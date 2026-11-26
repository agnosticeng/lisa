//! Cross-pass fences are exercised with tiny test-owned Untracked resources;
//! automatic hazard tracking must not mask a missing ordering edge. This does
//! not change the production allocator or revive an untracked scratch policy.
use super::*;
use crate::runtime::MetalRuntime;
use objc2_metal::{MTLHazardTrackingMode, MTLResourceOptions};

const SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;
kernel void fence_add(device const uint* x [[buffer(0)]],
                      device uint* y [[buffer(1)]],
                      constant uint& count [[buffer(2)]],
                      constant uint& add [[buffer(3)]],
                      uint i [[thread_position_in_grid]]) {
    if (i < count) y[i] = x[i] + add;
}
kernel void fence_read_only(device volatile const uint* x [[buffer(0)]],
                            constant uint& count [[buffer(1)]],
                            uint i [[thread_position_in_grid]]) {
    if (i < count) (void)x[i];
}
"#;

fn storage(runtime: &MetalRuntime, words: &[u32]) -> Arc<Buffer> {
    let raw = runtime
        .device()
        .newBufferWithLength_options(
            words.len() * 4,
            MTLResourceOptions::StorageModeShared | MTLResourceOptions::HazardTrackingModeUntracked,
        )
        .expect("tiny test-owned untracked buffer");
    assert_eq!(raw.hazardTrackingMode(), MTLHazardTrackingMode::Untracked);
    let buffer = Arc::new(Buffer::wrap(raw));
    // SAFETY: the new shared buffer holds exactly `words.len()` u32 values and is unbound.
    unsafe {
        std::ptr::copy_nonoverlapping(words.as_ptr(), buffer.contents().cast::<u32>(), words.len());
    }
    buffer
}
fn output(runtime: &MetalRuntime, count: usize) -> Arc<Buffer> {
    storage(runtime, &vec![u32::MAX; count + 4])
}
fn read(buffer: &Buffer, offset: usize, count: usize) -> Vec<u32> {
    // SAFETY: callers synchronize first and keep `offset + 4 * count` within the buffer.
    unsafe { std::slice::from_raw_parts(buffer.contents().add(offset).cast::<u32>(), count) }
        .to_vec()
}
fn waits(commands: &Commands) -> (usize, usize) {
    (
        commands.map_fence_waits.load(Ordering::Relaxed),
        commands.predecessor_fence_waits.load(Ordering::Relaxed),
    )
}
fn pass(runtime: &MetalRuntime, parallel: bool, body: impl FnOnce(&ComputeEncoder)) {
    if parallel {
        let scope = runtime.commands.recorded_encoder_scope();
        let guard = scope.encoder().unwrap();
        body(guard.encoder());
        drop(guard);
        drop(scope);
    } else {
        let guard = runtime.commands.encoder().unwrap();
        body(guard.encoder());
        drop(guard);
        runtime.commands.flush_encoder();
    }
}
fn add(
    runtime: &MetalRuntime,
    pipeline: &ComputePipeline,
    parallel: bool,
    x: &Arc<Buffer>,
    x_offset: usize,
    out: &Arc<Buffer>,
    out_offset: usize,
    count: usize,
    increment: u32,
) {
    pass(runtime, parallel, |encoder| {
        encoder.set_pipeline(pipeline);
        encoder.set_input(0, Some(x), x_offset);
        encoder.set_output(1, Some(out), out_offset);
        encoder.set_bytes(2, &(count as u32));
        encoder.set_bytes(3, &increment);
        encoder.dispatch_threads((count, 1, 1), (64, 1, 1));
    });
}
fn read_only(
    runtime: &MetalRuntime,
    pipeline: &ComputePipeline,
    parallel: bool,
    x: &Arc<Buffer>,
    offset: usize,
    count: usize,
) {
    pass(runtime, parallel, |encoder| {
        encoder.set_pipeline(pipeline);
        encoder.set_input(0, Some(x), offset);
        encoder.set_bytes(1, &(count as u32));
        encoder.dispatch_threads((count, 1, 1), (64, 1, 1));
    });
}
fn pipelines(runtime: &MetalRuntime) -> (ComputePipeline, ComputePipeline) {
    assert!(
        !super::super::env::gpu_probe_enabled(),
        "these checks exercise ordinary fence semantics"
    );
    (
        runtime.compile(SOURCE, "fence_add").unwrap(),
        runtime.compile(SOURCE, "fence_read_only").unwrap(),
    )
}

#[test]
fn predecessor_wait_covers_older_writers_and_read_only_pass_with_exact_counts() {
    let runtime = MetalRuntime::new(usize::MAX).unwrap();
    let (pipeline, reader) = pipelines(&runtime);
    let count = 1024;
    let words = (0..count + 4).map(|i| i as u32 * 3 + 7).collect::<Vec<_>>();
    let input = storage(&runtime, &words);
    let a = output(&runtime, count);
    let b = output(&runtime, count);
    let c = output(&runtime, count);
    let d = output(&runtime, count);
    let e = output(&runtime, count);
    let commands = &runtime.commands;
    let cb = commands.current_cb_id();
    add(&runtime, &pipeline, false, &input, 4, &a, 8, count, 1);
    assert_eq!(waits(commands), (0, 0));
    add(&runtime, &pipeline, false, &input, 4, &b, 4, count, 2);
    assert_eq!(
        waits(commands),
        (1, 0),
        "serial-to-serial map fallback remains"
    );
    read_only(&runtime, &reader, true, &a, 8, count);
    assert_eq!(
        commands.prev_outputs.lock().unwrap().len(),
        2,
        "two older writer fences remain live"
    );
    assert_eq!(
        waits(commands),
        (1, 1),
        "the required predecessor wait replaces two map waits"
    );
    add(&runtime, &pipeline, false, &a, 8, &c, 12, count, 3);
    assert_eq!(
        waits(commands),
        (1, 2),
        "a read-only predecessor still dominates the older writers"
    );
    add(&runtime, &pipeline, false, &b, 4, &d, 0, count, 4);
    assert_eq!(
        waits(commands),
        (4, 2),
        "three map waits remain at ordinary fallback"
    );
    // Test the unchanged missing-predecessor fallback with real queued writers.
    // Their fences remain retained in both the map and producer cleanup list.
    commands.state.lock().unwrap().predecessor = None;
    add(&runtime, &pipeline, true, &d, 0, &e, 4, count, 5);
    assert_eq!(waits(commands), (8, 2));
    assert_eq!(
        commands.current_cb_id(),
        cb,
        "no implicit commit supplied ordering"
    );
    assert_eq!(commands.completed_watermark(), 0);
    runtime.synchronize().unwrap();
    let expected = |increment: u32| {
        words[1..count + 1]
            .iter()
            .map(|&x| x + increment)
            .collect::<Vec<_>>()
    };
    assert_eq!(read(&a, 8, count), expected(1));
    assert_eq!(read(&b, 4, count), expected(2));
    assert_eq!(read(&c, 12, count), expected(4));
    assert_eq!(read(&d, 0, count), expected(6));
    assert_eq!(read(&e, 4, count), expected(11));
}

#[test]
fn predecessor_wait_preserves_raw_war_waw_rotations_and_held_outputs() {
    for per_buffer in [usize::MAX, 1, 2] {
        let runtime = MetalRuntime::new(per_buffer).unwrap();
        let (pipeline, reader) = pipelines(&runtime);
        let count = 4096;
        let words = (0..count + 4)
            .map(|i| i as u32 * 5 + 11)
            .collect::<Vec<_>>();
        let input = storage(&runtime, &words);
        let mutable = output(&runtime, count);
        let early = output(&runtime, count);
        let middle = output(&runtime, count);
        let late = output(&runtime, count);
        add(&runtime, &pipeline, true, &input, 4, &mutable, 8, count, 1);
        // RAW consumes the first write; this output stays live through later
        // writes to the same physical mutable allocation (WAR and WAW).
        add(&runtime, &pipeline, false, &mutable, 8, &early, 4, count, 0);
        read_only(&runtime, &reader, true, &mutable, 8, count);
        add(
            &runtime, &pipeline, false, &input, 4, &mutable, 8, count, 10,
        );
        add(
            &runtime, &pipeline, true, &mutable, 8, &middle, 12, count, 0,
        );
        add(&runtime, &pipeline, true, &input, 4, &mutable, 8, count, 20);
        add(&runtime, &pipeline, false, &mutable, 8, &late, 0, count, 0);
        assert_eq!(waits(&runtime.commands), (0, 6));
        if per_buffer == usize::MAX {
            assert_eq!(runtime.commands.current_cb_id(), 1);
        } else {
            assert!(
                runtime.commands.current_cb_id() > 1,
                "automatic rotation must be exercised"
            );
        }
        runtime.synchronize().unwrap();
        let expected = |increment: u32| {
            words[1..count + 1]
                .iter()
                .map(|&x| x + increment)
                .collect::<Vec<_>>()
        };
        assert_eq!(read(&early, 4, count), expected(1));
        assert_eq!(read(&middle, 12, count), expected(10));
        assert_eq!(read(&late, 0, count), expected(20));
        assert_eq!(read(&mutable, 8, count), expected(20));
        assert_eq!(read(&input, 0, words.len()), words);
    }
}

#[test]
fn predecessor_wait_keeps_independent_host_sequences_exact() {
    let runtime = Arc::new(MetalRuntime::new(2).unwrap());
    let (pipeline, reader) = pipelines(&runtime);
    let pipeline = Arc::new(pipeline);
    let reader = Arc::new(reader);
    let barrier = Arc::new(std::sync::Barrier::new(4));
    let workers = (0..4)
        .map(|worker| {
            let runtime = runtime.clone();
            let pipeline = pipeline.clone();
            let reader = reader.clone();
            let barrier = barrier.clone();
            std::thread::spawn(move || {
                let count = 1024;
                let words = (0..count + 4)
                    .map(|i| i as u32 + 10000 * worker)
                    .collect::<Vec<_>>();
                let input = storage(&runtime, &words);
                let mutable = output(&runtime, count);
                let held = output(&runtime, count);
                let final_out = output(&runtime, count);
                barrier.wait();
                add(
                    &runtime,
                    &pipeline,
                    worker % 2 == 0,
                    &input,
                    4,
                    &mutable,
                    8,
                    count,
                    3,
                );
                barrier.wait();
                add(
                    &runtime,
                    &pipeline,
                    worker % 2 != 0,
                    &mutable,
                    8,
                    &held,
                    4,
                    count,
                    5,
                );
                barrier.wait();
                read_only(&runtime, &reader, true, &mutable, 8, count);
                add(
                    &runtime, &pipeline, false, &input, 4, &mutable, 8, count, 17,
                );
                barrier.wait();
                add(
                    &runtime, &pipeline, true, &mutable, 8, &final_out, 0, count, 19,
                );
                (words, input, mutable, held, final_out)
            })
        })
        .collect::<Vec<_>>();
    let results = workers
        .into_iter()
        .map(|worker| worker.join().unwrap())
        .collect::<Vec<_>>();
    assert!(waits(&runtime.commands).1 > 0);
    runtime.synchronize().unwrap();
    for (words, input, mutable, held, final_out) in results {
        let count = words.len() - 4;
        assert_eq!(
            read(&held, 4, count),
            words[1..count + 1]
                .iter()
                .map(|&x| x + 8)
                .collect::<Vec<_>>()
        );
        assert_eq!(
            read(&final_out, 0, count),
            words[1..count + 1]
                .iter()
                .map(|&x| x + 36)
                .collect::<Vec<_>>()
        );
        assert_eq!(
            read(&mutable, 8, count),
            words[1..count + 1]
                .iter()
                .map(|&x| x + 17)
                .collect::<Vec<_>>()
        );
        assert_eq!(read(&input, 0, words.len()), words);
    }
}

#[test]
fn predecessor_wait_retains_order_after_partial_replay_error() {
    let runtime = MetalRuntime::new(usize::MAX).unwrap();
    let (pipeline, _) = pipelines(&runtime);
    let count = 1024;
    let words = vec![13; count + 4];
    let input = storage(&runtime, &words);
    let older = output(&runtime, count);
    let produced = output(&runtime, count);
    let readback = output(&runtime, count);
    add(&runtime, &pipeline, false, &input, 4, &older, 0, count, 2);
    let result: Result<()> = (|| {
        let scope = runtime.commands.recorded_encoder_scope();
        let guard = scope.encoder()?;
        let encoder = guard.encoder();
        encoder.set_pipeline(&pipeline);
        encoder.set_input(0, Some(&older), 0);
        encoder.set_output(1, Some(&produced), 4);
        encoder.set_bytes(2, &(count as u32));
        encoder.set_bytes(3, &3u32);
        encoder.dispatch_threads((count, 1, 1), (64, 1, 1));
        Err(err("injected host error after encoded producer"))
    })();
    assert_eq!(
        result.unwrap_err().to_string(),
        "injected host error after encoded producer"
    );
    assert!(!runtime.commands.recorded_encoder_is_open());
    add(
        &runtime, &pipeline, false, &produced, 4, &readback, 8, count, 7,
    );
    assert_eq!(waits(&runtime.commands), (0, 2));
    assert_eq!(runtime.commands.current_cb_id(), 1);
    assert_eq!(runtime.commands.completed_watermark(), 0);
    runtime.synchronize().unwrap();
    assert_eq!(read(&older, 0, count), vec![15; count]);
    assert_eq!(read(&produced, 4, count), vec![18; count]);
    assert_eq!(read(&readback, 8, count), vec![25; count]);
}
