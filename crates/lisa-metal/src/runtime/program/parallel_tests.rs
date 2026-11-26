//! Numerical checks for the optional recorded dependency-parallel schedule.
use super::*;

const SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;
kernel void parallel_affine(device const float* x [[buffer(0)]],
                            device const float* b [[buffer(1)]],
                            device float* y [[buffer(2)]],
                            constant uint& n [[buffer(3)]],
                            uint i [[thread_position_in_grid]]) {
    if (i < n) y[i] = x[i] * 2.0f + b[i];
}
"#;
fn values(runtime: &Arc<MetalRuntime>, data: &[f32]) -> Arc<Buffer> {
    let buffer = runtime.buffer(data.len() * 4, "parallel_input").unwrap();
    // SAFETY: fresh lease of `data.len() * 4` bytes with no GPU user.
    unsafe {
        std::ptr::copy_nonoverlapping(
            data.as_ptr().cast::<u8>(),
            buffer.contents(),
            data.len() * 4,
        );
    }
    buffer
}
fn read(buffer: &Buffer, offset: usize, count: usize) -> Vec<u32> {
    // SAFETY: callers synchronize first and keep `offset + 4 * count` in bounds.
    unsafe { std::slice::from_raw_parts(buffer.contents().add(offset).cast::<f32>(), count) }
        .iter()
        .map(|v| v.to_bits())
        .collect()
}
fn launch(
    runtime: &Arc<MetalRuntime>,
    p: &ComputePipeline,
    x: &Arc<Buffer>,
    x_offset: usize,
    b: &Arc<Buffer>,
    out: &Arc<Buffer>,
    n: u32,
) {
    let guard = runtime.commands.encoder().unwrap();
    let e = guard.encoder();
    e.set_pipeline(p);
    e.set_input(0, Some(x), x_offset);
    e.set_input(1, Some(b), 0);
    e.set_output(2, Some(out), 0);
    e.set_bytes(3, &n);
    e.dispatch_threads((n as usize, 1, 1), (64.min(n as usize), 1, 1));
}
fn encode(
    runtime: &Arc<MetalRuntime>,
    p: &ComputePipeline,
    x: &Arc<Buffer>,
    bias: &Arc<Buffer>,
    n: u32,
) -> Result<Vec<ProgramOutput>> {
    let bytes = n as usize * 4;
    let a = runtime.buffer_for_audited_full_gpu_write(bytes, "parallel_a")?;
    let b = runtime.buffer_for_audited_full_gpu_write(bytes, "parallel_b")?;
    {
        let guard = runtime.commands.encoder()?;
        let e = guard.encoder();
        e.set_pipeline(p);
        e.set_input(0, Some(x), 4);
        e.set_input(1, Some(bias), 0);
        e.set_output(2, Some(&a), 0);
        e.set_bytes(3, &n);
        e.dispatch_threads((n as usize, 1, 1), (64, 1, 1));
        // Inheritance across a pipeline change INSIDE one guard stays live.
        // An unused temporary argument is explicitly cleared through setBytes.
        e.set_input(4, Some(&a), 0);
        e.set_bytes(4, &0u32);
        e.set_pipeline(p);
        e.set_output(2, Some(&b), 0);
        e.dispatch_threads((n as usize, 1, 1), (64, 1, 1));
    }
    let c = runtime.buffer_for_audited_full_gpu_write(bytes, "parallel_c")?;
    launch(runtime, p, &a, 0, &b, &c, n);
    let d = runtime.buffer_for_audited_full_gpu_write(bytes, "parallel_d")?;
    // a/b's final read is in the preceding dispatch. d reuses one physical
    // scratch range: the plan must insert a WAR even though c does not feed d.
    launch(runtime, p, x, 4, bias, &d, n);
    runtime.commands.flush_encoder();
    let out = runtime.buffer(bytes + 16, "parallel_partial_output")?;
    launch(runtime, p, &c, 0, &d, &out, n);
    Ok(vec![
        ProgramOutput {
            buffer: out.clone(),
            offset: 0,
        },
        ProgramOutput {
            buffer: Arc::new(out.retain_shared()),
            offset: 4,
        },
        ProgramOutput {
            buffer: x.clone(),
            offset: 4,
        },
    ])
}

#[test]
fn parallel_replay_preserves_bits_packed_war_offsets_poison_and_live_outputs() {
    for per_buffer in [usize::MAX, 2] {
        let runtime = Arc::new(MetalRuntime::new(usize::MAX).unwrap());
        runtime.commands.set_per_buffer(per_buffer);
        let p = runtime.compile(SOURCE, "parallel_affine").unwrap();
        let n = 512u32;
        let data = (0..=n)
            .map(|i| (i as f32 - 257.) * 0.125)
            .collect::<Vec<_>>();
        let x = values(&runtime, &data);
        let bias = values(&runtime, &vec![0.25; n as usize]);
        let inputs = [
            ProgramInput {
                buffer: x.clone(),
                required_bytes: data.len() * 4,
            },
            ProgramInput {
                buffer: Arc::new(x.retain_shared()),
                required_bytes: data.len() * 4,
            },
        ];
        let (captured, program) = capture_dispatch_program(&runtime, &inputs, || {
            let out = encode(&runtime, &p, &x, &bias, n)?;
            Ok::<_, crate::error::Error>((out.clone(), out))
        })
        .unwrap();
        let mut program = program.unwrap();
        assert!(program.enable_parallel());
        let plan = program.parallel.as_ref().unwrap();
        assert!(
            !plan.barriers[1],
            "independent within-guard projections can overlap"
        );
        assert!(plan.barriers[2], "c must wait for a and b");
        assert!(
            plan.barriers[3],
            "reused scratch must wait for its former read"
        );
        let scratch = program.scratch.as_ref().unwrap();
        assert!(
            scratch.bytes < scratch.unaliased_bytes,
            "test requires actual physical lifetime aliasing"
        );
        let bytes = scratch.bytes;
        let before = PARALLEL_REPLAYS.load(Ordering::Relaxed);
        let first = program.replay(&runtime, &inputs).unwrap().unwrap();
        assert!(!runtime.commands.recorded_encoder_is_open());
        runtime.synchronize().unwrap();
        let expected = read(&captured[0].buffer, 0, n as usize + 4);
        assert_eq!(read(&first[0].buffer, 0, n as usize + 4), expected);
        assert_eq!(&expected[n as usize..], &[0; 4]);
        let arena = runtime
            .pool
            .take_program_scratch(bytes, runtime.commands.program_identity())
            .unwrap();
        // SAFETY: poison only completed private scratch (synchronized above). Every later
        // logical value must be produced afresh, despite physical lifetime aliasing.
        unsafe {
            std::ptr::write_bytes(arena.contents(), 0xff, arena.length());
        }
        drop(arena);
        let second = program.replay(&runtime, &inputs).unwrap().unwrap();
        let changed = data.iter().map(|v| v + 3.).collect::<Vec<_>>();
        let y = values(&runtime, &changed);
        let changed_inputs = [
            ProgramInput {
                buffer: y.clone(),
                required_bytes: changed.len() * 4,
            },
            ProgramInput {
                buffer: Arc::new(y.retain_shared()),
                required_bytes: changed.len() * 4,
            },
        ];
        let third = program.replay(&runtime, &changed_inputs).unwrap().unwrap();
        let ordinary = encode(&runtime, &p, &y, &bias, n).unwrap();
        runtime.synchronize().unwrap();
        assert_eq!(read(&second[0].buffer, 0, n as usize + 4), expected);
        assert_eq!(
            read(&third[0].buffer, 0, n as usize + 4),
            read(&ordinary[0].buffer, 0, n as usize + 4)
        );
        assert_eq!(
            read(&third[1].buffer, 4, n as usize - 1),
            read(&ordinary[1].buffer, 4, n as usize - 1)
        );
        assert_eq!(
            read(&x, 0, data.len()),
            data.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
        assert_ne!(
            resource_id(&first[0].buffer),
            resource_id(&second[0].buffer)
        );
        assert_eq!(resource_id(&third[0].buffer), resource_id(&third[1].buffer));
        assert!(PARALLEL_REPLAYS.load(Ordering::Relaxed) >= before + 3);
        // Mutated input alias patterns must decline before issuing a dispatch.
        assert!(
            program
                .replay(&runtime, &[inputs[0].clone(), changed_inputs[0].clone()])
                .unwrap()
                .is_none()
        );
    }
}

#[test]
fn parallel_partial_allocation_failure_closes_mode_before_private_scratch_reuse() {
    let runtime = Arc::new(MetalRuntime::new(usize::MAX).unwrap());
    let p = runtime.compile(SOURCE, "parallel_affine").unwrap();
    let x = values(&runtime, &[1.; 4]);
    let bias = values(&runtime, &[2.; 4]);
    let inputs = [ProgramInput {
        buffer: x.clone(),
        required_bytes: 16,
    }];
    let (mut retained, program) = capture_dispatch_program(&runtime, &inputs, || -> Result<_> {
        let a = runtime.buffer_for_audited_full_gpu_write(16, "parallel_error_a")?;
        launch(&runtime, &p, &x, 0, &bias, &a, 4);
        let b = runtime.buffer_for_audited_full_gpu_write(16, "parallel_error_b")?;
        launch(&runtime, &p, &x, 0, &bias, &b, 4);
        let out = runtime.buffer(4096, "parallel_error_output")?;
        launch(&runtime, &p, &a, 0, &b, &out, 4);
        Ok((
            vec![a, b, out.clone()],
            vec![ProgramOutput {
                buffer: out,
                offset: 0,
            }],
        ))
    })
    .unwrap();
    let mut program = program.unwrap();
    assert!(program.enable_parallel());
    runtime.synchronize().unwrap();
    let reservation = runtime
        .reserve_model_memory(1, runtime.pool.bytes() + 2048 + 1)
        .unwrap();
    let before = super::super::DISPATCH_COUNT.load(Ordering::Relaxed);
    assert!(matches!(
        program.replay(&runtime, &inputs),
        Err(crate::error::Error::AllocationUnavailable(_))
    ));
    assert_eq!(
        super::super::DISPATCH_COUNT.load(Ordering::Relaxed) - before,
        2
    );
    assert!(!runtime.commands.recorded_encoder_is_open());
    let arena = runtime
        .pool
        .take_program_scratch(2048, runtime.commands.program_identity())
        .unwrap();
    assert!(arena.last_used_cb() > runtime.commands.completed_watermark());
    let id = arena.id();
    drop(arena);
    drop(retained.pop());
    let out = program.replay(&runtime, &inputs).unwrap().unwrap();
    let arena = runtime
        .pool
        .take_program_scratch(2048, runtime.commands.program_identity())
        .unwrap();
    assert_eq!(arena.id(), id);
    drop(arena);
    runtime.synchronize().unwrap();
    assert_eq!(read(&out[0].buffer, 0, 4), vec![12f32.to_bits(); 4]);
    assert!(!runtime.commands.recorded_encoder_is_open());
    drop(reservation);
}

#[test]
fn parallel_replay_scopes_remain_isolated_between_host_threads() {
    let runtime = Arc::new(MetalRuntime::new(2).unwrap());
    let p = runtime.compile(SOURCE, "parallel_affine").unwrap();
    let n = 64;
    let x = values(&runtime, &vec![1.; n + 1]);
    let bias = values(&runtime, &vec![0.25; n]);
    let input = [ProgramInput {
        buffer: x.clone(),
        required_bytes: (n + 1) * 4,
    }];
    let (_, program) = capture_dispatch_program(&runtime, &input, || {
        let out = encode(&runtime, &p, &x, &bias, n as u32)?;
        Ok::<_, crate::error::Error>(((), out))
    })
    .unwrap();
    let mut program = program.unwrap();
    assert!(program.enable_parallel());
    let program = Arc::new(program);
    let barrier = Arc::new(std::sync::Barrier::new(4));
    let threads = (0..4)
        .map(|i| {
            let runtime = runtime.clone();
            let program = program.clone();
            let barrier = barrier.clone();
            std::thread::spawn(move || {
                let x = values(&runtime, &vec![i as f32; n + 1]);
                barrier.wait();
                let out = program
                    .replay(
                        &runtime,
                        &[ProgramInput {
                            buffer: x,
                            required_bytes: (n + 1) * 4,
                        }],
                    )
                    .unwrap()
                    .unwrap();
                (i, out)
            })
        })
        .collect::<Vec<_>>();
    let outputs = threads
        .into_iter()
        .map(|h| h.join().unwrap())
        .collect::<Vec<_>>();
    runtime.synchronize().unwrap();
    for (i, out) in outputs {
        let a = i as f32 * 2. + 0.25;
        let c = a * 2. + a;
        let expected = c * 2. + a;
        assert_eq!(read(&out[0].buffer, 0, n), vec![expected.to_bits(); n]);
    }
    assert!(!runtime.commands.recorded_encoder_is_open());
}
