//! Our own Metal runtime: device/queue, byte-accounted buffer pool with a real
//! sweep, command batching (`Commands`/`ComputeEncoder`) with in-session
//! barriers and cross-encoder fences, pipeline cache, and the `Math` compile
//! regimes.
//!
//! Everything the tree dispatches flows through `mdev.commands` on ONE queue:
//! Metal does not order work across queues, so any dispatch outside this
//! runtime would race the custom kernels it shares buffers with.
//!
//! The pool is byte-accounted. Ordinary allocations, including every CPU
//! writer, reuse storage only after its last GPU command completes. Explicit
//! ordered GPU scopes can reuse unleased outputs earlier on the same queue:
//! full-write producers overwrite them directly, while partial producers get
//! an ordered GPU clear. Immutable model reservations bound private storage
//! with the detected RAM budget; allocation never waits while holding a lock.

#[cfg(test)]
mod bench_gpu_slot;
mod buffer;
mod commands;
mod device;
mod env;
mod fence;
mod pipeline;
mod residency;

pub use crate::error::{Error, Result};
pub use buffer::{Buffer, BufferPool, ResidentMemoryReservation};
pub use commands::{Commands, ComputeEncoder, EncoderGuard};
pub use device::{
    ConstVal, JIT_COMPILES, Math, MetalRuntime, OrderedGpuReuseScope,
    enter_ordered_gpu_reuse_scope, fnv1a, jit_compiles, kernel_fingerprint, kernel_law,
    nax_available,
};
pub use env::{
    ALLOC_COUNT, DISPATCH_COUNT, ENCODER_COUNT, GPU_PROBE_RECORDS, LABEL_COUNTS, LABEL_DISPATCH,
    LABEL_GPU_NS, POOLHIT_COUNT, RuntimeError, WAIT_NS, ZERO_NS, gpu_probe_enabled,
};
pub use fence::Fence;
pub use pipeline::ComputePipeline;
pub use residency::ResidencySet;

// ───────────────────────────── tests ─────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const KERNEL: &str = r#"
#include <metal_stdlib>
using namespace metal;
[[kernel]] void double_it(
    const device float* x [[buffer(0)]],
    device float* out [[buffer(1)]],
    uint i [[thread_position_in_grid]])
{
    out[i] = x[i] * 2.0f;
}
"#;

    const CHAIN_KERNEL: &str = r#"
#include <metal_stdlib>
using namespace metal;
[[kernel]] void add_one(
    const device float* x [[buffer(0)]],
    device float* out [[buffer(1)]],
    uint i [[thread_position_in_grid]])
{
    out[i] = x[i] + 1.0f;
}
"#;

    #[test]
    fn dispatch_and_read_back() {
        let rt = MetalRuntime::new(4).unwrap();
        let x = rt.buffer(64, "x").unwrap();
        let out = rt.buffer(64, "out").unwrap();
        unsafe {
            let p = x.contents() as *mut f32;
            for i in 0..16 {
                *p.add(i) = i as f32;
            }
        }
        let pipe = rt.compile(KERNEL, "double_it").unwrap();
        {
            let guard = rt.commands.encoder().unwrap();
            let enc = guard.encoder();
            enc.set_pipeline(&pipe);
            enc.set_input(0, Some(&x), 0);
            enc.set_output(1, Some(&out), 0);
            enc.dispatch_threads((16, 1, 1), (16, 1, 1));
        }
        rt.commands.flush_and_wait().unwrap();
        let got: Vec<f32> = unsafe {
            let p = out.contents() as *const f32;
            (0..16).map(|i| *p.add(i)).collect()
        };
        assert_eq!(got, (0..16).map(|i| (i * 2) as f32).collect::<Vec<_>>());
    }

    #[test]
    fn cross_command_buffer_fences_order_reused_outputs() {
        let rt = MetalRuntime::new(4).unwrap();
        rt.commands.set_per_buffer(1);
        let a = rt.buffer(64, "fence_chain_a").unwrap();
        let b = rt.buffer(64, "fence_chain_b").unwrap();
        unsafe {
            let p = a.contents() as *mut f32;
            for i in 0..16 {
                *p.add(i) = i as f32;
            }
        }
        let pipe = rt.compile(CHAIN_KERNEL, "add_one").unwrap();

        // A dispatch budget of one rotates after every encoded operation.
        // Alternating the same two buffers forces each command buffer to wait
        // for the preceding writer before consuming or overwriting its output.
        for step in 0..32 {
            let (input, output) = if step % 2 == 0 { (&a, &b) } else { (&b, &a) };
            let guard = rt.commands.encoder().unwrap();
            let encoder = guard.encoder();
            encoder.set_pipeline(&pipe);
            encoder.set_input(0, Some(input), 0);
            encoder.set_output(1, Some(output), 0);
            encoder.dispatch_threads((16, 1, 1), (16, 1, 1));
        }
        rt.commands.flush_and_wait().unwrap();

        let got: Vec<f32> = unsafe {
            let p = a.contents() as *const f32;
            (0..16).map(|i| *p.add(i)).collect()
        };
        let expected: Vec<f32> = (0..16).map(|i| i as f32 + 32.0).collect();
        assert_eq!(got, expected);
    }

    #[test]
    fn ordered_gpu_reuse_preserves_reads_zeroing_and_cpu_write_isolation() {
        let rt = MetalRuntime::new(1024).unwrap();
        let input_a = rt.buffer(64, "ordered_input_a").unwrap();
        let input_b = rt.buffer(64, "ordered_input_b").unwrap();
        let scratch = rt.buffer(64, "ordered_scratch").unwrap();
        let snapshot_a = rt.buffer(64, "ordered_snapshot_a").unwrap();
        let snapshot_b = rt.buffer(64, "ordered_snapshot_b").unwrap();
        unsafe {
            for index in 0..16 {
                *(input_a.contents() as *mut f32).add(index) = index as f32;
                *(input_b.contents() as *mut f32).add(index) = (100 + index) as f32;
            }
        }
        let pipeline = rt.compile(KERNEL, "double_it").unwrap();
        let encode_double = |input: &std::sync::Arc<Buffer>, output: &std::sync::Arc<Buffer>| {
            let guard = rt.commands.encoder().unwrap();
            let enc = guard.encoder();
            enc.set_pipeline(&pipeline);
            enc.set_input(0, Some(input), 0);
            enc.set_output(1, Some(output), 0);
            enc.dispatch_threads((16, 1, 1), (16, 1, 1));
        };
        encode_double(&input_a, &scratch);
        encode_double(&scratch, &snapshot_a);
        let scratch_id = scratch.id();
        drop(scratch);
        // End the encoder without submitting its command buffer. This makes
        // the in-flight condition deterministic and exercises encoder fences.
        rt.commands.flush_encoder();
        let scope = enter_ordered_gpu_reuse_scope();
        let reused = rt.buffer_for_full_gpu_write(64, "ordered_reused").unwrap();
        assert_eq!(
            reused.id(),
            scratch_id,
            "GPU writer should reuse the released lease"
        );
        drop(reused);
        let reused = rt
            .buffer_for_gpu_write(64, "ordered_clear_before_write")
            .unwrap();
        assert_eq!(reused.id(), scratch_id);
        encode_double(&input_b, &reused);
        encode_double(&reused, &snapshot_b);
        drop(reused);
        let host = rt.buffer_for_full_write(64, "ordered_cpu_import").unwrap();
        assert_ne!(
            host.id(),
            scratch_id,
            "CPU writes must not reuse an unsubmitted GPU buffer"
        );
        unsafe { std::ptr::write_bytes(host.contents(), 0x7f, host.length()) };
        let zeroed = rt.buffer_for_gpu_write(64, "ordered_zeroed").unwrap();
        assert_eq!(
            zeroed.id(),
            scratch_id,
            "GPU clear should reuse the released lease"
        );
        drop(scope);
        let outside_scope = rt.buffer_for_gpu_write(64, "ordinary_after_scope").unwrap();
        assert_ne!(
            outside_scope.id(),
            scratch_id,
            "scope exit restores completion-gated allocation"
        );
        rt.commands.flush_and_wait().unwrap();
        let read = |buffer: &Buffer| unsafe {
            std::slice::from_raw_parts(buffer.contents() as *const f32, 16).to_vec()
        };
        assert_eq!(
            read(&snapshot_a),
            (0..16).map(|index| 4.0 * index as f32).collect::<Vec<_>>()
        );
        assert_eq!(
            read(&snapshot_b),
            (0..16)
                .map(|index| 4.0 * (100 + index) as f32)
                .collect::<Vec<_>>()
        );
        assert!(
            unsafe { std::slice::from_raw_parts(zeroed.contents(), zeroed.length()) }
                .iter()
                .all(|&byte| byte == 0)
        );
    }

    /// Micro-bench: CPU price of one dispatch, by post. `cargo test -p lisa-mlx
    /// --lib bench_dispatch_cpu -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn bench_dispatch_cpu_cost() {
        let rt = MetalRuntime::new(64).unwrap();
        let x = rt.buffer(1024, "x").unwrap();
        let out = rt.buffer(1024, "out").unwrap();
        let pipe = rt.compile(KERNEL, "double_it").unwrap();
        let n: usize = 10_000;
        let report = |name: &str, t: std::time::Instant, k: usize| {
            let ns = t.elapsed().as_nanos() as f64 / k as f64;
            println!("{name:<38} {ns:8.1} ns/dispatch");
        };

        // A: lock-only (encoder() open + drop, nothing encoded).
        // Rotation keeps command buffers flowing like the real path.
        let t = std::time::Instant::now();
        for _ in 0..n {
            drop(rt.commands.encoder().unwrap());
        }
        rt.commands.flush_and_wait().unwrap();
        report("A lock+bookkeeping (encoder())", t, n);

        // B: dispatch-only on the real per-op path: encoder() + set_pipeline +
        // 2 bindings + set_bytes + dispatch (rotation at per_buffer=16).
        let t = std::time::Instant::now();
        for _ in 0..n {
            let g = rt.commands.encoder().unwrap();
            let e = g.encoder();
            e.set_pipeline(&pipe);
            e.set_input(0, Some(&x), 0);
            e.set_output(1, Some(&out), 0);
            e.set_bytes(2, &0u32);
            e.dispatch_threads((16, 1, 1), (16, 1, 1));
        }
        rt.commands.flush_and_wait().unwrap();
        report("B full per-op path (realistic)", t, n);

        // C: bindings only (objc setBuffer + hazard bookkeeping, no dispatch).
        let t = std::time::Instant::now();
        for _ in 0..n {
            let g = rt.commands.encoder().unwrap();
            let e = g.encoder();
            e.set_input(0, Some(&x), 0);
            e.set_output(1, Some(&out), 0);
        }
        rt.commands.flush_and_wait().unwrap();
        report("C bindings only (no dispatch)", t, n);

        // D: set_pipeline only.
        let t = std::time::Instant::now();
        for _ in 0..n {
            let g = rt.commands.encoder().unwrap();
            g.encoder().set_pipeline(&pipe);
        }
        rt.commands.flush_and_wait().unwrap();
        report("D set_pipeline only", t, n);

        // E: set_bytes only.
        let t = std::time::Instant::now();
        for _ in 0..n {
            let g = rt.commands.encoder().unwrap();
            g.encoder().set_bytes(2, &0u32);
        }
        rt.commands.flush_and_wait().unwrap();
        report("E set_bytes only", t, n);

        // F: dispatch only (pipeline+bindings pre-set on a long-lived encoder).
        {
            let g = rt.commands.encoder().unwrap();
            let e = g.encoder();
            e.set_pipeline(&pipe);
            e.set_input(0, Some(&x), 0);
            e.set_output(1, Some(&out), 0);
            let t = std::time::Instant::now();
            for _ in 0..n {
                e.set_bytes(2, &0u32);
                e.dispatch_threads((16, 1, 1), (16, 1, 1));
            }
            drop(g);
            rt.commands.flush_and_wait().unwrap();
            report("F held encoder: bytes+dispatch", t, n);
        }
    }

    #[test]
    fn pool_reuses_then_caps() {
        let rt = MetalRuntime::new(4).unwrap();
        rt.pool.set_limit(1 << 20);
        // A released buffer is reused, not re-allocated.
        let a = rt.buffer(4096, "a").unwrap();
        let a_id = a.id();
        unsafe { std::ptr::write_bytes(a.contents(), 0x7f, a.length()) };
        drop(a);
        let before = rt.pool.bytes();
        let b = rt.buffer(4096, "b").unwrap();
        assert_eq!(b.id(), a_id, "the free buffer should be reused");
        assert_eq!(unsafe { *b.contents() }, 0, "reused storage starts cleared");
        assert_eq!(rt.pool.bytes(), before, "released buffer should be reused");
        let in_flight_id = b.id();
        b.mark_used_cb(100);
        drop(b);
        let c = rt.buffer(4096, "c").unwrap();
        assert_ne!(c.id(), in_flight_id, "in-flight storage must not be reused");
        drop(c);

        // Past the cap, `sweep_if_over` releases what nobody holds.
        let mut held = Vec::new();
        for _ in 0..8 {
            held.push(rt.buffer(512 * 1024, "big").unwrap());
        }
        let peak = rt.pool.bytes();
        held.truncate(2);
        assert!(rt.pool.bytes() > (1 << 20), "pool over cap: {peak}");
        assert!(rt.pool.sweep_if_over(), "sweep should fire");
        assert!(rt.pool.bytes() <= (1 << 20), "pool bounded after sweep");
    }

    #[test]
    fn concurrent_pool_take_never_leases_one_buffer_twice() {
        let rt = MetalRuntime::new(4).unwrap();
        let seed = rt.buffer(4096, "pool_concurrent_seed").unwrap();
        let seed_id = seed.id();
        drop(seed);

        let workers = 16;
        let barrier = std::sync::Arc::new(std::sync::Barrier::new(workers));
        let mut joins = Vec::with_capacity(workers);
        for _ in 0..workers {
            let pool = std::sync::Arc::clone(&rt.pool);
            let barrier = std::sync::Arc::clone(&barrier);
            joins.push(std::thread::spawn(move || {
                barrier.wait();
                pool.take(4096, 0)
            }));
        }

        let leases: Vec<_> = joins
            .into_iter()
            .filter_map(|join| join.join().expect("pool allocator thread panicked"))
            .collect();
        assert_eq!(leases.len(), 1, "one pooled buffer must yield one lease");
        assert_eq!(leases[0].id(), seed_id);
    }
}
