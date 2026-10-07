//! Our own Metal runtime: device/queue, byte-accounted buffer pool with a real
//! sweep, command batching (`Commands`/`ComputeEncoder`) with in-session
//! barriers and cross-encoder fences, pipeline cache, and the `Math` compile
//! regimes.
//!
//! Everything the tree dispatches flows through `mdev.commands` on ONE queue:
//! Metal does not order work across queues, so any dispatch outside this
//! runtime would race the custom kernels it shares buffers with.
//!
//! The pool is byte-accounted and swept once past a cap (8 GiB); reuse is
//! gated on a full flush since the buffer's last
//! bind, and taken buffers are zeroed (the eager arrays have no graph retaining
//! intermediates, so a stale region a kernel reads before its producer writes
//! would otherwise carry garbage).

mod bench_gpu_slot;
mod buffer;
mod commands;
mod device;
mod env;
mod fence;
mod pipeline;

pub use crate::error::{Error, Result};
pub use buffer::{Buffer, BufferPool};
pub use commands::{Commands, ComputeEncoder, EncoderGuard};
pub use device::{
    ConstVal, JIT_COMPILES, Math, MetalRuntime, fnv1a, jit_compiles, kernel_fingerprint, kernel_law,
    nax_available,
};
pub use env::{
    ALLOC_COUNT, DISPATCH_COUNT, ENCODER_COUNT, GPU_PROBE_RECORDS, LABEL_COUNTS,
    LABEL_DISPATCH, LABEL_GPU_NS,
    POOLHIT_COUNT, RuntimeError, WAIT_NS, ZERO_NS, gpu_probe_enabled,
};
pub use fence::Fence;
pub use pipeline::ComputePipeline;

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
        drop(a);
        let before = rt.pool.bytes();
        let b = rt.buffer(4096, "b").unwrap();
        assert_eq!(rt.pool.bytes(), before, "released buffer should be reused");
        drop(b);

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
}
