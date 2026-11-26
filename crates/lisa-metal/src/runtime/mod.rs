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

mod buffer;
mod commands;
mod device;
mod env;
mod fence;
mod library_cache;
mod pipeline;
mod program;
mod residency;

pub use crate::error::{Error, Result};
pub use buffer::{Buffer, BufferPool, ResidentMemoryReservation};
pub use commands::{Commands, ComputeEncoder, EncoderGuard};
pub use device::{
    ConstVal, JIT_COMPILES, Math, MetalRuntime, OrderedGpuReuseScope, apple_gpu_family,
    check_platform_support, enter_ordered_gpu_reuse_scope, fnv1a, jit_compiles, kernel_fingerprint,
    kernel_law, metal4_supported, nax_available,
};
pub use env::{
    ALLOC_COUNT, ALLOCATION_LABELS, DISPATCH_COUNT, ENCODER_COUNT, GPU_PROBE_RECORDS, LABEL_COUNTS,
    LABEL_DISPATCH, LABEL_GPU_NS, POOLHIT_COUNT, RuntimeError, WAIT_NS, ZERO_NS, gpu_probe_enabled,
};
pub use fence::Fence;
pub use pipeline::ComputePipeline;
pub use program::{
    DispatchProgram, ProgramInput, ProgramOutput, capture_dispatch_program,
    dispatch_program_parallel_replay_count, dispatch_program_replay_count,
};
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
        // SAFETY: fresh 64-byte lease with no GPU user; 16 f32 writes fit.
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
        // SAFETY: the queue was flushed and waited; the buffer holds 16 f32 values.
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
        // SAFETY: fresh 64-byte lease with no GPU user; 16 f32 writes fit.
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

        // SAFETY: the queue was flushed and waited; the buffer holds 16 f32 values.
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
        // SAFETY: fresh 64-byte lease with no GPU user; 16 f32 writes fit.
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
        // SAFETY: CPU-import lease distinct from the in-flight scratch; write stays in `length()`.
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
        // SAFETY: the queue was flushed and waited; the buffer holds 16 f32 values.
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
            // SAFETY: flushed above; the range is exactly `zeroed`'s allocation.
            unsafe { std::slice::from_raw_parts(zeroed.contents(), zeroed.length()) }
                .iter()
                .all(|&byte| byte == 0)
        );
    }

    #[test]
    fn audited_full_gpu_write_overwrites_poison_without_clearing_padding() {
        let rt = MetalRuntime::new(1024).unwrap();
        // A non-bucket-aligned logical shape exercises both payload and padding.
        let count = 259;
        let bytes = count * std::mem::size_of::<f32>();
        let input = rt.buffer(bytes, "audited_input").unwrap();
        let seed = rt.buffer(bytes, "audited_poison").unwrap();
        // SAFETY: fresh leases with no GPU user; writes stay within `bytes` and `seed.length()`.
        unsafe {
            for i in 0..count {
                *input.contents().cast::<f32>().add(i) = i as f32 - 127.0;
            }
            std::ptr::write_bytes(seed.contents(), 0xa5, seed.length());
        }
        let id = seed.id();
        drop(seed);
        let out = rt
            .buffer_for_audited_full_gpu_write(bytes, "audited_output")
            .unwrap();
        assert_eq!(out.id(), id, "completed poisoned storage should be reused");
        assert!(
            // SAFETY: no GPU work references `out` yet; the range is exactly its allocation.
            unsafe { std::slice::from_raw_parts(out.contents(), out.length()) }
                .iter()
                .all(|&v| v == 0xa5),
            "audited allocation must not clear either payload or padding"
        );
        let pipeline = rt.compile(KERNEL, "double_it").unwrap();
        {
            let guard = rt.commands.encoder().unwrap();
            let encoder = guard.encoder();
            encoder.set_pipeline(&pipeline);
            encoder.set_input(0, Some(&input), 0);
            encoder.set_output(1, Some(&out), 0);
            encoder.dispatch_threads((count, 1, 1), (256, 1, 1));
        }
        rt.synchronize().unwrap();
        // SAFETY: synchronized above; `out` holds at least `count` f32 values.
        let actual = unsafe { std::slice::from_raw_parts(out.contents().cast::<f32>(), count) };
        assert_eq!(
            actual,
            (0..count)
                .map(|i| 2.0 * (i as f32 - 127.0))
                .collect::<Vec<_>>()
        );
        assert!(
            // SAFETY: synchronized above; the padding after `bytes` lies within the allocation.
            unsafe { std::slice::from_raw_parts(out.contents().add(bytes), out.length() - bytes) }
                .iter()
                .all(|&v| v == 0xa5),
            "only the logical shape belongs to the GPU writer"
        );
        drop(out);
        let conservative = rt
            .buffer_for_full_gpu_write(bytes, "conservative_output")
            .unwrap();
        assert_eq!(conservative.id(), id);
        assert!(
            // SAFETY: completed reuse with no GPU user; the range is exactly the allocation.
            unsafe { std::slice::from_raw_parts(conservative.contents(), conservative.length()) }
                .iter()
                .all(|&v| v == 0),
            "the existing allocation contract remains zero-initialized"
        );
    }

    #[test]
    fn audited_full_gpu_write_respects_completion_and_ordered_scope() {
        let rt = MetalRuntime::new(1024).unwrap();
        let input = rt.buffer(64, "audited_chain_input").unwrap();
        let scratch = rt.buffer(64, "audited_chain_scratch").unwrap();
        let first = rt.buffer(64, "audited_chain_first").unwrap();
        let second = rt.buffer(64, "audited_chain_second").unwrap();
        // SAFETY: fresh 64-byte lease with no GPU user; 16 f32 writes fit.
        unsafe {
            for i in 0..16 {
                *input.contents().cast::<f32>().add(i) = i as f32;
            }
        }
        let pipeline = rt.compile(CHAIN_KERNEL, "add_one").unwrap();
        let encode = |x: &std::sync::Arc<Buffer>, y: &std::sync::Arc<Buffer>| {
            let guard = rt.commands.encoder().unwrap();
            let encoder = guard.encoder();
            encoder.set_pipeline(&pipeline);
            encoder.set_input(0, Some(x), 0);
            encoder.set_output(1, Some(y), 0);
            encoder.dispatch_threads((16, 1, 1), (16, 1, 1));
        };
        encode(&input, &scratch);
        encode(&scratch, &first);
        let id = scratch.id();
        drop(scratch);
        // Keep these writes unsubmitted: the allocator cannot accidentally
        // observe a fast GPU completion and hide an incorrect reuse policy.
        rt.commands.flush_encoder();
        let ordinary = rt
            .buffer_for_audited_full_gpu_write(64, "audited_completion_gate")
            .unwrap();
        assert_ne!(ordinary.id(), id);
        let scope = enter_ordered_gpu_reuse_scope();
        let reused = rt
            .buffer_for_audited_full_gpu_write(64, "audited_ordered_reuse")
            .unwrap();
        assert_eq!(reused.id(), id);
        encode(&first, &reused);
        encode(&reused, &second);
        drop(reused);
        drop(scope);
        let after_scope = rt
            .buffer_for_audited_full_gpu_write(64, "audited_after_scope")
            .unwrap();
        assert_ne!(after_scope.id(), id);
        rt.synchronize().unwrap();
        let completed = rt
            .buffer_for_audited_full_gpu_write(64, "audited_completed_reuse")
            .unwrap();
        assert_eq!(completed.id(), id);
        // SAFETY: synchronized above; each buffer holds 16 f32 values.
        let read = |b: &Buffer| unsafe {
            std::slice::from_raw_parts(b.contents().cast::<f32>(), 16).to_vec()
        };
        for (buffer, increment) in [(&first, 2.0), (&completed, 3.0), (&second, 4.0)] {
            assert_eq!(
                read(buffer),
                (0..16).map(|i| i as f32 + increment).collect::<Vec<_>>()
            );
        }
    }

    #[test]
    fn allocation_budget_drains_pending_work_without_changing_results() {
        fn run(bounded: bool) -> (Vec<f32>, u64) {
            let rt = MetalRuntime::new(10000).unwrap();
            rt.pool.set_limit(0);
            let pipe = rt.compile(CHAIN_KERNEL, "add_one").unwrap();
            let bytes = 64 * 1024;
            let mut value = rt.buffer(bytes, "budget_input").unwrap();
            if bounded {
                rt.swap_allocation_limit(rt.allocated_bytes() + 3 * bytes);
            }
            let start = rt.commands.current_cb_id();
            for _ in 0..24 {
                let output = rt.buffer(bytes, "budget_output").unwrap();
                {
                    let guard = rt.commands.encoder().unwrap();
                    let encoder = guard.encoder();
                    encoder.set_pipeline(&pipe);
                    encoder.set_input(0, Some(&value), 0);
                    encoder.set_output(1, Some(&output), 0);
                    encoder.dispatch_threads((bytes / 4, 1, 1), (256, 1, 1));
                }
                value = output;
            }
            let rotations = rt.commands.current_cb_id() - start;
            rt.synchronize().unwrap();
            let result =
                // SAFETY: synchronized above; `value` holds `bytes / 4` f32 values.
                unsafe { std::slice::from_raw_parts(value.contents().cast::<f32>(), bytes / 4) }
                    .to_vec();
            (result, rotations)
        }
        let (reference, _) = run(false);
        for _ in 0..2 {
            let (actual, rotations) = run(true);
            assert!(
                rotations > 0,
                "the budget must drain work before the final readback"
            );
            assert_eq!(actual, reference);
            assert!(actual.iter().all(|&value| value == 24.0));
        }
    }

    #[test]
    fn allocation_budget_refuses_live_storage_instead_of_overcommitting() {
        let rt = MetalRuntime::new(4).unwrap();
        let held = rt.buffer(64 * 1024, "budget_held").unwrap();
        rt.swap_allocation_limit(rt.allocated_bytes() + 64 * 1024);
        let error = rt
            .buffer(128 * 1024, "budget_excess")
            .err()
            .expect("must refuse an oversized allocation");
        assert!(error.to_string().contains("Metal memory budget exceeded"));
        assert_eq!(held.length(), 64 * 1024);
        rt.swap_allocation_limit(0);
        assert!(rt.buffer(128 * 1024, "budget_reset").is_ok());
    }

    #[test]
    fn compact_host_import_keeps_completion_gates_inside_ordered_scope() {
        // Simulated completion watermarks require no submitted GPU work.
        let rt = MetalRuntime::new(4).unwrap();
        let large = rt.buffer(23 * 1024 * 1024, "import_large_scratch").unwrap();
        let large_id = large.id();
        drop(large);
        let small = rt
            .buffer_for_compact_full_write(12, "import_small")
            .unwrap();
        assert_eq!(small.length(), 1024);
        assert_ne!(small.id(), large_id);
        let small_id = small.id();
        // SAFETY: fresh lease with no GPU user; the write stays within `length()`.
        unsafe { std::ptr::write_bytes(small.contents(), 0xa5, small.length()) };
        drop(small);
        let reused = rt
            .buffer_for_compact_full_write(12, "import_reused")
            .unwrap();
        assert_eq!(reused.id(), small_id);
        assert!(
            // SAFETY: completed CPU-only lease; the range is exactly the allocation.
            unsafe { std::slice::from_raw_parts(reused.contents(), reused.length()) }
                .iter()
                .all(|&v| v == 0xa5),
            "full-write imports must not clear storage before copying"
        );
        reused.mark_used_cb(100);
        drop(reused);
        let _scope = enter_ordered_gpu_reuse_scope();
        let pending = rt
            .buffer_for_compact_full_write(12, "import_pending")
            .unwrap();
        assert_ne!(
            pending.id(),
            small_id,
            "CPU imports still require completion"
        );
        assert_eq!(pending.length(), 1024);
        let ordinary = rt.buffer_for_full_write(4096, "ordinary_import").unwrap();
        assert_eq!(
            ordinary.id(),
            large_id,
            "ordinary best-fit imports are unchanged"
        );
    }

    #[test]
    fn compact_output_reuse_keeps_capacity_leases_and_completion_gates() {
        // Only host allocation/bookkeeping is exercised: no shader compilation
        // or GPU commands. Simulate completion with explicit buffer watermarks.
        let rt = MetalRuntime::new(4).unwrap();
        let large = rt.buffer(128 * 2304 * 3 * 4, "prefix_scratch").unwrap();
        let large_id = large.id();
        drop(large);
        let compact = rt
            .buffer_for_audited_compact_gpu_write(2304 * 3 * 4, "final_history")
            .unwrap();
        assert_eq!(compact.length(), 2304 * 3 * 4);
        assert_ne!(compact.id(), large_id);
        let compact_id = compact.id();
        // SAFETY: fresh lease not yet bound to any submitted command buffer; write stays in `length()`.
        unsafe { std::ptr::write_bytes(compact.contents(), 0xa5, compact.length()) };
        assert!(rt.pool.take_exact(2304 * 3 * 4, 100, false).is_none());
        assert!(rt.pool.take_exact(2304 * 3 * 4, 0, true).is_none());
        compact.mark_used_cb(100);
        drop(compact);
        assert!(rt.pool.take_exact(2304 * 3 * 4, 99, false).is_none());
        let ordered = rt.pool.take_exact(2304 * 3 * 4, 0, true).unwrap();
        assert_eq!(ordered.id(), compact_id);
        assert!(rt.pool.take_exact(2304 * 3 * 4, 100, false).is_none());
        drop(ordered);
        let completed = rt.pool.take_exact(2304 * 3 * 4, 100, false).unwrap();
        assert_eq!(completed.id(), compact_id);
        assert!(
            // SAFETY: no submitted command uses this lease; the range is exactly the allocation.
            unsafe { std::slice::from_raw_parts(completed.contents(), completed.length()) }
                .iter()
                .all(|&v| v == 0xa5),
            "audited reuse must not clear storage"
        );
        // The opt-in does not change ordinary best-fit allocation.
        let ordinary = rt.pool.take(64 * 1024, 0).unwrap();
        assert_eq!(ordinary.id(), large_id);
        let rounded = rt
            .buffer_for_audited_compact_gpu_write(1030, "rounded_final_history")
            .unwrap();
        assert_eq!(rounded.length(), 2048);
    }

    #[test]
    fn pool_best_fit_skips_busy_and_in_flight_buckets() {
        let rt = MetalRuntime::new(4).unwrap();
        let exact = rt.buffer(4096, "held_exact").unwrap();
        let busy = rt.buffer(5120, "in_flight").unwrap();
        busy.mark_used_cb(100);
        let busy_id = busy.id();
        drop(busy);
        let smallest = rt.buffer(6144, "smallest_eligible").unwrap();
        let smallest_id = smallest.id();
        let larger = rt.buffer(8192, "larger_eligible").unwrap();
        let larger_id = larger.id();
        drop(smallest);
        drop(larger);
        let before = rt.pool.bytes();
        let result = rt.buffer(4096, "best_fit").unwrap();
        assert_eq!(result.id(), smallest_id);
        assert_ne!(result.id(), exact.id());
        assert_ne!(result.id(), busy_id);
        assert_eq!(rt.pool.bytes(), before);
        let next = rt.buffer(4096, "next_fit").unwrap();
        assert_eq!(next.id(), larger_id);
        assert_eq!(rt.pool.bytes(), before);
        drop(result);
        drop(next);
        rt.pool.sweep_completed(0);
        assert_eq!(rt.pool.bytes(), 4096 + 5120);
        let completed = rt.pool.take(4096, 100).unwrap();
        assert_eq!(completed.id(), busy_id);
    }

    #[test]
    fn reused_storage_clears_requested_bucket_not_unused_capacity() {
        let rt = MetalRuntime::new(4).unwrap();
        let large = rt.buffer(8192, "large_storage").unwrap();
        let id = large.id();
        // SAFETY: fresh lease with no GPU user; the write stays within `length()`.
        unsafe { std::ptr::write_bytes(large.contents(), 0x7f, large.length()) };
        drop(large);
        let small = rt.buffer(1030, "small_output").unwrap();
        assert_eq!(small.id(), id);
        // SAFETY: the reused lease has no GPU user; the range is exactly the allocation.
        let bytes = unsafe { std::slice::from_raw_parts(small.contents(), small.length()) };
        assert!(bytes[..2048].iter().all(|&b| b == 0));
        assert!(bytes[2048..].iter().all(|&b| b == 0x7f));
        drop(small);
        // A later larger request clears its newly published region as well.
        let large = rt.buffer(8192, "large_again").unwrap();
        assert_eq!(large.id(), id);
        // SAFETY: the reused lease has no GPU user; the range is exactly the allocation.
        let bytes = unsafe { std::slice::from_raw_parts(large.contents(), large.length()) };
        assert!(bytes.iter().all(|&b| b == 0));
    }

    #[test]
    fn pool_reuses_then_caps() {
        let rt = MetalRuntime::new(4).unwrap();
        rt.pool.set_limit(1 << 20);
        // A released buffer is reused, not re-allocated.
        let a = rt.buffer(4096, "a").unwrap();
        let a_id = a.id();
        // SAFETY: fresh lease with no GPU user; the write stays within `length()`.
        unsafe { std::ptr::write_bytes(a.contents(), 0x7f, a.length()) };
        drop(a);
        let before = rt.pool.bytes();
        let b = rt.buffer(4096, "b").unwrap();
        assert_eq!(b.id(), a_id, "the free buffer should be reused");
        // SAFETY: the reused lease has no GPU user and holds at least one byte.
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
