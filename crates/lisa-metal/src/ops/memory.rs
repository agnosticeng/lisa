//! `lisa_metal::memory`.
//!
//! Active memory includes private pooled buffers and lifetime-scoped mapped
//! model reservations. With a resident reservation, new private allocations
//! share its RAM budget: the allocator reclaims unused buffers whose GPU work
//! has completed, then refuses an allocation that still exceeds the cap. It
//! does not wait for GPU work while holding the allocation lock.
//!
//! Explicit cache clearing and growth-based trimming synchronize before
//! sweeping unused pooled buffers. Their growth counters cover private pool
//! bytes only; mapped model pages are owned by the resident reservation.
use super::*;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Excess the pool may grow past the baseline before a sweep is forced.
/// `0` disables the automatic sweep.
static CACHE_LIMIT: AtomicUsize = AtomicUsize::new(0);
/// Device allocation observed on the first `trim_cache` (the weights + the
/// steady-state caches); only growth past this is pool churn.
static CACHE_BASELINE: AtomicUsize = AtomicUsize::new(0);

fn with_dev<T>(f: impl FnOnce(&crate::runtime::MetalRuntime) -> T) -> Option<T> {
    let rt = Stream::thread_local_or_default().runtime().clone();
    Some(f(&rt))
}

pub fn set_cache_limit(bytes: usize) {
    CACHE_LIMIT.store(bytes, Ordering::Relaxed);
}

/// Override the dispatch budget per command buffer (rotate experiments).
pub fn set_per_buffer(n: usize) {
    with_dev(|m| m.commands.set_per_buffer(n));
}

/// Commit, wait, and release every pooled buffer whose only owner is the
/// allocator. The public path to `drop_unused_buffers`.
pub fn clear_cache() {
    with_dev(|m| {
        let _ = m.synchronize();
        m.pool.sweep_completed(m.commands.completed_watermark());
    });
}

/// Release pooled buffers once the device's allocation has grown more than
/// the configured limit past the baseline. Cheap when under the limit (no
/// sync), so it is safe to call on chunk/round boundaries and periodically
/// during decode.
pub fn trim_cache() {
    if CACHE_LIMIT.load(Ordering::Relaxed) == 0 {
        return;
    }
    with_dev(trim_cache_on);
}

pub(crate) fn trim_cache_on(m: &crate::runtime::MetalRuntime) {
    let limit = CACHE_LIMIT.load(Ordering::Relaxed);
    if limit == 0 {
        return;
    }
    let now = m.pool.bytes();
    let base = CACHE_BASELINE.load(Ordering::Relaxed);
    if base == 0 {
        CACHE_BASELINE.store(now, Ordering::Relaxed);
    } else if now > base.saturating_add(limit) {
        let _ = m.synchronize();
        m.pool.sweep_completed(m.commands.completed_watermark());
    }
}

pub fn active_memory() -> Result<usize> {
    Ok(with_dev(|m| m.pool.total_bytes()).unwrap_or(0))
}

/// Private pooled allocations, excluding immutable mapped model reservations.
pub fn private_allocated_memory() -> Result<usize> {
    Ok(with_dev(|m| m.pool.bytes()).unwrap_or(0))
}

/// Reserve immutable mapped model bytes once for their complete lifetime.
/// Overlapping Metal views of the same pages must not be counted repeatedly.
pub fn reserve_model_memory(
    bytes: usize,
    budget: usize,
) -> Result<crate::runtime::ResidentMemoryReservation> {
    Stream::thread_local_or_default()
        .runtime()
        .reserve_model_memory(bytes, budget)
}

/// A caller-owned allowance for additional Metal allocations. Nested scopes
/// may only lower the limit. The caller must serialize work on this runtime
/// and drop nested scopes in reverse order. Kernel arithmetic/shapes are unchanged.
pub struct AllocationBudget {
    runtime: std::sync::Arc<crate::runtime::MetalRuntime>,
    previous: usize,
    _thread_bound: std::marker::PhantomData<std::rc::Rc<()>>,
}
impl Drop for AllocationBudget {
    fn drop(&mut self) {
        self.runtime.swap_allocation_limit(self.previous);
    }
}
pub fn allocation_budget(additional_bytes: usize) -> Result<AllocationBudget> {
    let runtime = Stream::thread_local_or_default().runtime().clone();
    // Retire old sessions before measuring a new allowance: otherwise unused
    // pages can consume most of the process limit before this scope begins.
    runtime.synchronize()?;
    runtime
        .pool
        .sweep_completed(runtime.commands.completed_watermark());
    let limit = runtime
        .allocated_bytes()
        .saturating_add(additional_bytes)
        .max(1);
    let previous = runtime.swap_allocation_limit(limit);
    if previous != 0 && previous < limit {
        runtime.swap_allocation_limit(previous);
    }
    Ok(AllocationBudget {
        runtime,
        previous,
        _thread_bound: std::marker::PhantomData,
    })
}
