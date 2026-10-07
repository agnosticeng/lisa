//! `lisa_mlx::memory`.
//!
//! The Metal allocator keeps every buffer it ever allocated in a
//! size-bucketed pool and only releases the ones with `strong_count == 1` from
//! `MetalDevice::drop_unused_buffers`, which is called from
//! `wait_until_completed`/`flush_and_wait_current` — **never** from
//! `new_buffer`/`allocate_buffer` (the field doc promises a sweep on every
//! allocation; the code does not do it). lisa pipelines a whole prefill chunk
//! (or MTP round) without intermediate evals, so without an explicit sweep the
//! pool grows for the entire run and drives the box past its RAM. These mirror
//! The buffer-pool cap (8 GiB).
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
        m.pool.sweep();
    });
}

/// Release pooled buffers once the device's allocation has grown more than
/// the configured limit past the baseline. Cheap when under the limit (no
/// sync), so it is safe to call on chunk/round boundaries and periodically
/// during decode.
pub fn trim_cache() {
    let limit = CACHE_LIMIT.load(Ordering::Relaxed);
    if limit == 0 {
        return;
    }
    with_dev(|m| {
        let now = m.pool.bytes();
        let base = CACHE_BASELINE.load(Ordering::Relaxed);
        if base == 0 {
            CACHE_BASELINE.store(now, Ordering::Relaxed);
        } else if now > base.saturating_add(limit) {
            let _ = m.synchronize();
            m.pool.sweep();
        }
    });
}

pub fn active_memory() -> Result<usize> {
    // The pool's byte accounting (the runtime owns all allocations).
    Ok(with_dev(|m| m.pool.bytes()).unwrap_or(0))
}

/// No high-water mark is tracked, so report the live allocation.
pub fn peak_memory() -> Result<usize> {
    active_memory()
}

pub fn cache_memory() -> Result<usize> {
    Ok(0)
}

pub fn memory_limit() -> Result<usize> {
    // No public working-set query on our runtime. `LISA_RAM_CAP_GB` implements
    // the documented machine-safety cap; when it is unset the historical
    // "no cap" (0) is preserved so existing callers are unaffected.
    if let Some(cap) = env_cap_bytes() {
        return Ok(cap);
    }
    Ok(0)
}

/// Default head-room kept free for the OS, the desktop and the driver when no
/// explicit cap is given (GiB).
const RAM_RESERVE_GIB: usize = 8;

/// `LISA_RAM_CAP_GB` parsed to bytes, if set to a positive integer.
fn env_cap_bytes() -> Option<usize> {
    let v = std::env::var("LISA_RAM_CAP_GB").ok()?;
    let g: usize = v.trim().parse().ok()?;
    (g > 0).then(|| g << 30)
}

/// Total physical RAM in bytes (`sysctl hw.memsize`), cached; 0 if unavailable.
pub fn physical_ram() -> usize {
    use std::sync::OnceLock;
    static RAM: OnceLock<usize> = OnceLock::new();
    *RAM.get_or_init(|| {
        std::process::Command::new("sysctl")
            .args(["-n", "hw.memsize"])
            .output()
            .ok()
            .and_then(|o| String::from_utf8(o.stdout).ok())
            .and_then(|s| s.trim().parse::<usize>().ok())
            .unwrap_or(0)
    })
}

/// The admission budget in bytes — ALWAYS non-zero, unlike [`memory_limit`].
///
/// `LISA_RAM_CAP_GB` if set, else physical RAM minus [`RAM_RESERVE_GIB`]. This is
/// the number an admission layer must refuse against: the machine-safety rule is
/// that allocations beyond the budget are refused, not attempted.
pub fn ram_budget() -> usize {
    env_cap_bytes().unwrap_or_else(|| physical_ram().saturating_sub(RAM_RESERVE_GIB << 30))
}
