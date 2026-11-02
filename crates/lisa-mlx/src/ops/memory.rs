//! `lisa_mlx::memory`.
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

/// Minimum live headroom kept outside a model's working set.
pub fn ram_reserve_bytes() -> usize {
    RAM_RESERVE_GIB << 30
}

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

/// Parse the available-memory percentage reported by `memory_pressure`.
fn parse_available_ram(report: &str, total_ram: usize) -> Option<usize> {
    let percent = report.lines().find_map(|line| {
        line.trim()
            .strip_prefix("System-wide memory free percentage:")?
            .trim()
            .strip_suffix('%')?
            .parse::<usize>()
            .ok()
    })?;
    if percent > 100 {
        return None;
    }
    total_ram.checked_mul(percent)?.checked_div(100)
}

/// Best-effort currently available physical memory in bytes, based on the
/// free-memory percentage `memory_pressure` reports after accounting for
/// reclaimable inactive and purgeable pages.
pub fn available_ram() -> Option<usize> {
    #[cfg(target_os = "macos")]
    {
        let output = std::process::Command::new("memory_pressure")
            .output()
            .ok()?;
        if !output.status.success() {
            return None;
        }
        let report = String::from_utf8(output.stdout).ok()?;
        parse_available_ram(&report, physical_ram())
    }
    #[cfg(not(target_os = "macos"))]
    {
        None
    }
}

#[cfg(all(test, target_os = "macos"))]
mod tests {
    use super::{available_ram, parse_available_ram, physical_ram};

    #[test]
    fn available_ram_parses_the_reported_free_fraction() {
        let total = 128usize << 30;
        let report = "System-wide memory free percentage: 86%\n";
        assert_eq!(parse_available_ram(report, total), Some(total * 86 / 100));
        assert_eq!(parse_available_ram("Pages free: 100\n", total), None);
    }

    #[test]
    fn available_ram_uses_live_memory_pressure_free_fraction() {
        let available = available_ram().expect("memory_pressure report should parse");
        assert!(available > 0);
        assert!(available <= physical_ram());
    }
}
