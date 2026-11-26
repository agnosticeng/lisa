//! Host memory admission policy. Device allocators receive explicit byte budgets.

/// Default head-room kept free for the OS, the desktop and the driver when no
/// explicit cap is given (GiB).
#[cfg(any(test, not(all(target_os = "ios", not(target_abi = "sim")))))]
const RAM_RESERVE_GIB: usize = 8;

/// Headroom for the UI, driver and transient work inside an iOS app's limit.
#[cfg(any(test, all(target_os = "ios", not(target_abi = "sim"))))]
const IOS_RESERVE_BYTES: usize = 512 << 20;

static RAM_CAP: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// Cap the RAM budget at `gib` GiB for this process. The application sets it
/// once from `--ram-cap-gb`; it can only lower the detected budget, and it
/// also arms HTTP request admission.
pub fn set_ram_cap_gib(gib: usize) {
    RAM_CAP.store(
        gib.saturating_mul(1 << 30),
        std::sync::atomic::Ordering::Relaxed,
    );
}

/// Storage format of attention key/value caches.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum KvFormat {
    /// 8-bit keys and values where the model implements them, else its own
    /// format.
    #[default]
    Auto,
    /// Each model's own key/value format (F16 or wider).
    Native,
    /// 8-bit keys and values with one F32 scale per token and head. Halves
    /// cache memory and is a separate numerical law; models without it
    /// refuse to load.
    Int8,
}

static KV_FORMAT: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

/// Select the key/value cache format for models loaded after this call. The
/// application sets it once from `--kv-format`.
pub fn set_kv_format(format: KvFormat) {
    KV_FORMAT.store(format as u8, std::sync::atomic::Ordering::Relaxed);
}

/// The key/value cache format the application selected.
pub fn kv_format() -> KvFormat {
    match KV_FORMAT.load(std::sync::atomic::Ordering::Relaxed) {
        1 => KvFormat::Native,
        2 => KvFormat::Int8,
        _ => KvFormat::Auto,
    }
}

/// Whether a model that implements 8-bit keys and values uses them.
pub fn kv_int8() -> bool {
    kv_format() != KvFormat::Native
}

/// The explicit RAM cap in bytes, if the application set one.
pub fn ram_cap() -> Option<usize> {
    match RAM_CAP.load(std::sync::atomic::Ordering::Relaxed) {
        0 => None,
        bytes => Some(bytes),
    }
}

/// Total physical RAM in bytes (`sysctl hw.memsize`), cached; 0 if unavailable.
pub fn physical_ram() -> usize {
    #[cfg(target_os = "ios")]
    {
        let mut bytes = 0u64;
        let mut size = std::mem::size_of_val(&bytes);
        // SAFETY: the name is a NUL-terminated literal and `size` is the byte size of
        // `bytes`, so sysctl writes at most that scalar output.
        let result = unsafe {
            libc::sysctlbyname(
                c"hw.memsize".as_ptr(),
                (&mut bytes as *mut u64).cast(),
                &mut size,
                std::ptr::null_mut(),
                0,
            )
        };
        return if result == 0 && size == 8 {
            usize::try_from(bytes).unwrap_or(0)
        } else {
            0
        };
    }
    #[cfg(target_os = "linux")]
    {
        // SAFETY: sysconf takes no pointers and only reads a system constant.
        let pages = unsafe { libc::sysconf(libc::_SC_PHYS_PAGES) };
        // SAFETY: sysconf takes no pointers and only reads a system constant.
        let size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
        return usize::try_from(pages)
            .ok()
            .and_then(|p| usize::try_from(size).ok().and_then(|s| p.checked_mul(s)))
            .unwrap_or(0);
    }
    #[cfg(not(any(target_os = "ios", target_os = "linux")))]
    {
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
}

/// The admission budget in bytes. Zero means RAM detection failed or the host
/// cannot provide the required reserve; admission then refuses new allocations.
///
/// Mac/Simulator: physical RAM minus 8 GiB. Physical iOS devices: the smaller
/// of physical RAM and the current additional per-app allowance, minus 512 MiB.
/// An explicit cap ([`set_ram_cap_gib`]) can only lower either budget. The iOS allowance is read on
/// every call, never cached; this is a conservative budget for new admissions,
/// not a measurement of total memory owned by an already loaded model.
pub fn ram_budget() -> usize {
    #[cfg(all(target_os = "ios", not(target_abi = "sim")))]
    {
        ios_admission_budget(physical_ram(), available_ram().unwrap_or(0), ram_cap())
    }
    #[cfg(not(all(target_os = "ios", not(target_abi = "sim"))))]
    {
        admission_budget(physical_ram(), ram_cap())
    }
}

#[cfg(any(test, not(all(target_os = "ios", not(target_abi = "sim")))))]
fn admission_budget(physical: usize, cap: Option<usize>) -> usize {
    let safe = physical.saturating_sub(RAM_RESERVE_GIB << 30);
    cap.map_or(safe, |cap| cap.min(safe))
}

#[cfg(any(test, all(target_os = "ios", not(target_abi = "sim"))))]
fn ios_admission_budget(physical: usize, available: usize, cap: Option<usize>) -> usize {
    let safe = physical.min(available).saturating_sub(IOS_RESERVE_BYTES);
    cap.map_or(safe, |cap| cap.min(safe))
}

#[cfg(test)]
mod budget_tests {
    use super::*;
    #[test]
    fn cap_can_only_lower_detected_budget_and_detection_failure_refuses() {
        let gib = 1usize << 30;
        assert_eq!(admission_budget(64 * gib, None), 56 * gib);
        assert_eq!(admission_budget(64 * gib, Some(32 * gib)), 32 * gib);
        assert_eq!(admission_budget(64 * gib, Some(128 * gib)), 56 * gib);
        assert_eq!(admission_budget(0, Some(128 * gib)), 0);
        assert_eq!(admission_budget(4 * gib, None), 0);
    }

    #[test]
    fn ios_admits_small_models_within_the_live_app_allowance() {
        let gib = 1usize << 30;
        assert_eq!(ios_admission_budget(8 * gib, 3 * gib, None), 5 * gib / 2);
        assert_eq!(ios_admission_budget(8 * gib, 3 * gib, Some(gib)), gib);
        assert_eq!(
            ios_admission_budget(8 * gib, 3 * gib, Some(16 * gib)),
            5 * gib / 2
        );
        assert_eq!(ios_admission_budget(8 * gib, gib, None), gib / 2);
    }

    #[test]
    fn ios_refuses_missing_or_exhausted_memory_and_bounds_by_physical_ram() {
        let gib = 1usize << 30;
        for available in [0, IOS_RESERVE_BYTES - 1, IOS_RESERVE_BYTES] {
            assert_eq!(ios_admission_budget(8 * gib, available, Some(gib)), 0);
        }
        assert_eq!(ios_admission_budget(0, 3 * gib, None), 0);
        assert_eq!(ios_admission_budget(2 * gib, 3 * gib, None), 3 * gib / 2);
    }
}

/// Parse the available-memory percentage reported by `memory_pressure`.
#[cfg(target_os = "macos")]
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

/// Current additional memory allowance. macOS uses `memory_pressure` including
/// reclaimable pages; physical iOS uses the live per-app limit minus footprint.
/// The latter can shrink during the app lifetime and must never be cached.
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
    #[cfg(all(target_os = "ios", not(target_abi = "sim")))]
    {
        // SAFETY: matches libSystem's `size_t os_proc_available_memory(void)`.
        unsafe extern "C" {
            fn os_proc_available_memory() -> usize;
        }
        // SAFETY: the call takes no arguments and only reads the jetsam allowance.
        Some(unsafe { os_proc_available_memory() })
    }
    // Simulator processes have no iOS jetsam limit; the API returns zero there.
    // Their physical-RAM budget still uses the detected Mac host minus 8 GiB.
    #[cfg(any(target_abi = "sim", not(any(target_os = "macos", target_os = "ios"))))]
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

/// Provider-neutral allocation diagnostics; live accounting, not an RSS peak.
pub fn report<D: lisa_backend::ManagedDevice>(device: &D, tag: &str) {
    let usage = device.memory_usage().ok();
    eprintln!(
        "[mlx-mem {tag}] active {} MiB | private {} MiB | budget {} MiB",
        usage.as_ref().map_or(0, |m| m.active >> 20),
        usage.as_ref().map_or(0, |m| m.private >> 20),
        ram_budget() >> 20
    );
}
