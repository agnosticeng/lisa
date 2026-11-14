//! Host memory admission policy. Device allocators receive explicit byte budgets.

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
    (g > 0).then(|| g.checked_mul(1usize << 30)).flatten()
}

/// Total physical RAM in bytes (`sysctl hw.memsize`), cached; 0 if unavailable.
pub fn physical_ram() -> usize {
    #[cfg(target_os = "ios")]
    {
        let mut bytes = 0u64;
        let mut size = std::mem::size_of_val(&bytes);
        // sysctl writes only the supplied scalar-sized output on Apple hosts.
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
        let pages = unsafe { libc::sysconf(libc::_SC_PHYS_PAGES) };
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
/// Physical RAM minus `RAM_RESERVE_GIB`, lowered by `LISA_RAM_CAP_GB` when set. This is
/// the number an admission layer must refuse against: the machine-safety rule is
/// that allocations beyond the budget are refused, not attempted.
pub fn ram_budget() -> usize {
    admission_budget(physical_ram(), env_cap_bytes())
}

fn admission_budget(physical: usize, cap: Option<usize>) -> usize {
    let safe = physical.saturating_sub(ram_reserve_bytes());
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
    #[cfg(all(target_os = "ios", not(target_abi = "sim")))]
    {
        unsafe extern "C" {
            fn os_proc_available_memory() -> usize;
        }
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
