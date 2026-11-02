use objc2::rc::Retained;
use objc2_foundation::NSString;

use crate::error::Error;

/// Kept for call sites that name the runtime error explicitly.
pub type RuntimeError = Error;

pub(super) fn err(m: impl Into<String>) -> Error {
    Error::Msg(m.into())
}

pub static WAIT_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static DISPATCH_COUNT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static ALLOC_COUNT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static POOLHIT_COUNT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static ZERO_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static LABEL_COUNTS: std::sync::Mutex<Vec<(String, u64)>> = std::sync::Mutex::new(Vec::new());
/// Cumulative GPU nanoseconds per kernel label, from `GPUStartTime`/`GPUEndTime`
/// on each completed command buffer, split proportionally to the number of
/// dispatches each label encoded into that buffer.
pub static LABEL_GPU_NS: std::sync::Mutex<Vec<(String, u64)>> = std::sync::Mutex::new(Vec::new());
/// Cumulative DISPATCH counts per kernel label, taken from the encoder-side
/// `cb_labels` at command-buffer completion. Unlike `LABEL_GPU_NS` this needs no
/// GPU duration to be attributed, so a kernel that ran is listed even when its
/// time rounds to zero — which is what makes a kernel-engagement census exact.
pub static LABEL_DISPATCH: std::sync::Mutex<Vec<(String, u64)>> = std::sync::Mutex::new(Vec::new());

pub static ENCODER_COUNT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// GPU-probe mode (`LISA_GPU_PROBE=1`): one dispatch per command buffer
/// (`per_buffer=1`), so each completed buffer's `GPUStartTime`/`GPUEndTime`
/// interval can be attributed to its single kernel. This is a diagnostic
/// command-buffer interval, not a hardware-counter measurement of exclusive
/// shader occupancy. Records hold the kernel label and absolute GPU start/end
/// timestamps for timeline analysis.
static GPU_PROBE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
pub static GPU_PROBE_RECORDS: std::sync::Mutex<Vec<(String, f64, f64)>> =
    std::sync::Mutex::new(Vec::new());

pub fn gpu_probe_enabled() -> bool {
    *GPU_PROBE.get_or_init(|| std::env::var_os("LISA_GPU_PROBE").is_some())
}

/// Whether explicit diagnostics need per-dispatch kernel labels. Cache the
/// environment-derived gate so the disabled production path never checks the
/// environment once per dispatch.
pub fn kernel_label_capture_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| gpu_probe_enabled() || cb_record_file().is_some())
}

/// Per-command-buffer recorder (`LISA_CB_RECORD=<path>`): appends one JSON
/// line per completed command buffer — its GPU duration and per-label
/// dispatch counts — without changing the configured submission cadence
/// (`LISA_PER_BUFFER`; its default is 65536). Thousands of cbs with varying
/// kernel mixes give a least-squares system whose solution estimates per-kernel
/// command-buffer intervals on the unmodified submission cadence. It is a
/// diagnostic model, not a hardware-counter measurement of exclusive shader
/// occupancy.
pub fn cb_record_file() -> Option<std::path::PathBuf> {
    static PATH: std::sync::OnceLock<Option<std::path::PathBuf>> = std::sync::OnceLock::new();
    PATH.get_or_init(|| {
        std::env::var_os("LISA_CB_RECORD")
            .map(std::path::PathBuf::from)
            .filter(|p| !p.as_os_str().is_empty())
    })
    .clone()
}

/// Interned kernel-label table (ids start at 1). Written rarely (first
/// `set_pipeline` per kernel), read per dispatch via the pipeline's cached id.
pub static LABEL_TABLE: std::sync::Mutex<Vec<String>> = std::sync::Mutex::new(Vec::new());

pub fn intern_label(name: &str) -> u32 {
    let mut t = LABEL_TABLE.lock().unwrap();
    if let Some(i) = t.iter().position(|n| n == name) {
        return i as u32 + 1;
    }
    t.push(name.to_string());
    t.len() as u32
}

pub(super) fn label_name(id: u32) -> String {
    LABEL_TABLE.lock().unwrap()[(id - 1) as usize].clone()
}

/// Attribute `dur_ns` of GPU time across the labels of one completed command
/// buffer, proportionally to each label's dispatch count in that buffer
/// (buffers batch ~16 dispatches of mixed kernels, so a per-label split by
/// dispatch count is the simplest attribution that stays useful).
/// Accumulate the per-kernel dispatch counts carried by `cb_labels`.
pub fn add_dispatch_counts(labels: &[(String, u64)]) {
    if labels.is_empty() {
        return;
    }
    let mut g = LABEL_DISPATCH.lock().unwrap();
    for (label, count) in labels {
        match g.iter_mut().find(|(l, _)| l == label) {
            Some(e) => e.1 += *count,
            None => g.push((label.to_string(), *count)),
        }
    }
}

pub fn add_gpu_ns_split(labels: &[(String, u64)], dur_ns: u64) {
    if labels.is_empty() {
        return;
    }
    let total: u64 = labels.iter().map(|(_, c)| *c).sum();
    if total == 0 {
        return;
    }
    let mut g = LABEL_GPU_NS.lock().unwrap();
    for (label, count) in labels {
        let share = ((dur_ns as u128) * (*count as u128) / (total as u128)) as u64;
        match g.iter_mut().find(|(l, _)| l == label) {
            Some(e) => e.1 += share,
            None => g.push((label.to_string(), share)),
        }
    }
}

pub(super) fn ns(s: &str) -> Retained<NSString> {
    NSString::from_str(s)
}
