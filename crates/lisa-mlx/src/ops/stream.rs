/// Diagnostic: nanoseconds the host spent blocked in GPU waits.
pub fn runtime_wait_ns() -> u64 {
    crate::runtime::WAIT_NS.load(std::sync::atomic::Ordering::Relaxed)
}

/// Diagnostics: dispatch + encoder-open counts since process start.
pub fn runtime_dispatch_count() -> u64 {
    crate::runtime::DISPATCH_COUNT.load(std::sync::atomic::Ordering::Relaxed)
}

pub fn runtime_encoder_count() -> u64 {
    crate::runtime::ENCODER_COUNT.load(std::sync::atomic::Ordering::Relaxed)
}

pub fn runtime_alloc_count() -> u64 {
    crate::runtime::ALLOC_COUNT.load(std::sync::atomic::Ordering::Relaxed)
}

pub fn runtime_poolhit_count() -> u64 {
    crate::runtime::POOLHIT_COUNT.load(std::sync::atomic::Ordering::Relaxed)
}

pub fn label_counts() -> Vec<(String, u64)> {
    crate::runtime::LABEL_COUNTS.lock().unwrap().clone()
}

pub fn runtime_label_counts() -> Vec<(String, u64)> {
    crate::runtime::LABEL_COUNTS.lock().unwrap().clone()
}

/// Diagnostic: cumulative GPU nanoseconds per kernel label (from command
/// buffer GPUStartTime/GPUEndTime, split by dispatch counts).
pub fn runtime_label_gpu_ms() -> Vec<(String, u64)> {
    crate::runtime::LABEL_GPU_NS.lock().unwrap().clone()
}

/// Every kernel label that was dispatched, with its count — the census source.
pub fn runtime_label_dispatches() -> Vec<(String, u64)> {
    crate::runtime::LABEL_DISPATCH.lock().unwrap().clone()
}

pub fn runtime_zero_ns() -> u64 {
    crate::runtime::ZERO_NS.load(std::sync::atomic::Ordering::Relaxed)
}

/// A Metal stream. One command queue per device, so this just
/// carries the device (MLX's thread-local stream semantics land with `eval`).
#[derive(Clone)]
pub struct Stream {
    rt: std::sync::Arc<crate::runtime::MetalRuntime>,
}

impl Stream {
    pub fn gpu() -> Self {
        static RT: std::sync::OnceLock<std::sync::Arc<crate::runtime::MetalRuntime>> =
            std::sync::OnceLock::new();
        let rt = RT
            .get_or_init(|| {
                // `LISA_GPU_PROBE=1`: one dispatch per command buffer so
                // the per-buffer GPUStartTime/GPUEndTime measure single
                // kernels (diagnostic; commits per dispatch are slow).
                let probe = std::env::var_os("LISA_GPU_PROBE").is_some();
                let per_buffer = if probe { 1 } else { 16 };
                std::sync::Arc::new(
                    crate::runtime::MetalRuntime::new(per_buffer).expect("metal runtime"),
                )
            })
            .clone();
        Self { rt }
    }

    pub fn thread_local_or_default() -> Self {
        Self::gpu()
    }

    /// The Metal runtime (the only device now).
    pub fn device(&self) -> &std::sync::Arc<crate::runtime::MetalRuntime> {
        &self.rt
    }

    /// Our own runtime (the flip's handle).
    pub fn runtime(&self) -> &std::sync::Arc<crate::runtime::MetalRuntime> {
        &self.rt
    }
}

impl Default for Stream {
    fn default() -> Self {
        Self::gpu()
    }
}
