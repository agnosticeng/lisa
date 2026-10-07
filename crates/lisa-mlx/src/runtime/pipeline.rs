use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::MTLComputePipelineState;

// ───────────────────────────── pipelines ─────────────────────────────

#[derive(Clone)]
pub struct ComputePipeline {
    pub(super) raw: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    /// The kernel function name; used as the per-label GPU-time attribution key.
    pub name: String,
    /// Interned id for `name` (None = not yet interned). `set_pipeline` reads
    /// this instead of cloning the String through a Mutex on every dispatch.
    pub(super) label: std::cell::OnceCell<u32>,
}

unsafe impl Send for ComputePipeline {}
unsafe impl Sync for ComputePipeline {}

impl ComputePipeline {
    /// Wrap an already-built pipeline.
    pub fn from_raw(raw: Retained<ProtocolObject<dyn MTLComputePipelineState>>) -> Self {
        Self {
            raw,
            name: "raw".to_string(),
            label: std::cell::OnceCell::new(),
        }
    }

    /// The interned label id (ids start at 1); registers `name` on first use.
    pub(super) fn label_id(&self) -> u32 {
        *self.label.get_or_init(|| super::env::intern_label(&self.name))
    }

    pub fn raw(&self) -> &ProtocolObject<dyn MTLComputePipelineState> {
        &self.raw
    }

    pub fn max_total_threads_per_threadgroup(&self) -> usize {
        self.raw.maxTotalThreadsPerThreadgroup()
    }
}
