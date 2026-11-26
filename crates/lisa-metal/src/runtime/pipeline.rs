use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::MTLComputePipelineState;

// ───────────────────────────── pipelines ─────────────────────────────

/// A compiled kernel. Cloning retains the Metal object and shares the name;
/// the attribution label is interned once, when the pipeline is created.
#[derive(Clone)]
pub struct ComputePipeline {
    pub(super) raw: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    /// The kernel function name; used as the per-label GPU-time attribution key.
    pub name: std::sync::Arc<str>,
    pub(super) label: u32,
}

// SAFETY: MTLComputePipelineState is immutable and thread-safe; `name` is `Arc<str>`
// and `label` a plain `u32`.
unsafe impl Send for ComputePipeline {}
unsafe impl Sync for ComputePipeline {}

impl ComputePipeline {
    pub(super) fn new(
        raw: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
        name: &str,
    ) -> Self {
        Self {
            raw,
            name: name.into(),
            label: super::env::intern_label(name),
        }
    }

    /// Wrap an already-built pipeline.
    pub fn from_raw(raw: Retained<ProtocolObject<dyn MTLComputePipelineState>>) -> Self {
        Self::new(raw, "raw")
    }

    /// The interned label id (ids start at 1).
    pub(super) fn label_id(&self) -> u32 {
        self.label
    }

    pub fn raw(&self) -> &ProtocolObject<dyn MTLComputePipelineState> {
        &self.raw
    }

    pub fn max_total_threads_per_threadgroup(&self) -> usize {
        self.raw.maxTotalThreadsPerThreadgroup()
    }
}
