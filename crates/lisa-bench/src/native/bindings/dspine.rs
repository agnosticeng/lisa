//! Application alias for the portable DSpine implementation.
pub use lisa_engine::bindings::dspine::{
    DSpineTarget, REPO, full_tables_enabled, resolve_selection, supports_target,
};
pub type DSpineDraft = lisa_engine::bindings::dspine::DSpineDraft<lisa_metal::device::MetalDevice>;
#[cfg(test)]
use crate::bindings::dspine_bridge as bound_bridge;
#[cfg(test)]
use crate::core::{cache::FullAttentionCache, norm::RmsNorm};
#[cfg(test)]
use anyhow::{Context, ensure};
#[cfg(test)]
use lisa_metal::{Array, Dtype};
#[cfg(test)]
use lisa_models::qwen3_5::dspine::bridge::Utility;
#[cfg(test)]
type DenseLinear = lisa_engine::bindings::dspine::DenseLinear<Array>;
#[cfg(test)]
type DraftAttention = lisa_engine::bindings::dspine::DraftAttention<Array>;
#[cfg(test)]
type DraftStash = lisa_engine::bindings::dspine::DraftStash<Array>;
#[cfg(test)]
fn compact_draft_caches(
    caches: &[FullAttentionCache],
    keep: usize,
) -> anyhow::Result<Vec<FullAttentionCache>> {
    lisa_engine::bindings::dspine::compact_draft_caches(
        &crate::metal_context::execution_context(),
        caches,
        keep,
    )
}
#[cfg(test)]
#[path = "dspine_tests.rs"]
mod tests;
