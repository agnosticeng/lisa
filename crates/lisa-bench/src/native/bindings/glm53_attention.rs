pub use super::glm53_indexer::Indexer;
use crate::core::{
    cache::{FullAttentionCache, IndexerTape},
    loader::{TensorSource, Weights},
    quant::QuantizedLinear,
};
use lisa_metal::{Array, Dtype};
pub type Dsa = lisa_engine::bindings::glm53_attention::Dsa<lisa_metal::device::MetalDevice>;
pub type DsaProjections = lisa_engine::bindings::glm53_attention::DsaProjections<Array>;
pub fn causal_mask(x: &Array, b: usize, t: usize, tc: usize) -> anyhow::Result<Array> {
    lisa_engine::bindings::glm53_attention::causal_mask(
        &crate::metal_context::execution_context(),
        x,
        b,
        t,
        tc,
    )
}
pub fn keep_mask_from_selection(
    x: &Array,
    s: &Array,
    t: usize,
    tc: usize,
) -> anyhow::Result<Array> {
    lisa_engine::bindings::glm53_attention::keep_mask_from_selection(
        &crate::metal_context::execution_context(),
        x,
        s,
        t,
        tc,
    )
}
#[cfg(test)]
fn glm53_weighted_rms_norm(x: &Array, w: &Array, e: f32) -> anyhow::Result<Array> {
    super::glm53_indexer::weighted_rms(x, w, e, false)
}
#[cfg(test)]
const LAYER_NORM_EPS: f32 = lisa_models::glm53::indexer::LAYER_NORM_EPS;
#[cfg(all(test, feature = "glm53-fixtures"))]
pub use lisa_engine::bindings::glm53_attention::{
    mtp_real_target_fixtures_enabled, use_mtp_real_target_fixtures,
};
#[cfg(test)]
#[path = "glm53_attention_fixture_tests.rs"]
mod tests;
