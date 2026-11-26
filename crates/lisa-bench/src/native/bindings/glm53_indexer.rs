//! Native specialization and conformance fixtures for portable GLM bindings.
use anyhow::Result;
use lisa_metal::Array;
#[cfg(test)]
use lisa_metal::Dtype;
use lisa_metal::device::MetalDevice;
pub type Indexer = lisa_engine::bindings::glm53::attention::indexer::Indexer<MetalDevice>;
#[cfg(test)]
use crate::core::cache::IndexerTape;
pub fn layer_norm(x: &Array, w: &Array, b: &Array, eps: f32) -> Result<Array> {
    lisa_engine::bindings::glm53::attention::indexer::layer_norm(
        &crate::metal_context::execution_context(),
        x,
        w,
        b,
        eps,
    )
}
#[cfg(test)]
pub(crate) fn weighted_rms(x: &Array, w: &Array, eps: f32, latent: bool) -> Result<Array> {
    lisa_engine::bindings::glm53::attention::indexer::weighted_rms(
        &crate::metal_context::execution_context(),
        x,
        w,
        eps,
        latent,
    )
}
#[cfg(test)]
#[path = "glm53_indexer_tests.rs"]
mod tests;
