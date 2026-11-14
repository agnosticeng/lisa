//! Native specialization and conformance fixtures for portable GLM bindings.
use anyhow::{Context, Result, ensure};
use lisa_metal::{Array, Dtype, Stream, device::MetalDevice};
pub type Indexer = lisa_engine::bindings::glm53_indexer::Indexer<MetalDevice>;
use crate::core::{cache::IndexerTape, capture::Capture};
use lisa_models::glm53::indexer::{self as model, Projection as P, Request as R, Spec};
pub(crate) fn run<const N: usize>(request: R, inputs: &[&Array]) -> Result<[Array; N]> {
    lisa_engine::bindings::glm53_indexer::run(
        &crate::metal_context::execution_context(),
        request,
        inputs,
    )
}
pub fn layer_norm(x: &Array, w: &Array, b: &Array, eps: f32) -> Result<Array> {
    lisa_engine::bindings::glm53_indexer::layer_norm(
        &crate::metal_context::execution_context(),
        x,
        w,
        b,
        eps,
    )
}
#[cfg(test)]
pub(crate) fn weighted_rms(x: &Array, w: &Array, eps: f32, latent: bool) -> Result<Array> {
    lisa_engine::bindings::glm53_indexer::weighted_rms(
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
