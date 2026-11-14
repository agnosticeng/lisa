//! Native specialization and conformance fixtures for portable GLM bindings.
use anyhow::{Context, Result, ensure};
use lisa_metal::{Array, Dtype, Stream, device::MetalDevice};
pub type GgufDsa = lisa_engine::bindings::glm53_packed_attention::GgufDsa<MetalDevice>;
use super::glm53_projection::{Projection, ProjectionExecution};
use crate::core::cache::{FullAttentionCache, IndexerTape};
use lisa_models::glm53::packed_attention as model;
fn run<const N: usize>(r: model::Request, inputs: &[&Array]) -> Result<[Array; N]> {
    lisa_engine::bindings::glm53_packed_attention::run(
        &crate::metal_context::execution_context(),
        r,
        inputs,
    )
}
#[cfg(test)]
#[path = "glm53_packed_attention_tests.rs"]
mod tests;
use super::glm53_indexer::Indexer;
