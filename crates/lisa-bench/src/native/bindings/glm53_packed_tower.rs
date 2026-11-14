//! Native specialization and conformance fixtures for portable GLM bindings.
use anyhow::{Context, Result, ensure};
use lisa_metal::{Array, Dtype, Stream, device::MetalDevice};
pub type Tower = lisa_engine::bindings::glm53_packed_tower::Tower<MetalDevice>;
use super::glm53_projection::{Projection, ProjectionExecution};
use crate::core::cache::{FullAttentionCache, IndexerTape, LayerCache};
use crate::language::LanguageModel;
use lisa_models::glm53::packed_tower::{self as model, DraftStep, Request, Step};
use std::path::Path;
fn run<const N: usize>(r: Request, inputs: &[&Array]) -> Result<[Array; N]> {
    lisa_engine::bindings::glm53_packed_tower::run(
        &crate::metal_context::execution_context(),
        r,
        inputs,
    )
}
pub trait TowerLoad {
    fn load(path: &Path) -> Result<Self>
    where
        Self: Sized;
}
impl TowerLoad for Tower {
    fn load(path: &Path) -> Result<Self> {
        Self::load_on(crate::metal_language::runtime(), path)
    }
}
#[cfg(test)]
#[path = "glm53_packed_tower_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "glm53_packed_integration_tests.rs"]
mod integration_tests;
