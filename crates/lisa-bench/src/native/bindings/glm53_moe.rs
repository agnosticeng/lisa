//! Native specialization and conformance fixtures for portable GLM bindings.
use anyhow::{Context, Result, ensure};
use lisa_metal::{Array, Dtype, Stream, device::MetalDevice};
pub type Mlp = lisa_engine::bindings::glm53_moe::Mlp<MetalDevice>;
use super::glm53_projection::{Projection, ProjectionExecution};
use lisa_models::glm53::packed as model;
#[cfg(test)]
#[path = "glm53_moe_tests.rs"]
mod tests;
