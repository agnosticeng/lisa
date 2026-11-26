//! Native specialization and conformance fixtures for portable GLM bindings.
use anyhow::Result;
use lisa_metal::{Array, device::MetalDevice};
pub type GgufDsa = lisa_engine::bindings::glm53::packed::attention::GgufDsa<MetalDevice>;
use super::glm53_projection::Projection;
use crate::core::cache::IndexerTape;
#[cfg(test)]
#[path = "glm53_packed_attention_tests.rs"]
mod tests;
use super::glm53_indexer::Indexer;
