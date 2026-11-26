//! Native specialization and conformance fixtures for portable GLM bindings.
use lisa_metal::{Array, Dtype, device::MetalDevice};
pub type Kda = lisa_engine::bindings::glm53::kda::recurrent::Kda<MetalDevice>;
#[cfg(test)]
use super::glm53_math::linear_projection;
pub use super::glm53_math::{gated_rms_norm, rms_norm_last};
use super::glm53_projection::Projection;
use crate::core::{cache::GdnCache, capture};
#[cfg(test)]
#[path = "glm53_recurrent_plan_tests.rs"]
pub(crate) mod plan_tests;
#[cfg(test)]
#[path = "glm53_recurrent_tests.rs"]
mod tests;
