//! Native specialization and conformance fixtures for portable GLM bindings.
use anyhow::{Context, Result, ensure};
use lisa_metal::{Array, Dtype, Stream, device::MetalDevice};
pub type Hc = lisa_engine::bindings::glm53_hyper::Hc<MetalDevice>;
use crate::core::{graph::PreparedGraph, loader::TensorSource};
use lisa_models::glm53::hyper::{self as model, Mode, Spec};
fn plans() -> std::cell::RefCell<
    crate::core::graph::PlanCache<MetalDevice, lisa_engine::bindings::glm53_hyper::Key>,
> {
    lisa_engine::bindings::glm53_hyper::plans(&crate::metal_context::execution_context())
}
#[cfg(test)]
#[path = "glm53_hyper_tests.rs"]
mod tests;
