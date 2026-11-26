//! Native specialization and conformance fixtures for portable GLM bindings.
#[cfg(test)]
use anyhow::Result;
#[cfg(test)]
use lisa_metal::Array;
#[cfg(test)]
use lisa_metal::Dtype;
use lisa_metal::device::MetalDevice;
pub type Hc = lisa_engine::bindings::glm53::hyper::Hc<MetalDevice>;
#[cfg(test)]
fn plans() -> std::cell::RefCell<
    crate::core::graph::PlanCache<MetalDevice, lisa_engine::bindings::glm53::hyper::Key>,
> {
    lisa_engine::bindings::glm53::hyper::plans(&crate::metal_context::execution_context())
}
#[cfg(test)]
#[path = "glm53_hyper_tests.rs"]
mod tests;
