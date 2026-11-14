//! Native application alias and independent retrieval conformance tests.
pub use lisa_engine::bindings::dspine_retrieval::enabled;
pub type Retrieval =
    lisa_engine::bindings::dspine_retrieval::Retrieval<lisa_metal::device::MetalDevice>;
#[cfg(test)]
use crate::core::quant::QuantizedLinear;
#[cfg(test)]
use anyhow::Context;
#[cfg(test)]
use lisa_metal::Array;
#[cfg(test)]
#[path = "dspine_retrieval_tests.rs"]
mod tests;
