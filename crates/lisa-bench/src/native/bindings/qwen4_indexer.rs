//! Native validation adapters for portable bindings.
#[cfg(test)]
use crate::core::{
    cache::IndexerTape,
    norm::{RmsNorm, Rotary},
    quant::QuantizedLinear,
};
use lisa_metal::{Array, device::MetalDevice};
pub type QsaIndexer = lisa_engine::bindings::qwen4::indexer::QsaIndexer<MetalDevice>;
pub type QsaSelection = lisa_engine::bindings::qwen4::indexer::QsaSelection<Array>;
#[cfg(test)]
#[path = "qwen4_indexer_tests.rs"]
pub(crate) mod tests;
