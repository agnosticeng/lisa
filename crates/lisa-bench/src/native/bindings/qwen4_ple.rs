//! Native validation adapters for portable bindings.
#[cfg(test)]
use crate::core::{norm::RmsNorm, quant::QuantizedLinear};
#[cfg(test)]
use lisa_metal::Array;
use lisa_metal::device::MetalDevice;
pub type PleLayer = lisa_engine::bindings::qwen4::ple::PleLayer<MetalDevice>;
pub type NgramEmbedding = lisa_engine::bindings::qwen4::ple::NgramEmbedding<MetalDevice>;
#[cfg(test)]
use half::bf16;
pub use lisa_engine::bindings::qwen4::ple::NgramConstants;
pub use lisa_engine::weights::qwen4::ple::{NgramTable, ShardMap};
#[cfg(test)]
use memmap2::Mmap;
#[cfg(test)]
use std::sync::Arc;
#[cfg(test)]
#[path = "qwen4_ple_tests.rs"]
pub(crate) mod tests;
