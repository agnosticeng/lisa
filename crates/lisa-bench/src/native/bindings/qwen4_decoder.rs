//! Native validation adapters for portable bindings.
#[cfg(test)]
use crate::core::{cache::LayerCache, graph::PreparedGraph, norm::Rotary};
#[cfg(test)]
use lisa_metal::Array;
use lisa_metal::device::MetalDevice;
pub type DecoderLayer = lisa_engine::bindings::qwen4::decoder::DecoderLayer<MetalDevice>;
pub type Block = lisa_engine::bindings::qwen4::decoder::Block<MetalDevice>;
#[cfg(test)]
use super::{qwen4_attention::Attention, qwen4_hyper::GatedResidual, qwen4_moe::SparseMoeBlock};
#[cfg(test)]
#[path = "qwen4_decoder_oracle.rs"]
mod original;
#[cfg(test)]
use original::OriginalDecoder;
#[cfg(test)]
#[path = "qwen4_decoder_tests.rs"]
pub(crate) mod tests;
