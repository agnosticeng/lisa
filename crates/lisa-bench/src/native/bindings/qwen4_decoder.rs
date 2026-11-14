//! Native validation adapters for portable bindings.
use crate::core::{
    cache::{FullAttentionCache, IndexerTape, LayerCache},
    graph::PreparedGraph,
    loader::TensorSource,
    norm::{RmsNorm, Rotary},
    quant::{QuantizedEmbedding, QuantizedLinear},
};
use lisa_metal::{Array, device::MetalDevice};
pub type DecoderLayer = lisa_engine::bindings::qwen4_decoder::DecoderLayer<MetalDevice>;
pub type Block = lisa_engine::bindings::qwen4_decoder::Block<MetalDevice>;
use super::{
    qwen4_attention::Attention, qwen4_hyper::GatedResidual, qwen4_moe::SparseMoeBlock,
    qwen4_ple::PleLayer,
};
#[cfg(test)]
#[path = "qwen4_decoder_oracle.rs"]
mod original;
#[cfg(test)]
use original::OriginalDecoder;
#[cfg(test)]
#[path = "qwen4_decoder_tests.rs"]
pub(crate) mod tests;
