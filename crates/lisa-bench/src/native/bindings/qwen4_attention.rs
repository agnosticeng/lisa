//! Native validation adapters for portable bindings.
use crate::core::{
    cache::{FullAttentionCache, IndexerTape, LayerCache},
    graph::PreparedGraph,
    loader::TensorSource,
    norm::{RmsNorm, Rotary},
    quant::{QuantizedEmbedding, QuantizedLinear},
};
use lisa_metal::{Array, device::MetalDevice};
pub type Attention = lisa_engine::bindings::qwen4_attention::Attention<MetalDevice>;
#[cfg(test)]
#[path = "qwen4_attention_tests.rs"]
pub(crate) mod tests;
