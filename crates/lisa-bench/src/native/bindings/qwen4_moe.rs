//! Native validation adapters for portable bindings.
use crate::core::{
    cache::{FullAttentionCache, IndexerTape, LayerCache},
    graph::PreparedGraph,
    loader::TensorSource,
    norm::{RmsNorm, Rotary},
    quant::{QuantizedEmbedding, QuantizedLinear},
};
use lisa_metal::{Array, device::MetalDevice};
pub type SparseMoeBlock = lisa_engine::bindings::qwen4_moe::SparseMoeBlock<MetalDevice>;
#[cfg(test)]
#[path = "qwen4_moe_tests.rs"]
pub(crate) mod tests;
