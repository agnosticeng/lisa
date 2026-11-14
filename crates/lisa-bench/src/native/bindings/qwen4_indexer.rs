//! Native validation adapters for portable bindings.
use crate::core::{
    cache::{FullAttentionCache, IndexerTape, LayerCache},
    graph::PreparedGraph,
    loader::TensorSource,
    norm::{RmsNorm, Rotary},
    quant::{QuantizedEmbedding, QuantizedLinear},
};
use lisa_metal::{Array, device::MetalDevice};
pub type QsaIndexer = lisa_engine::bindings::qwen4_indexer::QsaIndexer<MetalDevice>;
pub type QsaSelection = lisa_engine::bindings::qwen4_indexer::QsaSelection<Array>;
#[cfg(test)]
#[path = "qwen4_indexer_tests.rs"]
pub(crate) mod tests;
