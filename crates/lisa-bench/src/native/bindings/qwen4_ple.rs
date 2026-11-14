//! Native validation adapters for portable bindings.
use crate::core::{
    cache::{FullAttentionCache, IndexerTape, LayerCache},
    graph::PreparedGraph,
    loader::TensorSource,
    norm::{RmsNorm, Rotary},
    quant::{QuantizedEmbedding, QuantizedLinear},
};
use lisa_metal::{Array, device::MetalDevice};
pub type PleLayer = lisa_engine::bindings::qwen4_ple::PleLayer<MetalDevice>;
pub type NgramEmbedding = lisa_engine::bindings::qwen4_ple::NgramEmbedding<MetalDevice>;
use half::bf16;
pub use lisa_engine::bindings::qwen4_ple::{NgramConstants, NgramTable, ShardMap};
use memmap2::Mmap;
use std::sync::Arc;
#[cfg(test)]
#[path = "qwen4_ple_tests.rs"]
pub(crate) mod tests;
