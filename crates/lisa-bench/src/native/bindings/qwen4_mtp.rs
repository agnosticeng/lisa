//! Native validation adapters for portable bindings.
use crate::core::{
    cache::{FullAttentionCache, IndexerTape, LayerCache},
    graph::PreparedGraph,
    loader::TensorSource,
    norm::{RmsNorm, Rotary},
    quant::{QuantizedEmbedding, QuantizedLinear},
};
use lisa_metal::{Array, device::MetalDevice};
pub type MtpHead = lisa_engine::bindings::qwen4_mtp::MtpHead<MetalDevice>;
