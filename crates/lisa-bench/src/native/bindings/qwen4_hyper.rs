//! Native validation adapters for portable bindings.
use crate::core::{
    cache::{FullAttentionCache, IndexerTape, LayerCache},
    graph::PreparedGraph,
    loader::TensorSource,
    norm::{RmsNorm, Rotary},
    quant::{QuantizedEmbedding, QuantizedLinear},
};
use lisa_metal::{Array, device::MetalDevice};
pub type GatedResidual = lisa_engine::bindings::qwen4_hyper::GatedResidual<MetalDevice>;
use lisa_models::qwen4::hyper as model;
pub fn inject(residual: &Array, output: &Array, inject: &Array) -> anyhow::Result<Array> {
    lisa_engine::bindings::qwen4_hyper::inject(
        &crate::metal_context::execution_context(),
        residual,
        output,
        inject,
    )
}
pub fn inject_norm(
    residual: &Array,
    out: Option<&Array>,
    inject: Option<&Array>,
    scale: &Array,
    hidden: i32,
    streams: i32,
    epsilon: f32,
) -> anyhow::Result<(Array, Array)> {
    lisa_engine::bindings::qwen4_hyper::inject_norm(
        &crate::metal_context::execution_context(),
        residual,
        out,
        inject,
        scale,
        hidden,
        streams,
        epsilon,
    )
}
#[cfg(test)]
#[path = "qwen4_hyper_tests.rs"]
pub(crate) mod tests;
