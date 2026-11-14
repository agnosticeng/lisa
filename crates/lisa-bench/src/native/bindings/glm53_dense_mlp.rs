use crate::core::quant::QuantizedLinear;
use lisa_metal::{Array, Dtype};
pub type DenseMlp =
    lisa_engine::bindings::glm53_dense_mlp::DenseMlp<lisa_metal::device::MetalDevice>;
pub type GateUpProjection = lisa_engine::bindings::glm53_dense_mlp::GateUpProjection<Array>;
pub fn fuse_gate_up(g: &QuantizedLinear, u: &QuantizedLinear) -> anyhow::Result<QuantizedLinear> {
    lisa_engine::bindings::glm53_dense_mlp::fuse_gate_up(
        &crate::metal_context::execution_context(),
        g,
        u,
    )
}
pub fn split_gate_up(x: &Array, n: usize) -> anyhow::Result<(Array, Array)> {
    lisa_engine::bindings::glm53_dense_mlp::split_gate_up(
        &crate::metal_context::execution_context(),
        x,
        n,
    )
}
pub fn clamped_swiglu(u: &Array, g: &Array, l: f32) -> anyhow::Result<Array> {
    lisa_engine::bindings::glm53_dense_mlp::clamped_swiglu(
        &crate::metal_context::execution_context(),
        u,
        g,
        l,
    )
}
pub fn split_gate_up_linear(
    w: &QuantizedLinear,
    n: usize,
) -> anyhow::Result<(QuantizedLinear, QuantizedLinear)> {
    lisa_engine::bindings::glm53_dense_mlp::split_gate_up_linear(
        &crate::metal_context::execution_context(),
        w,
        n,
    )
}
#[cfg(test)]
#[path = "glm53_dense_mlp_tests.rs"]
mod tests;
