//! Application adapters for portable prepared plans.
use lisa_metal::Array;
pub(crate) fn lookup(
    ids: &Array,
    weight: &Array,
    scales: &Array,
    biases: &Array,
    group_size: i32,
    bits: i32,
) -> anyhow::Result<Array> {
    lisa_engine::bindings::affine::lookup(
        &crate::metal_context::execution_context(),
        ids,
        weight,
        scales,
        biases,
        group_size,
        bits,
    )
}
pub(crate) fn requantize(
    weight: &Array,
    scales: &Array,
    biases: &Array,
    source: (i32, i32),
    target: (i32, i32),
) -> anyhow::Result<(Array, Array, Array)> {
    lisa_engine::bindings::affine::requantize(
        &crate::metal_context::execution_context(),
        weight,
        scales,
        biases,
        source,
        target,
    )
}
pub(crate) fn linear(x: &Array, p: &crate::core::quant::QuantizedLinear) -> anyhow::Result<Array> {
    lisa_engine::bindings::affine::linear(&crate::metal_context::execution_context(), x, p)
}
