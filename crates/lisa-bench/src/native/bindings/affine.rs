//! Application adapters for portable prepared plans.
use lisa_metal::Array;
pub(crate) fn requantize(
    weight: &Array,
    scales: &Array,
    biases: &Array,
    source: (i32, i32),
    target: (i32, i32),
) -> anyhow::Result<(Array, Array, Array)> {
    lisa_engine::bindings::shared::affine::requantize(
        &crate::metal_context::execution_context(),
        weight,
        scales,
        biases,
        source,
        target,
    )
}
