use crate::core::quant::QuantizedLinear;
use lisa_metal::Array;
pub fn forward(w: &QuantizedLinear, x: &Array, native: bool) -> anyhow::Result<Array> {
    lisa_engine::bindings::glm53::affine::projection::forward(
        &crate::metal_context::execution_context(),
        w,
        x,
        native,
    )
}
#[cfg(test)]
#[path = "glm53_affine_projection_tests.rs"]
mod tests;
