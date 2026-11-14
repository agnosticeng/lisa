use crate::core::quant::{QuantizedEmbedding, QuantizedLinear};
use lisa_metal::Array;
use lisa_models::glm53::affine_tower::Request;
pub fn run(r: Request, a: &[&Array]) -> anyhow::Result<Vec<Array>> {
    lisa_engine::bindings::glm53_affine_tower_ops::run(
        &crate::metal_context::execution_context(),
        r,
        a,
    )
}
pub fn one(r: Request, a: &[&Array]) -> anyhow::Result<Array> {
    lisa_engine::bindings::glm53_affine_tower_ops::one(
        &crate::metal_context::execution_context(),
        r,
        a,
    )
}
pub fn embedding(w: &QuantizedEmbedding, a: &Array) -> anyhow::Result<Array> {
    lisa_engine::bindings::glm53_affine_tower_ops::embedding(
        &crate::metal_context::execution_context(),
        w,
        a,
    )
}
pub fn head(w: &QuantizedLinear, a: &Array) -> anyhow::Result<Array> {
    lisa_engine::bindings::glm53_affine_tower_ops::head(
        &crate::metal_context::execution_context(),
        w,
        a,
    )
}
