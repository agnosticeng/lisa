#[cfg(test)]
use crate::core::quant::QuantizedEmbedding;
#[cfg(test)]
use crate::core::quant::QuantizedLinear;
#[cfg(test)]
use lisa_metal::Array;
#[cfg(test)]
use lisa_models::glm53::affine_tower::Request;
#[cfg(test)]
pub fn run(r: Request, a: &[&Array]) -> anyhow::Result<Vec<Array>> {
    lisa_engine::bindings::glm53::affine::tower::ops::run(
        &crate::metal_context::execution_context(),
        r,
        a,
    )
}
#[cfg(test)]
pub fn one(r: Request, a: &[&Array]) -> anyhow::Result<Array> {
    lisa_engine::bindings::glm53::affine::tower::ops::one(
        &crate::metal_context::execution_context(),
        r,
        a,
    )
}
#[cfg(test)]
pub fn embedding(w: &QuantizedEmbedding, a: &Array) -> anyhow::Result<Array> {
    lisa_engine::bindings::glm53::affine::tower::ops::embedding(
        &crate::metal_context::execution_context(),
        w,
        a,
    )
}
#[cfg(test)]
pub fn head(w: &QuantizedLinear, a: &Array) -> anyhow::Result<Array> {
    lisa_engine::bindings::glm53::affine::tower::ops::head(
        &crate::metal_context::execution_context(),
        w,
        a,
    )
}
