use crate::core::cache::IndexerTape;
use lisa_metal::Array;
use lisa_models::glm53::attention::Request;
pub(crate) fn run(request: Request, inputs: &[&Array]) -> anyhow::Result<Vec<Array>> {
    lisa_engine::bindings::glm53_attention_ops::run(
        &crate::metal_context::execution_context(),
        request,
        inputs,
    )
}
pub(crate) fn one(request: Request, inputs: &[&Array]) -> anyhow::Result<Array> {
    lisa_engine::bindings::glm53_attention_ops::one(
        &crate::metal_context::execution_context(),
        request,
        inputs,
    )
}
pub(crate) fn normalize(
    inputs: [&Array; 4],
    batch: usize,
    tokens: usize,
    epsilon: f32,
    capture: bool,
    preserve: bool,
    tape: Option<&mut IndexerTape>,
    layer: usize,
) -> anyhow::Result<(Option<Array>, Array, Option<(usize, usize)>)> {
    lisa_engine::bindings::glm53_attention_ops::normalize(
        &crate::metal_context::execution_context(),
        inputs,
        batch,
        tokens,
        epsilon,
        capture,
        preserve,
        tape,
        layer,
    )
}
#[cfg(test)]
#[path = "glm53_attention_tests.rs"]
pub(crate) mod tests;
