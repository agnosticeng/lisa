use crate::core::cache::IndexerTape;
use lisa_metal::Array;
use lisa_models::glm53::attention::Request;
pub(crate) fn run(request: Request, inputs: &[&Array]) -> anyhow::Result<Vec<Array>> {
    lisa_engine::bindings::glm53::attention::ops::run(
        &crate::metal_context::execution_context(),
        request,
        inputs,
    )
}
pub(crate) fn one(request: Request, inputs: &[&Array]) -> anyhow::Result<Array> {
    lisa_engine::bindings::glm53::attention::ops::one(
        &crate::metal_context::execution_context(),
        request,
        inputs,
    )
}
#[cfg(test)]
#[path = "glm53_attention_tests.rs"]
pub(crate) mod tests;
