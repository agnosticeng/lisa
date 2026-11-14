//! Application adapters for portable prepared plans.
use lisa_backend::Outputs;
use lisa_metal::Array;
use lisa_models::qwen3_5::dspine::retrieval::{Head, Rescore};
pub(crate) fn rescore(spec: Rescore, inputs: &[&Array], candidates: bool) -> anyhow::Result<Array> {
    lisa_engine::bindings::dspine_retrieval_plans::rescore(
        &crate::metal_context::execution_context(),
        spec,
        inputs,
        candidates,
    )
}
pub(crate) fn allocate(spec: Head) -> anyhow::Result<Outputs<Array>> {
    lisa_engine::bindings::dspine_retrieval_plans::allocate(
        &crate::metal_context::execution_context(),
        spec,
    )
}
pub(crate) fn chunk(spec: Head, start: i32, end: i32, inputs: &[&Array]) -> anyhow::Result<()> {
    lisa_engine::bindings::dspine_retrieval_plans::chunk(
        &crate::metal_context::execution_context(),
        spec,
        start,
        end,
        inputs,
    )
}
