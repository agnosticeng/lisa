//! Native application adapters for generic draft plan bindings.
use crate::core::cache::FullAttentionCache;
use lisa_backend::Outputs;
use lisa_metal::Array;
use lisa_models::qwen3_5::dspine::bridge::{Refine, Utility};
use lisa_models::qwen3_5::dspine::layers::Attention;
pub(crate) fn bridge(
    spec: lisa_models::qwen3_5::dspine::bridge::Bridge,
    inputs: &[&Array],
) -> anyhow::Result<Outputs<Array>> {
    lisa_engine::bindings::dspine_bridge::bridge(
        &crate::metal_context::execution_context(),
        spec,
        inputs,
    )
}
pub(crate) fn refine(spec: Refine, inputs: &[&Array]) -> anyhow::Result<Array> {
    lisa_engine::bindings::dspine_bridge::refine(
        &crate::metal_context::execution_context(),
        spec,
        inputs,
    )
}
pub(crate) fn utility(spec: Utility, inputs: &[&Array]) -> anyhow::Result<Outputs<Array>> {
    lisa_engine::bindings::dspine_bridge::utility(
        &crate::metal_context::execution_context(),
        spec,
        inputs,
    )
}
pub(crate) fn one(spec: Utility, inputs: &[&Array]) -> anyhow::Result<Array> {
    lisa_engine::bindings::dspine_bridge::one(
        &crate::metal_context::execution_context(),
        spec,
        inputs,
    )
}
pub(crate) fn block(
    spec: lisa_models::qwen3_5::dspine::block::Block,
    inputs: &[&Array],
) -> anyhow::Result<Outputs<Array>> {
    lisa_engine::bindings::dspine_bridge::block(
        &crate::metal_context::execution_context(),
        spec,
        inputs,
    )
}
pub(crate) fn state(
    spec: lisa_models::qwen3_5::dspine::state::Plan,
    inputs: &[&Array],
) -> anyhow::Result<Outputs<Array>> {
    lisa_engine::bindings::dspine_bridge::state(
        &crate::metal_context::execution_context(),
        spec,
        inputs,
    )
}
pub(crate) fn context(
    spec: lisa_models::qwen3_5::dspine::block::Context,
    inputs: &[&Array],
) -> anyhow::Result<Outputs<Array>> {
    lisa_engine::bindings::dspine_bridge::context(
        &crate::metal_context::execution_context(),
        spec,
        inputs,
    )
}
