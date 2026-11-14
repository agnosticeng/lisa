//! Native application adapters for generic draft plan bindings.
use crate::core::cache::FullAttentionCache;
use lisa_backend::Outputs;
use lisa_metal::Array;
use lisa_models::qwen3_5::dspine::bridge::{Refine, Utility};
use lisa_models::qwen3_5::dspine::layers::Attention;
pub(crate) fn linear(
    spec: lisa_models::qwen3_5::dspine::layers::Linear,
    inputs: &[&Array],
) -> anyhow::Result<Array> {
    lisa_engine::bindings::dspine_layers::linear(
        &crate::metal_context::execution_context(),
        spec,
        inputs,
    )
}
pub(crate) fn prepare(weight: &Array, native_layout: bool) -> anyhow::Result<Array> {
    lisa_engine::bindings::dspine_layers::prepare(
        &crate::metal_context::execution_context(),
        weight,
        native_layout,
    )
}
pub(crate) fn attention(
    spec: Attention,
    inputs: &[&Array],
    cache: &mut FullAttentionCache,
) -> anyhow::Result<Option<Array>> {
    lisa_engine::bindings::dspine_layers::attention(
        &crate::metal_context::execution_context(),
        spec,
        inputs,
        cache,
    )
}
pub(crate) fn small_rows() -> bool {
    lisa_engine::bindings::dspine_layers::small_rows(&crate::metal_context::execution_context())
}
