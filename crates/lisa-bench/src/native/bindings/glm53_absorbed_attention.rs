use super::glm53_attention_mode::*;
use lisa_metal::{Array, Stream};
pub fn dsa_absorbed_attention(
    q: &Array,
    kv_weight: &Array,
    kv_scales: &Array,
    kv_biases: &Array,
    group_size: usize,
    compact_latents: &Array,
    selected: &Array,
    scale: f32,
    stream: &Stream,
) -> Option<Array> {
    lisa_engine::bindings::glm53_absorbed_attention::dsa_absorbed_attention(
        q,
        kv_weight,
        kv_scales,
        kv_biases,
        group_size,
        compact_latents,
        selected,
        scale,
        &crate::metal_context::execution_context(),
    )
}
pub fn dsa_absorbed_attention_decode(
    q: &Array,
    kv_weight: &Array,
    kv_scales: &Array,
    kv_biases: &Array,
    group_size: usize,
    compact_latents: &Array,
    selected: Option<&Array>,
    selected_valid_prefix: usize,
    scale: f32,
    stream: &Stream,
) -> Option<Array> {
    lisa_engine::bindings::glm53_absorbed_attention::dsa_absorbed_attention_decode(
        q,
        kv_weight,
        kv_scales,
        kv_biases,
        group_size,
        compact_latents,
        selected,
        selected_valid_prefix,
        scale,
        &crate::metal_context::execution_context(),
    )
}
pub fn dsa_absorbed_attention_decode_capture(
    q: &Array,
    kv_weight: &Array,
    kv_scales: &Array,
    kv_biases: &Array,
    group_size: usize,
    compact_latents: &Array,
    selected: Option<&Array>,
    selected_valid_prefix: usize,
    scale: f32,
    stream: &Stream,
) -> Option<(Array, Array)> {
    lisa_engine::bindings::glm53_absorbed_attention::dsa_absorbed_attention_decode_capture(
        q,
        kv_weight,
        kv_scales,
        kv_biases,
        group_size,
        compact_latents,
        selected,
        selected_valid_prefix,
        scale,
        &crate::metal_context::execution_context(),
    )
}
pub fn dsa_absorbed_attention_with_valid_prefix(
    q: &Array,
    kv_weight: &Array,
    kv_scales: &Array,
    kv_biases: &Array,
    group_size: usize,
    compact_latents: &Array,
    selected: &Array,
    selected_valid_prefix: usize,
    scale: f32,
    stream: &Stream,
    prefill_encoder_boundaries: bool,
) -> Option<Array> {
    lisa_engine::bindings::glm53_absorbed_attention::dsa_absorbed_attention_with_valid_prefix(
        q,
        kv_weight,
        kv_scales,
        kv_biases,
        group_size,
        compact_latents,
        selected,
        selected_valid_prefix,
        scale,
        &crate::metal_context::execution_context(),
        prefill_encoder_boundaries,
    )
}
pub fn dsa_absorbed_attention_dense(
    q: &Array,
    kv_weight: &Array,
    kv_scales: &Array,
    kv_biases: &Array,
    group_size: usize,
    compact_latents: &Array,
    scale: f32,
    stream: &Stream,
) -> Option<Array> {
    lisa_engine::bindings::glm53_absorbed_attention::dsa_absorbed_attention_dense(
        q,
        kv_weight,
        kv_scales,
        kv_biases,
        group_size,
        compact_latents,
        scale,
        &crate::metal_context::execution_context(),
    )
}
pub fn dsa_absorbed_attention_dense_stages(
    q: &Array,
    kv_weight: &Array,
    kv_scales: &Array,
    kv_biases: &Array,
    group_size: usize,
    compact_latents: &Array,
    scale: f32,
    stream: &Stream,
) -> Option<(Array, Array, Array)> {
    lisa_engine::bindings::glm53_absorbed_attention::dsa_absorbed_attention_dense_stages(
        q,
        kv_weight,
        kv_scales,
        kv_biases,
        group_size,
        compact_latents,
        scale,
        &crate::metal_context::execution_context(),
    )
}
pub fn dsa_absorbed_attention_stages(
    q: &Array,
    kv_weight: &Array,
    kv_scales: &Array,
    kv_biases: &Array,
    group_size: usize,
    compact_latents: &Array,
    selected: &Array,
    scale: f32,
    stream: &Stream,
    prefill_encoder_boundaries: bool,
) -> Option<(Array, Array, Array)> {
    lisa_engine::bindings::glm53_absorbed_attention::dsa_absorbed_attention_stages(
        q,
        kv_weight,
        kv_scales,
        kv_biases,
        group_size,
        compact_latents,
        selected,
        scale,
        &crate::metal_context::execution_context(),
        prefill_encoder_boundaries,
    )
}
pub fn dsa_absorbed_attention_flash_causal_prefill_stages(
    q: &Array,
    w: &Array,
    s: &Array,
    b: &Array,
    group: usize,
    cache: &Array,
    position: usize,
    scale: f32,
    stream: &Stream,
    boundaries: bool,
) -> Option<(Array, Array, Array)> {
    lisa_engine::bindings::glm53_absorbed_attention::dsa_absorbed_attention_flash_causal_prefill_stages(q,w,s,b,group,cache,position,scale,&crate::metal_context::execution_context(),boundaries)
}
#[cfg(test)]
pub fn value_rows(
    x: &Array,
    w: &Array,
    s: &Array,
    b: &Array,
    group: usize,
    heads: usize,
    stream: &Stream,
) -> Option<Array> {
    lisa_engine::bindings::glm53_absorbed_attention::value_rows(
        x,
        w,
        s,
        b,
        group,
        heads,
        &crate::metal_context::execution_context(),
    )
}
#[cfg(test)]
#[path = "glm53_absorbed_attention_tests.rs"]
mod tests;

fn dsa_absorbed_attention_stages_inner(
    q: &Array,
    w: &Array,
    s: &Array,
    b: &Array,
    group: usize,
    cache: &Array,
    selected: Option<&Array>,
    prefix: usize,
    scale: f32,
    stream: &Stream,
    capture: bool,
    fused: bool,
    capture_fused: bool,
    boundaries: bool,
) -> Option<(Array, Option<Array>, Option<Array>)> {
    lisa_engine::bindings::glm53_absorbed_attention::dsa_absorbed_attention_stages_inner(
        q,
        w,
        s,
        b,
        group,
        cache,
        selected,
        prefix,
        scale,
        &crate::metal_context::execution_context(),
        capture,
        fused,
        capture_fused,
        boundaries,
    )
}
