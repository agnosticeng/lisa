pub use super::glm53_indexer::Indexer;
use lisa_metal::{Array, Dtype};
pub type Dsa = lisa_engine::bindings::glm53::attention::Dsa<lisa_metal::device::MetalDevice>;
pub type DsaProjections = lisa_engine::bindings::glm53::attention::DsaProjections<Array>;
pub fn causal_mask(x: &Array, b: usize, t: usize, tc: usize) -> anyhow::Result<Array> {
    lisa_engine::bindings::glm53::attention::causal_mask(
        &crate::metal_context::execution_context(),
        x,
        b,
        t,
        tc,
    )
}
pub fn keep_mask_from_selection(
    x: &Array,
    s: &Array,
    t: usize,
    tc: usize,
) -> anyhow::Result<Array> {
    lisa_engine::bindings::glm53::attention::keep_mask_from_selection(
        &crate::metal_context::execution_context(),
        x,
        s,
        t,
        tc,
    )
}
#[cfg(test)]
#[path = "glm53_attention_fixture_tests.rs"]
mod tests;
