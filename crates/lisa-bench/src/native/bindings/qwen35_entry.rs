//! Native application adapter and input-plan conformance tests.
use crate::core::{norm::Rotary, quant::QuantizedEmbedding};
use lisa_metal::Array;
pub(crate) fn begin(
    ids: &Array,
    embedding: &QuantizedEmbedding,
    rope: &Rotary,
    vision: Option<&crate::vision::Vision>,
    offset: usize,
    count: usize,
    next: Option<&[usize]>,
) -> anyhow::Result<(Array, Array, Array)> {
    lisa_engine::bindings::qwen35_entry::begin(
        &crate::metal_context::execution_context(),
        ids,
        embedding,
        &rope.0,
        vision,
        offset,
        count,
        next,
    )
}
pub(crate) fn capture_hidden(x: &Array, residual: &Array) -> anyhow::Result<Array> {
    lisa_engine::bindings::qwen35_entry::capture_hidden(
        &crate::metal_context::execution_context(),
        x,
        residual,
    )
}
#[cfg(test)]
#[path = "qwen35_entry_tests.rs"]
mod tests;
