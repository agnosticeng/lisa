//! Application adapters for portable prepared plans.
use crate::core::quant::QuantizedLinear;
use lisa_metal::Array;
pub type Selected = lisa_engine::bindings::qwen35_readout::Selected<Array>;
pub(crate) fn select(
    hidden: &Array,
    target: &QuantizedLinear,
    coarse: Option<&QuantizedLinear>,
    confidence: bool,
) -> anyhow::Result<Selected> {
    lisa_engine::bindings::qwen35_readout::select(
        &crate::metal_context::execution_context(),
        hidden,
        target,
        coarse,
        confidence,
    )
}
#[cfg(test)]
#[path = "qwen35_readout_tests.rs"]
mod tests;
