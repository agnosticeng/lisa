//! Application adapters for portable prepared plans.
use crate::core::quant::QuantizedLinear;
use lisa_metal::Array;
pub type Selected = lisa_engine::bindings::qwen35::readout::Selected<Array>;
pub(crate) fn select(
    hidden: &Array,
    target: &QuantizedLinear,
    coarse: Option<&QuantizedLinear>,
) -> anyhow::Result<Selected> {
    lisa_engine::bindings::qwen35::readout::select(
        &crate::metal_context::execution_context(),
        hidden,
        target,
        coarse,
    )
}
#[cfg(test)]
#[path = "qwen35_readout_tests.rs"]
mod tests;
