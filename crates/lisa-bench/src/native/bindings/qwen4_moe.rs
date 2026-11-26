//! Native validation adapters for portable bindings.
#[cfg(test)]
use lisa_metal::Array;
use lisa_metal::device::MetalDevice;
pub type SparseMoeBlock = lisa_engine::bindings::qwen4::moe::SparseMoeBlock<MetalDevice>;
#[cfg(test)]
#[path = "qwen4_moe_tests.rs"]
pub(crate) mod tests;
