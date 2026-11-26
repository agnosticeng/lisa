//! Native validation adapters for portable bindings.
#[cfg(test)]
use crate::core::{cache::FullAttentionCache, norm::Rotary};
#[cfg(test)]
use lisa_metal::Array;
use lisa_metal::device::MetalDevice;
pub type Attention = lisa_engine::bindings::qwen4::attention::Attention<MetalDevice>;
#[cfg(test)]
#[path = "qwen4_attention_tests.rs"]
pub(crate) mod tests;
