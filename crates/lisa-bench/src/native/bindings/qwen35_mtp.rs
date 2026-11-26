//! Native application alias and independent MTP conformance tests.
pub type MtpHead = lisa_engine::bindings::qwen35::mtp::MtpHead<lisa_metal::device::MetalDevice>;
#[cfg(test)]
use crate::core::{norm::RmsNormExecution, quant::QuantizedEmbedding};
#[cfg(test)]
use lisa_metal::Array;
#[cfg(test)]
#[path = "qwen35_mtp_tests.rs"]
mod tests;
