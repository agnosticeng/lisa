//! Native GGUF adapters for the pinned DS4 DeepSeek V4.1 execution port.
//!
//! The text graph is integrated with `LanguageModel`, CLI, HTTP and UI.
//! Short stage captures and logits through an 8199-token prompt match the
//! pinned same-weight DS4 adapter on the measured workloads.
//! Wide GEMM prefill and broader context/performance certification remain
//! open; see specs/models/deepseek41.md for the acceptance boundary.

pub mod attention;
pub mod config;
pub mod engram;
pub mod graph;
pub mod moe;
pub mod projection;
pub mod state;
pub mod tokenizer;
pub mod tower;
pub mod weights;

#[cfg(test)]
mod kernel_tests;

pub const DS4_REVISION: &str = "b3f50180ec2c7324892be1f50bc83555ccb8d5e7";
pub const REPO: &str = "ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-Coder-GGUF";
pub const GGUF_ENTRY: &str =
    "IQ2_XXS/DeepSeek-V4.1-Flash-1017-GSQ-RCO-Coder-IQ2_XXS-00001-of-00006.gguf";
