//! Native GGUF adapters for the pinned DS4 DeepSeek V4.1 execution port.
//!
//! The text graph is integrated with `LanguageModel`, CLI, HTTP and UI.
//! Short stage captures and logits through an 8199-token prompt match the
//! pinned same-weight DS4 adapter on the measured workloads.
//! Wide GEMM prefill and broader context/performance certification remain
//! open; see specs/models/deepseek41.md for the acceptance boundary.

pub use crate::bindings::deepseek41_attention as attention;
pub use crate::bindings::deepseek41_engram as engram;
pub use crate::bindings::deepseek41_graph as graph;
pub use crate::bindings::deepseek41_moe as moe;
pub use crate::bindings::deepseek41_projection as projection;
pub use crate::bindings::deepseek41_state as state;
pub use lisa_models::deepseek41::config;
pub mod tokenizer {
    pub use crate::core::gguf_tokenizer::Tokenizer;
}
pub use crate::bindings::deepseek41_tower as tower;
pub use crate::bindings::deepseek41_weights as weights;

#[cfg(test)]
#[path = "deepseek41_kernel_tests.rs"]
mod kernel_tests;

pub use lisa_engine::bindings::deepseek41_tower::DS4_REVISION;

pub use crate::checkpoint::catalog::deepseek41::{GGUF_ENTRY, REPO};
