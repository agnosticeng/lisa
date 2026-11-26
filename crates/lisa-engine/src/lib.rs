//! Device-independent inference execution and state management.
//!
//! Model computation plans live in `lisa-models`. This crate imports original
//! checkpoint payloads, executes plans through `lisa-backend`, and owns shared
//! generation, batching, speculation and session policy. It has no concrete
//! device dependency. Applications choose providers through `loading`; native
//! provider construction lives at the application composition boundary.

pub mod batching;
pub mod bindings;
pub mod cache;
pub mod checkpoint;
pub mod cli;
pub mod conversation;
pub mod core;
pub mod decision;
pub mod device;
pub mod eos;
pub mod execution;
pub mod generation;
pub mod image_generation;
pub mod language;
pub mod laya;
pub mod loading;
pub mod memory;
pub mod model_binding;
pub mod recurrent;
pub mod sampling;
pub mod scheduling;
pub mod session_check;
pub mod speculation;
pub mod storage;
pub mod tensor_ops;
pub mod vision;
pub mod weights;

// Public aliases for callers using the flat module paths.
pub use cache::{
    attention as attention_cache, batching as cache_batching, indexer as indexer_cache,
    prefix as prefix_cache, recurrent as recurrent_cache,
};
pub use conversation::cache as conversation_cache;
pub use language::runtime as language_runtime;
pub use sampling::sampler;
pub use speculation::runtime as speculative_runtime;

use std::sync::OnceLock;
use std::time::Instant;

/// Process-start anchor for the TTFT decomposition ([`ttft_mark`]): every
/// mark prints seconds since the earliest reachable point of `main`.
static TTFT_T0: OnceLock<Instant> = OnceLock::new();

/// First-time initialization of the TTFT anchor. Call at the top of `main`.
pub fn ttft_init() {
    let _ = TTFT_T0.set(Instant::now());
}

/// One stderr milestone line for the TTFT decomposition: phase
/// name plus seconds since process start. One line per phase, per process.
pub fn ttft_mark(label: &str) {
    if let Some(t0) = TTFT_T0.get() {
        eprintln!("[ttft] +{:8.3}s  {label}", t0.elapsed().as_secs_f64());
    }
}

// Convenience re-exports for the common entry points.
pub use core::tokenizer;
pub use decision::DecisionModel;
pub use language::LanguageModel;
pub use loading::Loaded;

#[cfg(test)]
mod boundary_tests;
