//! Device-independent inference execution and state management.
//!
//! Model computation plans live in `lisa-models`. This crate imports original
//! checkpoint payloads, executes plans through `lisa-backend`, and owns shared
//! generation, batching, speculation and session policy. It has no concrete
//! device dependency. Applications choose providers through `loading`; native
//! provider construction lives at the application composition boundary.

pub mod batching;
pub mod checkpoint;
pub mod conversation;
pub mod core;
pub mod decision;
pub mod device;
pub mod eos;
pub mod language;
pub mod laya;
pub mod loading;

use std::sync::OnceLock;
use std::time::Instant;

/// Process-start anchor for the TTFT decomposition ([`ttft_mark`]): every
/// mark prints seconds since the earliest reachable point of `main`.
static TTFT_T0: OnceLock<Instant> = OnceLock::new();

/// First-time initialization of the TTFT anchor. Call at the top of `main`.
pub fn ttft_init() {
    let _ = TTFT_T0.set(Instant::now());
}

/// One stderr milestone line for the TTFT decomposition (specs/05): phase
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

pub mod generation;
pub mod speculation;
pub mod speculative_runtime;

pub mod scheduling;

pub mod weights;

pub mod execution;

pub mod storage;

pub mod memory;

pub mod indexer_cache;

pub mod attention_cache;

pub mod recurrent;

pub mod recurrent_cache;

pub mod cache;

pub mod prefix_cache;

pub mod sampling;
pub mod tensor_ops;

pub mod cache_batching;

pub mod bindings;

pub mod language_runtime;

pub mod sampler;

pub mod vision;

pub mod image_generation;

pub mod cli;

pub mod model_binding;

pub mod conversation_cache;

pub mod session_check;
