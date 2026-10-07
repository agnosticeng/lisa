//! lisa — a from-scratch Rust inference engine for Qwen 3.8 Flash-Next (and,
//! structurally, other decoder-only models).
//!
//! - [`core`] is model-agnostic: loading, quantization, norm/rope primitives,
//!   the layer caches, sampling, tokenization, and the generation / batching /
//!   scheduling loops.
//! - [`models`] holds the model implementations. The runtime drives them
//!   through [`models::LanguageModel`]; Qwen 3.8 Flash-Next lives in
//!   [`models::qwen4`].

pub mod cli;
pub mod core;
pub mod models;

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
pub use core::{batch, generate, loader, mem, norm, quant, sampler, sched, session, tokenizer};
pub use models::{DecisionModel, LanguageModel, Loaded, laya, qwen4};
