//! lisa — a from-scratch Rust inference engine for Qwen 3.8 Flash-Next (and,
//! structurally, other decoder-only models).
//!
//! - [`core`] is model-agnostic: loading, quantization, norm/rope primitives,
//!   the layer caches, sampling, tokenization, and the generation / batching /
//!   scheduling loops.
//! - [`models`] holds the model implementations. The runtime drives them
//!   through [`models::LanguageModel`]; Qwen 3.8 Flash-Next lives in
//!   [`models::qwen4`].

pub mod core;
pub mod models;

// Convenience re-exports for the common entry points.
pub use core::{batch, generate, loader, mem, norm, quant, sampler, sched, session, tokenizer};
pub use models::{laya, qwen4, DecisionModel, LanguageModel, Loaded};
