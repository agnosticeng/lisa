//! Qwen 3.8 Flash-Next (`qwen4_exp_text`): the hybrid full-attention + gated
//! deltanet text tower, its QSA sparse-attention indexer, hyper-connections,
//! sparse MoE, the PLE n-gram layer, and the embedded MTP head.

pub mod attention;
pub mod config;
pub mod gdn;
pub mod hyper;
pub mod indexer;
pub mod layerdiff;
pub mod moe;
pub mod mtp;
pub mod ple;
pub mod smoke;
pub mod speculate;
pub mod tower;

pub use config::ModelConfig;
pub use tower::Tower;
