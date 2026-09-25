//! Qwen 3.8 27B (`qwen3_5_text`): the dense hybrid GDN + full-attention tower
//! with a native MTP head. A smaller, simpler sibling of `qwen4_exp` — no MoE,
//! no hyper-connection, no PLE, no QSA indexer.

pub mod attention;
pub mod config;
pub mod tower;

pub use config::Qwen35Config;
pub use tower::Qwen35Tower;
