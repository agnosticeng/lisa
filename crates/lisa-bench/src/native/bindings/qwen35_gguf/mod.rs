//! Native Qwen GGUF execution, independent of the legacy MLX arithmetic law.
pub(crate) mod execution;

mod weights;
pub use lisa_engine::weights::qwen35::gguf::{inspect, tokenizer};

#[cfg(test)]
mod plan_oracle;
#[cfg(test)]
mod plan_tests;

pub use crate::checkpoint::catalog::qwen35::{ENTRY, REPO};
