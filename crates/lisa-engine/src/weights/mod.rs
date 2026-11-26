//! Engine-owned checkpoint imports parameterized by a model-unaware device.
pub mod affine;
pub mod gguf;
pub mod lazy;
pub mod mlx;
pub mod prepare;
pub mod prepared;
pub mod source;
pub mod weight_arena;

mod policy;
pub use policy::{ResidencyBytes, ResidencyPolicy};

pub mod deepseek41;
pub mod flux;
pub mod flux2;
pub mod flux3;
pub mod glm53;
pub mod inkling2;
pub mod laya;
pub mod lfm3;
pub mod olala;
pub mod qwen35;
pub mod qwen4;
pub mod vision;

// Preserve the existing public loader paths while callers adopt model modules.
pub use deepseek41::{engram as deepseek41_engram, plan as deepseek41_weights};
pub use glm53::{affine as glm53_affine, packed as glm53_packed};
pub use qwen4::ple as qwen4_ple;
pub use qwen35::{gguf as qwen35_gguf, mlx as qwen35_mlx};

#[cfg(test)]
mod tests;
