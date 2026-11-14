//! Engine-owned checkpoint imports parameterized by a model-unaware device.
pub mod gguf;
pub mod lazy;
pub mod mlx;
#[cfg(test)]
mod tests;

mod policy;
pub use policy::{ResidencyBytes, ResidencyPolicy};

pub mod affine;
pub mod source;
