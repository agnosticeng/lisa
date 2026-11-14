//! Sparse-attention, recurrence and routed-kernel contracts ported from DS4.

pub mod gguf;
pub mod kernels;

#[cfg(all(test, feature = "glm53-fixtures"))]
mod test_fixtures;
