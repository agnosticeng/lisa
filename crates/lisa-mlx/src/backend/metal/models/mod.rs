//! Model-specific kernels and their host wrappers.
//!
//! Each model family gets a module here; a backend only supplies the generic
//! runtime and op surface. The wrappers are re-exported at the crate root so
//! the engine reads `lisa_mlx::kernels::…` regardless of where they live.

pub mod qwen4;
