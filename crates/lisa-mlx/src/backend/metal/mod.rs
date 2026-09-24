//! The Metal backend: objc2 Metal runtime, the eager `Array`/op surface, the
//! generic Metal kernels, and the model-specific kernel wrappers.
//!
//! Modules are declared here and re-exported at the crate root (see `lib.rs`),
//! so call sites keep reading `crate::array::…`, `crate::kernels::…` as before.

pub mod array;
pub mod ffi;
pub mod mlx_rt;
pub mod models;
pub mod runtime;
pub mod shim_api;
