//! Native tensor runtime for the `lisa` engine: an eager `Array` over a device
//! backend, plus the kernels the model dispatches.
//!
//! Layout:
//!   - [`device`] selects the device backend (Metal today; CPU is a sibling
//!     module and a [`device::Backend`] variant).
//!   - The Metal backend: objc2 Metal runtime, the eager `Array`/op surface,
//!     the generic Metal kernels, and the model-specific kernel wrappers.
//!   - The Metal modules are re-exported here, so call sites keep reading
//!     `lisa_mlx::ops`, `lisa_mlx::kernels::…`, `lisa_mlx::qsa::…`.
//!   - [`error`] is the shared error type.
//!
//! No MLX, no external tensor framework: the runtime is our own objc2 Metal.

pub mod array;
pub mod device;
pub mod error;
pub mod ffi;
pub mod jit;
pub mod models;
pub mod ops;
pub mod runtime;
pub mod trace;

// Compat: the device module used to live at `backend`; keep
// `lisa_mlx::backend::Backend` working for external callers.
pub mod backend {
    pub use crate::device::*;
}

pub use jit as mlx_rt;
pub use models::qwen4::{
    kernels, moe_decode, moe_helpers, nax_helpers, prefill_indirect, qsa, warm_kernels,
};

pub use ops::{
    Array, Dtype, Stream, runtime_alloc_count, runtime_dispatch_count, runtime_encoder_count,
    runtime_label_counts, runtime_label_dispatches, runtime_label_gpu_ms, runtime_poolhit_count, runtime_wait_ns,
    runtime_zero_ns,
};
pub use ops::{fast, memory, nn, transforms};

// Let the moved engine-kernel modules keep their `use lisa_mlx::…` imports:
// inside the crate, `lisa_mlx` is an alias for this crate itself.
extern crate self as lisa_mlx;
