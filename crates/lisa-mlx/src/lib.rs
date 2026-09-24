//! Native tensor runtime for the `lisa` engine: an eager `Array` over a device
//! backend, plus the kernels the model dispatches.
//!
//! Layout:
//!   - [`backend`] selects the device backend (Metal today; CPU is a sibling
//!     module and a [`backend::Backend`] variant). [`backend::metal`] holds the
//!     Metal runtime, the `Array`/op surface and the Metal kernels.
//!   - The Metal modules are re-exported here, so call sites keep reading
//!     `lisa_mlx::ops`, `lisa_mlx::kernels::…`, `lisa_mlx::qsa::…`.
//!   - [`error`] is the shared error type.
//!
//! No MLX, no external tensor framework: the runtime is our own objc2 Metal.

pub mod backend;
pub mod error;

// The Metal backend's modules, re-exported at the crate root so existing
// `crate::<module>` and `lisa_mlx::<module>` paths are unchanged.
pub use backend::metal::{array, ffi, mlx_rt, runtime, shim_api};
pub use backend::metal::models::qwen4::{
    kernels, moe_decode, moe_helpers, nax_helpers, prefill_indirect, qsa,
};

pub use shim_api::ops;
pub use shim_api::{fast, memory, nn, transforms};
pub use shim_api::{
    runtime_alloc_count, runtime_dispatch_count, runtime_encoder_count, runtime_label_counts,
    runtime_poolhit_count, runtime_wait_ns, runtime_zero_ns, Array, Dtype, Stream,
};

// Let the moved engine-kernel modules keep their `use lisa_mlx::…` imports:
// inside the crate, `lisa_mlx` is an alias for this crate itself.
extern crate self as lisa_mlx;

pub use runtime::env_flag;