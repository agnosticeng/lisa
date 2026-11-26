//! Native tensor runtime for the `lisa` engine: an eager `Array` over a device
//! backend, plus the kernels the model dispatches.
//!
//! Layout:
//!   - [`device`] reports Metal availability. Backend selection is owned by
//!     the engine/application; this crate only executes Metal operations.
//!   - The Metal backend: objc2 Metal runtime, the eager `Array`/op surface,
//!     the generic Metal kernels, and the operation-specific kernel wrappers.
//!   - The Metal modules are re-exported here, so call sites keep reading
//!     `lisa_metal::ops`, `lisa_metal::kernels::…`, `lisa_metal::qsa::…`.
//!   - [`error`] is the shared error type.
//!
//! No MLX, no external tensor framework: the runtime is our own objc2 Metal.

pub mod array;
pub mod device;
pub mod diagnostics;
pub mod error;
pub mod ffi;
pub mod jit;
pub mod kernels;
pub mod ops;
pub mod runtime;
#[doc(hidden)]
pub mod shader_pack;
pub mod trace;

include!(concat!(env!("OUT_DIR"), "/kernel_index.rs"));

pub use jit as mlx_rt;
pub use kernels::{
    affine_moe as moe_decode, block_attention as qsa, indirect_moe as prefill_indirect,
    moe_helpers, nax_helpers, warm_kernels,
};

pub use ops::{
    Array, Dtype, Stream, enable_allocation_labels, runtime_alloc_count, runtime_dispatch_count,
    runtime_encoder_count, runtime_label_counts, runtime_label_dispatches, runtime_label_gpu_ms,
    runtime_poolhit_count, runtime_wait_ns, runtime_zero_ns,
};
pub use ops::{fast, memory, nn, transforms};

// Let the moved engine-kernel modules keep their `use lisa_metal::…` imports:
// inside the crate, `lisa_metal` is an alias for this crate itself.
extern crate self as lisa_metal;

mod access;
