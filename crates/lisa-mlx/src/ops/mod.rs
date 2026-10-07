//! Native implementation of the `lisa_mlx` surface used by the tree.
//!
//! This is built and A/B-validated family-by-family; `lib.rs` keeps
//! re-exporting the real bindings until the whole surface here is complete, at
//! which point this module is promoted to the crate root and the real
//! dependency is removed.
//!
//! Semantics notes:
//! - `eval` maps to a device synchronize: there is no "dispatch without
//!   wait", so unlike MLX this serializes the queue. Revisit before shipping.
//! - Fused/rounding-sensitive ops (`silu`, `sigmoid`, `logaddexp`, `log1p`)
//!   are *not* bit-exact via op chains (see `ab_probe`); they will be
//!   routed through `mlx_rt` from MLX's unary kernel instead.

pub type Result<T> = crate::error::Result<T>;

/// `lisa_mlx::error`-shaped module.
pub mod error {
    pub use crate::error::{Exception, Result};
}

/// mlx-like dtype enum — now the native one (`array::Dtype`), re-exported so
/// `lisa_mlx::Dtype` keeps its meaning.
pub use crate::array::Dtype;

mod array;
mod stream;
mod traits;

pub mod array_ops;
pub mod fast;
pub mod indexing;
pub mod memory;
pub mod nn;
pub use array_ops::*;
pub mod transforms;

pub use array::Array;
pub use stream::{
    Stream, label_counts, runtime_alloc_count, runtime_dispatch_count, runtime_encoder_count,
    runtime_label_counts, runtime_label_dispatches, runtime_label_gpu_ms, runtime_poolhit_count, runtime_wait_ns,
    runtime_zero_ns,
};
pub use traits::{ScalarVal, SliceElem, ZeroElem};
pub(crate) use traits::{
    index_to_u32, norm_axis, promote_dtype, promoted_binary, scalar_op, scalar_op_rev,
};
