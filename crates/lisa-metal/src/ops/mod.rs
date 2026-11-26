//! Native Metal operations used by model execution.
//!
//! The API retains MLX-style array operations, but allocation, dispatch and
//! synchronization are implemented by LISA. This module has no dependency on
//! the MLX checkpoint reader or an external MLX tensor runtime. Arithmetic
//! contracts and attributed kernel sources remain part of each operation.

pub type Result<T> = crate::error::Result<T>;

/// `lisa_metal::error`-shaped module.
pub mod error {
    pub use crate::error::{Exception, Result};
}

/// mlx-like dtype enum — now the native one (`array::Dtype`), re-exported so
/// `lisa_metal::Dtype` keeps its meaning.
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
mod topk;
pub mod transforms;
pub use topk::top32_indices;
pub(crate) use topk::top32_indices_on;

pub use array::Array;
pub use stream::{
    Stream, enable_allocation_labels, label_counts, runtime_alloc_count, runtime_dispatch_count,
    runtime_encoder_count, runtime_label_counts, runtime_label_dispatches, runtime_label_gpu_ms,
    runtime_poolhit_count, runtime_wait_ns, runtime_zero_ns,
};
pub use traits::{ScalarVal, SliceElem, ZeroElem};
pub(crate) use traits::{
    index_to_u32, norm_axis, promote_dtype, promoted_binary, scalar_op, scalar_op_rev,
};
