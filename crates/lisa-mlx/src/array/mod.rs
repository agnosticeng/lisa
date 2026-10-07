//! A native, eager `Array` over `MTLBuffer` with a
//! `Layout` (shape / element strides / element offset) and metadata-only views.
//!
//! Additive — the shim's `ops::Array` is still the one the engine uses.
//! Stage 3 migrates the `ops` families onto this type behind that same shim API,
//! each family is validated against the 256/256 golden.
//!
//! Eager here means an op dispatches its kernel into the runtime's batched
//! command buffer as soon as it is called (no lazy graph). The GPU still
//! pipelines while lisa builds a whole chunk, because the runtime only commits
//! every `per_buffer` dispatches (see `runtime::Commands`).

mod copy;
mod core;
mod dtype;
mod facade;
mod indexing;
mod layout;
mod ops;

pub use copy::MAX_COPY_RANK;
pub use core::Array;
pub use dtype::{Dtype, dtype_of};
pub use indexing::IdxElem;
pub use layout::Layout;

pub(super) fn err(m: impl Into<String>) -> Error {
    Error::Msg(m.into())
}

use crate::error::Error;

// ─────────────────────────────── tests ───────────────────────────────
