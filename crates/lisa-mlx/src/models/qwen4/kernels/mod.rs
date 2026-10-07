//! Custom Metal kernels.
//!
//! Numerics are preserved exactly: same source strings, same template
//! parameters, same grid/threadgroup shapes.

mod attn;
mod common;
// GDN is a shared architecture component: the host wrappers live at the
// crate-level `models::gdn`, re-imported here so the `kernels::gdn_*` pub
// use chain is unchanged.
use crate::models::gdn;
mod moe;
mod ple;

/// Force `get_or_init` on every kernel static in every submodule (see
/// [`crate::models::qwen4::warm_kernels`]).
pub fn warm(_stream: &lisa_mlx::Stream) {
    attn::warm();
    common::warm();
    gdn::warm();
    moe::warm();
    ple::warm();
}

pub use attn::{attn_gate, attn_prep_split, gated_rms, gated_rms_silu, hc_mix, inject_norm, silu_head, swiglu2};
pub use common::{EXACT_HEADER, copy_rows_into, indirect_bench};
pub use gdn::{
    GatedDeltaKernels, gated_delta_kernel, gated_delta_ops, gated_delta_ops_capture,
    gdn_decode_complete, gdn_kernel_available, gdn_prep, gdn_prep_fused, gdn_rows, gdn_two_row,
    gdn_two_row_into,
};
pub use moe::{moe_combine, route_counting_sort, router_gemv};
pub use ple::{ple_conv, ple_fuse2, ple_gated, ple_prod};
