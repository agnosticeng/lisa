//! Custom Metal kernels.
//!
//! Numerics are preserved exactly: same source strings, same template
//! parameters, same grid/threadgroup shapes.

pub mod attention;
mod common;
// GDN is a shared architecture component: the host wrappers live at the
// crate-level `kernels::gated_delta`, re-imported here so the `kernels::gdn_*` pub
// use chain is unchanged.
pub mod gated_delta;
pub mod product_lookup;
pub mod routing;

/// Force `get_or_init` on every kernel static in every submodule (see
/// [`crate::kernels::warm_kernels`]).
pub fn warm(_stream: &lisa_metal::Stream) {
    attention::warm();
    common::warm();
    gated_delta::warm();
    routing::warm();
    product_lookup::warm();
}

pub use attention::{
    attn_gate, attn_prep_split, gated_rms, gated_rms_silu, gated_rms_silu_with_serial_gate, hc_mix,
    inject_norm, silu_head, swiglu2,
};
pub use common::{EXACT_HEADER, copy_rows_concat_into, copy_rows_into, indirect_bench};
pub use gated_delta::{
    GatedDeltaKernels, gated_delta_kernel, gdn_decode_complete, gdn_kernel_available, gdn_prep,
    gdn_prep_fused, gdn_rows, gdn_two_row, gdn_two_row_into,
};
pub use product_lookup::{ple_conv, ple_fuse2, ple_gated, ple_prod};
pub use routing::{moe_combine, route_counting_sort, router_gemv};

pub mod affine_moe;
pub mod block_attention;
pub mod blockwise;
pub mod gated_pointwise;
pub mod ggml;
pub mod indirect_moe;
pub mod moe_helpers;
pub mod nax_helpers;
pub mod sparse;

/// Initialize operation handles without selecting a model or traversing layers.
/// Shape-specialized pipelines remain lazy in their existing cache.
pub fn warm_kernels() {
    let stream = crate::Stream::gpu();
    warm(&stream);
    affine_moe::warm(&stream);
    indirect_moe::warm();
    block_attention::warm();
}

pub mod recurrent_mimo;
