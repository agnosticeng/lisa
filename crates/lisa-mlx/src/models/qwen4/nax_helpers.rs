//! MLX `steel` GEMM infrastructure (verbatim) from the engine's
//! `TrackPrefillIndirectMetal.swift`. The NAX prefill kernels build on it.
//! Extracted mechanically; do not hand-edit.
pub const NAX_HEADER: &crate::shader_pack::Source =
    crate::kernel_source!("nax/prefill_indirect.metal");
