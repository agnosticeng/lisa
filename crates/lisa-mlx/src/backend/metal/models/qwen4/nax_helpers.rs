//! MLX `steel` GEMM infrastructure (verbatim) from the engine's
//! `TrackPrefillIndirectMetal.swift`. The NAX prefill kernels build on it.
//! Extracted mechanically; do not hand-edit.
pub const NAX_HEADER: &str = include_str!("../../kernels/nax/prefill_indirect.metal");
