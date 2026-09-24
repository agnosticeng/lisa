//! Verbatim Metal helper library from the engine (`TrackFastMoE.swift` and
//! `TrackFastKernels2.swift`): MLX's own quantized-GEMV building blocks.
//! Extracted mechanically from the reference; do not hand-edit.
pub const HELPERS_CORE: &str = include_str!("../../kernels/common/quantized_helpers.metal");

pub const HELPERS_MLX: &str = include_str!("../../kernels/common/mlx_helpers.metal");

pub const REG_HELPERS: &str = include_str!("kernels/reg_helpers.metal");

pub const GATE_UP_REUSE_HELPERS: &str = include_str!("kernels/gate_up_reuse_helpers.metal");

pub const WIDE_HELPERS: &str = include_str!("kernels/wide_helpers.metal");

pub const WIDE_DECLS: &str = include_str!("kernels/wide_decls.metal");

pub const PIPELINED_HELPERS: &str = include_str!("kernels/pipelined_helpers.metal");

pub const MIXER_HEAD_TAIL: &str = include_str!("kernels/mixer_head_tail.metal");

