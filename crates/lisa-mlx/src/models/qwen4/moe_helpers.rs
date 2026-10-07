//! Verbatim Metal helper library (`TrackFastMoE.swift` and
//! `TrackFastKernels2.swift`): MLX's own quantized-GEMV building blocks.
//! Extracted mechanically; do not hand-edit.
pub const HELPERS_CORE: &str = include_str!("../../shaders/common/matmul/quantized_helpers.metal");

pub const HELPERS_MLX: &str = include_str!("../../shaders/common/matmul/mlx_helpers.metal");

pub const REG_HELPERS: &str = include_str!("../../shaders/common/matmul/reg_helpers.metal");

pub const GATE_UP_REUSE_HELPERS: &str =
    include_str!("../../shaders/qwen4/gate_up_reuse_helpers.metal");

pub const WIDE_HELPERS: &str = include_str!("../../shaders/common/matmul/wide_helpers.metal");

pub const WIDE_DECLS: &str = include_str!("../../shaders/common/matmul/wide_decls.metal");

pub const PIPELINED_HELPERS: &str =
    include_str!("../../shaders/common/matmul/pipelined_helpers.metal");

pub const MIXER_HEAD_TAIL: &str = include_str!("../../shaders/qwen4/mixer_head_tail.metal");
