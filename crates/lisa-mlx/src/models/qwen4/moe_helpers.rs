//! Verbatim Metal helper library (`TrackFastMoE.swift` and
//! `TrackFastKernels2.swift`): MLX's own quantized-GEMV building blocks.
//! Extracted mechanically; do not hand-edit.
pub const HELPERS_CORE: &crate::shader_pack::Source =
    crate::kernel_source!("common/matmul/quantized_helpers.metal");

pub const HELPERS_MLX: &crate::shader_pack::Source =
    crate::kernel_source!("common/matmul/mlx_helpers.metal");

pub const REG_HELPERS: &crate::shader_pack::Source =
    crate::kernel_source!("common/matmul/reg_helpers.metal");

pub const GATE_UP_REUSE_HELPERS: &crate::shader_pack::Source =
    crate::kernel_source!("qwen4/gate_up_reuse_helpers.metal");

pub const WIDE_HELPERS: &crate::shader_pack::Source =
    crate::kernel_source!("common/matmul/wide_helpers.metal");

pub const WIDE_DECLS: &crate::shader_pack::Source =
    crate::kernel_source!("common/matmul/wide_decls.metal");

pub const PIPELINED_HELPERS: &crate::shader_pack::Source =
    crate::kernel_source!("common/matmul/pipelined_helpers.metal");

pub const MIXER_HEAD_TAIL: &crate::shader_pack::Source =
    crate::kernel_source!("qwen4/mixer_head_tail.metal");
