//! Generic kernel-source assembly and JIT pipelines.
//!
//! The engine's own kernels are single static `.metal` sources. The *generic*
//! op set (elementwise, reductions, softmax, sort, rope, sdpa, dequantize,
//! gemm/gemv) is instead built here, because Metal has no runtime type
//! dispatch: each element type / configuration is a separate C++ template
//! instantiation, selected by name.
//!
//! Each generic kernel is therefore assembled from three static pieces:
//!   1. a **preamble** (helpers and shared types, `include_str!`);
//!   2. the **kernel body** (`include_str!`);
//!   3. an **explicit-instantiation line** per variant
//!      (`[[host_name("name")]] … decltype(fn<args>) fn<args>;`), built by
//!      [`builtin_template_def`].
//! The assembled source is JIT-compiled with the Safe/precise compile options
//! and cached per generated kernel name. This mirrors how the GPU code is meant
//! to be specialized; the bodies stay static and the host only picks a variant.
//!
//! The signature generator is a faithful reproduction; see `NOTICE` for the
//! third-party attribution.

/// The generic kernels were written against a `Tensor`/`DType`/`Device` naming;
/// these aliases keep the bodies readable while everything behind them is the
/// native `Array`/`Dtype`/`MetalRuntime`.
pub type Tensor = Array;
pub type DType = Dtype;
pub type Device = std::sync::Arc<MetalRuntime>;

/// MLX `metal::utils()` — the preprocessed Metal preamble (`metal_stdlib`,
/// `using namespace metal`, `bfloat16_t`, bf16 math overloads). MLX prepends
/// this to every custom kernel source (`CustomKernel::eval_gpu`), so we do too.
pub(crate) const MLX_UTILS_PREAMBLE: &str = include_str!("../shaders/common/utils.metal");
// Flattened MLX Metal NAX header closures (auto-generated; quoted #include
// lines inlined in dependency order). These replace the former `mlx_include`
// tree: the AOT compiles prepend the right closure instead of `-I`.
const NAX_GEMM_HEADER: &str = include_str!("../shaders/nax/gemm_header.metal");
const NAX_QUANT_HEADER: &str = include_str!("../shaders/nax/quant_header.metal");
const NAX_ATTN_HEADER: &str = include_str!("../shaders/nax/attn_header.metal");

/// MLX `metal::gemm()` — preprocessed `kernels/steel/gemm/gemm.h`.
pub(crate) const MLX_GEMM_PREAMBLE: &str = include_str!("../shaders/common/matmul/gemm.metal");
/// MLX `metal::quantized_utils()` — preprocessed `kernels/quantized_utils.h`.
pub(crate) const MLX_QUANTIZED_UTILS_PREAMBLE: &str =
    include_str!("../shaders/common/matmul/quantized_utils.metal");
/// MLX `metal::quantized()` — preprocessed `kernels/quantized.h` (affine kernels).
pub(crate) const MLX_QUANTIZED_PREAMBLE: &str = include_str!("../shaders/common/matmul/quantized.metal");
/// MLX `metal::unary_ops()` — preprocessed `kernels/unary_ops.h`.
pub(crate) const MLX_UNARY_OPS_PREAMBLE: &str =
    include_str!("../shaders/common/elementwise/unary_ops.metal");
/// MLX `metal::unary()` — preprocessed `kernels/unary.h`.
pub(crate) const MLX_UNARY_PREAMBLE: &str =
    include_str!("../shaders/common/elementwise/unary.metal");
/// MLX `metal::binary_ops()` — preprocessed `kernels/binary_ops.h`.
const MLX_BINARY_OPS_PREAMBLE: &str =
    include_str!("../shaders/common/elementwise/binary_ops.metal");
/// MLX `metal::binary()` — preprocessed `kernels/binary.h`.
const MLX_BINARY_PREAMBLE: &str = include_str!("../shaders/common/elementwise/binary.metal");
/// MLX `metal::softmax()` — preprocessed `kernels/softmax.h`.
pub(crate) const MLX_SOFTMAX_PREAMBLE: &str =
    include_str!("../shaders/common/reduce/softmax.metal");
/// MLX `metal::reduce_utils()` — preprocessed `kernels/reduce_utils.h`.
pub(crate) const MLX_REDUCE_UTILS_PREAMBLE: &str =
    include_str!("../shaders/common/reduce/reduce_utils.metal");
/// MLX `metal::reduce()` — preprocessed `kernels/reduce.h`.
pub(crate) const MLX_REDUCE_PREAMBLE: &str = include_str!("../shaders/common/reduce/reduce.metal");
/// MLX `metal::sort()` — preprocessed `kernels/sort.h`.
const MLX_SORT_PREAMBLE: &str = include_str!("../shaders/common/reduce/sort.metal");
/// MLX `kernels/rms_norm.metal` (AOT source; project include stripped). Uses the
/// `has_w` function constant (index 20).
const MLX_RMS_NORM_SOURCE: &str = include_str!("../shaders/common/attention/rms_norm.metal");
/// Fused residual-add + RMSNorm (specs/08 §1).
const MLX_FUSED_ADD_RMS_SOURCE: &str =
    include_str!("../shaders/common/attention/fused_add_rms.metal");
/// Norm-in-qmv mega-kernel (specs/16 phase 1): fused add+rms prologue inside
/// `affine_qmv_fast`. Must come AFTER `MLX_QUANTIZED_PREAMBLE` (uses qdot).
pub(crate) const MLX_NORM_QMV_SOURCE: &str = include_str!("../shaders/common/matmul/norm_qmv.metal");
/// MLX `kernels/rope.metal` (AOT source; project include stripped). Function
/// constants: 1 = forward, 2 = traditional, 3 = head_seq_transpose.
const MLX_ROPE_SOURCE: &str = include_str!("../shaders/common/attention/rope.metal");
/// MLX `kernels/sdpa_vector.h` (AOT source). Function constants 20..25:
/// has_mask, query_transposed, do_causal, bool_mask, float_mask, has_sinks.
const MLX_SDPA_VECTOR_PREAMBLE: &str =
    include_str!("../shaders/common/attention/sdpa_vector.metal");
/// MLX `metal::scatter_axis()` — preprocessed `kernels/indexing/scatter_axis.h`.
const MLX_SCATTER_AXIS_PREAMBLE: &str = include_str!("../shaders/common/data/scatter_axis.metal");
/// MLX `arg_reduce.metal` (AOT source; project include stripped — the utils
/// preamble provides it).
pub(crate) const MLX_ARG_REDUCE_SOURCE: &str =
    include_str!("../shaders/common/reduce/arg_reduce.metal");

use crate::array::{Array, Dtype};
use crate::runtime::MetalRuntime;

mod attention;
mod buf;
#[cfg(test)]
mod mpp_probe;
mod compile;
mod elementwise;
mod kernel;
mod matmul;
mod nax;
mod norm;
mod reduce;
mod rope;
mod sort;

pub use attention::*;
pub use buf::*;
pub(crate) use compile::builtin_template_def;
pub use compile::*;
pub use elementwise::*;
pub use kernel::*;
pub use matmul::*;
pub use nax::*;
pub use norm::*;
pub use reduce::*;
pub use rope::*;
pub use sort::*;
