//! GLM-5.3 KDA Metal kernels (phase 3 port, ds4 `glm53_kda.metal` reference).
//!
//! The KDA prepare, recurrence, and output kernels port ds4's F32 execution
//! order. MLX-loaded weights and projections feed the same operation graph.

use crate::ffi::{KernelInput, MetalKernel, OutputArg, TemplateArg};
use crate::kernels::EXACT_HEADER;
use crate::{Array, Dtype, Stream};
use objc2_metal::MTLDevice;

thread_local! {
    static GLM53_DSA_NATIVE_OVERRIDE: std::cell::Cell<bool> = const { std::cell::Cell::new(true) };
    static GLM53_DSA_VALUE_DENSE_PREFIX: std::cell::Cell<Option<usize>> = const { std::cell::Cell::new(None) };
}

/// DS4 ends an attention/value slice at the dense-to-sparse cache boundary.
/// Preserve the resulting value-projection row count when selecting its
/// scalar (<32 rows) or matrix schedule.
pub struct Glm53DsaValueSliceScope {
    previous: Option<usize>,
}

impl Glm53DsaValueSliceScope {
    pub fn new(query_start: usize, dense_limit: usize) -> Self {
        let prefix = dense_limit.saturating_sub(query_start);
        let previous = GLM53_DSA_VALUE_DENSE_PREFIX.with(|value| value.replace(Some(prefix)));
        Self { previous }
    }
}

impl Drop for Glm53DsaValueSliceScope {
    fn drop(&mut self) {
        GLM53_DSA_VALUE_DENSE_PREFIX.with(|value| value.set(self.previous));
    }
}

/// Scoped override for DS4's native QK, value and indexed-decode schedules.
/// Production uses the native schedules; diagnostics may select the legacy path.
pub struct Glm53DsaNativeScope {
    previous: bool,
}

impl Glm53DsaNativeScope {
    pub fn new(enabled: bool) -> Self {
        let previous = GLM53_DSA_NATIVE_OVERRIDE.with(|value| value.replace(enabled));
        Self { previous }
    }
}

impl Drop for Glm53DsaNativeScope {
    fn drop(&mut self) {
        GLM53_DSA_NATIVE_OVERRIDE.with(|value| value.set(self.previous));
    }
}

/// Arithmetic mode used by GLM's explicit prefix/session cache identity.
pub fn glm53_dsa_native_enabled() -> bool {
    GLM53_DSA_NATIVE_OVERRIDE.with(std::cell::Cell::get)
}

const KDA_RECURRENCE_SOURCE: &str = include_str!("../../shaders/glm53/kda_recurrence.metal");
const KDA_PREPARE_SOURCE: &str = include_str!("../../shaders/glm53/kda_prepare.metal");
const KDA_OUTPUT_SOURCE: &str = include_str!("../../shaders/glm53/kda_output.metal");
const KDA_DECODE_SOURCE: &str = include_str!("../../shaders/glm53/kda_decode.metal");
const BF16_GEMV_SOURCE: &str = include_str!("../../shaders/glm53/bf16_gemv.metal");
const BF16_GEMV_ROWS2_SOURCE: &str = include_str!("../../shaders/glm53/bf16_gemv_rows2.metal");
const BF16_PAIR_GEMV_SOURCE: &str = include_str!("../../shaders/glm53/bf16_pair_gemv.metal");
const BF16_QKV_SOURCE: &str = include_str!("../../shaders/glm53/bf16_qkv.metal");
const BF16_GEMM_SOURCE: &str = include_str!("../../shaders/glm53/bf16_gemm.metal");
const BF16_KDA_PROJECT_SOURCE: &str = include_str!("../../shaders/glm53/bf16_kda_project.metal");
const AFFINE_Q8_MV_DS4_SOURCE: &str = include_str!("../../shaders/glm53/affine_q8_mv_ds4.metal");
const INDEXER_TOPK_SORT_SOURCE: &str = include_str!("../../shaders/glm53/indexer_topk_sort.metal");
const INDEXER_TOPK_MERGE_SOURCE: &str =
    include_str!("../../shaders/glm53/indexer_topk_merge.metal");
const INDEXER_EXPAND_SELECTION_SOURCE: &str =
    include_str!("../../shaders/glm53/indexer_expand_selection.metal");
const DSA_INDEXED_LORA_SOURCE: &str = include_str!("../../shaders/glm53/dsa_indexed_lora.metal");
const DSA_INDEXED_LORA_NATIVE_SOURCE: &str =
    include_str!("../../shaders/glm53/dsa_indexed_lora_native.metal");
const DSA_INDEXED_DECODE_FUSED_SOURCE: &str =
    include_str!("../../shaders/glm53/dsa_indexed_decode_fused.metal");
const DSA_INDEXED_DECODE_NATIVE_SOURCE: &str =
    include_str!("../../shaders/glm53/dsa_indexed_decode_native.metal");
const DSA_VALUE_PROJECT_SOURCE: &str = include_str!("../../shaders/glm53/dsa_value_project.metal");
const DSA_QK_LOWRANK_NATIVE_SOURCE: &str =
    include_str!("../../shaders/glm53/dsa_qk_lowrank_native.metal");
const DSA_VALUE_PROJECT_NATIVE_SOURCE: &str =
    include_str!("../../shaders/glm53/dsa_value_project_native.metal");
const DSA_VALUE_PROJECT_NATIVE_MMA_SOURCE: &str =
    include_str!("../../shaders/glm53/dsa_value_project_native_mma.metal");
const DSA_O_PROJECT_SOURCE: &str = include_str!("../../shaders/glm53/dsa_o_project.metal");

fn indexer_topk_sort_kernel() -> &'static Option<MetalKernel> {
    static K: std::sync::OnceLock<Option<MetalKernel>> = std::sync::OnceLock::new();
    K.get_or_init(|| {
        MetalKernel::new_with_math(
            "glm53_indexer_topk_sort_ds4_default_v3",
            &["n_comp", "n_rows", "work_width", "block_top_k", "scores"],
            &["partial"],
            INDEXER_TOPK_SORT_SOURCE,
            EXACT_HEADER,
            true,
            false,
            crate::runtime::Math::CompilerDefault,
        )
        .ok()
    })
}

fn indexer_topk_merge_kernel() -> &'static Option<MetalKernel> {
    static K: std::sync::OnceLock<Option<MetalKernel>> = std::sync::OnceLock::new();
    K.get_or_init(|| {
        MetalKernel::new_with_math(
            "glm53_indexer_topk_merge_ds4_default_v2",
            &[
                "n_comp",
                "n_rows",
                "work_width",
                "top_k",
                "run_len",
                "out_width",
                "scores",
                "partial",
            ],
            &["partial_out"],
            INDEXER_TOPK_MERGE_SOURCE,
            EXACT_HEADER,
            true,
            false,
            crate::runtime::Math::CompilerDefault,
        )
        .ok()
    })
}

/// Select rowwise descending top-k pool ids with ds4's block-sort plus merge
/// algorithm. `scores` is contiguous F32 `[rows, pools]`; output is I32
/// `[rows, top_k]`. This avoids globally sorting the full sparse-attention
/// history for every query row.
pub fn dsa_indexer_topk(scores: &Array, top_k: usize, stream: &Stream) -> Option<Array> {
    dsa_indexer_topk_with_shuffle(scores, top_k, false, stream)
}

/// DS4's blockwise indexer top-k with SIMD shuffle stages for the sub-32
/// bitonic network. The comparator and tie order are the same as the shared
/// threadgroup-memory implementation.
pub fn dsa_indexer_topk_shuffle(scores: &Array, top_k: usize, stream: &Stream) -> Option<Array> {
    dsa_indexer_topk_with_shuffle(scores, top_k, true, stream)
}

fn dsa_indexer_topk_with_shuffle(
    scores: &Array,
    top_k: usize,
    use_shuffle: bool,
    stream: &Stream,
) -> Option<Array> {
    if scores.rank() != 2 || scores.dtype() != Dtype::Float32 {
        return None;
    }
    let n_rows = scores.dim(0) as usize;
    let n_comp = scores.dim(1) as usize;
    if n_rows == 0 || n_comp == 0 || top_k == 0 || top_k > n_comp {
        return None;
    }

    // The reference DS4 Metal path uses a 1024-entry block on Apple Silicon.
    // Round up to the next power of two for shorter score rows.
    let mut nth = 1usize;
    while nth < n_comp && nth < 1024 {
        nth <<= 1;
    }
    let block_top_k = top_k.min(nth);
    let n_blocks = n_comp.div_ceil(nth);
    let work_width = if n_blocks <= 1 {
        top_k
    } else {
        let last_width = n_comp - (n_blocks - 1) * nth;
        (n_blocks - 1) * block_top_k + last_width.min(block_top_k)
    };
    let (Ok(scores), Some(sort_kernel)) =
        (scores.contiguous(), indexer_topk_sort_kernel().as_ref())
    else {
        return None;
    };
    scores.eval_enqueue().ok()?;

    let sort_inputs = [
        KernelInput::I32(n_comp as i32),
        KernelInput::I32(n_rows as i32),
        KernelInput::I32(work_width as i32),
        KernelInput::I32(block_top_k as i32),
        KernelInput::Tensor(&scores.t),
    ];
    let sort_out = sort_kernel
        .apply_with_inline_inputs(
            &sort_inputs,
            &[
                TemplateArg::Int("NTH", nth as i32),
                TemplateArg::Int("SHUFFLE", use_shuffle as i32),
            ],
            ((n_blocks * n_rows * nth) as i32, 1, 1),
            (nth as i32, 1, 1),
            &[OutputArg {
                shape: vec![n_rows as i32, work_width as i32],
                dtype: Dtype::Int32,
            }],
            stream,
        )
        .ok()?;
    let mut partial = sort_out.into_iter().next()?;

    let merge_kernel = indexer_topk_merge_kernel().as_ref()?;
    let mut run_len = block_top_k;
    while run_len < work_width {
        let n_merges = work_width.div_ceil(2 * run_len);
        let final_merge = n_merges == 1;
        let out_width = if final_merge { top_k } else { work_width };
        let merge_threads = run_len.min(512).max(1);
        let merge_inputs = [
            KernelInput::I32(n_comp as i32),
            KernelInput::I32(n_rows as i32),
            KernelInput::I32(work_width as i32),
            KernelInput::I32(out_width as i32),
            KernelInput::I32(run_len as i32),
            KernelInput::I32(out_width as i32),
            KernelInput::Tensor(&scores.t),
            KernelInput::Tensor(&partial.t),
        ];
        let merged = merge_kernel
            .apply_with_inline_inputs(
                &merge_inputs,
                &[],
                ((n_merges * n_rows * merge_threads) as i32, 1, 1),
                (merge_threads as i32, 1, 1),
                &[OutputArg {
                    shape: vec![n_rows as i32, out_width as i32],
                    dtype: Dtype::Int32,
                }],
                stream,
            )
            .ok()?;
        partial = merged.into_iter().next()?;
        run_len = run_len.saturating_mul(2);
    }

    Some(partial)
}

fn indexer_expand_selection_kernel() -> &'static Option<MetalKernel> {
    static K: std::sync::OnceLock<Option<MetalKernel>> = std::sync::OnceLock::new();
    K.get_or_init(|| {
        MetalKernel::new_with_math(
            "glm53_indexer_expand_selection_ds4_default_v3",
            &[
                "n_rows",
                "n_tokens",
                "pos0",
                "selected_pools",
                "index_topk",
                "pool_size",
                "output_width",
                "always_tail",
                "causal_prefix",
                "pool_selected",
            ],
            &["selected"],
            INDEXER_EXPAND_SELECTION_SOURCE,
            EXACT_HEADER,
            true,
            false,
            crate::runtime::Math::CompilerDefault,
        )
        .ok()
    })
}

/// Expand selected DSA pool ids into token ids and append the causal tail.
/// `pool_selected` is `[batch * tokens, selected_pools]`; the returned
/// contiguous tensor is `[batch * tokens, output_width]`.
pub fn glm53_expand_pool_selection(
    pool_selected: &Array,
    n_tokens: usize,
    pos0: usize,
    pool_size: usize,
    always_tail: bool,
    stream: &Stream,
) -> Option<Array> {
    glm53_expand_pool_selection_mode(
        pool_selected,
        n_tokens,
        pos0,
        pool_size,
        always_tail,
        false,
        stream,
    )
}

/// Expand pool IDs for absorbed attention. Queries in DS4's dense-causal
/// prefix use chronological token IDs so their lora reduction visits cache
/// rows in the same order as DS4's causal kernel.
pub fn glm53_expand_pool_selection_for_attention(
    pool_selected: &Array,
    n_tokens: usize,
    pos0: usize,
    pool_size: usize,
    always_tail: bool,
    stream: &Stream,
) -> Option<Array> {
    glm53_expand_pool_selection_mode(
        pool_selected,
        n_tokens,
        pos0,
        pool_size,
        always_tail,
        true,
        stream,
    )
}

fn glm53_expand_pool_selection_mode(
    pool_selected: &Array,
    n_tokens: usize,
    pos0: usize,
    pool_size: usize,
    always_tail: bool,
    causal_prefix: bool,
    stream: &Stream,
) -> Option<Array> {
    if pool_selected.rank() != 2
        || pool_selected.dtype() != Dtype::Int32
        || n_tokens == 0
        || pool_size == 0
    {
        return None;
    }
    let (n_rows, selected_pools) = (pool_selected.dim(0) as usize, pool_selected.dim(1) as usize);
    if n_rows == 0
        || selected_pools == 0
        || n_rows % n_tokens != 0
        || pos0 > i32::MAX as usize
        || selected_pools > i32::MAX as usize
        || pool_size > i32::MAX as usize
    {
        return None;
    }
    let index_topk = selected_pools.checked_mul(pool_size)?;
    let output_width = index_topk.checked_add(pool_size.saturating_sub(1))?;
    let total = n_rows.checked_mul(output_width)?;
    if output_width == 0
        || output_width > i32::MAX as usize
        || n_rows > i32::MAX as usize
        || total > i32::MAX as usize
    {
        return None;
    }
    let kernel = indexer_expand_selection_kernel().as_ref()?;
    let pool_selected = pool_selected.contiguous().ok()?;
    pool_selected.eval_enqueue().ok()?;
    let inputs = [
        KernelInput::I32(n_rows as i32),
        KernelInput::I32(n_tokens as i32),
        KernelInput::I32(pos0 as i32),
        KernelInput::I32(selected_pools as i32),
        KernelInput::I32(index_topk as i32),
        KernelInput::I32(pool_size as i32),
        KernelInput::I32(output_width as i32),
        KernelInput::I32(i32::from(always_tail)),
        KernelInput::I32(i32::from(causal_prefix)),
        KernelInput::Tensor(&pool_selected.t),
    ];
    let out = kernel
        .apply_with_inline_inputs(
            &inputs,
            &[],
            (total as i32, 1, 1),
            (256, 1, 1),
            &[OutputArg {
                shape: vec![n_rows as i32, output_width as i32],
                dtype: Dtype::Int32,
            }],
            stream,
        )
        .ok()?
        .into_iter()
        .next()?;
    Some(out)
}

fn bf16_gemv_kernel() -> &'static Option<MetalKernel> {
    static K: std::sync::OnceLock<Option<MetalKernel>> = std::sync::OnceLock::new();
    K.get_or_init(|| {
        match MetalKernel::new_with_math(
            "glm53_bf16_gemv_ds4_default_v3",
            &["in_dim", "out_dim", "n_rows", "weights", "x"],
            &["out"],
            BF16_GEMV_SOURCE,
            EXACT_HEADER,
            true,
            false,
            crate::runtime::Math::CompilerDefault,
        ) {
            // Full-write contract: Every projection row has one final SIMD-lane store.
            Ok(kernel) => Some(kernel.with_full_write_outputs()),
            Err(error) => {
                eprintln!("[glm53-bf16-gemv] kernel compilation failed: {error}");
                None
            }
        }
    })
}

fn bf16_gemv_rows2_kernel() -> &'static Option<MetalKernel> {
    static K: std::sync::OnceLock<Option<MetalKernel>> = std::sync::OnceLock::new();
    K.get_or_init(|| {
        MetalKernel::new_with_math(
            "glm53_bf16_gemv_rows2_ds4_order_v2",
            &["in_dim", "out_dim", "n_rows", "weights", "x"],
            &["out"],
            BF16_GEMV_ROWS2_SOURCE,
            EXACT_HEADER,
            true,
            false,
            crate::runtime::Math::CompilerDefault,
        )
        // Full-write contract: Each admitted channel writes both token rows.
        .map(MetalKernel::with_full_write_outputs)
        .ok()
    })
}

fn bf16_pair_gemv_kernel() -> &'static Option<MetalKernel> {
    static K: std::sync::OnceLock<Option<MetalKernel>> = std::sync::OnceLock::new();
    K.get_or_init(|| {
        match MetalKernel::new_with_math(
            "glm53_bf16_pair_gemv_ds4_default_v2",
            &[
                "in_dim_a",
                "out_dim_a",
                "in_dim_b",
                "out_dim_b",
                "n_rows",
                "weights_a",
                "x_a",
                "weights_b",
                "x_b",
            ],
            &["out_a", "out_b"],
            BF16_PAIR_GEMV_SOURCE,
            EXACT_HEADER,
            true,
            false,
            crate::runtime::Math::CompilerDefault,
        ) {
            // Full-write contract: Both projection banks cover every row and token.
            Ok(kernel) => Some(kernel.with_full_write_outputs()),
            Err(error) => {
                eprintln!("[glm53-bf16-pair-gemv] kernel compilation failed: {error}");
                None
            }
        }
    })
}

fn bf16_qkv_kernel() -> &'static Option<MetalKernel> {
    static K: std::sync::OnceLock<Option<MetalKernel>> = std::sync::OnceLock::new();
    K.get_or_init(|| {
        match MetalKernel::new_with_math(
            "glm53_bf16_qkv_ds4_default_v2",
            &[
                "in_dim",
                "out_dim",
                "n_rows",
                "weights_q",
                "weights_k",
                "weights_v",
                "x",
            ],
            &["out_q", "out_k", "out_v"],
            BF16_QKV_SOURCE,
            EXACT_HEADER,
            true,
            false,
            crate::runtime::Math::CompilerDefault,
        ) {
            // Full-write contract: The admitted single row writes every Q/K/V output channel.
            Ok(kernel) => Some(kernel.with_full_write_outputs()),
            Err(error) => {
                eprintln!("[glm53-bf16-qkv] kernel compilation failed: {error}");
                None
            }
        }
    })
}

fn bf16_gemm_kernel() -> &'static Option<MetalKernel> {
    static K: std::sync::OnceLock<Option<MetalKernel>> = std::sync::OnceLock::new();
    K.get_or_init(|| {
        match MetalKernel::new_with_math(
            "glm53_bf16_gemm_ds4_default_v2",
            &["in_dim", "out_dim", "n_rows", "weights", "x"],
            &["out"],
            BF16_GEMM_SOURCE,
            EXACT_HEADER,
            true,
            false,
            crate::runtime::Math::CompilerDefault,
        ) {
            // Full-write contract: Interior and boundary tiles both assign every admitted output element.
            Ok(kernel) => Some(kernel.with_full_write_outputs()),
            Err(error) => {
                eprintln!("[glm53-bf16-gemm] kernel compilation failed: {error}");
                None
            }
        }
    })
}

fn bf16_kda_project_kernel() -> &'static Option<MetalKernel> {
    static K: std::sync::OnceLock<Option<MetalKernel>> = std::sync::OnceLock::new();
    K.get_or_init(|| {
        MetalKernel::new_with_math(
            "glm53_bf16_kda_project_ds4_default_v2",
            &[
                "in_dim",
                "qkv_dim",
                "low_dim",
                "heads",
                "n_rows",
                "weights_q",
                "weights_k",
                "weights_v",
                "weights_fa",
                "weights_ga",
                "weights_beta",
                "x",
            ],
            &["proj", "a_low", "g_low"],
            BF16_KDA_PROJECT_SOURCE,
            EXACT_HEADER,
            true,
            false,
            crate::runtime::Math::CompilerDefault,
        )
        // Full-write contract: Small-bank and Q/K/V tasks cover all three packed outputs.
        .map(MetalKernel::with_full_write_outputs)
        .ok()
    })
}

/// ds4 uses four simdgroups per threadgroup only on M3 Ultra; other devices
/// use eight. Keep the same scheduling split for both BF16 projection kernels.
fn bf16_projection_nsg(x: &Array) -> usize {
    static NSG: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *NSG.get_or_init(|| {
        if x.t
            .device()
            .metal_device()
            .name()
            .to_string()
            .contains("M3 Ultra")
        {
            4
        } else {
            8
        }
    })
}

/// F32-accumulating BF16 GEMV ported from ds4's GLM BF16 row kernel.
/// `weights=[N,K]` BF16 and `x=[M,K]` F32 produce `[M,N]` F32.
pub fn glm53_bf16_gemv(weights: &Array, x: &Array, stream: &Stream) -> Option<Array> {
    if weights.rank() != 2
        || x.rank() != 2
        || weights.dtype() != Dtype::Bfloat16
        || x.dtype() != Dtype::Float32
    {
        return None;
    }
    let (out_dim, in_dim, n_rows) = (
        weights.dim(0) as usize,
        weights.dim(1) as usize,
        x.dim(0) as usize,
    );
    if in_dim == 0 || out_dim == 0 || n_rows == 0 || x.dim(1) as usize != in_dim {
        return None;
    }
    // Large two-row projections reuse each weight without changing either
    // row's reduction. Small output banks need the original independent-row
    // groups to retain occupancy; sharing regresses the KDA low-rank/beta banks.
    let shared_rows = n_rows == 2 && out_dim >= 1024;
    let kernel = if shared_rows {
        bf16_gemv_rows2_kernel()
    } else {
        bf16_gemv_kernel()
    }
    .as_ref()?;
    let weights = weights.contiguous().ok()?;
    let x = x.contiguous().ok()?;
    for a in [&weights, &x] {
        a.eval_enqueue().ok()?;
    }
    let inputs = [
        KernelInput::I32(in_dim as i32),
        KernelInput::I32(out_dim as i32),
        KernelInput::I32(n_rows as i32),
        KernelInput::Tensor(&weights.t),
        KernelInput::Tensor(&x.t),
    ];
    let nsg = bf16_projection_nsg(&x);
    // This MetalKernel wrapper dispatches a thread grid, while the shader
    // derives output rows and tokens from threadgroup positions. The two-row
    // kernel computes both tokens within each group.
    let grid_x = (out_dim.checked_add(nsg - 1)? / nsg).checked_mul(32)?;
    let token_groups = if shared_rows { 1 } else { n_rows };
    let grid_y = token_groups.checked_mul(nsg)?;
    let grid = (i32::try_from(grid_x).ok()?, i32::try_from(grid_y).ok()?, 1);
    let out = match kernel.apply_with_inline_inputs(
        &inputs,
        &[],
        grid,
        (32, nsg as i32, 1),
        &[OutputArg {
            shape: vec![n_rows as i32, out_dim as i32],
            dtype: Dtype::Float32,
        }],
        stream,
    ) {
        Ok(out) => out,
        Err(error) => {
            eprintln!("[glm53-bf16-gemv] dispatch failed: {error}");
            return None;
        }
    };
    out.into_iter().next()
}

fn affine_q8_mv_ds4_kernel() -> &'static Option<MetalKernel> {
    static K: std::sync::OnceLock<Option<MetalKernel>> = std::sync::OnceLock::new();
    K.get_or_init(|| {
        MetalKernel::new_with_math(
            "glm53_affine_q8_mv_ds4_v2",
            &[
                "in_dim",
                "out_dim",
                "n_rows",
                "bits",
                "group_size",
                "metadata_bf16",
                "weights",
                "scales",
                "biases",
                "x",
            ],
            &["out"],
            AFFINE_Q8_MV_DS4_SOURCE,
            EXACT_HEADER,
            true,
            false,
            crate::runtime::Math::CompilerDefault,
        )
        // Full-write contract: Each output row/token has one final lane-zero store.
        .map(MetalKernel::with_full_write_outputs)
        .ok()
    })
}

/// DS4's direct MLX-affine Q8 matvec schedule. Each output/token pair uses one
/// SIMD group: lanes walk the same modulo-32 input slices, dequantize each
/// code, FMA-accumulate against the activation, then use one SIMD reduction.
pub fn glm53_affine_q8_mv_ds4(
    weights: &Array,
    scales: &Array,
    biases: &Array,
    x: &Array,
    stream: &Stream,
) -> Option<Array> {
    let kernel = affine_q8_mv_ds4_kernel().as_ref()?;
    if weights.rank() != 2
        || scales.rank() != 2
        || biases.shape() != scales.shape()
        || x.rank() != 2
        || weights.dtype() != Dtype::Uint32
        || !matches!(scales.dtype(), Dtype::Float32 | Dtype::Bfloat16)
        || biases.dtype() != scales.dtype()
        || x.dtype() != Dtype::Float32
    {
        return None;
    }
    let (out_dim, packed_k) = (weights.dim(0) as usize, weights.dim(1) as usize);
    let in_dim = packed_k.checked_mul(4)?;
    let n_rows = x.dim(0) as usize;
    if out_dim == 0
        || in_dim == 0
        || n_rows == 0
        || x.dim(1) as usize != in_dim
        || in_dim % 64 != 0
        || scales.shape() != [out_dim as i32, (in_dim / 64) as i32]
    {
        return None;
    }
    let weights = weights.contiguous().ok()?;
    let scales = scales.contiguous().ok()?;
    let biases = biases.contiguous().ok()?;
    let x = x.contiguous().ok()?;
    for input in [&weights, &scales, &biases, &x] {
        input.eval_enqueue().ok()?;
    }
    let inputs = [
        KernelInput::I32(i32::try_from(in_dim).ok()?),
        KernelInput::I32(i32::try_from(out_dim).ok()?),
        KernelInput::I32(i32::try_from(n_rows).ok()?),
        KernelInput::I32(8),
        KernelInput::I32(64),
        KernelInput::I32(i32::from(scales.dtype() == Dtype::Bfloat16)),
        KernelInput::Tensor(&weights.t),
        KernelInput::Tensor(&scales.t),
        KernelInput::Tensor(&biases.t),
        KernelInput::Tensor(&x.t),
    ];
    let result = kernel.apply_with_inline_inputs(
        &inputs,
        &[],
        (
            i32::try_from(out_dim.checked_mul(32)?).ok()?,
            i32::try_from(n_rows).ok()?,
            1,
        ),
        (32, 1, 1),
        &[OutputArg {
            shape: vec![n_rows as i32, out_dim as i32],
            dtype: Dtype::Float32,
        }],
        stream,
    );
    let out = match result {
        Ok(out) => out,
        Err(error) => {
            eprintln!("[glm53-affine-q8-ds4] dispatch failed: {error}");
            return None;
        }
    };
    out.into_iter().next()
}

/// Paired BF16-weight/F32-activation GEMV. Each result uses the same DS4
/// lane-strided FMA and SIMD reduction as `glm53_bf16_gemv`, but both
/// projections share one dispatch. Inputs are `[M,K_a]` and `[M,K_b]`;
/// weights are `[N_a,K_a]` and `[N_b,K_b]`.
pub fn glm53_bf16_pair_gemv(
    weights_a: &Array,
    x_a: &Array,
    weights_b: &Array,
    x_b: &Array,
    stream: &Stream,
) -> Option<(Array, Array)> {
    let kernel = bf16_pair_gemv_kernel().as_ref()?;
    if [weights_a, weights_b]
        .iter()
        .any(|w| w.rank() != 2 || w.dtype() != Dtype::Bfloat16)
        || [x_a, x_b]
            .iter()
            .any(|x| x.rank() != 2 || x.dtype() != Dtype::Float32)
    {
        return None;
    }
    let (out_dim_a, in_dim_a, n_rows) = (
        weights_a.dim(0) as usize,
        weights_a.dim(1) as usize,
        x_a.dim(0) as usize,
    );
    let (out_dim_b, in_dim_b) = (weights_b.dim(0) as usize, weights_b.dim(1) as usize);
    if in_dim_a == 0
        || in_dim_b == 0
        || out_dim_a == 0
        || out_dim_b == 0
        || n_rows == 0
        || x_a.dim(1) as usize != in_dim_a
        || x_b.dim(0) as usize != n_rows
        || x_b.dim(1) as usize != in_dim_b
        || [in_dim_a, in_dim_b, out_dim_a, out_dim_b, n_rows]
            .iter()
            .any(|&v| v > i32::MAX as usize)
    {
        return None;
    }
    let weights_a = weights_a.contiguous().ok()?;
    let x_a = x_a.contiguous().ok()?;
    let weights_b = weights_b.contiguous().ok()?;
    let x_b = x_b.contiguous().ok()?;
    for array in [&weights_a, &x_a, &weights_b, &x_b] {
        array.eval_enqueue().ok()?;
    }
    let in_a = Array::from_int(in_dim_a as i32);
    let out_a = Array::from_int(out_dim_a as i32);
    let in_b = Array::from_int(in_dim_b as i32);
    let out_b = Array::from_int(out_dim_b as i32);
    let rows_a = Array::from_int(n_rows as i32);
    let total_out = out_dim_a.checked_add(out_dim_b)?;
    if total_out > i32::MAX as usize {
        return None;
    }
    let inputs: [&Array; 9] = [
        &in_a, &out_a, &in_b, &out_b, &rows_a, &weights_a, &x_a, &weights_b, &x_b,
    ];
    let nsg = bf16_projection_nsg(&x_a);
    let grid_x = total_out
        .checked_add(nsg - 1)?
        .checked_div(nsg)?
        .checked_mul(32)?;
    let grid_y = n_rows.checked_mul(nsg)?;
    let outputs = match kernel.apply(
        &inputs,
        &[],
        (i32::try_from(grid_x).ok()?, i32::try_from(grid_y).ok()?, 1),
        (32, nsg as i32, 1),
        &[
            OutputArg {
                shape: vec![n_rows as i32, out_dim_a as i32],
                dtype: Dtype::Float32,
            },
            OutputArg {
                shape: vec![n_rows as i32, out_dim_b as i32],
                dtype: Dtype::Float32,
            },
        ],
        stream,
    ) {
        Ok(outputs) => outputs,
        Err(error) => {
            eprintln!("[glm53-bf16-pair-gemv] dispatch failed: {error}");
            return None;
        }
    };
    let mut outputs = outputs.into_iter();
    Some((outputs.next()?, outputs.next()?))
}

/// One-dispatch KDA input projections for one to eight rows, preserving the
/// BF16 GEMV lane order used by `glm53_bf16_gemv`. The packed first output is
/// row-major `[q | k | v | beta]`; the other outputs are `f_a(x)` and `g_a(x)`.
pub fn glm53_bf16_kda_project(
    weights_q: &Array,
    weights_k: &Array,
    weights_v: &Array,
    weights_fa: &Array,
    weights_ga: &Array,
    weights_beta: &Array,
    x: &Array,
    stream: &Stream,
) -> Option<(Array, Array, Array)> {
    glm53_bf16_kda_project_with_nsg(
        weights_q,
        weights_k,
        weights_v,
        weights_fa,
        weights_ga,
        weights_beta,
        x,
        None,
        stream,
    )
}

/// Fused KDA projection with an optional simdgroup-count override used by the
/// serial-audit schedule experiment. `None` retains the DS4 device schedule.
pub fn glm53_bf16_kda_project_with_nsg(
    weights_q: &Array,
    weights_k: &Array,
    weights_v: &Array,
    weights_fa: &Array,
    weights_ga: &Array,
    weights_beta: &Array,
    x: &Array,
    nsg_override: Option<usize>,
    stream: &Stream,
) -> Option<(Array, Array, Array)> {
    let kernel = bf16_kda_project_kernel().as_ref()?;
    if x.rank() != 2 || !(1..=8).contains(&(x.dim(0) as usize)) || x.dtype() != Dtype::Float32 {
        return None;
    }
    let n_rows = x.dim(0) as usize;
    let in_dim = x.dim(1) as usize;
    let qkv_dim = weights_q.dim(0) as usize;
    let low_dim = weights_fa.dim(0) as usize;
    let heads = weights_beta.dim(0) as usize;
    let packed_dim = qkv_dim.checked_mul(3)?.checked_add(heads)?;
    if in_dim == 0
        || qkv_dim == 0
        || low_dim == 0
        || heads == 0
        || [
            weights_q,
            weights_k,
            weights_v,
            weights_fa,
            weights_ga,
            weights_beta,
        ]
        .iter()
        .any(|w| w.rank() != 2 || w.dtype() != Dtype::Bfloat16 || w.dim(1) as usize != in_dim)
        || [weights_k, weights_v]
            .iter()
            .any(|w| w.dim(0) as usize != qkv_dim)
        || weights_ga.dim(0) as usize != low_dim
        || [in_dim, qkv_dim, low_dim, heads, packed_dim]
            .iter()
            .any(|&v| v > i32::MAX as usize)
    {
        return None;
    }
    let q = weights_q.contiguous().ok()?;
    let k = weights_k.contiguous().ok()?;
    let v = weights_v.contiguous().ok()?;
    let fa = weights_fa.contiguous().ok()?;
    let ga = weights_ga.contiguous().ok()?;
    let beta = weights_beta.contiguous().ok()?;
    let x = x.contiguous().ok()?;
    for a in [&q, &k, &v, &fa, &ga, &beta, &x] {
        a.eval_enqueue().ok()?;
    }
    let inputs = [
        KernelInput::I32(in_dim as i32),
        KernelInput::I32(qkv_dim as i32),
        KernelInput::I32(low_dim as i32),
        KernelInput::I32(heads as i32),
        KernelInput::I32(n_rows as i32),
        KernelInput::Tensor(&q.t),
        KernelInput::Tensor(&k.t),
        KernelInput::Tensor(&v.t),
        KernelInput::Tensor(&fa.t),
        KernelInput::Tensor(&ga.t),
        KernelInput::Tensor(&beta.t),
        KernelInput::Tensor(&x.t),
    ];
    let nsg = nsg_override.unwrap_or_else(|| bf16_projection_nsg(&x));
    if !matches!(nsg, 4 | 8 | 16) {
        return None;
    }
    let small_rows = low_dim.checked_mul(2)?.checked_add(heads)?;
    let small_groups = small_rows.checked_add(nsg - 1)?.checked_div(nsg)?;
    let qkv_groups = qkv_dim.checked_add(nsg - 1)?.checked_div(nsg)?;
    let task_groups = small_groups.checked_add(qkv_groups.checked_mul(3)?)?;
    let grid_x = task_groups.checked_mul(n_rows)?.checked_mul(32)?;
    let grid = (i32::try_from(grid_x).ok()?, nsg as i32, 1);
    let out = match kernel.apply_with_inline_inputs(
        &inputs,
        &[],
        grid,
        (32, nsg as i32, 1),
        &[
            OutputArg {
                shape: vec![n_rows as i32, packed_dim as i32],
                dtype: Dtype::Float32,
            },
            OutputArg {
                shape: vec![n_rows as i32, low_dim as i32],
                dtype: Dtype::Float32,
            },
            OutputArg {
                shape: vec![n_rows as i32, low_dim as i32],
                dtype: Dtype::Float32,
            },
        ],
        stream,
    ) {
        Ok(out) => out,
        Err(error) => {
            eprintln!("[glm53-bf16-kda-project] dispatch failed: {error}");
            return None;
        }
    };
    let mut out = out.into_iter();
    Some((out.next()?, out.next()?, out.next()?))
}

/// Tiled BF16-weight/F32-activation GEMM for GLM prefill. This follows ds4's
/// large-row route: 64x32 output tiles across four simdgroups, 32-wide
/// reductions, F16 operand staging, and F32 accumulation. `weights=[N,K]`,
/// `x=[M,K]` -> `[M,N]`.
pub fn glm53_bf16_gemm(weights: &Array, x: &Array, stream: &Stream) -> Option<Array> {
    const TILE_N: usize = 64;
    const TILE_M: usize = 32;
    let kernel = bf16_gemm_kernel().as_ref()?;
    if weights.rank() != 2
        || x.rank() != 2
        || weights.dtype() != Dtype::Bfloat16
        || x.dtype() != Dtype::Float32
    {
        return None;
    }
    let (out_dim, in_dim, n_rows) = (
        weights.dim(0) as usize,
        weights.dim(1) as usize,
        x.dim(0) as usize,
    );
    if in_dim == 0
        || out_dim == 0
        || n_rows <= 8
        || x.dim(1) as usize != in_dim
        || [in_dim, out_dim, n_rows]
            .iter()
            .any(|&v| v > i32::MAX as usize)
    {
        return None;
    }
    let weights = weights.contiguous().ok()?;
    let x = x.contiguous().ok()?;
    weights.eval_enqueue().ok()?;
    x.eval_enqueue().ok()?;
    let in_a = Array::from_int(in_dim as i32);
    let out_a = Array::from_int(out_dim as i32);
    let rows_a = Array::from_int(n_rows as i32);
    let inputs: [&Array; 5] = [&in_a, &out_a, &rows_a, &weights, &x];
    let grid_x = n_rows.div_ceil(TILE_M).checked_mul(128)?;
    // The 128 threads are laid out only along x; y indexes output tiles with
    // one threadgroup per feature tile.
    let grid_y = out_dim.div_ceil(TILE_N);
    let grid = (i32::try_from(grid_x).ok()?, i32::try_from(grid_y).ok()?, 1);
    let out = match kernel.apply(
        &inputs,
        &[],
        grid,
        (128, 1, 1),
        &[OutputArg {
            shape: vec![n_rows as i32, out_dim as i32],
            dtype: Dtype::Float32,
        }],
        stream,
    ) {
        Ok(out) => out,
        Err(error) => {
            eprintln!("[glm53-bf16-gemm] dispatch failed: {error}");
            return None;
        }
    };
    out.into_iter().next()
}

/// Fused BF16 q/k/v projection ported from ds4's decode kernel.
/// Each matrix is `[N,K]` BF16; the common input is `[1,K]` F32.
pub fn glm53_bf16_qkv(
    weights_q: &Array,
    weights_k: &Array,
    weights_v: &Array,
    x: &Array,
    stream: &Stream,
) -> Option<(Array, Array, Array)> {
    // ds4 only uses its fused QKV lane on M3 Ultra. Other devices perform
    // three independent BF16 GEMVs, which benchmark better there.
    if bf16_projection_nsg(x) != 4 {
        return None;
    }
    glm53_bf16_qkv_inner(weights_q, weights_k, weights_v, x, stream)
}

fn glm53_bf16_qkv_inner(
    weights_q: &Array,
    weights_k: &Array,
    weights_v: &Array,
    x: &Array,
    stream: &Stream,
) -> Option<(Array, Array, Array)> {
    let kernel = bf16_qkv_kernel().as_ref()?;
    if x.rank() != 2 || x.dim(0) != 1 || x.dtype() != Dtype::Float32 {
        return None;
    }
    let (out_dim, in_dim) = (weights_q.dim(0) as usize, weights_q.dim(1) as usize);
    if [weights_q, weights_k, weights_v].iter().any(|w| {
        w.rank() != 2
            || w.dtype() != Dtype::Bfloat16
            || w.dim(0) as usize != out_dim
            || w.dim(1) as usize != in_dim
    }) || x.dim(1) as usize != in_dim
        || out_dim == 0
    {
        return None;
    }
    let q = weights_q.contiguous().ok()?;
    let k = weights_k.contiguous().ok()?;
    let v = weights_v.contiguous().ok()?;
    let x = x.contiguous().ok()?;
    let in_a = Array::from_int(in_dim as i32);
    let out_a = Array::from_int(out_dim as i32);
    let rows_a = Array::from_int(1);
    for a in [&q, &k, &v, &x] {
        a.eval_enqueue().ok()?;
    }
    let inputs: [&Array; 7] = [&in_a, &out_a, &rows_a, &q, &k, &v, &x];
    let nsg = bf16_projection_nsg(&x);
    let grid_x = (out_dim.checked_add(nsg - 1)? / nsg).checked_mul(32)?;
    let out = match kernel.apply(
        &inputs,
        &[],
        (i32::try_from(grid_x).ok()?, nsg as i32, 3),
        (32, nsg as i32, 1),
        &[
            OutputArg {
                shape: vec![1, out_dim as i32],
                dtype: Dtype::Float32,
            },
            OutputArg {
                shape: vec![1, out_dim as i32],
                dtype: Dtype::Float32,
            },
            OutputArg {
                shape: vec![1, out_dim as i32],
                dtype: Dtype::Float32,
            },
        ],
        stream,
    ) {
        Ok(out) => out,
        Err(error) => {
            eprintln!("[glm53-bf16-qkv] dispatch failed: {error}");
            return None;
        }
    };
    let mut it = out.into_iter();
    Some((it.next()?, it.next()?, it.next()?))
}

fn kda_prepare_kernel() -> &'static Option<MetalKernel> {
    static K: std::sync::OnceLock<Option<MetalKernel>> = std::sync::OnceLock::new();
    K.get_or_init(|| {
        MetalKernel::new_with_math(
            "kda_prepare_ds4_default_v6",
            &[
                "n_heads",
                "n_rows",
                "lower_bound",
                "norm_eps",
                "q_in",
                "k_in",
                "v_in",
                "gate_in",
                "q_conv",
                "k_conv",
                "v_conv",
                "a_log",
                "dt_bias",
                "conv_state_in",
            ],
            &["q_out", "k_out", "v_out", "g_out", "conv_state_out"],
            KDA_PREPARE_SOURCE,
            EXACT_HEADER,
            true,
            false,
            crate::runtime::Math::CompilerDefault,
        )
        // Full-write contract: Every token channel and every convolution-history cell is assigned.
        .map(MetalKernel::with_full_write_outputs)
        .ok()
    })
}

fn kda_decode_kernel() -> &'static Option<MetalKernel> {
    static K: std::sync::OnceLock<Option<MetalKernel>> = std::sync::OnceLock::new();
    K.get_or_init(|| {
        MetalKernel::new_with_math(
            "glm53_kda_decode_ds4_default_v2",
            &[
                "n_heads",
                "lower_bound",
                "norm_eps",
                "q",
                "k",
                "v",
                "raw_gate",
                "raw_beta",
                "output_gate",
                "q_conv",
                "k_conv",
                "v_conv",
                "a_log",
                "dt_bias",
                "output_norm",
                "conv_state_in",
                "state_in",
            ],
            &["out", "conv_state_out", "state_out"],
            KDA_DECODE_SOURCE,
            EXACT_HEADER,
            true,
            false,
            crate::runtime::Math::CompilerDefault,
        )
        // Full-write contract: Each head assigns all output, convolution-history, and state cells.
        .map(MetalKernel::with_full_write_outputs)
        .ok()
    })
}

/// Fused one-token KDA decode, following ds4's single-dispatch kernel.
/// All arrays are contiguous F32; state/cache layouts match ds4.
#[allow(clippy::too_many_arguments)]
pub fn kda_decode(
    q: &Array,
    k: &Array,
    v: &Array,
    raw_gate: &Array,
    raw_beta: &Array,
    output_gate: &Array,
    q_conv: &Array,
    k_conv: &Array,
    v_conv: &Array,
    a_log: &Array,
    dt_bias: &Array,
    output_norm: &Array,
    conv_state: &Array,
    state: &Array,
    n_heads: usize,
    lower_bound: f32,
    norm_eps: f32,
    stream: &Stream,
) -> Option<(Array, Array, Array)> {
    let kernel = kda_decode_kernel().as_ref()?;
    let d = 128usize;
    let projection = n_heads.checked_mul(d)?;
    let inputs = [
        KernelInput::I32(n_heads as i32),
        KernelInput::F32(lower_bound),
        KernelInput::F32(norm_eps),
        KernelInput::Tensor(&q.t),
        KernelInput::Tensor(&k.t),
        KernelInput::Tensor(&v.t),
        KernelInput::Tensor(&raw_gate.t),
        KernelInput::Tensor(&raw_beta.t),
        KernelInput::Tensor(&output_gate.t),
        KernelInput::Tensor(&q_conv.t),
        KernelInput::Tensor(&k_conv.t),
        KernelInput::Tensor(&v_conv.t),
        KernelInput::Tensor(&a_log.t),
        KernelInput::Tensor(&dt_bias.t),
        KernelInput::Tensor(&output_norm.t),
        KernelInput::Tensor(&conv_state.t),
        KernelInput::Tensor(&state.t),
    ];
    for a in [
        q,
        k,
        v,
        raw_gate,
        raw_beta,
        output_gate,
        q_conv,
        k_conv,
        v_conv,
        a_log,
        dt_bias,
        output_norm,
        conv_state,
        state,
    ] {
        if let Err(e) = a.eval_enqueue() {
            eprintln!("[kda-decode-eval] {e}");
            return None;
        }
    }
    let outputs = [
        OutputArg {
            shape: vec![projection as i32],
            dtype: Dtype::Float32,
        },
        OutputArg {
            shape: vec![3, (3 * projection) as i32],
            dtype: Dtype::Float32,
        },
        OutputArg {
            shape: vec![n_heads as i32, d as i32, d as i32],
            dtype: Dtype::Float32,
        },
    ];
    let out = match kernel.apply_with_inline_inputs(
        &inputs,
        &[],
        (d as i32, n_heads as i32, 1),
        (128, 1, 1),
        &outputs,
        stream,
    ) {
        Ok(o) => o,
        Err(e) => {
            eprintln!("[kda-decode-decline] {e}");
            return None;
        }
    };
    let mut it = out.into_iter();
    Some((it.next()?, it.next()?, it.next()?))
}

fn kda_output_kernel() -> &'static Option<MetalKernel> {
    static K: std::sync::OnceLock<Option<MetalKernel>> = std::sync::OnceLock::new();
    K.get_or_init(|| {
        MetalKernel::new_with_math(
            "kda_output_ds4_default_v4",
            &[
                "n_heads",
                "n_rows",
                "norm_eps",
                "in",
                "output_gate",
                "output_norm",
            ],
            &["out"],
            KDA_OUTPUT_SOURCE,
            EXACT_HEADER,
            true,
            false,
            crate::runtime::Math::CompilerDefault,
        )
        // Full-write contract: Every head/token channel is assigned by its lane.
        .map(MetalKernel::with_full_write_outputs)
        .ok()
    })
}

/// Fused KDA prefill prepare: depthwise conv + silu + l2norm + compute_g_safe.
/// All inputs f32. `q/k/v_in` are the raw projections [B, T, heads*D],
/// `gate_in` the raw f_b(f_a(x)), `conv_state` the planar [3, 3*heads*D]
/// window (q history, then k history, then v history; B=1 path). Returns
/// `(q, k, v, g, conv_state_out)` with q/k l2-normalized
/// (q scaled by D^-0.5), g already exponentiated (decayed gate), all f32.
#[allow(clippy::too_many_arguments)]
pub fn kda_prepare(
    q: &Array,
    k: &Array,
    v: &Array,
    gate: &Array,
    q_conv: &Array,
    k_conv: &Array,
    v_conv: &Array,
    a_log: &Array,
    dt_bias: &Array,
    conv_state: &Array,
    n_heads: usize,
    lower_bound: f32,
    norm_eps: f32,
    stream: &Stream,
) -> Option<(Array, Array, Array, Array, Array)> {
    let kernel = kda_prepare_kernel().as_ref()?;
    // Bind only materialised buffers: evaluate every input (lazy casts and
    // matmul outputs otherwise hand the kernel uninitialised memory).
    for a in [
        q, k, v, gate, q_conv, k_conv, v_conv, a_log, dt_bias, conv_state,
    ] {
        a.eval_enqueue()
            .map_err(|e| eprintln!("[kda-prepare-eval] {e}"))
            .ok();
    }
    let t = q.dim(1);
    let d = (q.dim(2) / n_heads as i32) as usize;
    let rows_a = crate::models::gdn::t_scalar_pub(t as i32);
    let n_heads_a = Array::from_int(n_heads as i32);
    let lb = Array::from_f32_as(lower_bound, Dtype::Float32);
    let eps = Array::from_f32_as(norm_eps, Dtype::Float32);
    let inputs: [&Array; 10] = [
        q, k, v, gate, q_conv, k_conv, v_conv, a_log, dt_bias, conv_state,
    ];
    let scalars: [&Array; 4] = [&n_heads_a, &rows_a, &lb, &eps];
    let proj = n_heads * d;
    let outputs = [
        OutputArg {
            shape: vec![t as i32, proj as i32],
            dtype: crate::Dtype::Float32,
        },
        OutputArg {
            shape: vec![t as i32, proj as i32],
            dtype: crate::Dtype::Float32,
        },
        OutputArg {
            shape: vec![t as i32, proj as i32],
            dtype: crate::Dtype::Float32,
        },
        OutputArg {
            shape: vec![t as i32, proj as i32],
            dtype: crate::Dtype::Float32,
        },
        OutputArg {
            shape: vec![3i32, (3 * proj) as i32],
            dtype: crate::Dtype::Float32,
        },
    ];
    let mut all: Vec<&Array> = scalars.iter().map(|a| &**a).collect();
    all.extend(inputs.iter().copied());
    let out = match kernel.apply(
        &all,
        &[],
        ((n_heads * 128) as i32, 1, 1),
        (128, 1, 1),
        &outputs,
        stream,
    ) {
        Ok(o) => o,
        Err(e) => {
            eprintln!("[kda-prepare-decline] {e}");
            return None;
        }
    };
    let mut it = out.into_iter();
    let qo = it.next()?;
    let ko = it.next()?;
    let vo = it.next()?;
    let go = it.next()?;
    let cs = it.next()?;
    Some((qo, ko, vo, go, cs))
}

/// Fused gated RMS-norm output stage: `in` [B, T, heads*D] f32 (recurrence
/// output), `gate` [B, T, heads*D] f32 pre-sigmoid, `w` [D] f32.
pub fn kda_output_norm(
    y: &Array,
    gate: &Array,
    w: &Array,
    n_heads: usize,
    norm_eps: f32,
    stream: &Stream,
) -> Option<Array> {
    let kernel = kda_output_kernel().as_ref()?;
    let t_ = (y.dim((y.rank() - 2) as i32)) as usize;
    let d = w.size();
    let t_scalar = crate::models::gdn::t_scalar_pub(t_ as i32);
    let nh = Array::from_int(n_heads as i32);
    let eps = Array::from_f32_as(norm_eps, Dtype::Float32);
    let inputs: [&Array; 3] = [y, gate, w];
    let scalars: [&Array; 3] = [&nh, &t_scalar, &eps];
    let mut all: Vec<&Array> = scalars.iter().map(|a| &**a).collect();
    all.extend(inputs.iter().copied());
    let outputs = [OutputArg {
        shape: vec![(t_ * n_heads * d) as i32],
        dtype: crate::Dtype::Float32,
    }];
    let out = match kernel.apply(
        &all,
        &[],
        ((n_heads * 128) as i32, t_ as i32, 1),
        (128, 1, 1),
        &outputs,
        stream,
    ) {
        Ok(o) => o,
        Err(e) => {
            eprintln!("[kda-output-decline] {e}");
            return None;
        }
    };
    Some(out.into_iter().next()?)
}

const RMS_ROW_SOURCE: &str = include_str!("../../shaders/glm53/rms_row.metal");

fn rms_row_kernel() -> &'static Option<MetalKernel> {
    static K: std::sync::OnceLock<Option<MetalKernel>> = std::sync::OnceLock::new();
    K.get_or_init(|| {
        MetalKernel::new_with_math(
            "glm53_rms_row_ds4_default_v3",
            &["n_groups", "eps", "x"],
            &["y"],
            RMS_ROW_SOURCE,
            EXACT_HEADER,
            true,
            false,
            crate::runtime::Math::CompilerDefault,
        )
        // Full-write contract: Each SIMD lane assigns its four channels in every admitted group.
        .map(MetalKernel::with_full_write_outputs)
        .ok()
    })
}

/// Fused row/group RMS norm (ds4 norm.metal port): x bf16 [..., w_full] with
/// w_full = h * n_groups, h = 128. One kernel replaces the ~30-op grouped
/// butterfly dance (two f32 casts materialised, per-lane sums, two butterflies)
/// with the same pinned association: 4 sequential squares per lane, simd_sum
/// over 32 lanes, rsqrt(total/128 + eps), bf16(x * inv). The bf16 weight
/// multiply is applied by the caller AFTER this (engine rounding order).
/// Returns None when the kernel declines (caller falls back to the op path).
pub fn rms_row_fused(x: &Array, eps: f32, h: usize, stream: &Stream) -> Option<Array> {
    if h != 128 || x.dtype() != Dtype::Bfloat16 {
        return None;
    }
    let kernel = rms_row_kernel().as_ref()?;
    x.eval_enqueue()
        .map_err(|e| eprintln!("[rms-row-eval] {e}"))
        .ok();
    let w_full = *x.shape().last().unwrap() as usize;
    if w_full % h != 0 {
        return None;
    }
    let n_groups = w_full / h;
    let lead: i32 = x.shape()[..x.shape().len() - 1].iter().product::<i32>();
    let ng = Array::from_int(n_groups as i32);
    let eps_a = Array::from_f32_as(eps, Dtype::Float32);
    let inputs: [&Array; 2] = [&ng, &eps_a];
    let outputs = [OutputArg {
        shape: x.shape().to_vec(),
        dtype: Dtype::Bfloat16,
    }];
    let all: Vec<&Array> = inputs
        .iter()
        .map(|a| &**a)
        .chain(std::iter::once(x))
        .collect();
    let out = match kernel.apply(
        &all,
        &[],
        ((n_groups * 32) as i32, lead, 1),
        (32, 1, 1),
        &outputs,
        stream,
    ) {
        Ok(o) => o,
        Err(e) => {
            eprintln!("[rms-row-decline] {e}");
            return None;
        }
    };
    Some(out.into_iter().next()?)
}

const HC_SPLIT_SOURCE: &str = include_str!("../../shaders/glm53/hc_split.metal");
const HC_SPLIT_NORM4_SOURCE: &str = include_str!("../../shaders/glm53/hc_split_norm4.metal");
const HC_EXPAND_SOURCE: &str = include_str!("../../shaders/glm53/hc_expand.metal");
const HC_WSUM_SOURCE: &str = include_str!("../../shaders/glm53/hc_wsum.metal");

fn hc_wsum_kernel() -> &'static Option<MetalKernel> {
    static K: std::sync::OnceLock<Option<MetalKernel>> = std::sync::OnceLock::new();
    K.get_or_init(|| {
        MetalKernel::new_with_math(
            "glm53_hc_wsum_ds4_default_v3",
            &["n_rows", "hidden", "residual", "pre"],
            &["dst"],
            HC_WSUM_SOURCE,
            EXACT_HEADER,
            true,
            false,
            crate::runtime::Math::CompilerDefault,
        )
        // Full-write contract: Each row/channel task assigns one collapsed output element.
        .map(MetalKernel::with_full_write_outputs)
        .ok()
    })
}

/// Fused mHC collapse (ds4 weighted_sum port): residual [B,T,4,H] f32,
/// pre [B,T,4] f32 -> x [B,T,H] f32. One dispatch replaces the narrow/add
/// op chain.
pub fn hc_weighted_sum(residual: &Array, pre: &Array, stream: &Stream) -> Option<Array> {
    let kernel = hc_wsum_kernel().as_ref()?;
    let sh = residual.shape();
    let b = sh[0];
    let t = sh[1];
    let h = *sh.last().unwrap();
    let n_rows = b * t;
    let nr = Array::from_int(n_rows as i32);
    let h_a = Array::from_int(h as i32);
    for a in [residual, pre] {
        if let Err(e) = a.eval_enqueue() {
            eprintln!("[hc-wsum-decline] {e}");
            return None;
        }
    }
    let inputs: [&Array; 4] = [&nr, &h_a, residual, pre];
    let outputs = [OutputArg {
        shape: vec![b, t, h],
        dtype: Dtype::Float32,
    }];
    let all: Vec<&Array> = inputs.iter().map(|a| &**a).collect();
    let out = match kernel.apply(
        &all,
        &[],
        ((n_rows * h) as i32, 1, 1),
        (1, 1, 1),
        &outputs,
        stream,
    ) {
        Ok(o) => o,
        Err(e) => {
            eprintln!("[hc-wsum-decline] {e}");
            return None;
        }
    };
    Some(out.into_iter().next()?)
}

fn hc_split_kernel() -> &'static Option<MetalKernel> {
    static K: std::sync::OnceLock<Option<MetalKernel>> = std::sync::OnceLock::new();
    K.get_or_init(|| {
        MetalKernel::new_with_math(
            "glm53_hc_split_sinkhorn_ds4_default_v3",
            &["n_rows", "sinkhorn_iters", "eps", "mixes", "scale", "base"],
            &["out"],
            HC_SPLIT_SOURCE,
            EXACT_HEADER,
            true,
            false,
            crate::runtime::Math::CompilerDefault,
        )
        // Full-write contract: Each row assigns all 24 split fields.
        .map(MetalKernel::with_full_write_outputs)
        .ok()
    })
}

fn hc_split_norm4_kernel() -> &'static Option<MetalKernel> {
    static K: std::sync::OnceLock<Option<MetalKernel>> = std::sync::OnceLock::new();
    K.get_or_init(|| {
        MetalKernel::new_with_math_and_threadgroup_memory(
            "glm53_hc_split_weighted_sum_norm4_ds4_default_v2",
            &[
                "n_rows",
                "hidden",
                "sinkhorn_iters",
                "eps",
                "norm_eps",
                "mixes",
                "scale",
                "base",
                "x",
                "norm_weight",
            ],
            &["split", "post", "comb", "dst", "normed"],
            HC_SPLIT_NORM4_SOURCE,
            EXACT_HEADER,
            true,
            false,
            crate::runtime::Math::CompilerDefault,
            (4096 + 4 + 32) * std::mem::size_of::<f32>(),
        )
        // Full-write contract: Each row assigns all split/post/comb fields and all dst/normed channels.
        .map(MetalKernel::with_full_write_outputs)
        .ok()
    })
}

fn hc_expand_kernel() -> &'static Option<MetalKernel> {
    static K: std::sync::OnceLock<Option<MetalKernel>> = std::sync::OnceLock::new();
    K.get_or_init(|| {
        MetalKernel::new_with_math(
            "glm53_hc_expand4_ds4_default_v3",
            &["n_rows", "hidden", "y", "residual", "post", "comb"],
            &["dst"],
            HC_EXPAND_SOURCE,
            EXACT_HEADER,
            true,
            false,
            crate::runtime::Math::CompilerDefault,
        )
        // Full-write contract: Each row/channel task assigns all four expanded streams.
        .map(MetalKernel::with_full_write_outputs)
        .ok()
    })
}

/// Fused HC split+sinkhorn: mixes [N,24] f32 -> split [N,24] f32
/// (pre out[0:4], post out[4:8], comb out[8:24] row-major [source][destination]).
pub fn hc_split_sinkhorn(
    mixes: &Array,
    scale: &Array,
    base: &Array,
    eps: f32,
    sinkhorn_iters: usize,
    stream: &Stream,
) -> Option<Array> {
    let kernel = hc_split_kernel().as_ref()?;
    let n = mixes.dim(0);
    let n_a = Array::from_int(n);
    let it_a = Array::from_int(sinkhorn_iters as i32);
    let eps_a = Array::from_f32_as(eps, Dtype::Float32);
    let s32 = match scale.as_dtype(Dtype::Float32) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("[hc-split-decline] {e}");
            return None;
        }
    };
    let b32 = match base.as_dtype(Dtype::Float32) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("[hc-split-decline] {e}");
            return None;
        }
    };
    if let Err(e) = s32.eval_enqueue() {
        eprintln!("[hc-split-decline] {e}");
        return None;
    }
    if let Err(e) = b32.eval_enqueue() {
        eprintln!("[hc-split-decline] {e}");
        return None;
    }
    if let Err(e) = mixes.eval_enqueue() {
        eprintln!("[hc-split-decline] {e}");
        return None;
    }
    let inputs: [&Array; 6] = [&n_a, &it_a, &eps_a, mixes, &s32, &b32];
    let outputs = [OutputArg {
        shape: vec![n, 24],
        dtype: Dtype::Float32,
    }];
    let all: Vec<&Array> = inputs.iter().map(|a| &**a).collect();
    let out = match kernel.apply(&all, &[], (n, 1, 1), (1, 1, 1), &outputs, stream) {
        Ok(o) => o,
        Err(e) => {
            eprintln!("[hc-split-decline] {e}");
            return None;
        }
    };
    Some(out.into_iter().next()?)
}

/// ds4's HC=4 fused coefficient split, stream collapse, and RMSNorm path.
/// `mixes=[B*T,24]`, `x=[B,T,4,4096]`; returns `(split, post, comb, collapsed, normed)`.
pub fn hc_split_weighted_sum_norm4(
    mixes: &Array,
    scale: &Array,
    base: &Array,
    x: &Array,
    norm_weight: &Array,
    eps: f32,
    norm_eps: f32,
    sinkhorn_iters: usize,
    stream: &Stream,
) -> Option<(Array, Array, Array, Array, Array)> {
    let kernel = hc_split_norm4_kernel().as_ref()?;
    if mixes.rank() != 2
        || mixes.dim(1) != 24
        || x.rank() != 4
        || x.dim(2) != 4
        || x.dim(3) != 4096
        || norm_weight.shape() != [4096]
    {
        return None;
    }
    let (b, t, hidden) = (x.dim(0), x.dim(1), x.dim(3));
    let n_rows = b.checked_mul(t)?;
    if mixes.dim(0) != n_rows || scale.size() != 3 || base.size() != 24 {
        return None;
    }
    let scale = scale.as_dtype(Dtype::Float32).ok()?;
    let base = base.as_dtype(Dtype::Float32).ok()?;
    let norm_weight = norm_weight.as_dtype(Dtype::Float32).ok()?;
    let x = x.as_dtype(Dtype::Float32).ok()?.contiguous().ok()?;
    for input in [mixes, &scale, &base, &x, &norm_weight] {
        input.eval_enqueue().ok()?;
    }
    let inputs = [
        KernelInput::I32(n_rows),
        KernelInput::I32(hidden),
        KernelInput::I32(sinkhorn_iters as i32),
        KernelInput::F32(eps),
        KernelInput::F32(norm_eps),
        KernelInput::Tensor(&mixes.t),
        KernelInput::Tensor(&scale.t),
        KernelInput::Tensor(&base.t),
        KernelInput::Tensor(&x.t),
        KernelInput::Tensor(&norm_weight.t),
    ];
    let out = kernel
        .apply_with_inline_inputs(
            &inputs,
            &[],
            // Match ds4_gpu_rms_norm_threads(hidden): it divides the hidden
            // width by four and rounds the lane count up to a power of two.
            // For GLM's hidden=4096 specialization this is 1024 threads. The
            // reduction grouping is numerically observable, so do not tune
            // this launch width independently of the DS4 reference.
            (n_rows.checked_mul(1024)?, 1, 1),
            (1024, 1, 1),
            &[
                OutputArg {
                    shape: vec![n_rows, 24],
                    dtype: Dtype::Float32,
                },
                OutputArg {
                    shape: vec![b, t, 4],
                    dtype: Dtype::Float32,
                },
                OutputArg {
                    shape: vec![b, t, 4, 4],
                    dtype: Dtype::Float32,
                },
                OutputArg {
                    shape: vec![n_rows, hidden],
                    dtype: Dtype::Float32,
                },
                OutputArg {
                    shape: vec![n_rows, hidden],
                    dtype: Dtype::Float32,
                },
            ],
            stream,
        )
        .ok()?;
    let mut out = out.into_iter();
    Some((
        out.next()?,
        out.next()?,
        out.next()?,
        out.next()?,
        out.next()?,
    ))
}

/// Fused HC expand4: y [B,T,H] f32, residual [B,T,4,H] f32, post [B,T,4] f32,
/// comb [B,T,4,4] f32 -> new stream [B,T,4,H] f32.
pub fn hc_expand4(
    y: &Array,
    residual: &Array,
    post: &Array,
    comb: &Array,
    stream: &Stream,
) -> Option<Array> {
    let kernel = hc_expand_kernel().as_ref()?;
    let sh = y.shape();
    let b = sh[0];
    let t = sh[1];
    let h = *sh.last().unwrap();
    let n_rows = b * t;
    for a in [y, residual, post, comb] {
        if let Err(e) = a.eval_enqueue() {
            eprintln!("[hc-expand-decline] {e}");
            return None;
        }
    }
    let inputs = [
        KernelInput::I32(n_rows),
        KernelInput::I32(h),
        KernelInput::Tensor(&y.t),
        KernelInput::Tensor(&residual.t),
        KernelInput::Tensor(&post.t),
        KernelInput::Tensor(&comb.t),
    ];
    let outputs = [OutputArg {
        shape: vec![b, t, 4, h],
        dtype: Dtype::Float32,
    }];
    let out = match kernel.apply_with_inline_inputs(
        &inputs,
        &[],
        ((n_rows * h) as i32, 1, 1),
        (1, 1, 1),
        &outputs,
        stream,
    ) {
        Ok(o) => o,
        Err(e) => {
            eprintln!("[hc-expand-decline] {e}");
            return None;
        }
    };
    Some(out.into_iter().next()?)
}

const SWIGLU_SOURCE: &str = include_str!("../../shaders/glm53/swiglu.metal");
const SHARED_Q8_SWIGLU_SOURCE: &str =
    include_str!("../../shaders/glm53/shared_q8_swiglu_ds4.metal");

fn shared_q8_swiglu_ds4_kernel() -> &'static Option<MetalKernel> {
    static K: std::sync::OnceLock<Option<MetalKernel>> = std::sync::OnceLock::new();
    K.get_or_init(|| {
        MetalKernel::new_with_math_and_threadgroup_memory(
            "glm53_shared_q8_swiglu_ds4_schedule_v1",
            &[
                "in_dim",
                "out_dim",
                "group_size",
                "x",
                "gate_up_weights",
                "gate_up_scales",
                "gate_up_biases",
                "limit",
            ],
            &["gate", "up", "mid"],
            SHARED_Q8_SWIGLU_SOURCE,
            EXACT_HEADER,
            true,
            false,
            crate::runtime::Math::CompilerDefault,
            4 * 32 * std::mem::size_of::<f32>(),
        )
        // Full-write contract: Every valid row assigns gate, up, and mid outputs.
        .map(MetalKernel::with_full_write_outputs)
        .ok()
    })
}

/// DS4's one-row Q8 shared-expert gate/up and SwiGLU schedule, reading MLX's
/// affine-Q8 weight representation directly. This is the decode/prefill M=1
/// path; multirow inputs use the stock affine-QMM implementation.
pub fn shared_q8_swiglu_ds4(
    x: &Array,
    gate_up_weights: &Array,
    gate_up_scales: &Array,
    gate_up_biases: &Array,
    in_dim: usize,
    out_dim: usize,
    group_size: usize,
    limit: f32,
    stream: &Stream,
) -> Option<(Array, Array, Array)> {
    let kernel = shared_q8_swiglu_ds4_kernel().as_ref()?;
    if in_dim == 0
        || in_dim > i32::MAX as usize
        || out_dim == 0
        || out_dim > (i32::MAX as usize) / 2
        || x.shape() != [in_dim as i32]
        || x.dtype() != Dtype::Float32
        || in_dim % 64 != 0
        || group_size != 64
    {
        return None;
    }
    let weight_shape = [(2 * out_dim) as i32, (in_dim / 4) as i32];
    let metadata_shape = [(2 * out_dim) as i32, (in_dim / group_size) as i32];
    for (a, shape, dtype) in [
        (gate_up_weights, weight_shape.as_slice(), Dtype::Uint32),
        (gate_up_scales, metadata_shape.as_slice(), Dtype::Float32),
        (gate_up_biases, metadata_shape.as_slice(), Dtype::Float32),
    ] {
        if a.shape() != shape || a.dtype() != dtype {
            return None;
        }
    }
    for input in [x, gate_up_weights, gate_up_scales, gate_up_biases] {
        input.eval_enqueue().ok()?;
    }
    let inputs = [
        KernelInput::I32(in_dim as i32),
        KernelInput::I32(out_dim as i32),
        KernelInput::I32(group_size as i32),
        KernelInput::Tensor(&x.t),
        KernelInput::Tensor(&gate_up_weights.t),
        KernelInput::Tensor(&gate_up_scales.t),
        KernelInput::Tensor(&gate_up_biases.t),
        KernelInput::F32(limit),
    ];
    let outputs = [
        OutputArg {
            shape: vec![out_dim as i32],
            dtype: Dtype::Float32,
        },
        OutputArg {
            shape: vec![out_dim as i32],
            dtype: Dtype::Float32,
        },
        OutputArg {
            shape: vec![out_dim as i32],
            dtype: Dtype::Float32,
        },
    ];
    let result = kernel
        .apply_with_inline_inputs(
            &inputs,
            &[],
            ((((out_dim + 1) / 2) * 32) as i32, 4, 1),
            (32, 4, 1),
            &outputs,
            stream,
        )
        .ok()?;
    let mut it = result.into_iter();
    Some((it.next()?, it.next()?, it.next()?))
}

fn swiglu_kernel() -> &'static Option<MetalKernel> {
    static K: std::sync::OnceLock<Option<MetalKernel>> = std::sync::OnceLock::new();
    K.get_or_init(|| {
        MetalKernel::new_with_math(
            "glm53_swiglu_ds4_default_v3",
            &["n", "limit", "alpha", "gate", "up"],
            &["out"],
            SWIGLU_SOURCE,
            EXACT_HEADER,
            true,
            false,
            crate::runtime::Math::CompilerDefault,
        )
        // Full-write contract: The flat element grid assigns every output element.
        .map(MetalKernel::with_full_write_outputs)
        .ok()
    })
}

/// Fused clamped SwiGLU (ds4 glu.metal port): gate/up f32 [N] -> f32 [N],
/// one dispatch replaces the min/max/silu/multiply op chain.
pub fn swiglu_fused(gate: &Array, up: &Array, limit: f32, stream: &Stream) -> Option<Array> {
    let kernel = swiglu_kernel().as_ref()?;
    let n = gate.size();
    if up.size() != n {
        return None;
    }
    let n_a = Array::from_int(n as i32);
    let lim_a = Array::from_f32_as(limit, Dtype::Float32);
    let alpha_a = Array::from_f32_as(1.0, Dtype::Float32);
    for a in [gate, up] {
        if let Err(e) = a.eval_enqueue() {
            eprintln!("[swiglu-decline] {e}");
            return None;
        }
    }
    let inputs: [&Array; 5] = [&n_a, &lim_a, &alpha_a, gate, up];
    let outputs = [OutputArg {
        shape: gate.shape().to_vec(),
        dtype: Dtype::Float32,
    }];
    let all: Vec<&Array> = inputs.iter().map(|a| &**a).collect();
    let out = match kernel.apply(&all, &[], (n as i32, 1, 1), (256, 1, 1), &outputs, stream) {
        Ok(o) => o,
        Err(e) => {
            eprintln!("[swiglu-decline] {e}");
            return None;
        }
    };
    Some(out.into_iter().next()?)
}

const MOE_GATHER_SOURCE: &str = include_str!("../../shaders/glm53/moe_gather.metal");
const MOE_PAIR_SWIGLU_SOURCE: &str = include_str!("../../shaders/glm53/moe_pair_swiglu.metal");
const MOE_GATHER_ADDR_SOURCE: &str = include_str!("../../shaders/glm53/moe_gather_addr.metal");
const MOE_GATHER_ADDR_DS4_SOURCE: &str =
    include_str!("../../shaders/glm53/moe_gather_addr_ds4.metal");
const MOE_GATHER_SUM8_ADDR_DS4_SOURCE: &str =
    include_str!("../../shaders/glm53/moe_gather_sum8_addr_ds4.metal");
const MOE_GATHER_SUM8_ADDR_SOURCE: &str =
    include_str!("../../shaders/glm53/moe_gather_sum8_addr.metal");
const MOE_PAIR_SWIGLU_ADDR_SOURCE: &str =
    include_str!("../../shaders/glm53/moe_pair_swiglu_addr.metal");
const MOE_EXPERT_ROUTE_MAP_SOURCE: &str =
    include_str!("../../shaders/glm53/moe_expert_route_map.metal");
const MOE_MAPPED_RESOURCE_HELPER: &str =
    include_str!("../../shaders/glm53/moe_mapped_resource.metal");
const MOE_PAIR_SWIGLU_MAPPED_BODY: &str =
    include_str!("../../shaders/glm53/moe_pair_swiglu_mapped.metal");
const MOE_GATHER_SUM8_MAPPED_BODY: &str =
    include_str!("../../shaders/glm53/moe_gather_sum8_mapped.metal");
const MOE_SUM8_SOURCE: &str = include_str!("../../shaders/glm53/moe_sum8.metal");
const MOE_MAPPED_RESOURCE_INPUTS: [&str; 16] = [
    "resource_00",
    "resource_01",
    "resource_02",
    "resource_03",
    "resource_04",
    "resource_05",
    "resource_06",
    "resource_07",
    "resource_08",
    "resource_09",
    "resource_10",
    "resource_11",
    "resource_12",
    "resource_13",
    "resource_14",
    "resource_15",
];

/// Build the per-expert `(resource slot, byte offset)` table used by the
/// ordinary buffer-bound MoE kernels. Each expert contributes the same nine
/// fields as the address-table path: gate/up/down weight, scale, and bias.
/// The high byte of each Int64 entry stores the resource slot; the lower 56
/// bits store the byte offset into that encoder-bound MTLBuffer.
pub fn moe_mapped_resource_table(fields: &[&Array]) -> Option<(Array, Vec<Array>)> {
    if fields.is_empty() || fields.len() % 9 != 0 {
        return None;
    }

    let mut resource_slots = std::collections::HashMap::<usize, u8>::new();
    let mut resources = Vec::<Array>::new();
    let mut entries = Vec::<i64>::with_capacity(fields.len());
    for field in fields {
        let buffer = field.t.buffer();
        let address = field.gpu_address()?;
        let offset = address.checked_sub(buffer.gpu_address())?;
        let offset = usize::try_from(offset).ok()?;
        let byte_len = field.size().checked_mul(field.dtype().size_of())?;
        if offset.checked_add(byte_len)? > buffer.length() || offset >= (1usize << 56) {
            return None;
        }

        let id = field.backing_buffer_id();
        let slot = if let Some(&slot) = resource_slots.get(&id) {
            slot
        } else {
            if resources.len() == MOE_MAPPED_RESOURCE_INPUTS.len() || buffer.length() < 8 {
                return None;
            }
            let slot = u8::try_from(resources.len()).ok()?;
            let base =
                Array::from_buffer_offset(field.t.buffer().clone(), 0, &[8], Dtype::Uint8).ok()?;
            resource_slots.insert(id, slot);
            resources.push(base);
            slot
        };
        let packed = (u64::from(slot) << 56) | offset as u64;
        entries.push(packed as i64);
    }
    if resources.is_empty() {
        return None;
    }
    Some((
        Array::from_slice(&entries, &[entries.len() as i32]),
        resources,
    ))
}

/// The MLX quantized-matmul preamble chain — the SAME headers the jit qmv
/// dispatch composes (`jit/matmul.rs`), so the gathered kernel calls the very
/// same `qdot`/`load_vector` helpers as the certified per-expert qmv.
fn moe_gather_header() -> String {
    format!(
        "{}{}{}",
        crate::jit::MLX_GEMM_PREAMBLE,
        crate::jit::MLX_QUANTIZED_UTILS_PREAMBLE,
        crate::jit::MLX_QUANTIZED_PREAMBLE,
    )
}

fn moe_gather_kernel() -> &'static Option<MetalKernel> {
    static K: std::sync::OnceLock<Option<MetalKernel>> = std::sync::OnceLock::new();
    K.get_or_init(|| {
        let header = moe_gather_header();
        MetalKernel::new_with_math(
            "glm53_moe_gather_qmv_ds4_default_v4",
            &[
                "n_slots",
                "groups_per_slot",
                "out_rows",
                "in_vec_size",
                "x_per_slot",
                "w",
                "scales",
                "biases",
                "x",
                "selected",
            ],
            &["y"],
            MOE_GATHER_SOURCE,
            &header,
            true,
            false,
            crate::runtime::Math::CompilerDefault,
        )
        .ok()
    })
}

fn moe_pair_swiglu_kernel() -> &'static Option<MetalKernel> {
    static K: std::sync::OnceLock<Option<MetalKernel>> = std::sync::OnceLock::new();
    K.get_or_init(|| {
        let header = moe_gather_header();
        MetalKernel::new_with_math(
            "glm53_moe_pair_swiglu_ds4_default_v2",
            &[
                "n_slots",
                "groups_per_slot",
                "mid_dim",
                "in_vec_size",
                "slots_per_token",
                "limit",
                "w",
                "scales",
                "biases",
                "x",
                "selected",
                "weights",
            ],
            &["y"],
            MOE_PAIR_SWIGLU_SOURCE,
            &header,
            true,
            false,
            crate::runtime::Math::CompilerDefault,
        )
        .ok()
    })
}

fn moe_gather_addr_kernel() -> &'static Option<MetalKernel> {
    static K: std::sync::OnceLock<Option<MetalKernel>> = std::sync::OnceLock::new();
    K.get_or_init(|| {
        let header = moe_gather_header();
        MetalKernel::new_with_math(
            "glm53_moe_gather_qmv_addr_ds4_default_v3",
            &[
                "n_slots",
                "groups_per_slot",
                "out_rows",
                "in_vec_size",
                "x_per_slot",
                "n_experts",
                "address_table",
                "x",
                "selected",
            ],
            &["y"],
            MOE_GATHER_ADDR_SOURCE,
            &header,
            true,
            false,
            crate::runtime::Math::CompilerDefault,
        )
        .ok()
    })
}

fn moe_gather_addr_ds4_kernel() -> &'static Option<MetalKernel> {
    static K: std::sync::OnceLock<Option<MetalKernel>> = std::sync::OnceLock::new();
    K.get_or_init(|| {
        let header = moe_gather_header();
        MetalKernel::new_with_math(
            "glm53_moe_gather_qmv_addr_ds4_exact_v1",
            &[
                "n_slots",
                "out_rows",
                "in_vec_size",
                "x_per_slot",
                "n_experts",
                "address_table",
                "x",
                "selected",
            ],
            &["y"],
            MOE_GATHER_ADDR_DS4_SOURCE,
            &header,
            true,
            false,
            crate::runtime::Math::CompilerDefault,
        )
        .ok()
    })
}

fn moe_gather_sum8_addr_ds4_kernel() -> &'static Option<MetalKernel> {
    static K: std::sync::OnceLock<Option<MetalKernel>> = std::sync::OnceLock::new();
    K.get_or_init(|| {
        let header = moe_gather_header();
        MetalKernel::new_with_math(
            "glm53_moe_gather_sum8_addr_ds4_exact_v5",
            &[
                "n_tokens",
                "out_rows",
                "in_vec_size",
                "n_experts",
                "has_shared",
                "address_table",
                "x",
                "selected",
                "shared",
            ],
            &["y"],
            MOE_GATHER_SUM8_ADDR_DS4_SOURCE,
            &header,
            true,
            false,
            crate::runtime::Math::CompilerDefault,
        )
        // Full-write contract: Every row/token is assigned, including zero sums for invalid expert IDs.
        .map(MetalKernel::with_full_write_outputs)
        .ok()
    })
}

fn moe_gather_sum8_addr_kernel() -> &'static Option<MetalKernel> {
    static K: std::sync::OnceLock<Option<MetalKernel>> = std::sync::OnceLock::new();
    K.get_or_init(|| {
        let header = moe_gather_header();
        match MetalKernel::new_with_math(
            "glm53_moe_gather_sum8_addr_ds4_default_v9",
            &[
                "n_tokens",
                "groups_per_token",
                "out_rows",
                "in_vec_size",
                "slots_per_token",
                "n_experts",
                "address_table",
                "x",
                "selected",
                "has_shared",
                "shared",
                "has_shared_down",
                "shared_down_in_vec_size",
                "shared_down_x",
                "shared_down_w",
                "shared_down_scales",
                "shared_down_biases",
            ],
            &["out"],
            MOE_GATHER_SUM8_ADDR_SOURCE,
            &header,
            true,
            false,
            crate::runtime::Math::CompilerDefault,
        ) {
            Ok(kernel) => Some(kernel),
            Err(error) => {
                eprintln!("[glm53-moe-sum8] kernel compilation failed: {error}");
                None
            }
        }
    })
}

fn moe_pair_swiglu_addr_kernel() -> &'static Option<MetalKernel> {
    static K: std::sync::OnceLock<Option<MetalKernel>> = std::sync::OnceLock::new();
    K.get_or_init(|| {
        let header = moe_gather_header();
        MetalKernel::new_with_math(
            "glm53_moe_pair_swiglu_addr_ds4_default_v9",
            &[
                "n_slots",
                "groups_per_slot",
                "mid_dim",
                "in_vec_size",
                "slots_per_token",
                "n_experts",
                "limit",
                "address_table",
                "x",
                "selected",
                "weights",
            ],
            &["y"],
            MOE_PAIR_SWIGLU_ADDR_SOURCE,
            &header,
            true,
            false,
            crate::runtime::Math::CompilerDefault,
        )
        .ok()
    })
}

fn moe_expert_route_map_kernel() -> &'static Option<MetalKernel> {
    static K: std::sync::OnceLock<Option<MetalKernel>> = std::sync::OnceLock::new();
    K.get_or_init(|| {
        MetalKernel::new_with_math(
            "glm53_moe_expert_route_map_ds4_v1",
            &["n_tokens", "top_k", "n_experts", "tile_size", "selected"],
            &["route_ids", "counts", "work_count", "work_items"],
            MOE_EXPERT_ROUTE_MAP_SOURCE,
            "",
            true,
            false,
            crate::runtime::Math::CompilerDefault,
        )
        .ok()
    })
}

/// Build the expert-major route lists and compact tile work map used by the
/// grouped affine-Q2 MoE prefill. Route IDs remain flattened token/slot
/// indices, so grouped projections can write results back in the existing
/// token-major order. This is a mapping primitive; the production per-slot
/// MoE path remains unchanged until grouped projection parity is established.
pub fn moe_expert_route_map(
    selected: &Array,
    n_tokens: usize,
    top_k: usize,
    n_experts: usize,
    tile_size: usize,
    stream: &Stream,
) -> Option<(Array, Array, Array, Array)> {
    let kernel = moe_expert_route_map_kernel().as_ref()?;
    if selected.rank() != 1
        || selected.dtype() != Dtype::Int32
        || n_tokens == 0
        || top_k == 0
        || n_experts == 0
        || n_experts > 1024
        || tile_size == 0
        || n_tokens > i32::MAX as usize
        || top_k > i32::MAX as usize
        || tile_size > i32::MAX as usize
        || selected.dim(0) as usize != n_tokens.checked_mul(top_k)?
        || n_tokens.checked_mul(n_experts)? > i32::MAX as usize
    {
        return None;
    }
    selected.eval_enqueue().ok()?;

    // Router top-k selection produces unique expert IDs within each token,
    // which bounds each expert's route list to n_tokens entries.
    let route_id_count = n_tokens.checked_mul(n_experts)?;
    let max_work_items = selected.dim(0).checked_div(tile_size as i32)? as usize
        + usize::from(selected.dim(0) as usize % tile_size != 0)
        + n_experts;
    let work_item_values = max_work_items.checked_mul(2)?;
    if work_item_values > i32::MAX as usize {
        return None;
    }
    let inputs = [
        KernelInput::I32(n_tokens as i32),
        KernelInput::I32(top_k as i32),
        KernelInput::I32(n_experts as i32),
        KernelInput::I32(tile_size as i32),
        KernelInput::Tensor(&selected.t),
    ];
    let outputs = [
        OutputArg {
            shape: vec![route_id_count as i32],
            dtype: Dtype::Int32,
        },
        OutputArg {
            shape: vec![n_experts as i32],
            dtype: Dtype::Int32,
        },
        OutputArg {
            shape: vec![1],
            dtype: Dtype::Int32,
        },
        OutputArg {
            shape: vec![work_item_values as i32],
            dtype: Dtype::Int32,
        },
    ];
    let mut result = kernel
        .apply_with_inline_inputs(
            &inputs,
            &[],
            (n_experts as i32, 1, 1),
            (n_experts as i32, 1, 1),
            &outputs,
            stream,
        )
        .ok()?
        .into_iter();
    Some((
        result.next()?,
        result.next()?,
        result.next()?,
        result.next()?,
    ))
}

fn moe_pair_swiglu_mapped_kernel() -> &'static Option<MetalKernel> {
    static K: std::sync::OnceLock<Option<MetalKernel>> = std::sync::OnceLock::new();
    K.get_or_init(|| {
        let header = format!("{}\n{}", moe_gather_header(), MOE_MAPPED_RESOURCE_HELPER);
        let source = MOE_PAIR_SWIGLU_MAPPED_BODY;
        let mut inputs = vec![
            "n_slots",
            "groups_per_slot",
            "mid_dim",
            "in_vec_size",
            "slots_per_token",
            "n_experts",
            "limit",
            "address_table",
            "x",
            "selected",
            "weights",
        ];
        inputs.extend(MOE_MAPPED_RESOURCE_INPUTS.iter().copied());
        MetalKernel::new_with_math(
            "glm53_moe_pair_swiglu_mapped_ds4_exact_v2",
            &inputs,
            &["y"],
            source,
            &header,
            true,
            false,
            crate::runtime::Math::CompilerDefault,
        )
        .ok()
    })
}

fn moe_gather_sum8_mapped_kernel() -> &'static Option<MetalKernel> {
    static K: std::sync::OnceLock<Option<MetalKernel>> = std::sync::OnceLock::new();
    K.get_or_init(|| {
        let header = format!("{}\n{}", moe_gather_header(), MOE_MAPPED_RESOURCE_HELPER);
        let source = MOE_GATHER_SUM8_MAPPED_BODY;
        let mut inputs = vec![
            "n_tokens",
            "out_rows",
            "in_vec_size",
            "n_experts",
            "has_shared",
            "address_table",
            "x",
            "selected",
            "shared",
        ];
        inputs.extend(MOE_MAPPED_RESOURCE_INPUTS.iter().copied());
        MetalKernel::new_with_math(
            "glm53_moe_gather_sum8_mapped_ds4_exact_v2",
            &inputs,
            &["out"],
            source,
            &header,
            true,
            false,
            crate::runtime::Math::CompilerDefault,
        )
        .ok()
    })
}

fn moe_sum8_kernel() -> &'static Option<MetalKernel> {
    static K: std::sync::OnceLock<Option<MetalKernel>> = std::sync::OnceLock::new();
    K.get_or_init(|| {
        MetalKernel::new_with_math(
            "glm53_moe_sum8_ds4_default_v2",
            &["n_tokens", "width", "src"],
            &["out"],
            MOE_SUM8_SOURCE,
            EXACT_HEADER,
            true,
            false,
            crate::runtime::Math::CompilerDefault,
        )
        // Full-write contract: Each token group covers every output column.
        .map(MetalKernel::with_full_write_outputs)
        .ok()
    })
}

/// Fused gate/up projection through a compact MLX-expert GPU address table.
/// `address_table` stores nine Int64 addresses per expert; the first six are
/// the separate gate/up weight, scale, and bias buffers. `keepalive` holds the
/// underlying arrays through completion because address indirection is
/// invisible to Metal's normal resource tracker.
pub fn moe_pair_swiglu_addr(
    address_table: &Array,
    x: &Array,
    selected: &Array,
    weights: &Array,
    keepalive: &[&Array],
    mid_dim: usize,
    slots_per_token: usize,
    limit: f32,
    metadata_dtype: Dtype,
    residency_set: Option<&crate::runtime::ResidencySet>,
    stream: &Stream,
) -> Option<Array> {
    if address_table.rank() != 1
        || address_table.dtype() != Dtype::Int64
        || address_table.dim(0) <= 0
        || address_table.dim(0) % 9 != 0
        || !matches!(metadata_dtype, Dtype::Bfloat16 | Dtype::Float32)
        || selected.rank() != 1
        || selected.dtype() != Dtype::Int32
        || weights.rank() != 1
        || weights.dtype() != Dtype::Float32
        || x.rank() != 2
        || !matches!(x.dtype(), Dtype::Float32 | Dtype::Float16)
        || mid_dim == 0
        || slots_per_token == 0
    {
        return None;
    }
    let n_slots = selected.dim(0) as usize;
    let in_vec_size = x.dim(1) as usize;
    let n_experts = address_table.dim(0) as usize / 9;
    if n_slots == 0
        || n_slots % slots_per_token != 0
        || x.dim(0) != (n_slots / slots_per_token) as i32
        || weights.dim(0) as usize != n_slots
        || in_vec_size == 0
        || in_vec_size % 512 != 0
        || n_experts == 0
    {
        return None;
    }
    let kernel = moe_pair_swiglu_addr_kernel().as_ref()?;
    for a in [address_table, x, selected, weights] {
        a.eval_enqueue().ok()?;
    }
    // Four output rows per SIMD group, two SIMD groups per threadgroup.
    let groups_per_slot = mid_dim.div_ceil(8);
    let grid_y = i32::try_from(groups_per_slot.checked_mul(n_slots)?.checked_mul(2)?).ok()?;
    let inputs = [
        KernelInput::I32(n_slots as i32),
        KernelInput::I32(groups_per_slot as i32),
        KernelInput::I32(mid_dim as i32),
        KernelInput::I32(in_vec_size as i32),
        KernelInput::I32(slots_per_token as i32),
        KernelInput::I32(n_experts as i32),
        KernelInput::F32(limit),
        KernelInput::Tensor(&address_table.t),
        KernelInput::Tensor(&x.t),
        KernelInput::Tensor(&selected.t),
        KernelInput::Tensor(&weights.t),
    ];
    let outputs = [OutputArg {
        shape: vec![n_slots as i32, mid_dim as i32],
        dtype: Dtype::Float32,
    }];
    let result = if residency_set.is_some() {
        kernel.apply_with_inline_inputs_and_residency_set(
            &inputs,
            &[
                TemplateArg::Dtype("T", Dtype::Float32),
                TemplateArg::Dtype("X", x.dtype()),
                TemplateArg::Dtype("M", metadata_dtype),
            ],
            (32, grid_y, 1),
            (32, 2, 1),
            &outputs,
            keepalive,
            stream,
        )
    } else {
        kernel.apply_with_inline_inputs_and_keepalive(
            &inputs,
            &[
                TemplateArg::Dtype("T", Dtype::Float32),
                TemplateArg::Dtype("M", metadata_dtype),
            ],
            (32, grid_y, 1),
            (32, 2, 1),
            &outputs,
            keepalive,
            stream,
        )
    };
    result.ok()?.into_iter().next()
}

fn padded_moe_resource_inputs(resources: &[Array]) -> Option<Vec<Array>> {
    if resources.is_empty() || resources.len() > MOE_MAPPED_RESOURCE_INPUTS.len() {
        return None;
    }
    if resources
        .iter()
        .any(|resource| resource.dtype() != Dtype::Uint8 || resource.shape() != [8])
    {
        return None;
    }
    let mut inputs = resources.to_vec();
    let first = inputs.first()?.clone();
    inputs.resize(MOE_MAPPED_RESOURCE_INPUTS.len(), first);
    Some(inputs)
}

/// Fused gate/up SwiGLU with MLX expert buffers bound as ordinary Metal
/// resources. The table stores one packed resource slot and byte offset for
/// each of the nine expert fields. This mirrors ds4's normal encoder bindings
/// and avoids both raw GPU-address dereferences and selected-ID readback.
pub fn moe_pair_swiglu_mapped(
    resource_table: &Array,
    x: &Array,
    selected: &Array,
    weights: &Array,
    resources: &[Array],
    mid_dim: usize,
    slots_per_token: usize,
    limit: f32,
    metadata_dtype: Dtype,
    stream: &Stream,
) -> Option<Array> {
    let kernel = moe_pair_swiglu_mapped_kernel().as_ref()?;
    if resource_table.rank() != 1
        || resource_table.dtype() != Dtype::Int64
        || resource_table.dim(0) <= 0
        || resource_table.dim(0) % 9 != 0
        || !matches!(metadata_dtype, Dtype::Bfloat16 | Dtype::Float32)
        || selected.rank() != 1
        || selected.dtype() != Dtype::Int32
        || weights.rank() != 1
        || weights.dtype() != Dtype::Float32
        || x.rank() != 2
        || x.dtype() != Dtype::Float32
        || mid_dim == 0
        || slots_per_token == 0
    {
        return None;
    }
    let bound = padded_moe_resource_inputs(resources)?;
    let n_slots = selected.dim(0) as usize;
    let in_vec_size = x.dim(1) as usize;
    let n_experts = resource_table.dim(0) as usize / 9;
    if n_slots == 0
        || n_slots % slots_per_token != 0
        || x.dim(0) != (n_slots / slots_per_token) as i32
        || weights.dim(0) as usize != n_slots
        || in_vec_size == 0
        || in_vec_size % 512 != 0
        || n_experts == 0
    {
        return None;
    }

    let slots = Array::from_int(n_slots as i32);
    let groups_per_slot = Array::from_int(mid_dim.div_ceil(4) as i32);
    let mid = Array::from_int(mid_dim as i32);
    let k = Array::from_int(in_vec_size as i32);
    let spt = Array::from_int(slots_per_token as i32);
    let ne = Array::from_int(n_experts as i32);
    let lim = Array::from_f32_as(limit, Dtype::Float32);
    let mut inputs: Vec<&Array> = vec![
        &slots,
        &groups_per_slot,
        &mid,
        &k,
        &spt,
        &ne,
        &lim,
        resource_table,
        x,
        selected,
        weights,
    ];
    inputs.extend(bound.iter());
    for input in &inputs {
        input.eval_enqueue().ok()?;
    }
    let outputs = [OutputArg {
        shape: vec![n_slots as i32, mid_dim as i32],
        dtype: Dtype::Float32,
    }];
    let result = match kernel.apply(
        &inputs,
        &[
            TemplateArg::Dtype("T", Dtype::Float32),
            TemplateArg::Dtype("M", metadata_dtype),
        ],
        (32, (mid_dim.div_ceil(4) * n_slots * 4) as i32, 1),
        (32, 4, 1),
        &outputs,
        stream,
    ) {
        Ok(result) => result,
        Err(error) => {
            eprintln!("[glm53-moe-mapped] pair dispatch declined: {error}");
            return None;
        }
    };
    result.into_iter().next()
}

/// Per-route down projection through the same compact nine-address-per-expert
/// table. Output is `[n_slots, out_rows]` in F32.
pub fn moe_gather_qmv_addr(
    address_table: &Array,
    x: &Array,
    selected: &Array,
    keepalive: &[&Array],
    out_rows: usize,
    x_per_slot: bool,
    metadata_dtype: Dtype,
    residency_set: Option<&crate::runtime::ResidencySet>,
    stream: &Stream,
) -> Option<Array> {
    let kernel = moe_gather_addr_kernel().as_ref()?;
    if address_table.rank() != 1
        || address_table.dtype() != Dtype::Int64
        || address_table.dim(0) <= 0
        || address_table.dim(0) % 9 != 0
        || !matches!(metadata_dtype, Dtype::Bfloat16 | Dtype::Float32)
        || selected.rank() != 1
        || selected.dtype() != Dtype::Int32
        || x.rank() != 2
        || !matches!(x.dtype(), Dtype::Float32 | Dtype::Float16)
        || out_rows == 0
        || out_rows % 8 != 0
    {
        return None;
    }
    let n_slots = selected.dim(0) as usize;
    let in_vec_size = x.dim(1) as usize;
    let n_experts = address_table.dim(0) as usize / 9;
    if n_slots == 0
        || x.dim(0) as usize != if x_per_slot { n_slots } else { 1 }
        || in_vec_size == 0
        || in_vec_size % 512 != 0
        || n_experts == 0
    {
        return None;
    }
    for a in [address_table, x, selected] {
        a.eval_enqueue().ok()?;
    }
    let slots = Array::from_int(n_slots as i32);
    let groups_per_slot = Array::from_int(out_rows.div_ceil(8) as i32);
    let rows = Array::from_int(out_rows as i32);
    let k = Array::from_int(in_vec_size as i32);
    let xps = Array::from_int(i32::from(x_per_slot));
    let ne = Array::from_int(n_experts as i32);
    let inputs: [&Array; 9] = [
        &slots,
        &groups_per_slot,
        &rows,
        &k,
        &xps,
        &ne,
        address_table,
        x,
        selected,
    ];
    let outputs = [OutputArg {
        shape: vec![n_slots as i32, out_rows as i32],
        dtype: Dtype::Float32,
    }];
    let result = if residency_set.is_some() {
        kernel.apply_with_residency_set(
            &inputs,
            &[
                TemplateArg::Dtype("T", Dtype::Float32),
                TemplateArg::Dtype("X", x.dtype()),
                TemplateArg::Dtype("M", metadata_dtype),
            ],
            (32, (out_rows.div_ceil(8) * n_slots * 2) as i32, 1),
            (32, 2, 1),
            &outputs,
            keepalive,
            stream,
        )
    } else {
        kernel.apply_with_keepalive(
            &inputs,
            &[
                TemplateArg::Dtype("T", Dtype::Float32),
                TemplateArg::Dtype("X", x.dtype()),
                TemplateArg::Dtype("M", metadata_dtype),
            ],
            (32, (out_rows.div_ceil(8) * n_slots * 2) as i32, 1),
            (32, 2, 1),
            &outputs,
            keepalive,
            stream,
        )
    };
    result.ok()?.into_iter().next()
}

/// Q2 affine per-route down projection with DS4's exact one-token arithmetic
/// schedule. Every output/route pair uses one SIMD group, whose lanes visit
/// `k = lane + 32n`, FMA-accumulate in order, then perform one SIMD reduction.
pub fn moe_gather_qmv_addr_ds4(
    address_table: &Array,
    x: &Array,
    selected: &Array,
    keepalive: &[&Array],
    out_rows: usize,
    x_per_slot: bool,
    metadata_dtype: Dtype,
    residency_set: Option<&crate::runtime::ResidencySet>,
    stream: &Stream,
) -> Option<Array> {
    let kernel = moe_gather_addr_ds4_kernel().as_ref()?;
    if address_table.rank() != 1
        || address_table.dtype() != Dtype::Int64
        || address_table.dim(0) <= 0
        || address_table.dim(0) % 9 != 0
        || !matches!(metadata_dtype, Dtype::Bfloat16 | Dtype::Float32)
        || selected.rank() != 1
        || selected.dtype() != Dtype::Int32
        || x.rank() != 2
        || x.dtype() != Dtype::Float32
        || out_rows == 0
        || out_rows > i32::MAX as usize
    {
        return None;
    }
    let n_slots = selected.dim(0) as usize;
    let in_vec_size = x.dim(1) as usize;
    let n_experts = address_table.dim(0) as usize / 9;
    if n_slots == 0
        || n_slots > i32::MAX as usize
        || x.dim(0) as usize != if x_per_slot { n_slots } else { 1 }
        || in_vec_size == 0
        || in_vec_size % 128 != 0
        || in_vec_size > i32::MAX as usize
        || n_experts == 0
        || n_experts > i32::MAX as usize
    {
        return None;
    }
    for input in [address_table, x, selected] {
        input.eval_enqueue().ok()?;
    }
    let slots = Array::from_int(n_slots as i32);
    let rows = Array::from_int(out_rows as i32);
    let k = Array::from_int(in_vec_size as i32);
    let xps = Array::from_int(i32::from(x_per_slot));
    let ne = Array::from_int(n_experts as i32);
    let inputs: [&Array; 8] = [&slots, &rows, &k, &xps, &ne, address_table, x, selected];
    let outputs = [OutputArg {
        shape: vec![n_slots as i32, out_rows as i32],
        dtype: Dtype::Float32,
    }];
    let grid_x = i32::try_from(out_rows.checked_mul(32)?).ok()?;
    let grid_y = i32::try_from(n_slots).ok()?;
    let template = [TemplateArg::Dtype("M", metadata_dtype)];
    let result = if residency_set.is_some() {
        kernel.apply_with_residency_set(
            &inputs,
            &template,
            (grid_x, grid_y, 1),
            (32, 1, 1),
            &outputs,
            keepalive,
            stream,
        )
    } else {
        kernel.apply_with_keepalive(
            &inputs,
            &template,
            (grid_x, grid_y, 1),
            (32, 1, 1),
            &outputs,
            keepalive,
            stream,
        )
    };
    result.ok()?.into_iter().next()
}

/// DS4-exact Q2 down projection for eight routed experts in one dispatch.
/// Each SIMD group computes one expert's DS4 affine matvec for four output
/// rows; the group leader adds the eight reduced values in route order.
pub fn moe_gather_qmv_sum8_addr_ds4(
    address_table: &Array,
    x: &Array,
    selected: &Array,
    keepalive: &[&Array],
    shared: Option<&Array>,
    out_rows: usize,
    metadata_dtype: Dtype,
    residency_set: Option<&crate::runtime::ResidencySet>,
    stream: &Stream,
) -> Option<Array> {
    let kernel = moe_gather_sum8_addr_ds4_kernel().as_ref()?;
    if address_table.rank() != 1
        || address_table.dtype() != Dtype::Int64
        || address_table.dim(0) <= 0
        || address_table.dim(0) % 9 != 0
        || !matches!(metadata_dtype, Dtype::Bfloat16 | Dtype::Float32)
        || selected.rank() != 1
        || selected.dtype() != Dtype::Int32
        || selected.dim(0) <= 0
        || selected.dim(0) % 8 != 0
        || x.rank() != 2
        || x.dtype() != Dtype::Float32
        || x.dim(0) != selected.dim(0)
        || out_rows == 0
        || out_rows > i32::MAX as usize
    {
        return None;
    }
    let n_slots = selected.dim(0) as usize;
    let n_tokens = n_slots / 8;
    let in_vec_size = x.dim(1) as usize;
    let n_experts = address_table.dim(0) as usize / 9;
    if in_vec_size == 0
        || in_vec_size % 128 != 0
        || in_vec_size > i32::MAX as usize
        || n_experts == 0
        || n_experts > i32::MAX as usize
        || n_tokens > i32::MAX as usize
    {
        return None;
    }
    for input in [address_table, x, selected] {
        input.eval_enqueue().ok()?;
    }
    let (has_shared, shared_values) = if let Some(shared) = shared {
        if shared.rank() != 2
            || shared.dtype() != Dtype::Float32
            || shared.dim(0) != n_tokens as i32
            || shared.dim(1) != out_rows as i32
        {
            return None;
        }
        (1, shared.contiguous().ok()?)
    } else {
        // The shader never reads this binding when has_shared is zero.
        (0, x.clone())
    };
    let inputs = [
        KernelInput::I32(n_tokens as i32),
        KernelInput::I32(out_rows as i32),
        KernelInput::I32(in_vec_size as i32),
        KernelInput::I32(n_experts as i32),
        KernelInput::I32(has_shared),
        KernelInput::Tensor(&address_table.t),
        KernelInput::Tensor(&x.t),
        KernelInput::Tensor(&selected.t),
        KernelInput::Tensor(&shared_values.t),
    ];
    let outputs = [OutputArg {
        shape: vec![n_tokens as i32, out_rows as i32],
        dtype: Dtype::Float32,
    }];
    let grid_x = i32::try_from(out_rows.div_ceil(4).checked_mul(32)?).ok()?;
    let grid_y = i32::try_from(n_tokens.checked_mul(8)?).ok()?;
    let template = [TemplateArg::Dtype("M", metadata_dtype)];
    let result = if residency_set.is_some() {
        kernel.apply_with_inline_inputs_and_residency_set(
            &inputs,
            &template,
            (grid_x, grid_y, 1),
            (32, 8, 1),
            &outputs,
            keepalive,
            stream,
        )
    } else {
        kernel.apply_with_inline_inputs_and_keepalive(
            &inputs,
            &template,
            (grid_x, grid_y, 1),
            (32, 8, 1),
            &outputs,
            keepalive,
            stream,
        )
    };
    result.ok()?.into_iter().next()
}

/// Selected-expert QMV for eight routed rows per token. This follows ds4's
/// down-projection schedule: each output tile visits expert slots in order,
/// keeps their F32 reductions local, and writes one `[token, out_rows]` row.
/// `x` contains one already route-weighted input row per selected expert.
pub fn moe_gather_qmv_sum8_addr(
    address_table: &Array,
    x: &Array,
    selected: &Array,
    keepalive: &[&Array],
    out_rows: usize,
    metadata_dtype: Dtype,
    residency_set: Option<&crate::runtime::ResidencySet>,
    stream: &Stream,
) -> Option<Array> {
    moe_gather_qmv_sum8_addr_impl(
        address_table,
        x,
        selected,
        keepalive,
        None,
        None,
        out_rows,
        metadata_dtype,
        residency_set,
        stream,
    )
}

/// Selected-expert sum8 down projection with the shared-expert output added
/// after the ordered routed sum in the same dispatch.
pub fn moe_gather_qmv_sum8_addr_add_shared(
    address_table: &Array,
    x: &Array,
    selected: &Array,
    keepalive: &[&Array],
    shared: &Array,
    out_rows: usize,
    metadata_dtype: Dtype,
    residency_set: Option<&crate::runtime::ResidencySet>,
    stream: &Stream,
) -> Option<Array> {
    moe_gather_qmv_sum8_addr_impl(
        address_table,
        x,
        selected,
        keepalive,
        Some(shared),
        None,
        out_rows,
        metadata_dtype,
        residency_set,
        stream,
    )
}

/// Selected-expert sum8 down projection with the shared Q8/group-64 down
/// projection fused after the routed sum. The shared activation is one row per
/// token and uses the stock MLX packed affine layout.
pub fn moe_gather_qmv_sum8_addr_add_shared_down(
    address_table: &Array,
    x: &Array,
    selected: &Array,
    keepalive: &[&Array],
    shared_x: &Array,
    shared_w: &Array,
    shared_scales: &Array,
    shared_biases: &Array,
    out_rows: usize,
    metadata_dtype: Dtype,
    residency_set: Option<&crate::runtime::ResidencySet>,
    stream: &Stream,
) -> Option<Array> {
    moe_gather_qmv_sum8_addr_impl(
        address_table,
        x,
        selected,
        keepalive,
        None,
        Some((shared_x, shared_w, shared_scales, shared_biases)),
        out_rows,
        metadata_dtype,
        residency_set,
        stream,
    )
}

fn moe_gather_qmv_sum8_addr_impl(
    address_table: &Array,
    x: &Array,
    selected: &Array,
    keepalive: &[&Array],
    shared: Option<&Array>,
    shared_down: Option<(&Array, &Array, &Array, &Array)>,
    out_rows: usize,
    metadata_dtype: Dtype,
    residency_set: Option<&crate::runtime::ResidencySet>,
    stream: &Stream,
) -> Option<Array> {
    let kernel = moe_gather_sum8_addr_kernel().as_ref()?;
    if address_table.rank() != 1
        || address_table.dtype() != Dtype::Int64
        || address_table.dim(0) <= 0
        || address_table.dim(0) % 9 != 0
        || !matches!(metadata_dtype, Dtype::Bfloat16 | Dtype::Float32)
        || selected.rank() != 1
        || selected.dtype() != Dtype::Int32
        || x.rank() != 2
        || !matches!(x.dtype(), Dtype::Float32 | Dtype::Float16)
        || out_rows == 0
        || out_rows % 8 != 0
    {
        return None;
    }
    let n_slots = selected.dim(0) as usize;
    let in_vec_size = x.dim(1) as usize;
    let n_experts = address_table.dim(0) as usize / 9;
    if n_slots == 0
        || n_slots % 8 != 0
        || x.dim(0) as usize != n_slots
        || in_vec_size == 0
        || in_vec_size % 512 != 0
        || n_experts == 0
    {
        return None;
    }
    for a in [address_table, x, selected] {
        a.eval_enqueue().ok()?;
    }
    let n_tokens = n_slots / 8;
    let shared_input = if let Some(shared) = shared {
        if shared.rank() != 2
            || shared.dtype() != Dtype::Float32
            || shared.dim(0) != n_tokens as i32
            || shared.dim(1) != out_rows as i32
        {
            return None;
        }
        shared.contiguous().ok()?
    } else {
        Array::from_slice(&[0.0f32; 8], &[8])
    };
    let (
        shared_down_in_vec_size,
        shared_down_x,
        shared_down_w,
        shared_down_scales,
        shared_down_biases,
    ) = if let Some((shared_x, shared_w, shared_scales, shared_biases)) = shared_down {
        let shared_in = shared_x.dim(-1) as usize;
        if shared_x.rank() != 2
            || shared_x.dtype() != Dtype::Float32
            || shared_x.dim(0) != n_tokens as i32
            || shared_in == 0
            || shared_in % 256 != 0
            || shared_w.rank() != 2
            || shared_w.dtype() != Dtype::Uint32
            || shared_w.dim(0) != out_rows as i32
            || shared_w.dim(1) as usize != shared_in / 4
            || shared_scales.rank() != 2
            || shared_scales.dtype() != Dtype::Float32
            || shared_scales.dim(0) != out_rows as i32
            || shared_scales.dim(1) as usize != shared_in / 64
            || shared_biases.shape() != shared_scales.shape()
            || shared_biases.dtype() != Dtype::Float32
        {
            return None;
        }
        (
            shared_in,
            shared_x.contiguous().ok()?,
            shared_w.contiguous().ok()?,
            shared_scales.contiguous().ok()?,
            shared_biases.contiguous().ok()?,
        )
    } else {
        let placeholder = Array::from_slice(&[0.0f32; 8], &[8]);
        (
            0,
            placeholder.clone(),
            placeholder.clone(),
            placeholder.clone(),
            placeholder,
        )
    };
    let inputs = [
        KernelInput::I32(n_tokens as i32),
        KernelInput::I32(out_rows.div_ceil(4) as i32),
        KernelInput::I32(out_rows as i32),
        KernelInput::I32(in_vec_size as i32),
        KernelInput::I32(8),
        KernelInput::I32(n_experts as i32),
        KernelInput::Tensor(&address_table.t),
        KernelInput::Tensor(&x.t),
        KernelInput::Tensor(&selected.t),
        KernelInput::I32(i32::from(shared.is_some())),
        KernelInput::Tensor(&shared_input.t),
        KernelInput::I32(i32::from(shared_down.is_some())),
        KernelInput::I32(shared_down_in_vec_size as i32),
        KernelInput::Tensor(&shared_down_x.t),
        KernelInput::Tensor(&shared_down_w.t),
        KernelInput::Tensor(&shared_down_scales.t),
        KernelInput::Tensor(&shared_down_biases.t),
    ];
    for a in [
        &shared_input,
        &shared_down_x,
        &shared_down_w,
        &shared_down_scales,
        &shared_down_biases,
    ] {
        a.eval_enqueue().ok()?;
    }
    let outputs = [OutputArg {
        shape: vec![n_tokens as i32, out_rows as i32],
        dtype: Dtype::Float32,
    }];
    let result = if residency_set.is_some() {
        kernel.apply_with_inline_inputs_and_residency_set(
            &inputs,
            &[
                TemplateArg::Dtype("T", Dtype::Float32),
                TemplateArg::Dtype("X", x.dtype()),
                TemplateArg::Dtype("M", metadata_dtype),
            ],
            (32, (out_rows.div_ceil(4) * n_tokens * 4) as i32, 1),
            (32, 4, 1),
            &outputs,
            keepalive,
            stream,
        )
    } else {
        kernel.apply_with_inline_inputs_and_keepalive(
            &inputs,
            &[
                TemplateArg::Dtype("T", Dtype::Float32),
                TemplateArg::Dtype("X", x.dtype()),
                TemplateArg::Dtype("M", metadata_dtype),
            ],
            (32, (out_rows.div_ceil(4) * n_tokens * 4) as i32, 1),
            (32, 4, 1),
            &outputs,
            keepalive,
            stream,
        )
    };
    match result {
        Ok(outputs) => outputs.into_iter().next(),
        Err(error) => {
            eprintln!("[glm53-moe-sum8] dispatch failed: {error}");
            None
        }
    }
}

/// Selected-expert sum8 down projection with each MLX backing buffer supplied
/// as an ordinary encoder input, matching ds4's normal resource binding.
pub fn moe_gather_qmv_sum8_mapped(
    resource_table: &Array,
    x: &Array,
    selected: &Array,
    resources: &[Array],
    out_rows: usize,
    metadata_dtype: Dtype,
    stream: &Stream,
) -> Option<Array> {
    let kernel = moe_gather_sum8_mapped_kernel().as_ref()?;
    if resource_table.rank() != 1
        || resource_table.dtype() != Dtype::Int64
        || resource_table.dim(0) <= 0
        || resource_table.dim(0) % 9 != 0
        || !matches!(metadata_dtype, Dtype::Bfloat16 | Dtype::Float32)
        || selected.rank() != 1
        || selected.dtype() != Dtype::Int32
        || x.rank() != 2
        || x.dtype() != Dtype::Float32
        || out_rows == 0
        || out_rows > i32::MAX as usize
    {
        return None;
    }
    let bound = padded_moe_resource_inputs(resources)?;
    let n_slots = selected.dim(0) as usize;
    let in_vec_size = x.dim(1) as usize;
    let n_experts = resource_table.dim(0) as usize / 9;
    if n_slots == 0
        || n_slots % 8 != 0
        || x.dim(0) as usize != n_slots
        || in_vec_size == 0
        || in_vec_size % 128 != 0
        || in_vec_size > i32::MAX as usize
        || n_experts == 0
        || n_experts > i32::MAX as usize
    {
        return None;
    }

    let n_tokens = n_slots / 8;
    let n_tokens_a = Array::from_int(n_tokens as i32);
    let rows = Array::from_int(out_rows as i32);
    let k = Array::from_int(in_vec_size as i32);
    let ne = Array::from_int(n_experts as i32);
    let has_shared = Array::from_int(0);
    let shared = Array::from_slice(&[0.0f32; 8], &[8]);
    let mut inputs: Vec<&Array> = vec![
        &n_tokens_a,
        &rows,
        &k,
        &ne,
        &has_shared,
        resource_table,
        x,
        selected,
        &shared,
    ];
    inputs.extend(bound.iter());
    for input in &inputs {
        input.eval_enqueue().ok()?;
    }
    let outputs = [OutputArg {
        shape: vec![n_tokens as i32, out_rows as i32],
        dtype: Dtype::Float32,
    }];
    let grid_x = i32::try_from(out_rows.div_ceil(4).checked_mul(32)?).ok()?;
    let grid_y = i32::try_from(n_tokens.checked_mul(8)?).ok()?;
    let result = match kernel.apply(
        &inputs,
        &[
            TemplateArg::Dtype("T", Dtype::Float32),
            TemplateArg::Dtype("M", metadata_dtype),
        ],
        (grid_x, grid_y, 1),
        (32, 8, 1),
        &outputs,
        stream,
    ) {
        Ok(result) => result,
        Err(error) => {
            eprintln!("[glm53-moe-mapped] down sum8 dispatch declined: {error}");
            return None;
        }
    };
    result.into_iter().next()
}

/// Reduce eight ordered routed-expert rows per token into one F32 row. This
/// follows ds4's `kernel_dsv4_moe_sum8_f32` and replaces seven separate tensor
/// additions after the down projections.
pub fn moe_sum8(experts: &Array, n_tokens: usize, width: usize, stream: &Stream) -> Option<Array> {
    let kernel = moe_sum8_kernel().as_ref()?;
    if experts.rank() != 2
        || experts.dtype() != Dtype::Float32
        || n_tokens == 0
        || width == 0
        || n_tokens > i32::MAX as usize
        || width > i32::MAX as usize
        || experts.dim(0) as usize != n_tokens.checked_mul(8)?
        || experts.dim(1) as usize != width
    {
        return None;
    }
    let grid_x = i32::try_from(n_tokens.checked_mul(256)?).ok()?;
    experts.eval_enqueue().ok()?;
    let n_tokens_a = Array::from_int(n_tokens as i32);
    let width_a = Array::from_int(width as i32);
    let inputs: [&Array; 3] = [&n_tokens_a, &width_a, experts];
    let outputs = [OutputArg {
        shape: vec![n_tokens as i32, width as i32],
        dtype: Dtype::Float32,
    }];
    kernel
        .apply(&inputs, &[], (grid_x, 1, 1), (256, 1, 1), &outputs, stream)
        .ok()?
        .into_iter()
        .next()
}

/// Gathered quantized matvec over stacked per-expert pools. `selected` is
/// read on the GPU, and `x_per_slot` controls whether each route uses the
/// shared input row or its corresponding route row. GLM-5.3 uses F32 for both
/// input and quantization metadata. Returns `None` when declined.
pub fn moe_gather_qmv(
    pools_w: &Array,
    pools_s: &Array,
    pools_b: &Array,
    x: &Array,
    selected: &Array,
    out_rows: usize,
    x_per_slot: bool,
    dtype: Dtype,
    stream: &Stream,
) -> Option<Array> {
    let Some(kernel) = moe_gather_kernel().as_ref() else {
        eprintln!("[moe-gather-decline] Metal gathered-QMV kernel failed to initialize");
        return None;
    };
    // Materialize every input before dispatch: in the lazy zero-flush decode
    // step nothing else forces these, and the kernel reads raw buffers.
    for a in [pools_w, pools_s, pools_b, x, selected] {
        if let Err(e) = a.eval_enqueue() {
            eprintln!("[moe-gather-decline] {e}");
            return None;
        }
    }
    let n_slots = selected.dim(0) as usize;
    let k = x.dim(x.rank() as i32 - 1) as usize;
    if n_slots == 0 || out_rows == 0 || out_rows % 8 != 0 || k == 0 || k % 512 != 0 {
        return None;
    }
    let template = [TemplateArg::Dtype("T", dtype)];
    let na = Array::from_int(n_slots as i32);
    let gps = out_rows / 8;
    let gps_a = Array::from_int(gps as i32);
    let orows = Array::from_int(out_rows as i32);
    let k_a = Array::from_int(k as i32);
    let xps = Array::from_int(if x_per_slot { 1 } else { 0 });
    let inputs: [&Array; 10] = [
        &na, &gps_a, &orows, &k_a, &xps, pools_w, pools_s, pools_b, x, selected,
    ];
    let outputs = [OutputArg {
        shape: vec![n_slots as i32, out_rows as i32],
        dtype,
    }];
    let all: Vec<&Array> = inputs.iter().map(|a| &**a).collect();
    let out = match kernel.apply(
        &all,
        &template,
        (32, (gps * n_slots * 2) as i32, 1),
        (32, 2, 1),
        &outputs,
        stream,
    ) {
        Ok(o) => o,
        Err(e) => {
            eprintln!("[moe-gather-decline] {e}");
            return None;
        }
    };
    Some(out.into_iter().next()?)
}

/// Gathered gate/up projection fused with ds4's clamped SwiGLU and route
/// weighting. MLX packed Q2 weights remain in their native layout; `selected`
/// and `weights` are read per slot on the GPU. Returns `[n_slots, mid_dim]`.
pub fn moe_pair_swiglu(
    pools_w: &Array,
    pools_s: &Array,
    pools_b: &Array,
    x: &Array,
    selected: &Array,
    weights: &Array,
    mid_dim: usize,
    slots_per_token: usize,
    limit: f32,
    stream: &Stream,
) -> Option<Array> {
    let kernel = moe_pair_swiglu_kernel().as_ref()?;
    if selected.rank() != 1
        || weights.rank() != 1
        || x.rank() != 2
        || pools_w.rank() != 3
        || pools_s.rank() != 3
        || pools_b.rank() != 3
        || mid_dim == 0
        || slots_per_token == 0
        || pools_w.dim(1) != (2 * mid_dim) as i32
        || pools_s.dtype() != Dtype::Float32
        || pools_b.dtype() != Dtype::Float32
        || x.dtype() != Dtype::Float32
        || weights.dtype() != Dtype::Float32
        || selected.dtype() != Dtype::Int32
    {
        return None;
    }
    let n_slots = selected.dim(0) as usize;
    let in_vec_size = x.dim(1) as usize;
    if n_slots == 0
        || n_slots % slots_per_token != 0
        || x.dim(0) != (n_slots / slots_per_token) as i32
        || in_vec_size == 0
        || in_vec_size % 512 != 0
        || pools_s.dim(1) != (2 * mid_dim) as i32
        || pools_s.dim(2) != (in_vec_size / 128) as i32
    {
        return None;
    }
    for a in [pools_w, pools_s, pools_b, x, selected, weights] {
        if a.eval_enqueue().is_err() {
            return None;
        }
    }
    let slots = Array::from_int(n_slots as i32);
    let groups_per_slot = Array::from_int(mid_dim.div_ceil(8) as i32);
    let mid = Array::from_int(mid_dim as i32);
    let k = Array::from_int(in_vec_size as i32);
    let spt = Array::from_int(slots_per_token as i32);
    let lim = Array::from_f32_as(limit, Dtype::Float32);
    let inputs: [&Array; 12] = [
        &slots,
        &groups_per_slot,
        &mid,
        &k,
        &spt,
        &lim,
        pools_w,
        pools_s,
        pools_b,
        x,
        selected,
        weights,
    ];
    let outputs = [OutputArg {
        shape: vec![n_slots as i32, mid_dim as i32],
        dtype: Dtype::Float32,
    }];
    let all: Vec<&Array> = inputs.iter().map(|a| &**a).collect();
    let out = kernel
        .apply(
            &all,
            &[TemplateArg::Dtype("T", Dtype::Float32)],
            (32, (mid_dim.div_ceil(8) * n_slots * 2) as i32, 1),
            (32, 2, 1),
            &outputs,
            stream,
        )
        .ok()?;
    out.into_iter().next()
}

const ROUTER_SELECT_SOURCE: &str = include_str!("../../shaders/glm53/router_select.metal");
const ROUTER_MV_SOURCE: &str = include_str!("../../shaders/glm53/router_mv.metal");
const ROUTER_MV_SMALL_SOURCE: &str = include_str!("../../shaders/glm53/router_mv_small.metal");

fn router_mv_kernel() -> &'static Option<MetalKernel> {
    static K: std::sync::OnceLock<Option<MetalKernel>> = std::sync::OnceLock::new();
    K.get_or_init(|| {
        MetalKernel::new_with_math_and_threadgroup_memory(
            "glm53_router_mv_ds4_f32_v1",
            &["out_dim", "in_dim", "n_tokens", "weights", "x"],
            &["logits"],
            ROUTER_MV_SOURCE,
            EXACT_HEADER,
            true,
            false,
            crate::runtime::Math::CompilerDefault,
            32 * std::mem::size_of::<f32>(),
        )
        // Full-write contract: Every output row/token receives its reduced value.
        .map(MetalKernel::with_full_write_outputs)
        .ok()
    })
}

fn router_mv_small_kernel() -> &'static Option<MetalKernel> {
    static K: std::sync::OnceLock<Option<MetalKernel>> = std::sync::OnceLock::new();
    K.get_or_init(|| {
        MetalKernel::new_with_math(
            "glm53_router_mv_ext_ds4_f32_v1",
            &["out_dim", "in_dim", "n_tokens", "weights", "x"],
            &["logits"],
            ROUTER_MV_SMALL_SOURCE,
            EXACT_HEADER,
            true,
            false,
            crate::runtime::Math::CompilerDefault,
        )
        // Full-write contract: The admitted row groups cover every output row/token.
        .map(MetalKernel::with_full_write_outputs)
        .ok()
    })
}

fn router_matvec_f32_small(
    weights: &Array,
    x: &Array,
    n_tokens: usize,
    stream: &Stream,
) -> Option<Array> {
    let kernel = router_mv_small_kernel().as_ref()?;
    let (out_dim, in_dim) = (weights.dim(0) as usize, weights.dim(1) as usize);
    if !(2..=8).contains(&n_tokens)
        || in_dim % 128 != 0
        || x.shape() != [n_tokens as i32, in_dim as i32]
    {
        return None;
    }
    let (r1ptg, nxpsg) = match n_tokens {
        2 => (2, 16),
        3 | 6 => (3, 8),
        4 | 7 | 8 => (4, 8),
        5 => (5, 8),
        _ => return None,
    };
    let nsg = 2usize;
    let nypsg = 32usize / nxpsg;
    let rows_per_group = nypsg * nsg;
    let group_rows = out_dim.div_ceil(rows_per_group);
    let token_groups = n_tokens.div_ceil(r1ptg);
    let out_dim_arg = Array::from_int(out_dim as i32);
    let in_dim_arg = Array::from_int(in_dim as i32);
    let n_tokens_arg = Array::from_int(n_tokens as i32);
    let weights = weights.contiguous().ok()?;
    let x = x.contiguous().ok()?;
    for array in [&weights, &x] {
        array.eval_enqueue().ok()?;
    }
    let inputs: [&Array; 5] = [&out_dim_arg, &in_dim_arg, &n_tokens_arg, &weights, &x];
    let outputs = [OutputArg {
        shape: vec![n_tokens as i32, out_dim as i32],
        dtype: Dtype::Float32,
    }];
    kernel
        .apply(
            &inputs,
            &[
                TemplateArg::Int("R1PTG", r1ptg as i32),
                TemplateArg::Int("NXPSG", nxpsg as i32),
            ],
            (
                group_rows.checked_mul(32)? as i32,
                token_groups.checked_mul(nsg)? as i32,
                1,
            ),
            (32, nsg as i32, 1),
            &outputs,
            stream,
        )
        .ok()?
        .into_iter()
        .next()
}

/// DS4's F32 dense matvec lanes used for GLM router projection.
/// `weights` is row-major `[out, in]`; `x` is `[n_tokens, in]`. DS4 uses the
/// 1-row path for decode, a specialized multi-token lane for 2–8 rows, and
/// one F32 matvec per token above eight rows.
pub fn router_matvec_f32(weights: &Array, x: &Array, stream: &Stream) -> Option<Array> {
    if weights.dtype() != Dtype::Float32
        || x.dtype() != Dtype::Float32
        || weights.rank() != 2
        || x.rank() != 2
    {
        return None;
    }
    let out_dim = weights.dim(0) as usize;
    let in_dim = weights.dim(1) as usize;
    let n_tokens = x.dim(0) as usize;
    if out_dim == 0 || in_dim == 0 || in_dim % 128 != 0 || n_tokens == 0 {
        return None;
    }
    if x.dim(1) as usize != in_dim {
        return None;
    }
    if (2..=8).contains(&n_tokens) {
        return router_matvec_f32_small(weights, x, n_tokens, stream);
    }
    let kernel = router_mv_kernel().as_ref()?;
    for array in [weights, x] {
        if let Err(error) = array.eval_enqueue() {
            eprintln!("[glm53-router-mv-decline] {error}");
            return None;
        }
    }
    let weights = weights.contiguous().ok()?;
    let x = x.contiguous().ok()?;
    let out_dim_arg = Array::from_int(out_dim as i32);
    let in_dim_arg = Array::from_int(in_dim as i32);
    let n_tokens_arg = Array::from_int(n_tokens as i32);
    let inputs: [&Array; 5] = [&out_dim_arg, &in_dim_arg, &n_tokens_arg, &weights, &x];
    let outputs = [OutputArg {
        shape: vec![n_tokens as i32, out_dim as i32],
        dtype: Dtype::Float32,
    }];
    kernel
        .apply(
            &inputs,
            &[],
            // MetalKernel::apply uses dispatch_threads, unlike DS4's
            // dispatchThreadgroups call. Scale the grid by the group shape so
            // each output row/token receives one full 32x8 group.
            (
                out_dim.checked_mul(32)? as i32,
                n_tokens.checked_mul(8)? as i32,
                1,
            ),
            (32, 8, 1),
            &outputs,
            stream,
        )
        .ok()?
        .into_iter()
        .next()
}

fn router_select_kernel() -> &'static Option<MetalKernel> {
    static K: std::sync::OnceLock<Option<MetalKernel>> = std::sync::OnceLock::new();
    K.get_or_init(|| {
        MetalKernel::new_with_math(
            "glm53_router_select_ds4_default_v3",
            &["n_expert", "top_k", "scale", "logits", "bias"],
            &["selected", "weights", "probs"],
            ROUTER_SELECT_SOURCE,
            EXACT_HEADER,
            true,
            false,
            crate::runtime::Math::CompilerDefault,
        )
        // Full-write contract: The wrapper requires top_k <= experts; all selected/weight/probability slots are assigned.
        .map(MetalKernel::with_full_write_outputs)
        .ok()
    })
}

/// Fused MoE router select (ds4 `kernel_glm_router_select_one` port): logits
/// [n, E] f32 (contiguous), bias [E] (any dtype, cast to f32) -> selected
/// [n, top_k] i32 (descending by sigmoid+bias, ties by lower index) and
/// weights [n, top_k] f32 (sigmoid renormalized over the selected list,
/// clamped at 2^-14, scaled). `None` = declined (host fallback). Covers the
/// n_group == 1 routing of glm5_next; grouped routing stays on the host path.
pub fn router_select_fused(
    logits: &Array,
    bias: &Array,
    top_k: usize,
    scale: f32,
    stream: &Stream,
) -> Option<(Array, Array)> {
    let Some(kernel) = router_select_kernel().as_ref() else {
        eprintln!("[router-decline] Metal router-selection kernel failed to initialize");
        return None;
    };
    if logits.dtype() != Dtype::Float32 || logits.rank() != 2 {
        return None;
    }
    let n = logits.dim(0) as usize;
    let e = logits.dim(1) as usize;
    if e == 0 || e > 512 || top_k == 0 || top_k > 16 || top_k > e {
        return None;
    }
    let ea = Array::from_int(e as i32);
    let ka = Array::from_int(top_k as i32);
    let sc = Array::from_f32_as(scale, Dtype::Float32);
    let b32 = match bias.as_dtype(Dtype::Float32) {
        Ok(a) => a,
        Err(err) => {
            eprintln!("[router-decline] {err}");
            return None;
        }
    };
    for a in [logits, &b32] {
        if let Err(err) = a.eval_enqueue() {
            eprintln!("[router-decline] {err}");
            return None;
        }
    }
    let inputs: [&Array; 5] = [&ea, &ka, &sc, logits, &b32];
    let outputs = [
        OutputArg {
            shape: vec![n as i32, top_k as i32],
            dtype: Dtype::Int32,
        },
        OutputArg {
            shape: vec![n as i32, top_k as i32],
            dtype: Dtype::Float32,
        },
        OutputArg {
            shape: vec![n as i32, e as i32],
            dtype: Dtype::Float32,
        },
    ];
    let all: Vec<&Array> = inputs.iter().map(|a| &**a).collect();
    let sort_width = if e > 256 { 512 } else { 256 };
    let out = match kernel.apply(
        &all,
        &[],
        ((n as i32 * sort_width), 1, 1),
        (sort_width, 1, 1),
        &outputs,
        stream,
    ) {
        Ok(o) => o,
        Err(err) => {
            eprintln!("[router-decline] {err}");
            return None;
        }
    };
    let mut it = out.into_iter();
    let sel = it.next()?;
    let wts = it.next()?;
    Some((sel, wts))
}

const INDEXER_POOL_SOURCE: &str = include_str!("../../shaders/glm53/indexer_pool.metal");

const MTP_RMS_NORM_SOURCE: &str = include_str!("../../shaders/glm53/mtp_rms_norm.metal");
const HC_MIX_RMS_SOURCE: &str = include_str!("../../shaders/glm53/hc_mix_rms.metal");
const HC_MIX_RMS_BF16_GEMV_SOURCE: &str =
    include_str!("../../shaders/glm53/hc_mix_rms_bf16_gemv.metal");
const MTP_ADD_RMS_NORM_SOURCE: &str = include_str!("../../shaders/glm53/mtp_add_rms_norm.metal");
const MTP_ADD3_SOURCE: &str = include_str!("../../shaders/glm53/mtp_add3.metal");

fn ds4_rms_norm_kernel() -> &'static Option<MetalKernel> {
    static K: std::sync::OnceLock<Option<MetalKernel>> = std::sync::OnceLock::new();
    K.get_or_init(|| {
        MetalKernel::new_with_math(
            "glm53_mtp_rms_norm_ds4_f32x4_v1",
            &["n_dim", "n_rows", "eps", "input", "weight"],
            &["output"],
            MTP_RMS_NORM_SOURCE,
            EXACT_HEADER,
            true,
            false,
            crate::runtime::Math::CompilerDefault,
        )
        // Full-write contract: Float4 lane walks cover every channel of every admitted row.
        .map(MetalKernel::with_full_write_outputs)
        .ok()
    })
}

/// GLM-5.3 weighted RMSNorm using DS4's float4/SIMD reduction order.
pub fn glm53_ds4_rms_norm(
    input: &Array,
    weight: &Array,
    eps: f32,
    stream: &Stream,
) -> Option<Array> {
    let kernel = ds4_rms_norm_kernel().as_ref()?;
    if input.dtype() != Dtype::Float32 || input.rank() == 0 {
        return None;
    }
    let n = input.dim(input.rank() as i32 - 1) as usize;
    if n == 0 || n % 4 != 0 || weight.size() != n {
        return None;
    }
    let rows = input.size() / n;
    if rows == 0 {
        return None;
    }
    let weight_f32 = weight.as_dtype(Dtype::Float32).ok()?;
    for a in [input, &weight_f32] {
        if let Err(err) = a.eval_enqueue() {
            eprintln!("[glm53-mtp-rms-decline] {err}");
            return None;
        }
    }
    let n_a = Array::from_int(n as i32);
    let rows_a = Array::from_int(rows as i32);
    let eps_a = Array::from_f32_as(eps, Dtype::Float32);
    let inputs: [&Array; 5] = [&n_a, &rows_a, &eps_a, input, &weight_f32];
    let output = [OutputArg {
        shape: input.shape().iter().map(|&d| d as i32).collect(),
        dtype: Dtype::Float32,
    }];
    let all: Vec<&Array> = inputs.iter().map(|a| &**a).collect();

    let n4 = n / 4;
    let mut threads = 32usize;
    while threads < n4 && threads < 1024 {
        threads *= 2;
    }
    if threads > n4 {
        threads = n4;
    }
    let total = rows.saturating_mul(threads);
    let out = match kernel.apply(
        &all,
        &[],
        (total as i32, 1, 1),
        (threads as i32, 1, 1),
        &output,
        stream,
    ) {
        Ok(out) => out.into_iter().next()?,
        Err(err) => {
            eprintln!("[glm53-mtp-rms-decline] {err}");
            return None;
        }
    };
    Some(out)
}

/// GLM-5.3 MTP weighted RMSNorm using DS4's float4/SIMD reduction order.
pub fn glm53_mtp_rms_norm(
    input: &Array,
    weight: &Array,
    eps: f32,
    stream: &Stream,
) -> Option<Array> {
    glm53_ds4_rms_norm(input, weight, eps, stream)
}

fn hc_mix_rms_kernel() -> &'static Option<MetalKernel> {
    static K: std::sync::OnceLock<Option<MetalKernel>> = std::sync::OnceLock::new();
    K.get_or_init(|| {
        MetalKernel::new_with_math(
            "glm53_hc_mix_rms_ds4_f32x4_v1",
            &["n_dim", "n_rows", "eps", "input"],
            &["output"],
            HC_MIX_RMS_SOURCE,
            EXACT_HEADER,
            true,
            false,
            crate::runtime::Math::CompilerDefault,
        )
        // Full-write contract: Float4 lane walks cover every channel of every admitted row.
        .map(MetalKernel::with_full_write_outputs)
        .ok()
    })
}

/// Plain F32 RMSNorm for the flattened mHC stream, using ds4's float4 and
/// threadgroup reduction order instead of a generic square/reduce/scale graph.
pub fn glm53_hc_mix_rms(input: &Array, eps: f32, stream: &Stream) -> Option<Array> {
    let kernel = hc_mix_rms_kernel().as_ref()?;
    if input.dtype() != Dtype::Float32 || input.rank() != 2 {
        return None;
    }
    let n = input.dim(1) as usize;
    let rows = input.dim(0) as usize;
    if n == 0 || n % 4 != 0 || rows == 0 {
        return None;
    }
    input.eval_enqueue().ok()?;
    let n_a = Array::from_int(n as i32);
    let rows_a = Array::from_int(rows as i32);
    let eps_a = Array::from_f32_as(eps, Dtype::Float32);
    let inputs: [&Array; 4] = [&n_a, &rows_a, &eps_a, input];
    let output = [OutputArg {
        shape: input.shape().iter().map(|&d| d as i32).collect(),
        dtype: Dtype::Float32,
    }];
    let all: Vec<&Array> = inputs.iter().map(|a| &**a).collect();
    let n4 = n / 4;
    let mut threads = 32usize;
    while threads < n4 && threads < 1024 {
        threads *= 2;
    }
    if threads > n4 {
        threads = n4;
    }
    let total = rows.checked_mul(threads)?;
    let out = kernel
        .apply(
            &all,
            &[],
            (total as i32, 1, 1),
            (threads as i32, 1, 1),
            &output,
            stream,
        )
        .ok()?
        .into_iter()
        .next()?;
    Some(out)
}

fn hc_mix_rms_bf16_gemv_kernel() -> &'static Option<MetalKernel> {
    static K: std::sync::OnceLock<Option<MetalKernel>> = std::sync::OnceLock::new();
    K.get_or_init(|| {
        MetalKernel::new_with_math(
            "glm53_hc_mix_rms_bf16_gemv_ds4_order_v1",
            &["n_dim", "out_dim", "n_rows", "eps", "weights", "input"],
            &["output"],
            HC_MIX_RMS_BF16_GEMV_SOURCE,
            EXACT_HEADER,
            true,
            false,
            crate::runtime::Math::CompilerDefault,
        )
        // Full-write contract: The admitted shape assigns all 24 projection outputs per token.
        .map(MetalKernel::with_full_write_outputs)
        .ok()
    })
}

/// Fused mHC input RMSNorm and BF16 projection for the 24-by-16384 decode and
/// short-verify shapes. The RMS tree and each output's DS4 lane-FMA/SIMD-sum
/// order match the separate `glm53_hc_mix_rms` + `glm53_bf16_gemv` chain.
pub fn glm53_hc_mix_rms_bf16_gemv(
    weights: &Array,
    input: &Array,
    eps: f32,
    stream: &Stream,
) -> Option<Array> {
    let kernel = hc_mix_rms_bf16_gemv_kernel().as_ref()?;
    if weights.rank() != 2
        || weights.dtype() != Dtype::Bfloat16
        || weights.dim(0) != 24
        || weights.dim(1) != 16_384
        || input.rank() != 2
        || input.dtype() != Dtype::Float32
        || input.dim(1) != 16_384
        || input.dim(0) <= 0
        || input.dim(0) > 8
    {
        return None;
    }
    let weights = weights.contiguous().ok()?;
    let input = input.contiguous().ok()?;
    weights.eval_enqueue().ok()?;
    input.eval_enqueue().ok()?;
    let inputs = [
        KernelInput::I32(16_384),
        KernelInput::I32(24),
        KernelInput::I32(input.dim(0)),
        KernelInput::F32(eps),
        KernelInput::Tensor(&weights.t),
        KernelInput::Tensor(&input.t),
    ];
    let outputs = [OutputArg {
        shape: vec![input.dim(0), 24],
        dtype: Dtype::Float32,
    }];
    let nsg = bf16_projection_nsg(&input);
    let grid_x = (24usize.checked_add(nsg - 1)? / nsg).checked_mul(32)?;
    let grid_y = input.dim(0).checked_mul(nsg as i32)?;
    kernel
        .apply_with_inline_inputs(
            &inputs,
            &[],
            (i32::try_from(grid_x).ok()?, grid_y, 1),
            (32, nsg as i32, 1),
            &outputs,
            stream,
        )
        .ok()?
        .into_iter()
        .next()
}

fn mtp_add_rms_norm_kernel() -> &'static Option<MetalKernel> {
    static K: std::sync::OnceLock<Option<MetalKernel>> = std::sync::OnceLock::new();
    K.get_or_init(|| {
        MetalKernel::new_with_math_and_threadgroup_memory(
            "glm53_mtp_add_rms_norm_ds4_f32x4_v1",
            &["n_dim", "n_rows", "eps", "a", "b", "weight"],
            &["sum_out", "norm_out"],
            MTP_ADD_RMS_NORM_SOURCE,
            EXACT_HEADER,
            true,
            false,
            crate::runtime::Math::CompilerDefault,
            32 * std::mem::size_of::<f32>(),
        )
        // Full-write contract: Each row assigns every sum channel before its norm channel reads it.
        .map(MetalKernel::with_full_write_outputs)
        .ok()
    })
}

/// DS4's one-row MTP residual add and weighted RMSNorm, returning both the
/// residual sum and normalized row so the same sum can feed the next residual.
pub fn glm53_mtp_add_rms_norm(
    a: &Array,
    b: &Array,
    weight: &Array,
    eps: f32,
    stream: &Stream,
) -> Option<(Array, Array)> {
    let kernel = mtp_add_rms_norm_kernel().as_ref()?;
    if a.dtype() != Dtype::Float32 || b.dtype() != Dtype::Float32 || a.shape() != b.shape() {
        return None;
    }
    if a.rank() == 0 {
        return None;
    }
    let n = a.dim(a.rank() as i32 - 1) as usize;
    if n == 0 || n % 4 != 0 || weight.size() != n {
        return None;
    }
    let rows = a.size() / n;
    if rows == 0 {
        return None;
    }
    let weight = weight.as_dtype(Dtype::Float32).ok()?;
    for value in [a, b, &weight] {
        value.eval_enqueue().ok()?;
    }
    let n_a = Array::from_int(n as i32);
    let rows_a = Array::from_int(rows as i32);
    let eps_a = Array::from_f32_as(eps, Dtype::Float32);
    let inputs: [&Array; 6] = [&n_a, &rows_a, &eps_a, a, b, &weight];
    let shape: Vec<i32> = a.shape().iter().map(|&dim| dim as i32).collect();
    let outputs = [
        OutputArg {
            shape: shape.clone(),
            dtype: Dtype::Float32,
        },
        OutputArg {
            shape,
            dtype: Dtype::Float32,
        },
    ];
    let n4 = n / 4;
    let mut threads = 32usize;
    while threads < n4 && threads < 1024 {
        threads *= 2;
    }
    if threads > n4 {
        threads = n4;
    }
    let total = rows.saturating_mul(threads);
    let all: Vec<&Array> = inputs.iter().map(|value| &**value).collect();
    let mut out = kernel
        .apply(
            &all,
            &[],
            (total as i32, 1, 1),
            (threads as i32, 1, 1),
            &outputs,
            stream,
        )
        .ok()?
        .into_iter();
    Some((out.next()?, out.next()?))
}

fn mtp_add3_kernel() -> &'static Option<MetalKernel> {
    static K: std::sync::OnceLock<Option<MetalKernel>> = std::sync::OnceLock::new();
    K.get_or_init(|| {
        MetalKernel::new_with_math(
            "glm53_mtp_add3_ds4_f32_v1",
            &["n", "a", "b", "c"],
            &["output"],
            MTP_ADD3_SOURCE,
            EXACT_HEADER,
            true,
            false,
            crate::runtime::Math::CompilerDefault,
        )
        // Full-write contract: The flat element grid assigns every output element.
        .map(MetalKernel::with_full_write_outputs)
        .ok()
    })
}

/// DS4's ordered F32 residual + routed + shared MTP output sum.
pub fn glm53_mtp_add3(a: &Array, b: &Array, c: &Array, stream: &Stream) -> Option<Array> {
    let kernel = mtp_add3_kernel().as_ref()?;
    if a.dtype() != Dtype::Float32
        || b.dtype() != Dtype::Float32
        || c.dtype() != Dtype::Float32
        || a.shape() != b.shape()
        || a.shape() != c.shape()
    {
        return None;
    }
    let n = a.size();
    if n == 0 || n > i32::MAX as usize {
        return None;
    }
    for value in [a, b, c] {
        value.eval_enqueue().ok()?;
    }
    let n_a = Array::from_int(n as i32);
    let inputs: [&Array; 4] = [&n_a, a, b, c];
    let outputs = [OutputArg {
        shape: a.shape().iter().map(|&dim| dim as i32).collect(),
        dtype: Dtype::Float32,
    }];
    let threads = n.min(256);
    let all: Vec<&Array> = inputs.iter().map(|value| &**value).collect();
    kernel
        .apply(
            &all,
            &[],
            (n as i32, 1, 1),
            (threads as i32, 1, 1),
            &outputs,
            stream,
        )
        .ok()?
        .into_iter()
        .next()
}

const MTP_KV_LORA_RMS_NORM_SOURCE: &str =
    include_str!("../../shaders/glm53/mtp_kv_lora_rms_norm.metal");
const QKV_NORM_COMPACT_SOURCE: &str = include_str!("../../shaders/glm53/qkv_norm_compact.metal");
const DSA_QKV_NORM_ROWS_SOURCE: &str =
    include_str!("../../shaders/glm53/dsa_qkv_rms_norm_rows.metal");
const DSA_QKV_PROJECT_PAIR_SOURCE: &str =
    include_str!("../../shaders/glm53/dsa_qkv_project_pair.metal");

fn mtp_kv_lora_rms_norm_kernel() -> &'static Option<MetalKernel> {
    static K: std::sync::OnceLock<Option<MetalKernel>> = std::sync::OnceLock::new();
    K.get_or_init(|| {
        MetalKernel::new_with_math(
            "glm53_mtp_kv_lora_rms_norm_ds4_scalar_v1",
            &["n_dim", "raw_dim", "n_rows", "eps", "input", "weight"],
            &["output"],
            MTP_KV_LORA_RMS_NORM_SOURCE,
            EXACT_HEADER,
            true,
            false,
            crate::runtime::Math::CompilerDefault,
        )
        // Full-write contract: The row/lane walk assigns every normalized KV channel.
        .map(MetalKernel::with_full_write_outputs)
        .ok()
    })
}

fn dsa_qkv_norm_rows_kernel() -> &'static Option<MetalKernel> {
    static K: std::sync::OnceLock<Option<MetalKernel>> = std::sync::OnceLock::new();
    K.get_or_init(|| {
        MetalKernel::new_with_math_and_threadgroup_memory(
            "glm53_dsa_qkv_norm_rows_ds4_f32x4_v1",
            &[
                "q_dim",
                "kv_dim",
                "n_rows",
                "eps",
                "q_input",
                "q_weight",
                "kv_input",
                "kv_weight",
            ],
            &["q_output", "kv_output"],
            DSA_QKV_NORM_ROWS_SOURCE,
            EXACT_HEADER,
            true,
            false,
            crate::runtime::Math::CompilerDefault,
            32 * std::mem::size_of::<f32>(),
        )
        .ok()
    })
}

/// DS4's fused DSA q/kv normalization, retaining its q-sized threadgroup for
/// both branches. The GLM q-rank has 384 float4 lanes; DS4 uses those lanes
/// when reducing the shorter 512-value KV row as well.
pub fn glm53_dsa_qkv_rms_norm_rows(
    q_input: &Array,
    q_weight: &Array,
    kv_input: &Array,
    kv_weight: &Array,
    eps: f32,
    stream: &Stream,
) -> Option<(Array, Array)> {
    let kernel = dsa_qkv_norm_rows_kernel().as_ref()?;
    if q_input.dtype() != Dtype::Float32
        || kv_input.dtype() != Dtype::Float32
        || q_input.rank() != 3
        || kv_input.rank() != 3
        || q_input.shape()[..2] != kv_input.shape()[..2]
        || q_weight.rank() != 1
        || kv_weight.rank() != 1
    {
        return None;
    }
    let q_dim = q_input.dim(2) as usize;
    let kv_dim = kv_input.dim(2) as usize;
    if q_dim != 1536 || kv_dim != 512 || q_weight.size() != q_dim || kv_weight.size() != kv_dim {
        return None;
    }
    let rows = q_input.size() / q_dim;
    if rows == 0 {
        return None;
    }
    let q_input = q_input.contiguous().ok()?;
    let kv_input = kv_input.contiguous().ok()?;
    let q_weight = q_weight.as_dtype(Dtype::Float32).ok()?.contiguous().ok()?;
    let kv_weight = kv_weight.as_dtype(Dtype::Float32).ok()?.contiguous().ok()?;
    for value in [&q_input, &kv_input, &q_weight, &kv_weight] {
        value.eval_enqueue().ok()?;
    }
    let q_dim_arg = Array::from_int(q_dim as i32);
    let kv_dim_arg = Array::from_int(kv_dim as i32);
    let rows_arg = Array::from_int(rows as i32);
    let eps_arg = Array::from_f32_as(eps, Dtype::Float32);
    let inputs: [&Array; 8] = [
        &q_dim_arg,
        &kv_dim_arg,
        &rows_arg,
        &eps_arg,
        &q_input,
        &q_weight,
        &kv_input,
        &kv_weight,
    ];
    let outputs = [
        OutputArg {
            shape: q_input.shape().to_vec(),
            dtype: Dtype::Float32,
        },
        OutputArg {
            shape: kv_input.shape().to_vec(),
            dtype: Dtype::Float32,
        },
    ];
    let all: Vec<&Array> = inputs.iter().map(|value| &**value).collect();
    let threads = q_dim / 4;
    let mut output = kernel
        .apply(
            &all,
            &[],
            ((rows * 2 * threads) as i32, 1, 1),
            (threads as i32, 1, 1),
            &outputs,
            stream,
        )
        .ok()?
        .into_iter();
    Some((output.next()?, output.next()?))
}

/// GLM MTP `kv_a_layernorm` using ds4's scalar strided reduction and tree.
pub fn glm53_mtp_kv_lora_rms_norm(
    input: &Array,
    weight: &Array,
    eps: f32,
    stream: &Stream,
) -> Option<Array> {
    let kernel = mtp_kv_lora_rms_norm_kernel().as_ref()?;
    if input.dtype() != Dtype::Float32 || input.rank() == 0 {
        return None;
    }
    let raw_dim = input.dim(input.rank() as i32 - 1) as usize;
    let n = weight.size();
    // GLM-5.3's MTP KV-LoRA vector is exactly 512 values. The shader's
    // reduction scratch and ds4 launch shape are both specialized to 128
    // threads; reject other shapes rather than allowing a mismatched
    // threadgroup to index beyond that scratch allocation.
    if n != 512 || raw_dim != 512 || weight.rank() != 1 {
        return None;
    }
    let rows = input.size() / raw_dim;
    if rows == 0 {
        return None;
    }
    let weight_f32 = weight.as_dtype(Dtype::Float32).ok()?;
    for a in [input, &weight_f32] {
        if let Err(err) = a.eval_enqueue() {
            eprintln!("[glm53-mtp-kv-rms-decline] {err}");
            return None;
        }
    }
    let n_a = Array::from_int(n as i32);
    let raw_a = Array::from_int(raw_dim as i32);
    let rows_a = Array::from_int(rows as i32);
    let eps_a = Array::from_f32_as(eps, Dtype::Float32);
    let inputs: [&Array; 6] = [&n_a, &raw_a, &rows_a, &eps_a, input, &weight_f32];
    let output = [OutputArg {
        shape: input
            .shape()
            .iter()
            .enumerate()
            .map(|(i, &d)| {
                if i + 1 == input.rank() {
                    n as i32
                } else {
                    d as i32
                }
            })
            .collect(),
        dtype: Dtype::Float32,
    }];
    let all: Vec<&Array> = inputs.iter().map(|a| &**a).collect();

    // ds4_gpu_rms_norm_threads(512) divides the dimension by four and
    // selects 128 threads for this exact GLM shape.
    let threads = 128usize;
    let total = rows.saturating_mul(threads);
    let out = match kernel.apply(
        &all,
        &[],
        (total as i32, 1, 1),
        (threads as i32, 1, 1),
        &output,
        stream,
    ) {
        Ok(out) => out.into_iter().next()?,
        Err(err) => {
            eprintln!("[glm53-mtp-kv-rms-decline] {err}");
            return None;
        }
    };
    Some(out)
}

fn dsa_qkv_project_pair_kernel() -> &'static Option<MetalKernel> {
    static K: std::sync::OnceLock<Option<MetalKernel>> = std::sync::OnceLock::new();
    K.get_or_init(|| {
        MetalKernel::new_with_math_and_threadgroup_memory(
            "glm53_dsa_qkv_project_pair_ds4_mlx_f32_v2",
            &[
                "in_dim",
                "q_rows",
                "kv_rows_arg",
                "nxpsg_arg",
                "r1ptg_arg",
                "q_weights",
                "q_scales",
                "q_biases",
                "kv_weights",
                "kv_scales",
                "kv_biases",
                "x",
            ],
            &["q_raw", "kv_raw"],
            DSA_QKV_PROJECT_PAIR_SOURCE,
            &moe_gather_header(),
            true,
            false,
            crate::runtime::Math::SafeNoLang,
            32 * std::mem::size_of::<f32>(),
        )
        .ok()
    })
}

/// Pair GLM DSA `q_a` and `kv_a` affine projections in one dispatch. The
/// query-rank branch keeps MLX qmv_fast arithmetic. For one row, `kv_a` uses
/// DS4's F32 affine matvec order; for 2..8 rows it uses DS4's small-batch row
/// grouping and shuffle reduction.
pub fn glm53_dsa_qkv_project_pair(
    q_weights: &Array,
    q_scales: &Array,
    q_biases: &Array,
    kv_weights: &Array,
    kv_scales: &Array,
    kv_biases: &Array,
    x: &Array,
    stream: &Stream,
) -> Option<(Array, Array)> {
    if x.rank() != 3
        || x.dim(0) <= 0
        || x.dim(1) != 1
        || x.dim(2) != 4096
        || x.dtype() != Dtype::Float32
        || q_weights.shape() != [1536, 1024]
        || kv_weights.shape() != [512, 1024]
        || q_weights.dtype() != Dtype::Uint32
        || kv_weights.dtype() != Dtype::Uint32
        || q_scales.shape() != [1536, 64]
        || q_biases.shape() != [1536, 64]
        || kv_scales.shape() != [512, 64]
        || kv_biases.shape() != [512, 64]
        || q_scales.dtype() != Dtype::Float32
        || q_biases.dtype() != Dtype::Float32
        || kv_scales.dtype() != Dtype::Float32
        || kv_biases.dtype() != Dtype::Float32
    {
        return None;
    }
    let q_weights = q_weights.contiguous().ok()?;
    let q_scales = q_scales.contiguous().ok()?;
    let q_biases = q_biases.contiguous().ok()?;
    let kv_weights = kv_weights.contiguous().ok()?;
    let kv_scales = kv_scales.contiguous().ok()?;
    let kv_biases = kv_biases.contiguous().ok()?;
    let x = x.contiguous().ok()?;
    let rows = x.dim(0) as usize;
    for input in [
        &q_weights,
        &q_scales,
        &q_biases,
        &kv_weights,
        &kv_scales,
        &kv_biases,
        &x,
    ] {
        input.eval_enqueue().ok()?;
    }
    let batch_schedule = (2..=8).contains(&rows);
    let nxpsg = if rows == 2 { 16usize } else { 8usize };
    let r1ptg = match rows {
        2 => 2usize,
        3 | 6 => 3,
        4 | 7 | 8 => 4,
        5 => 5,
        _ => 1,
    };
    let kv_groups = if batch_schedule {
        let kv_outputs_per_group = (32 / nxpsg) * 8;
        512usize.div_ceil(kv_outputs_per_group)
    } else if rows == 1 {
        64
    } else {
        512
    };
    let row_groups = if batch_schedule {
        rows.div_ceil(r1ptg)
    } else {
        1
    };
    let inputs = [
        KernelInput::I32(4096),
        KernelInput::I32(1536),
        KernelInput::I32(rows as i32),
        KernelInput::I32(nxpsg as i32),
        KernelInput::I32(r1ptg as i32),
        KernelInput::Tensor(&q_weights.t),
        KernelInput::Tensor(&q_scales.t),
        KernelInput::Tensor(&q_biases.t),
        KernelInput::Tensor(&kv_weights.t),
        KernelInput::Tensor(&kv_scales.t),
        KernelInput::Tensor(&kv_biases.t),
        KernelInput::Tensor(&x.t),
    ];
    let kernel = dsa_qkv_project_pair_kernel().as_ref()?;
    let mut outputs = kernel
        .apply_with_inline_inputs(
            &inputs,
            &[],
            (
                rows as i32 * 32,
                (48 + kv_groups as i32) * 8,
                row_groups as i32,
            ),
            (32, 8, 1),
            &[
                OutputArg {
                    shape: vec![rows as i32, 1, 1536],
                    dtype: Dtype::Float32,
                },
                OutputArg {
                    shape: vec![rows as i32, 1, 512],
                    dtype: Dtype::Float32,
                },
            ],
            stream,
        )
        .ok()?;
    if outputs.len() != 2 {
        return None;
    }
    let kv_raw = outputs.pop()?;
    let q_raw = outputs.pop()?;
    Some((q_raw, kv_raw))
}

fn qkv_norm_compact_kernel() -> &'static Option<MetalKernel> {
    static K: std::sync::OnceLock<Option<MetalKernel>> = std::sync::OnceLock::new();
    K.get_or_init(|| {
        MetalKernel::new_with_math_and_threadgroup_memory(
            "glm53_qkv_norm_compact_ds4_float4_v1",
            &[
                "q_dim",
                "kv_dim",
                "n_rows",
                "cache_pos_arg",
                "eps",
                "q_input",
                "q_weight",
                "kv_input",
                "kv_weight",
            ],
            &["q_norm", "kv_norm"],
            QKV_NORM_COMPACT_SOURCE,
            EXACT_HEADER,
            false,
            false,
            crate::runtime::Math::CompilerDefault,
            32 * std::mem::size_of::<f32>(),
        )
        .ok()
    })
}

/// Fused DS4-order q_a/kv_a normalization. Returns weighted q rows as F32 and
/// normalized compact latents as F16, matching DS4's fused norm-and-store path.
pub fn glm53_qkv_norm_compact(
    q_input: &Array,
    q_weight: &Array,
    kv_input: &Array,
    kv_weight: &Array,
    eps: f32,
    stream: &Stream,
) -> Option<(Array, Array)> {
    let kernel = qkv_norm_compact_kernel().as_ref()?;
    if q_input.dtype() != Dtype::Float32
        || kv_input.dtype() != Dtype::Float32
        || q_input.rank() == 0
        || q_input.rank() != kv_input.rank()
        || q_weight.rank() != 1
        || kv_weight.rank() != 1
        || kv_weight.size() != 512
    {
        return None;
    }
    let q_dim = q_input.dim(q_input.rank() as i32 - 1) as usize;
    let kv_dim = kv_input.dim(kv_input.rank() as i32 - 1) as usize;
    if q_dim == 0
        || q_dim % 4 != 0
        || kv_dim != kv_weight.size()
        || kv_dim % 4 != 0
        || q_weight.size() != q_dim
        || q_input.shape()[..q_input.rank() - 1] != kv_input.shape()[..kv_input.rank() - 1]
    {
        return None;
    }
    let rows = q_input.size() / q_dim;
    if rows == 0 || rows > i32::MAX as usize {
        return None;
    }
    let q_input = q_input.contiguous().ok()?;
    let kv_input = kv_input.contiguous().ok()?;
    let q_weight = q_weight.as_dtype(Dtype::Float32).ok()?.contiguous().ok()?;
    let kv_weight = kv_weight.as_dtype(Dtype::Float32).ok()?.contiguous().ok()?;
    for input in [&q_input, &kv_input, &q_weight, &kv_weight] {
        input.eval_enqueue().ok()?;
    }

    let q_dim_arg = Array::from_int(q_dim as i32);
    let kv_dim_arg = Array::from_int(kv_dim as i32);
    let rows_arg = Array::from_int(rows as i32);
    let cache_pos_arg = Array::from_int(0);
    let eps_arg = Array::from_f32_as(eps, Dtype::Float32);
    let inputs: [&Array; 9] = [
        &q_dim_arg,
        &kv_dim_arg,
        &rows_arg,
        &cache_pos_arg,
        &eps_arg,
        &q_input,
        &q_weight,
        &kv_input,
        &kv_weight,
    ];
    let outputs = [
        OutputArg {
            shape: q_input.shape().to_vec(),
            dtype: Dtype::Float32,
        },
        OutputArg {
            shape: kv_input.shape().to_vec(),
            dtype: Dtype::Float16,
        },
    ];
    let all: Vec<&Array> = inputs.iter().map(|value| &**value).collect();
    let n4 = q_dim / 4;
    let mut threads = 32usize;
    while threads < n4 && threads < 1024 {
        threads *= 2;
    }
    if threads > n4 {
        threads = n4;
    }
    let total_threads = rows.checked_mul(2)?.checked_mul(threads)?;
    if total_threads > i32::MAX as usize {
        return None;
    }
    let mut out = kernel
        .apply(
            &all,
            &[],
            (total_threads as i32, 1, 1),
            (threads as i32, 1, 1),
            &outputs,
            stream,
        )
        .ok()?
        .into_iter();
    Some((out.next()?, out.next()?))
}

/// Fused q_a/kv_a normalization with the F16 compact latent written directly
/// into an existing cache buffer at `cache_pos`, avoiding a temporary latent
/// array and the subsequent cache-copy dispatch.
pub fn glm53_qkv_norm_compact_into(
    q_input: &Array,
    q_weight: &Array,
    kv_input: &Array,
    kv_weight: &Array,
    cache: &Array,
    cache_pos: usize,
    eps: f32,
    stream: &Stream,
) -> Option<Array> {
    let kernel = qkv_norm_compact_kernel().as_ref()?;
    if q_input.dtype() != Dtype::Float32
        || kv_input.dtype() != Dtype::Float32
        || q_input.rank() != 3
        || kv_input.rank() != 3
        || q_input.dim(0) != 1
        || kv_input.dim(0) != 1
        || q_input.shape()[..2] != kv_input.shape()[..2]
        || q_weight.rank() != 1
        || kv_weight.rank() != 1
        || cache.dtype() != Dtype::Float16
        || cache.rank() != 3
        || cache.dim(0) != 1
    {
        return None;
    }
    let q_dim = q_input.dim(2) as usize;
    let kv_dim = kv_input.dim(2) as usize;
    let rows = q_input.dim(1) as usize;
    let cache_cap = cache.dim(1) as usize;
    if q_dim == 0
        || q_dim % 4 != 0
        || kv_dim != 512
        || kv_weight.size() != kv_dim
        || cache.dim(2) as usize != kv_dim
        || q_weight.size() != q_dim
        || cache_pos > i32::MAX as usize
        || cache_pos.checked_add(rows)? > cache_cap
    {
        return None;
    }
    let n4 = q_dim / 4;
    let mut threads = 32usize;
    while threads < n4 && threads < 1024 {
        threads *= 2;
    }
    if threads > n4 {
        threads = n4;
    }
    let total_threads = rows.checked_mul(2)?.checked_mul(threads)?;
    if rows == 0 || total_threads > i32::MAX as usize {
        return None;
    }

    let q_input = q_input.contiguous().ok()?;
    let kv_input = kv_input.contiguous().ok()?;
    let q_weight = q_weight.as_dtype(Dtype::Float32).ok()?.contiguous().ok()?;
    let kv_weight = kv_weight.as_dtype(Dtype::Float32).ok()?.contiguous().ok()?;
    for input in [&q_input, &kv_input, &q_weight, &kv_weight] {
        input.eval_enqueue().ok()?;
    }

    // `apply_into` binds preallocated outputs at buffer offset zero. The
    // compact-cache output is therefore the whole contiguous cache, and the
    // kernel applies `cache_pos` to each written row.
    let q_norm = crate::ops::zeros::<f32>(&q_input.shape().to_vec()).ok()?;
    let q_dim_arg = Array::from_int(q_dim as i32);
    let kv_dim_arg = Array::from_int(kv_dim as i32);
    let rows_arg = Array::from_int(rows as i32);
    let cache_pos_arg = Array::from_int(cache_pos as i32);
    let eps_arg = Array::from_f32_as(eps, Dtype::Float32);
    let inputs: [&Array; 9] = [
        &q_dim_arg,
        &kv_dim_arg,
        &rows_arg,
        &cache_pos_arg,
        &eps_arg,
        &q_input,
        &q_weight,
        &kv_input,
        &kv_weight,
    ];
    let outputs = [
        OutputArg {
            shape: q_input.shape().to_vec(),
            dtype: Dtype::Float32,
        },
        OutputArg {
            shape: cache.shape().to_vec(),
            dtype: Dtype::Float16,
        },
    ];
    let all: Vec<&Array> = inputs.iter().map(|value| &**value).collect();
    let mut out = kernel
        .apply_into(
            &all,
            &[],
            (total_threads as i32, 1, 1),
            (threads as i32, 1, 1),
            &outputs,
            &[&q_norm, cache],
            stream,
        )
        .ok()?
        .into_iter();
    out.next()
}

fn indexer_pool_kernel() -> &'static Option<MetalKernel> {
    static K: std::sync::OnceLock<Option<MetalKernel>> = std::sync::OnceLock::new();
    K.get_or_init(|| {
        MetalKernel::new_with_math_and_threadgroup_memory(
            "glm53_indexer_pool_v6_ds4_runtime_loops_f16",
            &[
                "n_rows",
                "hd_dim",
                "pool_dim",
                "eps",
                "raw_key",
                "gate",
                "ape",
                "norm_weight",
                "norm_bias",
            ],
            &["pool"],
            INDEXER_POOL_SOURCE,
            EXACT_HEADER,
            true,
            false,
            crate::runtime::Math::CompilerDefault,
            (4 * 128 + 2 * 4) * std::mem::size_of::<f32>(),
        )
        .ok()
    })
}

/// Fused DSA indexer pool update: raw keys and gates are separate F32
/// `[B,n,4,hd]` arrays, APE is checkpoint BF16, and layer-norm weights are
/// F32. The kernel writes the ds4-style F16 pooled-key cache.
pub fn indexer_pool_fused(
    raw_key: &Array,
    gate: &Array,
    ape: &Array,
    norm_weight: &Array,
    norm_bias: &Array,
    eps: f32,
    stream: &Stream,
) -> Option<Array> {
    let kernel = indexer_pool_kernel().as_ref()?;
    if raw_key.dtype() != Dtype::Float32
        || gate.dtype() != Dtype::Float32
        || raw_key.shape() != gate.shape()
        || raw_key.rank() != 4
        || raw_key.dim(3) != 128
        || ape.dtype() != Dtype::Bfloat16
    {
        return None;
    }
    let norm_w = norm_weight.as_dtype(Dtype::Float32).ok()?;
    let norm_b = norm_bias.as_dtype(Dtype::Float32).ok()?;
    if raw_key.dim(2) != 4 {
        return None;
    }
    let b = raw_key.dim(0) as usize;
    let n = raw_key.dim(1) as usize;
    let hd = raw_key.dim(3) as usize;
    if ape.size() != 4 * hd {
        return None;
    }
    if b == 0 || n == 0 || hd != 128 {
        return None;
    }
    let rows = b * n;
    for a in [raw_key, gate, ape, &norm_w, &norm_b] {
        if let Err(err) = a.eval_enqueue() {
            eprintln!("[indexer-pool-decline] {err}");
            return None;
        }
    }
    let inputs = [
        KernelInput::I32(rows as i32),
        KernelInput::I32(hd as i32),
        KernelInput::I32(4),
        KernelInput::F32(eps),
        KernelInput::Tensor(&raw_key.t),
        KernelInput::Tensor(&gate.t),
        KernelInput::Tensor(&ape.t),
        KernelInput::Tensor(&norm_w.t),
        KernelInput::Tensor(&norm_b.t),
    ];
    let outputs = [OutputArg {
        shape: vec![b as i32, n as i32, hd as i32],
        dtype: Dtype::Float16,
    }];
    let out = match kernel.apply_with_inline_inputs(
        &inputs,
        &[],
        ((rows * hd) as i32, 1, 1),
        (hd as i32, 1, 1),
        &outputs,
        stream,
    ) {
        Ok(o) => o,
        Err(err) => {
            eprintln!("[indexer-pool-decline] {err}");
            return None;
        }
    };
    Some(out.into_iter().next()?)
}

const DSA_SCORES_SOURCE: &str = include_str!("../../shaders/glm53/dsa_scores.metal");
const DSA_WV_SOURCE: &str = include_str!("../../shaders/glm53/dsa_wv.metal");
const INDEXER_SCORES_BATCH_SOURCE: &str =
    include_str!("../../shaders/glm53/indexer_scores_batch.metal");
const INDEXER_SCORES_DECODE_SOURCE: &str =
    include_str!("../../shaders/glm53/indexer_scores_decode.metal");
const INDEXER_SCORES_DECODE_ROWS4_SOURCE: &str =
    include_str!("../../shaders/glm53/indexer_scores_decode_rows4.metal");
const INDEXER_SCORES_TILED_SOURCE: &str =
    include_str!("../../shaders/glm53/indexer_scores_tiled.metal");
const DSA_QK_LOWRANK_SOURCE: &str = include_str!("../../shaders/glm53/dsa_qk_lowrank.metal");

fn indexer_scores_batch_kernel() -> &'static Option<MetalKernel> {
    static K: std::sync::OnceLock<Option<MetalKernel>> = std::sync::OnceLock::new();
    K.get_or_init(|| {
        MetalKernel::new_with_math(
            "glm53_indexer_scores_batch_ds4_default_v2",
            &[
                "n_rows",
                "n_tokens",
                "n_batches",
                "n_heads",
                "head_dim",
                "pos0",
                "pool_size",
                "scale",
                "q",
                "weights",
                "pooled",
            ],
            &["scores"],
            INDEXER_SCORES_BATCH_SOURCE,
            EXACT_HEADER,
            true,
            false,
            crate::runtime::Math::CompilerDefault,
        )
        .ok()
    })
}

fn indexer_scores_decode_kernel() -> &'static Option<MetalKernel> {
    static K: std::sync::OnceLock<Option<MetalKernel>> = std::sync::OnceLock::new();
    K.get_or_init(|| {
        MetalKernel::new_with_math(
            "glm53_indexer_scores_decode_ds4_sg8_v1",
            &["n_rows", "n_batches", "scale", "q", "weights", "pooled"],
            &["scores"],
            INDEXER_SCORES_DECODE_SOURCE,
            EXACT_HEADER,
            true,
            false,
            crate::runtime::Math::CompilerDefault,
        )
        .ok()
    })
}

fn indexer_scores_decode_rows4_kernel() -> &'static Option<MetalKernel> {
    static K: std::sync::OnceLock<Option<MetalKernel>> = std::sync::OnceLock::new();
    K.get_or_init(|| {
        MetalKernel::new_with_math(
            "glm53_indexer_scores_decode_rows4_ds4_v1",
            &["n_rows", "n_batches", "scale", "q", "weights", "pooled"],
            &["scores"],
            INDEXER_SCORES_DECODE_ROWS4_SOURCE,
            EXACT_HEADER,
            true,
            false,
            crate::runtime::Math::CompilerDefault,
        )
        .ok()
    })
}

fn indexer_scores_tiled_kernel() -> &'static Option<MetalKernel> {
    static K: std::sync::OnceLock<Option<MetalKernel>> = std::sync::OnceLock::new();
    K.get_or_init(|| {
        MetalKernel::new_with_math(
            "glm53_indexer_scores_tiled_ds4_default_v2",
            &[
                "n_rows",
                "n_tokens",
                "n_batches",
                "n_heads",
                "pos0",
                "pool_size",
                "scale",
                "q",
                "weights",
                "pooled",
            ],
            &["scores"],
            INDEXER_SCORES_TILED_SOURCE,
            EXACT_HEADER,
            true,
            false,
            crate::runtime::Math::CompilerDefault,
        )
        .ok()
    })
}

/// Compute DSA selector scores with ds4's headwise F32 reduction schedule.
/// `q=[B,32,T,128]`, `weights=[B,T,32]`, pooled F16 keys `[B,P,128]`;
/// return `[B,T,P]` F32 with future pools set to negative infinity.
pub fn glm53_indexer_scores(
    q: &Array,
    weights: &Array,
    pooled: &Array,
    pos0: usize,
    pool_size: usize,
    scale: f32,
    stream: &Stream,
) -> Option<Array> {
    glm53_indexer_scores_with_decode_rows4(
        q, weights, pooled, pos0, pool_size, scale, false, false, stream,
    )
}

/// Compute DSA selector scores with the DS4 prefill batch reduction schedule,
/// including a one-row prefill tail. DS4 uses its batch kernel for that tail;
/// the decode-only kernels have a different float accumulation order.
pub fn glm53_indexer_scores_batch(
    q: &Array,
    weights: &Array,
    pooled: &Array,
    pos0: usize,
    pool_size: usize,
    scale: f32,
    stream: &Stream,
) -> Option<Array> {
    glm53_indexer_scores_with_decode_rows4(
        q, weights, pooled, pos0, pool_size, scale, false, true, stream,
    )
}

/// Compute DSA selector scores using DS4's four-pool-per-SIMD-group scorer on
/// one-token decode rows; all other row shapes retain the existing schedule.
pub fn glm53_indexer_scores_rows4(
    q: &Array,
    weights: &Array,
    pooled: &Array,
    pos0: usize,
    pool_size: usize,
    scale: f32,
    stream: &Stream,
) -> Option<Array> {
    glm53_indexer_scores_with_decode_rows4(
        q, weights, pooled, pos0, pool_size, scale, true, false, stream,
    )
}

fn glm53_indexer_scores_with_decode_rows4(
    q: &Array,
    weights: &Array,
    pooled: &Array,
    pos0: usize,
    pool_size: usize,
    scale: f32,
    use_rows4: bool,
    force_batch_t1: bool,
    stream: &Stream,
) -> Option<Array> {
    if q.rank() != 4
        || weights.rank() != 3
        || pooled.rank() != 3
        || q.dtype() != Dtype::Float32
        || weights.dtype() != Dtype::Float32
        || pooled.dtype() != Dtype::Float16
        || pool_size != 4
    {
        return None;
    }
    let (b, h, t, d) = (
        q.dim(0) as usize,
        q.dim(1) as usize,
        q.dim(2) as usize,
        q.dim(3) as usize,
    );
    let p = pooled.dim(1) as usize;
    if b == 0
        || h != 32
        || t == 0
        || d != 128
        || p == 0
        || weights.shape() != [b as i32, t as i32, h as i32]
        || pooled.dim(0) as usize != b
        || pooled.dim(2) as usize != d
        || pos0 > i32::MAX as usize
        || p > i32::MAX as usize
    {
        return None;
    }

    let q = q.contiguous().ok()?;
    let weights = weights.contiguous().ok()?;
    let pooled = pooled.contiguous().ok()?;
    for input in [&q, &weights, &pooled] {
        input.eval_enqueue().ok()?;
    }
    let outputs = [OutputArg {
        shape: vec![b as i32, t as i32, p as i32],
        dtype: Dtype::Float32,
    }];

    if t == 1 && !force_batch_t1 {
        let kernel = if use_rows4 {
            indexer_scores_decode_rows4_kernel().as_ref()?
        } else {
            indexer_scores_decode_kernel().as_ref()?
        };
        let inputs = [
            KernelInput::I32(p as i32),
            KernelInput::I32(b as i32),
            KernelInput::F32(scale),
            KernelInput::Tensor(&q.t),
            KernelInput::Tensor(&weights.t),
            KernelInput::Tensor(&pooled.t),
        ];
        let grid = if use_rows4 {
            ((b * p.div_ceil(16) * 32) as i32, 4, 1)
        } else {
            ((b * p * 32) as i32, 8, 1)
        };
        let threadgroup = if use_rows4 { (32, 4, 1) } else { (32, 8, 1) };
        return kernel
            .apply_with_inline_inputs(&inputs, &[], grid, threadgroup, &outputs, stream)
            .ok()?
            .into_iter()
            .next();
    }

    if t >= 8 {
        let Some(kernel) = indexer_scores_tiled_kernel().as_ref() else {
            return None;
        };
        let inputs = [
            KernelInput::I32(p as i32),
            KernelInput::I32(t as i32),
            KernelInput::I32(b as i32),
            KernelInput::I32(h as i32),
            KernelInput::I32(pos0 as i32),
            KernelInput::I32(pool_size as i32),
            KernelInput::F32(scale),
            KernelInput::Tensor(&q.t),
            KernelInput::Tensor(&weights.t),
            KernelInput::Tensor(&pooled.t),
        ];
        let tile_x = p.div_ceil(32);
        let tile_y = t.div_ceil(8);
        return kernel
            .apply_with_inline_inputs(
                &inputs,
                &[],
                ((tile_x * 32) as i32, (tile_y * 4) as i32, b as i32),
                (32, 4, 1),
                &outputs,
                stream,
            )
            .ok()?
            .into_iter()
            .next();
    }

    let Some(kernel) = indexer_scores_batch_kernel().as_ref() else {
        return None;
    };
    let inputs = [
        KernelInput::I32(p as i32),
        KernelInput::I32(t as i32),
        KernelInput::I32(b as i32),
        KernelInput::I32(h as i32),
        KernelInput::I32(d as i32),
        KernelInput::I32(pos0 as i32),
        KernelInput::I32(pool_size as i32),
        KernelInput::F32(scale),
        KernelInput::Tensor(&q.t),
        KernelInput::Tensor(&weights.t),
        KernelInput::Tensor(&pooled.t),
    ];
    kernel
        .apply_with_inline_inputs(
            &inputs,
            &[],
            ((p * 32) as i32, t as i32, b as i32),
            (32, 1, 1),
            &outputs,
            stream,
        )
        .ok()?
        .into_iter()
        .next()
}

fn dsa_scores_kernel() -> &'static Option<MetalKernel> {
    static K: std::sync::OnceLock<Option<MetalKernel>> = std::sync::OnceLock::new();
    K.get_or_init(|| {
        MetalKernel::new_with_math(
            "glm53_dsa_scores_ds4_default_v3",
            &["n_rows", "n_heads", "tc", "scale", "q", "k", "keep"],
            &["scores"],
            DSA_SCORES_SOURCE,
            EXACT_HEADER,
            true,
            false,
            crate::runtime::Math::CompilerDefault,
        )
        .ok()
    })
}

fn dsa_wv_kernel() -> &'static Option<MetalKernel> {
    static K: std::sync::OnceLock<Option<MetalKernel>> = std::sync::OnceLock::new();
    K.get_or_init(|| {
        MetalKernel::new_with_math(
            "glm53_dsa_wv_ds4_default_v3",
            &["n_rows", "tc", "scores", "v"],
            &["out"],
            DSA_WV_SOURCE,
            EXACT_HEADER,
            true,
            false,
            crate::runtime::Math::CompilerDefault,
        )
        .ok()
    })
}

/// Fused DSA decode attention, stage 1: q [B,heads,256], k
/// [B,heads,tc,256], keep [B,tc] are F32. Returns F32 scores.
pub fn dsa_decode_scores(
    q: &Array,
    k: &Array,
    keep: &Array,
    scale: f32,
    stream: &Stream,
) -> Option<Array> {
    let kernel = dsa_scores_kernel().as_ref()?;
    for a in [q, k, keep] {
        if a.dtype() != Dtype::Float32 {
            return None;
        }
    }
    if q.rank() != 3 || k.rank() != 4 || q.dim(0) != k.dim(0) || q.dim(1) != k.dim(1) {
        return None;
    }
    if q.dim(2) != 256 || k.dim(3) != 256 {
        return None;
    }
    let b = q.dim(0) as usize;
    let heads = q.dim(1) as usize;
    let tc = k.dim(2) as usize;
    if keep.rank() != 2 || keep.dim(0) != q.dim(0) || keep.dim(1) != k.dim(2) {
        return None;
    }
    if b == 0 || heads == 0 || tc == 0 {
        return None;
    }
    let rows = b * heads;
    let nr = Array::from_int(rows as i32);
    let nh = Array::from_int(heads as i32);
    let tca = Array::from_int(tc as i32);
    let sc = Array::from_f32_as(scale, Dtype::Float32);
    for a in [q, k, keep] {
        if let Err(err) = a.eval_enqueue() {
            eprintln!("[dsa-scores-decline] {err}");
            return None;
        }
    }
    let inputs: [&Array; 7] = [&nr, &nh, &tca, &sc, q, k, keep];
    let outputs = [OutputArg {
        shape: vec![rows as i32, tc as i32],
        dtype: Dtype::Float32,
    }];
    let all: Vec<&Array> = inputs.iter().map(|a| &**a).collect();
    let out = match kernel.apply(
        &all,
        &[],
        ((rows * tc) as i32, 1, 1),
        (1, 1, 1),
        &outputs,
        stream,
    ) {
        Ok(o) => o,
        Err(err) => {
            eprintln!("[dsa-scores-decline] {err}");
            return None;
        }
    };
    Some(out.into_iter().next()?)
}

/// Fused DSA decode attention, stage 2: scores [B*heads,tc] and
/// v [B,heads,tc,256] are F32 -> out [B*heads,256] F32.
pub fn dsa_decode_weighted_v(scores: &Array, v: &Array, stream: &Stream) -> Option<Array> {
    let kernel = dsa_wv_kernel().as_ref()?;
    if scores.dtype() != Dtype::Float32 || v.dtype() != Dtype::Float32 {
        return None;
    }
    if scores.rank() != 2 || v.rank() != 4 {
        return None;
    }
    let rows = scores.dim(0) as usize;
    let tc = scores.dim(1) as usize;
    if v.dim(0) as usize * v.dim(1) as usize != rows || v.dim(2) as usize != tc || v.dim(3) != 256 {
        return None;
    }
    if rows == 0 || tc == 0 {
        return None;
    }
    let nr = Array::from_int(rows as i32);
    let tca = Array::from_int(tc as i32);
    for a in [scores, v] {
        if let Err(err) = a.eval_enqueue() {
            eprintln!("[dsa-wv-decline] {err}");
            return None;
        }
    }
    let inputs: [&Array; 4] = [&nr, &tca, scores, v];
    let outputs = [OutputArg {
        shape: vec![rows as i32, 256],
        dtype: Dtype::Float32,
    }];
    let all: Vec<&Array> = inputs.iter().map(|a| &**a).collect();
    let out = match kernel.apply(
        &all,
        &[],
        ((rows * 256) as i32, 1, 1),
        (1, 1, 1),
        &outputs,
        stream,
    ) {
        Ok(o) => o,
        Err(err) => {
            eprintln!("[dsa-wv-decline] {err}");
            return None;
        }
    };
    Some(out.into_iter().next()?)
}

fn dsa_qk_lowrank_native_kernel() -> &'static Option<MetalKernel> {
    static K: std::sync::OnceLock<Option<MetalKernel>> = std::sync::OnceLock::new();
    K.get_or_init(|| {
        MetalKernel::new_with_math(
            "glm53_dsa_qk_lowrank_native_serial_dot_v2",
            &[
                "n_batches",
                "n_queries",
                "n_heads",
                "group_size",
                "kv_lora_dim",
                "qk_nope",
                "qk_dim",
                "q",
                "kv_weight",
                "kv_scales",
                "kv_biases",
            ],
            &["output"],
            DSA_QK_LOWRANK_NATIVE_SOURCE,
            "",
            true,
            false,
            crate::runtime::Math::CompilerDefault,
        )
        .ok()
    })
}

/// QK projection with the native DS4 schedule available under its diagnostic
/// scope. Queries are [B,H,T,256], output is [B,T,H,512].
pub fn dsa_qk_lowrank_project(
    q: &Array,
    kv_weight: &Array,
    kv_scales: &Array,
    kv_biases: &Array,
    group_size: usize,
    stream: &Stream,
) -> Option<Array> {
    if q.rank() != 4 || q.dim(3) != 256 || q.dtype() != Dtype::Float32 {
        return None;
    }
    let (batches, heads, queries) = (q.dim(0) as usize, q.dim(1) as usize, q.dim(2) as usize);
    if batches == 0
        || heads == 0
        || queries == 0
        || group_size == 0
        || 512 % group_size != 0
        || kv_weight.dtype() != Dtype::Uint32
        || kv_weight.shape() != [(heads * 512) as i32, 128]
        || kv_scales.dtype() != Dtype::Float32
        || kv_biases.dtype() != Dtype::Float32
        || kv_scales.shape() != [(heads * 512) as i32, (512 / group_size) as i32]
        || kv_biases.shape() != kv_scales.shape()
    {
        return None;
    }
    let native =
        GLM53_DSA_NATIVE_OVERRIDE.with(std::cell::Cell::get) && heads == 64 && group_size == 64;
    let kernel = if native {
        dsa_qk_lowrank_native_kernel()
    } else {
        dsa_qk_lowrank_kernel()
    }
    .as_ref()?;
    let mut inputs = vec![
        KernelInput::I32(batches as i32),
        KernelInput::I32(queries as i32),
        KernelInput::I32(heads as i32),
        KernelInput::I32(group_size as i32),
    ];
    if native {
        inputs.extend([
            KernelInput::I32(512),
            KernelInput::I32(256),
            KernelInput::I32(256),
        ]);
    }
    inputs.extend([
        KernelInput::Tensor(&q.t),
        KernelInput::Tensor(&kv_weight.t),
        KernelInput::Tensor(&kv_scales.t),
        KernelInput::Tensor(&kv_biases.t),
    ]);
    let (grid, group) = if native {
        (
            ((heads * batches * 256) as i32, queries as i32, 1),
            (256, 1, 1),
        )
    } else {
        (
            (512 * 32, (queries * 2) as i32, (heads * batches) as i32),
            (32, 2, 1),
        )
    };
    match kernel.apply_with_inline_inputs(
        &inputs,
        &[],
        grid,
        group,
        &[OutputArg {
            shape: vec![batches as i32, queries as i32, heads as i32, 512],
            dtype: Dtype::Float32,
        }],
        stream,
    ) {
        Ok(mut outputs) => outputs.pop(),
        Err(error) => {
            eprintln!("[dsa-qk-lowrank-dispatch] {error}");
            None
        }
    }
}

fn dsa_qk_lowrank_kernel() -> &'static Option<MetalKernel> {
    static K: std::sync::OnceLock<Option<MetalKernel>> = std::sync::OnceLock::new();
    K.get_or_init(|| {
        match MetalKernel::new_with_math_and_threadgroup_memory(
            "glm53_dsa_qk_lowrank_ds4_mv4_v4",
            &[
                "n_batches",
                "n_queries",
                "n_heads",
                "group_size",
                "q",
                "kv_weight",
                "kv_scales",
                "kv_biases",
            ],
            &["output"],
            DSA_QK_LOWRANK_SOURCE,
            EXACT_HEADER,
            true,
            false,
            crate::runtime::Math::CompilerDefault,
            32 * std::mem::size_of::<f32>(),
        ) {
            Ok(kernel) => Some(kernel),
            Err(err) => {
                eprintln!("[dsa-qk-lowrank-compile] {err}");
                None
            }
        }
    })
}

fn dsa_indexed_lora_kernel() -> &'static Option<MetalKernel> {
    static K: std::sync::OnceLock<Option<MetalKernel>> = std::sync::OnceLock::new();
    K.get_or_init(|| {
        match MetalKernel::new_with_math(
            "glm53_dsa_indexed_lora_ds4_default_v4",
            &[
                "n_batches",
                "n_tokens",
                "n_heads",
                "n_selected",
                "selected_valid_prefix",
                "dense_all",
                "tc",
                "scale",
                "q_low",
                "latents",
                "selected",
            ],
            &["lora_out"],
            DSA_INDEXED_LORA_SOURCE,
            EXACT_HEADER,
            true,
            false,
            crate::runtime::Math::CompilerDefault,
        ) {
            Ok(kernel) => Some(kernel),
            Err(err) => {
                eprintln!("[dsa-indexed-lora-compile] {err}");
                None
            }
        }
    })
}

// Keep the complete DS4 ABI: runtime dimensions and no-RoPE branches are
// part of the compiled arithmetic, even for GLM-5.3's fixed geometry.
#[repr(C)]
#[derive(Clone, Copy)]
struct Ds4IndexedLoraArgs {
    n_tokens: u32,
    n_selected: u32,
    cache_cap: u32,
    cache_f16: u32,
    n_head: u32,
    kv_lora_dim: u32,
    qk_nope: u32,
    qk_rope: u32,
    value_dim: u32,
    n_ctx_orig: u32,
    value_row_bytes: u32,
    value_type: u32,
    pos0: u32,
    scale: f32,
    freq_base: f32,
    freq_scale: f32,
    ext_factor: f32,
    attn_factor: f32,
    beta_fast: f32,
    beta_slow: f32,
    head_base: u32,
}

/// DS4's split serial attention fallback for a private MTP window beyond
/// the fused kernel's 2051-row limit. Eight SIMD groups stage 16 F16 rows at
/// a time; scratch remains 16 KiB independent of the selected-window length.
fn dsa_indexed_lora_wide_native(
    q: &Array,
    q_low: &Array,
    compact_latents: &Array,
    selected: &Array,
    scale: f32,
    stream: &Stream,
) -> Option<Array> {
    if q.shape() != [1, 64, 1, 256]
        || q.dtype() != Dtype::Float32
        || q_low.shape() != [1, 1, 64, 512]
        || q_low.dtype() != Dtype::Float32
        || compact_latents.rank() != 3
        || compact_latents.dim(0) != 1
        || compact_latents.dim(2) != 512
        || compact_latents.dtype() != Dtype::Float16
        || selected.rank() != 3
        || selected.shape()[..2] != [1, 1]
        || selected.dtype() != Dtype::Int32
        || selected.dim(2) <= 2051
        || selected.dim(2) > compact_latents.dim(1)
        || !scale.is_finite()
        || scale <= 0.0
    {
        return None;
    }
    for tensor in [q, q_low, compact_latents, selected] {
        if !tensor.t.buffer_and_layout().1.is_contiguous() {
            return None;
        }
        tensor.eval_enqueue().ok()?;
    }
    let runtime = stream.runtime();
    let pipeline = runtime
        .compile_with(
            DSA_INDEXED_LORA_NATIVE_SOURCE,
            "kernel_glm_attention_indexed_batch_lora_group8_vec",
            crate::runtime::Math::CompilerDefault,
        )
        .ok()?;
    let output_buffer = runtime
        .buffer_for_full_gpu_write(64 * 512 * std::mem::size_of::<f32>(), "glm53_mtp_lora_wide")
        .ok()?;
    let output = Array::new(crate::array::Array::from_parts(
        runtime,
        output_buffer.clone(),
        &[1, 1, 64, 512],
        Dtype::Float32,
    ));
    let args = Ds4IndexedLoraArgs {
        n_tokens: 1,
        n_selected: selected.dim(2) as u32,
        cache_cap: compact_latents.dim(1) as u32,
        cache_f16: 1,
        n_head: 64,
        kv_lora_dim: 512,
        qk_nope: 256,
        qk_rope: 0,
        value_dim: 512,
        n_ctx_orig: 0,
        value_row_bytes: 0,
        value_type: 1,
        pos0: 0,
        scale,
        freq_base: 0.0,
        freq_scale: 1.0,
        ext_factor: 0.0,
        attn_factor: 1.0,
        beta_fast: 0.0,
        beta_slow: 0.0,
        head_base: 0,
    };
    {
        let guard = runtime.commands.encoder().ok()?;
        let encoder = guard.encoder();
        encoder.set_pipeline(&pipeline);
        encoder.set_bytes(0, &args);
        // No RoPE rows exist for GLM-5.3. Preserve DS4's argument binding;
        // qk_rope=0 prevents the aliased placeholder at slot 4 from being read.
        for (index, tensor) in [
            (1, q),
            (2, q_low),
            (3, compact_latents),
            (4, compact_latents),
            (5, selected),
        ] {
            let (buffer, layout) = tensor.t.buffer_and_layout();
            encoder.set_input(
                index,
                Some(buffer),
                layout.offset * tensor.dtype().size_of(),
            );
        }
        encoder.set_output(6, Some(&output_buffer), 0);
        encoder.set_threadgroup_memory_length(0, 16 * 512 * std::mem::size_of::<u16>());
        encoder.dispatch_groups((8, 1, 1), (32, 8, 1));
        // DS4 keeps this dispatch and its following value projection in the
        // active serial encoder. Do not commit, wait, or close the encoder.
    }
    Some(output)
}

fn dsa_indexed_decode_native_kernel(capture_lora: bool) -> &'static Option<MetalKernel> {
    static PLAIN: std::sync::OnceLock<Option<MetalKernel>> = std::sync::OnceLock::new();
    static CAPTURE: std::sync::OnceLock<Option<MetalKernel>> = std::sync::OnceLock::new();
    static CAPTURE_SOURCE: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    (if capture_lora { &CAPTURE } else { &PLAIN }).get_or_init(|| {
        let source = if capture_lora {
            CAPTURE_SOURCE.get_or_init(|| DSA_INDEXED_DECODE_NATIVE_SOURCE.replace(
                "// __DSA_FUSED_LORA_CAPTURE__",
                "for (uint k = tid; k < KV_DIM; k += nth) { lora_capture[head * KV_DIM + k] = lora[k]; }",
            )).as_str()
        } else { DSA_INDEXED_DECODE_NATIVE_SOURCE };
        let outputs: &[&str] = if capture_lora { &["output", "lora_capture"] } else { &["output"] };
        MetalKernel::new_with_math_and_threadgroup_memory(
            if capture_lora { "glm53_dsa_indexed_decode_native_capture_v2" } else { "glm53_dsa_indexed_decode_native_v2" },
            &["n_selected", "dense_all", "tc", "group_size", "scale", "n_heads", "kv_lora_dim", "value_dim", "q_low", "latents", "selected", "kv_weight", "kv_scales", "kv_biases"],
            outputs,
            source,
            EXACT_HEADER,
            true,
            false,
            crate::runtime::Math::CompilerDefault,
            (256 + 2051 + 512) * std::mem::size_of::<f32>(),
        ).ok()
    })
}

fn dsa_indexed_decode_fused_kernel() -> &'static Option<MetalKernel> {
    static K: std::sync::OnceLock<Option<MetalKernel>> = std::sync::OnceLock::new();
    K.get_or_init(|| {
        match MetalKernel::new_with_math_and_threadgroup_memory(
            "glm53_dsa_indexed_decode_fused_ds4_affine_v2",
            &[
                "n_selected",
                "dense_all",
                "tc",
                "group_size",
                "scale",
                "q_low",
                "latents",
                "selected",
                "kv_weight",
                "kv_scales",
                "kv_biases",
            ],
            &["output"],
            DSA_INDEXED_DECODE_FUSED_SOURCE,
            EXACT_HEADER,
            true,
            false,
            crate::runtime::Math::CompilerDefault,
            (256 + 2051 + 512) * std::mem::size_of::<f32>(),
        ) {
            Ok(kernel) => Some(kernel),
            Err(err) => {
                eprintln!("[dsa-fused-decode-compile] {err}");
                None
            }
        }
    })
}

fn dsa_indexed_decode_fused_capture_kernel() -> &'static Option<MetalKernel> {
    static SOURCE: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    let source = SOURCE.get_or_init(|| {
        DSA_INDEXED_DECODE_FUSED_SOURCE.replace(
            "// __DSA_FUSED_LORA_CAPTURE__",
            r#"for (uint k = tid; k < KV_DIM; k += nth) {
        lora_capture[head * KV_DIM + k] = lora[k];
    }"#,
        )
    });
    static K: std::sync::OnceLock<Option<MetalKernel>> = std::sync::OnceLock::new();
    K.get_or_init(|| {
        match MetalKernel::new_with_math_and_threadgroup_memory(
            "glm53_dsa_indexed_decode_fused_capture_ds4_affine_v2",
            &[
                "n_selected",
                "dense_all",
                "tc",
                "group_size",
                "scale",
                "q_low",
                "latents",
                "selected",
                "kv_weight",
                "kv_scales",
                "kv_biases",
            ],
            &["output", "lora_capture"],
            source,
            EXACT_HEADER,
            true,
            false,
            crate::runtime::Math::CompilerDefault,
            (256 + 2051 + 512) * std::mem::size_of::<f32>(),
        ) {
            Ok(kernel) => Some(kernel),
            Err(err) => {
                eprintln!("[dsa-fused-decode-capture-compile] {err}");
                None
            }
        }
    })
}

/// DS4's serial indexed-decode schedule for GLM-5.3: one threadgroup per
/// attention head fuses indexed softmax, latent reduction, and MLX-affine
/// value projection. `None` leaves callers free to use the general path.
#[allow(clippy::too_many_arguments)]
fn dsa_indexed_decode_fused(
    q_low: &Array,
    compact_latents: &Array,
    selected: &Array,
    n_selected: usize,
    dense_all: bool,
    kv_weight: &Array,
    kv_scales: &Array,
    kv_biases: &Array,
    group_size: usize,
    scale: f32,
    stream: &Stream,
    capture_lora: bool,
) -> Option<(Array, Option<Array>)> {
    if q_low.shape() != [1, 1, 64, 512]
        || q_low.dtype() != Dtype::Float32
        || compact_latents.rank() != 3
        || compact_latents.dim(0) != 1
        || compact_latents.dim(2) != 512
        || compact_latents.dtype() != Dtype::Float16
        || selected.rank() != 3
        || selected.shape()[..2] != [1, 1]
        || selected.dtype() != Dtype::Int32
        || n_selected == 0
        || n_selected > 2051
        || kv_weight.shape() != [64 * 512, 128]
        || kv_weight.dtype() != Dtype::Uint32
        || group_size == 0
        || 512 % group_size != 0
        || kv_scales.shape() != [64 * 512, (512 / group_size) as i32]
        || kv_biases.shape() != kv_scales.shape()
        || kv_scales.dtype() != Dtype::Float32
        || kv_biases.dtype() != Dtype::Float32
    {
        eprintln!(
            "[dsa-fused-decode-decline] q={:?}/{:?} cache={:?}/{:?} selected={:?}/{:?} n_selected={n_selected} dense={dense_all} weight={:?}/{:?} scales={:?}/{:?} group_size={group_size}",
            q_low.shape(),
            q_low.dtype(),
            compact_latents.shape(),
            compact_latents.dtype(),
            selected.shape(),
            selected.dtype(),
            kv_weight.shape(),
            kv_weight.dtype(),
            kv_scales.shape(),
            kv_scales.dtype()
        );
        return None;
    }
    let q_low = q_low.contiguous().ok()?;
    let compact_latents = compact_latents.contiguous().ok()?;
    let selected = selected.contiguous().ok()?;
    let kv_weight = kv_weight.contiguous().ok()?;
    let kv_scales = kv_scales.contiguous().ok()?;
    let kv_biases = kv_biases.contiguous().ok()?;
    for a in [
        &q_low,
        &compact_latents,
        &selected,
        &kv_weight,
        &kv_scales,
        &kv_biases,
    ] {
        a.eval_enqueue().ok()?;
    }
    let native = GLM53_DSA_NATIVE_OVERRIDE.with(std::cell::Cell::get) && group_size == 64;
    let mut inputs = vec![
        KernelInput::I32(n_selected as i32),
        KernelInput::I32(i32::from(dense_all)),
        KernelInput::I32(compact_latents.dim(1)),
        KernelInput::I32(group_size as i32),
        KernelInput::F32(scale),
    ];
    if native {
        inputs.extend([
            KernelInput::I32(64),
            KernelInput::I32(512),
            KernelInput::I32(256),
        ]);
    }
    inputs.extend([
        KernelInput::Tensor(&q_low.t),
        KernelInput::Tensor(&compact_latents.t),
        KernelInput::Tensor(&selected.t),
        KernelInput::Tensor(&kv_weight.t),
        KernelInput::Tensor(&kv_scales.t),
        KernelInput::Tensor(&kv_biases.t),
    ]);
    let kernel_option = if native {
        dsa_indexed_decode_native_kernel(capture_lora)
    } else if capture_lora {
        dsa_indexed_decode_fused_capture_kernel()
    } else {
        dsa_indexed_decode_fused_kernel()
    };
    let Some(kernel) = kernel_option.as_ref() else {
        eprintln!("[dsa-fused-decode-decline] kernel compilation unavailable");
        return None;
    };
    let outputs = if capture_lora {
        vec![
            OutputArg {
                shape: vec![1, 1, 64, 256],
                dtype: Dtype::Float32,
            },
            OutputArg {
                shape: vec![1, 1, 64, 512],
                dtype: Dtype::Float32,
            },
        ]
    } else {
        vec![OutputArg {
            shape: vec![1, 1, 64, 256],
            dtype: Dtype::Float32,
        }]
    };
    match kernel.apply_with_inline_inputs(
        &inputs,
        &[],
        (64 * 256, 1, 1),
        (256, 1, 1),
        &outputs,
        stream,
    ) {
        Ok(mut outputs) if capture_lora && outputs.len() == 2 => {
            let lora = outputs.pop()?;
            let output = outputs.pop()?;
            Some((output, Some(lora)))
        }
        Ok(mut outputs) if !capture_lora && outputs.len() == 1 => Some((outputs.pop()?, None)),
        Ok(_) => None,
        Err(err) => {
            eprintln!("[dsa-fused-decode-dispatch] {err}");
            None
        }
    }
}

fn dsa_value_project_native_kernel(mma: bool) -> &'static Option<MetalKernel> {
    static SCALAR: std::sync::OnceLock<Option<MetalKernel>> = std::sync::OnceLock::new();
    static MMA: std::sync::OnceLock<Option<MetalKernel>> = std::sync::OnceLock::new();
    (if mma { &MMA } else { &SCALAR }).get_or_init(|| {
        MetalKernel::new_with_math_and_threadgroup_memory(
            if mma {
                "glm53_dsa_value_native_mma_v2"
            } else {
                "glm53_dsa_value_native_serial_dot_v2"
            },
            &[
                "n_rows",
                "n_heads",
                "group_size",
                "kv_lora_dim",
                "value_dim",
                "lora",
                "kv_weight",
                "kv_scales",
                "kv_biases",
            ],
            &["output"],
            if mma {
                DSA_VALUE_PROJECT_NATIVE_MMA_SOURCE
            } else {
                DSA_VALUE_PROJECT_NATIVE_SOURCE
            },
            "#include <metal_simdgroup_matrix>\n",
            true,
            false,
            crate::runtime::Math::CompilerDefault,
            if mma {
                16 * 1024
            } else {
                512 * std::mem::size_of::<f32>()
            },
        )
        .ok()
    })
}

fn dsa_value_project_kernel() -> &'static Option<MetalKernel> {
    static K: std::sync::OnceLock<Option<MetalKernel>> = std::sync::OnceLock::new();
    K.get_or_init(|| {
        match MetalKernel::new_with_math_and_threadgroup_memory(
            "glm53_dsa_value_project_ds4_f32_mv4_v3",
            &[
                "n_rows",
                "n_heads",
                "group_size",
                "lora",
                "kv_weight",
                "kv_scales",
                "kv_biases",
            ],
            &["output"],
            DSA_VALUE_PROJECT_SOURCE,
            EXACT_HEADER,
            true,
            false,
            crate::runtime::Math::CompilerDefault,
            32 * std::mem::size_of::<f32>(),
        ) {
            Ok(kernel) => Some(kernel),
            Err(err) => {
                eprintln!("[dsa-value-project-compile] {err}");
                None
            }
        }
    })
}

fn dsa_prefill_value_slices(tokens: usize, dense_prefix: usize) -> Vec<(usize, usize)> {
    let mut slices = Vec::new();
    let mut start = 0;
    while start < tokens {
        let mut length = (tokens - start).min(2048);
        if start < dense_prefix {
            length = length.min(dense_prefix - start);
        }
        slices.push((start, length));
        start += length;
    }
    slices
}

pub fn dsa_value_project_rows(
    lora: &Array,
    kv_weight: &Array,
    kv_scales: &Array,
    kv_biases: &Array,
    group_size: usize,
    n_heads: usize,
    stream: &Stream,
) -> Option<Array> {
    if n_heads == 0
        || group_size == 0
        || 512 % group_size != 0
        || lora.rank() != 2
        || lora.dtype() != Dtype::Float32
        || lora.dim(1) != 512
        || kv_weight.rank() != 2
        || kv_weight.dtype() != Dtype::Uint32
        || kv_weight.dim(0) as usize != n_heads.checked_mul(512)?
        || kv_weight.dim(1) != 128
        || kv_scales.rank() != 2
        || kv_biases.rank() != 2
        || kv_scales.dtype() != Dtype::Float32
        || kv_biases.dtype() != Dtype::Float32
        || kv_scales.shape() != kv_biases.shape()
        || kv_scales.dim(0) != kv_weight.dim(0)
        || kv_scales.dim(1) as usize != 512usize.checked_div(group_size)?
    {
        return None;
    }
    let rows = lora.dim(0) as usize;
    if rows == 0 || rows % n_heads != 0 {
        return None;
    }
    let rows_i32 = i32::try_from(rows).ok()?;
    let lora = lora.contiguous().ok()?;
    let kv_weight = kv_weight.contiguous().ok()?;
    let kv_scales = kv_scales.contiguous().ok()?;
    let kv_biases = kv_biases.contiguous().ok()?;
    for a in [&lora, &kv_weight, &kv_scales, &kv_biases] {
        a.eval_enqueue().ok()?;
    }
    let tokens = rows / n_heads;
    let native =
        GLM53_DSA_NATIVE_OVERRIDE.with(std::cell::Cell::get) && n_heads == 64 && group_size == 64;
    if native && let Some(prefix) = GLM53_DSA_VALUE_DENSE_PREFIX.with(std::cell::Cell::get) {
        let slices = dsa_prefill_value_slices(tokens, prefix);
        if slices.len() > 1 {
            // DS4 caps each attention/value slice at 2048 tokens and starts
            // a new slice at the dense/sparse boundary. The value kernel's
            // scalar/MMA threshold applies to each resulting slice.
            let _slice = Glm53DsaValueSliceScope::new(0, 0);
            let mut outputs = Vec::with_capacity(slices.len());
            for (start, length) in slices {
                let input = Array::new(
                    lora.t
                        .narrow(0, start.checked_mul(n_heads)?, length.checked_mul(n_heads)?)
                        .ok()?,
                );
                outputs.push(dsa_value_project_rows(
                    &input, &kv_weight, &kv_scales, &kv_biases, group_size, n_heads, stream,
                )?);
            }
            return Array::concatenate(&outputs.iter().collect::<Vec<_>>(), 0).ok();
        }
    }
    let mut inputs = vec![
        KernelInput::I32(rows_i32),
        KernelInput::I32(n_heads as i32),
        KernelInput::I32(group_size as i32),
    ];
    if native {
        inputs.extend([KernelInput::I32(512), KernelInput::I32(256)]);
    }
    inputs.extend([
        KernelInput::Tensor(&lora.t),
        KernelInput::Tensor(&kv_weight.t),
        KernelInput::Tensor(&kv_scales.t),
        KernelInput::Tensor(&kv_biases.t),
    ]);
    let mma = native && tokens >= 32;
    let kernel = if native {
        dsa_value_project_native_kernel(mma)
    } else {
        dsa_value_project_kernel()
    }
    .as_ref()?;
    let (grid, group) = if mma {
        (
            ((tokens.div_ceil(32) * 128) as i32, 4, n_heads as i32),
            (128, 1, 1),
        )
    } else if native {
        (((n_heads * 256) as i32, tokens as i32, 1), (256, 1, 1))
    } else {
        ((256 * 128, rows_i32, 1), (128, 1, 1))
    };
    kernel
        .apply_with_inline_inputs(
            &inputs,
            &[],
            grid,
            group,
            &[OutputArg {
                shape: vec![rows_i32, 256],
                dtype: Dtype::Float32,
            }],
            stream,
        )
        .ok()?
        .into_iter()
        .next()
}

fn dsa_o_project_kernel() -> &'static Option<MetalKernel> {
    static K: std::sync::OnceLock<Option<MetalKernel>> = std::sync::OnceLock::new();
    K.get_or_init(|| {
        MetalKernel::new_with_math(
            "glm53_dsa_o_project_ds4_mlx_affine_mv_v1",
            &[
                "n_rows",
                "group_size",
                "input",
                "weight",
                "scales",
                "biases",
            ],
            &["output"],
            DSA_O_PROJECT_SOURCE,
            EXACT_HEADER,
            true,
            false,
            crate::runtime::Math::CompilerDefault,
        )
        .ok()
    })
}

/// GLM DSA output projection using DS4's raw MLX-affine F32 matvec schedule.
pub fn dsa_o_project(
    input: &Array,
    weight: &Array,
    scales: &Array,
    biases: &Array,
    group_size: usize,
    stream: &Stream,
) -> Option<Array> {
    const IN: usize = 16_384;
    const OUT: usize = 4_096;
    if group_size == 0 || IN % group_size != 0 || i32::try_from(group_size).is_err() {
        return None;
    }
    if input.rank() != 2
        || input.dim(1) != IN as i32
        || input.dtype() != Dtype::Float32
        || weight.rank() != 2
        || weight.shape() != [OUT as i32, (IN / 4) as i32]
        || weight.dtype() != Dtype::Uint32
        || scales.rank() != 2
        || biases.rank() != 2
        || scales.shape() != biases.shape()
        || scales.shape() != [OUT as i32, (IN / group_size) as i32]
        || scales.dtype() != Dtype::Float32
        || biases.dtype() != Dtype::Float32
    {
        return None;
    }
    let rows = input.dim(0) as usize;
    if rows == 0 {
        return None;
    }
    let rows_i32 = i32::try_from(rows).ok()?;
    let input = input.contiguous().ok()?;
    let weight = weight.contiguous().ok()?;
    let scales = scales.contiguous().ok()?;
    let biases = biases.contiguous().ok()?;
    for a in [&input, &weight, &scales, &biases] {
        a.eval_enqueue().ok()?;
    }
    let inputs = [
        KernelInput::I32(rows_i32),
        KernelInput::I32(group_size as i32),
        KernelInput::Tensor(&input.t),
        KernelInput::Tensor(&weight.t),
        KernelInput::Tensor(&scales.t),
        KernelInput::Tensor(&biases.t),
    ];
    dsa_o_project_kernel()
        .as_ref()?
        .apply_with_inline_inputs(
            &inputs,
            &[],
            ((OUT * 32) as i32, rows_i32, 1),
            (32, 1, 1),
            &[OutputArg {
                shape: vec![rows_i32, OUT as i32],
                dtype: Dtype::Float32,
            }],
            stream,
        )
        .ok()?
        .into_iter()
        .next()
}

/// GLM DSA attention in absorbed MLA form, reading the packed MLX affine
/// `kv_b` weights directly. `q=[B,heads,T,256]`, `kv_weight=[32768,128]`
/// packed U32 (8-bit, 512 inputs), scales/biases `[32768,8]` F32,
/// `compact_latents=[B,tc,512]` F16, and `selected=[B,T,M]` absolute token
/// ids. Returns `[B,heads,T,256]` F32. Native single-row decode fuses attention
/// and value projection through 2051 selected rows; longer private MTP windows
/// use DS4's staged indexed-lora kernel followed by its value projection.
pub fn dsa_absorbed_attention(
    q: &Array,
    kv_weight: &Array,
    kv_scales: &Array,
    kv_biases: &Array,
    group_size: usize,
    compact_latents: &Array,
    selected: &Array,
    scale: f32,
    stream: &Stream,
) -> Option<Array> {
    dsa_absorbed_attention_stages_inner(
        q,
        kv_weight,
        kv_scales,
        kv_biases,
        group_size,
        compact_latents,
        Some(selected),
        0,
        scale,
        stream,
        false,
        false,
        false,
        false,
    )
    .map(|(out, _, _)| out)
}

/// Explicit fused one-token decode path. The fused schedule is separate from
/// DS4's staged MLX-affine path and must only be selected by callers that have
/// validated its numerics for their weight representation.
#[allow(clippy::too_many_arguments)]
pub fn dsa_absorbed_attention_decode(
    q: &Array,
    kv_weight: &Array,
    kv_scales: &Array,
    kv_biases: &Array,
    group_size: usize,
    compact_latents: &Array,
    selected: Option<&Array>,
    selected_valid_prefix: usize,
    scale: f32,
    stream: &Stream,
) -> Option<Array> {
    dsa_absorbed_attention_stages_inner(
        q,
        kv_weight,
        kv_scales,
        kv_biases,
        group_size,
        compact_latents,
        selected,
        selected_valid_prefix,
        scale,
        stream,
        false,
        true,
        false,
        false,
    )
    .map(|(out, _, _)| out)
}

/// Capture the intermediate weighted latent from the native fused decode
/// kernel. This diagnostic entry point adds one output buffer and is never
/// used by production inference.
#[allow(clippy::too_many_arguments)]
pub fn dsa_absorbed_attention_decode_capture(
    q: &Array,
    kv_weight: &Array,
    kv_scales: &Array,
    kv_biases: &Array,
    group_size: usize,
    compact_latents: &Array,
    selected: Option<&Array>,
    selected_valid_prefix: usize,
    scale: f32,
    stream: &Stream,
) -> Option<(Array, Array)> {
    dsa_absorbed_attention_stages_inner(
        q,
        kv_weight,
        kv_scales,
        kv_biases,
        group_size,
        compact_latents,
        selected,
        selected_valid_prefix,
        scale,
        stream,
        false,
        true,
        true,
        false,
    )
    .and_then(|(out, _, fused_lora)| Some((out, fused_lora?)))
}

/// DSA decode variant for selectors that guarantee a live-cache prefix in
/// every selected row. Entries at and after `selected_valid_prefix` remain
/// bounds-checked, so the causal-tail padding can use `-1`/`UINT32_MAX`.
/// Prefill and arbitrary selections should use [`dsa_absorbed_attention`].
#[allow(clippy::too_many_arguments)]
pub fn dsa_absorbed_attention_with_valid_prefix(
    q: &Array,
    kv_weight: &Array,
    kv_scales: &Array,
    kv_biases: &Array,
    group_size: usize,
    compact_latents: &Array,
    selected: &Array,
    selected_valid_prefix: usize,
    scale: f32,
    stream: &Stream,
    prefill_encoder_boundaries: bool,
) -> Option<Array> {
    dsa_absorbed_attention_stages_inner(
        q,
        kv_weight,
        kv_scales,
        kv_biases,
        group_size,
        compact_latents,
        Some(selected),
        selected_valid_prefix,
        scale,
        stream,
        false,
        false,
        false,
        prefill_encoder_boundaries,
    )
    .map(|(out, _, _)| out)
}

/// Dense-equivalent one-token DSA decode with contiguous cache rows supplied
/// implicitly. This avoids uploading `[0..tc)` as a selected-ID tensor.
#[allow(clippy::too_many_arguments)]
pub fn dsa_absorbed_attention_dense(
    q: &Array,
    kv_weight: &Array,
    kv_scales: &Array,
    kv_biases: &Array,
    group_size: usize,
    compact_latents: &Array,
    scale: f32,
    stream: &Stream,
) -> Option<Array> {
    dsa_absorbed_attention_stages_inner(
        q,
        kv_weight,
        kv_scales,
        kv_biases,
        group_size,
        compact_latents,
        None,
        0,
        scale,
        stream,
        false,
        false,
        false,
        false,
    )
    .map(|(out, _, _)| out)
}

/// Dense compact-cache attention with QK-lowrank and attention-weighted latent
/// intermediates returned for non-perturbing differential capture. At short
/// windows this uses the same dispatch sequence as `dsa_absorbed_attention_dense`;
/// the capture flag only prevents the long-window fused decode branch.
#[allow(clippy::too_many_arguments)]
pub fn dsa_absorbed_attention_dense_stages(
    q: &Array,
    kv_weight: &Array,
    kv_scales: &Array,
    kv_biases: &Array,
    group_size: usize,
    compact_latents: &Array,
    scale: f32,
    stream: &Stream,
) -> Option<(Array, Array, Array)> {
    dsa_absorbed_attention_stages_inner(
        q,
        kv_weight,
        kv_scales,
        kv_biases,
        group_size,
        compact_latents,
        None,
        0,
        scale,
        stream,
        true,
        false,
        false,
        false,
    )
    .and_then(|(out, q_low, lora)| Some((out, q_low?, lora?)))
}

/// Same DSA path with the QK low-rank and attention-weighted latent outputs
/// returned for differential capture. Runtime callers should use
/// `dsa_absorbed_attention` unless they need those intermediate tensors.
#[allow(clippy::too_many_arguments)]
pub fn dsa_absorbed_attention_stages(
    q: &Array,
    kv_weight: &Array,
    kv_scales: &Array,
    kv_biases: &Array,
    group_size: usize,
    compact_latents: &Array,
    selected: &Array,
    scale: f32,
    stream: &Stream,
    prefill_encoder_boundaries: bool,
) -> Option<(Array, Array, Array)> {
    dsa_absorbed_attention_stages_inner(
        q,
        kv_weight,
        kv_scales,
        kv_biases,
        group_size,
        compact_latents,
        Some(selected),
        0,
        scale,
        stream,
        true,
        false,
        false,
        prefill_encoder_boundaries,
    )
    .and_then(|(out, q_low, lora)| Some((out, q_low?, lora?)))
}

/// DS4's dense-causal FlashAttention prefill path for MLX-affine DSA V-B.
/// It keeps the query projection and F16 compact cache separate, runs the
/// reference FlashAttention schedule for attention-lora, then applies the
/// same split affine value projection as DS4.
#[allow(clippy::too_many_arguments)]
pub fn dsa_absorbed_attention_flash_causal_prefill_stages(
    q: &Array,
    kv_weight: &Array,
    kv_scales: &Array,
    kv_biases: &Array,
    group_size: usize,
    compact_latents: &Array,
    q_row0: usize,
    scale: f32,
    stream: &Stream,
    prefill_encoder_boundaries: bool,
) -> Option<(Array, Array, Array)> {
    let batches = q.dim(0) as usize;
    let n_heads = q.dim(1) as usize;
    let n_queries = q.dim(2) as usize;
    if batches != 1 || n_heads != 64 || n_queries == 0 || q.dim(3) != 256 {
        return None;
    }
    let q_low = dsa_qk_lowrank_project(q, kv_weight, kv_scales, kv_biases, group_size, stream)?;
    if prefill_encoder_boundaries {
        stream.runtime().commands.flush_encoder();
    }
    let lora = Array::new(
        crate::jit::ds4_flash_attention_prefill_raw_heads(
            &q_low.t,
            &compact_latents.t,
            q_row0,
            scale,
        )
        .ok()?,
    );
    let n_rows = batches.checked_mul(n_queries)?.checked_mul(n_heads)?;
    let lora_rows = lora.reshape(&[n_rows as i32, 512]).ok()?;
    let projected = dsa_value_project_rows(
        &lora_rows, kv_weight, kv_scales, kv_biases, group_size, n_heads, stream,
    )?
    .reshape(&[batches as i32, n_queries as i32, n_heads as i32, 256])
    .ok()?;
    if prefill_encoder_boundaries {
        stream.runtime().commands.flush_encoder();
    }
    let output = projected
        .transpose_axes(&[0, 2, 1, 3])
        .and_then(|value| value.contiguous())
        .ok()?;
    Some((output, q_low, lora))
}

#[allow(clippy::too_many_arguments)]
fn dsa_absorbed_attention_stages_inner(
    q: &Array,
    kv_weight: &Array,
    kv_scales: &Array,
    kv_biases: &Array,
    group_size: usize,
    compact_latents: &Array,
    selected: Option<&Array>,
    selected_valid_prefix: usize,
    scale: f32,
    stream: &Stream,
    capture_stages: bool,
    force_fused_decode: bool,
    capture_fused_lora: bool,
    prefill_encoder_boundaries: bool,
) -> Option<(Array, Option<Array>, Option<Array>)> {
    if q.rank() != 4
        || compact_latents.rank() != 3
        || selected.is_some_and(|selected| selected.rank() != 3)
    {
        return None;
    }
    let batches = q.dim(0) as usize;
    let heads = q.dim(1) as usize;
    let n_queries = q.dim(2) as usize;
    let tc = compact_latents.dim(1) as usize;
    let dense_all = selected.is_none();
    let requested_selected = selected.map_or(tc, |selected| selected.dim(2) as usize);
    if q.dim(3) != 256
        || kv_weight.shape() != [(heads * 512) as i32, 128]
        || kv_weight.dtype() != Dtype::Uint32
        || group_size == 0
        || 512 % group_size != 0
        || kv_scales.shape() != [(heads * 512) as i32, (512 / group_size) as i32]
        || kv_biases.shape() != kv_scales.shape()
        || compact_latents.dim(0) != batches as i32
        || selected.is_some_and(|selected| {
            selected.dim(0) != batches as i32 || selected.dim(1) != n_queries as i32
        })
        || compact_latents.dim(2) != 512
        || selected.is_some_and(|selected| selected.dtype() != Dtype::Int32)
        || batches == 0
        || n_queries == 0
        || heads == 0
        || requested_selected == 0
        || tc == 0
        || selected_valid_prefix > requested_selected
    {
        return None;
    }
    if q.dtype() != Dtype::Float32
        || kv_scales.dtype() != Dtype::Float32
        || kv_biases.dtype() != Dtype::Float32
        || compact_latents.dtype() != Dtype::Float16
    {
        return None;
    }
    let q = q.contiguous().ok()?;
    let kv_weight = kv_weight.contiguous().ok()?;
    let kv_scales = kv_scales.contiguous().ok()?;
    let kv_biases = kv_biases.contiguous().ok()?;
    let compact_latents = compact_latents.contiguous().ok()?;
    let selected = if let Some(selected) = selected {
        selected.contiguous().ok()?
    } else {
        static DENSE_ID_DUMMY: std::sync::OnceLock<Array> = std::sync::OnceLock::new();
        DENSE_ID_DUMMY
            .get_or_init(|| Array::from_slice(&[0i32; 8], &[1, 1, 8]))
            .clone()
    };
    for a in [
        &q,
        &kv_weight,
        &kv_scales,
        &kv_biases,
        &compact_latents,
        &selected,
    ] {
        a.eval_enqueue().ok()?;
    }
    // DS4 switches a long private MTP window to its indexed batch-lora
    // kernel with n_tokens=1. Keep the already admitted shorter/batched
    // routes unchanged and reject unrelated unsupported wide layouts.
    let wide_mtp_window = requested_selected > 2051;
    if wide_mtp_window
        && (batches != 1
            || n_queries != 1
            || heads != 64
            || dense_all
            || requested_selected > tc
            || selected_valid_prefix != 0
            || prefill_encoder_boundaries)
    {
        return None;
    }

    // MLX places input buffers smaller than eight elements in constant
    // address space. The indexed-lora kernel reads selected ids through a
    // device pointer, so pad short dense/MTP windows only enough to reach the
    // eight-element threshold. The kernel skips invalid rows without changing
    // the online-softmax state.
    let selection_rows = batches * n_queries;
    let selection_elements = selection_rows * requested_selected;
    let selected = if !dense_all && selection_elements < 8 {
        let padding_per_row = (8 - selection_elements).div_ceil(selection_rows);
        let padding_values = vec![-1i32; selection_rows * padding_per_row];
        let padding = Array::from_slice(
            &padding_values,
            &[batches as i32, n_queries as i32, padding_per_row as i32],
        );
        Array::concatenate(&[&selected, &padding], 2)
            .and_then(|padded| padded.contiguous())
            .ok()?
    } else {
        selected
    };
    let n_selected = if dense_all {
        tc
    } else {
        selected.dim(2) as usize
    };

    let q_low = dsa_qk_lowrank_project(&q, &kv_weight, &kv_scales, &kv_biases, group_size, stream)?;
    q_low.eval_enqueue().ok()?;
    // DS4 indexed prefill closes the encoder after qk_low and again after
    // attention plus value projection. Both remain in the same command buffer.
    if prefill_encoder_boundaries {
        stream.runtime().commands.flush_encoder();
    }

    let native_decode = GLM53_DSA_NATIVE_OVERRIDE.with(std::cell::Cell::get)
        && !prefill_encoder_boundaries
        && group_size == 64;
    if batches == 1
        && n_queries == 1
        && heads == 64
        && !wide_mtp_window
        && (native_decode || (!capture_stages && force_fused_decode))
    {
        let (projected, fused_lora) = dsa_indexed_decode_fused(
            &q_low,
            &compact_latents,
            &selected,
            requested_selected,
            dense_all,
            &kv_weight,
            &kv_scales,
            &kv_biases,
            group_size,
            scale,
            stream,
            capture_fused_lora || (native_decode && capture_stages),
        )?;
        let output = projected
            .transpose_axes(&[0, 2, 1, 3])
            .and_then(|out| out.contiguous())
            .ok()?;
        return Some((
            output,
            if capture_stages { Some(q_low) } else { None },
            fused_lora,
        ));
    }

    let lora = if wide_mtp_window {
        dsa_indexed_lora_wide_native(&q, &q_low, &compact_latents, &selected, scale, stream)?
    } else {
        let lora_inputs = [
            KernelInput::I32(batches as i32),
            KernelInput::I32(n_queries as i32),
            KernelInput::I32(heads as i32),
            KernelInput::I32(n_selected as i32),
            KernelInput::I32(selected_valid_prefix as i32),
            KernelInput::I32(i32::from(dense_all)),
            KernelInput::I32(tc as i32),
            KernelInput::F32(scale),
            KernelInput::Tensor(&q_low.t),
            KernelInput::Tensor(&compact_latents.t),
            KernelInput::Tensor(&selected.t),
        ];
        let lora_kernel = dsa_indexed_lora_kernel().as_ref()?;
        match lora_kernel.apply_with_inline_inputs(
            &lora_inputs,
            &[],
            (
                heads.div_ceil(8).saturating_mul(256) as i32,
                n_queries as i32,
                batches as i32,
            ),
            (256, 1, 1),
            &[OutputArg {
                shape: vec![batches as i32, n_queries as i32, heads as i32, 512],
                dtype: Dtype::Float32,
            }],
            stream,
        ) {
            Ok(mut out) => out.pop()?,
            Err(err) => {
                eprintln!("[dsa-indexed-lora-dispatch] {err}");
                return None;
            }
        }
    };
    lora.eval_enqueue().ok()?;

    let n_rows = batches.checked_mul(n_queries)?.checked_mul(heads)?;
    let n_rows_i32 = i32::try_from(n_rows).ok()?;
    let lora_rows = lora.reshape(&[n_rows_i32, 512]).ok()?;
    let projected = dsa_value_project_rows(
        &lora_rows, &kv_weight, &kv_scales, &kv_biases, group_size, heads, stream,
    )?
    .reshape(&[batches as i32, n_queries as i32, heads as i32, 256])
    .ok()?;
    if prefill_encoder_boundaries {
        stream.runtime().commands.flush_encoder();
    }
    let output = projected
        .transpose_axes(&[0, 2, 1, 3])
        .and_then(|out| out.contiguous())
        .ok()?;
    Some((output, Some(q_low), Some(lora)))
}

fn kda_kernel() -> &'static Option<MetalKernel> {
    static K: std::sync::OnceLock<Option<MetalKernel>> = std::sync::OnceLock::new();
    K.get_or_init(|| {
        MetalKernel::new_with_math(
            "kda_recurrence_ds4_default_v3",
            &["q", "k", "v", "g", "beta", "state_in", "T"],
            &["y", "state_out"],
            KDA_RECURRENCE_SOURCE,
            EXACT_HEADER,
            true,
            false,
            crate::runtime::Math::CompilerDefault,
        )
        .ok()
    })
}

/// The vectorized-gate delta-rule recurrence (Metal kernel). `g` is
/// [B, T, Hv, Dk] (per-Dk decay), `state` is [B, Hv, Dv, Dk] F32 in/out.
/// Returns `(y [B,T,Hv,Dv], state)`. `None` = kernel declined.
#[allow(clippy::too_many_arguments)]
pub fn kda_recurrence(
    q: &Array,
    k: &Array,
    v: &Array,
    g: &Array,
    beta: &Array,
    state: &Array,
    capture: bool,
    stream: &Stream,
) -> Option<(Array, Array)> {
    let kernel = kda_kernel().as_ref()?;

    let b = k.dim(0);
    let t = k.dim(1);
    let hk = k.dim(2);
    let dk = k.dim(3);
    let hv = v.dim(2);
    let dv = v.dim(3);
    let input_type = q.dtype();
    let state_type = state.dtype();

    let t_scalar = crate::models::gdn::t_scalar_pub(t);
    let inputs: [&Array; 7] = [q, k, v, g, beta, state, &t_scalar];
    let template = [
        TemplateArg::Dtype("InT", input_type),
        TemplateArg::Dtype("StT", state_type),
        TemplateArg::Int("Dk", dk),
        TemplateArg::Int("Dv", dv),
        TemplateArg::Int("Hk", hk),
        TemplateArg::Int("Hv", hv),
        TemplateArg::Bool("CAPTURE", capture),
    ];
    let state_shape = if capture {
        vec![b * t, hv, dv, dk]
    } else {
        state.shape().to_vec()
    };
    let outputs = [
        OutputArg {
            shape: vec![b, t, hv, dv],
            dtype: input_type,
        },
        OutputArg {
            shape: state_shape,
            dtype: state_type,
        },
    ];
    let out = match kernel.apply(
        &inputs,
        &template,
        (32, dv, b * hv),
        (32, 4, 1),
        &outputs,
        stream,
    ) {
        Ok(o) => o,
        Err(e) => {
            eprintln!("[kda-decline] {e}");
            return None;
        }
    };
    let mut it = out.into_iter();
    let y = it.next()?;
    let state_out = it.next()?;
    Some((y, state_out))
}

#[cfg(test)]
#[path = "tests.rs"]
mod tests;

#[cfg(test)]
#[path = "tests_q2_decode.rs"]
mod tests_q2_decode;
