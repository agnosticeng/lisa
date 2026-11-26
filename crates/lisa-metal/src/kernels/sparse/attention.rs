//! Attention operations with the pinned reference arithmetic.
use super::indexing::{indexer_topk_merge_kernel, indexer_topk_sort_kernel};
use super::moe::moe_gather_header;
use super::*;

pub(super) const DSA_INDEXED_LORA_SOURCE: &crate::shader_pack::Source =
    crate::kernel_source!("sparse_attention/dsa_indexed_lora.metal");
pub(super) const DSA_INDEXED_LORA_NATIVE_SOURCE: &crate::shader_pack::Source =
    crate::kernel_source!("sparse_attention/dsa_indexed_lora_native.metal");
pub(super) const DSA_INDEXED_DECODE_FUSED_SOURCE: &crate::shader_pack::Source =
    crate::kernel_source!("sparse_attention/dsa_indexed_decode_fused.metal");
pub(super) const DSA_INDEXED_DECODE_NATIVE_SOURCE: &crate::shader_pack::Source =
    crate::kernel_source!("sparse_attention/dsa_indexed_decode_native.metal");
pub(super) const DSA_VALUE_PROJECT_SOURCE: &crate::shader_pack::Source =
    crate::kernel_source!("sparse_attention/dsa_value_project.metal");
pub(super) const DSA_QK_LOWRANK_NATIVE_SOURCE: &crate::shader_pack::Source =
    crate::kernel_source!("sparse_attention/dsa_qk_lowrank_native.metal");
pub(super) const DSA_VALUE_PROJECT_NATIVE_SOURCE: &crate::shader_pack::Source =
    crate::kernel_source!("sparse_attention/dsa_value_project_native.metal");
pub(super) const DSA_VALUE_PROJECT_NATIVE_MMA_SOURCE: &crate::shader_pack::Source =
    crate::kernel_source!("sparse_attention/dsa_value_project_native_mma.metal");
pub(super) const DSA_O_PROJECT_SOURCE: &crate::shader_pack::Source =
    crate::kernel_source!("sparse_attention/dsa_o_project.metal");

/// Select rowwise descending top-k pool ids with the reference's block-sort plus merge
/// algorithm. `scores` is contiguous F32 `[rows, pools]`; output is I32
/// `[rows, top_k]`. This avoids globally sorting the full sparse-attention
/// history for every query row.
pub fn dsa_indexer_topk(scores: &Array, top_k: usize, stream: &Stream) -> Option<Array> {
    dsa_indexer_topk_with_shuffle(scores, top_k, false, stream)
}

/// The reference's blockwise indexer top-k with SIMD shuffle stages for the sub-32
/// bitonic network. The comparator and tie order are the same as the shared
/// threadgroup-memory implementation.
pub fn dsa_indexer_topk_shuffle(scores: &Array, top_k: usize, stream: &Stream) -> Option<Array> {
    dsa_indexer_topk_with_shuffle(scores, top_k, true, stream)
}

pub(super) fn dsa_indexer_topk_with_shuffle(
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

    // The reference the reference Metal path uses a 1024-entry block on Apple Silicon.
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
        let merge_threads = run_len.clamp(1, 512);
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
pub(super) const DSA_QKV_NORM_ROWS_SOURCE: &crate::shader_pack::Source =
    crate::kernel_source!("sparse_attention/dsa_qkv_rms_norm_rows.metal");
pub(super) const DSA_QKV_PROJECT_PAIR_SOURCE: &crate::shader_pack::Source =
    crate::kernel_source!("sparse_attention/dsa_qkv_project_pair.metal");

pub(super) fn dsa_qkv_norm_rows_kernel() -> &'static Option<MetalKernel> {
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

/// The reference's fused DSA q/kv normalization, retaining its q-sized threadgroup for
/// both branches. The GLM q-rank has 384 float4 lanes; the reference uses those lanes
/// when reducing the shorter 512-value KV row as well.
pub fn dsa_qkv_rms_norm_rows(
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

pub(super) fn dsa_qkv_project_pair_kernel() -> &'static Option<MetalKernel> {
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
/// the reference's F32 affine matvec order; for 2..8 rows it uses the reference's small-batch row
/// grouping and shuffle reduction.
pub fn dsa_qkv_project_pair(
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

pub(super) const DSA_SCORES_SOURCE: &crate::shader_pack::Source =
    crate::kernel_source!("sparse_attention/dsa_scores.metal");
pub(super) const DSA_WV_SOURCE: &crate::shader_pack::Source =
    crate::kernel_source!("sparse_attention/dsa_wv.metal");
pub(super) const DSA_QK_LOWRANK_SOURCE: &crate::shader_pack::Source =
    crate::kernel_source!("sparse_attention/dsa_qk_lowrank.metal");

pub(super) fn dsa_scores_kernel() -> &'static Option<MetalKernel> {
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

pub(super) fn dsa_wv_kernel() -> &'static Option<MetalKernel> {
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

/// Fused DSA decode attention, stage 1: q `[B,heads,256]`, k
/// `[B,heads,tc,256]`, keep `[B,tc]` are F32. Returns F32 scores.
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
    out.into_iter().next()
}

/// Fused DSA decode attention, stage 2: scores `[B*heads,tc]` and
/// v `[B,heads,tc,256]` are F32 -> out `[B*heads,256]` F32.
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
    out.into_iter().next()
}

pub(super) fn dsa_qk_lowrank_native_kernel() -> &'static Option<MetalKernel> {
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

/// QK projection with the native reference schedule available under its diagnostic
/// scope. Queries are `[B,H,T,256]`, output is `[B,T,H,512]`.
pub fn dsa_qk_lowrank_project(
    q: &Array,
    kv_weight: &Array,
    kv_scales: &Array,
    kv_biases: &Array,
    group_size: usize,
    stream: &Stream,
    native: bool,
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
    if native && (heads != 64 || group_size != 64) {
        return None;
    }
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

pub(super) fn dsa_qk_lowrank_kernel() -> &'static Option<MetalKernel> {
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

pub(super) fn dsa_indexed_lora_kernel() -> &'static Option<MetalKernel> {
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

// Keep the complete reference ABI: runtime dimensions and no-RoPE branches are
// part of the compiled arithmetic, even for GLM-5.3's fixed geometry.
#[repr(C)]
#[derive(Clone, Copy)]
pub(super) struct Ds4IndexedLoraArgs {
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

/// The reference's split serial attention fallback for a private MTP window beyond
/// the fused kernel's 2051-row limit. Eight SIMD groups stage 16 F16 rows at
/// a time; scratch remains 16 KiB independent of the selected-window length.
pub fn dsa_indexed_lora_wide_native(
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
        // No RoPE rows exist for GLM-5.3. Preserve the reference's argument binding;
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
        // The reference keeps this dispatch and its following value projection in the
        // active serial encoder. Do not commit, wait, or close the encoder.
    }
    Some(output)
}

pub(super) fn dsa_indexed_decode_native_kernel(capture_lora: bool) -> &'static Option<MetalKernel> {
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

pub fn dsa_indexed_decode_fused_kernel() -> &'static Option<MetalKernel> {
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

pub fn dsa_indexed_decode_fused_capture_kernel() -> &'static Option<MetalKernel> {
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

/// The reference's serial indexed-decode schedule for GLM-5.3: one threadgroup per
/// attention head fuses indexed softmax, latent reduction, and MLX-affine
/// value projection. `None` leaves callers free to use the general path.
#[allow(clippy::too_many_arguments)]
pub fn dsa_indexed_decode_fused(
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
    native: bool,
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
    if native && group_size != 64 {
        return None;
    }
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

pub(super) fn dsa_value_project_native_kernel(mma: bool) -> &'static Option<MetalKernel> {
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

pub(super) fn dsa_value_project_kernel() -> &'static Option<MetalKernel> {
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

pub fn dsa_value_project_rows(
    lora: &Array,
    kv_weight: &Array,
    kv_scales: &Array,
    kv_biases: &Array,
    group_size: usize,
    n_heads: usize,
    stream: &Stream,
    native: bool,
    mma: bool,
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
    if rows == 0 || !rows.is_multiple_of(n_heads) {
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
    if native && (n_heads != 64 || group_size != 64) {
        return None;
    }
    if mma && !native {
        return None;
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

pub(super) fn dsa_o_project_kernel() -> &'static Option<MetalKernel> {
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

/// GLM DSA output projection using the reference's raw MLX-affine F32 matvec schedule.
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
    if group_size == 0 || !IN.is_multiple_of(group_size) || i32::try_from(group_size).is_err() {
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

/// Indexed attention over a compact latent cache. The caller selects the schedule.
pub fn dsa_indexed_lora_short(
    q_low: &Array,
    compact_latents: &Array,
    selected: &Array,
    n_selected: usize,
    selected_valid_prefix: usize,
    dense_all: bool,
    scale: f32,
    stream: &Stream,
) -> Option<Array> {
    if q_low.rank() != 4
        || q_low.dim(3) != 512
        || q_low.dtype() != Dtype::Float32
        || compact_latents.rank() != 3
        || compact_latents.dim(2) != 512
        || compact_latents.dtype() != Dtype::Float16
        || selected.rank() != 3
        || selected.dtype() != Dtype::Int32
    {
        return None;
    }
    let (batches, n_queries, heads) = (
        q_low.dim(0) as usize,
        q_low.dim(1) as usize,
        q_low.dim(2) as usize,
    );
    let tc = compact_latents.dim(1) as usize;
    if batches == 0
        || n_queries == 0
        || heads == 0
        || tc == 0
        || n_selected == 0
        || selected_valid_prefix > n_selected
        || compact_latents.dim(0) != batches as i32
        || (!dense_all && selected.shape()[..2] != [batches as i32, n_queries as i32])
        || (!dense_all && (selected.dim(2) as usize) < n_selected)
    {
        return None;
    }
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
        Ok(mut out) => out.pop(),
        Err(err) => {
            eprintln!("[dsa-indexed-lora-dispatch] {err}");
            None
        }
    }
}
