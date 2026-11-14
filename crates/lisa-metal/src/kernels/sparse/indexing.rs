//! Indexing operations with the pinned DS4 arithmetic.
use super::*;

pub(super) const INDEXER_TOPK_SORT_SOURCE: &crate::shader_pack::Source =
    crate::kernel_source!("glm53/indexer_topk_sort.metal");
pub(super) const INDEXER_TOPK_MERGE_SOURCE: &crate::shader_pack::Source =
    crate::kernel_source!("glm53/indexer_topk_merge.metal");
pub(super) const INDEXER_EXPAND_SELECTION_SOURCE: &crate::shader_pack::Source =
    crate::kernel_source!("glm53/indexer_expand_selection.metal");

pub(super) fn indexer_topk_sort_kernel() -> &'static Option<MetalKernel> {
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

pub(super) fn indexer_topk_merge_kernel() -> &'static Option<MetalKernel> {
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

pub(super) fn indexer_expand_selection_kernel() -> &'static Option<MetalKernel> {
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

pub(super) fn glm53_expand_pool_selection_mode(
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

pub(super) const INDEXER_POOL_SOURCE: &crate::shader_pack::Source =
    crate::kernel_source!("glm53/indexer_pool.metal");

pub(super) fn indexer_pool_kernel() -> &'static Option<MetalKernel> {
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
pub(super) const INDEXER_SCORES_BATCH_SOURCE: &crate::shader_pack::Source =
    crate::kernel_source!("glm53/indexer_scores_batch.metal");
pub(super) const INDEXER_SCORES_DECODE_SOURCE: &crate::shader_pack::Source =
    crate::kernel_source!("glm53/indexer_scores_decode.metal");
pub(super) const INDEXER_SCORES_DECODE_ROWS4_SOURCE: &crate::shader_pack::Source =
    crate::kernel_source!("glm53/indexer_scores_decode_rows4.metal");
pub(super) const INDEXER_SCORES_TILED_SOURCE: &crate::shader_pack::Source =
    crate::kernel_source!("glm53/indexer_scores_tiled.metal");

pub(super) fn indexer_scores_batch_kernel() -> &'static Option<MetalKernel> {
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

pub(super) fn indexer_scores_decode_kernel() -> &'static Option<MetalKernel> {
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

pub(super) fn indexer_scores_decode_rows4_kernel() -> &'static Option<MetalKernel> {
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

pub(super) fn indexer_scores_tiled_kernel() -> &'static Option<MetalKernel> {
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

pub(super) fn glm53_indexer_scores_with_decode_rows4(
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