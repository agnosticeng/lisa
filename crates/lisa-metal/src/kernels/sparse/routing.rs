//! Routing operations with the pinned reference arithmetic.
use super::*;

pub(super) const ROUTER_SELECT_SOURCE: &crate::shader_pack::Source =
    crate::kernel_source!("moe/router_select.metal");
pub(super) const ROUTER_MV_SOURCE: &crate::shader_pack::Source =
    crate::kernel_source!("moe/router_mv.metal");
pub(super) const ROUTER_MV_SMALL_SOURCE: &crate::shader_pack::Source =
    crate::kernel_source!("moe/router_mv_small.metal");

pub(super) fn router_mv_kernel() -> &'static Option<MetalKernel> {
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

pub(super) fn router_mv_small_kernel() -> &'static Option<MetalKernel> {
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

pub(super) fn router_matvec_f32_small(
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

/// The reference's F32 dense matvec lanes used for GLM router projection.
/// `weights` is row-major `[out, in]`; `x` is `[n_tokens, in]`. The reference uses the
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
    if out_dim == 0 || in_dim == 0 || !in_dim.is_multiple_of(128) || n_tokens == 0 {
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
            // MetalKernel::apply uses dispatch_threads, unlike the reference's
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

pub(super) fn router_select_kernel() -> &'static Option<MetalKernel> {
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

/// Fused MoE router select (the reference `kernel_glm_router_select_one` port): logits
/// `[n, E]` f32 (contiguous), bias `[E]` (any dtype, cast to f32) -> selected
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
