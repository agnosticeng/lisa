//! Hyperconnection operations with the pinned DS4 arithmetic.
use super::*;


pub(super) const HC_SPLIT_SOURCE: &crate::shader_pack::Source = crate::kernel_source!("glm53/hc_split.metal");
pub(super) const HC_SPLIT_NORM4_SOURCE: &crate::shader_pack::Source =
    crate::kernel_source!("glm53/hc_split_norm4.metal");
pub(super) const HC_EXPAND_SOURCE: &crate::shader_pack::Source =
    crate::kernel_source!("glm53/hc_expand.metal");
pub(super) const HC_WSUM_SOURCE: &crate::shader_pack::Source = crate::kernel_source!("glm53/hc_wsum.metal");

pub(super) fn hc_wsum_kernel() -> &'static Option<MetalKernel> {
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

/// Fused mHC collapse (ds4 weighted_sum port): residual `[B,T,4,H]` f32,
/// pre `[B,T,4]` f32 -> x `[B,T,H]` f32. One dispatch replaces the narrow/add
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

pub(super) fn hc_split_kernel() -> &'static Option<MetalKernel> {
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

pub(super) fn hc_split_norm4_kernel() -> &'static Option<MetalKernel> {
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

pub(super) fn hc_expand_kernel() -> &'static Option<MetalKernel> {
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

/// Fused HC split+sinkhorn: mixes `[N,24]` f32 -> split `[N,24]` f32
/// (pre `out[0:4]`, post `out[4:8]`, comb `out[8:24]` row-major `[source][destination]`).
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

/// Fused HC expand4: y `[B,T,H]` f32, residual `[B,T,4,H]` f32, post `[B,T,4]` f32,
/// comb `[B,T,4,4]` f32 -> new stream `[B,T,4,H]` f32.
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
pub(super) const SHARED_Q8_SWIGLU_SOURCE: &crate::shader_pack::Source =
    crate::kernel_source!("glm53/shared_q8_swiglu_ds4.metal");

pub(super) fn shared_q8_swiglu_ds4_kernel() -> &'static Option<MetalKernel> {
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
pub(super) const HC_MIX_RMS_SOURCE: &crate::shader_pack::Source =
    crate::kernel_source!("glm53/hc_mix_rms.metal");
pub(super) const HC_MIX_RMS_BF16_GEMV_SOURCE: &crate::shader_pack::Source =
    crate::kernel_source!("glm53/hc_mix_rms_bf16_gemv.metal");

pub(super) fn hc_mix_rms_kernel() -> &'static Option<MetalKernel> {
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

pub(super) fn hc_mix_rms_bf16_gemv_kernel() -> &'static Option<MetalKernel> {
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