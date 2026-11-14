//! Linear operations with the pinned DS4 arithmetic.
use super::*;

pub(super) const BF16_GEMV_SOURCE: &crate::shader_pack::Source =
    crate::kernel_source!("glm53/bf16_gemv.metal");
pub(super) const BF16_GEMV_ROWS2_SOURCE: &crate::shader_pack::Source =
    crate::kernel_source!("glm53/bf16_gemv_rows2.metal");
pub(super) const BF16_PAIR_GEMV_SOURCE: &crate::shader_pack::Source =
    crate::kernel_source!("glm53/bf16_pair_gemv.metal");
pub(super) const BF16_QKV_SOURCE: &crate::shader_pack::Source = crate::kernel_source!("glm53/bf16_qkv.metal");
pub(super) const BF16_GEMM_SOURCE: &crate::shader_pack::Source =
    crate::kernel_source!("glm53/bf16_gemm.metal");
pub(super) const BF16_KDA_PROJECT_SOURCE: &crate::shader_pack::Source =
    crate::kernel_source!("glm53/bf16_kda_project.metal");
pub(super) const AFFINE_Q8_MV_DS4_SOURCE: &crate::shader_pack::Source =
    crate::kernel_source!("glm53/affine_q8_mv_ds4.metal");

pub(super) fn bf16_gemv_kernel() -> &'static Option<MetalKernel> {
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

pub(super) fn bf16_gemv_rows2_kernel() -> &'static Option<MetalKernel> {
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

pub(super) fn bf16_pair_gemv_kernel() -> &'static Option<MetalKernel> {
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

pub(super) fn bf16_qkv_kernel() -> &'static Option<MetalKernel> {
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

pub(super) fn bf16_gemm_kernel() -> &'static Option<MetalKernel> {
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

pub(super) fn bf16_kda_project_kernel() -> &'static Option<MetalKernel> {
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
pub(super) fn bf16_projection_nsg(x: &Array) -> usize {
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

pub(super) fn affine_q8_mv_ds4_kernel() -> &'static Option<MetalKernel> {
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

pub(super) fn glm53_bf16_qkv_inner(
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