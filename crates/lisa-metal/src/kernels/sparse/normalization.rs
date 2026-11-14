//! Normalization operations with the pinned DS4 arithmetic.
use super::*;


pub(super) const RMS_ROW_SOURCE: &crate::shader_pack::Source = crate::kernel_source!("glm53/rms_row.metal");

pub(super) fn rms_row_kernel() -> &'static Option<MetalKernel> {
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

pub(super) const SWIGLU_SOURCE: &crate::shader_pack::Source = crate::kernel_source!("glm53/swiglu.metal");

pub(super) fn swiglu_kernel() -> &'static Option<MetalKernel> {
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

/// Fused clamped SwiGLU (ds4 glu.metal port): gate/up f32 `[N]` -> f32 `[N]`,
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

pub(super) const MTP_RMS_NORM_SOURCE: &crate::shader_pack::Source =
    crate::kernel_source!("glm53/mtp_rms_norm.metal");
pub(super) const MTP_ADD_RMS_NORM_SOURCE: &crate::shader_pack::Source =
    crate::kernel_source!("glm53/mtp_add_rms_norm.metal");
pub(super) const MTP_ADD3_SOURCE: &crate::shader_pack::Source = crate::kernel_source!("glm53/mtp_add3.metal");

pub(super) fn ds4_rms_norm_kernel() -> &'static Option<MetalKernel> {
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

pub(super) fn mtp_add_rms_norm_kernel() -> &'static Option<MetalKernel> {
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

pub(super) fn mtp_add3_kernel() -> &'static Option<MetalKernel> {
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

pub(super) const MTP_KV_LORA_RMS_NORM_SOURCE: &crate::shader_pack::Source =
    crate::kernel_source!("glm53/mtp_kv_lora_rms_norm.metal");
pub(super) const QKV_NORM_COMPACT_SOURCE: &crate::shader_pack::Source =
    crate::kernel_source!("glm53/qkv_norm_compact.metal");

pub(super) fn mtp_kv_lora_rms_norm_kernel() -> &'static Option<MetalKernel> {
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

pub(super) fn qkv_norm_compact_kernel() -> &'static Option<MetalKernel> {
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