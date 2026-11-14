//! Recurrent operations with the pinned DS4 arithmetic.
use super::*;


pub(super) const KDA_RECURRENCE_SOURCE: &crate::shader_pack::Source =
    crate::kernel_source!("glm53/kda_recurrence.metal");
pub(super) const KDA_PREPARE_SOURCE: &crate::shader_pack::Source =
    crate::kernel_source!("glm53/kda_prepare.metal");
pub(super) const KDA_OUTPUT_SOURCE: &crate::shader_pack::Source =
    crate::kernel_source!("glm53/kda_output.metal");
pub(super) const KDA_DECODE_SOURCE: &crate::shader_pack::Source =
    crate::kernel_source!("glm53/kda_decode.metal");

pub(super) fn kda_prepare_kernel() -> &'static Option<MetalKernel> {
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

pub(super) fn kda_decode_kernel() -> &'static Option<MetalKernel> {
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

pub(super) fn kda_output_kernel() -> &'static Option<MetalKernel> {
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
    let rows_a = crate::kernels::gated_delta::t_scalar_pub(t as i32);
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
/// output), `gate` `[B, T, heads*D]` f32 pre-sigmoid, `w` `[D]` f32.
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
    let t_scalar = crate::kernels::gated_delta::t_scalar_pub(t_ as i32);
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

pub(super) fn kda_kernel() -> &'static Option<MetalKernel> {
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

    let t_scalar = crate::kernels::gated_delta::t_scalar_pub(t);
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