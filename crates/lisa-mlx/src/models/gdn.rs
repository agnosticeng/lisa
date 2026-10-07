use crate::ffi::{MetalKernel, OutputArg, TemplateArg};
use lisa_mlx::ops::indexing::IndexOp;
use lisa_mlx::{Array, Dtype, Stream, ops};

use crate::kernels::EXACT_HEADER;

/// The gated delta rule scan kernel from `GatedDelta.swift`.
///
/// Inputs: q [B,T,Hk,Dk], k [B,T,Hk,Dk], v [B,T,Hv,Dv], g [B,T,Hv] f32,
/// beta [B,T,Hv] f32, state_in [B,Hv,Dv,Dk], T scalar i32.
/// Outputs: y [B,T,Hv,Dv] (activation dtype), state_out [B,Hv,Dv,Dk] (StT).
const GATED_DELTA_SOURCE: &str = include_str!("../shaders/gdn/gated_delta.metal");

/// `track_p12_gdn_prep_split_inputs`: the causal depthwise conv + silu, the q/k
/// RMS norms with their scales, and the decay/beta gates in ONE launch. The two
/// 48-wide gate rows come from their own projection buffers instead of a
/// concatenated `proj`.
///
/// Inputs: `qkv [B,T,CONV_DIM]`, `conv_state [B,KC-1,CONV_DIM]`,
/// `conv_w [CONV_DIM,KC,1]`, `neg_exp_alog [Hv] f32`, `dt_bias [Hv]`,
/// `b_gate [B,T,Hv]`, `a_gate [B,T,Hv]`.
/// Outputs: `qn [B,T,Hk,Dk]`, `kn [B,T,Hk,Dk]`, `vv [B,T,Hv,Dv]`,
/// `g [B,T,Hv] f32`, `beta [B,T,Hv] f32`, `conv_out [B,KC-1,CONV_DIM]`.
const GDN_PREP_SOURCE: &str = include_str!("../shaders/gdn/gdn_prep.metal");

static GDN_PREP: std::sync::OnceLock<Option<MetalKernel>> = std::sync::OnceLock::new();

fn gdn_prep_kernel() -> &'static Option<MetalKernel> {
    GDN_PREP.get_or_init(|| {
        MetalKernel::new(
            "track_p12_gdn_prep_split_inputs",
            &[
                "proj",
                "conv_state",
                "conv_w",
                "neg_exp_alog",
                "dt_bias",
                "b_gate",
                "a_gate",
            ],
            &["qn", "kn", "vv", "g", "beta", "conv_out"],
            GDN_PREP_SOURCE,
            EXACT_HEADER,
            true,
            false,
        )
        .ok()
    })
}

/// Run the fused GDN prep. Returns `(qn, kn, vv, g, beta, conv_out)` or `None`.
#[allow(clippy::too_many_arguments)]
pub fn gdn_prep(
    qkv: &Array,
    conv_state: &Array,
    conv_w: &Array,
    neg_exp_alog: &Array,
    dt_bias: &Array,
    b_gate: &Array,
    a_gate: &Array,
    t_len: i32,
    hk: i32,
    hv: i32,
    dk: i32,
    dv: i32,
    kc: i32,
    conv_dim: i32,
    stream: &Stream,
) -> Option<(Array, Array, Array, Array, Array, Array)> {
    let kernel = gdn_prep_kernel().as_ref()?;
    let b = qkv.dim(0);
    let inputs: [&Array; 7] = [
        qkv,
        conv_state,
        conv_w,
        neg_exp_alog,
        dt_bias,
        b_gate,
        a_gate,
    ];
    let template = [
        TemplateArg::Dtype("InT", Dtype::Bfloat16),
        TemplateArg::Int("T", t_len),
        TemplateArg::Int("Dk", dk),
        TemplateArg::Int("Dv", dv),
        TemplateArg::Int("Hk", hk),
        TemplateArg::Int("Hv", hv),
        TemplateArg::Int("KC", kc),
        TemplateArg::Int("PROJ_W", conv_dim),
        TemplateArg::Int("CONV_DIM", conv_dim),
        TemplateArg::Int("B_OFF", 0),
        TemplateArg::Int("A_OFF", 0),
    ];
    let outs = [
        OutputArg {
            shape: vec![b, t_len, hk, dk],
            dtype: Dtype::Bfloat16,
        },
        OutputArg {
            shape: vec![b, t_len, hk, dk],
            dtype: Dtype::Bfloat16,
        },
        OutputArg {
            shape: vec![b, t_len, hv, dv],
            dtype: Dtype::Bfloat16,
        },
        OutputArg {
            shape: vec![b, t_len, hv],
            dtype: Dtype::Float32,
        },
        OutputArg {
            shape: vec![b, t_len, hv],
            dtype: Dtype::Float32,
        },
        OutputArg {
            shape: vec![b, kc - 1, conv_dim],
            dtype: Dtype::Bfloat16,
        },
    ];
    let out = kernel
        .apply(
            &inputs,
            &template,
            (32, conv_dim / 128, b * t_len),
            (32, 4, 1),
            &outs,
            stream,
        )
        .ok()?;
    let mut it = out.into_iter();
    Some((
        it.next()?,
        it.next()?,
        it.next()?,
        it.next()?,
        it.next()?,
        it.next()?,
    ))
}

/// `track_gdn_prep`: the same prep with the two gate rows read from offsets
/// `B_OFF`/`A_OFF` in the concatenated `proj [B,T,PROJ_W]` (the verify/capture
/// and small-window path; the engine only splits for eligible wide prefill).
///
/// Inputs: `proj [B,T,PROJ_W]`, `conv_state [B,KC-1,CONV_DIM]`,
/// `conv_w [CONV_DIM,KC,1]`, `neg_exp_alog [Hv] f32`, `dt_bias [Hv]`.
const GDN_PREP_FUSED_SOURCE: &str = include_str!("../shaders/gdn/gdn_prep_fused.metal");

static GDN_PREP_FUSED: std::sync::OnceLock<Option<MetalKernel>> = std::sync::OnceLock::new();

fn gdn_prep_fused_kernel() -> &'static Option<MetalKernel> {
    GDN_PREP_FUSED.get_or_init(|| {
        MetalKernel::new(
            "track_gdn_prep",
            &["proj", "conv_state", "conv_w", "neg_exp_alog", "dt_bias"],
            &["qn", "kn", "vv", "g", "beta", "conv_out", "conv_in"],
            GDN_PREP_FUSED_SOURCE,
            EXACT_HEADER,
            true,
            false,
        )
        .ok()
    })
}

/// Run the fused-projection GDN prep. Returns
/// `(qn, kn, vv, g, beta, conv_out, conv_in)` where `conv_in` is the rollback
/// conv input `[B, KC-1+T, CONV_DIM]` (state rows then the proj rows — the
/// same values the composed `concatenate` produced, emitted copy-free).
#[allow(clippy::too_many_arguments)]
pub fn gdn_prep_fused(
    proj: &Array,
    conv_state: &Array,
    conv_w: &Array,
    neg_exp_alog: &Array,
    dt_bias: &Array,
    b_off: i32,
    a_off: i32,
    t_len: i32,
    hk: i32,
    hv: i32,
    dk: i32,
    dv: i32,
    kc: i32,
    proj_w: i32,
    conv_dim: i32,
    stream: &Stream,
) -> Option<(Array, Array, Array, Array, Array, Array, Array)> {
    let kernel = gdn_prep_fused_kernel().as_ref()?;
    let b = proj.dim(0);
    let inputs: [&Array; 5] = [proj, conv_state, conv_w, neg_exp_alog, dt_bias];
    let template = [
        TemplateArg::Dtype("InT", Dtype::Bfloat16),
        TemplateArg::Int("T", t_len),
        TemplateArg::Int("Dk", dk),
        TemplateArg::Int("Dv", dv),
        TemplateArg::Int("Hk", hk),
        TemplateArg::Int("Hv", hv),
        TemplateArg::Int("KC", kc),
        TemplateArg::Int("PROJ_W", proj_w),
        TemplateArg::Int("CONV_DIM", conv_dim),
        TemplateArg::Int("B_OFF", b_off),
        TemplateArg::Int("A_OFF", a_off),
    ];
    let outs = [
        OutputArg {
            shape: vec![b, t_len, hk, dk],
            dtype: Dtype::Bfloat16,
        },
        OutputArg {
            shape: vec![b, t_len, hk, dk],
            dtype: Dtype::Bfloat16,
        },
        OutputArg {
            shape: vec![b, t_len, hv, dv],
            dtype: Dtype::Bfloat16,
        },
        OutputArg {
            shape: vec![b, t_len, hv],
            dtype: Dtype::Float32,
        },
        OutputArg {
            shape: vec![b, t_len, hv],
            dtype: Dtype::Float32,
        },
        OutputArg {
            shape: vec![b, kc - 1, conv_dim],
            dtype: Dtype::Bfloat16,
        },
        OutputArg {
            shape: vec![b, kc - 1 + t_len, conv_dim],
            dtype: Dtype::Bfloat16,
        },
    ];
    let out = kernel
        .apply(
            &inputs,
            &template,
            (32, conv_dim / 128, b * t_len),
            (32, 4, 1),
            &outs,
            stream,
        )
        .ok()?;
    let mut it = out.into_iter();
    Some((
        it.next()?,
        it.next()?,
        it.next()?,
        it.next()?,
        it.next()?,
        it.next()?,
        it.next()?,
    ))
}
/// `track_gdn_lean_two_row`: the decode recurrence with two value rows per
/// simdgroup. The kernel body is static (see `kernels/models/qwen4/`).
const GDN_LEAN_TWO_ROW_SOURCE: &str = include_str!("../shaders/gdn/gdn_lean_two_row.metal");
/// `track_gdn_rows`: the prefill recurrence with four value rows per simdgroup.
const GDN_ROWS_SOURCE: &str = include_str!("../shaders/gdn/gdn_rows.metal");

static GDN_ROWS: std::sync::OnceLock<Option<MetalKernel>> = std::sync::OnceLock::new();
static GDN_TWO_ROW: std::sync::OnceLock<Option<MetalKernel>> = std::sync::OnceLock::new();

/// The `T` scalar argument, cached per value (specs/08 item 3): rebuilding it
/// per layer per step was a host-side graph-node reconstruction. `Array` is
/// immutable and `Arc`-backed, so sharing one instance is safe.
fn t_scalar(t: i32) -> Array {
    static CACHE: std::sync::OnceLock<std::sync::Mutex<std::collections::HashMap<i32, Array>>> =
        std::sync::OnceLock::new();
    let cache = CACHE.get_or_init(Default::default);
    cache
        .lock()
        .unwrap()
        .entry(t)
        .or_insert_with(|| Array::from_int(t))
        .clone()
}

fn gdn_two_row_kernel() -> &'static Option<MetalKernel> {
    GDN_TWO_ROW.get_or_init(|| {
        MetalKernel::new(
            "track_gdn_lean_two_row",
            &["q", "k", "v", "g", "beta", "state_in", "T"],
            &["y", "state_out"],
            GDN_LEAN_TWO_ROW_SOURCE,
            "",
            true,
            false,
        )
        .ok()
    })
}

#[allow(clippy::too_many_arguments)]
pub fn gdn_two_row(
    q: &Array,
    k: &Array,
    v: &Array,
    g: &Array,
    beta: &Array,
    state: &Array,
    capture: bool,
    stream: &Stream,
) -> Option<(Array, Array)> {
    gdn_two_row_impl(q, k, v, g, beta, state, capture, None, stream)
}

/// As [`gdn_two_row`] but writes `y`/`state_out` into the caller's persistent
/// buffers (the MTP verify reuses its large capture-state buffer across
/// rounds instead of reallocating it per GDN layer).
#[allow(clippy::too_many_arguments)]
pub fn gdn_two_row_into(
    q: &Array,
    k: &Array,
    v: &Array,
    g: &Array,
    beta: &Array,
    state: &Array,
    capture: bool,
    out_y: &Array,
    out_state: &Array,
    stream: &Stream,
) -> Option<(Array, Array)> {
    gdn_two_row_impl(
        q,
        k,
        v,
        g,
        beta,
        state,
        capture,
        Some((out_y, out_state)),
        stream,
    )
}

#[allow(clippy::too_many_arguments)]
fn gdn_two_row_impl(
    q: &Array,
    k: &Array,
    v: &Array,
    g: &Array,
    beta: &Array,
    state: &Array,
    capture: bool,
    prealloc: Option<(&Array, &Array)>,
    stream: &Stream,
) -> Option<(Array, Array)> {
    let kernel = gdn_two_row_kernel().as_ref()?;
    let b = k.dim(0);
    let t = k.dim(1);
    let hk = k.dim(2);
    let dk = k.dim(3);
    let hv = v.dim(2);
    let dv = v.dim(3);
    let rows = 2;
    let t_scalar = t_scalar(t);
    let inputs: [&Array; 7] = [q, k, v, g, beta, state, &t_scalar];
    let template = [
        TemplateArg::Dtype("InT", Dtype::Bfloat16),
        TemplateArg::Dtype("StT", Dtype::Float32),
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
    let outs = [
        OutputArg {
            shape: vec![b, t, hv, dv],
            dtype: Dtype::Bfloat16,
        },
        OutputArg {
            shape: state_shape,
            dtype: Dtype::Float32,
        },
    ];
    let grid = (32, dv / rows, b * hv);
    let tg = (32, 4, 1);
    let r = match prealloc {
        Some((py, ps)) => kernel.apply_into(&inputs, &template, grid, tg, &outs, &[py, ps], stream),
        None => kernel.apply(&inputs, &template, grid, tg, &outs, stream),
    }
    .ok()?;
    let mut it = r.into_iter();
    Some((it.next()?, it.next()?))
}

fn gdn_rows_kernel() -> &'static Option<MetalKernel> {
    GDN_ROWS.get_or_init(|| {
        MetalKernel::new(
            "track_gdn_rows",
            &["q", "k", "v", "g", "beta", "state_in", "T"],
            &["y", "state_out"],
            GDN_ROWS_SOURCE,
            "",
            true,
            false,
        )
        .ok()
    })
}

/// The prefill recurrence with 4 value rows per simdgroup (`track_gdn_rows`).
#[allow(clippy::too_many_arguments)]
pub fn gdn_rows(
    q: &Array,
    k: &Array,
    v: &Array,
    g: &Array,
    beta: &Array,
    state: &Array,
    capture: bool,
    stream: &Stream,
) -> Option<(Array, Array)> {
    let kernel = gdn_rows_kernel().as_ref()?;
    let b = k.dim(0);
    let t = k.dim(1);
    let hk = k.dim(2);
    let dk = k.dim(3);
    let hv = v.dim(2);
    let dv = v.dim(3);
    let rows = 4;
    let t_scalar = t_scalar(t);
    let inputs: [&Array; 7] = [q, k, v, g, beta, state, &t_scalar];
    let template = [
        TemplateArg::Dtype("InT", Dtype::Bfloat16),
        TemplateArg::Dtype("StT", Dtype::Float32),
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
    let outs = [
        OutputArg {
            shape: vec![b, t, hv, dv],
            dtype: Dtype::Bfloat16,
        },
        OutputArg {
            shape: state_shape,
            dtype: Dtype::Float32,
        },
    ];
    let r = kernel
        .apply(
            &inputs,
            &template,
            (32, dv / rows, b * hv),
            (32, 4, 1),
            &outs,
            stream,
        )
        .ok()?;
    let mut it = r.into_iter();
    Some((it.next()?, it.next()?))
}
/// Manages the JIT-compiled GDN kernels.
pub struct GatedDeltaKernels {
    kernel: Option<MetalKernel>,
}

static GDN_KERNEL: std::sync::OnceLock<GatedDeltaKernels> = std::sync::OnceLock::new();

impl GatedDeltaKernels {
    fn new() -> Self {
        let kernel = MetalKernel::new(
            "gated_delta_step",
            &["q", "k", "v", "g", "beta", "state_in", "T"],
            &["y", "state_out"],
            GATED_DELTA_SOURCE,
            "",
            true,
            false,
        )
        .ok();
        Self { kernel }
    }
}

fn gdn_kernels() -> &'static GatedDeltaKernels {
    GDN_KERNEL.get_or_init(GatedDeltaKernels::new)
}

pub fn gdn_kernel_available() -> bool {
    gdn_kernels().kernel.is_some()
}

/// The gated delta scan via the custom Metal kernel.
///
/// q/k: [B,T,Hk,Dk] (activation dtype), v: [B,T,Hv,Dv] (activation dtype),
/// g/beta: [B,T,Hv] f32, state: [B,Hv,Dv,Dk] f32.
///
/// `capture` writes the state after every position instead of only the last,
/// so `state_out` is `[B*T, Hv, Dv, Dk]` (slot `b*T + t`); the return value is
/// `(y, state_out)`. Without capture it is the usual `[B, Hv, Dv, Dk]`.
#[allow(clippy::too_many_arguments)]
pub fn gated_delta_kernel(
    q: &Array,
    k: &Array,
    v: &Array,
    g: &Array,
    beta: &Array,
    state: &Array,
    capture: bool,
    stream: &Stream,
) -> Option<(Array, Array)> {
    gated_delta_kernel_impl(q, k, v, g, beta, state, capture, stream)
}

#[allow(clippy::too_many_arguments)]
fn gated_delta_kernel_impl(
    q: &Array,
    k: &Array,
    v: &Array,
    g: &Array,
    beta: &Array,
    state: &Array,
    capture: bool,
    stream: &Stream,
) -> Option<(Array, Array)> {
    let kernels = gdn_kernels();
    let kernel = kernels.kernel.as_ref()?;

    let b = k.dim(0);
    let t = k.dim(1);
    let hk = k.dim(2);
    let dk = k.dim(3);
    let hv = v.dim(2);
    let dv = v.dim(3);
    let input_type = q.dtype();
    let state_type = state.dtype();

    let t_scalar = t_scalar(t);
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
            eprintln!("[gdn-decline] {e}");
            return None;
        }
    };
    let mut it = out.into_iter();
    let y = it.next()?;
    let state_out = it.next()?;
    Some((y, state_out))
}

/// Ops fallback matching the `gatedDeltaOps` kernel: a per-t loop.
///
/// Exact same math as the kernel modulo fp32 accumulation order.
pub fn gated_delta_ops(
    q: &Array,
    k: &Array,
    v: &Array,
    g: &Array,
    beta: &Array,
    state: Option<&Array>,
) -> lisa_mlx::error::Result<(Array, Array)> {
    let b = q.dim(0);
    let t = q.dim(1);
    let hk = q.dim(2);
    let dk = q.dim(3);
    let hv = v.dim(2);
    let dv = v.dim(3);

    let repeat = hv / hk;
    let q = if repeat > 1 {
        ops::repeat_axis::<half::bf16>(q.clone(), repeat as i32, -2)?
    } else {
        q.clone()
    };
    let k = if repeat > 1 {
        ops::repeat_axis::<half::bf16>(k.clone(), repeat as i32, -2)?
    } else {
        k.clone()
    };

    let mut state = match state {
        Some(s) if s.dtype() == Dtype::Float32 => s.clone(),
        Some(s) => s.as_dtype(Dtype::Float32)?,
        None => ops::full::<f32>(&[b, hv, dv, dk], Array::from_f32(0.0))?,
    };

    let mut ys = Vec::with_capacity(t as usize);
    for idx in 0..t {
        let q_t = q.index((.., idx));
        let k_t = k.index((.., idx));
        let v_t = v.index((.., idx));
        let g_t = g.index((.., idx));
        let beta_t = beta.index((.., idx));

        let decay = g_t.reshape(&[b, hv, 1, 1])?;
        let k_t = k_t.reshape(&[b, hv, 1, dk])?;
        let beta_t = beta_t.reshape(&[b, hv, 1])?;

        let new_state = state * decay;
        let kv_mem = (&new_state * &k_t).sum_axis(-1, None)?;
        let delta = (v_t - kv_mem) * beta_t;
        let new_state = new_state + k_t * delta.reshape(&[b, hv, dv, 1])?;
        let y = (&new_state * q_t.reshape(&[b, hv, 1, dk])?).sum_axis(-1, None)?;
        ys.push(y.as_dtype(q.dtype())?);
        state = new_state;
    }
    let y = ops::stack(&ys, 1)?;
    Ok((y, state))
}

/// Ops fallback for the capture path: returns `(y, state_seq)` where
/// `state_seq` is the SSM state after every position, `[B*T, Hv, Dv, Dk]`.
pub fn gated_delta_ops_capture(
    q: &Array,
    k: &Array,
    v: &Array,
    g: &Array,
    beta: &Array,
    state: &Array,
) -> lisa_mlx::error::Result<(Array, Array)> {
    let b = q.dim(0);
    let t = q.dim(1);
    let hk = q.dim(2);
    let dk = q.dim(3);
    let hv = v.dim(2);
    let dv = v.dim(3);

    let repeat = hv / hk;
    let q = if repeat > 1 {
        ops::repeat_axis::<half::bf16>(q.clone(), repeat as i32, -2)?
    } else {
        q.clone()
    };
    let k = if repeat > 1 {
        ops::repeat_axis::<half::bf16>(k.clone(), repeat as i32, -2)?
    } else {
        k.clone()
    };

    let mut state = if state.dtype() == Dtype::Float32 {
        state.clone()
    } else {
        state.as_dtype(Dtype::Float32)?
    };

    let mut ys = Vec::with_capacity(t as usize);
    let mut states = Vec::with_capacity(t as usize);
    for idx in 0..t {
        let q_t = q.index((.., idx));
        let k_t = k.index((.., idx));
        let v_t = v.index((.., idx));
        let g_t = g.index((.., idx));
        let beta_t = beta.index((.., idx));

        let decay = g_t.reshape(&[b, hv, 1, 1])?;
        let k_t = k_t.reshape(&[b, hv, 1, dk])?;
        let beta_t = beta_t.reshape(&[b, hv, 1])?;

        let new_state = state * decay;
        let kv_mem = (&new_state * &k_t).sum_axis(-1, None)?;
        let delta = (v_t - kv_mem) * beta_t;
        let new_state = new_state + k_t * delta.reshape(&[b, hv, dv, 1])?;
        let y = (&new_state * q_t.reshape(&[b, hv, 1, dk])?).sum_axis(-1, None)?;
        ys.push(y.as_dtype(q.dtype())?);
        states.push(new_state.reshape(&[b * hv * dv * dk])?);
        state = new_state;
    }
    let y = ops::stack(&ys, 1)?;
    let seq = ops::stack(&states, 0)?.reshape(&[b * t, hv, dv, dk])?;
    Ok((y, seq))
}
/// `track_gdn_decode_complete`: the S=1 GDN (causal conv + q/k l2norm + the
/// gated delta rule + gated RMS) in one launch. Ported verbatim from
/// `reference/Runner/FastModel/TrackFastGDNDecode.swift`.
///
/// `proj` is the concatenated `[qkv | z | b | a]` projection (one GEMM); the
/// kernel reads z/b/a from their offsets in it.
const GDN_DECODE_COMPLETE_SOURCE: &str = include_str!("../shaders/gdn/gdn_decode_complete.metal");

static GDN_DECODE_COMPLETE: std::sync::OnceLock<Option<MetalKernel>> = std::sync::OnceLock::new();

fn gdn_decode_complete_kernel() -> &'static Option<MetalKernel> {
    GDN_DECODE_COMPLETE.get_or_init(|| {
        MetalKernel::new(
            "track_gdn_decode_complete",
            &[
                "proj",
                "conv_state",
                "conv_w",
                "neg_exp_alog",
                "dt_bias",
                "state_in",
                "w",
            ],
            &["state_out", "gated", "conv_out"],
            GDN_DECODE_COMPLETE_SOURCE,
            EXACT_HEADER,
            true,
            false,
        )
        .ok()
    })
}

/// The S=1 fused GDN. Returns `(gated [b,1,hv*dv], state_out [b,hv,dv,dk], conv_out [b,KC-1,convDim])`.
///
/// `swish_gate` selects the qwen3_5 output gate (`silu(z) * rms`, SWISH
/// template flag) over the Flash-Next sigmoid gate.
#[allow(clippy::too_many_arguments)]
pub fn gdn_decode_complete(
    proj: &Array,
    conv_state: &Array,
    conv_w: &Array,
    neg_exp_alog: &Array,
    dt_bias: &Array,
    state_in: &Array,
    norm_w: &Array,
    hk: i32,
    hv: i32,
    dk: i32,
    dv: i32,
    kc: i32,
    conv_dim: i32,
    pw: i32,
    b_off: i32,
    a_off: i32,
    z_off: i32,
    eps: f32,
    swish_gate: bool,
    stream: &Stream,
) -> Option<(Array, Array, Array)> {
    let kernel = gdn_decode_complete_kernel().as_ref()?;
    let b = proj.dim(0);
    let inputs: [&Array; 7] = [
        proj,
        conv_state,
        conv_w,
        neg_exp_alog,
        dt_bias,
        state_in,
        norm_w,
    ];
    let template = [
        TemplateArg::Dtype("InT", Dtype::Bfloat16),
        TemplateArg::Dtype("StT", Dtype::Float32),
        TemplateArg::Int("Dk", dk),
        TemplateArg::Int("Dv", dv),
        TemplateArg::Int("Hk", hk),
        TemplateArg::Int("Hv", hv),
        TemplateArg::Int("KC", kc),
        TemplateArg::Int("CONV_DIM", conv_dim),
        TemplateArg::Int("PW", pw),
        TemplateArg::Int("B_OFF", b_off),
        TemplateArg::Int("A_OFF", a_off),
        TemplateArg::Int("Z_OFF", z_off),
        TemplateArg::Int("EPS_BITS", eps.to_bits() as i32),
        TemplateArg::Int("RPS", 4),
        TemplateArg::Bool("SWISH", swish_gate),
    ];
    let outs = [
        OutputArg {
            shape: vec![b, hv, dv, dk],
            dtype: Dtype::Float32,
        },
        OutputArg {
            shape: vec![b, 1, hv * dv],
            dtype: Dtype::Bfloat16,
        },
        OutputArg {
            shape: vec![b, kc - 1, conv_dim],
            dtype: Dtype::Bfloat16,
        },
    ];
    let r = kernel
        .apply(
            &inputs,
            &template,
            (32, dv / 4, b * hv),
            (32, dv / 4, 1),
            &outs,
            stream,
        )
        .ok()?;
    let mut it = r.into_iter();
    let state_out = it.next()?;
    let gated = it.next()?;
    let conv_out = it.next()?;
    Some((gated, state_out, conv_out))
}
#[cfg(test)]
mod avail_probe {
    #[test]
    fn which_gdn_kernels_available() {
        eprintln!("gdn_prep available: {}", super::gdn_prep_kernel().is_some());
        eprintln!("gdn_kernel available: {}", super::gdn_kernel_available());
    }
}

#[cfg(test)]
mod mach_ab {
    use lisa_mlx::Array;
    #[test]
    fn dec_machinery_ab() {
        let _ = run();
    }
    fn run() -> Option<()> {
        let stream = lisa_mlx::Stream::thread_local_or_default();
        let (b, hk, hv, dk, dv, kc, conv_dim, pw) =
            (1i32, 16i32, 48i32, 128i32, 128i32, 4i32, 10240i32, 16480i32);
        let (z_off, b_off, a_off) = (10240i32, 16384i32, 16432i32);
        let mut seed = 12345u64;
        let mut rnd = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((seed >> 40) as f32 / 8_388_608.0) - 1.0
        };
        let mk = |shape: &[i32], r: &mut dyn FnMut() -> f32| -> Array {
            let n: usize = shape.iter().map(|&d| d as usize).product();
            let v: Vec<f32> = (0..n).map(|_| half::bf16::from_f32(r()).to_f32()).collect();
            Array::from_slice(&v, shape)
                .as_dtype(lisa_mlx::Dtype::Bfloat16)
                .unwrap()
        };
        let proj = mk(&[b, pw], &mut rnd);
        let conv_state = mk(&[b, kc - 1, conv_dim], &mut rnd);
        let conv_w = mk(&[conv_dim, kc], &mut rnd);
        let negv: Vec<f32> = (0..hv as usize).map(|_| -rnd().abs()).collect();
        let neg_exp_alog = Array::from_slice(&negv, &[hv])
            .as_dtype(lisa_mlx::Dtype::Float32)
            .unwrap();
        let dt_bias = mk(&[hv], &mut rnd);
        let sv: Vec<f32> = (0..(b * hv * dv * dk) as usize)
            .map(|_| rnd() * 0.01)
            .collect();
        let state_in = Array::from_slice(&sv, &[b, hv, dv, dk])
            .as_dtype(lisa_mlx::Dtype::Float32)
            .unwrap();
        let norm_w = mk(&[dv], &mut rnd);
        let (gated, state_out, conv_out) = super::gdn_decode_complete(
            &proj,
            &conv_state,
            &conv_w,
            &neg_exp_alog,
            &dt_bias,
            &state_in,
            &norm_w,
            hk,
            hv,
            dk,
            dv,
            kc,
            conv_dim,
            pw,
            b_off,
            a_off,
            z_off,
            1e-6,
            false,
            &stream,
        )?;
        let dir = String::from("/tmp");
        let w = |name: &str, a: &Array| {
            let f = a.as_dtype(lisa_mlx::Dtype::Float32).unwrap();
            let s = f.as_slice::<f32>();
            let _ = std::fs::write(format!("{dir}/{name}.bin"), unsafe {
                std::slice::from_raw_parts(s.as_ptr() as *const u8, s.len() * 4)
            });
        };
        w("gated", &gated);
        w("state_out", &state_out);
        w("conv_out", &conv_out);
        Some(())
    }
}

/// Force `get_or_init` on every kernel static in this module (see
/// `crate::models::qwen4::warm_kernels`).
pub fn warm() {
    let _ = gdn_prep_kernel();
    let _ = gdn_prep_fused_kernel();
    let _ = gdn_two_row_kernel();
    let _ = gdn_rows_kernel();
    let _ = gdn_kernels();
    let _ = gdn_decode_complete_kernel();
}
