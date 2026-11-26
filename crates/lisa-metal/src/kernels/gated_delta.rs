use crate::ffi::{MetalKernel, OutputArg, TemplateArg};
#[cfg(test)]
use lisa_metal::ops::indexing::IndexOp;
use lisa_metal::{Array, Dtype, Stream};

use crate::kernels::EXACT_HEADER;

/// The gated delta rule scan kernel from `GatedDelta.swift`.
///
/// Inputs: q `[B,T,Hk,Dk]`, k `[B,T,Hk,Dk]`, v `[B,T,Hv,Dv]`, g `[B,T,Hv]` f32,
/// beta `[B,T,Hv]` f32, state_in `[B,Hv,Dv,Dk]`, T scalar i32.
/// Outputs: y `[B,T,Hv,Dv]` (activation dtype), state_out `[B,Hv,Dv,Dk]` (StT).
const GATED_DELTA_SOURCE: &crate::shader_pack::Source =
    crate::kernel_source!("gdn/gated_delta.metal");

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
const GDN_PREP_SOURCE: &crate::shader_pack::Source = crate::kernel_source!("gdn/gdn_prep.metal");

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
const GDN_PREP_FUSED_SOURCE: &crate::shader_pack::Source =
    crate::kernel_source!("gdn/gdn_prep_fused.metal");

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
const GDN_LEAN_TWO_ROW_SOURCE: &crate::shader_pack::Source =
    crate::kernel_source!("gdn/gdn_lean_two_row.metal");
/// `track_gdn_rows`: the prefill recurrence with four value rows per simdgroup.
const GDN_ROWS_SOURCE: &crate::shader_pack::Source = crate::kernel_source!("gdn/gdn_rows.metal");

static GDN_ROWS: std::sync::OnceLock<Option<MetalKernel>> = std::sync::OnceLock::new();
static GDN_TWO_ROW: std::sync::OnceLock<Option<MetalKernel>> = std::sync::OnceLock::new();

/// The `T` scalar argument, cached per value: rebuilding it
/// per layer per step was a host-side graph-node reconstruction. `Array` is
/// immutable and `Arc`-backed, so sharing one instance is safe.
pub fn t_scalar_pub(t: i32) -> Array {
    t_scalar(t)
}

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

/// Snapshot verification with the exact gated RMS epilogue in the same
/// dispatch. All recurrence outputs round to BF16 in threadgroup memory before
/// normalization; the FP32 per-prefix states retain the ordinary capture layout.
#[allow(clippy::too_many_arguments)]
pub fn gdn_verify_fused(
    q: &Array,
    k: &Array,
    v: &Array,
    g: &Array,
    beta: &Array,
    state: &Array,
    z: &Array,
    weight: &Array,
    eps: f32,
    prealloc: Option<(&Array, &Array)>,
    stream: &Stream,
) -> Option<(Array, Array)> {
    gdn_verify_fused_with_serial_gate(
        q, k, v, g, beta, state, z, weight, eps, prealloc, false, stream,
    )
}

/// Captured prefixes may require the ordinary fused decoder's gate law.
#[allow(clippy::too_many_arguments)]
pub fn gdn_verify_fused_with_serial_gate(
    q: &Array,
    k: &Array,
    v: &Array,
    g: &Array,
    beta: &Array,
    state: &Array,
    z: &Array,
    weight: &Array,
    eps: f32,
    prealloc: Option<(&Array, &Array)>,
    serial_gate: bool,
    stream: &Stream,
) -> Option<(Array, Array)> {
    let (b, t, hk, dk) = (k.dim(0), k.dim(1), k.dim(2), k.dim(3));
    let (hv, dv) = (v.dim(2), v.dim(3));
    if dk != 128
        || dv != 128
        || !(2..=16).contains(&t)
        || hk == 0
        || hv % hk != 0
        || state.dtype() != Dtype::Float32
        || !(state.shape() == [b, hv, dv, dk] || b == 1 && state.shape() == [hv, dv, dk])
        || [q, k, v, z, weight]
            .iter()
            .any(|x| x.dtype() != Dtype::Bfloat16)
        || q.shape() != k.shape()
        || v.shape() != [b, t, hv, dv]
        || weight.shape() != [dv]
        || g.shape() != [b, t, hv]
        || beta.shape() != [b, t, hv]
        || g.dtype() != Dtype::Float32
        || beta.dtype() != Dtype::Float32
    {
        return None;
    }
    let q = q.contiguous().ok()?;
    let k = k.contiguous().ok()?;
    let v = v.contiguous().ok()?;
    let g = g.contiguous().ok()?;
    let beta = beta.contiguous().ok()?;
    let state = state.contiguous().ok()?;
    let weight = weight.contiguous().ok()?;
    let z = z.reshape_rows_view((b * t) as usize, (hv * dv) as usize)?;
    static KERNEL: std::sync::OnceLock<Option<MetalKernel>> = std::sync::OnceLock::new();
    let kernel = KERNEL
        .get_or_init(|| {
            MetalKernel::new(
                "gdn_verify_fused",
                &["q", "k", "v", "g", "beta", "state_in", "T", "zproj", "w"],
                &["y", "state_out"],
                crate::kernel_source!("gdn/gdn_verify_fused.metal"),
                crate::kernel_source!("common/exact_header.metal"),
                true,
                false,
            )
            .ok()
        })
        .as_ref()?;
    let t_scalar = t_scalar(t);
    let inputs = [&q, &k, &v, &g, &beta, &state, &t_scalar, &z, &weight];
    let template = [
        TemplateArg::Dtype("InT", Dtype::Bfloat16),
        TemplateArg::Dtype("StT", Dtype::Float32),
        TemplateArg::Int("Dk", dk),
        TemplateArg::Int("Dv", dv),
        TemplateArg::Int("Hk", hk),
        TemplateArg::Int("Hv", hv),
        TemplateArg::Int("EPS_BITS", eps.to_bits() as i32),
        TemplateArg::Bool("CAPTURE", true),
        TemplateArg::Bool("SIGLU", true),
        TemplateArg::Bool("SERIAL_GATE", serial_gate),
    ];
    let outs = [
        OutputArg {
            shape: vec![b, t, hv, dv],
            dtype: Dtype::Bfloat16,
        },
        OutputArg {
            shape: vec![b * t, hv, dv, dk],
            dtype: Dtype::Float32,
        },
    ];
    let output = match prealloc {
        Some((y, state)) if y.shape() == outs[0].shape && state.shape() == outs[1].shape => kernel
            .apply_into(
                &inputs,
                &template,
                (32, 32, b * hv),
                (32, 32, 1),
                &outs,
                &[y, state],
                stream,
            ),
        _ => kernel.apply(
            &inputs,
            &template,
            (32, 32, b * hv),
            (32, 32, 1),
            &outs,
            stream,
        ),
    }
    .ok()?;
    let mut output = output.into_iter();
    Some((output.next()?, output.next()?))
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
/// q/k: `[B,T,Hk,Dk]` (activation dtype), v: `[B,T,Hv,Dv]` (activation dtype),
/// g/beta: `[B,T,Hv]` f32, state: `[B,Hv,Dv,Dk]` f32.
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

/// `track_gdn_decode_complete`: the S=1 GDN (causal conv + q/k l2norm + the
/// gated delta rule + gated RMS) in one launch. Ported verbatim from
/// upstream `Runner/FastModel/TrackFastGDNDecode.swift`.
///
/// `proj` is the concatenated `[qkv | z | b | a]` projection (one GEMM); the
/// kernel reads z/b/a from their offsets in it.
const GDN_DECODE_COMPLETE_SOURCE: &crate::shader_pack::Source =
    crate::kernel_source!("gdn/gdn_decode_complete.metal");

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
    delta_decode_checked(
        proj,
        conv_state,
        conv_w,
        neg_exp_alog,
        dt_bias,
        state_in,
        norm_w,
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
        eps,
        swish_gate,
        stream,
    )
    .ok()
}

#[allow(clippy::too_many_arguments)]
pub fn delta_decode_checked(
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
) -> crate::error::Result<(Array, Array, Array)> {
    let kernel = gdn_decode_complete_kernel()
        .as_ref()
        .ok_or_else(|| crate::error::Error::msg("delta decode kernel unavailable"))?;
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
    let r = kernel.apply(
        &inputs,
        &template,
        (32, dv / 4, b * hv),
        (32, dv / 4, 1),
        &outs,
        stream,
    )?;
    let mut it = r.into_iter();
    let state_out = it.next().expect("validated decode output");
    let gated = it.next().expect("validated decode output");
    let conv_out = it.next().expect("validated decode output");
    Ok((gated, state_out, conv_out))
}
/// Force `get_or_init` on every kernel static in this module (see
/// `crate::kernels::warm_kernels`).
pub fn warm() {
    let _ = gdn_prep_kernel();
    let _ = gdn_prep_fused_kernel();
    let _ = gdn_two_row_kernel();
    let _ = gdn_rows_kernel();
    let _ = gdn_kernels();
    let _ = gdn_decode_complete_kernel();
}

#[cfg(test)]
#[path = "gated_delta/parity_tests.rs"]
mod decode_capture_tests;
