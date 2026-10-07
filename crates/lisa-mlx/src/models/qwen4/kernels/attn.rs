use crate::ffi::{MetalKernel, OutputArg, TemplateArg};
use lisa_mlx::ops::indexing::IndexOp;
use lisa_mlx::{Array, Dtype, Stream};

use super::common::EXACT_HEADER;

/// `track_gated_rms` (split-z form): the output gated RMS over `Dv` per value/// head (the engine's butterfly reduction), then `sigmoid(z) * out` in one
/// launch. `y`, `zproj` and `out` are flat `[rows, Hv, Dv]`, `zproj` is
/// `[rows, Hv*Dv]`, `w` is `[Dv]`.
const GATED_RMS_SOURCE: &str = include_str!("../../../shaders/qwen4/gated_rms.metal");

static GATED_RMS: std::sync::OnceLock<Option<MetalKernel>> = std::sync::OnceLock::new();

fn gated_rms_kernel() -> &'static Option<MetalKernel> {
    GATED_RMS.get_or_init(|| {
        MetalKernel::new(
            "track_gated_rms",
            &["y", "zproj", "w"],
            &["out"],
            GATED_RMS_SOURCE,
            EXACT_HEADER,
            true,
            false,
        )
        .ok()
    })
}

/// Run the fused gated RMS. `y`/`zproj` are `[B,S,Hv,Dv]`/`[B,S,Hv*Dv]`, `w`
/// `[Dv]` -> `[B,S,Hv,Dv]`. Returns `None` if the kernel is unavailable.
#[allow(clippy::too_many_arguments)]
pub fn gated_rms(
    y: &Array,
    zproj: &Array,
    w: &Array,
    hv: i32,
    dv: i32,
    eps: f32,
    stream: &Stream,
) -> Option<Array> {
    let kernel = gated_rms_kernel().as_ref()?;
    let b = y.dim(0);
    let s = y.dim(1);
    let rows = b * s;
    let y_flat = y.reshape(&[rows * hv * dv]).ok()?;
    // The z view binds through its strides (kernel reads zproj_strides), so a
    // strided slice of a fused projection needs NO materializing reshape.
    let cols = (hv * dv) as usize;
    let z_flat = zproj
        .reshape_rows_view(rows as usize, cols)
        .ok_or(())
        .or_else(|_| zproj.reshape(&[rows, hv * dv]))
        .ok()?;
    let inputs: [&Array; 3] = [&y_flat, &z_flat, w];
    let template = [
        TemplateArg::Dtype("InT", Dtype::Bfloat16),
        TemplateArg::Int("Hv", hv),
        TemplateArg::Int("Dv", dv),
        TemplateArg::Int("EPS_BITS", eps.to_bits() as i32),
        TemplateArg::Bool("SIGLU", false),
    ];
    let outs = [OutputArg {
        shape: vec![rows * hv * dv],
        dtype: Dtype::Bfloat16,
    }];
    let out = kernel
        .apply(
            &inputs,
            &template,
            (32, hv, rows),
            (32, 1, 1),
            &outs,
            stream,
        )
        .ok()?;
    let flat = out.into_iter().next()?;
    flat.reshape(&[b, s, hv, dv]).ok()
}

/// Run the fused gated RMS with the SILU gate (qwen3_5 `output_gate_type=swish`
/// — specs/01): `silu(z) * RMSNorm(out)` in one launch, replacing the composed
/// `fast::rms_norm` + `nn::silu` (= Sigmoid + bmul) + `multiply` chain — 4
/// dispatches per GDN layer down to 1. The kernel's silu arm reproduces the
/// composed rounding points (sigmoid→bf16, z·sig→bf16, gate·normed→bf16);
/// pinned by `gated_rms_silu_matches_composed`. Same shapes as [`gated_rms`].
#[allow(clippy::too_many_arguments)]
pub fn gated_rms_silu(
    y: &Array,
    zproj: &Array,
    w: &Array,
    hv: i32,
    dv: i32,
    eps: f32,
    stream: &Stream,
) -> Option<Array> {
    let kernel = gated_rms_kernel().as_ref()?;
    let b = y.dim(0);
    let s = y.dim(1);
    let rows = b * s;
    let y_flat = y.reshape(&[rows * hv * dv]).ok()?;
    let cols = (hv * dv) as usize;
    let z_flat = zproj
        .reshape_rows_view(rows as usize, cols)
        .ok_or(())
        .or_else(|_| zproj.reshape(&[rows, hv * dv]))
        .ok()?;
    let inputs: [&Array; 3] = [&y_flat, &z_flat, w];
    let template = [
        TemplateArg::Dtype("InT", Dtype::Bfloat16),
        TemplateArg::Int("Hv", hv),
        TemplateArg::Int("Dv", dv),
        TemplateArg::Int("EPS_BITS", eps.to_bits() as i32),
        TemplateArg::Bool("SIGLU", true),
    ];
    let outs = [OutputArg {
        shape: vec![rows * hv * dv],
        dtype: Dtype::Bfloat16,
    }];
    let out = kernel
        .apply(
            &inputs,
            &template,
            (32, hv, rows),
            (32, 1, 1),
            &outs,
            stream,
        )
        .ok()?;
    let flat = out.into_iter().next()?;
    flat.reshape(&[b, s, hv, dv]).ok()
}
/// `track_hc_mix`: `input[d] = bf16(sum_s bf16(sigmoid(w[s,d])) * normed[s,d])`
/// with the stream fold accumulating in bf16, one rounding per add, in stream
/// order; when `HAS_INJECT`, `inject[s] = 2 * sigmoid(inj[s])` in bf16.
///
/// `w`/`normed` are `[rows, HC*H]`, `inj` is `[rows, HC]` -> `input`
/// `[rows, H]`, `inject` `[rows, HC]`.
const HC_MIX_SOURCE: &str = include_str!("../../../shaders/qwen4/hc_mix.metal");

static HC_MIX: std::sync::OnceLock<Option<MetalKernel>> = std::sync::OnceLock::new();

fn hc_mix_kernel() -> &'static Option<MetalKernel> {
    HC_MIX.get_or_init(|| {
        MetalKernel::new(
            "track_hc_mix",
            &["w", "normed", "inj"],
            &["input", "inject"],
            HC_MIX_SOURCE,
            EXACT_HEADER,
            true,
            false,
        )
        .ok()
    })
}

/// Run the fused hyper-connection mix. Returns `(input [rows,H], inject
/// [rows,HC])`, or `None` if the kernel is unavailable.
#[allow(clippy::too_many_arguments)]
pub fn hc_mix(
    w: &Array,
    normed: &Array,
    inj: &Array,
    hc: i32,
    hidden: i32,
    rows: i32,
    has_inject: bool,
    stream: &Stream,
) -> Option<(Array, Array)> {
    let kernel = hc_mix_kernel().as_ref()?;
    let w_flat = w.reshape(&[rows, hc * hidden]).ok()?;
    let n_flat = normed.reshape(&[rows, hc * hidden]).ok()?;
    let inj_flat = if has_inject {
        inj.reshape(&[rows, hc]).ok()?
    } else {
        n_flat.index((.., 0..hc)).contiguous().ok()?
    };
    let inputs: [&Array; 3] = [&w_flat, &n_flat, &inj_flat];
    let template = [
        TemplateArg::Dtype("InT", Dtype::Bfloat16),
        TemplateArg::Int("H", hidden),
        TemplateArg::Int("W", hc * hidden),
        TemplateArg::Int("HC", hc),
        TemplateArg::Int("LW", hc),
        TemplateArg::Bool("HAS_INJECT", has_inject),
    ];
    let outs = [
        OutputArg {
            shape: vec![rows, hidden],
            dtype: Dtype::Bfloat16,
        },
        OutputArg {
            shape: vec![rows, hc],
            dtype: Dtype::Bfloat16,
        },
    ];
    let out = kernel
        .apply(
            &inputs,
            &template,
            (hidden, rows, 1),
            (256, 1, 1),
            &outs,
            stream,
        )
        .ok()?;
    let mut it = out.into_iter();
    Some((it.next()?, it.next()?))
}
/// `track_swiglu2`: `mlx_silu(gate) * up` over separate gate/up arrays, bf16.
const SWIGLU2_SOURCE: &str = include_str!("../../../shaders/qwen4/swiglu2.metal");

static SWIGLU2: std::sync::OnceLock<Option<MetalKernel>> = std::sync::OnceLock::new();

fn swiglu2_kernel() -> &'static Option<MetalKernel> {
    SWIGLU2.get_or_init(|| {
        MetalKernel::new(
            "track_swiglu2",
            &["gate", "up"],
            &["out"],
            SWIGLU2_SOURCE,
            EXACT_HEADER,
            true,
            false,
        )
        .ok()
    })
}

/// `track_silu_head`: `mlx_silu` over the leading `LMIX` columns of `lo`.
const SILU_HEAD_SOURCE: &str = include_str!("../../../shaders/qwen4/silu_head.metal");

static SILU_HEAD: std::sync::OnceLock<Option<MetalKernel>> = std::sync::OnceLock::new();

fn silu_head_kernel() -> &'static Option<MetalKernel> {
    SILU_HEAD.get_or_init(|| {
        MetalKernel::new(
            "track_silu_head",
            &["lo"],
            &["out"],
            SILU_HEAD_SOURCE,
            EXACT_HEADER,
            true,
            false,
        )
        .ok()
    })
}

/// Run `track_swiglu2`. `gate`/`up` `[B,F]` -> `out` `[B,F]`.
pub fn swiglu2(gate: &Array, up: &Array, stream: &Stream) -> Option<Array> {
    let kernel = swiglu2_kernel().as_ref()?;
    let f = gate.dim(-1);
    let b = (gate.size() / f as usize) as i32;
    let inputs: [&Array; 2] = [gate, up];
    let template = [
        TemplateArg::Dtype("InT", Dtype::Bfloat16),
        TemplateArg::Int("B", b),
        TemplateArg::Int("F", f),
    ];
    let outs = [OutputArg {
        shape: vec![b, f],
        dtype: Dtype::Bfloat16,
    }];
    kernel
        .apply(
            &inputs,
            &template,
            (b * f, 1, 1),
            (256, 1, 1),
            &outs,
            stream,
        )
        .ok()?
        .into_iter()
        .next()
}

/// Run `track_silu_head` over `[rows, W]` -> `[rows, width]`.
pub fn silu_head(lo: &Array, width: i32, stream: &Stream) -> Option<Array> {
    let kernel = silu_head_kernel().as_ref()?;
    let lw = lo.dim(-1);
    let rows = (lo.size() / lw as usize) as i32;
    let inputs: [&Array; 1] = [lo];
    let template = [
        TemplateArg::Dtype("InT", Dtype::Bfloat16),
        TemplateArg::Int("LW", lw),
        TemplateArg::Int("LMIX", width),
    ];
    let outs = [OutputArg {
        shape: vec![rows, width],
        dtype: Dtype::Bfloat16,
    }];
    kernel
        .apply(
            &inputs,
            &template,
            (width, rows, 1),
            (256, 1, 1),
            &outs,
            stream,
        )
        .ok()?
        .into_iter()
        .next()
}
/// `track_inject_norm`: `stream = residual + out ⊗ inject` (bf16), and
/// `normed = rms(stream_group) * scale` with the engine's reduction (4
/// elements/thread, butterfly, `precise::rsqrt`, weight after the bf16 round).
/// `scale` is the hc_norm weight pre-divided by `HC`.
///
/// `residual` is `[rows, W]` (or `[rows, H]` when `TILE`), `out` `[rows, H]`,
/// `inject` `[rows, HC]`, `scale` `[W]` -> `stream`/`normed` `[rows, W]`.
const INJECT_NORM_SOURCE: &str = include_str!("../../../shaders/qwen4/inject_norm.metal");

static INJECT_NORM: std::sync::OnceLock<Option<MetalKernel>> = std::sync::OnceLock::new();

fn inject_norm_kernel() -> &'static Option<MetalKernel> {
    INJECT_NORM.get_or_init(|| {
        MetalKernel::new(
            "track_inject_norm",
            &["residual", "out", "inject", "scale"],
            &["stream", "normed"],
            INJECT_NORM_SOURCE,
            EXACT_HEADER,
            true,
            false,
        )
        .ok()
    })
}

/// Run the fused inject + group RMS. Returns `(stream, normed)` `[rows, W]`.
#[allow(clippy::too_many_arguments)]
pub fn inject_norm(
    residual: &Array,
    out: Option<&Array>,
    inject: Option<&Array>,
    scale: &Array,
    hc: i32,
    hidden: i32,
    rows: i32,
    tile: bool,
    eps: f32,
    stream: &Stream,
) -> Option<(Array, Array)> {
    let kernel = inject_norm_kernel().as_ref()?;
    let has_inject = out.is_some() && inject.is_some();
    let w = hc * hidden;
    let dummy = residual;
    let out_a = out.unwrap_or(dummy);
    let inject_a = inject.unwrap_or(dummy);
    let inputs: [&Array; 4] = [residual, out_a, inject_a, scale];
    let template = [
        TemplateArg::Dtype("InT", Dtype::Bfloat16),
        TemplateArg::Int("H", hidden),
        TemplateArg::Int("W", w),
        TemplateArg::Int("HC", hc),
        TemplateArg::Bool("HAS_INJECT", has_inject),
        TemplateArg::Bool("TILE", tile),
        TemplateArg::Int("EPS_BITS", eps.to_bits() as i32),
    ];
    let outs = [
        OutputArg {
            shape: vec![rows, w],
            dtype: Dtype::Bfloat16,
        },
        OutputArg {
            shape: vec![rows, w],
            dtype: Dtype::Bfloat16,
        },
    ];
    let result = kernel
        .apply(
            &inputs,
            &template,
            (hidden / 4, hc, rows),
            (hidden / 4, 1, 1),
            &outs,
            stream,
        )
        .ok()?;
    let mut it = result.into_iter();
    Some((it.next()?, it.next()?))
}
/// `track_p12_attn_prep_split_inputs`: q/k RMS + partial rope + v passthrough.
/// Inputs: `qkv` = [q|gate] rows `[B,S,2*HQ*D]`, `kproj`/`vproj` `[B,S,HK*D]`.
const ATTN_PREP_SPLIT_SOURCE: &str = include_str!("../../../shaders/qwen4/attn_prep_split.metal");

static ATTN_PREP_SPLIT: std::sync::OnceLock<Option<MetalKernel>> = std::sync::OnceLock::new();
fn attn_prep_split_kernel() -> &'static Option<MetalKernel> {
    ATTN_PREP_SPLIT.get_or_init(|| {
        MetalKernel::new(
            "track_p12_attn_prep_split_inputs",
            &["qkv", "kproj", "vproj", "qnorm", "knorm", "cosb", "sinb"],
            &["qout", "kout", "vout"],
            ATTN_PREP_SPLIT_SOURCE,
            EXACT_HEADER,
            true,
            false,
        )
        .ok()
    })
}

/// `(q [B,HQ,S,D], k [B,HK,S,D], v [B,HK,S,D])`. `qkv` = [q|gate].
#[allow(clippy::too_many_arguments)]
pub fn attn_prep_split(
    qkv: &Array,
    kproj: &Array,
    vproj: &Array,
    q_norm: &Array,
    k_norm: &Array,
    cos: &Array,
    sin: &Array,
    heads: i32,
    kv_heads: i32,
    head_dim: i32,
    rotary_dims: i32,
    eps: f32,
    stream: &Stream,
) -> Option<(Array, Array, Array)> {
    let kernel = attn_prep_split_kernel().as_ref()?;
    let b = qkv.dim(0);
    let s = qkv.dim(1);
    let qw = qkv.dim(2);
    let inputs: [&Array; 7] = [qkv, kproj, vproj, q_norm, k_norm, cos, sin];
    let template = [
        TemplateArg::Dtype("InT", Dtype::Bfloat16),
        TemplateArg::Int("D", head_dim),
        TemplateArg::Int("HQ", heads),
        TemplateArg::Int("HK", kv_heads),
        TemplateArg::Int("S", s),
        TemplateArg::Int("QW", qw),
        TemplateArg::Int("ROT", rotary_dims),
        TemplateArg::Int("EPS_BITS", eps.to_bits() as i32),
    ];
    let outs = [
        OutputArg {
            shape: vec![b, heads, s, head_dim],
            dtype: Dtype::Bfloat16,
        },
        OutputArg {
            shape: vec![b, kv_heads, s, head_dim],
            dtype: Dtype::Bfloat16,
        },
        OutputArg {
            shape: vec![b, kv_heads, s, head_dim],
            dtype: Dtype::Bfloat16,
        },
    ];
    let r = kernel
        .apply(
            &inputs,
            &template,
            (head_dim / 4, heads + 2 * kv_heads, b * s),
            (head_dim / 4, 1, 1),
            &outs,
            stream,
        )
        .ok()?;
    let mut it = r.into_iter();
    Some((it.next()?, it.next()?, it.next()?))
}
/// `track_attn_gate` (split-gate form): `out = att * sigmoid(gate)`.
const ATTN_GATE_SOURCE: &str = include_str!("../../../shaders/qwen4/attn_gate.metal");

static ATTN_GATE: std::sync::OnceLock<Option<MetalKernel>> = std::sync::OnceLock::new();
fn attn_gate_kernel() -> &'static Option<MetalKernel> {
    ATTN_GATE.get_or_init(|| {
        MetalKernel::new(
            "track_attn_gate",
            &["att", "gateb"],
            &["out"],
            ATTN_GATE_SOURCE,
            EXACT_HEADER,
            true,
            false,
        )
        .ok()
    })
}

/// `att`/`gateb` are `[rows, HQ*D]` -> `out` `[rows, HQ*D]`.
pub fn attn_gate(
    att: &Array,
    gate: &Array,
    heads: i32,
    head_dim: i32,
    stream: &Stream,
) -> Option<Array> {
    let kernel = attn_gate_kernel().as_ref()?;
    let hw = heads * head_dim;
    let rows = (att.size() / hw as usize) as i32;
    let a = att.reshape(&[rows, hw]).ok()?;
    let g = gate.reshape(&[rows, hw]).ok()?;
    let inputs: [&Array; 2] = [&a, &g];
    let template = [
        TemplateArg::Dtype("InT", Dtype::Bfloat16),
        TemplateArg::Int("HQ", heads),
        TemplateArg::Int("D", head_dim),
    ];
    let outs = [OutputArg {
        shape: vec![rows, hw],
        dtype: Dtype::Bfloat16,
    }];
    kernel
        .apply(
            &inputs,
            &template,
            (hw, rows, 1),
            (256, 1, 1),
            &outs,
            stream,
        )
        .ok()?
        .into_iter()
        .next()
}
mod mixer_avail {
    #[test]
    fn mixer_kernels_available() {
        // indirect probe: call with tiny inputs and see if the kernel path returns Some
        use lisa_mlx::Array;
        let stream = lisa_mlx::Stream::thread_local_or_default();
        let w = Array::from_slice(&vec![0f32; 8], &[2, 2, 2])
            .as_dtype(lisa_mlx::Dtype::Bfloat16)
            .unwrap();
        let normed = Array::from_slice(&vec![0f32; 8], &[1, 2, 4])
            .as_dtype(lisa_mlx::Dtype::Bfloat16)
            .unwrap();
        let inj = Array::from_slice(&vec![0f32; 4], &[1, 2, 2])
            .as_dtype(lisa_mlx::Dtype::Bfloat16)
            .unwrap();
        eprintln!(
            "hc_mix: {}",
            super::hc_mix(&w, &normed, &inj, 2, 2, 2, true, &stream).is_some()
        );
        let lo = Array::from_slice(&vec![0f32; 4], &[1, 2, 2])
            .as_dtype(lisa_mlx::Dtype::Bfloat16)
            .unwrap();
        eprintln!("silu_head: {}", super::silu_head(&lo, 2, &stream).is_some());
    }
}

#[cfg(test)]
mod mixer_mach_ab {
    use lisa_mlx::Array;
    #[test]
    fn mixer_machinery_ab() {
        let stream = lisa_mlx::Stream::thread_local_or_default();
        let (rows, width, hc) = (1024usize, 2560usize, 4usize);
        let mut seed = 999u64;
        let mut rnd = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((seed >> 40) as f32 / 8_388_608.0) - 1.0
        };
        let mk = |shape: &[i32], r: &mut dyn FnMut() -> f32| {
            let n: usize = shape.iter().map(|&d| d as usize).product();
            let v: Vec<f32> = (0..n)
                .map(|_| half::bf16::from_f32(r() * 30.0).to_f32())
                .collect();
            Array::from_slice(&v, shape)
                .as_dtype(lisa_mlx::Dtype::Bfloat16)
                .unwrap()
        };
        let lo = mk(&[rows as i32, width as i32], &mut rnd);
        let dir = String::from("/tmp");
        let w = |name: &str, a: &Array| {
            let f = a.as_dtype(lisa_mlx::Dtype::Float32).unwrap();
            let s = f.as_slice::<f32>();
            let _ = std::fs::write(format!("{dir}/{name}.bin"), unsafe {
                std::slice::from_raw_parts(s.as_ptr() as *const u8, s.len() * 4)
            });
        };
        if let Some(r) = super::silu_head(&lo, width as i32, &stream) {
            w("silu_head", &r);
        }
        // hc_mix: w [rows, hc*h], normed [rows, hc*h], inj [rows, hc]
        let ww = mk(&[rows as i32, (hc * width) as i32], &mut rnd);
        let normed = mk(&[rows as i32, (hc * width) as i32], &mut rnd);
        let inj = mk(&[rows as i32, hc as i32], &mut rnd);
        if let Some((input, inject_w)) = super::hc_mix(
            &ww,
            &normed,
            &inj,
            hc as i32,
            width as i32,
            rows as i32,
            true,
            &stream,
        ) {
            w("hc_mix_input", &input);
            w("hc_mix_inject", &inject_w);
        }
    }
}

#[cfg(test)]
mod gated_rms_silu_parity {
    use lisa_mlx::Array;

    /// specs/01 pin: the SILU-gate arm of `track_gated_rms` vs the composed
    /// chain it replaces on the qwen3_5 verify path (`fast::rms_norm` +
    /// `nn::silu` (= Sigmoid + bmul) + `multiply`). ~74k values per tensor,
    /// plus specials on the gate. The rms half must ALSO match the stock
    /// sigmoid-arm kernel (same butterfly reduction, shared source).
    #[test]
    fn gated_rms_silu_matches_composed() {
        use super::gated_rms_silu;
        use lisa_mlx::ops::nn;
        use lisa_mlx::{Dtype, Stream};
        let stream = Stream::thread_local_or_default();
        let (b, s, hv, dv) = (1i32, 12i32, 48i32, 128i32);
        let eps = 1e-6f32;
        let mut seed = 424243u64;
        let mut rnd = move || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((seed >> 40) as f32 / 8_388_608.0) - 1.0
        };
        let n = (b * s * hv * dv) as usize;
        let yv: Vec<f32> = (0..n).map(|_| rnd() * 4.0 - 2.0).collect();
        let mut zv: Vec<f32> = (0..n - 10).map(|_| rnd() * 16.0 - 8.0).collect();
        for sp in [0.0f32, -0.0, 1.0, -1.0, 20.0, -20.0, 88.0, -88.0, 0.5, -3.5] {
            zv.push(sp);
        }
        let wv: Vec<f32> = (0..dv as usize).map(|_| rnd() * 2.0 - 1.0).collect();
        let y = Array::from_slice(&yv, &[b, s, hv, dv])
            .as_dtype(Dtype::Bfloat16)
            .unwrap();
        let z = Array::from_slice(&zv, &[b, s, hv, dv])
            .as_dtype(Dtype::Bfloat16)
            .unwrap();
        let w = Array::from_slice(&wv, &[dv])
            .as_dtype(Dtype::Bfloat16)
            .unwrap();

        let fused = gated_rms_silu(&y, &z, &w, hv, dv, eps, &stream).expect("kernel");

        // Composed chain exactly as gdn.rs builds it today.
        let rms = lisa_mlx::fast::rms_norm(&y, Some(&w), eps).unwrap();
        let composed = nn::silu(&z).unwrap().multiply(&rms).unwrap();

        let fv = fused
            .as_dtype(Dtype::Float32)
            .unwrap()
            .reshape(&[(n) as i32])
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let cv = composed
            .as_dtype(Dtype::Float32)
            .unwrap()
            .reshape(&[(n) as i32])
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let bad: Vec<usize> = (0..n)
            .filter(|&i| fv[i].to_bits() != cv[i].to_bits())
            .collect();
        assert!(
            bad.is_empty(),
            "{} mismatches, first at {:?}: fused={} composed={}",
            bad.len(),
            bad.first(),
            fv[*bad.first().unwrap()],
            cv[*bad.first().unwrap()]
        );
    }

    /// specs/04 pin: `gated_rms`/`gated_rms_silu` over a STRIDED z (a slice
    /// view of a fused projection, bound through `zproj_strides`) must be
    /// word-exact vs the same kernel on the contiguous copy of that view —
    /// the addressing change is the only difference. ~65k gate values, plus
    /// specials. ~65k word-exact values per draw.
    #[test]
    fn gated_rms_strided_z_view_matches_contiguous() {
        use super::{gated_rms, gated_rms_silu};
        use lisa_mlx::ops::indexing::IndexOp;
        use lisa_mlx::{Dtype, Stream};
        let stream = Stream::thread_local_or_default();
        // Dv MUST be 128 (the kernel reads 32 lanes x N_READS=4 per head row);
        // hv/dv are free.
        let (b, s, hv, dv) = (16i32, 16i32, 4i32, 128i32);
        let eps = 1e-6f32;
        let width = (hv * dv + 8) as usize; // fused-proj row width
        let mut seed = 90210u64;
        let mut rnd = move || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((seed >> 40) as f32 / 8_388_608.0) - 1.0
        };
        let rows = (b * s) as usize;
        let cols = (hv * dv) as usize;
        let n = rows * cols;
        let yv: Vec<f32> = (0..n).map(|_| rnd() * 4.0 - 2.0).collect();
        // The fused projection rows: 8 junk lanes before the z block per row.
        let mut proj: Vec<f32> = {
            let v: Vec<f32> = (0..rows * width).map(|_| rnd() * 8.0 - 4.0).collect();
            v
        };
        for sp in [0.0f32, -0.0, 20.0, -20.0, 88.0, -88.0, 0.5, -3.5] {
            let idx = (sp as usize).wrapping_mul(7919) % proj.len();
            proj[idx] = sp;
        }
        let wv: Vec<f32> = (0..dv as usize).map(|_| rnd() * 2.0 - 1.0).collect();
        let y = Array::from_slice(&yv, &[b, s, hv, dv])
            .as_dtype(Dtype::Bfloat16)
            .unwrap();
        let proj_a = Array::from_slice(&proj, &[b, s, width as i32])
            .as_dtype(Dtype::Bfloat16)
            .unwrap();
        let w = Array::from_slice(&wv, &[dv])
            .as_dtype(Dtype::Bfloat16)
            .unwrap();
        // The z slice VIEW exactly as gdn.rs builds it.
        let z_view = proj_a.index((.., .., 8..(8 + hv * dv)));
        let z_contig = z_view.copied().unwrap();

        for (name, k) in [("silu", 0usize), ("sigmoid", 1)] {
            let (fused, reference) = if k == 0 {
                (
                    gated_rms_silu(&y, &z_view, &w, hv, dv, eps, &stream).expect("kernel"),
                    gated_rms_silu(&y, &z_contig, &w, hv, dv, eps, &stream).expect("kernel"),
                )
            } else {
                (
                    gated_rms(&y, &z_view, &w, hv, dv, eps, &stream).expect("kernel"),
                    gated_rms(&y, &z_contig, &w, hv, dv, eps, &stream).expect("kernel"),
                )
            };
            let fv = fused
                .as_dtype(Dtype::Float32)
                .unwrap()
                .reshape(&[n as i32])
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            let cv = reference
                .as_dtype(Dtype::Float32)
                .unwrap()
                .reshape(&[n as i32])
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            let bad: Vec<usize> = (0..n)
                .filter(|&i| fv[i].to_bits() != cv[i].to_bits())
                .collect();
            assert!(
                bad.is_empty(),
                "{name}: {} mismatches, first at {:?}",
                bad.len(),
                bad.first()
            );
        }
    }
}

#[cfg(test)]
mod attn_mach_ab {
    use lisa_mlx::Array;
    #[test]
    fn attn_gate_machinery_ab() {
        let stream = lisa_mlx::Stream::thread_local_or_default();
        let (rows, heads, d) = (1usize, 24usize, 256usize);
        let hw = heads * d;
        let mut seed = 31337u64;
        let mut rnd = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((seed >> 40) as f32 / 8_388_608.0) - 1.0
        };
        let mk = |shape: &[i32], r: &mut dyn FnMut() -> f32| {
            let n: usize = shape.iter().map(|&x| x as usize).product();
            let v: Vec<f32> = (0..n)
                .map(|_| half::bf16::from_f32(r() * 4.0).to_f32())
                .collect();
            Array::from_slice(&v, shape)
                .as_dtype(lisa_mlx::Dtype::Bfloat16)
                .unwrap()
        };
        let att = mk(&[rows as i32, hw as i32], &mut rnd);
        let gate = mk(&[rows as i32, hw as i32], &mut rnd);
        let dir = String::from("/tmp");
        if let Some(r) = super::attn_gate(&att, &gate, heads as i32, d as i32, &stream) {
            let f = r.as_dtype(lisa_mlx::Dtype::Float32).unwrap();
            let s = f.as_slice::<f32>();
            let _ = std::fs::write(format!("{dir}/attn_gate.bin"), unsafe {
                std::slice::from_raw_parts(s.as_ptr() as *const u8, s.len() * 4)
            });
        }
    }
}

/// Force `get_or_init` on every kernel static in this module (see
/// `crate::models::qwen4::warm_kernels`). The Metal pipeline itself is
/// compiled lazily on first `apply` (template-keyed), so this materialises
/// the kernel handles only.
pub fn warm() {
    let _ = gated_rms_kernel();
    let _ = hc_mix_kernel();
    let _ = swiglu2_kernel();
    let _ = silu_head_kernel();
    let _ = inject_norm_kernel();
    let _ = attn_prep_split_kernel();
    let _ = attn_gate_kernel();
}
