//! Custom Metal kernels ported from the reference engine fork.
//!
//! Numerics are preserved exactly: same source strings, same template
//! parameters, same grid/threadgroup shapes as the Swift reference.

use crate::ffi::{MetalKernel, OutputArg, TemplateArg};
use lisa_mlx::ops::indexing::IndexOp;
use lisa_mlx::{ops, Array, Dtype, Stream};

/// The gated delta rule scan kernel from `GatedDelta.swift`.
///
/// Inputs: q [B,T,Hk,Dk], k [B,T,Hk,Dk], v [B,T,Hv,Dv], g [B,T,Hv] f32,
/// beta [B,T,Hv] f32, state_in [B,Hv,Dv,Dk], T scalar i32.
/// Outputs: y [B,T,Hv,Dv] (activation dtype), state_out [B,Hv,Dv,Dk] (StT).
const GATED_DELTA_SOURCE: &str = include_str!("kernels/gated_delta.metal");

/// Shared helpers for the fused elementwise kernels, ported verbatim from the
/// engine's `TrackFastKernels.exactHeader`.
pub const EXACT_HEADER: &str = include_str!("kernels/exact_header.metal");

/// `track_p12_moe_sorted_combine`: the routed combine reads the SORTED expert
/// rows through the inverse permutation the sort already produced, keeping the
/// float32 products in original (token, slot) order and the same small-column
/// reduction tree. Replaces the f32 [B,S,K,H] materialisation + scatter + the
/// 10-op column reduce with one launch.
///
/// `routed` is the sorted down output `[rows*K, H]`, `w` the float32 weights in
/// ORIGINAL slot order `[rows*K]`, `inverse_order` u32 `[rows*K]`, `shared`
/// `[rows, H]`, `gate` bf16 `[rows]` -> `[rows, H]`.
const MOE_COMBINE_SOURCE: &str = include_str!("kernels/moe_combine.metal");

static MOE_COMBINE: std::sync::OnceLock<Option<MetalKernel>> = std::sync::OnceLock::new();

fn moe_combine_kernel() -> &'static Option<MetalKernel> {
    MOE_COMBINE.get_or_init(|| {
        MetalKernel::new(
            "track_p12_moe_sorted_combine",
            &["routed", "w", "shared", "gate", "inverse_order"],
            &["out"],
            MOE_COMBINE_SOURCE,
            EXACT_HEADER,
            true,
            false,
        )
        .ok()
    })
}

/// Run the sorted MoE combine. Returns `[rows, H]` bf16, or `None` if the
/// kernel could not be built.
#[allow(clippy::too_many_arguments)]
pub fn moe_combine(
    routed: &Array,
    w: &Array,
    inverse_order: &Array,
    shared: &Array,
    gate: &Array,
    top_k: i32,
    h: i32,
    rows: i32,
    stream: &Stream,
) -> Option<Array> {
    let kernel = moe_combine_kernel().as_ref()?;
    let inputs: [&Array; 5] = [routed, w, shared, gate, inverse_order];
    let template = [
        TemplateArg::Dtype("InT", Dtype::Bfloat16),
        TemplateArg::Int("K", top_k),
        TemplateArg::Int("H", h),
    ];
    let outputs = [OutputArg {
        shape: vec![rows, h],
        dtype: Dtype::Bfloat16,
    }];
    let out = kernel
        .apply(&inputs, &template, (h, rows, 1), (256, 1, 1), &outputs, stream)
        .ok()?;
    out.into_iter().next()
}

/// `track_copy_rows_into`: append `src` `[R, s, D]` into `dst` `[R, cap, D]` at
/// row offset `off`, touching only the `s` appended rows. Replaces the
/// `slice_assign` (O(cap)) in the KV cache and the indexer tape.
const COPY_ROWS_SOURCE: &str = include_str!("../../kernels/common/copy_rows.metal");

static COPY_ROWS: std::sync::OnceLock<Option<MetalKernel>> = std::sync::OnceLock::new();

fn copy_rows_kernel() -> &'static Option<MetalKernel> {
    COPY_ROWS.get_or_init(|| {
        MetalKernel::new(
            "track_copy_rows_into",
            &["src", "dst", "off"],
            &["out"],
            COPY_ROWS_SOURCE,
            "",
            true,
            false,
        )
        .ok()
    })
}

/// Append `src` `[.., s, D]` into the caller's `dst` `[.., cap, D]` buffer at
/// row offset `off`. `dst` is bound as both a shape-carrying input and the
/// preallocated output so the encoder sees the dependency on the cache buffer.
pub fn copy_rows_into(src: &Array, dst: &Array, off: i32, stream: &Stream) -> Option<Array> {
    let kernel = copy_rows_kernel().as_ref()?;
    let off_a = Array::from_int(off);
    let inputs: [&Array; 3] = [src, dst, &off_a];
    let outputs = [OutputArg {
        shape: dst.shape().to_vec(),
        dtype: Dtype::Bfloat16,
    }];
    let grid = (src.size() as i32, 1, 1);
    let out = kernel
        .apply_into(&inputs, &[], grid, (256, 1, 1), &outputs, &[dst], stream)
        .ok()?;
    out.into_iter().next()
}

/// `track_route_block_counts` + `track_route_counting_scatter`: the stable
/// counting sort that produces the routed-assignment permutation
/// `(sortedIDs, tokenRows, inverse)` in two launches, replacing MLX's multi-block
/// merge sort (7 launches per `argSort`, 14 per layer). MLX's argsort is stable
/// and a stable counting sort is the unique stable permutation, so the two agree
/// element for element.
const ROUTE_BLOCK_COUNTS_SOURCE: &str = include_str!("kernels/route_block_counts.metal");

const ROUTE_COUNTING_SCATTER_SOURCE: &str = include_str!("kernels/route_counting_scatter.metal");

static ROUTE_BLOCK_COUNTS: std::sync::OnceLock<Option<MetalKernel>> = std::sync::OnceLock::new();
static ROUTE_COUNTING_SCATTER: std::sync::OnceLock<Option<MetalKernel>> = std::sync::OnceLock::new();

/// Stable counting sort of `ids` over `e` expert buckets. `ids` is `[R]` (any
/// integer dtype; cast to uint32). Returns `(sorted_ids, token_rows, inverse)`,
/// each `[R]` uint32, or `None` if the shape is outside the supported window
/// (`E % 256 == 0`, `E <= 4096`, `R >= 256`, `R % 256 == 0`, `R % top_k == 0`).
pub fn route_counting_sort(
    ids: &Array,
    e: usize,
    top_k: usize,
    stream: &Stream,
) -> Option<(Array, Array, Array)> {
    if e == 0 || e % 256 != 0 || e > 4096 || top_k == 0 {
        return None;
    }
    let r = ids.dim(0) as usize;
    // The last block may be partial (the kernels pad it with the out-of-range
    // id E, which no bucket counts); only the expert count must divide R.
    if r == 0 || r % top_k != 0 {
        return None;
    }
    let blk = 256i32;
    let nb = r.div_ceil(256) as i32;
    let idsu = ids.as_dtype(Dtype::Uint32).ok()?;

    // `track_route_block_counts` only produces correct counts for FULL blocks;
    // a partial last block comes back all-zero (verified directly). Pad the ids
    // to a multiple of the block with the out-of-range sentinel `E`, which the
    // kernel's `valid` ballot excludes, so every block is full. The scatter
    // still sees the unpadded ids and the logical `r`.
    let padded = (nb as usize) * 256;
    let ids_counts = if padded == r {
        idsu.clone()
    } else {
        let pad = Array::from_slice(&vec![e as u32; padded - r], &[(padded - r) as i32]);
        ops::concatenate(&[&idsu, &pad], 0).ok()?
    };
    let r_counts = padded as i32;

    let count_kernel = ROUTE_BLOCK_COUNTS.get_or_init(|| {
        MetalKernel::new(
            "track_route_block_counts",
            &["ids"],
            &["counts"],
            ROUTE_BLOCK_COUNTS_SOURCE,
            "",
            true,
            false,
        )
        .ok()
    });
    let counts = count_kernel.as_ref()?.apply(
        &[&ids_counts],
        &[
            TemplateArg::Int("E", e as i32),
            TemplateArg::Int("BLK", blk),
            TemplateArg::Int("R", r_counts),
        ],
        (r_counts, 1, 1),
        (blk, 1, 1),
        &[OutputArg {
            shape: vec![nb * e as i32],
            dtype: Dtype::Uint32,
        }],
        stream,
    ).ok()?
    .into_iter()
    .next()?;

    let scatter_kernel = ROUTE_COUNTING_SCATTER.get_or_init(|| {
        MetalKernel::new(
            "track_route_counting_scatter",
            &["ids", "counts"],
            &["sorted_ids", "token_rows", "inverse"],
            ROUTE_COUNTING_SCATTER_SOURCE,
            "",
            true,
            false,
        )
        .ok()
    });
    let mut outs = scatter_kernel.as_ref()?.apply(
        &[&ids_counts, &counts],
        &[
            TemplateArg::Int("E", e as i32),
            TemplateArg::Int("BLK", blk),
            TemplateArg::Int("R", r_counts),
            TemplateArg::Int("NB", nb),
            TemplateArg::Int("TOPK", top_k as i32),
        ],
        (r_counts, 1, 1),
        (blk, 1, 1),
        &[
            OutputArg { shape: vec![r as i32], dtype: Dtype::Uint32 },
            OutputArg { shape: vec![r as i32], dtype: Dtype::Uint32 },
            OutputArg { shape: vec![r as i32], dtype: Dtype::Uint32 },
        ],
        stream,
    ).ok()?
    .into_iter();
    Some((outs.next()?, outs.next()?, outs.next()?))
}

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
const GDN_PREP_SOURCE: &str = include_str!("kernels/gdn_prep.metal");

static GDN_PREP: std::sync::OnceLock<Option<MetalKernel>> = std::sync::OnceLock::new();

fn gdn_prep_kernel() -> &'static Option<MetalKernel> {
    GDN_PREP.get_or_init(|| {
        MetalKernel::new(
            "track_p12_gdn_prep_split_inputs",
            &["proj", "conv_state", "conv_w", "neg_exp_alog", "dt_bias", "b_gate", "a_gate"],
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
    let inputs: [&Array; 7] = [qkv, conv_state, conv_w, neg_exp_alog, dt_bias, b_gate, a_gate];
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
        OutputArg { shape: vec![b, t_len, hk, dk], dtype: Dtype::Bfloat16 },
        OutputArg { shape: vec![b, t_len, hk, dk], dtype: Dtype::Bfloat16 },
        OutputArg { shape: vec![b, t_len, hv, dv], dtype: Dtype::Bfloat16 },
        OutputArg { shape: vec![b, t_len, hv], dtype: Dtype::Float32 },
        OutputArg { shape: vec![b, t_len, hv], dtype: Dtype::Float32 },
        OutputArg { shape: vec![b, kc - 1, conv_dim], dtype: Dtype::Bfloat16 },
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
        it.next()?, it.next()?, it.next()?, it.next()?, it.next()?, it.next()?,
    ))
}

/// `track_gdn_prep`: the same prep with the two gate rows read from offsets
/// `B_OFF`/`A_OFF` in the concatenated `proj [B,T,PROJ_W]` (the verify/capture
/// and small-window path; the engine only splits for eligible wide prefill).
///
/// Inputs: `proj [B,T,PROJ_W]`, `conv_state [B,KC-1,CONV_DIM]`,
/// `conv_w [CONV_DIM,KC,1]`, `neg_exp_alog [Hv] f32`, `dt_bias [Hv]`.
const GDN_PREP_FUSED_SOURCE: &str = include_str!("kernels/gdn_prep_fused.metal");

static GDN_PREP_FUSED: std::sync::OnceLock<Option<MetalKernel>> = std::sync::OnceLock::new();

fn gdn_prep_fused_kernel() -> &'static Option<MetalKernel> {
    GDN_PREP_FUSED.get_or_init(|| {
        MetalKernel::new(
            "track_gdn_prep",
            &["proj", "conv_state", "conv_w", "neg_exp_alog", "dt_bias"],
            &["qn", "kn", "vv", "g", "beta", "conv_out"],
            GDN_PREP_FUSED_SOURCE,
            EXACT_HEADER,
            true,
            false,
        )
        .ok()
    })
}

/// Run the fused-projection GDN prep. Returns `(qn, kn, vv, g, beta, conv_out)`.
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
) -> Option<(Array, Array, Array, Array, Array, Array)> {
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
        OutputArg { shape: vec![b, t_len, hk, dk], dtype: Dtype::Bfloat16 },
        OutputArg { shape: vec![b, t_len, hk, dk], dtype: Dtype::Bfloat16 },
        OutputArg { shape: vec![b, t_len, hv, dv], dtype: Dtype::Bfloat16 },
        OutputArg { shape: vec![b, t_len, hv], dtype: Dtype::Float32 },
        OutputArg { shape: vec![b, t_len, hv], dtype: Dtype::Float32 },
        OutputArg { shape: vec![b, kc - 1, conv_dim], dtype: Dtype::Bfloat16 },
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
        it.next()?, it.next()?, it.next()?, it.next()?, it.next()?, it.next()?,
    ))
}

/// `track_gated_rms` (split-z form): the output gated RMS over `Dv` per value/// head (the engine's butterfly reduction), then `sigmoid(z) * out` in one
/// launch. `y`, `zproj` and `out` are flat `[rows, Hv, Dv]`, `zproj` is
/// `[rows, Hv*Dv]`, `w` is `[Dv]`.
const GATED_RMS_SOURCE: &str = include_str!("kernels/gated_rms.metal");

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
    let z_flat = zproj.reshape(&[rows, hv * dv]).ok()?;
    let inputs: [&Array; 3] = [&y_flat, &z_flat, w];
    let template = [
        TemplateArg::Dtype("InT", Dtype::Bfloat16),
        TemplateArg::Int("Hv", hv),
        TemplateArg::Int("Dv", dv),
        TemplateArg::Int("EPS_BITS", eps.to_bits() as i32),
    ];
    let outs = [OutputArg {
        shape: vec![rows * hv * dv],
        dtype: Dtype::Bfloat16,
    }];
    let out = kernel
        .apply(&inputs, &template, (32, hv, rows), (32, 1, 1), &outs, stream)
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
const HC_MIX_SOURCE: &str = include_str!("kernels/hc_mix.metal");

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
        OutputArg { shape: vec![rows, hidden], dtype: Dtype::Bfloat16 },
        OutputArg { shape: vec![rows, hc], dtype: Dtype::Bfloat16 },
    ];
    let out = kernel
        .apply(&inputs, &template, (hidden, rows, 1), (256, 1, 1), &outs, stream)
        .ok()?;
    let mut it = out.into_iter();
    Some((it.next()?, it.next()?))
}

/// `track_swiglu2`: `mlx_silu(gate) * up` over separate gate/up arrays, bf16.
const SWIGLU2_SOURCE: &str = include_str!("kernels/swiglu2.metal");

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
const SILU_HEAD_SOURCE: &str = include_str!("kernels/silu_head.metal");

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
    let outs = [OutputArg { shape: vec![b, f], dtype: Dtype::Bfloat16 }];
    kernel
        .apply(&inputs, &template, (b * f, 1, 1), (256, 1, 1), &outs, stream)
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
    let outs = [OutputArg { shape: vec![rows, width], dtype: Dtype::Bfloat16 }];
    kernel
        .apply(&inputs, &template, (width, rows, 1), (256, 1, 1), &outs, stream)
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
const INJECT_NORM_SOURCE: &str = include_str!("kernels/inject_norm.metal");

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
        OutputArg { shape: vec![rows, w], dtype: Dtype::Bfloat16 },
        OutputArg { shape: vec![rows, w], dtype: Dtype::Bfloat16 },
    ];
    let result = kernel
        .apply(&inputs, &template, (hidden / 4, hc, rows), (hidden / 4, 1, 1), &outs, stream)
        .ok()?;
    let mut it = result.into_iter();
    Some((it.next()?, it.next()?))
}

/// PLE helper header (adds `mlx_maximum`/`mlx_sign`/abs/sqrt).
const PLE_HEADER_EXTRA: &str = include_str!("kernels/ple_header_extra.metal");

/// `track_ple_prod`: `norm_key(keyFlat) * norm_query(stream)` with the engine's
/// butterfly RMS on both, in one launch.
const PLE_PROD_SOURCE: &str = include_str!("kernels/ple_prod.metal");

/// `track_ple_gated`: the gate scalar chain, the gated value, norm_conv.
const PLE_GATED_SOURCE: &str = include_str!("kernels/ple_gated.metal");

/// `track_ple_conv`: dilated depthwise conv + silu + residual add.
const PLE_CONV_SOURCE: &str = include_str!("kernels/ple_conv.metal");

static PLE_PROD: std::sync::OnceLock<Option<MetalKernel>> = std::sync::OnceLock::new();
static PLE_GATED: std::sync::OnceLock<Option<MetalKernel>> = std::sync::OnceLock::new();
static PLE_CONV: std::sync::OnceLock<Option<MetalKernel>> = std::sync::OnceLock::new();

fn ple_prod_kernel() -> &'static Option<MetalKernel> {
    PLE_PROD.get_or_init(|| {
        MetalKernel::new(
            "track_ple_prod",
            &["keyFlat", "stream", "kscale", "qscale", "eps"],
            &["prod"],
            PLE_PROD_SOURCE,
            EXACT_HEADER,
            true,
            false,
        )
        .ok()
    })
}
fn ple_gated_kernel() -> &'static Option<MetalKernel> {
    PLE_GATED.get_or_init(|| {
        let header = format!("{EXACT_HEADER}{PLE_HEADER_EXTRA}");
        MetalKernel::new(
            "track_ple_gated",
            &["g0", "value", "cscale", "divisor", "floorv", "eps"],
            &["gated", "normed"],
            PLE_GATED_SOURCE,
            &header,
            true,
            false,
        )
        .ok()
    })
}
fn ple_conv_kernel() -> &'static Option<MetalKernel> {
    PLE_CONV.get_or_init(|| {
        MetalKernel::new(
            "track_ple_conv",
            &["full", "convw", "gated"],
            &["out"],
            PLE_CONV_SOURCE,
            &format!("{EXACT_HEADER}{PLE_HEADER_EXTRA}"),
            true,
            false,
        )
        .ok()
    })
}

/// `track_ple_prod`. `keyFlat`/`stream` `[B,S,W]`, `kScale`/`qScale` `[W]`,
/// `eps` `[1]` -> `prod` `[B,S,W]`.
#[allow(clippy::too_many_arguments)]
pub fn ple_prod(
    key_flat: &Array,
    stream: &Array,
    k_scale: &Array,
    q_scale: &Array,
    eps: &Array,
    hc: i32,
    hidden: i32,
    stream_: &Stream,
) -> Option<Array> {
    let kernel = ple_prod_kernel().as_ref()?;
    let b = key_flat.dim(0);
    let s = key_flat.dim(1);
    let w = hc * hidden;
    let inputs: [&Array; 5] = [key_flat, stream, k_scale, q_scale, eps];
    let template = [
        TemplateArg::Dtype("InT", Dtype::Bfloat16),
        TemplateArg::Int("H", hidden),
        TemplateArg::Int("W", w),
        TemplateArg::Int("HC", hc),
    ];
    let outs = [OutputArg { shape: vec![b, s, w], dtype: Dtype::Bfloat16 }];
    kernel
        .apply(&inputs, &template, (hidden / 4, hc, b * s), (hidden / 4, 1, 1), &outs, stream_)
        .ok()?
        .into_iter()
        .next()
}

/// `track_ple_gated`. `g0` `[B,S,HC]`, `value` `[B,S,H]`, `cScale` `[W]`,
/// `divisor`/`floor` `[1]`, `eps` `[1]` -> `(gated, normed)` `[B,S,W]`.
#[allow(clippy::too_many_arguments)]
pub fn ple_gated(
    g0: &Array,
    value: &Array,
    c_scale: &Array,
    divisor: &Array,
    floor: &Array,
    eps: &Array,
    hc: i32,
    hidden: i32,
    stream_: &Stream,
) -> Option<(Array, Array)> {
    let kernel = ple_gated_kernel().as_ref()?;
    let b = value.dim(0);
    let s = value.dim(1);
    let w = hc * hidden;
    let inputs: [&Array; 6] = [g0, value, c_scale, divisor, floor, eps];
    let template = [
        TemplateArg::Dtype("InT", Dtype::Bfloat16),
        TemplateArg::Int("H", hidden),
        TemplateArg::Int("W", w),
        TemplateArg::Int("HC", hc),
    ];
    let outs = [
        OutputArg { shape: vec![b, s, w], dtype: Dtype::Bfloat16 },
        OutputArg { shape: vec![b, s, w], dtype: Dtype::Bfloat16 },
    ];
    let r = kernel
        .apply(&inputs, &template, (hidden / 4, hc, b * s), (hidden / 4, 1, 1), &outs, stream_)
        .ok()?;
    let mut it = r.into_iter();
    Some((it.next()?, it.next()?))
}

/// `track_ple_conv`. `full` `[B,N+S,W]`, `convw` `[W,KC]`, `gated` `[B,S,W]`
/// -> `[B,S,W]`.
pub fn ple_conv(
    full: &Array,
    conv_w: &Array,
    gated: &Array,
    dilation: i32,
    stream_: &Stream,
) -> Option<Array> {
    let kernel = ple_conv_kernel().as_ref()?;
    let b = gated.dim(0);
    let s = gated.dim(1);
    let w = gated.dim(2);
    let kc = conv_w.dim(1);
    let nin = full.dim(1);
    let inputs: [&Array; 3] = [full, conv_w, gated];
    let template = [
        TemplateArg::Dtype("InT", Dtype::Bfloat16),
        TemplateArg::Int("W", w),
        TemplateArg::Int("S", s),
        TemplateArg::Int("KC", kc),
        TemplateArg::Int("DIL", dilation),
        TemplateArg::Int("NIN", nin),
    ];
    let outs = [OutputArg { shape: vec![b, s, w], dtype: Dtype::Bfloat16 }];
    kernel
        .apply(&inputs, &template, (w, s, b), (256, 1, 1), &outs, stream_)
        .ok()?
        .into_iter()
        .next()
}

/// `track_router_gemv`: the decode router as a bf16 GEMV with f32 accumulate,
/// one token (`x` f32 `[K]`, `w` bf16 `[N,K]`) -> logits f32 `[N]`.
const ROUTER_GEMV_SOURCE: &str = include_str!("kernels/router_gemv.metal");

static ROUTER_GEMV: std::sync::OnceLock<Option<MetalKernel>> = std::sync::OnceLock::new();

fn router_gemv_kernel() -> &'static Option<MetalKernel> {
    ROUTER_GEMV.get_or_init(|| {
        MetalKernel::new(
            "track_router_gemv",
            &["x", "w"],
            &["out"],
            ROUTER_GEMV_SOURCE,
            "",
            true,
            false,
        )
        .ok()
    })
}

/// Run the decode router GEMV. `x` f32 `[K]`, `w` bf16 `[N,K]` -> f32 `[N]`.
pub fn router_gemv(x: &Array, w: &Array, stream: &Stream) -> Option<Array> {
    let kernel = router_gemv_kernel().as_ref()?;
    let n = w.dim(0);
    let k = w.dim(1);
    let rps: i32 = if k == 2560 && n == 512 { 1 } else { 4 };
    let inputs: [&Array; 2] = [x, w];
    let template = [
        TemplateArg::Dtype("T", Dtype::Bfloat16),
        TemplateArg::Int("K", k),
        TemplateArg::Int("N", n),
        TemplateArg::Int("RPS", rps),
    ];
    let outs = [OutputArg { shape: vec![n], dtype: Dtype::Float32 }];
    kernel
        .apply(
            &inputs,
            &template,
            (32 * (n / (4 * rps)), 1, 4),
            (32, 1, 4),
            &outs,
            stream,
        )
        .ok()?
        .into_iter()
        .next()
}

/// `track_gdn_lean_two_row`: the decode recurrence with two value rows per
/// simdgroup. The kernel body is static (see `kernels/models/qwen4/`).
const GDN_LEAN_TWO_ROW_SOURCE: &str =
    include_str!("kernels/gdn_lean_two_row.metal");
/// `track_gdn_rows`: the prefill recurrence with four value rows per simdgroup.
const GDN_ROWS_SOURCE: &str = include_str!("kernels/gdn_rows.metal");

static GDN_ROWS: std::sync::OnceLock<Option<MetalKernel>> = std::sync::OnceLock::new();
static GDN_TWO_ROW: std::sync::OnceLock<Option<MetalKernel>> = std::sync::OnceLock::new();

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
    gdn_two_row_impl(q, k, v, g, beta, state, capture, Some((out_y, out_state)), stream)
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
    let t_scalar = Array::from_int(t);
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
    let state_shape = if capture { vec![b * t, hv, dv, dk] } else { state.shape().to_vec() };
    let outs = [
        OutputArg { shape: vec![b, t, hv, dv], dtype: Dtype::Bfloat16 },
        OutputArg { shape: state_shape, dtype: Dtype::Float32 },
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
    let t_scalar = Array::from_int(t);
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
        OutputArg { shape: vec![b, t, hv, dv], dtype: Dtype::Bfloat16 },
        OutputArg { shape: state_shape, dtype: Dtype::Float32 },
    ];
    let r = kernel
        .apply(&inputs, &template, (32, dv / rows, b * hv), (32, 4, 1), &outs, stream)
        .ok()?;
    let mut it = r.into_iter();
    Some((it.next()?, it.next()?))
}

/// `track_swiglu`: `mlx_silu(gate) * up` over a fused `[.., 2F]` gate|up array.
// --- PLE S=1 fusion: track_ple_prepare_fuse2 + track_ple_convolution_fuse2 ---

const PLE_FUSE_HEADER: &str = include_str!("kernels/ple_fuse_header.metal");

const PLE_PREPARE_SOURCE: &str = include_str!("kernels/ple_prepare.metal");

const PLE_CONV_FUSE_SOURCE: &str = include_str!("kernels/ple_conv_fuse.metal");

static PLE_PREPARE: std::sync::OnceLock<Option<MetalKernel>> = std::sync::OnceLock::new();
static PLE_CONV_FUSE: std::sync::OnceLock<Option<MetalKernel>> = std::sync::OnceLock::new();

fn ple_prepare_kernel() -> &'static Option<MetalKernel> {
    PLE_PREPARE.get_or_init(|| {
        let h = format!("{EXACT_HEADER}{PLE_FUSE_HEADER}");
        MetalKernel::new(
            "track_ple_prepare_fuse2",
            &["key", "query", "value", "keyScale", "queryScale", "convScale", "convState"],
            &["gated", "full"],
            PLE_PREPARE_SOURCE,
            &h,
            true,
            false,
        )
        .ok()
    })
}
fn ple_conv_fuse_kernel() -> &'static Option<MetalKernel> {
    PLE_CONV_FUSE.get_or_init(|| {
        MetalKernel::new(
            "track_ple_convolution_fuse2",
            &["full", "weight", "gated"],
            &["out"],
            PLE_CONV_FUSE_SOURCE,
            EXACT_HEADER,
            true,
            false,
        )
        .ok()
    })
}

/// The S=1 PLE fusion: `(full [1,10,10240], output [1,1,10240])`.
#[allow(clippy::too_many_arguments)]
pub fn ple_fuse2(
    key: &Array,
    stream_arr: &Array,
    value: &Array,
    key_scale: &Array,
    query_scale: &Array,
    conv_scale: &Array,
    conv_state: &Array,
    conv_w: &Array,
    eps: f32,
    stream: &Stream,
) -> Option<(Array, Array)> {
    let prep = ple_prepare_kernel().as_ref()?;
    let conv = ple_conv_fuse_kernel().as_ref()?;
    let p_inputs: [&Array; 7] = [key, stream_arr, value, key_scale, query_scale, conv_scale, conv_state];
    let p_tmpl = [
        TemplateArg::Dtype("InT", Dtype::Bfloat16),
        TemplateArg::Int("EPS_BITS", eps.to_bits() as i32),
        TemplateArg::Int("DIVISOR_BITS", (2560f32.sqrt()).to_bits() as i32),
    ];
    let p_outs = [
        OutputArg { shape: vec![1, 1, 10240], dtype: Dtype::Bfloat16 },
        OutputArg { shape: vec![1, 10, 10240], dtype: Dtype::Bfloat16 },
    ];
    let r = prep
        .apply(&p_inputs, &p_tmpl, (640, 4, 1), (640, 1, 1), &p_outs, stream)
        .ok()?;
    let mut it = r.into_iter();
    let gated = it.next()?;
    let full = it.next()?;
    let c_inputs: [&Array; 3] = [&full, conv_w, &gated];
    let c_tmpl = [TemplateArg::Dtype("InT", Dtype::Bfloat16)];
    let c_outs = [OutputArg { shape: vec![1, 1, 10240], dtype: Dtype::Bfloat16 }];
    let out = conv
        .apply(&c_inputs, &c_tmpl, (32, 1, 4 * 10240), (32, 1, 4), &c_outs, stream)
        .ok()?
        .into_iter()
        .next()?;
    Some((full, out))
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

    let t_scalar = Array::from_int(t);
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

/// Ops fallback matching `gatedDeltaOps` in the reference: a per-t loop.
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

/// `track_p12_attn_prep_split_inputs`: q/k RMS + partial rope + v passthrough.
/// Inputs: `qkv` = [q|gate] rows `[B,S,2*HQ*D]`, `kproj`/`vproj` `[B,S,HK*D]`.
const ATTN_PREP_SPLIT_SOURCE: &str = include_str!("kernels/attn_prep_split.metal");

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
    qkv: &Array, kproj: &Array, vproj: &Array,
    q_norm: &Array, k_norm: &Array, cos: &Array, sin: &Array,
    heads: i32, kv_heads: i32, head_dim: i32, rotary_dims: i32, eps: f32,
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
        OutputArg { shape: vec![b, heads, s, head_dim], dtype: Dtype::Bfloat16 },
        OutputArg { shape: vec![b, kv_heads, s, head_dim], dtype: Dtype::Bfloat16 },
        OutputArg { shape: vec![b, kv_heads, s, head_dim], dtype: Dtype::Bfloat16 },
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
const ATTN_GATE_SOURCE: &str = include_str!("kernels/attn_gate.metal");

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
pub fn attn_gate(att: &Array, gate: &Array, heads: i32, head_dim: i32, stream: &Stream) -> Option<Array> {
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
    let outs = [OutputArg { shape: vec![rows, hw], dtype: Dtype::Bfloat16 }];
    kernel
        .apply(&inputs, &template, (hw, rows, 1), (256, 1, 1), &outs, stream)
        .ok()?
        .into_iter()
        .next()
}

/// Diagnostic: time the NAX indirect expert GEMMs against a dense quantized
/// matmul of the same FLOPs (`lisa indirect-bench`).
pub fn indirect_bench() {
    use std::time::Instant;
    let m = 10240i32;
    let k = 2560i32;
    let n = 640i32;
    let e = 512i32;
    let mut seed = 7u64;
    let mut rnd = move || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((seed >> 40) as f32 / 8388608.0) - 1.0
    };
    let wv: Vec<f32> = (0..(e * n * k) as usize).map(|_| rnd() * 0.1).collect();
    let w = Array::from_slice(&wv, &[e, n, k]).as_dtype(Dtype::Bfloat16).unwrap();
    let (gw, gs, gb) = ops::quantize(&w, 32, 4).unwrap();
    let wdv: Vec<f32> = (0..(e * k * n) as usize).map(|_| rnd() * 0.1).collect();
    let wd = Array::from_slice(&wdv, &[e, k, n]).as_dtype(Dtype::Bfloat16).unwrap();
    let (dw, ds, db) = ops::quantize(&wd, 32, 4).unwrap();
    let xv: Vec<f32> = (0..(1024 * k) as usize).map(|_| rnd()).collect();
    let x = Array::from_slice(&xv, &[1, 1024, k]).as_dtype(Dtype::Bfloat16).unwrap();
    let _ = (gw.eval(), gs.eval(), gb.eval(), dw.eval(), ds.eval(), db.eval(), x.eval());

    let idx: Vec<u32> = (0..m).map(|i| (i / 20) as u32).collect();
    let sorted_idx = Array::from_slice(&idx, &[m]);
    let tok: Vec<u32> = (0..m).map(|i| (i / 10) as u32).collect();
    let token_rows = Array::from_slice(&tok, &[m]);
    let stream = lisa_mlx::Stream::thread_local_or_default();
    let tiles = crate::prefill_indirect::tile_table(&sorted_idx, m, e, &stream).unwrap();
    let max_t = crate::prefill_indirect::max_tiles(m, e);
    let _ = tiles.eval();

    // warmup
    let act = crate::prefill_indirect::gate_up(
        &x, &gw, &gs, &gb, &gw, &gs, &gb, &sorted_idx, &token_rows, &tiles, max_t, n, k, m, &stream,
    )
    .unwrap();
    let _ = act.eval();
    let dn = crate::prefill_indirect::down(
        &act, &dw, &ds, &db, &sorted_idx, &tiles, max_t, k, n, m, &stream,
    )
    .unwrap();
    let _ = dn.eval();

    for i in 0..6 {
        let t = Instant::now();
        let a = crate::prefill_indirect::gate_up(
            &x, &gw, &gs, &gb, &gw, &gs, &gb, &sorted_idx, &token_rows, &tiles, max_t, n, k, m, &stream,
        )
        .unwrap();
        let _ = a.eval();
        println!("  gate_up iter {i}: {:.2} ms", t.elapsed().as_secs_f64() * 1e3);
    }

    let t = Instant::now();
    for _ in 0..5 {
        let a = crate::prefill_indirect::down(
            &act, &dw, &ds, &db, &sorted_idx, &tiles, max_t, k, n, m, &stream,
        )
        .unwrap();
        let _ = a.eval();
    }
    println!("down indirect:    {:.2} ms/iter", t.elapsed().as_secs_f64() * 200.0);

    // dense reference: one expert's weight over all M rows (same FLOPs)
    let xd = x.reshape(&[1024, k]).unwrap();
    let (w2, s2, b2) = ops::quantize(&w.index((0, .., ..)), 32, 4).unwrap();
    let _ = (w2.eval(), s2.eval(), b2.eval());
    let f = || ops::quantized_matmul(&xd, &w2, &s2, Some(&b2), true, 32, 4);
    let _ = f().unwrap().eval();
    let t = Instant::now();
    for _ in 0..5 {
        let _ = f().unwrap().eval();
    }
    println!("dense qmm M=1024: {:.2} ms/iter (x10 to compare M=10240)", t.elapsed().as_secs_f64() * 200.0);
}

/// `track_gdn_decode_complete`: the S=1 GDN (causal conv + q/k l2norm + the
/// gated delta rule + gated RMS) in one launch. Ported verbatim from
/// `reference/Runner/FastModel/TrackFastGDNDecode.swift`.
///
/// `proj` is the concatenated `[qkv | z | b | a]` projection (one GEMM); the
/// kernel reads z/b/a from their offsets in it.
const GDN_DECODE_COMPLETE_SOURCE: &str = include_str!("kernels/gdn_decode_complete.metal");

static GDN_DECODE_COMPLETE: std::sync::OnceLock<Option<MetalKernel>> = std::sync::OnceLock::new();

fn gdn_decode_complete_kernel() -> &'static Option<MetalKernel> {
    GDN_DECODE_COMPLETE.get_or_init(|| {
        MetalKernel::new(
            "track_gdn_decode_complete",
            &["proj", "conv_state", "conv_w", "neg_exp_alog", "dt_bias", "state_in", "w"],
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
    stream: &Stream,
) -> Option<(Array, Array, Array)> {
    let kernel = gdn_decode_complete_kernel().as_ref()?;
    let b = proj.dim(0);
    let inputs: [&Array; 7] = [proj, conv_state, conv_w, neg_exp_alog, dt_bias, state_in, norm_w];
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
    ];
    let outs = [
        OutputArg { shape: vec![b, hv, dv, dk], dtype: Dtype::Float32 },
        OutputArg { shape: vec![b, 1, hv * dv], dtype: Dtype::Bfloat16 },
        OutputArg { shape: vec![b, kc - 1, conv_dim], dtype: Dtype::Bfloat16 },
    ];
    let r = kernel
        .apply(&inputs, &template, (32, dv / 4, b * hv), (32, dv / 4, 1), &outs, stream)
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
            Array::from_slice(&v, shape).as_dtype(super::Dtype::Bfloat16).unwrap()
        };
        let proj = mk(&[b, pw], &mut rnd);
        let conv_state = mk(&[b, kc - 1, conv_dim], &mut rnd);
        let conv_w = mk(&[conv_dim, kc], &mut rnd);
        let negv: Vec<f32> = (0..hv as usize).map(|_| -rnd().abs()).collect();
        let neg_exp_alog = Array::from_slice(&negv, &[hv]).as_dtype(super::Dtype::Float32).unwrap();
        let dt_bias = mk(&[hv], &mut rnd);
        let sv: Vec<f32> = (0..(b * hv * dv * dk) as usize).map(|_| rnd() * 0.01).collect();
        let state_in = Array::from_slice(&sv, &[b, hv, dv, dk]).as_dtype(super::Dtype::Float32).unwrap();
        let norm_w = mk(&[dv], &mut rnd);
        let (gated, state_out, conv_out) = super::gdn_decode_complete(
            &proj, &conv_state, &conv_w, &neg_exp_alog, &dt_bias, &state_in, &norm_w,
            hk, hv, dk, dv, kc, conv_dim, pw, b_off, a_off, z_off, 1e-6, &stream,
        )?;
        let dir = std::env::var("LISA_AB_OUT").unwrap_or_else(|_| "/tmp".into());
        let w = |name: &str, a: &Array| {
            let f = a.as_dtype(super::Dtype::Float32).unwrap();
            let s = f.as_slice::<f32>();
            let _ = std::fs::write(format!("{dir}/{name}.bin"),
                unsafe { std::slice::from_raw_parts(s.as_ptr() as *const u8, s.len()*4) });
        };
        w("gated", &gated);
        w("state_out", &state_out);
        w("conv_out", &conv_out);
        Some(())
    }
}

#[cfg(test)]
mod mixer_avail {
    #[test]
    fn mixer_kernels_available() {
        // indirect probe: call with tiny inputs and see if the kernel path returns Some
        use lisa_mlx::Array;
        let stream = lisa_mlx::Stream::thread_local_or_default();
        let w = Array::from_slice(&vec![0f32; 8], &[2, 2, 2]).as_dtype(lisa_mlx::Dtype::Bfloat16).unwrap();
        let normed = Array::from_slice(&vec![0f32; 8], &[1, 2, 4]).as_dtype(lisa_mlx::Dtype::Bfloat16).unwrap();
        let inj = Array::from_slice(&vec![0f32; 4], &[1, 2, 2]).as_dtype(lisa_mlx::Dtype::Bfloat16).unwrap();
        eprintln!("hc_mix: {}", super::hc_mix(&w, &normed, &inj, 2, 2, 2, true, &stream).is_some());
        let lo = Array::from_slice(&vec![0f32; 4], &[1, 2, 2]).as_dtype(lisa_mlx::Dtype::Bfloat16).unwrap();
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
        let mut rnd = || { seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1); ((seed>>40) as f32/8_388_608.0)-1.0 };
        let mk = |shape: &[i32], r: &mut dyn FnMut()->f32| { let n: usize = shape.iter().map(|&d| d as usize).product(); let v: Vec<f32> = (0..n).map(|_| half::bf16::from_f32(r()*30.0).to_f32()).collect(); Array::from_slice(&v, shape).as_dtype(lisa_mlx::Dtype::Bfloat16).unwrap() };
        let lo = mk(&[rows as i32, width as i32], &mut rnd);
        let dir = std::env::var("LISA_AB_OUT").unwrap_or_else(|_| "/tmp".into());
        let w = |name: &str, a: &Array| { let f = a.as_dtype(lisa_mlx::Dtype::Float32).unwrap(); let s = f.as_slice::<f32>(); let _ = std::fs::write(format!("{dir}/{name}.bin"), unsafe { std::slice::from_raw_parts(s.as_ptr() as *const u8, s.len()*4) }); };
        if let Some(r) = super::silu_head(&lo, width as i32, &stream) { w("silu_head", &r); }
        // hc_mix: w [rows, hc*h], normed [rows, hc*h], inj [rows, hc]
        let ww = mk(&[rows as i32, (hc*width) as i32], &mut rnd);
        let normed = mk(&[rows as i32, (hc*width) as i32], &mut rnd);
        let inj = mk(&[rows as i32, hc as i32], &mut rnd);
        if let Some((input, inject_w)) = super::hc_mix(&ww, &normed, &inj, hc as i32, width as i32, rows as i32, true, &stream) {
            w("hc_mix_input", &input); w("hc_mix_inject", &inject_w);
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
        let hw = heads*d;
        let mut seed = 31337u64;
        let mut rnd = || { seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1); ((seed>>40) as f32/8_388_608.0)-1.0 };
        let mk = |shape: &[i32], r: &mut dyn FnMut()->f32| { let n: usize = shape.iter().map(|&x| x as usize).product(); let v: Vec<f32> = (0..n).map(|_| half::bf16::from_f32(r()*4.0).to_f32()).collect(); Array::from_slice(&v, shape).as_dtype(lisa_mlx::Dtype::Bfloat16).unwrap() };
        let att = mk(&[rows as i32, hw as i32], &mut rnd);
        let gate = mk(&[rows as i32, hw as i32], &mut rnd);
        let dir = std::env::var("LISA_AB_OUT").unwrap_or_else(|_| "/tmp".into());
        if let Some(r) = super::attn_gate(&att, &gate, heads as i32, d as i32, &stream) {
            let f = r.as_dtype(lisa_mlx::Dtype::Float32).unwrap(); let s = f.as_slice::<f32>();
            let _ = std::fs::write(format!("{dir}/attn_gate.bin"), unsafe { std::slice::from_raw_parts(s.as_ptr() as *const u8, s.len()*4) });
        }
    }
}

#[cfg(test)]
mod counting_sort_check {
    use lisa_mlx::Array;

    /// Regression: the counting sort must be a stable permutation into sorted
    /// expert order for every row count, including a partial last block (a
    /// non-multiple of the 256-row block size). A partial block previously
    /// produced all-zero bucket counts, which corrupted the MoE gather and
    /// caused a GPU page fault during long-context prefill.
    #[test]
    fn counting_sort_matches_reference() {
        let stream = lisa_mlx::Stream::thread_local_or_default();
        let e = 512usize;
        for &r in &[
            10240usize, 12800, 15360, 15370, 15530, 16000, 20480, 20490, 23480, 25600, 32760,
        ] {
            let mut seed = 4242u64;
            let mut rnd = || {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                ((seed >> 33) as u32)
            };
            let ids: Vec<i32> = (0..r).map(|_| (rnd() % e as u32) as i32).collect();
            let arr = Array::from_slice(&ids, &[r as i32]);
            let (sorted, _token, inv) =
                super::route_counting_sort(&arr, e, 10, &stream).expect("sort");
            let _ = sorted.eval();
            let _ = inv.eval();
            let s = sorted.as_slice::<u32>().to_vec();
            let iv = inv.as_slice::<u32>().to_vec();
            assert!(s.windows(2).all(|w| w[0] <= w[1]), "sorted_ids not sorted at r={r}");
            let mut cnt_ref = vec![0u32; e];
            for &x in &ids {
                cnt_ref[x as usize] += 1;
            }
            let mut cnt_got = vec![0u32; e];
            for &x in &s {
                cnt_got[x as usize] += 1;
            }
            assert_eq!(cnt_ref, cnt_got, "bucket counts differ at r={r}");
            let mut seen = vec![false; r];
            for &d in &iv {
                assert!((d as usize) < r, "inverse out of range at r={r}");
                seen[d as usize] = true;
            }
            assert!(seen.iter().all(|&b| b), "inverse not a permutation at r={r}");
        }
    }
}

