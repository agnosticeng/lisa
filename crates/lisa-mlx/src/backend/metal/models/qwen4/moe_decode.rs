//! Decode MoE kernels, ported verbatim from the engine's `TrackFastMoE.swift`.
//!
//! The one-token (`S == 1`) path: `track_moe_route_1` (top-k + softmax + the
//! shared-expert gate), `track_moe_gate_up_reuse_2row` (gate|up GEMVs + SwiGLU
//! for every routed slot and the shared expert), `track_moe_down_combine_1`
//! (down GEMV + the `col_reduce_small` expert fold + the shared gate add).

use lisa_mlx::{Array, Dtype, Stream};

use crate::ffi::{MetalKernel, OutputArg, TemplateArg};
use crate::moe_helpers as H;

fn compose(parts: &[&str]) -> String {
    parts.concat()
}

// --- track_moe_route_1 -----------------------------------------------------

const ROUTE_SOURCE: &str = include_str!("kernels/route.metal");

static ROUTE: std::sync::OnceLock<Option<MetalKernel>> = std::sync::OnceLock::new();
static ROUTE_WIDE: std::sync::OnceLock<Option<MetalKernel>> = std::sync::OnceLock::new();

fn route_kernel(wide: bool) -> &'static Option<MetalKernel> {
    let cell = if wide { &ROUTE_WIDE } else { &ROUTE };
    cell.get_or_init(|| {
        let header = compose(&[
            H::HELPERS_CORE,
            crate::kernels::EXACT_HEADER,
            H::MIXER_HEAD_TAIL,
            if wide { H::WIDE_HELPERS } else { H::WIDE_DECLS },
        ]);
        MetalKernel::new(
            if wide { "track_moe_route" } else { "track_moe_route_1" },
            &["logits", "x", "wg", "sgw", "bgw"],
            &["idx", "w", "gate"],
            ROUTE_SOURCE,
            &header,
            true,
            false,
        )
        .ok()
    })
}

/// `logits` f32 `[rows, E]`, `x` bf16 `[rows, KD]`, the shared-gate quantized
/// weight -> `(idx u32 [rows,K], w f32 [rows,K], gate bf16 [rows])`.
#[allow(clippy::too_many_arguments)]
pub fn route(
    logits: &Array,
    x: &Array,
    sg_w: &Array,
    sg_s: &Array,
    sg_b: &Array,
    top_k: i32,
    e: i32,
    kd: i32,
    rows: i32,
    stream: &Stream,
) -> Option<(Array, Array, Array)> {
    let wide = rows != 1;
    let kernel = route_kernel(wide).as_ref()?;
    let inputs: [&Array; 5] = [logits, x, sg_w, sg_s, sg_b];
    let template = [
        TemplateArg::Int("E", e),
        TemplateArg::Int("K", top_k),
        TemplateArg::Dtype("T", Dtype::Bfloat16),
        TemplateArg::Int("GS", 32),
        TemplateArg::Int("BITS", 4),
        TemplateArg::Int("KD", kd),
        TemplateArg::Int("VPT", rows),
        TemplateArg::Bool("HAS_GATE", true),
    ];
    let outs = [
        OutputArg { shape: vec![rows, top_k], dtype: Dtype::Uint32 },
        OutputArg { shape: vec![rows, top_k], dtype: Dtype::Float32 },
        OutputArg { shape: vec![rows], dtype: Dtype::Bfloat16 },
    ];
    let r = kernel
        .apply(
            &inputs,
            &template,
            (32, rows * 2, 1),
            (32, 2, 1),
            &outs,
            stream,
        )
        .ok()?;
    let mut it = r.into_iter();
    Some((it.next()?, it.next()?, it.next()?))
}

// --- track_moe_gate_up_reuse_2row ------------------------------------------

const GATE_UP_REUSE_SOURCE: &str = include_str!("kernels/gate_up_reuse.metal");

static GATE_UP_REUSE: std::sync::OnceLock<Option<MetalKernel>> = std::sync::OnceLock::new();

fn gate_up_reuse_kernel() -> &'static Option<MetalKernel> {
    GATE_UP_REUSE.get_or_init(|| {
        let header = compose(&[
            H::HELPERS_CORE,
            crate::kernels::EXACT_HEADER,
            H::GATE_UP_REUSE_HELPERS,
        ]);
        MetalKernel::new(
            "track_moe_gate_up_reuse_2row",
            &["wg", "sg", "bg", "wu", "su", "bu", "wsh", "ssh", "bsh", "x", "idx", "xrow"],
            &["act"],
            GATE_UP_REUSE_SOURCE,
            &header,
            true,
            false,
        )
        .ok()
    })
}

/// Routed `gate`/`up` and the fused shared `gate|up` -> `act [BR + S, N]`.
#[allow(clippy::too_many_arguments)]
pub fn gate_up_act(
    wg: &Array, sg: &Array, bg: &Array,
    wu: &Array, su: &Array, bu: &Array,
    wsh: &Array, ssh: &Array, bsh: &Array,
    x: &Array, idx: &Array, xrow: &Array,
    br: i32, n: i32, kd: i32, s: i32,
    stream: &Stream,
) -> Option<Array> {
    let kernel = gate_up_reuse_kernel().as_ref()?;
    let rps = 2;
    let inputs: [&Array; 12] = [wg, sg, bg, wu, su, bu, wsh, ssh, bsh, x, idx, xrow];
    let template = [
        TemplateArg::Dtype("T", Dtype::Bfloat16),
        TemplateArg::Int("GS", 32),
        TemplateArg::Int("BITS", 4),
        TemplateArg::Int("N", n),
        TemplateArg::Int("KD", kd),
        TemplateArg::Int("BR", br),
        TemplateArg::Int("RPS", rps),
    ];
    let outs = [OutputArg { shape: vec![br + s, n], dtype: Dtype::Bfloat16 }];
    kernel
        .apply(
            &inputs,
            &template,
            (32, n / rps, br + 1),
            (32, 2, 1),
            &outs,
            stream,
        )
        .ok()?
        .into_iter()
        .next()
}

// --- track_moe_down_combine_1 ----------------------------------------------

const DOWN_COMBINE_SOURCE: &str = include_str!("kernels/down_combine.metal");

static DOWN_COMBINE: std::sync::OnceLock<Option<MetalKernel>> = std::sync::OnceLock::new();

fn down_combine_kernel() -> &'static Option<MetalKernel> {
    DOWN_COMBINE.get_or_init(|| {
        let header = compose(&[
            H::HELPERS_CORE,
            crate::kernels::EXACT_HEADER,
            H::REG_HELPERS,
            H::WIDE_DECLS,
        ]);
        MetalKernel::new(
            "track_moe_down_combine_1",
            &["wd", "sd", "bd", "wsd", "ssd", "bsd", "act", "idx", "w", "gate"],
            &["out"],
            DOWN_COMBINE_SOURCE,
            &header,
            true,
            false,
        )
        .ok()
    })
}

/// Routed `down` + the shared `down`, the expert fold, the shared-gate add.
#[allow(clippy::too_many_arguments)]
pub fn down_combine(
    wd: &Array, sd: &Array, bd: &Array,
    wsd: &Array, ssd: &Array, bsd: &Array,
    act: &Array, idx: &Array, w: &Array, gate: &Array,
    top_k: i32, f: i32, h: i32, br: i32, s: i32,
    stream: &Stream,
) -> Option<Array> {
    let kernel = down_combine_kernel().as_ref()?;
    let ksg: i32 = if top_k % 5 == 0 { 5 } else { 1 };
    let rps = 2;
    let fast = (f % 512 == 0) && (h % 8 == 0);
    let inputs: [&Array; 10] = [wd, sd, bd, wsd, ssd, bsd, act, idx, w, gate];
    let template = [
        TemplateArg::Dtype("T", Dtype::Bfloat16),
        TemplateArg::Int("GS", 32),
        TemplateArg::Int("BITS", 4),
        TemplateArg::Int("H", h),
        TemplateArg::Int("F", f),
        TemplateArg::Int("K", top_k),
        TemplateArg::Bool("FAST", fast),
        TemplateArg::Int("BR", br),
        TemplateArg::Int("VPT", 1),
        TemplateArg::Int("KSG", ksg),
        TemplateArg::Int("RPS", rps),
    ];
    let outs = [OutputArg { shape: vec![s, h], dtype: Dtype::Bfloat16 }];
    // OPT-SHAREDROWSG: the one-token path gives each of the RPS shared
    // rows its own simdgroup, so the threadgroup grows by RPS groups (the
    // kernel guards the routed walk with `sgi < KSG`). `grid` is in threads,
    // so it scales with the threadgroup to keep `h / rps` threadgroups.
    let tg_y = if top_k == 10 && ksg >= 5 { ksg + rps } else { ksg };
    kernel
        .apply(
            &inputs,
            &template,
            (32, (h / rps) * tg_y, s),
            (32, tg_y, 1),
            &outs,
            stream,
        )
        .ok()?
        .into_iter()
        .next()
}

// --- track_moe_gate_up_act (wide, S in 2..=8) ------------------------------

const GATE_UP_ACT_SOURCE: &str = include_str!("kernels/gate_up_act.metal");

static GATE_UP_ACT: std::sync::OnceLock<Option<MetalKernel>> = std::sync::OnceLock::new();

fn gate_up_act_kernel() -> &'static Option<MetalKernel> {
    GATE_UP_ACT.get_or_init(|| {
        let header = compose(&[
            H::HELPERS_CORE,
            crate::kernels::EXACT_HEADER,
            H::REG_HELPERS,
            H::WIDE_HELPERS,
        ]);
        MetalKernel::new(
            "track_moe_gate_up_act",
            &["wg", "sg", "bg", "wu", "su", "bu", "wsh", "ssh", "bsh", "x", "idx", "xrow"],
            &["act"],
            GATE_UP_ACT_SOURCE,
            &header,
            true,
            false,
        )
        .ok()
    })
}

/// Wide (2..=8 rows) routed + shared gate|up -> `act [BR + S, N]`.
#[allow(clippy::too_many_arguments)]
pub fn gate_up_act_wide(
    wg: &Array, sg: &Array, bg: &Array,
    wu: &Array, su: &Array, bu: &Array,
    wsh: &Array, ssh: &Array, bsh: &Array,
    x: &Array, idx: &Array, xrow: &Array,
    br: i32, n: i32, kd: i32, s: i32,
    stream: &Stream,
) -> Option<Array> {
    let kernel = gate_up_act_kernel().as_ref()?;
    let fast = (kd % 512 == 0) && (n % 8 == 0);
    let inputs: [&Array; 12] = [wg, sg, bg, wu, su, bu, wsh, ssh, bsh, x, idx, xrow];
    let template = [
        TemplateArg::Dtype("T", Dtype::Bfloat16),
        TemplateArg::Int("GS", 32),
        TemplateArg::Int("BITS", 4),
        TemplateArg::Int("N", n),
        TemplateArg::Int("KD", kd),
        TemplateArg::Bool("FAST", fast),
        TemplateArg::Int("BR", br),
        TemplateArg::Int("VPT", s),
    ];
    let outs = [OutputArg { shape: vec![br + s, n], dtype: Dtype::Bfloat16 }];
    kernel
        .apply(
            &inputs,
            &template,
            (32, (n / 8) * 2, br + 1),
            (32, 2, 1),
            &outs,
            stream,
        )
        .ok()?
        .into_iter()
        .next()
}

// --- track_moe_down_combine (wide) -----------------------------------------

static DOWN_COMBINE_WIDE: std::sync::OnceLock<Option<MetalKernel>> = std::sync::OnceLock::new();

fn down_combine_wide_kernel() -> &'static Option<MetalKernel> {
    DOWN_COMBINE_WIDE.get_or_init(|| {
        let header = compose(&[
            H::HELPERS_CORE,
            crate::kernels::EXACT_HEADER,
            H::REG_HELPERS,
            H::WIDE_HELPERS,
        ]);
        MetalKernel::new(
            "track_moe_down_combine",
            &["wd", "sd", "bd", "wsd", "ssd", "bsd", "act", "idx", "w", "gate"],
            &["out"],
            DOWN_COMBINE_SOURCE,
            &header,
            true,
            false,
        )
        .ok()
    })
}

/// Wide down + combine -> `[S, H]`.
#[allow(clippy::too_many_arguments)]
pub fn down_combine_wide(
    wd: &Array, sd: &Array, bd: &Array,
    wsd: &Array, ssd: &Array, bsd: &Array,
    act: &Array, idx: &Array, w: &Array, gate: &Array,
    top_k: i32, f: i32, h: i32, br: i32, s: i32,
    stream: &Stream,
) -> Option<Array> {
    let kernel = down_combine_wide_kernel().as_ref()?;
    let ksg: i32 = if top_k % 5 == 0 { 5 } else { 1 };
    let rps = 4;
    let fast = (f % 512 == 0) && (h % 8 == 0);
    let inputs: [&Array; 10] = [wd, sd, bd, wsd, ssd, bsd, act, idx, w, gate];
    let template = [
        TemplateArg::Dtype("T", Dtype::Bfloat16),
        TemplateArg::Int("GS", 32),
        TemplateArg::Int("BITS", 4),
        TemplateArg::Int("H", h),
        TemplateArg::Int("F", f),
        TemplateArg::Int("K", top_k),
        TemplateArg::Bool("FAST", fast),
        TemplateArg::Int("BR", br),
        TemplateArg::Int("VPT", s),
        TemplateArg::Int("KSG", ksg),
        TemplateArg::Int("RPS", rps),
    ];
    let outs = [OutputArg { shape: vec![s, h], dtype: Dtype::Bfloat16 }];
    kernel
        .apply(
            &inputs,
            &template,
            (32, (h / rps) * ksg, s),
            (32, ksg, 1),
            &outs,
            stream,
        )
        .ok()?
        .into_iter()
        .next()
}

// --- mixer: track_mixer_down_inject + track_mixer_up_mix (S <= 8) -----------

const DOWN_INJECT_SOURCE: &str = include_str!("kernels/down_inject.metal");
const DOWN_INJECT_1_SOURCE: &str = include_str!("kernels/down_inject_1.metal");

const UP_MIX_SOURCE: &str = include_str!("kernels/up_mix.metal");
const UP_MIX_1_SOURCE: &str = include_str!("kernels/up_mix_1.metal");

const BF16_SIGMOID_SOURCE: &str = include_str!("kernels/bf16_sigmoid.metal");

static BF16_SIGMOID: std::sync::OnceLock<Option<Array>> = std::sync::OnceLock::new();

/// OPT-SIGLUT: `mlx_sigmoid` over every bf16 bit pattern, built once with
/// the same in-kernel function, so the mixer's bf16 sigmoid is a table lookup
/// and stays bit-identical.
pub fn bf16_sigmoid_table(stream: &Stream) -> Option<&'static Array> {
    BF16_SIGMOID
        .get_or_init(|| {
            let kernel = MetalKernel::new(
                "track_bf16_sigmoid_table",
                &[],
                &["table"],
                BF16_SIGMOID_SOURCE,
                crate::kernels::EXACT_HEADER,
                true,
                false,
            )
            .ok()?;
            let outs = [OutputArg { shape: vec![65536], dtype: Dtype::Bfloat16 }];
            let table = kernel
                .apply(&[], &[], (65536, 1, 1), (256, 1, 1), &outs, stream)
                .ok()?
                .into_iter()
                .next()?;
            let _ = table.eval();
            Some(table)
        })
        .as_ref()
}

static DOWN_INJECT: std::sync::OnceLock<Option<MetalKernel>> = std::sync::OnceLock::new();
static DOWN_INJECT_1: std::sync::OnceLock<Option<MetalKernel>> = std::sync::OnceLock::new();
static UP_MIX: std::sync::OnceLock<Option<MetalKernel>> = std::sync::OnceLock::new();
static UP_MIX_1: std::sync::OnceLock<Option<MetalKernel>> = std::sync::OnceLock::new();

fn mixer_header(wide: bool) -> String {
    compose(&[
        H::HELPERS_CORE,
        crate::kernels::EXACT_HEADER,
        H::REG_HELPERS,
        H::MIXER_HEAD_TAIL,
        if wide { H::WIDE_HELPERS } else { H::WIDE_DECLS },
    ])
}

fn down_inject_kernel(single: bool) -> &'static Option<MetalKernel> {
    let cell = if single { &DOWN_INJECT_1 } else { &DOWN_INJECT };
    cell.get_or_init(|| {
        let header = mixer_header(!single);
        MetalKernel::new(
            if single { "track_mixer_down_inject_1" } else { "track_mixer_down_inject" },
            &["normed", "wd", "sd", "bd", "wi", "si", "bi"],
            &["lo", "act", "inj"],
            if single { DOWN_INJECT_1_SOURCE } else { DOWN_INJECT_SOURCE },
            &header,
            true,
            false,
        )
        .ok()
    })
}

fn up_mix_kernel(single: bool) -> &'static Option<MetalKernel> {
    let cell = if single { &UP_MIX_1 } else { &UP_MIX };
    cell.get_or_init(|| {
        let header = mixer_header(!single);
        MetalKernel::new(
            if single { "track_mixer_up_mix_1" } else { "track_mixer_up_mix" },
            &["act", "normed", "wu", "su", "bu", "inj", "sigmoid_lut"],
            &["input", "inject", "inputF"],
            if single { UP_MIX_1_SOURCE } else { UP_MIX_SOURCE },
            &header,
            true,
            false,
        )
        .ok()
    })
}

/// `(lo, act, inj)` from `normed [S, KD]` and the down/inject quantized weights.
#[allow(clippy::too_many_arguments)]
pub fn down_inject(
    normed: &Array,
    wd: &Array, sd: &Array, bd: &Array,
    wi: &Array, si: &Array, bi: &Array,
    kd: i32, nd: i32, hc: i32, s: i32, has_inject: bool,
    stream: &Stream,
) -> Option<(Array, Array, Array)> {
    // OPT-SPLITK: one-row windows split the K walk across SIMD groups.
    if s == 1 && MIXER_SPLIT_K > 0 {
        if let Some(o) = split_k_mixer(normed, wd, sd, bd, wi, si, bi, nd, hc, has_inject, MIXER_SPLIT_K, stream) {
            return Some(o);
        }
    }
    let kernel = down_inject_kernel(s == 1).as_ref()?;
    let rps = if s == 1 { 1 } else { 4 };
    let mut tiles = nd / (2 * rps);
    if has_inject {
        tiles += if s == 1 { 2 } else { 1 };
    }
    let inputs: [&Array; 7] = [normed, wd, sd, bd, wi, si, bi];
    let template = [
        TemplateArg::Dtype("T", Dtype::Bfloat16),
        TemplateArg::Int("GS", 32),
        TemplateArg::Int("BITS", 4),
        TemplateArg::Int("KD", kd),
        TemplateArg::Int("ND", nd),
        TemplateArg::Int("HC", hc),
        TemplateArg::Int("VPT", s),
        TemplateArg::Bool("HAS_INJECT", has_inject),
    ];
    let outs = [
        OutputArg { shape: vec![s, nd], dtype: Dtype::Bfloat16 },
        OutputArg { shape: vec![s, nd], dtype: Dtype::Bfloat16 },
        OutputArg { shape: vec![s, hc], dtype: Dtype::Bfloat16 },
    ];
    let r = kernel
        .apply(&inputs, &template, (32, tiles * 2, 1), (32, 2, 1), &outs, stream)
        .ok()?;
    let mut it = r.into_iter();
    Some((it.next()?, it.next()?, it.next()?))
}

/// `(input [S,H], inject [S,HC])` from `act [S,LW]`, `normed [S,HC*H]`, up weight.
#[allow(clippy::too_many_arguments)]
pub fn up_mix(
    act: &Array, normed: &Array,
    wu: &Array, su: &Array, bu: &Array, inj: &Array,
    hidden: i32, hc: i32, s: i32, has_inject: bool,
    packed_rows: bool,
    stream: &Stream,
) -> Option<(Array, Array)> {
    let kernel = up_mix_kernel(s == 1).as_ref()?;
    let lw = act.dim(-1);
    let lut = if act.dtype() == Dtype::Bfloat16 { bf16_sigmoid_table(stream)? } else { normed };
    let inputs: [&Array; 7] = [act, normed, wu, su, bu, inj, lut];
    let template = [
        TemplateArg::Dtype("T", Dtype::Bfloat16),
        TemplateArg::Int("GS", 32),
        TemplateArg::Int("BITS", 4),
        TemplateArg::Int("H", hidden),
        TemplateArg::Int("HC", hc),
        TemplateArg::Int("LW", lw),
        TemplateArg::Int("VPT", s),
        TemplateArg::Bool("HAS_INJECT", has_inject),
        TemplateArg::Bool("EMIT_F32", false),
        TemplateArg::Bool("PACKED_ROWS", packed_rows),
    ];
    let outs = [
        OutputArg { shape: vec![s, hidden], dtype: Dtype::Bfloat16 },
        OutputArg { shape: vec![s, hc], dtype: Dtype::Bfloat16 },
        OutputArg { shape: vec![1, 1], dtype: Dtype::Float32 },
    ];
    let r = kernel
        .apply(&inputs, &template, (32, (hidden / 2) * 2, 1), (32, 2, 1), &outs, stream)
        .ok()?;
    let mut it = r.into_iter();
    Some((it.next()?, it.next()?))
}

// --- track_split_k_mixer: S=1 mixer down+inject split across SIMD groups ----

const SPLIT_K_HELPER: &str = include_str!("kernels/split_k_mixer_helper.metal");

const SPLIT_K_SOURCE: &str = include_str!("kernels/split_k_mixer.metal");

static SPLIT_K: std::sync::OnceLock<Option<MetalKernel>> = std::sync::OnceLock::new();

/// Partitions of the K walk in the S=1 mixer split-K (0 selects the original).
/// OPT-SPLITK bumped this 4 -> 5 (engine 2026-09-23); the fold is ordered
/// (ascending block), so it stays bit-identical.
pub const MIXER_SPLIT_K: i32 = 5;

fn split_k_kernel() -> &'static Option<MetalKernel> {
    SPLIT_K.get_or_init(|| {
        let header = compose(&[H::HELPERS_CORE, crate::kernels::EXACT_HEADER, SPLIT_K_HELPER]);
        MetalKernel::new(
            "track_split_k_mixer",
            &["x", "wd", "sd", "bd", "wi", "si", "bi"],
            &["lo", "act", "inj"],
            SPLIT_K_SOURCE,
            &header,
            true,
            false,
        )
        .ok()
    })
}

/// Split-K down+inject for the S=1 mixer: same kernels, K partitioned across
/// SIMD groups with an ORDERED (ascending-block) fold, so it is bit-identical.
#[allow(clippy::too_many_arguments)]
pub fn split_k_mixer(
    x: &Array,
    wd: &Array, sd: &Array, bd: &Array,
    wi: &Array, si: &Array, bi: &Array,
    n: i32, hc: i32, has_inject: bool, partitions: i32,
    stream: &Stream,
) -> Option<(Array, Array, Array)> {
    let kernel = split_k_kernel().as_ref()?;
    let k = x.dim(-1);
    let rows = 2;
    let inputs: [&Array; 7] = [x, wd, sd, bd, wi, si, bi];
    let template = [
        TemplateArg::Dtype("T", Dtype::Bfloat16),
        TemplateArg::Int("K", k),
        TemplateArg::Int("ND", n),
        TemplateArg::Int("RPS", rows),
        TemplateArg::Int("SPLIT", partitions),
        TemplateArg::Bool("HAS_INJECT", has_inject),
    ];
    let outs = [
        OutputArg { shape: vec![1, n], dtype: Dtype::Bfloat16 },
        OutputArg { shape: vec![1, n], dtype: Dtype::Bfloat16 },
        OutputArg { shape: vec![1, hc], dtype: Dtype::Bfloat16 },
    ];
    let grid_y = (n / rows + if has_inject { hc } else { 0 }) * partitions;
    let r = kernel
        .apply(&inputs, &template, (32, grid_y, 1), (32, partitions, 1), &outs, stream)
        .ok()?;
    let mut it = r.into_iter();
    Some((it.next()?, it.next()?, it.next()?))
}

// --- qmv_wide check kernels (engine test helpers) --------------------------

/// Micro-benchmark for the S=1 decode MoE kernels (route / gate_up_act /
/// down_combine) at the real shapes. Repeated launches with a single terminal
/// eval, so it measures GPU time, not syncs.
pub fn decode_bench() {
    use std::time::Instant;
    let kd = 2560i32; // hidden
    let n = 640i32;   // expert intermediate
    let e = 512i32;
    let top_k = 10i32;
    let rows = 1i32;
    let br = rows * top_k;
    let mut seed = 11u64;
    let mut rnd = move || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((seed >> 40) as f32 / 8388608.0) - 1.0
    };
    let mut mk = |shape: &[i32]| -> (Array, Array, Array) {
        let num: i32 = shape.iter().product();
        let v: Vec<f32> = (0..num as usize).map(|_| rnd() * 0.1).collect();
        let w = Array::from_slice(&v, shape).as_dtype(Dtype::Bfloat16).unwrap();
        lisa_mlx::ops::quantize(&w, 32, 4).unwrap()
    };
    let (gw, gs, gb) = mk(&[e, n, kd]);
    let (uw, us, ub) = mk(&[e, n, kd]);
    let (dw, ds, db) = mk(&[e, kd, n]);
    let (fsw, fss, fsb) = mk(&[2 * n, kd]);
    let (sddw, sdds, sddb) = mk(&[kd, n]);
    let x = Array::from_slice(&vec![0.1f32; kd as usize], &[rows, kd]).as_dtype(Dtype::Bfloat16).unwrap();
    let logits = Array::from_slice(&vec![0.05f32; (rows * e) as usize], &[rows, e]);
    let (sgw, sgs, sgb) = mk(&[1, kd]);
    let _ = (gw.eval(), gs.eval(), gb.eval(), x.eval(), logits.eval());

    let stream = lisa_mlx::Stream::thread_local_or_default();

    // route
    let t = Instant::now();
    let mut idx = None;
    for _ in 0..100 {
        idx = Some(route(&logits, &x, &sgw, &sgs, &sgb, top_k, e, kd, rows, &stream).unwrap());
    }
    let (i0, w, gate) = idx.unwrap();
    let _ = i0.eval();
    let rt = t.elapsed().as_secs_f64() * 10.0;

    let idx_flat = i0.reshape(&[br]).unwrap();
    let xrow = Array::from_slice(&(0..br).map(|i| (i / top_k) as u32).collect::<Vec<u32>>(), &[br]);

    // gate_up_act
    let t = Instant::now();
    let mut act = None;
    for _ in 0..100 {
        act = Some(
            gate_up_act(&gw, &gs, &gb, &uw, &us, &ub, &fsw, &fss, &fsb, &x, &idx_flat, &xrow, br, n, kd, rows, &stream)
                .unwrap(),
        );
    }
    let act = act.unwrap();
    let _ = act.eval();
    let gt = t.elapsed().as_secs_f64() * 10.0;

    // down_combine
    let t = Instant::now();
    let mut out = None;
    for _ in 0..100 {
        out = Some(
            down_combine(&dw, &ds, &db, &sddw, &sdds, &sddb, &act, &idx_flat, &w.reshape(&[br]).unwrap(), &gate.reshape(&[rows]).unwrap(), top_k, n, kd, br, rows, &stream)
                .unwrap(),
        );
    }
    let o = out.unwrap();
    let _ = o.eval();
    let dt = t.elapsed().as_secs_f64() * 10.0;
    println!("  (build+eval, 100 iters x 10) route {rt:.3} gate_up {gt:.3} down {dt:.3} ms/iter");
    wide_moe_bench();
}

/// Throwaway µbench for the S=7 wide MoE shapes: reports per-kernel ms and the
/// effective gathered-weight bandwidth, so the occupancy/bandwidth ceiling is
/// visible without running the model.
pub fn wide_moe_bench() {
    use std::time::Instant;
    let kd = 2560i32;
    let n = 640i32;
    let e = 512i32;
    let top_k = 10i32;
    let rows = 7i32;
    let br = rows * top_k;
    let mut seed = 11u64;
    let mut rnd = move || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((seed >> 40) as f32 / 8388608.0) - 1.0
    };
    let mut mk = |shape: &[i32]| -> (Array, Array, Array) {
        let num: i32 = shape.iter().product();
        let v: Vec<f32> = (0..num as usize).map(|_| rnd() * 0.1).collect();
        let w = Array::from_slice(&v, shape).as_dtype(Dtype::Bfloat16).unwrap();
        lisa_mlx::ops::quantize(&w, 32, 4).unwrap()
    };
    let (gw, gs, gb) = mk(&[e, n, kd]);
    let (uw, us, ub) = mk(&[e, n, kd]);
    let (dw, ds, db) = mk(&[e, kd, n]);
    let (fsw, fss, fsb) = mk(&[2 * n, kd]);
    let (sddw, sdds, sddb) = mk(&[kd, n]);
    let x = Array::from_slice(&vec![0.1f32; (rows * kd) as usize], &[rows, kd]).as_dtype(Dtype::Bfloat16).unwrap();
    let logits = Array::from_slice(&vec![0.05f32; (rows * e) as usize], &[rows, e]);
    let (sgw, sgs, sgb) = mk(&[1, kd]);
    let _ = (gw.eval(), x.eval(), logits.eval());
    let stream = lisa_mlx::Stream::thread_local_or_default();

    let t = Instant::now();
    let mut idx = None;
    for _ in 0..100 {
        idx = Some(route(&logits, &x, &sgw, &sgs, &sgb, top_k, e, kd, rows, &stream).unwrap());
    }
    let (i0, w, gate) = idx.unwrap();
    let _ = i0.eval();
    let rt = t.elapsed().as_secs_f64() * 10.0;
    let idx_flat = i0.reshape(&[br]).unwrap();
    let xrow = Array::from_slice(&(0..br).map(|i| (i / top_k) as u32).collect::<Vec<u32>>(), &[br]);

    let t = Instant::now();
    let mut act = None;
    for _ in 0..100 {
        act = Some(gate_up_act_wide(&gw, &gs, &gb, &uw, &us, &ub, &fsw, &fss, &fsb, &x, &idx_flat, &xrow, br, n, kd, rows, &stream).unwrap());
    }
    let act = act.unwrap();
    let _ = act.eval();
    let gt = t.elapsed().as_secs_f64() * 10.0;
    let gu_bytes = (br as f64) * 2.0 * (n as f64) * (kd as f64) / 2.0;

    let t = Instant::now();
    let mut out = None;
    for _ in 0..100 {
        out = Some(down_combine_wide(&dw, &ds, &db, &sddw, &sdds, &sddb, &act, &idx_flat, &w.reshape(&[br]).unwrap(), &gate.reshape(&[rows]).unwrap(), top_k, n, kd, br, rows, &stream).unwrap());
    }
    let o = out.unwrap();
    let _ = o.eval();
    let dt = t.elapsed().as_secs_f64() * 10.0;
    let d_bytes = (br as f64) * (kd as f64) * (n as f64) / 2.0;
    println!("  wide S=7 (br={br}): route {rt:.3} gate_up {gt:.3} ({:.0} GB/s) down {dt:.3} ({:.0} GB/s) ms/iter",
        gu_bytes / (gt * 1e-3) / 1e9, d_bytes / (dt * 1e-3) / 1e9);

    // Verify-width projections at M=6 (the d5 verify width): attention q/k/v/o,
    // GDN fused, and lm_head. The reported bandwidth is on the weight bytes.
    let rows = 6i32;
    let x7 = Array::from_slice(&vec![0.1f32; (rows * kd) as usize], &[rows, kd]).as_dtype(Dtype::Bfloat16).unwrap();
    let _ = x7.eval();
    for (nm, nrows) in [
        ("q_proj", 12288i32), ("k_proj", 512i32), ("v_proj", 512i32), ("o_proj", 2560i32),
        ("gdn_fused", 16480i32), ("lm_head", 248320i32),
    ] {
        let (w, sc, bi) = mk(&[nrows, kd]);
        let _ = w.eval();
        let bytes = (nrows as f64) * (kd as f64) / 2.0;
        let time = |label: &str, f: &dyn Fn() -> Array| {
            let t = Instant::now();
            for _ in 0..100 { let _ = f().eval(); }
            let ms = t.elapsed().as_secs_f64() * 10.0;
            println!("    {label}: {ms:.3} ms ({:.0} GB/s)", bytes / (ms * 1e-3) / 1e9);
        };
        time("qmm      ", &|| lisa_mlx::ops::quantized_matmul(&x7, &w, &sc, Some(&bi), true, 32, 4).unwrap());
        time("qmv_wide ", &|| Array::new(crate::mlx_rt::qmv_wide(x7.t.device(), &x7.t, &w.t, &sc.t, &bi.t, 32, 4).unwrap()));
        let a = Array::new(crate::mlx_rt::qmv_wide(x7.t.device(), &x7.t, &w.t, &sc.t, &bi.t, 32, 4).unwrap());
        let b = lisa_mlx::ops::quantized_matmul(&x7, &w, &sc, Some(&bi), true, 32, 4).unwrap();
        let _ = (a.eval(), b.eval());
        let diff = a.t.subtract(&b.t).unwrap().abs().unwrap().to_dtype(lisa_mlx::Dtype::Float32).unwrap().max_all().unwrap().to_scalar::<f32>().unwrap();
        println!("  proj {nm} N={nrows}: qmm-vs-qmv_wide maxdiff {diff:.4}");
    }
}
