//! The engine's NAX prefill indirect GEMM (`TrackPrefillIndirect.swift`):
//! a tile table over the sorted expert ids, then `track_prefill_indirect_gu`
//! (gate|up + SwiGLU) and `track_prefill_indirect` (down), both built on MLX's
//! `steel` header (`src/nax_helpers.rs`).
//!
//! Every expert run of `r` rows takes `ceil(r / 32)` tiles aligned to the run
//! start, so each threadgroup reads one expert's weights once.

use lisa_mlx::{Array, Dtype, Stream};

use crate::ffi::{MetalKernel, OutputArg, TemplateArg};

pub const TILE_ROWS: i32 = 32;
pub const TILE_THREADS: i32 = 1024;

pub fn max_tiles(rows: i32, experts: i32) -> i32 {
    (rows + TILE_ROWS - 1) / TILE_ROWS + experts
}

const TILE_SOURCE: &str = include_str!("kernels/prefill_tile_table.metal");

const SOURCE_GU: &str = include_str!("kernels/prefill_gate_up.metal");

const DOWN_BLOCK_N: i32 = 128;

const SOURCE_DOWN: &str = include_str!("kernels/prefill_down.metal");

static TILE_TABLE: std::sync::OnceLock<Option<MetalKernel>> = std::sync::OnceLock::new();
static GATE_UP: std::sync::OnceLock<Option<MetalKernel>> = std::sync::OnceLock::new();
static DOWN: std::sync::OnceLock<Option<MetalKernel>> = std::sync::OnceLock::new();

fn tile_table_kernel() -> &'static Option<MetalKernel> {
    TILE_TABLE.get_or_init(|| {
        MetalKernel::new(
            "track_prefill_tile_table",
            &["sorted_ids"],
            &["tiles"],
            TILE_SOURCE,
            "",
            true,
            false,
        )
        .ok()
    })
}
fn gate_up_kernel() -> &'static Option<MetalKernel> {
    GATE_UP.get_or_init(|| {
        MetalKernel::new(
            "track_prefill_indirect_gate_up",
            &["x", "w0", "scales0", "biases0", "w1", "scales1", "biases1", "indices", "token_rows", "tiles"],
            &["y"],
            SOURCE_GU,
            crate::nax_helpers::NAX_HEADER,
            true,
            false,
        )
        .ok()
    })
}
fn down_kernel() -> &'static Option<MetalKernel> {
    DOWN.get_or_init(|| {
        MetalKernel::new(
            "track_prefill_indirect_down",
            &["x", "w", "scales", "biases", "indices", "token_rows", "tiles"],
            &["y"],
            SOURCE_DOWN,
            crate::nax_helpers::NAX_HEADER,
            true,
            false,
        )
        .ok()
    })
}

/// `[2 * maxT]` uint32 row ranges, one 32-row tile per slot.
pub fn tile_table(sorted_ids: &Array, rows: i32, experts: i32, stream: &Stream) -> Option<Array> {
    let kernel = tile_table_kernel().as_ref()?;
    let max_t = max_tiles(rows, experts);
    let inputs: [&Array; 1] = [sorted_ids];
    let template = [
        TemplateArg::Int("R", rows),
        TemplateArg::Int("E", experts),
        TemplateArg::Int("BM", TILE_ROWS),
        TemplateArg::Int("MAXT", max_t),
        TemplateArg::Int("TG", TILE_THREADS),
    ];
    let outs = [OutputArg { shape: vec![2 * max_t], dtype: Dtype::Uint32 }];
    kernel
        .apply(
            &inputs,
            &template,
            (TILE_THREADS, 1, 1),
            (TILE_THREADS, 1, 1),
            &outs,
            stream,
        )
        .ok()?
        .into_iter()
        .next()
}

/// `gate|up + SwiGLU` over the tile table -> `[rows, 1, 640]` bf16.
#[allow(clippy::too_many_arguments)]
pub fn gate_up(
    x: &Array,
    w0: &Array, s0: &Array, b0: &Array,
    w1: &Array, s1: &Array, b1: &Array,
    sorted_ids: &Array, token_rows: &Array, tiles: &Array,
    max_t: i32, n: i32, k: i32, rows: i32,
    stream: &Stream,
) -> Option<Array> {
    let kernel = gate_up_kernel().as_ref()?;
    let inputs: [&Array; 10] = [x, w0, s0, b0, w1, s1, b1, sorted_ids, token_rows, tiles];
    let template = [
        TemplateArg::Dtype("T", Dtype::Bfloat16),
        TemplateArg::Int("N", n),
        TemplateArg::Int("K", k),
        TemplateArg::Bool("SILU", true),
    ];
    let outs = [OutputArg { shape: vec![rows, 1, n], dtype: Dtype::Bfloat16 }];
    kernel
        .apply(
            &inputs,
            &template,
            ((n / 64) * 32, max_t * 2, 2),
            (32, 2, 2),
            &outs,
            stream,
        )
        .ok()?
        .into_iter()
        .next()
}

/// The down projection over the same tile table -> `[rows, 1, 2560]` bf16.
#[allow(clippy::too_many_arguments)]
pub fn down(
    activated: &Array,
    w: &Array, s: &Array, b: &Array,
    sorted_ids: &Array, tiles: &Array,
    max_t: i32, n: i32, k: i32, rows: i32,
    stream: &Stream,
) -> Option<Array> {
    let kernel = down_kernel().as_ref()?;
    // OPT-IDENTITYROWS: the down tile's token rows are `0..rows`, so the
    // kernel takes the identity branch and `token_rows` is never read.
    let inputs: [&Array; 7] = [activated, w, s, b, sorted_ids, sorted_ids, tiles];
    let template = [
        TemplateArg::Dtype("T", Dtype::Bfloat16),
        TemplateArg::Int("N", n),
        TemplateArg::Int("K", k),
    ];
    let outs = [OutputArg { shape: vec![rows, 1, n], dtype: Dtype::Bfloat16 }];
    kernel
        .apply(
            &inputs,
            &template,
            ((n / DOWN_BLOCK_N) * 32, max_t * 2, 2),
            (32, 2, 2),
            &outs,
            stream,
        )
        .ok()?
        .into_iter()
        .next()
}

// --- NAX prefill router: track_router_bf16_storage + track_router_split_sum ---

const ROUTER_PARTIAL_SOURCE: &str = include_str!("kernels/router_partial.metal");

const ROUTER_SUM_SOURCE: &str = include_str!("kernels/router_sum.metal");

const ROUTER_LOOP_HEADER: &str = include_str!("kernels/router_loop_header.metal");


static ROUTER_PARTIAL: std::sync::OnceLock<Option<MetalKernel>> = std::sync::OnceLock::new();
static ROUTER_SUM: std::sync::OnceLock<Option<MetalKernel>> = std::sync::OnceLock::new();

fn router_partial_kernel() -> &'static Option<MetalKernel> {
    ROUTER_PARTIAL.get_or_init(|| {
        let header = format!("{}{}", crate::nax_helpers::NAX_HEADER, ROUTER_LOOP_HEADER);
        MetalKernel::new(
            "track_router_bf16_storage",
            &["x", "w"],
            &["partials"],
            ROUTER_PARTIAL_SOURCE,
            &header,
            true,
            false,
        )
        .ok()
    })
}
fn router_sum_kernel() -> &'static Option<MetalKernel> {
    ROUTER_SUM.get_or_init(|| {
        MetalKernel::new(
            "track_router_split_sum",
            &["partials"],
            &["y"],
            ROUTER_SUM_SOURCE,
            "",
            true,
            false,
        )
        .ok()
    })
}

/// The NAX prefill router: `x` bf16 `[1, rows, 2560]`, `w` bf16 `[512, 2560]`
/// -> logits f32 `[1, rows, 512]`. `rows` in 32..=1024.
pub fn router(x: &Array, w: &Array, rows: i32, stream: &Stream) -> Option<Array> {
    let partial = router_partial_kernel().as_ref()?;
    let sum = router_sum_kernel().as_ref()?;
    let tiles_m = (rows + 63) / 64;
    let swizzle: i32 = if tiles_m <= 3 { 1 } else { 2 };
    let groups = 8 * swizzle * ((tiles_m + swizzle - 1) / swizzle) * 2;
    let tmpl = [TemplateArg::Int("M", rows)];
    let p_outs = [OutputArg { shape: vec![2, rows, 512], dtype: Dtype::Float32 }];
    let partials = partial
        .apply(&[x, w], &tmpl, (groups * 32, 2, 2), (32, 2, 2), &p_outs, stream)
        .ok()?
        .into_iter()
        .next()?;
    let s_outs = [OutputArg { shape: vec![1, rows, 512], dtype: Dtype::Float32 }];
    sum.apply(&[&partials], &tmpl, (512, rows, 1), (256, 1, 1), &s_outs, stream)
        .ok()?
        .into_iter()
        .next()
}
