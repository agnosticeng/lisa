use crate::ffi::{MetalKernel, OutputArg, TemplateArg};
use lisa_mlx::{Array, Dtype, Stream, ops};

use super::common::EXACT_HEADER;

/// `track_p12_moe_sorted_combine`: the routed combine reads the SORTED expert
/// rows through the inverse permutation the sort already produced, keeping the
/// float32 products in original (token, slot) order and the same small-column
/// reduction tree. Replaces the f32 [B,S,K,H] materialisation + scatter + the
/// 10-op column reduce with one launch.
///
/// `routed` is the sorted down output `[rows*K, H]`, `w` the float32 weights in
/// ORIGINAL slot order `[rows*K]`, `inverse_order` u32 `[rows*K]`, `shared`
/// `[rows, H]`, `gate` bf16 `[rows]` -> `[rows, H]`.
const MOE_COMBINE_SOURCE: &str = include_str!("../../../shaders/qwen4/moe_combine.metal");

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
        .apply(
            &inputs,
            &template,
            (h, rows, 1),
            (256, 1, 1),
            &outputs,
            stream,
        )
        .ok()?;
    out.into_iter().next()
}
/// `track_route_block_counts` + `track_route_counting_scatter`: the stable
/// counting sort that produces the routed-assignment permutation
/// `(sortedIDs, tokenRows, inverse)` in two launches, replacing MLX's multi-block
/// merge sort (7 launches per `argSort`, 14 per layer). MLX's argsort is stable
/// and a stable counting sort is the unique stable permutation, so the two agree
/// element for element.
const ROUTE_BLOCK_COUNTS_SOURCE: &str =
    include_str!("../../../shaders/qwen4/route_block_counts.metal");

const ROUTE_COUNTING_SCATTER_SOURCE: &str =
    include_str!("../../../shaders/qwen4/route_counting_scatter.metal");

static ROUTE_BLOCK_COUNTS: std::sync::OnceLock<Option<MetalKernel>> = std::sync::OnceLock::new();
static ROUTE_COUNTING_SCATTER: std::sync::OnceLock<Option<MetalKernel>> =
    std::sync::OnceLock::new();

fn route_block_counts_kernel() -> &'static Option<MetalKernel> {
    ROUTE_BLOCK_COUNTS.get_or_init(|| {
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
    })
}

fn route_counting_scatter_kernel() -> &'static Option<MetalKernel> {
    ROUTE_COUNTING_SCATTER.get_or_init(|| {
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
    })
}

/// Force `get_or_init` on every kernel static in this module. Compilation of
/// the Metal pipeline itself happens on first `apply` (template-keyed), so
/// this only materialises the kernel handles; see `super::super::warm_kernels`.
pub fn warm() {
    let _ = moe_combine_kernel();
    let _ = route_block_counts_kernel();
    let _ = route_counting_scatter_kernel();
    let _ = router_gemv_kernel();
}

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

    let count_kernel = route_block_counts_kernel();
    let counts = count_kernel
        .as_ref()?
        .apply(
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
        )
        .ok()?
        .into_iter()
        .next()?;

    let scatter_kernel = route_counting_scatter_kernel();
    let mut outs = scatter_kernel
        .as_ref()?
        .apply(
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
                OutputArg {
                    shape: vec![r as i32],
                    dtype: Dtype::Uint32,
                },
                OutputArg {
                    shape: vec![r as i32],
                    dtype: Dtype::Uint32,
                },
                OutputArg {
                    shape: vec![r as i32],
                    dtype: Dtype::Uint32,
                },
            ],
            stream,
        )
        .ok()?
        .into_iter();
    Some((outs.next()?, outs.next()?, outs.next()?))
}
/// `track_router_gemv`: the decode router as a bf16 GEMV with f32 accumulate,
/// one token (`x` f32 `[K]`, `w` bf16 `[N,K]`) -> logits f32 `[N]`.
const ROUTER_GEMV_SOURCE: &str = include_str!("../../../shaders/qwen4/router_gemv.metal");

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
    let outs = [OutputArg {
        shape: vec![n],
        dtype: Dtype::Float32,
    }];
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
mod counting_sort_check {
    #[allow(unused_imports)] // utilisé sous cfg(test) uniquement
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
                (seed >> 33) as u32
            };
            let ids: Vec<i32> = (0..r).map(|_| (rnd() % e as u32) as i32).collect();
            let arr = Array::from_slice(&ids, &[r as i32]);
            let (sorted, _token, inv) =
                super::route_counting_sort(&arr, e, 10, &stream).expect("sort");
            let _ = sorted.eval();
            let _ = inv.eval();
            let s = sorted.as_slice::<u32>().to_vec();
            let iv = inv.as_slice::<u32>().to_vec();
            assert!(
                s.windows(2).all(|w| w[0] <= w[1]),
                "sorted_ids not sorted at r={r}"
            );
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
            assert!(
                seen.iter().all(|&b| b),
                "inverse not a permutation at r={r}"
            );
        }
    }
}
