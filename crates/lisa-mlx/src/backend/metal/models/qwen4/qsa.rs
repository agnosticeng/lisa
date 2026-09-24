//! QSA block-sparse attention kernels.
//!
//! The block-sparse attention and fused selector Metal sources are adapted
//! from a third-party implementation (Apache-2.0; see `NOTICE` for the
//! attribution). This module adapts only the host dispatch to lisa's
//! `MetalKernel` API and the tower's tensor layout.
//!
//! The kernel reads the full K/V backing in place, streams only the selected
//! four-token blocks (plus the query row's visible causal tail) per
//! `(query row, kv head)` simdgroup, and never builds a dense `[S, L]` mask.

use std::sync::OnceLock;

use lisa_mlx::ops::indexing::IndexOp;
use lisa_mlx::{ops, Array, Stream};

use crate::ffi::{MetalKernel, OutputArg, TemplateArg};

const HEADER: &str = include_str!("kernels/qsa/prefill_header.metal");
const SOURCE: &str = include_str!("kernels/qsa/prefill.metal");

const SELECT_HEADER: &str = include_str!("kernels/qsa/select_header.metal");
const SELECT_SOURCE: &str = include_str!("kernels/qsa/select.metal");

static SELECT: OnceLock<Option<MetalKernel>> = OnceLock::new();

/// Geometry baked into `qsa_select.metal` (generated for this model): 4 indexer
/// heads, head dim 128, top-k 512 blocks, compress ratio 4, width 512, and a
/// `BACKING_BLOCKS` of 256 K / 4 (the model's full context in blocks).
const SELECT_BACKING: i32 = 262144 / 4;
const SELECT_WIDTH: i32 = 512;

fn select_kernel() -> &'static Option<MetalKernel> {
    SELECT.get_or_init(|| {
        match MetalKernel::new(
            "qsa_select_blocks_h4_d128_n65536_k512_r4_w512_t0_tf1_bf16_bf16",
            &["q", "pooled", "pos_start", "total_tokens", "logical_blocks"],
            &["block_ids", "block_valid", "adjusted_scores", "score_scratch"],
            SELECT_SOURCE,
            SELECT_HEADER,
            false,
            false,
        ) {
            Ok(k) => Some(k),
            Err(e) => {
                eprintln!("[qsa] indexer-select kernel unavailable: {e}");
                None
            }
        }
    })
}

/// Fused QSA block selection (mode `blocks`; see `NOTICE`).
///
/// `q` is `[1, S, 4, 128]` and `pooled` `[1, blocks, 128]`, both post-norm and
/// post-RoPE. Returns `(block_ids [S, 512] int32, block_valid [S, 512] bool)`,
/// or `None` when the kernel is unavailable.
pub fn select_blocks(
    q: &Array,
    pooled: &Array,
    pos_start: i32,
    total_tokens: i32,
    blocks: usize,
    stream: &Stream,
) -> Option<(Array, Array)> {
    let kernel = select_kernel().as_ref()?;
    let rows = q.dim(1);
    if q.dim(0) != 1 || q.dim(2) != 4 || q.dim(3) != 128 {
        return None;
    }
    if pooled.dim(0) != 1 || pooled.dim(2) != 128 {
        return None;
    }
    let logical = blocks as i32;
    if logical != total_tokens / 4 || logical > SELECT_BACKING {
        return None;
    }
    // Score tiling: the selector stages a
    // `[rows, BACKING]` fp32 score scratch; at 262K that is ~2.1 GB per layer
    // per 2048-chunk and the dominant prefill transient. Each query row's
    // selection is independent, so process rows in tiles and concatenate.
    const TILE: i32 = 512;
    let mut ids_out: Vec<Array> = Vec::new();
    let mut val_out: Vec<Array> = Vec::new();
    let mut q0 = 0i32;
    while q0 < rows {
        let qt = TILE.min(rows - q0);
        let q_t = q.index((.., q0..(q0 + qt), .., ..));
        let ps = Array::from_slice(&[pos_start + q0], &[1]);
        let tt = Array::from_slice(&[total_tokens], &[1]);
        let lb = Array::from_slice(&[logical], &[1]);
        let inputs: [&Array; 5] = [&q_t, pooled, &ps, &tt, &lb];
        let outputs = [
            OutputArg { shape: vec![qt, 512], dtype: lisa_mlx::Dtype::Int32 },
            OutputArg { shape: vec![qt, 512], dtype: lisa_mlx::Dtype::Bool },
            OutputArg { shape: vec![qt, 512], dtype: lisa_mlx::Dtype::Float32 },
            OutputArg { shape: vec![qt, SELECT_BACKING], dtype: lisa_mlx::Dtype::Float32 },
        ];
        match kernel.apply(
            &inputs,
            &[],
            (qt * SELECT_WIDTH, 1, 1),
            (SELECT_WIDTH, 1, 1),
            &outputs,
            stream,
        ) {
            Ok(mut outs) => {
                let _ = outs.pop();
                let _ = outs.pop();
                val_out.push(outs.pop()?);
                ids_out.push(outs.pop()?);
            }
            Err(e) => {
                static ONCE: std::sync::Once = std::sync::Once::new();
                ONCE.call_once(|| eprintln!("[qsa] indexer-select apply failed (falling back): {e}"));
                return None;
            }
        }
        q0 += qt;
    }
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| eprintln!("[qsa] fused indexer-select engaged"));
    let ids = if ids_out.len() == 1 {
        ids_out.pop().unwrap()
    } else {
        let refs: Vec<&Array> = ids_out.iter().collect();
        ops::concatenate(&refs, 0).ok()?
    };
    let valid = if val_out.len() == 1 {
        val_out.pop().unwrap()
    } else {
        let refs: Vec<&Array> = val_out.iter().collect();
        ops::concatenate(&refs, 0).ok()?
    };
    Some((ids, valid))
}


static KERNEL: OnceLock<Option<MetalKernel>> = OnceLock::new();

fn kernel() -> &'static Option<MetalKernel> {
    KERNEL.get_or_init(|| {
        match MetalKernel::new(
            "qsa_prefill_flash_nax_m16_n32_h24_kv2_d256_b4_k512",
            &["q", "k", "v", "block_ids", "block_valid", "params", "scale"],
            &["out"],
            SOURCE,
            HEADER,
            false,
            false,
        ) {
            Ok(k) => Some(k),
            Err(e) => {
                eprintln!("[qsa] block-sparse kernel unavailable: {e}");
                None
            }
        }
    })
}

/// Block-sparse QSA prefill attention.
///
/// `q` is `[1, 24, S, 256]` (post-norm, post-RoPE), `k`/`v` are the full cache
/// backings `[1, 2, capacity, 256]`, and `block_ids`/`block_valid` are
/// `[S, 512]`. `pos_start` is the absolute position of query row zero and
/// `total_tokens` the logical KV length (`pos_start + S == total_tokens`).
/// Returns `[1, 24, S, 256]`, or `None` when the kernel is unavailable or the
/// geometry does not meet the production contract (caller falls back).
#[allow(clippy::too_many_arguments)]
pub fn prefill_flash(
    q: &Array,
    k: &Array,
    v: &Array,
    block_ids: &Array,
    block_valid: &Array,
    pos_start: i32,
    total_tokens: i32,
    scale: f32,
    stream: &Stream,
) -> Option<Array> {
    let kernel = match kernel().as_ref() {
        Some(k) => k,
        None => return None,
    };
    let reject = |why: &str| -> Option<Array> {
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| {
            eprintln!(
                "[qsa] geometry reject ({why}): q{:?} k{:?} kvd{:?} bids{:?} pos_start={pos_start} total={total_tokens}",
                q.shape(), k.shape(), v.shape(), block_ids.shape()
            )
        });
        None
    };
    if q.dim(0) != 1 || q.dim(1) != 24 || q.dim(3) != 256 || q.dim(2) < 1 {
        return reject("q");
    }
    if k.dim(0) != 1 || k.dim(1) != 2 || k.dim(3) != 256 {
        return reject("k");
    }
    if block_ids.dim(0) != q.dim(2) || block_ids.dim(1) != 512 {
        return reject("bids");
    }
    if total_tokens <= 0 || pos_start + q.dim(2) != total_tokens || total_tokens > k.dim(2) {
        return reject("span");
    }
    let rows = q.dim(2);
    let params = Array::from_slice(&[pos_start, total_tokens, rows], &[3]);
    let scale_arr = Array::from_slice(&[scale], &[1]);
    let inputs: [&Array; 7] = [q, k, v, block_ids, block_valid, &params, &scale_arr];
    let template = [TemplateArg::Dtype("T", q.dtype())];
    let outputs = [OutputArg {
        shape: vec![1, 24, rows, 256],
        dtype: q.dtype(),
    }];
    const THREADS: i32 = 32;
    match kernel.apply(
        &inputs,
        &template,
        (rows * 2 * THREADS, 1, 1),
        (THREADS, 1, 1),
        &outputs,
        stream,
    ) {
        Ok(outs) => {
            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| eprintln!("[qsa] block-sparse attention engaged"));
            outs.into_iter().next()
        }
        Err(e) => {
            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| eprintln!("[qsa] block-sparse apply failed (falling back): {e}"));
            None
        }
    }
}
