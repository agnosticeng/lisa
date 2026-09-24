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

#[cfg(test)]
mod selector_tests {
    use super::*;
    use lisa_mlx::ops::indexing::{Ellipsis, IndexOp};
    use lisa_mlx::Dtype;

    fn lcg(seed: &mut u64) -> f32 {
        *seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*seed >> 40) as f32 / (1u32 << 24) as f32) - 0.5
    }

    /// The fused selector must pick the same `(block_id, valid)` set as the
    /// reference op chain (`score = sum_h relu(q . pooled) / sqrt(128)`, the
    /// causal-complete mask, the `score - block_id*1e-12` tie-break, top-k).
    #[test]
    fn fused_selector_matches_reference() {
        let stream = Stream::thread_local_or_default();
        let (s, blocks, heads, dim) = (8usize, 600usize, 4usize, 128usize);
        let total = blocks * 4;
        let pos_start = total - s;
        let mut seed = 0xabcd_1234_5678_9ef0u64;
        let qv: Vec<half::bf16> = (0..s * heads * dim)
            .map(|_| half::bf16::from_f32(lcg(&mut seed)))
            .collect();
        let pv: Vec<half::bf16> = (0..blocks * dim)
            .map(|_| half::bf16::from_f32(lcg(&mut seed)))
            .collect();
        let q = Array::from_slice(&qv, &[1, s as i32, heads as i32, dim as i32]);
        let pooled = Array::from_slice(&pv, &[1, blocks as i32, dim as i32]);

        // Reference scoring, in f32 from the bf16 operands (what the indexer
        // does); the kernel truncates to tf32, a no-op for bf16.
        let qf = q.as_dtype(Dtype::Float32).unwrap();
        let pf = pooled.as_dtype(Dtype::Float32).unwrap();
        let flat = qf.reshape(&[1, (s * heads) as i32, dim as i32]).unwrap();
        let scores = ops::matmul(&flat, &pf.transpose_axes(&[0, 2, 1]).unwrap())
            .unwrap()
            .reshape(&[1, s as i32, heads as i32, blocks as i32])
            .unwrap()
            .transpose_axes(&[0, 1, 3, 2])
            .unwrap();
        let scores = ops::maximum(&scores, &Array::from_f32(0.0))
            .unwrap()
            .sum_axis(-1, None)
            .unwrap()
            .divide(&Array::from_f32((dim as f32).sqrt()))
            .unwrap();
        let complete_v: Vec<f32> = (0..s)
            .map(|j| (((pos_start + j + 1) / 4) as f32))
            .collect();
        let complete = Array::from_slice(&complete_v, &[1, s as i32]);
        let bids_v: Vec<f32> = (0..blocks).map(|i| i as f32).collect();
        let bids = Array::from_slice(&bids_v, &[1, 1, blocks as i32]);
        let visible = ops::broadcast_to(&bids, &[1, s as i32, blocks as i32])
            .unwrap()
            .lt(&complete.expand_dims(-1).unwrap())
            .unwrap();
        let neg_inf = ops::broadcast_to(&Array::from_f32(f32::NEG_INFINITY), scores.shape()).unwrap();
        let masked = ops::r#where(&visible, &scores, &neg_inf).unwrap();
        let key = masked
            .subtract(&bids.multiply(Array::from_f32(1e-12)).unwrap())
            .unwrap();
        let k = 512usize;
        let top = ops::argpartition_axis(&(-&key), (k - 1) as i32, -1)
            .unwrap()
            .index((Ellipsis, 0..k as i32))
            .contiguous()
            .unwrap();
        let ref_ids: Vec<i32> = top.as_dtype(Dtype::Int32).unwrap().as_slice::<i32>().to_vec();
        // Validity is the causal-complete mask at the selected block's query row
        // (avoid a bool gather, which has no kernel instantiation).
        let ref_valid: Vec<u8> = ref_ids
            .iter()
            .enumerate()
            .map(|(i, &id)| if (id as f32) < complete_v[i / k] { 1 } else { 0 })
            .collect();

        let (ids, valid) = select_blocks(&q, &pooled, pos_start as i32, total as i32, blocks, &stream)
            .expect("fused selector unavailable");
        let got_ids: Vec<i32> = ids.as_dtype(Dtype::Int32).unwrap().as_slice::<i32>().to_vec();
        let got_valid: Vec<u8> = valid.as_dtype(Dtype::Uint8).unwrap().as_slice::<u8>().to_vec();

        // The kernel's total order is `qsa_index_before`: selected blocks by
        // ascending id, then invalid lanes, applied per query row. Compare the
        // exact ordered arrays row by row.
        for r in 0..s {
            let lo = r * k;
            let hi = lo + k;
            let mut expected: Vec<(i32, u8)> = ref_ids[lo..hi]
                .iter()
                .copied()
                .zip(ref_valid[lo..hi].iter().copied())
                .collect();
            expected.sort_by(|x, y| y.1.cmp(&x.1).then(x.0.cmp(&y.0)));
            let got: Vec<(i32, u8)> = got_ids[lo..hi]
                .iter()
                .copied()
                .zip(got_valid[lo..hi].iter().copied())
                .collect();
            assert_eq!(got, expected, "fused selector row {r} order/content differs");
        }
    }
}

#[cfg(test)]
mod selector_bench {
    use super::*;
    use std::time::Instant;

    fn lcg(seed: &mut u64) -> f32 {
        *seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*seed >> 40) as f32 / (1u32 << 24) as f32) - 0.5
    }

    /// No-sync selector benchmark: enqueue `n` applies then one eval. Run with
    /// `cargo test -p lisa-mlx --lib selector_bench -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn bench_select() {
        let stream = Stream::thread_local_or_default();
        let (heads, dim) = (4usize, 128usize);
        for &(s, blocks) in &[(1usize, 1506usize), (1, 15000), (1, 60000), (2048, 60000)] {
            let total = blocks * 4;
            let pos_start = total.saturating_sub(s);
            let mut seed = 0x1234_5678u64;
            let qv: Vec<half::bf16> = (0..s * heads * dim)
                .map(|_| half::bf16::from_f32(lcg(&mut seed)))
                .collect();
            let pv: Vec<half::bf16> = (0..blocks * dim)
                .map(|_| half::bf16::from_f32(lcg(&mut seed)))
                .collect();
            let q = Array::from_slice(&qv, &[1, s as i32, heads as i32, dim as i32]);
            let pooled = Array::from_slice(&pv, &[1, blocks as i32, dim as i32]);
            // warmup (compile)
            let _ = select_blocks(&q, &pooled, pos_start as i32, total as i32, blocks, &stream);
            let n = 30usize;
            let t = Instant::now();
            let mut last = None;
            for _ in 0..n {
                last = select_blocks(&q, &pooled, pos_start as i32, total as i32, blocks, &stream);
            }
            if let Some((ids, _)) = last {
                let _ = ids.eval();
            }
            let ms = t.elapsed().as_secs_f64() * 1e3 / n as f64;
            eprintln!("[sel-bench] s={s} blocks={blocks} {ms:.3} ms/call");
        }
    }
}

#[cfg(test)]
mod score_equiv {
    use super::*;
    use lisa_mlx::ops::indexing::{Ellipsis, IndexOp};
    use lisa_mlx::Dtype;

    fn lcg(seed: &mut u64) -> f32 {
        *seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*seed >> 40) as f32 / (1u32 << 24) as f32) - 0.5
    }

    /// Reference top-K selection from a score sheet computed with MLX's matmul
    /// (the tensor-core/NAX route a score-sheet kernel would use).
    fn matmul_reference(
        q: &Array,
        pooled: &Array,
        s: usize,
        blocks: usize,
        pos_start: usize,
        heads: usize,
        dim: usize,
    ) -> Vec<(i32, u8)> {
        let qf = q.as_dtype(Dtype::Float32).unwrap();
        let pf = pooled.as_dtype(Dtype::Float32).unwrap();
        let flat = qf.reshape(&[1, (s * heads) as i32, dim as i32]).unwrap();
        let scores = ops::matmul(&flat, &pf.transpose_axes(&[0, 2, 1]).unwrap())
            .unwrap()
            .reshape(&[1, s as i32, heads as i32, blocks as i32])
            .unwrap()
            .transpose_axes(&[0, 1, 3, 2])
            .unwrap();
        let scores = ops::maximum(&scores, &Array::from_f32(0.0))
            .unwrap()
            .sum_axis(-1, None)
            .unwrap()
            .divide(&Array::from_f32((dim as f32).sqrt()))
            .unwrap();
        let complete_v: Vec<f32> = (0..s).map(|j| ((pos_start + j + 1) / 4) as f32).collect();
        let complete = Array::from_slice(&complete_v, &[1, s as i32]);
        let bids_v: Vec<f32> = (0..blocks).map(|i| i as f32).collect();
        let bids = Array::from_slice(&bids_v, &[1, 1, blocks as i32]);
        let visible = ops::broadcast_to(&bids, &[1, s as i32, blocks as i32])
            .unwrap()
            .lt(&complete.expand_dims(-1).unwrap())
            .unwrap();
        let neg_inf =
            ops::broadcast_to(&Array::from_f32(f32::NEG_INFINITY), scores.shape()).unwrap();
        let masked = ops::r#where(&visible, &scores, &neg_inf).unwrap();
        let key = masked
            .subtract(&bids.multiply(Array::from_f32(1e-12)).unwrap())
            .unwrap();
        let k = 512usize;
        let top = ops::argpartition_axis(&(-&key), (k - 1) as i32, -1)
            .unwrap()
            .index((Ellipsis, 0..k as i32))
            .contiguous()
            .unwrap();
        let ids: Vec<i32> = top.as_dtype(Dtype::Int32).unwrap().as_slice::<i32>().to_vec();
        let valid: Vec<u8> = ids
            .iter()
            .enumerate()
            .map(|(i, &id)| if (id as f32) < complete_v[i / k] { 1 } else { 0 })
            .collect();
        ids.into_iter().zip(valid).collect()
    }

    /// Quantify how often a matmul-computed score sheet changes the top-K block
    /// selection versus the scalar kernel. Zero flips over realistic sizes and
    /// many seeds is the evidence a tensor-core score sheet is safe.
    #[test]
    #[ignore]
    fn matmul_score_selection_flip_rate() {
        let stream = Stream::thread_local_or_default();
        let (heads, dim) = (4usize, 128usize);
        let mut flips = 0usize;
        let mut rows = 0usize;
        let mut cases = 0usize;
        for &(s, blocks) in &[(1usize, 4096usize), (4, 8192), (8, 16384), (16, 32768), (1, 65536)] {
            for seed0 in 0..24u64 {
                let total = blocks * 4;
                let pos_start = total - s;
                let mut seed = 0x9e37_79b9 ^ seed0.wrapping_mul(0x1000_0000_01);
                let qv: Vec<half::bf16> = (0..s * heads * dim)
                    .map(|_| half::bf16::from_f32(lcg(&mut seed)))
                    .collect();
                let pv: Vec<half::bf16> = (0..blocks * dim)
                    .map(|_| half::bf16::from_f32(lcg(&mut seed)))
                    .collect();
                let q = Array::from_slice(&qv, &[1, s as i32, heads as i32, dim as i32]);
                let pooled = Array::from_slice(&pv, &[1, blocks as i32, dim as i32]);

                let (ids, valid) =
                    select_blocks(&q, &pooled, pos_start as i32, total as i32, blocks, &stream)
                        .expect("fused selector unavailable");
                let kid: Vec<i32> = ids.as_dtype(Dtype::Int32).unwrap().as_slice::<i32>().to_vec();
                let kvalid: Vec<u8> = valid.as_dtype(Dtype::Uint8).unwrap().as_slice::<u8>().to_vec();

                let reference = matmul_reference(&q, &pooled, s, blocks, pos_start, heads, dim);

                for r in 0..s {
                    let lo = r * 512;
                    let hi = lo + 512;
                    let a: std::collections::HashSet<i32> = kid[lo..hi]
                        .iter()
                        .zip(kvalid[lo..hi].iter())
                        .filter(|(_, v)| **v != 0)
                        .map(|(i, _)| *i)
                        .collect();
                    let b: std::collections::HashSet<i32> = reference[lo..hi]
                        .iter()
                        .filter(|(_, v)| *v != 0)
                        .map(|(i, _)| *i)
                        .collect();
                    flips += a.symmetric_difference(&b).count();
                    rows += 1;
                }
                cases += 1;
            }
        }
        eprintln!(
            "[score-equiv] cases={cases} rows={rows} block_flips={flips} ({:.4}% of selected)",
            flips as f64 / (rows as f64 * 512.0) * 100.0
        );
    }
}
