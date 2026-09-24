//! QSA sparse-attention indexer: pooled key blocks, top-k budget selection,
//! boolean keep mask combined with causal attention.

use lisa_mlx::ops::indexing::{Ellipsis, IndexOp};
use lisa_mlx::{ops, Array, Dtype};

use crate::core::loader::TensorSource;
use crate::core::cache::IndexerTape;
use crate::core::norm::{rope_partial, RmsNorm, Rotary};
use crate::core::quant::QuantizedLinear;

/// The indexer key tape: EXACT storage (no reserve), one row per token.


/// QSA indexer for one full-attention layer.
pub struct QsaIndexer {
    pub index_qk_proj: QuantizedLinear,
    pub q_layer_norm: RmsNorm,
    pub k_layer_norm: RmsNorm,
    pub heads: usize,
    pub head_dim: usize,
    pub token_budget: usize,
    pub compress_ratio: usize,
    pub block_top_k: usize,
}

impl QsaIndexer {
    pub fn load<S: TensorSource>(src: &mut S, prefix: &str, eps: f32) -> anyhow::Result<Self> {
        Ok(Self {
            index_qk_proj: QuantizedLinear::load(src, prefix, "index_qk_proj")?,
            q_layer_norm: RmsNorm::load(src, &format!("{prefix}.q_layernorm"), eps, None)?,
            k_layer_norm: RmsNorm::load(src, &format!("{prefix}.k_layernorm"), eps, None)?,
            heads: 4,
            head_dim: 128,
            token_budget: 2048,
            compress_ratio: 4,
            block_top_k: 2048 / 4,
        })
    }

    /// Returns the sparse block selection, or `None` while the visible context
    /// fits the budget.
    pub fn forward(
        &self,
        x: &Array,
        rope: &Rotary,
        tape: Option<&mut IndexerTape>,
        offset: usize,
    ) -> anyhow::Result<Option<QsaSelection>> {
        let b = x.dim(0);
        let s = x.dim(1);

        let qk = self.index_qk_proj.forward(x)?;
        let split = (self.heads * self.head_dim) as i32;
        let q = qk.index((Ellipsis, 0..split));
        let q = q.reshape(&[b, s, self.heads as i32, self.head_dim as i32])?;
        let raw_k = qk.index((Ellipsis, split..));
        let raw_k = raw_k.reshape(&[b, s, self.head_dim as i32])?;

        let cr = self.compress_ratio as i32;
        // Pooling is lazy and incremental: nothing is pooled until the visible
        // context exceeds the budget, and after that only the newly completed
        // blocks. The raw keys live in a pre-allocated, in-place buffer, so an
        // append is O(s) rather than a whole-tape copy. (The old path re-read
        // the whole tape every call: 4 ms + 1.3 ms per layer at 15K.)
        let (pooled, blocks, kv_len) = match tape {
            Some(t) => {
                t.append(&raw_k, self.compress_ratio)?;
                let total = t.total;
                if total <= self.token_budget {
                    return Ok(None);
                }
                let target = total / self.compress_ratio;
                if t.pooled_upto < target {
                    let start = t.pooled_upto;
                    let raw = t
                        .raw_slice(start * self.compress_ratio, target * self.compress_ratio)
                        .ok_or_else(|| anyhow::anyhow!("indexer tape raw missing"))?
                        .contiguous()?;
                    let n_new = target - start;
                    let blk = raw
                        .reshape(&[b, n_new as i32, cr, self.head_dim as i32])?
                        .as_dtype(Dtype::Float32)?
                        .mean_axis(2, None)?
                        .as_dtype(raw_k.dtype())?;
                    let blk = self.k_layer_norm.forward(&blk)?;
                    let starts: Vec<i32> = (start..target)
                        .map(|n| (n * self.compress_ratio) as i32)
                        .collect();
                    let starts = Array::from_slice(&starts, &[1i32, n_new as i32]);
                    let (cos_k, sin_k) = rope.cos_sin(&starts)?;
                    let blk = rope_partial(&blk, &cos_k, &sin_k)?;
                    t.push_pooled(&blk)?;
                }
                let pv = t
                    .pooled_view()
                    .ok_or_else(|| anyhow::anyhow!("indexer pooled missing"))?;
                (pv, target, total)
            }
            None => {
                let blocks = s as usize / self.compress_ratio;
                let kv_len = s as usize;
                if kv_len <= self.token_budget {
                    return Ok(None);
                }
                let pooled = raw_k
                    .index((.., 0..(blocks * self.compress_ratio) as i32, ..))
                    .reshape(&[b, blocks as i32, cr, self.head_dim as i32])?
                    .as_dtype(Dtype::Float32)?
                    .mean_axis(2, None)?
                    .as_dtype(raw_k.dtype())?;
                let pooled = self.k_layer_norm.forward(&pooled)?;
                let starts: Vec<i32> = (0..blocks as i32).map(|n| n * cr).collect();
                let starts = Array::from_slice(&starts, &[1i32, blocks as i32]);
                let (cos_k, sin_k) = rope.cos_sin(&starts)?;
                (rope_partial(&pooled, &cos_k, &sin_k)?, blocks, kv_len)
            }
        };

        // The queries are `[B, S, heads, headDim]`: cos/sin `[1, S, rot]` need a
        // heads axis inserted so they broadcast over the head dimension.
        let q_pos: Vec<i32> = (offset as i32..(offset + s as usize) as i32).collect();
        let q_pos = Array::from_slice(&q_pos, &[1i32, s]);
        let (cos_q, sin_q) = rope.cos_sin(&q_pos)?;
        let q = self.q_layer_norm.forward(&q)?;
        let cos_q = cos_q.expand_dims(2)?;
        let sin_q = sin_q.expand_dims(2)?;
        let q = rope_partial(&q, &cos_q, &sin_q)?;

        // Fused O(N) selector (mode blocks; see NOTICE). Falls
        // back to the MLX-op scoring + argpartition below when unavailable.
        {
            let stream = lisa_mlx::Stream::thread_local_or_default();
            let total = (offset + s as usize) as i32;
            let qsa_prof = lisa_mlx::env_flag("LISA_QSA_PROF");
            let t0 = std::time::Instant::now();
            let sel = lisa_mlx::qsa::select_blocks(&q, &pooled, offset as i32, total, blocks, &stream);
            if qsa_prof {
                if let Some((ref ids, _)) = sel {
                    let _ = ids.eval();
                }
                eprintln!(
                    "[qsa-prof] offset={offset} s={s} blocks={blocks} select {:.2} ms",
                    t0.elapsed().as_secs_f64() * 1e3
                );
            }
            if let Some((ids, valid)) = sel
            {
                let ids = ids.reshape(&[b, s, 512])?;
                let valid = valid.reshape(&[b, s, 512])?;
                // Host-side: Metal binary kernels have no i32 variant,
                // and these are [1, s] index arrays anyway.
                let q_pos_v: Vec<i32> = (offset as i32..(offset + s as usize) as i32).collect();
                let complete_v: Vec<f32> = q_pos_v.iter().map(|&p| ((p + 1).max(0) / cr) as f32).collect();
                let q_pos = Array::from_slice(&q_pos_v, &[1i32, s]);
                let complete = Array::from_slice(&complete_v, &[1i32, s]);
                return Ok(Some(QsaSelection {
                    block_ids: ids,
                    block_valid: valid,
                    complete,
                    q_pos,
                    cr,
                    kv_len: total,
                }));
            }
        }

        // scores[b, s, n] = (sum_h relu(q . pooled)) / sqrt(128), as one matmul.
        let qf = q.as_dtype(Dtype::Float32)?;
        let pf = pooled.as_dtype(Dtype::Float32)?;
        let flat = qf.reshape(&[b, s * self.heads as i32, self.head_dim as i32])?;
        let contracted = ops::matmul(&flat, pf.transpose_axes(&[0, 2, 1])?)?;
        let scores = contracted.reshape(&[b, s, self.heads as i32, blocks as i32])?;
        let scores = scores.transpose_axes(&[0, 1, 3, 2])?;
        let zero = Array::from_f32(0.0);
        let scores = ops::maximum(&scores, &zero)?.sum_axis(-1, None)?;
        let scores = scores / (self.head_dim as f32).sqrt();

        // A block is a candidate only when it lies entirely in the query's past.
        // FLOOR DIVISION — load-bearing. Host-side (no i32 Metal binary op).
        let complete_v: Vec<f32> = (offset as i32..(offset + s as usize) as i32)
            .map(|p| ((p + 1).max(0) / cr) as f32)
            .collect();
        let complete = Array::from_slice(&complete_v, &[1i32, s]);
        let block_ids: Vec<i32> = (0..blocks as i32).collect();
        let block_ids = Array::from_slice(&block_ids, &[1i32, 1, blocks as i32]);
        let visible = lisa_mlx::ops::broadcast_to(&block_ids, &[b, s, blocks as i32])?.lt(&complete.expand_dims(-1)?)?;
        let neg_inf = lisa_mlx::ops::broadcast_to(&Array::from_f32(f32::NEG_INFINITY), scores.shape())?;
        let scores = lisa_mlx::ops::r#where(&visible, &scores, &neg_inf)?;

        // Model contract tie-break: `score - block_id * 1e-12`. It makes the
        // common all-zero-ReLU tie prefer the LOWER block id (otherwise a
        // higher id wins the cutoff). Applied after the -inf mask, which stays
        // -inf. `block_id` is the same [1,1,blocks] row.
        let bid = block_ids.as_dtype(Dtype::Float32)?;
        let scores = scores.subtract(&bid.multiply(Array::from_f32(1e-12))?)?;

        let k = self.block_top_k.min(blocks);
        let top = ops::argpartition_axis(&(-&scores), (k - 1) as i32, -1)?;
        let top = top.index((Ellipsis, 0..k as i32)).contiguous()?;
        let picked = visible.take_along_axis(&top, -1)?;
        Ok(Some(QsaSelection {
            block_ids: top,
            block_valid: picked,
            complete,
            q_pos,
            cr,
            kv_len: kv_len as i32,
        }))
    }
}

/// The QSA indexer's per-query sparse selection: the `block_top_k` selected
/// complete blocks, which are valid, and where the query's own partial-block
/// tail begins. Attention consumes this either by streaming only the selected
/// tokens (the block-sparse kernel) or by expanding a dense keep mask (the
/// fallback when the kernel is unavailable).
pub struct QsaSelection {
    /// `[B, S, K]` int32 selected complete block ids.
    pub block_ids: Array,
    /// `[B, S, K]` bool — whether each selected block is fully in the past.
    pub block_valid: Array,
    /// `[B, S]` int32 — number of complete blocks visible to each query.
    pub complete: Array,
    /// `[1, S]` int32 — absolute query positions.
    pub q_pos: Array,
    pub cr: i32,
    pub kv_len: i32,
}
