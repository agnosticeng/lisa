#![cfg(test)]
#![allow(dead_code)]
//! Frozen pre-migration indexer oracle; never used in production.
use crate::bindings::glm53_oracle_helpers::{mm, narrow};
use crate::core::cache::IndexerTape;
use lisa_metal::{Array, Dtype};
const LAYER_NORM_EPS: f32 = 1e-6;
const RAW_ROLLBACK_ROWS: usize = 32;
fn indexer_topk(scores: &Array, top_k: usize, stream: &lisa_metal::Stream) -> Option<Array> {
    if crate::core::quant::glm53_indexer_topk_shuffle_override().unwrap_or(true) {
        lisa_metal::kernels::sparse::kernels::dsa_indexer_topk_shuffle(scores, top_k, stream)
    } else {
        lisa_metal::kernels::sparse::kernels::dsa_indexer_topk(scores, top_k, stream)
    }
}

pub struct Indexer {
    pub(super) wq_b_w: Array,    // [4096, 1536], source dtype
    pub(super) wk_w: Array,      // [128, 4096], source dtype
    pub(super) k_norm_w: Array,  // [128]
    pub(super) k_norm_b: Array,  // [128]
    pub(super) weights_w: Array, // [32, 4096], source dtype
    pub(super) gate_w: Array,    // [128, 4096], source dtype
    pub(super) ape: Array,       // [4, 128] f32
    pub(super) heads: usize,
    pub(super) head_dim: usize,
    pub(super) topk: usize,
    pub(super) kpool: usize,
    pub(super) always_tail: bool,
}

impl Indexer {
    /// Run a plain `[out,in]` indexer projection with ds4's BF16 dispatch:
    /// row GEMV for up to eight rows, tiled GEMM for larger prefill chunks.
    pub(super) fn project(&self, x: &Array, weight: &Array) -> anyhow::Result<Array> {
        anyhow::ensure!(
            x.rank() >= 2 && weight.rank() == 2,
            "invalid indexer projection rank"
        );
        let input_dim = x.dim((x.rank() - 1) as i32) as usize;
        anyhow::ensure!(
            input_dim == weight.dim(1) as usize,
            "indexer projection K mismatch"
        );
        let mut output_shape = x.shape().to_vec();
        let last = output_shape.len() - 1;
        let rows = output_shape[..last]
            .iter()
            .try_fold(1usize, |acc, &d| acc.checked_mul(d as usize))
            .ok_or_else(|| anyhow::anyhow!("indexer projection row count overflow"))?;
        output_shape[last] = weight.dim(0);

        if weight.dtype() == Dtype::Bfloat16 && rows > 0 {
            let input = x.reshape(&[rows as i32, input_dim as i32])?;
            let stream = lisa_metal::Stream::thread_local_or_default();
            let projected = if rows <= 8 {
                lisa_metal::kernels::sparse::kernels::glm53_bf16_gemv(weight, &input, &stream)
            } else {
                lisa_metal::kernels::sparse::kernels::glm53_bf16_gemm(weight, &input, &stream)
            }
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "GLM indexer BF16 projection failed for {rows} rows and {input_dim} inputs"
                )
            })?;
            return Ok(projected.reshape(&output_shape)?);
        }

        let transposed = weight.transpose_axes(&[1, 0])?.as_dtype(Dtype::Float32)?;
        Ok(mm(x, &transposed)?)
    }

    /// Pair short-batch indexer BF16 projections in one dispatch. Every output
    /// row keeps the DS4 GEMV reduction order even when the two projections
    /// use different input and output widths.
    pub(super) fn project_pair(
        &self,
        x_a: &Array,
        weight_a: &Array,
        x_b: &Array,
        weight_b: &Array,
    ) -> anyhow::Result<(Array, Array)> {
        let rows_a = x_a.shape()[..x_a.rank() - 1]
            .iter()
            .try_fold(1usize, |acc, &d| acc.checked_mul(d as usize))
            .ok_or_else(|| anyhow::anyhow!("indexer projection row count overflow"))?;
        let rows_b = x_b.shape()[..x_b.rank() - 1]
            .iter()
            .try_fold(1usize, |acc, &d| acc.checked_mul(d as usize))
            .ok_or_else(|| anyhow::anyhow!("indexer projection row count overflow"))?;
        if rows_a > 0
            && rows_a == rows_b
            && rows_a <= 8
            && weight_a.dtype() == Dtype::Bfloat16
            && weight_b.dtype() == Dtype::Bfloat16
        {
            anyhow::ensure!(weight_a.rank() == 2 && weight_b.rank() == 2);
            let input_dim_a = x_a.dim((x_a.rank() - 1) as i32) as usize;
            let input_dim_b = x_b.dim((x_b.rank() - 1) as i32) as usize;
            anyhow::ensure!(
                weight_a.dim(1) as usize == input_dim_a && weight_b.dim(1) as usize == input_dim_b,
                "paired indexer projection K mismatch"
            );
            let input_a = x_a.reshape(&[rows_a as i32, input_dim_a as i32])?;
            let input_b = x_b.reshape(&[rows_b as i32, input_dim_b as i32])?;
            let stream = lisa_metal::Stream::thread_local_or_default();
            let (a, b) = lisa_metal::kernels::sparse::kernels::glm53_bf16_pair_gemv(
                weight_a, &input_a, weight_b, &input_b, &stream,
            )
            .ok_or_else(|| {
                anyhow::anyhow!("GLM indexer paired BF16 projection failed for {rows_a} rows")
            })?;
            let mut shape_a = x_a.shape().to_vec();
            *shape_a.last_mut().expect("rank checked by caller") = weight_a.dim(0);
            let mut shape_b = x_b.shape().to_vec();
            *shape_b.last_mut().expect("rank checked by caller") = weight_b.dim(0);
            return Ok((a.reshape(&shape_a)?, b.reshape(&shape_b)?));
        }
        Ok((self.project(x_a, weight_a)?, self.project(x_b, weight_b)?))
    }

    /// Host-assisted selection: pooling + scoring as Array ops, top-k and
    /// tail on the host. Returns `sel [B, T, width]` i64 selection token
    /// indices (-1 = none).
    /// Host-assisted selection from the indexer tape: the packed
    /// `[ik | gs]` rows live in `tape.raw`; pooling is lazy and incremental
    /// (only newly completed kpool-groups are pooled). Top-k and the tail
    /// run on the host with ABSOLUTE token positions.
    /// Pooling + scoring, GPU-only (shared by the host top-k path and the
    /// decode GPU-mask path): returns `index_sc [B, T, P]` f32.
    pub(super) fn project_score_inputs(
        &self,
        x: &Array,
        qr: &Array,
    ) -> anyhow::Result<(Array, Array)> {
        let _trace = lisa_metal::trace::span("glm53.dsa.indexer_input_projection");
        let (b, t, heads, hd) = (
            x.dim(0) as usize,
            x.dim(1) as usize,
            self.heads,
            self.head_dim,
        );
        let (iq, weights) = self.project_pair(qr, &self.wq_b_w, x, &self.weights_w)?;
        let iq = iq
            .as_dtype(Dtype::Float32)?
            .reshape(&[b as i32, t as i32, heads as i32, hd as i32])?
            .transpose_axes(&[0, 2, 1, 3])?
            .contiguous()?;
        let weights = weights.as_dtype(Dtype::Float32)?;
        Ok((iq, weights))
    }

    pub(super) fn update_pooled_cache(
        &self,
        tape: &mut IndexerTape,
        capture: Option<&crate::core::capture::Capture>,
        layer: usize,
    ) -> anyhow::Result<()> {
        let kpool = self.kpool;
        let total = tape.total;
        let p_n = total / kpool;
        if p_n == 0 || tape.pooled_upto >= p_n {
            return Ok(());
        }

        let _trace = lisa_metal::trace::span_detail("glm53.dsa.pool_update", layer as u64);
        let start = tape.pooled_upto;
        let raw = tape
            .raw_slice(start * kpool, p_n * kpool)
            .ok_or_else(|| anyhow::anyhow!("indexer tape raw missing"))?;
        let (b, hd) = (raw.dim(0) as usize, self.head_dim);
        let raw = narrow(&raw, 2, 0, 2 * hd)?.contiguous()?;
        let n_new = p_n - start;
        let grouped = raw.reshape(&[b as i32, n_new as i32, kpool as i32, (2 * hd) as i32])?;
        let mut fused_done = false;
        if !std::env::var("LISA_GLM53_POOL_OPS")
            .map(|v| v == "1")
            .unwrap_or(false)
        {
            let raw_k = narrow(&grouped, 3, 0, hd)?.contiguous()?;
            let gate = narrow(&grouped, 3, hd, hd)?.contiguous()?;
            let stream = lisa_metal::Stream::thread_local_or_default();
            if let Some(pool_k) = lisa_metal::kernels::sparse::kernels::indexer_pool_fused(
                &raw_k,
                &gate,
                &self.ape,
                &self.k_norm_w,
                &self.k_norm_b,
                LAYER_NORM_EPS,
                &stream,
            ) {
                if let Some(c) = capture.filter(|capture| capture.wants(layer)) {
                    c.store(&format!("L{layer}.dsa_pooled_kernel_output"), &pool_k);
                }
                tape.push_pooled_f16(&pool_k)
                    .map_err(|e| anyhow::anyhow!("{e}"))?;
                fused_done = true;
            }
        }
        if !fused_done {
            let ik_g =
                narrow(&grouped, 3, 0, hd)?.reshape(&[(b * n_new * kpool) as i32, hd as i32])?;
            let ik_g = layer_norm(
                &ik_g.reshape(&[b as i32, (n_new * kpool) as i32, hd as i32])?,
                &self.k_norm_w,
                &self.k_norm_b,
                LAYER_NORM_EPS,
            )?
            .reshape(&[b as i32, n_new as i32, kpool as i32, hd as i32])?;
            let gs_g = narrow(&grouped, 3, hd, hd)?.contiguous()?;
            let ape = self
                .ape
                .as_dtype(Dtype::Float32)?
                .expand_dims(0)?
                .expand_dims(0)?
                .broadcast_to(&[b as i32, n_new as i32, kpool as i32, hd as i32])?;
            // Each channel gets a softmax across the four candidate tokens.
            let probs = gs_g.add(&ape)?.softmax_axis(2)?;
            let mut pool_k: Option<Array> = None;
            for j in 0..kpool {
                let pj = narrow(&probs, 2, j, 1)?;
                let kij = narrow(&ik_g, 2, j, 1)?;
                let term = pj.multiply(&kij)?;
                pool_k = Some(match pool_k {
                    Some(a) => a.add(&term)?,
                    None => term,
                });
            }
            let pool_k =
                pool_k
                    .expect("kpool > 0")
                    .reshape(&[b as i32, n_new as i32, hd as i32])?;
            tape.push_pooled_f16(&pool_k)
                .map_err(|e| anyhow::anyhow!("{e}"))?;
        }
        tape.compact_raw_history(RAW_ROLLBACK_ROWS)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        Ok(())
    }

    pub(super) fn index_scores(
        &self,
        tape: &mut IndexerTape,
        iq: &Array,
        weights: &Array,
        offset: usize,
        t: usize,
        capture: Option<&crate::core::capture::Capture>,
        layer: usize,
        prefill_mode: bool,
    ) -> anyhow::Result<Array> {
        let (b, heads, hd) = (iq.dim(0) as usize, self.heads, self.head_dim);
        anyhow::ensure!(
            iq.shape() == [b as i32, heads as i32, t as i32, hd as i32]
                && weights.shape() == [b as i32, t as i32, heads as i32],
            "GLM indexer score input shape mismatch: q={:?}, weights={:?}, T={t}",
            iq.shape(),
            weights.shape()
        );
        let kpool = self.kpool;
        let total = tape.total;
        let p_n = total / kpool; // complete pools only
        self.update_pooled_cache(tape, capture, layer)?;
        if p_n == 0 {
            // Nothing to pool: the tail alone is visible.
            let width = self.tail_width();
            let mut sel = vec![-1i64; b * t * width];
            for bi in 0..b {
                for qi in 0..t {
                    let abs_qi = offset.saturating_sub(t) + qi;
                    let base = (bi * t + qi) * width;
                    let vis = abs_qi + 1;
                    let tail_count = vis % kpool;
                    let tail_start = vis - tail_count;
                    for m in 0..width.min(kpool - 1) {
                        if m < tail_count {
                            sel[base + m] = (tail_start + m) as i64;
                        }
                    }
                }
            }
            let arr: Vec<i32> = sel.iter().map(|&v| v as i32).collect();
            return Ok(Array::from_slice(&arr, &[b as i32, t as i32, width as i32]));
        }
        let pooled = tape
            .pooled_view()
            .ok_or_else(|| anyhow::anyhow!("indexer pooled missing"))?; // [B,P,128]

        let score_scale = ((hd * heads) as f64).powf(-0.5) as f32;
        let stream = lisa_metal::Stream::thread_local_or_default();
        let scores = {
            let _trace = lisa_metal::trace::span_detail("glm53.dsa.indexer_scores", layer as u64);
            let scorer = if prefill_mode && t == 1 {
                lisa_metal::kernels::sparse::kernels::glm53_indexer_scores_batch
            } else if crate::core::quant::glm53_indexer_decode_rows4_override().unwrap_or(true) {
                lisa_metal::kernels::sparse::kernels::glm53_indexer_scores_rows4
            } else {
                lisa_metal::kernels::sparse::kernels::glm53_indexer_scores
            };
            scorer(
                &iq,
                &weights,
                &pooled,
                offset.saturating_sub(t),
                kpool,
                score_scale,
                &stream,
            )
            .ok_or_else(|| anyhow::anyhow!("glm53 dsa: ds4 indexer-score Metal dispatch failed"))?
        };
        if let Some(c) = capture.filter(|capture| capture.wants(layer)) {
            let pos0 = offset.saturating_sub(t);
            c.store(&format!("L{layer}.dsa_indexer_q_p{pos0}_t{t}"), &iq);
            c.store(
                &format!("L{layer}.dsa_indexer_weights_p{pos0}_t{t}"),
                &weights,
            );
            c.store(&format!("L{layer}.dsa_pooled_keys"), &pooled);
            c.store(&format!("L{layer}.dsa_index_scores_p{pos0}_t{t}"), &scores);
        }
        Ok(scores)
    }

    /// Decode GPU-mask path: indexer scores stay on the GPU, the top-`select_k` pools
    /// come from ds4's blockwise GPU top-k, and the additive keep mask is built with
    /// elementwise ops — membership (pool_of(t) among the selected pools) or
    /// the always-selected tail. Returns the additive mask [B, 1, T, tc]
    /// f32 (0.0 keep, -1e30 masked — the host path's exact values).
    pub(super) fn select_keep_gpu(
        &self,
        tape: &mut IndexerTape,
        x: &Array,
        qr: &Array,
        offset: usize,
        t: usize,
        tc: usize,
        layer: usize,
    ) -> anyhow::Result<Array> {
        let b = x.dim(0) as usize;
        let kpool = self.kpool;
        let total = tape.total;
        let p_n = total / kpool;
        let (iq, weights) = self.project_score_inputs(x, qr)?;
        let index_sc = self.index_scores(tape, &iq, &weights, offset, t, None, 0, false)?;
        debug_assert_eq!(t, 1);
        let select_k = (self.topk / kpool).min(p_n);
        // Causal visibility at t == 1: every COMPLETE pool is visible
        // (pool_end(pp) = pp*4+3 <= offset-1 holds for all pp < p_n), so the
        // candidates are exactly the pooled rows — the host path's filter is
        // vacuous here.
        let score_rows = index_sc.reshape(&[b as i32, p_n as i32])?;
        let stream = lisa_metal::Stream::thread_local_or_default();
        let top = {
            let _trace = lisa_metal::trace::span_detail("glm53.dsa.topk_keep", layer as u64);
            indexer_topk(&score_rows, select_k, &stream).ok_or_else(|| {
                anyhow::anyhow!("glm53 dsa: ds4 blockwise top-k Metal dispatch failed")
            })?
        };
        // Membership: pool_of(pos) among the selected pools, OR the tail.
        let _trace_keep =
            lisa_metal::trace::span_detail("glm53.dsa.keep_mask_expand", layer as u64);
        let vis = tc; // t == 1: the query sees tokens 0..tc-1
        let tail_count = vis % kpool;
        let tail_start = vis - tail_count;
        let mut ar_v = Vec::with_capacity(tc);
        for i in 0..tc {
            ar_v.push(i as i32);
        }
        let ar = Array::from_slice(&ar_v, &[tc as i32]); // [tc] i32
        let pool_of = lisa_metal::ops::floor_divide(&ar, &Array::from_int(kpool as i32))?; // [tc]
        let top_b = top
            .expand_dims(1)?
            .broadcast_to(&[b as i32, tc as i32, select_k as i32])?;
        let pool_b = pool_of.expand_dims(0)?.expand_dims(2)?.broadcast_to(&[
            b as i32,
            tc as i32,
            select_k as i32,
        ])?;
        let hi_b = pool_b.le(&top_b)?;
        let member = pool_b.ge(&top_b)?.logical_and(&hi_b)?; // u8 [tc, sk]
        let member_any = member
            .as_dtype(Dtype::Float32)?
            .sum_axis(2, false)?
            .gt(&Array::from_f32_as(0.0f32, Dtype::Float32))?
            .as_dtype(Dtype::Float32)?; // [B,tc] f32 0/1
        let tail = ar
            .ge(&Array::from_int(tail_start as i32))?
            .as_dtype(Dtype::Float32)?
            .expand_dims(0)?
            .broadcast_to(&[b as i32, tc as i32])?;
        let keep = member_any.maximum(&tail)?; // [B,tc] f32 0/1
        // Additive mask: 0.0 keep, -1e30 masked (host-path values).
        let mask_shape = [b as i32, tc as i32];
        let zero = Array::from_f32_as(0.0f32, Dtype::Float32).broadcast_to(&mask_shape)?;
        let neg_big = Array::from_f32_as(-1e30f32, Dtype::Float32).broadcast_to(&mask_shape)?;
        let m = lisa_metal::Array::where_(&keep, &zero, &neg_big)?;
        Ok(m.reshape(&[(b) as i32, 1, 1, (tc) as i32])?)
    }

    /// GPU causal top-pool selection expanded to absolute token ids consumed
    /// by absorbed MLA attention. No score or selection readback crosses to
    /// host. Each query only admits pools whose four tokens are all visible;
    /// the incomplete tail is appended in position order.
    pub(super) fn select_ids_gpu(
        &self,
        tape: &mut IndexerTape,
        x: &Array,
        qr: &Array,
        offset: usize,
        t: usize,
        capture: Option<&crate::core::capture::Capture>,
        layer: usize,
        prefill_mode: bool,
    ) -> anyhow::Result<(Array, usize)> {
        let b = x.dim(0) as usize;
        let kpool = self.kpool;
        let p_n = tape.total / kpool;
        // At one-token decode, every complete pool is visible. The pool
        // expansion therefore guarantees this leading token-ID prefix; its
        // separately reserved incomplete-tail slots may contain -1 sentinels.
        let selected_valid_prefix = if t == 1 {
            (self.topk / kpool).min(p_n) * kpool
        } else {
            0
        };
        const SCORE_SCRATCH_BYTES: usize = 256 * 1024 * 1024;
        const MAX_SCORE_TOKENS: usize = 2048;
        let score_row_bytes = b
            .saturating_mul(p_n)
            .saturating_mul(Dtype::Float32.size_of());
        let score_cap = (SCORE_SCRATCH_BYTES / score_row_bytes.max(1)).clamp(1, MAX_SCORE_TOKENS);
        let rows_per_chunk = score_cap.min(t.max(1));
        let query_start = offset.saturating_sub(t);
        // Match ds4's prefill order: project the complete query batch once,
        // then chunk only the score tensor's working set. Splitting the tail
        // before projection changes a large BF16 GEMM into a small-row GEMV.
        let (iq_all, weights_all) = self.project_score_inputs(x, qr)?;
        let mut selections = Vec::new();
        let mut q0 = 0usize;
        while q0 < t {
            let n = rows_per_chunk.min(t - q0);
            let iq_chunk = narrow(&iq_all, 2, q0, n)?;
            let weights_chunk = narrow(&weights_all, 1, q0, n)?;
            let chunk_offset = query_start + q0 + n;
            let scores = self.index_scores(
                tape,
                &iq_chunk,
                &weights_chunk,
                chunk_offset,
                n,
                capture,
                layer,
                prefill_mode,
            )?;
            let selected = self.expand_score_selection(&scores, b, n, p_n, chunk_offset, layer)?;
            if let Some(c) = capture.filter(|capture| capture.wants(layer)) {
                c.store(
                    &format!("L{layer}.dsa_selected_p{}_t{n}", query_start + q0),
                    &selected,
                );
            }
            selections.push(selected);
            q0 += n;
        }
        if selections.len() == 1 {
            return Ok((selections.remove(0), selected_valid_prefix));
        }
        let refs: Vec<&Array> = selections.iter().collect();
        Ok((
            Array::concatenate(&refs, 1)?.contiguous()?,
            selected_valid_prefix,
        ))
    }

    pub(super) fn expand_score_selection(
        &self,
        index_sc: &Array,
        b: usize,
        t: usize,
        p_n: usize,
        offset: usize,
        layer: usize,
    ) -> anyhow::Result<Array> {
        let kpool = self.kpool;
        let q_rows = b * t;
        let select_k = (self.topk / kpool).min(p_n);
        if select_k == 0 {
            anyhow::bail!("glm53 dsa: no complete indexer pools available");
        }

        // The index score tensor already has the [B,T,P] layout. Mask pools
        // that end after this query before sorting; otherwise a high-scoring
        // future pool would consume a top-k slot and hide a valid earlier one.
        let abs_start = offset.saturating_sub(t);
        let mut visible_host = Vec::with_capacity(q_rows);
        for _bi in 0..b {
            for qi in 0..t {
                visible_host.push((abs_start + qi + 1) as i32);
            }
        }
        let visible = Array::from_slice(&visible_host, &[q_rows as i32, 1]);
        let mut pool_end = Vec::with_capacity(p_n);
        for pp in 0..p_n {
            pool_end.push(((pp + 1) * kpool) as i32);
        }
        let pool_end = Array::from_slice(&pool_end, &[1, p_n as i32]);
        let eligible = pool_end
            .broadcast_to(&[q_rows as i32, p_n as i32])?
            .le(&visible.broadcast_to(&[q_rows as i32, p_n as i32])?)?;
        let scores = index_sc.reshape(&[q_rows as i32, p_n as i32])?;
        let masked_fill = Array::from_f32_as(f32::NEG_INFINITY, Dtype::Float32)
            .broadcast_to(&[q_rows as i32, p_n as i32])?;
        let masked = Array::where_(&eligible, &scores, &masked_fill)?;
        let stream = lisa_metal::Stream::thread_local_or_default();
        let top = {
            let _trace = lisa_metal::trace::span_detail("glm53.dsa.topk", layer as u64);
            indexer_topk(&masked, select_k, &stream).ok_or_else(|| {
                anyhow::anyhow!("glm53 dsa: ds4 blockwise top-k Metal dispatch failed")
            })?
        };
        let expanded = {
            let _trace =
                lisa_metal::trace::span_detail("glm53.dsa.selection_expansion", layer as u64);
            lisa_metal::kernels::sparse::kernels::glm53_expand_pool_selection_for_attention(
                &top,
                t,
                abs_start,
                kpool,
                self.always_tail,
                &stream,
            )
            .ok_or_else(|| {
                anyhow::anyhow!("glm53 dsa: ds4 pool-selection expansion Metal dispatch failed")
            })?
        };
        Ok(expanded.reshape(&[
            b as i32,
            t as i32,
            (select_k * kpool + kpool.saturating_sub(1)) as i32,
        ])?)
    }

    /// Host top-k per query with causal visibility (absolute positions) —
    /// prefill path and the decode fallback (LISA_GLM53_POOL_OPS=1).
    pub(super) fn select(
        &self,
        x: &Array,
        tape: &mut IndexerTape,
        qr: &Array,
        offset: usize,
        t: usize,
        tc: usize,
        prefill_mode: bool,
    ) -> anyhow::Result<Array> {
        let (b, kpool) = (x.dim(0) as usize, self.kpool);
        let (iq, weights) = self.project_score_inputs(x, qr)?;
        let index_sc = self.index_scores(tape, &iq, &weights, offset, t, None, 0, prefill_mode)?;
        let sc_h: Vec<f32> = index_sc.as_dtype(Dtype::Float32)?.to_vec1()?;
        let pool_end = |pp: usize| -> usize { pp * kpool + kpool - 1 };
        let p_n = tape.total / kpool;
        let select_k = (self.topk / kpool).min(p_n);
        let width = select_k * kpool + (kpool - 1);
        let off = offset.saturating_sub(t);
        let mut sel = vec![-1i64; b * t * width];
        for bi in 0..b {
            for qi in 0..t {
                let abs_qi = off + qi;
                let mut cand: Vec<(f32, usize)> = Vec::with_capacity(p_n);
                for pp in 0..p_n {
                    if pool_end(pp) <= abs_qi {
                        cand.push((sc_h[(bi * t + qi) * p_n + pp], pp));
                    }
                }
                cand.sort_by(|a, b2| b2.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
                let base = (bi * t + qi) * width;
                let mut w = 0usize;
                for (_, pp) in cand.iter().take(select_k) {
                    for j in 0..kpool {
                        let idx = (pp * kpool + j) as i64;
                        if idx < tc as i64 {
                            sel[base + w] = idx;
                            w += 1;
                        }
                    }
                }
                // Always-select tail: the last (vis % kpool) tokens.
                let vis = abs_qi + 1;
                let tail_count = vis % kpool;
                let tail_start = vis - tail_count;
                for m in 0..(kpool - 1) {
                    if m < tail_count {
                        sel[base + w] = (tail_start + m) as i64;
                    } else {
                        sel[base + w] = -1;
                    }
                    w += 1;
                }
            }
        }
        let arr: Vec<i32> = sel.iter().map(|&v| v as i32).collect();
        Ok(Array::from_slice(&arr, &[b as i32, t as i32, width as i32]))
    }

    pub(super) fn tail_width(&self) -> usize {
        self.kpool - 1
    }
}

pub fn layer_norm(x: &Array, w: &Array, b: &Array, eps: f32) -> anyhow::Result<Array> {
    let xf = x.as_dtype(Dtype::Float32)?;
    let mean = xf.mean_axis(2, true)?; // [B,T,1]
    let cent = xf.subtract(&mean)?;
    let var = cent.multiply(&cent)?.mean_axis(2, true)?;
    let inv = var.add(&Array::from_f32_as(eps, Dtype::Float32))?.rsqrt()?;
    let n = cent.multiply(&inv)?;
    let out = n
        .multiply(&w.as_dtype(Dtype::Float32)?)?
        .add(&b.as_dtype(Dtype::Float32)?)?;
    Ok(out)
}
