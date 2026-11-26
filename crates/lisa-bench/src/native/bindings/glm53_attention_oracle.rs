#![cfg(test)]
// Frozen GLM DSA orchestration before input/cache/fallback plan extraction.
//! DSA (`deepseek_sparse_attention`) layers for `glm5_next`.
//!
//! NoPE MLA (no RoPE at all):
//! q via a low-rank branch (`q_a` 4096->1536, RMS, `q_b` 1536->16384), the
//! shared latent `kv_a_proj_with_mqa` 4096->512 cached per token, per-head
//! k/v projections sliced out of the packed Q8 `kv_b` (512->64x(256+256)),
//! and a lightning indexer that picks the top-2048 key tokens per query.
//!
//! Long-context decode and prefill use absorbed MLA attention over selected
//! compact latents; contexts below the sparse crossover use dense projected
//! K/V. The long-context path must not fall back to a full-context projection.

use crate::bindings::glm53_oracle_helpers::narrow;
use crate::core::cache::{FullAttentionCache, IndexerTape};
use crate::core::loader::Weights;
use crate::core::quant::QuantizedLinear;
use crate::core::quant::QuantizedLinearExecution;
use lisa_metal::{Array, Dtype, ops};

pub use crate::bindings::glm53_indexer::Indexer;

pub struct Dsa {
    q_a: QuantizedLinear,
    q_b: QuantizedLinear,
    kv_a: QuantizedLinear,
    kv_b: QuantizedLinear,
    o_proj: QuantizedLinear,
    q_a_norm: Array,  // F32 [1536]
    kv_a_norm: Array, // F32 [512]
    indexer: Indexer,
    heads: usize,
    q_head_dim: usize,
    v_head_dim: usize,
    norm_eps: f32,
}

pub(super) struct DsaProjections {
    latent: Option<Array>,
    compact_written: Option<(usize, usize)>,
    query_rank: Array,
    query: Array,
    #[cfg(test)]
    pub(super) indexer_key: Array,
    pub(super) indexer_gate: Array,
}

impl Dsa {
    pub fn load(
        w: &mut Weights,
        prefix: &str,
        heads: usize,
        q_head_dim: usize,
        v_head_dim: usize,
        index_heads: usize,
        index_head_dim: usize,
        index_topk: usize,
        index_kpool: usize,
        always_tail: bool,
        norm_eps: f32,
    ) -> anyhow::Result<Self> {
        let take = |w: &mut Weights, name: &str| -> anyhow::Result<Array> {
            w.remove(name)
                .ok_or_else(|| anyhow::anyhow!("glm53 dsa: {name} missing"))
        };
        let p = |s: &str| format!("{prefix}.{s}");
        let mut q_a = QuantizedLinear::load(w, &p("q_a_proj"), "")?;
        q_a.set_quant(64, 8);
        q_a.scales = q_a.scales.as_dtype(Dtype::Float32)?;
        q_a.biases = q_a.biases.as_dtype(Dtype::Float32)?;
        let mut q_b = QuantizedLinear::load(w, &p("q_b_proj"), "")?;
        q_b.set_quant(64, 8);
        q_b.scales = q_b.scales.as_dtype(Dtype::Float32)?;
        q_b.biases = q_b.biases.as_dtype(Dtype::Float32)?;
        let mut kv_a = QuantizedLinear::load(w, &p("kv_a_proj_with_mqa"), "")?;
        kv_a.set_quant(64, 8);
        kv_a.scales = kv_a.scales.as_dtype(Dtype::Float32)?;
        kv_a.biases = kv_a.biases.as_dtype(Dtype::Float32)?;
        let mut kv_b = QuantizedLinear::load(w, &p("kv_b_proj"), "")?;
        kv_b.set_quant(64, 8);
        kv_b.scales = kv_b.scales.as_dtype(Dtype::Float32)?;
        kv_b.biases = kv_b.biases.as_dtype(Dtype::Float32)?;
        let mut o_proj = QuantizedLinear::load(w, &p("o_proj"), "")?;
        o_proj.set_quant(64, 8);
        o_proj.scales = o_proj.scales.as_dtype(Dtype::Float32)?;
        o_proj.biases = o_proj.biases.as_dtype(Dtype::Float32)?;
        let q_a_norm = take(w, &p("q_a_layernorm.weight"))?.as_dtype(Dtype::Float32)?;
        let kv_a_norm = take(w, &p("kv_a_layernorm.weight"))?.as_dtype(Dtype::Float32)?;

        let indexer = Indexer {
            context: crate::metal_context::execution_context(),
            wq_b_w: take(w, &p("indexer.wq_b.weight"))?,
            wk_w: take(w, &p("indexer.wk.weight"))?,
            k_norm_w: take(w, &p("indexer.k_norm.weight"))?.as_dtype(Dtype::Float32)?,
            k_norm_b: take(w, &p("indexer.k_norm.bias"))?.as_dtype(Dtype::Float32)?,
            weights_w: take(w, &p("indexer.weights_proj.weight"))?,
            gate_w: take(w, &p("indexer.index_kpool_compress_gate"))?,
            ape: take(w, &p("indexer.index_kpool_compress_ape"))?,
            heads: index_heads,
            head_dim: index_head_dim,
            topk: index_topk,
            kpool: index_kpool,
            always_tail,
        };
        Ok(Self {
            q_a,
            q_b,
            kv_a,
            kv_b,
            o_proj,
            q_a_norm,
            kv_a_norm,
            indexer,
            heads,
            q_head_dim,
            v_head_dim,
            norm_eps,
        })
    }

    /// GLM's MTP DSA step uses a private contiguous window and deliberately
    /// skips the learned target-layer indexer. `position` is the absolute
    /// target position represented by `x`; selected rows are `[min_position,
    /// position]`, matching ds4's `glm_graph_mtp_step`.
    pub fn forward_mtp(
        &self,
        x: &Array,
        tape: &mut IndexerTape,
        position: usize,
        min_position: usize,
    ) -> anyhow::Result<Array> {
        anyhow::ensure!(
            x.dim(0) == 1 && x.dim(1) == 1,
            "glm53 MTP DSA expects one token, got {:?}",
            x.shape()
        );
        anyhow::ensure!(min_position <= position, "glm53 MTP window is inverted");

        let heads = self.heads;
        let qd = self.q_head_dim;
        let vd = self.v_head_dim;
        let scale = (qd as f64).powf(-0.5) as f32;
        // MTP uses the same reference projection dispatcher as the trunk. Preserve
        // the Q-A -> Q norm -> Q-B -> KV-A order and the native one-row Q8
        // reduction before converting the private compact cache to F16.
        let q_a_raw = self.q_a.forward_glm53_prefill(x)?;
        let qr = glm53_weighted_rms_norm(&q_a_raw, &self.q_a_norm, self.norm_eps)?;
        let q = self
            .q_b
            .forward_glm53_prefill(&qr)?
            .reshape(&[1, 1, heads as i32, qd as i32])?
            .transpose_axes(&[0, 2, 1, 3])?
            .contiguous()?;
        let kv_raw = self.kv_a.forward_glm53_prefill(x)?;
        let c = glm53_kv_lora_rms_norm(&kv_raw, &self.kv_a_norm, self.norm_eps)?;
        tape.write_compact_f16_at(&c, position)
            .map_err(|e| anyhow::anyhow!("glm53 MTP compact cache: {e}"))?;

        let is_window_fixture = false;
        let compact_start = if is_window_fixture { min_position } else { 0 };
        let compact = tape
            .compact_slice(compact_start, position + 1)
            .ok_or_else(|| anyhow::anyhow!("glm53 MTP compact cache missing at {position}"))?;
        let width = position - min_position + 1;
        let selected_start = if is_window_fixture { 0 } else { min_position };
        let selected: Vec<i32> = (selected_start..selected_start + width)
            .map(|p| p as i32)
            .collect();
        let selected = Array::from_slice(&selected, &[1, 1, width as i32]);
        let stream = lisa_metal::Stream::thread_local_or_default();
        let attended = crate::bindings::glm53_absorbed_attention::dsa_absorbed_attention(
            &q,
            &self.kv_b.weight,
            &self.kv_b.scales,
            &self.kv_b.biases,
            self.kv_b.group_size as usize,
            &compact,
            &selected,
            scale,
            &stream,
        )
        .ok_or_else(|| {
            anyhow::anyhow!("glm53 MTP indexed absorbed attention declined at position {position}")
        })?;
        let out = attended
            .transpose_axes(&[0, 2, 1, 3])?
            .contiguous()?
            .reshape(&[1, 1, (heads * vd) as i32])?;
        let projected = self.output_project(&out)?;
        Ok(projected)
    }

    fn output_project(&self, x: &Array) -> anyhow::Result<Array> {
        if crate::core::quant::glm53_q8_native_prefill_enabled() {
            // The reference's native Q8 crossover applies to the attention output
            // projection as well as the input and feed-forward projections.
            return Ok(self.o_proj.forward_glm53_prefill(x)?);
        }
        if x.rank() == 3 && x.dim(2) == 16_384 && x.dtype() == Dtype::Float32 {
            let rows = (x.dim(0) as usize)
                .checked_mul(x.dim(1) as usize)
                .ok_or_else(|| anyhow::anyhow!("glm53 DSA output row count overflow"))?;
            if (1..=31).contains(&rows) {
                // The custom-affine reference route uses its per-row affine matvec
                // for small row counts; the generic F32 `_4` kernel has a
                // different reduction schedule even with identical weights.
                return Ok(self.o_proj.forward_glm53_prefill(x)?);
            }
            let rows_i32 = i32::try_from(rows)
                .map_err(|_| anyhow::anyhow!("glm53 DSA output row count exceeds i32"))?;
            let flat = x.reshape(&[rows_i32, 16_384])?;
            let stream = lisa_metal::Stream::thread_local_or_default();
            let out = lisa_metal::kernels::sparse::kernels::dsa_o_project(
                &flat,
                &self.o_proj.weight,
                &self.o_proj.scales,
                &self.o_proj.biases,
                self.o_proj.group_size as usize,
                &stream,
            )
            .ok_or_else(|| anyhow::anyhow!("glm53 DS4-scheduled DSA output projection declined"))?;
            return Ok(out.reshape(&[x.dim(0), x.dim(1), 4096])?);
        }
        Ok(self.o_proj.forward_glm53_prefill(x)?)
    }

    pub(super) fn project_inputs(
        &self,
        x: &Array,
        cap: Option<&crate::core::capture::Capture>,
        layer: usize,
        compact_tape: Option<&mut crate::core::cache::IndexerTape>,
    ) -> anyhow::Result<DsaProjections> {
        let (b, t) = (x.dim(0) as usize, x.dim(1) as usize);
        if let Some(capture) = cap.filter(|capture| capture.wants(layer)) {
            capture.store(&format!("L{layer}.dsa_indexer_input_x"), x);
        }
        let heads = self.heads;
        let qd = self.q_head_dim;

        let capture = cap.filter(|capture| capture.wants(layer));
        let project_separately = || -> anyhow::Result<(Array, Array)> {
            // Keep kv_a on the standard affine operator so each shape follows
            // the reference's graph schedule. The former MTP-specific reduction tree
            // changed F32 outputs and sometimes moved the stored F16 value.
            let c_raw = self.kv_a.forward_glm53_prefill(x)?;
            let query_rank_raw = self.q_a.forward_glm53_prefill(x)?;
            Ok((c_raw, query_rank_raw))
        };
        // Keep q_a and kv_a on their separate reference projection schedules.
        // The paired kernel uses MLX qmv_fast for q_a, while the reference's graph
        // dispatches its affine matvec path for each projection. That changes
        // the reduction order and is visible in the composed attention output.
        let (c_raw, query_rank_raw) = project_separately()?;
        let mut compact_written = None;
        let capture_production_qkv = capture.is_some_and(|capture| capture.preserves_dispatch());
        // The reference's decode graph fuses q/kv norm with the compact-cache write for
        // one row. Prefill and MTP verification use separate KV normalization
        // and cache-store kernels, whose reduction order is observable at the
        // F16 cache boundary.
        let batch_qkv_norm = if b > 1 && t == 1 {
            lisa_metal::kernels::sparse::kernels::dsa_qkv_rms_norm_rows(
                &query_rank_raw,
                &self.q_a_norm,
                &c_raw,
                &self.kv_a_norm,
                self.norm_eps,
                &lisa_metal::Stream::thread_local_or_default(),
            )
        } else {
            None
        };
        let (c, qr) = if let Some((qr, c)) = batch_qkv_norm {
            (Some(c), qr)
        } else if b == 1 && t == 1 && (capture.is_none() || capture_production_qkv) {
            if b == 1 {
                if let Some(tape) = compact_tape {
                    let (compact_cache, start) = tape
                        .reserve_compact_f16_append(b as i32, t, 512)
                        .map_err(|e| anyhow::anyhow!("glm53 compact cache reserve: {e}"))?;
                    let direct_qr = {
                        let _trace = lisa_metal::trace::span_detail(
                            "glm53.dsa.qkv_norm_compact_into",
                            layer as u64,
                        );
                        lisa_metal::kernels::sparse::kernels::qkv_norm_compact_into(
                            &query_rank_raw,
                            &self.q_a_norm,
                            &c_raw,
                            &self.kv_a_norm,
                            &compact_cache,
                            start,
                            self.norm_eps,
                            &lisa_metal::Stream::thread_local_or_default(),
                        )
                    };
                    if let Some(qr) = direct_qr {
                        compact_written = Some((start, t));
                        (None, qr)
                    } else if let Some((qr, compact)) =
                        lisa_metal::kernels::sparse::kernels::qkv_norm_compact(
                            &query_rank_raw,
                            &self.q_a_norm,
                            &c_raw,
                            &self.kv_a_norm,
                            self.norm_eps,
                            &lisa_metal::Stream::thread_local_or_default(),
                        )
                    {
                        (Some(compact), qr)
                    } else {
                        (
                            Some(glm53_kv_lora_rms_norm(
                                &c_raw,
                                &self.kv_a_norm,
                                self.norm_eps,
                            )?),
                            glm53_weighted_rms_norm(
                                &query_rank_raw,
                                &self.q_a_norm,
                                self.norm_eps,
                            )?,
                        )
                    }
                } else if let Some((qr, compact)) =
                    lisa_metal::kernels::sparse::kernels::qkv_norm_compact(
                        &query_rank_raw,
                        &self.q_a_norm,
                        &c_raw,
                        &self.kv_a_norm,
                        self.norm_eps,
                        &lisa_metal::Stream::thread_local_or_default(),
                    )
                {
                    (Some(compact), qr)
                } else {
                    (
                        Some(glm53_kv_lora_rms_norm(
                            &c_raw,
                            &self.kv_a_norm,
                            self.norm_eps,
                        )?),
                        glm53_weighted_rms_norm(&query_rank_raw, &self.q_a_norm, self.norm_eps)?,
                    )
                }
            } else if let Some((qr, compact)) =
                lisa_metal::kernels::sparse::kernels::qkv_norm_compact(
                    &query_rank_raw,
                    &self.q_a_norm,
                    &c_raw,
                    &self.kv_a_norm,
                    self.norm_eps,
                    &lisa_metal::Stream::thread_local_or_default(),
                )
            {
                (Some(compact), qr)
            } else {
                (
                    Some(glm53_kv_lora_rms_norm(
                        &c_raw,
                        &self.kv_a_norm,
                        self.norm_eps,
                    )?),
                    glm53_weighted_rms_norm(&query_rank_raw, &self.q_a_norm, self.norm_eps)?,
                )
            }
        } else {
            (
                Some(glm53_kv_lora_rms_norm(
                    &c_raw,
                    &self.kv_a_norm,
                    self.norm_eps,
                )?),
                glm53_weighted_rms_norm(&query_rank_raw, &self.q_a_norm, self.norm_eps)?,
            )
        };
        if let Some(capture) = capture {
            capture.store(&format!("L{layer}.dsa_kv_a_raw"), &c_raw);
            if let Some(c) = &c {
                capture.store(&format!("L{layer}.dsa_lat"), c);
            }
            capture.store(&format!("L{layer}.dsa_qr"), &qr);
        }
        let q = self.q_b.forward_glm53_prefill(&qr)?; // [B,T,64*256]
        if let Some(c) = cap
            && c.wants(layer)
        {
            c.store(&format!("L{layer}.dsa_q"), &q);
        }
        let q = q
            .reshape(&[(b) as i32, (t) as i32, (heads) as i32, (qd) as i32])?
            .transpose_axes(&[0, 2, 1, 3])?
            .contiguous()?; // [B,64,T,256]

        // Store [ik | gs] as F32 and the compact KV latent as F16. Both tapes
        // grow in place and share the same absolute token positions.
        let (ik, gs) = self
            .indexer
            .project_pair(x, &self.indexer.wk_w, x, &self.indexer.gate_w)?;
        // Normalize the raw key while pooling; `gs` is the pool gate.
        if let Some(capture) = cap.filter(|capture| capture.wants(layer)) {
            capture.store(&format!("L{layer}.dsa_indexer_raw_k"), &ik);
            capture.store(&format!("L{layer}.dsa_indexer_pool_gate"), &gs);
        }
        Ok(DsaProjections {
            latent: c,
            compact_written,
            query_rank: qr,
            query: q,
            #[cfg(test)]
            indexer_key: ik,
            indexer_gate: gs,
        })
    }

    /// Forward `x [B,T,H]`. The indexer tape stores F32 key/gate features,
    /// while compact normalized KV latents are held in a separate F16 buffer.
    pub fn forward(
        &self,
        x: &Array,
        cache: &mut FullAttentionCache,
        cap: Option<&crate::core::capture::Capture>,
        layer: usize,
        prefill_mode: bool,
    ) -> anyhow::Result<Array> {
        let projected = {
            let _trace =
                lisa_metal::trace::span_detail("glm53.dsa.input_projections", layer as u64);
            self.project_inputs(x, cap, layer, Some(&mut cache.indexer_tape))?
        };
        let pre_o = self.forward_projected(x, cache, cap, layer, &projected, None, prefill_mode)?;
        let out = {
            let _trace =
                lisa_metal::trace::span_detail("glm53.dsa.output_projection", layer as u64);
            self.output_project(&pre_o)?
        };
        if let Some(capture) = cap.filter(|capture| capture.wants(layer)) {
            capture.store(&format!("L{layer}.dsa_output_project"), &out);
        }
        Ok(out)
    }

    pub(super) fn append_projected_cache(
        &self,
        projected: &DsaProjections,
        cache: &mut FullAttentionCache,
        layer: usize,
        capture: Option<&crate::core::capture::Capture>,
    ) -> anyhow::Result<()> {
        let _trace = lisa_metal::trace::span_detail("glm53.dsa.cache_append", layer as u64);
        self.append_cache_projection_rows(
            &projected.indexer_key,
            &projected.indexer_gate,
            projected.latent.as_ref(),
            projected.compact_written,
            cache,
        )?;
        self.indexer
            .update_pooled_cache(&mut cache.indexer_tape, capture, layer)
    }

    fn append_cache_projection_rows(
        &self,
        indexer_key: &Array,
        indexer_gate: &Array,
        latent: Option<&Array>,
        compact_written: Option<(usize, usize)>,
        cache: &mut FullAttentionCache,
    ) -> anyhow::Result<()> {
        cache
            .indexer_tape
            .append_fields(indexer_key, indexer_gate, self.indexer.kpool)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        if let Some((start, steps)) = compact_written {
            cache
                .indexer_tape
                .commit_compact_f16_append(start, steps)
                .map_err(|e| anyhow::anyhow!("{e}"))?;
        } else {
            cache
                .indexer_tape
                .append_compact_f16(latent.expect("fallback projection has compact latent"))
                .map_err(|e| anyhow::anyhow!("{e}"))?;
        }
        let total = cache.indexer_tape.total;
        cache.offset = total;
        // GLM stores compact latents separately from expanded attention KV.
        cache.keys = None;
        cache.values = None;
        Ok(())
    }

    pub(super) fn select_projected_ids(
        &self,
        x: &Array,
        cache: &mut FullAttentionCache,
        projected: &DsaProjections,
        capture: Option<&crate::core::capture::Capture>,
        layer: usize,
        prefill_mode: bool,
    ) -> anyhow::Result<(Array, usize)> {
        self.indexer.select_ids_gpu(
            &mut cache.indexer_tape,
            x,
            &projected.query_rank,
            cache.offset,
            x.dim(1) as usize,
            capture,
            layer,
            prefill_mode,
        )
    }

    fn forward_projected(
        &self,
        x: &Array,
        cache: &mut FullAttentionCache,
        cap: Option<&crate::core::capture::Capture>,
        layer: usize,
        projected: &DsaProjections,
        native_batch_row: Option<usize>,
        prefill_mode: bool,
    ) -> anyhow::Result<Array> {
        let (b, t) = (x.dim(0) as usize, x.dim(1) as usize);
        let heads = self.heads;
        let qd = self.q_head_dim;
        let vd = self.v_head_dim;
        let scale = (qd as f64).powf(-0.5) as f32;

        let qr = &projected.query_rank;
        let q = &projected.query;

        self.append_projected_cache(projected, cache, layer, cap)?;
        let tc = cache.indexer_tape.total;
        let dense_limit = self.indexer.topk + self.indexer.kpool.saturating_sub(1);
        let _value_slices = (prefill_mode && b == 1 && native_batch_row.is_none()).then(|| {
            crate::bindings::glm53_attention_mode::Glm53DsaValueSliceScope::new(tc - t, dense_limit)
        });
        let compact_latents = cache
            .indexer_tape
            .compact_slice(0, tc)
            .ok_or_else(|| anyhow::anyhow!("glm53 dsa: compact latent cache missing"))?;
        if let Some(capture) = cap.filter(|capture| capture.wants(layer)) {
            let current = narrow(&compact_latents, 1, tc - t, t)?.contiguous()?;
            capture.store(&format!("L{layer}.dsa_cache_latent"), &current);
        }

        // The reference uses its raw FlashAttention kernel for dense-causal prefill
        // slices of at least 24 rows. Keep that path distinct from indexed
        // attention: its F16 query staging and tiled softmax are observable
        // in the layer output.
        if native_batch_row.is_none() && b == 1 && t >= 24 && tc <= dense_limit {
            let query_start = tc - t;
            let stream = lisa_metal::Stream::thread_local_or_default();
            let (attended, q_low, lora) =
                crate::bindings::glm53_absorbed_attention::dsa_absorbed_attention_flash_causal_prefill_stages(
                    q,
                    &self.kv_b.weight,
                    &self.kv_b.scales,
                    &self.kv_b.biases,
                    self.kv_b.group_size as usize,
                    &compact_latents,
                    query_start,
                    scale,
                    &stream,
                    prefill_mode,
                )
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "glm53 dsa: DS4 dense-causal FlashAttention declined at rows {t}, context {tc}"
                    )
                })?;
            let out = attended
                .transpose_axes(&[0, 2, 1, 3])?
                .contiguous()?
                .reshape(&[b as i32, t as i32, (heads * vd) as i32])?;
            if let Some(capture) = cap.filter(|capture| capture.wants(layer)) {
                capture.store(&format!("L{layer}.dsa_q_low"), &q_low);
                capture.store(&format!("L{layer}.dsa_attn_lora"), &lora);
                capture.store(&format!("L{layer}.dsa_pre_o"), &out);
                let tail = t.saturating_sub(4);
                if let Ok(q_tail) = narrow(&q_low, 1, tail, t.min(4)) {
                    capture.store(&format!("L{layer}.dsa_q_low_boundary"), &q_tail);
                }
                if let Ok(lora_tail) = narrow(&lora, 1, tail, t.min(4)) {
                    capture.store(&format!("L{layer}.dsa_attn_lora_boundary"), &lora_tail);
                }
            }
            return Ok(out);
        }

        // The reference uses indexed decode even below the sparse crossover: it fills
        // a contiguous [0, tc) selection there. Keep that explicit prefix path
        // so affine-MLX attention uses the same reduction schedule.
        let selected_absorbed = if t > 1 && tc <= dense_limit {
            // Each verification row sees the committed prefix through its own
            // absolute position. Pad the shared-width selection with invalid
            // IDs so the indexed-lora kernel can process the rows together.
            let query_start = tc.saturating_sub(t);
            let mut ids = Vec::with_capacity(b * t * tc);
            for _batch in 0..b {
                for query in 0..t {
                    let last = query_start + query;
                    ids.extend((0..tc).map(|position| {
                        if position <= last {
                            position as i32
                        } else {
                            -1
                        }
                    }));
                }
            }
            Some((Array::from_slice(&ids, &[b as i32, t as i32, tc as i32]), 0))
        } else if t == 1 || tc > dense_limit {
            Some(if tc <= dense_limit {
                let ids: Vec<i32> = (0..tc as i32).collect();
                (
                    Array::from_slice(&ids, &[1, 1, tc as i32])
                        .broadcast_to(&[b as i32, t as i32, tc as i32])?,
                    tc,
                )
            } else {
                let _trace =
                    lisa_metal::trace::span_detail("glm53.dsa.indexer_selection", layer as u64);
                self.select_projected_ids(x, cache, projected, cap, layer, prefill_mode)?
            })
        } else {
            None
        };
        if let Some((selected, selected_valid_prefix)) = selected_absorbed.as_ref() {
            let stream = lisa_metal::Stream::thread_local_or_default();
            let _trace_absorbed =
                lisa_metal::trace::span_detail("glm53.dsa.absorbed_attention", layer as u64);
            let absorbed = if let Some(capture) = cap.filter(|capture| capture.wants(layer)) {
                crate::bindings::glm53_absorbed_attention::dsa_absorbed_attention_stages(
                    q,
                    &self.kv_b.weight,
                    &self.kv_b.scales,
                    &self.kv_b.biases,
                    self.kv_b.group_size as usize,
                    &compact_latents,
                    selected,
                    scale,
                    &stream,
                    prefill_mode,
                )
                .map(|(attended, q_low, lora)| {
                    if let Some(capture) = cap.filter(|capture| capture.wants(layer)) {
                        capture.store(&format!("L{layer}.dsa_q_low"), &q_low);
                        capture.store(&format!("L{layer}.dsa_attn_lora"), &lora);
                    }
                    let query_rows = q_low.dim(1) as usize;
                    let tail = query_rows.saturating_sub(4);
                    let tail_rows = query_rows.min(4);
                    if let Ok(q_tail) = narrow(&q_low, 1, tail, tail_rows) {
                        capture.store(&format!("L{layer}.dsa_q_low_boundary"), &q_tail);
                    }
                    if let Ok(lora_tail) = narrow(&lora, 1, tail, tail_rows) {
                        capture.store(&format!("L{layer}.dsa_attn_lora_boundary"), &lora_tail);
                    }
                    attended
                })
            } else {
                // The reference keeps MLX affine V-B on the split attention-lora plus
                // value-projection path; its fused indexed decode is only
                // used for native value-weight representations.
                crate::bindings::glm53_absorbed_attention::dsa_absorbed_attention_with_valid_prefix(
                    q,
                    &self.kv_b.weight,
                    &self.kv_b.scales,
                    &self.kv_b.biases,
                    self.kv_b.group_size as usize,
                    &compact_latents,
                    selected,
                    *selected_valid_prefix,
                    scale,
                    &stream,
                    prefill_mode,
                )
            };
            if let Some(attended) = absorbed {
                let out = attended
                    .transpose_axes(&[0, 2, 1, 3])?
                    .contiguous()?
                    .reshape(&[b as i32, t as i32, (heads * vd) as i32])?;
                if let Some(capture) = cap
                    && capture.wants(layer)
                {
                    capture.store(&format!("L{layer}.dsa_pre_o"), &out);
                }
                return Ok(out);
            }
            static WARNED_ABSORBED_FALLBACK: std::sync::atomic::AtomicBool =
                std::sync::atomic::AtomicBool::new(false);
            if !WARNED_ABSORBED_FALLBACK.swap(true, std::sync::atomic::Ordering::Relaxed) {
                eprintln!(
                    "glm53 dsa: absorbed attention declined; dense fallback is allowed only within the dense-equivalent context limit"
                );
            }
            // The dense fallback materializes kv_b for every cached token. At
            // long context this turns one decode into a full-context projection
            // and hides a broken/declined fast kernel behind a severe slowdown.
            // Fail with the recorded Metal diagnostic instead of pretending the
            // the reference indexed path is active.
            anyhow::ensure!(
                tc <= dense_limit,
                "glm53 dsa: indexed absorbed attention declined at context {tc} (selection width {}); refusing the quadratic dense fallback",
                selected.dim(2)
            );
        }

        let keep = if t == 1 && tc > dense_limit {
            let _trace =
                lisa_metal::trace::span_detail("glm53.dsa.indexer_keep_mask", layer as u64);
            self.indexer.select_keep_gpu(
                &mut cache.indexer_tape,
                x,
                qr,
                cache.offset,
                t,
                tc,
                layer,
            )?
        } else if let Some((selected, _)) = selected_absorbed.as_ref() {
            keep_mask_from_selection(x, selected, t, tc)?
        } else if tc <= dense_limit {
            causal_mask(x, b, t, tc)?
        } else {
            let sel = {
                let _trace = lisa_metal::trace::span_detail(
                    "glm53.dsa.indexer_host_selection",
                    layer as u64,
                );
                self.indexer.select(
                    x,
                    &mut cache.indexer_tape,
                    qr,
                    cache.offset,
                    t,
                    tc,
                    prefill_mode,
                )?
            };
            keep_mask_from_selection(x, &sel, t, tc)?
        };

        // Prefill and unsupported decode hardware retain the dense reference
        // path. It is intentionally outside the common one-token decode path.
        let kv = {
            let _trace =
                lisa_metal::trace::span_detail("glm53.dsa.dense_kv_projection", layer as u64);
            self.kv_b
                .forward(&compact_latents.as_dtype(Dtype::Float32)?)?
        };
        let kv = kv.reshape(&[b as i32, tc as i32, heads as i32, (qd + vd) as i32])?;
        let k = narrow(&kv, 3, 0, qd)?
            .transpose_axes(&[0, 2, 1, 3])?
            .contiguous()?;
        let v = narrow(&kv, 3, qd, vd)?
            .transpose_axes(&[0, 2, 1, 3])?
            .contiguous()?;
        if let Some(c) = cap
            && c.wants(layer)
        {
            c.store(&format!("L{layer}.dsa_k"), &k);
            c.store(&format!("L{layer}.dsa_v"), &v);
        }

        if t == 1 {
            let step = crate::bindings::glm53_affine_tower::INC_STEP
                .load(std::sync::atomic::Ordering::Relaxed);
            if let Some(c2) = cap
                && c2.wants(layer)
            {
                let _ = k.eval_enqueue();
                let _ = v.eval_enqueue();
                c2.store(&format!("L{layer}.dec_dsa_k_s{step}"), &k);
                c2.store(&format!("L{layer}.dec_dsa_v_s{step}"), &v);
            }
        }
        // scores = q @ k^T * scale + keep, softmax, @ v — per batch (B=1 v1).
        // The reference flash_attn decode parity (t == 1): the attention runs as TWO
        // fused dispatches (scores, then softmax @ v) over the full cached
        // K/V with the keep mask baked in. Both stages preserve the reference F32
        // graph. Op path remains the fallback
        let mut out_acc: Option<Array> = None;
        let fused = if t == 1 {
            let stream = lisa_metal::Stream::thread_local_or_default();
            let q2 = q.reshape(&[(b) as i32, (heads) as i32, (qd) as i32])?;
            let k_c = k.contiguous()?;
            let v_c = v.contiguous()?;
            let keep2 = keep.reshape(&[(b) as i32, (tc) as i32])?;
            let _trace =
                lisa_metal::trace::span_detail("glm53.dsa.fused_decode_attention", layer as u64);
            lisa_metal::kernels::sparse::kernels::dsa_decode_scores(
                &q2, &k_c, &keep2, scale, &stream,
            )
            .and_then(|sc| {
                lisa_metal::kernels::sparse::kernels::dsa_decode_weighted_v(&sc, &v_c, &stream)
            })
        } else {
            None
        };
        if let Some(flat) = fused {
            // [B*heads, vd] == the per-batch concat the op path builds at t == 1.
            out_acc = Some(flat);
        } else {
            let kt = k.transpose_axes(&[0, 1, 3, 2])?.contiguous()?; // [B,64,256,Tc]
            let scale_a = ops::full::<i32>(&[1], &Array::from_f32_as(scale, Dtype::Float32))?;
            for bi in 0..b {
                let qb =
                    narrow(q, 0, bi, 1)?.reshape(&[(heads) as i32, (t) as i32, (qd) as i32])?;
                // kt is already [B, heads, qd, tc] — reshape straight to the
                // [heads, qd, tc] matmul operand (no re-permute).
                let kb_t =
                    narrow(&kt, 0, bi, 1)?.reshape(&[(heads) as i32, (qd) as i32, (tc) as i32])?;
                if let Some(cp) = cap
                    && cp.wants(layer)
                {
                    cp.store(&format!("L{layer}.dsa_qb"), &qb);
                    cp.store(&format!("L{layer}.dsa_kb_t"), &kb_t);
                }
                let vb =
                    narrow(&v, 0, bi, 1)?.reshape(&[(heads) as i32, (tc) as i32, (vd) as i32])?;
                let sc = qb.matmul(&kb_t)?; // [64,T,Tc]
                let sc = sc.multiply(&scale_a)?;
                let keep_b = narrow(&keep, 0, bi, 1)?; // [1,1,T,Tc] broadcasts over heads
                let sc = sc
                    .add(&keep_b)?
                    .reshape(&[(heads) as i32, (t) as i32, (tc) as i32])?; // back to rank 3
                let probs_b = sc.softmax_axis(-1)?;
                if let Some(cp) = cap
                    && cp.wants(layer)
                {
                    cp.store(&format!("L{layer}.dsa_sc"), &sc);
                    cp.store(&format!("L{layer}.dsa_probs"), &probs_b);
                    cp.store(&format!("L{layer}.dsa_vb"), &vb);
                }
                let ob = probs_b.matmul(&vb)?; // [64,T,256]
                let ob = ob.reshape(&[(heads * t * vd) as i32])?;
                out_acc = Some(match out_acc {
                    Some(a) => Array::concatenate(&[&a, &ob], 0)?,
                    None => ob,
                });
            }
        }
        let out = out_acc.expect("b>0").reshape(&[
            (b) as i32,
            (heads) as i32,
            (t) as i32,
            (vd) as i32,
        ])?;
        let out = out.transpose_axes(&[0, 2, 1, 3])?.reshape(&[
            (b) as i32,
            (t) as i32,
            (heads * vd) as i32,
        ])?;
        if let Some(c) = cap
            && c.wants(layer)
        {
            c.store(&format!("L{layer}.dsa_pre_o"), &out);
        }
        if t == 1
            && let Some(c2) = cap
            && c2.wants(layer)
        {
            let _ = out.eval_enqueue();
            c2.store(&format!("L{layer}.dec_dsa_pre_o"), &out);
        }
        Ok(out)
    }
}

/// Causal additive mask `[B,1,T,Tc]` f32 (0 visible, -1e30 hidden).
fn glm53_weighted_rms_norm(x: &Array, w: &Array, eps: f32) -> anyhow::Result<Array> {
    crate::bindings::glm53_indexer::weighted_rms(x, w, eps, false)
}
fn glm53_kv_lora_rms_norm(x: &Array, w: &Array, eps: f32) -> anyhow::Result<Array> {
    crate::bindings::glm53_indexer::weighted_rms(x, w, eps, true)
}

pub fn causal_mask(_rt_of: &Array, b: usize, t: usize, tc: usize) -> anyhow::Result<Array> {
    Ok(Array::from_slice(
        &lisa_models::glm53::indexer::causal_mask(b, t, tc)?,
        &[b as i32, 1, t as i32, tc as i32],
    )
    .as_dtype(Dtype::Float32)?)
}
pub fn keep_mask_from_selection(
    _rt_of: &Array,
    sel: &Array,
    t: usize,
    tc: usize,
) -> anyhow::Result<Array> {
    let b = sel.dim(0) as usize;
    let width = sel.dim(2) as usize;
    Ok(Array::from_slice(
        &lisa_models::glm53::indexer::keep_mask(&sel.to_vec1::<i32>()?, b, t, tc, width)?,
        &[b as i32, 1, t as i32, tc as i32],
    )
    .as_dtype(Dtype::Float32)?)
}
