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

use crate::bindings::shared::tensor_io::{CaptureTensor, pull_f32, pull_i32};
use crate::{bindings::BindingDevice, execution::DeviceContext};
use lisa_backend::{ManagedDevice, TensorAccess};

use crate::bindings::glm53::affine::projection::descriptor;
use crate::bindings::glm53::attention::ops::{one as plan_one, run as plan_run};
use crate::cache::{attention::FullAttentionCache, indexer::IndexerTape};
use crate::weights::affine::QuantizedLinear;
use crate::weights::source::{TensorSource, Weights};
fn narrow<D: BindingDevice + ManagedDevice>(
    context: &DeviceContext<D>,
    a: &D::Tensor,
    axis: usize,
    start: usize,
    len: usize,
) -> anyhow::Result<D::Tensor> {
    crate::bindings::glm53::affine::tower::ops::one(
        context,
        lisa_models::glm53::affine_tower::Request::Slice {
            axis: i32::try_from(axis)?,
            start: i32::try_from(start)?,
            end: i32::try_from(
                start
                    .checked_add(len)
                    .ok_or_else(|| anyhow::anyhow!("slice overflow"))?,
            )?,
        },
        &[a],
    )
}

use lisa_models::glm53::attention::{self as model, Request as R, Shape};

pub use crate::bindings::glm53::attention::indexer::Indexer;

pub struct Dsa<D: BindingDevice + ManagedDevice> {
    pub context: DeviceContext<D>,
    pub q_a: QuantizedLinear<D::Tensor>,
    pub q_b: QuantizedLinear<D::Tensor>,
    pub kv_a: QuantizedLinear<D::Tensor>,
    pub kv_b: QuantizedLinear<D::Tensor>,
    pub o_proj: QuantizedLinear<D::Tensor>,
    pub q_a_norm: D::Tensor,  // F32 [1536]
    pub kv_a_norm: D::Tensor, // F32 [512]
    pub indexer: Indexer<D>,
    pub heads: usize,
    pub q_head_dim: usize,
    pub v_head_dim: usize,
    pub norm_eps: f32,
}

pub struct DsaProjections<T> {
    pub latent: Option<T>,
    pub compact_written: Option<(usize, usize)>,
    pub query_rank: T,
    pub query: T,
    pub indexer_key: T,
    pub indexer_gate: T,
}

impl<D: BindingDevice + ManagedDevice> Dsa<D> {
    pub fn shape(&self, b: usize, t: usize, tc: usize) -> Shape {
        Shape {
            batch: b,
            tokens: t,
            heads: self.heads,
            query: self.q_head_dim,
            value: self.v_head_dim,
            cache: tc,
        }
    }

    pub fn load_on<S: TensorSource<D::Tensor>>(
        context: &DeviceContext<D>,
        w: &mut Weights<D::Tensor>,
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
        let take = |w: &mut Weights<D::Tensor>, name: &str| -> anyhow::Result<D::Tensor> {
            w.remove(name)
                .ok_or_else(|| anyhow::anyhow!("glm53 dsa: {name} missing"))
        };
        let p = |s: &str| format!("{prefix}.{s}");
        let mut q_a = QuantizedLinear::load(w, &p(model::PROJECTIONS[0]), "")?;
        q_a.set_quant(model::QUANTIZATION.0, model::QUANTIZATION.1);
        q_a.scales = plan_one(context, R::CastF32, &[&q_a.scales])?;
        q_a.biases = plan_one(context, R::CastF32, &[&q_a.biases])?;
        let mut q_b = QuantizedLinear::load(w, &p(model::PROJECTIONS[1]), "")?;
        q_b.set_quant(model::QUANTIZATION.0, model::QUANTIZATION.1);
        q_b.scales = plan_one(context, R::CastF32, &[&q_b.scales])?;
        q_b.biases = plan_one(context, R::CastF32, &[&q_b.biases])?;
        let mut kv_a = QuantizedLinear::load(w, &p(model::PROJECTIONS[2]), "")?;
        kv_a.set_quant(model::QUANTIZATION.0, model::QUANTIZATION.1);
        kv_a.scales = plan_one(context, R::CastF32, &[&kv_a.scales])?;
        kv_a.biases = plan_one(context, R::CastF32, &[&kv_a.biases])?;
        let mut kv_b = QuantizedLinear::load(w, &p(model::PROJECTIONS[3]), "")?;
        kv_b.set_quant(model::QUANTIZATION.0, model::QUANTIZATION.1);
        kv_b.scales = plan_one(context, R::CastF32, &[&kv_b.scales])?;
        kv_b.biases = plan_one(context, R::CastF32, &[&kv_b.biases])?;
        let mut o_proj = QuantizedLinear::load(w, &p(model::PROJECTIONS[4]), "")?;
        o_proj.set_quant(model::QUANTIZATION.0, model::QUANTIZATION.1);
        o_proj.scales = plan_one(context, R::CastF32, &[&o_proj.scales])?;
        o_proj.biases = plan_one(context, R::CastF32, &[&o_proj.biases])?;
        let q_a_norm = plan_one(context, R::CastF32, &[&take(w, &p(model::NORMS[0]))?])?;
        let kv_a_norm = plan_one(context, R::CastF32, &[&take(w, &p(model::NORMS[1]))?])?;

        let indexer = Indexer {
            context: context.clone(),
            wq_b_w: take(w, &p(model::INDEXER_WEIGHTS[0]))?,
            wk_w: take(w, &p(model::INDEXER_WEIGHTS[1]))?,
            k_norm_w: plan_one(
                context,
                R::CastF32,
                &[&take(w, &p(model::INDEXER_WEIGHTS[2]))?],
            )?,
            k_norm_b: plan_one(
                context,
                R::CastF32,
                &[&take(w, &p(model::INDEXER_WEIGHTS[3]))?],
            )?,
            weights_w: take(w, &p(model::INDEXER_WEIGHTS[4]))?,
            gate_w: take(w, &p(model::INDEXER_WEIGHTS[5]))?,
            ape: take(w, &p(model::INDEXER_WEIGHTS[6]))?,
            heads: index_heads,
            head_dim: index_head_dim,
            topk: index_topk,
            kpool: index_kpool,
            always_tail,
        };
        Ok(Self {
            context: context.clone(),
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
        x: &D::Tensor,
        tape: &mut IndexerTape<D>,
        position: usize,
        min_position: usize,
    ) -> anyhow::Result<D::Tensor> {
        anyhow::ensure!(
            x.dim(0) == 1 && x.dim(1) == 1,
            "glm53 MTP DSA expects one token, got {:?}",
            x.shape()
        );
        anyhow::ensure!(min_position <= position, "glm53 MTP window is inverted");

        let qd = self.q_head_dim;
        let scale = model::scale(qd);
        // MTP uses the same reference projection dispatcher as the trunk. Preserve
        // the Q-A -> Q norm -> Q-B -> KV-A order and the native one-row Q8
        // reduction before converting the private compact cache to F16.
        let native = crate::bindings::glm53::policy::glm53_q8_native_prefill_enabled();
        let qa = descriptor(&self.context, &self.q_a, x, native)?;
        let kv = descriptor(&self.context, &self.kv_a, x, native)?;
        let mut qb = descriptor(&self.context, &self.q_b, x, native)?;
        qb.shape = vec![1, 1, self.q_a.weight.dim(0)];
        let mut projections = plan_run(
            &self.context,
            R::MtpInputs {
                qa,
                qb,
                kv,
                shape: self.shape(1, 1, 0),
                epsilon: self.norm_eps.to_bits(),
            },
            &[
                x,
                &self.q_a.weight,
                &self.q_a.scales,
                &self.q_a.biases,
                &self.q_b.weight,
                &self.q_b.scales,
                &self.q_b.biases,
                &self.kv_a.weight,
                &self.kv_a.scales,
                &self.kv_a.biases,
                &self.q_a_norm,
                &self.kv_a_norm,
            ],
        )?
        .into_iter();
        let _q_a_raw = projections.next().unwrap();
        let _qr = projections.next().unwrap();
        let q = projections.next().unwrap();
        let _kv_raw = projections.next().unwrap();
        let c = projections.next().unwrap();
        tape.write_compact_f16_at(&c, position)
            .map_err(|e| anyhow::anyhow!("glm53 MTP compact cache: {e}"))?;

        let _is_window_fixture = false;
        let compact_start = if _is_window_fixture { min_position } else { 0 };
        let compact = tape
            .compact_slice(compact_start, position + 1)
            .ok_or_else(|| anyhow::anyhow!("glm53 MTP compact cache missing at {position}"))?;
        let width = position - min_position + 1;
        let selected_start = if _is_window_fixture { 0 } else { min_position };
        let selected: Vec<i32> = (selected_start..selected_start + width)
            .map(|p| p as i32)
            .collect();
        let selected = pull_i32(&self.context, &selected, &[1, 1, width as i32])?;
        let stream = &self.context;
        let _attended = crate::bindings::glm53::attention::absorbed::dsa_absorbed_attention::<D>(
            &q,
            &self.kv_b.weight,
            &self.kv_b.scales,
            &self.kv_b.biases,
            self.kv_b.group_size as usize,
            &compact,
            &selected,
            scale,
            stream,
        )
        .ok_or_else(|| {
            anyhow::anyhow!("glm53 MTP indexed absorbed attention declined at position {position}")
        })?;
        let out = plan_one(&self.context, R::Output(self.shape(1, 1, 0)), &[&_attended])?;
        let projected = self.output_project(&out)?;
        Ok(projected)
    }

    pub fn output_project(&self, x: &D::Tensor) -> anyhow::Result<D::Tensor> {
        let dtype = self
            .context
            .device()
            .dtype(x)
            .ok_or_else(|| anyhow::anyhow!("unsupported attention input dtype"))?;
        if model::custom_output(
            x.shape(),
            dtype,
            crate::bindings::glm53::policy::glm53_q8_native_prefill_enabled(),
        ) {
            plan_one(
                &self.context,
                R::OutputProjection {
                    shape: x.shape().to_vec(),
                    group: self.o_proj.group_size as usize,
                },
                &[
                    x,
                    &self.o_proj.weight,
                    &self.o_proj.scales,
                    &self.o_proj.biases,
                ],
            )
        } else {
            Ok(crate::bindings::glm53::affine::projection::forward(
                &self.context,
                &self.o_proj,
                x,
                crate::bindings::glm53::policy::glm53_q8_mpp_prefill_override() != Some(false),
            )?)
        }
    }

    pub fn project_inputs(
        &self,
        x: &D::Tensor,
        cap: Option<&crate::bindings::shared::capture::Capture<D::Tensor>>,
        layer: usize,
        compact_tape: Option<&mut crate::cache::indexer::IndexerTape<D>>,
    ) -> anyhow::Result<DsaProjections<D::Tensor>> {
        let (b, t) = (x.dim(0) as usize, x.dim(1) as usize);
        if let Some(capture) = cap.filter(|capture| capture.wants(layer)) {
            capture.store(&format!("L{layer}.dsa_indexer_input_x"), x);
        }

        let capture = cap.filter(|capture| capture.wants(layer));
        let native = crate::bindings::glm53::policy::glm53_q8_native_prefill_enabled();
        let mut raw = plan_run(
            &self.context,
            R::Inputs {
                latent: descriptor(&self.context, &self.kv_a, x, native)?,
                query: descriptor(&self.context, &self.q_a, x, native)?,
            },
            &[
                x,
                &self.kv_a.weight,
                &self.kv_a.scales,
                &self.kv_a.biases,
                &self.q_a.weight,
                &self.q_a.scales,
                &self.q_a.biases,
            ],
        )?
        .into_iter();
        let c_raw = raw.next().unwrap();
        let query_rank_raw = raw.next().unwrap();
        let capture_production_qkv = capture.is_some_and(|capture| capture.preserves_dispatch());
        let (c, qr, compact_written) = crate::bindings::glm53::attention::ops::normalize(
            &self.context,
            [&query_rank_raw, &self.q_a_norm, &c_raw, &self.kv_a_norm],
            b,
            t,
            self.norm_eps,
            capture.is_some(),
            capture_production_qkv,
            compact_tape,
            layer,
        )?;
        if let Some(capture) = capture {
            capture.store(&format!("L{layer}.dsa_kv_a_raw"), &c_raw);
            if let Some(c) = &c {
                capture.store(&format!("L{layer}.dsa_lat"), c);
            }
            capture.store(&format!("L{layer}.dsa_qr"), &qr);
        }
        let q = crate::bindings::glm53::affine::projection::forward(
            &self.context,
            &self.q_b,
            &qr,
            crate::bindings::glm53::policy::glm53_q8_mpp_prefill_override() != Some(false),
        )?; // [B,T,64*256]
        if let Some(c) = cap
            && c.wants(layer)
        {
            c.store(&format!("L{layer}.dsa_q"), &q);
        }
        let q = plan_one(&self.context, R::Query(self.shape(b, t, 0)), &[&q])?;

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
            indexer_key: ik,
            indexer_gate: gs,
        })
    }

    /// Forward `x [B,T,H]`. The indexer tape stores F32 key/gate features,
    /// while compact normalized KV latents are held in a separate F16 buffer.
    pub fn forward(
        &self,
        x: &D::Tensor,
        cache: &mut FullAttentionCache<D>,
        cap: Option<&crate::bindings::shared::capture::Capture<D::Tensor>>,
        layer: usize,
        prefill_mode: bool,
    ) -> anyhow::Result<D::Tensor> {
        let projected = {
            let _trace = self
                .context
                .trace_detail("glm53.dsa.input_projections", layer as u64);
            self.project_inputs(x, cap, layer, Some(&mut cache.indexer_tape))?
        };
        let pre_o = self.forward_projected(x, cache, cap, layer, &projected, None, prefill_mode)?;
        let out = {
            let _trace = self
                .context
                .trace_detail("glm53.dsa.output_projection", layer as u64);
            self.output_project(&pre_o)?
        };
        if let Some(capture) = cap.filter(|capture| capture.wants(layer)) {
            capture.store(&format!("L{layer}.dsa_output_project"), &out);
        }
        Ok(out)
    }

    /// Native mixed-position batch path: batch all DSA input projections and
    /// the output projection, but append/attend against each row's own tape.
    pub fn forward_ragged(
        &self,
        x: &D::Tensor,
        caches: &mut [&mut FullAttentionCache<D>],
        cap: Option<&crate::bindings::shared::capture::Capture<D::Tensor>>,
        layer: usize,
    ) -> anyhow::Result<D::Tensor> {
        let b = caches.len();
        anyhow::ensure!(
            b > 1 && x.dim(0) == b as i32 && x.dim(1) == 1,
            "glm53 DSA ragged decode expects B>1, one token per row, got {:?} for {b} caches",
            x.shape()
        );
        let projected = self.project_inputs(x, cap, layer, None)?;
        let mut pre_o_rows = Vec::with_capacity(b);
        for (row, cache) in caches.iter_mut().enumerate() {
            let x_row = narrow(&self.context, x, 0, row, 1)?;
            let row_projected = DsaProjections {
                latent: Some(narrow(
                    &self.context,
                    projected
                        .latent
                        .as_ref()
                        .expect("ragged projection retains compact latent"),
                    0,
                    row,
                    1,
                )?),
                compact_written: None,
                query_rank: narrow(&self.context, &projected.query_rank, 0, row, 1)?,
                query: narrow(&self.context, &projected.query, 0, row, 1)?,
                indexer_key: narrow(&self.context, &projected.indexer_key, 0, row, 1)?,
                indexer_gate: narrow(&self.context, &projected.indexer_gate, 0, row, 1)?,
            };
            let pre_o = self.forward_projected(
                &x_row,
                cache,
                cap,
                layer,
                &row_projected,
                Some(row),
                false,
            )?;
            pre_o_rows.push(pre_o);
        }
        let pre_o_refs: Vec<&D::Tensor> = pre_o_rows.iter().collect();
        let pre_o_batch = plan_one(
            &self.context,
            R::Join {
                count: pre_o_refs.len(),
                axis: 0,
            },
            &pre_o_refs,
        )?;
        let output = {
            let _trace = self
                .context
                .trace_detail("glm53.dsa.output_projection", layer as u64);
            self.output_project(&pre_o_batch)?
        };
        if let Some(capture) = cap.filter(|capture| capture.wants(layer)) {
            capture.store(&format!("L{layer}.dsa_output_project"), &output);
        }
        Ok(output)
    }

    pub fn append_projected_cache(
        &self,
        projected: &DsaProjections<D::Tensor>,
        cache: &mut FullAttentionCache<D>,
        layer: usize,
        capture: Option<&crate::bindings::shared::capture::Capture<D::Tensor>>,
    ) -> anyhow::Result<()> {
        let _trace = self
            .context
            .trace_detail("glm53.dsa.cache_append", layer as u64);
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

    pub fn append_cache_projection_rows(
        &self,
        indexer_key: &D::Tensor,
        indexer_gate: &D::Tensor,
        latent: Option<&D::Tensor>,
        compact_written: Option<(usize, usize)>,
        cache: &mut FullAttentionCache<D>,
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

    pub fn select_projected_ids(
        &self,
        x: &D::Tensor,
        cache: &mut FullAttentionCache<D>,
        projected: &DsaProjections<D::Tensor>,
        capture: Option<&crate::bindings::shared::capture::Capture<D::Tensor>>,
        layer: usize,
        prefill_mode: bool,
    ) -> anyhow::Result<(D::Tensor, usize)> {
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

    pub fn forward_projected(
        &self,
        x: &D::Tensor,
        cache: &mut FullAttentionCache<D>,
        cap: Option<&crate::bindings::shared::capture::Capture<D::Tensor>>,
        layer: usize,
        projected: &DsaProjections<D::Tensor>,
        native_batch_row: Option<usize>,
        prefill_mode: bool,
    ) -> anyhow::Result<D::Tensor> {
        let (b, t) = (x.dim(0) as usize, x.dim(1) as usize);
        let qd = self.q_head_dim;
        let scale = model::scale(qd);

        let qr = &projected.query_rank;
        let q = &projected.query;

        self.append_projected_cache(projected, cache, layer, cap)?;
        let tc = cache.indexer_tape.total;
        let dense_limit = model::dense_limit(self.indexer.topk, self.indexer.kpool);
        let _value_slices = model::value_partitioned(
            self.shape(b, t, tc),
            prefill_mode,
            native_batch_row.is_some(),
        )
        .then(|| {
            crate::bindings::glm53::attention::mode::Glm53DsaValueSliceScope::new(
                tc - t,
                dense_limit,
            )
        });
        let compact_latents = cache
            .indexer_tape
            .compact_slice(0, tc)
            .ok_or_else(|| anyhow::anyhow!("glm53 dsa: compact latent cache missing"))?;
        if let Some(capture) = cap.filter(|capture| capture.wants(layer)) {
            let current = plan_one(
                &self.context,
                R::Contiguous,
                &[&narrow(&self.context, &compact_latents, 1, tc - t, t)?],
            )?;
            capture.store(&format!("L{layer}.dsa_cache_latent"), &current);
        }

        // The reference uses its raw FlashAttention kernel for dense-causal prefill
        // slices of at least 24 rows. Keep that path distinct from indexed
        // attention: its F16 query staging and tiled softmax are observable
        // in the layer output.
        if model::flash(
            self.shape(b, t, tc),
            dense_limit,
            native_batch_row.is_some(),
        ) {
            let query_start = tc - t;
            let stream = &self.context;
            let (_attended, q_low, lora) =
                crate::bindings::glm53::attention::absorbed::dsa_absorbed_attention_flash_causal_prefill_stages::<D>(
                    q,
                    &self.kv_b.weight,
                    &self.kv_b.scales,
                    &self.kv_b.biases,
                    self.kv_b.group_size as usize,
                    &compact_latents,
                    query_start,
                    scale,
                    stream,
                    prefill_mode,
                )
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "glm53 dsa: DS4 dense-causal FlashAttention declined at rows {t}, context {tc}"
                    )
                })?;
            let out = plan_one(&self.context, R::Output(self.shape(b, t, 0)), &[&_attended])?;
            if let Some(capture) = cap.filter(|capture| capture.wants(layer)) {
                capture.store(&format!("L{layer}.dsa_q_low"), &q_low);
                capture.store(&format!("L{layer}.dsa_attn_lora"), &lora);
                capture.store(&format!("L{layer}.dsa_pre_o"), &out);
                let tail = t.saturating_sub(4);
                if let Ok(q_tail) = narrow(&self.context, &q_low, 1, tail, t.min(4)) {
                    capture.store(&format!("L{layer}.dsa_q_low_boundary"), &q_tail);
                }
                if let Ok(lora_tail) = narrow(&self.context, &lora, 1, tail, t.min(4)) {
                    capture.store(&format!("L{layer}.dsa_attn_lora_boundary"), &lora_tail);
                }
            }
            return Ok(out);
        }

        // The reference uses indexed decode even below the sparse crossover: it fills
        // a contiguous [0, tc) selection there. Keep that explicit prefix path
        // so affine-MLX attention uses the same reduction schedule.
        let selected_absorbed = match model::selection(self.shape(b, t, tc), dense_limit) {
            model::Selection::Causal => {
                // Each verification row sees the committed prefix through its own
                // absolute position. Pad the shared-width selection with invalid
                // IDs so the indexed-lora kernel can process the rows together.
                let ids = model::causal_ids(self.shape(b, t, tc))?;
                Some((
                    pull_i32(&self.context, &ids, &[b as i32, t as i32, tc as i32])?,
                    0,
                ))
            }
            model::Selection::Prefix => Some({
                let ids: Vec<i32> = (0..tc as i32).collect();
                (
                    plan_one(
                        &self.context,
                        R::BroadcastIds(self.shape(b, t, tc)),
                        &[&pull_i32(&self.context, &ids, &[1, 1, tc as i32])?],
                    )?,
                    tc,
                )
            }),
            model::Selection::Indexed => Some({
                let _trace = self
                    .context
                    .trace_detail("glm53.dsa.indexer_selection", layer as u64);
                self.select_projected_ids(x, cache, projected, cap, layer, prefill_mode)?
            }),
            model::Selection::None => None,
        };
        if let Some((selected, selected_valid_prefix)) = selected_absorbed.as_ref() {
            let stream = &self.context;
            let _trace_absorbed = self
                .context
                .trace_detail("glm53.dsa.absorbed_attention", layer as u64);
            let absorbed = if let Some(capture) = cap.filter(|capture| capture.wants(layer)) {
                crate::bindings::glm53::attention::absorbed::dsa_absorbed_attention_stages::<D>(
                    q,
                    &self.kv_b.weight,
                    &self.kv_b.scales,
                    &self.kv_b.biases,
                    self.kv_b.group_size as usize,
                    &compact_latents,
                    selected,
                    scale,
                    stream,
                    prefill_mode,
                )
                .map(|(_attended, q_low, lora)| {
                    if let Some(capture) = cap.filter(|capture| capture.wants(layer)) {
                        capture.store(&format!("L{layer}.dsa_q_low"), &q_low);
                        capture.store(&format!("L{layer}.dsa_attn_lora"), &lora);
                    }
                    let query_rows = q_low.dim(1) as usize;
                    let tail = query_rows.saturating_sub(4);
                    let tail_rows = query_rows.min(4);
                    if let Ok(q_tail) = narrow(&self.context, &q_low, 1, tail, tail_rows) {
                        capture.store(&format!("L{layer}.dsa_q_low_boundary"), &q_tail);
                    }
                    if let Ok(lora_tail) = narrow(&self.context, &lora, 1, tail, tail_rows) {
                        capture.store(&format!("L{layer}.dsa_attn_lora_boundary"), &lora_tail);
                    }
                    _attended
                })
            } else {
                // The reference keeps MLX affine V-B on the split attention-lora plus
                // value-projection path; its fused indexed decode is only
                // used for native value-weight representations.
                crate::bindings::glm53::attention::absorbed::dsa_absorbed_attention_with_valid_prefix::<
                    D,
                >(
                    q,
                    &self.kv_b.weight,
                    &self.kv_b.scales,
                    &self.kv_b.biases,
                    self.kv_b.group_size as usize,
                    &compact_latents,
                    selected,
                    *selected_valid_prefix,
                    scale,
                    stream,
                    prefill_mode,
                )
            };
            if let Some(_attended) = absorbed {
                let out = plan_one(&self.context, R::Output(self.shape(b, t, 0)), &[&_attended])?;
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
                model::dense_fallback_allowed(self.shape(b, t, tc), dense_limit),
                "glm53 dsa: indexed absorbed attention declined at context {tc} (selection width {}); refusing the quadratic dense fallback",
                selected.dim(2)
            );
        }

        let keep = match model::mask(
            self.shape(b, t, tc),
            dense_limit,
            false,
            selected_absorbed.is_some(),
        ) {
            model::Mask::GpuKeep => {
                let _trace = self
                    .context
                    .trace_detail("glm53.dsa.indexer_keep_mask", layer as u64);
                self.indexer.select_keep_gpu(
                    &mut cache.indexer_tape,
                    x,
                    qr,
                    cache.offset,
                    t,
                    tc,
                    layer,
                )?
            }
            model::Mask::Selected => keep_mask_from_selection(
                &self.context,
                x,
                &selected_absorbed.as_ref().unwrap().0,
                t,
                tc,
            )?,
            model::Mask::Causal => causal_mask(&self.context, x, b, t, tc)?,
            model::Mask::HostSelection => {
                let sel = {
                    let _trace = self
                        .context
                        .trace_detail("glm53.dsa.indexer_host_selection", layer as u64);
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
                keep_mask_from_selection(&self.context, x, &sel, t, tc)?
            }
        };

        // Prefill and unsupported decode hardware retain the dense reference
        // path. It is intentionally outside the common one-token decode path.
        let kv = {
            let _trace = self
                .context
                .trace_detail("glm53.dsa.dense_kv_projection", layer as u64);
            crate::bindings::shared::affine::linear(
                &self.context,
                &plan_one(&self.context, R::CastF32, &[&compact_latents])?,
                &self.kv_b,
            )?
        };
        let mut split =
            plan_run(&self.context, R::SplitKv(self.shape(b, t, tc)), &[&kv])?.into_iter();
        let k = split.next().unwrap();
        let v = split.next().unwrap();
        if let Some(c) = cap
            && c.wants(layer)
        {
            c.store(&format!("L{layer}.dsa_k"), &k);
            c.store(&format!("L{layer}.dsa_v"), &v);
        }

        if t == 1 {
            let step =
                crate::bindings::glm53::policy::INC_STEP.load(std::sync::atomic::Ordering::Relaxed);
            if let Some(c2) = cap
                && c2.wants(layer)
            {
                let _ = self.context.device().evaluate(&k, false);
                let _ = self.context.device().evaluate(&v, false);
                c2.store(&format!("L{layer}.dec_dsa_k_s{step}"), &k);
                c2.store(&format!("L{layer}.dec_dsa_v_s{step}"), &v);
            }
        }
        // scores = q @ k^T * scale + keep, softmax, @ v — per batch (B=1 v1).
        // The reference flash_attn decode parity (t == 1): the attention runs as TWO
        // fused dispatches (scores, then softmax @ v) over the full cached
        // K/V with the keep mask baked in. Both stages preserve the reference F32
        // graph; the op path remains the fallback when the shape declines.
        let mut out_acc: Option<D::Tensor> = None;
        let fused = if model::fused_dense(self.shape(b, t, tc), false) {
            let _trace = self
                .context
                .trace_detail("glm53.dsa.fused_decode_attention", layer as u64);
            plan_one(
                &self.context,
                R::DenseDecode {
                    shape: self.shape(b, t, tc),
                    scale: scale.to_bits(),
                },
                &[q, &k, &v, &keep],
            )
            .ok()
        } else {
            None
        };
        if let Some(flat) = fused {
            // [B*heads, vd] == the per-batch concat the op path builds at t == 1.
            out_acc = Some(flat);
        } else {
            let kt = plan_one(&self.context, R::DenseTranspose, &[&k])?;
            for bi in 0..b {
                let mut operands = plan_run(
                    &self.context,
                    R::DenseOperands {
                        shape: self.shape(b, t, tc),
                        row: bi,
                    },
                    &[q, &kt, &v, &keep],
                )?
                .into_iter();
                let qb = operands.next().unwrap();
                let kb_t = operands.next().unwrap();
                let vb = operands.next().unwrap();
                let keep_b = operands.next().unwrap();
                if let Some(cp) = cap
                    && cp.wants(layer)
                {
                    cp.store(&format!("L{layer}.dsa_qb"), &qb);
                    cp.store(&format!("L{layer}.dsa_kb_t"), &kb_t);
                }
                let mut scores = plan_run(
                    &self.context,
                    R::DenseScores {
                        shape: self.shape(b, t, tc),
                        scale: scale.to_bits(),
                    },
                    &[&qb, &kb_t, &keep_b],
                )?
                .into_iter();
                let sc = scores.next().unwrap();
                let probs_b = scores.next().unwrap();
                if let Some(cp) = cap
                    && cp.wants(layer)
                {
                    cp.store(&format!("L{layer}.dsa_sc"), &sc);
                    cp.store(&format!("L{layer}.dsa_probs"), &probs_b);
                    cp.store(&format!("L{layer}.dsa_vb"), &vb);
                }
                let ob = plan_one(
                    &self.context,
                    R::DenseValue {
                        shape: self.shape(b, t, tc),
                    },
                    &[&probs_b, &vb],
                )?;
                out_acc = Some(match out_acc {
                    Some(a) => plan_one(&self.context, R::Join { count: 2, axis: 0 }, &[&a, &ob])?,
                    None => ob,
                });
            }
        }
        let out = plan_one(
            &self.context,
            R::FinalFallback(self.shape(b, t, tc)),
            &[&out_acc.expect("b>0")],
        )?;
        if let Some(c) = cap
            && c.wants(layer)
        {
            c.store(&format!("L{layer}.dsa_pre_o"), &out);
        }
        if t == 1
            && let Some(c2) = cap
            && c2.wants(layer)
        {
            let _ = self.context.device().evaluate(&out, false);
            c2.store(&format!("L{layer}.dec_dsa_pre_o"), &out);
        }
        Ok(out)
    }
}

/// Causal additive mask `[B,1,T,Tc]` f32 (0 visible, -1e30 hidden).
pub fn causal_mask<D: BindingDevice + ManagedDevice>(
    context: &DeviceContext<D>,
    _rt_of: &D::Tensor,
    b: usize,
    t: usize,
    tc: usize,
) -> anyhow::Result<D::Tensor> {
    plan_one(
        context,
        R::CastF32,
        &[&pull_f32(
            context,
            &lisa_models::glm53::indexer::causal_mask(b, t, tc)?,
            &[b as i32, 1, t as i32, tc as i32],
        )?],
    )
}
pub fn keep_mask_from_selection<D: BindingDevice + ManagedDevice>(
    context: &DeviceContext<D>,
    _rt_of: &D::Tensor,
    sel: &D::Tensor,
    t: usize,
    tc: usize,
) -> anyhow::Result<D::Tensor> {
    let b = sel.dim(0) as usize;
    let width = sel.dim(2) as usize;
    plan_one(
        context,
        R::CastF32,
        &[&pull_f32(
            context,
            &lisa_models::glm53::indexer::keep_mask(&sel.read_i32(context)?, b, t, tc, width)?,
            &[b as i32, 1, t as i32, tc as i32],
        )?],
    )
}

pub mod absorbed;
pub mod indexer;
pub mod mode;
pub mod ops;
