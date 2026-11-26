//! The Qwen 3.8 Flash-Next tower (`qwen4_exp_text`): embeddings, 48 hybrid
//! decoder layers, the final hyper-connection mixer, and the lm head.

use crate::batching::BatchRuntime;
use crate::{
    bindings::BindingDevice, execution::DeviceContext, language::runtime::Runtime,
    sampling::sampler::Sampler,
};
use lisa_backend::{HostTensor, ManagedDevice, ReadbackDevice, ScalarType, TensorAccess};

use crate::core::graph::PlanCache;

use lisa_models::qwen4::tower::{self as model, Step};

use crate::bindings::qwen4::hyper::GatedResidual;
use crate::bindings::shared::normalization::Rotary;
use crate::cache::{LayerCache, attention::FullAttentionCache, recurrent::GdnCache};
use crate::weights::affine::{QuantizedEmbedding, QuantizedLinear};
use lisa_models::qwen4::config::ModelConfig;

pub use crate::bindings::qwen4::decoder::{Block, DecoderLayer};

/// The full text tower.
pub struct Tower<D: BindingDevice + ManagedDevice + ReadbackDevice> {
    pub context: DeviceContext<D>,
    pub(crate) runtime: Runtime<D, Sampler<D>>,
    pub(crate) vision: Option<crate::vision::Vision<D>>,
    pub embed_tokens: QuantizedEmbedding<D::Tensor>,
    pub layers: Vec<DecoderLayer<D>>,
    pub final_mixer: GatedResidual<D>,
    pub lm_head: QuantizedLinear<D::Tensor>,
    pub rope: Rotary,
    pub config: ModelConfig,
    /// The 2-token n-gram history per row (host side; eos-filled initially).
    pub ngram_history: Option<Vec<Vec<i64>>>,
    pub eos_token_id: i64,
    /// The embedded MTP head (loaded, driven by the speculative path).
    pub mtp: Option<crate::bindings::qwen4::mtp::MtpHead<D>>,
    pub(crate) warmed: bool,
}

impl<D: BindingDevice + ManagedDevice + ReadbackDevice> Tower<D>
where
    D::Error: From<anyhow::Error>,
{
    pub fn token_rows(ids: &D::Tensor) -> anyhow::Result<Vec<Vec<i64>>> {
        model::token_rows(
            &ids.read_raw_ids()?
                .into_iter()
                .map(|x| x as i32)
                .collect::<Vec<_>>(),
            ids.dim(0) as usize,
            ids.dim(1) as usize,
        )
    }

    /// Forward the tower. Returns `(mixed, multi)` where `mixed` is the
    /// collapsed hidden state [B, S, H] and `multi` is the pre-final-mixer
    /// hyper stream [B, S, hc*H] (consumed by the MTP head later).
    pub fn forward(
        &mut self,
        ids: &D::Tensor,
        caches: Option<&mut Vec<LayerCache<D>>>,
    ) -> anyhow::Result<(D::Tensor, D::Tensor)> {
        self.forward_capture(ids, caches, false)
    }

    /// Long-context prefill window. A single forward over a `S`-token prompt
    /// allocates MoE activation buffers of `[S*top_k, hidden]` and, once the
    /// QSA tape passes its budget, `O(S^2)` attention masks. Feeding the prompt
    /// in windows of this size keeps every kernel on the small/prefill shapes
    /// it already handles and bounds the activation peak. Windows are causally
    /// equivalent to the full prefill (the cache carries the prefix), which
    /// `session::verify_incremental` pins. Prompts at or below this size take
    /// the original single-forward path (so the golden is untouched).
    pub const PREFILL_CHUNK: usize = model::PREFILL_CHUNK;

    /// Chunked prefill: forward `ids` through `caches` in windows and return
    /// the hidden state at the last position.
    pub fn prefill(
        &mut self,
        ids: &[u32],
        caches: &mut Vec<LayerCache<D>>,
    ) -> anyhow::Result<D::Tensor> {
        anyhow::ensure!(!ids.is_empty(), "empty prefill");
        let chunk = Self::PREFILL_CHUNK;
        if ids.len() <= chunk {
            let arr = i32_arr(&self.context, ids)?;
            let (mixed, _) = self.forward(&arr, Some(caches))?;
            return last_row(&self.context, &mixed);
        }
        let mut last = None;
        for c in ids.chunks(chunk) {
            let arr = i32_arr(&self.context, c)?;
            let (mixed, _) = self.forward(&arr, Some(caches))?;
            last = Some(mixed);
            self.context.device().trim_unused();
        }
        last_row(&self.context, &last.expect("non-empty chunks"))
    }

    /// Chunked prefill returning the hyper stream `multi` for every fed token
    /// (needed to prime the MTP head over a long suffix).
    pub fn prefill_multi(
        &mut self,
        ids: &[u32],
        caches: &mut Vec<LayerCache<D>>,
    ) -> anyhow::Result<(D::Tensor, D::Tensor)> {
        anyhow::ensure!(!ids.is_empty(), "empty prefill");
        let chunk = Self::PREFILL_CHUNK;
        let mut multis: Vec<D::Tensor> = Vec::new();
        let mut last_mixed: Option<D::Tensor> = None;
        for c in ids.chunks(chunk) {
            let arr = i32_arr(&self.context, c)?;
            let (mixed, multi) = self.forward(&arr, Some(caches))?;
            last_mixed = Some(mixed);
            multis.push(multi);
            self.context.device().trim_unused();
        }
        let mixed = last_mixed.expect("non-empty chunks");
        let multi = if multis.len() == 1 {
            multis.pop().expect("one multi")
        } else {
            let refs: Vec<&D::Tensor> = multis.iter().collect();
            execute(&self.context, View::Join(refs.len()), &refs, || {
                Ok(lisa_models::traversal::concatenate(refs.len(), 1))
            })?
            .into_one()
            .map_err(anyhow::Error::msg)?
        };
        Ok((last_row(&self.context, &mixed)?, multi))
    }

    /// Forward with an explicit speculative-capture flag (see `GdnCache`).
    pub fn forward_capture(
        &mut self,
        ids: &D::Tensor,
        mut caches: Option<&mut Vec<LayerCache<D>>>,
        capture: bool,
    ) -> anyhow::Result<(D::Tensor, D::Tensor)> {
        let _trace_fwd = self.context.trace("tower.forward");
        let b = ids.dim(0) as usize;
        let s = ids.dim(1) as usize;
        let token_rows = Self::token_rows(ids)?;

        // The n-gram hash reads ids in order; the history carries across calls.
        let (previous_context, new_history) = model::history(
            &token_rows,
            self.ngram_history.as_deref(),
            self.config.ngram_size,
            self.eos_token_id,
        )?;
        self.ngram_history = Some(new_history);

        let offset = caches
            .as_deref()
            .and_then(|c| {
                c.iter().find_map(|lc| match lc {
                    LayerCache::Full(f) => Some(f.offset),
                    _ => None,
                })
            })
            .unwrap_or(0);
        // RoPE positions. Ragged batching carries a per-stream next position on
        // the packed attention caches; otherwise every stream shares `offset`.
        let positions: D::Tensor = match caches.as_deref().and_then(|c| {
            c.iter().find_map(|lc| match lc {
                LayerCache::Full(f) => f.next_pos.clone(),
                _ => None,
            })
        }) {
            Some(next) => {
                let (data, shape) = lisa_models::position::indices(offset, s, Some(&next))?;
                self.context.device().pull(
                    HostTensor {
                        bytes: &data
                            .iter()
                            .flat_map(|x| x.to_ne_bytes())
                            .collect::<Vec<_>>(),
                        shape: &shape,
                        dtype: ScalarType::F32,
                    }
                    .into(),
                    Default::default(),
                )?
            }
            None => crate::bindings::shared::normalization::positions(&self.context, offset, s)?,
        };

        let positions = self
            .vision
            .as_ref()
            .map(|v| v.positions(offset, s))
            .transpose()?
            .flatten()
            .unwrap_or(positions);
        let mut residual = None;
        let mut pending_out = None;
        let mut pending_inject = None;
        let mut finished = None;
        for step in model::traversal(self.layers.len(), s) {
            match step {
                Step::Begin => {
                    let splice = self.vision.as_ref().and_then(|v| {
                        v.splice_bindings(
                            vec![b as i32, s as i32, self.config.hidden_size as i32],
                            lisa_backend::ScalarType::BF16,
                            offset,
                        )
                    });
                    let spec = model::Entry {
                        embedding: (self.embed_tokens.group_size, self.embed_tokens.bits),
                        streams: self.config.hc_count as i32,
                        splice: splice.as_ref().map(|(s, _)| s.clone()),
                    };
                    let mut inputs = vec![
                        ids,
                        &self.embed_tokens.weight,
                        &self.embed_tokens.scales,
                        &self.embed_tokens.biases,
                    ];
                    if let Some((_, features)) = &splice {
                        inputs.extend(features);
                    }
                    residual = Some(
                        execute(&self.context, View::Entry(spec.clone()), &inputs, || {
                            spec.graph()
                        })?
                        .into_one()
                        .map_err(anyhow::Error::msg)?,
                    );
                }
                Step::Layer(i) => {
                    let _trace_layer = self.context.trace_detail("tower.layer", i as u64);
                    let layer = &mut self.layers[i];
                    let cache = caches.as_deref_mut().map(|c| &mut c[i]);
                    let ple_args = layer
                        .has_ple
                        .then_some((&token_rows[..], &previous_context[..]));
                    let (st, out, inject) = layer.forward_with_capture_law(
                        residual.as_ref().unwrap(),
                        pending_out.as_ref(),
                        pending_inject.as_ref(),
                        cache,
                        &self.rope,
                        offset,
                        &positions,
                        ple_args,
                        capture,
                        capture,
                    )?;
                    residual = Some(st);
                    pending_out = Some(out);
                    pending_inject = Some(inject);
                }
                Step::Submit => {
                    let mut outputs = vec![residual.as_ref().unwrap()];
                    outputs.extend(pending_out.as_ref());
                    outputs.extend(pending_inject.as_ref());
                    // Eager providers already encoded these outputs; retain the no-wait boundary.
                    drop(outputs);
                }
                Step::Finish => {
                    finished = Some(if capture {
                        self.final_mixer.finish_captured(
                            residual.as_ref().unwrap(),
                            pending_out.as_ref(),
                            pending_inject.as_ref(),
                            self.config.rms_norm_eps,
                        )?
                    } else {
                        self.final_mixer.finish(
                            residual.as_ref().unwrap(),
                            pending_out.as_ref(),
                            pending_inject.as_ref(),
                            self.config.rms_norm_eps,
                        )?
                    });
                }
            }
        }
        finished.ok_or_else(|| anyhow::anyhow!("tower plan has no result"))
    }

    /// Logits from the mixed hidden state: [B, S, vocab].
    pub fn head(&self, mixed: &D::Tensor) -> Result<D::Tensor, D::Error> {
        crate::bindings::shared::affine::linear(&self.context, mixed, &self.lm_head)
            .map_err(Into::into)
    }
    pub fn head_verified(&self, mixed: &D::Tensor) -> Result<D::Tensor, D::Error> {
        crate::bindings::shared::affine::linear_serial(&self.context, mixed, &self.lm_head, false)
            .map_err(Into::into)
    }

    /// JIT-compile and warm every kernel/MoE path (both the narrow decode paths
    /// and the wide prefill paths) so the first measured forward does not pay
    /// Metal kernel compilation.
    ///
    /// The warm inputs are drawn from `seed` (the real prompt) rather than
    /// synthetic zeros: an all-zero activation exercises a different branch and
    /// must not be used here.
    pub fn warmup(&mut self, seed: &[u32]) -> anyhow::Result<()> {
        if self.warmed {
            self.ngram_history = None;
            return Ok(());
        }
        for ids in model::warmup(seed) {
            let arr = self.runtime.tokens(&ids, &[1, ids.len() as i32])?;
            let mut caches = self.new_caches();
            let (mixed, multi) = self.forward(&arr, Some(&mut caches))?;
            let last = last_row(&self.context, &mixed)?;
            let logits = self.head(&last)?;
            logits.eval()?;
            multi.eval()?;
        }
        self.ngram_history = None;
        // Release the warmup buffers before the worker serves: keeps unified
        // memory free for the real KV/activations. Compiled kernels are
        // unaffected.
        self.context.device().clear_unused();
        self.warmed = true;
        Ok(())
    }

    /// Fresh caches for one request.
    pub fn new_caches(&self) -> Vec<LayerCache<D>> {
        model::caches(&self.config)
            .into_iter()
            .map(|c| match c {
                model::Cache::Full { heads, width } => {
                    LayerCache::Full(FullAttentionCache::new(self.context.clone(), heads, width))
                }
                model::Cache::Linear => LayerCache::Linear(GdnCache::new()),
            })
            .collect()
    }
}

/// `[1, len]` token ids.
fn i32_arr<D: BindingDevice>(context: &DeviceContext<D>, ids: &[u32]) -> anyhow::Result<D::Tensor> {
    let v: Vec<i32> = ids.iter().map(|&t| t as i32).collect();
    Ok(context.device().pull(
        HostTensor {
            bytes: &v.iter().flat_map(|x| x.to_ne_bytes()).collect::<Vec<_>>(),
            shape: &[1, ids.len() as i32],
            dtype: ScalarType::I32,
        }
        .into(),
        Default::default(),
    )?)
}

/// The last sequence position of a `[B, S, H]` hidden state.
fn last_row<D: BindingDevice>(
    context: &DeviceContext<D>,
    x: &D::Tensor,
) -> anyhow::Result<D::Tensor> {
    execute(context, View::Last(x.dim(1)), &[x], || {
        Ok(lisa_models::qwen3_5::tower::View::Last(x.dim(1)).graph())
    })?
    .into_one()
    .map_err(anyhow::Error::msg)
}
#[derive(PartialEq)]
enum View {
    Last(i32),
    Join(usize),
    Entry(model::Entry),
    Readout(model::Readout),
}
fn execute<D: BindingDevice>(
    context: &DeviceContext<D>,
    key: View,
    inputs: &[&D::Tensor],
    build: impl FnOnce() -> anyhow::Result<lisa_models::graph::Graph>,
) -> anyhow::Result<lisa_backend::Outputs<D::Tensor>> {
    context.with_slot(
        "qwen4.tower.views",
        || std::cell::RefCell::new(PlanCache::new(context.device().clone(), 64)),
        |p| {
            p.borrow_mut()
                .try_get_or_prepare(key, build)?
                .run(inputs, &mut [])
        },
    )
}

impl<D: BindingDevice + ManagedDevice + ReadbackDevice>
    crate::language::LanguageModel<Runtime<D, Sampler<D>>> for Tower<D>
where
    D::Error: From<anyhow::Error>,
{
    fn runtime(&self) -> &Runtime<D, Sampler<D>> {
        &self.runtime
    }
    fn cache_numerical_law(&self) -> Option<u64> {
        Some(lisa_models::qwen4::cache_law_with_norm_mode(
            &self.config,
            &self.context.device().identity().arithmetic,
            self.final_mixer.force_norm_operations,
        ))
    }
    fn supports_images(&self) -> bool {
        self.vision.is_some()
    }
    fn prepare_images(
        &mut self,
        tokens: &mut Vec<u32>,
        images: &[image::RgbImage],
    ) -> anyhow::Result<()> {
        let limit = self.max_position_embeddings();
        self.vision
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("checkpoint has no supported vision encoder"))?
            .prepare(tokens, images, limit)
    }
    fn clear_images(&mut self) {
        if let Some(v) = &mut self.vision {
            v.clear();
        }
    }

    fn prefix_cache_granularity(&self) -> Option<usize> {
        Some(Self::PREFILL_CHUNK)
    }
    fn max_position_embeddings(&self) -> usize {
        self.config.max_position_embeddings
    }
    fn eos_token_id(&self) -> i64 {
        self.eos_token_id
    }
    fn new_caches(&self) -> Vec<LayerCache<D>> {
        self.new_caches()
    }
    /// O3: full-attention layers × kv_heads × head_dim × (K+V) × bf16 (2 o).
    fn kv_bytes_per_token(&self) -> usize {
        model::kv_bytes(&self.config)
    }
    fn forward(
        &mut self,
        ids: &D::Tensor,
        caches: Option<&mut Vec<LayerCache<D>>>,
    ) -> anyhow::Result<(D::Tensor, D::Tensor)> {
        self.forward(ids, caches)
    }
    fn forward_capture(
        &mut self,
        ids: &D::Tensor,
        caches: Option<&mut Vec<LayerCache<D>>>,
        capture: bool,
    ) -> anyhow::Result<(D::Tensor, D::Tensor)> {
        self.forward_capture(ids, caches, capture)
    }
    fn prefill(
        &mut self,
        tokens: &[u32],
        caches: &mut Vec<LayerCache<D>>,
    ) -> anyhow::Result<D::Tensor> {
        self.prefill(tokens, caches)
    }
    fn prefill_multi(
        &mut self,
        tokens: &[u32],
        caches: &mut Vec<LayerCache<D>>,
    ) -> anyhow::Result<(D::Tensor, D::Tensor)> {
        self.prefill_multi(tokens, caches)
    }
    fn head(&self, mixed: &D::Tensor) -> Result<D::Tensor, D::Error> {
        self.head(mixed)
    }
    fn head_verified(&self, mixed: &D::Tensor) -> Result<D::Tensor, D::Error> {
        self.head_verified(mixed)
    }
    fn warmup(&mut self, seed: &[u32]) -> anyhow::Result<()> {
        self.warmup(seed)
    }

    // Host-side rolling n-gram context for the PLE layer.
    fn context_window(&self) -> usize {
        self.config.ngram_size.saturating_sub(1)
    }
    fn context_tails(&self) -> Vec<Vec<i64>> {
        self.ngram_history.clone().unwrap_or_default()
    }
    fn set_context_tails(&mut self, tails: Vec<Vec<i64>>) {
        self.ngram_history = if tails.is_empty() { None } else { Some(tails) };
    }
    fn clear_context(&mut self) {
        self.ngram_history = None;
    }

    // Speculative drafting via the embedded MTP head.
    fn has_drafter(&self) -> bool {
        self.mtp.is_some()
    }
    fn mtp_cost_key(&self) -> String {
        model::cost_key(&self.config)
    }
    fn drafter_reset(&mut self) {
        if let Some(h) = self.mtp.as_mut() {
            h.reset_caches();
        }
    }
    fn drafter_trim(&mut self, n: usize) {
        if let Some(h) = self.mtp.as_mut() {
            h.trim_caches(n);
        }
    }
    fn drafter_offset(&self) -> usize {
        self.mtp.as_ref().map(|h| h.cache_offset()).unwrap_or(0)
    }
    fn drafter_restore_offset(&mut self, n: usize) {
        if let Some(h) = self.mtp.as_mut() {
            h.restore_offset(n);
        }
    }
    fn draft_step(
        &mut self,
        tokens: &D::Tensor,
        multi: &D::Tensor,
    ) -> anyhow::Result<(D::Tensor, D::Tensor)> {
        let offset = self.mtp.as_ref().map(|h| h.cache_offset()).unwrap_or(0);
        let head = self
            .mtp
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("this model has no MTP head"))?;
        let (sample, head_multi) = head.forward(tokens, multi, &self.embed_tokens, offset)?;
        let projection = &self.lm_head;
        let spec = model::Readout {
            rows: sample.dim(1),
            multi_rows: head_multi.dim(1),
            quant: (projection.group_size, projection.bits),
        };
        let inputs = vec![
            &sample,
            &head_multi,
            &projection.weight,
            &projection.scales,
            &projection.biases,
        ];
        let mut result =
            execute(&self.context, View::Readout(spec), &inputs, || spec.graph())?.into_iter();
        let draft_id = result.next().unwrap();
        let m = result.next().unwrap();
        let logits = result.next().unwrap();
        // Oracle: record the draft readout so the driver can rank the
        // target's true token after verify. A no-op unless the CLI enabled it.
        crate::core::oracle::push_arrays(None, &logits)?;
        Ok((draft_id, m))
    }
}
