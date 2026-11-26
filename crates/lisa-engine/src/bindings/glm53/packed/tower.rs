use crate::{bindings::BindingDevice, execution::DeviceContext};
use crate::{
    cache::{LayerCache, attention::FullAttentionCache, indexer::IndexerTape, recurrent::GdnCache},
    language::runtime::Runtime,
    sampling::sampler::Sampler,
    weights::{prepared::PreparedWeights, source::Weights},
};
use lisa_backend::{ExecutionScope, ManagedDevice, ReadbackDevice, TensorAccess};
// GLM's native GGUF graph. Serial execution is the numerical baseline for
// wider prefill and verification dispatches; no affine weight adapter is used.
use crate::bindings::glm53::packed::attention::GgufDsa;
use crate::bindings::{glm53::hyper::Hc, glm53::kda::recurrent::Kda};
use crate::bindings::{
    glm53::packed::moe::Mlp,
    glm53::packed::projection::{Projection, plain},
};
use crate::core::graph::PlanCache;
use crate::weights::source::EmbeddingRows;
use crate::{bindings::shared::capture::Capture, language::LanguageModel};
use anyhow::{Context, Result, ensure};
use lisa_models::glm53::config::ModelConfig;
use lisa_models::glm53::packed_tower::{self as model, DraftStep, Request, Step};
pub fn run<D: BindingDevice, const N: usize>(
    context: &DeviceContext<D>,
    request: Request,
    inputs: &[&D::Tensor],
) -> Result<[D::Tensor; N]> {
    context.with_slot(
        "glm53.packed_tower.plans",
        || std::cell::RefCell::new(PlanCache::new(context.device().clone(), 64)),
        |slot| {
            let mut p = slot.borrow_mut();
            let recipe = request.clone();
            p.try_get_or_prepare(request, || recipe.graph())?
                .run(inputs, &mut [])?
                .into_iter()
                .collect::<Vec<_>>()
                .try_into()
                .map_err(|_| anyhow::anyhow!("tower output arity differs"))
        },
    )
}
fn rms<D: BindingDevice>(
    context: &DeviceContext<D>,
    x: &D::Tensor,
    w: &D::Tensor,
) -> Result<D::Tensor> {
    let [out] = run(context, Request::Rms, &[x, w])?;
    Ok(out)
}
fn reshape<D: BindingDevice>(
    context: &DeviceContext<D>,
    x: &D::Tensor,
    shape: &[i32],
) -> Result<D::Tensor> {
    let [out] = run(context, Request::Reshape(shape.to_vec()), &[x])?;
    Ok(out)
}

enum Attention<D: BindingDevice + ManagedDevice + ReadbackDevice> {
    Kda(Kda<D>),
    Dsa(GgufDsa<D>),
}
struct Layer<D: BindingDevice + ManagedDevice + ReadbackDevice> {
    attn: Attention<D>,
    mlp: Mlp<D>,
    hc_attn: Hc<D>,
    hc_ffn: Hc<D>,
    in_norm: D::Tensor,
    post_norm: D::Tensor,
}
struct Mtp<D: BindingDevice + ManagedDevice + ReadbackDevice> {
    eh: Projection<D>,
    enorm: D::Tensor,
    hnorm: D::Tensor,
    anorm: D::Tensor,
    fnorm: D::Tensor,
    norm: D::Tensor,
    attn: GgufDsa<D>,
    mlp: Mlp<D>,
}
pub struct Tower<D: BindingDevice + ManagedDevice + ReadbackDevice> {
    pub context: DeviceContext<D>,
    runtime: Runtime<D, Sampler<D>>,
    // Drain outstanding commands before any zero-copy weight views disappear.
    _weights: PreparedWeights<D::Tensor, D::Reservation>,
    config: ModelConfig,
    layers: Vec<Layer<D>>,
    mtp: Mtp<D>,
    embedding: Box<dyn EmbeddingRows<D::Tensor>>,
    output: Projection<D>,
    norm: D::Tensor,
    capacity: usize,
    law: u64,
    capture: Option<Capture<D::Tensor>>,
    step: std::cell::Cell<usize>,
    schedule: Vec<Step>,
}
impl<D: BindingDevice + ManagedDevice + ReadbackDevice> Tower<D> {
    pub fn from_prepared(
        runtime: Runtime<D, Sampler<D>>,
        config: ModelConfig,
        w: PreparedWeights<D::Tensor, D::Reservation>,
        embedding: Box<dyn EmbeddingRows<D::Tensor>>,
        law: u64,
    ) -> Result<Self> {
        let context = runtime.context.clone();
        let mut layers = Vec::new();
        for layer in 0..model::LAYERS {
            let hc = |half: &str| -> Result<Hc<D>> {
                let mut views = Weights::new();
                for (alias, name) in model::hc_weights(layer, half) {
                    views.insert(alias, plain(&w, &name)?);
                }
                Hc::load_on(
                    context.clone(),
                    &mut views,
                    "hc",
                    model::HYPER.streams,
                    model::HYPER.hidden,
                    f32::from_bits(model::HYPER.epsilon),
                    model::HYPER.iterations,
                    f32::from_bits(model::HYPER.rms_epsilon),
                )
            };
            layers.push(Layer {
                attn: if model::attention(&config, layer) == model::Attention::Recurrent {
                    Attention::Kda(Kda::load_gguf(context.clone(), &w, layer)?)
                } else {
                    Attention::Dsa(GgufDsa::load_on(context.clone(), &w, layer)?)
                },
                mlp: Mlp::load_on(context.clone(), &w, layer)?,
                hc_attn: hc("attn")?,
                hc_ffn: hc("ffn")?,
                in_norm: plain(&w, &model::block_weight(layer, "attn_norm"))?,
                post_norm: plain(&w, &model::block_weight(layer, "ffn_norm"))?,
            });
        }
        let p = |name: &str| plain(&w, &model::block_weight(model::LAYERS, name));
        let mtp = Mtp {
            eh: Projection::<D>::load(
                &w,
                &model::block_weight(model::LAYERS, model::MTP_WEIGHTS[5]),
            )?,
            enorm: p(model::MTP_WEIGHTS[0])?,
            hnorm: p(model::MTP_WEIGHTS[1])?,
            anorm: p(model::MTP_WEIGHTS[2])?,
            fnorm: p(model::MTP_WEIGHTS[3])?,
            norm: p(model::MTP_WEIGHTS[4])?,
            attn: GgufDsa::load_on(context.clone(), &w, model::LAYERS)?,
            mlp: Mlp::load_on(context.clone(), &w, model::LAYERS)?,
        };
        let output = Projection::<D>::load(&w, model::OUTPUT_WEIGHT)?;
        let norm = plain(&w, model::OUTPUT_NORM_WEIGHT)?;
        Ok(Self {
            runtime,
            context,
            _weights: w,
            config,
            layers,
            mtp,
            embedding,
            output,
            norm,
            capacity: model::DEFAULT_CAPACITY,
            law,
            capture: None,
            step: std::cell::Cell::new(0),
            schedule: model::schedule(),
        })
    }
    fn embed(&self, token: u32) -> Result<D::Tensor> {
        ensure!(
            (token as usize) < model::VOCAB,
            "GLM token outside vocabulary"
        );
        self.embedding.row(token)
    }

    fn one(&self, token: u32, caches: &mut [LayerCache<D>]) -> Result<(D::Tensor, D::Tensor)> {
        ensure!(
            caches.len() == model::LAYERS,
            "GLM cache layer count mismatch"
        );
        let position = match &caches[model::POSITION_LAYER] {
            LayerCache::Full(f) => f.offset,
            _ => anyhow::bail!("GLM DSA cache missing"),
        };
        ensure!(position < self.capacity, "GLM context capacity exceeded");
        let _trace = self.context.trace("glm53.gguf.forward");
        let _scratch = self
            .context
            .device()
            .enter_scope(ExecutionScope::OrderedOutputReuse);
        let embedded = self.embed(token)?;
        let mut state = embedded;
        let mut layer_index = 0;
        let mut input = None;
        let mut post = None;
        let mut comb = None;
        let mut attended = None;
        let mut output = None;
        let mut final_state = None;
        let mut layer_trace = None;
        for step in &self.schedule {
            match *step {
                Step::ExpandEmbedding => {
                    [state] = run(&self.context, Request::Expand, &[&state])?;
                }
                Step::BeginLayer(i) => {
                    layer_index = i;
                    layer_trace = Some(self.context.trace_detail("glm53.gguf.layer", i as u64));
                }
                Step::AttentionInput | Step::FfnInput => {
                    let l = &self.layers[layer_index];
                    let (h, n) = if *step == Step::AttentionInput {
                        (&l.hc_attn, &l.in_norm)
                    } else {
                        (&l.hc_ffn, &l.post_norm)
                    };
                    let (_, x, p, c) = h
                        .split_normed(&state, n, model::NORM_EPSILON)?
                        .context("GLM HC split declined")?;
                    input = Some(x);
                    post = Some(p);
                    comb = Some(c);
                }
                Step::Attention => {
                    let x = input.take().context("attention input missing")?;
                    attended = Some(
                        match (&self.layers[layer_index].attn, &mut caches[layer_index]) {
                            (Attention::Kda(k), LayerCache::Linear(g)) => k.forward_at_position(
                                &x,
                                &mut g.conv,
                                &mut g.ssm,
                                None,
                                layer_index,
                                position,
                            )?,
                            (Attention::Dsa(d), LayerCache::Full(f)) => {
                                d.forward(&x, f, layer_index)?
                            }
                            _ => anyhow::bail!("GLM cache kind mismatch at layer {layer_index}"),
                        },
                    );
                }
                Step::AttentionResidual | Step::FfnResidual => {
                    let l = &self.layers[layer_index];
                    let (h, x) = if *step == Step::AttentionResidual {
                        (&l.hc_attn, attended.as_ref())
                    } else {
                        (&l.hc_ffn, output.as_ref())
                    };
                    state = h.expand(
                        x.context("residual output missing")?,
                        &state,
                        &post.take().context("post mix missing")?,
                        &comb.take().context("combination missing")?,
                    )?;
                }
                Step::Ffn => {
                    output = Some(
                        self.layers[layer_index]
                            .mlp
                            .forward(&input.take().context("FFN input missing")?)?,
                    );
                }
                Step::CaptureLayer => {
                    if let Some(c) = self.capture.as_ref().filter(|c| c.wants(layer_index)) {
                        c.store_ephemeral(&format!("L{layer_index}.state_p{position}"), &state);
                        c.store_ephemeral(
                            &format!("L{layer_index}.attn_p{position}"),
                            attended.as_ref().context("capture attention missing")?,
                        );
                        c.store_ephemeral(
                            &format!("L{layer_index}.ffn_p{position}"),
                            output.as_ref().context("capture FFN missing")?,
                        );
                    }
                }
                Step::EndLayer => {
                    layer_trace.take();
                    attended.take();
                    output.take();
                }
                Step::FinalNorm => {
                    let [hidden, multi] =
                        run(&self.context, Request::Final, &[&state, &self.norm])?;
                    final_state = Some((hidden, multi));
                }
            }
        }
        let (hidden, multi) = final_state.context("decoder plan omitted final state")?;
        self.context.device().evaluate(&hidden, false)?;
        self.step.set(position);
        Ok((hidden, multi))
    }
    fn prompt_block(
        &self,
        tokens: &[u32],
        caches: &mut [LayerCache<D>],
    ) -> Result<(D::Tensor, D::Tensor)> {
        ensure!(
            !tokens.is_empty() && tokens.len() <= 8,
            "invalid GLM prompt block"
        );
        ensure!(
            caches.len() == model::LAYERS,
            "GLM cache layer count mismatch"
        );
        let position = caches[model::POSITION_LAYER]
            .full_offset()
            .context("GLM DSA cache missing")?;
        let _trace = self.context.trace("glm53.gguf.prefill_block");
        let _scratch = self
            .context
            .device()
            .enter_scope(ExecutionScope::OrderedOutputReuse);
        let mut rows = tokens
            .iter()
            .map(|&token| {
                let [state] = run(&self.context, Request::Expand, &[&self.embed(token)?])?;
                Ok(state)
            })
            .collect::<Result<Vec<_>>>()?;
        for (i, layer) in self.layers.iter().enumerate() {
            let _layer = self.context.trace_detail("glm53.gguf.layer", i as u64);
            let mut ffn_inputs = Vec::with_capacity(rows.len());
            let mut post = Vec::with_capacity(rows.len());
            let mut comb = Vec::with_capacity(rows.len());
            let capture = self.capture.as_ref().filter(|c| c.wants(i));
            let mut attention_capture = Vec::new();
            for (row, state) in rows.iter_mut().enumerate() {
                let (_, x, p, c) = layer
                    .hc_attn
                    .split_normed(state, &layer.in_norm, model::NORM_EPSILON)?
                    .context("GLM HC split declined")?;
                // Keep the reference's per-position dense/sparse boundaries,
                // KDA recurrence and DSA value-projection slices unchanged.
                let attended = match (&layer.attn, &mut caches[i]) {
                    (Attention::Kda(k), LayerCache::Linear(g)) => {
                        k.forward_at_position(&x, &mut g.conv, &mut g.ssm, None, i, position + row)?
                    }
                    (Attention::Dsa(d), LayerCache::Full(f)) => d.forward(&x, f, i)?,
                    _ => anyhow::bail!("GLM cache kind mismatch at layer {i}"),
                };
                *state = layer.hc_attn.expand(&attended, state, &p, &c)?;
                if capture.is_some() {
                    attention_capture.push(attended);
                }
                let (_, x, p, c) = layer
                    .hc_ffn
                    .split_normed(state, &layer.post_norm, model::NORM_EPSILON)?
                    .context("GLM HC split declined")?;
                ffn_inputs.push(x);
                post.push(p);
                comb.push(c);
            }
            let outputs = layer.mlp.prompt_rows(&ffn_inputs)?;
            ensure!(
                outputs.len() == rows.len(),
                "GLM prompt FFN output count differs"
            );
            for row in 0..rows.len() {
                rows[row] =
                    layer
                        .hc_ffn
                        .expand(&outputs[row], &rows[row], &post[row], &comb[row])?;
                if let Some(capture) = capture {
                    let p = position + row;
                    capture.store_ephemeral(&format!("L{i}.state_p{p}"), &rows[row]);
                    capture.store_ephemeral(&format!("L{i}.attn_p{p}"), &attention_capture[row]);
                    capture.store_ephemeral(&format!("L{i}.ffn_p{p}"), &outputs[row]);
                }
            }
            // Release temporary row leases before the next layer. The ordered
            // scope permits reuse only after all earlier reads are encoded.
            self.context.device().submit(false)?;
        }
        let [hidden, multi] = run(
            &self.context,
            Request::Final,
            &[rows.last().unwrap(), &self.norm],
        )?;
        self.context.device().evaluate(&hidden, false)?;
        self.step.set(position + tokens.len() - 1);
        Ok((hidden, multi))
    }
    fn feed(
        &self,
        tokens: &[u32],
        caches: &mut [LayerCache<D>],
        all: bool,
    ) -> Result<(D::Tensor, D::Tensor)> {
        ensure!(!tokens.is_empty(), "GLM requires nonempty input");
        let position = match caches.get(model::POSITION_LAYER) {
            Some(LayerCache::Full(f)) => f.offset,
            _ => anyhow::bail!("GLM cache missing"),
        };
        ensure!(
            model::validate_feed(tokens, position, self.capacity),
            "GLM token or context range invalid"
        );
        // Full forward/verification retains the independent serial
        // oracle. Prompt blocks bound temporary storage independently of layers.
        if !all && tokens.len() > 1 {
            let mut last = None;
            for chunk in tokens.chunks(8) {
                last = Some(self.prompt_block(chunk, caches)?);
                self.context.device().submit(true)?;
                self.context.device().trim_unused();
            }
            return Ok(last.unwrap());
        }
        let mut hidden = Vec::new();
        let mut multi = Vec::new();
        for (i, &token) in tokens.iter().enumerate() {
            let (h, m) = self.one(token, caches)?;
            if model::retain_row(i, tokens.len(), all) {
                hidden.push(h);
                multi.push(m);
            }
        }
        Ok((
            run::<D, 1>(
                &self.context,
                Request::Join {
                    count: hidden.len(),
                },
                &hidden.iter().collect::<Vec<_>>(),
            )?
            .into_iter()
            .next()
            .unwrap(),
            run::<D, 1>(
                &self.context,
                Request::Join { count: multi.len() },
                &multi.iter().collect::<Vec<_>>(),
            )?
            .into_iter()
            .next()
            .unwrap(),
        ))
    }
}
impl<D: BindingDevice + ManagedDevice + ReadbackDevice> LanguageModel<Runtime<D, Sampler<D>>>
    for Tower<D>
where
    D::Error: From<anyhow::Error>,
{
    fn runtime(&self) -> &Runtime<D, Sampler<D>> {
        &self.runtime
    }
    fn cache_numerical_law(&self) -> Option<u64> {
        Some(self.law)
    }
    fn prefix_cache_granularity(&self) -> Option<usize> {
        Some(1)
    }
    fn max_position_embeddings(&self) -> usize {
        self.capacity
    }
    fn set_context_capacity(&mut self, c: usize) -> Result<()> {
        ensure!(
            c > 0 && c <= self.config.max_position_embeddings,
            "invalid GLM context capacity"
        );
        self.capacity = c;
        Ok(())
    }
    fn eos_token_id(&self) -> i64 {
        self.config.eos_token_id
    }
    fn eos_token_ids(&self) -> Vec<i64> {
        self.config.eos_token_ids.clone()
    }
    fn new_caches(&self) -> Vec<LayerCache<D>> {
        (0..model::LAYERS)
            .map(|i| {
                if model::attention(&self.config, i) == model::Attention::Recurrent {
                    LayerCache::Linear(GdnCache::new())
                } else {
                    LayerCache::Full(FullAttentionCache::new(
                        self.context.clone(),
                        model::FULL_CACHE.0,
                        model::FULL_CACHE.1,
                    ))
                }
            })
            .collect()
    }
    fn forward(
        &mut self,
        ids: &D::Tensor,
        caches: Option<&mut Vec<LayerCache<D>>>,
    ) -> Result<(D::Tensor, D::Tensor)> {
        ensure!(
            ids.rank() == 2 && ids.dim(0) == 1,
            "GLM GGUF expects one stream"
        );
        let tokens = ids.read_ids()?;
        let mut fresh;
        let caches = match caches {
            Some(c) => c,
            None => {
                fresh = self.new_caches();
                &mut fresh
            }
        };
        self.feed(&tokens, caches, true)
    }
    fn forward_capture(
        &mut self,
        ids: &D::Tensor,
        caches: Option<&mut Vec<LayerCache<D>>>,
        _capture: bool,
    ) -> Result<(D::Tensor, D::Tensor)> {
        self.forward(ids, caches)
    }
    fn prefill(&mut self, tokens: &[u32], caches: &mut Vec<LayerCache<D>>) -> Result<D::Tensor> {
        reshape(
            &self.context,
            &self.feed(tokens, caches, false)?.0,
            &[1, model::WIDTH as i32],
        )
    }
    fn prefill_multi(
        &mut self,
        tokens: &[u32],
        caches: &mut Vec<LayerCache<D>>,
    ) -> Result<(D::Tensor, D::Tensor)> {
        let (h, m) = self.feed(tokens, caches, false)?;
        Ok((
            reshape(&self.context, &h, &[1, model::WIDTH as i32])?,
            reshape(
                &self.context,
                &m,
                &[1, (model::WIDTH * model::STREAMS) as i32],
            )?,
        ))
    }
    fn head(&self, x: &D::Tensor) -> Result<D::Tensor, D::Error> {
        let result = || -> Result<D::Tensor> {
            let [shaped, logits] = run(
                &self.context,
                Request::Head {
                    shape: x.shape().to_vec(),
                    projection: self.output.descriptor_on(self.context.device())?,
                },
                &[x, &self.output],
            )?;
            if let Some(c) = &self.capture {
                c.store(&format!("logits_p{}", self.step.get()), &logits);
            }
            Ok(shaped)
        };
        result().map_err(Into::into)
    }

    fn warmup(&mut self, seed: &[u32]) -> Result<()> {
        let t = *seed.first().context("empty GLM warmup seed")?;
        let mut c = self.new_caches();
        let h = self.prefill(&[t], &mut c)?;
        self.head(&h)?.eval()?;
        Ok(())
    }
    fn batch_decode_ok(&self) -> bool {
        false
    }
    fn kv_bytes_per_token(&self) -> usize {
        model::KV_BYTES_PER_TOKEN
    }
    fn stream_state_bytes(&self) -> usize {
        model::STREAM_STATE_BYTES
    }
    fn has_target_mtp(&self) -> bool {
        true
    }
    fn target_mtp_step(
        &mut self,
        tape: &mut IndexerTape<D>,
        target: &D::Tensor,
        next: u32,
        position: usize,
        min_position: usize,
    ) -> Result<D::Tensor> {
        let _trace = self.context.trace("glm53.gguf.mtp");
        let mut embed = None;
        let mut hidden = None;
        let mut x = None;
        let mut a = None;
        let mut residual = None;
        let mut routed = None;
        let mut shared = None;
        let mut result = None;
        for step in model::DRAFT {
            match step {
                DraftStep::EmbeddingNorm => {
                    embed = Some(rms(&self.context, &self.embed(next)?, &self.mtp.enorm)?)
                }
                DraftStep::HiddenNorm => {
                    let [h] = run(&self.context, Request::MeanNorm, &[target, &self.mtp.hnorm])?;
                    hidden = Some(h);
                }
                DraftStep::JoinProject => {
                    let [value] = run(
                        &self.context,
                        Request::JoinProject(self.mtp.eh.descriptor_on(self.context.device())?),
                        &[
                            &embed.take().context("draft embedding missing")?,
                            &hidden.take().context("draft hidden missing")?,
                            &self.mtp.eh,
                        ],
                    )?;
                    x = Some(value);
                }
                DraftStep::AttentionNorm => {
                    a = Some(rms(
                        &self.context,
                        x.as_ref().context("draft input missing")?,
                        &self.mtp.anorm,
                    )?)
                }
                DraftStep::Attention => {
                    a = Some(self.mtp.attn.mtp(
                        &a.take().context("draft attention input missing")?,
                        tape,
                        position,
                        min_position,
                    )?)
                }
                DraftStep::AddNorm => {
                    let [r, input] = run(
                        &self.context,
                        Request::AddNorm,
                        &[
                            &x.take().context("draft input missing")?,
                            &a.take().context("draft attention missing")?,
                            &self.mtp.fnorm,
                        ],
                    )?;
                    residual = Some(r);
                    x = Some(input);
                }
                DraftStep::Experts => {
                    let (r, s) = self
                        .mtp
                        .mlp
                        .parts(&x.take().context("draft experts input missing")?)?;
                    routed = r;
                    shared = Some(s);
                }
                DraftStep::AddThree => {
                    let [sum] = run(
                        &self.context,
                        Request::AddThree,
                        &[
                            &residual.take().context("draft residual missing")?,
                            &routed.take().context("draft routed experts missing")?,
                            &shared.take().context("draft shared expert missing")?,
                        ],
                    )?;
                    x = Some(sum);
                }
                DraftStep::OutputNorm => {
                    x = Some(rms(
                        &self.context,
                        &x.take().context("draft sum missing")?,
                        &self.mtp.norm,
                    )?)
                }
                DraftStep::Head => {
                    result =
                        Some(self.output.forward_on(
                            &self.context,
                            &x.take().context("draft output missing")?,
                        )?)
                }
            }
        }
        result.context("draft plan omitted head")
    }

    fn enable_logit_capture(&mut self) -> Result<()> {
        self.enable_stage_capture(Vec::new(), Vec::new())
    }
    fn enable_stage_capture(&mut self, layers: Vec<usize>, keys: Vec<String>) -> Result<()> {
        ensure!(
            layers.iter().all(|&i| i < model::LAYERS),
            "invalid GLM capture layer"
        );
        self.capture = Some(Capture {
            layers,
            key_filters: keys,
            preserve_dispatch: true,
            ..Default::default()
        });
        Ok(())
    }
    fn write_logit_capture(&self, path: &str) -> Result<()> {
        self.capture.as_ref().context("no GLM capture")?.flush(path)
    }
}
