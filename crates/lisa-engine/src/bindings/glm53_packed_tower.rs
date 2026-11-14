use crate::{
    attention_cache::FullAttentionCache,
    cache::LayerCache,
    indexer_cache::IndexerTape,
    language_runtime::Runtime,
    recurrent_cache::GdnCache,
    sampler::Sampler,
    weights::{gguf::ResidentWeights, source::Weights},
};
use crate::{bindings::BindingDevice, execution::DeviceContext};
use lisa_backend::{ExecutionScope, ManagedDevice, ReadbackDevice, TensorAccess};
// GLM's native GGUF graph. Serial execution is the numerical baseline for
// wider prefill and verification dispatches; no affine weight adapter is used.
use crate::bindings::glm53_packed_attention::GgufDsa;
use crate::bindings::{glm53_hyper::Hc, glm53_recurrent::Kda};
use crate::bindings::{
    glm53_moe::Mlp,
    glm53_projection::{Projection, plain},
};
use crate::checkpoint::glm53 as weights;
use crate::checkpoint::glm53::ModelConfig;
use crate::core::graph::PlanCache;
use crate::{core::capture::Capture, language::LanguageModel};
use anyhow::{Context, Result, ensure};
use lisa_gguf::{Package, Tensor, quant};
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

use std::{os::unix::fs::MetadataExt, path::Path};

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
    _weights: ResidentWeights<D, D::Reservation>,
    config: ModelConfig,
    layers: Vec<Layer<D>>,
    mtp: Mtp<D>,
    embedding: Tensor,
    output: Projection<D>,
    norm: D::Tensor,
    capacity: usize,
    law: u64,
    capture: Option<Capture<D::Tensor>>,
    step: std::cell::Cell<usize>,
    schedule: Vec<Step>,
}
impl<D: BindingDevice + ManagedDevice + ReadbackDevice> Tower<D> {
    pub fn load_on(runtime: Runtime<D, Sampler<D>>, path: &Path) -> Result<Self> {
        let context = runtime.context.clone();
        let package = Package::open(path)?;
        let config = weights::validate(&package)?;
        let names: Vec<_> = package
            .tensors
            .keys()
            .filter(|n| n.as_str() != model::EMBEDDING_WEIGHT)
            .map(String::as_str)
            .collect();
        let w = ResidentWeights::pull(context.device().clone(), &package, &names, |bytes| {
            context
                .device()
                .reserve_residency(bytes, crate::memory::ram_budget())
        })?;
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
            eh: Projection::load(
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
        let output = Projection::load(&w, model::OUTPUT_WEIGHT)?;
        let norm = plain(&w, model::OUTPUT_NORM_WEIGHT)?;
        let stat = std::fs::metadata(path)?;
        let mut law = 0xcbf29ce484222325u64;
        for byte in model::NUMERICAL_IDENTITY.iter().copied().chain(
            [
                stat.dev(),
                stat.ino(),
                stat.len(),
                stat.mtime() as u64,
                stat.mtime_nsec() as u64,
            ]
            .into_iter()
            .flat_map(u64::to_le_bytes),
        ) {
            law = (law ^ u64::from(byte)).wrapping_mul(0x100000001b3);
        }
        Ok(Self {
            runtime,
            context,
            _weights: w,
            config,
            layers,
            mtp,
            embedding: package.tensor(model::EMBEDDING_WEIGHT)?.clone(),
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
        let mut packed = vec![0u8; self.embedding.info.row_bytes()? as usize];
        self.embedding.read_rows(&[token as u64], &mut packed)?;
        let mut row = vec![0f32; model::WIDTH];
        quant::decode(self.embedding.info.kind, &packed, &mut row)?;
        Ok(self.context.device().pull(
            lisa_backend::HostTensor {
                bytes: &row.iter().flat_map(|x| x.to_ne_bytes()).collect::<Vec<_>>(),
                shape: &model::INPUT_SHAPE,
                dtype: lisa_backend::ScalarType::F32,
            }
            .into(),
            Default::default(),
        )?)
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
