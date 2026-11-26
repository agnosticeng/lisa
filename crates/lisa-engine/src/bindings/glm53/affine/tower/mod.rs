//! The `glm5_next` text tower, following the reference's F32 inference graph.
//!
//! Model configuration and the reference source define the execution path; the local
//! GLM specs are supporting notes rather than the implementation authority.
//! Embed -> 45 blocks (mHC collapse -> RMS -> KDA/DSA -> expand, then FFN
//! half) -> stream-mean -> final RMS -> lm_head. Caches reuse the shared
//! `LayerCache` variants: KDA layers use `Linear(GdnCache<D::Tensor>)` (conv window +
//! F32 recurrence state), DSA layers use `Full(FullAttentionCache<D>)`
//! (`keys` = latents `[B,Tc,512]`, `offset` = cached length).

use crate::{
    bindings::{BindingDevice, glm53::affine::moe::ExpertDevice},
    execution::DeviceContext,
    language::runtime::Runtime,
    sampling::sampler::Sampler,
};
use lisa_backend::{ReadbackDevice, TensorAccess};

use crate::cache::{
    LayerCache, attention::FullAttentionCache, indexer::IndexerTape, recurrent::GdnCache,
};

/// Counts the incremental (t==1) forwards; shared with the DSA dec captures.
pub use crate::bindings::glm53::policy::INC_STEP;

use crate::weights::affine::{QuantizedEmbedding, QuantizedLinear};
use crate::weights::source::{TensorBank, Weights};

use crate::bindings::glm53::affine::moe::{DenseMlp, MoE};
use crate::bindings::glm53::affine::tower::ops::{self as plans, one as plan_one};
use crate::bindings::glm53::attention::Dsa;
use crate::bindings::glm53::hyper::Hc;
use crate::bindings::glm53::kda::recurrent::Kda;
use crate::bindings::shared::capture::Capture;
use lisa_models::glm53::affine_tower::{self as model, Request as R};
use lisa_models::glm53::config::ModelConfig;

const DS4_GLM53_DEFAULT_CONTEXT: usize = model::DEFAULT_CONTEXT;

fn ds4_native_session_batch_position_supported(
    rows: usize,
    max_position: usize,
    dense_compact_limit: usize,
) -> bool {
    model::batch_position_supported(rows, max_position, dense_compact_limit)
}

#[allow(
    clippy::large_enum_variant,
    reason = "one value per layer; boxing would add an indirection to every dispatch"
)]
pub enum Attn<D: ExpertDevice + ReadbackDevice> {
    Kda(Kda<D>),
    Dsa(Dsa<D>),
}

pub enum Mlp<D: ExpertDevice + ReadbackDevice> {
    Dense(DenseMlp<D>),
    Sparse(MoE<D>),
}

pub struct Layer<D: ExpertDevice + ReadbackDevice> {
    pub attn: Attn<D>,
    pub mlp: Mlp<D>,
    pub hc_attn: Hc<D>,
    pub hc_ffn: Hc<D>,
    pub in_norm: D::Tensor,
    pub post_norm: D::Tensor,
}

pub struct MtpBlock<D: ExpertDevice + ReadbackDevice> {
    pub context: DeviceContext<D>,
    pub eh_proj: D::Tensor,
    pub enorm: D::Tensor,
    pub hnorm: D::Tensor,
    pub attn_norm: D::Tensor,
    pub ffn_norm: D::Tensor,
    pub shared_head_norm: D::Tensor,
    pub attn: Dsa<D>,
    pub mlp: MoE<D>,
}

impl<D: ExpertDevice + ReadbackDevice> MtpBlock<D>
where
    D::Error: From<anyhow::Error>,
{
    pub fn load_on(
        context: &DeviceContext<D>,
        w: &mut Weights<D::Tensor>,
        config: &ModelConfig,
        lazy_store: Option<std::sync::Arc<dyn TensorBank<D::Tensor>>>,
        map_all_expert_weights: bool,
    ) -> anyhow::Result<Self> {
        let h = config.hidden_size;
        let eps = config.rms_norm_eps;
        let n_layers = config.num_hidden_layers;
        let p = format!("model.layers.{n_layers}");
        let take = |w: &mut Weights<D::Tensor>, name: &str| -> anyhow::Result<D::Tensor> {
            w.remove(name)
                .ok_or_else(|| anyhow::anyhow!("glm53: tensor {name} missing"))
        };
        let eh_proj = take(w, &format!("{p}.eh_proj.weight"))?;
        anyhow::ensure!(
            eh_proj.rank() == 2 && eh_proj.dim(0) == h as i32 && eh_proj.dim(1) == (2 * h) as i32,
            "glm53 MTP eh_proj shape mismatch: {:?}",
            eh_proj.shape()
        );
        let enorm = promote(context, &take(w, &format!("{p}.enorm.weight"))?)?;
        let hnorm = promote(context, &take(w, &format!("{p}.hnorm.weight"))?)?;
        let attn_norm = promote(context, &take(w, &format!("{p}.input_layernorm.weight"))?)?;
        let ffn_norm = promote(
            context,
            &take(w, &format!("{p}.post_attention_layernorm.weight"))?,
        )?;
        let shared_head_norm =
            promote(context, &take(w, &format!("{p}.shared_head.norm.weight"))?)?;
        let attn = Dsa::load_on::<Weights<D::Tensor>>(
            context,
            w,
            &format!("{p}.self_attn"),
            config.num_attention_heads,
            config.qk_nope_head_dim,
            config.v_head_dim,
            config.index_n_heads,
            config.index_head_dim,
            config.index_topk,
            config.index_kpool,
            config.index_kpool_always_select_tail,
            eps,
        )?;
        anyhow::ensure!(
            config.is_sparse_mlp(n_layers),
            "glm53 MTP block must use sparse MoE"
        );
        let mlp = MoE::load_on::<Weights<D::Tensor>>(
            context,
            w,
            &format!("{p}.mlp"),
            config.n_routed_experts,
            config.num_experts_per_tok,
            config.n_group,
            config.topk_group,
            config.routed_scaling_factor,
            config.norm_topk_prob,
            config.n_shared_experts,
            config.moe_intermediate_size,
            h,
            config.swiglu_limit,
            lazy_store,
            map_all_expert_weights,
        )?;
        Ok(Self {
            context: context.clone(),
            eh_proj,
            enorm,
            hnorm,
            attn_norm,
            ffn_norm,
            shared_head_norm,
            attn,
            mlp,
        })
    }

    pub fn forward(
        &self,
        config: &ModelConfig,
        embed: &QuantizedEmbedding<D::Tensor>,
        lm_head: &QuantizedLinear<D::Tensor>,
        tape: &mut IndexerTape<D>,
        target_multi: &D::Tensor,
        next_token: u32,
        position: usize,
        min_position: usize,
    ) -> anyhow::Result<D::Tensor> {
        anyhow::ensure!(
            target_multi.dim(0) == 1 && target_multi.dim(1) == 1,
            "glm53 MTP expects one target hidden row"
        );
        let h = config.hidden_size;
        let streams = config.hc_mult;
        anyhow::ensure!(
            target_multi.dim(2) == (h * streams) as i32,
            "glm53 MTP target hidden width mismatch"
        );
        anyhow::ensure!(
            (next_token as usize) < config.vocab_size,
            "glm53 MTP token {next_token} exceeds vocabulary size {}",
            config.vocab_size
        );

        let ids = crate::bindings::shared::tensor_io::pull_i32(
            &self.context,
            &[next_token as i32],
            &[1, 1],
        )?;
        let mut embedded_slot: Option<D::Tensor> = None;
        let mut hidden_slot: Option<D::Tensor> = None;
        let mut x_slot: Option<D::Tensor> = None;
        let mut attn_in_slot: Option<D::Tensor> = None;
        let mut attended_slot: Option<D::Tensor> = None;
        let mut after_attn_slot: Option<D::Tensor> = None;
        let mut ffn_in_slot: Option<D::Tensor> = None;
        let mut routed_slot: Option<D::Tensor> = None;
        let mut shared_slot: Option<D::Tensor> = None;
        let mut next_sum_slot: Option<D::Tensor> = None;
        let mut next_slot: Option<D::Tensor> = None;
        for action in model::DRAFT {
            match action {
                model::DraftStep::EmbeddingNorm => {
                    let embedded = plans::embedding(&self.context, embed, &ids)?;
                    let embedded =
                        mtp_rms(&self.context, &embedded, &self.enorm, config.rms_norm_eps)?;

                    embedded_slot = Some(embedded);
                }
                model::DraftStep::HiddenNorm => {
                    let hidden = plan_one(
                        &self.context,
                        R::DraftHidden {
                            hidden: h,
                            streams,
                            epsilon: config.rms_norm_eps.to_bits(),
                        },
                        &[target_multi, &self.hnorm],
                    )?;

                    hidden_slot = Some(hidden);
                }
                model::DraftStep::JoinProject => {
                    let embedded = embedded_slot.as_ref().expect("model draft producer");
                    let hidden = hidden_slot.as_ref().expect("model draft producer");
                    let dtype = self
                        .context
                        .device()
                        .dtype(&self.eh_proj)
                        .ok_or_else(|| anyhow::anyhow!("unsupported MTP projection dtype"))?;
                    let x = plan_one(
                        &self.context,
                        R::DraftProject { hidden: h, dtype },
                        &[embedded, hidden, &self.eh_proj],
                    )?;

                    x_slot = Some(x);
                }
                model::DraftStep::AttentionNorm => {
                    let x = x_slot.as_ref().expect("model draft producer");
                    let attn_in = mtp_rms(&self.context, x, &self.attn_norm, config.rms_norm_eps)?;

                    attn_in_slot = Some(attn_in);
                }
                model::DraftStep::Attention => {
                    let attn_in = attn_in_slot.as_ref().expect("model draft producer");
                    let attended = self
                        .attn
                        .forward_mtp(attn_in, tape, position, min_position)?;

                    attended_slot = Some(attended);
                }
                model::DraftStep::AddNorm => {
                    let x = x_slot.as_ref().expect("model draft producer");
                    let attended = attended_slot.as_ref().expect("model draft producer");
                    let mut add_norm = plans::run(
                        &self.context,
                        R::AddNorm {
                            epsilon: config.rms_norm_eps.to_bits(),
                        },
                        &[x, attended, &self.ffn_norm],
                    )?
                    .into_iter();
                    let after_attn = add_norm.next().unwrap();
                    let ffn_in = add_norm.next().unwrap();

                    after_attn_slot = Some(after_attn);
                    ffn_in_slot = Some(ffn_in);
                }
                model::DraftStep::Experts => {
                    let ffn_in = ffn_in_slot.as_ref().expect("model draft producer");
                    let (routed, shared) = self.mlp.forward_parts_with_capture(ffn_in, None, 0)?;
                    let shared = shared.ok_or_else(|| {
                        anyhow::anyhow!("GLM MTP checkpoint has no shared expert")
                    })?;

                    routed_slot = Some(routed);
                    shared_slot = Some(shared);
                }
                model::DraftStep::AddThree => {
                    let after_attn = after_attn_slot.as_ref().expect("model draft producer");
                    let routed = routed_slot.as_ref().expect("model draft producer");
                    let shared = shared_slot.as_ref().expect("model draft producer");
                    let next_sum =
                        plan_one(&self.context, R::AddThree, &[after_attn, routed, shared])?;

                    next_sum_slot = Some(next_sum);
                }
                model::DraftStep::OutputNorm => {
                    let next_sum = next_sum_slot.as_ref().expect("model draft producer");
                    let next = mtp_rms(
                        &self.context,
                        next_sum,
                        &self.shared_head_norm,
                        config.rms_norm_eps,
                    )?;

                    next_slot = Some(next);
                }
                model::DraftStep::Head => {
                    let next = next_slot.as_ref().expect("model draft producer");
                    // The reference's shared output head uses the same native Q8 dispatcher in
                    // target and MTP execution, including the vocabulary-sized reduction.
                    let logits = crate::bindings::glm53::affine::projection::forward(
                        &self.context,
                        lm_head,
                        next,
                        crate::bindings::glm53::policy::glm53_q8_mpp_prefill_override()
                            != Some(false),
                    )?;
                    return Ok(logits);
                }
            }
        }
        anyhow::bail!("GLM draft plan has no output")
    }
}

pub struct Tower<D: ExpertDevice + ReadbackDevice> {
    pub context: DeviceContext<D>,
    pub runtime: Runtime<D, Sampler<D>>,
    pub vision: Option<crate::vision::Vision<D>>,
    pub config: ModelConfig,
    pub context_capacity: usize,
    pub capture: Option<Capture<D::Tensor>>,
    pub prefill_mode: bool,
    pub prefill_final_chunk: bool,
    pub embed: QuantizedEmbedding<D::Tensor>,
    pub layers: Vec<Layer<D>>,
    pub final_norm: D::Tensor,
    pub lm_head: QuantizedLinear<D::Tensor>,
    pub mtp: Option<MtpBlock<D>>,
    pub _expert_residency_set: Option<D::Residency>,
    pub n_layers: usize,
    // Dropped after model arrays and residency resources release their owners.
    pub _resident_memory: Option<D::Reservation>,
}

pub fn rms<D: BindingDevice>(
    context: &DeviceContext<D>,
    x: &D::Tensor,
    w: &D::Tensor,
    eps: f32,
) -> anyhow::Result<D::Tensor> {
    plan_one(
        context,
        R::Rms {
            epsilon: eps.to_bits(),
        },
        &[x, w],
    )
}
pub fn final_rms<D: BindingDevice>(
    context: &DeviceContext<D>,
    x: &D::Tensor,
    w: &D::Tensor,
    eps: f32,
) -> anyhow::Result<D::Tensor> {
    plan_one(
        context,
        R::OrderedRms {
            epsilon: eps.to_bits(),
        },
        &[x, w],
    )
}
pub fn final_hc_mean<D: BindingDevice>(
    context: &DeviceContext<D>,
    x: &D::Tensor,
) -> anyhow::Result<D::Tensor> {
    plan_one(
        context,
        R::Mean {
            shape: x.shape().to_vec(),
        },
        &[x],
    )
}
pub fn mtp_rms<D: BindingDevice>(
    context: &DeviceContext<D>,
    x: &D::Tensor,
    w: &D::Tensor,
    eps: f32,
) -> anyhow::Result<D::Tensor> {
    final_rms(context, x, w, eps)
}
pub fn reshape<D: BindingDevice>(
    context: &DeviceContext<D>,
    x: &D::Tensor,
    shape: &[i32],
) -> anyhow::Result<D::Tensor> {
    plan_one(context, R::Reshape(shape.to_vec()), &[x])
        .map_err(|e| anyhow::Error::msg(e.to_string()))
}
pub fn promote<D: BindingDevice>(
    context: &DeviceContext<D>,
    x: &D::Tensor,
) -> anyhow::Result<D::Tensor> {
    plan_one(context, R::Promote, &[x]).map_err(|e| anyhow::Error::msg(e.to_string()))
}

pub fn narrow<D: BindingDevice>(
    context: &DeviceContext<D>,
    x: &D::Tensor,
    axis: usize,
    start: usize,
    len: usize,
) -> anyhow::Result<D::Tensor> {
    plan_one(
        context,
        R::Slice {
            axis: axis as i32,
            start: start as i32,
            end: (start + len) as i32,
        },
        &[x],
    )
    .map_err(|e| anyhow::Error::msg(e.to_string()))
}
pub fn join<D: BindingDevice>(
    context: &DeviceContext<D>,
    arrays: &[&D::Tensor],
    axis: i32,
) -> anyhow::Result<D::Tensor> {
    plan_one(
        context,
        R::Join {
            count: arrays.len(),
            axis,
        },
        arrays,
    )
    .map_err(|e| anyhow::Error::msg(e.to_string()))
}

impl<D: ExpertDevice + ReadbackDevice> Tower<D>
where
    D::Error: From<anyhow::Error>,
{
    pub fn from_weights(
        runtime: Runtime<D, Sampler<D>>,
        w: Weights<D::Tensor>,
        config: ModelConfig,
    ) -> anyhow::Result<Self> {
        let _context = runtime.context.clone();
        Self::from_weights_with_lazy_experts(runtime, w, config, None, true, true)
    }

    pub fn from_weights_with_lazy_experts(
        runtime: Runtime<D, Sampler<D>>,
        mut w: Weights<D::Tensor>,
        config: ModelConfig,
        lazy_store: Option<std::sync::Arc<dyn TensorBank<D::Tensor>>>,
        enable_expert_residency: bool,
        request_residency: bool,
    ) -> anyhow::Result<Self> {
        let context = runtime.context.clone();
        let context_capacity = config
            .max_position_embeddings
            .min(DS4_GLM53_DEFAULT_CONTEXT);
        let h = config.hidden_size;
        let eps = config.rms_norm_eps;
        let take = |w: &mut Weights<D::Tensor>, name: &str| -> anyhow::Result<D::Tensor> {
            w.remove(name)
                .ok_or_else(|| anyhow::anyhow!("glm53: tensor {name} missing"))
        };

        let mut embed = QuantizedEmbedding::load(&mut w, "model.embed_tokens")?;
        embed.bits = 8;
        embed.group_size = 64;
        // The reference consumes GLM weights into F32 graph buffers. Promote affine
        // metadata once so embedding dequantization does not round to BF16.
        embed.scales = promote(&context, &embed.scales)?;
        embed.biases = promote(&context, &embed.biases)?;
        let mut lm_head = QuantizedLinear::load(&mut w, "lm_head", "")?;
        lm_head.set_quant(64, 8);
        lm_head.scales = promote(&context, &lm_head.scales)?;
        lm_head.biases = promote(&context, &lm_head.biases)?;
        let final_norm = promote(&context, &take(&mut w, "model.norm.weight")?)?;

        let n_layers = config.num_hidden_layers; // 45; MTP (layer 45) stays in the map
        let mut layers = Vec::with_capacity(n_layers);
        for i in 0..n_layers {
            let p = format!("model.layers.{i}");
            let in_norm = promote(
                &context,
                &take(&mut w, &format!("{p}.input_layernorm.weight"))?,
            )?;
            let post_norm = promote(
                &context,
                &take(&mut w, &format!("{p}.post_attention_layernorm.weight"))?,
            )?;
            let hc_attn = Hc::load_on(
                context.clone(),
                &mut w,
                &format!("{p}.hc_attn"),
                config.hc_mult,
                h,
                config.hc_eps,
                config.hc_sinkhorn_iters,
                eps,
            )?;
            let hc_ffn = Hc::load_on(
                context.clone(),
                &mut w,
                &format!("{p}.hc_ffn"),
                config.hc_mult,
                h,
                config.hc_eps,
                config.hc_sinkhorn_iters,
                eps,
            )?;
            let attn = if config.is_kda(i) {
                Attn::Kda(Kda::load_on::<Weights<D::Tensor>>(
                    context.clone(),
                    &mut w,
                    &format!("{p}.self_attn"),
                    config.linear_num_heads,
                    config.linear_head_dim,
                    config.linear_conv_kernel_dim,
                    config.linear_lower_bound,
                    eps,
                )?)
            } else {
                Attn::Dsa(Dsa::load_on::<Weights<D::Tensor>>(
                    &context,
                    &mut w,
                    &format!("{p}.self_attn"),
                    config.num_attention_heads,
                    config.qk_nope_head_dim,
                    config.v_head_dim,
                    config.index_n_heads,
                    config.index_head_dim,
                    config.index_topk,
                    config.index_kpool,
                    config.index_kpool_always_select_tail,
                    eps,
                )?)
            };
            let mlp = if config.is_sparse_mlp(i) {
                Mlp::Sparse(MoE::load_on::<Weights<D::Tensor>>(
                    &context,
                    &mut w,
                    &format!("{p}.mlp"),
                    config.n_routed_experts,
                    config.num_experts_per_tok,
                    config.n_group,
                    config.topk_group,
                    config.routed_scaling_factor,
                    config.norm_topk_prob,
                    config.n_shared_experts,
                    config.moe_intermediate_size,
                    h,
                    config.swiglu_limit,
                    lazy_store.clone(),
                    enable_expert_residency,
                )?)
            } else {
                Mlp::Dense(DenseMlp::load_on::<Weights<D::Tensor>>(
                    &context,
                    &mut w,
                    &format!("{p}.mlp"),
                    config.swiglu_limit,
                )?)
            };
            layers.push(Layer {
                attn,
                mlp,
                hc_attn,
                hc_ffn,
                in_norm,
                post_norm,
            });
        }
        let mut mtp = if config.num_nextn_predict_layers > 0 {
            Some(MtpBlock::load_on(
                &context,
                &mut w,
                &config,
                lazy_store.clone(),
                enable_expert_residency,
            )?)
        } else {
            None
        };
        // The reference reuses the grouped MoE gate/up/activation/down storage across
        // layers. Every consumer is encoded before the next layer overwrites
        // these full-write buffers on the same ordered Metal queue.
        let grouped_workspace = std::sync::Arc::new(
            crate::bindings::glm53::affine::moe::GroupedMoeWorkspace::default(),
        );
        for layer in &mut layers {
            if let Mlp::Sparse(moe) = &mut layer.mlp {
                moe.set_grouped_workspace(grouped_workspace.clone());
            }
        }
        if let Some(block) = mtp.as_mut() {
            block.mlp.set_grouped_workspace(grouped_workspace);
        }
        // The reference registers the model's mapped buffers through a single
        // queue-wide residency set. Metal limits the number of residency sets
        // attached to one queue, so combine all routed experts across the
        // tower and MTP block instead of creating one set per MoE layer.
        let resident_buffers = lazy_store
            .as_ref()
            .map_or_else(Vec::new, |store| store.resident_allocations());
        if let Some(store) = lazy_store.as_ref().filter(|store| store.is_resident()) {
            // Metadata promotion and paired projection buffers are ordinary
            // private allocations in addition to the immutable mapped arena.
            let bytes = store
                .resident_nbytes()
                .checked_add(context.device().memory_usage()?.private)
                .ok_or_else(|| anyhow::anyhow!("GLM resident memory accounting overflow"))?;
            anyhow::ensure!(
                bytes <= crate::memory::ram_budget(),
                "GLM resident weights and private allocations require {:.2} GiB, above the {:.2} GiB RAM budget",
                bytes as f64 / (1u64 << 30) as f64,
                crate::memory::ram_budget() as f64 / (1u64 << 30) as f64,
            );
        }
        let residency_capacity = resident_buffers.len()
            + layers
                .iter()
                .map(|layer| match &layer.mlp {
                    Mlp::Sparse(moe) => moe.expert_residency_capacity(),
                    Mlp::Dense(_) => 0,
                })
                .sum::<usize>()
            + mtp
                .as_ref()
                .map_or(0, |block| block.mlp.expert_residency_capacity());
        let expert_residency_set = if !enable_expert_residency || residency_capacity == 0 {
            None
        } else {
            context
                .device()
                .new_residency(residency_capacity)
                .and_then(|set| {
                    if !resident_buffers
                        .iter()
                        .all(|buffer| context.device().residency_add(&set, buffer))
                    {
                        return None;
                    }
                    let layers_added = layers.iter().all(|layer| match &layer.mlp {
                        Mlp::Sparse(moe) => moe.add_expert_resources_to_residency_set(&set),
                        Mlp::Dense(_) => true,
                    });
                    let mtp_added = mtp
                        .as_ref()
                        .is_none_or(|block| block.mlp.add_expert_resources_to_residency_set(&set));
                    if !layers_added || !mtp_added {
                        return None;
                    }
                    context.device().residency_commit(&set);
                    let added_to_queue = if request_residency {
                        context.device().residency_attach(&set, true)
                    } else {
                        context.device().residency_attach(&set, false)
                    };
                    added_to_queue.then_some(set)
                })
        };
        anyhow::ensure!(
            resident_buffers.is_empty() || expert_residency_set.is_some(),
            "GLM resident model requires a queue-wide Metal residency set"
        );
        if let Some(set) = expert_residency_set.as_ref() {
            for layer in &mut layers {
                if let Mlp::Sparse(moe) = &mut layer.mlp {
                    moe.set_model_residency_set(set.clone());
                }
            }
            if let Some(block) = mtp.as_mut() {
                block.mlp.set_model_residency_set(set.clone());
            }
        }
        if context.trace_enabled("glm53-loader") {
            eprintln!(
                "[glm53-loader] model expert queue residency: {} allocations",
                expert_residency_set
                    .as_ref()
                    .map_or(0, |set| context.device().residency_count(set))
            );
        }
        Ok(Self {
            context: context.clone(),
            runtime,
            vision: None,
            config,
            context_capacity,
            capture: None,
            prefill_mode: false,
            prefill_final_chunk: false,
            embed,
            layers,
            final_norm,
            lm_head,
            mtp,
            _expert_residency_set: expert_residency_set,
            n_layers,
            _resident_memory: None,
        })
    }

    pub fn mtp_step(
        &mut self,
        tape: &mut IndexerTape<D>,
        target_multi: &D::Tensor,
        next_token: u32,
        position: usize,
        min_position: usize,
    ) -> anyhow::Result<D::Tensor> {
        anyhow::ensure!(
            min_position <= position && position < self.context_capacity,
            "glm53 MTP position {position}/minimum {min_position} exceeds context capacity {}",
            self.context_capacity
        );
        let _scratch = self
            .context
            .device()
            .enter_scope(lisa_backend::ExecutionScope::OrderedOutputReuse);
        let mtp = self
            .mtp
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("glm53 checkpoint has no MTP block"))?;
        mtp.forward(
            &self.config,
            &self.embed,
            &self.lm_head,
            tape,
            target_multi,
            next_token,
            position,
            min_position,
        )
    }

    /// Decode one token for rows whose DSA tapes have different absolute
    /// lengths. mHC, KDA/DSA projections, output projections, and MoE work run
    /// across the cohort; stateful KDA recurrence and DSA cache/attention
    /// updates still advance each row against its own cache, matching the reference's
    /// native session-batch ownership model.
    pub fn forward_ragged_inner(
        &mut self,
        ids: &D::Tensor,
        per_row_caches: &mut [Vec<LayerCache<D>>],
    ) -> anyhow::Result<(D::Tensor, D::Tensor)> {
        let (b, s) = (ids.dim(0) as usize, ids.dim(1) as usize);
        anyhow::ensure!(
            b == per_row_caches.len() && s == 1,
            "GLM ragged decode expects one token and cache per row, got ids {:?}, caches {}",
            ids.shape(),
            per_row_caches.len()
        );
        anyhow::ensure!(b > 1, "GLM ragged decode requires at least two rows");
        anyhow::ensure!(
            per_row_caches
                .iter()
                .all(|row| row.len() == self.layers.len()),
            "GLM ragged decode cache depth does not match the tower"
        );
        // Validate the complete cohort before updating any row's cache.
        for caches in per_row_caches.iter() {
            self.check_context_span(caches, 1)?;
        }
        // Preserve-dispatch captures retain their D::Tensors in Capture::pending.
        // Their buffer leases remain live, so ordered reuse can recycle only
        // unrelated dead scratch while those snapshots wait for readback.
        let _scratch = self
            .capture
            .as_ref()
            .is_none_or(|c| c.layers.is_empty() || c.preserves_dispatch())
            .then(|| {
                self.context
                    .device()
                    .enter_scope(lisa_backend::ExecutionScope::OrderedOutputReuse)
            });
        let dense_compact_limit = self
            .config
            .index_topk
            .saturating_add(self.config.index_kpool.saturating_sub(1));
        let max_position = per_row_caches
            .iter()
            .flat_map(|row| row.iter())
            .filter_map(|cache| match cache {
                LayerCache::Full(cache) => Some(cache.offset),
                LayerCache::Linear(_) | LayerCache::Native(_) => None,
            })
            .max()
            .unwrap_or(0);
        if !ds4_native_session_batch_position_supported(b, max_position, dense_compact_limit) {
            // Match glm53_graph_native_session_batch_supported: the reference declines
            // its native cohort kernel at this row count or sparse-context
            // boundary and evaluates each session through the serial graph.
            let mut mixed_rows = Vec::with_capacity(b);
            let mut multi_rows = Vec::with_capacity(b);
            for (row, caches) in per_row_caches.iter_mut().enumerate() {
                let token = narrow(&self.context, ids, 0, row, 1)?;
                let (mixed, multi) = self.forward_inner(&token, caches)?;
                mixed_rows.push(mixed);
                multi_rows.push(multi);
            }
            let mixed_refs: Vec<&D::Tensor> = mixed_rows.iter().collect();
            let multi_refs: Vec<&D::Tensor> = multi_rows.iter().collect();
            return Ok((
                join(&self.context, &mixed_refs, 0)?,
                join(&self.context, &multi_refs, 0)?,
            ));
        }
        let h = self.config.hidden_size;
        let c = self.config.hc_mult;
        let x = plans::embedding(&self.context, &self.embed, ids)?;
        let mut state = plan_one(
            &self.context,
            R::Expand {
                batch: b,
                rows: 1,
                streams: c,
                hidden: h,
            },
            &[&x],
        )?;
        let _capture_mhc_expansions = false;

        for li in model::layer_indices(self.layers.len(), None) {
            let layer = &mut self.layers[li];
            let capture = self.capture.as_ref().filter(|capture| capture.wants(li));
            let mut attn_inputs = None;
            let mut ffn_inputs = None;
            let mut attended_value = None;
            let mut ffn_value = None;
            let mut step_register = 0;
            for action in model::LAYER {
                let step = step_register;
                match action {
                    model::Step::AttentionInput => {
                        let (_xc, x_in, post, comb, attn_mixes, attn_split) =
                            if _capture_mhc_expansions {
                                if let Some((xc, normed, post, comb, mixes, split)) =
                                    layer.hc_attn.split_normed_with_mixes(
                                        &state,
                                        &layer.in_norm,
                                        self.config.rms_norm_eps,
                                    )?
                                {
                                    (xc, normed, post, comb, Some(mixes), Some(split))
                                } else {
                                    let (xc, post, comb) = layer.hc_attn.split(&state)?;
                                    let normed = rms(
                                        &self.context,
                                        &xc,
                                        &layer.in_norm,
                                        self.config.rms_norm_eps,
                                    )?;
                                    (xc, normed, post, comb, None, None)
                                }
                            } else if let Some((xc, normed, post, comb)) = layer
                                .hc_attn
                                .split_normed(&state, &layer.in_norm, self.config.rms_norm_eps)?
                            {
                                (xc, normed, post, comb, None, None)
                            } else {
                                let (xc, post, comb) = layer.hc_attn.split(&state)?;
                                let normed = rms(
                                    &self.context,
                                    &xc,
                                    &layer.in_norm,
                                    self.config.rms_norm_eps,
                                )?;
                                (xc, normed, post, comb, None, None)
                            };
                        if _capture_mhc_expansions && let Some(capture) = capture {
                            capture.store(&format!("L{li}.mhc_attn_in"), &state);
                            capture.store(&format!("L{li}.mhc_attn_post"), &post);
                            capture.store(&format!("L{li}.mhc_attn_comb"), &comb);
                            capture.store(&format!("L{li}.mhc_attn_collapsed"), &_xc);
                            capture.store(&format!("L{li}.mhc_attn_normed"), &x_in);
                            if let Some(mixes) = &attn_mixes {
                                capture.store(&format!("L{li}.mhc_attn_mixes"), mixes);
                            }
                            if let Some(split) = &attn_split {
                                capture.store(&format!("L{li}.mhc_attn_split"), split);
                            }
                        }
                        let step = INC_STEP.load(std::sync::atomic::Ordering::Relaxed);
                        if let Some(capture) = capture {
                            capture.store(&format!("L{li}.dec_rms_in_s{step}"), &x_in);
                        }
                        if li == 0 {
                            INC_STEP.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        }

                        attn_inputs = Some((_xc, x_in, post, comb, attn_mixes, attn_split));
                        step_register = step;
                    }
                    model::Step::Attention => {
                        let (_, x_in, _, _, _, _) =
                            attn_inputs.as_ref().expect("attention input step");
                        let attended = match &layer.attn {
                            Attn::Kda(kda) => {
                                let mut row_caches = Vec::with_capacity(b);
                                for (row, caches) in per_row_caches.iter_mut().enumerate() {
                                    match &mut caches[li] {
                                        LayerCache::Linear(cache) => row_caches.push(cache),
                                        _ => anyhow::bail!(
                                            "glm53: ragged KDA cache kind mismatch at layer {li}, row {row}"
                                        ),
                                    }
                                }
                                kda.forward_ragged(
                                    x_in,
                                    &mut row_caches,
                                    self.capture.as_ref(),
                                    li,
                                )?
                            }
                            Attn::Dsa(dsa) => {
                                let mut row_caches = Vec::with_capacity(b);
                                for (row, caches) in per_row_caches.iter_mut().enumerate() {
                                    match &mut caches[li] {
                                        LayerCache::Full(cache) => row_caches.push(cache),
                                        _ => anyhow::bail!(
                                            "glm53: ragged DSA cache kind mismatch at layer {li}, row {row}"
                                        ),
                                    }
                                }
                                dsa.forward_ragged(
                                    x_in,
                                    &mut row_caches,
                                    self.capture.as_ref(),
                                    li,
                                )?
                            }
                        };
                        if let Some(capture) = capture {
                            capture.store(&format!("L{li}.dec_attn_out_s{step}"), &attended);
                        }

                        attended_value = Some(attended);
                    }
                    model::Step::AttentionResidual => {
                        let (_, _, post, comb, _, _) =
                            attn_inputs.as_ref().expect("attention input step");
                        let attended = attended_value.as_ref().expect("attention step");
                        state = layer.hc_attn.expand(attended, &state, post, comb)?;
                        if let Some(capture) = capture {
                            capture.store(&format!("L{li}.state_after_attn"), &state);
                            capture.store(&format!("L{li}.dec_state_after_attn_s{step}"), &state);
                        }
                    }
                    model::Step::FfnInput => {
                        let (_xc2, x_in2, post2, comb2, ffn_mixes, ffn_split) =
                            if _capture_mhc_expansions {
                                if let Some((xc, normed, post, comb, mixes, split)) =
                                    layer.hc_ffn.split_normed_with_mixes(
                                        &state,
                                        &layer.post_norm,
                                        self.config.rms_norm_eps,
                                    )?
                                {
                                    (xc, normed, post, comb, Some(mixes), Some(split))
                                } else {
                                    let (xc, post, comb) = layer.hc_ffn.split(&state)?;
                                    let normed = rms(
                                        &self.context,
                                        &xc,
                                        &layer.post_norm,
                                        self.config.rms_norm_eps,
                                    )?;
                                    (xc, normed, post, comb, None, None)
                                }
                            } else if let Some((xc, normed, post, comb)) = layer
                                .hc_ffn
                                .split_normed(&state, &layer.post_norm, self.config.rms_norm_eps)?
                            {
                                (xc, normed, post, comb, None, None)
                            } else {
                                let (xc, post, comb) = layer.hc_ffn.split(&state)?;
                                let normed = rms(
                                    &self.context,
                                    &xc,
                                    &layer.post_norm,
                                    self.config.rms_norm_eps,
                                )?;
                                (xc, normed, post, comb, None, None)
                            };
                        if _capture_mhc_expansions && let Some(capture) = capture {
                            capture.store(&format!("L{li}.mhc_ffn_post"), &post2);
                            capture.store(&format!("L{li}.mhc_ffn_comb"), &comb2);
                            capture.store(&format!("L{li}.mhc_ffn_collapsed"), &_xc2);
                            capture.store(&format!("L{li}.mhc_ffn_normed"), &x_in2);
                            if let Some(mixes) = &ffn_mixes {
                                capture.store(&format!("L{li}.mhc_ffn_mixes"), mixes);
                            }
                            if let Some(split) = &ffn_split {
                                capture.store(&format!("L{li}.mhc_ffn_split"), split);
                            }
                        }
                        if let Some(capture) = capture {
                            capture.store(&format!("L{li}.mhc_ffn_normed"), &x_in2);
                        }

                        ffn_inputs = Some((_xc2, x_in2, post2, comb2, ffn_mixes, ffn_split));
                    }
                    model::Step::Ffn => {
                        let (_, _, _, _, _, _) =
                            attn_inputs.as_ref().expect("attention input step");
                        let (_, x_in2, _, _, _, _) = ffn_inputs.as_ref().expect("FFN input step");
                        let ffn = match &layer.mlp {
                            Mlp::Dense(dense) => {
                                dense.forward_with_capture(x_in2, self.capture.as_ref(), li)?
                            }
                            Mlp::Sparse(moe) => {
                                moe.forward_with_capture(x_in2, self.capture.as_ref(), li)?
                            }
                        };
                        if let Some(capture) = capture {
                            capture.store(&format!("L{li}.dec_moe_out_s{step}"), &ffn);
                        }

                        ffn_value = Some(ffn);
                    }
                    model::Step::FfnResidual => {
                        let (_, _, _, _, _, _) =
                            attn_inputs.as_ref().expect("attention input step");
                        let (_, _, post2, comb2, _, _) =
                            ffn_inputs.as_ref().expect("FFN input step");
                        let ffn = ffn_value.as_ref().expect("FFN step");
                        state = layer.hc_ffn.expand(ffn, &state, post2, comb2)?;
                        if let Some(capture) = capture {
                            capture.store(&format!("L{li}.state_after_ffn"), &state);
                            capture.store(&format!("L{li}.dec_state_after_ffn_s{step}"), &state);
                        }
                    }
                }
            }
        }

        let mixed = reshape(
            &self.context,
            &final_hc_mean(&self.context, &state)?,
            &[b as i32, 1, h as i32],
        )?;
        let mixed = final_rms(
            &self.context,
            &mixed,
            &self.final_norm,
            self.config.rms_norm_eps,
        )?;
        let multi = reshape(&self.context, &state, &[b as i32, 1, (c * h) as i32])?;
        Ok((mixed, multi))
    }

    /// One forward: `ids [B,S] i32` -> `(mixed [B,S,H], stream [B,S,C*H])`.
    pub fn forward_inner(
        &mut self,
        ids: &D::Tensor,
        caches: &mut [LayerCache<D>],
    ) -> anyhow::Result<(D::Tensor, D::Tensor)> {
        self.forward_inner_until(ids, caches, None)
    }

    pub fn forward_prefill_chunk(
        &mut self,
        ids: &D::Tensor,
        caches: &mut [LayerCache<D>],
        final_chunk: bool,
    ) -> anyhow::Result<(D::Tensor, D::Tensor)> {
        let previous = std::mem::replace(&mut self.prefill_mode, true);
        let previous_final = std::mem::replace(&mut self.prefill_final_chunk, final_chunk);
        let result = self.forward_inner(ids, caches);
        self.prefill_mode = previous;
        self.prefill_final_chunk = previous_final;
        result
    }

    pub fn forward_inner_until(
        &mut self,
        ids: &D::Tensor,
        caches: &mut [LayerCache<D>],
        stop_after_layer: Option<usize>,
    ) -> anyhow::Result<(D::Tensor, D::Tensor)> {
        let _trace_forward = self.context.trace("glm53.forward");
        let (b, s) = (ids.dim(0) as usize, ids.dim(1) as usize);
        self.check_context_span(caches, s)?;
        let _scratch = self
            .capture
            .as_ref()
            .is_none_or(|c| c.layers.is_empty() || c.preserves_dispatch())
            .then(|| {
                self.context
                    .device()
                    .enter_scope(lisa_backend::ExecutionScope::OrderedOutputReuse)
            });
        let forward_position = caches.iter().find_map(|cache| match cache {
            LayerCache::Full(cache) => Some(cache.offset),
            LayerCache::Linear(_) | LayerCache::Native(_) => None,
        });
        let prefill_mode = self.prefill_mode;
        let h = self.config.hidden_size;
        let c = self.config.hc_mult;
        if let Some(layer) = stop_after_layer {
            anyhow::ensure!(
                layer < self.layers.len(),
                "GLM prefix layer {layer} exceeds {} layers",
                self.layers.len()
            );
        }
        let x = {
            let _trace_embed = self.context.trace("glm53.embed");
            let x = plans::embedding(&self.context, &self.embed, ids)?;
            if let Some(v) = &self.vision {
                v.splice(x, forward_position.unwrap_or(0))?
            } else {
                x
            } // [B,S,H]
        };
        if let Some(c) = &self.capture {
            c.store("embed_out", &x);
        }
        let mut state = plan_one(
            &self.context,
            R::Expand {
                batch: b,
                rows: s,
                streams: c,
                hidden: h,
            },
            &[&x],
        )?;

        let layer_count = self.layers.len();
        for li in model::layer_indices(self.layers.len(), stop_after_layer) {
            let layer = &mut self.layers[li];
            let _trace_layer = self.context.trace_detail("glm53.layer", li as u64 + 1);
            if self.context.trace_enabled("glm53.memory") {
                eprintln!(
                    "[glm53-memory] layer={li} rows={} private_bytes={}",
                    b * s,
                    self.context.device().memory_usage()?.private,
                );
            }
            let mut attn_inputs = None;
            let mut ffn_inputs = None;
            let mut attended_value = None;
            let mut ffn_value = None;
            let mut step_register = 0;
            let mut capture_mode_register = "dec";
            for action in model::LAYER {
                let step = step_register;
                let capture_mode = capture_mode_register;
                match action {
                    model::Step::AttentionInput => {
                        // Attention half.
                        if let Some(cp) = &self.capture
                            && cp.wants(li)
                        {
                            cp.store(&format!("L{li}.mhc_attn_in"), &state);
                        }
                        let capture_attn_mixes = self.capture.as_ref().is_some_and(|capture| {
                            capture.wants(li) && !capture.preserves_dispatch()
                        });
                        let (xc, x_in, post, comb, attn_mixes, attn_split) = {
                            let _trace_mhc = self.context.trace("glm53.mhc.attn_input");
                            if capture_attn_mixes {
                                if let Some((xc, normed, post, comb, mixes, split)) =
                                    layer.hc_attn.split_normed_with_mixes(
                                        &state,
                                        &layer.in_norm,
                                        self.config.rms_norm_eps,
                                    )?
                                {
                                    (xc, normed, post, comb, Some(mixes), Some(split))
                                } else {
                                    let (xc, post, comb) = layer.hc_attn.split(&state)?;
                                    let normed = rms(
                                        &self.context,
                                        &xc,
                                        &layer.in_norm,
                                        self.config.rms_norm_eps,
                                    )?;
                                    (xc, normed, post, comb, None, None)
                                }
                            } else if let Some((xc, normed, post, comb)) = layer
                                .hc_attn
                                .split_normed(&state, &layer.in_norm, self.config.rms_norm_eps)?
                            {
                                (xc, normed, post, comb, None, None)
                            } else {
                                let (xc, post, comb) = layer.hc_attn.split(&state)?;
                                let normed = rms(
                                    &self.context,
                                    &xc,
                                    &layer.in_norm,
                                    self.config.rms_norm_eps,
                                )?;
                                (xc, normed, post, comb, None, None)
                            }
                        };
                        if let Some(cp) = &self.capture
                            && cp.wants(li)
                        {
                            cp.store(&format!("L{li}.mhc_attn_collapsed"), &xc);
                            cp.store(&format!("L{li}.mhc_attn_post"), &post);
                            cp.store(&format!("L{li}.mhc_attn_comb"), &comb);
                            cp.store(&format!("L{li}.mhc_attn_normed"), &x_in);
                            if let Some(mixes) = &attn_mixes {
                                cp.store(&format!("L{li}.mhc_attn_mixes"), mixes);
                            }
                            if let Some(split) = &attn_split {
                                cp.store(&format!("L{li}.mhc_attn_split"), split);
                            }
                        }
                        // incremental-decode probe: norm of every stage, every layer
                        let step = if li == 0 {
                            INC_STEP.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
                        } else {
                            INC_STEP.load(std::sync::atomic::Ordering::Relaxed)
                        };
                        let capture_mode = if x_in.dim(1) == 1 { "dec" } else { "rep" };
                        if let Some(cp) = &self.capture
                            && cp.wants(li)
                        {
                            let t2 = if x_in.dim(1) == 1 { "dec" } else { "rep" };
                            cp.store(&format!("L{li}.{t2}_rms_in_s{step}"), &x_in);
                        }

                        attn_inputs = Some((xc, x_in, post, comb, attn_mixes, attn_split));
                        step_register = step;
                        capture_mode_register = capture_mode;
                    }
                    model::Step::Attention => {
                        let (xc, x_in, _, _, _, _) =
                            attn_inputs.as_ref().expect("attention input step");

                        let attended = {
                            let _trace_attn = self.context.trace("glm53.attention");
                            match (&layer.attn, &mut caches[li]) {
                                (Attn::Kda(k), LayerCache::Linear(g)) => {
                                    if prefill_mode {
                                        k.forward_prefill(
                                            x_in,
                                            &mut g.conv,
                                            &mut g.ssm,
                                            self.capture.as_ref(),
                                            li,
                                        )?
                                    } else if let Some(position) = forward_position {
                                        k.forward_at_position(
                                            x_in,
                                            &mut g.conv,
                                            &mut g.ssm,
                                            self.capture.as_ref(),
                                            li,
                                            position,
                                        )?
                                    } else {
                                        k.forward(
                                            x_in,
                                            &mut g.conv,
                                            &mut g.ssm,
                                            self.capture.as_ref(),
                                            li,
                                        )?
                                    }
                                }
                                (Attn::Dsa(d), LayerCache::Full(f)) => {
                                    d.forward(x_in, f, self.capture.as_ref(), li, prefill_mode)?
                                }
                                _ => anyhow::bail!("glm53: cache kind mismatch at layer {li}"),
                            }
                        };
                        if let Some(cp) = &self.capture
                            && cp.wants(li)
                        {
                            let t2 = if x_in.dim(1) == 1 { "dec" } else { "rep" };
                            cp.store(&format!("L{li}.{t2}_attn_out_s{step}"), &attended);
                            cp.store(&format!("L{li}.mhc_attn_xc"), xc);
                        }

                        attended_value = Some(attended);
                    }
                    model::Step::AttentionResidual => {
                        let (_, _, post, comb, _, _) =
                            attn_inputs.as_ref().expect("attention input step");

                        let attended = attended_value.as_ref().expect("attention step");
                        state = {
                            let _trace_expand = self.context.trace("glm53.mhc.attn_expand");
                            layer.hc_attn.expand(attended, &state, post, comb)?
                        };
                        if let Some(cp) = &self.capture
                            && cp.wants(li)
                        {
                            cp.store(&format!("L{li}.state_after_attn"), &state);
                            cp.store(
                                &format!("L{li}.{capture_mode}_state_after_attn_s{step}"),
                                &state,
                            );
                        }
                    }
                    model::Step::FfnInput => {
                        // FFN half.
                        let capture_ffn_mixes = self.capture.as_ref().is_some_and(|capture| {
                            capture.wants(li) && !capture.preserves_dispatch()
                        });
                        let (xc2, x_in2, post2, comb2, ffn_mixes, ffn_split) = {
                            let _trace_mhc = self.context.trace("glm53.mhc.ffn_input");
                            if capture_ffn_mixes {
                                if let Some((xc, normed, post, comb, mixes, split)) =
                                    layer.hc_ffn.split_normed_with_mixes(
                                        &state,
                                        &layer.post_norm,
                                        self.config.rms_norm_eps,
                                    )?
                                {
                                    (xc, normed, post, comb, Some(mixes), Some(split))
                                } else {
                                    let (xc, post, comb) = layer.hc_ffn.split(&state)?;
                                    let normed = rms(
                                        &self.context,
                                        &xc,
                                        &layer.post_norm,
                                        self.config.rms_norm_eps,
                                    )?;
                                    (xc, normed, post, comb, None, None)
                                }
                            } else if let Some((xc, normed, post, comb)) = layer
                                .hc_ffn
                                .split_normed(&state, &layer.post_norm, self.config.rms_norm_eps)?
                            {
                                (xc, normed, post, comb, None, None)
                            } else {
                                let (xc, post, comb) = layer.hc_ffn.split(&state)?;
                                let normed = rms(
                                    &self.context,
                                    &xc,
                                    &layer.post_norm,
                                    self.config.rms_norm_eps,
                                )?;
                                (xc, normed, post, comb, None, None)
                            }
                        };
                        if let Some(cp) = &self.capture
                            && cp.wants(li)
                        {
                            cp.store(&format!("L{li}.mhc_ffn_post"), &post2);
                            cp.store(&format!("L{li}.mhc_ffn_comb"), &comb2);
                            cp.store(&format!("L{li}.mhc_ffn_normed"), &x_in2);
                            if let Some(mixes) = &ffn_mixes {
                                cp.store(&format!("L{li}.mhc_ffn_mixes"), mixes);
                            }
                            if let Some(split) = &ffn_split {
                                cp.store(&format!("L{li}.mhc_ffn_split"), split);
                            }
                        }

                        ffn_inputs = Some((xc2, x_in2, post2, comb2, ffn_mixes, ffn_split));
                    }
                    model::Step::Ffn => {
                        let (_, x_in, _, _, _, _) =
                            attn_inputs.as_ref().expect("attention input step");

                        let (xc2, x_in2, _, _, _, _) = ffn_inputs.as_ref().expect("FFN input step");
                        let m = {
                            let _trace_ffn = self.context.trace("glm53.mlp");
                            match &layer.mlp {
                                Mlp::Dense(d) => {
                                    d.forward_with_capture(x_in2, self.capture.as_ref(), li)?
                                }
                                Mlp::Sparse(moe) => {
                                    moe.forward_with_capture(x_in2, self.capture.as_ref(), li)?
                                }
                            }
                        };
                        if let Some(cp) = &self.capture
                            && cp.wants(li)
                        {
                            let t2 = if x_in.dim(1) == 1 { "dec" } else { "rep" };
                            cp.store(&format!("L{li}.{t2}_moe_out_s{step}"), &m);
                            cp.store(&format!("L{li}.mhc_ffn_xc2"), xc2);
                        }

                        ffn_value = Some(m);
                    }
                    model::Step::FfnResidual => {
                        let (_, _, _, _, _, _) =
                            attn_inputs.as_ref().expect("attention input step");

                        let (_, _, post2, comb2, _, _) =
                            ffn_inputs.as_ref().expect("FFN input step");
                        let m = ffn_value.as_ref().expect("FFN step");
                        state = {
                            let _trace_expand = self.context.trace("glm53.mhc.ffn_expand");
                            layer.hc_ffn.expand(m, &state, post2, comb2)?
                        };
                        if let Some(cp) = &self.capture
                            && cp.wants(li)
                        {
                            cp.store(&format!("L{li}.state_after_ffn"), &state);
                            cp.store(
                                &format!("L{li}.{capture_mode}_state_after_ffn_s{step}"),
                                &state,
                            );
                        }
                        match model::submission(
                            b,
                            s,
                            prefill_mode,
                            self.prefill_final_chunk,
                            self.capture.as_ref().is_none_or(|c| c.preserves_dispatch()),
                            li,
                            layer_count,
                        ) {
                            model::Submission::None => {}
                            model::Submission::Flush => self.context.device().submit(false)?,
                            model::Submission::Drain => {
                                let _span = self.context.trace("glm53.prefill.layer_drain");
                                self.context.device().submit(true)?;
                            }
                        }
                        if stop_after_layer == Some(li) {
                            let mixed = plan_one(
                                &self.context,
                                R::PrefixMean {
                                    shape: state.shape().to_vec(),
                                },
                                &[&state],
                            )?;
                            let multi = reshape(
                                &self.context,
                                &state,
                                &[b as i32, s as i32, (c * h) as i32],
                            )?;
                            return Ok((mixed, multi));
                        }
                    }
                }
            }
        }

        if prefill_mode {
            let _span = self.context.trace("glm53.prefill.chunk_drain");
            self.context.device().submit(true)?;
        }
        // Final: mean over the streams, RMS, head.
        let mixed = final_hc_mean(&self.context, &state)?;
        let mixed = final_rms(
            &self.context,
            &mixed,
            &self.final_norm,
            self.config.rms_norm_eps,
        )?;
        if let Some(c) = &self.capture {
            let mode = if s == 1 { "dec" } else { "rep" };
            let step = INC_STEP.load(std::sync::atomic::Ordering::Relaxed);
            c.store(&format!("final_{mode}_mixed_s{step}"), &mixed);
        }
        let multi = reshape(
            &self.context,
            &state,
            &[(b) as i32, (s) as i32, (c * h) as i32],
        )?;
        Ok((mixed, multi))
    }

    pub fn prefill_chunks(
        &mut self,
        tokens: &[u32],
        caches: &mut [LayerCache<D>],
        retain_multi: bool,
    ) -> anyhow::Result<(D::Tensor, Option<D::Tensor>)> {
        anyhow::ensure!(!tokens.is_empty(), "empty GLM prefill");
        self.check_context_span(caches, tokens.len())?;

        // Full diagnostic captures intentionally retain the single-forward
        // layout. Normal sessions follow the reference's graph-selected 2048-token
        // boundary and carry the recurrent and indexer caches between calls.
        if self
            .capture
            .as_ref()
            .is_some_and(|capture| !capture.preserves_dispatch())
            || tokens.len() <= self.prefill_chunk_rows(caches, tokens.len())
        {
            let ids = i32_arr(&self.context, &self.embed.weight, tokens)?;
            let (mixed, multi) = self.forward_prefill_chunk(&ids, caches, true)?;
            let last = plan_one(
                &self.context,
                R::LastRow {
                    rows: mixed.dim(1) as usize,
                    hidden: self.config.hidden_size,
                },
                &[&mixed],
            )?;
            return Ok((last, retain_multi.then_some(multi)));
        }

        let rollback: Vec<_> = caches.iter().map(LayerCache::snapshot_rollback).collect();
        let result = (|| -> anyhow::Result<(D::Tensor, Option<D::Tensor>)> {
            let mut last = None;
            let mut multi_chunks = Vec::new();
            let mut done = 0;

            while done < tokens.len() {
                let rows = self.prefill_chunk_rows(caches, tokens.len() - done);
                let chunk = &tokens[done..done + rows];
                let final_chunk = done + rows == tokens.len();
                let ids = i32_arr(&self.context, &self.embed.weight, chunk)?;
                let (mixed, multi) = self.forward_prefill_chunk(&ids, caches, final_chunk)?;
                last = Some(plan_one(
                    &self.context,
                    R::LastRow {
                        rows: mixed.dim(1) as usize,
                        hidden: self.config.hidden_size,
                    },
                    &[&mixed],
                )?);
                if retain_multi {
                    multi_chunks.push(multi);
                }
                self.context.device().trim_unused();
                done += rows;
            }
            let last = last.expect("non-empty chunks");
            let multi = if retain_multi {
                let refs: Vec<&D::Tensor> = multi_chunks.iter().collect();
                Some(join(&self.context, &refs, 1)?)
            } else {
                None
            };
            Ok((last, multi))
        })();
        match result {
            Ok(value) => Ok(value),
            Err(error) => {
                for (cache, state) in caches.iter_mut().zip(&rollback) {
                    cache
                        .restore_rollback(state)
                        .map_err(|rollback_error| anyhow::anyhow!("{rollback_error}"))?;
                }
                Err(error)
            }
        }
    }

    pub fn prefill_chunk_rows(&self, caches: &[LayerCache<D>], remaining: usize) -> usize {
        let position = caches
            .iter()
            .find_map(|cache| match cache {
                LayerCache::Full(cache) => Some(cache.offset),
                LayerCache::Linear(_) | LayerCache::Native(_) => None,
            })
            .unwrap_or(0);
        model::prefill_rows(remaining, position, self.context_capacity)
    }

    pub fn check_context_span(&self, caches: &[LayerCache<D>], rows: usize) -> anyhow::Result<()> {
        for cache in caches {
            if let LayerCache::Full(cache) = cache {
                anyhow::ensure!(
                    cache
                        .offset
                        .checked_add(rows)
                        .is_some_and(|end| end <= self.context_capacity),
                    "GLM context overflow: append [{}, {}) exceeds capacity {}",
                    cache.offset,
                    cache.offset.saturating_add(rows),
                    self.context_capacity,
                );
            }
        }
        Ok(())
    }
}

pub fn i32_arr<D: BindingDevice>(
    context: &DeviceContext<D>,
    _rt_of: &D::Tensor,
    tokens: &[u32],
) -> anyhow::Result<D::Tensor> {
    let v: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
    crate::bindings::shared::tensor_io::pull_i32(context, &v, &[1, tokens.len() as i32])
}

impl<D: ExpertDevice + ReadbackDevice> crate::language::LanguageModel<Runtime<D, Sampler<D>>>
    for Tower<D>
where
    D::Error: From<anyhow::Error>,
{
    fn runtime(&self) -> &Runtime<D, Sampler<D>> {
        &self.runtime
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

    fn cache_numerical_law(&self) -> Option<u64> {
        // Read the projection modes retained by the loaded model. The separate
        // diagnostic schedule must not share caches with native paired fusion.
        let modes = self.layers.iter().filter_map(|layer| match &layer.mlp {
            Mlp::Dense(dense) => Some(dense.paired_storage()),
            Mlp::Sparse(moe) => moe.shared_paired_storage(),
        });
        let mtp_mode = self
            .mtp
            .as_ref()
            .and_then(|mtp| mtp.mlp.shared_paired_storage());
        Some(crate::bindings::glm53::law::cache_numerical_law(
            self.context.device(),
            model::loaded_pairing_law_bits(modes.chain(mtp_mode)),
        ))
    }

    fn max_position_embeddings(&self) -> usize {
        self.context_capacity
    }

    fn set_context_capacity(&mut self, capacity: usize) -> anyhow::Result<()> {
        anyhow::ensure!(
            capacity > 0 && capacity <= self.config.max_position_embeddings,
            "GLM context capacity {capacity} must be in 1..={}",
            self.config.max_position_embeddings,
        );
        self.context_capacity = capacity;
        Ok(())
    }

    fn eos_token_id(&self) -> i64 {
        self.config.eos_token_id
    }

    fn eos_token_ids(&self) -> Vec<i64> {
        self.config.eos_token_ids.clone()
    }

    fn new_caches(&self) -> Vec<LayerCache<D>> {
        (0..self.n_layers)
            .map(|i| {
                if self.config.is_kda(i) {
                    LayerCache::Linear(GdnCache::new())
                } else {
                    LayerCache::Full(FullAttentionCache::new(self.context.clone(), 1, 512))
                }
            })
            .collect()
    }

    fn forward(
        &mut self,
        ids: &D::Tensor,
        caches: Option<&mut Vec<LayerCache<D>>>,
    ) -> anyhow::Result<(D::Tensor, D::Tensor)> {
        match caches {
            Some(c) => self.forward_inner(ids, c),
            None => {
                let mut c = self.new_caches();
                self.forward_inner(ids, &mut c)
            }
        }
    }

    fn forward_ragged(
        &mut self,
        ids: &D::Tensor,
        per_row_caches: &mut [Vec<LayerCache<D>>],
    ) -> anyhow::Result<(D::Tensor, D::Tensor)> {
        self.forward_ragged_inner(ids, per_row_caches)
    }

    fn forward_capture(
        &mut self,
        ids: &D::Tensor,
        caches: Option<&mut Vec<LayerCache<D>>>,
        _capture: bool,
    ) -> anyhow::Result<(D::Tensor, D::Tensor)> {
        self.forward(ids, caches)
    }

    #[cfg(test)]
    fn set_debug_capture_for_test(&mut self, layers: Vec<usize>) -> anyhow::Result<()> {
        self.set_debug_capture_filtered_for_test(layers, Vec::new())
    }

    #[cfg(test)]
    fn set_debug_capture_filtered_for_test(
        &mut self,
        layers: Vec<usize>,
        key_filters: Vec<String>,
    ) -> anyhow::Result<()> {
        self.capture = Some(Capture {
            layers,
            key_filters,
            state_only: false,
            kda_prefill_only: false,
            preserve_dispatch: true,
            entries: std::sync::Mutex::new(std::collections::HashMap::new()),
            shapes: std::sync::Mutex::new(std::collections::HashMap::new()),
            flushed: std::sync::atomic::AtomicBool::new(false),
            ..Default::default()
        });
        Ok(())
    }

    #[cfg(test)]
    fn flush_debug_capture_for_test(&mut self, path: &str) -> anyhow::Result<()> {
        let capture = self
            .capture
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("debug capture was not installed"))?;
        capture.flush(path)?;
        capture
            .flushed
            .store(true, std::sync::atomic::Ordering::Release);
        Ok(())
    }

    fn enable_logit_capture(&mut self) -> anyhow::Result<()> {
        self.capture = Some(Capture {
            key_filters: vec!["final_logits".to_string()],
            preserve_dispatch: true,
            ..Default::default()
        });
        Ok(())
    }

    fn enable_stage_capture(
        &mut self,
        layers: Vec<usize>,
        mut keys: Vec<String>,
    ) -> anyhow::Result<()> {
        anyhow::ensure!(
            layers.iter().all(|&layer| layer < self.layers.len()),
            "stage capture layer exceeds the GLM trunk depth"
        );
        if !keys.is_empty() && !keys.iter().any(|key| key == "final_logits") {
            keys.push("final_logits".to_string());
        }
        self.capture = Some(Capture {
            layers,
            key_filters: keys,
            preserve_dispatch: true,
            ..Default::default()
        });
        Ok(())
    }

    fn write_logit_capture(&self, path: &str) -> anyhow::Result<()> {
        self.capture
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("logit capture is not enabled"))?
            .flush(path)
    }

    fn prefill(
        &mut self,
        tokens: &[u32],
        caches: &mut Vec<LayerCache<D>>,
    ) -> anyhow::Result<D::Tensor> {
        let (last, _) = self.prefill_chunks(tokens, caches, false)?;
        reshape(&self.context, &last, &[self.config.hidden_size as i32])
    }

    fn decode_token(
        &mut self,
        token: u32,
        caches: &mut Vec<LayerCache<D>>,
    ) -> anyhow::Result<D::Tensor> {
        // The reference's serial evaluator and MTP seed/replay use the same token graph.
        // A one-token prompt continuation still enters `prefill_chunks` above.
        let ids = i32_arr(&self.context, &self.embed.weight, &[token])?;
        let (mixed, _) = self.forward_inner(&ids, caches)?;
        reshape(&self.context, &mixed, &[self.config.hidden_size as i32])
    }

    fn prefill_multi(
        &mut self,
        tokens: &[u32],
        caches: &mut Vec<LayerCache<D>>,
    ) -> anyhow::Result<(D::Tensor, D::Tensor)> {
        let (last, multi) = self.prefill_chunks(tokens, caches, true)?;
        Ok((last, multi.expect("retained GLM prefill stream")))
    }

    fn head(&self, mixed: &D::Tensor) -> Result<D::Tensor, D::Error> {
        let logits = plans::head(&self.context, &self.lm_head, mixed)
            .map_err(|e| anyhow::Error::msg(e.to_string()))?;
        if let Some(capture) = self.capture.as_ref().filter(|c| c.preserves_dispatch()) {
            let step = INC_STEP.load(std::sync::atomic::Ordering::Relaxed);
            capture.store("final_logits", &logits);
            capture.store(&format!("final_logits_s{step}"), &logits);
        }
        Ok(logits)
    }

    fn head_ragged(
        &self,
        mixed: &D::Tensor,
        per_row_caches: &[Vec<LayerCache<D>>],
    ) -> Result<D::Tensor, D::Error> {
        let row_count = per_row_caches.len();
        let dense_compact_limit = self
            .config
            .index_topk
            .saturating_add(self.config.index_kpool.saturating_sub(1));
        let max_position = per_row_caches
            .iter()
            .flat_map(|row| row.iter())
            .filter_map(|cache| match cache {
                LayerCache::Full(cache) => Some(cache.offset),
                LayerCache::Linear(_) | LayerCache::Native(_) => None,
            })
            .max()
            .unwrap_or(0);
        // The caches have already advanced once. Subtract that just-fed token
        // to apply the reference's pre-forward eligibility condition to the output head.
        if ds4_native_session_batch_position_supported(
            row_count,
            max_position.saturating_sub(1),
            dense_compact_limit,
        ) {
            return self.head(mixed);
        }
        let mut logits_rows = Vec::with_capacity(row_count);
        for row in 0..row_count {
            logits_rows.push(self.head(&narrow(&self.context, mixed, 0, row, 1)?)?);
        }
        let logits_refs: Vec<&D::Tensor> = logits_rows.iter().collect();
        Ok(join(&self.context, &logits_refs, 0)?)
    }

    fn warmup(&mut self, _seed: &[u32]) -> anyhow::Result<()> {
        Ok(())
    }

    fn batch_decode_ok(&self) -> bool {
        // GLM batches preserve compact latent and indexer tapes through the
        // packed equal-position path or the row-local ragged cache path.
        true
    }

    fn batch_decode_equal_length_only(&self) -> bool {
        true
    }

    fn batch_decode_ragged_supported(&self) -> bool {
        true
    }

    fn has_target_mtp(&self) -> bool {
        self.mtp.is_some()
    }

    fn target_mtp_step(
        &mut self,
        cache: &mut IndexerTape<D>,
        target_multi: &D::Tensor,
        next_token: u32,
        position: usize,
        min_position: usize,
    ) -> anyhow::Result<D::Tensor> {
        self.mtp_step(cache, target_multi, next_token, position, min_position)
    }

    fn kv_bytes_per_token(&self) -> usize {
        model::kv_bytes_per_token(&self.config, self.n_layers)
    }

    fn stream_state_bytes(&self) -> usize {
        model::stream_state_bytes(&self.config, self.n_layers)
    }

    fn mtp_cost_key(&self) -> String {
        format!("glm53-{}L-{}h", self.n_layers, self.config.hidden_size)
    }
}

pub fn is_glm_routed_expert_tensor(name: &str) -> bool {
    let parts: Vec<&str> = name.split('.').collect();
    parts.len() == 8
        && matches!(parts[0], "model" | "mtp")
        && parts[1] == "layers"
        && parts[2].parse::<usize>().is_ok()
        && parts[3] == "mlp"
        && parts[4] == "experts"
        && parts[5].parse::<usize>().is_ok()
        && matches!(parts[6], "gate_proj" | "up_proj" | "down_proj")
        && matches!(parts[7], "weight" | "scales" | "biases")
}

pub mod ops;
