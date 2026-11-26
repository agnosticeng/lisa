//! Flash-Next checkpoint import and weight binding.
pub mod ple;
use crate::bindings::{
    qwen4::{
        attention::Attention,
        decoder::{Block, DecoderLayer},
        hyper::GatedResidual,
        ple::PleLayer,
        tower::Tower,
    },
    shared::{gated_delta::GatedDeltaNet, normalization::Rotary},
};
use crate::weights::{
    affine::{QuantizedEmbedding, QuantizedLinear},
    mlx::{Checkpoint, Shard, get_tensor},
    qwen4::ple::NgramTable,
    source::Weights,
};
use crate::{bindings::BindingDevice, language::runtime::Runtime, sampling::sampler::Sampler};
use lisa_backend::{ManagedDevice, ReadbackDevice};
use lisa_models::qwen4::{config::ModelConfig, tower as model};
use std::{
    collections::HashMap,
    path::Path,
    sync::{Arc, RwLock},
};

/// Load the whole tower from a model directory.
pub fn load_on<D: BindingDevice + ManagedDevice + ReadbackDevice>(
    runtime: Runtime<D, Sampler<D>>,
    dir: &Path,
    config: ModelConfig,
) -> anyhow::Result<Tower<D>>
where
    D::Error: From<anyhow::Error>,
{
    let context = runtime.context.clone();
    let checkpoint = Checkpoint::open(dir)?;
    let map = checkpoint.weight_map.clone();

    // The n-gram table is read from disk, never loaded as parameters.
    // Two checkpoint layouts: sharded tensors in the index (HF/MLX), or one
    // merged `ngram.safetensors` (the merged pack).
    let ngram_prefix = model::NGRAM_PREFIX;
    let sharded = map.keys().any(|n| n.contains("ngram_embedding.shard_"));
    let ngram_table: crate::bindings::qwen4::ple::RowSource<D> =
        Arc::new(RwLock::new(if sharded {
            NgramTable::open(dir, &map, ngram_prefix, config.split_ngram_parts)?
        } else {
            NgramTable::open_merged(&dir.join("ngram.safetensors"), config.split_ngram_parts)?
        }));

    // Stream the shards into a sanitized weight map, excluding the
    // n-gram shards (they stay on disk behind the row source) and the
    // vision tower (dropped by sanitize_name).
    let mut shard_order: Vec<String> = Vec::new();
    for shard in map.values() {
        if !shard_order.contains(shard) {
            shard_order.push(shard.clone());
        }
    }
    let mut expected: Vec<String> = map
        .keys()
        .filter_map(|n| lisa_models::checkpoint::sanitize_name(n))
        .collect();
    expected.sort();

    let mut weights: Weights<D::Tensor> = HashMap::new();
    for shard_name in &shard_order {
        let shard = Shard::open(context.device().clone(), &dir.join(shard_name))?;
        for name in shard.tensor_names() {
            if let Some(sanitized) = lisa_models::checkpoint::sanitize_name(name) {
                if weights.contains_key(&sanitized) {
                    continue;
                }
                let arr = get_tensor(&shard, name)?;
                weights.insert(sanitized, arr);
            }
        }
    }

    // t0: the full checkpoint materialized into the map — the baseline every
    // later ledger point is read against.
    crate::memory::report(context.device(), "t0-map-only");

    let missing: Vec<&String> = expected
        .iter()
        .filter(|n| !weights.contains_key(*n))
        .collect();
    if !missing.is_empty() {
        anyhow::bail!(
            "missing {} tensors after load, e.g. {:?}",
            missing.len(),
            &missing[..missing.len().min(5)]
        );
    }

    let mut tower = from_weights_on(runtime, weights, config, ngram_table)?;
    tower.vision = crate::weights::vision::load(context.clone(), dir, tower.rope.dimensions)?;
    Ok(tower)
}

/// Build the tower from a materialized weight map.
pub fn from_weights_on<D: BindingDevice + ManagedDevice + ReadbackDevice>(
    runtime: Runtime<D, Sampler<D>>,
    mut w: Weights<D::Tensor>,
    config: ModelConfig,
    ngram_table: crate::bindings::qwen4::ple::RowSource<D>,
) -> anyhow::Result<Tower<D>>
where
    D::Error: From<anyhow::Error>,
{
    let context = runtime.context.clone();
    let force_norm_operations = false;
    let hidden = config.hidden_size;
    let hc = config.hc_count;
    let eps = config.rms_norm_eps;

    let embed_tokens = QuantizedEmbedding::load(&mut w, model::EMBEDDING)?;
    let lm_head = QuantizedLinear::load(&mut w, model::HEAD, "")?;

    let ngram_table_for_ple = ngram_table.clone();
    let mut layers = Vec::with_capacity(config.num_hidden_layers);
    for description in model::layers(&config) {
        let block = if description.linear {
            Block::Linear(GatedDeltaNet::load(
                context.clone(),
                &mut w,
                &description.mixer,
                eps,
                config.linear_num_key_heads,
                config.linear_num_value_heads,
                config.linear_key_head_dim,
                config.linear_value_head_dim,
                config.linear_conv_kernel_dim,
                crate::weights::affine::GROUP_SIZE,
                crate::weights::affine::BITS,
            )?)
        } else {
            Block::Attention(Attention::load_on(
                context.clone(),
                &mut w,
                &description.mixer,
                eps,
                config.num_attention_heads,
                config.num_key_value_heads,
                config.head_dim,
            )?)
        };
        let mlp = crate::bindings::qwen4::moe::SparseMoeBlock::load_on(
            context.clone(),
            &mut w,
            &description.experts,
            config.num_experts_per_tok,
        )?;
        let attn_hc = GatedResidual::load_on(
            context.clone(),
            &mut w,
            &description.attn_hyper,
            hidden,
            hc,
            true,
            force_norm_operations,
        )?;
        let mlp_hc = GatedResidual::load_on(
            context.clone(),
            &mut w,
            &description.mlp_hyper,
            hidden,
            hc,
            true,
            force_norm_operations,
        )?;
        let ple = if let Some(ordinal) = description.ple_ordinal {
            Some(PleLayer::load_on(
                context.clone(),
                &mut w,
                &description.ple,
                eps,
                hidden,
                hc,
                config.ngram_size,
                config.heads_per_ngram,
                config.ple_embed_dim,
                config.ple_conv_kernel_size,
                ordinal, // index INTO ple_layer_ids (NOT the layer number)
                config.seed,
                config.vocab_size,
                config.ngram_vocab_size_base,
                config.make_ngram_vocab_size_divisible_by,
                config.split_ngram_parts,
                ngram_table_for_ple.clone(),
            )?)
        } else {
            None
        };
        let has_ple = ple.is_some();
        layers.push(DecoderLayer {
            context: context.clone(),
            block,
            mlp,
            attn_hc,
            mlp_hc,
            ple,
            has_ple,
        });
    }
    let final_mixer = GatedResidual::load_on(
        context.clone(),
        &mut w,
        model::FINAL_MIXER,
        hidden,
        hc,
        false,
        force_norm_operations,
    )?;
    crate::memory::report(context.device(), "t2-layers");

    // The embedded MTP head is a sibling of `model` in the checkpoint.
    let mtp = crate::bindings::qwen4::mtp::MtpHead::load_on(
        context.clone(),
        &mut w,
        &config,
        force_norm_operations,
    )?;
    crate::memory::report(context.device(), "t4-mtp-head");

    crate::memory::report(context.device(), "t5-map-check");
    let leftover: Vec<String> = w.keys().cloned().collect();
    if !leftover.is_empty() {
        anyhow::bail!(
            "{} weight tensors unused, e.g. {:?}",
            leftover.len(),
            &leftover[..leftover.len().min(5)]
        );
    }

    let mut tower = Tower {
        runtime,
        context,
        embed_tokens,
        layers,
        final_mixer,
        lm_head,
        rope: Rotary::new(config.rotary_dimensions(), config.rope_theta),
        eos_token_id: config.eos_token_id,
        config,
        ngram_history: None,
        mtp: Some(mtp),
        vision: None,
        warmed: false,
    };
    for layer in tower.layers.iter_mut() {
        if let Some(ple) = layer.ple.as_mut() {
            ple.set_eos(tower.eos_token_id);
        }
    }
    Ok(tower)
}
