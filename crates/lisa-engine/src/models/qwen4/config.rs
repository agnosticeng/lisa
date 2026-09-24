//! Model configuration (`qwen4_exp` / `qwen4_exp_text`).

use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct RopeParameters {
    #[serde(rename = "rope_theta", default)]
    pub rope_theta: Option<f32>,
    #[serde(rename = "partial_rotary_factor", default)]
    pub partial_rotary_factor: Option<f32>,
}

/// Flattened text-tower configuration. Accepts both the raw HF layout
/// (`text_config` nested) and the transformed layout (flattened at the root).
#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub rms_norm_eps: f32,
    pub rms_norm_weight_offset: f32,
    pub layer_types: Vec<String>,
    pub full_attention_interval: usize,
    pub num_experts: usize,
    pub num_experts_per_tok: usize,
    pub moe_intermediate_size: usize,
    pub shared_expert_intermediate_size: usize,
    pub linear_num_key_heads: usize,
    pub linear_num_value_heads: usize,
    pub linear_key_head_dim: usize,
    pub linear_value_head_dim: usize,
    pub linear_conv_kernel_dim: usize,
    pub output_gate_type: String,
    pub hc_count: usize,
    pub hc_lowrank: usize,
    pub indexer_heads: usize,
    pub indexer_kv_heads: usize,
    pub indexer_head_dim: usize,
    pub indexer_budget: usize,
    pub indexer_compress_ratio: usize,
    pub ngram_size: usize,
    pub heads_per_ngram: usize,
    pub ngram_vocab_size_base: usize,
    pub make_ngram_vocab_size_divisible_by: usize,
    pub split_ngram_parts: usize,
    pub ple_embed_dim: usize,
    pub ple_layer_ids: Vec<usize>,
    pub ple_conv_kernel_size: usize,
    pub seed: i64,
    pub eos_token_id: i64,
    pub partial_rotary_factor: f32,
    pub rope_theta: f32,
    pub max_position_embeddings: usize,
    pub tie_word_embeddings: bool,
}

// Fields we read from either the root or `text_config`.
#[derive(Debug, Deserialize)]
struct RawText {
    #[serde(rename = "hidden_size", default)]
    pub hidden_size: Option<usize>,
    #[serde(rename = "num_hidden_layers", default)]
    pub num_hidden_layers: Option<usize>,
    #[serde(rename = "num_attention_heads", default)]
    pub num_attention_heads: Option<usize>,
    #[serde(rename = "num_key_value_heads", default)]
    pub num_key_value_heads: Option<usize>,
    #[serde(rename = "head_dim", default)]
    pub head_dim: Option<usize>,
    #[serde(rename = "vocab_size", default)]
    pub vocab_size: Option<usize>,
    #[serde(rename = "rms_norm_eps", default)]
    pub rms_norm_eps: Option<f32>,
    #[serde(rename = "rms_norm_weight_offset", default)]
    pub rms_norm_weight_offset: Option<f32>,
    #[serde(rename = "layer_types", default)]
    pub layer_types: Option<Vec<String>>,
    #[serde(rename = "full_attention_interval", default)]
    pub full_attention_interval: Option<usize>,
    #[serde(rename = "num_experts", default)]
    pub num_experts: Option<usize>,
    #[serde(rename = "num_experts_per_tok", default)]
    pub num_experts_per_tok: Option<usize>,
    #[serde(rename = "moe_intermediate_size", default)]
    pub moe_intermediate_size: Option<usize>,
    #[serde(rename = "shared_expert_intermediate_size", default)]
    pub shared_expert_intermediate_size: Option<usize>,
    #[serde(rename = "linear_num_key_heads", default)]
    pub linear_num_key_heads: Option<usize>,
    #[serde(rename = "linear_num_value_heads", default)]
    pub linear_num_value_heads: Option<usize>,
    #[serde(rename = "linear_key_head_dim", default)]
    pub linear_key_head_dim: Option<usize>,
    #[serde(rename = "linear_value_head_dim", default)]
    pub linear_value_head_dim: Option<usize>,
    #[serde(rename = "linear_conv_kernel_dim", default)]
    pub linear_conv_kernel_dim: Option<usize>,
    #[serde(rename = "output_gate_type", default)]
    pub output_gate_type: Option<String>,
    #[serde(rename = "hc_count", default)]
    pub hc_count: Option<usize>,
    #[serde(rename = "hc_lowrank", default)]
    pub hc_lowrank: Option<usize>,
    #[serde(rename = "indexer_n_heads", default)]
    pub indexer_heads: Option<usize>,
    #[serde(rename = "indexer_kv_heads", default)]
    pub indexer_kv_heads: Option<usize>,
    #[serde(rename = "indexer_head_dim", default)]
    pub indexer_head_dim: Option<usize>,
    #[serde(rename = "indexer_budget", default)]
    pub indexer_budget: Option<usize>,
    #[serde(rename = "indexer_compress_ratio", default)]
    pub indexer_compress_ratio: Option<usize>,
    #[serde(rename = "ngram_size", default)]
    pub ngram_size: Option<usize>,
    #[serde(rename = "heads_per_ngram", default)]
    pub heads_per_ngram: Option<usize>,
    #[serde(rename = "ngram_vocab_size_base", default)]
    pub ngram_vocab_size_base: Option<usize>,
    #[serde(rename = "make_ngram_vocab_size_divisible_by", default)]
    pub make_ngram_vocab_size_divisible_by: Option<usize>,
    #[serde(rename = "split_ngram_parts", default)]
    pub split_ngram_parts: Option<usize>,
    #[serde(rename = "ple_embed_dim", default)]
    pub ple_embed_dim: Option<usize>,
    #[serde(rename = "ple_layer_ids", default)]
    pub ple_layer_ids: Option<Vec<usize>>,
    #[serde(rename = "ple_conv_kernel_size", default)]
    pub ple_conv_kernel_size: Option<usize>,
    #[serde(default)]
    pub seed: Option<i64>,
    #[serde(rename = "eos_token_id", default)]
    pub eos_token_id: Option<serde_json::Value>,
    #[serde(rename = "partial_rotary_factor", default)]
    pub partial_rotary_factor: Option<f32>,
    #[serde(rename = "rope_theta", default)]
    pub rope_theta: Option<f32>,
    #[serde(rename = "max_position_embeddings", default)]
    pub max_position_embeddings: Option<usize>,
    #[serde(rename = "tie_word_embeddings", default)]
    pub tie_word_embeddings: Option<bool>,
    #[serde(rename = "rope_parameters", default)]
    pub rope_parameters: Option<RopeParameters>,
}

#[derive(Debug, Deserialize)]
struct RawConfig {
    #[serde(rename = "text_config", default)]
    text_config: Option<RawText>,
    #[serde(flatten)]
    root: RawText,
}

impl ModelConfig {
    pub fn from_json(path: &std::path::Path) -> anyhow::Result<Self> {
        let data = std::fs::read_to_string(path)?;
        Self::from_str(&data)
    }

    pub fn from_str(data: &str) -> anyhow::Result<Self> {
        let raw: RawConfig = serde_json::from_str(data)?;
        // The nested text_config wins when present; the root is the fallback
        // (and the transformed tree's only source).
        let merged = MergedText::merge(raw.text_config.as_ref(), &raw.root);
        Ok(Self::from_merged(&merged))
    }

    fn from_merged(m: &MergedText) -> Self {
        let hidden_size = m.hidden_size.unwrap_or(2560);
        let num_hidden_layers = m.num_hidden_layers.unwrap_or(48);
        let full_attention_interval = m.full_attention_interval.unwrap_or(4);
        let layer_types = match m.layer_types {
            Some(ref types) if !types.is_empty() => types.clone(),
            _ => (0..num_hidden_layers)
                .map(|i| {
                    if (i + 1) % full_attention_interval == 0 {
                        "full_attention".to_string()
                    } else {
                        "linear_attention".to_string()
                    }
                })
                .collect(),
        };
        let eos_token_id = match &m.eos_token_id {
            Some(serde_json::Value::Number(n)) => n.as_i64().unwrap_or(248_044),
            Some(serde_json::Value::Array(a)) => a
                .first()
                .and_then(|v| v.as_i64())
                .unwrap_or(248_044),
            _ => 248_044,
        };
        // rope_parameters wins over the top level for both fields.
        let mut rope_theta = m.rope_theta.unwrap_or(10_000_000.0);
        let mut partial_rotary_factor = m.partial_rotary_factor.unwrap_or(0.25);
        if let Some(rp) = &m.rope_parameters {
            if let Some(t) = rp.rope_theta {
                rope_theta = t;
            }
            if let Some(p) = rp.partial_rotary_factor {
                partial_rotary_factor = p;
            }
        }

        Self {
            hidden_size,
            num_hidden_layers,
            num_attention_heads: m.num_attention_heads.unwrap_or(24),
            num_key_value_heads: m.num_key_value_heads.unwrap_or(2),
            head_dim: m.head_dim.unwrap_or(256),
            vocab_size: m.vocab_size.unwrap_or(248_320),
            rms_norm_eps: m.rms_norm_eps.unwrap_or(1e-6),
            rms_norm_weight_offset: m.rms_norm_weight_offset.unwrap_or(0.0),
            layer_types,
            full_attention_interval,
            num_experts: m.num_experts.unwrap_or(512),
            num_experts_per_tok: m.num_experts_per_tok.unwrap_or(10),
            moe_intermediate_size: m.moe_intermediate_size.unwrap_or(640),
            shared_expert_intermediate_size: m.shared_expert_intermediate_size.unwrap_or(640),
            linear_num_key_heads: m.linear_num_key_heads.unwrap_or(16),
            linear_num_value_heads: m.linear_num_value_heads.unwrap_or(48),
            linear_key_head_dim: m.linear_key_head_dim.unwrap_or(128),
            linear_value_head_dim: m.linear_value_head_dim.unwrap_or(128),
            linear_conv_kernel_dim: m.linear_conv_kernel_dim.unwrap_or(4),
            output_gate_type: m
                .output_gate_type
                .clone()
                .unwrap_or_else(|| "sigmoid".to_string()),
            hc_count: m.hc_count.unwrap_or(4),
            hc_lowrank: m.hc_lowrank.unwrap_or(320),
            indexer_heads: m.indexer_heads.unwrap_or(4),
            indexer_kv_heads: m.indexer_kv_heads.unwrap_or(1),
            indexer_head_dim: m.indexer_head_dim.unwrap_or(128),
            indexer_budget: m.indexer_budget.unwrap_or(2048),
            indexer_compress_ratio: m.indexer_compress_ratio.unwrap_or(4),
            ngram_size: m.ngram_size.unwrap_or(3),
            heads_per_ngram: m.heads_per_ngram.unwrap_or(8),
            ngram_vocab_size_base: m.ngram_vocab_size_base.unwrap_or(20_000_000),
            make_ngram_vocab_size_divisible_by: m.make_ngram_vocab_size_divisible_by.unwrap_or(128),
            split_ngram_parts: m.split_ngram_parts.unwrap_or(128),
            ple_embed_dim: m.ple_embed_dim.unwrap_or(2560),
            ple_layer_ids: m.ple_layer_ids.clone().unwrap_or_else(|| vec![2]),
            ple_conv_kernel_size: m.ple_conv_kernel_size.unwrap_or(4),
            seed: m.seed.unwrap_or(0),
            eos_token_id,
            partial_rotary_factor,
            rope_theta,
            max_position_embeddings: m.max_position_embeddings.unwrap_or(262_144),
            tie_word_embeddings: m.tie_word_embeddings.unwrap_or(false),
        }
    }

    /// Dimensions receiving rotary embedding (first quarter of each head).
    pub fn rotary_dimensions(&self) -> usize {
        ((self.head_dim as f32) * self.partial_rotary_factor).max(1.0) as usize
    }

    /// One PLE layer index per entry of `ple_layer_ids`, which is 1-based.
    pub fn ple_layer_indices(&self) -> Vec<usize> {
        self.ple_layer_ids.iter().map(|i| i - 1).collect()
    }
}

/// Field-wise merge: nested text_config fields win when present, root fills gaps.
struct MergedText {
    hidden_size: Option<usize>,
    num_hidden_layers: Option<usize>,
    num_attention_heads: Option<usize>,
    num_key_value_heads: Option<usize>,
    head_dim: Option<usize>,
    vocab_size: Option<usize>,
    rms_norm_eps: Option<f32>,
    rms_norm_weight_offset: Option<f32>,
    layer_types: Option<Vec<String>>,
    full_attention_interval: Option<usize>,
    num_experts: Option<usize>,
    num_experts_per_tok: Option<usize>,
    moe_intermediate_size: Option<usize>,
    shared_expert_intermediate_size: Option<usize>,
    linear_num_key_heads: Option<usize>,
    linear_num_value_heads: Option<usize>,
    linear_key_head_dim: Option<usize>,
    linear_value_head_dim: Option<usize>,
    linear_conv_kernel_dim: Option<usize>,
    output_gate_type: Option<String>,
    hc_count: Option<usize>,
    hc_lowrank: Option<usize>,
    indexer_heads: Option<usize>,
    indexer_kv_heads: Option<usize>,
    indexer_head_dim: Option<usize>,
    indexer_budget: Option<usize>,
    indexer_compress_ratio: Option<usize>,
    ngram_size: Option<usize>,
    heads_per_ngram: Option<usize>,
    ngram_vocab_size_base: Option<usize>,
    make_ngram_vocab_size_divisible_by: Option<usize>,
    split_ngram_parts: Option<usize>,
    ple_embed_dim: Option<usize>,
    ple_layer_ids: Option<Vec<usize>>,
    ple_conv_kernel_size: Option<usize>,
    seed: Option<i64>,
    eos_token_id: Option<serde_json::Value>,
    partial_rotary_factor: Option<f32>,
    rope_theta: Option<f32>,
    max_position_embeddings: Option<usize>,
    tie_word_embeddings: Option<bool>,
    rope_parameters: Option<RopeParameters>,
}

impl MergedText {
    fn merge(nested: Option<&RawText>, root: &RawText) -> MergedText {
        macro_rules! pick {
            ($($field:ident),* $(,)?) => {
                MergedText {
                    $($field: nested.and_then(|n| n.$field.clone()).or(root.$field.clone()),)*
                }
            };
        }
        pick!(
            hidden_size,
            num_hidden_layers,
            num_attention_heads,
            num_key_value_heads,
            head_dim,
            vocab_size,
            rms_norm_eps,
            rms_norm_weight_offset,
            layer_types,
            full_attention_interval,
            num_experts,
            num_experts_per_tok,
            moe_intermediate_size,
            shared_expert_intermediate_size,
            linear_num_key_heads,
            linear_num_value_heads,
            linear_key_head_dim,
            linear_value_head_dim,
            linear_conv_kernel_dim,
            output_gate_type,
            hc_count,
            hc_lowrank,
            indexer_heads,
            indexer_kv_heads,
            indexer_head_dim,
            indexer_budget,
            indexer_compress_ratio,
            ngram_size,
            heads_per_ngram,
            ngram_vocab_size_base,
            make_ngram_vocab_size_divisible_by,
            split_ngram_parts,
            ple_embed_dim,
            ple_layer_ids,
            ple_conv_kernel_size,
            seed,
            eos_token_id,
            partial_rotary_factor,
            rope_theta,
            max_position_embeddings,
            tie_word_embeddings,
            rope_parameters,
        )
    }
}
