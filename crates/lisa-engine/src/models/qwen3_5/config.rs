//! `qwen3_5_text` configuration (Qwen 3.8 27B).

use serde::Deserialize;

#[derive(Debug, Clone, Deserialize, Default)]
struct RopeParameters {
    #[serde(rename = "rope_theta", default)]
    rope_theta: Option<f32>,
    #[serde(rename = "partial_rotary_factor", default)]
    partial_rotary_factor: Option<f32>,
}

#[derive(Debug, Clone, Deserialize, Default)]
struct RawText {
    hidden_size: Option<usize>,
    num_hidden_layers: Option<usize>,
    num_attention_heads: Option<usize>,
    num_key_value_heads: Option<usize>,
    head_dim: Option<usize>,
    vocab_size: Option<usize>,
    rms_norm_eps: Option<f32>,
    layer_types: Option<Vec<String>>,
    full_attention_interval: Option<usize>,
    intermediate_size: Option<usize>,
    linear_num_key_heads: Option<usize>,
    linear_num_value_heads: Option<usize>,
    linear_key_head_dim: Option<usize>,
    linear_value_head_dim: Option<usize>,
    linear_conv_kernel_dim: Option<usize>,
    output_gate_type: Option<String>,
    attn_output_gate: Option<bool>,
    mtp_num_hidden_layers: Option<usize>,
    mtp_use_dedicated_embeddings: Option<bool>,
    eos_token_id: Option<serde_json::Value>,
    partial_rotary_factor: Option<f32>,
    rope_theta: Option<f32>,
    max_position_embeddings: Option<usize>,
    tie_word_embeddings: Option<bool>,
    rope_parameters: Option<RopeParameters>,
}

#[derive(Debug, Clone, Deserialize, Default)]
struct RawQuant {
    group_size: Option<i32>,
    bits: Option<i32>,
}

#[derive(Debug, Clone, Deserialize, Default)]
struct RawConfig {
    #[serde(default)]
    text_config: Option<RawText>,
    #[serde(default)]
    root: RawText,
    #[serde(rename = "quantization", default)]
    quantization: Option<RawQuant>,
    #[serde(rename = "quantization_config", default)]
    quantization_config: Option<RawQuant>,
}

#[derive(Debug, Clone)]
pub struct Qwen35Config {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub rms_norm_eps: f32,
    pub full_attention_interval: usize,
    pub intermediate_size: usize,
    pub layer_types: Vec<String>,
    pub linear_num_key_heads: usize,
    pub linear_num_value_heads: usize,
    pub linear_key_head_dim: usize,
    pub linear_value_head_dim: usize,
    pub linear_conv_kernel_dim: usize,
    pub attn_output_gate: bool,
    pub output_gate_type: String,
    pub mtp_num_hidden_layers: usize,
    pub mtp_use_dedicated_embeddings: bool,
    pub quant_group_size: i32,
    pub quant_bits: i32,
    pub eos_token_id: i64,
    pub partial_rotary_factor: f32,
    pub rope_theta: f32,
    pub max_position_embeddings: usize,
    pub tie_word_embeddings: bool,
}

impl Qwen35Config {
    pub fn from_json(path: &std::path::Path) -> anyhow::Result<Self> {
        Ok(Self::from_str(&std::fs::read_to_string(path)?)?)
    }

    pub fn from_str(data: &str) -> anyhow::Result<Self> {
        let raw: RawConfig = serde_json::from_str(data)?;
        let t = raw.text_config.clone().unwrap_or_default();
        let root = &raw.root;
        // text_config wins; root is the fallback (transformed flat layouts).
        let g = |f: fn(&RawText) -> Option<usize>| f(&t).or_else(|| f(root));
        let hidden_size = g(|x| x.hidden_size).unwrap_or(5120);
        let num_hidden_layers = g(|x| x.num_hidden_layers).unwrap_or(64);
        let full_attention_interval = g(|x| x.full_attention_interval).unwrap_or(4);
        let layer_types = t
            .layer_types
            .clone()
            .or_else(|| root.layer_types.clone())
            .filter(|v| !v.is_empty())
            .unwrap_or_else(|| {
                (0..num_hidden_layers)
                    .map(|i| {
                        if (i + 1) % full_attention_interval == 0 {
                            "full_attention".into()
                        } else {
                            "linear_attention".into()
                        }
                    })
                    .collect()
            });
        let eos = match g(|x| x.eos_token_id.clone().map(|_| 0)) {
            _ => t
                .eos_token_id
                .clone()
                .or_else(|| root.eos_token_id.clone()),
        };
        let eos_token_id = match eos {
            Some(serde_json::Value::Number(n)) => n.as_i64().unwrap_or(248046),
            Some(serde_json::Value::Array(a)) => {
                a.first().and_then(|v| v.as_i64()).unwrap_or(248046)
            }
            _ => 248046,
        };
        let rp = t.rope_parameters.clone().or_else(|| root.rope_parameters.clone());
        let mut rope_theta = t
            .rope_theta
            .or(root.rope_theta)
            .unwrap_or(10_000_000.0);
        let mut partial_rotary_factor = t
            .partial_rotary_factor
            .or(root.partial_rotary_factor)
            .unwrap_or(0.25);
        if let Some(rp) = &rp {
            if let Some(v) = rp.rope_theta {
                rope_theta = v;
            }
            if let Some(v) = rp.partial_rotary_factor {
                partial_rotary_factor = v;
            }
        }
        let q = raw
            .quantization
            .clone()
            .or(raw.quantization_config.clone())
            .unwrap_or_default();
        Ok(Self {
            hidden_size,
            num_hidden_layers,
            num_attention_heads: g(|x| x.num_attention_heads).unwrap_or(24),
            num_key_value_heads: g(|x| x.num_key_value_heads).unwrap_or(4),
            head_dim: g(|x| x.head_dim).unwrap_or(256),
            vocab_size: g(|x| x.vocab_size).unwrap_or(248_320),
            rms_norm_eps: t.rms_norm_eps.or(root.rms_norm_eps).unwrap_or(1e-6),
            full_attention_interval,
            intermediate_size: g(|x| x.intermediate_size).unwrap_or(17408),
            layer_types,
            linear_num_key_heads: g(|x| x.linear_num_key_heads).unwrap_or(16),
            linear_num_value_heads: g(|x| x.linear_num_value_heads).unwrap_or(48),
            linear_key_head_dim: g(|x| x.linear_key_head_dim).unwrap_or(128),
            linear_value_head_dim: g(|x| x.linear_value_head_dim).unwrap_or(128),
            linear_conv_kernel_dim: g(|x| x.linear_conv_kernel_dim).unwrap_or(4),
            attn_output_gate: t.attn_output_gate.or(root.attn_output_gate).unwrap_or(true),
            output_gate_type: t
                .output_gate_type
                .clone()
                .or_else(|| root.output_gate_type.clone())
                .unwrap_or_else(|| "swish".into()),
            mtp_num_hidden_layers: t
                .mtp_num_hidden_layers
                .or(root.mtp_num_hidden_layers)
                .unwrap_or(1),
            mtp_use_dedicated_embeddings: t
                .mtp_use_dedicated_embeddings
                .or(root.mtp_use_dedicated_embeddings)
                .unwrap_or(false),
            quant_group_size: q.group_size.unwrap_or(64),
            quant_bits: q.bits.unwrap_or(4),
            eos_token_id,
            partial_rotary_factor,
            rope_theta,
            max_position_embeddings: g(|x| x.max_position_embeddings).unwrap_or(262_144),
            tie_word_embeddings: t
                .tie_word_embeddings
                .or(root.tie_word_embeddings)
                .unwrap_or(false),
        })
    }

    pub fn rotary_dimensions(&self) -> usize {
        ((self.head_dim as f32) * self.partial_rotary_factor).max(1.0) as usize
    }

    pub fn is_full(&self, layer: usize) -> bool {
        self.layer_types
            .get(layer)
            .map(|t| t == "full_attention")
            .unwrap_or(false)
    }
}
