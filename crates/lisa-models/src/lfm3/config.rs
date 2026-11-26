//! Validated, device-independent LFM3 checkpoint configuration.
use anyhow::{Result, ensure};
use serde::Deserialize;
use std::collections::HashMap;
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Layer {
    Conv,
    FullAttention,
    SlidingAttention,
}
#[derive(Clone, Debug, Deserialize)]
pub struct Rope {
    pub rope_theta: f32,
    #[serde(default = "one")]
    pub partial_rotary_factor: f32,
    #[serde(default)]
    pub rope_type: Option<String>,
}
fn one() -> f32 {
    1.0
}
#[derive(Clone, Debug, Deserialize)]
pub struct Quantization {
    pub bits: i32,
    pub group_size: i32,
    pub mode: String,
}
#[derive(Clone, Debug, Deserialize)]
pub struct Config {
    pub model_type: String,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub norm_eps: f32,
    #[serde(rename = "conv_L_cache")]
    pub conv_length: usize,
    pub conv_bias: bool,
    pub use_sandwich_norm: bool,
    pub tie_word_embeddings: bool,
    pub max_position_embeddings: usize,
    pub sliding_window: usize,
    pub eos_token_id: Vec<i64>,
    pub layer_types: Vec<Layer>,
    pub rope_parameters: HashMap<String, Rope>,
    pub quantization: Quantization,
}
impl Config {
    pub fn parse(bytes: &[u8]) -> Result<Self> {
        let c: Self = serde_json::from_slice(bytes)?;
        c.validate()?;
        Ok(c)
    }
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.model_type == "lfm3",
            "expected an LFM3 text checkpoint"
        );
        ensure!(
            self.quantization.bits == 4
                && self.quantization.group_size == 64
                && self.quantization.mode == "affine",
            "LFM3 requires original affine 4-bit/group-64 weights"
        );
        ensure!(
            !self.conv_bias && self.use_sandwich_norm && self.tie_word_embeddings,
            "unsupported LFM3 bias, norm or embedding layout"
        );
        ensure!(
            self.num_hidden_layers > 0 && self.layer_types.len() == self.num_hidden_layers,
            "LFM3 layer count differs"
        );
        ensure!(
            self.hidden_size > 0
                && self.intermediate_size > 0
                && self.num_key_value_heads > 0
                && self.num_attention_heads > 0
                && self.head_dim > 0
                && self.head_dim.is_multiple_of(2),
            "invalid LFM3 dimensions"
        );
        ensure!(
            self.hidden_size == self.num_attention_heads * self.head_dim
                && self
                    .num_attention_heads
                    .is_multiple_of(self.num_key_value_heads),
            "invalid LFM3 attention heads"
        );
        ensure!(
            self.hidden_size.is_multiple_of(512)
                && self.intermediate_size.is_multiple_of(512)
                && self.vocab_size.is_multiple_of(32),
            "unsupported LFM3 projection dimensions"
        );
        ensure!(
            self.conv_length > 1
                && self.conv_length <= 64
                && self.sliding_window > 0
                && self.max_position_embeddings > 0
                && self.max_position_embeddings <= i32::MAX as usize,
            "invalid LFM3 cache dimensions"
        );
        ensure!(
            self.norm_eps.is_finite()
                && self.norm_eps > 0.0
                && !self.eos_token_id.is_empty()
                && self
                    .eos_token_id
                    .iter()
                    .all(|&v| v >= 0 && v < self.vocab_size as i64),
            "invalid LFM3 norm or stop tokens"
        );
        for name in ["full_attention", "sliding_attention"] {
            let r = self
                .rope_parameters
                .get(name)
                .ok_or_else(|| anyhow::anyhow!("missing {name} rotary parameters"))?;
            ensure!(
                r.rope_type.as_deref().is_none_or(|v| v == "default")
                    && r.rope_theta.is_finite()
                    && r.rope_theta > 0.0
                    && r.partial_rotary_factor.is_finite()
                    && r.partial_rotary_factor > 0.0
                    && r.partial_rotary_factor <= 1.0
                    && (self.head_dim as f32 * r.partial_rotary_factor) as usize >= 2,
                "unsupported LFM3 rotary parameters"
            );
        }
        Ok(())
    }
    pub fn rope(&self, layer: Layer) -> &Rope {
        &self.rope_parameters[if layer == Layer::SlidingAttention {
            "sliding_attention"
        } else {
            "full_attention"
        }]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn checkpoint_contract_and_unsupported_layouts() {
        let bytes = include_bytes!("fixtures/config.json");
        let c = Config::parse(bytes).unwrap();
        assert_eq!(
            c.layer_types.iter().filter(|&&k| k == Layer::Conv).count(),
            13
        );
        assert_eq!(
            c.layer_types
                .iter()
                .filter(|&&k| k == Layer::SlidingAttention)
                .count(),
            5
        );
        for (key, value) in [
            ("conv_bias", serde_json::json!(true)),
            ("tie_word_embeddings", serde_json::json!(false)),
            ("num_hidden_layers", serde_json::json!(23)),
            ("sliding_window", serde_json::json!(0)),
        ] {
            let mut v: serde_json::Value = serde_json::from_slice(bytes).unwrap();
            v[key] = value;
            assert!(
                Config::parse(&serde_json::to_vec(&v).unwrap()).is_err(),
                "{key}"
            );
        }
        let mut c = c;
        c.quantization.bits = 8;
        assert!(c.validate().is_err());
    }
}
