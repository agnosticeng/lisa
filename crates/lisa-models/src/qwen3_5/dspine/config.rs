//! Checkpoint contract and validation; input bytes are supplied by the engine.
use anyhow::{Context, ensure};
use serde::Deserialize;
const TARGET_LAYERS: [usize; 5] = [1, 16, 31, 46, 61];
const SERVED_BLOCK_SIZE: usize = 7;
#[derive(Debug, Deserialize)]
pub struct DSpineConfig {
    #[serde(default)]
    pub quantization: Option<DraftQuantization>,
    pub architectures: Vec<String>,
    pub model_type: String,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    #[serde(default)]
    pub layer_types: Vec<String>,
    #[serde(default)]
    pub sliding_window: Option<usize>,
    #[serde(default)]
    pub use_sliding_window: bool,
    #[serde(default)]
    pub is_causal: bool,
    pub block_size: usize,
    #[serde(rename = "dflash_config")]
    pub dflash: DSpineSubconfig,
}

/// MLX affine quantization applies to attention/MLP projections and fc only.
/// Bridge, normalization and codebook tensors retain their original BF16 data.
#[derive(Debug, Deserialize)]
pub struct DraftQuantization {
    pub bits: i32,
    pub group_size: i32,
    pub mode: String,
}

#[derive(Debug, Deserialize)]
pub struct DSpineSubconfig {
    pub mask_token_id: usize,
    pub target_layer_ids: Vec<usize>,
    pub selected_prefix_code_bridge: DSpineBridgeConfig,
    pub selected_prefix_feature_layers: Vec<usize>,
    pub selected_prefix_feature_writes: bool,
    pub selected_prefix_feature_write_mode: String,
    pub selected_prefix_terminal: bool,
    pub selected_prefix_decode_mode: String,
    pub draft_block_causal: bool,
    pub selected_prefix_last_write_refine: DSpineRefineConfig,
}

#[derive(Debug, Deserialize)]
pub struct DSpineBridgeConfig {
    pub code_size: usize,
    pub rank: usize,
    pub eps: f32,
}

#[derive(Debug, Deserialize)]
pub struct DSpineRefineConfig {
    pub kind: String,
    pub head_topk: usize,
}

impl DSpineConfig {
    pub fn parse(bytes: &[u8], target_vocab: usize, target_hidden: usize) -> anyhow::Result<Self> {
        let config: Self = serde_json::from_slice(bytes).context("reading DSpine config")?;
        ensure!(
            config
                .architectures
                .iter()
                .any(|name| { name == "DFlashDraftModel" || name == "DSpineDraftModel" }),
            "unsupported DSpine architecture {:?}",
            config.architectures
        );
        ensure!(
            config.model_type == "qwen3",
            "DSpine model_type must be qwen3"
        );
        ensure!(
            config.num_hidden_layers == 5,
            "DSpine requires its five-layer draft"
        );
        ensure!(
            config.hidden_size == target_hidden,
            "DSpine hidden size does not match the target"
        );
        ensure!(
            config.vocab_size == target_vocab,
            "DSpine vocabulary does not match the target"
        );
        ensure!(
            config.dflash.target_layer_ids == TARGET_LAYERS,
            "unsupported DSpine target taps {:?}",
            config.dflash.target_layer_ids
        );
        ensure!(
            config.dflash.selected_prefix_feature_layers == [1, 2, 3, 4, 5],
            "DSpine bridge writes must be present after all five layers"
        );
        ensure!(
            config.dflash.selected_prefix_feature_writes
                && config.dflash.selected_prefix_feature_write_mode == "independent"
                && !config.dflash.selected_prefix_terminal
                && config.dflash.selected_prefix_decode_mode == "serial"
                && config.dflash.draft_block_causal,
            "unsupported DSpine causal bridge configuration"
        );
        ensure!(
            config.dflash.selected_prefix_last_write_refine.kind == "exact_sender",
            "unsupported DSpine final-write refinement"
        );
        ensure!(
            config.dflash.selected_prefix_last_write_refine.head_topk == 16,
            "unsupported DSpine refinement width"
        );
        ensure!(
            config.hidden_size == 5120
                && config.dflash.selected_prefix_code_bridge.code_size == 1024
                && config.dflash.selected_prefix_code_bridge.rank == 512,
            "unsupported DSpine 27B bridge geometry"
        );
        ensure!(
            config.block_size == 16,
            "unexpected DSpine trained block size"
        );
        ensure!(
            config.head_dim > 0
                && config.num_attention_heads > 0
                && config.num_key_value_heads > 0
                && config.num_attention_heads % config.num_key_value_heads == 0,
            "invalid DSpine attention geometry"
        );
        if config.use_sliding_window
            || config
                .layer_types
                .iter()
                .any(|kind| kind == "sliding_attention")
        {
            ensure!(
                config.sliding_window == Some(8192)
                    && config.layer_types == vec!["sliding_attention"; config.num_hidden_layers]
                    && config.is_causal,
                "DSpine 27B requires five causal sliding-attention layers with window 8192"
            );
        }
        if let Some(q) = &config.quantization {
            ensure!(
                q.bits == 4 && q.group_size == 64 && q.mode == "affine",
                "DSpine supports only MLX affine 4-bit/group-64 quantization"
            );
        }
        Ok(config)
    }

    pub fn attention_window(&self) -> Option<usize> {
        self.sliding_window.filter(|_| {
            self.use_sliding_window
                || self
                    .layer_types
                    .iter()
                    .any(|kind| kind == "sliding_attention")
        })
    }

    pub fn target_layers(&self) -> &[usize] {
        &self.dflash.target_layer_ids
    }

    pub fn served_block_size(&self) -> usize {
        SERVED_BLOCK_SIZE.min(self.block_size)
    }
}

pub fn supports_target(config: &crate::qwen3_5::config::Qwen35Config) -> bool {
    config.hidden_size == 5120 && config.num_hidden_layers == 64 && config.vocab_size == 248320
}
