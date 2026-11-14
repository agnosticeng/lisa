use crate::metadata::Metadata;
use anyhow::{Result, ensure};

/// Geometry read from the file, including the pruned Coder's 192 experts.
/// DS4's unpruned model has 384; this is never inferred from a repository name.
#[derive(Clone, Debug)]
pub struct Config {
    pub layers: usize,
    pub width: usize,
    pub vocab: usize,
    pub experts: usize,
    pub selected_experts: usize,
    pub expert_width: usize,
    pub context: usize,
    pub rms_epsilon: f32,
    pub hc_epsilon: f32,
    pub expert_scale: f32,
    pub compress_ratios: Vec<u32>,
    pub kv_sources: Vec<u32>,
    pub index_sources: Vec<u32>,
}

impl Config {
    pub fn from_metadata(m: &impl Metadata) -> Result<Self> {
        ensure!(
            m.architecture()? == "deepseek41",
            "expected DeepSeek V4.1 GGUF"
        );
        for (keys, expected) in [
            (&["block_count", "num_hidden_layers"][..], 40),
            (&["embedding_length", "hidden_size"][..], 5120),
            (
                &["expert_feed_forward_length", "moe_intermediate_size"][..],
                2304,
            ),
            (&["attention.head_count", "num_attention_heads"][..], 64),
            (&["attention.head_count_kv", "num_key_value_heads"][..], 1),
            (&["attention.key_length", "head_dim"][..], 512),
            (&["rope.dimension_count", "qk_rope_head_dim"][..], 64),
            (&["attention.q_lora_rank", "q_lora_rank"][..], 1280),
            (&["attention.output_lora_rank", "o_lora_rank"][..], 1024),
            (&["attention.output_group_count", "o_groups"][..], 8),
            (&["attention.sliding_window", "sliding_window"][..], 128),
            (&["attention.indexer.head_count", "index_n_heads"][..], 32),
            (&["attention.indexer.key_length", "index_head_dim"][..], 128),
            (&["attention.indexer.top_k", "index_topk"][..], 512),
            (&["expert_used_count", "num_experts_per_tok"][..], 6),
            (&["expert_shared_count", "n_shared_experts"][..], 1),
            (&["hyper_connection.count", "hc_mult"][..], 4),
            (
                &["hyper_connection.sinkhorn_iterations", "hc_sinkhorn_iters"][..],
                20,
            ),
            (
                &[
                    "attention.indexer.candidate_source_layer",
                    "candidate_source_layer_id",
                ][..],
                20,
            ),
            (
                &[
                    "attention.indexer.candidate_top_k_blocks",
                    "candidate_topk_blocks",
                ][..],
                2048,
            ),
            (
                &[
                    "attention.indexer.candidate_block_size",
                    "candidate_block_size",
                ][..],
                8,
            ),
        ] {
            m.expect(keys, expected)?;
        }
        let experts = m.integer(&["expert_count", "n_routed_experts"])? as usize;
        ensure!(
            matches!(experts, 192 | 384),
            "unsupported DeepSeek expert count {experts}"
        );
        ensure!(
            m.boolean(&["expert_weights_norm", "norm_topk_prob"])?,
            "DeepSeek requires normalized routed weights"
        );
        if m.contains("expert_gating_func") {
            m.expect(&["expert_gating_func"], 4)?;
        } else {
            ensure!(
                m.string(&["scoring_func"])? == "sqrtsoftplus",
                "unsupported DeepSeek router activation"
            );
        }
        for (keys, expected) in [
            (&["rope.freq_base", "rope_theta"][..], 10000.0),
            (
                &["attention.compress_rope_freq_base", "compress_rope_theta"][..],
                160000.0,
            ),
            (&["rope.scaling.factor", "rope_scaling.factor"][..], 16.0),
            (
                &["rope.scaling.yarn_beta_fast", "rope_scaling.beta_fast"][..],
                32.0,
            ),
            (
                &["rope.scaling.yarn_beta_slow", "rope_scaling.beta_slow"][..],
                1.0,
            ),
            (
                &[
                    "rope.scaling.original_context_length",
                    "rope_scaling.original_max_position_embeddings",
                ][..],
                65536.0,
            ),
        ] {
            ensure!(
                m.float(keys)? == expected,
                "unsupported DeepSeek RoPE metadata {keys:?}"
            );
        }
        if m.contains("swiglu_limit") {
            ensure!(
                m.float(&["swiglu_limit"])? == 10.0,
                "unsupported DeepSeek SwiGLU clamp"
            );
        } else {
            for key in ["swiglu_clamp_exp", "swiglu_clamp_shexp"] {
                let values = m.floats(&[key])?;
                ensure!(
                    values.len() == 40 && values.iter().all(|&v| v == 10.0),
                    "unsupported DeepSeek SwiGLU clamp schedule"
                );
            }
        }
        let vocab = m.vocabulary_len()?;
        ensure!(
            vocab == 129280,
            "unsupported DeepSeek vocabulary size {vocab}"
        );
        if m.contains("vocab_size") {
            ensure!(
                m.integer(&["vocab_size"])? == vocab as u64,
                "vocabulary metadata mismatch"
            );
        }
        let ratios = m.integers(&["attention.compress_ratios", "compress_ratios"])?;
        // The standard converter retains three draft-layer ratios in metadata
        // even when the GGUF contains only the 40-layer trunk. Metadata alone
        // does not establish that the checkpoint has an MTP/DSpark head.
        ensure!(
            ratios.len() == 40 || ratios.len() == 43 && ratios[40..] == [0, 0, 0],
            "unsupported compressor-ratio metadata"
        );
        let expected: Vec<u64> = (0..40)
            .map(|i| {
                if i < 2 {
                    0
                } else if i < 20 {
                    2
                } else {
                    1
                }
            })
            .collect();
        ensure!(
            ratios[..40] == expected,
            "unsupported DeepSeek compression schedule"
        );
        let kv_sources = m.integers(&["attention.kv_source_layers", "kv_source_layer_ids"])?;
        let index_sources =
            m.integers(&["attention.indexer.source_layers", "index_source_layer_ids"])?;
        ensure!(
            kv_sources == [2, 8, 14, 20],
            "unsupported DeepSeek shared KV sources"
        );
        ensure!(
            index_sources == [2, 8, 14, 20, 24, 28, 32, 36],
            "unsupported DeepSeek index sources"
        );
        let rms_epsilon = m.float(&["attention.layer_norm_rms_epsilon", "rms_norm_eps"])?;
        let hc_epsilon = m.float(&["hyper_connection.epsilon", "hc_eps"])?;
        let expert_scale = m.float(&["expert_weights_scale", "routed_scaling_factor"])?;
        ensure!(
            rms_epsilon == 1e-20 && hc_epsilon == 1e-6 && expert_scale == 1.5,
            "unsupported DeepSeek normalization or routing constants"
        );
        let context = usize::try_from(m.integer(&["context_length", "max_position_embeddings"])?)?;
        ensure!(context == 1048576, "unsupported DeepSeek context length");
        Ok(Self {
            layers: 40,
            width: 5120,
            vocab,
            experts,
            selected_experts: 6,
            expert_width: 2304,
            context,
            rms_epsilon,
            hc_epsilon,
            expert_scale,
            compress_ratios: ratios[..40].iter().map(|&v| v as u32).collect(),
            kv_sources: kv_sources.into_iter().map(|v| v as u32).collect(),
            index_sources: index_sources.into_iter().map(|v| v as u32).collect(),
        })
    }

    /// DS4's bounded prefill chunk. Carry storage is accounted separately.
    pub fn prefill_capacity(context: usize) -> usize {
        if context < 8192 {
            context.min(2048)
        } else if context < 16384 {
            4096
        } else {
            8192
        }
    }
}
