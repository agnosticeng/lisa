//! Model configuration (`glm5_next` / `glm5_next_text`).
//!
//! GLM-5.3 Flash: 45 layers (34 KDA `linear_attention` + 11 NoPE-MLA
//! `deepseek_sparse_attention`), 288-expert MoE from layer 3, mHC
//! hyper-connections, one MTP block. Field names follow the official HF
//! config; see `specs/models/glm53.md`.

use serde::Deserialize;

#[derive(Debug, Clone, Deserialize, serde::Serialize)]
struct RawText {
    #[serde(rename = "hidden_size", default)]
    hidden_size: Option<usize>,
    #[serde(rename = "num_hidden_layers", default)]
    num_hidden_layers: Option<usize>,
    #[serde(rename = "max_position_embeddings", default)]
    max_position_embeddings: Option<usize>,
    #[serde(rename = "vocab_size", default)]
    vocab_size: Option<usize>,
    #[serde(rename = "rms_norm_eps", default)]
    rms_norm_eps: Option<f32>,
    #[serde(rename = "layer_types", default)]
    layer_types: Option<Vec<String>>,
    #[serde(rename = "mlp_layer_types", default)]
    mlp_layer_types: Option<Vec<String>>,
    #[serde(rename = "first_k_dense_replace", default)]
    first_k_dense_replace: Option<usize>,
    #[serde(rename = "num_attention_heads", default)]
    num_attention_heads: Option<usize>,
    #[serde(rename = "q_lora_rank", default)]
    q_lora_rank: Option<usize>,
    #[serde(rename = "kv_lora_rank", default)]
    kv_lora_rank: Option<usize>,
    #[serde(rename = "qk_nope_head_dim", default)]
    qk_nope_head_dim: Option<usize>,
    #[serde(rename = "qk_rope_head_dim", default)]
    qk_rope_head_dim: Option<usize>,
    #[serde(rename = "v_head_dim", default)]
    v_head_dim: Option<usize>,
    #[serde(rename = "mla_use_nope", default)]
    mla_use_nope: Option<bool>,
    #[serde(rename = "index_n_heads", default)]
    index_n_heads: Option<usize>,
    #[serde(rename = "index_head_dim", default)]
    index_head_dim: Option<usize>,
    #[serde(rename = "index_topk", default)]
    index_topk: Option<usize>,
    #[serde(rename = "index_kpool", default)]
    index_kpool: Option<usize>,
    #[serde(rename = "index_kpool_always_select_tail", default)]
    index_kpool_always_select_tail: Option<bool>,
    #[serde(rename = "linear_num_heads", default)]
    linear_num_heads: Option<usize>,
    #[serde(rename = "linear_head_dim", default)]
    linear_head_dim: Option<usize>,
    #[serde(rename = "linear_conv_kernel_dim", default)]
    linear_conv_kernel_dim: Option<usize>,
    #[serde(rename = "linear_lower_bound", default)]
    linear_lower_bound: Option<f32>,
    #[serde(rename = "n_routed_experts", default)]
    n_routed_experts: Option<usize>,
    #[serde(rename = "num_experts_per_tok", default)]
    num_experts_per_tok: Option<usize>,
    #[serde(rename = "n_shared_experts", default)]
    n_shared_experts: Option<usize>,
    #[serde(rename = "n_group", default)]
    n_group: Option<usize>,
    #[serde(rename = "topk_group", default)]
    topk_group: Option<usize>,
    #[serde(rename = "linear_attn_config", default)]
    linear_attn_config: Option<RawLinearAttn>,
    #[serde(rename = "routed_scaling_factor", default)]
    routed_scaling_factor: Option<f32>,
    #[serde(rename = "norm_topk_prob", default)]
    norm_topk_prob: Option<bool>,
    #[serde(rename = "moe_intermediate_size", default)]
    moe_intermediate_size: Option<usize>,
    #[serde(rename = "intermediate_size", default)]
    intermediate_size: Option<usize>,
    #[serde(rename = "swiglu_limit", default)]
    swiglu_limit: Option<f32>,
    #[serde(rename = "moe_router_dtype", default)]
    moe_router_dtype: Option<String>,
    #[serde(rename = "hc_mult", default)]
    hc_mult: Option<usize>,
    #[serde(rename = "hc_eps", default)]
    hc_eps: Option<f32>,
    #[serde(rename = "hc_sinkhorn_iters", default)]
    hc_sinkhorn_iters: Option<usize>,
    #[serde(rename = "num_nextn_predict_layers", default)]
    num_nextn_predict_layers: Option<usize>,
    #[serde(rename = "index_share_for_mtp_iteration", default)]
    index_share_for_mtp_iteration: Option<bool>,
    #[serde(rename = "eos_token_id", default)]
    eos_token_id: Option<serde_json::Value>,
    #[serde(rename = "tie_word_embeddings", default)]
    tie_word_embeddings: Option<bool>,
}

#[derive(Debug, Clone, Deserialize, serde::Serialize)]
struct RawLinearAttn {
    #[serde(rename = "num_heads", default)]
    num_heads: Option<usize>,
    #[serde(rename = "head_dim", default)]
    head_dim: Option<usize>,
    #[serde(rename = "short_conv_kernel_size", default)]
    short_conv_kernel_size: Option<usize>,
    #[serde(rename = "gate_lower_bound", default)]
    gate_lower_bound: Option<f32>,
}

#[derive(Debug, Clone, Deserialize)]
struct RawConfig {
    #[serde(rename = "text_config", default)]
    text_config: Option<RawText>,
    #[serde(flatten)]
    root: RawText,
}

/// Flattened text-tower configuration for `glm5_next`.
#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub max_position_embeddings: usize,
    pub vocab_size: usize,
    pub rms_norm_eps: f32,
    pub layer_types: Vec<String>,
    pub mlp_layer_types: Vec<String>,
    pub first_k_dense_replace: usize,
    // NoPE MLA
    pub num_attention_heads: usize,
    pub q_lora_rank: usize,
    pub kv_lora_rank: usize,
    pub qk_nope_head_dim: usize,
    pub qk_rope_head_dim: usize,
    pub v_head_dim: usize,
    pub mla_use_nope: bool,
    // Indexer
    pub index_n_heads: usize,
    pub index_head_dim: usize,
    pub index_topk: usize,
    pub index_kpool: usize,
    pub index_kpool_always_select_tail: bool,
    // KDA
    pub linear_num_heads: usize,
    pub linear_head_dim: usize,
    pub linear_conv_kernel_dim: usize,
    pub linear_lower_bound: f32,
    // MoE
    pub n_routed_experts: usize,
    pub num_experts_per_tok: usize,
    pub n_shared_experts: usize,
    pub n_group: usize,
    pub topk_group: usize,
    pub routed_scaling_factor: f32,
    pub norm_topk_prob: bool,
    pub moe_intermediate_size: usize,
    pub intermediate_size: usize,
    pub swiglu_limit: f32,
    pub moe_router_dtype: String,
    // mHC
    pub hc_mult: usize,
    pub hc_eps: f32,
    pub hc_sinkhorn_iters: usize,
    // MTP
    pub num_nextn_predict_layers: usize,
    pub index_share_for_mtp_iteration: bool,
    pub eos_token_ids: Vec<i64>,
    pub eos_token_id: i64,
    pub tie_word_embeddings: bool,
}

fn merge(nested: &Option<RawText>, root: &RawText) -> RawText {
    // Field-wise merge; the nested text_config wins over the root.
    let mut v = serde_json::to_value(root).expect("root to value");
    if let Some(t) = nested {
        if let (Some(a), Ok(b)) = (v.as_object_mut(), serde_json::to_value(t).map_err(|_| ())) {
            if let Some(b) = b.as_object() {
                for (k, val) in b {
                    a.insert(k.clone(), val.clone());
                }
            }
        }
    }
    serde_json::from_value(v).expect("merged config")
}

impl ModelConfig {
    pub fn from_json(path: &std::path::Path) -> anyhow::Result<Self> {
        let data = std::fs::read_to_string(path)?;
        Self::from_str(&data)
    }

    pub fn from_str(data: &str) -> anyhow::Result<Self> {
        let raw: RawConfig = serde_json::from_str(data)?;
        let m = merge(&raw.text_config, &raw.root);
        let eos_token_ids: Vec<i64> = match &m.eos_token_id {
            Some(serde_json::Value::Number(n)) => n.as_i64().into_iter().collect(),
            Some(serde_json::Value::Array(a)) => {
                a.iter().filter_map(serde_json::Value::as_i64).collect()
            }
            _ => Vec::new(),
        };
        let mut unique_eos = Vec::with_capacity(eos_token_ids.len());
        for id in eos_token_ids {
            if !unique_eos.contains(&id) {
                unique_eos.push(id);
            }
        }
        let mut eos_token_ids = unique_eos;
        if eos_token_ids.is_empty() {
            eos_token_ids.push(154_820);
        }
        let eos_token_id = eos_token_ids[0];

        Ok(Self {
            hidden_size: m.hidden_size.unwrap_or(4096),
            num_hidden_layers: m.num_hidden_layers.unwrap_or(45),
            max_position_embeddings: m.max_position_embeddings.unwrap_or(1_048_576),
            vocab_size: m.vocab_size.unwrap_or(154_880),
            rms_norm_eps: m.rms_norm_eps.unwrap_or(1e-5),
            layer_types: m.layer_types.clone().unwrap_or_default(),
            mlp_layer_types: m.mlp_layer_types.clone().unwrap_or_default(),
            first_k_dense_replace: m.first_k_dense_replace.unwrap_or(3),
            num_attention_heads: m.num_attention_heads.unwrap_or(64),
            q_lora_rank: m.q_lora_rank.unwrap_or(1536),
            kv_lora_rank: m.kv_lora_rank.unwrap_or(512),
            qk_nope_head_dim: m.qk_nope_head_dim.unwrap_or(256),
            qk_rope_head_dim: m.qk_rope_head_dim.unwrap_or(0),
            v_head_dim: m.v_head_dim.unwrap_or(256),
            mla_use_nope: m.mla_use_nope.unwrap_or(true),
            index_n_heads: m.index_n_heads.unwrap_or(32),
            index_head_dim: m.index_head_dim.unwrap_or(128),
            index_topk: m.index_topk.unwrap_or(2048),
            index_kpool: m.index_kpool.unwrap_or(4),
            index_kpool_always_select_tail: m.index_kpool_always_select_tail.unwrap_or(true),
            linear_num_heads: m
                .linear_num_heads
                .or_else(|| m.linear_attn_config.as_ref().and_then(|l| l.num_heads))
                .unwrap_or(64),
            linear_head_dim: m
                .linear_head_dim
                .or_else(|| m.linear_attn_config.as_ref().and_then(|l| l.head_dim))
                .unwrap_or(128),
            linear_conv_kernel_dim: m
                .linear_conv_kernel_dim
                .or_else(|| {
                    m.linear_attn_config
                        .as_ref()
                        .and_then(|l| l.short_conv_kernel_size)
                })
                .unwrap_or(4),
            linear_lower_bound: m
                .linear_lower_bound
                .or_else(|| {
                    m.linear_attn_config
                        .as_ref()
                        .and_then(|l| l.gate_lower_bound)
                })
                .unwrap_or(-5.0),
            n_routed_experts: m.n_routed_experts.unwrap_or(288),
            num_experts_per_tok: m.num_experts_per_tok.unwrap_or(8),
            n_shared_experts: m.n_shared_experts.unwrap_or(1),
            n_group: m.n_group.unwrap_or(1),
            topk_group: m.topk_group.unwrap_or(1),
            routed_scaling_factor: m.routed_scaling_factor.unwrap_or(2.5),
            norm_topk_prob: m.norm_topk_prob.unwrap_or(true),
            moe_intermediate_size: m.moe_intermediate_size.unwrap_or(2048),
            intermediate_size: m.intermediate_size.unwrap_or(12_288),
            swiglu_limit: m.swiglu_limit.unwrap_or(10.0),
            moe_router_dtype: m
                .moe_router_dtype
                .clone()
                .unwrap_or_else(|| "float32".to_string()),
            hc_mult: m.hc_mult.unwrap_or(4),
            hc_eps: m.hc_eps.unwrap_or(1e-6),
            hc_sinkhorn_iters: m.hc_sinkhorn_iters.unwrap_or(20),
            num_nextn_predict_layers: m.num_nextn_predict_layers.unwrap_or(1),
            index_share_for_mtp_iteration: m.index_share_for_mtp_iteration.unwrap_or(true),
            eos_token_ids,
            eos_token_id,
            tie_word_embeddings: m.tie_word_embeddings.unwrap_or(false),
        })
    }

    pub fn is_sparse_mlp(&self, layer: usize) -> bool {
        match self.mlp_layer_types.get(layer) {
            Some(t) => t == "sparse",
            None => layer >= self.first_k_dense_replace,
        }
    }

    pub fn is_kda(&self, layer: usize) -> bool {
        self.layer_types.get(layer).map(String::as_str) == Some("linear_attention")
    }

    /// Number of blocks including the MTP block (config layer list carries it).
    pub fn total_blocks(&self) -> usize {
        self.num_hidden_layers + self.num_nextn_predict_layers
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ARTIFACT: &str = concat!(
        env!("HOME"),
        "/.cache/huggingface/hub/models--lisa--glm53-flash-2bit/snapshots/main"
    );

    #[test]
    fn parses_real_artifact_config() {
        let path = std::path::Path::new(ARTIFACT).join("config.json");
        if !path.exists() {
            eprintln!("artifact not present; skipping");
            return;
        }
        let c = ModelConfig::from_json(&path).expect("config parse");
        assert_eq!(c.hidden_size, 4096);
        assert_eq!(c.num_hidden_layers, 45);
        assert_eq!(c.vocab_size, 154_880);
        assert_eq!(c.total_blocks(), 46);
        assert_eq!(c.n_routed_experts, 288);
        assert_eq!(c.num_experts_per_tok, 8);
        assert_eq!(c.eos_token_ids, vec![154_820, 154_827, 154_829]);
        assert_eq!(c.kv_lora_rank, 512);
        assert_eq!(c.index_topk, 2048);
        assert_eq!(c.index_kpool, 4);
        assert_eq!(c.hc_mult, 4);
        assert_eq!(c.hc_sinkhorn_iters, 20);
        assert_eq!(c.swiglu_limit, 10.0);
        assert!((c.linear_lower_bound + 5.0).abs() < 1e-6);
        assert!(c.mla_use_nope);
        assert_eq!(c.qk_rope_head_dim, 0);
        // Layer types: 3 dense KDA, DSA at 3, 7, ..., 43, KDA at 44, MTP DSA.
        assert!(c.is_kda(0) && c.is_kda(2));
        assert!(!c.is_kda(3));
        assert!(!c.is_kda(43));
        assert!(c.is_kda(44));
        assert!(!c.is_kda(45));
        // MLP: dense first 3, sparse from 3 on (incl. MTP).
        assert!(!c.is_sparse_mlp(0));
        assert!(c.is_sparse_mlp(3));
        assert!(c.is_sparse_mlp(45));
        // Router dtype must be fp32 (our recipe requirement).
        assert_eq!(c.moe_router_dtype, "float32");
    }
}
