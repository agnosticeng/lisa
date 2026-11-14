//! Portable Laya checkpoint schema and calibration rules.
use anyhow::Context;
use std::collections::BTreeMap;
/// Parsed `encoder/config.json` + `rl_agent_config.json`.
#[derive(Debug, Clone)]
pub struct LayaConfig {
    // --- encoder (ModernBERT) ---
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub norm_eps: f32,
    pub local_attention: usize,
    /// Per layer: true = full attention, false = sliding window.
    pub layer_global: Vec<bool>,
    pub rope_theta_global: f32,
    pub rope_theta_local: f32,
    pub max_position_embeddings: usize,
    pub tie_word_embeddings: bool,

    // --- special tokens ---
    pub cls_id: u32,
    pub sep_id: u32,
    pub pad_id: u32,
    pub mask_id: u32,
    pub mask_token: String,

    // --- decision head / agent ---
    pub head_layers: usize,
    pub max_len: usize,
    pub head_max_len: usize,
    /// Output columns of the action head (`len(act_costs) + 1`).
    pub n_actions: usize,
    /// Per-question-type temperatures `[choice, score, noul]`.
    pub temperature: [f32; 3],
    /// Bucketed temperatures keyed like `"choice:3-5"`.
    pub temperature_by_options: BTreeMap<String, f32>,
}

impl LayaConfig {
    /// Parse checkpoint metadata supplied by the engine.
    pub fn from_values(
        enc: &serde_json::Value,
        agent: &serde_json::Value,
        tcfg: Option<&serde_json::Value>,
    ) -> anyhow::Result<Self> {
        let model_type = str_of(&enc, "model_type").unwrap_or_default();
        anyhow::ensure!(
            model_type == "modernbert",
            "unsupported Laya encoder model_type {model_type:?} (want modernbert)"
        );
        if let Some(act) = str_of(&enc, "hidden_activation") {
            anyhow::ensure!(act == "gelu", "unsupported hidden_activation {act:?}");
        }

        let hidden_size = int_of(&enc, "hidden_size").context("hidden_size")?;
        let num_heads = int_of(&enc, "num_attention_heads").context("num_attention_heads")?;
        let num_layers = int_of(&enc, "num_hidden_layers").context("num_hidden_layers")?;
        anyhow::ensure!(
            num_heads > 0 && hidden_size % num_heads == 0,
            "hidden_size % heads"
        );
        let head_dim = hidden_size / num_heads;
        anyhow::ensure!(head_dim % 2 == 0, "head_dim must be even for RoPE");

        // Layer attention pattern: `layer_types` if present, else every Nth full.
        let every_n = int_of(&enc, "global_attn_every_n_layers").unwrap_or(3);
        anyhow::ensure!(every_n > 0, "global_attn_every_n_layers is 0");
        let mut layer_global = vec![false; num_layers];
        for (i, g) in layer_global.iter_mut().enumerate() {
            *g = i % every_n == 0;
        }
        if let Some(types) = enc.get("layer_types").and_then(|v| v.as_array()) {
            anyhow::ensure!(types.len() == num_layers, "layer_types length");
            for (i, v) in types.iter().enumerate() {
                match v.as_str().unwrap_or_default() {
                    "full_attention" => layer_global[i] = true,
                    "sliding_attention" => layer_global[i] = false,
                    other => anyhow::bail!("unknown layer type {other:?}"),
                }
            }
        }

        let (mut theta_global, mut theta_local) = (
            f32_of(&enc, "global_rope_theta").unwrap_or(160_000.0),
            f32_of(&enc, "local_rope_theta").unwrap_or(10_000.0),
        );
        if let Some(rp) = enc.get("rope_parameters") {
            if let Some(v) = rp
                .pointer("/full_attention/rope_theta")
                .and_then(|v| v.as_f64())
            {
                theta_global = v as f32;
            }
            if let Some(v) = rp
                .pointer("/sliding_attention/rope_theta")
                .and_then(|v| v.as_f64())
            {
                theta_local = v as f32;
            }
        }

        // Special tokens: ids from the encoder config, `mask_token` text from the
        // tokenizer config (the id is looked up from the tokenizer at run time).
        let cls_id = int_of(&enc, "cls_token_id").unwrap_or(0) as u32;
        let sep_id = int_of(&enc, "sep_token_id").unwrap_or(0) as u32;
        let pad_id = int_of(&enc, "pad_token_id").unwrap_or(0) as u32;
        let mask_token = tcfg
            .as_ref()
            .and_then(|t| t.get("mask_token"))
            .and_then(mask_text)
            .unwrap_or_else(|| "[MASK]".to_string());

        // --- agent ---
        let head_layers = int_of(&agent, "head_layers").unwrap_or(2);
        let max_len = int_of(&agent, "max_len").unwrap_or(512);
        let head_max_len = int_of(&agent, "head_max_len").unwrap_or(192);
        let max_pos = int_of(&enc, "max_position_embeddings").unwrap_or(8192);
        anyhow::ensure!(
            4 < head_max_len && head_max_len < max_len && max_len <= max_pos,
            "want 4 < head_max_len ({head_max_len}) < max_len ({max_len}) <= max_position_embeddings ({max_pos})"
        );
        let n_actions = agent
            .get("act_costs")
            .and_then(|v| v.as_object())
            .map(|o| o.len() + 1)
            .unwrap_or(1);

        let mut temperature = [1.0f32; 3];
        if let Some(t) = agent.get("temperature").and_then(|v| v.as_array()) {
            anyhow::ensure!(t.len() == 3, "temperature must have 3 entries");
            for (i, v) in t.iter().enumerate() {
                temperature[i] = v.as_f64().unwrap_or(1.0) as f32;
            }
        }
        for (i, t) in temperature.iter_mut().enumerate() {
            clamp_temperature(["choice", "score", "noul"][i], t);
        }
        let mut temperature_by_options = BTreeMap::new();
        if let Some(m) = agent
            .get("temperature_by_options")
            .and_then(|v| v.as_object())
        {
            for (k, v) in m {
                let mut t = v.as_f64().unwrap_or(1.0) as f32;
                clamp_temperature(k, &mut t);
                temperature_by_options.insert(k.clone(), t);
            }
        }

        Ok(Self {
            vocab_size: int_of(&enc, "vocab_size").unwrap_or(0),
            hidden_size,
            intermediate_size: int_of(&enc, "intermediate_size").unwrap_or(0),
            num_layers,
            num_heads,
            head_dim,
            norm_eps: f32_of(&enc, "norm_eps")
                .or_else(|| f32_of(&enc, "layer_norm_eps"))
                .unwrap_or(1e-5),
            local_attention: int_of(&enc, "local_attention").unwrap_or(128),
            layer_global,
            rope_theta_global: theta_global,
            rope_theta_local: theta_local,
            max_position_embeddings: max_pos,
            tie_word_embeddings: enc
                .get("tie_word_embeddings")
                .and_then(|v| v.as_bool())
                .unwrap_or(true),
            cls_id,
            sep_id,
            pad_id,
            // Resolved from the tokenizer at load time; 0 until then.
            mask_id: 0,
            mask_token,
            head_layers,
            max_len,
            head_max_len,
            n_actions,
            temperature,
            temperature_by_options,
        })
    }

    /// A one-line structural summary.
    pub fn summary(&self) -> String {
        let global = self.layer_global.iter().filter(|g| **g).count();
        format!(
            "laya: ModernBERT {}L h{} heads{} f{} vocab{} | {} global / {} sliding, \
             local {} | head {}L max_len {} head_max_len {} actions {} | eps {:.0e}",
            self.num_layers,
            self.hidden_size,
            self.num_heads,
            self.intermediate_size,
            self.vocab_size,
            global,
            self.num_layers - global,
            self.local_attention,
            self.head_layers,
            self.max_len,
            self.head_max_len,
            self.n_actions,
            self.norm_eps,
        )
    }
}

/// A fitted temperature below 0.5 sharpens
/// the logits enough to report a coin flip as a certainty.
pub fn clamp_temperature(name: &str, t: &mut f32) {
    const TEMP_MIN: f32 = 0.5;
    const TEMP_MAX: f32 = 5.0;
    if !(*t > 0.0) || !t.is_finite() {
        *t = 1.0;
        return;
    }
    if name.starts_with("noul") {
        return;
    }
    *t = t.clamp(TEMP_MIN, TEMP_MAX);
}

fn int_of(v: &serde_json::Value, key: &str) -> Option<usize> {
    v.get(key)?.as_u64().map(|x| x as usize)
}

fn f32_of(v: &serde_json::Value, key: &str) -> Option<f32> {
    v.get(key)?.as_f64().map(|x| x as f32)
}

fn str_of(v: &serde_json::Value, key: &str) -> Option<String> {
    v.get(key)?.as_str().map(|s| s.to_string())
}

/// `mask_token` is a plain string or `{ "content": "..." }`.
fn mask_text(v: &serde_json::Value) -> Option<String> {
    match v {
        serde_json::Value::String(s) => Some(s.clone()),
        serde_json::Value::Object(o) => o.get("content")?.as_str().map(|s| s.to_string()),
        _ => None,
    }
}
