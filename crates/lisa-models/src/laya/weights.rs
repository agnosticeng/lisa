//! Required checkpoint names; the engine validates their presence during binding.
use super::config::LayaConfig;
pub fn required_tensors(cfg: &LayaConfig) -> Vec<String> {
    let mut required = vec![
        "encoder.embeddings.norm.weight".to_string(),
        "encoder.final_norm.weight".to_string(),
        "type_emb.weight".to_string(),
        "scorer.0.weight".to_string(),
        "act_head.0.weight".to_string(),
    ];
    for i in 0..cfg.num_layers {
        let l = format!("encoder.layers.{i}");
        required.push(format!("{l}.attn.Wqkv.weight"));
        required.push(format!("{l}.attn.Wo.weight"));
        required.push(format!("{l}.mlp.Wi.weight"));
        required.push(format!("{l}.mlp.Wo.weight"));
        required.push(format!("{l}.mlp_norm.weight"));
        if i > 0 {
            required.push(format!("{l}.attn_norm.weight"));
        }
    }
    for i in 0..cfg.head_layers {
        for t in [
            "norm1.weight",
            "norm2.weight",
            "self_attn.in_proj_weight",
            "self_attn.out_proj.weight",
            "linear1.weight",
            "linear2.weight",
        ] {
            required.push(format!("head.layers.{i}.{t}"));
        }
    }
    required
}
