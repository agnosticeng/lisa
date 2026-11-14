//! Portable model definition and computation plans.
pub mod attention;
pub mod config;
pub mod decoder;
pub mod hyper;
pub mod indexer;
pub mod moe;
pub mod mtp;
pub mod ngram;
pub mod ple;
pub mod tower;

/// Stable arithmetic identity, available before any lazy kernel is prepared.
pub fn cache_law(config: &config::ModelConfig, device_arithmetic: &[u8]) -> u64 {
    let mut bytes = b"lisa.qwen4.ordered-hybrid.v1".to_vec();
    bytes.extend_from_slice(device_arithmetic);
    for value in [
        config.hidden_size,
        config.num_hidden_layers,
        config.num_attention_heads,
        config.num_key_value_heads,
        config.head_dim,
        config.num_experts,
        config.num_experts_per_tok,
        config.hc_count,
        config.linear_conv_kernel_dim,
        config.linear_num_key_heads,
        config.linear_num_value_heads,
        config.linear_key_head_dim,
        config.linear_value_head_dim,
        config.hc_lowrank,
        config.indexer_heads,
        config.indexer_kv_heads,
        config.indexer_head_dim,
        config.indexer_budget,
        config.indexer_compress_ratio,
        config.ple_embed_dim,
        config.ple_conv_kernel_size,
        config.moe_intermediate_size,
        config.shared_expert_intermediate_size,
    ] {
        bytes.extend_from_slice(&(value as u64).to_le_bytes());
    }
    for value in [
        config.rms_norm_eps,
        config.rms_norm_weight_offset,
        config.partial_rotary_factor,
        config.rope_theta,
    ] {
        bytes.extend_from_slice(&value.to_bits().to_le_bytes());
    }
    for kind in &config.layer_types {
        bytes.extend_from_slice(kind.as_bytes());
        bytes.push(0);
    }
    bytes.extend_from_slice(config.output_gate_type.as_bytes());
    bytes.iter().fold(0xcbf29ce484222325, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(0x100000001b3)
    })
}

#[cfg(test)]
mod law_tests {
    use super::*;
    #[test]
    fn loaded_arithmetic_and_topology_partition_prefix_state() {
        let base = config::ModelConfig::from_str("{}").unwrap();
        let law = cache_law(&base, b"metal:strict:1");
        assert_eq!(law, cache_law(&base.clone(), b"metal:strict:1"));
        assert_ne!(law, cache_law(&base, b"rocm:strict:1"));
        let mut changed = base.clone();
        changed.rms_norm_eps *= 2.;
        assert_ne!(law, cache_law(&changed, b"metal:strict:1"));
        let mut changed = base.clone();
        changed.layer_types[0] = "full_attention".into();
        assert_ne!(law, cache_law(&changed, b"metal:strict:1"));
        let mut changed = base.clone();
        changed.indexer_budget += 1;
        assert_ne!(law, cache_law(&changed, b"metal:strict:1"));
    }
}
