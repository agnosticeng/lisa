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
    let history_law = crate::gated_delta::complete_history_cache_law(arithmetic_geometry_identity(
        config,
        device_arithmetic,
    ));
    b"lisa.qwen4.serial-capture.v1"
        .iter()
        .fold(history_law, |hash, byte| {
            (hash ^ u64::from(*byte)).wrapping_mul(0x100000001b3)
        })
}

fn arithmetic_geometry_identity(config: &config::ModelConfig, device_arithmetic: &[u8]) -> u64 {
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

/// Bind the loaded normalization mode only where it selects a different graph.
/// Hyper normalization uses BF16 and `hidden_size` rows. The forced-operation
/// mode changes execution only at width 128; wider rows, including the curated
/// 2560-wide model, always use the same explicit reduction chain.
pub fn cache_law_with_norm_mode(
    config: &config::ModelConfig,
    device_arithmetic: &[u8],
    force_norm_operations: bool,
) -> u64 {
    let law = cache_law(config, device_arithmetic);
    if config.hidden_size == 128 && force_norm_operations {
        b"lisa.qwen4.hyper.norm-operations.v1"
            .iter()
            .fold(law, |hash, byte| {
                (hash ^ u64::from(*byte)).wrapping_mul(0x100000001b3)
            })
    } else {
        law
    }
}

#[cfg(test)]
mod law_tests {
    use super::*;
    #[test]
    fn complete_convolution_history_excludes_legacy_prefixes() {
        let config = config::ModelConfig::parse("{}").unwrap();
        let legacy = arithmetic_geometry_identity(&config, b"metal:test");
        assert_ne!(cache_law(&config, b"metal:test"), legacy);
        assert_ne!(
            cache_law(&config, b"metal:test"),
            crate::gated_delta::complete_history_cache_law(legacy)
        );
        assert_ne!(
            cache_law_with_norm_mode(&config, b"metal:test", true),
            legacy
        );
    }
    #[test]
    fn loaded_arithmetic_and_topology_partition_prefix_state() {
        let base = config::ModelConfig::parse("{}").unwrap();
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
    #[test]
    fn loaded_norm_mode_partitions_only_distinct_graphs() -> anyhow::Result<()> {
        use lisa_backend::{Kernel, ScalarType, TensorOperation};
        let base = config::ModelConfig::parse("{}")?;
        for width in [128, 2560] {
            let mut config = base.clone();
            config.hidden_size = width;
            let graph = |forced| {
                crate::normalization::grouped_rms(
                    &[1, 1, (4 * width) as i32],
                    ScalarType::BF16,
                    config.rms_norm_eps,
                    width,
                    true,
                    forced,
                )
            };
            let normal = graph(false)?;
            let forced = graph(true)?;
            let has_fused = |graph: &crate::graph::Graph| {
                graph.calls.iter().any(|call| {
                    matches!(
                        call.kernel,
                        Kernel::Tensor(TensorOperation::GroupedRms128 { .. })
                    )
                })
            };
            assert!(!has_fused(&forced));
            assert_eq!(has_fused(&normal), width == 128);
            let original = cache_law(&config, b"strict");
            let normal_law = cache_law_with_norm_mode(&config, b"strict", false);
            let forced_law = cache_law_with_norm_mode(&config, b"strict", true);
            assert_eq!(normal_law, original);
            if width == 128 {
                assert_ne!(normal_law, forced_law);
            } else {
                assert_eq!(normal_law, forced_law);
                assert_eq!(normal.outputs, forced.outputs);
                assert_eq!(
                    normal
                        .calls
                        .iter()
                        .map(|call| (&call.kernel, &call.inputs))
                        .collect::<Vec<_>>(),
                    forced
                        .calls
                        .iter()
                        .map(|call| (&call.kernel, &call.inputs))
                        .collect::<Vec<_>>()
                );
            }
        }
        Ok(())
    }
}
