//! Validate the DS4 checkpoint before mapping any tensor payload.

use super::super::config::ModelConfig;
use anyhow::{Context, Result, ensure};
use lisa_gguf::{Header, Package, TensorType, Value};

pub fn config(header: &Header) -> Result<ModelConfig> {
    ensure!(
        header.architecture()? == "glm5-next",
        "expected GLM-5.3 Flash GGUF"
    );
    let mut fields = serde_json::Map::new();
    for (key, field, expected) in [
        ("block_count", "total_blocks", 46),
        ("trunk_block_count", "num_hidden_layers", 45),
        ("nextn_predict_layers", "num_nextn_predict_layers", 1),
        ("context_length", "max_position_embeddings", 1048576),
        ("embedding_length", "hidden_size", 4096),
        ("vocab_size", "vocab_size", 154880),
        ("feed_forward_length", "intermediate_size", 12288),
        ("expert_feed_forward_length", "moe_intermediate_size", 2048),
        ("expert_count", "n_routed_experts", 288),
        ("expert_used_count", "num_experts_per_tok", 8),
        ("expert_shared_count", "n_shared_experts", 1),
        ("leading_dense_block_count", "first_k_dense_replace", 3),
        ("attention.head_count", "num_attention_heads", 64),
        ("attention.key_length", "qk_nope_head_dim", 256),
        ("attention.value_length", "v_head_dim", 256),
        ("attention.q_lora_rank", "q_lora_rank", 1536),
        ("attention.kv_lora_rank", "kv_lora_rank", 512),
        ("attention.rope_dimension_count", "qk_rope_head_dim", 0),
        ("attention.indexer.head_count", "index_n_heads", 32),
        ("attention.indexer.key_length", "index_head_dim", 128),
        ("attention.indexer.top_k", "index_topk", 2048),
        ("attention.indexer.pool_size", "index_kpool", 4),
        ("linear_attention.head_count", "linear_num_heads", 64),
        ("linear_attention.head_dimension", "linear_head_dim", 128),
        ("linear_attention.conv_kernel", "linear_conv_kernel_dim", 4),
        ("hyper_connection.count", "hc_mult", 4),
        (
            "hyper_connection.sinkhorn_iterations",
            "hc_sinkhorn_iters",
            20,
        ),
    ] {
        let value = header.integer(&format!("glm5-next.{key}"))?;
        ensure!(
            value == expected,
            "unsupported GLM GGUF {key}: {value}, expected {expected}"
        );
        fields.insert(field.into(), value.into());
    }
    for (key, field, expected) in [
        ("expert_weights_scale", "routed_scaling_factor", 2.5f32),
        ("swiglu_limit", "swiglu_limit", 10.0),
        ("attention.layer_norm_rms_epsilon", "rms_norm_eps", 1e-5),
        (
            "linear_attention.gate_lower_bound",
            "linear_lower_bound",
            -5.0,
        ),
        ("hyper_connection.epsilon", "hc_eps", 1e-6),
    ] {
        let value = header
            .metadata
            .get(&format!("glm5-next.{key}"))
            .and_then(Value::as_f64)
            .with_context(|| format!("missing GLM GGUF {key}"))? as f32;
        ensure!(
            value.to_bits() == expected.to_bits(),
            "unsupported GLM GGUF {key}: {value}"
        );
        fields.insert(field.into(), serde_json::json!(value));
    }
    ensure!(
        header.metadata.get("glm5-next.expert_weights_norm") == Some(&Value::Bool(true)),
        "GLM GGUF requires normalized route weights"
    );
    let types = header
        .metadata
        .get("glm5-next.layer_types")
        .and_then(Value::as_array)
        .context("missing GLM layer types")?;
    ensure!(types.len() == 46, "GLM GGUF layer type count mismatch");
    let mut layer_types = Vec::new();
    for (layer, value) in types.iter().enumerate() {
        let kda = layer < 45 && layer % 4 != 3;
        ensure!(
            value.as_u64() == Some(u64::from(!kda)),
            "GLM GGUF layer type mismatch at {layer}"
        );
        layer_types.push(if kda {
            "linear_attention"
        } else {
            "deepseek_sparse_attention"
        });
    }
    fields.insert("layer_types".into(), serde_json::json!(layer_types));
    let tokens = header
        .metadata
        .get("tokenizer.ggml.tokens")
        .and_then(Value::as_array)
        .context("missing GLM vocabulary")?;
    ensure!(
        tokens.len() == 154880,
        "GLM GGUF vocabulary length mismatch"
    );
    let eos: Vec<_> = ["<|endoftext|>", "<|user|>", "<|observation|>"]
        .into_iter()
        .map(|name| {
            tokens
                .iter()
                .position(|v| v.as_str() == Some(name))
                .with_context(|| format!("missing GLM stop marker {name}"))
        })
        .collect::<Result<_>>()?;
    fields.insert("eos_token_id".into(), serde_json::json!(eos));
    ModelConfig::from_str(&serde_json::Value::Object(fields).to_string())
}

/// Fixed package contract for DS4's imatrix Q2 file. All names, formats and
/// shapes are checked before residency, including the embedded MTP block.
pub fn validate(package: &Package) -> Result<ModelConfig> {
    let config = config(&package.header)?;
    let mut expected = std::collections::BTreeSet::new();
    let mut tensor = |name: String, dims: &[u64], kind: TensorType| -> Result<()> {
        let info = &package.tensor(&name)?.info;
        ensure!(
            info.dimensions == dims && info.kind == kind,
            "GLM GGUF tensor {name}: expected {dims:?} {kind:?}, got {:?} {:?}",
            info.dimensions,
            info.kind
        );
        expected.insert(name);
        Ok(())
    };
    tensor(
        "token_embd.weight".into(),
        &[4096, 154880],
        TensorType::Q8_0,
    )?;
    tensor("output.weight".into(), &[4096, 154880], TensorType::Q8_0)?;
    tensor("output_norm.weight".into(), &[4096], TensorType::F32)?;
    for layer in 0..46 {
        let mut add =
            |suffix: &str, dims: &[u64], kind| tensor(format!("blk.{layer}.{suffix}"), dims, kind);
        for name in ["attn_norm.weight", "ffn_norm.weight"] {
            add(name, &[4096], TensorType::F32)?;
        }
        if layer < 45 {
            for half in ["attn", "ffn"] {
                add(
                    &format!("hc_{half}_fn.weight"),
                    &[16384, 24],
                    TensorType::BF16,
                )?;
                add(&format!("hc_{half}_scale.weight"), &[3], TensorType::F32)?;
                add(&format!("hc_{half}_base.weight"), &[24], TensorType::F32)?;
            }
        }
        if config.is_kda(layer) {
            for name in ["q", "k"] {
                add(
                    &format!("kda_{name}.weight"),
                    &[4096, 8192],
                    TensorType::Q4_K,
                )?;
            }
            add("kda_v.weight", &[4096, 8192], TensorType::Q8_0)?;
            for name in ["q", "k", "v"] {
                add(
                    &format!("kda_{name}_conv.weight"),
                    &[4, 1, 8192],
                    TensorType::F32,
                )?;
            }
            for name in ["f", "g"] {
                add(
                    &format!("kda_{name}_a.weight"),
                    &[4096, 128],
                    TensorType::Q8_0,
                )?;
                add(
                    &format!("kda_{name}_b.weight"),
                    &[128, 8192],
                    TensorType::Q8_0,
                )?;
            }
            add("kda_beta.weight", &[4096, 64], TensorType::Q8_0)?;
            add("kda_output.weight", &[8192, 4096], TensorType::Q8_0)?;
            for (name, dims) in [
                ("kda_a_log.weight", &[64][..]),
                ("kda_dt_bias.weight", &[8192][..]),
                ("kda_o_norm.weight", &[128][..]),
            ] {
                add(name, dims, TensorType::F32)?;
            }
        } else {
            for (name, dims) in [
                ("attn_q_a.weight", &[4096, 1536][..]),
                ("attn_q_b.weight", &[1536, 16384][..]),
                ("attn_kv_a_mqa.weight", &[4096, 512][..]),
                ("attn_output.weight", &[16384, 4096][..]),
                ("attn_k_b.weight", &[256, 512, 64][..]),
                ("attn_v_b.weight", &[512, 256, 64][..]),
            ] {
                add(name, dims, TensorType::Q8_0)?;
            }
            for (name, dims) in [
                ("attn_q_a_norm.weight", &[1536][..]),
                ("attn_kv_a_norm.weight", &[512][..]),
                ("indexer.k_norm.weight", &[128][..]),
                ("indexer.k_norm.bias", &[128][..]),
            ] {
                add(name, dims, TensorType::F32)?;
            }
            for (name, dims) in [
                ("indexer.attn_q_b.weight", &[1536, 4096][..]),
                ("indexer.attn_k.weight", &[4096, 128][..]),
                ("indexer.proj.weight", &[4096, 32][..]),
                ("indexer.pool_ape.weight", &[128, 4][..]),
                ("indexer.pool_gate.weight", &[4096, 128][..]),
            ] {
                add(name, dims, TensorType::BF16)?;
            }
        }
        if layer < 3 {
            for name in ["gate", "up"] {
                add(
                    &format!("ffn_{name}.weight"),
                    &[4096, 12288],
                    TensorType::Q8_0,
                )?;
            }
            add("ffn_down.weight", &[12288, 4096], TensorType::Q8_0)?;
        } else {
            add("ffn_gate_inp.weight", &[4096, 288], TensorType::F32)?;
            add("exp_probs_b.bias", &[288], TensorType::F32)?;
            for name in ["gate", "up"] {
                add(
                    &format!("ffn_{name}_exps.weight"),
                    &[4096, 2048, 288],
                    TensorType::IQ2_XXS,
                )?;
                add(
                    &format!("ffn_{name}_shexp.weight"),
                    &[4096, 2048],
                    TensorType::Q8_0,
                )?;
            }
            add("ffn_down_exps.weight", &[2048, 4096, 288], TensorType::Q2_K)?;
            add("ffn_down_shexp.weight", &[2048, 4096], TensorType::Q8_0)?;
        }
        if layer == 45 {
            add("nextn.eh_proj.weight", &[8192, 4096], TensorType::BF16)?;
            for name in ["enorm", "hnorm", "shared_head_norm"] {
                add(&format!("nextn.{name}.weight"), &[4096], TensorType::F32)?;
            }
        }
    }
    ensure!(
        expected.len() == package.tensors.len(),
        "unexpected tensors in GLM GGUF package"
    );
    Ok(config)
}
