//! Validate the complete checkpoint before mapping resident tensor payloads.

use super::{config::Config, engram::Layout};
use anyhow::{Context, Result, ensure};
use lisa_gguf::{Package, TensorType};
use std::collections::BTreeSet;

#[derive(Debug)]
pub struct WeightPlan {
    pub resident: Vec<String>,
    pub disk: Vec<String>,
    pub resident_payload_bytes: u64,
    pub disk_payload_bytes: u64,
}

impl WeightPlan {
    pub fn inspect(package: &Package, config: &Config, engram: &Layout) -> Result<Self> {
        let mut seen = BTreeSet::new();
        let mut require = |name: String, shape: &[u64], plain: bool| -> Result<()> {
            let tensor = package.tensor(&name)?;
            ensure!(
                tensor.info.dimensions == shape,
                "DeepSeek tensor {name} has shape {:?}, expected {shape:?}",
                tensor.info.dimensions
            );
            let kind = tensor.info.kind;
            if plain {
                ensure!(kind == TensorType::F32, "{name} must retain F32 parameters");
            } else if shape.len() == 3 {
                super::projection::block_format(kind)
                    .with_context(|| format!("unsupported expert tensor {name}"))?;
            } else {
                ensure!(
                    matches!(
                        kind,
                        TensorType::F32
                            | TensorType::F16
                            | TensorType::BF16
                            | TensorType::Q8_0
                            | TensorType::Q5_K
                            | TensorType::IQ2_XXS
                            | TensorType::IQ2_XS
                            | TensorType::IQ1_M
                    ),
                    "unsupported projection format {kind:?} for {name}"
                );
                if !matches!(
                    kind,
                    TensorType::F32 | TensorType::F16 | TensorType::BF16 | TensorType::Q8_0
                ) {
                    ensure!(
                        shape[1] % 8 == 0,
                        "packed projection {name} requires complete eight-row tiles"
                    );
                }
            }
            seen.insert(name);
            Ok(())
        };
        require(
            "token_embd.weight".into(),
            &[5120, config.vocab as u64],
            false,
        )?;
        require("output.weight".into(), &[5120, config.vocab as u64], false)?;
        require("output_norm.weight".into(), &[5120], true)?;
        for layer in 0..config.layers {
            for (suffix, shape, plain) in [
                ("attn_kv.weight", &[5120, 512][..], false),
                ("attn_kv_a_norm.weight", &[512][..], true),
                ("attn_norm.weight", &[5120][..], true),
                ("attn_output_a.weight", &[4096, 8192][..], false),
                ("attn_output_b.weight", &[8192, 5120][..], false),
                ("attn_q_a.weight", &[5120, 1280][..], false),
                ("attn_q_a_norm.weight", &[1280][..], true),
                ("attn_q_b.weight", &[1280, 32768][..], false),
                ("attn_sinks.weight", &[64][..], true),
                ("ffn_down_shexp.weight", &[2304, 5120][..], false),
                ("ffn_gate_shexp.weight", &[5120, 2304][..], false),
                ("ffn_up_shexp.weight", &[5120, 2304][..], false),
                ("ffn_norm.weight", &[5120][..], true),
            ] {
                require(format!("blk.{layer}.{suffix}"), shape, plain)?;
            }
            for name in ["attn", "ffn"] {
                require(format!("blk.{layer}.hc_{name}_base.weight"), &[24], true)?;
                require(
                    format!("blk.{layer}.hc_{name}_fn.weight"),
                    &[20480, 24],
                    true,
                )?;
                require(format!("blk.{layer}.hc_{name}_scale.weight"), &[3], true)?;
            }
            let experts = config.experts as u64;
            require(format!("blk.{layer}.exp_probs_b.bias"), &[experts], true)?;
            let vision_bias = format!("blk.{layer}.exp_probs_b_vl.bias");
            if package.tensors.contains_key(&vision_bias) {
                require(vision_bias, &[experts], true)?;
            }
            require(
                format!("blk.{layer}.ffn_gate_inp.weight"),
                &[5120, experts],
                false,
            )?;
            require(
                format!("blk.{layer}.ffn_down_exps.weight"),
                &[2304, 5120, experts],
                false,
            )?;
            for name in ["gate", "up"] {
                require(
                    format!("blk.{layer}.ffn_{name}_exps.weight"),
                    &[5120, 2304, experts],
                    false,
                )?;
            }
            if config.kv_sources.contains(&(layer as u32)) {
                require(
                    format!("blk.{layer}.attn_compressor_kv.weight"),
                    &[5120, 512],
                    false,
                )?;
                require(
                    format!("blk.{layer}.attn_compressor_norm.weight"),
                    &[512],
                    true,
                )?;
                if config.compress_ratios[layer] == 2 {
                    require(
                        format!("blk.{layer}.attn_compressor_gate.weight"),
                        &[5120, 512],
                        false,
                    )?;
                }
                require(
                    format!("blk.{layer}.indexer.attn_k.weight"),
                    &[512, 128],
                    false,
                )?;
                require(format!("blk.{layer}.indexer.k_norm.weight"), &[128], true)?;
            }
            if config.index_sources.contains(&(layer as u32)) {
                require(
                    format!("blk.{layer}.indexer.attn_q_b.weight"),
                    &[1280, 4096],
                    false,
                )?;
                require(
                    format!("blk.{layer}.indexer.proj.weight"),
                    &[5120, 32],
                    false,
                )?;
            }
            if super::engram::LAYERS.contains(&layer) {
                require(
                    format!("blk.{layer}.engram_kv.weight"),
                    &[6144, 25600],
                    false,
                )?;
                if engram.combined {
                    require(format!("blk.{layer}.engram_gate.weight"), &[5120, 4], true)?;
                } else {
                    for name in ["q", "k"] {
                        require(
                            format!("blk.{layer}.engram_{name}_norm.weight"),
                            &[5120, 4],
                            true,
                        )?;
                    }
                }
            }
        }
        let resident: Vec<String> = seen.iter().cloned().collect();
        let disk = if engram.combined {
            vec!["per_layer_token_embd.weight".to_owned()]
        } else {
            super::engram::LAYERS
                .iter()
                .map(|i| format!("blk.{i}.engram_embd.weight"))
                .collect()
        };
        for (i, name) in disk.iter().enumerate() {
            let tensor = package.tensor(name)?;
            let (shape, kind) = if engram.combined {
                (
                    [256, engram.rows[0] as u64 + engram.rows[1] as u64],
                    TensorType::IQ4_NL,
                )
            } else {
                ([264, engram.rows[i] as u64], TensorType(24))
            };
            ensure!(
                tensor.info.dimensions == shape && tensor.info.kind == kind,
                "unsupported Engram table {name}"
            );
            seen.insert(name.clone());
        }
        let unknown: Vec<&str> = package
            .tensors
            .keys()
            .filter(|n| !seen.contains(*n))
            .map(String::as_str)
            .collect();
        ensure!(
            unknown.is_empty(),
            "unbound DeepSeek GGUF tensors: {unknown:?}; additional draft/vision tensors require their own graph"
        );
        let bytes = |names: &[String]| {
            names.iter().try_fold(0u64, |total, name| {
                total
                    .checked_add(package.tensors[name].info.bytes)
                    .context("DeepSeek payload size overflow")
            })
        };
        Ok(Self {
            resident_payload_bytes: bytes(&resident)?,
            disk_payload_bytes: bytes(&disk)?,
            resident,
            disk,
        })
    }
}
