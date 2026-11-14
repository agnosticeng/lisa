//! Model-owned weight names, shapes and storage requirements.
use super::{config::Config, engram::Layout};
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Storage {
    F32,
    Projection,
    Experts,
    Iq4Nl,
    Mxfp8,
}
pub struct Requirement {
    pub name: String,
    /// Canonical row-major shape, independent of GGUF dimension ordering.
    pub shape: Vec<u64>,
    pub storage: Storage,
    pub optional: bool,
    pub disk: bool,
}
pub fn requirements(config: &Config, engram: &Layout) -> Vec<Requirement> {
    let mut weights = Vec::new();
    let mut optional = Vec::new();
    let mut require = |name: String, shape: &[u64], plain: bool| {
        weights.push(Requirement {
            name,
            shape: shape.iter().rev().copied().collect(),
            storage: if plain {
                Storage::F32
            } else if shape.len() == 3 {
                Storage::Experts
            } else {
                Storage::Projection
            },
            optional: false,
            disk: false,
        });
    };
    require(
        "token_embd.weight".into(),
        &[5120, config.vocab as u64],
        false,
    );
    require("output.weight".into(), &[5120, config.vocab as u64], false);
    require("output_norm.weight".into(), &[5120], true);
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
            require(format!("blk.{layer}.{suffix}"), shape, plain);
        }
        for name in ["attn", "ffn"] {
            require(format!("blk.{layer}.hc_{name}_base.weight"), &[24], true);
            require(
                format!("blk.{layer}.hc_{name}_fn.weight"),
                &[20480, 24],
                true,
            );
            require(format!("blk.{layer}.hc_{name}_scale.weight"), &[3], true);
        }
        let experts = config.experts as u64;
        require(format!("blk.{layer}.exp_probs_b.bias"), &[experts], true);
        let vision_bias = format!("blk.{layer}.exp_probs_b_vl.bias");
        optional.push((vision_bias, vec![experts], true));
        require(
            format!("blk.{layer}.ffn_gate_inp.weight"),
            &[5120, experts],
            false,
        );
        require(
            format!("blk.{layer}.ffn_down_exps.weight"),
            &[2304, 5120, experts],
            false,
        );
        for name in ["gate", "up"] {
            require(
                format!("blk.{layer}.ffn_{name}_exps.weight"),
                &[5120, 2304, experts],
                false,
            );
        }
        if config.kv_sources.contains(&(layer as u32)) {
            require(
                format!("blk.{layer}.attn_compressor_kv.weight"),
                &[5120, 512],
                false,
            );
            require(
                format!("blk.{layer}.attn_compressor_norm.weight"),
                &[512],
                true,
            );
            if config.compress_ratios[layer] == 2 {
                require(
                    format!("blk.{layer}.attn_compressor_gate.weight"),
                    &[5120, 512],
                    false,
                );
            }
            require(
                format!("blk.{layer}.indexer.attn_k.weight"),
                &[512, 128],
                false,
            );
            require(format!("blk.{layer}.indexer.k_norm.weight"), &[128], true);
        }
        if config.index_sources.contains(&(layer as u32)) {
            require(
                format!("blk.{layer}.indexer.attn_q_b.weight"),
                &[1280, 4096],
                false,
            );
            require(
                format!("blk.{layer}.indexer.proj.weight"),
                &[5120, 32],
                false,
            );
        }
        if super::engram::LAYERS.contains(&layer) {
            require(
                format!("blk.{layer}.engram_kv.weight"),
                &[6144, 25600],
                false,
            );
            if engram.combined {
                require(format!("blk.{layer}.engram_gate.weight"), &[5120, 4], true);
            } else {
                for name in ["q", "k"] {
                    require(
                        format!("blk.{layer}.engram_{name}_norm.weight"),
                        &[5120, 4],
                        true,
                    );
                }
            }
        }
    }

    for (name, shape, _) in optional {
        weights.push(Requirement {
            name,
            shape,
            storage: Storage::F32,
            optional: true,
            disk: false,
        });
    }
    if engram.combined {
        weights.push(Requirement {
            name: "per_layer_token_embd.weight".into(),
            shape: vec![engram.rows[0] as u64 + engram.rows[1] as u64, 256],
            storage: Storage::Iq4Nl,
            optional: false,
            disk: true,
        });
    } else {
        for (i, layer) in super::engram::LAYERS.into_iter().enumerate() {
            weights.push(Requirement {
                name: format!("blk.{layer}.engram_embd.weight"),
                shape: vec![engram.rows[i] as u64, 264],
                storage: Storage::Mxfp8,
                optional: false,
                disk: true,
            });
        }
    }
    weights
}
