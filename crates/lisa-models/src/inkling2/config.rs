//! Geometry of the original Unsloth Inkling 2 Mini GGUF release.
use crate::metadata::Metadata;
use anyhow::{Result, ensure};
#[derive(Clone, Debug)]
pub struct Config {
    pub context: usize,
}
impl Config {
    pub fn from_metadata(m: &impl Metadata) -> Result<Self> {
        ensure!(
            m.architecture()? == "inkling2",
            "expected original Inkling 2 GGUF layout"
        );
        for (key, value) in [
            ("block_count", 48),
            ("embedding_length", 2304),
            ("num_loops", 3),
            ("loop.core_start", 6),
            ("loop.core_end", 18),
            ("loop.gate_rank", 64),
            ("expert_count", 128),
            ("expert_used_count", 8),
            ("expert_feed_forward_length", 1152),
            ("attention.key_length", 128),
            ("attention.value_length", 128),
            ("attention.relative_length", 16),
            ("attention.relative_extent", 1024),
            ("attention.sliding_window", 1024),
            ("shortconv.l_cache", 4),
            ("attn_res.block_size", 6),
            ("attention.log_scale_floor", 8192),
            ("vocab_size", 201024),
            ("vocab_size_unpadded", 200058),
            ("rope.dimension_count", 0),
        ] {
            m.expect(&[key], value)?;
        }
        for (key, value) in [("attention.head_count", 24), ("attention.head_count_kv", 8)] {
            ensure!(
                m.integers(&[key])? == vec![value; 48],
                "unsupported Inkling {key}"
            );
        }
        for (key, value) in [
            ("attention.layer_norm_rms_epsilon", 1e-6f32),
            ("attention.qk_clip", 22.627416610717773f32),
            ("attention.relative_clip", 8.),
            ("attention.log_scale_alpha", 0.1),
            ("expert_weights_scale", 8.),
            ("activation.situ_beta", 4.),
            ("activation.situ_linear_beta", 25.),
            ("final_logit_softcapping", 30.),
        ] {
            ensure!(
                m.float(&[key])? == value,
                "unsupported Inkling arithmetic: {key}"
            );
        }
        ensure!(
            m.vocabulary_len()? == 200058,
            "unsupported Inkling vocabulary"
        );
        let context = usize::try_from(m.integer(&["context_length"])?)?;
        ensure!(
            context > 0 && context <= 1048576,
            "unsupported Inkling context"
        );
        Ok(Self { context })
    }
}
pub fn tensors() -> Vec<(String, Vec<u64>)> {
    let mut out = vec![];
    for (name, shape) in [
        ("token_embd.weight", vec![2304, 201024]),
        ("output.weight", vec![2304, 201024]),
        ("token_embd_norm.weight", vec![2304]),
        ("output_norm.weight", vec![2304]),
        ("output_res_query.weight", vec![2304]),
        ("loop_res_query.weight", vec![2304, 24]),
        ("loop_kv_res_query.weight", vec![2304, 24]),
    ] {
        out.push((name.into(), shape));
    }
    for i in 0..24 {
        let mut add =
            |name: &str, shape: &[u64]| out.push((format!("blk.{i}.{name}"), shape.to_vec()));
        for (name, cols, rows) in [
            ("attn_q", 2304, 3072),
            ("attn_k", 2304, 1024),
            ("attn_v", 2304, 1024),
            ("attn_gate", 2304, 3072),
            ("attn_output", 3072, 2304),
            ("attn_rel", 2304, 384),
            ("attn_sink_proj", 2304, 24),
            ("ffn_gate_inp", 2304, 128),
        ] {
            add(&format!("{name}.weight"), &[cols, rows]);
        }
        for (name, shape) in [
            ("attn_q_clip", vec![128, 24]),
            ("attn_rel_clip", vec![16, 24]),
            ("attn_rel_bias", vec![16, 1024]),
            ("attn_k_conv", vec![4, 1024]),
            ("attn_v_conv", vec![4, 1024]),
            ("attn_post_conv", vec![4, 2304]),
            ("ffn_post_conv", vec![4, 2304]),
            ("ffn_norm", vec![2304]),
            ("ffn_res_query", vec![2304]),
        ] {
            add(&format!("{name}.weight"), &shape);
        }
        if i % 6 == 5 {
            for k in ["q", "k", "v"] {
                add(&format!("attn_{k}_inp_norm.weight"), &[2304]);
            }
            for k in ["k", "v"] {
                add(&format!("attn_{k}_conv_proj.weight"), &[2304, 32]);
            }
        } else {
            add("attn_norm.weight", &[2304]);
        }
        if i > 0 {
            add(
                "attn_res_query.weight",
                if i % 6 == 5 { &[2304, 3] } else { &[2304] },
            );
        }
        add("exp_probs_b.bias", &[128]);
        for (name, cols) in [("ffn_gate_up_exps", 2304), ("ffn_down_exps", 1152)] {
            add(&format!("{name}.weight"), &[cols, 2304, 128]);
            add(&format!("{name}.scale"), &[128]);
        }
        if (6..18).contains(&i) {
            for ctrl in ["attn", "k", "v", "mlp"] {
                add(&format!("loop_{ctrl}_down.weight"), &[4608, 64]);
                for gate in ["gain", "forget"] {
                    add(&format!("loop_{ctrl}_{gate}.weight"), &[64, 2304]);
                }
            }
        }
    }
    out
}

/// The supported original F16 projector's numerical and geometry contract.
pub fn validate_vision(m: &impl Metadata) -> Result<()> {
    ensure!(
        m.architecture()? == "clip" && m.string(&["projector_type"])? == "inkling2v",
        "expected Inkling 2 vision projector"
    );
    for (key, n) in [
        ("patch_size", 20),
        ("embedding_length", 1024),
        ("feed_forward_length", 4096),
        ("block_count", 23),
        ("attention.head_count", 16),
        ("spatial_merge_size", 2),
        ("projection_dim", 2304),
    ] {
        m.expect(&[key], n)?;
    }
    ensure!(
        m.float(&["attention.layer_norm_epsilon"])? == 1e-5f32,
        "unsupported Inkling vision normalization"
    );
    for key in ["image_mean", "image_std"] {
        ensure!(
            m.floats(&[key])? == vec![0.5; 3],
            "unsupported Inkling vision {key}"
        );
    }
    Ok(())
}
