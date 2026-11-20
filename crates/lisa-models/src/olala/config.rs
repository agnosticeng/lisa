//! Released Olala mini geometry, validated before importing weights.
use crate::metadata::Metadata;
use anyhow::{Result, ensure};
#[derive(Clone, Debug)]
pub struct Config {
    pub layers: Vec<bool>,
    pub context: usize,
    pub epsilon: f32,
    pub scale: f32,
    pub gate_bias: f32,
    pub softcap: f32,
    pub window: usize,
}
impl Config {
    pub fn from_metadata(m: &impl Metadata) -> Result<Self> {
        ensure!(m.architecture()? == "olala", "expected Olala GGUF");
        for (key, value) in [
            ("block_count", 36),
            ("embedding_length", 1536),
            ("attention.head_count", 48),
            ("attention.head_count_kv", 12),
            ("attention.key_length", 128),
            ("attention.value_length", 128),
            ("expert_count", 256),
            ("expert_used_count", 6),
            ("expert_feed_forward_length", 768),
            ("expert_shared_feed_forward_length", 1536),
            ("expert_shared_count", 1),
            ("expert_gating_func", 2),
            ("moe_latent_size", 384),
            ("ssm.inner_size", 3072),
            ("ssm.state_size", 128),
            ("ssm.group_count", 1),
            ("ssm.time_step_rank", 64),
            ("mamba_mimo_dim", 4),
            ("mamba_headdim", 64),
            ("num_signal_heads", 36),
            ("num_noise_heads", 12),
            ("tpa_rank", 4),
            ("num_rope_angles", 32),
        ] {
            m.expect(&[key], value)?;
        }
        ensure!(
            m.vocabulary_len()? == 151936,
            "unsupported Olala vocabulary"
        );
        for (key, value) in [
            ("zero_centered_gamma", true),
            ("gate_attn", true),
            ("geodesic_update", true),
            ("final_norm", false),
        ] {
            ensure!(m.boolean(&[key])? == value, "unsupported Olala {key}");
        }
        let layers = m.string(&["layer_types"])?;
        ensure!(
            layers == "MMMMVMMMMVMMMMVMMMMVMMMMVMMMMVMMMMVM",
            "unsupported Olala layer schedule"
        );
        let context = usize::try_from(m.integer(&["context_length"])?)?;
        let window = usize::try_from(m.integer(&["attention.sliding_window"])?)?;
        ensure!(
            (1..=100000).contains(&context) && window == 32768,
            "unsupported Olala context geometry"
        );
        let epsilon = m.float(&["attention.layer_norm_rms_epsilon"])?;
        let scale = m.float(&["expert_weights_scale"])?;
        let gate_bias = m.float(&["gate_bias"])?;
        let softcap = m.float(&["attn_logit_softcapping"])?;
        ensure!(
            epsilon == 1e-6 && scale == 2.45 && gate_bias == 1.15 && softcap == 150.0,
            "unsupported Olala arithmetic metadata"
        );
        Ok(Self {
            layers: layers.bytes().map(|v| v == b'M').collect(),
            context,
            epsilon,
            scale,
            gate_bias,
            softcap,
            window,
        })
    }
    pub fn recurrent_floats(&self) -> usize {
        48 * (64 * 128 + 4 * 128 + 4 * 64 + 32)
    }
}

#[derive(Clone, Copy)]
pub enum Role {
    Parameter,
    Projection,
    ExpertUp,
    ExpertDown,
}
/// GGUF logical dimensions (innermost first); device views reverse these axes.
pub fn tensors(c: &Config) -> Vec<(String, Vec<u64>, Role)> {
    let mut out = vec![
        (
            "token_embd.weight".into(),
            vec![1536, 151936],
            Role::Projection,
        ),
        ("output.weight".into(), vec![1536, 151936], Role::Projection),
    ];
    for (i, &mamba) in c.layers.iter().enumerate() {
        let mut add = |name: &str, shape: &[u64], role| {
            out.push((format!("blk.{i}.{name}"), shape.to_vec(), role))
        };
        for name in [
            "geo_mixer_scale",
            "geo_mixer_bias",
            "geo_mlp_scale",
            "geo_mlp_bias",
        ] {
            add(name, &[1], Role::Parameter);
        }
        add(
            "attn_output.weight",
            &[if mamba { 3072 } else { 4608 }, 1536],
            Role::Projection,
        );
        if mamba {
            add("ssm_in.weight", &[1536, 6288], Role::Projection);
            add("ssm_in_dyn.weight", &[1536, 1056], Role::Projection);
            add("ssm_m3_bias", &[128, 8, 48], Role::Parameter);
            add("ssm_m3_mxz", &[64, 8, 48], Role::Parameter);
            add("ssm_mimo_o", &[64, 4, 48], Role::Parameter);
            add("ssm_m3_norms", &[128, 2], Role::Parameter);
            add("ssm_m3_misc", &[48, 2], Role::Parameter);
        } else {
            for (name, width) in [
                ("attn_q", 6144),
                ("attn_wa_k", 48),
                ("attn_wa_v", 48),
                ("attn_wb_k", 512),
                ("attn_wb_v", 512),
                ("attn_shift_k", 12),
                ("attn_shift_v", 12),
                ("attn_lambda", 12),
                ("attn_gate", 4608),
            ] {
                add(&format!("{name}.weight"), &[1536, width], Role::Projection);
            }
            add("attn_q_norm.weight", &[128], Role::Parameter);
            add("attn_k_norm.weight", &[128], Role::Parameter);
            add("attn_softmax_scaler", &[48], Role::Parameter);
        }
        add("ffn_gate_inp.weight", &[1536, 256], Role::Projection);
        add("exp_probs_b", &[256], Role::Parameter);
        add("ffn_latent_down.weight", &[1536, 384], Role::Projection);
        add("ffn_latent_up.weight", &[384, 1536], Role::Projection);
        add("ffn_up_exps.weight", &[384, 768, 256], Role::ExpertUp);
        add("ffn_down_exps.weight", &[768, 384, 256], Role::ExpertDown);
        add("ffn_up_shexp.weight", &[1536, 1536], Role::Projection);
        add("ffn_down_shexp.weight", &[1536, 1536], Role::Projection);
    }
    out
}
