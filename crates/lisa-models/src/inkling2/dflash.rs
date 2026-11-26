//! The DFlash2 block-diffusion draft of Inkling 2 Mini: its geometry,
//! configuration contract, tensor schema and the projections the step's
//! draft image repacks. The pinned draft checkpoint and its reference
//! integration are recorded in [provenance](provenance.json).
//!
//! One cycle drafts a block of eight rows (the anchor and seven mask rows)
//! over the committed context and selects candidate trees from the target
//! head's top candidates and the selector's bilinear edges.
use crate::program::Schema;
use anyhow::ensure;

/// Target visits whose inputs (and, for 48, the final-norm output) the draft reads.
pub const TAPS: [usize; 4] = [5, 23, 41, 48];
/// Block rows: the anchor and seven proposals.
pub const BLOCK: usize = 8;
pub const PROPOSALS: usize = BLOCK - 1;
pub const WIDTH: i32 = 2304;
pub const LAYERS: usize = 5;
pub const HEADS: i32 = 32;
pub const KV_HEADS: i32 = 8;
pub const HEAD_DIM: i32 = 128;
pub const INTERMEDIATE: i32 = 6912;
/// Dynamic convolution groups of sixteen channels.
pub const GROUPS: i32 = WIDTH / 16;
pub const TOP_K: i32 = 16;
pub const RANK: i32 = 256;
/// The draft attends to context positions within this window of each row.
pub const WINDOW: usize = 2048;
pub const MASK_TOKEN: u32 = 200056;
pub const VALID_VOCABULARY: i32 = 200058;
pub const PADDED_VOCABULARY: i32 = 201024;
/// RoPE base of the draft heads.
pub const ROPE_BASE: f32 = 10000.;

pub fn validate_config(config: &serde_json::Value) -> anyhow::Result<()> {
    for (key, value) in [
        ("hidden_size", WIDTH as u64),
        ("num_hidden_layers", LAYERS as u64),
        ("num_attention_heads", HEADS as u64),
        ("num_key_value_heads", KV_HEADS as u64),
        ("head_dim", HEAD_DIM as u64),
        ("intermediate_size", INTERMEDIATE as u64),
        ("sliding_window", WINDOW as u64),
        ("vocab_size", PADDED_VOCABULARY as u64),
    ] {
        ensure!(
            config[key].as_u64() == Some(value),
            "unsupported DFlash2 {key}"
        );
    }
    ensure!(
        config["architectures"] == serde_json::json!(["DFlash2DraftModel"])
            && config["is_causal"] == false
            && config["attention_bias"] == false
            && config["hidden_act"] == "silu",
        "unsupported DFlash2 architecture"
    );
    ensure!(
        config["rms_norm_eps"].as_f64() == Some(1e-6)
            && config["rope_parameters"]["rope_theta"].as_f64() == Some(f64::from(ROPE_BASE))
            && config["rope_parameters"]["rope_type"] == "default",
        "unsupported DFlash2 numerical configuration"
    );
    let c = &config["dflash_config"];
    for (key, value) in [
        ("block_size", BLOCK as u64),
        ("conv_group_size", 16),
        ("conv_kernel_size", 2),
        ("mask_token_id", u64::from(MASK_TOKEN)),
        ("selector_rank", RANK as u64),
        ("selector_top_k", TOP_K as u64),
    ] {
        ensure!(c[key].as_u64() == Some(value), "unsupported DFlash2 {key}");
    }
    ensure!(
        c["target_layer_ids"] == serde_json::json!(TAPS)
            && c["final_logit_softcapping"].as_f64() == Some(30.)
            && c["selector_softcap_mode"] == "after_correction",
        "unsupported DFlash2 selector or feature geometry"
    );
    Ok(())
}

pub fn schema() -> Schema {
    let mut s = Schema::new();
    for (n, shape) in [
        ("fc.weight", vec![WIDTH, 4 * WIDTH]),
        ("hidden_norm.weight", vec![WIDTH]),
        ("norm.weight", vec![WIDTH]),
        (
            "candidate_selector.hidden_projection.weight",
            vec![RANK, WIDTH],
        ),
        (
            "candidate_selector.predecessor_codebook",
            vec![PADDED_VOCABULARY, RANK],
        ),
        (
            "candidate_selector.successor_codebook",
            vec![PADDED_VOCABULARY, RANK],
        ),
    ] {
        s.insert(n.into(), shape);
    }
    for l in 0..LAYERS {
        for n in ["input_layernorm", "post_attention_layernorm"] {
            s.insert(format!("layers.{l}.{n}.weight"), vec![WIDTH]);
        }
        for n in ["q_norm", "k_norm"] {
            s.insert(format!("layers.{l}.self_attn.{n}.weight"), vec![HEAD_DIM]);
        }
        for n in ["attention_conv", "mlp_conv"] {
            s.insert(format!("layers.{l}.{n}.base_kernel"), vec![2, 2, WIDTH]);
            s.insert(
                format!("layers.{l}.{n}.kernel_projection.weight"),
                vec![4 * GROUPS, WIDTH],
            );
        }
        for (n, o, i) in [
            ("self_attn.q_proj", HEADS * HEAD_DIM, WIDTH),
            ("self_attn.k_proj", KV_HEADS * HEAD_DIM, WIDTH),
            ("self_attn.v_proj", KV_HEADS * HEAD_DIM, WIDTH),
            ("self_attn.o_proj", WIDTH, HEADS * HEAD_DIM),
            ("mlp.gate_proj", INTERMEDIATE, WIDTH),
            ("mlp.up_proj", INTERMEDIATE, WIDTH),
            ("mlp.down_proj", WIDTH, INTERMEDIATE),
        ] {
            s.insert(format!("layers.{l}.{n}.weight"), vec![o, i]);
        }
    }
    s
}

/// A projection the pinned reference's draft image stores in MLX affine
/// 4-bit groups of 64 (its `af4g64` planes): its plane tensor's name, the
/// checkpoint tensors whose rows it holds in order, its rows and inputs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReferenceProjection {
    pub name: String,
    pub parts: Vec<String>,
    pub rows: i32,
    pub inputs: i32,
}

/// The draft's projections as the pinned reference's image lays them out:
/// each layer's attention dynamic taps, fused q|k|v, attention output, MLP
/// dynamic taps, gate, up and down, then the context projection and the
/// selector's hidden projection.
pub fn reference_projections() -> Vec<ReferenceProjection> {
    let one = |name: &str, rows, inputs| ReferenceProjection {
        name: format!("{name}.af4g64"),
        parts: vec![format!("{name}.weight")],
        rows,
        inputs,
    };
    let mut out = vec![];
    for l in 0..LAYERS {
        let p = format!("layers.{l}.");
        out.push(one(
            &format!("{p}attention_conv.kernel_projection"),
            4 * GROUPS,
            WIDTH,
        ));
        out.push(ReferenceProjection {
            name: format!("{p}self_attn.qkv.af4g64"),
            parts: ["q_proj", "k_proj", "v_proj"]
                .map(|n| format!("{p}self_attn.{n}.weight"))
                .to_vec(),
            rows: (HEADS + 2 * KV_HEADS) * HEAD_DIM,
            inputs: WIDTH,
        });
        out.push(one(
            &format!("{p}self_attn.o_proj"),
            WIDTH,
            HEADS * HEAD_DIM,
        ));
        out.push(one(
            &format!("{p}mlp_conv.kernel_projection"),
            4 * GROUPS,
            WIDTH,
        ));
        out.push(one(&format!("{p}mlp.gate_proj"), INTERMEDIATE, WIDTH));
        out.push(one(&format!("{p}mlp.up_proj"), INTERMEDIATE, WIDTH));
        out.push(one(&format!("{p}mlp.down_proj"), WIDTH, INTERMEDIATE));
    }
    out.push(one("fc", WIDTH, 4 * WIDTH));
    out.push(one("candidate_selector.hidden_projection", RANK, WIDTH));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schema_matches_the_pinned_checkpoint_tensor_count() {
        // 6 shared tensors and 15 per layer: the checkpoint header lists 81.
        assert_eq!(schema().len(), 6 + 15 * LAYERS);
    }

    #[test]
    fn reference_projections_cover_the_schema_projections() {
        let s = schema();
        let projections = reference_projections();
        assert_eq!(projections.len(), 7 * LAYERS + 2);
        let mut parts = 0;
        for p in &projections {
            let rows: i32 = p.parts.iter().map(|n| s[n][0]).sum();
            assert_eq!(rows, p.rows, "{}", p.name);
            assert!(p.parts.iter().all(|n| s[n][1] == p.inputs), "{}", p.name);
            assert!(p.rows % 64 == 0 && p.inputs % 64 == 0, "{}", p.name);
            parts += p.parts.len();
        }
        // Every 2-D weight but the codebooks and the base kernels.
        let linear = s
            .iter()
            .filter(|(n, shape)| shape.len() == 2 && !n.contains("codebook"))
            .count();
        assert_eq!(parts, linear);
    }
}
