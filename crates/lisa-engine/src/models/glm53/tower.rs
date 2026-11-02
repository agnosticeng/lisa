//! The `glm5_next` text tower, following ds4's F32 inference graph.
//!
//! Model configuration and ds4 source define the execution path; the local
//! GLM specs are supporting notes rather than the implementation authority.
//! Embed -> 45 blocks (mHC collapse -> RMS -> KDA/DSA -> expand, then FFN
//! half) -> stream-mean -> final RMS -> lm_head. Caches reuse the shared
//! `LayerCache` variants: KDA layers use `Linear(GdnCache)` (conv window +
//! F32 recurrence state), DSA layers use `Full(FullAttentionCache)`
//! (`keys` = latents `[B,Tc,512]`, `offset` = cached length).

#[cfg(test)]
use std::collections::HashMap;
use std::path::Path;
#[cfg(test)]
use std::path::PathBuf;

use crate::core::cache::{FullAttentionCache, GdnCache, IndexerTape, LayerCache};

/// Counts the incremental (t==1) forwards; shared with the DSA dec captures.
pub(crate) static INC_STEP: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

#[cfg(test)]
fn capture_mtp_verify_layer_state(
    prefix: &std::ffi::OsStr,
    capture_index: usize,
    layer: usize,
    state: &Array,
) -> anyhow::Result<()> {
    let values = state
        .as_dtype(Dtype::Float32)?
        .contiguous()?
        .reshape(&[state.size() as i32])?
        .to_vec1::<f32>()?;
    let bytes: Vec<u8> = values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();
    let prefix = std::path::PathBuf::from(prefix).display().to_string();
    std::fs::write(
        format!("{prefix}.layer-call-{capture_index:03}-L{layer:02}.f32"),
        bytes,
    )?;
    Ok(())
}

#[cfg(test)]
fn capture_single_step_layer_state(
    prefix: &std::ffi::OsStr,
    position: usize,
    layer: usize,
    stage: Option<&str>,
    state: &Array,
) -> anyhow::Result<()> {
    let values = state
        .as_dtype(Dtype::Float32)?
        .contiguous()?
        .reshape(&[state.size() as i32])?
        .to_vec1::<f32>()?;
    let bytes: Vec<u8> = values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();
    let prefix = std::path::PathBuf::from(prefix).display().to_string();
    let suffix = stage.map_or_else(String::new, |stage| format!("-{stage}"));
    std::fs::write(
        format!("{prefix}.layer-step-pos-{position:05}{suffix}-L{layer:02}.f32"),
        bytes,
    )?;
    Ok(())
}
use super::mhc::{mm, narrow};
use crate::core::loader::{self, LazyTensorStore, Weights};
use crate::core::quant::{QuantizedEmbedding, QuantizedLinear};
use lisa_mlx::{Array, Dtype};

use super::config::ModelConfig;
use super::kda::Kda;
use super::mhc::Hc;
use super::moe::{DenseMlp, MoE};
use super::sparse::Dsa;
use crate::core::capture::Capture;

const DS4_GLM53_NATIVE_BATCH_ROW_LIMIT: usize = 2048;
const DS4_GLM53_PREFILL_CHUNK_TOKENS: usize = 2048;
const DS4_GLM53_DEFAULT_CONTEXT: usize = 32_768;

fn ds4_native_session_batch_position_supported(
    rows: usize,
    max_position: usize,
    dense_compact_limit: usize,
) -> bool {
    (2..=DS4_GLM53_NATIVE_BATCH_ROW_LIMIT).contains(&rows) && max_position < dense_compact_limit
}

enum Attn {
    Kda(Kda),
    Dsa(Dsa),
}

enum Mlp {
    Dense(DenseMlp),
    Sparse(MoE),
}

struct Layer {
    attn: Attn,
    mlp: Mlp,
    hc_attn: Hc,
    hc_ffn: Hc,
    in_norm: Array,
    post_norm: Array,
}

struct MtpBlock {
    eh_proj: Array,
    enorm: Array,
    hnorm: Array,
    attn_norm: Array,
    ffn_norm: Array,
    shared_head_norm: Array,
    attn: Dsa,
    mlp: MoE,
}

impl MtpBlock {
    fn load(
        w: &mut Weights,
        config: &ModelConfig,
        lazy_store: Option<std::sync::Arc<LazyTensorStore>>,
        map_all_expert_weights: bool,
    ) -> anyhow::Result<Self> {
        let h = config.hidden_size;
        let eps = config.rms_norm_eps;
        let n_layers = config.num_hidden_layers;
        let p = format!("model.layers.{n_layers}");
        let take = |w: &mut Weights, name: &str| -> anyhow::Result<Array> {
            w.remove(name)
                .ok_or_else(|| anyhow::anyhow!("glm53: tensor {name} missing"))
        };
        let eh_proj = take(w, &format!("{p}.eh_proj.weight"))?;
        anyhow::ensure!(
            eh_proj.rank() == 2 && eh_proj.dim(0) == h as i32 && eh_proj.dim(1) == (2 * h) as i32,
            "glm53 MTP eh_proj shape mismatch: {:?}",
            eh_proj.shape()
        );
        let enorm = take(w, &format!("{p}.enorm.weight"))?.as_dtype(Dtype::Float32)?;
        let hnorm = take(w, &format!("{p}.hnorm.weight"))?.as_dtype(Dtype::Float32)?;
        let attn_norm =
            take(w, &format!("{p}.input_layernorm.weight"))?.as_dtype(Dtype::Float32)?;
        let ffn_norm =
            take(w, &format!("{p}.post_attention_layernorm.weight"))?.as_dtype(Dtype::Float32)?;
        let shared_head_norm =
            take(w, &format!("{p}.shared_head.norm.weight"))?.as_dtype(Dtype::Float32)?;
        let attn = Dsa::load::<Weights>(
            w,
            &format!("{p}.self_attn"),
            config.num_attention_heads,
            config.qk_nope_head_dim,
            config.v_head_dim,
            config.index_n_heads,
            config.index_head_dim,
            config.index_topk,
            config.index_kpool,
            config.index_kpool_always_select_tail,
            eps,
        )?;
        anyhow::ensure!(
            config.is_sparse_mlp(n_layers),
            "glm53 MTP block must use sparse MoE"
        );
        let mlp = MoE::load::<Weights>(
            w,
            &format!("{p}.mlp"),
            config.n_routed_experts,
            config.num_experts_per_tok,
            config.n_group,
            config.topk_group,
            config.routed_scaling_factor,
            config.norm_topk_prob,
            config.n_shared_experts,
            config.moe_intermediate_size,
            h,
            config.swiglu_limit,
            lazy_store,
            map_all_expert_weights,
        )?;
        Ok(Self {
            eh_proj,
            enorm,
            hnorm,
            attn_norm,
            ffn_norm,
            shared_head_norm,
            attn,
            mlp,
        })
    }

    fn forward(
        &self,
        config: &ModelConfig,
        embed: &QuantizedEmbedding,
        lm_head: &QuantizedLinear,
        tape: &mut IndexerTape,
        target_multi: &Array,
        next_token: u32,
        position: usize,
        min_position: usize,
    ) -> anyhow::Result<Array> {
        anyhow::ensure!(
            target_multi.dim(0) == 1 && target_multi.dim(1) == 1,
            "glm53 MTP expects one target hidden row"
        );
        let h = config.hidden_size;
        let streams = config.hc_mult;
        anyhow::ensure!(
            target_multi.dim(2) == (h * streams) as i32,
            "glm53 MTP target hidden width mismatch"
        );
        anyhow::ensure!(
            (next_token as usize) < config.vocab_size,
            "glm53 MTP token {next_token} exceeds vocabulary size {}",
            config.vocab_size
        );

        let ids = Array::from_slice(&[next_token as i32], &[1i32, 1]);
        let embedded = embed.forward(&ids)?.as_dtype(Dtype::Float32)?;
        let embedded = mtp_rms(&embedded, &self.enorm, config.rms_norm_eps)?;

        // ds4 reduces target mHC streams with uniform 1/C weights via its
        // fused weighted-sum kernel; preserve that kernel and accumulation order.
        anyhow::ensure!(streams == 4, "glm53 MTP mHC width must be four");
        let multi =
            target_multi
                .as_dtype(Dtype::Float32)?
                .reshape(&[1, 1, streams as i32, h as i32])?;
        let weights = Array::from_slice(&[0.25f32; 4], &[1, 1, 4]);
        let stream = lisa_mlx::Stream::thread_local_or_default();
        let hidden =
            lisa_mlx::models::glm53::kernels::hc_weighted_sum(&multi, &weights, &stream)
                .ok_or_else(|| anyhow::anyhow!("glm53 MTP target mHC reduction kernel declined"))?;
        let hidden = mtp_rms(&hidden, &self.hnorm, config.rms_norm_eps)?;

        let joined = Array::concatenate(&[&embedded, &hidden], 2)?;
        let rows = joined.reshape(&[1, joined.dim(2)])?;
        let projected = if self.eh_proj.dtype() == Dtype::Bfloat16 {
            let stream = lisa_mlx::Stream::thread_local_or_default();
            lisa_mlx::models::glm53::kernels::glm53_bf16_gemv(&self.eh_proj, &rows, &stream)
                .ok_or_else(|| anyhow::anyhow!("GLM MTP BF16 eh_proj GEMV declined"))?
        } else {
            let weight = self
                .eh_proj
                .transpose_axes(&[1, 0])?
                .as_dtype(Dtype::Float32)?;
            mm(&joined, &weight)?
        };
        let x = projected.reshape(&[1, 1, h as i32])?;

        let attn_in = mtp_rms(&x, &self.attn_norm, config.rms_norm_eps)?;
        let attended = self
            .attn
            .forward_mtp(&attn_in, tape, position, min_position)?;
        let stream = lisa_mlx::Stream::thread_local_or_default();
        let (after_attn, ffn_in) = lisa_mlx::models::glm53::kernels::glm53_mtp_add_rms_norm(
            &x,
            &attended,
            &self.ffn_norm,
            config.rms_norm_eps,
            &stream,
        )
        .ok_or_else(|| anyhow::anyhow!("GLM MTP DS4 add+RMSNorm kernel declined"))?;
        #[cfg(all(test, feature = "glm53-fixtures"))]
        if position == 15 || std::env::var_os("LISA_GLM53_MTP_RESIDUAL_CAPTURE_DIR").is_some() {
            capture_or_assert_mtp_residual_ds4(
                &x,
                &attended,
                &self.ffn_norm,
                &after_attn,
                &ffn_in,
                config.rms_norm_eps,
            )?;
        }
        let (routed, shared) = self.mlp.forward_parts_with_capture(&ffn_in, None, 0)?;
        #[cfg(all(test, feature = "glm53-fixtures"))]
        if position == 15
            || std::env::var_os("LISA_GLM53_MTP_ROUTER_CAPTURE_DIR").is_some()
            || std::env::var_os("LISA_GLM53_MTP_ROUTED_CAPTURE_DIR").is_some()
        {
            self.mlp.capture_or_assert_mtp_router_ds4(&ffn_in)?;
            self.mlp
                .capture_or_assert_mtp_routed_ds4(&ffn_in, &routed)?;
        }
        let shared =
            shared.ok_or_else(|| anyhow::anyhow!("GLM MTP checkpoint has no shared expert"))?;
        #[cfg(all(test, feature = "glm53-fixtures"))]
        if position == 15 || std::env::var_os("LISA_GLM53_MTP_SHARED_CAPTURE_DIR").is_some() {
            self.mlp
                .capture_or_assert_mtp_shared_ds4(&ffn_in, &shared)?;
        }
        #[cfg(test)]
        if position == 15 {
            let combined = self.mlp.forward_with_capture(&ffn_in, None, 0)?;
            let branch_sum = routed.add(&shared)?;
            let combined_values = combined.to_vec1::<f32>()?;
            let branch_values = branch_sum.to_vec1::<f32>()?;
            let delta2 = combined_values
                .iter()
                .zip(&branch_values)
                .map(|(actual, expected)| f64::from(actual - expected).powi(2))
                .sum::<f64>();
            let expected2 = branch_values
                .iter()
                .map(|value| f64::from(*value).powi(2))
                .sum::<f64>();
            let max_abs = combined_values
                .iter()
                .zip(&branch_values)
                .map(|(actual, expected)| (actual - expected).abs())
                .fold(0.0f32, f32::max);
            let rel_l2 = (delta2 / expected2.max(f64::MIN_POSITIVE)).sqrt();
            anyhow::ensure!(
                max_abs <= 1e-4 && rel_l2 <= 1e-5,
                "fused production MoE differs from the DS4-validated branches: max_abs={max_abs:e}, rel_l2={rel_l2:e}"
            );
            eprintln!(
                "[glm53-mtp-moe-fused] output max_abs={max_abs:.3e} rel_l2={rel_l2:.3e} against DS4-validated branch sum"
            );
        }
        let stream = lisa_mlx::Stream::thread_local_or_default();
        let next_sum = lisa_mlx::models::glm53::kernels::glm53_mtp_add3(
            &after_attn,
            &routed,
            &shared,
            &stream,
        )
        .ok_or_else(|| anyhow::anyhow!("GLM MTP DS4 ordered add3 kernel declined"))?;
        let next = mtp_rms(&next_sum, &self.shared_head_norm, config.rms_norm_eps)?;
        #[cfg(all(test, feature = "glm53-fixtures"))]
        // The checked-in tail fixture is anchored at position 15. Keep the
        // capture hook on that same row; allowing the test's later generated
        // feedback rows to overwrite it silently changes the fixture input.
        if position == 15 {
            capture_or_assert_mtp_tail_ds4(
                &after_attn,
                &routed,
                &shared,
                &self.shared_head_norm,
                &next_sum,
                &next,
                config.rms_norm_eps,
            )?;
        }
        // DS4's shared output head uses the same native Q8 dispatcher in
        // target and MTP execution, including the vocabulary-sized reduction.
        let logits = lm_head.forward_glm53_prefill(&next)?;
        #[cfg(all(test, feature = "glm53-fixtures"))]
        if position == 15
            || (super::sparse::mtp_real_target_fixtures_enabled()
                && (position == 5 || position == 6))
        {
            capture_or_assert_mtp_lm_head_ds4(&next, lm_head, &logits, position)?;
        }
        Ok(logits)
    }
}

#[cfg(all(test, feature = "glm53-fixtures"))]
fn capture_or_assert_mtp_residual_ds4(
    x: &Array,
    attended: &Array,
    weight: &Array,
    sum: &Array,
    norm: &Array,
    eps: f32,
) -> anyhow::Result<()> {
    const WIDTH: usize = 4096;
    const PART_BYTES: usize = WIDTH * std::mem::size_of::<f32>();
    const FIXTURE: &[u8] = include_bytes!("fixtures/mtp_residual_layer45_ds4.bin");
    let values = [
        x.to_vec1::<f32>()?,
        attended.to_vec1::<f32>()?,
        weight.to_vec1::<f32>()?,
        sum.to_vec1::<f32>()?,
        norm.to_vec1::<f32>()?,
    ];
    anyhow::ensure!(
        values.iter().all(|values| values.len() == WIDTH),
        "GLM MTP residual DS4 fixture shape mismatch"
    );
    let mut payload = Vec::with_capacity(5 * PART_BYTES + 4);
    for values in &values {
        for value in values {
            payload.extend_from_slice(&value.to_le_bytes());
        }
    }
    payload.extend_from_slice(&eps.to_le_bytes());

    if let Some(path) = std::env::var_os("LISA_GLM53_MTP_RESIDUAL_CAPTURE_DIR") {
        let path = Path::new(&path);
        std::fs::create_dir_all(path)?;
        for (name, values) in [
            ("x.f32", &values[0]),
            ("attended.f32", &values[1]),
            ("ffn_norm.f32", &values[2]),
            ("lisa_sum.f32", &values[3]),
            ("lisa_norm.f32", &values[4]),
        ] {
            let bytes: Vec<u8> = values
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect();
            std::fs::write(path.join(name), bytes)?;
        }
        std::fs::write(path.join("eps.f32"), eps.to_le_bytes())?;
        return Ok(());
    }

    anyhow::ensure!(
        FIXTURE.len() == payload.len(),
        "DS4 MTP residual fixture has an invalid size"
    );
    for (index, (actual, expected)) in payload
        .chunks_exact(4)
        .zip(FIXTURE.chunks_exact(4))
        .enumerate()
    {
        anyhow::ensure!(
            actual == expected,
            "layer-45 MTP residual DS4 fixture differs at F32 word {index}"
        );
    }
    eprintln!("[glm53-mtp-residual] add and RMSNorm outputs match DS4 bit-for-bit");
    Ok(())
}

#[cfg(all(test, feature = "glm53-fixtures"))]
fn capture_or_assert_mtp_tail_ds4(
    residual: &Array,
    routed: &Array,
    shared: &Array,
    head_norm_weight: &Array,
    sum: &Array,
    normalized: &Array,
    eps: f32,
) -> anyhow::Result<()> {
    const WIDTH: usize = 4096;
    const PART_BYTES: usize = WIDTH * std::mem::size_of::<f32>();
    const FIXTURE: &[u8] = include_bytes!("fixtures/mtp_tail_layer45_ds4.bin");
    let values = [
        residual.to_vec1::<f32>()?,
        routed.to_vec1::<f32>()?,
        shared.to_vec1::<f32>()?,
        head_norm_weight.to_vec1::<f32>()?,
        sum.to_vec1::<f32>()?,
        normalized.to_vec1::<f32>()?,
    ];
    anyhow::ensure!(
        values.iter().all(|values| values.len() == WIDTH),
        "GLM MTP tail DS4 fixture shape mismatch"
    );
    let mut payload = Vec::with_capacity(6 * PART_BYTES + 4);
    for values in &values {
        for value in values {
            payload.extend_from_slice(&value.to_le_bytes());
        }
    }
    payload.extend_from_slice(&eps.to_le_bytes());

    if let Some(path) = std::env::var_os("LISA_GLM53_MTP_TAIL_CAPTURE_DIR") {
        let path = Path::new(&path);
        std::fs::create_dir_all(path)?;
        for (name, values) in [
            ("residual.f32", &values[0]),
            ("routed.f32", &values[1]),
            ("shared.f32", &values[2]),
            ("head_norm.f32", &values[3]),
            ("lisa_sum.f32", &values[4]),
            ("lisa_norm.f32", &values[5]),
        ] {
            let bytes: Vec<u8> = values
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect();
            std::fs::write(path.join(name), bytes)?;
        }
        std::fs::write(path.join("eps.f32"), eps.to_le_bytes())?;
        return Ok(());
    }

    anyhow::ensure!(
        FIXTURE.len() == payload.len(),
        "DS4 MTP tail fixture has an invalid size"
    );
    for (index, (actual, expected)) in payload
        .chunks_exact(4)
        .zip(FIXTURE.chunks_exact(4))
        .enumerate()
    {
        anyhow::ensure!(
            actual == expected,
            "layer-45 MTP tail DS4 fixture differs at F32 word {index}"
        );
    }
    eprintln!("[glm53-mtp-tail] residual and head norm outputs match DS4 bit-for-bit");
    Ok(())
}

#[cfg(all(test, feature = "glm53-fixtures"))]
fn capture_or_assert_mtp_lm_head_ds4(
    input: &Array,
    lm_head: &QuantizedLinear,
    actual_logits: &Array,
    position: usize,
) -> anyhow::Result<()> {
    const HIDDEN: usize = 4096;
    let fixture: &[u8] = if super::sparse::mtp_real_target_fixtures_enabled() && position == 6 {
        include_bytes!("fixtures/mtp_composed_natural_target_step6_ds4.bin")
    } else if super::sparse::mtp_real_target_fixtures_enabled() {
        include_bytes!("fixtures/mtp_composed_natural_target_ds4.bin")
    } else {
        include_bytes!("fixtures/mtp_lm_head_layer45_ds4.bin")
    };
    let vocab = lm_head.dims_out();
    anyhow::ensure!(
        input.size() == HIDDEN && lm_head.group_size == 64 && lm_head.bits == 8,
        "layer-45 MTP output-head fixture geometry differs from DS4"
    );
    let input_values = input.as_dtype(Dtype::Float32)?.to_vec1::<f32>()?;
    let actual_values = actual_logits.as_dtype(Dtype::Float32)?.to_vec1::<f32>()?;
    anyhow::ensure!(
        input_values.len() == HIDDEN && actual_values.len() == vocab,
        "layer-45 MTP output-head fixture shape mismatch"
    );
    let encode_f32 = |values: &[f32]| -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect()
    };
    let encode_u32 = |values: &[u32]| -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect()
    };

    if let Some(path) = std::env::var_os("LISA_GLM53_MTP_LM_HEAD_CAPTURE_DIR") {
        let path = Path::new(&path);
        std::fs::create_dir_all(path)?;
        std::fs::write(
            path.join("geometry.u32"),
            [
                HIDDEN as u32,
                vocab as u32,
                lm_head.group_size as u32,
                lm_head.bits as u32,
            ]
            .into_iter()
            .flat_map(u32::to_le_bytes)
            .collect::<Vec<_>>(),
        )?;
        std::fs::write(path.join("input.f32"), encode_f32(&input_values))?;
        std::fs::write(
            path.join("weight.u32"),
            encode_u32(&lm_head.weight.to_vec1::<u32>()?),
        )?;
        std::fs::write(
            path.join("scales.f32"),
            encode_f32(&lm_head.scales.as_dtype(Dtype::Float32)?.to_vec1::<f32>()?),
        )?;
        std::fs::write(
            path.join("biases.f32"),
            encode_f32(&lm_head.biases.as_dtype(Dtype::Float32)?.to_vec1::<f32>()?),
        )?;
        std::fs::write(path.join("lisa_logits.f32"), encode_f32(&actual_values))?;
        return Ok(());
    }

    let header_bytes = 4usize;
    let input_bytes = HIDDEN * 4;
    anyhow::ensure!(
        fixture.len() == header_bytes + input_bytes + vocab * 4,
        "DS4 layer-45 MTP output-head fixture has an invalid size"
    );
    let expected_vocab = u32::from_le_bytes(fixture[..header_bytes].try_into().unwrap()) as usize;
    anyhow::ensure!(
        expected_vocab == vocab,
        "MTP output-head vocabulary changed"
    );
    let input_end = header_bytes + input_bytes;
    let expected_input = fixture[header_bytes..input_end]
        .chunks_exact(4)
        .map(|bytes| f32::from_le_bytes(bytes.try_into().expect("F32 input word")))
        .collect::<Vec<_>>();
    let input_max_abs = input_values
        .iter()
        .zip(&expected_input)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0f32, f32::max);
    if super::sparse::mtp_real_target_fixtures_enabled() {
        anyhow::ensure!(
            input_max_abs <= 4e-6,
            "composed real-target MTP hidden differs from DS4: max_abs={input_max_abs:e}"
        );
    } else {
        anyhow::ensure!(
            input_values
                .iter()
                .zip(&expected_input)
                .all(|(actual, expected)| actual.to_bits() == expected.to_bits()),
            "layer-45 MTP output-head input changed"
        );
    }
    let expected_logits = fixture[input_end..]
        .chunks_exact(4)
        .map(|bytes| f32::from_le_bytes(bytes.try_into().expect("F32 logit word")))
        .collect::<Vec<_>>();
    let max_abs = actual_values
        .iter()
        .zip(&expected_logits)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0f32, f32::max);
    let delta2 = actual_values
        .iter()
        .zip(&expected_logits)
        .map(|(actual, expected)| f64::from(actual - expected).powi(2))
        .sum::<f64>();
    let ref2 = expected_logits
        .iter()
        .map(|expected| f64::from(*expected).powi(2))
        .sum::<f64>();
    let rel_l2 = (delta2 / ref2.max(f64::MIN_POSITIVE)).sqrt();
    let (max_abs_limit, rel_l2_limit) = if super::sparse::mtp_real_target_fixtures_enabled() {
        (1e-5, 1e-6)
    } else {
        (1e-4, 1e-5)
    };
    anyhow::ensure!(
        max_abs <= max_abs_limit && rel_l2 <= rel_l2_limit,
        "layer-45 MTP output-head logits differ from DS4: max_abs={max_abs:e}, rel_l2={rel_l2:e}"
    );
    anyhow::ensure!(
        actual_values
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(i, _)| i)
            == expected_logits
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map(|(i, _)| i),
        "layer-45 MTP output-head argmax differs from DS4"
    );
    eprintln!(
        "[glm53-mtp-lm-head] vocab={vocab}; logits max_abs={max_abs:.3e} rel_l2={rel_l2:.3e}; argmax matches DS4 (input max_abs={input_max_abs:.3e})"
    );
    Ok(())
}

pub struct Tower {
    config: ModelConfig,
    context_capacity: usize,
    pub(crate) capture: Option<Capture>,
    prefill_mode: bool,
    prefill_final_chunk: bool,
    embed: QuantizedEmbedding,
    layers: Vec<Layer>,
    final_norm: Array,
    lm_head: QuantizedLinear,
    mtp: Option<MtpBlock>,
    _expert_residency_set: Option<std::sync::Arc<lisa_mlx::runtime::ResidencySet>>,
    n_layers: usize,
    // Dropped after model arrays and residency resources release their owners.
    _resident_memory: Option<lisa_mlx::runtime::ResidentMemoryReservation>,
}

fn rms(x: &Array, w: &Array, eps: f32) -> anyhow::Result<Array> {
    let xf = x.as_dtype(Dtype::Float32)?;
    let ms = xf.multiply(&xf)?.contiguous()?.mean_axis(2, true)?; // [B,T,1]
    let n = xf.multiply(&ms.add(&Array::from_f32_as(eps, Dtype::Float32))?.rsqrt()?)?;
    Ok(n.multiply(&w.as_dtype(Dtype::Float32)?)?)
}

fn final_rms(x: &Array, w: &Array, eps: f32) -> anyhow::Result<Array> {
    let stream = lisa_mlx::Stream::thread_local_or_default();
    lisa_mlx::models::glm53::kernels::glm53_ds4_rms_norm(x, w, eps, &stream).ok_or_else(|| {
        anyhow::anyhow!(
            "GLM final DS4-order RMSNorm declined for shape {:?}",
            x.shape()
        )
    })
}

fn final_hc_mean(state: &Array) -> anyhow::Result<Array> {
    let shape = state.shape();
    anyhow::ensure!(
        shape.len() == 4 && shape[2] == 4,
        "GLM final mHC collapse expects [B,T,4,H], got {shape:?}"
    );
    let weight_count = usize::try_from(shape[0] * shape[1] * shape[2])?;
    let weights = Array::from_slice(
        &vec![0.25f32; weight_count],
        &[shape[0], shape[1], shape[2]],
    );
    let stream = lisa_mlx::Stream::thread_local_or_default();
    lisa_mlx::models::glm53::kernels::hc_weighted_sum(state, &weights, &stream)
        .ok_or_else(|| anyhow::anyhow!("GLM final DS4-order mHC collapse declined"))
}

fn mtp_rms(x: &Array, w: &Array, eps: f32) -> anyhow::Result<Array> {
    let stream = lisa_mlx::Stream::thread_local_or_default();
    lisa_mlx::models::glm53::kernels::glm53_mtp_rms_norm(x, w, eps, &stream).ok_or_else(|| {
        anyhow::anyhow!(
            "GLM MTP ds4-order RMSNorm declined for shape {:?}",
            x.shape()
        )
    })
}

impl Tower {
    pub fn load(dir: &Path, config: ModelConfig) -> anyhow::Result<Self> {
        // DS4's resident graph owns a small set of immutable model views. The
        // aligned cache preserves the original MLX payloads while allowing
        // every expert address to remain valid for the model's lifetime.
        let checkpoint = loader::Checkpoint::open(dir)?;
        let store = std::sync::Arc::new(LazyTensorStore::open_resident(dir)?);
        let weight_bytes = store.resident_nbytes();
        let budget = lisa_mlx::memory::ram_budget();
        anyhow::ensure!(
            weight_bytes < budget,
            "GLM resident weights require {:.2} GiB, exceeding the {:.2} GiB RAM budget",
            weight_bytes as f64 / (1u64 << 30) as f64,
            budget as f64 / (1u64 << 30) as f64,
        );
        let reservation = lisa_mlx::memory::reserve_model_memory(weight_bytes, budget)?;
        let mut weights = Weights::new();
        for original in checkpoint.weight_map.keys() {
            let Some(name) = loader::sanitize_name(original) else {
                continue;
            };
            if loader::is_glm_routed_expert_tensor(&name) || weights.contains_key(&name) {
                continue;
            }
            // The resident reader rejects missing or unaligned ranges. There
            // is no individual-allocation fallback on this execution path.
            weights.insert(name.clone(), store.get_mapped(&name)?);
        }
        let mut tower =
            Self::from_weights_with_lazy_experts(weights, config, Some(store.clone()), true, true)?;
        tower._resident_memory = Some(reservation);
        eprintln!(
            "[glm53-loader] resident MLX model: {:.2} GiB, {} shared views, GPU expert routing",
            weight_bytes as f64 / (1u64 << 30) as f64,
            store.resident_buffers().len(),
        );
        Ok(tower)
    }

    #[cfg(test)]
    fn load_with_expert_residency(
        dir: &Path,
        config: ModelConfig,
        enable_expert_residency: bool,
    ) -> anyhow::Result<Self> {
        Self::load_with_expert_residency_mode(dir, config, enable_expert_residency, true, false)
    }

    #[cfg(test)]
    fn load_with_expert_residency_mode(
        dir: &Path,
        config: ModelConfig,
        enable_expert_residency: bool,
        request_residency: bool,
        enable_bounded_cache_residency: bool,
    ) -> anyhow::Result<Self> {
        let checkpoint = loader::Checkpoint::open(dir)?;
        let lazy_store = std::sync::Arc::new(LazyTensorStore::from_checkpoint(dir, &checkpoint)?);
        let mut shard_order: Vec<String> = Vec::new();
        for shard in checkpoint.weight_map.values() {
            if !shard_order.contains(shard) {
                shard_order.push(shard.clone());
            }
        }
        let mut w: Weights = HashMap::new();
        for shard_name in &shard_order {
            let shard = lazy_store.shard(shard_name)?;
            for name in shard.tensor_names() {
                if let Some(sanitized) = loader::sanitize_name(name) {
                    if loader::is_glm_routed_expert_tensor(&sanitized) {
                        continue;
                    }
                    if w.contains_key(&sanitized) {
                        continue;
                    }
                    let (bytes, dtype, shape) = shard
                        .tensor_bytes(name)
                        .ok_or_else(|| anyhow::anyhow!("tensor {name} missing from shard"))?;
                    // Full-bank diagnostics already retain every routed
                    // expert shard. Reuse those exact backing buffers for
                    // ordinary tensors in the same safetensors file instead
                    // of creating overlapping page-window Metal buffers.
                    let shard_view = if enable_expert_residency {
                        shard.tensor_array_shard_view(name)?
                    } else {
                        None
                    };
                    let individual_view = if shard_view.is_none() {
                        shard.tensor_array_individual(name)?
                    } else {
                        None
                    };
                    let window_view = if shard_view.is_none() && individual_view.is_none() {
                        shard.tensor_array(name)?
                    } else {
                        None
                    };
                    let arr = if let Some(mapped) = shard_view.or(individual_view).or(window_view) {
                        mapped
                    } else {
                        if lisa_mlx::trace::enabled_for("glm53-loader") {
                            eprintln!(
                                "[glm53-loader] copying tensor {name} from {shard_name}: no legal mapped Metal window"
                            );
                        }
                        loader::array_from_bytes(bytes, dtype, shape)
                            .map_err(|e| anyhow::anyhow!("loading GLM tensor {sanitized}: {e}"))?
                    };
                    w.insert(sanitized, arr);
                }
            }
        }
        let mut tower = Self::from_weights_with_lazy_experts(
            w,
            config,
            Some(lazy_store.clone()),
            enable_expert_residency,
            request_residency,
        )?;
        let sparse_layers = tower
            .layers
            .iter()
            .filter(|layer| matches!(layer.mlp, Mlp::Sparse(_)))
            .count();
        let sparse_count = sparse_layers + usize::from(tower.mtp.is_some());
        if sparse_count > 0 {
            let ram_cap = lisa_mlx::memory::ram_budget();
            let available_ram = lisa_mlx::memory::available_ram().unwrap_or(0);
            let cache_headroom =
                available_ram.saturating_sub(lisa_mlx::memory::ram_reserve_bytes());
            let expert_cache_budget = (ram_cap / 3).min(cache_headroom);
            let per_pool_budget = expert_cache_budget / sparse_count;
            for layer in &tower.layers {
                if let Mlp::Sparse(moe) = &layer.mlp {
                    moe.set_expert_cache_capacity(per_pool_budget);
                }
            }
            if let Some(block) = tower.mtp.as_ref() {
                block.mlp.set_expert_cache_capacity(per_pool_budget);
            }
            if lisa_mlx::trace::enabled_for("glm53-loader") {
                eprintln!(
                    "[glm53-loader] routed expert cache budget {:.2} GiB ({sparse_count} pools, {:.2} GiB each)",
                    expert_cache_budget as f64 / (1024.0 * 1024.0 * 1024.0),
                    per_pool_budget as f64 / (1024.0 * 1024.0 * 1024.0),
                );
            }
        }
        if enable_bounded_cache_residency && tower._expert_residency_set.is_none() {
            // A residency update reprocesses the set's allocations. Keep each
            // set local to its expert LRU so one layer's cache churn does not
            // revisit every other layer's resident weights.
            let mut residency_sets = 0usize;
            let mut allocations = 0usize;
            for layer in &mut tower.layers {
                if let Mlp::Sparse(moe) = &mut layer.mlp {
                    if let Some(set) = moe.new_cache_residency_set() {
                        allocations = allocations.saturating_add(set.allocation_count());
                        residency_sets = residency_sets.saturating_add(1);
                        moe.set_model_residency_set(set);
                    }
                }
            }
            if let Some(block) = tower.mtp.as_mut() {
                if let Some(set) = block.mlp.new_cache_residency_set() {
                    allocations = allocations.saturating_add(set.allocation_count());
                    residency_sets = residency_sets.saturating_add(1);
                    block.mlp.set_model_residency_set(set);
                }
            }
            if lisa_mlx::trace::enabled_for("glm53-loader") {
                eprintln!(
                    "[glm53-loader] bounded-cache residency: {residency_sets} per-pool sets, {allocations} allocation slots"
                );
            }
        }
        Ok(tower)
    }

    #[cfg(test)]
    pub(crate) fn load_without_expert_residency(
        dir: &Path,
        config: ModelConfig,
    ) -> anyhow::Result<Self> {
        Self::load_with_expert_residency(dir, config, false)
    }

    #[cfg(test)]
    pub(crate) fn load_mapped_experts_without_residency(
        dir: &Path,
        config: ModelConfig,
    ) -> anyhow::Result<Self> {
        // Diagnostic A/B: map every expert weight but do not pin all of them
        // in Metal residency or use the per-layer bounded expert cache.
        Self::load_with_expert_residency_mode(dir, config, true, false, false)
    }

    #[cfg(test)]
    pub(crate) fn load_with_deferred_expert_residency(
        dir: &Path,
        config: ModelConfig,
    ) -> anyhow::Result<Self> {
        Self::load_with_expert_residency_mode(dir, config, true, false, false)
    }

    pub fn from_weights(w: Weights, config: ModelConfig) -> anyhow::Result<Self> {
        Self::from_weights_with_lazy_experts(w, config, None, true, true)
    }

    fn from_weights_with_lazy_experts(
        mut w: Weights,
        config: ModelConfig,
        lazy_store: Option<std::sync::Arc<LazyTensorStore>>,
        enable_expert_residency: bool,
        request_residency: bool,
    ) -> anyhow::Result<Self> {
        let context_capacity = config
            .max_position_embeddings
            .min(DS4_GLM53_DEFAULT_CONTEXT);
        let h = config.hidden_size;
        let eps = config.rms_norm_eps;
        let take = |w: &mut Weights, name: &str| -> anyhow::Result<Array> {
            w.remove(name)
                .ok_or_else(|| anyhow::anyhow!("glm53: tensor {name} missing"))
        };

        let mut embed = QuantizedEmbedding::load(&mut w, "model.embed_tokens")?;
        embed.bits = 8;
        embed.group_size = 64;
        // ds4 consumes GLM weights into F32 graph buffers. Promote affine
        // metadata once so embedding dequantization does not round to BF16.
        embed.scales = embed.scales.as_dtype(Dtype::Float32)?;
        embed.biases = embed.biases.as_dtype(Dtype::Float32)?;
        let mut lm_head = QuantizedLinear::load(&mut w, "lm_head", "")?;
        lm_head.set_quant(64, 8);
        lm_head.scales = lm_head.scales.as_dtype(Dtype::Float32)?;
        lm_head.biases = lm_head.biases.as_dtype(Dtype::Float32)?;
        let final_norm = take(&mut w, "model.norm.weight")?.as_dtype(Dtype::Float32)?;

        let n_layers = config.num_hidden_layers; // 45; MTP (layer 45) stays in the map
        let mut layers = Vec::with_capacity(n_layers);
        for i in 0..n_layers {
            let p = format!("model.layers.{i}");
            let in_norm =
                take(&mut w, &format!("{p}.input_layernorm.weight"))?.as_dtype(Dtype::Float32)?;
            let post_norm = take(&mut w, &format!("{p}.post_attention_layernorm.weight"))?
                .as_dtype(Dtype::Float32)?;
            let hc_attn = Hc::load(
                &mut w,
                &format!("{p}.hc_attn"),
                config.hc_mult,
                h,
                config.hc_eps,
                config.hc_sinkhorn_iters,
                eps,
            )?;
            let hc_ffn = Hc::load(
                &mut w,
                &format!("{p}.hc_ffn"),
                config.hc_mult,
                h,
                config.hc_eps,
                config.hc_sinkhorn_iters,
                eps,
            )?;
            let attn = if config.is_kda(i) {
                Attn::Kda(Kda::load::<Weights>(
                    &mut w,
                    &format!("{p}.self_attn"),
                    config.linear_num_heads,
                    config.linear_head_dim,
                    config.linear_conv_kernel_dim,
                    config.linear_lower_bound,
                    eps,
                )?)
            } else {
                Attn::Dsa(Dsa::load::<Weights>(
                    &mut w,
                    &format!("{p}.self_attn"),
                    config.num_attention_heads,
                    config.qk_nope_head_dim,
                    config.v_head_dim,
                    config.index_n_heads,
                    config.index_head_dim,
                    config.index_topk,
                    config.index_kpool,
                    config.index_kpool_always_select_tail,
                    eps,
                )?)
            };
            let mlp = if config.is_sparse_mlp(i) {
                Mlp::Sparse(MoE::load::<Weights>(
                    &mut w,
                    &format!("{p}.mlp"),
                    config.n_routed_experts,
                    config.num_experts_per_tok,
                    config.n_group,
                    config.topk_group,
                    config.routed_scaling_factor,
                    config.norm_topk_prob,
                    config.n_shared_experts,
                    config.moe_intermediate_size,
                    h,
                    config.swiglu_limit,
                    lazy_store.clone(),
                    enable_expert_residency,
                )?)
            } else {
                Mlp::Dense(DenseMlp::load::<Weights>(
                    &mut w,
                    &format!("{p}.mlp"),
                    config.swiglu_limit,
                )?)
            };
            layers.push(Layer {
                attn,
                mlp,
                hc_attn,
                hc_ffn,
                in_norm,
                post_norm,
            });
        }
        let mut mtp = if config.num_nextn_predict_layers > 0 {
            Some(MtpBlock::load(
                &mut w,
                &config,
                lazy_store.clone(),
                enable_expert_residency,
            )?)
        } else {
            None
        };
        // DS4 reuses the grouped MoE gate/up/activation/down storage across
        // layers. Every consumer is encoded before the next layer overwrites
        // these full-write buffers on the same ordered Metal queue.
        let grouped_workspace = std::sync::Arc::new(super::moe::GroupedMoeWorkspace::default());
        for layer in &mut layers {
            if let Mlp::Sparse(moe) = &mut layer.mlp {
                moe.set_grouped_workspace(grouped_workspace.clone());
            }
        }
        if let Some(block) = mtp.as_mut() {
            block.mlp.set_grouped_workspace(grouped_workspace);
        }
        // DS4 registers the model's mapped buffers through a single
        // queue-wide residency set. Metal limits the number of residency sets
        // attached to one queue, so combine all routed experts across the
        // tower and MTP block instead of creating one set per MoE layer.
        let resident_buffers = lazy_store
            .as_ref()
            .map_or_else(Vec::new, |store| store.resident_buffers());
        if let Some(store) = lazy_store.as_ref().filter(|store| store.is_resident()) {
            // Metadata promotion and paired projection buffers are ordinary
            // private allocations in addition to the immutable mapped arena.
            let bytes = store
                .resident_nbytes()
                .checked_add(lisa_mlx::memory::private_allocated_memory()?)
                .ok_or_else(|| anyhow::anyhow!("GLM resident memory accounting overflow"))?;
            anyhow::ensure!(
                bytes <= lisa_mlx::memory::ram_budget(),
                "GLM resident weights and private allocations require {:.2} GiB, above the {:.2} GiB RAM budget",
                bytes as f64 / (1u64 << 30) as f64,
                lisa_mlx::memory::ram_budget() as f64 / (1u64 << 30) as f64,
            );
        }
        let residency_capacity = resident_buffers.len()
            + layers
                .iter()
                .map(|layer| match &layer.mlp {
                    Mlp::Sparse(moe) => moe.expert_residency_capacity(),
                    Mlp::Dense(_) => 0,
                })
                .sum::<usize>()
            + mtp
                .as_ref()
                .map_or(0, |block| block.mlp.expert_residency_capacity());
        let expert_residency_set = if !enable_expert_residency || residency_capacity == 0 {
            None
        } else {
            Array::new_residency_set(residency_capacity).and_then(|set| {
                if !resident_buffers.iter().all(|buffer| set.add(buffer)) {
                    return None;
                }
                let layers_added = layers.iter().all(|layer| match &layer.mlp {
                    Mlp::Sparse(moe) => moe.add_expert_resources_to_residency_set(&set),
                    Mlp::Dense(_) => true,
                });
                let mtp_added = mtp
                    .as_ref()
                    .is_none_or(|block| block.mlp.add_expert_resources_to_residency_set(&set));
                if !layers_added || !mtp_added {
                    return None;
                }
                set.commit();
                let added_to_queue = if request_residency {
                    Array::add_residency_set_to_queue(&set)
                } else {
                    Array::add_residency_set_to_queue_without_request(&set)
                };
                added_to_queue.then(|| std::sync::Arc::new(set))
            })
        };
        anyhow::ensure!(
            resident_buffers.is_empty() || expert_residency_set.is_some(),
            "GLM resident model requires a queue-wide Metal residency set"
        );
        if let Some(set) = expert_residency_set.as_ref() {
            for layer in &mut layers {
                if let Mlp::Sparse(moe) = &mut layer.mlp {
                    moe.set_model_residency_set(std::sync::Arc::clone(set));
                }
            }
            if let Some(block) = mtp.as_mut() {
                block
                    .mlp
                    .set_model_residency_set(std::sync::Arc::clone(set));
            }
        }
        if lisa_mlx::trace::enabled_for("glm53-loader") {
            eprintln!(
                "[glm53-loader] model expert queue residency: {} allocations",
                expert_residency_set
                    .as_deref()
                    .map_or(0, lisa_mlx::runtime::ResidencySet::allocation_count)
            );
        }
        Ok(Self {
            config,
            context_capacity,
            capture: Capture::from_env(),
            prefill_mode: false,
            prefill_final_chunk: false,
            embed,
            layers,
            final_norm,
            lm_head,
            mtp,
            _expert_residency_set: expert_residency_set,
            n_layers,
            _resident_memory: None,
        })
    }

    fn mtp_step(
        &mut self,
        tape: &mut IndexerTape,
        target_multi: &Array,
        next_token: u32,
        position: usize,
        min_position: usize,
    ) -> anyhow::Result<Array> {
        anyhow::ensure!(
            min_position <= position && position < self.context_capacity,
            "glm53 MTP position {position}/minimum {min_position} exceeds context capacity {}",
            self.context_capacity
        );
        let _scratch = lisa_mlx::runtime::enter_ordered_gpu_reuse_scope();
        let mtp = self
            .mtp
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("glm53 checkpoint has no MTP block"))?;
        mtp.forward(
            &self.config,
            &self.embed,
            &self.lm_head,
            tape,
            target_multi,
            next_token,
            position,
            min_position,
        )
    }

    /// Decode one token for rows whose DSA tapes have different absolute
    /// lengths. mHC, KDA/DSA projections, output projections, and MoE work run
    /// across the cohort; stateful KDA recurrence and DSA cache/attention
    /// updates still advance each row against its own cache, matching ds4's
    /// native session-batch ownership model.
    fn forward_ragged_inner(
        &mut self,
        ids: &Array,
        per_row_caches: &mut [Vec<LayerCache>],
    ) -> anyhow::Result<(Array, Array)> {
        let (b, s) = (ids.dim(0) as usize, ids.dim(1) as usize);
        anyhow::ensure!(
            b == per_row_caches.len() && s == 1,
            "GLM ragged decode expects one token and cache per row, got ids {:?}, caches {}",
            ids.shape(),
            per_row_caches.len()
        );
        anyhow::ensure!(b > 1, "GLM ragged decode requires at least two rows");
        anyhow::ensure!(
            per_row_caches
                .iter()
                .all(|row| row.len() == self.layers.len()),
            "GLM ragged decode cache depth does not match the tower"
        );
        // Validate the complete cohort before updating any row's cache.
        for caches in per_row_caches.iter() {
            self.check_context_span(caches, 1)?;
        }
        // Preserve-dispatch captures retain their Arrays in Capture::pending.
        // Their buffer leases remain live, so ordered reuse can recycle only
        // unrelated dead scratch while those snapshots wait for readback.
        let _scratch = self
            .capture
            .as_ref()
            .is_none_or(|c| c.layers.is_empty() || c.preserves_dispatch())
            .then(lisa_mlx::runtime::enter_ordered_gpu_reuse_scope);
        let dense_compact_limit = self
            .config
            .index_topk
            .saturating_add(self.config.index_kpool.saturating_sub(1));
        let max_position = per_row_caches
            .iter()
            .flat_map(|row| row.iter())
            .filter_map(|cache| match cache {
                LayerCache::Full(cache) => Some(cache.offset),
                LayerCache::Linear(_) => None,
            })
            .max()
            .unwrap_or(0);
        if !ds4_native_session_batch_position_supported(b, max_position, dense_compact_limit) {
            // Match glm53_graph_native_session_batch_supported: ds4 declines
            // its native cohort kernel at this row count or sparse-context
            // boundary and evaluates each session through the serial graph.
            let mut mixed_rows = Vec::with_capacity(b);
            let mut multi_rows = Vec::with_capacity(b);
            for (row, caches) in per_row_caches.iter_mut().enumerate() {
                let token = narrow(ids, 0, row, 1)?;
                let (mixed, multi) = self.forward_inner(&token, caches)?;
                mixed_rows.push(mixed);
                multi_rows.push(multi);
            }
            let mixed_refs: Vec<&Array> = mixed_rows.iter().collect();
            let multi_refs: Vec<&Array> = multi_rows.iter().collect();
            return Ok((
                Array::concatenate(&mixed_refs, 0)?,
                Array::concatenate(&multi_refs, 0)?,
            ));
        }
        let h = self.config.hidden_size;
        let c = self.config.hc_mult;
        let x = self.embed.forward(ids)?.as_dtype(Dtype::Float32)?;
        let mut state = x
            .expand_dims(2)?
            .broadcast_to(&[b as i32, 1, c as i32, h as i32])?
            .contiguous()?;
        #[cfg(test)]
        let capture_mhc_expansions = std::env::var_os("LISA_GLM53_MHC_BATCH_CAPTURE").is_some();
        #[cfg(not(test))]
        let capture_mhc_expansions = false;

        for (li, layer) in self.layers.iter_mut().enumerate() {
            let (_xc, x_in, post, comb, attn_mixes, attn_split) = if capture_mhc_expansions {
                if let Some((xc, normed, post, comb, mixes, split)) = layer
                    .hc_attn
                    .split_normed_with_mixes(&state, &layer.in_norm, self.config.rms_norm_eps)?
                {
                    (xc, normed, post, comb, Some(mixes), Some(split))
                } else {
                    let (xc, post, comb) = layer.hc_attn.split(&state)?;
                    let normed = rms(&xc, &layer.in_norm, self.config.rms_norm_eps)?;
                    (xc, normed, post, comb, None, None)
                }
            } else if let Some((xc, normed, post, comb)) =
                layer
                    .hc_attn
                    .split_normed(&state, &layer.in_norm, self.config.rms_norm_eps)?
            {
                (xc, normed, post, comb, None, None)
            } else {
                let (xc, post, comb) = layer.hc_attn.split(&state)?;
                let normed = rms(&xc, &layer.in_norm, self.config.rms_norm_eps)?;
                (xc, normed, post, comb, None, None)
            };
            let capture = self.capture.as_ref().filter(|capture| capture.wants(li));
            if capture_mhc_expansions && let Some(capture) = capture {
                capture.store(&format!("L{li}.mhc_attn_in"), &state);
                capture.store(&format!("L{li}.mhc_attn_post"), &post);
                capture.store(&format!("L{li}.mhc_attn_comb"), &comb);
                capture.store(&format!("L{li}.mhc_attn_collapsed"), &_xc);
                capture.store(&format!("L{li}.mhc_attn_normed"), &x_in);
                if let Some(mixes) = &attn_mixes {
                    capture.store(&format!("L{li}.mhc_attn_mixes"), mixes);
                }
                if let Some(split) = &attn_split {
                    capture.store(&format!("L{li}.mhc_attn_split"), split);
                }
            }
            let step = INC_STEP.load(std::sync::atomic::Ordering::Relaxed);
            if let Some(capture) = capture {
                capture.store(&format!("L{li}.dec_rms_in_s{step}"), &x_in);
            }
            if li == 0 {
                INC_STEP.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
            let attended = match &layer.attn {
                Attn::Kda(kda) => {
                    let mut row_caches = Vec::with_capacity(b);
                    for (row, caches) in per_row_caches.iter_mut().enumerate() {
                        match &mut caches[li] {
                            LayerCache::Linear(cache) => row_caches.push(cache),
                            _ => anyhow::bail!(
                                "glm53: ragged KDA cache kind mismatch at layer {li}, row {row}"
                            ),
                        }
                    }
                    kda.forward_ragged(&x_in, &mut row_caches, self.capture.as_ref(), li)?
                }
                Attn::Dsa(dsa) => {
                    let mut row_caches = Vec::with_capacity(b);
                    for (row, caches) in per_row_caches.iter_mut().enumerate() {
                        match &mut caches[li] {
                            LayerCache::Full(cache) => row_caches.push(cache),
                            _ => anyhow::bail!(
                                "glm53: ragged DSA cache kind mismatch at layer {li}, row {row}"
                            ),
                        }
                    }
                    dsa.forward_ragged(&x_in, &mut row_caches, self.capture.as_ref(), li)?
                }
            };
            if let Some(capture) = capture {
                capture.store(&format!("L{li}.dec_attn_out_s{step}"), &attended);
            }
            state = layer.hc_attn.expand(&attended, &state, &post, &comb)?;
            if let Some(capture) = capture {
                capture.store(&format!("L{li}.state_after_attn"), &state);
                capture.store(&format!("L{li}.dec_state_after_attn_s{step}"), &state);
            }

            let (_xc2, x_in2, post2, comb2, ffn_mixes, ffn_split) = if capture_mhc_expansions {
                if let Some((xc, normed, post, comb, mixes, split)) = layer
                    .hc_ffn
                    .split_normed_with_mixes(&state, &layer.post_norm, self.config.rms_norm_eps)?
                {
                    (xc, normed, post, comb, Some(mixes), Some(split))
                } else {
                    let (xc, post, comb) = layer.hc_ffn.split(&state)?;
                    let normed = rms(&xc, &layer.post_norm, self.config.rms_norm_eps)?;
                    (xc, normed, post, comb, None, None)
                }
            } else if let Some((xc, normed, post, comb)) =
                layer
                    .hc_ffn
                    .split_normed(&state, &layer.post_norm, self.config.rms_norm_eps)?
            {
                (xc, normed, post, comb, None, None)
            } else {
                let (xc, post, comb) = layer.hc_ffn.split(&state)?;
                let normed = rms(&xc, &layer.post_norm, self.config.rms_norm_eps)?;
                (xc, normed, post, comb, None, None)
            };
            if capture_mhc_expansions && let Some(capture) = capture {
                capture.store(&format!("L{li}.mhc_ffn_post"), &post2);
                capture.store(&format!("L{li}.mhc_ffn_comb"), &comb2);
                capture.store(&format!("L{li}.mhc_ffn_collapsed"), &_xc2);
                capture.store(&format!("L{li}.mhc_ffn_normed"), &x_in2);
                if let Some(mixes) = &ffn_mixes {
                    capture.store(&format!("L{li}.mhc_ffn_mixes"), mixes);
                }
                if let Some(split) = &ffn_split {
                    capture.store(&format!("L{li}.mhc_ffn_split"), split);
                }
            }
            if let Some(capture) = capture {
                capture.store(&format!("L{li}.mhc_ffn_normed"), &x_in2);
            }
            let ffn = match &layer.mlp {
                Mlp::Dense(dense) => {
                    dense.forward_with_capture(&x_in2, self.capture.as_ref(), li)?
                }
                Mlp::Sparse(moe) => moe.forward_with_capture(&x_in2, self.capture.as_ref(), li)?,
            };
            if let Some(capture) = capture {
                capture.store(&format!("L{li}.dec_moe_out_s{step}"), &ffn);
            }
            state = layer.hc_ffn.expand(&ffn, &state, &post2, &comb2)?;
            if let Some(capture) = capture {
                capture.store(&format!("L{li}.state_after_ffn"), &state);
                capture.store(&format!("L{li}.dec_state_after_ffn_s{step}"), &state);
            }
        }

        let mixed = final_hc_mean(&state)?.reshape(&[b as i32, 1, h as i32])?;
        let mixed = final_rms(&mixed, &self.final_norm, self.config.rms_norm_eps)?;
        let multi = state.reshape(&[b as i32, 1, (c * h) as i32])?;
        Ok((mixed, multi))
    }

    /// One forward: `ids [B,S] i32` -> `(mixed [B,S,H], stream [B,S,C*H])`.
    fn forward_inner(
        &mut self,
        ids: &Array,
        caches: &mut Vec<LayerCache>,
    ) -> anyhow::Result<(Array, Array)> {
        self.forward_inner_until(ids, caches, None)
    }

    fn forward_prefill_chunk(
        &mut self,
        ids: &Array,
        caches: &mut Vec<LayerCache>,
        final_chunk: bool,
    ) -> anyhow::Result<(Array, Array)> {
        let previous = std::mem::replace(&mut self.prefill_mode, true);
        let previous_final = std::mem::replace(&mut self.prefill_final_chunk, final_chunk);
        let result = self.forward_inner(ids, caches);
        self.prefill_mode = previous;
        self.prefill_final_chunk = previous_final;
        result
    }

    fn forward_inner_until(
        &mut self,
        ids: &Array,
        caches: &mut Vec<LayerCache>,
        stop_after_layer: Option<usize>,
    ) -> anyhow::Result<(Array, Array)> {
        let _trace_forward = lisa_mlx::trace::span("glm53.forward");
        let (b, s) = (ids.dim(0) as usize, ids.dim(1) as usize);
        self.check_context_span(caches, s)?;
        let _scratch = self
            .capture
            .as_ref()
            .is_none_or(|c| c.layers.is_empty() || c.preserves_dispatch())
            .then(lisa_mlx::runtime::enter_ordered_gpu_reuse_scope);
        let forward_position = caches.iter().find_map(|cache| match cache {
            LayerCache::Full(cache) => Some(cache.offset),
            LayerCache::Linear(_) => None,
        });
        #[cfg(test)]
        let mtp_layer_capture = if b == 1 && s == 2 {
            static NEXT_CAPTURE: std::sync::atomic::AtomicUsize =
                std::sync::atomic::AtomicUsize::new(0);
            static REQUESTED_CALLS: std::sync::OnceLock<Option<Vec<usize>>> =
                std::sync::OnceLock::new();
            std::env::var_os("LISA_GLM53_MTP_LAYER_CAPTURE_PREFIX").and_then(|prefix| {
                let capture_index = NEXT_CAPTURE.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                let requested_calls = REQUESTED_CALLS.get_or_init(|| {
                    std::env::var("LISA_GLM53_MTP_LAYER_CAPTURE_CALLS")
                        .ok()
                        .map(|value| {
                            value
                                .split(',')
                                .filter_map(|item| item.trim().parse::<usize>().ok())
                                .collect()
                        })
                });
                let selected = requested_calls
                    .as_ref()
                    .map_or(capture_index < 2, |calls| calls.contains(&capture_index));
                selected.then_some((prefix, capture_index))
            })
        } else {
            None
        };
        #[cfg(test)]
        let single_step_layer_capture = if b == 1 && s == 1 {
            static REQUESTED_POSITIONS: std::sync::OnceLock<Option<Vec<usize>>> =
                std::sync::OnceLock::new();
            let position = caches.iter().find_map(|cache| match cache {
                LayerCache::Full(cache) => Some(cache.offset),
                LayerCache::Linear(_) => None,
            });
            std::env::var_os("LISA_GLM53_MTP_LAYER_CAPTURE_PREFIX").and_then(|prefix| {
                let requested_positions = REQUESTED_POSITIONS.get_or_init(|| {
                    std::env::var("LISA_GLM53_MTP_LAYER_CAPTURE_POSITIONS")
                        .ok()
                        .map(|value| {
                            value
                                .split(',')
                                .filter_map(|item| item.trim().parse::<usize>().ok())
                                .collect()
                        })
                });
                let selected = position.is_some_and(|position| {
                    requested_positions
                        .as_ref()
                        .is_some_and(|positions| positions.contains(&position))
                });
                selected.then_some((prefix, position.expect("selected cache position")))
            })
        } else {
            None
        };
        let prefill_mode = self.prefill_mode;
        let h = self.config.hidden_size;
        let c = self.config.hc_mult;
        if let Some(layer) = stop_after_layer {
            anyhow::ensure!(
                layer < self.layers.len(),
                "GLM prefix layer {layer} exceeds {} layers",
                self.layers.len()
            );
        }
        // DEBUG-ONLY probes: every readback below forces a full GPU→CPU sync
        // per layer — 90 syncs per decode token when left on.
        let probe = std::env::var_os("LISA_GLM53_DEBUG").is_some();
        let flush_layers = std::env::var_os("LISA_GLM53_FLUSH_LAYERS").is_some();
        let sync_layers = std::env::var("LISA_GLM53_SYNC_LAYERS").ok().map(|spec| {
            if spec == "1" || spec.eq_ignore_ascii_case("all") {
                None
            } else {
                Some(
                    spec.split(',')
                        .filter_map(|item| item.trim().parse::<usize>().ok())
                        .collect::<std::collections::HashSet<_>>(),
                )
            }
        });

        let x = {
            let _trace_embed = lisa_mlx::trace::span("glm53.embed");
            self.embed.forward(ids)?.as_dtype(Dtype::Float32)? // [B,S,H]
        };
        if let Some(c) = &self.capture {
            c.store("embed_out", &x);
        }
        let mut state = x
            .expand_dims(2)?
            .broadcast_to(&[(b) as i32, (s) as i32, (c) as i32, (h) as i32])?
            .contiguous()?; // [B,S,C,H]

        let layer_count = self.layers.len();
        for (li, layer) in self.layers.iter_mut().enumerate() {
            let _trace_layer = lisa_mlx::trace::span_detail("glm53.layer", li as u64 + 1);
            if lisa_mlx::trace::enabled_for("glm53.memory") {
                eprintln!(
                    "[glm53-memory] layer={li} rows={} private_bytes={}",
                    b * s,
                    lisa_mlx::memory::private_allocated_memory()?,
                );
            }
            // Attention half.
            if let Some(cp) = &self.capture {
                if cp.wants(li) {
                    cp.store(&format!("L{li}.mhc_attn_in"), &state);
                }
            }
            let capture_attn_mixes = self
                .capture
                .as_ref()
                .is_some_and(|capture| capture.wants(li) && !capture.preserves_dispatch());
            let (xc, x_in, post, comb, attn_mixes, attn_split) = {
                let _trace_mhc = lisa_mlx::trace::span("glm53.mhc.attn_input");
                if capture_attn_mixes {
                    if let Some((xc, normed, post, comb, mixes, split)) = layer
                        .hc_attn
                        .split_normed_with_mixes(&state, &layer.in_norm, self.config.rms_norm_eps)?
                    {
                        (xc, normed, post, comb, Some(mixes), Some(split))
                    } else {
                        let (xc, post, comb) = layer.hc_attn.split(&state)?;
                        let normed = rms(&xc, &layer.in_norm, self.config.rms_norm_eps)?;
                        (xc, normed, post, comb, None, None)
                    }
                } else if let Some((xc, normed, post, comb)) =
                    layer
                        .hc_attn
                        .split_normed(&state, &layer.in_norm, self.config.rms_norm_eps)?
                {
                    (xc, normed, post, comb, None, None)
                } else {
                    let (xc, post, comb) = layer.hc_attn.split(&state)?;
                    let normed = rms(&xc, &layer.in_norm, self.config.rms_norm_eps)?;
                    (xc, normed, post, comb, None, None)
                }
            };
            #[cfg(test)]
            if li == 1 {
                if let Some((prefix, position)) = single_step_layer_capture.as_ref() {
                    capture_single_step_layer_state(
                        prefix,
                        *position,
                        li,
                        Some("attn-norm"),
                        &x_in,
                    )?;
                }
            }
            if let Some(cp) = &self.capture {
                if cp.wants(li) {
                    cp.store(&format!("L{li}.mhc_attn_collapsed"), &xc);
                    cp.store(&format!("L{li}.mhc_attn_post"), &post);
                    cp.store(&format!("L{li}.mhc_attn_comb"), &comb);
                    cp.store(&format!("L{li}.mhc_attn_normed"), &x_in);
                    if let Some(mixes) = &attn_mixes {
                        cp.store(&format!("L{li}.mhc_attn_mixes"), mixes);
                    }
                    if let Some(split) = &attn_split {
                        cp.store(&format!("L{li}.mhc_attn_split"), split);
                    }
                }
            }
            // incremental-decode probe: norm of every stage, every layer
            let step = if li == 0 {
                INC_STEP.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
            } else {
                INC_STEP.load(std::sync::atomic::Ordering::Relaxed)
            };
            let capture_mode = if x_in.dim(1) == 1 { "dec" } else { "rep" };
            if probe && x_in.dim(1) == 1 {
                let n = |a: &Array| -> f32 {
                    let v = a
                        .as_dtype(Dtype::Float32)
                        .unwrap()
                        .to_vec1::<f32>()
                        .unwrap_or_default();
                    v.iter().map(|x| x * x).sum::<f32>().sqrt()
                };
                let nan = |a: &Array| -> bool {
                    a.as_dtype(Dtype::Float32)
                        .unwrap()
                        .to_vec1::<f32>()
                        .unwrap_or_default()
                        .iter()
                        .any(|x| x.is_nan())
                };
                eprintln!(
                    "[inc-probe] s{step} T{} L{li:>2} rms_in {:9.4} nan {}",
                    x_in.dim(1),
                    n(&x_in),
                    nan(&x_in)
                );
            }
            if let Some(cp) = &self.capture {
                if cp.wants(li) {
                    let t2 = if x_in.dim(1) == 1 { "dec" } else { "rep" };
                    cp.store(&format!("L{li}.{t2}_rms_in_s{step}"), &x_in);
                }
            }
            let attended = {
                let _trace_attn = lisa_mlx::trace::span("glm53.attention");
                match (&layer.attn, &mut caches[li]) {
                    (Attn::Kda(k), LayerCache::Linear(g)) => {
                        if prefill_mode {
                            k.forward_prefill(
                                &x_in,
                                &mut g.conv,
                                &mut g.ssm,
                                self.capture.as_ref(),
                                li,
                            )?
                        } else if let Some(position) = forward_position {
                            k.forward_at_position(
                                &x_in,
                                &mut g.conv,
                                &mut g.ssm,
                                self.capture.as_ref(),
                                li,
                                position,
                            )?
                        } else {
                            k.forward(&x_in, &mut g.conv, &mut g.ssm, self.capture.as_ref(), li)?
                        }
                    }
                    (Attn::Dsa(d), LayerCache::Full(f)) => {
                        d.forward(&x_in, f, self.capture.as_ref(), li, prefill_mode)?
                    }
                    _ => anyhow::bail!("glm53: cache kind mismatch at layer {li}"),
                }
            };
            if probe && x_in.dim(1) == 1 {
                let n = |a: &Array| -> f32 {
                    let v = a
                        .as_dtype(Dtype::Float32)
                        .unwrap()
                        .to_vec1::<f32>()
                        .unwrap_or_default();
                    v.iter().map(|x| x * x).sum::<f32>().sqrt()
                };
                let nan = |a: &Array| -> bool {
                    a.as_dtype(Dtype::Float32)
                        .unwrap()
                        .to_vec1::<f32>()
                        .unwrap_or_default()
                        .iter()
                        .any(|x| x.is_nan())
                };
                eprintln!(
                    "[inc-probe] s{step} L{li:>2} attn_out {:9.4} nan {}",
                    n(&attended),
                    nan(&attended)
                );
            }
            if let Some(cp) = &self.capture {
                if cp.wants(li) {
                    let t2 = if x_in.dim(1) == 1 { "dec" } else { "rep" };
                    cp.store(&format!("L{li}.{t2}_attn_out_s{step}"), &attended);
                    cp.store(&format!("L{li}.mhc_attn_xc"), &xc);
                }
            }
            state = {
                let _trace_expand = lisa_mlx::trace::span("glm53.mhc.attn_expand");
                layer.hc_attn.expand(&attended, &state, &post, &comb)?
            };
            if let Some(cp) = &self.capture {
                if cp.wants(li) {
                    cp.store(&format!("L{li}.state_after_attn"), &state);
                    cp.store(
                        &format!("L{li}.{capture_mode}_state_after_attn_s{step}"),
                        &state,
                    );
                }
            }

            // FFN half.
            let capture_ffn_mixes = self
                .capture
                .as_ref()
                .is_some_and(|capture| capture.wants(li) && !capture.preserves_dispatch());
            let (xc2, x_in2, post2, comb2, ffn_mixes, ffn_split) = {
                let _trace_mhc = lisa_mlx::trace::span("glm53.mhc.ffn_input");
                if capture_ffn_mixes {
                    if let Some((xc, normed, post, comb, mixes, split)) =
                        layer.hc_ffn.split_normed_with_mixes(
                            &state,
                            &layer.post_norm,
                            self.config.rms_norm_eps,
                        )?
                    {
                        (xc, normed, post, comb, Some(mixes), Some(split))
                    } else {
                        let (xc, post, comb) = layer.hc_ffn.split(&state)?;
                        let normed = rms(&xc, &layer.post_norm, self.config.rms_norm_eps)?;
                        (xc, normed, post, comb, None, None)
                    }
                } else if let Some((xc, normed, post, comb)) =
                    layer
                        .hc_ffn
                        .split_normed(&state, &layer.post_norm, self.config.rms_norm_eps)?
                {
                    (xc, normed, post, comb, None, None)
                } else {
                    let (xc, post, comb) = layer.hc_ffn.split(&state)?;
                    let normed = rms(&xc, &layer.post_norm, self.config.rms_norm_eps)?;
                    (xc, normed, post, comb, None, None)
                }
            };
            if let Some(cp) = &self.capture {
                if cp.wants(li) {
                    cp.store(&format!("L{li}.mhc_ffn_post"), &post2);
                    cp.store(&format!("L{li}.mhc_ffn_comb"), &comb2);
                    cp.store(&format!("L{li}.mhc_ffn_normed"), &x_in2);
                    if let Some(mixes) = &ffn_mixes {
                        cp.store(&format!("L{li}.mhc_ffn_mixes"), mixes);
                    }
                    if let Some(split) = &ffn_split {
                        cp.store(&format!("L{li}.mhc_ffn_split"), split);
                    }
                }
            }
            let m = {
                let _trace_ffn = lisa_mlx::trace::span("glm53.mlp");
                match &layer.mlp {
                    Mlp::Dense(d) => d.forward_with_capture(&x_in2, self.capture.as_ref(), li)?,
                    Mlp::Sparse(moe) => {
                        moe.forward_with_capture(&x_in2, self.capture.as_ref(), li)?
                    }
                }
            };
            if let Some(cp) = &self.capture {
                if cp.wants(li) {
                    let t2 = if x_in.dim(1) == 1 { "dec" } else { "rep" };
                    cp.store(&format!("L{li}.{t2}_moe_out_s{step}"), &m);
                    cp.store(&format!("L{li}.mhc_ffn_xc2"), &xc2);
                }
            }
            state = {
                let _trace_expand = lisa_mlx::trace::span("glm53.mhc.ffn_expand");
                layer.hc_ffn.expand(&m, &state, &post2, &comb2)?
            };
            #[cfg(test)]
            if let Some((prefix, capture_index)) = mtp_layer_capture.as_ref() {
                capture_mtp_verify_layer_state(prefix, *capture_index, li, &state)?;
            }
            #[cfg(test)]
            if let Some((prefix, position)) = single_step_layer_capture.as_ref() {
                capture_single_step_layer_state(prefix, *position, li, None, &state)?;
            }
            if let Some(cp) = &self.capture {
                if cp.wants(li) {
                    cp.store(&format!("L{li}.state_after_ffn"), &state);
                    cp.store(
                        &format!("L{li}.{capture_mode}_state_after_ffn_s{step}"),
                        &state,
                    );
                }
            }
            if sync_layers
                .as_ref()
                .is_some_and(|selected| selected.as_ref().is_none_or(|layers| layers.contains(&li)))
            {
                state.eval()?;
            }
            if flush_layers {
                lisa_mlx::Stream::thread_local_or_default()
                    .runtime()
                    .commands
                    .flush()?;
            }
            // DS4 submits indexed GLM decode work every four layers without
            // waiting for completion. This lets Metal execute early layers
            // while the host encodes the remainder of the token. A single
            // command buffer for the whole 45-layer forward adds a large
            // submission bubble even though both paths are otherwise async.
            if b == 1
                && s == 1
                && !prefill_mode
                && self
                    .capture
                    .as_ref()
                    .is_none_or(|capture| capture.preserves_dispatch())
                && !probe
                && (li + 1) % 4 == 0
                && li + 1 < layer_count
            {
                lisa_mlx::Stream::thread_local_or_default()
                    .runtime()
                    .commands
                    .flush()?;
            }
            // DS4's final prefill chunk submits every layer above eight
            // rows and drains at each sixteen-layer boundary. Intermediate
            // chunks retain one command stream until the chunk-end drain.
            if prefill_mode
                && (s > DS4_GLM53_PREFILL_CHUNK_TOKENS || (self.prefill_final_chunk && s > 8))
                && li + 1 < layer_count
                && self.capture.as_ref().is_none_or(|c| c.preserves_dispatch())
                && !probe
            {
                let stream = lisa_mlx::Stream::thread_local_or_default();
                if (li + 1) % 16 == 0 {
                    let _span = lisa_mlx::trace::span("glm53.prefill.layer_drain");
                    stream.runtime().commands.flush_and_wait()?;
                } else {
                    stream.runtime().commands.flush()?;
                }
            }
            if stop_after_layer == Some(li) {
                let mixed = state
                    .contiguous()?
                    .mean_axis(2, true)?
                    .reshape(&[b as i32, s as i32, h as i32])?;
                let multi = state.reshape(&[b as i32, s as i32, (c * h) as i32])?;
                return Ok((mixed, multi));
            }
        }

        if prefill_mode {
            let _span = lisa_mlx::trace::span("glm53.prefill.chunk_drain");
            lisa_mlx::Stream::thread_local_or_default()
                .runtime()
                .commands
                .flush_and_wait()?;
        }
        // Final: mean over the streams, RMS, head.
        if let Some(c) = &self.capture {
            if std::env::var_os("LISA_GLM53_CAPTURE").is_some() {
                c.store("final_state", &state);
                let mode = if s == 1 { "dec" } else { "rep" };
                let step = INC_STEP.load(std::sync::atomic::Ordering::Relaxed);
                c.store(&format!("final_{mode}_state_s{step}"), &state);
            }
        }
        let mixed = final_hc_mean(&state)?;
        let mixed = final_rms(&mixed, &self.final_norm, self.config.rms_norm_eps)?;
        if let Some(c) = &self.capture {
            let mode = if s == 1 { "dec" } else { "rep" };
            let step = INC_STEP.load(std::sync::atomic::Ordering::Relaxed);
            c.store(&format!("final_{mode}_mixed_s{step}"), &mixed);
            if !c.preserves_dispatch()
                && let Ok(p) = std::env::var("LISA_GLM53_CAPTURE")
            {
                // Re-flush on every forward so step-tagged decode captures
                // include the final mixed representation used by the head.
                let _ = c.flush(&p);
            }
        }
        let multi = state.reshape(&[(b) as i32, (s) as i32, (c * h) as i32])?;
        Ok((mixed, multi))
    }

    fn prefill_chunks(
        &mut self,
        tokens: &[u32],
        caches: &mut Vec<LayerCache>,
        retain_multi: bool,
    ) -> anyhow::Result<(Array, Option<Array>)> {
        anyhow::ensure!(!tokens.is_empty(), "empty GLM prefill");
        self.check_context_span(caches, tokens.len())?;
        static PREFILL_CAPTURE_ID: std::sync::atomic::AtomicUsize =
            std::sync::atomic::AtomicUsize::new(0);
        let diagnostic_capture =
            std::env::var_os("LISA_GLM53_RAGGED_PREFILL_CAPTURE_PREFIX").is_some()
                && self.capture.as_ref().is_some_and(|capture| {
                    capture.kda_prefill_only && capture.preserves_dispatch()
                });
        let diagnostic_capture_id = diagnostic_capture
            .then(|| PREFILL_CAPTURE_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed));
        let flush_diagnostic_chunk = |capture: &Capture,
                                      chunk_index: usize|
         -> anyhow::Result<()> {
            let Some(prefix) = std::env::var_os("LISA_GLM53_RAGGED_PREFILL_CAPTURE_PREFIX") else {
                return Ok(());
            };
            let capture_id = diagnostic_capture_id
                .ok_or_else(|| anyhow::anyhow!("KDA prefill capture id is missing"))?;
            let mut path = prefix;
            path.push(format!(".prefill-{capture_id}.chunk-{chunk_index}.npz"));
            capture.flush(
                Path::new(&path)
                    .to_str()
                    .ok_or_else(|| anyhow::anyhow!("ragged KDA chunk capture path is not UTF-8"))?,
            )
        };

        // Full diagnostic captures intentionally retain the single-forward
        // layout. Normal sessions follow DS4's graph-selected 2048-token
        // boundary and carry the recurrent and indexer caches between calls.
        if self
            .capture
            .as_ref()
            .is_some_and(|capture| !capture.preserves_dispatch())
            || tokens.len() <= self.prefill_chunk_rows(caches, tokens.len())
        {
            let ids = i32_arr(&self.embed.weight, tokens)?;
            let (mixed, multi) = self.forward_prefill_chunk(&ids, caches, true)?;
            if diagnostic_capture {
                flush_diagnostic_chunk(
                    self.capture
                        .as_ref()
                        .expect("diagnostic capture is installed"),
                    0,
                )?;
            }
            let last = narrow(&mixed, 1, mixed.dim(1) as usize - 1, 1)?
                .reshape(&[1, self.config.hidden_size as i32])?;
            return Ok((last, retain_multi.then_some(multi)));
        }

        let rollback: Vec<_> = caches.iter().map(LayerCache::snapshot_rollback).collect();
        let result = (|| -> anyhow::Result<(Array, Option<Array>)> {
            let mut last = None;
            let mut multi_chunks = Vec::new();
            let mut done = 0;
            let mut chunk_index = 0;
            while done < tokens.len() {
                let rows = self.prefill_chunk_rows(caches, tokens.len() - done);
                let chunk = &tokens[done..done + rows];
                let final_chunk = done + rows == tokens.len();
                let ids = i32_arr(&self.embed.weight, chunk)?;
                let (mixed, multi) = self.forward_prefill_chunk(&ids, caches, final_chunk)?;
                last = Some(
                    narrow(&mixed, 1, mixed.dim(1) as usize - 1, 1)?
                        .reshape(&[1, self.config.hidden_size as i32])?,
                );
                if retain_multi {
                    multi_chunks.push(multi);
                }
                if diagnostic_capture {
                    flush_diagnostic_chunk(
                        self.capture
                            .as_ref()
                            .expect("diagnostic capture is installed"),
                        chunk_index,
                    )?;
                    if !final_chunk {
                        let previous = self
                            .capture
                            .take()
                            .expect("diagnostic capture is installed");
                        self.capture = Some(Capture {
                            layers: previous.layers.clone(),
                            state_only: previous.state_only,
                            kda_prefill_only: previous.kda_prefill_only,
                            preserve_dispatch: previous.preserves_dispatch(),
                            ..Default::default()
                        });
                    }
                }
                lisa_mlx::memory::trim_cache();
                done += rows;
                chunk_index += 1;
            }
            let last = last.expect("non-empty chunks");
            let multi = if retain_multi {
                let refs: Vec<&Array> = multi_chunks.iter().collect();
                Some(Array::concatenate(&refs, 1)?)
            } else {
                None
            };
            Ok((last, multi))
        })();
        match result {
            Ok(value) => Ok(value),
            Err(error) => {
                for (cache, state) in caches.iter_mut().zip(&rollback) {
                    cache
                        .restore_rollback(state)
                        .map_err(|rollback_error| anyhow::anyhow!("{rollback_error}"))?;
                }
                Err(error)
            }
        }
    }

    fn prefill_chunk_rows(&self, caches: &[LayerCache], remaining: usize) -> usize {
        let position = caches
            .iter()
            .find_map(|cache| match cache {
                LayerCache::Full(cache) => Some(cache.offset),
                LayerCache::Linear(_) => None,
            })
            .unwrap_or(0);
        let work_capacity = self.context_capacity.min(4096);
        let rows = remaining.min(DS4_GLM53_PREFILL_CHUNK_TOKENS);
        if position < work_capacity {
            rows.min(work_capacity - position)
        } else {
            rows
        }
    }

    fn check_context_span(&self, caches: &[LayerCache], rows: usize) -> anyhow::Result<()> {
        for cache in caches {
            if let LayerCache::Full(cache) = cache {
                anyhow::ensure!(
                    cache
                        .offset
                        .checked_add(rows)
                        .is_some_and(|end| end <= self.context_capacity),
                    "GLM context overflow: append [{}, {}) exceeds capacity {}",
                    cache.offset,
                    cache.offset.saturating_add(rows),
                    self.context_capacity,
                );
            }
        }
        Ok(())
    }
}

fn i32_arr(_rt_of: &Array, tokens: &[u32]) -> anyhow::Result<Array> {
    let v: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
    Ok(Array::from_slice(&v, &[1, tokens.len() as i32]))
}

fn loaded_pairing_law_bits(modes: impl IntoIterator<Item = bool>) -> u64 {
    modes.into_iter().fold(0, |mask, paired| {
        mask | if paired { 1 << 4 } else { 1 << 5 }
    })
}

#[cfg(test)]
mod loaded_pairing_law_tests {
    #[test]
    fn loaded_projection_modes_have_distinct_cache_identities() {
        use super::loaded_pairing_law_bits;
        assert_eq!(loaded_pairing_law_bits([true, true]), 1 << 4);
        assert_eq!(loaded_pairing_law_bits([false, false]), 1 << 5);
        assert_eq!(loaded_pairing_law_bits([true, false]), (1 << 4) | (1 << 5));
    }
}

impl crate::models::LanguageModel for Tower {
    fn cache_numerical_law(&self) -> Option<u64> {
        // Read the projection modes retained by the loaded model. The separate
        // diagnostic schedule must not share caches with native paired fusion.
        let modes = self.layers.iter().filter_map(|layer| match &layer.mlp {
            Mlp::Dense(dense) => Some(dense.paired_storage()),
            Mlp::Sparse(moe) => moe.shared_paired_storage(),
        });
        let mtp_mode = self
            .mtp
            .as_ref()
            .and_then(|mtp| mtp.mlp.shared_paired_storage());
        Some(super::cache_numerical_law() | loaded_pairing_law_bits(modes.chain(mtp_mode)))
    }

    fn max_position_embeddings(&self) -> usize {
        self.context_capacity
    }

    fn set_context_capacity(&mut self, capacity: usize) -> anyhow::Result<()> {
        anyhow::ensure!(
            capacity > 0 && capacity <= self.config.max_position_embeddings,
            "GLM context capacity {capacity} must be in 1..={}",
            self.config.max_position_embeddings,
        );
        self.context_capacity = capacity;
        Ok(())
    }

    fn eos_token_id(&self) -> i64 {
        self.config.eos_token_id
    }

    fn eos_token_ids(&self) -> Vec<i64> {
        self.config.eos_token_ids.clone()
    }

    fn new_caches(&self) -> Vec<LayerCache> {
        (0..self.n_layers)
            .map(|i| {
                if self.config.is_kda(i) {
                    LayerCache::Linear(GdnCache::new())
                } else {
                    LayerCache::Full(FullAttentionCache::new(1, 512))
                }
            })
            .collect()
    }

    fn forward(
        &mut self,
        ids: &Array,
        caches: Option<&mut Vec<LayerCache>>,
    ) -> anyhow::Result<(Array, Array)> {
        match caches {
            Some(c) => self.forward_inner(ids, c),
            None => {
                let mut c = self.new_caches();
                self.forward_inner(ids, &mut c)
            }
        }
    }

    fn forward_ragged(
        &mut self,
        ids: &Array,
        per_row_caches: &mut [Vec<LayerCache>],
    ) -> anyhow::Result<(Array, Array)> {
        self.forward_ragged_inner(ids, per_row_caches)
    }

    fn forward_capture(
        &mut self,
        ids: &Array,
        caches: Option<&mut Vec<LayerCache>>,
        _capture: bool,
    ) -> anyhow::Result<(Array, Array)> {
        self.forward(ids, caches)
    }

    #[cfg(test)]
    fn set_debug_capture_for_test(&mut self, layers: Vec<usize>) -> anyhow::Result<()> {
        self.set_debug_capture_filtered_for_test(layers, Vec::new())
    }

    #[cfg(test)]
    fn set_debug_capture_filtered_for_test(
        &mut self,
        layers: Vec<usize>,
        key_filters: Vec<String>,
    ) -> anyhow::Result<()> {
        self.capture = Some(Capture {
            layers,
            key_filters,
            state_only: false,
            kda_prefill_only: false,
            preserve_dispatch: true,
            entries: std::sync::Mutex::new(std::collections::HashMap::new()),
            shapes: std::sync::Mutex::new(std::collections::HashMap::new()),
            flushed: std::sync::atomic::AtomicBool::new(false),
            ..Default::default()
        });
        Ok(())
    }

    #[cfg(test)]
    fn flush_debug_capture_for_test(&mut self, path: &str) -> anyhow::Result<()> {
        let capture = self
            .capture
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("debug capture was not installed"))?;
        capture.flush(path)?;
        capture
            .flushed
            .store(true, std::sync::atomic::Ordering::Release);
        Ok(())
    }

    fn enable_logit_capture(&mut self) -> anyhow::Result<()> {
        self.capture = Some(Capture {
            key_filters: vec!["final_logits".to_string()],
            preserve_dispatch: true,
            ..Default::default()
        });
        Ok(())
    }

    fn enable_stage_capture(
        &mut self,
        layers: Vec<usize>,
        mut keys: Vec<String>,
    ) -> anyhow::Result<()> {
        anyhow::ensure!(
            layers.iter().all(|&layer| layer < self.layers.len()),
            "stage capture layer exceeds the GLM trunk depth"
        );
        if !keys.is_empty() && !keys.iter().any(|key| key == "final_logits") {
            keys.push("final_logits".to_string());
        }
        self.capture = Some(Capture {
            layers,
            key_filters: keys,
            preserve_dispatch: true,
            ..Default::default()
        });
        Ok(())
    }

    fn write_logit_capture(&self, path: &str) -> anyhow::Result<()> {
        self.capture
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("logit capture is not enabled"))?
            .flush(path)
    }

    fn prefill(&mut self, tokens: &[u32], caches: &mut Vec<LayerCache>) -> anyhow::Result<Array> {
        let (last, _) = self.prefill_chunks(tokens, caches, false)?;
        Ok(last.reshape(&[self.config.hidden_size as i32])?)
    }

    fn decode_token(&mut self, token: u32, caches: &mut Vec<LayerCache>) -> anyhow::Result<Array> {
        // DS4's serial evaluator and MTP seed/replay use the same token graph.
        // A one-token prompt continuation still enters `prefill_chunks` above.
        let ids = i32_arr(&self.embed.weight, &[token])?;
        let (mixed, _) = self.forward_inner(&ids, caches)?;
        Ok(mixed.reshape(&[self.config.hidden_size as i32])?)
    }

    fn prefill_multi(
        &mut self,
        tokens: &[u32],
        caches: &mut Vec<LayerCache>,
    ) -> anyhow::Result<(Array, Array)> {
        let (last, multi) = self.prefill_chunks(tokens, caches, true)?;
        Ok((last, multi.expect("retained GLM prefill stream")))
    }

    fn head(&self, mixed: &Array) -> lisa_mlx::error::Result<Array> {
        let shape = mixed.shape();
        let rows: usize = shape[..shape.len() - 1]
            .iter()
            .map(|&d| d as usize)
            .product();
        let hdim = *shape.last().expect("non-empty") as usize;
        let flat = mixed.reshape(&[(rows) as i32, (hdim) as i32])?;
        let logits = self.lm_head.forward_glm53_prefill(&flat)?;
        let mut out_shape = shape[..shape.len() - 1].to_vec();
        if out_shape.is_empty() {
            out_shape.push(1);
        }
        out_shape.push(logits.dim(1));
        let logits = logits.reshape(&out_shape)?;
        if let Some(capture) = self.capture.as_ref().filter(|c| c.preserves_dispatch()) {
            let step = INC_STEP.load(std::sync::atomic::Ordering::Relaxed);
            capture.store("final_logits", &logits);
            capture.store(&format!("final_logits_s{step}"), &logits);
            // In dispatch-preserving capture mode, defer all tensor readbacks
            // until the production logits have completed. This prevents the
            // trace from inserting GPU/CPU synchronization points between
            // model layers.
            logits.eval()?;
            if let Ok(path) = std::env::var("LISA_GLM53_CAPTURE") {
                if let Err(error) = capture.flush(&path) {
                    eprintln!("[glm53-capture] deferred flush failed: {error}");
                }
            }
        }
        Ok(logits)
    }

    fn head_ragged(
        &self,
        mixed: &Array,
        per_row_caches: &[Vec<LayerCache>],
    ) -> lisa_mlx::error::Result<Array> {
        let row_count = per_row_caches.len();
        let dense_compact_limit = self
            .config
            .index_topk
            .saturating_add(self.config.index_kpool.saturating_sub(1));
        let max_position = per_row_caches
            .iter()
            .flat_map(|row| row.iter())
            .filter_map(|cache| match cache {
                LayerCache::Full(cache) => Some(cache.offset),
                LayerCache::Linear(_) => None,
            })
            .max()
            .unwrap_or(0);
        // The caches have already advanced once. Subtract that just-fed token
        // to apply ds4's pre-forward eligibility condition to the output head.
        if ds4_native_session_batch_position_supported(
            row_count,
            max_position.saturating_sub(1),
            dense_compact_limit,
        ) {
            return self.head(mixed);
        }
        let mut logits_rows = Vec::with_capacity(row_count);
        for row in 0..row_count {
            logits_rows.push(self.head(&narrow(mixed, 0, row, 1)?)?);
        }
        let logits_refs: Vec<&Array> = logits_rows.iter().collect();
        Array::concatenate(&logits_refs, 0)
    }

    fn warmup(&mut self, _seed: &[u32]) -> anyhow::Result<()> {
        Ok(())
    }

    fn batch_decode_ok(&self) -> bool {
        // GLM batches preserve compact latent and indexer tapes through the
        // packed equal-position path or the row-local ragged cache path.
        true
    }

    fn batch_decode_equal_length_only(&self) -> bool {
        true
    }

    fn batch_decode_ragged_supported(&self) -> bool {
        true
    }

    fn has_target_mtp(&self) -> bool {
        self.mtp.is_some()
    }

    fn target_mtp_step(
        &mut self,
        cache: &mut IndexerTape,
        target_multi: &Array,
        next_token: u32,
        position: usize,
        min_position: usize,
    ) -> anyhow::Result<Array> {
        self.mtp_step(cache, target_multi, next_token, position, min_position)
    }

    fn kv_bytes_per_token(&self) -> usize {
        // Admission conservatively charges F32 indexer key/gate rows per input
        // token because long prefill holds them until the first pool update.
        // After pooling, only a bounded rollback tail remains; compact MLA
        // latents and pooled keys stay F16 for the context lifetime.
        let dsa = (0..self.n_layers)
            .filter(|&i| !self.config.is_kda(i))
            .count();
        let raw_indexer = 2 * self.config.index_head_dim * 4;
        let compact_latent = self.config.kv_lora_rank * 2;
        let pooled_per_token = self.config.index_head_dim * 2 / self.config.index_kpool;
        dsa * (raw_indexer + compact_latent + pooled_per_token)
    }

    fn stream_state_bytes(&self) -> usize {
        // KDA: ds4-compatible F32 conv window + F32 recurrence state.
        let qkv = self.config.linear_num_heads * self.config.linear_head_dim;
        let per_kda = (self.config.linear_conv_kernel_dim - 1) * qkv * 3 * 4
            + self.config.linear_num_heads * self.config.linear_head_dim.pow(2) * 4;
        let kda = (0..self.n_layers)
            .filter(|&i| self.config.is_kda(i))
            .count();
        kda * per_kda
    }

    fn mtp_cost_key(&self) -> String {
        format!("glm53-{}L-{}h", self.n_layers, self.config.hidden_size)
    }
}

#[cfg(test)]
#[path = "tower_tests.rs"]
mod tests;
