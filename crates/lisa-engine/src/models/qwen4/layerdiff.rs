//! Layer-diff against the Python reference dumps in /tmp/lisa-diff.

use anyhow::Context;
use lisa_mlx::ops::indexing::IndexOp;
use lisa_mlx::{ops, Array, Dtype};
use std::path::{Path, PathBuf};

use crate::models::qwen4::config::ModelConfig;
use crate::models::qwen4::tower::{Block, Tower};

fn load_bin(name: &str) -> anyhow::Result<Vec<f32>> {
    let p = PathBuf::from("/tmp/lisa-diff").join(format!("{name}.bin"));
    let data = std::fs::read(&p).with_context(|| format!("reading {}", p.display()))?;
    Ok(data
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect())
}

fn diff(name: &str, arr: &Array) -> anyhow::Result<f32> {
    // save my side for elementwise analysis
    {
        let got_f: Vec<f32> = arr
            .as_dtype(Dtype::Float32)
            .map_err(|e| anyhow::anyhow!("{e}"))?
            .reshape(&[-1])?
            .as_slice::<f32>()
            .to_vec();
        let path = format!("/tmp/lisa-diff/rs_{name}.bin");
        std::fs::write(&path, unsafe {
            std::slice::from_raw_parts(got_f.as_ptr() as *const u8, got_f.len() * 4)
        })
        .ok();
    }
    let got: Vec<f32> = arr
        .as_dtype(Dtype::Float32)
        .map_err(|e| anyhow::anyhow!("{e}"))?
        .reshape(&[-1])?
        .as_slice::<f32>()
        .to_vec();
    let want = match load_bin(name) {
        Ok(w) if w.len() == got.len() => w,
        _ => {
            println!("{name}: (no python reference at this length; saved rs side)");
            return Ok(0.0);
        }
    };
    let mut max = 0f32;
    for (a, b) in got.iter().zip(want.iter()) {
        max = max.max((a - b).abs());
    }
    println!("{name}: max abs diff {max:.6}");
    Ok(max)
}

pub fn run(model: &Path) -> anyhow::Result<()> {
    {
        // Weight sanity: final mixer hc_norm
        let config = ModelConfig::from_json(&model.join("config.json"))?;
        let tower = Tower::load(model, config)?;
        let w = &tower.final_mixer.hc_norm.weight;
        let wf: Vec<f32> = w
            .as_dtype(Dtype::Float32)
            .map_err(|e| anyhow::anyhow!("{e}"))?
            .as_slice::<f32>()
            .to_vec();
        let mn = wf.iter().cloned().fold(f32::INFINITY, f32::min);
        let mx = wf.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mean = wf.iter().sum::<f32>() / wf.len() as f32;
        println!("rs mixer hc_norm weight: min {mn:.4} max {mx:.4} mean {mean:.4}");
        println!(
            "rs mixer hc_norm group_size: {:?} eps {}",
            tower.final_mixer.hc_norm.group_size, tower.final_mixer.hc_norm.eps
        );
        drop(tower);
    }
    // (tower reloaded below for the actual diff)
    let config = ModelConfig::from_json(&model.join("config.json"))?;
    let mut tower = Tower::load(model, config)?;
    let _ = &tower;

    // Re-run the tiny forward manually using the tower's modules.
    let golden: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string("reference/correctness_prompts/public_longcopy_gate_english_1024_256.json")?)?;
    let take: usize = std::env::var("LISA_LDTOKENS").ok().and_then(|v| v.parse().ok()).unwrap_or(48);
    let ptok: Vec<i32> = golden["cases"][0]["prompt_tokens"]
        .as_array().unwrap().iter().take(take)
        .filter_map(|v| v.as_i64().map(|x| x as i32)).collect();
    let n_tok = ptok.len() as i32;
    let ids = Array::from_slice(&ptok, &[1i32, n_tok]);
    let hidden = tower.embed_tokens.forward(&ids)?;
    diff("01_embed", &hidden)?;
    let hidden = ops::tile(&hidden, &[1, 1, tower.config.hc_count as i32])?;
    diff("02_tile", &hidden)?;

    let layer = &tower.layers[0];
    let normed = layer.attn_hc.hc_norm.forward(&hidden)?;
    if std::env::var("LISA_DUMP03").is_ok() {
        let f = normed.as_dtype(Dtype::Float32)?;
        let v = f.as_slice::<f32>().to_vec();
        std::fs::write("/tmp/rms-trace/zz03.bin", unsafe {
            std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4)
        }).ok();
    }
    diff("03_hc_norm", &normed)?;
    let lo = layer.attn_hc.mix_down.forward(&normed)?;
    diff("04_down", &lo)?;
    let lo4 = lo / crate::core::norm::bf16_scalar(4.0);
    let w = lisa_mlx::nn::silu(&lo4)?;
    let w = layer.attn_hc.mix_up.forward(&w)?;
    let w = ops::sigmoid(&w)?;
    diff("05_up", &w)?;
    let mut lead = w.shape().to_vec();
    lead.pop();
    lead.push(4);
    lead.push(2560);
    let mixed = (w.reshape(&lead)?.multiply(&normed.reshape(&lead)?)?).mean_axis(-2, None)?;
    diff("06_mixed", &mixed)?;
    let inj = layer.attn_hc.block_inject.as_ref().unwrap().forward(&normed)?;
    let inj = ops::sigmoid(&(inj / crate::core::norm::bf16_scalar(4.0)))? * crate::core::norm::bf16_scalar(2.0);
    diff("07_inject_w", &inj)?;

    // GDN
    let gdn = match &layer.block {
        Block::Linear(g) => g,
        _ => anyhow::bail!("layer 0 not linear"),
    };
    let b = ids.dim(0);
    let s = ids.dim(1);
    let mixed_qkv = gdn.in_proj_qkv.forward(&mixed)?;
    diff("08_qkv", &mixed_qkv)?;
    let z = gdn.in_proj_z.forward(&mixed)?;
    let z = z.reshape(&[b, s, gdn.value_heads as i32, gdn.value_head_dim as i32])?;
    let b_proj = gdn.in_proj_b.forward(&mixed)?;
    let a_proj = gdn.in_proj_a.forward(&mixed)?;

    let zeros = ops::zeros::<half::bf16>(&[b, (gdn.conv_kernel_size - 1) as i32, gdn.conv_dim as i32])?;
    let conv_input = ops::concatenate(&[&zeros, &mixed_qkv], 1)?;
    let conv_out = ops::conv1d(
        &conv_input,
        &gdn.conv1d_weight,
        None,
        None,
        None,
        Some(gdn.conv_dim as i32),
    )?;
    let conv_out = lisa_mlx::nn::silu(&conv_out)?;
    diff("09_conv_out", &conv_out)?;

    let key_dim = (gdn.key_heads * gdn.key_head_dim) as i32;
    let q = conv_out.index((.., .., 0..key_dim));
    let k = conv_out.index((.., .., key_dim..(2 * key_dim)));
    let v = conv_out.index((.., .., (2 * key_dim)..));
    let q = q.reshape(&[b, s, gdn.key_heads as i32, gdn.key_head_dim as i32])?;
    let k = k.reshape(&[b, s, gdn.key_heads as i32, gdn.key_head_dim as i32])?;
    let v = v.reshape(&[b, s, gdn.value_heads as i32, gdn.value_head_dim as i32])?;
    let inv_scale = (gdn.key_head_dim as f32).powf(-0.5);
    let q = lisa_mlx::fast::rms_norm(&q, None, 1e-6)? * crate::core::norm::bf16_scalar(inv_scale * inv_scale);
    let k = lisa_mlx::fast::rms_norm(&k, None, 1e-6)? * crate::core::norm::bf16_scalar(inv_scale);
    diff("10_q", &q)?;
    diff("11_k", &k)?;
    diff("12_v", &v)?;

    let beta = ops::sigmoid(&b_proj)?.as_dtype(Dtype::Float32)?;
    diff("g_beta", &beta)?;
    let a_log = gdn.a_log.as_dtype(Dtype::Float32)?;
    let neg_exp_alog = -(ops::exp(&a_log)?);
    let ax = a_proj.add(&gdn.dt_bias)?;
    let sp = crate::core::norm::bf16_logaddexp0(&ax)?;
    let sp_f = sp.as_dtype(Dtype::Float32)?;
    let g = ops::exp(&(neg_exp_alog * sp_f))?;
    diff("g_g", &g)?;

    let zeros_state = ops::zeros::<f32>(&[
        b,
        gdn.value_heads as i32,
        gdn.value_head_dim as i32,
        gdn.key_head_dim as i32,
    ])?;
    let stream = lisa_mlx::Stream::thread_local_or_default();
    let (out, state) =
        lisa_mlx::kernels::gated_delta_kernel(&q, &k, &v, &g, &beta, &zeros_state, false, &stream)
            .ok_or_else(|| anyhow::anyhow!("gdn kernel failed"))?;
    diff("14_gdn_out", &out)?;
    diff("15_gdn_state", &state)?;

    let normed_out = gdn.norm.forward(&out, &z)?;
    let normed_out = normed_out.reshape(&[b, s, -1])?;
    diff("16_gated", &normed_out)?;
    let attended = gdn.out_proj.forward(&normed_out)?;
    diff("17_attended", &attended)?;

    let stream2 = crate::models::qwen4::hyper::inject(&hidden, &attended, &inj).map_err(|e| anyhow::anyhow!("{e}"))?;
    diff("18_after_attn_inject", &stream2)?;

    // MLP side
    let normed2 = layer.mlp_hc.hc_norm.forward(&stream2)?;
    let lo2 = layer.mlp_hc.mix_down.forward(&normed2)?;
    let lo2 = lo2 / crate::core::norm::bf16_scalar(4.0);
    let w2 = lisa_mlx::nn::silu(&lo2)?;
    let w2 = layer.mlp_hc.mix_up.forward(&w2)?;
    let w2 = ops::sigmoid(&w2)?;
    let mixed2 = (w2.reshape(&lead)?.multiply(&normed2.reshape(&lead)?)?).mean_axis(-2, None)?;
    diff("19_moe_in", &mixed2)?;
    let inj2 = layer.mlp_hc.block_inject.as_ref().unwrap().forward(&normed2)?;
    let inj2 = ops::sigmoid(&(inj2 / crate::core::norm::bf16_scalar(4.0)))? * crate::core::norm::bf16_scalar(2.0);

    let moe = &layer.mlp;
    let logits_r = ops::matmul(&mixed2.as_dtype(Dtype::Float32)?, moe.gate.transpose_axes(&[1, 0])?)?;
    diff("20_routerlogits", &logits_r)?;
    let indices = ops::argpartition_axis(&(-&logits_r), (moe.top_k - 1) as i32, -1)?;
    let indices = indices.index((.., .., 0..moe.top_k as i32));
    let selected = logits_r.take_along_axis(&indices, -1)?;
    let weights = ops::softmax_axis(&selected, -1, true)?;
    diff("20_router_w", &weights)?;
    let idx_f = indices.as_dtype(Dtype::Float32).map_err(|e| anyhow::anyhow!("{e}"))?;
    diff("21_router_idx", &idx_f)?;

    let rows = b * s;
    let x_rows = mixed2.reshape(&[rows, mixed2.dim(-1)])?;
    let x_rep = ops::repeat_axis::<half::bf16>(x_rows, moe.top_k as i32, 0)?;
    let x_rep = x_rep.reshape(&[rows * moe.top_k as i32, 1, mixed2.dim(-1)])?;
    let idx_flat = indices.reshape(&[rows * moe.top_k as i32])?;
    let gq = |inp: &Array, key: &str| -> lisa_mlx::error::Result<Array> {
        let (sw, ss, sb) = match key {
            "gate_proj" => (&moe.switch_mlp.gate_proj, &moe.switch_mlp.gate_scales, &moe.switch_mlp.gate_biases),
            "up_proj" => (&moe.switch_mlp.up_proj, &moe.switch_mlp.up_scales, &moe.switch_mlp.up_biases),
            _ => (&moe.switch_mlp.down_proj, &moe.switch_mlp.down_scales, &moe.switch_mlp.down_biases),
        };
        ops::gather_qmm(inp, sw, ss, Some(sb), None, Some(&idx_flat), true, 32, 4, false)?
            .reshape(&[rows * moe.top_k as i32, -1])
    };
    let gate_out = gq(&x_rep, "gate_proj")?;
    let up_out = gq(&x_rep, "up_proj")?;
    let act = lisa_mlx::nn::silu(&gate_out)?.multiply(&up_out)?;
    let down_out = gq(&act.reshape(&[rows * moe.top_k as i32, 1, act.dim(-1)])?, "down_proj")?;
    let down = down_out.reshape(&[b, s, moe.top_k as i32, -1])?;
    let routed = down.multiply(&weights.expand_dims(-1)?)?.sum_axis(-2, None)?;
    let routed = routed.as_dtype(mixed2.dtype())?;
    diff("22_routed", &routed)?;

    let sg = moe.shared_expert_gate.forward(&mixed2)?;
    let sh = moe.shared_gate_proj.forward(&mixed2)?;
    let sh = lisa_mlx::nn::silu(&sh)?.multiply(&moe.shared_up_proj.forward(&mixed2)?)?;
    let sh = moe.shared_down_proj.forward(&sh)?;
    diff("23_moe_out", &routed.add(&sh.multiply(&ops::sigmoid(&sg)?)?)?)?;

    let moe_added = routed.add(&sh.multiply(&ops::sigmoid(&sg)?)?)?;
    let layer_out = crate::models::qwen4::hyper::inject(&stream2, &moe_added, &inj2)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    diff("24_layer0_out", &layer_out)?;
    diff_rest(&mut tower, layer_out, &ids)
}

/// Layers 1-2 (with PLE) then layer-3 attention internals.
fn diff_rest(tower: &mut Tower, mut hidden: Array, ids: &Array) -> anyhow::Result<()> {
    let _ = &mut hidden;
    let cfg = &tower.config;
    let b = ids.dim(0);
    let s = ids.dim(1);

    // --- layer 1: PLE + GDN + MoE ---
    let token_rows = Tower::token_rows(ids)?;
    let prev_ctx = vec![vec![248044i64; 2]; b as usize];
    let (ple_out, last_embed, last_gated) = {
        let ple = tower.layers[1].ple.as_mut().unwrap();
        let out = ple.forward(&hidden, &token_rows, &prev_ctx, &mut None, false, &mut None)?;
        (out, ple.last_embed.clone().unwrap(), ple.last_gated.clone().unwrap())
    };
    diff("30_ple_embed", &last_embed)?;
    diff("31_ple_gated", &last_gated)?;
    let hidden = hidden.add(&ple_out)?;

    // attn hc + gdn + mlp hc + moe via the real layer forward pieces
    let hidden = run_linear_body(tower, &1, hidden)?;
    diff("39_layer1_out", &hidden)?;
    let hidden = run_linear_body(tower, &2, hidden)?;
    diff("40_layer2_out", &hidden)?;

    // --- layer 3 attention internals ---
    let layer3 = &tower.layers[3];
    let attn = match &layer3.block {
        Block::Attention(a) => a,
        _ => anyhow::bail!("layer 3 not attention"),
    };
    let prefix = "model.layers.3";
    let _ = prefix;
    let (mixed3, inj3) = {
        let normed = layer3.attn_hc.hc_norm.forward(&hidden)?;
        let lo = layer3.attn_hc.mix_down.forward(&normed)?;
        let wmix = ops::sigmoid(&layer3.attn_hc.mix_up.forward(&lisa_mlx::nn::silu(&(lo / crate::core::norm::bf16_scalar(4.0)))?)?)?;
        let mut lead = wmix.shape().to_vec();
        lead.pop();
        lead.push(4);
        lead.push(2560);
        let mixed = (wmix.reshape(&lead)?.multiply(&normed.reshape(&lead)?)?).mean_axis(-2, None)?;
        let inj = ops::sigmoid(&(layer3.attn_hc.block_inject.as_ref().unwrap().forward(&normed)? / crate::core::norm::bf16_scalar(4.0)))? * crate::core::norm::bf16_scalar(2.0);
        (mixed, inj)
    };
    diff("41_layer3_mixed", &mixed3)?;
    // Surgical: feed the PYTHON-computed layer-3 input so any downstream diff
    // is purely the attention path.
    let mixed3_py = {
        let data = load_bin("41_layer3_mixed")?;
        Array::from_slice(&data, &[b, s, mixed3.dim(-1)])
            .as_dtype(Dtype::Bfloat16)
            .map_err(|e| anyhow::anyhow!("{e}"))?
    };
    let mixed3 = mixed3_py;

    let projected = attn.q_proj.forward(&mixed3)?;
    let projected = projected.reshape(&[b, s, attn.heads as i32, -1])?;
    let split = projected.split_equal(2, -1)?;
    let queries = &split[0];
    let gate = split[1].reshape(&[b, s, -1])?;
    diff("42_q_proj", &queries)?;
    let queries = attn.q_norm.forward(queries)?;
    diff("43_q_norm", &queries)?;
    let keys = attn.k_proj.forward(&mixed3)?;
    let keys = keys.reshape(&[b, s, attn.kv_heads as i32, attn.head_dim as i32])?;
    let keys = attn.k_norm.forward(&keys)?;
    diff("44_k_norm", &keys)?;
    let values = attn.v_proj.forward(&mixed3)?;
    let values = values.reshape(&[b, s, attn.kv_heads as i32, attn.head_dim as i32])?;
    diff("45_v", &values)?;

    let rope = &tower.rope;
    let (cos, sin) = rope.cos_sin(&crate::core::norm::positions(0, s as usize)?)?;
    diff("46_cos", &cos)?;
    let cos4 = cos.expand_dims(1)?;
    let sin4 = sin.expand_dims(1)?;
    let q4 = queries.transpose_axes(&[0, 2, 1, 3])?;
    let k4 = keys.transpose_axes(&[0, 2, 1, 3])?;
    let v4 = values.transpose_axes(&[0, 2, 1, 3])?;
    let q4 = crate::core::norm::rope_partial(&q4, &cos4, &sin4)?;
    let k4 = crate::core::norm::rope_partial(&k4, &cos4, &sin4)?;
    diff("47_q_rope", &q4)?;
    diff("48_k_rope", &k4)?;
    let out = lisa_mlx::fast::scaled_dot_product_attention(
        &q4, &k4, &v4, attn.scale,
        lisa_mlx::fast::ScaledDotProductAttentionMask::Causal,
        None,
    )?;
    diff("49_sdpa", &out)?;
    let out = out.transpose_axes(&[0, 2, 1, 3])?.reshape(&[b, s, -1])?;
    let gated = out.multiply(&ops::sigmoid(&gate)?)?;
    diff("50_gated", &gated)?;
    let att = attn.o_proj.forward(&gated)?;
    diff("51_o_proj", &att)?;
    let _ = (inj3, cfg);

    // --- final mixer + lm_head on the SAME hidden Python used (layer-2 out)
    // Feed PYTHON's hidden so the mixer + head are compared on identical input
    let hidden_py = {
        let data = load_bin("40_layer2_out")?;
        Array::from_slice(&data, &[1i32, s, hidden.dim(-1)])
            .as_dtype(Dtype::Bfloat16)
            .map_err(|e| anyhow::anyhow!("{e}"))?
    };
    let normed_final = tower.final_mixer.hc_norm.forward(&hidden_py)?;
    diff("60_mixer_normed", &normed_final)?;
    let mixed_out = tower.final_mixer.mixed_from_normed(&normed_final)?;
    diff("61_mixed", &mixed_out)?;
    let logits = tower.lm_head.forward(&mixed_out)?;
    diff("62_logits", &logits)?;
    Ok(())
}

/// hc mix -> GDN (no cache) -> inject -> hc mix -> MoE -> inject.
fn run_linear_body(tower: &Tower, idx: &usize, hidden: Array) -> anyhow::Result<Array> {
    let layer = &tower.layers[*idx];
    let lp = format!("model.layers.{idx}");
    let _ = &lp;
    let (input, residual, attn_inject) = layer.attn_hc.mix_with_inject(&hidden)?;
    let attended = match &layer.block {
        Block::Linear(gdn) => gdn.forward(&input, None, false)?,
        _ => anyhow::bail!("expected linear"),
    };
    let stream = crate::models::qwen4::hyper::inject(&residual, &attended, &attn_inject)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    let (input2, residual2, mlp_inject) = layer.mlp_hc.mix_with_inject(&stream)?;
    let out = layer.mlp.forward(&input2)?;
    crate::models::qwen4::hyper::inject(&residual2, &out, &mlp_inject).map_err(|e| anyhow::anyhow!("{e}"))
}
