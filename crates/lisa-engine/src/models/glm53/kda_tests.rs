use super::*;
use crate::core::loader::{Checkpoint, Shard, sanitize_name};
use crate::core::quant::get_tensor;
use lisa_mlx::ops::indexing::IndexOp;
use std::collections::HashMap;

fn load_layer_kda_weights_for(dir: &std::path::Path, layer: usize) -> anyhow::Result<Weights> {
    let checkpoint = Checkpoint::open(dir)?;
    let prefix = format!("model.layers.{layer}.self_attn.");
    let mut by_shard: HashMap<String, Vec<(String, String)>> = HashMap::new();
    for (raw, shard) in &checkpoint.weight_map {
        let Some(name) = sanitize_name(raw) else {
            continue;
        };
        if name.starts_with(&prefix) {
            by_shard
                .entry(shard.clone())
                .or_default()
                .push((raw.clone(), name));
        }
    }
    let mut weights = Weights::new();
    for (shard_name, names) in by_shard {
        let shard = Shard::open(&dir.join(shard_name))?;
        for (raw, name) in names {
            weights.insert(name, get_tensor(&shard, &raw)?);
        }
    }
    Ok(weights)
}

fn load_layer_kda_weights(dir: &std::path::Path) -> anyhow::Result<Weights> {
    load_layer_kda_weights_for(dir, 0)
}

fn assert_projection_fixture(
    expected: &[u8],
    names: &[&str],
    outputs: &[Vec<f32>],
) -> anyhow::Result<()> {
    anyhow::ensure!(
        names.len() == outputs.len(),
        "projection fixture arity differs"
    );
    anyhow::ensure!(
        expected.len()
            == outputs
                .iter()
                .map(|output| output.len() * std::mem::size_of::<f32>())
                .sum::<usize>(),
        "DS4 projection fixture has {} bytes",
        expected.len()
    );
    let mut offset = 0;
    for (name, output) in names.iter().zip(outputs) {
        for (index, actual) in output.iter().enumerate() {
            let start = offset + index * std::mem::size_of::<f32>();
            let want = f32::from_le_bytes(
                expected[start..start + 4]
                    .try_into()
                    .expect("four-byte float"),
            );
            anyhow::ensure!(
                actual.to_bits() == want.to_bits(),
                "KDA {name} output {index} differs from DS4: {actual} vs {want}"
            );
        }
        offset += output.len() * std::mem::size_of::<f32>();
    }
    Ok(())
}

#[test]
fn glm53_kda_layer21_input_projections_match_direct_ds4_fixture() -> anyhow::Result<()> {
    if !lisa_mlx::device::metal_available() {
        eprintln!("[glm53-kda-ds4-projections] skipping: no Metal device available");
        return Ok(());
    }
    let Some(model_dir) = std::env::var_os("LISA_GLM53_MODEL") else {
        eprintln!("[glm53-kda-ds4-projections] skipping: LISA_GLM53_MODEL is unset");
        return Ok(());
    };
    let dir = std::path::Path::new(&model_dir);
    let config = super::super::config::ModelConfig::from_json(&dir.join("config.json"))?;
    anyhow::ensure!(config.is_kda(21), "GLM layer 21 is not a KDA layer");
    let mut weights = load_layer_kda_weights_for(dir, 21)?;
    let kda = Kda::load::<Weights>(
        &mut weights,
        "model.layers.21.self_attn",
        config.linear_num_heads,
        config.linear_head_dim,
        config.linear_conv_kernel_dim,
        config.linear_lower_bound,
        config.rms_norm_eps,
    )?;

    let input_values: Vec<f32> = (0..config.hidden_size)
        .map(|i| (((i * 37 + 13) % 251) as f32 - 125.0) * 0.0004)
        .collect();
    let input = Array::from_slice(&input_values, &[1, 1, config.hidden_size as i32]);
    let names = ["q", "k", "v", "f_a", "g_a", "beta"];
    let staged = vec![
        linear_projection(&input, &kda.q_w)?.to_vec1::<f32>()?,
        linear_projection(&input, &kda.k_w)?.to_vec1::<f32>()?,
        linear_projection(&input, &kda.v_w)?.to_vec1::<f32>()?,
        linear_projection(&input, &kda.f_a_w)?.to_vec1::<f32>()?,
        linear_projection(&input, &kda.g_a_w)?.to_vec1::<f32>()?,
        linear_projection(&input, &kda.b_w)?.to_vec1::<f32>()?,
    ];
    let (q, k, v, fa, ga, beta) = kda.input_projections(&input)?;
    let fused = vec![
        q.to_vec1::<f32>()?,
        k.to_vec1::<f32>()?,
        v.to_vec1::<f32>()?,
        fa.to_vec1::<f32>()?,
        ga.to_vec1::<f32>()?,
        beta.to_vec1::<f32>()?,
    ];
    let expected = &crate::models::glm53::test_fixtures::read("kda_l21_inputproj_ds4.f32");
    assert_projection_fixture(expected, &names, &staged)?;
    assert_projection_fixture(expected, &names, &fused)?;
    Ok(())
}

#[test]
fn glm53_kda_layer21_multirow_projections_match_direct_ds4_fixtures() -> anyhow::Result<()> {
    if !lisa_mlx::device::metal_available() {
        eprintln!("[glm53-kda-ds4-batch] skipping: no Metal device available");
        return Ok(());
    }
    let Some(model_dir) = std::env::var_os("LISA_GLM53_MODEL") else {
        eprintln!("[glm53-kda-ds4-batch] skipping: LISA_GLM53_MODEL is unset");
        return Ok(());
    };
    let dir = std::path::Path::new(&model_dir);
    let config = super::super::config::ModelConfig::from_json(&dir.join("config.json"))?;
    anyhow::ensure!(config.is_kda(21), "GLM layer 21 is not a KDA layer");
    let mut weights = load_layer_kda_weights_for(dir, 21)?;
    let kda = Kda::load::<Weights>(
        &mut weights,
        "model.layers.21.self_attn",
        config.linear_num_heads,
        config.linear_head_dim,
        config.linear_conv_kernel_dim,
        config.linear_lower_bound,
        config.rms_norm_eps,
    )?;

    for (rows, expected) in [
        (
            2usize,
            crate::models::glm53::test_fixtures::read("kda_l21_inputproj_ds4_2.f32"),
        ),
        (
            8usize,
            crate::models::glm53::test_fixtures::read("kda_l21_inputproj_ds4_8.f32"),
        ),
    ] {
        let input_values: Vec<f32> = (0..rows * config.hidden_size)
            .map(|i| (((i * 37 + 13) % 251) as f32 - 125.0) * 0.0004)
            .collect();
        let input = Array::from_slice(
            &input_values,
            &[rows as i32, 1, config.hidden_size as i32],
        );
        let (q, k, v, fa, ga, beta) = kda.input_projections(&input)?;
        let outputs = vec![
            q.to_vec1::<f32>()?,
            k.to_vec1::<f32>()?,
            v.to_vec1::<f32>()?,
            fa.to_vec1::<f32>()?,
            ga.to_vec1::<f32>()?,
            beta.to_vec1::<f32>()?,
        ];
        assert_projection_fixture(&expected, &["q", "k", "v", "f_a", "g_a", "beta"], &outputs)
            .map_err(|error| anyhow::anyhow!("{rows}-row projections: {error}"))?;
    }
    Ok(())
}

#[test]
fn glm53_kda_layer21_captured_activation_matches_direct_ds4_fixture() -> anyhow::Result<()> {
    if !lisa_mlx::device::metal_available() {
        eprintln!("[glm53-kda-ds4-captured] skipping: no Metal device available");
        return Ok(());
    }
    let Some(model_dir) = std::env::var_os("LISA_GLM53_MODEL") else {
        eprintln!("[glm53-kda-ds4-captured] skipping: LISA_GLM53_MODEL is unset");
        return Ok(());
    };
    let dir = std::path::Path::new(&model_dir);
    let config = super::super::config::ModelConfig::from_json(&dir.join("config.json"))?;
    anyhow::ensure!(config.is_kda(21), "GLM layer 21 is not a KDA layer");
    let mut weights = load_layer_kda_weights_for(dir, 21)?;
    let kda = Kda::load::<Weights>(
        &mut weights,
        "model.layers.21.self_attn",
        config.linear_num_heads,
        config.linear_head_dim,
        config.linear_conv_kernel_dim,
        config.linear_lower_bound,
        config.rms_norm_eps,
    )?;

    let input_bytes =
        &crate::models::glm53::test_fixtures::read("kda_l21_captured_input_ids1234.f32");
    anyhow::ensure!(
        input_bytes.len() == 4 * config.hidden_size * std::mem::size_of::<f32>(),
        "captured layer-21 input fixture has {} bytes",
        input_bytes.len()
    );
    let input_values = input_bytes
        .chunks_exact(std::mem::size_of::<f32>())
        .map(|bytes| f32::from_le_bytes(bytes.try_into().expect("four-byte float")))
        .collect::<Vec<_>>();
    let input = Array::from_slice(&input_values, &[1, 4, config.hidden_size as i32]);
    let (q, k, v, fa, ga, beta) = kda.input_projections(&input)?;
    let outputs = vec![
        q.to_vec1::<f32>()?,
        k.to_vec1::<f32>()?,
        v.to_vec1::<f32>()?,
        fa.to_vec1::<f32>()?,
        ga.to_vec1::<f32>()?,
        beta.to_vec1::<f32>()?,
    ];
    let expected =
        &crate::models::glm53::test_fixtures::read("kda_l21_captured_inputproj_ds4_ids1234.f32");
    assert_projection_fixture(&expected, &["q", "k", "v", "f_a", "g_a", "beta"], &outputs)
}

#[test]
fn glm53_kda_layer21_captured_prefill_matches_direct_ds4_fixture() -> anyhow::Result<()> {
    if !lisa_mlx::device::metal_available() {
        eprintln!("[glm53-kda-ds4-prefill] skipping: no Metal device available");
        return Ok(());
    }
    let Some(model_dir) = std::env::var_os("LISA_GLM53_MODEL") else {
        eprintln!("[glm53-kda-ds4-prefill] skipping: LISA_GLM53_MODEL is unset");
        return Ok(());
    };
    let dir = std::path::Path::new(&model_dir);
    let config = super::super::config::ModelConfig::from_json(&dir.join("config.json"))?;
    anyhow::ensure!(config.is_kda(21), "GLM layer 21 is not a KDA layer");
    let mut weights = load_layer_kda_weights_for(dir, 21)?;
    let kda = Kda::load::<Weights>(
        &mut weights,
        "model.layers.21.self_attn",
        config.linear_num_heads,
        config.linear_head_dim,
        config.linear_conv_kernel_dim,
        config.linear_lower_bound,
        config.rms_norm_eps,
    )?;

    let input_bytes =
        &crate::models::glm53::test_fixtures::read("kda_l21_captured_input_ids1234.f32");
    anyhow::ensure!(
        input_bytes.len() == 4 * config.hidden_size * std::mem::size_of::<f32>(),
        "captured layer-21 input fixture has {} bytes",
        input_bytes.len()
    );
    let input_values = input_bytes
        .chunks_exact(std::mem::size_of::<f32>())
        .map(|bytes| f32::from_le_bytes(bytes.try_into().expect("four-byte float")))
        .collect::<Vec<_>>();
    let input = Array::from_slice(&input_values, &[1, 4, config.hidden_size as i32]);
    let capture = capture::Capture {
        layers: vec![21],
        state_only: false,
        kda_prefill_only: false,
        entries: std::sync::Mutex::new(HashMap::new()),
        shapes: std::sync::Mutex::new(HashMap::new()),
        flushed: std::sync::atomic::AtomicBool::new(false),
        ..Default::default()
    };
    let mut conv_state = None;
    let mut recurrent_state = None;
    let projected = kda.forward(
        &input,
        &mut conv_state,
        &mut recurrent_state,
        Some(&capture),
        21,
    )?;

    let keys = [
        "L21.kda_qc",
        "L21.kda_kc",
        "L21.kda_vc",
        "L21.kda_g",
        "L21.kda_conv_state",
        "L21.fused_state",
        "L21.kda_outnorm",
    ];
    let entries = capture.entries.lock().expect("capture entries lock");
    let outputs = keys
        .iter()
        .map(|key| {
            entries
                .get(*key)
                .cloned()
                .ok_or_else(|| anyhow::anyhow!("missing DS4 comparison capture {key}"))
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    assert_projection_fixture(
        &crate::models::glm53::test_fixtures::read("kda_l21_prefill_ds4_ids1234.f32"),
        &[
            "q",
            "k",
            "v",
            "decay gate",
            "conv state",
            "recurrent state",
            "output",
        ],
        &outputs,
    )?;
    assert_projection_fixture(
        &crate::models::glm53::test_fixtures::read("kda_l21_output_proj_ds4_ids1234.f32"),
        &["output projection"],
        &[projected
            .reshape(&[4 * config.hidden_size as i32])?
            .to_vec1::<f32>()?],
    )?;

    let first_chunk = narrow(&input, 1, 0, 3)?.contiguous()?;
    let final_token = narrow(&input, 1, 3, 1)?.contiguous()?;
    let mut chunk_conv = None;
    let mut chunk_state = None;
    let _ = kda.forward_prefill(&first_chunk, &mut chunk_conv, &mut chunk_state, None, 21)?;
    let final_output =
        kda.forward_prefill(&final_token, &mut chunk_conv, &mut chunk_state, None, 21)?;
    let chunk_conv = chunk_conv
        .ok_or_else(|| anyhow::anyhow!("chunked DS4 prefill lost convolution state"))?
        .to_vec1::<f32>()?;
    let chunk_state = chunk_state
        .ok_or_else(|| anyhow::anyhow!("chunked DS4 prefill lost recurrent state"))?
        .to_vec1::<f32>()?;
    let projection = config.linear_num_heads * config.linear_head_dim;
    let conv_bytes = 9 * projection * std::mem::size_of::<f32>();
    let state_bytes = config.linear_num_heads
        * config.linear_head_dim
        * config.linear_head_dim
        * std::mem::size_of::<f32>();
    let prefill_fixture =
        &crate::models::glm53::test_fixtures::read("kda_l21_prefill_ds4_ids1234.f32");
    let conv_start = 4 * 4 * projection * std::mem::size_of::<f32>();
    let state_start = conv_start + conv_bytes;
    assert_projection_fixture(
        &prefill_fixture[conv_start..state_start + state_bytes],
        &["chunked convolution state", "chunked recurrent state"],
        &[chunk_conv, chunk_state],
    )?;
    let expected_output =
        &crate::models::glm53::test_fixtures::read("kda_l21_output_proj_ds4_ids1234.f32");
    let output_start = 3 * config.hidden_size * std::mem::size_of::<f32>();
    assert_projection_fixture(
        &expected_output[output_start..],
        &["chunked final output projection"],
        &[final_output
            .reshape(&[config.hidden_size as i32])?
            .to_vec1::<f32>()?],
    )
}

#[test]
fn glm53_kda_layer1_mtp_rejection_verify_matches_direct_ds4() -> anyhow::Result<()> {
    if !lisa_mlx::device::metal_available() {
        eprintln!("[glm53-kda-mtp-verify] skipping: no Metal device available");
        return Ok(());
    }
    let Some(model_dir) = std::env::var_os("LISA_GLM53_MODEL") else {
        eprintln!("[glm53-kda-mtp-verify] skipping: LISA_GLM53_MODEL is unset");
        return Ok(());
    };
    let dir = std::path::Path::new(&model_dir);
    let config = super::super::config::ModelConfig::from_json(&dir.join("config.json"))?;
    anyhow::ensure!(config.is_kda(1), "GLM layer 1 is not a KDA layer");
    let mut weights = load_layer_kda_weights_for(dir, 1)?;
    let kda = Kda::load::<Weights>(
        &mut weights,
        "model.layers.1.self_attn",
        config.linear_num_heads,
        config.linear_head_dim,
        config.linear_conv_kernel_dim,
        config.linear_lower_bound,
        config.rms_norm_eps,
    )?;

    let projection = config.linear_num_heads * config.linear_head_dim;
    let input_count = 2 * config.hidden_size;
    let conv_count = 9 * projection;
    let state_count = config.linear_num_heads * config.linear_head_dim * config.linear_head_dim;
    let input_fixture =
        &crate::models::glm53::test_fixtures::read("kda_l1_mtp_reject_verify_input.bin");
    let expected_bytes = (input_count + conv_count + state_count) * std::mem::size_of::<f32>();
    anyhow::ensure!(
        input_fixture.len() == expected_bytes,
        "layer-1 MTP verify input fixture has {} bytes, expected {expected_bytes}",
        input_fixture.len()
    );
    let values = input_fixture
        .chunks_exact(std::mem::size_of::<f32>())
        .map(|bytes| f32::from_le_bytes(bytes.try_into().expect("four-byte float")))
        .collect::<Vec<_>>();
    let input = Array::from_slice(&values[..input_count], &[1, 2, config.hidden_size as i32]);
    let conv_start = input_count;
    let state_start = conv_start + conv_count;
    let mut conv_state = Some(Array::from_slice(
        &values[conv_start..state_start],
        &[1, 3, (3 * projection) as i32],
    ));
    let mut recurrent_state = Some(Array::from_slice(
        &values[state_start..],
        &[
            1,
            config.linear_num_heads as i32,
            config.linear_head_dim as i32,
            config.linear_head_dim as i32,
        ],
    ));
    let capture = capture::Capture {
        layers: vec![1],
        state_only: false,
        kda_prefill_only: false,
        entries: std::sync::Mutex::new(HashMap::new()),
        shapes: std::sync::Mutex::new(HashMap::new()),
        flushed: std::sync::atomic::AtomicBool::new(false),
        ..Default::default()
    };
    let _ = kda.forward(
        &input,
        &mut conv_state,
        &mut recurrent_state,
        Some(&capture),
        1,
    )?;

    let keys = [
        "L1.kda_qc",
        "L1.kda_kc",
        "L1.kda_vc",
        "L1.kda_g",
        "L1.kda_conv_state",
        "L1.fused_state",
        "L1.kda_outnorm",
    ];
    let entries = capture.entries.lock().expect("capture entries lock");
    let outputs = keys
        .iter()
        .map(|key| {
            entries
                .get(*key)
                .cloned()
                .ok_or_else(|| anyhow::anyhow!("missing DS4 comparison capture {key}"))
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    assert_projection_fixture(
        &crate::models::glm53::test_fixtures::read("kda_l1_mtp_reject_verify_ds4.bin"),
        &[
            "prepared q",
            "prepared k",
            "prepared v",
            "decay gate",
            "convolution cache",
            "recurrent cache",
            "gated normalized output",
        ],
        &outputs,
    )
}

#[test]
fn glm53_kda_layer21_decode_from_nonzero_cache_matches_direct_ds4_fixture() -> anyhow::Result<()> {
    if !lisa_mlx::device::metal_available() {
        eprintln!("[glm53-kda-ds4-decode] skipping: no Metal device available");
        return Ok(());
    }
    let Some(model_dir) = std::env::var_os("LISA_GLM53_MODEL") else {
        eprintln!("[glm53-kda-ds4-decode] skipping: LISA_GLM53_MODEL is unset");
        return Ok(());
    };
    let dir = std::path::Path::new(&model_dir);
    let config = super::super::config::ModelConfig::from_json(&dir.join("config.json"))?;
    anyhow::ensure!(config.is_kda(21), "GLM layer 21 is not a KDA layer");
    let mut weights = load_layer_kda_weights_for(dir, 21)?;
    let kda = Kda::load::<Weights>(
        &mut weights,
        "model.layers.21.self_attn",
        config.linear_num_heads,
        config.linear_head_dim,
        config.linear_conv_kernel_dim,
        config.linear_lower_bound,
        config.rms_norm_eps,
    )?;

    let read_f32 = |bytes: &[u8], offset: usize, count: usize| -> anyhow::Result<Vec<f32>> {
        let end = offset
            .checked_add(count * std::mem::size_of::<f32>())
            .ok_or_else(|| anyhow::anyhow!("fixture offset overflow"))?;
        anyhow::ensure!(end <= bytes.len(), "fixture slice exceeds its byte buffer");
        Ok(bytes[offset..end]
            .chunks_exact(std::mem::size_of::<f32>())
            .map(|value| f32::from_le_bytes(value.try_into().expect("four-byte float")))
            .collect())
    };
    let activation =
        &crate::models::glm53::test_fixtures::read("kda_l21_captured_input_ids1234.f32");
    let x_values = read_f32(activation, 0, config.hidden_size)?;
    let x = Array::from_slice(&x_values, &[1, 1, config.hidden_size as i32]);

    let prefill = &crate::models::glm53::test_fixtures::read("kda_l21_prefill_ds4_ids1234.f32");
    let projection = config.linear_num_heads * config.linear_head_dim;
    let conv_count = 9 * projection;
    let state_count = config.linear_num_heads * config.linear_head_dim * config.linear_head_dim;
    let conv_offset = 4 * 4 * projection * std::mem::size_of::<f32>();
    let state_offset = conv_offset + conv_count * std::mem::size_of::<f32>();
    let conv_values = read_f32(prefill, conv_offset, conv_count)?;
    let state_values = read_f32(prefill, state_offset, state_count)?;
    let mut conv_state = Some(Array::from_slice(
        &conv_values,
        &[1, 3, (3 * projection) as i32],
    ));
    let mut recurrent_state = Some(Array::from_slice(
        &state_values,
        &[
            1,
            config.linear_num_heads as i32,
            config.linear_head_dim as i32,
            config.linear_head_dim as i32,
        ],
    ));
    let capture = capture::Capture {
        layers: vec![21],
        state_only: false,
        kda_prefill_only: false,
        entries: std::sync::Mutex::new(HashMap::new()),
        shapes: std::sync::Mutex::new(HashMap::new()),
        flushed: std::sync::atomic::AtomicBool::new(false),
        ..Default::default()
    };
    let projected = kda.forward(
        &x,
        &mut conv_state,
        &mut recurrent_state,
        Some(&capture),
        21,
    )?;

    let keys = [
        "L21.kda_decode_out",
        "L21.kda_decode_conv_state",
        "L21.kda_decode_state",
    ];
    let entries = capture.entries.lock().expect("capture entries lock");
    let outputs = keys
        .iter()
        .map(|key| {
            entries
                .get(*key)
                .cloned()
                .ok_or_else(|| anyhow::anyhow!("missing DS4 decode capture {key}"))
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    assert_projection_fixture(
        &crate::models::glm53::test_fixtures::read("kda_l21_decode_ds4_ids1234.f32"),
        &["output", "conv state", "recurrent state"],
        &outputs,
    )?;
    assert_projection_fixture(
        &crate::models::glm53::test_fixtures::read("kda_l21_decode_output_proj_ds4_ids1234.f32"),
        &["decode output projection"],
        &[projected
            .reshape(&[config.hidden_size as i32])?
            .to_vec1::<f32>()?],
    )
}

#[test]
fn glm53_kda_layer21_nine_row_projections_match_direct_ds4_fixture() -> anyhow::Result<()> {
    if !lisa_mlx::device::metal_available() {
        eprintln!("[glm53-kda-ds4-prefill] skipping: no Metal device available");
        return Ok(());
    }
    let Some(model_dir) = std::env::var_os("LISA_GLM53_MODEL") else {
        eprintln!("[glm53-kda-ds4-prefill] skipping: LISA_GLM53_MODEL is unset");
        return Ok(());
    };
    let dir = std::path::Path::new(&model_dir);
    let config = super::super::config::ModelConfig::from_json(&dir.join("config.json"))?;
    anyhow::ensure!(config.is_kda(21), "GLM layer 21 is not a KDA layer");
    let mut weights = load_layer_kda_weights_for(dir, 21)?;
    let kda = Kda::load::<Weights>(
        &mut weights,
        "model.layers.21.self_attn",
        config.linear_num_heads,
        config.linear_head_dim,
        config.linear_conv_kernel_dim,
        config.linear_lower_bound,
        config.rms_norm_eps,
    )?;

    let rows = 9usize;
    let input_values: Vec<f32> = (0..rows * config.hidden_size)
        .map(|i| (((i * 37 + 13) % 251) as f32 - 125.0) * 0.0004)
        .collect();
    let input = Array::from_slice(&input_values, &[1, rows as i32, config.hidden_size as i32]);
    let (q, k, v, fa, ga, beta) = kda.input_projections(&input)?;
    let outputs = vec![
        q.to_vec1::<f32>()?,
        k.to_vec1::<f32>()?,
        v.to_vec1::<f32>()?,
        fa.to_vec1::<f32>()?,
        ga.to_vec1::<f32>()?,
        beta.to_vec1::<f32>()?,
    ];
    let expected = &crate::models::glm53::test_fixtures::read("kda_l21_inputproj_ds4_9.f32");
    assert_projection_fixture(&expected, &["q", "k", "v", "f_a", "g_a", "beta"], &outputs)?;
    Ok(())
}

#[test]
fn glm53_kda_projection_uses_eight_row_boundary_and_preserves_shape() -> anyhow::Result<()> {
    if !lisa_mlx::device::metal_available() {
        eprintln!("[glm53-kda-projection] skipping: no Metal device available");
        return Ok(());
    }
    let (k, n) = (323usize, 9usize);
    let weights_f32: Vec<f32> = (0..n * k)
        .map(|i| (((i * 7 + 3) % 53) as f32 - 26.0) * 0.003)
        .collect();
    let weights =
        Array::from_slice(&weights_f32, &[n as i32, k as i32]).as_dtype(Dtype::Bfloat16)?;
    let rounded_weights = weights.as_dtype(Dtype::Float32)?.to_vec1::<f32>()?;

    // Eight rows use ds4's row GEMV. Nine rows cross into its tiled GEMM,
    // which rounds both operands through F16 before F32 accumulation.
    for rows in [8usize, 9] {
        let x_values: Vec<f32> = (0..rows * k)
            .map(|i| (((i * 11 + 5) % 47) as f32 - 23.0) * 0.002)
            .collect();
        let x = Array::from_slice(&x_values, &[1, rows as i32, k as i32]);
        let got = linear_projection(&x, &weights)?;
        assert_eq!(got.shape(), [1, rows as i32, n as i32]);
        let got = got.to_vec1::<f32>()?;
        for row in 0..rows {
            for out in 0..n {
                let mut want = 0.0f64;
                for inner in 0..k {
                    let (weight, activation) = if rows > 8 {
                        (
                            half::f16::from_f32(rounded_weights[out * k + inner]).to_f32(),
                            half::f16::from_f32(x_values[row * k + inner]).to_f32(),
                        )
                    } else {
                        (rounded_weights[out * k + inner], x_values[row * k + inner])
                    };
                    want += f64::from(weight) * f64::from(activation);
                }
                assert!(
                    (f64::from(got[row * n + out]) - want).abs() < 2e-4,
                    "rows={rows} output=({row},{out}): {} vs {want}",
                    got[row * n + out]
                );
            }
        }
    }
    Ok(())
}

#[test]
fn glm53_kda_prefill_matches_eight_incremental_decode_steps() -> anyhow::Result<()> {
    if !lisa_mlx::device::metal_available() {
        eprintln!("[glm53-kda-cache] skipping: no Metal device available");
        return Ok(());
    }
    let Some(model_dir) = std::env::var_os("LISA_GLM53_MODEL") else {
        eprintln!("[glm53-kda-cache] skipping: LISA_GLM53_MODEL is unset");
        return Ok(());
    };
    let dir = std::path::Path::new(&model_dir);
    let config = super::super::config::ModelConfig::from_json(&dir.join("config.json"))?;
    anyhow::ensure!(config.is_kda(0), "GLM layer 0 is not a KDA layer");
    let weights = load_layer_kda_weights(dir)?;
    let mut prefill_weights = weights.clone();
    let mut decode_weights = weights;
    let prefill_kda = Kda::load::<Weights>(
        &mut prefill_weights,
        "model.layers.0.self_attn",
        config.linear_num_heads,
        config.linear_head_dim,
        config.linear_conv_kernel_dim,
        config.linear_lower_bound,
        config.rms_norm_eps,
    )?;
    let decode_kda = Kda::load::<Weights>(
        &mut decode_weights,
        "model.layers.0.self_attn",
        config.linear_num_heads,
        config.linear_head_dim,
        config.linear_conv_kernel_dim,
        config.linear_lower_bound,
        config.rms_norm_eps,
    )?;
    // Use identical BF16-projection inputs for one eight-token prefill and
    // eight one-token calls. This isolates KDA's state transition from the
    // surrounding 45-layer network and checks both returned activations
    // and persistent convolution/recurrent state.
    let (steps, hidden) = (8usize, config.hidden_size);
    let input_values: Vec<f32> = (0..steps * hidden)
        .map(|i| (((i * 37 + 13) % 251) as f32 - 125.0) * 0.0004)
        .collect();
    let input = Array::from_slice(&input_values, &[1, steps as i32, hidden as i32]);
    let mut prefill_conv = None;
    let mut prefill_ssm = None;
    let prefill_out = prefill_kda.forward(&input, &mut prefill_conv, &mut prefill_ssm, None, 0)?;
    let mut decode_conv = None;
    let mut decode_ssm = None;
    let mut decoded_rows = Vec::with_capacity(steps);
    for row in 0..steps {
        let token = input
            .index((.., row as i32..(row + 1) as i32, ..))
            .contiguous()?;
        decoded_rows.push(decode_kda.forward(
            &token,
            &mut decode_conv,
            &mut decode_ssm,
            None,
            0,
        )?);
    }
    let expected = prefill_out.to_vec1::<f32>()?;
    let mut actual = Vec::with_capacity(expected.len());
    for row in decoded_rows {
        actual.extend(row.to_vec1::<f32>()?);
    }
    let output_error = expected
        .iter()
        .zip(&actual)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    anyhow::ensure!(
        output_error <= 2e-4,
        "GLM KDA prefill/decode output max abs error is {output_error}"
    );
    for (name, prefill, decode) in [
        ("conv", prefill_conv, decode_conv),
        ("ssm", prefill_ssm, decode_ssm),
    ] {
        let (Some(prefill), Some(decode)) = (prefill, decode) else {
            anyhow::bail!("GLM KDA {name} state was not initialized");
        };
        let a = prefill.to_vec1::<f32>()?;
        let b = decode.to_vec1::<f32>()?;
        let error = a
            .iter()
            .zip(&b)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        anyhow::ensure!(error <= 2e-4, "GLM KDA {name} state error is {error}");
    }
    Ok(())
}

#[test]
fn glm53_kda_decode_captures_real_nonzero_state_fixture() -> anyhow::Result<()> {
    if !lisa_mlx::device::metal_available() {
        eprintln!("[glm53-kda-decode] skipping: no Metal device available");
        return Ok(());
    }
    let Some(model_dir) = std::env::var_os("LISA_GLM53_MODEL") else {
        eprintln!("[glm53-kda-decode] skipping: LISA_GLM53_MODEL is unset");
        return Ok(());
    };
    let Some(fixture_dir) = std::env::var_os("LISA_GLM53_KDA_DECODE_FIXTURE") else {
        eprintln!("[glm53-kda-decode] skipping: real decode fixture is unset");
        return Ok(());
    };
    let fixture_dir = std::path::Path::new(&fixture_dir);
    let read_f32 = |name: &str, count: usize| -> anyhow::Result<Vec<f32>> {
        let bytes = std::fs::read(fixture_dir.join(name))?;
        anyhow::ensure!(
            bytes.len() == count * std::mem::size_of::<f32>(),
            "{name} has {} bytes, expected {}",
            bytes.len(),
            count * std::mem::size_of::<f32>()
        );
        Ok(bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("four-byte float")))
            .collect())
    };

    let dir = std::path::Path::new(&model_dir);
    let config = super::super::config::ModelConfig::from_json(&dir.join("config.json"))?;
    anyhow::ensure!(config.is_kda(0), "GLM layer 0 is not a KDA layer");
    let mut weights = load_layer_kda_weights(dir)?;
    let kda = Kda::load::<Weights>(
        &mut weights,
        "model.layers.0.self_attn",
        config.linear_num_heads,
        config.linear_head_dim,
        config.linear_conv_kernel_dim,
        config.linear_lower_bound,
        config.rms_norm_eps,
    )?;

    let hidden = config.hidden_size;
    let heads = config.linear_num_heads;
    let dim = config.linear_head_dim;
    let conv_width = heads * dim * 3;
    let input = Array::from_slice(&read_f32("input.f32", hidden)?, &[1, 1, hidden as i32]);
    let conv = Array::from_slice(
        &read_f32("conv_state.f32", 3 * conv_width)?,
        &[1, 3, conv_width as i32],
    );
    let state = Array::from_slice(
        &read_f32("recurrent_state.f32", heads * dim * dim)?,
        &[1, heads as i32, dim as i32, dim as i32],
    );
    let mut conv_state = Some(conv);
    let mut ssm_state = Some(state);
    let stages = capture::Capture {
        layers: vec![0],
        state_only: false,
        kda_prefill_only: false,
        entries: std::sync::Mutex::new(std::collections::HashMap::new()),
        shapes: std::sync::Mutex::new(std::collections::HashMap::new()),
        flushed: std::sync::atomic::AtomicBool::new(false),
        ..Default::default()
    };
    let output = kda.forward(&input, &mut conv_state, &mut ssm_state, Some(&stages), 0)?;
    let _ = output.to_vec1::<f32>()?;
    let conv_state = conv_state.ok_or_else(|| anyhow::anyhow!("KDA decode lost conv state"))?;
    let ssm_state = ssm_state.ok_or_else(|| anyhow::anyhow!("KDA decode lost recurrent state"))?;
    stages.store("L0.kda_decode_conv_state", &conv_state);
    stages.store("L0.kda_decode_final_state", &ssm_state);
    stages.flush(
        fixture_dir
            .join("lisa_decode.npz")
            .to_str()
            .ok_or_else(|| anyhow::anyhow!("decode fixture path is not UTF-8"))?,
    )?;
    Ok(())
}
