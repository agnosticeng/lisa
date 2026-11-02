use super::*;
use crate::core::loader::{Checkpoint, Shard, sanitize_name};
use crate::core::quant::get_tensor;
use lisa_mlx::ops::indexing::IndexOp;
use std::collections::HashMap;

fn load_layer_dsa_weights_at(dir: &std::path::Path, layer: usize) -> anyhow::Result<Weights> {
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

fn load_layer_dsa_weights(dir: &std::path::Path) -> anyhow::Result<Weights> {
    load_layer_dsa_weights_at(dir, 3)
}

fn model_long_dsa_fixture(layer: usize) -> anyhow::Result<Vec<u8>> {
    let fixture = match layer {
        3 => crate::models::glm53::test_fixtures::read("dsa_attention_layer3_model_long_ds4.bin"),
        7 => crate::models::glm53::test_fixtures::read("dsa_attention_layer7_model_long_ds4.bin"),
        11 => crate::models::glm53::test_fixtures::read("dsa_attention_layer11_model_long_ds4.bin"),
        15 => crate::models::glm53::test_fixtures::read("dsa_attention_layer15_model_long_ds4.bin"),
        19 => crate::models::glm53::test_fixtures::read("dsa_attention_layer19_model_long_ds4.bin"),
        23 => crate::models::glm53::test_fixtures::read("dsa_attention_layer23_model_long_ds4.bin"),
        27 => crate::models::glm53::test_fixtures::read("dsa_attention_layer27_model_long_ds4.bin"),
        31 => crate::models::glm53::test_fixtures::read("dsa_attention_layer31_model_long_ds4.bin"),
        35 => crate::models::glm53::test_fixtures::read("dsa_attention_layer35_model_long_ds4.bin"),
        39 => crate::models::glm53::test_fixtures::read("dsa_attention_layer39_model_long_ds4.bin"),
        43 => crate::models::glm53::test_fixtures::read("dsa_attention_layer43_model_long_ds4.bin"),
        _ => anyhow::bail!("layer {layer} has no captured long-context DSA fixture"),
    };
    Ok(fixture)
}

fn model_long_dsa_production_fixture(layer: usize) -> anyhow::Result<Vec<u8>> {
    let fixture = match layer {
        3 => crate::models::glm53::test_fixtures::read(
            "dsa_production_decode_layer3_model_long_ds4.bin",
        ),
        7 => crate::models::glm53::test_fixtures::read(
            "dsa_production_decode_layer7_model_long_ds4.bin",
        ),
        11 => crate::models::glm53::test_fixtures::read(
            "dsa_production_decode_layer11_model_long_ds4.bin",
        ),
        15 => crate::models::glm53::test_fixtures::read(
            "dsa_production_decode_layer15_model_long_ds4.bin",
        ),
        19 => crate::models::glm53::test_fixtures::read(
            "dsa_production_decode_layer19_model_long_ds4.bin",
        ),
        23 => crate::models::glm53::test_fixtures::read(
            "dsa_production_decode_layer23_model_long_ds4.bin",
        ),
        27 => crate::models::glm53::test_fixtures::read(
            "dsa_production_decode_layer27_model_long_ds4.bin",
        ),
        31 => crate::models::glm53::test_fixtures::read(
            "dsa_production_decode_layer31_model_long_ds4.bin",
        ),
        35 => crate::models::glm53::test_fixtures::read(
            "dsa_production_decode_layer35_model_long_ds4.bin",
        ),
        39 => crate::models::glm53::test_fixtures::read(
            "dsa_production_decode_layer39_model_long_ds4.bin",
        ),
        43 => crate::models::glm53::test_fixtures::read(
            "dsa_production_decode_layer43_model_long_ds4.bin",
        ),
        _ => anyhow::bail!("layer {layer} has no production DSA fixture"),
    };
    Ok(fixture)
}

fn decode_fixture_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("fixture F32 word")))
        .collect()
}

#[test]
fn glm53_dsa_serial_row_batch_projection_matches_real_checkpoint_rows() -> anyhow::Result<()> {
    if !lisa_mlx::device::metal_available() {
        eprintln!("[glm53-dsa-row-projections] skipping: no Metal device available");
        return Ok(());
    }
    let Some(model_dir) = std::env::var_os("LISA_GLM53_MODEL") else {
        eprintln!("[glm53-dsa-row-projections] skipping: LISA_GLM53_MODEL is unset");
        return Ok(());
    };
    let dir = std::path::Path::new(&model_dir);
    let config = super::super::config::ModelConfig::from_json(&dir.join("config.json"))?;
    let mut weights = load_layer_dsa_weights(dir)?;
    let dsa = Dsa::load::<Weights>(
        &mut weights,
        "model.layers.3.self_attn",
        config.num_attention_heads,
        config.qk_nope_head_dim,
        config.v_head_dim,
        config.index_n_heads,
        config.index_head_dim,
        config.index_topk,
        config.index_kpool,
        config.index_kpool_always_select_tail,
        config.rms_norm_eps,
    )?;

    // The row-batched qmv kernel is intended to retain serial arithmetic
    // while sharing one dispatch. Exercise the actual layer-3 packed
    // checkpoint projections, then compare every produced stage with an
    // independent one-row forward that uses DS4's serial DSA path.
    let values: Vec<f32> = (0..2 * 4096)
        .map(|i| (((i * 37 + 11) % 251) as f32 - 125.0) * 0.0003)
        .collect();
    let x = Array::from_slice(&values, &[2, 1, 4096]);
    let batched = dsa.project_inputs(&x, None, 3, None)?;
    for row in 0..2 {
        let x_row = x.index((row..row + 1, ..));
        let serial = dsa.project_inputs(&x_row, None, 3, None)?;
        let compare = |name: &str, actual: &Array, expected: &Array| -> anyhow::Result<()> {
            // The serial decode kernel returns the compact latent in its
            // cache dtype, while the multirow path returns the normalized
            // F32 value before cache append. Compare both at the DS4 F16
            // cache boundary, where they become observable to attention.
            let dtype = if name == "compact latent" {
                Dtype::Float16
            } else {
                Dtype::Float32
            };
            let actual = actual.index((row..row + 1, ..)).as_dtype(dtype)?;
            let expected = expected.as_dtype(dtype)?;
            let matches = if dtype == Dtype::Float16 {
                let actual = actual.to_vec1::<half::f16>()?;
                let expected = expected.to_vec1::<half::f16>()?;
                actual
                    .iter()
                    .map(|v| v.to_bits())
                    .eq(expected.iter().map(|v| v.to_bits()))
            } else {
                let actual = actual.to_vec1::<f32>()?;
                let expected = expected.to_vec1::<f32>()?;
                actual
                    .iter()
                    .map(|v| v.to_bits())
                    .eq(expected.iter().map(|v| v.to_bits()))
            };
            anyhow::ensure!(
                matches,
                "layer-3 row-batched DSA {name} differs from serial row {row}"
            );
            Ok(())
        };
        compare(
            "compact latent",
            batched.latent.as_ref().expect("latent is present"),
            serial.latent.as_ref().expect("serial latent is present"),
        )?;
        compare("query rank", &batched.query_rank, &serial.query_rank)?;
        compare("query", &batched.query, &serial.query)?;
        compare("indexer key", &batched.indexer_key, &serial.indexer_key)?;
        compare("indexer gate", &batched.indexer_gate, &serial.indexer_gate)?;
    }

    // Exercise the production decode kernel with capture enabled. Capture
    // must not alter the projection route or its DS4 arithmetic schedule.
    let make_capture = || crate::core::capture::Capture {
        layers: vec![3],
        state_only: false,
        kda_prefill_only: false,
        entries: std::sync::Mutex::new(HashMap::new()),
        shapes: std::sync::Mutex::new(HashMap::new()),
        flushed: std::sync::atomic::AtomicBool::new(false),
        ..Default::default()
    };
    let batch_capture = make_capture();
    let captured_batch = dsa.project_inputs(&x, Some(&batch_capture), 3, None)?;
    let compare_captured = |name: &str, actual: &Array, expected: &Array| -> anyhow::Result<()> {
        let actual = actual.as_dtype(Dtype::Float32)?.to_vec1::<f32>()?;
        let expected = expected.as_dtype(Dtype::Float32)?.to_vec1::<f32>()?;
        anyhow::ensure!(
            actual
                .iter()
                .map(|value| value.to_bits())
                .eq(expected.iter().map(|value| value.to_bits())),
            "layer-3 captured row-batched DSA {name} differs from serial"
        );
        Ok(())
    };
    for row in 0..2 {
        let x_row = x.index((row..row + 1, ..));
        let serial_capture = make_capture();
        let captured_serial = dsa.project_inputs(&x_row, Some(&serial_capture), 3, None)?;
        compare_captured(
            "kv raw",
            &captured_batch.kv_raw.index((row..row + 1, ..)),
            &captured_serial.kv_raw,
        )?;
        compare_captured(
            "normalized latent",
            &captured_batch
                .latent
                .as_ref()
                .expect("staged latent is present")
                .index((row..row + 1, ..)),
            captured_serial
                .latent
                .as_ref()
                .expect("staged serial latent is present"),
        )?;
    }
    Ok(())
}

#[test]
fn glm53_dsa_value_projection_matches_ds4_fixture_on_real_weights() -> anyhow::Result<()> {
    if !lisa_mlx::device::metal_available() {
        eprintln!("[glm53-dsa-value] skipping: no Metal device available");
        return Ok(());
    }
    let Some(model_dir) = std::env::var_os("LISA_GLM53_MODEL") else {
        eprintln!("[glm53-dsa-value] skipping: LISA_GLM53_MODEL is unset");
        return Ok(());
    };
    let dir = std::path::Path::new(&model_dir);
    let config = super::super::config::ModelConfig::from_json(&dir.join("config.json"))?;
    let mut weights = load_layer_dsa_weights(dir)?;
    let dsa = Dsa::load::<Weights>(
        &mut weights,
        "model.layers.3.self_attn",
        config.num_attention_heads,
        config.qk_nope_head_dim,
        config.v_head_dim,
        config.index_n_heads,
        config.index_head_dim,
        config.index_topk,
        config.index_kpool,
        config.index_kpool_always_select_tail,
        config.rms_norm_eps,
    )?;

    // Feed a deterministic compact cache through the production QK and
    // online indexed-lora stages, then compare the value half against a
    // direct DS4 F32 matvec using the same dequantized MLX checkpoint rows.
    let (heads, tokens) = (config.num_attention_heads, 5usize);
    let q_values: Vec<f32> = (0..heads * 256)
        .map(|i| (((i * 31 + 7) % 127) as f32 - 63.0) * 0.0008)
        .collect();
    let latent_values: Vec<half::f16> = (0..tokens * 512)
        .map(|i| half::f16::from_f32((((i * 17 + 3) % 131) as f32 - 65.0) * 0.001))
        .collect();
    let selected_values: Vec<i32> = (0..tokens as i32).collect();
    let q = Array::from_slice(&q_values, &[1, heads as i32, 1, 256]);
    let latents = Array::from_slice(&latent_values, &[1, tokens as i32, 512]);
    let selected = Array::from_slice(&selected_values, &[1, 1, tokens as i32]);
    let stream = lisa_mlx::Stream::thread_local_or_default();
    let (actual, q_low, lora) = lisa_mlx::models::glm53::kernels::dsa_absorbed_attention_stages(
        &q,
        &dsa.kv_b.weight,
        &dsa.kv_b.scales,
        &dsa.kv_b.biases,
        dsa.kv_b.group_size as usize,
        &latents,
        &selected,
        (config.qk_nope_head_dim as f64).powf(-0.5) as f32,
        &stream,
        false,
    )
    .ok_or_else(|| anyhow::anyhow!("real-weight DSA value path declined"))?;
    let actual = actual.to_vec1::<f32>()?;
    let q_low = q_low.to_vec1::<f32>()?;
    let lora = lora.to_vec1::<f32>()?;
    let pre_o = Array::from_slice(&actual, &[1, 1, (heads * 256) as i32]);
    let lisa_o = dsa.output_project(&pre_o)?.to_vec1::<f32>()?;
    let loaded_o_project_fixture: &[u8] =
        &crate::models::glm53::test_fixtures::read("dsa_o_project_layer3_ds4.bin");
    const O_INPUT_BYTES: usize = 16_384 * std::mem::size_of::<f32>();
    const O_OUTPUT_BYTES: usize = 4_096 * std::mem::size_of::<f32>();
    anyhow::ensure!(loaded_o_project_fixture.len() == O_INPUT_BYTES + O_OUTPUT_BYTES);
    let o_fixture_f32 = |bytes: &[u8]| {
        bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("F32 fixture word")))
            .collect::<Vec<_>>()
    };
    let expected_o_input = o_fixture_f32(&loaded_o_project_fixture[..O_INPUT_BYTES]);
    let expected_o_output = o_fixture_f32(&loaded_o_project_fixture[O_INPUT_BYTES..]);
    anyhow::ensure!(
        actual
            .iter()
            .map(|x| x.to_bits())
            .eq(expected_o_input.iter().map(|x| x.to_bits())),
        "layer-3 DSA output-projection input changed from DS4 fixture"
    );
    anyhow::ensure!(
        lisa_o
            .iter()
            .map(|x| x.to_bits())
            .eq(expected_o_output.iter().map(|x| x.to_bits())),
        "real-weight DSA output projection differs from direct DS4 F32 matvec fixture"
    );
    const O_MULTI_ROWS: usize = 9;
    let multi_input: Vec<f32> = (0..O_MULTI_ROWS * actual.len())
        .map(|i| {
            let row = i / actual.len();
            let col = i % actual.len();
            actual[(col + row * 131) % actual.len()] * (0.75 + row as f32 * 0.01)
                + row as f32 * 0.0001
        })
        .collect();
    let multi_array = Array::from_slice(&multi_input, &[1, O_MULTI_ROWS as i32, 16_384]);
    let multi_output = dsa.output_project(&multi_array)?.to_vec1::<f32>()?;
    let mut small_batches = Vec::new();
    for rows in 2..=8usize {
        let input: Vec<f32> = (0..rows * actual.len())
            .map(|i| {
                let row = i / actual.len();
                let col = i % actual.len();
                actual[(col + row * 131) % actual.len()] * (0.75 + row as f32 * 0.01)
                    + row as f32 * 0.0001
            })
            .collect();
        let array = Array::from_slice(&input, &[1, rows as i32, 16_384]);
        let output = dsa.output_project(&array)?.to_vec1::<f32>()?;
        small_batches.push((rows, input, output));
    }
    if std::env::var_os("LISA_GLM53_DSA_VPROJ_CAPTURE_DIR").is_none() {
        let loaded_o_multi_fixture: &[u8] =
            &crate::models::glm53::test_fixtures::read("dsa_o_project_rows9_layer3_ds4.bin");
        const O_MULTI_INPUT_BYTES: usize = O_MULTI_ROWS * 16_384 * std::mem::size_of::<f32>();
        const O_MULTI_OUTPUT_BYTES: usize = O_MULTI_ROWS * 4_096 * std::mem::size_of::<f32>();
        anyhow::ensure!(loaded_o_multi_fixture.len() == O_MULTI_INPUT_BYTES + O_MULTI_OUTPUT_BYTES);
        let expected_multi_input = o_fixture_f32(&loaded_o_multi_fixture[..O_MULTI_INPUT_BYTES]);
        let expected_multi_output = o_fixture_f32(&loaded_o_multi_fixture[O_MULTI_INPUT_BYTES..]);
        anyhow::ensure!(
            multi_input
                .iter()
                .map(|x| x.to_bits())
                .eq(expected_multi_input.iter().map(|x| x.to_bits())),
            "nine-row DSA output-projection input changed from DS4 fixture"
        );
        anyhow::ensure!(
            multi_output
                .iter()
                .map(|x| x.to_bits())
                .eq(expected_multi_output.iter().map(|x| x.to_bits())),
            "nine-row DSA output projection differs from direct DS4 F32 matmul fixture"
        );
        let loaded_o_small_fixture: &[u8] =
            &crate::models::glm53::test_fixtures::read("dsa_o_project_rows2to8_layer3_ds4.bin");
        let mut offset = 0usize;
        for (rows, input, output) in &small_batches {
            let input_bytes = rows * 16_384 * std::mem::size_of::<f32>();
            let output_bytes = rows * 4_096 * std::mem::size_of::<f32>();
            anyhow::ensure!(offset + input_bytes + output_bytes <= loaded_o_small_fixture.len());
            let expected_input =
                o_fixture_f32(&loaded_o_small_fixture[offset..offset + input_bytes]);
            let output_offset = offset + input_bytes;
            let expected_output =
                o_fixture_f32(&loaded_o_small_fixture[output_offset..output_offset + output_bytes]);
            anyhow::ensure!(
                input
                    .iter()
                    .map(|x| x.to_bits())
                    .eq(expected_input.iter().map(|x| x.to_bits())),
                "{rows}-row DSA output-projection input changed from DS4 fixture"
            );
            anyhow::ensure!(
                output
                    .iter()
                    .map(|x| x.to_bits())
                    .eq(expected_output.iter().map(|x| x.to_bits())),
                "{rows}-row DSA output projection differs from direct DS4 mv_ext fixture"
            );
            offset += input_bytes + output_bytes;
        }
        anyhow::ensure!(offset == loaded_o_small_fixture.len());
    }
    if let Some(path) = std::env::var_os("LISA_GLM53_DSA_VPROJ_CAPTURE_DIR") {
        let path = std::path::Path::new(&path);
        std::fs::create_dir_all(path)?;
        let write_f32 = |name: &str, values: &[f32]| -> anyhow::Result<()> {
            let bytes: Vec<u8> = values
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect();
            std::fs::write(path.join(name), bytes)?;
            Ok(())
        };
        let write_u32 = |name: &str, values: &[u32]| -> anyhow::Result<()> {
            let bytes: Vec<u8> = values
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect();
            std::fs::write(path.join(name), bytes)?;
            Ok(())
        };
        write_f32("lora.f32", &lora)?;
        write_f32("q.f32", &q_values)?;
        write_f32("q_low_lisa.f32", &q_low)?;
        write_f32("attended_lora_lisa.f32", &lora)?;
        write_f32("value_output_lisa.f32", &actual)?;
        let compact_bytes: Vec<u8> = latent_values
            .iter()
            .flat_map(|value| value.to_bits().to_le_bytes())
            .collect();
        std::fs::write(path.join("compact_cache.f16"), compact_bytes)?;
        let selected_bytes: Vec<u8> = selected_values
            .iter()
            .flat_map(|value| (*value as u32).to_le_bytes())
            .collect();
        std::fs::write(path.join("selected.u32"), selected_bytes)?;
        write_u32("weight.u32", &dsa.kv_b.weight.to_vec1::<u32>()?)?;
        write_f32("scales.f32", &dsa.kv_b.scales.to_vec1::<f32>()?)?;
        write_f32("biases.f32", &dsa.kv_b.biases.to_vec1::<f32>()?)?;
        write_f32("o_proj_input.f32", &actual)?;
        write_f32("o_proj_lisa_output.f32", &lisa_o)?;
        write_f32("o_proj_multi_input.f32", &multi_input)?;
        write_f32("o_proj_multi_lisa_output.f32", &multi_output)?;
        for (rows, input, output) in &small_batches {
            write_f32(&format!("o_proj_rows{rows}_input.f32"), input)?;
            write_f32(&format!("o_proj_rows{rows}_lisa_output.f32"), output)?;
        }
        write_u32("o_proj_weight.u32", &dsa.o_proj.weight.to_vec1::<u32>()?)?;
        write_f32("o_proj_scales.f32", &dsa.o_proj.scales.to_vec1::<f32>()?)?;
        write_f32("o_proj_biases.f32", &dsa.o_proj.biases.to_vec1::<f32>()?)?;
    }
    let loaded_fixture: &[u8] =
        &crate::models::glm53::test_fixtures::read("dsa_value_project_layer3_ds4.bin");
    const LORA_BYTES: usize = 64 * 512 * std::mem::size_of::<f32>();
    const OUTPUT_BYTES: usize = 64 * 256 * std::mem::size_of::<f32>();
    anyhow::ensure!(loaded_fixture.len() == LORA_BYTES + OUTPUT_BYTES);
    let fixture_f32 = |bytes: &[u8]| {
        bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("F32 fixture word")))
            .collect::<Vec<_>>()
    };
    let expected_lora = fixture_f32(&loaded_fixture[..LORA_BYTES]);
    let expected_output = fixture_f32(&loaded_fixture[LORA_BYTES..]);
    anyhow::ensure!(
        lora.iter()
            .map(|x| x.to_bits())
            .eq(expected_lora.iter().map(|x| x.to_bits())),
        "layer-3 indexed-lora input changed from DS4 value fixture"
    );
    anyhow::ensure!(
        actual
            .iter()
            .map(|x| x.to_bits())
            .eq(expected_output.iter().map(|x| x.to_bits())),
        "real-weight DSA value projection differs from direct DS4 F32 matvec fixture"
    );
    let loaded_absorbed_fixture: &[u8] =
        &crate::models::glm53::test_fixtures::read("dsa_absorbed_layer3_ds4.bin");
    const Q_LOW_BYTES: usize = 64 * 512 * std::mem::size_of::<f32>();
    const ABSORBED_LORA_BYTES: usize = 64 * 512 * std::mem::size_of::<f32>();
    const VALUE_BYTES: usize = 64 * 256 * std::mem::size_of::<f32>();
    anyhow::ensure!(
        loaded_absorbed_fixture.len() == Q_LOW_BYTES + ABSORBED_LORA_BYTES + VALUE_BYTES,
        "DS4 absorbed-attention fixture has an invalid size"
    );
    let ds4_q_low = fixture_f32(&loaded_absorbed_fixture[..Q_LOW_BYTES]);
    let lora_start = Q_LOW_BYTES;
    let ds4_lora =
        fixture_f32(&loaded_absorbed_fixture[lora_start..lora_start + ABSORBED_LORA_BYTES]);
    let value_start = lora_start + ABSORBED_LORA_BYTES;
    let ds4_value = fixture_f32(&loaded_absorbed_fixture[value_start..]);
    let q_low_max_abs = q_low
        .iter()
        .zip(&ds4_q_low)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0f32, f32::max);
    let q_low_ref_norm = ds4_q_low.iter().map(|x| x * x).sum::<f32>().sqrt();
    let q_low_diff_norm = q_low
        .iter()
        .zip(&ds4_q_low)
        .map(|(actual, expected)| (actual - expected).powi(2))
        .sum::<f32>()
        .sqrt();
    anyhow::ensure!(
        q_low_max_abs <= 2e-6 && q_low_diff_norm <= q_low_ref_norm * 1e-5,
        "DS4 QK-lowrank projection differs: max_abs={q_low_max_abs:e}, rel_l2={:e}",
        q_low_diff_norm / (q_low_ref_norm + f32::MIN_POSITIVE)
    );
    anyhow::ensure!(
        lora.iter()
            .map(|x| x.to_bits())
            .eq(ds4_lora.iter().map(|x| x.to_bits())),
        "real-weight DSA indexed-lora output differs from direct DS4 Metal"
    );
    anyhow::ensure!(
        actual
            .iter()
            .map(|x| x.to_bits())
            .eq(ds4_value.iter().map(|x| x.to_bits())),
        "real-weight DSA value output differs from direct DS4 Metal"
    );
    eprintln!(
        "[glm53-dsa-absorbed] q_low max_abs={q_low_max_abs:.3e} rel_l2={:.3e}; indexed-lora and value outputs bit-identical",
        q_low_diff_norm / (q_low_ref_norm + f32::MIN_POSITIVE)
    );
    let wide_tokens = 2051usize;
    let wide_latent_values: Vec<half::f16> = (0..wide_tokens * 512)
        .map(|i| half::f16::from_f32((((i * 23 + 11) % 127) as f32 - 63.0) * 0.0007))
        .collect();
    let wide_selected_ids: Vec<i32> = (0..wide_tokens as i32).collect();
    let wide_latents = Array::from_slice(&wide_latent_values, &[1, wide_tokens as i32, 512]);
    let wide_selected = Array::from_slice(&wide_selected_ids, &[1, 1, wide_tokens as i32]);
    let wide_fused = lisa_mlx::models::glm53::kernels::dsa_absorbed_attention(
        &q,
        &dsa.kv_b.weight,
        &dsa.kv_b.scales,
        &dsa.kv_b.biases,
        dsa.kv_b.group_size as usize,
        &wide_latents,
        &wide_selected,
        (config.qk_nope_head_dim as f64).powf(-0.5) as f32,
        &stream,
    )
    .ok_or_else(|| anyhow::anyhow!("real-weight fused DSA decode declined"))?
    .to_vec1::<f32>()?;
    let (wide_staged, wide_q_low, _) =
        lisa_mlx::models::glm53::kernels::dsa_absorbed_attention_stages(
            &q,
            &dsa.kv_b.weight,
            &dsa.kv_b.scales,
            &dsa.kv_b.biases,
            dsa.kv_b.group_size as usize,
            &wide_latents,
            &wide_selected,
            (config.qk_nope_head_dim as f64).powf(-0.5) as f32,
            &stream,
            false,
        )
        .ok_or_else(|| anyhow::anyhow!("real-weight staged DSA decode declined"))?;
    let wide_staged = wide_staged.to_vec1::<f32>()?;
    let wide_q_low = wide_q_low.to_vec1::<f32>()?;
    let wide_selected_u32: Vec<u32> = wide_selected_ids
        .iter()
        .map(|&value| value as u32)
        .collect();
    let wide_max_abs = wide_fused
        .iter()
        .zip(&wide_staged)
        .map(|(fused, staged)| (fused - staged).abs())
        .fold(0.0f32, f32::max);
    anyhow::ensure!(
        wide_max_abs <= 2e-4,
        "real-weight fused DSA decode differs from staged reference: max_abs={wide_max_abs:e}"
    );
    let loaded_production_decode_fixture: &[u8] =
        &crate::models::glm53::test_fixtures::read("dsa_production_decode_layer3_ds4.bin");
    anyhow::ensure!(
        loaded_production_decode_fixture.len() == 64 * 256 * std::mem::size_of::<f32>(),
        "DS4 production DSA decode fixture has an invalid size"
    );
    let ds4_production_decode = fixture_f32(loaded_production_decode_fixture);
    anyhow::ensure!(
        wide_fused
            .iter()
            .map(|value| value.to_bits())
            .eq(ds4_production_decode.iter().map(|value| value.to_bits())),
        "real-weight fused DSA decode differs from DS4 production kernel"
    );
    eprintln!(
        "[glm53-dsa-production-decode] 2,051 selected rows match DS4 production kernel bit-for-bit"
    );
    if let Some(path) = std::env::var_os("LISA_GLM53_DSA_PRODUCTION_CAPTURE_DIR") {
        let path = std::path::Path::new(&path);
        std::fs::create_dir_all(path)?;
        let write_f32 = |name: &str, values: &[f32]| -> anyhow::Result<()> {
            let bytes: Vec<u8> = values
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect();
            std::fs::write(path.join(name), bytes)?;
            Ok(())
        };
        let write_u32 = |name: &str, values: &[u32]| -> anyhow::Result<()> {
            let bytes: Vec<u8> = values
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect();
            std::fs::write(path.join(name), bytes)?;
            Ok(())
        };
        write_f32("wide_q.f32", &q_values)?;
        write_f32("wide_q_low.f32", &wide_q_low)?;
        write_f32("wide_lisa_output.f32", &wide_fused)?;
        write_u32("wide_selected.u32", &wide_selected_u32)?;
        let latent_bytes: Vec<u8> = wide_latent_values
            .iter()
            .flat_map(|value| value.to_bits().to_le_bytes())
            .collect();
        std::fs::write(path.join("wide_latents.f16"), latent_bytes)?;
        write_u32("weight.u32", &dsa.kv_b.weight.to_vec1::<u32>()?)?;
        write_f32("scales.f32", &dsa.kv_b.scales.to_vec1::<f32>()?)?;
        write_f32("biases.f32", &dsa.kv_b.biases.to_vec1::<f32>()?)?;
    }
    eprintln!(
        "[glm53-dsa-fused] 2,051 selected rows with real kv_b weights; staged-path max_abs={wide_max_abs:.3e}"
    );
    // Supplying the same three invalid padding ids explicitly must be
    // identical to the wrapper's automatic short-buffer padding.
    let mut padded_ids = selected_values.clone();
    padded_ids.extend([-1, -1, -1]);
    let explicit_selected = Array::from_slice(&padded_ids, &[1, 1, 8]);
    let (explicit_out, _, explicit_lora) =
        lisa_mlx::models::glm53::kernels::dsa_absorbed_attention_stages(
            &q,
            &dsa.kv_b.weight,
            &dsa.kv_b.scales,
            &dsa.kv_b.biases,
            dsa.kv_b.group_size as usize,
            &latents,
            &explicit_selected,
            (config.qk_nope_head_dim as f64).powf(-0.5) as f32,
            &stream,
            false,
        )
        .ok_or_else(|| anyhow::anyhow!("explicitly padded DSA value path declined"))?;
    let explicit_out = explicit_out.to_vec1::<f32>()?;
    let explicit_lora = explicit_lora.to_vec1::<f32>()?;
    anyhow::ensure!(
        actual
            .iter()
            .map(|x| x.to_bits())
            .eq(explicit_out.iter().map(|x| x.to_bits()))
            && lora
                .iter()
                .map(|x| x.to_bits())
                .eq(explicit_lora.iter().map(|x| x.to_bits())),
        "automatic short-window padding changed indexed attention output"
    );

    let packed = dsa.kv_b.weight.to_vec1::<u32>()?;
    let scales = dsa.kv_b.scales.to_vec1::<f32>()?;
    let biases = dsa.kv_b.biases.to_vec1::<f32>()?;
    let group_size = dsa.kv_b.group_size as usize;
    let groups = 512 / group_size;
    let mut max_abs = 0.0f32;
    for head in 0..heads {
        for d in 0..256 {
            let row = head * 512 + 256 + d;
            let mut expected = 0.0f32;
            for j in 0..512 {
                let word = packed[row * 128 + j / 4];
                let code = ((word >> ((j & 3) * 8)) & 0xff) as f32;
                let group = row * groups + j / group_size;
                let weight = code.mul_add(scales[group], biases[group]);
                expected = lora[head * 512 + j].mul_add(weight, expected);
            }
            let got = actual[head * 256 + d];
            max_abs = max_abs.max((got - expected).abs());
        }
    }
    anyhow::ensure!(
        max_abs <= 2e-8,
        "real-weight DSA value projection exceeds DS4 reduction-order tolerance: max_abs={max_abs:e}"
    );
    eprintln!("[glm53-dsa-value] direct DS4 fixture exact; scalar-FMA max_abs={max_abs:e}");

    // Exercise the actual pooled selector just beyond its dense crossover.
    // At one-token decode every complete pool is live; only the reserved
    // incomplete-tail slots may contain the DS4 UINT32_MAX sentinel.
    let kpool = dsa.indexer.kpool;
    let selector_tokens = dsa.indexer.topk + kpool + 1;
    let complete_pools = selector_tokens / kpool;
    let head_dim = dsa.indexer.head_dim;
    let raw_key = Array::from_slice(
        &vec![0.0f32; selector_tokens * head_dim],
        &[1, selector_tokens as i32, head_dim as i32],
    );
    let raw_gate = Array::from_slice(
        &vec![0.0f32; selector_tokens * head_dim],
        &[1, selector_tokens as i32, head_dim as i32],
    );
    let pooled_values: Vec<half::f16> = (0..complete_pools * head_dim)
        .map(|i| half::f16::from_f32((((i * 19 + 7) % 127) as f32 - 63.0) * 0.0002))
        .collect();
    let pooled = Array::from_slice(&pooled_values, &[1, complete_pools as i32, head_dim as i32]);
    let mut selector_tape = IndexerTape::new();
    selector_tape.append_fields(&raw_key, &raw_gate, kpool)?;
    selector_tape.push_pooled_f16(&pooled)?;
    let selector_input_values: Vec<f32> = (0..config.hidden_size)
        .map(|i| (((i * 29 + 11) % 251) as f32 - 125.0) * 0.0003)
        .collect();
    let selector_input =
        Array::from_slice(&selector_input_values, &[1, 1, config.hidden_size as i32]);
    let selector_qr_raw = dsa.q_a.forward(&selector_input)?;
    let selector_qr = glm53_weighted_rms_norm(&selector_qr_raw, &dsa.q_a_norm, dsa.norm_eps)?;
    let (selected, valid_prefix) = dsa.indexer.select_ids_gpu(
        &mut selector_tape,
        &selector_input,
        &selector_qr,
        selector_tokens,
        1,
        None,
        3,
        false,
    )?;
    let expected_prefix = (dsa.indexer.topk / kpool).min(complete_pools) * kpool;
    anyhow::ensure!(
        valid_prefix == expected_prefix
            && selected.dim(2) as usize == expected_prefix + kpool.saturating_sub(1),
        "one-token selector returned an invalid DS4 prefix contract: prefix={valid_prefix}, width={}, expected_prefix={expected_prefix}",
        selected.dim(2)
    );
    let selected_ids = selected.to_vec1::<i32>()?;
    anyhow::ensure!(
        selected_ids[..valid_prefix]
            .iter()
            .all(|&id| id >= 0 && (id as usize) < selector_tokens),
        "pooled DSA selector placed an invalid token in its guaranteed prefix"
    );
    anyhow::ensure!(
        selected_ids[valid_prefix..]
            .iter()
            .all(|&id| id == -1 || (id >= 0 && (id as usize) < selector_tokens)),
        "pooled DSA selector emitted a non-causal padded-tail id"
    );
    eprintln!(
        "[glm53-dsa-selection] decode prefix={valid_prefix}, total_ids={}, tail={:?}",
        selected_ids.len(),
        &selected_ids[valid_prefix..]
    );

    let loaded_model_capture: &[u8] =
        &crate::models::glm53::test_fixtures::read("dsa_attention_layer3_model_long_ds4.bin");
    const MODEL_Q_BYTES: usize = 64 * 256 * 4;
    const MODEL_Q_LOW_BYTES: usize = 64 * 512 * 4;
    const MODEL_CACHE_ROWS: usize = 16_385;
    const MODEL_CACHE_BYTES: usize = MODEL_CACHE_ROWS * 512 * 2;
    const MODEL_SELECTION_BYTES: usize = 2_051 * 4;
    const MODEL_LORA_BYTES: usize = 64 * 512 * 4;
    anyhow::ensure!(
        loaded_model_capture.len()
            == MODEL_Q_BYTES
                + MODEL_Q_LOW_BYTES
                + MODEL_CACHE_BYTES
                + MODEL_SELECTION_BYTES
                + MODEL_LORA_BYTES,
        "model-derived DS4 DSA fixture has an invalid layout"
    );
    let decode_f32 = |bytes: &[u8]| {
        bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("model F32 word")))
            .collect::<Vec<_>>()
    };
    let q_start = 0;
    let q_low_start = q_start + MODEL_Q_BYTES;
    let cache_start = q_low_start + MODEL_Q_LOW_BYTES;
    let selection_start = cache_start + MODEL_CACHE_BYTES;
    let lora_start = selection_start + MODEL_SELECTION_BYTES;
    let model_q = decode_f32(&loaded_model_capture[q_start..q_low_start]);
    let captured_q_low = decode_f32(&loaded_model_capture[q_low_start..cache_start]);
    let model_cache = loaded_model_capture[cache_start..selection_start]
        .chunks_exact(2)
        .map(|chunk| half::f16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]])))
        .collect::<Vec<_>>();
    let model_selected = loaded_model_capture[selection_start..lora_start]
        .chunks_exact(4)
        .map(|chunk| i32::from_le_bytes(chunk.try_into().expect("model selection word")))
        .collect::<Vec<_>>();
    let captured_lora = decode_f32(&loaded_model_capture[lora_start..]);
    anyhow::ensure!(
        model_selected
            .iter()
            .all(|&id| id == -1 || (0..MODEL_CACHE_ROWS as i32).contains(&id))
            && model_selected[..2048].iter().all(|&id| id >= 0)
            && model_selected[2048] >= 0
            && model_selected[2049..].iter().all(|&id| id == -1),
        "model-derived DSA selection has invalid sparse-boundary IDs"
    );
    let model_q_array = Array::from_slice(&model_q, &[1, 64, 1, 256]);
    let model_cache_array = Array::from_slice(&model_cache, &[1, MODEL_CACHE_ROWS as i32, 512]);
    let model_selected_array = Array::from_slice(&model_selected, &[1, 1, 2_051]);
    let model_scale = (config.qk_nope_head_dim as f64).powf(-0.5) as f32;
    let (model_staged, model_q_low, model_lora_actual) =
        lisa_mlx::models::glm53::kernels::dsa_absorbed_attention_stages(
            &model_q_array,
            &dsa.kv_b.weight,
            &dsa.kv_b.scales,
            &dsa.kv_b.biases,
            dsa.kv_b.group_size as usize,
            &model_cache_array,
            &model_selected_array,
            model_scale,
            &stream,
            false,
        )
        .ok_or_else(|| anyhow::anyhow!("model-derived staged DSA decode declined"))?;
    let model_q_low = model_q_low.to_vec1::<f32>()?;
    let model_lora_actual = model_lora_actual.to_vec1::<f32>()?;
    anyhow::ensure!(
        model_q_low
            .iter()
            .map(|x| x.to_bits())
            .eq(captured_q_low.iter().map(|x| x.to_bits())),
        "model-derived QK-lowrank output changed from captured output"
    );
    anyhow::ensure!(
        model_lora_actual
            .iter()
            .map(|x| x.to_bits())
            .eq(captured_lora.iter().map(|x| x.to_bits())),
        "model-derived staged indexed-lora output changed from captured output"
    );
    let model_staged = model_staged.to_vec1::<f32>()?;
    let model_fused = lisa_mlx::models::glm53::kernels::dsa_absorbed_attention_decode(
        &model_q_array,
        &dsa.kv_b.weight,
        &dsa.kv_b.scales,
        &dsa.kv_b.biases,
        dsa.kv_b.group_size as usize,
        &model_cache_array,
        Some(&model_selected_array),
        2_048,
        model_scale,
        &stream,
    )
    .ok_or_else(|| anyhow::anyhow!("model-derived fused DSA decode declined"))?
    .to_vec1::<f32>()?;
    let model_max_abs = model_fused
        .iter()
        .zip(&model_staged)
        .map(|(fused, staged)| (fused - staged).abs())
        .fold(0.0f32, f32::max);
    anyhow::ensure!(
        model_max_abs <= 2e-4,
        "model-derived fused DSA decode differs from staged path: max_abs={model_max_abs:e}"
    );
    let loaded_model_production_fixture: &[u8] = &crate::models::glm53::test_fixtures::read(
        "dsa_production_decode_layer3_model_long_ds4.bin",
    );
    anyhow::ensure!(
        loaded_model_production_fixture.len() == 64 * 256 * std::mem::size_of::<f32>(),
        "model-derived DS4 production DSA fixture has an invalid size"
    );
    let model_ds4_output = decode_f32(loaded_model_production_fixture);
    anyhow::ensure!(
        model_fused
            .iter()
            .map(|value| value.to_bits())
            .eq(model_ds4_output.iter().map(|value| value.to_bits())),
        "model-derived fused DSA decode differs from DS4 production kernel"
    );
    eprintln!(
        "[glm53-dsa-model-production] 16K prefix, 2,049 selected IDs match DS4 bit-for-bit; fused/staged max_abs={model_max_abs:.3e}"
    );
    if let Some(output_dir) = std::env::var_os("LISA_GLM53_DSA_PRODUCTION_CAPTURE_DIR") {
        let output_dir = std::path::Path::new(&output_dir);
        std::fs::create_dir_all(output_dir)?;
        let write_f32 = |name: &str, values: &[f32]| -> anyhow::Result<()> {
            let bytes: Vec<u8> = values.iter().flat_map(|x| x.to_le_bytes()).collect();
            std::fs::write(output_dir.join(name), bytes)?;
            Ok(())
        };
        let write_u32 = |name: &str, values: &[u32]| -> anyhow::Result<()> {
            let bytes: Vec<u8> = values.iter().flat_map(|x| x.to_le_bytes()).collect();
            std::fs::write(output_dir.join(name), bytes)?;
            Ok(())
        };
        write_f32("wide_q.f32", &model_q)?;
        write_f32("wide_q_low.f32", &model_q_low)?;
        write_f32("wide_lisa_output.f32", &model_fused)?;
        write_u32(
            "wide_selected.u32",
            &model_selected
                .iter()
                .map(|&id| id as u32)
                .collect::<Vec<_>>(),
        )?;
        let cache_bytes: Vec<u8> = model_cache
            .iter()
            .flat_map(|x| x.to_bits().to_le_bytes())
            .collect();
        std::fs::write(output_dir.join("wide_latents.f16"), cache_bytes)?;
        write_u32("weight.u32", &dsa.kv_b.weight.to_vec1::<u32>()?)?;
        write_f32("scales.f32", &dsa.kv_b.scales.to_vec1::<f32>()?)?;
        write_f32("biases.f32", &dsa.kv_b.biases.to_vec1::<f32>()?)?;
    }
    Ok(())
}

#[test]
fn glm53_dsa_production_decode_matches_ds4_model_long_all_layers() -> anyhow::Result<()> {
    if !lisa_mlx::device::metal_available() {
        eprintln!("[glm53-dsa-model-production] skipping: no Metal device available");
        return Ok(());
    }
    let Some(model_dir) = std::env::var_os("LISA_GLM53_MODEL") else {
        eprintln!("[glm53-dsa-model-production] skipping: LISA_GLM53_MODEL is unset");
        return Ok(());
    };
    const LAYERS: [usize; 11] = [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43];
    const Q_BYTES: usize = 64 * 256 * 4;
    const Q_LOW_BYTES: usize = 64 * 512 * 4;
    const CACHE_ROWS: usize = 16_385;
    const CACHE_BYTES: usize = CACHE_ROWS * 512 * 2;
    const SELECTION_BYTES: usize = 2_051 * 4;
    const LORA_BYTES: usize = 64 * 512 * 4;
    let dir = std::path::Path::new(&model_dir);
    let config = super::super::config::ModelConfig::from_json(&dir.join("config.json"))?;
    let output_root =
        std::env::var_os("LISA_GLM53_DSA_PRODUCTION_CAPTURE_DIR").map(std::path::PathBuf::from);
    let capture_only = std::env::var_os("LISA_GLM53_DSA_PRODUCTION_CAPTURE_ONLY").is_some();
    anyhow::ensure!(
        !capture_only || output_root.is_some(),
        "capture-only mode requires LISA_GLM53_DSA_PRODUCTION_CAPTURE_DIR"
    );
    if let Some(output_root) = &output_root {
        std::fs::create_dir_all(output_root)?;
    }
    let stream = lisa_mlx::Stream::thread_local_or_default();
    for layer in LAYERS {
        anyhow::ensure!(!config.is_kda(layer), "configured DSA layer {layer} is KDA");
        let fixture = model_long_dsa_fixture(layer)?;
        anyhow::ensure!(
            fixture.len() == Q_BYTES + Q_LOW_BYTES + CACHE_BYTES + SELECTION_BYTES + LORA_BYTES,
            "layer-{layer} model-derived DSA fixture has an invalid size"
        );
        let q_start = 0;
        let q_low_start = q_start + Q_BYTES;
        let cache_start = q_low_start + Q_LOW_BYTES;
        let selection_start = cache_start + CACHE_BYTES;
        let lora_start = selection_start + SELECTION_BYTES;
        let q = decode_fixture_f32(&fixture[q_start..q_low_start]);
        let q_low_expected = decode_fixture_f32(&fixture[q_low_start..cache_start]);
        let cache = fixture[cache_start..selection_start]
            .chunks_exact(2)
            .map(|chunk| half::f16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]])))
            .collect::<Vec<_>>();
        let selected = fixture[selection_start..lora_start]
            .chunks_exact(4)
            .map(|chunk| i32::from_le_bytes(chunk.try_into().expect("selection word")))
            .collect::<Vec<_>>();
        let lora_expected = decode_fixture_f32(&fixture[lora_start..]);
        anyhow::ensure!(
            selected
                .iter()
                .all(|&id| id == -1 || (0..CACHE_ROWS as i32).contains(&id))
                && selected[..2048].iter().all(|&id| id >= 0)
                && selected[2048] >= 0,
            "layer-{layer} model-derived sparse selection is invalid"
        );
        {
            let mut weights = load_layer_dsa_weights_at(dir, layer)?;
            let prefix = format!("model.layers.{layer}.self_attn");
            let dsa = Dsa::load::<Weights>(
                &mut weights,
                &prefix,
                config.num_attention_heads,
                config.qk_nope_head_dim,
                config.v_head_dim,
                config.index_n_heads,
                config.index_head_dim,
                config.index_topk,
                config.index_kpool,
                config.index_kpool_always_select_tail,
                config.rms_norm_eps,
            )?;
            let q_array = Array::from_slice(&q, &[1, 64, 1, 256]);
            let cache_array = Array::from_slice(&cache, &[1, CACHE_ROWS as i32, 512]);
            let selected_array = Array::from_slice(&selected, &[1, 1, 2_051]);
            let scale = (config.qk_nope_head_dim as f64).powf(-0.5) as f32;
            let (staged, q_low, lora) =
                lisa_mlx::models::glm53::kernels::dsa_absorbed_attention_stages(
                    &q_array,
                    &dsa.kv_b.weight,
                    &dsa.kv_b.scales,
                    &dsa.kv_b.biases,
                    dsa.kv_b.group_size as usize,
                    &cache_array,
                    &selected_array,
                    scale,
                    &stream,
                    false,
                )
                .ok_or_else(|| anyhow::anyhow!("layer-{layer} staged decode declined"))?;
            let q_low = q_low.to_vec1::<f32>()?;
            let lora = lora.to_vec1::<f32>()?;
            anyhow::ensure!(
                q_low
                    .iter()
                    .map(|x| x.to_bits())
                    .eq(q_low_expected.iter().map(|x| x.to_bits())),
                "layer-{layer} QK-lowrank output differs from the captured model value"
            );
            anyhow::ensure!(
                lora.iter()
                    .map(|x| x.to_bits())
                    .eq(lora_expected.iter().map(|x| x.to_bits())),
                "layer-{layer} indexed-lora output differs from DS4 capture"
            );
            let staged = staged.to_vec1::<f32>()?;
            let fused = lisa_mlx::models::glm53::kernels::dsa_absorbed_attention_decode(
                &q_array,
                &dsa.kv_b.weight,
                &dsa.kv_b.scales,
                &dsa.kv_b.biases,
                dsa.kv_b.group_size as usize,
                &cache_array,
                Some(&selected_array),
                2_048,
                scale,
                &stream,
            )
            .ok_or_else(|| anyhow::anyhow!("layer-{layer} fused decode declined"))?
            .to_vec1::<f32>()?;
            let max_abs = fused
                .iter()
                .zip(&staged)
                .map(|(actual, expected)| (actual - expected).abs())
                .fold(0.0f32, f32::max);
            anyhow::ensure!(
                max_abs <= 2e-4,
                "layer-{layer} fused/staged DSA output differs: max_abs={max_abs:e}"
            );
            if !capture_only {
                let ds4_fixture = model_long_dsa_production_fixture(layer)?;
                anyhow::ensure!(
                    ds4_fixture.len() == 64 * 256 * std::mem::size_of::<f32>(),
                    "layer-{layer} DS4 production DSA fixture has an invalid size"
                );
                let ds4_output = decode_fixture_f32(&ds4_fixture);
                anyhow::ensure!(
                    fused
                        .iter()
                        .map(|value| value.to_bits())
                        .eq(ds4_output.iter().map(|value| value.to_bits())),
                    "layer-{layer} model-derived DSA output differs from DS4 production kernel"
                );
            }
            if let Some(output_root) = &output_root {
                let layer_dir = output_root.join(format!("layer-{layer}"));
                std::fs::create_dir_all(&layer_dir)?;
                let write_f32 = |name: &str, values: &[f32]| -> anyhow::Result<()> {
                    let bytes: Vec<u8> = values.iter().flat_map(|x| x.to_le_bytes()).collect();
                    std::fs::write(layer_dir.join(name), bytes)?;
                    Ok(())
                };
                let write_u32 = |name: &str, values: &[u32]| -> anyhow::Result<()> {
                    let bytes: Vec<u8> = values.iter().flat_map(|x| x.to_le_bytes()).collect();
                    std::fs::write(layer_dir.join(name), bytes)?;
                    Ok(())
                };
                write_f32("wide_q.f32", &q)?;
                write_f32("wide_q_low.f32", &q_low)?;
                write_f32("wide_lisa_output.f32", &fused)?;
                write_u32(
                    "wide_selected.u32",
                    &selected.iter().map(|&id| id as u32).collect::<Vec<_>>(),
                )?;
                let cache_bytes: Vec<u8> = cache
                    .iter()
                    .flat_map(|x| x.to_bits().to_le_bytes())
                    .collect();
                std::fs::write(layer_dir.join("wide_latents.f16"), cache_bytes)?;
                write_u32("weight.u32", &dsa.kv_b.weight.to_vec1::<u32>()?)?;
                write_f32("scales.f32", &dsa.kv_b.scales.to_vec1::<f32>()?)?;
                write_f32("biases.f32", &dsa.kv_b.biases.to_vec1::<f32>()?)?;
            }
            eprintln!(
                "[glm53-dsa-model-production] layer={layer} selected=2051 capture_only={capture_only} fused/staged max_abs={max_abs:.3e}"
            );
        }
        stream.runtime().commands.flush_and_wait()?;
    }
    Ok(())
}

#[test]
fn glm53_dsa_dense_prefill_matches_incremental_absorbed_decode() -> anyhow::Result<()> {
    lisa_mlx::Stream::thread_local_or_default()
        .runtime()
        .commands
        .flush_and_wait()?;
    if !lisa_mlx::device::metal_available() {
        eprintln!("[glm53-dsa-cache] skipping: no Metal device available");
        return Ok(());
    }
    let Some(model_dir) = std::env::var_os("LISA_GLM53_MODEL") else {
        eprintln!("[glm53-dsa-cache] skipping: LISA_GLM53_MODEL is unset");
        return Ok(());
    };
    let dir = std::path::Path::new(&model_dir);
    let config = super::super::config::ModelConfig::from_json(&dir.join("config.json"))?;
    anyhow::ensure!(!config.is_kda(3), "GLM layer 3 is not a DSA layer");
    let mut prefill_weights = load_layer_dsa_weights(dir)?;
    let mut decode_weights = prefill_weights.clone();
    let load = |weights: &mut Weights| {
        Dsa::load::<Weights>(
            weights,
            "model.layers.3.self_attn",
            config.num_attention_heads,
            config.qk_nope_head_dim,
            config.v_head_dim,
            config.index_n_heads,
            config.index_head_dim,
            config.index_topk,
            config.index_kpool,
            config.index_kpool_always_select_tail,
            config.rms_norm_eps,
        )
    };
    let prefill_dsa = load(&mut prefill_weights)?;
    let decode_dsa = load(&mut decode_weights)?;

    // This layer's short-context prefill uses dense projected K/V while
    // one-token decode uses absorbed MLA over the compact F16 tape. Compare
    // their output for identical real-checkpoint weights and input rows.
    let (steps, hidden) = (8usize, config.hidden_size);
    let input_values: Vec<f32> = (0..steps * hidden)
        .map(|i| (((i * 37 + 13) % 251) as f32 - 125.0) * 0.0004)
        .collect();
    let input = Array::from_slice(&input_values, &[1, steps as i32, hidden as i32]);
    let expected_q_raw = prefill_dsa.q_a.forward_glm53_prefill(&input)?;
    let expected_kv_raw = prefill_dsa.kv_a.forward_glm53_prefill(&input)?;
    let (_, expected_compact) = lisa_mlx::models::glm53::kernels::glm53_qkv_norm_compact(
        &expected_q_raw,
        &prefill_dsa.q_a_norm,
        &expected_kv_raw,
        &prefill_dsa.kv_a_norm,
        prefill_dsa.norm_eps,
        &lisa_mlx::Stream::thread_local_or_default(),
    )
    .ok_or_else(|| anyhow::anyhow!("temporary-output fused norm path declined"))?;
    let expected_compact = expected_compact.to_vec1::<half::f16>()?;
    let mut prefill_cache = FullAttentionCache::new(1, config.kv_lora_rank);
    let prefill_projection =
        prefill_dsa.project_inputs(&input, None, 3, Some(&mut prefill_cache.indexer_tape))?;
    let prefill_pre_o = prefill_dsa.forward_projected(
        &input,
        &mut prefill_cache,
        None,
        3,
        &prefill_projection,
        None,
        true,
    )?;
    let prefill_out = prefill_dsa.output_project(&prefill_pre_o)?;
    let mut decode_cache = FullAttentionCache::new(1, config.kv_lora_rank);
    let mut decoded_rows = Vec::with_capacity(steps);
    for row in 0..steps {
        let token = input
            .index((.., row as i32..(row + 1) as i32, ..))
            .contiguous()?;
        decoded_rows.push(decode_dsa.forward(&token, &mut decode_cache, None, 3, false)?);
    }

    let prefill_latents = prefill_cache
        .indexer_tape
        .compact_slice(0, steps)
        .ok_or_else(|| anyhow::anyhow!("direct DSA prefill cache is missing"))?;
    let prefill_latents = prefill_latents.to_vec1::<half::f16>()?;
    let cache_mismatch = prefill_latents
        .iter()
        .zip(&expected_compact)
        .enumerate()
        .find(|(_, (prefill, expected))| prefill.to_bits() != expected.to_bits());
    anyhow::ensure!(
        cache_mismatch.is_none(),
        "direct-write DSA prefill cache differs from temporary-output fused cache: first={cache_mismatch:?}"
    );

    let expected = prefill_out.to_vec1::<f32>()?;
    let mut actual = Vec::with_capacity(expected.len());
    for row in decoded_rows {
        actual.extend(row.to_vec1::<f32>()?);
    }
    anyhow::ensure!(
        expected.iter().all(|x| x.is_finite()) && actual.iter().all(|x| x.is_finite()),
        "GLM DSA prefill/decode produced non-finite output"
    );
    let max_error = expected
        .iter()
        .zip(&actual)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let expected_norm = expected.iter().map(|x| x * x).sum::<f32>().sqrt();
    let actual_norm = actual.iter().map(|x| x * x).sum::<f32>().sqrt();
    let cosine = expected
        .iter()
        .zip(&actual)
        .map(|(a, b)| a * b)
        .sum::<f32>()
        / (expected_norm * actual_norm + 1e-9);
    eprintln!("[glm53-dsa-cache] max_abs={max_error:.6e} cosine={cosine:.8}");
    anyhow::ensure!(
        max_error <= 2e-3 && cosine >= 0.995,
        "GLM DSA dense/absorbed prefill-decode mismatch: max abs {max_error}, cosine {cosine}"
    );
    if lisa_mlx::trace::enabled() {
        lisa_mlx::trace::report();
    }
    Ok(())
}

#[test]
fn glm53_indexer_projection_uses_bf16_tiled_prefill() -> anyhow::Result<()> {
    if !lisa_mlx::device::metal_available() {
        eprintln!("[glm53-indexer-projection] skipping: no Metal device available");
        return Ok(());
    }
    let (in_dim, out_dim, rows) = (323usize, 13usize, 9usize);
    let source_weights: Vec<f32> = (0..out_dim * in_dim)
        .map(|i| (((i * 17 + 5) % 113) as f32 - 56.0) * 0.001)
        .collect();
    let weights = Array::from_slice(&source_weights, &[out_dim as i32, in_dim as i32])
        .as_dtype(Dtype::Bfloat16)?;
    let input_values: Vec<f32> = (0..rows * in_dim)
        .map(|i| (((i * 11 + 9) % 97) as f32 - 48.0) * 0.002)
        .collect();
    let input = Array::from_slice(&input_values, &[1, rows as i32, in_dim as i32]);
    let projection = Indexer {
        wq_b_w: weights.clone(),
        wk_w: weights.clone(),
        k_norm_w: Array::from_slice(&vec![1.0f32; 128], &[128]),
        k_norm_b: Array::from_slice(&vec![0.0f32; 128], &[128]),
        weights_w: weights.clone(),
        gate_w: weights.clone(),
        ape: Array::from_slice(&vec![0.0f32; 512], &[4, 128]),
        heads: 32,
        head_dim: 128,
        topk: 2048,
        kpool: 4,
        always_tail: true,
    };
    let actual = projection.project(&input, &weights)?.to_vec1::<f32>()?;
    let rounded_weights = weights.as_dtype(Dtype::Float32)?.to_vec1::<f32>()?;
    for row in 0..rows {
        for col in 0..out_dim {
            let mut expected = 0.0f64;
            for k in 0..in_dim {
                let w = half::f16::from_f32(rounded_weights[col * in_dim + k]).to_f32();
                let x = half::f16::from_f32(input_values[row * in_dim + k]).to_f32();
                expected += f64::from(w) * f64::from(x);
            }
            assert!(
                (f64::from(actual[row * out_dim + col]) - expected).abs() < 2e-4,
                "indexer projection ({row},{col}): {} vs {expected}",
                actual[row * out_dim + col]
            );
        }
    }
    Ok(())
}

#[test]
fn glm53_indexer_pair_projection_matches_separate_gemv() -> anyhow::Result<()> {
    if !lisa_mlx::device::metal_available() {
        eprintln!("[glm53-indexer-pair] skipping: no Metal device available");
        return Ok(());
    }
    let make_weights = |out_dim: usize, in_dim: usize, salt: usize| {
        let values: Vec<f32> = (0..out_dim * in_dim)
            .map(|i| (((i * 19 + salt * 23) % 127) as f32 - 63.0) * 0.001)
            .collect();
        Array::from_slice(&values, &[out_dim as i32, in_dim as i32])
            .as_dtype(Dtype::Bfloat16)
            .expect("cast paired indexer weights to BF16")
    };
    let weights_a = make_weights(17, 323, 3);
    let weights_b = make_weights(9, 197, 7);
    let x_a_values: Vec<f32> = (0..2 * 323)
        .map(|i| (((i * 11 + 5) % 97) as f32 - 48.0) * 0.002)
        .collect();
    let x_b_values: Vec<f32> = (0..2 * 197)
        .map(|i| (((i * 13 + 9) % 103) as f32 - 51.0) * 0.0015)
        .collect();
    let x_a = Array::from_slice(&x_a_values, &[1, 2, 323]);
    let x_b = Array::from_slice(&x_b_values, &[1, 2, 197]);
    let projection = Indexer {
        wq_b_w: weights_a.clone(),
        wk_w: weights_a.clone(),
        k_norm_w: Array::from_slice(&vec![1.0f32; 128], &[128]),
        k_norm_b: Array::from_slice(&vec![0.0f32; 128], &[128]),
        weights_w: weights_b.clone(),
        gate_w: weights_b.clone(),
        ape: Array::from_slice(&vec![0.0f32; 512], &[4, 128]),
        heads: 32,
        head_dim: 128,
        topk: 2048,
        kpool: 4,
        always_tail: true,
    };

    let before = lisa_mlx::runtime_dispatch_count();
    let (paired_a, paired_b) = projection.project_pair(&x_a, &weights_a, &x_b, &weights_b)?;
    assert_eq!(
        lisa_mlx::runtime_dispatch_count() - before,
        1,
        "paired indexer projections should use one dispatch"
    );
    let separate_a = projection.project(&x_a, &weights_a)?.to_vec1::<f32>()?;
    let separate_b = projection.project(&x_b, &weights_b)?.to_vec1::<f32>()?;
    let paired_a = paired_a.to_vec1::<f32>()?;
    let paired_b = paired_b.to_vec1::<f32>()?;
    assert_eq!(
        paired_a.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        separate_a.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        "paired A projection changed the DS4 GEMV result"
    );
    assert_eq!(
        paired_b.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        separate_b.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        "paired B projection changed the DS4 GEMV result"
    );
    Ok(())
}

#[test]
fn glm53_indexer_bf16_gemm_accepts_real_mlx_checkpoint_weights() -> anyhow::Result<()> {
    if !lisa_mlx::device::metal_available() {
        eprintln!("[glm53-indexer-mlx-weight] skipping: no Metal device available");
        return Ok(());
    }
    let Some(model_dir) = std::env::var_os("LISA_GLM53_MODEL") else {
        eprintln!("[glm53-indexer-mlx-weight] skipping: LISA_GLM53_MODEL is unset");
        return Ok(());
    };
    let checkpoint = crate::core::loader::Checkpoint::open(std::path::Path::new(&model_dir))?;
    let tensors = [
        "model.language_model.layers.11.self_attn.indexer.wq_b.weight",
        "model.language_model.layers.11.self_attn.indexer.wk.weight",
        "model.language_model.layers.11.self_attn.indexer.weights_proj.weight",
        "model.language_model.layers.11.self_attn.indexer.index_kpool_compress_gate",
    ];
    let shard_name = checkpoint
        .weight_map
        .get(tensors[0])
        .ok_or_else(|| anyhow::anyhow!("checkpoint has no {}", tensors[0]))?;
    let shard =
        crate::core::loader::Shard::open(&std::path::Path::new(&model_dir).join(shard_name))?;
    let rows = 9usize;
    let stream = lisa_mlx::Stream::thread_local_or_default();
    for tensor_name in tensors {
        let mapped_shard = checkpoint
            .weight_map
            .get(tensor_name)
            .ok_or_else(|| anyhow::anyhow!("checkpoint has no {tensor_name}"))?;
        anyhow::ensure!(
            mapped_shard == shard_name,
            "fixture tensors must share one shard"
        );
        let weight = crate::core::quant::get_tensor(&shard, tensor_name)?;
        anyhow::ensure!(
            weight.dtype() == Dtype::Bfloat16,
            "unexpected checkpoint indexer weight dtype {:?} for {tensor_name}",
            weight.dtype()
        );
        let in_dim = weight.dim(1) as usize;
        let out_dim = weight.dim(0) as usize;
        let values: Vec<f32> = (0..rows * in_dim)
            .map(|i| (((i * 29 + 7) % 251) as f32 - 125.0) * 0.0002)
            .collect();
        let input = Array::from_slice(&values, &[rows as i32, in_dim as i32]);
        let output = lisa_mlx::models::glm53::kernels::glm53_bf16_gemm(&weight, &input, &stream)
            .ok_or_else(|| {
                anyhow::anyhow!("BF16 tiled GEMM declined real checkpoint tensor {tensor_name}")
            })?;
        let projected = output.to_vec1::<f32>()?;
        anyhow::ensure!(
            projected.len() == rows * out_dim
                && projected.iter().all(|v| v.is_finite())
                && projected.iter().any(|v| *v != 0.0),
            "real checkpoint projection returned invalid output for {tensor_name}"
        );
        let decode_input = Array::from_slice(&values[..in_dim], &[1, in_dim as i32]);
        let decode_output =
            lisa_mlx::models::glm53::kernels::glm53_bf16_gemv(&weight, &decode_input, &stream)
                .ok_or_else(|| {
                    anyhow::anyhow!("BF16 GEMV declined real checkpoint tensor {tensor_name}")
                })?;
        let decode_values = decode_output.to_vec1::<f32>()?;
        anyhow::ensure!(
            decode_values.len() == out_dim
                && decode_values.iter().all(|v| v.is_finite())
                && decode_values.iter().any(|v| *v != 0.0),
            "real checkpoint decode projection returned invalid output for {tensor_name}"
        );
    }

    let pool_tensors = [
        "model.language_model.layers.11.self_attn.indexer.k_norm.weight",
        "model.language_model.layers.11.self_attn.indexer.k_norm.bias",
        "model.language_model.layers.11.self_attn.indexer.index_kpool_compress_ape",
    ];
    let mut pool_weights = Vec::with_capacity(pool_tensors.len());
    for tensor_name in pool_tensors {
        let mapped_shard = checkpoint
            .weight_map
            .get(tensor_name)
            .ok_or_else(|| anyhow::anyhow!("checkpoint has no {tensor_name}"))?;
        anyhow::ensure!(
            mapped_shard == shard_name,
            "fixture tensors must share one shard"
        );
        pool_weights.push(crate::core::quant::get_tensor(&shard, tensor_name)?);
    }
    anyhow::ensure!(
        pool_weights
            .iter()
            .all(|weight| weight.dtype() == Dtype::Bfloat16),
        "unexpected checkpoint indexer pooling weight dtype"
    );
    let raw_key_values: Vec<f32> = (0..2 * 4 * 128)
        .map(|i| (((i * 37 + 13) % 257) as f32 - 128.0) * 0.001)
        .collect();
    let gate_values: Vec<f32> = (0..2 * 4 * 128)
        .map(|i| (((i * 19 + 7) % 193) as f32 - 96.0) * 0.002)
        .collect();
    let raw_key = Array::from_slice(&raw_key_values, &[1, 2, 4, 128]);
    let gate = Array::from_slice(&gate_values, &[1, 2, 4, 128]);
    let pooled = lisa_mlx::models::glm53::kernels::indexer_pool_fused(
        &raw_key,
        &gate,
        &pool_weights[2],
        &pool_weights[0],
        &pool_weights[1],
        LAYER_NORM_EPS,
        &stream,
    )
    .ok_or_else(|| anyhow::anyhow!("fused pool kernel declined real checkpoint weights"))?;
    let pooled_values = pooled.as_dtype(Dtype::Float32)?.to_vec1::<f32>()?;
    anyhow::ensure!(
        pooled.shape() == [1, 2, 128]
            && pooled_values.iter().all(|v| v.is_finite())
            && pooled_values.iter().any(|v| *v != 0.0),
        "real checkpoint pooling returned invalid output"
    );
    Ok(())
}
