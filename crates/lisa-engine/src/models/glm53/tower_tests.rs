use super::*;
use crate::models::LanguageModel;

#[test]
fn ds4_native_glm_batch_position_guard_matches_reference_boundaries() {
    let dense_compact_limit = 2048 + 4 - 1;
    assert!(ds4_native_session_batch_position_supported(
        2,
        dense_compact_limit - 1,
        dense_compact_limit
    ));
    assert!(!ds4_native_session_batch_position_supported(
        2,
        dense_compact_limit,
        dense_compact_limit
    ));
    assert!(!ds4_native_session_batch_position_supported(
        1,
        0,
        dense_compact_limit
    ));
    assert!(!ds4_native_session_batch_position_supported(
        DS4_GLM53_NATIVE_BATCH_ROW_LIMIT + 1,
        0,
        dense_compact_limit
    ));
}

#[test]
fn glm53_full_checkpoint_load_without_expert_residency_probe() -> anyhow::Result<()> {
    if !lisa_mlx::device::metal_available() {
        eprintln!("[glm53-loader-probe] skipping: no Metal device available");
        return Ok(());
    }
    let Some(model_dir) = std::env::var_os("LISA_GLM53_MODEL") else {
        eprintln!("[glm53-loader-probe] skipping: LISA_GLM53_MODEL is unset");
        return Ok(());
    };
    let model_dir = Path::new(&model_dir);
    let config = ModelConfig::from_json(&model_dir.join("config.json"))?;
    let started = std::time::Instant::now();
    let tower = Tower::load_without_expert_residency(model_dir, config.clone())?;
    anyhow::ensure!(tower.n_layers == config.num_hidden_layers);
    anyhow::ensure!(
        tower._expert_residency_set.is_none(),
        "diagnostic GLM loading unexpectedly created a queue residency set"
    );
    let diagnostic_set = Array::new_residency_set(1)
        .ok_or_else(|| anyhow::anyhow!("Metal residency sets are unavailable"))?;
    let mut lazy_pools = 0;
    for layer in &tower.layers {
        if let Mlp::Sparse(moe) = &layer.mlp {
            anyhow::ensure!(!moe.add_expert_resources_to_residency_set(&diagnostic_set));
            lazy_pools += 1;
        }
    }
    if let Some(block) = &tower.mtp {
        anyhow::ensure!(
            !block
                .mlp
                .add_expert_resources_to_residency_set(&diagnostic_set)
        );
        lazy_pools += 1;
    }
    eprintln!(
        "[glm53-loader-probe] deferred expert tower loaded without residency in {:.1}s; {} lazy expert pools register no model-wide buffers",
        started.elapsed().as_secs_f64(),
        lazy_pools,
    );
    eprintln!(
        "[glm53-loader-probe] tower remains usable; elapsed {:.1}s",
        started.elapsed().as_secs_f64()
    );
    drop(tower);
    Ok(())
}

#[test]
fn glm53_full_checkpoint_load_with_expert_residency_probe() -> anyhow::Result<()> {
    if !lisa_mlx::device::metal_available() {
        eprintln!("[glm53-loader-probe] skipping: no Metal device available");
        return Ok(());
    }
    let Some(model_dir) = std::env::var_os("LISA_GLM53_MODEL") else {
        eprintln!("[glm53-loader-probe] skipping: LISA_GLM53_MODEL is unset");
        return Ok(());
    };
    let model_dir = Path::new(&model_dir);
    let config = ModelConfig::from_json(&model_dir.join("config.json"))?;
    let started = std::time::Instant::now();
    let tower = Tower::load_with_expert_residency(model_dir, config.clone(), true)?;
    anyhow::ensure!(tower.n_layers == config.num_hidden_layers);
    let set = tower
        ._expert_residency_set
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("model-wide expert residency set was not created"))?;
    eprintln!(
        "[glm53-loader-probe] full tower loaded with {} model-wide expert residency allocations in {:.1}s",
        set.allocation_count(),
        started.elapsed().as_secs_f64()
    );
    drop(tower);
    Ok(())
}

#[test]
fn glm53_full_checkpoint_load_with_deferred_residency_probe() -> anyhow::Result<()> {
    if !lisa_mlx::device::metal_available() {
        eprintln!("[glm53-loader-probe] skipping: no Metal device available");
        return Ok(());
    }
    let Some(model_dir) = std::env::var_os("LISA_GLM53_MODEL") else {
        eprintln!("[glm53-loader-probe] skipping: LISA_GLM53_MODEL is unset");
        return Ok(());
    };
    let model_dir = Path::new(&model_dir);
    let config = ModelConfig::from_json(&model_dir.join("config.json"))?;
    let started = std::time::Instant::now();
    let tower =
        Tower::load_with_expert_residency_mode(model_dir, config.clone(), true, false, false)?;
    anyhow::ensure!(tower.n_layers == config.num_hidden_layers);
    let set = tower
        ._expert_residency_set
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("deferred model residency set was not created"))?;
    eprintln!(
        "[glm53-loader-probe] full tower loaded with deferred residency: {} allocations, {:.2} GiB, {:.1}s",
        set.allocation_count(),
        set.allocation_bytes() as f64 / (1024.0 * 1024.0 * 1024.0),
        started.elapsed().as_secs_f64(),
    );
    drop(tower);
    Ok(())
}

fn load_prefix_tower(
    dir: &Path,
    source_config: &ModelConfig,
    last_layer: usize,
) -> anyhow::Result<Tower> {
    let checkpoint = loader::Checkpoint::open(dir)?;
    let lazy_store = std::sync::Arc::new(LazyTensorStore::open(dir)?);
    let full_depth = last_layer + 1 == source_config.num_hidden_layers;
    let mut by_shard: HashMap<String, Vec<(String, String)>> = HashMap::new();
    for (raw, shard) in &checkpoint.weight_map {
        let Some(sanitized) = loader::sanitize_name(raw) else {
            continue;
        };
        let selected = sanitized == "model.norm.weight"
            || sanitized.starts_with("model.embed_tokens.")
            || (full_depth && sanitized.starts_with("lm_head."))
            || sanitized
                .strip_prefix("model.layers.")
                .and_then(|tail| tail.split_once('.'))
                .and_then(|(layer, _)| layer.parse::<usize>().ok())
                .is_some_and(|layer| {
                    layer <= last_layer && !loader::is_glm_routed_expert_tensor(&sanitized)
                });
        if selected {
            by_shard
                .entry(shard.clone())
                .or_default()
                .push((raw.clone(), sanitized));
        }
    }

    let mut weights = Weights::new();
    for (shard_name, tensors) in by_shard {
        let shard = loader::Shard::open(&dir.join(&shard_name))?;
        for (raw, sanitized) in tensors {
            if weights.contains_key(&sanitized) {
                continue;
            }
            let (bytes, dtype, shape) = shard
                .tensor_bytes(&raw)
                .ok_or_else(|| anyhow::anyhow!("tensor {raw} missing from {shard_name}"))?;
            let array = if let Some(mapped) = shard.tensor_array(&raw)? {
                mapped
            } else {
                loader::array_from_bytes(bytes, dtype, shape)?
            };
            weights.insert(sanitized, array);
        }
        // Keep the mapped prefix weights under the normal VM cache policy,
        // matching ds4's ordinary model-mapping path.
    }
    if !full_depth {
        // Prefix execution never reaches the vocabulary projection. Keep
        // a tiny valid head instead of faulting the large tensor in.
        weights.insert("lm_head.weight".into(), Array::from_slice(&[0u32], &[1, 1]));
        weights.insert(
            "lm_head.scales".into(),
            Array::from_slice(&[1.0f32], &[1, 1]),
        );
        weights.insert(
            "lm_head.biases".into(),
            Array::from_slice(&[0.0f32], &[1, 1]),
        );
    }

    let mut config = source_config.clone();
    config.num_hidden_layers = last_layer + 1;
    config.layer_types.truncate(config.num_hidden_layers);
    config.mlp_layer_types.truncate(config.num_hidden_layers);
    config.num_nextn_predict_layers = 0;
    // Prefix captures are correctness diagnostics. Avoid pinning every
    // routed expert in the truncated tower; the per-dispatch fallback
    // keeps only selected expert buffers alive and bounds residency.
    Tower::from_weights_with_lazy_experts(weights, config, Some(lazy_store), false, false)
}

#[test]
fn capture_real_target_logits_and_q8_head_for_ds4() -> anyhow::Result<()> {
    let Some(capture_path) = std::env::var_os("LISA_GLM53_TARGET_LM_HEAD_CAPTURE_DIR") else {
        eprintln!("LISA_GLM53_TARGET_LM_HEAD_CAPTURE_DIR=<dir> to capture full-target logits");
        return Ok(());
    };
    let capture_path = Path::new(&capture_path).to_path_buf();
    std::fs::create_dir_all(&capture_path)?;
    if !lisa_mlx::device::metal_available() {
        eprintln!("[glm53-target-lm-head] skipping: no Metal device available");
        return Ok(());
    }
    let dir = std::env::var_os("LISA_GLM53_MODEL")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| {
            std::path::PathBuf::from(concat!(
                env!("HOME"),
                "/.cache/huggingface/hub/models--lisa--glm53-flash-2bit/snapshots/main"
            ))
        });
    let config = ModelConfig::from_json(&dir.join("config.json"))?;
    let fullbank_experiment = std::env::var_os("LISA_GLM53_TEST_FULLBANK_EXPERTS").is_some();
    let mut tower = if fullbank_experiment {
        Tower::load_mapped_experts_without_residency(&dir, config)?
    } else {
        Tower::load(&dir, config)?
    };
    if fullbank_experiment {
        eprintln!("[glm53-target-lm-head] using diagnostic all-expert shard mappings");
    }
    anyhow::ensure!(
        tower.lm_head.group_size == 64 && tower.lm_head.bits == 8,
        "GLM target output head is not affine Q8 group-64"
    );
    let tokenizer = crate::core::tokenizer::Tokenizer::load_file(&dir.join("tokenizer.json"))?;
    let prompt_text = if let Some(path) = std::env::var_os("LISA_GLM53_LOGIT_CAPTURE_PROMPT_FILE") {
        std::fs::read_to_string(path)?
    } else {
        std::env::var("LISA_GLM53_LOGIT_CAPTURE_PROMPT")
            .unwrap_or_else(|_| "The capital of France is".to_string())
    };
    let token_ids = tokenizer.encode(&prompt_text, false)?;
    anyhow::ensure!(
        !token_ids.is_empty(),
        "target logit prompt tokenized to empty input"
    );
    let input_ids: Vec<i32> = token_ids.iter().map(|&id| id as i32).collect();
    let mut caches = tower.new_caches();
    let final_token_layer_trace =
        std::env::var_os("LISA_GLM53_LOGIT_CAPTURE_FINAL_TOKEN_LAYER_TRACE_PATH");
    let hidden = if final_token_layer_trace.is_some() {
        anyhow::ensure!(
            token_ids.len() > 1,
            "final-token layer trace needs a prefix"
        );
        let final_token = *token_ids.last().expect("non-empty token IDs");
        tower.prefill(&token_ids[..token_ids.len() - 1], &mut caches)?;
        tower.set_debug_capture_for_test((0..tower.config.num_hidden_layers).collect())?;
        let final_token = Array::from_slice(&[final_token as i32], &[1, 1]);
        let (mixed, _) = tower.forward(&final_token, Some(&mut caches))?;
        mixed.reshape(&[tower.config.hidden_size as i32])?
    } else {
        tower.prefill(&token_ids, &mut caches)?
    };
    let logits = tower.head(&hidden)?;
    if let Some(path) = final_token_layer_trace {
        tower.flush_debug_capture_for_test(
            Path::new(&path)
                .to_str()
                .ok_or_else(|| anyhow::anyhow!("layer-trace path is not UTF-8"))?,
        )?;
    }
    let lm_head = &tower.lm_head;
    let vocab = usize::try_from(lm_head.dims_out())?;
    anyhow::ensure!(hidden.size() == tower.config.hidden_size && logits.size() == vocab);
    let input = hidden.to_vec1::<f32>()?;
    let packed = lm_head.weight.to_vec1::<u32>()?;
    let scales = lm_head.scales.as_dtype(Dtype::Float32)?.to_vec1::<f32>()?;
    let biases = lm_head.biases.as_dtype(Dtype::Float32)?.to_vec1::<f32>()?;
    let logits = logits.as_dtype(Dtype::Float32)?.to_vec1::<f32>()?;
    std::fs::create_dir_all(&capture_path)?;
    let write_f32 = |name: &str, values: &[f32]| -> anyhow::Result<()> {
        let mut bytes = Vec::with_capacity(values.len() * 4);
        for value in values {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        std::fs::write(capture_path.join(name), bytes)?;
        Ok(())
    };
    let geometry = [
        tower.config.hidden_size as u32,
        vocab as u32,
        lm_head.group_size as u32,
        lm_head.bits as u32,
    ];
    std::fs::write(
        capture_path.join("geometry.u32"),
        geometry
            .into_iter()
            .flat_map(u32::to_le_bytes)
            .collect::<Vec<_>>(),
    )?;
    std::fs::write(
        capture_path.join("input_ids.i32"),
        input_ids
            .iter()
            .flat_map(|&token| token.to_le_bytes())
            .collect::<Vec<_>>(),
    )?;
    std::fs::write(
        capture_path.join("weight.u32"),
        packed
            .into_iter()
            .flat_map(u32::to_le_bytes)
            .collect::<Vec<_>>(),
    )?;
    write_f32("input.f32", &input)?;
    write_f32("scales.f32", &scales)?;
    write_f32("biases.f32", &biases)?;
    write_f32("lisa_logits.f32", &logits)?;
    let argmax = logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(index, _)| index)
        .ok_or_else(|| anyhow::anyhow!("GLM target logits are empty"))?;
    eprintln!(
        "[glm53-target-lm-head] tokens={} vocab={} argmax={argmax}; captured same-weight Q8 head inputs and logits",
        token_ids.len(),
        logits.len()
    );
    Ok(())
}

#[test]
fn glm53_tower_prefix_matches_mlx_reference_stages() -> anyhow::Result<()> {
    if !lisa_mlx::device::metal_available() {
        eprintln!("[glm53-prefix] skipping: no Metal device available");
        return Ok(());
    }
    let Some(model_dir) = std::env::var_os("LISA_GLM53_MODEL") else {
        eprintln!("[glm53-prefix] skipping: LISA_GLM53_MODEL is unset");
        return Ok(());
    };
    let Some(capture_path) = std::env::var_os("LISA_GLM53_CAPTURE") else {
        eprintln!("[glm53-prefix] skipping: LISA_GLM53_CAPTURE is unset");
        return Ok(());
    };
    let Some(reference_dir) = std::env::var_os("LISA_GLM53_MOE_REFERENCE_DIR") else {
        eprintln!("[glm53-prefix] skipping: MLX reference fixture is unset");
        return Ok(());
    };
    let reference_dir = Path::new(&reference_dir);
    let metadata: serde_json::Value =
        serde_json::from_slice(&std::fs::read(reference_dir.join("metadata.json"))?)?;
    let ids: Vec<i32> = metadata["token_ids"]
        .as_array()
        .ok_or_else(|| anyhow::anyhow!("prefix fixture has no token_ids"))?
        .iter()
        .map(|id| {
            id.as_i64()
                .and_then(|id| i32::try_from(id).ok())
                .ok_or_else(|| anyhow::anyhow!("prefix fixture token id is out of range"))
        })
        .collect::<anyhow::Result<_>>()?;
    anyhow::ensure!(!ids.is_empty(), "GLM prefix test has no input ids");
    let dir = Path::new(&model_dir);
    let config = ModelConfig::from_json(&dir.join("config.json"))?;
    let last_layer = metadata["last_layer"]
        .as_u64()
        .ok_or_else(|| anyhow::anyhow!("prefix fixture has no last_layer"))?
        as usize;
    anyhow::ensure!(
        last_layer < config.num_hidden_layers,
        "prefix fixture last layer {last_layer} exceeds the model depth {}",
        config.num_hidden_layers
    );
    anyhow::ensure!(
        metadata["hidden"].as_u64() == Some(config.hidden_size as u64)
            && metadata["tokens"].as_u64() == Some(ids.len() as u64),
        "prefix fixture dimensions do not match the checkpoint"
    );
    let mut tower = load_prefix_tower(dir, &config, last_layer)?;
    anyhow::ensure!(
        tower.capture.is_some(),
        "GLM prefix capture was not enabled"
    );
    let ids = Array::from_slice(&ids, &[1, ids.len() as i32]);
    let mut caches = tower.new_caches();
    if std::env::var_os("LISA_GLM53_CAPTURE_ONLY").is_some() {
        let tokens: Vec<u32> = ids
            .to_vec1::<i32>()?
            .into_iter()
            .map(|token| token as u32)
            .collect();
        let _ = tower.prefill_chunks(&tokens, &mut caches, false)?;
        let capture = tower.capture.as_ref().expect("checked capture");
        capture.flush(
            Path::new(&capture_path)
                .to_str()
                .ok_or_else(|| anyhow::anyhow!("capture path is not UTF-8"))?,
        )?;
        eprintln!(
            "[glm53-prefix] captured chunked prefill layers through {last_layer} without fixture comparison"
        );
        return Ok(());
    }
    let full_depth = last_layer + 1 == tower.config.num_hidden_layers;
    let (mixed, multi) = if full_depth {
        tower.forward_inner(&ids, &mut caches)?
    } else {
        tower.forward_inner_until(&ids, &mut caches, Some(last_layer))?
    };
    anyhow::ensure!(
        mixed.shape() == [1, ids.dim(1), tower.config.hidden_size as i32]
            && multi.shape()
                == [
                    1,
                    ids.dim(1),
                    (tower.config.hidden_size * tower.config.hc_mult) as i32
                ],
        "GLM layer-prefix forward returned invalid hidden shapes"
    );
    if full_depth {
        let logits = tower.head(&mixed)?;
        let capture = tower.capture.as_ref().expect("checked capture");
        capture.store("final_mixed", &mixed);
        capture.store("final_logits", &logits);
    }
    let capture = tower.capture.as_ref().expect("checked capture");
    capture.flush(
        Path::new(&capture_path)
            .to_str()
            .ok_or_else(|| anyhow::anyhow!("capture path is not UTF-8"))?,
    )?;
    let read_f32 = |name: &str| -> anyhow::Result<Vec<f32>> {
        let bytes = std::fs::read(reference_dir.join(name))?;
        anyhow::ensure!(bytes.len() % 4 == 0, "invalid F32 reference file {name}");
        Ok(bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("four-byte chunk")))
            .collect())
    };
    let tokens = ids.dim(1) as usize;
    let top_k = metadata["top_k"]
        .as_u64()
        .ok_or_else(|| anyhow::anyhow!("prefix fixture has no top_k"))? as usize;
    let compared_layers = capture.layers.clone();
    anyhow::ensure!(
        compared_layers.iter().all(|layer| *layer <= last_layer),
        "captured layer selection exceeds fixture prefix"
    );
    let entries = capture.entries.lock().expect("capture entries lock");
    let compare = |stage: &str, key: &str, file: &str, tolerance: f32| -> anyhow::Result<()> {
        let actual = entries
            .get(key)
            .ok_or_else(|| anyhow::anyhow!("missing prefix capture {key}"))?;
        let expected = read_f32(file)?;
        anyhow::ensure!(
            actual.len() == expected.len(),
            "{stage} element count differs: {} vs {}",
            actual.len(),
            expected.len()
        );
        let error = actual
            .iter()
            .zip(&expected)
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f32>()
            .sqrt();
        let reference = expected.iter().map(|x| x * x).sum::<f32>().sqrt();
        let relative = error / reference.max(1e-20);
        anyhow::ensure!(
            relative <= tolerance,
            "{stage} relative L2 error {relative:.6} exceeds {tolerance:.6}"
        );
        eprintln!("[glm53-prefix] {stage}: rel_l2={relative:.6}");
        Ok(())
    };
    let compare_routed_slots = |stage: &str,
                                key: &str,
                                file: &str,
                                width: usize,
                                expected_to_actual: &[Vec<usize>],
                                tolerance: f32|
     -> anyhow::Result<()> {
        let actual = entries
            .get(key)
            .ok_or_else(|| anyhow::anyhow!("missing prefix capture {key}"))?;
        let expected = read_f32(file)?;
        anyhow::ensure!(
            width > 0
                && actual.len() == tokens * top_k * width
                && expected.len() == tokens * top_k * width
                && expected_to_actual.len() == tokens,
            "{stage} routed slot dimensions do not match"
        );
        let mut error_sq = 0.0f32;
        let mut reference_sq = 0.0f32;
        for token in 0..tokens {
            anyhow::ensure!(
                expected_to_actual[token].len() == top_k,
                "{stage} route permutation has the wrong width"
            );
            for expected_slot in 0..top_k {
                let actual_slot = expected_to_actual[token][expected_slot];
                let actual_start = (token * top_k + actual_slot) * width;
                let expected_start = (token * top_k + expected_slot) * width;
                for column in 0..width {
                    let a = actual[actual_start + column];
                    let b = expected[expected_start + column];
                    error_sq += (a - b) * (a - b);
                    reference_sq += b * b;
                }
            }
        }
        let relative = error_sq.sqrt() / reference_sq.sqrt().max(1e-20);
        anyhow::ensure!(
            relative <= tolerance,
            "{stage} relative L2 error {relative:.6} exceeds {tolerance:.6}"
        );
        eprintln!("[glm53-prefix] {stage}: rel_l2={relative:.6} (routes aligned by expert id)");
        Ok(())
    };

    compare("embedding", "embed_out", "embed_out.f32", 0.0017)?;
    for layer in compared_layers {
        let step = if layer == 0 { 0 } else { 1 };
        let sparse = layer >= 3;
        let common_checks = [
            (
                "input norm",
                format!("L{layer}.rep_rms_in_s{step}"),
                format!("l{layer}_input_layernorm.f32"),
                if sparse {
                    0.00001
                } else {
                    [0.0025f32, 0.0018, 0.0015][layer]
                },
            ),
            (
                "attention output",
                format!("L{layer}.rep_attn_out_s{step}"),
                format!("l{layer}_self_attn.f32"),
                if sparse {
                    0.005
                } else {
                    [0.0021f32, 0.0014, 0.0011][layer]
                },
            ),
            (
                "post-attention state",
                format!("L{layer}.state_after_attn"),
                format!("l{layer}_state_after_attn.f32"),
                if sparse {
                    0.002
                } else {
                    [0.0024f32, 0.0021, 0.0020][layer]
                },
            ),
            (
                "FFN input",
                format!("L{layer}.mhc_ffn_xc2"),
                format!("l{layer}_ffn_hc.f32"),
                if sparse {
                    0.003
                } else {
                    [0.0030f32, 0.0026, 0.0022][layer]
                },
            ),
            (
                "post-FFN state",
                format!("L{layer}.state_after_ffn"),
                format!("l{layer}_state_after_ffn.f32"),
                if sparse {
                    0.003
                } else {
                    [0.0022f32, 0.0023, 0.0013][layer]
                },
            ),
        ];
        for (stage, key, file, tolerance) in common_checks {
            compare(&format!("layer {layer} {stage}"), &key, &file, tolerance)?;
        }
        if !sparse {
            compare(
                &format!("layer {layer} FFN output"),
                &format!("L{layer}.rep_moe_out_s{step}"),
                &format!("l{layer}_mlp.f32"),
                [0.0033f32, 0.0038, 0.0011][layer],
            )?;
            continue;
        }

        let selected = entries
            .get(&format!("L{layer}.moe_selected"))
            .ok_or_else(|| anyhow::anyhow!("missing layer-{layer} selected ids"))?;
        anyhow::ensure!(
            selected.len() == tokens * top_k,
            "layer-{layer} selected id count differs from reference"
        );
        let expected_selected_bytes =
            std::fs::read(reference_dir.join(format!("l{layer}_moe_selected.i32")))?;
        anyhow::ensure!(
            expected_selected_bytes.len() == selected.len() * 4,
            "layer-{layer} selected id count differs from reference"
        );
        let expected_selected: Vec<i32> = expected_selected_bytes
            .chunks_exact(4)
            .map(|chunk| i32::from_le_bytes(chunk.try_into().expect("four-byte id")))
            .collect();
        let actual_selected: Vec<i32> = selected.iter().map(|value| *value as i32).collect();
        let mut expected_to_actual = Vec::with_capacity(tokens);
        let mut route_order_differences = 0;
        for token in 0..tokens {
            let actual_row = &actual_selected[token * top_k..(token + 1) * top_k];
            let expected_row = &expected_selected[token * top_k..(token + 1) * top_k];
            let mut sorted_actual = actual_row.to_vec();
            let mut sorted_expected = expected_row.to_vec();
            sorted_actual.sort_unstable();
            sorted_expected.sort_unstable();
            anyhow::ensure!(
                sorted_actual == sorted_expected,
                "layer-{layer} selected expert set differs at token {token}: {actual_row:?} vs {expected_row:?}"
            );
            if actual_row != expected_row {
                route_order_differences += 1;
            }
            let mut permutation = Vec::with_capacity(top_k);
            for expected_id in expected_row {
                let actual_slot = actual_row
                    .iter()
                    .position(|actual_id| actual_id == expected_id)
                    .expect("selected expert sets were checked equal");
                permutation.push(actual_slot);
            }
            expected_to_actual.push(permutation);
        }
        eprintln!(
            "[glm53-prefix] layer {layer} selected expert sets: exact; route-order differences={route_order_differences}/{tokens}"
        );

        let sparse_checks = [
            ("MoE input", "moe_input", "moe_input", 0.003),
            ("router", "moe_router_logits", "moe_router_logits", 0.001),
            ("routed sum", "moe_routed", "moe_routed", 0.003),
            ("shared expert", "moe_shared", "moe_shared", 0.002),
            ("combined MoE", "rep_moe_out_s", "moe_output", 0.003),
        ];
        for (stage, actual_suffix, file_suffix, tolerance) in sparse_checks {
            let actual_key = if actual_suffix == "rep_moe_out_s" {
                format!("L{layer}.{actual_suffix}{step}")
            } else {
                format!("L{layer}.{actual_suffix}")
            };
            compare(
                &format!("layer {layer} {stage}"),
                &actual_key,
                &format!("l{layer}_{file_suffix}.f32"),
                tolerance,
            )?;
        }
        compare_routed_slots(
            &format!("layer {layer} route weights"),
            &format!("L{layer}.moe_weights"),
            &format!("l{layer}_moe_weights.f32"),
            1,
            &expected_to_actual,
            0.001,
        )?;
        compare_routed_slots(
            &format!("layer {layer} weighted pair"),
            &format!("L{layer}.moe_pair_weighted"),
            &format!("l{layer}_moe_pair_weighted.f32"),
            metadata["intermediate"].as_u64().unwrap() as usize,
            &expected_to_actual,
            0.003,
        )?;
        compare_routed_slots(
            &format!("layer {layer} weighted down"),
            &format!("L{layer}.moe_down_slots"),
            &format!("l{layer}_moe_down_weighted.f32"),
            tower.config.hidden_size,
            &expected_to_actual,
            0.003,
        )?;
    }
    Ok(())
}

fn load_checkpoint_subset(dir: &Path) -> anyhow::Result<Weights> {
    let checkpoint = loader::Checkpoint::open(dir)?;
    let wanted = |name: &str| {
        name.starts_with("model.layers.45.")
            || name.starts_with("model.embed_tokens.")
            || name.starts_with("lm_head.")
    };
    let mut by_shard: HashMap<String, Vec<(String, String)>> = HashMap::new();
    for (raw, shard) in &checkpoint.weight_map {
        let Some(sanitized) = loader::sanitize_name(raw) else {
            continue;
        };
        if wanted(&sanitized) {
            by_shard
                .entry(shard.clone())
                .or_default()
                .push((raw.clone(), sanitized));
        }
    }

    let mut weights = Weights::new();
    for (shard_name, names) in by_shard {
        let shard = loader::Shard::open(&dir.join(shard_name))?;
        for (raw, sanitized) in names {
            anyhow::ensure!(
                !weights.contains_key(&sanitized),
                "duplicate selected GLM tensor {sanitized}"
            );
            weights.insert(sanitized, crate::core::quant::get_tensor(&shard, &raw)?);
        }
    }
    Ok(weights)
}

#[test]
fn glm53_mtp_advances_recent_window_with_real_mlx_checkpoint_weights() -> anyhow::Result<()> {
    if !lisa_mlx::device::metal_available() {
        eprintln!("[glm53-mtp-mlx-weight] skipping: no Metal device available");
        return Ok(());
    }
    let Some(model_dir) = std::env::var_os("LISA_GLM53_MODEL") else {
        eprintln!("[glm53-mtp-mlx-weight] skipping: LISA_GLM53_MODEL is unset");
        return Ok(());
    };
    let model_dir = Path::new(&model_dir);
    let config = ModelConfig::from_json(&model_dir.join("config.json"))?;
    let mut weights = load_checkpoint_subset(model_dir)?;

    let mut embed = QuantizedEmbedding::load(&mut weights, "model.embed_tokens")?;
    embed.bits = 8;
    embed.group_size = 64;
    embed.scales = embed.scales.as_dtype(Dtype::Float32)?;
    embed.biases = embed.biases.as_dtype(Dtype::Float32)?;
    let mut lm_head = QuantizedLinear::load(&mut weights, "lm_head", "")?;
    lm_head.set_quant(64, 8);
    lm_head.scales = lm_head.scales.as_dtype(Dtype::Float32)?;
    lm_head.biases = lm_head.biases.as_dtype(Dtype::Float32)?;
    let mtp = MtpBlock::load(&mut weights, &config, None, false)?;

    // Check the real plain-BF16 eh_proj tensor against ds4's row-GEMV
    // accumulation schedule independently of the surrounding MTP graph.
    // This isolates its checkpoint dtype/loader mapping and Metal dispatch.
    anyhow::ensure!(
        mtp.eh_proj.shape() == [config.hidden_size as i32, (2 * config.hidden_size) as i32],
        "GLM MTP eh_proj has unexpected shape {:?}",
        mtp.eh_proj.shape()
    );
    let streams = config.hc_mult;
    let target_width = config.hidden_size * streams;
    let target_values: Vec<f32> = (0..target_width)
        .map(|i| (((i * 37 + 11) % 251) as f32 - 125.0) * 0.0003)
        .collect();
    let target_multi = Array::from_slice(&target_values, &[1, 1, target_width as i32]);
    anyhow::ensure!(streams == 4, "GLM MTP fixture expects four mHC streams");
    let target_streams = target_multi.as_dtype(Dtype::Float32)?.reshape(&[
        1,
        1,
        streams as i32,
        config.hidden_size as i32,
    ])?;
    let mean_weights = Array::from_slice(&[0.25f32; 4], &[1, 1, 4]);
    let stream = lisa_mlx::Stream::thread_local_or_default();
    let target_hidden =
        lisa_mlx::models::glm53::kernels::hc_weighted_sum(&target_streams, &mean_weights, &stream)
            .ok_or_else(|| anyhow::anyhow!("GLM MTP fixture target reduction declined"))?;
    let target_hidden = mtp_rms(&target_hidden, &mtp.hnorm, config.rms_norm_eps)?;
    let token_ids = Array::from_slice(&[42i32], &[1, 1]);
    let embedded = embed.forward(&token_ids)?.as_dtype(Dtype::Float32)?;
    let embedded = mtp_rms(&embedded, &mtp.enorm, config.rms_norm_eps)?;
    let eh_input = Array::concatenate(&[&embedded, &target_hidden], 2)?.to_vec1::<f32>()?;
    let eh_input_array = Array::from_slice(&eh_input, &[1, eh_input.len() as i32]);
    let stream = lisa_mlx::Stream::thread_local_or_default();
    let eh_actual =
        lisa_mlx::models::glm53::kernels::glm53_bf16_gemv(&mtp.eh_proj, &eh_input_array, &stream)
            .ok_or_else(|| anyhow::anyhow!("GLM MTP real-weight eh_proj GEMV declined"))?
            .to_vec1::<f32>()?;
    let eh_weights = mtp.eh_proj.as_dtype(Dtype::Float32)?.to_vec1::<f32>()?;
    if let Some(path) = std::env::var_os("LISA_GLM53_MTP_EH_CAPTURE_DIR") {
        let path = Path::new(&path);
        std::fs::create_dir_all(path)?;
        let write_f32 = |name: &str, values: &[f32]| -> anyhow::Result<()> {
            let bytes: Vec<u8> = values
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect();
            std::fs::write(path.join(name), bytes)?;
            Ok(())
        };
        write_f32("eh_weight.f32", &eh_weights)?;
        write_f32("eh_input.f32", &eh_input)?;
        write_f32("lisa_eh_output.f32", &eh_actual)?;
    }
    let in_dim = eh_input.len();
    let mut eh_expected = vec![0.0f32; config.hidden_size];
    for out_row in 0..config.hidden_size {
        let row = &eh_weights[out_row * in_dim..(out_row + 1) * in_dim];
        let mut lanes = [0.0f32; 32];
        for lane in 0..32 {
            let mut k = lane;
            while k + 224 < in_dim {
                for offset in (0..256).step_by(32) {
                    lanes[lane] = row[k + offset].mul_add(eh_input[k + offset], lanes[lane]);
                }
                k += 256;
            }
            while k < in_dim {
                lanes[lane] = row[k].mul_add(eh_input[k], lanes[lane]);
                k += 32;
            }
        }
        eh_expected[out_row] = lanes.into_iter().sum();
    }
    let max_abs = eh_actual
        .iter()
        .zip(&eh_expected)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0f32, f32::max);
    anyhow::ensure!(
        max_abs <= 2e-4,
        "GLM MTP real-weight eh_proj differs from ds4 row-GEMV schedule: max abs {max_abs}"
    );
    let loaded_eh_proj_fixture: &[u8] =
        &crate::models::glm53::test_fixtures::read("mtp_eh_proj_ds4.bin");
    if std::env::var_os("LISA_GLM53_MTP_EH_CAPTURE_DIR").is_none() {
        anyhow::ensure!(
            loaded_eh_proj_fixture.len() == config.hidden_size * std::mem::size_of::<f32>(),
            "DS4 MTP eh_proj fixture has an invalid size"
        );
        let expected_ds4: Vec<f32> = loaded_eh_proj_fixture
            .chunks_exact(4)
            .map(|bytes| f32::from_le_bytes(bytes.try_into().expect("F32 fixture word")))
            .collect();
        anyhow::ensure!(
            eh_actual
                .iter()
                .map(|value| value.to_bits())
                .eq(expected_ds4.iter().map(|value| value.to_bits())),
            "real-weight MTP eh_proj differs from direct DS4 Metal BF16 GEMV"
        );
    }

    // Exercise MTP's real layer-45 DSA with the just-verified eh_proj
    // activation. A private five-row tape makes the contiguous MTP
    // attention window directly comparable with ds4's Metal stages.
    let mtp_attn_input = Array::from_slice(&eh_actual, &[1, 1, config.hidden_size as i32]);
    let mtp_attn_input = mtp_rms(&mtp_attn_input, &mtp.attn_norm, config.rms_norm_eps)?;
    let mut mtp_dsa_tape = IndexerTape::new();
    for position in 0..5 {
        let _ = mtp
            .attn
            .forward_mtp(&mtp_attn_input, &mut mtp_dsa_tape, position, 0)?;
    }

    let mut tape = IndexerTape::new();
    let h = config.hidden_size * config.hc_mult;
    let mut last_values = Vec::new();
    for (step, (position, min_position)) in [(15, 15), (16, 15), (17, 15), (18, 15), (19, 15)]
        .into_iter()
        .enumerate()
    {
        let target_values: Vec<f32> = (0..h)
            .map(|i| (((i * 37 + 11 + step * 19) % 251) as f32 - 125.0) * 0.0003)
            .collect();
        let target_multi = Array::from_slice(&target_values, &[1, 1, h as i32]);
        let logits = mtp.forward(
            &config,
            &embed,
            &lm_head,
            &mut tape,
            &target_multi,
            42 + step as u32,
            position,
            min_position,
        )?;
        anyhow::ensure!(
            logits.shape() == [1, 1, config.vocab_size as i32],
            "unexpected GLM MTP logits shape {:?}",
            logits.shape()
        );
        last_values = logits.to_vec1::<f32>()?;
        anyhow::ensure!(
            last_values.len() == config.vocab_size
                && last_values.iter().all(|value| value.is_finite()),
            "real-weight GLM MTP produced invalid logits at position {position}"
        );
    }
    eprintln!(
        "[glm53-mtp-mlx-weight] five recent-window MTP forwards passed; final argmax={}",
        last_values
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(index, _)| index)
            .unwrap_or(0)
    );

    // Continue one step with the previous MTP argmax as its token input.
    // The earlier five forwards keep the pinned teacher-forced fixture
    // sequence; this extra transition checks the generated-token feedback
    // path while extending the same recent-window cache.
    let feedback_token = last_values
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(index, _)| index)
        .ok_or_else(|| anyhow::anyhow!("real-weight GLM MTP produced empty logits"))?;
    let target_values: Vec<f32> = (0..h)
        .map(|i| (((i * 37 + 11 + 4 * 19) % 251) as f32 - 125.0) * 0.0003)
        .collect();
    let target_multi = Array::from_slice(&target_values, &[1, 1, h as i32]);
    let feedback_logits = mtp.forward(
        &config,
        &embed,
        &lm_head,
        &mut tape,
        &target_multi,
        feedback_token as u32,
        20,
        15,
    )?;
    let feedback_logits = feedback_logits.to_vec1::<f32>()?;
    anyhow::ensure!(
        feedback_logits.len() == config.vocab_size
            && feedback_logits.iter().all(|value| value.is_finite()),
        "generated-token MTP feedback produced invalid logits"
    );
    eprintln!(
        "[glm53-mtp-mlx-weight] generated-token feedback token={feedback_token}; next argmax={}",
        feedback_logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(index, _)| index)
            .unwrap_or(0)
    );
    Ok(())
}

#[cfg(feature = "glm53-fixtures")]
#[test]
fn glm53_mtp_eh_proj_matches_ds4_with_full_model_target_hidden() -> anyhow::Result<()> {
    if !lisa_mlx::device::metal_available() {
        eprintln!("[glm53-mtp-real-target] skipping: no Metal device available");
        return Ok(());
    }
    let Some(model_dir) = std::env::var_os("LISA_GLM53_MODEL") else {
        eprintln!("[glm53-mtp-real-target] skipping: LISA_GLM53_MODEL is unset");
        return Ok(());
    };
    let model_dir = Path::new(&model_dir);
    let config = ModelConfig::from_json(&model_dir.join("config.json"))?;
    let mut weights = load_checkpoint_subset(model_dir)?;
    let mut embed = QuantizedEmbedding::load(&mut weights, "model.embed_tokens")?;
    embed.bits = 8;
    embed.group_size = 64;
    embed.scales = embed.scales.as_dtype(Dtype::Float32)?;
    embed.biases = embed.biases.as_dtype(Dtype::Float32)?;
    let mut lm_head = QuantizedLinear::load(&mut weights, "lm_head", "")?;
    lm_head.set_quant(64, 8);
    lm_head.scales = lm_head.scales.as_dtype(Dtype::Float32)?;
    lm_head.biases = lm_head.biases.as_dtype(Dtype::Float32)?;
    let mtp = MtpBlock::load(&mut weights, &config, None, false)?;

    let loaded_target_multi: &[u8] =
        &crate::models::glm53::test_fixtures::read("mtp_target_hidden_natural_ids_pos0.f32");
    anyhow::ensure!(
        loaded_target_multi.len() == config.hidden_size * config.hc_mult * 4,
        "captured full-model target mHC row has an invalid size"
    );
    let target_values: Vec<f32> = loaded_target_multi
        .chunks_exact(4)
        .map(|bytes| f32::from_le_bytes(bytes.try_into().expect("F32 target value")))
        .collect();
    let target_multi = Array::from_slice(
        &target_values,
        &[1, 1, (config.hidden_size * config.hc_mult) as i32],
    );
    anyhow::ensure!(
        config.hc_mult == 4,
        "captured full-model target row expects four mHC streams"
    );
    let target_streams =
        target_multi.reshape(&[1, 1, config.hc_mult as i32, config.hidden_size as i32])?;
    let mean_weights = Array::from_slice(&[0.25f32; 4], &[1, 1, 4]);
    let stream = lisa_mlx::Stream::thread_local_or_default();
    let target_hidden =
        lisa_mlx::models::glm53::kernels::hc_weighted_sum(&target_streams, &mean_weights, &stream)
            .ok_or_else(|| anyhow::anyhow!("GLM MTP target reduction declined"))?;
    let target_hidden = mtp_rms(&target_hidden, &mtp.hnorm, config.rms_norm_eps)?;

    // The saved trunk state is for natural prompt token 785. Its next
    // teacher-forced token is 3974, so the MTP input uses that exact pair.
    let token_ids = Array::from_slice(&[3974i32], &[1, 1]);
    let embedded = embed.forward(&token_ids)?.as_dtype(Dtype::Float32)?;
    let embedded = mtp_rms(&embedded, &mtp.enorm, config.rms_norm_eps)?;
    let eh_input = Array::concatenate(&[&embedded, &target_hidden], 2)?.to_vec1::<f32>()?;
    let eh_input_array = Array::from_slice(&eh_input, &[1, eh_input.len() as i32]);
    let stream = lisa_mlx::Stream::thread_local_or_default();
    let eh_actual =
        lisa_mlx::models::glm53::kernels::glm53_bf16_gemv(&mtp.eh_proj, &eh_input_array, &stream)
            .ok_or_else(|| anyhow::anyhow!("GLM MTP real-target eh_proj GEMV declined"))?
            .to_vec1::<f32>()?;
    let eh_weights = mtp.eh_proj.as_dtype(Dtype::Float32)?.to_vec1::<f32>()?;

    if let Some(path) = std::env::var_os("LISA_GLM53_MTP_REAL_EH_CAPTURE_DIR") {
        let path = Path::new(&path);
        std::fs::create_dir_all(path)?;
        let write_f32 = |name: &str, values: &[f32]| -> anyhow::Result<()> {
            let bytes: Vec<u8> = values
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect();
            std::fs::write(path.join(name), bytes)?;
            Ok(())
        };
        write_f32("eh_weight.f32", &eh_weights)?;
        write_f32("eh_input.f32", &eh_input)?;
        write_f32("lisa_eh_output.f32", &eh_actual)?;
        return Ok(());
    }

    let loaded_ds4_eh: &[u8] =
        &crate::models::glm53::test_fixtures::read("mtp_eh_proj_natural_target_ds4.bin");
    anyhow::ensure!(
        eh_actual.len() == config.hidden_size
            && loaded_ds4_eh.len() == config.hidden_size * std::mem::size_of::<f32>(),
        "real-target MTP eh_proj fixture shape mismatch"
    );
    for (index, (actual, expected)) in eh_actual
        .iter()
        .zip(loaded_ds4_eh.chunks_exact(4))
        .enumerate()
    {
        let expected = f32::from_le_bytes(expected.try_into().expect("F32 DS4 value"));
        anyhow::ensure!(
            actual.to_bits() == expected.to_bits(),
            "real-target MTP eh_proj differs from DS4 at {index}: {actual} vs {expected}"
        );
    }

    // Replay five consecutive target-produced trunk rows with their
    // following teacher-forced tokens, preserving their absolute prompt
    // positions in the private MTP DSA cache.
    let loaded_target_sequence: &[u8] =
        &crate::models::glm53::test_fixtures::read("mtp_target_hidden_natural_ids_pos0to4.f32");
    const NEXT_TOKENS: [i32; 5] = [3974, 13867, 38627, 34041, 916];
    let target_width = config.hidden_size * config.hc_mult;
    anyhow::ensure!(
        loaded_target_sequence.len() == 5 * target_width * std::mem::size_of::<f32>(),
        "captured full-model target sequence has an invalid size"
    );
    let target_sequence: Vec<f32> = loaded_target_sequence
        .chunks_exact(4)
        .map(|bytes| f32::from_le_bytes(bytes.try_into().expect("F32 target value")))
        .collect();
    let _fixture_scope = crate::models::glm53::sparse::use_mtp_real_target_fixtures();
    let run_target_row = |tape: &mut IndexerTape, row: usize| -> anyhow::Result<Array> {
        let target_multi = Array::from_slice(
            &target_sequence[row * target_width..(row + 1) * target_width],
            &[1, 1, target_width as i32],
        );
        let target_streams =
            target_multi.reshape(&[1, 1, config.hc_mult as i32, config.hidden_size as i32])?;
        let mean_weights = Array::from_slice(&[0.25f32; 4], &[1, 1, 4]);
        let stream = lisa_mlx::Stream::thread_local_or_default();
        let target_hidden = lisa_mlx::models::glm53::kernels::hc_weighted_sum(
            &target_streams,
            &mean_weights,
            &stream,
        )
        .ok_or_else(|| anyhow::anyhow!("GLM MTP target reduction declined"))?;
        let target_hidden = mtp_rms(&target_hidden, &mtp.hnorm, config.rms_norm_eps)?;
        let token_ids = Array::from_slice(&[NEXT_TOKENS[row]], &[1, 1]);
        let embedded = embed.forward(&token_ids)?.as_dtype(Dtype::Float32)?;
        let embedded = mtp_rms(&embedded, &mtp.enorm, config.rms_norm_eps)?;
        let eh_input = Array::concatenate(&[&embedded, &target_hidden], 2)?
            .reshape(&[1, (2 * config.hidden_size) as i32])?;
        let stream = lisa_mlx::Stream::thread_local_or_default();
        let eh =
            lisa_mlx::models::glm53::kernels::glm53_bf16_gemv(&mtp.eh_proj, &eh_input, &stream)
                .ok_or_else(|| anyhow::anyhow!("GLM MTP real-target eh_proj GEMV declined"))?;
        let eh = eh.reshape(&[1, 1, config.hidden_size as i32])?;
        let mtp_attn_input = mtp_rms(&eh, &mtp.attn_norm, config.rms_norm_eps)?;
        mtp.attn.forward_mtp(&mtp_attn_input, tape, row + 1, 1)
    };
    let mut tape = IndexerTape::new();
    let mut last_attention = None;
    let mut replay_snapshot = None;
    let mut uninterrupted_tail = Vec::new();
    for row in 0..5 {
        let attention = run_target_row(&mut tape, row)?;
        if row == 1 {
            replay_snapshot = Some(tape.snapshot());
        }
        if row >= 2 {
            uninterrupted_tail.push(attention.to_vec1::<f32>()?);
        }
        last_attention = Some(attention);
    }
    let replay_snapshot = replay_snapshot.expect("mid-window target cache snapshot");
    let mut replayed_tape = IndexerTape::new();
    replayed_tape.restore(&replay_snapshot);
    for (tail_index, row) in (2..5).enumerate() {
        let replayed = run_target_row(&mut replayed_tape, row)?.to_vec1::<f32>()?;
        anyhow::ensure!(
            replayed.len() == uninterrupted_tail[tail_index].len()
                && replayed
                    .iter()
                    .zip(&uninterrupted_tail[tail_index])
                    .all(|(actual, expected)| actual.to_bits() == expected.to_bits()),
            "real-weight MTP cache restore/replay differs at target row {row}"
        );
    }
    // Compose the final MTP output from the replayed cache branch. The
    // downstream DS4 fixture checks the restored tape's numeric path too.
    tape = replayed_tape;
    let last_attention = last_attention.expect("five MTP DSA positions were forwarded");
    anyhow::ensure!(
        last_attention.shape() == [1, 1, config.hidden_size as i32],
        "real-target MTP DSA returned unexpected shape {:?}",
        last_attention.shape()
    );
    let final_target_multi = Array::from_slice(
        &target_sequence[4 * target_width..5 * target_width],
        &[1, 1, target_width as i32],
    );
    let logits = mtp.forward(
        &config,
        &embed,
        &lm_head,
        &mut tape,
        &final_target_multi,
        NEXT_TOKENS[4] as u32,
        5,
        1,
    )?;
    let logits = logits.to_vec1::<f32>()?;
    anyhow::ensure!(
        logits.len() == config.vocab_size && logits.iter().all(|value| value.is_finite()),
        "composed real-target MTP produced invalid logits"
    );
    let argmax = logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(index, _)| index)
        .ok_or_else(|| anyhow::anyhow!("composed real-target MTP produced empty logits"))?;
    eprintln!("[glm53-mtp-real-target] composed DS4 logits match at position 5; argmax={argmax}");
    let feedback_logits = mtp.forward(
        &config,
        &embed,
        &lm_head,
        &mut tape,
        &final_target_multi,
        argmax as u32,
        6,
        1,
    )?;
    let feedback_logits = feedback_logits.to_vec1::<f32>()?;
    anyhow::ensure!(
        feedback_logits.len() == config.vocab_size
            && feedback_logits.iter().all(|value| value.is_finite()),
        "generated real-target MTP feedback produced invalid logits"
    );
    let feedback_argmax = feedback_logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(index, _)| index)
        .ok_or_else(|| anyhow::anyhow!("generated real-target MTP logits are empty"))?;
    eprintln!(
        "[glm53-mtp-real-target] generated feedback token={argmax}; next argmax={feedback_argmax}"
    );
    eprintln!("[glm53-mtp-real-target] eh_proj matches DS4 bit-for-bit");
    eprintln!("[glm53-mtp-real-target] five varying target rows passed through MTP DSA");
    Ok(())
}

#[test]
fn glm53_capture_dsa_indexer_from_saved_layer_state() -> anyhow::Result<()> {
    if !lisa_mlx::device::metal_available() {
        eprintln!("[glm53-indexer-capture] skipping: no Metal device available");
        return Ok(());
    }
    let Some(model_dir) = std::env::var_os("LISA_GLM53_MODEL") else {
        eprintln!("[glm53-indexer-capture] skipping: LISA_GLM53_MODEL is unset");
        return Ok(());
    };
    let Some(state_path) = std::env::var_os("LISA_GLM53_LAYER_INPUT_STATE") else {
        eprintln!("[glm53-indexer-capture] skipping: saved layer state is unset");
        return Ok(());
    };
    let Some(capture_path) = std::env::var_os("LISA_GLM53_CAPTURE") else {
        eprintln!("[glm53-indexer-capture] skipping: LISA_GLM53_CAPTURE is unset");
        return Ok(());
    };

    let dir = Path::new(&model_dir);
    let config = ModelConfig::from_json(&dir.join("config.json"))?;
    let layer = std::env::var("LISA_GLM53_LAYER")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(3);
    let tower = load_prefix_tower(dir, &config, layer)?;
    let width = config
        .hidden_size
        .checked_mul(config.hc_mult)
        .ok_or_else(|| anyhow::anyhow!("GLM layer-state width overflow"))?;
    let state_bytes = std::fs::read(state_path)?;
    anyhow::ensure!(
        state_bytes.len() % std::mem::size_of::<f32>() == 0,
        "saved GLM layer state is not F32-aligned"
    );
    let state_values: Vec<f32> = state_bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("four-byte chunk")))
        .collect();
    anyhow::ensure!(
        state_values.len() % width == 0,
        "saved GLM layer state has an invalid width"
    );
    let tokens = state_values.len() / width;
    let state = Array::from_slice(
        &state_values,
        &[
            1,
            tokens as i32,
            config.hc_mult as i32,
            config.hidden_size as i32,
        ],
    );
    let (x, dsa) = {
        let current = &tower.layers[layer];
        let (_, x, _, _) = current
            .hc_attn
            .split_normed(&state, &current.in_norm, config.rms_norm_eps)?
            .ok_or_else(|| anyhow::anyhow!("GLM mHC split-norm path declined"))?;
        let dsa = match &current.attn {
            Attn::Dsa(dsa) => dsa,
            Attn::Kda(_) => anyhow::bail!("requested layer {layer} is KDA, not DSA"),
        };
        (x, dsa)
    };
    let capture = tower
        .capture
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("GLM capture was not enabled"))?;
    anyhow::ensure!(
        capture.wants(layer),
        "capture does not include layer {layer}"
    );
    capture.store(&format!("L{layer}.dsa_indexer_input_x"), &x);
    let mut caches = tower.new_caches();
    let cache = match &mut caches[layer] {
        LayerCache::Full(cache) => cache,
        LayerCache::Linear(_) | LayerCache::Native(_) => {
            anyhow::bail!("DSA cache mismatch at layer {layer}")
        }
    };
    let _ = dsa.forward(&x, cache, Some(capture), layer, true)?;
    capture.flush(
        Path::new(&capture_path)
            .to_str()
            .ok_or_else(|| anyhow::anyhow!("capture path is not UTF-8"))?,
    )?;
    eprintln!("[glm53-indexer-capture] captured layer {layer} from saved state");
    Ok(())
}

#[test]
fn diagnose_prefill_chunk_boundary_vs_one_shot() -> anyhow::Result<()> {
    if !lisa_mlx::device::metal_available() {
        eprintln!("[glm53-prefill-chunks] skipping: no Metal device available");
        return Ok(());
    }
    let Some(model_dir) = std::env::var_os("LISA_GLM53_MODEL") else {
        eprintln!("[glm53-prefill-chunks] skipping: LISA_GLM53_MODEL is unset");
        return Ok(());
    };
    let dir = Path::new(&model_dir);
    let config = ModelConfig::from_json(&dir.join("config.json"))?;
    let mut tower = load_prefix_tower(dir, &config, 3)?;
    let tokenizer = crate::core::tokenizer::Tokenizer::load_file(&dir.join("tokenizer.json"))?;
    let seed = tokenizer.encode("The capital of France is Paris. ", false)?;
    anyhow::ensure!(!seed.is_empty());
    let tokens: Vec<u32> = (0..DS4_GLM53_PREFILL_CHUNK_TOKENS + 1)
        .map(|index| seed[index % seed.len()])
        .collect();
    let ids = i32_arr(&tower.embed.weight, &tokens)?;

    let mut one_shot_caches = tower.new_caches();
    let (one_shot_mixed, _one_shot_multi) = tower.forward_inner(&ids, &mut one_shot_caches)?;
    let expected_last = narrow(&one_shot_mixed, 1, one_shot_mixed.dim(1) as usize - 1, 1)?
        .reshape(&[1, config.hidden_size as i32])?
        .to_vec1::<f32>()?;

    let mut chunked_caches = tower.new_caches();
    let (actual_last, retained_multi) =
        tower.prefill_chunks(&tokens, &mut chunked_caches, false)?;
    anyhow::ensure!(retained_multi.is_none());
    let actual_last_values = actual_last.to_vec1::<f32>()?;

    // Exercise the exact sequence selected by DS4's GLM session loop
    // independently of the production chunk driver. This distinguishes a
    // chunk-driver/cache bug from the expected difference versus a single
    // 2,049-row forward.
    let mut explicit_caches = tower.new_caches();
    let mut explicit_last = None;
    for chunk in tokens.chunks(DS4_GLM53_PREFILL_CHUNK_TOKENS) {
        let chunk_ids = i32_arr(&tower.embed.weight, chunk)?;
        let (chunk_mixed, _) = tower.forward_inner(&chunk_ids, &mut explicit_caches)?;
        explicit_last = Some(
            narrow(&chunk_mixed, 1, chunk_mixed.dim(1) as usize - 1, 1)?
                .reshape(&[1, config.hidden_size as i32])?
                .to_vec1::<f32>()?,
        );
    }
    let explicit_last = explicit_last.expect("non-empty explicit prefill chunks");
    anyhow::ensure!(
        actual_last_values.len() == explicit_last.len()
            && actual_last_values
                .iter()
                .zip(&explicit_last)
                .all(|(actual, expected)| actual.to_bits() == expected.to_bits()),
        "production chunk driver differs from explicit DS4-sized forwards"
    );

    fn assert_array_equal(a: &Array, b: &Array, label: &str) -> anyhow::Result<()> {
        anyhow::ensure!(a.shape() == b.shape(), "{label} shape differs");
        let a = a.as_dtype(Dtype::Float32)?.to_vec1::<f32>()?;
        let b = b.as_dtype(Dtype::Float32)?.to_vec1::<f32>()?;
        anyhow::ensure!(
            a.iter().zip(&b).all(|(a, b)| a.to_bits() == b.to_bits()),
            "{label} values differ"
        );
        Ok(())
    }
    fn assert_optional_arrays_equal(
        a: Option<Array>,
        b: Option<Array>,
        label: &str,
    ) -> anyhow::Result<()> {
        match (a, b) {
            (Some(a), Some(b)) => assert_array_equal(&a, &b, label),
            (None, None) => Ok(()),
            _ => anyhow::bail!("{label} presence differs"),
        }
    }
    anyhow::ensure!(chunked_caches.len() == explicit_caches.len());
    for (layer, (actual, expected)) in chunked_caches.iter().zip(&explicit_caches).enumerate() {
        match (actual, expected) {
            (LayerCache::Linear(actual), LayerCache::Linear(expected)) => {
                assert_optional_arrays_equal(
                    actual.conv.clone(),
                    expected.conv.clone(),
                    &format!("layer {layer} KDA convolution cache"),
                )?;
                assert_optional_arrays_equal(
                    actual.ssm.clone(),
                    expected.ssm.clone(),
                    &format!("layer {layer} KDA recurrent cache"),
                )?;
                assert_optional_arrays_equal(
                    actual.ple_conv.clone(),
                    expected.ple_conv.clone(),
                    &format!("layer {layer} PLE convolution cache"),
                )?;
            }
            (LayerCache::Full(actual), LayerCache::Full(expected)) => {
                anyhow::ensure!(
                    actual.offset == expected.offset,
                    "layer {layer} DSA cache offsets differ"
                );
                assert_optional_arrays_equal(
                    actual.keys.clone(),
                    expected.keys.clone(),
                    &format!("layer {layer} DSA key cache"),
                )?;
                assert_optional_arrays_equal(
                    actual.values.clone(),
                    expected.values.clone(),
                    &format!("layer {layer} DSA value cache"),
                )?;
                assert_optional_arrays_equal(
                    actual.indexer_tape.raw_live_for_test(),
                    expected.indexer_tape.raw_live_for_test(),
                    &format!("layer {layer} DSA indexer raw cache"),
                )?;
                assert_optional_arrays_equal(
                    actual.indexer_tape.compact_live_for_test(),
                    expected.indexer_tape.compact_live_for_test(),
                    &format!("layer {layer} DSA compact cache"),
                )?;
                assert_optional_arrays_equal(
                    actual.indexer_tape.pooled_view(),
                    expected.indexer_tape.pooled_view(),
                    &format!("layer {layer} DSA pooled cache"),
                )?;
                assert_optional_arrays_equal(
                    actual.ple_conv.clone(),
                    expected.ple_conv.clone(),
                    &format!("layer {layer} DSA PLE convolution cache"),
                )?;
            }
            _ => anyhow::bail!("layer {layer} cache kind differs"),
        }
    }

    let mut output_mismatches = 0usize;
    let mut output_max_abs = 0.0f32;
    let mut output_reference_norm = 0.0f64;
    let mut output_delta_norm = 0.0f64;
    for (&actual, &expected) in actual_last_values.iter().zip(&expected_last) {
        output_mismatches += usize::from(actual.to_bits() != expected.to_bits());
        let delta = (actual - expected).abs();
        output_max_abs = output_max_abs.max(delta);
        output_delta_norm += f64::from(delta) * f64::from(delta);
        output_reference_norm += f64::from(expected) * f64::from(expected);
    }
    eprintln!(
        "[glm53-prefill-chunks] one-shot output delta: {output_mismatches}/{} values, max_abs={output_max_abs:.8e}, rel_l2={:.8e}",
        actual_last_values.len(),
        (output_delta_norm / output_reference_norm).sqrt()
    );

    anyhow::ensure!(one_shot_caches.len() == chunked_caches.len());
    eprintln!(
        "[glm53-prefill-chunks] diagnostic completed for {} rows; one-shot is not the DS4 chunked-path oracle",
        tokens.len(),
    );
    Ok(())
}

#[cfg(feature = "glm53-fixtures")]
#[test]
fn capture_real_16k_dsa_decode_selector_inputs_for_ds4() -> anyhow::Result<()> {
    const HISTORY: usize = 16_384;
    let target_layer = match std::env::var("LISA_GLM53_DSA_CAPTURE_LAYER") {
        Ok(value) => value.parse::<usize>()?,
        Err(_) => 3,
    };
    let Some(output_dir) = std::env::var_os("LISA_GLM53_LONG_DSA_CAPTURE_DIR") else {
        eprintln!("LISA_GLM53_LONG_DSA_CAPTURE_DIR=<dir> to capture a real 16K DSA selector step");
        return Ok(());
    };
    if !lisa_mlx::device::metal_available() {
        eprintln!("[glm53-long-indexer] skipping: no Metal device available");
        return Ok(());
    }
    let Some(model_dir) = std::env::var_os("LISA_GLM53_MODEL") else {
        eprintln!("[glm53-long-indexer] skipping: LISA_GLM53_MODEL is unset");
        return Ok(());
    };
    let output_dir = Path::new(&output_dir);
    std::fs::create_dir_all(output_dir)?;
    let dir = Path::new(&model_dir);
    let config = ModelConfig::from_json(&dir.join("config.json"))?;
    anyhow::ensure!(
        config.num_hidden_layers > target_layer && !config.is_kda(target_layer),
        "GLM layer {target_layer} is not a configured DSA layer"
    );
    let tokenizer = crate::core::tokenizer::Tokenizer::load_file(&dir.join("tokenizer.json"))?;
    let seed = tokenizer.encode("The capital of France is Paris. ", false)?;
    anyhow::ensure!(
        !seed.is_empty(),
        "long-window selector prompt tokenized to empty input"
    );
    let mut ids = Vec::with_capacity(HISTORY);
    while ids.len() < HISTORY {
        let take = (HISTORY - ids.len()).min(seed.len());
        ids.extend_from_slice(&seed[..take]);
    }
    let query_id = seed[HISTORY % seed.len()];

    // DS4 caps GLM prefill at 2048 rows. Preserve that cache/update boundary
    // while streaming the prefix so this diagnostic does not require a
    // 16K x HC x hidden activation for the entire prefix at once.
    let mut tower = load_prefix_tower(dir, &config, target_layer)?;
    let mut caches = tower.new_caches();
    let mut history_input = Vec::with_capacity(HISTORY * 4096);
    let mut history_indexer_key = Vec::with_capacity(HISTORY * 128);
    let mut history_indexer_gate = Vec::with_capacity(HISTORY * 128);
    let mut history_kv_raw = Vec::with_capacity(HISTORY * 512);
    let mut history_kv_norm = Vec::with_capacity(HISTORY * 512);
    for start in (0..HISTORY).step_by(DS4_GLM53_PREFILL_CHUNK_TOKENS) {
        let end = (start + DS4_GLM53_PREFILL_CHUNK_TOKENS).min(HISTORY);
        let chunk_ids: Vec<i32> = ids[start..end].iter().map(|&id| id as i32).collect();
        let chunk_ids = Array::from_slice(&chunk_ids, &[1, (end - start) as i32]);
        let (_, multi) =
            tower.forward_inner_until(&chunk_ids, &mut caches, Some(target_layer - 1))?;
        let state = multi.reshape(&[
            1,
            (end - start) as i32,
            config.hc_mult as i32,
            config.hidden_size as i32,
        ])?;
        let x = {
            let layer = &tower.layers[target_layer];
            let (_, x, _, _) = layer
                .hc_attn
                .split_normed(&state, &layer.in_norm, config.rms_norm_eps)?
                .ok_or_else(|| {
                    anyhow::anyhow!("GLM layer-{target_layer} mHC split/norm declined")
                })?;
            x
        };
        let (kv_raw, indexer_key, indexer_gate, kv_norm) = {
            let dsa = match &tower.layers[target_layer].attn {
                Attn::Dsa(dsa) => dsa,
                Attn::Kda(_) => anyhow::bail!("GLM layer {target_layer} is not DSA"),
            };
            let cache = match &mut caches[target_layer] {
                LayerCache::Full(cache) => cache,
                LayerCache::Linear(_) | LayerCache::Native(_) => {
                    anyhow::bail!("GLM layer-{target_layer} cache is not DSA")
                }
            };
            let (kv_raw, indexer_key, indexer_gate) =
                dsa.project_prefill_cache_for_capture(&x, cache, target_layer)?;
            let kv_norm = dsa.kv_a_norm_for_capture(&kv_raw)?.to_vec1::<f32>()?;
            (kv_raw, indexer_key, indexer_gate, kv_norm)
        };
        history_input.extend(x.to_vec1::<f32>()?);
        history_indexer_key.extend(indexer_key.to_vec1::<f32>()?);
        history_indexer_gate.extend(indexer_gate.to_vec1::<f32>()?);
        history_kv_raw.extend(kv_raw.to_vec1::<f32>()?);
        history_kv_norm.extend(kv_norm);
    }
    let target_cache_len = match &caches[target_layer] {
        LayerCache::Full(cache) => cache.indexer_tape.total,
        LayerCache::Linear(_) | LayerCache::Native(_) => {
            anyhow::bail!("GLM layer-{target_layer} cache is not DSA")
        }
    };
    anyhow::ensure!(
        target_cache_len == HISTORY,
        "chunked layer-{target_layer} DSA cache did not reach the 16K prefix"
    );
    let mut kda_initials = Vec::with_capacity(3);
    for layer_id in 0..3 {
        let (conv, state) = match &caches[layer_id] {
            LayerCache::Linear(cache) => (
                cache
                    .conv
                    .as_ref()
                    .ok_or_else(|| anyhow::anyhow!("KDA layer {layer_id} has no conv prefix"))?
                    .to_vec1::<f32>()?,
                cache
                    .ssm
                    .as_ref()
                    .ok_or_else(|| anyhow::anyhow!("KDA layer {layer_id} has no SSM prefix"))?
                    .to_vec1::<f32>()?,
            ),
            LayerCache::Full(_) | LayerCache::Native(_) => {
                anyhow::bail!("KDA layer {layer_id} has an incompatible cache")
            }
        };
        let weights = match &tower.layers[layer_id].attn {
            Attn::Kda(kda) => kda.ds4_decode_weights_for_capture()?,
            Attn::Dsa(_) => anyhow::bail!("layer {layer_id} is not a KDA layer"),
        };
        kda_initials.push((conv, state, weights));
    }
    tower.capture = Some(Capture {
        layers: vec![0, 1, 2],
        state_only: false,
        kda_prefill_only: false,
        entries: std::sync::Mutex::new(HashMap::new()),
        shapes: std::sync::Mutex::new(HashMap::new()),
        flushed: std::sync::atomic::AtomicBool::new(false),
        ..Default::default()
    });
    let query_ids = Array::from_slice(&[query_id as i32], &[1, 1]);
    let (_, query_multi) =
        tower.forward_inner_until(&query_ids, &mut caches, Some(target_layer - 1))?;
    let kda_capture = tower
        .capture
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("KDA decode capture was not installed"))?;
    let entries = kda_capture.entries.lock().expect("KDA decode capture lock");
    for layer_id in 0..3 {
        let get = |stage: &str| -> anyhow::Result<&Vec<f32>> {
            entries
                .get(&format!("L{layer_id}.{stage}"))
                .ok_or_else(|| anyhow::anyhow!("missing KDA decode capture L{layer_id}.{stage}"))
        };
        let q = get("kda_q_proj")?;
        let k = get("kda_k_proj")?;
        let v = get("kda_v_proj")?;
        let raw_gate = get("kda_gate_raw")?;
        let raw_beta = get("kda_beta_logits")?;
        let output_gate = get("kda_gb_proj")?;
        let output = get("kda_decode_out")?;
        let next_conv = get("kda_decode_conv_state")?;
        let next_state = get("kda_decode_state")?;
        let (initial_conv, initial_state, weights) = &kda_initials[layer_id];
        let (q_conv, k_conv, v_conv, a_log, dt_bias, output_norm) = weights;
        let projection = config.linear_num_heads * config.linear_head_dim;
        let state_count = projection * config.linear_head_dim;
        let conv_history = 3 * projection * 3;
        let conv_weights = config.linear_conv_kernel_dim * projection;
        anyhow::ensure!(
            [
                q.len(),
                k.len(),
                v.len(),
                raw_gate.len(),
                output_gate.len(),
                output.len()
            ]
            .iter()
            .all(|&length| length == projection),
            "KDA layer {layer_id} decode projection width differs from model geometry"
        );
        anyhow::ensure!(raw_beta.len() == config.linear_num_heads);
        anyhow::ensure!(
            initial_conv.len() == conv_history && next_conv.len() == conv_history,
            "KDA layer {layer_id} convolution cache width differs from DS4 geometry"
        );
        anyhow::ensure!(
            initial_state.len() == state_count && next_state.len() == state_count,
            "KDA layer {layer_id} recurrent cache width differs from model geometry"
        );
        anyhow::ensure!(
            [q_conv.len(), k_conv.len(), v_conv.len()]
                .iter()
                .all(|&length| length == conv_weights)
                && a_log.len() == config.linear_num_heads
                && dt_bias.len() == projection
                && output_norm.len() == config.linear_head_dim,
            "KDA layer {layer_id} decode weight width differs from model geometry"
        );
        let mut kda_bytes = Vec::new();
        for value in [
            layer_id as u32,
            config.linear_num_heads as u32,
            config.linear_head_dim as u32,
            (config.linear_num_heads * config.linear_head_dim) as u32,
            config.linear_conv_kernel_dim as u32,
            config.linear_lower_bound.to_bits(),
            config.rms_norm_eps.to_bits(),
        ] {
            kda_bytes.extend_from_slice(&value.to_le_bytes());
        }
        for values in [
            q.as_slice(),
            k.as_slice(),
            v.as_slice(),
            raw_gate.as_slice(),
            raw_beta.as_slice(),
            output_gate.as_slice(),
            initial_conv.as_slice(),
            initial_state.as_slice(),
            q_conv.as_slice(),
            k_conv.as_slice(),
            v_conv.as_slice(),
            a_log.as_slice(),
            dt_bias.as_slice(),
            output_norm.as_slice(),
            output.as_slice(),
            next_conv.as_slice(),
            next_state.as_slice(),
        ] {
            for value in values {
                kda_bytes.extend_from_slice(&value.to_le_bytes());
            }
        }
        std::fs::write(
            output_dir.join(format!("kda_layer{layer_id}_decode_capture.bin")),
            &kda_bytes,
        )?;
        let fixture: Option<Vec<u8>> = match layer_id {
            0 => Some(crate::models::glm53::test_fixtures::read(
                "kda_decode_layer0_model_long_ds4.bin",
            )),
            1 => Some(crate::models::glm53::test_fixtures::read(
                "kda_decode_layer1_model_long_ds4.bin",
            )),
            2 => Some(crate::models::glm53::test_fixtures::read(
                "kda_decode_layer2_model_long_ds4.bin",
            )),
            _ => None,
        };
        if let Some(fixture) = fixture {
            anyhow::ensure!(
                fixture.len() >= 28,
                "KDA layer {layer_id} fixture is truncated"
            );
            anyhow::ensure!(
                fixture[..28] == kda_bytes[..28],
                "KDA layer {layer_id} fixture header differs from loaded model geometry"
            );
            anyhow::ensure!(
                fixture.len() == 28 + (output.len() + next_conv.len() + next_state.len()) * 4,
                "KDA layer {layer_id} DS4 fixture has an unexpected output size"
            );
            let mut expected_offset = 28;
            for (label, values) in [
                ("output", output.as_slice()),
                ("convolution state", next_conv.as_slice()),
                ("recurrent state", next_state.as_slice()),
            ] {
                anyhow::ensure!(
                    expected_offset + values.len() * 4 <= fixture.len(),
                    "KDA layer {layer_id} fixture is truncated at {label}"
                );
                for (index, (actual, bytes)) in values
                    .iter()
                    .zip(
                        fixture[expected_offset..expected_offset + values.len() * 4]
                            .chunks_exact(4),
                    )
                    .enumerate()
                {
                    let expected = f32::from_le_bytes(bytes.try_into().expect("F32 fixture word"));
                    anyhow::ensure!(
                        actual.to_bits() == expected.to_bits(),
                        "KDA layer {layer_id} {label} differs from DS4 at {index}: {actual} vs {expected}"
                    );
                }
                expected_offset += values.len() * 4;
            }
            anyhow::ensure!(expected_offset == fixture.len());
        }
    }
    drop(entries);
    let query_state =
        query_multi.reshape(&[1, 1, config.hc_mult as i32, config.hidden_size as i32])?;
    let (query_x, dsa) = {
        let layer = &tower.layers[target_layer];
        let (_, x, _, _) = layer
            .hc_attn
            .split_normed(&query_state, &layer.in_norm, config.rms_norm_eps)?
            .ok_or_else(|| {
                anyhow::anyhow!("GLM layer-{target_layer} decode mHC split/norm declined")
            })?;
        let dsa = match &layer.attn {
            Attn::Dsa(dsa) => dsa,
            Attn::Kda(_) => anyhow::bail!("GLM layer {target_layer} is not DSA"),
        };
        (x, dsa)
    };
    let cache = match &mut caches[target_layer] {
        LayerCache::Full(cache) => cache,
        LayerCache::Linear(_) | LayerCache::Native(_) => {
            anyhow::bail!("GLM layer-{target_layer} cache is not DSA")
        }
    };
    let loaded_prefill_fixture: &[u8] =
        &crate::models::glm53::test_fixtures::read("indexer_prefill_projection_model_long_ds4.bin");
    const PREFILL_HEADER_BYTES: usize = 4 * std::mem::size_of::<u32>();
    const PREFILL_PROJECTION_BYTES: usize = HISTORY * 128 * std::mem::size_of::<f32>();
    let fixture_f32 = |bytes: &[u8]| {
        bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("F32 fixture word")))
            .collect::<Vec<_>>()
    };
    let assert_fixture_f32 = |label: &str, actual: &[f32], expected: &[u8]| {
        anyhow::ensure!(expected.len() == actual.len() * 4);
        for (index, (value, bytes)) in actual.iter().zip(expected.chunks_exact(4)).enumerate() {
            let expected = f32::from_le_bytes(bytes.try_into().expect("F32 fixture word"));
            anyhow::ensure!(
                value.to_bits() == expected.to_bits(),
                "layer-{target_layer} {label} differs from DS4 at {index}: {value} vs {expected}"
            );
        }
        Ok::<(), anyhow::Error>(())
    };
    let (wk_weight, gate_weight) = dsa.indexer_projection_weights_for_capture()?;
    anyhow::ensure!(history_input.len() == HISTORY * 4096);
    anyhow::ensure!(wk_weight.len() == 128 * 4096 && gate_weight.len() == 128 * 4096);
    anyhow::ensure!(
        history_indexer_key.len() == HISTORY * 128 && history_indexer_gate.len() == HISTORY * 128
    );
    let mut prefill_projection_bytes =
        Vec::with_capacity(16 + HISTORY * 4096 * 4 + 2 * 128 * 4096 * 2 + 2 * HISTORY * 128 * 4);
    for value in [HISTORY as u32, 4096u32, 128u32, 2u32] {
        prefill_projection_bytes.extend_from_slice(&value.to_le_bytes());
    }
    for value in &history_input {
        prefill_projection_bytes.extend_from_slice(&value.to_le_bytes());
    }
    for value in wk_weight.iter().chain(&gate_weight) {
        prefill_projection_bytes.extend_from_slice(&value.to_le_bytes());
    }
    for value in history_indexer_key.iter().chain(&history_indexer_gate) {
        prefill_projection_bytes.extend_from_slice(&value.to_le_bytes());
    }
    std::fs::write(
        output_dir.join("lisa_long_prefill_projection_capture.bin"),
        prefill_projection_bytes,
    )?;
    if target_layer == 3 {
        anyhow::ensure!(
            loaded_prefill_fixture.len() == PREFILL_HEADER_BYTES + 2 * PREFILL_PROJECTION_BYTES,
            "unexpected long-prefill projection fixture length"
        );
        anyhow::ensure!(
            loaded_prefill_fixture[..PREFILL_HEADER_BYTES]
                == [
                    (HISTORY as u32).to_le_bytes(),
                    4096u32.to_le_bytes(),
                    128u32.to_le_bytes(),
                    2u32.to_le_bytes(),
                ]
                .concat(),
            "unexpected long-prefill projection fixture header"
        );
        let expected_key = fixture_f32(
            &loaded_prefill_fixture
                [PREFILL_HEADER_BYTES..PREFILL_HEADER_BYTES + PREFILL_PROJECTION_BYTES],
        );
        let expected_gate =
            fixture_f32(&loaded_prefill_fixture[PREFILL_HEADER_BYTES + PREFILL_PROJECTION_BYTES..]);
        anyhow::ensure!(history_indexer_key.len() == expected_key.len());
        anyhow::ensure!(history_indexer_gate.len() == expected_gate.len());
        for (index, (&actual, &expected)) in
            history_indexer_key.iter().zip(&expected_key).enumerate()
        {
            anyhow::ensure!(
                actual.to_bits() == expected.to_bits(),
                "16K prefill key projection differs from DS4 at {index}: {actual} vs {expected}"
            );
        }
        for (index, (&actual, &expected)) in
            history_indexer_gate.iter().zip(&expected_gate).enumerate()
        {
            anyhow::ensure!(
                actual.to_bits() == expected.to_bits(),
                "16K prefill gate projection differs from DS4 at {index}: {actual} vs {expected}"
            );
        }
    }
    let prefill_fixture = match target_layer {
        43 => Some(crate::models::glm53::test_fixtures::read(
            "indexer_prefill_projection_layer43_model_long_ds4.bin",
        )),
        39 => Some(crate::models::glm53::test_fixtures::read(
            "indexer_prefill_projection_layer39_model_long_ds4.bin",
        )),
        35 => Some(crate::models::glm53::test_fixtures::read(
            "indexer_prefill_projection_layer35_model_long_ds4.bin",
        )),
        31 => Some(crate::models::glm53::test_fixtures::read(
            "indexer_prefill_projection_layer31_model_long_ds4.bin",
        )),
        27 => Some(crate::models::glm53::test_fixtures::read(
            "indexer_prefill_projection_layer27_model_long_ds4.bin",
        )),
        15 => Some(crate::models::glm53::test_fixtures::read(
            "indexer_prefill_projection_layer15_model_long_ds4.bin",
        )),
        11 => Some(crate::models::glm53::test_fixtures::read(
            "indexer_prefill_projection_layer11_model_long_ds4.bin",
        )),
        7 => Some(crate::models::glm53::test_fixtures::read(
            "indexer_prefill_projection_layer7_model_long_ds4.bin",
        )),
        19 => Some(crate::models::glm53::test_fixtures::read(
            "indexer_prefill_projection_layer19_model_long_ds4.bin",
        )),
        23 => Some(crate::models::glm53::test_fixtures::read(
            "indexer_prefill_projection_layer23_model_long_ds4.bin",
        )),
        _ => None,
    };
    if let Some(fixture) = prefill_fixture {
        anyhow::ensure!(
            fixture.len() == PREFILL_HEADER_BYTES + 2 * PREFILL_PROJECTION_BYTES
                && fixture[..PREFILL_HEADER_BYTES]
                    == [
                        (HISTORY as u32).to_le_bytes(),
                        4096u32.to_le_bytes(),
                        128u32.to_le_bytes(),
                        2u32.to_le_bytes(),
                    ]
                    .concat(),
            "layer-{target_layer} prefill projection fixture has an unexpected header"
        );
        assert_fixture_f32(
            "16K indexer key projection",
            &history_indexer_key,
            &fixture[PREFILL_HEADER_BYTES..PREFILL_HEADER_BYTES + PREFILL_PROJECTION_BYTES],
        )?;
        assert_fixture_f32(
            "16K indexer gate projection",
            &history_indexer_gate,
            &fixture[PREFILL_HEADER_BYTES + PREFILL_PROJECTION_BYTES..],
        )?;
    }
    let (kv_a_weight, kv_a_scales, kv_a_biases, kv_a_norm_weight, kv_a_group, kv_a_bits, kv_a_eps) =
        dsa.kv_a_quant_params_for_capture()?;
    let kv_a_raw = history_kv_raw;
    let kv_a_norm = history_kv_norm;
    let kv_a_compact = cache
        .indexer_tape
        .compact_slice(0, HISTORY)
        .ok_or_else(|| anyhow::anyhow!("missing model-produced 16K compact KV"))?
        .to_vec1::<u16>()?;
    anyhow::ensure!(kv_a_raw.len() == HISTORY * 512 && kv_a_norm.len() == HISTORY * 512);
    anyhow::ensure!(kv_a_compact.len() == HISTORY * 512);
    let kv_a_compact_fixture = match target_layer {
        43 => Some(crate::models::glm53::test_fixtures::read(
            "dsa_compact_cache_layer43_model_long_ds4.f16",
        )),
        39 => Some(crate::models::glm53::test_fixtures::read(
            "dsa_compact_cache_layer39_model_long_ds4.f16",
        )),
        35 => Some(crate::models::glm53::test_fixtures::read(
            "dsa_compact_cache_layer35_model_long_ds4.f16",
        )),
        31 => Some(crate::models::glm53::test_fixtures::read(
            "dsa_compact_cache_layer31_model_long_ds4.f16",
        )),
        11 => Some(crate::models::glm53::test_fixtures::read(
            "dsa_compact_cache_layer11_model_long_ds4.f16",
        )),
        15 => Some(crate::models::glm53::test_fixtures::read(
            "dsa_compact_cache_layer15_model_long_ds4.f16",
        )),
        _ => None,
    };
    if let Some(fixture) = kv_a_compact_fixture {
        anyhow::ensure!(fixture.len() == kv_a_compact.len() * 2);
        for (index, (actual, bytes)) in kv_a_compact.iter().zip(fixture.chunks_exact(2)).enumerate()
        {
            let expected = u16::from_le_bytes(bytes.try_into().expect("F16 compact word"));
            anyhow::ensure!(
                *actual == expected,
                "layer-{target_layer} 16K compact KV differs from DS4 at {index}: {actual:#06x} vs {expected:#06x}"
            );
        }
    }
    let write_u32 = |name: &str, values: &[u32]| -> anyhow::Result<()> {
        let bytes: Vec<u8> = values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        std::fs::write(output_dir.join(name), bytes)?;
        Ok(())
    };
    let write_f32 = |name: &str, values: &[f32]| -> anyhow::Result<()> {
        let bytes: Vec<u8> = values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        std::fs::write(output_dir.join(name), bytes)?;
        Ok(())
    };
    let write_f16 = |name: &str, values: &[u16]| -> anyhow::Result<()> {
        let bytes: Vec<u8> = values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        std::fs::write(output_dir.join(name), bytes)?;
        Ok(())
    };
    write_u32(
        "lisa_long_kv_a_meta.u32",
        &[
            HISTORY as u32,
            4096,
            512,
            kv_a_group as u32,
            kv_a_bits as u32,
            kv_a_eps.to_bits(),
        ],
    )?;
    write_u32("lisa_long_kv_a_weight.u32", &kv_a_weight)?;
    write_f32("lisa_long_kv_a_scales.f32", &kv_a_scales)?;
    write_f32("lisa_long_kv_a_biases.f32", &kv_a_biases)?;
    write_f32("lisa_long_kv_a_norm_weight.f32", &kv_a_norm_weight)?;
    write_f32("lisa_long_kv_a_raw.f32", &kv_a_raw)?;
    write_f32("lisa_long_kv_a_norm.f32", &kv_a_norm)?;
    write_f16("lisa_long_kv_a_compact.f16", &kv_a_compact)?;

    let capture = Capture {
        layers: vec![target_layer],
        state_only: false,
        kda_prefill_only: false,
        entries: std::sync::Mutex::new(HashMap::new()),
        shapes: std::sync::Mutex::new(HashMap::new()),
        flushed: std::sync::atomic::AtomicBool::new(false),
        ..Default::default()
    };
    let dsa_output = dsa.forward(&query_x, cache, Some(&capture), target_layer, true)?;
    anyhow::ensure!(dsa_output.shape() == [1, 1, config.hidden_size as i32]);
    anyhow::ensure!(cache.indexer_tape.total == HISTORY + 1);

    let entries = capture.entries.lock().expect("capture entries lock");
    let get = |stage: &str| -> anyhow::Result<&Vec<f32>> {
        entries
            .get(&format!("L{target_layer}.{stage}"))
            .ok_or_else(|| {
                anyhow::anyhow!("missing model-produced layer-{target_layer} stage {stage}")
            })
    };
    let query = get("dsa_indexer_q_p16384_t1")?;
    let projection_input = get("dsa_indexer_input_x")?;
    let projected_key = get("dsa_indexer_raw_k")?;
    let projected_gate = get("dsa_indexer_pool_gate")?;
    let weights = get("dsa_indexer_weights_p16384_t1")?;
    let scores = get("dsa_index_scores_p16384_t1")?;
    let query_full = get("dsa_q")?;
    let q_low = get("dsa_q_low")?;
    let lora = get("dsa_attn_lora")?;
    let selected = get("dsa_selected_p16384_t1")?;
    anyhow::ensure!(query.len() == 32 * 128 && weights.len() == 32 && scores.len() == 4096);
    anyhow::ensure!(projection_input.len() == 4096);
    anyhow::ensure!(projected_key.len() == 128 && projected_gate.len() == 128);
    anyhow::ensure!(query_full.len() == 64 * 256 && q_low.len() == 64 * 512);
    anyhow::ensure!(lora.len() == 64 * 512 && selected.len() == 2051);
    let external_projection_fixture: &[u8] =
        &crate::models::glm53::test_fixtures::read("indexer_projection_model_long_ds4.bin");
    const PROJECTION_HEADER_BYTES: usize = 4 * std::mem::size_of::<u32>();
    const PROJECTION_OUTPUT_START: usize = PROJECTION_HEADER_BYTES
        + 4096 * std::mem::size_of::<f32>()
        + 2 * 128 * 4096 * std::mem::size_of::<u16>();
    let projection_fixture = match target_layer {
        3 => Some(external_projection_fixture.to_vec()),
        43 => Some(crate::models::glm53::test_fixtures::read(
            "indexer_projection_layer43_model_long_ds4.bin",
        )),
        39 => Some(crate::models::glm53::test_fixtures::read(
            "indexer_projection_layer39_model_long_ds4.bin",
        )),
        35 => Some(crate::models::glm53::test_fixtures::read(
            "indexer_projection_layer35_model_long_ds4.bin",
        )),
        31 => Some(crate::models::glm53::test_fixtures::read(
            "indexer_projection_layer31_model_long_ds4.bin",
        )),
        15 => Some(crate::models::glm53::test_fixtures::read(
            "indexer_projection_layer15_model_long_ds4.bin",
        )),
        11 => Some(crate::models::glm53::test_fixtures::read(
            "indexer_projection_layer11_model_long_ds4.bin",
        )),
        7 => Some(crate::models::glm53::test_fixtures::read(
            "indexer_projection_layer7_model_long_ds4.bin",
        )),
        19 => Some(crate::models::glm53::test_fixtures::read(
            "indexer_projection_layer19_model_long_ds4.bin",
        )),
        27 => Some(crate::models::glm53::test_fixtures::read(
            "indexer_projection_layer27_model_long_ds4.bin",
        )),
        23 => Some(crate::models::glm53::test_fixtures::read(
            "indexer_projection_layer23_model_long_ds4.bin",
        )),
        _ => None,
    };
    if let Some(fixture) = projection_fixture {
        anyhow::ensure!(
            fixture.len() == PROJECTION_OUTPUT_START + 2 * 128 * 4,
            "layer-{target_layer} projection fixture has an unexpected size"
        );
        anyhow::ensure!(
            fixture[..PROJECTION_HEADER_BYTES]
                == [
                    1u32.to_le_bytes(),
                    4096u32.to_le_bytes(),
                    128u32.to_le_bytes(),
                    2u32.to_le_bytes()
                ]
                .concat(),
            "layer-{target_layer} projection fixture has an unexpected header"
        );
        assert_fixture_f32(
            "indexer key projection",
            projected_key,
            &fixture[PROJECTION_OUTPUT_START..PROJECTION_OUTPUT_START + 128 * 4],
        )?;
        assert_fixture_f32(
            "indexer gate projection",
            projected_gate,
            &fixture[PROJECTION_OUTPUT_START + 128 * 4..],
        )?;
    }
    let (kv_b_weight, kv_b_scales, kv_b_biases) = dsa.kv_b_quant_params_for_capture()?;
    anyhow::ensure!(kv_b_weight.len() == 64 * 512 * 128);
    anyhow::ensure!(kv_b_scales.len() == 64 * 512 * 8);
    anyhow::ensure!(kv_b_biases.len() == 64 * 512 * 8);
    let mut q_low_bytes = Vec::with_capacity(
        16 + query_full.len() * 4 + kv_b_weight.len() * 4 + kv_b_scales.len() * 8 + q_low.len() * 4,
    );
    for value in [64u32, 256u32, 512u32, 64u32] {
        q_low_bytes.extend_from_slice(&value.to_le_bytes());
    }
    for value in query_full {
        q_low_bytes.extend_from_slice(&value.to_le_bytes());
    }
    for value in &kv_b_weight {
        q_low_bytes.extend_from_slice(&value.to_le_bytes());
    }
    for value in kv_b_scales.iter().chain(&kv_b_biases) {
        q_low_bytes.extend_from_slice(&value.to_le_bytes());
    }
    for value in q_low {
        q_low_bytes.extend_from_slice(&value.to_le_bytes());
    }
    let loaded_q_low_fixture: &[u8] =
        &crate::models::glm53::test_fixtures::read("q_low_model_long_ds4.bin");
    let q_low_fixture = match target_layer {
        3 => Some(loaded_q_low_fixture.to_vec()),
        43 => Some(crate::models::glm53::test_fixtures::read(
            "q_low_layer43_model_long_ds4.bin",
        )),
        39 => Some(crate::models::glm53::test_fixtures::read(
            "q_low_layer39_model_long_ds4.bin",
        )),
        35 => Some(crate::models::glm53::test_fixtures::read(
            "q_low_layer35_model_long_ds4.bin",
        )),
        31 => Some(crate::models::glm53::test_fixtures::read(
            "q_low_layer31_model_long_ds4.bin",
        )),
        15 => Some(crate::models::glm53::test_fixtures::read(
            "q_low_layer15_model_long_ds4.bin",
        )),
        11 => Some(crate::models::glm53::test_fixtures::read(
            "q_low_layer11_model_long_ds4.bin",
        )),
        7 => Some(crate::models::glm53::test_fixtures::read(
            "q_low_layer7_model_long_ds4.bin",
        )),
        19 => Some(crate::models::glm53::test_fixtures::read(
            "q_low_layer19_model_long_ds4.bin",
        )),
        27 => Some(crate::models::glm53::test_fixtures::read(
            "q_low_layer27_model_long_ds4.bin",
        )),
        23 => Some(crate::models::glm53::test_fixtures::read(
            "q_low_layer23_model_long_ds4.bin",
        )),
        _ => None,
    };
    if let Some(fixture) = q_low_fixture {
        anyhow::ensure!(
            fixture.len() == 16 + q_low.len() * 4,
            "layer-{target_layer} q-low fixture has an unexpected length"
        );
        anyhow::ensure!(
            fixture[..16]
                == [
                    64u32.to_le_bytes(),
                    256u32.to_le_bytes(),
                    512u32.to_le_bytes(),
                    64u32.to_le_bytes()
                ]
                .concat(),
            "layer-{target_layer} q-low fixture has an unexpected header"
        );
        assert_fixture_f32("q-low projection", q_low, &fixture[16..])?;
    }
    std::fs::write(output_dir.join("lisa_long_q_low_capture.bin"), q_low_bytes)?;
    let pooled = cache
        .indexer_tape
        .pooled_view()
        .ok_or_else(|| anyhow::anyhow!("missing model-produced long-window pooled keys"))?
        .to_vec1::<u16>()?;
    anyhow::ensure!(pooled.len() == 4096 * 128);
    let pool_fixture = match target_layer {
        3 => Some(crate::models::glm53::test_fixtures::read(
            "indexer_pool_layer3_model_long_ds4.bin",
        )),
        43 => Some(crate::models::glm53::test_fixtures::read(
            "indexer_pool_layer43_model_long_ds4.bin",
        )),
        39 => Some(crate::models::glm53::test_fixtures::read(
            "indexer_pool_layer39_model_long_ds4.bin",
        )),
        35 => Some(crate::models::glm53::test_fixtures::read(
            "indexer_pool_layer35_model_long_ds4.bin",
        )),
        31 => Some(crate::models::glm53::test_fixtures::read(
            "indexer_pool_layer31_model_long_ds4.bin",
        )),
        15 => Some(crate::models::glm53::test_fixtures::read(
            "indexer_pool_layer15_model_long_ds4.bin",
        )),
        11 => Some(crate::models::glm53::test_fixtures::read(
            "indexer_pool_layer11_model_long_ds4.bin",
        )),
        7 => Some(crate::models::glm53::test_fixtures::read(
            "indexer_pool_layer7_model_long_ds4.bin",
        )),
        19 => Some(crate::models::glm53::test_fixtures::read(
            "indexer_pool_layer19_model_long_ds4.bin",
        )),
        27 => Some(crate::models::glm53::test_fixtures::read(
            "indexer_pool_layer27_model_long_ds4.bin",
        )),
        23 => Some(crate::models::glm53::test_fixtures::read(
            "indexer_pool_layer23_model_long_ds4.bin",
        )),
        _ => None,
    };
    if let Some(fixture) = pool_fixture {
        let output_bytes = pooled.len() * 2;
        anyhow::ensure!(fixture.len() >= output_bytes);
        for (index, (actual, bytes)) in pooled
            .iter()
            .zip(fixture[fixture.len() - output_bytes..].chunks_exact(2))
            .enumerate()
        {
            let expected = u16::from_le_bytes(bytes.try_into().expect("F16 pool word"));
            anyhow::ensure!(
                *actual == expected,
                "layer-{target_layer} pooled F16 cache differs from DS4 at {index}: {actual:#06x} vs {expected:#06x}"
            );
        }
    }
    let selected_values: Vec<i32> = selected.iter().map(|&value| value as i32).collect();
    anyhow::ensure!(selected_values.len() == 2051);

    let mut bytes = Vec::with_capacity(1_100_000);
    for value in [HISTORY as u32, 4096u32, selected_values.len() as u32] {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    for value in query.iter().chain(weights).chain(scores) {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    for value in &pooled {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    for value in &selected_values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    let selection_fixture = match target_layer {
        3 => Some(crate::models::glm53::test_fixtures::read(
            "indexer_selection_layer3_model_long_ds4.bin",
        )),
        43 => Some(crate::models::glm53::test_fixtures::read(
            "indexer_selection_layer43_model_long_ds4.bin",
        )),
        39 => Some(crate::models::glm53::test_fixtures::read(
            "indexer_selection_layer39_model_long_ds4.bin",
        )),
        35 => Some(crate::models::glm53::test_fixtures::read(
            "indexer_selection_layer35_model_long_ds4.bin",
        )),
        31 => Some(crate::models::glm53::test_fixtures::read(
            "indexer_selection_layer31_model_long_ds4.bin",
        )),
        15 => Some(crate::models::glm53::test_fixtures::read(
            "indexer_selection_layer15_model_long_ds4.bin",
        )),
        11 => Some(crate::models::glm53::test_fixtures::read(
            "indexer_selection_layer11_model_long_ds4.bin",
        )),
        7 => Some(crate::models::glm53::test_fixtures::read(
            "indexer_selection_layer7_model_long_ds4.bin",
        )),
        19 => Some(crate::models::glm53::test_fixtures::read(
            "indexer_selection_layer19_model_long_ds4.bin",
        )),
        27 => Some(crate::models::glm53::test_fixtures::read(
            "indexer_selection_layer27_model_long_ds4.bin",
        )),
        23 => Some(crate::models::glm53::test_fixtures::read(
            "indexer_selection_layer23_model_long_ds4.bin",
        )),
        _ => None,
    };
    if let Some(fixture) = selection_fixture {
        let selected_bytes = selected_values.len() * 4;
        let top_ids_bytes = 512 * 4;
        let score_bytes = scores.len() * 4;
        anyhow::ensure!(fixture.len() >= selected_bytes + top_ids_bytes + score_bytes);
        let score_start = fixture.len() - selected_bytes - top_ids_bytes - score_bytes;
        assert_fixture_f32(
            "indexer scores",
            scores,
            &fixture[score_start..score_start + score_bytes],
        )?;
        let selected_start = fixture.len() - selected_bytes;
        for (index, (actual, encoded)) in selected_values
            .iter()
            .zip(fixture[selected_start..].chunks_exact(4))
            .enumerate()
        {
            let expected = i32::from_le_bytes(encoded.try_into().expect("selected ID word"));
            anyhow::ensure!(
                *actual == expected,
                "layer-{target_layer} selected token ID differs from DS4 at {index}: {actual} vs {expected}"
            );
        }
    }
    std::fs::write(output_dir.join("lisa_long_decode_capture.bin"), bytes)?;

    anyhow::ensure!(wk_weight.len() == 128 * 4096 && gate_weight.len() == 128 * 4096);
    let mut projection_bytes = Vec::with_capacity(16 + 4096 * 4 + 2 * 128 * 4096 * 2 + 2 * 128 * 4);
    for value in [1u32, 4096u32, 128u32, 2u32] {
        projection_bytes.extend_from_slice(&value.to_le_bytes());
    }
    for value in projection_input {
        projection_bytes.extend_from_slice(&value.to_le_bytes());
    }
    for value in wk_weight.iter().chain(&gate_weight) {
        projection_bytes.extend_from_slice(&value.to_le_bytes());
    }
    for value in projected_key.iter().chain(projected_gate) {
        projection_bytes.extend_from_slice(&value.to_le_bytes());
    }
    std::fs::write(
        output_dir.join("lisa_long_projection_capture.bin"),
        projection_bytes,
    )?;
    let token_bytes: Vec<u8> = ids
        .iter()
        .copied()
        .chain(std::iter::once(query_id))
        .flat_map(|value| value.to_le_bytes())
        .collect();
    std::fs::write(output_dir.join("prompt_ids.u32"), token_bytes)?;
    let compact_latents = cache
        .indexer_tape
        .compact_slice(0, HISTORY + 1)
        .ok_or_else(|| anyhow::anyhow!("missing model-produced long-window compact KV"))?
        .to_vec1::<u16>()?;
    anyhow::ensure!(compact_latents.len() == (HISTORY + 1) * 512);
    let mut attention_bytes = Vec::with_capacity(17_000_000);
    for value in [HISTORY as u32 + 1, selected_values.len() as u32] {
        attention_bytes.extend_from_slice(&value.to_le_bytes());
    }
    for value in query_full.iter().chain(q_low).chain(lora) {
        attention_bytes.extend_from_slice(&value.to_le_bytes());
    }
    for value in &compact_latents {
        attention_bytes.extend_from_slice(&value.to_le_bytes());
    }
    for value in &selected_values {
        attention_bytes.extend_from_slice(&value.to_le_bytes());
    }
    std::fs::write(
        output_dir.join("lisa_long_attention_capture.bin"),
        attention_bytes,
    )?;
    let attention_fixture = match target_layer {
        3 => Some(crate::models::glm53::test_fixtures::read(
            "dsa_attention_layer3_model_long_ds4.bin",
        )),
        43 => Some(crate::models::glm53::test_fixtures::read(
            "dsa_attention_layer43_model_long_ds4.bin",
        )),
        39 => Some(crate::models::glm53::test_fixtures::read(
            "dsa_attention_layer39_model_long_ds4.bin",
        )),
        35 => Some(crate::models::glm53::test_fixtures::read(
            "dsa_attention_layer35_model_long_ds4.bin",
        )),
        31 => Some(crate::models::glm53::test_fixtures::read(
            "dsa_attention_layer31_model_long_ds4.bin",
        )),
        15 => Some(crate::models::glm53::test_fixtures::read(
            "dsa_attention_layer15_model_long_ds4.bin",
        )),
        11 => Some(crate::models::glm53::test_fixtures::read(
            "dsa_attention_layer11_model_long_ds4.bin",
        )),
        7 => Some(crate::models::glm53::test_fixtures::read(
            "dsa_attention_layer7_model_long_ds4.bin",
        )),
        19 => Some(crate::models::glm53::test_fixtures::read(
            "dsa_attention_layer19_model_long_ds4.bin",
        )),
        27 => Some(crate::models::glm53::test_fixtures::read(
            "dsa_attention_layer27_model_long_ds4.bin",
        )),
        23 => Some(crate::models::glm53::test_fixtures::read(
            "dsa_attention_layer23_model_long_ds4.bin",
        )),
        _ => None,
    };
    if let Some(fixture) = attention_fixture {
        assert_fixture_f32(
            "indexed-lora output",
            lora,
            &fixture[fixture.len() - lora.len() * 4..],
        )?;
    }
    let (norm_weight, norm_bias, ape) = dsa.indexer_pool_params_for_capture()?;
    let pooled_expected = cache
        .indexer_tape
        .pooled_view()
        .ok_or_else(|| anyhow::anyhow!("missing model-produced long-window pooled keys"))?
        .to_vec1::<u16>()?;
    anyhow::ensure!(history_indexer_key.len() == HISTORY * 128);
    anyhow::ensure!(history_indexer_gate.len() == HISTORY * 128);
    anyhow::ensure!(norm_weight.len() == 128 && norm_bias.len() == 128 && ape.len() == 4 * 128);
    anyhow::ensure!(pooled_expected.len() == 4096 * 128);
    let mut pool_bytes = Vec::with_capacity(17_000_000);
    for value in [HISTORY as u32, 4096u32] {
        pool_bytes.extend_from_slice(&value.to_le_bytes());
    }
    for value in history_indexer_key
        .iter()
        .chain(&history_indexer_gate)
        .chain(&norm_weight)
        .chain(&norm_bias)
    {
        pool_bytes.extend_from_slice(&value.to_le_bytes());
    }
    for value in ape.iter().chain(&pooled_expected) {
        pool_bytes.extend_from_slice(&value.to_le_bytes());
    }
    std::fs::write(output_dir.join("lisa_long_pool_capture.bin"), pool_bytes)?;
    eprintln!(
        "[glm53-long-indexer] captured layer-{target_layer} pool evolution, scores, selection, and indexed attention at position {HISTORY} over 4096 model-produced pools"
    );
    Ok(())
}

#[test]
fn capture_real_16k_kda_prefill_inputs_for_ds4() -> anyhow::Result<()> {
    const DEFAULT_HISTORY: usize = 16_384;
    let Some(output_dir) = std::env::var_os("LISA_GLM53_LONG_DSA_CAPTURE_DIR") else {
        eprintln!("set LISA_GLM53_LONG_DSA_CAPTURE_DIR to capture real KDA prefill inputs");
        return Ok(());
    };
    if std::env::var_os("LISA_GLM53_KDA_PREFILL_DS4_CAPTURE").is_none() {
        eprintln!("set LISA_GLM53_KDA_PREFILL_DS4_CAPTURE=1 for the bounded KDA capture mode");
        return Ok(());
    }
    if !lisa_mlx::device::metal_available() {
        eprintln!("[glm53-kda-prefill-capture] skipping: no Metal device available");
        return Ok(());
    }
    let Some(model_dir) = std::env::var_os("LISA_GLM53_MODEL") else {
        eprintln!("[glm53-kda-prefill-capture] skipping: LISA_GLM53_MODEL is unset");
        return Ok(());
    };
    let layer_id = std::env::var("LISA_GLM53_KDA_PREFILL_LAYER")?.parse::<usize>()?;
    std::fs::create_dir_all(&output_dir)?;
    let dir = Path::new(&model_dir);
    let config = ModelConfig::from_json(&dir.join("config.json"))?;
    anyhow::ensure!(
        layer_id < config.num_hidden_layers,
        "configured layer {layer_id} exceeds model depth {}",
        config.num_hidden_layers
    );
    anyhow::ensure!(
        config.is_kda(layer_id)
            || std::env::var_os("LISA_GLM53_KDA_PREFILL_CHUNK_CAPTURE_PREFIX").is_some(),
        "configured layer {layer_id} is not KDA"
    );
    let tokenizer = crate::core::tokenizer::Tokenizer::load_file(&dir.join("tokenizer.json"))?;
    let seed = tokenizer.encode("The capital of France is Paris. ", false)?;
    anyhow::ensure!(
        !seed.is_empty(),
        "KDA prefill prompt tokenized to empty input"
    );
    let ids = if let Some(path) = std::env::var_os("LISA_GLM53_KDA_PREFILL_TOKEN_FILE") {
        let bytes = std::fs::read(path)?;
        anyhow::ensure!(
            bytes.len() % 4 == 0,
            "KDA prefill token file is not u32 aligned"
        );
        bytes
            .chunks_exact(4)
            .map(|value| u32::from_le_bytes(value.try_into().expect("four-byte token")))
            .collect::<Vec<_>>()
    } else {
        let mut repeated = Vec::with_capacity(DEFAULT_HISTORY);
        while repeated.len() < DEFAULT_HISTORY {
            let take = (DEFAULT_HISTORY - repeated.len()).min(seed.len());
            repeated.extend_from_slice(&seed[..take]);
        }
        repeated
    };
    anyhow::ensure!(!ids.is_empty(), "KDA prefill token file is empty");
    let history = ids.len();
    let tower = &mut load_prefix_tower(dir, &config, layer_id)?;
    tower.capture = Some(Capture {
        layers: vec![layer_id],
        state_only: false,
        kda_prefill_only: true,
        entries: std::sync::Mutex::new(HashMap::new()),
        shapes: std::sync::Mutex::new(HashMap::new()),
        flushed: std::sync::atomic::AtomicBool::new(false),
        ..Default::default()
    });
    let input = Array::from_slice(
        &ids.iter().map(|&id| id as i32).collect::<Vec<_>>(),
        &[1, history as i32],
    );
    let mut caches = tower.new_caches();
    if let Some(prefix) = std::env::var_os("LISA_GLM53_KDA_PREFILL_CHUNK_CAPTURE_PREFIX") {
        let prefix = PathBuf::from(prefix);
        let capture_keys = std::env::var("LISA_GLM53_KDA_PREFILL_CHUNK_CAPTURE_KEY")
            .unwrap_or_else(|_| "kda_input".to_string())
            .split(',')
            .map(str::trim)
            .filter(|key| !key.is_empty())
            .map(str::to_string)
            .collect::<Vec<_>>();
        anyhow::ensure!(
            !capture_keys.is_empty(),
            "KDA chunk capture key list is empty"
        );
        let chunk_count = ids.chunks(DS4_GLM53_PREFILL_CHUNK_TOKENS).len();
        let selected_chunk = std::env::var("LISA_GLM53_KDA_PREFILL_CHUNK_CAPTURE_INDEX")
            .ok()
            .map(|value| value.parse::<usize>())
            .transpose()?;
        if let Some(index) = selected_chunk {
            anyhow::ensure!(
                index < chunk_count,
                "selected KDA capture chunk {index} exceeds {chunk_count} chunks"
            );
        }
        for (chunk_index, chunk_ids) in ids.chunks(DS4_GLM53_PREFILL_CHUNK_TOKENS).enumerate() {
            let should_capture = selected_chunk.is_none_or(|selected| selected == chunk_index);
            tower.capture = should_capture.then(|| Capture {
                layers: vec![layer_id],
                key_filters: capture_keys.clone(),
                preserve_dispatch: true,
                entries: std::sync::Mutex::new(HashMap::new()),
                shapes: std::sync::Mutex::new(HashMap::new()),
                pending: std::sync::Mutex::new(HashMap::new()),
                flushed: std::sync::atomic::AtomicBool::new(false),
                ..Default::default()
            });
            let chunk_input = Array::from_slice(
                &chunk_ids.iter().map(|&id| id as i32).collect::<Vec<_>>(),
                &[1, chunk_ids.len() as i32],
            );
            tower.prefill_mode = true;
            let result = tower.forward_inner_until(&chunk_input, &mut caches, Some(layer_id));
            tower.prefill_mode = false;
            result?;
            if should_capture {
                let capture = tower
                    .capture
                    .as_ref()
                    .ok_or_else(|| anyhow::anyhow!("KDA chunk capture was not installed"))?;
                let path = PathBuf::from(format!(
                    "{}.chunk-{chunk_index:03}.npz",
                    prefix.to_string_lossy()
                ));
                capture.flush(
                    path.to_str()
                        .ok_or_else(|| anyhow::anyhow!("KDA chunk capture path is not UTF-8"))?,
                )?;
            }
            lisa_mlx::memory::trim_cache();
        }
        eprintln!(
            "[glm53-kda-prefill-capture] processed layer {layer_id} across {chunk_count} production-sized chunks"
        );
        return Ok(());
    }
    let _ = tower.forward_inner_until(&input, &mut caches, Some(layer_id))?;
    let capture = tower
        .capture
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("KDA prefill capture was not installed"))?;
    let entries = capture.entries.lock().expect("KDA prefill capture lock");
    let get = |stage: &str| -> anyhow::Result<&Vec<f32>> {
        entries
            .get(&format!("L{layer_id}.{stage}"))
            .ok_or_else(|| anyhow::anyhow!("missing KDA prefill capture L{layer_id}.{stage}"))
    };
    let q = get("kda_q_proj")?;
    let k = get("kda_k_proj")?;
    let v = get("kda_v_proj")?;
    let raw_beta = get("kda_beta_logits")?;
    let raw_gate = get("kda_gate_raw")?;
    let output_gate = get("kda_gb_proj")?;
    let output = get("kda_outnorm")?;
    let next_conv = get("kda_conv_state")?;
    let next_state = get("fused_state")?;
    let projection = config.linear_num_heads * config.linear_head_dim;
    let state_count = projection * config.linear_head_dim;
    anyhow::ensure!(
        [
            q.len(),
            k.len(),
            v.len(),
            raw_gate.len(),
            output_gate.len(),
            output.len()
        ]
        .iter()
        .all(|&length| length == history * projection),
        "captured KDA prefill projection does not match model geometry"
    );
    anyhow::ensure!(raw_beta.len() == history * config.linear_num_heads);
    anyhow::ensure!(next_conv.len() == 9 * projection);
    anyhow::ensure!(next_state.len() == state_count);
    let weights = match &tower.layers[layer_id].attn {
        Attn::Kda(kda) => kda.ds4_decode_weights_for_capture()?,
        Attn::Dsa(_) => anyhow::bail!("configured layer {layer_id} is not KDA"),
    };
    let (q_conv, k_conv, v_conv, a_log, dt_bias, output_norm) = &weights;
    let expected_conv_weights = config.linear_conv_kernel_dim * projection;
    anyhow::ensure!(
        [q_conv.len(), k_conv.len(), v_conv.len()]
            .iter()
            .all(|&length| length == expected_conv_weights)
            && a_log.len() == config.linear_num_heads
            && dt_bias.len() == projection
            && output_norm.len() == config.linear_head_dim,
        "captured KDA prefill weights do not match model geometry"
    );
    fn write_f32_le(file: &mut std::fs::File, values: &[f32]) -> anyhow::Result<()> {
        use std::io::Write;
        #[cfg(target_endian = "little")]
        {
            let bytes = unsafe {
                std::slice::from_raw_parts(
                    values.as_ptr().cast::<u8>(),
                    std::mem::size_of_val(values),
                )
            };
            file.write_all(bytes)?;
        }
        #[cfg(target_endian = "big")]
        for value in values {
            file.write_all(&value.to_le_bytes())?;
        }
        Ok(())
    }
    let capture_path =
        Path::new(&output_dir).join(format!("kda_layer{layer_id}_prefill_capture.bin"));
    let mut file = std::fs::File::create(&capture_path)?;
    for value in [
        layer_id as u32,
        config.linear_num_heads as u32,
        config.linear_head_dim as u32,
        config.linear_conv_kernel_dim as u32,
        config.linear_lower_bound.to_bits(),
        config.rms_norm_eps.to_bits(),
        history as u32,
    ] {
        use std::io::Write;
        file.write_all(&value.to_le_bytes())?;
    }
    for values in [
        q.as_slice(),
        k.as_slice(),
        v.as_slice(),
        raw_gate.as_slice(),
        raw_beta.as_slice(),
        output_gate.as_slice(),
        q_conv.as_slice(),
        k_conv.as_slice(),
        v_conv.as_slice(),
        a_log.as_slice(),
        dt_bias.as_slice(),
        output_norm.as_slice(),
        output.as_slice(),
        next_conv.as_slice(),
        next_state.as_slice(),
    ] {
        write_f32_le(&mut file, values)?;
    }
    let decode_initial_conv = next_conv.clone();
    let decode_initial_state = next_state.clone();
    drop(file);
    drop(entries);
    eprintln!(
        "[glm53-kda-prefill-capture] saved layer {layer_id}, {history} positions, {projection}-wide projections"
    );

    if std::env::var_os("LISA_GLM53_KDA_DECODE_DS4_CAPTURE").is_some() {
        let query_id = seed[history % seed.len()];
        tower.capture = Some(Capture {
            layers: vec![layer_id],
            state_only: false,
            kda_prefill_only: false,
            entries: std::sync::Mutex::new(HashMap::new()),
            shapes: std::sync::Mutex::new(HashMap::new()),
            flushed: std::sync::atomic::AtomicBool::new(false),
            ..Default::default()
        });
        let query = Array::from_slice(&[query_id as i32], &[1, 1]);
        let _ = tower.forward_inner_until(&query, &mut caches, Some(layer_id))?;
        let decode_capture = tower
            .capture
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("KDA decode capture was not installed"))?;
        let entries = decode_capture
            .entries
            .lock()
            .expect("KDA decode capture lock");
        let get_decode = |stage: &str| -> anyhow::Result<&Vec<f32>> {
            entries
                .get(&format!("L{layer_id}.{stage}"))
                .ok_or_else(|| anyhow::anyhow!("missing KDA decode capture L{layer_id}.{stage}"))
        };
        let q = get_decode("kda_q_proj")?;
        let k = get_decode("kda_k_proj")?;
        let v = get_decode("kda_v_proj")?;
        let raw_gate = get_decode("kda_gate_raw")?;
        let raw_beta = get_decode("kda_beta_logits")?;
        let output_gate = get_decode("kda_gb_proj")?;
        let output = get_decode("kda_decode_out")?;
        let next_conv = get_decode("kda_decode_conv_state")?;
        let next_state = get_decode("kda_decode_state")?;
        anyhow::ensure!(
            [
                q.len(),
                k.len(),
                v.len(),
                raw_gate.len(),
                output_gate.len(),
                output.len()
            ]
            .iter()
            .all(|&length| length == projection)
                && raw_beta.len() == config.linear_num_heads
                && next_conv.len() == 9 * projection
                && next_state.len() == state_count,
            "captured KDA decode does not match model geometry"
        );
        let decode_path =
            Path::new(&output_dir).join(format!("kda_layer{layer_id}_decode_capture.bin"));
        let mut decode_file = std::fs::File::create(&decode_path)?;
        let decode_header = [
            layer_id as u32,
            config.linear_num_heads as u32,
            config.linear_head_dim as u32,
            projection as u32,
            config.linear_conv_kernel_dim as u32,
            config.linear_lower_bound.to_bits(),
            config.rms_norm_eps.to_bits(),
        ];
        for value in decode_header {
            use std::io::Write;
            decode_file.write_all(&value.to_le_bytes())?;
        }
        for values in [
            q.as_slice(),
            k.as_slice(),
            v.as_slice(),
            raw_gate.as_slice(),
            raw_beta.as_slice(),
            output_gate.as_slice(),
            decode_initial_conv.as_slice(),
            decode_initial_state.as_slice(),
            weights.0.as_slice(),
            weights.1.as_slice(),
            weights.2.as_slice(),
            weights.3.as_slice(),
            weights.4.as_slice(),
            weights.5.as_slice(),
            output.as_slice(),
            next_conv.as_slice(),
            next_state.as_slice(),
        ] {
            write_f32_le(&mut decode_file, values)?;
        }
        let fixture: Option<Vec<u8>> = match layer_id {
            0 => Some(crate::models::glm53::test_fixtures::read(
                "kda_decode_layer0_model_long_ds4.bin",
            )),
            4 => Some(crate::models::glm53::test_fixtures::read(
                "kda_decode_layer4_model_long_ds4.bin",
            )),
            5 => Some(crate::models::glm53::test_fixtures::read(
                "kda_decode_layer5_model_long_ds4.bin",
            )),
            6 => Some(crate::models::glm53::test_fixtures::read(
                "kda_decode_layer6_model_long_ds4.bin",
            )),
            8 => Some(crate::models::glm53::test_fixtures::read(
                "kda_decode_layer8_model_long_ds4.bin",
            )),
            12 => Some(crate::models::glm53::test_fixtures::read(
                "kda_decode_layer12_model_long_ds4.bin",
            )),
            16 => Some(crate::models::glm53::test_fixtures::read(
                "kda_decode_layer16_model_long_ds4.bin",
            )),
            20 => Some(crate::models::glm53::test_fixtures::read(
                "kda_decode_layer20_model_long_ds4.bin",
            )),
            _ => None,
        };
        if let Some(fixture) =
            fixture.filter(|_| std::env::var_os("LISA_GLM53_KDA_LONG_CAPTURE_ONLY").is_none())
        {
            anyhow::ensure!(
                fixture.len() == 28 + (projection + 9 * projection + state_count) * 4,
                "layer-{layer_id} KDA decode fixture has an unexpected output size"
            );
            for (index, word) in decode_header.iter().enumerate() {
                let expected = u32::from_le_bytes(
                    fixture[index * 4..index * 4 + 4]
                        .try_into()
                        .expect("KDA fixture header word"),
                );
                anyhow::ensure!(
                    *word == expected,
                    "layer-{layer_id} KDA decode fixture header differs at word {index}"
                );
            }
            let mut offset = 28;
            for (label, values) in [
                ("output", output.as_slice()),
                ("convolution state", next_conv.as_slice()),
                ("recurrent state", next_state.as_slice()),
            ] {
                for (index, (actual, bytes)) in values
                    .iter()
                    .zip(fixture[offset..offset + values.len() * 4].chunks_exact(4))
                    .enumerate()
                {
                    let expected = f32::from_le_bytes(bytes.try_into().expect("KDA fixture value"));
                    anyhow::ensure!(
                        actual.to_bits() == expected.to_bits(),
                        "layer-{layer_id} KDA decode {label} differs from DS4 at {index}: {actual} vs {expected}"
                    );
                }
                offset += values.len() * 4;
            }
            anyhow::ensure!(offset == fixture.len());
        }
        eprintln!(
            "[glm53-kda-prefill-capture] saved layer {layer_id} decode from the retained 16K state"
        );
    }
    Ok(())
}
