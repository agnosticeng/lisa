use super::*;

#[test]
fn prefetch_budget_uses_selected_payload_bytes() {
    let expert_bytes = vec![8; 64];
    let many_small_experts = (0..20).collect::<Vec<_>>();
    assert_eq!(
        selected_prefetch_bytes(&many_small_experts, &expert_bytes),
        Some(160)
    );
    assert_eq!(
        selected_prefetch_bytes(&many_small_experts, &expert_bytes)
            .is_some_and(|bytes| bytes <= 160),
        true,
        "a prefill selection above the old 16-expert cutoff should fit by bytes"
    );
    assert_eq!(
        selected_prefetch_bytes(&many_small_experts, &expert_bytes)
            .is_some_and(|bytes| bytes <= 159),
        false,
        "read-ahead must stay within the per-pool byte budget"
    );
}

#[test]
fn prefetch_budget_rejects_invalid_or_overflowing_expert_sizes() {
    assert_eq!(selected_prefetch_bytes(&[0, 1], &[4]), None);
    assert_eq!(selected_prefetch_bytes(&[0, 1], &[usize::MAX, 1]), None);
}

#[test]
fn capture_real_single_layer_moe_route_for_ds4() -> anyhow::Result<()> {
    let (Some(model_dir), Some(input_path), Some(capture_path)) = (
        std::env::var_os("LISA_GLM53_MODEL"),
        std::env::var_os("LISA_GLM53_MOE_SINGLE_LAYER_INPUT_F32"),
        std::env::var_os("LISA_GLM53_MOE_SINGLE_LAYER_CAPTURE_NPZ"),
    ) else {
        eprintln!(
            "set LISA_GLM53_MODEL, LISA_GLM53_MOE_SINGLE_LAYER_INPUT_F32, and LISA_GLM53_MOE_SINGLE_LAYER_CAPTURE_NPZ to run the isolated real-weight MoE capture"
        );
        return Ok(());
    };
    if !lisa_mlx::device::metal_available() {
        eprintln!("[glm53-moe-single-layer] skipping: no Metal device available");
        return Ok(());
    }

    let dir = std::path::Path::new(&model_dir);
    let config = super::super::config::ModelConfig::from_json(&dir.join("config.json"))?;
    let layer = std::env::var("LISA_GLM53_MOE_SINGLE_LAYER")
        .ok()
        .map(|value| value.parse::<usize>())
        .transpose()?
        .unwrap_or(9);
    anyhow::ensure!(
        layer < config.num_hidden_layers && config.is_sparse_mlp(layer),
        "layer {layer} is not a sparse GLM layer"
    );
    let store = Arc::new(LazyTensorStore::open(dir)?);
    let prefix = format!("model.layers.{layer}.mlp");
    let mut weights = Weights::new();
    for name in [
        format!("{prefix}.gate.weight"),
        format!("{prefix}.gate.e_score_correction_bias"),
        format!("{prefix}.shared_experts.gate_proj.weight"),
        format!("{prefix}.shared_experts.gate_proj.scales"),
        format!("{prefix}.shared_experts.gate_proj.biases"),
        format!("{prefix}.shared_experts.up_proj.weight"),
        format!("{prefix}.shared_experts.up_proj.scales"),
        format!("{prefix}.shared_experts.up_proj.biases"),
        format!("{prefix}.shared_experts.down_proj.weight"),
        format!("{prefix}.shared_experts.down_proj.scales"),
        format!("{prefix}.shared_experts.down_proj.biases"),
    ] {
        weights.insert(name.clone(), store.get_mapped(&name)?);
    }
    let map_all_expert_weights = std::env::var_os("LISA_GLM53_MOE_SINGLE_LAYER_MAP_ALL").is_some();
    let mut moe = MoE::load::<Weights>(
        &mut weights,
        &prefix,
        config.n_routed_experts,
        config.num_experts_per_tok,
        config.n_group,
        config.topk_group,
        config.routed_scaling_factor,
        config.norm_topk_prob,
        config.n_shared_experts,
        config.moe_intermediate_size,
        config.hidden_size,
        config.swiglu_limit,
        Some(Arc::clone(&store)),
        map_all_expert_weights,
    )?;
    if !map_all_expert_weights {
        moe.set_expert_cache_capacity(1024 * 1024 * 1024);
        if std::env::var_os("LISA_GLM53_MOE_SINGLE_LAYER_CACHE_RESIDENCY").is_some() {
            let residency = moe
                .new_cache_residency_set()
                .ok_or_else(|| anyhow::anyhow!("single-layer cache residency set unavailable"))?;
            if std::env::var_os("LISA_GLM53_MOE_SINGLE_LAYER_QUEUE_REGISTER").is_some() {
                anyhow::ensure!(
                    Array::add_residency_set_to_queue_without_request(&residency),
                    "single-layer cache residency set could not be attached to the Metal queue"
                );
            }
            moe.set_model_residency_set(residency);
        }
    }

    let input_bytes = std::fs::read(&input_path)?;
    anyhow::ensure!(
        input_bytes.len() == config.hidden_size * std::mem::size_of::<f32>(),
        "single-layer MoE input has {} bytes, expected {}",
        input_bytes.len(),
        config.hidden_size * std::mem::size_of::<f32>()
    );
    let input = input_bytes
        .chunks_exact(4)
        .map(|word| f32::from_le_bytes(word.try_into().expect("four-byte input")))
        .collect::<Vec<_>>();
    let input = Array::from_slice(&input, &[1, 1, config.hidden_size as i32]);
    let capture = Capture {
        layers: vec![layer],
        preserve_dispatch: std::env::var_os("LISA_GLM53_MOE_SINGLE_LAYER_PRESERVE_DISPATCH")
            .is_some(),
        ..Default::default()
    };
    let output = moe.forward_with_capture(&input, Some(&capture), layer)?;
    capture.flush(
        std::path::Path::new(&capture_path)
            .to_str()
            .ok_or_else(|| anyhow::anyhow!("single-layer MoE capture path is not UTF-8"))?,
    )?;
    if let Some(output_path) = std::env::var_os("LISA_GLM53_MOE_SINGLE_LAYER_OUTPUT_F32") {
        let values = output
            .as_dtype(Dtype::Float32)?
            .contiguous()?
            .to_vec1::<f32>()?;
        let bytes = values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>();
        std::fs::write(output_path, bytes)?;
    }
    store.discard_cached_pages()?;
    eprintln!(
        "[glm53-moe-single-layer] captured real layer {layer} route (map_all_expert_weights={map_all_expert_weights}) to {}",
        capture_path.to_string_lossy()
    );
    Ok(())
}

#[test]
fn glm53_dense_q8_concatenated_gate_up_matches_separate_real_checkpoint_projections()
-> anyhow::Result<()> {
    if !lisa_mlx::device::metal_available() {
        eprintln!("[glm53-dense-q8-pair] skipping: no Metal device available");
        return Ok(());
    }
    let Some(model_dir) = std::env::var_os("LISA_GLM53_MODEL") else {
        eprintln!("[glm53-dense-q8-pair] skipping: LISA_GLM53_MODEL is unset");
        return Ok(());
    };
    let dir = std::path::Path::new(&model_dir);
    let config = super::super::config::ModelConfig::from_json(&dir.join("config.json"))?;
    let store = LazyTensorStore::open(dir)?;
    let prefix = "model.layers.0.mlp";
    let mut weights = Weights::new();
    for projection in ["gate_proj", "up_proj", "down_proj"] {
        for field in ["weight", "scales", "biases"] {
            let name = format!("{prefix}.{projection}.{field}");
            weights.insert(name.clone(), store.get_owned(&name)?);
        }
    }
    let mut gate = QuantizedLinear::load(&mut weights, prefix, "gate_proj")?;
    let mut up = QuantizedLinear::load(&mut weights, prefix, "up_proj")?;
    let mut down = QuantizedLinear::load(&mut weights, prefix, "down_proj")?;
    for projection in [&mut gate, &mut up, &mut down] {
        projection.set_quant(64, 8);
        projection.scales = projection.scales.as_dtype(Dtype::Float32)?;
        projection.biases = projection.biases.as_dtype(Dtype::Float32)?;
    }
    let paired_linear = fuse_gate_up(&gate, &up)?;
    let input: Vec<f32> = (0..config.hidden_size)
        .map(|i| (((i * 29 + 17) % 257) as f32 - 128.0) * 0.0002)
        .collect();
    let x = Array::from_slice(&input, &[1, config.hidden_size as i32]);

    let gate_array = gate.forward_glm53_prefill(&x)?;
    let up_array = up.forward_glm53_prefill(&x)?;
    let gate_values = gate_array.to_vec1::<f32>()?;
    let up_values = up_array.to_vec1::<f32>()?;
    let paired = paired_linear.forward_glm53_prefill(&x)?;
    let (paired_gate, paired_up) = split_gate_up(&paired, gate.dims_out())?;
    let paired_gate_values = paired_gate.to_vec1::<f32>()?;
    let paired_up_values = paired_up.to_vec1::<f32>()?;
    anyhow::ensure!(
        gate_values
            .iter()
            .map(|value| value.to_bits())
            .eq(paired_gate_values.iter().map(|value| value.to_bits())),
        "concatenated dense Q8 gate projection differs from its separate real-checkpoint projection"
    );
    anyhow::ensure!(
        up_values
            .iter()
            .map(|value| value.to_bits())
            .eq(paired_up_values.iter().map(|value| value.to_bits())),
        "concatenated dense Q8 up projection differs from its separate real-checkpoint projection"
    );
    let expected_activation = clamped_swiglu(&up_array, &gate_array, config.swiglu_limit)?;
    let paired_activation = clamped_swiglu(&paired_up, &paired_gate, config.swiglu_limit)?;
    let expected_activation_values = expected_activation.to_vec1::<f32>()?;
    let paired_activation_values = paired_activation.to_vec1::<f32>()?;
    anyhow::ensure!(
        expected_activation_values
            .iter()
            .map(|value| value.to_bits())
            .eq(paired_activation_values.iter().map(|value| value.to_bits())),
        "concatenated dense Q8 gate/up activation differs from separate real-checkpoint projections"
    );

    let mut mlp_weights = Weights::new();
    for projection in ["gate_proj", "up_proj", "down_proj"] {
        for field in ["weight", "scales", "biases"] {
            let name = format!("{prefix}.{projection}.{field}");
            mlp_weights.insert(name.clone(), store.get_owned(&name)?);
        }
    }
    let mlp = DenseMlp::load::<Weights>(&mut mlp_weights, prefix, config.swiglu_limit)?;
    let expected_output = down.forward_glm53_prefill(&expected_activation)?;
    let actual_output = mlp.forward(&x)?;
    anyhow::ensure!(
        expected_output
            .to_vec1::<f32>()?
            .iter()
            .map(|value| value.to_bits())
            .eq(actual_output
                .to_vec1::<f32>()?
                .iter()
                .map(|value| value.to_bits())),
        "concatenated dense Q8 MLP output differs from separate real-checkpoint projections"
    );

    let mut batch_input = input.clone();
    batch_input.extend(input.iter().map(|value| -*value * 0.75 + 0.001));
    let batch_x = Array::from_slice(&batch_input, &[2, 1, config.hidden_size as i32]);
    let batch_gate_array = gate.forward_glm53_prefill(&batch_x)?;
    let batch_up_array = up.forward_glm53_prefill(&batch_x)?;
    let batch_pair = paired_linear.forward_glm53_prefill(&batch_x)?;
    let (batch_pair_gate, batch_pair_up) = split_gate_up(&batch_pair, gate.dims_out())?;
    let batch_gate = batch_gate_array.reshape(&[(2 * gate.dims_out()) as i32])?;
    let batch_up = batch_up_array.reshape(&[(2 * up.dims_out()) as i32])?;
    let batch_pair_gate = batch_pair_gate.reshape(&[(2 * gate.dims_out()) as i32])?;
    let batch_pair_up = batch_pair_up.reshape(&[(2 * up.dims_out()) as i32])?;
    anyhow::ensure!(
        batch_gate
            .to_vec1::<f32>()?
            .iter()
            .map(|value| value.to_bits())
            .eq(batch_pair_gate
                .to_vec1::<f32>()?
                .iter()
                .map(|value| value.to_bits()))
            && batch_up
                .to_vec1::<f32>()?
                .iter()
                .map(|value| value.to_bits())
                .eq(batch_pair_up
                    .to_vec1::<f32>()?
                    .iter()
                    .map(|value| value.to_bits())),
        "concatenated dense Q8 batch projections differ from separate real-checkpoint projections"
    );
    let batch_expected_activation =
        clamped_swiglu(&batch_up_array, &batch_gate_array, config.swiglu_limit)?;
    let batch_expected = down.forward_glm53_prefill(&batch_expected_activation)?;
    let batch_actual = mlp.forward(&batch_x)?;
    anyhow::ensure!(
        batch_expected
            .reshape(&[(2 * config.hidden_size) as i32])?
            .to_vec1::<f32>()?
            .iter()
            .map(|value| value.to_bits())
            .eq(batch_actual
                .reshape(&[(2 * config.hidden_size) as i32])?
                .to_vec1::<f32>()?
                .iter()
                .map(|value| value.to_bits())),
        "concatenated dense Q8 batch MLP output differs from separate real-checkpoint projections"
    );

    // Exercise the production Q8 prefill route on a non-tile-aligned row
    // count, separately from the one-row and batch decode checks above.
    let prefill_rows = 4usize;
    let prefill_input: Vec<f32> = (0..prefill_rows * config.hidden_size)
        .map(|i| (((i * 31 + 23) % 509) as f32 - 254.0) * 0.0001)
        .collect();
    let prefill_x = Array::from_slice(
        &prefill_input,
        &[1, prefill_rows as i32, config.hidden_size as i32],
    );
    let prefill_gate = gate.forward_glm53_prefill(&prefill_x)?;
    let prefill_up = up.forward_glm53_prefill(&prefill_x)?;
    let prefill_pair = paired_linear.forward_glm53_prefill(&prefill_x)?;
    let (prefill_pair_gate, prefill_pair_up) = split_gate_up(&prefill_pair, gate.dims_out())?;
    anyhow::ensure!(
        prefill_gate
            .to_vec1::<f32>()?
            .iter()
            .map(|value| value.to_bits())
            .eq(prefill_pair_gate
                .to_vec1::<f32>()?
                .iter()
                .map(|value| value.to_bits()))
            && prefill_up
                .to_vec1::<f32>()?
                .iter()
                .map(|value| value.to_bits())
                .eq(prefill_pair_up
                    .to_vec1::<f32>()?
                    .iter()
                    .map(|value| value.to_bits())),
        "concatenated dense Q8 prefill projections differ from separate real-checkpoint projections"
    );
    let prefill_expected_activation =
        clamped_swiglu(&prefill_up, &prefill_gate, config.swiglu_limit)?;
    let prefill_expected = down.forward_glm53_prefill(&prefill_expected_activation)?;
    let prefill_actual = mlp.forward(&prefill_x)?;
    anyhow::ensure!(
        prefill_expected
            .to_vec1::<f32>()?
            .iter()
            .map(|value| value.to_bits())
            .eq(prefill_actual
                .to_vec1::<f32>()?
                .iter()
                .map(|value| value.to_bits())),
        "concatenated dense Q8 prefill MLP output differs from separate real-checkpoint projections"
    );
    Ok(())
}

#[test]
fn glm53_dense_q8_ds4_replay_capture() -> anyhow::Result<()> {
    let Some(capture_root) = std::env::var_os("LISA_GLM53_DENSE_Q8_REPLAY_CAPTURE") else {
        eprintln!("[glm53-dense-q8-replay] skipping: capture path is unset");
        return Ok(());
    };
    let Some(input_path) = std::env::var_os("LISA_GLM53_DENSE_Q8_REPLAY_INPUT") else {
        eprintln!("[glm53-dense-q8-replay] skipping: input path is unset");
        return Ok(());
    };
    if !lisa_mlx::device::metal_available() {
        eprintln!("[glm53-dense-q8-replay] skipping: no Metal device available");
        return Ok(());
    }
    let model_dir = std::env::var_os("LISA_GLM53_MODEL")
        .ok_or_else(|| anyhow::anyhow!("LISA_GLM53_MODEL is required for Q8 replay"))?;
    let dir = std::path::Path::new(&model_dir);
    let config = super::super::config::ModelConfig::from_json(&dir.join("config.json"))?;
    let input_bytes = std::fs::read(input_path)?;
    anyhow::ensure!(
        input_bytes.len() % 4 == 0,
        "Q8 replay input is not F32 aligned"
    );
    let input: Vec<f32> = input_bytes
        .chunks_exact(4)
        .map(|bytes| f32::from_le_bytes(bytes.try_into().expect("four-byte F32")))
        .collect();
    anyhow::ensure!(
        input.len() % config.hidden_size == 0,
        "Q8 replay input length does not match the hidden size"
    );
    let rows = input.len() / config.hidden_size;
    anyhow::ensure!(rows >= 32, "Q8 replay needs a prefill-sized input");
    let x = Array::from_slice(&input, &[rows as i32, config.hidden_size as i32]);
    let store = LazyTensorStore::open(dir)?;
    let capture_root = std::path::Path::new(&capture_root);
    std::fs::create_dir_all(capture_root)?;
    let mut gate_output = None;
    let mut up_output = None;

    for name in ["gate_proj", "up_proj"] {
        let prefix = format!("model.layers.0.mlp.{name}");
        let mut weights = Weights::new();
        for field in ["weight", "scales", "biases"] {
            let key = format!("{prefix}.{field}");
            weights.insert(key.clone(), store.get_owned(&key)?);
        }
        let mut projection = QuantizedLinear::load(&mut weights, "model.layers.0.mlp", name)?;
        projection.set_quant(64, 8);
        let output_array = projection.forward_glm53_prefill(&x)?;
        let output = output_array.to_vec1::<f32>()?;
        if name == "gate_proj" {
            gate_output = Some(output_array);
        } else {
            up_output = Some(output_array);
        }
        let stage_dir = capture_root.join(name);
        std::fs::create_dir_all(&stage_dir)?;
        std::fs::write(stage_dir.join("input.f32"), encode_f32(&input))?;
        std::fs::write(stage_dir.join("lisa_output.f32"), encode_f32(&output))?;
        eprintln!(
            "[glm53-dense-q8-replay] wrote layer-0 {name}: {rows} rows, {} outputs",
            output.len()
        );
    }

    let gate = gate_output.ok_or_else(|| anyhow::anyhow!("missing dense gate replay"))?;
    let up = up_output.ok_or_else(|| anyhow::anyhow!("missing dense up replay"))?;
    let activation = clamped_swiglu(&up, &gate, config.swiglu_limit)?;
    let swiglu_dir = capture_root.join("swiglu");
    std::fs::create_dir_all(&swiglu_dir)?;
    std::fs::write(
        swiglu_dir.join("gate.f32"),
        encode_f32(&gate.to_vec1::<f32>()?),
    )?;
    std::fs::write(swiglu_dir.join("up.f32"), encode_f32(&up.to_vec1::<f32>()?))?;
    std::fs::write(
        swiglu_dir.join("lisa_output.f32"),
        encode_f32(&activation.to_vec1::<f32>()?),
    )?;

    let prefix = "model.layers.0.mlp.down_proj";
    let mut weights = Weights::new();
    for field in ["weight", "scales", "biases"] {
        let key = format!("{prefix}.{field}");
        weights.insert(key.clone(), store.get_owned(&key)?);
    }
    let mut down = QuantizedLinear::load(&mut weights, "model.layers.0.mlp", "down_proj")?;
    down.set_quant(64, 8);
    let output = down.forward_glm53_prefill(&activation)?.to_vec1::<f32>()?;
    let down_dir = capture_root.join("down_proj");
    std::fs::create_dir_all(&down_dir)?;
    std::fs::write(
        down_dir.join("input.f32"),
        encode_f32(&activation.to_vec1::<f32>()?),
    )?;
    std::fs::write(down_dir.join("lisa_output.f32"), encode_f32(&output))?;
    eprintln!(
        "[glm53-dense-q8-replay] wrote layer-0 SwiGLU and down output: {} values",
        output.len()
    );
    Ok(())
}

#[cfg(feature = "glm53-fixtures")]
#[test]
fn glm53_shared_q8_mlp_matches_separate_ds4_path_real_mtp_weights() -> anyhow::Result<()> {
    if !lisa_mlx::device::metal_available() {
        eprintln!("[glm53-shared-q8-real] skipping: no Metal device available");
        return Ok(());
    }
    let Some(model_dir) = std::env::var_os("LISA_GLM53_MODEL") else {
        eprintln!("[glm53-shared-q8-real] skipping: LISA_GLM53_MODEL is unset");
        return Ok(());
    };
    let external_fixture: &[u8] =
        &crate::models::glm53::test_fixtures::read("mtp_shared_layer45_ds4.bin");
    let dir = std::path::Path::new(&model_dir);
    let config = super::super::config::ModelConfig::from_json(&dir.join("config.json"))?;
    let store = LazyTensorStore::open(dir)?;
    let prefix = "model.layers.45.mlp.shared_experts";
    let mut weights = Weights::new();
    for projection in ["gate_proj", "up_proj", "down_proj"] {
        for field in ["weight", "scales", "biases"] {
            let name = format!("{prefix}.{projection}.{field}");
            weights.insert(name.clone(), store.get_owned(&name)?);
        }
    }
    let mlp = DenseMlp::load_shared::<Weights>(&mut weights, prefix, config.swiglu_limit)?;
    let (gate, up) = mlp.gate_up.split_linears();
    let hidden = config.hidden_size;
    anyhow::ensure!(
        external_fixture.len() >= 16 + hidden * 4,
        "invalid MTP shared fixture"
    );
    let fixture_hidden =
        u32::from_le_bytes(external_fixture[0..4].try_into().expect("hidden width"));
    anyhow::ensure!(
        fixture_hidden as usize == hidden,
        "MTP fixture hidden width changed"
    );
    let input = external_fixture[16..16 + hidden * 4]
        .chunks_exact(4)
        .map(|bytes| f32::from_le_bytes(bytes.try_into().expect("F32 input word")))
        .collect::<Vec<_>>();
    let x = Array::from_slice(&input, &[1, 1, hidden as i32]);
    let gate_output = gate.forward_glm53_prefill(&x)?;
    let up_output = up.forward_glm53_prefill(&x)?;
    let expected = clamped_swiglu(&up_output, &gate_output, mlp.limit)?;
    let expected_output = mlp.down.forward_glm53_prefill(&expected)?;
    let actual_output = mlp.forward(&x)?;
    anyhow::ensure!(
        actual_output
            .to_vec1::<f32>()?
            .iter()
            .map(|value| value.to_bits())
            .eq(expected_output
                .to_vec1::<f32>()?
                .iter()
                .map(|value| value.to_bits())),
        "real shared Q8 MLP differs from its separate DS4-path projections"
    );
    Ok(())
}

#[test]
fn glm53_moe_mapped_checkpoint_expert_matches_gathered_q2_path() -> anyhow::Result<()> {
    if !lisa_mlx::device::metal_available() {
        eprintln!("[glm53-moe-mapped] skipping: no Metal device available");
        return Ok(());
    }
    let Some(model_dir) = std::env::var_os("LISA_GLM53_MODEL") else {
        eprintln!("[glm53-moe-mapped] skipping: LISA_GLM53_MODEL is unset");
        return Ok(());
    };
    let dir = std::path::Path::new(&model_dir);
    let config = super::super::config::ModelConfig::from_json(&dir.join("config.json"))?;
    let store = Arc::new(LazyTensorStore::open(dir)?);
    // Confirm the mapped routed checkpoint keeps packed Q2 bytes exact
    // instead of exposing its safetensors U32 shape.
    let fallback_name = "model.layers.7.mlp.experts.87.gate_proj.weight";
    let fallback_bytes = store.get_mapped_weight_bytes(fallback_name)?;
    anyhow::ensure!(
        fallback_bytes.dtype() == Dtype::Uint8,
        "fallback expert weight should be exposed as bytes, got {:?}",
        fallback_bytes.dtype()
    );
    let fallback_words = store.get_owned(fallback_name)?.to_vec1::<u32>()?;
    let expected_bytes: Vec<u8> = fallback_words
        .iter()
        .flat_map(|word| word.to_le_bytes())
        .collect();
    anyhow::ensure!(
        fallback_bytes.to_vec1::<u8>()? == expected_bytes,
        "fallback expert weight byte view changed the checkpoint payload"
    );
    // Cover experts on both shard sides, the shard transition, and the
    // final tensor near EOF through page-scoped mappings.
    let expert_ids = [17usize, 174, 175, 200, 255];
    let (bank_experts, _) = ExpertPools::load_experts(
        &store,
        "model.layers.3.mlp",
        config.n_routed_experts,
        true,
        false,
    )?;
    let bank_address_table = ExpertPools::address_table(&bank_experts)?.0;
    let bank_fields = bank_experts
        .iter()
        .flat_map(ExpertPools::arrays)
        .collect::<Vec<_>>();
    let (bank_resource_table, bank_resources) =
        lisa_mlx::models::glm53::kernels::moe_mapped_resource_table(&bank_fields)
            .ok_or_else(|| anyhow::anyhow!("full expert bank exceeds mapped resource limits"))?;
    anyhow::ensure!(
        bank_resources.len() <= 8,
        "layer-3 full expert bank should share shard-backed resources; found {} resources",
        bank_resources.len()
    );
    let bank_resource_refs = bank_resources.iter().collect::<Vec<_>>();
    let (mapped_experts, expert_cache_bytes) = ExpertPools::load_experts(
        &store,
        "model.layers.3.mlp",
        config.n_routed_experts,
        false,
        false,
    )?;
    let mapped_buffers: std::collections::HashSet<usize> = mapped_experts
        .iter()
        .flat_map(|expert| {
            [
                &expert.gate.weight,
                &expert.gate.scales,
                &expert.gate.biases,
                &expert.up.weight,
                &expert.up.scales,
                &expert.up.biases,
                &expert.down.weight,
                &expert.down.scales,
                &expert.down.biases,
            ]
        })
        .map(Array::backing_buffer_id)
        .collect();
    eprintln!(
        "[glm53-moe-mapped] Metal max buffer length={} bytes",
        Array::max_buffer_length()
    );
    anyhow::ensure!(
        mapped_buffers.len() == 2,
        "unloaded layer-3 GLM experts should use weight and metadata placeholders; found {}",
        mapped_buffers.len()
    );
    store.discard_cached_pages()?;
    eprintln!(
        "[glm53-moe-mapped] unloaded layer-3 expert bank uses {} placeholder allocations",
        mapped_buffers.len(),
    );
    let (hidden, inter) = (config.hidden_size, config.moe_intermediate_size);
    let (pool_address_table, pool_metadata_dtype) = ExpertPools::address_table(&mapped_experts)?;
    let cache_capacity = *expert_cache_bytes
        .first()
        .ok_or_else(|| anyhow::anyhow!("layer-3 expert bank is empty"))?;
    let staged_pools = Arc::new(ExpertPools {
        experts: mapped_experts.clone(),
        expert_cache_bytes,
        lazy_store: Some(Arc::clone(&store)),
        expert_prefix: Some("model.layers.3.mlp".to_string()),
        inter,
        address_table: pool_address_table,
        metadata_dtype: pool_metadata_dtype,
        residency_set: std::sync::OnceLock::new(),
        gpu_address_all_experts: false,
        cache_capacity_bytes: AtomicUsize::new(cache_capacity),
        cache: Mutex::new(ExpertCacheState::default()),
    });
    let input_values: Vec<f32> = (0..hidden)
        .map(|i| (((i * 31 + 17) % 251) as f32 - 125.0) * 0.0007)
        .collect();
    let input = Array::from_slice(&input_values, &[1, hidden as i32]);
    let selected = Array::from_slice(&[0i32], &[1]);
    let route_value = 0.73f32;
    let route = Array::from_slice(&[route_value], &[1]);
    let stream = lisa_mlx::Stream::thread_local_or_default();
    let mut cache_experts = mapped_experts.clone();
    for &expert_id in &expert_ids {
        let expert = staged_pools.load_lazy_expert_weights(expert_id)?;
        cache_experts[expert_id] = expert.clone();
        let mut direct_experts = mapped_experts.clone();
        direct_experts[expert_id] = expert.clone();
        let indirect_expert_buffers =
            ExpertPools::indirect_resources(&direct_experts, &[expert_id as i32])?;
        anyhow::ensure!(
            indirect_expert_buffers.len() == 9,
            "selected layer-3 expert should expose three weights and six mapped metadata buffers; found {}",
            indirect_expert_buffers.len()
        );
        let selected_keepalive: Vec<&Array> = indirect_expert_buffers.iter().collect();
        let kernel_keepalive: &[&Array] = &selected_keepalive;
        let (address_table, metadata_dtype) =
            ExpertPools::address_table(std::slice::from_ref(&expert))?;
        anyhow::ensure!(
            expert.gate.weight.size() * expert.gate.weight.dtype().size_of() == inter * hidden / 4,
            "unexpected deferred layer-3 expert gate shape {:?} dtype {:?}",
            expert.gate.weight.shape(),
            expert.gate.weight.dtype()
        );

        let pair_addr = lisa_mlx::models::glm53::kernels::moe_pair_swiglu_addr(
            &address_table,
            &input,
            &selected,
            &route,
            kernel_keepalive,
            inter,
            1,
            config.swiglu_limit,
            metadata_dtype,
            None,
            &stream,
        )
        .ok_or_else(|| anyhow::anyhow!("deferred expert pair-address dispatch declined"))?;
        let gate_weight = store.get_owned(&format!(
            "model.layers.3.mlp.experts.{expert_id}.gate_proj.weight"
        ))?;
        let up_weight = store.get_owned(&format!(
            "model.layers.3.mlp.experts.{expert_id}.up_proj.weight"
        ))?;
        let pair_weight = ops::concatenate(&[&gate_weight, &up_weight], 0)?
            .expand_dims(0)?
            .contiguous()?;
        let pair_scale = ops::concatenate(
            &[
                &expert.gate.scales.as_dtype(Dtype::Float32)?,
                &expert.up.scales.as_dtype(Dtype::Float32)?,
            ],
            0,
        )?
        .expand_dims(0)?
        .contiguous()?;
        let pair_bias = ops::concatenate(
            &[
                &expert.gate.biases.as_dtype(Dtype::Float32)?,
                &expert.up.biases.as_dtype(Dtype::Float32)?,
            ],
            0,
        )?
        .expand_dims(0)?
        .contiguous()?;
        let pair_gathered = lisa_mlx::models::glm53::kernels::moe_pair_swiglu(
            &pair_weight,
            &pair_scale,
            &pair_bias,
            &input,
            &selected,
            &route,
            inter,
            1,
            config.swiglu_limit,
            &stream,
        )
        .ok_or_else(|| anyhow::anyhow!("deferred expert gathered pair dispatch declined"))?;
        let pair_a = pair_addr.to_vec1::<f32>()?;
        let pair_b = pair_gathered.to_vec1::<f32>()?;
        anyhow::ensure!(
            pair_b.iter().any(|value| value.abs() > 1e-6),
            "real layer-3 expert pair path unexpectedly produced all zeros"
        );
        let pair_error = pair_a
            .iter()
            .zip(&pair_b)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        anyhow::ensure!(
            pair_error <= 3e-4,
            "real deferred expert pair max abs error is {pair_error}"
        );

        // Exercise the same selected expert through the full layer bank,
        // with each backing safetensors shard bound once as an ordinary
        // Metal resource. This catches invalid byte offsets at shard
        // transitions as well as mapped-kernel arithmetic differences.
        let bank_selected = Array::from_slice(&[expert_id as i32], &[1]);
        let bank_address_pair = lisa_mlx::models::glm53::kernels::moe_pair_swiglu_addr(
            &bank_address_table,
            &input,
            &bank_selected,
            &route,
            &bank_resource_refs,
            inter,
            1,
            config.swiglu_limit,
            metadata_dtype,
            None,
            &stream,
        )
        .ok_or_else(|| anyhow::anyhow!("full-bank address pair dispatch declined"))?;
        let bank_address_values = bank_address_pair.to_vec1::<f32>()?;
        let address_bank_error = bank_address_values
            .iter()
            .zip(&pair_a)
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0f32, f32::max);
        eprintln!(
            "[glm53-moe-mapped] expert {expert_id} full-bank address max abs delta={address_bank_error}"
        );
        let bank_pair = lisa_mlx::models::glm53::kernels::moe_pair_swiglu_mapped(
            &bank_resource_table,
            &input,
            &bank_selected,
            &route,
            &bank_resources,
            inter,
            1,
            config.swiglu_limit,
            metadata_dtype,
            &stream,
        )
        .ok_or_else(|| anyhow::anyhow!("full-bank mapped pair dispatch declined"))?;
        let bank_pair_values = bank_pair.to_vec1::<f32>()?;
        let mapped_pair_first = bank_pair_values
            .iter()
            .zip(&bank_address_values)
            .position(|(actual, expected)| actual.to_bits() != expected.to_bits());
        if let Some(index) = mapped_pair_first {
            let max_abs = bank_pair_values
                .iter()
                .zip(&bank_address_values)
                .map(|(actual, expected)| (actual - expected).abs())
                .fold(0.0f32, f32::max);
            anyhow::bail!(
                "expert {expert_id} full-bank mapped pair differs at {index}: {} vs {} (max abs {max_abs})",
                bank_pair_values[index],
                bank_address_values[index]
            );
        }

        let selected8_values = [expert_id as i32; 8];
        let selected8 = Array::from_slice(&selected8_values, &[8]);
        let routed_slots = (0..8).map(|_| &pair_addr).collect::<Vec<_>>();
        let routed_x = ops::concatenate(&routed_slots, 0)?;
        let bank_down = lisa_mlx::models::glm53::kernels::moe_gather_qmv_sum8_mapped(
            &bank_resource_table,
            &routed_x,
            &selected8,
            &bank_resources,
            hidden,
            metadata_dtype,
            &stream,
        )
        .ok_or_else(|| anyhow::anyhow!("full-bank mapped down dispatch declined"))?;
        let address_down = lisa_mlx::models::glm53::kernels::moe_gather_qmv_sum8_addr_ds4(
            &bank_address_table,
            &routed_x,
            &selected8,
            &bank_resource_refs,
            None,
            hidden,
            metadata_dtype,
            None,
            &stream,
        )
        .ok_or_else(|| anyhow::anyhow!("full-bank address down dispatch declined"))?;
        let bank_down_values = bank_down.to_vec1::<f32>()?;
        let address_down_values = address_down.to_vec1::<f32>()?;
        anyhow::ensure!(
            bank_down_values
                .iter()
                .map(|value| value.to_bits())
                .eq(address_down_values.iter().map(|value| value.to_bits())),
            "expert {expert_id} full-bank mapped down differs from the address path"
        );

        let (staged_address_table, staged_resources) =
            staged_pools.dispatch_resources(&[expert_id as i32])?;
        let staged_keepalive: Vec<&Array> = staged_resources.iter().collect();
        let staged_selected = Array::from_slice(&[expert_id as i32], &[1]);
        let staged_pair = lisa_mlx::models::glm53::kernels::moe_pair_swiglu_addr(
            &staged_address_table,
            &input,
            &staged_selected,
            &route,
            &staged_keepalive,
            inter,
            1,
            config.swiglu_limit,
            metadata_dtype,
            None,
            &stream,
        )
        .ok_or_else(|| anyhow::anyhow!("staged expert pair-address dispatch declined"))?
        .to_vec1::<f32>()?;
        anyhow::ensure!(
            pair_a
                .iter()
                .map(|value| value.to_bits())
                .eq(staged_pair.iter().map(|value| value.to_bits())),
            "expert {expert_id} staged gate/up output differs from mapped dispatch"
        );

        let down_addr = lisa_mlx::models::glm53::kernels::moe_gather_qmv_addr(
            &address_table,
            &pair_addr,
            &selected,
            kernel_keepalive,
            hidden,
            true,
            metadata_dtype,
            None,
            &stream,
        )
        .ok_or_else(|| anyhow::anyhow!("deferred expert down-address dispatch declined"))?;
        let down_weight = store
            .get_owned(&format!(
                "model.layers.3.mlp.experts.{expert_id}.down_proj.weight"
            ))?
            .expand_dims(0)?
            .contiguous()?;
        let down_scale = expert
            .down
            .scales
            .as_dtype(Dtype::Float32)?
            .expand_dims(0)?
            .contiguous()?;
        let down_bias = expert
            .down
            .biases
            .as_dtype(Dtype::Float32)?
            .expand_dims(0)?
            .contiguous()?;
        let down_gathered = lisa_mlx::models::glm53::kernels::moe_gather_qmv(
            &down_weight,
            &down_scale,
            &down_bias,
            &pair_addr,
            &selected,
            hidden,
            true,
            Dtype::Float32,
            &stream,
        )
        .ok_or_else(|| anyhow::anyhow!("deferred expert gathered down dispatch declined"))?;
        let down_a = down_addr.to_vec1::<f32>()?;
        let down_b = down_gathered.to_vec1::<f32>()?;
        anyhow::ensure!(
            down_b.iter().any(|value| value.abs() > 1e-6),
            "real layer-3 expert down path unexpectedly produced all zeros"
        );
        let down_error = down_a
            .iter()
            .zip(&down_b)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        anyhow::ensure!(
            down_error <= 3e-4,
            "real deferred expert {expert_id} down max abs error is {down_error}"
        );
        let staged_down = lisa_mlx::models::glm53::kernels::moe_gather_qmv_addr(
            &staged_address_table,
            &pair_addr,
            &staged_selected,
            &staged_keepalive,
            hidden,
            true,
            metadata_dtype,
            None,
            &stream,
        )
        .ok_or_else(|| anyhow::anyhow!("staged expert down-address dispatch declined"))?
        .to_vec1::<f32>()?;
        anyhow::ensure!(
            down_a
                .iter()
                .map(|value| value.to_bits())
                .eq(staged_down.iter().map(|value| value.to_bits())),
            "expert {expert_id} staged down output differs from mapped dispatch"
        );
    }
    let selected_resources = ExpertPools::indirect_resources(
        &cache_experts,
        &expert_ids.iter().map(|&id| id as i32).collect::<Vec<_>>(),
    )?;
    anyhow::ensure!(
        selected_resources.len() <= expert_ids.len() * 9,
        "five cached layer-3 experts should bind at most nine buffers per expert; found {}",
        selected_resources.len()
    );
    eprintln!(
        "[glm53-moe-individual-views] five selected layer-3 experts bind {} Metal resources",
        selected_resources.len()
    );
    Ok(())
}

#[test]
fn glm53_layer3_moe_matches_mlx_reference_stages() -> anyhow::Result<()> {
    if !lisa_mlx::device::metal_available() {
        eprintln!("[glm53-moe-reference] skipping: no Metal device available");
        return Ok(());
    }
    let Some(model_dir) = std::env::var_os("LISA_GLM53_MODEL") else {
        eprintln!("[glm53-moe-reference] skipping: LISA_GLM53_MODEL is unset");
        return Ok(());
    };
    let Some(reference_dir) = std::env::var_os("LISA_GLM53_MOE_REFERENCE_DIR") else {
        eprintln!("[glm53-moe-reference] skipping: reference fixture is unset");
        return Ok(());
    };
    let model_dir = std::path::Path::new(&model_dir);
    let reference_dir = std::path::Path::new(&reference_dir);
    let config = super::super::config::ModelConfig::from_json(&model_dir.join("config.json"))?;
    let store = Arc::new(LazyTensorStore::open(model_dir)?);
    let mut weights = Weights::new();
    let prefix = "model.layers.3.mlp";
    for name in [
        format!("{prefix}.gate.weight"),
        format!("{prefix}.gate.e_score_correction_bias"),
        format!("{prefix}.shared_experts.gate_proj.weight"),
        format!("{prefix}.shared_experts.gate_proj.scales"),
        format!("{prefix}.shared_experts.gate_proj.biases"),
        format!("{prefix}.shared_experts.up_proj.weight"),
        format!("{prefix}.shared_experts.up_proj.scales"),
        format!("{prefix}.shared_experts.up_proj.biases"),
        format!("{prefix}.shared_experts.down_proj.weight"),
        format!("{prefix}.shared_experts.down_proj.scales"),
        format!("{prefix}.shared_experts.down_proj.biases"),
    ] {
        weights.insert(name.clone(), store.get_owned(&name)?);
    }
    let moe = MoE::load::<Weights>(
        &mut weights,
        prefix,
        config.n_routed_experts,
        config.num_experts_per_tok,
        config.n_group,
        config.topk_group,
        config.routed_scaling_factor,
        config.norm_topk_prob,
        config.n_shared_experts,
        config.moe_intermediate_size,
        config.hidden_size,
        config.swiglu_limit,
        Some(Arc::clone(&store)),
        false,
    )?;

    let metadata: serde_json::Value =
        serde_json::from_slice(&std::fs::read(reference_dir.join("metadata.json"))?)?;
    let tokens = metadata["tokens"].as_u64().unwrap_or_default() as usize;
    anyhow::ensure!(tokens > 0, "MoE reference fixture has no token count");
    anyhow::ensure!(
        metadata["hidden"].as_u64() == Some(config.hidden_size as u64)
            && metadata["intermediate"].as_u64() == Some(config.moe_intermediate_size as u64)
            && metadata["top_k"].as_u64() == Some(config.num_experts_per_tok as u64),
        "MoE reference fixture dimensions do not match the checkpoint"
    );
    let read_f32 = |name: &str| -> anyhow::Result<Vec<f32>> {
        let bytes = std::fs::read(reference_dir.join(name))?;
        anyhow::ensure!(bytes.len() % 4 == 0, "invalid F32 reference file {name}");
        Ok(bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("four-byte chunk")))
            .collect())
    };
    let read_i32 = |name: &str| -> anyhow::Result<Vec<i32>> {
        let bytes = std::fs::read(reference_dir.join(name))?;
        anyhow::ensure!(bytes.len() % 4 == 0, "invalid I32 reference file {name}");
        Ok(bytes
            .chunks_exact(4)
            .map(|chunk| i32::from_le_bytes(chunk.try_into().expect("four-byte chunk")))
            .collect())
    };
    let input = read_f32("moe_input.f32")?;
    anyhow::ensure!(
        input.len() == tokens * config.hidden_size,
        "MoE reference input has the wrong element count"
    );
    let input = Array::from_slice(&input, &[1, tokens as i32, config.hidden_size as i32]);
    let capture = Capture {
        layers: vec![3],
        state_only: false,
        kda_prefill_only: false,
        entries: std::sync::Mutex::new(std::collections::HashMap::new()),
        shapes: std::sync::Mutex::new(std::collections::HashMap::new()),
        flushed: std::sync::atomic::AtomicBool::new(false),
        ..Default::default()
    };
    let output = moe.forward_with_capture(&input, Some(&capture), 3)?;
    output.eval()?;

    let actual = capture.entries.lock().expect("capture entries lock");
    let actual = |name: &str| -> anyhow::Result<Vec<f32>> {
        actual
            .get(name)
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("missing captured stage {name}"))
    };
    let compare = |stage: &str, actual: &[f32], expected: &[f32], tolerance: f32| {
        assert_eq!(actual.len(), expected.len(), "{stage} element count");
        let err = actual
            .iter()
            .zip(expected)
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f32>()
            .sqrt();
        let reference = expected.iter().map(|x| x * x).sum::<f32>().sqrt();
        let relative = err / reference.max(1e-20);
        assert!(
            relative <= tolerance,
            "{stage} relative L2 error {relative:.6} exceeds {tolerance:.6}"
        );
        eprintln!("[glm53-moe-reference] {stage}: rel_l2={relative:.6}");
    };

    let expected_selected = read_i32("moe_selected.i32")?;
    let selected = actual("L3.moe_selected")?;
    let selected: Vec<i32> = selected.into_iter().map(|id| id as i32).collect();
    anyhow::ensure!(
        selected == expected_selected,
        "layer-3 selected expert ids differ from MLX reference"
    );
    compare(
        "route weights",
        &actual("L3.moe_weights")?,
        &read_f32("moe_weights.f32")?,
        0.0002,
    );
    compare(
        "weighted gate/up",
        &actual("L3.moe_pair_weighted")?,
        &read_f32("moe_pair_weighted.f32")?,
        0.0002,
    );
    compare(
        "weighted down slots",
        &actual("L3.moe_down_slots")?,
        &read_f32("moe_down_weighted.f32")?,
        0.0002,
    );
    compare(
        "routed expert sum",
        &actual("L3.moe_routed")?,
        &read_f32("moe_routed.f32")?,
        0.0002,
    );
    compare(
        "shared expert",
        &actual("L3.moe_shared")?,
        &read_f32("moe_shared.f32")?,
        0.00001,
    );
    compare(
        "combined MoE output",
        &output.to_vec1::<f32>()?,
        &read_f32("moe_output.f32")?,
        0.00012,
    );
    Ok(())
}

#[test]
fn glm53_moe_affine_prefill_capture_for_ds4() -> anyhow::Result<()> {
    let Some(capture_dir) = std::env::var_os("LISA_GLM53_AFFINE_CAPTURE") else {
        eprintln!("[glm53-affine-capture] skipping: LISA_GLM53_AFFINE_CAPTURE is unset");
        return Ok(());
    };
    if !lisa_mlx::device::metal_available() {
        eprintln!("[glm53-affine-capture] skipping: no Metal device available");
        return Ok(());
    }
    let model_dir = std::env::var_os("LISA_GLM53_MODEL")
        .ok_or_else(|| anyhow::anyhow!("LISA_GLM53_MODEL is required for affine capture"))?;
    let dir = std::path::Path::new(&model_dir);
    let config = super::super::config::ModelConfig::from_json(&dir.join("config.json"))?;
    let store = LazyTensorStore::open(dir)?;
    let prefix = "model.layers.3.mlp.experts.0.gate_proj";
    let mut projection = QuantizedLinear {
        weight: store.get_mapped(&format!("{prefix}.weight"))?,
        scales: store.get_mapped(&format!("{prefix}.scales"))?,
        biases: store.get_mapped(&format!("{prefix}.biases"))?,
        group_size: 128,
        bits: 2,
    };
    projection.set_quant(128, 2);
    anyhow::ensure!(
        projection.weight.shape()[1] as usize * (32 / projection.bits as usize)
            == config.hidden_size
            && projection.dims_out() == config.moe_intermediate_size
            && projection.group_size == 128
            && projection.bits == 2,
        "selected routed gate tensor does not have the expected MLX Q2 geometry"
    );
    const ROWS: usize = 9;
    let input: Vec<f32> = (0..ROWS)
        .flat_map(|row| {
            (0..config.hidden_size)
                .map(move |i| (((i * 31 + row * 47 + 17) % 251) as f32 - 125.0) * 0.0007)
        })
        .collect();
    let x = Array::from_slice(&input, &[ROWS as i32, config.hidden_size as i32]);
    let output = projection.forward(&x)?.to_vec1::<f32>()?;
    let capture_dir = std::path::Path::new(&capture_dir);
    std::fs::create_dir_all(capture_dir)?;
    std::fs::write(capture_dir.join("input.f32"), encode_f32(&input))?;
    std::fs::write(capture_dir.join("lisa_output.f32"), encode_f32(&output))?;
    eprintln!(
        "[glm53-affine-capture] wrote production Q2 gate prefill: {ROWS} rows, {} -> {}",
        input.len(),
        output.len()
    );
    Ok(())
}

#[test]
fn glm53_attention_q8_affine_prefill_capture_for_ds4() -> anyhow::Result<()> {
    let Some(capture_root) = std::env::var_os("LISA_GLM53_AFFINE_CAPTURE") else {
        eprintln!("[glm53-affine-capture] skipping: LISA_GLM53_AFFINE_CAPTURE is unset");
        return Ok(());
    };
    if !lisa_mlx::device::metal_available() {
        eprintln!("[glm53-affine-capture] skipping: no Metal device available");
        return Ok(());
    }
    let model_dir = std::env::var_os("LISA_GLM53_MODEL")
        .ok_or_else(|| anyhow::anyhow!("LISA_GLM53_MODEL is required for affine capture"))?;
    let dir = std::path::Path::new(&model_dir);
    let config = super::super::config::ModelConfig::from_json(&dir.join("config.json"))?;
    let store = LazyTensorStore::open(dir)?;
    let prefix = "model.layers.3.self_attn.q_a_proj";
    let mut projection = QuantizedLinear {
        weight: store.get_mapped(&format!("{prefix}.weight"))?,
        scales: store.get_mapped(&format!("{prefix}.scales"))?,
        biases: store.get_mapped(&format!("{prefix}.biases"))?,
        group_size: 64,
        bits: 8,
    };
    projection.set_quant(64, 8);
    anyhow::ensure!(
        projection.weight.shape()[1] as usize * (32 / projection.bits as usize)
            == config.hidden_size
            && projection.dims_out() == 1536
            && projection.group_size == 64
            && projection.bits == 8,
        "selected attention q_a tensor does not have the expected MLX Q8 geometry"
    );
    const ROWS: usize = 9;
    let input: Vec<f32> = (0..ROWS)
        .flat_map(|row| {
            (0..config.hidden_size)
                .map(move |i| (((i * 31 + row * 47 + 17) % 251) as f32 - 125.0) * 0.0007)
        })
        .collect();
    let x = Array::from_slice(&input, &[ROWS as i32, config.hidden_size as i32]);
    let output = projection.forward(&x)?.to_vec1::<f32>()?;
    let capture_dir = std::path::Path::new(&capture_root).join("q8_qa");
    std::fs::create_dir_all(&capture_dir)?;
    std::fs::write(capture_dir.join("input.f32"), encode_f32(&input))?;
    std::fs::write(capture_dir.join("lisa_output.f32"), encode_f32(&output))?;
    eprintln!(
        "[glm53-affine-capture] wrote production Q8 q_a prefill: {ROWS} rows, {} -> {}",
        input.len(),
        output.len()
    );
    Ok(())
}

#[test]
fn glm53_layer3_routed_moe_capture_for_ds4() -> anyhow::Result<()> {
    let Some(capture_dir) = std::env::var_os("LISA_GLM53_MOE_DS4_CAPTURE_DIR") else {
        eprintln!("[glm53-moe-ds4-capture] skipping: capture directory is unset");
        return Ok(());
    };
    if !lisa_mlx::device::metal_available() {
        eprintln!("[glm53-moe-ds4-capture] skipping: no Metal device available");
        return Ok(());
    }
    let model_dir = std::env::var_os("LISA_GLM53_MODEL")
        .ok_or_else(|| anyhow::anyhow!("LISA_GLM53_MODEL is required for MoE capture"))?;
    let dir = std::path::Path::new(&model_dir);
    let config = super::super::config::ModelConfig::from_json(&dir.join("config.json"))?;
    anyhow::ensure!(config.hidden_size == 4096 && config.moe_intermediate_size == 2048);
    let store = Arc::new(LazyTensorStore::open(dir)?);
    let prefix = "model.layers.3.mlp";
    let mut weights = Weights::new();
    for name in [
        format!("{prefix}.gate.weight"),
        format!("{prefix}.gate.e_score_correction_bias"),
        format!("{prefix}.shared_experts.gate_proj.weight"),
        format!("{prefix}.shared_experts.gate_proj.scales"),
        format!("{prefix}.shared_experts.gate_proj.biases"),
        format!("{prefix}.shared_experts.up_proj.weight"),
        format!("{prefix}.shared_experts.up_proj.scales"),
        format!("{prefix}.shared_experts.up_proj.biases"),
        format!("{prefix}.shared_experts.down_proj.weight"),
        format!("{prefix}.shared_experts.down_proj.scales"),
        format!("{prefix}.shared_experts.down_proj.biases"),
    ] {
        weights.insert(name.clone(), store.get_owned(&name)?);
    }
    let moe = MoE::load::<Weights>(
        &mut weights,
        prefix,
        config.n_routed_experts,
        config.num_experts_per_tok,
        config.n_group,
        config.topk_group,
        config.routed_scaling_factor,
        config.norm_topk_prob,
        config.n_shared_experts,
        config.moe_intermediate_size,
        config.hidden_size,
        config.swiglu_limit,
        Some(Arc::clone(&store)),
        false,
    )?;
    anyhow::ensure!(moe.top_k == 8 && moe.pools.experts.len() == config.n_routed_experts);
    anyhow::ensure!(
        moe.pools.experts.iter().all(|expert| {
            expert.gate.bits == 2
                && expert.gate.group_size == 128
                && expert.up.bits == 2
                && expert.up.group_size == 128
                && expert.down.bits == 2
                && expert.down.group_size == 128
        }),
        "layer-3 routed expert affine geometry differs from Q2/group-128"
    );

    let rows = std::env::var("LISA_GLM53_MOE_DS4_ROWS")
        .ok()
        .map(|value| value.parse::<usize>())
        .transpose()?
        .unwrap_or(96);
    anyhow::ensure!((1..=96).contains(&rows));
    let input_values: Vec<f32> = if let Some(path) = std::env::var_os("LISA_GLM53_MOE_DS4_INPUT") {
        let bytes = std::fs::read(&path)?;
        anyhow::ensure!(
            bytes.len() == rows * config.hidden_size * std::mem::size_of::<f32>(),
            "captured layer-3 MoE input has an invalid length"
        );
        bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("captured F32")))
            .collect()
    } else {
        let one_input: Vec<f32> = (0..config.hidden_size)
            .map(|i| (((i * 31 + 17) % 251) as f32 - 125.0) * 0.0007)
            .collect();
        (0..rows).flat_map(|_| one_input.iter().copied()).collect()
    };
    let input_shape = if rows <= 8 {
        [rows as i32, 1, config.hidden_size as i32]
    } else {
        [1, rows as i32, config.hidden_size as i32]
    };
    let input = Array::from_slice(&input_values, &input_shape);
    let capture = Capture {
        layers: vec![3],
        state_only: false,
        kda_prefill_only: false,
        entries: std::sync::Mutex::new(std::collections::HashMap::new()),
        shapes: std::sync::Mutex::new(std::collections::HashMap::new()),
        flushed: std::sync::atomic::AtomicBool::new(false),
        ..Default::default()
    };
    let output = moe.forward_with_capture(&input, Some(&capture), 3)?;
    output.eval()?;
    let captured_output = output.to_vec1::<f32>()?;
    let production_output = moe.forward(&input)?;
    production_output.eval()?;
    let production_output_values = production_output.to_vec1::<f32>()?;
    anyhow::ensure!(production_output_values.len() == captured_output.len());
    let mut output_mismatches = 0usize;
    let mut output_max_abs = 0.0f32;
    let mut output_delta_sq = 0.0f64;
    let mut output_expected_sq = 0.0f64;
    for (&actual, &expected) in production_output_values.iter().zip(&captured_output) {
        output_mismatches += usize::from(actual.to_bits() != expected.to_bits());
        let delta = (actual - expected).abs();
        output_max_abs = output_max_abs.max(delta);
        output_delta_sq += f64::from(delta) * f64::from(delta);
        output_expected_sq += f64::from(expected) * f64::from(expected);
    }
    eprintln!(
        "[glm53-moe-ds4-capture] uncaptured fused route vs capture route: mismatches={output_mismatches}/{}, max_abs={output_max_abs:e}, rel_l2={:e}",
        captured_output.len(),
        output_delta_sq.sqrt() / output_expected_sq.sqrt().max(1e-30)
    );
    anyhow::ensure!(
        output_mismatches == 0,
        "captured per-slot and uncaptured fused layer-3 MoE paths differ"
    );

    if rows == 1 {
        let shared = moe
            .shared
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("layer-3 shared expert is missing"))?;
        let projection = match &shared.gate_up {
            GateUpProjection::Paired { projection, .. } => projection,
            GateUpProjection::Separate { .. } => {
                anyhow::bail!("DS4 shared-kernel capture requires paired gate/up weights")
            }
        };
        let flat = input
            .as_dtype(Dtype::Float32)?
            .reshape(&[config.hidden_size as i32])?
            .contiguous()?;
        let inter = config.moe_intermediate_size;
        let (ds4_gate, ds4_up, ds4_mid) = lisa_mlx::models::glm53::kernels::shared_q8_swiglu_ds4(
            &flat,
            &projection.weight,
            &projection.scales,
            &projection.biases,
            config.hidden_size,
            inter,
            projection.group_size as usize,
            config.swiglu_limit,
            &lisa_mlx::Stream::thread_local_or_default(),
        )
        .ok_or_else(|| anyhow::anyhow!("DS4 shared Q8 diagnostic kernel declined"))?;
        let ds4_activation = ds4_mid.clone();
        let (stock_gate, stock_up) = shared.gate_up.forward(&input)?;
        let stock_mid = clamped_swiglu(&stock_up, &stock_gate, config.swiglu_limit)?;
        let stages = [
            ("gate", ds4_gate, stock_gate),
            ("up", ds4_up, stock_up),
            ("SwiGLU", ds4_mid, stock_mid),
        ];
        let capture_path = std::path::Path::new(&capture_dir);
        std::fs::create_dir_all(capture_path)?;
        for (name, candidate, stock) in stages {
            let candidate = candidate.to_vec1::<f32>()?;
            let stock = stock.to_vec1::<f32>()?;
            let (max_abs, delta_sq, expected_sq) = candidate.iter().zip(&stock).fold(
                (0.0f32, 0.0f64, 0.0f64),
                |(max_abs, delta_sq, expected_sq), (&a, &b)| {
                    let delta = f64::from(a - b);
                    (
                        max_abs.max((a - b).abs()),
                        delta_sq + delta * delta,
                        expected_sq + f64::from(b) * f64::from(b),
                    )
                },
            );
            eprintln!(
                "[glm53-shared-ds4-real] {name}: max_abs={max_abs:e} rel_l2={:e}",
                delta_sq.sqrt() / expected_sq.sqrt().max(1e-30)
            );
            let bytes: Vec<u8> = candidate
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect();
            std::fs::write(
                capture_path.join(format!("ds4_schedule_{}.f32", name.to_lowercase())),
                bytes,
            )?;
        }
        let mut activation_shape = input.shape().to_vec();
        *activation_shape
            .last_mut()
            .expect("MoE input has a feature axis") = inter as i32;
        let ds4_activation = ds4_activation.reshape(&activation_shape)?;
        let ds4_down = shared.forward_projection(&shared.down, &ds4_activation)?;
        let ds4_down_values = ds4_down.to_vec1::<f32>()?;
        let routed_values = capture
            .entries
            .lock()
            .expect("capture entries lock")
            .get("L3.moe_routed")
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("missing captured routed branch"))?;
        let ds4_combined: Vec<f32> = routed_values
            .iter()
            .zip(&ds4_down_values)
            .map(|(routed, shared)| routed + shared)
            .collect();
        let encode_f32 = |values: &[f32]| -> Vec<u8> {
            values
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect()
        };
        std::fs::write(
            capture_path.join("ds4_schedule_down.f32"),
            encode_f32(&ds4_down_values),
        )?;
        std::fs::write(
            capture_path.join("ds4_schedule_shared.f32"),
            encode_f32(&ds4_down_values),
        )?;
        std::fs::write(
            capture_path.join("ds4_schedule_combined.f32"),
            encode_f32(&ds4_combined),
        )?;
    }

    // Save the old NAX schedule beside the production output so the
    // captured real-weight case can be replayed against upstream MLX
    // without temporarily editing the production dispatcher.
    let nax_shared = if input.t.device().nax() {
        let shared = moe
            .shared
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("layer-3 shared expert is missing"))?;
        let projection = match &shared.gate_up {
            GateUpProjection::Paired { projection, .. } => projection,
            GateUpProjection::Separate { .. } => {
                anyhow::bail!("NAX capture requires the paired gate/up schedule")
            }
        };
        let x = input
            .as_dtype(Dtype::Float32)?
            .reshape(&[rows as i32, config.hidden_size as i32])?
            .contiguous()?;
        let gate_up = Array::new(lisa_mlx::jit::qmm_nax(
            x.t.device(),
            &x.t,
            &projection.weight.t,
            &projection.scales.t,
            &projection.biases.t,
            projection.group_size,
            projection.bits,
        )?);
        let (gate, up) = split_gate_up(&gate_up, config.moe_intermediate_size)?;
        let activated = clamped_swiglu(&up, &gate, config.swiglu_limit)?;
        let activated = activated
            .reshape(&[rows as i32, config.moe_intermediate_size as i32])?
            .contiguous()?;
        let down = Array::new(lisa_mlx::jit::qmm_nax(
            activated.t.device(),
            &activated.t,
            &shared.down.weight.t,
            &shared.down.scales.t,
            &shared.down.biases.t,
            shared.down.group_size,
            shared.down.bits,
        )?);
        Some((gate, up, activated, down))
    } else {
        None
    };
    let stock_shared = if input.t.device().nax() {
        let shared = moe
            .shared
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("layer-3 shared expert is missing"))?;
        let projection = match &shared.gate_up {
            GateUpProjection::Paired { projection, .. } => projection,
            GateUpProjection::Separate { .. } => {
                anyhow::bail!("stock affine capture requires the paired gate/up schedule")
            }
        };
        let x = input
            .as_dtype(Dtype::Float32)?
            .reshape(&[rows as i32, config.hidden_size as i32])?
            .contiguous()?;
        let gate_up = Array::new(lisa_mlx::jit::affine_qmm(
            x.t.device(),
            &x.t,
            &projection.weight.t,
            &projection.scales.t,
            &projection.biases.t,
            projection.group_size,
            projection.bits,
        )?);
        let (gate, up) = split_gate_up(&gate_up, config.moe_intermediate_size)?;
        let activated = clamped_swiglu(&up, &gate, config.swiglu_limit)?;
        let activated = activated
            .reshape(&[rows as i32, config.moe_intermediate_size as i32])?
            .contiguous()?;
        let down = Array::new(lisa_mlx::jit::affine_qmm(
            activated.t.device(),
            &activated.t,
            &shared.down.weight.t,
            &shared.down.scales.t,
            &shared.down.biases.t,
            shared.down.group_size,
            shared.down.bits,
        )?);
        Some((gate, up, activated, down))
    } else {
        None
    };

    let entries = capture.entries.lock().expect("capture entries lock");
    let values = |name: &str| -> anyhow::Result<&Vec<f32>> {
        entries
            .get(name)
            .ok_or_else(|| anyhow::anyhow!("missing captured MoE value {name}"))
    };
    let selected_f32 = values("L3.moe_selected")?;
    let selected_all: Vec<i32> = selected_f32.iter().map(|&id| id as i32).collect();
    let selected: Vec<i32> = selected_all[..moe.top_k].to_vec();
    anyhow::ensure!(
        selected_all.len() == rows * moe.top_k
            && selected_all.chunks_exact(moe.top_k).all(|row| {
                row.iter()
                    .all(|&id| (0..moe.n_experts as i32).contains(&id))
                    && row
                        .iter()
                        .enumerate()
                        .all(|(slot, id)| !row[..slot].contains(id))
            }),
        "captured GLM layer-3 selection is invalid or contains a per-row duplicate"
    );
    let route_weights_all = values("L3.moe_weights")?;
    let route_weights = &route_weights_all[..moe.top_k];
    let routed = values("L3.moe_routed")?;
    anyhow::ensure!(
        route_weights_all.len() == rows * moe.top_k
            && route_weights_all.iter().all(|weight| weight.is_finite()),
        "captured layer-3 route weights are invalid"
    );
    anyhow::ensure!(routed.len() == rows * config.hidden_size);

    let encode_f32 = |values: &[f32]| -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect()
    };
    let encode_i32 = |values: &[i32]| -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect()
    };
    let capture_dir = std::path::Path::new(&capture_dir);
    std::fs::create_dir_all(capture_dir)?;
    std::fs::write(capture_dir.join("input.f32"), encode_f32(&input_values))?;
    std::fs::write(
        capture_dir.join("router_weight.f32"),
        encode_f32(
            &moe.gate_w
                .reshape(&[(moe.n_experts * config.hidden_size) as i32])?
                .to_vec1::<f32>()?,
        ),
    )?;
    std::fs::write(
        capture_dir.join("router_bias.f32"),
        encode_f32(&moe.bias.to_vec1::<f32>()?),
    )?;
    std::fs::write(
        capture_dir.join("router_logits.f32"),
        encode_f32(values("L3.moe_router_logits")?),
    )?;
    std::fs::write(
        capture_dir.join("selected_all.i32"),
        encode_i32(&selected_all),
    )?;
    std::fs::write(
        capture_dir.join("routed_scaling.f32"),
        moe.routed_scaling.to_le_bytes(),
    )?;
    std::fs::write(capture_dir.join("selected.i32"), encode_i32(&selected))?;
    std::fs::write(
        capture_dir.join("route_weights_all.f32"),
        encode_f32(route_weights_all),
    )?;
    std::fs::write(
        capture_dir.join("route_weights.f32"),
        encode_f32(route_weights),
    )?;
    std::fs::write(capture_dir.join("lisa_routed.f32"), encode_f32(routed))?;
    std::fs::write(
        capture_dir.join("lisa_shared.f32"),
        encode_f32(values("L3.moe_shared")?),
    )?;
    let shared_gate_up = values("L3.dense_gate_up")?;
    let shared_swiglu = values("L3.dense_swiglu")?;
    let shared_down = values("L3.dense_out")?;
    let mut shared_gate = Vec::with_capacity(rows * config.moe_intermediate_size);
    let mut shared_up = Vec::with_capacity(rows * config.moe_intermediate_size);
    for row in shared_gate_up.chunks_exact(2 * config.moe_intermediate_size) {
        shared_gate.extend_from_slice(&row[..config.moe_intermediate_size]);
        shared_up.extend_from_slice(&row[config.moe_intermediate_size..]);
    }
    std::fs::write(
        capture_dir.join("lisa_shared_gate.f32"),
        encode_f32(&shared_gate),
    )?;
    std::fs::write(
        capture_dir.join("lisa_shared_up.f32"),
        encode_f32(&shared_up),
    )?;
    std::fs::write(
        capture_dir.join("lisa_shared_swiglu.f32"),
        encode_f32(shared_swiglu),
    )?;
    std::fs::write(
        capture_dir.join("lisa_shared_down.f32"),
        encode_f32(shared_down),
    )?;
    std::fs::write(
        capture_dir.join("lisa_combined.f32"),
        encode_f32(&captured_output),
    )?;
    std::fs::write(
        capture_dir.join("lisa_combined_uncaptured.f32"),
        encode_f32(&production_output_values),
    )?;
    std::fs::write(
        capture_dir.join("lisa_pair_weighted.f32"),
        encode_f32(values("L3.moe_pair_weighted")?),
    )?;
    std::fs::write(
        capture_dir.join("lisa_down_slots.f32"),
        encode_f32(values("L3.moe_down_slots")?),
    )?;
    std::fs::write(
        capture_dir.join("limit.f32"),
        config.swiglu_limit.to_le_bytes(),
    )?;
    let metadata = serde_json::json!({
        "layer": 3,
        "hidden": config.hidden_size,
        "intermediate": config.moe_intermediate_size,
        "top_k": moe.top_k,
        "tokens": rows,
        "n_experts": moe.n_experts,
        "routed_scaling": moe.routed_scaling,
        "bits": 2,
        "group_size": 128,
        "swiglu_limit": config.swiglu_limit,
        "selected": selected,
    });
    std::fs::write(
        capture_dir.join("metadata.json"),
        serde_json::to_vec_pretty(&metadata)?,
    )?;
    if let Some((nax_gate, nax_up, nax_activation, nax_down)) = nax_shared {
        let nax_dir = capture_dir.join("nax");
        std::fs::create_dir_all(&nax_dir)?;
        std::fs::write(nax_dir.join("input.f32"), encode_f32(&input_values))?;
        std::fs::write(
            nax_dir.join("lisa_shared_gate.f32"),
            encode_f32(&nax_gate.to_vec1::<f32>()?),
        )?;
        std::fs::write(
            nax_dir.join("lisa_shared_up.f32"),
            encode_f32(&nax_up.to_vec1::<f32>()?),
        )?;
        std::fs::write(
            nax_dir.join("lisa_shared_swiglu.f32"),
            encode_f32(&nax_activation.to_vec1::<f32>()?),
        )?;
        std::fs::write(
            nax_dir.join("lisa_shared_down.f32"),
            encode_f32(&nax_down.to_vec1::<f32>()?),
        )?;
        std::fs::write(
            nax_dir.join("metadata.json"),
            serde_json::to_vec_pretty(&metadata)?,
        )?;
    }
    if let Some((stock_gate, stock_up, stock_activation, stock_down)) = stock_shared {
        let stock_dir = capture_dir.join("stock");
        std::fs::create_dir_all(&stock_dir)?;
        std::fs::write(stock_dir.join("input.f32"), encode_f32(&input_values))?;
        std::fs::write(
            stock_dir.join("lisa_shared_gate.f32"),
            encode_f32(&stock_gate.to_vec1::<f32>()?),
        )?;
        std::fs::write(
            stock_dir.join("lisa_shared_up.f32"),
            encode_f32(&stock_up.to_vec1::<f32>()?),
        )?;
        std::fs::write(
            stock_dir.join("lisa_shared_swiglu.f32"),
            encode_f32(&stock_activation.to_vec1::<f32>()?),
        )?;
        std::fs::write(
            stock_dir.join("lisa_shared_down.f32"),
            encode_f32(&stock_down.to_vec1::<f32>()?),
        )?;
        std::fs::write(
            stock_dir.join("metadata.json"),
            serde_json::to_vec_pretty(&metadata)?,
        )?;
    }
    eprintln!(
        "[glm53-moe-ds4-capture] captured {rows} layer-3 MoE rows; first row experts {selected:?}, unique experts {}",
        selected_all
            .iter()
            .collect::<std::collections::HashSet<_>>()
            .len()
    );
    lisa_mlx::trace::finish();
    Ok(())
}

#[test]
fn glm53_layer3_router_rows_capture_for_ds4() -> anyhow::Result<()> {
    let Some(capture_root) = std::env::var_os("LISA_GLM53_ROUTER_DS4_CAPTURE_DIR") else {
        eprintln!("[glm53-router-ds4-capture] skipping: capture directory is unset");
        return Ok(());
    };
    if !lisa_mlx::device::metal_available() {
        eprintln!("[glm53-router-ds4-capture] skipping: no Metal device available");
        return Ok(());
    }
    let model_dir = std::env::var_os("LISA_GLM53_MODEL")
        .ok_or_else(|| anyhow::anyhow!("LISA_GLM53_MODEL is required for router capture"))?;
    let dir = std::path::Path::new(&model_dir);
    let config = super::super::config::ModelConfig::from_json(&dir.join("config.json"))?;
    anyhow::ensure!(config.hidden_size == 4096 && config.n_routed_experts == 288);
    let store = LazyTensorStore::open(dir)?;
    let gate_w = store
        .get_owned("model.layers.3.mlp.gate.weight")?
        .as_dtype(Dtype::Float32)?;
    let bias = store
        .get_owned("model.layers.3.mlp.gate.e_score_correction_bias")?
        .as_dtype(Dtype::Float32)?;
    let router_weight_values = gate_w
        .reshape(&[(config.n_routed_experts * config.hidden_size) as i32])?
        .to_vec1::<f32>()?;
    let bias_values = bias.to_vec1::<f32>()?;
    anyhow::ensure!(
        router_weight_values.len() == config.n_routed_experts * config.hidden_size
            && bias_values.len() == config.n_routed_experts,
        "layer-3 router weight geometry differs from the pinned GLM checkpoint"
    );
    let root = std::path::Path::new(&capture_root);
    std::fs::create_dir_all(root)?;
    let encode_f32 = |values: &[f32]| -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect()
    };
    let encode_i32 = |values: &[i32]| -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect()
    };
    for rows in 1..=9usize {
        let input_values: Vec<f32> = if rows == 2 {
            if let Some(path) = std::env::var_os("LISA_GLM53_ROUTER_ROWS2_INPUT") {
                let bytes = std::fs::read(path)?;
                anyhow::ensure!(
                    bytes.len() == rows * config.hidden_size * std::mem::size_of::<f32>(),
                    "captured layer-3 two-row router input has an invalid length"
                );
                bytes
                    .chunks_exact(4)
                    .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("captured F32")))
                    .collect()
            } else {
                (0..rows * config.hidden_size)
                    .map(|i| (((i * 31 + 17) % 251) as f32 - 125.0) * 0.0007)
                    .collect()
            }
        } else {
            (0..rows * config.hidden_size)
                .map(|i| (((i * 31 + 17) % 251) as f32 - 125.0) * 0.0007)
                .collect()
        };
        let input = Array::from_slice(&input_values, &[rows as i32, config.hidden_size as i32]);
        let stream = lisa_mlx::Stream::thread_local_or_default();
        let logits = lisa_mlx::models::glm53::kernels::router_matvec_f32(&gate_w, &input, &stream)
            .ok_or_else(|| anyhow::anyhow!("layer-3 router projection declined at {rows} rows"))?;
        let (selected, route_weights) = lisa_mlx::models::glm53::kernels::router_select_fused(
            &logits,
            &bias,
            config.num_experts_per_tok,
            config.routed_scaling_factor,
            &stream,
        )
        .ok_or_else(|| anyhow::anyhow!("layer-3 router selection declined at {rows} rows"))?;
        let capture = root.join(format!("rows-{rows}"));
        std::fs::create_dir_all(&capture)?;
        std::fs::write(capture.join("input.f32"), encode_f32(&input_values))?;
        std::fs::write(
            capture.join("router_logits.f32"),
            encode_f32(&logits.to_vec1::<f32>()?),
        )?;
        std::fs::write(
            capture.join("selected.i32"),
            encode_i32(&selected.to_vec1::<i32>()?),
        )?;
        std::fs::write(
            capture.join("route_weights.f32"),
            encode_f32(&route_weights.to_vec1::<f32>()?),
        )?;
        std::fs::write(
            capture.join("router_weight.f32"),
            encode_f32(&router_weight_values),
        )?;
        std::fs::write(capture.join("router_bias.f32"), encode_f32(&bias_values))?;
        std::fs::write(
            capture.join("routed_scaling.f32"),
            config.routed_scaling_factor.to_le_bytes(),
        )?;
    }
    eprintln!("[glm53-router-ds4-capture] captured real layer-3 router rows 1 through 9");
    Ok(())
}
