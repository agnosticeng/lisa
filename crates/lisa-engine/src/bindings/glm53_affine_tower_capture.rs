use super::*;
pub fn capture_mtp_verify_layer_state<D: ExpertDevice + ReadbackDevice>(
    context: &DeviceContext<D>,
    prefix: &std::ffi::OsStr,
    capture_index: usize,
    layer: usize,
    state: &D::Tensor,
) -> anyhow::Result<()> {
    let values = state
        .cast_on(context, lisa_backend::ScalarType::F32)?
        .contiguous()?
        .reshape(&[state.size() as i32])?
        .read_f32()?;
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

pub fn capture_single_step_layer_state<D: ExpertDevice + ReadbackDevice>(
    context: &DeviceContext<D>,
    prefix: &std::ffi::OsStr,
    position: usize,
    layer: usize,
    stage: Option<&str>,
    state: &D::Tensor,
) -> anyhow::Result<()> {
    let values = state
        .cast_on(context, lisa_backend::ScalarType::F32)?
        .contiguous()?
        .reshape(&[state.size() as i32])?
        .read_f32()?;
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
#[cfg(feature = "glm53-fixtures")]
pub fn capture_or_assert_mtp_residual_ds4<D: ExpertDevice + ReadbackDevice>(
    _context: &DeviceContext<D>,
    x: &D::Tensor,
    attended: &D::Tensor,
    weight: &D::Tensor,
    sum: &D::Tensor,
    norm: &D::Tensor,
    eps: f32,
) -> anyhow::Result<()> {
    const WIDTH: usize = 4096;
    const PART_BYTES: usize = WIDTH * std::mem::size_of::<f32>();
    let external_fixture: &[u8] =
        &crate::bindings::glm53_test_fixtures::read("mtp_residual_layer45_ds4.bin");
    let values = [
        x.read_f32()?,
        attended.read_f32()?,
        weight.read_f32()?,
        sum.read_f32()?,
        norm.read_f32()?,
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
        external_fixture.len() == payload.len(),
        "DS4 MTP residual fixture has an invalid size"
    );
    for (index, (actual, expected)) in payload
        .chunks_exact(4)
        .zip(external_fixture.chunks_exact(4))
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

#[cfg(feature = "glm53-fixtures")]
pub fn capture_or_assert_mtp_tail_ds4<D: ExpertDevice + ReadbackDevice>(
    _context: &DeviceContext<D>,
    residual: &D::Tensor,
    routed: &D::Tensor,
    shared: &D::Tensor,
    head_norm_weight: &D::Tensor,
    sum: &D::Tensor,
    normalized: &D::Tensor,
    eps: f32,
) -> anyhow::Result<()> {
    const WIDTH: usize = 4096;
    const PART_BYTES: usize = WIDTH * std::mem::size_of::<f32>();
    let external_fixture: &[u8] =
        &crate::bindings::glm53_test_fixtures::read("mtp_tail_layer45_ds4.bin");
    let values = [
        residual.read_f32()?,
        routed.read_f32()?,
        shared.read_f32()?,
        head_norm_weight.read_f32()?,
        sum.read_f32()?,
        normalized.read_f32()?,
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
        external_fixture.len() == payload.len(),
        "DS4 MTP tail fixture has an invalid size"
    );
    for (index, (actual, expected)) in payload
        .chunks_exact(4)
        .zip(external_fixture.chunks_exact(4))
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

#[cfg(feature = "glm53-fixtures")]
pub fn capture_or_assert_mtp_lm_head_ds4<D: ExpertDevice + ReadbackDevice>(
    context: &DeviceContext<D>,
    input: &D::Tensor,
    lm_head: &QuantizedLinear<D::Tensor>,
    actual_logits: &D::Tensor,
    position: usize,
) -> anyhow::Result<()> {
    const HIDDEN: usize = 4096;
    let fixture: &[u8] =
        if crate::bindings::glm53_attention::mtp_real_target_fixtures_enabled() && position == 6 {
            &crate::bindings::glm53_test_fixtures::read("mtp_composed_natural_target_step6_ds4.bin")
        } else if crate::bindings::glm53_attention::mtp_real_target_fixtures_enabled() {
            &crate::bindings::glm53_test_fixtures::read("mtp_composed_natural_target_ds4.bin")
        } else {
            &crate::bindings::glm53_test_fixtures::read("mtp_lm_head_layer45_ds4.bin")
        };
    let vocab = lm_head.dims_out();
    anyhow::ensure!(
        input.size() == HIDDEN && lm_head.group_size == 64 && lm_head.bits == 8,
        "layer-45 MTP output-head fixture geometry differs from DS4"
    );
    let input_values = input
        .cast_on(context, lisa_backend::ScalarType::F32)?
        .read_f32()?;
    let actual_values = actual_logits
        .cast_on(context, lisa_backend::ScalarType::F32)?
        .read_f32()?;
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
            encode_u32(&lm_head.weight.read_raw_ids()?),
        )?;
        std::fs::write(
            path.join("scales.f32"),
            encode_f32(&promote(context, &lm_head.scales)?.read_f32()?),
        )?;
        std::fs::write(
            path.join("biases.f32"),
            encode_f32(&promote(context, &lm_head.biases)?.read_f32()?),
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
    if crate::bindings::glm53_attention::mtp_real_target_fixtures_enabled() {
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
    let (max_abs_limit, rel_l2_limit) =
        if crate::bindings::glm53_attention::mtp_real_target_fixtures_enabled() {
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
