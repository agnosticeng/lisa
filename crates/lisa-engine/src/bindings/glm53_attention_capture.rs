//! Explicit capture I/O and external-reference diagnostics.
use super::*;

std::thread_local! {
    static MTP_REAL_TARGET_FIXTURES: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

pub struct MtpRealTargetFixtureScope(bool);

impl Drop for MtpRealTargetFixtureScope {
    fn drop(&mut self) {
        MTP_REAL_TARGET_FIXTURES.with(|enabled| enabled.set(self.0));
    }
}

pub fn use_mtp_real_target_fixtures() -> MtpRealTargetFixtureScope {
    let previous = MTP_REAL_TARGET_FIXTURES.with(|enabled| enabled.replace(true));
    MtpRealTargetFixtureScope(previous)
}

pub fn mtp_real_target_fixtures_enabled() -> bool {
    MTP_REAL_TARGET_FIXTURES.with(std::cell::Cell::get)
}

pub fn capture_mtp_dsa_inputs<D: BindingDevice + ManagedDevice>(
    context: &DeviceContext<D>,
    path: &std::path::Path,
    dsa: &Dsa<D>,
    x: &D::Tensor,
    kv_raw: &D::Tensor,
    kv_norm: &D::Tensor,
    q_a_raw: &D::Tensor,
    q_a_normed: &D::Tensor,
    q: &D::Tensor,
    q_low: &D::Tensor,
    lora: &D::Tensor,
    attended: &D::Tensor,
    compact: &D::Tensor,
    selected: &D::Tensor,
) -> anyhow::Result<()> {
    std::fs::create_dir_all(path)?;
    let write_u32 = |name: &str, values: &[u32]| -> anyhow::Result<()> {
        let bytes: Vec<u8> = values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        std::fs::write(path.join(name), bytes)?;
        Ok(())
    };
    let write_i32 = |name: &str, values: &[i32]| -> anyhow::Result<()> {
        let bytes: Vec<u8> = values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        std::fs::write(path.join(name), bytes)?;
        Ok(())
    };
    let write_f32 = |name: &str, values: &[f32]| -> anyhow::Result<()> {
        let bytes: Vec<u8> = values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        std::fs::write(path.join(name), bytes)?;
        Ok(())
    };
    let write_f16 = |name: &str, values: &[u16]| -> anyhow::Result<()> {
        let bytes: Vec<u8> = values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        std::fs::write(path.join(name), bytes)?;
        Ok(())
    };
    write_u32("kv_a_weight.u32", &dsa.kv_a.weight.read_raw_ids()?)?;
    write_f32("kv_a_scales.f32", &dsa.kv_a.scales.read_f32()?)?;
    write_f32("kv_a_biases.f32", &dsa.kv_a.biases.read_f32()?)?;
    write_f32("kv_a_norm.f32", &dsa.kv_a_norm.read_f32()?)?;
    write_f32("kv_a_input.f32", &x.read_f32()?)?;
    write_f32("kv_a_eps.f32", &[dsa.norm_eps])?;
    write_f32("lisa_kv_a_raw.f32", &kv_raw.read_f32()?)?;
    write_f32("lisa_kv_a_norm.f32", &kv_norm.read_f32()?)?;
    write_f16("lisa_mtp_compact_cache.f16", &compact.read_u16(context)?)?;
    write_u32("q_a_weight.u32", &dsa.q_a.weight.read_raw_ids()?)?;
    write_f32("q_a_scales.f32", &dsa.q_a.scales.read_f32()?)?;
    write_f32("q_a_biases.f32", &dsa.q_a.biases.read_f32()?)?;
    write_f32("q_a_norm.f32", &dsa.q_a_norm.read_f32()?)?;
    write_f32("lisa_q_a_raw.f32", &q_a_raw.read_f32()?)?;
    write_f32("lisa_q_a_norm.f32", &q_a_normed.read_f32()?)?;
    write_u32("q_b_weight.u32", &dsa.q_b.weight.read_raw_ids()?)?;
    write_f32("q_b_scales.f32", &dsa.q_b.scales.read_f32()?)?;
    write_f32("q_b_biases.f32", &dsa.q_b.biases.read_f32()?)?;
    write_u32("weight.u32", &dsa.kv_b.weight.read_raw_ids()?)?;
    write_f32("scales.f32", &dsa.kv_b.scales.read_f32()?)?;
    write_f32("biases.f32", &dsa.kv_b.biases.read_f32()?)?;
    write_f32("q.f32", &q.read_f32()?)?;
    write_f32("lisa_q_low.f32", &q_low.read_f32()?)?;
    write_f32("lisa_lora.f32", &lora.read_f32()?)?;
    write_f32("lisa_pre_o.f32", &attended.read_f32()?)?;
    write_f16("compact_cache.f16", &compact.read_u16(context)?)?;
    write_i32("selected.u32", &selected.read_i32(context)?)?;
    Ok(())
}

const DS4_DSA_VERIFY_LAYERS: [usize; 11] = [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43];

pub fn ds4_dsa_verify_fixture_layer_index(layer: usize) -> Option<usize> {
    DS4_DSA_VERIFY_LAYERS
        .iter()
        .position(|&candidate| candidate == layer)
}

pub fn ds4_dsa_verify_stages_fixture(layer: usize) -> Option<Vec<u8>> {
    const PER_LAYER: usize = 655_360;
    let external_fixture: &[u8] = &crate::bindings::glm53_test_fixtures::read(
        "dsa_verify_target_mtp_all_layers_stages_ds4.bin",
    );
    let index = ds4_dsa_verify_fixture_layer_index(layer)?;
    Some(external_fixture[index * PER_LAYER..(index + 1) * PER_LAYER].to_vec())
}

pub fn ds4_dsa_verify_kv_a_fixture(layer: usize) -> Option<Vec<u8>> {
    const PER_LAYER: usize = 10_240;
    let external_fixture: &[u8] = &crate::bindings::glm53_test_fixtures::read(
        "dsa_verify_target_mtp_all_layers_kv_a_ds4.bin",
    );
    let index = ds4_dsa_verify_fixture_layer_index(layer)?;
    Some(external_fixture[index * PER_LAYER..(index + 1) * PER_LAYER].to_vec())
}

pub fn capture_dsa_verify_attention_inputs<D: BindingDevice + ManagedDevice>(
    context: &DeviceContext<D>,
    path: &std::path::Path,
    dsa: &Dsa<D>,
    layer: usize,
    q: &D::Tensor,
    compact: &D::Tensor,
    lisa_pre_o: &D::Tensor,
    pos0: usize,
) -> anyhow::Result<()> {
    std::fs::create_dir_all(path)?;
    let write_u32 = |name: &str, values: &[u32]| -> anyhow::Result<()> {
        let bytes: Vec<u8> = values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        std::fs::write(path.join(name), bytes)?;
        Ok(())
    };
    let write_f32 = |name: &str, values: &[f32]| -> anyhow::Result<()> {
        let bytes: Vec<u8> = values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        std::fs::write(path.join(name), bytes)?;
        Ok(())
    };
    let write_f16 = |name: &str, values: &[u16]| -> anyhow::Result<()> {
        let bytes: Vec<u8> = values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        std::fs::write(path.join(name), bytes)?;
        Ok(())
    };
    anyhow::ensure!(
        q.dim(0) == 1 && compact.dim(0) == 1 && lisa_pre_o.dim(0) == 1,
        "DS4 dense verify capture currently requires batch size one"
    );
    let (tokens, cache_tokens) = (q.dim(2) as usize, compact.dim(1) as usize);
    anyhow::ensure!(
        tokens > 0 && cache_tokens >= tokens && cache_tokens - tokens == pos0,
        "invalid DS4 dense verify capture positions: pos0={pos0}, queries={tokens}, cache={cache_tokens}"
    );
    let q_rows = q.transpose_on(context, &[0, 2, 1, 3])?.contiguous()?;
    write_u32("kv_b_weight.u32", &dsa.kv_b.weight.read_raw_ids()?)?;
    write_f32("kv_b_scales.f32", &dsa.kv_b.scales.read_f32()?)?;
    write_f32("kv_b_biases.f32", &dsa.kv_b.biases.read_f32()?)?;
    write_f32("q.f32", &q_rows.read_f32()?)?;
    write_f16("compact_cache.f16", &compact.read_u16(context)?)?;
    write_f32("lisa_pre_o.f32", &lisa_pre_o.read_f32()?)?;
    let actual = lisa_pre_o.read_f32()?;
    if tokens == 2
        && pos0 == 6
        && dsa.heads == 64
        && dsa.v_head_dim == 256
        && std::env::var_os("LISA_GLM53_DSA_KVA_VERIFY_CAPTURE_DIR").is_some()
    {
        let ds4_fixture = ds4_dsa_verify_stages_fixture(layer).ok_or_else(|| {
            anyhow::anyhow!("no direct-DS4 target-MTP DSA fixture for layer {layer}")
        })?;
        let output_start = 2 * tokens * dsa.heads * 512 * std::mem::size_of::<f32>();
        let output_fixture = &ds4_fixture[output_start..];
        let expected = output_fixture
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("F32 fixture word")));
        anyhow::ensure!(
            output_fixture.len() == actual.len() * std::mem::size_of::<f32>()
                && actual
                    .iter()
                    .map(|x| x.to_bits())
                    .eq(expected.map(|x| x.to_bits())),
            "real layer-{layer} target-MTP DSA output differs from the direct DS4 fixture"
        );
    }
    let meta = [
        tokens as u32,
        cache_tokens as u32,
        pos0 as u32,
        dsa.heads as u32,
        dsa.q_head_dim as u32,
        dsa.v_head_dim as u32,
        dsa.kv_b.group_size as u32,
        dsa.kv_b.bits as u32,
    ];
    write_u32("meta.u32", &meta)?;
    Ok(())
}

pub fn capture_dsa_native_decode_attention_inputs<D: BindingDevice + ManagedDevice>(
    context: &DeviceContext<D>,
    path: &std::path::Path,
    dsa: &Dsa<D>,
    q: &D::Tensor,
    compact: &D::Tensor,
    selected: &D::Tensor,
    q_low: &D::Tensor,
    lora: &D::Tensor,
    lisa_fused_lora: &D::Tensor,
    lisa_pre_o: &D::Tensor,
    lisa_fused_pre_o: &D::Tensor,
) -> anyhow::Result<()> {
    std::fs::create_dir_all(path)?;
    let write_u32 = |name: &str, values: &[u32]| -> anyhow::Result<()> {
        let bytes: Vec<u8> = values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        std::fs::write(path.join(name), bytes)?;
        Ok(())
    };
    let write_i32 = |name: &str, values: &[i32]| -> anyhow::Result<()> {
        let bytes: Vec<u8> = values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        std::fs::write(path.join(name), bytes)?;
        Ok(())
    };
    let write_f32 = |name: &str, values: &[f32]| -> anyhow::Result<()> {
        let bytes: Vec<u8> = values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        std::fs::write(path.join(name), bytes)?;
        Ok(())
    };
    let write_f16 = |name: &str, values: &[u16]| -> anyhow::Result<()> {
        let bytes: Vec<u8> = values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        std::fs::write(path.join(name), bytes)?;
        Ok(())
    };
    anyhow::ensure!(
        q.shape() == [1, dsa.heads as i32, 1, dsa.q_head_dim as i32]
            && compact.dim(0) == 1
            && compact.dim(2) == dsa.kv_a_norm.dim(0)
            && selected.rank() == 3
            && selected.dim(0) == 1
            && selected.dim(1) == 1
            && q_low.shape() == [1, 1, dsa.heads as i32, dsa.kv_a_norm.dim(0)]
            && lora.shape() == q_low.shape()
            && lisa_fused_lora.shape() == q_low.shape()
            && lisa_pre_o.shape() == [1, 1, (dsa.heads * dsa.v_head_dim) as i32]
            && lisa_fused_pre_o.shape() == lisa_pre_o.shape(),
        "native DSA attention capture has unexpected geometry"
    );
    let q_rows = q.transpose_on(context, &[0, 2, 1, 3])?.contiguous()?;
    let meta = [
        compact.dim(1) as u32,
        selected.dim(2) as u32,
        dsa.heads as u32,
        dsa.q_head_dim as u32,
        dsa.v_head_dim as u32,
        dsa.kv_b.group_size as u32,
        dsa.kv_b.bits as u32,
    ];
    write_u32("meta.u32", &meta)?;
    write_u32("kv_b_weight.u32", &dsa.kv_b.weight.read_raw_ids()?)?;
    write_f32("kv_b_scales.f32", &dsa.kv_b.scales.read_f32()?)?;
    write_f32("kv_b_biases.f32", &dsa.kv_b.biases.read_f32()?)?;
    write_f32("q.f32", &q_rows.read_f32()?)?;
    write_f16("compact_cache.f16", &compact.read_u16(context)?)?;
    write_i32("selected.i32", &selected.read_i32(context)?)?;
    write_f32("lisa_q_low.f32", &q_low.read_f32()?)?;
    write_f32("lisa_lora.f32", &lora.read_f32()?)?;
    write_f32("lisa_fused_lora.f32", &lisa_fused_lora.read_f32()?)?;
    write_f32("lisa_pre_o.f32", &lisa_pre_o.read_f32()?)?;
    write_f32("lisa_fused_pre_o.f32", &lisa_fused_pre_o.read_f32()?)?;
    Ok(())
}

pub fn capture_dsa_native_batch_o_project_inputs<D: BindingDevice + ManagedDevice>(
    _context: &DeviceContext<D>,
    path: &std::path::Path,
    dsa: &Dsa<D>,
    input: &D::Tensor,
    lisa_output: &D::Tensor,
) -> anyhow::Result<()> {
    std::fs::create_dir_all(path)?;
    let write_u32 = |name: &str, values: &[u32]| -> anyhow::Result<()> {
        let bytes: Vec<u8> = values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        std::fs::write(path.join(name), bytes)?;
        Ok(())
    };
    let write_f32 = |name: &str, values: &[f32]| -> anyhow::Result<()> {
        let bytes: Vec<u8> = values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        std::fs::write(path.join(name), bytes)?;
        Ok(())
    };
    anyhow::ensure!(
        input.shape() == [2, 1, 16_384] && lisa_output.shape() == [2, 1, 4096],
        "native B=2 DSA output-projection capture has unexpected geometry"
    );
    write_u32("o_proj_weight.u32", &dsa.o_proj.weight.read_raw_ids()?)?;
    write_f32("o_proj_scales.f32", &dsa.o_proj.scales.read_f32()?)?;
    write_f32("o_proj_biases.f32", &dsa.o_proj.biases.read_f32()?)?;
    write_f32("input.f32", &input.reshape(&[2, 16_384])?.read_f32()?)?;
    write_f32(
        "lisa_output.f32",
        &lisa_output.reshape(&[2, 4096])?.read_f32()?,
    )?;
    Ok(())
}

pub fn capture_dsa_verify_kv_a_inputs<D: BindingDevice + ManagedDevice>(
    context: &DeviceContext<D>,
    path: &std::path::Path,
    dsa: &Dsa<D>,
    layer: usize,
    pos0: usize,
    x: &D::Tensor,
    kv_raw: &D::Tensor,
    kv_norm: Option<&D::Tensor>,
) -> anyhow::Result<()> {
    std::fs::create_dir_all(path)?;
    anyhow::ensure!(
        (x.dim(0) == 1 || (x.dim(0) == 2 && x.dim(1) == 1))
            && x.dim(1) > 0
            && x.dim(2) == 4096
            && kv_raw.dim(2) == 512
            && kv_norm.is_none_or(|norm| norm.dim(2) == 512),
        "DS4 target verify kv_a capture has unexpected geometry"
    );
    let rows = (x.dim(0) * x.dim(1)) as usize;
    let input_rows = x.reshape(&[rows as i32, 4096])?;
    let raw_rows = kv_raw.reshape(&[rows as i32, 512])?;
    let norm_rows = kv_norm
        .map(|norm| norm.reshape(&[rows as i32, 512]))
        .transpose()?;
    let write_u32 = |name: &str, values: &[u32]| -> anyhow::Result<()> {
        let bytes: Vec<u8> = values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        std::fs::write(path.join(name), bytes)?;
        Ok(())
    };
    let write_f32 = |name: &str, values: &[f32]| -> anyhow::Result<()> {
        let bytes: Vec<u8> = values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        std::fs::write(path.join(name), bytes)?;
        Ok(())
    };
    let write_u16 = |name: &str, values: &[u16]| -> anyhow::Result<()> {
        let bytes: Vec<u8> = values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        std::fs::write(path.join(name), bytes)?;
        Ok(())
    };
    write_u32("kv_a_weight.u32", &dsa.kv_a.weight.read_raw_ids()?)?;
    write_f32("kv_a_scales.f32", &dsa.kv_a.scales.read_f32()?)?;
    write_f32("kv_a_biases.f32", &dsa.kv_a.biases.read_f32()?)?;
    write_f32("kv_a_norm.f32", &dsa.kv_a_norm.read_f32()?)?;
    write_f32("kv_a_eps.f32", &[dsa.norm_eps])?;
    for row in 0..rows {
        write_f32(
            &format!("kv_a_input_{row}.f32"),
            &narrow(context, &input_rows, 0, row, 1)?
                .reshape(&[4096])?
                .read_f32()?,
        )?;
        write_f32(
            &format!("lisa_kv_a_raw_{row}.f32"),
            &narrow(context, &raw_rows, 0, row, 1)?
                .reshape(&[512])?
                .read_f32()?,
        )?;
        if let Some(norm_rows) = &norm_rows {
            write_f32(
                &format!("lisa_kv_a_norm_{row}.f32"),
                &narrow(context, norm_rows, 0, row, 1)?
                    .reshape(&[512])?
                    .read_f32()?,
            )?;
        }
    }
    if rows == 2
        && x.dim(0) == 1
        && pos0 == 6
        && std::env::var_os("LISA_GLM53_DSA_KVA_VERIFY_CAPTURE_DIR").is_some()
    {
        let ds4_fixture = ds4_dsa_verify_kv_a_fixture(layer).ok_or_else(|| {
            anyhow::anyhow!("no direct-DS4 target-MTP kv_a fixture for layer {layer}")
        })?;
        let values = raw_rows.read_f32()?;
        let raw_bytes = rows * 512 * std::mem::size_of::<f32>();
        anyhow::ensure!(
            ds4_fixture.len() == raw_bytes * 2 + rows * 512 * std::mem::size_of::<u16>(),
            "real layer-{layer} target-MTP kv_a fixture has an unexpected size"
        );
        let matches_f32 = |values: &[f32], bytes: &[u8]| {
            values
                .iter()
                .map(|x| x.to_bits())
                .eq(bytes.chunks_exact(4).map(|chunk| {
                    f32::from_le_bytes(chunk.try_into().expect("F32 fixture word")).to_bits()
                }))
        };
        anyhow::ensure!(
            matches_f32(&values, &ds4_fixture[..raw_bytes]),
            "real layer-{layer} target-MTP kv_a projection differs from DS4"
        );
        if kv_norm.is_some() {
            let norm_values = norm_rows
                .as_ref()
                .expect("normalized capture rows")
                .read_f32()?;
            anyhow::ensure!(
                matches_f32(&norm_values, &ds4_fixture[raw_bytes..2 * raw_bytes]),
                "real layer-{layer} target-MTP KV-LoRA norm differs from DS4"
            );
        }
    }
    write_u32(
        "meta.u32",
        &[
            rows as u32,
            4096,
            512,
            dsa.kv_a.group_size as u32,
            dsa.kv_a.bits as u32,
        ],
    )?;
    if x.dim(0) == 2 && x.dim(1) == 1 {
        let Some(norm_rows) = norm_rows else {
            anyhow::bail!("native batch DSA capture requires normalized KV rows");
        };
        let compact = norm_rows
            .reshape(&[(rows * 512) as i32])?
            .cast_on(context, lisa_backend::ScalarType::F16)?
            .read_u16(context)?;
        write_u16("lisa_compact_cache.f16", &compact)?;
    }
    Ok(())
}

pub fn capture_dsa_verify_compact_cache<D: BindingDevice + ManagedDevice>(
    context: &DeviceContext<D>,
    path: &std::path::Path,
    layer: usize,
    pos0: usize,
    compact: &D::Tensor,
) -> anyhow::Result<()> {
    let values = compact.read_u16(context)?;
    if values.len() == 2 * 512
        && pos0 == 6
        && std::env::var_os("LISA_GLM53_DSA_KVA_VERIFY_CAPTURE_DIR").is_some()
    {
        let ds4_fixture = ds4_dsa_verify_kv_a_fixture(layer)
            .expect("direct-DS4 target-MTP compact-cache fixture for captured layer");
        let cache_start = 2 * 512 * 2 * std::mem::size_of::<f32>();
        let expected = ds4_fixture[cache_start..]
            .chunks_exact(2)
            .map(|chunk| u16::from_le_bytes(chunk.try_into().expect("F16 fixture word")));
        anyhow::ensure!(
            values.iter().copied().eq(expected),
            "real layer-{layer} target-MTP compact-cache append differs from DS4"
        );
    }
    let bytes: Vec<u8> = values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();
    std::fs::write(path.join("lisa_mtp_compact_cache.f16"), bytes)?;
    Ok(())
}

pub fn capture_dsa_verify_latent_stages<D: BindingDevice + ManagedDevice>(
    _context: &DeviceContext<D>,
    path: &std::path::Path,
    q_low: &D::Tensor,
    lora: &D::Tensor,
) -> anyhow::Result<()> {
    std::fs::create_dir_all(path)?;
    for (name, stage) in [("lisa_q_low.f32", q_low), ("lisa_lora.f32", lora)] {
        let bytes: Vec<u8> = stage
            .read_f32()?
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        std::fs::write(path.join(name), bytes)?;
    }
    Ok(())
}

pub fn capture_mtp_dsa_kv_a_input_row<D: BindingDevice + ManagedDevice>(
    _context: &DeviceContext<D>,
    path: &std::path::Path,
    x: &D::Tensor,
    kv_raw: &D::Tensor,
    kv_norm: &D::Tensor,
    min_position: usize,
    position: usize,
) -> anyhow::Result<()> {
    std::fs::create_dir_all(path)?;
    anyhow::ensure!(
        position >= min_position,
        "MTP DSA capture position precedes its cache window"
    );
    let values = x.read_f32()?;
    let row = position - min_position;
    let bytes: Vec<u8> = values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();
    std::fs::write(path.join(format!("kv_a_input_{row}.f32")), bytes)?;
    for (name, stage) in [("kv_a_raw", kv_raw), ("kv_a_norm", kv_norm)] {
        let values = stage.read_f32()?;
        let bytes: Vec<u8> = values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        std::fs::write(path.join(format!("lisa_{name}_{row}.f32")), bytes)?;
    }
    Ok(())
}

pub fn assert_mtp_dsa_compact_cache_ds4_fixture<D: BindingDevice + ManagedDevice>(
    context: &DeviceContext<D>,
    kv_raw: &D::Tensor,
    kv_norm: &D::Tensor,
    compact: &D::Tensor,
) -> anyhow::Result<()> {
    let fixture: &[u8] = if mtp_real_target_fixtures_enabled() {
        &crate::bindings::glm53_test_fixtures::read("mtp_dsa_compact_cache_natural_target_ds4.bin")
    } else {
        &crate::bindings::glm53_test_fixtures::read("mtp_dsa_compact_cache_layer45_ds4.bin")
    };
    const STAGE_BYTES: usize = 5 * 512 * std::mem::size_of::<f32>();
    const CACHE_BYTES: usize = 5 * 512 * std::mem::size_of::<u16>();
    anyhow::ensure!(
        fixture.len() == 2 * STAGE_BYTES + CACHE_BYTES,
        "DS4 layer-45 MTP compact-cache fixture has an invalid size"
    );
    let decode_f32 = |bytes: &[u8]| {
        bytes
            .chunks_exact(4)
            .map(|bytes| f32::from_le_bytes(bytes.try_into().expect("F32 fixture word")))
            .collect::<Vec<_>>()
    };
    let expected_raw = decode_f32(&fixture[..STAGE_BYTES]);
    let expected_norm = decode_f32(&fixture[STAGE_BYTES..2 * STAGE_BYTES]);
    let expected_cache: Vec<u16> = fixture[2 * STAGE_BYTES..]
        .chunks_exact(2)
        .map(|bytes| u16::from_le_bytes(bytes.try_into().expect("F16 fixture word")))
        .collect();
    let actual_raw = kv_raw.read_f32()?;
    let actual_norm = kv_norm.read_f32()?;
    let actual_cache = compact.read_u16(context)?;
    anyhow::ensure!(
        actual_raw.len() == 512 && actual_norm.len() == 512,
        "layer-45 MTP KV-LoRA projection stage has an invalid width"
    );
    let stage_error = |actual: &[f32], expected: &[f32]| {
        let max_abs = actual
            .iter()
            .zip(expected)
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0f32, f32::max);
        let ref_norm = expected
            .iter()
            .map(|value| value * value)
            .sum::<f32>()
            .sqrt();
        let diff_norm = actual
            .iter()
            .zip(expected)
            .map(|(actual, expected)| (actual - expected).powi(2))
            .sum::<f32>()
            .sqrt();
        (max_abs, diff_norm / (ref_norm + f32::MIN_POSITIVE))
    };
    let (raw_max_abs, raw_rel_l2) = stage_error(&actual_raw, &expected_raw[4 * 512..]);
    let (norm_max_abs, norm_rel_l2) = stage_error(&actual_norm, &expected_norm[4 * 512..]);
    anyhow::ensure!(
        raw_max_abs <= 1e-5 && raw_rel_l2 <= 1e-5,
        "layer-45 MTP kv_a projection differs from DS4: max_abs={raw_max_abs:e}, rel_l2={raw_rel_l2:e}"
    );
    anyhow::ensure!(
        norm_max_abs <= 1e-6 && norm_rel_l2 <= 1e-5,
        "layer-45 MTP KV-LoRA norm differs from DS4: max_abs={norm_max_abs:e}, rel_l2={norm_rel_l2:e}"
    );
    let mismatch = actual_cache
        .iter()
        .zip(&expected_cache)
        .position(|(actual, expected)| actual != expected);
    anyhow::ensure!(
        actual_cache.len() == expected_cache.len() && mismatch.is_none(),
        "layer-45 MTP compact cache differs from direct DS4 Metal at element {mismatch:?}"
    );
    eprintln!(
        "[glm53-mtp-dsa] kv_a max_abs={raw_max_abs:.3e} rel_l2={raw_rel_l2:.3e}; KV-LoRA norm max_abs={norm_max_abs:.3e} rel_l2={norm_rel_l2:.3e}; all 2,560 compact-cache F16 values match DS4"
    );
    Ok(())
}

pub fn assert_mtp_dsa_kv_a_row_ds4_fixture<D: BindingDevice + ManagedDevice>(
    _context: &DeviceContext<D>,
    kv_raw: &D::Tensor,
    kv_norm: &D::Tensor,
    row: usize,
) -> anyhow::Result<()> {
    const WIDTH: usize = 512;
    const ROW_BYTES: usize = WIDTH * std::mem::size_of::<f32>();
    const STAGE_BYTES: usize = 5 * ROW_BYTES;
    const CACHE_BYTES: usize = 5 * WIDTH * std::mem::size_of::<u16>();
    let fixture =
        &crate::bindings::glm53_test_fixtures::read("mtp_dsa_compact_cache_natural_target_ds4.bin");
    anyhow::ensure!(
        row < 5,
        "MTP DSA fixture row is outside its five-row window"
    );
    anyhow::ensure!(
        fixture.len() == 2 * STAGE_BYTES + CACHE_BYTES,
        "real-target MTP DSA KV fixture has an invalid size"
    );
    let expected_raw = &fixture[row * ROW_BYTES..(row + 1) * ROW_BYTES];
    let norm_offset = STAGE_BYTES + row * ROW_BYTES;
    let expected_norm = &fixture[norm_offset..norm_offset + ROW_BYTES];
    let compare = |name: &str, actual: Vec<f32>, expected: &[u8]| -> anyhow::Result<()> {
        anyhow::ensure!(
            actual.len() == WIDTH
                && actual
                    .iter()
                    .zip(expected.chunks_exact(4))
                    .all(|(actual, bytes)| {
                        actual.to_bits()
                            == f32::from_le_bytes(bytes.try_into().expect("F32 fixture value"))
                                .to_bits()
                    }),
            "real-target MTP DSA {name} row {row} differs from DS4"
        );
        Ok(())
    };
    compare("kv_a", kv_raw.read_f32()?, expected_raw)?;
    compare("KV-LoRA norm", kv_norm.read_f32()?, expected_norm)?;
    Ok(())
}

pub fn assert_mtp_dsa_ds4_fixture<D: BindingDevice + ManagedDevice>(
    _context: &DeviceContext<D>,
    q_low: &D::Tensor,
    lora: &D::Tensor,
    value: &D::Tensor,
) -> anyhow::Result<()> {
    let fixture: &[u8] = if mtp_real_target_fixtures_enabled() {
        &crate::bindings::glm53_test_fixtures::read("dsa_absorbed_mtp_natural_target_ds4.bin")
    } else {
        &crate::bindings::glm53_test_fixtures::read("dsa_absorbed_mtp_layer45_ds4.bin")
    };
    const LOW_BYTES: usize = 64 * 512 * std::mem::size_of::<f32>();
    const LORA_BYTES: usize = 64 * 512 * std::mem::size_of::<f32>();
    const VALUE_BYTES: usize = 64 * 256 * std::mem::size_of::<f32>();
    anyhow::ensure!(
        fixture.len() == LOW_BYTES + LORA_BYTES + VALUE_BYTES,
        "DS4 layer-45 MTP absorbed-attention fixture has an invalid size"
    );
    let fixture_f32 = |bytes: &[u8]| {
        bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("F32 fixture word")))
            .collect::<Vec<_>>()
    };
    let expected_q_low = fixture_f32(&fixture[..LOW_BYTES]);
    let expected_lora = fixture_f32(&fixture[LOW_BYTES..LOW_BYTES + LORA_BYTES]);
    let expected_value = fixture_f32(&fixture[LOW_BYTES + LORA_BYTES..]);
    let actual_q_low = q_low.read_f32()?;
    let actual_lora = lora.read_f32()?;
    let actual_value = value.read_f32()?;
    anyhow::ensure!(
        actual_q_low.len() == expected_q_low.len()
            && actual_lora.len() == expected_lora.len()
            && actual_value.len() == expected_value.len(),
        "DS4 layer-45 MTP absorbed-attention fixture shape mismatch"
    );
    let q_low_max_abs = actual_q_low
        .iter()
        .zip(&expected_q_low)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0f32, f32::max);
    let q_low_ref_norm = expected_q_low
        .iter()
        .map(|value| value * value)
        .sum::<f32>()
        .sqrt();
    let q_low_diff_norm = actual_q_low
        .iter()
        .zip(&expected_q_low)
        .map(|(actual, expected)| (actual - expected).powi(2))
        .sum::<f32>()
        .sqrt();
    anyhow::ensure!(
        q_low_max_abs <= 1e-5 && q_low_diff_norm <= q_low_ref_norm * 1e-5,
        "layer-45 MTP QK-lowrank differs from DS4: max_abs={q_low_max_abs:e}, rel_l2={:e}",
        q_low_diff_norm / (q_low_ref_norm + f32::MIN_POSITIVE)
    );
    anyhow::ensure!(
        actual_lora
            .iter()
            .map(|value| value.to_bits())
            .eq(expected_lora.iter().map(|value| value.to_bits())),
        "layer-45 MTP indexed-lora output differs from direct DS4 Metal"
    );
    anyhow::ensure!(
        actual_value
            .iter()
            .map(|value| value.to_bits())
            .eq(expected_value.iter().map(|value| value.to_bits())),
        "layer-45 MTP value projection differs from direct DS4 Metal"
    );
    eprintln!(
        "[glm53-mtp-dsa] q_low max_abs={q_low_max_abs:.3e} rel_l2={:.3e}; indexed-lora and value outputs bit-identical",
        q_low_diff_norm / (q_low_ref_norm + f32::MIN_POSITIVE)
    );
    Ok(())
}

pub fn assert_mtp_dsa_window_ds4_fixture<D: BindingDevice + ManagedDevice>(
    context: &DeviceContext<D>,
    q: &D::Tensor,
    compact: &D::Tensor,
    selected: &D::Tensor,
    q_low: &D::Tensor,
    lora: &D::Tensor,
    value: &D::Tensor,
    min_position: usize,
    position: usize,
) -> anyhow::Result<()> {
    const HEADS: usize = 64;
    const WIDTH: usize = 512;
    const QDIM: usize = 256;
    const VALUE: usize = 256;
    const HEADER_BYTES: usize = 3 * 4;
    const Q_BYTES: usize = HEADS * QDIM * 4;
    const LOW_BYTES: usize = HEADS * WIDTH * 4;
    const LORA_BYTES: usize = HEADS * WIDTH * 4;
    const VALUE_BYTES: usize = HEADS * VALUE * 4;
    let tokens = position.saturating_sub(min_position) + 1;
    let cache_bytes = tokens * WIDTH * 2;
    let selected_bytes = tokens * 4;
    let fixture: &[u8] = if mtp_real_target_fixtures_enabled() && position == 6 {
        &crate::bindings::glm53_test_fixtures::read("mtp_dsa_window_natural_target_step6_ds4.bin")
    } else if mtp_real_target_fixtures_enabled() {
        &crate::bindings::glm53_test_fixtures::read("mtp_dsa_window_natural_target_ds4.bin")
    } else {
        &crate::bindings::glm53_test_fixtures::read("mtp_dsa_window_layer45_ds4.bin")
    };
    anyhow::ensure!(
        position >= min_position
            && (tokens == 5 || (mtp_real_target_fixtures_enabled() && tokens == 6)),
        "MTP DSA window fixture has unexpected positions {min_position}..={position}"
    );
    anyhow::ensure!(
        fixture.len()
            == HEADER_BYTES
                + Q_BYTES
                + cache_bytes
                + selected_bytes
                + LOW_BYTES
                + LORA_BYTES
                + VALUE_BYTES,
        "DS4 layer-45 MTP DSA window fixture has an invalid size"
    );
    let header = fixture[..HEADER_BYTES]
        .chunks_exact(4)
        .map(|bytes| u32::from_le_bytes(bytes.try_into().expect("U32 window metadata")))
        .collect::<Vec<_>>();
    anyhow::ensure!(
        header == [min_position as u32, position as u32, tokens as u32],
        "layer-45 MTP DSA window position changed"
    );
    let mut offset = HEADER_BYTES;
    let expected_q = &fixture[offset..offset + Q_BYTES];
    offset += Q_BYTES;
    let expected_cache = &fixture[offset..offset + cache_bytes];
    offset += cache_bytes;
    let expected_selected = &fixture[offset..offset + selected_bytes];
    offset += selected_bytes;
    let expected_low = &fixture[offset..offset + LOW_BYTES];
    offset += LOW_BYTES;
    let expected_lora = &fixture[offset..offset + LORA_BYTES];
    offset += LORA_BYTES;
    let expected_value = &fixture[offset..];

    let q_values = q.read_f32()?;
    let cache_values = compact.read_u16(context)?;
    let selected_values = selected.read_i32(context)?;
    let q_low_values = q_low.read_f32()?;
    let lora_values = lora.read_f32()?;
    let value_values = value.read_f32()?;
    let bytes_f32 = |values: &[f32]| -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect()
    };
    let bytes_u16 = |values: &[u16]| -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect()
    };
    let bytes_i32 = |values: &[i32]| -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect()
    };
    let q_matches = bytes_f32(&q_values) == expected_q;
    let cache_matches = bytes_u16(&cache_values) == expected_cache;
    let selected_matches = bytes_i32(&selected_values) == expected_selected;
    anyhow::ensure!(
        q_matches && cache_matches && selected_matches,
        "layer-45 MTP DSA window inputs differ from DS4 fixture: q={q_matches}, cache={cache_matches}, selected={selected_matches}"
    );

    let decode_f32 = |bytes: &[u8]| {
        bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("F32 fixture word")))
            .collect::<Vec<_>>()
    };
    let compare_low = |actual: &[f32],
                       expected_bytes: &[u8],
                       name: &str|
     -> anyhow::Result<(f32, f64)> {
        let expected = decode_f32(expected_bytes);
        let max_abs = actual
            .iter()
            .zip(&expected)
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0f32, f32::max);
        let (max_index, max_actual, max_expected) = actual
            .iter()
            .zip(&expected)
            .enumerate()
            .max_by(|(_, (a0, e0)), (_, (a1, e1))| (*a0 - *e0).abs().total_cmp(&(*a1 - *e1).abs()))
            .map(|(index, (actual, expected))| (index, *actual, *expected))
            .unwrap_or((0, 0.0, 0.0));
        let delta2 = actual
            .iter()
            .zip(&expected)
            .map(|(actual, expected)| f64::from(actual - expected).powi(2))
            .sum::<f64>();
        let ref2 = expected
            .iter()
            .map(|value| f64::from(*value).powi(2))
            .sum::<f64>();
        let rel_l2 = (delta2 / ref2.max(f64::MIN_POSITIVE)).sqrt();
        let mismatches = bytes_f32(actual)
            .iter()
            .zip(expected_bytes)
            .filter(|(actual, expected)| actual != expected)
            .count();
        anyhow::ensure!(
            mismatches == 0,
            "MTP DSA window {name} differs from DS4: mismatches={mismatches}, max_abs={max_abs:e}, rel_l2={rel_l2:e}, at {max_index} LISA={max_actual:e} DS4={max_expected:e}"
        );
        Ok((max_abs, rel_l2))
    };
    let (q_low_max_abs, q_low_rel_l2) = compare_low(&q_low_values, expected_low, "QK-lowrank")?;
    for (name, actual, expected) in [
        ("indexed-lora", &lora_values, expected_lora),
        ("value projection", &value_values, expected_value),
    ] {
        let expected_values = decode_f32(expected);
        let max_abs = actual
            .iter()
            .zip(&expected_values)
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0f32, f32::max);
        let diff2 = actual
            .iter()
            .zip(&expected_values)
            .map(|(actual, expected)| f64::from(actual - expected).powi(2))
            .sum::<f64>();
        let ref2 = expected_values
            .iter()
            .map(|value| f64::from(*value).powi(2))
            .sum::<f64>();
        let rel_l2 = (diff2 / ref2.max(f64::MIN_POSITIVE)).sqrt();
        let mismatches = bytes_f32(actual)
            .iter()
            .zip(expected)
            .filter(|(actual, expected)| actual != expected)
            .count();
        anyhow::ensure!(
            mismatches == 0,
            "layer-45 MTP DSA window {name} differs from direct DS4 Metal: mismatches={mismatches}, max_abs={max_abs:e}, rel_l2={rel_l2:e}"
        );
        eprintln!(
            "[glm53-mtp-dsa-window] {name} max_abs={max_abs:.3e} rel_l2={rel_l2:.3e}; bit-identical"
        );
    }
    eprintln!(
        "[glm53-mtp-dsa-window] positions {min_position}..={position}; QK-lowrank max_abs={q_low_max_abs:.3e} rel_l2={q_low_rel_l2:.3e}; all compared stages are bit-identical to DS4"
    );
    Ok(())
}

pub fn capture_mtp_dsa_o_project_inputs<D: BindingDevice + ManagedDevice>(
    _context: &DeviceContext<D>,
    path: &std::path::Path,
    dsa: &Dsa<D>,
    input: &D::Tensor,
    output: &D::Tensor,
) -> anyhow::Result<()> {
    std::fs::create_dir_all(path)?;
    let write_u32 = |name: &str, values: &[u32]| -> anyhow::Result<()> {
        let bytes: Vec<u8> = values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        std::fs::write(path.join(name), bytes)?;
        Ok(())
    };
    let write_f32 = |name: &str, values: &[f32]| -> anyhow::Result<()> {
        let bytes: Vec<u8> = values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        std::fs::write(path.join(name), bytes)?;
        Ok(())
    };
    write_u32("o_proj_weight.u32", &dsa.o_proj.weight.read_raw_ids()?)?;
    write_f32("o_proj_scales.f32", &dsa.o_proj.scales.read_f32()?)?;
    write_f32("o_proj_biases.f32", &dsa.o_proj.biases.read_f32()?)?;
    write_f32("o_proj_input.f32", &input.read_f32()?)?;
    write_f32("lisa_o_proj_output.f32", &output.read_f32()?)?;
    Ok(())
}

pub fn capture_mtp_dsa_window_metadata(
    path: &std::path::Path,
    min_position: usize,
    position: usize,
    width: usize,
) -> anyhow::Result<()> {
    std::fs::create_dir_all(path)?;
    let metadata = [min_position as u32, position as u32, width as u32];
    let bytes: Vec<u8> = metadata
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();
    std::fs::write(path.join("window.u32"), bytes)?;
    Ok(())
}

pub fn assert_mtp_dsa_o_project_ds4_fixture<D: BindingDevice + ManagedDevice>(
    context: &DeviceContext<D>,
    input: &D::Tensor,
    output: &D::Tensor,
) -> anyhow::Result<()> {
    assert_mtp_dsa_o_project_fixture(
        context,
        input,
        output,
        &crate::bindings::glm53_test_fixtures::read(if mtp_real_target_fixtures_enabled() {
            "dsa_o_project_mtp_natural_target_ds4.bin"
        } else {
            "dsa_o_project_mtp_layer45_ds4.bin"
        }),
        if mtp_real_target_fixtures_enabled() {
            "layer-45 MTP natural target"
        } else {
            "layer-45 MTP"
        },
    )
}

pub fn assert_mtp_dsa_o_project_window_ds4_fixture<D: BindingDevice + ManagedDevice>(
    context: &DeviceContext<D>,
    input: &D::Tensor,
    output: &D::Tensor,
    position: usize,
) -> anyhow::Result<()> {
    assert_mtp_dsa_o_project_fixture(
        context,
        input,
        output,
        &crate::bindings::glm53_test_fixtures::read(
            if mtp_real_target_fixtures_enabled() && position == 6 {
                "dsa_o_project_mtp_natural_target_step6_ds4.bin"
            } else if mtp_real_target_fixtures_enabled() {
                "dsa_o_project_mtp_natural_target_window_ds4.bin"
            } else {
                "dsa_o_project_mtp_window_layer45_ds4.bin"
            },
        ),
        if mtp_real_target_fixtures_enabled() && position == 6 {
            "layer-45 MTP natural-target generated-step window"
        } else if mtp_real_target_fixtures_enabled() {
            "layer-45 MTP natural-target window"
        } else {
            "layer-45 MTP window"
        },
    )
}

pub fn assert_mtp_dsa_o_project_fixture<D: BindingDevice + ManagedDevice>(
    _context: &DeviceContext<D>,
    input: &D::Tensor,
    output: &D::Tensor,
    fixture: &[u8],
    label: &str,
) -> anyhow::Result<()> {
    const INPUT_BYTES: usize = 16_384 * std::mem::size_of::<f32>();
    const OUTPUT_BYTES: usize = 4_096 * std::mem::size_of::<f32>();
    anyhow::ensure!(
        fixture.len() == INPUT_BYTES + OUTPUT_BYTES,
        "DS4 {label} output-projection fixture has an invalid size"
    );
    let decode = |bytes: &[u8]| {
        bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("F32 fixture word")))
            .collect::<Vec<_>>()
    };
    let expected_input = decode(&fixture[..INPUT_BYTES]);
    let expected_output = decode(&fixture[INPUT_BYTES..]);
    let actual_input = input.read_f32()?;
    let actual_output = output.read_f32()?;
    anyhow::ensure!(
        actual_input
            .iter()
            .map(|value| value.to_bits())
            .eq(expected_input.iter().map(|value| value.to_bits())),
        "{label} DSA output-projection input changed from DS4 fixture"
    );
    anyhow::ensure!(
        actual_output
            .iter()
            .map(|value| value.to_bits())
            .eq(expected_output.iter().map(|value| value.to_bits())),
        "{label} DSA output projection differs from direct DS4 Metal"
    );
    eprintln!("[glm53-mtp-dsa] {label} o_proj output matches DS4 bit-for-bit");
    Ok(())
}
