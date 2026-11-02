//! DSA (`deepseek_sparse_attention`) layers for `glm5_next`.
//!
//! NoPE MLA (no RoPE at all):
//! q via a low-rank branch (`q_a` 4096->1536, RMS, `q_b` 1536->16384), the
//! shared latent `kv_a_proj_with_mqa` 4096->512 cached per token, per-head
//! k/v projections sliced out of the packed Q8 `kv_b` (512->64x(256+256)),
//! and a lightning indexer that picks the top-2048 key tokens per query.
//!
//! Long-context decode and prefill use absorbed MLA attention over selected
//! compact latents; contexts below the sparse crossover use dense projected
//! K/V. The long-context path must not fall back to a full-context projection.

use super::mhc::{mm, narrow};
use crate::core::cache::{FullAttentionCache, IndexerTape};
use crate::core::loader::{TensorSource, Weights};
use crate::core::quant::QuantizedLinear;
use lisa_mlx::{Array, Dtype, ops};

const LAYER_NORM_EPS: f32 = 1e-6;
const RAW_ROLLBACK_ROWS: usize = 32;

fn indexer_topk(scores: &Array, top_k: usize, stream: &lisa_mlx::Stream) -> Option<Array> {
    if crate::core::quant::glm53_indexer_topk_shuffle_override().unwrap_or(true) {
        lisa_mlx::models::glm53::kernels::dsa_indexer_topk_shuffle(scores, top_k, stream)
    } else {
        lisa_mlx::models::glm53::kernels::dsa_indexer_topk(scores, top_k, stream)
    }
}

#[cfg(all(test, feature = "glm53-fixtures"))]
std::thread_local! {
    static MTP_REAL_TARGET_FIXTURES: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

#[cfg(all(test, feature = "glm53-fixtures"))]
pub(crate) struct MtpRealTargetFixtureScope(bool);

#[cfg(all(test, feature = "glm53-fixtures"))]
impl Drop for MtpRealTargetFixtureScope {
    fn drop(&mut self) {
        MTP_REAL_TARGET_FIXTURES.with(|enabled| enabled.set(self.0));
    }
}

#[cfg(all(test, feature = "glm53-fixtures"))]
pub(crate) fn use_mtp_real_target_fixtures() -> MtpRealTargetFixtureScope {
    let previous = MTP_REAL_TARGET_FIXTURES.with(|enabled| enabled.replace(true));
    MtpRealTargetFixtureScope(previous)
}

#[cfg(all(test, feature = "glm53-fixtures"))]
pub(crate) fn mtp_real_target_fixtures_enabled() -> bool {
    MTP_REAL_TARGET_FIXTURES.with(std::cell::Cell::get)
}

pub struct Indexer {
    wq_b_w: Array,    // [4096, 1536], source dtype
    wk_w: Array,      // [128, 4096], source dtype
    k_norm_w: Array,  // [128]
    k_norm_b: Array,  // [128]
    weights_w: Array, // [32, 4096], source dtype
    gate_w: Array,    // [128, 4096], source dtype
    ape: Array,       // [4, 128] f32
    heads: usize,
    head_dim: usize,
    topk: usize,
    kpool: usize,
    always_tail: bool,
}

pub struct Dsa {
    q_a: QuantizedLinear,
    q_b: QuantizedLinear,
    kv_a: QuantizedLinear,
    kv_b: QuantizedLinear,
    o_proj: QuantizedLinear,
    q_a_norm: Array,  // F32 [1536]
    kv_a_norm: Array, // F32 [512]
    indexer: Indexer,
    heads: usize,
    q_head_dim: usize,
    v_head_dim: usize,
    norm_eps: f32,
}

#[cfg(all(test, feature = "glm53-fixtures"))]
fn capture_mtp_dsa_inputs(
    path: &std::path::Path,
    dsa: &Dsa,
    x: &Array,
    kv_raw: &Array,
    kv_norm: &Array,
    q_a_raw: &Array,
    q_a_normed: &Array,
    q: &Array,
    q_low: &Array,
    lora: &Array,
    attended: &Array,
    compact: &Array,
    selected: &Array,
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
    write_u32("kv_a_weight.u32", &dsa.kv_a.weight.to_vec1::<u32>()?)?;
    write_f32("kv_a_scales.f32", &dsa.kv_a.scales.to_vec1::<f32>()?)?;
    write_f32("kv_a_biases.f32", &dsa.kv_a.biases.to_vec1::<f32>()?)?;
    write_f32("kv_a_norm.f32", &dsa.kv_a_norm.to_vec1::<f32>()?)?;
    write_f32("kv_a_input.f32", &x.to_vec1::<f32>()?)?;
    write_f32("kv_a_eps.f32", &[dsa.norm_eps])?;
    write_f32("lisa_kv_a_raw.f32", &kv_raw.to_vec1::<f32>()?)?;
    write_f32("lisa_kv_a_norm.f32", &kv_norm.to_vec1::<f32>()?)?;
    write_f16("lisa_mtp_compact_cache.f16", &compact.to_vec1::<u16>()?)?;
    write_u32("q_a_weight.u32", &dsa.q_a.weight.to_vec1::<u32>()?)?;
    write_f32("q_a_scales.f32", &dsa.q_a.scales.to_vec1::<f32>()?)?;
    write_f32("q_a_biases.f32", &dsa.q_a.biases.to_vec1::<f32>()?)?;
    write_f32("q_a_norm.f32", &dsa.q_a_norm.to_vec1::<f32>()?)?;
    write_f32("lisa_q_a_raw.f32", &q_a_raw.to_vec1::<f32>()?)?;
    write_f32("lisa_q_a_norm.f32", &q_a_normed.to_vec1::<f32>()?)?;
    write_u32("q_b_weight.u32", &dsa.q_b.weight.to_vec1::<u32>()?)?;
    write_f32("q_b_scales.f32", &dsa.q_b.scales.to_vec1::<f32>()?)?;
    write_f32("q_b_biases.f32", &dsa.q_b.biases.to_vec1::<f32>()?)?;
    write_u32("weight.u32", &dsa.kv_b.weight.to_vec1::<u32>()?)?;
    write_f32("scales.f32", &dsa.kv_b.scales.to_vec1::<f32>()?)?;
    write_f32("biases.f32", &dsa.kv_b.biases.to_vec1::<f32>()?)?;
    write_f32("q.f32", &q.to_vec1::<f32>()?)?;
    write_f32("lisa_q_low.f32", &q_low.to_vec1::<f32>()?)?;
    write_f32("lisa_lora.f32", &lora.to_vec1::<f32>()?)?;
    write_f32("lisa_pre_o.f32", &attended.to_vec1::<f32>()?)?;
    write_f16("compact_cache.f16", &compact.to_vec1::<u16>()?)?;
    write_i32("selected.u32", &selected.to_vec1::<i32>()?)?;
    Ok(())
}

#[cfg(all(test, feature = "glm53-fixtures"))]
const DS4_DSA_VERIFY_LAYERS: [usize; 11] = [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43];

#[cfg(all(test, feature = "glm53-fixtures"))]
fn ds4_dsa_verify_fixture_layer_index(layer: usize) -> Option<usize> {
    DS4_DSA_VERIFY_LAYERS
        .iter()
        .position(|&candidate| candidate == layer)
}

#[cfg(all(test, feature = "glm53-fixtures"))]
fn ds4_dsa_verify_stages_fixture(layer: usize) -> Option<&'static [u8]> {
    const PER_LAYER: usize = 655_360;
    const FIXTURE: &[u8] =
        include_bytes!("fixtures/dsa_verify_target_mtp_all_layers_stages_ds4.bin");
    let index = ds4_dsa_verify_fixture_layer_index(layer)?;
    Some(&FIXTURE[index * PER_LAYER..(index + 1) * PER_LAYER])
}

#[cfg(all(test, feature = "glm53-fixtures"))]
fn ds4_dsa_verify_kv_a_fixture(layer: usize) -> Option<&'static [u8]> {
    const PER_LAYER: usize = 10_240;
    const FIXTURE: &[u8] = include_bytes!("fixtures/dsa_verify_target_mtp_all_layers_kv_a_ds4.bin");
    let index = ds4_dsa_verify_fixture_layer_index(layer)?;
    Some(&FIXTURE[index * PER_LAYER..(index + 1) * PER_LAYER])
}

#[cfg(all(test, feature = "glm53-fixtures"))]
fn capture_dsa_verify_attention_inputs(
    path: &std::path::Path,
    dsa: &Dsa,
    layer: usize,
    q: &Array,
    compact: &Array,
    lisa_pre_o: &Array,
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
    let q_rows = q.transpose_axes(&[0, 2, 1, 3])?.contiguous()?;
    write_u32("kv_b_weight.u32", &dsa.kv_b.weight.to_vec1::<u32>()?)?;
    write_f32("kv_b_scales.f32", &dsa.kv_b.scales.to_vec1::<f32>()?)?;
    write_f32("kv_b_biases.f32", &dsa.kv_b.biases.to_vec1::<f32>()?)?;
    write_f32("q.f32", &q_rows.to_vec1::<f32>()?)?;
    write_f16("compact_cache.f16", &compact.to_vec1::<u16>()?)?;
    write_f32("lisa_pre_o.f32", &lisa_pre_o.to_vec1::<f32>()?)?;
    let actual = lisa_pre_o.to_vec1::<f32>()?;
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

#[cfg(all(test, feature = "glm53-fixtures"))]
fn capture_dsa_native_decode_attention_inputs(
    path: &std::path::Path,
    dsa: &Dsa,
    q: &Array,
    compact: &Array,
    selected: &Array,
    q_low: &Array,
    lora: &Array,
    lisa_fused_lora: &Array,
    lisa_pre_o: &Array,
    lisa_fused_pre_o: &Array,
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
    let q_rows = q.transpose_axes(&[0, 2, 1, 3])?.contiguous()?;
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
    write_u32("kv_b_weight.u32", &dsa.kv_b.weight.to_vec1::<u32>()?)?;
    write_f32("kv_b_scales.f32", &dsa.kv_b.scales.to_vec1::<f32>()?)?;
    write_f32("kv_b_biases.f32", &dsa.kv_b.biases.to_vec1::<f32>()?)?;
    write_f32("q.f32", &q_rows.to_vec1::<f32>()?)?;
    write_f16("compact_cache.f16", &compact.to_vec1::<u16>()?)?;
    write_i32("selected.i32", &selected.to_vec1::<i32>()?)?;
    write_f32("lisa_q_low.f32", &q_low.to_vec1::<f32>()?)?;
    write_f32("lisa_lora.f32", &lora.to_vec1::<f32>()?)?;
    write_f32("lisa_fused_lora.f32", &lisa_fused_lora.to_vec1::<f32>()?)?;
    write_f32("lisa_pre_o.f32", &lisa_pre_o.to_vec1::<f32>()?)?;
    write_f32("lisa_fused_pre_o.f32", &lisa_fused_pre_o.to_vec1::<f32>()?)?;
    Ok(())
}

#[cfg(all(test, feature = "glm53-fixtures"))]
fn capture_dsa_native_batch_o_project_inputs(
    path: &std::path::Path,
    dsa: &Dsa,
    input: &Array,
    lisa_output: &Array,
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
    write_u32("o_proj_weight.u32", &dsa.o_proj.weight.to_vec1::<u32>()?)?;
    write_f32("o_proj_scales.f32", &dsa.o_proj.scales.to_vec1::<f32>()?)?;
    write_f32("o_proj_biases.f32", &dsa.o_proj.biases.to_vec1::<f32>()?)?;
    write_f32("input.f32", &input.reshape(&[2, 16_384])?.to_vec1::<f32>()?)?;
    write_f32(
        "lisa_output.f32",
        &lisa_output.reshape(&[2, 4096])?.to_vec1::<f32>()?,
    )?;
    Ok(())
}

#[cfg(all(test, feature = "glm53-fixtures"))]
fn capture_dsa_verify_kv_a_inputs(
    path: &std::path::Path,
    dsa: &Dsa,
    layer: usize,
    pos0: usize,
    x: &Array,
    kv_raw: &Array,
    kv_norm: Option<&Array>,
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
    write_u32("kv_a_weight.u32", &dsa.kv_a.weight.to_vec1::<u32>()?)?;
    write_f32("kv_a_scales.f32", &dsa.kv_a.scales.to_vec1::<f32>()?)?;
    write_f32("kv_a_biases.f32", &dsa.kv_a.biases.to_vec1::<f32>()?)?;
    write_f32("kv_a_norm.f32", &dsa.kv_a_norm.to_vec1::<f32>()?)?;
    write_f32("kv_a_eps.f32", &[dsa.norm_eps])?;
    for row in 0..rows {
        write_f32(
            &format!("kv_a_input_{row}.f32"),
            &narrow(&input_rows, 0, row, 1)?
                .reshape(&[4096])?
                .to_vec1::<f32>()?,
        )?;
        write_f32(
            &format!("lisa_kv_a_raw_{row}.f32"),
            &narrow(&raw_rows, 0, row, 1)?
                .reshape(&[512])?
                .to_vec1::<f32>()?,
        )?;
        if let Some(norm_rows) = &norm_rows {
            write_f32(
                &format!("lisa_kv_a_norm_{row}.f32"),
                &narrow(norm_rows, 0, row, 1)?
                    .reshape(&[512])?
                    .to_vec1::<f32>()?,
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
        let values = raw_rows.to_vec1::<f32>()?;
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
                .to_vec1::<f32>()?;
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
            .as_dtype(lisa_mlx::Dtype::Float16)?
            .to_vec1::<u16>()?;
        write_u16("lisa_compact_cache.f16", &compact)?;
    }
    Ok(())
}

#[cfg(all(test, feature = "glm53-fixtures"))]
fn capture_dsa_verify_compact_cache(
    path: &std::path::Path,
    layer: usize,
    pos0: usize,
    compact: &Array,
) -> anyhow::Result<()> {
    let values = compact.to_vec1::<u16>()?;
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

#[cfg(all(test, feature = "glm53-fixtures"))]
fn capture_dsa_verify_latent_stages(
    path: &std::path::Path,
    q_low: &Array,
    lora: &Array,
) -> anyhow::Result<()> {
    std::fs::create_dir_all(path)?;
    for (name, stage) in [("lisa_q_low.f32", q_low), ("lisa_lora.f32", lora)] {
        let bytes: Vec<u8> = stage
            .to_vec1::<f32>()?
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        std::fs::write(path.join(name), bytes)?;
    }
    Ok(())
}

#[cfg(all(test, feature = "glm53-fixtures"))]
fn capture_mtp_dsa_kv_a_input_row(
    path: &std::path::Path,
    x: &Array,
    kv_raw: &Array,
    kv_norm: &Array,
    min_position: usize,
    position: usize,
) -> anyhow::Result<()> {
    std::fs::create_dir_all(path)?;
    anyhow::ensure!(
        position >= min_position,
        "MTP DSA capture position precedes its cache window"
    );
    let values = x.to_vec1::<f32>()?;
    let row = position - min_position;
    let bytes: Vec<u8> = values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();
    std::fs::write(path.join(format!("kv_a_input_{row}.f32")), bytes)?;
    for (name, stage) in [("kv_a_raw", kv_raw), ("kv_a_norm", kv_norm)] {
        let values = stage.to_vec1::<f32>()?;
        let bytes: Vec<u8> = values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        std::fs::write(path.join(format!("lisa_{name}_{row}.f32")), bytes)?;
    }
    Ok(())
}

#[cfg(all(test, feature = "glm53-fixtures"))]
fn assert_mtp_dsa_compact_cache_ds4_fixture(
    kv_raw: &Array,
    kv_norm: &Array,
    compact: &Array,
) -> anyhow::Result<()> {
    let fixture: &[u8] = if mtp_real_target_fixtures_enabled() {
        include_bytes!("fixtures/mtp_dsa_compact_cache_natural_target_ds4.bin")
    } else {
        include_bytes!("fixtures/mtp_dsa_compact_cache_layer45_ds4.bin")
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
    let actual_raw = kv_raw.to_vec1::<f32>()?;
    let actual_norm = kv_norm.to_vec1::<f32>()?;
    let actual_cache = compact.to_vec1::<u16>()?;
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

#[cfg(all(test, feature = "glm53-fixtures"))]
fn assert_mtp_dsa_kv_a_row_ds4_fixture(
    kv_raw: &Array,
    kv_norm: &Array,
    row: usize,
) -> anyhow::Result<()> {
    const WIDTH: usize = 512;
    const ROW_BYTES: usize = WIDTH * std::mem::size_of::<f32>();
    const STAGE_BYTES: usize = 5 * ROW_BYTES;
    const CACHE_BYTES: usize = 5 * WIDTH * std::mem::size_of::<u16>();
    let fixture = include_bytes!("fixtures/mtp_dsa_compact_cache_natural_target_ds4.bin");
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
    compare("kv_a", kv_raw.to_vec1::<f32>()?, expected_raw)?;
    compare("KV-LoRA norm", kv_norm.to_vec1::<f32>()?, expected_norm)?;
    Ok(())
}

#[cfg(all(test, feature = "glm53-fixtures"))]
fn assert_mtp_dsa_ds4_fixture(q_low: &Array, lora: &Array, value: &Array) -> anyhow::Result<()> {
    let fixture: &[u8] = if mtp_real_target_fixtures_enabled() {
        include_bytes!("fixtures/dsa_absorbed_mtp_natural_target_ds4.bin")
    } else {
        include_bytes!("fixtures/dsa_absorbed_mtp_layer45_ds4.bin")
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
    let actual_q_low = q_low.to_vec1::<f32>()?;
    let actual_lora = lora.to_vec1::<f32>()?;
    let actual_value = value.to_vec1::<f32>()?;
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

#[cfg(all(test, feature = "glm53-fixtures"))]
fn assert_mtp_dsa_window_ds4_fixture(
    q: &Array,
    compact: &Array,
    selected: &Array,
    q_low: &Array,
    lora: &Array,
    value: &Array,
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
        include_bytes!("fixtures/mtp_dsa_window_natural_target_step6_ds4.bin")
    } else if mtp_real_target_fixtures_enabled() {
        include_bytes!("fixtures/mtp_dsa_window_natural_target_ds4.bin")
    } else {
        include_bytes!("fixtures/mtp_dsa_window_layer45_ds4.bin")
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

    let q_values = q.to_vec1::<f32>()?;
    let cache_values = compact.to_vec1::<u16>()?;
    let selected_values = selected.to_vec1::<i32>()?;
    let q_low_values = q_low.to_vec1::<f32>()?;
    let lora_values = lora.to_vec1::<f32>()?;
    let value_values = value.to_vec1::<f32>()?;
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

#[cfg(all(test, feature = "glm53-fixtures"))]
fn capture_mtp_dsa_o_project_inputs(
    path: &std::path::Path,
    dsa: &Dsa,
    input: &Array,
    output: &Array,
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
    write_u32("o_proj_weight.u32", &dsa.o_proj.weight.to_vec1::<u32>()?)?;
    write_f32("o_proj_scales.f32", &dsa.o_proj.scales.to_vec1::<f32>()?)?;
    write_f32("o_proj_biases.f32", &dsa.o_proj.biases.to_vec1::<f32>()?)?;
    write_f32("o_proj_input.f32", &input.to_vec1::<f32>()?)?;
    write_f32("lisa_o_proj_output.f32", &output.to_vec1::<f32>()?)?;
    Ok(())
}

#[cfg(all(test, feature = "glm53-fixtures"))]
fn capture_mtp_dsa_window_metadata(
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

#[cfg(all(test, feature = "glm53-fixtures"))]
fn assert_mtp_dsa_o_project_ds4_fixture(input: &Array, output: &Array) -> anyhow::Result<()> {
    assert_mtp_dsa_o_project_fixture(
        input,
        output,
        if mtp_real_target_fixtures_enabled() {
            include_bytes!("fixtures/dsa_o_project_mtp_natural_target_ds4.bin")
        } else {
            include_bytes!("fixtures/dsa_o_project_mtp_layer45_ds4.bin")
        },
        if mtp_real_target_fixtures_enabled() {
            "layer-45 MTP natural target"
        } else {
            "layer-45 MTP"
        },
    )
}

#[cfg(all(test, feature = "glm53-fixtures"))]
fn assert_mtp_dsa_o_project_window_ds4_fixture(
    input: &Array,
    output: &Array,
    position: usize,
) -> anyhow::Result<()> {
    assert_mtp_dsa_o_project_fixture(
        input,
        output,
        if mtp_real_target_fixtures_enabled() && position == 6 {
            include_bytes!("fixtures/dsa_o_project_mtp_natural_target_step6_ds4.bin")
        } else if mtp_real_target_fixtures_enabled() {
            include_bytes!("fixtures/dsa_o_project_mtp_natural_target_window_ds4.bin")
        } else {
            include_bytes!("fixtures/dsa_o_project_mtp_window_layer45_ds4.bin")
        },
        if mtp_real_target_fixtures_enabled() && position == 6 {
            "layer-45 MTP natural-target generated-step window"
        } else if mtp_real_target_fixtures_enabled() {
            "layer-45 MTP natural-target window"
        } else {
            "layer-45 MTP window"
        },
    )
}

#[cfg(all(test, feature = "glm53-fixtures"))]
fn assert_mtp_dsa_o_project_fixture(
    input: &Array,
    output: &Array,
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
    let actual_input = input.to_vec1::<f32>()?;
    let actual_output = output.to_vec1::<f32>()?;
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

pub(super) struct DsaProjections {
    latent: Option<Array>,
    compact_written: Option<(usize, usize)>,
    query_rank: Array,
    query: Array,
    #[cfg(test)]
    pub(super) kv_raw: Array,
    pub(super) indexer_key: Array,
    pub(super) indexer_gate: Array,
}

impl Dsa {
    pub fn load<S: TensorSource>(
        w: &mut Weights,
        prefix: &str,
        heads: usize,
        q_head_dim: usize,
        v_head_dim: usize,
        index_heads: usize,
        index_head_dim: usize,
        index_topk: usize,
        index_kpool: usize,
        always_tail: bool,
        norm_eps: f32,
    ) -> anyhow::Result<Self> {
        let take = |w: &mut Weights, name: &str| -> anyhow::Result<Array> {
            w.remove(name)
                .ok_or_else(|| anyhow::anyhow!("glm53 dsa: {name} missing"))
        };
        let p = |s: &str| format!("{prefix}.{s}");
        let mut q_a = QuantizedLinear::load(w, &p("q_a_proj"), "")?;
        q_a.set_quant(64, 8);
        q_a.scales = q_a.scales.as_dtype(Dtype::Float32)?;
        q_a.biases = q_a.biases.as_dtype(Dtype::Float32)?;
        let mut q_b = QuantizedLinear::load(w, &p("q_b_proj"), "")?;
        q_b.set_quant(64, 8);
        q_b.scales = q_b.scales.as_dtype(Dtype::Float32)?;
        q_b.biases = q_b.biases.as_dtype(Dtype::Float32)?;
        let mut kv_a = QuantizedLinear::load(w, &p("kv_a_proj_with_mqa"), "")?;
        kv_a.set_quant(64, 8);
        kv_a.scales = kv_a.scales.as_dtype(Dtype::Float32)?;
        kv_a.biases = kv_a.biases.as_dtype(Dtype::Float32)?;
        let mut kv_b = QuantizedLinear::load(w, &p("kv_b_proj"), "")?;
        kv_b.set_quant(64, 8);
        kv_b.scales = kv_b.scales.as_dtype(Dtype::Float32)?;
        kv_b.biases = kv_b.biases.as_dtype(Dtype::Float32)?;
        let mut o_proj = QuantizedLinear::load(w, &p("o_proj"), "")?;
        o_proj.set_quant(64, 8);
        o_proj.scales = o_proj.scales.as_dtype(Dtype::Float32)?;
        o_proj.biases = o_proj.biases.as_dtype(Dtype::Float32)?;
        let q_a_norm = take(w, &p("q_a_layernorm.weight"))?.as_dtype(Dtype::Float32)?;
        let kv_a_norm = take(w, &p("kv_a_layernorm.weight"))?.as_dtype(Dtype::Float32)?;

        let indexer = Indexer {
            wq_b_w: take(w, &p("indexer.wq_b.weight"))?,
            wk_w: take(w, &p("indexer.wk.weight"))?,
            k_norm_w: take(w, &p("indexer.k_norm.weight"))?.as_dtype(Dtype::Float32)?,
            k_norm_b: take(w, &p("indexer.k_norm.bias"))?.as_dtype(Dtype::Float32)?,
            weights_w: take(w, &p("indexer.weights_proj.weight"))?,
            gate_w: take(w, &p("indexer.index_kpool_compress_gate"))?,
            ape: take(w, &p("indexer.index_kpool_compress_ape"))?,
            heads: index_heads,
            head_dim: index_head_dim,
            topk: index_topk,
            kpool: index_kpool,
            always_tail,
        };
        Ok(Self {
            q_a,
            q_b,
            kv_a,
            kv_b,
            o_proj,
            q_a_norm,
            kv_a_norm,
            indexer,
            heads,
            q_head_dim,
            v_head_dim,
            norm_eps,
        })
    }

    #[cfg(all(test, feature = "glm53-fixtures"))]
    pub(super) fn indexer_pool_params_for_capture(
        &self,
    ) -> anyhow::Result<(Vec<f32>, Vec<f32>, Vec<u16>)> {
        Ok((
            self.indexer.k_norm_w.to_vec1::<f32>()?,
            self.indexer.k_norm_b.to_vec1::<f32>()?,
            self.indexer.ape.to_vec1::<u16>()?,
        ))
    }

    #[cfg(all(test, feature = "glm53-fixtures"))]
    pub(super) fn indexer_projection_weights_for_capture(
        &self,
    ) -> anyhow::Result<(Vec<u16>, Vec<u16>)> {
        anyhow::ensure!(
            self.indexer.wk_w.dtype() == Dtype::Bfloat16
                && self.indexer.gate_w.dtype() == Dtype::Bfloat16,
            "GLM indexer capture expects BF16 key and gate weights"
        );
        Ok((
            self.indexer.wk_w.to_vec1::<u16>()?,
            self.indexer.gate_w.to_vec1::<u16>()?,
        ))
    }

    #[cfg(all(test, feature = "glm53-fixtures"))]
    pub(super) fn kv_b_quant_params_for_capture(
        &self,
    ) -> anyhow::Result<(Vec<u32>, Vec<f32>, Vec<f32>)> {
        Ok((
            self.kv_b.weight.to_vec1::<u32>()?,
            self.kv_b.scales.to_vec1::<f32>()?,
            self.kv_b.biases.to_vec1::<f32>()?,
        ))
    }

    #[cfg(all(test, feature = "glm53-fixtures"))]
    pub(super) fn kv_a_quant_params_for_capture(
        &self,
    ) -> anyhow::Result<(Vec<u32>, Vec<f32>, Vec<f32>, Vec<f32>, i32, i32, f32)> {
        Ok((
            self.kv_a.weight.to_vec1::<u32>()?,
            self.kv_a.scales.to_vec1::<f32>()?,
            self.kv_a.biases.to_vec1::<f32>()?,
            self.kv_a_norm.to_vec1::<f32>()?,
            self.kv_a.group_size,
            self.kv_a.bits,
            self.norm_eps,
        ))
    }

    #[cfg(all(test, feature = "glm53-fixtures"))]
    pub(super) fn kv_a_norm_for_capture(&self, raw: &Array) -> anyhow::Result<Array> {
        glm53_kv_lora_rms_norm(raw, &self.kv_a_norm, self.norm_eps)
    }

    /// GLM's MTP DSA step uses a private contiguous window and deliberately
    /// skips the learned target-layer indexer. `position` is the absolute
    /// target position represented by `x`; selected rows are `[min_position,
    /// position]`, matching ds4's `glm_graph_mtp_step`.
    pub fn forward_mtp(
        &self,
        x: &Array,
        tape: &mut IndexerTape,
        position: usize,
        min_position: usize,
    ) -> anyhow::Result<Array> {
        anyhow::ensure!(
            x.dim(0) == 1 && x.dim(1) == 1,
            "glm53 MTP DSA expects one token, got {:?}",
            x.shape()
        );
        anyhow::ensure!(min_position <= position, "glm53 MTP window is inverted");

        let heads = self.heads;
        let qd = self.q_head_dim;
        let vd = self.v_head_dim;
        let scale = (qd as f64).powf(-0.5) as f32;
        // MTP uses the same DS4 projection dispatcher as the trunk. Preserve
        // the Q-A -> Q norm -> Q-B -> KV-A order and the native one-row Q8
        // reduction before converting the private compact cache to F16.
        let q_a_raw = self.q_a.forward_glm53_prefill(x)?;
        let qr = glm53_weighted_rms_norm(&q_a_raw, &self.q_a_norm, self.norm_eps)?;
        let q = self
            .q_b
            .forward_glm53_prefill(&qr)?
            .reshape(&[1, 1, heads as i32, qd as i32])?
            .transpose_axes(&[0, 2, 1, 3])?
            .contiguous()?;
        let kv_raw = self.kv_a.forward_glm53_prefill(x)?;
        let c = glm53_kv_lora_rms_norm(&kv_raw, &self.kv_a_norm, self.norm_eps)?;
        #[cfg(all(test, feature = "glm53-fixtures"))]
        if mtp_real_target_fixtures_enabled()
            && std::env::var_os("LISA_GLM53_MTP_DSA_WINDOW_CAPTURE_DIR").is_none()
            && position - min_position < 5
        {
            assert_mtp_dsa_kv_a_row_ds4_fixture(&kv_raw, &c, position - min_position)?;
        }
        #[cfg(all(test, feature = "glm53-fixtures"))]
        let kv_a_capture_path =
            std::env::var_os("LISA_GLM53_MTP_DSA_WINDOW_CAPTURE_DIR").or_else(|| {
                if min_position == 0 {
                    std::env::var_os("LISA_GLM53_MTP_DSA_CAPTURE_DIR")
                } else {
                    None
                }
            });
        #[cfg(all(test, feature = "glm53-fixtures"))]
        if let Some(path) = kv_a_capture_path {
            let path = std::path::Path::new(&path);
            capture_mtp_dsa_kv_a_input_row(path, x, &kv_raw, &c, min_position, position)?;
        }
        tape.write_compact_f16_at(&c, position)
            .map_err(|e| anyhow::anyhow!("glm53 MTP compact cache: {e}"))?;

        #[cfg(all(test, feature = "glm53-fixtures"))]
        let is_window_fixture = (min_position == 15 && position == 19)
            || (mtp_real_target_fixtures_enabled()
                && min_position == 1
                && (position == 5 || position == 6));
        #[cfg(not(all(test, feature = "glm53-fixtures")))]
        let is_window_fixture = false;
        let compact_start = if is_window_fixture { min_position } else { 0 };
        let compact = tape
            .compact_slice(compact_start, position + 1)
            .ok_or_else(|| anyhow::anyhow!("glm53 MTP compact cache missing at {position}"))?;
        let width = position - min_position + 1;
        let selected_start = if is_window_fixture { 0 } else { min_position };
        let selected: Vec<i32> = (selected_start..selected_start + width)
            .map(|p| p as i32)
            .collect();
        let selected = Array::from_slice(&selected, &[1, 1, width as i32]);
        let stream = lisa_mlx::Stream::thread_local_or_default();
        #[cfg(all(test, feature = "glm53-fixtures"))]
        let attended = if (min_position == 0 && position == 4) || is_window_fixture {
            let (attended, q_low, lora) =
                lisa_mlx::models::glm53::kernels::dsa_absorbed_attention_stages(
                    &q,
                    &self.kv_b.weight,
                    &self.kv_b.scales,
                    &self.kv_b.biases,
                    self.kv_b.group_size as usize,
                    &compact,
                    &selected,
                    scale,
                    &stream,
                    false,
                )
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "glm53 MTP indexed absorbed attention declined at position {position}"
                    )
                })?;
            if is_window_fixture {
                let path = std::env::var_os("LISA_GLM53_MTP_DSA_WINDOW_CAPTURE_DIR");
                if let Some(path) = path {
                    let path = std::path::Path::new(&path).join(format!("position-{position}"));
                    capture_mtp_dsa_inputs(
                        &path, self, x, &kv_raw, &c, &q_a_raw, &qr, &q, &q_low, &lora, &attended,
                        &compact, &selected,
                    )?;
                    capture_mtp_dsa_window_metadata(&path, min_position, position, width)?;
                } else {
                    assert_mtp_dsa_window_ds4_fixture(
                        &q,
                        &compact,
                        &selected,
                        &q_low,
                        &lora,
                        &attended,
                        min_position,
                        position,
                    )?;
                    if mtp_real_target_fixtures_enabled() && position == 5 {
                        assert_mtp_dsa_compact_cache_ds4_fixture(&kv_raw, &c, &compact)?;
                    }
                }
            } else if let Some(path) = std::env::var_os("LISA_GLM53_MTP_DSA_CAPTURE_DIR") {
                capture_mtp_dsa_inputs(
                    std::path::Path::new(&path),
                    self,
                    x,
                    &kv_raw,
                    &c,
                    &q_a_raw,
                    &qr,
                    &q,
                    &q_low,
                    &lora,
                    &attended,
                    &compact,
                    &selected,
                )?;
            } else {
                assert_mtp_dsa_compact_cache_ds4_fixture(&kv_raw, &c, &compact)?;
                assert_mtp_dsa_ds4_fixture(&q_low, &lora, &attended)?;
            }
            attended
        } else {
            lisa_mlx::models::glm53::kernels::dsa_absorbed_attention(
                &q,
                &self.kv_b.weight,
                &self.kv_b.scales,
                &self.kv_b.biases,
                self.kv_b.group_size as usize,
                &compact,
                &selected,
                scale,
                &stream,
            )
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "glm53 MTP indexed absorbed attention declined at position {position}"
                )
            })?
        };
        #[cfg(not(all(test, feature = "glm53-fixtures")))]
        let attended = lisa_mlx::models::glm53::kernels::dsa_absorbed_attention(
            &q,
            &self.kv_b.weight,
            &self.kv_b.scales,
            &self.kv_b.biases,
            self.kv_b.group_size as usize,
            &compact,
            &selected,
            scale,
            &stream,
        )
        .ok_or_else(|| {
            anyhow::anyhow!("glm53 MTP indexed absorbed attention declined at position {position}")
        })?;
        let out = attended
            .transpose_axes(&[0, 2, 1, 3])?
            .contiguous()?
            .reshape(&[1, 1, (heads * vd) as i32])?;
        let projected = self.output_project(&out)?;
        #[cfg(all(test, feature = "glm53-fixtures"))]
        if (min_position == 0 && position == 4) || is_window_fixture {
            let capture_path = if is_window_fixture {
                std::env::var_os("LISA_GLM53_MTP_DSA_WINDOW_CAPTURE_DIR")
            } else {
                std::env::var_os("LISA_GLM53_MTP_DSA_CAPTURE_DIR")
            };
            if let Some(path) = capture_path {
                let path = std::path::Path::new(&path).join(format!("position-{position}"));
                capture_mtp_dsa_o_project_inputs(&path, self, &out, &projected)?;
            } else if is_window_fixture {
                assert_mtp_dsa_o_project_window_ds4_fixture(&out, &projected, position)?;
            } else {
                assert_mtp_dsa_o_project_ds4_fixture(&out, &projected)?;
            }
        }
        Ok(projected)
    }

    fn output_project(&self, x: &Array) -> anyhow::Result<Array> {
        if crate::core::quant::glm53_q8_native_prefill_enabled() {
            // DS4's native Q8 crossover applies to the attention output
            // projection as well as the input and feed-forward projections.
            return Ok(self.o_proj.forward_glm53_prefill(x)?);
        }
        if x.rank() == 3 && x.dim(2) == 16_384 && x.dtype() == Dtype::Float32 {
            let rows = (x.dim(0) as usize)
                .checked_mul(x.dim(1) as usize)
                .ok_or_else(|| anyhow::anyhow!("glm53 DSA output row count overflow"))?;
            if (1..=31).contains(&rows) {
                // The custom-affine DS4 route uses its per-row affine matvec
                // for small row counts; the generic F32 `_4` kernel has a
                // different reduction schedule even with identical weights.
                return Ok(self.o_proj.forward_glm53_prefill(x)?);
            }
            let rows_i32 = i32::try_from(rows)
                .map_err(|_| anyhow::anyhow!("glm53 DSA output row count exceeds i32"))?;
            let flat = x.reshape(&[rows_i32, 16_384])?;
            let stream = lisa_mlx::Stream::thread_local_or_default();
            let out = lisa_mlx::models::glm53::kernels::dsa_o_project(
                &flat,
                &self.o_proj.weight,
                &self.o_proj.scales,
                &self.o_proj.biases,
                self.o_proj.group_size as usize,
                &stream,
            )
            .ok_or_else(|| anyhow::anyhow!("glm53 DS4-scheduled DSA output projection declined"))?;
            return Ok(out.reshape(&[x.dim(0), x.dim(1), 4096])?);
        }
        Ok(self.o_proj.forward_glm53_prefill(x)?)
    }

    pub(super) fn project_inputs(
        &self,
        x: &Array,
        cap: Option<&crate::core::capture::Capture>,
        layer: usize,
        compact_tape: Option<&mut crate::core::cache::IndexerTape>,
    ) -> anyhow::Result<DsaProjections> {
        let (b, t) = (x.dim(0) as usize, x.dim(1) as usize);
        if let Some(capture) = cap.filter(|capture| capture.wants(layer)) {
            capture.store(&format!("L{layer}.dsa_indexer_input_x"), x);
        }
        let heads = self.heads;
        let qd = self.q_head_dim;

        let capture = cap.filter(|capture| capture.wants(layer));
        let project_separately = || -> anyhow::Result<(Array, Array)> {
            // Keep kv_a on the standard affine operator so each shape follows
            // DS4's graph schedule. The former MTP-specific reduction tree
            // changed F32 outputs and sometimes moved the stored F16 value.
            let c_raw = self.kv_a.forward_glm53_prefill(x)?;
            let query_rank_raw = self.q_a.forward_glm53_prefill(x)?;
            Ok((c_raw, query_rank_raw))
        };
        // Keep q_a and kv_a on their separate DS4 projection schedules.
        // The paired kernel uses MLX qmv_fast for q_a, while DS4's graph
        // dispatches its affine matvec path for each projection. That changes
        // the reduction order and is visible in the composed attention output.
        let (c_raw, query_rank_raw) = project_separately()?;
        #[cfg(all(test, feature = "glm53-fixtures"))]
        let compact_position = compact_tape.as_ref().map(|tape| tape.total);
        let mut compact_written = None;
        let capture_production_qkv = capture.is_some_and(|capture| capture.preserves_dispatch())
            || (cfg!(test) && std::env::var_os("LISA_GLM53_DSA_KVA_VERIFY_CAPTURE_DIR").is_some());
        // DS4's decode graph fuses q/kv norm with the compact-cache write for
        // one row. Prefill and MTP verification use separate KV normalization
        // and cache-store kernels, whose reduction order is observable at the
        // F16 cache boundary.
        let batch_qkv_norm = if b > 1 && t == 1 {
            lisa_mlx::models::glm53::kernels::glm53_dsa_qkv_rms_norm_rows(
                &query_rank_raw,
                &self.q_a_norm,
                &c_raw,
                &self.kv_a_norm,
                self.norm_eps,
                &lisa_mlx::Stream::thread_local_or_default(),
            )
        } else {
            None
        };
        let (c, qr) = if let Some((qr, c)) = batch_qkv_norm {
            (Some(c), qr)
        } else if b == 1 && t == 1 && (capture.is_none() || capture_production_qkv) {
            if b == 1 {
                if let Some(tape) = compact_tape {
                    let (compact_cache, start) = tape
                        .reserve_compact_f16_append(b as i32, t, 512)
                        .map_err(|e| anyhow::anyhow!("glm53 compact cache reserve: {e}"))?;
                    let direct_qr = {
                        let _trace = lisa_mlx::trace::span_detail(
                            "glm53.dsa.qkv_norm_compact_into",
                            layer as u64,
                        );
                        lisa_mlx::models::glm53::kernels::glm53_qkv_norm_compact_into(
                            &query_rank_raw,
                            &self.q_a_norm,
                            &c_raw,
                            &self.kv_a_norm,
                            &compact_cache,
                            start,
                            self.norm_eps,
                            &lisa_mlx::Stream::thread_local_or_default(),
                        )
                    };
                    if let Some(qr) = direct_qr {
                        compact_written = Some((start, t));
                        (None, qr)
                    } else if let Some((qr, compact)) =
                        lisa_mlx::models::glm53::kernels::glm53_qkv_norm_compact(
                            &query_rank_raw,
                            &self.q_a_norm,
                            &c_raw,
                            &self.kv_a_norm,
                            self.norm_eps,
                            &lisa_mlx::Stream::thread_local_or_default(),
                        )
                    {
                        (Some(compact), qr)
                    } else {
                        (
                            Some(glm53_kv_lora_rms_norm(
                                &c_raw,
                                &self.kv_a_norm,
                                self.norm_eps,
                            )?),
                            glm53_weighted_rms_norm(
                                &query_rank_raw,
                                &self.q_a_norm,
                                self.norm_eps,
                            )?,
                        )
                    }
                } else if let Some((qr, compact)) =
                    lisa_mlx::models::glm53::kernels::glm53_qkv_norm_compact(
                        &query_rank_raw,
                        &self.q_a_norm,
                        &c_raw,
                        &self.kv_a_norm,
                        self.norm_eps,
                        &lisa_mlx::Stream::thread_local_or_default(),
                    )
                {
                    (Some(compact), qr)
                } else {
                    (
                        Some(glm53_kv_lora_rms_norm(
                            &c_raw,
                            &self.kv_a_norm,
                            self.norm_eps,
                        )?),
                        glm53_weighted_rms_norm(&query_rank_raw, &self.q_a_norm, self.norm_eps)?,
                    )
                }
            } else if let Some((qr, compact)) =
                lisa_mlx::models::glm53::kernels::glm53_qkv_norm_compact(
                    &query_rank_raw,
                    &self.q_a_norm,
                    &c_raw,
                    &self.kv_a_norm,
                    self.norm_eps,
                    &lisa_mlx::Stream::thread_local_or_default(),
                )
            {
                (Some(compact), qr)
            } else {
                (
                    Some(glm53_kv_lora_rms_norm(
                        &c_raw,
                        &self.kv_a_norm,
                        self.norm_eps,
                    )?),
                    glm53_weighted_rms_norm(&query_rank_raw, &self.q_a_norm, self.norm_eps)?,
                )
            }
        } else {
            (
                Some(glm53_kv_lora_rms_norm(
                    &c_raw,
                    &self.kv_a_norm,
                    self.norm_eps,
                )?),
                glm53_weighted_rms_norm(&query_rank_raw, &self.q_a_norm, self.norm_eps)?,
            )
        };
        #[cfg(all(test, feature = "glm53-fixtures"))]
        if let Some(path) = std::env::var_os("LISA_GLM53_DSA_KVA_VERIFY_CAPTURE_DIR") {
            if capture.is_some() && ((b == 1 && t == 2) || (b == 2 && t == 1)) {
                let layer_path = std::path::Path::new(&path).join(format!("layer-{layer}"));
                let norm_for_capture = match c.as_ref() {
                    Some(norm) => norm.clone(),
                    None => glm53_kv_lora_rms_norm(&c_raw, &self.kv_a_norm, self.norm_eps)?,
                };
                capture_dsa_verify_kv_a_inputs(
                    &layer_path,
                    self,
                    layer,
                    compact_position.unwrap_or(0),
                    x,
                    &c_raw,
                    Some(&norm_for_capture),
                )?;
            }
        }
        if let Some(capture) = capture {
            capture.store(&format!("L{layer}.dsa_kv_a_raw"), &c_raw);
            if let Some(c) = &c {
                capture.store(&format!("L{layer}.dsa_lat"), c);
            }
            capture.store(&format!("L{layer}.dsa_qr"), &qr);
        }
        let q = self.q_b.forward_glm53_prefill(&qr)?; // [B,T,64*256]
        if let Some(c) = cap {
            if c.wants(layer) {
                c.store(&format!("L{layer}.dsa_q"), &q);
            }
        }
        let q = q
            .reshape(&[(b) as i32, (t) as i32, (heads) as i32, (qd) as i32])?
            .transpose_axes(&[0, 2, 1, 3])?
            .contiguous()?; // [B,64,T,256]

        // Store [ik | gs] as F32 and the compact KV latent as F16. Both tapes
        // grow in place and share the same absolute token positions.
        let (ik, gs) =
            self.indexer
                .project_pair(&x, &self.indexer.wk_w, &x, &self.indexer.gate_w)?;
        // Normalize the raw key while pooling; `gs` is the pool gate.
        if let Some(capture) = cap.filter(|capture| capture.wants(layer)) {
            capture.store(&format!("L{layer}.dsa_indexer_raw_k"), &ik);
            capture.store(&format!("L{layer}.dsa_indexer_pool_gate"), &gs);
        }
        Ok(DsaProjections {
            latent: c,
            compact_written,
            query_rank: qr,
            query: q,
            #[cfg(test)]
            kv_raw: c_raw,
            indexer_key: ik,
            indexer_gate: gs,
        })
    }

    /// Forward `x [B,T,H]`. The indexer tape stores F32 key/gate features,
    /// while compact normalized KV latents are held in a separate F16 buffer.
    pub fn forward(
        &self,
        x: &Array,
        cache: &mut FullAttentionCache,
        cap: Option<&crate::core::capture::Capture>,
        layer: usize,
        prefill_mode: bool,
    ) -> anyhow::Result<Array> {
        let projected = {
            let _trace = lisa_mlx::trace::span_detail("glm53.dsa.input_projections", layer as u64);
            self.project_inputs(x, cap, layer, Some(&mut cache.indexer_tape))?
        };
        let pre_o = self.forward_projected(x, cache, cap, layer, &projected, None, prefill_mode)?;
        let out = {
            let _trace = lisa_mlx::trace::span_detail("glm53.dsa.output_projection", layer as u64);
            self.output_project(&pre_o)?
        };
        if let Some(capture) = cap.filter(|capture| capture.wants(layer)) {
            capture.store(&format!("L{layer}.dsa_output_project"), &out);
        }
        Ok(out)
    }

    /// Native mixed-position batch path: batch all DSA input projections and
    /// the output projection, but append/attend against each row's own tape.
    pub fn forward_ragged(
        &self,
        x: &Array,
        caches: &mut [&mut FullAttentionCache],
        cap: Option<&crate::core::capture::Capture>,
        layer: usize,
    ) -> anyhow::Result<Array> {
        let b = caches.len();
        anyhow::ensure!(
            b > 1 && x.dim(0) == b as i32 && x.dim(1) == 1,
            "glm53 DSA ragged decode expects B>1, one token per row, got {:?} for {b} caches",
            x.shape()
        );
        let projected = self.project_inputs(x, cap, layer, None)?;
        let mut pre_o_rows = Vec::with_capacity(b);
        for (row, cache) in caches.iter_mut().enumerate() {
            let x_row = narrow(x, 0, row, 1)?;
            let row_projected = DsaProjections {
                latent: Some(narrow(
                    projected
                        .latent
                        .as_ref()
                        .expect("ragged projection retains compact latent"),
                    0,
                    row,
                    1,
                )?),
                compact_written: None,
                query_rank: narrow(&projected.query_rank, 0, row, 1)?,
                query: narrow(&projected.query, 0, row, 1)?,
                #[cfg(test)]
                kv_raw: narrow(&projected.kv_raw, 0, row, 1)?,
                indexer_key: narrow(&projected.indexer_key, 0, row, 1)?,
                indexer_gate: narrow(&projected.indexer_gate, 0, row, 1)?,
            };
            let pre_o = self.forward_projected(
                &x_row,
                cache,
                cap,
                layer,
                &row_projected,
                Some(row),
                false,
            )?;
            pre_o_rows.push(pre_o);
        }
        let pre_o_refs: Vec<&Array> = pre_o_rows.iter().collect();
        let pre_o_batch = Array::concatenate(&pre_o_refs, 0)?;
        let output = {
            let _trace = lisa_mlx::trace::span_detail("glm53.dsa.output_projection", layer as u64);
            self.output_project(&pre_o_batch)?
        };
        #[cfg(all(test, feature = "glm53-fixtures"))]
        if cap.is_some_and(|capture| capture.wants(layer))
            && let Some(root) = std::env::var_os("LISA_GLM53_DSA_NATIVE_BATCH_O_CAPTURE_DIR")
        {
            let path = std::path::Path::new(&root).join(format!("layer-{layer}"));
            capture_dsa_native_batch_o_project_inputs(&path, self, &pre_o_batch, &output)?;
        }
        if let Some(capture) = cap.filter(|capture| capture.wants(layer)) {
            capture.store(&format!("L{layer}.dsa_output_project"), &output);
        }
        Ok(output)
    }

    pub(super) fn append_projected_cache(
        &self,
        projected: &DsaProjections,
        cache: &mut FullAttentionCache,
        layer: usize,
        capture: Option<&crate::core::capture::Capture>,
    ) -> anyhow::Result<()> {
        let _trace = lisa_mlx::trace::span_detail("glm53.dsa.cache_append", layer as u64);
        self.append_cache_projection_rows(
            &projected.indexer_key,
            &projected.indexer_gate,
            projected.latent.as_ref(),
            projected.compact_written,
            cache,
        )?;
        self.indexer
            .update_pooled_cache(&mut cache.indexer_tape, capture, layer)
    }

    fn append_cache_projection_rows(
        &self,
        indexer_key: &Array,
        indexer_gate: &Array,
        latent: Option<&Array>,
        compact_written: Option<(usize, usize)>,
        cache: &mut FullAttentionCache,
    ) -> anyhow::Result<()> {
        cache
            .indexer_tape
            .append_fields(indexer_key, indexer_gate, self.indexer.kpool)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        if let Some((start, steps)) = compact_written {
            cache
                .indexer_tape
                .commit_compact_f16_append(start, steps)
                .map_err(|e| anyhow::anyhow!("{e}"))?;
        } else {
            cache
                .indexer_tape
                .append_compact_f16(latent.expect("fallback projection has compact latent"))
                .map_err(|e| anyhow::anyhow!("{e}"))?;
        }
        let total = cache.indexer_tape.total;
        cache.offset = total;
        // GLM stores compact latents separately from expanded attention KV.
        cache.keys = None;
        cache.values = None;
        Ok(())
    }

    #[cfg(all(test, feature = "glm53-fixtures"))]
    pub(super) fn project_prefill_cache_for_capture(
        &self,
        x: &Array,
        cache: &mut FullAttentionCache,
        layer: usize,
    ) -> anyhow::Result<(Array, Array, Array)> {
        let _trace = lisa_mlx::trace::span_detail("glm53.dsa.input_projections", layer as u64);
        let kv_raw = self.kv_a.forward_glm53_prefill(x)?;
        let latent = glm53_kv_lora_rms_norm(&kv_raw, &self.kv_a_norm, self.norm_eps)?;
        let (indexer_key, indexer_gate) =
            self.indexer
                .project_pair(x, &self.indexer.wk_w, x, &self.indexer.gate_w)?;
        self.append_cache_projection_rows(&indexer_key, &indexer_gate, Some(&latent), None, cache)?;
        self.indexer
            .update_pooled_cache(&mut cache.indexer_tape, None, layer)?;
        Ok((kv_raw, indexer_key, indexer_gate))
    }

    pub(super) fn select_projected_ids(
        &self,
        x: &Array,
        cache: &mut FullAttentionCache,
        projected: &DsaProjections,
        capture: Option<&crate::core::capture::Capture>,
        layer: usize,
        prefill_mode: bool,
    ) -> anyhow::Result<(Array, usize)> {
        self.indexer.select_ids_gpu(
            &mut cache.indexer_tape,
            x,
            &projected.query_rank,
            cache.offset,
            x.dim(1) as usize,
            capture,
            layer,
            prefill_mode,
        )
    }

    fn forward_projected(
        &self,
        x: &Array,
        cache: &mut FullAttentionCache,
        cap: Option<&crate::core::capture::Capture>,
        layer: usize,
        projected: &DsaProjections,
        native_batch_row: Option<usize>,
        prefill_mode: bool,
    ) -> anyhow::Result<Array> {
        let (b, t) = (x.dim(0) as usize, x.dim(1) as usize);
        let heads = self.heads;
        let qd = self.q_head_dim;
        let vd = self.v_head_dim;
        let scale = (qd as f64).powf(-0.5) as f32;

        let qr = &projected.query_rank;
        let q = &projected.query;

        self.append_projected_cache(projected, cache, layer, cap)?;
        let tc = cache.indexer_tape.total;
        let dense_limit = self.indexer.topk + self.indexer.kpool.saturating_sub(1);
        let _value_slices = (prefill_mode && b == 1 && native_batch_row.is_none()).then(|| {
            lisa_mlx::models::glm53::kernels::Glm53DsaValueSliceScope::new(tc - t, dense_limit)
        });
        let compact_latents = cache
            .indexer_tape
            .compact_slice(0, tc)
            .ok_or_else(|| anyhow::anyhow!("glm53 dsa: compact latent cache missing"))?;
        if let Some(capture) = cap.filter(|capture| capture.wants(layer)) {
            let current = narrow(&compact_latents, 1, tc - t, t)?.contiguous()?;
            capture.store(&format!("L{layer}.dsa_cache_latent"), &current);
        }

        // DS4 uses its raw FlashAttention kernel for dense-causal prefill
        // slices of at least 24 rows. Keep that path distinct from indexed
        // attention: its F16 query staging and tiled softmax are observable
        // in the layer output.
        if native_batch_row.is_none() && b == 1 && t >= 24 && tc <= dense_limit {
            let query_start = tc - t;
            let stream = lisa_mlx::Stream::thread_local_or_default();
            let (attended, q_low, lora) =
                lisa_mlx::models::glm53::kernels::dsa_absorbed_attention_flash_causal_prefill_stages(
                    q,
                    &self.kv_b.weight,
                    &self.kv_b.scales,
                    &self.kv_b.biases,
                    self.kv_b.group_size as usize,
                    &compact_latents,
                    query_start,
                    scale,
                    &stream,
                    prefill_mode,
                )
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "glm53 dsa: DS4 dense-causal FlashAttention declined at rows {t}, context {tc}"
                    )
                })?;
            let out = attended
                .transpose_axes(&[0, 2, 1, 3])?
                .contiguous()?
                .reshape(&[b as i32, t as i32, (heads * vd) as i32])?;
            if let Some(capture) = cap.filter(|capture| capture.wants(layer)) {
                capture.store(&format!("L{layer}.dsa_q_low"), &q_low);
                capture.store(&format!("L{layer}.dsa_attn_lora"), &lora);
                capture.store(&format!("L{layer}.dsa_pre_o"), &out);
                let tail = t.saturating_sub(4);
                if let Ok(q_tail) = narrow(&q_low, 1, tail, t.min(4)) {
                    capture.store(&format!("L{layer}.dsa_q_low_boundary"), &q_tail);
                }
                if let Ok(lora_tail) = narrow(&lora, 1, tail, t.min(4)) {
                    capture.store(&format!("L{layer}.dsa_attn_lora_boundary"), &lora_tail);
                }
            }
            return Ok(out);
        }
        #[cfg(all(test, feature = "glm53-fixtures"))]
        if let Some(path) = std::env::var_os("LISA_GLM53_DSA_KVA_VERIFY_CAPTURE_DIR") {
            if b == 1 && t == 2 && cap.is_some_and(|c| c.wants(layer)) {
                let layer_path = std::path::Path::new(&path).join(format!("layer-{layer}"));
                let appended = narrow(&compact_latents, 1, tc - t, t)?.contiguous()?;
                capture_dsa_verify_compact_cache(
                    &layer_path,
                    layer,
                    tc.saturating_sub(t),
                    &appended,
                )?;
            }
        }

        // DS4 uses indexed decode even below the sparse crossover: it fills
        // a contiguous [0, tc) selection there. Keep that explicit prefix path
        // so affine-MLX attention uses the same reduction schedule.
        let selected_absorbed = if t > 1 && tc <= dense_limit {
            // Each verification row sees the committed prefix through its own
            // absolute position. Pad the shared-width selection with invalid
            // IDs so the indexed-lora kernel can process the rows together.
            let query_start = tc.saturating_sub(t);
            let mut ids = Vec::with_capacity(b * t * tc);
            for _batch in 0..b {
                for query in 0..t {
                    let last = query_start + query;
                    ids.extend((0..tc).map(|position| {
                        if position <= last {
                            position as i32
                        } else {
                            -1
                        }
                    }));
                }
            }
            Some((Array::from_slice(&ids, &[b as i32, t as i32, tc as i32]), 0))
        } else if t == 1 || tc > dense_limit {
            Some(if tc <= dense_limit {
                let ids: Vec<i32> = (0..tc as i32).collect();
                (
                    Array::from_slice(&ids, &[1, 1, tc as i32])
                        .broadcast_to(&[b as i32, t as i32, tc as i32])?,
                    tc,
                )
            } else {
                let _trace =
                    lisa_mlx::trace::span_detail("glm53.dsa.indexer_selection", layer as u64);
                self.select_projected_ids(x, cache, projected, cap, layer, prefill_mode)?
            })
        } else {
            None
        };
        if let Some((selected, selected_valid_prefix)) = selected_absorbed.as_ref() {
            let stream = lisa_mlx::Stream::thread_local_or_default();
            let _trace_absorbed =
                lisa_mlx::trace::span_detail("glm53.dsa.absorbed_attention", layer as u64);
            let absorbed = if let Some(capture) = cap.filter(|capture| capture.wants(layer)) {
                lisa_mlx::models::glm53::kernels::dsa_absorbed_attention_stages(
                    &q,
                    &self.kv_b.weight,
                    &self.kv_b.scales,
                    &self.kv_b.biases,
                    self.kv_b.group_size as usize,
                    &compact_latents,
                    selected,
                    scale,
                    &stream,
                    prefill_mode,
                )
                .map(|(attended, q_low, lora)| {
                    if let Some(capture) = cap.filter(|capture| capture.wants(layer)) {
                        capture.store(&format!("L{layer}.dsa_q_low"), &q_low);
                        capture.store(&format!("L{layer}.dsa_attn_lora"), &lora);
                    }
                    #[cfg(all(test, feature = "glm53-fixtures"))]
                    if let (Some(row), Some(root)) = (
                        native_batch_row,
                        std::env::var_os("LISA_GLM53_DSA_NATIVE_ATTN_CAPTURE_DIR"),
                    ) {
                        if cap.is_some_and(|capture| capture.wants(layer)) {
                            let pre_o = attended
                                .transpose_axes(&[0, 2, 1, 3])
                                .and_then(|value| value.contiguous())
                                .and_then(|value| {
                                    value.reshape(&[b as i32, t as i32, (heads * vd) as i32])
                                })
                                .expect("reshape captured native DSA pre-output");
                            let (fused_attended, fused_lora) =
                                lisa_mlx::models::glm53::kernels::dsa_absorbed_attention_decode_capture(
                                    &q,
                                    &self.kv_b.weight,
                                    &self.kv_b.scales,
                                    &self.kv_b.biases,
                                    self.kv_b.group_size as usize,
                                    &compact_latents,
                                    Some(selected),
                                    *selected_valid_prefix,
                                    scale,
                                    &stream,
                                )
                                .expect("native DS4-scheduled fused DSA decode dispatches");
                            let fused_pre_o = fused_attended
                                .transpose_axes(&[0, 2, 1, 3])
                                .and_then(|value| value.contiguous())
                                .and_then(|value| {
                                    value.reshape(&[b as i32, t as i32, (heads * vd) as i32])
                                })
                                .expect("reshape captured fused native DSA pre-output");
                            let path = std::path::Path::new(&root)
                                .join(format!("layer-{layer}"))
                                .join(format!("row-{row}"));
                            capture_dsa_native_decode_attention_inputs(
                                &path,
                                self,
                                &q,
                                &compact_latents,
                                selected,
                                &q_low,
                                &lora,
                                &fused_lora,
                                &pre_o,
                                &fused_pre_o,
                            )
                            .expect("write native DSA attention stages for DS4 replay");
                        }
                    }
                    #[cfg(all(test, feature = "glm53-fixtures"))]
                    if b == 1
                        && t == 2
                        && std::env::var_os("LISA_GLM53_DSA_VERIFY_DS4_CAPTURE_DIR").is_some()
                        && cap.is_some_and(|capture| capture.wants(layer))
                    {
                        let path = std::env::var_os("LISA_GLM53_DSA_VERIFY_DS4_CAPTURE_DIR")
                            .expect("DSA verify capture directory");
                        let layer_path = std::path::Path::new(&path).join(format!("layer-{layer}"));
                        capture_dsa_verify_latent_stages(&layer_path, &q_low, &lora).expect(
                            "write real DSA verify latent-stage capture before fixture checks",
                        );
                        let pre_o = attended
                            .transpose_axes(&[0, 2, 1, 3])
                            .and_then(|value| value.contiguous())
                            .and_then(|value| {
                                value.reshape(&[b as i32, t as i32, (heads * vd) as i32])
                            })
                            .expect("reshape captured DSA verify pre-output");
                        capture_dsa_verify_attention_inputs(
                            &layer_path,
                            self,
                            layer,
                            &q,
                            &compact_latents,
                            &pre_o,
                            tc - t,
                        )
                        .expect("write real DSA verify attention capture before fixture checks");
                    }
                    #[cfg(all(test, feature = "glm53-fixtures"))]
                    if b == 1
                        && t == 2
                        && tc == 8
                        && tc - t == 6
                        && heads == 64
                        && vd == 256
                        && std::env::var_os("LISA_GLM53_DSA_KVA_VERIFY_CAPTURE_DIR").is_some()
                    {
                        let ds4_fixture = ds4_dsa_verify_stages_fixture(layer)
                            .expect("direct-DS4 target-MTP absorbed-attention fixture");
                        let q_values = q_low
                            .to_vec1::<f32>()
                            .expect("real target-MTP QK-lowrank fixture readback");
                        let lora_values = lora
                            .to_vec1::<f32>()
                            .expect("real target-MTP indexed-lora fixture readback");
                        let q_bytes = q_values.len() * std::mem::size_of::<f32>();
                        let lora_bytes = lora_values.len() * std::mem::size_of::<f32>();
                        let output_bytes = b * t * heads * vd * std::mem::size_of::<f32>();
                        assert_eq!(ds4_fixture.len(), q_bytes + lora_bytes + output_bytes);
                        let matches_fixture = |values: &[f32], bytes: &[u8]| {
                            values
                                .iter()
                                .map(|x| x.to_bits())
                                .eq(bytes.chunks_exact(4).map(|chunk| {
                                    f32::from_le_bytes(chunk.try_into().expect("F32 fixture word"))
                                        .to_bits()
                                }))
                        };
                        assert!(
                            matches_fixture(&q_values, &ds4_fixture[..q_bytes]),
                            "real layer-{layer} target-MTP DSA QK-lowrank differs from DS4"
                        );
                        assert!(
                            matches_fixture(
                                &lora_values,
                                &ds4_fixture[q_bytes..q_bytes + lora_bytes]
                            ),
                            "real layer-{layer} target-MTP DSA indexed-lora differs from DS4"
                        );
                    }
                    let query_rows = q_low.dim(1) as usize;
                    #[cfg(all(test, feature = "glm53-fixtures"))]
                    {
                        if let Some(path) =
                            std::env::var_os("LISA_GLM53_DSA_VERIFY_DS4_CAPTURE_DIR")
                        {
                            let layer_path =
                                std::path::Path::new(&path).join(format!("layer-{layer}"));
                            capture_dsa_verify_latent_stages(&layer_path, &q_low, &lora)
                                .expect("write real DSA verify latent-stage capture");
                        }
                    }
                    let tail = query_rows.saturating_sub(4);
                    let tail_rows = query_rows.min(4);
                    if let Ok(q_tail) = narrow(&q_low, 1, tail, tail_rows) {
                        capture.store(&format!("L{layer}.dsa_q_low_boundary"), &q_tail);
                    }
                    if let Ok(lora_tail) = narrow(&lora, 1, tail, tail_rows) {
                        capture.store(&format!("L{layer}.dsa_attn_lora_boundary"), &lora_tail);
                    }
                    attended
                })
            } else {
                // DS4 keeps MLX affine V-B on the split attention-lora plus
                // value-projection path; its fused indexed decode is only
                // used for native value-weight representations.
                lisa_mlx::models::glm53::kernels::dsa_absorbed_attention_with_valid_prefix(
                    &q,
                    &self.kv_b.weight,
                    &self.kv_b.scales,
                    &self.kv_b.biases,
                    self.kv_b.group_size as usize,
                    &compact_latents,
                    selected,
                    *selected_valid_prefix,
                    scale,
                    &stream,
                    prefill_mode,
                )
            };
            if let Some(attended) = absorbed {
                let out = attended
                    .transpose_axes(&[0, 2, 1, 3])?
                    .contiguous()?
                    .reshape(&[b as i32, t as i32, (heads * vd) as i32])?;
                #[cfg(all(test, feature = "glm53-fixtures"))]
                if let Some(path) = std::env::var_os("LISA_GLM53_DSA_VERIFY_DS4_CAPTURE_DIR") {
                    if cap.is_some_and(|capture| capture.wants(layer)) && t > 1 && tc <= dense_limit
                    {
                        let layer_path = std::path::Path::new(&path).join(format!("layer-{layer}"));
                        capture_dsa_verify_attention_inputs(
                            &layer_path,
                            self,
                            layer,
                            &q,
                            &compact_latents,
                            &out,
                            tc - t,
                        )?;
                    }
                }
                if let Some(capture) = cap {
                    if capture.wants(layer) {
                        capture.store(&format!("L{layer}.dsa_pre_o"), &out);
                    }
                }
                return Ok(out);
            }
            static WARNED_ABSORBED_FALLBACK: std::sync::atomic::AtomicBool =
                std::sync::atomic::AtomicBool::new(false);
            if !WARNED_ABSORBED_FALLBACK.swap(true, std::sync::atomic::Ordering::Relaxed) {
                eprintln!(
                    "glm53 dsa: absorbed attention declined; dense fallback is allowed only within the dense-equivalent context limit"
                );
            }
            // The dense fallback materializes kv_b for every cached token. At
            // long context this turns one decode into a full-context projection
            // and hides a broken/declined fast kernel behind a severe slowdown.
            // Fail with the recorded Metal diagnostic instead of pretending the
            // ds4 indexed path is active.
            anyhow::ensure!(
                tc <= dense_limit,
                "glm53 dsa: indexed absorbed attention declined at context {tc} (selection width {}); refusing the quadratic dense fallback",
                selected.dim(2)
            );
        }

        let pool_ops = std::env::var("LISA_GLM53_POOL_OPS")
            .map(|v| v == "1")
            .unwrap_or(false);
        let keep = if t == 1 && tc > dense_limit && !pool_ops {
            let _trace = lisa_mlx::trace::span_detail("glm53.dsa.indexer_keep_mask", layer as u64);
            self.indexer.select_keep_gpu(
                &mut cache.indexer_tape,
                x,
                &qr,
                cache.offset,
                t,
                tc,
                layer,
            )?
        } else if let Some((selected, _)) = selected_absorbed.as_ref() {
            keep_mask_from_selection(x, selected, t, tc)?
        } else if tc <= dense_limit {
            causal_mask(x, b, t, tc)?
        } else {
            let sel = {
                let _trace =
                    lisa_mlx::trace::span_detail("glm53.dsa.indexer_host_selection", layer as u64);
                self.indexer.select(
                    x,
                    &mut cache.indexer_tape,
                    &qr,
                    cache.offset,
                    t,
                    tc,
                    prefill_mode,
                )?
            };
            keep_mask_from_selection(x, &sel, t, tc)?
        };

        // Prefill and unsupported decode hardware retain the dense reference
        // path. It is intentionally outside the common one-token decode path.
        let kv = {
            let _trace =
                lisa_mlx::trace::span_detail("glm53.dsa.dense_kv_projection", layer as u64);
            self.kv_b
                .forward(&compact_latents.as_dtype(Dtype::Float32)?)?
        };
        let kv = kv.reshape(&[b as i32, tc as i32, heads as i32, (qd + vd) as i32])?;
        let k = narrow(&kv, 3, 0, qd)?
            .transpose_axes(&[0, 2, 1, 3])?
            .contiguous()?;
        let v = narrow(&kv, 3, qd, vd)?
            .transpose_axes(&[0, 2, 1, 3])?
            .contiguous()?;
        if let Some(c) = cap {
            if c.wants(layer) {
                c.store(&format!("L{layer}.dsa_k"), &k);
                c.store(&format!("L{layer}.dsa_v"), &v);
            }
        }

        if t == 1 {
            let step = super::tower::INC_STEP.load(std::sync::atomic::Ordering::Relaxed);
            if let Some(c2) = cap {
                if c2.wants(layer) {
                    let _ = k.eval_enqueue();
                    let _ = v.eval_enqueue();
                    c2.store(&format!("L{layer}.dec_dsa_k_s{step}"), &k);
                    c2.store(&format!("L{layer}.dec_dsa_v_s{step}"), &v);
                }
            }
        }
        // scores = q @ k^T * scale + keep, softmax, @ v — per batch (B=1 v1).
        // ds4 flash_attn decode parity (t == 1): the attention runs as TWO
        // fused dispatches (scores, then softmax @ v) over the full cached
        // K/V with the keep mask baked in. Both stages preserve the ds4 F32
        // graph. Op path remains the fallback
        // (LISA_GLM53_ATTN_OPS=1 forces it).
        let mut out_acc: Option<Array> = None;
        let fused = if t == 1
            && !std::env::var("LISA_GLM53_ATTN_OPS")
                .map(|v| v == "1")
                .unwrap_or(false)
        {
            let stream = lisa_mlx::Stream::thread_local_or_default();
            let q2 = q.reshape(&[(b) as i32, (heads) as i32, (qd) as i32])?;
            let k_c = k.contiguous()?;
            let v_c = v.contiguous()?;
            let keep2 = keep.reshape(&[(b) as i32, (tc) as i32])?;
            let _trace =
                lisa_mlx::trace::span_detail("glm53.dsa.fused_decode_attention", layer as u64);
            lisa_mlx::models::glm53::kernels::dsa_decode_scores(&q2, &k_c, &keep2, scale, &stream)
                .and_then(|sc| {
                    lisa_mlx::models::glm53::kernels::dsa_decode_weighted_v(&sc, &v_c, &stream)
                })
        } else {
            None
        };
        if let Some(flat) = fused {
            // [B*heads, vd] == the per-batch concat the op path builds at t == 1.
            out_acc = Some(flat);
        } else {
            let kt = k.transpose_axes(&[0, 1, 3, 2])?.contiguous()?; // [B,64,256,Tc]
            let scale_a = ops::full::<i32>(&[1], &Array::from_f32_as(scale, Dtype::Float32))?;
            for bi in 0..b {
                let qb =
                    narrow(&q, 0, bi, 1)?.reshape(&[(heads) as i32, (t) as i32, (qd) as i32])?;
                // kt is already [B, heads, qd, tc] — reshape straight to the
                // [heads, qd, tc] matmul operand (no re-permute).
                let kb_t =
                    narrow(&kt, 0, bi, 1)?.reshape(&[(heads) as i32, (qd) as i32, (tc) as i32])?;
                if let Some(cp) = cap {
                    if cp.wants(layer) {
                        cp.store(&format!("L{layer}.dsa_qb"), &qb);
                        cp.store(&format!("L{layer}.dsa_kb_t"), &kb_t);
                    }
                }
                let vb =
                    narrow(&v, 0, bi, 1)?.reshape(&[(heads) as i32, (tc) as i32, (vd) as i32])?;
                let sc = qb.matmul(&kb_t)?; // [64,T,Tc]
                let sc = sc.multiply(&scale_a)?;
                let keep_b = narrow(&keep, 0, bi, 1)?; // [1,1,T,Tc] broadcasts over heads
                let sc = sc
                    .add(&keep_b)?
                    .reshape(&[(heads) as i32, (t) as i32, (tc) as i32])?; // back to rank 3
                let probs_b = sc.softmax_axis(-1)?;
                if let Some(cp) = cap {
                    if cp.wants(layer) {
                        cp.store(&format!("L{layer}.dsa_sc"), &sc);
                        cp.store(&format!("L{layer}.dsa_probs"), &probs_b);
                        cp.store(&format!("L{layer}.dsa_vb"), &vb);
                    }
                }
                let ob = probs_b.matmul(&vb)?; // [64,T,256]
                let ob = ob.reshape(&[(heads * t * vd) as i32])?;
                out_acc = Some(match out_acc {
                    Some(a) => Array::concatenate(&[&a, &ob], 0)?,
                    None => ob,
                });
            }
        }
        let out = out_acc.expect("b>0").reshape(&[
            (b) as i32,
            (heads) as i32,
            (t) as i32,
            (vd) as i32,
        ])?;
        let out = out.transpose_axes(&[0, 2, 1, 3])?.reshape(&[
            (b) as i32,
            (t) as i32,
            (heads * vd) as i32,
        ])?;
        #[cfg(all(test, feature = "glm53-fixtures"))]
        if let Some(path) = std::env::var_os("LISA_GLM53_DSA_VERIFY_DS4_CAPTURE_DIR") {
            if cap.is_some_and(|capture| capture.wants(layer)) && t > 1 && tc <= dense_limit {
                let layer_path = std::path::Path::new(&path).join(format!("layer-{layer}"));
                capture_dsa_verify_attention_inputs(
                    &layer_path,
                    self,
                    layer,
                    q,
                    &compact_latents,
                    &out,
                    tc - t,
                )?;
            }
        }
        if let Some(c) = cap {
            if c.wants(layer) {
                c.store(&format!("L{layer}.dsa_pre_o"), &out);
            }
        }
        if t == 1 {
            if let Some(c2) = cap {
                if c2.wants(layer) {
                    let _ = out.eval_enqueue();
                    c2.store(&format!("L{layer}.dec_dsa_pre_o"), &out);
                }
            }
        }
        Ok(out)
    }
}

impl Indexer {
    /// Run a plain `[out,in]` indexer projection with ds4's BF16 dispatch:
    /// row GEMV for up to eight rows, tiled GEMM for larger prefill chunks.
    fn project(&self, x: &Array, weight: &Array) -> anyhow::Result<Array> {
        anyhow::ensure!(
            x.rank() >= 2 && weight.rank() == 2,
            "invalid indexer projection rank"
        );
        let input_dim = x.dim((x.rank() - 1) as i32) as usize;
        anyhow::ensure!(
            input_dim == weight.dim(1) as usize,
            "indexer projection K mismatch"
        );
        let mut output_shape = x.shape().to_vec();
        let last = output_shape.len() - 1;
        let rows = output_shape[..last]
            .iter()
            .try_fold(1usize, |acc, &d| acc.checked_mul(d as usize))
            .ok_or_else(|| anyhow::anyhow!("indexer projection row count overflow"))?;
        output_shape[last] = weight.dim(0);

        if weight.dtype() == Dtype::Bfloat16 && rows > 0 {
            let input = x.reshape(&[rows as i32, input_dim as i32])?;
            let stream = lisa_mlx::Stream::thread_local_or_default();
            let projected = if rows <= 8 {
                lisa_mlx::models::glm53::kernels::glm53_bf16_gemv(weight, &input, &stream)
            } else {
                lisa_mlx::models::glm53::kernels::glm53_bf16_gemm(weight, &input, &stream)
            }
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "GLM indexer BF16 projection failed for {rows} rows and {input_dim} inputs"
                )
            })?;
            return Ok(projected.reshape(&output_shape)?);
        }

        let transposed = weight.transpose_axes(&[1, 0])?.as_dtype(Dtype::Float32)?;
        Ok(mm(x, &transposed)?)
    }

    /// Pair short-batch indexer BF16 projections in one dispatch. Every output
    /// row keeps the DS4 GEMV reduction order even when the two projections
    /// use different input and output widths.
    fn project_pair(
        &self,
        x_a: &Array,
        weight_a: &Array,
        x_b: &Array,
        weight_b: &Array,
    ) -> anyhow::Result<(Array, Array)> {
        let rows_a = x_a.shape()[..x_a.rank() - 1]
            .iter()
            .try_fold(1usize, |acc, &d| acc.checked_mul(d as usize))
            .ok_or_else(|| anyhow::anyhow!("indexer projection row count overflow"))?;
        let rows_b = x_b.shape()[..x_b.rank() - 1]
            .iter()
            .try_fold(1usize, |acc, &d| acc.checked_mul(d as usize))
            .ok_or_else(|| anyhow::anyhow!("indexer projection row count overflow"))?;
        if rows_a > 0
            && rows_a == rows_b
            && rows_a <= 8
            && weight_a.dtype() == Dtype::Bfloat16
            && weight_b.dtype() == Dtype::Bfloat16
        {
            anyhow::ensure!(weight_a.rank() == 2 && weight_b.rank() == 2);
            let input_dim_a = x_a.dim((x_a.rank() - 1) as i32) as usize;
            let input_dim_b = x_b.dim((x_b.rank() - 1) as i32) as usize;
            anyhow::ensure!(
                weight_a.dim(1) as usize == input_dim_a && weight_b.dim(1) as usize == input_dim_b,
                "paired indexer projection K mismatch"
            );
            let input_a = x_a.reshape(&[rows_a as i32, input_dim_a as i32])?;
            let input_b = x_b.reshape(&[rows_b as i32, input_dim_b as i32])?;
            let stream = lisa_mlx::Stream::thread_local_or_default();
            let (a, b) = lisa_mlx::models::glm53::kernels::glm53_bf16_pair_gemv(
                weight_a, &input_a, weight_b, &input_b, &stream,
            )
            .ok_or_else(|| {
                anyhow::anyhow!("GLM indexer paired BF16 projection failed for {rows_a} rows")
            })?;
            let mut shape_a = x_a.shape().to_vec();
            *shape_a.last_mut().expect("rank checked by caller") = weight_a.dim(0);
            let mut shape_b = x_b.shape().to_vec();
            *shape_b.last_mut().expect("rank checked by caller") = weight_b.dim(0);
            return Ok((a.reshape(&shape_a)?, b.reshape(&shape_b)?));
        }
        Ok((self.project(x_a, weight_a)?, self.project(x_b, weight_b)?))
    }

    /// Host-assisted selection: pooling + scoring as Array ops, top-k and
    /// tail on the host. Returns `sel [B, T, width]` i64 selection token
    /// indices (-1 = none).
    /// Host-assisted selection from the indexer tape: the packed
    /// `[ik | gs]` rows live in `tape.raw`; pooling is lazy and incremental
    /// (only newly completed kpool-groups are pooled). Top-k and the tail
    /// run on the host with ABSOLUTE token positions.
    /// Pooling + scoring, GPU-only (shared by the host top-k path and the
    /// decode GPU-mask path): returns `index_sc [B, T, P]` f32.
    fn project_score_inputs(&self, x: &Array, qr: &Array) -> anyhow::Result<(Array, Array)> {
        let _trace = lisa_mlx::trace::span("glm53.dsa.indexer_input_projection");
        let (b, t, heads, hd) = (
            x.dim(0) as usize,
            x.dim(1) as usize,
            self.heads,
            self.head_dim,
        );
        let (iq, weights) = self.project_pair(qr, &self.wq_b_w, x, &self.weights_w)?;
        let iq = iq
            .as_dtype(Dtype::Float32)?
            .reshape(&[b as i32, t as i32, heads as i32, hd as i32])?
            .transpose_axes(&[0, 2, 1, 3])?
            .contiguous()?;
        let weights = weights.as_dtype(Dtype::Float32)?;
        Ok((iq, weights))
    }

    fn update_pooled_cache(
        &self,
        tape: &mut IndexerTape,
        capture: Option<&crate::core::capture::Capture>,
        layer: usize,
    ) -> anyhow::Result<()> {
        let kpool = self.kpool;
        let total = tape.total;
        let p_n = total / kpool;
        if p_n == 0 || tape.pooled_upto >= p_n {
            return Ok(());
        }

        let _trace = lisa_mlx::trace::span_detail("glm53.dsa.pool_update", layer as u64);
        let start = tape.pooled_upto;
        let raw = tape
            .raw_slice(start * kpool, p_n * kpool)
            .ok_or_else(|| anyhow::anyhow!("indexer tape raw missing"))?;
        let (b, hd) = (raw.dim(0) as usize, self.head_dim);
        let raw = narrow(&raw, 2, 0, 2 * hd)?.contiguous()?;
        let n_new = p_n - start;
        let grouped = raw.reshape(&[b as i32, n_new as i32, kpool as i32, (2 * hd) as i32])?;
        let mut fused_done = false;
        if !std::env::var("LISA_GLM53_POOL_OPS")
            .map(|v| v == "1")
            .unwrap_or(false)
        {
            let raw_k = narrow(&grouped, 3, 0, hd)?.contiguous()?;
            let gate = narrow(&grouped, 3, hd, hd)?.contiguous()?;
            let stream = lisa_mlx::Stream::thread_local_or_default();
            if let Some(pool_k) = lisa_mlx::models::glm53::kernels::indexer_pool_fused(
                &raw_k,
                &gate,
                &self.ape,
                &self.k_norm_w,
                &self.k_norm_b,
                LAYER_NORM_EPS,
                &stream,
            ) {
                if let Some(c) = capture.filter(|capture| capture.wants(layer)) {
                    c.store(&format!("L{layer}.dsa_pooled_kernel_output"), &pool_k);
                }
                tape.push_pooled_f16(&pool_k)
                    .map_err(|e| anyhow::anyhow!("{e}"))?;
                fused_done = true;
            }
        }
        if !fused_done {
            let ik_g =
                narrow(&grouped, 3, 0, hd)?.reshape(&[(b * n_new * kpool) as i32, hd as i32])?;
            let ik_g = layer_norm(
                &ik_g.reshape(&[b as i32, (n_new * kpool) as i32, hd as i32])?,
                &self.k_norm_w,
                &self.k_norm_b,
                LAYER_NORM_EPS,
            )?
            .reshape(&[b as i32, n_new as i32, kpool as i32, hd as i32])?;
            let gs_g = narrow(&grouped, 3, hd, hd)?.contiguous()?;
            let ape = self
                .ape
                .as_dtype(Dtype::Float32)?
                .expand_dims(0)?
                .expand_dims(0)?
                .broadcast_to(&[b as i32, n_new as i32, kpool as i32, hd as i32])?;
            // Each channel gets a softmax across the four candidate tokens.
            let probs = gs_g.add(&ape)?.softmax_axis(2)?;
            let mut pool_k: Option<Array> = None;
            for j in 0..kpool {
                let pj = narrow(&probs, 2, j, 1)?;
                let kij = narrow(&ik_g, 2, j, 1)?;
                let term = pj.multiply(&kij)?;
                pool_k = Some(match pool_k {
                    Some(a) => a.add(&term)?,
                    None => term,
                });
            }
            let pool_k =
                pool_k
                    .expect("kpool > 0")
                    .reshape(&[b as i32, n_new as i32, hd as i32])?;
            tape.push_pooled_f16(&pool_k)
                .map_err(|e| anyhow::anyhow!("{e}"))?;
        }
        tape.compact_raw_history(RAW_ROLLBACK_ROWS)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        Ok(())
    }

    fn index_scores(
        &self,
        tape: &mut IndexerTape,
        iq: &Array,
        weights: &Array,
        offset: usize,
        t: usize,
        capture: Option<&crate::core::capture::Capture>,
        layer: usize,
        prefill_mode: bool,
    ) -> anyhow::Result<Array> {
        let (b, heads, hd) = (iq.dim(0) as usize, self.heads, self.head_dim);
        anyhow::ensure!(
            iq.shape() == [b as i32, heads as i32, t as i32, hd as i32]
                && weights.shape() == [b as i32, t as i32, heads as i32],
            "GLM indexer score input shape mismatch: q={:?}, weights={:?}, T={t}",
            iq.shape(),
            weights.shape()
        );
        let kpool = self.kpool;
        let total = tape.total;
        let p_n = total / kpool; // complete pools only
        self.update_pooled_cache(tape, capture, layer)?;
        if p_n == 0 {
            // Nothing to pool: the tail alone is visible.
            let width = self.tail_width();
            let mut sel = vec![-1i64; b * t * width];
            for bi in 0..b {
                for qi in 0..t {
                    let abs_qi = offset.saturating_sub(t) + qi;
                    let base = (bi * t + qi) * width;
                    let vis = abs_qi + 1;
                    let tail_count = vis % kpool;
                    let tail_start = vis - tail_count;
                    for m in 0..width.min(kpool - 1) {
                        if m < tail_count {
                            sel[base + m] = (tail_start + m) as i64;
                        }
                    }
                }
            }
            let arr: Vec<i32> = sel.iter().map(|&v| v as i32).collect();
            return Ok(Array::from_slice(&arr, &[b as i32, t as i32, width as i32]));
        }
        let pooled = tape
            .pooled_view()
            .ok_or_else(|| anyhow::anyhow!("indexer pooled missing"))?; // [B,P,128]

        let score_scale = ((hd * heads) as f64).powf(-0.5) as f32;
        let stream = lisa_mlx::Stream::thread_local_or_default();
        let scores = {
            let _trace = lisa_mlx::trace::span_detail("glm53.dsa.indexer_scores", layer as u64);
            let scorer = if prefill_mode && t == 1 {
                lisa_mlx::models::glm53::kernels::glm53_indexer_scores_batch
            } else if crate::core::quant::glm53_indexer_decode_rows4_override().unwrap_or(true) {
                lisa_mlx::models::glm53::kernels::glm53_indexer_scores_rows4
            } else {
                lisa_mlx::models::glm53::kernels::glm53_indexer_scores
            };
            scorer(
                &iq,
                &weights,
                &pooled,
                offset.saturating_sub(t),
                kpool,
                score_scale,
                &stream,
            )
            .ok_or_else(|| anyhow::anyhow!("glm53 dsa: ds4 indexer-score Metal dispatch failed"))?
        };
        if let Some(c) = capture.filter(|capture| capture.wants(layer)) {
            let pos0 = offset.saturating_sub(t);
            c.store(&format!("L{layer}.dsa_indexer_q_p{pos0}_t{t}"), &iq);
            c.store(
                &format!("L{layer}.dsa_indexer_weights_p{pos0}_t{t}"),
                &weights,
            );
            c.store(&format!("L{layer}.dsa_pooled_keys"), &pooled);
            c.store(&format!("L{layer}.dsa_index_scores_p{pos0}_t{t}"), &scores);
        }
        Ok(scores)
    }

    /// Decode GPU-mask path: indexer scores stay on the GPU, the top-`select_k` pools
    /// come from ds4's blockwise GPU top-k, and the additive keep mask is built with
    /// elementwise ops — membership (pool_of(t) among the selected pools) or
    /// the always-selected tail. Returns the additive mask [B, 1, T, tc]
    /// f32 (0.0 keep, -1e30 masked — the host path's exact values).
    fn select_keep_gpu(
        &self,
        tape: &mut IndexerTape,
        x: &Array,
        qr: &Array,
        offset: usize,
        t: usize,
        tc: usize,
        layer: usize,
    ) -> anyhow::Result<Array> {
        let b = x.dim(0) as usize;
        let kpool = self.kpool;
        let total = tape.total;
        let p_n = total / kpool;
        let (iq, weights) = self.project_score_inputs(x, qr)?;
        let index_sc = self.index_scores(tape, &iq, &weights, offset, t, None, 0, false)?;
        debug_assert_eq!(t, 1);
        let select_k = (self.topk / kpool).min(p_n);
        // Causal visibility at t == 1: every COMPLETE pool is visible
        // (pool_end(pp) = pp*4+3 <= offset-1 holds for all pp < p_n), so the
        // candidates are exactly the pooled rows — the host path's filter is
        // vacuous here.
        let score_rows = index_sc.reshape(&[b as i32, p_n as i32])?;
        let stream = lisa_mlx::Stream::thread_local_or_default();
        let top = {
            let _trace = lisa_mlx::trace::span_detail("glm53.dsa.topk_keep", layer as u64);
            indexer_topk(&score_rows, select_k, &stream).ok_or_else(|| {
                anyhow::anyhow!("glm53 dsa: ds4 blockwise top-k Metal dispatch failed")
            })?
        };
        // Membership: pool_of(pos) among the selected pools, OR the tail.
        let _trace_keep = lisa_mlx::trace::span_detail("glm53.dsa.keep_mask_expand", layer as u64);
        let vis = tc; // t == 1: the query sees tokens 0..tc-1
        let tail_count = vis % kpool;
        let tail_start = vis - tail_count;
        let mut ar_v = Vec::with_capacity(tc);
        for i in 0..tc {
            ar_v.push(i as i32);
        }
        let ar = Array::from_slice(&ar_v, &[tc as i32]); // [tc] i32
        let pool_of = lisa_mlx::ops::floor_divide(&ar, &Array::from_int(kpool as i32))?; // [tc]
        let top_b = top
            .expand_dims(1)?
            .broadcast_to(&[b as i32, tc as i32, select_k as i32])?;
        let pool_b = pool_of.expand_dims(0)?.expand_dims(2)?.broadcast_to(&[
            b as i32,
            tc as i32,
            select_k as i32,
        ])?;
        let hi_b = pool_b.le(&top_b)?;
        let member = pool_b.ge(&top_b)?.logical_and(&hi_b)?; // u8 [tc, sk]
        let member_any = member
            .as_dtype(Dtype::Float32)?
            .sum_axis(2, false)?
            .gt(&Array::from_f32_as(0.0f32, Dtype::Float32))?
            .as_dtype(Dtype::Float32)?; // [B,tc] f32 0/1
        let tail = ar
            .ge(&Array::from_int(tail_start as i32))?
            .as_dtype(Dtype::Float32)?
            .expand_dims(0)?
            .broadcast_to(&[b as i32, tc as i32])?;
        let keep = member_any.maximum(&tail)?; // [B,tc] f32 0/1
        // Additive mask: 0.0 keep, -1e30 masked (host-path values).
        let mask_shape = [b as i32, tc as i32];
        let zero = Array::from_f32_as(0.0f32, Dtype::Float32).broadcast_to(&mask_shape)?;
        let neg_big = Array::from_f32_as(-1e30f32, Dtype::Float32).broadcast_to(&mask_shape)?;
        let m = lisa_mlx::Array::where_(&keep, &zero, &neg_big)?;
        Ok(m.reshape(&[(b) as i32, 1, 1, (tc) as i32])?)
    }

    /// GPU causal top-pool selection expanded to absolute token ids consumed
    /// by absorbed MLA attention. No score or selection readback crosses to
    /// host. Each query only admits pools whose four tokens are all visible;
    /// the incomplete tail is appended in position order.
    fn select_ids_gpu(
        &self,
        tape: &mut IndexerTape,
        x: &Array,
        qr: &Array,
        offset: usize,
        t: usize,
        capture: Option<&crate::core::capture::Capture>,
        layer: usize,
        prefill_mode: bool,
    ) -> anyhow::Result<(Array, usize)> {
        let b = x.dim(0) as usize;
        let kpool = self.kpool;
        let p_n = tape.total / kpool;
        // At one-token decode, every complete pool is visible. The pool
        // expansion therefore guarantees this leading token-ID prefix; its
        // separately reserved incomplete-tail slots may contain -1 sentinels.
        let selected_valid_prefix = if t == 1 {
            (self.topk / kpool).min(p_n) * kpool
        } else {
            0
        };
        const SCORE_SCRATCH_BYTES: usize = 256 * 1024 * 1024;
        const MAX_SCORE_TOKENS: usize = 2048;
        let score_row_bytes = b
            .saturating_mul(p_n)
            .saturating_mul(Dtype::Float32.size_of());
        let score_cap = (SCORE_SCRATCH_BYTES / score_row_bytes.max(1)).clamp(1, MAX_SCORE_TOKENS);
        let rows_per_chunk = score_cap.min(t.max(1));
        let query_start = offset.saturating_sub(t);
        // Match ds4's prefill order: project the complete query batch once,
        // then chunk only the score tensor's working set. Splitting the tail
        // before projection changes a large BF16 GEMM into a small-row GEMV.
        let (iq_all, weights_all) = self.project_score_inputs(x, qr)?;
        let mut selections = Vec::new();
        let mut q0 = 0usize;
        while q0 < t {
            let n = rows_per_chunk.min(t - q0);
            let iq_chunk = narrow(&iq_all, 2, q0, n)?;
            let weights_chunk = narrow(&weights_all, 1, q0, n)?;
            let chunk_offset = query_start + q0 + n;
            let scores = self.index_scores(
                tape,
                &iq_chunk,
                &weights_chunk,
                chunk_offset,
                n,
                capture,
                layer,
                prefill_mode,
            )?;
            let selected = self.expand_score_selection(&scores, b, n, p_n, chunk_offset, layer)?;
            if let Some(c) = capture.filter(|capture| capture.wants(layer)) {
                c.store(
                    &format!("L{layer}.dsa_selected_p{}_t{n}", query_start + q0),
                    &selected,
                );
            }
            selections.push(selected);
            q0 += n;
        }
        if selections.len() == 1 {
            return Ok((selections.remove(0), selected_valid_prefix));
        }
        let refs: Vec<&Array> = selections.iter().collect();
        Ok((
            Array::concatenate(&refs, 1)?.contiguous()?,
            selected_valid_prefix,
        ))
    }

    fn expand_score_selection(
        &self,
        index_sc: &Array,
        b: usize,
        t: usize,
        p_n: usize,
        offset: usize,
        layer: usize,
    ) -> anyhow::Result<Array> {
        let kpool = self.kpool;
        let q_rows = b * t;
        let select_k = (self.topk / kpool).min(p_n);
        if select_k == 0 {
            anyhow::bail!("glm53 dsa: no complete indexer pools available");
        }

        // The index score tensor already has the [B,T,P] layout. Mask pools
        // that end after this query before sorting; otherwise a high-scoring
        // future pool would consume a top-k slot and hide a valid earlier one.
        let abs_start = offset.saturating_sub(t);
        let mut visible_host = Vec::with_capacity(q_rows);
        for _bi in 0..b {
            for qi in 0..t {
                visible_host.push((abs_start + qi + 1) as i32);
            }
        }
        let visible = Array::from_slice(&visible_host, &[q_rows as i32, 1]);
        let mut pool_end = Vec::with_capacity(p_n);
        for pp in 0..p_n {
            pool_end.push(((pp + 1) * kpool) as i32);
        }
        let pool_end = Array::from_slice(&pool_end, &[1, p_n as i32]);
        let eligible = pool_end
            .broadcast_to(&[q_rows as i32, p_n as i32])?
            .le(&visible.broadcast_to(&[q_rows as i32, p_n as i32])?)?;
        let scores = index_sc.reshape(&[q_rows as i32, p_n as i32])?;
        let masked_fill = Array::from_f32_as(f32::NEG_INFINITY, Dtype::Float32)
            .broadcast_to(&[q_rows as i32, p_n as i32])?;
        let masked = Array::where_(&eligible, &scores, &masked_fill)?;
        let stream = lisa_mlx::Stream::thread_local_or_default();
        let top = {
            let _trace = lisa_mlx::trace::span_detail("glm53.dsa.topk", layer as u64);
            indexer_topk(&masked, select_k, &stream).ok_or_else(|| {
                anyhow::anyhow!("glm53 dsa: ds4 blockwise top-k Metal dispatch failed")
            })?
        };
        let expanded = {
            let _trace =
                lisa_mlx::trace::span_detail("glm53.dsa.selection_expansion", layer as u64);
            lisa_mlx::models::glm53::kernels::glm53_expand_pool_selection_for_attention(
                &top,
                t,
                abs_start,
                kpool,
                self.always_tail,
                &stream,
            )
            .ok_or_else(|| {
                anyhow::anyhow!("glm53 dsa: ds4 pool-selection expansion Metal dispatch failed")
            })?
        };
        Ok(expanded.reshape(&[
            b as i32,
            t as i32,
            (select_k * kpool + kpool.saturating_sub(1)) as i32,
        ])?)
    }

    /// Host top-k per query with causal visibility (absolute positions) —
    /// prefill path and the decode fallback (LISA_GLM53_POOL_OPS=1).
    fn select(
        &self,
        x: &Array,
        tape: &mut IndexerTape,
        qr: &Array,
        offset: usize,
        t: usize,
        tc: usize,
        prefill_mode: bool,
    ) -> anyhow::Result<Array> {
        let (b, kpool) = (x.dim(0) as usize, self.kpool);
        let (iq, weights) = self.project_score_inputs(x, qr)?;
        let index_sc = self.index_scores(tape, &iq, &weights, offset, t, None, 0, prefill_mode)?;
        let sc_h: Vec<f32> = index_sc.as_dtype(Dtype::Float32)?.to_vec1()?;
        let pool_end = |pp: usize| -> usize { pp * kpool + kpool - 1 };
        let p_n = tape.total / kpool;
        let select_k = (self.topk / kpool).min(p_n);
        let width = select_k * kpool + (kpool - 1);
        let off = offset.saturating_sub(t);
        let mut sel = vec![-1i64; b * t * width];
        for bi in 0..b {
            for qi in 0..t {
                let abs_qi = off + qi;
                let mut cand: Vec<(f32, usize)> = Vec::with_capacity(p_n);
                for pp in 0..p_n {
                    if pool_end(pp) <= abs_qi {
                        cand.push((sc_h[(bi * t + qi) * p_n + pp], pp));
                    }
                }
                cand.sort_by(|a, b2| b2.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
                let base = (bi * t + qi) * width;
                let mut w = 0usize;
                for (_, pp) in cand.iter().take(select_k) {
                    for j in 0..kpool {
                        let idx = (pp * kpool + j) as i64;
                        if idx < tc as i64 {
                            sel[base + w] = idx;
                            w += 1;
                        }
                    }
                }
                // Always-select tail: the last (vis % kpool) tokens.
                let vis = abs_qi + 1;
                let tail_count = vis % kpool;
                let tail_start = vis - tail_count;
                for m in 0..(kpool - 1) {
                    if m < tail_count {
                        sel[base + w] = (tail_start + m) as i64;
                    } else {
                        sel[base + w] = -1;
                    }
                    w += 1;
                }
            }
        }
        let arr: Vec<i32> = sel.iter().map(|&v| v as i32).collect();
        Ok(Array::from_slice(&arr, &[b as i32, t as i32, width as i32]))
    }

    fn tail_width(&self) -> usize {
        self.kpool - 1
    }
}

/// Causal additive mask `[B,1,T,Tc]` f32 (0 visible, -1e30 hidden).
fn glm53_weighted_rms_norm(x: &Array, weight: &Array, eps: f32) -> anyhow::Result<Array> {
    let stream = lisa_mlx::Stream::thread_local_or_default();
    lisa_mlx::models::glm53::kernels::glm53_mtp_rms_norm(x, weight, eps, &stream).ok_or_else(|| {
        anyhow::anyhow!(
            "GLM ds4-order weighted RMSNorm declined for shape {:?}",
            x.shape()
        )
    })
}

fn glm53_kv_lora_rms_norm(x: &Array, weight: &Array, eps: f32) -> anyhow::Result<Array> {
    let stream = lisa_mlx::Stream::thread_local_or_default();
    lisa_mlx::models::glm53::kernels::glm53_mtp_kv_lora_rms_norm(x, weight, eps, &stream)
        .ok_or_else(|| {
            anyhow::anyhow!(
                "GLM ds4-order KV-LoRA RMSNorm declined for shape {:?}",
                x.shape()
            )
        })
}

pub fn causal_mask(_rt_of: &Array, b: usize, t: usize, tc: usize) -> anyhow::Result<Array> {
    let mut m = vec![0f32; (b as usize) * (t as usize) * (tc as usize)];
    let offset = tc - t; // queries attend up to (offset + qi) inclusive
    for bi in 0..b {
        for qi in 0..t {
            let lim = offset + qi;
            for kv in 0..tc {
                if kv > lim {
                    m[(bi * t + qi) * tc + kv] = -1e30;
                }
            }
        }
    }
    Ok(Array::from_slice(&m, &[b as i32, 1, t as i32, tc as i32]).as_dtype(Dtype::Float32)?)
}

/// Build the additive keep mask `[B,1,T,Tc]` from the selection `[B,T,width]`
/// (i32, -1 = none): visible iff causal AND selected.
pub fn keep_mask_from_selection(
    _rt_of: &Array,
    sel: &Array,
    t: usize,
    tc: usize,
) -> anyhow::Result<Array> {
    let b = sel.dim(0) as usize;
    let width = sel.dim(2) as usize;
    let sel_h: Vec<i32> = sel.to_vec1()?;
    let mut m = vec![-1e30f32; (b as usize) * (t as usize) * (tc as usize)];
    let offset = tc - t;
    for bi in 0..b {
        for qi in 0..t {
            let lim = offset + qi;
            let base = (bi * t + qi) * width;
            for w in 0..width {
                let idx = sel_h[base + w];
                if idx >= 0 && (idx as usize) <= lim {
                    m[(bi * t + qi) * tc + idx as usize] = 0.0;
                }
            }
        }
    }
    Ok(Array::from_slice(&m, &[b as i32, 1, t as i32, tc as i32]).as_dtype(Dtype::Float32)?)
}

/// LayerNorm (mean/var, weight + bias) over the last axis, F32 math.
pub fn layer_norm(x: &Array, w: &Array, b: &Array, eps: f32) -> anyhow::Result<Array> {
    let xf = x.as_dtype(Dtype::Float32)?;
    let mean = xf.mean_axis(2, true)?; // [B,T,1]
    let cent = xf.subtract(&mean)?;
    let var = cent.multiply(&cent)?.mean_axis(2, true)?;
    let inv = var.add(&Array::from_f32_as(eps, Dtype::Float32))?.rsqrt()?;
    let n = cent.multiply(&inv)?;
    let out = n
        .multiply(&w.as_dtype(Dtype::Float32)?)?
        .add(&b.as_dtype(Dtype::Float32)?)?;
    Ok(out)
}

#[cfg(test)]
#[path = "sparse_tests.rs"]
mod tests;
