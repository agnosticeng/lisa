//! Loading boundary for the pinned DFlash2 drafter: BF16 safetensors only,
//! each tensor checked against the draft schema. The draft's projections are
//! stored as the reference's draft image stores them: each group of 64 BF16
//! weights quantized to MLX affine 4 bits (`af4g64_group`) and repacked into
//! its planes (`<projection>.af4g64`, [`model::reference_projections`]),
//! beside the original BF16 tensors.
use crate::{
    checkpoint::catalog::inkling2::{DFLASH_REPO as REPO, DFLASH_REVISION as REVISION},
    weights::mlx::Shard,
};
use anyhow::{Context, Result, ensure};
use lisa_backend::{ManagedDevice, ScalarType as Dtype};
use lisa_models::inkling2::dflash as model;
use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
};

/// A `dflash2` directory beside the target, else the pinned Hub snapshot.
pub fn selection(target: &Path) -> Result<Option<PathBuf>> {
    let sibling = target
        .parent()
        .context("missing target directory")?
        .join("dflash2");
    if sibling.join("model.safetensors").is_file() {
        return Ok(Some(sibling));
    }
    let p = crate::checkpoint::hf::repo_dir(REPO)
        .join("snapshots")
        .join(REVISION);
    Ok((p.join("model.safetensors").is_file() && p.join("config.json").is_file()).then_some(p))
}

/// The draft's tensors for the draft step: every BF16 tensor of the schema
/// (checked against it), and each projection's af4g64 planes as the
/// reference's draft image stores them.
pub(crate) fn load<D: crate::bindings::BindingDevice + ManagedDevice>(
    device: &D,
    path: &Path,
    target_bytes: usize,
) -> Result<BTreeMap<String, D::Tensor>> {
    let config: serde_json::Value =
        serde_json::from_slice(&std::fs::read(path.join("config.json"))?)?;
    model::validate_config(&config)?;
    let bytes = std::fs::metadata(path.join("model.safetensors"))?.len() as usize;
    ensure!(
        target_bytes
            .checked_add(bytes)
            .and_then(|n| n.checked_add(256 << 20))
            .is_some_and(|n| n <= crate::memory::ram_budget()),
        "Inkling and DFlash2 exceed RAM budget"
    );
    let shard = Shard::open(device.clone(), &path.join("model.safetensors"))?;
    let mut weights = BTreeMap::new();
    for (name, shape) in &model::schema() {
        let (_, dtype, actual) = shard
            .tensor_bytes(name)
            .with_context(|| format!("missing DFlash2 tensor {name}"))?;
        ensure!(
            dtype == Dtype::BF16 && actual == shape,
            "invalid DFlash2 tensor {name}"
        );
        weights.insert(
            name.clone(),
            shard
                .tensor_array(name)?
                .context("missing DFlash2 weight")?,
        );
    }
    let ops = crate::tensor_ops::TensorOps::new(device, crate::tensor_ops::TensorLaw::Native);
    for p in model::reference_projections() {
        let mut bf16 = Vec::with_capacity(p.rows as usize * p.inputs as usize * 2);
        for part in &p.parts {
            let (bytes, _, _) = shard
                .tensor_bytes(part)
                .with_context(|| format!("missing DFlash2 tensor {part}"))?;
            bf16.extend_from_slice(bytes);
        }
        let native = af4g64_rows(&bf16, p.inputs as usize)?;
        let row_bytes = p.inputs / 64 * 36;
        let source = ops.pull(&native, &[1, p.rows, row_bytes], Dtype::U8)?;
        let planes = ops.required(
            lisa_backend::Kernel::PackedCompute(lisa_backend::PackedCompute::RepackPlanes {
                ggml_type: AF4G64_TYPE,
                first_row: 0,
                rows: p.rows as u32,
                columns: p.inputs as u32,
            }),
            &[&source],
        )?;
        weights.insert(p.name, planes);
    }
    Ok(weights)
}

/// The reference loader's tensor type of an af4g64 native block
/// (`QUANT_AFFINE_TYPE(4, 64)`).
const AF4G64_TYPE: u32 = 0x4D4C_0000 | (4 << 8) | 64;

/// The BF16 nearest a finite value, ties to even.
fn nearest_bf16(value: f32) -> u16 {
    let bits = value.to_bits();
    ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16) as u16
}

/// A group of 64 BF16 weights (little-endian bytes) as the reference
/// quantizes it into a native af4g64 block, MLX's affine rounding in F32:
/// the range runs from the minimum to the maximum or 0, whichever is
/// greater; the end farther from zero is the bias and the scale is adjusted
/// so that 0 falls on a code unless that code is 0; each code is
/// `round((w - bias) / scale)` (halves away from zero) clamped to [0, 15].
/// The block holds the BF16 scale, the BF16 bias, then the codes two a byte,
/// the even element's in the low nibble.
pub(crate) fn af4g64_group(weights: &[u8]) -> Result<[u8; 36]> {
    ensure!(
        weights.len() == 128,
        "an af4g64 group holds 64 BF16 weights"
    );
    let mut w = [0f32; 64];
    for (i, value) in w.iter_mut().enumerate() {
        *value = f32::from_bits(
            u32::from(u16::from_le_bytes([weights[2 * i], weights[2 * i + 1]])) << 16,
        );
    }
    ensure!(
        w.iter().all(|x| x.is_finite()),
        "non-finite weight in a BF16 projection"
    );
    let (mut minimum, mut maximum) = (w[0], w[0]);
    for &x in &w {
        minimum = minimum.min(x);
        maximum = maximum.max(x);
    }
    maximum = maximum.max(0.0);
    let minimum_edge = minimum.abs() > maximum.abs();
    let mut step = ((maximum - minimum) / 15.0).max(1e-7);
    if !minimum_edge {
        step = -step;
    }
    let edge = if minimum_edge { minimum } else { maximum };
    let q0 = (edge / step).round();
    let mut offset = 0.0f32;
    if q0 != 0.0 {
        step = edge / q0;
        offset = edge;
    }
    let mut block = [0u8; 36];
    block[..2].copy_from_slice(&nearest_bf16(step).to_le_bytes());
    block[2..4].copy_from_slice(&nearest_bf16(offset).to_le_bytes());
    let code = |x: f32| ((x - offset) / step).round().clamp(0.0, 15.0) as u8;
    for i in (0..64).step_by(2) {
        block[4 + i / 2] = code(w[i]) | (code(w[i + 1]) << 4);
    }
    Ok(block)
}

/// BF16 rows of `inputs` weights each as native af4g64 rows, on every core.
pub(crate) fn af4g64_rows(bf16: &[u8], inputs: usize) -> Result<Vec<u8>> {
    ensure!(
        inputs > 0 && inputs.is_multiple_of(64) && bf16.len().is_multiple_of(2 * inputs),
        "af4g64 rows of whole 64-weight groups"
    );
    let groups = bf16.len() / 128;
    let mut out = vec![0u8; groups * 36];
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
    let per = groups.div_ceil(threads).max(1);
    std::thread::scope(|scope| -> Result<()> {
        let workers = out
            .chunks_mut(per * 36)
            .zip(bf16.chunks(per * 128))
            .map(|(out, source)| {
                scope.spawn(move || -> Result<()> {
                    for (block, group) in out.chunks_mut(36).zip(source.chunks(128)) {
                        block.copy_from_slice(&af4g64_group(group)?);
                    }
                    Ok(())
                })
            })
            .collect::<Vec<_>>();
        for worker in workers {
            worker
                .join()
                .map_err(|_| anyhow::anyhow!("af4g64 quantization panicked"))??;
        }
        Ok(())
    })?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bf16_bytes(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
            .collect()
    }
    fn decode(block: &[u8; 36]) -> Vec<f32> {
        let half = |i: usize| {
            f32::from_bits(u32::from(u16::from_le_bytes([block[i], block[i + 1]])) << 16)
        };
        let (scale, bias) = (half(0), half(2));
        (0..64)
            .map(|e| {
                let code = (block[4 + e / 2] >> (4 * (e % 2))) & 15;
                scale * f32::from(code) + bias
            })
            .collect()
    }

    #[test]
    fn af4g64_keeps_a_positive_edge_as_its_bias() -> Result<()> {
        // The maximum is the farther end: the bias is the maximum and the
        // scale negative, so code 0 decodes to it exactly.
        let mut w = [0f32; 64];
        for (i, v) in w.iter_mut().enumerate() {
            *v = (i as f32 - 20.0) / 32.0;
        }
        let block = af4g64_group(&bf16_bytes(&w))?;
        let decoded = decode(&block);
        let max = w.iter().fold(f32::MIN, |m, &x| m.max(x));
        assert!(decoded.contains(&max));
        // The scale is adjusted so that 0 falls on a code, then rounded to
        // bf16: within six tenths of a step everywhere.
        let step = (max - w[0]) / 15.0;
        for (d, x) in decoded.iter().zip(w) {
            assert!((d - x).abs() <= step * 0.6, "{d} {x}");
        }
        Ok(())
    }

    #[test]
    fn af4g64_codes_zero_exactly_and_packs_low_nibbles_first() -> Result<()> {
        // A negative edge: the bias is the minimum, the scale 1.5 / 13 so
        // that 0 falls on code 13 before the scale's bf16 rounding.
        let mut w = [0f32; 64];
        w[0] = -1.5;
        w[1] = 0.25;
        let block = af4g64_group(&bf16_bytes(&w))?;
        let decoded = decode(&block);
        assert_eq!(decoded[0], -1.5);
        assert_eq!(
            u16::from_le_bytes([block[0], block[1]]),
            nearest_bf16(1.5 / 13.0)
        );
        // Element 0 holds code 0 in the low nibble of byte 4, element 1 its
        // code in the high nibble, and every zero code 13.
        assert_eq!(block[4] & 15, 0);
        assert_eq!(block[4] >> 4, (0.25f32 / (1.5 / 13.0) + 13.0).round() as u8);
        assert!(block[5..].iter().all(|&b| b == 13 | (13 << 4)));
        assert!(decoded[2..].iter().all(|&x| x.abs() < 0.01));
        // All zeros: a zero scale floor, bias 0, codes 0.
        let zero = af4g64_group(&[0u8; 128])?;
        assert_eq!(&zero[2..], &[0u8; 34][..]);
        assert_eq!(
            nearest_bf16(1e-7),
            u16::from_le_bytes([zero[0], zero[1]]) ^ 0x8000
        );
        Ok(())
    }

    #[test]
    fn af4g64_rows_match_groups_on_every_core() -> Result<()> {
        let values: Vec<f32> = (0..64 * 37)
            .map(|i| ((i * 7919) % 211) as f32 / 97.0 - 1.0)
            .collect();
        let bf16 = bf16_bytes(&values);
        let rows = af4g64_rows(&bf16, 64 * 37)?;
        for g in 0..37 {
            assert_eq!(rows[g * 36..][..36], af4g64_group(&bf16[g * 128..][..128])?);
        }
        assert!(af4g64_rows(&bf16, 96).is_err());
        assert!(af4g64_group(&bf16_bytes(&[f32::NAN; 64])).is_err());
        Ok(())
    }
}
