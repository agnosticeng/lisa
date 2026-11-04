//! Reference decoding for original GGML blocks. Used for small CPU lookups
//! (Engram/embedding rows) and kernel comparisons, never a whole-model repack.
//! Ported from ggml-quants.c; see LICENSE-GGML and provenance.json.

#[path = "tables.rs"]
#[rustfmt::skip]
mod tables;

use super::TensorType;
use anyhow::{Result, bail, ensure};
use tables::*;

fn u16_at(b: &[u8], at: usize) -> u16 {
    u16::from_le_bytes([b[at], b[at + 1]])
}

fn f16_at(b: &[u8], at: usize) -> f32 {
    half::f16::from_bits(u16_at(b, at)).to_f32()
}

pub fn decode(kind: TensorType, packed: &[u8], output: &mut [f32]) -> Result<()> {
    let (_, width, bytes) = kind.layout()?;
    ensure!(
        !output.is_empty() && output.len() as u64 % width == 0,
        "invalid GGUF decode output length"
    );
    ensure!(
        packed.len() as u64 == kind.row_bytes(output.len() as u64)?,
        "GGUF packed row length mismatch"
    );
    for (b, out) in packed
        .chunks_exact(bytes as usize)
        .zip(output.chunks_exact_mut(width as usize))
    {
        match kind {
            TensorType::F32 => out[0] = f32::from_le_bytes(b.try_into().unwrap()),
            TensorType::F16 => out[0] = f16_at(b, 0),
            TensorType::BF16 => out[0] = half::bf16::from_bits(u16_at(b, 0)).to_f32(),
            TensorType::Q8_0 => {
                let scale = f16_at(b, 0);
                for i in 0..32 {
                    out[i] = scale * b[i + 2] as i8 as f32;
                }
            }
            TensorType::IQ4_NL => {
                let scale = f16_at(b, 0);
                for i in 0..16 {
                    out[i] = scale * KVALUES_IQ4NL[(b[i + 2] & 15) as usize] as f32;
                    out[i + 16] = scale * KVALUES_IQ4NL[(b[i + 2] >> 4) as usize] as f32;
                }
            }
            TensorType::IQ2_XXS => iq2_xxs(b, out),
            TensorType::IQ2_XS => iq2_xs(b, out),
            TensorType::IQ1_M => iq1_m(b, out),
            TensorType::Q2_K => q2_k(b, out),
            TensorType::Q4_K => q4_k(b, out),
            TensorType::Q5_K => q5_k(b, out),
            _ => bail!("CPU row decoder not implemented for {}", kind.layout()?.0),
        }
    }
    Ok(())
}

fn q2_k(b: &[u8], out: &mut [f32]) {
    let d = f16_at(b, 80);
    let min = f16_at(b, 82);
    for half in 0..2 {
        for shift in 0..4 {
            for group in 0..2 {
                let sc = b[half * 8 + shift * 2 + group];
                let dl = d * (sc & 15) as f32;
                let ml = min * (sc >> 4) as f32;
                for l in 0..16 {
                    let q = (b[16 + half * 32 + group * 16 + l] >> (2 * shift)) & 3;
                    out[half * 128 + shift * 32 + group * 16 + l] = dl * q as f32 - ml;
                }
            }
        }
    }
}

fn q4_k(b: &[u8], out: &mut [f32]) {
    let d = f16_at(b, 0);
    let min = f16_at(b, 2);
    for group in 0..8 {
        let (scale, bias) = if group < 4 {
            (b[4 + group] & 63, b[8 + group] & 63)
        } else {
            (
                (b[8 + group] & 15) | ((b[group] >> 6) << 4),
                (b[8 + group] >> 4) | ((b[4 + group] >> 6) << 4),
            )
        };
        let dl = d * scale as f32;
        let ml = min * bias as f32;
        for l in 0..32 {
            let q = (b[16 + (group / 2) * 32 + l] >> (4 * (group % 2))) & 15;
            out[group * 32 + l] = dl * q as f32 - ml;
        }
    }
}

fn iq2_xxs(b: &[u8], out: &mut [f32]) {
    let d = f16_at(b, 0);
    for group in 0..8 {
        let at = 2 + group * 8;
        let aux = u32::from_le_bytes(b[at + 4..at + 8].try_into().unwrap());
        let scale = d * (0.5 + (aux >> 28) as f32) * 0.25;
        for l in 0..4 {
            let grid = IQ2XXS_GRID[b[at + l] as usize].to_le_bytes();
            let signs = KSIGNS_IQ2XS[((aux >> (7 * l)) & 127) as usize];
            for j in 0..8 {
                out[group * 32 + l * 8 + j] =
                    scale * grid[j] as f32 * if signs & (1 << j) != 0 { -1.0 } else { 1.0 };
            }
        }
    }
}

fn iq2_xs(b: &[u8], out: &mut [f32]) {
    let d = f16_at(b, 0);
    for group in 0..8 {
        let scales = [
            d * (0.5 + (b[66 + group] & 15) as f32) * 0.25,
            d * (0.5 + (b[66 + group] >> 4) as f32) * 0.25,
        ];
        for l in 0..4 {
            let q = u16_at(b, 2 + 2 * (4 * group + l));
            let grid = IQ2XS_GRID[(q & 511) as usize].to_le_bytes();
            let signs = KSIGNS_IQ2XS[(q >> 9) as usize];
            for j in 0..8 {
                out[group * 32 + l * 8 + j] =
                    scales[l / 2] * grid[j] as f32 * if signs & (1 << j) != 0 { -1.0 } else { 1.0 };
            }
        }
    }
}

fn iq1_m(b: &[u8], out: &mut [f32]) {
    let sc = [u16_at(b, 48), u16_at(b, 50), u16_at(b, 52), u16_at(b, 54)];
    let bits = (sc[0] >> 12) | ((sc[1] >> 8) & 0xf0) | ((sc[2] >> 4) & 0xf00) | (sc[3] & 0xf000);
    let d = half::f16::from_bits(bits).to_f32();
    for group in 0..8 {
        let scales = [
            d * (2 * ((sc[group / 2] >> (6 * (group % 2))) & 7) + 1) as f32,
            d * (2 * ((sc[group / 2] >> (6 * (group % 2) + 3)) & 7) + 1) as f32,
        ];
        for l in 0..4 {
            let high = b[32 + 2 * group + l / 2];
            let nibble = high >> (4 * (l % 2));
            let index = b[4 * group + l] as usize | (((nibble & 7) as usize) << 8);
            let grid = IQ1S_GRID[index].to_le_bytes();
            let delta = if nibble & 8 != 0 { -0.125 } else { 0.125 };
            for j in 0..8 {
                out[32 * group + 8 * l + j] = scales[l / 2] * (grid[j] as i8 as f32 + delta);
            }
        }
    }
}

fn q5_k(b: &[u8], out: &mut [f32]) {
    let d = f16_at(b, 0);
    let dmin = f16_at(b, 2);
    let scales = &b[4..16];
    for group in 0..8 {
        let (scale, min) = if group < 4 {
            (scales[group] & 63, scales[group + 4] & 63)
        } else {
            (
                (scales[group + 4] & 15) | ((scales[group - 4] >> 6) << 4),
                (scales[group + 4] >> 4) | ((scales[group] >> 6) << 4),
            )
        };
        let ds = d * scale as f32;
        let ms = dmin * min as f32;
        for j in 0..32 {
            let low = (b[48 + (group / 2) * 32 + j] >> (4 * (group % 2))) & 15;
            let high = if b[16 + j] & (1 << group) != 0 { 16 } else { 0 };
            out[group * 32 + j] = ds * (low + high) as f32 - ms;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Frozen from the unmodified GGML C dequantizers at the revision in
    /// provenance.json, compiled with clang -O2 -ffp-contract=off. This checks
    /// 204,800 scalar values across the original packed layouts, not a second
    /// Rust implementation of the same formulas.
    #[test]
    fn gguf_block_decoders_match_ggml_c() {
        for (kind, expected) in [
            (TensorType::Q8_0, 0xc0087397958e8331u64),
            (TensorType::Q5_K, 0xf8a0e7f3f1b1aee0),
            (TensorType::Q2_K, 0x92f0e7a5daf379e9),
            (TensorType::Q4_K, 0xa8fd5258d144add5),
            (TensorType::IQ2_XXS, 0xef5d016f9a4d2e0f),
            (TensorType::IQ2_XS, 0x66d89c9304e90e7b),
            (TensorType::IQ1_M, 0x0c854b853679c2a4),
            (TensorType::IQ4_NL, 0xd3639c1e16eb09e0),
        ] {
            let (_, width, bytes) = kind.layout().unwrap();
            let mut state = 0x12345678u32;
            let mut hash = 0xcbf29ce484222325u64;
            for block in 0..128 {
                let mut packed = vec![0u8; bytes as usize];
                for byte in &mut packed {
                    state ^= state << 13;
                    state ^= state >> 17;
                    state ^= state << 5;
                    *byte = (state >> 24) as u8;
                }
                let scale = (0x2800 + (block % 8) * 0x200) as u16;
                if kind == TensorType::Q2_K {
                    packed[80..82].copy_from_slice(&scale.to_le_bytes());
                    packed[82..84].copy_from_slice(&0x2400u16.to_le_bytes());
                } else if kind == TensorType::IQ1_M {
                    for i in 0..4 {
                        let at = 48 + 2 * i;
                        let value =
                            (u16_at(&packed, at) & 0xfff) | (((scale >> (4 * i)) & 15) << 12);
                        packed[at..at + 2].copy_from_slice(&value.to_le_bytes());
                    }
                } else {
                    packed[..2].copy_from_slice(&scale.to_le_bytes());
                    if matches!(kind, TensorType::Q5_K | TensorType::Q4_K) {
                        packed[2..4].copy_from_slice(&0x2400u16.to_le_bytes());
                    }
                }
                let mut decoded = vec![0f32; width as usize];
                decode(kind, &packed, &mut decoded).unwrap();
                for value in decoded {
                    for byte in value.to_bits().to_le_bytes() {
                        hash = (hash ^ byte as u64).wrapping_mul(0x100000001b3);
                    }
                }
            }
            assert_eq!(hash, expected, "GGML reference mismatch for {kind:?}");
        }
    }

    #[test]
    fn gguf_decode_rejects_invalid_buffers() {
        assert!(decode(TensorType::IQ1_M, &[0; 55], &mut [0.0; 256]).is_err());
        assert!(decode(TensorType::IQ4_NL, &[0; 18], &mut [0.0; 31]).is_err());
        assert!(decode(TensorType::Q8_0, &[], &mut []).is_err());
    }
}
