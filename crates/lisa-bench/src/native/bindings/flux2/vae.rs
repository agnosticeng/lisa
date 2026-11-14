//! Native conformance adapters for portable image bindings.
use super::weights::Weights;
use lisa_metal::{Array, Dtype, device::MetalDevice};
use std::{
    collections::HashMap,
    path::{Path, PathBuf},
};
pub fn decode(dir: &Path, packed: &Array) -> anyhow::Result<Array> {
    lisa_engine::bindings::flux2::vae::decode(
        &crate::metal_context::execution_context(),
        dir,
        packed,
    )
}
pub(crate) fn decode_weights(
    weights: &Weights,
    packed: &Array,
    dtype: Dtype,
) -> anyhow::Result<Array> {
    let dtype = match dtype {
        Dtype::Bfloat16 => lisa_backend::ScalarType::BF16,
        Dtype::Float32 => lisa_backend::ScalarType::F32,
        _ => anyhow::bail!("unsupported VAE output dtype"),
    };
    lisa_engine::bindings::flux2::vae::decode_weights(weights, packed, dtype)
}
#[cfg(test)]
mod plan_tests {
    use super::*;
    use sha2::{Digest, Sha256};
    fn fixture(dtype: Dtype) -> anyhow::Result<Weights> {
        let mut shapes = std::collections::BTreeMap::<String, (Vec<i32>, f32)>::new();
        let mut pair = |prefix: &str, shape: Vec<i32>, shift: f32| {
            let n = shape[0];
            shapes.insert(format!("{prefix}.weight"), (shape, shift));
            shapes.insert(format!("{prefix}.bias"), (vec![n], 0.));
        };
        pair("post_quant_conv", vec![32, 1, 1, 32], 0.);
        pair("decoder.conv_in", vec![32, 3, 3, 32], 0.);
        let mut blocks = vec![
            "decoder.mid_block.resnets.0".to_string(),
            "decoder.mid_block.resnets.1".into(),
        ];
        for b in 0..4 {
            for l in 0..3 {
                blocks.push(format!("decoder.up_blocks.{b}.resnets.{l}"));
            }
        }
        for name in blocks {
            for i in 1..=2 {
                pair(&format!("{name}.norm{i}"), vec![32], 1.);
                pair(&format!("{name}.conv{i}"), vec![32, 3, 3, 32], 0.);
            }
        }
        // Exercise both the identity and projected residual paths.
        pair(
            "decoder.up_blocks.1.resnets.0.conv_shortcut",
            vec![32, 1, 1, 32],
            0.,
        );
        for b in 0..3 {
            pair(
                &format!("decoder.up_blocks.{b}.upsamplers.0.conv"),
                vec![32, 3, 3, 32],
                0.,
            );
        }
        pair("decoder.mid_block.attentions.0.group_norm", vec![32], 1.);
        for name in ["to_q", "to_k", "to_v", "to_out"] {
            pair(
                &format!("decoder.mid_block.attentions.0.{name}"),
                vec![32, 32],
                0.,
            );
        }
        pair("decoder.conv_norm_out", vec![32], 1.);
        pair("decoder.conv_out", vec![3, 3, 3, 32], 0.);
        shapes.insert("bn.running_mean".into(), (vec![128], 0.));
        shapes.insert("bn.running_var".into(), (vec![128], 1.));
        let mut seed = 319u64;
        let weights = shapes
            .into_iter()
            .map(|(name, (shape, shift))| {
                let data = (0..shape.iter().product::<i32>())
                    .map(|_| {
                        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                        ((seed >> 40) as f32 / 8_388_608. - 1.) * 0.025 + shift
                    })
                    .collect::<Vec<_>>();
                Ok((name, Array::from_slice(&data, &shape).as_dtype(dtype)?))
            })
            .collect::<anyhow::Result<_>>()?;
        Ok(Weights::from_map(
            crate::metal_context::execution_context(),
            weights,
        ))
    }
    #[test]
    fn vae_plan_matches_original_bits() -> anyhow::Result<()> {
        for dtype in [Dtype::Bfloat16, Dtype::Float32] {
            let weights = fixture(dtype)?;
            let data = (0..128 * 2 * 3)
                .map(|i| ((i * 7 % 31) as f32 - 15.) / 16.)
                .collect::<Vec<_>>();
            let input = Array::from_slice(&data, &[1, 128, 2, 3]).as_dtype(dtype)?;
            let output = decode_weights(&weights, &input, dtype)?.as_dtype(Dtype::Float32)?;
            let bytes = output
                .as_slice::<f32>()
                .iter()
                .flat_map(|x| x.to_bits().to_le_bytes())
                .collect::<Vec<_>>();
            assert_eq!(output.shape(), [1, 32, 48, 3]);
            let expected = if dtype == Dtype::Bfloat16 {
                "ef125f1d320f9f26181db2d3870cd5a16e09d6d760714c0b2d843b4a7ccc1819"
            } else {
                "b7350346964b409596dc36940c598ab1c980ed9bdf6cc5d2951061de090504f5"
            };
            assert_eq!(format!("{:x}", Sha256::digest(bytes)), expected);
        }
        Ok(())
    }
}
