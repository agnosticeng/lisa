//! Native conformance fixtures for the portable glm encoder.
pub type Encoder = lisa_engine::vision::glm::Encoder<lisa_metal::device::MetalDevice>;
#[cfg(test)]
use lisa_metal::Array;
pub use lisa_models::vision::glm::Config;
#[cfg(test)]
mod plan_tests {
    use super::*;
    use lisa_metal::Dtype;
    fn fixture() -> (Config, std::collections::HashMap<String, Array>) {
        let mut shapes = std::collections::BTreeMap::<String, (Vec<i32>, f32)>::new();
        for (name, shape) in [
            ("patch_embed.proj.weight", vec![8, 1, 1, 1, 3]),
            ("patch_embed.proj.bias", vec![8]),
            ("downsample.weight", vec![8, 2, 2, 8]),
            ("downsample.bias", vec![8]),
        ] {
            shapes.insert(name.into(), (shape, 0.));
        }
        shapes.insert("post_layernorm.weight".into(), (vec![8], 1.));
        for i in 0..2 {
            for (name, size) in [
                ("norm1", 8),
                ("norm2", 8),
                ("attn.q_norm", 4),
                ("attn.k_norm", 4),
            ] {
                shapes.insert(format!("blocks.{i}.{name}.weight"), (vec![size], 1.));
            }
            for (name, o, n) in [
                ("attn.qkv", 24, 8),
                ("attn.proj", 8, 8),
                ("mlp.gate_proj", 16, 8),
                ("mlp.up_proj", 16, 8),
                ("mlp.down_proj", 8, 16),
            ] {
                shapes.insert(format!("blocks.{i}.{name}.weight"), (vec![o, n], 0.));
                shapes.insert(format!("blocks.{i}.{name}.bias"), (vec![o], 0.));
            }
        }
        for (suffix, shift) in [("weight", 1.), ("bias", 0.)] {
            shapes.insert(
                format!("merger.post_projection_norm.{suffix}"),
                (vec![8], shift),
            );
        }
        for (name, o, n) in [
            ("merger.proj", 8, 8),
            ("merger.gate_proj", 16, 8),
            ("merger.up_proj", 16, 8),
            ("merger.down_proj", 8, 16),
        ] {
            shapes.insert(format!("{name}.weight"), (vec![o, n], 0.));
            shapes.insert(format!("{name}.bias"), (vec![o], 0.));
        }
        let mut seed = 319u64;
        let weights = shapes
            .into_iter()
            .map(|(name, (shape, shift))| {
                let data = (0..shape.iter().product::<i32>())
                    .map(|_| {
                        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                        ((seed >> 40) as f32 / 8_388_608. - 1.) * 0.25 + shift
                    })
                    .collect::<Vec<_>>();
                (
                    name,
                    Array::from_slice(&data, &shape)
                        .as_dtype(Dtype::Bfloat16)
                        .unwrap(),
                )
            })
            .collect();
        (
            Config {
                depth: 2,
                hidden_size: 8,
                num_heads: 2,
                patch_size: 1,
                temporal_patch_size: 1,
                spatial_merge_size: 2,
                out_hidden_size: 8,
                rms_norm_eps: 1e-5,
                swiglu_limit: 0.3,
            },
            weights,
        )
    }
    #[test]
    fn glm_vision_plan_matches_original_bits() -> anyhow::Result<()> {
        let (c, w) = fixture();
        let encoder = Encoder::from_weights(crate::metal_context::execution_context(), c, w)?;
        for (gh, gw, expected) in [
            (
                2,
                2,
                vec![
                    3171024896, 1020723200, 3186884608, 1045168128, 3176398848, 1042939904,
                    3174105088, 3183280128,
                ],
            ),
            (
                2,
                4,
                vec![
                    3171024896, 1020788736, 3186950144, 1045168128, 3176398848, 1042939904,
                    3174039552, 3183214592, 3171024896, 1018036224, 3186819072, 1045233664,
                    3175153664, 1043202048, 3173842944, 3183017984,
                ],
            ),
        ] {
            let input = (0..gh * gw * 3)
                .map(|i| ((i * 7 % 31) as f32 - 15.) / 16.)
                .collect::<Vec<_>>();
            for _ in 0..2 {
                let (out, _, _) = encoder.encode_pixels(&input, gh, gw)?;
                assert_eq!(
                    out.as_dtype(Dtype::Float32)?
                        .as_slice::<f32>()
                        .iter()
                        .map(|v| v.to_bits())
                        .collect::<Vec<_>>(),
                    expected
                );
            }
        }
        Ok(())
    }
}
