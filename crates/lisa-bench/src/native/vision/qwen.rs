//! Native conformance fixtures for the portable qwen encoder.
pub type Encoder = lisa_engine::vision::qwen::Encoder<lisa_metal::device::MetalDevice>;
#[cfg(test)]
use crate::core::program;
#[cfg(test)]
use lisa_metal::Array;
#[cfg(test)]
use lisa_models::vision::qwen as definition;
pub use lisa_models::vision::qwen::Config;
#[cfg(test)]
use serde_json::Value;
#[cfg(test)]
use std::{cell::RefCell, collections::HashMap};
#[cfg(test)]
mod plan_tests {
    use super::*;
    use lisa_metal::Dtype;
    #[test]
    fn qwen_vision_plan_matches_original_bits() {
        let mut seed = 319u64;
        let mut tensor = |shape: &[i32], shift: f32| {
            let data = (0..shape.iter().product::<i32>())
                .map(|_| {
                    seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                    ((seed >> 40) as f32 / 8_388_608.0 - 1.0) * 0.25 + shift
                })
                .collect::<Vec<_>>();
            Array::from_slice(&data, shape)
                .as_dtype(Dtype::Bfloat16)
                .unwrap()
        };
        let mut weights = HashMap::new();
        // Sorted keys pin fixture generation independently of loading order.
        let mut shapes = std::collections::BTreeMap::<String, (Vec<i32>, f32)>::new();
        shapes.insert("patch_embed.proj.weight".into(), (vec![8, 3], 0.));
        shapes.insert("patch_embed.proj.bias".into(), (vec![8], 0.));
        shapes.insert("pos_embed.weight".into(), (vec![4, 8], 0.));
        for i in 0..2 {
            for name in ["norm1", "norm2"] {
                for (suffix, shift) in [("weight", 1.), ("bias", 0.)] {
                    shapes.insert(format!("blocks.{i}.{name}.{suffix}"), (vec![8], shift));
                }
            }
            for (name, o, n) in [
                ("attn.qkv", 24, 8),
                ("attn.proj", 8, 8),
                ("mlp.linear_fc1", 16, 8),
                ("mlp.linear_fc2", 8, 16),
            ] {
                shapes.insert(format!("blocks.{i}.{name}.weight"), (vec![o, n], 0.));
                shapes.insert(format!("blocks.{i}.{name}.bias"), (vec![o], 0.));
            }
        }
        for (suffix, shift) in [("weight", 1.), ("bias", 0.)] {
            shapes.insert(format!("merger.norm.{suffix}"), (vec![8], shift));
        }
        for (name, o, n) in [("merger.linear_fc1", 16, 32), ("merger.linear_fc2", 8, 16)] {
            shapes.insert(format!("{name}.weight"), (vec![o, n], 0.));
            shapes.insert(format!("{name}.bias"), (vec![o], 0.));
        }
        for (name, (shape, shift)) in shapes {
            weights.insert(name, tensor(&shape, shift));
        }
        let schema = weights
            .iter()
            .map(|(k, v)| (k.clone(), v.shape().to_vec()))
            .collect();
        let encoder = Encoder {
            context: crate::metal_context::execution_context(),
            config: Config {
                depth: 2,
                hidden_size: 8,
                num_heads: 2,
                patch_size: 1,
                temporal_patch_size: 1,
                spatial_merge_size: 2,
                num_position_embeddings: 4,
                out_hidden_size: 8,
                deepstack_visual_indexes: vec![],
            },
            processor: Value::Null,
            weights,
            schema,
            prepared: RefCell::new(None),
        };
        for (gh, gw, expected) in [
            (
                2,
                2,
                vec![
                    1040580608, 1055260672, 1049231360, 3196256256, 1043857408, 1040252928,
                    3187933184, 3204513792,
                ],
            ),
            (
                2,
                4,
                vec![
                    1038352384, 1055981568, 1056964608, 3199991808, 1032716288, 1042087936,
                    1030488064, 3201499136, 1033240576, 1052901376, 1035862016, 1042415616,
                    1041760256, 1043202048, 3199336448, 3181772800,
                ],
            ),
        ] {
            let input = (0..gh * gw * 3)
                .map(|i| ((i * 7 % 31) as f32 - 15.) / 16.)
                .collect::<Vec<_>>();
            for _ in 0..2 {
                let (out, _, _) = encoder.encode_pixels(&input, gh, gw).unwrap();
                assert_eq!(
                    out.as_dtype(Dtype::Float32)
                        .unwrap()
                        .as_slice::<f32>()
                        .iter()
                        .map(|v| v.to_bits())
                        .collect::<Vec<_>>(),
                    expected
                );
            }
        }
    }
}
