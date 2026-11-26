//! Application construction of portable DenseMlp bindings.
#[cfg(test)]
use lisa_metal::Array;
pub struct DenseMlp(
    pub lisa_engine::bindings::shared::dense::DenseMlp<lisa_metal::device::MetalDevice>,
);
impl std::ops::Deref for DenseMlp {
    type Target = lisa_engine::bindings::shared::dense::DenseMlp<lisa_metal::device::MetalDevice>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl std::ops::DerefMut for DenseMlp {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}
impl DenseMlp {
    pub fn load(
        w: &mut crate::core::loader::Weights,
        prefix: &str,
        gs: i32,
        bits: i32,
    ) -> anyhow::Result<Self> {
        Ok(Self(lisa_engine::bindings::shared::dense::DenseMlp::load(
            crate::metal_context::execution_context(),
            w,
            prefix,
            gs,
            bits,
        )?))
    }
}
#[cfg(test)]
use crate::core::{
    loader::Weights,
    quant::{QuantizedLinear, QuantizedLinearExecution},
};
#[cfg(test)]
mod model_plan_tests {
    use super::*;
    use lisa_metal::Dtype;

    #[test]
    fn wide_joined_27b_mlp_matches_separate_projections_bitwise() -> anyhow::Result<()> {
        let (h, f, group, bits) = (5120, 17408, 64, 4);
        let projection = |n: i32, k: i32, salt: u32| QuantizedLinear {
            weight: Array::from_slice(
                &(0..n * k / 8)
                    .map(|i| (i as u32).wrapping_mul(0x9e3779b9).wrapping_add(salt))
                    .collect::<Vec<_>>(),
                &[n, k / 8],
            ),
            scales: Array::from_slice(
                &(0..n * k / group)
                    .map(|i| half::bf16::from_f32((1 + i % 7) as f32 / 1024.))
                    .collect::<Vec<_>>(),
                &[n, k / group],
            ),
            biases: Array::from_slice(
                &(0..n * k / group)
                    .map(|i| half::bf16::from_f32(-((1 + i % 7) as f32) / 128.))
                    .collect::<Vec<_>>(),
                &[n, k / group],
            ),
            group_size: group,
            bits,
        };
        let (gate, up, down) = (
            projection(f, h, 0x89abcdef),
            projection(f, h, 0x12347890),
            projection(h, f, 0xfedcba98),
        );
        let mut weights = Weights::new();
        for (name, p) in [("gate_proj", &gate), ("up_proj", &up), ("down_proj", &down)] {
            weights.insert(format!("mlp.{name}.weight"), p.weight.clone());
            weights.insert(format!("mlp.{name}.scales"), p.scales.clone());
            weights.insert(format!("mlp.{name}.biases"), p.biases.clone());
        }
        let mlp = DenseMlp::load(&mut weights, "mlp", group, bits)?;
        let stream = lisa_metal::Stream::thread_local_or_default();
        let bits_of = |a: &Array| -> anyhow::Result<Vec<u32>> {
            Ok(a.as_dtype(Dtype::Float32)?
                .as_slice::<f32>()
                .iter()
                .map(|v| v.to_bits())
                .collect())
        };
        for verify in [false, true] {
            let _scope = verify.then(lisa_metal::ops::enter_verify_splitk_scope);
            for (b, s) in (2..=9)
                .map(|s| (1, s))
                .chain([(1, 1), (2, 1), (2, 3), (1, 16)])
            {
                let x = Array::from_slice(
                    &(0..b * s * h)
                        .map(|i| half::bf16::from_f32(((i * 37 % 257) - 128) as f32 / 128.))
                        .collect::<Vec<_>>(),
                    &[b, s, h],
                );
                let g = gate.forward(&x)?;
                let u = up.forward(&x)?;
                let a = lisa_metal::kernels::swiglu2(&g, &u, &stream)
                    .expect("BF16 SwiGLU oracle")
                    .reshape(g.shape())?;
                let expected = down.forward(&a)?;
                let actual = mlp.forward(&x)?;
                assert_eq!(actual.shape(), expected.shape());
                assert_eq!(
                    bits_of(&actual)?,
                    bits_of(&expected)?,
                    "verify={verify}, B={b}, S={s}"
                );
            }
        }
        Ok(())
    }

    #[test]
    fn dense_plan_matches_prior_projection_and_activation_chain() -> anyhow::Result<()> {
        let stream = lisa_metal::Stream::thread_local_or_default();
        for bits in [2, 4, 8] {
            let (h, f, group) = (128, 256, 64);
            let projection = |n: i32, k: i32, salt: u32| QuantizedLinear {
                weight: Array::from_slice(
                    &(0..n * k * bits / 32)
                        .map(|i| (i as u32).wrapping_mul(0x1234567).wrapping_add(salt))
                        .collect::<Vec<_>>(),
                    &[n, k * bits / 32],
                ),
                scales: Array::from_slice(
                    &vec![1.0f32 / 64.0; (n * k / group) as usize],
                    &[n, k / group],
                )
                .as_dtype(Dtype::Bfloat16)
                .unwrap(),
                biases: Array::from_slice(
                    &vec![-0.125f32; (n * k / group) as usize],
                    &[n, k / group],
                )
                .as_dtype(Dtype::Bfloat16)
                .unwrap(),
                group_size: group,
                bits,
            };
            let (gate, up, down) = (
                projection(f, h, 0x89abcdef),
                projection(f, h, 0x12347890),
                projection(h, f, 0xfedcba98),
            );
            let mut weights = Weights::new();
            for (name, p) in [("gate_proj", &gate), ("up_proj", &up), ("down_proj", &down)] {
                weights.insert(format!("mlp.{name}.weight"), p.weight.clone());
                weights.insert(format!("mlp.{name}.scales"), p.scales.clone());
                weights.insert(format!("mlp.{name}.biases"), p.biases.clone());
            }
            let mlp = DenseMlp::load(&mut weights, "mlp", group, bits)?;
            let joined = QuantizedLinear {
                weight: Array::concatenate(&[&gate.weight, &up.weight], 0)?,
                scales: Array::concatenate(&[&gate.scales, &up.scales], 0)?,
                biases: Array::concatenate(&[&gate.biases, &up.biases], 0)?,
                group_size: group,
                bits,
            };
            for (b, s) in [(1, 1), (2, 1), (1, 3), (1, 7), (2, 3)] {
                let x = Array::from_slice(
                    &(0..b * s * h)
                        .map(|i| ((i % 31) as f32 - 15.0) / 32.0)
                        .collect::<Vec<_>>(),
                    &[b, s, h],
                )
                .as_dtype(Dtype::Bfloat16)?;
                let expected = if s == 1 {
                    let gu = joined.forward(&x)?;
                    let a = lisa_metal::kernels::gated_pointwise::swiglu2_packed(&gu, &stream)
                        .expect("packed oracle kernel");
                    down.forward(&a)?
                } else {
                    let g = gate.forward(&x)?;
                    let u = up.forward(&x)?;
                    let a = match lisa_metal::kernels::swiglu2(&g, &u, &stream) {
                        Some(a) => a.reshape(g.shape())?,
                        None => lisa_metal::nn::silu(&g)?.multiply(&u)?,
                    };
                    down.forward(&a)?
                };
                let actual = mlp.forward(&x)?;
                assert_eq!(actual.shape(), expected.shape(), "bits={bits},B={b},S={s}");
                let bits_of = |a: &Array| -> anyhow::Result<Vec<u32>> {
                    Ok(a.as_dtype(Dtype::Float32)?
                        .as_slice::<f32>()
                        .iter()
                        .map(|v| v.to_bits())
                        .collect())
                };
                assert_eq!(
                    bits_of(&actual)?,
                    bits_of(&expected)?,
                    "bits={bits},B={b},S={s}"
                );
            }
        }
        Ok(())
    }
}
