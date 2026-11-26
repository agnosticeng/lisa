//! Application aliases and native conformance oracles for portable normalization bindings.
use crate::core::loader::TensorSource;
use lisa_engine::bindings::shared::normalization as portable;
use lisa_metal::Array;
#[cfg(test)]
use lisa_metal::Dtype;
fn context() -> lisa_engine::execution::DeviceContext<lisa_metal::device::MetalDevice> {
    crate::metal_context::execution_context()
}
fn native<T>(result: anyhow::Result<T>) -> lisa_metal::error::Result<T> {
    result.map_err(|e| lisa_metal::error::Error::msg(e.to_string()))
}
pub type RmsNorm = portable::RmsNorm<Array>;
pub type RmsNormGated = portable::RmsNormGated<Array>;
pub struct Rotary(pub portable::Rotary);
impl std::ops::Deref for Rotary {
    type Target = portable::Rotary;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl Clone for Rotary {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}
pub trait RmsNormExecution {
    fn load<S: TensorSource>(
        src: &mut S,
        name: &str,
        eps: f32,
        group_size: Option<usize>,
    ) -> anyhow::Result<Self>
    where
        Self: Sized;
    fn forward(&self, x: &Array) -> lisa_metal::error::Result<Array>;
    fn forward_residual_add(
        &self,
        x: &Array,
        r: &Array,
    ) -> lisa_metal::error::Result<(Array, Array)>;
}
impl RmsNormExecution for RmsNorm {
    fn load<S: TensorSource>(
        src: &mut S,
        name: &str,
        eps: f32,
        group_size: Option<usize>,
    ) -> anyhow::Result<Self> {
        Self::load_on(&context(), src, name, eps, group_size)
    }
    fn forward(&self, x: &Array) -> lisa_metal::error::Result<Array> {
        native(self.forward_on(&context(), x))
    }
    fn forward_residual_add(
        &self,
        x: &Array,
        r: &Array,
    ) -> lisa_metal::error::Result<(Array, Array)> {
        native(self.forward_residual_add_on(&context(), x, r))
    }
}
pub trait RmsNormGatedExecution {
    fn load<S: TensorSource>(src: &mut S, name: &str, eps: f32) -> anyhow::Result<Self>
    where
        Self: Sized;
    fn forward(&self, x: &Array, g: &Array) -> lisa_metal::error::Result<Array>;
}
impl RmsNormGatedExecution for RmsNormGated {
    fn load<S: TensorSource>(src: &mut S, name: &str, eps: f32) -> anyhow::Result<Self> {
        Self::load_on(&context(), src, name, eps)
    }
    fn forward(&self, x: &Array, g: &Array) -> lisa_metal::error::Result<Array> {
        native(self.forward_on(&context(), x, g))
    }
}
impl Rotary {
    pub fn new(d: usize, b: f32) -> Self {
        Self(portable::Rotary::new(d, b))
    }
    pub fn cos_sin(&self, p: &Array) -> lisa_metal::error::Result<(Array, Array)> {
        native(self.0.cos_sin(&context(), p))
    }
    pub fn cos_sin_multimodal(&self, p: &Array) -> lisa_metal::error::Result<(Array, Array)> {
        native(self.0.cos_sin_multimodal(&context(), p))
    }
}
pub fn bf16_scalar(v: f32) -> Array {
    portable::bf16_scalar(&context(), v).expect("BF16 scalar import")
}
pub fn bf16_silu(x: &Array) -> lisa_metal::error::Result<Array> {
    native(portable::bf16_silu(&context(), x))
}
pub fn bf16_logaddexp0(x: &Array) -> lisa_metal::error::Result<Array> {
    native(portable::bf16_logaddexp0(&context(), x))
}
pub fn rms_row_exact(
    x: &Array,
    w: Option<&Array>,
    e: f32,
    h: usize,
) -> lisa_metal::error::Result<Array> {
    native(portable::rms_row_exact(&context(), x, w, e, h))
}
pub fn rope_partial(x: &Array, c: &Array, s: &Array) -> lisa_metal::error::Result<Array> {
    native(portable::rope_partial(&context(), x, c, s))
}
pub fn positions(offset: usize, count: usize) -> lisa_metal::error::Result<Array> {
    native(portable::positions(&context(), offset, count))
}
#[cfg(test)]
use lisa_metal::{ops, ops::indexing::Ellipsis};
#[cfg(test)]
fn grouped_rms_plan(
    x: &Array,
    w: Option<&Array>,
    e: f32,
    h: usize,
    f: bool,
) -> lisa_metal::error::Result<Array> {
    native(portable::grouped_rms_plan(&context(), x, w, e, h, f))
}
#[cfg(test)]
mod rope_ab {
    use super::*;
    #[test]
    fn rope_partial_ab() {
        let (b, h, s, d, dims) = (1i32, 24i32, 1i32, 256i32, 64i32);
        let mut seed = 555u64;
        let mut rnd = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((seed >> 40) as f32 / 8_388_608.0) - 1.0
        };
        let xv: Vec<f32> = (0..(b * h * s * d) as usize)
            .map(|_| half::bf16::from_f32(rnd()).to_f32())
            .collect();
        let cv: Vec<f32> = (0..(b * s * dims) as usize).map(|_| rnd()).collect();
        let sv: Vec<f32> = (0..(b * s * dims) as usize).map(|_| rnd()).collect();
        let x = Array::from_slice(&xv, &[b, h, s, d])
            .as_dtype(Dtype::Bfloat16)
            .unwrap();
        let cos = Array::from_slice(&cv, &[b, s, dims])
            .as_dtype(Dtype::Float32)
            .unwrap()
            .expand_dims(1)
            .unwrap();
        let sin = Array::from_slice(&sv, &[b, s, dims])
            .as_dtype(Dtype::Float32)
            .unwrap()
            .expand_dims(1)
            .unwrap();
        let out = rope_partial(&x, &cos, &sin).unwrap();
        let f = out.as_dtype(Dtype::Float32).unwrap();
        let _a = f.as_slice::<f32>();
    }
}

#[cfg(test)]
mod plan_parity {
    use super::*;
    use lisa_metal::ops::indexing::IndexOp;
    fn equal(a: &Array, b: &Array) {
        assert_eq!(a.shape(), b.shape());
        assert_eq!(
            a.as_dtype(Dtype::Float32)
                .unwrap()
                .as_slice::<f32>()
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            b.as_dtype(Dtype::Float32)
                .unwrap()
                .as_slice::<f32>()
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>()
        );
    }
    #[test]
    fn shared_plans_preserve_original_operation_chains() {
        for dtype in [Dtype::Float16, Dtype::Bfloat16, Dtype::Float32] {
            for shape in [[1, 1, 128], [2, 3, 128], [1, 7, 256]] {
                let data: Vec<f32> = (0..shape.iter().product::<i32>())
                    .map(|i| ((i * 13 % 101) as f32 - 50.0) / 37.0)
                    .collect();
                let x = Array::from_slice(&data, &shape).as_dtype(dtype).unwrap();
                let r = x.multiply(Array::from_f32_as(0.25, dtype)).unwrap();
                let weight = Array::from_slice(&vec![0.875f32; shape[2] as usize], &[shape[2]])
                    .as_dtype(dtype)
                    .unwrap();
                let norm = RmsNorm {
                    weight: weight.clone(),
                    eps: 1e-6,
                    group_size: None,
                };
                equal(
                    &norm.forward(&x).unwrap(),
                    &lisa_metal::fast::rms_norm(&x, Some(&weight), 1e-6).unwrap(),
                );
                let (s, n) = norm.forward_residual_add(&x, &r).unwrap();
                let (s0, n0) = lisa_metal::fast::fused_add_rms_norm(&x, &r, &weight, 1e-6).unwrap();
                let flat_r = r.reshape(&[-1, shape[2]]).unwrap();
                let (flat_s, flat_n) = norm.forward_residual_add(&x, &flat_r).unwrap();
                equal(&flat_s, &s0);
                equal(&flat_n, &n0);
                equal(&s, &s0);
                equal(&n, &n0);
                let gated = RmsNormGated {
                    weight: weight.clone(),
                    eps: 1e-6,
                };
                let g = ops::sigmoid(r.as_dtype(Dtype::Float32).unwrap()).unwrap();
                let expected = (g * lisa_metal::fast::rms_norm(&x, Some(&weight), 1e-6)
                    .unwrap()
                    .as_dtype(Dtype::Float32)
                    .unwrap())
                .as_dtype(dtype)
                .unwrap();
                equal(&gated.forward(&x, &r).unwrap(), &expected);
                for d in [64, shape[2]] {
                    let cos = Array::from_slice(&vec![0.6f32; d as usize], &[1, 1, d]);
                    let sin = Array::from_slice(&vec![0.8f32; d as usize], &[1, 1, d]);
                    let rotated = x.index((Ellipsis, 0..d));
                    let first = rotated.index((Ellipsis, 0..d / 2));
                    let second = rotated.index((Ellipsis, d / 2..d));
                    let swapped = ops::concatenate(&[&(-second), &first], -1).unwrap();
                    let out = rotated
                        .multiply(cos.as_dtype(dtype).unwrap())
                        .unwrap()
                        .add(swapped.multiply(sin.as_dtype(dtype).unwrap()).unwrap())
                        .unwrap();
                    let expected = if d == shape[2] {
                        out
                    } else {
                        ops::concatenate(&[&out, &x.index((Ellipsis, d..shape[2]))], -1).unwrap()
                    };
                    equal(&rope_partial(&x, &cos, &sin).unwrap(), &expected);
                }
            }
        }
    }
}

#[cfg(test)]
mod frequency_plan_parity {
    use super::*;
    #[test]
    fn rotary_frequency_plans_match_native_text_and_multimodal_bits() -> anyhow::Result<()> {
        for dimensions in [2, 64, 128] {
            for base in [10000., 1000000.] {
                for multimodal in [false, true] {
                    let shape = if multimodal {
                        vec![2, 7, dimensions / 2]
                    } else {
                        vec![2, 7]
                    };
                    let values = (0..shape.iter().product::<i32>())
                        .map(|i| (i * 11 % 1009) as f32)
                        .collect::<Vec<_>>();
                    let positions = Array::from_slice(&values, &shape);
                    let even = (0..dimensions)
                        .step_by(2)
                        .map(|i| i as f32)
                        .collect::<Vec<_>>();
                    let even = Array::from_slice(&even, &[even.len() as i32]);
                    let inverse = ops::exp(&(even * (-f32::ln(base) / dimensions as f32)))?;
                    let pos = positions.as_dtype(Dtype::Float32)?;
                    let pos = if multimodal {
                        pos
                    } else {
                        pos.expand_dims(-1)?
                    };
                    let freq = pos.multiply(&inverse)?;
                    let emb = ops::concatenate(&[&freq, &freq], -1)?;
                    let expected = [ops::cos(&emb)?, ops::sin(&emb)?];
                    let rotary = Rotary::new(dimensions as usize, base);
                    for _ in 0..2 {
                        let (cos, sin) = if multimodal {
                            rotary.cos_sin_multimodal(&positions)?
                        } else {
                            rotary.cos_sin(&positions)?
                        };
                        for (got, want) in [cos, sin].iter().zip(&expected) {
                            assert_eq!(got.shape(), want.shape());
                            assert_eq!(
                                got.as_slice::<f32>()
                                    .iter()
                                    .map(|v| v.to_bits())
                                    .collect::<Vec<_>>(),
                                want.as_slice::<f32>()
                                    .iter()
                                    .map(|v| v.to_bits())
                                    .collect::<Vec<_>>()
                            );
                        }
                    }
                }
            }
        }
        Ok(())
    }
    #[test]
    fn softplus_plan_preserves_all_bf16_inputs() -> anyhow::Result<()> {
        let values = (0..=u16::MAX)
            .map(half::bf16::from_bits)
            .collect::<Vec<_>>();
        let x = Array::from_slice(&values, &[65536]);
        let zero = bf16_scalar(0.);
        let max = ops::maximum(&x, &zero)?;
        let min = ops::minimum(&x, &zero)?;
        let d = min - &max;
        let e = Array::new(lisa_metal::mlx_rt::exp(d.t.device(), &d.t)?);
        let l = Array::new(lisa_metal::mlx_rt::log1p(e.t.device(), &e.t)?);
        let expected = max.add(&l)?.as_dtype(Dtype::Float32)?;
        for _ in 0..2 {
            let got = bf16_logaddexp0(&x)?.as_dtype(Dtype::Float32)?;
            assert_eq!(
                got.as_slice::<f32>()
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>(),
                expected
                    .as_slice::<f32>()
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>()
            );
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "norm_plan_tests.rs"]
mod norm_plan_tests;
