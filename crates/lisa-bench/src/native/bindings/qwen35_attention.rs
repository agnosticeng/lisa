//! Application construction and native oracles for portable attention bindings.
use crate::core::{
    cache::FullAttentionCache, graph::PreparedGraph, loader::TensorSource, norm::RmsNorm,
    quant::QuantizedLinear,
};
use lisa_metal::{Array, device::MetalDevice};
pub struct Qwen35Attention(
    pub lisa_engine::bindings::qwen35_attention::Qwen35Attention<MetalDevice>,
);
impl std::ops::Deref for Qwen35Attention {
    type Target = lisa_engine::bindings::qwen35_attention::Qwen35Attention<MetalDevice>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl Qwen35Attention {
    pub fn load<S: TensorSource>(
        src: &mut S,
        prefix: &str,
        eps: f32,
        heads: usize,
        kv_heads: usize,
        head_dim: usize,
        gated: bool,
        group_size: i32,
        bits: i32,
    ) -> anyhow::Result<Self> {
        Ok(Self(
            lisa_engine::bindings::qwen35_attention::Qwen35Attention::load(
                crate::metal_context::execution_context(),
                src,
                prefix,
                eps,
                heads,
                kv_heads,
                head_dim,
                gated,
                group_size,
                bits,
            )?,
        ))
    }
}
#[cfg(test)]
fn join_qkv(
    q: &mut QuantizedLinear,
    k: &mut QuantizedLinear,
    v: &mut QuantizedLinear,
) -> anyhow::Result<Option<QuantizedLinear>> {
    lisa_engine::bindings::qwen35_attention::join_qkv(
        &crate::metal_context::execution_context(),
        q,
        k,
        v,
    )
}
#[cfg(test)]
use crate::core::{norm::RmsNormExecution, quant::QuantizedLinearExecution};
#[cfg(test)]
mod grouped_projection_tests {
    use super::*;
    use half::bf16;
    use lisa_metal::ops::indexing::IndexOp;

    #[test]
    fn joined_qkv_preserves_projection_bits_and_strided_views() {
        let mut seed = 31u32;
        let mut random = || {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            seed
        };
        for bits in [2, 4] {
            let mut make = |n: i32| QuantizedLinear {
                weight: Array::from_slice(
                    &(0..n * 5120 / (32 / bits))
                        .map(|_| random())
                        .collect::<Vec<_>>(),
                    &[n, 5120 / (32 / bits)],
                ),
                scales: Array::from_slice(
                    &(0..n * 80)
                        .map(|_| bf16::from_f32((random() % 127 + 1) as f32 / 8192.))
                        .collect::<Vec<_>>(),
                    &[n, 80],
                ),
                biases: Array::from_slice(
                    &(0..n * 80)
                        .map(|_| bf16::from_f32((random() % 63) as f32 / 1024. - 0.03))
                        .collect::<Vec<_>>(),
                    &[n, 80],
                ),
                group_size: 64,
                bits,
            };
            let (mut q, mut k, mut v) = (make(12288), make(1024), make(1024));
            let originals = [q.clone(), k.clone(), v.clone()];
            let joined = join_qkv(&mut q, &mut k, &mut v).unwrap().unwrap();
            for rows in [1, 2, 3, 6, 7, 8, 16, 64] {
                let input = Array::from_slice(
                    &(0..rows * 5120)
                        .map(|i| bf16::from_f32(((i * 37 % 521) as f32 - 260.) / 521.))
                        .collect::<Vec<_>>(),
                    &[1, rows, 5120],
                );
                let all = joined.forward(&input).unwrap();
                let mut offset = 0;
                for (original, view) in originals.iter().zip([&q, &k, &v]) {
                    let reference = original.forward(&input).unwrap().to_vec1::<bf16>().unwrap();
                    assert_eq!(
                        view.forward(&input).unwrap().to_vec1::<bf16>().unwrap(),
                        reference,
                        "packed weight view bits={bits} rows={rows} offset={offset}"
                    );
                    if rows <= 8 {
                        let selected = all
                            .index((.., .., offset..offset + original.dims_out() as i32))
                            .contiguous()
                            .unwrap();
                        assert_eq!(
                            selected.to_vec1::<bf16>().unwrap(),
                            reference,
                            "grouped output bits={bits} rows={rows} offset={offset}"
                        );
                    }
                    offset += original.dims_out() as i32;
                }
            }
        }
    }
}

#[cfg(test)]
#[path = "qwen35_attention_tests.rs"]
mod plan_parity;
