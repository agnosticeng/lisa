#![cfg(test)]
use super::*;
use crate::bindings::deepseek41_moe::tests::{counts, difference, projection_check, tensor};
use crate::core::gguf_resident::PackedTensor;
use lisa_gguf::TensorType as T;
use lisa_metal::{Dtype, Stream, ops::indexing::IndexOp};
#[path = "glm53_moe_oracle.rs"]
mod oracle;
fn projection(w: &PackedTensor) -> Result<Projection> {
    Ok(if matches!(w.info.kind, T::F32 | T::F16 | T::BF16) {
        Projection::Plain(w.plain_view()?)
    } else {
        Projection::Packed(w.clone())
    })
}
#[test]
fn packed_glm_ffn_preserves_dense_and_sparse_parts_and_combination() -> Result<()> {
    let stream = Stream::gpu();
    for (layer, kinds) in [
        (0, [T::Q8_0, T::Q8_0, T::Q8_0]),
        (3, [T::Q8_0, T::Q8_0, T::Q8_0]),
        (3, [T::BF16, T::BF16, T::Q4_K]),
    ] {
        let sparse = layer >= 3;
        let mid = if sparse { 2048 } else { 12288 };
        let experts = 16;
        let shapes = [
            vec![mid, 4096],
            vec![mid, 4096],
            vec![4096, mid],
            vec![experts, 4096],
            vec![experts],
            vec![experts, 2048, 4096],
            vec![experts, 2048, 4096],
            vec![experts, 4096, 2048],
        ];
        let formats = [
            kinds[0],
            kinds[1],
            kinds[2],
            T::F32,
            T::F32,
            T::IQ2_XXS,
            T::IQ2_XXS,
            T::Q2_K,
        ];
        let weights = shapes
            .iter()
            .zip(formats)
            .take(if sparse { 8 } else { 3 })
            .enumerate()
            .map(|(i, (s, k))| tensor(k, s, 301 + i as u32))
            .collect::<Result<Vec<_>>>()?;
        let old = oracle::from_weights(&weights, sparse)?;
        let mut arrays = Vec::new();
        let mut descriptors = Vec::new();
        for (i, w) in weights.iter().enumerate() {
            if i < 3 {
                let p = projection(w)?;
                descriptors.push(p.descriptor()?);
                arrays.push((*p).clone());
            } else if i < 5 {
                arrays.push(w.plain_view()?);
            } else {
                arrays.push(w.packed_bytes().clone());
            }
        }
        let new = Mlp::bound(
            crate::metal_context::execution_context(),
            arrays,
            model::Mlp::for_layer(layer, descriptors.try_into().unwrap()),
        );
        for round in 0..2 {
            let data = (0..4096)
                .map(|i| ((i * 37 + round * 17) % 4093) as f32 / 4093. - 0.5)
                .collect::<Vec<_>>();
            let x = Array::from_slice(&data, &[1, 1, 4096]);
            stream.runtime().commands.flush_and_wait()?;
            let before = counts();
            let a = old.parts(&x)?;
            stream.runtime().commands.flush_and_wait()?;
            let middle = counts();
            let b = new.parts(&x)?;
            stream.runtime().commands.flush_and_wait()?;
            let after = counts();
            assert_eq!(a.1.to_vec1::<u32>()?, b.1.to_vec1::<u32>()?);
            assert_eq!(a.0.is_some(), b.0.is_some());
            if let (Some(a), Some(b)) = (a.0, b.0) {
                assert_eq!(a.to_vec1::<u32>()?, b.to_vec1::<u32>()?);
            }
            if !after.is_empty() {
                let a = difference(&before, &middle);
                let b = difference(&middle, &after);
                assert_eq!(a, b, "layer {layer} parts dispatch order");
                eprintln!(
                    "GLM FFN layer={layer}: {} dispatches, {} variants",
                    b.values().sum::<u64>(),
                    b.len()
                );
            }
            projection_check(&stream, || old.forward(&x), || new.forward(&x))?;
        }
    }
    Ok(())
}
#[test]
fn packed_glm_projection_preserves_bf16_gemv_gemm_and_strided_inputs() -> Result<()> {
    let stream = Stream::gpu();
    for kind in [T::F32, T::F16, T::BF16, T::Q8_0, T::Q4_K] {
        let w = tensor(kind, &[32, 256], 431)?;
        let old = if matches!(kind, T::F32 | T::F16 | T::BF16) {
            oracle::projection::Projection::Plain(w.plain_view()?)
        } else {
            oracle::projection::Projection::Packed(w.clone())
        };
        let new = projection(&w)?;
        for (batch, rows) in [(1, 1), (1, 8), (1, 9), (2, 4)] {
            if kind == T::Q8_0 && batch * rows > 8 {
                continue;
            }
            let x = Array::from_slice(
                &(0..batch * rows * 512)
                    .map(|i| (i % 4093) as f32 / 4093. - 0.5)
                    .collect::<Vec<_>>(),
                &[batch, rows, 512],
            );
            let x = x.index((.., .., 128..384));
            assert_eq!(x.dtype(), Dtype::Float32);
            projection_check(&stream, || old.forward(&x), || new.forward(&x))?;
        }
    }
    Ok(())
}
