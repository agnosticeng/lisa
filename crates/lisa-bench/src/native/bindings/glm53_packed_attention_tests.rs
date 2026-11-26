#![cfg(test)]
use super::*;
use crate::bindings::deepseek41_moe::tests::tensor;
use lisa_gguf::TensorType as T;
use lisa_metal::Dtype;
#[path = "glm53_packed_attention_oracle.rs"]
mod oracle;
fn values(shape: &[i32], seed: usize) -> Array {
    Array::from_slice(
        &(0..shape.iter().product::<i32>() as usize)
            .map(|i| ((i * 17 + seed) % 127) as f32 / 151. - 0.3)
            .collect::<Vec<_>>(),
        shape,
    )
}
fn equal(a: &Array, b: &Array) -> Result<()> {
    assert_eq!(a.shape(), b.shape());
    assert_eq!(a.dtype(), b.dtype());
    assert_eq!(a.to_vec1::<u32>()?, b.to_vec1::<u32>()?);
    Ok(())
}
#[test]
fn packed_attention_plans_preserve_short_sparse_and_wide_bits() -> Result<()> {
    let weight = |shape: &[i32], seed| -> Result<Projection> {
        Ok(Projection::Packed(
            tensor(T::Q8_0, shape, seed)?.into_prepared()?,
        ))
    };
    let make_indexer = || -> Result<Indexer> {
        Ok(Indexer {
            context: crate::metal_context::execution_context(),
            wq_b_w: values(&[4096, 1536], 1).as_dtype(Dtype::Bfloat16)?,
            wk_w: values(&[128, 4096], 7).as_dtype(Dtype::Bfloat16)?,
            k_norm_w: values(&[128], 11),
            k_norm_b: values(&[128], 17),
            weights_w: values(&[32, 4096], 23).as_dtype(Dtype::Bfloat16)?,
            gate_w: values(&[128, 4096], 31).as_dtype(Dtype::Bfloat16)?,
            ape: values(&[4, 128], 41).as_dtype(Dtype::Bfloat16)?,
            heads: 32,
            head_dim: 128,
            topk: 2048,
            kpool: 4,
            always_tail: true,
        })
    };
    let old = oracle::GgufDsa {
        qa: weight(&[1536, 4096], 1)?,
        qb: weight(&[16384, 1536], 3)?,
        kva: weight(&[512, 4096], 5)?,
        output: weight(&[4096, 16384], 7)?,
        key: tensor(T::Q8_0, &[64, 512, 256], 11)?,
        value: tensor(T::Q8_0, &[64, 256, 512], 13)?,
        qnorm: values(&[1536], 17),
        kvnorm: values(&[512], 19),
    };
    let clone = |p: &Projection| match p {
        Projection::Plain(a) => Projection::Plain(a.clone()),
        Projection::Packed(a) => Projection::Packed(a.clone()),
    };
    let new = GgufDsa {
        context: crate::metal_context::execution_context(),
        qa: clone(&old.qa),
        qb: clone(&old.qb),
        kva: clone(&old.kva),
        output: clone(&old.output),
        key: old.key.clone().into_prepared()?,
        value: old.value.clone().into_prepared()?,
        qnorm: old.qnorm.clone(),
        kvnorm: old.kvnorm.clone(),
        indexer: make_indexer()?,
    };
    let x = values(&[1, 1, 4096], 31);
    let a = old.project(&x)?;
    let b = new.project(&x)?;
    equal(&a.0, &b.0)?;
    equal(&a.1, &b.1)?;
    equal(&a.2, &b.2)?;
    for n in [1, 7, 2051, 2052, 4097] {
        let latent = values(&[1, n, 512], 37).as_dtype(Dtype::Float16)?;
        for dense in [false, true] {
            let ids = Array::from_slice(
                &(0..n)
                    .map(|i| if !dense && i % 5 == 4 { -1 } else { i })
                    .collect::<Vec<_>>(),
                &[1, 1, n],
            );
            let a = old.attend(&a.1, &latent, &ids, dense)?;
            let b = new.attend(&b.1, &latent, &ids, dense)?;
            equal(&a, &b)?;
        }
    }
    let (mut ta, mut tb) = (
        IndexerTape::new(crate::metal_context::execution_context()),
        IndexerTape::new(crate::metal_context::execution_context()),
    );
    for position in 0usize..8 {
        let start = position.saturating_sub(3usize);
        let a = old.mtp(&x, &mut ta, position, start)?;
        let b = new.mtp(&x, &mut tb, position, start)?;
        equal(&a, &b)?;
        equal(
            &ta.compact_slice(0, position + 1)
                .unwrap()
                .as_dtype(Dtype::Float32)?,
            &tb.compact_slice(0, position + 1)
                .unwrap()
                .as_dtype(Dtype::Float32)?,
        )?;
    }
    Ok(())
}
