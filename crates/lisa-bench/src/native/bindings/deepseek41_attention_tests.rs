#![cfg(test)]
//! Original scalar graph, packed bytes and non-dyadic state are the oracle.
use super::*;
use crate::bindings::deepseek41_moe::tests::{counts, difference, tensor};
use lisa_gguf::TensorType as T;
use lisa_metal::kernels::ggml as native;
#[path = "deepseek41_attention_oracle.rs"]
mod oracle;
fn values(shape: &[i32], seed: usize) -> Array {
    let count = shape.iter().map(|&d| d as usize).product();
    Array::from_slice(
        &(0..count)
            .map(|i| ((i * 17 + seed) % 113) as f32 / 71. - 0.7)
            .collect::<Vec<_>>(),
        shape,
    )
}
fn equal(a: &Array, b: &Array) -> Result<()> {
    assert_eq!(a.shape(), b.shape());
    assert_eq!(a.dtype(), b.dtype());
    if a.dtype().size_of() == 2 {
        assert_eq!(a.to_vec1::<u16>()?, b.to_vec1::<u16>()?);
    } else {
        assert_eq!(a.to_vec1::<u32>()?, b.to_vec1::<u32>()?);
    }
    Ok(())
}
#[test]
fn scalar_attention_plans_preserve_cache_and_projection_bits() -> Result<()> {
    let stream = Stream::gpu();
    let geometry = [
        (T::Q8_0, vec![1280, 5120]),
        (T::F32, vec![1280]),
        (T::Q8_0, vec![512, 5120]),
        (T::F32, vec![512]),
        (T::Q8_0, vec![32768, 1280]),
        (T::Q5_K, vec![8192, 4096]),
        (T::Q8_0, vec![5120, 8192]),
        (T::F32, vec![64]),
        (T::Q8_0, vec![512, 5120]),
        (T::F32, vec![512, 5120]),
        (T::F32, vec![512]),
        (T::Q8_0, vec![128, 512]),
        (T::F32, vec![128]),
        (T::Q8_0, vec![4096, 1280]),
        (T::F32, vec![32, 5120]),
    ];
    let weights = geometry
        .iter()
        .enumerate()
        .map(|(i, (t, s))| tensor(*t, s, (i + 23) as u32))
        .collect::<Result<Vec<_>>>()?;
    let mut matrices = [None; 15];
    for i in 0..15 {
        if model::is_projection(i) {
            matrices[i] = Some(projection::matrix(&weights[i])?);
        }
    }
    let bound = weights
        .iter()
        .map(projection::bound)
        .collect::<Result<Vec<_>>>()?;
    let mut scratch = Scratch::new(&stream)?;
    let mut old_scratch = native::DecodeAttentionScratch::new(&stream)?;
    for (layer, ratio, compressor, gate, indexer, positions) in [
        (0, 0, false, false, false, vec![0, 127, 128]),
        (2, 2, true, true, true, vec![0, 1, 127, 128]),
        (20, 1, true, false, true, vec![0, 8191, 16384]),
        (21, 1, false, false, true, vec![16384]),
        (22, 1, false, false, false, vec![16384]),
    ] {
        let spec = model::Spec {
            layer,
            ratio,
            matrices,
            compressor,
            gate,
            indexer,
        };
        let new = Attention::bound(spec, bound.clone(), &stream)?;
        let old = oracle::fixture(&weights, layer, ratio, compressor, gate, indexer)?;
        for rows in [1, 3, 8] {
            let norm = values(&[rows, 5120], 29);
            let before = counts();
            let a = old.project(&norm, &stream)?;
            let between = counts();
            let b = new.project(&norm, &stream)?;
            let after = counts();
            for r in 0..rows as usize {
                let a = a.row(r)?;
                let b = b.row(r)?;
                oracle::compare_projections(&a, &b.qr, &b.kv, &b.q)?;
            }
            if std::env::var_os("LISA_TRACE").is_some() {
                assert_eq!(difference(&before, &between), difference(&between, &after));
            }
        }
        for pos in positions {
            let capacity = pos + 2;
            let mut a = State::new(crate::metal_context::device(), capacity)?;
            a.position = pos;
            let owner = lisa_models::deepseek41::state::owner(layer);
            a.raw[layer] = values(&[128, 512], 13);
            let c = &mut a.owners[owner];
            c.compressed = values(c.compressed.shape(), 19);
            c.index = values(c.index.shape(), 47);
            if c.index_bf16.is_some() {
                c.index_bf16 = Some(native::index_keys_bf16(&c.index, &stream)?);
            }
            c.previous_kv = values(&[1, 512], 31);
            c.previous_score = values(&[1, 512], 73);
            let mut b = a.snapshot()?;
            let p = Position::new(layer, ratio, pos)?;
            let mut ca = if p.compressed_rows > 16384 {
                Some(native::index_candidates(
                    &values(&[1, p.compressed_rows as i32], 97),
                    &stream,
                )?)
            } else {
                None
            };
            let mut cb = if let Some(c) = ca.as_ref() {
                Some(oracle::candidate_tensor(c))
            } else {
                None
            };
            let mut sa = if p.compressed_rows > 0 {
                Some(Array::from_slice(
                    &(0..p.selected as i32).collect::<Vec<_>>(),
                    &[p.selected as i32],
                ))
            } else {
                None
            };
            let mut sb = sa.clone();
            let norm = values(&[1, 5120], 91);
            let qa = old.project(&norm, &stream)?.row(0)?;
            let qb = new.project(&norm, &stream)?.row(0)?;
            let rope = rope_table(pos as u32, &stream)?;
            let before = counts();
            let x = old.forward(
                &norm,
                qa,
                &mut a,
                &mut sa,
                &mut ca,
                &mut old_scratch,
                &rope,
                None,
                &stream,
            )?;
            let between = counts();
            let y = new.forward(
                &norm,
                qb,
                &mut b,
                &mut sb,
                &mut cb,
                &mut scratch,
                &rope,
                None,
                &stream,
            )?;
            let after = counts();
            equal(&x, &y)?;
            equal(&a.raw[layer], &b.raw[layer])?;
            let ac = &a.owners[owner];
            let bc = &b.owners[owner];
            for (x, y) in [
                (&ac.compressed, &bc.compressed),
                (&ac.index, &bc.index),
                (&ac.previous_kv, &bc.previous_kv),
                (&ac.previous_score, &bc.previous_score),
            ] {
                equal(x, y)?;
            }
            if let (Some(x), Some(y)) = (&ac.index_bf16, &bc.index_bf16) {
                equal(x, y)?;
            }
            if let (Some(x), Some(y)) = (&sa, &sb) {
                equal(x, y)?;
            }
            if let (Some(x), Some(y)) = (&ca, &cb) {
                equal(&oracle::candidate_tensor(x), y)?;
            }
            if std::env::var_os("LISA_TRACE").is_some() {
                assert_eq!(
                    difference(&before, &between),
                    difference(&between, &after),
                    "layer {layer} position {pos}"
                );
            }
        }
    }
    Ok(())
}
