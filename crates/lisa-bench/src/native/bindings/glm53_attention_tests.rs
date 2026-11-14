#![cfg(test)]
use super::*;
use crate::bindings::deepseek41_moe::tests::{counts, difference};
use crate::bindings::glm53_attention::Dsa;
use crate::core::{cache::FullAttentionCache, loader::Weights};
use lisa_metal::Dtype;
#[path = "glm53_attention_oracle.rs"]
mod oracle;
fn values(shape: &[i32], seed: usize) -> Array {
    Array::from_slice(
        &(0..shape.iter().product::<i32>() as usize)
            .map(|i| ((i * 17 + seed) % 127) as f32 / 151. - 0.3)
            .collect::<Vec<_>>(),
        shape,
    )
}
fn equal(a: &Array, b: &Array) -> anyhow::Result<()> {
    assert_eq!(a.shape(), b.shape());
    assert_eq!(a.dtype(), b.dtype());
    match a.dtype() {
        Dtype::Float16 | Dtype::Bfloat16 => assert_eq!(a.to_vec1::<u16>()?, b.to_vec1::<u16>()?),
        Dtype::Float32 => assert_eq!(a.to_vec1::<u32>()?, b.to_vec1::<u32>()?),
        d => panic!("unexpected result dtype {d:?}"),
    }
    Ok(())
}
fn weights() -> anyhow::Result<Weights> {
    let mut w = Weights::new();
    for (i, (name, n, k)) in [
        ("q_a_proj", 1536, 4096),
        ("q_b_proj", 16384, 1536),
        ("kv_a_proj_with_mqa", 512, 4096),
        ("kv_b_proj", 32768, 512),
        ("o_proj", 4096, 16384),
    ]
    .into_iter()
    .enumerate()
    {
        let packed = Array::from_slice(
            &(0..n * k / 4)
                .map(|j| (j as u32).wrapping_mul(1729).wrapping_add(i as u32))
                .collect::<Vec<_>>(),
            &[n, k / 4],
        );
        let scales = Array::from_slice(
            &(0..n * k / 64)
                .map(|j| 0.00001 + ((j + i as i32) % 7) as f32 * 0.0000001)
                .collect::<Vec<_>>(),
            &[n, k / 64],
        );
        let biases = Array::from_slice(
            &(0..n * k / 64)
                .map(|j| -0.00128 - ((j + i as i32) % 7) as f32 * 0.0000128)
                .collect::<Vec<_>>(),
            &[n, k / 64],
        );
        for (suffix, a) in [("weight", packed), ("scales", scales), ("biases", biases)] {
            w.insert(format!("attn.{name}.{suffix}"), a);
        }
    }
    for (name, shape, seed, bf16) in [
        ("q_a_layernorm.weight", vec![1536], 1, false),
        ("kv_a_layernorm.weight", vec![512], 3, false),
        ("indexer.wq_b.weight", vec![4096, 1536], 5, true),
        ("indexer.wk.weight", vec![128, 4096], 7, true),
        ("indexer.k_norm.weight", vec![128], 11, false),
        ("indexer.k_norm.bias", vec![128], 13, false),
        ("indexer.weights_proj.weight", vec![32, 4096], 17, true),
        (
            "indexer.index_kpool_compress_gate",
            vec![128, 4096],
            19,
            true,
        ),
        ("indexer.index_kpool_compress_ape", vec![4, 128], 23, true),
    ] {
        w.insert(
            format!("attn.{name}"),
            values(&shape, seed).as_dtype(if bf16 {
                Dtype::Bfloat16
            } else {
                Dtype::Float32
            })?,
        );
    }
    Ok(w)
}
#[test]
fn dsa_input_cache_and_target_plans_preserve_bits() -> anyhow::Result<()> {
    let w = weights()?;
    let old = oracle::Dsa::load::<Weights>(
        &mut w.clone(),
        "attn",
        64,
        256,
        256,
        32,
        128,
        2048,
        4,
        true,
        1e-5,
    )?;
    let new = Dsa::load_on::<Weights>(
        &crate::metal_context::execution_context(),
        &mut w.clone(),
        "attn",
        64,
        256,
        256,
        32,
        128,
        2048,
        4,
        true,
        1e-5,
    )?;
    for native in [true, false] {
        let _q8 = crate::core::quant::Glm53Q8PrefillScope::new(native);
        let _attention = crate::bindings::glm53_attention_mode::Glm53DsaNativeScope::new(native);
        for (b, t) in [(1, 1), (2, 1), (1, 8), (1, 24), (1, 32), (2, 3)] {
            let mut ac = FullAttentionCache::new(crate::metal_context::execution_context(), 1, 512);
            let mut bc = FullAttentionCache::new(crate::metal_context::execution_context(), 1, 512);
            for (n, prefill) in [(t, true), (1, false)] {
                let x = values(&[b, n, 4096], 31);
                let before = counts();
                let a = old.forward(&x, &mut ac, None, 3, prefill)?;
                let between = counts();
                let z = new.forward(&x, &mut bc, None, 3, prefill)?;
                let after = counts();
                equal(&a, &z)?;
                assert_eq!(ac.offset, bc.offset);
                equal(
                    &ac.indexer_tape.compact_slice(0, ac.offset).unwrap(),
                    &bc.indexer_tape.compact_slice(0, bc.offset).unwrap(),
                )?;
                if std::env::var_os("LISA_TRACE").is_some() {
                    assert_eq!(
                        difference(&before, &between),
                        difference(&between, &after),
                        "native{native} B{b} T{n} prefill{prefill}"
                    );
                }
            }
        }
        for length in [2051usize, 2052] {
            let mut ac = FullAttentionCache::new(crate::metal_context::execution_context(), 1, 512);
            let mut bc = FullAttentionCache::new(crate::metal_context::execution_context(), 1, 512);
            let keys = values(&[1, length as i32, 128], 47);
            let gates = values(&[1, length as i32, 128], 53);
            let latent = values(&[1, length as i32, 512], 59);
            for cache in [&mut ac, &mut bc] {
                cache.indexer_tape.append_fields(&keys, &gates, 4)?;
                cache.indexer_tape.append_compact_f16(&latent)?;
                cache.offset = length;
            }
            let x = values(&[1, 1, 4096], 61);
            let before = counts();
            let a = old.forward(&x, &mut ac, None, 3, false)?;
            let between = counts();
            let b = new.forward(&x, &mut bc, None, 3, false)?;
            let after = counts();
            equal(&a, &b)?;
            if std::env::var_os("LISA_TRACE").is_some() {
                assert_eq!(difference(&before, &between), difference(&between, &after));
            }
        }
        for preserve in [false, true] {
            let acap = crate::core::capture::Capture {
                layers: vec![3],
                preserve_dispatch: preserve,
                ..Default::default()
            };
            let bcap = crate::core::capture::Capture {
                layers: vec![3],
                preserve_dispatch: preserve,
                ..Default::default()
            };
            let mut ac = FullAttentionCache::new(crate::metal_context::execution_context(), 1, 512);
            let mut bc = FullAttentionCache::new(crate::metal_context::execution_context(), 1, 512);
            let x = values(&[1, 1, 4096], 71);
            let before = counts();
            let a = old.forward(&x, &mut ac, Some(&acap), 3, false)?;
            let between = counts();
            let b = new.forward(&x, &mut bc, Some(&bcap), 3, false)?;
            let after = counts();
            equal(&a, &b)?;
            if std::env::var_os("LISA_TRACE").is_some() {
                assert_eq!(difference(&before, &between), difference(&between, &after));
            }
            if preserve {
                let a = acap.pending.lock().unwrap();
                let b = bcap.pending.lock().unwrap();
                assert_eq!(
                    a.len(),
                    b.len(),
                    "old keys {:?}, new keys {:?}",
                    a.keys().collect::<Vec<_>>(),
                    b.keys().collect::<Vec<_>>()
                );
                for (k, v) in a.iter() {
                    equal(v, &b[k])?;
                }
            } else {
                let a = acap.entries.lock().unwrap();
                let b = bcap.entries.lock().unwrap();
                assert_eq!(
                    a.len(),
                    b.len(),
                    "old keys {:?}, new keys {:?}",
                    a.keys().collect::<Vec<_>>(),
                    b.keys().collect::<Vec<_>>()
                );
                for (k, v) in a.iter() {
                    assert_eq!(
                        v.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                        b[k].iter().map(|v| v.to_bits()).collect::<Vec<_>>()
                    );
                }
            }
        }
        for n in [1, 7, 2052] {
            let mut ac = IndexerTape::new(crate::metal_context::execution_context());
            let mut bc = IndexerTape::new(crate::metal_context::execution_context());
            let c = values(&[1, n, 512], 41);
            for row in 0..n {
                let v = Array::new(c.t.narrow(1, row as usize, 1)?);
                ac.write_compact_f16_at(&v, row as usize)?;
                bc.write_compact_f16_at(&v, row as usize)?;
            }
            let x = values(&[1, 1, 4096], 43);
            let before = counts();
            let a = old.forward_mtp(&x, &mut ac, n as usize - 1, 0)?;
            let between = counts();
            let b = new.forward_mtp(&x, &mut bc, n as usize - 1, 0)?;
            let after = counts();
            equal(&a, &b)?;
            equal(
                &ac.compact_slice(0, n as usize).unwrap(),
                &bc.compact_slice(0, n as usize).unwrap(),
            )?;
            if std::env::var_os("LISA_TRACE").is_some() {
                assert_eq!(difference(&before, &between), difference(&between, &after));
            }
        }
    }
    Ok(())
}

#[test]
fn dense_fallback_graphs_keep_original_order() -> anyhow::Result<()> {
    use crate::bindings::glm53_oracle_helpers::narrow;
    use lisa_metal::ops;
    use lisa_models::glm53::attention::{Request as R, Shape};
    for batch in [1, 2] {
        for tokens in [1, 3] {
            let s = Shape {
                batch,
                tokens,
                heads: 2,
                query: 8,
                value: 12,
                cache: 7,
            };
            let q = values(&[batch as i32, 2, tokens as i32, 8], 3);
            let k = values(&[batch as i32, 2, 7, 8], 5);
            let v = values(&[batch as i32, 2, 7, 12], 7);
            let keep = values(&[batch as i32, 1, tokens as i32, 7], 11);
            let scale = 0.125f32;
            let before = counts();
            let kt = k.transpose_axes(&[0, 1, 3, 2])?.contiguous()?;
            let scale_a = ops::full::<i32>(&[1], &Array::from_f32_as(scale, Dtype::Float32))?;
            let mut old = None;
            let mut expected = vec![];
            for row in 0..batch {
                let qb = narrow(&q, 0, row, 1)?.reshape(&[2, tokens as i32, 8])?;
                let kb = narrow(&kt, 0, row, 1)?.reshape(&[2, 8, 7])?;
                let vb = narrow(&v, 0, row, 1)?.reshape(&[2, 7, 12])?;
                let scores = qb
                    .matmul(&kb)?
                    .multiply(&scale_a)?
                    .add(&narrow(&keep, 0, row, 1)?)?
                    .reshape(&[2, tokens as i32, 7])?;
                let probs = scores.softmax_axis(-1)?;
                let y = probs.matmul(&vb)?.reshape(&[(2 * tokens * 12) as i32])?;
                old = Some(match old {
                    Some(a) => Array::concatenate(&[&a, &y], 0)?,
                    None => y,
                });
                expected.push([qb, kb, vb, scores, probs]);
            }
            let old = old
                .unwrap()
                .reshape(&[batch as i32, 2, tokens as i32, 12])?
                .transpose_axes(&[0, 2, 1, 3])?
                .reshape(&[batch as i32, tokens as i32, 24])?;
            let middle = counts();
            let kt = one(R::DenseTranspose, &[&k])?;
            let mut new = None;
            let mut actual = vec![];
            for row in 0..batch {
                let mut op =
                    run(R::DenseOperands { shape: s, row }, &[&q, &kt, &v, &keep])?.into_iter();
                let qb = op.next().unwrap();
                let kb = op.next().unwrap();
                let vb = op.next().unwrap();
                let keep = op.next().unwrap();
                let mut sc = run(
                    R::DenseScores {
                        shape: s,
                        scale: scale.to_bits(),
                    },
                    &[&qb, &kb, &keep],
                )?
                .into_iter();
                let scores = sc.next().unwrap();
                let probs = sc.next().unwrap();
                let y = one(R::DenseValue { shape: s }, &[&probs, &vb])?;
                new = Some(match new {
                    Some(a) => one(R::Join { count: 2, axis: 0 }, &[&a, &y])?,
                    None => y,
                });
                actual.push([qb, kb, vb, scores, probs]);
            }
            let new = one(R::FinalFallback(s), &[&new.unwrap()])?;
            let after = counts();
            equal(&old, &new)?;
            for (a, b) in expected.iter().flatten().zip(actual.iter().flatten()) {
                equal(a, b)?;
            }
            if std::env::var_os("LISA_TRACE").is_some() {
                assert_eq!(difference(&before, &middle), difference(&middle, &after));
            }
        }
    }
    Ok(())
}

pub(crate) fn tower_fixture() -> anyhow::Result<Dsa> {
    Dsa::load_on::<Weights>(
        &crate::metal_context::execution_context(),
        &mut weights()?,
        "attn",
        64,
        256,
        256,
        32,
        128,
        2048,
        4,
        true,
        1e-5,
    )
}
