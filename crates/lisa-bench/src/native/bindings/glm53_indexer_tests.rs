#![cfg(test)]
use super::*;
use crate::bindings::deepseek41_moe::tests::{counts, difference};
#[path = "glm53_indexer_oracle.rs"]
mod oracle;
fn values(shape: &[i32], seed: usize) -> Array {
    Array::from_slice(
        &(0..shape.iter().product::<i32>() as usize)
            .map(|i| ((i * 17 + seed) % 127) as f32 / 151. - 0.3)
            .collect::<Vec<_>>(),
        shape,
    )
}
fn pair(dtype: Dtype, ape_dtype: Dtype) -> anyhow::Result<(oracle::Indexer, Indexer)> {
    let a = oracle::Indexer {
        wq_b_w: values(&[4096, 1536], 1).as_dtype(dtype)?,
        wk_w: values(&[128, 4096], 7).as_dtype(dtype)?,
        k_norm_w: values(&[128], 11),
        k_norm_b: values(&[128], 17),
        weights_w: values(&[32, 4096], 23).as_dtype(dtype)?,
        gate_w: values(&[128, 4096], 31).as_dtype(dtype)?,
        ape: values(&[4, 128], 41).as_dtype(ape_dtype)?,
        heads: 32,
        head_dim: 128,
        topk: 16,
        kpool: 4,
        always_tail: true,
    };
    let b = Indexer {
        context: crate::metal_context::execution_context(),
        wq_b_w: a.wq_b_w.clone(),
        wk_w: a.wk_w.clone(),
        k_norm_w: a.k_norm_w.clone(),
        k_norm_b: a.k_norm_b.clone(),
        weights_w: a.weights_w.clone(),
        gate_w: a.gate_w.clone(),
        ape: a.ape.clone(),
        heads: a.heads,
        head_dim: a.head_dim,
        topk: a.topk,
        kpool: a.kpool,
        always_tail: a.always_tail,
    };
    Ok((a, b))
}
fn equal(a: &Array, b: &Array) -> anyhow::Result<()> {
    assert_eq!(a.shape(), b.shape());
    assert_eq!(a.dtype(), b.dtype());
    assert_eq!(
        a.as_dtype(Dtype::Float32)?.to_vec1::<u32>()?,
        b.as_dtype(Dtype::Float32)?.to_vec1::<u32>()?
    );
    Ok(())
}
fn check<const N: usize>(
    old: impl FnOnce() -> anyhow::Result<[Array; N]>,
    new: impl FnOnce() -> anyhow::Result<[Array; N]>,
) -> anyhow::Result<()> {
    let before = counts();
    let a = old()?;
    let between = counts();
    let b = new()?;
    let after = counts();
    for (a, b) in a.iter().zip(&b) {
        equal(a, b)?;
    }
    if std::env::var_os("LISA_TRACE").is_some() {
        assert_eq!(difference(&before, &between), difference(&between, &after));
    }
    Ok(())
}
#[test]
fn indexer_projections_preserve_bits_and_kernel_counts() -> anyhow::Result<()> {
    for dtype in [Dtype::Bfloat16, Dtype::Float32] {
        let (a, b) = pair(dtype, Dtype::Bfloat16)?;
        for (batch, tokens) in [(1, 1), (1, 8), (1, 9), (1, 24), (2, 3)] {
            let x = values(&[batch, tokens, 4096], 51);
            let q = values(&[batch, tokens, 1536], 61);
            check(
                || Ok([a.project(&x, &a.wk_w)?]),
                || Ok([b.project(&x, &b.wk_w)?]),
            )?;
            check(
                || {
                    let (q, w) = a.project_score_inputs(&x, &q)?;
                    Ok([q, w])
                },
                || {
                    let (q, w) = b.project_score_inputs(&x, &q)?;
                    Ok([q, w])
                },
            )?;
        }
        let x = values(&[3, 4096], 81);
        check(
            || Ok([a.project(&x, &a.wk_w)?]),
            || Ok([b.project(&x, &b.wk_w)?]),
        )?;
    }
    Ok(())
}
#[test]
fn indexer_pool_score_selection_preserve_bits_and_kernel_counts() -> anyhow::Result<()> {
    for ape_dtype in [Dtype::Bfloat16, Dtype::Float32] {
        let (a, b) = pair(Dtype::Bfloat16, ape_dtype)?;
        for batch in [1, 2] {
            let (mut ta, mut tb) = (
                IndexerTape::new(crate::metal_context::execution_context()),
                IndexerTape::new(crate::metal_context::execution_context()),
            );
            for end in [3, 4, 7, 12, 35, 300] {
                let rows = end - ta.total;
                let raw = values(&[batch, rows as i32, 256], end + 71);
                ta.append(&raw, 4)?;
                tb.append(&raw, 4)?;
                let before = counts();
                a.update_pooled_cache(&mut ta, None, 0)?;
                let between = counts();
                b.update_pooled_cache(&mut tb, None, 0)?;
                let after = counts();
                if std::env::var_os("LISA_TRACE").is_some() {
                    assert_eq!(
                        difference(&before, &between),
                        difference(&between, &after),
                        "pool {ape_dtype:?} B{batch} end{end}"
                    );
                }
                assert_eq!(ta.pooled_upto, tb.pooled_upto);
                assert_eq!(ta.raw_start(), tb.raw_start());
                if let (Some(a), Some(b)) = (ta.pooled_view(), tb.pooled_view()) {
                    equal(&a, &b)?;
                }
                for (t, prefill) in [(1, false), (1, true), (3, true), (9, true)] {
                    if t > end {
                        continue;
                    }
                    let iq = values(&[batch, 32, t as i32, 128], 83);
                    let w = values(&[batch, t as i32, 32], 89);
                    check(
                        || Ok([a.index_scores(&mut ta, &iq, &w, end, t, None, 0, prefill)?]),
                        || Ok([b.index_scores(&mut tb, &iq, &w, end, t, None, 0, prefill)?]),
                    )?;
                    let x = values(&[batch, t as i32, 4096], 97);
                    let q = values(&[batch, t as i32, 1536], 101);
                    check(
                        || Ok([a.select(&x, &mut ta, &q, end, t, end, prefill)?]),
                        || Ok([b.select(&x, &mut tb, &q, end, t, end, prefill)?]),
                    )?;
                    if end < 4 {
                        continue;
                    }
                    check(
                        || {
                            Ok([a
                                .select_ids_gpu(&mut ta, &x, &q, end, t, None, 0, prefill)?
                                .0])
                        },
                        || {
                            Ok([b
                                .select_ids_gpu(&mut tb, &x, &q, end, t, None, 0, prefill)?
                                .0])
                        },
                    )?;
                    if t == 1 {
                        // The legacy GPU-mask fallback feeds F32 to where, which
                        // the runtime does not support. Preserve the rejection;
                        // production uses token-ID selection checked above.
                        let old = a
                            .select_keep_gpu(&mut ta, &x, &q, end, t, end, 0)
                            .err()
                            .expect("legacy float mask must reject");
                        let new = b
                            .select_keep_gpu(&mut tb, &x, &q, end, t, end, 0)
                            .err()
                            .expect("plan float mask must reject");
                        assert!(old.to_string().contains("where_cond"));
                        assert!(new.to_string().contains("Where"));
                    }
                }
            }
        }
    }
    Ok(())
}
#[test]
fn indexer_chunk_and_schedule_overrides_preserve_selection() -> anyhow::Result<()> {
    let (a, b) = pair(Dtype::Bfloat16, Dtype::Bfloat16)?;
    let (mut ta, mut tb) = (
        IndexerTape::new(crate::metal_context::execution_context()),
        IndexerTape::new(crate::metal_context::execution_context()),
    );
    let raw = values(&[1, 2052, 256], 111);
    ta.append(&raw, 4)?;
    tb.append(&raw, 4)?;
    let x = values(&[1, 2049, 4096], 113);
    let q = values(&[1, 2049, 1536], 127);
    check(
        || {
            Ok([a
                .select_ids_gpu(&mut ta, &x, &q, 2052, 2049, None, 0, true)?
                .0])
        },
        || {
            Ok([b
                .select_ids_gpu(&mut tb, &x, &q, 2052, 2049, None, 0, true)?
                .0])
        },
    )?;
    for rows4 in [false, true] {
        for shuffle in [false, true] {
            let _score = crate::core::quant::Glm53IndexerDecodeRows4Scope::new(rows4);
            let _sort = crate::core::quant::Glm53IndexerTopkShuffleScope::new(shuffle);
            let x = values(&[1, 1, 4096], 131);
            let q = values(&[1, 1, 1536], 137);
            check(
                || {
                    Ok([a
                        .select_ids_gpu(&mut ta, &x, &q, 2052, 1, None, 0, false)?
                        .0])
                },
                || {
                    Ok([b
                        .select_ids_gpu(&mut tb, &x, &q, 2052, 1, None, 0, false)?
                        .0])
                },
            )?;
        }
    }
    Ok(())
}
