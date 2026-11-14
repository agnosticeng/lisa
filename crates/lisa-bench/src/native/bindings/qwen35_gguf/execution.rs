//! Provider-native conformance tests for the portable GGUF binding.
use crate::bindings::qwen35_tower::Qwen35Load;
use crate::core::cache::{FullAttentionCache, LayerCache};
use anyhow::{Result, ensure};
use lisa_metal::{Array, Dtype, Stream, ops::indexing::IndexOp};
use std::path::Path;
pub type GgufExecution =
    lisa_engine::bindings::qwen35_gguf::execution::GgufExecution<lisa_metal::device::MetalDevice>;
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn f32_attention_keeps_serial_reductions_in_wide_windows() -> Result<()> {
        for prefix in [0, 37, 2048] {
            for rows in [1, 7, 16] {
                let len = prefix + rows;
                let q = Array::from_slice(
                    &(0..24 * rows * 256)
                        .map(|i| (i % 113) as f32 * 0.017 - 0.7)
                        .collect::<Vec<_>>(),
                    &[1, 24, rows, 256],
                );
                let data: Vec<_> = (0..4 * (len + 16) * 256)
                    .map(|i| (i % 131) as f32 * 0.011 - 0.6)
                    .collect();
                // A live cache is a strided view of a larger allocation.
                let k =
                    Array::from_slice(&data, &[1, 4, len + 16, 256]).index((.., .., 0..len, ..));
                let v = k.multiply(Array::from_f32(-0.7))?;
                let apply = |q: &Array, k: &Array, v: &Array, causal| {
                    lisa_metal::jit::sdpa_vector(
                        q.t.device(),
                        &q.t,
                        &k.t,
                        &v.t,
                        1. / 16.,
                        causal,
                        None,
                    )
                    .map(Array::new)
                };
                let wide = apply(&q, &k, &v, true)?;
                for i in 0..rows {
                    let single = apply(
                        &q.index((.., .., i..i + 1, ..)).contiguous()?,
                        &k.index((.., .., 0..prefix + i + 1, ..)),
                        &v.index((.., .., 0..prefix + i + 1, ..)),
                        false,
                    )?;
                    assert_eq!(
                        single.reshape(&[-1])?.to_vec1::<f32>()?,
                        wide.index((.., .., i..i + 1, ..))
                            .contiguous()?
                            .reshape(&[-1])?
                            .to_vec1::<f32>()?,
                        "prefix={prefix}, rows={rows}, query={i}"
                    );
                }
            }
        }
        Ok(())
    }
    #[test]
    fn f32_cache_keeps_precision_across_growth_and_restore() -> Result<()> {
        let values: Vec<f32> = (0..260 * 256)
            .map(|i| 1.0001 + (i % 19) as f32 * 0.0001)
            .collect();
        let all = Array::from_slice(&values, &[1, 1, 260, 256]);
        let mut cache = FullAttentionCache::with_dtype(
            crate::metal_context::execution_context(),
            1,
            256,
            Dtype::Float32,
        );
        cache.update(
            &all.index((.., .., 0..1, ..)),
            &all.index((.., .., 0..1, ..)),
        )?;
        let layer = LayerCache::Full(cache);
        let saved = layer.snapshot_prefix();
        let mut restored = LayerCache::Full(FullAttentionCache::with_dtype(
            crate::metal_context::execution_context(),
            1,
            256,
            Dtype::Float32,
        ));
        restored.restore_prefix(&saved);
        let LayerCache::Full(mut cache) = restored else {
            unreachable!()
        };
        let (k, v) = cache.update(
            &all.index((.., .., 1..260, ..)),
            &all.index((.., .., 1..260, ..)),
        )?;
        for array in [k, v] {
            assert_eq!(array.dtype(), Dtype::Float32);
            array.eval()?;
            assert_eq!(array.reshape(&[-1])?.to_vec1::<f32>()?, values);
        }
        Ok(())
    }
}

#[cfg(test)]
mod checkpoint_tests {
    use super::*;
    use crate::language::LanguageModel;
    #[test]
    #[ignore = "requires the exact Unsloth checkpoint and independent llama.cpp F32 logits"]
    fn qwen_gguf_checkpoint_matches_reference_and_replays_state() -> Result<()> {
        let dir = crate::checkpoint::hf::cached_snapshot(super::super::REPO)
            .ok_or_else(|| anyhow::anyhow!("download the curated Qwen GGUF first"))?;
        let file = dir.join(super::super::ENTRY);
        let tokenizer = crate::core::tokenizer::Tokenizer::load(&file)?;
        let prompt = "<|im_start|>user\nExplain why the sky is blue in two short sentences.<|im_end|>\n<|im_start|>assistant\n<think>\n</think>\n\n";
        let tokens = tokenizer.encode_special(prompt)?;
        let expected_path = std::env::var("LISA_QWEN_GGUF_REFERENCE")?;
        let bytes = std::fs::read(expected_path)?;
        ensure!(
            bytes.len() == 248320 * 4,
            "reference must contain 248320 F32 logits"
        );
        let expected: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        let mut tower = crate::bindings::qwen35::Qwen35Tower::load_gguf(&file)?;
        tower.set_context_capacity(512)?;
        let mut caches = tower.new_caches();
        let mut last = None;
        for &token in &tokens {
            last = Some(
                tower
                    .forward(&Array::from_slice(&[token], &[1, 1]), Some(&mut caches))?
                    .0,
            );
        }
        let last = last.unwrap();
        let logits = tower.head(&last)?;
        logits.eval()?;
        let actual = logits.reshape(&[-1])?.to_vec1::<f32>()?;
        let max = actual
            .iter()
            .zip(&expected)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        let dot = actual
            .iter()
            .zip(&expected)
            .map(|(a, b)| *a as f64 * *b as f64)
            .sum::<f64>();
        let length = |x: &[f32]| x.iter().map(|v| (*v as f64).powi(2)).sum::<f64>().sqrt();
        let cosine = dot / (length(&actual) * length(&expected));
        let argmax = |x: &[f32]| {
            x.iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .unwrap()
                .0
        };
        eprintln!(
            "reference logits: max_abs={max} cosine={cosine} argmax={}/{}",
            argmax(&actual),
            argmax(&expected)
        );
        ensure!(
            actual.iter().all(|v| v.is_finite()),
            "non-finite Qwen logits"
        );
        ensure!(
            max <= 0.005 && cosine >= 0.999999 && argmax(&actual) == argmax(&expected),
            "Qwen GGUF reference mismatch"
        );
        let mut wide_caches = tower.new_caches();
        let (wide, history) = tower.prefill_multi(&tokens, &mut wide_caches)?;
        let wide_logits = tower.head(&wide)?.reshape(&[-1])?.to_vec1::<f32>()?;
        let wide_max = actual
            .iter()
            .zip(&wide_logits)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        eprintln!("serial/wide logits max_abs={wide_max}");
        ensure!(actual == wide_logits, "serial/prefill arithmetic differs");
        drop(wide_caches);
        ensure!(
            history.dim(1) as usize == tokens.len(),
            "prefill lost draft history"
        );
        let mut pairs = tokens[1..].to_vec();
        pairs.push(argmax(&actual) as u32);
        let pairs = Array::from_slice(&pairs, &[1, pairs.len() as i32]);
        let (primed_token, primed_hidden) = tower.draft_step(&pairs, &history)?;
        tower.drafter_reset();
        let mut serial_draft = None;
        for i in 0..pairs.dim(1) {
            serial_draft = Some(tower.draft_step(
                &pairs.index((.., i..i + 1)),
                &history.index((.., i..i + 1, ..)).contiguous()?,
            )?);
        }
        let (serial_token, serial_hidden) = serial_draft.unwrap();
        ensure!(
            primed_token.to_vec1::<u32>()? == serial_token.to_vec1::<u32>()?,
            "MTP priming token mismatch"
        );
        ensure!(
            primed_hidden.to_vec1::<f32>()? == serial_hidden.to_vec1::<f32>()?,
            "MTP priming state mismatch"
        );
        let invalid = Array::from_slice(&[11u32; 4], &[1, 4]);
        ensure!(
            tower
                .forward_capture(&invalid, Some(&mut caches), true)
                .is_err(),
            "oversized verify window was admitted"
        );
        let saved: Vec<_> = caches.iter().map(LayerCache::snapshot_rollback).collect();
        let ids = Array::from_slice(&[argmax(&actual) as u32, 11, 13], &[1, 3]);
        let (verified, _) = tower.forward_capture(&ids, Some(&mut caches), true)?;
        for c in &mut caches {
            match c {
                LayerCache::Full(f) => f.trim(2),
                LayerCache::Linear(g) => {
                    g.rollback_to(&crate::metal_context::execution_context(), 1, 4)?
                }
                _ => unreachable!(),
            }
        }
        let continued = tower.forward(&ids.index((.., 1..3)), Some(&mut caches))?.0;
        continued.eval()?;
        let verified = verified.index((.., 1..3, ..)).contiguous()?;
        verified.eval()?;
        let delta = continued
            .subtract(&verified)?
            .abs()?
            .max(None)?
            .item::<f32>();
        ensure!(
            delta < 0.0001,
            "recurrent rollback changed hidden states: {delta}"
        );
        for (cache, state) in caches.iter_mut().zip(&saved) {
            cache.restore_rollback(state)?;
        }
        let single = tower.forward(&ids.index((.., 0..1)), Some(&mut caches))?.0;
        tower.drafter_reset();
        let (token, hidden) = tower.draft_step(&ids.index((.., 0..1)), &single)?;
        token.eval()?;
        hidden.eval()?;
        ensure!(tower.drafter_offset() == 1, "MTP failed to advance");
        tower.drafter_restore_offset(0);
        let (again, replayed) = tower.draft_step(&ids.index((.., 0..1)), &single)?;
        again.eval()?;
        replayed.eval()?;
        ensure!(
            token.as_slice::<u32>() == again.as_slice::<u32>(),
            "MTP replay token mismatch"
        );
        ensure!(
            hidden.reshape(&[-1])?.to_vec1::<f32>()?
                == replayed.reshape(&[-1])?.to_vec1::<f32>()?,
            "MTP replay state mismatch"
        );
        Ok(())
    }
}
