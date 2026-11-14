//! Application alias for the portable sampler.
use lisa_engine::{sampling::SamplingContext, tensor_ops::TensorLaw};
use lisa_metal::device::MetalDevice;
pub type Sampler = lisa_engine::sampler::Sampler<MetalDevice>;
#[allow(non_snake_case)]
pub fn Sampler(config: lisa_engine::sampling::Sampler) -> Sampler {
    Sampler::new(config)
}
#[cfg(test)]
pub(crate) fn context(device: &MetalDevice) -> SamplingContext<'_, MetalDevice> {
    SamplingContext::new(device, TensorLaw::Native, |_| None)
}
pub fn token_logprob(
    runtime: &crate::metal_language::MetalLanguageRuntime,
    logits: &lisa_metal::Array,
    token: u32,
) -> anyhow::Result<f32> {
    lisa_engine::sampling::token_logprob(&runtime.sampling(), logits, token)
}
#[cfg(test)]
use lisa_metal::Array;
#[cfg(test)]
mod keyed_tests {
    use super::*;
    use lisa_metal::{
        Dtype, ops,
        ops::indexing::{IndexOp, argmax_axis},
    };

    fn fake_logits(v: usize, seed: u64) -> Array {
        let mut s = seed;
        let mut rnd = move || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((s >> 40) as f32 / 8_388_608.0) - 1.0
        };
        let v: Vec<f32> = (0..v).map(|_| rnd() * 10.0).collect();
        Array::from_slice(&v, &[v.len() as i32])
            .as_dtype(Dtype::Float32)
            .unwrap()
    }

    #[test]
    fn ds4_sparse_penalty_matches_vocab_scan() {
        let device = crate::metal_context::device();
        lisa_engine::sampling::oracles::ds4_sparse_penalty_matches_vocab_scan(&context(&device));
    }

    #[test]
    // RESOLVED (specs/04 §6): failed ≥3× in CI-style parallel runs with a
    // MASSIVE divergence (3388 vs 783 at token 11). Root cause was the runtime
    // readback race: `flush_*` drained `in_flight` before waiting, so a
    // concurrent thread's `eval` returned without waiting committed GPU work
    // (plus a dropped completion-handler block in `ComputeEncoder::end` that
    // corrupted the cross-encoder fence map). Both fixed in commands.rs; the
    // parallel suite is green (extinction runs in specs/04 §6).
    fn keyed_same_seed_same_tokens() {
        let logits = fake_logits(4096, 1);
        let history: Vec<u32> = (0..37).collect();
        let mut a = Sampler(lisa_engine::sampling::Sampler {
            temperature: 0.7,
            ..Default::default()
        });
        let mut b = Sampler(lisa_engine::sampling::Sampler {
            temperature: 0.7,
            ..Default::default()
        });
        for pos in 0..24 {
            let hist: Vec<u32> = history[..pos].to_vec();
            let ta = a
                .draw(&crate::metal_language::runtime(), &logits, &hist)
                .unwrap();
            let tb = b
                .draw(&crate::metal_language::runtime(), &logits, &hist)
                .unwrap();
            assert_eq!(ta, tb, "position {pos}");
        }
    }

    #[test]
    fn keyed_seed_varies() {
        let logits = fake_logits(4096, 2);
        let mut a = Sampler(lisa_engine::sampling::Sampler {
            temperature: 0.7,
            seed: 0xDEAD,
            ..Default::default()
        });
        let mut b = Sampler(lisa_engine::sampling::Sampler {
            temperature: 0.7,
            seed: 0xBEEF,
            ..Default::default()
        });
        let diffs = (0..16)
            .filter(|pos| {
                let hist: Vec<u32> = (0..*pos as u32).collect();
                a.draw(&crate::metal_language::runtime(), &logits, &hist)
                    .unwrap()
                    != b.draw(&crate::metal_language::runtime(), &logits, &hist)
                        .unwrap()
            })
            .count();
        assert!(diffs > 0, "different seeds drew identical streams");
    }

    #[test]
    fn greedy_unchanged() {
        let v: Vec<f32> = (0..64).map(|i| (i as f32 * 0.5).sin()).collect();
        let arr = Array::from_slice(&v, &[64])
            .as_dtype(Dtype::Float32)
            .unwrap();
        let mut g = Sampler::default();
        let want = v
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap()
            .0 as u32;
        assert_eq!(
            g.draw(&crate::metal_language::runtime(), &arr, &[])
                .unwrap(),
            want
        );
    }

    #[test]
    fn exact_speculation_distribution_matches_top_k_top_p_and_min_p() {
        let device = crate::metal_context::device();
        lisa_engine::sampling::oracles::exact_speculation_distribution_matches_top_k_top_p_and_min_p(&context(&device));
    }

    #[test]
    fn glm_ds4_sampling_order_matches_c_reference() {
        let device = crate::metal_context::device();
        lisa_engine::sampling::oracles::glm_ds4_sampling_order_matches_c_reference(&context(
            &device,
        ));
    }

    #[test]
    fn glm_ds4_exact_delta_sampling_matches_c_reference() {
        let device = crate::metal_context::device();
        lisa_engine::sampling::oracles::glm_ds4_exact_delta_sampling_matches_c_reference(&context(
            &device,
        ));
    }

    #[test]
    fn glm_ds4_filter_edges_match_c_reference() {
        let device = crate::metal_context::device();
        lisa_engine::sampling::oracles::glm_ds4_filter_edges_match_c_reference(&context(&device));
    }

    // Pre-migration native operation sequence; independent of TensorOps dispatch.
    fn legacy_next(seed: &mut u64) -> u64 {
        *seed = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = *seed;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn legacy_draw(
        config: &mut lisa_engine::sampling::Sampler,
        logits: &Array,
        history: &[u32],
    ) -> anyhow::Result<u32> {
        let _trace = lisa_metal::trace::span("sampler");
        let flat = logits.reshape(&[-1])?;
        let mut logits = flat.as_dtype(lisa_metal::Dtype::Float32)?;
        if config.greedy() {
            return Ok(argmax_axis(&logits, -1, false)?.item_cast::<i32>() as u32);
        }

        if config.repetition_penalty == 1.0 {
            let stream = lisa_metal::Stream::thread_local_or_default();
            if let Some(tok) = lisa_metal::kernels::gated_pointwise::keyed_gumbel_sample(
                &logits,
                1.0 / config.temperature,
                config.seed,
                history.len() as u64,
                &stream,
            ) {
                return Ok(tok.item::<i32>() as u32);
            }
        }

        if config.repetition_penalty != 1.0 && !history.is_empty() {
            let start = history.len().saturating_sub(config.penalty_window);
            let penalty = config.repetition_penalty;
            let mut mask = vec![false; logits.dim(0) as usize];
            for &t in &history[start..] {
                if (t as i32) < logits.dim(0) {
                    mask[t as usize] = true;
                }
            }
            let mask = Array::from_slice(&mask, &[logits.dim(0)]);
            let pos =
                logits.divide(&ops::full::<f32>(logits.shape(), Array::from_f32(penalty))?)?;
            let neg =
                logits.multiply(&ops::full::<f32>(logits.shape(), Array::from_f32(penalty))?)?;
            logits = lisa_metal::ops::r#where(&mask, &pos, &neg)?;
        }
        logits = logits.divide(&ops::full::<f32>(
            logits.shape(),
            Array::from_f32(config.temperature),
        )?)?;

        let k = if config.top_k == 0 {
            64usize
        } else {
            config.top_k.min(logits.dim(0) as usize)
        };
        let neg = -&logits;
        let idx = ops::argpartition_axis(&neg, (k - 1) as i32, -1)?;
        let idx = idx.index(0..k as i32).contiguous()?;
        let vals = logits.take_along_axis(&idx, -1)?;
        let _ = (idx.eval(), vals.eval());
        let ids: Vec<u32> = idx.as_slice::<u32>().to_vec();
        let mut cand: Vec<(u32, f32)> = vals
            .as_slice::<f32>()
            .iter()
            .copied()
            .zip(ids.iter().copied())
            .map(|(v, t)| (t, v))
            .collect();
        drop(ids);

        cand.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let max = cand.first().map(|c| c.1).unwrap_or(f32::NEG_INFINITY);

        if config.min_p > 0.0 {
            let floor = max + config.min_p.max(1e-9).ln();
            cand.retain(|c| c.1 >= floor);
        }
        let mut probs: Vec<(u32, f32)> = {
            let m = cand.iter().map(|c| c.1).fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = cand.iter().map(|c| (c.1 - m).exp()).collect();
            let sum: f32 = exps.iter().sum();
            cand.iter()
                .zip(exps.iter())
                .map(|(c, e)| (c.0, e / sum.max(1e-30)))
                .collect()
        };
        if config.top_p < 1.0 && probs.len() > 1 {
            let mut cum = 0.0f32;
            let mut keep = 0usize;
            for (i, (_, p)) in probs.iter().enumerate() {
                cum += *p;
                keep = i + 1;
                if cum >= config.top_p {
                    break;
                }
            }
            probs.truncate(keep.max(1));
        }
        probs.retain(|(_, p)| *p > 0.0);
        if probs.is_empty() {
            return Ok(cand[0].0);
        }

        let total: f32 = probs.iter().map(|(_, p)| *p).sum();
        let mut r = (legacy_next(&mut config.seed) >> 11) as f32 / (1u64 << 53) as f32 * total;
        for (t, p) in &probs {
            r -= *p;
            if r <= 0.0 {
                return Ok(*t);
            }
        }
        Ok(probs.last().unwrap().0)
    }
    #[test]
    fn portable_sampling_preserves_native_draws_and_rng() -> anyhow::Result<()> {
        for width in [32, 64, 257] {
            for dtype in [Dtype::Float32, Dtype::Bfloat16] {
                let logits = fake_logits(width, 13).as_dtype(dtype)?;
                let history: Vec<u32> = (0..37).collect();
                for (temperature, top_k, top_p, repetition_penalty) in [
                    (0.0, 0, 1.0, 1.0),
                    (0.7, 0, 1.0, 1.0),
                    (0.7, 17, 0.8, 1.2),
                    (1.2, 33, 0.9, 0.8),
                ] {
                    for seed in [0, 7, 19] {
                        let mut actual = Sampler(lisa_engine::sampling::Sampler {
                            temperature,
                            top_k,
                            top_p,
                            repetition_penalty,
                            seed,
                            min_p: 0.02,
                            ..Default::default()
                        });
                        let mut expected = actual.0.clone();
                        for _ in 0..3 {
                            assert_eq!(
                                actual.draw(
                                    &crate::metal_language::runtime(),
                                    &logits,
                                    &history
                                )?,
                                legacy_draw(&mut expected, &logits, &history)?
                            );
                            assert_eq!(actual.seed, expected.seed);
                        }
                    }
                }
            }
        }
        Ok(())
    }

    #[test]
    fn portable_acceptance_preserves_native_reduction_bits() -> anyhow::Result<()> {
        for width in [64, 257, 4096] {
            for dtype in [Dtype::Float32, Dtype::Bfloat16] {
                let logits = fake_logits(width, 37).as_dtype(dtype)?;
                for temperature in [0.3, 0.7, 1.2] {
                    let sampler = Sampler(lisa_engine::sampling::Sampler {
                        temperature,
                        ..Default::default()
                    });
                    let flat = logits.reshape(&[-1])?.as_dtype(Dtype::Float32)?;
                    let inverse =
                        ops::full::<f32>(flat.shape(), Array::from_f32(1.0 / temperature))?;
                    let scaled = flat.multiply(&inverse)?;
                    let max = scaled.max(None)?;
                    let exp = scaled.subtract(&max)?.exp()?;
                    let sum = exp.sum(None)?;
                    max.eval()?;
                    sum.eval()?;
                    let m = max.item::<f32>();
                    let z = sum.item::<f32>();
                    for draft in [0, width / 2, width - 1] {
                        let index = Array::from_slice(&[draft as i32], &[1]);
                        let x = scaled.take_along_axis(&index, -1)?.item::<f32>();
                        let expected = ((x - m).exp() / z).min(1.0);
                        assert_eq!(
                            sampler
                                .draft_accept_prob(
                                    &crate::metal_language::runtime(),
                                    &logits,
                                    draft as u32
                                )
                                .unwrap()
                                .to_bits(),
                            expected.to_bits()
                        );
                    }
                }
            }
        }
        Ok(())
    }
}
