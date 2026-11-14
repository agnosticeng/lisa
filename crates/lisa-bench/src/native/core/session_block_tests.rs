use super::*;

struct BlockTarget {
    runtime: crate::metal_language::MetalLanguageRuntime,
    history: Vec<u32>,
    hidden: Option<Array>,
    capture: bool,
    append: bool,
    widths: Vec<usize>,
    capture_modes: Vec<bool>,
    wrong: bool,
}

impl LanguageModel<crate::metal_language::MetalLanguageRuntime> for BlockTarget {
    fn runtime(&self) -> &crate::metal_language::MetalLanguageRuntime {
        &self.runtime
    }
    fn max_position_embeddings(&self) -> usize {
        512
    }
    fn eos_token_id(&self) -> i64 {
        31
    }
    fn max_speculative_depth(&self) -> usize {
        5
    }
    fn has_block_drafter(&self) -> bool {
        true
    }
    fn new_caches(&self) -> Vec<LayerCache> {
        vec![LayerCache::Full(
            crate::core::cache::FullAttentionCache::new(
                crate::metal_context::execution_context(),
                1,
                1,
            ),
        )]
    }
    fn forward(
        &mut self,
        ids: &Array,
        caches: Option<&mut Vec<LayerCache>>,
    ) -> anyhow::Result<(Array, Array)> {
        let hidden = ids
            .as_dtype(lisa_metal::Dtype::Float32)?
            .reshape(&[1, ids.dim(1), 1])?;
        if let Some(caches) = caches {
            let LayerCache::Full(cache) = &mut caches[0] else {
                unreachable!()
            };
            let kv =
                hidden
                    .as_dtype(lisa_metal::Dtype::Bfloat16)?
                    .reshape(&[1, 1, ids.dim(1), 1])?;
            let _ = cache.update(&kv, &kv)?;
        }
        if self.capture {
            self.hidden = Some(hidden.clone());
        }
        Ok((hidden.clone(), hidden))
    }
    fn forward_capture(
        &mut self,
        ids: &Array,
        caches: Option<&mut Vec<LayerCache>>,
        capture: bool,
    ) -> anyhow::Result<(Array, Array)> {
        self.widths.push(ids.dim(1) as usize);
        self.capture_modes.push(capture);
        self.forward(ids, caches)
    }
    fn prefill(&mut self, tokens: &[u32], caches: &mut Vec<LayerCache>) -> anyhow::Result<Array> {
        let ids = Array::from_slice(
            &tokens.iter().map(|&t| t as i32).collect::<Vec<_>>(),
            &[1, tokens.len() as i32],
        );
        let (h, _) = self.forward(&ids, Some(caches))?;
        if self.append {
            self.history.extend_from_slice(tokens);
        }
        Ok(h.index((.., tokens.len() as i32 - 1, ..)).contiguous()?)
    }
    fn prefill_multi(
        &mut self,
        tokens: &[u32],
        caches: &mut Vec<LayerCache>,
    ) -> anyhow::Result<(Array, Array)> {
        let last = self.prefill(tokens, caches)?;
        Ok((last.clone(), last))
    }
    fn head(&self, mixed: &Array) -> lisa_metal::error::Result<Array> {
        let values = mixed.to_vec1::<f32>()?;
        let mut logits = vec![-40f32; values.len() * 32];
        for (row, x) in values.iter().enumerate() {
            logits[row * 32 + (*x as usize + 1) % 32] = 2.0;
            logits[row * 32 + (*x as usize + 2) % 32] = 1.0;
        }
        Ok(Array::from_slice(&logits, &[1, values.len() as i32, 32]))
    }
    fn warmup(&mut self, _: &[u32]) -> anyhow::Result<()> {
        Ok(())
    }
    fn set_block_drafter_capture(&mut self, on: bool) {
        self.capture = on;
    }
    fn set_block_drafter_prefill_append(&mut self, on: bool) {
        self.append = on;
    }
    fn take_drafter_hidden(&mut self) -> Option<Array> {
        self.hidden.take()
    }
    fn block_drafter_context_matches(&self, tokens: &[u32]) -> bool {
        self.history == tokens
    }
    fn rebuild_block_drafter_context(&mut self, tokens: &[u32]) -> anyhow::Result<()> {
        self.history = tokens.to_vec();
        Ok(())
    }
    fn drafter_reset(&mut self) {
        self.history.clear();
    }
    fn append_drafter_context(&mut self, hidden: &Array, tokens: &[u32]) -> anyhow::Result<()> {
        assert_eq!(
            hidden.to_vec1::<f32>()?,
            tokens.iter().map(|&x| x as f32).collect::<Vec<_>>()
        );
        self.history.extend_from_slice(tokens);
        Ok(())
    }
    fn draft_block(&mut self, anchor: u32) -> anyhow::Result<Array> {
        let ids: Vec<u32> = (1..=5)
            .map(|i| (anchor + i + if self.wrong { 7 } else { 0 }) % 32)
            .collect();
        Ok(Array::from_slice(&ids, &[1, 5]))
    }
}

#[test]
fn block_policy_serial_probes_preserve_tokens_rng_cache_and_resume() -> anyhow::Result<()> {
    let _eos = crate::core::generate::ignore_eos_scope();
    for temperature in [0., 0.8] {
        for wrong in [false, true] {
            let mut reference = None;
            for (depth, chunks) in [
                (MtpDepth::Fixed(0), vec![40]),
                (MtpDepth::Auto, vec![40]),
                // Resume at a configured-mode boundary: a new request starts
                // with its configured sampler rather than a prior callback.
                (MtpDepth::Auto, vec![17, 23]),
            ] {
                let mut tower = BlockTarget {
                    runtime: crate::metal_language::runtime(),
                    wrong,
                    ..Default::default()
                };
                let mut session = Session::new(&mut tower);
                let mut sampler = Sampler(lisa_engine::sampling::Sampler {
                    temperature,
                    seed: 731,
                    ..Default::default()
                });
                let mut output = Vec::new();
                let mut suffix = vec![1];
                let mut emitted = 0;
                for count in chunks {
                    let tokens = session.generate_mtp_depth_with_sampling_mode(
                        &mut tower,
                        &suffix,
                        count,
                        depth,
                        None,
                        Some(&mut sampler),
                        &mut |_| {
                            emitted += 1;
                            Ok(if emitted % 7 < 3 {
                                TargetMtpSamplingMode::Greedy
                            } else {
                                TargetMtpSamplingMode::Configured
                            })
                        },
                    )?;
                    suffix = vec![*tokens.last().unwrap()];
                    output.extend(tokens);
                    if matches!(depth, MtpDepth::Auto) {
                        assert_eq!(tower.history, session.fed);
                    }
                }
                let LayerCache::Full(cache) = &session.caches[0] else {
                    unreachable!()
                };
                assert_eq!(cache.offset, session.fed.len());
                let actual = cache
                    .keys
                    .as_ref()
                    .unwrap()
                    .index((0, 0, 0..cache.offset as i32, 0))
                    .as_dtype(lisa_metal::Dtype::Float32)?
                    .to_vec1::<f32>()?;
                assert_eq!(
                    actual,
                    session.fed.iter().map(|&x| x as f32).collect::<Vec<_>>()
                );
                let result = (output, sampler.seed, session.fed);
                if let Some(expected) = &reference {
                    assert_eq!(&result, expected);
                } else {
                    reference = Some(result);
                }
                if matches!(depth, MtpDepth::Auto) {
                    assert_eq!(&tower.widths[..3], &[1, 1, 1]);
                    assert_eq!(&tower.capture_modes[..3], &[false, false, false]);
                    assert!(tower.widths.iter().any(|&w| w > 1));
                    assert!(!tower.capture && !tower.append);
                }
            }
        }
    }
    Ok(())
}

#[test]
fn block_policy_callback_failure_invalidates_and_clears_capture() {
    let _eos = crate::core::generate::ignore_eos_scope();
    let mut tower = BlockTarget::default();
    let mut session = Session::new(&mut tower);
    let mut count = 0;
    let result = session.generate_mtp_depth(
        &mut tower,
        &[1],
        40,
        MtpDepth::Auto,
        None,
        None,
        &mut |_| {
            count += 1;
            anyhow::ensure!(count < 10, "cancelled");
            Ok(())
        },
    );
    assert!(result.is_err());
    assert!(!session.is_valid());
    assert!(!tower.capture && !tower.append);
}

#[test]
fn block_policy_eos_and_shortened_rounds_match_serial_boundaries() -> anyhow::Result<()> {
    let _restore_eos = crate::core::generate::ignore_eos_scope();
    crate::core::generate::set_eos_ids(vec![31]);
    for prompt in [18, 22, 23, 24, 25, 26, 27, 28, 29, 30] {
        for budget in 0..=12 {
            for wrong in [false, true] {
                let mut reference = None;
                for depth in [MtpDepth::Fixed(0), MtpDepth::Fixed(5), MtpDepth::Auto] {
                    let mut tower = BlockTarget {
                        runtime: crate::metal_language::runtime(),
                        wrong,
                        ..Default::default()
                    };
                    let mut session = Session::new(&mut tower);
                    let mut callbacks = Vec::new();
                    let output = session.generate_mtp_depth(
                        &mut tower,
                        &[prompt],
                        budget,
                        depth,
                        None,
                        None,
                        &mut |token| {
                            callbacks.push(token);
                            Ok(())
                        },
                    )?;
                    assert_eq!(callbacks, output);
                    assert!(
                        output
                            .iter()
                            .position(|&t| t == 31)
                            .is_none_or(|i| i + 1 == output.len())
                    );
                    assert!(!session.fed.contains(&31));
                    assert!(!tower.capture && !tower.append);
                    if matches!(depth, MtpDepth::Auto) && budget > 1 {
                        assert_eq!(tower.history, session.fed);
                    }
                    let result = (output, session.fed);
                    if let Some(expected) = &reference {
                        assert_eq!(
                            &result, expected,
                            "prompt={prompt} budget={budget} wrong={wrong}"
                        );
                    } else {
                        reference = Some(result);
                    }
                }
            }
        }
    }
    Ok(())
}

#[test]
#[ignore = "requires explicit local Qwen 27B and DSpine checkpoints"]
fn block_policy_real_serial_probe_matches_serial_entry() -> anyhow::Result<()> {
    use crate::{
        checkpoint,
        loading::{self, Loaded},
    };
    use anyhow::Context;
    let target =
        std::env::var("LISA_QWEN_DSPINE_TARGET_MODEL").context("target checkpoint required")?;
    let draft = std::env::var("LISA_QWEN_DSPINE_MODEL").context("draft checkpoint required")?;
    let target = checkpoint::resolve_model_dir(&target)?;
    let draft = checkpoint::resolve_model_dir(&draft)?;
    let Loaded::Language(mut tower) = loading::load_dir_with_dspine(&target, &draft)? else {
        anyhow::bail!("expected language model");
    };
    for prefix_len in [1, 17, 256] {
        let prefix: Vec<u32> = (0..prefix_len).map(|i| 100 + (i % 97) as u32).collect();
        let mut expected = None;
        for serial_entry in [true, false] {
            tower.set_block_drafter_capture(false);
            let mut caches = tower.new_caches();
            tower.prefill(&prefix, &mut caches)?;
            tower.set_block_drafter_capture(true);
            // Auto serial probes must obey the ordinary serial entry's law.
            // One-row speculative capture is a different path, not its oracle.
            let hidden = if serial_entry {
                tower.decode_token(2028, &mut caches)?
            } else {
                tower
                    .forward_capture(
                        &Array::from_slice(&[2028i32], &[1, 1]),
                        Some(&mut caches),
                        false,
                    )?
                    .0
            };
            let hidden_bits = hidden.to_vec1::<u16>()?;
            let logits_bits = tower.head(&hidden)?.to_vec1::<u16>()?;
            let taps = tower
                .take_drafter_hidden()
                .context("missing target taps")?
                .to_vec1::<u16>()?;
            let mut state = Vec::new();
            for cache in &caches {
                match cache {
                    LayerCache::Linear(c) => {
                        state.push(
                            c.ssm
                                .as_ref()
                                .context("missing SSM")?
                                .to_vec1::<f32>()?
                                .into_iter()
                                .map(f32::to_bits)
                                .collect::<Vec<_>>(),
                        );
                        state.push(
                            c.conv
                                .as_ref()
                                .context("missing convolution")?
                                .to_vec1::<u16>()?
                                .into_iter()
                                .map(u32::from)
                                .collect(),
                        );
                    }
                    LayerCache::Full(c) => {
                        for a in [&c.keys, &c.values] {
                            state.push(
                                a.as_ref()
                                    .context("missing KV")?
                                    .index((.., .., 0..c.offset as i32, ..))
                                    .to_vec1::<u16>()?
                                    .into_iter()
                                    .map(u32::from)
                                    .collect(),
                            );
                        }
                    }
                    LayerCache::Native(_) => anyhow::bail!("expected MLX target"),
                }
            }
            let actual = (hidden_bits, logits_bits, taps, state);
            if let Some(expected) = &expected {
                assert!(
                    expected == &actual,
                    "serial probe differs at prefix {prefix_len}"
                );
            } else {
                expected = Some(actual);
            }
        }
    }
    tower.set_block_drafter_capture(false);
    Ok(())
}

impl Default for BlockTarget {
    fn default() -> Self {
        Self {
            runtime: crate::metal_language::runtime(),
            history: Vec::new(),
            hidden: None,
            capture: false,
            append: false,
            widths: Vec::new(),
            capture_modes: Vec::new(),
            wrong: false,
        }
    }
}
