//! Independent-MTP ownership checks using exact host prediction markers.
use super::{Host, Tensor, speculative::Greedy};
use crate::{
    conversation::{Conversation, TargetMtpSamplingMode},
    language::LanguageModel,
    speculation::{MtpDepth, runtime::SpeculativeTensor},
};

#[derive(Default)]
struct Independent {
    wrong: bool,
    wrong_after: Option<u32>,
    pairs: Vec<(u32, u32)>,
    tails: Vec<Vec<i64>>,
    forwards: Vec<(usize, Vec<u32>, Vec<Vec<i64>>)>,
    draft_calls: usize,
    capacity: Option<usize>,
}
impl LanguageModel<Host> for Independent {
    fn runtime(&self) -> &Host {
        &Host
    }
    fn cache_numerical_law(&self) -> Option<u64> {
        Some(7)
    }
    fn max_position_embeddings(&self) -> usize {
        self.capacity.unwrap_or(128)
    }
    fn eos_token_id(&self) -> i64 {
        1000
    }
    fn new_caches(&self) -> Vec<usize> {
        vec![0]
    }
    fn forward(
        &mut self,
        ids: &Tensor,
        cache: Option<&mut Vec<usize>>,
    ) -> anyhow::Result<(Tensor, Tensor)> {
        let cache = cache.expect("independent MTP requires its target cache");
        self.forwards
            .push((cache[0], ids.values.clone(), self.tails.clone()));
        cache[0] += ids.values.len();
        anyhow::ensure!(
            cache[0] <= self.max_position_embeddings(),
            "forward exceeded context limit"
        );
        for tail in &mut self.tails {
            tail.extend(ids.values.iter().map(|&v| i64::from(v)));
            tail.drain(..tail.len().saturating_sub(2));
        }
        let rows = Tensor {
            shape: vec![1, ids.values.len() as i32, 1],
            values: ids.values.clone(),
        };
        Ok((rows.clone(), rows))
    }
    fn forward_capture(
        &mut self,
        ids: &Tensor,
        cache: Option<&mut Vec<usize>>,
        _: bool,
    ) -> anyhow::Result<(Tensor, Tensor)> {
        self.forward(ids, cache)
    }
    fn prefill(&mut self, ids: &[u32], cache: &mut Vec<usize>) -> anyhow::Result<Tensor> {
        Ok(self.prefill_multi(ids, cache)?.0)
    }
    fn prefill_multi(
        &mut self,
        ids: &[u32],
        cache: &mut Vec<usize>,
    ) -> anyhow::Result<(Tensor, Tensor)> {
        let input = Tensor {
            shape: vec![1, ids.len() as i32],
            values: ids.to_vec(),
        };
        let (mixed, multi) = self.forward(&input, Some(cache))?;
        Ok((mixed.select(1, ids.len() as i32 - 1)?, multi))
    }
    fn head(&self, hidden: &Tensor) -> Result<Tensor, std::io::Error> {
        Ok(Tensor {
            shape: hidden.shape.clone(),
            values: hidden.values.iter().map(|&v| v + 1).collect(),
        })
    }
    fn warmup(&mut self, _: &[u32]) -> anyhow::Result<()> {
        Ok(())
    }
    fn context_window(&self) -> usize {
        2
    }
    fn context_tails(&self) -> Vec<Vec<i64>> {
        self.tails.clone()
    }
    fn set_context_tails(&mut self, tails: Vec<Vec<i64>>) {
        self.tails = tails;
    }
    fn clear_context(&mut self) {
        self.tails = vec![vec![]];
    }
    fn has_drafter(&self) -> bool {
        true
    }
    fn max_speculative_depth(&self) -> usize {
        6
    }
    fn mtp_cost_key(&self) -> String {
        "host-independent-mtp-transaction-test".into()
    }
    fn drafter_reset(&mut self) {
        self.pairs.clear();
    }
    fn drafter_offset(&self) -> usize {
        self.pairs.len()
    }
    fn drafter_trim(&mut self, rows: usize) {
        self.pairs.truncate(self.pairs.len() - rows);
    }
    fn drafter_restore_offset(&mut self, rows: usize) {
        self.pairs.truncate(rows);
    }
    fn draft_step(&mut self, ids: &Tensor, hidden: &Tensor) -> anyhow::Result<(Tensor, Tensor)> {
        anyhow::ensure!(
            ids.values.len() == hidden.values.len(),
            "draft token/hidden alignment"
        );
        self.draft_calls += 1;
        self.pairs.extend(
            ids.values
                .iter()
                .copied()
                .zip(hidden.values.iter().copied()),
        );
        let token = *ids.values.last().expect("nonempty draft input");
        let wrong = self.wrong || self.wrong_after == Some(token);
        Ok((
            Tensor {
                shape: vec![1, 1],
                values: vec![token + if wrong { 2 } else { 1 }],
            },
            Tensor {
                shape: vec![1, 1, 1],
                values: vec![token],
            },
        ))
    }
}

fn assert_target_boundary(model: &Independent, session: &Conversation<Host>, fed: &[u32]) {
    assert!(session.is_valid());
    assert_eq!(session.fed, fed);
    assert_eq!(session.caches, [fed.len()]);
    assert_eq!(
        session.last_mixed.as_ref().unwrap().values,
        [*fed.last().unwrap()]
    );
    assert_eq!(
        model.tails,
        [fed[fed.len().saturating_sub(2)..]
            .iter()
            .map(|&v| i64::from(v))
            .collect::<Vec<_>>()]
    );
}

fn assert_independent_boundary(model: &Independent, session: &Conversation<Host>, fed: &[u32]) {
    assert_target_boundary(model, session, fed);
    assert_eq!(session.mtp_tokens, fed);
    let multi = session
        .mtp_multi
        .iter()
        .flat_map(|t| t.values.clone())
        .collect::<Vec<_>>();
    assert_eq!(
        multi, fed,
        "hidden store must contain exactly one row per fed input"
    );
    let expected_pairs = fed.windows(2).map(|p| (p[1], p[0])).collect::<Vec<_>>();
    assert_eq!(
        model.pairs, expected_pairs,
        "draft head must exclude the un-fed carry"
    );
    assert_eq!(session.mtp_head_len, expected_pairs.len());
}

#[test]
fn independent_zero_and_one_output_budgets_do_not_prime_the_head() -> anyhow::Result<()> {
    let _eos = crate::eos::ignore_eos_scope();
    for depth in [0, 2, 6] {
        let mut model = Independent::default();
        let mut session = Conversation::new(&mut model);
        assert!(
            session
                .generate_mtp(&mut model, &[1, 2], 0, depth, &mut |_| unreachable!())?
                .is_empty()
        );
        assert_eq!(session.caches, [0]);
        assert!(session.fed.is_empty());
        assert!(model.forwards.is_empty());
        assert_eq!(model.draft_calls, 0);
        assert_eq!(
            session.generate_mtp(&mut model, &[1, 2], 1, depth, &mut |_| Ok(()))?,
            [3]
        );
        assert_target_boundary(&model, &session, &[1, 2]);
        assert_eq!(model.draft_calls, 0);
        assert!(model.pairs.is_empty());
    }
    Ok(())
}

#[test]
fn independent_invalid_depth_fails_without_feeding_the_target() {
    for depth in [1, 7] {
        let mut model = Independent::default();
        let mut session = Conversation::new(&mut model);
        assert!(
            session
                .generate_mtp(&mut model, &[1, 2], 8, depth, &mut |_| Ok(()))
                .is_err()
        );
        assert!(model.forwards.is_empty());
        assert!(model.pairs.is_empty());
        assert_eq!(session.caches, [0]);
    }
}

#[test]
fn independent_full_and_rejected_rounds_preserve_every_budget_boundary() -> anyhow::Result<()> {
    let _eos = crate::eos::ignore_eos_scope();
    for wrong in [0, 1, 2] {
        for depth in [0, 2, 6] {
            for budget in 2..=10 {
                let mut model = Independent {
                    wrong: wrong == 1,
                    wrong_after: (wrong == 2).then_some(4),
                    ..Default::default()
                };
                let mut session = Conversation::new(&mut model);
                let generated =
                    session.generate_mtp(&mut model, &[1, 2], budget, depth, &mut |_| Ok(()))?;
                assert_eq!(generated, (3..3 + budget as u32).collect::<Vec<_>>());
                let fed = (1..2 + budget as u32).collect::<Vec<_>>();
                if depth == 0 {
                    assert_target_boundary(&model, &session, &fed);
                    assert_eq!(model.draft_calls, 0);
                } else {
                    assert_independent_boundary(&model, &session, &fed);
                }
                for (position, _, tail) in model.forwards.iter().skip(1) {
                    assert_eq!(
                        tail,
                        &[fed[position.saturating_sub(2)..*position]
                            .iter()
                            .map(|&v| i64::from(v))
                            .collect::<Vec<_>>()]
                    );
                }
            }
        }
    }
    Ok(())
}

#[test]
fn independent_eos_is_emitted_but_never_fed_or_reused_as_a_head_pair() -> anyhow::Result<()> {
    let _eos = crate::eos::ignore_eos_scope();
    for wrong in [false, true] {
        for stop in 3..=9 {
            crate::eos::set_eos_ids(vec![stop]);
            let mut model = Independent {
                wrong,
                ..Default::default()
            };
            let mut session = Conversation::new(&mut model);
            let out = session.generate_mtp(&mut model, &[1, 2], 12, 6, &mut |_| Ok(()))?;
            assert_eq!(out, (3..=stop).collect::<Vec<_>>());
            let fed = (1..stop).collect::<Vec<_>>();
            if stop == 3 {
                assert_target_boundary(&model, &session, &fed);
                assert!(model.pairs.is_empty());
            } else {
                assert_independent_boundary(&model, &session, &fed);
            }
        }
    }
    Ok(())
}

#[test]
fn independent_resume_replaces_the_previous_unfed_carry_with_the_actual_suffix()
-> anyhow::Result<()> {
    let _eos = crate::eos::ignore_eos_scope();
    for wrong in [false, true] {
        for first_depth in [0, 2] {
            let mut model = Independent {
                wrong,
                ..Default::default()
            };
            let mut session = Conversation::new(&mut model);
            assert_eq!(
                session.generate_mtp(&mut model, &[1, 2], 5, first_depth, &mut |_| Ok(()))?,
                [3, 4, 5, 6, 7]
            );
            // A new chat suffix need not start with the previous un-fed carry.
            assert_eq!(
                session.generate_mtp(&mut model, &[20, 21], 5, 2, &mut |_| Ok(()))?,
                [22, 23, 24, 25, 26]
            );
            assert_independent_boundary(
                &model,
                &session,
                &[1, 2, 3, 4, 5, 6, 20, 21, 22, 23, 24, 25],
            );
            assert_eq!(
                session.generate_mtp(&mut model, &[40], 2, 0, &mut |_| Ok(()))?,
                [41, 42]
            );
            assert_eq!(
                session.generate_mtp(&mut model, &[60], 3, 2, &mut |_| Ok(()))?,
                [61, 62, 63]
            );
            assert_independent_boundary(
                &model,
                &session,
                &[1, 2, 3, 4, 5, 6, 20, 21, 22, 23, 24, 25, 40, 41, 60, 61, 62],
            );
        }
    }
    Ok(())
}

#[test]
fn independent_verify_is_bounded_by_the_remaining_context() -> anyhow::Result<()> {
    let _eos = crate::eos::ignore_eos_scope();
    for capacity in 2..=8 {
        let mut model = Independent {
            capacity: Some(capacity),
            ..Default::default()
        };
        let mut session = Conversation::new(&mut model);
        let out = session.generate_mtp(&mut model, &[1, 2], 100, 6, &mut |_| Ok(()))?;
        assert_eq!(out, (3..=capacity as u32 + 1).collect::<Vec<_>>());
        assert_target_boundary(&model, &session, &(1..=capacity as u32).collect::<Vec<_>>());
        assert!(
            model
                .forwards
                .iter()
                .all(|(position, ids, _)| position + ids.len() <= capacity)
        );
    }
    Ok(())
}

#[test]
fn independent_sampling_mode_and_draw_history_follow_only_emitted_tokens() -> anyhow::Result<()> {
    let _eos = crate::eos::ignore_eos_scope();
    for wrong in [false, true] {
        let mut model = Independent {
            wrong,
            ..Default::default()
        };
        let mut session = Conversation::new(&mut model);
        let mut sampler = Greedy {
            configured: true,
            ..Default::default()
        };
        let mut callbacks = vec![];
        let output = session.generate_mtp_depth_with_sampling_mode(
            &mut model,
            &[1, 2],
            8,
            MtpDepth::Fixed(6),
            None,
            Some(&mut sampler),
            &mut |token| {
                callbacks.push(token);
                Ok(if matches!(token, 3 | 4) {
                    TargetMtpSamplingMode::Greedy
                } else {
                    TargetMtpSamplingMode::Configured
                })
            },
        )?;
        assert_eq!(output, (3..=10).collect::<Vec<_>>());
        assert_eq!(callbacks, output);
        // First draw is configured; tokens 4/5 use the live callback's greedy
        // mode. Rejected suffixes and unused bonus rows consume no draws.
        assert_eq!(
            sampler.draws,
            [2usize, 5, 6, 7, 8, 9].map(|n| (false, (1..=n as u32).collect::<Vec<_>>()))
        );
        assert_independent_boundary(&model, &session, &(1..=9).collect::<Vec<_>>());
    }
    Ok(())
}

#[test]
fn independent_callback_failure_invalidates_target_and_resets_draft_head() -> anyhow::Result<()> {
    let _eos = crate::eos::ignore_eos_scope();
    for wrong in [false, true] {
        for fail_at in 1..=5 {
            let mut model = Independent {
                wrong,
                ..Default::default()
            };
            let mut session = Conversation::new(&mut model);
            let mut sampler = Greedy {
                configured: true,
                ..Default::default()
            };
            let mut callbacks = 0usize;
            let result = session.generate_mtp_depth_with_sampling_mode(
                &mut model,
                &[1, 2],
                8,
                MtpDepth::Fixed(6),
                None,
                Some(&mut sampler),
                &mut |_| {
                    callbacks += 1;
                    anyhow::ensure!(callbacks != fail_at, "injected callback failure");
                    Ok(TargetMtpSamplingMode::Configured)
                },
            );
            assert!(result.is_err());
            assert_eq!(callbacks, fail_at);
            assert_eq!(sampler.draws.len(), fail_at, "no RNG draws after failure");
            assert!(!session.is_valid());
            assert!(model.pairs.is_empty());
            assert!(
                session
                    .generate_mtp(&mut model, &[20], 3, 2, &mut |_| Ok(()))
                    .is_err()
            );
        }
    }
    Ok(())
}

#[test]
fn independent_snapshot_retains_priming_rows_after_private_head_reset() -> anyhow::Result<()> {
    let _eos = crate::eos::ignore_eos_scope();
    for initial_outputs in [1, 8] {
        let mut model = Independent::default();
        let mut session = Conversation::new(&mut model);
        session.enable_snapshots();
        session.generate_mtp(&mut model, &[1, 2], initial_outputs, 2, &mut |_| Ok(()))?;
        let prefix = session.fed.clone();
        session.capture_snapshot(1, model.drafter_offset());
        // A one-output turn clears private head KV but keeps target priming
        // rows. Rewinding its offset cannot reconstruct that cleared storage.
        session.generate_mtp(&mut model, &[30, 40], 1, 2, &mut |_| Ok(()))?;
        assert_eq!(model.drafter_offset(), 0);
        assert_eq!(
            session.restore_snapshot_exact(&mut model, 1),
            Some(prefix.len())
        );
        assert_target_boundary(&model, &session, &prefix);
        assert_eq!(session.mtp_tokens, prefix);
        assert_eq!(
            session
                .mtp_multi
                .iter()
                .flat_map(|t| t.values.iter().copied())
                .collect::<Vec<_>>(),
            prefix
        );
        assert_eq!(session.mtp_head_len, 0);
        assert_eq!(model.drafter_offset(), 0);
        let output = session.generate_mtp(&mut model, &[50, 60], 4, 2, &mut |_| Ok(()))?;
        assert_eq!(output, [61, 62, 63, 64]);
        let fed = prefix
            .into_iter()
            .chain([50, 60, 61, 62, 63])
            .collect::<Vec<_>>();
        assert_independent_boundary(&model, &session, &fed);
    }
    Ok(())
}
