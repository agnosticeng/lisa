//! Host-only tests of shared conversation transactions, independent of a GPU.
use super::*;

#[derive(Clone, Debug, PartialEq)]
struct Tensor {
    shape: Vec<i32>,
    values: Vec<u32>,
}
impl Tensor {
    fn rows(values: &[u32]) -> Self {
        Self {
            shape: vec![values.len() as i32, 1],
            values: values.to_vec(),
        }
    }
}
struct Host;
impl LanguageRuntime for Host {
    type Tensor = Tensor;
    type Cache = usize;
    type DraftCache = Vec<u32>;
    type Error = std::io::Error;
    fn shape(t: &Tensor) -> &[i32] {
        &t.shape
    }
    fn row(t: &Tensor, index: usize) -> anyhow::Result<Tensor> {
        Ok(Tensor::rows(&[t.values[index]]))
    }
    fn join_rows(t: &[&Tensor]) -> anyhow::Result<Tensor> {
        Ok(Tensor::rows(
            &t.iter().flat_map(|t| t.values.clone()).collect::<Vec<_>>(),
        ))
    }
}
impl SessionRuntime for Host {
    type Snapshot = usize;
    type DraftSnapshot = Vec<u32>;
    type PrefixState = usize;
    fn snapshot_cache(cache: &usize) -> usize {
        *cache
    }
    fn restore_cache(cache: &mut usize, snapshot: &usize) {
        *cache = *snapshot;
    }
    fn snapshot_draft(cache: &Vec<u32>) -> Vec<u32> {
        cache.clone()
    }
    fn restore_draft(cache: &mut Vec<u32>, snapshot: &Vec<u32>) {
        *cache = snapshot.clone();
    }
    fn restore_prefix(cache: &mut usize, prefix: &usize) {
        *cache = *prefix;
    }
    fn hidden_prefix(t: &Tensor, _rows: usize, keep: usize) -> anyhow::Result<Tensor> {
        Ok(Tensor::rows(&t.values[..keep]))
    }
    fn new_draft_cache(&self) -> Vec<u32> {
        Vec::new()
    }
}
#[derive(Default)]
struct Target {
    row_local: bool,
    law: u64,
    fail: bool,
    tail: Vec<Vec<i64>>,
    restored_head: usize,
    calls: Vec<&'static str>,
}
impl LanguageModel<Host> for Target {
    fn runtime(&self) -> &Host {
        &Host
    }
    fn batch_decode_equal_length_only(&self) -> bool {
        self.row_local
    }
    fn cache_numerical_law(&self) -> Option<u64> {
        Some(self.law)
    }
    fn max_position_embeddings(&self) -> usize {
        8
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
        if let Some(c) = cache {
            c[0] += ids.values.len();
        }
        for t in &mut self.tail {
            t.extend(ids.values.iter().map(|&v| v as i64));
        }
        Ok((ids.clone(), ids.clone()))
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
        self.calls.push("prefill");
        cache[0] += ids.len();
        Ok(Tensor::rows(&ids[ids.len() - 1..]))
    }
    fn decode_token(&mut self, id: u32, cache: &mut Vec<usize>) -> anyhow::Result<Tensor> {
        self.calls.push("decode");
        cache[0] += 1;
        Ok(Tensor::rows(&[id]))
    }
    fn prefill_multi(
        &mut self,
        ids: &[u32],
        cache: &mut Vec<usize>,
    ) -> anyhow::Result<(Tensor, Tensor)> {
        Ok((self.prefill(ids, cache)?, Tensor::rows(ids)))
    }
    fn head(&self, x: &Tensor) -> Result<Tensor, std::io::Error> {
        if self.fail {
            return Err(std::io::Error::other("injected failure after cache write"));
        }
        Ok(x.clone())
    }
    fn warmup(&mut self, _: &[u32]) -> anyhow::Result<()> {
        Ok(())
    }
    fn clear_context(&mut self) {
        self.tail.clear();
    }
    fn context_window(&self) -> usize {
        2
    }
    fn context_tails(&self) -> Vec<Vec<i64>> {
        self.tail.clone()
    }
    fn set_context_tails(&mut self, tails: Vec<Vec<i64>>) {
        self.tail = tails;
    }
    fn drafter_restore_offset(&mut self, offset: usize) {
        self.restored_head = offset;
    }
}

#[test]
fn committed_transactions_reject_law_changes_and_invalidate_partial_writes() -> anyhow::Result<()> {
    let mut target = Target::default();
    let mut session = Conversation::<Host>::new(&mut target);
    session.feed(&mut target, &[2, 3])?;
    session.decode_token(&mut target, 4)?;
    assert_eq!(session.fed, [2, 3, 4]);
    assert_eq!(session.caches, [3]);
    assert_eq!(target.calls, ["prefill", "decode"]);
    target.law = 1;
    assert!(session.decode_token(&mut target, 5).is_err());
    assert_eq!(session.caches, [3]);
    target.law = 0;
    assert!(session.feed(&mut target, &[]).is_err());
    assert!(session.feed(&mut target, &[0; 6]).is_err());
    assert!(session.is_valid());
    target.fail = true;
    assert!(session.decode_token(&mut target, 5).is_err());
    assert_eq!(session.caches, [4]); // The backend did write before failure.
    assert_eq!(session.fed, [2, 3, 4]); // No successful logical commit was claimed.
    assert!(!session.is_valid());
    target.fail = false;
    assert!(session.feed(&mut target, &[6]).is_err());
    Ok(())
}

#[test]
fn snapshots_restore_private_draft_state_and_hidden_history_with_target() -> anyhow::Result<()> {
    let mut target = Target::default();
    let mut session = Conversation::<Host>::new(&mut target);
    session.feed(&mut target, &[1, 2])?;
    session.enable_snapshots();
    session.mtp_tokens = vec![1, 2];
    session.mtp_multi = vec![Tensor::rows(&[1, 2])];
    session.glm_mtp_cache = vec![7, 8];
    session.glm_mtp_min_position = Some(2);
    session.capture_snapshot(1, 1);
    session.decode_token(&mut target, 3)?;
    session.mtp_tokens.push(3);
    session.mtp_multi.push(Tensor::rows(&[3]));
    session.glm_mtp_cache.push(9);
    session.glm_mtp_min_position = Some(3);
    assert_eq!(session.restore_snapshot(&mut target, 1), Some(2));
    assert_eq!(session.caches, [2]);
    assert_eq!(session.fed, [1, 2]);
    assert_eq!(session.mtp_tokens, [1, 2]);
    assert_eq!(session.mtp_multi, [Tensor::rows(&[1, 2])]);
    assert_eq!(session.glm_mtp_cache, [7, 8]);
    assert_eq!(session.glm_mtp_min_position, Some(2));
    assert_eq!(target.restored_head, 1);
    let resumed = Conversation::<Host>::from_prefix(&mut target, &[2], &[1, 2, 3], 2);
    assert_eq!(resumed.fed, [1, 2]);
    assert_eq!(target.tail, [vec![1, 2]]);
    Ok(())
}

#[test]
fn ragged_default_keeps_row_cache_and_rolling_context_separate() -> anyhow::Result<()> {
    let mut target = Target {
        tail: vec![vec![10], vec![20]],
        ..Default::default()
    };
    let mut caches = vec![vec![2], vec![4]];
    let (mixed, multi) = target.forward_ragged(&Tensor::rows(&[11, 21]), &mut caches)?;
    assert_eq!(mixed, Tensor::rows(&[11, 21]));
    assert_eq!(multi, mixed);
    assert_eq!(caches, [vec![3], vec![5]]);
    assert_eq!(target.tail, [vec![10, 11], vec![20, 21]]);
    assert!(
        target
            .forward_ragged(&Tensor::rows(&[11]), &mut caches)
            .is_err()
    );
    Ok(())
}

#[derive(Default)]
struct Sampler {
    draws: Vec<&'static str>,
}
impl TokenSampler<Host> for Sampler {
    fn greedy(&mut self, _: &Host, x: &Tensor) -> anyhow::Result<u32> {
        self.draws.push("greedy");
        Ok(x.values[0] + 1)
    }
    fn draw(&mut self, _: &Host, x: &Tensor, _: &[u32]) -> anyhow::Result<u32> {
        self.draws.push("configured");
        Ok(x.values[0] + 1)
    }
    fn draw_ds4(&mut self, _: &Host, x: &Tensor, _: &[u32]) -> anyhow::Result<u32> {
        self.draws.push("ds4");
        Ok(x.values[0] + 1)
    }
}
#[test]
fn portable_serial_generation_keeps_live_sampling_and_commit_boundaries() -> anyhow::Result<()> {
    let _eos = crate::eos::ignore_eos_scope();
    let mut target = Target::default();
    let mut session = Conversation::<Host>::new(&mut target);
    let mut sampler = Sampler::default();
    let mut seen = Vec::new();
    let out = session.generate_with_sampling_mode(
        &mut target,
        &[1],
        3,
        &mut sampler,
        None,
        &mut |token| {
            seen.push(token);
            Ok(TargetMtpSamplingMode::Greedy)
        },
    )?;
    assert_eq!(out, [2, 3, 4]);
    assert_eq!(seen, out);
    assert_eq!(session.fed, [1, 2, 3]);
    assert_eq!(sampler.draws, ["configured", "greedy", "greedy"]);
    assert_eq!(target.calls, ["prefill", "decode", "decode"]);
    assert!(
        session
            .generate_with_sampling_mode(
                &mut target,
                &[7],
                0,
                &mut sampler,
                None,
                &mut |_| unreachable!()
            )?
            .is_empty()
    );
    assert_eq!(session.fed, [1, 2, 3]);
    Ok(())
}

impl crate::batching::BatchRuntime for Host {
    type Sampler = Sampler;
    fn tokens(&self, values: &[i32], shape: &[i32]) -> anyhow::Result<Tensor> {
        Ok(Tensor {
            shape: shape.to_vec(),
            values: values.iter().map(|&v| v as u32).collect(),
        })
    }
    fn mask(&self, values: &[bool], shape: &[i32]) -> anyhow::Result<Tensor> {
        Ok(Tensor {
            shape: shape.to_vec(),
            values: values.iter().map(|&v| u32::from(v)).collect(),
        })
    }
    fn last_hidden(t: &Tensor) -> anyhow::Result<Tensor> {
        Ok(t.clone())
    }
    fn full_offset(_: &usize) -> Option<usize> {
        None
    }
    fn has_positions(_: &usize) -> bool {
        false
    }
    fn set_mask(_: &mut usize, _: &Tensor) {
        panic!("host test uses row-local storage");
    }
    fn advance_positions(_: &mut usize) {
        panic!("host test uses row-local storage");
    }
    fn clear_positions(_: &mut usize) {
        panic!("host test uses row-local storage");
    }
    fn trim_memory(&self) {}
    fn pack_layer(
        &self,
        _: &[Vec<usize>],
        _: usize,
        _: &[usize],
        _: usize,
        _: usize,
    ) -> anyhow::Result<usize> {
        anyhow::bail!("packed storage is not implemented by this host test adapter")
    }
    fn extract_streams(
        &self,
        _: &[usize],
        _: &[usize],
        _: &[usize],
        _: usize,
    ) -> anyhow::Result<Vec<Vec<usize>>> {
        anyhow::bail!("packed storage is not implemented by this host test adapter")
    }
    fn draw_rows(
        &self,
        t: &Tensor,
        rows: usize,
        sampler: &mut Sampler,
        history: &[Vec<u32>],
        ds4: bool,
    ) -> anyhow::Result<Vec<u32>> {
        (0..rows)
            .map(|i| {
                if ds4 {
                    sampler.draw_ds4(self, &Tensor::rows(&[t.values[i]]), &history[i])
                } else {
                    sampler.draw(self, &Tensor::rows(&[t.values[i]]), &history[i])
                }
            })
            .collect()
    }
    fn draw_rows_multi(
        &self,
        t: &Tensor,
        rows: usize,
        samplers: &mut [Sampler],
        history: &[Vec<u32>],
        ds4: bool,
    ) -> anyhow::Result<Vec<u32>> {
        (0..rows)
            .map(|i| {
                if ds4 {
                    samplers[i].draw_ds4(self, &Tensor::rows(&[t.values[i]]), &history[i])
                } else {
                    samplers[i].draw(self, &Tensor::rows(&[t.values[i]]), &history[i])
                }
            })
            .collect()
    }
}

#[test]
fn portable_continuous_batch_preserves_row_ownership_across_retirement() -> anyhow::Result<()> {
    use crate::batching::ContinuousBatch;
    let mut target = Target {
        row_local: true,
        ..Default::default()
    };
    let mut batch = ContinuousBatch::<Host>::empty(8);
    let mut sampler = Sampler::default();
    assert_eq!(batch.admit(&mut target, &[1, 2], &mut sampler)?, 3);
    assert!(!batch.admits_equal_length(1));
    assert_eq!(batch.admit(&mut target, &[4], &mut sampler)?, 5);
    assert_eq!(batch.kv_lens(), [2, 1]);
    let mut samplers = [Sampler::default(), Sampler::default()];
    assert_eq!(batch.step(&mut target, &[3, 5], &mut samplers)?, [4, 6]);
    assert_eq!(batch.fed, [vec![1, 2, 3], vec![4, 5]]);
    batch.retire(&mut target, &[false, true])?;
    assert_eq!(batch.len(), 1);
    assert_eq!(batch.fed, [vec![4, 5]]);
    assert_eq!(batch.step(&mut target, &[6], &mut samplers[..1])?, [7]);
    assert_eq!(batch.fed, [vec![4, 5, 6]]);
    batch.retire(&mut target, &[false])?;
    assert!(batch.is_empty());
    assert!(target.context_tails().is_empty());
    Ok(())
}

mod speculative {
    use super::*;
    use crate::speculation::MtpDepth;
    use crate::speculative_runtime::{SpeculativeRuntime, SpeculativeSampler, SpeculativeTensor};
    // Prediction-marker tensors keep this test about transactions and ownership,
    // independently of a provider's arithmetic or vocabulary reduction kernels.
    impl SpeculativeTensor for Tensor {
        fn independent_copy(&self) -> anyhow::Result<Self> {
            Ok(self.clone())
        }
        fn shape(&self) -> &[i32] {
            &self.shape
        }
        fn select(&self, axis: i32, row: i32) -> anyhow::Result<Self> {
            let mut out = self.slice(axis, row, Some(row + 1))?;
            out.shape.remove(axis as usize);
            Ok(out)
        }
        fn slice(&self, axis: i32, start: i32, end: Option<i32>) -> anyhow::Result<Self> {
            let axis = axis as usize;
            let end = end.unwrap_or(self.shape[axis]);
            anyhow::ensure!(
                start >= 0 && end >= start && end <= self.shape[axis],
                "invalid slice"
            );
            let outer: usize = self.shape[..axis].iter().map(|&v| v as usize).product();
            let inner: usize = self.shape[axis + 1..].iter().map(|&v| v as usize).product();
            let mut values = Vec::new();
            for row in 0..outer {
                let base = row * self.shape[axis] as usize * inner;
                values.extend_from_slice(
                    &self.values[base + start as usize * inner..base + end as usize * inner],
                );
            }
            let mut shape = self.shape.clone();
            shape[axis] = end - start;
            Ok(Self { shape, values })
        }
        fn reshape(&self, shape: &[i32]) -> anyhow::Result<Self> {
            anyhow::ensure!(
                shape.iter().map(|&v| v as usize).product::<usize>() == self.values.len(),
                "invalid reshape"
            );
            Ok(Self {
                shape: shape.to_vec(),
                values: self.values.clone(),
            })
        }
        fn expand_dims(&self, axis: i32) -> anyhow::Result<Self> {
            let mut out = self.clone();
            out.shape.insert(axis as usize, 1);
            Ok(out)
        }
        fn contiguous(&self) -> anyhow::Result<Self> {
            Ok(self.clone())
        }
        fn eval(&self) -> anyhow::Result<()> {
            Ok(())
        }
        fn read_raw_ids(&self) -> anyhow::Result<Vec<u32>> {
            Ok(self.values.clone())
        }
        fn read_ids(&self) -> anyhow::Result<Vec<u32>> {
            Ok(self.values.clone())
        }
        fn read_f32(&self) -> anyhow::Result<Vec<f32>> {
            Ok(self.values.iter().map(|&x| x as f32).collect())
        }
        fn argmax_id(&self) -> anyhow::Result<u32> {
            Ok(self.values[0])
        }
        fn argmax_axis(&self, axis: i32) -> anyhow::Result<Self> {
            anyhow::ensure!(
                axis == -1 && self.shape.last() == Some(&1),
                "marker prediction axis"
            );
            let mut out = self.clone();
            out.shape.pop();
            Ok(out)
        }
        fn concatenate(tensors: &[&Self], axis: i32) -> anyhow::Result<Self> {
            anyhow::ensure!(
                axis == 1 && tensors.iter().all(|t| t.shape[0] == 1),
                "test sequence concatenation"
            );
            let mut out = tensors[0].clone();
            out.shape[1] = tensors.iter().map(|t| t.shape[1]).sum();
            out.values = tensors.iter().flat_map(|t| t.values.clone()).collect();
            Ok(out)
        }
    }
    #[derive(Clone)]
    pub(super) struct Greedy;
    impl TokenSampler<Host> for Greedy {
        fn greedy(&mut self, _: &Host, t: &Tensor) -> anyhow::Result<u32> {
            t.argmax_id()
        }
        fn draw(&mut self, _: &Host, t: &Tensor, _: &[u32]) -> anyhow::Result<u32> {
            t.argmax_id()
        }
        fn draw_ds4(&mut self, _: &Host, t: &Tensor, _: &[u32]) -> anyhow::Result<u32> {
            t.argmax_id()
        }
    }
    impl SpeculativeSampler for Greedy {
        type Context = Host;
        type Tensor = Tensor;
        fn is_greedy(&self) -> bool {
            true
        }
        fn exact_sampling(&self) -> bool {
            false
        }
        fn temperature(&self) -> f32 {
            0.
        }
        fn repetition_penalty(&self) -> f32 {
            1.
        }
        fn penalty_window(&self) -> usize {
            16
        }
        fn draw(&mut self, _: &Host, t: &Tensor, _: &[u32]) -> anyhow::Result<u32> {
            t.argmax_id()
        }
        fn draw_ds4(&mut self, _: &Host, t: &Tensor, _: &[u32]) -> anyhow::Result<u32> {
            t.argmax_id()
        }
        fn draw_at(&self, _: &Host, t: &Tensor, _: u64) -> Option<u32> {
            Some(t.values[0])
        }
        fn draw_row(&mut self, _: &Host, t: &Tensor) -> Option<u32> {
            Some(t.values[0])
        }
        fn draft_accept_prob(&self, _: &Host, t: &Tensor, d: u32) -> Option<f32> {
            Some(f32::from(t.values[0] == d))
        }
        fn accept_draw(&mut self, p: f32) -> bool {
            p == 1.
        }
        fn draw_residual(&mut self, _: &Host, t: &Tensor, _: u32) -> Option<u32> {
            Some(t.values[0])
        }
        fn verify_draft_with_history(
            &mut self,
            _: &Host,
            t: &Tensor,
            d: u32,
            _: &[u32],
        ) -> Option<(bool, u32)> {
            Some((t.values[0] == d, t.values[0]))
        }
    }
    impl SpeculativeRuntime for Host {
        type Sampler = Greedy;
        type Rollback = usize;
        type SerialSnapshot = usize;
        type Trace = ();
        type VerifyScope = ();
        fn tokens(&self, ids: &[i32], shape: &[i32]) -> anyhow::Result<Tensor> {
            Ok(Tensor {
                shape: shape.to_vec(),
                values: ids.iter().map(|&x| x as u32).collect(),
            })
        }
        fn tokens_u32(&self, ids: &[u32], shape: &[i32]) -> anyhow::Result<Tensor> {
            Ok(Tensor {
                shape: shape.to_vec(),
                values: ids.to_vec(),
            })
        }
        fn snapshot_rollback(c: &usize) -> usize {
            *c
        }
        fn restore_rollback(c: &mut usize, s: &usize) -> anyhow::Result<()> {
            *c = *s;
            Ok(())
        }
        fn serial_snapshot(c: &usize) -> usize {
            *c
        }
        fn restore_serial_eos(c: &mut usize, s: &usize) {
            *c = *s
        }
        fn rollback(
            &self,
            c: &mut usize,
            advance: usize,
            keep: usize,
            _: bool,
        ) -> anyhow::Result<()> {
            *c -= advance - keep;
            Ok(())
        }
        fn trim_memory(&self) {}
        fn verify_scope(&self) {}
        fn trace(&self, _: &'static str) {}
        fn trace_detail(&self, _: &'static str, _: u64) {}
        fn trace_enabled(&self, _: &str) -> bool {
            false
        }
    }
    #[derive(Default)]
    struct SpecModel {
        block: bool,
        wrong: bool,
        hidden: Option<Tensor>,
        context: Vec<u32>,
        prefill_append: bool,
    }
    impl LanguageModel<Host> for SpecModel {
        fn runtime(&self) -> &Host {
            &Host
        }
        fn max_position_embeddings(&self) -> usize {
            128
        }
        fn eos_token_id(&self) -> i64 {
            1000
        }
        fn new_caches(&self) -> Vec<usize> {
            vec![0]
        }
        fn forward(
            &mut self,
            t: &Tensor,
            c: Option<&mut Vec<usize>>,
        ) -> anyhow::Result<(Tensor, Tensor)> {
            if let Some(c) = c {
                c[0] += t.values.len();
            }
            let t = Tensor {
                shape: vec![1, t.values.len() as i32, 1],
                values: t.values.clone(),
            };
            self.hidden = Some(t.clone());
            Ok((t.clone(), t))
        }
        fn forward_capture(
            &mut self,
            t: &Tensor,
            c: Option<&mut Vec<usize>>,
            _: bool,
        ) -> anyhow::Result<(Tensor, Tensor)> {
            self.forward(t, c)
        }
        fn prefill(&mut self, t: &[u32], c: &mut Vec<usize>) -> anyhow::Result<Tensor> {
            let (x, _) = self.prefill_multi(t, c)?;
            Ok(x)
        }
        fn prefill_multi(
            &mut self,
            t: &[u32],
            c: &mut Vec<usize>,
        ) -> anyhow::Result<(Tensor, Tensor)> {
            if self.prefill_append {
                self.context.extend_from_slice(t);
            }
            let (x, m) = self.forward(
                &Tensor {
                    shape: vec![1, t.len() as i32],
                    values: t.to_vec(),
                },
                Some(c),
            )?;
            Ok((x.select(1, t.len() as i32 - 1)?, m))
        }
        fn head(&self, t: &Tensor) -> Result<Tensor, std::io::Error> {
            Ok(Tensor {
                shape: t.shape.clone(),
                values: t.values.iter().map(|&v| v + 1).collect(),
            })
        }
        fn warmup(&mut self, _: &[u32]) -> anyhow::Result<()> {
            Ok(())
        }
        fn has_drafter(&self) -> bool {
            true
        }
        fn has_target_mtp(&self) -> bool {
            !self.block
        }
        fn has_block_drafter(&self) -> bool {
            self.block
        }
        fn max_speculative_depth(&self) -> usize {
            3
        }
        fn target_mtp_step(
            &mut self,
            c: &mut Vec<u32>,
            _: &Tensor,
            next: u32,
            position: usize,
            _: usize,
        ) -> anyhow::Result<Tensor> {
            c.resize(position + 1, 0);
            c[position] = next;
            Ok(Tensor {
                shape: vec![1, 1, 1],
                values: vec![next + if self.wrong { 50 } else { 1 }],
            })
        }
        fn take_drafter_hidden(&mut self) -> Option<Tensor> {
            self.hidden.take()
        }
        fn set_block_drafter_prefill_append(&mut self, on: bool) {
            self.prefill_append = on;
        }
        fn append_drafter_context(&mut self, _: &Tensor, t: &[u32]) -> anyhow::Result<()> {
            self.context.extend_from_slice(t);
            Ok(())
        }
        fn block_drafter_context_matches(&self, t: &[u32]) -> bool {
            self.context == t
        }
        fn rebuild_block_drafter_context(&mut self, t: &[u32]) -> anyhow::Result<()> {
            self.context = t.to_vec();
            Ok(())
        }
        fn draft_block(&mut self, a: u32) -> anyhow::Result<Tensor> {
            Ok(Tensor {
                shape: vec![1, 3],
                values: vec![a + 1, a + if self.wrong { 50 } else { 2 }, a + 3],
            })
        }
    }
    #[test]
    fn portable_target_mtp_resumes_acceptance_and_rejection_without_a_device() -> anyhow::Result<()>
    {
        let _eos = crate::eos::ignore_eos_scope();
        for wrong in [false, true] {
            let mut model = SpecModel {
                wrong,
                ..Default::default()
            };
            let mut s = Conversation::new(&mut model);
            let mut all = Vec::new();
            while all.len() < 12 {
                let prompt = if all.is_empty() { &[1, 2][..] } else { &[] };
                let out =
                    s.target_mtp_cycle(&mut model, prompt, 12 - all.len(), None, &mut |_| {
                        Ok(TargetMtpSamplingMode::Configured)
                    })?;
                all.extend(out);
                assert_eq!(s.caches[0], s.fed.len());
            }
            assert_eq!(all, (3..15).collect::<Vec<_>>());
            assert!(s.is_valid());
        }
        Ok(())
    }
    #[test]
    fn portable_block_verification_commits_only_accepted_prefix() -> anyhow::Result<()> {
        let _eos = crate::eos::ignore_eos_scope();
        for wrong in [false, true] {
            let mut model = SpecModel {
                block: true,
                wrong,
                ..Default::default()
            };
            let mut s = Conversation::new(&mut model);
            let out = s.generate_mtp_depth(
                &mut model,
                &[1, 2],
                12,
                MtpDepth::Fixed(3),
                None,
                None,
                &mut |_| Ok(()),
            )?;
            assert_eq!(out, (3..15).collect::<Vec<_>>());
            assert_eq!(s.caches[0], s.fed.len());
            assert_eq!(s.fed, (1..14).collect::<Vec<_>>());
            assert_eq!(model.context, s.fed);
        }
        Ok(())
    }
}

#[test]
fn exact_snapshot_rejects_missing_message_boundary_without_rewinding() -> anyhow::Result<()> {
    let mut target = Target::default();
    let mut session = Conversation::<Host>::new(&mut target);
    session.enable_snapshots();
    session.capture_snapshot(0, 0);
    session.feed(&mut target, &[1, 2])?;
    session.capture_snapshot(2, 1);
    session.decode_token(&mut target, 3)?;
    // Editing the first assistant shares one message, but only boundaries zero
    // and two exist. Reusing boundary zero with a suffix after one loses a turn.
    assert_eq!(session.restore_snapshot_exact(&mut target, 1), None);
    assert_eq!(session.fed, [1, 2, 3]);
    assert_eq!(session.caches, [3]);
    assert_eq!(session.restore_snapshot_exact(&mut target, 2), Some(2));
    assert_eq!(session.fed, [1, 2]);
    assert_eq!(session.caches, [2]);
    Ok(())
}
