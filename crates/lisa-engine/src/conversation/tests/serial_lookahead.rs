//! Host transaction oracle for the shared HTTP/UI Conversation::generate path.
use super::{Conversation, LanguageModel, LanguageRuntime, SessionRuntime, Tensor, TokenSampler};
use crate::speculation::runtime::{QueuedGreedyToken, SpeculativeRuntime, SpeculativeSampler};
use std::{
    cell::{Cell, RefCell},
    rc::Rc,
};

const EOS: u32 = 99;

#[derive(Clone, Debug, PartialEq, Eq)]
enum Event {
    Draw(u32),
    Host(u32),
    Device(u32),
    Queue(u32),
    Read(u32),
    Drain,
}

#[derive(Default)]
struct Runtime {
    events: Rc<RefCell<Vec<Event>>>,
    fail_queue: Cell<Option<u32>>,
    fail_read: Cell<Option<u32>>,
    fail_finish: Cell<bool>,
}
impl LanguageRuntime for Runtime {
    type Tensor = Tensor;
    type Cache = Vec<u32>;
    type DraftCache = Vec<u32>;
    type Error = std::io::Error;
    fn shape(t: &Tensor) -> &[i32] {
        &t.shape
    }
    fn row(t: &Tensor, n: usize) -> anyhow::Result<Tensor> {
        Ok(Tensor::rows(&[t.values[n]]))
    }
    fn join_rows(t: &[&Tensor]) -> anyhow::Result<Tensor> {
        Ok(Tensor::rows(
            &t.iter().flat_map(|t| t.values.clone()).collect::<Vec<_>>(),
        ))
    }
}
impl SessionRuntime for Runtime {
    type Snapshot = Vec<u32>;
    type DraftSnapshot = Vec<u32>;
    type PrefixState = Vec<u32>;
    fn snapshot_cache(c: &Vec<u32>) -> Vec<u32> {
        c.clone()
    }
    fn restore_cache(c: &mut Vec<u32>, s: &Vec<u32>) {
        *c = s.clone();
    }
    fn snapshot_draft(c: &Vec<u32>) -> Vec<u32> {
        c.clone()
    }
    fn restore_draft(c: &mut Vec<u32>, s: &Vec<u32>) {
        *c = s.clone();
    }
    fn restore_prefix(c: &mut Vec<u32>, s: &Vec<u32>) {
        *c = s.clone();
    }
    fn hidden_prefix(t: &Tensor, _: usize, keep: usize) -> anyhow::Result<Tensor> {
        Ok(Tensor::rows(&t.values[..keep]))
    }
    fn new_draft_cache(&self) -> Vec<u32> {
        Vec::new()
    }
}
impl SpeculativeRuntime for Runtime {
    type Sampler = Sampler;
    type Rollback = Vec<u32>;
    type SerialSnapshot = Vec<u32>;
    type Trace = ();
    fn supports_queued_greedy(&self) -> bool {
        true
    }
    fn queue_greedy_token(&self, logits: &Tensor) -> anyhow::Result<QueuedGreedyToken<Tensor>> {
        let id = logits.values[0];
        self.events.borrow_mut().push(Event::Queue(id));
        anyhow::ensure!(self.fail_queue.get() != Some(id), "injected queue failure");
        let ids = Tensor {
            shape: vec![1, 1],
            values: vec![id],
        };
        Ok(QueuedGreedyToken {
            mirror: ids.clone(),
            ids,
        })
    }
    fn read_queued_token(&self, token: &QueuedGreedyToken<Tensor>) -> anyhow::Result<u32> {
        let id = token.mirror.values[0];
        self.events.borrow_mut().push(Event::Read(id));
        anyhow::ensure!(self.fail_read.get() != Some(id), "injected read failure");
        Ok(id)
    }
    fn finish_queued_work(&self) -> anyhow::Result<()> {
        self.events.borrow_mut().push(Event::Drain);
        anyhow::ensure!(!self.fail_finish.get(), "injected completion failure");
        Ok(())
    }
    fn tokens(&self, v: &[i32], shape: &[i32]) -> anyhow::Result<Tensor> {
        Ok(Tensor {
            shape: shape.to_vec(),
            values: v.iter().map(|&x| x as u32).collect(),
        })
    }
    fn tokens_u32(&self, v: &[u32], shape: &[i32]) -> anyhow::Result<Tensor> {
        Ok(Tensor {
            shape: shape.to_vec(),
            values: v.to_vec(),
        })
    }
    fn snapshot_rollback(c: &Vec<u32>) -> Vec<u32> {
        c.clone()
    }
    fn restore_rollback(c: &mut Vec<u32>, s: &Vec<u32>) -> anyhow::Result<()> {
        *c = s.clone();
        Ok(())
    }
    fn serial_snapshot(c: &Vec<u32>) -> Vec<u32> {
        c.clone()
    }
    fn restore_serial_eos(c: &mut Vec<u32>, s: &Vec<u32>) {
        *c = s.clone();
    }
    fn rollback(
        &self,
        c: &mut Vec<u32>,
        advance: usize,
        keep: usize,
        _: bool,
    ) -> anyhow::Result<()> {
        c.truncate(c.len() - advance + keep);
        Ok(())
    }
    fn trim_memory(&self) {}
    fn trace(&self, _: &'static str) {}
    fn trace_detail(&self, _: &'static str, _: u64) {}
    fn trace_enabled(&self, _: &str) -> bool {
        false
    }
}

#[derive(Clone, Default)]
struct Sampler {
    configured: bool,
    penalty: bool,
    seed: u64,
}
impl Sampler {
    fn sample(&mut self, r: &Runtime, t: &Tensor) -> anyhow::Result<u32> {
        if self.configured {
            self.seed += 1;
        }
        r.events.borrow_mut().push(Event::Draw(t.values[0]));
        Ok(t.values[0])
    }
}
impl TokenSampler<Runtime> for Sampler {
    fn greedy(&mut self, r: &Runtime, t: &Tensor) -> anyhow::Result<u32> {
        self.sample(r, t)
    }
    fn draw(&mut self, r: &Runtime, t: &Tensor, _: &[u32]) -> anyhow::Result<u32> {
        self.sample(r, t)
    }
    fn draw_host(&mut self, r: &Runtime, t: &Tensor, _: &[u32]) -> anyhow::Result<u32> {
        self.sample(r, t)
    }
}
impl SpeculativeSampler for Sampler {
    type Tensor = Tensor;
    type Context = Runtime;
    fn is_greedy(&self) -> bool {
        !self.configured
    }
    fn supports_queued_greedy(&self) -> bool {
        !self.configured && !self.penalty
    }
    fn exact_sampling(&self) -> bool {
        false
    }
    fn temperature(&self) -> f32 {
        if self.configured { 1.0 } else { 0.0 }
    }
    fn repetition_penalty(&self) -> f32 {
        if self.penalty { 1.1 } else { 1.0 }
    }
    fn penalty_window(&self) -> usize {
        16
    }
    fn draw(&mut self, r: &Runtime, t: &Tensor, _: &[u32]) -> anyhow::Result<u32> {
        self.sample(r, t)
    }
    fn draw_host(&mut self, r: &Runtime, t: &Tensor, _: &[u32]) -> anyhow::Result<u32> {
        self.sample(r, t)
    }
    fn draw_at(&self, _: &Runtime, _: &Tensor, _: u64) -> Option<u32> {
        panic!("not speculative")
    }
    fn draw_row(&mut self, _: &Runtime, _: &Tensor) -> Option<u32> {
        panic!("not speculative")
    }
    fn draft_accept_prob(&self, _: &Runtime, _: &Tensor, _: u32) -> Option<f32> {
        panic!("not speculative")
    }
    fn accept_draw(&mut self, _: f32) -> bool {
        panic!("not speculative")
    }
    fn draw_residual(&mut self, _: &Runtime, _: &Tensor, _: u32) -> Option<u32> {
        panic!("not speculative")
    }
    fn verify_draft_with_history(
        &mut self,
        _: &Runtime,
        _: &Tensor,
        _: u32,
        _: &[u32],
    ) -> Option<(bool, u32)> {
        panic!("not speculative")
    }
}

struct Model {
    runtime: Runtime,
    eligible: bool,
    ds4: bool,
    capacity: usize,
    eos_after: Option<u32>,
    fail_device: Option<u32>,
    tails: Vec<Vec<i64>>,
}
impl Model {
    fn new(eligible: bool) -> Self {
        Self {
            runtime: Runtime::default(),
            eligible,
            ds4: false,
            capacity: 32,
            eos_after: None,
            fail_device: None,
            tails: Vec::new(),
        }
    }
    fn append(&mut self, ids: &[u32], caches: &mut [Vec<u32>]) -> anyhow::Result<Tensor> {
        anyhow::ensure!(
            caches[0].len() + ids.len() <= self.capacity,
            "context overflow"
        );
        caches[0].extend_from_slice(ids);
        self.tails = vec![
            caches[0]
                .iter()
                .rev()
                .take(2)
                .copied()
                .collect::<Vec<_>>()
                .into_iter()
                .rev()
                .map(i64::from)
                .collect(),
        ];
        Ok(Tensor::rows(&ids[ids.len() - 1..]))
    }
}
impl LanguageModel<Runtime> for Model {
    fn runtime(&self) -> &Runtime {
        &self.runtime
    }
    fn uses_host_sampling(&self) -> bool {
        self.ds4
    }
    fn cache_numerical_law(&self) -> Option<u64> {
        Some(18)
    }
    fn max_position_embeddings(&self) -> usize {
        self.capacity
    }
    fn eos_token_id(&self) -> i64 {
        EOS.into()
    }
    fn new_caches(&self) -> Vec<Vec<u32>> {
        vec![Vec::new()]
    }
    fn forward(
        &mut self,
        ids: &Tensor,
        caches: Option<&mut Vec<Vec<u32>>>,
    ) -> anyhow::Result<(Tensor, Tensor)> {
        let h = self.append(&ids.values, caches.unwrap())?;
        Ok((h.clone(), h))
    }
    fn forward_capture(
        &mut self,
        ids: &Tensor,
        caches: Option<&mut Vec<Vec<u32>>>,
        _: bool,
    ) -> anyhow::Result<(Tensor, Tensor)> {
        self.forward(ids, caches)
    }
    fn prefill(&mut self, ids: &[u32], caches: &mut Vec<Vec<u32>>) -> anyhow::Result<Tensor> {
        self.append(ids, caches)
    }
    fn prefill_multi(
        &mut self,
        ids: &[u32],
        caches: &mut Vec<Vec<u32>>,
    ) -> anyhow::Result<(Tensor, Tensor)> {
        let h = self.append(ids, caches)?;
        Ok((h.clone(), h))
    }
    fn decode_token(&mut self, id: u32, caches: &mut Vec<Vec<u32>>) -> anyhow::Result<Tensor> {
        self.runtime.events.borrow_mut().push(Event::Host(id));
        self.append(&[id], caches)
    }
    fn supports_device_serial_decode(&self) -> bool {
        self.eligible
    }
    fn decode_generated_token(
        &mut self,
        ids: &Tensor,
        caches: &mut Vec<Vec<u32>>,
    ) -> anyhow::Result<Tensor> {
        let id = ids.values[0];
        self.runtime.events.borrow_mut().push(Event::Device(id));
        let hidden = self.append(&[id], caches)?;
        anyhow::ensure!(
            self.fail_device != Some(id),
            "injected pending append failure"
        );
        Ok(hidden)
    }
    fn head(&self, hidden: &Tensor) -> Result<Tensor, std::io::Error> {
        let id = hidden.values[0];
        Ok(Tensor::rows(&[if self.eos_after == Some(id) {
            EOS
        } else {
            id + 1
        }]))
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
        self.tails.clear();
    }
}

fn same(a: &Conversation<Runtime>, b: &Conversation<Runtime>, ma: &Model, mb: &Model) {
    assert_eq!(a.fed, b.fed);
    assert_eq!(a.caches, b.caches);
    assert_eq!(a.last_mixed, b.last_mixed);
    assert_eq!(ma.tails, mb.tails);
    assert_eq!(a.is_valid(), b.is_valid());
}

#[test]
fn queued_serial_respects_context_end_and_restores_snapshots_across_turns() -> anyhow::Result<()> {
    let _scope = crate::eos::ignore_eos_scope();
    let mut serial = Model::new(false);
    let mut queued = Model::new(true);
    serial.capacity = 5;
    queued.capacity = 5;
    let mut a = Conversation::new(&mut serial);
    let mut b = Conversation::new(&mut queued);
    a.enable_snapshots();
    b.enable_snapshots();
    let mut sa = Sampler::default();
    let mut sb = sa.clone();
    for (model, session) in [(&mut serial, &mut a), (&mut queued, &mut b)] {
        session.feed(model, &[1])?;
        session.capture_snapshot(1, 0);
    }
    for _ in 0..2 {
        let ao = a.generate(&mut serial, &[2], 100, &mut sa, None, &mut |_| Ok(()))?;
        let bo = b.generate(&mut queued, &[2], 100, &mut sb, None, &mut |_| Ok(()))?;
        assert_eq!(ao, bo);
        assert_eq!(ao.len(), 4);
        same(&a, &b, &serial, &queued);
        assert_eq!(a.fed.len(), 5);
        assert_eq!(b.fed.last(), Some(&5));
        assert!(
            !queued.runtime.events.borrow().contains(&Event::Drain),
            "context termination must not add a cleanup wait"
        );
        assert_eq!(a.restore_snapshot_exact(&mut serial, 1), Some(1));
        assert_eq!(b.restore_snapshot_exact(&mut queued, 1), Some(1));
        same(&a, &b, &serial, &queued);
    }
    serial.capacity = 32;
    queued.capacity = 32;
    for suffix in [&[2u32][..], &[6u32, 20][..]] {
        let ao = a.generate(&mut serial, suffix, 4, &mut sa, None, &mut |_| Ok(()))?;
        let bo = b.generate(&mut queued, suffix, 4, &mut sb, None, &mut |_| Ok(()))?;
        assert_eq!(ao, bo);
        same(&a, &b, &serial, &queued);
    }
    Ok(())
}

#[test]
fn queued_serial_restores_pending_append_on_callback_or_model_failure() -> anyhow::Result<()> {
    let _scope = crate::eos::ignore_eos_scope();
    for stop in [1, 2, 3] {
        let mut serial = Model::new(false);
        let mut queued = Model::new(true);
        let mut a = Conversation::new(&mut serial);
        let mut b = Conversation::new(&mut queued);
        let mut sa = Sampler::default();
        let mut sb = sa.clone();
        let mut na = 0;
        let mut nb = 0;
        assert!(
            a.generate(&mut serial, &[1, 2], 6, &mut sa, None, &mut |_| {
                na += 1;
                anyhow::ensure!(na != stop, "cancel");
                Ok(())
            })
            .is_err()
        );
        assert!(
            b.generate(&mut queued, &[1, 2], 6, &mut sb, None, &mut |_| {
                nb += 1;
                anyhow::ensure!(nb != stop, "cancel");
                Ok(())
            })
            .is_err()
        );
        assert_eq!(na, nb);
        same(&a, &b, &serial, &queued);
        assert!(!b.is_valid());
        assert!(b.feed(&mut queued, &[10]).is_err());
    }
    let mut model = Model::new(true);
    model.fail_device = Some(4);
    let mut session = Conversation::new(&mut model);
    let mut emitted = Vec::new();
    assert!(
        session
            .generate(
                &mut model,
                &[1, 2],
                6,
                &mut Sampler::default(),
                None,
                &mut |id| {
                    emitted.push(id);
                    Ok(())
                }
            )
            .is_err()
    );
    assert_eq!(emitted, [3, 4]);
    assert_eq!(session.fed, [1, 2, 3]);
    assert_eq!(session.caches[0], session.fed);
    assert_eq!(model.tails, vec![vec![2, 3]]);
    assert!(!session.is_valid());
    Ok(())
}

#[test]
fn queued_serial_emits_current_token_before_future_error_and_drains_terminal_work()
-> anyhow::Result<()> {
    let _scope = crate::eos::ignore_eos_scope();
    crate::eos::set_eos_ids(vec![EOS]);
    for (eos, cancel, fail_finish) in [
        (true, false, false),
        (false, false, false),
        (false, true, false),
        (true, false, true),
    ] {
        let mut model = Model::new(true);
        model.eos_after = eos.then_some(3);
        let current = if eos { EOS } else { 4 };
        model.fail_device = Some(current);
        model.runtime.fail_finish.set(fail_finish);
        let mut session = Conversation::new(&mut model);
        let mut emitted = Vec::new();
        let result = session.generate(
            &mut model,
            &[1, 2],
            6,
            &mut Sampler::default(),
            None,
            &mut |id| {
                emitted.push(id);
                anyhow::ensure!(!cancel || emitted.len() != 2, "injected callback stop");
                Ok(())
            },
        );
        assert_eq!(emitted, [3, current]);
        assert_eq!(session.fed, [1, 2, 3]);
        assert_eq!(session.caches[0], session.fed);
        assert_eq!(model.tails, vec![vec![2, 3]]);
        assert_eq!(
            model
                .runtime
                .events
                .borrow()
                .iter()
                .filter(|e| **e == Event::Drain)
                .count(),
            1
        );
        if fail_finish {
            assert!(
                result
                    .unwrap_err()
                    .to_string()
                    .contains("completion failure")
            );
            assert!(!session.is_valid());
        } else if eos {
            assert_eq!(result?, emitted);
            assert!(session.is_valid());
        } else {
            let err = result.unwrap_err().to_string();
            assert!(err.contains(if cancel {
                "callback stop"
            } else {
                "pending append failure"
            }));
            assert!(!session.is_valid());
        }
    }
    for read_failure in [false, true] {
        let mut model = Model::new(true);
        if read_failure {
            model.runtime.fail_read.set(Some(4));
        } else {
            model.runtime.fail_queue.set(Some(5));
        }
        let mut session = Conversation::new(&mut model);
        let mut emitted = Vec::new();
        assert!(
            session
                .generate(
                    &mut model,
                    &[1, 2],
                    6,
                    &mut Sampler::default(),
                    None,
                    &mut |id| {
                        emitted.push(id);
                        Ok(())
                    }
                )
                .is_err()
        );
        assert_eq!(emitted, if read_failure { vec![3] } else { vec![3, 4] });
        assert_eq!(session.fed, [1, 2, 3]);
        assert_eq!(session.caches[0], session.fed);
        assert_eq!(
            model
                .runtime
                .events
                .borrow()
                .iter()
                .filter(|e| **e == Event::Drain)
                .count(),
            1
        );
        assert!(!session.is_valid());
    }
    Ok(())
}
