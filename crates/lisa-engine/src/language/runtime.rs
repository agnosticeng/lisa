//! Concrete engine session adapter over portable device and tensor contracts.
use crate::{
    cache::{
        LayerCache, LayerRollbackState, LayerState, SessionSnapshot,
        indexer::{CacheDevice, IndexerTape, TapeSnapshot},
    },
    conversation::{SessionRuntime, TokenSampler},
    execution::DeviceContext,
    language::LanguageRuntime,
    sampling::{Sampler, SamplingContext},
    speculation::runtime::{SpeculativeRuntime, SpeculativeSampler},
    tensor_ops::{TensorLaw, TensorOps},
};
use lisa_backend::{ManagedDevice, ReadbackDevice, ScalarType, TensorAccess};
use std::marker::PhantomData;

pub struct Runtime<D: CacheDevice, S> {
    pub context: DeviceContext<D>,
    law: TensorLaw,
    sampler: PhantomData<fn() -> S>,
}
impl<D: CacheDevice, S> Runtime<D, S> {
    pub fn new(context: DeviceContext<D>, law: TensorLaw) -> Self {
        Self {
            context,
            law,
            sampler: PhantomData,
        }
    }
}
impl<D: CacheDevice, S> LanguageRuntime for Runtime<D, S> {
    type Tensor = D::Tensor;
    type Cache = LayerCache<D>;
    type DraftCache = IndexerTape<D>;
    type Error = D::Error;
    fn shape(t: &D::Tensor) -> &[i32] {
        t.shape()
    }
    fn row(t: &D::Tensor, index: usize) -> anyhow::Result<D::Tensor> {
        let i = i32::try_from(index)?;
        anyhow::ensure!(t.rank() >= 2 && i < t.dim(0), "language row out of bounds");
        t.slice(0, i, Some(i + 1))
    }
    fn join_rows(t: &[&D::Tensor]) -> anyhow::Result<D::Tensor> {
        D::Tensor::concatenate(t, 0)
    }
}
impl<D: CacheDevice, S> SessionRuntime for Runtime<D, S> {
    type Snapshot = SessionSnapshot<D::Tensor>;
    type DraftSnapshot = TapeSnapshot<D::Tensor>;
    type PrefixState = LayerState<D::Tensor>;
    fn snapshot_cache(c: &Self::Cache) -> Self::Snapshot {
        c.checkpoint()
    }
    fn restore_cache(c: &mut Self::Cache, s: &Self::Snapshot) {
        c.restore_checkpoint(s)
    }
    fn snapshot_draft(c: &Self::DraftCache) -> Self::DraftSnapshot {
        c.snapshot()
    }
    fn restore_draft(c: &mut Self::DraftCache, s: &Self::DraftSnapshot) {
        c.restore(s)
    }
    fn restore_prefix(c: &mut Self::Cache, s: &Self::PrefixState) {
        c.restore_prefix(s)
    }
    fn hidden_prefix(t: &D::Tensor, rows: usize, keep: usize) -> anyhow::Result<D::Tensor> {
        t.reshape(&[1, rows as i32, t.dim(-1)])?
            .slice(1, 0, Some(keep as i32))?
            .contiguous()
    }
    fn new_draft_cache(&self) -> Self::DraftCache {
        IndexerTape::new(self.context.clone())
    }
}
impl<D, S> Runtime<D, S>
where
    D: CacheDevice + ManagedDevice + ReadbackDevice,
{
    pub fn sampling(&self) -> SamplingContext<'_, D> {
        SamplingContext::with_observer(self.context.device(), self.law, self.context.observer())
    }
    fn pull(&self, bytes: &[u8], shape: &[i32], dtype: ScalarType) -> anyhow::Result<D::Tensor> {
        TensorOps::new(self.context.device(), self.law).pull(bytes, shape, dtype)
    }
}
impl<D, S> crate::batching::BatchRuntime for Runtime<D, S>
where
    D: CacheDevice + ManagedDevice + ReadbackDevice,
    S: TokenSampler<Self> + AsMut<Sampler>,
{
    type Sampler = S;
    fn tokens(&self, v: &[i32], shape: &[i32]) -> anyhow::Result<D::Tensor> {
        self.pull(
            &v.iter().flat_map(|v| v.to_ne_bytes()).collect::<Vec<_>>(),
            shape,
            ScalarType::I32,
        )
    }
    fn mask(&self, v: &[bool], shape: &[i32]) -> anyhow::Result<D::Tensor> {
        self.pull(
            &v.iter().map(|v| u8::from(*v)).collect::<Vec<_>>(),
            shape,
            ScalarType::Bool,
        )
    }
    fn last_hidden(t: &D::Tensor) -> anyhow::Result<D::Tensor> {
        t.select(1, t.dim(1) - 1)
    }
    fn full_offset(c: &Self::Cache) -> Option<usize> {
        c.full_offset()
    }
    fn has_positions(c: &Self::Cache) -> bool {
        c.has_positions()
    }
    fn set_mask(c: &mut Self::Cache, m: &D::Tensor) {
        c.set_mask(m)
    }
    fn advance_positions(c: &mut Self::Cache) {
        c.advance_positions()
    }
    fn clear_positions(c: &mut Self::Cache) {
        c.clear_positions()
    }
    fn trim_memory(&self) {
        self.context.device().trim_unused()
    }
    fn pack_layer(
        &self,
        per: &[Vec<Self::Cache>],
        layer: usize,
        lengths: &[usize],
        max_length: usize,
        capacity: usize,
    ) -> anyhow::Result<Self::Cache> {
        crate::cache::batching::pack_layer(&self.context, per, layer, lengths, max_length, capacity)
    }
    fn extract_streams(
        &self,
        caches: &[Self::Cache],
        survivors: &[usize],
        lengths: &[usize],
        max_length: usize,
    ) -> anyhow::Result<Vec<Vec<Self::Cache>>> {
        crate::cache::batching::extract_streams(caches, survivors, lengths, max_length)
    }
    fn draw_rows(
        &self,
        logits: &D::Tensor,
        rows: usize,
        sampler: &mut S,
        history: &[Vec<u32>],
        ds4: bool,
    ) -> anyhow::Result<Vec<u32>> {
        crate::sampling::draw_rows(
            &self.sampling(),
            logits,
            rows,
            sampler.as_mut(),
            history,
            ds4,
        )
    }
    fn draw_rows_multi(
        &self,
        logits: &D::Tensor,
        rows: usize,
        samplers: &mut [S],
        history: &[Vec<u32>],
        ds4: bool,
    ) -> anyhow::Result<Vec<u32>> {
        crate::sampling::draw_rows_multi(&self.sampling(), logits, rows, samplers, history, ds4)
    }
}
impl<D, S> SpeculativeRuntime for Runtime<D, S>
where
    D: CacheDevice + ManagedDevice + ReadbackDevice,
    S: SpeculativeSampler<Tensor = D::Tensor, Context = Self> + TokenSampler<Self>,
{
    type Sampler = S;
    type Rollback = LayerRollbackState<D::Tensor>;
    type SerialSnapshot = Option<(Option<D::Tensor>, Option<D::Tensor>, Option<D::Tensor>)>;
    type Trace = Option<Box<dyn std::any::Any>>;
    fn sampling_logits(&self, logits: &D::Tensor) -> anyhow::Result<D::Tensor> {
        Ok(logits.clone())
    }
    fn supports_queued_greedy(&self) -> bool {
        true
    }
    fn queue_greedy_token(
        &self,
        logits: &D::Tensor,
    ) -> anyhow::Result<crate::speculation::runtime::QueuedGreedyToken<D::Tensor>> {
        use lisa_backend::TensorOperation as Op;
        let device = self.context.device();
        let ops = TensorOps::new(device, self.law);
        // This is exactly draw_on's ordinary greedy chain. The integer cast
        // serves the embedding ABI and does not change its selected index.
        let flat = self.sampling().flat_logits(logits)?;
        let selected = ops.op(
            Op::Argmax {
                axis: -1,
                keep_dims: false,
            },
            &[&flat],
        )?;
        let ids = ops
            .op(Op::Cast(ScalarType::I32), &[&selected])?
            .reshape(&[1, 1])?;
        let mirror = device.copy_tensor(&ids)?;
        // Separate the host mirror's producer from the following forward,
        // whose embedding will consume ids and advance its last-use fence.
        device.submit(false)?;
        Ok(crate::speculation::runtime::QueuedGreedyToken { ids, mirror })
    }
    fn read_queued_token(
        &self,
        token: &crate::speculation::runtime::QueuedGreedyToken<D::Tensor>,
    ) -> anyhow::Result<u32> {
        let values = self.context.device().read_i32_through(&token.mirror)?;
        anyhow::ensure!(
            values.len() == 1 && values[0] >= 0,
            "invalid queued greedy token"
        );
        Ok(values[0] as u32)
    }
    fn finish_queued_work(&self) -> anyhow::Result<()> {
        self.context.device().submit(true)
    }
    fn tokens_u32(&self, v: &[u32], shape: &[i32]) -> anyhow::Result<D::Tensor> {
        self.pull(
            &v.iter().flat_map(|v| v.to_ne_bytes()).collect::<Vec<_>>(),
            shape,
            ScalarType::U32,
        )
    }
    fn tokens(&self, v: &[i32], shape: &[i32]) -> anyhow::Result<D::Tensor> {
        self.pull(
            &v.iter().flat_map(|v| v.to_ne_bytes()).collect::<Vec<_>>(),
            shape,
            ScalarType::I32,
        )
    }
    fn snapshot_rollback(c: &Self::Cache) -> Self::Rollback {
        c.snapshot_rollback()
    }
    fn restore_rollback(c: &mut Self::Cache, s: &Self::Rollback) -> anyhow::Result<()> {
        c.restore_rollback(s)
    }
    fn serial_snapshot(c: &Self::Cache) -> Self::SerialSnapshot {
        c.serial_snapshot()
    }
    fn restore_serial_eos(c: &mut Self::Cache, s: &Self::SerialSnapshot) {
        c.restore_serial_eos(s)
    }
    fn rollback(
        &self,
        c: &mut Self::Cache,
        advance: usize,
        keep: usize,
        captured: bool,
    ) -> anyhow::Result<()> {
        c.rollback_verified(&self.context, advance, keep, captured)
    }
    fn trim_memory(&self) {
        self.context.device().trim_unused()
    }
    fn trace(&self, name: &'static str) -> Self::Trace {
        self.context.trace(name)
    }
    fn trace_detail(&self, name: &'static str, detail: u64) -> Self::Trace {
        self.context.trace_detail(name, detail)
    }
    fn trace_enabled(&self, name: &str) -> bool {
        self.context.trace_enabled(name)
    }
}
