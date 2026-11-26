//! Engine storage interface for speculative transactions. Providers expose only
//! tensor operations; acceptance, sampling history and cache commits stay in
//! the engine. All methods preserve the selected model's numerical law.
use crate::conversation::{SessionRuntime, TokenSampler};

pub use lisa_backend::TensorAccess as SpeculativeTensor;

/// A greedy device ID and an independent host mirror. Only `ids` may enter a
/// later forward; `mirror` must never gain another GPU consumer before readback.
pub struct QueuedGreedyToken<T> {
    pub ids: T,
    pub mirror: T,
}

pub trait SpeculativeSampler: Clone {
    type Tensor;
    type Context;
    fn is_greedy(&self) -> bool;
    /// A configured greedy sampler can still transform logits using history.
    /// Such a sampler must not be replaced by an unmodified argmax.
    fn can_use_raw_argmax(&self) -> bool {
        self.is_greedy() && self.repetition_penalty() == 1.0
    }
    /// The ordinary greedy branch is the runtime's exact argmax law and has
    /// no sampling-state side effects. Other samplers keep the host draw path.
    fn supports_queued_greedy(&self) -> bool {
        false
    }
    fn exact_sampling(&self) -> bool;
    fn temperature(&self) -> f32;
    fn repetition_penalty(&self) -> f32;
    fn penalty_window(&self) -> usize;
    fn draw(
        &mut self,
        context: &Self::Context,
        logits: &Self::Tensor,
        history: &[u32],
    ) -> anyhow::Result<u32>;
    fn draw_host(
        &mut self,
        context: &Self::Context,
        logits: &Self::Tensor,
        history: &[u32],
    ) -> anyhow::Result<u32>;
    fn draw_at(&self, context: &Self::Context, logits: &Self::Tensor, position: u64)
    -> Option<u32>;
    fn draw_row(&mut self, context: &Self::Context, logits: &Self::Tensor) -> Option<u32>;
    fn draft_accept_prob(
        &self,
        context: &Self::Context,
        logits: &Self::Tensor,
        draft: u32,
    ) -> Option<f32>;
    fn accept_draw(&mut self, p: f32) -> bool;
    fn draw_residual(
        &mut self,
        context: &Self::Context,
        logits: &Self::Tensor,
        draft: u32,
    ) -> Option<u32>;
    fn verify_draft_with_history(
        &mut self,
        context: &Self::Context,
        logits: &Self::Tensor,
        draft: u32,
        history: &[u32],
    ) -> Option<(bool, u32)>;
}

pub trait SpeculativeRuntime: SessionRuntime<Tensor: SpeculativeTensor> + Sized {
    type Sampler: SpeculativeSampler<Tensor = Self::Tensor, Context = Self> + TokenSampler<Self>;
    type Rollback;
    type SerialSnapshot;
    type Trace;
    /// Prepare target logits for sampling without changing the model's raw
    /// head. The identity default preserves runtimes with no output policy.
    fn sampling_logits(&self, logits: &Self::Tensor) -> anyhow::Result<Self::Tensor> {
        Ok(logits.clone())
    }
    fn supports_queued_greedy(&self) -> bool {
        false
    }
    /// Select with the existing ordinary greedy law, create an independent
    /// I32 host mirror, and submit its producer before any later forward.
    fn queue_greedy_token(
        &self,
        _logits: &Self::Tensor,
    ) -> anyhow::Result<QueuedGreedyToken<Self::Tensor>> {
        anyhow::bail!("queued greedy sampling is not supported")
    }
    /// Wait only for the mirror's producer where the provider supports it.
    fn read_queued_token(&self, _token: &QueuedGreedyToken<Self::Tensor>) -> anyhow::Result<u32> {
        anyhow::bail!("queued greedy sampling is not supported")
    }
    /// Complete submitted and partially encoded queued work on terminal/error
    /// exits. A runtime opting into queued selection must override this unless
    /// its operations are synchronous. Completion failures must remain errors.
    fn finish_queued_work(&self) -> anyhow::Result<()> {
        Ok(())
    }
    fn tokens_u32(&self, ids: &[u32], shape: &[i32]) -> anyhow::Result<Self::Tensor>;
    fn tokens(&self, ids: &[i32], shape: &[i32]) -> anyhow::Result<Self::Tensor>;
    fn snapshot_rollback(cache: &Self::Cache) -> Self::Rollback;
    fn restore_rollback(cache: &mut Self::Cache, state: &Self::Rollback) -> anyhow::Result<()>;
    fn serial_snapshot(cache: &Self::Cache) -> Self::SerialSnapshot;
    fn restore_serial_eos(cache: &mut Self::Cache, snapshot: &Self::SerialSnapshot);
    fn rollback(
        &self,
        cache: &mut Self::Cache,
        advance: usize,
        keep: usize,
        captured: bool,
    ) -> anyhow::Result<()>;
    fn trim_memory(&self);
    fn trace(&self, name: &'static str) -> Self::Trace;
    fn trace_detail(&self, name: &'static str, detail: u64) -> Self::Trace;
    fn trace_enabled(&self, name: &str) -> bool;
}
