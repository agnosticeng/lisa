//! Engine storage interface for speculative transactions. Providers expose only
//! tensor operations; acceptance, sampling history and cache commits stay in
//! the engine. All methods preserve the selected model's numerical law.
use crate::conversation::{SessionRuntime, TokenSampler};

pub use lisa_backend::TensorAccess as SpeculativeTensor;

pub trait SpeculativeSampler: Clone {
    type Tensor;
    type Context;
    fn is_greedy(&self) -> bool;
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
    fn draw_ds4(
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
    type VerifyScope;
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
    fn verify_scope(&self) -> Self::VerifyScope;
    fn trace(&self, name: &'static str) -> Self::Trace;
    fn trace_detail(&self, name: &'static str, detail: u64) -> Self::Trace;
    fn trace_enabled(&self, name: &str) -> bool;
}
