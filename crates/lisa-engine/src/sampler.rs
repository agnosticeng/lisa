//! Sampling configuration with explicit session execution ownership.
use crate::{indexer_cache::CacheDevice, language_runtime::Runtime};
use lisa_backend::{ManagedDevice, ReadbackDevice};
use std::marker::PhantomData;
#[derive(Clone)]
pub struct Sampler<D: CacheDevice>(pub crate::sampling::Sampler, PhantomData<fn() -> D>);
impl<D: CacheDevice> Sampler<D> {
    pub fn new(config: crate::sampling::Sampler) -> Self {
        Self(config, PhantomData)
    }
}
impl<D: CacheDevice> Default for Sampler<D> {
    fn default() -> Self {
        Self::new(Default::default())
    }
}
impl<D: CacheDevice> std::fmt::Debug for Sampler<D> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}
impl<D: CacheDevice> std::ops::Deref for Sampler<D> {
    type Target = crate::sampling::Sampler;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl<D: CacheDevice> std::ops::DerefMut for Sampler<D> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}
impl<D: CacheDevice> AsMut<crate::sampling::Sampler> for Sampler<D> {
    fn as_mut(&mut self) -> &mut crate::sampling::Sampler {
        &mut self.0
    }
}
impl<D: CacheDevice + ManagedDevice + ReadbackDevice> Sampler<D> {
    pub fn greedy(&self) -> bool {
        self.0.greedy()
    }
    pub fn accept_draw(&mut self, p: f32) -> bool {
        self.0.accept_draw(p)
    }
    pub fn accept_draw_ds4(&mut self, p: f32) -> bool {
        self.0.accept_draw_ds4(p)
    }

    pub fn draft_accept_prob(
        &self,
        runtime: &Runtime<D, Self>,
        logits: &D::Tensor,
        draft: u32,
    ) -> Option<f32> {
        self.draft_accept_prob_on(&runtime.sampling(), logits, draft)
    }
    pub fn draft_accept_prob_with_history(
        &self,
        runtime: &Runtime<D, Self>,
        logits: &D::Tensor,
        draft: u32,
        history: &[u32],
    ) -> Option<f32> {
        self.draft_accept_prob_with_history_on(&runtime.sampling(), logits, draft, history)
    }
    pub fn verify_draft_with_history(
        &mut self,
        runtime: &Runtime<D, Self>,
        logits: &D::Tensor,
        draft: u32,
        history: &[u32],
    ) -> Option<(bool, u32)> {
        self.verify_draft_with_history_on(&runtime.sampling(), logits, draft, history)
    }
    pub fn draw_residual(
        &mut self,
        runtime: &Runtime<D, Self>,
        logits: &D::Tensor,
        draft: u32,
    ) -> Option<u32> {
        self.draw_residual_on(&runtime.sampling(), logits, draft)
    }
    pub fn draw_residual_with_history(
        &mut self,
        runtime: &Runtime<D, Self>,
        logits: &D::Tensor,
        draft: u32,
        history: &[u32],
    ) -> Option<u32> {
        self.draw_residual_with_history_on(&runtime.sampling(), logits, draft, history)
    }
    pub fn draw_row(&mut self, runtime: &Runtime<D, Self>, logits: &D::Tensor) -> Option<u32> {
        self.draw_row_on(&runtime.sampling(), logits)
    }
    pub fn draw_row_with_history(
        &mut self,
        runtime: &Runtime<D, Self>,
        logits: &D::Tensor,
        history: &[u32],
    ) -> Option<u32> {
        self.draw_row_with_history_on(&runtime.sampling(), logits, history)
    }
    pub fn draw_ds4(
        &mut self,
        runtime: &Runtime<D, Self>,
        logits: &D::Tensor,
        history: &[u32],
    ) -> anyhow::Result<u32> {
        self.draw_ds4_on(&runtime.sampling(), logits, history)
    }
    pub fn draw_at(
        &self,
        runtime: &Runtime<D, Self>,
        logits: &D::Tensor,
        position: u64,
    ) -> Option<u32> {
        self.draw_at_on(&runtime.sampling(), logits, position)
    }
    pub fn draw(
        &mut self,
        runtime: &Runtime<D, Self>,
        logits: &D::Tensor,
        history: &[u32],
    ) -> anyhow::Result<u32> {
        self.draw_on(&runtime.sampling(), logits, history)
    }
}
impl<D: CacheDevice + ManagedDevice + ReadbackDevice>
    crate::conversation::TokenSampler<Runtime<D, Self>> for Sampler<D>
{
    fn greedy(&mut self, _runtime: &Runtime<D, Self>, logits: &D::Tensor) -> anyhow::Result<u32> {
        lisa_backend::TensorAccess::argmax_id(logits)
    }
    fn draw(
        &mut self,
        runtime: &Runtime<D, Self>,
        logits: &D::Tensor,
        history: &[u32],
    ) -> anyhow::Result<u32> {
        Self::draw(self, runtime, logits, history)
    }
    fn draw_ds4(
        &mut self,
        runtime: &Runtime<D, Self>,
        logits: &D::Tensor,
        history: &[u32],
    ) -> anyhow::Result<u32> {
        Self::draw_ds4(self, runtime, logits, history)
    }
}

impl<D: CacheDevice + ManagedDevice + ReadbackDevice> crate::speculative_runtime::SpeculativeSampler
    for Sampler<D>
{
    type Tensor = D::Tensor;
    type Context = Runtime<D, Self>;
    fn is_greedy(&self) -> bool {
        self.greedy()
    }
    fn exact_sampling(&self) -> bool {
        self.mtp_exact_sampling
    }
    fn temperature(&self) -> f32 {
        self.temperature
    }
    fn repetition_penalty(&self) -> f32 {
        self.repetition_penalty
    }
    fn penalty_window(&self) -> usize {
        self.penalty_window
    }
    fn draw(
        &mut self,
        runtime: &Self::Context,
        logits: &D::Tensor,
        history: &[u32],
    ) -> anyhow::Result<u32> {
        Self::draw(self, runtime, logits, history)
    }
    fn draw_ds4(
        &mut self,
        runtime: &Self::Context,
        logits: &D::Tensor,
        history: &[u32],
    ) -> anyhow::Result<u32> {
        Self::draw_ds4(self, runtime, logits, history)
    }
    fn draw_at(&self, runtime: &Self::Context, logits: &D::Tensor, position: u64) -> Option<u32> {
        Self::draw_at(self, runtime, logits, position)
    }
    fn draw_row(&mut self, runtime: &Self::Context, logits: &D::Tensor) -> Option<u32> {
        Self::draw_row(self, runtime, logits)
    }
    fn draft_accept_prob(
        &self,
        runtime: &Self::Context,
        logits: &D::Tensor,
        draft: u32,
    ) -> Option<f32> {
        Self::draft_accept_prob(self, runtime, logits, draft)
    }
    fn accept_draw(&mut self, p: f32) -> bool {
        Self::accept_draw(self, p)
    }
    fn draw_residual(
        &mut self,
        runtime: &Self::Context,
        logits: &D::Tensor,
        draft: u32,
    ) -> Option<u32> {
        Self::draw_residual(self, runtime, logits, draft)
    }
    fn verify_draft_with_history(
        &mut self,
        runtime: &Self::Context,
        logits: &D::Tensor,
        draft: u32,
        history: &[u32],
    ) -> Option<(bool, u32)> {
        Self::verify_draft_with_history(self, runtime, logits, draft, history)
    }
}

pub fn token_logprob<D: CacheDevice + ManagedDevice + ReadbackDevice>(
    runtime: &Runtime<D, Sampler<D>>,
    logits: &D::Tensor,
    token: u32,
) -> anyhow::Result<f32> {
    crate::sampling::token_logprob(&runtime.sampling(), logits, token)
}
