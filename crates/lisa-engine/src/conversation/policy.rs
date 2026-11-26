//! Request-level driver selection, independent of model formats and devices.
use super::Conversation;
use crate::{
    language::LanguageModel,
    speculation::{
        MtpDepth,
        runtime::{SpeculativeRuntime, SpeculativeSampler},
    },
};

/// A frontend's permitted generation strategy for the current request.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum GenerationPolicy {
    /// Preserve ordinary sampling and the model's serial lookahead capability.
    #[default]
    Serial,
    /// Use automatic block drafting only when both model and sampler support
    /// it. Other draft policies and configured sampling stay serial.
    AdaptiveBlock,
}

impl<R: SpeculativeRuntime> Conversation<R> {
    /// Select an eligible shared driver without changing request callbacks,
    /// sampling history or the model's current capabilities. In particular,
    /// frontends can keep image/tool requests serial without inspecting the
    /// loaded checkpoint's format or concrete runtime.
    pub fn generate_with_policy(
        &mut self,
        tower: &mut dyn LanguageModel<R>,
        tokens: &[u32],
        max_tokens: usize,
        sampler: &mut R::Sampler,
        policy: GenerationPolicy,
        after_prefill: Option<&mut dyn FnMut(&[R::Cache])>,
        on_token: &mut dyn FnMut(u32) -> anyhow::Result<()>,
    ) -> anyhow::Result<Vec<u32>> {
        if policy == GenerationPolicy::AdaptiveBlock
            && tower.has_block_drafter()
            && tower.supports_tree_verify()
            && sampler.can_use_raw_argmax()
        {
            self.generate_mtp_depth(
                tower,
                tokens,
                max_tokens,
                MtpDepth::Auto,
                after_prefill,
                Some(sampler),
                on_token,
            )
        } else {
            self.generate(tower, tokens, max_tokens, sampler, after_prefill, on_token)
        }
    }
}
