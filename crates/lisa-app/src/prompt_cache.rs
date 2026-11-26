//! Text-only prompt reuse, scoped to the loaded model and active conversation.
use lisa_engine::language::LanguageModel;
use lisa_serve::application::{LanguageRuntime, PrefixCache, Session};

#[derive(Default)]
pub(crate) struct PromptCache {
    inner: Option<PrefixCache>,
}

impl PromptCache {
    pub(crate) fn clear(&mut self) {
        self.inner = None;
    }

    /// Preserve each admitted model's natural prefill chunk boundaries.
    /// Keep two immutable prompt snapshots in RAM, never generated tokens or
    /// image embeddings. Exact token matching and the model law gate restore.
    pub(crate) fn prepare(
        &mut self,
        model: &mut dyn LanguageModel<LanguageRuntime>,
        ids: &[u32],
        enabled: bool,
        mut progress: impl FnMut(usize, usize) -> anyhow::Result<()>,
    ) -> anyhow::Result<(Session, usize)> {
        anyhow::ensure!(!ids.is_empty(), "empty chat prompt");
        anyhow::ensure!(
            ids.len() < model.max_position_embeddings(),
            "chat prompt fills the context window"
        );
        let grain = model.prefix_cache_granularity().filter(|g| *g > 0);
        if !enabled || grain.is_none() || model.cache_numerical_law().is_none() {
            return Ok((crate::backend::new_chat_session(model), 0));
        }
        let grain = grain.expect("admitted above");
        let cache = self.inner.get_or_insert_with(|| {
            PrefixCache::new(model.runtime().context.device().clone(), 2, None)
        });
        let (reused, mut session) = cache
            .restore_session(model, ids)
            .unwrap_or_else(|| (0, crate::backend::new_chat_session(model)));
        session.set_target_mtp_eos_policy(
            lisa_engine::speculation::TargetMtpEosPolicy::StopBeforeVerified,
        );
        progress(reused, reused)?;
        // Keep the original last chunk intact: N-1 is safe only for serial
        // prefill. Repartitioning a batched tail changes its arithmetic.
        let split = (ids.len() - 1) / grain * grain;
        let step = model.preferred_prefill_step().max(128).div_ceil(grain) * grain;
        let mut fed = reused;
        if split >= lisa_engine::conversation::cache::MIN_PREFIX_TOKENS {
            while fed < split {
                let end = (fed + step).min(split);
                session.feed(model, &ids[fed..end])?;
                cache.insert_model(model, &ids[..end], &session.caches);
                fed = end;
                progress(reused, fed)?;
            }
        }
        Ok((session, fed))
    }
}
