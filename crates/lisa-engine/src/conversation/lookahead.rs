//! One pending serial append over an explicitly opted-in model and sampler.
//! Sampling and commit policy stay here; providers only submit tensor work.
use crate::{
    conversation::Conversation,
    eos::is_eos,
    language::LanguageModel,
    speculation::runtime::{SpeculativeRuntime, SpeculativeSampler},
};

impl<R: SpeculativeRuntime> Conversation<R> {
    pub(crate) fn can_queue_serial(
        &self,
        tower: &dyn LanguageModel<R>,
        sampler: &R::Sampler,
    ) -> bool {
        sampler.can_use_raw_argmax()
            && sampler.supports_queued_greedy()
            && !tower.uses_host_sampling()
            && tower.supports_device_serial_decode()
            && tower.runtime().supports_queued_greedy()
    }

    /// The first token has already been sampled and emitted synchronously.
    /// Every subsequent candidate may have one uncommitted S1 append in flight.
    /// `fed` and `last_mixed` advance only after the candidate is accepted by
    /// the callback and is not EOS. The final budget token is never fed.
    pub(crate) fn generate_serial_lookahead(
        &mut self,
        tower: &mut dyn LanguageModel<R>,
        out: &mut Vec<u32>,
        max_tokens: usize,
        on_token: &mut dyn FnMut(u32) -> anyhow::Result<()>,
    ) -> anyhow::Result<()> {
        let first = *out
            .last()
            .ok_or_else(|| anyhow::anyhow!("missing synchronous carry"))?;
        if out.len() >= max_tokens || is_eos(first) {
            return Ok(());
        }
        let _trace = tower
            .runtime()
            .trace_detail("decode.serial_lookahead", max_tokens as u64);
        // A terminal/error path must finish any partially encoded future work.
        // The final ordinary mirror read clears this flag only when no later
        // append exists; the first-token early exits above never set it.
        let mut cleanup_needed = true;
        let result = (|| -> anyhow::Result<()> {
            // This known carry is already accepted. Its ordinary serial append
            // preserves the existing first-token timing and prefix boundary.
            let logits = self.decode_token(tower, first)?;
            let mut token = tower.runtime().queue_greedy_token(&logits)?;
            loop {
                let can_advance =
                    out.len() + 1 < max_tokens && self.fed.len() < tower.max_position_embeddings();
                let saved = can_advance.then(|| {
                    (
                        self.caches
                            .iter()
                            .map(R::snapshot_rollback)
                            .collect::<Vec<_>>(),
                        tower.context_tails(),
                    )
                });
                // Keep a future encoding error separate from the earlier,
                // already submitted selection. Serial execution emits that
                // token before attempting this append, and EOS needs no append.
                let advance = (|| -> anyhow::Result<_> {
                    if !can_advance {
                        return Ok(None);
                    }
                    cleanup_needed = true;
                    let hidden = tower.decode_generated_token(&token.ids, &mut self.caches)?;
                    let logits = tower.head(&hidden)?;
                    let next = tower.runtime().queue_greedy_token(&logits)?;
                    Ok(Some((hidden, next)))
                })();
                let emitted = (|| -> anyhow::Result<_> {
                    // The independent mirror has no later consumer, even though
                    // the device ID entered the pending append.
                    let id = tower.runtime().read_queued_token(&token)?;
                    if !can_advance {
                        cleanup_needed = false;
                    }
                    out.push(id);
                    on_token(id)?;
                    Ok(id)
                })();
                let stop = match &emitted {
                    Ok(id) => is_eos(*id),
                    Err(_) => true,
                };
                if (stop || advance.is_err())
                    && let Some((snapshots, tails)) = saved
                {
                    for (cache, snapshot) in self.caches.iter_mut().zip(&snapshots) {
                        R::restore_rollback(cache, snapshot)?;
                    }
                    tower.set_context_tails(tails);
                }
                let id = emitted?;
                if is_eos(id) {
                    // The failed/successful speculative append is unnecessary.
                    // The outer cleanup still checks its GPU completion.
                    return Ok(());
                }
                let Some((hidden, next)) = advance? else {
                    return Ok(());
                };
                self.fed.push(id);
                self.last_mixed = Some(hidden);
                token = next;
            }
        })();
        if cleanup_needed {
            // Run even when rollback failed. Never hide a provider completion
            // error behind a successful EOS or a callback cancellation.
            tower.runtime().finish_queued_work()?;
        }
        result
    }
}
