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

    /// Olala's serial prefill law is invariant to these chunk boundaries.
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
        if !enabled {
            return Ok((crate::backend::new_chat_session(model), 0));
        }
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
        // Leave a real final token for generate(), even for identical prompts.
        // Feeding it twice after restoring a full-prompt state would be wrong.
        let split = ids.len() - 1;
        let mut fed = reused;
        if split >= lisa_engine::conversation_cache::MIN_PREFIX_TOKENS {
            while fed < split {
                let end = (fed + 128).min(split);
                session.feed(model, &ids[fed..end])?;
                cache.insert_model(model, &ids[..end], &session.caches);
                fed = end;
                progress(reused, fed)?;
            }
        }
        Ok((session, fed))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires LISA_OLALA_MODEL pointing to the pinned Q5_K_M GGUF"]
    fn olala_prompt_reuse_matches_full_prefill() -> anyhow::Result<()> {
        let path = std::env::var_os("LISA_OLALA_MODEL")
            .ok_or_else(|| anyhow::anyhow!("set LISA_OLALA_MODEL to the pinned Q5_K_M GGUF"))?;
        let path = std::path::Path::new(&path);
        anyhow::ensure!(
            lisa_engine::checkpoint::model_type_of(path)? == "olala",
            "expected Olala"
        );
        let lisa_serve::application::loading::Loaded::Language(mut model) =
            lisa_serve::application::loading::load_dir(path)?
        else {
            anyhow::bail!("expected a language model");
        };
        let tokenizer = lisa_engine::tokenizer::Tokenizer::load(path)?;
        let template = lisa_serve::ChatTemplate::new(
            &lisa_engine::tokenizer::load_chat_template(path)?
                .ok_or_else(|| anyhow::anyhow!("missing template"))?,
        )?;
        let msg = |role: &str, content: &str| lisa_serve::ChatMessage {
            role: role.into(),
            content: content.into(),
            reasoning_content: None,
            tool_calls: vec![],
            tool_call_id: None,
            parts: None,
        };
        let mut messages = vec![msg("user", "What is the capital of France?")];
        let ids = tokenizer.encode(&template.render(&messages, true, &[], false)?, false)?;
        // Warmup must never replay the complete request or validate/use its tail.
        model.warmup(&[ids[0], u32::MAX])?;
        model.warmup(&[u32::MAX])?;
        let mut prefix = PromptCache::default();
        let bits = |logits: &lisa_metal::Array| {
            logits
                .as_slice::<f32>()
                .iter()
                .map(|x| x.to_bits())
                .collect::<Vec<_>>()
        };
        let mut first_bits = None;
        for turn in 0..3 {
            let prompt = if turn < 2 {
                ids.clone()
            } else {
                messages.push(msg("assistant", "Paris is the capital of France."));
                messages.push(msg("user", "Which country is that in?"));
                tokenizer.encode(&template.render(&messages, true, &[], false)?, false)?
            };
            let mut reused = None;
            let (mut session, offset) = prefix.prepare(&mut *model, &prompt, true, |n, _| {
                reused.get_or_insert(n);
                Ok(())
            })?;
            assert_eq!(offset, prompt.len() - 1);
            if turn == 0 {
                assert_eq!(reused, Some(0));
            } else {
                assert!(reused.unwrap() >= ids.len() - 1);
            }
            let resumed = session.feed(&mut *model, &prompt[offset..])?;
            let resumed = bits(&resumed);
            let mut fresh = crate::backend::new_chat_session(&mut *model);
            let full = fresh.feed(&mut *model, &prompt)?;
            assert_eq!(resumed, bits(&full), "turn {turn}");
            if turn == 0 {
                first_bits = Some(resumed);
            } else if turn == 1 {
                assert_eq!(Some(resumed), first_bits);
            }
            // Generation can advance independently without corrupting the saved prompt.
            session.decode_token(&mut *model, 40)?;
        }
        let mut changed = ids.clone();
        changed[0] ^= 1;
        assert!(
            prefix
                .inner
                .as_mut()
                .unwrap()
                .restore_session(&mut *model, &changed)
                .is_none()
        );
        let (_, offset) = prefix.prepare(&mut *model, &ids, false, |_, _| unreachable!())?;
        assert_eq!(
            offset, 0,
            "image/non-admitted requests must not restore text state"
        );
        prefix.clear();
        assert!(prefix.inner.is_none());
        Ok(())
    }
}
