//! Portable incremental-prefill diagnostic, with explicit model ownership.
use crate::{
    conversation::Conversation, language::LanguageModel, speculation::runtime::SpeculativeRuntime,
};
use lisa_backend::TensorAccess;

pub fn verify_incremental<R: SpeculativeRuntime>(
    model: &mut dyn LanguageModel<R>,
    tokens: &[u32],
    chunk: usize,
) -> anyhow::Result<(bool, f32, f32)> {
    anyhow::ensure!(!tokens.is_empty(), "incremental check needs tokens");
    let saved = model.context_tails();
    let result = (|| {
        let incremental =
            |model: &mut dyn LanguageModel<R>| -> anyhow::Result<(R::Tensor, R::Tensor)> {
                model.clear_context();
                let mut session = Conversation::new(model);
                let mut logits = None;
                for part in tokens.chunks(chunk.max(1)) {
                    logits = Some(session.feed(model, part)?);
                }
                Ok((logits.unwrap(), session.last_mixed.unwrap()))
            };
        let full = |model: &mut dyn LanguageModel<R>| -> anyhow::Result<(R::Tensor, R::Tensor)> {
            model.clear_context();
            let ids = model
                .runtime()
                .tokens_u32(tokens, &[1, tokens.len() as i32])?;
            let mut caches = model.new_caches();
            let (mixed, _) = model.forward(&ids, Some(&mut caches))?;
            let last = mixed.select(1, mixed.dim(1) - 1)?;
            Ok((model.head(&last)?, last))
        };
        let _ = incremental(model)?;
        let _ = full(model)?;
        let (inc, inc_hidden) = incremental(model)?;
        let (reference, reference_hidden) = full(model)?;
        let diff = |a: &R::Tensor, b: &R::Tensor| -> anyhow::Result<f32> {
            let a = a.read_f32()?;
            let b = b.read_f32()?;
            anyhow::ensure!(a.len() == b.len(), "diagnostic shape mismatch");
            Ok(a.iter()
                .zip(&b)
                .map(|(a, b)| (a - b).abs())
                .fold(0., f32::max))
        };
        Ok((
            inc.argmax_id()? == reference.argmax_id()?,
            diff(&inc, &reference)?,
            diff(&inc_hidden, &reference_hidden)?,
        ))
    })();
    model.set_context_tails(saved);
    result
}
