//! Generation loop: prefill, decode, greedy and temperature sampling.
use crate::speculation::runtime::{SpeculativeRuntime, SpeculativeSampler, SpeculativeTensor};

use std::time::Instant;

use crate::language::LanguageModel;

pub struct GenerationStats {
    pub prompt_tokens: usize,
    /// All emitted tokens, including the first token produced by prefill.
    pub generated_tokens: usize,
    /// Prefill, output head and first sample, ending before its callback.
    /// Kernel warmup and the first token's callback are excluded.
    pub prefill_seconds: f64,
    /// Wall time from the first token's callback returning through the final
    /// decoded token's callback returning. Includes intervening scheduling,
    /// sampling and callbacks; excludes later cleanup and telemetry. Zero
    /// when generation emits no token after the prefill-produced first token.
    pub decode_seconds: f64,
}

impl GenerationStats {
    pub fn prefill_tps(&self) -> f64 {
        if self.prefill_seconds > 0.0 {
            self.prompt_tokens as f64 / self.prefill_seconds
        } else {
            0.0
        }
    }
    /// Decode throughput excludes the first token, which is timed as prefill.
    pub fn decode_tps(&self) -> f64 {
        if self.decode_seconds > 0.0 {
            self.generated_tokens.saturating_sub(1) as f64 / self.decode_seconds
        } else {
            0.0
        }
    }
}

pub use crate::eos::{
    IgnoreEosScope, ignore_eos_scope, is_eos, set_eos_ids, set_eos_ids_for_model,
};

/// Greedy (temperature 0) or temperature-scaled argmax generation.
///
/// Returns the generated ids (excluding the prompt).
#[allow(clippy::too_many_arguments)]
pub fn generate<R: SpeculativeRuntime>(
    tower: &mut dyn LanguageModel<R>,
    prompt: &[u32],
    max_tokens: usize,
    sampler: &mut R::Sampler,
    on_token: &mut dyn FnMut(u32) -> anyhow::Result<()>,
) -> anyhow::Result<(Vec<u32>, GenerationStats)> {
    if max_tokens == 0 {
        return Ok((
            Vec::new(),
            GenerationStats {
                prompt_tokens: prompt.len(),
                generated_tokens: 0,
                prefill_seconds: 0.0,
                decode_seconds: 0.0,
            },
        ));
    }
    // Compile/warm every kernel before the timed section (the engine warms at
    // init too; without this the first prefill pays ~0.45s of Metal JIT).
    anyhow::ensure!(
        prompt
            .len()
            .checked_add(max_tokens)
            .is_some_and(|end| end <= tower.max_position_embeddings()),
        "context overflow: prompt {} + max_tokens {} > max_position_embeddings {}",
        prompt.len(),
        max_tokens,
        tower.max_position_embeddings()
    );
    tower.warmup(prompt)?;
    let mut caches: Vec<R::Cache> = tower.new_caches();

    // Prefill in windows (bounded activations for long prompts).
    let _trace_prefill = tower.runtime().trace("prefill");
    let t0 = Instant::now();
    let last = tower.prefill_for_generation(prompt, &mut caches)?;
    let logits = tower.head(&last)?;
    let token = if tower.uses_host_sampling() {
        sampler.draw_host(tower.runtime(), &logits, prompt)?
    } else {
        sampler.draw(tower.runtime(), &logits, prompt)?
    };

    // Force eval of the whole prefill graph before timing decode.
    logits.eval().map_err(|e| anyhow::anyhow!("{e}"))?;
    let prefill_seconds = t0.elapsed().as_secs_f64();
    drop(_trace_prefill);

    let mut generated = Vec::new();
    generated.push(token);
    on_token(token)?;
    let decode_start = Instant::now();
    let mut decode_seconds = 0.0;
    let mut on_decode_token = |token| -> anyhow::Result<()> {
        on_token(token)?;
        decode_seconds = decode_start.elapsed().as_secs_f64();
        Ok(())
    };

    while generated.len() < max_tokens && !is_eos(*generated.last().unwrap()) {
        let _trace_step = tower.runtime().trace("decode.step");
        let token = *generated.last().unwrap();
        let step = tower.runtime().tokens(&[token as i32], &[1i32, 1])?;
        let (mixed, _) = tower.forward(&step, Some(&mut caches))?;
        let last = mixed.select(1, mixed.dim(1) - 1)?;
        let logits = tower.head(&last)?;
        let mut history: Vec<u32> = prompt.to_vec();
        history.extend_from_slice(&generated);
        let token = if tower.uses_host_sampling() {
            sampler.draw_host(tower.runtime(), &logits, &history)?
        } else {
            sampler.draw(tower.runtime(), &logits, &history)?
        };
        generated.push(token);
        on_decode_token(token)?;
        // Release pooled temporaries periodically: the pool only sweeps
        // buffer pool from `wait_until_completed`, so a long decode would pin
        // every intermediate it ever allocated.
        if generated.len() % 32 == 0 {
            tower.runtime().trim_memory();
        }
    }

    let stats = GenerationStats {
        prompt_tokens: prompt.len(),
        generated_tokens: generated.len(),
        prefill_seconds,
        decode_seconds,
    };
    Ok((generated, stats))
}

#[cfg(test)]
mod stats_tests {
    use super::GenerationStats;

    #[test]
    fn zero_budget_reports_no_throughput() {
        let stats = GenerationStats {
            prompt_tokens: 32,
            generated_tokens: 0,
            prefill_seconds: 0.0,
            decode_seconds: 0.0,
        };
        assert_eq!(stats.prefill_tps(), 0.0);
        assert_eq!(stats.decode_tps(), 0.0);
    }

    #[test]
    fn one_token_has_prefill_but_no_decode_throughput() {
        let stats = GenerationStats {
            prompt_tokens: 32,
            generated_tokens: 1,
            prefill_seconds: 0.5,
            decode_seconds: 0.0,
        };
        assert_eq!(stats.prefill_tps(), 64.0);
        assert_eq!(stats.decode_tps(), 0.0);
        assert_eq!(
            GenerationStats {
                decode_seconds: 0.25,
                ..stats
            }
            .decode_tps(),
            0.0
        );
    }

    #[test]
    fn many_tokens_count_only_the_tokens_timed_as_decode() {
        let stats = GenerationStats {
            prompt_tokens: 32,
            generated_tokens: 5,
            prefill_seconds: 0.5,
            decode_seconds: 0.25,
        };
        assert_eq!(stats.decode_tps(), 16.0);
        assert_eq!(
            GenerationStats {
                decode_seconds: 0.0,
                ..stats
            }
            .decode_tps(),
            0.0
        );
    }
}

#[cfg(test)]
mod eos_contract_tests {
    use super::*;

    /// Regression: set_eos_ids used OnceLock::set — the FIRST call stuck and
    /// any later call (e.g. `--ignore-eos` clearing im_end) was silently
    /// dropped, so the serve/run decode loop kept stopping at EOS at ~199/204
    /// tokens. The last call must win.
    #[test]
    fn eos_set_is_overwritable() {
        set_eos_ids(vec![1, 2, 3]);
        assert!(is_eos(2));
        set_eos_ids(Vec::new());
        assert!(!is_eos(2));
        set_eos_ids(vec![9]);
        assert!(is_eos(9) && !is_eos(1));
        set_eos_ids(Vec::new());
    }
}
