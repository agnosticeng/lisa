//! Token sampling: greedy, temperature, top-k, top-p (nucleus), min-p, with a
//! deterministic seeded RNG.
//!
//! The filtering runs on the GPU (argpartition + take_along_axis over the
//! vocabulary) and only the surviving candidates are read back, so the cost is
//! independent of the vocabulary size. Greedy (temperature <= 0) is the exact
//! argmax path the golden relies on and is bit-identical to the old `sample`.

use lisa_mlx::ops::indexing::{argmax_axis, IndexOp};
use lisa_mlx::{ops, Array};

/// Sampling configuration.
#[derive(Clone, Debug)]
pub struct Sampler {
    pub temperature: f32,
    /// 1 == greedy; 0 disables the cutoff.
    pub top_k: usize,
    /// 1.0 disables nucleus truncation.
    pub top_p: f32,
    /// 0.0 disables min-p.
    pub min_p: f32,
    /// `1.0` disables the penalty; applied over the last `penalty_window` tokens.
    pub repetition_penalty: f32,
    pub penalty_window: usize,
    pub seed: u64,
}

impl Default for Sampler {
    fn default() -> Self {
        Sampler {
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            min_p: 0.0,
            repetition_penalty: 1.0,
            penalty_window: 64,
            seed: 0x9E37_79B9_7F4A_7C15,
        }
    }
}

impl Sampler {
    pub fn greedy(&self) -> bool {
        self.temperature <= 1e-5 || self.top_k == 1
    }

    /// SplitMix64: small, deterministic, good enough for token sampling.
    fn next_u64(&mut self) -> u64 {
        self.seed = self.seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.seed;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Draw one token from `logits` (`[.., vocab]`, f32; the last axis is the
    /// vocabulary). `history` is the stream's committed tokens, used by the
    /// repetition penalty.
    pub fn draw(&mut self, logits: &Array, history: &[u32]) -> anyhow::Result<u32> {
        let flat = logits.reshape(&[-1])?;
        let mut logits = flat.as_dtype(lisa_mlx::Dtype::Float32)?;
        if self.greedy() {
            return Ok(argmax_axis(&logits, -1, false)?.item_cast::<i32>() as u32);
        }

        // Repetition penalty: positive logits are divided, negative multiplied.
        if self.repetition_penalty != 1.0 && !history.is_empty() {
            let start = history.len().saturating_sub(self.penalty_window);
            let penalty = self.repetition_penalty;
            let mut mask = vec![false; logits.dim(0) as usize];
            for &t in &history[start..] {
                if (t as i32) < logits.dim(0) {
                    mask[t as usize] = true;
                }
            }
            let mask = Array::from_slice(&mask, &[logits.dim(0)]);
            let pos = logits.divide(&ops::full::<f32>(logits.shape(), Array::from_f32(penalty))?)?;
            let neg = logits.multiply(&ops::full::<f32>(logits.shape(), Array::from_f32(penalty))?)?;
            logits = lisa_mlx::ops::r#where(&mask, &pos, &neg)?;
        }
        logits = logits.divide(&ops::full::<f32>(logits.shape(), Array::from_f32(self.temperature))?)?;

        // Top-k by argpartition on the negated logits, then read the candidates.
        let k = if self.top_k == 0 {
            64usize
        } else {
            self.top_k.min(logits.dim(0) as usize)
        };
        let neg = -&logits;
        let idx = ops::argpartition_axis(&neg, (k - 1) as i32, -1)?;
        let idx = idx.index(0..k as i32).contiguous()?;
        let vals = logits.take_along_axis(&idx, -1)?;
        let _ = (idx.eval(), vals.eval());
        let ids: Vec<u32> = idx.as_slice::<u32>().to_vec();
        let mut cand: Vec<(u32, f32)> = vals
            .as_slice::<f32>()
            .iter()
            .copied()
            .zip(ids.iter().copied())
            .map(|(v, t)| (t, v))
            .collect();
        drop(ids);

        // Sort descending so top-p / min-p see the tail in order.
        cand.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let max = cand.first().map(|c| c.1).unwrap_or(f32::NEG_INFINITY);

        // min-p: drop anything below `min_p * max`.
        if self.min_p > 0.0 {
            let floor = max + self.min_p.max(1e-9).ln();
            cand.retain(|c| c.1 >= floor);
        }
        // Softmax over the survivors.
        let mut probs: Vec<(u32, f32)> = {
            let m = cand.iter().map(|c| c.1).fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = cand.iter().map(|c| (c.1 - m).exp()).collect();
            let sum: f32 = exps.iter().sum();
            cand.iter()
                .zip(exps.iter())
                .map(|(c, e)| (c.0, e / sum.max(1e-30)))
                .collect()
        };
        // top-p: keep the smallest prefix whose cumulative probability >= top_p.
        if self.top_p < 1.0 && probs.len() > 1 {
            let mut cum = 0.0f32;
            let mut keep = 0usize;
            for (i, (_, p)) in probs.iter().enumerate() {
                cum += *p;
                keep = i + 1;
                if cum >= self.top_p {
                    break;
                }
            }
            probs.truncate(keep.max(1));
        }
        probs.retain(|(_, p)| *p > 0.0);
        if probs.is_empty() {
            return Ok(cand[0].0);
        }

        // Categorical draw.
        let total: f32 = probs.iter().map(|(_, p)| *p).sum();
        let mut r = (self.next_u64() >> 11) as f32 / (1u64 << 53) as f32 * total;
        for (t, p) in &probs {
            r -= *p;
            if r <= 0.0 {
                return Ok(*t);
            }
        }
        Ok(probs.last().unwrap().0)
    }
}
