//! Token sampling: greedy, temperature, top-k, top-p (nucleus), min-p, with a
//! deterministic seeded RNG.
//!
//! The filtering runs on the GPU (argpartition + take_along_axis over the
//! vocabulary) and only the surviving candidates are read back, so the cost is
//! independent of the vocabulary size. Greedy (temperature <= 0) is the exact
//! argmax path the golden relies on and is bit-identical to the old `sample`.

use lisa_mlx::ops::indexing::{IndexOp, argmax_axis};
use lisa_mlx::{Array, ops};

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

    /// Uniform f32 in `[0,1)` off the sampler's own RNG — the acceptance draw
    /// of stochastic speculative decoding.
    fn next_unit(&mut self) -> f32 {
        (self.next_u64() >> 11) as f32 / (1u64 << 53) as f32
    }

    /// Temperature-scaled softmax weights of one row (`exp(v/T - max)`), on the
    /// host. `None` when the sampler is greedy or the row is unusable — the
    /// caller then falls back to the greedy path.
    fn row_weights(&self, logits: &Array) -> Option<Vec<f32>> {
        if self.temperature <= 1e-5 {
            return None;
        }
        let v = logits
            .reshape(&[-1])
            .ok()?
            .as_dtype(lisa_mlx::Dtype::Float32)
            .ok()?
            .to_vec1::<f32>()
            .ok()?;
        let t = 1.0 / self.temperature;
        let m = v.iter().map(|&z| z * t).fold(f32::NEG_INFINITY, f32::max);
        if !m.is_finite() {
            return None;
        }
        Some(v.iter().map(|&z| ((z * t) - m).exp()).collect())
    }

    /// The acceptance probability of a DETERMINISTIC draft under the target's
    /// own row distribution: `min(1, p_target[draft])`.
    ///
    /// The prompt-lookup draft is deterministic (`q = delta`), so the exact
    /// speculative-sampling test collapses to the target's own probability of
    /// that token — this is the whole of lisa's stochastic PLD
    /// (`accept_prob = min(1, target_p[draft[i]])`).
    pub fn draft_accept_prob(&self, logits: &Array, draft: u32) -> Option<f32> {
        if self.temperature <= 1e-5 {
            return None;
        }
        let flat = logits
            .reshape(&[-1])
            .ok()?
            .as_dtype(lisa_mlx::Dtype::Float32)
            .ok()?;
        let inv = lisa_mlx::ops::full::<f32>(
            flat.shape(),
            Array::from_f32(1.0 / self.temperature),
        )
        .ok()?;
        let scaled = flat.multiply(&inv).ok()?;
        // GPU side: max + sum(exp(v/T - max)) — TWO scalars read back instead of
        // the whole vocabulary. Pulling 248k f32 to the host once per verify row
        // (up to 7 rows a round) measured 0.73 s against 0.53 s greedy on a
        // 24-token request; the residual/bonus draws legitimately need the full
        // vector but fire only once per round.
        let m = scaled.max(None).ok()?;
        let ex = scaled.subtract(&m).ok()?.exp().ok()?;
        let z = ex.sum(None).ok()?;
        let _ = (m.eval(), z.eval());
        let m_v: f32 = m.item::<f32>();
        let z_v: f32 = z.item::<f32>();
        if !(z_v > 0.0) || !z_v.is_finite() {
            return None;
        }
        // The draft's own scaled logit — ONE element, not the vector.
        let idx = Array::from_slice(&[draft as i32], &[1i32]);
        let x: f32 = scaled.take_along_axis(&idx, -1).ok()?.item::<f32>();
        Some(((x - m_v).exp() / z_v).min(1.0))
    }

    /// Draw from the residual `norm(max(p_target - one_hot(draft), 0))` — what
    /// the target wants where the draft is wrong. Never returns `draft` (its
    /// mass is zeroed), so it always differs from the rejected token.
    pub fn draw_residual(&mut self, logits: &Array, draft: u32) -> Option<u32> {
        let mut w = self.row_weights(logits)?;
        if (draft as usize) < w.len() {
            w[draft as usize] = 0.0;
        }
        self.categorical(&w)
    }

    /// Plain categorical draw from the row's own distribution (the bonus token
    /// a fully-accepted round earns).
    pub fn draw_row(&mut self, logits: &Array) -> Option<u32> {
        let w = self.row_weights(logits)?;
        self.categorical(&w)
    }

    fn categorical(&mut self, w: &[f32]) -> Option<u32> {
        let total: f32 = w.iter().sum();
        if total <= 0.0 || !total.is_finite() {
            return None;
        }
        let mut r = self.next_unit() * total;
        let mut last = None;
        for (i, p) in w.iter().enumerate() {
            if *p > 0.0 {
                last = Some(i as u32);
            }
            r -= *p;
            if r <= 0.0 {
                return Some(i as u32);
            }
        }
        last
    }

    /// The acceptance draw: `u < p` with `u` uniform on the sampler's RNG.
    pub fn accept_draw(&mut self, p: f32) -> bool {
        self.next_unit() < p
    }

    /// [`Self::draw`]'s keyed Gumbel draw at an EXPLICIT absolute position —
    /// the GPU-side equivalent of `draw(logits, history)` with
    /// `history.len() == position`. Used where the caller knows the position but
    /// does not hold a history slice (the MTP loop's serial steps), so it never
    /// pulls the vocabulary to the host.
    pub fn draw_at(&self, logits: &Array, position: u64) -> Option<u32> {
        if self.greedy() || self.repetition_penalty != 1.0 {
            return None;
        }
        let flat = logits
            .reshape(&[-1])
            .ok()?
            .as_dtype(lisa_mlx::Dtype::Float32)
            .ok()?;
        let stream = lisa_mlx::Stream::thread_local_or_default();
        let tok = lisa_mlx::models::qwen35::kernels::keyed_gumbel_sample(
            &flat,
            1.0 / self.temperature,
            self.seed,
            position,
            &stream,
        )?;
        Some(tok.item::<i32>() as u32)
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
        let _trace = lisa_mlx::trace::span("sampler");
        let flat = logits.reshape(&[-1])?;
        let mut logits = flat.as_dtype(lisa_mlx::Dtype::Float32)?;
        if self.greedy() {
            return Ok(argmax_axis(&logits, -1, false)?.item_cast::<i32>() as u32);
        }

        // Keyed Gumbel-max (specs/08 §5):
        // the whole temp>0 draw in ONE kernel — argmax_i(v/T + g(seed, p, i))
        // with g a splitmix64 hash of (seed, absolute position, token id).
        // Exact multinomial over softmax(v/T) on the full vocabulary, no
        // vocab-wide sort, and the draw is a pure function of (seed, position)
        // so verify rows compose. Top-k/top-p/min-p are not applied here (the
        // reference ships without min-p either); the repetition-penalty path
        // below keeps the composed GPU filter.
        if self.repetition_penalty == 1.0 {
            let stream = lisa_mlx::Stream::thread_local_or_default();
            if let Some(tok) = lisa_mlx::models::qwen35::kernels::keyed_gumbel_sample(
                &logits,
                1.0 / self.temperature,
                self.seed,
                history.len() as u64,
                &stream,
            ) {
                return Ok(tok.item::<i32>() as u32);
            }
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
            let pos =
                logits.divide(&ops::full::<f32>(logits.shape(), Array::from_f32(penalty))?)?;
            let neg =
                logits.multiply(&ops::full::<f32>(logits.shape(), Array::from_f32(penalty))?)?;
            logits = lisa_mlx::ops::r#where(&mask, &pos, &neg)?;
        }
        logits = logits.divide(&ops::full::<f32>(
            logits.shape(),
            Array::from_f32(self.temperature),
        )?)?;

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

/// Log-softmax probability of `token` under `logits` (`[.., vocab]`).
///
/// Lives in lisa-engine so serving layers can compute logprobs without
/// depending on the tensor runtime (`lisa-mlx`) directly.
pub fn token_logprob(logits: &Array, token: u32) -> anyhow::Result<f32> {
    let v = logits
        .as_dtype(lisa_mlx::Dtype::Float32)?
        .to_vec1::<f32>()?;
    let m = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let sum: f32 = v.iter().map(|x| (x - m).exp()).sum();
    let lse = m + sum.ln();
    let idx = token as usize;
    Ok(if idx < v.len() { v[idx] - lse } else { 0.0 })
}

#[cfg(test)]
mod keyed_tests {
    use super::*;
    use lisa_mlx::Dtype;

    fn fake_logits(v: usize, seed: u64) -> Array {
        let mut s = seed;
        let mut rnd = move || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((s >> 40) as f32 / 8_388_608.0) - 1.0
        };
        let v: Vec<f32> = (0..v).map(|_| rnd() * 10.0).collect();
        Array::from_slice(&v, &[v.len() as i32]).as_dtype(Dtype::Float32).unwrap()
    }

    #[test]
    // RESOLVED (specs/04 §6): failed ≥3× in CI-style parallel runs with a
    // MASSIVE divergence (3388 vs 783 at token 11). Root cause was the runtime
    // readback race: `flush_*` drained `in_flight` before waiting, so a
    // concurrent thread's `eval` returned without waiting committed GPU work
    // (plus a dropped completion-handler block in `ComputeEncoder::end` that
    // corrupted the cross-encoder fence map). Both fixed in commands.rs; the
    // parallel suite is green (extinction runs in specs/04 §6).
    fn keyed_same_seed_same_tokens() {
        let logits = fake_logits(4096, 1);
        let history: Vec<u32> = (0..37).collect();
        let mut a = Sampler { temperature: 0.7, ..Default::default() };
        let mut b = Sampler { temperature: 0.7, ..Default::default() };
        for pos in 0..24 {
            let hist: Vec<u32> = history[..pos].to_vec();
            let ta = a.draw(&logits, &hist).unwrap();
            let tb = b.draw(&logits, &hist).unwrap();
            assert_eq!(ta, tb, "position {pos}");
        }
    }

    #[test]
    fn keyed_seed_varies() {
        let logits = fake_logits(4096, 2);
        let mut a = Sampler { temperature: 0.7, seed: 0xDEAD, ..Default::default() };
        let mut b = Sampler { temperature: 0.7, seed: 0xBEEF, ..Default::default() };
        let diffs = (0..16)
            .filter(|pos| {
                let hist: Vec<u32> = (0..*pos as u32).collect();
                a.draw(&logits, &hist).unwrap() != b.draw(&logits, &hist).unwrap()
            })
            .count();
        assert!(diffs > 0, "different seeds drew identical streams");
    }

    #[test]
    fn greedy_unchanged() {
        let v: Vec<f32> = (0..64).map(|i| (i as f32 * 0.5).sin()).collect();
        let arr = Array::from_slice(&v, &[64]).as_dtype(Dtype::Float32).unwrap();
        let mut g = Sampler::default();
        let want = v.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0 as u32;
        assert_eq!(g.draw(&arr, &[]).unwrap(), want);
    }
}
