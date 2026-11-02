//! Token sampling: greedy, temperature, top-k, top-p (nucleus), min-p, with a
//! deterministic seeded RNG.
//!
//! Generic sampling uses GPU-side candidate selection where available. GLM
//! sampling follows DS4's host-side top-k/top-p/min-p and xorshift path.

use lisa_mlx::ops::indexing::{IndexOp, argmax_axis};
use lisa_mlx::{Array, ops};

struct FilteredDistribution {
    /// Filtered candidates in the order DS4 uses for categorical draws.
    ordered: Vec<(usize, f32)>,
    /// Sum accumulated in `ordered` order, matching DS4's sampling loop.
    total: f32,
    /// DS4's return value if rounding leaves the categorical scan unfinished.
    fallback: usize,
}

// DS4 sampling helpers follow ds4.c at b4e3faf5262b21db94ec309cf321640ff1eed460.
// A candidate is (token ID, logit, unnormalized probability). Comparisons use
// numeric equality, including +0 == -0, then the lower token ID as the tie-break.
type Ds4Candidate = (usize, f32, f32);

fn ds4_candidate_order(a: &Ds4Candidate, b: &Ds4Candidate) -> std::cmp::Ordering {
    b.1.partial_cmp(&a.1)
        .unwrap_or(std::cmp::Ordering::Equal)
        .then_with(|| a.0.cmp(&b.0))
}

fn ds4_candidate_gt(a: Ds4Candidate, b: Ds4Candidate) -> bool {
    if a.1 != b.1 { a.1 > b.1 } else { a.0 < b.0 }
}

fn ds4_heap_sift_up(heap: &mut [Ds4Candidate], mut index: usize) {
    while index > 0 {
        let parent = (index - 1) / 2;
        if !ds4_candidate_gt(heap[parent], heap[index]) {
            break;
        }
        heap.swap(parent, index);
        index = parent;
    }
}

fn ds4_heap_sift_down(heap: &mut [Ds4Candidate], mut index: usize) {
    loop {
        let left = index * 2 + 1;
        let right = left + 1;
        let mut smallest = index;
        if left < heap.len() && ds4_candidate_gt(heap[smallest], heap[left]) {
            smallest = left;
        }
        if right < heap.len() && ds4_candidate_gt(heap[smallest], heap[right]) {
            smallest = right;
        }
        if smallest == index {
            break;
        }
        heap.swap(index, smallest);
        index = smallest;
    }
}

fn ds4_argmax_host(logits: &[f32]) -> usize {
    let mut best = 0;
    let mut best_value = -1.0e30f32;
    for (index, &value) in logits.iter().enumerate() {
        if value > best_value {
            best = index;
            best_value = value;
        }
    }
    best
}

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
    /// Use exact point-mass p/q acceptance for target-conditioned MTP. The
    /// default follows ds4's opportunistic GLM path, which accepts only drafts
    /// that equal the target argmax.
    pub mtp_exact_sampling: bool,
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
            mtp_exact_sampling: false,
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

    /// Read one GLM row and apply LISA's optional history penalty before the
    /// reference sampling operations. DS4 itself has no repetition-penalty knob.
    fn ds4_logits(&self, logits: &Array, history: &[u32]) -> Option<Vec<f32>> {
        let flat = logits
            .reshape(&[-1])
            .ok()?
            .as_dtype(lisa_mlx::Dtype::Float32)
            .ok()?;
        // Wait only through the command buffer that produced this logits
        // array. A global flush also serializes unrelated later work on the
        // shared runtime queue, which is avoidable at a sampling boundary.
        let mut values = flat.to_vec1_wait_through::<f32>().ok()?;
        if values.is_empty() {
            return None;
        }
        let vocab_len = values.len();
        let start = history.len().saturating_sub(self.penalty_window);
        let mut seen = vec![false; vocab_len];
        for &token in &history[start..] {
            if let Some(is_seen) = seen.get_mut(token as usize) {
                *is_seen = true;
            }
        }
        for (value, is_seen) in values.iter_mut().zip(seen) {
            if self.repetition_penalty != 1.0 && is_seen {
                *value = if *value > 0.0 {
                    *value / self.repetition_penalty
                } else {
                    *value * self.repetition_penalty
                };
            }
        }

        Some(values)
    }

    /// Ordinary DS4 categorical sampling. Exact MTP has a separate builder:
    /// its heap denominator and raw min-p comparisons are different operations.
    fn sampling_distribution(
        &self,
        logits: &Array,
        history: &[u32],
    ) -> Option<FilteredDistribution> {
        let _trace = lisa_mlx::trace::span("glm53.sampler.filter");
        if self.temperature <= 0.0 {
            return None;
        }
        let values = self.ds4_logits(logits, history)?;
        let full_vocab_fallback = ds4_argmax_host(&values);

        let max = values
            .iter()
            .copied()
            .filter(|value| value.is_finite())
            .fold(
                if self.top_k > 0 {
                    f32::NEG_INFINITY
                } else {
                    -1.0e30f32
                },
                f32::max,
            );
        if !max.is_finite() {
            return None;
        }
        let mut candidates: Vec<(usize, f32, f32)> = values
            .into_iter()
            .enumerate()
            .filter(|(_, value)| value.is_finite())
            .map(|(token, value)| {
                let weight = ((value - max) / self.temperature).exp();
                (token, value, weight)
            })
            .collect();
        if candidates.is_empty() {
            return None;
        }
        // DS4 accumulates the full-vocabulary denominator in token order
        // before applying top-p. Preserve that order when top-k is disabled.
        let full_sum: f32 = candidates.iter().map(|(_, _, weight)| *weight).sum();
        let min_p = self.min_p.max(0.0);
        let min_weight = candidates
            .iter()
            .map(|(_, _, weight)| *weight)
            .fold(0.0f32, f32::max)
            * min_p;
        let top_p = if self.top_p <= 0.0 || self.top_p > 1.0 {
            1.0
        } else {
            self.top_p
        };

        let select_prefix = |sorted: &[(usize, f32, f32)], base_sum: f32| {
            let mut selected = Vec::with_capacity(sorted.len());
            let mut selected_sum = 0.0f32;
            let mut stopped_by_min_p = false;
            let mut reached_top_p = false;
            let min_probability = sorted
                .first()
                .map(|candidate| (candidate.2 / base_sum) * min_p)
                .unwrap_or(0.0);
            for (index, candidate) in sorted.iter().enumerate() {
                if index > 0 && candidate.2 / base_sum < min_probability {
                    stopped_by_min_p = true;
                    break;
                }
                selected_sum += candidate.2;
                selected.push((candidate.0, candidate.2));
                if selected_sum / base_sum >= top_p {
                    reached_top_p = true;
                    break;
                }
            }
            (selected, selected_sum, stopped_by_min_p, reached_top_p)
        };

        let (selected, selected_sum) = if self.top_k > 0 {
            let k = self.top_k.min(1024).min(candidates.len());
            if k < candidates.len() {
                candidates.select_nth_unstable_by(k, ds4_candidate_order);
                candidates.truncate(k);
            }
            candidates.sort_by(ds4_candidate_order);
            let base_sum: f32 = candidates.iter().map(|(_, _, weight)| *weight).sum();
            let (selected, selected_sum, _, _) = select_prefix(&candidates, base_sum);
            (selected, selected_sum)
        } else if top_p >= 1.0 {
            let mut selected = Vec::with_capacity(candidates.len());
            let mut selected_sum = 0.0f32;
            for (token, _, weight) in candidates {
                if weight >= min_weight {
                    selected.push((token, weight));
                    selected_sum += weight;
                }
            }
            (selected, selected_sum)
        } else {
            const FAST_TOP_P_CAP: usize = 512;
            let cap = candidates.len().min(FAST_TOP_P_CAP);
            if cap < candidates.len() {
                candidates.select_nth_unstable_by(cap, ds4_candidate_order);
            }
            candidates[..cap].sort_by(ds4_candidate_order);
            let (fast_selected, fast_sum, stopped_by_min_p, reached_top_p) =
                select_prefix(&candidates[..cap], full_sum);
            let tail_proven_filtered =
                stopped_by_min_p && cap < candidates.len() && candidates[cap - 1].2 < min_weight;
            if reached_top_p || tail_proven_filtered || cap == candidates.len() {
                (fast_selected, fast_sum)
            } else {
                candidates.sort_by(ds4_candidate_order);
                let (selected, selected_sum, _, _) = select_prefix(&candidates, full_sum);
                (selected, selected_sum)
            }
        };
        if !(selected_sum > 0.0) || !selected_sum.is_finite() {
            return None;
        }

        Some(FilteredDistribution {
            fallback: if self.top_k == 0 && top_p >= 1.0 {
                full_vocab_fallback
            } else {
                selected.last()?.0
            },
            ordered: selected,
            total: selected_sum,
        })
    }

    /// Port of DS4 sample_build_probabilities. Keep the token-order heap-sum
    /// updates, raw min-p threshold and sorted filtered sum distinct from the
    /// ordinary sampler; changing their order can alter exact-MTP acceptance.
    fn speculation_weights(&self, logits: &Array, history: &[u32]) -> Option<Vec<f32>> {
        let values = self.ds4_logits(logits, history)?;
        let mut probabilities = vec![0.0; values.len()];
        let point_mass = |mut probabilities: Vec<f32>| {
            probabilities[ds4_argmax_host(&values)] = 1.0;
            probabilities
        };
        if self.temperature <= 0.0 {
            return Some(point_mass(probabilities));
        }
        let top_p = if self.top_p <= 0.0 || self.top_p > 1.0 {
            1.0
        } else {
            self.top_p
        };
        let min_p = if self.min_p < 0.0 { 0.0 } else { self.min_p };
        let top_k = self.top_k.min(1024).min(values.len());
        let heap_cap = if top_k > 0 { top_k } else { 512 };
        let mut finite = 0;
        let mut max_logit = -1.0e30f32;
        for &value in &values {
            if value.is_finite() {
                finite += 1;
                if value > max_logit {
                    max_logit = value;
                }
            }
        }
        if finite == 0 {
            return Some(point_mass(probabilities));
        }

        let mut heap: Vec<Ds4Candidate> = Vec::with_capacity(heap_cap);
        let mut full_sum = 0.0f32;
        let mut heap_sum = 0.0f32;
        for (token, &value) in values.iter().enumerate() {
            if !value.is_finite() {
                continue;
            }
            let weight = ((value - max_logit) / self.temperature).exp();
            full_sum += weight;
            let candidate = (token, value, weight);
            if heap.len() < heap_cap {
                heap.push(candidate);
                heap_sum += weight;
                let index = heap.len() - 1;
                ds4_heap_sift_up(&mut heap, index);
            } else if ds4_candidate_gt(candidate, heap[0]) {
                heap_sum -= heap[0].2;
                heap[0] = candidate;
                heap_sum += weight;
                ds4_heap_sift_down(&mut heap, 0);
            }
        }
        if full_sum <= 0.0 || !full_sum.is_finite() || heap.is_empty() {
            return Some(point_mass(probabilities));
        }
        heap.sort_by(ds4_candidate_order);
        let filter = |candidates: &[Ds4Candidate], base_sum: f32| {
            let min_raw = candidates[0].2 * min_p;
            let mut filtered_sum = 0.0f32;
            let mut filtered = 0;
            let mut stopped_by_min_p = false;
            let mut reached_top_p = false;
            for (index, candidate) in candidates.iter().enumerate() {
                if index > 0 && candidate.2 < min_raw {
                    stopped_by_min_p = true;
                    break;
                }
                filtered_sum += candidate.2;
                filtered += 1;
                if filtered_sum / base_sum >= top_p {
                    reached_top_p = true;
                    break;
                }
            }
            (filtered, filtered_sum, stopped_by_min_p, reached_top_p)
        };
        let base_sum = if top_k > 0 { heap_sum } else { full_sum };
        let (mut filtered, mut filtered_sum, stopped, reached) = filter(&heap, base_sum);
        let min_raw = heap[0].2 * min_p;
        let fast_complete = top_k > 0
            || heap.len() == finite
            || reached
            || (stopped && heap[heap.len() - 1].2 < min_raw);
        if !fast_complete {
            heap = values
                .iter()
                .copied()
                .enumerate()
                .filter(|(_, value)| value.is_finite())
                .map(|(token, value)| {
                    (token, value, ((value - max_logit) / self.temperature).exp())
                })
                .collect();
            heap.sort_by(ds4_candidate_order);
            (filtered, filtered_sum, _, _) = filter(&heap, full_sum);
        }
        if filtered == 0 || filtered_sum <= 0.0 || !filtered_sum.is_finite() {
            probabilities[heap[0].0] = 1.0;
        } else {
            let inv_sum = 1.0 / filtered_sum;
            for &(token, _, weight) in &heap[..filtered] {
                probabilities[token] = weight * inv_sum;
            }
        }
        Some(probabilities)
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
        let inv = lisa_mlx::ops::full::<f32>(flat.shape(), Array::from_f32(1.0 / self.temperature))
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

    /// History-aware GLM MTP acceptance probability. The history contains the
    /// committed tokens before the row whose logits are being verified.
    pub fn draft_accept_prob_with_history(
        &self,
        logits: &Array,
        draft: u32,
        history: &[u32],
    ) -> Option<f32> {
        let weights = self.speculation_weights(logits, history)?;
        let probability = *weights.get(draft as usize)?;
        (probability >= 0.0 && probability.is_finite()).then_some(probability.min(1.0))
    }

    /// Verify GLM's point-mass proposal using one DS4 probability construction.
    /// A rejection removes the proposal's mass from that same distribution
    /// before drawing its replacement, preserving DS4's RNG consumption.
    pub fn verify_draft_with_history(
        &mut self,
        logits: &Array,
        draft: u32,
        history: &[u32],
    ) -> Option<(bool, u32)> {
        let mut weights = self.speculation_weights(logits, history)?;
        let probability = *weights.get(draft as usize)?;
        if probability < 0.0 || !probability.is_finite() {
            return None;
        }
        if self.accept_draw_ds4(probability) {
            return Some((true, draft));
        }
        weights[draft as usize] = 0.0;
        self.categorical_ds4(&weights)
            .map(|replacement| (false, replacement))
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

    /// History-aware residual draw for GLM's point-mass MTP proposal.
    pub fn draw_residual_with_history(
        &mut self,
        logits: &Array,
        draft: u32,
        history: &[u32],
    ) -> Option<u32> {
        let mut weights = self.speculation_weights(logits, history)?;
        *weights.get_mut(draft as usize)? = 0.0;
        self.categorical_ds4(&weights)
    }

    /// Plain categorical draw from the row's own distribution (the bonus token
    /// a fully-accepted round earns).
    pub fn draw_row(&mut self, logits: &Array) -> Option<u32> {
        let w = self.row_weights(logits)?;
        self.categorical(&w)
    }

    /// Draw a verified bonus row under the same history-dependent target
    /// distribution used by ordinary sampling.
    pub fn draw_row_with_history(&mut self, logits: &Array, history: &[u32]) -> Option<u32> {
        let weights = self.speculation_weights(logits, history)?;
        self.categorical_ds4(&weights)
    }

    /// GLM sampling path matching DS4's top-k/top-p/min-p filters and
    /// xorshift64* RNG. This intentionally reads logits to the host; the
    /// keyed-Gumbel path remains the fast full-softmax sampler for other paths.
    pub fn draw_ds4(&mut self, logits: &Array, history: &[u32]) -> anyhow::Result<u32> {
        let _trace = lisa_mlx::trace::span("glm53.sampler.draw");
        if self.temperature <= 0.0 {
            if self.repetition_penalty == 1.0 || history.is_empty() {
                return self.draw(logits, history);
            }
            let values = self
                .ds4_logits(logits, history)
                .ok_or_else(|| anyhow::anyhow!("cannot read GLM logits for sampling"))?;
            return Ok(ds4_argmax_host(&values) as u32);
        }
        if let Some(distribution) = self.sampling_distribution(logits, history)
            && let Some(token) = self.categorical_ds4_ordered(
                &distribution.ordered,
                distribution.total,
                distribution.fallback,
            )
        {
            return Ok(token);
        }
        let values = self
            .ds4_logits(logits, history)
            .ok_or_else(|| anyhow::anyhow!("cannot read GLM logits for sampling"))?;
        Ok(ds4_argmax_host(&values) as u32)
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

    /// DS4 sample_probabilities: sum and scan in token order, skip invalid or
    /// zero mass before testing the draw, and fall back to the highest mass.
    fn categorical_ds4(&mut self, w: &[f32]) -> Option<u32> {
        let mut best_probability = *w.first()?;
        let mut best = 0;
        let mut total = 0.0f32;
        for (index, &probability) in w.iter().enumerate() {
            if probability > 0.0 && probability.is_finite() {
                total += probability;
            }
            if probability > best_probability {
                best_probability = probability;
                best = index;
            }
        }
        if total <= 0.0 || !total.is_finite() {
            return Some(best as u32);
        }
        let mut r = self.next_ds4_unit() * total;
        for (index, &probability) in w.iter().enumerate() {
            if probability <= 0.0 || !probability.is_finite() {
                continue;
            }
            r -= probability;
            if r <= 0.0 {
                return Some(index as u32);
            }
        }
        Some(best as u32)
    }

    fn categorical_ds4_ordered(
        &mut self,
        candidates: &[(usize, f32)],
        total: f32,
        fallback: usize,
    ) -> Option<u32> {
        if total <= 0.0 || !total.is_finite() {
            return None;
        }
        let mut r = self.next_ds4_unit() * total;
        for &(token, weight) in candidates {
            r -= weight;
            if r <= 0.0 {
                return Some(token as u32);
            }
        }
        Some(fallback as u32)
    }

    /// The acceptance draw: `u < p` with `u` uniform on the sampler's RNG.
    pub fn accept_draw(&mut self, p: f32) -> bool {
        self.next_unit() < p
    }

    /// DS4-compatible acceptance draw for GLM exact target-MTP.
    pub fn accept_draw_ds4(&mut self, p: f32) -> bool {
        if !(p > 0.0 && p.is_finite()) {
            return false;
        }
        if p >= 1.0 {
            return true;
        }
        self.next_ds4_unit() <= p
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

    fn next_ds4_unit(&mut self) -> f32 {
        let mut state = self.seed;
        if state == 0 {
            state = 0x9E37_79B9_7F4A_7C15;
        }
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        self.seed = state;
        let value = state.wrapping_mul(0x2545_F491_4F6C_DD1D);
        ((value >> 40) as u32 & 0x00FF_FFFF) as f32 / 16_777_216.0
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
        Array::from_slice(&v, &[v.len() as i32])
            .as_dtype(Dtype::Float32)
            .unwrap()
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
        let mut a = Sampler {
            temperature: 0.7,
            ..Default::default()
        };
        let mut b = Sampler {
            temperature: 0.7,
            ..Default::default()
        };
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
        let mut a = Sampler {
            temperature: 0.7,
            seed: 0xDEAD,
            ..Default::default()
        };
        let mut b = Sampler {
            temperature: 0.7,
            seed: 0xBEEF,
            ..Default::default()
        };
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
        let arr = Array::from_slice(&v, &[64])
            .as_dtype(Dtype::Float32)
            .unwrap();
        let mut g = Sampler::default();
        let want = v
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap()
            .0 as u32;
        assert_eq!(g.draw(&arr, &[]).unwrap(), want);
    }

    #[test]
    fn exact_speculation_distribution_matches_top_k_top_p_and_min_p() {
        let logits = Array::from_slice(&[3.0f32, 2.0, 0.0], &[3]);
        let mut sampler = Sampler {
            temperature: 1.0,
            top_p: 0.8,
            ..Default::default()
        };

        let weights = sampler.speculation_weights(&logits, &[]).unwrap();
        assert!(weights[0] > weights[1]);
        let top_p_sum = weights[0] + weights[1];
        assert!((weights[0] / top_p_sum - 1.0 / (1.0 + (-1.0f32).exp())).abs() < 1e-6);
        assert_eq!(weights[2], 0.0, "top-p must use full-vocabulary mass");

        sampler.top_k = 1;
        sampler.top_p = 1.0;
        let weights = sampler.speculation_weights(&logits, &[]).unwrap();
        assert_eq!(weights, vec![1.0, 0.0, 0.0]);
        assert_eq!(sampler.draw_ds4(&logits, &[]).unwrap(), 0);

        sampler.top_k = 0;
        sampler.min_p = 0.2;
        let weights = sampler.speculation_weights(&logits, &[]).unwrap();
        assert!(weights[0] > 0.0 && weights[1] > 0.0);
        assert_eq!(weights[2], 0.0, "min-p must cut the low-probability tail");
    }

    #[test]
    fn ds4_rng_matches_xorshift64star_reference() {
        let mut sampler = Sampler {
            seed: 0x5eed,
            ..Default::default()
        };
        let expected = [0.59004927f32, 0.7808038, 0.91030985];
        for value in expected {
            assert_eq!(sampler.next_ds4_unit(), value);
        }
    }

    #[test]
    fn ds4_accept_draw_preserves_point_mass_rng_semantics() {
        let mut sampler = Sampler {
            seed: 0x5eed,
            ..Default::default()
        };
        let initial = sampler.seed;
        assert!(!sampler.accept_draw_ds4(0.0));
        assert_eq!(sampler.seed, initial, "zero target mass consumes no draw");
        assert!(sampler.accept_draw_ds4(1.0));
        assert_eq!(sampler.seed, initial, "certain acceptance consumes no draw");

        let mut oracle = sampler.clone();
        let boundary = oracle.next_ds4_unit();
        assert!(sampler.accept_draw_ds4(boundary));
        assert_eq!(sampler.seed, oracle.seed);
    }

    #[test]
    fn glm_ds4_sampling_order_matches_c_reference() {
        // Oracle tokens and post-draw RNG states from ds4.c at
        // ee7e8405232c8971000cc7ba2b945b00bf6d20d3, using
        // ds4_test_sample_logits with top_k=0, top_p=0.95, min_p=0.
        let logits = Array::from_slice(&[3.0f32, 2.0, 4.0, 1.0, 0.0, -1.0], &[6]);
        let expected = [
            (2, 285_734_347_287_933_762),
            (2, 33_554_433),
            (2, 67_108_866),
            (0, 100_663_299),
            (2, 134_217_733),
            (2, 167_772_164),
            (0, 201_326_599),
            (0, 234_881_030),
            (2, 268_435_466),
            (0, 301_989_899),
            (2, 335_544_328),
            (0, 369_098_761),
            (0, 402_653_199),
            (0, 436_207_630),
            (0, 469_762_061),
            (0, 503_316_492),
        ];
        for (seed, (expected_token, expected_state)) in expected.into_iter().enumerate() {
            let mut sampler = Sampler {
                temperature: 1.0,
                top_p: 0.95,
                seed: seed as u64,
                ..Default::default()
            };
            assert_eq!(sampler.draw_ds4(&logits, &[]).unwrap(), expected_token);
            assert_eq!(sampler.seed, expected_state);
        }

        // The 2048-entry fixtures exercise DS4's 512-candidate fast top-p
        // path and its full-sort fallback. Expected values come from the same
        // C hook and reference revision above.
        let mut fast_logits: Vec<f32> = (0..2048)
            .map(|i| -0.02 * i as f32 + ((i * 17) % 11) as f32 * 0.001)
            .collect();
        let mut fallback_logits: Vec<f32> =
            (0..2048).map(|i| ((i * 37) % 101) as f32 * 0.001).collect();
        let fast_tokens = [
            2, 15, 38, 80, 14, 14, 77, 75, 35, 75, 33, 71, 72, 70, 68, 66,
        ];
        let fallback_tokens = [
            1665, 1992, 1388, 107, 1760, 1831, 1764, 1835, 722, 1532, 763, 1371, 1199, 1068, 937,
            1008,
        ];
        for (logits, expected_tokens) in [
            (&mut fast_logits, &fast_tokens),
            (&mut fallback_logits, &fallback_tokens),
        ] {
            let logits = Array::from_slice(logits, &[2048]);
            for (seed, expected_token) in expected_tokens.iter().copied().enumerate() {
                let mut sampler = Sampler {
                    temperature: 1.0,
                    top_p: 0.95,
                    seed: seed as u64,
                    ..Default::default()
                };
                assert_eq!(sampler.draw_ds4(&logits, &[]).unwrap(), expected_token);
                assert_eq!(sampler.seed, expected[seed].1);
            }
        }
    }

    #[test]
    fn glm_ds4_exact_delta_sampling_matches_c_reference() {
        // Oracle from ds4_test_speculative_delta_sample with draft token 2,
        // top_k=0, top_p=0.95, min_p=0, at the DS4 revision above.
        let logits = Array::from_slice(&[3.0f32, 2.0, 4.0, 1.0, 0.0, -1.0], &[6]);
        let expected = [
            (2, 285_734_347_287_933_762),
            (2, 33_554_433),
            (2, 67_108_866),
            (0, 3_378_524_379_445_251),
            (2, 134_217_733),
            (2, 167_772_164),
            (0, 6_757_048_725_336_071),
            (1, 7_883_223_518_468_102),
            (2, 268_435_466),
            (0, 10_135_573_071_226_891),
            (2, 335_544_328),
            (1, 12_387_922_657_490_953),
            (0, 13_514_097_417_117_711),
            (1, 14_640_272_210_249_742),
            (1, 15_766_447_003_381_773),
            (1, 16_892_621_796_513_804),
        ];
        for (seed, (expected_token, expected_state)) in expected.into_iter().enumerate() {
            let mut sampler = Sampler {
                temperature: 1.0,
                top_p: 0.95,
                seed: seed as u64,
                ..Default::default()
            };
            let p = sampler
                .draft_accept_prob_with_history(&logits, 2, &[])
                .unwrap();
            assert!((p - 0.665_240_943).abs() <= 1e-7, "p={p}");
            let token = if sampler.accept_draw_ds4(p) {
                2
            } else {
                sampler.draw_residual_with_history(&logits, 2, &[]).unwrap()
            };
            assert_eq!(token, expected_token);
            assert_eq!(sampler.seed, expected_state);
        }
    }

    #[test]
    fn glm_ds4_filter_edges_match_c_reference() {
        // Direct fixtures from ds4_test_sample_logits and
        // ds4_test_speculative_delta_sample at
        // ee7e8405232c8971000cc7ba2b945b00bf6d20d3. These cover top-k plus
        // nucleus, top-p=1 with min-p, top-k=1, the 512-candidate fast path,
        // and the full-sort fallback for both ordinary and exact draws.
        let draw_states = [
            285_734_347_287_933_762,
            33_554_433,
            67_108_866,
            100_663_299,
            134_217_733,
            167_772_164,
            201_326_599,
            234_881_030,
            268_435_466,
            301_989_899,
            335_544_328,
            369_098_761,
            402_653_199,
            436_207_630,
            469_762_061,
            503_316_492,
        ];
        let cases: [(
            &[f32],
            usize,
            f32,
            usize,
            f32,
            f32,
            u32,
            f32,
            &[u32],
            &[u32],
            &[u64],
        ); 2] = [
            (
                &[3.0, 2.0, 4.0, 1.0, 0.0, -1.0, -2.0, f32::NAN],
                8,
                0.7,
                3,
                0.8,
                0.05,
                2,
                0.806_678_653,
                &[2, 2, 2, 0, 2, 2, 0, 0, 2, 0, 2, 2, 0, 2, 2, 2],
                &[2, 2, 2, 0, 2, 2, 0, 0, 2, 0, 2, 2, 0, 2, 2, 2],
                &[
                    285_734_347_287_933_762,
                    33_554_433,
                    67_108_866,
                    3_378_524_379_445_251,
                    134_217_733,
                    167_772_164,
                    6_757_048_725_336_071,
                    7_883_223_518_468_102,
                    268_435_466,
                    10_135_573_071_226_891,
                    335_544_328,
                    369_098_761,
                    13_514_097_417_117_711,
                    436_207_630,
                    469_762_061,
                    503_316_492,
                ],
            ),
            (
                &[3.0, 2.0, 4.0, 1.0, 0.0, -1.0, -2.0, -3.0],
                8,
                1.0,
                0,
                1.0,
                0.01,
                2,
                0.636_408_627,
                &[0, 1, 2, 2, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
                &[2, 2, 2, 0, 2, 2, 0, 3, 2, 0, 2, 1, 0, 4, 1, 1],
                &[
                    285_734_347_287_933_762,
                    33_554_433,
                    67_108_866,
                    3_378_524_379_445_251,
                    134_217_733,
                    167_772_164,
                    6_757_048_725_336_071,
                    7_883_223_518_468_102,
                    268_435_466,
                    10_135_573_071_226_891,
                    335_544_328,
                    12_387_922_657_490_953,
                    13_514_097_417_117_711,
                    14_640_272_210_249_742,
                    15_766_447_003_381_773,
                    16_892_621_796_513_804,
                ],
            ),
        ];

        for (
            case_index,
            (
                values,
                len,
                temperature,
                top_k,
                top_p,
                min_p,
                draft,
                expected_p,
                ordinary_tokens,
                exact_tokens,
                exact_states,
            ),
        ) in cases.iter().copied().enumerate()
        {
            let logits = Array::from_slice(values, &[len as i32]);
            for seed in 0..16 {
                let mut ordinary = Sampler {
                    temperature,
                    top_k,
                    top_p,
                    min_p,
                    seed,
                    ..Default::default()
                };
                assert_eq!(
                    ordinary.draw_ds4(&logits, &[]).unwrap(),
                    ordinary_tokens[seed as usize],
                    "ordinary case={case_index} seed={seed}"
                );
                assert_eq!(ordinary.seed, draw_states[seed as usize]);
            }

            for seed in 0..16 {
                let mut exact = Sampler {
                    temperature,
                    top_k,
                    top_p,
                    min_p,
                    seed,
                    ..Default::default()
                };
                let p = exact
                    .draft_accept_prob_with_history(&logits, draft, &[])
                    .unwrap();
                assert!(
                    (p - expected_p).abs() <= 1e-6,
                    "target probability case={case_index} seed={seed}: {p}"
                );
                let token = if exact.accept_draw_ds4(p) {
                    draft
                } else {
                    exact
                        .draw_residual_with_history(&logits, draft, &[])
                        .unwrap()
                };
                assert_eq!(
                    token, exact_tokens[seed as usize],
                    "exact case={case_index} seed={seed}"
                );
                assert_eq!(
                    exact.seed, exact_states[seed as usize],
                    "exact RNG case={case_index} seed={seed}"
                );
            }
        }

        let logits = Array::from_slice(&[3.0f32, 2.0, 4.0, 1.0, 0.0, -1.0, -2.0, -3.0], &[8]);
        for seed in 0..16 {
            let mut sampler = Sampler {
                temperature: 0.8,
                top_k: 1,
                top_p: 0.95,
                seed,
                ..Default::default()
            };
            assert_eq!(sampler.draw_ds4(&logits, &[]).unwrap(), 2);
            assert_eq!(sampler.seed, draw_states[seed as usize]);
        }

        let fast_logits: Vec<f32> = (0..2048)
            .map(|i| -0.02 * i as f32 + ((i * 17) % 11) as f32 * 0.001)
            .collect();
        let fast_ordinary = [
            1, 11, 26, 50, 10, 10, 49, 48, 25, 48, 24, 46, 46, 45, 44, 43,
        ];
        let fast_exact = [14, 35, 15, 1, 37, 29, 2, 61, 17, 3, 7, 51, 4, 68, 53, 42];
        let fast_exact_states = [
            6_354_739_634_455_712_083,
            1_126_174_793_148_417,
            2_252_349_586_296_834,
            3_378_524_379_445_251,
            4_504_699_139_039_237,
            5_630_873_932_171_268,
            6_757_048_725_336_071,
            7_883_223_518_468_102,
            9_009_398_278_078_474,
            10_135_573_071_226_891,
            11_261_747_864_342_536,
            12_387_922_657_490_953,
            13_514_097_417_117_711,
            14_640_272_210_249_742,
            15_766_447_003_381_773,
            16_892_621_796_513_804,
        ];
        let fast = Array::from_slice(&fast_logits, &[2048]);
        for seed in 0..16 {
            let mut ordinary = Sampler {
                temperature: 0.9,
                top_p: 0.95,
                min_p: 0.2,
                seed,
                ..Default::default()
            };
            assert_eq!(
                ordinary.draw_ds4(&fast, &[]).unwrap(),
                fast_ordinary[seed as usize]
            );
            assert_eq!(ordinary.seed, draw_states[seed as usize]);

            let mut exact = Sampler {
                temperature: 0.9,
                top_p: 0.95,
                min_p: 0.2,
                seed,
                ..Default::default()
            };
            let p = exact.draft_accept_prob_with_history(&fast, 0, &[]).unwrap();
            assert!((p - 0.027_237_312_9).abs() <= 1e-6, "p={p}");
            let token = if exact.accept_draw_ds4(p) {
                0
            } else {
                exact.draw_residual_with_history(&fast, 0, &[]).unwrap()
            };
            assert_eq!(token, fast_exact[seed as usize], "fast exact seed={seed}");
            assert_eq!(exact.seed, fast_exact_states[seed as usize]);
        }

        let fallback_logits: Vec<f32> =
            (0..2048).map(|i| ((i * 37) % 101) as f32 * 0.001).collect();
        let fallback_ordinary = [
            1261, 1558, 217, 1876, 1528, 1700, 1543, 1715, 1945, 1412, 198, 1453, 1180, 1150, 1322,
            1393,
        ];
        let fallback_exact = [
            675, 1370, 698, 27, 1420, 1195, 76, 1899, 795, 125, 351, 1725, 174, 1997, 1773, 1550,
        ];
        let fallback_exact_states = [
            6_354_739_634_455_712_083,
            1_126_174_793_148_417,
            2_252_349_586_296_834,
            3_378_524_379_445_251,
            4_504_699_139_039_237,
            5_630_873_932_171_268,
            6_757_048_725_336_071,
            7_883_223_518_468_102,
            9_009_398_278_078_474,
            10_135_573_071_226_891,
            11_261_747_864_342_536,
            12_387_922_657_490_953,
            13_514_097_417_117_711,
            14_640_272_210_249_742,
            15_766_447_003_381_773,
            16_892_621_796_513_804,
        ];
        let fallback = Array::from_slice(&fallback_logits, &[2048]);
        for seed in 0..16 {
            let mut ordinary = Sampler {
                temperature: 1.3,
                top_p: 0.9,
                seed,
                ..Default::default()
            };
            assert_eq!(
                ordinary.draw_ds4(&fallback, &[]).unwrap(),
                fallback_ordinary[seed as usize]
            );
            assert_eq!(ordinary.seed, draw_states[seed as usize]);

            let mut exact = Sampler {
                temperature: 1.3,
                top_p: 0.9,
                seed,
                ..Default::default()
            };
            let p = exact
                .draft_accept_prob_with_history(&fallback, 100, &[])
                .unwrap();
            assert!((p - 0.000_548_244_687).abs() <= 1e-6, "p={p}");
            let token = if exact.accept_draw_ds4(p) {
                100
            } else {
                exact
                    .draw_residual_with_history(&fallback, 100, &[])
                    .unwrap()
            };
            assert_eq!(
                token, fallback_exact[seed as usize],
                "fallback exact seed={seed}"
            );
            assert_eq!(exact.seed, fallback_exact_states[seed as usize]);
        }
    }
}
