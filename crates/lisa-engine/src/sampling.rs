//! Token sampling: greedy, temperature, top-k, top-p (nucleus), min-p, with a
//! deterministic seeded RNG.
//!
//! Generic sampling uses GPU-side candidate selection where available. GLM
//! sampling follows DS4's host-side top-k/top-p/min-p and xorshift path.

use crate::tensor_ops::{TensorLaw, TensorOps};
use lisa_backend::{
    Binary, HostAccessDevice, Kernel, ReadbackDevice, ScalarType, TensorOperation as Op, Unary,
};

/// Optional diagnostic observer supplied by the application, outside device policy.
pub type TraceHook = fn(&'static str) -> Option<Box<dyn std::any::Any>>;
pub struct SamplingContext<'a, D: HostAccessDevice + ReadbackDevice> {
    pub ops: TensorOps<'a, D>,
    trace: TraceHook,
    observer: Option<crate::execution::Observer>,
}
impl<'a, D: HostAccessDevice + ReadbackDevice> SamplingContext<'a, D> {
    pub fn new(device: &'a D, law: TensorLaw, trace: TraceHook) -> Self {
        Self {
            ops: TensorOps::new(device, law),
            trace,
            observer: None,
        }
    }
    fn span(&self, name: &'static str) -> Option<Box<dyn std::any::Any>> {
        match self.observer {
            Some(observer) => (observer.span)(name, None),
            None => (self.trace)(name),
        }
    }
    pub fn with_observer(
        device: &'a D,
        law: TensorLaw,
        observer: crate::execution::Observer,
    ) -> Self {
        Self {
            ops: TensorOps::new(device, law),
            trace: |_| None,
            observer: Some(observer),
        }
    }
}

struct FilteredDistribution {
    /// Filtered candidates in the order DS4 uses for categorical draws.
    pub ordered: Vec<(usize, f32)>,
    /// Sum accumulated in `ordered` order, matching DS4's sampling loop.
    pub total: f32,
    /// DS4's return value if rounding leaves the categorical scan unfinished.
    pub fallback: usize,
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
    fn row_weights_on<D: HostAccessDevice + ReadbackDevice>(
        &self,
        ctx: &SamplingContext<'_, D>,
        logits: &D::Tensor,
    ) -> Option<Vec<f32>> {
        if self.temperature <= 1e-5 {
            return None;
        }
        let v = ctx.ops.read_f32(&ctx.ops.flat_f32(logits).ok()?).ok()?;
        let t = 1.0 / self.temperature;
        let m = v.iter().map(|&z| z * t).fold(f32::NEG_INFINITY, f32::max);
        if !m.is_finite() {
            return None;
        }
        Some(v.iter().map(|&z| ((z * t) - m).exp()).collect())
    }

    /// Read one GLM row and apply LISA's optional history penalty before the
    /// reference sampling operations. DS4 itself has no repetition-penalty knob.
    fn ds4_logits_on<D: HostAccessDevice + ReadbackDevice>(
        &self,
        ctx: &SamplingContext<'_, D>,
        logits: &D::Tensor,
        history: &[u32],
    ) -> Option<Vec<f32>> {
        let flat = ctx.ops.flat_f32(logits).ok()?;
        // Wait only through the command buffer that produced this logits
        // array. A global flush also serializes unrelated later work on the
        // shared runtime queue, which is avoidable at a sampling boundary.
        let mut values = ctx.ops.device.read_f32_through(&flat).ok()?;
        if values.is_empty() {
            return None;
        }
        if self.repetition_penalty == 1.0 || history.is_empty() || self.penalty_window == 0 {
            return Some(values);
        }
        let start = history.len().saturating_sub(self.penalty_window);
        let mut seen = std::collections::HashSet::with_capacity(history.len() - start);
        for &token in &history[start..] {
            if let Some(value) = values.get_mut(token as usize)
                && seen.insert(token)
            {
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
    fn sampling_distribution_on<D: HostAccessDevice + ReadbackDevice>(
        &self,
        ctx: &SamplingContext<'_, D>,
        logits: &D::Tensor,
        history: &[u32],
    ) -> Option<FilteredDistribution> {
        let _trace = ctx.span("glm53.sampler.filter");
        if self.temperature <= 0.0 {
            return None;
        }
        let values = self.ds4_logits_on(ctx, logits, history)?;
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
    fn speculation_weights_on<D: HostAccessDevice + ReadbackDevice>(
        &self,
        ctx: &SamplingContext<'_, D>,
        logits: &D::Tensor,
        history: &[u32],
    ) -> Option<Vec<f32>> {
        let values = self.ds4_logits_on(ctx, logits, history)?;
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
    pub fn draft_accept_prob_on<D: HostAccessDevice + ReadbackDevice>(
        &self,
        ctx: &SamplingContext<'_, D>,
        logits: &D::Tensor,
        draft: u32,
    ) -> Option<f32> {
        if self.temperature <= 1e-5 {
            return None;
        }
        let flat = ctx.ops.flat_f32(logits).ok()?;
        let inv = ctx
            .ops
            .full(ctx.ops.device.shape(&flat), 1.0 / self.temperature)
            .ok()?;
        let scaled = ctx
            .ops
            .op(Op::Binary(Binary::Multiply), &[&flat, &inv])
            .ok()?;
        let m = ctx.ops.op(Op::MaxAll, &[&scaled]).ok()?;
        let shifted = ctx
            .ops
            .op(Op::Binary(Binary::Subtract), &[&scaled, &m])
            .ok()?;
        let ex = ctx.ops.op(Op::Unary(Unary::Exp), &[&shifted]).ok()?;
        let z = ctx.ops.op(Op::SumAll, &[&ex]).ok()?;
        let _ = (
            ctx.ops.device.evaluate(&m, true),
            ctx.ops.device.evaluate(&z, true),
        );
        let m_v = ctx.ops.value(&m).ok()?;
        let z_v = ctx.ops.value(&z).ok()?;
        if !(z_v > 0.0) || !z_v.is_finite() {
            return None;
        }
        let idx = ctx
            .ops
            .pull(&(draft as i32).to_ne_bytes(), &[1], ScalarType::I32)
            .ok()?;
        let selected = ctx
            .ops
            .op(Op::TakeAlong { axis: -1 }, &[&scaled, &idx])
            .ok()?;
        let x = ctx.ops.value(&selected).ok()?;
        Some(((x - m_v).exp() / z_v).min(1.0))
    }

    /// History-aware GLM MTP acceptance probability. The history contains the
    /// committed tokens before the row whose logits are being verified.
    pub fn draft_accept_prob_with_history_on<D: HostAccessDevice + ReadbackDevice>(
        &self,
        ctx: &SamplingContext<'_, D>,
        logits: &D::Tensor,
        draft: u32,
        history: &[u32],
    ) -> Option<f32> {
        let weights = self.speculation_weights_on(ctx, logits, history)?;
        let probability = *weights.get(draft as usize)?;
        (probability >= 0.0 && probability.is_finite()).then_some(probability.min(1.0))
    }

    /// Verify GLM's point-mass proposal using one DS4 probability construction.
    /// A rejection removes the proposal's mass from that same distribution
    /// before drawing its replacement, preserving DS4's RNG consumption.
    pub fn verify_draft_with_history_on<D: HostAccessDevice + ReadbackDevice>(
        &mut self,
        ctx: &SamplingContext<'_, D>,
        logits: &D::Tensor,
        draft: u32,
        history: &[u32],
    ) -> Option<(bool, u32)> {
        let mut weights = self.speculation_weights_on(ctx, logits, history)?;
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
    pub fn draw_residual_on<D: HostAccessDevice + ReadbackDevice>(
        &mut self,
        ctx: &SamplingContext<'_, D>,
        logits: &D::Tensor,
        draft: u32,
    ) -> Option<u32> {
        let mut w = self.row_weights_on(ctx, logits)?;
        if (draft as usize) < w.len() {
            w[draft as usize] = 0.0;
        }
        self.categorical(&w)
    }

    /// History-aware residual draw for GLM's point-mass MTP proposal.
    pub fn draw_residual_with_history_on<D: HostAccessDevice + ReadbackDevice>(
        &mut self,
        ctx: &SamplingContext<'_, D>,
        logits: &D::Tensor,
        draft: u32,
        history: &[u32],
    ) -> Option<u32> {
        let mut weights = self.speculation_weights_on(ctx, logits, history)?;
        *weights.get_mut(draft as usize)? = 0.0;
        self.categorical_ds4(&weights)
    }

    /// Plain categorical draw from the row's own distribution (the bonus token
    /// a fully-accepted round earns).
    pub fn draw_row_on<D: HostAccessDevice + ReadbackDevice>(
        &mut self,
        ctx: &SamplingContext<'_, D>,
        logits: &D::Tensor,
    ) -> Option<u32> {
        let w = self.row_weights_on(ctx, logits)?;
        self.categorical(&w)
    }

    /// Draw a verified bonus row under the same history-dependent target
    /// distribution used by ordinary sampling.
    pub fn draw_row_with_history_on<D: HostAccessDevice + ReadbackDevice>(
        &mut self,
        ctx: &SamplingContext<'_, D>,
        logits: &D::Tensor,
        history: &[u32],
    ) -> Option<u32> {
        let weights = self.speculation_weights_on(ctx, logits, history)?;
        self.categorical_ds4(&weights)
    }

    /// GLM sampling path matching DS4's top-k/top-p/min-p filters and
    /// xorshift64* RNG. This intentionally reads logits to the host; the
    /// keyed-Gumbel path remains the fast full-softmax sampler for other paths.
    pub fn draw_ds4_on<D: HostAccessDevice + ReadbackDevice>(
        &mut self,
        ctx: &SamplingContext<'_, D>,
        logits: &D::Tensor,
        history: &[u32],
    ) -> anyhow::Result<u32> {
        let _trace = ctx.span("glm53.sampler.draw");
        if self.temperature <= 0.0 {
            if self.repetition_penalty == 1.0 || history.is_empty() {
                return self.draw_on(ctx, logits, history);
            }
            let values = self
                .ds4_logits_on(ctx, logits, history)
                .ok_or_else(|| anyhow::anyhow!("cannot read GLM logits for sampling"))?;
            return Ok(ds4_argmax_host(&values) as u32);
        }
        if let Some(distribution) = self.sampling_distribution_on(ctx, logits, history)
            && let Some(token) = self.categorical_ds4_ordered(
                &distribution.ordered,
                distribution.total,
                distribution.fallback,
            )
        {
            return Ok(token);
        }
        let values = self
            .ds4_logits_on(ctx, logits, history)
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

    /// [`Self::draw_on`]'s keyed Gumbel draw at an EXPLICIT absolute position —
    /// the GPU-side equivalent of `draw(logits, history)` with
    /// `history.len() == position`. Used where the caller knows the position but
    /// does not hold a history slice (the MTP loop's serial steps), so it never
    /// pulls the vocabulary to the host.
    pub fn draw_at_on<D: HostAccessDevice + ReadbackDevice>(
        &self,
        ctx: &SamplingContext<'_, D>,
        logits: &D::Tensor,
        position: u64,
    ) -> Option<u32> {
        if self.greedy() || self.repetition_penalty != 1.0 {
            return None;
        }
        let flat = ctx.ops.flat_f32(logits).ok()?;
        let tok = ctx
            .ops
            .optional(
                Kernel::KeyedGumbel {
                    inverse_temperature: (1.0 / self.temperature).to_bits(),
                    seed: self.seed,
                    position,
                },
                &[&flat],
            )
            .ok()??;
        ctx.ops.id(&tok).ok()
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
    pub fn draw_on<D: HostAccessDevice + ReadbackDevice>(
        &mut self,
        ctx: &SamplingContext<'_, D>,
        logits: &D::Tensor,
        history: &[u32],
    ) -> anyhow::Result<u32> {
        let _trace = ctx.span("sampler");
        let mut logits = ctx.ops.flat_f32(logits)?;
        if self.greedy() {
            let token = ctx.ops.op(
                Op::Argmax {
                    axis: -1,
                    keep_dims: false,
                },
                &[&logits],
            )?;
            return ctx.ops.id(&token);
        }
        if self.repetition_penalty == 1.0 {
            if let Some(tok) = ctx.ops.optional(
                Kernel::KeyedGumbel {
                    inverse_temperature: (1.0 / self.temperature).to_bits(),
                    seed: self.seed,
                    position: history.len() as u64,
                },
                &[&logits],
            )? {
                return ctx.ops.id(&tok);
            }
        }
        // Preserve the existing composed penalty selection, including its mask
        // semantics. Changing the sampling distribution requires a separate law.
        let width = ctx.ops.device.shape(&logits)[0];
        if self.repetition_penalty != 1.0 && !history.is_empty() {
            let start = history.len().saturating_sub(self.penalty_window);
            let mut mask = vec![0u8; width as usize];
            for &t in &history[start..] {
                if (t as i32) < width {
                    mask[t as usize] = 1;
                }
            }
            let mask = ctx.ops.pull(&mask, &[width], ScalarType::Bool)?;
            let denominator = ctx.ops.full(&[width], self.repetition_penalty)?;
            let pos = ctx
                .ops
                .op(Op::Binary(Binary::Divide), &[&logits, &denominator])?;
            let multiplier = ctx.ops.full(&[width], self.repetition_penalty)?;
            let neg = ctx
                .ops
                .op(Op::Binary(Binary::Multiply), &[&logits, &multiplier])?;
            logits = ctx.ops.op(Op::Where, &[&mask, &pos, &neg])?;
        }
        let temperature = ctx.ops.full(&[width], self.temperature)?;
        logits = ctx
            .ops
            .op(Op::Binary(Binary::Divide), &[&logits, &temperature])?;
        // Native argpartition sorts the whole row and its subsequent slice clamps
        // to the vocabulary width. Make that effective range explicit.
        let k = if self.top_k == 0 {
            64usize.min(width as usize)
        } else {
            self.top_k.min(width as usize)
        };
        let neg = ctx.ops.op(Op::Unary(Unary::Negate), &[&logits])?;
        let idx = ctx.ops.op(
            Op::ArgPartition {
                kth: (k - 1) as i32,
                axis: -1,
            },
            &[&neg],
        )?;
        let idx = ctx.ops.op(
            Op::Slice {
                axis: 0,
                start: 0,
                end: k as i32,
            },
            &[&idx],
        )?;
        let idx = ctx.ops.op(Op::Contiguous, &[&idx])?;
        let vals = ctx.ops.op(Op::TakeAlong { axis: -1 }, &[&logits, &idx])?;
        let _ = (
            ctx.ops.device.evaluate(&idx, true),
            ctx.ops.device.evaluate(&vals, true),
        );
        let ids = ctx.ops.read_ids(&idx)?;
        let mut cand: Vec<(u32, f32)> = ctx
            .ops
            .read_f32(&vals)?
            .into_iter()
            .zip(ids)
            .map(|(v, t)| (t, v))
            .collect();

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
/// Device binding and readback are supplied explicitly by the application.
pub fn token_logprob<D: HostAccessDevice + ReadbackDevice>(
    ctx: &SamplingContext<'_, D>,
    logits: &D::Tensor,
    token: u32,
) -> anyhow::Result<f32> {
    let cast = ctx.ops.op(Op::Cast(ScalarType::F32), &[logits])?;
    let v = ctx.ops.read_f32(&cast)?;
    let m = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let sum: f32 = v.iter().map(|x| (x - m).exp()).sum();
    let lse = m + sum.ln();
    let idx = token as usize;
    Ok(if idx < v.len() { v[idx] - lse } else { 0.0 })
}

#[cfg(test)]
mod rng_tests {
    use super::*;
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
}

/// One token per stream: a single batched argmax when greedy, otherwise a
/// per-stream draw over that stream's (filtered) distribution.
pub fn draw_rows<D: HostAccessDevice + ReadbackDevice>(
    ctx: &SamplingContext<'_, D>,
    logits: &D::Tensor,
    n: usize,
    sampler: &mut Sampler,
    history: &[Vec<u32>],
    ds4_sampling: bool,
) -> anyhow::Result<Vec<u32>> {
    let rows = ctx.ops.op(Op::Reshape(vec![n as i32, -1]), &[logits])?;
    if sampler.greedy() {
        let am = ctx.ops.op(
            Op::Argmax {
                axis: -1,
                keep_dims: false,
            },
            &[&rows],
        )?; // [N, S]
        let _ = ctx.ops.device.evaluate(&am, true);
        let v = ctx.ops.read_ids(&am)?;
        anyhow::ensure!(v.len() == n, "expected {n} tokens, got {}", v.len());
        return Ok(v);
    }
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let row = ctx.ops.op(
            Op::Select {
                axis: 0,
                index: i as i32,
            },
            &[&rows],
        )?;
        let row_history = history.get(i).map(|h| h.as_slice()).unwrap_or(&[]);
        out.push(if ds4_sampling {
            sampler.draw_ds4_on(ctx, &row, row_history)?
        } else {
            sampler.draw_on(ctx, &row, row_history)?
        });
    }
    Ok(out)
}

/// One token per stream with a per-stream sampler: a single batched argmax when
/// every stream is greedy, otherwise a per-stream draw over that stream's
/// (filtered) distribution.
pub fn draw_rows_multi<D: HostAccessDevice + ReadbackDevice, S: AsMut<Sampler>>(
    ctx: &SamplingContext<'_, D>,
    logits: &D::Tensor,
    n: usize,
    samplers: &mut [S],
    history: &[Vec<u32>],
    ds4_sampling: bool,
) -> anyhow::Result<Vec<u32>> {
    let rows = ctx.ops.op(Op::Reshape(vec![n as i32, -1]), &[logits])?;
    if samplers.iter_mut().all(|s| s.as_mut().greedy()) {
        let am = ctx.ops.op(
            Op::Argmax {
                axis: -1,
                keep_dims: false,
            },
            &[&rows],
        )?;
        let _ = ctx.ops.device.evaluate(&am, true);
        let v = ctx.ops.read_ids(&am)?;
        anyhow::ensure!(v.len() == n, "expected {n} tokens, got {}", v.len());
        return Ok(v);
    }
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let row = ctx.ops.op(
            Op::Select {
                axis: 0,
                index: i as i32,
            },
            &[&rows],
        )?;
        let row_history = history.get(i).map(|h| h.as_slice()).unwrap_or(&[]);
        out.push(if ds4_sampling {
            samplers[i].as_mut().draw_ds4_on(ctx, &row, row_history)?
        } else {
            samplers[i].as_mut().draw_on(ctx, &row, row_history)?
        });
    }
    Ok(out)
}

impl AsMut<Sampler> for Sampler {
    fn as_mut(&mut self) -> &mut Sampler {
        self
    }
}

#[cfg(any(test, feature = "diagnostics"))]
#[path = "sampling_oracles.rs"]
pub mod oracles;
#[cfg(test)]
mod host_oracles {
    use super::*;
    #[test]
    fn ds4_sparse_penalty_matches_vocab_scan() {
        let device = lisa_cpu::CpuDevice;
        let ctx = SamplingContext::new(&device, TensorLaw::OrderedF32, |_| None);
        oracles::ds4_sparse_penalty_matches_vocab_scan(&ctx);
    }
    #[test]
    fn exact_speculation_distribution_matches_top_k_top_p_and_min_p() {
        let device = lisa_cpu::CpuDevice;
        let ctx = SamplingContext::new(&device, TensorLaw::OrderedF32, |_| None);
        oracles::exact_speculation_distribution_matches_top_k_top_p_and_min_p(&ctx);
    }
    #[test]
    fn glm_ds4_sampling_order_matches_c_reference() {
        let device = lisa_cpu::CpuDevice;
        let ctx = SamplingContext::new(&device, TensorLaw::OrderedF32, |_| None);
        oracles::glm_ds4_sampling_order_matches_c_reference(&ctx);
    }
    #[test]
    fn glm_ds4_exact_delta_sampling_matches_c_reference() {
        let device = lisa_cpu::CpuDevice;
        let ctx = SamplingContext::new(&device, TensorLaw::OrderedF32, |_| None);
        oracles::glm_ds4_exact_delta_sampling_matches_c_reference(&ctx);
    }
    #[test]
    fn glm_ds4_filter_edges_match_c_reference() {
        let device = lisa_cpu::CpuDevice;
        let ctx = SamplingContext::new(&device, TensorLaw::OrderedF32, |_| None);
        oracles::glm_ds4_filter_edges_match_c_reference(&ctx);
    }
}
