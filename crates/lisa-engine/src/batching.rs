//! Shared cohort/ragged/continuous batching, independent of device storage.
use crate::conversation::TokenSampler;
use crate::eos::is_eos;
use crate::language::{LanguageModel, LanguageRuntime};

/// Engine storage adapter. The device API itself never receives layer caches,
/// token histories, sampler state, or batch membership policy.
pub trait BatchRuntime: LanguageRuntime + Sized {
    type Sampler: TokenSampler<Self>;
    fn tokens(&self, values: &[i32], shape: &[i32]) -> anyhow::Result<Self::Tensor>;
    fn mask(&self, values: &[bool], shape: &[i32]) -> anyhow::Result<Self::Tensor>;
    fn last_hidden(tensor: &Self::Tensor) -> anyhow::Result<Self::Tensor>;
    fn full_offset(cache: &Self::Cache) -> Option<usize>;
    fn has_positions(cache: &Self::Cache) -> bool;
    fn set_mask(cache: &mut Self::Cache, mask: &Self::Tensor);
    fn advance_positions(cache: &mut Self::Cache);
    fn clear_positions(cache: &mut Self::Cache);
    fn trim_memory(&self);
    fn pack_layer(
        &self,
        per: &[Vec<Self::Cache>],
        layer: usize,
        lengths: &[usize],
        max_length: usize,
        capacity: usize,
    ) -> anyhow::Result<Self::Cache>;
    fn extract_streams(
        &self,
        caches: &[Self::Cache],
        survivors: &[usize],
        lengths: &[usize],
        max_length: usize,
    ) -> anyhow::Result<Vec<Vec<Self::Cache>>>;
    fn draw_rows(
        &self,
        logits: &Self::Tensor,
        rows: usize,
        sampler: &mut Self::Sampler,
        history: &[Vec<u32>],
        ds4: bool,
    ) -> anyhow::Result<Vec<u32>>;
    fn draw_rows_multi(
        &self,
        logits: &Self::Tensor,
        rows: usize,
        samplers: &mut [Self::Sampler],
        history: &[Vec<u32>],
        ds4: bool,
    ) -> anyhow::Result<Vec<u32>>;
}

/// Prompt prefill window (`Tower::prefill_multi` chunks at the same size).
pub const PREFILL_CHUNK: usize = 2048;

/// The padded tensor may be at most this multiple of the bytes the group
/// actually needs. Must stay
/// BELOW 2.0 or a two-slot group is never vetoable: one 1-token slot beside
/// one huge slot pads to exactly 2x — the pathological pair this cap exists
/// for.
pub const MAX_PAD_WASTE: f64 = 1.5;

/// Per-slot batched attention floor: past the longest
/// slot's live window the attention reads are per-slot (nothing padded), so
/// the pad-waste cap is void for the group. Must match
/// `Qwen35Attention::per_slot_attention`.
pub const BATCHED_PER_SLOT_ATTN_MIN_KV: usize = 1024;

/// Largest ascending-sorted prefix of `kv_lens_asc` whose padded tensor stays
/// within [`MAX_PAD_WASTE`] of its useful bytes. The slots that fall out are the LONGEST ones — the ones
/// dominating the pad. 0 or 1 = nobody batches.
pub fn batched_kv_keep_count(kv_lens_asc: &[usize]) -> usize {
    if kv_lens_asc.len() < 2 {
        return 0;
    }
    let mut k = kv_lens_asc.len();
    while k >= 2 {
        let sum: usize = kv_lens_asc[..k].iter().sum();
        if sum == 0 {
            return k; // nothing prefilled yet: no padding to waste
        }
        let padded = (k * kv_lens_asc[k - 1]) as f64;
        if padded <= MAX_PAD_WASTE * sum as f64 {
            return k;
        }
        k -= 1;
    }
    0
}

/// [`batched_kv_keep_count`] for a group whose attention may go per-slot: past
/// the per-slot floor nothing is padded, so every slot batches.
pub fn group_keep_count(kv_lens_asc: &[usize], per_slot_capable: bool) -> usize {
    if per_slot_capable
        && kv_lens_asc.len() >= 2
        && kv_lens_asc[kv_lens_asc.len() - 1] >= BATCHED_PER_SLOT_ATTN_MIN_KV
    {
        return kv_lens_asc.len();
    }
    batched_kv_keep_count(kv_lens_asc)
}

/// A prefill narrows to at most [`PREFILL_CHUNK`] while anyone decodes, so an
/// admission prefill cannot monopolize the GPU against live decode ticks. At
/// our default chunk the narrow is a no-op — kept as the policy's safety net.
pub fn company_prefill_chunk(chunk: usize, decoding: usize) -> usize {
    if decoding == 0 {
        return chunk;
    }
    chunk.min(PREFILL_CHUNK)
}

pub struct Batch<R: BatchRuntime> {
    /// Packed cache rows for models whose batched decode uses packed caches.
    /// GLM keeps one cache per session, matching ds4's native batch ownership.
    pub caches: Vec<R::Cache>,
    pub(crate) unpacked_caches: Option<Vec<Vec<R::Cache>>>,
    /// Tokens fed per stream (the committed conversation of each).
    pub fed: Vec<Vec<u32>>,
    pub done: Vec<bool>,
    pub(crate) tails: Vec<Vec<i64>>,
    /// Ragged mode: the per-stream context length after prefill. `None` for a
    /// cohort (equal lengths), where a single shared causal mask suffices.
    pub(crate) ragged: Option<Vec<usize>>,
    /// Packed attention length all streams are right-aligned to.
    pub(crate) lmax: usize,
}

impl<R: BatchRuntime> Batch<R> {
    /// Prefill `prompts` (all of equal length) in one batched forward and
    /// return the first sampled token per stream.
    pub fn prefill(
        tower: &mut dyn LanguageModel<R>,
        prompts: &[Vec<u32>],
        sampler: &mut R::Sampler,
    ) -> anyhow::Result<(Self, Vec<u32>)> {
        let n = prompts.len();
        anyhow::ensure!(n > 0, "empty batch");
        let s = prompts[0].len();
        anyhow::ensure!(
            prompts.iter().all(|p| p.len() == s),
            "batched prefill needs equal-length prompts"
        );

        tower.clear_context();
        let mut caches = tower.new_caches();

        // [N, S] token ids, row-major.
        let flat: Vec<i32> = prompts
            .iter()
            .flat_map(|p| p.iter().map(|&t| t as i32))
            .collect();
        let arr = tower.runtime().tokens(&flat, &[n as i32, s as i32])?;
        let (mixed, _) = tower.forward(&arr, Some(&mut caches))?;
        let last = R::last_hidden(&mixed)?;
        let logits = tower.head(&last)?; // [N, 1, vocab]
        let toks = tower.runtime().draw_rows(
            &logits,
            n,
            sampler,
            &prompts.to_vec(),
            tower.uses_ds4_sampling(),
        )?;

        Ok((
            Batch {
                caches,
                unpacked_caches: None,
                fed: prompts.to_vec(),
                done: vec![false; n],
                tails: Vec::new(),
                ragged: None,
                lmax: s,
            },
            toks,
        ))
    }

    /// Prefill each stream on its own (any length), then prepare lockstep
    /// decode. GLM keeps the per-stream caches independent at every position
    /// because ds4 advances each session's DSA tape and KDA state separately.
    ///
    /// Attention is packed right-aligned: stream `i`'s `L_i` keys occupy
    /// `[lmax-L_i, lmax)` of a shared buffer, so every stream appends its next
    /// key at the same index and a single `offset` works. A per-stream keep
    /// mask hides the unused prefix, and `next_pos` carries each stream's true
    /// token position for RoPE.
    pub fn prefill_ragged(
        tower: &mut dyn LanguageModel<R>,
        prompts: &[Vec<u32>],
        sampler: &mut R::Sampler,
        cap_hint: usize,
    ) -> anyhow::Result<(Self, Vec<u32>)> {
        let n = prompts.len();
        anyhow::ensure!(n > 0, "empty batch");
        let equal_positions = prompts.windows(2).all(|w| w[0].len() == w[1].len());
        if tower.batch_decode_equal_length_only() && equal_positions {
            let mut continuous = ContinuousBatch::empty(cap_hint);
            let mut firsts = Vec::with_capacity(n);
            for prompt in prompts {
                firsts.push(continuous.admit(tower, prompt, sampler)?);
            }
            let batch = Self {
                caches: continuous.caches,
                unpacked_caches: continuous.unpacked_caches,
                fed: continuous.fed,
                done: continuous.done,
                tails: continuous.tails,
                ragged: None,
                lmax: continuous.lmax,
            };
            return Ok((batch, firsts));
        }

        let mut per: Vec<Vec<R::Cache>> = Vec::with_capacity(n);
        let mut firsts = Vec::with_capacity(n);
        let mut tails: Vec<Vec<i64>> = Vec::with_capacity(n);
        for p in prompts {
            tower.clear_context();
            let mut c = tower.new_caches();
            let last = tower.prefill(p, &mut c)?;
            let logits = tower.head(&last)?;
            let t = if tower.uses_ds4_sampling() {
                sampler.draw_ds4(tower.runtime(), &logits, p)?
            } else {
                sampler.draw(tower.runtime(), &logits, p)?
            };
            firsts.push(t);
            per.push(c);
            let ctx = tower.context_window();
            let tailh = p[p.len().saturating_sub(ctx)..]
                .iter()
                .map(|&t| t as i64)
                .collect::<Vec<i64>>();
            tails.push(tailh);
        }
        let lens: Vec<usize> = prompts.iter().map(|p| p.len()).collect();
        let lmax = *lens.iter().max().unwrap_or(&0);
        let cap = lmax + cap_hint + 8;

        let keep_rows_separate = tower.batch_decode_equal_length_only();
        let mut caches: Vec<R::Cache> = Vec::new();
        let unpacked_caches = if keep_rows_separate {
            Some(per)
        } else {
            caches.reserve(per[0].len());
            for li in 0..per[0].len() {
                caches.push(tower.runtime().pack_layer(&per, li, &lens, lmax, cap)?);
            }
            None
        };
        tower.set_context_tails(tails.clone());

        Ok((
            Batch {
                caches,
                unpacked_caches,
                fed: prompts.to_vec(),
                done: vec![false; n],
                tails,
                ragged: Some(lens),
                lmax,
            },
            firsts,
        ))
    }

    /// One batched decode step: feed `tokens` (one per stream), return the next
    /// token per stream. Finished streams are fed their own last token again and
    /// their output is ignored by the caller.
    pub fn step(
        &mut self,
        tower: &mut dyn LanguageModel<R>,
        tokens: &[u32],
        sampler: &mut R::Sampler,
    ) -> anyhow::Result<Vec<u32>> {
        let n = self.fed.len();
        anyhow::ensure!(tokens.len() == n, "step needs one token per stream");

        if let Some(per) = self.unpacked_caches.as_mut() {
            anyhow::ensure!(per.len() == n, "unpacked cache row count mismatch");
            anyhow::ensure!(
                self.tails.len() == n,
                "unpacked context-tail count mismatch"
            );
            tower.set_context_tails(self.tails.clone());
            let flat: Vec<i32> = tokens.iter().map(|&token| token as i32).collect();
            let input = tower.runtime().tokens(&flat, &[n as i32, 1])?;
            let (mixed, _) = tower.forward_ragged(&input, per)?;
            let updated_tails = tower.context_tails();
            if updated_tails.len() == n {
                self.tails = updated_tails;
            }
            for (row, token) in tokens.iter().enumerate() {
                self.fed[row].push(*token);
            }
            tower.set_context_tails(self.tails.clone());
            let last = R::last_hidden(&mixed)?;
            let logits = tower.head_ragged(&last, per)?;
            return tower.runtime().draw_rows(
                &logits,
                n,
                sampler,
                &self.fed,
                tower.uses_ds4_sampling(),
            );
        }

        // Ragged: rebuild the per-stream keep mask for the (growing) key length.
        // Stream i's keys occupy [lmax - L_i, offset + 1); everything before is
        // padding and is masked out.
        if let Some(lens) = &self.ragged {
            let l_after = self
                .caches
                .iter()
                .find_map(|c| R::full_offset(c).map(|offset| offset + 1))
                .unwrap_or(1) as i32;
            let mut m = vec![false; n * l_after as usize];
            for (i, l) in lens.iter().enumerate() {
                let lo = self.lmax - l;
                for j in 0..l_after as usize {
                    m[i * l_after as usize + j] = j >= lo;
                }
            }
            let mask = tower.runtime().mask(&m, &[n as i32, 1, 1, l_after])?;
            for c in self.caches.iter_mut() {
                R::set_mask(c, &mask);
            }
        }

        let flat: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
        let arr = tower.runtime().tokens(&flat, &[n as i32, 1i32])?;
        let (mixed, _) = tower.forward(&arr, Some(&mut self.caches))?;
        let last = R::last_hidden(&mixed)?;
        let logits = tower.head(&last)?;
        let out =
            tower
                .runtime()
                .draw_rows(&logits, n, sampler, &self.fed, tower.uses_ds4_sampling())?;
        for (i, t) in tokens.iter().enumerate() {
            self.fed[i].push(*t);
        }
        if self.ragged.is_some() {
            for c in self.caches.iter_mut() {
                R::advance_positions(c);
            }
        }
        Ok(out)
    }

    /// Batched generation: prefill, then decode in lockstep until every stream
    /// hits EOS or `max_tokens`. Emits `(stream, token)` as they are produced.
    pub fn generate(
        tower: &mut dyn LanguageModel<R>,
        prompts: &[Vec<u32>],
        max_tokens: usize,
        sampler: &mut R::Sampler,
        on_token: &mut dyn FnMut(usize, u32) -> anyhow::Result<()>,
    ) -> anyhow::Result<(Self, Vec<Vec<u32>>)> {
        anyhow::ensure!(!prompts.is_empty(), "empty batch");
        let equal = prompts.windows(2).all(|w| w[0].len() == w[1].len());
        let session_prefill = tower.batch_decode_equal_length_only();
        let (mut batch, first) = if equal && !session_prefill {
            Batch::prefill(tower, prompts, sampler)?
        } else {
            Batch::prefill_ragged(tower, prompts, sampler, max_tokens)?
        };
        let n = prompts.len();
        let mut out: Vec<Vec<u32>> = vec![Vec::new(); n];
        let mut next = first;
        let pack_lens: Vec<usize> = batch.fed.iter().map(Vec::len).collect();
        for i in 0..n {
            out[i].push(next[i]);
            on_token(i, next[i])?;
            batch.done[i] = is_eos(next[i]);
        }
        let mut len = 1usize;
        while len < max_tokens && batch.done.iter().any(|d| !d) {
            if tower.batch_decode_ragged_supported() && batch.done.iter().any(|&done| done) {
                let active: Vec<usize> = (0..n).filter(|&i| !batch.done[i]).collect();
                if batch.unpacked_caches.is_none() {
                    let rows: Vec<usize> = (0..n).collect();
                    batch.unpacked_caches = Some(tower.runtime().extract_streams(
                        &batch.caches,
                        &rows,
                        &pack_lens,
                        batch.lmax,
                    )?);
                    batch.caches.clear();
                }

                let per = batch
                    .unpacked_caches
                    .as_mut()
                    .expect("GLM live-row caches were extracted");
                let active_tokens: Vec<u32> = active.iter().map(|&row| next[row]).collect();
                let active_fed: Vec<Vec<u32>> = active
                    .iter()
                    .map(|&row| {
                        let mut history = batch.fed[row].clone();
                        history.push(next[row]);
                        history
                    })
                    .collect();
                let active_tails: Vec<Vec<i64>> =
                    active.iter().map(|&row| batch.tails[row].clone()).collect();
                let flat: Vec<i32> = active_tokens.iter().map(|&token| token as i32).collect();
                let input = tower.runtime().tokens(&flat, &[active.len() as i32, 1])?;
                let logits = if active.len() == 1 {
                    let row = active[0];
                    tower.set_context_tails(vec![batch.tails[row].clone()]);
                    let (mixed, _) = tower.forward(&input, Some(&mut per[row]))?;
                    let updated_tails = tower.context_tails();
                    if let Some(tail) = updated_tails.first() {
                        batch.tails[row] = tail.clone();
                    }
                    tower.head(&R::last_hidden(&mixed)?)?
                } else {
                    let mut active_caches: Vec<Vec<R::Cache>> = active
                        .iter()
                        .map(|&row| std::mem::take(&mut per[row]))
                        .collect();
                    tower.set_context_tails(active_tails);
                    let (mixed, _) = tower.forward_ragged(&input, &mut active_caches)?;
                    let logits = tower.head_ragged(&R::last_hidden(&mixed)?, &active_caches)?;
                    for (slot, &row) in active.iter().enumerate() {
                        per[row] = std::mem::take(&mut active_caches[slot]);
                    }
                    let updated_tails = tower.context_tails();
                    if updated_tails.len() == active.len() {
                        for (slot, &row) in active.iter().enumerate() {
                            batch.tails[row] = updated_tails[slot].clone();
                        }
                    }
                    logits
                };
                for &row in &active {
                    batch.fed[row].push(next[row]);
                }
                let next_active = tower.runtime().draw_rows(
                    &logits,
                    active.len(),
                    sampler,
                    &active_fed,
                    tower.uses_ds4_sampling(),
                )?;
                for (slot, &row) in active.iter().enumerate() {
                    next[row] = next_active[slot];
                    out[row].push(next[row]);
                    on_token(row, next[row])?;
                    if is_eos(next[row]) {
                        batch.done[row] = true;
                    }
                }
                len += 1;
                continue;
            }

            let step: Vec<u32> = (0..n)
                .map(|i| {
                    if batch.done[i] {
                        *out[i].last().unwrap()
                    } else {
                        next[i]
                    }
                })
                .collect();
            next = batch.step(tower, &step, sampler)?;
            for i in 0..n {
                if batch.done[i] {
                    continue;
                }
                let t = next[i];
                out[i].push(t);
                on_token(i, t)?;
                if is_eos(t) {
                    batch.done[i] = true;
                }
            }
            len += 1;
        }
        Ok((batch, out))
    }
}

pub struct ContinuousBatch<R: BatchRuntime> {
    pub caches: Vec<R::Cache>,
    /// GLM rows with distinct absolute DSA positions retain independent tapes;
    /// the tower still batches their stateless mHC/MLP work via
    /// `LanguageModel::forward_ragged`.
    pub(crate) unpacked_caches: Option<Vec<Vec<R::Cache>>>,
    /// Committed tokens per stream (prompt + generated).
    pub fed: Vec<Vec<u32>>,
    pub done: Vec<bool>,
    /// Per-stream PLE n-gram tails (kept in sync with the caches in `Tower`).
    pub tails: Vec<Vec<i64>>,
    /// Pack-time lengths: stream `i`'s right-align origin is `lmax - lens[i]`,
    /// which stays fixed while every stream advances by one token per step.
    pub(crate) lens: Vec<usize>,
    pub(crate) lmax: usize,
    pub(crate) cap_hint: usize,
}

impl<R: BatchRuntime> ContinuousBatch<R> {
    pub fn empty(cap_hint: usize) -> Self {
        Self {
            caches: Vec::new(),
            unpacked_caches: None,
            fed: Vec::new(),
            done: Vec::new(),
            tails: Vec::new(),
            lens: Vec::new(),
            lmax: 0,
            cap_hint,
        }
    }

    pub fn len(&self) -> usize {
        self.fed.len()
    }

    pub fn is_empty(&self) -> bool {
        self.fed.is_empty()
    }

    /// Live KV lens per stream (committed tokens — each full-attention layer's
    /// live window length at decode time).
    pub fn kv_lens(&self) -> Vec<usize> {
        self.fed.iter().map(|f| f.len()).collect()
    }

    /// May a new stream of `new_len` prompt tokens join without pushing the
    /// group's padded attention past [`MAX_PAD_WASTE`]?
    ///
    /// The keep-count cap is normally applied when forming each tick's decode
    /// group, decoding the longest rejects serially that tick. Our packed
    /// cache makes per-tick re-formation a copy storm (specs/13 §3b), and —
    /// because every stream advances one token per tick — the KV skew is
    /// FIXED at pack time, so the same cap applied at ADMISSION gives the
    /// same protection: a would-be reject waits instead of joining. Past the
    /// per-slot attention floor nothing is padded, so the cap is void —
    /// mirroring the per-slot-capable arm.
    pub fn admits_within_pad_waste(&self, new_len: usize) -> bool {
        if self.len() == 0 {
            return true;
        }
        let mut lens = self.kv_lens();
        lens.push(new_len);
        lens.sort_unstable();
        let per_slot = *lens.last().unwrap() >= BATCHED_PER_SLOT_ATTN_MIN_KV;
        group_keep_count(&lens, per_slot) == lens.len()
    }

    /// GLM's compact DSA tape currently packs only when every stream occupies
    /// the same absolute token positions.
    pub fn admits_equal_length(&self, new_len: usize) -> bool {
        self.fed.iter().all(|stream| stream.len() == new_len)
    }

    /// Prefill `prompt` alone (any length) and merge it into the batch; returns
    /// the first sampled token.
    pub fn admit(
        &mut self,
        tower: &mut dyn LanguageModel<R>,
        prompt: &[u32],
        sampler: &mut R::Sampler,
    ) -> anyhow::Result<u32> {
        anyhow::ensure!(!prompt.is_empty(), "empty prompt");
        let per = prefill_one(tower, prompt)?;
        self.admit_prefilled(tower, per, prompt, sampler)
    }

    /// Prefill `prompt` alone in [`PREFILL_CHUNK`] windows, yielding to the
    /// caller between chunks via `on_chunk` (the prefill/decode interleave:
    /// a long prefill must not stall streams that are already decoding).
    /// Returns the first sampled
    /// token.
    ///
    /// The chunk loop mirrors `Tower::prefill_multi` exactly (same windows,
    /// same per-chunk forward), so the admission result is identical to
    /// [`Self::admit`] — only the wall-clock interleaving differs.
    pub fn admit_chunked(
        &mut self,
        tower: &mut dyn LanguageModel<R>,
        prompt: &[u32],
        sampler: &mut R::Sampler,
        on_chunk: &mut dyn FnMut(&mut Self, &mut dyn LanguageModel<R>) -> anyhow::Result<()>,
    ) -> anyhow::Result<u32> {
        anyhow::ensure!(!prompt.is_empty(), "empty prompt");
        tower.clear_context();
        let mut caches = tower.new_caches();
        let mut last = None;
        for chunk in prompt.chunks(PREFILL_CHUNK) {
            let ids: Vec<i32> = chunk.iter().map(|&t| t as i32).collect();
            let arr = tower.runtime().tokens(&ids, &[1i32, chunk.len() as i32])?;
            let (h, _) = tower.forward(&arr, Some(&mut caches))?;
            last = Some(R::last_hidden(&h)?);
            tower.runtime().trim_memory();
            on_chunk(self, tower)?;
        }
        let per = Prefilled {
            caches,
            last_hidden: last.expect("non-empty prompt"),
        };
        self.admit_prefilled(tower, per, prompt, sampler)
    }

    /// Merge an already-prefilled stream into the batch (the shared tail of
    /// [`Self::admit`] / [`Self::admit_chunked`]).
    fn admit_prefilled(
        &mut self,
        tower: &mut dyn LanguageModel<R>,
        per: Prefilled<R>,
        prompt: &[u32],
        sampler: &mut R::Sampler,
    ) -> anyhow::Result<u32> {
        let logits = tower.head(&per.last_hidden)?;
        let tok = if tower.uses_ds4_sampling() {
            sampler.draw_ds4(tower.runtime(), &logits, prompt)?
        } else {
            sampler.draw(tower.runtime(), &logits, prompt)?
        };
        let ctx = tower.context_window();
        let tail: Vec<i64> = prompt[prompt.len().saturating_sub(ctx)..]
            .iter()
            .map(|&t| t as i64)
            .collect();
        let keep = vec![true; self.len()];
        self.repack(tower, &keep, vec![(per.caches, prompt.to_vec(), tail)])?;
        Ok(tok)
    }

    /// Drop the streams whose `keep[i]` is false, compacting the packed caches.
    pub fn retire(
        &mut self,
        tower: &mut dyn LanguageModel<R>,
        keep: &[bool],
    ) -> anyhow::Result<()> {
        anyhow::ensure!(keep.len() == self.len(), "keep length mismatch");
        if keep.iter().all(|&k| k) {
            return Ok(());
        }
        self.repack(tower, keep, Vec::new())
    }

    /// One batched decode step (one token per stream, one sampler per stream);
    /// returns the next token per stream. Callers re-feed a finished stream's
    /// last token (its output is ignored) until they retire it.
    pub fn step(
        &mut self,
        tower: &mut dyn LanguageModel<R>,
        tokens: &[u32],
        samplers: &mut [R::Sampler],
    ) -> anyhow::Result<Vec<u32>> {
        let n = self.len();
        anyhow::ensure!(tokens.len() == n, "step needs one token per stream");
        anyhow::ensure!(samplers.len() == n, "step needs one sampler per stream");
        anyhow::ensure!(n > 0, "empty batch");

        tower.set_context_tails(self.tails.clone());

        // Ragged packing: stream i's keys occupy `[lmax - lens[i], offset + 1)`;
        // the mask hides the padding prefix. Collapsed (single-stream) batches
        // carry no `next_pos`, so they take the plain causal path.
        let ragged = self.caches.iter().any(R::has_positions);
        if ragged {
            let l_after = self
                .caches
                .iter()
                .find_map(|c| R::full_offset(c).map(|offset| offset + 1))
                .unwrap_or(1) as i32;
            let mut m = vec![false; n * l_after as usize];
            for i in 0..n {
                let lo = self.lmax - self.lens[i];
                for j in 0..l_after as usize {
                    m[i * l_after as usize + j] = j >= lo;
                }
            }
            let mask = tower.runtime().mask(&m, &[n as i32, 1, 1, l_after])?;
            for c in self.caches.iter_mut() {
                R::set_mask(c, &mask);
            }
        }

        let flat: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
        let arr = tower.runtime().tokens(&flat, &[n as i32, 1i32])?;
        let row_local_cache = self.unpacked_caches.is_some();
        let (mixed, _) = if let Some(per) = self.unpacked_caches.as_mut() {
            if n == 1 {
                // GLM's row-aware path is for cohorts with distinct cache
                // positions. Once EOS retirement leaves one stream, resume
                // the normal single-session path instead of asking the model
                // to run a one-row ragged forward.
                tower.forward(&arr, Some(&mut per[0]))?
            } else {
                tower.forward_ragged(&arr, per)?
            }
        } else {
            tower.forward(&arr, Some(&mut self.caches))?
        };
        // Advance the committed history before drawing, so the sampler sees the
        // token it just consumed (matching `generate`/`Session`).
        for (i, t) in tokens.iter().enumerate() {
            self.fed[i].push(*t);
        }
        // The forward rewrote `Tower::ngram_history`; keep the stream tails in
        // sync so a later admit (which clobbers it) can restore them.
        let tails = tower.context_tails();
        if let Some(t) = tails.first().map(|_| tails.clone()) {
            self.tails = t;
        }
        let last = R::last_hidden(&mixed)?;
        let logits = if row_local_cache && n > 1 {
            let per = self
                .unpacked_caches
                .as_ref()
                .expect("row-local caches were present before forward");
            tower.head_ragged(&last, per)?
        } else {
            tower.head(&last)?
        };
        let out = tower.runtime().draw_rows_multi(
            &logits,
            n,
            samplers,
            &self.fed,
            tower.uses_ds4_sampling(),
        )?;
        if ragged {
            for c in self.caches.iter_mut() {
                R::advance_positions(c);
            }
        }
        Ok(out)
    }

    /// Rebuild the packed caches from the surviving streams plus `extra`
    /// (freshly prefilled streams: caches, committed tokens, PLE tail).
    fn repack(
        &mut self,
        tower: &mut dyn LanguageModel<R>,
        keep: &[bool],
        extra: Vec<(Vec<R::Cache>, Vec<u32>, Vec<i64>)>,
    ) -> anyhow::Result<()> {
        let survivors: Vec<usize> = (0..self.len()).filter(|&i| keep[i]).collect();

        let mut per: Vec<Vec<R::Cache>> = Vec::with_capacity(survivors.len() + extra.len());
        let mut lens: Vec<usize> = Vec::with_capacity(survivors.len() + extra.len());
        let mut tails: Vec<Vec<i64>> = Vec::new();
        let mut fed: Vec<Vec<u32>> = Vec::new();
        let mut done: Vec<bool> = Vec::new();

        if let Some(mut unpacked) = self.unpacked_caches.take() {
            for &i in &survivors {
                per.push(std::mem::replace(&mut unpacked[i], Vec::new()));
                lens.push(self.fed[i].len());
                tails.push(self.tails[i].clone());
                fed.push(self.fed[i].clone());
                done.push(self.done[i]);
            }
        } else if !self.caches.is_empty() {
            let mut extracted = tower
                .runtime()
                .extract_streams(&self.caches, &survivors, &self.lens, self.lmax)?
                .into_iter();
            for &i in survivors.iter() {
                per.push(extracted.next().expect("one per survivor"));
                lens.push(self.fed[i].len());
                tails.push(self.tails[i].clone());
                fed.push(self.fed[i].clone());
                done.push(self.done[i]);
            }
        }
        for (c, f, t) in extra {
            lens.push(f.len());
            fed.push(f);
            tails.push(t);
            done.push(false);
            per.push(c);
        }

        if per.is_empty() {
            self.caches.clear();
            self.unpacked_caches = None;
            self.fed.clear();
            self.done.clear();
            self.tails.clear();
            self.lens.clear();
            self.lmax = 0;
            tower.clear_context();
            return Ok(());
        }

        let nlayers = per[0].len();
        let lmax = *lens.iter().max().unwrap_or(&0);
        let cap = lmax + self.cap_hint + 8;
        let keep_rows_separate = tower.batch_decode_equal_length_only();
        let mut caches: Vec<R::Cache> = Vec::new();
        if keep_rows_separate {
            self.unpacked_caches = Some(per);
        } else {
            caches.reserve(nlayers);
            for li in 0..nlayers {
                caches.push(tower.runtime().pack_layer(&per, li, &lens, lmax, cap)?);
            }
            // A lone stream is contiguously left-aligned (origin 0): drop the ragged
            // mask so it uses the plain causal (fused) path.
            if per.len() == 1 {
                for c in caches.iter_mut() {
                    R::clear_positions(c);
                }
            }
            self.unpacked_caches = None;
        }

        self.caches = caches;
        self.fed = fed;
        self.done = done;
        self.tails = tails;
        self.lens = lens;
        self.lmax = lmax;
        tower.set_context_tails(self.tails.clone());
        Ok(())
    }
}

pub struct Prefilled<R: BatchRuntime> {
    pub caches: Vec<R::Cache>,
    pub last_hidden: R::Tensor,
}
/// Prefill `prompt` alone (B=1, fresh caches) and return the caches plus the
/// last-position hidden state.
pub fn prefill_one<R: BatchRuntime>(
    tower: &mut dyn LanguageModel<R>,
    prompt: &[u32],
) -> anyhow::Result<Prefilled<R>> {
    tower.clear_context();
    let mut caches = tower.new_caches();
    let last = tower.prefill(prompt, &mut caches)?;
    Ok(Prefilled {
        caches,
        last_hidden: last,
    })
}

#[cfg(any(test, feature = "diagnostics"))]
impl<R: BatchRuntime> ContinuousBatch<R> {
    pub fn unpacked_caches_for_test(&self) -> &Option<Vec<Vec<R::Cache>>> {
        &self.unpacked_caches
    }
    pub fn unpacked_caches_mut_for_test(&mut self) -> &mut Option<Vec<Vec<R::Cache>>> {
        &mut self.unpacked_caches
    }
}
