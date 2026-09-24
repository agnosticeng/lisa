//! Batching: cohort, ragged, and continuous.
//!
//! [`Batch`] is the fixed-membership case: all streams are prefilled up front
//! (together for equal lengths, or alone then packed when ragged) and then
//! advanced in lockstep. [`ContinuousBatch`] is the serving case: streams are
//! admitted and retired as they arrive and finish, and the packed caches are
//! compacted on every membership change (the reference engine's CBv2 idea).
//!
//! The model is written with `B` as a batch dimension throughout (attention,
//! GDN state, PLE history), so a batched forward is a single `Tower::forward`
//! over `[N, S]` tokens. The MoE falls back to MLX's generic (batched) path
//! because the engine's fused decode/prefill kernels are single-batch.

use lisa_mlx::ops::indexing::{IndexMutOp, IndexOp};
use lisa_mlx::{ops, Array};

use crate::core::generate::is_eos;
use crate::core::sampler::Sampler;
use crate::core::cache::{FullAttentionCache, GdnCache, IndexerTape, LayerCache};
use crate::models::LanguageModel;

pub struct Batch {
    pub caches: Vec<LayerCache>,
    /// Tokens fed per stream (the committed conversation of each).
    pub fed: Vec<Vec<u32>>,
    pub done: Vec<bool>,
    /// Ragged mode: the per-stream context length after prefill. `None` for a
    /// cohort (equal lengths), where a single shared causal mask suffices.
    ragged: Option<Vec<usize>>,
    /// Packed attention length all streams are right-aligned to.
    lmax: usize,
}

impl Batch {
    /// Prefill `prompts` (all of equal length) in one batched forward and
    /// return the first sampled token per stream.
    pub fn prefill(
        tower: &mut dyn LanguageModel,
        prompts: &[Vec<u32>],
        sampler: &mut Sampler,
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
        let arr = Array::from_slice(&flat, &[n as i32, s as i32]);
        let (mixed, _) = tower.forward(&arr, Some(&mut caches))?;
        let last = mixed.index((.., mixed.dim(1) - 1, ..));
        let logits = tower.head(&last)?; // [N, 1, vocab]
        let toks = draw_rows(&logits, n, sampler, &prompts.to_vec())?;

        Ok((
            Batch {
                caches,
                fed: prompts.to_vec(),
                done: vec![false; n],
                ragged: None,
                lmax: s,
            },
            toks,
        ))
    }


    /// Prefill each stream on its own (any length), then pack the per-stream
    /// caches into one batched cache for lockstep decode.
    ///
    /// Attention is packed right-aligned: stream `i`'s `L_i` keys occupy
    /// `[lmax-L_i, lmax)` of a shared buffer, so every stream appends its next
    /// key at the same index and a single `offset` works. A per-stream keep
    /// mask hides the unused prefix, and `next_pos` carries each stream's true
    /// token position for RoPE.
    pub fn prefill_ragged(
        tower: &mut dyn LanguageModel,
        prompts: &[Vec<u32>],
        sampler: &mut Sampler,
        cap_hint: usize,
    ) -> anyhow::Result<(Self, Vec<u32>)> {
        let n = prompts.len();
        let mut per: Vec<Vec<LayerCache>> = Vec::with_capacity(n);
        let mut firsts = Vec::with_capacity(n);
        let mut tails: Vec<Vec<i64>> = Vec::with_capacity(n);
        for p in prompts {
            tower.clear_context();
            let mut c = tower.new_caches();
            let last = tower.prefill(p, &mut c)?;
            let logits = tower.head(&last)?;
            let t = sampler.draw(&logits, p)?;
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

        // Pack layer by layer.
        let mut caches: Vec<LayerCache> = Vec::with_capacity(per[0].len());
        for li in 0..per[0].len() {
            caches.push(pack_layer(&per, li, &lens, lmax, cap)?);
        }
        tower.set_context_tails(tails);

        Ok((
            Batch {
                caches,
                fed: prompts.to_vec(),
                done: vec![false; n],
                ragged: Some(lens),
                lmax,
            },
            firsts,
        ))
    }

    /// One batched decode step: feed `tokens` (one per stream), return the next
    /// token per stream. Finished streams are fed their own last token again and
    /// their output is ignored by the caller.
    pub fn step(&mut self, tower: &mut dyn LanguageModel, tokens: &[u32], sampler: &mut Sampler) -> anyhow::Result<Vec<u32>> {
        let n = self.fed.len();
        anyhow::ensure!(tokens.len() == n, "step needs one token per stream");

        // Ragged: rebuild the per-stream keep mask for the (growing) key length.
        // Stream i's keys occupy [lmax - L_i, offset + 1); everything before is
        // padding and is masked out.
        if let Some(lens) = &self.ragged {
            let l_after = self
                .caches
                .iter()
                .find_map(|c| match c {
                    LayerCache::Full(f) => Some(f.offset + 1),
                    _ => None,
                })
                .unwrap_or(1) as i32;
            let mut m = vec![false; n * l_after as usize];
            for (i, l) in lens.iter().enumerate() {
                let lo = self.lmax - l;
                for j in 0..l_after as usize {
                    m[i * l_after as usize + j] = j >= lo;
                }
            }
            let mask = Array::from_slice(&m, &[n as i32, 1, 1, l_after]);
            for c in self.caches.iter_mut() {
                if let LayerCache::Full(f) = c {
                    f.attn_mask = Some(mask.clone());
                }
            }
        }

        let flat: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
        let arr = Array::from_slice(&flat, &[n as i32, 1i32]);
        let (mixed, _) = tower.forward(&arr, Some(&mut self.caches))?;
        let last = mixed.index((.., mixed.dim(1) - 1, ..));
        let logits = tower.head(&last)?;
        let out = draw_rows(&logits, n, sampler, &self.fed)?;
        for (i, t) in tokens.iter().enumerate() {
            self.fed[i].push(*t);
        }
        if self.ragged.is_some() {
            for c in self.caches.iter_mut() {
                if let LayerCache::Full(f) = c {
                    if let Some(v) = f.next_pos.as_mut() {
                        for p in v.iter_mut() {
                            *p += 1;
                        }
                    }
                }
            }
        }
        Ok(out)
    }

    /// Batched generation: prefill, then decode in lockstep until every stream
    /// hits EOS or `max_tokens`. Emits `(stream, token)` as they are produced.
    pub fn generate(
        tower: &mut dyn LanguageModel,
        prompts: &[Vec<u32>],
        max_tokens: usize,
        sampler: &mut Sampler,
        on_token: &mut dyn FnMut(usize, u32) -> anyhow::Result<()>,
    ) -> anyhow::Result<(Self, Vec<Vec<u32>>)> {
        let equal = prompts.windows(2).all(|w| w[0].len() == w[1].len());
        let (mut batch, first) = if equal {
            Batch::prefill(tower, prompts, sampler)?
        } else {
            Batch::prefill_ragged(tower, prompts, sampler, max_tokens)?
        };
        let n = prompts.len();
        let mut out: Vec<Vec<u32>> = vec![Vec::new(); n];
        let mut next = first;
        for i in 0..n {
            out[i].push(next[i]);
            on_token(i, next[i])?;
        }
        let mut len = 1usize;
        while len < max_tokens && batch.done.iter().any(|d| !d) {
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

/// One token per stream: a single batched argmax when greedy, otherwise a
/// per-stream draw over that stream's (filtered) distribution.
fn draw_rows(
    logits: &Array,
    n: usize,
    sampler: &mut Sampler,
    history: &[Vec<u32>],
) -> anyhow::Result<Vec<u32>> {
    let rows = logits.reshape(&[n as i32, -1])?;
    if sampler.greedy() {
        let am = lisa_mlx::ops::indexing::argmax_axis(&rows, -1, false)?; // [N, S]
        let _ = am.eval();
        let v: Vec<u32> = am.as_slice::<u32>().to_vec();
        anyhow::ensure!(v.len() == n, "expected {n} tokens, got {}", v.len());
        return Ok(v);
    }
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let row = rows.index((i as i32, ..));
        out.push(sampler.draw(&row, history.get(i).map(|h| h.as_slice()).unwrap_or(&[]))?);
    }
    Ok(out)
}

/// One token per stream with a per-stream sampler: a single batched argmax when
/// every stream is greedy, otherwise a per-stream draw over that stream's
/// (filtered) distribution.
pub(crate) fn draw_rows_multi(
    logits: &Array,
    n: usize,
    samplers: &mut [Sampler],
    history: &[Vec<u32>],
) -> anyhow::Result<Vec<u32>> {
    let rows = logits.reshape(&[n as i32, -1])?;
    if samplers.iter().all(|s| s.greedy()) {
        let am = lisa_mlx::ops::indexing::argmax_axis(&rows, -1, false)?;
        let _ = am.eval();
        let v: Vec<u32> = am.as_slice::<u32>().to_vec();
        anyhow::ensure!(v.len() == n, "expected {n} tokens, got {}", v.len());
        return Ok(v);
    }
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let row = rows.index((i as i32, ..));
        out.push(samplers[i].draw(&row, history.get(i).map(|h| h.as_slice()).unwrap_or(&[]))?);
    }
    Ok(out)
}

/// Pack one layer's per-stream caches into a single batched cache.
fn pack_layer(
    per: &[Vec<LayerCache>],
    li: usize,
    lens: &[usize],
    lmax: usize,
    cap: usize,
) -> anyhow::Result<LayerCache> {
    match &per[0][li] {
        LayerCache::Full(tmpl) => {
            let h = tmpl.kv_heads as i32;
            let d = tmpl.head_dim as i32;
            let n = per.len() as i32;
            let mut keys = ops::zeros::<half::bf16>(&[n, h, cap as i32, d])?;
            let mut values = ops::zeros::<half::bf16>(&[n, h, cap as i32, d])?;
            for (i, caches) in per.iter().enumerate() {
                let LayerCache::Full(c) = &caches[li] else {
                    anyhow::bail!("cache kind mismatch");
                };
                let li_len = lens[i] as i32;
                let dst = lmax as i32 - li_len;
                if let Some(k) = &c.keys {
                    let src = k.index((.., .., 0..li_len, ..));
                    keys.index_mut((i as i32, .., dst..(dst + li_len), ..), src);
                }
                if let Some(v) = &c.values {
                    let src = v.index((.., .., 0..li_len, ..));
                    values.index_mut((i as i32, .., dst..(dst + li_len), ..), src);
                }
            }
            Ok(LayerCache::Full(FullAttentionCache {
                keys: Some(keys),
                values: Some(values),
                offset: lmax,
                capacity: cap,
                kv_heads: tmpl.kv_heads,
                head_dim: tmpl.head_dim,
                indexer_tape: IndexerTape::new(),
                indexer_ok: false,
                ple_conv: None,
                capture_ple_full: None,
                next_pos: Some(lens.to_vec()),
                attn_mask: None,
            }))
        }
        LayerCache::Linear(_) => {
            let conv = concat_opt(per, li, |c| c.conv.as_ref())?;
            let ssm = concat_opt(per, li, |c| c.ssm.as_ref())?;
            let ple = concat_opt(per, li, |c| c.ple_conv.as_ref())?;
            Ok(LayerCache::Linear(GdnCache {
                conv,
                ssm,
                capture_ssm: None,
                capture_conv_input: None,
                rec_y_buf: None,
                ple_conv: ple,
                capture_ple_full: None,
            }))
        }
    }
}

fn concat_opt(
    per: &[Vec<LayerCache>],
    li: usize,
    get: impl Fn(&GdnCache) -> Option<&Array>,
) -> anyhow::Result<Option<Array>> {
    let mut parts: Vec<&Array> = Vec::new();
    for caches in per {
        let LayerCache::Linear(g) = &caches[li] else {
            anyhow::bail!("cache kind mismatch");
        };
        match get(g) {
            Some(a) => parts.push(a),
            None => return Ok(None),
        }
    }
    if parts.is_empty() {
        return Ok(None);
    }
    Ok(Some(ops::concatenate(&parts, 0)?))
}

/// Continuous batching: streams are admitted and retired as they arrive and
/// finish, and the packed caches are compacted on every membership change.
///
/// The layout is the same right-aligned packing as [`Batch::prefill_ragged`]
/// (`next_pos` + per-stream `attn_mask`). When only one stream is live the
/// packing is collapsed to a contiguously left-aligned cache with no mask, so a
/// lone stream takes the exact single-stream decode path.
pub struct ContinuousBatch {
    pub caches: Vec<LayerCache>,
    /// Committed tokens per stream (prompt + generated).
    pub fed: Vec<Vec<u32>>,
    pub done: Vec<bool>,
    /// Per-stream PLE n-gram tails (kept in sync with the caches in `Tower`).
    pub tails: Vec<Vec<i64>>,
    /// Pack-time lengths: stream `i`'s right-align origin is `lmax - lens[i]`,
    /// which stays fixed while every stream advances by one token per step.
    lens: Vec<usize>,
    lmax: usize,
    cap_hint: usize,
}

impl ContinuousBatch {
    pub fn empty(cap_hint: usize) -> Self {
        Self {
            caches: Vec::new(),
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

    /// Prefill `prompt` alone (any length) and merge it into the batch; returns
    /// the first sampled token.
    pub fn admit(
        &mut self,
        tower: &mut dyn LanguageModel,
        prompt: &[u32],
        sampler: &mut Sampler,
    ) -> anyhow::Result<u32> {
        anyhow::ensure!(!prompt.is_empty(), "empty prompt");
        let per = prefill_one(tower, prompt)?;
        let logits = tower.head(&per.last_hidden)?;
        let tok = sampler.draw(&logits, prompt)?;
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
    pub fn retire(&mut self, tower: &mut dyn LanguageModel, keep: &[bool]) -> anyhow::Result<()> {
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
        tower: &mut dyn LanguageModel,
        tokens: &[u32],
        samplers: &mut [Sampler],
    ) -> anyhow::Result<Vec<u32>> {
        let n = self.len();
        anyhow::ensure!(tokens.len() == n, "step needs one token per stream");
        anyhow::ensure!(samplers.len() == n, "step needs one sampler per stream");
        anyhow::ensure!(n > 0, "empty batch");

        tower.set_context_tails(self.tails.clone());

        // Ragged packing: stream i's keys occupy `[lmax - lens[i], offset + 1)`;
        // the mask hides the padding prefix. Collapsed (single-stream) batches
        // carry no `next_pos`, so they take the plain causal path.
        let ragged = self.caches.iter().any(|c| matches!(c, LayerCache::Full(f) if f.next_pos.is_some()));
        if ragged {
            let l_after = self
                .caches
                .iter()
                .find_map(|c| match c {
                    LayerCache::Full(f) => Some(f.offset + 1),
                    _ => None,
                })
                .unwrap_or(1) as i32;
            let mut m = vec![false; n * l_after as usize];
            for i in 0..n {
                let lo = self.lmax - self.lens[i];
                for j in 0..l_after as usize {
                    m[i * l_after as usize + j] = j >= lo;
                }
            }
            let mask = Array::from_slice(&m, &[n as i32, 1, 1, l_after]);
            for c in self.caches.iter_mut() {
                if let LayerCache::Full(f) = c {
                    f.attn_mask = Some(mask.clone());
                }
            }
        }

        let flat: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
        let arr = Array::from_slice(&flat, &[n as i32, 1i32]);
        let (mixed, _) = tower.forward(&arr, Some(&mut self.caches))?;
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
        let last = mixed.index((.., mixed.dim(1) - 1, ..));
        let logits = tower.head(&last)?;
        let out = draw_rows_multi(&logits, n, samplers, &self.fed)?;
        if ragged {
            for c in self.caches.iter_mut() {
                if let LayerCache::Full(f) = c {
                    if let Some(v) = f.next_pos.as_mut() {
                        for p in v.iter_mut() {
                            *p += 1;
                        }
                    }
                }
            }
        }
        Ok(out)
    }

    /// Rebuild the packed caches from the surviving streams plus `extra`
    /// (freshly prefilled streams: caches, committed tokens, PLE tail).
    fn repack(
        &mut self,
        tower: &mut dyn LanguageModel,
        keep: &[bool],
        extra: Vec<(Vec<LayerCache>, Vec<u32>, Vec<i64>)>,
    ) -> anyhow::Result<()> {
        let survivors: Vec<usize> = (0..self.len()).filter(|&i| keep[i]).collect();

        let mut per: Vec<Vec<LayerCache>> = Vec::with_capacity(survivors.len() + extra.len());
        let mut lens: Vec<usize> = Vec::with_capacity(survivors.len() + extra.len());
        let mut tails: Vec<Vec<i64>> = Vec::new();
        let mut fed: Vec<Vec<u32>> = Vec::new();
        let mut done: Vec<bool> = Vec::new();

        if !self.caches.is_empty() {
            let mut extracted = extract_streams(&self.caches, &survivors, &self.lens, self.lmax)?.into_iter();
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
        let mut caches: Vec<LayerCache> = Vec::with_capacity(nlayers);
        for li in 0..nlayers {
            caches.push(pack_layer(&per, li, &lens, lmax, cap)?);
        }
        // A lone stream is contiguously left-aligned (origin 0): drop the ragged
        // mask so it uses the plain causal (fused) path.
        if per.len() == 1 {
            for c in caches.iter_mut() {
                if let LayerCache::Full(f) = c {
                    f.next_pos = None;
                    f.attn_mask = None;
                }
            }
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

pub(crate) struct Prefilled {
    caches: Vec<LayerCache>,
    last_hidden: Array,
}
/// Prefill `prompt` alone (B=1, fresh caches) and return the caches plus the
/// last-position hidden state.
pub(crate) fn prefill_one(tower: &mut dyn LanguageModel, prompt: &[u32]) -> anyhow::Result<Prefilled> {
    tower.clear_context();
    let mut caches = tower.new_caches();
    let last = tower.prefill(prompt, &mut caches)?;
    Ok(Prefilled { caches, last_hidden: last })
}

/// Slice the packed caches back into per-stream (B=1) caches for `survivors`,
/// each holding exactly its live tokens (right-aligned `[origin, offset)`).
fn extract_streams(
    packed: &[LayerCache],
    survivors: &[usize],
    pack_lens: &[usize],
    lmax: usize,
) -> anyhow::Result<Vec<Vec<LayerCache>>> {
    let offset = packed
        .iter()
        .find_map(|c| match c {
            LayerCache::Full(f) => Some(f.offset),
            _ => None,
        })
        .unwrap_or(0);
    let mut out = Vec::with_capacity(survivors.len());
    for &i in survivors {
        let origin = lmax - pack_lens[i];
        let range = origin as i32..offset as i32;
        let row = i as i32..i as i32 + 1;
        let mut layers = Vec::with_capacity(packed.len());
        for c in packed {
            match c {
                LayerCache::Full(f) => {
                    let k = f.keys.as_ref().unwrap().index((row.clone(), .., range.clone(), ..)).contiguous()?;
                    let v = f.values.as_ref().unwrap().index((row.clone(), .., range.clone(), ..)).contiguous()?;
                    layers.push(LayerCache::Full(FullAttentionCache {
                        keys: Some(k),
                        values: Some(v),
                        offset: offset - origin,
                        capacity: offset - origin,
                        kv_heads: f.kv_heads,
                        head_dim: f.head_dim,
                        indexer_tape: IndexerTape::new(),
                        indexer_ok: false,
                        ple_conv: None,
                        capture_ple_full: None,
                        next_pos: None,
                        attn_mask: None,
                    }));
                }
                LayerCache::Linear(g) => {
                    let one = |a: &Option<Array>| -> anyhow::Result<Option<Array>> {
                        match a {
                            Some(x) => Ok(Some(x.index(row.clone()).contiguous()?)),
                            None => Ok(None),
                        }
                    };
                    layers.push(LayerCache::Linear(GdnCache {
                        conv: one(&g.conv)?,
                        ssm: one(&g.ssm)?,
                        ple_conv: one(&g.ple_conv)?,
                        capture_ssm: None,
                        capture_conv_input: None,
                rec_y_buf: None,
                        capture_ple_full: None,
                    }));
                }
            }
        }
        out.push(layers);
    }
    Ok(out)
}
