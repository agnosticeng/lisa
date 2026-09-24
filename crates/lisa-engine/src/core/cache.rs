//! Per-layer KV / recurrent caches. These are the interface between a model
//! and the batching runtime, so they live in `core` even though the concrete
//! variants (full attention, gated deltanet) are Qwen4-shaped for now.

use lisa_mlx::ops::indexing::{IndexMutOp, IndexOp};
use lisa_mlx::{ops, Array};

pub struct IndexerTape {
    /// Raw indexer keys `[B, cap, D]`, appended in place (doubling), so an
    /// append is O(s) rather than a whole-tape copy.
    raw: Option<Array>,
    cap: usize,
    /// Total raw tokens appended.
    pub total: usize,
    /// Normalized + roped pooled keys `[B, pcap, D]`, written in place.
    pooled: Option<Array>,
    pcap: usize,
    /// Number of blocks already pooled. Pooling is lazy: nothing is pooled
    /// until the visible context exceeds the budget.
    pub pooled_upto: usize,
    cr: usize,
}

impl IndexerTape {
    pub fn new() -> Self {
        Self { raw: None, cap: 0, total: 0, pooled: None, pcap: 0, pooled_upto: 0, cr: 0 }
    }

    fn grow_raw(&mut self, needed: usize, b: i32, d: i32) -> lisa_mlx::error::Result<()> {
        let same_batch = self.raw.as_ref().map_or(true, |r| r.dim(0) == b);
        if self.cap >= needed && same_batch {
            return Ok(());
        }
        let mut new_cap = if self.cap == 0 { needed + 256 } else { self.cap };
        while new_cap < needed {
            new_cap *= 2;
        }
        let mut np = ops::zeros::<half::bf16>(&[b, new_cap as i32, d])?;
        if let Some(old) = &self.raw {
            let keep = self.total as i32;
            if keep > 0 {
                np.index_mut((.., 0..keep, ..), old.index((.., 0..keep, ..)));
            }
        }
        self.raw = Some(np);
        self.cap = new_cap;
        Ok(())
    }

    /// Append `raw_k` `[B, s, D]` in place.
    pub fn append(&mut self, raw_k: &Array, cr: usize) -> lisa_mlx::error::Result<()> {
        self.cr = cr;
        let (b, s, d) = (raw_k.dim(0), raw_k.dim(1), raw_k.dim(2));
        self.grow_raw(self.total + s as usize, b, d)?;
        let stream = lisa_mlx::Stream::thread_local_or_default();
        let raw = self.raw.as_ref().unwrap();
        if lisa_mlx::kernels::copy_rows_into(raw_k, raw, self.total as i32, &stream).is_none() {
            let range = self.total as i32..(self.total + s as usize) as i32;
            self.raw
                .as_mut()
                .unwrap()
                .index_mut((.., range, ..), raw_k.clone());
        }
        self.total += s as usize;
        Ok(())
    }

    /// The raw keys for token range `[start, end)`, `[B, end-start, D]`.
    pub fn raw_slice(&self, start: usize, end: usize) -> Option<Array> {
        self.raw
            .as_ref()
            .map(|r| r.index((.., start as i32..end as i32, ..)))
    }

    /// Append `pooled` `[B, n_new, D]` into the pooled buffer.
    pub fn push_pooled(&mut self, pooled: &Array) -> lisa_mlx::error::Result<()> {
        let b = pooled.dim(0);
        let d = pooled.dim(2);
        let n_new = pooled.dim(1) as usize;
        let needed = self.pooled_upto + n_new;
        let same_batch = self.pooled.as_ref().map_or(true, |p| p.dim(0) == b);
        if self.pcap < needed || !same_batch {
            let mut new_cap = if self.pcap == 0 { needed + 256 } else { self.pcap };
            while new_cap < needed {
                new_cap *= 2;
            }
            let mut np = ops::zeros::<half::bf16>(&[b, new_cap as i32, d])?;
            if let Some(old) = &self.pooled {
                let keep = self.pooled_upto as i32;
                if keep > 0 {
                    np.index_mut((.., 0..keep, ..), old.index((.., 0..keep, ..)));
                }
            }
            self.pooled = Some(np);
            self.pcap = new_cap;
        }
        let range = self.pooled_upto as i32..needed as i32;
        let stream = lisa_mlx::Stream::thread_local_or_default();
        let pooled_buf = self.pooled.as_ref().unwrap();
        if lisa_mlx::kernels::copy_rows_into(pooled, pooled_buf, self.pooled_upto as i32, &stream)
            .is_none()
        {
            self.pooled
                .as_mut()
                .unwrap()
                .index_mut((.., range, ..), pooled.clone());
        }
        self.pooled_upto = needed;
        Ok(())
    }

    /// The live pooled keys `[B, pooled_upto, D]` (a view of the buffer).
    pub fn pooled_view(&self) -> Option<Array> {
        self.pooled
            .as_ref()
            .map(|p| p.index((.., 0..self.pooled_upto as i32, ..)))
    }

    /// Roll the tape back to `offset` raw tokens (MTP verify).
    pub fn trim(&mut self, offset: usize) {
        if self.cr == 0 {
            return;
        }
        self.total = offset;
        self.pooled_upto = self.pooled_upto.min(offset / self.cr);
    }
}

/// Contiguous KV cache for one full-attention layer: `[B, kvHeads, capacity, D]`
/// buffers for K and V, grown by doubling, with an offset for the live length.
pub struct FullAttentionCache {
    /// Unused here; present so `LayerCache::ple_conv_mut` is uniform.
    pub ple_conv: Option<Array>,
    pub capture_ple_full: Option<Array>,
    pub keys: Option<Array>,
    pub values: Option<Array>,
    pub offset: usize,
    pub capacity: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub indexer_tape: IndexerTape,
    /// Whether `indexer_tape` is the full token history of this stream. False for
    /// packed/compacted caches (their tape is not maintained), so the QSA
    /// indexer is skipped and attention stays exact causal.
    pub indexer_ok: bool,
    /// Ragged batching: each stream's next absolute position. `None` = a single
    /// stream whose position is `offset`. When set, `offset` is the packed
    /// (shared) index and this carries the per-stream token positions for RoPE.
    pub next_pos: Option<Vec<usize>>,
    /// Ragged batching: the per-stream attention keep mask `[B,1,S,L]` (bool),
    /// rebuilt each step because L grows. Replaces the causal/varied mask.
    pub attn_mask: Option<Array>,
}

impl FullAttentionCache {
    pub fn new(kv_heads: usize, head_dim: usize) -> Self {
        Self {
            keys: None,
            values: None,
            offset: 0,
            capacity: 0,
            kv_heads,
            head_dim,
            indexer_tape: IndexerTape::new(),
            indexer_ok: true,
            ple_conv: None,
            capture_ple_full: None,
            next_pos: None,
            attn_mask: None,
        }
    }

    fn grow(&mut self, needed: usize, b: i32) -> lisa_mlx::error::Result<()> {
        let same_batch = self.keys.as_ref().map_or(true, |k| k.dim(0) == b);
        if self.capacity >= needed && same_batch {
            return Ok(());
        }
        let mut new_cap = if self.capacity == 0 {
            needed + 256
        } else {
            self.capacity
        };
        while new_cap < needed {
            new_cap *= 2;
        }
        let shape = &[b, self.kv_heads as i32, new_cap as i32, self.head_dim as i32];
        let mut new_keys = ops::zeros::<half::bf16>(shape)?;
        let mut new_values = ops::zeros::<half::bf16>(shape)?;
        if let (Some(old_k), Some(old_v)) = (&self.keys, &self.values) {
            let range = 0..self.offset as i32;
            let old_k = old_k.index((.., .., range.clone(), ..));
            let old_v = old_v.index((.., .., range.clone(), ..));
            new_keys.index_mut((.., .., range.clone(), ..), old_k);
            new_values.index_mut((.., .., range, ..), old_v);
        }
        self.keys = Some(new_keys);
        self.values = Some(new_values);
        self.capacity = new_cap;
        Ok(())
    }

    /// Append `keys`/`values` `[B, kvHeads, S, D]`; returns the live views
    /// `[B, kvHeads, offset', D]`.
    pub fn update(&mut self, keys: &Array, values: &Array) -> lisa_mlx::error::Result<(Array, Array)> {
        let s = keys.dim(2) as usize;
        self.grow(self.offset + s, keys.dim(0))?;
        let stream = lisa_mlx::Stream::thread_local_or_default();
        let off = self.offset as i32;
        let mut done = true;
        {
            let k = self.keys.as_ref().unwrap();
            done &= lisa_mlx::kernels::copy_rows_into(keys, k, off, &stream).is_some();
            let v = self.values.as_ref().unwrap();
            done &= lisa_mlx::kernels::copy_rows_into(values, v, off, &stream).is_some();
        }
        if !done {
            let range = self.offset as i32..(self.offset + s) as i32;
            let k = self.keys.as_mut().unwrap();
            k.index_mut((.., .., range.clone(), ..), keys.clone());
            let v = self.values.as_mut().unwrap();
            v.index_mut((.., .., range, ..), values.clone());
        }
        self.offset += s as usize;
        let live = 0..self.offset as i32;
        Ok((
            self.keys.as_ref().unwrap().index((.., .., live.clone(), ..)),
            self.values.as_ref().unwrap().index((.., .., live, ..)),
        ))
    }

    /// Roll the offset back (speculative rollback); bytes past the offset stay.
    pub fn trim(&mut self, n: usize) {
        self.offset = self.offset.saturating_sub(n);
        self.indexer_tape.trim(self.offset);
    }
}

/// Per-layer recurrent state for the gated deltanet: a short-convolution
/// state `[B, 3, 10240]` (bf16) and the fp32 SSM state `[B, 48, 128, 128]`.
pub struct GdnCache {
    pub conv: Option<Array>,
    pub ssm: Option<Array>,
    /// Speculative-verify capture (only set on a capture forward): the SSM
    /// state after EVERY position `[B*S, Hv, Dv, Dk]` and the raw conv input
    /// `[B, K-1+S, convDim]`. Used to roll the recurrent state back to the
    /// accepted prefix.
    pub capture_ssm: Option<Array>,
    pub capture_conv_input: Option<Array>,
    /// Persistent recurrence-output buffer for the capture window. The MTP
    /// verify rebuilds it every round; reusing the buffer avoids a fresh
    /// (large) Metal allocation per GDN layer per round.
    pub rec_y_buf: Option<Array>,
    /// PLE short-conv state `[B, 9, wide]`, per request (it used to live on the
    /// layer, which made it global mutable state shared by every forward).
    pub ple_conv: Option<Array>,
    /// Speculative-verify capture of the PLE conv *input* `[B, 9+S, wide]`, so
    /// the PLE state can be rolled back to the accepted prefix exactly like the
    /// GDN conv. Without this the PLE state keeps the rejected drafts and MTP
    /// drifts on free-form text.
    pub capture_ple_full: Option<Array>,
}

impl GdnCache {
    pub fn new() -> Self {
        Self {
            conv: None,
            ssm: None,
            capture_ssm: None,
            capture_conv_input: None,
            rec_y_buf: None,
            ple_conv: None,
            capture_ple_full: None,
        }
    }

    /// Roll the recurrent state back to the state after `n` positions of the
    /// captured verify window (`n >= 1`). No-op when nothing was captured.
    pub fn rollback_to(&mut self, n: usize, conv_kernel: usize) -> lisa_mlx::error::Result<()> {
        let (Some(seq), Some(ci)) = (&self.capture_ssm, &self.capture_conv_input) else {
            return Ok(());
        };
        let n = n as i32;
        self.ssm = Some(seq.index((n - 1, .., .., ..)).contiguous()?);
        let k = (conv_kernel - 1) as i32;
        self.conv = Some(ci.index((.., n..(n + k), ..)).contiguous()?);
        // Roll the PLE short-conv state back too (its own state length).
        if std::env::var("LISA_DEBUG_MTP").is_ok() {
            eprintln!(
                "[rollback] n={n} ple_full={} ple_conv={:?}",
                self.capture_ple_full.is_some(),
                self.ple_conv.as_ref().map(|a| a.shape().to_vec())
            );
        }
        if let (Some(full), Some(side)) = (self.capture_ple_full.as_ref(), self.ple_conv.as_ref()) {
            let k = side.dim(1);
            self.ple_conv = Some(full.index((.., n..(n + k), ..)).contiguous()?);
        }
        Ok(())
    }
}

impl Default for GdnCache {
    fn default() -> Self {
        Self::new()
    }
}

/// Per-layer recurrent cache.
pub enum LayerCache {
    Full(FullAttentionCache),
    Linear(GdnCache),
}

impl LayerCache {
    /// The PLE short-conv state and (capture) conv-input slots for this layer.
    #[allow(clippy::type_complexity)]
    pub fn ple_conv_mut(&mut self) -> (&mut Option<Array>, &mut Option<Array>) {
        match self {
            LayerCache::Full(f) => (&mut f.ple_conv, &mut f.capture_ple_full),
            LayerCache::Linear(g) => (&mut g.ple_conv, &mut g.capture_ple_full),
        }
    }
}
