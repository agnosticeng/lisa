//! Per-layer KV / recurrent caches. These are the interface between a model
//! and the batching runtime, so they live in `core` even though the concrete
//! variants (full attention, gated deltanet) are Qwen4-shaped for now.

use lisa_mlx::ops::indexing::{IndexMutOp, IndexOp};
use lisa_mlx::{Array, Dtype, ops};

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
        Self {
            raw: None,
            cap: 0,
            total: 0,
            pooled: None,
            pcap: 0,
            pooled_upto: 0,
            cr: 0,
        }
    }

    fn grow_raw(&mut self, needed: usize, b: i32, d: i32) -> lisa_mlx::error::Result<()> {
        let same_batch = self.raw.as_ref().map_or(true, |r| r.dim(0) == b);
        if self.cap >= needed && same_batch {
            return Ok(());
        }
        let mut new_cap = if self.cap == 0 {
            needed + 256
        } else {
            self.cap
        };
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
            let mut new_cap = if self.pcap == 0 {
                needed + 256
            } else {
                self.pcap
            };
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

    /// Host-side clone of the tape (prefix-cache entries outlive the session,
    /// so the buffers are cloned, not borrowed).
    pub fn snapshot(&self) -> TapeSnapshot {
        TapeSnapshot {
            raw: self.raw.clone(),
            total: self.total,
            pooled: self.pooled.clone(),
            pooled_upto: self.pooled_upto,
            cr: self.cr,
        }
    }

    pub fn restore(&mut self, s: &TapeSnapshot) {
        self.raw = s.raw.clone();
        self.total = s.total;
        self.pooled = s.pooled.clone();
        self.pooled_upto = s.pooled_upto;
        self.cr = s.cr;
    }
}

/// A [`IndexerTape`] clone for a prefix-cache entry.
#[derive(Clone)]
pub struct TapeSnapshot {
    raw: Option<Array>,
    total: usize,
    pooled: Option<Array>,
    pooled_upto: usize,
    cr: usize,
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
    /// packed/compacted caches (the tape is not maintained there), so the QSA
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
        let shape = &[
            b,
            self.kv_heads as i32,
            new_cap as i32,
            self.head_dim as i32,
        ];
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
    pub fn update(
        &mut self,
        keys: &Array,
        values: &Array,
    ) -> lisa_mlx::error::Result<(Array, Array)> {
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
            self.keys
                .as_ref()
                .unwrap()
                .index((.., .., live.clone(), ..)),
            self.values.as_ref().unwrap().index((.., .., live, ..)),
        ))
    }

    /// Roll the offset back (speculative rollback); bytes past the offset stay.
    pub fn trim(&mut self, n: usize) {
        self.offset = self.offset.saturating_sub(n);
        self.indexer_tape.trim(self.offset);
    }

    /// Set the live length back to `boundary` (prefix-snapshot restore); the
    /// KV bytes past it stay and are overwritten by the next append.
    pub fn restore_offset(&mut self, boundary: usize) {
        if boundary <= self.offset {
            self.indexer_tape.trim(boundary);
            self.offset = boundary;
        }
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
        if let (Some(full), Some(side)) = (self.capture_ple_full.as_ref(), self.ple_conv.as_ref()) {
            let k = side.dim(1);
            self.ple_conv = Some(full.index((.., n..(n + k), ..)).contiguous()?);
        }
        Ok(())
    }
    /// Clone the recurrent state for a prefix snapshot.
    pub fn snapshot_state(&self) -> (Option<Array>, Option<Array>, Option<Array>) {
        (self.conv.clone(), self.ssm.clone(), self.ple_conv.clone())
    }

    /// Restore the recurrent state from a prefix snapshot.
    pub fn restore_state(&mut self, s: &(Option<Array>, Option<Array>, Option<Array>)) {
        self.conv = s.0.clone();
        self.ssm = s.1.clone();
        self.ple_conv = s.2.clone();
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

    /// Clone the live state at a prefix boundary for a cross-request cache
    /// entry. Only the live KV rows are copied (the entry outlives the
    /// session's buffers); the GDN/PLE recurrent states clone the proven
    /// `snapshot_state` way.
    pub fn snapshot_prefix(&self) -> LayerState {
        match self {
            LayerCache::Full(f) => {
                let keys = f
                    .keys
                    .as_ref()
                    .map(|k| k.index((.., .., 0..f.offset as i32, ..)).contiguous().ok());
                let values = f
                    .values
                    .as_ref()
                    .map(|v| v.index((.., .., 0..f.offset as i32, ..)).contiguous().ok());
                LayerState::Full(FullPrefixState {
                    keys: keys.unwrap_or(None),
                    values: values.unwrap_or(None),
                    offset: f.offset,
                    tape: f.indexer_tape.snapshot(),
                    ple_conv: f.ple_conv.clone(),
                })
            }
            LayerCache::Linear(g) => {
                let (conv, ssm, ple_conv) = g.snapshot_state();
                LayerState::Linear(LinearPrefixState {
                    conv,
                    ssm,
                    ple_conv,
                })
            }
        }
    }

    /// Install a prefix-cache entry's state into this (fresh) cache.
    pub fn restore_prefix(&mut self, s: &LayerState) {
        match (self, s) {
            (LayerCache::Full(f), LayerState::Full(st)) => {
                f.keys = st.keys.clone();
                f.values = st.values.clone();
                f.offset = st.offset;
                f.capacity = st.keys.as_ref().map(|k| k.dim(2) as usize).unwrap_or(0);
                f.indexer_tape.restore(&st.tape);
                f.ple_conv = st.ple_conv.clone();
                // The tape is the full history of the restored prefix, so the
                // indexer stays exact-causal-correct (contract kept uniform).
                f.indexer_ok = true;
            }
            (LayerCache::Linear(g), LayerState::Linear(st)) => {
                g.restore_state(&(st.conv.clone(), st.ssm.clone(), st.ple_conv.clone()))
            }
            _ => {}
        }
    }
}

/// Cross-request prefix state for one full-attention layer.
#[derive(Clone)]
pub struct FullPrefixState {
    keys: Option<Array>,
    values: Option<Array>,
    offset: usize,
    tape: TapeSnapshot,
    ple_conv: Option<Array>,
}

/// Cross-request prefix state for one GDN layer.
#[derive(Clone)]
pub struct LinearPrefixState {
    conv: Option<Array>,
    ssm: Option<Array>,
    ple_conv: Option<Array>,
}

/// Cross-request prefix state for one layer (see [`LayerCache`]).
#[derive(Clone)]
pub enum LayerState {
    Full(FullPrefixState),
    Linear(LinearPrefixState),
}

// --- SSD-spill codec (core/prefix_cache.rs) ---------------------------------
// A `LayerState` serializes to host bytes: one u8 presence tag per array,
// then dtype tag + shape + raw LE payload. `decode` rebuilds the arrays as
// host-backed (`Array::from_raw_data`); MLX copies them to the GPU on first
// use, exactly like the safetensors-restore path in the reference spill
// design. Keyed by a content hash of the entry's tokens in prefix_cache.rs.

fn dtype_tag(d: Dtype) -> u8 {
    match d {
        Dtype::Bool => 1,
        Dtype::Uint8 => 2,
        Dtype::Uint16 => 3,
        Dtype::Uint32 => 4,
        Dtype::Int8 => 5,
        Dtype::Int16 => 6,
        Dtype::Int32 => 7,
        Dtype::Int64 => 8,
        Dtype::Float16 => 9,
        Dtype::Float32 => 10,
        Dtype::Float64 => 11,
        Dtype::Bfloat16 => 12,
        _ => 0,
    }
}

fn dtype_from_tag(t: u8) -> Option<Dtype> {
    Some(match t {
        1 => Dtype::Bool,
        2 => Dtype::Uint8,
        3 => Dtype::Uint16,
        4 => Dtype::Uint32,
        5 => Dtype::Int8,
        6 => Dtype::Int16,
        7 => Dtype::Int32,
        8 => Dtype::Int64,
        9 => Dtype::Float16,
        10 => Dtype::Float32,
        11 => Dtype::Float64,
        12 => Dtype::Bfloat16,
        _ => return None,
    })
}

macro_rules! enc_payload {
    ($a:expr, $out:expr) => {{
        $a.eval()?;
        macro_rules! one {
            ($t:ty) => {{
                let s = $a.as_slice::<$t>();
                let n = s.len() * std::mem::size_of::<$t>();
                $out.extend((n as u64).to_le_bytes());
                let ptr = s.as_ptr() as *const u8;
                $out.extend_from_slice(unsafe { std::slice::from_raw_parts(ptr, n) });
            }};
        }
        match $a.dtype() {
            Dtype::Bfloat16 => one!(half::bf16),
            Dtype::Float16 => one!(half::f16),
            Dtype::Float32 => one!(f32),
            Dtype::Float64 => one!(f64),
            Dtype::Bool | Dtype::Uint8 => one!(u8),
            Dtype::Uint16 => one!(u16),
            Dtype::Uint32 => one!(u32),
            Dtype::Int8 => one!(i8),
            Dtype::Int16 => one!(i16),
            Dtype::Int32 => one!(i32),
            Dtype::Int64 => one!(i64),
            other => anyhow::bail!("prefix spill: unsupported dtype {other:?}"),
        }
    }};
}

fn enc_arr(out: &mut Vec<u8>, a: &Option<Array>) -> anyhow::Result<()> {
    let Some(a) = a else {
        out.push(0);
        return Ok(());
    };
    out.push(1);
    out.push(dtype_tag(a.dtype()));
    let shape = a.shape();
    out.extend((shape.len() as u32).to_le_bytes());
    for &d in shape {
        out.extend(d.to_le_bytes());
    }
    enc_payload!(a, out);
    Ok(())
}

fn take_bytes<'a>(cur: &mut &'a [u8], n: usize) -> anyhow::Result<&'a [u8]> {
    if cur.len() < n {
        anyhow::bail!("prefix spill: truncated record");
    }
    let (h, t) = cur.split_at(n);
    *cur = t;
    Ok(h)
}

fn rd_u8(cur: &mut &[u8]) -> anyhow::Result<u8> {
    Ok(take_bytes(cur, 1)?[0])
}

fn rd_u32(cur: &mut &[u8]) -> anyhow::Result<u32> {
    Ok(u32::from_le_bytes(take_bytes(cur, 4)?.try_into().unwrap()))
}

fn rd_u64(cur: &mut &[u8]) -> anyhow::Result<u64> {
    Ok(u64::from_le_bytes(take_bytes(cur, 8)?.try_into().unwrap()))
}

fn dec_arr(cur: &mut &[u8]) -> anyhow::Result<Option<Array>> {
    if rd_u8(cur)? == 0 {
        return Ok(None);
    }
    let dtype = dtype_from_tag(rd_u8(cur)?)
        .ok_or_else(|| anyhow::anyhow!("prefix spill: bad dtype tag"))?;
    let ndim = rd_u32(cur)? as usize;
    let mut shape = Vec::with_capacity(ndim);
    for _ in 0..ndim {
        shape.push(rd_u32(cur)?);
    }
    let n: usize = shape.iter().map(|&d| d as usize).product();
    let nbytes = rd_u64(cur)? as usize;
    let bytes = take_bytes(cur, nbytes)?.to_vec();
    if n * dtype.size_of() != nbytes {
        anyhow::bail!("prefix spill: payload size mismatch");
    }
    let shape_i: Vec<i32> = shape.iter().map(|&d| d as i32).collect();
    let a = unsafe { Array::from_raw_data(bytes.as_ptr() as *const std::ffi::c_void, &shape_i, dtype) };
    Ok(Some(a))
}

impl LayerState {
    /// Host-byte encoding for the SSD spill tier. Eval + readback of every
    /// array: the spill runs on eviction only, off the serve hot path.
    pub fn encode(&self) -> anyhow::Result<Vec<u8>> {
        let mut out = Vec::new();
        match self {
            LayerState::Full(s) => {
                out.push(0);
                out.extend((s.offset as u64).to_le_bytes());
                enc_arr(&mut out, &s.keys)?;
                enc_arr(&mut out, &s.values)?;
                enc_arr(&mut out, &s.tape.raw)?;
                out.extend((s.tape.total as u64).to_le_bytes());
                enc_arr(&mut out, &s.tape.pooled)?;
                out.extend((s.tape.pooled_upto as u64).to_le_bytes());
                out.extend((s.tape.cr as u64).to_le_bytes());
                enc_arr(&mut out, &s.ple_conv)?;
            }
            LayerState::Linear(s) => {
                out.push(1);
                enc_arr(&mut out, &s.conv)?;
                enc_arr(&mut out, &s.ssm)?;
                enc_arr(&mut out, &s.ple_conv)?;
            }
        }
        Ok(out)
    }

    /// Rebuild a `LayerState` from [`LayerState::encode`] bytes, advancing
    /// `cur` past the consumed record (layers are variable-length, so the
    /// spill reader decodes in sequence off one flat buffer).
    pub fn decode(cur: &mut &[u8]) -> anyhow::Result<LayerState> {
        let kind = if cur.is_empty() {
            anyhow::bail!("prefix spill: empty layer record")
        } else {
            let k = cur[0];
            *cur = &cur[1..];
            k
        };
        Ok(match kind {
            0 => {
                let offset = rd_u64(cur)? as usize;
                let keys = dec_arr(cur)?;
                let values = dec_arr(cur)?;
                let tape_raw = dec_arr(cur)?;
                let total = rd_u64(cur)? as usize;
                let pooled = dec_arr(cur)?;
                let pooled_upto = rd_u64(cur)? as usize;
                let cr = rd_u64(cur)? as usize;
                let ple_conv = dec_arr(cur)?;
                LayerState::Full(FullPrefixState {
                    keys,
                    values,
                    offset,
                    tape: TapeSnapshot {
                        raw: tape_raw,
                        total,
                        pooled,
                        pooled_upto,
                        cr,
                    },
                    ple_conv,
                })
            }
            1 => {
                let conv = dec_arr(cur)?;
                let ssm = dec_arr(cur)?;
                let ple_conv = dec_arr(cur)?;
                LayerState::Linear(LinearPrefixState { conv, ssm, ple_conv })
            }
            k => anyhow::bail!("prefix spill: bad layer kind {k}"),
        })
    }
}

#[cfg(test)]
mod spill_codec_tests {
    use super::*;

    /// The spill codec round-trips a Full layer state: shapes, dtypes and
    /// offset/tape scalars survive; the restored arrays are host-backed and
    /// carry the same values.
    #[test]
    fn layer_state_codec_round_trips() -> anyhow::Result<()> {
        let keys: Option<Array> = Some(Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[1, 2, 1, 2]));
        let values = Some(Array::from_slice(
            &[
                half::bf16::from_f32(0.5),
                half::bf16::from_f32(0.25),
                half::bf16::from_f32(1.0),
                half::bf16::from_f32(2.0),
            ],
            &[1, 1, 2, 2],
        ));
        let tape_raw = Some(Array::from_slice(&[7u32, 8, 9], &[3]));
        let st = LayerState::Full(FullPrefixState {
            keys,
            values,
            offset: 2,
            tape: TapeSnapshot {
                raw: tape_raw,
                total: 3,
                pooled: None,
                pooled_upto: 1,
                cr: 4,
            },
            ple_conv: None,
        });
        let bytes = st.encode()?;
        let mut cur: &[u8] = &bytes;
        let back = LayerState::decode(&mut cur)?;
        assert!(cur.is_empty(), "one record consumes the whole buffer");
        let LayerState::Full(b) = back else {
            anyhow::bail!("kind mismatch");
        };
        assert_eq!(b.offset, 2);
        assert_eq!(b.tape.total, 3);
        assert_eq!(b.tape.pooled_upto, 1);
        assert_eq!(b.tape.cr, 4);
        let k = b.keys.as_ref().expect("keys");
        assert_eq!(k.shape(), &[1, 2, 1, 2]);
        assert_eq!(k.as_slice::<f32>(), &[1.0, 2.0, 3.0, 4.0]);
        let t = b.tape.raw.as_ref().expect("tape");
        assert_eq!(t.as_slice::<u32>(), &[7, 8, 9]);
        let v = b.values.as_ref().expect("values");
        assert_eq!(v.dtype(), Dtype::Bfloat16);
        Ok(())
    }
}
