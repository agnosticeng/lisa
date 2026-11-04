//! Per-layer KV / recurrent caches. These are the interface between a model
//! and the batching runtime, so they live in `core` even though the concrete
//! variants (full attention, gated deltanet) are Qwen4-shaped for now.

use lisa_mlx::ops::indexing::{IndexMutOp, IndexOp};
use lisa_mlx::{Array, Dtype, ops};

/// Model-owned session state for architectures whose cache cannot be described
/// as independent attention/GDN layers. Snapshots share immutable ownership;
/// the model obtains a private GPU copy before its next in-place write.
pub trait NativeState: std::any::Any + Send + Sync {
    fn as_any(&self) -> &dyn std::any::Any;
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any;
    fn copy_for_write(&self) -> anyhow::Result<std::sync::Arc<dyn NativeState>>;
    fn position(&self) -> usize;
}

#[derive(Clone)]
pub struct NativeCache {
    state: std::sync::Arc<dyn NativeState>,
    ple_conv: Option<Array>,
    capture_ple_full: Option<Array>,
}

impl NativeCache {
    pub fn new(state: impl NativeState) -> Self {
        Self {
            state: std::sync::Arc::new(state),
            ple_conv: None,
            capture_ple_full: None,
        }
    }
    pub fn position(&self) -> usize {
        self.state.position()
    }
    pub fn get<T: NativeState>(&self) -> anyhow::Result<&T> {
        self.state.as_any().downcast_ref().ok_or_else(|| {
            anyhow::anyhow!("native cache belongs to a different model architecture")
        })
    }
    pub fn get_mut<T: NativeState>(&mut self) -> anyhow::Result<&mut T> {
        anyhow::ensure!(
            self.state.as_any().is::<T>(),
            "native cache belongs to a different model architecture"
        );
        if std::sync::Arc::get_mut(&mut self.state).is_none() {
            self.state = self.state.copy_for_write()?;
        }
        std::sync::Arc::get_mut(&mut self.state)
            .ok_or_else(|| anyhow::anyhow!("native cache copy must return independent ownership"))?
            .as_any_mut()
            .downcast_mut()
            .ok_or_else(|| {
                anyhow::anyhow!("native cache belongs to a different model architecture")
            })
    }
}

pub struct IndexerTape {
    /// Raw per-token features `[B, cap, D]`, appended in place (doubling), so
    /// an append is O(s) rather than a whole-tape copy. GLM stores its raw
    /// indexer key and pool gate here; compact MLA latents have a separate F16
    /// buffer below. `raw_start` is the absolute token represented by row zero.
    raw: Option<Array>,
    cap: usize,
    raw_start: usize,
    /// Total raw tokens appended.
    pub total: usize,
    /// Normalized + roped pooled keys `[B, pcap, D]`, written in place.
    pooled: Option<Array>,
    pcap: usize,
    /// Number of blocks already pooled. Pooling is lazy: nothing is pooled
    /// until the visible context exceeds the budget.
    pub pooled_upto: usize,
    cr: usize,
    /// Optional compact GLM MLA latent cache `[B, cap, D]` in IEEE F16.
    compact: Option<Array>,
    compact_cap: usize,
    compact_total: usize,
}

impl IndexerTape {
    pub fn new() -> Self {
        Self {
            raw: None,
            cap: 0,
            raw_start: 0,
            total: 0,
            pooled: None,
            pcap: 0,
            pooled_upto: 0,
            cr: 0,
            compact: None,
            compact_cap: 0,
            compact_total: 0,
        }
    }

    fn grow_raw(
        &mut self,
        needed: usize,
        b: i32,
        d: i32,
        dtype: Dtype,
    ) -> lisa_mlx::error::Result<()> {
        let same_batch = self.raw.as_ref().map_or(true, |r| r.dim(0) == b);
        let needed_rows = needed.saturating_sub(self.raw_start);
        if self.cap >= needed_rows && same_batch {
            return Ok(());
        }
        let mut new_cap = if self.cap == 0 {
            needed_rows + 256
        } else {
            self.cap
        };
        while new_cap < needed_rows {
            new_cap *= 2;
        }
        let mut np = match dtype {
            Dtype::Float32 => ops::zeros::<f32>(&[b, new_cap as i32, d])?,
            Dtype::Float16 => ops::zeros::<half::f16>(&[b, new_cap as i32, d])?,
            _ => ops::zeros::<half::bf16>(&[b, new_cap as i32, d])?,
        };
        if let Some(old) = &self.raw {
            let keep = self.total.saturating_sub(self.raw_start) as i32;
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
        self.grow_raw(self.total + s as usize, b, d, raw_k.dtype())?;
        let stream = lisa_mlx::Stream::thread_local_or_default();
        let raw = self.raw.as_ref().unwrap();
        let write_at = self.total.saturating_sub(self.raw_start);
        if lisa_mlx::kernels::copy_rows_into(raw_k, raw, write_at as i32, &stream).is_none() {
            let range = write_at as i32..(write_at + s as usize) as i32;
            self.raw
                .as_mut()
                .unwrap()
                .index_mut((.., range, ..), raw_k.clone());
        }
        self.total += s as usize;
        Ok(())
    }

    /// Append two same-position feature arrays into adjacent fields of the raw
    /// tape without first allocating their concatenation.
    pub fn append_fields(
        &mut self,
        first: &Array,
        second: &Array,
        cr: usize,
    ) -> lisa_mlx::error::Result<()> {
        if first.rank() != 3
            || second.rank() != 3
            || first.dim(0) != second.dim(0)
            || first.dim(1) != second.dim(1)
            || first.dtype() != second.dtype()
        {
            return Err(lisa_mlx::error::Error::msg(
                "indexer tape fields must be rank-3 tensors with matching rows and dtypes",
            ));
        }
        self.cr = cr;
        let (batch, steps) = (first.dim(0), first.dim(1) as usize);
        let width = first.dim(2) + second.dim(2);
        self.grow_raw(self.total + steps, batch, width, first.dtype())?;
        let stream = lisa_mlx::Stream::thread_local_or_default();
        let write_at = self.total.saturating_sub(self.raw_start) as i32;
        let dst = self.raw.as_ref().expect("raw tape was grown");
        if lisa_mlx::kernels::copy_rows_concat_into(first, second, dst, write_at, &stream).is_none()
        {
            let packed = ops::concatenate(&[first, second], 2)?;
            if lisa_mlx::kernels::copy_rows_into(&packed, dst, write_at, &stream).is_none() {
                let range = write_at..(write_at + steps as i32);
                self.raw
                    .as_mut()
                    .expect("raw tape was grown")
                    .index_mut((.., range, ..), packed);
            }
        }
        self.total += steps;
        Ok(())
    }

    /// The raw features for token range `[start, end)`, `[B, end-start, D]`.
    pub fn raw_slice(&self, start: usize, end: usize) -> Option<Array> {
        if start < self.raw_start || start > end || end > self.total {
            return None;
        }
        let first = start - self.raw_start;
        let last = end - self.raw_start;
        self.raw
            .as_ref()
            .map(|r| r.index((.., first as i32..last as i32, ..)))
    }

    /// Absolute origin of the retained raw indexer rows.
    pub fn raw_start(&self) -> usize {
        self.raw_start
    }

    /// Number of live compact-cache rows.
    pub fn compact_total(&self) -> usize {
        self.compact_total
    }

    /// Pooling ratio used to build the indexer tape.
    pub fn compression_ratio(&self) -> usize {
        self.cr
    }

    /// Discard raw indexer rows older than the completed-pool rollback window.
    /// Pooled keys and compact latents remain full length; only the raw inputs
    /// needed to finish a pool after a short speculative rollback are retained.
    pub fn compact_raw_history(&mut self, rollback_rows: usize) -> lisa_mlx::error::Result<()> {
        if self.cr == 0 || self.raw.is_none() {
            return Ok(());
        }
        let desired = self.total.saturating_sub(rollback_rows);
        let group_aligned = desired / self.cr * self.cr;
        let pooled_end = self.pooled_upto.saturating_mul(self.cr);
        let new_start = group_aligned.min(pooled_end).min(self.total);
        if new_start.saturating_sub(self.raw_start) < 256 {
            return Ok(());
        }
        let retained = self.total - new_start;
        let source = self
            .raw_slice(new_start, self.total)
            .expect("raw range is present before compaction");
        let batch = source.dim(0);
        let width = source.dim(2);
        let new_cap = retained + 256;
        let mut next = match source.dtype() {
            Dtype::Float32 => ops::zeros::<f32>(&[batch, new_cap as i32, width])?,
            Dtype::Float16 => ops::zeros::<half::f16>(&[batch, new_cap as i32, width])?,
            _ => ops::zeros::<half::bf16>(&[batch, new_cap as i32, width])?,
        };
        let stream = lisa_mlx::Stream::thread_local_or_default();
        if lisa_mlx::kernels::copy_rows_into(&source, &next, 0, &stream).is_none() {
            next.index_mut((.., 0..retained as i32, ..), source);
        }
        self.raw = Some(next);
        self.raw_start = new_start;
        self.cap = new_cap;
        Ok(())
    }

    /// Append compact GLM MLA latents as F16. The indexer feature tape must
    /// already have been appended so both histories share the same positions.
    pub fn append_compact_f16(&mut self, latents: &Array) -> lisa_mlx::error::Result<()> {
        let latents = latents.as_dtype(Dtype::Float16)?;
        let (b, s, d) = (latents.dim(0), latents.dim(1), latents.dim(2));
        let needed = self.compact_total + s as usize;
        if needed != self.total {
            return Err(lisa_mlx::error::Error::msg(format!(
                "compact cache length {needed} does not match indexer tape length {}",
                self.total
            )));
        }
        let same_batch = self.compact.as_ref().map_or(true, |r| r.dim(0) == b);
        if self.compact_cap < needed || !same_batch {
            let mut new_cap = if self.compact_cap == 0 {
                needed + 256
            } else {
                self.compact_cap
            };
            while new_cap < needed {
                new_cap *= 2;
            }
            let mut next = ops::zeros::<half::f16>(&[b, new_cap as i32, d])?;
            if let Some(old) = &self.compact {
                let keep = self.compact_total as i32;
                if keep > 0 {
                    next.index_mut((.., 0..keep, ..), old.index((.., 0..keep, ..)));
                }
            }
            self.compact = Some(next);
            self.compact_cap = new_cap;
        }
        let start = self.compact_total as i32;
        let range = start..needed as i32;
        let stream = lisa_mlx::Stream::thread_local_or_default();
        let dst = self.compact.as_ref().unwrap();
        if lisa_mlx::kernels::copy_rows_into(&latents, dst, start, &stream).is_none() {
            self.compact
                .as_mut()
                .unwrap()
                .index_mut((.., range, ..), latents);
        }
        self.compact_total = needed;
        Ok(())
    }

    /// Reserve the compact F16 cache for a kernel that writes directly into
    /// its storage. The logical length is committed separately, after the
    /// matching raw indexer rows have been appended successfully.
    pub fn reserve_compact_f16_append(
        &mut self,
        batch: i32,
        steps: usize,
        width: i32,
    ) -> lisa_mlx::error::Result<(Array, usize)> {
        let needed = self
            .compact_total
            .checked_add(steps)
            .ok_or_else(|| lisa_mlx::error::Error::msg("compact cache length overflow"))?;
        if self.compact_total != self.total {
            return Err(lisa_mlx::error::Error::msg(format!(
                "compact cache length {} does not match indexer tape length {} before reserve",
                self.compact_total, self.total
            )));
        }
        if batch <= 0 || steps == 0 || width <= 0 || needed > i32::MAX as usize {
            return Err(lisa_mlx::error::Error::msg(
                "invalid compact cache append geometry",
            ));
        }
        let same_geometry = self
            .compact
            .as_ref()
            .map_or(true, |cache| cache.dim(0) == batch && cache.dim(2) == width);
        if self.compact_cap < needed || !same_geometry {
            let mut new_cap = if self.compact_cap == 0 || !same_geometry {
                needed
                    .checked_add(256)
                    .ok_or_else(|| lisa_mlx::error::Error::msg("compact cache capacity overflow"))?
            } else {
                self.compact_cap
            };
            while new_cap < needed {
                new_cap = new_cap.checked_mul(2).ok_or_else(|| {
                    lisa_mlx::error::Error::msg("compact cache capacity overflow")
                })?;
            }
            if new_cap > i32::MAX as usize {
                return Err(lisa_mlx::error::Error::msg(
                    "compact cache capacity exceeds Metal array dimensions",
                ));
            }
            let mut next = ops::zeros::<half::f16>(&[batch, new_cap as i32, width])?;
            if let Some(old) = &self.compact {
                let keep = self.compact_total as i32;
                if keep > 0 && old.dim(0) == batch && old.dim(2) == width {
                    next.index_mut((.., 0..keep, ..), old.index((.., 0..keep, ..)));
                }
            }
            self.compact = Some(next);
            self.compact_cap = new_cap;
        }
        Ok((
            self.compact
                .as_ref()
                .expect("compact cache was allocated")
                .clone(),
            self.compact_total,
        ))
    }

    /// Commit a compact-cache range written directly by a Metal kernel.
    pub fn commit_compact_f16_append(
        &mut self,
        start: usize,
        steps: usize,
    ) -> lisa_mlx::error::Result<()> {
        let has_capacity = self.compact.as_ref().is_some_and(|cache| {
            cache.dtype() == Dtype::Float16
                && cache.dim(0) > 0
                && cache.dim(1) as usize >= self.total
        });
        if start != self.compact_total
            || steps == 0
            || start.checked_add(steps) != Some(self.total)
            || self.total > self.compact_cap
            || !has_capacity
        {
            return Err(lisa_mlx::error::Error::msg(format!(
                "invalid direct compact cache commit: start={start}, steps={steps}, compact={}, tape={}, capacity={}",
                self.compact_total, self.total, self.compact_cap
            )));
        }
        self.compact_total = self.total;
        Ok(())
    }

    /// Write one compact GLM MLA latent row at its absolute token position.
    ///
    /// The GLM MTP head has a private attention cache whose first live row may
    /// be well past zero (the current prompt boundary). Keeping absolute slots
    /// lets its selected-token list match ds4's `[min_pos..=pos]` convention.
    pub fn write_compact_f16_at(
        &mut self,
        latent: &Array,
        position: usize,
    ) -> lisa_mlx::error::Result<()> {
        let latent = latent.as_dtype(Dtype::Float16)?;
        let (b, s, d) = (latent.dim(0), latent.dim(1), latent.dim(2));
        if s != 1 {
            return Err(lisa_mlx::error::Error::msg(format!(
                "compact cache positional write requires one row, got {s}"
            )));
        }
        let needed = position.saturating_add(1);
        let same_batch = self.compact.as_ref().map_or(true, |r| r.dim(0) == b);
        let same_dim = self.compact.as_ref().map_or(true, |r| r.dim(2) == d);
        if self.compact_cap < needed || !same_batch || !same_dim {
            let mut new_cap = self.compact_cap.max(256);
            while new_cap < needed {
                new_cap = new_cap.saturating_mul(2);
            }
            let mut next = ops::zeros::<half::f16>(&[b, new_cap as i32, d])?;
            if let Some(old) = &self.compact {
                let keep = self.compact_total.min(new_cap) as i32;
                if keep > 0 && same_batch && same_dim {
                    next.index_mut((.., 0..keep, ..), old.index((.., 0..keep, ..)));
                }
            }
            self.compact = Some(next);
            self.compact_cap = new_cap;
        }
        let stream = lisa_mlx::Stream::thread_local_or_default();
        let dst = self.compact.as_ref().unwrap();
        if lisa_mlx::kernels::copy_rows_into(&latent, dst, position as i32, &stream).is_none() {
            self.compact
                .as_mut()
                .unwrap()
                .index_mut((.., position as i32..needed as i32, ..), latent);
        }
        self.compact_total = self.compact_total.max(needed);
        Ok(())
    }

    /// The live compact GLM MLA latents `[B, end-start, D]` in F16.
    pub fn compact_slice(&self, start: usize, end: usize) -> Option<Array> {
        if end > self.compact_total || start > end {
            return None;
        }
        self.compact
            .as_ref()
            .map(|r| r.index((.., start as i32..end as i32, ..)))
    }

    #[cfg(test)]
    pub(crate) fn raw_live_for_test(&self) -> Option<Array> {
        self.raw_slice(self.raw_start, self.total)
    }

    #[cfg(test)]
    pub(crate) fn compact_live_for_test(&self) -> Option<Array> {
        self.compact_slice(0, self.compact_total)
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
            let mut np = match pooled.dtype() {
                Dtype::Float32 => ops::zeros::<f32>(&[b, new_cap as i32, d])?,
                Dtype::Float16 => ops::zeros::<half::f16>(&[b, new_cap as i32, d])?,
                _ => ops::zeros::<half::bf16>(&[b, new_cap as i32, d])?,
            };
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

    /// Append a GLM pooled key after the ds4-compatible F16 cache cast.
    pub fn push_pooled_f16(&mut self, pooled: &Array) -> lisa_mlx::error::Result<()> {
        self.push_pooled(&pooled.as_dtype(Dtype::Float16)?)
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
        debug_assert!(
            offset >= self.raw_start,
            "rollback exceeds retained raw indexer history"
        );
        self.total = offset;
        self.pooled_upto = self.pooled_upto.min(offset / self.cr);
        self.compact_total = self.compact_total.min(offset);
    }

    /// Host-side clone of the tape (prefix-cache entries outlive the session,
    /// so the buffers are cloned, not borrowed).
    pub fn snapshot(&self) -> TapeSnapshot {
        TapeSnapshot {
            raw: self.raw.clone(),
            total: self.total,
            raw_start: self.raw_start,
            pooled: self.pooled.clone(),
            pooled_upto: self.pooled_upto,
            cr: self.cr,
            compact: self.compact.clone(),
            compact_total: self.compact_total,
        }
    }

    pub fn restore(&mut self, s: &TapeSnapshot) {
        self.raw = s.raw.clone();
        self.cap = self.raw.as_ref().map_or(0, |a| a.dim(1) as usize);
        self.total = s.total;
        self.raw_start = s.raw_start;
        self.pooled = s.pooled.clone();
        self.pcap = self.pooled.as_ref().map_or(0, |a| a.dim(1) as usize);
        self.pooled_upto = s.pooled_upto;
        self.cr = s.cr;
        self.compact = s.compact.clone();
        self.compact_cap = self.compact.as_ref().map_or(0, |a| a.dim(1) as usize);
        self.compact_total = s.compact_total;
    }

    /// Pack same-position GLM tapes for a lockstep decode batch. Ragged tape
    /// lengths are rejected because pool boundaries and absolute raw origins
    /// are shared metadata in DS4's GLM indexer path.
    pub fn pack_equal(tapes: &[&Self]) -> lisa_mlx::error::Result<Self> {
        let Some(first) = tapes.first().copied() else {
            return Ok(Self::new());
        };
        for tape in &tapes[1..] {
            if tape.total != first.total
                || tape.raw_start != first.raw_start
                || tape.pooled_upto != first.pooled_upto
                || tape.cr != first.cr
                || tape.compact_total != first.compact_total
            {
                return Err(lisa_mlx::error::Error::msg(
                    "GLM batched indexer tapes must have equal position metadata",
                ));
            }
        }
        // Do not concatenate allocation slack. Equal-length streams can have
        // different backing capacities after prefix-cache restore or chunked
        // prefill; only their live rows have to share a packed shape.
        let raw = concat_live_tape_field(
            tapes,
            |tape| tape.raw_slice(tape.raw_start, tape.total),
            "raw",
        )?;
        let pooled = concat_live_tape_field(tapes, |tape| tape.pooled_view(), "pooled")?;
        let compact = concat_live_tape_field(
            tapes,
            |tape| tape.compact_slice(0, tape.compact_total),
            "compact",
        )?;
        Ok(Self {
            cap: raw.as_ref().map_or(0, |array| array.dim(1) as usize),
            pcap: pooled.as_ref().map_or(0, |array| array.dim(1) as usize),
            compact_cap: compact.as_ref().map_or(0, |array| array.dim(1) as usize),
            raw,
            raw_start: first.raw_start,
            total: first.total,
            pooled,
            pooled_upto: first.pooled_upto,
            cr: first.cr,
            compact,
            compact_total: first.compact_total,
        })
    }

    /// Extract one row from a packed lockstep tape without changing its
    /// absolute position or rollback metadata.
    pub fn extract_row(&self, row: usize) -> lisa_mlx::error::Result<Self> {
        let raw = extract_tape_field(self.raw.as_ref(), row, "raw")?;
        let pooled = extract_tape_field(self.pooled.as_ref(), row, "pooled")?;
        let compact = extract_tape_field(self.compact.as_ref(), row, "compact")?;
        Ok(Self {
            cap: raw.as_ref().map_or(0, |array| array.dim(1) as usize),
            pcap: pooled.as_ref().map_or(0, |array| array.dim(1) as usize),
            compact_cap: compact.as_ref().map_or(0, |array| array.dim(1) as usize),
            raw,
            raw_start: self.raw_start,
            total: self.total,
            pooled,
            pooled_upto: self.pooled_upto,
            cr: self.cr,
            compact,
            compact_total: self.compact_total,
        })
    }
}

fn concat_live_tape_field(
    tapes: &[&IndexerTape],
    get: impl Fn(&IndexerTape) -> Option<Array>,
    label: &str,
) -> lisa_mlx::error::Result<Option<Array>> {
    let mut arrays = Vec::with_capacity(tapes.len());
    for tape in tapes {
        match get(tape) {
            Some(array) => arrays.push(array),
            None if arrays.is_empty() && tapes.iter().all(|candidate| get(candidate).is_none()) => {
                return Ok(None);
            }
            None => {
                return Err(lisa_mlx::error::Error::msg(format!(
                    "GLM batched indexer {label} storage is missing from one stream",
                )));
            }
        }
    }
    let refs: Vec<&Array> = arrays.iter().collect();
    Ok(Some(ops::concatenate(&refs, 0)?))
}

fn extract_tape_field(
    array: Option<&Array>,
    row: usize,
    label: &str,
) -> lisa_mlx::error::Result<Option<Array>> {
    let Some(array) = array else {
        return Ok(None);
    };
    if array.rank() != 3 || row >= array.dim(0) as usize {
        return Err(lisa_mlx::error::Error::msg(format!(
            "GLM batched indexer {label} row {row} is outside shape {:?}",
            array.shape(),
        )));
    }
    let row_range = row as i32..(row as i32 + 1);
    Ok(Some(array.index((row_range, .., ..)).contiguous()?))
}

/// A [`IndexerTape`] clone for a prefix-cache entry.
#[derive(Clone)]
pub struct TapeSnapshot {
    raw: Option<Array>,
    total: usize,
    raw_start: usize,
    pooled: Option<Array>,
    pooled_upto: usize,
    cr: usize,
    compact: Option<Array>,
    compact_total: usize,
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

/// Per-layer recurrent state for linear-attention layers. Qwen GDN keeps its
/// model-specific conv dtype; GLM KDA stores both conv and SSM state in F32.
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
    Native(NativeCache),
}

impl LayerCache {
    /// The PLE short-conv state and (capture) conv-input slots for this layer.
    #[allow(clippy::type_complexity)]
    pub fn ple_conv_mut(&mut self) -> (&mut Option<Array>, &mut Option<Array>) {
        match self {
            LayerCache::Native(n) => (&mut n.ple_conv, &mut n.capture_ple_full),
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
            LayerCache::Native(n) => LayerState::Native(n.clone()),
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

    /// Capture the small amount of state needed to undo an append-only verify
    /// transaction. Unlike `snapshot_prefix`, this does not copy live KV rows:
    /// full-attention buffers remain in place and only their logical offsets
    /// and indexer metadata are saved. Linear-attention recurrent arrays are
    /// cloned by reference, as in prefix snapshots.
    pub fn snapshot_rollback(&self) -> LayerRollbackState {
        match self {
            LayerCache::Native(n) => LayerRollbackState::Native(n.clone()),
            LayerCache::Full(f) => LayerRollbackState::Full {
                offset: f.offset,
                tape: f.indexer_tape.snapshot(),
                ple_conv: f.ple_conv.clone(),
                next_pos: f.next_pos.clone(),
            },
            LayerCache::Linear(g) => {
                let (conv, ssm, ple_conv) = g.snapshot_state();
                LayerRollbackState::Linear {
                    conv,
                    ssm,
                    ple_conv,
                }
            }
        }
    }

    /// Restore an append-only verify transaction captured by
    /// `snapshot_rollback`. Rows written past the restored boundary remain in
    /// the backing buffers and are overwritten by the next append.
    pub fn restore_rollback(&mut self, state: &LayerRollbackState) -> lisa_mlx::error::Result<()> {
        match (self, state) {
            (LayerCache::Native(n), LayerRollbackState::Native(saved)) => {
                *n = saved.clone();
                Ok(())
            }
            (
                LayerCache::Full(f),
                LayerRollbackState::Full {
                    offset,
                    tape,
                    ple_conv,
                    next_pos,
                },
            ) => {
                if *offset > f.offset {
                    return Err(lisa_mlx::error::Error::msg(format!(
                        "full-attention rollback boundary {offset} exceeds live offset {}",
                        f.offset
                    )));
                }
                f.offset = *offset;
                f.indexer_tape.restore(tape);
                f.ple_conv = ple_conv.clone();
                f.next_pos = next_pos.clone();
                Ok(())
            }
            (
                LayerCache::Linear(g),
                LayerRollbackState::Linear {
                    conv,
                    ssm,
                    ple_conv,
                },
            ) => {
                g.restore_state(&(conv.clone(), ssm.clone(), ple_conv.clone()));
                Ok(())
            }
            _ => Err(lisa_mlx::error::Error::msg(
                "cache kind changed during verify transaction",
            )),
        }
    }

    /// Install a prefix-cache entry's state into this (fresh) cache.
    pub fn restore_prefix(&mut self, s: &LayerState) {
        match (self, s) {
            (LayerCache::Native(n), LayerState::Native(saved)) => *n = saved.clone(),
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
    Native(NativeCache),
}

/// Lightweight append-only cache checkpoint for speculative verification.
pub enum LayerRollbackState {
    Native(NativeCache),
    Full {
        offset: usize,
        tape: TapeSnapshot,
        ple_conv: Option<Array>,
        next_pos: Option<Vec<usize>>,
    },
    Linear {
        conv: Option<Array>,
        ssm: Option<Array>,
        ple_conv: Option<Array>,
    },
}

// --- SSD-spill codec (core/prefix_cache.rs) ---------------------------------
// A `LayerState` serializes to host bytes: one u8 presence tag per array,
// then dtype tag + shape + raw LE payload. `decode` rebuilds owned device
// arrays through `Array::from_raw_data`. Entries are keyed by their tokens
// and numerical law in prefix_cache.rs.

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
    let a =
        unsafe { Array::from_raw_data(bytes.as_ptr() as *const std::ffi::c_void, &shape_i, dtype) };
    Ok(Some(a))
}

impl LayerState {
    /// Host-byte encoding for the SSD spill tier. Eval + readback of every
    /// array: the spill runs on eviction only, off the serve hot path.
    pub fn encode(&self) -> anyhow::Result<Vec<u8>> {
        let mut out = Vec::new();
        match self {
            LayerState::Native(_) => {
                anyhow::bail!("this model's native cache has no SSD spill codec")
            }
            LayerState::Full(s) => {
                // Tag 3 adds the raw-tail base. Tags 0 and 2 remain readable
                // for spill records written before raw-history compaction.
                out.push(3);
                out.extend((s.offset as u64).to_le_bytes());
                enc_arr(&mut out, &s.keys)?;
                enc_arr(&mut out, &s.values)?;
                enc_arr(&mut out, &s.tape.raw)?;
                out.extend((s.tape.total as u64).to_le_bytes());
                out.extend((s.tape.raw_start as u64).to_le_bytes());
                enc_arr(&mut out, &s.tape.pooled)?;
                out.extend((s.tape.pooled_upto as u64).to_le_bytes());
                out.extend((s.tape.cr as u64).to_le_bytes());
                enc_arr(&mut out, &s.tape.compact)?;
                out.extend((s.tape.compact_total as u64).to_le_bytes());
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
            0 | 2 | 3 => {
                let has_compact = kind >= 2;
                let has_raw_start = kind >= 3;
                let offset = rd_u64(cur)? as usize;
                let keys = dec_arr(cur)?;
                let values = dec_arr(cur)?;
                let tape_raw = dec_arr(cur)?;
                let total = rd_u64(cur)? as usize;
                let raw_start = if has_raw_start {
                    rd_u64(cur)? as usize
                } else {
                    0
                };
                let pooled = dec_arr(cur)?;
                let pooled_upto = rd_u64(cur)? as usize;
                let cr = rd_u64(cur)? as usize;
                let compact = if has_compact { dec_arr(cur)? } else { None };
                let compact_total = if has_compact {
                    rd_u64(cur)? as usize
                } else {
                    0
                };
                let ple_conv = dec_arr(cur)?;
                LayerState::Full(FullPrefixState {
                    keys,
                    values,
                    offset,
                    tape: TapeSnapshot {
                        raw: tape_raw,
                        total,
                        raw_start,
                        pooled,
                        pooled_upto,
                        cr,
                        compact,
                        compact_total,
                    },
                    ple_conv,
                })
            }
            1 => {
                let conv = dec_arr(cur)?;
                let ssm = dec_arr(cur)?;
                let ple_conv = dec_arr(cur)?;
                LayerState::Linear(LinearPrefixState {
                    conv,
                    ssm,
                    ple_conv,
                })
            }
            k => anyhow::bail!("prefix spill: bad layer kind {k}"),
        })
    }
}

#[cfg(test)]
mod cache_state_tests {
    use super::*;

    #[test]
    fn indexer_tape_append_fields_matches_packed_append() -> lisa_mlx::error::Result<()> {
        let key = Array::from_slice(&[0.25f32, -1.0, 2.0, 3.5], &[1, 2, 2]);
        let gate = Array::from_slice(&[-0.5f32, 0.75], &[1, 2, 1]);
        let packed = ops::concatenate(&[&key, &gate], 2)?;
        let mut expected = IndexerTape::new();
        let mut actual = IndexerTape::new();
        expected.append(&packed, 4)?;
        actual.append_fields(&key, &gate, 4)?;
        let want = expected.raw_slice(0, 2).expect("packed raw rows");
        let got = actual.raw_slice(0, 2).expect("field raw rows");
        let want = want.to_vec1::<f32>()?;
        let got = got.to_vec1::<f32>()?;
        assert_eq!(
            want.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            got.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
        );
        Ok(())
    }

    #[test]
    fn glm_mtp_compact_cache_replaces_rejected_draft_at_same_position()
    -> lisa_mlx::error::Result<()> {
        let mut tape = IndexerTape::new();
        tape.write_compact_f16_at(&Array::from_slice(&[1.25f32], &[1, 1, 1]), 10)?;
        tape.write_compact_f16_at(&Array::from_slice(&[2.5f32], &[1, 1, 1]), 11)?;

        // A rejected proposal is retried at the same absolute target position
        // with the replacement token's latent. The later row must stay intact.
        tape.write_compact_f16_at(&Array::from_slice(&[9.75f32], &[1, 1, 1]), 10)?;
        let rows = tape.compact_slice(10, 12).expect("written MTP cache rows");
        rows.eval()?;
        assert_eq!(
            rows.reshape(&[2])?.to_vec1::<half::f16>()?,
            &[half::f16::from_f32(9.75), half::f16::from_f32(2.5)]
        );
        assert_eq!(tape.compact_total, 12);
        Ok(())
    }

    #[cfg(feature = "glm53-fixtures")]
    #[test]
    fn glm_compact_cache_matches_ds4_metal_positional_f16_writes() -> lisa_mlx::error::Result<()> {
        const ROWS: usize = 8;
        const DIM: usize = 512;
        const START: usize = 254;
        const REPLACED_ROW: usize = 3;
        let external_fixture: &[u8] =
            &crate::models::glm53::test_fixtures::read("dsa_compact_cache_ds4_metal.bin");
        assert_eq!(external_fixture.len(), ROWS * DIM * 2);

        fn row_values(row_index: usize) -> Vec<f32> {
            (0..DIM)
                .map(|i| {
                    let logical = row_index * DIM + i;
                    let centered = ((logical * 73 + 29) % 2001) as i32 - 1000;
                    centered as f32 / 997.0
                })
                .collect()
        }

        let mut tape = IndexerTape::new();
        for row in 0..ROWS {
            tape.write_compact_f16_at(
                &Array::from_slice(&row_values(row), &[1, 1, DIM as i32]),
                START + row,
            )?;
        }
        tape.write_compact_f16_at(
            &Array::from_slice(&row_values(ROWS + REPLACED_ROW), &[1, 1, DIM as i32]),
            START + REPLACED_ROW,
        )?;

        let rows = tape
            .compact_slice(START, START + ROWS)
            .expect("DS4 comparison cache rows");
        rows.eval()?;
        let actual = rows
            .reshape(&[(ROWS * DIM) as i32])?
            .to_vec1::<half::f16>()?;
        let actual_bits: Vec<u16> = actual.iter().map(|value| value.to_bits()).collect();
        let expected_bits: Vec<u16> = external_fixture
            .chunks_exact(2)
            .map(|bytes| u16::from_le_bytes([bytes[0], bytes[1]]))
            .collect();
        assert_eq!(actual_bits, expected_bits);
        Ok(())
    }

    #[test]
    fn direct_compact_append_reservation_preserves_rows_across_growth()
    -> lisa_mlx::error::Result<()> {
        const WIDTH: usize = 512;
        let mut tape = IndexerTape::new();
        let key0 = Array::from_slice(&vec![0.0f32; 257], &[1, 257, 1]);
        let gate0 = Array::from_slice(&vec![0.0f32; 257], &[1, 257, 1]);
        let (cache0, start0) = tape.reserve_compact_f16_append(1, 257, WIDTH as i32)?;
        assert_eq!(start0, 0);
        assert_eq!(cache0.dim(1), 513);
        let first = Array::from_slice(
            &vec![half::f16::from_f32(1.25); 257 * WIDTH],
            &[1, 257, WIDTH as i32],
        );
        let stream = lisa_mlx::Stream::thread_local_or_default();
        assert!(lisa_mlx::kernels::copy_rows_into(&first, &cache0, 0, &stream).is_some());
        tape.append_fields(&key0, &gate0, 1)?;
        tape.commit_compact_f16_append(start0, 257)?;

        let (cache1, start1) = tape.reserve_compact_f16_append(1, 257, WIDTH as i32)?;
        assert_eq!(start1, 257);
        assert_eq!(cache1.dim(1), 1026);
        let replacement = Array::from_slice(
            &vec![half::f16::from_f32(2.5); 257 * WIDTH],
            &[1, 257, WIDTH as i32],
        );
        assert!(
            lisa_mlx::kernels::copy_rows_into(&replacement, &cache1, start1 as i32, &stream)
                .is_some()
        );
        let key1 = Array::from_slice(&vec![0.0f32; 257], &[1, 257, 1]);
        let gate1 = Array::from_slice(&vec![0.0f32; 257], &[1, 257, 1]);
        tape.append_fields(&key1, &gate1, 1)?;
        tape.commit_compact_f16_append(start1, 257)?;

        let rows = tape
            .compact_slice(0, 514)
            .expect("reserved compact rows after growth");
        let values = rows.to_vec1::<half::f16>()?;
        assert!(
            values[..257 * WIDTH]
                .iter()
                .all(|value| *value == half::f16::from_f32(1.25))
        );
        assert!(
            values[257 * WIDTH..514 * WIDTH]
                .iter()
                .all(|value| *value == half::f16::from_f32(2.5))
        );
        Ok(())
    }

    #[test]
    fn indexer_tape_packs_and_extracts_equal_position_rows() -> lisa_mlx::error::Result<()> {
        let mut tapes = Vec::new();
        for stream in 0..2 {
            let raw: Vec<f32> = (0..8).map(|i| (stream * 100 + i) as f32).collect();
            let compact: Vec<f32> = (0..12).map(|i| (stream * 1000 + i) as f32).collect();
            let pooled = [stream as f32 + 0.5, stream as f32 + 1.5];
            let mut tape = IndexerTape::new();
            tape.append(&Array::from_slice(&raw, &[1, 4, 2]), 4)?;
            tape.append_compact_f16(&Array::from_slice(&compact, &[1, 4, 3]))?;
            tape.push_pooled_f16(&Array::from_slice(&pooled, &[1, 1, 2]))?;
            tapes.push(tape);
        }

        let packed = IndexerTape::pack_equal(&[&tapes[0], &tapes[1]])?;
        assert_eq!(packed.total, 4);
        assert_eq!(packed.raw_start, 0);
        assert_eq!(packed.pooled_upto, 1);
        assert_eq!(packed.compact_total, 4);
        assert_eq!(packed.raw.as_ref().unwrap().dim(0), 2);
        assert_eq!(packed.pooled.as_ref().unwrap().dim(0), 2);
        assert_eq!(packed.compact.as_ref().unwrap().dim(0), 2);

        for stream in 0..2 {
            let row = packed.extract_row(stream)?;
            let raw = row.raw_slice(0, 4).expect("packed raw row");
            let compact = row.compact_slice(0, 4).expect("packed compact row");
            let pooled = row.pooled_view().expect("packed pooled row");
            assert_eq!(
                raw.to_vec1::<f32>()?,
                (0..8)
                    .map(|i| (stream * 100 + i) as f32)
                    .collect::<Vec<_>>()
            );
            assert_eq!(
                compact.to_vec1::<half::f16>()?,
                (0..12)
                    .map(|i| half::f16::from_f32((stream * 1000 + i) as f32))
                    .collect::<Vec<_>>()
            );
            assert_eq!(
                pooled.to_vec1::<half::f16>()?,
                [
                    half::f16::from_f32(stream as f32 + 0.5),
                    half::f16::from_f32(stream as f32 + 1.5),
                ]
            );
        }
        Ok(())
    }

    #[test]
    fn indexer_compaction_snapshot_and_rollback_preserve_absolute_rows()
    -> lisa_mlx::error::Result<()> {
        let values: Vec<f32> = (0..600 * 2).map(|v| v as f32).collect();
        let rows = Array::from_slice(&values, &[1, 600, 2]);
        let mut tape = IndexerTape::new();
        tape.append(&rows, 4)?;
        tape.pooled_upto = 150;
        tape.compact_raw_history(32)?;

        assert_eq!(tape.total, 600);
        assert_eq!(tape.raw_start, 568);
        assert!(tape.raw_slice(0, 4).is_none());
        let tail = tape.raw_slice(568, 600).expect("retained rollback tail");
        tail.eval()?;
        let expected: Vec<f32> = values[568 * 2..].to_vec();
        assert_eq!(tail.as_slice::<f32>(), expected.as_slice());

        let snapshot = tape.snapshot();
        let mut restored = IndexerTape::new();
        restored.restore(&snapshot);
        assert_eq!(restored.raw_start, 568);
        assert_eq!(restored.total, 600);
        assert_eq!(restored.pooled_upto, 150);
        assert!(restored.raw_slice(567, 569).is_none());

        // Roll back by two complete pools, then append replacement rows. The
        // absolute origin stays fixed while the retained tail is overwritten.
        restored.trim(592);
        let replacement_values = [9000.0f32, 9001.0, 9002.0, 9003.0];
        let replacement = Array::from_slice(&replacement_values, &[1, 2, 2]);
        restored.append(&replacement, 4)?;
        let replaced = restored.raw_slice(592, 594).expect("replacement rows");
        replaced.eval()?;
        assert_eq!(replaced.as_slice::<f32>(), &replacement_values);
        assert_eq!(restored.total, 594);
        assert_eq!(restored.raw_start, 568);
        Ok(())
    }

    #[test]
    fn glm_indexer_prefix_restore_replays_partial_pool_after_rollback()
    -> lisa_mlx::error::Result<()> {
        let raw_values: Vec<f32> = (0..600).map(|v| v as f32).collect();
        let mut full = FullAttentionCache::new(1, 1);
        full.offset = 600;
        full.indexer_tape
            .append(&Array::from_slice(&raw_values, &[1, 600, 1]), 4)?;

        let compact_values: Vec<f32> = (0..600).map(|v| (v as f32) * 0.5).collect();
        full.indexer_tape
            .append_compact_f16(&Array::from_slice(&compact_values, &[1, 600, 1]))?;
        let pooled_values: Vec<f32> = (0..150).map(|v| v as f32).collect();
        full.indexer_tape
            .push_pooled_f16(&Array::from_slice(&pooled_values, &[1, 150, 1]))?;
        full.indexer_tape.compact_raw_history(32)?;
        assert_eq!(full.indexer_tape.raw_start, 568);

        let prefix = LayerCache::Full(full).snapshot_prefix();
        let mut restored = LayerCache::Full(FullAttentionCache::new(1, 1));
        restored.restore_prefix(&prefix);
        let LayerCache::Full(restored) = &mut restored else {
            unreachable!();
        };

        // Roll back into the final pool, then replace the two speculative rows.
        // The pool is rebuilt from the preserved prefix rows plus the new rows.
        restored.trim(6);
        let replacement_raw = Array::from_slice(&[900.0f32, 901.0], &[1, 2, 1]);
        restored.indexer_tape.append(&replacement_raw, 4)?;
        let replacement_compact = Array::from_slice(&[450.0f32, 450.5], &[1, 2, 1]);
        restored
            .indexer_tape
            .append_compact_f16(&replacement_compact)?;

        let pool_raw = restored
            .indexer_tape
            .raw_slice(592, 596)
            .expect("partial-pool prefix remains available after restore");
        pool_raw.eval()?;
        assert_eq!(
            pool_raw.reshape(&[4])?.to_vec1::<f32>()?,
            &[592.0, 593.0, 900.0, 901.0]
        );
        let rebuilt_pool = Array::from_slice(&[846.5f32], &[1, 1, 1]);
        restored.indexer_tape.push_pooled_f16(&rebuilt_pool)?;
        let pooled_tail = restored
            .indexer_tape
            .pooled_view()
            .expect("restored pooled-key cache")
            .index((.., 148..149, ..))
            .reshape(&[1])?;
        pooled_tail.eval()?;
        assert_eq!(
            pooled_tail.to_vec1::<half::f16>()?,
            &[half::f16::from_f32(846.5)]
        );

        let live_compact = restored
            .indexer_tape
            .compact_slice(592, 596)
            .expect("compact cache was truncated and replayed with raw rows");
        live_compact.eval()?;
        assert_eq!(
            live_compact.reshape(&[4])?.to_vec1::<half::f16>()?,
            &[
                half::f16::from_f32(296.0),
                half::f16::from_f32(296.5),
                half::f16::from_f32(450.0),
                half::f16::from_f32(450.5),
            ]
        );
        assert_eq!(restored.offset, 594);
        assert_eq!(restored.indexer_tape.total, 596);
        assert_eq!(restored.indexer_tape.pooled_upto, 149);
        assert_eq!(restored.indexer_tape.raw_start, 568);
        Ok(())
    }

    #[cfg(feature = "glm53-fixtures")]
    #[test]
    fn glm_indexer_pool_rollback_and_replay_match_ds4_incremental_cache() -> anyhow::Result<()> {
        if !lisa_mlx::device::metal_available() {
            eprintln!("[glm-indexer-pool-replay] skipping: no Metal device available");
            return Ok(());
        }
        const POOLS: usize = 2;
        const POOL_SIZE: usize = 4;
        const DIM: usize = 128;
        const ROW_VALUES: usize = POOLS * POOL_SIZE * DIM;
        const ROW_BYTES: usize = ROW_VALUES * std::mem::size_of::<f32>();
        const NORM_BYTES: usize = DIM * std::mem::size_of::<f32>();
        const APE_BYTES: usize = POOL_SIZE * DIM * std::mem::size_of::<u16>();
        const POOL_BYTES: usize = POOLS * DIM * std::mem::size_of::<u16>();
        let external_fixture: &[u8] =
            &crate::models::glm53::test_fixtures::read("indexer_pool_lifecycle_ds4.bin");
        assert_eq!(
            external_fixture.len(),
            ROW_BYTES + NORM_BYTES + ROW_BYTES + NORM_BYTES + APE_BYTES + POOL_BYTES
        );

        let decode_f32 = |bytes: &[u8]| {
            bytes
                .chunks_exact(std::mem::size_of::<f32>())
                .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("F32 fixture word")))
                .collect::<Vec<_>>()
        };
        let decode_bf16 = |bytes: &[u8]| {
            bytes
                .chunks_exact(std::mem::size_of::<u16>())
                .map(|chunk| half::bf16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]])))
                .collect::<Vec<_>>()
        };
        let expected: Vec<half::f16> = external_fixture[external_fixture.len() - POOL_BYTES..]
            .chunks_exact(std::mem::size_of::<u16>())
            .map(|chunk| half::f16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]])))
            .collect();
        let raw = decode_f32(&external_fixture[..ROW_BYTES]);
        let norm_weight = decode_f32(&external_fixture[ROW_BYTES..ROW_BYTES + NORM_BYTES]);
        let gate_start = ROW_BYTES + NORM_BYTES;
        let gate = decode_f32(&external_fixture[gate_start..gate_start + ROW_BYTES]);
        let bias_start = gate_start + ROW_BYTES;
        let norm_bias = decode_f32(&external_fixture[bias_start..bias_start + NORM_BYTES]);
        let ape_start = bias_start + NORM_BYTES;
        let ape = decode_bf16(&external_fixture[ape_start..ape_start + APE_BYTES]);

        let append_rows = |tape: &mut IndexerTape,
                           start: usize,
                           count: usize,
                           raw_rows: &[f32],
                           gate_rows: &[f32]|
         -> anyhow::Result<()> {
            let start_at = start * DIM;
            let end_at = (start + count) * DIM;
            let raw =
                Array::from_slice(&raw_rows[start_at..end_at], &[1, count as i32, DIM as i32]);
            let gate =
                Array::from_slice(&gate_rows[start_at..end_at], &[1, count as i32, DIM as i32]);
            tape.append_fields(&raw, &gate, POOL_SIZE)?;
            Ok(())
        };
        let pool_rows = |raw_rows: &[f32], gate_rows: &[f32]| -> anyhow::Result<Array> {
            let raw = Array::from_slice(raw_rows, &[1, 1, POOL_SIZE as i32, DIM as i32]);
            let gate = Array::from_slice(gate_rows, &[1, 1, POOL_SIZE as i32, DIM as i32]);
            let norm_weight = Array::from_slice(&norm_weight, &[DIM as i32]);
            let norm_bias = Array::from_slice(&norm_bias, &[DIM as i32]);
            let ape = Array::from_slice(&ape, &[POOL_SIZE as i32, DIM as i32]);
            lisa_mlx::models::glm53::kernels::indexer_pool_fused(
                &raw,
                &gate,
                &ape,
                &norm_weight,
                &norm_bias,
                1e-6,
                &lisa_mlx::Stream::thread_local_or_default(),
            )
            .ok_or_else(|| anyhow::anyhow!("DS4-shaped DSA pool kernel declined"))
        };

        let mut tape = IndexerTape::new();
        for (start, count) in [(0, 1), (1, 2), (3, 1)] {
            append_rows(&mut tape, start, count, &raw, &gate)?;
        }
        let first_pool = pool_rows(&raw[..POOL_SIZE * DIM], &gate[..POOL_SIZE * DIM])?;
        tape.push_pooled_f16(&first_pool)?;
        let checkpoint = tape.snapshot();

        let mut speculative_raw = vec![0.0f32; POOL_SIZE * DIM];
        let mut speculative_gate = vec![0.0f32; POOL_SIZE * DIM];
        for row in 0..POOL_SIZE {
            for channel in 0..DIM {
                let index = row * DIM + channel;
                let source = ((row + 1) % POOL_SIZE) * DIM + channel;
                speculative_raw[index] = raw[source] * -0.5 + (row + 2) as f32 * 0.0625;
                speculative_gate[index] = gate[source] * 0.25 + (row + 1) as f32 * 0.125;
            }
        }
        let mut speculative_full_raw = vec![0.0f32; ROW_VALUES];
        let mut speculative_full_gate = vec![0.0f32; ROW_VALUES];
        speculative_full_raw[POOL_SIZE * DIM..].copy_from_slice(&speculative_raw);
        speculative_full_gate[POOL_SIZE * DIM..].copy_from_slice(&speculative_gate);
        append_rows(
            &mut tape,
            4,
            2,
            &speculative_full_raw,
            &speculative_full_gate,
        )?;
        append_rows(
            &mut tape,
            6,
            2,
            &speculative_full_raw,
            &speculative_full_gate,
        )?;
        let speculative_pool = pool_rows(&speculative_raw, &speculative_gate)?;
        let speculative_bits: Vec<u16> = speculative_pool
            .to_vec1::<half::f16>()?
            .iter()
            .map(|value| value.to_bits())
            .collect();
        tape.push_pooled_f16(&speculative_pool)?;
        tape.trim(4);
        assert_eq!(tape.total, 4);
        assert_eq!(tape.pooled_upto, 1);

        for (start, count) in [(4, 1), (5, 2), (7, 1)] {
            append_rows(&mut tape, start, count, &raw, &gate)?;
        }
        let replayed_pool = pool_rows(&raw[POOL_SIZE * DIM..], &gate[POOL_SIZE * DIM..])?;
        tape.push_pooled_f16(&replayed_pool)?;
        let actual: Vec<u16> = tape
            .pooled_view()
            .expect("replayed pooled cache")
            .to_vec1::<half::f16>()?
            .iter()
            .map(|value| value.to_bits())
            .collect();
        let expected_bits: Vec<u16> = expected.iter().map(|value| value.to_bits()).collect();
        anyhow::ensure!(
            actual == expected_bits,
            "LISA pooled cache after rollback/replay differs from DS4 incremental Metal"
        );
        anyhow::ensure!(
            speculative_bits != expected_bits[DIM..],
            "speculative pool fixture unexpectedly equals the replacement replay"
        );
        anyhow::ensure!(
            tape.total == 8 && tape.pooled_upto == 2 && tape.raw_start == 0,
            "unexpected live DSA tape state after replay"
        );
        anyhow::ensure!(checkpoint.total == 4 && checkpoint.pooled_upto == 1);
        eprintln!(
            "[glm-indexer-pool-replay] all {} F16 cache values match DS4 after speculative completion, rollback, and replacement replay",
            actual.len()
        );
        Ok(())
    }

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
                raw_start: 0,
                pooled: None,
                pooled_upto: 1,
                cr: 4,
                compact: None,
                compact_total: 0,
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

    #[test]
    fn glm_compacted_tape_spill_restore_preserves_live_absolute_rows() -> anyhow::Result<()> {
        let raw_values: Vec<f32> = (0..600 * 2).map(|v| v as f32 * 0.25).collect();
        let compact_values: Vec<f32> = (0..600 * 2).map(|v| v as f32 * 0.5).collect();
        let pooled_values: Vec<f32> = (0..150 * 2).map(|v| v as f32 * 0.125).collect();

        let mut full = FullAttentionCache::new(1, 2);
        full.offset = 600;
        full.indexer_tape
            .append(&Array::from_slice(&raw_values, &[1, 600, 2]), 4)?;
        full.indexer_tape
            .append_compact_f16(&Array::from_slice(&compact_values, &[1, 600, 2]))?;
        full.indexer_tape
            .push_pooled_f16(&Array::from_slice(&pooled_values, &[1, 150, 2]))?;
        full.indexer_tape.compact_raw_history(32)?;
        assert_eq!(full.indexer_tape.raw_start, 568);

        // Exercise the exact layer-state codec used by SSD prefix spill, then
        // restore into a fresh GLM full-attention cache.
        let state = LayerCache::Full(full).snapshot_prefix();
        let bytes = state.encode()?;
        let mut cursor: &[u8] = &bytes;
        let restored_state = LayerState::decode(&mut cursor)?;
        assert!(cursor.is_empty());
        let mut restored = LayerCache::Full(FullAttentionCache::new(1, 2));
        restored.restore_prefix(&restored_state);
        let LayerCache::Full(restored) = &mut restored else {
            anyhow::bail!("decoded cache kind mismatch");
        };

        assert_eq!(restored.offset, 600);
        assert_eq!(restored.indexer_tape.raw_start, 568);
        assert_eq!(restored.indexer_tape.total, 600);
        assert_eq!(restored.indexer_tape.pooled_upto, 150);
        let raw = restored
            .indexer_tape
            .raw_slice(568, 600)
            .expect("retained raw rollback window");
        raw.eval()?;
        assert_eq!(
            raw.reshape(&[64])?.to_vec1::<f32>()?,
            &raw_values[568 * 2..]
        );
        let compact = restored
            .indexer_tape
            .compact_slice(568, 600)
            .expect("compact latent history");
        compact.eval()?;
        let expected_compact: Vec<half::f16> = compact_values[568 * 2..]
            .iter()
            .map(|&value| half::f16::from_f32(value))
            .collect();
        assert_eq!(
            compact.reshape(&[64])?.to_vec1::<half::f16>()?,
            expected_compact
        );
        let pooled = restored
            .indexer_tape
            .pooled_view()
            .expect("pooled key history")
            .index((.., 149..150, ..));
        pooled.eval()?;
        let expected_pooled: Vec<half::f16> = pooled_values[149 * 2..150 * 2]
            .iter()
            .map(|&value| half::f16::from_f32(value))
            .collect();
        assert_eq!(
            pooled.reshape(&[2])?.to_vec1::<half::f16>()?,
            expected_pooled
        );

        // Roll back across the final pool and replay replacement rows after
        // spill restore; absolute positions and the retained pool prefix must
        // remain valid.
        restored.trim(6);
        let replacement_raw = Array::from_slice(&[9000.0f32, 9001.0, 9002.0, 9003.0], &[1, 2, 2]);
        restored.indexer_tape.append(&replacement_raw, 4)?;
        let replacement_compact =
            Array::from_slice(&[9100.0f32, 9101.0, 9102.0, 9103.0], &[1, 2, 2]);
        restored
            .indexer_tape
            .append_compact_f16(&replacement_compact)?;
        let replayed_raw = restored
            .indexer_tape
            .raw_slice(594, 596)
            .expect("replacement raw rows");
        replayed_raw.eval()?;
        assert_eq!(
            replayed_raw.reshape(&[4])?.to_vec1::<f32>()?,
            &[9000.0, 9001.0, 9002.0, 9003.0]
        );
        let replayed_compact = restored
            .indexer_tape
            .compact_slice(594, 596)
            .expect("replacement compact rows");
        replayed_compact.eval()?;
        assert_eq!(
            replayed_compact.reshape(&[4])?.to_vec1::<half::f16>()?,
            &[
                half::f16::from_f32(9100.0),
                half::f16::from_f32(9101.0),
                half::f16::from_f32(9102.0),
                half::f16::from_f32(9103.0),
            ]
        );
        assert_eq!(restored.indexer_tape.raw_start, 568);
        assert_eq!(restored.indexer_tape.total, 596);
        assert_eq!(restored.indexer_tape.pooled_upto, 148);
        Ok(())
    }
}
