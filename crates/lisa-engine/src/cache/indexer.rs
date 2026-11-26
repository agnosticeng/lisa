//! Engine-owned indexer history, compaction, batching and rollback.
use lisa_backend::{Device, Kernel, ScalarType as Dtype, TensorAccess, TensorOperation};
pub trait CacheDevice:
    Device<Tensor: TensorAccess + 'static, KernelHandle: 'static, Stream: 'static> + Clone + 'static
{
}
impl<D> CacheDevice for D
where
    D: Device + Clone + 'static,
    D::Tensor: TensorAccess + 'static,
    D::KernelHandle: 'static,
    D::Stream: 'static,
{
}

pub struct IndexerTape<D: CacheDevice> {
    pub(crate) context: crate::execution::DeviceContext<D>,
    device: D,
    stream: D::Stream,
    /// Raw per-token features `[B, cap, D]`, appended in place (doubling), so
    /// an append is O(s) rather than a whole-tape copy. GLM stores its raw
    /// indexer key and pool gate here; compact MLA latents have a separate F16
    /// buffer below. `raw_start` is the absolute token represented by row zero.
    raw: Option<D::Tensor>,
    cap: usize,
    raw_start: usize,
    /// Total raw tokens appended.
    pub total: usize,
    /// Normalized + roped pooled keys `[B, pcap, D]`, written in place.
    pooled: Option<D::Tensor>,
    pcap: usize,
    /// Number of blocks already pooled. Pooling is lazy: nothing is pooled
    /// until the visible context exceeds the budget.
    pub pooled_upto: usize,
    cr: usize,
    /// Optional compact GLM MLA latent cache `[B, cap, D]` in IEEE F16.
    compact: Option<D::Tensor>,
    compact_cap: usize,
    compact_total: usize,
}

impl<D: CacheDevice> IndexerTape<D>
where
    D::Tensor: TensorAccess,
{
    pub fn new(context: crate::execution::DeviceContext<D>) -> Self {
        let device = context.device().clone();
        Self {
            context,
            stream: device.stream(),
            device,
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

    fn grow_raw(&mut self, needed: usize, b: i32, d: i32, dtype: Dtype) -> anyhow::Result<()> {
        let same_batch = self.raw.as_ref().is_none_or(|r| r.dim(0) == b);
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
            Dtype::F32 => self.zeros(&[b, new_cap as i32, d], Dtype::F32)?,
            Dtype::F16 => self.zeros(&[b, new_cap as i32, d], Dtype::F16)?,
            _ => self.zeros(&[b, new_cap as i32, d], Dtype::BF16)?,
        };
        if let Some(old) = &self.raw {
            let keep = self.total.saturating_sub(self.raw_start) as i32;
            if keep > 0 {
                self.assign(&mut np, 0..keep, &old.slice(1, 0, Some(keep))?)?;
            }
        }
        self.raw = Some(np);
        self.cap = new_cap;
        Ok(())
    }

    /// Append `raw_k` `[B, s, D]` in place.
    pub fn append(&mut self, raw_k: &D::Tensor, cr: usize) -> anyhow::Result<()> {
        self.cr = cr;
        let (b, s, d) = (raw_k.dim(0), raw_k.dim(1), raw_k.dim(2));
        self.grow_raw(
            self.total + s as usize,
            b,
            d,
            self.device.dtype(raw_k).expect("typed cache tensor"),
        )?;

        let raw = self.raw.as_ref().unwrap();
        let write_at = self.total.saturating_sub(self.raw_start);
        if !self.copy_rows(raw_k, raw, write_at as i32)? {
            let range = write_at as i32..(write_at + s as usize) as i32;
            self.raw = Some(self.assigned(self.raw.as_ref().unwrap(), range, &raw_k.clone())?);
        }
        self.total += s as usize;
        Ok(())
    }

    /// Append two same-position feature arrays into adjacent fields of the raw
    /// tape without first allocating their concatenation.
    pub fn append_fields(
        &mut self,
        first: &D::Tensor,
        second: &D::Tensor,
        cr: usize,
    ) -> anyhow::Result<()> {
        if first.rank() != 3
            || second.rank() != 3
            || first.dim(0) != second.dim(0)
            || first.dim(1) != second.dim(1)
            || self.device.dtype(first).expect("typed cache tensor")
                != self.device.dtype(second).expect("typed cache tensor")
        {
            return Err(anyhow::Error::msg(
                "indexer tape fields must be rank-3 tensors with matching rows and dtypes",
            ));
        }
        self.cr = cr;
        let (batch, steps) = (first.dim(0), first.dim(1) as usize);
        let width = first.dim(2) + second.dim(2);
        self.grow_raw(
            self.total + steps,
            batch,
            width,
            self.device.dtype(first).expect("typed cache tensor"),
        )?;

        let write_at = self.total.saturating_sub(self.raw_start) as i32;
        let dst = self.raw.as_ref().expect("raw tape was grown");
        if !self.copy_fields(first, second, dst, write_at)? {
            let packed = D::Tensor::concatenate(&[first, second], 2)?;
            if !self.copy_rows(&packed, dst, write_at)? {
                let range = write_at..(write_at + steps as i32);
                self.raw = Some(self.assigned(self.raw.as_ref().unwrap(), range, &packed)?);
            }
        }
        self.total += steps;
        Ok(())
    }

    /// The raw features for token range `[start, end)`, `[B, end-start, D]`.
    pub fn raw_slice(&self, start: usize, end: usize) -> Option<D::Tensor> {
        if start < self.raw_start || start > end || end > self.total {
            return None;
        }
        let first = start - self.raw_start;
        let last = end - self.raw_start;
        self.raw.as_ref().map(|r| {
            r.slice(1, first as i32, Some(last as i32))
                .expect("validated tape range")
        })
    }

    /// Absolute origin of the retained raw indexer rows.
    pub fn raw_start(&self) -> usize {
        self.raw_start
    }

    /// Number of live compact-cache rows.
    pub fn compact_total(&self) -> usize {
        self.compact_total
    }

    /// Discard raw indexer rows older than the completed-pool rollback window.
    /// Pooled keys and compact latents remain full length; only the raw inputs
    /// needed to finish a pool after a short speculative rollback are retained.
    pub fn compact_raw_history(&mut self, rollback_rows: usize) -> anyhow::Result<()> {
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
        let mut next = match self.device.dtype(&source).expect("typed cache tensor") {
            Dtype::F32 => self.zeros(&[batch, new_cap as i32, width], Dtype::F32)?,
            Dtype::F16 => self.zeros(&[batch, new_cap as i32, width], Dtype::F16)?,
            _ => self.zeros(&[batch, new_cap as i32, width], Dtype::BF16)?,
        };

        if !self.copy_rows(&source, &next, 0)? {
            self.assign(&mut next, 0..retained as i32, &source)?;
        }
        self.raw = Some(next);
        self.raw_start = new_start;
        self.cap = new_cap;
        Ok(())
    }

    /// Append compact GLM MLA latents as F16. The indexer feature tape must
    /// already have been appended so both histories share the same positions.
    pub fn append_compact_f16(&mut self, latents: &D::Tensor) -> anyhow::Result<()> {
        let latents = self.cast(latents, Dtype::F16)?;
        let (b, s, d) = (latents.dim(0), latents.dim(1), latents.dim(2));
        let needed = self.compact_total + s as usize;
        if needed != self.total {
            return Err(anyhow::Error::msg(format!(
                "compact cache length {needed} does not match indexer tape length {}",
                self.total
            )));
        }
        let same_batch = self.compact.as_ref().is_none_or(|r| r.dim(0) == b);
        if self.compact_cap < needed || !same_batch {
            let mut new_cap = if self.compact_cap == 0 {
                needed + 256
            } else {
                self.compact_cap
            };
            while new_cap < needed {
                new_cap *= 2;
            }
            let mut next = self.zeros(&[b, new_cap as i32, d], Dtype::F16)?;
            if let Some(old) = &self.compact {
                let keep = self.compact_total as i32;
                if keep > 0 {
                    self.assign(&mut next, 0..keep, &old.slice(1, 0, Some(keep))?)?;
                }
            }
            self.compact = Some(next);
            self.compact_cap = new_cap;
        }
        let start = self.compact_total as i32;
        let range = start..needed as i32;

        let dst = self.compact.as_ref().unwrap();
        if !self.copy_rows(&latents, dst, start)? {
            self.compact = Some(self.assigned(self.compact.as_ref().unwrap(), range, &latents)?);
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
    ) -> anyhow::Result<(D::Tensor, usize)> {
        let needed = self
            .compact_total
            .checked_add(steps)
            .ok_or_else(|| anyhow::Error::msg("compact cache length overflow"))?;
        if self.compact_total != self.total {
            return Err(anyhow::Error::msg(format!(
                "compact cache length {} does not match indexer tape length {} before reserve",
                self.compact_total, self.total
            )));
        }
        if batch <= 0 || steps == 0 || width <= 0 || needed > i32::MAX as usize {
            return Err(anyhow::Error::msg("invalid compact cache append geometry"));
        }
        let same_geometry = self
            .compact
            .as_ref()
            .is_none_or(|cache| cache.dim(0) == batch && cache.dim(2) == width);
        if self.compact_cap < needed || !same_geometry {
            let mut new_cap = if self.compact_cap == 0 || !same_geometry {
                needed
                    .checked_add(256)
                    .ok_or_else(|| anyhow::Error::msg("compact cache capacity overflow"))?
            } else {
                self.compact_cap
            };
            while new_cap < needed {
                new_cap = new_cap
                    .checked_mul(2)
                    .ok_or_else(|| anyhow::Error::msg("compact cache capacity overflow"))?;
            }
            if new_cap > i32::MAX as usize {
                return Err(anyhow::Error::msg(
                    "compact cache capacity exceeds device tensor dimensions",
                ));
            }
            let mut next = self.zeros(&[batch, new_cap as i32, width], Dtype::F16)?;
            if let Some(old) = &self.compact {
                let keep = self.compact_total as i32;
                if keep > 0 && old.dim(0) == batch && old.dim(2) == width {
                    self.assign(&mut next, 0..keep, &old.slice(1, 0, Some(keep))?)?;
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

    /// Commit a compact-cache range written directly by a device kernel.
    pub fn commit_compact_f16_append(&mut self, start: usize, steps: usize) -> anyhow::Result<()> {
        let has_capacity = self.compact.as_ref().is_some_and(|cache| {
            self.device.dtype(cache).expect("typed cache tensor") == Dtype::F16
                && cache.dim(0) > 0
                && cache.dim(1) as usize >= self.total
        });
        if start != self.compact_total
            || steps == 0
            || start.checked_add(steps) != Some(self.total)
            || self.total > self.compact_cap
            || !has_capacity
        {
            return Err(anyhow::Error::msg(format!(
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
    /// lets its selected-token list match the reference's `[min_pos..=pos]` convention.
    pub fn write_compact_f16_at(
        &mut self,
        latent: &D::Tensor,
        position: usize,
    ) -> anyhow::Result<()> {
        let latent = self.cast(latent, Dtype::F16)?;
        let (b, s, d) = (latent.dim(0), latent.dim(1), latent.dim(2));
        if s != 1 {
            return Err(anyhow::Error::msg(format!(
                "compact cache positional write requires one row, got {s}"
            )));
        }
        let needed = position.saturating_add(1);
        let same_batch = self.compact.as_ref().is_none_or(|r| r.dim(0) == b);
        let same_dim = self.compact.as_ref().is_none_or(|r| r.dim(2) == d);
        if self.compact_cap < needed || !same_batch || !same_dim {
            let mut new_cap = self.compact_cap.max(256);
            while new_cap < needed {
                new_cap = new_cap.saturating_mul(2);
            }
            let mut next = self.zeros(&[b, new_cap as i32, d], Dtype::F16)?;
            if let Some(old) = &self.compact {
                let keep = self.compact_total.min(new_cap) as i32;
                if keep > 0 && same_batch && same_dim {
                    self.assign(&mut next, 0..keep, &old.slice(1, 0, Some(keep))?)?;
                }
            }
            self.compact = Some(next);
            self.compact_cap = new_cap;
        }

        let dst = self.compact.as_ref().unwrap();
        if !self.copy_rows(&latent, dst, position as i32)? {
            self.compact = Some(self.assigned(
                self.compact.as_ref().unwrap(),
                position as i32..needed as i32,
                &latent,
            )?);
        }
        self.compact_total = self.compact_total.max(needed);
        Ok(())
    }

    /// The live compact GLM MLA latents `[B, end-start, D]` in F16.
    pub fn compact_slice(&self, start: usize, end: usize) -> Option<D::Tensor> {
        if end > self.compact_total || start > end {
            return None;
        }
        self.compact.as_ref().map(|r| {
            r.slice(1, start as i32, Some(end as i32))
                .expect("validated tape range")
        })
    }

    pub fn raw_live_for_test(&self) -> Option<D::Tensor> {
        self.raw_slice(self.raw_start, self.total)
    }

    pub fn compact_live_for_test(&self) -> Option<D::Tensor> {
        self.compact_slice(0, self.compact_total)
    }

    /// Append `pooled` `[B, n_new, D]` into the pooled buffer.
    pub fn push_pooled(&mut self, pooled: &D::Tensor) -> anyhow::Result<()> {
        let b = pooled.dim(0);
        let d = pooled.dim(2);
        let n_new = pooled.dim(1) as usize;
        let needed = self.pooled_upto + n_new;
        let same_batch = self.pooled.as_ref().is_none_or(|p| p.dim(0) == b);
        if self.pcap < needed || !same_batch {
            let mut new_cap = if self.pcap == 0 {
                needed + 256
            } else {
                self.pcap
            };
            while new_cap < needed {
                new_cap *= 2;
            }
            let mut np = match self.device.dtype(pooled).expect("typed cache tensor") {
                Dtype::F32 => self.zeros(&[b, new_cap as i32, d], Dtype::F32)?,
                Dtype::F16 => self.zeros(&[b, new_cap as i32, d], Dtype::F16)?,
                _ => self.zeros(&[b, new_cap as i32, d], Dtype::BF16)?,
            };
            if let Some(old) = &self.pooled {
                let keep = self.pooled_upto as i32;
                if keep > 0 {
                    self.assign(&mut np, 0..keep, &old.slice(1, 0, Some(keep))?)?;
                }
            }
            self.pooled = Some(np);
            self.pcap = new_cap;
        }
        let range = self.pooled_upto as i32..needed as i32;

        let pooled_buf = self.pooled.as_ref().unwrap();
        if !self.copy_rows(pooled, pooled_buf, self.pooled_upto as i32)? {
            self.pooled =
                Some(self.assigned(self.pooled.as_ref().unwrap(), range, &pooled.clone())?);
        }
        self.pooled_upto = needed;
        Ok(())
    }

    /// Append a GLM pooled key after the reference-compatible F16 cache cast.
    pub fn push_pooled_f16(&mut self, pooled: &D::Tensor) -> anyhow::Result<()> {
        self.push_pooled(&self.cast(pooled, Dtype::F16)?)
    }

    /// The live pooled keys `[B, pooled_upto, D]` (a view of the buffer).
    pub fn pooled_view(&self) -> Option<D::Tensor> {
        self.pooled.as_ref().map(|p| {
            p.slice(1, 0, Some(self.pooled_upto as i32))
                .expect("validated pooled range")
        })
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
    pub fn snapshot(&self) -> TapeSnapshot<D::Tensor> {
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

    pub fn restore(&mut self, s: &TapeSnapshot<D::Tensor>) {
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
    /// are shared metadata in the reference's GLM indexer path.
    pub fn pack_equal(
        context: crate::execution::DeviceContext<D>,
        tapes: &[&Self],
    ) -> anyhow::Result<Self> {
        let Some(first) = tapes.first().copied() else {
            return Ok(Self::new(context));
        };
        for tape in &tapes[1..] {
            if tape.total != first.total
                || tape.raw_start != first.raw_start
                || tape.pooled_upto != first.pooled_upto
                || tape.cr != first.cr
                || tape.compact_total != first.compact_total
            {
                return Err(anyhow::Error::msg(
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
        let device = context.device().clone();
        Ok(Self {
            context,
            stream: device.stream(),
            device,
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
    pub fn extract_row(&self, row: usize) -> anyhow::Result<Self> {
        let raw = extract_tape_field::<D>(self.raw.as_ref(), row, "raw")?;
        let pooled = extract_tape_field::<D>(self.pooled.as_ref(), row, "pooled")?;
        let compact = extract_tape_field::<D>(self.compact.as_ref(), row, "compact")?;
        Ok(Self {
            context: self.context.clone(),
            device: self.device.clone(),
            stream: self.device.stream(),
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

fn concat_live_tape_field<D: CacheDevice>(
    tapes: &[&IndexerTape<D>],
    get: impl Fn(&IndexerTape<D>) -> Option<D::Tensor>,
    label: &str,
) -> anyhow::Result<Option<D::Tensor>>
where
    D::Tensor: TensorAccess,
{
    let mut arrays = Vec::with_capacity(tapes.len());
    for tape in tapes {
        match get(tape) {
            Some(array) => arrays.push(array),
            None if arrays.is_empty() && tapes.iter().all(|candidate| get(candidate).is_none()) => {
                return Ok(None);
            }
            None => {
                return Err(anyhow::Error::msg(format!(
                    "GLM batched indexer {label} storage is missing from one stream",
                )));
            }
        }
    }
    let refs: Vec<&D::Tensor> = arrays.iter().collect();
    Ok(Some(D::Tensor::concatenate(&refs, 0)?))
}

fn extract_tape_field<D: Device>(
    array: Option<&D::Tensor>,
    row: usize,
    label: &str,
) -> anyhow::Result<Option<D::Tensor>>
where
    D::Tensor: TensorAccess,
{
    let Some(array) = array else {
        return Ok(None);
    };
    if array.rank() != 3 || row >= array.dim(0) as usize {
        return Err(anyhow::Error::msg(format!(
            "GLM batched indexer {label} row {row} is outside shape {:?}",
            array.shape(),
        )));
    }
    let row_range = row as i32..(row as i32 + 1);
    Ok(Some(
        array
            .slice(0, row_range.start, Some(row_range.end))?
            .contiguous()?,
    ))
}

/// A [`IndexerTape`] clone for a prefix-cache entry.
#[derive(Clone)]
pub struct TapeSnapshot<T> {
    pub raw: Option<T>,
    pub total: usize,
    pub raw_start: usize,
    pub pooled: Option<T>,
    pub pooled_upto: usize,
    pub cr: usize,
    pub compact: Option<T>,
    pub compact_total: usize,
}

impl<D: CacheDevice> IndexerTape<D>
where
    D::Tensor: TensorAccess,
{
    fn optional(&self, kernel: Kernel, inputs: &[&D::Tensor]) -> anyhow::Result<Option<D::Tensor>> {
        let Some(handle) = self.device.prepare(kernel)? else {
            return Ok(None);
        };
        self.device
            .dispatch(&handle, inputs, &self.stream)?
            .map(|v| v.output.into_one().map_err(anyhow::Error::msg))
            .transpose()
    }
    fn required(&self, kernel: Kernel, inputs: &[&D::Tensor]) -> anyhow::Result<D::Tensor> {
        self.optional(kernel, inputs)?
            .ok_or_else(|| anyhow::anyhow!("device declined a required cache storage operation"))
    }
    fn zeros(&self, shape: &[i32], dtype: Dtype) -> anyhow::Result<D::Tensor> {
        self.required(
            Kernel::Zeros {
                shape: shape.to_vec(),
                dtype,
            },
            &[],
        )
    }
    fn cast(&self, tensor: &D::Tensor, dtype: Dtype) -> anyhow::Result<D::Tensor> {
        self.required(Kernel::Tensor(TensorOperation::Cast(dtype)), &[tensor])
    }
    fn assigned(
        &self,
        target: &D::Tensor,
        range: std::ops::Range<i32>,
        source: &D::Tensor,
    ) -> anyhow::Result<D::Tensor> {
        self.required(
            Kernel::Tensor(TensorOperation::AssignSlice {
                axis: 1,
                start: range.start,
                end: range.end,
            }),
            &[target, source],
        )
    }
    fn assign(
        &self,
        target: &mut D::Tensor,
        range: std::ops::Range<i32>,
        source: &D::Tensor,
    ) -> anyhow::Result<()> {
        *target = self.assigned(target, range, source)?;
        Ok(())
    }
    fn copy_rows(
        &self,
        source: &D::Tensor,
        target: &D::Tensor,
        offset: i32,
    ) -> anyhow::Result<bool> {
        Ok(self
            .optional(Kernel::CopyRowsInto { offset }, &[source, target])?
            .is_some())
    }
    fn copy_fields(
        &self,
        first: &D::Tensor,
        second: &D::Tensor,
        target: &D::Tensor,
        offset: i32,
    ) -> anyhow::Result<bool> {
        Ok(self
            .optional(
                Kernel::CopyRowsConcatInto { offset },
                &[first, second, target],
            )?
            .is_some())
    }
}
