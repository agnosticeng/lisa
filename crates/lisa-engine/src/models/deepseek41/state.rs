//! V4.1 session storage. Raw rings, shared compressed owners and compressor
//! carry are session-local; a failed token invalidates the entire session.

use super::engram::History;
use anyhow::{Result, ensure};
use lisa_mlx::{Array, Dtype, ops};

pub const MAX_CONTEXT: usize = 1_048_576;

pub struct Owner {
    pub compressed: Array,
    pub index: Array,
    pub index_bf16: Option<Array>,
    pub previous_kv: Array,
    pub previous_score: Array,
}

pub struct State {
    pub(crate) position: usize,
    pub(crate) capacity: usize,
    pub(crate) valid: bool,
    pub(crate) history: History,
    pub(crate) raw: Vec<Array>,
    pub(crate) owners: Vec<Owner>,
}

impl State {
    /// Native caches reserve their full configured capacity on first use.
    /// Admission must count this even for a short request or restored prefix.
    pub fn storage_bytes(capacity: usize) -> usize {
        let fixed = 40 * 128 * 512 * 4 + 4 * 2 * 512 * 4;
        (0..4).fold(fixed, |total, owner| {
            let count = if owner < 3 { capacity / 2 } else { capacity }.max(1);
            total + count * (640 * 4 + if count >= 8192 { 128 * 2 } else { 0 })
        })
    }

    /// Infallible empty session descriptor; allocation is deferred until the
    /// model's first forward, where allocation failures can be returned.
    pub fn empty(capacity: usize) -> Self {
        Self {
            position: 0,
            capacity,
            valid: true,
            history: History::default(),
            raw: Vec::new(),
            owners: Vec::new(),
        }
    }

    pub fn initialize(&mut self) -> Result<()> {
        if self.raw.is_empty() {
            *self = Self::new(self.capacity)?;
        }
        Ok(())
    }
    /// Reserve the configured context under the runtime's RAM budget. The
    /// BF16 mirror is needed only if a compressed owner can reach 8192 keys.
    pub fn new(capacity: usize) -> Result<Self> {
        ensure!(
            (1..=MAX_CONTEXT).contains(&capacity),
            "DeepSeek context capacity must be between 1 and {MAX_CONTEXT}"
        );
        let raw = (0..40)
            .map(|_| ops::zeros::<f32>(&[128, 512]))
            .collect::<lisa_mlx::error::Result<Vec<_>>>()?;
        let owners = (0..4)
            .map(|owner| -> Result<_> {
                let count = if owner < 3 { capacity / 2 } else { capacity }.max(1) as i32;
                Ok(Owner {
                    compressed: ops::zeros::<f32>(&[count, 512])?,
                    index: ops::zeros::<f32>(&[count, 128])?,
                    index_bf16: if count >= 8192 {
                        Some(ops::zeros::<half::bf16>(&[count, 128])?)
                    } else {
                        None
                    },
                    previous_kv: ops::zeros::<f32>(&[1, 512])?,
                    previous_score: ops::zeros::<f32>(&[1, 512])?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            position: 0,
            capacity,
            valid: true,
            history: History::default(),
            raw,
            owners,
        })
    }

    pub fn position(&self) -> usize {
        self.position
    }
    pub fn capacity(&self) -> usize {
        self.capacity
    }
    pub fn is_valid(&self) -> bool {
        self.valid
    }

    /// An independent GPU copy, including raw ring contents and odd-token
    /// compressor carry. Cloning handles alone would corrupt rollback at wrap.
    pub fn snapshot(&self) -> Result<Self> {
        ensure!(self.valid, "cannot snapshot an invalid DeepSeek session");
        let copy = |x: &Array| -> Result<Array> { Ok(Array::new(x.t.copied()?)) };
        let raw = self.raw.iter().map(copy).collect::<Result<Vec<_>>>()?;
        let owners = self
            .owners
            .iter()
            .map(|owner| -> Result<_> {
                Ok(Owner {
                    compressed: copy(&owner.compressed)?,
                    index: copy(&owner.index)?,
                    index_bf16: owner.index_bf16.as_ref().map(copy).transpose()?,
                    previous_kv: copy(&owner.previous_kv)?,
                    previous_score: copy(&owner.previous_score)?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            position: self.position,
            capacity: self.capacity,
            valid: true,
            history: self.history,
            raw,
            owners,
        })
    }
}

impl crate::core::cache::NativeState for State {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
    fn copy_for_write(&self) -> Result<std::sync::Arc<dyn crate::core::cache::NativeState>> {
        Ok(std::sync::Arc::new(self.snapshot()?))
    }
    fn position(&self) -> usize {
        self.position
    }
}

/// Checked contiguous view; offsets are bytes at the outer Array boundary.
pub(super) fn view(x: &Array, first: usize, shape: &[i32]) -> Result<Array> {
    ensure!(
        matches!(x.dtype(), Dtype::Float32 | Dtype::Bfloat16 | Dtype::Int32)
            && x.t.buffer_and_layout().1.is_contiguous(),
        "DeepSeek state view requires contiguous F32, BF16 or I32 storage"
    );
    let count = shape.iter().try_fold(1usize, |n, &d| {
        ensure!(d > 0, "invalid DeepSeek state view shape");
        n.checked_mul(d as usize)
            .ok_or_else(|| anyhow::anyhow!("state view size overflow"))
    })?;
    ensure!(
        first <= x.size() && count <= x.size() - first,
        "DeepSeek state view out of bounds"
    );
    x.eval_enqueue()?;
    let (buffer, layout) = x.t.buffer_and_layout();
    Ok(Array::from_buffer_offset(
        buffer.clone(),
        (layout.offset + first) * x.dtype().size_of(),
        shape,
        x.dtype(),
    )?)
}

pub(super) fn store_row(dst: &mut Array, row: usize, src: &Array) -> Result<()> {
    ensure!(
        dst.rank() == 2
            && src.shape() == [1, dst.dim(1)]
            && row < dst.dim(0) as usize
            && dst.dtype() == src.dtype(),
        "invalid DeepSeek cache row write"
    );
    let target = view(dst, row * dst.dim(1) as usize, src.shape())?;
    src.eval_enqueue()?;
    src.t.copy_strided_into(&target.t)?;
    *dst = Array::new(dst.t.clone());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deepseek41_admission_counts_full_native_storage() -> Result<()> {
        for capacity in [1, 257, 8191, 8192, 16384] {
            let state = State::new(capacity)?;
            let bytes = |x: &Array| x.size() * x.dtype().size_of();
            let actual = state.raw.iter().map(bytes).sum::<usize>()
                + state
                    .owners
                    .iter()
                    .map(|owner| {
                        bytes(&owner.compressed)
                            + bytes(&owner.index)
                            + owner.index_bf16.as_ref().map_or(0, bytes)
                            + bytes(&owner.previous_kv)
                            + bytes(&owner.previous_score)
                    })
                    .sum::<usize>();
            assert_eq!(State::storage_bytes(capacity), actual);
        }
        Ok(())
    }

    #[test]
    fn deepseek41_bf16_index_mirror_snapshot_preserves_offsets() -> Result<()> {
        let mut state = State::new(8193)?;
        assert!(state.owners[0].index_bf16.is_none());
        let mirror = state.owners[3].index_bf16.as_mut().unwrap();
        let row = Array::from_slice(&vec![half::bf16::from_f32(1.25); 128], &[1, 128]);
        store_row(mirror, 8191, &row)?;
        let snapshot = state.snapshot()?;
        let zero = ops::zeros::<half::bf16>(&[1, 128])?;
        store_row(state.owners[3].index_bf16.as_mut().unwrap(), 8191, &zero)?;
        let restored = view(
            snapshot.owners[3].index_bf16.as_ref().unwrap(),
            8191 * 128,
            &[1, 128],
        )?;
        assert!(
            restored
                .as_slice::<half::bf16>()
                .iter()
                .all(|v| v.to_f32() == 1.25)
        );
        Ok(())
    }

    #[test]
    fn deepseek41_state_snapshot_owns_wrapped_ring_and_carry() -> Result<()> {
        let mut state = State::new(257)?;
        for row in 0..128 {
            store_row(
                &mut state.raw[0],
                row,
                &Array::from_slice(&vec![row as f32; 512], &[1, 512]),
            )?;
        }
        state.position = 129;
        state.owners[0].previous_kv = Array::from_slice(&vec![7f32; 512], &[1, 512]);
        store_row(
            &mut state.owners[0].compressed,
            63,
            &Array::from_slice(&vec![11f32; 512], &[1, 512]),
        )?;
        let snapshot = state.snapshot()?;
        // Queue writes after the copy, before a readback, to check dependency
        // tracking and storage ownership rather than a CPU clone.
        store_row(
            &mut state.raw[0],
            0,
            &Array::from_slice(&vec![99f32; 512], &[1, 512]),
        )?;
        state.owners[0].previous_kv = Array::from_slice(&vec![13f32; 512], &[1, 512]);
        store_row(
            &mut state.owners[0].compressed,
            63,
            &Array::from_slice(&vec![17f32; 512], &[1, 512]),
        )?;
        assert!(
            view(&snapshot.raw[0], 0, &[1, 512])?
                .as_slice::<f32>()
                .iter()
                .all(|&v| v == 0.0)
        );
        assert!(
            view(&snapshot.owners[0].compressed, 63 * 512, &[1, 512])?
                .as_slice::<f32>()
                .iter()
                .all(|&v| v == 11.0)
        );
        assert!(
            snapshot.owners[0]
                .previous_kv
                .as_slice::<f32>()
                .iter()
                .all(|&v| v == 7.0)
        );
        assert_eq!(snapshot.position(), 129);
        assert!(State::new(MAX_CONTEXT + 1).is_err());
        state.valid = false;
        assert!(state.snapshot().is_err());
        Ok(())
    }

    #[test]
    fn deepseek41_native_prefix_and_rollback_are_copy_on_write() -> Result<()> {
        use crate::core::cache::{LayerCache, NativeCache};
        let mut cache = LayerCache::Native(NativeCache::new(State::new(129)?));
        let prefix = cache.snapshot_prefix();
        let rollback = cache.snapshot_rollback();
        let LayerCache::Native(native) = &mut cache else {
            unreachable!()
        };
        let state = native.get_mut::<State>()?;
        state.position = 1;
        store_row(
            &mut state.raw[1],
            0,
            &Array::from_slice(&vec![31f32; 512], &[1, 512]),
        )?;
        cache.restore_rollback(&rollback)?;
        let LayerCache::Native(native) = &cache else {
            unreachable!()
        };
        assert_eq!(native.position(), 0);
        assert!(
            native.get::<State>()?.raw[1]
                .as_slice::<f32>()
                .iter()
                .all(|&v| v == 0.0)
        );
        let mut second = LayerCache::Native(NativeCache::new(State::empty(129)));
        second.restore_prefix(&prefix);
        let LayerCache::Native(native) = &mut second else {
            unreachable!()
        };
        let state = native.get_mut::<State>()?;
        state.position = 1;
        store_row(
            &mut state.raw[1],
            0,
            &Array::from_slice(&vec![47f32; 512], &[1, 512]),
        )?;
        cache.restore_prefix(&prefix);
        let LayerCache::Native(native) = &cache else {
            unreachable!()
        };
        assert_eq!(native.position(), 0);
        assert!(
            native.get::<State>()?.raw[1]
                .as_slice::<f32>()
                .iter()
                .all(|&v| v == 0.0)
        );
        assert!(
            prefix.encode().is_err(),
            "native caches must not silently use the attention spill format"
        );
        Ok(())
    }
}
