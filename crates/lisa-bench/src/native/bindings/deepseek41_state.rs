//! Native conformance tests for portable DeepSeek session storage.
pub use lisa_engine::bindings::deepseek41_state::MAX_CONTEXT;
pub type State = lisa_engine::bindings::deepseek41_state::State<lisa_metal::device::MetalDevice>;
#[cfg(test)]
pub(crate) use crate::core::storage::store_row;
pub(crate) use crate::core::storage::view;
#[cfg(test)]
use {
    anyhow::Result,
    lisa_metal::{Array, ops},
};
#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn deepseek41_admission_counts_full_native_storage() -> Result<()> {
        for capacity in [1, 257, 8191, 8192, 16384] {
            let state = State::new(crate::metal_context::device(), capacity)?;
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
        let mut state = State::new(crate::metal_context::device(), 8193)?;
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
        let mut state = State::new(crate::metal_context::device(), 257)?;
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
        assert!(State::new(crate::metal_context::device(), MAX_CONTEXT + 1).is_err());
        state.valid = false;
        assert!(state.snapshot().is_err());
        Ok(())
    }

    #[test]
    fn deepseek41_native_prefix_and_rollback_are_copy_on_write() -> Result<()> {
        use crate::core::cache::{LayerCache, NativeCache};
        let mut cache = LayerCache::Native(NativeCache::new(State::new(
            crate::metal_context::device(),
            129,
        )?));
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
        let mut second = LayerCache::Native(NativeCache::new(State::empty(
            crate::metal_context::device(),
            129,
        )));
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
            prefix.encode(&crate::metal_context::device()).is_err(),
            "native caches must not silently use the attention spill format"
        );
        Ok(())
    }
}
