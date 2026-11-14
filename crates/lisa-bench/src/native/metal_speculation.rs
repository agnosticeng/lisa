//! Native speculative storage conformance tests.
#[cfg(test)]
use crate::{core::cache::LayerCache, speculative_runtime::SpeculativeRuntime};
#[cfg(test)]
use lisa_metal::Array;
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn speculative_rollback_uses_loaded_history_extent() -> anyhow::Result<()> {
        for history in [1, 3, 5] {
            let mut state = crate::core::cache::GdnCache::new();
            state.conv = Some(Array::from_slice(
                &vec![0f32; history],
                &[1, history as i32, 1],
            ));
            state.capture_ssm = Some(Array::from_slice(&[10f32, 20., 30.], &[3, 1, 1, 1]));
            let values: Vec<f32> = (0..history + 3).map(|v| v as f32).collect();
            state.capture_conv_input =
                Some(Array::from_slice(&values, &[1, (history + 3) as i32, 1]));
            let mut cache = LayerCache::Linear(state);
            crate::metal_language::runtime().rollback(&mut cache, 3, 1, true)?;
            let LayerCache::Linear(state) = cache else {
                unreachable!()
            };
            assert_eq!(
                state.conv.unwrap().to_vec1::<f32>()?,
                values[1..history + 1]
            );
            assert_eq!(state.ssm.unwrap().to_vec1::<f32>()?, [10.]);
        }
        Ok(())
    }
}
