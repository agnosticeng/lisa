//! Native request-state conformance fixtures.
pub type Vision = lisa_engine::vision::Vision<lisa_metal::device::MetalDevice>;
#[cfg(test)]
use lisa_engine::vision::context::{ImageSpan, Prompt};
#[cfg(test)]
use lisa_metal::Array;
#[cfg(test)]
use lisa_models::vision::prompt::{Config, Positions};
#[cfg(test)]
mod tests {
    use super::*;
    struct NoEncoder;
    impl lisa_engine::vision::context::EncoderSource<lisa_metal::device::MetalDevice> for NoEncoder {
        fn load(
            &self,
        ) -> anyhow::Result<lisa_engine::vision::context::Encoder<lisa_metal::device::MetalDevice>>
        {
            anyhow::bail!("splice-only fixture must not acquire image encoder weights")
        }
    }

    #[test]
    fn image_spans_cross_prefill_chunks_and_clear_between_requests() -> anyhow::Result<()> {
        let features = Array::from_slice(&[10f32, 11., 20., 21., 30., 31.], &[1, 3, 2]);
        let mut vision = Vision::from_prepared(
            crate::metal_context::execution_context(),
            Config {
                glm: false,
                image_id: 1,
                start_id: 2,
                end_id: 3,
                selector: vec![0, 1, 2],
            },
            Box::new(NoEncoder),
        );
        vision.prompt = Some(Prompt {
            positions: Positions {
                values: vec![[0, 0, 0], [1, 1, 1], [2, 2, 2], [2, 2, 3], [2, 3, 2]],
                delta: -1,
            },
            spans: vec![ImageSpan { start: 2, features }],
        });
        let hidden = Array::from_slice(&[1f32; 8], &[1, 4, 2]);
        let first = vision.splice(hidden.clone(), 0)?;
        first.eval()?;
        assert_eq!(
            first.as_slice::<f32>(),
            &[1., 1., 1., 1., 10., 11., 20., 21.]
        );
        let next = vision.splice(hidden.clone(), 4)?;
        next.eval()?;
        assert_eq!(next.as_slice::<f32>(), &[30., 31., 1., 1., 1., 1., 1., 1.]);
        let positions = vision.positions(3, 3)?.unwrap();
        positions.eval()?;
        assert_eq!(
            positions.as_slice::<f32>(),
            &[2., 2., 3., 2., 3., 2., 4., 4., 4.]
        );
        vision.clear();
        assert!(vision.positions(0, 4)?.is_none());
        let plain = vision.splice(hidden, 0)?;
        plain.eval()?;
        assert_eq!(plain.as_slice::<f32>(), &[1.; 8]);
        Ok(())
    }
}
