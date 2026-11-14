#[cfg(test)]
mod tests {
    use lisa_engine::core::traversal::*;
    use lisa_metal::{Array, device::MetalDevice};
    struct Recorder {
        calls: Vec<String>,
        fail: Option<usize>,
    }
    impl Blocks for Recorder {
        type Tensor = Array;
        type State = Array;
        fn begin(&mut self) -> anyhow::Result<Array> {
            self.calls.push("begin".into());
            Ok(Array::from_slice(&[0f32, 0.], &[1, 2]))
        }
        fn layer(&mut self, i: usize, state: &mut Array) -> anyhow::Result<()> {
            self.calls.push(format!("layer {i}"));
            anyhow::ensure!(self.fail != Some(i), "injected layer failure");
            *state = Array::from_slice(&[i as f32, 10. + i as f32], &[1, 2]);
            Ok(())
        }
        fn capture(&self, state: &Array) -> anyhow::Result<Array> {
            Ok(state.clone())
        }
        fn finish(&self, state: Array) -> anyhow::Result<Array> {
            Ok(state)
        }
    }
    #[test]
    fn traversal_keeps_captured_values_alive_and_stops_at_failure() -> anyhow::Result<()> {
        let plan = PreparedTraversal::new(
            MetalDevice::system_default(),
            lisa_models::qwen3_5::tower::traversal(4, &[3, 1]),
        )?;
        let mut r = Recorder {
            calls: vec![],
            fail: None,
        };
        let (out, capture) = plan.run(&mut r)?;
        assert_eq!(
            r.calls,
            vec!["begin", "layer 0", "layer 1", "layer 2", "layer 3"]
        );
        assert_eq!(out.to_vec1::<f32>()?, vec![3., 13.]);
        assert_eq!(capture.unwrap().to_vec1::<f32>()?, vec![1., 11., 3., 13.]);
        r.calls.clear();
        r.fail = Some(2);
        assert!(plan.run(&mut r).is_err());
        assert_eq!(r.calls, vec!["begin", "layer 0", "layer 1", "layer 2"]);
        Ok(())
    }
}
