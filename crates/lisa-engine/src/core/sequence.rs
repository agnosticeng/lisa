//! Generic execution of model-owned block dataflow, with no model selection.
use lisa_models::sequence::Plan;
pub trait Blocks<O> {
    type Tensor: Clone;
    fn call(
        &mut self,
        operation: &O,
        inputs: &[Option<&Self::Tensor>],
    ) -> anyhow::Result<Vec<Option<Self::Tensor>>>;
}
pub struct Prepared<O, const N: usize> {
    plan: Plan<O>,
}
impl<O, const N: usize> Prepared<O, N> {
    pub fn new(plan: Plan<O>) -> anyhow::Result<Self> {
        plan.validate().map_err(anyhow::Error::msg)?;
        anyhow::ensure!(plan.slots <= N, "block register capacity exceeded");
        Ok(Self { plan })
    }
    pub fn run<B: Blocks<O>>(
        &self,
        blocks: &mut B,
        inputs: &[Option<&B::Tensor>],
    ) -> anyhow::Result<Vec<Option<B::Tensor>>> {
        anyhow::ensure!(
            inputs.len() == self.plan.inputs,
            "block input count differs"
        );
        let mut values: [Option<B::Tensor>; N] = std::array::from_fn(|_| None);
        for (dst, src) in values.iter_mut().zip(inputs) {
            *dst = src.cloned();
        }
        for call in &self.plan.calls {
            let mut stack: [Option<&B::Tensor>; 16] = [None; 16];
            let heap;
            let args = if call.inputs.len() <= stack.len() {
                for (dst, &i) in stack.iter_mut().zip(&call.inputs) {
                    *dst = values[i].as_ref();
                }
                &stack[..call.inputs.len()]
            } else {
                heap = call
                    .inputs
                    .iter()
                    .map(|&i| values[i].as_ref())
                    .collect::<Vec<_>>();
                &heap
            };
            let output = blocks.call(&call.operation, args)?;
            anyhow::ensure!(
                output.len() == call.outputs.len(),
                "bound block returned incorrect output count"
            );
            for (&i, v) in call.outputs.iter().zip(output) {
                values[i] = v;
            }
        }
        Ok(self
            .plan
            .outputs
            .iter()
            .map(|&i| values[i].clone())
            .collect())
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use lisa_models::sequence::Call;
    struct Recorder(Vec<usize>);
    impl Blocks<usize> for Recorder {
        type Tensor = i32;
        fn call(
            &mut self,
            op: &usize,
            inputs: &[Option<&i32>],
        ) -> anyhow::Result<Vec<Option<i32>>> {
            self.0.push(*op);
            Ok(vec![Some(
                inputs.iter().flatten().map(|&&i| i).sum::<i32>() + *op as i32,
            )])
        }
    }
    #[test]
    fn block_executor_obeys_order_and_optional_inputs() -> anyhow::Result<()> {
        let p = Prepared::<_, 4>::new(Plan {
            inputs: 2,
            slots: 4,
            calls: vec![
                Call {
                    operation: 7,
                    inputs: vec![0, 1],
                    outputs: vec![2],
                },
                Call {
                    operation: 3,
                    inputs: vec![2],
                    outputs: vec![3],
                },
            ],
            outputs: vec![3, 2],
        })?;
        let mut b = Recorder(vec![]);
        assert_eq!(p.run(&mut b, &[Some(&5), None])?, vec![Some(15), Some(12)]);
        assert_eq!(b.0, vec![7, 3]);
        Ok(())
    }
}
