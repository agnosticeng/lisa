//! Engine-owned preparation and execution over a model-unaware device endpoint.
use lisa_backend::Device;
use lisa_models::{Alternatives, DimensionGuard, Plan, Value};

struct PreparedBranch<D: Device> {
    plan: Plan,
    handles: Vec<D::KernelHandle>,
}
pub struct PreparedPlan<D: Device> {
    device: D,
    stream: D::Stream,
    primary: PreparedBranch<D>,
    guarded: Option<(DimensionGuard, PreparedBranch<D>)>,
    weights: Vec<D::Tensor>,
}
impl<D: Device> PreparedPlan<D>
where
    D::Tensor: Clone,
{
    pub fn new(device: D, plans: Alternatives, weights: Vec<D::Tensor>) -> anyhow::Result<Self> {
        plans
            .primary
            .validate(weights.len())
            .map_err(anyhow::Error::msg)?;
        if let Some((guard, plan)) = &plans.guarded {
            anyhow::ensure!(
                guard.axis < guard.min_rank,
                "plan shape guard axis exceeds minimum rank"
            );
            plan.validate(weights.len()).map_err(anyhow::Error::msg)?;
        }
        let prepare = |plan: Plan| -> anyhow::Result<Option<PreparedBranch<D>>> {
            let mut handles = Vec::with_capacity(plan.calls.len());
            for call in &plan.calls {
                let Some(handle) = device.prepare(call.kernel.clone())? else {
                    return Ok(None);
                };
                handles.push(handle);
            }
            Ok(Some(PreparedBranch { plan, handles }))
        };
        let primary = prepare(plans.primary)?
            .ok_or_else(|| anyhow::anyhow!("device lacks a required model-plan kernel contract"))?;
        let guarded = if let Some((guard, plan)) = plans.guarded {
            prepare(plan)?.map(|branch| (guard, branch))
        } else {
            None
        };
        let stream = device.stream();
        Ok(Self {
            device,
            stream,
            primary,
            guarded,
            weights,
        })
    }
    pub fn weights(&self) -> &[D::Tensor] {
        &self.weights
    }
    pub fn run(&self, input: &D::Tensor) -> anyhow::Result<D::Tensor> {
        if let Some((guard, branch)) = &self.guarded {
            let shape = self.device.shape(input);
            if shape.len() >= guard.min_rank && shape[guard.axis] == guard.equals {
                if let Some(output) = self.execute(branch, input)? {
                    return Ok(output);
                }
            }
        }
        self.execute(&self.primary, input)?
            .ok_or_else(|| anyhow::anyhow!("device declined a required model-plan kernel"))
    }
    fn execute(
        &self,
        branch: &PreparedBranch<D>,
        input: &D::Tensor,
    ) -> anyhow::Result<Option<D::Tensor>> {
        let plan = &branch.plan;
        let mut temporaries = Vec::with_capacity(plan.calls.len());
        for (call, handle) in plan.calls.iter().zip(&branch.handles) {
            let value = |v| match v {
                Value::Input => input,
                Value::Weight(i) => &self.weights[i],
                Value::Temporary(i) => &temporaries[i],
            };
            let mut inputs = [input; 4];
            for (slot, &binding) in call.inputs.iter().enumerate() {
                inputs[slot] = value(binding);
            }
            match self
                .device
                .dispatch(handle, &inputs[..call.inputs.len()], &self.stream)?
            {
                Some(result) => {
                    temporaries.push(result.output.into_one().map_err(anyhow::Error::msg)?)
                }
                None => return Ok(None),
            }
        }
        Ok(Some(match plan.output {
            Value::Input => input.clone(),
            Value::Weight(i) => self.weights[i].clone(),
            Value::Temporary(i) => temporaries.swap_remove(i),
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lisa_backend::{Dispatch, ImportOptions, Kernel, KernelCapability, TensorSource};
    use std::{cell::RefCell, rc::Rc};
    #[derive(Clone, Debug)]
    struct Tensor {
        shape: Vec<i32>,
    }
    struct Recorder {
        calls: Rc<RefCell<Vec<Kernel>>>,
        fail: bool,
        prepared: Rc<RefCell<usize>>,
    }
    impl Device for Recorder {
        type Tensor = Tensor;
        type Error = std::io::Error;
        type KernelHandle = Kernel;
        type Stream = ();
        type Completion = ();
        fn capabilities(&self) -> &[KernelCapability] {
            &[]
        }
        fn dtype(&self, _: &Tensor) -> Option<lisa_backend::ScalarType> {
            Some(lisa_backend::ScalarType::BF16)
        }
        fn stream(&self) {}
        fn prepare(&self, op: Kernel) -> Result<Option<Kernel>, Self::Error> {
            *self.prepared.borrow_mut() += 1;
            Ok(Some(op))
        }
        fn wait(&self, _: &()) -> Result<(), Self::Error> {
            Ok(())
        }
        fn pull(&self, source: TensorSource<'_>, _: ImportOptions) -> Result<Tensor, Self::Error> {
            let source = source
                .host_tensor()
                .ok_or_else(|| std::io::Error::other("invalid source"))?;
            if !source.validate() {
                return Err(std::io::Error::other("invalid source"));
            }
            Ok(Tensor {
                shape: source.shape.to_vec(),
            })
        }
        fn shape<'a>(&self, tensor: &'a Tensor) -> &'a [i32] {
            &tensor.shape
        }
        fn dispatch(
            &self,
            handle: &Kernel,
            inputs: &[&Tensor],
            _: &(),
        ) -> Result<Option<Dispatch<lisa_backend::Outputs<Tensor>, ()>>, Self::Error> {
            let op = handle.clone();
            self.calls.borrow_mut().push(op.clone());
            if op == Kernel::PackedSwiGlu {
                if self.fail {
                    return Err(std::io::Error::other("device failure"));
                }
                return Ok(None);
            }
            Ok(Some(Dispatch {
                output: lisa_backend::Outputs::One(inputs[0].clone()),
                completion: (),
            }))
        }
    }
    #[test]
    fn executor_obeys_model_guard_fallback_and_propagates_device_errors() -> anyhow::Result<()> {
        for (tokens, fail) in [(8, false), (1, false), (1, true)] {
            let calls = Rc::new(RefCell::new(Vec::new()));
            let prepared_count = Rc::new(RefCell::new(0));
            let device = Recorder {
                calls: calls.clone(),
                fail,
                prepared: prepared_count.clone(),
            };
            let plans = lisa_models::dense::swiglu([(64, 4); 3], Some((64, 4)));
            let prepared =
                PreparedPlan::new(device, plans, vec![Tensor { shape: vec![1, 1] }; 12])?;
            assert_eq!(*prepared_count.borrow(), 7);
            let result = prepared.run(&Tensor {
                shape: vec![1, tokens, 128],
            });
            let count = calls.borrow().len();
            if fail {
                assert!(result.unwrap_err().to_string().contains("device failure"));
                assert_eq!(count, 2);
            } else {
                assert!(result.is_ok());
                assert_eq!(count, if tokens == 1 { 6 } else { 4 });
                prepared.run(&Tensor {
                    shape: vec![1, tokens, 128],
                })?;
                assert_eq!(
                    *prepared_count.borrow(),
                    7,
                    "execution must reuse prepared handles"
                );
            }
        }
        Ok(())
    }
}
