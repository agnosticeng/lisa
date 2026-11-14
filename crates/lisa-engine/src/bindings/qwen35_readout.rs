//! Draft readout bindings and diagnostic outputs; selection belongs to the model.
use crate::{bindings::BindingDevice, execution::DeviceContext};
use crate::{core::graph::PlanCache, weights::affine::QuantizedLinear};
use lisa_backend::TensorAccess;
use lisa_models::qwen3_5::readout::Plan;
pub struct Selected<T> {
    pub token: T,
    pub scores: T,
    pub candidates: Option<T>,
    pub confidence: Option<T>,
}
pub fn select<D: BindingDevice>(
    context: &DeviceContext<D>,
    hidden: &D::Tensor,
    target: &QuantizedLinear<D::Tensor>,
    coarse: Option<&QuantizedLinear<D::Tensor>>,
    confidence: bool,
) -> anyhow::Result<Selected<D::Tensor>> {
    anyhow::ensure!(
        hidden.rank() == 2 && hidden.dim(0) == 1,
        "native draft readout requires one hidden row"
    );
    let plan = Plan {
        target: (target.group_size, target.bits),
        coarse: coarse.map(|p| (p.group_size, p.bits)),
        confidence,
    };
    let mut inputs = vec![hidden, &target.weight, &target.scales, &target.biases];
    if let Some(p) = coarse {
        inputs.extend([&p.weight, &p.scales, &p.biases]);
    }
    let mut out = context
        .with_slot(
            "qwen35_readout.plans",
            || std::cell::RefCell::new(PlanCache::new(context.device().clone(), 16)),
            |p| {
                p.borrow_mut()
                    .get_or_prepare(plan, || plan.graph())?
                    .run(&inputs, &mut [])
            },
        )?
        .into_iter();
    Ok(Selected {
        token: out.next().unwrap(),
        scores: out.next().unwrap(),
        candidates: coarse.and_then(|_| out.next()),
        confidence: confidence.then(|| out.next().unwrap()),
    })
}
