use crate::core::graph::PlanCache;
use crate::{bindings::BindingDevice, execution::DeviceContext};
use lisa_models::glm53::routed_moe::Request;
pub fn run<D: BindingDevice + lisa_backend::IndirectDevice>(
    context: &DeviceContext<D>,
    request: Request,
    inputs: &[&D::Tensor],
    resources: Option<&D::Resources>,
) -> anyhow::Result<Vec<D::Tensor>> {
    context.with_slot(
        "glm53.moe.plans",
        || std::cell::RefCell::new(PlanCache::<D, Request>::new(context.device().clone(), 64)),
        |slot| {
            let mut p = slot.borrow_mut();
            let recipe = request.clone();
            let plan = p.try_get_or_prepare(request, || recipe.graph())?;
            let output = if let Some(r) = resources {
                plan.run_with_resources(inputs, &mut [], r)?
            } else {
                plan.run(inputs, &mut [])?
            };
            Ok(output.into_iter().collect())
        },
    )
}
