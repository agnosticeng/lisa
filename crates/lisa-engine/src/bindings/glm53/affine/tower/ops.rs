use crate::{
    bindings::BindingDevice,
    core::graph::PlanCache,
    execution::DeviceContext,
    weights::affine::{QuantizedEmbedding, QuantizedLinear},
};
use lisa_models::glm53::affine_tower::Request;
pub fn run<D: BindingDevice>(
    context: &DeviceContext<D>,
    request: Request,
    inputs: &[&D::Tensor],
) -> anyhow::Result<Vec<D::Tensor>> {
    context.with_slot(
        "glm53.affine_tower.plans",
        || std::cell::RefCell::new(PlanCache::<D, Request>::new(context.device().clone(), 64)),
        |slot| {
            let recipe = request.clone();
            Ok(slot
                .borrow_mut()
                .try_get_or_prepare(request, || recipe.graph())?
                .run(inputs, &mut [])?
                .into_iter()
                .collect())
        },
    )
}
pub fn one<D: BindingDevice>(
    context: &DeviceContext<D>,
    request: Request,
    inputs: &[&D::Tensor],
) -> anyhow::Result<D::Tensor> {
    let mut out = run(context, request, inputs)?;
    anyhow::ensure!(out.len() == 1, "GLM decoder graph output count differs");
    Ok(out.remove(0))
}
pub fn embedding<D: BindingDevice>(
    context: &DeviceContext<D>,
    w: &QuantizedEmbedding<D::Tensor>,
    ids: &D::Tensor,
) -> anyhow::Result<D::Tensor> {
    one(
        context,
        Request::Embedding {
            group: w.group_size,
            bits: w.bits,
        },
        &[ids, &w.weight, &w.scales, &w.biases],
    )
}
pub fn head<D: BindingDevice>(
    context: &DeviceContext<D>,
    w: &QuantizedLinear<D::Tensor>,
    x: &D::Tensor,
) -> anyhow::Result<D::Tensor> {
    let projection = crate::bindings::glm53::affine::projection::descriptor(
        context,
        w,
        x,
        crate::bindings::glm53::policy::glm53_q8_native_prefill_enabled(),
    )?;
    one(
        context,
        Request::Head { projection },
        &[x, &w.weight, &w.scales, &w.biases],
    )
}
