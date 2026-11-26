use crate::{bindings::BindingDevice, execution::DeviceContext};
use crate::{core::graph::PlanCache, weights::affine::QuantizedLinear};
use lisa_backend::TensorAccess;
use lisa_models::glm53::affine_projection::Spec;
pub fn run<D: BindingDevice>(
    context: &DeviceContext<D>,
    request: Spec,
    inputs: &[&D::Tensor],
) -> anyhow::Result<Vec<D::Tensor>> {
    context.with_slot(
        "glm53.affine_projection.plans",
        || std::cell::RefCell::new(PlanCache::<D, Spec>::new(context.device().clone(), 64)),
        |slot| {
            let mut p = slot.borrow_mut();
            let recipe = request.clone();
            Ok(p.try_get_or_prepare(request, || recipe.graph())?
                .run(inputs, &mut [])?
                .into_iter()
                .collect())
        },
    )
}
pub fn forward<D: BindingDevice>(
    context: &DeviceContext<D>,
    w: &QuantizedLinear<D::Tensor>,
    x: &D::Tensor,
    native: bool,
) -> anyhow::Result<D::Tensor> {
    let spec = descriptor(context, w, x, native)?;
    Ok(run(context, spec, &[x, &w.weight, &w.scales, &w.biases])?.remove(0))
}

pub fn descriptor<D: BindingDevice>(
    context: &DeviceContext<D>,
    w: &QuantizedLinear<D::Tensor>,
    x: &D::Tensor,
    native: bool,
) -> anyhow::Result<Spec> {
    let device = context.device();
    let dtype = |a: &D::Tensor| {
        device
            .dtype(a)
            .ok_or_else(|| anyhow::anyhow!("unsupported affine dtype"))
    };
    Ok(Spec {
        shape: x.shape().to_vec(),
        weight: w.weight.shape().to_vec(),
        input_dtype: dtype(x)?,
        weight_dtype: dtype(&w.weight)?,
        scales_dtype: dtype(&w.scales)?,
        biases_dtype: dtype(&w.biases)?,
        bits: w.bits,
        group_size: w.group_size,
        native,
    })
}
