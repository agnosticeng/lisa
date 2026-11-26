use crate::{bindings::BindingDevice, execution::DeviceContext};
use lisa_backend::TensorAccess;
// Binding of legacy GLM operation chains to the generic graph executor.
use crate::core::graph::PlanCache;

use lisa_models::glm53::math as model;
#[derive(Clone, Debug, PartialEq)]
enum Key {
    Rms(usize, u32, bool, bool),
    L2(u32, lisa_backend::ScalarType),
    Projection(Vec<i32>, Vec<i32>, lisa_backend::ScalarType),
}
fn run<D: BindingDevice>(
    context: &DeviceContext<D>,
    key: Key,
    inputs: &[&D::Tensor],
) -> anyhow::Result<D::Tensor> {
    context.with_slot(
        "glm53.math.plans",
        || std::cell::RefCell::new(PlanCache::<D, Key>::new(context.device().clone(), 64)),
        |slot| {
            let mut p = slot.borrow_mut();
            let recipe = key.clone();
            p.try_get_or_prepare(key, || match recipe {
                Key::Rms(rank, eps, w, g) => model::rms(rank, f32::from_bits(eps), w, g),
                Key::L2(eps, dtype) => model::l2(f32::from_bits(eps), dtype),
                Key::Projection(shape, wshape, dtype) => model::projection(&shape, &wshape, dtype),
            })?
            .run(inputs, &mut [])?
            .into_one()
            .map_err(anyhow::Error::msg)
        },
    )
}
fn dtype<D: BindingDevice>(
    context: &DeviceContext<D>,
    x: &D::Tensor,
) -> anyhow::Result<lisa_backend::ScalarType> {
    context
        .device()
        .dtype(x)
        .ok_or_else(|| anyhow::anyhow!("unsupported GLM scalar storage"))
}
pub fn l2norm_last<D: BindingDevice>(
    context: &DeviceContext<D>,
    a: &D::Tensor,
    eps: f32,
) -> anyhow::Result<D::Tensor> {
    run(
        context,
        Key::L2(
            eps.to_bits(),
            dtype(context, a).map_err(|e| anyhow::Error::msg(e.to_string()))?,
        ),
        &[a],
    )
    .map_err(|e| anyhow::Error::msg(e.to_string()))
}
pub fn linear_projection<D: BindingDevice>(
    context: &DeviceContext<D>,
    x: &D::Tensor,
    w: &D::Tensor,
) -> anyhow::Result<D::Tensor> {
    run(
        context,
        Key::Projection(x.shape().to_vec(), w.shape().to_vec(), dtype(context, w)?),
        &[x, w],
    )
}
pub fn gated_rms_norm<D: BindingDevice>(
    context: &DeviceContext<D>,
    x: &D::Tensor,
    weight: &D::Tensor,
    gate: &D::Tensor,
    eps: f32,
) -> anyhow::Result<D::Tensor> {
    run(
        context,
        Key::Rms(x.rank(), eps.to_bits(), true, true),
        &[x, weight, gate],
    )
}
pub fn rms_norm_last<D: BindingDevice>(
    context: &DeviceContext<D>,
    x: &D::Tensor,
    weight: Option<&D::Tensor>,
    eps: f32,
) -> anyhow::Result<D::Tensor> {
    let mut a = vec![x];
    a.extend(weight);
    run(
        context,
        Key::Rms(x.rank(), eps.to_bits(), weight.is_some(), false),
        &a,
    )
}
