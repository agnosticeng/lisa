//! Affine table inputs bound to model-owned lookup and preparation graphs.
use crate::core::graph::PlanCache;
use crate::{bindings::BindingDevice, execution::DeviceContext};
#[derive(PartialEq)]
enum Key {
    Linear(i32, i32),
    Lookup(i32, i32),
    Requantize((i32, i32), (i32, i32)),
}
pub fn lookup<D: BindingDevice>(
    context: &DeviceContext<D>,
    ids: &D::Tensor,
    weight: &D::Tensor,
    scales: &D::Tensor,
    biases: &D::Tensor,
    group_size: i32,
    bits: i32,
) -> anyhow::Result<D::Tensor> {
    context.with_slot(
        "affine.plans",
        || std::cell::RefCell::new(PlanCache::new(context.device().clone(), 16)),
        |p| {
            p.borrow_mut()
                .get_or_prepare(Key::Lookup(group_size, bits), || {
                    lisa_models::embedding::affine(group_size, bits)
                })?
                .run(&[ids, weight, scales, biases], &mut [])?
                .into_one()
                .map_err(anyhow::Error::msg)
        },
    )
}
pub fn requantize<D: BindingDevice>(
    context: &DeviceContext<D>,
    weight: &D::Tensor,
    scales: &D::Tensor,
    biases: &D::Tensor,
    source: (i32, i32),
    target: (i32, i32),
) -> anyhow::Result<(D::Tensor, D::Tensor, D::Tensor)> {
    let output = context.with_slot(
        "affine.plans",
        || std::cell::RefCell::new(PlanCache::new(context.device().clone(), 16)),
        |p| {
            p.borrow_mut()
                .get_or_prepare(Key::Requantize(source, target), || {
                    lisa_models::embedding::requantize(source, target)
                })?
                .run(&[weight, scales, biases], &mut [])
        },
    )?;
    let mut out = output.into_iter();
    Ok((
        out.next().unwrap(),
        out.next().unwrap(),
        out.next().unwrap(),
    ))
}

pub fn linear<D: BindingDevice>(
    context: &DeviceContext<D>,
    x: &D::Tensor,
    p: &crate::weights::affine::QuantizedLinear<D::Tensor>,
) -> anyhow::Result<D::Tensor> {
    context.with_slot(
        "affine.plans",
        || std::cell::RefCell::new(PlanCache::new(context.device().clone(), 16)),
        |cache| {
            cache
                .borrow_mut()
                .get_or_prepare(Key::Linear(p.group_size, p.bits), || {
                    lisa_models::embedding::linear(p.group_size, p.bits)
                })?
                .run(&[x, &p.weight, &p.scales, &p.biases], &mut [])?
                .into_one()
                .map_err(anyhow::Error::msg)
        },
    )
}
