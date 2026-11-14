//! Prepared retrieval and auxiliary-head plan bindings.
use crate::core::graph::PlanCache;
use crate::{bindings::BindingDevice, execution::DeviceContext};
use lisa_backend::Outputs;
use lisa_models::qwen3_5::dspine::retrieval::{Head, Rescore};
#[derive(PartialEq)]
enum Key {
    Rescore(Rescore),
    Candidates(Rescore),
    Allocate(Head),
    Chunk(Head, i32, i32),
}
pub fn rescore<D: BindingDevice>(
    context: &DeviceContext<D>,
    spec: Rescore,
    inputs: &[&D::Tensor],
    candidates: bool,
) -> anyhow::Result<D::Tensor> {
    let key = if candidates {
        Key::Candidates(spec.clone())
    } else {
        Key::Rescore(spec.clone())
    };
    context.with_slot(
        "dspine_retrieval_plans.plans",
        || std::cell::RefCell::new(PlanCache::new(context.device().clone(), 16)),
        |p| {
            p.borrow_mut()
                .try_get_or_prepare(key, || {
                    if candidates {
                        spec.candidates()
                    } else {
                        spec.graph()
                    }
                })?
                .run(inputs, &mut [])?
                .into_one()
                .map_err(anyhow::Error::msg)
        },
    )
}
pub fn allocate<D: BindingDevice>(
    context: &DeviceContext<D>,
    spec: Head,
) -> anyhow::Result<Outputs<D::Tensor>> {
    context.with_slot(
        "dspine_retrieval_plans.plans",
        || std::cell::RefCell::new(PlanCache::new(context.device().clone(), 16)),
        |p| {
            p.borrow_mut()
                .try_get_or_prepare(Key::Allocate(spec.clone()), || spec.allocate())?
                .run(&[], &mut [])
        },
    )
}
pub fn chunk<D: BindingDevice>(
    context: &DeviceContext<D>,
    spec: Head,
    start: i32,
    end: i32,
    inputs: &[&D::Tensor],
) -> anyhow::Result<()> {
    context.with_slot(
        "dspine_retrieval_plans.plans",
        || std::cell::RefCell::new(PlanCache::new(context.device().clone(), 16)),
        |p| {
            p.borrow_mut()
                .try_get_or_prepare(Key::Chunk(spec.clone(), start, end), || {
                    spec.chunk(start, end)
                })?
                .run(inputs, &mut [])
        },
    )?;
    Ok(())
}
