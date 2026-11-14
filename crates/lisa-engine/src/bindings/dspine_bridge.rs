//! The engine supplies bridge tensors; the model supplies complete arithmetic.
use crate::core::graph::PlanCache;
use crate::{bindings::BindingDevice, execution::DeviceContext};
use lisa_backend::Outputs;
use lisa_models::qwen3_5::dspine::bridge::{Refine, Utility};
#[derive(PartialEq)]
enum Key {
    Context(lisa_models::qwen3_5::dspine::block::Context),
    State(lisa_models::qwen3_5::dspine::state::Plan),
    Block(lisa_models::qwen3_5::dspine::block::Block),
    Bridge(lisa_models::qwen3_5::dspine::bridge::Bridge),
    Refine(Refine),
    Utility(Utility),
}
pub fn bridge<D: BindingDevice>(
    context: &DeviceContext<D>,
    spec: lisa_models::qwen3_5::dspine::bridge::Bridge,
    inputs: &[&D::Tensor],
) -> anyhow::Result<Outputs<D::Tensor>> {
    context.with_slot(
        "dspine_bridge.plans",
        || std::cell::RefCell::new(PlanCache::new(context.device().clone(), 32)),
        |p| {
            p.borrow_mut()
                .try_get_or_prepare(Key::Bridge(spec.clone()), || spec.graph())?
                .run(inputs, &mut [])
        },
    )
}
pub fn refine<D: BindingDevice>(
    context: &DeviceContext<D>,
    spec: Refine,
    inputs: &[&D::Tensor],
) -> anyhow::Result<D::Tensor> {
    context.with_slot(
        "dspine_bridge.plans",
        || std::cell::RefCell::new(PlanCache::new(context.device().clone(), 32)),
        |p| {
            p.borrow_mut()
                .try_get_or_prepare(Key::Refine(spec.clone()), || spec.graph())?
                .run(inputs, &mut [])?
                .into_one()
                .map_err(anyhow::Error::msg)
        },
    )
}
pub fn utility<D: BindingDevice>(
    context: &DeviceContext<D>,
    spec: Utility,
    inputs: &[&D::Tensor],
) -> anyhow::Result<Outputs<D::Tensor>> {
    context.with_slot(
        "dspine_bridge.plans",
        || std::cell::RefCell::new(PlanCache::new(context.device().clone(), 32)),
        |p| {
            p.borrow_mut()
                .try_get_or_prepare(Key::Utility(spec.clone()), || spec.graph())?
                .run(inputs, &mut [])
        },
    )
}
pub fn one<D: BindingDevice>(
    context: &DeviceContext<D>,
    spec: Utility,
    inputs: &[&D::Tensor],
) -> anyhow::Result<D::Tensor> {
    utility(context, spec, inputs)?
        .into_one()
        .map_err(anyhow::Error::msg)
}

pub fn block<D: BindingDevice>(
    context: &DeviceContext<D>,
    spec: lisa_models::qwen3_5::dspine::block::Block,
    inputs: &[&D::Tensor],
) -> anyhow::Result<Outputs<D::Tensor>> {
    context.with_slot(
        "dspine_bridge.plans",
        || std::cell::RefCell::new(PlanCache::new(context.device().clone(), 32)),
        |p| {
            p.borrow_mut()
                .try_get_or_prepare(Key::Block(spec.clone()), || spec.graph())?
                .run(inputs, &mut [])
        },
    )
}

pub fn state<D: BindingDevice>(
    context: &DeviceContext<D>,
    spec: lisa_models::qwen3_5::dspine::state::Plan,
    inputs: &[&D::Tensor],
) -> anyhow::Result<Outputs<D::Tensor>> {
    context.with_slot(
        "dspine_bridge.plans",
        || std::cell::RefCell::new(PlanCache::new(context.device().clone(), 32)),
        |p| {
            p.borrow_mut()
                .try_get_or_prepare(Key::State(spec.clone()), || spec.graph())?
                .run(inputs, &mut [])
        },
    )
}

pub fn context<D: BindingDevice>(
    context: &DeviceContext<D>,
    spec: lisa_models::qwen3_5::dspine::block::Context,
    inputs: &[&D::Tensor],
) -> anyhow::Result<Outputs<D::Tensor>> {
    context.with_slot(
        "dspine_bridge.plans",
        || std::cell::RefCell::new(PlanCache::new(context.device().clone(), 32)),
        |p| {
            p.borrow_mut()
                .try_get_or_prepare(Key::Context(spec.clone()), || spec.graph())?
                .run(inputs, &mut [])
        },
    )
}
