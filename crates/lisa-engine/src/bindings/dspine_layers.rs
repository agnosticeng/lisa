//! Runtime bindings for model-owned draft projection and attention graphs.
use crate::attention_cache::FullAttentionCache;
use crate::core::graph::PlanCache;
use crate::{bindings::BindingDevice, execution::DeviceContext};
use lisa_backend::Outputs;
use lisa_backend::TensorAccess;
use lisa_models::qwen3_5::dspine::layers;
use lisa_models::qwen3_5::dspine::layers::Attention;
#[derive(PartialEq)]
enum Key {
    Linear(lisa_models::qwen3_5::dspine::layers::Linear),
    Prepare(usize, bool),
    Attention(Attention),
}
fn one<T>(output: Outputs<T>) -> anyhow::Result<T> {
    output.into_one().map_err(anyhow::Error::msg)
}
pub fn linear<D: BindingDevice>(
    context: &DeviceContext<D>,
    spec: lisa_models::qwen3_5::dspine::layers::Linear,
    inputs: &[&D::Tensor],
) -> anyhow::Result<D::Tensor> {
    context.with_slot(
        "dspine_layers.plans",
        || std::cell::RefCell::new(PlanCache::new(context.device().clone(), 32)),
        |p| {
            one(p
                .borrow_mut()
                .try_get_or_prepare(Key::Linear(spec.clone()), || spec.graph())?
                .run(inputs, &mut [])?)
        },
    )
}
pub fn prepare<D: BindingDevice>(
    context: &DeviceContext<D>,
    weight: &D::Tensor,
    native_layout: bool,
) -> anyhow::Result<D::Tensor> {
    context.with_slot(
        "dspine_layers.plans",
        || std::cell::RefCell::new(PlanCache::new(context.device().clone(), 32)),
        |p| {
            one(p
                .borrow_mut()
                .try_get_or_prepare(Key::Prepare(weight.rank(), native_layout), || {
                    layers::prepare_weight(weight.rank(), native_layout)
                })?
                .run(&[weight], &mut [])?)
        },
    )
}

pub fn attention<D: BindingDevice>(
    context: &DeviceContext<D>,
    spec: Attention,
    inputs: &[&D::Tensor],
    cache: &mut FullAttentionCache<D>,
) -> anyhow::Result<Option<D::Tensor>> {
    let (end, capacity, _) = spec.cache.layout().map_err(anyhow::Error::msg)?;
    let mut inputs = inputs.to_vec();
    if spec.cache.previous_batch.is_some() {
        inputs.push(
            cache
                .keys
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("missing draft keys"))?,
        );
        inputs.push(
            cache
                .values
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("missing draft values"))?,
        );
    }
    let output = context.with_slot(
        "dspine_layers.plans",
        || std::cell::RefCell::new(PlanCache::new(context.device().clone(), 32)),
        |p| {
            p.borrow_mut()
                .try_get_or_prepare(Key::Attention(spec.clone()), || spec.graph())?
                .run(&inputs, &mut [])
        },
    )?;
    let mut output = output.into_iter();
    let activation = if spec.context_only {
        None
    } else {
        output.next()
    };
    let keys = output
        .next()
        .ok_or_else(|| anyhow::anyhow!("missing planned keys"))?;
    let values = output
        .next()
        .ok_or_else(|| anyhow::anyhow!("missing planned values"))?;
    cache.keys = Some(keys);
    cache.values = Some(values);
    cache.capacity = capacity;
    cache.offset = if spec.context_only {
        end
    } else {
        spec.cache.offset
    };
    Ok(activation)
}

/// Query the exact prepared-operation capability.
pub fn small_rows<D: BindingDevice>(context: &DeviceContext<D>) -> bool {
    context.with_slot(
        "dspine.small_rows",
        || {
            context
                .device()
                .capabilities()
                .iter()
                .any(|c| c.kind == lisa_backend::KernelKind::Bf16SmallRowLinear)
        },
        |v| *v,
    )
}
