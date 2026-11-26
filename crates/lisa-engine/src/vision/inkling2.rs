//! Execute the prepared F32 image law over tensors only. The loading adapter
//! owns checkpoint access, original payloads and request-bounded residency.
use crate::{bindings::BindingDevice, core::graph::PreparedGraph, execution::DeviceContext};
use anyhow::{Result, ensure};
use lisa_backend::ManagedDevice;

pub(crate) fn execute<D: BindingDevice + ManagedDevice>(
    context: &DeviceContext<D>,
    plans: Vec<PreparedGraph<D>>,
    patches: &D::Tensor,
    positions: &D::Tensor,
    cos: &D::Tensor,
    sin: &D::Tensor,
) -> Result<D::Tensor> {
    ensure!(
        plans.len() == 25,
        "invalid prepared Inkling vision plan count"
    );
    let mut plans = plans.into_iter();
    let mut x = plans
        .next()
        .unwrap()
        .run(&[patches, positions], &mut [])?
        .into_one()
        .map_err(anyhow::Error::msg)?;
    for layer in plans.by_ref().take(23) {
        x = layer
            .run(&[&x, cos, sin], &mut [])?
            .into_one()
            .map_err(anyhow::Error::msg)?;
        context.device().submit(true)?;
        context.device().trim_unused();
    }
    let out = plans
        .next()
        .unwrap()
        .run(&[&x], &mut [])?
        .into_one()
        .map_err(anyhow::Error::msg)?;
    context.device().submit(true)?;
    Ok(out)
}
