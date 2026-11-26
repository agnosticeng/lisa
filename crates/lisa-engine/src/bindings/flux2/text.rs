use crate::{bindings::BindingDevice, execution::DeviceContext};
use lisa_backend::{ScalarType, TensorAccess};
// Prepared tensors and token IDs for model-owned FLUX.2 text conditioning.
use super::weights::Weights;

use lisa_models::flux::text2 as model;
pub fn encode<D: BindingDevice + lisa_backend::ManagedDevice>(
    context: &DeviceContext<D>,
    weights: &Weights<D>,
    ids: &[u32],
    real: usize,
    bounded: bool,
) -> anyhow::Result<D::Tensor> {
    let schema = weights.schema()?;
    let geometry = model::Geometry::from_schema(&schema)?;
    let program = model::program(&schema, geometry, real)?;
    let ids = crate::bindings::flux2::weights::pull(
        context,
        &ids.iter().flat_map(|x| x.to_ne_bytes()).collect::<Vec<_>>(),
        &[ids.len() as i32],
        ScalarType::U32,
    )?;
    let prepared = weights.prepare(program)?;
    let out = if bounded {
        prepared.run_bounded(&[&ids], &mut [], 24)?
    } else {
        prepared.run(&[&ids], &mut [])?
    }
    .into_one()
    .map_err(anyhow::Error::msg)?;
    out.eval()?;
    Ok(out)
}
