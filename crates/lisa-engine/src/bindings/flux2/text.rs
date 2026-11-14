use crate::{bindings::BindingDevice, execution::DeviceContext};
use lisa_backend::{ScalarType, TensorAccess};
// Tokenization and loaded weights for model-owned FLUX.2 text conditioning.
use super::weights::Weights;

use lisa_models::flux::text2 as model;
use std::path::Path;
pub fn encode<D: BindingDevice>(
    context: &DeviceContext<D>,
    dir: &Path,
    prompt: &str,
) -> anyhow::Result<D::Tensor> {
    let tokenizer = crate::core::tokenizer::Tokenizer::load(&dir.join("tokenizer"))?;
    let mut ids = tokenizer.encode(&model::prompt(prompt), false)?;
    let real = model::pad(&mut ids);
    let weights = Weights::load(context.clone(), &dir.join("text_encoder"))?;
    let schema = weights.schema()?;
    let geometry = model::Geometry::from_schema(&schema)?;
    let program = model::program(&schema, geometry, real)?;
    let ids = crate::bindings::flux2::weights::pull(
        context,
        &ids.iter().flat_map(|x| x.to_ne_bytes()).collect::<Vec<_>>(),
        &[ids.len() as i32],
        ScalarType::U32,
    )?;
    let out = weights
        .prepare(program)?
        .run(&[&ids], &mut [])?
        .into_one()
        .map_err(anyhow::Error::msg)?;
    out.eval()?;
    Ok(out)
}
