use crate::{bindings::BindingDevice, execution::DeviceContext};
use lisa_backend::{ScalarType, TensorAccess};
// Tokenization and loaded weights for model-owned FLUX.2 text conditioning.
use super::weights::Weights;

use lisa_models::flux::text2 as model;
use std::path::Path;
pub fn encode<D: BindingDevice + lisa_backend::ManagedDevice>(
    context: &DeviceContext<D>,
    dir: &Path,
    prompt: &str,
) -> anyhow::Result<D::Tensor> {
    let tokenizer = crate::core::tokenizer::Tokenizer::load(&dir.join("tokenizer"))?;
    let mut ids = tokenizer.encode(&model::prompt(prompt), false)?;
    let real = model::pad(&mut ids);
    let binary = crate::checkpoint::catalog::bonsai::is_checkpoint(dir);
    let mut weights = Weights::load_filtered(
        context.clone(),
        &dir.join(if binary {
            "text_encoder-mlx-4bit"
        } else {
            "text_encoder"
        }),
        |name| !binary || model::conditioning_weight(name.strip_prefix("model.").unwrap_or(name)),
    )?;
    if binary {
        weights.0 = weights
            .0
            .into_iter()
            .map(|(name, value)| {
                (
                    name.strip_prefix("model.").unwrap_or(&name).to_owned(),
                    value,
                )
            })
            .collect();
    }
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
    let out = if binary {
        prepared.run_bounded(&[&ids], &mut [], 24)?
    } else {
        prepared.run(&[&ids], &mut [])?
    }
    .into_one()
    .map_err(anyhow::Error::msg)?;
    out.eval()?;
    Ok(out)
}
