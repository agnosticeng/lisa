use crate::{bindings::BindingDevice, execution::DeviceContext};
use lisa_backend::{ScalarType, TensorAccess};
// Checkpoint bindings and execution for the model-owned FLUX autoencoder.
use super::weights::Weights;

use std::path::Path;
pub fn decode<D: BindingDevice>(
    context: &DeviceContext<D>,
    dir: &Path,
    packed: &D::Tensor,
) -> anyhow::Result<D::Tensor> {
    let weights = Weights::load(context.clone(), &dir.join("vae"))?;
    decode_weights(&weights, packed, ScalarType::BF16)
}
pub fn decode_weights<D: BindingDevice>(
    weights: &Weights<D>,
    packed: &D::Tensor,
    dtype: ScalarType,
) -> anyhow::Result<D::Tensor> {
    anyhow::ensure!(
        packed.rank() == 4 && packed.dim(0) == 1 && packed.dim(1) == 128,
        "invalid packed VAE latent"
    );
    let dtype = match dtype {
        ScalarType::BF16 => ScalarType::BF16,
        ScalarType::F32 => ScalarType::F32,
        _ => anyhow::bail!("unsupported VAE output dtype"),
    };
    let schema = weights
        .0
        .iter()
        .map(|(k, v)| (k.clone(), v.shape().to_vec()))
        .collect();
    let quantized = weights
        .0
        .iter()
        .filter(|(_, v)| weights.2.device().dtype(v) == Some(ScalarType::U32))
        .map(|(k, _)| k.clone())
        .collect();
    let plan =
        lisa_models::flux::vae::decoder(&schema, &quantized, packed.dim(2), packed.dim(3), dtype)?;
    let prepared =
        crate::core::program::prepare(weights.2.device().clone(), plan, |name| weights.get(name))?;
    let output = prepared
        .run(&[packed], &mut [])?
        .into_one()
        .map_err(anyhow::Error::msg)?;
    output.eval()?;
    Ok(output)
}
