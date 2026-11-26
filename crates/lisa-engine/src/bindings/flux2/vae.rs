use crate::{bindings::BindingDevice, execution::DeviceContext};
use lisa_backend::{ScalarType, TensorAccess};
// Prepared tensors and execution for the model-owned FLUX autoencoder.
use super::weights::Weights;

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

/// Decode Bonsai's original Diffusers VAE in overlapping 128px tiles. Only one
/// tile's activations is live; the final RGB-sized accumulation stays on host.
pub fn decode_tiled<D: BindingDevice + lisa_backend::ManagedDevice>(
    context: &DeviceContext<D>,
    weights: &Weights<D>,
    packed: &D::Tensor,
    progress: &mut dyn FnMut(&str, u32, u32) -> anyhow::Result<()>,
) -> anyhow::Result<Vec<f32>> {
    use super::weights::run;
    use lisa_models::flux::{pipeline, tiles};
    let (h, w) = (packed.dim(2) as usize, packed.dim(3) as usize);
    let mut output = vec![0.0; h * w * 256 * 3];
    let mut count = vec![0.0; h * w * 256];
    let tiles = tiles::tiles(h, w);
    for (i, tile) in tiles.iter().enumerate() {
        progress("Decoding image", i as u32, tiles.len() as u32)?;
        let input = run(context, tile.graph(), &[packed])?;
        let decoded = decode_weights(weights, &input, ScalarType::BF16)?;
        let pixels = run(context, pipeline::pixel_readback(), &[&decoded])?;
        pixels.eval()?;
        tile.blend(h * 16, w * 16, &pixels.read_f32()?, &mut output, &mut count)?;
        drop(pixels);
        drop(decoded);
        drop(input);
        context.device().clear_unused();
    }
    for (i, value) in output.iter_mut().enumerate() {
        *value /= count[i / 3].max(1e-6);
    }
    Ok(output)
}
