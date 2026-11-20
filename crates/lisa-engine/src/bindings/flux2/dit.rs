use crate::{bindings::BindingDevice, execution::DeviceContext};
// Loaded checkpoint bindings for the model-owned FLUX 2 denoiser graph.
use crate::bindings::flux2::weights::{Weights, info};

use lisa_models::flux::{dit2 as model, symbolic::Schema};
use std::path::Path;
pub struct Dit<D: BindingDevice> {
    pub context: DeviceContext<D>,
    weights: Weights<D>,
    schema: Schema,
    geometry: model::Geometry,
    bounded: bool,
}
impl<D: BindingDevice + lisa_backend::ManagedDevice> Dit<D> {
    pub fn load(context: DeviceContext<D>, dir: &Path) -> anyhow::Result<Self> {
        let bounded = crate::checkpoint::catalog::bonsai::is_checkpoint(dir);
        let mut weights = if bounded {
            Weights::load_file(
                context.clone(),
                &dir.join("transformer-packed-mflux/diffusion_pytorch_model.safetensors"),
                |_| true,
            )?
        } else {
            Weights::load(context.clone(), &dir.join("transformer"))?
        };
        weights.0 = weights
            .0
            .into_iter()
            .map(|(name, value)| {
                (
                    name.replace(
                        "time_guidance_embed.timestep_embedder.",
                        "time_guidance_embed.",
                    )
                    .replace(".to_out.0.", ".to_out."),
                    value,
                )
            })
            .collect();
        let schema = weights.schema()?;
        let geometry = model::Geometry::from_schema(&schema)?;
        Ok(Self {
            context,
            weights,
            schema,
            geometry,
            bounded,
        })
    }
    pub fn forward(
        &self,
        latent: &D::Tensor,
        enc: &D::Tensor,
        time: f32,
        h: i32,
        w: i32,
    ) -> anyhow::Result<D::Tensor> {
        let program = model::program(
            &self.schema,
            self.geometry,
            info(&self.context, latent)?,
            info(&self.context, enc)?,
            time,
            h,
            w,
        )?;
        let prepared = self.weights.prepare(program)?;
        let output = if self.bounded {
            prepared.run_bounded(&[latent, enc], &mut [], 24)?
        } else {
            prepared.run(&[latent, enc], &mut [])?
        };
        output.into_one().map_err(anyhow::Error::msg)
    }
}
