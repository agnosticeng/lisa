use crate::{bindings::BindingDevice, execution::DeviceContext};
// Loaded checkpoint bindings for the model-owned FLUX 3 denoiser graph.
use crate::bindings::flux2::weights::{Weights, info};

use lisa_models::flux::{dit3 as model, symbolic::Schema};
use std::path::Path;
pub struct Dit<D: BindingDevice> {
    pub context: DeviceContext<D>,
    weights: Weights<D>,
    schema: Schema,
    geometry: model::Geometry,
}
impl<D: BindingDevice> Dit<D> {
    pub fn load(context: DeviceContext<D>, dir: &Path) -> anyhow::Result<Self> {
        let weights = Weights::load_file(context.clone(), &dir.join("model.safetensors"), |key| {
            !key.contains("image_cond")
        })?;
        let schema = weights.schema()?;
        let geometry = model::Geometry::from_schema(&schema)?;
        Ok(Self {
            context,
            weights,
            schema,
            geometry,
        })
    }
    pub fn forward(
        &self,
        latent: &D::Tensor,
        enc: &D::Tensor,
        time: f32,
        target: f32,
        h: i32,
        w: i32,
    ) -> anyhow::Result<D::Tensor> {
        let program = model::program(
            &self.schema,
            self.geometry,
            info(&self.context, latent)?,
            info(&self.context, enc)?,
            time,
            target,
            h,
            w,
        )?;
        self.weights
            .prepare(program)?
            .run(&[latent, enc], &mut [])?
            .into_one()
            .map_err(anyhow::Error::msg)
    }
}
