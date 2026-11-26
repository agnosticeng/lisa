use crate::{bindings::BindingDevice, execution::DeviceContext};
// Prepared tensor bindings for the model-owned FLUX 2 denoiser graph.
use crate::bindings::flux2::weights::{Weights, info};

use lisa_models::flux::{dit2 as model, symbolic::Schema};
pub struct Dit<D: BindingDevice> {
    pub context: DeviceContext<D>,
    weights: Weights<D>,
    schema: Schema,
    geometry: model::Geometry,
    bounded: bool,
}
impl<D: BindingDevice + lisa_backend::ManagedDevice> Dit<D> {
    pub fn from_weights(
        context: DeviceContext<D>,
        weights: Weights<D>,
        bounded: bool,
    ) -> anyhow::Result<Self> {
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
