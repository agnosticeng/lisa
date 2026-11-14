//! Checkpoint and request bindings for the model-owned GLM vision program.
use super::weights::{get, load_weights};
use crate::core::{graph::PreparedGraph, program};
use crate::{bindings::BindingDevice, execution::DeviceContext};
use anyhow::Context;
pub use definition::Config;
use lisa_backend::{HostTensor, ManagedDevice, ScalarType, TensorAccess};
use lisa_models::{program::Schema, vision::glm as definition};
use serde_json::Value;
use std::{cell::RefCell, collections::HashMap, path::Path};
pub struct Encoder<D: BindingDevice + ManagedDevice> {
    pub context: DeviceContext<D>,
    pub config: Config,
    pub weights: HashMap<String, D::Tensor>,
    pub schema: Schema,
    pub prepared: RefCell<Option<(usize, usize, PreparedGraph<D>)>>,
}
impl<D: BindingDevice + ManagedDevice> Encoder<D> {
    pub fn load(context: DeviceContext<D>, dir: &Path) -> anyhow::Result<Self> {
        let root: Value = serde_json::from_slice(&std::fs::read(dir.join("config.json"))?)?;
        let config: Config = serde_json::from_value(
            root.get("vision_config")
                .context("checkpoint has no vision_config")?
                .clone(),
        )?;
        config.validate()?;
        super::admit_scratch(&context, 2usize << 30)?;
        Self::from_weights(context.clone(), config, load_weights(&context, dir)?)
    }
    pub fn from_weights(
        context: DeviceContext<D>,
        config: Config,
        mut weights: HashMap<String, D::Tensor>,
    ) -> anyhow::Result<Self> {
        let schema = weights
            .iter()
            .map(|(k, v)| (k.clone(), v.shape().to_vec()))
            .collect();
        for (name, plan) in definition::prepare_weights(&config, &schema)? {
            let prepared =
                program::prepare(context.device().clone(), plan, |name| get(&weights, name))?;
            let output = prepared
                .run(&[], &mut [])?
                .into_one()
                .map_err(anyhow::Error::msg)?;
            weights.insert(name, output);
        }
        let schema = weights
            .iter()
            .map(|(k, v)| (k.clone(), v.shape().to_vec()))
            .collect();
        Ok(Self {
            context,
            config,
            weights,
            schema,
            prepared: RefCell::new(None),
        })
    }
    pub fn encode(&self, image: &image::RgbImage) -> anyhow::Result<(D::Tensor, usize, usize)> {
        let pixels = super::processor::glm(
            image,
            self.config.patch_size,
            self.config.temporal_patch_size,
            self.config.spatial_merge_size,
        )?;
        self.encode_pixels(&pixels.values, pixels.height, pixels.width)
    }
    pub fn encode_pixels(
        &self,
        values: &[f32],
        gh: usize,
        gw: usize,
    ) -> anyhow::Result<(D::Tensor, usize, usize)> {
        let (n, width) = self.config.geometry(gh, gw, values.len())?;
        super::admit_scratch(&self.context, self.config.scratch_bytes(n as usize)?)?;
        let mut cache = self.prepared.borrow_mut();
        if cache.as_ref().is_none_or(|(h, w, _)| *h != gh || *w != gw) {
            let plan = definition::encoder(&self.config, &self.schema, gh, gw)?;
            let prepared = program::prepare(self.context.device().clone(), plan, |name| {
                get(&self.weights, name)
            })?;
            *cache = Some((gh, gw, prepared));
        }
        let input = self.context.device().pull(
            HostTensor {
                bytes: &values
                    .iter()
                    .flat_map(|v| v.to_ne_bytes())
                    .collect::<Vec<_>>(),
                shape: &[n, width],
                dtype: ScalarType::F32,
            }
            .into(),
            Default::default(),
        )?;
        let output = cache
            .as_ref()
            .unwrap()
            .2
            .run(&[&input], &mut [])?
            .into_one()
            .map_err(anyhow::Error::msg)?;
        output.eval()?;
        Ok((output, gh, gw))
    }
}
