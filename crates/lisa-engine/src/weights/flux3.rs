//! Checkpoint adaptation and staged residency for FLUX 3.
use crate::{
    bindings::{
        BindingDevice,
        flux2::weights::{Weights, run},
        flux3,
    },
    execution::DeviceContext,
};
use anyhow::ensure;
use lisa_backend::{ManagedDevice, TensorAccess};
use lisa_models::flux::{pipeline as model, text3};
use std::path::{Path, PathBuf};

pub use crate::checkpoint::catalog::{flux3::REPO, is_flux3 as is_checkpoint};
struct Source<D: BindingDevice + ManagedDevice> {
    context: DeviceContext<D>,
    dir: PathBuf,
}
impl<D: BindingDevice + ManagedDevice> flux3::StageSource<D> for Source<D> {
    fn required_ram(&self) -> anyhow::Result<usize> {
        usize::try_from(std::fs::metadata(self.dir.join("model.safetensors"))?.len())?
            .checked_mul(2)
            .and_then(|bytes| bytes.checked_add(8usize << 30))
            .ok_or_else(|| anyhow::anyhow!("image memory estimate overflow"))
    }
    fn text_encoder(&self) -> anyhow::Result<flux3::text::Encoder<D>> {
        load_text(self.context.clone(), &self.dir)
    }
    fn denoiser_weights(&self) -> anyhow::Result<Weights<D>> {
        denoiser_weights(self.context.clone(), &self.dir)
    }
    fn decoder_weights(&self) -> anyhow::Result<Weights<D>> {
        decoder_weights(&self.context, &self.dir)
    }
}
pub fn load<D: BindingDevice + ManagedDevice>(
    context: DeviceContext<D>,
    dir: &Path,
) -> anyhow::Result<flux3::Pipeline<D>> {
    ensure!(
        is_checkpoint(dir),
        "expected the distilled FLUX 3 Image Klein native checkpoint"
    );
    let config: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.join("config.json"))?)?;
    model::validate3(&config)?;
    for name in [
        "model.safetensors",
        "ae.safetensors",
        "text_encoder/config.json",
        "text_encoder/tokenizer.json",
    ] {
        ensure!(
            dir.join(name).is_file(),
            "missing FLUX 3 file {name}; complete the model download"
        );
    }
    let source = Box::new(Source {
        context: context.clone(),
        dir: dir.into(),
    });
    Ok(flux3::Pipeline::from_prepared(context, source))
}

pub fn load_text<D: BindingDevice>(
    context: DeviceContext<D>,
    dir: &Path,
) -> anyhow::Result<flux3::text::Encoder<D>> {
    let dir = dir.join("text_encoder");
    let config = serde_json::from_slice(&std::fs::read(dir.join("config.json"))?)?;
    let loaded = super::flux::load_filtered(context.clone(), &dir, |key| {
        key.starts_with("model.language_model.")
    })?;
    let weights = Weights::from_map(
        context.clone(),
        loaded
            .0
            .into_iter()
            .map(|(key, value)| {
                (
                    key.trim_start_matches("model.language_model.").to_owned(),
                    value,
                )
            })
            .collect(),
    );
    let schema = weights.schema()?;
    text3::validate(&config, &schema)?;
    let tokenizer = crate::checkpoint::tokenizer::load(&dir)?;
    Ok(flux3::text::Encoder::from_prepared(
        context, weights, schema, tokenizer,
    ))
}

pub fn denoiser_weights<D: BindingDevice>(
    context: DeviceContext<D>,
    dir: &Path,
) -> anyhow::Result<Weights<D>> {
    super::flux::load_file(context, &dir.join("model.safetensors"), |key| {
        !key.contains("image_cond")
    })
}

/// Rename the original f32 decoder into the shared VAE operation layout.
/// Only convolution storage is transposed (OIHW -> OHWI); payload precision is retained.
pub fn decoder_weights<D: BindingDevice>(
    context: &DeviceContext<D>,
    dir: &Path,
) -> anyhow::Result<Weights<D>> {
    let source = super::flux::load_file(context.clone(), &dir.join("ae.safetensors"), |k| {
        k.starts_with("decoder.") || k.starts_with("bn.")
    })?;
    let mut out = std::collections::HashMap::new();
    for (key, value) in source.0 {
        let (name, preparation) = model::decoder_weight(&key, value.shape())?;
        let value = match preparation {
            Some(graph) => run(context, graph, &[&value])?,
            None => value,
        };
        out.insert(name, value);
    }
    Ok(Weights::from_map(context.clone(), out))
}
