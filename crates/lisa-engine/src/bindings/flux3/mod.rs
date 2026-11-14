use crate::{bindings::BindingDevice, execution::DeviceContext};
use lisa_backend::{ManagedDevice, ScalarType, TensorAccess};
// Native FLUX 3 Image Klein text-to-image inference.
// Adapted from BFL's Apache-2.0 reference bfc8889a4497f846416f253a3972a1941ac698b9.
// Copyright 2026 Black Forest Labs, Inc. See LICENSE for the full notice.
pub mod dit;
pub mod text;
use crate::bindings::flux2::weights::run;
use crate::bindings::flux2::{Options, vae, weights::Weights};
use anyhow::ensure;

use lisa_models::flux::pipeline::{self as model, Family};
use model::schedule3 as schedule;
use std::path::{Path, PathBuf};

pub const DEFAULT_STEPS: u32 = 16;
pub struct Pipeline<D: BindingDevice + ManagedDevice> {
    pub context: DeviceContext<D>,
    dir: PathBuf,
}

pub use crate::checkpoint::catalog::is_flux3 as is_checkpoint;
impl<D: BindingDevice + ManagedDevice> Pipeline<D> {
    pub fn load(context: DeviceContext<D>, dir: &Path) -> anyhow::Result<Self> {
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
        Ok(Self {
            context,
            dir: dir.into(),
        })
    }
    pub fn generate(
        &self,
        prompt: &str,
        options: &Options,
        progress: &mut dyn FnMut(&str, u32, u32) -> anyhow::Result<()>,
    ) -> anyhow::Result<Vec<u8>> {
        options.validate()?;
        ensure!(
            !prompt.trim().is_empty() && prompt.len() <= 16384,
            "prompt must contain 1..=16384 bytes"
        );
        let steps = options.steps.unwrap_or(DEFAULT_STEPS);
        // Original mapped weights, their cached dense layouts, and bounded
        // attention/VAE scratch. This is a model estimate, never a machine cap.
        let needed = usize::try_from(std::fs::metadata(self.dir.join("model.safetensors"))?.len())?
            .checked_mul(2)
            .and_then(|bytes| bytes.checked_add(8usize << 30))
            .ok_or_else(|| anyhow::anyhow!("image memory estimate overflow"))?;
        ensure!(
            self.context
                .device()
                .memory_usage()?
                .active
                .saturating_add(needed)
                < crate::memory::ram_budget(),
            "insufficient RAM budget for FLUX 3"
        );
        if let Some(available) = crate::memory::available_ram() {
            ensure!(available >= needed, "insufficient available RAM for FLUX 3");
        }
        let enc = {
            let _trace = self.context.trace("image.text_encoder");
            progress("Loading text encoder", 0, steps)?;
            let encoder = text::Encoder::load(self.context.clone(), &self.dir)?;
            let caption = if options.prompt_is_caption {
                prompt.to_owned()
            } else {
                let _trace = self.context.trace("image.prompt_expansion");
                encoder.expand(prompt, options.width, options.height, &mut || {
                    progress("Expanding prompt", 0, steps)
                })?
            };
            progress("Encoding prompt", 0, steps)?;
            encoder.encode(&caption)?
        };
        self.context.device().clear_unused();
        let (h, w) = ((options.height / 16) as i32, (options.width / 16) as i32);
        // Noise is bf16-rounded; the recurrent latent state and Euler update remain f32.
        let noise = model::normal_noise((h * w * 128) as usize, options.seed);
        let noise = crate::bindings::flux2::weights::pull(
            &self.context,
            &noise
                .iter()
                .flat_map(|x| x.to_ne_bytes())
                .collect::<Vec<_>>(),
            &model::noise_shape(Family::Klein3, h, w),
            ScalarType::F32,
        )?;
        let mut latent = run(
            &self.context,
            model::initialize(Family::Klein3, h, w)?,
            &[&noise],
        )?;
        {
            progress("Loading image model", 0, steps)?;
            let model = {
                let _trace = self.context.trace("image.denoiser.load");
                dit::Dit::load(self.context.clone(), &self.dir)?
            };
            let times = schedule(steps);
            for i in 0..steps as usize {
                progress("Generating", i as u32, steps)?;
                let _trace = self.context.trace("image.denoiser.step");
                let velocity = model.forward(&latent, &enc, times[i], times[i + 1], h, w)?;
                latent = run(
                    &self.context,
                    model::step(Family::Klein3, times[i + 1] - times[i]),
                    &[&latent, &velocity],
                )?;
                latent.eval()?;
            }
        }
        drop(enc);
        self.context.device().clear_unused();
        progress("Decoding image", steps, steps)?;
        let pixels = {
            let _trace = self.context.trace("image.vae");
            let weights = decoder_weights(&self.context, &self.dir)?;
            vae::decode_weights(
                &weights,
                &run(&self.context, model::decoder_input(h, w), &[&latent])?,
                ScalarType::F32,
            )?
        };
        let pixels = run(&self.context, model::pixel_readback(), &[&pixels])?;
        pixels.eval()?;
        let values = pixels.read_f32()?;
        let rgb = model::rgb(Family::Klein3, &values)?;
        let image = image::RgbImage::from_raw(options.width, options.height, rgb)
            .ok_or_else(|| anyhow::anyhow!("FLUX 3 output shape mismatch"))?;
        let mut out = std::io::Cursor::new(Vec::new());
        image.write_to(&mut out, image::ImageFormat::Png)?;
        Ok(out.into_inner())
    }
}

/// Rename the original f32 decoder into the shared VAE operation layout.
/// Only convolution storage is transposed (OIHW -> OHWI); payload precision is retained.
pub fn decoder_weights<D: BindingDevice>(
    context: &DeviceContext<D>,
    dir: &Path,
) -> anyhow::Result<Weights<D>> {
    let source = Weights::load_file(context.clone(), &dir.join("ae.safetensors"), |k| {
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

pub use crate::checkpoint::catalog::flux3::REPO;
