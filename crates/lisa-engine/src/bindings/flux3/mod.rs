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

pub const DEFAULT_STEPS: u32 = 16;
/// Each stage supplies prepared tensors and releases them before the next one.
pub trait StageSource<D: BindingDevice + ManagedDevice> {
    fn required_ram(&self) -> anyhow::Result<usize>;
    fn text_encoder(&self) -> anyhow::Result<text::Encoder<D>>;
    fn denoiser_weights(&self) -> anyhow::Result<Weights<D>>;
    fn decoder_weights(&self) -> anyhow::Result<Weights<D>>;
}
pub struct Pipeline<D: BindingDevice + ManagedDevice> {
    pub context: DeviceContext<D>,
    source: Box<dyn StageSource<D>>,
}

impl<D: BindingDevice + ManagedDevice> Pipeline<D> {
    pub fn from_prepared(context: DeviceContext<D>, source: Box<dyn StageSource<D>>) -> Self {
        Self { context, source }
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
        let needed = self.source.required_ram()?;
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
            let encoder = self.source.text_encoder()?;
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
                dit::Dit::from_weights(self.context.clone(), self.source.denoiser_weights()?)?
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
            let weights = self.source.decoder_weights()?;
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
