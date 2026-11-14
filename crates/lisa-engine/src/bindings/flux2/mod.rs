use crate::{bindings::BindingDevice, execution::DeviceContext};
use lisa_backend::{ManagedDevice, ScalarType, TensorAccess};
// Native FLUX.2 Klein 4B text-to-image pipeline on LISA's Metal runtime.
// Architecture and operation order: mlx-serve 6c430614 (MIT), flux2 plugin.
pub mod dit;
pub mod text;
pub mod vae;
pub mod weights;
use anyhow::ensure;

use lisa_models::flux::pipeline::{self as model, Family};
use std::path::{Path, PathBuf};
use weights::run;

pub const DEFAULT_MODEL: &str = "mlx-community/flux2-klein-4b-4bit";
pub const DEFAULT_STEPS: u32 = 4;
pub use lisa_models::flux::pipeline::Options;
pub struct Pipeline<D: BindingDevice + ManagedDevice> {
    pub context: DeviceContext<D>,
    dir: PathBuf,
}
impl<D: BindingDevice + ManagedDevice> Pipeline<D> {
    pub fn load(context: DeviceContext<D>, dir: &Path) -> anyhow::Result<Self> {
        ensure!(
            is_checkpoint(dir),
            "expected an MLX FLUX.2 Klein checkpoint with text_encoder, transformer, tokenizer and vae"
        );
        Ok(Self {
            context,
            dir: dir.into(),
        })
    }
    /// Each stage releases its weights before the next stage loads. The caller
    /// must release any language model before entering this separate pipeline.
    pub fn generate(
        &self,
        prompt: &str,
        options: &Options,
        progress: &mut dyn FnMut(&str, u32, u32) -> anyhow::Result<()>,
    ) -> anyhow::Result<Vec<u8>> {
        options.validate()?;
        let steps = options.steps.unwrap_or(DEFAULT_STEPS);
        ensure!(
            !prompt.trim().is_empty() && prompt.len() <= 16_384,
            "prompt must contain 1..=16384 bytes"
        );
        // Leave room for convolution intermediates, attention, and allocator
        // overlap. Check both the configured cap and current system headroom.
        let needed = 8usize << 30;
        ensure!(
            self.context
                .device()
                .memory_usage()?
                .active
                .saturating_add(needed)
                < crate::memory::ram_budget(),
            "insufficient RAM budget for image generation"
        );
        if let Some(available) = crate::memory::available_ram() {
            ensure!(
                available >= needed,
                "insufficient available memory for image generation"
            );
        }
        progress("Encoding prompt", 0, steps)?;
        let enc = {
            let _trace = self.context.trace("image.text_encoder");
            text::encode(&self.context, &self.dir, prompt)?
        };
        let (h, w) = ((options.height / 16) as i32, (options.width / 16) as i32);
        let noise = model::normal_noise((128 * h * w) as usize, options.seed);
        let noise = weights::pull(
            &self.context,
            &noise
                .iter()
                .flat_map(|x| x.to_ne_bytes())
                .collect::<Vec<_>>(),
            &model::noise_shape(Family::Klein2, h, w),
            ScalarType::F32,
        )?;
        let mut latents = run(
            &self.context,
            model::initialize(Family::Klein2, h, w)?,
            &[&noise],
        )?;
        {
            let model = {
                let _trace = self.context.trace("image.denoiser.load");
                dit::Dit::load(self.context.clone(), &self.dir)?
            };
            let sigma = model::schedule2((h * w) as usize, steps as usize);
            for i in 0..steps as usize {
                let _trace = self.context.trace("image.denoiser.step");
                progress("Generating", i as u32, steps)?;
                let velocity = model.forward(&latents, &enc, sigma[i] * 1000.0, h, w)?;
                latents = run(
                    &self.context,
                    model::step(Family::Klein2, sigma[i + 1] - sigma[i]),
                    &[&latents, &velocity],
                )?;
                latents.eval()?;
            }
        }
        drop(enc);
        progress("Decoding image", steps, steps)?;
        let pixels = {
            let _trace = self.context.trace("image.vae");
            vae::decode(
                &self.context,
                &self.dir,
                &run(&self.context, model::decoder_input(h, w), &[&latents])?,
            )?
        };
        let pixels = run(&self.context, model::pixel_readback(), &[&pixels])?;
        pixels.eval()?;
        let values = pixels.read_f32()?;
        let rgb = model::rgb(Family::Klein2, &values)?;
        let image = image::RgbImage::from_raw(options.width, options.height, rgb)
            .ok_or_else(|| anyhow::anyhow!("VAE output dimensions do not match request"))?;
        let mut bytes = std::io::Cursor::new(Vec::new());
        image.write_to(&mut bytes, image::ImageFormat::Png)?;
        Ok(bytes.into_inner())
    }
}
pub use crate::checkpoint::catalog::is_flux2 as is_checkpoint;
