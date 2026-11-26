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
use weights::run;

pub const DEFAULT_MODEL: &str = "mlx-community/flux2-klein-4b-4bit";
pub const DEFAULT_STEPS: u32 = 4;
pub use lisa_models::flux::pipeline::Options;
/// Stage acquisition returns imported tensors without exposing their container.
pub trait StageSource<D: BindingDevice + ManagedDevice> {
    fn tokenizer(&self) -> anyhow::Result<crate::core::tokenizer::Tokenizer>;
    fn text_weights(&self) -> anyhow::Result<weights::Weights<D>>;
    fn denoiser_weights(&self) -> anyhow::Result<weights::Weights<D>>;
    fn decoder_weights(&self) -> anyhow::Result<weights::Weights<D>>;
}
pub struct Policy {
    pub bounded: bool,
    pub required_ram: usize,
}
pub struct Pipeline<D: BindingDevice + ManagedDevice> {
    pub context: DeviceContext<D>,
    source: Box<dyn StageSource<D>>,
    policy: Policy,
}
impl<D: BindingDevice + ManagedDevice> Pipeline<D> {
    pub fn from_prepared(
        context: DeviceContext<D>,
        source: Box<dyn StageSource<D>>,
        policy: Policy,
    ) -> Self {
        Self {
            context,
            source,
            policy,
        }
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
        let bounded = self.policy.bounded;
        if bounded {
            self.context.device().clear_unused();
            ensure!(
                options.width <= 512 && options.height <= 512,
                "Bonsai currently supports up to 512 × 512 pixels"
            );
        }
        let needed = self.policy.required_ram;
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
            let tokenizer = self.source.tokenizer()?;
            let mut ids = tokenizer.encode(&lisa_models::flux::text2::prompt(prompt), false)?;
            let real = lisa_models::flux::text2::pad(&mut ids);
            let weights = self.source.text_weights()?;
            text::encode(&self.context, &weights, &ids, real, bounded)?
        };
        if bounded {
            self.context.device().clear_unused();
        }
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
                dit::Dit::from_weights(
                    self.context.clone(),
                    self.source.denoiser_weights()?,
                    bounded,
                )?
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
        if bounded {
            self.context.device().clear_unused();
        }
        progress("Decoding image", steps, steps)?;
        let values = if bounded {
            let _trace = self.context.trace("image.vae.tiled");
            let packed = run(&self.context, model::decoder_input(h, w), &[&latents])?;
            let weights = self.source.decoder_weights()?;
            vae::decode_tiled(&self.context, &weights, &packed, progress)?
        } else {
            let _trace = self.context.trace("image.vae");
            let pixels = {
                let packed = run(&self.context, model::decoder_input(h, w), &[&latents])?;
                let weights = self.source.decoder_weights()?;
                vae::decode_weights(&weights, &packed, ScalarType::BF16)?
            };
            let pixels = run(&self.context, model::pixel_readback(), &[&pixels])?;
            pixels.eval()?;
            pixels.read_f32()?
        };
        let rgb = model::rgb(Family::Klein2, &values)?;
        let image = image::RgbImage::from_raw(options.width, options.height, rgb)
            .ok_or_else(|| anyhow::anyhow!("VAE output dimensions do not match request"))?;
        let mut bytes = std::io::Cursor::new(Vec::new());
        image.write_to(&mut bytes, image::ImageFormat::Png)?;
        Ok(bytes.into_inner())
    }
}
