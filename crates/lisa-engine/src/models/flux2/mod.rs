//! Native FLUX.2 Klein 4B text-to-image pipeline on LISA's Metal runtime.
//! Architecture and operation order: mlx-serve 6c430614 (MIT), flux2 plugin.
mod dit;
#[cfg(test)]
mod reference_tests;
mod text;
pub(crate) mod vae;
pub(crate) mod weights;
use anyhow::ensure;
use lisa_mlx::{Array, Dtype};
use std::path::{Path, PathBuf};

pub const DEFAULT_MODEL: &str = "mlx-community/flux2-klein-4b-4bit";
pub const DEFAULT_STEPS: u32 = 4;
#[derive(Clone, Debug)]
pub struct Options {
    pub width: u32,
    pub height: u32,
    /// None selects the loaded model's native default (4 for FLUX.2, 16 for FLUX 3).
    pub steps: Option<u32>,
    pub seed: u64,
    /// Skip FLUX 3's local prompt expansion when supplying a complete caption.
    pub prompt_is_caption: bool,
}
impl Default for Options {
    fn default() -> Self {
        Self {
            width: 512,
            height: 512,
            steps: None,
            seed: 0,
            prompt_is_caption: false,
        }
    }
}
impl Options {
    pub fn validate(&self) -> anyhow::Result<()> {
        ensure!(
            (64..=1024).contains(&self.width)
                && (64..=1024).contains(&self.height)
                && self.width % 32 == 0
                && self.height % 32 == 0,
            "image width and height must be multiples of 32 between 64 and 1024"
        );
        ensure!(
            self.steps.is_none_or(|steps| (1..=50).contains(&steps)),
            "steps must be between 1 and 50"
        );
        Ok(())
    }
}
pub struct Pipeline {
    dir: PathBuf,
}
impl Pipeline {
    pub fn load(dir: &Path) -> anyhow::Result<Self> {
        ensure!(
            is_checkpoint(dir),
            "expected an MLX FLUX.2 Klein checkpoint with text_encoder, transformer, tokenizer and vae"
        );
        Ok(Self { dir: dir.into() })
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
            lisa_mlx::memory::active_memory()?.saturating_add(needed)
                < lisa_mlx::memory::ram_budget(),
            "insufficient RAM budget for image generation"
        );
        if let Some(available) = lisa_mlx::memory::available_ram() {
            ensure!(
                available >= needed,
                "insufficient available memory for image generation"
            );
        }
        progress("Encoding prompt", 0, steps)?;
        let enc = {
            let _trace = lisa_mlx::trace::span("image.text_encoder");
            text::encode(&self.dir, prompt)?
        };
        let (h, w) = ((options.height / 16) as i32, (options.width / 16) as i32);
        let noise = normal_noise((128 * h * w) as usize, options.seed);
        let mut latents = Array::from_slice(&noise, &[1, 128, h * w])
            .as_dtype(Dtype::Bfloat16)?
            .transpose_axes(&[0, 2, 1])?
            .contiguous()?;
        {
            let model = {
                let _trace = lisa_mlx::trace::span("image.denoiser.load");
                dit::Dit::load(&self.dir)?
            };
            let sigma = schedule((h * w) as usize, steps as usize);
            for i in 0..steps as usize {
                let _trace = lisa_mlx::trace::span("image.denoiser.step");
                progress("Generating", i as u32, steps)?;
                let velocity = model.forward(&latents, &enc, sigma[i] * 1000.0, h, w)?;
                latents =
                    latents.add(&velocity.multiply(&Array::from_f32(sigma[i + 1] - sigma[i]))?)?;
                latents.eval()?;
            }
        }
        drop(enc);
        progress("Decoding image", steps, steps)?;
        let pixels = {
            let _trace = lisa_mlx::trace::span("image.vae");
            vae::decode(
                &self.dir,
                &latents
                    .reshape(&[1, h, w, 128])?
                    .transpose_axes(&[0, 3, 1, 2])?,
            )?
        };
        let pixels = pixels.as_dtype(Dtype::Float32)?.contiguous()?;
        pixels.eval()?;
        let values = pixels.as_slice::<f32>();
        ensure!(
            values.iter().all(|v| v.is_finite()),
            "image generation produced non-finite pixels"
        );
        let rgb: Vec<u8> = values
            .iter()
            .map(|v| ((v * 0.5 + 0.5).clamp(0.0, 1.0) * 255.0).round() as u8)
            .collect();
        let image = image::RgbImage::from_raw(options.width, options.height, rgb)
            .ok_or_else(|| anyhow::anyhow!("VAE output dimensions do not match request"))?;
        let mut bytes = std::io::Cursor::new(Vec::new());
        image.write_to(&mut bytes, image::ImageFormat::Png)?;
        Ok(bytes.into_inner())
    }
}
pub fn is_checkpoint(dir: &Path) -> bool {
    let Ok(bytes) = std::fs::read(dir.join("transformer/model.safetensors.index.json")) else {
        return false;
    };
    let Ok(index) = serde_json::from_slice::<serde_json::Value>(&bytes) else {
        return false;
    };
    index["weight_map"]
        .get("double_stream_modulation_img.linear.weight")
        .is_some()
        && dir.join("tokenizer/tokenizer.json").is_file()
        && dir.join("vae/model.safetensors.index.json").is_file()
        && dir
            .join("text_encoder/model.safetensors.index.json")
            .is_file()
}
fn schedule(tokens: usize, steps: usize) -> Vec<f32> {
    let a = 0.00016927 * tokens as f64 + 0.45666666;
    let b = 8.73809524e-5 * tokens as f64 + 1.89833333;
    let slope = (b - a) / (10.0 - 200.0);
    let mu = if tokens > 4300 {
        a
    } else {
        slope * steps as f64 + (a - slope * 200.0)
    };
    let emu = mu.exp();
    let mut out = Vec::new();
    for i in 0..steps {
        let t = if steps == 1 {
            1.0
        } else {
            1.0 + (1.0 / steps as f64 - 1.0) * i as f64 / (steps - 1) as f64
        };
        out.push((emu / (emu + (1.0 / t - 1.0))) as f32);
    }
    out.push(0.0);
    out
}
/// SplitMix64 + Box–Muller: stable LISA seeds, independent of process-global RNG.
/// The same numeric seed is not a promise of MLX's Threefry noise sequence.
pub(crate) fn normal_noise(count: usize, mut state: u64) -> Vec<f32> {
    let mut uniform = || {
        state = state.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^= z >> 31;
        ((z >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    };
    let mut out = Vec::with_capacity(count);
    while out.len() < count {
        let r = (-2.0 * uniform().ln()).sqrt();
        let a = std::f64::consts::TAU * uniform();
        out.push((r * a.cos()) as f32);
        if out.len() < count {
            out.push((r * a.sin()) as f32);
        }
    }
    out
}
pub(crate) fn silu(x: &Array) -> anyhow::Result<Array> {
    Ok(x.multiply(&lisa_mlx::ops::sigmoid(x)?)?)
}
pub(crate) fn layer_norm(x: &Array) -> anyhow::Result<Array> {
    let x = x.as_dtype(Dtype::Float32)?;
    let xc = x.subtract(&x.mean_axis(-1, true)?)?;
    let inv = xc
        .multiply(&xc)?
        .mean_axis(-1, true)?
        .add(&Array::from_f32(1e-6))?
        .rsqrt()?;
    Ok(xc.multiply(&inv)?.as_dtype(Dtype::Bfloat16)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn noise_is_reproducible_and_schedule_reaches_zero() {
        let a = normal_noise(101, 42);
        assert_eq!(a, normal_noise(101, 42));
        assert_ne!(a, normal_noise(101, 43));
        assert!(a.iter().all(|x| x.is_finite()));
        for steps in [1, 4, 50] {
            for tokens in [16, 1024, 4096] {
                let s = schedule(tokens, steps);
                assert_eq!(s.len(), steps + 1);
                assert_eq!(s[0], 1.0);
                assert_eq!(s[steps], 0.0);
                assert!(s.windows(2).all(|pair| pair[0] > pair[1]));
            }
        }
    }
}
