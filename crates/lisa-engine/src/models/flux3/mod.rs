//! Native FLUX 3 Image Klein text-to-image inference.
//! Adapted from BFL's Apache-2.0 reference bfc8889a4497f846416f253a3972a1941ac698b9.
//! Copyright 2026 Black Forest Labs, Inc. See LICENSE for the full notice.
mod dit;
#[cfg(test)]
pub(crate) mod tests;
mod text;
use crate::models::flux2::{Options, normal_noise, vae, weights::Weights};
use anyhow::ensure;
use lisa_mlx::{Array, Dtype};
use std::path::{Path, PathBuf};

pub const REPO: &str = "black-forest-labs/flux-3-image-klein";
pub const DEFAULT_STEPS: u32 = 16;
pub struct Pipeline {
    dir: PathBuf,
}

pub fn is_checkpoint(dir: &Path) -> bool {
    std::fs::read(dir.join("config.json"))
        .ok()
        .and_then(|b| serde_json::from_slice::<serde_json::Value>(&b).ok())
        .is_some_and(|v| v["model"] == "flux-3-image-klein" && v["dit"]["flow_map"] == true)
}
impl Pipeline {
    pub fn load(dir: &Path) -> anyhow::Result<Self> {
        ensure!(
            is_checkpoint(dir),
            "expected the distilled FLUX 3 Image Klein native checkpoint"
        );
        let config: serde_json::Value =
            serde_json::from_slice(&std::fs::read(dir.join("config.json"))?)?;
        let d = &config["dit"];
        ensure!(
            config["quantization"].is_null()
                && d["hidden"] == 3072
                && d["heads"] == 24
                && d["depth"] == 5
                && d["depth_single"] == 28
                && d["theta"] == 10000
                && d["image_rope_scale"] == 0.5,
            "unsupported FLUX 3 architecture or quantization; select the original BF16 package"
        );
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
        Ok(Self { dir: dir.into() })
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
            lisa_mlx::memory::active_memory()?.saturating_add(needed)
                < lisa_mlx::memory::ram_budget(),
            "insufficient RAM budget for FLUX 3"
        );
        if let Some(available) = lisa_mlx::memory::available_ram() {
            ensure!(available >= needed, "insufficient available RAM for FLUX 3");
        }
        let enc = {
            let _trace = lisa_mlx::trace::span("image.text_encoder");
            progress("Loading text encoder", 0, steps)?;
            let encoder = text::Encoder::load(&self.dir)?;
            let caption = if options.prompt_is_caption {
                prompt.to_owned()
            } else {
                let _trace = lisa_mlx::trace::span("image.prompt_expansion");
                encoder.expand(prompt, options.width, options.height, &mut || {
                    progress("Expanding prompt", 0, steps)
                })?
            };
            progress("Encoding prompt", 0, steps)?;
            encoder.encode(&caption)?
        };
        lisa_mlx::memory::clear_cache();
        let (h, w) = ((options.height / 16) as i32, (options.width / 16) as i32);
        // Noise is bf16-rounded; the recurrent latent state and Euler update remain f32.
        let noise = normal_noise((h * w * 128) as usize, options.seed);
        let mut latent = Array::from_slice(&noise, &[1, h * w, 128])
            .as_dtype(Dtype::Bfloat16)?
            .as_dtype(Dtype::Float32)?;
        {
            progress("Loading image model", 0, steps)?;
            let model = {
                let _trace = lisa_mlx::trace::span("image.denoiser.load");
                dit::Dit::load(&self.dir)?
            };
            let times = schedule(steps);
            for i in 0..steps as usize {
                progress("Generating", i as u32, steps)?;
                let _trace = lisa_mlx::trace::span("image.denoiser.step");
                let velocity = model.forward(&latent, &enc, times[i], times[i + 1], h, w)?;
                // Reference multiplies the bf16 prediction by dt before adding to f32 state.
                let delta = velocity
                    .multiply(&Array::from_f32(times[i + 1] - times[i]))?
                    .as_dtype(Dtype::Bfloat16)?;
                latent = latent.add(&delta.as_dtype(Dtype::Float32)?)?;
                latent.eval()?;
            }
        }
        drop(enc);
        lisa_mlx::memory::clear_cache();
        progress("Decoding image", steps, steps)?;
        let pixels = {
            let _trace = lisa_mlx::trace::span("image.vae");
            let weights = decoder_weights(&self.dir)?;
            vae::decode_weights(
                &weights,
                &latent
                    .reshape(&[1, h, w, 128])?
                    .transpose_axes(&[0, 3, 1, 2])?,
                Dtype::Float32,
            )?
        }
        .as_dtype(Dtype::Float32)?
        .contiguous()?;
        pixels.eval()?;
        let values = pixels.as_slice::<f32>();
        ensure!(
            values.iter().all(|v| v.is_finite()),
            "FLUX 3 generated non-finite pixels"
        );
        let rgb: Vec<u8> = values
            .iter()
            .map(|v| ((v.clamp(-1.0, 1.0) + 1.0) * 127.5) as u8)
            .collect();
        let image = image::RgbImage::from_raw(options.width, options.height, rgb)
            .ok_or_else(|| anyhow::anyhow!("FLUX 3 output shape mismatch"))?;
        let mut out = std::io::Cursor::new(Vec::new());
        image.write_to(&mut out, image::ImageFormat::Png)?;
        Ok(out.into_inner())
    }
}

/// Follow the executable reference's alpha=16. The exported config says 6.93,
/// but pipeline.generate uses 16 and the public CLI does not override it.
fn schedule(steps: u32) -> Vec<f32> {
    (0..=steps)
        .map(|i| {
            let t = 1.0 - i as f32 / steps as f32;
            16.0 * t / (1.0 + 15.0 * t)
        })
        .collect()
}

/// Rename the original f32 decoder into the shared VAE operation layout.
/// Only convolution storage is transposed (OIHW -> OHWI); payload precision is retained.
fn decoder_weights(dir: &Path) -> anyhow::Result<Weights> {
    let source = Weights::load_file(&dir.join("ae.safetensors"), |k| {
        k.starts_with("decoder.") || k.starts_with("bn.")
    })?;
    let mut out = std::collections::HashMap::new();
    for (key, value) in source.0 {
        let mut name = key
            .replace("decoder.post_quant_conv", "post_quant_conv")
            .replace("decoder.mid.block_1", "decoder.mid_block.resnets.0")
            .replace("decoder.mid.block_2", "decoder.mid_block.resnets.1")
            .replace(
                "decoder.mid.attn_1.norm",
                "decoder.mid_block.attentions.0.group_norm",
            )
            .replace(
                "decoder.mid.attn_1.proj_out",
                "decoder.mid_block.attentions.0.to_out",
            )
            .replace(
                "decoder.mid.attn_1.q",
                "decoder.mid_block.attentions.0.to_q",
            )
            .replace(
                "decoder.mid.attn_1.k",
                "decoder.mid_block.attentions.0.to_k",
            )
            .replace(
                "decoder.mid.attn_1.v",
                "decoder.mid_block.attentions.0.to_v",
            )
            .replace("decoder.norm_out", "decoder.conv_norm_out")
            .replace("nin_shortcut", "conv_shortcut");
        for level in 0..4 {
            name = name
                .replace(
                    &format!("decoder.up.{level}.block."),
                    &format!("decoder.up_blocks.{}.resnets.", 3 - level),
                )
                .replace(
                    &format!("decoder.up.{level}.upsample.conv"),
                    &format!("decoder.up_blocks.{}.upsamplers.0.conv", 3 - level),
                );
        }
        let value = if value.ndim() == 4 {
            if name.contains(".attentions.") {
                ensure!(
                    value.dim(2) == 1 && value.dim(3) == 1,
                    "expected pointwise VAE attention projection"
                );
                value.reshape(&[value.dim(0), value.dim(1)])?
            } else {
                value.transpose_axes(&[0, 2, 3, 1])?.contiguous()?
            }
        } else {
            value
        };
        out.insert(name, value);
    }
    Ok(Weights::from_map(out))
}
