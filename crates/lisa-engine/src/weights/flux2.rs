//! Checkpoint adaptation and staged residency for FLUX.2.
use crate::core::tokenizer::Tokenizer;
use crate::{
    bindings::{
        BindingDevice,
        flux2::{self, weights::Weights},
    },
    execution::DeviceContext,
};
use anyhow::ensure;
use lisa_backend::{ManagedDevice, TensorAccess};
use lisa_models::flux::{pipeline, text2};
use std::path::{Path, PathBuf};

pub use crate::checkpoint::catalog::is_flux2 as is_checkpoint;
struct Source<D: BindingDevice + ManagedDevice> {
    context: DeviceContext<D>,
    dir: PathBuf,
    bounded: bool,
}
impl<D: BindingDevice + ManagedDevice> flux2::StageSource<D> for Source<D> {
    fn tokenizer(&self) -> anyhow::Result<Tokenizer> {
        crate::checkpoint::tokenizer::load(&self.dir.join("tokenizer"))
    }
    fn text_weights(&self) -> anyhow::Result<Weights<D>> {
        text_weights(self.context.clone(), &self.dir, self.bounded)
    }
    fn denoiser_weights(&self) -> anyhow::Result<Weights<D>> {
        denoiser_weights(self.context.clone(), &self.dir, self.bounded)
    }
    fn decoder_weights(&self) -> anyhow::Result<Weights<D>> {
        decoder_weights(self.context.clone(), &self.dir, self.bounded)
    }
}
pub fn load<D: BindingDevice + ManagedDevice>(
    context: DeviceContext<D>,
    dir: &Path,
) -> anyhow::Result<flux2::Pipeline<D>> {
    ensure!(
        is_checkpoint(dir),
        "expected an MLX FLUX.2 Klein checkpoint with text_encoder, transformer, tokenizer and vae"
    );
    let bounded = crate::checkpoint::catalog::bonsai::is_checkpoint(dir);
    let policy = flux2::Policy {
        bounded,
        required_ram: if bounded {
            crate::checkpoint::catalog::bonsai::REQUIRED_RAM
        } else {
            8usize << 30
        },
    };
    let source = Box::new(Source {
        context: context.clone(),
        dir: dir.into(),
        bounded,
    });
    Ok(flux2::Pipeline::from_prepared(context, source, policy))
}

pub fn text_weights<D: BindingDevice>(
    context: DeviceContext<D>,
    dir: &Path,
    bounded: bool,
) -> anyhow::Result<Weights<D>> {
    let mut weights = super::flux::load_filtered(
        context,
        &dir.join(if bounded {
            "text_encoder-mlx-4bit"
        } else {
            "text_encoder"
        }),
        |name| !bounded || text2::conditioning_weight(name.strip_prefix("model.").unwrap_or(name)),
    )?;
    if bounded {
        weights.0 = weights
            .0
            .into_iter()
            .map(|(name, value)| {
                (
                    name.strip_prefix("model.").unwrap_or(&name).to_owned(),
                    value,
                )
            })
            .collect();
    }
    Ok(weights)
}

pub fn denoiser_weights<D: BindingDevice>(
    context: DeviceContext<D>,
    dir: &Path,
    bounded: bool,
) -> anyhow::Result<Weights<D>> {
    let mut weights = if bounded {
        super::flux::load_file(
            context,
            &dir.join("transformer-packed-mflux/diffusion_pytorch_model.safetensors"),
            |_| true,
        )?
    } else {
        super::flux::load(context, &dir.join("transformer"))?
    };
    weights.0 = weights
        .0
        .into_iter()
        .map(|(name, value)| {
            (
                name.replace(
                    "time_guidance_embed.timestep_embedder.",
                    "time_guidance_embed.",
                )
                .replace(".to_out.0.", ".to_out."),
                value,
            )
        })
        .collect();
    Ok(weights)
}

pub fn decoder_weights<D: BindingDevice>(
    context: DeviceContext<D>,
    dir: &Path,
    bounded: bool,
) -> anyhow::Result<Weights<D>> {
    if !bounded {
        return super::flux::load(context, &dir.join("vae"));
    }
    let original = super::flux::load_file(
        context.clone(),
        &dir.join("vae/diffusion_pytorch_model.safetensors"),
        |key| {
            key.starts_with("decoder.")
                || key.starts_with("post_quant_conv.")
                || key.starts_with("bn.running_")
        },
    )?;
    let mut tensors = std::collections::HashMap::new();
    for (name, value) in original.0 {
        let (name, prepare) =
            pipeline::decoder_weight(&name.replace(".to_out.0.", ".to_out."), value.shape())?;
        let value = if let Some(graph) = prepare {
            flux2::weights::run(&context, graph, &[&value])?
        } else {
            value
        };
        tensors.insert(name, value);
    }
    Ok(Weights::from_map(context, tensors))
}
