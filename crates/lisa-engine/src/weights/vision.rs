//! Checkpoint adaptation and lazy residency for image encoders.
use crate::{
    bindings::BindingDevice,
    execution::DeviceContext,
    tensor_ops::{TensorLaw, TensorOps},
    vision::{
        self,
        context::{Encoder, EncoderSource, Vision},
    },
    weights::mlx::{Checkpoint, Shard},
};
use anyhow::{Context, ensure};
use lisa_backend::{ManagedDevice, ScalarType as Dtype, TensorOperation};
use lisa_models::vision::prompt::Config;
use std::{
    collections::HashMap,
    path::{Path, PathBuf},
};

pub fn load_weights<D: BindingDevice>(
    context: &DeviceContext<D>,
    dir: &Path,
) -> anyhow::Result<HashMap<String, D::Tensor>> {
    let checkpoint = Checkpoint::open(dir)?;
    let mut weights = HashMap::new();
    let mut shards = HashMap::<String, Shard<D>>::new();
    for (name, file) in &checkpoint.weight_map {
        let Some(key) = ["vision_tower.", "model.visual.", "visual."]
            .iter()
            .find_map(|p| name.strip_prefix(p))
        else {
            continue;
        };
        if !shards.contains_key(file) {
            shards.insert(
                file.clone(),
                Shard::open(context.device().clone(), &dir.join(file))?,
            );
        }
        let tensor = crate::weights::mlx::get_tensor(&shards[file], name)?;
        ensure!(
            context.device().dtype(&tensor) == Some(Dtype::BF16)
                || context.device().dtype(&tensor) == Some(Dtype::F32)
                || context.device().dtype(&tensor) == Some(Dtype::F16),
            "quantized vision tensor {name} requires a matching quantized encoder"
        );
        ensure!(
            weights
                .insert(
                    key.to_string(),
                    TensorOps::new(context.device(), TensorLaw::Native)
                        .op(TensorOperation::Cast(Dtype::BF16), &[&tensor])?
                )
                .is_none(),
            "duplicate vision tensor {key}"
        );
    }
    Ok(weights)
}

/// Retain only the source identity until an image request needs its weights.
struct Source<D: BindingDevice + ManagedDevice> {
    context: DeviceContext<D>,
    dir: PathBuf,
    glm: bool,
}
impl<D: BindingDevice + ManagedDevice> EncoderSource<D> for Source<D> {
    fn load(&self) -> anyhow::Result<Encoder<D>> {
        if self.glm {
            Ok(Encoder::Glm(load_glm(self.context.clone(), &self.dir)?))
        } else {
            Ok(Encoder::Qwen(load_qwen(self.context.clone(), &self.dir)?))
        }
    }
}

pub fn load<D: BindingDevice + ManagedDevice>(
    context: DeviceContext<D>,
    dir: &Path,
    rotary_dimensions: usize,
) -> anyhow::Result<Option<Vision<D>>> {
    let root = serde_json::from_slice(&std::fs::read(dir.join("config.json"))?)?;
    Ok(Config::parse(&root, rotary_dimensions)?.map(|config| {
        let source = Box::new(Source {
            context: context.clone(),
            dir: dir.into(),
            glm: config.glm,
        });
        Vision::from_prepared(context, config, source)
    }))
}

pub fn load_qwen<D: BindingDevice + ManagedDevice>(
    context: DeviceContext<D>,
    dir: &Path,
) -> anyhow::Result<vision::qwen::Encoder<D>> {
    let root: serde_json::Value = serde_json::from_slice(&std::fs::read(dir.join("config.json"))?)?;
    let config: vision::qwen::Config = serde_json::from_value(
        root.get("vision_config")
            .context("checkpoint has no vision_config")?
            .clone(),
    )?;
    config.validate()?;
    vision::admit_scratch(&context, 2usize << 30)?;
    let mut encoder =
        vision::qwen::Encoder::from_weights(context.clone(), config, load_weights(&context, dir)?)?;
    encoder.processor = match std::fs::read(dir.join("preprocessor_config.json")) {
        Ok(bytes) => serde_json::from_slice(&bytes)?,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => serde_json::Value::Null,
        Err(e) => return Err(e.into()),
    };
    Ok(encoder)
}

pub fn load_glm<D: BindingDevice + ManagedDevice>(
    context: DeviceContext<D>,
    dir: &Path,
) -> anyhow::Result<vision::glm::Encoder<D>> {
    let root: serde_json::Value = serde_json::from_slice(&std::fs::read(dir.join("config.json"))?)?;
    let config: vision::glm::Config = serde_json::from_value(
        root.get("vision_config")
            .context("checkpoint has no vision_config")?
            .clone(),
    )?;
    config.validate()?;
    vision::admit_scratch(&context, 2usize << 30)?;
    vision::glm::Encoder::from_weights(context.clone(), config, load_weights(&context, dir)?)
}
