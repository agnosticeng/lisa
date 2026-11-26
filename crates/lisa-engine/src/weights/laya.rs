//! Original Laya checkpoint loading and lazy tensor import.
use super::source::TensorSource;
use crate::{
    checkpoint::laya::LayaConfigIo,
    laya::{Laya, LayaConfig},
};
use anyhow::{Context, Result};
use lisa_backend::{
    HostTensor, ImportOptions, Kernel, ReadbackDevice, ScalarType, TensorOperation,
};
use lisa_mlx::Shard;
use lisa_models::laya::plan::Arithmetic;
use std::path::Path;

pub fn load_on<D>(dir: &Path, device: D, arithmetic: Arithmetic) -> Result<Laya>
where
    D: ReadbackDevice + Clone + 'static,
    D::Tensor: Clone,
{
    let mut config = LayaConfig::from_dir(dir)?;

    let tokenizer = crate::checkpoint::tokenizer::load_file(&dir.join("tokenizer/tokenizer.json"))
        .map_err(|e| anyhow::anyhow!("laya tokenizer: {e}"))?;
    config.mask_id = tokenizer
        .inner
        .token_to_id(&config.mask_token)
        .ok_or_else(|| anyhow::anyhow!("laya: token {} not in tokenizer", config.mask_token))?;

    let weights = open_weights(dir)?;
    let mut tensor_names: Vec<String> = weights.tensor_names().cloned().collect();
    tensor_names.sort();
    verify_encoder_tensors(&weights, &config)?;

    let (_, _, shape) = weights
        .tensor_bytes("type_emb.weight")
        .context("missing type embedding")?;
    anyhow::ensure!(shape.len() == 2, "invalid Laya type embedding shape");
    let question_types = shape[0];
    let source = Box::new(Source {
        device: device.clone(),
        arithmetic,
        shard: weights,
    });
    Ok(Laya::from_prepared(
        dir.to_path_buf(),
        config,
        tokenizer,
        tensor_names,
        question_types,
        source,
        device,
        arithmetic,
    ))
}

struct Source<D> {
    device: D,
    arithmetic: Arithmetic,
    shard: Shard,
}
impl<D> TensorSource<D::Tensor> for Source<D>
where
    D: ReadbackDevice + Clone,
    D::Tensor: Clone,
{
    fn get(&mut self, name: &str) -> Result<D::Tensor> {
        let (bytes, dtype, shape) = self
            .shard
            .tensor_bytes(name)
            .with_context(|| format!("laya: missing weight {name}"))?;
        let dtype = super::mlx::scalar_type(dtype);
        if self.arithmetic == Arithmetic::HostF32 {
            let data = match dtype {
                ScalarType::F32 => bytes.to_vec(),
                ScalarType::F16 => bytes
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .flat_map(|b| half::f16::from_le_bytes(*b).to_f32().to_le_bytes())
                    .collect(),
                ScalarType::BF16 => bytes
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .flat_map(|b| half::bf16::from_le_bytes(*b).to_f32().to_le_bytes())
                    .collect(),
                _ => anyhow::bail!("unsupported Laya weight dtype"),
            };
            return Ok(self.device.pull(
                HostTensor {
                    bytes: &data,
                    shape,
                    dtype: ScalarType::F32,
                }
                .into(),
                ImportOptions::default(),
            )?);
        }
        let input = self.device.pull(
            HostTensor {
                bytes,
                shape,
                dtype,
            }
            .into(),
            ImportOptions::default(),
        )?;
        if dtype == ScalarType::F32 {
            return Ok(input);
        }
        let cast = self
            .device
            .prepare(Kernel::Tensor(TensorOperation::Cast(ScalarType::F32)))?
            .context("device does not support the required weight cast")?;
        let result = self
            .device
            .dispatch(&cast, &[&input], &self.device.stream())?
            .context("device rejected the required weight cast")?;
        result.output.into_one().map_err(anyhow::Error::msg)
    }
}

/// Open the checkpoint weights: a single `model.safetensors`, else the sharded
/// index (the original HF layout).
fn open_weights(dir: &Path) -> anyhow::Result<Shard> {
    let single = dir.join("model.safetensors");
    if single.is_file() {
        return Shard::open(&single);
    }
    anyhow::bail!(
        "laya: no model.safetensors under {} (sharded checkpoints not handled yet)",
        dir.display()
    );
}

/// Check that the original-name tensors the forward will need are present.
///
/// Original layout: `encoder.*` (ModernBERT), `head.layers.{0,1}.*` (decision
/// head), `scorer.*` (option-marker scorer), `act_head.*`, `type_emb.weight`.
/// Layer 0 shares `encoder.embeddings.norm` as its attention norm; later
/// layers carry `attn_norm`/`mlp_norm`.
fn verify_encoder_tensors(shard: &Shard, cfg: &LayaConfig) -> anyhow::Result<()> {
    let emb = "encoder.embeddings.tok_embeddings.weight";
    let (_, dtype, shape) = shard
        .tensor_bytes(emb)
        .ok_or_else(|| anyhow::anyhow!("laya: missing {emb}"))?;
    anyhow::ensure!(
        shape == [cfg.vocab_size as i32, cfg.hidden_size as i32],
        "laya: {emb} shape {shape:?}, expected [{}, {}]",
        cfg.vocab_size,
        cfg.hidden_size
    );
    anyhow::ensure!(
        matches!(
            dtype,
            lisa_mlx::Dtype::Float16 | lisa_mlx::Dtype::Float32 | lisa_mlx::Dtype::Bfloat16
        ),
        "laya: {emb} has unsupported dtype {dtype:?}"
    );

    let required = lisa_models::laya::weights::required_tensors(cfg);
    let missing: Vec<String> = required
        .into_iter()
        .filter(|n| shard.tensor_bytes(n).is_none())
        .take(5)
        .collect();
    anyhow::ensure!(missing.is_empty(), "laya: missing tensors {missing:?}");
    Ok(())
}
