//! GGUF checkpoint adaptation for the Qwen 27B text and MTP graph.
pub use crate::checkpoint::catalog::qwen35::{ENTRY, REPO};
pub use crate::checkpoint::qwen35::tokenizer::from_header as tokenizer;
use crate::{
    bindings::{
        BindingDevice,
        qwen35::gguf::{
            execution::{EmbeddingRows, GgufExecution},
            weights::Bound,
        },
    },
    execution::DeviceContext,
    weights::gguf::ResidentWeights,
};
use anyhow::{Result, ensure};
use lisa_backend::{HostTensor, ManagedDevice, ScalarType};
use lisa_gguf::{Package, Tensor, quant};
use lisa_models::qwen3_5::gguf as model;
pub use lisa_models::qwen3_5::gguf::Config;
use std::path::Path;
pub fn validate(p: &Package) -> Result<Config> {
    use lisa_models::qwen3_5::gguf as model;
    let metadata = &p.header.metadata;
    model::validate(&model::Checkpoint {
        architecture: p.header.architecture()?.to_owned(),
        integers: metadata
            .iter()
            .filter_map(|(k, v)| v.as_u64().map(|n| (k.clone(), n)))
            .collect(),
        reals: metadata
            .iter()
            .filter_map(|(k, v)| v.as_f64().map(|n| (k.clone(), n)))
            .collect(),
        tensors: p
            .tensors
            .iter()
            .map(|(k, t)| {
                (
                    k.clone(),
                    model::TensorInfo {
                        dimensions: t.info.dimensions.clone(),
                        encoding: tensor_encoding(t.info.kind),
                    },
                )
            })
            .collect(),
    })
}

/// Validate the selected text package without creating a device runtime.
/// Embeddings are decoded a row at a time from disk and are not GPU-resident.
pub fn inspect(path: &std::path::Path) -> anyhow::Result<u64> {
    let package = lisa_gguf::Package::open(path)?;
    validate(&package)?;
    package
        .tensors
        .iter()
        .filter(|(name, _)| name.as_str() != "token_embd.weight")
        .try_fold(0u64, |total, (_, tensor)| {
            total
                .checked_add(tensor.mapped_region_bytes()? as u64)
                .ok_or_else(|| anyhow::anyhow!("Qwen resident size overflow"))
        })
}

pub fn load<D: BindingDevice + ManagedDevice>(
    context: DeviceContext<D>,
    path: &Path,
) -> Result<GgufExecution<D>> {
    let package = Package::open(path)?;
    let mut identity = format!(
        "{}:{:?}",
        model::CACHE_LAW,
        std::fs::metadata(path)?.modified()?
    );
    for (key, value) in &package.header.metadata {
        if key.starts_with("qwen35.") {
            identity.push_str(&format!(":{key}={value:?}"));
        }
    }
    for tensor in package.tensors.values() {
        identity.push_str(&format!(":{:?}", tensor.info));
    }
    let law = model::cache_law(identity.as_bytes());
    let config = validate(&package)?;
    let mut model_config = lisa_models::qwen3_5::config::Qwen35Config::parse("{}")?;
    model_config.rms_norm_eps = config.eps;
    model_config.rope_theta = config.theta;
    model_config.max_position_embeddings = config.context;
    model_config.eos_token_id = config.eos;

    let names: Vec<_> = package
        .tensors
        .keys()
        .filter(|n| n.as_str() != "token_embd.weight")
        .map(String::as_str)
        .collect();
    let w = ResidentWeights::pull(context.device().clone(), &package, &names, |bytes| {
        context
            .device()
            .reserve_residency(bytes, crate::memory::ram_budget())
    })?;
    let bound = std::sync::Arc::new(bind(context.clone(), &w)?);
    let embedding = Box::new(Embedding {
        context: context.clone(),
        tensor: package.tensor("token_embd.weight")?.clone(),
    });
    let mapped_bytes = w.mapped_bytes;
    let (_reservation, tensors) = w.into_parts();
    let weights = crate::weights::source::PreparedTensors {
        _reservation,
        tensors: tensors
            .into_iter()
            .map(|(name, t)| (name, t.packed_bytes().clone()))
            .collect(),
    };
    GgufExecution::from_prepared(
        context,
        weights,
        mapped_bytes,
        bound,
        embedding,
        model_config,
        config,
        law,
    )
}
pub fn bind<D: BindingDevice, L>(
    context: DeviceContext<D>,
    w: &ResidentWeights<D, L>,
) -> Result<Bound<D>> {
    let mut arrays = std::collections::BTreeMap::new();
    let mut schema = lisa_models::program::Schema::new();
    let mut projections = lisa_models::qwen3_5::gguf::Projections::new();
    for (name, t) in &w.tensors {
        let shape = t
            .info
            .dimensions
            .iter()
            .rev()
            .map(|&n| i32::try_from(n))
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let array = if t.info.kind == lisa_gguf::TensorType::F32 {
            t.plain_view()?
        } else {
            ensure!(shape.len() == 2, "expected a packed matrix {name}");
            // Required projections were validated before import. Keep unrelated
            // payloads resident without assigning them an executable encoding.
            if let Ok(encoding) = ordered_encoding(t.info.kind) {
                projections.insert(
                    name.clone(),
                    lisa_models::qwen3_5::gguf::Projection {
                        encoding,
                        input: shape[1],
                        output: shape[0],
                    },
                );
            }
            t.packed_bytes().clone()
        };
        schema.insert(name.clone(), shape);
        arrays.insert(name.clone(), array);
    }
    Ok(Bound {
        context,
        arrays,
        schema,
        projections,
    })
}
struct Embedding<D: BindingDevice> {
    context: DeviceContext<D>,
    tensor: Tensor,
}
impl<D: BindingDevice> EmbeddingRows<D::Tensor> for Embedding<D> {
    fn rows(&self, rows: &[u64]) -> Result<D::Tensor> {
        let mut bytes = vec![0u8; rows.len() * self.tensor.info.row_bytes()? as usize];
        self.tensor.read_rows(rows, &mut bytes)?;
        let shape = model::embedding_shape(rows.len())?;
        let mut values = vec![0f32; shape.iter().map(|&n| n as usize).product()];
        quant::decode(self.tensor.info.kind, &bytes, &mut values)?;
        Ok(self.context.device().pull(
            HostTensor {
                bytes: &values
                    .iter()
                    .flat_map(|x| x.to_ne_bytes())
                    .collect::<Vec<_>>(),
                shape: &shape,
                dtype: ScalarType::F32,
            }
            .into(),
            Default::default(),
        )?)
    }
}

fn ordered_encoding(kind: lisa_gguf::TensorType) -> Result<lisa_backend::OrderedBlockEncoding> {
    use lisa_backend::OrderedBlockEncoding as E;
    Ok(match kind.0 {
        8 => E::Q8_0,
        10 => E::Q2K,
        11 => E::Q3K,
        12 => E::Q4K,
        13 => E::Q5K,
        14 => E::Q6K,
        16 => E::Iq2Xxs,
        17 => E::Iq2Xs,
        18 => E::Iq3Xxs,
        19 => E::Iq1S,
        21 => E::Iq3S,
        22 => E::Iq2S,
        23 => E::Iq4Xs,
        29 => E::Iq1M,
        _ => anyhow::bail!("unsupported ordered block encoding {:?}", kind),
    })
}
fn tensor_encoding(kind: lisa_gguf::TensorType) -> model::TensorEncoding {
    if kind == lisa_gguf::TensorType::F32 {
        model::TensorEncoding::F32
    } else {
        ordered_encoding(kind)
            .map(model::TensorEncoding::Block)
            .unwrap_or(model::TensorEncoding::Unsupported)
    }
}
