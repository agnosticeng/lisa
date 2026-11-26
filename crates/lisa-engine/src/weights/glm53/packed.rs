//! Adapt validated GLM packed checkpoint tensors to the execution boundary.
use crate::{
    bindings::{BindingDevice, glm53::packed::tower::Tower},
    language::runtime::Runtime,
    sampling::sampler::Sampler,
    weights::{gguf::ResidentWeights, source::EmbeddingRows},
};
use anyhow::Result;
use lisa_backend::{ManagedDevice, ReadbackDevice};
use lisa_gguf::{Package, Tensor, quant};
use lisa_models::glm53::packed_tower as model;
use std::{os::unix::fs::MetadataExt, path::Path};

pub fn load<D: BindingDevice + ManagedDevice + ReadbackDevice>(
    runtime: Runtime<D, Sampler<D>>,
    path: &Path,
) -> Result<Tower<D>> {
    let context = runtime.context.clone();
    let package = Package::open(path)?;
    let config = crate::checkpoint::glm53::validate(&package)?;
    let names: Vec<_> = package
        .tensors
        .keys()
        .filter(|n| n.as_str() != model::EMBEDDING_WEIGHT)
        .map(String::as_str)
        .collect();
    let w = ResidentWeights::pull(context.device().clone(), &package, &names, |bytes| {
        context
            .device()
            .reserve_residency(bytes, crate::memory::ram_budget())
    })?;
    let stat = std::fs::metadata(path)?;
    let mut law = 0xcbf29ce484222325u64;
    for byte in model::NUMERICAL_IDENTITY.iter().copied().chain(
        [
            stat.dev(),
            stat.ino(),
            stat.len(),
            stat.mtime() as u64,
            stat.mtime_nsec() as u64,
        ]
        .into_iter()
        .flat_map(u64::to_le_bytes),
    ) {
        law = (law ^ u64::from(byte)).wrapping_mul(0x100000001b3);
    }

    let embedding = package.tensor(model::EMBEDDING_WEIGHT)?.clone();
    let provider = Rows {
        device: context.device().clone(),
        source: embedding,
    };
    Tower::from_prepared(runtime, config, w.into_prepared()?, Box::new(provider), law)
}

struct Rows<D: lisa_backend::StorageDevice> {
    device: D,
    source: Tensor,
}
impl<D: lisa_backend::StorageDevice + Send + Sync> EmbeddingRows<D::Tensor> for Rows<D> {
    fn row(&self, token: u32) -> Result<D::Tensor> {
        let mut packed = vec![0u8; self.source.info.row_bytes()? as usize];
        self.source.read_rows(&[token as u64], &mut packed)?;
        let mut row = vec![0f32; model::WIDTH];
        quant::decode(self.source.info.kind, &packed, &mut row)?;
        Ok(self.device.pull(
            lisa_backend::HostTensor {
                bytes: &row.iter().flat_map(|x| x.to_ne_bytes()).collect::<Vec<_>>(),
                shape: &model::INPUT_SHAPE,
                dtype: lisa_backend::ScalarType::F32,
            }
            .into(),
            Default::default(),
        )?)
    }
}
