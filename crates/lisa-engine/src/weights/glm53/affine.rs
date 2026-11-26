//! GLM affine checkpoint loading and tensor-bank adaptation.
use crate::{
    bindings::glm53::affine::{
        moe::ExpertDevice,
        tower::{Tower, is_glm_routed_expert_tensor},
    },
    language::runtime::Runtime,
    sampling::sampler::Sampler,
    weights::{
        lazy::LazyTensorStore,
        mlx::Checkpoint,
        source::{TensorBank, Weights},
    },
};
use lisa_backend::ReadbackDevice;
use lisa_models::{checkpoint::sanitize_name, glm53::config::ModelConfig};
use std::path::Path;

pub fn load<D: ExpertDevice + ReadbackDevice>(
    runtime: Runtime<D, Sampler<D>>,
    dir: &Path,
    config: ModelConfig,
) -> anyhow::Result<Tower<D>>
where
    D::Error: From<anyhow::Error>,
{
    let context = runtime.context.clone();
    // The reference's resident graph owns a small set of immutable model views. The
    // aligned cache preserves the original MLX payloads while allowing
    // every expert address to remain valid for the model's lifetime.
    let checkpoint = Checkpoint::open(dir)?;
    let store = std::sync::Arc::new(LazyTensorStore::open_resident(
        context.device().clone(),
        dir,
        sanitize_name,
    )?);
    let weight_bytes = store.resident_nbytes();
    let budget = crate::memory::ram_budget();
    anyhow::ensure!(
        weight_bytes < budget,
        "GLM resident weights require {:.2} GiB, exceeding the {:.2} GiB RAM budget",
        weight_bytes as f64 / (1u64 << 30) as f64,
        budget as f64 / (1u64 << 30) as f64,
    );
    let reservation = context.device().reserve_residency(weight_bytes, budget)?;
    let mut weights = Weights::new();
    for original in checkpoint.weight_map.keys() {
        let Some(name) = sanitize_name(original) else {
            continue;
        };
        if is_glm_routed_expert_tensor(&name) || weights.contains_key(&name) {
            continue;
        }
        // The resident reader rejects missing or unaligned ranges. There
        // is no individual-allocation fallback on this execution path.
        weights.insert(name.clone(), store.get_mapped(&name)?);
    }
    let mut tower = Tower::from_weights_with_lazy_experts(
        runtime,
        weights,
        config,
        Some(store.clone()),
        true,
        true,
    )?;
    tower._resident_memory = Some(reservation);
    tower.vision = crate::weights::vision::load(context.clone(), dir, 0)?;
    eprintln!(
        "[glm53-loader] resident MLX model: {:.2} GiB, {} shared views, GPU expert routing",
        weight_bytes as f64 / (1u64 << 30) as f64,
        store.resident_allocations().len(),
    );
    Ok(tower)
}

impl<D: lisa_backend::StorageDevice + Clone + Send + Sync> TensorBank<D::Tensor>
    for LazyTensorStore<D>
where
    D::Tensor: Clone + Send + Sync,
{
    fn tensor_dtype(&self, name: &str) -> anyhow::Result<lisa_backend::ScalarType> {
        LazyTensorStore::tensor_dtype(self, name)
    }
    fn tensor_nbytes(&self, name: &str) -> anyhow::Result<usize> {
        LazyTensorStore::tensor_nbytes(self, name)
    }
    fn get_mapped(&self, name: &str) -> anyhow::Result<D::Tensor> {
        LazyTensorStore::get_mapped(self, name)
    }
    fn get_mapped_individual(&self, name: &str) -> anyhow::Result<D::Tensor> {
        LazyTensorStore::get_mapped_individual(self, name)
    }
    fn get_mapped_weight_bytes(&self, name: &str) -> anyhow::Result<D::Tensor> {
        LazyTensorStore::get_mapped_weight_bytes(self, name)
    }
    fn get_mapped_weight_bytes_individual(&self, name: &str) -> anyhow::Result<D::Tensor> {
        LazyTensorStore::get_mapped_weight_bytes_individual(self, name)
    }
    fn get_packed(&self, names: &[String]) -> anyhow::Result<D::Tensor> {
        LazyTensorStore::get_packed(self, names)
    }
    fn prefetch_mapped_tensors(&self, names: &[String]) -> anyhow::Result<(usize, usize)> {
        LazyTensorStore::prefetch_mapped_tensors(self, names)
    }
    fn resident_allocations(&self) -> Vec<D::Tensor> {
        LazyTensorStore::resident_allocations(self)
    }
    fn resident_nbytes(&self) -> usize {
        LazyTensorStore::resident_nbytes(self)
    }
    fn is_resident(&self) -> bool {
        LazyTensorStore::is_resident(self)
    }
}
