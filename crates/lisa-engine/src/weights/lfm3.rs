//! LFM3 checkpoint parsing, admission and original tensor import.
use super::{
    mlx::get_tensor,
    mlx::{Checkpoint, Shard},
    source::Weights,
};
use crate::{
    bindings::{BindingDevice, lfm3::Tower},
    language::runtime::Runtime,
    sampling::sampler::Sampler,
};
use anyhow::{Result, ensure};
use lisa_backend::{ManagedDevice, ReadbackDevice};
use lisa_models::lfm3::config::Config;
use std::path::Path;
pub fn load<D>(runtime: Runtime<D, Sampler<D>>, path: &Path) -> Result<Tower<D>>
where
    D: BindingDevice + ManagedDevice + ReadbackDevice,
    D::Error: From<anyhow::Error>,
{
    let config = Config::parse(&std::fs::read(path.join("config.json"))?)?;
    let checkpoint = Checkpoint::open(path)?;
    let mut files = checkpoint.weight_map.values().cloned().collect::<Vec<_>>();
    files.sort();
    files.dedup();
    let bytes = files.iter().try_fold(0u64, |n, f| {
        Ok::<_, anyhow::Error>(n + std::fs::metadata(path.join(f))?.len())
    })?;
    ensure!(
        bytes.saturating_add(256 * 1024 * 1024) <= crate::memory::ram_budget() as u64,
        "LFM3 weights and working memory exceed the device memory budget"
    );
    let device = runtime.context.device();
    let mut weights = Weights::new();
    for file in files {
        let shard = Shard::open(device.clone(), &path.join(file))?;
        for name in shard.tensor_names() {
            let key = name
                .strip_prefix("language_model.")
                .unwrap_or(name)
                .to_string();
            weights.insert(key, get_tensor(&shard, name)?);
        }
    }
    Tower::from_prepared(runtime, config, weights)
}
