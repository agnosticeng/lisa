//! Original tensor import for image checkpoint adapters.
use crate::weights::mlx::{Checkpoint, Shard};
use crate::{
    bindings::{BindingDevice, flux2::weights::Weights},
    execution::DeviceContext,
};
use std::{collections::HashMap, path::Path};

pub fn load<D: BindingDevice>(context: DeviceContext<D>, dir: &Path) -> anyhow::Result<Weights<D>> {
    load_filtered(context.clone(), dir, |_| true)
}
pub fn load_file<D: BindingDevice>(
    context: DeviceContext<D>,
    path: &Path,
    keep: impl Fn(&str) -> bool,
) -> anyhow::Result<Weights<D>> {
    let shard = Shard::open(context.device().clone(), path)?;
    let mut out = HashMap::new();
    for key in shard.tensor_names().filter(|key| keep(key)) {
        out.insert(key.clone(), crate::weights::mlx::get_tensor(&shard, key)?);
    }
    Ok(Weights::from_map(context.clone(), out))
}
pub fn load_filtered<D: BindingDevice>(
    context: DeviceContext<D>,
    dir: &Path,
    keep: impl Fn(&str) -> bool,
) -> anyhow::Result<Weights<D>> {
    if !dir.join("model.safetensors.index.json").is_file() {
        return load_file(context.clone(), &dir.join("model.safetensors"), keep);
    }
    let cp = Checkpoint::open(dir)?;
    let mut shards = HashMap::new();
    let mut out = HashMap::new();
    for (key, file) in cp.weight_map {
        if !keep(&key) {
            continue;
        }
        if !shards.contains_key(&file) {
            shards.insert(
                file.clone(),
                Shard::open(context.device().clone(), &dir.join(&file))?,
            );
        }
        out.insert(
            key.clone(),
            crate::weights::mlx::get_tensor(&shards[&file], &key)?,
        );
    }
    Ok(Weights::from_map(context.clone(), out))
}
