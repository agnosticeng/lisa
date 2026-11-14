//! Checkpoint I/O and device bindings shared by the vision plans.
use crate::core::loader::{Checkpoint, Shard};
use anyhow::{Context, ensure};
use lisa_metal::{Array, Dtype};
use std::{collections::HashMap, path::Path};
pub(super) fn get(w: &HashMap<String, Array>, name: &str) -> anyhow::Result<Array> {
    w.get(name)
        .cloned()
        .with_context(|| format!("missing vision tensor {name}"))
}

pub(super) fn load_weights(dir: &Path) -> anyhow::Result<HashMap<String, Array>> {
    let checkpoint = Checkpoint::open(dir)?;
    let mut weights = HashMap::new();
    let mut shards = HashMap::<String, Shard>::new();
    for (name, file) in &checkpoint.weight_map {
        let Some(key) = ["vision_tower.", "model.visual.", "visual."]
            .iter()
            .find_map(|p| name.strip_prefix(p))
        else {
            continue;
        };
        if !shards.contains_key(file) {
            shards.insert(file.clone(), Shard::open(&dir.join(file))?);
        }
        let tensor = crate::core::quant::get_tensor(&shards[file], name)?;
        ensure!(
            tensor.dtype() == Dtype::Bfloat16
                || tensor.dtype() == Dtype::Float32
                || tensor.dtype() == Dtype::Float16,
            "quantized vision tensor {name} requires a matching quantized encoder"
        );
        ensure!(
            weights
                .insert(key.to_string(), tensor.as_dtype(Dtype::Bfloat16)?)
                .is_none(),
            "duplicate vision tensor {key}"
        );
    }
    Ok(weights)
}
