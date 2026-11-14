//! Checkpoint I/O and device bindings shared by the vision plans.
use crate::{
    bindings::BindingDevice,
    execution::DeviceContext,
    tensor_ops::{TensorLaw, TensorOps},
    weights::mlx::{Checkpoint, Shard},
};
use anyhow::{Context, ensure};
use lisa_backend::{ScalarType as Dtype, TensorOperation};
use std::{collections::HashMap, path::Path};
pub(super) fn get<T: Clone>(w: &HashMap<String, T>, name: &str) -> anyhow::Result<T> {
    w.get(name)
        .cloned()
        .with_context(|| format!("missing vision tensor {name}"))
}

pub(super) fn load_weights<D: BindingDevice>(
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
        let tensor = crate::weights::source::get_tensor(&shards[file], name)?;
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
