//! Original Olala checkpoint validation, tensor import and embedding provider.
use super::gguf::{RowEmbedding, load_resident};
use crate::{
    bindings::{BindingDevice, olala::Tower},
    language::runtime::Runtime,
    sampling::sampler::Sampler,
};
use anyhow::{Result, ensure};
use lisa_backend::{ManagedDevice, ReadbackDevice, ScalarType as Dtype};
use lisa_gguf::{Package, TensorType as Type};
use lisa_models::olala::{
    config::{Config, Role},
    graph::Format,
};
use std::{collections::BTreeMap, path::Path};

fn formats(package: &Package, c: &Config) -> Result<BTreeMap<String, Format>> {
    let mut result = BTreeMap::new();
    for (name, shape, role) in lisa_models::olala::config::tensors(c) {
        let t = &package.tensor(&name)?.info;
        ensure!(
            t.dimensions == shape,
            "Olala tensor {name} shape mismatch: {:?}, expected {shape:?}",
            t.dimensions
        );
        let valid = match role {
            Role::Parameter => t.kind == Type::F32,
            Role::ExpertUp => t.kind == Type::Q8_0,
            Role::ExpertDown => matches!(t.kind, Type::Q5_K | Type::Q6_K),
            Role::Projection => matches!(
                t.kind,
                Type::F32 | Type::BF16 | Type::Q5_K | Type::Q6_K | Type::Q8_0
            ),
        };
        ensure!(
            valid,
            "unsupported Olala tensor encoding for {name}: {:?}",
            t.kind
        );
        let dtype = match t.kind {
            Type::F32 => Dtype::F32,
            Type::BF16 => Dtype::BF16,
            _ => Dtype::U8,
        };
        result.insert(
            name,
            Format {
                projection: super::gguf::projection(t.kind)?,
                dtype,
                row_bytes: i32::try_from(t.row_bytes()?)?,
            },
        );
    }
    Ok(result)
}
pub fn load<D>(runtime: Runtime<D, Sampler<D>>, path: &Path) -> Result<Tower<D>>
where
    D: BindingDevice + ManagedDevice + ReadbackDevice,
    D::Error: From<anyhow::Error>,
{
    ensure!(
        path.is_file(),
        "Olala currently requires an original GGUF file"
    );
    let package = Package::open(path)?;
    let config = crate::checkpoint::olala::config(&package.header)?;
    let formats = formats(&package, &config)?;
    let device = runtime.context.device();
    let embedding = package.tensor("token_embd.weight")?.disk_reader()?;
    let names = formats
        .keys()
        .filter(|k| k.as_str() != "token_embd.weight")
        .map(String::as_str)
        .collect::<Vec<_>>();
    let resident = load_resident(device.clone(), &package, &names)?.into_prepared()?;
    let embedding = Box::new(RowEmbedding::new(device.clone(), embedding)?);
    Tower::from_prepared(runtime, config, formats, resident, embedding)
}
