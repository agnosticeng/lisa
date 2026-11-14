//! Validate original checkpoint payloads against model-owned requirements.
use anyhow::{Context, Result, ensure};
use lisa_gguf::{Package, TensorType};
use lisa_models::deepseek41::{
    config::Config,
    engram::Layout,
    weights::{Storage, requirements},
};
use std::collections::BTreeSet;

#[derive(Debug)]
pub struct WeightPlan {
    pub resident: Vec<String>,
    pub disk: Vec<String>,
    pub resident_payload_bytes: u64,
    pub disk_payload_bytes: u64,
}
impl WeightPlan {
    pub fn inspect(package: &Package, config: &Config, engram: &Layout) -> Result<Self> {
        let mut seen = BTreeSet::new();
        let mut resident = Vec::new();
        let mut disk = Vec::new();
        for requirement in requirements(config, engram) {
            let name = requirement.name;
            if requirement.optional && !package.tensors.contains_key(&name) {
                continue;
            }
            let tensor = package.tensor(&name)?;
            ensure!(
                tensor
                    .info
                    .dimensions
                    .iter()
                    .rev()
                    .eq(requirement.shape.iter()),
                "DeepSeek tensor {name} does not match its required shape"
            );
            let kind = tensor.info.kind;
            match requirement.storage {
                Storage::F32 => {
                    ensure!(kind == TensorType::F32, "{name} must retain F32 parameters")
                }
                Storage::Iq4Nl => ensure!(
                    kind == TensorType::IQ4_NL,
                    "unsupported Engram table {name}"
                ),
                Storage::Mxfp8 => {
                    ensure!(kind == TensorType(24), "unsupported Engram table {name}")
                }
                Storage::Experts => {
                    super::deepseek41_projection::block_format(kind)
                        .with_context(|| format!("unsupported expert tensor {name}"))?;
                }
                Storage::Projection => {
                    let projection = super::packed::projection(kind)?;
                    ensure!(
                        lisa_models::deepseek41::projection::supported(projection),
                        "unsupported projection format {kind:?} for {name}"
                    );
                    if matches!(projection, lisa_models::packed::Projection::Block(_)) {
                        ensure!(
                            requirement.shape[0] % 8 == 0,
                            "packed projection {name} requires complete eight-row tiles"
                        );
                    }
                }
            }
            seen.insert(name.clone());
            if requirement.disk {
                disk.push(name);
            } else {
                resident.push(name);
            }
        }
        // Preserve the old loader's resident mapping order and on-disk table order.
        resident.sort();
        let unknown: Vec<&str> = package
            .tensors
            .keys()
            .filter(|n| !seen.contains(*n))
            .map(String::as_str)
            .collect();
        ensure!(
            unknown.is_empty(),
            "unbound DeepSeek GGUF tensors: {unknown:?}; additional draft/vision tensors require their own graph"
        );
        let bytes = |names: &[String]| {
            names.iter().try_fold(0u64, |total, name| {
                total
                    .checked_add(package.tensors[name].info.bytes)
                    .context("DeepSeek payload size overflow")
            })
        };
        Ok(Self {
            resident_payload_bytes: bytes(&resident)?,
            disk_payload_bytes: bytes(&disk)?,
            resident,
            disk,
        })
    }
}
