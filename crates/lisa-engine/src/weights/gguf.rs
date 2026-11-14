//! Immutable zero-copy GGUF weight views with lifetime-scoped RAM admission.

use super::{ResidencyBytes, ResidencyPolicy};
use anyhow::{Context, Result, ensure};
use lisa_backend::{ScalarType as Dtype, StorageDevice};
use lisa_gguf::{Package, TensorInfo};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

#[derive(Clone)]
pub struct PackedTensor<D: StorageDevice> {
    pub(crate) device: D,
    /// Original packed bytes, shaped [outer dimensions reversed, row bytes].
    /// Kernel dispatch must use `info.kind`; this is not an affine MLX tensor.
    pub(crate) bytes: D::Tensor,
    pub info: TensorInfo,
}

impl<D: StorageDevice> PackedTensor<D> {
    /// Construct an already imported packed tensor for device conformance tests.
    #[cfg(any(test, feature = "diagnostics"))]
    pub fn from_parts_for_test(device: D, bytes: D::Tensor, info: TensorInfo) -> Self {
        Self {
            device,
            bytes,
            info,
        }
    }

    /// Typed view of an original unquantized payload, with unchanged byte offset.
    pub fn plain_view(&self) -> Result<D::Tensor> {
        let dtype = match self.info.kind {
            lisa_gguf::TensorType::F32 => Dtype::F32,
            lisa_gguf::TensorType::F16 => Dtype::F16,
            lisa_gguf::TensorType::BF16 => Dtype::BF16,
            _ => anyhow::bail!("{} is not a plain GGUF tensor", self.info.name),
        };
        let shape: Vec<i32> = self
            .info
            .dimensions
            .iter()
            .rev()
            .map(|&d| i32::try_from(d))
            .collect::<std::result::Result<_, _>>()?;
        Ok(self.device.byte_view(&self.bytes, 0, &shape, dtype)?)
    }

    /// Borrow the immutable payload while the owning ResidentWeights lives.
    /// Model code must keep that owner for all encoded uses of the view.
    pub fn packed_bytes(&self) -> &D::Tensor {
        &self.bytes
    }
}

pub struct ResidentWeights<D: StorageDevice, L> {
    // Drain on the owning thread before dropping any weight buffers. Putting
    // this lease in a Metal deallocator could wait on its own completion.
    _reservation: L,
    pub tensors: BTreeMap<String, PackedTensor<D>>,
    pub mapped_bytes: usize,
    pub residency: ResidencyBytes,
}

impl<D: StorageDevice + Clone, L> ResidentWeights<D, L> {
    /// `names` is explicit: the graph must exclude disk-only tables. Budgeting
    /// happens before the first mmap or GPU view, including page padding.
    /// Runtime allocations are charged separately by the same buffer pool.
    pub fn pull(
        device: D,
        package: &Package,
        names: &[&str],
        reserve: impl FnOnce(usize) -> Result<L>,
    ) -> Result<Self> {
        Self::pull_with_policy(device, package, names, ResidencyPolicy::Mapped, |bytes| {
            reserve(bytes.host_resident)
        })
    }

    /// Admit the selected host/device memory costs before any import. Uploads
    /// preserve packed encodings; no quantization or format conversion occurs.
    pub fn pull_with_policy(
        device: D,
        package: &Package,
        names: &[&str],
        policy: ResidencyPolicy,
        reserve: impl FnOnce(ResidencyBytes) -> Result<L>,
    ) -> Result<Self> {
        ensure!(!names.is_empty(), "empty resident GGUF tensor list");
        let mut seen = BTreeSet::new();
        let mut mapped_bytes = 0usize;
        let mut residency = ResidencyBytes::default();
        for &name in names {
            ensure!(seen.insert(name), "duplicate resident GGUF tensor {name}");
            let tensor = package.tensor(name)?;
            mapped_bytes = mapped_bytes
                .checked_add(tensor.mapped_region_bytes()?)
                .context("resident GGUF size overflow")?;
            match policy {
                ResidencyPolicy::Mapped => residency.host_resident = mapped_bytes,
                ResidencyPolicy::Uploaded => {
                    let bytes = usize::try_from(tensor.info.bytes)?;
                    residency.device_resident = residency
                        .device_resident
                        .checked_add(bytes)
                        .context("uploaded byte count overflow")?;
                    residency.host_staging =
                        residency.host_staging.max(tensor.mapped_region_bytes()?);
                }
            }
        }
        let reservation = reserve(residency)?;
        let mut tensors = BTreeMap::new();
        for &name in names {
            let tensor = package.tensor(name)?;
            let (mapping, inner) = tensor.mapped_region()?;
            let pointer =
                std::ptr::NonNull::new(mapping.as_ptr() as *mut u8).context("null GGUF mapping")?;
            let length = mapping.len();
            let owner: Arc<dyn std::any::Any + Send + Sync> = mapping;
            // The GGUF adapter supplies bytes and a retained mapping only.
            let storage = unsafe { lisa_backend::HostStorage::new(pointer, length, owner) };
            let mut shape: Vec<i32> = tensor.info.dimensions[1..]
                .iter()
                .rev()
                .map(|&n| i32::try_from(n))
                .collect::<std::result::Result<_, _>>()?;
            shape.push(i32::try_from(tensor.info.row_bytes()?)?);
            let bytes = device.pull(
                lisa_backend::TensorSource::Shared {
                    storage,
                    offset: inner,
                    shape: &shape,
                    dtype: lisa_backend::ScalarType::U8,
                },
                lisa_backend::ImportOptions {
                    mode: policy.import_mode(),
                },
            )?;
            tensors.insert(
                name.to_owned(),
                PackedTensor {
                    device: device.clone(),
                    bytes,
                    info: tensor.info.clone(),
                },
            );
        }
        Ok(Self {
            _reservation: reservation,
            tensors,
            mapped_bytes,
            residency,
        })
    }
}

/// Admit and import mapped weights using only provider-owned allocation leases.
pub fn load_resident<D: lisa_backend::ManagedDevice + Clone>(
    device: D,
    package: &Package,
    names: &[&str],
) -> Result<ResidentWeights<D, D::Reservation>> {
    ResidentWeights::pull(device.clone(), package, names, |mapped_bytes| {
        let budget = crate::memory::ram_budget();
        ensure!(
            mapped_bytes <= budget,
            "resident GGUF weights need {} GiB, RAM budget is {} GiB",
            mapped_bytes as f64 / (1u64 << 30) as f64,
            budget as f64 / (1u64 << 30) as f64
        );
        device.reserve_residency(mapped_bytes, budget)
    })
}
