//! Immutable zero-copy GGUF weight views with lifetime-scoped RAM admission.

use super::{ResidencyBytes, ResidencyPolicy};
use anyhow::{Context, Result, ensure};
use lisa_backend::{ScalarType as Dtype, StorageDevice};
use lisa_gguf::{Package, TensorInfo};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

pub use super::prepared::PreparedLayout;

pub fn projection(kind: lisa_gguf::TensorType) -> Result<lisa_models::packed::Projection> {
    use lisa_backend::PackedEncoding as E;
    use lisa_gguf::TensorType as T;
    use lisa_models::packed::Projection as P;
    Ok(match kind {
        T::F32 | T::F16 | T::BF16 => P::Plain,
        T::Q8_0 => P::Q8,
        T::IQ2_XXS => P::Block(E::Iq2Xxs),
        T::IQ2_XS => P::Block(E::Iq2Xs),
        T::IQ1_M => P::Block(E::Iq1M),
        T::Q5_K => P::Block(E::Q5K),
        T::Q6_K => P::Block(E::Q6K),
        T::Q2_K => P::Block(E::Q2K),
        T::Q4_K => P::Block(E::Q4K),
        T::NVFP4 => P::Block(E::Nvfp4),
        _ => anyhow::bail!("unsupported packed compute encoding {kind:?}"),
    })
}

pub fn prepared_layout(info: &TensorInfo) -> Result<PreparedLayout> {
    ensure!(
        !info.dimensions.is_empty() && info.dimensions.iter().all(|&n| n > 0),
        "invalid packed tensor dimensions"
    );
    let logical_shape = info
        .dimensions
        .iter()
        .rev()
        .map(|&n| i32::try_from(n))
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let row_bytes = i32::try_from(info.row_bytes()?)?;
    let dtype = match info.kind {
        lisa_gguf::TensorType::F32 => Dtype::F32,
        lisa_gguf::TensorType::F16 => Dtype::F16,
        lisa_gguf::TensorType::BF16 => Dtype::BF16,
        _ => Dtype::U8,
    };
    let storage_shape = if dtype == Dtype::U8 {
        let mut bytes = logical_shape[..logical_shape.len() - 1].to_vec();
        bytes.push(row_bytes);
        bytes
    } else {
        // Plain compute views are typed elements; row_bytes still describes
        // their original byte span. Only packed U8 views use byte dimensions.
        logical_shape.clone()
    };
    Ok(PreparedLayout {
        projection: projection(info.kind)?,
        dtype,
        logical_shape,
        storage_shape,
        row_bytes,
    })
}

/// Original row decoder retained behind a tensor-producing loading adapter.
/// Its arithmetic and import policy are unchanged from the legacy F32 path.
pub struct RowEmbedding<D: StorageDevice> {
    device: D,
    source: lisa_gguf::Tensor,
    width: usize,
    row_bytes: usize,
}
impl<D: StorageDevice> RowEmbedding<D> {
    pub fn new(device: D, source: lisa_gguf::Tensor) -> Result<Self> {
        ensure!(
            source.info.dimensions.len() == 2,
            "embedding must be a matrix"
        );
        let width = usize::try_from(source.info.dimensions[0])?;
        let row_bytes = usize::try_from(source.info.row_bytes()?)?;
        Ok(Self {
            device,
            source,
            width,
            row_bytes,
        })
    }
}
impl<D: StorageDevice + Send + Sync> super::source::EmbeddingRows<D::Tensor> for RowEmbedding<D> {
    fn row(&self, token: u32) -> Result<D::Tensor> {
        let mut bytes = vec![0; self.row_bytes];
        self.source.read_rows(&[u64::from(token)], &mut bytes)?;
        let mut values = vec![0.; self.width];
        lisa_gguf::quant::decode(self.source.info.kind, &bytes, &mut values)?;
        let bytes = values
            .iter()
            .flat_map(|v| v.to_ne_bytes())
            .collect::<Vec<_>>();
        Ok(self.device.pull(
            lisa_backend::HostTensor {
                bytes: &bytes,
                shape: &[1, i32::try_from(self.width)?],
                dtype: Dtype::F32,
            }
            .into(),
            Default::default(),
        )?)
    }
}

#[derive(Clone)]
pub struct PackedTensor<D: StorageDevice> {
    pub(crate) device: D,
    /// Original packed bytes, shaped [outer dimensions reversed, row bytes].
    /// Kernel dispatch must use `info.kind`; this is not an affine MLX tensor.
    pub(crate) bytes: D::Tensor,
    pub info: TensorInfo,
}

impl<D: StorageDevice> PackedTensor<D> {
    /// Adapt file metadata once, retaining the exact imported payload.
    pub fn into_prepared(self) -> Result<super::prepared::PreparedTensor<D::Tensor>> {
        let layout = self.prepared_layout()?;
        let plain = if layout.projection == lisa_models::packed::Projection::Plain {
            Some(self.plain_view()?)
        } else {
            None
        };
        super::prepared::PreparedTensor::new(layout, self.bytes, plain)
    }
    pub fn prepared_layout(&self) -> Result<PreparedLayout> {
        prepared_layout(&self.info)
    }
    /// Construct an already imported packed tensor for device conformance tests.
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
    /// Remove all file-reader metadata before handing weights to execution.
    pub fn into_prepared(self) -> Result<super::prepared::PreparedWeights<D::Tensor, L>>
    where
        D::Tensor: Clone,
    {
        let mut prepared = super::prepared::PreparedWeights {
            reservation: self._reservation,
            tensors: BTreeMap::new(),
            mapped_bytes: self.mapped_bytes,
            residency: self.residency,
        };
        // Borrow the original owners until every fallible view is prepared.
        // On error this local drops first, draining the reservation before
        // either its new views or the original bank's buffers are released.
        for (name, tensor) in &self.tensors {
            let layout = tensor.prepared_layout()?;
            let plain = if layout.projection == lisa_models::packed::Projection::Plain {
                Some(tensor.plain_view()?)
            } else {
                None
            };
            prepared.tensors.insert(
                name.clone(),
                super::prepared::PreparedTensor::new(layout, tensor.bytes.clone(), plain)?,
            );
        }
        Ok(prepared)
    }
    /// Transfer the unchanged residency lease and original views to a loading
    /// adapter that materializes its container-independent tensor bank.
    pub(crate) fn into_parts(self) -> (L, BTreeMap<String, PackedTensor<D>>) {
        (self._reservation, self.tensors)
    }

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
            // SAFETY: `owner` is the exact read-only page-rounded GGUF mapping, whose
            // last page is partly file-backed and so readable; checkpoint files stay
            // immutable while mapped, and the device retains `owner` through last use.
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

#[cfg(test)]
mod prepared_tests {
    use super::*;
    #[test]
    fn prepared_packed_layout_retains_original_encoding_and_byte_geometry() -> Result<()> {
        use lisa_backend::PackedEncoding as E;
        use lisa_gguf::TensorType as T;
        use lisa_models::packed::Projection as P;
        for (kind, width, bytes, projection) in [
            (T::Q5_K, 2304, 1584, P::Block(E::Q5K)),
            (T::Q8_0, 2304, 2448, P::Q8),
            (T::NVFP4, 2304, 1296, P::Block(E::Nvfp4)),
        ] {
            let info = TensorInfo {
                name: "expert".into(),
                dimensions: vec![width, 2304, 128],
                kind,
                offset: 4096,
                bytes: bytes * 2304 * 128,
            };
            let layout = prepared_layout(&info)?;
            assert_eq!(layout.projection, projection);
            assert_eq!(layout.dtype, Dtype::U8);
            assert_eq!(layout.logical_shape, [128, 2304, 2304]);
            assert_eq!(layout.storage_shape, [128, 2304, bytes as i32]);
            assert_eq!(layout.row_bytes, bytes as i32);
            assert_eq!(info.kind, kind);
            assert_eq!(info.offset, 4096);
        }
        for (kind, dtype, element_bytes) in [
            (T::F32, Dtype::F32, 4),
            (T::F16, Dtype::F16, 2),
            (T::BF16, Dtype::BF16, 2),
        ] {
            let info = TensorInfo {
                name: "plain".into(),
                dimensions: vec![32, 3],
                kind,
                offset: 4096,
                bytes: 32 * 3 * element_bytes,
            };
            let layout = prepared_layout(&info)?;
            assert_eq!(layout.projection, P::Plain);
            assert_eq!(layout.dtype, dtype);
            assert_eq!(layout.storage_shape, [3, 32]);
            assert_eq!(layout.logical_shape, layout.storage_shape);
            assert_eq!(layout.row_bytes as u64, 32 * element_bytes);
        }
        Ok(())
    }
}
