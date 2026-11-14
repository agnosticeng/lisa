//! Device-independent checked views of cache and workspace storage.
use anyhow::{Result, ensure};
use lisa_backend::{HostAccessDevice, ScalarType};

/// Retain a contiguous subrange without copying or changing its scalar type.
/// `first` counts elements relative to the source view, not its allocation.
pub fn view<D: HostAccessDevice>(
    device: &D,
    source: &D::Tensor,
    first: usize,
    shape: &[i32],
) -> Result<D::Tensor> {
    let dtype = device
        .dtype(source)
        .ok_or_else(|| anyhow::anyhow!("unknown storage type"))?;
    ensure!(
        matches!(dtype, ScalarType::F32 | ScalarType::BF16 | ScalarType::I32),
        "Tensor view requires contiguous F32, BF16 or I32 storage"
    );
    let source_shape = device.shape(source);
    let layout = device.layout(source);
    ensure!(
        layout.is_contiguous(source_shape),
        "Tensor view requires contiguous storage"
    );
    let count = shape.iter().try_fold(1usize, |n, &d| {
        ensure!(d > 0, "invalid Tensor view shape");
        n.checked_mul(d as usize)
            .ok_or_else(|| anyhow::anyhow!("state view size overflow"))
    })?;
    let size = lisa_backend::tensor_bytes(source_shape, dtype)
        .ok_or_else(|| anyhow::anyhow!("invalid source shape"))?
        / dtype.size_of();
    ensure!(
        first <= size && count <= size - first,
        "Tensor view out of bounds"
    );
    device.evaluate(source, false)?;
    Ok(device.byte_view(
        source,
        first
            .checked_mul(dtype.size_of())
            .ok_or_else(|| anyhow::anyhow!("state view offset overflow"))?,
        shape,
        dtype,
    )?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use lisa_backend::{Device, HostTensor, ReadbackDevice, erased::DeviceHandle};
    #[test]
    fn nested_views_retain_offsets_and_storage_after_source_drop() -> Result<()> {
        let device = DeviceHandle::new(lisa_cpu::CpuDevice);
        let bytes: Vec<u8> = [99f32, 1., 2., 3., 4., 88.]
            .iter()
            .flat_map(|v| v.to_ne_bytes())
            .collect();
        let source = device.pull(
            HostTensor {
                bytes: &bytes,
                shape: &[6],
                dtype: ScalarType::F32,
            }
            .into(),
            Default::default(),
        )?;
        let outer = view(&device, &source, 1, &[2, 2])?;
        let inner = view(&device, &outer, 1, &[2])?;
        assert!(view(&device, &outer, 3, &[2]).is_err());
        assert!(view(&device, &outer, usize::MAX, &[1]).is_err());
        assert!(view(&device, &outer, 0, &[-1]).is_err());
        drop(source);
        drop(outer);
        assert_eq!(device.read_f32(&inner)?, [2., 3.]);
        Ok(())
    }
}
