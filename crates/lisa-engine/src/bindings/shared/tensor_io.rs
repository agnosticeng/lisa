//! Explicit provider access for typed host captures and storage-preserving casts.
use crate::bindings::BindingDevice;
use crate::execution::DeviceContext;
use crate::tensor_ops::{TensorLaw, TensorOps};
use lisa_backend::{ScalarType, TensorAccess, TensorOperation};
pub trait CaptureTensor: TensorAccess {
    fn read_u16<D: BindingDevice<Tensor = Self>>(
        &self,
        context: &DeviceContext<D>,
    ) -> anyhow::Result<Vec<u16>> {
        anyhow::ensure!(
            matches!(
                context.device().dtype(self),
                Some(ScalarType::U16 | ScalarType::F16 | ScalarType::BF16)
            ),
            "capture requires 16-bit storage"
        );
        let bytes = context.device().read_bytes(self)?;
        anyhow::ensure!(bytes.len() % 2 == 0, "unaligned capture");
        Ok(bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|v| u16::from_ne_bytes(*v))
            .collect())
    }
    fn read_i32<D: BindingDevice<Tensor = Self>>(
        &self,
        _context: &DeviceContext<D>,
    ) -> anyhow::Result<Vec<i32>> {
        Ok(self.read_raw_ids()?.into_iter().map(|x| x as i32).collect())
    }
    fn transpose_on<D: BindingDevice<Tensor = Self>>(
        &self,
        context: &DeviceContext<D>,
        axes: &[i32],
    ) -> anyhow::Result<Self> {
        TensorOps::new(context.device(), TensorLaw::Native)
            .op(TensorOperation::Transpose(axes.to_vec()), &[self])
    }
    fn cast_on<D: BindingDevice<Tensor = Self>>(
        &self,
        context: &DeviceContext<D>,
        dtype: ScalarType,
    ) -> anyhow::Result<Self> {
        TensorOps::new(context.device(), TensorLaw::Native)
            .op(TensorOperation::Cast(dtype), &[self])
    }
}
impl<T: TensorAccess> CaptureTensor for T {}
pub fn pull_i32<D: BindingDevice>(
    context: &DeviceContext<D>,
    data: &[i32],
    shape: &[i32],
) -> anyhow::Result<D::Tensor> {
    let bytes = data
        .iter()
        .flat_map(|x| x.to_ne_bytes())
        .collect::<Vec<_>>();
    Ok(context.device().pull(
        lisa_backend::HostTensor {
            bytes: &bytes,
            shape,
            dtype: ScalarType::I32,
        }
        .into(),
        Default::default(),
    )?)
}
pub fn pull_f32<D: BindingDevice>(
    context: &DeviceContext<D>,
    data: &[f32],
    shape: &[i32],
) -> anyhow::Result<D::Tensor> {
    let bytes = data
        .iter()
        .flat_map(|x| x.to_ne_bytes())
        .collect::<Vec<_>>();
    Ok(context.device().pull(
        lisa_backend::HostTensor {
            bytes: &bytes,
            shape,
            dtype: ScalarType::F32,
        }
        .into(),
        Default::default(),
    )?)
}
