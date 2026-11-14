//! Explicit device binding for individual tensor operations used by engine policy.
use lisa_backend::{
    Device, HostAccessDevice, HostTensor, Kernel, ReadbackDevice, ScalarType, TensorOperation,
};

/// Callers select an arithmetic contract; unsupported laws never fall back.
#[derive(Clone, Copy)]
pub enum TensorLaw {
    Native,
    OrderedF32,
}

pub struct TensorOps<'a, D: Device> {
    pub device: &'a D,
    stream: D::Stream,
    law: TensorLaw,
}
impl<'a, D: Device> TensorOps<'a, D> {
    pub fn new(device: &'a D, law: TensorLaw) -> Self {
        Self {
            device,
            stream: device.stream(),
            law,
        }
    }
    pub fn optional(
        &self,
        kernel: Kernel,
        inputs: &[&D::Tensor],
    ) -> anyhow::Result<Option<D::Tensor>> {
        let Some(handle) = self.device.prepare(kernel)? else {
            return Ok(None);
        };
        self.device
            .dispatch(&handle, inputs, &self.stream)?
            .map(|v| v.output.into_one().map_err(anyhow::Error::msg))
            .transpose()
    }
    pub fn required(&self, kernel: Kernel, inputs: &[&D::Tensor]) -> anyhow::Result<D::Tensor> {
        let kind = kernel.kind();
        self.optional(kernel, inputs)?
            .ok_or_else(|| anyhow::anyhow!("device declined required tensor operation {kind:?}"))
    }
    pub fn op(
        &self,
        operation: TensorOperation,
        inputs: &[&D::Tensor],
    ) -> anyhow::Result<D::Tensor> {
        self.required(
            match self.law {
                TensorLaw::Native => Kernel::Tensor(operation),
                TensorLaw::OrderedF32 => Kernel::HostTensor(operation),
            },
            inputs,
        )
    }
    pub fn pull(
        &self,
        bytes: &[u8],
        shape: &[i32],
        dtype: ScalarType,
    ) -> anyhow::Result<D::Tensor> {
        Ok(self.device.pull(
            HostTensor {
                bytes,
                shape,
                dtype,
            }
            .into(),
            Default::default(),
        )?)
    }
    pub fn scalar(&self, value: f32) -> anyhow::Result<D::Tensor> {
        self.pull(&value.to_ne_bytes(), &[], ScalarType::F32)
    }
    /// Preserve the broadcast-then-materialize chain used by native full().
    pub fn full(&self, shape: &[i32], value: f32) -> anyhow::Result<D::Tensor> {
        let scalar = self.scalar(value)?;
        let broadcast = self.op(TensorOperation::Broadcast(shape.to_vec()), &[&scalar])?;
        self.op(TensorOperation::Contiguous, &[&broadcast])
    }
    pub fn flat_f32(&self, tensor: &D::Tensor) -> anyhow::Result<D::Tensor> {
        let flat = self.op(TensorOperation::Reshape(vec![-1]), &[tensor])?;
        self.op(TensorOperation::Cast(ScalarType::F32), &[&flat])
    }
}
impl<D: HostAccessDevice> TensorOps<'_, D> {
    pub fn read_ids(&self, tensor: &D::Tensor) -> anyhow::Result<Vec<u32>> {
        anyhow::ensure!(
            matches!(
                self.device.dtype(tensor),
                Some(ScalarType::I32 | ScalarType::U32)
            ),
            "token IDs require 32-bit integer storage"
        );
        let bytes = self.device.read_bytes(tensor)?;
        anyhow::ensure!(bytes.len() % 4 == 0, "unaligned token IDs");
        Ok(bytes
            .chunks_exact(4)
            .map(|b| u32::from_ne_bytes(b.try_into().unwrap()))
            .collect())
    }
    pub fn id(&self, tensor: &D::Tensor) -> anyhow::Result<u32> {
        let ids = self.read_ids(tensor)?;
        anyhow::ensure!(ids.len() == 1, "expected one token ID");
        Ok(ids[0])
    }
}
impl<D: ReadbackDevice> TensorOps<'_, D> {
    pub fn read_f32(&self, tensor: &D::Tensor) -> anyhow::Result<Vec<f32>> {
        Ok(self.device.read_f32(tensor)?)
    }
    pub fn value(&self, tensor: &D::Tensor) -> anyhow::Result<f32> {
        let values = self.read_f32(tensor)?;
        anyhow::ensure!(values.len() == 1, "expected one scalar");
        Ok(values[0])
    }
}
