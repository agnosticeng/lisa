//! Tensor access and view operations, independent of models or inference policy.
pub trait TensorAccess: Clone {
    fn shape(&self) -> &[i32];
    fn rank(&self) -> usize {
        self.shape().len()
    }
    fn dim(&self, axis: i32) -> i32 {
        self.shape()[axis.rem_euclid(self.rank() as i32) as usize]
    }
    fn size(&self) -> usize {
        self.shape().iter().map(|&d| d as usize).product()
    }
    fn select(&self, axis: i32, row: i32) -> anyhow::Result<Self>;
    fn slice(&self, axis: i32, start: i32, end: Option<i32>) -> anyhow::Result<Self>;
    fn reshape(&self, shape: &[i32]) -> anyhow::Result<Self>;
    fn expand_dims(&self, axis: i32) -> anyhow::Result<Self>;
    fn contiguous(&self) -> anyhow::Result<Self>;
    /// Independent storage, retaining the original element bits and shape.
    fn independent_copy(&self) -> anyhow::Result<Self>;
    fn eval(&self) -> anyhow::Result<()>;
    fn read_raw_ids(&self) -> anyhow::Result<Vec<u32>>;
    fn read_ids(&self) -> anyhow::Result<Vec<u32>>;
    fn read_f32(&self) -> anyhow::Result<Vec<f32>>;
    fn argmax_id(&self) -> anyhow::Result<u32>;
    fn argmax_axis(&self, axis: i32) -> anyhow::Result<Self>;
    fn concatenate(tensors: &[&Self], axis: i32) -> anyhow::Result<Self>;
}

/// Host interaction with opaque device storage. No implicit scalar conversion.
pub trait HostAccessDevice: crate::StorageDevice {
    /// Whether views retain the same backing allocation. False is conservative.
    fn same_storage(&self, _a: &Self::Tensor, _b: &Self::Tensor) -> bool {
        false
    }
    fn read_bytes(&self, tensor: &Self::Tensor) -> anyhow::Result<Vec<u8>>;
    /// Independent, byte-preserving storage for a snapshot. Providers may queue
    /// a device copy; the default imports the synchronized logical payload.
    fn copy_tensor(&self, tensor: &Self::Tensor) -> anyhow::Result<Self::Tensor> {
        let bytes = self.read_bytes(tensor)?;
        let dtype = self
            .dtype(tensor)
            .ok_or_else(|| anyhow::anyhow!("unknown snapshot storage type"))?;
        Ok(self.pull(
            crate::HostTensor {
                bytes: &bytes,
                shape: self.shape(tensor),
                dtype,
            }
            .into(),
            Default::default(),
        )?)
    }
    /// Read I32 selector storage through its last producer without casting.
    fn read_i32_through(&self, tensor: &Self::Tensor) -> anyhow::Result<Vec<i32>> {
        anyhow::ensure!(
            self.dtype(tensor) == Some(crate::ScalarType::I32),
            "I32 readback requires I32 storage"
        );
        let bytes = self.read_bytes(tensor)?;
        anyhow::ensure!(bytes.len() % 4 == 0, "unaligned I32 readback");
        Ok(bytes
            .chunks_exact(4)
            .map(|b| i32::from_ne_bytes(b.try_into().unwrap()))
            .collect())
    }
    /// Read F32 elements after their producer completes. Providers with a serial
    /// queue may leave later work in flight; the default uses a stronger wait.
    fn read_f32_through(&self, tensor: &Self::Tensor) -> anyhow::Result<Vec<f32>> {
        anyhow::ensure!(
            self.dtype(tensor) == Some(crate::ScalarType::F32),
            "F32 readback requires F32 storage"
        );
        let bytes = self.read_bytes(tensor)?;
        anyhow::ensure!(bytes.len() % 4 == 0, "unaligned F32 readback");
        Ok(bytes
            .chunks_exact(4)
            .map(|b| f32::from_ne_bytes(b.try_into().unwrap()))
            .collect())
    }
    fn layout(&self, tensor: &Self::Tensor) -> TensorLayout;
    /// Submit producers without waiting when `wait` is false; otherwise make
    /// host access to this tensor safe. Do not wait while holding allocator locks.
    fn evaluate(&self, tensor: &Self::Tensor, wait: bool) -> anyhow::Result<()>;
}

/// Strides and offset are in elements, matching the scalar interpretation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorLayout {
    pub strides: Vec<usize>,
    pub offset: usize,
}

impl TensorLayout {
    pub fn is_contiguous(&self, shape: &[i32]) -> bool {
        if shape.len() != self.strides.len() {
            return false;
        }
        let mut expected = 1usize;
        for (&dim, &stride) in shape.iter().zip(&self.strides).rev() {
            let Ok(dim) = usize::try_from(dim) else {
                return false;
            };
            if dim != 1 && stride != expected {
                return false;
            }
            let Some(next) = expected.checked_mul(dim) else {
                return false;
            };
            expected = next;
        }
        true
    }
}
