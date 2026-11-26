//! Tensor storage imports, independent of any checkpoint or tensor role.
use crate::ScalarType;
use std::{any::Any, ptr::NonNull, sync::Arc};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ImportMode {
    /// Copy exactly the supplied bytes. The source read finishes before return.
    #[default]
    Copy,
    /// Copy exactly the supplied bytes into compact storage. The source read
    /// finishes before return. Retained capacity may include allocator bucket
    /// padding, but must not inherit an unrelated larger reusable allocation.
    CopyCompact,
    /// Retain shared storage without copying. Unsupported alignment/storage is
    /// an error; the engine decides whether a separate copy is acceptable.
    SharedRequired,
}
#[derive(Clone, Copy, Debug, Default)]
pub struct ImportOptions {
    pub mode: ImportMode,
}

/// Immutable, stable, externally owned host memory. The owner is retained by
/// the imported allocation through the last device use, including queued work.
#[derive(Clone)]
pub struct HostStorage {
    pointer: NonNull<u8>,
    length: usize,
    owner: Arc<dyn Any + Send + Sync>,
}
// SAFETY: `new` requires immutable, stable memory, and `owner` is Send + Sync;
// moving the handle to another thread moves only that retained ownership.
unsafe impl Send for HostStorage {}
// SAFETY: shared access only reads the immutable bytes or clones the
// thread-safe `owner` Arc.
unsafe impl Sync for HostStorage {}
impl HostStorage {
    /// # Safety
    /// `pointer..pointer+length` must remain readable and immutable until the
    /// last clone of `owner` is released. `owner` must own that exact allocation.
    /// Page padding is allowed only if it is mapped and readable. Device writes
    /// to imported storage are forbidden. Rust heap memory must not be presented
    /// as a larger page-rounded allocation.
    pub unsafe fn new(
        pointer: NonNull<u8>,
        length: usize,
        owner: Arc<dyn Any + Send + Sync>,
    ) -> Self {
        Self {
            pointer,
            length,
            owner,
        }
    }
    pub fn pointer(&self) -> NonNull<u8> {
        self.pointer
    }
    pub fn len(&self) -> usize {
        self.length
    }
    pub fn is_empty(&self) -> bool {
        self.length == 0
    }
    pub fn owner(&self) -> Arc<dyn Any + Send + Sync> {
        self.owner.clone()
    }
    pub fn bytes(&self) -> &[u8] {
        // SAFETY: the constructor contract and retained owner cover this slice.
        unsafe { std::slice::from_raw_parts(self.pointer.as_ptr(), self.length) }
    }
}

pub enum TensorSource<'a> {
    Borrowed(crate::HostTensor<'a>),
    Shared {
        storage: HostStorage,
        offset: usize,
        shape: &'a [i32],
        dtype: ScalarType,
    },
}
impl<'a> From<crate::HostTensor<'a>> for TensorSource<'a> {
    fn from(value: crate::HostTensor<'a>) -> Self {
        Self::Borrowed(value)
    }
}
impl TensorSource<'_> {
    pub fn host_tensor(&self) -> Option<crate::HostTensor<'_>> {
        match self {
            Self::Borrowed(t) => Some(crate::HostTensor {
                bytes: t.bytes,
                shape: t.shape,
                dtype: t.dtype,
            }),
            Self::Shared {
                storage,
                offset,
                shape,
                dtype,
            } => {
                let len = crate::tensor_bytes(shape, *dtype)?;
                let end = offset.checked_add(len)?;
                Some(crate::HostTensor {
                    bytes: storage.bytes().get(*offset..end)?,
                    shape,
                    dtype: *dtype,
                })
            }
        }
    }
}

/// Byte-preserving views of an imported allocation. This contract has no
/// checkpoint or tensor-role knowledge. Views retain the source allocation,
/// cannot exceed its logical byte range, and do not change its mutability.
pub trait StorageDevice: crate::Device {
    fn max_allocation_bytes(&self) -> usize;
    fn byte_view(
        &self,
        source: &Self::Tensor,
        offset: usize,
        shape: &[i32],
        dtype: ScalarType,
    ) -> Result<Self::Tensor, Self::Error>;
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn shared_tensor_checks_range_and_retains_exact_storage() {
        let bytes = Arc::new(vec![1u8, 2, 3, 4, 5, 6, 7, 8]);
        let weak = Arc::downgrade(&bytes);
        let pointer = NonNull::new(bytes.as_ptr() as *mut u8).unwrap();
        // SAFETY: the Arc<Vec<u8>> owner holds exactly this immutable buffer.
        let storage = unsafe { HostStorage::new(pointer, bytes.len(), bytes.clone()) };
        drop(bytes);
        let valid = TensorSource::Shared {
            storage: storage.clone(),
            offset: 2,
            shape: &[2],
            dtype: ScalarType::U8,
        };
        assert_eq!(valid.host_tensor().unwrap().bytes, &[3, 4]);
        for (offset, shape) in [
            (7, &[2][..]),
            (usize::MAX, &[1][..]),
            (0, &[-1][..]),
            (0, &[i32::MAX, i32::MAX, i32::MAX][..]),
        ] {
            let source = TensorSource::Shared {
                storage: storage.clone(),
                offset,
                shape,
                dtype: ScalarType::F32,
            };
            assert!(source.host_tensor().is_none());
        }
        drop(storage);
        assert!(weak.upgrade().is_some());
        drop(valid);
        assert!(weak.upgrade().is_none());
    }
}
