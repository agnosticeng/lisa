//! Immutable zero-copy GGUF weight views with lifetime-scoped RAM admission.

use anyhow::{Context, Result, ensure};
use lisa_gguf::{Package, TensorInfo};
use lisa_mlx::runtime::ResidentMemoryReservation;
use lisa_mlx::{Array, Dtype, Stream};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

#[derive(Clone)]
pub struct PackedTensor {
    /// Original packed bytes, shaped [outer dimensions reversed, row bytes].
    /// Kernel dispatch must use `info.kind`; this is not an affine MLX tensor.
    pub(crate) bytes: Array,
    pub info: TensorInfo,
}

impl PackedTensor {
    /// Typed view of an original unquantized payload, with unchanged byte offset.
    pub fn plain_view(&self) -> Result<Array> {
        let dtype = match self.info.kind {
            lisa_gguf::TensorType::F32 => Dtype::Float32,
            lisa_gguf::TensorType::F16 => Dtype::Float16,
            lisa_gguf::TensorType::BF16 => Dtype::Bfloat16,
            _ => anyhow::bail!("{} is not a plain GGUF tensor", self.info.name),
        };
        let shape: Vec<i32> = self
            .info
            .dimensions
            .iter()
            .rev()
            .map(|&d| i32::try_from(d))
            .collect::<std::result::Result<_, _>>()?;
        let (buffer, layout) = self.packed_bytes().t.buffer_and_layout();
        Ok(Array::from_buffer_offset(
            buffer.clone(),
            layout.offset,
            &shape,
            dtype,
        )?)
    }

    /// Borrow the immutable payload while the owning ResidentWeights lives.
    /// Model code must keep that owner for all encoded uses of the view.
    pub fn packed_bytes(&self) -> &Array {
        &self.bytes
    }
}

pub struct ResidentWeights {
    // Drain on the owning thread before dropping any weight buffers. Putting
    // this lease in a Metal deallocator could wait on its own completion.
    _reservation: ResidentMemoryReservation,
    pub tensors: BTreeMap<String, PackedTensor>,
    pub mapped_bytes: usize,
}

impl ResidentWeights {
    /// `names` is explicit: the graph must exclude disk-only tables. Budgeting
    /// happens before the first mmap or GPU view, including page padding.
    /// Runtime allocations are charged separately by the same buffer pool.
    pub fn load(package: &Package, names: &[&str], stream: &Stream) -> Result<Self> {
        ensure!(!names.is_empty(), "empty resident GGUF tensor list");
        let mut seen = BTreeSet::new();
        let mut mapped_bytes = 0usize;
        for &name in names {
            ensure!(seen.insert(name), "duplicate resident GGUF tensor {name}");
            let tensor = package.tensor(name)?;
            mapped_bytes = mapped_bytes
                .checked_add(tensor.mapped_region_bytes()?)
                .context("resident GGUF size overflow")?;
        }
        let budget = lisa_mlx::memory::ram_budget();
        ensure!(
            mapped_bytes <= budget,
            "resident GGUF weights need {} GiB, RAM budget is {} GiB",
            mapped_bytes as f64 / (1u64 << 30) as f64,
            budget as f64 / (1u64 << 30) as f64
        );
        let reservation = stream
            .runtime()
            .reserve_model_memory(mapped_bytes, budget)?;
        let mut tensors = BTreeMap::new();
        for &name in names {
            let tensor = package.tensor(name)?;
            let (mapping, inner) = tensor.mapped_region()?;
            let pointer = std::ptr::NonNull::new(mapping.as_ptr() as *mut std::ffi::c_void)
                .context("null GGUF mapping")?;
            let length = mapping.len();
            let owner: Arc<dyn std::any::Any + Send + Sync> = mapping;
            let buffer =
                stream
                    .runtime()
                    .buffer_from_external(pointer, length, owner, "gguf_weight")?;
            let mut shape: Vec<i32> = tensor.info.dimensions[1..]
                .iter()
                .rev()
                .map(|&n| i32::try_from(n))
                .collect::<std::result::Result<_, _>>()?;
            shape.push(i32::try_from(tensor.info.row_bytes()?)?);
            let bytes = Array::from_buffer_offset(buffer, inner, &shape, Dtype::Uint8)?;
            tensors.insert(
                name.to_owned(),
                PackedTensor {
                    bytes,
                    info: tensor.info.clone(),
                },
            );
        }
        Ok(Self {
            _reservation: reservation,
            tensors,
            mapped_bytes,
        })
    }
}
