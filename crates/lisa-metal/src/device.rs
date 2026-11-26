//! Model-unaware Metal storage, prepared operations and completion fences.
use crate::{
    Array, Dtype, Stream,
    error::{Error, Result},
};
use lisa_backend::*;
use std::sync::Arc;
mod affine_experts;
mod attention;
mod block_attention;
mod delta;
mod indexed_attention;
mod indirect;
mod ordered_f32;
mod packed;
mod program;
mod projection;
mod residual_bf16;
mod sparse_f32;
pub use indirect::IndirectResources;
mod resources;
mod tensor;
pub use resources::{IndirectTable, Residency, SerialWorkspace};

pub fn metal_available() -> bool {
    objc2_metal::MTLCreateSystemDefaultDevice().is_some()
}

#[derive(Clone)]
pub struct MetalDevice {
    stream: Stream,
    runtime: Arc<crate::runtime::MetalRuntime>,
}
impl ReadbackDevice for MetalDevice {
    fn read_f32(&self, tensor: &Array) -> Result<Vec<f32>> {
        if !Arc::ptr_eq(tensor.t.runtime(), &self.runtime) {
            return Err(Error::msg("foreign tensor context"));
        }
        if tensor.dtype() != Dtype::Float32 {
            return Err(Error::msg("read_f32 requires F32 storage"));
        }
        tensor.to_vec1::<f32>()
    }
}
/// Opaque operation handle. Pipeline specialization remains cached by shape,
/// dtype, math mode and source identity inside the existing launch wrappers.
pub struct KernelHandle {
    runtime: Arc<crate::runtime::MetalRuntime>,
    kernel: Kernel,
    constraints: KernelCapability,
    invoke: fn(&Kernel, &[&Array], &Stream) -> Result<Option<Outputs<Array>>>,
}
pub struct Completion {
    runtime: Arc<crate::runtime::MetalRuntime>,
    id: u64,
}
impl MetalDevice {
    pub fn on_stream(stream: &Stream) -> Self {
        Self {
            runtime: stream.runtime().clone(),
            stream: stream.clone(),
        }
    }
    pub fn system_default() -> Self {
        Self::on_stream(&Stream::thread_local_or_default())
    }
}
const BF16: TensorConstraint = TensorConstraint {
    dtypes: &[ScalarType::BF16],
    min_rank: 1,
    max_rank: None,
    layout: LayoutSupport::RowContiguous,
    alignment: 2,
};
const FLOAT: TensorConstraint = TensorConstraint {
    dtypes: &[ScalarType::BF16, ScalarType::F32],
    min_rank: 1,
    max_rank: None,
    layout: LayoutSupport::RowContiguous,
    alignment: 2,
};
const PACKED: TensorConstraint = TensorConstraint {
    dtypes: &[ScalarType::U32],
    min_rank: 2,
    max_rank: Some(2),
    layout: LayoutSupport::RowContiguous,
    alignment: 4,
};
const META: TensorConstraint = TensorConstraint {
    min_rank: 2,
    max_rank: Some(2),
    ..FLOAT
};
static CAPABILITIES: [KernelCapability; 4] = [
    KernelCapability {
        kind: KernelKind::AffineSerialLinear,
        law: NumericalLaw::AffineSerialV1,
        inputs: &[FLOAT, PACKED, META, META],
        variadic: false,
        shape: ShapeRelation::Affine,
        encoding: Some(AffineEncoding {
            word: ScalarType::U32,
            bits: &[1, 2, 4, 8],
            group_sizes: &[32, 64, 128],
        }),
        workspace: Workspace::ProviderManaged,
    },
    KernelCapability {
        kind: KernelKind::AffineLinear,
        law: NumericalLaw::AffineV1,
        inputs: &[FLOAT, PACKED, META, META],
        variadic: false,
        shape: ShapeRelation::Affine,
        encoding: Some(AffineEncoding {
            word: ScalarType::U32,
            bits: &[1, 2, 4, 8],
            group_sizes: &[32, 64, 128],
        }),
        workspace: Workspace::ProviderManaged,
    },
    KernelCapability {
        kind: KernelKind::SwiGlu,
        law: NumericalLaw::Bf16SwiGluV1,
        inputs: &[BF16, BF16],
        variadic: false,
        shape: ShapeRelation::EqualInputs,
        encoding: None,
        workspace: Workspace::ProviderManaged,
    },
    KernelCapability {
        kind: KernelKind::PackedSwiGlu,
        law: NumericalLaw::Bf16SwiGluV1,
        inputs: &[BF16],
        variadic: false,
        shape: ShapeRelation::PackedHalves,
        encoding: None,
        workspace: Workspace::ProviderManaged,
    },
];
fn dtype(value: ScalarType) -> Dtype {
    match value {
        ScalarType::Bool => Dtype::Bool,
        ScalarType::U8 => Dtype::Uint8,
        ScalarType::U32 => Dtype::Uint32,
        ScalarType::I32 => Dtype::Int32,
        ScalarType::I64 => Dtype::Int64,
        ScalarType::F16 => Dtype::Float16,
        ScalarType::BF16 => Dtype::Bfloat16,
        ScalarType::U16 => Dtype::Uint16,
        ScalarType::I8 => Dtype::Int8,
        ScalarType::I16 => Dtype::Int16,
        ScalarType::F64 => Dtype::Float64,
        ScalarType::F32 => Dtype::Float32,
    }
}
impl Device for MetalDevice {
    fn gpu_cores(&self) -> Option<u32> {
        Some(crate::jit::native_geometry::gpu_core_count() as u32)
    }
    fn apple_gpu_family(&self) -> Option<u32> {
        Some(crate::runtime::apple_gpu_family()).filter(|&family| family > 0)
    }
    fn metal4(&self) -> bool {
        crate::runtime::metal4_supported()
    }
    fn hold_resident(&self, tensors: &[&Array]) -> Option<Box<dyn std::any::Any + Send + Sync>> {
        let set = Arc::new(crate::runtime::ResidencySet::new(
            self.runtime.device(),
            tensors.len(),
        )?);
        for tensor in tensors {
            if Arc::ptr_eq(tensor.t.runtime(), &self.runtime) {
                set.add(tensor.t.buffer());
            }
        }
        // Scratch, sequence state and step outputs come from the pool: they
        // stay wired too.
        self.runtime.pool.keep_resident(&set);
        set.commit();
        set.hold();
        Some(Box::new(set))
    }
    type Tensor = Array;
    type KernelHandle = KernelHandle;
    type Stream = Stream;
    type Completion = Completion;
    type Error = Error;
    fn capabilities(&self) -> &[KernelCapability] {
        static ALL: std::sync::OnceLock<Vec<KernelCapability>> = std::sync::OnceLock::new();
        ALL.get_or_init(|| {
            let mut all = CAPABILITIES.to_vec();
            all.push(KernelCapability {
                kind: KernelKind::AffineDequantize,
                law: NumericalLaw::AffineDequantizeV1,
                inputs: &[
                    TensorConstraint {
                        max_rank: None,
                        ..PACKED
                    },
                    TensorConstraint {
                        max_rank: None,
                        ..META
                    },
                    TensorConstraint {
                        max_rank: None,
                        ..META
                    },
                ],
                variadic: false,
                shape: ShapeRelation::TensorOperation,
                encoding: Some(AffineEncoding {
                    word: ScalarType::U32,
                    bits: &[1, 2, 4, 8],
                    group_sizes: &[32, 64, 128],
                }),
                workspace: Workspace::ProviderManaged,
            });
            all.push(KernelCapability {
                kind: KernelKind::AffineQuantize,
                law: NumericalLaw::AffineQuantizeV1,
                inputs: &[FLOAT],
                variadic: false,
                shape: ShapeRelation::TensorOperation,
                encoding: Some(AffineEncoding {
                    word: ScalarType::U32,
                    bits: &[2, 4, 8],
                    group_sizes: &[32, 64, 128],
                }),
                workspace: Workspace::ProviderManaged,
            });
            all.extend(tensor::capabilities());
            all.extend(delta::capabilities());
            all.extend(attention::capabilities());
            all.extend(projection::capabilities());
            all.extend(ordered_f32::capabilities());
            all.extend(packed::capabilities());
            all.extend(sparse_f32::capabilities(
                self.runtime.device_name().contains("M3 Ultra"),
            ));
            all.extend(residual_bf16::capabilities());
            all.extend(affine_experts::capabilities());
            all.extend(indirect::capabilities());
            all.extend(block_attention::capabilities());
            all.extend(indexed_attention::capabilities());
            for kind in [KernelKind::Zeros, KernelKind::DeviceZeros] {
                all.push(KernelCapability {
                    kind,
                    law: NumericalLaw::TensorV1,
                    inputs: &[],
                    variadic: false,
                    shape: ShapeRelation::TensorOperation,
                    encoding: None,
                    workspace: Workspace::ProviderManaged,
                });
            }
            all
        })
    }
    fn dtype(&self, tensor: &Array) -> Option<ScalarType> {
        Some(match tensor.dtype() {
            Dtype::Bool => ScalarType::Bool,
            Dtype::Uint8 => ScalarType::U8,
            Dtype::Uint32 => ScalarType::U32,
            Dtype::Int32 => ScalarType::I32,
            Dtype::Int64 => ScalarType::I64,
            Dtype::Float16 => ScalarType::F16,
            Dtype::Bfloat16 => ScalarType::BF16,
            Dtype::Float32 => ScalarType::F32,
            Dtype::Uint16 => ScalarType::U16,
            Dtype::Int8 => ScalarType::I8,
            Dtype::Int16 => ScalarType::I16,
            Dtype::Float64 => ScalarType::F64,
        })
    }
    fn stream(&self) -> Stream {
        self.stream.clone()
    }
    fn shape<'a>(&self, tensor: &'a Array) -> &'a [i32] {
        tensor.shape()
    }
    fn pull(&self, source: TensorSource<'_>, options: ImportOptions) -> Result<Array> {
        let host = source
            .host_tensor()
            .filter(|t| t.validate())
            .ok_or_else(|| Error::msg("pull: invalid tensor size or storage range"))?;
        let dims: Vec<usize> = host.shape.iter().map(|&d| d as usize).collect();
        let dt = dtype(host.dtype);
        let (buffer, offset) = match options.mode {
            ImportMode::Copy | ImportMode::CopyCompact => {
                let buffer = if options.mode == ImportMode::CopyCompact {
                    self.runtime
                        .buffer_for_compact_full_write(host.bytes.len(), "array")?
                } else {
                    self.runtime
                        .buffer_for_full_write(host.bytes.len(), "array")?
                };
                // SAFETY: `buffer` is fresh shared storage of at least `host.bytes.len()` bytes with
                // no GPU user; the validated source slice is live for this call and cannot overlap it.
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        host.bytes.as_ptr(),
                        buffer.contents(),
                        host.bytes.len(),
                    );
                }
                (buffer, 0)
            }
            ImportMode::SharedRequired => {
                let TensorSource::Shared {
                    storage, offset, ..
                } = &source
                else {
                    return Err(Error::msg("pull: shared import requires an owned source"));
                };
                if offset % dt.size_of() != 0 {
                    return Err(Error::msg("pull: unaligned tensor offset"));
                }
                let buffer = self.runtime.buffer_from_external(
                    storage.pointer().cast(),
                    storage.len(),
                    storage.owner(),
                    "tensor_import",
                )?;
                (buffer, offset / dt.size_of())
            }
        };
        Ok(Array::new(crate::array::Array::from_parts_offset(
            &self.runtime,
            buffer,
            &dims,
            dt,
            offset,
        )))
    }
    fn prepare(&self, kernel: Kernel) -> Result<Option<KernelHandle>> {
        let Some(constraints) = self
            .capabilities()
            .iter()
            .find(|c| c.accepts(&kernel))
            .copied()
        else {
            return Ok(None);
        };
        let invoke = match &kernel {
            Kernel::Tensor(_) => tensor::invoke as _,
            Kernel::OrderedF32(_) => ordered_f32::invoke as _,
            Kernel::PackedCompute(_) => packed::invoke as _,
            Kernel::SparseF32(_) => sparse_f32::invoke as _,
            Kernel::ResidualBf16(_) => residual_bf16::invoke as _,
            Kernel::AffineExperts(_) => affine_experts::invoke as _,
            Kernel::IndirectAffine(_) => indirect::without_resources as _,
            Kernel::IndexedAttention(_) => indexed_attention::invoke as _,
            Kernel::BlockAttention(_) => block_attention::invoke as _,
            Kernel::Affine4WideLinear => projection::invoke as _,
            Kernel::DeltaDecode(_)
            | Kernel::DeltaPrep(_)
            | Kernel::DeltaScan { .. }
            | Kernel::GatedRms { .. } => delta::invoke as _,
            Kernel::Zeros { .. } | Kernel::DeviceZeros { .. } => zeros as _,
            Kernel::KeyedGumbel { .. }
            | Kernel::CopyRowsInto { .. }
            | Kernel::CopyRowsConcatInto { .. }
            | Kernel::QkNormRotary { .. }
            | Kernel::SigmoidGateTail { .. } => attention::invoke as _,
            Kernel::HostTensor(_) => return Ok(None),
            Kernel::AffineDequantize { .. } => affine_dequantize as _,
            Kernel::AffineQuantize { .. } => affine_quantize as _,
            Kernel::AffineLinear { .. } | Kernel::AffineSerialLinear { .. } => affine as _,
            Kernel::SwiGlu => {
                crate::kernels::attention::swiglu2_kernel()?;
                swiglu as _
            }
            Kernel::PackedSwiGlu => {
                crate::kernels::gated_pointwise::swiglu2_packed_kernel()?;
                packed_swiglu as _
            }
        };
        Ok(Some(KernelHandle {
            runtime: self.runtime.clone(),
            kernel,
            constraints,
            invoke,
        }))
    }
    fn dispatch(
        &self,
        handle: &KernelHandle,
        inputs: &[&Array],
        stream: &Stream,
    ) -> Result<Option<Dispatch<Outputs<Array>, Completion>>> {
        self.dispatch_resources(handle, inputs, stream, None)
    }
    fn wait(&self, completion: &Completion) -> Result<()> {
        if !Arc::ptr_eq(&completion.runtime, &self.runtime) {
            return Err(Error::msg("wait: foreign completion"));
        }
        if completion.id == 0 {
            return Ok(());
        }
        self.runtime.commands.flush_wait_through(completion.id)
    }
    fn capture_program(
        &self,
        inputs: &[&Array],
        stream: &Stream,
        encode: &mut dyn FnMut() -> anyhow::Result<Vec<Array>>,
    ) -> anyhow::Result<ProgramCapture<Array>> {
        program::capture(self, inputs, stream, encode)
    }
}
fn affine(kernel: &Kernel, inputs: &[&Array], _stream: &Stream) -> Result<Option<Outputs<Array>>> {
    let (group_size, bits, serial) = match kernel {
        Kernel::AffineLinear { group_size, bits } => (*group_size, *bits, None),
        Kernel::AffineSerialLinear {
            group_size,
            bits,
            allow_verify_splitk,
        } => (*group_size, *bits, Some(*allow_verify_splitk)),
        _ => unreachable!(),
    };
    let output = {
        let [x, w, scales, biases] = inputs else {
            unreachable!()
        };
        if x.rank() == 0 || w.rank() != 2 {
            return Err(Error::msg(
                "affine linear expects a ranked input and matrix weights",
            ));
        }
        if group_size <= 0 || !matches!(bits, 1 | 2 | 4 | 8) || x.shape().iter().any(|&d| d <= 0) {
            return Err(Error::msg(
                "affine linear: invalid shape or quantization geometry",
            ));
        }

        if w.dtype() != Dtype::Uint32
            || !matches!(x.dtype(), Dtype::Bfloat16 | Dtype::Float32)
            || scales.dtype() != x.dtype()
            || biases.dtype() != x.dtype()
            || scales.rank() != 2
            || biases.shape() != scales.shape()
            || scales.dim(0) != w.dim(0)
            || x.dim(-1) % group_size != 0
            || scales.dim(1) != x.dim(-1) / group_size
            || i64::from(w.dim(1)) * 32 != i64::from(x.dim(-1)) * i64::from(bits)
        {
            return Err(Error::msg(
                "affine linear: incompatible packed matrix or metadata",
            ));
        }
        let shape = x.shape();
        let rows = shape[..shape.len() - 1].iter().product::<i32>().max(1);
        let flat = x.reshape(&[rows, x.dim(-1)])?;
        let out = if let Some(allow_verify_splitk) = serial {
            crate::ops::quantized_matmul_serial_capture(
                &flat,
                w,
                scales,
                Some(biases),
                true,
                group_size,
                bits,
                allow_verify_splitk && (x.rank() <= 2 || x.dim(0) == 1),
            )?
        } else {
            crate::ops::quantized_matmul(&flat, w, scales, Some(biases), true, group_size, bits)?
        };
        out.reshape(&[&shape[..shape.len() - 1], &[w.dim(0)]].concat())?
    };
    Ok(Some(Outputs::One(output)))
}
fn swiglu(_kernel: &Kernel, inputs: &[&Array], stream: &Stream) -> Result<Option<Outputs<Array>>> {
    let output = {
        let [g, u] = inputs else { unreachable!() };
        if g.dtype() != Dtype::Bfloat16
            || u.dtype() != Dtype::Bfloat16
            || g.shape() != u.shape()
            || g.rank() == 0
            || g.dim(-1) <= 0
        {
            return Err(Error::msg("SwiGlu requires equal nonempty BF16 shapes"));
        }
        match crate::kernels::attention::try_swiglu2(g, u, stream)? {
            Some(a) => a.reshape(g.shape())?,
            None => crate::nn::silu(g)?.multiply(u)?,
        }
    };
    Ok(Some(Outputs::One(output)))
}
fn packed_swiglu(
    _kernel: &Kernel,
    inputs: &[&Array],
    stream: &Stream,
) -> Result<Option<Outputs<Array>>> {
    if inputs[0].rank() == 0 || inputs[0].dim(-1) <= 0 {
        return Ok(None);
    }
    crate::kernels::gated_pointwise::try_swiglu2_packed(inputs[0], stream)
        .map(|out| out.map(Outputs::One))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        alloc::{Layout, alloc_zeroed, dealloc},
        ptr::NonNull,
    };

    struct Pages {
        pointer: NonNull<u8>,
        layout: Layout,
    }
    // SAFETY: `Pages` uniquely owns its allocation, which is only read after initialisation
    // and freed once in `Drop`; shared access is read-only, so Send and Sync hold.
    unsafe impl Send for Pages {}
    unsafe impl Sync for Pages {}
    impl Drop for Pages {
        fn drop(&mut self) {
            // SAFETY: `pointer` came from `alloc_zeroed(layout)` with this same `layout` and is freed only here.
            unsafe {
                dealloc(self.pointer.as_ptr(), self.layout);
            }
        }
    }

    #[test]
    fn compact_pull_preserves_bytes_without_retaining_large_scratch() -> Result<()> {
        let device = MetalDevice::system_default();
        device.runtime.synchronize()?;
        let large = device
            .runtime
            .buffer_for_audited_compact_gpu_write(23 * 1024 * 1024, "import_test_scratch")?;
        let large_id = large.id();
        assert!(large.reusable_after(device.runtime.commands.completed_watermark()));
        drop(large);
        let bits = [0x80000000u32, 0x3f800001, 0x7fc01234];
        let bytes: Vec<_> = bits.iter().flat_map(|v| v.to_le_bytes()).collect();
        let tensor = device.pull(
            HostTensor {
                bytes: &bytes,
                shape: &[3],
                dtype: ScalarType::F32,
            }
            .into(),
            ImportOptions {
                mode: ImportMode::CopyCompact,
            },
        )?;
        assert_eq!(tensor.t.buffer().length(), 1024);
        assert_ne!(tensor.t.buffer().id(), large_id);
        // Inspect the copied payload directly: no conversion may canonicalize
        // signed zero, low mantissa bits or the NaN payload.
        assert_eq!(
            // SAFETY: the copy import finished synchronously into this idle shared buffer, which
            // holds at least `bytes.len()` bytes.
            unsafe { std::slice::from_raw_parts(tensor.t.buffer().contents(), bytes.len()) },
            bytes
        );
        Ok(())
    }

    #[test]
    fn shared_pull_keeps_owner_through_queued_kernel_and_completion() -> Result<()> {
        unsafe extern "C" {
            fn getpagesize() -> i32;
        }
        // SAFETY: `getpagesize` takes no arguments and only reports the VM page size.
        let page = unsafe { getpagesize() } as usize;
        let layout = Layout::from_size_align(page, page).unwrap();
        // SAFETY: `layout` is one non-zero, page-aligned page; null is rejected by `NonNull::new`.
        let pointer = NonNull::new(unsafe { alloc_zeroed(layout) }).unwrap();
        let owner = Arc::new(Pages { pointer, layout });
        for i in 0..128 {
            // SAFETY: 128 u16 writes stay inside the zeroed page, which nothing else accesses yet.
            unsafe {
                pointer.as_ptr().cast::<u16>().add(i).write(0x3f80);
            }
        }
        let weak = Arc::downgrade(&owner);
        // SAFETY: the page is owned by `owner`, is not written after this point and stays
        // mapped until the last clone of `owner` is dropped.
        let storage = unsafe { HostStorage::new(pointer, page, owner.clone()) };
        let device = MetalDevice::system_default();
        let input = device.pull(
            TensorSource::Shared {
                storage,
                offset: 0,
                shape: &[1, 128],
                dtype: ScalarType::BF16,
            },
            ImportOptions {
                mode: ImportMode::SharedRequired,
            },
        )?;
        assert_eq!(input.t.buffer().contents(), pointer.as_ptr());
        drop(owner);
        let handle = device.prepare(Kernel::SwiGlu)?.unwrap();
        let result = device
            .dispatch(&handle, &[&input, &input], &device.stream())?
            .unwrap();
        drop(input);
        // The command encoder retains shared input resources even when every
        // tensor handle and the original host owner have been dropped.
        assert!(weak.upgrade().is_some());
        device.wait(&result.completion)?;
        let values = result
            .output
            .into_one()
            .map_err(Error::msg)?
            .as_dtype(Dtype::Float32)?;
        assert!(values.as_slice::<f32>().iter().all(|&v| v == 0.73046875));
        Ok(())
    }

    #[test]
    fn residency_shrink_drains_pending_shared_reads_and_preserves_private_cap() -> anyhow::Result<()>
    {
        unsafe extern "C" {
            fn getpagesize() -> i32;
        }
        // SAFETY: `getpagesize` takes no arguments and only reports the VM page size.
        let page = unsafe { getpagesize() } as usize;
        let layout = Layout::from_size_align(page, page)?;
        // SAFETY: `layout` is one non-zero, page-aligned page; null is rejected by `NonNull::new`.
        let pointer = NonNull::new(unsafe { alloc_zeroed(layout) }).unwrap();
        let owner = Arc::new(Pages { pointer, layout });
        for i in 0..128 {
            // SAFETY: 128 u16 writes stay inside the zeroed page, which nothing else accesses yet.
            unsafe {
                pointer.as_ptr().cast::<u16>().add(i).write(0x3f80);
            }
        }
        let weak = Arc::downgrade(&owner);
        let device = MetalDevice::system_default();
        device.submit(true)?;
        let baseline = device.memory_usage()?.active;
        // Keep both source+copy charge during import. Shrinking cannot admit
        // the pending source's bytes before all queued users have completed.
        let cap = baseline + 4 * page;
        let mut reservation = device.reserve_residency(3 * page, cap)?;
        // SAFETY: the page is owned by `owner`, is not written after this point and stays
        // mapped until the last clone of `owner` is dropped.
        let storage = unsafe { HostStorage::new(pointer, page, owner.clone()) };
        let input = device.pull(
            TensorSource::Shared {
                storage,
                offset: 0,
                shape: &[1, 128],
                dtype: ScalarType::BF16,
            },
            ImportOptions {
                mode: ImportMode::SharedRequired,
            },
        )?;
        let handle = device.prepare(Kernel::SwiGlu)?.unwrap();
        let result = device
            .dispatch(&handle, &[&input, &input], &device.stream())?
            .unwrap();
        drop(input);
        drop(owner);
        assert!(
            weak.upgrade().is_some(),
            "pending Metal reads retain the independent owner"
        );
        assert!(device.shrink_residency(&mut reservation, page)?);
        device.wait(&result.completion)?;
        let output = result.output.into_one().map_err(anyhow::Error::msg)?;
        assert!(
            output
                .as_dtype(Dtype::Float32)?
                .as_slice::<f32>()
                .iter()
                .all(|&x| x == 0.73046875)
        );
        assert!(
            device.shrink_residency(&mut reservation, 2 * page).is_err(),
            "reservation growth is never admitted"
        );
        assert!(device.shrink_residency(&mut reservation, 0)?);
        assert!(device.runtime.pool.has_resident_admission());
        assert!(
            device
                .runtime
                .buffer_for_compact_full_write(cap + page, "shrink_must_keep_cap")
                .is_err()
        );
        drop(reservation);
        Ok(())
    }

    #[test]
    fn pull_and_prepare_reject_unsupported_requests() -> Result<()> {
        let device = MetalDevice::system_default();
        assert!(
            device
                .prepare(Kernel::AffineLinear {
                    group_size: 3,
                    bits: 4
                })?
                .is_none()
        );
        assert!(
            device
                .prepare(Kernel::AffineLinear {
                    group_size: 64,
                    bits: 3
                })?
                .is_none()
        );
        assert!(
            device
                .pull(
                    HostTensor {
                        bytes: &[0],
                        shape: &[1],
                        dtype: ScalarType::BF16
                    }
                    .into(),
                    ImportOptions::default()
                )
                .is_err()
        );
        assert!(
            device
                .pull(
                    HostTensor {
                        bytes: &[0],
                        shape: &[1],
                        dtype: ScalarType::U8
                    }
                    .into(),
                    ImportOptions {
                        mode: ImportMode::SharedRequired
                    }
                )
                .is_err()
        );
        let handle = device.prepare(Kernel::SwiGlu)?.unwrap();
        let bad = Array::from_slice(&[1.0f32; 8], &[1, 8]);
        assert!(
            device
                .dispatch(&handle, &[&bad, &bad], &device.stream())
                .is_err()
        );
        assert!(device.dispatch(&handle, &[&bad], &device.stream()).is_err());
        let packed = device.prepare(Kernel::PackedSwiGlu)?.unwrap();
        let odd = Array::from_slice(&[1.0f32; 7], &[1, 7]).as_dtype(Dtype::Bfloat16)?;
        assert!(
            device
                .dispatch(&packed, &[&odd], &device.stream())?
                .is_none()
        );
        Ok(())
    }
}

fn zeros(kernel: &Kernel, _inputs: &[&Array], stream: &Stream) -> Result<Option<Outputs<Array>>> {
    let (Kernel::Zeros { shape, dtype: dt } | Kernel::DeviceZeros { shape, dtype: dt }) = kernel
    else {
        unreachable!()
    };
    let shape = shape
        .iter()
        .map(|&d| usize::try_from(d).map_err(|_| Error::msg("negative zero tensor dimension")))
        .collect::<Result<Vec<_>>>()?;
    let array = if matches!(kernel, Kernel::DeviceZeros { .. }) {
        crate::array::Array::zeros_on_device(stream.runtime(), &shape, dtype(*dt))?
    } else {
        crate::array::Array::zeros(stream.runtime(), &shape, dtype(*dt))?
    };
    Ok(Some(Outputs::One(Array::new(array))))
}

fn affine_dequantize(
    kernel: &Kernel,
    inputs: &[&Array],
    _: &Stream,
) -> Result<Option<Outputs<Array>>> {
    let Kernel::AffineDequantize { group_size, bits } = kernel else {
        unreachable!()
    };
    let [weight, scales, biases] = inputs else {
        unreachable!()
    };
    if weight.rank() < 2
        || scales.rank() != weight.rank()
        || biases.shape() != scales.shape()
        || scales.dtype() != biases.dtype()
        || weight.shape()[..weight.rank() - 1] != scales.shape()[..scales.rank() - 1]
        || i64::from(weight.dim(-1)) * 32
            != i64::from(scales.dim(-1)) * i64::from(*group_size) * i64::from(*bits)
    {
        return Err(Error::msg("incompatible affine dequantization geometry"));
    }
    Ok(Some(Outputs::One(crate::ops::dequantize(
        weight,
        scales,
        biases,
        *group_size,
        *bits,
    )?)))
}

fn affine_quantize(
    kernel: &Kernel,
    inputs: &[&Array],
    _: &Stream,
) -> Result<Option<Outputs<Array>>> {
    let Kernel::AffineQuantize { group_size, bits } = kernel else {
        unreachable!()
    };
    if inputs[0].rank() < 2 || inputs[0].dim(-1) % group_size != 0 {
        return Err(Error::msg("invalid affine quantization geometry"));
    }
    let (w, s, b) = crate::ops::quantize(inputs[0], *group_size, *bits)?;
    Ok(Some(Outputs::Many(vec![w, s, b])))
}

impl MetalDevice {
    fn dispatch_resources(
        &self,
        handle: &KernelHandle,
        inputs: &[&Array],
        stream: &Stream,
        resources: Option<&IndirectResources>,
    ) -> Result<Option<Dispatch<Outputs<Array>, Completion>>> {
        if self.runtime.commands.is_capturing_program() && !program::supported(&handle.kernel) {
            self.runtime.commands.decline_program_capture();
        }
        if inputs.len() != handle.kernel.arity() {
            return Err(Error::msg("dispatch: wrong input count"));
        }
        if !Arc::ptr_eq(&handle.runtime, &self.runtime)
            || !Arc::ptr_eq(stream.runtime(), &self.runtime)
            || inputs
                .iter()
                .any(|a| !Arc::ptr_eq(a.t.runtime(), &self.runtime))
        {
            return Err(Error::msg("dispatch: foreign device context"));
        }
        for (i, input) in inputs.iter().enumerate() {
            let constraint = &handle.constraints.inputs[i.min(handle.constraints.inputs.len() - 1)];
            if !constraint
                .dtypes
                .iter()
                .any(|&dt| dtype(dt) == input.dtype())
                || input.rank() < constraint.min_rank
                || constraint.max_rank.is_some_and(|rank| input.rank() > rank)
            {
                return Err(Error::msg(format!(
                    "dispatch {:?}: input {i} {:?} {:?} violates prepared dtype/rank contract",
                    handle.kernel,
                    input.dtype(),
                    input.shape()
                )));
            }
            if constraint.layout == LayoutSupport::RowContiguous
                && !input.t.buffer_and_layout().1.is_contiguous()
            {
                return Err(Error::msg(
                    "dispatch: tensor violates prepared layout contract",
                ));
            }
        }
        let output = if let Kernel::IndirectAffine(op) = &handle.kernel {
            indirect::invoke(op, inputs, stream, resources)?
        } else {
            (handle.invoke)(&handle.kernel, inputs, stream)?
        };
        let Some(output) = output else {
            return Ok(None);
        };
        let completion = Completion {
            runtime: self.runtime.clone(),
            id: (0..output.len())
                .filter_map(|i| output.get(i))
                .map(|a| a.t.buffer().last_used_cb())
                .max()
                .unwrap_or(0),
        };
        Ok(Some(Dispatch { output, completion }))
    }
}

impl lisa_backend::StorageDevice for MetalDevice {
    fn max_allocation_bytes(&self) -> usize {
        Array::max_buffer_length()
    }
    fn byte_view(
        &self,
        source: &Array,
        offset: usize,
        shape: &[i32],
        scalar: ScalarType,
    ) -> Result<Array> {
        let bytes = lisa_backend::tensor_bytes(shape, scalar)
            .ok_or_else(|| crate::error::Exception::custom("invalid byte-view shape".into()))?;
        let (buffer, layout) = source.t.buffer_and_layout();
        if !layout.is_contiguous()
            || !offset.is_multiple_of(scalar.size_of())
            || offset
                .checked_add(bytes)
                .is_none_or(|end| end > source.size() * source.dtype().size_of())
        {
            crate::bail!("invalid byte-view range or alignment");
        }
        Array::from_buffer_offset(
            buffer.clone(),
            layout.offset * source.dtype().size_of() + offset,
            shape,
            dtype(scalar),
        )
    }
}

impl lisa_backend::HostAccessDevice for MetalDevice {
    fn same_storage(&self, a: &Array, b: &Array) -> bool {
        Arc::ptr_eq(a.t.runtime(), &self.runtime)
            && Arc::ptr_eq(b.t.runtime(), &self.runtime)
            && Arc::ptr_eq(a.t.buffer(), b.t.buffer())
    }
    fn copy_tensor(&self, tensor: &Array) -> anyhow::Result<Array> {
        anyhow::ensure!(
            Arc::ptr_eq(tensor.t.runtime(), &self.runtime),
            "foreign tensor context"
        );
        Ok(Array::new(tensor.t.copied()?))
    }
    fn read_bytes(&self, tensor: &Array) -> anyhow::Result<Vec<u8>> {
        anyhow::ensure!(
            Arc::ptr_eq(tensor.t.runtime(), &self.runtime),
            "foreign tensor context"
        );
        let tensor = tensor.contiguous()?;
        tensor.eval()?;
        let (buffer, layout) = tensor.t.buffer_and_layout();
        let offset = layout.offset * tensor.dtype().size_of();
        let bytes = tensor.size() * tensor.dtype().size_of();
        // SAFETY: `tensor` is contiguous and evaluated, so its retained shared buffer holds
        // `bytes` initialised bytes at `offset` and no GPU work is pending on it.
        Ok(unsafe { std::slice::from_raw_parts(buffer.contents().add(offset), bytes) }.to_vec())
    }
    fn read_i32_through(&self, tensor: &Array) -> anyhow::Result<Vec<i32>> {
        anyhow::ensure!(
            tensor.dtype() == Dtype::Int32 && Arc::ptr_eq(tensor.t.runtime(), &self.runtime),
            "I32 readback requires local I32 storage"
        );
        Ok(tensor.to_vec1_wait_through::<i32>()?)
    }
    fn read_f32_through(&self, tensor: &Array) -> anyhow::Result<Vec<f32>> {
        anyhow::ensure!(
            Arc::ptr_eq(tensor.t.runtime(), &self.runtime),
            "foreign tensor context"
        );
        anyhow::ensure!(
            tensor.dtype() == Dtype::Float32,
            "F32 readback requires F32 storage"
        );
        Ok(tensor.to_vec1_wait_through::<f32>()?)
    }
    fn layout(&self, tensor: &Array) -> lisa_backend::TensorLayout {
        let (_, layout) = tensor.t.buffer_and_layout();
        lisa_backend::TensorLayout {
            strides: layout.strides.clone(),
            offset: layout.offset,
        }
    }
    fn evaluate(&self, tensor: &Array, wait: bool) -> anyhow::Result<()> {
        anyhow::ensure!(
            Arc::ptr_eq(tensor.t.runtime(), &self.runtime),
            "foreign tensor context"
        );
        if wait {
            tensor.eval()?;
        } else {
            tensor.eval_enqueue()?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod erased_tests;

impl TryFrom<Dtype> for ScalarType {
    type Error = Error;
    fn try_from(dtype: Dtype) -> Result<Self> {
        match dtype {
            Dtype::Bool => Ok(Self::Bool),
            Dtype::Uint8 => Ok(Self::U8),
            Dtype::Uint32 => Ok(Self::U32),
            Dtype::Int32 => Ok(Self::I32),
            Dtype::Int64 => Ok(Self::I64),
            Dtype::Float16 => Ok(Self::F16),
            Dtype::Float32 => Ok(Self::F32),
            Dtype::Bfloat16 => Ok(Self::BF16),
            Dtype::Uint16 => Ok(Self::U16),
            Dtype::Int8 => Ok(Self::I8),
            Dtype::Int16 => Ok(Self::I16),
            Dtype::Float64 => Ok(Self::F64),
        }
    }
}

pub enum DeviceScope {
    Ordered(crate::runtime::OrderedGpuReuseScope),
    Verify(crate::ops::VerifySplitkGuard),
}
impl ManagedDevice for MetalDevice {
    fn identity(&self) -> lisa_backend::DeviceIdentity {
        let architecture = self.runtime.architecture_name();
        let mut arithmetic = b"metal:".to_vec();
        arithmetic.extend_from_slice(architecture.as_bytes());
        arithmetic.push(u8::from(self.runtime.nax()));
        lisa_backend::DeviceIdentity {
            provider: "metal".into(),
            device: self.runtime.device_name(),
            architecture,
            arithmetic,
        }
    }
    type Reservation = crate::runtime::ResidentMemoryReservation;
    type Scope = DeviceScope;
    fn memory_usage(&self) -> anyhow::Result<MemoryUsage> {
        Ok(MemoryUsage {
            active: self.runtime.pool.total_bytes(),
            private: self.runtime.pool.bytes(),
        })
    }
    fn reserve_residency(&self, bytes: usize, budget: usize) -> anyhow::Result<Self::Reservation> {
        Ok(self.runtime.reserve_model_memory(bytes, budget)?)
    }
    fn shrink_residency(
        &self,
        reservation: &mut Self::Reservation,
        bytes: usize,
    ) -> anyhow::Result<bool> {
        reservation.shrink_to(&self.runtime.pool, bytes)?;
        Ok(true)
    }
    fn execution_boundary(&self) {
        self.runtime.commands.flush_encoder();
    }
    fn submit(&self, wait: bool) -> anyhow::Result<()> {
        if wait {
            self.runtime.commands.flush_and_wait()?;
        } else {
            self.runtime.commands.flush()?;
        }
        Ok(())
    }
    fn submit_after(&self, previous: Option<u64>) -> anyhow::Result<u64> {
        let commands = &self.runtime.commands;
        let mark = commands.current_cb_id();
        commands.flush()?;
        if let Some(previous) = previous {
            commands.flush_wait_through(previous)?;
        }
        Ok(mark)
    }
    fn enter_scope(&self, scope: ExecutionScope) -> Self::Scope {
        match scope {
            ExecutionScope::OrderedOutputReuse => {
                DeviceScope::Ordered(crate::runtime::enter_ordered_gpu_reuse_scope())
            }
            ExecutionScope::AffineVerifySplitK => {
                DeviceScope::Verify(crate::ops::enter_verify_splitk_scope())
            }
        }
    }
    fn clear_unused(&self) {
        let _ = self.runtime.synchronize();
        self.runtime
            .pool
            .sweep_completed(self.runtime.commands.completed_watermark());
    }
    fn trim_unused(&self) {
        crate::ops::memory::trim_cache_on(&self.runtime);
    }
}
