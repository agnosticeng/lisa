//! Device contracts only: no model, checkpoint parser, plan executor or policy.
mod capability;
mod indirect;
pub use indirect::*;
mod affine_experts;
mod block_attention;
mod delta;
mod import;
mod indexed_attention;
mod ordered_f32;
mod packed;
mod residual_bf16;
mod sparse_f32;
pub use affine_experts::*;
pub use block_attention::*;
pub use indexed_attention::*;
pub use packed::*;
pub use residual_bf16::*;
pub use sparse_f32::*;
mod refinement;
pub use ordered_f32::*;
mod tensor;
pub use capability::*;
pub use delta::*;
pub use import::*;
pub use refinement::*;
pub use tensor::*;

/// Scalar storage interpretation. Packed quantized weights remain U32/bytes;
/// quantization parameters belong to a kernel invocation, never an implicit cast.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScalarType {
    Bool,
    U8,
    U16,
    U32,
    I8,
    I16,
    I32,
    I64,
    F16,
    BF16,
    F32,
    F64,
}
impl ScalarType {
    pub const fn size_of(self) -> usize {
        match self {
            Self::Bool | Self::U8 | Self::I8 => 1,
            Self::F16 | Self::BF16 | Self::U16 | Self::I16 => 2,
            Self::U32 | Self::I32 | Self::F32 => 4,
            Self::I64 | Self::F64 => 8,
        }
    }
}

/// A checked host tensor request. `pull` completes its source read before return;
/// asynchronous import needs a separate owned-source/completion contract.
pub struct HostTensor<'a> {
    pub bytes: &'a [u8],
    pub shape: &'a [i32],
    pub dtype: ScalarType,
}
impl HostTensor<'_> {
    pub fn validate(&self) -> bool {
        tensor_bytes(self.shape, self.dtype) == Some(self.bytes.len())
    }
}

/// Exact existing MLX-affine numerical semantics, independent of hardware.
/// A backend must reject an unsupported contract rather than silently changing
/// casts or reduction order. Additional numerical laws need distinct variants.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Kernel {
    Tensor(TensorOperation),
    Refinement(Refinement),
    OrderedF32(OrderedF32),
    PackedCompute(PackedCompute),
    SparseF32(SparseF32),
    ResidualBf16(ResidualBf16),
    AffineExperts(AffineExperts),
    IndirectAffine(IndirectAffine),
    BlockAttention(BlockAttention),
    IndexedAttention(IndexedAttention),
    /// MLX 0.32.4 BF16 small-row projection with its wide/thin reduction law.
    Bf16SmallRowLinear,
    /// MLX 0.32.4 Q4/group-64 projection, including its F32 split-K accumulation.
    Affine4WideLinear,
    /// Strided source rows into contiguous destination storage; preserves bytes.
    /// One keyed Gumbel-max draw over F32 logits; no sampler policy or RNG mutation.
    KeyedGumbel {
        inverse_temperature: u32,
        seed: u64,
        position: u64,
    },
    CopyRowsInto {
        offset: i32,
    },
    /// Concatenate two strided fields directly into a destination row range.
    CopyRowsConcatInto {
        offset: i32,
    },
    /// Exact BF16 Q/K RMS and 64-wide partial rotary, optionally reading an
    /// interleaved Q/gate projection and returning deinterleaved gates.
    QkNormRotary {
        query_heads: i32,
        key_heads: i32,
        rows: i32,
        epsilon: u32,
        interleaved: bool,
    },
    SigmoidGateTail {
        batch: i32,
        rows: i32,
        heads: i32,
        width: i32,
    },
    Zeros {
        shape: Vec<i32>,
        dtype: ScalarType,
    },
    DeltaDecode(DeltaDecode),
    DeltaPrep(DeltaPrep),
    DeltaScan {
        mode: DeltaScan,
        capture: bool,
        reuse: bool,
    },
    GatedRms {
        heads: i32,
        width: i32,
        epsilon: u32,
        swish: bool,
    },
    /// Ordered host f32 arithmetic, including its sequential reductions and libm erf.
    HostTensor(TensorOperation),
    /// Decode original affine packed rows without changing their group metadata.
    AffineDequantize {
        group_size: i32,
        bits: i32,
    },
    /// Existing ordered host affine encoder, including source precision and rounding.
    AffineQuantize {
        group_size: i32,
        bits: i32,
    },
    AffineLinear {
        group_size: i32,
        bits: i32,
    },
    /// BF16 SiLU(gate) * up, including the established intermediate rounding.
    SwiGlu,
    /// The same operation over a row-concatenated gate/up projection.
    PackedSwiGlu,
    /// Sequential delta recurrence returning activation, final state and compact delta history.
    DeltaReplayCapture,
    /// Reapply compact history to the base state for the supplied prefix.
    DeltaReplayRestore {
        prefix: usize,
    },
}
impl Kernel {
    /// Fresh storage suitable for a later explicitly effectful write. Views
    /// and computed immutable values are not writable merely because they are results.
    pub fn requires_resources(&self) -> bool {
        matches!(self, Self::IndirectAffine(op) if op.requires_resources())
    }
    pub fn fresh_output(&self, output: usize) -> bool {
        if let Self::PackedCompute(op) = self {
            return !matches!(op, PackedCompute::GatedResidual4 { .. }) && output < op.outputs();
        }
        output == 0
            && matches!(
                self,
                Self::Zeros { .. } | Self::Tensor(TensorOperation::AssignSlice { .. })
            )
    }
    /// Storage-preserving views used to trace an explicit write back to its
    /// declared input workspace or a fresh allocation in this graph.
    pub fn alias_input(&self, output: usize) -> Option<usize> {
        if output == 0
            && matches!(
                self,
                Self::Tensor(
                    TensorOperation::Reshape(_)
                        | TensorOperation::SplitLastAxis(_)
                        | TensorOperation::Transpose(_)
                        | TensorOperation::Broadcast(_)
                        | TensorOperation::Slice { .. }
                        | TensorOperation::Select { .. }
                )
            )
        {
            Some(0)
        } else {
            None
        }
    }
    /// Caller-supplied output storage overwritten by this invocation. These
    /// effects are explicit and are not rolled back if a later call fails.
    pub fn overwritten_inputs(&self) -> &'static [usize] {
        match self {
            Self::IndirectAffine(op) => op.overwritten_inputs(),
            Self::SparseF32(SparseF32::QueryLatentRmsInto { .. }) => &[4],
            Self::PackedCompute(PackedCompute::GatedResidual4 { .. }) => &[0],
            Self::IndexedAttention(op) => op.overwritten_inputs(),
            Self::CopyRowsConcatInto { .. } => &[2],
            Self::CopyRowsInto { .. } | Self::Tensor(TensorOperation::CopySliceInto { .. }) => &[1],
            Self::DeltaScan {
                mode: DeltaScan::Verify { .. },
                reuse: true,
                ..
            } => &[8, 9],
            Self::DeltaScan { reuse: true, .. } => &[6, 7],
            _ => &[],
        }
    }
    pub fn outputs(&self) -> usize {
        match self {
            Self::Tensor(op) | Self::HostTensor(op) => op.outputs(),
            Self::Refinement(op) => op.outputs(),
            Self::OrderedF32(op) => op.outputs(),
            Self::PackedCompute(op) => op.outputs(),
            Self::SparseF32(op) => op.outputs(),
            Self::AffineExperts(op) => op.outputs(),
            Self::IndirectAffine(op) => op.outputs(),
            Self::IndexedAttention(_) => 1,
            Self::BlockAttention(op) => op.outputs(),
            Self::ResidualBf16(op) => op.outputs(),
            Self::DeltaReplayCapture | Self::DeltaDecode(_) | Self::AffineQuantize { .. } => 3,
            Self::QkNormRotary { interleaved, .. } => {
                if *interleaved {
                    3
                } else {
                    2
                }
            }
            Self::DeltaScan { .. } => 2,
            Self::DeltaPrep(s) => {
                if s.separate {
                    6
                } else {
                    7
                }
            }
            _ => 1,
        }
    }
    pub fn kind(&self) -> KernelKind {
        match self {
            Self::Tensor(op) | Self::HostTensor(op) => KernelKind::Tensor(op.kind()),
            Self::Refinement(op) => KernelKind::Refinement(op.kind()),
            Self::OrderedF32(op) => KernelKind::OrderedF32(op.kind()),
            Self::PackedCompute(op) => KernelKind::PackedCompute(op.kind()),
            Self::SparseF32(op) => KernelKind::SparseF32(op.kind()),
            Self::AffineExperts(op) => KernelKind::AffineExperts(op.kind()),
            Self::IndirectAffine(op) => KernelKind::IndirectAffine(op.kind()),
            Self::IndexedAttention(op) => KernelKind::IndexedAttention(op.kind()),
            Self::BlockAttention(op) => KernelKind::BlockAttention(op.kind()),
            Self::ResidualBf16(op) => KernelKind::ResidualBf16(op.kind()),
            Self::Zeros { .. } => KernelKind::Zeros,
            Self::KeyedGumbel { .. } => KernelKind::KeyedGumbel,
            Self::CopyRowsInto { .. } => KernelKind::CopyRowsInto,
            Self::CopyRowsConcatInto { .. } => KernelKind::CopyRowsConcatInto,
            Self::QkNormRotary { .. } => KernelKind::QkNormRotary,
            Self::SigmoidGateTail { .. } => KernelKind::SigmoidGateTail,
            Self::DeltaScan { mode, reuse, .. } => KernelKind::DeltaScan {
                gated: mode.gated(),
                reuse: *reuse,
            },
            Self::GatedRms { .. } => KernelKind::GatedRms,
            Self::DeltaDecode(_) => KernelKind::DeltaDecode,
            Self::DeltaPrep(s) => {
                if s.separate {
                    KernelKind::DeltaPrepSeparate
                } else {
                    KernelKind::DeltaPrepPacked
                }
            }
            Self::AffineDequantize { .. } => KernelKind::AffineDequantize,
            Self::AffineQuantize { .. } => KernelKind::AffineQuantize,
            Self::AffineLinear { .. } => KernelKind::AffineLinear,
            Self::Bf16SmallRowLinear => KernelKind::Bf16SmallRowLinear,
            Self::Affine4WideLinear => KernelKind::Affine4WideLinear,
            Self::SwiGlu => KernelKind::SwiGlu,
            Self::PackedSwiGlu => KernelKind::PackedSwiGlu,
            Self::DeltaReplayCapture => KernelKind::DeltaReplayCapture,
            Self::DeltaReplayRestore { .. } => KernelKind::DeltaReplayRestore,
        }
    }
    pub fn law(&self) -> NumericalLaw {
        match self {
            Self::Tensor(_) | Self::Zeros { .. } => NumericalLaw::TensorV1,
            Self::HostTensor(_) => NumericalLaw::HostSequentialF32,
            Self::Refinement(_) => NumericalLaw::MlxRefinementV1,
            Self::OrderedF32(_) => NumericalLaw::OrderedF32V1,
            Self::PackedCompute(_) => NumericalLaw::PackedScalarV1,
            Self::SparseF32(_) => NumericalLaw::SparseF32V1,
            Self::AffineExperts(_) => NumericalLaw::AffineExpertsV1,
            Self::IndirectAffine(_) => NumericalLaw::IndirectAffineV1,
            Self::IndexedAttention(_) => NumericalLaw::IndexedAttentionV1,
            Self::BlockAttention(_) => NumericalLaw::BlockAttentionV1,
            Self::ResidualBf16(_) => NumericalLaw::ResidualBf16V1,
            Self::KeyedGumbel { .. } => NumericalLaw::KeyedGumbelV1,
            Self::CopyRowsInto { .. } | Self::CopyRowsConcatInto { .. } => {
                NumericalLaw::StorageBytesV1
            }
            Self::QkNormRotary { .. } => NumericalLaw::Bf16QkRotaryV1,
            Self::SigmoidGateTail { .. } => NumericalLaw::Bf16SigmoidGateV1,
            Self::AffineDequantize { .. } => NumericalLaw::AffineDequantizeV1,
            Self::AffineQuantize { .. } => NumericalLaw::AffineQuantizeV1,
            Self::AffineLinear { .. } => NumericalLaw::AffineV1,
            Self::Bf16SmallRowLinear => NumericalLaw::MlxSmallRowBf16V1,
            Self::Affine4WideLinear => NumericalLaw::MlxWideAffine4V1,
            Self::SwiGlu | Self::PackedSwiGlu => NumericalLaw::Bf16SwiGluV1,
            Self::GatedRms { .. } => NumericalLaw::Bf16GatedRmsV1,
            Self::DeltaPrep(_) => NumericalLaw::DeltaPrepV1,
            Self::DeltaDecode(_) => NumericalLaw::DeltaDecodeV1,
            Self::DeltaScan { .. } | Self::DeltaReplayCapture | Self::DeltaReplayRestore { .. } => {
                NumericalLaw::DeltaSequentialV1
            }
        }
    }
    pub fn arity(&self) -> usize {
        match self {
            Self::Tensor(op) | Self::HostTensor(op) => op.arity(),
            Self::Refinement(op) => op.arity(),
            Self::OrderedF32(op) => op.arity(),
            Self::PackedCompute(op) => op.arity(),
            Self::SparseF32(op) => op.arity(),
            Self::AffineExperts(op) => op.arity(),
            Self::IndirectAffine(op) => op.arity(),
            Self::IndexedAttention(op) => op.arity(),
            Self::BlockAttention(op) => op.arity(),
            Self::ResidualBf16(op) => op.arity(),
            Self::AffineDequantize { .. } => 3,
            Self::AffineQuantize { .. } => 1,
            Self::AffineLinear { .. } | Self::Affine4WideLinear => 4,
            Self::Bf16SmallRowLinear => 2,
            Self::KeyedGumbel { .. } => 1,
            Self::CopyRowsConcatInto { .. } => 3,
            Self::CopyRowsInto { .. } | Self::SigmoidGateTail { .. } => 2,
            Self::QkNormRotary { .. } => 6,
            Self::SwiGlu => 2,
            Self::PackedSwiGlu => 1,
            Self::DeltaReplayCapture => 6,
            Self::Zeros { .. } => 0,
            Self::DeltaScan { mode, reuse, .. } => {
                6 + 2 * usize::from(mode.gated()) + 2 * usize::from(*reuse)
            }
            Self::GatedRms { .. } => 3,
            Self::DeltaDecode(_) => 7,
            Self::DeltaPrep(s) => {
                if s.separate {
                    7
                } else {
                    5
                }
            }
            Self::DeltaReplayRestore { .. } => 4,
        }
    }
}

/// Checked storage size, also used before constructing shared views.
pub fn tensor_bytes(shape: &[i32], dtype: ScalarType) -> Option<usize> {
    shape.iter().try_fold(dtype.size_of(), |n, &d| {
        usize::try_from(d).ok().and_then(|d| n.checked_mul(d))
    })
}

/// Enqueued output and a fence for its producer. Dropping the fence must not
/// cancel work or release storage still in use. No implicit host synchronization.
pub struct Dispatch<T, C> {
    pub output: T,
    pub completion: C,
}

/// Device storage and individual operations. The engine owns plan traversal,
/// bindings, scheduling, fallbacks and the choice of stream. Handles are opaque,
/// context-bound and prepared once; dispatch never negotiates a capability.
pub trait Device {
    type Tensor;
    type KernelHandle;
    type Stream;
    type Completion;
    type Error: std::error::Error + Send + Sync + 'static;
    fn capabilities(&self) -> &[KernelCapability];
    fn pull(
        &self,
        source: TensorSource<'_>,
        options: ImportOptions,
    ) -> Result<Self::Tensor, Self::Error>;
    fn shape<'a>(&self, tensor: &'a Self::Tensor) -> &'a [i32];
    fn dtype(&self, tensor: &Self::Tensor) -> Option<ScalarType>;
    fn stream(&self) -> Self::Stream;
    /// None means the contract is unsupported. Allocation/compilation failures
    /// are errors, never a silent fallback. Shape-dependent specialization can
    /// remain lazy inside the prepared operation implementation.
    fn prepare(&self, kernel: Kernel) -> Result<Option<Self::KernelHandle>, Self::Error>;
    /// Shape-dependent optional kernels may decline before encoding any work.
    fn dispatch(
        &self,
        handle: &Self::KernelHandle,
        inputs: &[&Self::Tensor],
        stream: &Self::Stream,
    ) -> Result<Option<Dispatch<Outputs<Self::Tensor>, Self::Completion>>, Self::Error>;
    fn wait(&self, completion: &Self::Completion) -> Result<(), Self::Error>;
}

/// Explicit synchronized readback. No implicit cast, sampling or model policy.
pub trait ReadbackDevice: Device {
    fn read_f32(&self, tensor: &Self::Tensor) -> Result<Vec<f32>, Self::Error>;
}

mod access;
pub use access::{HostAccessDevice, TensorAccess, TensorLayout};

mod control;
pub mod erased;
pub use control::{DeviceIdentity, ExecutionScope, ManagedDevice, MemoryUsage};

mod resources;
pub use resources::ResourceDevice;
