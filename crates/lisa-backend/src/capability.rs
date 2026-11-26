//! Typed contracts for the portable operation surface. Backends must advertise
//! only contracts they implement; native-only entry points are not capabilities.
use crate::{Kernel, ScalarType};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KernelKind {
    IndexedAttention(crate::IndexedAttentionKind),
    Tensor(crate::TensorOperationKind),
    OrderedF32(crate::OrderedF32Kind),
    PackedCompute(crate::PackedComputeKind),
    SparseF32(crate::SparseF32Kind),
    AffineExperts(crate::AffineExpertsKind),
    IndirectAffine(crate::IndirectAffineKind),
    BlockAttention(crate::BlockAttentionKind),
    ResidualBf16(crate::ResidualBf16Kind),
    AffineLinear,
    AffineSerialLinear,
    Affine4WideLinear,
    AffineDequantize,
    AffineQuantize,
    KeyedGumbel,
    CopyRowsInto,
    CopyRowsConcatInto,
    QkNormRotary,
    SigmoidGateTail,
    SwiGlu,
    PackedSwiGlu,
    Zeros,
    DeviceZeros,
    DeltaDecode,
    DeltaScan { gated: bool, reuse: bool },
    GatedRms,
    DeltaPrepSeparate,
    DeltaPrepPacked,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NumericalLaw {
    IndexedAttentionV1,
    /// The reference scalar projections, original GGML block reductions and ordered routing.
    PackedScalarV1,
    /// Matrix expert reductions and online attention with F32 accumulation.
    /// Internal reduced precision and reassociation are allowed explicitly.
    PackedFastV1,
    /// Original reference GLM F32 sparse arithmetic, separate from six-expert routing.
    SparseF32V1,
    /// Original affine expert reductions, routing selection and sorted combine.
    AffineExpertsV1,
    /// Original reference affine Q2/group-128 indirect projections and ordered slot sums.
    IndirectAffineV1,
    BlockAttentionV1,
    /// Original affine residual projection, stepped SiLU and sequential BF16 stream fold.
    ResidualBf16V1,
    /// Original packed GGML projections and F32 primitive arithmetic.
    OrderedF32V1,
    /// Native tensor primitive law; operation variants preserve distinct cast/order semantics.
    TensorV1,
    KeyedGumbelV1,
    StorageBytesV1,
    Bf16QkRotaryV1,
    Bf16SigmoidGateV1,
    /// Original host f32 operation order, separate from native GPU arithmetic.
    HostSequentialF32,
    /// Existing affine QMV/QMM law, including the explicitly scoped verify
    /// exception. The engine must keep the scope in its session identity.
    AffineV1,
    /// Ordinary single-query affine arithmetic for every independent row,
    /// with only the operation's explicit, entered verify-scope exception.
    AffineSerialV1,
    MlxWideAffine4V1,
    AffineDequantizeV1,
    AffineQuantizeV1,
    /// BF16 intermediates: round SiLU(gate), then multiply and round again.
    Bf16SwiGluV1,
    /// Sequential f32 delta updates with BF16 activation rounding and original reduction order.
    DeltaSequentialV1,
    /// Original BF16 convolution/activation, query-key normalization and gates.
    DeltaPrepV1,
    /// The fused one-token preparation, recurrence and gated RMS law.
    DeltaDecodeV1,
    /// Butterfly RMS and the selected gate's original per-operation casts.
    Bf16GatedRmsV1,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LayoutSupport {
    Strided,
    RowContiguous,
}
#[derive(Clone, Copy, Debug)]
pub struct TensorConstraint {
    pub dtypes: &'static [ScalarType],
    pub min_rank: usize,
    pub max_rank: Option<usize>,
    pub layout: LayoutSupport,
    pub alignment: usize,
}
#[derive(Clone, Copy, Debug)]
pub enum ShapeRelation {
    /// Shape follows the typed tensor operation attributes.
    TensorOperation,
    /// q/k `[B,T,Hk,Dk]`, value/history `[B,T,Hv,Dv]`, state `[B,Hv,Dv,Dk]`.
    DeltaRecurrence,
    /// Last input dimension is the packed matrix input width; scale/bias rows
    /// match output rows and their columns match the group count.
    Affine,
    EqualInputs,
    /// Final dimension is even; output flattens leading dimensions to rows.
    PackedHalves,
}
#[derive(Clone, Copy, Debug)]
pub struct AffineEncoding {
    pub word: ScalarType,
    pub bits: &'static [i32],
    pub group_sizes: &'static [i32],
}
#[derive(Clone, Copy, Debug)]
pub enum Workspace {
    /// Output and internal scratch are allocated by the provider. No persistent
    /// workspace or state is hidden in the kernel handle.
    ProviderManaged,
    /// These input slots are output storage. Their bytes may be overwritten
    /// before dispatch returns; failed later calls do not undo those writes.
    CallerOutputs { inputs: &'static [usize] },
}
#[derive(Clone, Copy, Debug)]
pub struct KernelCapability {
    pub kind: KernelKind,
    pub law: NumericalLaw,
    pub inputs: &'static [TensorConstraint],
    /// Repeat the final input constraint for a variadic operation.
    pub variadic: bool,
    pub shape: ShapeRelation,
    pub encoding: Option<AffineEncoding>,
    pub workspace: Workspace,
}
impl KernelCapability {
    pub fn accepts(&self, kernel: &Kernel) -> bool {
        (if self.variadic {
            kernel.arity() >= self.inputs.len()
        } else {
            kernel.arity() == self.inputs.len()
        }) && self.kind == kernel.kind()
            && self.law == kernel.law()
            && match self.workspace {
                Workspace::ProviderManaged => kernel.overwritten_inputs().is_empty(),
                Workspace::CallerOutputs { inputs } => inputs == kernel.overwritten_inputs(),
            }
            && match kernel {
                Kernel::Tensor(op) | Kernel::HostTensor(op) => {
                    self.encoding.is_none() && op.validate()
                }
                Kernel::AffineQuantize { bits, group_size }
                | Kernel::AffineDequantize { bits, group_size }
                | Kernel::AffineLinear { bits, group_size } => self
                    .encoding
                    .is_some_and(|e| e.bits.contains(bits) && e.group_sizes.contains(group_size)),
                Kernel::AffineSerialLinear {
                    bits, group_size, ..
                } => self
                    .encoding
                    .is_some_and(|e| e.bits.contains(bits) && e.group_sizes.contains(group_size)),
                Kernel::OrderedF32(op) => self.encoding.is_none() && op.validate(),
                Kernel::PackedCompute(op) => self.encoding.is_none() && op.validate(),
                Kernel::SparseF32(op) => self.encoding.is_none() && op.validate(),
                Kernel::AffineExperts(op) => self.encoding.is_none() && op.validate(),
                Kernel::IndirectAffine(op) => self.encoding.is_none() && op.validate(),
                Kernel::IndexedAttention(op) => self.encoding.is_none() && op.validate(),
                Kernel::BlockAttention(op) => self.encoding.is_none() && op.validate(),
                Kernel::ResidualBf16(op) => self.encoding.is_none() && op.validate(),
                Kernel::CopyRowsInto { offset } | Kernel::CopyRowsConcatInto { offset } => {
                    *offset >= 0
                }
                Kernel::QkNormRotary {
                    query_heads,
                    key_heads,
                    rows,
                    epsilon,
                    interleaved,
                } => {
                    *query_heads > 0
                        && *key_heads > 0
                        && query_heads % key_heads == 0
                        && (if *interleaved {
                            (2..=8).contains(rows)
                        } else {
                            *rows == 1
                        })
                        && f32::from_bits(*epsilon).is_finite()
                        && f32::from_bits(*epsilon) >= 0.
                }
                Kernel::SigmoidGateTail {
                    batch,
                    rows,
                    heads,
                    width,
                } => [*batch, *rows, *heads, *width].iter().all(|&d| d > 0),
                Kernel::Zeros { shape, dtype } | Kernel::DeviceZeros { shape, dtype } => {
                    self.encoding.is_none() && crate::tensor_bytes(shape, *dtype).is_some()
                }
                Kernel::DeltaDecode(spec) => self.encoding.is_none() && spec.validate(),
                Kernel::DeltaPrep(spec) => self.encoding.is_none() && spec.validate(),
                Kernel::GatedRms {
                    heads,
                    width,
                    epsilon,
                    ..
                } => {
                    *heads > 0
                        && *width == 128
                        && f32::from_bits(*epsilon).is_finite()
                        && f32::from_bits(*epsilon) >= 0.
                }
                Kernel::DeltaScan {
                    mode,
                    capture,
                    reuse,
                } => match mode {
                    crate::DeltaScan::Verify { epsilon, .. } => {
                        *capture
                            && f32::from_bits(*epsilon).is_finite()
                            && f32::from_bits(*epsilon) >= 0.
                    }
                    crate::DeltaScan::TwoRows => true,
                    _ => !*reuse,
                },
                _ => self.encoding.is_none(),
            }
    }
}
