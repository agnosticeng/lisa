//! Original packed-block projection and ordered routing arithmetic contracts.
//! Encodings describe payloads, never a checkpoint reader or model family.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PackedEncoding {
    Iq2Xxs,
    Iq2Xs,
    Iq1M,
    Q5K,
    Q6K,
    Q2K,
    Q4K,
    Nvfp4,
}
/// Packed expert weight formats with expert-grouped prefill projections.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExpertFormat {
    Nvfp4,
    Q5K,
    Q8,
}
/// Vendored reference kernel files a [`VendoredDispatch`] names.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum VendoredSource {
    RelativeAttention,
    ShortConvolution,
    BlockResidualMix,
    /// The block residual mixes of more rows than one mix call takes.
    BlockResidualMixRows,
    MessageGate,
    Moe,
    MoeGguf,
    GgufFloat,
    GgufLinear,
    /// The int8 prefill tile and its activation quantizer.
    GgufPrefillInt8,
    LinearColumn,
    LinearRegister,
    Normalization,
    RowCopy,
    Embedding,
    LogitFinish,
    Sampling,
    /// The DFlash2 draft's layers, attention and context commit.
    Draft,
    /// The DFlash2 draft's context prefill.
    DraftContext,
    /// RoPE tables.
    Rope,
    /// LISA's copies of the Inkling step's paged state (snapshots, growth).
    StateGlue,
}
/// A function constant of a [`VendoredDispatch`] pipeline.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum VendoredConstant {
    Bool(bool),
    UInt(u32),
}
/// One dispatch of a vendored reference entry point, as its host code
/// encodes it: input `i` binds buffer slot `slots[i]` at its view's offset,
/// `bytes` bind slot `bytes_slot`, and `addressed` inputs are reached only
/// through GPU addresses the kernel reads (made resident read-write). The
/// grid is `grid` threadgroups of `threads`, or read from `indirect` (input
/// index, byte offset). Kernels write their outputs into the bound inputs;
/// the operation returns input 0 as its output.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct VendoredDispatch {
    pub source: VendoredSource,
    pub entry: String,
    pub constants: Vec<(u32, VendoredConstant)>,
    pub slots: Vec<u32>,
    pub addressed: Vec<u32>,
    pub bytes: Vec<u8>,
    pub bytes_slot: u32,
    pub grid: [u32; 3],
    pub threads: [u32; 3],
    pub indirect: Option<(u32, u32)>,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PackedCompute {
    /// See [`VendoredDispatch`].
    Vendored(Box<VendoredDispatch>),
    /// Page-table entries of KV extents for the vendored reference's paged
    /// attention: input 0 is one extent of `extent_pages` pages; output U64
    /// `[extent_pages]`, entry p = the extent's GPU address | p.
    KvPageTable {
        extent_pages: u32,
    },
    /// GPU addresses of each input's view plus `offsets[i]` bytes: U32
    /// `[2 * inputs]` (little-endian u64 pairs), for tables kernels read
    /// addresses from. The inputs stay resident while kernels use them only
    /// when bound as `addressed` inputs of those dispatches.
    PlaneAddresses {
        offsets: Vec<u64>,
    },
    /// Load-time repack of GGUF expert rows `[first_row, first_row + rows)`
    /// of every expert (input: the stored tensor `[experts, rows', bytes]`,
    /// GGML type `ggml_type`, `columns` inputs per row) into the vendored
    /// reference's column-interleaved planes: one U8 allocation, plane0 then
    /// plane1 then meta. Values are unchanged (a lossless relayout).
    RepackPlanes {
        ggml_type: u32,
        first_row: u32,
        rows: u32,
        columns: u32,
    },
    /// Draft tree selection of the vendored reference for one greedy lane:
    /// the best-first tree over a draft's candidate lattice and a host
    /// lookup chain, then its best `rows` rows in preorder. Inputs I32
    /// candidate IDs `[7, 16]`, F32 logits `[7, 16]`, F32 edges
    /// `[7, 16, 16]`, I32 chain tokens `[15]` and F32 chain rates `[15, 4]`.
    /// With `selected`, the first three inputs are instead the reference
    /// selector's own outputs, already in its order: candidates I32
    /// `[7, 16]`, unary scores F32 `[7, 16]` and its partial values F32
    /// `[7 * 8 * 16 + 7 * 16 * 16]` (shard partials, then the edge tables).
    /// `model` holds F32 bits of the softcap, the acceptance slope, seven
    /// intercepts and the row floor. Outputs: verify IDs I32 `[rows]`, the
    /// tree I32 `[2, rows]` (parents, -1 for a root; depths), the rows'
    /// sources U32 `[rows]`, and the grown tree's tokens U32 `[16]`, rows
    /// U32 `[16, 2]`, scores F32 `[16, 2]`, caps F32 `[16]`, size U32 `[1]` and
    /// the sorted candidates U32 `[7, 16]`, then the verify rows as the
    /// reference's tree rows U32 `[rows, 2]` and its lane rows U32 `[2]`.
    DraftTree {
        anchor: u32,
        depth: u32,
        chain: u32,
        children: u32,
        rows: u32,
        lane_cap: u32,
        model: [u32; 10],
        selected: bool,
    },
    /// Full-copy concatenation; input layouts may be strided, rank at most three.
    ConcatF32 {
        axis: u32,
        inputs: u32,
    },
    GatedResidual4 {
        epsilon: u32,
        masked: bool,
    },
    RmsNorm {
        weighted: bool,
        round_output: bool,
        epsilon: u32,
    },
    RmsNormPair {
        epsilon: u32,
    },
    ResidualInput4 {
        epsilon: u32,
        norm_epsilon: u32,
    },
    ResidualExpand4 {
        add: bool,
        round_block: bool,
    },
    ResidualCollapse4,
    PlainLinear {
        round_output: bool,
    },
    Q8Linear {
        round_output: bool,
        round_input: bool,
    },
    BlockLinear {
        encoding: PackedEncoding,
        round_output: bool,
    },
    GroupedLinear {
        encoding: PackedEncoding,
        round_output: bool,
    },
    /// Original Q8_0 experts with GPU indices and the unrounded dense dot-product law.
    Q8ExpertLinear,
    Q8Pair,
    ExpertLinear {
        encoding: PackedEncoding,
    },
    Router {
        top_k: u32,
        scale: u32,
    },
    /// Flat clamped SiLU product, optionally rounded to BF16 in F32 storage.
    SharedSwiGlu {
        round_output: bool,
    },
    WeightedSwiGlu {
        clamp: u32,
    },
    /// Two independent F32 row blends, with native sigmoid and separate products.
    RowGateBlendPair {
        zero: bool,
    },
    DifferentialSiluGate {
        bias: u32,
    },
    /// Batched F32 products with 2..=8 rows and the original eight-wide MMA order.
    SmallRowMatmul {
        transpose_rhs: bool,
    },
    ReluSquared,
    /// Six separately rounded products followed by five left-associated F32 adds.
    WeightedSumSix,
    /// Six stable routes from 256 finite logits/biases; outputs I32 ids and F32 weights.
    SigmoidRouter256 {
        scale: u32,
    },
    SumSix,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PackedComputeKind {
    Vendored,
    KvPageTable,
    PlaneAddresses,
    RepackPlanes,
    DraftTree,
    ConcatF32,
    GatedResidual4(bool),
    RmsNorm(bool),
    RmsNormPair,
    ResidualInput4,
    ResidualExpand4(bool),
    ResidualCollapse4,
    PlainLinear,
    Q8Linear,
    BlockLinear(PackedEncoding),
    GroupedLinear(PackedEncoding),
    Q8ExpertLinear,
    Q8Pair,
    ExpertLinear(PackedEncoding),
    Router,
    SharedSwiGlu,
    WeightedSwiGlu,
    RowGateBlendPair,
    DifferentialSiluGate,
    SmallRowMatmul,
    ReluSquared,
    WeightedSumSix,
    SigmoidRouter256,
    SumSix,
}
impl PackedCompute {
    pub fn kind(&self) -> PackedComputeKind {
        use PackedComputeKind as K;
        match self {
            Self::RepackPlanes { .. } => K::RepackPlanes,
            Self::DraftTree { .. } => K::DraftTree,
            Self::Vendored(_) => K::Vendored,
            Self::KvPageTable { .. } => K::KvPageTable,
            Self::PlaneAddresses { .. } => K::PlaneAddresses,
            Self::ConcatF32 { .. } => K::ConcatF32,
            Self::GatedResidual4 { masked, .. } => K::GatedResidual4(*masked),
            Self::RmsNorm { weighted, .. } => K::RmsNorm(*weighted),
            Self::RmsNormPair { .. } => K::RmsNormPair,
            Self::ResidualInput4 { .. } => K::ResidualInput4,
            Self::ResidualExpand4 { add, .. } => K::ResidualExpand4(*add),
            Self::ResidualCollapse4 => K::ResidualCollapse4,
            Self::PlainLinear { .. } => K::PlainLinear,
            Self::Q8Linear { .. } => K::Q8Linear,
            Self::BlockLinear { encoding, .. } => K::BlockLinear(*encoding),
            Self::GroupedLinear { encoding, .. } => K::GroupedLinear(*encoding),
            Self::Q8ExpertLinear => K::Q8ExpertLinear,
            Self::Q8Pair => K::Q8Pair,
            Self::ExpertLinear { encoding } => K::ExpertLinear(*encoding),
            Self::Router { .. } => K::Router,
            Self::SharedSwiGlu { .. } => K::SharedSwiGlu,
            Self::WeightedSwiGlu { .. } => K::WeightedSwiGlu,
            Self::RowGateBlendPair { .. } => K::RowGateBlendPair,
            Self::DifferentialSiluGate { .. } => K::DifferentialSiluGate,
            Self::SmallRowMatmul { .. } => K::SmallRowMatmul,
            Self::ReluSquared => K::ReluSquared,
            Self::WeightedSumSix => K::WeightedSumSix,
            Self::SigmoidRouter256 { .. } => K::SigmoidRouter256,
            Self::SumSix => K::SumSix,
        }
    }
    pub fn arity(&self) -> usize {
        match self {
            Self::RepackPlanes { .. } => 1,
            Self::DraftTree { .. } => 5,
            Self::Vendored(d) => d.slots.len(),
            Self::KvPageTable { .. } => 1,
            Self::PlaneAddresses { offsets } => offsets.len(),
            Self::ConcatF32 { inputs, .. } => *inputs as usize,
            Self::GatedResidual4 { masked, .. } => 4 + usize::from(*masked),
            Self::RmsNorm { weighted, .. } => 1 + usize::from(*weighted),
            Self::RmsNormPair { .. } => 4,
            Self::ResidualInput4 { .. } | Self::RowGateBlendPair { .. } => 6,
            Self::ResidualExpand4 { add, .. } => 3 + usize::from(*add),
            Self::SumSix | Self::ReluSquared => 1,
            Self::DifferentialSiluGate { .. }
            | Self::Q8ExpertLinear
            | Self::Q8Pair
            | Self::ExpertLinear { .. }
            | Self::WeightedSwiGlu { .. } => 3,
            _ => 2,
        }
    }
    pub fn outputs(&self) -> usize {
        match self {
            Self::RmsNormPair { .. }
            | Self::Q8Pair
            | Self::SigmoidRouter256 { .. }
            | Self::RowGateBlendPair { .. } => 2,
            Self::ResidualInput4 { .. } => 3,
            Self::Router { .. } => 3,
            _ => 1,
        }
    }
    pub fn validate(&self) -> bool {
        let positive = |bits| {
            let x = f32::from_bits(bits);
            x.is_finite() && x > 0.
        };
        match self {
            Self::RepackPlanes { rows, columns, .. } => *rows > 0 && columns % 32 == 0,
            Self::Vendored(d) => {
                !d.slots.is_empty()
                    && d.addressed.iter().all(|&i| (i as usize) < d.slots.len())
                    && d.indirect.is_none_or(|(i, _)| (i as usize) < d.slots.len())
                    && d.threads.iter().all(|&t| t > 0)
                    && d.threads.iter().product::<u32>() <= 1024
            }
            Self::KvPageTable { extent_pages } => (1..=16384).contains(extent_pages),
            Self::DraftTree {
                depth,
                chain,
                children,
                rows,
                lane_cap,
                model,
                ..
            } => {
                (1..=16).contains(rows)
                    && rows <= lane_cap
                    && *lane_cap <= 16
                    && *chain <= 15
                    && *chain <= *depth
                    && *children <= 4
                    && model.iter().all(|&b| f32::from_bits(b).is_finite())
            }
            Self::ConcatF32 { axis, inputs } => *axis < 3 && (2..=8).contains(inputs),
            Self::GatedResidual4 { epsilon, .. } => positive(*epsilon),
            Self::RmsNorm {
                weighted,
                round_output,
                epsilon,
            } => positive(*epsilon) && (!round_output || *weighted),
            Self::RmsNormPair { epsilon } => positive(*epsilon),
            Self::ResidualInput4 {
                epsilon,
                norm_epsilon,
            } => positive(*epsilon) && positive(*norm_epsilon),
            Self::Q8Linear {
                round_output,
                round_input,
            } => !round_input || *round_output,
            Self::Router { top_k, scale } => {
                (1..=32).contains(top_k)
                    && f32::from_bits(*scale).is_finite()
                    && f32::from_bits(*scale) > 0.
            }
            Self::SigmoidRouter256 { scale } => positive(*scale),
            Self::DifferentialSiluGate { bias } => f32::from_bits(*bias).is_finite(),
            Self::WeightedSwiGlu { clamp } => {
                f32::from_bits(*clamp).is_finite() && f32::from_bits(*clamp) >= 0.
            }
            _ => true,
        }
    }
}
