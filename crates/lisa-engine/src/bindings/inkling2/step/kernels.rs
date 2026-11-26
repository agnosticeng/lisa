//! Parameter blocks and dispatches of the vendored reference kernels one
//! Inkling decode/verify step runs.
//!
//! Every parameter block is written field by field in little-endian order
//! with the layout of the pinned reference's `metal/abi/*.h` structure of the
//! same name less its project prefix (sizes checked against its static
//! assertions in the tests).
//! Every builder returns a [`VendoredDispatch`] whose inputs the caller supplies
//! in the documented order: input `i` binds the slot listed for it, the
//! parameter block follows at the slot after the kernel's last buffer.
//! Nothing here touches a device.
use anyhow::{Result, ensure};
use lisa_backend::{VendoredConstant, VendoredDispatch, VendoredSource};

/// Lane bindings of a dispatch (`MAXIMUM_BATCH_WIDTH`).
pub(crate) const MAXIMUM_LANES: u32 = 8;
/// Rows of a verify block (`TARGET_VERIFY_ROWS`).
pub(crate) const VERIFY_ROWS: u32 = 16;
/// Tokens of a KV page (`TARGET_KV_BLOCK_TOKENS`).
pub(crate) const PAGE_TOKENS: u32 = 32;
/// Key/value heads and their dimension of the relative attention.
pub(crate) const KV_HEADS: u32 = 8;
pub(crate) const HEAD_DIMENSION: u32 = 128;
/// Query heads of the relative attention (three a KV head).
pub(crate) const QUERY_HEADS: u32 = 24;
/// INT8 data and F32 scale bytes of one KV tensor's page slab.
pub(crate) const KV_PAGE_DATA_BYTES: u32 = KV_HEADS * PAGE_TOKENS * HEAD_DIMENSION;
pub(crate) const KV_PAGE_SCALE_BYTES: u32 = KV_HEADS * PAGE_TOKENS * 4;
/// Rows of an attention tile (`ATTENTION_TILE_ROWS`).
pub(crate) const ATTENTION_TILE_ROWS: u32 = 8;
/// History splits a decode dispatch takes at most
/// (`RELATIVE_ATTENTION_DECODE_SPLITS`).
pub(crate) const DECODE_SPLITS: u32 = 32;
/// Pages a lane's tile takes a split for, times the GPU cores
/// (`RelativeAttention::kDecodeCorePages`).
pub(crate) const DECODE_CORE_PAGES: u64 = 80;
/// Window of the sliding layers (keys before a row).
pub(crate) const WINDOW_TOKENS: u32 = 1023;
/// Argmax shards of a target row (`TARGET_SAMPLING_SHARDS`).
pub(crate) const SAMPLING_SHARDS: u32 = 16;
/// Stop tokens a selection compares (`STOP_TOKENS`).
pub(crate) const STOP_TOKENS: usize = 4;
/// No token (`NO_TOKEN`).
pub(crate) const NO_TOKEN: u32 = u32::MAX;
/// A tree root's parent (`TREE_ROOT`).
pub(crate) const TREE_ROOT: u16 = 0xffff;
/// `GGUF_FMT_Q80`, Q8_0's run-time format id, and its kernel name token.
pub(crate) const Q80: u32 = 6;
pub(crate) const Q80_TOKEN: &str = "q80";
/// Threads of a short convolution's threadgroup (`kConvolutionThreads`).
const CONVOLUTION_THREADS: u32 = 256;
/// Channels and rows of a combine-convolution threadgroup
/// (`MOE_COMBINE_CONVOLUTION_CHANNELS`, `_ROWS`).
const COMBINE_CHANNELS: u32 = 32;
const COMBINE_ROWS: u32 = 32;
/// Expert slots of a routing (`MOE_EXPERT_SLOTS`): the select and
/// grouping threadgroups and the router's score stride.
pub(crate) const EXPERT_SLOTS: u32 = 256;
/// Byte offsets of the expert passes' grids in `MoeTileCount` (80 bytes).
pub(crate) const TILE_COUNT_BYTES: u32 = 80;
pub(crate) const GATE_UP_GRID: u32 = 8;
pub(crate) const DOWN_GRID: u32 = 20;
/// `MoeTileDescriptor` bytes.
pub(crate) const TILE_DESCRIPTOR_BYTES: u32 = 12;
/// Threads of the staged and expert tiles (`GGUF_STAGED_THREADS`), the
/// register tile (`GGUF_REGISTER_THREADS`) and the column tile.
const STAGED_THREADS: u32 = 64;
const REGISTER_THREADS: u32 = 128;
const COLUMN_THREADS: u32 = 256;
/// Columns of a GGUF tile and of a column-tile threadgroup.
pub(crate) const TILE_COLUMNS: u32 = 64;
const COLUMN_COLUMNS: u32 = 32;
/// Rows of a Table16 tile (`GGUF_TABLE16_ROWS`).
pub(crate) const TABLE16_ROWS: u32 = 8;
/// Fused projection segments (`GGUF_SEGMENTS`).
const SEGMENTS: usize = 3;
/// Rows of a mix call that runs a threadgroup a row and query
/// (`BLOCK_RESIDUAL_MIX_QUERY_ROWS`), its threads, and its limits.
pub(crate) const MIX_QUERY_ROWS: u32 = 64;
const MIX_THREADS: u32 = 1024;
const MIX_ROW_THREADS: u32 = 256;
const MIX_MAXIMUM_QUERIES: u32 = 4;
const MIX_MAXIMUM_STATES: u32 = 5;
/// Message gate geometry (`MESSAGE_GATE_*`).
const GATE_THREADS: u32 = 256;
const GATE_GROUP_ROWS: u32 = 8;
const GATE_DOWN_OUTPUTS: u32 = 2;
/// Rows of a row-kernel gate tile (`MESSAGE_GATE_TILE_ROWS`).
const GATE_TILE_ROWS: u32 = 32;
pub(crate) const GATE_MAXIMUM_RANK: u32 = 64;
pub(crate) const GATE_MAXIMUM_GATES: u32 = 3;
/// Rows a fused gate call takes; more run the vendored file's row kernel.
pub(crate) const GATE_FUSED_ROWS: u32 = 128;
const DEFAULT_THREADS: u32 = 256;

// ---------------------------------------------------------------------------
// Parameter blocks

/// A parameter structure of the reference ABI, written little-endian.
pub(crate) trait Abi {
    /// `sizeof` of the reference structure.
    const SIZE: usize;
    fn put(&self, out: &mut Vec<u8>);
    fn bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(Self::SIZE);
        self.put(&mut out);
        debug_assert_eq!(out.len(), Self::SIZE);
        out
    }
}
fn put_u32(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_le_bytes());
}
fn put_f32(out: &mut Vec<u8>, value: f32) {
    out.extend_from_slice(&value.to_le_bytes());
}
fn put_u32s(out: &mut Vec<u8>, values: &[u32]) {
    for &v in values {
        put_u32(out, v);
    }
}
fn put_f32s(out: &mut Vec<u8>, values: &[f32]) {
    for &v in values {
        put_f32(out, v);
    }
}

/// `BlockResidualMixParams` (24 bytes): `rows` rows mixed over `stack`
/// stack states `stack_plane` values apart and the current rows, by
/// `queries` queries, scores scaled by `scale` (hidden^-1/2).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct BlockResidualMixParams {
    pub rows: u32,
    pub hidden: u32,
    pub stack: u32,
    pub queries: u32,
    pub stack_plane: u32,
    pub scale: f32,
}
impl Abi for BlockResidualMixParams {
    const SIZE: usize = 24;
    fn put(&self, out: &mut Vec<u8>) {
        put_u32s(
            out,
            &[
                self.rows,
                self.hidden,
                self.stack,
                self.queries,
                self.stack_plane,
            ],
        );
        put_f32(out, self.scale);
    }
}

/// `MessageGateFusedParams` (48 bytes).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct MessageGateFusedParams {
    pub rows: u32,
    pub hidden: u32,
    pub x_stride: u32,
    pub message_stride: u32,
    pub rank: u32,
    pub gates: u32,
    pub x_columns: [u32; 3],
    pub message_columns: [u32; 3],
}
impl Abi for MessageGateFusedParams {
    const SIZE: usize = 48;
    fn put(&self, out: &mut Vec<u8>) {
        put_u32s(
            out,
            &[
                self.rows,
                self.hidden,
                self.x_stride,
                self.message_stride,
                self.rank,
                self.gates,
            ],
        );
        put_u32s(out, &self.x_columns);
        put_u32s(out, &self.message_columns);
    }
}

/// Outputs of a short convolution (`SHORT_CONVOLUTION_OUTPUT_*`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(u32)]
pub(crate) enum ConvolutionOutput {
    /// bf16(y) as bf16 rows.
    #[default]
    Plain = 0,
    /// fp32 rows of a bf16 residual (or none) plus bf16(y).
    Sum = 1,
    /// bf16 of an fp32 residual plus bf16(y).
    RoundedSum = 2,
}

/// `ShortConvolutionParams` (104 bytes).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct ShortConvolutionParams {
    pub rows: u32,
    pub channels: u32,
    pub heads: u32,
    pub output: ConvolutionOutput,
    pub residual: u32,
    pub write_state: u32,
    pub tree: u32,
    pub sequences: u32,
    pub state_stride: u32,
    pub state_column: u32,
    pub divisor: f32,
    pub captures: u32,
    pub captures_only: u32,
    pub dynamic_stride: u32,
    pub capture_sequences: [u32; 4],
    pub capture_rows: [u32; 4],
    pub input_stride: u32,
    pub input_column: u32,
    pub pair_input_column: u32,
    pub pair_state_column: u32,
}
impl Abi for ShortConvolutionParams {
    const SIZE: usize = 104;
    fn put(&self, out: &mut Vec<u8>) {
        put_u32s(
            out,
            &[
                self.rows,
                self.channels,
                self.heads,
                self.output as u32,
                self.residual,
                self.write_state,
                self.tree,
                self.sequences,
                self.state_stride,
                self.state_column,
            ],
        );
        put_f32(out, self.divisor);
        put_u32s(
            out,
            &[self.captures, self.captures_only, self.dynamic_stride],
        );
        put_u32s(out, &self.capture_sequences);
        put_u32s(out, &self.capture_rows);
        put_u32s(
            out,
            &[
                self.input_stride,
                self.input_column,
                self.pair_input_column,
                self.pair_state_column,
            ],
        );
    }
}

/// `MoeCombineParams` (12 bytes).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct MoeCombineParams {
    pub rows: u32,
    pub hidden_size: u32,
    pub routes_per_row: u32,
}
impl Abi for MoeCombineParams {
    const SIZE: usize = 12;
    fn put(&self, out: &mut Vec<u8>) {
        put_u32s(out, &[self.rows, self.hidden_size, self.routes_per_row]);
    }
}

/// `MoeCombineConvolutionParams` (116 bytes): the combine, then the
/// convolution at byte 12.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct MoeCombineConvolutionParams {
    pub combine: MoeCombineParams,
    pub convolution: ShortConvolutionParams,
}
impl Abi for MoeCombineConvolutionParams {
    const SIZE: usize = 116;
    fn put(&self, out: &mut Vec<u8>) {
        self.combine.put(out);
        self.convolution.put(out);
    }
}

/// `KvLayer` (8 bytes): a layer's region in the KV extents.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct KvLayer {
    pub extent_pages: u32,
    pub offset: u32,
}
impl Abi for KvLayer {
    const SIZE: usize = 8;
    fn put(&self, out: &mut Vec<u8>) {
        put_u32s(out, &[self.extent_pages, self.offset]);
    }
}

/// `AttentionSequence` (44 bytes).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct AttentionSequence {
    pub committed_tokens: u32,
    pub row_bound: u32,
    pub layout_row: u32,
    pub stage_row: u32,
    pub chunk_stride: u32,
    pub page_table_entries: u32,
    pub split_count: u32,
    pub first_tile: u32,
    pub first_slot: u32,
    pub block_stride: u32,
    pub block_list: u32,
}
impl Abi for AttentionSequence {
    const SIZE: usize = 44;
    fn put(&self, out: &mut Vec<u8>) {
        put_u32s(
            out,
            &[
                self.committed_tokens,
                self.row_bound,
                self.layout_row,
                self.stage_row,
                self.chunk_stride,
                self.page_table_entries,
                self.split_count,
                self.first_tile,
                self.first_slot,
                self.block_stride,
                self.block_list,
            ],
        );
    }
}

/// `AttentionParams` (384 bytes): six words, the layer's KV placement
/// at byte 24, then eight sequences from byte 32.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct AttentionParams {
    pub sequences: u32,
    pub first_sequence: u32,
    pub rows: u32,
    pub tiles: u32,
    pub trees: u32,
    pub last_rows: u32,
    pub kv: KvLayer,
    pub sequence: [AttentionSequence; MAXIMUM_LANES as usize],
}
impl Abi for AttentionParams {
    const SIZE: usize = 32 + 44 * MAXIMUM_LANES as usize;
    fn put(&self, out: &mut Vec<u8>) {
        put_u32s(
            out,
            &[
                self.sequences,
                self.first_sequence,
                self.rows,
                self.tiles,
                self.trees,
                self.last_rows,
            ],
        );
        self.kv.put(out);
        for s in &self.sequence {
            s.put(out);
        }
    }
}

/// `KvPathCommitParams` (40 bytes).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct KvPathCommitParams {
    pub extent_pages: u32,
    pub regions: u32,
    pub committed_tokens: [u32; MAXIMUM_LANES as usize],
}
impl Abi for KvPathCommitParams {
    const SIZE: usize = 8 + 4 * MAXIMUM_LANES as usize;
    fn put(&self, out: &mut Vec<u8>) {
        put_u32s(out, &[self.extent_pages, self.regions]);
        put_u32s(out, &self.committed_tokens);
    }
}

/// `MoeRouteParams` (16 bytes).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct MoeRouteParams {
    pub rows: u32,
    pub input_size: u32,
    pub experts: u32,
    pub top_k: u32,
}
impl Abi for MoeRouteParams {
    const SIZE: usize = 16;
    fn put(&self, out: &mut Vec<u8>) {
        put_u32s(out, &[self.rows, self.input_size, self.experts, self.top_k]);
    }
}

/// `MoeBiasedRouteParams` (20 bytes): the routing, then its weights' scale.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct MoeBiasedRouteParams {
    pub route: MoeRouteParams,
    pub scale: f32,
}
impl Abi for MoeBiasedRouteParams {
    const SIZE: usize = 20;
    fn put(&self, out: &mut Vec<u8>) {
        self.route.put(out);
        put_f32(out, self.scale);
    }
}

/// `MoeGroupParams` (32 bytes).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct MoeGroupParams {
    pub rows: u32,
    pub top_k: u32,
    pub tile_rows: u32,
    pub experts: u32,
    /// Intermediate / `GGUF_STAGED_COLUMNS` (32).
    pub gate_up_columns: u32,
    /// Hidden / `GGUF_TILE_COLUMNS` (64).
    pub down_columns: u32,
    pub streamed: u32,
    pub lanes: u32,
}
impl Abi for MoeGroupParams {
    const SIZE: usize = 32;
    fn put(&self, out: &mut Vec<u8>) {
        put_u32s(
            out,
            &[
                self.rows,
                self.top_k,
                self.tile_rows,
                self.experts,
                self.gate_up_columns,
                self.down_columns,
                self.streamed,
                self.lanes,
            ],
        );
    }
}

/// `MoeGatherParams` (8 bytes).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct MoeGatherParams {
    pub input_size: u32,
    pub routes_per_row: u32,
}
impl Abi for MoeGatherParams {
    const SIZE: usize = 8;
    fn put(&self, out: &mut Vec<u8>) {
        put_u32s(out, &[self.input_size, self.routes_per_row]);
    }
}

/// Expert tiles a pass reads (`MOE_TILES_*`).
pub(crate) const MOE_TILES_ALL: u32 = 0;

/// `MoeGgufGateUpParams` (40 bytes).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct MoeGgufGateUpParams {
    pub input_size: u32,
    pub output_size: u32,
    pub experts: u32,
    pub gate_routed_format: u32,
    pub gate_shared_format: u32,
    pub up_routed_format: u32,
    pub up_shared_format: u32,
    pub floor_slots: u32,
    pub chunk_slots: u32,
    pub tiles: u32,
}
impl Abi for MoeGgufGateUpParams {
    const SIZE: usize = 40;
    fn put(&self, out: &mut Vec<u8>) {
        put_u32s(
            out,
            &[
                self.input_size,
                self.output_size,
                self.experts,
                self.gate_routed_format,
                self.gate_shared_format,
                self.up_routed_format,
                self.up_shared_format,
                self.floor_slots,
                self.chunk_slots,
                self.tiles,
            ],
        );
    }
}

/// `MoeGgufExpertParams` (32 bytes), the down pass's.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct MoeGgufExpertParams {
    pub input_size: u32,
    pub output_size: u32,
    pub experts: u32,
    pub routed_format: u32,
    pub shared_format: u32,
    pub floor_slots: u32,
    pub chunk_slots: u32,
    pub tiles: u32,
}
impl Abi for MoeGgufExpertParams {
    const SIZE: usize = 32;
    fn put(&self, out: &mut Vec<u8>) {
        put_u32s(
            out,
            &[
                self.input_size,
                self.output_size,
                self.experts,
                self.routed_format,
                self.shared_format,
                self.floor_slots,
                self.chunk_slots,
                self.tiles,
            ],
        );
    }
}

/// `GgufFloatParams` (32 bytes): up to two projections of one input.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct GgufFloatParams {
    pub rows: u32,
    pub input_size: u32,
    pub output_size: [u32; 2],
    pub out_stride: [u32; 2],
    pub out_offset: [u32; 2],
}
impl Abi for GgufFloatParams {
    const SIZE: usize = 32;
    fn put(&self, out: &mut Vec<u8>) {
        put_u32s(out, &[self.rows, self.input_size]);
        put_u32s(out, &self.output_size);
        put_u32s(out, &self.out_stride);
        put_u32s(out, &self.out_offset);
    }
}

/// `GgufEmbedParams` (12 bytes).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct GgufEmbedParams {
    pub rows: u32,
    pub vocabulary: u32,
    pub hidden: u32,
}
impl Abi for GgufEmbedParams {
    const SIZE: usize = 12;
    fn put(&self, out: &mut Vec<u8>) {
        put_u32s(out, &[self.rows, self.vocabulary, self.hidden]);
    }
}

/// `RowCopyParams` (32 bytes).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct RowCopyParams {
    pub rows: u32,
    pub width: u32,
    pub source_row: u32,
    pub source_stride: u32,
    pub source_column: u32,
    pub destination_row: u32,
    pub destination_stride: u32,
    pub destination_column: u32,
}
impl Abi for RowCopyParams {
    const SIZE: usize = 32;
    fn put(&self, out: &mut Vec<u8>) {
        put_u32s(
            out,
            &[
                self.rows,
                self.width,
                self.source_row,
                self.source_stride,
                self.source_column,
                self.destination_row,
                self.destination_stride,
                self.destination_column,
            ],
        );
    }
}

/// `LogitFinishParams` (16 bytes).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct LogitFinishParams {
    pub rows: u32,
    pub vocabulary: u32,
    pub tokens: u32,
    pub softcap: f32,
}
impl Abi for LogitFinishParams {
    const SIZE: usize = 16;
    fn put(&self, out: &mut Vec<u8>) {
        put_u32s(out, &[self.rows, self.vocabulary, self.tokens]);
        put_f32(out, self.softcap);
    }
}

/// `GgufColumnParams` (20 bytes).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct GgufColumnParams {
    pub rows: u32,
    pub input_size: u32,
    pub out_stride: u32,
    pub out_offset: u32,
    pub format: u32,
}
impl Abi for GgufColumnParams {
    const SIZE: usize = 20;
    fn put(&self, out: &mut Vec<u8>) {
        put_u32s(
            out,
            &[
                self.rows,
                self.input_size,
                self.out_stride,
                self.out_offset,
                self.format,
            ],
        );
    }
}

/// A fused projection's segment: its columns, format and first output
/// column. Slots past the last take no columns, format 0 and offset 0, as
/// the reference's host leaves them.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Segment {
    pub columns: u32,
    pub format: u32,
    pub offset: u32,
}
fn segment_fields(segments: &[Segment]) -> Result<([u32; 3], [u32; 3], [u32; 3])> {
    ensure!(
        (1..=SEGMENTS).contains(&segments.len()),
        "a fused projection takes one to three segments"
    );
    let (mut cols, mut fmt, mut offset) = ([0; 3], [0; 3], [0; 3]);
    for (i, s) in segments.iter().enumerate() {
        cols[i] = s.columns;
        fmt[i] = s.format;
        offset[i] = s.offset;
    }
    Ok((cols, fmt, offset))
}

/// `GgufColumnFusedParams` (48 bytes).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct GgufColumnFusedParams {
    pub rows: u32,
    pub input_size: u32,
    pub out_stride: u32,
    pub cols: [u32; 3],
    pub fmt: [u32; 3],
    pub offset: [u32; 3],
}
impl GgufColumnFusedParams {
    pub(crate) fn new(
        rows: u32,
        input_size: u32,
        out_stride: u32,
        segments: &[Segment],
    ) -> Result<Self> {
        let (cols, fmt, offset) = segment_fields(segments)?;
        Ok(Self {
            rows,
            input_size,
            out_stride,
            cols,
            fmt,
            offset,
        })
    }
}
impl Abi for GgufColumnFusedParams {
    const SIZE: usize = 48;
    fn put(&self, out: &mut Vec<u8>) {
        put_u32s(out, &[self.rows, self.input_size, self.out_stride]);
        put_u32s(out, &self.cols);
        put_u32s(out, &self.fmt);
        put_u32s(out, &self.offset);
    }
}

/// `GgufDecodeParams` (20 bytes).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct GgufDecodeParams {
    pub input_size: u32,
    pub splits: u32,
    pub out_stride: u32,
    pub out_offset: u32,
    pub spread: u32,
}
impl Abi for GgufDecodeParams {
    const SIZE: usize = 20;
    fn put(&self, out: &mut Vec<u8>) {
        put_u32s(
            out,
            &[
                self.input_size,
                self.splits,
                self.out_stride,
                self.out_offset,
                self.spread,
            ],
        );
    }
}

/// `GgufDecodeFusedParams` (52 bytes).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct GgufDecodeFusedParams {
    pub input_size: u32,
    pub splits: u32,
    pub out_stride: u32,
    pub cols: [u32; 3],
    pub fmt: [u32; 3],
    pub offset: [u32; 3],
    pub spread: u32,
}
impl GgufDecodeFusedParams {
    pub(crate) fn new(
        input_size: u32,
        splits: u32,
        out_stride: u32,
        segments: &[Segment],
        spread: u32,
    ) -> Result<Self> {
        let (cols, fmt, offset) = segment_fields(segments)?;
        Ok(Self {
            input_size,
            splits,
            out_stride,
            cols,
            fmt,
            offset,
            spread,
        })
    }
    fn segments(&self) -> usize {
        self.cols.iter().take_while(|&&c| c > 0).count()
    }
}
impl Abi for GgufDecodeFusedParams {
    const SIZE: usize = 52;
    fn put(&self, out: &mut Vec<u8>) {
        put_u32s(out, &[self.input_size, self.splits, self.out_stride]);
        put_u32s(out, &self.cols);
        put_u32s(out, &self.fmt);
        put_u32s(out, &self.offset);
        put_u32(out, self.spread);
    }
}

/// `TargetSamplingParams` (172 bytes). A greedy step's lanes leave the
/// sampling fields zero.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct TargetSamplingParams {
    pub vocabulary: u32,
    pub mask_words: u32,
    pub mask_row: u32,
    pub top_k: [u32; MAXIMUM_LANES as usize],
    pub temperature: [f32; MAXIMUM_LANES as usize],
    pub top_p: [f32; MAXIMUM_LANES as usize],
    pub min_p: [f32; MAXIMUM_LANES as usize],
    pub sampling_mask: u32,
    pub constrained_mask: u32,
    pub exclude_stop_mask: u32,
    pub stop_tokens: [u32; STOP_TOKENS],
    pub trees: u32,
}
impl TargetSamplingParams {
    /// Every lane greedy over verify trees: `vocabulary` logits a row
    /// (`mask_words` its constraint masks' words, unread by unconstrained
    /// lanes), the stop tokens padded with [`NO_TOKEN`].
    pub(crate) fn greedy(vocabulary: u32, stop: &[u32]) -> Result<Self> {
        ensure!(stop.len() <= STOP_TOKENS, "at most four stop tokens");
        let mut stop_tokens = [NO_TOKEN; STOP_TOKENS];
        stop_tokens[..stop.len()].copy_from_slice(stop);
        Ok(Self {
            vocabulary,
            mask_words: vocabulary.div_ceil(32),
            stop_tokens,
            trees: 1,
            ..Default::default()
        })
    }
}
impl Abi for TargetSamplingParams {
    const SIZE: usize = 44 + 16 * MAXIMUM_LANES as usize;
    fn put(&self, out: &mut Vec<u8>) {
        put_u32s(out, &[self.vocabulary, self.mask_words, self.mask_row]);
        put_u32s(out, &self.top_k);
        put_f32s(out, &self.temperature);
        put_f32s(out, &self.top_p);
        put_f32s(out, &self.min_p);
        put_u32s(
            out,
            &[
                self.sampling_mask,
                self.constrained_mask,
                self.exclude_stop_mask,
            ],
        );
        put_u32s(out, &self.stop_tokens);
        put_u32(out, self.trees);
    }
}

/// `AcceptBatchParams` (48 bytes): each lane's most retained tokens, and the
/// stop tokens padded with [`NO_TOKEN`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct AcceptBatchParams {
    pub remaining: [u32; MAXIMUM_LANES as usize],
    pub stop_tokens: [u32; STOP_TOKENS],
}
impl Default for AcceptBatchParams {
    fn default() -> Self {
        Self {
            remaining: [0; MAXIMUM_LANES as usize],
            stop_tokens: [NO_TOKEN; STOP_TOKENS],
        }
    }
}
impl Abi for AcceptBatchParams {
    const SIZE: usize = 16 + 4 * MAXIMUM_LANES as usize;
    fn put(&self, out: &mut Vec<u8>) {
        put_u32s(out, &self.remaining);
        put_u32s(out, &self.stop_tokens);
    }
}

/// `TreeRow` (8 bytes): a verify row's ancestors (itself included,
/// rows of its lane's block), its parent ([`TREE_ROOT`] for the anchor) and
/// depth.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct TreeRow {
    pub ancestors: u32,
    pub parent: u16,
    pub depth: u16,
}
impl TreeRow {
    /// A chain of `rows` rows as a tree: row r's parent r - 1, depth r.
    pub(crate) fn chain(rows: u32) -> Result<Vec<Self>> {
        ensure!(
            (1..=32).contains(&rows),
            "a verify block holds 1 to 32 rows"
        );
        Ok((0..rows)
            .map(|r| Self {
                ancestors: if r == 31 {
                    u32::MAX
                } else {
                    (1u32 << (r + 1)) - 1
                },
                parent: if r == 0 { TREE_ROOT } else { (r - 1) as u16 },
                depth: r as u16,
            })
            .collect())
    }
}
impl Abi for TreeRow {
    const SIZE: usize = 8;
    fn put(&self, out: &mut Vec<u8>) {
        put_u32(out, self.ancestors);
        out.extend_from_slice(&self.parent.to_le_bytes());
        out.extend_from_slice(&self.depth.to_le_bytes());
    }
}

/// `LaneRows` (8 bytes): a lane's first packed row and its rows.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct LaneRows {
    pub first: u32,
    pub rows: u32,
}
impl Abi for LaneRows {
    const SIZE: usize = 8;
    fn put(&self, out: &mut Vec<u8>) {
        put_u32s(out, &[self.first, self.rows]);
    }
}

/// The bytes of consecutive ABI records (tree rows, lane rows).
pub(crate) fn pack<T: Abi>(records: &[T]) -> Vec<u8> {
    let mut out = Vec::with_capacity(records.len() * T::SIZE);
    for r in records {
        r.put(&mut out);
    }
    out
}

/// A `constant uint &` parameter (the norms' and the Table16 prepare's width).
pub(crate) fn width_bytes(width: u32) -> Vec<u8> {
    width.to_le_bytes().to_vec()
}

// ---------------------------------------------------------------------------
// Dispatch assembly

/// A dispatch binding its `inputs` inputs at slots `0..inputs` and its
/// parameters at slot `inputs`.
fn contiguous(
    source: VendoredSource,
    entry: impl Into<String>,
    inputs: u32,
    bytes: Vec<u8>,
    grid: [u32; 3],
    threads: [u32; 3],
) -> VendoredDispatch {
    VendoredDispatch {
        source,
        entry: entry.into(),
        constants: Vec::new(),
        slots: (0..inputs).collect(),
        addressed: Vec::new(),
        bytes,
        bytes_slot: inputs,
        grid,
        threads,
        indirect: None,
    }
}

fn uint_constants(values: &[u32]) -> Vec<(u32, VendoredConstant)> {
    values
        .iter()
        .enumerate()
        .map(|(i, &v)| (i as u32, VendoredConstant::UInt(v)))
        .collect()
}

/// The formats a specialized pipeline decodes (the reference's `passFormats`
/// and fused-format constants): each distinct id once in ascending order,
/// the last repeated to `N`.
pub(crate) fn pass_formats<const N: usize>(formats: &[u32]) -> Result<[u32; N]> {
    ensure!(
        !formats.is_empty() && formats.len() <= N,
        "one to {N} formats"
    );
    let mut distinct = formats.to_vec();
    distinct.sort_unstable();
    distinct.dedup();
    let last = *distinct.last().unwrap_or(&0);
    let mut out = [last; N];
    out[..distinct.len()].copy_from_slice(&distinct);
    Ok(out)
}

fn lanes_valid(lanes: u32) -> Result<()> {
    ensure!(
        (1..=MAXIMUM_LANES).contains(&lanes),
        "a step takes 1 to 8 lanes"
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// Embedding, norms, copies

/// `gguf_embed_<token>`: rows of a GGUF token table into bf16 rows.
/// Inputs: 0 token ids U32 `[rows]`, 1 table rows as stored, 2 output bf16
/// `[rows][hidden]`; parameters at 3.
pub(crate) fn gguf_embed(token: &str, p: &GgufEmbedParams) -> VendoredDispatch {
    let elements = p.rows * p.hidden;
    contiguous(
        VendoredSource::Embedding,
        format!("gguf_embed_{token}"),
        3,
        p.bytes(),
        [elements.div_ceil(DEFAULT_THREADS), 1, 1],
        [DEFAULT_THREADS, 1, 1],
    )
}

/// `norm_rms_f32`: a weighted RMS norm of bf16 rows with F32 weights, a
/// threadgroup a row. Inputs: 0 input bf16 `[rows][width]`, 1 weight F32
/// `[width]`, 2 output bf16; the width at 3.
pub(crate) fn norm_rms(rows: u32, width: u32) -> VendoredDispatch {
    contiguous(
        VendoredSource::Normalization,
        "norm_rms_f32",
        3,
        width_bytes(width),
        [rows, 1, 1],
        [DEFAULT_THREADS, 1, 1],
    )
}

/// `copy_rows_bf16`. Inputs: 0 source, 1 destination; parameters at 2.
pub(crate) fn copy_rows_bf16(p: &RowCopyParams) -> VendoredDispatch {
    contiguous(
        VendoredSource::RowCopy,
        "copy_rows_bf16",
        2,
        p.bytes(),
        [(p.rows * p.width).div_ceil(DEFAULT_THREADS), 1, 1],
        [DEFAULT_THREADS, 1, 1],
    )
}

/// `logit_finish`, in place. Input: 0 logits F32 `[rows][vocabulary]`;
/// parameters at 1.
pub(crate) fn logit_finish(p: &LogitFinishParams) -> VendoredDispatch {
    contiguous(
        VendoredSource::LogitFinish,
        "logit_finish",
        1,
        p.bytes(),
        [p.vocabulary.div_ceil(DEFAULT_THREADS), p.rows, 1],
        [DEFAULT_THREADS, 1, 1],
    )
}

// ---------------------------------------------------------------------------
// Block residual mixes and message gates

/// The current rows' type of a mix: bf16 (a layer's input) or fp32 (the
/// residual inside a layer, `_f32`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum MixCurrent {
    Bf16,
    F32,
}
impl MixCurrent {
    fn suffix(self) -> &'static str {
        match self {
            Self::Bf16 => "",
            Self::F32 => "_f32",
        }
    }
}

fn mix_valid(p: &BlockResidualMixParams, queries: u32) -> Result<()> {
    ensure!(p.rows > 0, "a mix of no rows");
    ensure!(
        (1..=queries).contains(&p.queries) && p.stack < MIX_MAXIMUM_STATES,
        "invalid mix queries or states"
    );
    Ok(())
}

/// A mix of `rows` rows: up to [`MIX_QUERY_ROWS`] a threadgroup a row and
/// query of 1024 threads, more the row kernels (`_rows`, of
/// `block_residual_mix_rows.metal`) a threadgroup a row of 256.
fn mix_launch(
    name: &str,
    p: &BlockResidualMixParams,
    current: MixCurrent,
    inputs: u32,
) -> VendoredDispatch {
    if p.rows <= MIX_QUERY_ROWS {
        contiguous(
            VendoredSource::BlockResidualMix,
            format!("{name}{}", current.suffix()),
            inputs,
            p.bytes(),
            [p.rows, p.queries, 1],
            [MIX_THREADS, 1, 1],
        )
    } else {
        contiguous(
            VendoredSource::BlockResidualMixRows,
            format!("{name}_rows{}", current.suffix()),
            inputs,
            p.bytes(),
            [p.rows, 1, 1],
            [MIX_ROW_THREADS, 1, 1],
        )
    }
}

/// `block_residual_mix[_rows][_f32]`: each query's output as fp32 rows
/// `[rows][queries][hidden]`. Inputs: 0 stack bf16 (the current rows when
/// `stack` is 0), 1 current rows, 2 queries F32 `[queries][hidden]`, 3 output
/// F32; parameters at 4.
pub(crate) fn block_residual_mix(
    p: &BlockResidualMixParams,
    current: MixCurrent,
) -> Result<VendoredDispatch> {
    mix_valid(p, MIX_MAXIMUM_QUERIES)?;
    Ok(mix_launch("block_residual_mix", p, current, 4))
}

/// `block_residual_mix_norm[_rows][_f32]`: each query's output through its
/// own weighted RMS norm into its own bf16 rows. Inputs: 0 stack, 1
/// current, 2 queries F32, 3-5 norms F32 (unused slots repeat input 3), 6-8
/// outputs bf16 (unused slots repeat input 6); parameters at 9.
pub(crate) fn block_residual_mix_norm(
    p: &BlockResidualMixParams,
    current: MixCurrent,
) -> Result<VendoredDispatch> {
    mix_valid(p, 3)?;
    Ok(mix_launch("block_residual_mix_norm", p, current, 9))
}

/// The stored type of a message gate's projections, one per call.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum GateWeights {
    F32,
    Bf16,
}
impl GateWeights {
    fn suffix(self) -> &'static str {
        match self {
            Self::F32 => "_f32",
            Self::Bf16 => "_bf16",
        }
    }
    fn bytes(self) -> u64 {
        match self {
            Self::F32 => 4,
            Self::Bf16 => 2,
        }
    }
}

fn gate_valid(p: &MessageGateFusedParams) -> Result<()> {
    ensure!(
        (1..=GATE_FUSED_ROWS).contains(&p.rows),
        "a gate call of more than 128 rows runs the row kernel"
    );
    gate_shape_valid(p)
}

fn gate_shape_valid(p: &MessageGateFusedParams) -> Result<()> {
    ensure!(
        p.rows > 0
            && (1..=GATE_MAXIMUM_GATES).contains(&p.gates)
            && p.rank > 0
            && p.rank <= GATE_MAXIMUM_RANK
            && p.rank.is_multiple_of(4)
            && p.hidden.is_multiple_of(4)
            && p.x_stride.is_multiple_of(4)
            && p.message_stride.is_multiple_of(4)
            && p.x_columns
                .iter()
                .chain(&p.message_columns)
                .all(|c| c.is_multiple_of(4)),
        "message gates the fused kernels do not take"
    );
    Ok(())
}

/// Scratch bytes of a fused gate call: rank values bf16 `[gates][rows][rank]`,
/// the concatenations then gated rows fp32 `[gates][rows][hidden]`, partial
/// sums `[gates][groups][slices][8]` fp32, and counters `[gates][groups]`
/// (zero at allocation, zero after every call).
pub(crate) fn message_gate_scratch(p: &MessageGateFusedParams) -> [u64; 4] {
    let (gates, rows) = (u64::from(p.gates), u64::from(p.rows));
    let groups = u64::from(p.rows.div_ceil(GATE_GROUP_ROWS));
    let slices = u64::from(p.hidden.div_ceil(GATE_THREADS));
    [
        gates * rows * u64::from(p.rank) * 2,
        gates * rows * u64::from(p.hidden) * 4,
        gates * groups * slices * u64::from(GATE_GROUP_ROWS) * 4,
        gates * groups * 4,
    ]
}

/// `message_gate_normalize`. Inputs: 0 x F32, 1 messages F32, 2 concat bf16
/// `[gates][rows][2 hidden]` (the gated rows' buffer); parameters at 3.
/// Grid (rows, 1, gates).
pub(crate) fn message_gate_normalize(p: &MessageGateFusedParams) -> Result<VendoredDispatch> {
    gate_valid(p)?;
    Ok(contiguous(
        VendoredSource::MessageGate,
        "message_gate_normalize",
        3,
        p.bytes(),
        [p.rows, 1, p.gates],
        [GATE_THREADS, 1, 1],
    ))
}

/// `message_gate_down_<f32|bf16>`. Inputs: 0 concat, 1-3 down weights
/// `[rank][2 hidden]` of gates 0-2 (unused slots repeat gate 0's), 4 rank
/// values bf16 `[gates][rows][rank]`; parameters at 5.
pub(crate) fn message_gate_down(
    p: &MessageGateFusedParams,
    weights: GateWeights,
) -> Result<VendoredDispatch> {
    gate_valid(p)?;
    Ok(contiguous(
        VendoredSource::MessageGate,
        format!("message_gate_down{}", weights.suffix()),
        5,
        p.bytes(),
        [
            p.rank.div_ceil(GATE_DOWN_OUTPUTS),
            p.rows.div_ceil(GATE_GROUP_ROWS),
            p.gates,
        ],
        [GATE_THREADS, 1, 1],
    ))
}

/// `message_gate_apply_<f32|bf16>`. Inputs: 0 x, 1 messages, 2-4 forget
/// `[hidden][rank]`, 5-7 gain `[hidden][rank]`, 8 rank values, 9-11 norms
/// F32, 12 gated values F32 (the concat's buffer), 13 partial sums, 14
/// counters, 15-17 outputs bf16 (unused slots repeat gate 0's buffers);
/// parameters at 18.
pub(crate) fn message_gate_apply(
    p: &MessageGateFusedParams,
    weights: GateWeights,
) -> Result<VendoredDispatch> {
    gate_valid(p)?;
    Ok(contiguous(
        VendoredSource::MessageGate,
        format!("message_gate_apply{}", weights.suffix()),
        18,
        p.bytes(),
        [
            p.hidden.div_ceil(GATE_THREADS),
            p.rows.div_ceil(GATE_GROUP_ROWS),
            p.gates,
        ],
        [GATE_THREADS, 1, 1],
    ))
}

/// `message_gate_rows_<f32|bf16>`: a gate call of more than
/// [`GATE_FUSED_ROWS`] rows, tiles of 32 rows. Inputs: 0 x F32, 1 messages
/// F32, 2-4 down weights, 5-7 forget, 8-10 gain, 11-13 norms F32, 14-16 each
/// gate's gated rows F32 `[rows][hidden]`, 17-19 outputs bf16 (unused slots
/// repeat gate 0's buffers); parameters at 20. Grid (row tiles, 1, gates).
pub(crate) fn message_gate_rows(
    p: &MessageGateFusedParams,
    weights: GateWeights,
) -> Result<VendoredDispatch> {
    gate_shape_valid(p)?;
    Ok(contiguous(
        VendoredSource::MessageGate,
        format!("message_gate_rows{}", weights.suffix()),
        20,
        p.bytes(),
        [p.rows.div_ceil(GATE_TILE_ROWS), 1, p.gates],
        [GATE_THREADS, 1, 1],
    ))
}

/// Bytes of one gate's down, forget and gain weights.
pub(crate) fn message_gate_weight_bytes(hidden: u32, rank: u32, weights: GateWeights) -> [u64; 3] {
    let (h, r, e) = (u64::from(hidden), u64::from(rank), weights.bytes());
    [r * 2 * h * e, h * r * e, h * r * e]
}

// ---------------------------------------------------------------------------
// Short convolutions

fn convolution_valid(p: &ShortConvolutionParams) -> Result<()> {
    lanes_valid(p.sequences)?;
    ensure!(
        p.channels > 0 && p.state_stride > 0,
        "invalid short convolution"
    );
    Ok(())
}

/// `short_convolution_pair`: a verify step's K and V convolutions, plain,
/// in tree mode. Inputs: 0 keys input, 1 values input, 2-9 lane states in
/// (unused lanes repeat lane 0's), 10/11 static taps F32 `[channels][4]`,
/// 12/13 dynamic taps bf16 (the static taps again without), 14/15 outputs
/// bf16, 16 lane layout, 17 tree rows, 18 tape; parameters at 19. Grid
/// (lane cap, both convolutions' channel groups, lanes).
pub(crate) fn short_convolution_pair(
    p: &ShortConvolutionParams,
    lane_cap: u32,
) -> Result<VendoredDispatch> {
    convolution_valid(p)?;
    ensure!(
        p.output == ConvolutionOutput::Plain && p.tree == 1,
        "a pair is plain, in tree mode"
    );
    Ok(contiguous(
        VendoredSource::ShortConvolution,
        "short_convolution_pair",
        19,
        p.bytes(),
        [
            lane_cap,
            2 * p.channels.div_ceil(CONVOLUTION_THREADS),
            p.sequences,
        ],
        [CONVOLUTION_THREADS, 1, 1],
    ))
}

/// `short_convolution_sum` (fp32 output of a bf16 residual) or
/// `short_convolution` (bf16 output of an fp32 residual). Inputs: 0 input
/// bf16, 1-8 lane states in, 9-16 lane states out (states in again without
/// `write_state`), 17 static taps F32, 18 dynamic taps (the static taps
/// without), 19 residual, 20 output, 21 lane layout, 22 tree rows, 23 tape,
/// 24-27 captures (any written buffer without); parameters at 28. Grid
/// (`grid_rows`: a decode step's lane cap, a prompt's bound plus state rows,
/// channel groups, lanes).
pub(crate) fn short_convolution(
    p: &ShortConvolutionParams,
    grid_rows: u32,
) -> Result<VendoredDispatch> {
    convolution_valid(p)?;
    let entry = match p.output {
        ConvolutionOutput::Sum => "short_convolution_sum",
        ConvolutionOutput::Plain | ConvolutionOutput::RoundedSum => "short_convolution",
    };
    Ok(contiguous(
        VendoredSource::ShortConvolution,
        entry,
        28,
        p.bytes(),
        [
            grid_rows,
            p.channels.div_ceil(CONVOLUTION_THREADS),
            p.sequences,
        ],
        [CONVOLUTION_THREADS, 1, 1],
    ))
}

/// `short_convolution_commit`: each lane's state after its accepted path,
/// every column of its state matrix (`channels` = the state stride).
/// Inputs: 0 tape, 1-8 lane states in (current), 9-16 lane states out
/// (next), 17 retained rows U32 `[lanes]`, 18 lane layout; parameters at 19.
/// Grid (3 state rows, channel groups, lanes).
pub(crate) fn short_convolution_commit(p: &ShortConvolutionParams) -> Result<VendoredDispatch> {
    convolution_valid(p)?;
    Ok(contiguous(
        VendoredSource::ShortConvolution,
        "short_convolution_commit",
        19,
        p.bytes(),
        [3, p.channels.div_ceil(CONVOLUTION_THREADS), p.sequences],
        [CONVOLUTION_THREADS, 1, 1],
    ))
}

// ---------------------------------------------------------------------------
// Relative attention

/// The decode pipeline of a layer and step.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct AttentionVariant {
    /// INT8 queries (`q8q`) over the INT8 pages, else bf16 queries (`q8`).
    pub int8_queries: bool,
    /// A sliding layer's window of 1023 keys (`_w1023`), else the whole
    /// history, log-scaled (`_scaled`).
    pub window: bool,
    /// Also write the out-projection's Table16 input (`_table16`).
    pub table16: bool,
}

/// Rows of the decode pipeline for a lane cap: 8 or 16 (`_r16`).
pub(crate) fn attention_rows_per_tile(lane_cap: u32) -> u32 {
    if lane_cap > ATTENTION_TILE_ROWS {
        16
    } else {
        8
    }
}

/// `relative_attention_{q8|q8q}_decode[_r16]_{w1023|scaled}[_table16]`.
pub(crate) fn relative_attention_decode_entry(variant: AttentionVariant, lane_cap: u32) -> String {
    format!(
        "relative_attention_{}_decode{}{}{}",
        if variant.int8_queries { "q8q" } else { "q8" },
        if lane_cap > ATTENTION_TILE_ROWS {
            "_r16"
        } else {
            ""
        },
        if variant.window { "_w1023" } else { "_scaled" },
        if variant.table16 { "_table16" } else { "" },
    )
}

/// The relative attention's single decode pass. Inputs: 0 projection rows
/// bf16 (stride 6592), 1 convolved keys bf16, 2 convolved values bf16, 3
/// query scale F32 `[24][128]`, 4 relative-state scale F32 `[24][16]`, 5
/// relative bias F32 `[1024][16]`, 6 partials F32, 7 statistics F32, 8
/// counters U32 `[8 lanes][8 heads]`, 9 hidden bf16 `[rows][3072]`, 10
/// Table16 table (the hidden rows without), 11 Table16 sums (the hidden rows
/// without), 12-19 lane page tables U64 (unused lanes repeat lane 0's), 20
/// tree rows, 21 lane layout; parameters at 22. Grid (KV heads, lanes, most
/// splits) of 256 threads.
pub(crate) fn relative_attention_decode(
    p: &AttentionParams,
    variant: AttentionVariant,
    lane_cap: u32,
) -> Result<VendoredDispatch> {
    lanes_valid(p.sequences)?;
    let lanes = &p.sequence[..p.sequences as usize];
    let rows_per_tile = attention_rows_per_tile(lane_cap);
    ensure!(
        lanes.iter().all(|s| {
            (1..=DECODE_SPLITS).contains(&s.split_count)
                && (1..=rows_per_tile).contains(&s.row_bound)
        }),
        "attention lanes past the decode pipeline's splits or rows"
    );
    let splits = lanes.iter().map(|s| s.split_count).max().unwrap_or(1);
    Ok(contiguous(
        VendoredSource::RelativeAttention,
        relative_attention_decode_entry(variant, lane_cap),
        22,
        p.bytes(),
        [KV_HEADS, p.sequences, splits],
        [256, 1, 1],
    ))
}

/// `relative_attention_q8_commit`: moves each lane's accepted path rows'
/// keys and values to their committed slots. Inputs: 0-7 lane page tables
/// (unused lanes repeat lane 0's), 8 retained rows U32 `[lanes]`;
/// parameters at 9. Grid (regions x KV heads, lanes) of a thread a
/// dimension.
pub(crate) fn relative_attention_q8_commit(
    p: &KvPathCommitParams,
    lanes: u32,
) -> Result<VendoredDispatch> {
    lanes_valid(lanes)?;
    Ok(contiguous(
        VendoredSource::RelativeAttention,
        "relative_attention_q8_commit",
        9,
        p.bytes(),
        [p.regions * KV_HEADS, lanes, 1],
        [HEAD_DIMENSION, 1, 1],
    ))
}

/// Pages of a KV region's tensor of `extent_pages` pages, per tensor kind.
fn region_bytes(extent_pages: u32) -> u64 {
    u64::from(extent_pages) * 2 * u64::from(KV_PAGE_DATA_BYTES + KV_PAGE_SCALE_BYTES)
}

/// Region `region`'s placement in extents of `extent_pages` pages
/// (`kv_layer` of the vendored ABI).
pub(crate) fn kv_layer(extent_pages: u32, region: u32) -> Result<KvLayer> {
    let offset = u64::from(region) * region_bytes(extent_pages);
    ensure!(
        offset <= u64::from(u32::MAX),
        "KV region offset past the kernel ABI"
    );
    Ok(KvLayer {
        extent_pages,
        offset: offset as u32,
    })
}

/// Bytes of one extent of `regions` regions.
pub(crate) fn kv_extent_bytes(extent_pages: u32, regions: u32) -> u64 {
    u64::from(regions) * region_bytes(extent_pages)
}

/// Pages a lane that sees `visible` tokens reads, at most a window's.
pub(crate) fn read_pages(visible: u64, window_tokens: u32) -> u64 {
    let seen = visible.div_ceil(u64::from(PAGE_TOKENS));
    if window_tokens == 0 {
        seen
    } else {
        seen.min(u64::from(window_attention_pages(window_tokens)))
    }
}

/// Pages a window of `window_tokens` keys before a tile's rows spans.
pub(crate) fn window_attention_pages(window_tokens: u32) -> u32 {
    (window_tokens + ATTENTION_TILE_ROWS - 1).div_ceil(PAGE_TOKENS) + 1
}

/// Page-table entries a lane's step needs.
pub(crate) fn page_table_entries(committed: u32, lane_cap: u32) -> u32 {
    (committed + lane_cap).div_ceil(PAGE_TOKENS)
}

/// The most history splits of a lane's decode tile.
pub(crate) fn decode_most_splits(lane_cap: u32, cores: u32) -> u32 {
    let most = if lane_cap > ATTENTION_TILE_ROWS {
        (cores + 8) / 4
    } else {
        (cores + 16) / 3
    };
    most.clamp(1, DECODE_SPLITS)
}

/// The history splits of a lane's decode tile over `pages` pages on a GPU
/// of `cores` cores (`RelativeAttention::decodeSplits`), every split reading
/// at least one page.
pub(crate) fn decode_splits(pages: u64, cores: u32, lane_cap: u32) -> u32 {
    let most = u64::from(decode_most_splits(lane_cap, cores));
    let fewest = pages.clamp(1, u64::from(decode_most_splits(VERIFY_ROWS, cores)));
    let splits = (pages * u64::from(cores))
        .div_ceil(DECODE_CORE_PAGES)
        .clamp(fewest.min(most), most);
    let per = pages.div_ceil(splits).max(1);
    pages.div_ceil(per).max(1) as u32
}

/// Pages the most loaded split reads (INT8 queries from a calibrated count).
pub(crate) fn split_reads(pages: u64, splits: u32) -> u64 {
    pages.div_ceil(u64::from(splits.max(1)))
}

/// One lane's slots in the attention workspace, from its `first_slot` (in
/// 24-row tile slots) at its pipeline's rows a tile.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct AttentionSlots {
    pub first_slot: u32,
    pub splits: u32,
    pub rows_per_tile: u32,
}
impl AttentionSlots {
    fn tiles(&self) -> u64 {
        u64::from(self.rows_per_tile / ATTENTION_TILE_ROWS)
    }
    /// Fused rows of the pipeline's tile: three query heads a row.
    pub(crate) fn fused_rows(&self) -> u64 {
        3 * u64::from(self.rows_per_tile)
    }
    fn head_slot(&self, kv_head: u32) -> u64 {
        u64::from(self.first_slot) / self.tiles() + u64::from(kv_head) * u64::from(self.splits)
    }
    /// F32 offset of a split's partials of a KV head.
    pub(crate) fn partials(&self, kv_head: u32, split: u32) -> u64 {
        (self.head_slot(kv_head) + u64::from(split)) * self.fused_rows() * u64::from(HEAD_DIMENSION)
    }
    /// F32 offset of a fused row's `{max, sum}` statistics.
    pub(crate) fn statistics(&self, kv_head: u32, split: u32, fused: u32) -> u64 {
        ((self.head_slot(kv_head) + u64::from(split)) * self.fused_rows() + u64::from(fused)) * 2
    }
    /// F32 offset of a split's staging: INT8 query codes, then F32 relative
    /// states at byte `M * 128`, then bf16 queries at byte `2 M * 128`.
    pub(crate) fn staging(&self, kv_head: u32, split: u32) -> u64 {
        (u64::from(self.first_slot) / self.tiles()
            + u64::from(KV_HEADS + kv_head) * u64::from(self.splits)
            + u64::from(split))
            * self.fused_rows()
            * u64::from(HEAD_DIMENSION)
    }
}

/// Counter index of a lane's KV head.
pub(crate) fn attention_counter(first_sequence: u32, lane: u32, kv_head: u32) -> u32 {
    (first_sequence + lane) * KV_HEADS + kv_head
}

/// Each lane's first slot: the lanes before it each take a slot of partials
/// and one of staging per tile, KV head and split.
pub(crate) fn attention_first_slots(splits: &[u32], lane_cap: u32) -> Vec<u32> {
    let tiles = lane_cap.div_ceil(ATTENTION_TILE_ROWS);
    let mut slot = 0;
    splits
        .iter()
        .map(|&s| {
            let first = slot;
            slot += 2 * tiles * KV_HEADS * s;
            first
        })
        .collect()
}

/// Partials and statistics bytes of a step's lanes (one dispatch).
pub(crate) fn attention_workspace_bytes(splits: &[u32], lane_cap: u32) -> (u64, u64) {
    let tiles = u64::from(lane_cap.div_ceil(ATTENTION_TILE_ROWS));
    let slots: u64 = splits.iter().map(|&s| 2 * tiles * u64::from(s)).sum();
    let rows = slots * u64::from(ATTENTION_TILE_ROWS * QUERY_HEADS);
    (rows * u64::from(HEAD_DIMENSION) * 4, rows * 2 * 4)
}

/// Bytes of the attention counters (8 lanes x 8 KV heads), zero at
/// allocation and after every dispatch.
pub(crate) const ATTENTION_COUNTER_BYTES: u64 = (MAXIMUM_LANES * KV_HEADS * 4) as u64;

/// A lane's attention sequence of a decode step: every row of its block
/// from its anchor's position `committed`.
pub(crate) fn attention_sequence(
    lane: u32,
    committed: u32,
    lane_cap: u32,
    splits: u32,
    first_slot: u32,
) -> AttentionSequence {
    AttentionSequence {
        committed_tokens: committed,
        row_bound: lane_cap,
        layout_row: 0,
        stage_row: lane * KV_HEADS * PAGE_TOKENS,
        chunk_stride: PAGE_TOKENS,
        page_table_entries: page_table_entries(committed, lane_cap),
        split_count: splits,
        first_tile: lane,
        first_slot,
        block_stride: 0,
        block_list: 0,
    }
}

// ---------------------------------------------------------------------------
// MoE

/// Expert tile rows the vendored passes are compiled for.
pub(crate) const EXPERT_TILE_ROWS: [u32; 3] = [8, 32, 64];

/// The most expert tiles of `rows` rows routed to `top_k` of `experts`
/// experts, without shared slices (`moeMaximumTiles`).
pub(crate) fn moe_maximum_tiles(rows: u32, top_k: u32, experts: u32, tile_rows: u32) -> u32 {
    let routed = rows * top_k;
    (routed / tile_rows + experts).min(routed)
}

/// `gguf_float_<weights>_<output>`: fp32 projections of F32 or BF16 weights
/// on the float tile. Inputs: 0 input bf16, 1 weights of projection 0, 2 its
/// output, 3 weights of projection 1, 4 its output (projection 0's again
/// without one); parameters at 5. Grid (output columns / 8 summed, rows / 32).
pub(crate) fn gguf_float(
    weights_f32: bool,
    output_f32: bool,
    p: &GgufFloatParams,
) -> VendoredDispatch {
    let tiles = p.output_size.iter().map(|n| n.div_ceil(8)).sum::<u32>();
    contiguous(
        VendoredSource::GgufFloat,
        format!(
            "gguf_float_{}_{}",
            if weights_f32 { "f32" } else { "bf16" },
            if output_f32 { "f32" } else { "bf16" }
        ),
        5,
        p.bytes(),
        [tiles, p.rows.div_ceil(32), 1],
        [256, 1, 1],
    )
}

/// `moe_route_select_f32_sigmoid_biased_s0`. Inputs: 0 router scores F32
/// (rows of 256), 1 selection bias F32 `[experts]`, 2 selected U32
/// `[rows][top_k]`, 3 routing weights F32; parameters at 4.
pub(crate) fn moe_route_select(p: &MoeBiasedRouteParams) -> Result<VendoredDispatch> {
    ensure!(p.route.experts <= EXPERT_SLOTS, "more experts than slots");
    Ok(contiguous(
        VendoredSource::Moe,
        "moe_route_select_f32_sigmoid_biased_s0",
        4,
        p.bytes(),
        [p.route.rows, 1, 1],
        [EXPERT_SLOTS, 1, 1],
    ))
}

/// `moe_group_routes_verified_s0`: the step's routes of its lanes' rows in
/// expert tiles, writing the expert passes' indirect grids. Inputs: 0
/// selected, 1 tile descriptors, 2 tile count (`MoeTileCount`), 3 grouped
/// routes, 4 route rows, 5 slot table, 6 expert slots, 7 lane layout;
/// parameters at 8. One threadgroup.
pub(crate) fn moe_group_routes_verified(p: &MoeGroupParams) -> Result<VendoredDispatch> {
    lanes_valid(p.lanes)?;
    ensure!(
        p.experts <= EXPERT_SLOTS && EXPERT_TILE_ROWS.contains(&p.tile_rows),
        "invalid expert grouping"
    );
    Ok(contiguous(
        VendoredSource::Moe,
        "moe_group_routes_verified_s0",
        8,
        p.bytes(),
        [1, 1, 1],
        [EXPERT_SLOTS, 1, 1],
    ))
}

/// `moe_gather_rows`: each grouped route's input row. Inputs: 0 input bf16,
/// 1 grouped routes, 2 tile descriptors, 3 tile count, 4 grouped input;
/// parameters at 5. Grid (most tiles, input / 256).
pub(crate) fn moe_gather_rows(p: &MoeGatherParams, maximum_tiles: u32) -> Result<VendoredDispatch> {
    ensure!(
        p.input_size.is_multiple_of(256),
        "gathered rows of whole 256 columns"
    );
    Ok(contiguous(
        VendoredSource::Moe,
        "moe_gather_rows",
        5,
        p.bytes(),
        [maximum_tiles, p.input_size / 256, 1],
        [256, 1, 1],
    ))
}

/// An expert pass: inputs `0..params_slot` at their own slots, the chunk
/// table at the slot after the parameters, then `planes` plane storages
/// reached through the chunk table's addresses at the slots after it; the
/// grid read from the tile count (input 2) at `grid_offset`.
fn expert_pass(
    entry: String,
    params: Vec<u8>,
    params_slot: u32,
    constants: Vec<(u32, VendoredConstant)>,
    planes: u32,
    nominal: [u32; 3],
    grid_offset: u32,
) -> VendoredDispatch {
    let mut slots: Vec<u32> = (0..params_slot).collect();
    slots.push(params_slot + 1);
    let fixed = slots.len() as u32;
    slots.extend((0..planes).map(|i| params_slot + 2 + i));
    VendoredDispatch {
        source: VendoredSource::MoeGguf,
        entry,
        constants,
        slots,
        addressed: (fixed..fixed + planes).collect(),
        bytes: params,
        bytes_slot: params_slot,
        grid: nominal,
        threads: [STAGED_THREADS, 1, 1],
        indirect: Some((2, grid_offset)),
    }
}

/// `moe_expert_gguf_m<tile>_gate_up[_mixed]_s0_softcap4_25`, on the grid the
/// grouping wrote. Inputs: 0 grouped input bf16, 1 tile descriptors, 2 tile
/// count, 3-8 shared planes (the slot table: no shared expert), 9
/// intermediate bf16, 10 slot table, 11 expert scales F32 `[3][experts]`, 12
/// chunk table U64 (slot 13), then `planes` plane storages (slots 14 on,
/// resident through their addresses); parameters at 12. Function constants
/// 0-3: the gate's and up's formats (`pass_formats`).
pub(crate) fn moe_expert_gate_up(
    p: &MoeGgufGateUpParams,
    tile_rows: u32,
    maximum_tiles: u32,
    planes: u32,
) -> Result<VendoredDispatch> {
    ensure!(
        EXPERT_TILE_ROWS.contains(&tile_rows),
        "no expert pass of {tile_rows} rows"
    );
    let mixed =
        p.gate_routed_format != p.up_routed_format || p.gate_shared_format != p.up_shared_format;
    let formats = pass_formats::<4>(&[
        p.gate_routed_format,
        p.gate_shared_format,
        p.up_routed_format,
        p.up_shared_format,
    ])?;
    Ok(expert_pass(
        format!(
            "moe_expert_gguf_m{tile_rows}_gate_up{}_s0_softcap4_25",
            if mixed { "_mixed" } else { "" }
        ),
        p.bytes(),
        12,
        uint_constants(&formats),
        planes,
        [p.output_size / 32, maximum_tiles, 1],
        GATE_UP_GRID,
    ))
}

/// `moe_expert_gguf_m<tile>_down_s0`, on the grid the grouping wrote.
/// Inputs: 0 intermediate bf16, 1 tile descriptors, 2 tile count, 3-5
/// shared planes (the slot table), 6 expert outputs bf16, 7 slot table, 8
/// expert scales, 9 chunk table U64 (slot 10), then `planes` plane storages
/// (slots 11 on); parameters at 9. Function constants 0-1: the down formats.
pub(crate) fn moe_expert_down(
    p: &MoeGgufExpertParams,
    tile_rows: u32,
    maximum_tiles: u32,
    planes: u32,
) -> Result<VendoredDispatch> {
    ensure!(
        EXPERT_TILE_ROWS.contains(&tile_rows),
        "no expert pass of {tile_rows} rows"
    );
    let formats = pass_formats::<2>(&[p.routed_format, p.shared_format])?;
    Ok(expert_pass(
        format!("moe_expert_gguf_m{tile_rows}_down_s0"),
        p.bytes(),
        9,
        uint_constants(&formats),
        planes,
        [p.output_size / TILE_COLUMNS, maximum_tiles, 1],
        DOWN_GRID,
    ))
}

/// `moe_combine_convolution`: the routes' weighted sum, then the FFN output
/// convolution along each row's path into the layer's output rows. Inputs: 0
/// expert outputs bf16, 1 route rows, 2 routing weights, 3 combine residual
/// (zero rows, never written), 4-11 lane states in, 12 taps F32
/// `[hidden][4]`, 13 residual F32, 14 output bf16, 15 lane layout, 16 tree
/// rows, 17 tape; parameters at 18. Grid (hidden / 32, lanes) of (32, lane
/// cap) threads.
pub(crate) fn moe_combine_convolution(
    p: &MoeCombineConvolutionParams,
    lane_cap: u32,
) -> Result<VendoredDispatch> {
    let c = &p.convolution;
    convolution_valid(c)?;
    ensure!(
        c.channels.is_multiple_of(COMBINE_CHANNELS)
            && (1..=COMBINE_ROWS).contains(&lane_cap)
            && c.tree == 1
            && c.heads == 0,
        "invalid combine convolution"
    );
    Ok(contiguous(
        VendoredSource::Moe,
        "moe_combine_convolution",
        18,
        p.bytes(),
        [c.channels / COMBINE_CHANNELS, c.sequences, 1],
        [COMBINE_CHANNELS, lane_cap, 1],
    ))
}

// ---------------------------------------------------------------------------
// Dense GGUF projections

/// The output type of a plain projection: bf16 (`_a`) or fp32 (`_a_f32`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Destination {
    Bf16,
    F32,
}
impl Destination {
    fn suffix(self) -> &'static str {
        match self {
            Self::Bf16 => "a",
            Self::F32 => "a_f32",
        }
    }
}

/// Table16 scratch bytes (table, sums) of `tiles` 8-row tiles of `width`
/// inputs.
pub(crate) fn table16_bytes(width: u32, tiles: u32) -> (u64, u64) {
    let (w, t) = (u64::from(width), u64::from(tiles));
    let sums_per_tile = w / 16 * 8 + w / 32 * 8;
    (w * u64::from(TABLE16_ROWS) * t * 2, t * sums_per_tile * 4)
}

/// `decode_linear_gguf_prepare`: the register tile's Table16 input of
/// `tiles` 8-row tiles. Inputs: 0 input bf16 (whole tiles of readable rows),
/// 1 table bf16, 2 sums F32; the width at 3.
pub(crate) fn decode_linear_gguf_prepare(width: u32, tiles: u32) -> Result<VendoredDispatch> {
    ensure!(
        width.is_multiple_of(TILE_COLUMNS),
        "Table16 rows of whole 64-input spans"
    );
    Ok(contiguous(
        VendoredSource::LinearRegister,
        "decode_linear_gguf_prepare",
        3,
        width_bytes(width),
        [width / 32, tiles, 1],
        [REGISTER_THREADS, 1, 1],
    ))
}

/// `gguf_column_a[_f32]`: the column tile of 1-2 rows over `columns`
/// columns. Inputs: 0 input bf16, 1 plane0, 2 plane1 (meta without), 3
/// meta, 4 output; parameters at 5.
pub(crate) fn gguf_column(
    p: &GgufColumnParams,
    destination: Destination,
    columns: u32,
) -> Result<VendoredDispatch> {
    ensure!(
        (1..=2).contains(&p.rows) && columns.is_multiple_of(COLUMN_COLUMNS),
        "invalid column tile"
    );
    Ok(contiguous(
        VendoredSource::LinearColumn,
        format!("gguf_column_{}", destination.suffix()),
        5,
        p.bytes(),
        [columns / COLUMN_COLUMNS, 1, 1],
        [COLUMN_THREADS, 1, 1],
    ))
}

/// `gguf_column_fused`. Inputs: 0 input bf16, 1-9 the segments' plane
/// triples (slots past the last repeat its planes), 10 output bf16;
/// parameters at 11.
pub(crate) fn gguf_column_fused(p: &GgufColumnFusedParams) -> Result<VendoredDispatch> {
    let columns: u32 = p.cols.iter().sum();
    ensure!(
        (1..=2).contains(&p.rows) && p.cols.iter().all(|c| c.is_multiple_of(COLUMN_COLUMNS)),
        "invalid fused column tile"
    );
    Ok(contiguous(
        VendoredSource::LinearColumn,
        "gguf_column_fused",
        11,
        p.bytes(),
        [columns / COLUMN_COLUMNS, 1, 1],
        [COLUMN_THREADS, 1, 1],
    ))
}

fn decode_valid(input_size: u32, splits: u32, columns: &[u32]) -> Result<()> {
    ensure!(
        splits >= 1 && columns.iter().all(|c| c.is_multiple_of(TILE_COLUMNS)),
        "invalid decode tile"
    );
    ensure!(
        input_size.is_multiple_of(TILE_COLUMNS),
        "inputs of whole 64-input spans"
    );
    Ok(())
}

/// `gguf_decode_sg_<token>_l<lanes>_<a|a_f32>`: the register tile. Inputs: 0
/// table, 1 sums, 2 plane0, 3 plane1 (meta without), 4 meta, 5 output, 6
/// partials, 7 counters (the output in both with one split), 8 aux (the
/// output); parameters at 9. Grid (columns / 64, splits) of 128 threads.
pub(crate) fn gguf_decode_sg(
    token: &str,
    lanes: u32,
    destination: Destination,
    p: &GgufDecodeParams,
    columns: u32,
) -> Result<VendoredDispatch> {
    decode_valid(p.input_size, p.splits, &[columns])?;
    ensure!(
        (1..=4).contains(&lanes) && p.input_size / 64 >= p.splits,
        "invalid register tile"
    );
    Ok(contiguous(
        VendoredSource::LinearRegister,
        format!("gguf_decode_sg_{token}_l{lanes}_{}", destination.suffix()),
        9,
        p.bytes(),
        [columns / TILE_COLUMNS, p.splits, 1],
        [REGISTER_THREADS, 1, 1],
    ))
}

/// `gguf_decode_sg_fused_l<lanes>`. Inputs: 0 table, 1 sums, 2-10 the
/// segments' plane triples, 11 output bf16, 12 partials, 13 counters;
/// parameters at 14.
pub(crate) fn gguf_decode_sg_fused(
    lanes: u32,
    p: &GgufDecodeFusedParams,
) -> Result<VendoredDispatch> {
    decode_valid(p.input_size, p.splits, &p.cols)?;
    ensure!(
        (1..=4).contains(&lanes) && p.input_size / 64 >= p.splits,
        "invalid register tile"
    );
    Ok(contiguous(
        VendoredSource::LinearRegister,
        format!("gguf_decode_sg_fused_l{lanes}"),
        14,
        p.bytes(),
        [p.cols.iter().sum::<u32>() / TILE_COLUMNS, p.splits, 1],
        [REGISTER_THREADS, 1, 1],
    ))
}

/// Staged tile heights the vendored file instantiates.
pub(crate) const STAGED_ROWS: [u32; 5] = [8, 16, 32, 48, 64];

/// The staged tile height of `rows` rows.
pub(crate) fn staged_rows(rows: u32) -> u32 {
    STAGED_ROWS.into_iter().find(|&r| rows <= r).unwrap_or(64)
}

fn staged_valid(height: u32, input_size: u32, splits: u32, columns: &[u32]) -> Result<()> {
    decode_valid(input_size, splits, columns)?;
    ensure!(
        STAGED_ROWS.contains(&height),
        "no staged tile of {height} rows"
    );
    ensure!(
        (input_size / 32).is_multiple_of(splits),
        "staged K partitions of whole 32-input steps"
    );
    Ok(())
}

/// `gguf_decode_<token>_m<height>_<a|a_f32>`: the staged tile. Inputs: 0
/// input bf16 (`height` readable rows), 1 plane0, 2 plane1 (meta without),
/// 3 meta, 4 output (`height` rows), 5 partials, 6 counters (the output in
/// both with one split), 7 aux (the output); parameters at 8. Grid
/// (columns / 64, splits) of 64 threads.
pub(crate) fn gguf_decode(
    token: &str,
    height: u32,
    destination: Destination,
    p: &GgufDecodeParams,
    columns: u32,
) -> Result<VendoredDispatch> {
    staged_valid(height, p.input_size, p.splits, &[columns])?;
    Ok(contiguous(
        VendoredSource::GgufLinear,
        format!("gguf_decode_{token}_m{height}_{}", destination.suffix()),
        8,
        p.bytes(),
        [columns / TILE_COLUMNS, p.splits, 1],
        [STAGED_THREADS, 1, 1],
    ))
}

/// `gguf_decode_fused_m<height>`, specialized to its segments' formats
/// (function constants 0-2). Inputs: 0 input bf16, 1-9 the segments' plane
/// triples, 10 output bf16, 11 partials, 12 counters; parameters at 13.
pub(crate) fn gguf_decode_fused(
    height: u32,
    p: &GgufDecodeFusedParams,
) -> Result<VendoredDispatch> {
    staged_valid(height, p.input_size, p.splits, &p.cols)?;
    let formats = pass_formats::<3>(&p.fmt[..p.segments().max(1)])?;
    let mut d = contiguous(
        VendoredSource::GgufLinear,
        format!("gguf_decode_fused_m{height}"),
        13,
        p.bytes(),
        [p.cols.iter().sum::<u32>() / TILE_COLUMNS, p.splits, 1],
        [STAGED_THREADS, 1, 1],
    );
    d.constants = uint_constants(&formats);
    Ok(d)
}

// ---------------------------------------------------------------------------
// Selection and acceptance

/// Argmax workspace bytes (values F32, indices U32) of `rows` logits rows.
pub(crate) fn argmax_workspace_bytes(rows: u32) -> u64 {
    u64::from(rows) * u64::from(SAMPLING_SHARDS) * 4
}

/// `decode_sample_argmax_sharded`: each greedy row's shards. Inputs: 0
/// logits F32, 1 constraint masks (any buffer when no lane is constrained),
/// 2 partial values, 3 partial indices, 4 lane layout; parameters at 5.
/// Grid (lane cap x shards, lanes).
pub(crate) fn decode_sample_argmax_sharded(
    p: &TargetSamplingParams,
    lane_cap: u32,
    lanes: u32,
) -> Result<VendoredDispatch> {
    lanes_valid(lanes)?;
    Ok(contiguous(
        VendoredSource::Sampling,
        "decode_sample_argmax_sharded",
        5,
        p.bytes(),
        [lane_cap * SAMPLING_SHARDS, lanes, 1],
        [DEFAULT_THREADS, 1, 1],
    ))
}

/// `decode_sample_argmax_reduce`: the largest logit's token (lowest id on
/// ties). Inputs: 0 partial values, 1 partial indices, 2 output tokens U32,
/// 3 lane layout; parameters at 4. Grid (lane cap, lanes) of 32 threads.
pub(crate) fn decode_sample_argmax_reduce(
    p: &TargetSamplingParams,
    lane_cap: u32,
    lanes: u32,
) -> Result<VendoredDispatch> {
    lanes_valid(lanes)?;
    Ok(contiguous(
        VendoredSource::Sampling,
        "decode_sample_argmax_reduce",
        4,
        p.bytes(),
        [lane_cap, lanes, 1],
        [32, 1, 1],
    ))
}

/// `decode_accept_tree`: each lane's accepted path. Inputs: 0 input tokens,
/// 1 tree rows, 2 output tokens (selections in, retained tokens out), 3
/// retained counts, 4 accepted counts, 5 retained rows (path masks), 6 lane
/// layout; parameters at 7. A thread a lane.
pub(crate) fn decode_accept_tree(p: &AcceptBatchParams, lanes: u32) -> Result<VendoredDispatch> {
    lanes_valid(lanes)?;
    Ok(contiguous(
        VendoredSource::Sampling,
        "decode_accept_tree",
        7,
        p.bytes(),
        [lanes, 1, 1],
        [1, 1, 1],
    ))
}

#[cfg(test)]
#[path = "kernels_tests.rs"]
mod tests;

// ---------------------------------------------------------------------------
// The DFlash2 draft (`decode/draft.metal`, `shared/rope.metal` and the
// selector of `decode/sampling.metal`)

/// Query rows of a lane's draft: the anchor and seven mask rows
/// (`DRAFT_QUERY_ROWS`), and its proposal positions.
pub(crate) const DRAFT_ROWS: u32 = 8;
pub(crate) const DRAFT_POSITIONS: u32 = 7;
/// The compiled draft attention (`metal/abi/DraftAttention.h`): 32 query
/// heads over 8 KV heads of 128 dimensions.
pub(crate) const DRAFT_KV_HEADS: u32 = 8;
pub(crate) const DRAFT_HEAD_DIMENSION: u32 = 128;
pub(crate) const DRAFT_ATTENTION_WIDTH: u32 = 32 * DRAFT_HEAD_DIMENSION;
pub(crate) const DRAFT_QKV_WIDTH: u32 = DRAFT_ATTENTION_WIDTH + 2 * DRAFT_KV_WIDTH;
/// The key (or value) columns of a q|k|v row.
pub(crate) const DRAFT_KV_WIDTH: u32 = DRAFT_KV_HEADS * DRAFT_HEAD_DIMENSION;
/// Attention rows of a KV head's tile: its four query heads' eight rows.
const DRAFT_ATTENTION_ROWS: u32 = 32;
/// Splits of the attention core and ring slots per KV head
/// (`DRAFT_ATTENTION_SPLITS`, `DRAFT_SLIDING_WINDOW`).
pub(crate) const DRAFT_SPLITS: u32 = 4;
pub(crate) const DRAFT_WINDOW: u32 = 2048;
/// Rotated pairs of a draft head (`DRAFT_ROPE_PAIRS`).
pub(crate) const DRAFT_ROPE_PAIRS: u32 = DRAFT_HEAD_DIMENSION / 2;
/// Groups of sixteen channels whose dynamic taps a convolution shares.
const DRAFT_CONVOLUTION_GROUP: u32 = 16;
/// Selector shards, candidates and rank (`DRAFT_SAMPLING_SHARDS`,
/// `DRAFT_CANDIDATES`, `DRAFT_SELECTOR_RANK`).
pub(crate) const DRAFT_SHARDS: u32 = 8;
pub(crate) const DRAFT_CANDIDATES: u32 = 16;
pub(crate) const DRAFT_RANK: u32 = 256;
/// Threads of an edge-table group (`kEdgeThreads`).
const EDGE_THREADS: u32 = 256;

/// The dynamic taps a draft layer projects per row: two stages, two taps,
/// a weight per group of sixteen channels (`draft_dynamic_width`).
pub(crate) const fn draft_dynamic_width(hidden: u32) -> u32 {
    2 * 2 * (hidden / DRAFT_CONVOLUTION_GROUP)
}

/// Bytes of the attention's grouped query rows of `lanes` lanes and the
/// split partials behind them (`DraftAttentionPlan::workspace`).
pub(crate) fn draft_grouped_queries_bytes(lanes: u32) -> u64 {
    let rows = u64::from(lanes) * u64::from(DRAFT_ROWS);
    let partial = u64::from(DRAFT_ATTENTION_ROWS) * u64::from(DRAFT_HEAD_DIMENSION + 2) * 4;
    rows * u64::from(DRAFT_ATTENTION_WIDTH) * 2
        + u64::from(lanes) * u64::from(DRAFT_KV_HEADS) * u64::from(DRAFT_SPLITS) * partial
}

/// Bytes of one lane's keys (or values) ring of one layer: every KV head's
/// window of slots.
pub(crate) fn draft_ring_bytes() -> u64 {
    u64::from(DRAFT_KV_HEADS) * u64::from(DRAFT_WINDOW) * u64::from(DRAFT_HEAD_DIMENSION) * 2
}

/// The stage of a draft convolution: before the sublayer, or after it with
/// the residual.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum DraftConvolutionStage {
    Prepare,
    Residual,
}

/// `DraftConvBatchParams` (8 bytes).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct DraftConvBatchParams {
    pub finish: u32,
    pub hidden: u32,
}
impl Abi for DraftConvBatchParams {
    const SIZE: usize = 8;
    fn put(&self, out: &mut Vec<u8>) {
        put_u32s(out, &[self.finish, self.hidden]);
    }
}

/// `DraftAttentionBatchParams` (40 bytes).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct DraftAttentionBatchParams {
    pub value_stride: u32,
    pub lanes: u32,
    pub cache_length: [u32; MAXIMUM_LANES as usize],
}
impl Abi for DraftAttentionBatchParams {
    const SIZE: usize = 8 + 4 * MAXIMUM_LANES as usize;
    fn put(&self, out: &mut Vec<u8>) {
        put_u32s(out, &[self.value_stride, self.lanes]);
        put_u32s(out, &self.cache_length);
    }
}

/// `DraftContextBatchParams` (32 bytes).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct DraftContextBatchParams {
    pub start_position: [u32; MAXIMUM_LANES as usize],
}
impl Abi for DraftContextBatchParams {
    const SIZE: usize = 4 * MAXIMUM_LANES as usize;
    fn put(&self, out: &mut Vec<u8>) {
        put_u32s(out, &self.start_position);
    }
}

/// `RopeRangeParams` (8 bytes).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct RopeRangeParams {
    pub rows: u32,
    pub start_position: u32,
}
impl Abi for RopeRangeParams {
    const SIZE: usize = 8;
    fn put(&self, out: &mut Vec<u8>) {
        put_u32s(out, &[self.rows, self.start_position]);
    }
}

/// `SelectorBatchParams` (52 bytes): the draft's logits rows hold `columns`
/// tokens of the vocabulary (`leading`, `trailing`: none past the whole
/// vocabulary's).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct SelectorBatchParams {
    pub anchor: [u32; MAXIMUM_LANES as usize],
    pub lanes: u32,
    pub vocabulary: u32,
    pub columns: u32,
    pub leading: u32,
    pub trailing: u32,
}
impl SelectorBatchParams {
    /// One lane of `anchor` over the whole vocabulary of `vocabulary` tokens.
    pub(crate) fn whole(anchor: u32, vocabulary: u32) -> Self {
        let mut anchors = [0; MAXIMUM_LANES as usize];
        anchors[0] = anchor;
        Self {
            anchor: anchors,
            lanes: 1,
            vocabulary,
            columns: vocabulary,
            leading: 0,
            trailing: 0,
        }
    }
}
impl Abi for SelectorBatchParams {
    const SIZE: usize = 20 + 4 * MAXIMUM_LANES as usize;
    fn put(&self, out: &mut Vec<u8>) {
        put_u32s(out, &self.anchor);
        put_u32s(
            out,
            &[
                self.lanes,
                self.vocabulary,
                self.columns,
                self.leading,
                self.trailing,
            ],
        );
    }
}

/// Selector scratch of `positions` proposal positions
/// (`DraftSelector::workspace`): partial IDs U32, partial values F32 (the
/// shards' partials, then each position's 16 x 16 edge table), candidates
/// U32 and unary scores F32, in elements.
pub(crate) fn draft_selector_elements(positions: u32) -> [u64; 4] {
    let candidates = u64::from(positions) * u64::from(DRAFT_CANDIDATES);
    [
        candidates * u64::from(DRAFT_SHARDS),
        candidates * u64::from(DRAFT_SHARDS + DRAFT_CANDIDATES),
        candidates,
        candidates,
    ]
}

/// One 256-thread group per 256 elements, at least one per task
/// (`phaseGroups`).
fn phase_groups(elements: u64, tasks: u64) -> u32 {
    elements.div_ceil(u64::from(DEFAULT_THREADS)).max(tasks) as u32
}

/// `norm_rms`: a weighted RMS norm of bf16 rows with bf16 weights, a
/// threadgroup a row. Inputs: 0 input bf16 `[rows][width]`, 1 weight bf16
/// `[width]`, 2 output bf16; the width at 3.
pub(crate) fn norm_rms_bf16(rows: u32, width: u32) -> VendoredDispatch {
    contiguous(
        VendoredSource::Normalization,
        "norm_rms",
        3,
        width_bytes(width),
        [rows, 1, 1],
        [DEFAULT_THREADS, 1, 1],
    )
}

/// `norm_rms_table16_decode`: [`norm_rms_bf16`] that also writes the
/// register tile's Table16 input of its rows. Inputs: 0 input, 1 weight
/// bf16, 2 output, 3 table bf16, 4 sums F32; the width at 5.
pub(crate) fn norm_rms_table16(rows: u32, width: u32) -> Result<VendoredDispatch> {
    ensure!(
        width.is_multiple_of(TILE_COLUMNS),
        "Table16 rows of whole 64-input spans"
    );
    Ok(contiguous(
        VendoredSource::Normalization,
        "norm_rms_table16_decode",
        5,
        width_bytes(width),
        [rows, 1, 1],
        [DEFAULT_THREADS, 1, 1],
    ))
}

/// `draft_conv` of one lane's eight rows `hidden` wide. Inputs: 0 input bf16,
/// 1 dynamic taps bf16 `[8][dynamic]`, 2 base kernel bf16 `[2][2][hidden]`,
/// 3 residual bf16 (read by the residual stage), 4 output bf16; parameters
/// at 5.
pub(crate) fn draft_conv(stage: DraftConvolutionStage, hidden: u32) -> Result<VendoredDispatch> {
    ensure!(
        hidden > 0 && hidden.is_multiple_of(DRAFT_CONVOLUTION_GROUP),
        "draft convolution of whole channel groups"
    );
    let p = DraftConvBatchParams {
        finish: u32::from(stage == DraftConvolutionStage::Residual),
        hidden,
    };
    Ok(contiguous(
        VendoredSource::Draft,
        "draft_conv",
        5,
        p.bytes(),
        [
            phase_groups(u64::from(DRAFT_ROWS) * u64::from(hidden), 0),
            1,
            1,
        ],
        [DEFAULT_THREADS, 1, 1],
    ))
}

/// `draft_attention_qkv` of one lane: normalized, rotated queries (grouped
/// by head) and keys and the values of its q|k|v rows. Inputs: 0 q|k|v bf16
/// `[8][6144]`, 1 grouped queries bf16, 2 query norm bf16 `[128]`, 3 key
/// norm, 4 RoPE cosines F32 `[16][64]`, 5 sines, 6 query keys bf16
/// `[8 heads][8][128]`, 7 query values bf16 `[8 heads][128][8]`.
pub(crate) fn draft_attention_qkv() -> VendoredDispatch {
    let groups = phase_groups(
        u64::from(DRAFT_ROWS * DRAFT_KV_WIDTH),
        u64::from(DRAFT_ROWS * (32 + DRAFT_KV_HEADS)),
    );
    contiguous(
        VendoredSource::Draft,
        "draft_attention_qkv",
        8,
        Vec::new(),
        [groups, 1, 1],
        [DEFAULT_THREADS, 1, 1],
    )
}

/// The attention parameters of one lane at `cache_length` context tokens.
pub(crate) fn draft_attention_params(cache_length: u32) -> DraftAttentionBatchParams {
    let mut lengths = [0; MAXIMUM_LANES as usize];
    lengths[0] = cache_length;
    DraftAttentionBatchParams {
        value_stride: DRAFT_WINDOW,
        lanes: 1,
        cache_length: lengths,
    }
}

/// `draft_attention_bf16_split`: every split of each KV head over the live
/// ring tiles, the last adding the eight block rows. Inputs: 0 grouped
/// queries (and the partials behind them), 1-8 each lane's keys ring, 9-16
/// its values ring, 17 query keys, 18 query values; parameters at 19.
pub(crate) fn draft_attention_split(p: &DraftAttentionBatchParams) -> Result<VendoredDispatch> {
    ensure!(
        p.lanes == 1 && p.value_stride == DRAFT_WINDOW,
        "one draft lane over its window"
    );
    Ok(contiguous(
        VendoredSource::Draft,
        "draft_attention_bf16_split",
        19,
        p.bytes(),
        [DRAFT_KV_HEADS, p.lanes, DRAFT_SPLITS],
        [DEFAULT_THREADS, 1, 1],
    ))
}

/// `draft_attention_bf16_reduce`: the splits' fixed-order combine into the
/// output projection's input rows, in place of the grouped queries. Input:
/// 0 grouped queries; parameters at 1.
pub(crate) fn draft_attention_reduce(p: &DraftAttentionBatchParams) -> VendoredDispatch {
    contiguous(
        VendoredSource::Draft,
        "draft_attention_bf16_reduce",
        1,
        p.bytes(),
        [DRAFT_KV_HEADS, p.lanes, 1],
        [DEFAULT_THREADS, 1, 1],
    )
}

/// `draft_context_kv_commit` of one lane whose path may hold up to
/// `lane_cap` rows: its retained context rows' normalized, rotated keys and
/// values into its rings from `start_position` on. Inputs: 0 context K|V
/// bf16 `[rows][2048]`, 1 key norm, 2 RoPE cosines, 3 sines, 4-11 each lane's
/// keys ring, 12-19 its values ring, 20 retained rows U32 (the path mask),
/// 21 lane rows (`LaneRows`); parameters at 22.
pub(crate) fn draft_context_kv_commit(
    start_position: u32,
    lane_cap: u32,
) -> Result<VendoredDispatch> {
    ensure!(
        (1..=VERIFY_ROWS).contains(&lane_cap),
        "a lane commits at most a verify block's rows"
    );
    let mut positions = [0; MAXIMUM_LANES as usize];
    positions[0] = start_position;
    let p = DraftContextBatchParams {
        start_position: positions,
    };
    Ok(contiguous(
        VendoredSource::Draft,
        "draft_context_kv_commit",
        22,
        p.bytes(),
        [lane_cap * DRAFT_KV_HEADS, 1, 1],
        [DEFAULT_THREADS, 1, 1],
    ))
}

/// `rope_build_draft_range_tables`: the draft's RoPE rows of `rows`
/// consecutive positions from `start_position`. Inputs: 0 inverse
/// frequencies F32 `[64]`, 1 cosines F32 `[rows][64]`, 2 sines; parameters
/// at 3.
pub(crate) fn rope_draft_range(rows: u32, start_position: u32) -> VendoredDispatch {
    let p = RopeRangeParams {
        rows,
        start_position,
    };
    contiguous(
        VendoredSource::Rope,
        "rope_build_draft_range_tables",
        3,
        p.bytes(),
        [(rows * DRAFT_ROPE_PAIRS).div_ceil(DEFAULT_THREADS), 1, 1],
        [DEFAULT_THREADS, 1, 1],
    )
}

/// The draft's inverse frequencies of base `theta`, each `theta^(-i / 64)`
/// in F32 as the reference's host computes them.
pub(crate) fn draft_inverse_frequencies(theta: f32) -> Vec<f32> {
    (0..DRAFT_ROPE_PAIRS)
        .map(|i| theta.powf(-(i as f32) / DRAFT_ROPE_PAIRS as f32))
        .collect()
}

/// `draft_select_top16_sharded`: each proposal row's shards' sixteen best
/// columns. Inputs: 0 logits F32 `[8][columns]`, 1 partial IDs, 2 partial
/// values; parameters at 3.
pub(crate) fn draft_select_top16(p: &SelectorBatchParams) -> Result<VendoredDispatch> {
    ensure!(p.lanes == 1, "one draft lane");
    Ok(contiguous(
        VendoredSource::Sampling,
        "draft_select_top16_sharded",
        3,
        p.bytes(),
        [p.lanes * DRAFT_POSITIONS * DRAFT_SHARDS, 1, 1],
        [DEFAULT_THREADS, 1, 1],
    ))
}

/// `draft_select_edges`: each position's sixteen candidates, unary scores
/// and edge table. Inputs: 0 partial IDs, 1 partial values (the edge tables
/// follow the partials), 2 candidates U32 `[7][16]`, 3 unary F32 `[7][16]`,
/// 4 selector hidden bf16 `[8][256]`, 5 predecessor codebook bf16, 6
/// successor codebook bf16; parameters at 7.
pub(crate) fn draft_select_edges(p: &SelectorBatchParams) -> Result<VendoredDispatch> {
    ensure!(p.lanes == 1, "one draft lane");
    Ok(contiguous(
        VendoredSource::Sampling,
        "draft_select_edges",
        7,
        p.bytes(),
        [p.lanes * DRAFT_POSITIONS, 1, 1],
        [EDGE_THREADS, 1, 1],
    ))
}

/// The epilogue of a dense projection's pass: plain, or the up pass of a
/// gate/up pair that multiplies `silu(gate)` from the gate scratch (`_g`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Epilogue {
    Plain(Destination),
    UpWithGate,
}
impl Epilogue {
    fn suffix(self) -> &'static str {
        match self {
            Self::Plain(d) => d.suffix(),
            Self::UpWithGate => "g",
        }
    }
}

/// `gguf_decode_sg_<token>_l<lanes>_<suffix>`: [`gguf_decode_sg`] of any
/// epilogue. Inputs as [`gguf_decode_sg`], 8 the aux input (the gate
/// scratch of an up pass).
pub(crate) fn gguf_decode_sg_epilogue(
    token: &str,
    lanes: u32,
    epilogue: Epilogue,
    p: &GgufDecodeParams,
    columns: u32,
) -> Result<VendoredDispatch> {
    decode_valid(p.input_size, p.splits, &[columns])?;
    ensure!(
        (1..=4).contains(&lanes) && p.input_size / 64 >= p.splits,
        "invalid register tile"
    );
    Ok(contiguous(
        VendoredSource::LinearRegister,
        format!("gguf_decode_sg_{token}_l{lanes}_{}", epilogue.suffix()),
        9,
        p.bytes(),
        [columns / TILE_COLUMNS, p.splits, 1],
        [REGISTER_THREADS, 1, 1],
    ))
}

/// `gguf_decode_<token>_m<height>_<suffix>`: [`gguf_decode`] of any
/// epilogue. Inputs as [`gguf_decode`], 7 the aux input.
pub(crate) fn gguf_decode_epilogue(
    token: &str,
    height: u32,
    epilogue: Epilogue,
    p: &GgufDecodeParams,
    columns: u32,
) -> Result<VendoredDispatch> {
    staged_valid(height, p.input_size, p.splits, &[columns])?;
    Ok(contiguous(
        VendoredSource::GgufLinear,
        format!("gguf_decode_{token}_m{height}_{}", epilogue.suffix()),
        8,
        p.bytes(),
        [columns / TILE_COLUMNS, p.splits, 1],
        [STAGED_THREADS, 1, 1],
    ))
}

/// `gguf_decode_<token>_m<height>_gate_up`: a gate/up pair of one format in
/// one pass, `bf16(up) * silu(bf16(gate))`. Inputs: 0 input bf16, 1-3 the
/// gate's planes (plane0, plane1 or meta, meta), 4-6 the up's, 7 output
/// bf16, 8 partials, 9 counters (the output in both with one split);
/// parameters at 10. Grid (columns / 32, splits) of one simdgroup.
pub(crate) fn gguf_decode_gate_up(
    token: &str,
    height: u32,
    p: &GgufDecodeParams,
    columns: u32,
) -> Result<VendoredDispatch> {
    staged_valid(height, p.input_size, p.splits, &[columns])?;
    ensure!(height <= 32, "a one-pass pair holds up to 32 rows");
    Ok(contiguous(
        VendoredSource::GgufLinear,
        format!("gguf_decode_{token}_m{height}_gate_up"),
        10,
        p.bytes(),
        [columns / GATE_UP_COLUMNS, p.splits, 1],
        [GATE_UP_THREADS, 1, 1],
    ))
}
/// Columns and threads of a one-pass gate/up threadgroup
/// (`GGUF_STAGED_COLUMNS`, `GGUF_GATE_UP_THREADS`).
const GATE_UP_COLUMNS: u32 = 32;
const GATE_UP_THREADS: u32 = 32;

/// `gguf_column_gate_up`: a gate/up pair of 1-2 rows in one pass. Inputs: 0
/// input bf16, 1-3 the gate's planes, 4 output bf16, 5-7 the up's planes;
/// parameters at 8.
pub(crate) fn gguf_column_gate_up(p: &GgufColumnParams, columns: u32) -> Result<VendoredDispatch> {
    ensure!(
        (1..=2).contains(&p.rows) && columns.is_multiple_of(COLUMN_COLUMNS),
        "invalid column tile"
    );
    Ok(contiguous(
        VendoredSource::LinearColumn,
        "gguf_column_gate_up",
        8,
        p.bytes(),
        [columns / COLUMN_COLUMNS, 1, 1],
        [COLUMN_THREADS, 1, 1],
    ))
}

// ---------------------------------------------------------------------------
// Prompt chunks

/// Rows of a direct relative attention tile
/// (`RELATIVE_ATTENTION_DIRECT_ROWS`).
pub(crate) const DIRECT_ROWS: u32 = 16;
/// Tile splits a prompt chunk's sequence takes at most
/// (`kAttentionSequenceSplits`), which its workspace holds.
pub(crate) const SEQUENCE_SPLITS: u32 = 256;
/// Splits of a one-tile sequence, at least and at most (`kSharedSplits`,
/// `kAttentionMaximumSplits`), and its pages a split (`kTilePagesPerSplit`).
const SHARED_SPLITS: u32 = 32;
const ATTENTION_MAXIMUM_SPLITS: u32 = 128;
const TILE_PAGES_PER_SPLIT: u64 = 16;

/// The history splits of each of a prompt sequence's `tiles` query tiles
/// reading `pages` pages (`attentionSplits`), with a window of
/// `window_tokens` keys before each row or none.
pub(crate) fn sequence_splits(
    tiles: u32,
    pages: u64,
    window_tokens: u32,
    pages_per_split: u32,
) -> u32 {
    let fewest = (SHARED_SPLITS / tiles).clamp(1, SHARED_SPLITS);
    let most = (SEQUENCE_SPLITS / tiles).clamp(fewest, SHARED_SPLITS);
    let splits = if tiles == 1 {
        pages.div_ceil(TILE_PAGES_PER_SPLIT).clamp(
            u64::from(SHARED_SPLITS),
            u64::from(ATTENTION_MAXIMUM_SPLITS),
        ) as u32
    } else {
        (pages / u64::from(pages_per_split)).clamp(u64::from(fewest), u64::from(most)) as u32
    };
    if window_tokens > 0 {
        splits.min(window_attention_pages(window_tokens))
    } else {
        splits
    }
}

/// Attention partials and statistics bytes of a prompt sequence's workspace
/// (`RelativeAttention::sequenceWorkspace`).
pub(crate) fn sequence_workspace_bytes() -> (u64, u64) {
    let rows = u64::from(SEQUENCE_SPLITS) * u64::from(ATTENTION_TILE_ROWS * QUERY_HEADS);
    (rows * u64::from(HEAD_DIMENSION) * 4, rows * 2 * 4)
}

/// Query, relative-state and sink staging bytes of a prompt sequence staged
/// `chunk_stride` rows a KV head (`RelativeAttention::stagingBytes`).
pub(crate) fn staging_bytes(chunk_stride: u32) -> (u64, u64, u64) {
    let fused = u64::from(KV_HEADS) * u64::from(chunk_stride) * 3;
    (
        fused * u64::from(HEAD_DIMENSION) * 2,
        fused * u64::from(RELATIVE_STATES) * 4,
        fused * 4,
    )
}
/// Relative states a query head (`SPLASH_RELATIVE_ATTENTION_STATES`).
const RELATIVE_STATES: u32 = 16;

/// A prompt chunk's sequence of `rows` rows after `committed` tokens, staged
/// in whole KV pages (`sequenceRows` of `chunkParams`).
pub(crate) fn prompt_sequence(
    committed: u32,
    rows: u32,
    page_table_entries: u32,
    splits: u32,
) -> AttentionSequence {
    AttentionSequence {
        committed_tokens: committed,
        row_bound: rows,
        layout_row: 0,
        stage_row: 0,
        chunk_stride: rows.next_multiple_of(PAGE_TOKENS),
        page_table_entries,
        split_count: splits,
        first_tile: 0,
        first_slot: 0,
        block_stride: 0,
        block_list: 0,
    }
}

/// `relative_attention_q8_prepare[_scaled]`: a prompt chunk's queries,
/// relative states and sinks staged, its keys and values clipped,
/// normalized and stored in their pages. Inputs: 0 projection rows bf16
/// (stride 6592), 1 convolved keys, 2 convolved values, 3 query scale, 4
/// relative-state scale, 5 staged queries bf16, 6 staged states F32, 7 staged
/// sinks F32, 8-15 lane page tables, 16 tree rows (input 0 without), 17 lane
/// layout; parameters at 18. Grid (24 + 8 heads, rows, sequences) of 128.
pub(crate) fn relative_attention_prepare(
    p: &AttentionParams,
    scaled: bool,
) -> Result<VendoredDispatch> {
    lanes_valid(p.sequences)?;
    let rows = p.sequence[..p.sequences as usize]
        .iter()
        .map(|s| s.row_bound)
        .max()
        .unwrap_or(0);
    Ok(contiguous(
        VendoredSource::RelativeAttention,
        format!(
            "relative_attention_q8_prepare{}",
            if scaled { "_scaled" } else { "" }
        ),
        18,
        p.bytes(),
        [QUERY_HEADS + KV_HEADS, rows, p.sequences],
        [HEAD_DIMENSION, 1, 1],
    ))
}

fn window_suffix(window: bool) -> &'static str {
    if window { "_w1023" } else { "" }
}
fn query_format(int8_queries: bool) -> &'static str {
    if int8_queries { "q8q" } else { "q8" }
}

/// `relative_attention_{q8|q8q}_split[_w1023]`: each tile split's partials
/// and statistics. Inputs: 0 staged queries, 1 partials F32, 2 statistics
/// F32, 3-10 lane page tables, 11 tree rows (input 1 without), 12 lane
/// layout, 13 staged states, 14 relative bias F32; parameters at 15. Grid
/// (KV heads, tiles, splits).
pub(crate) fn relative_attention_split(
    p: &AttentionParams,
    int8_queries: bool,
    window: bool,
    splits: u32,
) -> Result<VendoredDispatch> {
    lanes_valid(p.sequences)?;
    Ok(contiguous(
        VendoredSource::RelativeAttention,
        format!(
            "relative_attention_{}_split{}",
            query_format(int8_queries),
            window_suffix(window)
        ),
        15,
        p.bytes(),
        [KV_HEADS, p.tiles, splits],
        [DEFAULT_THREADS, 1, 1],
    ))
}

/// `relative_attention_reduce[_w1023][_table16]`: each fused row's splits
/// with its sink, gated, into the hidden rows (and with `table16` the
/// out-projection's Table16). Inputs: 0 partials, 1 statistics, 2
/// projection rows (the gates), 3 hidden bf16 `[rows][3072]`, 4 table (the
/// hidden rows without), 5 sums (the hidden rows without), 6 lane layout, 7
/// staged sinks; parameters at 8. Grid (KV heads, 24 fused rows, tiles) of
/// 128.
pub(crate) fn relative_attention_reduce(
    p: &AttentionParams,
    window: bool,
    table16: bool,
) -> Result<VendoredDispatch> {
    lanes_valid(p.sequences)?;
    Ok(contiguous(
        VendoredSource::RelativeAttention,
        format!(
            "relative_attention_reduce{}{}",
            window_suffix(window),
            if table16 { "_table16" } else { "" }
        ),
        8,
        p.bytes(),
        [KV_HEADS, ATTENTION_TILE_ROWS * 3, p.tiles],
        [HEAD_DIMENSION, 1, 1],
    ))
}

/// `relative_attention_{q8|q8q}_direct[_w1023]`: a sequence of one split
/// attended and reduced in one pass over tiles of [`DIRECT_ROWS`] rows
/// (`p.tiles` of them). Inputs: 0 staged queries, 1 partials (INT8 query
/// codes), 2-9 lane page tables, 10 tree rows (input 1 without), 11 lane
/// layout, 12 staged states, 13 relative bias, 14 projection rows, 15
/// hidden, 16 table, 17 sums (the hidden rows in both), 18 staged sinks;
/// parameters at 19. Grid (KV heads, tiles).
pub(crate) fn relative_attention_direct(
    p: &AttentionParams,
    int8_queries: bool,
    window: bool,
) -> Result<VendoredDispatch> {
    lanes_valid(p.sequences)?;
    Ok(contiguous(
        VendoredSource::RelativeAttention,
        format!(
            "relative_attention_{}_direct{}",
            query_format(int8_queries),
            window_suffix(window)
        ),
        19,
        p.bytes(),
        [KV_HEADS, p.tiles, 1],
        [DEFAULT_THREADS, 1, 1],
    ))
}

/// `moe_group_routes_s0`: a prompt chunk's routes in expert tiles, writing
/// the expert passes' indirect grids. Inputs: 0 selected, 1 tile
/// descriptors, 2 tile count, 3 grouped routes, 4 route rows, 5 slot table,
/// 6 expert slots; parameters at 7. One threadgroup.
pub(crate) fn moe_group_routes(p: &MoeGroupParams) -> Result<VendoredDispatch> {
    ensure!(
        p.lanes == 0 && p.experts <= EXPERT_SLOTS && EXPERT_TILE_ROWS.contains(&p.tile_rows),
        "invalid expert grouping"
    );
    Ok(contiguous(
        VendoredSource::Moe,
        "moe_group_routes_s0",
        7,
        p.bytes(),
        [1, 1, 1],
        [EXPERT_SLOTS, 1, 1],
    ))
}

/// `moe_combine`: each row's routes' weighted sum plus the residual rows.
/// Inputs: 0 expert outputs bf16, 1 route rows, 2 routing weights, 3
/// residual bf16 (zero rows), 4 output bf16; parameters at 5. Grid (rows,
/// hidden / 256).
pub(crate) fn moe_combine(p: &MoeCombineParams) -> Result<VendoredDispatch> {
    ensure!(
        p.hidden_size.is_multiple_of(256),
        "combined rows of whole 256 columns"
    );
    Ok(contiguous(
        VendoredSource::Moe,
        "moe_combine",
        5,
        p.bytes(),
        [p.rows, p.hidden_size / 256, 1],
        [DEFAULT_THREADS, 1, 1],
    ))
}

/// `gguf_float_na_<weights>_<output>`: [`gguf_float`] on the neural
/// accelerator tile (64 x 32 tiles). Inputs as [`gguf_float`]. Grid
/// (output columns / 32 summed, rows / 64) of 128 threads.
pub(crate) fn gguf_float_na(
    weights_f32: bool,
    output_f32: bool,
    p: &GgufFloatParams,
) -> Result<VendoredDispatch> {
    ensure!(
        p.rows >= 16 && p.input_size.is_multiple_of(32),
        "the accelerator float tile takes 16 rows of whole 32-input spans"
    );
    let tiles = p.output_size.iter().map(|n| n.div_ceil(32)).sum::<u32>();
    Ok(contiguous(
        VendoredSource::GgufFloat,
        format!(
            "gguf_float_na_{}_{}",
            if weights_f32 { "f32" } else { "bf16" },
            if output_f32 { "f32" } else { "bf16" }
        ),
        5,
        p.bytes(),
        [tiles, p.rows.div_ceil(64), 1],
        [128, 1, 1],
    ))
}

/// `GgufPrefillSegmentsParams` (36 bytes): up to three segments of one
/// format.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct GgufPrefillSegmentsParams {
    pub input_size: u32,
    pub rows: u32,
    pub out_stride: u32,
    pub cols: [u32; 3],
    pub offset: [u32; 3],
}
impl Abi for GgufPrefillSegmentsParams {
    const SIZE: usize = 36;
    fn put(&self, out: &mut Vec<u8>) {
        put_u32s(out, &[self.input_size, self.rows, self.out_stride]);
        put_u32s(out, &self.cols);
        put_u32s(out, &self.offset);
    }
}

/// Rows of a prefill tile (`GGUF_PREFILL_ROWS`) and its threads.
pub(crate) const PREFILL_ROWS: u32 = 128;
const PREFILL_THREADS: u32 = 128;

/// `gguf_prefill_<token>_a`: the bf16 prefill tile over every segment of
/// one format, 128-row tiles of `storage_rows`. Inputs: 0 input bf16
/// (`storage_rows` readable rows), 1-9 the segments' plane triples (slots
/// past the last repeat its planes), 10 output bf16; parameters at 11.
pub(crate) fn gguf_prefill(
    token: &str,
    p: &GgufPrefillSegmentsParams,
    storage_rows: u32,
) -> Result<VendoredDispatch> {
    decode_valid(p.input_size, 1, &p.cols)?;
    ensure!(
        storage_rows.is_multiple_of(PREFILL_ROWS) && storage_rows >= p.rows,
        "prefill tiles of whole 128-row tiles"
    );
    Ok(contiguous(
        VendoredSource::GgufLinear,
        format!("gguf_prefill_{token}_a"),
        11,
        p.bytes(),
        [
            storage_rows / PREFILL_ROWS,
            p.cols.iter().sum::<u32>() / TILE_COLUMNS,
            1,
        ],
        [PREFILL_THREADS, 1, 1],
    ))
}

/// `GgufPrefillInt8QuantizeParams` (12 bytes).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct GgufPrefillInt8QuantizeParams {
    pub input_size: u32,
    pub rows: u32,
    pub storage_rows: u32,
}
impl Abi for GgufPrefillInt8QuantizeParams {
    const SIZE: usize = 12;
    fn put(&self, out: &mut Vec<u8>) {
        put_u32s(out, &[self.input_size, self.rows, self.storage_rows]);
    }
}

/// `GgufPrefillInt8Params` (28 bytes).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct GgufPrefillInt8Params {
    pub input_size: u32,
    pub rows: u32,
    pub storage_rows: u32,
    pub out_stride: u32,
    pub out_offset: u32,
    pub plane_input_size: u32,
    pub column_tiles: u32,
}
impl Abi for GgufPrefillInt8Params {
    const SIZE: usize = 28;
    fn put(&self, out: &mut Vec<u8>) {
        put_u32s(
            out,
            &[
                self.input_size,
                self.rows,
                self.storage_rows,
                self.out_stride,
                self.out_offset,
                self.plane_input_size,
                self.column_tiles,
            ],
        );
    }
}

/// Inputs an int8 code group covers, the int8 tile's rows and threads, its
/// walk down a column tile, and the quantizer's simdgroups.
const PREFILL_INT8_GROUP: u32 = 64;
const PREFILL_INT8_ROWS: u32 = 32;
const PREFILL_INT8_THREADS: u32 = 64;
const PREFILL_INT8_WALK: u32 = 16;
const PREFILL_INT8_QUANTIZE_SIMDGROUPS: u32 = 8;

/// Int8 codes and scale bytes of `storage_rows` rows of `width` inputs
/// (`gguf_prefill_int8_table_bytes`, `_scale_bytes`).
pub(crate) fn prefill_int8_bytes(width: u32, storage_rows: u32) -> (u64, u64) {
    let (w, s) = (u64::from(width), u64::from(storage_rows));
    (w * s, w / u64::from(PREFILL_INT8_GROUP) * s * 8)
}

/// `gguf_prefill_int8_quantize`: each 64-input group of each row as int8
/// codes and its scales. Inputs: 0 input bf16, 1 codes, 2 scales; parameters
/// at 3. Grid (group blocks, rows) of 8 simdgroups.
pub(crate) fn gguf_prefill_int8_quantize(
    p: &GgufPrefillInt8QuantizeParams,
) -> Result<VendoredDispatch> {
    ensure!(
        p.input_size.is_multiple_of(PREFILL_INT8_GROUP) && p.storage_rows >= p.rows,
        "int8 rows of whole 64-input groups"
    );
    let groups = p.input_size / PREFILL_INT8_GROUP;
    Ok(contiguous(
        VendoredSource::GgufPrefillInt8,
        "gguf_prefill_int8_quantize",
        3,
        p.bytes(),
        [groups.div_ceil(PREFILL_INT8_QUANTIZE_SIMDGROUPS), p.rows, 1],
        [PREFILL_INT8_QUANTIZE_SIMDGROUPS * 32, 1, 1],
    ))
}

/// `gguf_prefill_int8_<token>_a`: one segment on the int8 prefill tile over
/// the quantized rows. Inputs: 0 codes, 1 scales, 2 plane0, 3 plane1 (meta
/// without), 4 meta, 5 output bf16, 6 aux (the output); parameters at 7.
/// Grid (16, tiles / 16) of the walk over 32-row by 64-column tiles.
pub(crate) fn gguf_prefill_int8(
    token: &str,
    p: &GgufPrefillInt8Params,
) -> Result<VendoredDispatch> {
    ensure!(
        p.input_size.is_multiple_of(PREFILL_INT8_GROUP) && p.storage_rows >= p.rows,
        "invalid int8 prefill tile"
    );
    let tiles = p.column_tiles * p.rows.div_ceil(PREFILL_INT8_ROWS);
    Ok(contiguous(
        VendoredSource::GgufPrefillInt8,
        format!("gguf_prefill_int8_{token}_a"),
        7,
        p.bytes(),
        [PREFILL_INT8_WALK, tiles.div_ceil(PREFILL_INT8_WALK), 1],
        [PREFILL_INT8_THREADS, 1, 1],
    ))
}

/// `DraftContextParams` (8 bytes).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct DraftContextParams {
    pub tokens: u32,
    pub start_position: u32,
}
impl Abi for DraftContextParams {
    const SIZE: usize = 8;
    fn put(&self, out: &mut Vec<u8>) {
        put_u32s(out, &[self.tokens, self.start_position]);
    }
}

/// `prefill_draft_context_kv`: `tokens` consecutive context rows'
/// normalized, rotated keys and values into one lane's rings from
/// `start_position` on. Inputs: 0 context K|V bf16 `[tokens][2048]`, 1 key
/// norm, 2 RoPE cosines, 3 sines (of those rows), 4 keys ring, 5 values
/// ring; parameters at 6. Grid (tokens x KV heads).
pub(crate) fn prefill_draft_context_kv(tokens: u32, start_position: u32) -> VendoredDispatch {
    let p = DraftContextParams {
        tokens,
        start_position,
    };
    contiguous(
        VendoredSource::DraftContext,
        "prefill_draft_context_kv",
        6,
        p.bytes(),
        [tokens * DRAFT_KV_HEADS, 1, 1],
        [DEFAULT_THREADS, 1, 1],
    )
}
