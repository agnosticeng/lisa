//! The pinned reference's Inkling decode/verify step of one lane: 1 to 16
//! rows in tree mode (a chain is a tree), every dispatch its vendored
//! kernels as its host encodes them ([`lisa_backend::VendoredDispatch`]),
//! over the persistent state of [`super::state`] and per-step scratch sized
//! for sixteen storage rows. A forward writes softcapped logits and the
//! draft's captures; once the host has chosen the accepted path, the commit
//! moves its keys and values to their committed slots and writes the next
//! convolution state along it.
//!
//! Numerics are the reference's own (bf16 activations, its dense tiles,
//! expert passes, attention and convolutions); bits depend on the GPU's
//! cores through the attention splits and the dense tiles' K splits. Two
//! addition of LISA's own: image rows replace token rows after the
//! embedding norm.
use super::{
    calibration::{Calibration, FloatRows, FloatTile, expert_tile_rows, float_tile},
    dense::{
        DENSE_ROWS, DenseShape, DenseTile, PACKED_SEGMENTS, ROW_GROUP_ROWS, STEP_PROJECTIONS,
        Selection, StepKind, Unit, split_scratch,
    },
    kernels::{
        self as k, AttentionParams, AttentionVariant, BlockResidualMixParams, ConvolutionOutput,
        Destination, GateWeights, GgufColumnFusedParams, GgufColumnParams, GgufDecodeFusedParams,
        GgufDecodeParams, GgufEmbedParams, GgufFloatParams, GgufPrefillInt8Params,
        GgufPrefillInt8QuantizeParams, GgufPrefillSegmentsParams, KvPathCommitParams,
        LogitFinishParams, MessageGateFusedParams, MixCurrent, MoeBiasedRouteParams,
        MoeCombineConvolutionParams, MoeCombineParams, MoeGatherParams, MoeGgufExpertParams,
        MoeGgufGateUpParams, MoeGroupParams, MoeRouteParams, RowCopyParams, Segment,
        ShortConvolutionParams,
    },
    schedule::{
        self as sched, ATTENTION_WIDTH, CAPTURES, EXPERTS, FINAL_CAPTURE, FIRST_PASS_PLANE,
        GATE_RANK, HIDDEN, INTERMEDIATE, KV_WIDTH, MESSAGES, PACKED_WIDTH, PREVIOUS_PLANES,
        ROUTED_SCALE, Rows, SINK_COLUMN, SINKS, SOFTCAP, STACK_PLANES, TAP_STRIDE, TAPS, TOKENS,
        TOP_K, VOCABULARY,
    },
    state::{
        APPLICATION_COLUMNS, APPLICATIONS, FFN_COLUMN, KEY_COLUMN, MIXER_COLUMN, PAGE_TOKENS,
        STATE_COLUMNS, StepState, VALUE_COLUMN, region,
    },
};
use crate::tensor_ops::TensorOps;
use anyhow::{Context, Result, bail, ensure};
use lisa_backend::{
    Device, Kernel, PackedCompute as P, ScalarType as Dtype, TensorAccess, TensorOperation,
    VendoredDispatch,
};
use lisa_models::inkling2::planes::{self, Format};
use std::collections::BTreeMap;

/// Storage rows of every row buffer of a decode step: a verify block's
/// most.
pub(crate) const STORAGE_ROWS: u32 = DENSE_ROWS;
/// The attention window of the sliding layers.
const WINDOW: u32 = k::WINDOW_TOKENS;
/// Bytes of the message gates' counters (`[3 gates][16 groups]` u32).
const GATE_COUNTER_BYTES: u64 = 192;
/// Rows the decode step's expert passes run in a tile.
const EXPERT_TILE: u32 = 8;

// ---------------------------------------------------------------------------
// Pure geometry

/// The attention of one variant for a step: its history splits and whether
/// it scores INT8 queries (`RelativeAttention::plan` for a decode step).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct AttentionChoice {
    pub splits: u32,
    pub int8_queries: bool,
}

/// The decode attention of a lane at `committed` tokens and lane cap `cap`
/// on the device of `calibration`, windowed (sliding layers) or over every
/// page.
pub(crate) fn attention_choice(
    committed: u32,
    cap: u32,
    window: bool,
    calibration: &Calibration,
) -> AttentionChoice {
    let pages = k::read_pages(
        u64::from(committed) + u64::from(cap),
        if window { WINDOW } else { 0 },
    );
    let splits = k::decode_splits(pages, calibration.cores, cap);
    AttentionChoice {
        splits,
        int8_queries: k::split_reads(pages, splits)
            >= u64::from(calibration.attention.int8_queries_from_pages),
    }
}

/// The attention of one variant for a prompt chunk of `rows` rows after
/// `committed` tokens (`RelativeAttention::plan` for a prefill step): its
/// tiles' history splits, whether it scores INT8 queries, and whether its
/// one split runs the direct kernel.
pub(crate) fn prompt_attention(
    committed: u32,
    rows: u32,
    window: bool,
    calibration: &Calibration,
) -> AttentionChoice {
    let window = if window { WINDOW } else { 0 };
    let pages = k::read_pages(u64::from(committed) + u64::from(rows), window);
    let tiles = rows.div_ceil(k::ATTENTION_TILE_ROWS);
    let splits = k::sequence_splits(tiles, pages, window, calibration.attention.pages_per_split);
    AttentionChoice {
        splits,
        int8_queries: k::split_reads(pages, splits)
            >= u64::from(calibration.attention.int8_queries_from_pages),
    }
}

/// `sqrt(taps + 1)`, or `sqrt(2 taps + 1)` = 3 with dynamic taps.
fn convolution_divisor(dynamic: bool) -> f32 {
    (if dynamic { 9.0f64 } else { 5.0f64 }).sqrt() as f32
}

/// A routed expert projection's plane format (`kQuantFormats`): the run-time
/// id, plane0 and plane1 bytes per row and 32-element group, meta bytes per
/// unit and groups per unit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct PlaneFormat {
    pub id: u32,
    pub plane0: u64,
    pub plane1: u64,
    pub meta: u64,
    pub meta_groups: u64,
}

/// The plane format the reference's loader stores a GGML type of `columns`
/// inputs a row in.
pub(crate) fn plane_format(ggml_type: u32, columns: u32) -> Result<PlaneFormat> {
    let (id, plane0, plane1, meta, meta_groups) = match ggml_type {
        40 if columns.is_multiple_of(256) => (37, 16, 0, 20, 8),
        40 => (44, 16, 0, 8, 2),
        13 => (3, 16, 4, 16, 8),
        8 => (6, 32, 0, 2, 1),
        other => bail!("no vendored expert planes for GGML type {other}"),
    };
    Ok(PlaneFormat {
        id,
        plane0,
        plane1,
        meta,
        meta_groups,
    })
}

/// Byte offsets of plane0, plane1 (meta's for a format without one) and meta
/// of `experts x rows` rows of `columns` inputs, and the total bytes: the
/// chunk table's addresses of one projection.
pub(crate) fn plane_offsets(
    f: PlaneFormat,
    experts: u64,
    rows: u64,
    columns: u64,
) -> ([u64; 3], u64) {
    let groups = columns / 32;
    let total_rows = experts * rows;
    let plane1 = total_rows * groups * f.plane0;
    let meta = plane1 + total_rows * groups * f.plane1;
    let total = meta + total_rows * (groups / f.meta_groups) * f.meta;
    ([0, if f.plane1 == 0 { meta } else { plane1 }, meta], total)
}

/// Bytes of a dense Q8_0 projection's planes of `n` rows and `k` inputs,
/// and meta's offset.
pub(crate) fn dense_plane_bytes(n: u64, k: u64) -> (u64, u64) {
    let groups = k / 32;
    (n * groups * 32, n * groups * 34)
}

/// The sizes of a step's scratch buffers, in elements of their types (the
/// table in bytes).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ScratchSizes {
    pub attention_partials: u64,
    pub attention_statistics: u64,
    pub table_bytes: u64,
    pub sums: u64,
    pub split_partials: u64,
    pub split_counters: u64,
    pub maximum_tiles: u64,
    pub grouped_input_bytes: u64,
    pub intermediate: u64,
    pub expert_output: u64,
    pub gate_low: u64,
    pub gate_values: u64,
    pub gate_partials: u64,
}

/// Scratch of a step of up to `rows` storage rows on the device of
/// `calibration`: a decode step's ([`STORAGE_ROWS`]) or a prompt chunk's.
pub(crate) fn scratch_sizes(calibration: &Calibration, rows: u32, prompt: bool) -> ScratchSizes {
    let s = u64::from(rows);
    let (partials, statistics) = if prompt {
        k::sequence_workspace_bytes()
    } else {
        k::attention_workspace_bytes(&[k::DECODE_SPLITS], k::VERIFY_ROWS)
    };
    // Table16 input of the widest projection (the attention output's) over
    // every 8-row tile the register tile holds; a prompt chunk's int8 codes
    // and scales of its rows.
    let register_rows = rows.min(32);
    let (mut table, mut sums) =
        k::table16_bytes(ATTENTION_WIDTH, register_rows.div_ceil(k::TABLE16_ROWS));
    if prompt {
        let (codes, scales) = k::prefill_int8_bytes(ATTENTION_WIDTH, rows);
        table = table.max(codes);
        sums = sums.max(scales);
    }
    let kind = if prompt {
        StepKind::Prefill
    } else {
        StepKind::Decode
    };
    let selection = Selection {
        calibration,
        kind,
        int8: true,
    };
    let (split_partials, split_counters) = split_scratch(
        &selection,
        if prompt {
            ROW_GROUP_ROWS.min(rows)
        } else {
            STORAGE_ROWS
        },
    );
    let routes = s * u64::from(TOP_K);
    // A chunk of up to 16 rows runs 8-row expert tiles, a longer one tiles
    // of 32 rows or more.
    let (narrow, wide) = (rows.min(16), if rows > 16 { 32 } else { 8 });
    let maximum_tiles = k::moe_maximum_tiles(narrow, TOP_K, EXPERTS, EXPERT_TILE)
        .max(k::moe_maximum_tiles(rows, TOP_K, EXPERTS, wide));
    // The passes' matmuls read up to a tile's rows less one past the routes.
    let most_tile = if prompt { 64 } else { EXPERT_TILE };
    let matmul_rows = routes + u64::from(most_tile) - 1;
    let scores = s * u64::from(k::EXPERT_SLOTS) * 4;
    let gate = MessageGateFusedParams {
        rows: rows.min(k::GATE_FUSED_ROWS),
        hidden: HIDDEN,
        x_stride: HIDDEN,
        message_stride: HIDDEN,
        rank: GATE_RANK,
        gates: k::GATE_MAXIMUM_GATES,
        ..Default::default()
    };
    let [low, values, gate_partials, _] = k::message_gate_scratch(&gate);
    ScratchSizes {
        attention_partials: partials / 4,
        attention_statistics: statistics / 4,
        table_bytes: table,
        sums: sums.div_ceil(4),
        split_partials,
        split_counters,
        maximum_tiles: u64::from(maximum_tiles),
        grouped_input_bytes: (matmul_rows * u64::from(HIDDEN) * 2).max(scores),
        intermediate: matmul_rows * u64::from(INTERMEDIATE),
        expert_output: routes * u64::from(HIDDEN),
        gate_low: low / 2,
        gate_values: values / 4,
        gate_partials: gate_partials / 4,
    }
}

// ---------------------------------------------------------------------------
// Weights

/// A dense Q8_0 projection's planes (one "expert" of `rows` rows): plane0 at
/// its start, meta, which kernels also bind in plane1's slot.
#[derive(Clone)]
struct Planes<T> {
    plane0: T,
    meta: T,
    rows: u32,
}

/// A routed expert block's planes, its chunk table and scales.
struct Experts<T> {
    /// Gate, up and down planes, reached through the chunk table.
    planes: [T; 3],
    /// U64 `[9]` addresses (as U32 pairs): projection x {plane0, plane1, meta}.
    chunks: T,
    /// F32 `[3, 128]`: gate, up and down scales, as the expert passes apply them.
    scales: T,
    gate_format: u32,
    up_format: u32,
    down_format: u32,
}

/// A loop layer's message queries (q, k, v, FFN) and its four gates' down,
/// forget and gain weights (q, k, v, FFN), and their stored types.
struct Messages<T> {
    queries: T,
    gates: [[T; 3]; 4],
    weights: [GateWeights; 4],
}

struct Layer<T> {
    /// F32 `[queries][hidden]`; none in layer 0.
    mixer_queries: Option<T>,
    /// The q, k and v inputs' norms (a sliding layer's one norm thrice).
    norms: [T; 3],
    /// A full layer's key and value tap projections `[32][hidden]`.
    taps: Option<[T; 2]>,
    taps_f32: bool,
    packed: [Planes<T>; 3],
    sinks: T,
    sinks_f32: bool,
    key: Planes<T>,
    value: Planes<T>,
    query_scale: T,
    relative_scale: T,
    relative_bias: T,
    key_convolution: T,
    value_convolution: T,
    output: Planes<T>,
    output_convolution: T,
    ffn_query: T,
    ffn_norm: T,
    router: T,
    router_f32: bool,
    selection_bias: T,
    ffn_convolution: T,
    experts: Experts<T>,
    messages: Option<Messages<T>>,
}

struct Head<T> {
    embedding: T,
    embedding_norm: T,
    output_query: T,
    final_norm: T,
    head: Planes<T>,
}

/// A step's scratch (the reference's decode or prompt buffers and Inkling
/// scratch), every row buffer of `rows` rows: a decode step's
/// [`STORAGE_ROWS`] or a prompt chunk's whole prefill tiles.
struct Scratch<T> {
    /// Rows of every row buffer and block state plane.
    rows: u32,
    hidden: [T; 2],
    normalized: T,
    full_packed: T,
    attention_hidden: T,
    attention_output: T,
    attention_partials: T,
    attention_statistics: T,
    /// The register tile's Table16 input, or a prompt chunk's int8 codes.
    table: T,
    sums: T,
    partials: T,
    counters: T,
    selected: T,
    routing_weights: T,
    tile_descriptors: T,
    tile_count: T,
    grouped_routes: T,
    route_rows: T,
    slot_table: T,
    expert_slots: T,
    grouped_input: T,
    intermediate: T,
    expert_output: T,
    final_hidden: T,
    captured: T,
    stack: T,
    stack_planes: Vec<T>,
    previous: T,
    previous_planes: Vec<T>,
    residual: T,
    mixed: T,
    messages: T,
    key_input: T,
    value_input: T,
    keys: T,
    values: T,
    key_taps: T,
    value_taps: T,
    convolved_keys: T,
    convolved_values: T,
    gate_low: T,
    gate_values: T,
    gate_partials: T,
    gate_counters: T,
    zeros: T,
    /// A decode step's own buffers, or a prompt chunk's.
    own: Own<T>,
}

/// The buffers only a decode step or only a prompt chunk binds.
enum Own<T> {
    Decode {
        /// F32 `[16, 201024]` logits.
        logits: T,
        attention_counters: T,
        key_values: T,
        /// Every convolution's input rows, which the commit reads.
        tape: T,
    },
    Prompt {
        /// The attention's staged queries (bf16), relative states and row
        /// sinks (F32).
        queries: T,
        states: T,
        sinks: T,
        /// Each gate's gated rows of a row-kernel gate call, F32
        /// `[rows][hidden]`.
        gated: [T; 3],
    },
}

/// A step's outputs, views of the step's scratch that the next forward
/// overwrites.
pub(crate) struct StepOutputs<T> {
    /// F32 `[16, 201024]` softcapped logits; rows past the step's are garbage.
    pub logits: T,
}

pub(crate) struct Step<T> {
    calibration: Calibration,
    layers: Vec<Layer<T>>,
    head: Head<T>,
    s: Scratch<T>,
    /// A prompt chunk's scratch, allocated for the longest chunk yet.
    prompt: Option<Scratch<T>>,
}

/// Rows replacing a step's token rows after the embedding norm: bf16
/// `[n, hidden]` rows and runs of (step row, source row, rows).
pub(crate) struct Injected<'a, T> {
    pub rows: &'a T,
    pub runs: &'a [(u32, u32, u32)],
}

fn get<'a, T>(weights: &'a BTreeMap<String, T>, name: &str) -> Result<&'a T> {
    weights
        .get(name)
        .with_context(|| format!("missing Inkling step tensor {name}"))
}

/// `tensor` as F32 values (widened exactly where stored narrower), flat.
fn float<D: Device<Tensor = T>, T: TensorAccess>(ops: &TensorOps<'_, D>, tensor: &T) -> Result<T> {
    let flat = tensor.reshape(&[tensor.size() as i32])?;
    match ops.dtype(&flat) {
        Some(Dtype::F32) => Ok(flat),
        Some(Dtype::BF16 | Dtype::F16) => {
            let wide = ops.op(TensorOperation::Cast(Dtype::F32), &[&flat])?;
            ops.op(TensorOperation::Contiguous, &[&wide])
        }
        other => bail!("Inkling step tensor of type {other:?} is not a float tensor"),
    }
}

/// Whether float rows are stored F32 (else BF16); other types are refused.
fn stored_f32<D: Device<Tensor = T>, T>(
    ops: &TensorOps<'_, D>,
    tensor: &T,
    name: &str,
) -> Result<bool> {
    match ops.dtype(tensor) {
        Some(Dtype::F32) => Ok(true),
        Some(Dtype::BF16) => Ok(false),
        other => bail!("Inkling step tensor {name} is {other:?}, not F32 or BF16 rows"),
    }
}

fn dense_planes<T: TensorAccess>(
    weights: &BTreeMap<String, T>,
    weight: &str,
    n: u32,
    k: u32,
) -> Result<Planes<T>> {
    let name = planes::dense_name(weight);
    let storage = get(weights, &name)?;
    let (meta, total) = dense_plane_bytes(u64::from(n), u64::from(k));
    ensure!(
        storage.rank() == 1 && storage.size() as u64 == total,
        "Inkling planes {name} do not hold [{n}, {k}] Q8_0 weights"
    );
    Ok(Planes {
        plane0: storage.clone(),
        meta: storage.slice(0, meta as i32, None)?,
        rows: n,
    })
}

fn experts<D: Device<Tensor = T>, T: TensorAccess>(
    ops: &TensorOps<'_, D>,
    weights: &BTreeMap<String, T>,
    formats: &BTreeMap<String, Format>,
    prefix: &str,
) -> Result<Experts<T>> {
    let ggml = |name: &str| -> Result<u32> {
        let name = format!("{prefix}{name}");
        let format = formats
            .get(&name)
            .with_context(|| format!("missing Inkling format of {name}"))?;
        planes::ggml_type(format.projection)
            .with_context(|| format!("no vendored expert planes for {name}"))
    };
    let gate_up = plane_format(ggml("ffn_gate_up_exps.weight")?, HIDDEN)?;
    let down = plane_format(ggml("ffn_down_exps.weight")?, INTERMEDIATE)?;
    let shapes = [
        (gate_up, INTERMEDIATE, HIDDEN),
        (gate_up, INTERMEDIATE, HIDDEN),
        (down, HIDDEN, INTERMEDIATE),
    ];
    let mut planes = Vec::with_capacity(3);
    let mut offsets = Vec::with_capacity(9);
    for (name, (format, rows, columns)) in planes::EXPERTS.iter().zip(shapes) {
        let storage = get(weights, &format!("{prefix}{name}"))?;
        let (at, total) = plane_offsets(
            format,
            u64::from(EXPERTS),
            u64::from(rows),
            u64::from(columns),
        );
        ensure!(
            storage.size() as u64 == total,
            "Inkling expert planes {prefix}{name} do not match their format"
        );
        planes.push(storage.clone());
        offsets.extend(at);
    }
    let addressed = [0, 0, 0, 1, 1, 1, 2, 2, 2].map(|i| &planes[i]);
    let chunks = ops.required(
        Kernel::PackedCompute(P::PlaneAddresses { offsets }),
        &addressed,
    )?;
    let scales = float(
        ops,
        get(weights, &format!("{prefix}{}", planes::EXPERT_SCALES))?,
    )?;
    ensure!(
        scales.size() == 3 * EXPERTS as usize,
        "Inkling expert scales are not [3, 128]"
    );
    let [gate, up, down_planes] =
        <[T; 3]>::try_from(planes).map_err(|_| anyhow::anyhow!("Inkling expert planes"))?;
    Ok(Experts {
        planes: [gate, up, down_planes],
        chunks,
        scales,
        gate_format: gate_up.id,
        up_format: gate_up.id,
        down_format: down.id,
    })
}

fn layer<D: Device<Tensor = T>, T: TensorAccess>(
    ops: &TensorOps<'_, D>,
    weights: &BTreeMap<String, T>,
    formats: &BTreeMap<String, Format>,
    index: usize,
) -> Result<Layer<T>> {
    let p = format!("blk.{index}.");
    let w = |name: &str| get(weights, &format!("{p}{name}"));
    let values = |name: &str| float(ops, w(name)?);
    let full = sched::full_layer(index);
    let mixer_queries = (index > 0)
        .then(|| values("attn_res_query.weight"))
        .transpose()?;
    if let Some(q) = &mixer_queries {
        ensure!(
            q.size() == (sched::mixer_queries(index) * HIDDEN) as usize,
            "Inkling layer {index} mixer queries mismatch"
        );
    }
    let (norms, taps, taps_f32) = if full {
        let norms = [
            values("attn_q_inp_norm.weight")?,
            values("attn_k_inp_norm.weight")?,
            values("attn_v_inp_norm.weight")?,
        ];
        let (key, value) = (w("attn_k_conv_proj.weight")?, w("attn_v_conv_proj.weight")?);
        let key_f32 = stored_f32(ops, key, "attn_k_conv_proj")?;
        ensure!(
            stored_f32(ops, value, "attn_v_conv_proj")? == key_f32
                && key.size() == (TAPS * HIDDEN) as usize
                && value.size() == key.size(),
            "Inkling layer {index} tap projections mismatch"
        );
        (norms, Some([key.clone(), value.clone()]), key_f32)
    } else {
        let norm = values("attn_norm.weight")?;
        ([norm.clone(), norm.clone(), norm], None, true)
    };
    let dense = |name: &str, n: u32, k: u32| dense_planes(weights, &format!("{p}{name}"), n, k);
    let sinks = w("attn_sink_proj.weight")?.clone();
    let sinks_f32 = stored_f32(ops, &sinks, "attn_sink_proj")?;
    let router = w("ffn_gate_inp.weight")?.clone();
    let router_f32 = stored_f32(ops, &router, "ffn_gate_inp")?;
    let messages = if (6..18).contains(&index) {
        let queries = values(planes::MESSAGE_QUERIES)?;
        ensure!(
            queries.size() == (MESSAGES * HIDDEN) as usize,
            "Inkling layer {index} message queries mismatch"
        );
        // The gates of the q, k, v and FFN inputs, by the GGUF's names.
        let mut gates = Vec::with_capacity(4);
        let mut types = [GateWeights::F32; 4];
        for (g, input) in ["attn", "k", "v", "mlp"].into_iter().enumerate() {
            let part = |kind: &str| w(&format!("loop_{input}_{kind}.weight")).cloned();
            let triple = [part("down")?, part("forget")?, part("gain")?];
            let f32s = triple
                .iter()
                .map(|t| stored_f32(ops, t, "message gate"))
                .collect::<Result<Vec<_>>>()?;
            ensure!(
                f32s.iter().all(|&f| f == f32s[0]),
                "Inkling layer {index} gate {input} mixes stored types"
            );
            types[g] = if f32s[0] {
                GateWeights::F32
            } else {
                GateWeights::Bf16
            };
            gates.push(triple);
        }
        ensure!(
            types[0] == types[1] && types[1] == types[2],
            "Inkling layer {index}'s q, k and v gates are of different types"
        );
        let gates =
            <[[T; 3]; 4]>::try_from(gates).map_err(|_| anyhow::anyhow!("Inkling message gates"))?;
        Some(Messages {
            queries,
            gates,
            weights: types,
        })
    } else {
        None
    };
    Ok(Layer {
        mixer_queries,
        norms,
        taps,
        taps_f32,
        packed: [
            dense("attn_q.weight", sched::QUERY_COLUMNS, HIDDEN)?,
            dense("attn_gate.weight", sched::QUERY_COLUMNS, HIDDEN)?,
            dense("attn_rel.weight", sched::RELATIVE_COLUMNS, HIDDEN)?,
        ],
        sinks,
        sinks_f32,
        key: dense("attn_k.weight", KV_WIDTH, HIDDEN)?,
        value: dense("attn_v.weight", KV_WIDTH, HIDDEN)?,
        query_scale: values("attn_q_clip.weight")?,
        relative_scale: values("attn_rel_clip.weight")?,
        relative_bias: values("attn_rel_bias.weight")?,
        key_convolution: values("attn_k_conv.weight")?,
        value_convolution: values("attn_v_conv.weight")?,
        output: dense("attn_output.weight", HIDDEN, ATTENTION_WIDTH)?,
        output_convolution: values("attn_post_conv.weight")?,
        ffn_query: values("ffn_res_query.weight")?,
        ffn_norm: values("ffn_norm.weight")?,
        router,
        router_f32,
        selection_bias: values("exp_probs_b.bias")?,
        ffn_convolution: values("ffn_post_conv.weight")?,
        experts: experts(ops, weights, formats, &p)?,
        messages,
    })
}

/// One gate of a fused gate call: its message (0-3: q, k, v, FFN), its x
/// column (in hidden widths), its norm and output rows.
struct Gate<'a, T> {
    message: u32,
    column: u32,
    norm: &'a T,
    output: &'a T,
}

impl<T> Scratch<T> {
    fn rows(&self, rows: Rows) -> &T {
        match rows {
            Rows::Hidden(i) => &self.hidden[i as usize],
            Rows::Stack(p) => &self.stack_planes[p as usize],
            Rows::Previous(p) => &self.previous_planes[p as usize],
        }
    }
}

impl<T: TensorAccess> Scratch<T> {
    /// A decode step's scratch (`prompt` false, [`STORAGE_ROWS`] rows) or a
    /// prompt chunk's of `rows` rows, on the device of `calibration`.
    fn new<D: Device<Tensor = T>>(
        ops: &TensorOps<'_, D>,
        calibration: &Calibration,
        rows: u32,
        prompt: bool,
    ) -> Result<Self> {
        let sizes = scratch_sizes(calibration, rows, prompt);
        // Cleared on the device: a prompt chunk's scratch is hundreds of
        // megabytes, and only device kernels and readbacks touch it.
        let zeros = |shape: &[u64], dtype| -> Result<T> {
            ops.required(
                Kernel::DeviceZeros {
                    shape: shape.iter().map(|&d| d as i32).collect(),
                    dtype,
                },
                &[],
            )
        };
        let height = rows;
        let rows = u64::from(rows);
        let bf16 = |width: u32| zeros(&[rows, u64::from(width)], Dtype::BF16);
        let f32s = |width: u32| zeros(&[rows, u64::from(width)], Dtype::F32);
        let flat = |n: u64, dtype| zeros(&[n], dtype);
        let routes = rows * u64::from(TOP_K);
        let stack = zeros(
            &[u64::from(STACK_PLANES) * rows, u64::from(HIDDEN)],
            Dtype::BF16,
        )?;
        let previous = zeros(
            &[u64::from(PREVIOUS_PLANES) * rows, u64::from(HIDDEN)],
            Dtype::BF16,
        )?;
        let planes = |t: &T, n: u32| -> Result<Vec<T>> {
            (0..n)
                .map(|p| {
                    t.slice(
                        0,
                        (u64::from(p) * rows) as i32,
                        Some((u64::from(p + 1) * rows) as i32),
                    )
                })
                .collect()
        };
        let slots = (0..EXPERTS).flat_map(u32::to_le_bytes).collect::<Vec<_>>();
        let own = if prompt {
            let (queries, states, sinks) =
                k::staging_bytes(height.next_multiple_of(PAGE_TOKENS as u32));
            Own::Prompt {
                queries: flat(queries / 2, Dtype::BF16)?,
                states: flat(states / 4, Dtype::F32)?,
                sinks: flat(sinks / 4, Dtype::F32)?,
                gated: if height > k::GATE_FUSED_ROWS {
                    [f32s(HIDDEN)?, f32s(HIDDEN)?, f32s(HIDDEN)?]
                } else {
                    let one = || flat(1, Dtype::F32);
                    [one()?, one()?, one()?]
                },
            }
        } else {
            Own::Decode {
                logits: f32s(VOCABULARY)?,
                attention_counters: flat(k::ATTENTION_COUNTER_BYTES / 4, Dtype::U32)?,
                key_values: bf16(2 * KV_WIDTH)?,
                tape: zeros(&[rows, STATE_COLUMNS as u64], Dtype::BF16)?,
            }
        };
        Ok(Scratch {
            rows: height,
            hidden: [bf16(HIDDEN)?, bf16(HIDDEN)?],
            normalized: bf16(HIDDEN)?,
            full_packed: bf16(PACKED_WIDTH)?,
            attention_hidden: bf16(ATTENTION_WIDTH)?,
            attention_output: bf16(HIDDEN)?,
            attention_partials: flat(sizes.attention_partials, Dtype::F32)?,
            attention_statistics: flat(sizes.attention_statistics, Dtype::F32)?,
            table: flat(sizes.table_bytes, Dtype::U8)?,
            sums: flat(sizes.sums, Dtype::F32)?,
            partials: flat(sizes.split_partials, Dtype::F32)?,
            counters: flat(sizes.split_counters, Dtype::U32)?,
            selected: flat(routes, Dtype::U32)?,
            routing_weights: flat(routes, Dtype::F32)?,
            tile_descriptors: flat(
                sizes.maximum_tiles * u64::from(k::TILE_DESCRIPTOR_BYTES),
                Dtype::U8,
            )?,
            tile_count: flat(u64::from(k::TILE_COUNT_BYTES), Dtype::U8)?,
            grouped_routes: flat(routes, Dtype::U32)?,
            route_rows: flat(routes, Dtype::U32)?,
            slot_table: ops.pull(&slots, &[EXPERTS as i32], Dtype::U32)?,
            expert_slots: flat(u64::from(EXPERTS), Dtype::U32)?,
            grouped_input: flat(sizes.grouped_input_bytes, Dtype::U8)?,
            intermediate: flat(sizes.intermediate, Dtype::BF16)?,
            expert_output: flat(sizes.expert_output, Dtype::BF16)?,
            final_hidden: bf16(HIDDEN)?,
            captured: bf16(CAPTURES * HIDDEN)?,
            stack_planes: planes(&stack, STACK_PLANES)?,
            stack,
            previous_planes: planes(&previous, PREVIOUS_PLANES)?,
            previous,
            residual: f32s(HIDDEN)?,
            mixed: f32s(3 * HIDDEN)?,
            messages: f32s(MESSAGES * HIDDEN)?,
            key_input: bf16(HIDDEN)?,
            value_input: bf16(HIDDEN)?,
            keys: bf16(KV_WIDTH)?,
            values: bf16(KV_WIDTH)?,
            key_taps: bf16(TAP_STRIDE)?,
            value_taps: bf16(TAP_STRIDE)?,
            convolved_keys: bf16(KV_WIDTH)?,
            convolved_values: bf16(KV_WIDTH)?,
            gate_low: flat(sizes.gate_low, Dtype::BF16)?,
            gate_values: flat(sizes.gate_values, Dtype::F32)?,
            gate_partials: flat(sizes.gate_partials, Dtype::F32)?,
            gate_counters: flat(GATE_COUNTER_BYTES / 4, Dtype::U32)?,
            zeros: bf16(HIDDEN)?,
            own,
        })
    }

    /// A decode step's logits rows.
    fn logits(&self) -> Result<&T> {
        match &self.own {
            Own::Decode { logits, .. } => Ok(logits),
            Own::Prompt { .. } => bail!("a prompt chunk's scratch has no logits"),
        }
    }
}

impl<T: TensorAccess> Step<T> {
    /// Bind the tower's resident weights (GGUF names, with the loader's
    /// vendored planes) and allocate a decode step's scratch on the device
    /// of `calibration`.
    pub(crate) fn new<D: Device<Tensor = T>>(
        ops: &TensorOps<'_, D>,
        weights: &BTreeMap<String, T>,
        formats: &BTreeMap<String, Format>,
        calibration: Calibration,
    ) -> Result<Self> {
        let embedding = get(weights, "token_embd.weight")?.clone();
        ensure!(
            formats.get("token_embd.weight").map(|f| f.projection)
                == Some(lisa_models::packed::Projection::Q8),
            "the Inkling step gathers a Q8_0 token table"
        );
        let head = Head {
            embedding,
            embedding_norm: float(ops, get(weights, "token_embd_norm.weight")?)?,
            output_query: float(ops, get(weights, "output_res_query.weight")?)?,
            final_norm: float(ops, get(weights, "output_norm.weight")?)?,
            head: dense_planes(weights, "output.weight", VOCABULARY, HIDDEN)?,
        };
        let layers = (0..24)
            .map(|i| layer(ops, weights, formats, i))
            .collect::<Result<Vec<_>>>()?;
        let s = Scratch::new(ops, &calibration, STORAGE_ROWS, false)?;
        Ok(Self {
            calibration,
            layers,
            head,
            s,
            prompt: None,
        })
    }

    /// The draft captures of the last decode forward, bf16
    /// `[16, 4 * 2304]`: the inputs of applications 5, 23 and 41, then the
    /// final norm's output.
    pub(crate) fn captures(&self) -> &T {
        &self.s.captured
    }

    /// Encode one decode step of `rows` rows of one lane: `tokens` U32
    /// `[rows]`, `tree` the rows' `TreeRow`s (U32 `[2 rows]`, topological,
    /// each row's ancestors including itself), `lane` its `LaneRows`
    /// (U32 `[2]`: 0, rows). Row 0, the anchor, sits at `state.committed`.
    /// `injected` rows replace token rows after the embedding norm.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn forward<D: Device<Tensor = T>>(
        &self,
        ops: &TensorOps<'_, D>,
        state: &StepState<T>,
        tokens: &T,
        tree: &T,
        lane: &T,
        rows: usize,
        injected: Option<Injected<'_, T>>,
        submit: &dyn Fn() -> Result<()>,
    ) -> Result<StepOutputs<T>> {
        ensure!(
            (1..=STORAGE_ROWS as usize).contains(&rows),
            "an Inkling step takes 1 to 16 rows"
        );
        ensure!(
            state.committed + STORAGE_ROWS as usize <= state.extent_pages * PAGE_TOKENS,
            "Inkling step past its KV pages"
        );
        let r = rows as u32;
        let cap = sched::lane_cap(r);
        let committed = u32::try_from(state.committed)?;
        let selection = Selection {
            calibration: &self.calibration,
            kind: StepKind::Decode,
            int8: true,
        };
        let run = Run {
            ops,
            state,
            s: &self.s,
            lane,
            rows: r,
            committed,
            selection,
            table16: matches!(
                selection.tile(Unit::Q80, HIDDEN, ATTENTION_WIDTH, r),
                DenseTile::Register { .. }
            ),
            mode: Mode::Decode {
                tree,
                cap,
                full: attention_choice(committed, cap, false, &self.calibration),
                sliding: attention_choice(committed, cap, true, &self.calibration),
            },
        };
        self.body(&run, tokens, injected, submit)?;
        let s = &self.s;
        let logits = s.logits()?;
        self.project(
            &run.dense(),
            &s.final_hidden,
            &[(&self.head.head, 0)],
            STEP_PROJECTIONS[4],
            logits,
            Destination::F32,
            false,
        )?;
        run.dispatch(
            k::logit_finish(&LogitFinishParams {
                rows: r,
                vocabulary: VOCABULARY,
                tokens: TOKENS,
                softcap: SOFTCAP,
            }),
            &[logits],
        )?;
        Ok(StepOutputs {
            logits: logits.clone(),
        })
    }

    /// Encode a prompt chunk of `rows` rows (1 to [`PROMPT_ROWS`]) of
    /// one sequence from `state.committed`, as the reference's prefill runs
    /// it: its keys and values into their pages, the convolution state after
    /// its rows into the state's next cell, then the state advanced past
    /// them. The draft captures of rows `[captured, rows)` go to
    /// [`Self::prompt_captures`]' rows from 0; the final norm's rows to
    /// [`Self::prompt_hidden`].
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn prefill<D: Device<Tensor = T>>(
        &mut self,
        ops: &TensorOps<'_, D>,
        state: &mut StepState<T>,
        tokens: &T,
        rows: usize,
        captured: usize,
        injected: Option<Injected<'_, T>>,
        submit: &dyn Fn() -> Result<()>,
    ) -> Result<()> {
        ensure!(
            (1..=PROMPT_ROWS as usize).contains(&rows) && captured <= rows,
            "an Inkling prompt chunk takes 1 to 2048 rows"
        );
        ensure!(
            state.committed + rows <= state.extent_pages * PAGE_TOKENS,
            "Inkling prompt chunk past its KV pages"
        );
        let r = rows as u32;
        // Whole prefill tiles of rows: the tiles read and write them.
        let height = r.next_multiple_of(k::PREFILL_ROWS);
        if self.prompt.as_ref().is_none_or(|p| p.rows < height) {
            self.prompt = None;
            self.prompt = Some(Scratch::new(ops, &self.calibration, height, true)?);
        }
        let committed = u32::try_from(state.committed)?;
        let lane = ops.pull(
            &k::pack(&[k::LaneRows { first: 0, rows: r }]),
            &[2],
            Dtype::U32,
        )?;
        let selection = Selection {
            calibration: &self.calibration,
            kind: StepKind::Prefill,
            int8: true,
        };
        let run = Run {
            ops,
            state,
            s: self.prompt.as_ref().context("missing prompt scratch")?,
            lane: &lane,
            rows: r,
            committed,
            selection,
            table16: matches!(
                selection.tile(Unit::Q80, HIDDEN, ATTENTION_WIDTH, r),
                DenseTile::Register { .. }
            ),
            mode: Mode::Prompt {
                full: prompt_attention(committed, r, false, &self.calibration),
                sliding: prompt_attention(committed, r, true, &self.calibration),
                captured: (captured as u32, r - captured as u32),
            },
        };
        self.body(&run, tokens, injected, submit)?;
        state.swap();
        state.committed += rows;
        Ok(())
    }

    /// The last prompt chunk's draft captures, bf16 `[rows, 4 * 2304]`
    /// from row 0 (its captured rows).
    pub(crate) fn prompt_captures(&self) -> Result<&T> {
        Ok(&self.prompt.as_ref().context("no prompt chunk")?.captured)
    }

    /// The last prompt chunk's final norm rows, bf16 `[rows, 2304]`.
    pub(crate) fn prompt_hidden(&self) -> Result<&T> {
        Ok(&self
            .prompt
            .as_ref()
            .context("no prompt chunk")?
            .final_hidden)
    }

    /// Softcapped logits F32 `[rows, 201024]` (rows 1 to 16) of the last
    /// prompt chunk's final norm rows from `first`, through the head as the
    /// reference runs a prompt's first selection; a view of the decode
    /// scratch that the next head or step overwrites.
    pub(crate) fn prompt_logits<D: Device<Tensor = T>>(
        &self,
        ops: &TensorOps<'_, D>,
        first: u32,
        rows: u32,
    ) -> Result<T> {
        ensure!((1..=STORAGE_ROWS).contains(&rows), "a head of 1 to 16 rows");
        let hidden = self.prompt_hidden()?;
        let s = &self.s;
        let mut copy = row_copy(rows, 0);
        copy.source_row = first;
        dispatch(ops, k::copy_rows_bf16(&copy), &[hidden, &s.final_hidden])?;
        let dense = Dense {
            ops,
            s,
            rows,
            selection: Selection {
                calibration: &self.calibration,
                kind: StepKind::Prefill,
                int8: true,
            },
        };
        let logits = s.logits()?;
        self.project(
            &dense,
            &s.final_hidden,
            &[(&self.head.head, 0)],
            STEP_PROJECTIONS[4],
            logits,
            Destination::F32,
            false,
        )?;
        dispatch(
            ops,
            k::logit_finish(&LogitFinishParams {
                rows,
                vocabulary: VOCABULARY,
                tokens: TOKENS,
                softcap: SOFTCAP,
            }),
            &[logits],
        )?;
        logits.slice(0, 0, Some(rows as i32))
    }

    /// The rows' embedding (and injected rows), every application, the
    /// head's mix of every plane and the last output into the final norm's
    /// rows, and its capture.
    fn body<D: Device<Tensor = T>>(
        &self,
        run: &Run<'_, '_, D, T>,
        tokens: &T,
        injected: Option<Injected<'_, T>>,
        submit: &dyn Fn() -> Result<()>,
    ) -> Result<()> {
        let (s, r) = (run.s, run.rows);
        // The tokens' rows, then the token table's norm into stack plane 0.
        run.dispatch(
            k::gguf_embed(
                k::Q80_TOKEN,
                &GgufEmbedParams {
                    rows: r,
                    vocabulary: VOCABULARY,
                    hidden: HIDDEN,
                },
            ),
            &[tokens, &self.head.embedding, &s.hidden[0]],
        )?;
        run.dispatch(
            k::norm_rms(r, HIDDEN),
            &[&s.hidden[0], &self.head.embedding_norm, &s.stack_planes[0]],
        )?;
        if let Some(injected) = injected {
            for &(row, source, count) in injected.runs {
                ensure!(
                    count > 0 && row + count <= r,
                    "injected rows outside the step"
                );
                let mut copy = row_copy(count, 0);
                copy.source_row = source;
                copy.destination_row = row;
                run.dispatch(
                    k::copy_rows_bf16(&copy),
                    &[injected.rows, &s.stack_planes[0]],
                )?;
            }
        }
        // As the reference encodes its target while the GPU runs what came
        // before, submit the step in parts as it is encoded.
        for a in 0..APPLICATIONS {
            self.application(run, a)?;
            if a % SUBMIT_APPLICATIONS == SUBMIT_APPLICATIONS - 1 {
                submit()?;
            }
        }
        // The head's mix of every plane and the last output, and its
        // capture.
        let last = s.rows(sched::output(APPLICATIONS - 1));
        self.mix_norm(
            run,
            Some(&s.stack),
            last,
            MixCurrent::Bf16,
            &self.head.output_query,
            &[&self.head.final_norm],
            &[&s.final_hidden],
            STACK_PLANES,
        )?;
        self.capture(run, &s.final_hidden, FINAL_CAPTURE)
    }

    /// After the host chose the accepted path of the step just encoded:
    /// encode the commit of the path `retained_rows` (U32 `[1]`, preorder
    /// bits) selects: the paths' keys and values move down and the next
    /// convolution state is built. [`Self::advance`] completes it on the
    /// host once the path's length is known.
    pub(crate) fn encode_commit<D: Device<Tensor = T>>(
        &self,
        ops: &TensorOps<'_, D>,
        state: &StepState<T>,
        retained_rows: &T,
        lane: &T,
        rows: usize,
    ) -> Result<()>
    where
        T: Clone,
    {
        ensure!(
            (1..=STORAGE_ROWS as usize).contains(&rows),
            "invalid Inkling commit"
        );
        let Own::Decode { tape, .. } = &self.s.own else {
            bail!("a decode step's scratch has its tape")
        };
        let mut committed = [0; k::MAXIMUM_LANES as usize];
        committed[0] = u32::try_from(state.committed)?;
        let commit = KvPathCommitParams {
            extent_pages: u32::try_from(state.extent_pages)?,
            regions: APPLICATIONS as u32,
            committed_tokens: committed,
        };
        let tables = [&state.page_table; 8];
        let mut inputs = tables.to_vec();
        inputs.push(retained_rows);
        let d = with_extent(
            k::relative_attention_q8_commit(&commit, 1)?,
            &mut inputs,
            &state.extent,
        );
        dispatch(ops, d, &inputs)?;
        let convolution = ShortConvolutionParams {
            rows: rows as u32,
            channels: STATE_COLUMNS as u32,
            sequences: 1,
            state_stride: STATE_COLUMNS as u32,
            ..Default::default()
        };
        let mut inputs = vec![tape];
        inputs.extend([state.current(); 8]);
        inputs.extend([state.next(); 8]);
        inputs.extend([retained_rows, lane]);
        dispatch(ops, k::short_convolution_commit(&convolution)?, &inputs)?;
        Ok(())
    }
    /// The host side of a commit of `retained` rows.
    pub(crate) fn advance(&self, state: &mut StepState<T>, retained: usize) {
        state.swap();
        state.committed += retained;
    }
    /// Commit the path `retained_rows` selects, `retained` rows long.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn commit<D: Device<Tensor = T>>(
        &self,
        ops: &TensorOps<'_, D>,
        state: &mut StepState<T>,
        retained_rows: &T,
        lane: &T,
        retained: usize,
        rows: usize,
    ) -> Result<()>
    where
        T: Clone,
    {
        ensure!((1..=rows).contains(&retained), "invalid Inkling commit");
        self.encode_commit(ops, state, retained_rows, lane, rows)?;
        self.advance(state, retained);
        Ok(())
    }

    fn application<D: Device<Tensor = T>>(&self, run: &Run<'_, '_, D, T>, a: usize) -> Result<()> {
        let s = run.s;
        let layer = &self.layers[sched::weight_layer(a)];
        if sched::restarts(a) {
            // The pass before's block state after the loop's first: one copy
            // of the step's rows (the planes are contiguous).
            run.dispatch(
                k::copy_rows_bf16(&row_copy(run.rows, 0)),
                &[
                    &s.stack_planes[FIRST_PASS_PLANE as usize],
                    &s.previous_planes[0],
                ],
            )?;
        }
        let input = s.rows(sched::input(a));
        if let Some(slot) = sched::capture_slot(a) {
            self.capture(run, input, slot)?;
        }
        let inputs = self.mixer_inputs(run, a, layer, input)?;
        self.attention(run, a, layer, inputs)?;
        // The attention output's convolution plus the input rows (none at a
        // block boundary), into the fp32 residual.
        let boundary = sched::boundary(a);
        let residual = if boundary { &s.residual } else { input };
        match &run.mode {
            Mode::Decode { tree, cap, .. } => {
                let Own::Decode { tape, .. } = &s.own else {
                    bail!("a decode step's scratch has its tape")
                };
                let convolution = ShortConvolutionParams {
                    rows: run.rows,
                    channels: HIDDEN,
                    output: ConvolutionOutput::Sum,
                    residual: u32::from(!boundary),
                    tree: 1,
                    sequences: 1,
                    state_stride: STATE_COLUMNS as u32,
                    state_column: column(a, MIXER_COLUMN),
                    divisor: convolution_divisor(false),
                    input_stride: HIDDEN,
                    ..Default::default()
                };
                let state = run.state.current();
                let mut bindings = vec![&s.attention_output];
                bindings.extend([state; 16]);
                bindings.extend([
                    &layer.output_convolution,
                    &layer.output_convolution,
                    residual,
                    &s.residual,
                    run.lane,
                    *tree,
                    tape,
                ]);
                bindings.extend([&s.residual; 4]);
                run.dispatch(k::short_convolution(&convolution, *cap)?, &bindings)?;
            }
            Mode::Prompt { .. } => self.prompt_convolution(
                run,
                PromptConvolution {
                    input: &s.attention_output,
                    taps: &layer.output_convolution,
                    dynamic: None,
                    residual: (!boundary).then_some(input),
                    output: &s.residual,
                    channels: HIDDEN,
                    column: column(a, MIXER_COLUMN),
                    kind: ConvolutionOutput::Sum,
                },
            )?,
        }
        // The FFN's input: the mix of the stack and the residual, gated by
        // its message in a later pass.
        if sched::gated(a) {
            self.mix(
                run,
                Some(&s.stack),
                &s.residual,
                MixCurrent::F32,
                &layer.ffn_query,
                &s.mixed,
                sched::ffn_states(a),
                1,
            )?;
            let messages = layer.messages.as_ref().context("ungated loop layer")?;
            self.gates(
                run,
                messages,
                1,
                &[Gate {
                    message: 3,
                    column: 0,
                    norm: &layer.ffn_norm,
                    output: &s.normalized,
                }],
            )?;
        } else {
            self.mix_norm(
                run,
                Some(&s.stack),
                &s.residual,
                MixCurrent::F32,
                &layer.ffn_query,
                &[&layer.ffn_norm],
                &[&s.normalized],
                sched::ffn_states(a),
            )?;
        }
        self.experts(run, a, layer)
    }

    /// The mixer's q, k and v inputs: layer 0 normalizes its input; a
    /// sliding layer mixes one query and one norm for all three; a full
    /// layer three, each through its own norm; a later pass gates each by
    /// its message.
    fn mixer_inputs<'s, D: Device<Tensor = T>>(
        &'s self,
        run: &Run<'s, '_, D, T>,
        a: usize,
        layer: &'s Layer<T>,
        input: &'s T,
    ) -> Result<[&'s T; 3]> {
        let s = run.s;
        let queries = sched::mixer_queries(sched::weight_layer(a));
        let full = sched::full(a);
        let separate = [&s.normalized, &s.key_input, &s.value_input];
        let shared = [&s.normalized; 3];
        if sched::gated(a) {
            let messages = layer.messages.as_ref().context("ungated loop layer")?;
            self.mix(
                run,
                Some(&s.previous),
                &s.previous_planes[PREVIOUS_PLANES as usize - 1],
                MixCurrent::Bf16,
                &messages.queries,
                &s.messages,
                PREVIOUS_PLANES - 1,
                MESSAGES,
            )?;
            let mixer_queries = layer.mixer_queries.as_ref().context("layer mix queries")?;
            self.mix(
                run,
                Some(&s.stack),
                input,
                MixCurrent::Bf16,
                mixer_queries,
                &s.mixed,
                sched::mixer_states(a),
                queries,
            )?;
            let gates = (0..3)
                .map(|g| Gate {
                    message: g as u32,
                    column: if full { g as u32 } else { 0 },
                    norm: &layer.norms[g],
                    output: separate[g],
                })
                .collect::<Vec<_>>();
            self.gates(run, messages, queries, &gates)?;
            return Ok(separate);
        }
        if queries == 0 {
            run.dispatch(
                k::norm_rms(run.rows, HIDDEN),
                &[input, &layer.norms[0], &s.normalized],
            )?;
            return Ok(shared);
        }
        let mixer_queries = layer.mixer_queries.as_ref().context("layer mix queries")?;
        let n = queries as usize;
        let norms = layer.norms.each_ref();
        self.mix_norm(
            run,
            Some(&s.stack),
            input,
            MixCurrent::Bf16,
            mixer_queries,
            &norms[..n],
            &separate[..n],
            sched::mixer_states(a),
        )?;
        Ok(if full { separate } else { shared })
    }

    /// A float projection (`gguf_float[_na]`) of the run's rows of `input`
    /// through `weights` (F32 or BF16 rows of `outputs` outputs) into
    /// `output` rows `stride` wide from `offset`, on the tile the device's
    /// row selects for them.
    #[allow(clippy::too_many_arguments)]
    fn float_projection<D: Device<Tensor = T>>(
        &self,
        run: &Run<'_, '_, D, T>,
        input: &T,
        weights: &T,
        weights_f32: bool,
        output: &T,
        output_f32: bool,
        outputs: u32,
        stride: u32,
        offset: u32,
    ) -> Result<()> {
        let p = GgufFloatParams {
            rows: run.rows,
            input_size: HIDDEN,
            output_size: [outputs, 0],
            out_stride: [stride, 0],
            out_offset: [offset, 0],
        };
        let stored = if weights_f32 {
            FloatRows::F32
        } else {
            FloatRows::Bf16
        };
        let d = match float_tile(&self.calibration, run.rows, outputs, stored) {
            FloatTile::Simdgroup => k::gguf_float(weights_f32, output_f32, &p),
            FloatTile::NeuralAccelerator => k::gguf_float_na(weights_f32, output_f32, &p)?,
        };
        run.dispatch(d, &[input, weights, output, weights, output])
    }

    /// The packed q projection, the keys and values (a decode step's side by
    /// side where one input feeds both), a full layer's taps, the K/V
    /// convolutions, the relative attention and its output projection.
    fn attention<D: Device<Tensor = T>>(
        &self,
        run: &Run<'_, '_, D, T>,
        a: usize,
        layer: &Layer<T>,
        [q_input, k_input, v_input]: [&T; 3],
    ) -> Result<()> {
        let s = run.s;
        let full = sched::full(a);
        let decode = matches!(run.mode, Mode::Decode { .. });
        // The sinks' float segment first, then the fused Q8_0 segments.
        self.float_projection(
            run,
            q_input,
            &layer.sinks,
            layer.sinks_f32,
            &s.full_packed,
            false,
            SINKS,
            PACKED_WIDTH,
            SINK_COLUMN,
        )?;
        let packed = PACKED_SEGMENTS
            .iter()
            .zip(&layer.packed)
            .map(|(&(_, offset), planes)| (planes, offset))
            .collect::<Vec<_>>();
        let dense = run.dense();
        self.project(
            &dense,
            q_input,
            &packed,
            STEP_PROJECTIONS[0],
            &s.full_packed,
            Destination::Bf16,
            false,
        )?;
        // A decode step's sliding layer's ungated keys and values read one
        // input: one projection side by side.
        let side = decode && !full && !sched::gated(a);
        let (keys_in, values_in, input_stride, pair_column) = match &s.own {
            Own::Decode { key_values, .. } if side => {
                self.project(
                    &dense,
                    k_input,
                    &[(&layer.key, 0), (&layer.value, KV_WIDTH)],
                    STEP_PROJECTIONS[1],
                    key_values,
                    Destination::Bf16,
                    false,
                )?;
                (key_values, key_values, 2 * KV_WIDTH, KV_WIDTH)
            }
            _ => {
                for (input, planes, output) in [
                    (k_input, &layer.key, &s.keys),
                    (v_input, &layer.value, &s.values),
                ] {
                    self.project(
                        &dense,
                        input,
                        &[(planes, 0)],
                        STEP_PROJECTIONS[2],
                        output,
                        Destination::Bf16,
                        false,
                    )?;
                }
                (&s.keys, &s.values, KV_WIDTH, 0)
            }
        };
        let (key_dynamic, value_dynamic) = match &layer.taps {
            Some([key_taps, value_taps]) if full => {
                for (input, weights, output) in [
                    (k_input, key_taps, &s.key_taps),
                    (v_input, value_taps, &s.value_taps),
                ] {
                    self.float_projection(
                        run,
                        input,
                        weights,
                        layer.taps_f32,
                        output,
                        false,
                        TAPS,
                        TAP_STRIDE,
                        0,
                    )?;
                }
                (&s.key_taps, &s.value_taps)
            }
            _ => (&layer.key_convolution, &layer.value_convolution),
        };
        let choice = match &run.mode {
            Mode::Decode {
                tree,
                cap,
                full: f,
                sliding,
            } => {
                let Own::Decode {
                    tape,
                    attention_counters,
                    ..
                } = &s.own
                else {
                    bail!("a decode step's scratch has its tape")
                };
                let pair = ShortConvolutionParams {
                    rows: run.rows,
                    channels: KV_WIDTH,
                    heads: if full { k::KV_HEADS } else { 0 },
                    output: ConvolutionOutput::Plain,
                    tree: 1,
                    sequences: 1,
                    state_stride: STATE_COLUMNS as u32,
                    state_column: column(a, KEY_COLUMN),
                    divisor: convolution_divisor(full),
                    dynamic_stride: if full { TAP_STRIDE } else { 0 },
                    input_stride,
                    input_column: 0,
                    pair_input_column: pair_column,
                    pair_state_column: column(a, VALUE_COLUMN),
                    ..Default::default()
                };
                let mut bindings = vec![keys_in, values_in];
                bindings.extend([run.state.current(); 8]);
                bindings.extend([
                    &layer.key_convolution,
                    &layer.value_convolution,
                    key_dynamic,
                    value_dynamic,
                    &s.convolved_keys,
                    &s.convolved_values,
                    run.lane,
                    *tree,
                    tape,
                ]);
                run.dispatch(k::short_convolution_pair(&pair, *cap)?, &bindings)?;
                // The single pass: stores the rows' keys and values in their
                // slots, attends, and with a register out-projection writes
                // its Table16.
                let choice = if full { *f } else { *sliding };
                let mut params = AttentionParams {
                    sequences: 1,
                    first_sequence: 0,
                    rows: run.rows,
                    tiles: cap.div_ceil(k::ATTENTION_TILE_ROWS),
                    trees: 1,
                    last_rows: 0,
                    kv: k::kv_layer(u32::try_from(run.state.extent_pages)?, region(a) as u32)?,
                    ..Default::default()
                };
                params.sequence[0] =
                    k::attention_sequence(0, run.committed, *cap, choice.splits, 0);
                params.sequence[0].page_table_entries = u32::try_from(run.state.extent_pages)?;
                let variant = AttentionVariant {
                    int8_queries: choice.int8_queries,
                    window: !full,
                    table16: run.table16,
                };
                let (table, sums) = if run.table16 {
                    (&s.table, &s.sums)
                } else {
                    (&s.attention_hidden, &s.attention_hidden)
                };
                let mut bindings = vec![
                    &s.full_packed,
                    &s.convolved_keys,
                    &s.convolved_values,
                    &layer.query_scale,
                    &layer.relative_scale,
                    &layer.relative_bias,
                    &s.attention_partials,
                    &s.attention_statistics,
                    attention_counters,
                    &s.attention_hidden,
                    table,
                    sums,
                ];
                bindings.extend([&run.state.page_table; 8]);
                bindings.extend([*tree, run.lane]);
                let d = with_extent(
                    k::relative_attention_decode(&params, variant, *cap)?,
                    &mut bindings,
                    &run.state.extent,
                );
                run.dispatch(d, &bindings)?;
                None
            }
            Mode::Prompt {
                full: f, sliding, ..
            } => Some(if full { *f } else { *sliding }),
        };
        if let Some(choice) = choice {
            for (input, taps, dynamic, output, convolution) in [
                (
                    keys_in,
                    &layer.key_convolution,
                    key_dynamic,
                    &s.convolved_keys,
                    KEY_COLUMN,
                ),
                (
                    values_in,
                    &layer.value_convolution,
                    value_dynamic,
                    &s.convolved_values,
                    VALUE_COLUMN,
                ),
            ] {
                self.prompt_convolution(
                    run,
                    PromptConvolution {
                        input,
                        taps,
                        dynamic: full.then_some(dynamic),
                        residual: None,
                        output,
                        channels: KV_WIDTH,
                        column: column(a, convolution),
                        kind: ConvolutionOutput::Plain,
                    },
                )?;
            }
            self.prompt_relative_attention(run, a, layer, full, choice)?;
        }
        self.project(
            &dense,
            &s.attention_hidden,
            &[(&layer.output, 0)],
            STEP_PROJECTIONS[3],
            &s.attention_output,
            Destination::Bf16,
            run.table16,
        )
    }

    /// A prompt chunk's relative attention (`RelativeAttention::addProjection`
    /// and `addAttention`): the prepare stages its queries, states and sinks
    /// and stores its keys and values in their pages, then the split and
    /// reduce, or the direct kernel when the sequence takes one split.
    fn prompt_relative_attention<D: Device<Tensor = T>>(
        &self,
        run: &Run<'_, '_, D, T>,
        a: usize,
        layer: &Layer<T>,
        full: bool,
        choice: AttentionChoice,
    ) -> Result<()> {
        let s = run.s;
        let Own::Prompt {
            queries,
            states,
            sinks,
            ..
        } = &s.own
        else {
            bail!("a prompt chunk's scratch stages its attention")
        };
        let extent_pages = u32::try_from(run.state.extent_pages)?;
        let tiles = run.rows.div_ceil(k::ATTENTION_TILE_ROWS);
        let mut params = AttentionParams {
            sequences: 1,
            first_sequence: 0,
            rows: run.rows,
            tiles,
            trees: 0,
            last_rows: 0,
            kv: k::kv_layer(extent_pages, region(a) as u32)?,
            ..Default::default()
        };
        params.sequence[0] = k::prompt_sequence(run.committed, run.rows, extent_pages, 0);
        let mut bindings = vec![
            &s.full_packed,
            &s.convolved_keys,
            &s.convolved_values,
            &layer.query_scale,
            &layer.relative_scale,
            queries,
            states,
            sinks,
        ];
        bindings.extend([&run.state.page_table; 8]);
        // Without tree rows the prepare binds the projection rows there.
        bindings.extend([&s.full_packed, run.lane]);
        let d = with_extent(
            k::relative_attention_prepare(&params, full)?,
            &mut bindings,
            &run.state.extent,
        );
        run.dispatch(d, &bindings)?;
        params.sequence[0].split_count = choice.splits;
        let (table, sums) = if run.table16 {
            (&s.table, &s.sums)
        } else {
            (&s.attention_hidden, &s.attention_hidden)
        };
        if choice.splits == 1 {
            ensure!(
                !run.table16,
                "the direct relative attention writes no out-projection table"
            );
            params.tiles = run.rows.div_ceil(k::DIRECT_ROWS);
            let mut bindings = vec![queries, &s.attention_partials];
            bindings.extend([&run.state.page_table; 8]);
            bindings.extend([
                &s.attention_partials,
                run.lane,
                states,
                &layer.relative_bias,
                &s.full_packed,
                &s.attention_hidden,
                table,
                sums,
                sinks,
            ]);
            let d = with_extent(
                k::relative_attention_direct(&params, choice.int8_queries, !full)?,
                &mut bindings,
                &run.state.extent,
            );
            return run.dispatch(d, &bindings);
        }
        ensure!(
            u64::from(tiles) * u64::from(choice.splits) <= u64::from(k::SEQUENCE_SPLITS),
            "a prompt chunk's attention splits past its workspace"
        );
        let mut bindings = vec![queries, &s.attention_partials, &s.attention_statistics];
        bindings.extend([&run.state.page_table; 8]);
        bindings.extend([
            &s.attention_partials,
            run.lane,
            states,
            &layer.relative_bias,
        ]);
        let d = with_extent(
            k::relative_attention_split(&params, choice.int8_queries, !full, choice.splits)?,
            &mut bindings,
            &run.state.extent,
        );
        run.dispatch(d, &bindings)?;
        run.dispatch(
            k::relative_attention_reduce(&params, !full, run.table16)?,
            &[
                &s.attention_partials,
                &s.attention_statistics,
                &s.full_packed,
                &s.attention_hidden,
                table,
                sums,
                run.lane,
                sinks,
            ],
        )
    }

    /// A prompt chunk's short convolution (`ShortConvolution::addConvolution`
    /// of a sequence's rows): its output rows and the state after them,
    /// read from the state's current cell, into its next cell.
    fn prompt_convolution<D: Device<Tensor = T>>(
        &self,
        run: &Run<'_, '_, D, T>,
        c: PromptConvolution<'_, T>,
    ) -> Result<()> {
        let p = ShortConvolutionParams {
            rows: run.rows,
            channels: c.channels,
            heads: if c.dynamic.is_some() { k::KV_HEADS } else { 0 },
            output: c.kind,
            residual: u32::from(c.residual.is_some()),
            write_state: 1,
            tree: 0,
            sequences: 1,
            state_stride: STATE_COLUMNS as u32,
            state_column: c.column,
            divisor: convolution_divisor(c.dynamic.is_some()),
            dynamic_stride: if c.dynamic.is_some() { TAP_STRIDE } else { 0 },
            input_stride: c.channels,
            ..Default::default()
        };
        let mut bindings = vec![c.input];
        bindings.extend([run.state.current(); 8]);
        bindings.extend([run.state.next(); 8]);
        bindings.extend([
            c.taps,
            c.dynamic.unwrap_or(c.taps),
            c.residual.unwrap_or(c.output),
            c.output,
            run.lane,
            c.taps,
            c.output,
        ]);
        bindings.extend([c.output; 4]);
        // The rows, then the state's rows after them.
        run.dispatch(
            k::short_convolution(&p, run.rows + CONVOLUTION_STATE_ROWS)?,
            &bindings,
        )
    }

    /// Routing, the routed experts and the combine with the FFN output's
    /// convolution into the application's output rows: a decode step's
    /// in one dispatch along its tree, a prompt chunk's as a combine and a
    /// convolution writing the state after its rows.
    fn experts<D: Device<Tensor = T>>(
        &self,
        run: &Run<'_, '_, D, T>,
        a: usize,
        layer: &Layer<T>,
    ) -> Result<()> {
        let s = run.s;
        let e = &layer.experts;
        let r = run.rows;
        let prompt = matches!(run.mode, Mode::Prompt { .. });
        // fp32 scores live in the grouped input until the gather.
        self.float_projection(
            run,
            &s.normalized,
            &layer.router,
            layer.router_f32,
            &s.grouped_input,
            true,
            EXPERTS,
            k::EXPERT_SLOTS,
            0,
        )?;
        let route = MoeRouteParams {
            rows: r,
            input_size: HIDDEN,
            experts: EXPERTS,
            top_k: TOP_K,
        };
        run.dispatch(
            k::moe_route_select(&MoeBiasedRouteParams {
                route,
                scale: ROUTED_SCALE,
            })?,
            &[
                &s.grouped_input,
                &layer.selection_bias,
                &s.selected,
                &s.routing_weights,
            ],
        )?;
        let tile_rows = expert_tile_rows(&self.calibration, prompt, r, TOP_K, EXPERTS);
        let group = MoeGroupParams {
            rows: r,
            top_k: TOP_K,
            tile_rows,
            experts: EXPERTS,
            gate_up_columns: INTERMEDIATE / 32,
            down_columns: HIDDEN / k::TILE_COLUMNS,
            streamed: 0,
            lanes: u32::from(!prompt),
        };
        let mut grouping = vec![
            &s.selected,
            &s.tile_descriptors,
            &s.tile_count,
            &s.grouped_routes,
            &s.route_rows,
            &s.slot_table,
            &s.expert_slots,
        ];
        if prompt {
            run.dispatch(k::moe_group_routes(&group)?, &grouping)?;
        } else {
            grouping.push(run.lane);
            run.dispatch(k::moe_group_routes_verified(&group)?, &grouping)?;
        }
        let tiles = k::moe_maximum_tiles(r, TOP_K, EXPERTS, tile_rows);
        run.dispatch(
            k::moe_gather_rows(
                &MoeGatherParams {
                    input_size: HIDDEN,
                    routes_per_row: TOP_K,
                },
                tiles,
            )?,
            &[
                &s.normalized,
                &s.grouped_routes,
                &s.tile_descriptors,
                &s.tile_count,
                &s.grouped_input,
            ],
        )?;
        let gate_up = MoeGgufGateUpParams {
            input_size: HIDDEN,
            output_size: INTERMEDIATE,
            experts: EXPERTS,
            gate_routed_format: e.gate_format,
            gate_shared_format: e.gate_format,
            up_routed_format: e.up_format,
            up_shared_format: e.up_format,
            floor_slots: EXPERTS,
            chunk_slots: EXPERTS,
            tiles: k::MOE_TILES_ALL,
        };
        // Without a shared expert its plane slots bind the slot table.
        let mut bindings = vec![&s.grouped_input, &s.tile_descriptors, &s.tile_count];
        bindings.extend([&s.slot_table; 6]);
        bindings.extend([&s.intermediate, &s.slot_table, &e.scales, &e.chunks]);
        bindings.extend(e.planes.each_ref());
        run.dispatch(
            k::moe_expert_gate_up(&gate_up, tile_rows, tiles, 3)?,
            &bindings,
        )?;
        let down = MoeGgufExpertParams {
            input_size: INTERMEDIATE,
            output_size: HIDDEN,
            experts: EXPERTS,
            routed_format: e.down_format,
            shared_format: e.down_format,
            floor_slots: EXPERTS,
            chunk_slots: EXPERTS,
            tiles: k::MOE_TILES_ALL,
        };
        let mut bindings = vec![&s.intermediate, &s.tile_descriptors, &s.tile_count];
        bindings.extend([&s.slot_table; 3]);
        bindings.extend([&s.expert_output, &s.slot_table, &e.scales, &e.chunks]);
        bindings.extend(e.planes.each_ref());
        run.dispatch(k::moe_expert_down(&down, tile_rows, tiles, 3)?, &bindings)?;
        let combine = MoeCombineParams {
            rows: r,
            hidden_size: HIDDEN,
            routes_per_row: TOP_K,
        };
        let output = s.rows(sched::output(a));
        let Mode::Decode { tree, cap, .. } = &run.mode else {
            // The routes' weighted sum alone (added to zero rows), then its
            // convolution.
            run.dispatch(
                k::moe_combine(&combine)?,
                &[
                    &s.expert_output,
                    &s.route_rows,
                    &s.routing_weights,
                    &s.zeros,
                    &s.attention_output,
                ],
            )?;
            return self.prompt_convolution(
                run,
                PromptConvolution {
                    input: &s.attention_output,
                    taps: &layer.ffn_convolution,
                    dynamic: None,
                    residual: Some(&s.residual),
                    output,
                    channels: HIDDEN,
                    column: column(a, FFN_COLUMN),
                    kind: ConvolutionOutput::RoundedSum,
                },
            );
        };
        let Own::Decode { tape, .. } = &s.own else {
            bail!("a decode step's scratch has its tape")
        };
        let combine = MoeCombineConvolutionParams {
            combine,
            convolution: ShortConvolutionParams {
                rows: r,
                channels: HIDDEN,
                output: ConvolutionOutput::RoundedSum,
                residual: 1,
                tree: 1,
                sequences: 1,
                state_stride: STATE_COLUMNS as u32,
                state_column: column(a, FFN_COLUMN),
                divisor: convolution_divisor(false),
                input_stride: HIDDEN,
                ..Default::default()
            },
        };
        let mut bindings = vec![
            &s.expert_output,
            &s.route_rows,
            &s.routing_weights,
            &s.zeros,
        ];
        bindings.extend([run.state.current(); 8]);
        bindings.extend([
            &layer.ffn_convolution,
            &s.residual,
            output,
            run.lane,
            *tree,
            tape,
        ]);
        run.dispatch(k::moe_combine_convolution(&combine, *cap)?, &bindings)
    }

    /// `copy_rows_bf16` of the step's captured rows into capture slot
    /// `slot`: a decode step's every row, a prompt chunk's captured rows
    /// from capture row 0.
    fn capture<D: Device<Tensor = T>>(
        &self,
        run: &Run<'_, '_, D, T>,
        rows: &T,
        slot: u32,
    ) -> Result<()> {
        let (first, count) = match run.mode {
            Mode::Decode { .. } => (0, run.rows),
            Mode::Prompt { captured, .. } => captured,
        };
        if count == 0 {
            return Ok(());
        }
        let mut copy = row_copy(count, 0);
        copy.source_row = first;
        copy.destination_stride = CAPTURES * HIDDEN;
        copy.destination_column = slot * HIDDEN;
        run.dispatch(k::copy_rows_bf16(&copy), &[rows, &run.s.captured])
    }

    #[allow(clippy::too_many_arguments)]
    fn mix<D: Device<Tensor = T>>(
        &self,
        run: &Run<'_, '_, D, T>,
        stack: Option<&T>,
        current: &T,
        kind: MixCurrent,
        queries: &T,
        output: &T,
        states: u32,
        count: u32,
    ) -> Result<()> {
        let p = mix_params(run.rows, run.s.rows, states, count);
        let stack = if states > 0 {
            stack.unwrap_or(current)
        } else {
            current
        };
        run.dispatch(
            k::block_residual_mix(&p, kind)?,
            &[stack, current, queries, output],
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn mix_norm<D: Device<Tensor = T>>(
        &self,
        run: &Run<'_, '_, D, T>,
        stack: Option<&T>,
        current: &T,
        kind: MixCurrent,
        queries: &T,
        norms: &[&T],
        outputs: &[&T],
        states: u32,
    ) -> Result<()> {
        ensure!(
            (1..=3).contains(&norms.len()) && norms.len() == outputs.len(),
            "a mix takes one to three norms"
        );
        let p = mix_params(run.rows, run.s.rows, states, norms.len() as u32);
        let stack = if states > 0 {
            stack.unwrap_or(current)
        } else {
            current
        };
        let mut bindings = vec![stack, current, queries];
        bindings.extend((0..3).map(|q| norms[if q < norms.len() { q } else { 0 }]));
        bindings.extend((0..3).map(|q| outputs[if q < outputs.len() { q } else { 0 }]));
        run.dispatch(k::block_residual_mix_norm(&p, kind)?, &bindings)
    }

    /// One gate call over `gates` (x rows of `queries` mixes): the fused
    /// kernels up to 128 rows, the row kernel past them.
    fn gates<D: Device<Tensor = T>>(
        &self,
        run: &Run<'_, '_, D, T>,
        messages: &Messages<T>,
        queries: u32,
        gates: &[Gate<'_, T>],
    ) -> Result<()> {
        let s = run.s;
        let n = gates.len();
        ensure!((1..=3).contains(&n), "a gate call takes one to three gates");
        let weights = messages.weights[gates[0].message as usize];
        ensure!(
            gates
                .iter()
                .all(|g| messages.weights[g.message as usize] == weights),
            "a gate call's weights are of one type"
        );
        let mut p = MessageGateFusedParams {
            rows: run.rows,
            hidden: HIDDEN,
            x_stride: queries * HIDDEN,
            message_stride: MESSAGES * HIDDEN,
            rank: GATE_RANK,
            gates: n as u32,
            ..Default::default()
        };
        for (i, g) in gates.iter().enumerate() {
            p.x_columns[i] = g.column * HIDDEN;
            p.message_columns[i] = g.message * HIDDEN;
        }
        // Slots past the call's gates bind its first gate's buffers.
        let slot = |i: usize| &gates[if i < n { i } else { 0 }];
        let weight = |i: usize, part: usize| &messages.gates[slot(i).message as usize][part];
        if run.rows > k::GATE_FUSED_ROWS {
            let Own::Prompt { gated, .. } = &s.own else {
                bail!("a gate call of more than 128 rows is a prompt chunk's")
            };
            let mut bindings = vec![&s.mixed, &s.messages];
            for part in 0..3 {
                bindings.extend((0..3).map(|i| weight(i, part)));
            }
            bindings.extend((0..3).map(|i| slot(i).norm));
            bindings.extend((0..3).map(|i| &gated[if i < n { i } else { 0 }]));
            bindings.extend((0..3).map(|i| slot(i).output));
            return run.dispatch(k::message_gate_rows(&p, weights)?, &bindings);
        }
        run.dispatch(
            k::message_gate_normalize(&p)?,
            &[&s.mixed, &s.messages, &s.gate_values],
        )?;
        run.dispatch(
            k::message_gate_down(&p, weights)?,
            &[
                &s.gate_values,
                weight(0, 0),
                weight(1, 0),
                weight(2, 0),
                &s.gate_low,
            ],
        )?;
        let mut bindings = vec![&s.mixed, &s.messages];
        bindings.extend((0..3).map(|i| weight(i, 1)));
        bindings.extend((0..3).map(|i| weight(i, 2)));
        bindings.push(&s.gate_low);
        bindings.extend((0..3).map(|i| slot(i).norm));
        bindings.extend([&s.gate_values, &s.gate_partials, &s.gate_counters]);
        bindings.extend((0..3).map(|i| slot(i).output));
        run.dispatch(k::message_gate_apply(&p, weights)?, &bindings)
    }

    /// A dense Q8_0 projection of `input` (one or more segments, each its
    /// planes and first output column) into `output`, on the tile the
    /// reference selects for `shape` and the rows. `prepared` when the
    /// input's Table16 is already in the scratch (the attention wrote it for
    /// the register out-projection).
    #[allow(clippy::too_many_arguments)]
    fn project<D: Device<Tensor = T>>(
        &self,
        dense: &Dense<'_, '_, D, T>,
        input: &T,
        segments: &[(&Planes<T>, u32)],
        shape: DenseShape,
        output: &T,
        destination: Destination,
        prepared: bool,
    ) -> Result<()> {
        let (s, ops, rows) = (dense.s, dense.ops, dense.rows);
        let fused = segments.len() > 1;
        ensure!(
            !fused || destination == Destination::Bf16,
            "fused projections write bf16"
        );
        let list = segments
            .iter()
            .map(|&(planes, offset)| Segment {
                columns: planes.rows,
                format: k::Q80,
                offset,
            })
            .collect::<Vec<_>>();
        // Plane triples (plane0, plane1 = meta, meta); slots past the last
        // segment bind its planes again.
        let triples = || {
            (0..3).flat_map(|i| {
                let (planes, _) = segments[i.min(segments.len() - 1)];
                [&planes.plane0, &planes.meta, &planes.meta]
            })
        };
        let tile = dense
            .selection
            .tile(Unit::Q80, shape.selection, shape.k, rows);
        let (partials, counters) = if tile.splits() > 1 {
            (&s.partials, &s.counters)
        } else {
            (output, output)
        };
        let (single, offset) = segments[0];
        match tile {
            DenseTile::Column => {
                if fused {
                    let p = GgufColumnFusedParams::new(rows, shape.k, shape.out_stride, &list)?;
                    let mut bindings = vec![input];
                    bindings.extend(triples());
                    bindings.push(output);
                    dispatch(ops, k::gguf_column_fused(&p)?, &bindings)
                } else {
                    let p = GgufColumnParams {
                        rows,
                        input_size: shape.k,
                        out_stride: shape.out_stride,
                        out_offset: offset,
                        format: k::Q80,
                    };
                    dispatch(
                        ops,
                        k::gguf_column(&p, destination, single.rows)?,
                        &[input, &single.plane0, &single.meta, &single.meta, output],
                    )
                }
            }
            DenseTile::Register { splits } => {
                let lanes = rows.div_ceil(8);
                if !prepared {
                    dispatch(
                        ops,
                        k::decode_linear_gguf_prepare(shape.k, lanes)?,
                        &[input, &s.table, &s.sums],
                    )?;
                }
                if fused {
                    let p =
                        GgufDecodeFusedParams::new(shape.k, splits, shape.out_stride, &list, 0)?;
                    let mut bindings = vec![&s.table, &s.sums];
                    bindings.extend(triples());
                    bindings.extend([output, partials, counters]);
                    dispatch(ops, k::gguf_decode_sg_fused(lanes, &p)?, &bindings)
                } else {
                    let p = GgufDecodeParams {
                        input_size: shape.k,
                        splits,
                        out_stride: shape.out_stride,
                        out_offset: offset,
                        spread: 0,
                    };
                    dispatch(
                        ops,
                        k::gguf_decode_sg(k::Q80_TOKEN, lanes, destination, &p, single.rows)?,
                        &[
                            &s.table,
                            &s.sums,
                            &single.plane0,
                            &single.meta,
                            &single.meta,
                            output,
                            partials,
                            counters,
                            output,
                        ],
                    )
                }
            }
            DenseTile::Staged {
                rows: height,
                splits,
                spread,
            } => {
                let spread = u32::from(spread);
                if fused {
                    let p = GgufDecodeFusedParams::new(
                        shape.k,
                        splits,
                        shape.out_stride,
                        &list,
                        spread,
                    )?;
                    let mut bindings = vec![input];
                    bindings.extend(triples());
                    bindings.extend([output, partials, counters]);
                    dispatch(ops, k::gguf_decode_fused(height, &p)?, &bindings)
                } else {
                    let p = GgufDecodeParams {
                        input_size: shape.k,
                        splits,
                        out_stride: shape.out_stride,
                        out_offset: offset,
                        spread,
                    };
                    dispatch(
                        ops,
                        k::gguf_decode(k::Q80_TOKEN, height, destination, &p, single.rows)?,
                        &[
                            input,
                            &single.plane0,
                            &single.meta,
                            &single.meta,
                            output,
                            partials,
                            counters,
                            output,
                        ],
                    )
                }
            }
            DenseTile::Prefill => {
                ensure!(
                    destination == Destination::Bf16,
                    "the prefill tile writes bf16"
                );
                // Every segment, all Q8_0, in one dispatch.
                let mut p = GgufPrefillSegmentsParams {
                    input_size: shape.k,
                    rows,
                    out_stride: shape.out_stride,
                    ..Default::default()
                };
                for (i, segment) in list.iter().enumerate() {
                    p.cols[i] = segment.columns;
                    p.offset[i] = segment.offset;
                }
                let mut bindings = vec![input];
                bindings.extend(triples());
                bindings.push(output);
                dispatch(
                    ops,
                    k::gguf_prefill(k::Q80_TOKEN, &p, tile.storage_rows(rows))?,
                    &bindings,
                )
            }
            DenseTile::PrefillInt8 => {
                ensure!(
                    destination == Destination::Bf16,
                    "the int8 prefill tile writes bf16"
                );
                let storage_rows = tile.storage_rows(rows);
                dispatch(
                    ops,
                    k::gguf_prefill_int8_quantize(&GgufPrefillInt8QuantizeParams {
                        input_size: shape.k,
                        rows,
                        storage_rows,
                    })?,
                    &[input, &s.table, &s.sums],
                )?;
                for &(planes, offset) in segments {
                    let p = GgufPrefillInt8Params {
                        input_size: shape.k,
                        rows,
                        storage_rows,
                        out_stride: shape.out_stride,
                        out_offset: offset,
                        plane_input_size: shape.k,
                        column_tiles: planes.rows / k::TILE_COLUMNS,
                    };
                    dispatch(
                        ops,
                        k::gguf_prefill_int8(k::Q80_TOKEN, &p)?,
                        &[
                            &s.table,
                            &s.sums,
                            &planes.plane0,
                            &planes.meta,
                            &planes.meta,
                            output,
                            output,
                        ],
                    )?;
                }
                Ok(())
            }
        }
    }
}

/// A prompt chunk's short convolution: its input rows, static taps, a full
/// layer's dynamic taps, the residual it adds (none for a plain output),
/// its output rows, channels, state column and output kind.
struct PromptConvolution<'a, T> {
    input: &'a T,
    taps: &'a T,
    dynamic: Option<&'a T>,
    residual: Option<&'a T>,
    output: &'a T,
    channels: u32,
    column: u32,
    kind: ConvolutionOutput,
}

/// Rows a prompt chunk takes at most (`prefillTokenBudget`).
pub(crate) const PROMPT_ROWS: u32 = 2048;

/// Rows of a convolution's state (`SHORT_CONVOLUTION_STATE_ROWS`).
const CONVOLUTION_STATE_ROWS: u32 = 3;

/// A step's or a chunk's rows: a decode step's tree, lane cap and attention
/// splits, or a prompt chunk's attention plans and the rows it captures
/// (first row, rows).
enum Mode<'a, T> {
    Decode {
        tree: &'a T,
        cap: u32,
        full: AttentionChoice,
        sliding: AttentionChoice,
    },
    Prompt {
        full: AttentionChoice,
        sliding: AttentionChoice,
        captured: (u32, u32),
    },
}

/// One step's encoding context.
struct Run<'a, 'o, D: Device<Tensor = T>, T> {
    ops: &'a TensorOps<'o, D>,
    state: &'a StepState<T>,
    s: &'a Scratch<T>,
    /// The lane's `LaneRows`.
    lane: &'a T,
    rows: u32,
    committed: u32,
    selection: Selection<'a>,
    /// Whether the attention output projection runs the register tile, whose
    /// Table16 input the attention writes.
    table16: bool,
    mode: Mode<'a, T>,
}

impl<'a, 'o, D: Device<Tensor = T>, T> Run<'a, 'o, D, T> {
    fn dispatch(&self, d: VendoredDispatch, inputs: &[&T]) -> Result<()> {
        dispatch(self.ops, d, inputs)
    }
    fn dense(&self) -> Dense<'a, 'o, D, T> {
        Dense {
            ops: self.ops,
            s: self.s,
            rows: self.rows,
            selection: self.selection,
        }
    }
}

/// What a dense projection reads of its step: its rows, scratch and plans.
struct Dense<'a, 'o, D: Device<Tensor = T>, T> {
    ops: &'a TensorOps<'o, D>,
    s: &'a Scratch<T>,
    rows: u32,
    selection: Selection<'a>,
}

/// `d` with `extent` bound at an unused slot and made resident read-write:
/// the kernel reaches its pages only through page-table addresses.
fn with_extent<'a, T>(
    mut d: VendoredDispatch,
    inputs: &mut Vec<&'a T>,
    extent: &'a T,
) -> VendoredDispatch {
    d.addressed.push(inputs.len() as u32);
    d.slots.push(EXTENT_SLOT);
    inputs.push(extent);
    d
}
/// Applications encoded between submissions of a step.
const SUBMIT_APPLICATIONS: usize = 8;

/// A buffer slot no vendored kernel binds.
const EXTENT_SLOT: u32 = 30;

pub(super) fn dispatch<D: Device<Tensor = T>, T>(
    ops: &TensorOps<'_, D>,
    d: VendoredDispatch,
    inputs: &[&T],
) -> Result<()> {
    ensure!(
        inputs.len() == d.slots.len(),
        "{} binds {} buffers, given {}",
        d.entry,
        d.slots.len(),
        inputs.len()
    );
    let entry = d.entry.clone();
    ops.required(Kernel::PackedCompute(P::Vendored(Box::new(d))), inputs)
        .with_context(|| format!("Inkling step dispatch {entry}"))?;
    Ok(())
}

fn row_copy(rows: u32, column: u32) -> RowCopyParams {
    RowCopyParams {
        rows,
        width: HIDDEN,
        source_row: 0,
        source_stride: HIDDEN,
        source_column: 0,
        destination_row: 0,
        destination_stride: HIDDEN,
        destination_column: column,
    }
}

/// A mix of `rows` rows over `stack` states of planes of `plane_rows` rows.
fn mix_params(rows: u32, plane_rows: u32, stack: u32, queries: u32) -> BlockResidualMixParams {
    BlockResidualMixParams {
        rows,
        hidden: HIDDEN,
        stack,
        queries,
        stack_plane: plane_rows * HIDDEN,
        scale: (1.0 / f64::from(HIDDEN).sqrt()) as f32,
    }
}

/// State column of an application's convolution.
fn column(a: usize, convolution: usize) -> u32 {
    (a * APPLICATION_COLUMNS + convolution) as u32
}

#[cfg(test)]
#[path = "target_tests.rs"]
mod tests;
