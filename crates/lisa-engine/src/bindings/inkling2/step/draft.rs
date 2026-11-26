//! The pinned reference's DFlash2 draft step of one lane, every dispatch
//! its vendored kernels as its host encodes them
//! ([`lisa_backend::VendoredDispatch`]).
//!
//! The draft's context is the reference's, in rings the sequence state owns
//! ([`super::state::StepState::rings`]): each layer's keys and values of the
//! last [`k::DRAFT_WINDOW`] positions (keys
//! `[head][slot][dimension]`, values `[head][dimension][slot]`, position
//! `p` at slot `p % 2048`). A [`DraftStep::forward`] embeds the anchor and
//! seven mask tokens through the target's token table and norm, runs the
//! draft layers over the rings (dynamic two-tap convolutions, q|k|v with
//! head norms and RoPE, the split attention), the final norm, the target's
//! vocabulary head and the selector's projection, then the selector's top
//! sixteen and edge tables, which the tree selection reads as they are. A
//! [`DraftStep::commit`] projects captured target rows through the context
//! projection and hidden norm and writes the keys and values of the rows a
//! path retained into the rings, as the reference's context commit does.
//!
//! The draft's projections run on the reference's af4g64 planes (its draft
//! image, see the loader), the head on the target's Q8_0 planes; tiles,
//! splits and passes follow the reference's selection
//! ([`super::dense`]). Numerics are the reference's: bf16 activations, its
//! convolutions, attention and selector; bits depend on the GPU's cores
//! through the tiles' K splits.
use super::{
    calibration::Calibration,
    dense::{DenseTile, PairTile, ROW_GROUP_ROWS, Selection as Plans, StepKind, Unit},
    kernels::{
        self as k, Destination, DraftConvolutionStage, Epilogue, GgufColumnParams,
        GgufDecodeParams, GgufEmbedParams, GgufPrefillSegmentsParams, LogitFinishParams,
        SelectorBatchParams,
    },
    schedule::{HIDDEN, TOKENS, VOCABULARY},
    target::dispatch,
};
use crate::tensor_ops::TensorOps;
use anyhow::{Context, Result, bail, ensure};
use lisa_backend::{Device, Kernel, ScalarType as Dtype, TensorAccess, TensorOperation};
use lisa_models::inkling2::{dflash as model, planes};
use std::collections::BTreeMap;

/// Storage rows of the step's row buffers: a verify block's, which a context
/// commit projects.
const STORAGE_ROWS: u32 = k::VERIFY_ROWS;
/// The draft's dynamic taps a row.
const DYNAMIC: u32 = k::draft_dynamic_width(HIDDEN);
/// The draft's MLP width.
const INTERMEDIATE: u32 = model::INTERMEDIATE as u32;
/// Target features a context row reads: four captured rows side by side.
const FEATURES: u32 = 4 * HIDDEN;
/// RoPE base of the draft's heads.
const ROPE_THETA: f32 = model::ROPE_BASE;

/// A projection's weight format.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Format {
    Af4g64,
    Q80,
}
impl Format {
    fn id(self) -> u32 {
        match self {
            Self::Af4g64 => 26,
            Self::Q80 => k::Q80,
        }
    }
    fn token(self) -> &'static str {
        match self {
            Self::Af4g64 => "af4g64",
            Self::Q80 => k::Q80_TOKEN,
        }
    }
    /// Plane0 bytes per row and 32-input group, meta bytes per row and meta
    /// unit, and the groups of a meta unit.
    fn bytes(self) -> (u64, u64, u64) {
        match self {
            Self::Af4g64 => (16, 4, 2),
            Self::Q80 => (32, 2, 1),
        }
    }
}

/// A projection's planes: plane0 at its first row, meta (bound in plane1's
/// slot too: neither format has a plane1), `rows` outputs of `inputs`.
#[derive(Clone)]
struct Planes<T> {
    plane0: T,
    meta: T,
    rows: u32,
    inputs: u32,
    format: Format,
}

impl<T: TensorAccess> Planes<T> {
    /// The planes of a whole tensor of `rows` rows stored in one allocation.
    fn whole(storage: &T, format: Format, rows: u32, inputs: u32, name: &str) -> Result<Self> {
        let [_, meta, total] = plane_offsets(format, rows, inputs, 0);
        ensure!(
            storage.rank() == 1 && storage.size() as u64 == total,
            "{name} does not hold [{rows}, {inputs}] {format:?} planes"
        );
        Ok(Self {
            plane0: storage.clone(),
            meta: storage.slice(0, meta as i32, None)?,
            rows,
            inputs,
            format,
        })
    }
    /// A view of `count` rows from `first`, whole 64-row plane tiles.
    fn rows(&self, first: u32, count: u32, storage: &T) -> Result<Self> {
        ensure!(
            first.is_multiple_of(64) && count.is_multiple_of(64) && first + count <= self.rows,
            "a view of whole plane tiles"
        );
        let [plane0, meta, _] = plane_offsets(self.format, self.rows, self.inputs, first);
        Ok(Self {
            plane0: storage.slice(0, plane0 as i32, None)?,
            meta: storage.slice(0, meta as i32, None)?,
            rows: count,
            inputs: self.inputs,
            format: self.format,
        })
    }
    fn unit(&self) -> Unit {
        match self.format {
            Format::Af4g64 => Unit::af4g64(self.inputs),
            Format::Q80 => Unit::Q80,
        }
    }
}

/// Byte offsets of row `first`'s plane0 and meta in the planes of a tensor of
/// `rows` rows and `inputs` inputs, and the planes' bytes: plane0
/// `[rows/64][groups][64][..]`, then meta `[rows/64][units][64][..]`.
fn plane_offsets(format: Format, rows: u32, inputs: u32, first: u32) -> [u64; 3] {
    let (plane0, meta, meta_groups) = format.bytes();
    let groups = u64::from(inputs / 32);
    let units = groups / meta_groups;
    let meta_at = u64::from(rows) * groups * plane0;
    let total = meta_at + u64::from(rows) * units * meta;
    let first = u64::from(first);
    [
        first * groups * plane0,
        meta_at + first * units * meta,
        total,
    ]
}

/// The draft's plans of `rows` rows: a decode step's or a commit's up to a
/// verify block's rows, a prompt chunk's past them (`draftLinear`: bf16
/// activations).
fn selection(calibration: &Calibration, rows: u32) -> Plans<'_> {
    Plans {
        calibration,
        kind: if rows > STORAGE_ROWS {
            StepKind::Prefill
        } else {
            StepKind::Decode
        },
        int8: false,
    }
}

/// The tile scratch every plan of the draft needs on the device of
/// `calibration`, in elements: Table16 bf16, sums F32, split partials F32,
/// counters U32.
pub(crate) fn tile_scratch(calibration: &Calibration) -> [u64; 4] {
    let af4 = Unit::af4g64;
    let unit_tile = |unit: Unit, n: u32, kk: u32, rows: u32| {
        selection(calibration, rows).tile(unit, n, kk, rows)
    };
    let mut need = [1u64; 4];
    let mut take = |tile: DenseTile, n: u32, kk: u32, rows: u32, columns: u64, per_counter: u32| {
        if let DenseTile::Register { .. } = tile {
            let (table, sums) = k::table16_bytes(kk, rows.div_ceil(8));
            need[0] = need[0].max(table / 2);
            need[1] = need[1].max(sums / 4);
        }
        if tile.splits() > 1 {
            need[2] = need[2]
                .max(u64::from(tile.splits()) * u64::from(tile.storage_rows(rows)) * columns);
            need[3] = need[3].max(u64::from(n / per_counter));
        }
    };
    let rows8 = k::DRAFT_ROWS;
    for (unit, n, kk) in [
        (af4(HIDDEN), DYNAMIC, HIDDEN),
        (af4(HIDDEN), k::DRAFT_QKV_WIDTH, HIDDEN),
        (
            af4(k::DRAFT_ATTENTION_WIDTH),
            HIDDEN,
            k::DRAFT_ATTENTION_WIDTH,
        ),
        (af4(INTERMEDIATE), HIDDEN, INTERMEDIATE),
        (af4(HIDDEN), k::DRAFT_RANK, HIDDEN),
        (Unit::Q80, VOCABULARY, HIDDEN),
    ] {
        take(
            unit_tile(unit, n, kk, rows8),
            n,
            kk,
            rows8,
            u64::from(n),
            64,
        );
    }
    // The context projections of a commit's rows and of a prompt chunk's
    // rows on the decode tiles.
    for rows in 1..=ROW_GROUP_ROWS {
        for (n, kk) in [(HIDDEN, FEATURES), (2 * k::DRAFT_KV_WIDTH, HIDDEN)] {
            take(
                unit_tile(af4(kk), n, kk, rows),
                n,
                kk,
                rows,
                u64::from(n),
                64,
            );
        }
    }
    let (n, kk) = (INTERMEDIATE, HIDDEN);
    match selection(calibration, rows8).pair(af4(kk), n, kk, rows8) {
        PairTile::Column => {}
        PairTile::Register { splits } => take(
            DenseTile::Register { splits },
            n,
            kk,
            rows8,
            u64::from(n),
            64,
        ),
        PairTile::Staged {
            rows,
            splits,
            one_pass,
            spread,
        } => take(
            DenseTile::Staged {
                rows,
                splits,
                spread,
            },
            n,
            kk,
            rows8,
            if one_pass {
                2 * u64::from(n)
            } else {
                u64::from(n)
            },
            if one_pass { 32 } else { 64 },
        ),
    }
    need
}

struct Layer<T> {
    input_norm: T,
    attention_convolution: T,
    attention_dynamic: Planes<T>,
    qkv: Planes<T>,
    /// The key and value rows of `qkv`, which the context commit projects.
    context_kv: Planes<T>,
    query_norm: T,
    key_norm: T,
    output: Planes<T>,
    post_norm: T,
    mlp_convolution: T,
    mlp_dynamic: Planes<T>,
    gate: Planes<T>,
    up: Planes<T>,
    down: Planes<T>,
}

struct Scratch<T> {
    hidden: [T; 2],
    normalized: T,
    dynamic: T,
    convolved: T,
    qkv: T,
    grouped: T,
    projected: T,
    residual: T,
    intermediate: T,
    gate: T,
    final_hidden: T,
    logits: T,
    selector_hidden: T,
    query_keys: T,
    query_values: T,
    rope_cos: T,
    rope_sin: T,
    context_kv: T,
    table: T,
    sums: T,
    partials: T,
    counters: T,
    partial_ids: T,
    partial_values: T,
    candidates: T,
    unary: T,
}

/// The selector's outputs of a forward, views of the step's scratch that
/// the next forward overwrites: candidates I32 `[7, 16]` (the selector
/// writes them as U32 token IDs), unary scores F32 `[7, 16]` and the partial
/// values F32 with the edge tables after the shard partials.
pub(crate) struct Selection<T> {
    pub candidates: T,
    pub unary: T,
    pub partial_values: T,
}

pub(crate) struct DraftStep<T> {
    calibration: Calibration,
    layers: Vec<Layer<T>>,
    context: Planes<T>,
    hidden_norm: T,
    final_norm: T,
    selector: Planes<T>,
    predecessor: T,
    successor: T,
    embedding: T,
    embedding_norm: T,
    head: Planes<T>,
    frequencies: T,
    s: Scratch<T>,
    /// A prompt chunk's context rows, allocated for the longest chunk yet.
    prompt: Option<PromptScratch<T>>,
}

/// A prompt chunk's context rows: projected and normalized rows, their keys
/// and values, and their RoPE tables, of `rows` rows.
struct PromptScratch<T> {
    rows: u32,
    projected: T,
    hidden: T,
    context_kv: T,
    rope_cos: T,
    rope_sin: T,
}

fn get<'a, T>(weights: &'a BTreeMap<String, T>, name: &str) -> Result<&'a T> {
    weights
        .get(name)
        .with_context(|| format!("missing DFlash2 reference tensor {name}"))
}

/// A BF16 vector of `n` values, flat.
fn bf16_vector<D: Device<Tensor = T>, T: TensorAccess>(
    ops: &TensorOps<'_, D>,
    weights: &BTreeMap<String, T>,
    name: &str,
    n: usize,
) -> Result<T> {
    let t = get(weights, name)?;
    ensure!(
        t.size() == n && ops.dtype(t) == Some(Dtype::BF16),
        "DFlash2 tensor {name} is not {n} BF16 values"
    );
    let flat = t.reshape(&[n as i32])?;
    ops.op(TensorOperation::Contiguous, &[&flat])
}

impl<T: TensorAccess> DraftStep<T> {
    /// Bind the draft's reference planes and BF16 tensors, the target's token
    /// table, embedding norm and Q8_0 head planes, allocate the rings and the
    /// scratch, on the device of `calibration`.
    pub(crate) fn new<D: Device<Tensor = T>>(
        ops: &TensorOps<'_, D>,
        draft: &BTreeMap<String, T>,
        target: &BTreeMap<String, T>,
        calibration: Calibration,
    ) -> Result<Self> {
        let planes = |name: &str, rows: i32, inputs: i32| -> Result<Planes<T>> {
            let name = format!("{name}.af4g64");
            Planes::whole(
                get(draft, &name)?,
                Format::Af4g64,
                rows as u32,
                inputs as u32,
                &name,
            )
        };
        let (h, w) = (model::WIDTH, HIDDEN as usize);
        let mut layers = Vec::with_capacity(model::LAYERS);
        for l in 0..model::LAYERS {
            let p = format!("layers.{l}.");
            let qkv_name = format!("{p}self_attn.qkv.af4g64");
            let qkv = planes(&format!("{p}self_attn.qkv"), k::DRAFT_QKV_WIDTH as i32, h)?;
            let context_kv = qkv.rows(
                k::DRAFT_ATTENTION_WIDTH,
                2 * k::DRAFT_KV_WIDTH,
                get(draft, &qkv_name)?,
            )?;
            layers.push(Layer {
                input_norm: bf16_vector(ops, draft, &format!("{p}input_layernorm.weight"), w)?,
                attention_convolution: bf16_vector(
                    ops,
                    draft,
                    &format!("{p}attention_conv.base_kernel"),
                    4 * w,
                )?,
                attention_dynamic: planes(
                    &format!("{p}attention_conv.kernel_projection"),
                    DYNAMIC as i32,
                    h,
                )?,
                qkv,
                context_kv,
                query_norm: bf16_vector(ops, draft, &format!("{p}self_attn.q_norm.weight"), 128)?,
                key_norm: bf16_vector(ops, draft, &format!("{p}self_attn.k_norm.weight"), 128)?,
                output: planes(
                    &format!("{p}self_attn.o_proj"),
                    h,
                    k::DRAFT_ATTENTION_WIDTH as i32,
                )?,
                post_norm: bf16_vector(
                    ops,
                    draft,
                    &format!("{p}post_attention_layernorm.weight"),
                    w,
                )?,
                mlp_convolution: bf16_vector(
                    ops,
                    draft,
                    &format!("{p}mlp_conv.base_kernel"),
                    4 * w,
                )?,
                mlp_dynamic: planes(&format!("{p}mlp_conv.kernel_projection"), DYNAMIC as i32, h)?,
                gate: planes(&format!("{p}mlp.gate_proj"), model::INTERMEDIATE, h)?,
                up: planes(&format!("{p}mlp.up_proj"), model::INTERMEDIATE, h)?,
                down: planes(&format!("{p}mlp.down_proj"), h, model::INTERMEDIATE)?,
            });
        }
        let codebook = |name: &str| -> Result<T> {
            let t = get(draft, name)?;
            ensure!(
                t.shape() == [model::PADDED_VOCABULARY, model::RANK]
                    && ops.dtype(t) == Some(Dtype::BF16),
                "DFlash2 codebook {name} is not BF16 [201024, 256]"
            );
            ops.op(TensorOperation::Contiguous, &[t])
        };
        let embedding_norm = {
            let t = get(target, "token_embd_norm.weight")?;
            let flat = t.reshape(&[t.size() as i32])?;
            let wide = ops.op(TensorOperation::Cast(Dtype::F32), &[&flat])?;
            ops.op(TensorOperation::Contiguous, &[&wide])?
        };
        let head_name = planes::dense_name("output.weight");
        let head = Planes::whole(
            get(target, &head_name)?,
            Format::Q80,
            VOCABULARY,
            HIDDEN,
            &head_name,
        )?;
        let frequencies = k::draft_inverse_frequencies(ROPE_THETA)
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect::<Vec<_>>();
        let frequencies = ops.pull(&frequencies, &[k::DRAFT_ROPE_PAIRS as i32], Dtype::F32)?;
        Ok(Self {
            calibration,
            layers,
            context: planes("fc", h, model::WIDTH * 4)?,
            hidden_norm: bf16_vector(ops, draft, "hidden_norm.weight", w)?,
            final_norm: bf16_vector(ops, draft, "norm.weight", w)?,
            selector: planes("candidate_selector.hidden_projection", model::RANK, h)?,
            predecessor: codebook("candidate_selector.predecessor_codebook")?,
            successor: codebook("candidate_selector.successor_codebook")?,
            embedding: get(target, "token_embd.weight")?.clone(),
            embedding_norm,
            head,
            frequencies,
            s: Self::scratch(ops, &calibration)?,
            prompt: None,
        })
    }

    fn scratch<D: Device<Tensor = T>>(
        ops: &TensorOps<'_, D>,
        calibration: &Calibration,
    ) -> Result<Scratch<T>> {
        let zeros = |shape: &[u64], dtype| -> Result<T> {
            ops.required(
                Kernel::Zeros {
                    shape: shape.iter().map(|&d| d as i32).collect(),
                    dtype,
                },
                &[],
            )
        };
        let rows = u64::from(STORAGE_ROWS);
        let bf16 = |width: u32| zeros(&[rows, u64::from(width)], Dtype::BF16);
        let flat = |n: u64, dtype| zeros(&[n], dtype);
        let [table, sums, partials, counters] = tile_scratch(calibration);
        let [ids, values, _, _] = k::draft_selector_elements(k::DRAFT_POSITIONS);
        let p = u64::from(k::DRAFT_POSITIONS);
        let c = u64::from(k::DRAFT_CANDIDATES);
        let kv = u64::from(k::DRAFT_ROWS * k::DRAFT_KV_WIDTH);
        let rope = u64::from(k::VERIFY_ROWS * k::DRAFT_ROPE_PAIRS);
        Ok(Scratch {
            hidden: [bf16(HIDDEN)?, bf16(HIDDEN)?],
            normalized: bf16(HIDDEN)?,
            dynamic: bf16(DYNAMIC)?,
            convolved: bf16(HIDDEN)?,
            qkv: bf16(k::DRAFT_QKV_WIDTH)?,
            grouped: flat(k::draft_grouped_queries_bytes(1) / 2, Dtype::BF16)?,
            projected: bf16(HIDDEN)?,
            residual: bf16(HIDDEN)?,
            intermediate: bf16(INTERMEDIATE)?,
            gate: bf16(INTERMEDIATE)?,
            final_hidden: bf16(HIDDEN)?,
            logits: zeros(&[rows, u64::from(VOCABULARY)], Dtype::F32)?,
            selector_hidden: bf16(k::DRAFT_RANK)?,
            query_keys: flat(kv, Dtype::BF16)?,
            query_values: flat(kv, Dtype::BF16)?,
            rope_cos: flat(rope, Dtype::F32)?,
            rope_sin: flat(rope, Dtype::F32)?,
            context_kv: bf16(2 * k::DRAFT_KV_WIDTH)?,
            table: flat(table, Dtype::BF16)?,
            sums: flat(sums, Dtype::F32)?,
            partials: flat(partials, Dtype::F32)?,
            counters: flat(counters, Dtype::U32)?,
            partial_ids: flat(ids, Dtype::U32)?,
            partial_values: flat(values, Dtype::F32)?,
            candidates: zeros(&[p, c], Dtype::I32)?,
            unary: zeros(&[p, c], Dtype::F32)?,
        })
    }

    /// The draft's rows of the RoPE tables: positions `start` to `start + 15`.
    fn rope<D: Device<Tensor = T>>(&self, ops: &TensorOps<'_, D>, start: u32) -> Result<()> {
        let s = &self.s;
        dispatch(
            ops,
            k::rope_draft_range(k::VERIFY_ROWS, start),
            &[&self.frequencies, &s.rope_cos, &s.rope_sin],
        )
    }

    /// A plain RMS norm of `rows` rows with bf16 `weight`, writing the
    /// register tile's table too when `table` (the next plan's input).
    fn norm<D: Device<Tensor = T>>(
        &self,
        ops: &TensorOps<'_, D>,
        input: &T,
        weight: &T,
        output: &T,
        rows: u32,
        table: bool,
    ) -> Result<bool> {
        let s = &self.s;
        if table {
            dispatch(
                ops,
                k::norm_rms_table16(rows, HIDDEN)?,
                &[input, weight, output, &s.table, &s.sums],
            )?;
        } else {
            dispatch(
                ops,
                k::norm_rms_bf16(rows, HIDDEN),
                &[input, weight, output],
            )?;
        }
        Ok(table)
    }

    /// Whether `planes`' plan of `rows` rows reads the register tile's table.
    fn registers(&self, planes: &Planes<T>, rows: u32) -> bool {
        matches!(
            selection(&self.calibration, rows).tile(
                planes.unit(),
                planes.rows,
                planes.inputs,
                rows
            ),
            DenseTile::Register { .. }
        )
    }

    /// One projection of `rows` rows of `input` through `w` into `output`
    /// (rows `w.rows` wide), as the reference's plan of its shape runs it.
    /// `prepared`: the table already holds `input`'s rows. Returns whether
    /// the table holds them after it.
    #[allow(clippy::too_many_arguments)]
    fn project<D: Device<Tensor = T>>(
        &self,
        ops: &TensorOps<'_, D>,
        input: &T,
        w: &Planes<T>,
        rows: u32,
        output: &T,
        destination: Destination,
        prepared: bool,
    ) -> Result<bool> {
        let s = &self.s;
        let kk = w.inputs;
        let tile = selection(&self.calibration, rows).tile(w.unit(), w.rows, kk, rows);
        let (partials, counters) = if tile.splits() > 1 {
            (&s.partials, &s.counters)
        } else {
            (output, output)
        };
        let token = w.format.token();
        match tile {
            DenseTile::Column => {
                let p = GgufColumnParams {
                    rows,
                    input_size: kk,
                    out_stride: w.rows,
                    out_offset: 0,
                    format: w.format.id(),
                };
                dispatch(
                    ops,
                    k::gguf_column(&p, destination, w.rows)?,
                    &[input, &w.plane0, &w.meta, &w.meta, output],
                )?;
                Ok(prepared)
            }
            DenseTile::Register { splits } => {
                let lanes = rows.div_ceil(8);
                if !prepared {
                    dispatch(
                        ops,
                        k::decode_linear_gguf_prepare(kk, lanes)?,
                        &[input, &s.table, &s.sums],
                    )?;
                }
                let p = GgufDecodeParams {
                    input_size: kk,
                    splits,
                    out_stride: w.rows,
                    out_offset: 0,
                    spread: 0,
                };
                dispatch(
                    ops,
                    k::gguf_decode_sg(token, lanes, destination, &p, w.rows)?,
                    &[
                        &s.table, &s.sums, &w.plane0, &w.meta, &w.meta, output, partials, counters,
                        output,
                    ],
                )?;
                Ok(true)
            }
            DenseTile::Staged {
                rows: height,
                splits,
                spread,
            } => {
                let p = GgufDecodeParams {
                    input_size: kk,
                    splits,
                    out_stride: w.rows,
                    out_offset: 0,
                    spread: u32::from(spread),
                };
                dispatch(
                    ops,
                    k::gguf_decode(token, height, destination, &p, w.rows)?,
                    &[
                        input, &w.plane0, &w.meta, &w.meta, output, partials, counters, output,
                    ],
                )?;
                Ok(prepared)
            }
            DenseTile::Prefill => {
                ensure!(
                    destination == Destination::Bf16,
                    "the prefill tile writes bf16"
                );
                let p = GgufPrefillSegmentsParams {
                    input_size: kk,
                    rows,
                    out_stride: w.rows,
                    cols: [w.rows, 0, 0],
                    offset: [0; 3],
                };
                let mut bindings = vec![input];
                for _ in 0..3 {
                    bindings.extend([&w.plane0, &w.meta, &w.meta]);
                }
                bindings.push(output);
                dispatch(
                    ops,
                    k::gguf_prefill(token, &p, tile.storage_rows(rows))?,
                    &bindings,
                )?;
                Ok(prepared)
            }
            DenseTile::PrefillInt8 => bail!("the draft's projections keep bf16 activations"),
        }
    }

    /// The MLP's gate/up pair of `rows` rows of `input` into
    /// `bf16(up) * silu(bf16(gate))` rows of `output`.
    fn gate_up<D: Device<Tensor = T>>(
        &self,
        ops: &TensorOps<'_, D>,
        input: &T,
        gate: &Planes<T>,
        up: &Planes<T>,
        rows: u32,
        output: &T,
    ) -> Result<()> {
        let s = &self.s;
        let (n, kk) = (up.rows, up.inputs);
        ensure!(
            gate.rows == n && gate.inputs == kk && gate.format == up.format,
            "a gate/up pair of one shape and format"
        );
        let token = up.format.token();
        let params = |splits, spread| GgufDecodeParams {
            input_size: kk,
            splits,
            out_stride: n,
            out_offset: 0,
            spread,
        };
        let split_scratch = |splits: u32| {
            if splits > 1 {
                (&s.partials, &s.counters)
            } else {
                (output, output)
            }
        };
        match selection(&self.calibration, rows).pair(up.unit(), n, kk, rows) {
            PairTile::Column => {
                let p = GgufColumnParams {
                    rows,
                    input_size: kk,
                    out_stride: n,
                    out_offset: 0,
                    format: up.format.id(),
                };
                dispatch(
                    ops,
                    k::gguf_column_gate_up(&p, n)?,
                    &[
                        input,
                        &gate.plane0,
                        &gate.meta,
                        &gate.meta,
                        output,
                        &up.plane0,
                        &up.meta,
                        &up.meta,
                    ],
                )
            }
            PairTile::Register { splits } => {
                let lanes = rows.div_ceil(8);
                dispatch(
                    ops,
                    k::decode_linear_gguf_prepare(kk, lanes)?,
                    &[input, &s.table, &s.sums],
                )?;
                let p = params(splits, 0);
                let (partials, counters) = if splits > 1 {
                    (&s.partials, &s.counters)
                } else {
                    (&s.gate, &s.gate)
                };
                dispatch(
                    ops,
                    k::gguf_decode_sg_epilogue(
                        token,
                        lanes,
                        Epilogue::Plain(Destination::Bf16),
                        &p,
                        n,
                    )?,
                    &[
                        &s.table,
                        &s.sums,
                        &gate.plane0,
                        &gate.meta,
                        &gate.meta,
                        &s.gate,
                        partials,
                        counters,
                        &s.gate,
                    ],
                )?;
                let (partials, counters) = split_scratch(splits);
                dispatch(
                    ops,
                    k::gguf_decode_sg_epilogue(token, lanes, Epilogue::UpWithGate, &p, n)?,
                    &[
                        &s.table, &s.sums, &up.plane0, &up.meta, &up.meta, output, partials,
                        counters, &s.gate,
                    ],
                )
            }
            PairTile::Staged {
                rows: height,
                splits,
                one_pass,
                spread,
            } => {
                let p = params(splits, u32::from(spread));
                let (partials, counters) = split_scratch(splits);
                if one_pass {
                    return dispatch(
                        ops,
                        k::gguf_decode_gate_up(token, height, &p, n)?,
                        &[
                            input,
                            &gate.plane0,
                            &gate.meta,
                            &gate.meta,
                            &up.plane0,
                            &up.meta,
                            &up.meta,
                            output,
                            partials,
                            counters,
                        ],
                    );
                }
                let (gate_partials, gate_counters) = if splits > 1 {
                    (&s.partials, &s.counters)
                } else {
                    (&s.gate, &s.gate)
                };
                dispatch(
                    ops,
                    k::gguf_decode_epilogue(
                        token,
                        height,
                        Epilogue::Plain(Destination::Bf16),
                        &p,
                        n,
                    )?,
                    &[
                        input,
                        &gate.plane0,
                        &gate.meta,
                        &gate.meta,
                        &s.gate,
                        gate_partials,
                        gate_counters,
                        &s.gate,
                    ],
                )?;
                dispatch(
                    ops,
                    k::gguf_decode_epilogue(token, height, Epilogue::UpWithGate, &p, n)?,
                    &[
                        input, &up.plane0, &up.meta, &up.meta, output, partials, counters, &s.gate,
                    ],
                )
            }
        }
    }

    fn convolution<D: Device<Tensor = T>>(
        ops: &TensorOps<'_, D>,
        stage: DraftConvolutionStage,
        inputs: [&T; 5],
    ) -> Result<()> {
        dispatch(ops, k::draft_conv(stage, HIDDEN)?, &inputs)
    }

    /// Draft after `anchor`, the token at position `context` (the context
    /// holds positions `0..context`): the eight block rows through the
    /// layers, the head and the selector.
    pub(crate) fn forward<D: Device<Tensor = T>>(
        &self,
        ops: &TensorOps<'_, D>,
        rings: &[[T; 2]],
        anchor: u32,
        context: usize,
    ) -> Result<Selection<T>> {
        ensure!(rings.len() == self.layers.len(), "missing DFlash2 rings");
        ensure!(context > 0, "a DFlash2 draft needs committed context");
        let s = &self.s;
        let rows = k::DRAFT_ROWS;
        let position = u32::try_from(context).context("DFlash2 position exceeds U32")?;
        self.rope(ops, position)?;
        // The anchor and the mask tokens' rows of the target's table, then its
        // norm.
        let mut tokens = vec![model::MASK_TOKEN; rows as usize];
        tokens[0] = anchor;
        let tokens = tokens
            .iter()
            .flat_map(|t| t.to_le_bytes())
            .collect::<Vec<_>>();
        let tokens = ops.pull(&tokens, &[rows as i32], Dtype::U32)?;
        dispatch(
            ops,
            k::gguf_embed(
                k::Q80_TOKEN,
                &GgufEmbedParams {
                    rows,
                    vocabulary: VOCABULARY,
                    hidden: HIDDEN,
                },
            ),
            &[&tokens, &self.embedding, &s.normalized],
        )?;
        dispatch(
            ops,
            k::norm_rms(rows, HIDDEN),
            &[&s.normalized, &self.embedding_norm, &s.hidden[0]],
        )?;
        let attention = k::draft_attention_params(position);
        for (l, w) in self.layers.iter().enumerate() {
            let (current, next) = (&s.hidden[l & 1], &s.hidden[(l + 1) & 1]);
            // Attention sublayer.
            let table = self.registers(&w.attention_dynamic, rows);
            let prepared = self.norm(ops, current, &w.input_norm, &s.normalized, rows, table)?;
            self.project(
                ops,
                &s.normalized,
                &w.attention_dynamic,
                rows,
                &s.dynamic,
                Destination::Bf16,
                prepared,
            )?;
            Self::convolution(
                ops,
                DraftConvolutionStage::Prepare,
                [
                    &s.normalized,
                    &s.dynamic,
                    &w.attention_convolution,
                    current,
                    &s.convolved,
                ],
            )?;
            self.project(
                ops,
                &s.convolved,
                &w.qkv,
                rows,
                &s.qkv,
                Destination::Bf16,
                false,
            )?;
            dispatch(
                ops,
                k::draft_attention_qkv(),
                &[
                    &s.qkv,
                    &s.grouped,
                    &w.query_norm,
                    &w.key_norm,
                    &s.rope_cos,
                    &s.rope_sin,
                    &s.query_keys,
                    &s.query_values,
                ],
            )?;
            let mut split = vec![&s.grouped];
            split.extend([&rings[l][0]; k::MAXIMUM_LANES as usize]);
            split.extend([&rings[l][1]; k::MAXIMUM_LANES as usize]);
            split.extend([&s.query_keys, &s.query_values]);
            dispatch(ops, k::draft_attention_split(&attention)?, &split)?;
            dispatch(ops, k::draft_attention_reduce(&attention), &[&s.grouped])?;
            self.project(
                ops,
                &s.grouped,
                &w.output,
                rows,
                &s.projected,
                Destination::Bf16,
                false,
            )?;
            Self::convolution(
                ops,
                DraftConvolutionStage::Residual,
                [
                    &s.projected,
                    &s.dynamic,
                    &w.attention_convolution,
                    current,
                    &s.residual,
                ],
            )?;
            // MLP sublayer.
            let table = self.registers(&w.mlp_dynamic, rows);
            let prepared = self.norm(ops, &s.residual, &w.post_norm, &s.normalized, rows, table)?;
            self.project(
                ops,
                &s.normalized,
                &w.mlp_dynamic,
                rows,
                &s.dynamic,
                Destination::Bf16,
                prepared,
            )?;
            Self::convolution(
                ops,
                DraftConvolutionStage::Prepare,
                [
                    &s.normalized,
                    &s.dynamic,
                    &w.mlp_convolution,
                    &s.residual,
                    &s.convolved,
                ],
            )?;
            self.gate_up(ops, &s.convolved, &w.gate, &w.up, rows, &s.intermediate)?;
            self.project(
                ops,
                &s.intermediate,
                &w.down,
                rows,
                &s.projected,
                Destination::Bf16,
                false,
            )?;
            Self::convolution(
                ops,
                DraftConvolutionStage::Residual,
                [
                    &s.projected,
                    &s.dynamic,
                    &w.mlp_convolution,
                    &s.residual,
                    next,
                ],
            )?;
        }
        // The final norm feeds the target's head, its rows past the valid
        // tokens left out (no softcap: it follows the selector's edges), then
        // the selector's projection.
        let last = &s.hidden[self.layers.len() & 1];
        let table = self.registers(&self.head, rows);
        let prepared = self.norm(ops, last, &self.final_norm, &s.final_hidden, rows, table)?;
        let prepared = self.project(
            ops,
            &s.final_hidden,
            &self.head,
            rows,
            &s.logits,
            Destination::F32,
            prepared,
        )?;
        dispatch(
            ops,
            k::logit_finish(&LogitFinishParams {
                rows,
                vocabulary: VOCABULARY,
                tokens: TOKENS,
                softcap: 0.0,
            }),
            &[&s.logits],
        )?;
        self.project(
            ops,
            &s.final_hidden,
            &self.selector,
            rows,
            &s.selector_hidden,
            Destination::Bf16,
            prepared,
        )?;
        let selector = SelectorBatchParams::whole(anchor, VOCABULARY);
        dispatch(
            ops,
            k::draft_select_top16(&selector)?,
            &[&s.logits, &s.partial_ids, &s.partial_values],
        )?;
        dispatch(
            ops,
            k::draft_select_edges(&selector)?,
            &[
                &s.partial_ids,
                &s.partial_values,
                &s.candidates,
                &s.unary,
                &s.selector_hidden,
                &self.predecessor,
                &self.successor,
            ],
        )?;
        Ok(Selection {
            candidates: s.candidates.clone(),
            unary: s.unary.clone(),
            partial_values: s.partial_values.clone(),
        })
    }

    /// Commit context rows: `captures` holds `rows` (1 to 16) captured target
    /// rows, bf16 `[rows, 4 * 2304]` contiguous from row 0; `retained` U32
    /// `[1]` is the mask of those a path retained and `lane` U32 `[2]` their
    /// `LaneRows` (0, rows). The path's i-th row goes to position
    /// `start + i`.
    pub(crate) fn commit<D: Device<Tensor = T>>(
        &self,
        ops: &TensorOps<'_, D>,
        rings: &[[T; 2]],
        captures: &T,
        rows: usize,
        retained: &T,
        lane: &T,
        start: usize,
    ) -> Result<()> {
        ensure!(
            (1..=STORAGE_ROWS as usize).contains(&rows),
            "a DFlash2 context commit takes 1 to 16 rows"
        );
        let s = &self.s;
        let r = rows as u32;
        let start = u32::try_from(start).context("DFlash2 position exceeds U32")?;
        self.rope(ops, start)?;
        self.project(
            ops,
            captures,
            &self.context,
            r,
            &s.projected,
            Destination::Bf16,
            false,
        )?;
        // Every layer's key and value projection reads the same normalized
        // rows, prepared for the first layer's plan.
        let table = self.registers(&self.layers[0].context_kv, r);
        let mut prepared =
            self.norm(ops, &s.projected, &self.hidden_norm, &s.hidden[0], r, table)?;
        ensure!(rings.len() == self.layers.len(), "missing DFlash2 rings");
        for (w, [keys, values]) in self.layers.iter().zip(rings) {
            prepared = self.project(
                ops,
                &s.hidden[0],
                &w.context_kv,
                r,
                &s.context_kv,
                Destination::Bf16,
                prepared,
            )?;
            let mut inputs = vec![&s.context_kv, &w.key_norm, &s.rope_cos, &s.rope_sin];
            inputs.extend([keys; k::MAXIMUM_LANES as usize]);
            inputs.extend([values; k::MAXIMUM_LANES as usize]);
            inputs.extend([retained, lane]);
            dispatch(
                ops,
                k::draft_context_kv_commit(start, k::VERIFY_ROWS)?,
                &inputs,
            )?;
        }
        Ok(())
    }
}

impl<T: TensorAccess> DraftStep<T> {
    /// The context of a prompt chunk's captured rows
    /// (`DFlashDraft::addContextPrefill`; LISA keeps no context window
    /// beside the rings): `rows` rows (1 to 2048) of the target's captures,
    /// bf16 `[rows, 4 * 2304]`, at positions from `start`, through the
    /// context projection and the hidden norm, then each layer's keys and
    /// values into its rings.
    pub(crate) fn prefill<D: Device<Tensor = T>>(
        &mut self,
        ops: &TensorOps<'_, D>,
        rings: &[[T; 2]],
        captures: &T,
        rows: usize,
        start: usize,
    ) -> Result<()> {
        ensure!(
            (1..=k::DRAFT_WINDOW as usize).contains(&rows),
            "a DFlash2 context prefill takes 1 to 2048 rows"
        );
        ensure!(rings.len() == self.layers.len(), "missing DFlash2 rings");
        let r = rows as u32;
        let start = u32::try_from(start).context("DFlash2 position exceeds U32")?;
        // Whole prefill tiles of rows: the tiles read and write them.
        let height = r.next_multiple_of(k::PREFILL_ROWS);
        if self.prompt.as_ref().is_none_or(|p| p.rows < height) {
            self.prompt = None;
            let zeros = |shape: &[u64], dtype| -> Result<T> {
                ops.required(
                    Kernel::Zeros {
                        shape: shape.iter().map(|&d| d as i32).collect(),
                        dtype,
                    },
                    &[],
                )
            };
            let h = u64::from(height);
            let rope = h * u64::from(k::DRAFT_ROPE_PAIRS);
            self.prompt = Some(PromptScratch {
                rows: height,
                projected: zeros(&[h, u64::from(HIDDEN)], Dtype::BF16)?,
                hidden: zeros(&[h, u64::from(HIDDEN)], Dtype::BF16)?,
                context_kv: zeros(&[h, u64::from(2 * k::DRAFT_KV_WIDTH)], Dtype::BF16)?,
                rope_cos: zeros(&[rope], Dtype::F32)?,
                rope_sin: zeros(&[rope], Dtype::F32)?,
            });
        }
        let p = self
            .prompt
            .as_ref()
            .context("missing DFlash2 prompt scratch")?;
        dispatch(
            ops,
            k::rope_draft_range(r, start),
            &[&self.frequencies, &p.rope_cos, &p.rope_sin],
        )?;
        self.project(
            ops,
            captures,
            &self.context,
            r,
            &p.projected,
            Destination::Bf16,
            false,
        )?;
        dispatch(
            ops,
            k::norm_rms_bf16(r, HIDDEN),
            &[&p.projected, &self.hidden_norm, &p.hidden],
        )?;
        for (w, [keys, values]) in self.layers.iter().zip(rings) {
            self.project(
                ops,
                &p.hidden,
                &w.context_kv,
                r,
                &p.context_kv,
                Destination::Bf16,
                false,
            )?;
            dispatch(
                ops,
                k::prefill_draft_context_kv(r, start),
                &[
                    &p.context_kv,
                    &w.key_norm,
                    &p.rope_cos,
                    &p.rope_sin,
                    keys,
                    values,
                ],
            )?;
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "draft_tests.rs"]
mod tests;
