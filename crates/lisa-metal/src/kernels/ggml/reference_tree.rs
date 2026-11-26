//! Draft tree selection of the vendored reference
//! (`shaders/reference/sampling.metal`, revision and hashes in its
//! `provenance.json`) for one greedy lane, on the device: the best-first
//! tree over a draft's candidate lattice and a host lookup chain
//! (`draft_select_tree`), then the verify block's rows, the tree's best
//! `rows` by estimate in preorder (`draft_verify_rows`). Glue in
//! `shaders/gguf/reference_tree_glue.metal` sorts the lattice into the order
//! the reference's selector writes and converts the rows to LISA's tree
//! tensor. Nothing is read back.
use super::output_full_write;
use crate::{
    Array, Dtype, Stream,
    error::{Error, Result},
    runtime::Math,
};

const SAMPLING_SOURCE: &crate::shader_pack::Source =
    crate::kernel_source!("reference/sampling.metal");
const GLUE_SOURCE: &crate::shader_pack::Source =
    crate::kernel_source!("gguf/reference_tree_glue.metal");

/// `DRAFT_PROPOSAL_TOKENS`.
pub const TREE_POSITIONS: usize = 7;
/// `DRAFT_CANDIDATES`.
pub const TREE_CANDIDATES: usize = 16;
/// `TARGET_VERIFY_ROWS`: a lane's most rows.
pub const TREE_ROWS: usize = 16;
/// `TREE_CHAIN_TOKENS`.
pub const TREE_CHAIN_TOKENS: usize = TREE_ROWS - 1;
/// `TREE_AGREEMENTS`.
pub const TREE_AGREEMENTS: usize = 4;
/// `TREE_CHILDREN`.
pub const TREE_CHILDREN: usize = 4;
const LANES: usize = 8;
const SHARDS: usize = 8;

/// One lane's tree and row layout parameters (`DraftTreeParams` and
/// `VerifyRowsParams` of a greedy lane).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TreeParams {
    pub anchor: u32,
    /// Deepest proposal a walk may retain (tokens left past the anchor).
    pub depth: u32,
    /// Tokens of the host chain (0 for none).
    pub chain: u32,
    /// Children a draft row takes (at most [`TREE_CHILDREN`]).
    pub children: u32,
    /// The draft head's softcap (0 for none).
    pub softcap: f32,
    /// Acceptance model: `sigmoid(slope * logit(q) + intercepts[p])`.
    pub slope: f32,
    pub intercepts: [f32; TREE_POSITIONS],
    /// Verify rows of the step, the anchor's included.
    pub rows: u32,
    /// Most rows of the lane.
    pub lane_cap: u32,
    /// Lowest estimate a row is verified at.
    pub floor: f32,
    /// Whether the lattice is the reference selector's own output
    /// (candidates, unary scores, partial values with the edge tables),
    /// already in its order: no glue sorts it.
    pub selected: bool,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct DraftTreeParams {
    anchor: [u32; LANES],
    depth: [u32; LANES],
    chain: [u32; LANES],
    children: [u32; LANES],
    lanes: u32,
    softcap: f32,
    slope: [f32; LANES],
    intercept: [[f32; TREE_POSITIONS]; LANES],
    temperature: [f32; LANES],
    noise: [[u32; 2]; LANES],
}
const _: () = assert!(std::mem::size_of::<DraftTreeParams>() == 8 + (32 + 4 * 7) * 8);

#[repr(C)]
#[derive(Clone, Copy)]
struct VerifyRowsParams {
    lanes: u32,
    rows: u32,
    lane_cap: u32,
    floor: f32,
}

/// The tree and verify rows of one lane.
pub struct DraftTreeOutputs {
    /// Verify tokens `I32 [rows]`.
    pub ids: Array,
    /// LISA tree tensor `I32 [2, rows]`: parents (-1 for the anchor and for
    /// rows of no lane), then depths.
    pub tree: Array,
    /// The verify rows' tree rows (`U32 [rows]`, `u32::MAX` for no row).
    pub sources: Array,
    /// The whole tree in the order it grew: tokens `U32 [16]`, rows
    /// `U32 [16, 2]` (ancestors; parent | depth << 16), scores
    /// `F32 [16, 2]` (draft probability, estimate), caps `F32 [16]` and the
    /// row count `U32 [1]`.
    pub tokens: Array,
    pub rows: Array,
    pub scores: Array,
    pub caps: Array,
    pub size: Array,
    /// The lattice's candidates in the reference's order `U32 [7, 16]`.
    pub candidates: Array,
    /// The verify rows as the reference's tree rows (`TreeRow`,
    /// `U32 [rows, 2]`) and the lane's rows (`LaneRows`, `U32 [2]`).
    pub verify_tree: Array,
    pub lane_rows: Array,
}

/// Grow one greedy lane's draft tree and lay out its verify rows.
/// `ids I32 [7, 16]`, `logits F32 [7, 16]` and `edges F32 [7, 16, 16]` are
/// the draft's lattice in any candidate order, or with `params.selected` the
/// reference selector's candidates (I32 `[7, 16]`), unary scores and partial
/// values (F32 `[7 * 8 * 16 + 7 * 16 * 16]`); `chain I32 [15]` the host
/// chain's tokens (the first `params.chain` read) and `rates F32 [15, 4]`
/// their estimates by agreement with the draft.
pub fn draft_tree(
    ids: &Array,
    logits: &Array,
    edges: &Array,
    chain: &Array,
    rates: &Array,
    params: &TreeParams,
    stream: &Stream,
) -> Result<DraftTreeOutputs> {
    let (p, k) = (TREE_POSITIONS as i32, TREE_CANDIDATES as i32);
    let contiguous = |x: &Array, dtype: Dtype, shape: &[i32]| -> Result<()> {
        if x.dtype() != dtype || x.shape() != shape || !x.t.buffer_and_layout().1.is_contiguous() {
            return Err(Error::msg(format!(
                "draft tree input expects contiguous {dtype:?} {shape:?}, got {:?} {:?}",
                x.dtype(),
                x.shape()
            )));
        }
        x.eval_enqueue()
    };
    contiguous(ids, Dtype::Int32, &[p, k])?;
    contiguous(logits, Dtype::Float32, &[p, k])?;
    let partials = (TREE_POSITIONS * SHARDS * TREE_CANDIDATES) as i32 + p * k * k;
    if params.selected {
        contiguous(edges, Dtype::Float32, &[partials])?;
    } else {
        contiguous(edges, Dtype::Float32, &[p, k, k])?;
    }
    contiguous(chain, Dtype::Int32, &[TREE_CHAIN_TOKENS as i32])?;
    contiguous(
        rates,
        Dtype::Float32,
        &[TREE_CHAIN_TOKENS as i32, TREE_AGREEMENTS as i32],
    )?;
    let rows = params.rows as usize;
    if !(1..=TREE_ROWS).contains(&rows)
        || params.lane_cap as usize > TREE_ROWS
        || params.lane_cap < params.rows
        || params.children as usize > TREE_CHILDREN
        || params.chain as usize > TREE_CHAIN_TOKENS
        || !params.floor.is_finite()
    {
        return Err(Error::msg("invalid draft tree parameters"));
    }
    let runtime = stream.runtime();
    let inputs_pipeline =
        runtime.compile_with(GLUE_SOURCE, "lisa_tree_inputs", Math::CompilerDefault)?;
    let tree_pipeline =
        runtime.compile_with(SAMPLING_SOURCE, "draft_select_tree", Math::CompilerDefault)?;
    let rows_pipeline =
        runtime.compile_with(SAMPLING_SOURCE, "draft_verify_rows", Math::CompilerDefault)?;
    let out_pipeline =
        runtime.compile_with(GLUE_SOURCE, "lisa_tree_rows", Math::CompilerDefault)?;

    let full = |shape: &[i32], dtype| output_full_write(stream, shape, dtype);
    let (candidates, unary, partial_values) = if params.selected {
        (ids.clone(), logits.clone(), edges.clone())
    } else {
        // Only the edge table after the shard partials is read.
        (
            full(&[p, k], Dtype::Uint32)?,
            full(&[p, k], Dtype::Float32)?,
            full(&[partials], Dtype::Float32)?,
        )
    };
    let n = TREE_ROWS as i32;
    // The tree kernel writes its rows' entries up to the count it made; the
    // rest are never read.
    let tokens = full(&[n], Dtype::Uint32)?;
    let tree_rows = full(&[n, 2], Dtype::Uint32)?;
    let scores = full(&[n, 2], Dtype::Float32)?;
    let size = full(&[1], Dtype::Uint32)?;
    let draws = full(&[n * TREE_CHILDREN as i32], Dtype::Uint8)?;
    let next = full(&[n, k], Dtype::Float32)?;
    let caps = full(&[n], Dtype::Float32)?;
    let r = rows as i32;
    let verify_tokens = full(&[r], Dtype::Uint32)?;
    let verify_tree = full(&[r, 2], Dtype::Uint32)?;
    let lane_rows = full(&[2], Dtype::Uint32)?;
    let sources = full(&[r], Dtype::Uint32)?;
    let out_ids = full(&[r], Dtype::Int32)?;
    let out_tree = full(&[2, r], Dtype::Int32)?;

    let mut tree_params = DraftTreeParams {
        anchor: [0; LANES],
        depth: [0; LANES],
        chain: [0; LANES],
        children: [0; LANES],
        lanes: 1,
        softcap: params.softcap,
        slope: [0.; LANES],
        intercept: [[0.; TREE_POSITIONS]; LANES],
        temperature: [0.; LANES],
        noise: [[0; 2]; LANES],
    };
    tree_params.anchor[0] = params.anchor;
    tree_params.depth[0] = params.depth;
    tree_params.chain[0] = params.chain;
    tree_params.children[0] = params.children;
    tree_params.slope[0] = params.slope;
    tree_params.intercept[0] = params.intercepts;
    let rows_params = VerifyRowsParams {
        lanes: 1,
        rows: params.rows,
        lane_cap: params.lane_cap,
        floor: params.floor,
    };

    let bind = |a: &Array| {
        let (b, l) = a.t.buffer_and_layout();
        (b.clone(), l.offset * a.dtype().size_of())
    };
    let guard = runtime.commands.encoder()?;
    let enc = guard.encoder();
    if !params.selected {
        enc.set_pipeline(&inputs_pipeline);
        for (slot, a) in [ids, logits, edges].into_iter().enumerate() {
            let (b, at) = bind(a);
            enc.set_input(slot, Some(&b), at);
        }
        for (slot, a) in [&candidates, &unary, &partial_values]
            .into_iter()
            .enumerate()
        {
            let (b, at) = bind(a);
            enc.set_output(3 + slot, Some(&b), at);
        }
        enc.dispatch_groups((1, 1, 1), (TREE_POSITIONS * TREE_CANDIDATES, 1, 1));
    }

    enc.set_pipeline(&tree_pipeline);
    for (slot, a) in [&candidates, &unary, &partial_values, chain, rates]
        .into_iter()
        .enumerate()
    {
        let (b, at) = bind(a);
        enc.set_input(slot, Some(&b), at);
    }
    for (slot, a) in [&tokens, &tree_rows, &scores, &size, &draws, &next, &caps]
        .into_iter()
        .enumerate()
    {
        let (b, at) = bind(a);
        enc.set_output(5 + slot, Some(&b), at);
    }
    enc.set_bytes(12, &tree_params);
    // Its first phase takes a thread per (position, predecessor).
    enc.dispatch_groups((1, 1, 1), (128, 1, 1));

    enc.set_pipeline(&rows_pipeline);
    for (slot, a) in [&tokens, &tree_rows, &scores, &size]
        .into_iter()
        .enumerate()
    {
        let (b, at) = bind(a);
        enc.set_input(slot, Some(&b), at);
    }
    for (slot, a) in [&verify_tokens, &verify_tree, &lane_rows, &sources]
        .into_iter()
        .enumerate()
    {
        let (b, at) = bind(a);
        enc.set_output(4 + slot, Some(&b), at);
    }
    enc.set_bytes(8, &rows_params);
    enc.dispatch_groups((1, 1, 1), (1, 1, 1));

    enc.set_pipeline(&out_pipeline);
    for (slot, a) in [&verify_tokens, &verify_tree].into_iter().enumerate() {
        let (b, at) = bind(a);
        enc.set_input(slot, Some(&b), at);
    }
    for (slot, a) in [&out_ids, &out_tree].into_iter().enumerate() {
        let (b, at) = bind(a);
        enc.set_output(2 + slot, Some(&b), at);
    }
    enc.set_bytes(4, &params.rows);
    enc.dispatch_groups((1, 1, 1), (TREE_ROWS, 1, 1));
    drop(guard);
    Ok(DraftTreeOutputs {
        ids: out_ids,
        tree: out_tree,
        sources,
        tokens,
        rows: tree_rows,
        scores,
        caps,
        size,
        candidates,
        verify_tree,
        lane_rows,
    })
}

#[cfg(test)]
#[path = "reference_tree_tests.rs"]
mod tests;
