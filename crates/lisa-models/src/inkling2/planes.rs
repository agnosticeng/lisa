//! Names and formats of the tensors the Inkling step binds: the checkpoint's
//! own, and the planes the loader repacks beside them for the pinned
//! reference's kernels.
use crate::packed::Projection;
use lisa_backend::ScalarType as D;

/// A checkpoint tensor's stored projection, element type and row bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Format {
    pub projection: Projection,
    pub dtype: D,
    pub row_bytes: i32,
}

/// Dense Q8_0 tensors whose planes the loader repacks: suffixes of each
/// layer's attention projections; the head is `output.weight`.
pub const DENSE: [&str; 6] = [
    "attn_q.weight",
    "attn_k.weight",
    "attn_v.weight",
    "attn_rel.weight",
    "attn_gate.weight",
    "attn_output.weight",
];

/// Name of a dense Q8_0 tensor's repacked planes.
pub fn dense_name(weight: &str) -> String {
    format!(
        "{}.planes",
        weight.strip_suffix(".weight").unwrap_or(weight)
    )
}

/// Suffixes of a layer's repacked expert planes: gate, up and down.
pub const EXPERTS: [&str; 3] = [
    "ffn_exps.planes_gate",
    "ffn_exps.planes_up",
    "ffn_exps.planes_down",
];

/// Suffix of a layer's F32 `[3, 128]` expert scale table as the step's
/// expert passes apply it (gate, up and down, each pass times its scale).
pub const EXPERT_SCALES: &str = "ffn_exps.planes_scales";

/// Suffix of a loop layer's F32 `[4, hidden]` message queries (q, k, v,
/// FFN), gathered by the loader from `loop_res_query` rows 2i and 2i + 1 and
/// `loop_kv_res_query` rows 2i and 2i + 1 (i = layer - 6).
pub const MESSAGE_QUERIES: &str = "loop_msg_query.planes";

/// GGML type of an expert projection format, as the repack names it.
pub fn ggml_type(projection: Projection) -> Option<u32> {
    match projection {
        Projection::Block(lisa_backend::PackedEncoding::Nvfp4) => Some(40),
        Projection::Block(lisa_backend::PackedEncoding::Q5K) => Some(13),
        Projection::Q8 => Some(8),
        _ => None,
    }
}

/// Bytes of the repacked planes of `experts x rows` rows of `columns`
/// inputs stored as `ggml_type` (the reference's quantization format sizes).
pub fn plane_bytes(ggml_type: u32, experts: usize, rows: usize, columns: usize) -> usize {
    // (plane0 + plane1 bytes per 32-input group, meta bytes, groups per meta unit)
    let (group_bytes, meta_bytes, meta_groups) = match ggml_type {
        40 if columns.is_multiple_of(256) => (16, 20, 8),
        40 => (16, 8, 2),
        13 => (20, 16, 8),
        _ => (32, 2, 1),
    };
    let groups = columns / 32;
    experts * rows * (groups * group_bytes + groups / meta_groups * meta_bytes)
}
