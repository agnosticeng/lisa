//! Load-time repack of GGUF and MLX affine weights into the vendored
//! reference's planes (`shaders/reference/`, revision and hashes in its
//! `provenance.json`).
//!
//! Preparation rebuilds each tensor into the reference loader's native rows
//! and runs its `gguf_repack` kernel, which writes the column-interleaved
//! planes `plane0 [N/64][K/32][64][..]`, `plane1` and
//! `meta [N/64][units][64][..]` of every expert, back to back. NVFP4 rows
//! whose width is a multiple of 256 become 256-element native blocks
//! (16 E4M3 scales, an F32 `g`, 128 code bytes), other widths 64-element
//! blocks; `g` is 1 because the per-expert `.scale` stays in a separate
//! table. Q5_K, Q8_0 and MLX affine 4-bit (af4g64) rows are repacked as
//! stored.
use super::output_full_write;
use crate::{
    Array, Dtype, Stream,
    error::{Error, Result},
    runtime::Math,
};

const REPACK_SOURCE: &crate::shader_pack::Source =
    crate::kernel_source!("reference/gguf_repack.metal");

/// GGUF tensor types the preparation accepts.
pub const GGML_TYPE_Q8_0: u32 = 8;
pub const GGML_TYPE_Q5_K: u32 = 13;
pub const GGML_TYPE_NVFP4: u32 = 40;
/// The reference loader's MLX affine 4-bit, 64-element native block
/// (`QUANT_AFFINE_TYPE(4, 64)`: bf16 scale, bf16 bias, 32 code bytes).
pub const GGML_TYPE_AF4G64: u32 = 0x4D4C_0000 | (4 << 8) | 64;

/// `QUANT_TILE_ROWS`: rows of one plane tile.
const TILE_ROWS: usize = 64;
/// Rows the repack stages per dispatch, bounding host staging memory.
const STAGING_BYTES: usize = 32 << 20;

/// One `kQuantFormats` entry (the reference's `abi/QuantFormat.h`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PlaneFormat {
    /// `GGUF_FMT_*` id, the kernels' run-time format.
    pub id: u32,
    pub block_elements: usize,
    pub block_bytes: usize,
    /// Bytes per row and 32-element group.
    pub plane0_bytes: usize,
    pub plane1_bytes: usize,
    /// Bytes per row and meta unit (one native block).
    pub meta_bytes: usize,
    pub meta_groups: usize,
}

pub const PLANE_Q5K: PlaneFormat = PlaneFormat {
    id: 3,
    block_elements: 256,
    block_bytes: 176,
    plane0_bytes: 16,
    plane1_bytes: 4,
    meta_bytes: 16,
    meta_groups: 8,
};
pub const PLANE_Q80: PlaneFormat = PlaneFormat {
    id: 6,
    block_elements: 32,
    block_bytes: 34,
    plane0_bytes: 32,
    plane1_bytes: 0,
    meta_bytes: 2,
    meta_groups: 1,
};
pub const PLANE_NVFP4: PlaneFormat = PlaneFormat {
    id: 37,
    block_elements: 256,
    block_bytes: 148,
    plane0_bytes: 16,
    plane1_bytes: 0,
    meta_bytes: 20,
    meta_groups: 8,
};
pub const PLANE_AF4G64: PlaneFormat = PlaneFormat {
    id: 26,
    block_elements: 64,
    block_bytes: 36,
    plane0_bytes: 16,
    plane1_bytes: 0,
    meta_bytes: 4,
    meta_groups: 2,
};
pub const PLANE_NVFP464: PlaneFormat = PlaneFormat {
    id: 44,
    block_elements: 64,
    block_bytes: 40,
    plane0_bytes: 16,
    plane1_bytes: 0,
    meta_bytes: 8,
    meta_groups: 2,
};

/// A GGUF expert tensor `[experts, rows, row_bytes]` as stored.
#[derive(Clone, Copy)]
pub struct GgufExpertTensor<'a> {
    pub bytes: &'a [u8],
    pub ggml_type: u32,
    pub experts: usize,
    /// Rows of one expert.
    pub rows: usize,
    /// Inputs of one row (K).
    pub columns: usize,
}

impl GgufExpertTensor<'_> {
    /// Stored bytes of one row.
    pub fn row_bytes(&self) -> Result<usize> {
        let (elements, bytes) = match self.ggml_type {
            GGML_TYPE_NVFP4 => (64, 36),
            GGML_TYPE_Q5_K => (256, 176),
            GGML_TYPE_Q8_0 => (32, 34),
            GGML_TYPE_AF4G64 => (64, 36),
            other => {
                return Err(Error::msg(format!(
                    "unsupported expert tensor type {other}"
                )));
            }
        };
        if self.columns == 0 || !self.columns.is_multiple_of(elements) {
            return Err(Error::msg("expert row width is not whole blocks"));
        }
        Ok(self.columns / elements * bytes)
    }

    /// The plane format the reference loader stores this tensor in.
    pub fn plane_format(&self) -> Result<PlaneFormat> {
        Ok(match self.ggml_type {
            GGML_TYPE_NVFP4 if self.columns.is_multiple_of(256) => PLANE_NVFP4,
            GGML_TYPE_NVFP4 => PLANE_NVFP464,
            GGML_TYPE_Q5_K => PLANE_Q5K,
            GGML_TYPE_Q8_0 => PLANE_Q80,
            GGML_TYPE_AF4G64 => PLANE_AF4G64,
            other => {
                return Err(Error::msg(format!(
                    "unsupported expert tensor type {other}"
                )));
            }
        })
    }
}

/// The plane format the reference loader stores a GGML expert type with
/// `columns` inputs per row in.
pub fn plane_format_for(ggml_type: u32, columns: usize) -> Result<PlaneFormat> {
    GgufExpertTensor {
        bytes: &[],
        ggml_type,
        experts: 0,
        rows: 0,
        columns,
    }
    .plane_format()
}

/// Byte offsets of plane1 and meta and the total size of `experts x rows`
/// rows of `columns` inputs in `format`.
pub fn plane_layout(
    format: PlaneFormat,
    experts: usize,
    rows: usize,
    columns: usize,
) -> (usize, usize, usize) {
    let groups = columns / 32;
    let total_rows = experts * rows;
    let plane1 = total_rows * groups * format.plane0_bytes;
    let meta = plane1 + total_rows * groups * format.plane1_bytes;
    let total = meta + total_rows * (groups / format.meta_groups) * format.meta_bytes;
    (plane1, meta, total)
}

/// One projection of every expert in plane layout: plane0, then plane1 (if
/// the format has one), then meta, in one allocation. Expert `e`'s planes
/// start at row `e * rows` of each plane.
pub struct RepackedPlanes {
    storage: Array,
    format: PlaneFormat,
    experts: usize,
    rows: usize,
    columns: usize,
    plane1: usize,
    meta: usize,
}

impl RepackedPlanes {
    /// Planes held in `storage` (as [`repack_planes`] writes
    /// them) of `experts x rows` rows of `columns` inputs in `format`.
    pub fn from_storage(
        storage: Array,
        format: PlaneFormat,
        experts: usize,
        rows: usize,
        columns: usize,
    ) -> Result<Self> {
        let (plane1, meta, total) = plane_layout(format, experts, rows, columns);
        if storage.dtype() != Dtype::Uint8 || storage.size() != total {
            return Err(Error::msg("expert planes do not match their layout"));
        }
        Ok(Self {
            storage,
            format,
            experts,
            rows,
            columns,
            plane1,
            meta,
        })
    }
    pub fn format(&self) -> PlaneFormat {
        self.format
    }
    pub fn experts(&self) -> usize {
        self.experts
    }
    /// Output rows of one expert.
    pub fn rows(&self) -> usize {
        self.rows
    }
    /// Inputs of one row.
    pub fn columns(&self) -> usize {
        self.columns
    }
    /// The whole allocation: plane0 at 0, plane1 and meta at their offsets.
    pub fn storage(&self) -> &Array {
        &self.storage
    }
    pub fn plane1_offset(&self) -> usize {
        self.plane1
    }
    pub fn meta_offset(&self) -> usize {
        self.meta
    }
    /// Bytes of the planes.
    pub fn bytes(&self) -> usize {
        self.storage.size()
    }
}

/// Rows `[first_row, first_row + rows)` of every expert of `source`, as the
/// reference loader prepares them: native rows on the host, on every core,
/// then `gguf_repack` on the GPU into the planes.
pub fn repack_planes(
    source: &GgufExpertTensor,
    first_row: usize,
    rows: usize,
    stream: &Stream,
) -> Result<RepackedPlanes> {
    let row_bytes = source.row_bytes()?;
    let format = source.plane_format()?;
    let experts = source.experts;
    let columns = source.columns;
    let groups = columns / 32;
    if experts == 0
        || rows == 0
        || !rows.is_multiple_of(TILE_ROWS)
        || first_row + rows > source.rows
        || !groups.is_multiple_of(format.meta_groups)
        || experts
            .checked_mul(source.rows)
            .and_then(|n| n.checked_mul(row_bytes))
            != Some(source.bytes.len())
    {
        return Err(Error::msg("invalid expert tensor geometry for repacking"));
    }
    let native_row_bytes = columns / format.block_elements * format.block_bytes;
    let total_rows = experts * rows;
    let plane0_total = total_rows * groups * format.plane0_bytes;
    let plane1_total = total_rows * groups * format.plane1_bytes;
    let meta_total = total_rows * (groups / format.meta_groups) * format.meta_bytes;
    let plane1 = plane0_total;
    let meta = plane1 + plane1_total;
    let total = meta + meta_total;
    if total > u32::MAX as usize {
        return Err(Error::msg(
            "expert planes exceed the repack's 4 GiB offsets",
        ));
    }
    let storage = output_full_write(stream, &[total as i32], Dtype::Uint8)?;
    let pipeline =
        stream
            .runtime()
            .compile_with(REPACK_SOURCE, "gguf_repack", Math::CompilerDefault)?;
    // Stage whole plane tiles of rows, across experts, within the budget.
    let tile_bytes = TILE_ROWS * native_row_bytes;
    let rows_per_chunk = (STAGING_BYTES / tile_bytes).max(1) * TILE_ROWS;
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
    let mut staging = Vec::new();
    for r0 in (0..total_rows).step_by(rows_per_chunk) {
        let chunk_rows = rows_per_chunk.min(total_rows - r0);
        staging.clear();
        staging.resize(chunk_rows * native_row_bytes, 0);
        // Native rows on a thread per core, as the reference's writers stage
        // them; every row is independent, so the bytes do not depend on the
        // split.
        let per = chunk_rows.div_ceil(threads);
        std::thread::scope(|scope| {
            for (t, part) in staging.chunks_mut(per * native_row_bytes).enumerate() {
                let format = &format;
                scope.spawn(move || {
                    for (j, out) in part.chunks_exact_mut(native_row_bytes).enumerate() {
                        let i = r0 + t * per + j;
                        let (expert, row) = (i / rows, i % rows);
                        let at = (expert * source.rows + first_row + row) * row_bytes;
                        let src = &source.bytes[at..at + row_bytes];
                        if source.ggml_type == GGML_TYPE_NVFP4 {
                            nvfp4_native_row(src, format, out);
                        } else {
                            out.copy_from_slice(src);
                        }
                    }
                });
            }
        });
        let staged = Array::from_slice(&staging, &[staging.len() as i32]);
        // A chunk starts on a plane tile: after the planes of the rows
        // above it in each plane.
        let at0 = r0 * groups * format.plane0_bytes;
        let at1 = plane1 + r0 * groups * format.plane1_bytes;
        let at2 = meta + r0 * (groups / format.meta_groups) * format.meta_bytes;
        let params: [u32; 6] = [
            chunk_rows as u32,
            columns as u32,
            format.id,
            native_row_bytes as u32,
            if format.plane1_bytes == 0 {
                0
            } else {
                (at1 - at0) as u32
            },
            (at2 - at0) as u32,
        ];
        let guard = stream.runtime().commands.encoder()?;
        let enc = guard.encoder();
        enc.set_pipeline(&pipeline);
        let (b, l) = staged.t.buffer_and_layout();
        enc.set_input(0, Some(b), l.offset);
        let (b, l) = storage.t.buffer_and_layout();
        enc.set_output(1, Some(b), l.offset + at0);
        enc.set_bytes(2, &params);
        enc.dispatch_threads((chunk_rows * groups, 1, 1), (256, 1, 1));
    }
    Ok(RepackedPlanes {
        storage,
        format,
        experts,
        rows,
        columns,
        plane1,
        meta,
    })
}

/// A row of llama.cpp NVFP4 blocks (4 UE4M3 scale bytes, then per 16
/// elements 8 bytes holding element j in the low nibble of byte j and
/// element j + 8 in its high one) as the reference loader's native blocks:
/// per native block the E4M3 scales (0x7F read as 0, bit 7 cleared), `g`
/// (1 here), then the codes with element e in nibble e % 2 of byte e / 2.
fn nvfp4_native_row(src: &[u8], format: &PlaneFormat, out: &mut [u8]) {
    let per_native = format.block_elements / 64;
    let g_at = format.meta_bytes - 4;
    let codes_at = format.meta_bytes;
    for (b, native) in out.chunks_exact_mut(format.block_bytes).enumerate() {
        for q in 0..per_native {
            let block = &src[(b * per_native + q) * 36..][..36];
            for s in 0..4 {
                native[4 * q + s] = if block[s] == 0x7F { 0 } else { block[s] & 0x7F };
                let codes = &block[4 + 8 * s..][..8];
                let pairs = &mut native[codes_at + 32 * q + 8 * s..][..8];
                for i in 0..4 {
                    pairs[i] = (codes[2 * i] & 15) | (codes[2 * i + 1] & 15) << 4;
                    pairs[4 + i] = codes[2 * i] >> 4 | (codes[2 * i + 1] & 0xF0);
                }
            }
        }
        native[g_at..g_at + 4].copy_from_slice(&1.0f32.to_le_bytes());
    }
}

#[cfg(test)]
#[path = "reference_planes_tests.rs"]
mod tests;
