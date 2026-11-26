//! Generic dispatch of the vendored reference's kernels
//! (`shaders/reference/`, revision and hashes in its `provenance.json`): an
//! entry point of one vendored file with its buffers, parameter block,
//! function constants and grid, as the reference's host code encodes it.
//! The device knows no model here; callers own the meaning of every slot.
use crate::{
    Array, Dtype, Stream,
    error::{Error, Result},
    runtime::{ConstVal, Math},
};
use lisa_backend::{VendoredConstant, VendoredDispatch, VendoredSource};

fn source(source: VendoredSource) -> &'static crate::shader_pack::Source {
    match source {
        VendoredSource::RelativeAttention => {
            crate::kernel_source!("reference/relative_attention.metal")
        }
        VendoredSource::ShortConvolution => {
            crate::kernel_source!("reference/short_convolution.metal")
        }
        VendoredSource::BlockResidualMix => {
            crate::kernel_source!("reference/block_residual_mix.metal")
        }
        VendoredSource::BlockResidualMixRows => {
            crate::kernel_source!("reference/block_residual_mix_rows.metal")
        }
        VendoredSource::MessageGate => crate::kernel_source!("reference/message_gate.metal"),
        VendoredSource::Moe => crate::kernel_source!("reference/moe.metal"),
        VendoredSource::MoeGguf => crate::kernel_source!("reference/moe_gguf.metal"),
        VendoredSource::GgufFloat => crate::kernel_source!("reference/gguf_float.metal"),
        VendoredSource::GgufLinear => crate::kernel_source!("reference/gguf_linear.metal"),
        VendoredSource::GgufPrefillInt8 => {
            crate::kernel_source!("reference/gguf_prefill_int8.metal")
        }
        VendoredSource::LinearColumn => crate::kernel_source!("reference/linear_gguf_column.metal"),
        VendoredSource::LinearRegister => {
            crate::kernel_source!("reference/linear_gguf_sgmatrix.metal")
        }
        VendoredSource::Normalization => crate::kernel_source!("reference/normalization.metal"),
        VendoredSource::RowCopy => crate::kernel_source!("reference/row_copy.metal"),
        VendoredSource::Embedding => crate::kernel_source!("reference/embedding.metal"),
        VendoredSource::LogitFinish => crate::kernel_source!("reference/logit_finish.metal"),
        VendoredSource::Sampling => crate::kernel_source!("reference/sampling.metal"),
        VendoredSource::Draft => crate::kernel_source!("reference/draft.metal"),
        VendoredSource::DraftContext => crate::kernel_source!("reference/draft_context.metal"),
        VendoredSource::Rope => crate::kernel_source!("reference/rope.metal"),
        VendoredSource::StateGlue => crate::kernel_source!("gguf/reference_state_glue.metal"),
    }
}

/// Encode `d` over `buffers` (one per slot, in order). Returns buffer 0.
pub fn vendored_dispatch(
    d: &VendoredDispatch,
    buffers: &[&Array],
    stream: &Stream,
) -> Result<Array> {
    if buffers.len() != d.slots.len() || buffers.is_empty() {
        return Err(Error::msg("vendored dispatch buffer count mismatch"));
    }
    for b in buffers {
        b.eval_enqueue()?;
    }
    let runtime = stream.runtime();
    let constants = d
        .constants
        .iter()
        .map(|&(i, c)| {
            (
                i as usize,
                match c {
                    VendoredConstant::Bool(b) => ConstVal::Bool(b),
                    VendoredConstant::UInt(u) => ConstVal::UInt(u),
                },
            )
        })
        .collect::<Vec<_>>();
    let pipeline = if constants.is_empty() {
        runtime.compile_with(source(d.source), &d.entry, Math::CompilerDefault)?
    } else {
        runtime.compile_with_constants(
            source(d.source),
            &d.entry,
            Math::CompilerDefault,
            &constants,
        )?
    };
    let guard = runtime.commands.encoder()?;
    let enc = guard.encoder();
    enc.set_pipeline(&pipeline);
    for (i, (b, &slot)) in buffers.iter().zip(&d.slots).enumerate() {
        let (buffer, layout) = b.t.buffer_and_layout();
        let at = layout.offset * b.dtype().size_of();
        // Every binding may be written: the reference's kernels write into
        // their bound buffers.
        enc.set_output(slot as usize, Some(buffer), at);
        if d.addressed.contains(&(i as u32)) {
            enc.retain_written_until_completion(buffer);
        }
    }
    if !d.bytes.is_empty() {
        enc.set_bytes_directly(
            d.bytes_slot as usize,
            d.bytes.len(),
            d.bytes.as_ptr().cast(),
        );
    }
    let threads = (
        d.threads[0] as usize,
        d.threads[1] as usize,
        d.threads[2] as usize,
    );
    match d.indirect {
        Some((input, offset)) => {
            let (buffer, layout) = buffers[input as usize].t.buffer_and_layout();
            let at = layout.offset * buffers[input as usize].dtype().size_of() + offset as usize;
            enc.dispatch_groups_indirect(buffer, at, threads);
        }
        None => enc.dispatch_groups(
            (d.grid[0] as usize, d.grid[1] as usize, d.grid[2] as usize),
            threads,
        ),
    }
    Ok(buffers[0].clone())
}

/// Page-table entries of one extent: its GPU address | page index.
pub fn kv_page_table(extent: &Array, extent_pages: usize) -> Result<Array> {
    extent.eval_enqueue()?;
    let (buffer, layout) = extent.t.buffer_and_layout();
    let base = buffer.gpu_address() + (layout.offset * extent.dtype().size_of()) as u64;
    if base & 0x3FFF != 0 {
        return Err(Error::msg("KV extent is not 16 KiB aligned"));
    }
    let entries = (0..extent_pages as u64)
        .map(|p| base | p)
        .collect::<Vec<_>>();
    let words = entries
        .iter()
        .flat_map(|e| [*e as u32, (*e >> 32) as u32])
        .collect::<Vec<_>>();
    Array::from_slice(&words, &[(2 * extent_pages) as i32]).as_dtype(Dtype::Uint32)
}

/// GPU addresses of each view plus its byte offset, as U32 pairs.
pub fn plane_addresses(arrays: &[&Array], offsets: &[u64]) -> Result<Array> {
    if arrays.len() != offsets.len() {
        return Err(Error::msg("address table size mismatch"));
    }
    let mut words = Vec::with_capacity(2 * arrays.len());
    for (a, &offset) in arrays.iter().zip(offsets) {
        a.eval_enqueue()?;
        let (buffer, layout) = a.t.buffer_and_layout();
        let at = buffer.gpu_address() + (layout.offset * a.dtype().size_of()) as u64 + offset;
        words.extend([at as u32, (at >> 32) as u32]);
    }
    Ok(Array::from_slice(&words, &[words.len() as i32]))
}
