//! Bind original GGUF payloads to native DS4/GGML Metal dispatches.
//! No affine conversion, CPU expert selection or whole-matrix dequantization.

use crate::core::gguf_resident::PackedTensor;
use anyhow::{Context, Result, bail, ensure};
use lisa_gguf::TensorType;
use lisa_mlx::models::deepseek41::{self as gpu, BlockFormat};
use lisa_mlx::{Array, Dtype, Stream};

pub fn block_format(kind: TensorType) -> Result<BlockFormat> {
    match kind {
        TensorType::IQ2_XXS => Ok(BlockFormat::Iq2Xxs),
        TensorType::IQ2_XS => Ok(BlockFormat::Iq2Xs),
        TensorType::IQ1_M => Ok(BlockFormat::Iq1M),
        TensorType::Q5_K => Ok(BlockFormat::Q5K),
        _ => bail!("GGUF expert format {kind:?} has no DeepSeek dispatch"),
    }
}

/// Typed, immutable view of an original unquantized GGUF tensor. The caller
/// must retain its ResidentWeights owner until all encoded operations finish.
pub fn plain_view(weight: &PackedTensor) -> Result<Array> {
    weight.plain_view()
}

pub fn linear(weight: &PackedTensor, input: &Array, round: bool, stream: &Stream) -> Result<Array> {
    ensure!(
        weight.info.dimensions.len() == 2,
        "{} is not a matrix",
        weight.info.name
    );
    linear_rows(weight, 0, weight.info.dimensions[1], input, round, stream)
}

/// Original DS4 paired query/KV gate. Other formats retain their independent
/// projections, including the same BF16 output boundary.
pub fn query_kv_pair(
    query: &PackedTensor,
    kv: &PackedTensor,
    input: &Array,
    stream: &Stream,
) -> Result<(Array, Array)> {
    if query.info.kind == TensorType::Q8_0
        && kv.info.kind == TensorType::Q8_0
        && input.rank() == 2
        && input.dim(0) == 1
    {
        return Ok(gpu::q8_matvec_pair(
            query.packed_bytes(),
            kv.packed_bytes(),
            input,
            stream,
        )?);
    }
    Ok((
        linear(query, input, true, stream)?,
        linear(kv, input, true, stream)?,
    ))
}

/// DS4's eight O_A groups. Q5_K uses the original GGML group strides in one
/// launch; formats with a different reduction retain their scalar dispatch.
pub fn attention_output_groups(
    weight: &PackedTensor,
    input: &Array,
    stream: &Stream,
) -> Result<Array> {
    ensure!(
        weight.info.dimensions == [4096, 8192] && input.shape() == [1, 64, 512],
        "invalid DeepSeek O_A group geometry"
    );
    if weight.info.kind == TensorType::Q5_K {
        let weights =
            weight
                .packed_bytes()
                .reshape(&[8, 1024, i32::try_from(weight.info.row_bytes()?)?])?;
        let mut out = gpu::grouped_packed_matvec(
            BlockFormat::Q5K,
            &weights,
            &input.reshape(&[8, 1, 4096])?,
            stream,
        )?
        .reshape(&[1, 8192])?;
        gpu::quantize_in_place(&mut out, gpu::ActivationFormat::Bf16, stream)?;
        return Ok(out);
    }
    let mut groups = Vec::with_capacity(8);
    for group in 0..8 {
        let input = super::state::view(input, group * 4096, &[1, 4096])?;
        groups.push(linear_rows(
            weight,
            group as u64 * 1024,
            1024,
            &input,
            true,
            stream,
        )?);
    }
    Ok(Array::concatenate(&groups.iter().collect::<Vec<_>>(), 1)?)
}

/// Slice complete output rows without changing a block, scale or row stride.
/// This is also the adapter for the eight independent attention O_A groups.
pub fn linear_rows(
    weight: &PackedTensor,
    first: u64,
    count: u64,
    input: &Array,
    round: bool,
    stream: &Stream,
) -> Result<Array> {
    ensure!(
        weight.info.dimensions.len() == 2
            && count > 0
            && first <= weight.info.dimensions[1]
            && count <= weight.info.dimensions[1] - first,
        "invalid GGUF output-row slice for {}",
        weight.info.name
    );
    ensure!(
        input.rank() == 2 && input.dim(0) > 0 && input.dim(0) <= 8,
        "DeepSeek scalar projection supports one to eight rows; grouped prefill needs its own schedule"
    );
    let row_bytes = weight.info.row_bytes()?;
    let (buffer, layout) = weight.packed_bytes().t.buffer_and_layout();
    let offset = (first
        .checked_mul(row_bytes)
        .context("GGUF row offset overflow")? as usize)
        .checked_add(layout.offset)
        .context("GGUF row offset overflow")?;
    let bytes = Array::from_buffer_offset(
        buffer.clone(),
        offset,
        &[i32::try_from(count)?, i32::try_from(row_bytes)?],
        Dtype::Uint8,
    )?;
    let kind = weight.info.kind;
    if kind == TensorType::Q8_0 {
        return Ok(gpu::q8_matvec(&bytes, input, round, false, stream)?);
    }
    let mut result = if matches!(kind, TensorType::F32 | TensorType::F16 | TensorType::BF16) {
        let dtype = match kind {
            TensorType::F32 => Dtype::Float32,
            TensorType::F16 => Dtype::Float16,
            _ => Dtype::Bfloat16,
        };
        let view = Array::from_buffer_offset(
            buffer.clone(),
            offset,
            &[
                i32::try_from(count)?,
                i32::try_from(weight.info.dimensions[0])?,
            ],
            dtype,
        )?;
        gpu::plain_matvec(&view, input, stream)?
    } else {
        gpu::packed_matvec(block_format(kind)?, &bytes, input, stream)?
    };
    if round {
        gpu::quantize_in_place(&mut result, gpu::ActivationFormat::Bf16, stream)?;
    }
    Ok(result)
}

pub fn experts(
    weight: &PackedTensor,
    input: &Array,
    selected: &Array,
    stream: &Stream,
) -> Result<Array> {
    ensure!(
        weight.info.dimensions.len() == 3,
        "{} is not an expert tensor",
        weight.info.name
    );
    Ok(gpu::expert_matvec(
        block_format(weight.info.kind)?,
        weight.packed_bytes(),
        input,
        selected,
        stream,
    )?)
}
