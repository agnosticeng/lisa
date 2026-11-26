//! Strided F32 concatenation as one full writer, preserving every bit.
use super::output_full_write as output;
use crate::{
    Array, Dtype, Stream,
    error::{Error, Result},
    runtime::Math,
};
const SOURCE: &crate::shader_pack::Source =
    crate::kernel_source!("packed_pointwise/ordered_fusions.metal");
#[cfg(test)]
#[path = "ordered_fusions_tests.rs"]
mod tests;

#[repr(C)]
struct ConcatArgs {
    rank: u32,
    axis: u32,
    inputs: u32,
    count: u32,
    shape: [u32; 3],
    ends: [u32; 8],
    strides: [u64; 24],
}
/// One full writer for 2..=8 arbitrary nonnegative-strided F32 inputs.
pub fn concat_f32(inputs: &[&Array], axis: usize, stream: &Stream) -> Result<Array> {
    if !(2..=8).contains(&inputs.len()) {
        return Err(Error::msg("concat requires 2..=8 inputs"));
    }
    let first = inputs[0];
    let rank = first.rank();
    if !(1..=3).contains(&rank) || axis >= rank {
        return Err(Error::msg("unsupported fused concat rank/axis"));
    }
    let mut shape = first.shape().to_vec();
    shape[axis] = 0;
    let mut args = ConcatArgs {
        rank: rank as u32,
        axis: axis as u32,
        inputs: inputs.len() as u32,
        count: 0,
        shape: [1; 3],
        ends: [0; 8],
        strides: [0; 24],
    };
    for (i, x) in inputs.iter().enumerate() {
        if x.dtype() != Dtype::Float32
            || x.rank() != rank
            || x.shape().iter().any(|&d| d <= 0)
            || x.shape()
                .iter()
                .enumerate()
                .any(|(d, &v)| d != axis && v != first.dim(d as i32))
        {
            return Err(Error::msg("invalid fused concat input"));
        }
        x.eval_enqueue()?;
        let (_, l) = x.t.buffer_and_layout();
        for d in 0..rank {
            args.strides[i * 3 + d] =
                u64::try_from(l.strides[d]).map_err(|_| Error::msg("negative concat stride"))?;
        }
        shape[axis] = shape[axis]
            .checked_add(x.dim(axis as i32))
            .ok_or_else(|| Error::msg("concat axis overflow"))?;
        args.ends[i] = shape[axis] as u32;
    }
    let count = shape
        .iter()
        .try_fold(1u32, |n, &d| n.checked_mul(d as u32))
        .ok_or_else(|| Error::msg("concat size overflow"))?;
    args.count = count;
    for d in 0..rank {
        args.shape[d] = shape[d] as u32;
    }
    let y = output(stream, &shape, Dtype::Float32)?;
    let pipeline = stream
        .runtime()
        .compile_with(SOURCE, "lisa_concat_f32", Math::Safe)?;
    let guard = stream.runtime().commands.encoder()?;
    let e = guard.encoder();
    e.set_pipeline(&pipeline);
    e.set_bytes(0, &args);
    for i in 0..8 {
        let x = inputs.get(i).copied().unwrap_or(first);
        let (b, l) = x.t.buffer_and_layout();
        e.set_input(i + 1, Some(b), l.offset * 4);
    }
    let (b, l) = y.t.buffer_and_layout();
    e.set_output(9, Some(b), l.offset * 4);
    e.dispatch_groups(((count as usize).div_ceil(256), 1, 1), (256, 1, 1));
    Ok(y)
}
