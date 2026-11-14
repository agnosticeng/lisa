//! DS4's standalone routed activation and ordered six-expert reduction.

use super::{f32_input, output};
use crate::error::{Error, Result};
use crate::runtime::Math;
use crate::{Array, Dtype, Stream};

const SOURCE: &crate::shader_pack::Source = crate::kernel_source!("deepseek41/moe.metal");

/// Original shared-expert flat activation. This has its own dispatch and
/// code generation; the routed kernel with a unit weight is not its oracle.
pub fn shared_swiglu(gate: &Array, up: &Array, stream: &Stream) -> Result<Array> {
    if gate.rank() != 2 || gate.dim(0) != 1 || gate.dim(1) < 1 {
        return Err(Error::msg("invalid DeepSeek shared activation shape"));
    }
    f32_input(gate, gate.shape())?;
    f32_input(up, gate.shape())?;
    let width = gate.dim(1);
    #[repr(C)]
    struct Args {
        ne00: i32,
        nb01: u64,
        ne10: i32,
        nb11: u64,
        ne0: i32,
        nb1: u64,
        i00: i32,
        i10: i32,
        alpha: f32,
        limit: f32,
    }
    let row = width as u64 * 4;
    let args = Args {
        ne00: width,
        nb01: row,
        ne10: width,
        nb11: row,
        ne0: width,
        nb1: row,
        i00: 0,
        i10: 0,
        alpha: 1.0,
        limit: 10.0,
    };
    let runtime = stream.runtime();
    let pipeline = runtime.compile_with(SOURCE, "kernel_swiglu_flat_f32", Math::CompilerDefault)?;
    let out = output(stream, gate.shape(), Dtype::Float32)?;
    let guard = runtime.commands.encoder()?;
    let encoder = guard.encoder();
    encoder.set_pipeline(&pipeline);
    encoder.set_bytes(0, &args);
    for (slot, array) in [(1, gate), (2, up)] {
        let (buffer, layout) = array.t.buffer_and_layout();
        encoder.set_input(slot, Some(buffer), layout.offset * 4);
    }
    let (buffer, layout) = out.t.buffer_and_layout();
    encoder.set_output(3, Some(buffer), layout.offset * 4);
    let threads = 256
        .min(width as usize)
        .min(pipeline.max_total_threads_per_threadgroup());
    encoder.dispatch_groups(((width as usize).div_ceil(threads), 1, 1), (threads, 1, 1));
    Ok(out)
}

/// Clamp, SwiGLU and route weighting in DS4's order. No BF16 boundary is
/// inserted in the routed experts; the shared expert has separate boundaries.
pub fn swiglu_weight(
    gate: &Array,
    up: &Array,
    weights: &Array,
    clamp: f32,
    stream: &Stream,
) -> Result<Array> {
    if gate.rank() != 3
        || gate.dim(0) < 1
        || gate.dim(0) > 65535
        || gate.dim(1) < 1
        || gate.dim(1) > 32
        || gate.dim(2) < 1
        || !clamp.is_finite()
        || clamp < 0.0
    {
        return Err(Error::msg("invalid DS4 SwiGLU geometry"));
    }
    f32_input(gate, gate.shape())?;
    f32_input(up, gate.shape())?;
    f32_input(weights, &[gate.dim(0), gate.dim(1)])?;
    let width = gate.dim(2) as u32;
    let rows = (gate.dim(0) * gate.dim(1)) as u32;
    #[repr(C)]
    struct Args {
        width: u32,
        rows: u32,
        gate_stride: u64,
        up_stride: u64,
        mid_stride: u64,
        weight_stride: u64,
        write_clamped: u32,
        clamp: f32,
    }
    let args = Args {
        width,
        rows,
        gate_stride: width as u64 * 4,
        up_stride: width as u64 * 4,
        mid_stride: width as u64 * 4,
        weight_stride: 4,
        write_clamped: 0,
        clamp,
    };
    let runtime = stream.runtime();
    let pipeline = runtime.compile_with(
        SOURCE,
        "kernel_dsv4_moe_swiglu_weight",
        Math::CompilerDefault,
    )?;
    let out = output(stream, gate.shape(), Dtype::Float32)?;
    let guard = runtime.commands.encoder()?;
    let encoder = guard.encoder();
    encoder.set_pipeline(&pipeline);
    encoder.set_bytes(0, &args);
    for (slot, array) in [(1, gate), (2, up), (4, weights)] {
        let (buffer, layout) = array.t.buffer_and_layout();
        encoder.set_input(slot, Some(buffer), layout.offset * 4);
    }
    let (buffer, layout) = out.t.buffer_and_layout();
    encoder.set_output(3, Some(buffer), layout.offset * 4);
    encoder.dispatch_groups((rows as usize, 1, 1), (256, 1, 1));
    Ok(out)
}

pub fn sum_six(input: &Array, stream: &Stream) -> Result<Array> {
    if input.rank() != 3
        || input.dim(0) < 1
        || input.dim(0) > 65535
        || input.dim(1) != 6
        || input.dim(2) < 1
    {
        return Err(Error::msg("invalid DS4 six-expert reduction geometry"));
    }
    f32_input(input, input.shape())?;
    let (tokens, width) = (input.dim(0), input.dim(2));
    #[repr(C)]
    struct Args {
        width: u32,
        tokens: u32,
        src_stride: u64,
        dst_stride: u64,
    }
    let args = Args {
        width: width as u32,
        tokens: tokens as u32,
        src_stride: width as u64 * 6 * 4,
        dst_stride: width as u64 * 4,
    };
    let runtime = stream.runtime();
    let pipeline =
        runtime.compile_with(SOURCE, "kernel_dsv4_moe_sum6_f32", Math::CompilerDefault)?;
    let out = output(stream, &[tokens, width], Dtype::Float32)?;
    let guard = runtime.commands.encoder()?;
    let encoder = guard.encoder();
    encoder.set_pipeline(&pipeline);
    encoder.set_bytes(0, &args);
    let (buffer, layout) = input.t.buffer_and_layout();
    encoder.set_input(1, Some(buffer), layout.offset * 4);
    let (buffer, layout) = out.t.buffer_and_layout();
    encoder.set_output(2, Some(buffer), layout.offset * 4);
    encoder.dispatch_groups((tokens as usize, 1, 1), (256, 1, 1));
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deepseek41_routed_clamp_and_ordered_sum() {
        let stream = Stream::gpu();
        let width = 256;
        let gate: Vec<f32> = (0..12 * width).map(|i| (i % 47) as f32 - 23.0).collect();
        let up: Vec<f32> = (0..12 * width).map(|i| (i % 43) as f32 - 21.0).collect();
        let weights: Vec<f32> = (0..12).map(|i| (i % 7) as f32 / 8.0).collect();
        let out = swiglu_weight(
            &Array::from_slice(&gate, &[2, 6, width as i32]),
            &Array::from_slice(&up, &[2, 6, width as i32]),
            &Array::from_slice(&weights, &[2, 6]),
            10.0,
            &stream,
        )
        .unwrap();
        for (i, &actual) in out.as_slice::<f32>().iter().enumerate() {
            let g = gate[i].min(10.0);
            let expected = (g / (1.0 + (-g).exp())) * up[i].clamp(-10.0, 10.0) * weights[i / width];
            assert!((actual - expected).abs() <= 1e-5 * expected.abs().max(1.0));
        }
        let values: Vec<f32> = [1e10f32, 1.0, -1e10, 2.0, 3.0, 4.0]
            .iter()
            .flat_map(|&x| vec![x; width])
            .collect();
        let sum = sum_six(&Array::from_slice(&values, &[1, 6, width as i32]), &stream).unwrap();
        assert!(sum.as_slice::<f32>().iter().all(|&x| x == 9.0));
    }
}
