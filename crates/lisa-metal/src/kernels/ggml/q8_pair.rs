//! DS4 paired Q8 query/KV projection, including its literal decode shape.

use super::{MatvecArgs, Q8_SOURCE, f32_input, output};
use crate::error::{Error, Result};
use crate::runtime::{ConstVal, Math};
use crate::{Array, Dtype, Stream};

pub fn q8_matvec_pair(a: &Array, b: &Array, x: &Array, stream: &Stream) -> Result<(Array, Array)> {
    if x.rank() != 2 || x.dim(0) != 1 || x.dim(1) < 32 || x.dim(1) % 32 != 0 {
        return Err(Error::msg("DeepSeek paired Q8 projection requires one row"));
    }
    let width = x.dim(1);
    let row_bytes = width as u64 / 32 * 34;
    f32_input(x, &[1, width])?;
    for w in [a, b] {
        if w.dtype() != Dtype::Uint8
            || w.rank() != 2
            || w.dim(0) < 2
            || w.dim(0) > 65536
            || w.dim(0) % 2 != 0
            || w.dim(1) as u64 != row_bytes
            || !w.t.buffer_and_layout().1.is_contiguous()
            || w.t.buffer_and_layout().1.offset % 2 != 0
        {
            return Err(Error::msg("unsupported DeepSeek paired Q8 weight layout"));
        }
        w.eval_enqueue()?;
    }
    let make_args = |rows: i32| MatvecArgs {
        ne00: width,
        ne01: rows,
        ne02: 1,
        nb00: 34,
        nb01: row_bytes,
        nb02: row_bytes * rows as u64,
        nb03: row_bytes * rows as u64,
        ne10: width,
        ne11: 1,
        ne12: 1,
        nb10: 4,
        nb11: width as u64 * 4,
        nb12: width as u64 * 4,
        nb13: width as u64 * 4,
        ne0: rows,
        ne1: 1,
        nr0: 2,
        r2: 1,
        r3: 1,
    };
    let literal = width == 5120 && a.dim(0) == 1280 && b.dim(0) == 512;
    let function = if literal {
        "kernel_mul_mv_q8_0_f32_bf16_pair_qa_kv"
    } else {
        "kernel_mul_mv_q8_0_f32_bf16_pair"
    };
    let runtime = stream.runtime();
    let pipeline = runtime.compile_with_constants(
        Q8_SOURCE,
        function,
        Math::CompilerDefault,
        &[(600, ConstVal::Short(4))],
    )?;
    let ya = output(stream, &[1, a.dim(0)], Dtype::Float32)?;
    let yb = output(stream, &[1, b.dim(0)], Dtype::Float32)?;
    let guard = runtime.commands.encoder()?;
    let encoder = guard.encoder();
    encoder.set_pipeline(&pipeline);
    encoder.set_bytes(0, &make_args(a.dim(0)));
    encoder.set_bytes(1, &make_args(b.dim(0)));
    for (slot, input) in [(2, a), (3, b), (4, x)] {
        let (buffer, layout) = input.t.buffer_and_layout();
        encoder.set_input(slot, Some(buffer), layout.offset * input.dtype().size_of());
    }
    for (slot, output) in [(5, &ya), (6, &yb)] {
        let (buffer, layout) = output.t.buffer_and_layout();
        encoder.set_output(slot, Some(buffer), layout.offset * 4);
    }
    encoder.set_threadgroup_memory_length(0, 32 * 2 * 4);
    encoder.dispatch_groups(((a.dim(0) + b.dim(0)) as usize / 2, 1, 1), (32, 4, 1));
    Ok((ya, yb))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deepseek41_paired_q8_matches_separate_bits() {
        let stream = Stream::gpu();
        for (width, rows_a, rows_b) in [(128, 2, 14), (1280, 192, 512), (5120, 1280, 512)] {
            let make = |rows: i32, seed: usize| {
                let mut bytes = Vec::new();
                for block in 0..rows as usize * width as usize / 32 {
                    bytes.extend_from_slice(
                        &half::f16::from_f32(((block * 17 + seed) % 127) as f32 / 173.0)
                            .to_bits()
                            .to_le_bytes(),
                    );
                    bytes.extend((0..32).map(|i| ((i * 13 + block * 19 + seed) % 255) as u8));
                }
                Array::from_slice(&bytes, &[rows, width / 32 * 34])
            };
            let a = make(rows_a, 3);
            let b = make(rows_b, 73);
            let x = Array::from_slice(
                &(0..width)
                    .map(|i| ((i * 113 % 997) as f32 - 499.0) / 73.0)
                    .collect::<Vec<_>>(),
                &[1, width],
            );
            let expected_a = super::super::q8_matvec(&a, &x, true, false, &stream).unwrap();
            let expected_b = super::super::q8_matvec(&b, &x, true, false, &stream).unwrap();
            let (actual_a, actual_b) = q8_matvec_pair(&a, &b, &x, &stream).unwrap();
            for (actual, expected) in [(actual_a, expected_a), (actual_b, expected_b)] {
                assert_eq!(
                    actual
                        .as_slice::<f32>()
                        .iter()
                        .map(|x| x.to_bits())
                        .collect::<Vec<_>>(),
                    expected
                        .as_slice::<f32>()
                        .iter()
                        .map(|x| x.to_bits())
                        .collect::<Vec<_>>()
                );
            }
        }
    }
}
