//! Fusions that preserve the standalone F32 tensor operation boundaries.
use super::{f32_input, output};
use crate::{
    Array, Dtype, Stream,
    error::{Error, Result},
    runtime::Math,
};
const SOURCE: &crate::shader_pack::Source =
    crate::kernel_source!("packed_pointwise/pointwise.metal");
#[repr(C)]
struct Args {
    width: u32,
    rows: u32,
    scale: f32,
    pad: u32,
}
fn dispatch(
    name: &str,
    args: Args,
    inputs: &[&Array],
    outputs: &[&Array],
    router: bool,
    stream: &Stream,
) -> Result<()> {
    let pipeline = stream.runtime().compile_with(SOURCE, name, Math::Safe)?;
    let guard = stream.runtime().commands.encoder()?;
    let e = guard.encoder();
    e.set_pipeline(&pipeline);
    e.set_bytes(0, &args);
    for (i, a) in inputs.iter().enumerate() {
        let (b, l) = a.t.buffer_and_layout();
        e.set_input(i + 1, Some(b), l.offset * a.dtype().size_of());
    }
    for (i, a) in outputs.iter().enumerate() {
        let (b, l) = a.t.buffer_and_layout();
        e.set_output(
            inputs.len() + i + 1,
            Some(b),
            l.offset * a.dtype().size_of(),
        );
    }
    let threads = if router { 32 } else { 256 };
    let groups = if router {
        args.rows as usize
    } else {
        (args.width as usize * args.rows as usize).div_ceil(threads)
    };
    e.dispatch_groups((groups, 1, 1), (threads, 1, 1));
    Ok(())
}
pub fn relu_squared(x: &Array, stream: &Stream) -> Result<Array> {
    if x.size() == 0 || x.size() > u32::MAX as usize {
        return Err(Error::msg("invalid squared ReLU size"));
    }
    f32_input(x, x.shape())?;
    let out = output(stream, x.shape(), Dtype::Float32)?;
    dispatch(
        "lisa_relu_squared",
        Args {
            width: x.size() as u32,
            rows: 1,
            scale: 0.,
            pad: 0,
        },
        &[x],
        &[&out],
        false,
        stream,
    )?;
    Ok(out)
}
pub fn weighted_sum6(x: &Array, weights: &Array, stream: &Stream) -> Result<Array> {
    if x.rank() != 3
        || x.dim(0) < 1
        || x.dim(0) > 65535
        || x.dim(1) != 6
        || x.dim(2) < 1
        || (x.dim(0) as u64 * x.dim(2) as u64) > u32::MAX as u64
    {
        return Err(Error::msg("invalid weighted six-expert reduction"));
    }
    f32_input(x, x.shape())?;
    f32_input(weights, &[x.dim(0), 6])?;
    let out = output(stream, &[x.dim(0), x.dim(2)], Dtype::Float32)?;
    dispatch(
        "lisa_weighted_sum6",
        Args {
            width: x.dim(2) as u32,
            rows: x.dim(0) as u32,
            scale: 0.,
            pad: 0,
        },
        &[x, weights],
        &[&out],
        false,
        stream,
    )?;
    Ok(out)
}
/// Stable descending sigmoid+bias selection over 256 experts, six routes.
/// Scores are normalized with the existing four-read F32 reduction law.
pub fn sigmoid_router256(
    logits: &Array,
    bias: &Array,
    scale: f32,
    stream: &Stream,
) -> Result<(Array, Array)> {
    if logits.rank() != 2
        || logits.dim(0) < 1
        || logits.dim(0) > 65535
        || logits.dim(1) != 256
        || !scale.is_finite()
        || scale <= 0.
    {
        return Err(Error::msg("invalid sigmoid router geometry"));
    }
    f32_input(logits, logits.shape())?;
    f32_input(bias, &[256])?;
    let ids = output(stream, &[logits.dim(0), 6], Dtype::Int32)?;
    let weights = output(stream, &[logits.dim(0), 6], Dtype::Float32)?;
    dispatch(
        "lisa_sigmoid_router256",
        Args {
            width: 256,
            rows: logits.dim(0) as u32,
            scale,
            pad: 0,
        },
        &[logits, bias],
        &[&ids, &weights],
        true,
        stream,
    )?;
    Ok((ids, weights))
}

/// Two independent sigmoid-gated row blends; the original raw rows stay immutable.
pub fn row_gate_blend_pair(a: &[&Array], zero: bool, stream: &Stream) -> Result<(Array, Array)> {
    if a.len() != 6
        || a[0].rank() != 2
        || a[0].dim(0) < 1
        || a[0].dim(0) > 65535
        || a[0].dim(1) < 1
        || a[0].size() > u32::MAX as usize
    {
        return Err(Error::msg("invalid row gate blend geometry"));
    }
    let (rows, width) = (a[0].dim(0), a[0].dim(1));
    for x in &a[..4] {
        f32_input(x, &[rows, width])?;
    }
    for x in &a[4..] {
        f32_input(x, &[1, rows])?;
    }
    let k = output(stream, &[rows, width], Dtype::Float32)?;
    let v = output(stream, &[rows, width], Dtype::Float32)?;
    dispatch(
        "lisa_row_gate_blend_pair",
        Args {
            width: width as u32,
            rows: rows as u32,
            scale: 0.,
            pad: zero as u32,
        },
        a,
        &[&k, &v],
        false,
        stream,
    )?;
    Ok((k, v))
}
/// Three signal rows minus one sigmoid-scaled noise row, then a SiLU output gate.
pub fn differential_silu_gate(
    a: &Array,
    lambda: &Array,
    gate: &Array,
    bias: f32,
    stream: &Stream,
) -> Result<Array> {
    if a.rank() != 3
        || a.dim(0) < 1
        || a.dim(0) > 65535
        || a.dim(1) != 4
        || a.dim(2) < 1
        || a.size() > i32::MAX as usize
        || !bias.is_finite()
    {
        return Err(Error::msg("invalid differential gate geometry"));
    }
    let (heads, width) = (a.dim(0), a.dim(2));
    f32_input(a, a.shape())?;
    f32_input(lambda, &[1, heads])?;
    f32_input(gate, &[1, heads * 3 * width])?;
    let out = output(stream, &[1, heads * 3 * width], Dtype::Float32)?;
    dispatch(
        "lisa_differential_silu_gate",
        Args {
            width: width as u32,
            rows: heads as u32 * 3,
            scale: bias,
            pad: 0,
        },
        &[a, lambda, gate],
        &[&out],
        false,
        stream,
    )?;
    Ok(out)
}

/// Batched small-row F32 matrix product, optionally interpreting RHS as transposed.
/// Explicitly scoped to 2..=8 query rows; ordinary matmul keeps its old dispatch.
pub fn small_row_matmul(
    lhs: &Array,
    rhs: &Array,
    transpose_rhs: bool,
    stream: &Stream,
) -> Result<Array> {
    if lhs.rank() != 3
        || rhs.rank() != 3
        || lhs.dim(0) < 1
        || lhs.dim(0) != rhs.dim(0)
        || !(2..=8).contains(&lhs.dim(1))
        || lhs.dim(2) < 1
        || rhs.dim(1) < 1
        || rhs.dim(2) < 1
        || lhs.dim(2) != rhs.dim(if transpose_rhs { 2 } else { 1 })
    {
        return Err(Error::msg("invalid small-row matmul geometry"));
    }
    f32_input(lhs, lhs.shape())?;
    f32_input(rhs, rhs.shape())?;
    let n = rhs.dim(if transpose_rhs { 1 } else { 2 });
    let out = output(stream, &[lhs.dim(0), lhs.dim(1), n], Dtype::Float32)?;
    crate::jit::dense_gemm_small_rows(&lhs.t, &rhs.t, &out.t, transpose_rhs)?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::{self, indexing::IndexOp};
    fn exact(a: &Array, b: &Array) {
        let a = a.as_slice::<f32>();
        let b = b.as_slice::<f32>();
        assert_eq!(a.len(), b.len());
        for (i, (a, b)) in a.iter().zip(b).enumerate() {
            assert_eq!(a.to_bits(), b.to_bits(), "element {i}: {a} != {b}");
        }
    }
    #[test]
    fn packed_small_row_matmul_preserves_original_mma_bits() {
        let stream = Stream::gpu();
        let mut seed = 631498u32;
        let mut data = |n: usize| {
            (0..n)
                .map(|_| {
                    seed ^= seed << 13;
                    seed ^= seed >> 17;
                    seed ^= seed << 5;
                    seed as i32 as f32 / 76391234.
                })
                .collect::<Vec<_>>()
        };
        for b in [1, 12] {
            for m in [2, 4, 8] {
                for length in [1, 2, 7, 16, 17, 31, 32, 33, 128, 257, 1021, 4097] {
                    for trans in [false, true] {
                        let (k, n) = if trans { (128, length) } else { (length, 128) };
                        let x =
                            Array::from_slice(&data(b * m * k), &[b as i32, m as i32, k as i32]);
                        let shape = if trans {
                            vec![b as i32, n as i32, k as i32]
                        } else {
                            vec![b as i32, k as i32, n as i32]
                        };
                        let w = Array::from_slice(&data(b * k * n), &shape);
                        let reference = if trans {
                            w.transpose_axes(&[0, 2, 1]).unwrap()
                        } else {
                            w.clone()
                        };
                        let expected = x.matmul(&reference).unwrap();
                        let actual = small_row_matmul(&x, &w, trans, &stream).unwrap();
                        exact(&actual, &expected);
                    }
                }
            }
        }
    }

    #[test]
    fn packed_attention_gates_match_separate_operations() {
        let stream = Stream::gpu();
        let mut seed = 13791u32;
        let mut values = |n: usize| {
            (0..n)
                .map(|_| {
                    seed ^= seed << 13;
                    seed ^= seed >> 17;
                    seed ^= seed << 5;
                    seed as i32 as f32 / 97398123.
                })
                .collect::<Vec<_>>()
        };
        for (h, w) in [(1, 31), (12, 128)] {
            let rows: Vec<Array> = (0..4)
                .map(|_| Array::from_slice(&values(h * w), &[h as i32, w as i32]))
                .collect();
            let kg = Array::from_slice(&values(h), &[1, h as i32]);
            let vg = Array::from_slice(&values(h), &[1, h as i32]);
            for zero in [false, true] {
                let (k, v) = row_gate_blend_pair(
                    &[&rows[0], &rows[1], &rows[2], &rows[3], &kg, &vg],
                    zero,
                    &stream,
                )
                .unwrap();
                for (actual, raw, prev, gate) in
                    [(&k, &rows[0], &rows[2], &kg), (&v, &rows[1], &rows[3], &vg)]
                {
                    let mut a = gate.sigmoid().unwrap().reshape(&[h as i32, 1]).unwrap();
                    if zero {
                        a = a.multiply(Array::from_slice(&[0f32], &[1])).unwrap();
                    }
                    let inv = Array::from_slice(&[1f32], &[1]).subtract(&a).unwrap();
                    let expected = prev
                        .multiply(&a)
                        .unwrap()
                        .add(raw.multiply(&inv).unwrap())
                        .unwrap();
                    exact(actual, &expected);
                }
            }
            let a = Array::from_slice(&values(h * 4 * w), &[h as i32, 4, w as i32]);
            let gate = Array::from_slice(&values(h * 3 * w), &[1, (h * 3 * w) as i32]);
            let l = kg.sigmoid().unwrap().reshape(&[h as i32, 1, 1]).unwrap();
            let mixed = a
                .index((.., 0..3, ..))
                .subtract(l.multiply(a.index((.., 3..4, ..))).unwrap())
                .unwrap()
                .reshape(&[1, (h * 3 * w) as i32])
                .unwrap();
            let g = gate.add(Array::from_slice(&[1.15f32], &[1])).unwrap();
            let act = g.multiply(g.sigmoid().unwrap()).unwrap();
            let expected = mixed.multiply(&act).unwrap();
            exact(
                &differential_silu_gate(&a, &kg, &gate, 1.15, &stream).unwrap(),
                &expected,
            );
        }
    }

    #[test]
    fn packed_fusions_preserve_standalone_f32_boundaries() {
        let stream = Stream::gpu();
        // Exhaust all BF16 bit patterns in F32 storage, including signed zeros,
        // infinities and NaNs, against the actual two-dispatch activation.
        let values: Vec<f32> = (0..65536u32).map(|b| f32::from_bits(b << 16)).collect();
        let x = Array::from_slice(&values, &[256, 256]);
        let z = Array::from_slice(&[0f32], &[1]);
        let a = x.maximum(&z).unwrap();
        exact(
            &relu_squared(&x, &stream).unwrap(),
            &a.multiply(&a).unwrap(),
        );
        let mut seed = 0x78123u32;
        let mut random = || {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            seed as i32 as f32 / 981234567.
        };
        for width in [1, 31, 384, 1536] {
            let mut v: Vec<f32> = (0..3 * 6 * width).map(|_| random()).collect();
            if width == 1536 {
                for (i, x) in v.iter_mut().enumerate() {
                    // Sweep subnormal through large finite exponents, with
                    // cancellation and signs; exercise intermediate rounding.
                    *x = f32::from_bits((x.to_bits() & 0x807fffff) | ((i as u32 % 254) << 23));
                }
            }
            for (j, value) in [1e10, 1., -1e10, 2., 3., 4.].iter().enumerate() {
                v[j * width] = *value;
            }
            let x = Array::from_slice(&v, &[3, 6, width as i32]);
            let w = Array::from_slice(&(0..18).map(|_| random()).collect::<Vec<_>>(), &[3, 6]);
            let mut expected: Option<Array> = None;
            for k in 0..6 {
                let y = x
                    .index((.., k, ..))
                    .multiply(w.index((.., k..k + 1)))
                    .unwrap();
                expected = Some(match expected {
                    None => y,
                    Some(a) => a.add(&y).unwrap(),
                });
            }
            exact(&weighted_sum6(&x, &w, &stream).unwrap(), &expected.unwrap());
        }
    }
    #[test]
    fn packed_sigmoid_router_matches_sort_and_reduce_bits() {
        let stream = Stream::gpu();
        let mut seed = 9876123u32;
        let mut random = || {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            seed as i32 as f32 / 187654321.
        };
        for case in 0..5 {
            let rows = 129;
            let x = Array::from_slice(
                &(0..rows * 256)
                    .map(|i| match case {
                        0 => random(),
                        1 => 0.,
                        2 => -100.,
                        3 => {
                            if i % 3 == 0 {
                                88.
                            } else {
                                -88.
                            }
                        }
                        _ => ((i % 17) as f32 - 8.) / 7.,
                    })
                    .collect::<Vec<_>>(),
                &[rows, 256],
            );
            let bias = Array::from_slice(
                &(0..256)
                    .map(|i| match case {
                        0 => random() * 0.1,
                        4 => (i % 11) as f32 / 99.,
                        _ => 0.,
                    })
                    .collect::<Vec<_>>(),
                &[256],
            );
            let p = x.sigmoid().unwrap();
            let neg = -&p.add(&bias).unwrap();
            let ids = ops::argsort(&neg).unwrap().index((.., 0..6));
            let p = p.take_along_axis(&ids, -1).unwrap();
            let denom = p
                .sum_axis(-1, true)
                .unwrap()
                .maximum(Array::from_slice(&[6.103515625e-5f32], &[1]))
                .unwrap();
            let expected = p
                .divide(&denom)
                .unwrap()
                .multiply(Array::from_slice(&[2.45f32], &[1]))
                .unwrap();
            let (got_ids, weights) = sigmoid_router256(&x, &bias, 2.45, &stream).unwrap();
            assert!(
                ids.as_slice::<u32>()
                    .iter()
                    .zip(got_ids.as_slice::<i32>())
                    .all(|(a, b)| *a == *b as u32),
                "case={case}"
            );
            exact(&weights, &expected);
        }
    }
}
