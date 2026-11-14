//! Launch fusion for the existing F32 graph, without changing its reductions.
use crate::error::{Error, Result};
use crate::ops::indexing::IndexOp;
use crate::runtime::Math;
use crate::{Array, Dtype, Stream};

const SOURCE: &crate::shader_pack::Source = crate::kernel_source!("gguf/pointwise.metal");

fn dispatch(
    name: &str,
    inputs: &[&Array],
    shape: &[i32],
    args: &[u32; 2],
    threads: usize,
    stream: &Stream,
) -> Result<Array> {
    if inputs
        .iter()
        .any(|a| a.dtype() != Dtype::Float32 || !a.t.buffer_and_layout().1.is_contiguous())
    {
        return Err(Error::msg("GGUF pointwise inputs must be contiguous F32"));
    }
    let rt = stream.runtime();
    let pipeline = rt.compile_with(SOURCE, name, Math::SafeNoLang)?;
    let dims: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
    let size: usize = dims.iter().product();
    let buffer = rt.buffer_for_full_gpu_write(size * 4, "gguf_pointwise")?;
    let out = Array::new(crate::array::Array::from_parts(
        rt,
        buffer,
        &dims,
        Dtype::Float32,
    ));
    let guard = rt.commands.encoder()?;
    let enc = guard.encoder();
    enc.set_pipeline(&pipeline);
    for (slot, a) in inputs.iter().enumerate() {
        let (b, l) = a.t.buffer_and_layout();
        enc.set_input(slot, Some(b), l.offset * 4);
    }
    let (b, l) = out.t.buffer_and_layout();
    enc.set_output(inputs.len(), Some(b), l.offset * 4);
    enc.set_bytes(inputs.len() + 1, args);
    enc.dispatch_threads((threads, 1, 1), (256.min(threads), 1, 1));
    Ok(out)
}

/// Four-tap depthwise convolution followed by the original F32 SiLU chain.
/// `x` includes the three history rows; the caller retains them for rollback.
pub fn conv_silu(x: &Array, w: &Array) -> Result<Array> {
    conv_silu_on(x, w, &Stream::thread_local_or_default())
}
pub fn conv_silu_on(x: &Array, w: &Array, stream: &Stream) -> Result<Array> {
    if x.rank() != 3
        || x.dim(0) != 1
        || x.dim(1) < 4
        || x.dim(2) <= 0
        || w.shape() != [x.dim(2), 4, 1]
    {
        return Err(Error::msg("invalid GGUF convolution shape"));
    }
    let shape = [1, x.dim(1) - 3, x.dim(2)];
    dispatch(
        "lisa_gguf_conv_silu",
        &[x, w],
        &shape,
        &[x.dim(2) as u32, 0],
        (shape[1] * shape[2]) as usize,
        stream,
    )
}

/// Stable softplus/decay and sigmoid, retaining the threshold and math mode.
pub fn gates(alpha: &Array, beta: &Array, dt: &Array, a: &Array) -> Result<(Array, Array)> {
    gates_on(alpha, beta, dt, a, &Stream::thread_local_or_default())
}
pub fn gates_on(
    alpha: &Array,
    beta: &Array,
    dt: &Array,
    a: &Array,
    stream: &Stream,
) -> Result<(Array, Array)> {
    if alpha.rank() != 3
        || alpha.dim(0) != 1
        || alpha.dim(1) <= 0
        || alpha.dim(2) <= 0
        || alpha.shape() != beta.shape()
        || dt.shape() != [alpha.dim(2)]
        || a.shape() != dt.shape()
    {
        return Err(Error::msg("invalid GGUF gate shape"));
    }
    let (t, h) = (alpha.dim(1), alpha.dim(2));
    let out = dispatch(
        "lisa_gguf_gates",
        &[alpha, beta, dt, a],
        &[2, t, h],
        &[h as u32, (t * h) as u32],
        (t * h) as usize,
        stream,
    )?;
    Ok((out.index(0), out.index(1)))
}

/// `x.silu().multiply(y)` with both F32 products kept in their original order.
pub fn silu_mul(x: &Array, y: &Array) -> Result<Array> {
    silu_product_on(x, y, false, &Stream::thread_local_or_default())
}

/// `x.multiply(y.silu())`, preserving operand order as well as rounding.
pub fn mul_silu(x: &Array, y: &Array) -> Result<Array> {
    silu_product_on(x, y, true, &Stream::thread_local_or_default())
}

pub fn silu_product_on(x: &Array, y: &Array, reverse: bool, stream: &Stream) -> Result<Array> {
    let (x, y) = if reverse { (y, x) } else { (x, y) };
    if x.shape() != y.shape() || x.shape().iter().any(|&n| n <= 0) {
        return Err(Error::msg("invalid GGUF SiLU product shape"));
    }
    dispatch(
        "lisa_gguf_silu_mul",
        &[x, y],
        x.shape(),
        &[reverse as u32, 0],
        x.size(),
        stream,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops;

    fn exact(actual: &Array, expected: &Array) -> Result<()> {
        let a = actual.reshape(&[-1])?.to_vec1::<f32>()?;
        let e = expected.reshape(&[-1])?.to_vec1::<f32>()?;
        assert_eq!(a.len(), e.len());
        for (i, (a, e)) in a.iter().zip(e).enumerate() {
            assert!(
                a.to_bits() == e.to_bits() || (a.is_nan() && e.is_nan()),
                "element {i}: {a:?} ({:08x}) vs {e:?} ({:08x})",
                a.to_bits(),
                e.to_bits()
            );
        }
        Ok(())
    }

    fn values(n: usize, seed: u32) -> Vec<f32> {
        let mut rng = seed;
        (0..n)
            .map(|_| {
                rng ^= rng << 13;
                rng ^= rng >> 17;
                rng ^= rng << 5;
                (rng as i32 as f32) * (4.0 / i32::MAX as f32)
            })
            .collect()
    }

    #[test]
    fn fused_conv_matches_separate_f32_chain() -> Result<()> {
        for channels in [8, 10240] {
            let w = Array::from_slice(&values(channels * 4, 12345), &[channels as i32, 4, 1]);
            for t in [1, 2, 3, 7, 16] {
                let full = Array::from_slice(
                    &values((t + 4) * channels, 67890),
                    &[1, (t + 4) as i32, channels as i32],
                );
                // Nonzero storage offset must survive the fusion's bindings.
                let x = full.index((.., 1.., ..));
                let want = ops::conv1d(&x, &w, None, None, None, Some(channels as i32))?.silu()?;
                exact(&conv_silu(&x, &w)?, &want)?;
            }
        }
        Ok(())
    }

    #[test]
    fn fused_gates_and_silu_match_f32_chains() -> Result<()> {
        // Exhaust the finite bf16 inputs represented in F32, then exercise
        // full-mantissa F32 values, threshold neighbors and offset views.
        let mut data: Vec<f32> = (0..=u16::MAX)
            .map(|b| f32::from_bits((b as u32) << 16))
            .filter(|v| v.is_finite())
            .collect();
        data.extend(values(65536, 98891).iter().map(|x| *x * 20.0));
        data.extend([
            20.0,
            f32::from_bits(20f32.to_bits() - 1),
            f32::from_bits(20f32.to_bits() + 1),
            0.0,
            -0.0,
        ]);
        data.splice(0..0, [0.0; 48]);
        data.resize(data.len().div_ceil(48) * 48, 0.0);
        let rows = (data.len() / 48) as i32;
        let input = Array::from_slice(&data, &[1, rows, 48]);
        let x = input.index((.., 1.., ..));
        let y = Array::from_slice(&values(x.size(), 7783), x.shape());
        exact(&silu_mul(&x, &y)?, &x.silu()?.multiply(&y)?)?;
        exact(&mul_silu(&y, &x)?, &y.multiply(&x.silu()?)?)?;
        for bias in [vec![0.0; 48], values(48, 871)] {
            let dt = Array::from_slice(&bias, &[48]);
            let a = Array::from_slice(&values(48, 783), &[48]);
            let alpha = x.add(&dt)?;
            let soft = alpha
                .minimum(Array::from_f32(20.0))?
                .exp()?
                .add(Array::from_f32(1.0))?
                .log()?;
            let soft = ops::r#where(&alpha.gt(Array::from_f32(20.0))?, &alpha, &soft)?;
            let (g, b) = gates(&x, &y, &dt, &a)?;
            exact(&g, &soft.multiply(&a)?.reshape(&[rows - 1, 48])?)?;
            exact(&b, &y.sigmoid()?.reshape(&[rows - 1, 48])?)?;
        }
        Ok(())
    }

    #[test]
    fn fused_pointwise_rejects_invalid_layouts() -> Result<()> {
        let x = Array::from_slice(&[1f32; 48], &[1, 4, 12]);
        let strided = x.transpose_axes(&[0, 2, 1])?;
        assert!(silu_mul(&strided, &strided).is_err());
        assert!(conv_silu(&x, &Array::from_slice(&[1f32; 12], &[12])).is_err());
        assert!(gates(&x, &x, &x, &x).is_err());
        Ok(())
    }
}
