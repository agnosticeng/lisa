//! Packed GGML projections. The payload and accumulation order are retained.
use crate::error::{Error, Result};
use crate::runtime::{ConstVal, Math};
use crate::{Array, Dtype, Stream};
use lisa_backend::OrderedBlockEncoding;
const SOURCE: &crate::shader_pack::Source = crate::kernel_source!("gguf/matvec.metal");
mod pointwise;
pub use pointwise::{
    conv_silu, conv_silu_on, gates, gates_on, mul_silu, silu_mul, silu_product_on,
};

/// Block encoding, bytes per block, elements per block, output rows per SIMD,
/// SIMD groups, and shared scratch. Only explicitly ported formats are accepted.
fn geometry(kind: OrderedBlockEncoding) -> Result<(&'static str, i32, i32, i32, usize, usize)> {
    Ok(match kind {
        OrderedBlockEncoding::Q8_0 => ("q8_0", 34, 32, 2, 4, 256),
        OrderedBlockEncoding::Q2K => ("q2_K", 84, 256, 4, 2, 16),
        OrderedBlockEncoding::Q3K => ("q3_K", 110, 256, 2, 2, 16),
        OrderedBlockEncoding::Q4K => ("q4_K", 144, 256, 2, 2, 16),
        OrderedBlockEncoding::Q5K => ("q5_K", 176, 256, 1, 2, 16),
        OrderedBlockEncoding::Q6K => ("q6_K", 210, 256, 2, 2, 16),
        OrderedBlockEncoding::Iq2Xxs => ("iq2_xxs", 66, 256, 4, 2, 2176),
        OrderedBlockEncoding::Iq2Xs => ("iq2_xs", 74, 256, 4, 2, 4224),
        OrderedBlockEncoding::Iq3Xxs => ("iq3_xxs", 98, 256, 4, 2, 1152),
        OrderedBlockEncoding::Iq1S => ("iq1_s", 50, 256, 4, 2, 16),
        OrderedBlockEncoding::Iq3S => ("iq3_s", 110, 256, 4, 2, 2048),
        OrderedBlockEncoding::Iq2S => ("iq2_s", 82, 256, 4, 2, 16),
        OrderedBlockEncoding::Iq4Xs => ("iq4_xs", 136, 256, 2, 2, 128),
        OrderedBlockEncoding::Iq1M => ("iq1_m", 56, 256, 4, 2, 16),
    })
}

#[repr(C)]
struct Args {
    ne00: i32,
    ne01: i32,
    ne02: i32,
    nb00: u64,
    nb01: u64,
    nb02: u64,
    nb03: u64,
    ne10: i32,
    ne11: i32,
    ne12: i32,
    nb10: u64,
    nb11: u64,
    nb12: u64,
    nb13: u64,
    ne0: i32,
    ne1: i32,
    nr0: i32,
    r2: i16,
    r3: i16,
}

/// Original packed `[output, row_bytes]` times contiguous F32 `[tokens, input]`.
/// Independent rows use the same reduction in decode and prefill.
pub fn matvec(
    kind: OrderedBlockEncoding,
    weights: &Array,
    input: &Array,
    stream: &Stream,
) -> Result<Array> {
    let (name, bytes, block, rows_per_simd, groups, scratch) = geometry(kind)?;
    let (_, wl) = weights.t.buffer_and_layout();
    let (_, xl) = input.t.buffer_and_layout();
    if weights.rank() != 2
        || weights.dtype() != Dtype::Uint8
        || !wl.is_contiguous()
        || wl.offset % 2 != 0
        || weights.dim(0) <= 0
        || weights.dim(0) % (rows_per_simd * groups as i32) != 0
        || weights.dim(1) <= 0
        || weights.dim(1) % bytes != 0
        || input.rank() != 2
        || input.dtype() != Dtype::Float32
        || !xl.is_contiguous()
        || input.dim(0) <= 0
        || input.dim(0) > 65535
        || input.dim(1) != weights.dim(1) / bytes * block
    {
        return Err(Error::msg("unsupported packed GGML matvec geometry"));
    }
    let k = input.dim(1);
    let n = weights.dim(0);
    let m = input.dim(0);
    let row = weights.dim(1) as u64;
    let args = Args {
        ne00: k,
        ne01: n,
        ne02: 1,
        nb00: bytes as u64,
        nb01: row,
        nb02: row * n as u64,
        nb03: row * n as u64,
        ne10: k,
        ne11: m,
        ne12: 1,
        nb10: 4,
        nb11: k as u64 * 4,
        nb12: k as u64 * m as u64 * 4,
        nb13: k as u64 * m as u64 * 4,
        ne0: n,
        ne1: m,
        nr0: rows_per_simd,
        r2: 1,
        r3: 1,
    };
    let pipeline = stream.runtime().compile_with_constants(
        SOURCE,
        &format!("lisa_packed_{name}"),
        Math::CompilerDefault,
        &[
            (600, ConstVal::Short(groups as i16)),
            (602, ConstVal::Short(1)),
            (603, ConstVal::Short(1)),
            (604, ConstVal::Short(1)),
            (605, ConstVal::Bool(false)),
        ],
    )?;
    weights.eval_enqueue()?;
    input.eval_enqueue()?;
    let runtime = stream.runtime();
    let buffer = runtime.buffer_for_full_gpu_write(m as usize * n as usize * 4, "gguf_matvec")?;
    let out = Array::new(crate::array::Array::from_parts(
        runtime,
        buffer,
        &[m as usize, n as usize],
        Dtype::Float32,
    ));
    let guard = stream.runtime().commands.encoder()?;
    let e = guard.encoder();
    e.set_pipeline(&pipeline);
    e.set_bytes(0, &args);
    for (slot, a) in [(1, weights), (2, input)] {
        let (b, l) = a.t.buffer_and_layout();
        e.set_input(slot, Some(b), l.offset * a.dtype().size_of());
    }
    let (b, l) = out.t.buffer_and_layout();
    e.set_output(3, Some(b), l.offset * 4);
    e.set_threadgroup_memory_length(0, scratch);
    e.dispatch_groups(
        (
            n as usize
                / (rows_per_simd as usize
                    * if kind == OrderedBlockEncoding::Q8_0 {
                        1
                    } else {
                        groups
                    }),
            m as usize,
            1,
        ),
        (32, groups, 1),
    );
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn packed_formats_execute_zero_rows() -> Result<()> {
        let stream = Stream::thread_local_or_default();
        for kind in OrderedBlockEncoding::ALL {
            let (_, bytes, block, _, _, _) = geometry(kind)?;
            let w = Array::from_slice(
                &vec![0u8; (8 * 2048 / block * bytes) as usize],
                &[8, 2048 / block * bytes],
            );
            let x = Array::from_slice(&vec![1f32; 2048 * 2], &[2, 2048]);
            let y = matvec(kind, &w, &x, &stream)?;
            y.eval()?;
            assert!(
                y.as_slice::<f32>().iter().all(|v| *v == 0.0),
                "type {kind:?}"
            );
        }
        Ok(())
    }
}

#[cfg(test)]
mod numerical_tests {
    use super::*;
    include!("blockwise/oracle.rs");
    #[test]
    fn mixed_projections_match_independent_ggml_cpu() -> Result<()> {
        let stream = Stream::thread_local_or_default();
        for &(kind, width, expected) in ORACLE {
            let (_, bytes, block, _, _, _) = geometry(kind)?;
            let mut rng = 0x12345678u32;
            let mut packed = vec![0u8; (8 * width / block * bytes) as usize];
            for (b, p) in packed.chunks_exact_mut(bytes as usize).enumerate() {
                for byte in p.iter_mut() {
                    rng ^= rng << 13;
                    rng ^= rng >> 17;
                    rng ^= rng << 5;
                    *byte = (rng >> 24) as u8;
                }
                let scale = 0x2800u16 + (b % 8) as u16 * 0x200;
                let at = if kind == OrderedBlockEncoding::Q2K {
                    80
                } else if kind == OrderedBlockEncoding::Q3K || kind == OrderedBlockEncoding::Q6K {
                    bytes as usize - 2
                } else {
                    0
                };
                if kind == OrderedBlockEncoding::Iq1M {
                    for j in 0..4 {
                        let q = 48 + j * 2;
                        let v = u16::from_le_bytes([p[q], p[q + 1]]);
                        p[q..q + 2].copy_from_slice(
                            &((v & 0xfff) | (((scale >> (4 * j)) & 15) << 12)).to_le_bytes(),
                        );
                    }
                } else {
                    p[at..at + 2].copy_from_slice(&scale.to_le_bytes());
                }
                if [
                    OrderedBlockEncoding::Q2K,
                    OrderedBlockEncoding::Q4K,
                    OrderedBlockEncoding::Q5K,
                ]
                .contains(&kind)
                {
                    let at = if kind == OrderedBlockEncoding::Q2K {
                        82
                    } else {
                        2
                    };
                    p[at..at + 2].copy_from_slice(&0x2400u16.to_le_bytes());
                }
            }
            let w = Array::from_slice(&packed, &[8, width / block * bytes]);
            let values: Vec<f32> = (0..2)
                .flat_map(|t| (0..width).map(move |i| ((i * 13 + t * 7) % 67 - 33) as f32 / 32.0))
                .collect();
            let x = Array::from_slice(&values, &[2, width]);
            let y = matvec(kind, &w, &x, &stream)?;
            y.eval()?;
            for (i, (&actual, &want)) in y.as_slice::<f32>().iter().zip(&expected).enumerate() {
                assert!(
                    (actual - want).abs() <= 0.002 + want.abs() * 0.00002,
                    "kind={kind:?} K={width} row={i}: {actual} vs {want}"
                );
            }
        }
        Ok(())
    }
}

/// Original F32 recurrent update for Qwen's 128-channel GDN heads.
/// Input rows are contiguous [tokens, heads, width], state [heads, width, width].
pub fn gdn(
    q: &Array,
    k: &Array,
    v: &Array,
    g: &Array,
    b: &Array,
    state: &Array,
    stream: &Stream,
) -> Result<(Array, Array)> {
    const SOURCE: &crate::shader_pack::Source = crate::kernel_source!("gguf/gdn.metal");
    if q.rank() != 3
        || k.shape() != q.shape()
        || v.rank() != 3
        || v.dim(0) != q.dim(0)
        || q.dim(2) != 128
        || v.dim(2) != 128
        || v.dim(1) % q.dim(1) != 0
        || q.dim(0) < 1
        || q.dim(0) > 32767
        || g.shape() != [q.dim(0), v.dim(1)]
        || b.shape() != g.shape()
        || state.shape() != [v.dim(1), 128, 128]
        || [q, k, v, g, b, state]
            .iter()
            .any(|a| a.dtype() != Dtype::Float32 || !a.t.buffer_and_layout().1.is_contiguous())
    {
        return Err(Error::msg("unsupported GGUF GDN layout"));
    }
    #[repr(C)]
    #[derive(Default)]
    struct GdnArgs {
        ne00: i32,
        ne01: i32,
        ne02: i32,
        ne03: i32,
        nb00: u64,
        nb01: u64,
        nb02: u64,
        nb03: u64,
        ne10: i32,
        ne11: i32,
        ne12: i32,
        ne13: i32,
        nb10: u64,
        nb11: u64,
        nb12: u64,
        nb13: u64,
        ne20: i32,
        ne21: i32,
        ne22: i32,
        ne23: i32,
        nb20: u64,
        nb21: u64,
        nb22: u64,
        nb23: u64,
        ns02: i32,
        ns12: i32,
        ns22: i32,
        ne0: i32,
        ne1: i32,
        ne2: i32,
        ne3: i32,
        nb0: u64,
        nb1: u64,
        nb2: u64,
        nb3: u64,
        nb_out: u64,
    }
    let t = q.dim(0);
    let h = v.dim(1);
    let kh = q.dim(1);
    let d = 128;
    let args = GdnArgs {
        ne00: d,
        ne01: kh,
        ne02: t,
        ne03: 1,
        nb00: 4,
        nb01: 512,
        nb02: (kh * 512) as u64,
        nb03: (kh * t * 512) as u64,
        ne10: d,
        ne11: kh,
        ne12: t,
        ne13: 1,
        nb10: 4,
        nb11: 512,
        nb12: (kh * 512) as u64,
        nb13: (kh * t * 512) as u64,
        ne20: d,
        ne21: h,
        ne22: t,
        ne23: 1,
        nb20: 4,
        nb21: 512,
        nb22: (h * 512) as u64,
        nb23: (h * t * 512) as u64,
        ns02: kh * d,
        ns12: kh * d,
        ns22: h * d,
        ..Default::default()
    };
    let runtime = stream.runtime();
    let attn = (t * h * d) as usize;
    let states = (h * d * d) as usize;
    let buffer = runtime.buffer_for_full_gpu_write((attn + states) * 4, "gguf_gdn")?;
    let y = Array::from_buffer_offset(buffer.clone(), 0, &[t, h, d], Dtype::Float32)?;
    let next = Array::from_buffer_offset(buffer, attn * 4, &[h, d, d], Dtype::Float32)?;
    let p = runtime.compile_with_constants(
        SOURCE,
        "kernel_gated_delta_net_f32_4",
        Math::CompilerDefault,
        &[
            (1700, ConstVal::Short(128)),
            (1701, ConstVal::Short(1)),
            (1702, ConstVal::Short(1)),
        ],
    )?;
    for a in [q, k, v, g, b, state] {
        a.eval_enqueue()?;
    }
    let guard = runtime.commands.encoder()?;
    let e = guard.encoder();
    e.set_pipeline(&p);
    e.set_bytes(0, &args);
    for (i, a) in [q, k, v, g, b, state].into_iter().enumerate() {
        let (b, l) = a.t.buffer_and_layout();
        e.set_input(i + 1, Some(b), l.offset * 4);
    }
    let (buf, l) = y.t.buffer_and_layout();
    e.set_output(7, Some(buf), l.offset * 4);
    e.set_output(8, Some(buf), l.offset * 4);
    e.dispatch_groups((32, h as usize, 1), (32, 4, 1));
    Ok((y, next))
}

#[cfg(test)]
mod recurrent_tests {
    use super::*;
    use crate::ops::indexing::IndexOp;
    #[test]
    fn recurrence_matches_scalar_reference_and_serial_rows() -> Result<()> {
        let (t, kh, h, d) = (3, 2, 4, 128);
        let vals = |n: usize, salt: usize, scale: f32| -> Vec<f32> {
            (0..n)
                .map(|i| ((i * 17 + salt) % 79) as f32 * scale - 39.0 * scale)
                .collect()
        };
        let qv = vals(t * kh * d, 1, 0.002);
        let kv = vals(t * kh * d, 9, 0.002);
        let vv = vals(t * h * d, 4, 0.025);
        let sv = vals(h * d * d, 11, 0.001);
        let gv = vec![-0.15f32; t * h];
        let bv = vec![0.55f32; t * h];
        let q = Array::from_slice(&qv, &[t as i32, kh as i32, d as i32]);
        let k = Array::from_slice(&kv, q.shape());
        let v = Array::from_slice(&vv, &[t as i32, h as i32, d as i32]);
        let g = Array::from_slice(&gv, &[t as i32, h as i32]);
        let b = Array::from_slice(&bv, g.shape());
        let s = Array::from_slice(&sv, &[h as i32, d as i32, d as i32]);
        let stream = Stream::thread_local_or_default();
        let (y, next) = gdn(&q, &k, &v, &g, &b, &s, &stream)?;
        y.eval()?;
        next.eval()?;
        let mut oracle: Vec<f64> = sv.iter().map(|&v| v as f64).collect();
        let mut expected = Vec::new();
        for ti in 0..t {
            for hi in 0..h {
                for row in 0..d {
                    let head = ti * kh * d + (hi % kh) * d;
                    let base = hi * d * d + row * d;
                    let decay = (-0.15f32).exp() as f64;
                    let mut sk = 0.0;
                    for j in 0..d {
                        oracle[base + j] *= decay;
                        sk += oracle[base + j] * kv[head + j] as f64;
                    }
                    let delta = (vv[(ti * h + hi) * d + row] as f64 - sk) * 0.55f32 as f64;
                    let mut value = 0.0;
                    for j in 0..d {
                        oracle[base + j] += kv[head + j] as f64 * delta;
                        value += oracle[base + j] * qv[head + j] as f64;
                    }
                    expected.push((value / (d as f64).sqrt()) as f32);
                }
            }
        }
        for (&a, &b) in y.as_slice::<f32>().iter().zip(&expected) {
            assert!((a - b).abs() < 1e-6, "{a} vs {b}");
        }
        for (&a, &b) in next.as_slice::<f32>().iter().zip(&oracle) {
            assert!((a as f64 - b).abs() < 1e-6);
        }
        let mut state = s;
        for i in 0..t as i32 {
            let (single, new) = gdn(
                &q.index((i..i + 1, .., ..)).contiguous()?,
                &k.index((i..i + 1, .., ..)).contiguous()?,
                &v.index((i..i + 1, .., ..)).contiguous()?,
                &g.index((i..i + 1, ..)).contiguous()?,
                &b.index((i..i + 1, ..)).contiguous()?,
                &state,
                &stream,
            )?;
            single.eval()?;
            assert_eq!(
                single.as_slice::<f32>(),
                &y.as_slice::<f32>()[i as usize * h * d..(i as usize + 1) * h * d]
            );
            state = new;
        }
        state.eval()?;
        assert_eq!(state.as_slice::<f32>(), next.as_slice::<f32>());
        Ok(())
    }
}
