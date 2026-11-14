//! Explicit DSpine projection path; target-model matmul dispatch is unchanged.
use objc2_metal::MTLSize;
use std::sync::OnceLock;

use super::{Device, MLX_UTILS_PREAMBLE, NAX_GEMM_HEADER, Tensor, nax::compile_nax_jit};
use crate::{
    array::{Array, Dtype},
    error::Result,
};

const THIN: &crate::shader_pack::Source = crate::kernel_source!("nax/dspine_thin.metal");
const WIDE: &crate::shader_pack::Source = crate::kernel_source!("common/matmul/dspine_wide.metal");

/// MLX v0.32.4 small-row BF16 linear: A[M,K] @ weight[N,K].
/// M<=15 uses wide GEMV; M=16..64 uses the 2x2, SM=SN=1, KS=4
/// NAX specialization. Both preserve the upstream reduction order.
pub fn dspine_linear_nax(device: &Device, a: &Tensor, weight: &Tensor) -> Result<Tensor> {
    if !device.nax()
        || a.dtype() != Dtype::Bfloat16
        || weight.dtype() != a.dtype()
        || a.rank() != 2
        || weight.rank() != 2
        || a.dim(1) != weight.dim(1)
        || !(1..=64).contains(&a.dim(0))
    {
        crate::bail!("DSpine NAX linear requires BF16 A[1..64,K], weight[N,K] on NAX");
    }
    let a = a.contiguous()?;
    let weight = weight.contiguous()?;
    let (m, k, n) = (a.dim(0), a.dim(1), weight.dim(0));
    if m == 1 {
        // A committed single token uses the ordinary GEMV with the native
        // [N,K] weight layout, avoiding a full-weight transpose copy.
        let rhs = weight.transpose(0, 1)?;
        let out = Array::zeros(device, &[m, n], a.dtype())?;
        let (ab, al) = a.buffer_and_layout();
        let (wb, wl) = rhs.buffer_and_layout();
        super::dense_gemm(
            device,
            (1, m, n, k),
            al.stride(),
            al.offset * 2,
            ab,
            wl.stride(),
            wl.offset * 2,
            wb,
            &out,
            a.dtype(),
        )?;
        return Ok(out);
    }
    // Matmul::eval_gpu tries gemv_wide before the NAX thin specialization.
    if m <= 15 && k % 4 == 0 && n > 1 {
        return wide(device, &a, &weight);
    }
    let name = "dspine_gemm_thin_nax_nt_bf16_um2_un2_sm1_sn1_ks4";
    static SOURCE: OnceLock<String> = OnceLock::new();
    let source = SOURCE.get_or_init(|| format!("{NAX_GEMM_HEADER}\n{THIN}\ninstantiate_kernel(\"{name}\", gemm_thin_nax, bfloat, 2, 2, 1, 1, 4, true)\n"));
    let pipeline = compile_nax_jit(
        device,
        &source,
        name,
        &[
            (100, false),
            (110, false),
            (200, m % 32 == 0),
            (201, n % 32 == 0),
            (202, k % 64 == 0),
        ],
    )?;
    let out_buffer = device.buffer_for_full_gpu_write(m * n * 2, "dspine_linear")?;
    let out = Array::from_parts(device, out_buffer.clone(), &[m, n], a.dtype());
    let (tm, tn) = (m.div_ceil(32), n.div_ceil(32));
    // GEMMParams: six dimensions, two tile counts, three 64-bit batch strides,
    // swizzle, K iterations per partition, batch rank, then ABI padding.
    let mut params = Vec::with_capacity(72);
    for v in [m, n, k, k, k, n, tn, tm] {
        params.extend_from_slice(&(v as i32).to_le_bytes());
    }
    for _ in 0..3 {
        params.extend_from_slice(&0i64.to_le_bytes());
    }
    for v in [0, k.div_ceil(64).div_ceil(4) as i32, 0] {
        params.extend_from_slice(&v.to_le_bytes());
    }
    params.resize(72, 0);
    let guard = device.commands.encoder()?;
    let enc = guard.encoder();
    enc.set_pipeline(&pipeline);
    for (index, tensor) in [(0, &a), (1, &weight)] {
        let (buffer, layout) = tensor.buffer_and_layout();
        enc.set_input(index, Some(buffer), layout.offset * 2);
    }
    enc.set_output(3, Some(&out_buffer), 0);
    enc.set_bytes_directly(4, params.len(), params.as_ptr().cast());
    enc.dispatch_groups_size(
        MTLSize {
            width: tm * tn,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 128,
            height: 1,
            depth: 1,
        },
    );
    Ok(out)
}

fn wide(device: &Device, a: &Tensor, weight: &Tensor) -> Result<Tensor> {
    let (m, k, n) = (a.dim(0), a.dim(1), weight.dim(0));
    let passes = m.div_ceil(5);
    let nv = m.div_ceil(passes);
    let lanes = if passes == 1 || n <= 64 { 32 } else { 16 };
    static SOURCES: [OnceLock<(String, String)>; 10] = [const { OnceLock::new() }; 10];
    let (name, source) = SOURCES[(nv - 1) * 2 + usize::from(lanes == 32)].get_or_init(|| {
        let name = format!("dspine_gemv_wide_bf16_nv{nv}_kl{lanes}");
        let source = format!(
            "{MLX_UTILS_PREAMBLE}\n{WIDE}\n{}",
            super::builtin_template_def(
                &name,
                "gemv_wide",
                &["bfloat16_t".into(), nv.to_string(), lanes.to_string()]
            )
        );
        (name, source)
    });
    let pipeline =
        super::compile_builtin_bool_consts(device, &source, &name, &[(0, false), (1, false)])?;
    let buffer = device.buffer_for_full_gpu_write(m * n * 2, "dspine_wide")?;
    let out = Array::from_parts(device, buffer.clone(), &[m, n], a.dtype());
    let guard = device.commands.encoder()?;
    let enc = guard.encoder();
    enc.set_pipeline(&pipeline);
    for (index, tensor) in [(0, weight), (1, a)] {
        let (buffer, layout) = tensor.buffer_and_layout();
        enc.set_input(index, Some(buffer), layout.offset * 2);
    }
    enc.set_output(3, Some(&buffer), 0);
    for (index, value) in [(4, k), (5, n), (6, m), (7, k), (8, k), (11, 1), (12, 1)] {
        enc.set_bytes(index, &(value as i32));
    }
    enc.set_bytes(13, &0i64);
    enc.set_bytes(14, &0i64);
    enc.dispatch_groups_size(
        MTLSize {
            width: if n >= 65536 { 1 } else { passes },
            height: n.div_ceil(4),
            depth: 1,
        },
        MTLSize {
            width: 32,
            height: lanes / 8,
            depth: 1,
        },
    );
    Ok(out)
}

/// MLX 0.32.4 affine draft schedule. The single-row input sum promotes BF16
/// operands before addition, and split-K partials remain F32 until the final
/// sequential accumulation. This is deliberately separate from target kernels.
pub fn dspine_affine(
    device: &Device,
    x: &Tensor,
    w: &Tensor,
    scales: &Tensor,
    biases: &Tensor,
) -> Result<Tensor> {
    if x.rank() != 2 || w.rank() != 2 {
        crate::bail!("DSpine affine expects matrices");
    }
    let (m, k) = x.dims2();
    let (n, _) = w.dims2();
    if m == 0
        || n == 0
        || k == 0
        || x.dtype() != Dtype::Bfloat16
        || w.dtype() != Dtype::Uint32
        || scales.dtype() != Dtype::Bfloat16
        || biases.dtype() != Dtype::Bfloat16
        || k % 512 != 0
        || n % 32 != 0
        || w.dim(1) * 8 != k
        || scales.dims() != [n, k / 64]
        || biases.dims() != scales.dims()
    {
        crate::bail!("unsupported DSpine affine layout");
    }
    let mut split = (512 / (n.div_ceil(32) * m.div_ceil(32))).max(1).min(k / 64);
    while split > 1 && k % (split * 64) != 0 {
        split -= 1;
    }
    if m > 1 && m < super::qmv_batch_limit(k, n) {
        return super::qmv_wide(device, x, w, scales, biases, 64, 4);
    }
    if m > 1 && split <= 1 {
        return if device.nax() {
            super::qmm_nax(device, x, w, scales, biases, 64, 4)
        } else {
            super::affine_qmm(device, x, w, scales, biases, 64, 4)
        };
    }
    const QUANT: &crate::shader_pack::Source =
        crate::kernel_source!("common/matmul/dspine_quantized.metal");
    static SOURCE: OnceLock<String> = OnceLock::new();
    let source = SOURCE.get_or_init(|| {
        use super::{MLX_GEMM_PREAMBLE, MLX_QUANTIZED_UTILS_PREAMBLE};
        let qmv = super::builtin_template_def("dspine_affine_qmv", "affine_qmv_fast",
            &["bfloat16_t", "64", "4", "false", "false", "false", "4"].map(String::from));
        let qmm = super::builtin_template_def("dspine_affine_splitk", "affine_qmm_t_splitk",
            &["bfloat16_t", "64", "4", "true"].map(String::from));
        let sum = super::builtin_template_def("dspine_affine_accum", "gemm_splitk_accum",
            &["float", "bfloat16_t"].map(String::from));
        format!("{MLX_UTILS_PREAMBLE}{MLX_GEMM_PREAMBLE}{MLX_QUANTIZED_UTILS_PREAMBLE}{QUANT}{qmv}{qmm}{sum}")
    });
    let name = if m == 1 {
        "dspine_affine_qmv"
    } else {
        "dspine_affine_splitk"
    };
    let pipeline = super::compile_builtin(device, source, name)?;
    let out_dtype = if m == 1 {
        Dtype::Bfloat16
    } else {
        Dtype::Float32
    };
    let partitions = if m == 1 { 1 } else { split };
    let buffer = device
        .buffer_for_full_gpu_write(partitions * m * n * out_dtype.size_of(), "dspine_affine")?;
    let out = Array::from_parts(device, buffer.clone(), &[partitions, m, n], out_dtype);
    let guard = device.commands.encoder()?;
    let enc = guard.encoder();
    enc.set_pipeline(&pipeline);
    for (slot, tensor) in [w, scales, biases, x].into_iter().enumerate() {
        let (b, l) = tensor.buffer_and_layout();
        if !l.is_contiguous() {
            crate::bail!("DSpine affine expects contiguous inputs");
        }
        enc.set_input(slot, Some(b), l.offset * tensor.dtype().size_of());
    }
    enc.set_output(4, Some(&buffer), 0);
    enc.set_bytes(5, &(k as i32));
    enc.set_bytes(6, &(n as i32));
    if m == 1 {
        enc.dispatch_groups((1, n / 8, 1), (32, 2, 1));
        return out.reshape(&[m, n]);
    }
    enc.set_bytes(7, &(m as i32));
    enc.set_bytes(8, &((k / split) as i32));
    enc.set_bytes(9, &((m * n) as i32));
    enc.dispatch_groups((n.div_ceil(32), m.div_ceil(32), split), (32, 2, 2));
    drop(guard);
    let accum = super::compile_builtin(device, source, "dspine_affine_accum")?;
    let result_buffer = device.buffer_for_full_gpu_write(m * n * 2, "dspine_affine_accum")?;
    let result = Array::from_parts(device, result_buffer.clone(), &[m, n], Dtype::Bfloat16);
    let guard = device.commands.encoder()?;
    let enc = guard.encoder();
    enc.set_pipeline(&accum);
    enc.set_input(0, Some(&buffer), 0);
    enc.set_output(1, Some(&result_buffer), 0);
    enc.set_bytes(2, &(split as i32));
    enc.set_bytes(3, &((m * n) as i32));
    enc.set_bytes(4, &(n as i32));
    enc.dispatch_threads((n, m, 1), (32, 1, 1));
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dspine_projections_match_exact_dyadic_oracle() -> Result<()> {
        use crate::ops::{Array as A, Stream};
        let rt = Stream::thread_local_or_default().runtime().clone();
        if !rt.nax() {
            return Ok(());
        }
        // Small dyadic values keep every FP32 partial sum exact, so a CPU
        // f64 dot product is an independent bitwise oracle. Exercise both
        // kernels, K/N tails, alignment constants and offset bindings.
        for m in [1, 2, 5, 7, 15, 16, 32, 63, 64] {
            for (k, n) in [(128, 96), (260, 37)] {
                let av: Vec<f32> = (0..(m + 1) * k)
                    .map(|i| ((i * 13 + 5) % 31) as f32 / 32.0 - 0.5)
                    .collect();
                let wv: Vec<f32> = (0..(n + 1) * k)
                    .map(|i| ((i * 7 + 3) % 29) as f32 / 32.0 - 0.5)
                    .collect();
                let a =
                    A::from_slice(&av, &[(m + 1) as i32, k as i32]).as_dtype(Dtype::Bfloat16)?;
                let w =
                    A::from_slice(&wv, &[(n + 1) as i32, k as i32]).as_dtype(Dtype::Bfloat16)?;
                let a = a.t.narrow(0, 1, m)?;
                let w = w.t.narrow(0, 1, n)?;
                let actual = dspine_linear_nax(&rt, &a, &w)?.to_vec::<u16>()?;
                let mut expected = Vec::new();
                for row in 0..m {
                    for col in 0..n {
                        let sum: f64 = (0..k)
                            .map(|i| av[(row + 1) * k + i] as f64 * wv[(col + 1) * k + i] as f64)
                            .sum();
                        expected.push(half::bf16::from_f64(sum).to_bits());
                    }
                }
                assert_eq!(actual, expected, "M={m}, K={k}, N={n}");
            }
        }
        Ok(())
    }
}
