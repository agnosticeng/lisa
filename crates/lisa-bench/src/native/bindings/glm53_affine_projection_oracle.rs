#![cfg(test)]
use crate::core::quant::QuantizedLinear;
use crate::core::quant::QuantizedLinearExecution;
use lisa_metal::{Array, Dtype};
pub fn original(w: &QuantizedLinear, x: &Array, native: bool) -> lisa_metal::error::Result<Array> {
    let shape = x.shape();
    let rows: i32 = shape[..shape.len() - 1].iter().product::<i32>().max(1);
    let k = x.dim(-1) as usize;
    let n = w.dims_out();
    let mpp_override = Some(native);
    if mpp_override != Some(false)
        && w.bits == 8
        && w.group_size == 64
        && x.dtype() == Dtype::Float32
        && matches!(w.scales.dtype(), Dtype::Float32 | Dtype::Bfloat16)
        && w.biases.dtype() == w.scales.dtype()
        && w.weight.dtype() == Dtype::Uint32
        && k.is_multiple_of(64)
        && w.weight.shape()[1] as usize * 4 == k
    {
        let flat = x.reshape(&[rows, k as i32])?.contiguous()?;
        let output = if rows > 16 {
            Array::new(lisa_metal::jit::affine_q8_prefill(
                x.t.device(),
                &flat.t,
                &w.weight.t,
                &w.scales.contiguous()?.t,
                &w.biases.contiguous()?.t,
            )?)
        } else if rows == 1 || k.is_multiple_of(128) {
            Array::new(lisa_metal::jit::affine_q8_tiny_native(
                x.t.device(),
                &flat.t,
                &w.weight.t,
                &w.scales.contiguous()?.t,
                &w.biases.contiguous()?.t,
            )?)
        } else {
            lisa_metal::kernels::sparse::kernels::affine_q8_mv_ds4(
                &w.weight,
                &w.scales,
                &w.biases,
                &flat,
                &lisa_metal::Stream::thread_local_or_default(),
            )
            .ok_or_else(|| {
                lisa_metal::error::Error::msg(
                    "GLM-5.3 DS4 affine-Q8 prefill kernel declined a supported shape",
                )
            })?
        };
        return output.reshape(&[&shape[..shape.len() - 1], &[n as i32]].concat());
    }
    w.forward_serial_rows(x)
}
