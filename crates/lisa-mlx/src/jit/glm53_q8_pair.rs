//! Native DS4 fused Q8 gate/up and SwiGLU for one to sixteen rows.
use objc2_metal::MTLSize;

use super::{DType, Device, Tensor};
use crate::array::Array;
use crate::error::Result;
use crate::runtime::{ConstVal, Math};

const EXT_SOURCE: &crate::shader_pack::Source =
    crate::kernel_source!("common/matmul/glm53_affine_q8_ext_small_native.metal");
const PAIR_SOURCE: &crate::shader_pack::Source =
    crate::kernel_source!("common/matmul/glm53_affine_q8_pair_native.metal");

#[allow(clippy::too_many_arguments)]
pub fn glm53_affine_q8_gate_up_native(
    device: &Device,
    x: &Tensor,
    gate_w: &Tensor,
    gate_s: &Tensor,
    gate_b: &Tensor,
    up_w: &Tensor,
    up_s: &Tensor,
    up_b: &Tensor,
    limit: f32,
) -> Result<(Tensor, Tensor, Tensor)> {
    let (gate, up, mid) = gate_up_impl(
        device, x, gate_w, gate_s, gate_b, up_w, up_s, up_b, limit, true,
    )?;
    Ok((
        gate.expect("projection output"),
        up.expect("projection output"),
        mid,
    ))
}

/// Production activation-only path: retain the exact paired arithmetic and
/// omit diagnostic gate/up buffers and stores.
#[allow(clippy::too_many_arguments)]
pub fn glm53_affine_q8_swiglu_native(
    device: &Device,
    x: &Tensor,
    gate_w: &Tensor,
    gate_s: &Tensor,
    gate_b: &Tensor,
    up_w: &Tensor,
    up_s: &Tensor,
    up_b: &Tensor,
    limit: f32,
) -> Result<Tensor> {
    Ok(gate_up_impl(
        device, x, gate_w, gate_s, gate_b, up_w, up_s, up_b, limit, false,
    )?
    .2)
}

#[allow(clippy::too_many_arguments)]
fn gate_up_impl(
    device: &Device,
    x: &Tensor,
    gate_w: &Tensor,
    gate_s: &Tensor,
    gate_b: &Tensor,
    up_w: &Tensor,
    up_s: &Tensor,
    up_b: &Tensor,
    limit: f32,
    emit_projections: bool,
) -> Result<(Option<Tensor>, Option<Tensor>, Tensor)> {
    if [x, gate_w, gate_s, gate_b, up_w, up_s, up_b]
        .iter()
        .any(|value| value.rank() != 2)
    {
        crate::bail!("native Q8 gate/up requires two-dimensional inputs");
    }
    let (rows, k) = x.dims2();
    let (n, packed_k) = gate_w.dims2();
    if !(1..=16).contains(&rows)
        || k == 0
        || k % 128 != 0
        || n == 0
        || n % 2 != 0
        || x.dtype() != DType::F32
        || gate_w.dtype() != DType::U32
        || up_w.dtype() != DType::U32
        || packed_k.checked_mul(4) != Some(k)
        || up_w.dims2() != (n, packed_k)
        || !matches!(gate_s.dtype(), DType::F32 | DType::BF16)
        || !limit.is_finite()
        || limit < 0.0
        || [rows, k, n]
            .into_iter()
            .any(|value| value > i32::MAX as usize)
    {
        crate::bail!("unsupported native Q8 gate/up geometry or dtype");
    }
    for value in [gate_s, gate_b, up_s, up_b] {
        if value.dtype() != gate_s.dtype() || value.dims2() != (n, k / 64) {
            crate::bail!("native Q8 gate/up metadata must share dtype and [N,K/64] shape");
        }
    }
    for value in [x, gate_w, gate_s, gate_b, up_w, up_s, up_b] {
        if !value.buffer_and_layout().1.is_contiguous() {
            crate::bail!("native Q8 gate/up input must be contiguous");
        }
    }
    static SOURCE: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    let source = SOURCE.get_or_init(|| format!("{EXT_SOURCE}\n{PAIR_SOURCE}"));
    let r1ptg = match rows {
        2 => 2,
        3 | 6 => 3,
        5 => 5,
        _ => 4,
    };
    let nxpsg = if k % 256 == 0 && rows == 2 { 16 } else { 8 };
    let nsg = if rows == 1 { 4 } else { 2 };
    let name = if rows == 1 {
        "glm53_affine_q8_pair_native_m1".to_owned()
    } else {
        format!("glm53_affine_q8_pair_native_r{r1ptg}")
    };
    let mut constants = if rows == 1 {
        Vec::new()
    } else {
        vec![(0, ConstVal::Int(nsg)), (1, ConstVal::Int(nxpsg))]
    };
    constants.push((2, ConstVal::Bool(emit_projections)));
    let pipeline =
        device.compile_with_constants(source, &name, Math::CompilerDefault, &constants)?;
    let bytes = rows
        .checked_mul(n)
        .and_then(|count| count.checked_mul(4))
        .ok_or_else(|| crate::error::Error::msg("native Q8 gate/up output size overflow"))?;
    let gate_buffer = emit_projections
        .then(|| device.buffer_for_full_gpu_write(bytes, "glm53_q8_native_gate"))
        .transpose()?;
    let up_buffer = emit_projections
        .then(|| device.buffer_for_full_gpu_write(bytes, "glm53_q8_native_up"))
        .transpose()?;
    let mid_buffer = device.buffer_for_full_gpu_write(bytes, "glm53_q8_native_mid")?;
    let gate = gate_buffer
        .as_ref()
        .map(|buffer| Array::from_parts(device, buffer.clone(), &[rows, n], DType::F32));
    let up = up_buffer
        .as_ref()
        .map(|buffer| Array::from_parts(device, buffer.clone(), &[rows, n], DType::F32));
    let mid = Array::from_parts(device, mid_buffer.clone(), &[rows, n], DType::F32);
    let guard = device.commands.encoder()?;
    let enc = guard.encoder();
    enc.set_pipeline(&pipeline);
    enc.set_bytes(
        0,
        &[
            k as u32,
            n as u32,
            rows as u32,
            u32::from(gate_s.dtype() == DType::BF16),
        ],
    );
    for (slot, value) in [
        (1, gate_w),
        (2, up_w),
        (3, x),
        (8, gate_s),
        (9, gate_b),
        (10, up_s),
        (11, up_b),
    ] {
        let (buffer, layout) = value.buffer_and_layout();
        enc.set_input(slot, Some(buffer), layout.offset * value.dtype().size_of());
    }
    for (slot, buffer) in [(4, gate_buffer.as_ref()), (5, up_buffer.as_ref())] {
        if let Some(buffer) = buffer {
            enc.set_output(slot, Some(buffer), 0);
        } else {
            // The function constant removes these diagnostic writes entirely.
            enc.set_input(slot, None, 0);
        }
    }
    enc.set_output(6, Some(&mid_buffer), 0);
    enc.set_bytes(7, &limit);
    if rows == 1 {
        enc.set_threadgroup_memory_length(0, 2 * 2 * 32 * 4);
    }
    enc.dispatch_groups_size(
        MTLSize {
            width: n.div_ceil(if rows == 1 {
                2
            } else {
                ((32 / nxpsg) * nsg) as usize
            }),
            height: if rows == 1 { 1 } else { rows.div_ceil(r1ptg) },
            depth: 1,
        },
        MTLSize {
            width: 32,
            height: nsg as usize,
            depth: 1,
        },
    );
    Ok((gate, up, mid))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::Array as OpArray;

    #[test]
    fn q8_activation_only_matches_full_projection_outputs() -> Result<()> {
        for metadata in [DType::F32, DType::BF16] {
            for rows in [1, 2, 3, 5, 6, 8, 16] {
                for k in [128, 256, 512, 4096] {
                    let n = if k == 4096 {
                        if rows <= 2 { 12288 } else { 2048 }
                    } else {
                        18
                    };
                    let packed = (0..n * k / 4)
                        .map(|i| (i as u32).wrapping_mul(0x9e3779b9))
                        .collect::<Vec<_>>();
                    let w = OpArray::from_slice(&packed, &[n, k / 4]);
                    let up = OpArray::from_slice(
                        &packed.iter().map(|v| !v).collect::<Vec<_>>(),
                        &[n, k / 4],
                    );
                    let scales = (0..n * k / 64)
                        .map(|i| 0.0017 + (i % 11) as f32 * 0.0003)
                        .collect::<Vec<_>>();
                    let biases = (0..n * k / 64)
                        .map(|i| -0.27 + (i % 7) as f32 * 0.013)
                        .collect::<Vec<_>>();
                    let s = OpArray::from_slice(&scales, &[n, k / 64]).as_dtype(metadata)?;
                    let b = OpArray::from_slice(&biases, &[n, k / 64]).as_dtype(metadata)?;
                    let input = (0..rows * k)
                        .map(|i| ((i * 29 % 127) as f32 - 63.0) * 0.07)
                        .collect::<Vec<_>>();
                    let x = OpArray::from_slice(&input, &[rows, k]);
                    for array in [&x, &w, &up, &s, &b] {
                        array.eval_enqueue()?;
                    }
                    for limit in [0.0, 0.1, 10.0] {
                        let (_, _, expected) = glm53_affine_q8_gate_up_native(
                            x.t.device(),
                            &x.t,
                            &w.t,
                            &s.t,
                            &b.t,
                            &up.t,
                            &s.t,
                            &b.t,
                            limit,
                        )?;
                        let actual = glm53_affine_q8_swiglu_native(
                            x.t.device(),
                            &x.t,
                            &w.t,
                            &s.t,
                            &b.t,
                            &up.t,
                            &s.t,
                            &b.t,
                            limit,
                        )?;
                        assert_eq!(
                            actual
                                .to_vec::<f32>()?
                                .iter()
                                .map(|v| v.to_bits())
                                .collect::<Vec<_>>(),
                            expected
                                .to_vec::<f32>()?
                                .iter()
                                .map(|v| v.to_bits())
                                .collect::<Vec<_>>(),
                            "M={rows} K={k} metadata={metadata:?} limit={limit}"
                        );
                    }
                }
            }
        }
        Ok(())
    }
}
