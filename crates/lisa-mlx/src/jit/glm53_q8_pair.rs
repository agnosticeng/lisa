//! Native DS4 fused Q8 gate/up and SwiGLU for one to sixteen rows.
use objc2_metal::MTLSize;

use super::{DType, Device, Tensor};
use crate::array::Array;
use crate::error::Result;
use crate::runtime::{ConstVal, Math};

const EXT_SOURCE: &str =
    include_str!("../shaders/common/matmul/glm53_affine_q8_ext_small_native.metal");
const PAIR_SOURCE: &str =
    include_str!("../shaders/common/matmul/glm53_affine_q8_pair_native.metal");

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
    let constants = if rows == 1 {
        Vec::new()
    } else {
        vec![(0, ConstVal::Int(nsg)), (1, ConstVal::Int(nxpsg))]
    };
    let pipeline =
        device.compile_with_constants(source, &name, Math::CompilerDefault, &constants)?;
    let bytes = rows
        .checked_mul(n)
        .and_then(|count| count.checked_mul(4))
        .ok_or_else(|| crate::error::Error::msg("native Q8 gate/up output size overflow"))?;
    let gate_buffer = device.buffer_for_full_gpu_write(bytes, "glm53_q8_native_gate")?;
    let up_buffer = device.buffer_for_full_gpu_write(bytes, "glm53_q8_native_up")?;
    let mid_buffer = device.buffer_for_full_gpu_write(bytes, "glm53_q8_native_mid")?;
    let gate = Array::from_parts(device, gate_buffer.clone(), &[rows, n], DType::F32);
    let up = Array::from_parts(device, up_buffer.clone(), &[rows, n], DType::F32);
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
    enc.set_output(4, Some(&gate_buffer), 0);
    enc.set_output(5, Some(&up_buffer), 0);
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
