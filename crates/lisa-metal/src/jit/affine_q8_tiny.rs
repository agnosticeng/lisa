//! Native reference Q8 schedules over original MLX affine storage.

use super::{DType, Device, Tensor};
use crate::array::Array;
use crate::error::Result;
use crate::runtime::{ConstVal, Math};

const MV_SOURCE: &crate::shader_pack::Source =
    crate::kernel_source!("common/matmul/affine_q8_mv_native.metal");
const EXT_SOURCE: &crate::shader_pack::Source =
    crate::kernel_source!("common/matmul/affine_q8_ext_small_native.metal");

/// Preserve upstream's M=1 block-dot and M=2..16 vector-dot dispatch gates.
/// GLM chooses this path after the corresponding native fixture and complete
/// model-logit comparisons; other models do not call this wrapper.
pub fn affine_q8_tiny_native(
    device: &Device,
    x: &Tensor,
    weights: &Tensor,
    scales: &Tensor,
    biases: &Tensor,
) -> Result<Tensor> {
    if [x, weights, scales, biases].iter().any(|t| t.rank() != 2) {
        crate::bail!("affine_q8_tiny_native requires rank-2 tensors");
    }
    let (m, k) = x.dims2();
    let (n, packed_k) = weights.dims2();
    if !(1..=16).contains(&m)
        || n == 0
        || k == 0
        || k % 64 != 0
        || (m > 1 && k % 128 != 0)
        || packed_k.checked_mul(4) != Some(k)
        || x.dtype() != DType::F32
        || weights.dtype() != DType::U32
        || !matches!(scales.dtype(), DType::F32 | DType::BF16)
        || biases.dtype() != scales.dtype()
        || scales.dims2() != (n, k / 64)
        || biases.dims2() != (n, k / 64)
        || [m, k, n].into_iter().any(|d| d > i32::MAX as usize)
    {
        crate::bail!("unsupported native tiny Q8 shape/dtype: M={m}, K={k}, N={n}");
    }
    for tensor in [x, weights, scales, biases] {
        if !tensor.buffer_and_layout().1.is_contiguous() {
            crate::bail!("native tiny Q8 requires contiguous inputs");
        }
    }
    let (pipeline, groups, nsg, shared_bytes) = if m == 1 {
        let (name, nsg) = if n > 65536 {
            ("glm53_affine_q8_mv_native_nsg8", 8)
        } else {
            ("glm53_affine_q8_mv_native_nsg4", 4)
        };
        (
            device.compile_with(MV_SOURCE, name, Math::CompilerDefault)?,
            (n.div_ceil(2), 1, 1),
            nsg,
            32 * 2 * 4,
        )
    } else {
        let nxpsg = if k % 256 == 0 && m == 2 { 16 } else { 8 };
        let r1ptg = match m {
            2 => 2,
            3 | 6 => 3,
            5 => 5,
            _ => 4,
        };
        let name = format!("glm53_affine_q8_ext_small_native_r{r1ptg}");
        (
            device.compile_with_constants(
                EXT_SOURCE,
                &name,
                Math::CompilerDefault,
                &[(0, ConstVal::Int(2)), (1, ConstVal::Int(nxpsg))],
            )?,
            (n.div_ceil((32 / nxpsg as usize) * 2), m.div_ceil(r1ptg), 1),
            2,
            0,
        )
    };
    let output_bytes = m
        .checked_mul(n)
        .and_then(|elements| elements.checked_mul(4))
        .ok_or_else(|| crate::error::Error::msg("native tiny Q8 output size overflow"))?;
    let output_buffer = device.buffer_for_full_gpu_write(output_bytes, "glm53_q8_tiny_native")?;
    let output = Array::from_parts(device, output_buffer.clone(), &[m, n], DType::F32);
    let guard = device.commands.encoder()?;
    let encoder = guard.encoder();
    encoder.set_pipeline(&pipeline);
    encoder.set_bytes(
        0,
        &[
            k as u32,
            n as u32,
            m as u32,
            u32::from(scales.dtype() == DType::BF16),
        ],
    );
    for (index, tensor) in [(1, weights), (2, x), (4, scales), (5, biases)] {
        let (buffer, layout) = tensor.buffer_and_layout();
        encoder.set_input(
            index,
            Some(buffer),
            layout.offset * tensor.dtype().size_of(),
        );
    }
    encoder.set_output(3, Some(&output_buffer), 0);
    if shared_bytes != 0 {
        encoder.set_threadgroup_memory_length(0, shared_bytes);
    }
    encoder.dispatch_groups(groups, (32, nsg, 1));
    Ok(output)
}

#[cfg(test)]
mod tests {}
