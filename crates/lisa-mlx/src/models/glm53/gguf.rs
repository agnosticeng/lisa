//! Native GGUF GLM projections; packed DS4 weights are read in place.
use crate::error::{Error, Result};
use crate::ffi::{KernelInput, MetalKernel, OutputArg};
use crate::{Array, Dtype, Stream};
use std::sync::OnceLock;
const HEADER: &crate::shader_pack::Source = crate::kernel_source!("glm53/gguf_q8.h");
fn kernel(decode: bool) -> Result<&'static MetalKernel> {
    static QK: OnceLock<std::result::Result<MetalKernel, String>> = OnceLock::new();
    static DECODE: OnceLock<std::result::Result<MetalKernel, String>> = OnceLock::new();
    let k = if decode { &DECODE } else { &QK };
    k.get_or_init(|| {
        MetalKernel::new_with_math_and_threadgroup_memory(
            if decode {
                "glm53_gguf_indexed_decode_v1"
            } else {
                "glm53_gguf_qk_v1"
            },
            if decode {
                &[
                    "n_selected",
                    "dense_all",
                    "tc",
                    "scale",
                    "q_low",
                    "latents",
                    "selected",
                    "weight",
                ]
            } else {
                &["q", "weight"]
            },
            &["output"],
            if decode {
                crate::kernel_source!("glm53/gguf_decode.metal")
            } else {
                crate::kernel_source!("glm53/gguf_qk.metal")
            },
            HEADER,
            true,
            false,
            crate::runtime::Math::CompilerDefault,
            if decode { (256 + 2051 + 512) * 4 } else { 0 },
        )
        .map_err(|e| e.to_string())
    })
    .as_ref()
    .map_err(|e| Error::msg(e.clone()))
}
pub fn qk(q: &Array, weight: &Array, s: &Stream) -> Result<Array> {
    if q.size() != 64 * 256
        || q.dtype() != Dtype::Float32
        || weight.shape() != [64, 512, 272]
        || weight.dtype() != Dtype::Uint8
    {
        return Err(Error::msg("GLM GGUF QK layout mismatch"));
    }
    let inputs = [KernelInput::Tensor(&q.t), KernelInput::Tensor(&weight.t)];
    Ok(kernel(false)?
        .apply_with_inline_inputs(
            &inputs,
            &[],
            (64 * 256, 1, 1),
            (256, 1, 1),
            &[OutputArg {
                shape: vec![1, 1, 64, 512],
                dtype: Dtype::Float32,
            }],
            s,
        )?
        .remove(0))
}
pub fn decode(
    q: &Array,
    latents: &Array,
    selected: &Array,
    weight: &Array,
    dense: bool,
    s: &Stream,
) -> Result<Array> {
    let n = selected.size();
    if q.shape() != [1, 1, 64, 512]
        || q.dtype() != Dtype::Float32
        || latents.rank() != 3
        || latents.dim(0) != 1
        || latents.dim(2) != 512
        || latents.dtype() != Dtype::Float16
        || selected.dtype() != Dtype::Int32
        || n == 0
        || n > 2051
        || weight.shape() != [64, 256, 544]
        || weight.dtype() != Dtype::Uint8
    {
        return Err(Error::msg("GLM GGUF indexed decode layout mismatch"));
    }
    let inputs = [
        KernelInput::I32(n as i32),
        KernelInput::I32(i32::from(dense)),
        KernelInput::I32(latents.dim(1)),
        KernelInput::F32(0.0625),
        KernelInput::Tensor(&q.t),
        KernelInput::Tensor(&latents.t),
        KernelInput::Tensor(&selected.t),
        KernelInput::Tensor(&weight.t),
    ];
    Ok(kernel(true)?
        .apply_with_inline_inputs(
            &inputs,
            &[],
            (64 * 256, 1, 1),
            (256, 1, 1),
            &[OutputArg {
                shape: vec![1, 1, 16384],
                dtype: Dtype::Float32,
            }],
            s,
        )?
        .remove(0))
}

/// The original DS4 long private MTP window path uses bounded 16 KiB attention
/// scratch, followed by a separate Q8 value projection.
pub fn wide_decode(
    q: &Array,
    low: &Array,
    latents: &Array,
    selected: &Array,
    weight: &Array,
    s: &Stream,
) -> Result<Array> {
    if weight.shape() != [64, 256, 544] || weight.dtype() != Dtype::Uint8 {
        return Err(Error::msg("invalid GGUF value layout"));
    }
    let lora = super::kernels::dsa_indexed_lora_wide_native(
        &q.reshape(&[1, 64, 1, 256])?,
        low,
        latents,
        selected,
        0.0625,
        s,
    )
    .ok_or_else(|| Error::msg("GLM wide MTP attention declined"))?;
    static VALUE: OnceLock<std::result::Result<MetalKernel, String>> = OnceLock::new();
    let k = VALUE
        .get_or_init(|| {
            MetalKernel::new_with_math_and_threadgroup_memory(
                "glm53_gguf_value_v1",
                &["lora", "weight"],
                &["output"],
                crate::kernel_source!("glm53/gguf_value.metal"),
                HEADER,
                true,
                false,
                crate::runtime::Math::CompilerDefault,
                512 * 4,
            )
            .map_err(|e| e.to_string())
        })
        .as_ref()
        .map_err(|e| Error::msg(e.clone()))?;
    Ok(k.apply_with_inline_inputs(
        &[KernelInput::Tensor(&lora.t), KernelInput::Tensor(&weight.t)],
        &[],
        (64 * 256, 1, 1),
        (256, 1, 1),
        &[OutputArg {
            shape: vec![1, 1, 16384],
            dtype: Dtype::Float32,
        }],
        s,
    )?
    .remove(0))
}
