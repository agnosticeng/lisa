use objc2_metal::MTLSize;

use std::sync::OnceLock;

use crate::array::Array;

/// The threadgroup size the looped rms family actually launched with (the
/// pipeline's `max_total_threads_per_threadgroup` at launch time). The
/// norm-in-qmv prologue must emulate the SAME emulated-lane count to stay
/// bit-exact with the reduction association (specs/16 phase 1). Set on the
/// first looped rms / fused_add_rms launch; falls back to 1024 (the
/// Apple-silicon device maximum) when no looped launch has happened yet.
static LOOPED_RMS_TGS: OnceLock<u32> = OnceLock::new();

pub fn looped_rms_tgs() -> u32 {
    *LOOPED_RMS_TGS.get_or_init(|| 1024)
}

fn record_looped_tgs(pipeline: &crate::runtime::ComputePipeline) {
    let tgs = pipeline.max_total_threads_per_threadgroup() as u32;
    let _ = LOOPED_RMS_TGS.set(tgs);
}

use super::compile::{
    builtin_template_def, compile_builtin_bool_consts, type_string, type_to_name,
};
use super::{Device, Tensor};
use super::{MLX_FUSED_ADD_RMS_SOURCE, MLX_RMS_NORM_SOURCE, MLX_UTILS_PREAMBLE};
use crate::error::Result;

/// MLX LayerNorm forward, including its two-pass reduction and cast before
/// the affine transform. Vision models use this with original checkpoint data.
pub fn layer_norm(
    device: &Device,
    x: &Tensor,
    weight: &Tensor,
    bias: &Tensor,
    eps: f32,
) -> Result<Tensor> {
    let (_, layout) = x.buffer_and_layout();
    if x.dims().is_empty() || !layout.is_contiguous() {
        crate::bail!("layer_norm requires contiguous non-scalar input");
    }
    let axis = *x.dims().last().unwrap();
    if axis == 0
        || weight.dims().len() != 1
        || bias.dims().len() != 1
        || weight.elem_count() != axis
        || bias.elem_count() != axis
        || x.dtype() != weight.dtype()
        || x.dtype() != bias.dtype()
    {
        crate::bail!("layer_norm requires matching input, weight and bias shapes/dtypes");
    }
    let looped = axis > 6656;
    let fun = if looped {
        "layer_norm_looped"
    } else {
        "layer_norm_single_row"
    };
    let name = format!("lisa_{fun}_{}", type_to_name(x.dtype())?);
    let defs = builtin_template_def(&name, fun, &[type_string(x.dtype())?.to_string()]);
    let source = format!(
        "{MLX_UTILS_PREAMBLE}{}{defs}",
        crate::kernel_source!("common/attention/layer_norm.metal")
    );
    let pipeline = super::compile::compile_builtin(device, &source, &name)?;
    let dims = x.dims().to_vec();
    let output =
        device.buffer_for_full_gpu_write(x.elem_count() * x.dtype().size_of(), "layer_norm_out")?;
    let out = Array::from_parts(device, output.clone(), &dims, x.dtype());
    let guard = device.commands.encoder()?;
    let enc = guard.encoder();
    enc.set_pipeline(&pipeline);
    for (index, tensor) in [x, weight, bias].iter().enumerate() {
        let (buffer, layout) = tensor.buffer_and_layout();
        enc.set_input(
            index,
            Some(buffer),
            layout.offset * tensor.dtype().size_of(),
        );
    }
    enc.set_output(3, Some(&output), 0);
    enc.set_bytes(4, &eps);
    enc.set_bytes(5, &(axis as u32));
    enc.set_bytes(6, &(weight.buffer_and_layout().1.stride()[0] as u32));
    enc.set_bytes(7, &(bias.buffer_and_layout().1.stride()[0] as u32));
    let threads = if looped {
        pipeline.max_total_threads_per_threadgroup()
    } else {
        axis.div_ceil(8).div_ceil(32) * 32
    };
    if threads > pipeline.max_total_threads_per_threadgroup() {
        crate::bail!("layer_norm threadgroup exceeds device limit");
    }
    enc.dispatch_threads_size(
        MTLSize {
            width: x.elem_count() / axis * threads,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: threads,
            height: 1,
            depth: 1,
        },
    );
    Ok(out)
}

/// The weightless-rms ones scalar, cached per dtype (specs/08 item 3). The
/// runtime is single-GPU (the device argument is identical across calls in a
/// process), matching the `BF16_SIGMOID` static precedent in moe_decode.
fn ones_scalar(device: &Device, dt: crate::array::Dtype) -> Result<Array> {
    use std::sync::OnceLock;
    static F32: OnceLock<Option<Array>> = OnceLock::new();
    static BF16: OnceLock<Option<Array>> = OnceLock::new();
    static F16: OnceLock<Option<Array>> = OnceLock::new();
    let slot = match dt {
        crate::array::Dtype::Float32 => &F32,
        crate::array::Dtype::Bfloat16 => &BF16,
        crate::array::Dtype::Float16 => &F16,
        // rms_norm only runs on float lanes; a `_` arm keeps the match total
        // without caching non-float dtypes.
        _ => {
            return Array::scalar_of(device, 1f32, dt);
        }
    };
    Ok(slot
        .get_or_init(|| Array::scalar_of(device, 1f32, dt).ok())
        .clone()
        .expect("ones scalar alloc"))
}

/// Fused residual-add + RMSNorm (specs/08 §1): one dispatch returns
/// `(bf16(x + r), rms_norm(bf16(x + r)))`, bit-identical to the composed
/// `add` → `rms_norm` pair by construction (the kernel rounds the sum to `T`
/// before the reduction, exactly what the composed chain reads back).
/// Only the non-looped / looped split of `rms_norm` is mirrored; inputs must
/// be contiguous with rank ≥ 1.
pub fn fused_add_rms_norm(
    device: &Device,
    x: &Tensor,
    r: &Tensor,
    weight: &Tensor,
    eps: f32,
) -> Result<(Tensor, Tensor)> {
    let (_s, layout) = x.buffer_and_layout();
    if x.dims().is_empty() || !layout.is_contiguous() {
        crate::bail!("fused_add_rms_norm: non-contiguous or scalar input");
    }
    let dims = x.dims().to_vec();
    let dt = x.dtype();
    let axis_size = *dims.last().unwrap();
    let n_rows = x.elem_count() / axis_size.max(1);
    let in_t = type_string(dt)?;
    let ty = type_to_name(dt)?;
    let looped = axis_size > 4096;
    let kernel_name = if looped {
        format!("fused_add_rms_looped_{ty}")
    } else {
        format!("fused_add_rms_{ty}")
    };
    let mut defs = builtin_template_def(
        &format!("fused_add_rms_{ty}"),
        "fused_add_rms_single_row",
        &[in_t.to_string()],
    );
    defs.push_str(&builtin_template_def(
        &format!("fused_add_rms_looped_{ty}"),
        "fused_add_rms_looped",
        &[in_t.to_string()],
    ));
    let source = format!("{MLX_UTILS_PREAMBLE}{MLX_FUSED_ADD_RMS_SOURCE}{defs}");

    let pipeline = compile_builtin_bool_consts(device, &source, &kernel_name, &[])?;

    if looped {
        record_looped_tgs(&pipeline);
    }

    let count = x.elem_count();
    let sum_buf = device.buffer_for_full_gpu_write(count * dt.size_of(), "fused_add_rms_sum")?;
    let norm_buf = device.buffer_for_full_gpu_write(count * dt.size_of(), "fused_add_rms_norm")?;
    let sum = Array::from_parts(device, sum_buf.clone(), &dims.clone(), dt);
    let normed = Array::from_parts(device, norm_buf.clone(), &dims.clone(), dt);

    let guard = device.commands.encoder()?;
    let enc = guard.encoder();
    enc.set_pipeline(&pipeline);
    {
        let (ms, layout) = x.buffer_and_layout();
        enc.set_input(0, Some(ms), layout.offset * x.dtype().size_of());
    }
    {
        let (ms, layout) = r.buffer_and_layout();
        enc.set_input(1, Some(ms), layout.offset * r.dtype().size_of());
    }
    let w_stride: u32 = {
        let (ms, wl) = weight.buffer_and_layout();
        enc.set_input(2, Some(ms), wl.offset * weight.dtype().size_of());
        if weight.rank() == 1 {
            wl.stride()[0] as u32
        } else {
            0
        }
    };
    enc.set_output(3, Some(&sum_buf), 0);
    enc.set_output(4, Some(&norm_buf), 0);
    enc.set_bytes(5, &eps);
    let asize = axis_size as u32;
    enc.set_bytes(6, &asize);
    enc.set_bytes(7, &w_stride);

    let simd = 32usize;
    if !looped {
        let tg_needed = axis_size.div_ceil(4);
        let tgs = simd * tg_needed.div_ceil(simd);
        let n_threads = n_rows * tgs;
        enc.dispatch_threads_size(
            MTLSize {
                width: n_threads,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: tgs,
                height: 1,
                depth: 1,
            },
        );
    } else {
        let tgs = pipeline.max_total_threads_per_threadgroup();
        let n_threads = n_rows * tgs;
        enc.dispatch_threads_size(
            MTLSize {
                width: n_threads,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: tgs,
                height: 1,
                depth: 1,
            },
        );
    }
    Ok((sum, normed))
}

/// MLX `fast::rms_norm` (`RMSNorm::eval_gpu`, `normalization.cpp:17`). The
/// `has_w` function constant selects the weighted scheme.
pub fn rms_norm(device: &Device, x: &Tensor, weight: Option<&Tensor>, eps: f32) -> Result<Tensor> {
    let mdev = device;
    let (_s, layout) = x.buffer_and_layout();
    if x.dims().is_empty() || !layout.is_contiguous() {
        crate::bail!("rms_norm: non-contiguous or scalar input");
    }
    let dims = x.dims().to_vec();
    let dt = x.dtype();
    let axis_size = *dims.last().unwrap();
    let n_rows = x.elem_count() / axis_size.max(1);
    let in_t = type_string(dt)?;
    let ty = type_to_name(dt)?;
    let looped = axis_size > 4096;
    let kernel_name = if looped {
        format!("rms_looped_{ty}")
    } else {
        format!("rms_{ty}")
    };
    let mut defs =
        builtin_template_def(&format!("rms_{ty}"), "rms_single_row", &[in_t.to_string()]);
    defs.push_str(&builtin_template_def(
        &format!("rms_looped_{ty}"),
        "rms_looped",
        &[in_t.to_string()],
    ));
    let source = format!("{MLX_UTILS_PREAMBLE}{MLX_RMS_NORM_SOURCE}{defs}");
    // function constant 20 is `has_w`. MLX passes a scalar `1` weight when no
    // weight is given (`fast.cpp:112`), so the kernel always runs the weighted
    // path with that scalar. The ones scalar is a per-step constant (specs/08
    // item 3): cached per dtype instead of rebuilt per call.
    let ones;
    let weight = match weight {
        Some(w) => w,
        None => {
            ones = ones_scalar(device, dt)?;
            &ones
        }
    };
    let pipeline = compile_builtin_bool_consts(device, &source, &kernel_name, &[(20, true)])?;
    if looped {
        record_looped_tgs(&pipeline);
    }

    let count = x.elem_count();
    let obuf = mdev.buffer_for_full_gpu_write((count) as usize * (dt).size_of(), "rms_out")?;
    let out = Array::from_parts(mdev, obuf.clone(), &dims.clone(), dt);

    let guard = mdev.commands.encoder()?;
    let enc = guard.encoder();
    enc.set_pipeline(&pipeline);
    {
        let (ms, layout) = x.buffer_and_layout();
        enc.set_input(0, Some(ms), layout.offset * x.dtype().size_of());
    }
    let w_stride: u32 = {
        let (ms, wl) = weight.buffer_and_layout();
        enc.set_input(1, Some(ms), wl.offset * weight.dtype().size_of());
        if weight.rank() == 1 {
            wl.stride()[0] as u32
        } else {
            0
        }
    };
    enc.set_output(2, Some(&obuf), 0);
    enc.set_bytes(3, &eps);
    let asize = axis_size as u32;
    enc.set_bytes(4, &asize);
    enc.set_bytes(5, &w_stride);

    let simd = 32usize;
    if !looped {
        let tg_needed = axis_size.div_ceil(4);
        let tgs = simd * tg_needed.div_ceil(simd);
        let n_threads = n_rows * tgs;
        enc.dispatch_threads_size(
            MTLSize {
                width: n_threads,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: tgs,
                height: 1,
                depth: 1,
            },
        );
    } else {
        let tgs = pipeline.max_total_threads_per_threadgroup();
        let n_threads = n_rows * tgs;
        enc.dispatch_threads_size(
            MTLSize {
                width: n_threads,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: tgs,
                height: 1,
                depth: 1,
            },
        );
    }
    Ok(out)
}
