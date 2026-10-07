use objc2_metal::MTLSize;

use crate::array::Array;

use super::compile::{
    builtin_template_def, compile_builtin, type_string, type_to_name, work_per_thread,
};
use super::{Device, Tensor};
use super::{
    MLX_BINARY_OPS_PREAMBLE, MLX_BINARY_PREAMBLE, MLX_UNARY_OPS_PREAMBLE, MLX_UNARY_PREAMBLE,
    MLX_UTILS_PREAMBLE,
};
use crate::error::Result;

/// Run MLX's built-in unary op kernel (`unary.cpp:unary_op_gpu`, contiguous
/// path) bit-exactly. `op` is the MLX op identifier, e.g. `"Sigmoid"`.
pub fn unary_op(device: &Device, x: &Tensor, op: &str) -> Result<Tensor> {
    let mdev = device;
    // The kernel walks the buffer contiguously; a transposed/sliced view must
    // be materialised first or it would read the wrong elements.
    let x = x.contiguous()?;
    let x = &x;
    let in_dt = x.dtype();
    let out_dt = in_dt;
    let size = x.elem_count();
    if size == 0 {
        crate::bail!("unary_op: empty input");
    }
    let in_t = type_string(in_dt)?;
    let out_t = type_string(out_dt)?;
    let wpt = work_per_thread(in_dt, size);
    let ty_in = type_to_name(in_dt)?;
    let ty_out = type_to_name(out_dt)?;

    let mut kernel_name = String::from(if wpt > 1 { "vn" } else { "v" });
    kernel_name.push('_');
    kernel_name.push_str(op);
    kernel_name.push_str(ty_in);
    kernel_name.push_str(ty_out);
    let lib_name = kernel_name
        .split_once('_')
        .map(|(_, rest)| rest)
        .unwrap_or(&kernel_name)
        .to_string();

    let arg = |s: &str| s.to_string();
    let mut defs = builtin_template_def(
        &format!("v_{lib_name}"),
        "unary_v",
        &[arg(in_t), arg(out_t), arg(op), arg("1")],
    );
    if wpt > 1 {
        defs.push_str(&builtin_template_def(
            &format!("vn_{lib_name}"),
            "unary_v",
            &[arg(in_t), arg(out_t), arg(op)],
        ));
    }
    defs.push_str(&builtin_template_def(
        &format!("v2_{lib_name}"),
        "unary_v2",
        &[arg(in_t), arg(out_t), arg(op)],
    ));
    defs.push_str(&builtin_template_def(
        &format!("gn1_{lib_name}"),
        "unary_g",
        &[arg(in_t), arg(out_t), arg(op), arg("1"), arg("int")],
    ));
    defs.push_str(&builtin_template_def(
        &format!("gn4large_{lib_name}"),
        "unary_g",
        &[arg(in_t), arg(out_t), arg(op), arg("4")],
    ));

    let source = format!("{MLX_UTILS_PREAMBLE}{MLX_UNARY_OPS_PREAMBLE}{MLX_UNARY_PREAMBLE}{defs}");
    let pipeline = compile_builtin(device, &source, &kernel_name)?;

    let ybuf = mdev.buffer((size) as usize * (out_dt).size_of(), "unary_out")?;
    let y = Array::from_parts(mdev, ybuf.clone(), &x.dims().to_vec(), out_dt);

    let guard = mdev.commands.encoder()?;
    let enc = guard.encoder();
    enc.set_pipeline(&pipeline);
    {
        let (ms, layout) = x.buffer_and_layout();
        enc.set_input(0, Some(ms), layout.offset * x.dtype().size_of());
    }
    enc.set_output(1, Some(&ybuf), 0);
    let n = size as i32;
    enc.set_bytes(2, &n);

    let nthreads = size.div_ceil(wpt);
    let tg = pipeline.max_total_threads_per_threadgroup().min(nthreads);
    enc.dispatch_threads_size(
        MTLSize {
            width: nthreads,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: tg,
            height: 1,
            depth: 1,
        },
    );
    Ok(y)
}

/// MLX `ops::sigmoid` (`Sigmoid` primitive).
pub fn sigmoid(device: &Device, x: &Tensor) -> Result<Tensor> {
    unary_op(device, x, "Sigmoid")
}

/// MLX `nn.silu` is `x * sigmoid(x)`.
pub fn silu(device: &Device, x: &Tensor) -> Result<Tensor> {
    let s = sigmoid(device, x)?;
    Ok(x.mul(&s)?)
}

/// MLX's other built-in unary ops (all via the generic `unary_op` kernel path).
pub fn exp(device: &Device, x: &Tensor) -> Result<Tensor> {
    unary_op(device, x, "Exp")
}
pub fn log(device: &Device, x: &Tensor) -> Result<Tensor> {
    unary_op(device, x, "Log")
}
pub fn log1p(device: &Device, x: &Tensor) -> Result<Tensor> {
    unary_op(device, x, "Log1p")
}
pub fn sin(device: &Device, x: &Tensor) -> Result<Tensor> {
    unary_op(device, x, "Sin")
}
pub fn cos(device: &Device, x: &Tensor) -> Result<Tensor> {
    unary_op(device, x, "Cos")
}
pub fn sqrt(device: &Device, x: &Tensor) -> Result<Tensor> {
    unary_op(device, x, "Sqrt")
}
pub fn rsqrt(device: &Device, x: &Tensor) -> Result<Tensor> {
    unary_op(device, x, "Rsqrt")
}
pub fn abs(device: &Device, x: &Tensor) -> Result<Tensor> {
    unary_op(device, x, "Abs")
}
pub fn sign(device: &Device, x: &Tensor) -> Result<Tensor> {
    unary_op(device, x, "Sign")
}
pub fn negative(device: &Device, x: &Tensor) -> Result<Tensor> {
    unary_op(device, x, "Negative")
}
pub fn square(device: &Device, x: &Tensor) -> Result<Tensor> {
    unary_op(device, x, "Square")
}

/// Run MLX's built-in binary op kernel (`binary.cpp:binary_op_gpu`, the
/// VectorVector contiguous path) bit-exactly. `op` is the MLX primitive name,
/// e.g. `"LogAddExp"`. Both operands must have the same shape and dtype.
pub fn binary_op(device: &Device, a: &Tensor, b: &Tensor, op: &str) -> Result<Tensor> {
    if a.dims() != b.dims() || a.dtype() != b.dtype() {
        crate::bail!("binary_op: operands must match in shape and dtype");
    }
    let mdev = device;
    let a = a.contiguous()?;
    let b = b.contiguous()?;
    let (a, b) = (&a, &b);
    let in_dt = a.dtype();
    let out_dt = in_dt;
    let size = a.elem_count();
    if size == 0 {
        crate::bail!("binary_op: empty input");
    }
    let in_t = type_string(in_dt)?;
    let out_t = type_string(out_dt)?;
    let wpt = work_per_thread(in_dt, size);
    let ty = type_to_name(in_dt)?;

    let mut kernel_name = String::from(if wpt > 1 { "vvn" } else { "vv" });
    kernel_name.push('_');
    kernel_name.push_str(op);
    kernel_name.push_str(ty);
    let lib_name = kernel_name
        .split_once('_')
        .map(|(_, r)| r)
        .unwrap_or(&kernel_name)
        .to_string();

    let arg = |s: &str| s.to_string();
    let mut defs = builtin_template_def(
        &format!("vv_{lib_name}"),
        "binary_vv",
        &[arg(in_t), arg(out_t), arg(op), arg("1")],
    );
    if wpt > 1 {
        defs.push_str(&builtin_template_def(
            &format!("vvn_{lib_name}"),
            "binary_vv",
            &[arg(in_t), arg(out_t), arg(op)],
        ));
    }
    let source =
        format!("{MLX_UTILS_PREAMBLE}{MLX_BINARY_OPS_PREAMBLE}{MLX_BINARY_PREAMBLE}{defs}");
    let pipeline = compile_builtin(device, &source, &kernel_name)?;

    let cbuf = mdev.buffer((size) as usize * (out_dt).size_of(), "binary_out")?;
    let c = Array::from_parts(mdev, cbuf.clone(), &a.dims().to_vec(), out_dt);

    let guard = mdev.commands.encoder()?;
    let enc = guard.encoder();
    enc.set_pipeline(&pipeline);
    let bind = |index: usize, t: &Tensor| -> Result<()> {
        let (ms, layout) = t.buffer_and_layout();
        enc.set_input(index, Some(ms), layout.offset * t.dtype().size_of());
        Ok(())
    };
    bind(0, a)?;
    bind(1, b)?;
    enc.set_output(2, Some(&cbuf), 0);
    let n = size as i32;
    enc.set_bytes(3, &n);

    let nthreads = size.div_ceil(wpt);
    let tg = pipeline.max_total_threads_per_threadgroup().min(nthreads);
    enc.dispatch_threads_size(
        MTLSize {
            width: nthreads,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: tg,
            height: 1,
            depth: 1,
        },
    );
    Ok(c)
}

/// MLX `ops::logaddexp` (the `LogAddExp` binary primitive).
pub fn logaddexp(device: &Device, a: &Tensor, b: &Tensor) -> Result<Tensor> {
    binary_op(device, a, b, "LogAddExp")
}
