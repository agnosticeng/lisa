use objc2_metal::MTLSize;

use crate::array::Array;

use super::compile::{
    builtin_template_def, compile_builtin, get_2d_grid_dims, tg_from_row_size, type_string,
    type_to_name,
};
use super::{DType, Device, Tensor};
use super::{
    MLX_ARG_REDUCE_SOURCE, MLX_REDUCE_PREAMBLE, MLX_REDUCE_UTILS_PREAMBLE, MLX_UTILS_PREAMBLE,
};
use crate::error::Result;

/// MLX `ops::argmax_axis` — `ArgReduce::eval_gpu` with the `argmax_` kernel.
pub fn argmax_axis(device: &Device, x: &Tensor, axis: i32) -> Result<Tensor> {
    arg_reduce_axis(device, x, axis, "ArgMax", "argmax")
}

/// MLX `ops::argmin_axis`.
pub fn argmin_axis(device: &Device, x: &Tensor, axis: i32) -> Result<Tensor> {
    arg_reduce_axis(device, x, axis, "ArgMin", "argmin")
}

/// MLX `ops::sum_axis`/`max`/`min`/`mean` over the last axis, `row_reduce_simple`
/// (`reduce.cpp:467`). `op_name` is `"sum"`, `"max"`, `"min"` or `"mean"`; MLX's
/// `mean` is `multiply(sum, 1/n)` with the reciprocal computed in f32 and cast
/// to the array dtype (`ops.cpp:2340`), so it rides the sum kernel.
pub fn reduce_last_axis(device: &Device, x: &Tensor, op_name: &str) -> Result<Tensor> {
    if op_name == "mean" {
        let n = *x.dims().last().unwrap() as f32;
        let s = reduce_last_axis(device, x, "sum")?;
        let norm = Array::scalar_of(device, 1.0f32 / n, x.dtype())?;
        return s.broadcast_mul(&norm);
    }
    let mdev = device;
    let (_s, layout) = x.buffer_and_layout();
    if x.dims().is_empty() || !layout.is_contiguous() {
        crate::bail!("reduce: non-contiguous or scalar input");
    }
    let dims = x.dims().to_vec();
    let dt = x.dtype();
    let row_size = *dims.last().unwrap();
    let out_dims = dims[..dims.len() - 1].to_vec();
    let in_t = type_string(dt)?;
    let out_t = in_t;
    let ty = type_to_name(dt)?;
    let op_type = {
        let mut c = op_name.chars();
        match c.next() {
            Some(f) => f.to_uppercase().collect::<String>() + c.as_str(),
            None => crate::bail!("reduce: empty op"),
        }
    };
    let op = format!("{op_type}<{out_t}>");
    let kernel_name = format!("row_reduce_simple_{op_name}{ty}");
    let def = builtin_template_def(
        &kernel_name,
        "row_reduce_simple",
        &[
            in_t.to_string(),
            out_t.to_string(),
            op,
            "size_t".to_string(),
        ],
    );
    let source =
        format!("{MLX_UTILS_PREAMBLE}{MLX_REDUCE_UTILS_PREAMBLE}{MLX_REDUCE_PREAMBLE}{def}");
    let pipeline = compile_builtin(device, &source, &kernel_name)?;

    let out_count: usize = out_dims.iter().product::<usize>().max(1);
    let obuf = mdev.buffer((out_count) as usize * (dt).size_of(), "reduce_out")?;
    let out = Array::from_parts(mdev, obuf.clone(), &out_dims.clone(), dt);

    let guard = mdev.commands.encoder()?;
    let enc = guard.encoder();
    enc.set_pipeline(&pipeline);
    {
        let (ms, layout) = x.buffer_and_layout();
        enc.set_input(0, Some(ms), layout.offset * x.dtype().size_of());
    }
    enc.set_output(1, Some(&obuf), 0);
    let rs = row_size;
    enc.set_bytes(2, &rs);
    let os = out_count as i64;
    enc.set_bytes(3, &os);

    let tgs = tg_from_row_size(row_size).min(pipeline.max_total_threads_per_threadgroup());
    let mut out_strides = vec![1i64; out_dims.len()];
    let mut acc = 1i64;
    for i in (0..out_dims.len()).rev() {
        out_strides[i] = acc;
        acc *= out_dims[i] as i64;
    }
    let (gx, gy) = get_2d_grid_dims(&out_dims, &out_strides);
    let gw = gx.div_ceil(4);
    enc.dispatch_threads_size(
        MTLSize {
            width: tgs,
            height: gw,
            depth: gy,
        },
        MTLSize {
            width: tgs,
            height: 1,
            depth: 1,
        },
    );
    Ok(out)
}

/// Reduction over an arbitrary axis, implemented as a transpose to the last
/// axis + [`reduce_last_axis`] + transpose back (all bit-exact steps).
pub fn reduce_axis(device: &Device, x: &Tensor, axis: i32, op_name: &str) -> Result<Tensor> {
    let rank = x.rank();
    let ax = if axis < 0 {
        (rank as i32 + axis) as usize
    } else {
        axis as usize
    };
    if ax == rank - 1 {
        return reduce_last_axis(device, x, op_name);
    }
    let t = x.transpose(ax, rank - 1)?.contiguous()?;
    let r = reduce_last_axis(device, &t, op_name)?;
    // t has the reduced axis now last; move it back to `ax`.
    r.transpose(ax, rank - 2)?.contiguous()
}

/// MLX `ops::mean_axis` over the last axis: `multiply(sum(a), 1/n)` where the
/// normaliser is `1/n` computed in f32 and cast to the array dtype (`ops.cpp:2340`).
pub fn mean_last_axis(device: &Device, x: &Tensor) -> Result<Tensor> {
    let dims = x.dims().to_vec();
    let n = *dims.last().unwrap() as f32;
    let s = reduce_last_axis(device, x, "sum")?;
    let norm = Array::scalar_of(device, 1.0f32 / n, x.dtype())?;
    Ok(s.broadcast_mul(&norm)?)
}

fn arg_reduce_axis(
    device: &Device,
    x: &Tensor,
    axis: i32,
    op_struct: &str,
    prefix: &str,
) -> Result<Tensor> {
    let mdev = device;
    let (_s, layout) = x.buffer_and_layout();
    if !layout.is_contiguous() {
        crate::bail!("arg_reduce: non-contiguous input");
    }
    let dims = x.dims().to_vec();
    let rank = dims.len();
    if rank == 0 {
        crate::bail!("arg_reduce: scalar input");
    }
    let ax = if axis < 0 {
        (rank as i32 + axis) as usize
    } else {
        axis as usize
    };
    let in_dt = x.dtype();
    let ty = type_to_name(in_dt)?;
    let t = type_string(in_dt)?;
    let kernel_name = format!("{prefix}_{ty}");
    let inst = format!(
        "\ntemplate [[host_name(\"{kernel_name}\")]] [[kernel]] decltype(arg_reduce_general<{t}, {op_struct}<{t}>>) arg_reduce_general<{t}, {op_struct}<{t}>>;\n"
    );
    let source = format!("{MLX_UTILS_PREAMBLE}{MLX_ARG_REDUCE_SOURCE}{inst}");
    let pipeline = compile_builtin(device, &source, &kernel_name)?;

    let strides = layout.stride().to_vec();
    let mut shape: Vec<i32> = Vec::new();
    let mut in_strides: Vec<i64> = Vec::new();
    let mut axis_stride = 1i64;
    let mut axis_size = 1usize;
    for i in 0..rank {
        if i == ax {
            axis_stride = strides[i] as i64;
            axis_size = dims[i];
            continue;
        }
        shape.push(dims[i] as i32);
        in_strides.push(strides[i] as i64);
    }
    let out_dims: Vec<usize> = dims
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != ax)
        .map(|(_, d)| *d)
        .collect();
    let mut out_strides = vec![1i64; out_dims.len()];
    let mut acc = 1i64;
    for i in (0..out_dims.len()).rev() {
        out_strides[i] = acc;
        acc *= out_dims[i] as i64;
    }
    let ndim = out_dims.len();

    let out_count: usize = out_dims.iter().product::<usize>().max(1);
    let obuf = mdev.buffer((out_count) as usize * (DType::U32).size_of(), "arg_out")?;
    let out = Array::from_parts(mdev, obuf.clone(), &out_dims.clone(), DType::U32);

    let guard = mdev.commands.encoder()?;
    let enc = guard.encoder();
    enc.set_pipeline(&pipeline);
    {
        let (ms, layout) = x.buffer_and_layout();
        enc.set_input(0, Some(ms), layout.offset * x.dtype().size_of());
    }
    enc.set_output(1, Some(&obuf), 0);
    if ndim == 0 {
        let shape_: i32 = 0;
        let stride_: i64 = 0;
        enc.set_bytes(2, &shape_);
        enc.set_bytes(3, &stride_);
        enc.set_bytes(4, &stride_);
    } else {
        enc.set_bytes_directly(
            2,
            std::mem::size_of_val(shape.as_slice()),
            shape.as_ptr().cast(),
        );
        enc.set_bytes_directly(
            3,
            std::mem::size_of_val(in_strides.as_slice()),
            in_strides.as_ptr().cast(),
        );
        enc.set_bytes_directly(
            4,
            std::mem::size_of_val(out_strides.as_slice()),
            out_strides.as_ptr().cast(),
        );
    }
    enc.set_bytes(5, &ndim);
    enc.set_bytes(6, &axis_stride);
    enc.set_bytes(7, &axis_size);

    // `get_2d_grid_dims(out.shape(), out.strides())` for rank <= 2.
    let (gd_w, gd_h) = match ndim {
        0 => (1usize, 1usize),
        1 => (out_dims[0], 1usize),
        2 => (out_dims[1], out_dims[0]),
        _ => crate::bail!("arg_reduce: out rank > 2 not implemented"),
    };
    let simd = 32usize;
    let mut tgs = axis_size
        .div_ceil(4)
        .min(pipeline.max_total_threads_per_threadgroup());
    tgs = tgs.div_ceil(simd) * simd;
    enc.dispatch_threads_size(
        MTLSize {
            width: tgs,
            height: gd_w,
            depth: gd_h,
        },
        MTLSize {
            width: tgs,
            height: 1,
            depth: 1,
        },
    );
    Ok(out)
}

/// MLX `strided_reduce_small` for reducing axis 0 of a row-contiguous
/// `[R, ...]` intermediate (`reduce.cpp:584` + `ColReduceArgs(intermediate)`):
/// sum over the outermost axis with `col_reduce_small`.
pub fn reduce_axis0_sum(device: &Device, inter: &Tensor) -> Result<Tensor> {
    let mdev = device;
    let dims = inter.dims().to_vec();
    let r = dims[0];
    let rest: usize = dims[1..].iter().product();
    let in_t = type_string(inter.dtype())?;
    let out_t = in_t;
    let ty = type_to_name(inter.dtype())?;
    let kname = format!("col_reduce_small_1_reduce_sum{ty}");
    let def = builtin_template_def(
        &kname,
        "col_reduce_small",
        &[
            in_t.to_string(),
            out_t.to_string(),
            format!("Sum<{out_t}>"),
            "int".to_string(),
            "1".to_string(),
        ],
    );
    let source =
        format!("{MLX_UTILS_PREAMBLE}{MLX_REDUCE_UTILS_PREAMBLE}{MLX_REDUCE_PREAMBLE}{def}");
    let pipeline = compile_builtin(device, &source, &kname)?;

    let out_count = rest;
    let obuf = mdev.buffer(
        (out_count) as usize * (inter.dtype()).size_of(),
        "colred_out",
    )?;
    let out_dims = dims[1..].to_vec();
    let out = Array::from_parts(mdev, obuf.clone(), &out_dims, inter.dtype());

    let guard = mdev.commands.encoder()?;
    let enc = guard.encoder();
    enc.set_pipeline(&pipeline);
    {
        let (ms, layout) = inter.buffer_and_layout();
        enc.set_input(0, Some(ms), layout.offset * inter.dtype().size_of());
    }
    enc.set_output(1, Some(&obuf), 0);
    let rs = r;
    enc.set_bytes(2, &rs);
    let rstride = rest as i64;
    enc.set_bytes(3, &rstride);
    let shape_z: [i32; 1] = [0];
    let stride_z: [i64; 1] = [0];
    enc.set_bytes_directly(4, std::mem::size_of_val(&shape_z), shape_z.as_ptr().cast());
    enc.set_bytes_directly(
        5,
        std::mem::size_of_val(&stride_z),
        stride_z.as_ptr().cast(),
    );
    let ndim: i32 = 0;
    enc.set_bytes(6, &ndim);
    let reduce_shape: [i32; 1] = [r as i32];
    let reduce_strides: [i64; 1] = [rstride];
    enc.set_bytes_directly(
        7,
        std::mem::size_of_val(&reduce_shape),
        reduce_shape.as_ptr().cast(),
    );
    enc.set_bytes_directly(
        8,
        std::mem::size_of_val(&reduce_strides),
        reduce_strides.as_ptr().cast(),
    );
    let reduce_ndim: i32 = 1;
    enc.set_bytes(9, &reduce_ndim);
    let ncr: usize = 1;
    enc.set_bytes(10, &ncr);

    let n_reads = 4usize;
    let blocks = (rest).div_ceil(n_reads);
    let tg_x = blocks.min(32);
    let tg_y = 8usize
        .min(pipeline.max_total_threads_per_threadgroup() / tg_x)
        .min(r);
    enc.dispatch_groups_size(
        MTLSize {
            width: blocks.div_ceil(tg_x),
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: tg_x,
            height: tg_y,
            depth: 1,
        },
    );
    Ok(out)
}
