use objc2_metal::MTLSize;

use super::compile::compile_builtin_lazy;
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

/// Specialize the row count without changing each row's reduction tree.
pub(crate) fn row_reduce_simple_spec(
    op_name: &str,
    in_t: &str,
    out_t: &str,
    ty: &str,
    rows: usize,
) -> Result<(String, String, usize)> {
    let mut chars = op_name.chars();
    let op = match chars.next() {
        Some(first) => first.to_uppercase().collect::<String>() + chars.as_str(),
        None => crate::bail!("reduce: empty op"),
    };
    // The four-row kernel backs up its last tile. Fewer than four rows
    // would underflow its unsigned row index and write before the output.
    let writes = if rows < 4 { 1 } else { 4 };
    let mut name = format!("row_reduce_simple_{op_name}{ty}");
    let mut args = vec![
        in_t.to_string(),
        out_t.to_string(),
        format!("{op}<{out_t}>"),
        "size_t".to_string(),
    ];
    if writes == 1 {
        name.push_str("_w1");
        args.extend(["REDUCE_N_READS".to_string(), "1".to_string()]);
    }
    Ok((
        name.clone(),
        builtin_template_def(&name, "row_reduce_simple", &args),
        writes,
    ))
}

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
    let out_count: usize = out_dims.iter().product::<usize>().max(1);
    let (kernel_name, def, writes) = row_reduce_simple_spec(op_name, in_t, out_t, ty, out_count)?;
    let source =
        format!("{MLX_UTILS_PREAMBLE}{MLX_REDUCE_UTILS_PREAMBLE}{MLX_REDUCE_PREAMBLE}{def}");
    let pipeline = compile_builtin(device, &source, &kernel_name)?;

    let obuf = mdev.buffer_for_gpu_write((out_count) * (dt).size_of(), "reduce_out")?;
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
    let gw = gx.div_ceil(writes);
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
    let source = || format!("{MLX_UTILS_PREAMBLE}{MLX_ARG_REDUCE_SOURCE}{inst}");
    let pipeline = compile_builtin_lazy(device, &kernel_name, source)?;

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
    let obuf = mdev.buffer_for_gpu_write((out_count) * (DType::U32).size_of(), "arg_out")?;
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
    let obuf = mdev.buffer_for_gpu_write((out_count) * (inter.dtype()).size_of(), "colred_out")?;
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

#[cfg(test)]
mod row_bounds_tests {
    use super::*;
    use crate::Dtype;

    #[test]
    fn small_row_reductions_preserve_guard_bytes_and_four_row_arithmetic() -> Result<()> {
        let stream = crate::Stream::thread_local_or_default();
        let rt = stream.runtime();
        for dtype in [Dtype::Float32, Dtype::Bfloat16, Dtype::Float16] {
            for width in [1usize, 7, 128, 257, 1025, 4096, 16385] {
                for rows in 1..=5usize {
                    // Prefix rows make an erroneous negative input offset readable;
                    // prefix output sentinels detect the corresponding underwrite.
                    let data = (0..(rows + 6) * width)
                        .map(|i| ((i * 17 % 61) as f32 - 30.) / 31.7)
                        .collect::<Vec<_>>();
                    let input = Array::from_slice(rt, &data, &[rows + 6, width])?.cast(dtype)?;
                    let view = input.narrow(0, 3, rows)?;
                    for op in ["sum", "max", "min"] {
                        let output = Array::from_slice(rt, &vec![11.25f32; rows + 6], &[rows + 6])?
                            .cast(dtype)?;
                        let (name, def, writes) = row_reduce_simple_spec(
                            op,
                            type_string(dtype)?,
                            type_string(dtype)?,
                            type_to_name(dtype)?,
                            rows,
                        )?;
                        let source = || {
                            format!(
                                "{MLX_UTILS_PREAMBLE}{MLX_REDUCE_UTILS_PREAMBLE}{MLX_REDUCE_PREAMBLE}{def}"
                            )
                        };
                        let pipeline = compile_builtin_lazy(rt, &name, source)?;
                        {
                            let guard = rt.commands.encoder()?;
                            let enc = guard.encoder();
                            enc.set_pipeline(&pipeline);
                            enc.set_input(0, Some(input.buffer()), 3 * width * dtype.size_of());
                            enc.set_output(1, Some(output.buffer()), 3 * dtype.size_of());
                            enc.set_bytes(2, &width);
                            enc.set_bytes(3, &(rows as i64));
                            let threads = tg_from_row_size(width)
                                .min(pipeline.max_total_threads_per_threadgroup());
                            enc.dispatch_groups((1, rows.div_ceil(writes), 1), (threads, 1, 1));
                        }
                        let actual = output.cast(Dtype::Float32)?.to_vec::<f32>()?;
                        assert_eq!(
                            &actual[..3],
                            &[11.25; 3],
                            "prefix {dtype:?} rows={rows} width={width} {op}"
                        );
                        assert_eq!(
                            &actual[rows + 3..],
                            &[11.25; 3],
                            "suffix {dtype:?} rows={rows} width={width} {op}"
                        );
                        // Compare each valid row against the unchanged four-row
                        // specialization, with the same input bytes and group width.
                        let reference =
                            reduce_last_axis(rt, &input.narrow(0, 3, rows.max(4))?, op)?;
                        let expected = reference.cast(Dtype::Float32)?.to_vec::<f32>()?;
                        let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
                        assert_eq!(
                            bits(&actual[3..rows + 3]),
                            bits(&expected[..rows]),
                            "{dtype:?} rows={rows} width={width} {op}"
                        );
                        for result in [reduce_last_axis(rt, &view, op)?, view.reduce_last_axis(op)?]
                        {
                            assert_eq!(
                                bits(&result.cast(Dtype::Float32)?.to_vec::<f32>()?),
                                bits(&expected[..rows])
                            );
                        }
                    }
                }
            }
        }
        Ok(())
    }
}
