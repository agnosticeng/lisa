//! Generic kernel-source assembly and JIT pipelines.
//!
//! The engine's own kernels are single static `.metal` sources. The *generic*
//! op set (elementwise, reductions, softmax, sort, rope, sdpa, dequantize,
//! gemm/gemv) is instead built here, because Metal has no runtime type
//! dispatch: each element type / configuration is a separate C++ template
//! instantiation, selected by name.
//!
//! Each generic kernel is therefore assembled from three static pieces:
//!   1. a **preamble** (helpers and shared types, `include_str!`);
//!   2. the **kernel body** (`include_str!`);
//!   3. an **explicit-instantiation line** per variant
//!      (`[[host_name("name")]] … decltype(fn<args>) fn<args>;`), built by
//!      [`builtin_template_def`].
//! The assembled source is JIT-compiled with the Safe/precise compile options
//! and cached per generated kernel name. This mirrors how the GPU code is meant
//! to be specialized; the bodies stay static and the host only picks a variant.
//!
//! The signature generator is a faithful reproduction; see `NOTICE` for the
//! third-party attribution.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::Mutex;

use crate::array::{Array, Dtype};
use crate::error::{Error, Result};
use crate::runtime::{Buffer, ComputePipeline, ConstVal, Math, MetalRuntime};

/// The generic kernels were written against a `Tensor`/`DType`/`Device` naming;
/// these aliases keep the bodies readable while everything behind them is the
/// native `Array`/`Dtype`/`MetalRuntime`.
pub type Tensor = Array;
pub type DType = Dtype;
pub type Device = std::sync::Arc<MetalRuntime>;
use objc2_metal::MTLSize;

/// Inputs with fewer elements are passed in the `constant` address space.
const MAX_CONSTANT_ARRAY_SIZE: usize = 8;

/// MLX `metal::utils()` — the preprocessed Metal preamble (`metal_stdlib`,
/// `using namespace metal`, `bfloat16_t`, bf16 math overloads). MLX prepends
/// this to every custom kernel source (`CustomKernel::eval_gpu`), so we do too.
pub(crate) const MLX_UTILS_PREAMBLE: &str = include_str!("kernels/common/utils.metal");
// Flattened MLX Metal NAX header closures (auto-generated; quoted #include
// lines inlined in dependency order). These replace the former `mlx_include`
// tree: the AOT compiles prepend the right closure instead of `-I`.
const NAX_GEMM_HEADER: &str = include_str!("kernels/nax/gemm_header.metal");
const NAX_QUANT_HEADER: &str = include_str!("kernels/nax/quant_header.metal");
const NAX_ATTN_HEADER: &str = include_str!("kernels/nax/attn_header.metal");

/// MLX `metal::gemm()` — preprocessed `kernels/steel/gemm/gemm.h`.
const MLX_GEMM_PREAMBLE: &str = include_str!("kernels/common/gemm.metal");
/// MLX `metal::quantized_utils()` — preprocessed `kernels/quantized_utils.h`.
const MLX_QUANTIZED_UTILS_PREAMBLE: &str = include_str!("kernels/common/quantized_utils.metal");
/// MLX `metal::quantized()` — preprocessed `kernels/quantized.h` (affine kernels).
const MLX_QUANTIZED_PREAMBLE: &str = include_str!("kernels/common/quantized.metal");
/// MLX `metal::unary_ops()` — preprocessed `kernels/unary_ops.h`.
pub(crate) const MLX_UNARY_OPS_PREAMBLE: &str = include_str!("kernels/common/unary_ops.metal");
/// MLX `metal::unary()` — preprocessed `kernels/unary.h`.
pub(crate) const MLX_UNARY_PREAMBLE: &str = include_str!("kernels/common/unary.metal");
/// MLX `metal::binary_ops()` — preprocessed `kernels/binary_ops.h`.
const MLX_BINARY_OPS_PREAMBLE: &str = include_str!("kernels/common/binary_ops.metal");
/// MLX `metal::binary()` — preprocessed `kernels/binary.h`.
const MLX_BINARY_PREAMBLE: &str = include_str!("kernels/common/binary.metal");
/// MLX `metal::softmax()` — preprocessed `kernels/softmax.h`.
pub(crate) const MLX_SOFTMAX_PREAMBLE: &str = include_str!("kernels/common/softmax.metal");
/// MLX `metal::reduce_utils()` — preprocessed `kernels/reduce_utils.h`.
pub(crate) const MLX_REDUCE_UTILS_PREAMBLE: &str = include_str!("kernels/common/reduce_utils.metal");
/// MLX `metal::reduce()` — preprocessed `kernels/reduce.h`.
pub(crate) const MLX_REDUCE_PREAMBLE: &str = include_str!("kernels/common/reduce.metal");
/// MLX `metal::sort()` — preprocessed `kernels/sort.h`.
const MLX_SORT_PREAMBLE: &str = include_str!("kernels/common/sort.metal");
/// MLX `kernels/rms_norm.metal` (AOT source; project include stripped). Uses the
/// `has_w` function constant (index 20).
const MLX_RMS_NORM_SOURCE: &str = include_str!("kernels/common/rms_norm.metal");
/// MLX `kernels/rope.metal` (AOT source; project include stripped). Function
/// constants: 1 = forward, 2 = traditional, 3 = head_seq_transpose.
const MLX_ROPE_SOURCE: &str = include_str!("kernels/common/rope.metal");
/// MLX `kernels/sdpa_vector.h` (AOT source). Function constants 20..25:
/// has_mask, query_transposed, do_causal, bool_mask, float_mask, has_sinks.
const MLX_SDPA_VECTOR_PREAMBLE: &str = include_str!("kernels/common/sdpa_vector.metal");
/// MLX `metal::scatter_axis()` — preprocessed `kernels/indexing/scatter_axis.h`.
const MLX_SCATTER_AXIS_PREAMBLE: &str = include_str!("kernels/common/scatter_axis.metal");
/// MLX `arg_reduce.metal` (AOT source; project include stripped — the utils
/// preamble provides it).
pub(crate) const MLX_ARG_REDUCE_SOURCE: &str = include_str!("kernels/common/arg_reduce.metal");

/// Process-wide cache of compiled built-in (non custom-kernel) pipelines,
/// keyed by generated kernel name.
/// MLX `get_template_definition`: the explicit-instantiation line appended
/// after the kernel source.
pub(crate) fn builtin_template_def(name: &str, func: &str, args: &[String]) -> String {
    let joined = args.join(", ");
    format!(
        "\ntemplate [[host_name(\"{name}\")]] [[kernel]] decltype({func}<{joined}>) {func}<{joined}>;\n"
    )
}

/// Compile (or fetch) a pipeline from a generated built-in library source.
fn compile_builtin(device: &Device, source: &str, name: &str) -> Result<ComputePipeline> {
    compile_builtin_bool_consts(device, source, name, &[])
}

/// Like [`compile_builtin`] but can specialise Metal bool function constants.
fn compile_builtin_bool_consts(
    device: &Device,
    source: &str,
    name: &str,
    consts: &[(usize, bool)],
) -> Result<ComputePipeline> {
    let c: Vec<(usize, ConstVal)> = consts.iter().map(|(i, v)| (*i, ConstVal::Bool(*v))).collect();
    device.compile_with_constants(source, name, Math::SafeNoLang, &c)
}

/// Compile a library and fetch `name`, specialising bool/int function
/// constants.
fn compile_builtin_typed_consts(
    device: &Device,
    source: &str,
    name: &str,
    consts: &[(usize, ConstVal)],
) -> Result<ComputePipeline> {
    device.compile_with_constants(source, name, Math::SafeNoLang, consts)
}

/// Mirror of MLX `qmv_fast_k_alignment` (`quantized.cpp:147`): the K step in
/// `qmv_fast_impl` must divide K for the fast path to be valid.
fn qmv_fast_k_alignment(bits: i32) -> i32 {
    let pack_factor = 32 / bits.max(1);
    pack_factor * (if bits == 2 { 1 } else { 2 }) * 32
}

/// Affine `quantized_matmul` for a single-batch (B = 1) vector input, matching
/// MLX's `qmv`/`qmv_fast` path (`quantized.cpp:461`). `w` is the packed uint32
/// weight `[N, K*bits/32]`; `scales`/`biases` are `[N, K/group_size]`.
///
/// Returns `(kernel_name, generated_source, func)` for the chosen variant.
pub fn affine_qmv_source(
    x: &Tensor,
    w: &Tensor,
    group_size: i32,
    bits: i32,
) -> Result<(String, String)> {
    let (_, k) = x.dims2();
    let (n, _) = w.dims2();
    let type_str = type_string(x.dtype())?;
    let fast = n % 8 == 0 && (k as i32) % qmv_fast_k_alignment(bits) == 0;
    let func = if fast { "affine_qmv_fast" } else { "affine_qmv" };
    let mut kname = String::from("affine_");
    kname.push_str(if fast { "qmv_fast_" } else { "qmv_" });
    kname.push_str(type_str);
    kname.push_str(&format!("_gs_{group_size}_b_{bits}_batch_0"));
    // template args: type, group_size, bits, batched=false, has_global_scale=false,
    // results_per_simdgroup=4 (bool prints as 0/1, as in C++ ostream).
    let template_def = builtin_template_def(
        &kname,
        func,
        &[
            type_str.to_string(),
            group_size.to_string(),
            bits.to_string(),
            "0".to_string(),
            "0".to_string(),
            "4".to_string(),
        ],
    );
    let source = format!(
        "{MLX_UTILS_PREAMBLE}{MLX_GEMM_PREAMBLE}{MLX_QUANTIZED_UTILS_PREAMBLE}{MLX_QUANTIZED_PREAMBLE}{template_def}"
    );
    Ok((kname, source))
}

/// JIT-compile the affine qmv variant for these shapes through the runtime.
pub fn affine_qmv_pipeline_jit(
    device: &Device,
    x: &Tensor,
    w: &Tensor,
    group_size: i32,
    bits: i32,
) -> Result<ComputePipeline> {
    let (kname, source) = affine_qmv_source(x, w, group_size, bits)?;
    compile_builtin(device, &source, &kname)
}

/// Affine `quantized_matmul` using a JIT-compiled pipeline (see
/// [`affine_qmv_pipeline_jit`]).
pub fn affine_qmv_fast(
    device: &Device,
    x: &Tensor,
    w: &Tensor,
    scales: &Tensor,
    biases: &Tensor,
    group_size: i32,
    bits: i32,
) -> Result<Tensor> {
    let pipeline = affine_qmv_pipeline_jit(device, x, w, group_size, bits)?;
    affine_qmv_dispatch(device, &pipeline, x, w, scales, biases)
}

/// Dispatch a prebuilt affine qmv pipeline over these inputs.
pub fn affine_qmv_dispatch(
    device: &Device,
    pipeline: &ComputePipeline,
    x: &Tensor,
    w: &Tensor,
    scales: &Tensor,
    biases: &Tensor,
) -> Result<Tensor> {
    let mdev = device;
    let (m, k) = x.dims2();
    let (n, _) = w.dims2();
    let out_dtype = x.dtype();

    let ycount = m * n;
    let ybuf = mdev.buffer((ycount) as usize * (out_dtype).size_of(), "qmv_out")?;
    let y = Array::from_parts(mdev, ybuf.clone(), &vec![m, n], out_dtype);

    let guard = mdev.commands.encoder()?;
    let enc = guard.encoder();
    enc.set_pipeline(pipeline);

    let bind = |index: usize, t: &Tensor| -> Result<()> {
        let (ms, layout) = t.buffer_and_layout();
        if !layout.is_contiguous() {
            crate::bail!("affine_qmv: non-contiguous input");
        }
        enc.set_input(index, Some(ms), layout.offset * t.dtype().size_of());
        Ok(())
    };
    bind(0, w)?;
    bind(1, scales)?;
    bind(2, biases)?;
    bind(3, x)?;
    enc.set_output(4, Some(&ybuf), 0);
    let kk = k as i32;
    enc.set_bytes(5, &kk);
    let nn = n as i32;
    enc.set_bytes(6, &nn);

    let bn = 8usize;
    enc.dispatch_groups_size(
        MTLSize {
            width: m,
            height: n.div_ceil(bn),
            depth: 1,
        },
        MTLSize {
            width: 32,
            height: 2,
            depth: 1,
        },
    );
    Ok(y)
}

/// MLX `backend/metal/utils.cpp:8` `type_to_name` — used in the kernel name.
fn type_to_name(dt: DType) -> Result<&'static str> {
    Ok(match dt {
        DType::F32 => "float32",
        DType::F16 => "float16",
        DType::BF16 => "bfloat16",
        DType::F64 => "double",
        DType::U8 => "uint8",
        DType::U32 => "uint32",
        DType::I32 => "int32",
        DType::I64 => "int64",
        DType::I16 => "int16",
        other => crate::bail!("type_to_name: unsupported {other:?}"),
    })
}

/// MLX `backend/metal/utils.h:75` `get_work_per_thread`.
fn work_per_thread(dt: DType, size: usize) -> usize {
    const WPT_THRESHOLD: usize = 1 << 16;
    if size < WPT_THRESHOLD {
        1
    } else {
        (8 / dt.size_of()).max(1)
    }
}

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

    let source = format!(
        "{MLX_UTILS_PREAMBLE}{MLX_UNARY_OPS_PREAMBLE}{MLX_UNARY_PREAMBLE}{defs}"
    );
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

/// MLX `ops::argmax_axis` — `ArgReduce::eval_gpu` with the `argmax_` kernel.
pub fn argmax_axis(device: &Device, x: &Tensor, axis: i32) -> Result<Tensor> {
    arg_reduce_axis(device, x, axis, "ArgMax", "argmax")
}

/// MLX `ops::argmin_axis`.
pub fn argmin_axis(device: &Device, x: &Tensor, axis: i32) -> Result<Tensor> {
    arg_reduce_axis(device, x, axis, "ArgMin", "argmin")
}

/// MLX `get_2d_grid_dims` (`common/utils.cpp`).
fn get_2d_grid_dims(shape: &[usize], strides: &[i64]) -> (usize, usize) {
    let mut gx = 1usize;
    let mut gy = 1usize;
    for i in 0..shape.len() {
        if strides[i] == 0 {
            continue;
        }
        if (gx as u64) * (shape[i] as u64) < u32::MAX as u64 {
            gx *= shape[i];
        } else {
            gy *= shape[i];
        }
    }
    if gy > gx {
        std::mem::swap(&mut gx, &mut gy);
    }
    (gx, gy)
}

fn tg_from_row_size(row_size: usize) -> usize {
    if row_size <= 512 {
        32
    } else if row_size <= 1024 {
        128
    } else {
        ((row_size.div_ceil(4) + 31) / 32 * 32).min(1024)
    }
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
        &[in_t.to_string(), out_t.to_string(), op, "size_t".to_string()],
    );
    let source = format!(
        "{MLX_UTILS_PREAMBLE}{MLX_REDUCE_UTILS_PREAMBLE}{MLX_REDUCE_PREAMBLE}{def}"
    );
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

/// MLX `ops::argsort` along the last axis (`ArgSort::eval_gpu` →
/// `single_block_sort`, `sort.cpp:15`). Returns uint32 indices.
pub fn argsort_last(device: &Device, x: &Tensor) -> Result<Tensor> {
    argsort_axis(device, x, true)
}

/// MLX `ops::argpartition_axis` — MLX directs partition to sort
/// (`sort.cpp:342`), so this is the same kernel.
pub fn argpartition_axis(device: &Device, x: &Tensor, _kth: i32) -> Result<Tensor> {
    argsort_axis(device, x, true)
}

fn argsort_axis(device: &Device, x: &Tensor, argsort: bool) -> Result<Tensor> {
    let mdev = device;
    let (_s, layout) = x.buffer_and_layout();
    if x.dims().is_empty() || !layout.is_contiguous() {
        crate::bail!("argsort: non-contiguous or scalar input");
    }
    let dims = x.dims().to_vec();
    let in_dt = x.dtype();
    let size_sorted_axis = *dims.last().unwrap();
    let n_rows: usize = dims[..dims.len() - 1].iter().product::<usize>().max(1);
    // bn/tn selection from gpu_merge_sort.
    let tn = 4usize;
    let potential_bn = size_sorted_axis.div_ceil(tn);
    let mut bn = if potential_bn > 256 {
        512
    } else if potential_bn > 128 {
        256
    } else if potential_bn > 64 {
        128
    } else if potential_bn > 32 {
        64
    } else {
        32
    };
    if bn == 512 && in_dt.size_of() > 4 {
        bn = 256;
    }
    if size_sorted_axis.div_ceil(bn * tn) > 1 {
        crate::bail!("argsort: multi-block sort not implemented");
    }
    let in_t = type_string(in_dt)?;
    let out_t = type_string(DType::U32)?;
    let in_ty = type_to_name(in_dt)?;
    let out_ty = type_to_name(DType::U32)?;
    let mut kernel_name = String::from("c");
    if argsort {
        kernel_name.push_str("arg");
    }
    kernel_name.push_str(&format!(
        "_block_sort_{in_ty}_{out_ty}_bn{bn}_tn{tn}"
    ));
    let lib_name = kernel_name
        .split_once('_')
        .map(|(_, r)| r)
        .unwrap_or(&kernel_name)
        .to_string();
    let a = |s: &str| s.to_string();
    let mut defs = String::new();
    for (prefix, arg_sort) in [("carg_", "true"), ("c_", "false")] {
        defs.push_str(&builtin_template_def(
            &format!("{prefix}{lib_name}"),
            "block_sort",
            &[a(in_t), a(out_t), a(arg_sort), bn.to_string(), tn.to_string()],
        ));
        defs.push_str(&builtin_template_def(
            &format!("n{prefix}{lib_name}"),
            "block_sort_nc",
            &[a(in_t), a(out_t), a(arg_sort), bn.to_string(), tn.to_string()],
        ));
    }
    let source = format!("{MLX_UTILS_PREAMBLE}{MLX_SORT_PREAMBLE}{defs}");
    let pipeline = compile_builtin(device, &source, &kernel_name)?;

    let count = x.elem_count();
    let obuf = mdev.buffer((count) as usize * (DType::U32).size_of(), "sort_out")?;
    let out = Array::from_parts(mdev, obuf.clone(), &dims.clone(), DType::U32);

    let guard = mdev.commands.encoder()?;
    let enc = guard.encoder();
    enc.set_pipeline(&pipeline);
    {
        let (ms, layout) = x.buffer_and_layout();
        enc.set_input(0, Some(ms), layout.offset * x.dtype().size_of());
    }
    enc.set_output(1, Some(&obuf), 0);
    let ssa = size_sorted_axis as i32;
    enc.set_bytes(2, &ssa);
    let in_stride: i32 = 1;
    enc.set_bytes(3, &in_stride);
    enc.set_bytes(4, &in_stride);
    // contiguous: min non-singleton non-sorted-axis stride (== the row stride)
    let seg: i32 = if dims.len() >= 2 {
        dims[..dims.len() - 1]
            .iter()
            .rposition(|&d| d != 1)
            .map(|i| layout.stride()[i] as i32)
            .unwrap_or(i32::MAX)
    } else {
        i32::MAX
    };
    enc.set_bytes(5, &seg);
    enc.set_bytes(6, &seg);

    enc.dispatch_groups_size(
        MTLSize {
            width: 1,
            height: n_rows,
            depth: 1,
        },
        MTLSize {
            width: bn,
            height: 1,
            depth: 1,
        },
    );
    Ok(out)
}

/// MLX `fast::rms_norm` (`RMSNorm::eval_gpu`, `normalization.cpp:17`). The
/// `has_w` function constant selects the weighted scheme.
pub fn rms_norm(
    device: &Device,
    x: &Tensor,
    weight: Option<&Tensor>,
    eps: f32,
) -> Result<Tensor> {
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
    let mut defs = builtin_template_def(
        &format!("rms_{ty}"),
        "rms_single_row",
        &[in_t.to_string()],
    );
    defs.push_str(&builtin_template_def(
        &format!("rms_looped_{ty}"),
        "rms_looped",
        &[in_t.to_string()],
    ));
    let source = format!("{MLX_UTILS_PREAMBLE}{MLX_RMS_NORM_SOURCE}{defs}");
    // function constant 20 is `has_w`. MLX passes a scalar `1` weight when no
    // weight is given (`fast.cpp:112`), so the kernel always runs the weighted
    // path with that scalar.
    let ones;
    let weight = match weight {
        Some(w) => w,
        None => {
            ones = Array::scalar_of(device, 1f32, dt)?;
            &ones
        }
    };
    let pipeline =
        compile_builtin_bool_consts(device, &source, &kernel_name, &[(20, true)])?;

    let count = x.elem_count();
    let obuf = mdev.buffer((count) as usize * (dt).size_of(), "rms_out")?;
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
        enc.set_input(1, Some(ms), wl.start_offset());
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

/// MLX `get_block_dims` (`common/utils.cpp:83`).
pub fn get_block_dims(dim0: usize, dim1: usize, dim2: usize, pow2: i32) -> (usize, usize, usize) {
    let mut pows = [0i32; 3];
    let mut sum = 0i32;
    loop {
        let presum = sum;
        if dim0 as i64 >= 1i64 << (pows[0] + 1) {
            pows[0] += 1;
            sum += 1;
        }
        if sum == 10 {
            break;
        }
        if dim1 as i64 >= 1i64 << (pows[1] + 1) {
            pows[1] += 1;
            sum += 1;
        }
        if sum == 10 {
            break;
        }
        if dim2 as i64 >= 1i64 << (pows[2] + 1) {
            pows[2] += 1;
            sum += 1;
        }
        if sum == presum || sum == pow2 {
            break;
        }
    }
    (1usize << pows[0], 1usize << pows[1], 1usize << pows[2])
}

/// MLX `fast::scaled_dot_product_attention` vector path (`sdpa_vector`,
/// `scaled_dot_product_attention.cpp:364`) — single/few query rows, no array
/// mask, no sinks. `q`/`k`/`v` are `[B, H, L, D]`, `[B, Hk, N, D]`, `[B, Hk, N, V]`.
pub fn sdpa_vector(
    device: &Device,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    scale: f32,
    do_causal: bool,
) -> Result<Tensor> {
    let mdev = device;
    let qt = type_string(q.dtype())?;
    let d = q.dims()[3];
    let vd = v.dims()[3];
    let kernel_name = format!("sdpa_vector_{qt}_{d}_{vd}");
    let def = builtin_template_def(
        &kernel_name,
        "sdpa_vector",
        &[qt.to_string(), d.to_string(), vd.to_string()],
    );
    let source = format!("{MLX_UTILS_PREAMBLE}{MLX_SDPA_VECTOR_PREAMBLE}{def}");
    let pipeline = compile_builtin_bool_consts(
        device,
        &source,
        &kernel_name,
        &[
            (20, false), // has_mask
            (21, false), // query_transposed
            (22, do_causal),
            (23, false), // bool_mask
            (24, false), // float_mask
            (25, false), // has_sinks
        ],
    )?;

    let qd = q.dims().to_vec();
    let (b, h, ql) = (qd[0], qd[1], qd[2]);
    let out_dims = vec![b, h, ql, vd];
    let count: usize = out_dims.iter().product();
    let obuf = mdev.buffer((count) as usize * (q.dtype()).size_of(), "sdpa_out")?;
    let out = Array::from_parts(mdev, obuf.clone(), &out_dims, q.dtype());

    let gqa_factor = (q.dims()[1] / k.dims()[1]) as i32;
    let n = k.dims()[2] as i32;
    let k_head_stride: usize = if k.dims()[1] == 1 {
        k.stride()[0]
    } else {
        k.stride()[1]
    };
    let k_seq_stride: usize = k.stride()[2];
    let v_head_stride: usize = if v.dims()[1] == 1 {
        v.stride()[0]
    } else {
        v.stride()[1]
    };
    let v_seq_stride: usize = v.stride()[2];

    let guard = mdev.commands.encoder()?;
    let enc = guard.encoder();
    enc.set_pipeline(&pipeline);
    let bind = |index: usize, t: &Tensor| -> Result<()> {
        let (ms, layout) = t.buffer_and_layout();
        enc.set_input(index, Some(ms), layout.offset * t.dtype().size_of());
        Ok(())
    };
    bind(0, q)?;
    bind(1, k)?;
    bind(2, v)?;
    enc.set_output(3, Some(&obuf), 0);
    enc.set_bytes(4, &gqa_factor);
    enc.set_bytes(5, &n);
    enc.set_bytes(6, &k_head_stride);
    enc.set_bytes(7, &k_seq_stride);
    enc.set_bytes(8, &v_head_stride);
    enc.set_bytes(9, &v_seq_stride);
    enc.set_bytes(10, &scale);

    enc.dispatch_groups_size(
        MTLSize {
            width: b * h,
            height: ql,
            depth: 1,
        },
        MTLSize {
            width: 1024,
            height: 1,
            depth: 1,
        },
    );
    Ok(out)
}


/// MLX `steel_matmul_regular_axpby_nax` (`matmul.cpp:178`), no batch, no
/// out-source. `a` is `[M,K]`; `b` is `[K,N]` (or `[N,K]` when `transpose_b`).
/// Source for one `steel_gemm_fused_nax` instantiation (the NAX GEMM).
pub fn matmul_nax_source(
    base_name: &str,
    a_ty: &str,
    out_ty: &str,
    bm: usize,
    bn: usize,
    bk: usize,
    wm: usize,
    wn: usize,
    transpose_a: bool,
    transpose_b: bool,
) -> String {
    let _ = (a_ty, out_ty);
    // MLX instantiates the NAX GEMM on the *output* dtype's Metal type
    // (`get_type_string(out.dtype())`), so f32 matmuls use `gemm<float, ...>`.
    let metal_ty = match out_ty {
        "float32" => "float",
        "bfloat16" => "bfloat16_t",
        "float16" => "half",
        other => other,
    };
    format!(
        "{NAX_GEMM_HEADER}\nusing namespace metal;\n\
         instantiate_kernel(\"{base_name}\", gemm, {metal_ty}, {bm}, {bn}, {bk}, {wm}, {wn}, {}, {})\n",
        if transpose_a { "true" } else { "false" },
        if transpose_b { "true" } else { "false" },
    )
}

pub fn matmul_nax(
    device: &Device,
    metallib: Option<&str>,
    a: &Tensor,
    b: &Tensor,
    transpose_a: bool,
    transpose_b: bool,
) -> Result<Tensor> {
    let mdev = device;
    let ar = a.rank();
    let br = b.rank();
    if ar != 2 || br != 2 {
        crate::bail!("matmul_nax: 2-D only");
    }
    let m = a.dims()[0];
    let k = a.dims()[1];
    let n = if transpose_b { b.dims()[0] } else { b.dims()[1] };
    let lda = if transpose_a { m } else { k } as i32;
    let ldb = if transpose_b { k } else { n } as i32;
    let ldd = n as i32;

    let a_ty = type_to_name(a.dtype())?;
    let out_ty = type_to_name(a.dtype())?;
    let devc = mdev.architecture_name().chars().last().unwrap_or('s');
    let (mut bm, bn, mut bk, mut wm, wn) = (128usize, 128usize, 512usize, 4usize, 4usize);
    if devc == 's' || devc == 'c' || devc == 'd' {
        bk = if k >= 8192 && k > m + n { 64 } else { 256 };
        bm = 64;
        wm = 2;
    }
    let base_name = format!(
        "steel_gemm_fused_nax_{}{}_{a_ty}_{out_ty}_bm{bm}_bn{bn}_bk{bk}_wm{wm}_wn{wn}",
        if transpose_a { 't' } else { 'n' },
        if transpose_b { 't' } else { 'n' },
    );
    let align_m = m % bm == 0;
    let align_n = n % bn == 0;
    let align_k = k % bk == 0;
    let pipeline = compile_nax_jit(
        device,
        &matmul_nax_source(&base_name, a_ty, out_ty, bm, bn, bk, wm, wn, transpose_a, transpose_b),
        &base_name,
        &[
            (10, false),
            (100, false),
            (110, false),
            (200, align_m),
            (201, align_n),
            (202, align_k),
        ],
    )?;
    let _ = metallib;

    let count = m * n;
    let obuf = mdev.buffer((count) as usize * (a.dtype()).size_of(), "gemm_out")?;
    let out = Array::from_parts(mdev, obuf.clone(), &vec![m, n], a.dtype());

    let tn = n.div_ceil(bn);
    let tm = m.div_ceil(bm);
    let swizzle_log: i32 = if devc == 's' || devc == 'c' || devc == 'd' { 2 } else if tm <= 3 { 0 } else { 1 };
    let mut p: Vec<u8> = Vec::with_capacity(72);
    for v in [m as i32, n as i32, k as i32, lda, ldb, ldd, tn as i32, tm as i32] {
        p.extend_from_slice(&v.to_le_bytes());
    }
    for v in [0i64, 0, 0] {
        p.extend_from_slice(&v.to_le_bytes());
    }
    for v in [swizzle_log, (k / bk) as i32, 0i32] {
        p.extend_from_slice(&v.to_le_bytes());
    }
    p.resize(72, 0);

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
    enc.set_output(3, Some(&obuf), 0);
    enc.set_bytes_directly(4, p.len(), p.as_ptr().cast());

    let tile = 1usize << swizzle_log;
    let tm2 = tm.div_ceil(tile);
    let tn2 = tn * tile;
    enc.dispatch_groups_size(
        MTLSize { width: tn2, height: tm2, depth: 1 },
        MTLSize { width: 32, height: wn, depth: wm },
    );
    Ok(out)
}

/// MLX `collapse_contiguous_dims` (`backend/common/utils.cpp:24`).
///
/// Collapses runs of axes that are contiguous for every input into a single
/// axis, returning one shape plus the collapsed strides per input. Used by the
/// gather/qmm kernels to describe batched shapes and gather-index strides.
pub fn collapse_contiguous_dims(
    shape: &[i32],
    strides: &[Vec<i64>],
    size_cap: i64,
) -> (Vec<i32>, Vec<Vec<i64>>) {
    let mut to_collapse: Vec<i32> = Vec::new();
    if !shape.is_empty() {
        if shape[0] != 1 {
            to_collapse.push(0);
        }
        let mut size: i64 = shape[0] as i64;
        for i in 1..shape.len() {
            let mut contiguous = true;
            size *= shape[i] as i64;
            for st in strides {
                if st[i] * shape[i] as i64 != st[i - 1] || size > size_cap {
                    contiguous = false;
                    size = shape[i] as i64;
                    break;
                }
            }
            if !contiguous {
                to_collapse.push(-1);
            }
            if shape[i] != 1 {
                to_collapse.push(i as i32);
            }
        }
        to_collapse.push(-1);
    }

    let mut out_shape: Vec<i32> = Vec::new();
    let mut out_strides: Vec<Vec<i64>> = vec![Vec::new(); strides.len()];
    let mut i = 0usize;
    loop {
        while i < to_collapse.len() && to_collapse[i] == -1 {
            i += 1;
        }
        if i == to_collapse.len() {
            break;
        }
        let mut current_shape = shape[to_collapse[i] as usize];
        let mut k = i;
        loop {
            k += 1;
            if to_collapse[k] == -1 {
                break;
            }
            current_shape *= shape[to_collapse[k] as usize];
        }
        out_shape.push(current_shape);
        for (j, st) in strides.iter().enumerate() {
            out_strides[j].push(st[to_collapse[k - 1] as usize]);
        }
        i = k + 1;
    }

    if !shape.is_empty() && out_shape.is_empty() {
        out_shape.push(1);
        for os in out_strides.iter_mut() {
            os.push(0);
        }
    }
    (out_shape, out_strides)
}

pub fn collapse_default_cap() -> i64 {
    i64::from(i32::MAX)
}

/// Accumulates the small scalar/vector blobs MLX writes with `set_bytes`
/// (`int`) and `set_vector_bytes` (`vector<int>` / `vector<int64_t>`), keeping
/// the order in which they were written so they can be bound to consecutive
/// buffer indices. Metal copies on set, but the bytes must outlive dispatch.
pub struct ByteBlob {
    buf: Vec<u8>,
    fields: Vec<(usize, usize)>,
}

impl ByteBlob {
    pub fn new() -> Self {
        Self { buf: Vec::new(), fields: Vec::new() }
    }
    fn push(&mut self, bytes: &[u8]) {
        self.fields.push((self.buf.len(), bytes.len()));
        self.buf.extend_from_slice(bytes);
    }
    pub fn i32(&mut self, v: i32) {
        self.push(&v.to_le_bytes());
    }
    pub fn i32s(&mut self, vs: &[i32]) {
        self.push(&vs.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>());
    }
    pub fn i64s(&mut self, vs: &[i64]) {
        self.push(&vs.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>());
    }
    pub fn len_fields(&self) -> usize {
        self.fields.len()
    }
    /// Bind each field to `base`, `base+1`, … via `set_bytes_directly`.
    pub fn bind(
        &self,
        enc: &crate::runtime::ComputeEncoder,
        base: usize,
    ) {
        for (i, (off, len)) in self.fields.iter().enumerate() {
            enc.set_bytes_directly(base + i, *len, self.buf[*off..].as_ptr().cast());
        }
    }
}

impl Default for ByteBlob {
    fn default() -> Self {
        Self::new()
    }
}

/// MLX `add_strides_and_shapes` (`quantized.cpp:151`). Each argument is a
/// `(shape, strides)` pair in elements (strides stored `int64_t`).
pub fn add_strides_and_shapes(
    x: (&[usize], &[usize]),
    w: (&[usize], &[usize]),
    scales_strides: &[usize],
    biases_strides: Option<&[usize]>,
) -> ByteBlob {
    let mut b = ByteBlob::new();
    b.i32(x.0.len() as i32 - 2);
    b.i32s(&x.0.iter().map(|&v| v as i32).collect::<Vec<_>>());
    b.i64s(&x.1.iter().map(|&v| v as i64).collect::<Vec<_>>());
    b.i32(w.0.len() as i32 - 2);
    b.i32s(&w.0.iter().map(|&v| v as i32).collect::<Vec<_>>());
    b.i64s(&w.1.iter().map(|&v| v as i64).collect::<Vec<_>>());
    b.i64s(&scales_strides.iter().map(|&v| v as i64).collect::<Vec<_>>());
    if let Some(bs) = biases_strides {
        b.i64s(&bs.iter().map(|&v| v as i64).collect::<Vec<_>>());
    }
    b
}

/// MLX `add_gather_strides_and_shapes` (`quantized.cpp:181`). Collapses the lhs
/// index shape against the lhs/rhs index strides, then writes the shared shape
/// and both collapsed stride vectors.
pub fn add_gather_strides_and_shapes(
    lhs: (&[usize], &[usize]),
    rhs: (&[usize], &[usize]),
) -> ByteBlob {
    let shape: Vec<i32> = lhs.0.iter().map(|&v| v as i32).collect();
    let strides = vec![
        lhs.1.iter().map(|&v| v as i64).collect::<Vec<_>>(),
        rhs.1.iter().map(|&v| v as i64).collect::<Vec<_>>(),
    ];
    let (shape, strides) =
        collapse_contiguous_dims(&shape, &strides, collapse_default_cap());
    let mut b = ByteBlob::new();
    b.i32(shape.len() as i32);
    b.i32s(&shape);
    b.i64s(&strides[0]);
    b.i64s(&strides[1]);
    b
}

/// Compile `source` to a content-addressed `.metallib` under the temp dir and
/// return its path. Compiling one instantiation takes ~0.2 s.
/// `mlx_gemm.metal` / `gemv.metal` (Apple MLX steel kernels vendored
/// through the kernels crate; see NOTICE). The general `matmul` path uses
/// these so its accumulation matches the reference bit-for-bit.
const DENSE_GEMM: &str = include_str!("kernels/common/dense_gemm.metal");
const DENSE_GEMV: &str = include_str!("kernels/common/dense_gemv.metal");

/// `GemmParams` (`kernels/mlx_gemm.rs`).
#[repr(C)]
struct GemmParams {
    m: i32,
    n: i32,
    k: i32,
    lda: i32,
    ldb: i32,
    ldd: i32,
    tiles_n: i32,
    tiles_m: i32,
    batch_stride_a: isize,
    batch_stride_b: isize,
    batch_stride_d: isize,
    swizzle_log: i32,
    gemm_k_iterations_aligned: i32,
    batch_ndim: i32,
}

/// The Metal device class tile selection keys on: the last character
/// of the architecture name (`applegpu_*`).
fn device_type(rt: &Arc<crate::runtime::MetalRuntime>) -> char {
    rt.architecture_name()
        .chars()
        .last()
        .unwrap_or('m')
}



/// MLX `affine_dequantize` (quantized.cpp): JIT-compile the dequantize kernel
/// for these template args and dispatch it. Replaces the host dequantize in
/// `ops::dequantize` (a full GPU->CPU->GPU round trip per call; the embedding
/// forward runs it per token, which showed up as ~130 `cast_host` allocs/token).
pub fn affine_dequantize(
    device: &Device,
    w: &Tensor,
    scales: &Tensor,
    biases: &Tensor,
    group_size: i32,
    bits: i32,
) -> Result<Tensor> {
    
    let mdev = device;
    // w is [.., kq] u32 (packed); scales/biases [.., groups]; out [.., k].
    let kq = *w.dims().last().unwrap();
    let per_word = 32 / bits as usize;
    let k = kq * per_word;
    let count: usize = w.size() / kq * k;
    let out_dims: Vec<usize> = {
        let mut d2 = w.dims().to_vec();
        *d2.last_mut().unwrap() = k;
        d2
    };
    let obuf = mdev.buffer(count * scales.dtype().size_of(), "deq_out")?;
    let out = Array::from_parts(mdev, obuf.clone(), &out_dims, scales.dtype());

    let ty = type_string(scales.dtype())?;
    // MLX instantiates the *out* dtype as T; w is read as bytes.
    let kname = format!("affine_dequantize_{ty}_gs_{group_size}_b_{bits}");
    let template_def = builtin_template_def(
        &kname,
        "affine_dequantize",
        &[
            ty.to_string(),
            group_size.to_string(),
            bits.to_string(),
            "0".to_string(),
        ],
    );
    let source = format!(
        "{MLX_UTILS_PREAMBLE}{MLX_GEMM_PREAMBLE}{MLX_QUANTIZED_UTILS_PREAMBLE}{MLX_QUANTIZED_PREAMBLE}{template_def}"
    );
    let pipeline = compile_builtin(device, &source, &kname)?;

    // MLX dims: one thread per pack (8 values for 4-bit); 2-D grid.
    let packs_per_int: usize = if bits == 3 || bits == 5 { 8 } else { 8 / bits as usize };
    let nthreads = count / packs_per_int.max(1);
    let mut grid_shape = w.dims().to_vec();
    *grid_shape.last_mut().unwrap() *= 4; // uint8 per uint32
    let (gw, gh) = {
        // get_2d_grid_dims over the byte-expanded shape
        let strides: Vec<i64> = {
            let mut sv = vec![1i64; grid_shape.len()];
            for i in (0..grid_shape.len().saturating_sub(1)).rev() {
                sv[i] = sv[i + 1] * grid_shape[i + 1] as i64;
            }
            sv
        };
        get_2d_grid_dims(&grid_shape, &strides)
    };
    let tg = pipeline.max_total_threads_per_threadgroup().min(nthreads).max(1);

    let guard = mdev.commands.encoder()?;
    let enc = guard.encoder();
    enc.set_pipeline(&pipeline);
    enc.set_input(0, Some(w.buffer_and_layout().0), w.buffer_and_layout().1.offset * 4);
    enc.set_input(1, Some(scales.buffer_and_layout().0), scales.buffer_and_layout().1.offset * scales.dtype().size_of());
    enc.set_input(2, Some(biases.buffer_and_layout().0), biases.buffer_and_layout().1.offset * biases.dtype().size_of());
    enc.set_output(3, Some(out.buffer_and_layout().0), 0);
    enc.dispatch_threads_3d((gw.max(1), gh.max(1), 1), (tg, 1, 1));
    Ok(out)
}

/// `call_mlx_gemv` (kernels/mlx_gemm.rs:270): the M=1 / N=1 path of the
/// reference `matmul`, so its numerics match bit-for-bit.
#[allow(clippy::too_many_arguments)]
fn dense_gemv(
    rt: &Arc<MetalRuntime>,
    (b, m, n, k): (usize, usize, usize, usize),
    lhs_stride: &[usize],
    lhs_offset: usize,
    lhs_buffer: &Arc<Buffer>,
    rhs_stride: &[usize],
    rhs_offset: usize,
    rhs_buffer: &Arc<Buffer>,
    out: &crate::array::Array,
    dt: Dtype,
) -> Result<()> {
    let rhs_m1 = rhs_stride[rhs_stride.len() - 1];
    let rhs_m2 = rhs_stride[rhs_stride.len() - 2];
    let lhs_m1 = lhs_stride[lhs_stride.len() - 1];
    let lhs_m2 = lhs_stride[lhs_stride.len() - 2];

    let (lda, a_trans) = if (lhs_m1 == 1 || k == 1) && (lhs_m2 == k || m == 1) {
        (k as i32, false)
    } else if (lhs_m1 == m || k == 1) && (lhs_m2 == 1 || m == 1) {
        (m as i32, true)
    } else {
        return Err(Error::Msg(format!(
            "mlx gemv: non-contiguous lhs {lhs_stride:?} mnk {m},{n},{k}"
        )));
    };
    let (ldb, b_trans) = if (rhs_m1 == 1 || n == 1) && (rhs_m2 == n || k == 1) {
        (n as i32, false)
    } else if (rhs_m1 == k || n == 1) && (rhs_m2 == 1 || k == 1) {
        (k as i32, true)
    } else {
        return Err(Error::Msg(format!(
            "mlx gemv: non-contiguous rhs {rhs_stride:?} mnk {m},{n},{k}"
        )));
    };

    let is_b_matrix = n != 1;
    let transpose_mat = if is_b_matrix { !b_trans } else { a_trans };
    let mat_ld = if is_b_matrix { ldb as usize } else { lda as usize };
    let in_vec_size = k;
    let out_vec_size = if is_b_matrix { n } else { m };

    let (mat_buffer, mat_offset, vec_buffer, vec_offset) = if is_b_matrix {
        (rhs_buffer, rhs_offset, lhs_buffer, lhs_offset)
    } else {
        (lhs_buffer, lhs_offset, rhs_buffer, rhs_offset)
    };

    let vec_batch_stride: i64 = if is_b_matrix {
        if lhs_stride.len() > 2 { lhs_stride[lhs_stride.len() - 3] as i64 } else { k as i64 }
    } else {
        if rhs_stride.len() > 2 { rhs_stride[rhs_stride.len() - 3] as i64 } else { k as i64 }
    };
    let mat_batch_stride: i64 = if is_b_matrix {
        if rhs_stride.len() > 2 { rhs_stride[rhs_stride.len() - 3] as i64 } else { 0 }
    } else {
        if lhs_stride.len() > 2 { lhs_stride[lhs_stride.len() - 3] as i64 } else { 0 }
    };

    let (bm, bn, sm, sn, tm, tn) = if transpose_mat {
        let (sm, sn) = if in_vec_size >= 8192 && out_vec_size >= 2048 {
            (4usize, 8usize)
        } else {
            (8, 4)
        };
        let bn = if out_vec_size >= 2048 {
            16usize
        } else if out_vec_size >= 512 {
            4
        } else {
            2
        };
        let tn: usize = if out_vec_size < 4 { 1 } else { 4 };
        (1usize, bn, sm, sn, 4usize, tn)
    } else {
        let (bm, bn, sm, sn): (usize, usize, usize, usize) = if in_vec_size <= 64 {
            (1, 1, 8, 4)
        } else if in_vec_size >= 16 * out_vec_size {
            (1, 8, 1, 32)
        } else if out_vec_size >= 4096 {
            (8, 1, 1, 32)
        } else {
            (4, 1, 1, 32)
        };
        let tm: usize = if out_vec_size < 4 { 1 } else { 4 };
        (bm, bn, sm, sn, tm, 4usize)
    };

    let dtype_str = match dt {
        Dtype::Float32 => "float32",
        Dtype::Float16 => "float16",
        Dtype::Bfloat16 => "bfloat16",
        other => return Err(Error::Msg(format!("mlx gemv dtype {other:?}"))),
    };
    let kernel_prefix = if transpose_mat { "gemv_t" } else { "gemv" };
    let name = format!(
        "{}_{}_bm{}_bn{}_sm{}_sn{}_tm{}_tn{}_nc0_axpby0",
        kernel_prefix, dtype_str, bm, bn, sm, sn, tm, tn
    );
    let source = if transpose_mat { DENSE_GEMV } else { DENSE_GEMV };
    let pipeline = rt.compile_with_constants(
        source,
        &name,
        crate::runtime::Math::SafeNoLang,
        &[],
    )?;

    let _esz = dt.size_of();
    let guard = rt.commands.encoder()?;
    let enc = guard.encoder();
    enc.set_pipeline(&pipeline);
    enc.set_input(0, Some(mat_buffer), mat_offset);
    enc.set_input(1, Some(vec_buffer), vec_offset);
    enc.set_output(3, Some(out.buffer_and_layout().0), 0);
    enc.set_bytes(4, &(in_vec_size as i32));
    enc.set_bytes(5, &(out_vec_size as i32));
    enc.set_bytes(6, &(mat_ld as i32));
    enc.set_bytes(7, &1.0f32); // alpha
    enc.set_bytes(8, &0.0f32); // beta
    enc.set_bytes(9, &1i32); // batch_ndim
    let batch_shape = [b as i32];
    let vec_strides = [vec_batch_stride];
    let mat_strides = [mat_batch_stride];
    let bias_strides = [0i64];
    enc.set_bytes_directly(10, 4, batch_shape.as_ptr().cast());
    enc.set_bytes_directly(11, 8, vec_strides.as_ptr().cast());
    enc.set_bytes_directly(12, 8, mat_strides.as_ptr().cast());
    enc.set_bytes_directly(13, 8, bias_strides.as_ptr().cast());
    enc.set_bytes(14, &1i32); // bias_stride

    let n_out_per_tgp = if transpose_mat { bn * sn * tn } else { bm * sm * tm };
    let n_tgp = out_vec_size.div_ceil(n_out_per_tgp);
    enc.dispatch_groups_3d((n_tgp, 1, b), (32, bn, bm));
    Ok(())
}

/// `call_mlx_gemm` (kernels/mlx_gemm.rs:453): the reference `matmul`
/// kernel, tile selection, batch collapse and parameters, so the general
/// matmul path is bit-identical to the reference engine's.
#[allow(clippy::too_many_arguments)]
pub fn dense_gemm(
    rt: &Arc<MetalRuntime>,
    (b, m, n, k): (usize, usize, usize, usize),
    lhs_stride: &[usize],
    lhs_offset: usize,
    lhs_buffer: &Arc<Buffer>,
    rhs_stride: &[usize],
    rhs_offset: usize,
    rhs_buffer: &Arc<Buffer>,
    out: &crate::array::Array,
    dt: Dtype,
) -> Result<()> {
    let rhs_m1 = rhs_stride[rhs_stride.len() - 1];
    let rhs_m2 = rhs_stride[rhs_stride.len() - 2];
    let lhs_m1 = lhs_stride[lhs_stride.len() - 1];
    let lhs_m2 = lhs_stride[lhs_stride.len() - 2];

    let (lda, a_trans) = if (lhs_m1 == 1 || k == 1) && (lhs_m2 == k || m == 1) {
        (k as i32, false)
    } else if (lhs_m1 == m || k == 1) && (lhs_m2 == 1 || m == 1) {
        (m as i32, true)
    } else {
        return Err(Error::Msg(format!(
            "mlx gemm: non-contiguous lhs {lhs_stride:?} mnk {m},{n},{k}"
        )));
    };
    let (ldb, b_trans) = if (rhs_m1 == 1 || n == 1) && (rhs_m2 == n || k == 1) {
        (n as i32, false)
    } else if (rhs_m1 == k || n == 1) && (rhs_m2 == 1 || k == 1) {
        (k as i32, true)
    } else {
        return Err(Error::Msg(format!(
            "mlx gemm: non-contiguous rhs {rhs_stride:?} mnk {m},{n},{k}"
        )));
    };

    if m == 1 || n == 1 {
        return dense_gemv(
            rt, (b, m, n, k), lhs_stride, lhs_offset, lhs_buffer, rhs_stride,
            rhs_offset, rhs_buffer, out, dt,
        );
    }

    // Batch collapse: A contiguous in batch + B broadcast (2-D rhs) -> [b*m, k].
    let (effective_batch, effective_m, batch_collapsed) = {
        let mut eb = b;
        let mut em = m;
        let mut collapsed = false;
        if b > 1 && !a_trans {
            let a_batch = if lhs_stride.len() > 2 { lhs_stride[lhs_stride.len() - 3] } else { m * k };
            let b_batch = if rhs_stride.len() > 2 { rhs_stride[rhs_stride.len() - 3] } else { 0 };
            if a_batch == m * k && b_batch == 0 {
                eb = 1;
                em = b * m;
                collapsed = true;
            }
        }
        (eb, em, collapsed)
    };
    let m = effective_m;
    let b = effective_batch;

    // Tile selection (MLX GEMM_TPARAM_MACRO; medium device = arch ending 's').
    let total_output = b * m * n;
    let is_large = total_output >= (1 << 20);
    let devc = device_type(rt);
    let (bm, bn, bk, wm, wn) = if m < 16 {
        (32, 32, 16, 2, 2)
    } else if devc == 's' || devc == 'm' {
        if dt == Dtype::Float32 {
            if !is_large {
                if !a_trans && b_trans { (32, 64, 16, 1, 2) } else { (64, 32, 32, 2, 2) }
            } else {
                (64, 64, 16, 2, 2)
            }
        } else if is_large {
            if 2 * m.max(n) > k {
                (64, 64, 16, 1, 2)
            } else if !a_trans && b_trans {
                (64, 32, 32, 2, 2)
            } else {
                (32, 64, 16, 1, 2)
            }
        } else if !a_trans && b_trans {
            (64, 32, 32, 2, 2)
        } else {
            (64, 64, 16, 1, 2)
        }
    } else if devc == 'd' {
        if is_large {
            if dt != Dtype::Float32 {
                if 2 * m.max(n) > k {
                    (64, 64, 16, 1, 2)
                } else if !a_trans && b_trans {
                    (64, 32, 32, 2, 2)
                } else {
                    (32, 64, 16, 1, 2)
                }
            } else {
                (64, 64, 16, 2, 2)
            }
        } else if dt != Dtype::Float32 {
            if !a_trans && b_trans { (64, 32, 32, 2, 2) } else { (64, 64, 16, 1, 2) }
        } else if !a_trans && b_trans {
            (32, 64, 16, 1, 2)
        } else {
            (64, 32, 32, 2, 2)
        }
    } else {
        if !a_trans && b_trans {
            (64, 32, 32, 2, 2)
        } else if dt != Dtype::Float32 {
            (64, 64, 16, 1, 2)
        } else {
            (64, 64, 16, 2, 2)
        }
    };

    let has_batch = b > 1;
    let swizzle_log = 0;
    let tile_swizzle = 1usize << swizzle_log;
    let tn = n.div_ceil(bn);
    let tm = m.div_ceil(bm);
    let tn = tn * tile_swizzle;
    let tm = tm.div_ceil(tile_swizzle);

    let (batch_stride_a, batch_stride_b) = if batch_collapsed {
        (0isize, 0isize)
    } else {
        let a_stride = if lhs_stride.len() > 2 {
            lhs_stride[lhs_stride.len() - 3] as isize
        } else {
            (m * k) as isize
        };
        let b_stride = if rhs_stride.len() > 2 {
            rhs_stride[rhs_stride.len() - 3] as isize
        } else {
            (n * k) as isize
        };
        (a_stride, b_stride)
    };

    let params = GemmParams {
        m: m as i32,
        n: n as i32,
        k: k as i32,
        lda: if batch_collapsed { k as i32 } else { lda },
        ldb,
        ldd: n as i32,
        tiles_n: tn as i32,
        tiles_m: tm as i32,
        swizzle_log,
        batch_stride_a,
        batch_stride_b,
        batch_stride_d: (m * n) as isize,
        batch_ndim: 1,
        gemm_k_iterations_aligned: (k / bk) as i32,
    };

    let dtype_str = match dt {
        Dtype::Float32 => "f32",
        Dtype::Float16 => "f16",
        Dtype::Bfloat16 => "bf16",
        other => return Err(Error::Msg(format!("mlx gemm dtype {other:?}"))),
    };
    let trans_str = match (a_trans, b_trans) {
        (false, false) => "nn",
        (true, false) => "tn",
        (false, true) => "nt",
        (true, true) => "tt",
    };
    let name = format!(
        "gemm_{}_{}_{}_{}_{}_{}_{}_{}",
        trans_str, dtype_str, dtype_str, bm, bn, bk, wm, wn
    );
    let consts: Vec<(usize, ConstVal)> = vec![
        (10, ConstVal::Bool(has_batch)),
        (100, ConstVal::Bool(false)),
        (110, ConstVal::Bool(false)),
        (200, ConstVal::Bool(m % bm == 0)),
        (201, ConstVal::Bool(n % bn == 0)),
        (202, ConstVal::Bool(k % bk == 0)),
        (300, ConstVal::Bool(false)),
    ];
    let pipeline =
        rt.compile_with_constants(DENSE_GEMM, &name, crate::runtime::Math::SafeNoLang, &consts)?;

    let guard = rt.commands.encoder()?;
    let enc = guard.encoder();
    enc.set_pipeline(&pipeline);
    enc.set_input(0, Some(lhs_buffer), lhs_offset);
    enc.set_input(1, Some(rhs_buffer), rhs_offset);
    enc.set_output(3, Some(out.buffer_and_layout().0), 0);
    enc.set_bytes(4, &params);
    enc.set_bytes(6, &(b as i32));
    let batch_strides = [batch_stride_a, batch_stride_b];
    enc.set_bytes_directly(7, 16, batch_strides.as_ptr().cast());
    enc.dispatch_groups_3d((tn, tm, b), (32, wn, wm));
    Ok(())
}

/// JIT-compile an NAX kernel from its flattened source and memoize the
/// pipeline in the runtime (replaces the AOT `xcrun metal` -> metallib path).
/// Same compile options as the reference (`set_compile_options`: Safe + Precise
/// + language version), function constants supplied at pipeline creation.
fn compile_nax_jit(
    device: &Device,
    source: &str,
    name: &str,
    consts: &[(usize, bool)],
) -> Result<ComputePipeline> {
    let consts: Vec<(usize, ConstVal)> = consts
        .iter()
        .map(|&(i, v)| (i, ConstVal::Bool(v)))
        .collect();
    device.compile_with_constants(source, name, crate::runtime::Math::Safe, &consts)
}

/// Source for one `affine_gather_qmm_rhs_nax` instantiation (transpose path).
pub fn gather_qmm_rhs_nax_source(kname: &str, bm: usize, group_size: i32, bits: i32) -> String {
    format!(
        "{NAX_QUANT_HEADER}\nusing namespace metal;\n\
         instantiate_kernel(\"{kname}\", affine_gather_qmm_rhs_nax, bfloat, {group_size}, {bits}, {bm}, 64, 64, 2, 2, true)\n"
    )
}

/// MLX `gather_qmm_rhs_nax` (`quantized.cpp:1450`): the NAX affine gather-GEMM
/// with right-sorted indices. This is the branch the sorted MoE expert matmuls
/// actually take (`M == 1`, `B >= 16`, `B / E >= 4`, `right_sorted`), because
/// the engine feeds a `[rows, 1, K]` lhs so `M` collapses to 1 and `B` becomes
/// the row count. `indices` is one expert id per output row, walked in order so
/// consecutive rows sharing an expert reuse the loaded weight block.
///
/// `x` is `[M, 1, K]` (or `[M, K]`), `w` is `[E, N, K]` quantized, and
/// `out_dims` is the caller's output shape. `M`/`N`/`K` are derived from the
/// tensors, matching the arguments `GatherQMM::eval_gpu` passes down.
pub fn gather_qmm_rhs_nax(
    device: &Device,
    metallib: Option<&str>,
    x: &Tensor,
    w: &Tensor,
    scales: &Tensor,
    biases: &Tensor,
    indices: &Tensor,
    out_dims: Vec<usize>,
    group_size: i32,
    bits: i32,
) -> Result<Tensor> {
    let mdev = device;
    let k = *x.dims().last().unwrap();
    let n = *out_dims.last().unwrap();
    let m = x.elem_count() / k;
    let total: usize = out_dims.iter().product();

    if !matches!(indices.dtype(), DType::U32 | DType::I32) {
        crate::bail!("gather_qmm_rhs_nax: indices must be U32/I32");
    }

    let e = w.elem_count() / w.dims()[w.rank() - 1] / w.dims()[w.rank() - 2];
    let bm = if m / e < 64 { 32usize } else { 64usize };
    let (bn, bk, wm, wn) = (64usize, 64usize, 2usize, 2usize);
    let align_m = m % bm == 0;
    let align_n = n % bn == 0;
    let align_k = k % bk == 0;

    let ty = type_to_name(x.dtype())?;
    let kname = format!(
        "affine_gather_qmm_rhs_nax_nt_{ty}_gs_{group_size}_b_{bits}_bm_{bm}_bn_{bn}_bk_{bk}_wm_{wm}_wn_{wn}"
    );
    let pipeline = compile_nax_jit(
        device,
        &gather_qmm_rhs_nax_source(&kname, bm, group_size, bits),
        &kname,
        &[(200, align_m), (201, align_n), (202, align_k)],
    )?;
    let _ = metallib;

    let obuf = mdev.buffer((total) as usize * (x.dtype()).size_of(), "gather_qmm_rhs_nax_out")?;
    let out = Array::from_parts(mdev, obuf.clone(), &out_dims, x.dtype());

    let guard = mdev.commands.encoder()?;
    let enc = guard.encoder();
    enc.set_pipeline(&pipeline);
    let bind = |index: usize, t: &Tensor| -> Result<()> {
        let (ms, layout) = t.buffer_and_layout();
        enc.set_input(index, Some(ms), layout.offset * t.dtype().size_of());
        Ok(())
    };

    bind(0, x)?;
    bind(1, w)?;
    bind(2, scales)?;
    bind(3, biases)?;
    bind(4, indices)?;
    enc.set_output(5, Some(&obuf), 0);
    for (i, v) in [m as i32, n as i32, k as i32].into_iter().enumerate() {
        enc.set_bytes(6 + i, &v);
    }

    enc.dispatch_groups_size(
        MTLSize { width: n.div_ceil(bn), height: m.div_ceil(bm), depth: 1 },
        MTLSize { width: 32, height: wn, depth: wm },
    );
    Ok(out)
}

/// MLX `qmv_wide` (`quantized.cpp:544`): the affine qmv that reuses each weight
/// group across up to 5 input vectors. MLX takes this for `2 <= M <
/// vector_limit` (`use_qmv_wide` is true for affine on gen>=15).
pub fn qmv_wide(
    device: &Device,
    x: &Tensor,
    w: &Tensor,
    scales: &Tensor,
    biases: &Tensor,
    group_size: i32,
    bits: i32,
) -> Result<Tensor> {
    let mdev = device;
    let (m, k) = x.dims2();
    let (n, _) = w.dims2();
    let n_tiles = m.div_ceil(5);
    let vecs_per_tg = m.div_ceil(n_tiles);
    let k_lanes = 8usize;
    let rows_per_tg = (32 / k_lanes) * 2;
    let type_str = type_string(x.dtype())?;
    let kname = format!(
        "affine_qmv_wide_{type_str}_gs_{group_size}_b_{bits}_nv_{vecs_per_tg}_kl_{k_lanes}_batch_0"
    );
    let template_def = builtin_template_def(
        &kname,
        "affine_qmv_wide",
        &[
            type_str.to_string(),
            group_size.to_string(),
            bits.to_string(),
            vecs_per_tg.to_string(),
            k_lanes.to_string(),
            "0".to_string(),
        ],
    );
    let source = format!(
        "{MLX_UTILS_PREAMBLE}{MLX_GEMM_PREAMBLE}{MLX_QUANTIZED_UTILS_PREAMBLE}{MLX_QUANTIZED_PREAMBLE}{template_def}"
    );
    let pipeline = compile_builtin(device, &source, &kname)?;

    let count = m * n;
    let obuf = mdev.buffer((count) as usize * (x.dtype()).size_of(), "qmv_wide_out")?;
    let out = Array::from_parts(mdev, obuf.clone(), &vec![m, n], x.dtype());
    let guard = mdev.commands.encoder()?;
    let enc = guard.encoder();
    enc.set_pipeline(&pipeline);
    let bind = |index: usize, t: &Tensor| -> Result<()> {
        let (ms, layout) = t.buffer_and_layout();
        enc.set_input(
            index,
            Some(ms),
            layout.offset * t.dtype().size_of(),
        );
        Ok(())
    };
    bind(0, w)?;
    bind(1, scales)?;
    bind(2, biases)?;
    bind(3, x)?;
    enc.set_output(4, Some(&obuf), 0);
    for (i, v) in [k as i32, n as i32, m as i32].into_iter().enumerate() {
        enc.set_bytes(5 + i, &v);
    }
    enc.dispatch_groups_size(
        MTLSize {
            width: m.div_ceil(vecs_per_tg),
            height: n.div_ceil(rows_per_tg),
            depth: 1,
        },
        MTLSize { width: 32, height: 2, depth: 1 },
    );
    Ok(out)
}

/// MLX `get_qmv_batch_limit` (`quantized.cpp:85`): the M threshold below which
/// quantized matmul takes the qmv path instead of a GEMM.
pub fn qmv_batch_limit(k: usize, n: usize) -> usize {
    if k <= 2048 && n <= 2048 {
        33
    } else if k <= 4096 && n <= 4096 {
        25
    } else {
        13
    }
}

/// MLX `fast::scaled_dot_product_attention` 2-pass vector path
/// (`sdpa_vector_2pass`, `scaled_dot_product_attention.cpp:454`).
pub fn sdpa_vector_2pass(
    device: &Device,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    scale: f32,
    do_causal: bool,
) -> Result<Tensor> {
    let mdev = device;
    let qt = type_string(q.dtype())?;
    let d = q.dims()[3];
    let vd = v.dims()[3];
    let kname1 = format!("sdpa_vector_2pass_1_{qt}_{d}_{vd}");
    let kname2 = format!("sdpa_vector_2pass_2_{qt}_{vd}");
    let mut defs = builtin_template_def(
        &kname1,
        "sdpa_vector_2pass_1",
        &[qt.to_string(), d.to_string(), vd.to_string()],
    );
    defs.push_str(&builtin_template_def(
        &kname2,
        "sdpa_vector_2pass_2",
        &[qt.to_string(), vd.to_string()],
    ));
    let source = format!("{MLX_UTILS_PREAMBLE}{MLX_SDPA_VECTOR_PREAMBLE}{defs}");

    let gqa_factor = q.dims()[1] / k.dims()[1];
    let n_simds = gqa_factor * q.dims()[2];
    let n = k.dims()[2];
    let devc = mdev
        .architecture_name()
        .chars()
        .last()
        .unwrap_or('s');
    let blocks: usize = if devc == 's' {
        let mut bl = 64;
        if n > 1024 && n_simds > 4 {
            if n <= 8192 {
                bl = 128;
            } else if n <= 32768 {
                bl = 256;
            } else if n <= 65536 {
                bl = 512;
            } else {
                bl = 1024;
            }
        }
        bl
    } else if devc == 'd' {
        let mut bl = 128;
        if n_simds <= 2 && n > 8192 {
            bl = 256;
        } else if n_simds >= 6 {
            if (16384..65536).contains(&n) {
                bl = 512;
            } else if n >= 65536 {
                bl = 1024;
            }
        }
        bl
    } else if n_simds >= 4 {
        64
    } else {
        32
    };

    let pipeline1 = compile_builtin_typed_consts(
        device,
        &source,
        &kname1,
        &[
            (20, ConstVal::Bool(false)),
            (21, ConstVal::Bool(false)),
            (22, ConstVal::Bool(do_causal)),
            (23, ConstVal::Bool(false)),
            (24, ConstVal::Bool(false)),
            (25, ConstVal::Bool(false)),
            (26, ConstVal::Int(blocks as i32)),
        ],
    )?;
    let pipeline2 = compile_builtin(device, &source, &kname2)?;

    let qd = q.dims().to_vec();
    let (b, h, ql) = (qd[0], qd[1], qd[2]);
    let out_dims = vec![b, h, ql, vd];
    let out_count: usize = out_dims.iter().product();
    let obuf = mdev.buffer((out_count) as usize * (q.dtype()).size_of(), "sdpa_out")?;
    let out = Array::from_parts(mdev, obuf.clone(), &out_dims.clone(), q.dtype());

    // intermediates: [b,h,ql,blocks,vd] (q dtype) and [b,h,ql,blocks] (f32)
    let interm_dims = vec![b, h, ql, blocks, vd];
    let interm_count: usize = interm_dims.iter().product();
    let ibuf = mdev.buffer((interm_count) as usize * (q.dtype()).size_of(), "sdpa_interm")?;
    let red_dims = vec![b, h, ql, blocks];
    let red_count: usize = red_dims.iter().product();
    let sbuf = mdev.buffer((red_count) as usize * (DType::F32).size_of(), "sdpa_sums")?;
    let mbuf = mdev.buffer((red_count) as usize * (DType::F32).size_of(), "sdpa_maxs")?;

    let k_head_stride: usize = if k.dims()[1] == 1 { k.stride()[0] } else { k.stride()[1] };
    let k_seq_stride: usize = k.stride()[2];
    let v_head_stride: usize = if v.dims()[1] == 1 { v.stride()[0] } else { v.stride()[1] };
    let v_seq_stride: usize = v.stride()[2];
    let n_i = n as i32;

    let guard = mdev.commands.encoder()?;
    let enc = guard.encoder();

    // pass 1
    enc.set_pipeline(&pipeline1);
    let bind = |index: usize, t: &Tensor| -> Result<()> {
        let (ms, layout) = t.buffer_and_layout();
        enc.set_input(index, Some(ms), layout.offset * t.dtype().size_of());
        Ok(())
    };
    bind(0, q)?;
    bind(1, k)?;
    bind(2, v)?;
    enc.set_output(3, Some(&ibuf), 0);
    enc.set_output(4, Some(&sbuf), 0);
    enc.set_output(5, Some(&mbuf), 0);
    enc.set_bytes(7, &n_i);
    enc.set_bytes(8, &k_head_stride);
    enc.set_bytes(9, &k_seq_stride);
    enc.set_bytes(10, &v_head_stride);
    enc.set_bytes(11, &v_seq_stride);
    enc.set_bytes(12, &scale);
    enc.dispatch_groups_size(
        MTLSize {
            width: k.dims()[1],
            height: b,
            depth: blocks,
        },
        MTLSize {
            width: 32,
            height: gqa_factor,
            depth: ql,
        },
    );

    // pass 2
    enc.set_pipeline(&pipeline2);
    enc.set_input(0, Some(&ibuf), 0);
    enc.set_input(1, Some(&sbuf), 0);
    enc.set_input(2, Some(&mbuf), 0);
    enc.set_output(3, Some(&obuf), 0);
    let blocks_i = blocks as i32;
    enc.set_bytes(4, &blocks_i);
    enc.dispatch_groups_size(
        MTLSize {
            width: b * h,
            height: ql,
            depth: 1,
        },
        MTLSize {
            width: 1024,
            height: 1,
            depth: 1,
        },
    );
    let _ = interm_dims;
    Ok(out)
}

/// MLX `fast::rope` (`RoPE::eval_gpu`, `rope.cpp:12`). `offset` is an int array.
#[allow(clippy::too_many_arguments)]
pub fn rope(
    device: &Device,
    x: &Tensor,
    dims: i32,
    traditional: bool,
    base: f32,
    scale: f32,
    offset: &Tensor,
    forward: bool,
) -> Result<Tensor> {
    let mdev = device;
    let (_s, layout) = x.buffer_and_layout();
    let rank = x.rank();
    if rank < 2 || !layout.is_contiguous() {
        crate::bail!("rope: non-contiguous or rank < 2 input");
    }
    let dims_v = x.dims().to_vec();
    let b = dims_v[0];
    let t = dims_v[rank - 2];
    let d = dims_v[rank - 1];
    let n: usize = dims_v[1..rank.saturating_sub(2)].iter().product();
    let mat_size = (t * d) as i64;
    let single = t == 1 && offset.elem_count() == 1;
    let in_dt = x.dtype();
    let in_t = type_string(in_dt)?;
    let ty = type_to_name(in_dt)?;

    let kernel_name = if single {
        format!("rope_single_{ty}")
    } else {
        format!("rope_{ty}")
    };
    let mut defs = builtin_template_def(
        &format!("rope_{ty}"),
        "rope",
        &[in_t.to_string(), "int32_t".to_string()],
    );
    defs.push_str(&builtin_template_def(
        &format!("rope_single_{ty}"),
        "rope_single",
        &[in_t.to_string()],
    ));
    let source = format!("{MLX_UTILS_PREAMBLE}{MLX_ROPE_SOURCE}{defs}");
    let pipeline = compile_builtin_bool_consts(
        device,
        &source,
        &kernel_name,
        &[(1, forward), (2, traditional), (3, false)],
    )?;

    let count = x.elem_count();
    let obuf = mdev.buffer((count) as usize * (in_dt).size_of(), "rope_out")?;
    let out = Array::from_parts(mdev, obuf.clone(), &dims_v.clone(), in_dt);

    let in_strides = layout.stride().to_vec();
    let strides: Vec<i64> = vec![
        mat_size,
        in_strides[rank - 2] as i64,
        in_strides[rank - 1] as i64,
    ];
    // out is contiguous
    let out_strides_raw = {
        let mut s = vec![1i64; rank];
        let mut acc = 1i64;
        for i in (0..rank).rev() {
            s[i] = acc;
            acc *= dims_v[i] as i64;
        }
        s
    };
    let out_strides: Vec<i64> = vec![
        mat_size,
        out_strides_raw[rank - 2],
        out_strides_raw[rank - 1],
    ];

    let guard = mdev.commands.encoder()?;
    let enc = guard.encoder();
    enc.set_pipeline(&pipeline);
    {
        let (ms, l) = x.buffer_and_layout();
        enc.set_input(0, Some(ms), l.start_offset());
    }
    enc.set_output(1, Some(&obuf), 0);
    {
        let (ms, l) = offset.buffer_and_layout();
        enc.set_input(2, Some(ms), l.start_offset());
    }
    enc.set_bytes(3, &scale);

    let (d0, d1, d2);
    if single {
        enc.set_bytes_directly(
            4,
            std::mem::size_of_val(out_strides.as_slice()),
            out_strides.as_ptr().cast(),
        );
        d0 = dims as usize / 2;
        d1 = b * n;
        d2 = 1;
    } else {
        enc.set_bytes_directly(
            4,
            std::mem::size_of_val(strides.as_slice()),
            strides.as_ptr().cast(),
        );
        enc.set_bytes_directly(
            5,
            std::mem::size_of_val(out_strides.as_slice()),
            out_strides.as_ptr().cast(),
        );
        let offset_stride: i64 = if offset.rank() > 0 {
            offset.stride()[0] as i64
        } else {
            0
        };
        enc.set_bytes(6, &offset_stride);
        let nn = n as i32;
        enc.set_bytes(7, &nn);
        d0 = dims as usize / 2;
        d1 = t;
        d2 = b * n.div_ceil(4);
    }
    let log2base = base.log2();
    enc.set_bytes(10, &log2base);

    let group = get_block_dims(d0, d1, d2, 10);
    enc.dispatch_threads_size(
        MTLSize {
            width: d0,
            height: d1,
            depth: d2,
        },
        MTLSize {
            width: group.0,
            height: group.1,
            depth: group.2,
        },
    );
    Ok(out)
}

/// MLX `ops::put_along_axis` / `ScatterAxis::eval_gpu` (`indexing.cpp:522`).
/// `out = copy(src)` then scatter `updates` at `indices` along `axis`.
pub fn put_along_axis(
    device: &Device,
    src: &Tensor,
    indices: &Tensor,
    updates: &Tensor,
    axis: i32,
) -> Result<Tensor> {
    let mdev = device;
    let out = src.copied()?;
    let dims = indices.dims().to_vec();
    let rank = dims.len();
    let ax = if axis < 0 {
        (rank as i32 + axis) as usize
    } else {
        axis as usize
    };
    let out_t = type_string(src.dtype())?;
    let idx_t = type_string(indices.dtype())?;
    let out_ty = type_to_name(src.dtype())?;
    let idx_ty = type_to_name(indices.dtype())?;
    let lib_name = format!("scatter_axis{out_ty}{idx_ty}_none_int");
    let kernel_name = format!("{lib_name}cc");
    let a = |s: &str| s.to_string();
    let mut defs = String::new();
    for (uc, ic) in [(true, true), (true, false), (false, true), (false, false)] {
        defs.push_str(&builtin_template_def(
            &format!(
                "{lib_name}{}{}",
                if uc { "c" } else { "nc" },
                if ic { "c" } else { "nc" }
            ),
            "scatter_axis",
            &[
                a(out_t),
                a(idx_t),
                a("int"),
                a("None"),
                a(if uc { "true" } else { "false" }),
                a(if ic { "true" } else { "false" }),
            ],
        ));
    }
    let source = format!(
        "{MLX_UTILS_PREAMBLE}{MLX_REDUCE_UTILS_PREAMBLE}{MLX_SCATTER_AXIS_PREAMBLE}{defs}"
    );
    let pipeline = compile_builtin(device, &source, &kernel_name)?;

    let mut shape: Vec<i32> = Vec::new();
    let mut upd_strides: Vec<i64> = Vec::new();
    let mut idx_strides: Vec<i64> = Vec::new();
    let (_s, idx_layout) = indices.buffer_and_layout();
    let (_s, upd_layout) = updates.buffer_and_layout();
    let mut out_axis_size = 0i32;
    let mut upd_ax_stride = 0usize;
    let mut idx_ax_stride = 0usize;
    for i in 0..rank {
        if i == ax {
            out_axis_size = src.dims()[i] as i32;
            upd_ax_stride = upd_layout.stride()[i];
            idx_ax_stride = idx_layout.stride()[i];
            continue;
        }
        shape.push(dims[i] as i32);
        upd_strides.push(upd_layout.stride()[i] as i64);
        idx_strides.push(idx_layout.stride()[i] as i64);
    }
    let ndim = rank - 1;

    let guard = mdev.commands.encoder()?;
    let enc = guard.encoder();
    enc.set_pipeline(&pipeline);
    let bind = |index: usize, t: &Tensor| -> Result<()> {
        let (ms, layout) = t.buffer_and_layout();
        enc.set_input(index, Some(ms), layout.offset * t.dtype().size_of());
        Ok(())
    };
    bind(0, updates)?;
    bind(1, indices)?;
    {
        let (ms, layout) = out.buffer_and_layout();
        enc.set_output(2, Some(ms), layout.offset * out.dtype().size_of());
    }
    if ndim == 0 {
        let z: i32 = 0;
        let zl: i64 = 0;
        enc.set_bytes(3, &z);
        enc.set_bytes(4, &zl);
        enc.set_bytes(5, &zl);
    } else {
        enc.set_bytes_directly(3, std::mem::size_of_val(shape.as_slice()), shape.as_ptr().cast());
        enc.set_bytes_directly(
            4,
            std::mem::size_of_val(upd_strides.as_slice()),
            upd_strides.as_ptr().cast(),
        );
        enc.set_bytes_directly(
            5,
            std::mem::size_of_val(idx_strides.as_slice()),
            idx_strides.as_ptr().cast(),
        );
    }
    enc.set_bytes(6, &ndim);
    enc.set_bytes(7, &(ax as i32));
    enc.set_bytes(8, &out_axis_size);
    enc.set_bytes(9, &upd_ax_stride);
    enc.set_bytes(10, &idx_ax_stride);

    let size_pre: usize = dims[..ax].iter().product();
    let size_post: usize = dims[ax + 1..].iter().product();
    let idx_ax_size = dims[ax];
    let group = get_block_dims(size_post, idx_ax_size, size_pre, 10);
    enc.dispatch_threads_size(
        MTLSize {
            width: size_post,
            height: idx_ax_size,
            depth: size_pre,
        },
        MTLSize {
            width: group.0,
            height: group.1,
            depth: group.2,
        },
    );
    Ok(out)
}

/// Reduction over an arbitrary axis, implemented as a transpose to the last
/// axis + [`reduce_last_axis`] + transpose back (all bit-exact steps).
pub fn reduce_axis(
    device: &Device,
    x: &Tensor,
    axis: i32,
    op_name: &str,
) -> Result<Tensor> {
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

/// MLX `ops::softmax_axis` over the last axis, `Softmax::eval_gpu` block path
/// (`softmax.cpp:16`). `precise` forces an f32 accumulator.
pub fn softmax_last_axis(device: &Device, x: &Tensor, precise: bool) -> Result<Tensor> {
    let mdev = device;
    let (_s, layout) = x.buffer_and_layout();
    if x.dims().is_empty() || !layout.is_contiguous() {
        crate::bail!("softmax: non-contiguous or scalar input");
    }
    let dims = x.dims().to_vec();
    let dt = x.dtype();
    let axis_size = *dims.last().unwrap();
    let n_rows = x.elem_count() / axis_size.max(1);
    if axis_size > 4096 {
        crate::bail!("softmax: axis_size > 4096 (looped path) not implemented");
    }
    let in_t = type_string(dt)?;
    let acc_t = if precise { "float" } else { in_t };
    let ty = type_to_name(dt)?;

    let mut kernel_name = String::from("block_softmax_");
    if dt != DType::F32 && precise {
        kernel_name.push_str("precise_");
    }
    kernel_name.push_str(ty);
    let lib_name = kernel_name
        .split_once('_')
        .map(|(_, r)| r)
        .unwrap_or(&kernel_name)
        .to_string();

    let mut defs = builtin_template_def(
        &format!("block_{lib_name}"),
        "softmax_single_row",
        &[in_t.to_string(), acc_t.to_string()],
    );
    defs.push_str(&builtin_template_def(
        &format!("looped_{lib_name}"),
        "softmax_looped",
        &[in_t.to_string(), acc_t.to_string()],
    ));
    let source = format!("{MLX_UTILS_PREAMBLE}{MLX_SOFTMAX_PREAMBLE}{defs}");
    let pipeline = compile_builtin(device, &source, &kernel_name)?;

    let count = x.elem_count();
    let obuf = mdev.buffer((count) as usize * (dt).size_of(), "softmax_out")?;
    let out = Array::from_parts(mdev, obuf.clone(), &dims.clone(), dt);

    let guard = mdev.commands.encoder()?;
    let enc = guard.encoder();
    enc.set_pipeline(&pipeline);
    {
        let (ms, layout) = x.buffer_and_layout();
        enc.set_input(0, Some(ms), layout.offset * x.dtype().size_of());
    }
    enc.set_output(1, Some(&obuf), 0);
    let asize = axis_size as i32;
    enc.set_bytes(2, &asize);

    let simd = 32usize;
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
    Ok(out)
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
        enc.set_bytes_directly(2, std::mem::size_of_val(shape.as_slice()), shape.as_ptr().cast());
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
    let mut tgs = axis_size.div_ceil(4).min(pipeline.max_total_threads_per_threadgroup());
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
    let source = format!(
        "{MLX_UTILS_PREAMBLE}{MLX_REDUCE_UTILS_PREAMBLE}{MLX_REDUCE_PREAMBLE}{def}"
    );
    let pipeline = compile_builtin(device, &source, &kname)?;

    let out_count = rest;
    let obuf = mdev.buffer((out_count) as usize * (inter.dtype()).size_of(), "colred_out")?;
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
    enc.set_bytes_directly(5, std::mem::size_of_val(&stride_z), stride_z.as_ptr().cast());
    let ndim: i32 = 0;
    enc.set_bytes(6, &ndim);
    let reduce_shape: [i32; 1] = [r as i32];
    let reduce_strides: [i64; 1] = [rstride];
    enc.set_bytes_directly(7, std::mem::size_of_val(&reduce_shape), reduce_shape.as_ptr().cast());
    enc.set_bytes_directly(8, std::mem::size_of_val(&reduce_strides), reduce_strides.as_ptr().cast());
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
        MTLSize { width: blocks.div_ceil(tg_x), height: 1, depth: 1 },
        MTLSize { width: tg_x, height: tg_y, depth: 1 },
    );
    Ok(out)
}

/// MLX affine `qmm_splitk` (`quantized.cpp:1120`) — the prefill dense path
/// (transpose=true, B=1, M >= vector_limit). Split-K over K with a sum reduce.
pub fn affine_qmm_splitk(
    device: &Device,
    x: &Tensor,
    w: &Tensor,
    scales: &Tensor,
    biases: &Tensor,
    group_size: i32,
    bits: i32,
) -> Result<Tensor> {
    let mdev = device;
    let (m, k) = x.dims2();
    let (n, _) = w.dims2();
    let (bm, bn) = (32usize, 32usize);
    let n_tiles = n.div_ceil(bn);
    let m_tiles = m.div_ceil(bm);
    let current_tgs = n_tiles * m_tiles;
    let k_align = (group_size.max(32)) as usize;
    let mut split_k = (512 / current_tgs).max(1);
    split_k = split_k.min(k / k_align);
    while split_k > 1 && k % (split_k * k_align) != 0 {
        split_k -= 1;
    }
    if split_k <= 1 {
        crate::bail!("affine_qmm_splitk: split_k <= 1 (use qmm path)");
    }
    let k_partition_size = (k / split_k) as i32;

    let type_str = type_string(x.dtype())?;
    let aligned = n % 32 == 0;
    let kname = format!(
        "affine_qmm_t_splitk_{type_str}_gs_{group_size}_b_{bits}_{}",
        if aligned { "_alN_true" } else { "_alN_false" }
    );
    let def = builtin_template_def(
        &kname,
        "affine_qmm_t_splitk",
        &[
            type_str.to_string(),
            group_size.to_string(),
            bits.to_string(),
            (if aligned { "true" } else { "false" }).to_string(),
        ],
    );
    let source = format!(
        "{MLX_UTILS_PREAMBLE}{MLX_GEMM_PREAMBLE}{MLX_QUANTIZED_UTILS_PREAMBLE}{MLX_QUANTIZED_PREAMBLE}{def}"
    );
    let pipeline = compile_builtin(device, &source, &kname)?;

    // intermediate [split_k, m, n]
    let icount = split_k * m * n;
    let ibuf = mdev.buffer((icount) as usize * (x.dtype()).size_of(), "qmm_splitk_inter")?;
    let inter = Array::from_parts(mdev, ibuf.clone(), &vec![split_k, m, n], x.dtype());

    let guard = mdev.commands.encoder()?;
    let enc = guard.encoder();
    enc.set_pipeline(&pipeline);
    let bind = |index: usize, t: &Tensor| -> Result<()> {
        let (ms, layout) = t.buffer_and_layout();
        enc.set_input(index, Some(ms), layout.offset * t.dtype().size_of());
        Ok(())
    };
    bind(0, w)?;
    bind(1, scales)?;
    bind(2, biases)?;
    bind(3, x)?;
    enc.set_output(4, Some(&ibuf), 0);
    let kk = k as i32;
    enc.set_bytes(5, &kk);
    let nn = n as i32;
    enc.set_bytes(6, &nn);
    let mm = m as i32;
    enc.set_bytes(7, &mm);
    enc.set_bytes(8, &k_partition_size);
    let skps = (m * n) as i32;
    enc.set_bytes(9, &skps);
    enc.dispatch_groups_size(
        MTLSize { width: n_tiles, height: m_tiles, depth: split_k },
        MTLSize { width: 32, height: 2, depth: 2 },
    );
    drop(guard);

    // Sum the split-K partials (axis 0) with MLX's strided reduce.
    reduce_axis0_sum(device, &inter)
}

/// Source for one `affine_qmm_t_nax` instantiation.
pub fn qmm_nax_source(
    kname: &str,
    group_size: i32,
    bits: i32,
    aligned_n: bool,
    batched: bool,
    bm: usize,
    bk: usize,
    bn: usize,
    wm: usize,
    wn: usize,
) -> String {
    format!(
        "{NAX_QUANT_HEADER}\nusing namespace metal;\n\
         instantiate_kernel(\"{kname}\", affine_qmm_t_nax, bfloat16_t, {group_size}, {bits}, {}, {}, {bm}, {bk}, {bn}, {wm}, {wn})\n",
        if aligned_n { "true" } else { "false" },
        if batched { "true" } else { "false" },
    )
}

/// MLX `qmm_nax` (`quantized.cpp:814`): the NAX affine quantized matmul used
/// for `M > 1` when NAX is available, `transpose`, and `K % 64 == 0`. It is the
/// non-split counterpart of `affine_qmm_splitk`. `x`/`w` are 2-D and `B == 1`
/// (the engine always flattens to `[M, K]`).
pub fn qmm_nax(
    device: &Device,
    x: &Tensor,
    w: &Tensor,
    scales: &Tensor,
    biases: &Tensor,
    group_size: i32,
    bits: i32,
) -> Result<Tensor> {
    let mdev = device;
    let (m, k) = x.dims2();
    let (n, _) = w.dims2();
    let (wm, wn) = (2usize, 2usize);
    let bm = if m <= 32 { 32usize } else { 64usize };
    let (bn, bk) = (64usize, 64usize);
    let aligned = n % 64 == 0;
    let batched = false;
    let ty = type_to_name(x.dtype())?;
    let kname = format!(
        "affine_qmm_t_nax_{ty}_gs_{group_size}_b_{bits}_bm{bm}_bn{bn}_bk{bk}_wm{wm}_wn{wn}{}{}",
        if aligned { "_alN_true" } else { "_alN_false" },
        if batched { "_batch_1" } else { "_batch_0" },
    );
    let pipeline = compile_nax_jit(
        device,
        &qmm_nax_source(&kname, group_size, bits, aligned, batched, bm, bk, bn, wm, wn),
        &kname,
        &[],
    )?;

    let count = m * n;
    let obuf = mdev.buffer((count) as usize * (x.dtype()).size_of(), "qmm_nax_out")?;
    let out = Array::from_parts(mdev, obuf.clone(), &vec![m, n], x.dtype());

    let guard = mdev.commands.encoder()?;
    let enc = guard.encoder();
    enc.set_pipeline(&pipeline);
    let bind = |index: usize, t: &Tensor| -> Result<()> {
        let (ms, layout) = t.buffer_and_layout();
        enc.set_input(
            index,
            Some(ms),
            layout.offset * t.dtype().size_of(),
        );
        Ok(())
    };
    bind(0, w)?;
    bind(1, scales)?;
    bind(2, biases)?;
    bind(3, x)?;
    enc.set_output(4, Some(&obuf), 0);
    for (i, v) in [k as i32, n as i32, m as i32].into_iter().enumerate() {
        enc.set_bytes(5 + i, &v);
    }

    enc.dispatch_groups_size(
        MTLSize { width: n.div_ceil(bn), height: m.div_ceil(bm), depth: 1 },
        MTLSize { width: 32, height: wn, depth: wm },
    );
    Ok(out)
}

/// Load a compiled `.metallib`, fetch `name`, and specialise bool function
/// constants before building the pipeline. Unlike the JIT path this avoids
/// recompiling MLX's NAX kernels (whose JIT preamble hangs the Metal compiler).
/// Source for one `attention_nax` / `attention_nax_dsplit` instantiation.
pub fn sdpa_full_nax_source(
    base_name: &str,
    split_d: bool,
    bq: usize,
    bk: usize,
    bd: usize,
    wm: usize,
    wn: usize,
    q_ty: &str,
    mask_ty: &str,
) -> String {
    format!(
        "{NAX_ATTN_HEADER}\nusing namespace metal;\n\
         instantiate_kernel(\"{base_name}\", {}, {q_ty}, {bq}, {bk}, {bd}, {wm}, {wn}, {mask_ty})\n",
        if split_d { "attention_nax_dsplit" } else { "attention_nax" }
    )
}

/// MLX `sdpa_full_self_attention_nax` (`scaled_dot_product_attention.cpp:18`)
/// loaded from an AOT metallib (built from the vendored MLX `.metal` source).
pub fn sdpa_full_nax(
    device: &Device,
    metallib: Option<&str>,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    scale: f32,
    do_causal: bool,
) -> Result<Tensor> {
    let mdev = device;
    let bd = q.dims()[3];
    let bq = 64usize;
    let bk = 32usize;
    let split_d = bd == 256;
    let wm = 4usize;
    let wn = if split_d { 2 } else { 1 };
    let (b, h, ql) = (q.dims()[0], q.dims()[1], q.dims()[2]);
    let k_l = k.dims()[2];
    let gqa = q.dims()[1] / k.dims()[1];
    let qt = type_to_name(q.dtype())?;

    let prefix = if split_d {
        "steel_attention_dsplit_"
    } else {
        "steel_attention_"
    };
    let base_name = format!("{prefix}{qt}_bq{bq}_bk{bk}_bd{bd}_wm{wm}_wn{wn}_mask{qt}");
    let align_q = ql % bq == 0;
    let align_k = k_l % bk == 0;
    let pipeline = compile_nax_jit(
        device,
        &sdpa_full_nax_source(&base_name, split_d, bq, bk, bd, wm, wn, "bfloat", "bfloat"),
        &base_name,
        &[
            (200, align_q),
            (201, align_k),
            (300, false),
            (301, do_causal),
            (302, false),
        ],
    )?;
    let _ = metallib;

    let store_dims = vec![b, h, ql, bd];
    let o_str = [(h * ql * bd) as i64, (ql * bd) as i64, bd as i64];
    let count: usize = store_dims.iter().product();
    let obuf = mdev.buffer((count) as usize * (q.dtype()).size_of(), "sdpa_full_out")?;
    let store = Array::from_parts(mdev, obuf.clone(), &store_dims, q.dtype());

    let nq = ql.div_ceil(bq);
    let nk = k_l.div_ceil(bk);
    let nq_aligned = ql / bq;
    let nk_aligned = k_l / bk;
    let ql_rem = ql - nq_aligned * bq;
    let kl_rem = k_l - nk_aligned * bk;
    let ql_off = k_l as i64 - ql as i64;

    let mut p: Vec<u8> = Vec::with_capacity(152);
    for val in [b as i32, h as i32, bd as i32, ql as i32, k_l as i32, gqa as i32] {
        p.extend_from_slice(&val.to_le_bytes());
    }
    p.extend_from_slice(&scale.to_le_bytes());
    for val in [
        nq as i32, nk as i32, nq_aligned as i32, nk_aligned as i32, ql_rem as i32,
        kl_rem as i32, ql_off as i32,
    ] {
        p.extend_from_slice(&val.to_le_bytes());
    }
    let q_str = [q.stride()[0] as i64, q.stride()[1] as i64, q.stride()[2] as i64];
    let k_str = [k.stride()[0] as i64, k.stride()[1] as i64, k.stride()[2] as i64];
    let v_str = [v.stride()[0] as i64, v.stride()[1] as i64, v.stride()[2] as i64];
    for s in [q_str, k_str, v_str, o_str] {
        for val in s {
            p.extend_from_slice(&val.to_le_bytes());
        }
    }

    let guard = mdev.commands.encoder()?;
    let enc = guard.encoder();
    enc.set_pipeline(&pipeline);
    let bind = |index: usize, t: &Tensor| -> Result<()> {
        let (ms, layout) = t.buffer_and_layout();
        enc.set_input(index, Some(ms), layout.offset * t.dtype().size_of());
        Ok(())
    };
    bind(0, q)?;
    bind(1, k)?;
    bind(2, v)?;
    enc.set_output(3, Some(&obuf), 0);
    enc.set_bytes_directly(4, p.len(), p.as_ptr().cast());

    enc.dispatch_groups_size(
        MTLSize { width: nq, height: h, depth: b },
        MTLSize { width: 32, height: wm, depth: wn },
    );
    store.contiguous()
}

/// MLX `fast::scaled_dot_product_attention` fused dispatch for the causal,
/// no-array-mask case: the vector kernel for short query runs (and
/// `qL * gqa <= 32`), the NAX full kernel for wider ones. Both branches are
/// bit-exact vs mlx-c; the dense op-chain fallback (`use_fallback == true`) is
/// not ported here and is reported rather than silently substituted.
pub fn sdpa(
    device: &Device,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    scale: f32,
    do_causal: bool,
    mask: Option<&Tensor>,
) -> Result<Tensor> {
    // The KV cache hands us non-contiguous views (a [0..len] slice of a larger
    // capacity buffer); our vector/full kernels' stride handling diverges from
    // MLX's there. Materialise contiguous copies -- same data, same result.
    let (q, k, v) = (q.contiguous()?, k.contiguous()?, v.contiguous()?);
    let (q, k, v) = (&q, &k, &v);
    let ql = q.dims()[2];
    let d = q.dims()[3];
    let gqa = q.dims()[1] / k.dims()[1];
    if mask.is_some() {
        // The fused kernels' array-mask branches are not ported; MLX itself
        // routes d == 192/256 array-mask attention to the dense op chain.
        if matches!(d, 192 | 256) {
            return sdpa_dense(device, q, k, v, scale, false, mask);
        }
        crate::bail!("sdpa: array mask with head_dim {d} not ported");
    }
    if ql <= 8 {
        if ql * gqa > 32 {
            if d == 256 && do_causal {
                return sdpa_full_nax(device, None, q, k, v, scale, do_causal);
            }
            return sdpa_dense(device, q, k, v, scale, do_causal, None);
        }
        // MLX eval_gpu: 2-pass on 's'/'d' with kL >= 1024 (or short KV heads
        // with kL >= 4096); causal is dropped for a single query.
        let kl = k.dims()[2];
        let devc = device
            .architecture_name()
            .chars()
            .last()
            .unwrap_or('s');
        let two_pass = ((devc == 'd' || devc == 's') && kl >= 1024)
            || (k.dims()[1] < q.dims()[1] && kl >= 4096);
        let dc = do_causal && ql > 1;
        if two_pass {
            sdpa_vector_2pass(device, q, k, v, scale, dc)
        } else {
            sdpa_vector(device, q, k, v, scale, dc)
        }
    } else if ql >= 1024 && d == 256 && do_causal {
        sdpa_full_nax(device, None, q, k, v, scale, do_causal)
    } else if matches!(d, 192 | 256) {
        sdpa_dense(device, q, k, v, scale, do_causal, None)
    } else if matches!(d, 64 | 96 | 128) {
        sdpa_full_nax(device, None, q, k, v, scale, do_causal)
    } else {
        crate::bail!("sdpa: dense fallback for head_dim {d} not ported")
    }
}

/// MLX's dense (unfused) SDPA op chain from `fast.cpp` (`use_fallback ==
/// true`): `softmax(q*scale @ kᵀ [+ mask]) @ v`. Used for MTP verify
/// (`qL*gqa > 32`) and for array-mask / ragged batching. The batched matmuls
/// are the MLX NAX GEMM (`matmul_nax`) applied per (batch, kv-head, repeat)
/// slice; the softmax is the ported precise softmax kernel.
pub fn sdpa_dense(
    device: &Device,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    scale: f32,
    do_causal: bool,
    mask: Option<&Tensor>,
) -> Result<Tensor> {
        let (b, h, ql, d) = (q.dims()[0], q.dims()[1], q.dims()[2], q.dims()[3]);
    let hk = k.dims()[1];
    let kl = k.dims()[2];
    let r = h / hk;
    let dt = q.dtype();

    // `multiply(array(scale, dtype), q)` — a bf16 scalar times q.
    let scal = Array::scalar_of(device, scale, dt)?;
    let qs = q.broadcast_mul(&scal)?;

    // scores = matmul(q, swapaxes(k, -1, -2)) -> [B, Hk, R, qL, kL]
    let k_t = k.transpose(2, 3)?.contiguous()?;
    let mut rows: Vec<Tensor> = Vec::with_capacity(b * hk * r);
    for bi in 0..b {
        for hi in 0..hk {
            let kt = k_t.i((bi, hi))?.contiguous()?;
            for ri in 0..r {
                let qq = qs.i((bi, hi * r + ri))?.contiguous()?;
                rows.push(matmul_nax(device, None, &qq, &kt, false, false)?);
            }
        }
    }
    let mut scores = Array::stack(&rows, 0)?.reshape(&[b, hk, r, ql, kl])?;

    let has_mask = mask.is_some() || do_causal;
    if has_mask {
        let m = match mask {
            Some(m) => m.contiguous()?,
            None => {
                let offset = kl as i64 - ql as i64;
                let mut mv = vec![0u8; ql * kl];
                for i in 0..ql {
                    for j in 0..kl {
                        mv[i * kl + j] = ((offset + i as i64) >= j as i64) as u8;
                    }
                }
                Array::from_slice_dt(device, &mv, &[ql as usize, kl as usize], Dtype::Uint8)?
            }
        };
        // bool mask -> `where(mask, scores, finfo(dtype).min)`
        let minv = Array::scalar_of(device, half::bf16::MIN.to_f32(), dt)?;
        let mb = m.broadcast_as(scores.shape())?;
        let minb = minv.broadcast_as(scores.shape())?;
        scores = mb.where_cond(&scores, &minb)?;
    }

    let probs = softmax_last_axis(device, &scores, true)?;

    let mut outs: Vec<Tensor> = Vec::with_capacity(b * hk * r);
    for bi in 0..b {
        for hi in 0..hk {
            let vv = v.i((bi, hi))?.contiguous()?;
            for ri in 0..r {
                let sc = probs.i3(bi, hi, ri)?.contiguous()?;
                outs.push(matmul_nax(device, None, &sc, &vv, false, false)?);
            }
        }
    }
    let out = Array::stack(&outs, 0)?.reshape(&[b, hk, r, ql, d])?;
    out.reshape(&[b, h, ql, d])
}

/// Load a compiled `.metallib` and fetch `name`. This is how the engine's
/// built-in quantized ops run (AOT), and the path we mirror so codegen matches.
/// Metal attributes MLX auto-declares when the source references them, in the
/// exact order of `metal_kernel.cpp`.
const METAL_ATTRIBUTES: &[(&str, &str)] = &[
    ("dispatch_quadgroups_per_threadgroup", "uint"),
    ("dispatch_simdgroups_per_threadgroup", "uint"),
    ("dispatch_threads_per_threadgroup", "uint3"),
    ("grid_origin", "uint3"),
    ("grid_size", "uint3"),
    ("quadgroup_index_in_threadgroup", "uint"),
    ("quadgroups_per_threadgroup", "uint"),
    ("simdgroup_index_in_threadgroup", "uint"),
    ("simdgroups_per_threadgroup", "uint"),
    ("thread_execution_width", "uint"),
    ("thread_index_in_quadgroup", "uint"),
    ("thread_index_in_simdgroup", "uint"),
    ("thread_index_in_threadgroup", "uint"),
    ("thread_position_in_grid", "uint3"),
    ("thread_position_in_threadgroup", "uint3"),
    ("threadgroup_position_in_grid", "uint3"),
    ("threadgroups_per_grid", "uint3"),
    ("threads_per_grid", "uint3"),
    ("threads_per_simdgroup", "uint"),
    ("threads_per_threadgroup", "uint3"),
];

/// Template argument for a Metal kernel (mirrors MLX's `mx.fast.metal_kernel`).
#[derive(Clone)]
pub enum TemplateArg {
    Dtype(&'static str, DType),
    Int(&'static str, i32),
    Bool(&'static str, bool),
}

impl TemplateArg {
    fn name(&self) -> &'static str {
        match self {
            TemplateArg::Dtype(n, _) | TemplateArg::Int(n, _) | TemplateArg::Bool(n, _) => n,
        }
    }
}

/// Output argument: shape + dtype.
#[derive(Clone)]
pub struct OutputArg {
    pub shape: Vec<i32>,
    pub dtype: DType,
}

/// MLX `get_type_string` (`backend/common/compiled.cpp:31`).
fn type_string(dt: DType) -> Result<&'static str> {
    Ok(match dt {
        DType::F32 => "float",
        DType::F16 => "float16_t",
        DType::BF16 => "bfloat16_t",
        DType::F64 => "double",
        DType::U8 => "uint8_t",
        DType::U32 => "uint32_t",
        DType::I32 => "int32_t",
        DType::I64 => "int64_t",
        DType::I16 => "int16_t",
        other => crate::bail!("unsupported compilation type {other:?}"),
    })
}

/// The `template <...>` value list MLX appends to the kernel name and uses to
/// explicitly instantiate the kernel (`write_template`).
fn write_template(args: &[TemplateArg]) -> Result<String> {
    let mut s = String::from("<");
    for (i, arg) in args.iter().enumerate() {
        if i > 0 {
            s.push_str(", ");
        }
        match arg {
            TemplateArg::Int(_, v) => s.push_str(&v.to_string()),
            TemplateArg::Bool(_, v) => s.push_str(if *v { "1" } else { "0" }),
            TemplateArg::Dtype(_, dt) => s.push_str(type_string(*dt)?),
        }
    }
    s.push('>');
    Ok(s)
}

/// `make_template_hash`: `<>` -> `_`, `", "` -> `_`, then drop the last char.
fn make_template_hash(template_def: &str) -> String {
    let bytes = template_def.as_bytes();
    let mut s = String::with_capacity(template_def.len());
    let mut i = 0;
    while i < bytes.len() {
        let c = bytes[i] as char;
        if c == '<' || c == '>' {
            s.push('_');
        } else if c == ',' && i + 1 < bytes.len() && bytes[i + 1] == b' ' {
            s.push('_');
            i += 1;
        } else {
            s.push(c);
        }
        i += 1;
    }
    s.pop();
    s
}

#[allow(clippy::too_many_arguments)]
fn write_signature(
    func_name: &str,
    header: &str,
    source: &str,
    input_names: &[String],
    input_dtypes: &[DType],
    input_ndims: &[usize],
    input_sizes: &[usize],
    output_names: &[String],
    output_dtypes: &[DType],
    template_args: &[TemplateArg],
    attributes: &[String],
    shape_infos: &[(bool, bool, bool)],
    atomic_outputs: bool,
) -> Result<String> {
    let mut ks = String::with_capacity(header.len() + source.len() + 16384);
    ks.push_str(header);
    if !template_args.is_empty() {
        ks.push_str("template <");
        for (i, arg) in template_args.iter().enumerate() {
            let param_type = match arg {
                TemplateArg::Int(..) => "int",
                TemplateArg::Bool(..) => "bool",
                TemplateArg::Dtype(..) => "typename",
            };
            if i > 0 {
                ks.push_str(", ");
            }
            ks.push_str(param_type);
            ks.push(' ');
            ks.push_str(arg.name());
        }
        ks.push_str(">\n");
    }
    ks.push_str("[[kernel]] void ");
    ks.push_str(func_name);
    ks.push_str("(\n");

    let mut index = 0usize;
    for i in 0..input_names.len() {
        let name = &input_names[i];
        let dtype = type_string(input_dtypes[i])?;
        let location = if input_sizes[i] < MAX_CONSTANT_ARRAY_SIZE {
            "constant"
        } else {
            "device"
        };
        let reference = if input_ndims[i] == 0 { "&" } else { "*" };
        ks.push_str("  const ");
        ks.push_str(location);
        ks.push(' ');
        ks.push_str(dtype);
        ks.push_str(reference);
        ks.push(' ');
        ks.push_str(name);
        ks.push_str(&format!(" [[buffer({index})]],\n"));
        index += 1;
        if input_ndims[i] > 0 {
            let (shape, strides, ndim) = shape_infos[i];
            if shape {
                ks.push_str(&format!(
                    "  const constant int* {name}_shape [[buffer({index})]],\n"
                ));
                index += 1;
            }
            if strides {
                ks.push_str(&format!(
                    "  const constant int64_t* {name}_strides [[buffer({index})]],\n"
                ));
                index += 1;
            }
            if ndim {
                ks.push_str(&format!(
                    "  const constant int& {name}_ndim [[buffer({index})]],\n"
                ));
                index += 1;
            }
        }
    }
    for i in 0..output_names.len() {
        let name = &output_names[i];
        let ts = type_string(output_dtypes[i])?;
        ks.push_str("  device ");
        if atomic_outputs {
            ks.push_str("atomic<");
        }
        ks.push_str(ts);
        if atomic_outputs {
            ks.push('>');
        }
        ks.push_str("* ");
        ks.push_str(name);
        ks.push_str(&format!(" [[buffer({index})]]"));
        if index < input_names.len() + output_names.len() - 1 || !attributes.is_empty() {
            ks.push_str(",\n");
        } else {
            ks.push_str(") {\n");
        }
        index += 1;
    }
    for (i, attr) in attributes.iter().enumerate() {
        ks.push_str(attr);
        if i < attributes.len() - 1 {
            ks.push_str(",\n");
        } else {
            ks.push_str(") {\n");
        }
    }
    ks.push_str(source);
    ks.push_str("\n}\n");
    Ok(ks)
}

/// A JIT-compiled custom Metal kernel; the replacement for MLX's
/// `mlx_fast_metal_kernel`.
pub struct MetalKernel {
    name: String,
    input_names: Vec<String>,
    output_names: Vec<String>,
    source: String,
    header: String,
    atomic_outputs: bool,
    shape_infos: Vec<(bool, bool, bool)>,
    attributes: Vec<String>,
    pipelines: Mutex<HashMap<String, ComputePipeline>>,
    /// `apply` fast path: cheap u64 key -> (kernel name, pipeline). Avoids
    /// rebuilding the template string + hash per dispatch.
    apply_keys: Mutex<HashMap<u64, (String, ComputePipeline)>>,
}

unsafe impl Send for MetalKernel {}
unsafe impl Sync for MetalKernel {}

impl MetalKernel {
    pub fn new(
        name: &str,
        input_names: &[&str],
        output_names: &[&str],
        source: &str,
        header: &str,
        _ensure_row_contiguous: bool,
        atomic_outputs: bool,
    ) -> Result<Self> {
        if output_names.is_empty() {
            crate::bail!("[metal_kernel] must specify at least one output");
        }
        let shape_infos = input_names
            .iter()
            .map(|n| {
                (
                    source.contains(&format!("{n}_shape")),
                    source.contains(&format!("{n}_strides")),
                    source.contains(&format!("{n}_ndim")),
                )
            })
            .collect();
        let attributes = METAL_ATTRIBUTES
            .iter()
            .filter(|(attr, _)| source.contains(attr))
            .map(|(attr, ty)| format!("  {ty} {attr} [[{attr}]]"))
            .collect();
        Ok(Self {
            name: name.to_string(),
            input_names: input_names.iter().map(|s| s.to_string()).collect(),
            output_names: output_names.iter().map(|s| s.to_string()).collect(),
            source: source.to_string(),
            header: header.to_string(),
            atomic_outputs,
            shape_infos,
            attributes,
            pipelines: Mutex::new(HashMap::new()),
            apply_keys: Mutex::new(HashMap::new()),
        })
    }

    /// Match MLX's `device.cpp::set_compile_options` + `build_library_`
    /// exactly. On macOS >= 15 MLX sets **only** `MathModeSafe` (never
    /// `fastMathEnabled`) and additionally sets the language version
    /// (`get_metal_version`: 4.1 on macOS 27, 4.0 on 26, 3.2 on 15). Calling
    /// `setFastMathEnabled(false)` and leaving the language version at its
    /// default changed `metal::exp` codegen and made `silu_head` differ from
    /// the engine by ~6e-5 on large inputs.
     fn pipeline(&self, device: &Device, kernel_source: &str, kernel_name: &str) -> Result<ComputePipeline> {
        {
            let cache = self.pipelines.lock().unwrap();
            if let Some(p) = cache.get(kernel_name) {
                return Ok(p.clone());
            }
        }
        let mdev = device;
        let full_source = format!("{MLX_UTILS_PREAMBLE}{kernel_source}");
        let pipeline = mdev.compile_with(&full_source, kernel_name, Math::Jit)?;
        self.pipelines
            .lock()
            .unwrap()
            .insert(kernel_name.to_string(), pipeline.clone());
        Ok(pipeline)
    }

    /// Apply the kernel. `inputs` are Metal tensors; the returned tensors
    /// are allocated on the same device.
    pub fn apply(
        &self,
        device: &Device,
        inputs: &[&Tensor],
        template: &[TemplateArg],
        grid: (i32, i32, i32),
        thread_group: (i32, i32, i32),
        outputs: &[OutputArg],
        prealloc: Option<&[Tensor]>,
    ) -> Result<Vec<Tensor>> {
        if inputs.len() != self.input_names.len() {
            crate::bail!(
                "kernel {}: expected {} inputs, got {}",
                self.name,
                self.input_names.len(),
                inputs.len()
            );
        }
        let _mdev = device;

        let mut input_dtypes = Vec::with_capacity(inputs.len());
        let mut input_ndims = Vec::with_capacity(inputs.len());
        let mut input_sizes = Vec::with_capacity(inputs.len());
        for t in inputs {
            let _ = t.device();
            input_dtypes.push(t.dtype());
            input_ndims.push(t.rank());
            input_sizes.push(t.elem_count());
        }
        let output_dtypes: Vec<DType> = outputs.iter().map(|o| o.dtype).collect();

        // Cheap u64 cache key (name, template args, input dtypes + the
        // scalar/buffer flag, output dtypes). `apply` runs for every dispatch;
        // the template string + hash + `write_signature` are only needed on a
        // miss (first launch), and the name string itself only to compile.
        let mut key: u64 = 0xcbf29ce484222325;
        for b in self.name.as_bytes() {
            key = (key ^ *b as u64).wrapping_mul(0x100000001b3);
        }
        for arg in template {
            match arg {
                TemplateArg::Dtype(n, d) => {
                    for b in n.as_bytes() {
                        key = (key ^ *b as u64).wrapping_mul(0x100000001b3);
                    }
                    key = (key ^ (*d as u32 as u64)).wrapping_mul(0x100000001b3);
                }
                TemplateArg::Int(n, v) => {
                    for b in n.as_bytes() {
                        key = (key ^ *b as u64).wrapping_mul(0x100000001b3);
                    }
                    key = (key ^ (*v as u64)).wrapping_mul(0x100000001b3);
                }
                TemplateArg::Bool(n, v) => {
                    for b in n.as_bytes() {
                        key = (key ^ *b as u64).wrapping_mul(0x100000001b3);
                    }
                    key = (key ^ (*v as u64)).wrapping_mul(0x100000001b3);
                }
            }
        }
        for (i, dt) in input_dtypes.iter().enumerate() {
            key = (key ^ (*dt as u32 as u64)).wrapping_mul(0x100000001b3);
            if input_ndims[i] == 0 {
                key = (key ^ 0x11).wrapping_mul(0x100000001b3);
            } else if input_sizes[i] < MAX_CONSTANT_ARRAY_SIZE {
                key = (key ^ 0x22).wrapping_mul(0x100000001b3);
            }
        }
        for dt in output_dtypes.iter() {
            key = (key ^ (*dt as u32 as u64)).wrapping_mul(0x100000001b3);
        }

        // Pipeline fast path: `apply_keys` maps the u64 key to the kernel name
        // (which encodes dtypes/templates, as MLX builds it) + pipeline.
        {
            let cache = self.apply_keys.lock().unwrap();
            if let Some((kernel_name, p)) = cache.get(&key) {
                let (_kernel_name, p) = (kernel_name.clone(), p.clone());
                drop(cache);
                return self.apply_with_pipeline(
                    device, inputs, grid, thread_group, outputs, prealloc, p,
                    &input_ndims,
                );
            }
        }

        let mut kernel_name = format!("custom_kernel_{}", self.name);
        if !template.is_empty() {
            let template_def = write_template(template)?;
            let hash = make_template_hash(&template_def);
            kernel_name.push('_');
            kernel_name.push_str(&hash);
        }
        for (i, dt) in input_dtypes.iter().enumerate() {
            kernel_name.push('_');
            kernel_name.push_str(type_string(*dt)?);
            if input_ndims[i] == 0 {
                kernel_name.push('s');
            } else if input_sizes[i] < MAX_CONSTANT_ARRAY_SIZE {
                kernel_name.push('c');
            }
        }
        for dt in output_dtypes.iter() {
            kernel_name.push('_');
            kernel_name.push_str(type_string(*dt)?);
        }

        let mut kernel_source = write_signature(
            &kernel_name,
            &self.header,
            &self.source,
            &self.input_names,
            &input_dtypes,
            &input_ndims,
            &input_sizes,
            &self.output_names,
            &output_dtypes,
            template,
            &self.attributes,
            &self.shape_infos,
            self.atomic_outputs,
        )?;
        if !template.is_empty() {
            let template_def = write_template(template)?;
            let template_def = format!("{kernel_name}{template_def}");
            kernel_source.push_str(&format!(
                "\ntemplate [[host_name(\"{kernel_name}\")]] [[kernel]] decltype({template_def}) {template_def};\n"
            ));
        }

        let pipeline = self.pipeline(device, &kernel_source, &kernel_name)?;
        {
            let mut cache = self.apply_keys.lock().unwrap();
            cache.insert(key, (kernel_name.clone(), pipeline.clone()));
        }

        self.apply_with_pipeline(
            device, inputs, grid, thread_group, outputs, prealloc, pipeline, &input_ndims,
        )
    }

    /// The tail of `apply` once the pipeline is known: allocate outputs, bind,
    /// dispatch. Shared by the cache-hit fast path.
    fn apply_with_pipeline(
        &self,
        device: &Device,
        inputs: &[&Tensor],
        grid: (i32, i32, i32),
        thread_group: (i32, i32, i32),
        outputs: &[OutputArg],
        prealloc: Option<&[Tensor]>,
        pipeline: ComputePipeline,
        input_ndims: &[usize],
    ) -> Result<Vec<Tensor>> {
        let mdev = device;
        if crate::runtime::env_flag("LISA_NOOP_DISPATCH") {
            // Timing probe: skip binding + dispatch entirely (garbage results).
            let mut out_tensors = Vec::with_capacity(outputs.len());
            for out in outputs {
                let count: usize = out.shape.iter().map(|&d| d.max(0) as usize).product();
                let buffer = mdev.buffer(count * out.dtype.size_of(), "noop")?;
                let shape: Vec<usize> = out.shape.iter().map(|&d| d as usize).collect();
                out_tensors.push(Array::from_parts(mdev, buffer, &shape, out.dtype));
            }
            return Ok(out_tensors);
        }
        // Allocate outputs up front so their buffers can be bound.
        let mut out_tensors = Vec::with_capacity(outputs.len());
        for (oi, out) in outputs.iter().enumerate() {
            let count: usize = out.shape.iter().map(|&d| d.max(0) as usize).product();
            if let Some(pa) = prealloc {
                out_tensors.push(pa[oi].clone());
                continue;
            }
            let buffer = mdev.buffer((count) as usize * (out.dtype).size_of(), "lisa_kernel_out")?;
            let shape: Vec<usize> = out.shape.iter().map(|&d| d as usize).collect();
            out_tensors.push(Array::from_parts(mdev, buffer, &shape, out.dtype));
        }

        let guard = mdev.commands.encoder()?;
        let enc = guard.encoder();
        enc.set_pipeline(&pipeline);

        let mut index = 0usize;
        for (i, t) in inputs.iter().enumerate() {
            let (ms, layout) = t.buffer_and_layout();
            // A kernel that declares `{name}_strides` indexes through them, so a
            // non-contiguous view (e.g. a KV-cache slice) is fine; the shape /
            // stride / ndim buffers below carry its layout.
            if !layout.is_contiguous() && !self.shape_infos[i].1 {
                crate::bail!(
                    "kernel {}: input {} is not contiguous and the kernel declares no strides",
                    self.name,
                    self.input_names[i]
                );
            }
            enc.set_input(index, Some(ms), layout.offset * t.dtype().size_of());
            index += 1;
            if input_ndims[i] > 0 {
                let (want_shape, want_strides, want_ndim) = self.shape_infos[i];
                if want_shape {
                    let dims: Vec<i32> =
                        layout.shape().iter().map(|&d| d as i32).collect();
                    enc.set_bytes_directly(
                        index,
                        std::mem::size_of_val(dims.as_slice()),
                        dims.as_ptr().cast(),
                    );
                    index += 1;
                }
                if want_strides {
                    let strides: Vec<i64> =
                        layout.stride().iter().map(|&s| s as i64).collect();
                    enc.set_bytes_directly(
                        index,
                        std::mem::size_of_val(strides.as_slice()),
                        strides.as_ptr().cast(),
                    );
                    index += 1;
                }
                if want_ndim {
                    let ndim = input_ndims[i] as i32;
                    enc.set_bytes(index, &ndim);
                    index += 1;
                }
            }
        }
        for t in &out_tensors {
            let (ms, _) = t.buffer_and_layout();
            enc.set_output(index, Some(ms), 0);
            index += 1;
        }

        let _ = index;

        let tg_size = (thread_group.0 as usize)
            .saturating_mul(thread_group.1 as usize)
            .saturating_mul(thread_group.2 as usize);
        if tg_size > pipeline.max_total_threads_per_threadgroup() {
            crate::bail!(
                "kernel {}: thread group size {tg_size} exceeds max {}",
                self.name,
                pipeline.max_total_threads_per_threadgroup()
            );
        }
        let (gx, gy, gz) = grid;
        let (tx, ty, tz) = thread_group;
        let group = MTLSize {
            width: tx.min(gx) as usize,
            height: ty.min(gy) as usize,
            depth: tz.min(gz) as usize,
        };
        let grid_dims = MTLSize {
            width: gx.max(0) as usize,
            height: gy.max(0) as usize,
            depth: gz.max(0) as usize,
        };
        enc.dispatch_threads_size(grid_dims, group);

        Ok(out_tensors)
    }
}
























