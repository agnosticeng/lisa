use crate::error::Result;
use crate::runtime::{ComputePipeline, ConstVal, Math};

use super::{DType, Device};

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
pub(super) fn compile_builtin(
    device: &Device,
    source: &str,
    name: &str,
) -> Result<ComputePipeline> {
    compile_builtin_bool_consts(device, source, name, &[])
}

/// Like [`compile_builtin`] with explicit math compile options (tail-ULP
/// experiments only — the stock built-in class is `Math::SafeNoLang`).
pub(super) fn compile_builtin_math(
    device: &Device,
    source: &str,
    name: &str,
    math: Math,
) -> Result<ComputePipeline> {
    device.compile_with(source, name, math)
}

/// Like [`compile_builtin`] but can specialise Metal bool function constants.
pub(super) fn compile_builtin_bool_consts(
    device: &Device,
    source: &str,
    name: &str,
    consts: &[(usize, bool)],
) -> Result<ComputePipeline> {
    let c: Vec<(usize, ConstVal)> = consts
        .iter()
        .map(|(i, v)| (*i, ConstVal::Bool(*v)))
        .collect();
    device.compile_with_constants(source, name, Math::SafeNoLang, &c)
}

/// Compile a library and fetch `name`, specialising bool/int function
/// constants.
pub(super) fn compile_builtin_typed_consts(
    device: &Device,
    source: &str,
    name: &str,
    consts: &[(usize, ConstVal)],
) -> Result<ComputePipeline> {
    device.compile_with_constants(source, name, Math::SafeNoLang, consts)
}

/// MLX `backend/metal/utils.cpp:8` `type_to_name` — used in the kernel name.
pub(super) fn type_to_name(dt: DType) -> Result<&'static str> {
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
pub(super) fn work_per_thread(dt: DType, size: usize) -> usize {
    const WPT_THRESHOLD: usize = 1 << 16;
    if size < WPT_THRESHOLD {
        1
    } else {
        (8 / dt.size_of()).max(1)
    }
}

/// MLX `get_2d_grid_dims` (`common/utils.cpp`).
pub(super) fn get_2d_grid_dims(shape: &[usize], strides: &[i64]) -> (usize, usize) {
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

pub(super) fn tg_from_row_size(row_size: usize) -> usize {
    if row_size <= 512 {
        32
    } else if row_size <= 1024 {
        128
    } else {
        ((row_size.div_ceil(4) + 31) / 32 * 32).min(1024)
    }
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

/// MLX `get_type_string` (`backend/common/compiled.cpp:31`).
pub(super) fn type_string(dt: DType) -> Result<&'static str> {
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
        DType::Bool => "bool",
        other => crate::bail!("unsupported compilation type {other:?}"),
    })
}
