//! DS4 dense, normalization and activation-cache dispatches.

use super::{MatvecArgs, f32_input, output};
use crate::error::{Error, Result};
use crate::runtime::{ConstVal, Math};
use crate::{Array, Dtype, Stream};
use objc2_metal::MTLDevice;

const SOURCE: &crate::shader_pack::Source = crate::kernel_source!("deepseek41/base.metal");

/// Native vectorized dense matvec for original F32/F16/BF16 weight views.
/// Multiple activation rows retain the scalar reduction tree per row.
pub fn plain_matvec(weights: &Array, input: &Array, stream: &Stream) -> Result<Array> {
    plain_matvec_impl(weights, input, true, stream)
}

fn plain_matvec_impl(
    weights: &Array,
    input: &Array,
    blend: bool,
    stream: &Stream,
) -> Result<Array> {
    if weights.rank() != 2
        || weights.dim(0) < 1
        || weights.dim(1) < 32
        || weights.dim(1) % 32 != 0
        || input.rank() != 2
        || input.dim(0) < 1
        || input.dim(0) > 65535
        || !weights.t.buffer_and_layout().1.is_contiguous()
        || weights.t.buffer_and_layout().1.offset * weights.dtype().size_of() % 16 != 0
    {
        return Err(Error::msg("unsupported DS4 plain matvec geometry"));
    }
    let blend = blend && weights.dtype() == Dtype::Bfloat16 && input.dim(0) == 1;
    let function = match weights.dtype() {
        Dtype::Float32 => "kernel_mul_mv_f32_f32_4",
        Dtype::Float16 => "kernel_mul_mv_f16_f32_4",
        Dtype::Bfloat16 if blend => "kernel_router_blend_r1",
        Dtype::Bfloat16 => "kernel_mul_mv_bf16_f32_4",
        _ => return Err(Error::msg("DS4 plain matvec requires F32/F16/BF16 weights")),
    };
    let (rows, columns, tokens) = (weights.dim(0), weights.dim(1), input.dim(0));
    f32_input(input, &[tokens, columns])?;
    weights.eval_enqueue()?;
    // DS4 uses one row on Apple10+ for the small output matrices. The scalar
    // reduction is identical; only independent output rows share a group.
    let tile = if blend || (rows <= 1024 && stream.runtime().nax()) {
        1
    } else {
        2
    };
    if rows % tile != 0 {
        return Err(Error::msg("DS4 plain matvec requires complete row tiles"));
    }
    let groups = ((columns + 127) / 128).min(8) as i16;
    let bytes = weights.dtype().size_of() as u64;
    let row_bytes = columns as u64 * bytes;
    let args = MatvecArgs {
        ne00: columns,
        ne01: rows,
        ne02: 1,
        nb00: bytes,
        nb01: row_bytes,
        nb02: row_bytes * rows as u64,
        nb03: row_bytes * rows as u64,
        ne10: columns,
        ne11: tokens,
        ne12: 1,
        nb10: 4,
        nb11: columns as u64 * 4,
        nb12: columns as u64 * tokens as u64 * 4,
        nb13: columns as u64 * tokens as u64 * 4,
        ne0: rows,
        ne1: tokens,
        nr0: tile,
        r2: 1,
        r3: 1,
    };
    let runtime = stream.runtime();
    let pipeline = runtime.compile_with_constants(
        SOURCE,
        function,
        Math::CompilerDefault,
        &[(600, ConstVal::Short(groups))],
    )?;
    let out = output(stream, &[tokens, rows], Dtype::Float32)?;
    let guard = runtime.commands.encoder()?;
    let encoder = guard.encoder();
    encoder.set_pipeline(&pipeline);
    encoder.set_bytes(0, &args);
    for (slot, array) in [(1, weights), (2, input)] {
        let (buffer, layout) = array.t.buffer_and_layout();
        encoder.set_input(slot, Some(buffer), layout.offset * array.dtype().size_of());
    }
    let (buffer, layout) = out.t.buffer_and_layout();
    encoder.set_output(3, Some(buffer), layout.offset * 4);
    encoder.set_threadgroup_memory_length(0, 32 * tile as usize * 4);
    encoder.dispatch_groups(
        (rows as usize / tile as usize, tokens as usize, 1),
        (32, groups as usize, 1),
    );
    Ok(out)
}

#[repr(C)]
struct NormArgs {
    ne00: i32,
    ne00_t: i32,
    nb1: u64,
    nb2: u64,
    nb3: u64,
    eps: f32,
    nef1: [i32; 3],
    nef2: [i32; 3],
    nef3: [i32; 3],
    nbf1: [u64; 3],
    nbf2: [u64; 3],
    nbf3: [u64; 3],
}

/// DS4's float4 RMSNorm tree, including its width-dependent thread count.
pub fn rms_norm(
    input: &Array,
    weight: Option<&Array>,
    eps: f32,
    round: bool,
    stream: &Stream,
) -> Result<Array> {
    if input.rank() != 2
        || input.dim(0) < 1
        || input.dim(0) > 65535
        || input.dim(1) < 128
        || input.dim(1) % 128 != 0
        || !eps.is_finite()
        || eps <= 0.0
        || (round && weight.is_none())
    {
        return Err(Error::msg("unsupported DS4 RMSNorm geometry"));
    }
    let (rows, width) = (input.dim(0), input.dim(1));
    f32_input(input, &[rows, width])?;
    if let Some(weight) = weight {
        f32_input(weight, &[width])?;
    }
    let row_bytes = width as u64 * 4;
    let args = NormArgs {
        ne00: width,
        ne00_t: width / 4,
        nb1: row_bytes,
        nb2: row_bytes * rows as u64,
        nb3: row_bytes * rows as u64,
        eps,
        nef1: [rows, 1, 1],
        nef2: [1; 3],
        nef3: [1; 3],
        nbf1: [row_bytes; 3],
        nbf2: [row_bytes * rows as u64, row_bytes, row_bytes],
        nbf3: [row_bytes * rows as u64, row_bytes, row_bytes],
    };
    let function = if round {
        "kernel_rms_norm_mul_f32_4_bf16"
    } else if weight.is_some() {
        "kernel_rms_norm_mul_f32_4"
    } else {
        "kernel_rms_norm_f32_4"
    };
    let runtime = stream.runtime();
    let pipeline = runtime.compile_with(SOURCE, function, Math::CompilerDefault)?;
    let width4 = width as usize / 4;
    let threads = width4.next_power_of_two().min(1024).min(width4);
    if threads > pipeline.max_total_threads_per_threadgroup() {
        return Err(Error::msg(
            "device cannot execute the DS4 RMSNorm thread count",
        ));
    }
    let out = output(stream, &[rows, width], Dtype::Float32)?;
    let guard = runtime.commands.encoder()?;
    let encoder = guard.encoder();
    encoder.set_pipeline(&pipeline);
    encoder.set_bytes(0, &args);
    for (slot, array) in [(1, input), (2, weight.unwrap_or(input)), (3, input)] {
        let (buffer, layout) = array.t.buffer_and_layout();
        encoder.set_input(slot, Some(buffer), layout.offset * 4);
    }
    let (buffer, layout) = out.t.buffer_and_layout();
    encoder.set_output(4, Some(buffer), layout.offset * 4);
    encoder.set_threadgroup_memory_length(0, 32 * 4);
    encoder.dispatch_groups((rows as usize, 1, 1), (threads, 1, 1));
    Ok(out)
}

/// Two DS4 weighted norms in one launch, retaining each row's independent
/// thread count and reduction tree. Inputs and outputs use f32 storage.
pub fn rms_norm_pair(
    x0: &Array,
    w0: &Array,
    x1: &Array,
    w1: &Array,
    eps: f32,
    stream: &Stream,
) -> Result<(Array, Array)> {
    if x0.rank() != 2
        || x1.rank() != 2
        || x0.dim(0) < 1
        || x0.dim(0) > 65535
        || x0.dim(0) != x1.dim(0)
        || !eps.is_finite()
        || eps <= 0.0
    {
        return Err(Error::msg("invalid DeepSeek paired normalization shape"));
    }
    let rows = x0.dim(0);
    let mut threads = [0usize; 2];
    for (i, (x, w)) in [(x0, w0), (x1, w1)].into_iter().enumerate() {
        let width = x.dim(1);
        if width < 128 || width % 128 != 0 {
            return Err(Error::msg("unsupported DeepSeek paired norm width"));
        }
        f32_input(x, &[rows, width])?;
        f32_input(w, &[width])?;
        let width4 = width as usize / 4;
        threads[i] = width4.next_power_of_two().min(1024).min(width4);
    }
    let runtime = stream.runtime();
    let pipeline = runtime.compile_with(SOURCE, "kernel_dsv41_norm_pair", Math::CompilerDefault)?;
    let total = threads[0] + threads[1];
    if total > 1024 || total > pipeline.max_total_threads_per_threadgroup() {
        return Err(Error::msg(
            "DeepSeek paired norm exceeds device thread count",
        ));
    }
    #[repr(C)]
    struct Args {
        n0: u32,
        n1: u32,
        ntg0: u32,
        ntg1: u32,
        eps: f32,
    }
    let args = Args {
        n0: x0.dim(1) as u32,
        n1: x1.dim(1) as u32,
        ntg0: threads[0] as u32,
        ntg1: threads[1] as u32,
        eps,
    };
    let y0 = output(stream, x0.shape(), Dtype::Float32)?;
    let y1 = output(stream, x1.shape(), Dtype::Float32)?;
    let guard = runtime.commands.encoder()?;
    let encoder = guard.encoder();
    encoder.set_pipeline(&pipeline);
    encoder.set_bytes(0, &args);
    for (slot, x) in [(1, x0), (2, w0), (4, x1), (5, w1)] {
        let (b, l) = x.t.buffer_and_layout();
        encoder.set_input(slot, Some(b), l.offset * 4);
    }
    for (slot, y) in [(3, &y0), (6, &y1)] {
        let (b, l) = y.t.buffer_and_layout();
        encoder.set_output(slot, Some(b), l.offset * 4);
    }
    encoder.set_threadgroup_memory_length(0, 64 * 4);
    encoder.dispatch_groups((1, rows as usize, 1), (total, 1, 1));
    Ok((y0, y1))
}

#[derive(Clone, Copy, Debug)]
#[repr(u32)]
pub enum ActivationFormat {
    Bf16 = 0,
    Fp8E8M0 = 1,
    Fp4E8M0 = 2,
    Fp4E4M3 = 3,
}

/// Raw and compressed (cos, sin) pairs shared by all layers at one position.
pub fn rope_coefficient_table(start: u32, stream: &Stream) -> Result<Array> {
    if start >= 1048576 {
        return Err(Error::msg("invalid DeepSeek RoPE position"));
    }
    #[repr(C)]
    struct Args {
        width: u32,
        start: u32,
        inverse: u32,
        frequencies: [f32; 32],
    }
    let args = |compressed| Args {
        width: 512,
        start,
        inverse: 1,
        frequencies: *super::rope_frequencies(compressed),
    };
    let runtime = stream.runtime();
    let pipeline = runtime.compile_with(
        SOURCE,
        "kernel_dsv41_rope_coeff_table",
        Math::CompilerDefault,
    )?;
    let table = output(stream, &[64, 2], Dtype::Float32)?;
    let guard = runtime.commands.encoder()?;
    let encoder = guard.encoder();
    encoder.set_pipeline(&pipeline);
    encoder.set_bytes(0, &args(false));
    encoder.set_bytes(1, &args(true));
    let (buffer, layout) = table.t.buffer_and_layout();
    encoder.set_output(2, Some(buffer), layout.offset * 4);
    encoder.dispatch_groups((1, 1, 1), (64, 1, 1));
    Ok(table)
}

/// Original DS4 decode index-query fusion, using this position's table.
pub fn index_query_rope_quantize(input: &mut Array, table: &Array, stream: &Stream) -> Result<()> {
    f32_input(input, &[32, 128])?;
    f32_input(table, &[64, 2])?;
    #[repr(C)]
    struct Args {
        width: u32,
        start: u32,
        inverse: u32,
        mode: u32,
        frequencies: [f32; 32],
    }
    let args = Args {
        width: 128,
        start: 0,
        inverse: 0,
        mode: 2,
        frequencies: [0.0; 32],
    };
    let runtime = stream.runtime();
    let pipeline = runtime.compile_with(
        SOURCE,
        "kernel_dsv41_rope_quantize_heads_tab",
        Math::CompilerDefault,
    )?;
    let guard = runtime.commands.encoder()?;
    let encoder = guard.encoder();
    encoder.set_pipeline(&pipeline);
    encoder.set_bytes(0, &args);
    let (buffer, layout) = input.t.buffer_and_layout();
    encoder.set_output(1, Some(buffer), layout.offset * 4);
    let (buffer, layout) = table.t.buffer_and_layout();
    encoder.set_input(2, Some(buffer), (layout.offset + 64) * 4);
    encoder.dispatch_groups((4, 32, 1), (32, 1, 1));
    *input = Array::new(input.t.clone());
    Ok(())
}

/// DS4's fused rotation, activation quantization and cache write. Its fast
/// divisions are part of the reference arithmetic; the standalone quantizer
/// uses precise divisions and can differ at FP4 rounding boundaries.
pub fn rope_quantize_store(
    input: &Array,
    cache: &mut Array,
    row: usize,
    start: u32,
    compressed: bool,
    format: ActivationFormat,
    stream: &Stream,
) -> Result<()> {
    if cache.rank() != 2 || row >= cache.dim(0) as usize || start >= 1048576 {
        return Err(Error::msg("invalid DeepSeek fused cache write"));
    }
    let width = cache.dim(1);
    let mode = format as u32;
    let block = if mode == 3 { 16 } else { 32 };
    if width < 64 || width % block != 0 || !(1..=3).contains(&mode) {
        return Err(Error::msg("invalid DeepSeek fused cache format"));
    }
    f32_input(input, &[1, width])?;
    f32_input(cache, &[cache.dim(0), width])?;
    #[repr(C)]
    struct Args {
        width: u32,
        start: u32,
        inverse: u32,
        mode: u32,
        frequencies: [f32; 32],
    }
    let args = Args {
        width: width as u32,
        start,
        inverse: 0,
        mode,
        frequencies: *super::rope_frequencies(compressed),
    };
    let runtime = stream.runtime();
    let pipeline =
        runtime.compile_with(SOURCE, "kernel_dsv41_rope_quantize", Math::CompilerDefault)?;
    let guard = runtime.commands.encoder()?;
    let encoder = guard.encoder();
    encoder.set_pipeline(&pipeline);
    encoder.set_bytes(0, &args);
    let (buffer, layout) = input.t.buffer_and_layout();
    encoder.set_input(1, Some(buffer), layout.offset * 4);
    let (buffer, layout) = cache.t.buffer_and_layout();
    encoder.set_output(2, Some(buffer), (layout.offset + row * width as usize) * 4);
    encoder.dispatch_groups(((width / block) as usize, 1, 1), (32, 1, 1));
    *cache = Array::new(cache.t.clone());
    Ok(())
}

/// DS4 activation quantization in f32 storage, including the BF16 boundaries.
/// This never changes or repacks a model weight tensor.
pub fn quantize_in_place(
    input: &mut Array,
    format: ActivationFormat,
    stream: &Stream,
) -> Result<()> {
    quantize_impl(input, format, true, stream)
}

fn quantize_impl(
    input: &mut Array,
    format: ActivationFormat,
    linear: bool,
    stream: &Stream,
) -> Result<()> {
    if input.rank() != 2 || input.dim(0) < 1 || input.dim(0) > 65535 || input.dim(1) < 1 {
        return Err(Error::msg("invalid DeepSeek activation quantization shape"));
    }
    f32_input(input, input.shape())?;
    #[repr(C)]
    struct Args {
        width: u32,
        rows: u32,
        mode: u32,
    }
    let args = Args {
        width: input.dim(1) as u32,
        rows: input.dim(0) as u32,
        mode: format as u32,
    };
    let runtime = stream.runtime();
    let count = u64::from(args.width) * u64::from(args.rows);
    let linear = linear
        && matches!(format, ActivationFormat::Bf16)
        && count <= u32::MAX as u64
        && input.t.buffer_and_layout().1.offset * 4 % 16 == 0;
    let function = if linear {
        "kernel_dsv41_bf16_linear"
    } else {
        "kernel_dsv41_quantize"
    };
    let pipeline = runtime.compile_with(SOURCE, function, Math::CompilerDefault)?;
    let guard = runtime.commands.encoder()?;
    let encoder = guard.encoder();
    encoder.set_pipeline(&pipeline);
    if linear {
        encoder.set_bytes(0, &count);
    } else {
        encoder.set_bytes(0, &args);
    }
    let (buffer, layout) = input.t.buffer_and_layout();
    encoder.set_output(1, Some(buffer), layout.offset * 4);
    let block = if matches!(format, ActivationFormat::Fp4E4M3) {
        16
    } else {
        32
    };
    if linear {
        encoder.dispatch_groups(((count as usize).div_ceil(1024), 1, 1), (256, 1, 1));
    } else {
        encoder.dispatch_groups(
            ((args.width as usize).div_ceil(block), args.rows as usize, 1),
            (32, 1, 1),
        );
    }
    *input = Array::new(input.t.clone());
    Ok(())
}

#[repr(C)]
struct HcInputArgs {
    n_embd: i64,
    n_hc: i32,
    sinkhorn_iters: i32,
    n_rows: i64,
    mix_hc: i64,
    nb_mix1: u64,
    nb_split1: u64,
    nb_x0: u64,
    nb_x1: u64,
    nb_x2: u64,
    nb0: u64,
    nb1: u64,
    nb_norm1: u64,
    eps: f32,
    norm_eps: f32,
}

pub struct HcInput {
    pub split: Array,
    pub collapsed: Array,
    pub norm: Array,
}

/// Final head collapse. Keep DS4's separate BF16 dispatch after the ordered
/// four-stream sum, rather than using a different fused reduction.
pub fn hc_collapse(residual: &Array, pre: &Array, stream: &Stream) -> Result<Array> {
    if residual.rank() != 3 || residual.dim(0) != 1 || residual.dim(1) != 4 || residual.dim(2) < 1 {
        return Err(Error::msg("invalid DeepSeek head collapse shape"));
    }
    let width = residual.dim(2);
    f32_input(residual, &[1, 4, width])?;
    f32_input(pre, &[4])?;
    #[repr(C)]
    struct Args {
        width: i64,
        hc: i64,
        tokens: i64,
        x0: u64,
        x1: u64,
        x2: u64,
        w0: u64,
        w1: u64,
        y0: u64,
        y1: u64,
        round: u64,
    }
    let row = width as u64 * 4;
    let args = Args {
        width: width as i64,
        hc: 4,
        tokens: 1,
        x0: 4,
        x1: row,
        x2: row * 4,
        w0: 4,
        w1: 16,
        y0: 4,
        y1: row,
        round: 0,
    };
    let runtime = stream.runtime();
    let pipeline =
        runtime.compile_with(SOURCE, "kernel_dsv4_hc_weighted_sum", Math::CompilerDefault)?;
    let mut out = output(stream, &[1, width], Dtype::Float32)?;
    {
        let guard = runtime.commands.encoder()?;
        let encoder = guard.encoder();
        encoder.set_pipeline(&pipeline);
        encoder.set_bytes(0, &args);
        for (slot, array) in [(1, residual), (2, pre)] {
            let (buffer, layout) = array.t.buffer_and_layout();
            encoder.set_input(slot, Some(buffer), layout.offset * 4);
        }
        let (buffer, layout) = out.t.buffer_and_layout();
        encoder.set_output(3, Some(buffer), layout.offset * 4);
        encoder.dispatch_groups(((width as usize).div_ceil(256), 1, 1), (256, 1, 1));
    }
    quantize_in_place(&mut out, ActivationFormat::Bf16, stream)?;
    Ok(out)
}

/// Single-token V4.1 HC split/collapse/norm, using the previous block's pre
/// weights. GLM's HC kernel uses the current block's pre weights instead.
pub fn hc_input(
    mixes: &Array,
    residual: &Array,
    previous_pre: &Array,
    scale: &Array,
    base: &Array,
    norm_weight: &Array,
    hc_eps: f32,
    norm_eps: f32,
    stream: &Stream,
) -> Result<HcInput> {
    if residual.rank() != 3
        || residual.dim(0) != 1
        || residual.dim(1) != 4
        || residual.dim(2) < 128
        || residual.dim(2) % 128 != 0
        || !hc_eps.is_finite()
        || hc_eps <= 0.0
        || !norm_eps.is_finite()
        || norm_eps <= 0.0
    {
        return Err(Error::msg("unsupported DeepSeek HC input geometry"));
    }
    let width = residual.dim(2);
    f32_input(residual, &[1, 4, width])?;
    for (array, shape) in [
        (mixes, &[1, 24][..]),
        (previous_pre, &[4][..]),
        (scale, &[3][..]),
        (base, &[24][..]),
        (norm_weight, &[width][..]),
    ] {
        f32_input(array, shape)?;
    }
    let row_bytes = width as u64 * 4;
    let args = HcInputArgs {
        n_embd: width as i64,
        n_hc: 4,
        sinkhorn_iters: 20,
        n_rows: 1,
        mix_hc: 24,
        nb_mix1: 96,
        nb_split1: 96,
        nb_x0: 4,
        nb_x1: row_bytes,
        nb_x2: row_bytes * 4,
        nb0: 4,
        nb1: row_bytes,
        nb_norm1: row_bytes,
        eps: hc_eps,
        norm_eps,
    };
    let runtime = stream.runtime();
    let pipeline = runtime.compile_with(
        SOURCE,
        "kernel_dsv41_hc_split_sum_norm",
        Math::CompilerDefault,
    )?;
    let threads = (width as usize / 4)
        .next_power_of_two()
        .min(1024)
        .min(width as usize / 4);
    let scratch = (width as usize + 4 + 32) * 4;
    if threads > pipeline.max_total_threads_per_threadgroup()
        || scratch > runtime.metal_device().maxThreadgroupMemoryLength()
    {
        return Err(Error::msg(
            "device cannot execute DS4 HC thread/scratch requirements",
        ));
    }
    let out = HcInput {
        split: output(stream, &[1, 24], Dtype::Float32)?,
        collapsed: output(stream, &[1, width], Dtype::Float32)?,
        norm: output(stream, &[1, width], Dtype::Float32)?,
    };
    let guard = runtime.commands.encoder()?;
    let encoder = guard.encoder();
    encoder.set_pipeline(&pipeline);
    encoder.set_bytes(0, &args);
    for (slot, array) in [
        (1, mixes),
        (2, scale),
        (3, base),
        (4, residual),
        (7, norm_weight),
        (9, previous_pre),
    ] {
        let (buffer, layout) = array.t.buffer_and_layout();
        encoder.set_input(slot, Some(buffer), layout.offset * 4);
    }
    for (slot, array) in [(5, &out.split), (6, &out.collapsed), (8, &out.norm)] {
        let (buffer, layout) = array.t.buffer_and_layout();
        encoder.set_output(slot, Some(buffer), layout.offset * 4);
    }
    encoder.set_threadgroup_memory_length(0, scratch);
    encoder.dispatch_groups((1, 1, 1), (threads, 1, 1));
    Ok(out)
}

#[repr(C)]
struct HcExpandArgs {
    n_embd: i64,
    n_hc: i64,
    n_tokens: i64,
    nb_block0: u64,
    nb_block1: u64,
    nb_add0: u64,
    nb_add1: u64,
    nb_res0: u64,
    nb_res1: u64,
    nb_res2: u64,
    nb_post0: u64,
    nb_post1: u64,
    nb_comb0: u64,
    nb_comb1: u64,
    nb_comb2: u64,
    nb0: u64,
    nb1: u64,
    nb2: u64,
    has_add: i32,
    round_bf16: i32,
    round_block_in: i32,
}

/// DS4's four-stream residual expansion, with the ordered expert add and its
/// separate input/output rounding boundaries preserved.
pub fn hc_expand(
    block: &Array,
    add: Option<&Array>,
    residual: &Array,
    split: &Array,
    round_block: bool,
    stream: &Stream,
) -> Result<Array> {
    if residual.rank() != 3
        || residual.dim(0) < 1
        || residual.dim(0) > 65535
        || residual.dim(1) != 4
        || residual.dim(2) < 1
    {
        return Err(Error::msg("invalid DS4 HC expansion geometry"));
    }
    let (tokens, width) = (residual.dim(0), residual.dim(2));
    f32_input(residual, &[tokens, 4, width])?;
    f32_input(block, &[tokens, width])?;
    f32_input(split, &[tokens, 24])?;
    if let Some(add) = add {
        f32_input(add, &[tokens, width])?;
    }
    let row = width as u64 * 4;
    let args = HcExpandArgs {
        n_embd: width as i64,
        n_hc: 4,
        n_tokens: tokens as i64,
        nb_block0: 4,
        nb_block1: row,
        nb_add0: 4,
        nb_add1: row,
        nb_res0: 4,
        nb_res1: row,
        nb_res2: row * 4,
        nb_post0: 4,
        nb_post1: 96,
        nb_comb0: 4,
        nb_comb1: 16,
        nb_comb2: 96,
        nb0: 4,
        nb1: row,
        nb2: row * 4,
        has_add: add.is_some() as i32,
        round_bf16: 1,
        round_block_in: round_block as i32,
    };
    let runtime = stream.runtime();
    let pipeline = runtime.compile_with(SOURCE, "kernel_dsv4_hc_expand4", Math::CompilerDefault)?;
    let out = output(stream, &[tokens, 4, width], Dtype::Float32)?;
    let guard = runtime.commands.encoder()?;
    let encoder = guard.encoder();
    encoder.set_pipeline(&pipeline);
    encoder.set_bytes(0, &args);
    for (slot, array, extra) in [
        (1, block, 0),
        (2, residual, 0),
        (3, split, 16),
        (4, split, 32),
        (5, add.unwrap_or(block), 0),
    ] {
        let (buffer, layout) = array.t.buffer_and_layout();
        encoder.set_input(slot, Some(buffer), layout.offset * 4 + extra);
    }
    let (buffer, layout) = out.t.buffer_and_layout();
    encoder.set_output(6, Some(buffer), layout.offset * 4);
    let count = tokens as usize * width as usize;
    if count > u32::MAX as usize {
        return Err(Error::msg("DS4 HC expansion index overflow"));
    }
    encoder.dispatch_groups((count.div_ceil(256), 1, 1), (256, 1, 1));
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bits(x: &Array) -> Vec<u32> {
        x.as_slice::<f32>().iter().map(|x| x.to_bits()).collect()
    }

    #[test]
    fn deepseek41_fused_cache_preserves_reference_fp4_midpoint() {
        let stream = Stream::gpu();
        // Original DS4 capture: token 128803 at position 1, layer 14,
        // compressor columns 176..192. Fast division rounds 0.146484375
        // above the E2M1 tie; the standalone precise division rounds below.
        let block = [
            -0.330078125f32,
            -0.06396484375,
            -0.6953125,
            0.043212890625,
            0.201171875,
            0.1357421875,
            -0.2373046875,
            -0.345703125,
            0.08642578125,
            -0.384765625,
            0.146484375,
            -0.26171875,
            0.412109375,
            -0.6640625,
            -0.369140625,
            -0.359375,
        ];
        let mut values = vec![0f32; 512];
        values[176..192].copy_from_slice(&block);
        let input = Array::from_slice(&values, &[1, 512]);
        let mut cache = Array::from_slice(&vec![7f32; 3 * 512], &[3, 512]);
        rope_quantize_store(
            &input,
            &mut cache,
            1,
            0,
            true,
            ActivationFormat::Fp4E4M3,
            &stream,
        )
        .unwrap();
        let actual = cache.as_slice::<f32>();
        assert!(actual[..512].iter().all(|&v| v == 7.0));
        assert!(actual[1024..].iter().all(|&v| v == 7.0));
        assert_eq!(actual[512 + 186].to_bits(), 0.17578125f32.to_bits());
        assert!(
            rope_quantize_store(
                &input,
                &mut cache,
                3,
                0,
                true,
                ActivationFormat::Fp4E4M3,
                &stream
            )
            .is_err()
        );
    }

    #[test]
    fn deepseek41_bf16_blend_matches_narrow_bits() {
        let stream = Stream::gpu();
        for (rows, width) in [
            (32, 32),
            (192, 5120),
            (512, 5120),
            (4096, 1280),
            (24, 20480),
        ] {
            let w: Vec<f32> = (0..rows * width)
                .map(|i| ((i * 19 % 997) as f32 - 499.0) / 73.0)
                .collect();
            let x: Vec<f32> = (0..width)
                .map(|i| ((i * 113 % 997) as f32 - 499.0) / 97.0)
                .collect();
            let w = Array::from_slice(&w, &[rows, width])
                .as_dtype(Dtype::Bfloat16)
                .unwrap();
            let x = Array::from_slice(&x, &[1, width]);
            let narrow = plain_matvec_impl(&w, &x, false, &stream).unwrap();
            let blend = plain_matvec(&w, &x, &stream).unwrap();
            assert_eq!(bits(&blend), bits(&narrow), "rows={rows} width={width}");
        }
    }

    #[test]
    fn deepseek41_linear_bf16_matches_scalar_bits() {
        let stream = Stream::gpu();
        // Exhaust all high words at both rounding ties and just below/above.
        // Include nonfinite patterns; DS4 truncates their low payload bits.
        let values: Vec<f32> = (0u32..65536)
            .flat_map(|hi| {
                [0, 0x7fff, 0x8000, 0x8001, 0xffff].map(|lo| f32::from_bits(hi << 16 | lo))
            })
            .collect();
        for length in [1, 3, 4, 1023, 1024, 1025, values.len()] {
            let mut reference = Array::from_slice(&values[..length], &[1, length as i32]);
            let mut actual = Array::from_slice(&values[..length], &[1, length as i32]);
            quantize_impl(&mut reference, ActivationFormat::Bf16, false, &stream).unwrap();
            quantize_in_place(&mut actual, ActivationFormat::Bf16, &stream).unwrap();
            assert_eq!(bits(&actual), bits(&reference), "length={length}");
        }
    }

    #[test]
    fn deepseek41_paired_norm_matches_separate_bits() {
        let stream = Stream::gpu();
        for rows in [1, 3, 8] {
            for (n0, n1) in [(128, 128), (1280, 512), (2048, 2048)] {
                let input = |width| {
                    Array::from_slice(
                        &(0..rows * width)
                            .map(|i| ((i * 113 % 997) as f32 - 499.0) / 73.0)
                            .collect::<Vec<_>>(),
                        &[rows, width],
                    )
                };
                let weight = |width| {
                    Array::from_slice(
                        &(0..width)
                            .map(|i| (i % 29 + 1) as f32 / 23.0)
                            .collect::<Vec<_>>(),
                        &[width],
                    )
                };
                let (x0, x1, w0, w1) = (input(n0), input(n1), weight(n0), weight(n1));
                let y0 = rms_norm(&x0, Some(&w0), 1e-20, true, &stream).unwrap();
                let y1 = rms_norm(&x1, Some(&w1), 1e-20, true, &stream).unwrap();
                let (a0, a1) = rms_norm_pair(&x0, &w0, &x1, &w1, 1e-20, &stream).unwrap();
                assert_eq!(bits(&a0), bits(&y0));
                assert_eq!(bits(&a1), bits(&y1));
            }
        }
    }

    #[test]
    fn deepseek41_plain_original_weight_dtypes() {
        let stream = Stream::gpu();
        for width in [128, 512, 1280, 5120, 20480] {
            let rows = 24;
            let w: Vec<f32> = (0..rows * width)
                .map(|i| ((i * 19 % 15) as f32 - 7.0) / 16.0)
                .collect();
            let x: Vec<f32> = (0..3 * width)
                .map(|i| ((i * 13 % 7) as f32 - 3.0) / 8.0)
                .collect();
            for dtype in [Dtype::Float32, Dtype::Float16, Dtype::Bfloat16] {
                let weights = Array::from_slice(&w, &[rows as i32, width as i32])
                    .as_dtype(dtype)
                    .unwrap();
                let input = Array::from_slice(&x, &[3, width as i32]);
                let out = plain_matvec(&weights, &input, &stream).unwrap();
                for (index, &value) in out.as_slice::<f32>().iter().enumerate() {
                    let (token, row) = (index / rows, index % rows);
                    let expected: f32 = w[row * width..(row + 1) * width]
                        .iter()
                        .zip(&x[token * width..(token + 1) * width])
                        .map(|(w, x)| w * x)
                        .sum();
                    assert_eq!(value, expected, "{dtype:?} width={width} index={index}");
                }
            }
        }
    }

    #[test]
    fn deepseek41_norm_widths_and_hc_previous_pre() {
        let stream = Stream::gpu();
        for width in [128, 512, 1280, 5120, 20480] {
            let x: Vec<f32> = (0..3 * width)
                .map(|i| if i % 3 == 0 { -1.0 } else { 1.0 })
                .collect();
            let weights: Vec<f32> = (0..width).map(|i| (i % 31) as f32 / 16.0).collect();
            let input = Array::from_slice(&x, &[3, width as i32]);
            let w = Array::from_slice(&weights, &[width as i32]);
            for round in [false, true] {
                let out = rms_norm(&input, Some(&w), 1e-20, round, &stream).unwrap();
                for (i, &actual) in out.as_slice::<f32>().iter().enumerate() {
                    assert_eq!(actual, x[i] * weights[i % width]);
                }
            }
        }
        let width = 5120;
        let values: Vec<f32> = (0..4 * width)
            .map(|i| ((i * 7 % 23) as f32 - 11.0) / 4.0)
            .collect();
        let residual = Array::from_slice(&values, &[1, 4, width as i32]);
        let first = Array::from_slice(&values[..width], &[1, width as i32]);
        let norm_weight = Array::from_slice(&vec![1f32; width], &[width as i32]);
        let expected = rms_norm(&first, Some(&norm_weight), 1e-20, true, &stream).unwrap();
        let mixes = Array::from_slice(&[0.0f32; 24], &[1, 24]);
        let base = Array::from_slice(&[0.0f32; 24], &[24]);
        let scale = Array::from_slice(&[1.0f32; 3], &[3]);
        let pre = Array::from_slice(&[1.0f32, 0.0, 0.0, 0.0], &[4]);
        let out = hc_input(
            &mixes,
            &residual,
            &pre,
            &scale,
            &base,
            &norm_weight,
            1e-6,
            1e-20,
            &stream,
        )
        .unwrap();
        assert_eq!(out.collapsed.as_slice::<f32>(), &values[..width]);
        assert_eq!(out.norm.as_slice::<f32>(), expected.as_slice::<f32>());
        assert!(
            out.split.as_slice::<f32>()[..4]
                .iter()
                .all(|&x| x == 0.5 + 1e-6)
        );
        let expanded = hc_expand(&first, None, &residual, &out.split, false, &stream).unwrap();
        let head = hc_collapse(&residual, &pre, &stream).unwrap();
        assert_eq!(head.as_slice::<f32>(), &values[..width]);
        let split = out.split.as_slice::<f32>();
        for channel in 0..4 {
            for i in 0..width {
                let mut value = values[i] * split[4 + channel];
                for src in 0..4 {
                    value += split[8 + src * 4 + channel] * values[src * width + i];
                }
                assert_eq!(
                    expanded.as_slice::<f32>()[channel * width + i],
                    half::bf16::from_f32(value).to_f32()
                );
            }
        }
    }

    #[test]
    fn deepseek41_activation_quantization_matches_codebook_rounding() {
        fn bf(x: f32) -> f32 {
            half::bf16::from_f32(x).to_f32()
        }
        fn e4(code: u32) -> f32 {
            let (exponent, mantissa) = (code >> 3, code & 7);
            if exponent == 0 {
                mantissa as f32 / 512.0
            } else {
                (1.0 + mantissa as f32 / 8.0) * 2f32.powi(exponent as i32 - 7)
            }
        }
        fn nearest(x: f32, values: &[f32]) -> f32 {
            let mut best = 0;
            for i in 1..values.len() {
                let distance = (x.abs() - values[i]).abs();
                let old = (x.abs() - values[best]).abs();
                if distance < old || (distance == old && i % 2 == 0 && best % 2 != 0) {
                    best = i;
                }
            }
            values[best].copysign(x)
        }
        fn pow2(x: f32) -> f32 {
            let bits = x.to_bits();
            f32::from_bits((bits & 0x7f800000) + if bits & 0x7fffff != 0 { 0x800000 } else { 0 })
        }
        let stream = Stream::gpu();
        let e4: Vec<f32> = (0..127).map(e4).collect();
        let e2 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
        let input: Vec<f32> = (0..512 * 128)
            .map(|i| ((i * 73 % 499) as f32 - 249.0) / 17.0 * 2f32.powi((i / 128 % 13) - 6))
            .collect();
        for format in [
            ActivationFormat::Bf16,
            ActivationFormat::Fp8E8M0,
            ActivationFormat::Fp4E8M0,
            ActivationFormat::Fp4E4M3,
        ] {
            let mut x = Array::from_slice(&input, &[512, 128]);
            quantize_in_place(&mut x, format, &stream).unwrap();
            let block = if matches!(format, ActivationFormat::Fp4E4M3) {
                16
            } else {
                32
            };
            for (original, actual) in input
                .chunks_exact(block)
                .zip(x.as_slice::<f32>().chunks_exact(block))
            {
                let max = original.iter().map(|&x| bf(x).abs()).fold(0f32, f32::max);
                let scale = match format {
                    ActivationFormat::Bf16 => 1.0,
                    ActivationFormat::Fp8E8M0 => pow2(max.max(1e-4) * (1.0 / 448.0)),
                    ActivationFormat::Fp4E8M0 => pow2(max.max(7.052966e-38) * (1.0 / 6.0)),
                    ActivationFormat::Fp4E4M3 => nearest(max.max(0.01171875) / 6.0, &e4),
                };
                for (&original, &actual) in original.iter().zip(actual) {
                    let value = bf(original);
                    let expected = if matches!(format, ActivationFormat::Bf16) {
                        value
                    } else {
                        bf(nearest(
                            value / scale,
                            if matches!(format, ActivationFormat::Fp8E8M0) {
                                &e4
                            } else {
                                &e2
                            },
                        ) * scale)
                    };
                    assert_eq!(
                        actual, expected,
                        "{format:?} input={original} scale={scale}"
                    );
                }
            }
        }
    }
}
