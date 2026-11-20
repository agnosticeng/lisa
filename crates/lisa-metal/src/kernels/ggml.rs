//! Native DS4 dispatch ABI. Kernel bodies remain byte-pinned beside the shader.
//! In-place operations require caller-owned scratch, not immutable model views.

use crate::error::{Error, Result};
use crate::runtime::Math;
use crate::{Array, Dtype, Stream};

mod base;
pub use base::*;
mod q8_pair;
pub use q8_pair::*;
mod pointwise;
pub use pointwise::*;
mod moe;
pub use moe::*;
mod attention;
pub use attention::*;
mod indexer;
pub use indexer::*;

const SOURCE: &crate::shader_pack::Source = crate::kernel_source!("deepseek41/pointwise.metal");
const Q8_SOURCE: &crate::shader_pack::Source = crate::kernel_source!("deepseek41/q8.metal");
const IQ_SOURCE: &crate::shader_pack::Source = crate::kernel_source!("deepseek41/iq.metal");

/// Original GGML block formats. These are not MLX affine quantization modes.
#[derive(Clone, Copy, Debug)]
pub enum BlockFormat {
    Iq2Xxs,
    Iq2Xs,
    Iq1M,
    Q5K,
    Q6K,
    Q2K,
    Q4K,
}

impl BlockFormat {
    fn geometry(self) -> (i32, &'static str, &'static str, usize) {
        match self {
            Self::Iq2Xxs => (66, "lisa_gguf_iq2_xxs_mv", "lisa_gguf_iq2_xxs_id", 2176),
            Self::Iq2Xs => (74, "lisa_gguf_iq2_xs_mv", "lisa_gguf_iq2_xs_id", 4224),
            Self::Iq1M => (56, "lisa_gguf_iq1_m_mv", "lisa_gguf_iq1_m_id", 16),
            Self::Q5K => (176, "lisa_gguf_q5_K_mv", "lisa_gguf_q5_K_id", 16),
            Self::Q6K => (210, "lisa_gguf_q6_K_mv", "lisa_gguf_q6_K_id", 16),
            Self::Q2K => (84, "lisa_gguf_q2_K_mv", "lisa_gguf_q2_K_id", 32),
            Self::Q4K => (144, "lisa_gguf_q4_K_mv", "lisa_gguf_q4_K_id", 32),
        }
    }

    fn rows_per_simd(self) -> i32 {
        if matches!(self, Self::Q4K) { 2 } else { 4 }
    }

    fn columns(self, weights: &Array, rank: usize) -> Result<i32> {
        let block_bytes = self.geometry().0;
        if weights.rank() != rank
            || weights.dtype() != Dtype::Uint8
            || weights.shape().iter().any(|&d| d <= 0)
            || weights.dim(rank as i32 - 2) % 8 != 0
            || weights.dim(rank as i32 - 1) % block_bytes != 0
            || !weights.t.buffer_and_layout().1.is_contiguous()
            || weights.t.buffer_and_layout().1.offset % 2 != 0
        {
            return Err(Error::msg("unsupported GGUF quantized matvec layout"));
        }
        (weights.dim(rank as i32 - 1) / block_bytes)
            .checked_mul(256)
            .ok_or_else(|| Error::msg("GGUF quantized input width overflow"))
    }
}

fn iq_pipeline(
    format: BlockFormat,
    indirect: bool,
    groups: i16,
    stream: &Stream,
) -> Result<crate::runtime::ComputePipeline> {
    use crate::runtime::ConstVal;
    let (_, dense, expert, _) = format.geometry();
    stream.runtime().compile_with_constants(
        IQ_SOURCE,
        if indirect { expert } else { dense },
        Math::CompilerDefault,
        &[
            (600, ConstVal::Short(2)),
            (602, ConstVal::Short(groups)),
            (603, ConstVal::Short(1)),
            (604, ConstVal::Short(1)),
            (605, ConstVal::Bool(false)),
        ],
    )
}

/// Native packed GGML matvec, with one independent matvec per input row.
/// The grouped prefill schedule is a separate operation. Eight complete output
/// rows are required because the reference helpers read their full row tile.
pub fn packed_matvec(
    format: BlockFormat,
    weights: &Array,
    input: &Array,
    stream: &Stream,
) -> Result<Array> {
    format.columns(weights, 2)?;
    if input.rank() != 2 || input.dim(0) < 1 || input.dim(0) > 65535 {
        return Err(Error::msg("invalid GGUF matvec input rows"));
    }
    let out = grouped_packed_matvec(
        format,
        &weights.reshape(&[1, weights.dim(0), weights.dim(1)])?,
        &input.reshape(&[1, input.dim(0), input.dim(1)])?,
        stream,
    )?;
    out.reshape(&[input.dim(0), weights.dim(0)])
}

/// Independent matrices in one launch, using GGML's original group addressing.
/// Weights `[groups, output, packed_row_bytes]`, inputs `[groups, tokens, width]`.
/// Each group's reduction is identical to `packed_matvec`; no cross-group sum.
pub fn grouped_packed_matvec(
    format: BlockFormat,
    weights: &Array,
    input: &Array,
    stream: &Stream,
) -> Result<Array> {
    let columns = format.columns(weights, 3)?;
    let groups = weights.dim(0);
    if groups > i16::MAX as i32 || input.rank() != 3 || input.dim(1) < 1 || input.dim(1) > 65535 {
        return Err(Error::msg("invalid grouped GGUF matvec geometry"));
    }
    let tokens = input.dim(1);
    let rows = weights.dim(1);
    f32_input(input, &[groups, tokens, columns])?;
    weights.eval_enqueue()?;
    let row_bytes = weights.dim(2) as u64;
    let args = MatvecArgs {
        ne00: columns,
        ne01: rows,
        ne02: groups,
        nb00: format.geometry().0 as u64,
        nb01: row_bytes,
        nb02: row_bytes * rows as u64,
        nb03: row_bytes * rows as u64 * groups as u64,
        ne10: columns,
        ne11: tokens,
        ne12: groups,
        nb10: 4,
        nb11: columns as u64 * 4,
        nb12: columns as u64 * tokens as u64 * 4,
        nb13: columns as u64 * tokens as u64 * groups as u64 * 4,
        ne0: rows,
        ne1: tokens,
        nr0: format.rows_per_simd(),
        r2: 1,
        r3: 1,
    };
    let pipeline = iq_pipeline(format, false, groups as i16, stream)?;
    let out = output(stream, &[groups, tokens, rows], Dtype::Float32)?;
    let guard = stream.runtime().commands.encoder()?;
    let encoder = guard.encoder();
    encoder.set_pipeline(&pipeline);
    encoder.set_bytes(0, &args);
    for (slot, array) in [(1, weights), (2, input)] {
        let (buffer, layout) = array.t.buffer_and_layout();
        encoder.set_input(slot, Some(buffer), layout.offset * array.dtype().size_of());
    }
    let (buffer, layout) = out.t.buffer_and_layout();
    encoder.set_output(3, Some(buffer), layout.offset * 4);
    encoder.set_threadgroup_memory_length(0, format.geometry().3);
    encoder.dispatch_groups(
        (
            rows as usize / (2 * format.rows_per_simd()) as usize,
            tokens as usize,
            groups as usize,
        ),
        (32, 2, 1),
    );
    Ok(out)
}

#[repr(C)]
struct ExpertMatvecArgs {
    nei0: i32,
    nei1: i32,
    nbi1: u64,
    ne00: i32,
    ne01: i32,
    ne02: i32,
    nb00: u64,
    nb01: u64,
    nb02: u64,
    ne10: i32,
    ne11: i32,
    ne12: i32,
    ne13: i32,
    nb10: u64,
    nb11: u64,
    nb12: u64,
    ne0: i32,
    ne1: i32,
    nb1: u64,
    nr0: i32,
    tp_rank: i32,
    tp_world: i32,
    tp_addend: i32,
    tp_expert_base: i32,
}

/// GPU-selected expert matvec. Weights are `[experts, output, packed_row_bytes]`,
/// input `[tokens, 1 or top_k, width]`, ids `[tokens, top_k]`. Invalid ids write
/// zero without touching weight memory; the model supplies ids from `router`.
pub fn expert_matvec(
    format: BlockFormat,
    weights: &Array,
    input: &Array,
    ids: &Array,
    stream: &Stream,
) -> Result<Array> {
    let columns = format.columns(weights, 3)?;
    if ids.rank() != 2
        || ids.dim(0) < 1
        || ids.dim(0) > 65535
        || ids.dim(1) < 1
        || ids.dim(1) > 32
        || ids.dtype() != Dtype::Int32
        || !ids.t.buffer_and_layout().1.is_contiguous()
        || input.rank() != 3
        || (input.dim(1) != 1 && input.dim(1) != ids.dim(1))
    {
        return Err(Error::msg("invalid GGUF expert matvec geometry"));
    }
    let tokens = ids.dim(0);
    let top_k = ids.dim(1);
    let slots = input.dim(1);
    f32_input(input, &[tokens, slots, columns])?;
    weights.eval_enqueue()?;
    ids.eval_enqueue()?;
    let rows = weights.dim(1);
    let row_bytes = weights.dim(2) as u64;
    let args = ExpertMatvecArgs {
        nei0: top_k,
        nei1: tokens,
        nbi1: top_k as u64 * 4,
        ne00: columns,
        ne01: rows,
        ne02: weights.dim(0),
        nb00: format.geometry().0 as u64,
        nb01: row_bytes,
        nb02: row_bytes * rows as u64,
        ne10: columns,
        ne11: slots,
        ne12: tokens,
        ne13: 1,
        nb10: 4,
        nb11: columns as u64 * 4,
        nb12: columns as u64 * slots as u64 * 4,
        ne0: rows,
        ne1: top_k,
        nb1: rows as u64 * 4,
        nr0: format.rows_per_simd(),
        tp_rank: 0,
        tp_world: 1,
        tp_addend: 0,
        tp_expert_base: 0,
    };
    let pipeline = iq_pipeline(format, true, 1, stream)?;
    let out = output(stream, &[tokens, top_k, rows], Dtype::Float32)?;
    let guard = stream.runtime().commands.encoder()?;
    let encoder = guard.encoder();
    encoder.set_pipeline(&pipeline);
    encoder.set_bytes(0, &args);
    for (slot, array) in [(1, weights), (2, input), (4, ids)] {
        let (buffer, layout) = array.t.buffer_and_layout();
        encoder.set_input(slot, Some(buffer), layout.offset * array.dtype().size_of());
    }
    let (buffer, layout) = out.t.buffer_and_layout();
    encoder.set_output(3, Some(buffer), layout.offset * 4);
    encoder.set_threadgroup_memory_length(0, format.geometry().3);
    encoder.dispatch_groups(
        (
            rows as usize / (2 * format.rows_per_simd()) as usize,
            1,
            tokens as usize * top_k as usize,
        ),
        (32, 2, 1),
    );
    Ok(out)
}

#[repr(C)]
struct MatvecArgs {
    ne00: i32,
    ne01: i32,
    ne02: i32,
    nb00: u64,
    nb01: u64,
    nb02: u64,
    nb03: u64,
    ne10: i32,
    ne11: i32,
    ne12: i32,
    nb10: u64,
    nb11: u64,
    nb12: u64,
    nb13: u64,
    ne0: i32,
    ne1: i32,
    nr0: i32,
    r2: i16,
    r3: i16,
}

/// Original GGML Q8_0 blocks through DS4's native scalar matvec reductions.
/// Two complete output rows are required because the kernel reads both before
/// its output bounds check. Prefill and small-batch schedules are separate.
pub fn q8_matvec(
    weights: &Array,
    input: &Array,
    round_output: bool,
    round_input: bool,
    stream: &Stream,
) -> Result<Array> {
    if weights.rank() != 2
        || weights.dtype() != Dtype::Uint8
        || weights.dim(0) < 2
        || weights.dim(0) % 2 != 0
        || weights.dim(1) < 34
        || weights.dim(1) % 34 != 0
        || round_input && !round_output
        || !weights.t.buffer_and_layout().1.is_contiguous()
        || weights.t.buffer_and_layout().1.offset % 2 != 0
    {
        return Err(Error::msg("unsupported DS4 Q8_0 matvec layout"));
    }
    let columns = weights
        .dim(1)
        .checked_div(34)
        .and_then(|n| n.checked_mul(32))
        .ok_or_else(|| Error::msg("Q8_0 input width overflow"))?;
    let rows = weights.dim(0);
    if input.rank() != 2 || input.dim(0) < 1 || input.dim(0) > 8 {
        return Err(Error::msg(
            "DS4 Q8_0 scalar matvec supports one to eight input rows",
        ));
    }
    let tokens = input.dim(0);
    f32_input(input, &[tokens, columns])?;
    weights.eval_enqueue()?;
    let row_bytes = weights.dim(1) as u64;
    let args = MatvecArgs {
        ne00: columns,
        ne01: rows,
        ne02: 1,
        nb00: 34,
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
        nr0: 2,
        r2: 1,
        r3: 1,
    };
    let function = if round_input {
        "kernel_mul_mv_q8_0_f32_bf16io"
    } else if round_output {
        "kernel_mul_mv_q8_0_f32_bf16"
    } else {
        "kernel_mul_mv_q8_0_f32"
    };
    let groups = if rows > 65536 { 8 } else { 4 };
    let runtime = stream.runtime();
    let pipeline = runtime.compile_with_constants(
        Q8_SOURCE,
        function,
        Math::CompilerDefault,
        &[(600, crate::runtime::ConstVal::Short(groups))],
    )?;
    let out = output(stream, &[tokens, rows], Dtype::Float32)?;
    let guard = runtime.commands.encoder()?;
    let encoder = guard.encoder();
    encoder.set_pipeline(&pipeline);
    encoder.set_bytes(0, &args);
    for (slot, input) in [(1, weights), (2, input)] {
        let (buffer, layout) = input.t.buffer_and_layout();
        encoder.set_input(slot, Some(buffer), layout.offset * input.dtype().size_of());
    }
    let (buffer, layout) = out.t.buffer_and_layout();
    encoder.set_output(3, Some(buffer), layout.offset * 4);
    encoder.set_threadgroup_memory_length(0, 32 * 2 * 4);
    encoder.dispatch_groups(
        (rows as usize / 2, tokens as usize, 1),
        (32, groups as usize, 1),
    );
    Ok(out)
}

/// GPU-selected Q8_0 projections with the exact dense scalar reduction.
/// Weights `[experts, rows, bytes]`, inputs `[tokens, 1 or top_k, columns]`,
/// indices `[tokens, top_k]`. No selected weight payloads are copied.
pub fn q8_expert_matvec(
    weights: &Array,
    input: &Array,
    ids: &Array,
    stream: &Stream,
) -> Result<Array> {
    if weights.rank() != 3
        || weights.dtype() != Dtype::Uint8
        || weights.shape().iter().any(|&n| n <= 0)
        || weights.dim(1) % 2 != 0
        || weights.dim(2) % 34 != 0
        || !weights.t.buffer_and_layout().1.is_contiguous()
        || weights.t.buffer_and_layout().1.offset % 2 != 0
        || ids.rank() != 2
        || ids.dtype() != Dtype::Int32
        || ids.dim(0) < 1
        || ids.dim(0) > 65535
        || ids.dim(1) < 1
        || ids.dim(1) > 32
        || !ids.t.buffer_and_layout().1.is_contiguous()
        || input.rank() != 3
        || (input.dim(1) != 1 && input.dim(1) != ids.dim(1))
    {
        return Err(Error::msg("invalid Q8 expert matvec geometry"));
    }
    let columns = weights
        .dim(2)
        .checked_div(34)
        .and_then(|n| n.checked_mul(32))
        .ok_or_else(|| Error::msg("Q8 expert input width overflow"))?;
    let rows = weights.dim(1);
    let tokens = ids.dim(0);
    let top_k = ids.dim(1);
    f32_input(input, &[tokens, input.dim(1), columns])?;
    weights.eval_enqueue()?;
    ids.eval_enqueue()?;
    let row_bytes = weights.dim(2) as u64;
    let args = MatvecArgs {
        ne00: columns,
        ne01: rows,
        ne02: 1,
        nb00: 34,
        nb01: row_bytes,
        nb02: row_bytes * rows as u64,
        nb03: row_bytes * rows as u64,
        ne10: columns,
        ne11: 1,
        ne12: 1,
        nb10: 4,
        nb11: columns as u64 * 4,
        nb12: columns as u64 * 4,
        nb13: columns as u64 * 4,
        ne0: rows,
        ne1: tokens,
        nr0: 2,
        r2: 1,
        r3: 1,
    };
    // At K <= 512, the old extra SIMD groups only contributed positive zeros.
    // Active groups still visit exactly the same blocks once, and the final
    // 32-lane sum still has the same positive-zero padding. Wider inputs keep
    // the original schedule because changing its stride would reorder sums.
    let groups = if columns <= 256 {
        1
    } else if columns <= 512 {
        2
    } else if rows > 65536 {
        8
    } else {
        4
    };
    let pipeline = stream.runtime().compile_with_constants(
        Q8_SOURCE,
        "lisa_q8_expert_f32",
        Math::CompilerDefault,
        &[(600, crate::runtime::ConstVal::Short(groups))],
    )?;
    let out = output(stream, &[tokens, top_k, rows], Dtype::Float32)?;
    let guard = stream.runtime().commands.encoder()?;
    let encoder = guard.encoder();
    encoder.set_pipeline(&pipeline);
    encoder.set_bytes(0, &args);
    encoder.set_bytes(
        5,
        &[top_k as u32, input.dim(1) as u32, weights.dim(0) as u32, 0],
    );
    for (slot, array) in [(1, weights), (2, input), (4, ids)] {
        let (buffer, layout) = array.t.buffer_and_layout();
        encoder.set_input(slot, Some(buffer), layout.offset * array.dtype().size_of());
    }
    let (buffer, layout) = out.t.buffer_and_layout();
    encoder.set_output(3, Some(buffer), layout.offset * 4);
    encoder.set_threadgroup_memory_length(0, 32 * 2 * 4);
    encoder.dispatch_groups(
        (rows as usize / 2, 1, tokens as usize * top_k as usize),
        (32, groups as usize, 1),
    );
    Ok(out)
}

fn f32_input(x: &Array, shape: &[i32]) -> Result<()> {
    if x.dtype() != Dtype::Float32
        || x.shape() != shape
        || !x.t.buffer_and_layout().1.is_contiguous()
    {
        return Err(Error::msg(format!(
            "DeepSeek kernel expects contiguous f32 {shape:?}, got {:?}",
            x.shape()
        )));
    }
    x.eval_enqueue()
}

fn output(stream: &Stream, shape: &[i32], dtype: Dtype) -> Result<Array> {
    let bytes = shape
        .iter()
        .try_fold(dtype.size_of(), |n, &d| n.checked_mul(d as usize))
        .ok_or_else(|| Error::msg("DeepSeek output size overflow"))?;
    let runtime = stream.runtime();
    let buffer = runtime.buffer_for_full_gpu_write(bytes, "deepseek41_output")?;
    let shape: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
    Ok(Array::new(crate::array::Array::from_parts(
        runtime, buffer, &shape, dtype,
    )))
}

pub struct RouterOutput {
    pub probabilities: Array,
    pub selected: Array,
    pub weights: Array,
}

/// Exact DS4 softplus/sqrt, biased bitonic selection, and normalized weights.
pub fn router(
    logits: &Array,
    bias: &Array,
    top_k: u32,
    scale: f32,
    stream: &Stream,
) -> Result<RouterOutput> {
    if logits.rank() != 2
        || logits.dim(0) < 1
        || logits.dim(0) > 65535
        || logits.dim(1) < 1
        || logits.dim(1) > 1024
        || top_k == 0
        || top_k > 32
        || top_k > logits.dim(1) as u32
        || !scale.is_finite()
        || scale <= 0.0
    {
        return Err(Error::msg("unsupported DeepSeek router geometry"));
    }
    let rows = logits.dim(0);
    let experts = logits.dim(1);
    f32_input(logits, &[rows, experts])?;
    f32_input(bias, &[experts])?;
    let runtime = stream.runtime();
    let pipeline =
        runtime.compile_with(SOURCE, "kernel_dsv41_router_one", Math::CompilerDefault)?;
    let threads = (experts as usize).next_power_of_two();
    if threads > pipeline.max_total_threads_per_threadgroup() {
        return Err(Error::msg(
            "DeepSeek router exceeds device threadgroup limit",
        ));
    }
    let result = RouterOutput {
        probabilities: output(stream, &[rows, experts], Dtype::Float32)?,
        selected: output(stream, &[rows, top_k as i32], Dtype::Int32)?,
        weights: output(stream, &[rows, top_k as i32], Dtype::Float32)?,
    };
    #[repr(C)]
    struct Args {
        experts: u32,
        top_k: u32,
        has_bias: u32,
        scale: f32,
    }
    let args = Args {
        experts: experts as u32,
        top_k,
        has_bias: 1,
        scale,
    };
    let guard = runtime.commands.encoder()?;
    let encoder = guard.encoder();
    encoder.set_pipeline(&pipeline);
    encoder.set_bytes(0, &args);
    for (slot, input) in [(1, logits), (2, bias)] {
        let (buffer, layout) = input.t.buffer_and_layout();
        encoder.set_input(slot, Some(buffer), layout.offset * 4);
    }
    for (slot, out) in [
        (3, &result.probabilities),
        (4, &result.selected),
        (5, &result.weights),
    ] {
        let (buffer, layout) = out.t.buffer_and_layout();
        encoder.set_output(slot, Some(buffer), layout.offset * 4);
    }
    encoder.set_threadgroup_memory_length(0, (threads * 12 + 15 + 128) & !15);
    encoder.dispatch_groups((1, rows as usize, 1), (threads, 1, 1));
    Ok(result)
}

/// Rotate the last 64 channels of caller-owned `[rows, heads, width]` scratch.
/// No command-buffer boundary is introduced by this operation.
pub fn rope_in_place(
    x: &mut Array,
    start: u32,
    stride: u32,
    frequencies: &[f32; 32],
    inverse: bool,
    round_all: bool,
    stream: &Stream,
) -> Result<()> {
    if x.rank() != 3 || x.dim(0) < 1 || x.dim(1) < 1 || x.dim(2) < 64 || stride == 0 {
        return Err(Error::msg("invalid DeepSeek RoPE geometry"));
    }
    let rows = x.dim(0) as u32;
    let heads = x.dim(1) as u32;
    let width = x.dim(2) as u32;
    if start as u64 + (rows - 1) as u64 * stride as u64 >= 1048576 {
        return Err(Error::msg("DeepSeek RoPE exceeds context capacity"));
    }
    f32_input(x, &[rows as i32, heads as i32, width as i32])?;
    #[repr(C)]
    struct Args {
        width: u32,
        heads: u32,
        rows: u32,
        start: u32,
        inverse: u32,
        stride: u32,
        frequencies: [f32; 32],
        round_all: u32,
    }
    let args = Args {
        width,
        heads,
        rows,
        start,
        inverse: inverse as u32,
        stride,
        frequencies: *frequencies,
        round_all: round_all as u32,
    };
    let runtime = stream.runtime();
    let pipeline = runtime.compile_with(SOURCE, "kernel_dsv41_rope", Math::CompilerDefault)?;
    let guard = runtime.commands.encoder()?;
    let encoder = guard.encoder();
    encoder.set_pipeline(&pipeline);
    encoder.set_bytes(0, &args);
    let (buffer, layout) = x.t.buffer_and_layout();
    encoder.set_output(1, Some(buffer), layout.offset * 4);
    encoder.dispatch_groups((heads as usize, rows as usize, 1), (32, 1, 1));
    *x = Array::new(x.t.clone());
    Ok(())
}

/// DS4's Engram gate, including the unmasked width-5120, rows <=16 split
/// reduction gate. Both kernels retain each lane's original accumulation order.
pub fn engram_in_place(
    residual: &mut Array,
    kv: &Array,
    q_weight: &Array,
    k_weight: &Array,
    mask: Option<&Array>,
    eps: f32,
    stream: &Stream,
) -> Result<()> {
    if residual.rank() != 3
        || residual.dim(0) < 1
        || residual.dim(1) != 4
        || residual.dim(2) < 1
        || !eps.is_finite()
        || eps <= 0.0
    {
        return Err(Error::msg("invalid DeepSeek Engram geometry"));
    }
    let rows = residual.dim(0);
    let width = residual.dim(2);
    f32_input(residual, &[rows, 4, width])?;
    f32_input(kv, &[rows, 5, width])?;
    f32_input(q_weight, &[4, width])?;
    f32_input(k_weight, &[4, width])?;
    if let Some(mask) = mask {
        if mask.shape() != [rows]
            || mask.dtype() != Dtype::Uint8
            || !mask.t.buffer_and_layout().1.is_contiguous()
        {
            return Err(Error::msg("invalid DeepSeek Engram mask"));
        }
        mask.eval_enqueue()?;
    }
    #[repr(C)]
    struct Args {
        width: u32,
        rows: u32,
        eps: f32,
        masked: u32,
    }
    let args = Args {
        width: width as u32,
        rows: rows as u32,
        eps,
        masked: mask.is_some() as u32,
    };
    let runtime = stream.runtime();
    let split = mask.is_none() && width == 5120 && rows <= 16;
    let pipeline = runtime.compile_with(
        SOURCE,
        if split {
            "kernel_dsv41_engram_add_tg256"
        } else {
            "kernel_dsv41_engram_add"
        },
        Math::CompilerDefault,
    )?;
    let guard = runtime.commands.encoder()?;
    let encoder = guard.encoder();
    encoder.set_pipeline(&pipeline);
    encoder.set_bytes(0, &args);
    let (buffer, layout) = residual.t.buffer_and_layout();
    encoder.set_output(1, Some(buffer), layout.offset * 4);
    for (slot, input) in [
        (2, kv),
        (3, q_weight),
        (4, k_weight),
        (5, mask.unwrap_or(residual)),
    ] {
        let (buffer, layout) = input.t.buffer_and_layout();
        encoder.set_input(slot, Some(buffer), layout.offset * input.dtype().size_of());
    }
    if split {
        encoder.set_threadgroup_memory_length(0, 128 * 4);
    }
    encoder.dispatch_groups((rows as usize, 4, 1), (if split { 256 } else { 32 }, 1, 1));
    *residual = Array::new(residual.t.clone());
    Ok(())
}

// Test data captured from the pre-extraction DS4 host coefficient routine.
// Device production code receives coefficients from the caller.
#[cfg(test)]
fn fixture_frequencies(compressed: bool) -> [f32; 32] {
    const BITS: [[u32; 32]; 2] = [
        [
            1065353216, 1061157137, 1058010522, 1054337179, 1050798235, 1047712803, 1043732615,
            1040747895, 1036831949, 1033475085, 1030116802, 1026341397, 1023510243, 1019364175,
            1016180025, 1012562930, 1008981770, 1005959600, 1001931932, 998911607, 995049016,
            991652109, 988353832, 984533390, 981668463, 977573004, 974350870, 970790693, 967166816,
            964208661, 960132947, 957076594,
        ],
        [
            1065353216, 1060112954, 1056054303, 1051098369, 1046804782, 1042117749, 1037601972,
            1033169251, 1028443341, 1024251131, 1019326489, 1015361742, 1010249151, 1006366088,
            1001209182, 997081655, 991197921, 985076960, 979287478, 973418702, 966505391,
            959469944, 952012877, 943627103, 933484300, 917783504, 913828774, 908753896, 904558591,
            899759068, 895336240, 890797133,
        ],
    ];
    BITS[usize::from(compressed)].map(f32::from_bits)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn indirect_experts_match_dense_reductions_bitwise() {
        let stream = Stream::gpu();
        // Non-dyadic activations exercise rounding and cancellation, not merely
        // exactly representable sums. Include duplicate, reordered and invalid ids.
        let ids_data = [3, 0, 2, 3, -1, 4, 1, 3, 0, 2, 1, 0];
        let mut seed = 0x1783a9d5u32;
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            seed
        };
        for (format, columns, rows) in [
            (None, 32, 16),
            (None, 256, 16),
            (None, 384, 768),
            (None, 512, 16),
            (None, 544, 16),
            (None, 1280, 16),
            (Some(BlockFormat::Q5K), 768, 384),
            (Some(BlockFormat::Q6K), 768, 384),
        ] {
            let (block, block_bytes, scale_offset) = match format {
                None => (32, 34, 0),
                Some(BlockFormat::Q5K) => (256, 176, 0),
                Some(BlockFormat::Q6K) => (256, 210, 208),
                _ => unreachable!(),
            };
            let row_bytes = columns / block * block_bytes;
            let mut bytes = vec![0u8; 2 + 4 * rows * row_bytes];
            for b in &mut bytes {
                *b = next() as u8;
            }
            for b in bytes[2..].chunks_exact_mut(block_bytes) {
                let scale = half::f16::from_f32((next() % 127 + 1) as f32 / 3971.);
                b[scale_offset..scale_offset + 2].copy_from_slice(&scale.to_bits().to_le_bytes());
                if matches!(format, Some(BlockFormat::Q5K)) {
                    b[2..4].copy_from_slice(&half::f16::from_f32(0.0137).to_bits().to_le_bytes());
                }
            }
            let storage = Array::from_slice(&bytes, &[bytes.len() as i32]);
            let (buffer, _) = storage.t.buffer_and_layout();
            let weights = Array::from_buffer_offset(
                buffer.clone(),
                2,
                &[4, rows as i32, row_bytes as i32],
                Dtype::Uint8,
            )
            .unwrap();
            for tokens in [1, 2] {
                let ids = Array::from_slice(&ids_data[..tokens * 6], &[tokens as i32, 6]);
                for slots in [1, 6] {
                    let values = (0..tokens * slots * columns)
                        .map(|_| (next() as i32 as f32) / 765432109.)
                        .collect::<Vec<_>>();
                    let input =
                        Array::from_slice(&values, &[tokens as i32, slots as i32, columns as i32]);
                    let actual = match format {
                        None => q8_expert_matvec(&weights, &input, &ids, &stream),
                        Some(f) => expert_matvec(f, &weights, &input, &ids, &stream),
                    }
                    .unwrap();
                    let got = actual.as_slice::<f32>();
                    for token in 0..tokens {
                        for slot in 0..6 {
                            let expert = ids_data[token * 6 + slot];
                            let range = (token * 6 + slot) * rows..(token * 6 + slot + 1) * rows;
                            if !(0..4).contains(&expert) {
                                assert!(got[range].iter().all(|v| v.to_bits() == 0));
                                continue;
                            }
                            let w = Array::from_buffer_offset(
                                buffer.clone(),
                                2 + expert as usize * rows * row_bytes,
                                &[rows as i32, row_bytes as i32],
                                Dtype::Uint8,
                            )
                            .unwrap();
                            let start =
                                (token * slots + if slots == 1 { 0 } else { slot }) * columns;
                            let x = Array::from_slice(
                                &values[start..start + columns],
                                &[1, columns as i32],
                            );
                            let expected = match format {
                                None => q8_matvec(&w, &x, false, false, &stream),
                                Some(f) => packed_matvec(f, &w, &x, &stream),
                            }
                            .unwrap();
                            assert!(
                                got[range]
                                    .iter()
                                    .map(|v| v.to_bits())
                                    .eq(expected.as_slice::<f32>().iter().map(|v| v.to_bits())),
                                "format={format:?} tokens={tokens} slots={slots} token={token} slot={slot}"
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn deepseek41_engram_split_matches_original_lane_chains() {
        let stream = Stream::gpu();
        for rows in [1, 3, 16] {
            let values = |count: usize, stride: usize| {
                (0..count)
                    .map(|i| ((i * stride % 997) as f32 - 499.0) / 73.0)
                    .collect::<Vec<_>>()
            };
            let h = values(rows * 4 * 5120, 113);
            let mut split = Array::from_slice(&h, &[rows as i32, 4, 5120]);
            let mut original = Array::from_slice(&h, &[rows as i32, 4, 5120]);
            let kv = Array::from_slice(&values(rows * 5 * 5120, 19), &[rows as i32, 5, 5120]);
            let q = Array::from_slice(&values(4 * 5120, 23), &[4, 5120]);
            let k = Array::from_slice(&values(4 * 5120, 37), &[4, 5120]);
            // An all-one mask selects the original 32-lane implementation.
            let mask = Array::from_slice(&vec![1u8; rows], &[rows as i32]);
            engram_in_place(&mut original, &kv, &q, &k, Some(&mask), 1e-20, &stream).unwrap();
            engram_in_place(&mut split, &kv, &q, &k, None, 1e-20, &stream).unwrap();
            assert!(
                split
                    .as_slice::<f32>()
                    .iter()
                    .map(|v| v.to_bits())
                    .eq(original.as_slice::<f32>().iter().map(|v| v.to_bits())),
                "rows={rows}"
            );
        }
    }

    #[test]
    fn deepseek41_q8_native_packed_blocks_and_offsets() {
        let stream = Stream::gpu();
        assert_eq!(std::mem::size_of::<MatvecArgs>(), 112);
        for (columns, rows) in [
            (32, 2),
            (1280, 128),
            (5120, 16),
            (2304, 8),
            (8192, 8),
            (32, 65538),
        ] {
            let row_bytes = columns / 32 * 34;
            let mut bytes = vec![0u8; rows * row_bytes + 2];
            let input: Vec<f32> = (0..columns).map(|i| (i % 13) as f32 / 8.0 - 0.75).collect();
            let mut expected = vec![0f32; rows];
            for row in 0..rows {
                for block in 0..columns / 32 {
                    let start = 2 + row * row_bytes + block * 34;
                    let scale = half::f16::from_f32(if block % 2 == 0 { 0.125 } else { 0.25 });
                    bytes[start..start + 2].copy_from_slice(&scale.to_bits().to_le_bytes());
                    for lane in 0..32 {
                        let code = ((row * 11 + block * 7 + lane * 3) % 15) as i8 - 7;
                        bytes[start + 2 + lane] = code as u8;
                        expected[row] += scale.to_f32() * code as f32 * input[block * 32 + lane];
                    }
                }
            }
            let storage = Array::from_slice(&bytes, &[bytes.len() as i32]);
            let (buffer, _) = storage.t.buffer_and_layout();
            // GGUF tensors need only half alignment, not uint/float alignment.
            let weights = Array::from_buffer_offset(
                buffer.clone(),
                2,
                &[rows as i32, row_bytes as i32],
                Dtype::Uint8,
            )
            .unwrap();
            for tokens in [1, 3, 8] {
                let inputs: Vec<f32> = (0..tokens)
                    .flat_map(|t| input.iter().map(move |x| x * (t + 1) as f32))
                    .collect();
                let x = Array::from_slice(&inputs, &[tokens as i32, columns as i32]);
                for (round_out, round_in) in [(false, false), (true, false), (true, true)] {
                    let actual = q8_matvec(&weights, &x, round_out, round_in, &stream).unwrap();
                    actual.eval().unwrap();
                    let got = actual.as_slice::<f32>();
                    for token in 0..tokens {
                        for (i, &expected) in expected.iter().enumerate() {
                            let expected = expected * (token + 1) as f32;
                            let want = if round_out {
                                half::bf16::from_f32(expected).to_f32()
                            } else {
                                expected
                            };
                            assert_eq!(
                                got[token * rows + i],
                                want,
                                "Q8 K={columns} N={rows} row={i} token={token} round={round_out}/{round_in}"
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn deepseek41_router_pruned_and_full_expert_counts() {
        let stream = Stream::gpu();
        for experts in [192, 384] {
            let logits = Array::from_slice(&vec![0f32; 3 * experts], &[3, experts as i32]);
            let bias = Array::from_slice(
                &(0..experts).map(|i| i as f32).collect::<Vec<_>>(),
                &[experts as i32],
            );
            let out = router(&logits, &bias, 6, 1.5, &stream).unwrap();
            out.weights.eval().unwrap();
            for row in 0..3 {
                for col in 0..6 {
                    assert_eq!(
                        out.selected.as_slice::<i32>()[row * 6 + col],
                        (experts - 1 - col) as i32
                    );
                    assert!((out.weights.as_slice::<f32>()[row * 6 + col] - 0.25).abs() < 1e-6);
                }
            }
        }
    }

    #[test]
    fn deepseek41_rope_zero_position_and_host_readback() {
        let stream = Stream::gpu();
        let values: Vec<f32> = (0..2 * 512).map(|i| i as f32 * 0.03127).collect();
        for round_all in [false, true] {
            let mut x = Array::from_slice(&values, &[1, 2, 512]);
            assert_eq!(x.as_slice::<f32>(), values);
            rope_in_place(
                &mut x,
                0,
                1,
                &fixture_frequencies(false),
                false,
                round_all,
                &stream,
            )
            .unwrap();
            x.eval().unwrap();
            for (i, &got) in x.as_slice::<f32>().iter().enumerate() {
                let want = if round_all || i % 512 >= 448 {
                    half::bf16::from_f32(values[i]).to_f32()
                } else {
                    values[i]
                };
                assert_eq!(got, want);
            }
        }
    }

    #[test]
    fn deepseek41_engram_mask_and_bf16_boundary() {
        let stream = Stream::gpu();
        let width = 5120;
        let original: Vec<f32> = (0..2 * 4 * width)
            .map(|i| 1.001 + (i % 37) as f32 * 0.011)
            .collect();
        let mut residual = Array::from_slice(&original, &[2, 4, width as i32]);
        let kv = Array::from_slice(&vec![0f32; 2 * 5 * width], &[2, 5, width as i32]);
        let weight = Array::from_slice(&vec![1f32; 4 * width], &[4, width as i32]);
        let mask = Array::from_slice(&[0u8, 1], &[2]);
        assert_eq!(residual.as_slice::<f32>(), original);
        engram_in_place(
            &mut residual,
            &kv,
            &weight,
            &weight,
            Some(&mask),
            1e-20,
            &stream,
        )
        .unwrap();
        residual.eval().unwrap();
        for (i, &got) in residual.as_slice::<f32>().iter().enumerate() {
            let want = if i < 4 * width {
                original[i]
            } else {
                half::bf16::from_f32(original[i]).to_f32()
            };
            assert_eq!(got, want);
        }
    }
}
