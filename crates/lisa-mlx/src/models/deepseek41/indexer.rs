//! DS4's scalar index scorer and block-sort/merge selection.

use super::{f32_input, output};
use crate::error::{Error, Result};
use crate::runtime::Math;
use crate::{Array, Dtype, Stream};

const SOURCE: &crate::shader_pack::Source = crate::kernel_source!("deepseek41/indexer.metal");
const SORT: &crate::shader_pack::Source = crate::kernel_source!("deepseek41/topk.metal");

pub fn index_scores(q: &Array, weights: &Array, keys: &Array, stream: &Stream) -> Result<Array> {
    f32_input(q, &[32, 128])?;
    f32_input(weights, &[32])?;
    if keys.rank() != 2 || keys.dim(0) < 1 || keys.dim(0) > 1048576 {
        return Err(Error::msg("invalid DeepSeek index key count"));
    }
    let rows = keys.dim(0);
    let wide = rows >= 8192;
    let bf16 = keys.dtype() == Dtype::Bfloat16;
    if keys.shape() != [rows, 128]
        || !keys.t.buffer_and_layout().1.is_contiguous()
        || (bf16 && !wide)
        || (!bf16 && keys.dtype() != Dtype::Float32)
    {
        return Err(Error::msg("invalid DeepSeek index key storage"));
    }
    keys.eval_enqueue()?;
    #[repr(C)]
    struct Args {
        rows: u32,
        heads: u32,
        width: u32,
        half: u32,
        scale: f32,
    }
    let args = Args {
        rows: rows as u32,
        heads: 32,
        width: 128,
        half: 0,
        scale: 1.0 / 64.0,
    };
    let runtime = stream.runtime();
    let (function, threads, keys_per_group) = if !wide {
        ("kernel_glm_indexer_score_one_rows4", 128, 16)
    } else if !bf16 {
        ("kernel_glm_indexer_score_one_wide", 256, 256)
    } else if rows < 10240 {
        ("kernel_dsv41_indexer_score_rows_split4_bf16", 256, 128)
    } else if rows < 65536 {
        ("kernel_dsv41_indexer_score_rows_split2_bf16", 256, 256)
    } else if rows < 98304 {
        ("kernel_dsv41_indexer_score_bf16", 512, 1024)
    } else {
        ("kernel_dsv41_indexer_score_bf16", 128, 256)
    };
    let pipeline = runtime.compile_with(SOURCE, function, Math::CompilerDefault)?;
    if pipeline.max_total_threads_per_threadgroup() < threads {
        return Err(Error::msg("device cannot execute DS4 index score geometry"));
    }
    let out = output(stream, &[1, rows], Dtype::Float32)?;
    let guard = runtime.commands.encoder()?;
    let encoder = guard.encoder();
    encoder.set_pipeline(&pipeline);
    encoder.set_bytes(0, &args);
    for (slot, array) in [(1, q), (2, weights), (3, keys)] {
        let (buffer, layout) = array.t.buffer_and_layout();
        encoder.set_input(slot, Some(buffer), layout.offset * array.dtype().size_of());
    }
    let (buffer, layout) = out.t.buffer_and_layout();
    encoder.set_output(4, Some(buffer), layout.offset * 4);
    if wide {
        encoder.set_threadgroup_memory_length(0, (4096 + 32) * 4);
    }
    encoder.dispatch_groups(
        ((rows as usize).div_ceil(keys_per_group), 1, 1),
        if wide { (threads, 1, 1) } else { (32, 4, 1) },
    );
    Ok(out)
}

/// FP4 E8M0 keys are already exactly representable in BF16. Preserve DS4's
/// truncating store instead of introducing a new rounded cache conversion.
pub fn index_keys_bf16(keys: &Array, stream: &Stream) -> Result<Array> {
    if keys.rank() != 2 || keys.dim(0) < 1 || keys.dim(0) > 1048576 || keys.dim(1) != 128 {
        return Err(Error::msg("invalid DeepSeek BF16 key shape"));
    }
    f32_input(keys, keys.shape())?;
    #[repr(C)]
    struct Args {
        vectors: u32,
        check: u32,
    }
    let args = Args {
        vectors: keys.size() as u32 / 4,
        check: 0,
    };
    let runtime = stream.runtime();
    let pipeline = runtime.compile_with(
        SOURCE,
        "kernel_dsv41_index_keys_bf16",
        Math::CompilerDefault,
    )?;
    let out = output(stream, keys.shape(), Dtype::Bfloat16)?;
    let guard = runtime.commands.encoder()?;
    let encoder = guard.encoder();
    encoder.set_pipeline(&pipeline);
    encoder.set_bytes(0, &args);
    let (buffer, layout) = keys.t.buffer_and_layout();
    encoder.set_input(1, Some(buffer), layout.offset * 4);
    encoder.set_input(3, Some(buffer), layout.offset * 4); // check=0 never reads the flag.
    let (buffer, layout) = out.t.buffer_and_layout();
    encoder.set_output(2, Some(buffer), layout.offset * 2);
    encoder.dispatch_groups(((args.vectors as usize).div_ceil(256), 1, 1), (256, 1, 1));
    Ok(out)
}

/// One completed compressor pair. The first row lives in persistent state;
/// neither state row is mutated here, allowing transactional token rollback.
pub fn pool_pair(
    previous_kv: &Array,
    previous_score: &Array,
    kv: &Array,
    score: &Array,
    stream: &Stream,
) -> Result<Array> {
    for array in [previous_kv, previous_score, kv, score] {
        f32_input(array, &[1, 512])?;
    }
    #[repr(C)]
    struct Args {
        width: u32,
        pairs: u32,
        tail: u32,
    }
    let args = Args {
        width: 512,
        pairs: 1,
        tail: 1,
    };
    let runtime = stream.runtime();
    let pipeline = runtime.compile_with(SOURCE, "kernel_dsv41_pool2", Math::CompilerDefault)?;
    let out = output(stream, &[1, 512], Dtype::Float32)?;
    let guard = runtime.commands.encoder()?;
    let encoder = guard.encoder();
    encoder.set_pipeline(&pipeline);
    encoder.set_bytes(0, &args);
    for (slot, array) in [(2, kv), (3, score), (4, previous_kv), (5, previous_score)] {
        let (buffer, layout) = array.t.buffer_and_layout();
        encoder.set_input(slot, Some(buffer), layout.offset * 4);
    }
    let (buffer, layout) = out.t.buffer_and_layout();
    encoder.set_output(1, Some(buffer), layout.offset * 4);
    encoder.dispatch_groups((2, 1, 1), (256, 1, 1));
    Ok(out)
}

#[repr(C)]
struct SortArgs {
    ne00: i32,
    ne01: i32,
    ne02: i32,
    ne03: i32,
    nb00: u64,
    nb01: u64,
    nb02: u64,
    nb03: u64,
    ne0: i32,
    ne1: i32,
    ne2: i32,
    ne3: i32,
    top_k: i32,
    causal_start: u32,
    causal_ratio: u32,
}
#[repr(C)]
struct MergeArgs {
    ne00: i64,
    ne01: i64,
    ne02: i64,
    ne03: i64,
    nb00: u64,
    nb01: u64,
    nb02: u64,
    nb03: u64,
    ne0: i32,
    ne1: i32,
    ne2: i32,
    ne3: i32,
    top_k: i32,
    len: i32,
    causal_start: u32,
    causal_ratio: u32,
    block_width: u32,
    block_top_k: u32,
}

/// DS4's shape gates: descending block sort below 16384 columns, radix above.
/// Radix uses score-descending/id-ascending ties; short rows retain the
/// non-stable bitonic order. The fused final merge emulates the short ladder.
pub fn index_topk(scores: &Array, top_k: i32, stream: &Stream) -> Result<Array> {
    if scores.rank() != 2
        || scores.dim(0) < 1
        || scores.dim(0) > 8192
        || scores.dim(1) < 1
        || scores.dim(1) > 1048576
        || top_k < 1
        || top_k > scores.dim(1)
        || top_k > 2048
    {
        return Err(Error::msg("invalid DeepSeek top-k shape"));
    }
    f32_input(scores, scores.shape())?;
    let (tokens, columns) = (scores.dim(0), scores.dim(1));
    if columns >= 16384 {
        return radix_topk(scores, top_k, None, stream);
    }
    let runtime = stream.runtime();
    let pipeline =
        runtime.compile_with(SORT, "kernel_argsort_f32_i32_desc", Math::CompilerDefault)?;
    let mut threads = 1usize;
    while threads < columns as usize && threads * 2 <= pipeline.max_total_threads_per_threadgroup()
    {
        threads *= 2;
    }
    let blocks = (columns as usize).div_ceil(threads);
    let block_top = (top_k as usize).min(threads);
    let width = if blocks == 1 {
        top_k as usize
    } else {
        (blocks - 1) * block_top + (columns as usize - (blocks - 1) * threads).min(block_top)
    };
    let row_bytes = columns as u64 * 4;
    let args = SortArgs {
        ne00: columns,
        ne01: tokens,
        ne02: 1,
        ne03: 1,
        nb00: 4,
        nb01: row_bytes,
        nb02: row_bytes * tokens as u64,
        nb03: row_bytes * tokens as u64,
        ne0: width as i32,
        ne1: tokens,
        ne2: 1,
        ne3: 1,
        top_k: block_top as i32,
        causal_start: 0,
        causal_ratio: 0,
    };
    let mut partial = output(stream, &[tokens, width as i32], Dtype::Int32)?;
    {
        let guard = runtime.commands.encoder()?;
        let encoder = guard.encoder();
        encoder.set_pipeline(&pipeline);
        encoder.set_bytes(0, &args);
        let (buffer, layout) = scores.t.buffer_and_layout();
        encoder.set_input(1, Some(buffer), layout.offset * 4);
        let (buffer, layout) = partial.t.buffer_and_layout();
        encoder.set_output(2, Some(buffer), layout.offset * 4);
        encoder.set_threadgroup_memory_length(0, (threads * 8 + 15) & !15);
        encoder.dispatch_groups((blocks * tokens as usize, 1, 1), (threads, 1, 1));
    }
    let staged = top_k == 512 && tokens == 1;
    let merge = runtime.compile_with(
        SORT,
        if staged {
            "kernel_argsort_merge_f32_i32_desc_prefix_staged"
        } else {
            "kernel_argsort_merge_f32_i32_desc"
        },
        Math::CompilerDefault,
    )?;
    let mut len = block_top;
    while len < width {
        let runs = width.div_ceil(len);
        if staged && threads == 1024 && block_top == 512 && runs <= 10 {
            let fused = runtime.compile_with(
                SORT,
                "kernel_argsort_merge_f32_i32_desc_prefix_fused",
                Math::CompilerDefault,
            )?;
            if fused.max_total_threads_per_threadgroup() >= 1024 {
                let out = output(stream, &[1, top_k], Dtype::Int32)?;
                let args = MergeArgs {
                    ne00: columns as i64,
                    ne01: 1,
                    ne02: 1,
                    ne03: 1,
                    nb00: 4,
                    nb01: row_bytes,
                    nb02: row_bytes,
                    nb03: row_bytes,
                    ne0: width as i32,
                    ne1: 1,
                    ne2: 1,
                    ne3: 1,
                    top_k,
                    len: len as i32,
                    causal_start: 0,
                    causal_ratio: 0,
                    block_width: 1024,
                    block_top_k: 512,
                };
                let guard = runtime.commands.encoder()?;
                let encoder = guard.encoder();
                encoder.set_pipeline(&fused);
                encoder.set_bytes(0, &args);
                encoder.set_bytes(1, &512u32);
                for (slot, array) in [(2, scores), (3, &partial)] {
                    let (buffer, layout) = array.t.buffer_and_layout();
                    encoder.set_input(slot, Some(buffer), layout.offset * 4);
                }
                let (buffer, layout) = out.t.buffer_and_layout();
                encoder.set_output(4, Some(buffer), layout.offset * 4);
                encoder.set_threadgroup_memory_length(0, (runs * 512 * 6 + 15) & !15);
                encoder.dispatch_groups((1, 1, 1), (1024, 1, 1));
                return Ok(out);
            }
        }
        let groups = width.div_ceil(2 * len);
        let out_width = if groups == 1 { top_k as usize } else { width };
        let next = output(stream, &[tokens, out_width as i32], Dtype::Int32)?;
        let args = MergeArgs {
            ne00: columns as i64,
            ne01: tokens as i64,
            ne02: 1,
            ne03: 1,
            nb00: 4,
            nb01: row_bytes,
            nb02: row_bytes * tokens as u64,
            nb03: row_bytes * tokens as u64,
            ne0: width as i32,
            ne1: tokens,
            ne2: 1,
            ne3: 1,
            top_k: out_width as i32,
            len: len as i32,
            causal_start: 0,
            causal_ratio: 0,
            block_width: threads as u32,
            block_top_k: block_top as u32,
        };
        {
            let guard = runtime.commands.encoder()?;
            let encoder = guard.encoder();
            encoder.set_pipeline(&merge);
            encoder.set_bytes(0, &args);
            for (slot, array) in [(1, scores), (2, &partial)] {
                let (buffer, layout) = array.t.buffer_and_layout();
                encoder.set_input(slot, Some(buffer), layout.offset * 4);
            }
            let (buffer, layout) = next.t.buffer_and_layout();
            encoder.set_output(3, Some(buffer), layout.offset * 4);
            if staged {
                encoder.set_threadgroup_memory_length(0, 8192);
            }
            encoder.dispatch_groups(
                (groups * tokens as usize, 1, 1),
                (
                    len.min(512).min(merge.max_total_threads_per_threadgroup()),
                    1,
                    1,
                ),
            );
        }
        partial = next;
        len *= 2;
    }
    Ok(partial)
}

#[repr(C)]
struct RadixArgs {
    n: u32,
    rows: u32,
    top_k: u32,
    pass: u32,
    parity: u32,
    n_tg: u32,
}

fn radix_topk(
    scores: &Array,
    top_k: i32,
    candidate_map: Option<&Array>,
    stream: &Stream,
) -> Result<Array> {
    let (rows, n) = (scores.dim(0), scores.dim(1));
    let groups = (n as usize).div_ceil(2048);
    let runtime = stream.runtime();
    let pipeline = |name| runtime.compile_with(SORT, name, Math::CompilerDefault);
    let hist_pipeline = pipeline("kernel_topk_select_hist")?;
    let count_pipeline = pipeline("kernel_topk_select_count")?;
    let write_pipeline = pipeline("kernel_topk_select_write")?;
    let sort_pipeline = pipeline(if top_k <= 512 {
        "kernel_topk_select_sort512"
    } else {
        "kernel_topk_select_sort"
    })?;
    if [&hist_pipeline, &count_pipeline, &write_pipeline]
        .iter()
        .any(|p| p.max_total_threads_per_threadgroup() < 1024)
        || sort_pipeline.max_total_threads_per_threadgroup() < if top_k <= 512 { 512 } else { 1024 }
    {
        return Err(Error::msg("device cannot execute DS4 radix selection"));
    }
    // Independent scratch gives each queued call its own histogram lifetime.
    // DS4's parity ping-pong is unnecessary until this scratch is reused.
    let hist = crate::ops::zeros::<u32>(&[rows, 4 * 2048])?;
    let state = output(stream, &[rows, 4], Dtype::Uint32)?;
    let counts = output(stream, &[rows, groups as i32, 2], Dtype::Uint32)?;
    let out = output(stream, &[rows, top_k], Dtype::Int32)?;
    let mut args = RadixArgs {
        n: n as u32,
        rows: rows as u32,
        top_k: top_k as u32,
        pass: 0,
        parity: 0,
        n_tg: groups as u32,
    };
    hist.eval_enqueue()?;
    let bind_input = |encoder: &crate::runtime::ComputeEncoder, slot, array: &Array| {
        let (buffer, layout) = array.t.buffer_and_layout();
        encoder.set_input(slot, Some(buffer), layout.offset * 4);
    };
    let bind_output = |encoder: &crate::runtime::ComputeEncoder, slot, array: &Array| {
        let (buffer, layout) = array.t.buffer_and_layout();
        encoder.set_output(slot, Some(buffer), layout.offset * 4);
    };
    for pass in 0..3 {
        args.pass = pass;
        let guard = runtime.commands.encoder()?;
        let encoder = guard.encoder();
        encoder.set_pipeline(&hist_pipeline);
        encoder.set_bytes(0, &args);
        bind_input(encoder, 1, scores);
        bind_output(encoder, 2, &hist);
        encoder.set_threadgroup_memory_length(0, 2048 * 4);
        encoder.dispatch_groups((groups, rows as usize, 1), (1024, 1, 1));
    }
    args.pass = 3;
    {
        let guard = runtime.commands.encoder()?;
        let encoder = guard.encoder();
        encoder.set_pipeline(&count_pipeline);
        encoder.set_bytes(0, &args);
        bind_input(encoder, 1, scores);
        bind_input(encoder, 2, &hist);
        bind_output(encoder, 3, &state);
        bind_output(encoder, 4, &counts);
        encoder.dispatch_groups((groups, rows as usize, 1), (1024, 1, 1));
    }
    {
        let guard = runtime.commands.encoder()?;
        let encoder = guard.encoder();
        encoder.set_pipeline(&write_pipeline);
        encoder.set_bytes(0, &args);
        bind_input(encoder, 1, scores);
        bind_input(encoder, 2, &state);
        bind_input(encoder, 3, &counts);
        bind_output(encoder, 4, &out);
        encoder.dispatch_groups((groups, rows as usize, 1), (1024, 1, 1));
    }
    {
        let guard = runtime.commands.encoder()?;
        let encoder = guard.encoder();
        encoder.set_pipeline(&sort_pipeline);
        encoder.set_bytes(0, &args);
        bind_input(encoder, 1, scores);
        bind_output(encoder, 2, &out);
        if top_k <= 512 {
            bind_input(encoder, 3, candidate_map.unwrap_or(scores));
            encoder.set_bytes(4, &u32::from(candidate_map.is_some()));
        } else {
            encoder.set_threadgroup_memory_length(0, 2 * 2048 * 4);
        }
        encoder.dispatch_groups(
            (rows as usize, 1, 1),
            (if top_k <= 512 { 512 } else { 1024 }, 1, 1),
        );
    }
    Ok(out)
}

/// Validated candidate list built from layer 20's score row. Keeping its
/// construction private prevents partially initialized arbitrary GPU lists.
pub struct IndexCandidates {
    rows: i32,
    list: Array,
}

pub fn index_candidates(scores: &Array, stream: &Stream) -> Result<IndexCandidates> {
    if scores.rank() != 2 || scores.dim(0) != 1 || scores.dim(1) <= 16384 || scores.dim(1) > 1048576
    {
        return Err(Error::msg(
            "candidate selection requires one wide DeepSeek score row",
        ));
    }
    f32_input(scores, scores.shape())?;
    let rows = scores.dim(1);
    let blocks = (rows as u32).div_ceil(8);
    #[repr(C)]
    struct BlockArgs {
        width: u32,
        rows: u32,
        start: u32,
        ratio: u32,
        mask_stride: u32,
    }
    let args = BlockArgs {
        width: rows as u32,
        rows: 1,
        start: rows as u32 - 1,
        ratio: 1,
        mask_stride: blocks,
    };
    let runtime = stream.runtime();
    let block_scores = output(stream, &[1, blocks as i32], Dtype::Float32)?;
    {
        let pipeline = runtime.compile_with(
            SOURCE,
            "kernel_dsv41_candidate_blocks",
            Math::CompilerDefault,
        )?;
        let guard = runtime.commands.encoder()?;
        let encoder = guard.encoder();
        encoder.set_pipeline(&pipeline);
        encoder.set_bytes(0, &args);
        let (buffer, layout) = scores.t.buffer_and_layout();
        encoder.set_input(1, Some(buffer), layout.offset * 4);
        encoder.set_input(3, Some(buffer), layout.offset * 4);
        let (buffer, layout) = block_scores.t.buffer_and_layout();
        encoder.set_output(2, Some(buffer), layout.offset * 4);
        encoder.dispatch_groups(((blocks as usize).div_ceil(256), 1, 1), (256, 1, 1));
    }
    let selected = index_topk(&block_scores, 2048, stream)?;
    let mask = output(stream, &[blocks as i32], Dtype::Float32)?;
    #[repr(C)]
    struct MaskArgs {
        ne00: i64,
        ne01: i64,
        nb00: u64,
        nb01: u64,
        ne0: i64,
        ne1: i64,
        nb0: u64,
        nb1: u64,
    }
    let args = MaskArgs {
        ne00: 2048,
        ne01: 1,
        nb00: 4,
        nb01: 2048 * 4,
        ne0: blocks as i64,
        ne1: 1,
        nb0: 4,
        nb1: blocks as u64 * 4,
    };
    for (function, count) in [
        ("kernel_dsv4_topk_mask", blocks),
        ("kernel_dsv4_topk_mask_scatter", 2048),
    ] {
        let pipeline = runtime.compile_with(SOURCE, function, Math::CompilerDefault)?;
        let guard = runtime.commands.encoder()?;
        let encoder = guard.encoder();
        encoder.set_pipeline(&pipeline);
        encoder.set_bytes(0, &args);
        let (buffer, layout) = selected.t.buffer_and_layout();
        encoder.set_input(1, Some(buffer), layout.offset * 4);
        let (buffer, layout) = mask.t.buffer_and_layout();
        encoder.set_output(2, Some(buffer), layout.offset * 4);
        encoder.dispatch_groups(((count as usize).div_ceil(256), 1, 1), (256, 1, 1));
    }
    let list = output(stream, &[2048], Dtype::Uint32)?;
    {
        let pipeline =
            runtime.compile_with(SOURCE, "kernel_dsv41_candidate_list", Math::CompilerDefault)?;
        let guard = runtime.commands.encoder()?;
        let encoder = guard.encoder();
        encoder.set_pipeline(&pipeline);
        encoder.set_bytes(0, &blocks);
        let (buffer, layout) = mask.t.buffer_and_layout();
        encoder.set_input(1, Some(buffer), layout.offset * 4);
        let (buffer, layout) = list.t.buffer_and_layout();
        encoder.set_output(2, Some(buffer), layout.offset * 4);
        encoder.dispatch_groups((1, 1, 1), (1024, 1, 1));
    }
    Ok(IndexCandidates { rows, list })
}

/// Re-score only the 2048 candidate blocks, and map winners back to cache IDs
/// in the radix winners' sort. The candidate list is reused across layers.
pub fn index_candidates_select(
    q: &Array,
    weights: &Array,
    keys: &Array,
    candidates: &IndexCandidates,
    stream: &Stream,
) -> Result<Array> {
    f32_input(q, &[32, 128])?;
    f32_input(weights, &[32])?;
    f32_input(keys, &[candidates.rows, 128])?;
    let count = 16384 - ((candidates.rows as u32).div_ceil(8) * 8 - candidates.rows as u32);
    #[repr(C)]
    struct Args {
        rows: u32,
        heads: u32,
        width: u32,
        half: u32,
        scale: f32,
    }
    let args = Args {
        rows: count,
        heads: 32,
        width: 128,
        half: 0,
        scale: 1.0 / 64.0,
    };
    let runtime = stream.runtime();
    let scores = output(stream, &[1, count as i32], Dtype::Float32)?;
    {
        let pipeline = runtime.compile_with(
            SOURCE,
            "kernel_dsv41_indexer_score_candidates",
            Math::CompilerDefault,
        )?;
        let guard = runtime.commands.encoder()?;
        let encoder = guard.encoder();
        encoder.set_pipeline(&pipeline);
        encoder.set_bytes(0, &args);
        for (slot, array) in [(1, q), (2, weights), (3, keys), (5, &candidates.list)] {
            let (buffer, layout) = array.t.buffer_and_layout();
            encoder.set_input(slot, Some(buffer), layout.offset * 4);
        }
        let (buffer, layout) = scores.t.buffer_and_layout();
        encoder.set_output(4, Some(buffer), layout.offset * 4);
        encoder.set_threadgroup_memory_length(0, (4096 + 32) * 4);
        encoder.dispatch_groups(((count as usize).div_ceil(256), 1, 1), (256, 1, 1));
    }
    radix_topk(&scores, 512, Some(&candidates.list), stream)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Emulate the original comparison network, including its deliberately
    // non-stable ordering of equal scores. Sorting by (score, id) is a different
    // oracle: equal-score swaps in the descending/ascending bitonic stages
    // produce a different permutation even before the merge.
    fn reference_topk(scores: &[f32], top: usize, threads: usize) -> Vec<i32> {
        let mut runs = Vec::new();
        for first in (0..scores.len()).step_by(threads) {
            let mut ids: Vec<usize> = (first..first + threads).collect();
            let mut k = 2;
            while k <= threads {
                let mut j = k / 2;
                while j > 0 {
                    for col in 0..threads {
                        let other = col ^ j;
                        if other <= col {
                            continue;
                        }
                        let (a, b) = (ids[col], ids[other]);
                        let swap = if col & k == 0 {
                            a >= scores.len() || b < scores.len() && scores[a] < scores[b]
                        } else {
                            b >= scores.len() || a < scores.len() && scores[a] > scores[b]
                        };
                        if swap {
                            ids.swap(col, other);
                        }
                    }
                    j /= 2;
                }
                k *= 2;
            }
            ids.truncate(top.min(threads).min(scores.len() - first));
            runs.push(ids);
        }
        while runs.len() > 1 {
            runs = runs
                .chunks(2)
                .map(|pair| {
                    if pair.len() == 1 {
                        return pair[0].clone();
                    }
                    let (mut a, mut b) = (0, 0);
                    let mut merged = Vec::new();
                    while a < pair[0].len() || b < pair[1].len() {
                        if b == pair[1].len()
                            || a < pair[0].len() && scores[pair[0][a]] >= scores[pair[1][b]]
                        {
                            merged.push(pair[0][a]);
                            a += 1;
                        } else {
                            merged.push(pair[1][b]);
                            b += 1;
                        }
                    }
                    merged.truncate(top);
                    merged
                })
                .collect();
        }
        runs.remove(0)
            .into_iter()
            .take(top)
            .map(|x| x as i32)
            .collect()
    }

    #[test]
    fn deepseek41_indexer_selection_ties_and_partial_blocks() {
        let stream = Stream::gpu();
        for width in [
            1, 31, 511, 512, 513, 1023, 1024, 1025, 2049, 8193, 16383, 16384, 65537,
        ] {
            for tokens in [1, 3] {
                for top in [1, 512.min(width), 2048.min(width)] {
                    let scores: Vec<f32> = (0..tokens * width)
                        .map(|i| ((i * 71 + i / 11) % 97) as f32 - 48.0)
                        .collect();
                    let input = Array::from_slice(&scores, &[tokens, width]);
                    let out = index_topk(&input, top, &stream).unwrap();
                    for (row, actual) in
                        out.as_slice::<i32>().chunks_exact(top as usize).enumerate()
                    {
                        let source = &scores[row * width as usize..(row + 1) * width as usize];
                        let threads = (width as usize).next_power_of_two().min(1024);
                        let ids = if width < 16384 {
                            reference_topk(source, top as usize, threads)
                        } else {
                            let mut ids: Vec<i32> = (0..width).collect();
                            ids.sort_by(|&a, &b| {
                                source[b as usize]
                                    .total_cmp(&source[a as usize])
                                    .then(a.cmp(&b))
                            });
                            ids.truncate(top as usize);
                            ids
                        };
                        assert_eq!(actual, ids, "width={width} top={top} row={row}");
                    }
                }
            }
        }
    }

    #[test]
    fn deepseek41_index_scores_and_pool_pair() {
        let stream = Stream::gpu();
        let q: Vec<f32> = (0..4096)
            .map(|i| ((i * 3 % 17) as f32 - 8.0) / 8.0)
            .collect();
        let weights: Vec<f32> = (0..32).map(|i| (i % 11) as f32 / 8.0 - 0.5).collect();
        let keys: Vec<f32> = (0..513 * 128)
            .map(|i| ((i * 7 % 13) as f32 - 6.0) / 8.0)
            .collect();
        let out = index_scores(
            &Array::from_slice(&q, &[32, 128]),
            &Array::from_slice(&weights, &[32]),
            &Array::from_slice(&keys, &[513, 128]),
            &stream,
        )
        .unwrap();
        for (row, &actual) in out.as_slice::<f32>().iter().enumerate() {
            let mut expected = 0.0f32;
            for head in 0..32 {
                let dot: f32 = q[head * 128..(head + 1) * 128]
                    .iter()
                    .zip(&keys[row * 128..(row + 1) * 128])
                    .map(|(q, k)| q * k)
                    .sum();
                expected += (dot / 64.0).max(0.0) * weights[head];
            }
            assert_eq!(actual, expected, "index key {row}");
        }
        let a = Array::from_slice(&vec![1.25f32; 512], &[1, 512]);
        let b = Array::from_slice(&vec![-0.5f32; 512], &[1, 512]);
        let scores = Array::from_slice(&vec![0f32; 512], &[1, 512]);
        let out = pool_pair(&a, &scores, &b, &scores, &stream).unwrap();
        assert!(out.as_slice::<f32>().iter().all(|&x| x == 0.375));
    }

    #[test]
    fn deepseek41_wide_bf16_scores_match_original_f32_at_every_gate() {
        let stream = Stream::gpu();
        let q: Vec<f32> = (0..4096)
            .map(|i| ((i * 71 % 257) as f32 - 128.0) / 64.0)
            .collect();
        let weights: Vec<f32> = (0..32).map(|i| (i * 19 % 31) as f32 / 97.0 - 0.1).collect();
        let q = Array::from_slice(&q, &[32, 128]);
        let weights = Array::from_slice(&weights, &[32]);
        for rows in [8192, 8193, 10239, 10240, 65535, 65536, 98303, 98304, 131073] {
            let values: Vec<f32> = (0..rows * 128)
                .map(|i| ((i * 13 + i / 7) % 31) as f32 / 8.0 - 2.0)
                .collect();
            let keys = Array::from_slice(&values, &[rows, 128]);
            let packed = index_keys_bf16(&keys, &stream).unwrap();
            let original = index_scores(&q, &weights, &keys, &stream).unwrap();
            let actual = index_scores(&q, &weights, &packed, &stream).unwrap();
            for (i, (a, b)) in actual
                .as_slice::<f32>()
                .iter()
                .zip(original.as_slice::<f32>())
                .enumerate()
            {
                assert_eq!(a.to_bits(), b.to_bits(), "wide index rows={rows} key={i}");
            }
        }
    }

    #[test]
    fn deepseek41_candidates_keep_newest_block_and_map_radix_ids() {
        let stream = Stream::gpu();
        let q = Array::from_slice(
            &(0..4096)
                .map(|i| ((i * 11 % 17) as f32 - 8.0) / 8.0)
                .collect::<Vec<_>>(),
            &[32, 128],
        );
        let weights = Array::from_slice(
            &(0..32)
                .map(|i| (i % 11) as f32 / 16.0 - 0.25)
                .collect::<Vec<_>>(),
            &[32],
        );
        for rows in [16385, 32769, 131073] {
            let seed_scores: Vec<f32> =
                (0..rows).map(|i| ((i * 17 + i / 7) % 173) as f32).collect();
            let candidates =
                index_candidates(&Array::from_slice(&seed_scores, &[1, rows]), &stream).unwrap();
            let block_scores: Vec<f32> = seed_scores
                .chunks(8)
                .map(|row| row.iter().copied().fold(f32::NEG_INFINITY, f32::max))
                .collect();
            let mut block_scores = block_scores;
            *block_scores.last_mut().unwrap() = f32::INFINITY;
            let mut chosen = if block_scores.len() < 16384 {
                reference_topk(
                    &block_scores,
                    2048,
                    block_scores.len().next_power_of_two().min(1024),
                )
            } else {
                let mut ids: Vec<i32> = (0..block_scores.len() as i32).collect();
                ids.sort_by(|&a, &b| {
                    block_scores[b as usize]
                        .total_cmp(&block_scores[a as usize])
                        .then(a.cmp(&b))
                });
                ids.truncate(2048);
                ids
            };
            chosen.sort();
            assert_eq!(
                candidates.list.as_slice::<u32>(),
                chosen.iter().map(|&i| i as u32).collect::<Vec<_>>()
            );
            assert_eq!(chosen.last(), Some(&((rows - 1) / 8)));
            let keys: Vec<f32> = (0..rows * 128)
                .map(|i| ((i * 13 + i / 7) % 31) as f32 / 8.0 - 2.0)
                .collect();
            let keys = Array::from_slice(&keys, &[rows, 128]);
            let full = index_scores(&q, &weights, &keys, &stream).unwrap();
            let full = full.as_slice::<f32>();
            let mut expected: Vec<i32> = chosen
                .iter()
                .flat_map(|&b| b * 8..((b + 1) * 8).min(rows))
                .collect();
            expected.sort_by(|&a, &b| {
                full[b as usize]
                    .total_cmp(&full[a as usize])
                    .then(a.cmp(&b))
            });
            expected.truncate(512);
            for _ in 0..2 {
                let out =
                    index_candidates_select(&q, &weights, &keys, &candidates, &stream).unwrap();
                assert_eq!(out.as_slice::<i32>(), expected, "candidate rows={rows}");
            }
        }
    }
}
