//! DS4's F16 staging and 32-way split-K attention for V4.1 scalar decode.

use super::{f32_input, output};
use crate::error::{Error, Result};
use crate::runtime::{ConstVal, Math};
use crate::{Array, Dtype, Stream};

const SOURCE: &crate::shader_pack::Source = crate::kernel_source!("deepseek41/attention.metal");

/// Reused serial-queue scratch, independent of model layer and context counts.
/// Every writer/reader must be encoded on this model's one command stream.
pub struct DecodeAttentionScratch {
    staged: Array,
    mask: Array,
    pad: Array,
    partials: Array,
}

impl DecodeAttentionScratch {
    pub fn new(stream: &Stream) -> Result<Self> {
        Ok(Self {
            staged: output(stream, &[640, 512], Dtype::Float16)?,
            mask: Array::from_slice(&[0i16; 640], &[640]),
            pad: output(stream, &[2 * 32 * 512 + 32], Dtype::Float16)?,
            partials: output(stream, &[64 * (512 * 32 + 64)], Dtype::Float32)?,
        })
    }
}

#[repr(C)]
struct StageArgs {
    raw_cap: u32,
    raw_start: u32,
    n_raw: u32,
    n_comp: u32,
    pad_rows: u32,
    shared_pad: u32,
    cache_rows: u32,
}

#[repr(C)]
struct AttentionArgs {
    ne01: i32,
    ne02: i32,
    ne03: i32,
    nb01: u64,
    nb02: u64,
    nb03: u64,
    ne11: i32,
    ne_12_2: i32,
    ne_12_3: i32,
    ns10: i32,
    nb11: u64,
    nb12: u64,
    nb13: u64,
    ns20: i32,
    nb21: u64,
    nb22: u64,
    nb23: u64,
    ne31: i32,
    ne32: i32,
    ne33: i32,
    nb31: u64,
    nb32: u64,
    nb33: u64,
    ne1: i32,
    ne2: i32,
    ne3: i32,
    scale: f32,
    max_bias: f32,
    m0: f32,
    m1: f32,
    n_head_log2: i32,
    logit_softcap: f32,
}

/// Attend to the logical raw ring followed by GPU-selected compressed rows.
/// Uses the original DS4 F16 conversion, padding, vector attention and reduce.
/// Invalid compressed ids stage zeros without reading outside the cache.
pub fn attention_decode(
    q: &Array,
    sinks: &Array,
    raw: &Array,
    raw_count: u32,
    raw_start: u32,
    compressed: Option<(&Array, &Array)>,
    scratch: &mut DecodeAttentionScratch,
    stream: &Stream,
) -> Result<Array> {
    attention_decode_with_rope(
        q, sinks, raw, raw_count, raw_start, compressed, scratch, None, stream,
    )
}

/// Optional DS4 heads transform in the split-K reduction, under its original
/// 64-head / width-512 / 32-partition gate (the only shapes this API accepts).
pub fn attention_decode_with_rope(
    q: &Array,
    sinks: &Array,
    raw: &Array,
    raw_count: u32,
    raw_start: u32,
    compressed: Option<(&Array, &Array)>,
    scratch: &mut DecodeAttentionScratch,
    rope: Option<(&Array, bool)>,
    stream: &Stream,
) -> Result<Array> {
    if let Some((table, _)) = rope {
        f32_input(table, &[64, 2])?;
    }
    f32_input(q, &[64, 512])?;
    f32_input(sinks, &[64])?;
    f32_input(raw, &[128, 512])?;
    if raw_count == 0 || raw_count > 128 || raw_start >= 128 {
        return Err(Error::msg("invalid DeepSeek raw attention ring range"));
    }
    let (comp, ids, count, cache_rows) = if let Some((cache, ids)) = compressed {
        if cache.rank() != 2
            || cache.dim(0) < 1
            || ids.rank() != 1
            || ids.dim(0) < 1
            || ids.dim(0) > 512
            || ids.dim(0) > cache.dim(0)
            || ids.dtype() != Dtype::Int32
            || !ids.t.buffer_and_layout().1.is_contiguous()
        {
            return Err(Error::msg(
                "invalid DeepSeek compressed attention selection",
            ));
        }
        f32_input(cache, &[cache.dim(0), 512])?;
        ids.eval_enqueue()?;
        (cache, ids, ids.dim(0) as u32, cache.dim(0) as u32)
    } else {
        (raw, raw, 0, 0)
    };
    let keys = raw_count + count;
    let has_pad = keys % 32 != 0;
    // At most 128 raw + 512 selected keys: DS4's NSG selector is one at
    // these widths; its NWG remains 32 even for short contexts.
    let stage_args = StageArgs {
        raw_cap: 128,
        raw_start,
        n_raw: raw_count,
        n_comp: count,
        pad_rows: if has_pad { 32 } else { 0 },
        shared_pad: 0,
        cache_rows,
    };
    let runtime = stream.runtime();
    let stage_pipeline =
        runtime.compile_with(SOURCE, "kernel_dsv41_flash_kv_stage", Math::CompilerDefault)?;
    let vec_pipeline = runtime.compile_with_constants(
        SOURCE,
        "kernel_flash_attn_ext_vec_f16_dk512_dv512",
        Math::CompilerDefault,
        &[
            (400, ConstVal::Bool(true)),
            (401, ConstVal::Bool(true)),
            (402, ConstVal::Bool(false)),
            (403, ConstVal::Bool(false)),
            (404, ConstVal::Bool(has_pad)),
            (405, ConstVal::Bool(false)),
            (420, ConstVal::Int(512)),
            (421, ConstVal::Int(512)),
            (422, ConstVal::Int(1)),
            (423, ConstVal::Int(32)),
        ],
    )?;
    let reduce_pipeline = runtime.compile_with_constants(
        SOURCE,
        if rope.is_some() {
            "kernel_dsv41_flash_attn_vec_reduce_heads"
        } else {
            "kernel_flash_attn_ext_vec_reduce"
        },
        Math::CompilerDefault,
        &[(500, ConstVal::Int(512)), (501, ConstVal::Int(32))],
    )?;
    if reduce_pipeline.max_total_threads_per_threadgroup() < 1024 {
        return Err(Error::msg(
            "device cannot execute DS4 split-K attention reduction",
        ));
    }
    let args = AttentionArgs {
        ne01: 1,
        ne02: 64,
        ne03: 1,
        nb01: 64 * 2048,
        nb02: 2048,
        nb03: 64 * 2048,
        ne11: keys as i32,
        ne_12_2: 1,
        ne_12_3: 1,
        ns10: 512,
        nb11: 1024,
        nb12: keys as u64 * 1024,
        nb13: keys as u64 * 1024,
        ns20: 512,
        nb21: 1024,
        nb22: keys as u64 * 1024,
        nb23: keys as u64 * 1024,
        ne31: 1,
        ne32: 1,
        ne33: 1,
        nb31: keys as u64 * 2,
        nb32: keys as u64 * 2,
        nb33: keys as u64 * 2,
        ne1: 64,
        ne2: 1,
        ne3: 1,
        scale: 1.0 / 512f32.sqrt(),
        max_bias: 0.0,
        m0: 0.0,
        m1: 0.0,
        n_head_log2: 0,
        logit_softcap: 0.0,
    };
    let out = output(stream, &[64, 512], Dtype::Float32)?;
    {
        let guard = runtime.commands.encoder()?;
        let encoder = guard.encoder();
        encoder.set_pipeline(&stage_pipeline);
        encoder.set_bytes(0, &stage_args);
        for (slot, array) in [(1, raw), (2, comp), (4, &scratch.mask), (6, ids)] {
            let (buffer, layout) = array.t.buffer_and_layout();
            encoder.set_input(slot, Some(buffer), layout.offset * array.dtype().size_of());
        }
        for (slot, array) in [(3, &scratch.staged), (5, &scratch.pad)] {
            let (buffer, layout) = array.t.buffer_and_layout();
            encoder.set_output(slot, Some(buffer), layout.offset * 2);
        }
        let work = keys as usize * 128 + if has_pad { 32 * 128 + 32 } else { 0 };
        encoder.dispatch_groups((work.div_ceil(256), 1, 1), (256, 1, 1));
    }
    {
        let guard = runtime.commands.encoder()?;
        let encoder = guard.encoder();
        encoder.set_pipeline(&vec_pipeline);
        encoder.set_bytes(0, &args);
        for (slot, array) in [
            (1, q),
            (2, &scratch.staged),
            (3, &scratch.staged),
            (4, &scratch.mask),
            (5, sinks),
            (6, &scratch.pad),
        ] {
            let (buffer, layout) = array.t.buffer_and_layout();
            encoder.set_input(slot, Some(buffer), layout.offset * array.dtype().size_of());
        }
        let (buffer, layout) = scratch.partials.t.buffer_and_layout();
        encoder.set_output(7, Some(buffer), layout.offset * 4);
        encoder.set_threadgroup_memory_length(0, (512 + 4 * 32 + 2 * 512) * 2);
        encoder.dispatch_groups((1, 64, 32), (32, 1, 1));
    }
    {
        let guard = runtime.commands.encoder()?;
        let encoder = guard.encoder();
        encoder.set_pipeline(&reduce_pipeline);
        encoder.set_bytes(0, &64i32);
        let (buffer, layout) = scratch.partials.t.buffer_and_layout();
        encoder.set_input(1, Some(buffer), layout.offset * 4);
        let (buffer, layout) = out.t.buffer_and_layout();
        encoder.set_output(2, Some(buffer), layout.offset * 4);
        if let Some((table, compressed)) = rope {
            let (buffer, layout) = table.t.buffer_and_layout();
            encoder.set_input(
                3,
                Some(buffer),
                (layout.offset + if compressed { 64 } else { 0 }) * 4,
            );
        }
        encoder.dispatch_groups((64, 1, 1), (1024, 1, 1));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deepseek41_transforming_reduce_matches_original_chain() {
        let stream = Stream::gpu();
        let mut scratch = DecodeAttentionScratch::new(&stream).unwrap();
        let values = |n: usize, stride: usize| {
            (0..n)
                .map(|i| ((i * stride % 997) as f32 - 499.0) / 199.0)
                .collect::<Vec<_>>()
        };
        let q = Array::from_slice(&values(64 * 512, 19), &[64, 512]);
        let raw = Array::from_slice(&values(128 * 512, 113), &[128, 512]);
        let sinks = Array::from_slice(&values(64, 23), &[64]);
        for position in [0, 1, 7, 8192, 131072, 1048575] {
            let table = super::super::rope_coefficient_table(position, &stream).unwrap();
            for compressed in [false, true] {
                let mut expected =
                    attention_decode(&q, &sinks, &raw, 128, 71, None, &mut scratch, &stream)
                        .unwrap()
                        .reshape(&[1, 64, 512])
                        .unwrap();
                super::super::rope_in_place(
                    &mut expected,
                    position,
                    1,
                    compressed,
                    true,
                    true,
                    &stream,
                )
                .unwrap();
                let actual = attention_decode_with_rope(
                    &q,
                    &sinks,
                    &raw,
                    128,
                    71,
                    None,
                    &mut scratch,
                    Some((&table, compressed)),
                    &stream,
                )
                .unwrap();
                assert!(
                    actual
                        .as_slice::<f32>()
                        .iter()
                        .map(|v| v.to_bits())
                        .eq(expected.as_slice::<f32>().iter().map(|v| v.to_bits())),
                    "position={position} compressed={compressed}"
                );
            }
        }
    }

    #[test]
    fn deepseek41_attention_ring_padding_sinks_and_scratch_reuse() {
        let stream = Stream::gpu();
        let mut scratch = DecodeAttentionScratch::new(&stream).unwrap();
        let q = Array::from_slice(&[0f32; 64 * 512], &[64, 512]);
        let sinks = Array::from_slice(&[0f32; 64], &[64]);
        let raw_values: Vec<f32> = (0..128 * 512)
            .map(|i| ((i / 512 * 3 + i % 512) % 17) as f32 / 8.0 - 1.0)
            .collect();
        let comp_values: Vec<f32> = (0..512 * 512)
            .map(|i| ((i / 512 * 7 + i % 512) % 13) as f32 / 8.0 - 0.75)
            .collect();
        let raw = Array::from_slice(&raw_values, &[128, 512]);
        let comp = Array::from_slice(&comp_values, &[512, 512]);
        let mut pending = Vec::new();
        for (raw_count, compressed) in [
            (1, 0),
            (31, 0),
            (32, 0),
            (127, 0),
            (128, 0),
            (1, 1),
            (31, 31),
            (32, 32),
            (127, 511),
            (128, 512),
        ] {
            let ids: Vec<i32> = (0..compressed).map(|i| (i * 7 + 3) % 512).collect();
            let ids_gpu = Array::from_slice(&ids, &[compressed]);
            let selected = if compressed > 0 {
                Some((&comp, &ids_gpu))
            } else {
                None
            };
            let out = attention_decode(
                &q,
                &sinks,
                &raw,
                raw_count,
                125,
                selected,
                &mut scratch,
                &stream,
            )
            .unwrap();
            let expected: Vec<f32> = (0..512)
                .map(|col| {
                    let mut value = 0.0;
                    for r in 0..raw_count {
                        value += raw_values[((125 + r) % 128) as usize * 512 + col];
                    }
                    for &r in &ids {
                        value += comp_values[r as usize * 512 + col];
                    }
                    // Sink contributes exp(0) to the denominator and no value.
                    value / (raw_count as f32 + compressed as f32 + 1.0)
                })
                .collect();
            pending.push((out, expected));
        }
        // All ten calls reuse scratch before any readback or explicit wait.
        for (out, expected) in pending {
            for row in out.as_slice::<f32>().chunks_exact(512) {
                for (&actual, &expected) in row.iter().zip(&expected) {
                    assert!(
                        (actual - expected).abs() < 2e-6,
                        "attention {actual} != {expected}"
                    );
                }
            }
        }
    }
}
