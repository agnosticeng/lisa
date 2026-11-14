//! DSpine selection kernels. Floating-point projections and reductions remain
//! in their independently checked paths; these kernels only move integer IDs.
use std::sync::OnceLock;

use crate::ffi::{MetalKernel, OutputArg, TemplateArg};
use crate::{Array, Dtype, Stream, error::Result};

/// Normalize the 1024 code channels and replace each block's first row by its
/// anchor code. The scale is sqrt(1024)=32, exactly representable in F32.
pub fn normalize_codes(
    projected: &Array,
    input_ids: &Array,
    codebook: &Array,
    eps: f32,
) -> Result<Array> {
    normalize_codes_on(
        projected,
        input_ids,
        codebook,
        eps,
        &Stream::thread_local_or_default(),
    )
}

pub(crate) fn normalize_codes_on(
    projected: &Array,
    input_ids: &Array,
    codebook: &Array,
    eps: f32,
    stream: &Stream,
) -> Result<Array> {
    if projected.rank() != 3
        || projected.dim(2) != 1024
        || projected.dtype() != Dtype::Bfloat16
        || input_ids.shape() != &projected.shape()[..2]
        || !matches!(input_ids.dtype(), Dtype::Int32 | Dtype::Uint32)
        || codebook.rank() != 2
        || codebook.dim(1) != 1024
        || codebook.dtype() != Dtype::Bfloat16
    {
        return Err(crate::error::Error::msg(
            "invalid DSpine code normalization layout",
        ));
    }
    let projected = projected.contiguous()?;
    let input_ids = input_ids.contiguous()?;
    let codebook = codebook.contiguous()?;
    let eps = Array::new(crate::array::Array::from_slice(
        stream.runtime(),
        &[eps],
        &[1],
    )?);
    let (b, t) = (projected.dim(0), projected.dim(1));
    static KERNEL: OnceLock<MetalKernel> = OnceLock::new();
    let kernel = KERNEL.get_or_init(|| {
        MetalKernel::new_with_math(
            "dspine_normalize_codes",
            &["projected", "input_ids", "codebook", "eps"],
            &["codes"],
            crate::kernel_source!("dspine/normalize_codes.metal"),
            "",
            true,
            false,
            crate::runtime::Math::SafeNoLang,
        )
        .expect("DSpine code normalization kernel")
    });
    Ok(kernel
        .apply(
            &[&projected, &input_ids, &codebook, &eps],
            &[TemplateArg::Int("C", 1024), TemplateArg::Int("T", t)],
            (b * t * 128, 1, 1),
            (128, 1, 1),
            &[OutputArg {
                shape: projected.shape().to_vec(),
                dtype: Dtype::Bfloat16,
            }],
            stream,
        )?
        .remove(0))
}

/// The bridge's read RMS and weighted RMS normalization share the exact MLX
/// reduction tree. `read=true` returns (normalized, RMS); otherwise the second
/// output is the inverse RMS and the BF16 intermediate cast precedes weighting.
pub fn normalize(raw: &Array, weight: &Array, eps: f32, read: bool) -> Result<(Array, Array)> {
    normalize_on(raw, weight, eps, read, &Stream::thread_local_or_default())
}

pub(crate) fn normalize_on(
    raw: &Array,
    weight: &Array,
    eps: f32,
    read: bool,
    stream: &Stream,
) -> Result<(Array, Array)> {
    if raw.rank() != 3
        || raw.dtype() != Dtype::Bfloat16
        || raw.dim(2) != 5120
        || weight.dtype() != Dtype::Bfloat16
        || weight.shape() != [5120]
    {
        return Err(crate::error::Error::msg(
            "DSpine normalization expects BF16 [B,T,5120] and weight [5120]",
        ));
    }
    let raw = raw.contiguous()?;
    let weight = weight.contiguous()?;
    let (b, t, h) = (raw.dim(0), raw.dim(1), raw.dim(2));
    static KERNEL: OnceLock<MetalKernel> = OnceLock::new();
    let kernel = KERNEL.get_or_init(|| {
        MetalKernel::new_with_math(
            "dspine_normalize",
            &["raw", "weight", "eps", "reciprocal"],
            &["normalized", "rms"],
            crate::kernel_source!("dspine/normalize.metal"),
            "",
            true,
            false,
            crate::runtime::Math::SafeNoLang,
        )
        .expect("DSpine normalization kernel")
    });
    let eps = Array::new(crate::array::Array::from_slice(
        stream.runtime(),
        &[eps],
        &[1],
    )?);
    let reciprocal = Array::new(crate::array::Array::from_slice(
        stream.runtime(),
        &[1.0f32 / h as f32],
        &[1],
    )?);
    let mut output = kernel.apply(
        &[&raw, &weight, &eps, &reciprocal],
        &[
            TemplateArg::Int("H", h),
            TemplateArg::Int("THREADS", 1024),
            TemplateArg::Bool("READ", read),
        ],
        (b * t * 1024, 1, 1),
        (1024, 1, 1),
        &[
            OutputArg {
                shape: raw.shape().to_vec(),
                dtype: Dtype::Bfloat16,
            },
            OutputArg {
                shape: vec![b, t],
                dtype: Dtype::Float32,
            },
        ],
        stream,
    )?;
    let rms = output.pop().unwrap();
    Ok((output.pop().unwrap(), rms))
}

/// Causal code-bridge message with the exact composed MLX sigmoid and BF16 cast.
pub fn feature_message(packed: &Array, bias: &Array, batch: i32, steps: i32) -> Result<Array> {
    feature_message_on(
        packed,
        bias,
        batch,
        steps,
        &Stream::thread_local_or_default(),
    )
}

pub(crate) fn feature_message_on(
    packed: &Array,
    bias: &Array,
    batch: i32,
    steps: i32,
    stream: &Stream,
) -> Result<Array> {
    let rank = bias.size() as i32;
    if batch < 1
        || steps < 1
        || packed.shape() != [batch * steps, 3 * rank]
        || bias.rank() != 1
        || packed.dtype() != Dtype::Bfloat16
        || bias.dtype() != Dtype::Bfloat16
    {
        return Err(crate::error::Error::msg(
            "invalid DSpine feature message layout",
        ));
    }
    let packed = packed.contiguous()?;
    let bias = bias.contiguous()?;
    static KERNEL: OnceLock<MetalKernel> = OnceLock::new();
    let kernel = KERNEL.get_or_init(|| {
        MetalKernel::new_with_math(
            "dspine_feature_message",
            &["packed", "bias"],
            &["message"],
            crate::kernel_source!("dspine/feature_message.metal"),
            "",
            true,
            false,
            crate::runtime::Math::SafeNoLang,
        )
        .expect("DSpine feature message kernel")
    });
    Ok(kernel
        .apply(
            &[&packed, &bias],
            &[
                TemplateArg::Int("B", batch),
                TemplateArg::Int("T", steps),
                TemplateArg::Int("R", rank),
            ],
            (batch * steps * rank, 1, 1),
            (256, 1, 1),
            &[OutputArg {
                shape: vec![batch, steps, rank],
                dtype: Dtype::Bfloat16,
            }],
            stream,
        )?
        .remove(0))
}

/// Apply the causal residual write without reassociating multiplication or FMA.
pub fn feature_update(raw: &Array, delta: &Array, rms: &Array) -> Result<Array> {
    feature_update_on(raw, delta, rms, &Stream::thread_local_or_default())
}

pub(crate) fn feature_update_on(
    raw: &Array,
    delta: &Array,
    rms: &Array,
    stream: &Stream,
) -> Result<Array> {
    if raw.rank() != 3
        || raw.shape() != delta.shape()
        || raw.dtype() != Dtype::Bfloat16
        || delta.dtype() != Dtype::Bfloat16
        || rms.dtype() != Dtype::Float32
        || rms.shape() != &raw.shape()[..2]
    {
        return Err(crate::error::Error::msg(
            "invalid DSpine feature update layout",
        ));
    }
    let raw = raw.contiguous()?;
    let delta = delta.contiguous()?;
    let rms = rms.contiguous()?;
    let (b, t, h) = (raw.dim(0), raw.dim(1), raw.dim(2));
    static KERNEL: OnceLock<MetalKernel> = OnceLock::new();
    let kernel = KERNEL.get_or_init(|| {
        MetalKernel::new_with_math(
            "dspine_feature_update",
            &["raw", "delta", "rms"],
            &["updated"],
            crate::kernel_source!("dspine/feature_update.metal"),
            "",
            true,
            false,
            crate::runtime::Math::SafeNoLang,
        )
        .expect("DSpine feature update kernel")
    });
    Ok(kernel
        .apply(
            &[&raw, &delta, &rms],
            &[
                TemplateArg::Int("B", b),
                TemplateArg::Int("T", t),
                TemplateArg::Int("H", h),
            ],
            (b * t * h, 1, 1),
            (256, 1, 1),
            &[OutputArg {
                shape: raw.shape().to_vec(),
                dtype: Dtype::Bfloat16,
            }],
            stream,
        )?
        .remove(0))
}

const SELECT_HEADER: &str = r#"
inline bool dspine_before(float a, uint ai, float b, uint bi) {
    if (ai == UINT_MAX) return false;
    if (bi == UINT_MAX) return true;
    const bool an = metal::isnan(a), bn = metal::isnan(b);
    if (an != bn) return !an;
    if (!an && a != b) return a > b;
    return ai < bi;
}
inline uint dspine_group_best(uint id, float value, uint lane,
                              threadgroup uint* ids, threadgroup float* values) {
    for (ushort stride = 16; stride; stride >>= 1) {
        uint other = simd_shuffle_xor(id, stride);
        float ov = simd_shuffle_xor(value, stride);
        if (dspine_before(ov, other, value, id)) { id = other; value = ov; }
    }
    if ((lane & 31) == 0) { ids[lane / 32] = id; values[lane / 32] = value; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    id = UINT_MAX;
    value = 0.0f;
    for (uint group = 0; group < 8; ++group) {
        if (dspine_before(values[group], ids[group], value, id)) {
            id = ids[group]; value = values[group];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return id;
}
"#;

/// Exact two-pass retrieval: choose sixteen 64-token chunks by their maxima,
/// then select the best sixteen tokens in that union. No approximate head or
/// floating-point reduction is introduced. Cutoff ties retain lower token IDs.
pub fn top16_candidates(scores: &Array) -> Result<Array> {
    top16_candidates_with_workspace(scores, &mut SelectionWorkspace::default())
}

/// Two shape-bounded retrieval buffers. Reuse requires GPU completion and no
/// surviving output/view lease. This does not use GLM's ordered-reuse scope.
#[derive(Default)]
pub struct SelectionWorkspace {
    maxima: Option<Array>,
    selected: Option<Array>,
}

impl SelectionWorkspace {
    fn output(slot: &mut Option<Array>, shape: &[i32], stream: &Stream) -> Result<Array> {
        let runtime = stream.runtime();
        let reusable = slot.as_ref().is_some_and(|a| {
            let (buffer, _) = a.t.buffer_and_layout();
            a.shape() == shape && std::sync::Arc::ptr_eq(a.t.device(), runtime)
                // The allocator retains one owner for every leased buffer;
                // the workspace holds the other. Any view adds another Arc.
                && std::sync::Arc::strong_count(buffer) == 2
                && buffer.reusable_after(runtime.commands.completed_watermark())
        });
        if !reusable {
            let dims = shape.iter().map(|&d| d as usize).collect::<Vec<_>>();
            *slot = Some(Array::new(crate::array::Array::zeros(
                runtime,
                &dims,
                Dtype::Uint32,
            )?));
        }
        let storage = slot.as_ref().unwrap();
        let (buffer, _) = storage.t.buffer_and_layout();
        let dims: Vec<usize> = shape.iter().map(|&n| n as usize).collect();
        // Fresh host-read memoization is essential when recycling storage.
        Ok(Array::new(crate::array::Array::from_parts(
            runtime,
            buffer.clone(),
            &dims,
            Dtype::Uint32,
        )))
    }
}

pub fn top16_candidates_with_workspace(
    scores: &Array,
    workspace: &mut SelectionWorkspace,
) -> Result<Array> {
    top16_candidates_with_workspace_on(scores, workspace, &Stream::thread_local_or_default())
}

pub(crate) fn top16_candidates_with_workspace_on(
    scores: &Array,
    workspace: &mut SelectionWorkspace,
    stream: &Stream,
) -> Result<Array> {
    if scores.rank() != 2
        || scores.dim(0) < 1
        || scores.dim(1) < 32
        || scores.dim(1) > 262144
        || !matches!(
            scores.dtype(),
            Dtype::Float32 | Dtype::Bfloat16 | Dtype::Float16
        )
    {
        return Err(crate::error::Error::msg(
            "DSpine retrieval expects floating [N,32..262144]",
        ));
    }
    let scores = scores.contiguous()?;
    let (n, v) = (scores.dim(0), scores.dim(1));
    let chunks = (v + 63) / 64;
    static CHUNK: OnceLock<MetalKernel> = OnceLock::new();
    static SELECT: OnceLock<MetalKernel> = OnceLock::new();
    let chunk = CHUNK.get_or_init(|| {
        MetalKernel::new_with_math(
            "dspine_top16_chunks",
            &["scores"],
            &["maxima"],
            crate::kernel_source!("dspine/top16_chunks.metal"),
            SELECT_HEADER,
            true,
            false,
            crate::runtime::Math::Safe,
        )
        .expect("DSpine chunk maxima kernel")
    });
    let select = SELECT.get_or_init(|| {
        MetalKernel::new_with_math(
            "dspine_top16_select",
            &["scores", "maxima"],
            &["selected"],
            crate::kernel_source!("dspine/top16_select.metal"),
            SELECT_HEADER,
            true,
            false,
            crate::runtime::Math::Safe,
        )
        .expect("DSpine top16 selection kernel")
    });
    let params = [
        TemplateArg::Int("V", v),
        TemplateArg::Int("CHUNKS", chunks),
        TemplateArg::Int("SLOTS", (chunks + 255) / 256),
    ];
    let maxima_out = SelectionWorkspace::output(&mut workspace.maxima, &[n, chunks], stream)?;
    let selected_out = SelectionWorkspace::output(&mut workspace.selected, &[n, 16], stream)?;
    let maxima = chunk
        .apply_into(
            &[&scores],
            &params,
            (chunks * 32, n, 1),
            (32, 1, 1),
            &[OutputArg {
                shape: vec![n, chunks],
                dtype: Dtype::Uint32,
            }],
            &[&maxima_out],
            stream,
        )?
        .remove(0);
    Ok(select
        .apply_into(
            &[&scores, &maxima],
            &params,
            (n * 256, 1, 1),
            (256, 1, 1),
            &[OutputArg {
                shape: vec![n, 16],
                dtype: Dtype::Uint32,
            }],
            &[&selected_out],
            stream,
        )?
        .remove(0))
}

/// Coarse-head shortlist. Same stable comparison law as top16, with 128
/// selected chunks and a 128-token result ordered by global token ID.
pub fn top128_candidates_with_workspace(
    scores: &Array,
    workspace: &mut SelectionWorkspace,
) -> Result<Array> {
    top128_candidates_with_workspace_on(scores, workspace, &Stream::thread_local_or_default())
}

pub(crate) fn top128_candidates_with_workspace_on(
    scores: &Array,
    workspace: &mut SelectionWorkspace,
    stream: &Stream,
) -> Result<Array> {
    if scores.rank() != 2
        || scores.dim(0) < 1
        || scores.dim(1) < 128
        || scores.dim(1) > 262144
        || !matches!(
            scores.dtype(),
            Dtype::Float32 | Dtype::Bfloat16 | Dtype::Float16
        )
    {
        return Err(crate::error::Error::msg(
            "DSpine retrieval expects floating [N,128..262144]",
        ));
    }
    let scores = scores.contiguous()?;
    let (n, v) = (scores.dim(0), scores.dim(1));
    let chunks = (v + 63) / 64;
    static CHUNK: OnceLock<MetalKernel> = OnceLock::new();
    static SELECT: OnceLock<MetalKernel> = OnceLock::new();
    let chunk = CHUNK.get_or_init(|| {
        MetalKernel::new_with_math(
            "dspine_top16_chunks",
            &["scores"],
            &["maxima"],
            crate::kernel_source!("dspine/top16_chunks.metal"),
            SELECT_HEADER,
            true,
            false,
            crate::runtime::Math::Safe,
        )
        .expect("DSpine chunk maxima kernel")
    });
    let select = SELECT.get_or_init(|| {
        MetalKernel::new_with_math(
            "dspine_top128_select",
            &["scores", "maxima"],
            &["selected"],
            crate::kernel_source!("dspine/top128_select.metal"),
            SELECT_HEADER,
            true,
            false,
            crate::runtime::Math::Safe,
        )
        .expect("DSpine top128 selection kernel")
    });
    let params = [
        TemplateArg::Int("V", v),
        TemplateArg::Int("CHUNKS", chunks),
        TemplateArg::Int("SLOTS", (chunks + 255) / 256),
    ];
    let maxima_out = SelectionWorkspace::output(&mut workspace.maxima, &[n, chunks], stream)?;
    let selected_out = SelectionWorkspace::output(&mut workspace.selected, &[n, 128], stream)?;
    let maxima = chunk
        .apply_into(
            &[&scores],
            &params,
            (chunks * 32, n, 1),
            (32, 1, 1),
            &[OutputArg {
                shape: vec![n, chunks],
                dtype: Dtype::Uint32,
            }],
            &[&maxima_out],
            stream,
        )?
        .remove(0);
    Ok(select
        .apply_into(
            &[&scores, &maxima],
            &params,
            (n * 256, 1, 1),
            (256, 1, 1),
            &[OutputArg {
                shape: vec![n, 128],
                dtype: Dtype::Uint32,
            }],
            &[&selected_out],
            stream,
        )?
        .remove(0))
}

/// Sort the first sixteen IDs of the ordinary top-32 result in one dispatch.
pub fn sorted_candidates(selected: &Array) -> Result<Array> {
    if selected.rank() != 2 || selected.dim(1) != 32 || selected.dtype() != Dtype::Uint32 {
        return Err(crate::error::Error::msg(
            "DSpine candidate ordering expects U32 [N,32]",
        ));
    }
    let selected = selected.contiguous()?;
    let n = selected.dim(0);
    static KERNEL: OnceLock<MetalKernel> = OnceLock::new();
    let kernel = KERNEL.get_or_init(|| {
        MetalKernel::new(
            "dspine_sort_candidates",
            &["selected"],
            &["candidates"],
            crate::kernel_source!("dspine/sort_candidates.metal"),
            "",
            true,
            false,
        )
        .expect("DSpine candidate ordering kernel")
    });
    Ok(kernel
        .apply(
            &[&selected],
            &[TemplateArg::Int("N", n)],
            (n, 1, 1),
            (32, 1, 1),
            &[OutputArg {
                shape: vec![n, 16],
                dtype: Dtype::Uint32,
            }],
            &Stream::thread_local_or_default(),
        )?
        .remove(0))
}

/// Resolve the serial chain through a parallel table of conditional choices.
pub fn greedy_chain(choices: &Array, candidates: &Array) -> Result<Array> {
    greedy_chain_on(choices, candidates, &Stream::thread_local_or_default())
}

pub(crate) fn greedy_chain_on(
    choices: &Array,
    candidates: &Array,
    stream: &Stream,
) -> Result<Array> {
    if choices.rank() != 2
        || choices.shape() != candidates.shape()
        || choices.dtype() != Dtype::Uint32
        || candidates.dtype() != Dtype::Uint32
        || !(1..=15).contains(&choices.dim(0))
        || choices.dim(1) != 16
    {
        return Err(crate::error::Error::msg(
            "DSpine chain expects U32 [1..15,16] choices and candidates",
        ));
    }
    let choices = choices.contiguous()?;
    let candidates = candidates.contiguous()?;
    let n = choices.dim(0);
    static KERNEL: OnceLock<MetalKernel> = OnceLock::new();
    let kernel = KERNEL.get_or_init(|| {
        MetalKernel::new(
            "dspine_greedy_chain",
            &["choices", "candidates"],
            &["tokens"],
            crate::kernel_source!("dspine/greedy_chain.metal"),
            "",
            true,
            false,
        )
        .expect("DSpine greedy chain kernel")
    });
    Ok(kernel
        .apply(
            &[&choices, &candidates],
            &[TemplateArg::Int("N", n), TemplateArg::Int("K", 16)],
            (1, 1, 1),
            (1, 1, 1),
            &[OutputArg {
                shape: vec![1, n],
                dtype: Dtype::Uint32,
            }],
            stream,
        )?
        .remove(0))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn selection_workspace_preserves_live_views_and_refreshes_host_reads() -> Result<()> {
        use crate::ops::indexing::IndexOp;
        let scores = |shift: i32| {
            Array::from_slice(
                &(0..1024)
                    .map(|i| ((i + shift) % 1024) as f32)
                    .collect::<Vec<_>>(),
                &[1, 1024],
            )
        };
        let mut workspace = SelectionWorkspace::default();
        let first = top16_candidates_with_workspace(&scores(0), &mut workspace)?;
        let original = first.to_vec1::<u32>()?;
        let view = first.index((.., 0..8));
        let buffer = first.t.buffer_and_layout().0.id();
        drop(first);
        let second = top16_candidates_with_workspace(&scores(20), &mut workspace)?;
        assert_ne!(
            buffer,
            second.t.buffer_and_layout().0.id(),
            "live view must prevent reuse"
        );
        assert_eq!(view.to_vec1::<u32>()?, original[..8]);
        assert_eq!(
            second.to_vec1::<u32>()?,
            top16_candidates(&scores(20))?.to_vec1::<u32>()?
        );
        let buffer = second.t.buffer_and_layout().0.id();
        drop(second);
        let third = top16_candidates_with_workspace(&scores(40), &mut workspace)?;
        assert_eq!(
            buffer,
            third.t.buffer_and_layout().0.id(),
            "completed unleased storage should be reused"
        );
        assert_eq!(
            third.to_vec1::<u32>()?,
            top16_candidates(&scores(40))?.to_vec1::<u32>()?
        );
        Ok(())
    }
    use crate::ops::indexing::IndexOp;

    fn assert_bits_equal(a: &Array, b: &Array) -> Result<()> {
        let a = a.as_dtype(Dtype::Float32)?.to_vec1::<f32>()?;
        let b = b.as_dtype(Dtype::Float32)?.to_vec1::<f32>()?;
        assert_eq!(a.len(), b.len());
        for (i, (a, b)) in a.into_iter().zip(b).enumerate() {
            if a.is_nan() && b.is_nan() {
                continue;
            }
            assert_eq!(a.to_bits(), b.to_bits(), "element {i}: {a} != {b}");
        }
        Ok(())
    }

    #[test]
    fn code_normalization_preserves_projection_and_anchor_rows() -> Result<()> {
        for (b, t) in [(1, 3), (2, 7), (1, 16)] {
            let ids: Vec<i32> = (0..b * t).map(|i| (i * 17 + 3) % 64).collect();
            let ids = Array::from_slice(&ids, &[b, t]);
            let book: Vec<f32> = (0..64 * 1024)
                .map(|i| ((i * 37 % 251) as f32 - 125.0) / 127.0)
                .collect();
            let book = Array::from_slice(&book, &[64, 1024]).as_dtype(Dtype::Bfloat16)?;
            for variant in 0..3 {
                let values: Vec<f32> = (0..b * t * 1024)
                    .map(|i| match variant {
                        0 => ((i * 7919 % 997) as f32 - 498.0) / 127.0,
                        1 => f32::from_bits(((i * 7 % 65536) as u32) << 16),
                        _ => 0.0,
                    })
                    .collect();
                let projected =
                    Array::from_slice(&values, &[b, t, 1024]).as_dtype(Dtype::Bfloat16)?;
                let f = projected.as_dtype(Dtype::Float32)?;
                let norm = f.multiply(&f)?.sum_axis(-1, true)?.sqrt()?;
                let expected = f
                    .multiply(&Array::from_f32(32.0))?
                    .divide(&norm.add(&Array::from_f32(1e-6))?)?
                    .as_dtype(Dtype::Bfloat16)?;
                let anchor_ids = ids.index((.., 0)).reshape(&[b])?;
                let anchor = book
                    .take_axis(&anchor_ids, 0)?
                    .as_dtype(Dtype::Float32)?
                    .multiply(&Array::from_f32(32.0))?
                    .as_dtype(Dtype::Bfloat16)?
                    .expand_dims(1)?;
                let expected = Array::concatenate(&[&anchor, &expected.index((.., 1.., ..))], 1)?;
                assert_bits_equal(&normalize_codes(&projected, &ids, &book, 1e-6)?, &expected)?;
            }
        }
        Ok(())
    }

    #[test]
    fn bridge_normalization_preserves_mlx_reduction_and_casts() -> Result<()> {
        for (b, t) in [(1, 1), (1, 2), (1, 3), (1, 7), (2, 7), (1, 16), (1, 64)] {
            for variant in 0..4 {
                let values: Vec<f32> = (0..b * t * 5120)
                    .map(|i| match variant {
                        0 => ((i * 7919 % 997) as f32 - 498.0) / 127.0,
                        1 => f32::from_bits(((i % 65536) as u32) << 16),
                        2 => {
                            if i % 2 == 0 {
                                1.0e-20
                            } else {
                                -1.0e-20
                            }
                        }
                        _ => 0.0,
                    })
                    .collect();
                let raw = Array::from_slice(&values, &[b, t, 5120]).as_dtype(Dtype::Bfloat16)?;
                let w: Vec<f32> = (0..5120).map(|i| (i % 37) as f32 / 16.0).collect();
                let weight = Array::from_slice(&w, &[5120]).as_dtype(Dtype::Bfloat16)?;
                let f = raw.as_dtype(Dtype::Float32)?;
                let mean = f.multiply(&f)?.mean_axis(-1, false)?;
                for eps in [1.0e-6f32, 1.0e-5] {
                    let rms = mean.maximum(&Array::from_f32(f32::MIN_POSITIVE))?.sqrt()?;
                    let expected = f
                        .divide(&rms.expand_dims(-1)?.add(&Array::from_f32(eps))?)?
                        .as_dtype(Dtype::Bfloat16)?;
                    let (actual, actual_rms) = normalize(&raw, &weight, eps, true)?;
                    assert_bits_equal(&actual_rms, &rms)?;
                    assert_bits_equal(&actual, &expected)?;
                    let inv = mean.add(&Array::from_f32(eps))?.rsqrt()?;
                    let expected = f
                        .multiply(&inv.expand_dims(-1)?)?
                        .as_dtype(Dtype::Bfloat16)?
                        .as_dtype(Dtype::Float32)?
                        .multiply(&weight.as_dtype(Dtype::Float32)?)?
                        .as_dtype(Dtype::Bfloat16)?;
                    let (actual, actual_inv) = normalize(&raw, &weight, eps, false)?;
                    assert_bits_equal(&actual_inv, &inv)?;
                    assert_bits_equal(&actual, &expected)?;
                }
            }
        }
        Ok(())
    }

    #[test]
    fn verify_row_argmax_matches_separate_completions() -> Result<()> {
        use crate::ops::indexing::{argmax, argmax_axis};
        for rows in [1, 2, 6, 7, 16] {
            let width = 248320;
            let data: Vec<f32> = (0..rows * width)
                .map(|i| {
                    let row = i / width;
                    let col = i % width;
                    if row % 3 == 0 {
                        (col % 79) as f32
                    } else {
                        ((i * 43 + row) % 997) as f32 / 31.0
                    }
                })
                .collect();
            for dtype in [Dtype::Float32, Dtype::Bfloat16] {
                let logits = Array::from_slice(&data, &[1, rows, width]).as_dtype(dtype)?;
                let together = argmax_axis(&logits, -1, false)?.to_vec1::<u32>()?;
                for r in 0..rows {
                    let row = logits.index((.., r, ..)).contiguous()?;
                    let separate = argmax(&row, None)?.item_cast::<u32>();
                    assert_eq!(together[r as usize], separate);
                    // Sampler::draw promotes to F32 before its greedy argmax.
                    assert_eq!(
                        separate,
                        argmax(&row.as_dtype(Dtype::Float32)?, None)?.item_cast::<u32>()
                    );
                }
            }
        }
        Ok(())
    }

    #[test]
    fn bridge_fusions_match_composed_ops_exhaustive_bf16() -> Result<()> {
        use crate::ops;
        // Every BF16 gate operand, including subnormals and infinities. The
        // realistic shapes separately cover shifted rows and batch boundaries.
        for (b, t, r) in [(1, 3, 65536), (2, 7, 512), (1, 16, 512)] {
            for shift in [0.0f32, -3.125, 0.00390625] {
                let packed: Vec<f32> = (0..b * t * 3 * r)
                    .map(|i| {
                        let col = i % (3 * r);
                        if col < r {
                            f32::from_bits(((i % 65536) as u32) << 16)
                        } else if col < 2 * r {
                            shift
                        } else {
                            ((i % 101) as f32 - 50.0) / 32.0
                        }
                    })
                    .collect();
                let packed =
                    Array::from_slice(&packed, &[b * t, 3 * r]).as_dtype(Dtype::Bfloat16)?;
                let bias =
                    Array::from_slice(&vec![shift; r as usize], &[r]).as_dtype(Dtype::Bfloat16)?;
                let x = packed.as_dtype(Dtype::Float32)?.reshape(&[b, t, 3 * r])?;
                let gate = x
                    .index((.., 1.., 0..r))
                    .add(&x.index((.., 0..t - 1, r..2 * r)))?
                    .add(&bias.as_dtype(Dtype::Float32)?)?
                    .sigmoid()?;
                let value = gate
                    .multiply(&x.index((.., 0..t - 1, 2 * r..3 * r)))?
                    .as_dtype(Dtype::Bfloat16)?;
                let zeros = ops::zeros::<f32>(&[b, 1, r])?.as_dtype(Dtype::Bfloat16)?;
                let expected = Array::concatenate(&[&zeros, &value], 1)?;
                assert_bits_equal(&feature_message(&packed, &bias, b, t)?, &expected)?;
            }
        }
        for (b, t, h) in [(1, 3, 65536), (2, 7, 5120), (1, 16, 5120)] {
            let raw: Vec<f32> = (0..b * t * h)
                .map(|i| f32::from_bits(((i % 65536) as u32) << 16))
                .collect();
            let delta: Vec<f32> = (0..b * t * h)
                .map(|i| ((i % 37) as f32 - 18.0) / 7.0)
                .collect();
            let rms: Vec<f32> = (0..b * t).map(|i| (i as f32 + 1.0) / 11.0).collect();
            let raw = Array::from_slice(&raw, &[b, t, h]).as_dtype(Dtype::Bfloat16)?;
            let delta = Array::from_slice(&delta, &[b, t, h]).as_dtype(Dtype::Bfloat16)?;
            let rms = Array::from_slice(&rms, &[b, t]);
            let mask = Array::from_slice(
                &(0..t)
                    .map(|i| if i == 0 { 0.0f32 } else { 1.0 })
                    .collect::<Vec<_>>(),
                &[1, t, 1],
            );
            let oracle = raw
                .as_dtype(Dtype::Float32)?
                .add(
                    &rms.expand_dims(-1)?
                        .multiply(&delta.as_dtype(Dtype::Float32)?)?
                        .multiply(&mask)?,
                )?
                .as_dtype(Dtype::Bfloat16)?;
            assert_bits_equal(&feature_update(&raw, &delta, &rms)?, &oracle)?;
        }
        Ok(())
    }

    #[test]
    fn chunk_retrieval_matches_stable_sort_and_existing_top32() -> Result<()> {
        for width in [32usize, 63, 64, 65, 511, 1024, 1025, 248320, 262144] {
            let rows = 4;
            let mut data = vec![0.0f32; (rows + 1) * width];
            for r in 0..rows {
                for j in 0..width {
                    data[(r + 1) * width + j] = match r {
                        0 => ((j.wrapping_mul(7919) ^ (j >> 4)) % 4096) as f32 / 64.0,
                        1 => {
                            if j % 7 == 0 {
                                -0.0
                            } else {
                                0.0
                            }
                        }
                        2 => match j % 5 {
                            0 => f32::NAN,
                            1 => f32::INFINITY,
                            2 => f32::NEG_INFINITY,
                            _ => 1.0,
                        },
                        _ => f32::NAN,
                    };
                }
            }
            for dtype in [Dtype::Float32, Dtype::Bfloat16, Dtype::Float16] {
                let input = Array::from_slice(&data, &[rows as i32 + 1, width as i32])
                    .as_dtype(dtype)?
                    .index((1..rows as i32 + 1, ..));
                let result = top16_candidates(&input)?.to_vec1::<u32>()?;
                let old =
                    sorted_candidates(&crate::ops::top32_indices(&input)?)?.to_vec1::<u32>()?;
                assert_eq!(result, old, "width={width} dtype={dtype:?}");
                let values = input.as_dtype(Dtype::Float32)?.to_vec1::<f32>()?;
                let mut oracle = Vec::new();
                for row in values.chunks_exact(width) {
                    let mut ids: Vec<_> = (0..width as u32).collect();
                    ids.sort_by(|&a, &b| {
                        let (x, y) = (row[a as usize], row[b as usize]);
                        x.is_nan()
                            .cmp(&y.is_nan())
                            .then_with(|| y.partial_cmp(&x).unwrap_or(std::cmp::Ordering::Equal))
                            .then(a.cmp(&b))
                    });
                    ids.truncate(16);
                    ids.sort_unstable();
                    oracle.extend(ids);
                }
                assert_eq!(result, oracle, "host width={width} dtype={dtype:?}");
                if width >= 128 {
                    let result = top128_candidates_with_workspace(
                        &input,
                        &mut SelectionWorkspace::default(),
                    )?
                    .to_vec1::<u32>()?;
                    let mut expected = Vec::new();
                    for row in values.chunks_exact(width) {
                        let mut ids: Vec<_> = (0..width as u32).collect();
                        ids.sort_by(|&a, &b| {
                            let (x, y) = (row[a as usize], row[b as usize]);
                            x.is_nan()
                                .cmp(&y.is_nan())
                                .then_with(|| {
                                    y.partial_cmp(&x).unwrap_or(std::cmp::Ordering::Equal)
                                })
                                .then(a.cmp(&b))
                        });
                        ids.truncate(128);
                        ids.sort_unstable();
                        expected.extend(ids);
                    }
                    assert_eq!(
                        result, expected,
                        "top128 host width={width} dtype={dtype:?}"
                    );
                }
            }
        }
        Ok(())
    }

    #[test]
    fn integer_selection_matches_host_for_all_draft_widths() -> Result<()> {
        for n in 1..=15usize {
            let input: Vec<u32> = (0..n)
                .flat_map(|r| (0..32).map(move |j| ((31 - j) * 7919 + r) as u32))
                .collect();
            let top = Array::from_slice(&input, &[n as i32, 32]);
            let ids = sorted_candidates(&top)?;
            let mut expected = Vec::new();
            for row in input.chunks_exact(32) {
                let mut v = row[..16].to_vec();
                v.sort_unstable();
                expected.extend(v);
            }
            assert_eq!(ids.to_vec1::<u32>()?, expected);
            // Every possible initial choice, with a changing conditional path.
            for first in 0..16 {
                let table: Vec<u32> = (0..n)
                    .flat_map(|r| {
                        (0..16).map(move |j| {
                            if r == 0 {
                                first
                            } else {
                                ((j * 7 + r * 3) % 16) as u32
                            }
                        })
                    })
                    .collect();
                let mut sender = 0;
                let oracle: Vec<u32> = (0..n)
                    .map(|r| {
                        sender = table[r * 16 + sender] as usize;
                        expected[r * 16 + sender]
                    })
                    .collect();
                let result = greedy_chain(&Array::from_slice(&table, &[n as i32, 16]), &ids)?;
                assert_eq!(result.to_vec1::<u32>()?, oracle);
            }
        }
        Ok(())
    }
}
