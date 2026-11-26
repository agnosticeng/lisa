#![cfg(test)]
//! Frozen pre-migration compositor. Only primitive adapter names changed.
use crate::bindings::glm53_attention_mode::*;
use lisa_metal::kernels::sparse::kernels::{self as native, dsa_indexed_lora_wide_native};
use lisa_metal::{Array, Dtype, Stream};
pub fn dsa_prefill_value_slices(tokens: usize, dense_prefix: usize) -> Vec<(usize, usize)> {
    let mut slices = Vec::new();
    let mut start = 0;
    while start < tokens {
        let mut length = (tokens - start).min(2048);
        if start < dense_prefix {
            length = length.min(dense_prefix - start);
        }
        slices.push((start, length));
        start += length;
    }
    slices
}

fn dsa_qk_lowrank_project(
    q: &Array,
    w: &Array,
    s: &Array,
    b: &Array,
    g: usize,
    stream: &Stream,
) -> Option<Array> {
    native::dsa_qk_lowrank_project(
        q,
        w,
        s,
        b,
        g,
        stream,
        glm53_dsa_native_enabled() && q.dim(1) == 64 && g == 64,
    )
}
fn dsa_value_project_rows(
    x: &Array,
    w: &Array,
    s: &Array,
    b: &Array,
    g: usize,
    h: usize,
    stream: &Stream,
) -> Option<Array> {
    let native = glm53_dsa_native_enabled() && h == 64 && g == 64;
    let tokens = x.dim(0) as usize / h;
    if native && let Some(prefix) = GLM53_DSA_VALUE_DENSE_PREFIX.with(std::cell::Cell::get) {
        let slices = dsa_prefill_value_slices(tokens, prefix);
        if slices.len() > 1 {
            let _scope = Glm53DsaValueSliceScope::new(0, 0);
            let mut out = vec![];
            for (start, len) in slices {
                let x = Array::new(x.t.narrow(0, start * h, len * h).ok()?);
                out.push(dsa_value_project_rows(&x, w, s, b, g, h, stream)?);
            }
            return Array::concatenate(&out.iter().collect::<Vec<_>>(), 0).ok();
        }
    }
    native::dsa_value_project_rows(x, w, s, b, g, h, stream, native, native && tokens >= 32)
}
#[allow(clippy::too_many_arguments)]
fn dsa_indexed_decode_fused(
    q: &Array,
    c: &Array,
    ids: &Array,
    n: usize,
    dense: bool,
    w: &Array,
    s: &Array,
    b: &Array,
    g: usize,
    scale: f32,
    stream: &Stream,
    capture: bool,
) -> Option<(Array, Option<Array>)> {
    native::dsa_indexed_decode_fused(
        q,
        c,
        ids,
        n,
        dense,
        w,
        s,
        b,
        g,
        scale,
        stream,
        capture,
        glm53_dsa_native_enabled() && g == 64,
    )
}

/// The reference's dense-causal FlashAttention prefill path for MLX-affine DSA V-B.
/// It keeps the query projection and F16 compact cache separate, runs the
/// reference FlashAttention schedule for attention-lora, then applies the
/// same split affine value projection as the reference.
#[allow(clippy::too_many_arguments)]
pub fn dsa_absorbed_attention_flash_causal_prefill_stages(
    q: &Array,
    kv_weight: &Array,
    kv_scales: &Array,
    kv_biases: &Array,
    group_size: usize,
    compact_latents: &Array,
    q_row0: usize,
    scale: f32,
    stream: &Stream,
    prefill_encoder_boundaries: bool,
) -> Option<(Array, Array, Array)> {
    let batches = q.dim(0) as usize;
    let n_heads = q.dim(1) as usize;
    let n_queries = q.dim(2) as usize;
    if batches != 1 || n_heads != 64 || n_queries == 0 || q.dim(3) != 256 {
        return None;
    }
    let q_low = dsa_qk_lowrank_project(q, kv_weight, kv_scales, kv_biases, group_size, stream)?;
    if prefill_encoder_boundaries {
        stream.runtime().commands.flush_encoder();
    }
    let lora = Array::new(
        lisa_metal::jit::ds4_flash_attention_prefill_raw_heads(
            &q_low.t,
            &compact_latents.t,
            q_row0,
            scale,
        )
        .ok()?,
    );
    let n_rows = batches.checked_mul(n_queries)?.checked_mul(n_heads)?;
    let lora_rows = lora.reshape(&[n_rows as i32, 512]).ok()?;
    let projected = dsa_value_project_rows(
        &lora_rows, kv_weight, kv_scales, kv_biases, group_size, n_heads, stream,
    )?
    .reshape(&[batches as i32, n_queries as i32, n_heads as i32, 256])
    .ok()?;
    if prefill_encoder_boundaries {
        stream.runtime().commands.flush_encoder();
    }
    let output = projected
        .transpose_axes(&[0, 2, 1, 3])
        .and_then(|value| value.contiguous())
        .ok()?;
    Some((output, q_low, lora))
}

#[allow(clippy::too_many_arguments)]
pub fn dsa_absorbed_attention_stages_inner(
    q: &Array,
    kv_weight: &Array,
    kv_scales: &Array,
    kv_biases: &Array,
    group_size: usize,
    compact_latents: &Array,
    selected: Option<&Array>,
    selected_valid_prefix: usize,
    scale: f32,
    stream: &Stream,
    capture_stages: bool,
    force_fused_decode: bool,
    capture_fused_lora: bool,
    prefill_encoder_boundaries: bool,
) -> Option<(Array, Option<Array>, Option<Array>)> {
    if q.rank() != 4
        || compact_latents.rank() != 3
        || selected.is_some_and(|selected| selected.rank() != 3)
    {
        return None;
    }
    let batches = q.dim(0) as usize;
    let heads = q.dim(1) as usize;
    let n_queries = q.dim(2) as usize;
    let tc = compact_latents.dim(1) as usize;
    let dense_all = selected.is_none();
    let requested_selected = selected.map_or(tc, |selected| selected.dim(2) as usize);
    if q.dim(3) != 256
        || kv_weight.shape() != [(heads * 512) as i32, 128]
        || kv_weight.dtype() != Dtype::Uint32
        || group_size == 0
        || 512 % group_size != 0
        || kv_scales.shape() != [(heads * 512) as i32, (512 / group_size) as i32]
        || kv_biases.shape() != kv_scales.shape()
        || compact_latents.dim(0) != batches as i32
        || selected.is_some_and(|selected| {
            selected.dim(0) != batches as i32 || selected.dim(1) != n_queries as i32
        })
        || compact_latents.dim(2) != 512
        || selected.is_some_and(|selected| selected.dtype() != Dtype::Int32)
        || batches == 0
        || n_queries == 0
        || heads == 0
        || requested_selected == 0
        || tc == 0
        || selected_valid_prefix > requested_selected
    {
        return None;
    }
    if q.dtype() != Dtype::Float32
        || kv_scales.dtype() != Dtype::Float32
        || kv_biases.dtype() != Dtype::Float32
        || compact_latents.dtype() != Dtype::Float16
    {
        return None;
    }
    let q = q.contiguous().ok()?;
    let kv_weight = kv_weight.contiguous().ok()?;
    let kv_scales = kv_scales.contiguous().ok()?;
    let kv_biases = kv_biases.contiguous().ok()?;
    let compact_latents = compact_latents.contiguous().ok()?;
    let selected = if let Some(selected) = selected {
        selected.contiguous().ok()?
    } else {
        static DENSE_ID_DUMMY: std::sync::OnceLock<Array> = std::sync::OnceLock::new();
        DENSE_ID_DUMMY
            .get_or_init(|| Array::from_slice(&[0i32; 8], &[1, 1, 8]))
            .clone()
    };
    for a in [
        &q,
        &kv_weight,
        &kv_scales,
        &kv_biases,
        &compact_latents,
        &selected,
    ] {
        a.eval_enqueue().ok()?;
    }
    // The reference switches a long private MTP window to its indexed batch-lora
    // kernel with n_tokens=1. Keep the already admitted shorter/batched
    // routes unchanged and reject unrelated unsupported wide layouts.
    let wide_mtp_window = requested_selected > 2051;
    if wide_mtp_window
        && (batches != 1
            || n_queries != 1
            || heads != 64
            || dense_all
            || requested_selected > tc
            || selected_valid_prefix != 0
            || prefill_encoder_boundaries)
    {
        return None;
    }

    // MLX places input buffers smaller than eight elements in constant
    // address space. The indexed-lora kernel reads selected ids through a
    // device pointer, so pad short dense/MTP windows only enough to reach the
    // eight-element threshold. The kernel skips invalid rows without changing
    // the online-softmax state.
    let selection_rows = batches * n_queries;
    let selection_elements = selection_rows * requested_selected;
    let selected = if !dense_all && selection_elements < 8 {
        let padding_per_row = (8 - selection_elements).div_ceil(selection_rows);
        let padding_values = vec![-1i32; selection_rows * padding_per_row];
        let padding = Array::from_slice(
            &padding_values,
            &[batches as i32, n_queries as i32, padding_per_row as i32],
        );
        Array::concatenate(&[&selected, &padding], 2)
            .and_then(|padded| padded.contiguous())
            .ok()?
    } else {
        selected
    };
    let n_selected = if dense_all {
        tc
    } else {
        selected.dim(2) as usize
    };

    let q_low = dsa_qk_lowrank_project(&q, &kv_weight, &kv_scales, &kv_biases, group_size, stream)?;
    q_low.eval_enqueue().ok()?;
    // The reference indexed prefill closes the encoder after qk_low and again after
    // attention plus value projection. Both remain in the same command buffer.
    if prefill_encoder_boundaries {
        stream.runtime().commands.flush_encoder();
    }

    let native_decode = GLM53_DSA_NATIVE_OVERRIDE.with(std::cell::Cell::get)
        && !prefill_encoder_boundaries
        && group_size == 64;
    if batches == 1
        && n_queries == 1
        && heads == 64
        && !wide_mtp_window
        && (native_decode || (!capture_stages && force_fused_decode))
    {
        let (projected, fused_lora) = dsa_indexed_decode_fused(
            &q_low,
            &compact_latents,
            &selected,
            requested_selected,
            dense_all,
            &kv_weight,
            &kv_scales,
            &kv_biases,
            group_size,
            scale,
            stream,
            capture_fused_lora || (native_decode && capture_stages),
        )?;
        let output = projected
            .transpose_axes(&[0, 2, 1, 3])
            .and_then(|out| out.contiguous())
            .ok()?;
        return Some((
            output,
            if capture_stages { Some(q_low) } else { None },
            fused_lora,
        ));
    }

    let lora = if wide_mtp_window {
        dsa_indexed_lora_wide_native(&q, &q_low, &compact_latents, &selected, scale, stream)?
    } else {
        native::dsa_indexed_lora_short(
            &q_low,
            &compact_latents,
            &selected,
            n_selected,
            selected_valid_prefix,
            dense_all,
            scale,
            stream,
        )?
    };
    lora.eval_enqueue().ok()?;

    let n_rows = batches.checked_mul(n_queries)?.checked_mul(heads)?;
    let n_rows_i32 = i32::try_from(n_rows).ok()?;
    let lora_rows = lora.reshape(&[n_rows_i32, 512]).ok()?;
    let projected = dsa_value_project_rows(
        &lora_rows, &kv_weight, &kv_scales, &kv_biases, group_size, heads, stream,
    )?
    .reshape(&[batches as i32, n_queries as i32, heads as i32, 256])
    .ok()?;
    if prefill_encoder_boundaries {
        stream.runtime().commands.flush_encoder();
    }
    let output = projected
        .transpose_axes(&[0, 2, 1, 3])
        .and_then(|out| out.contiguous())
        .ok()?;
    Some((output, Some(q_low), Some(lora)))
}
