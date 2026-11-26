#![cfg(test)]
use super::*;
use crate::core::quant::QuantizedEmbeddingExecution;
pub(super) fn rms(x: &Array, w: &Array, eps: f32) -> anyhow::Result<Array> {
    let xf = x.as_dtype(Dtype::Float32)?;
    let ms = xf.multiply(&xf)?.contiguous()?.mean_axis(2, true)?; // [B,T,1]
    let n = xf.multiply(&ms.add(Array::from_f32_as(eps, Dtype::Float32))?.rsqrt()?)?;
    Ok(n.multiply(&w.as_dtype(Dtype::Float32)?)?)
}

pub(super) fn final_rms(x: &Array, w: &Array, eps: f32) -> anyhow::Result<Array> {
    let stream = lisa_metal::Stream::thread_local_or_default();
    lisa_metal::kernels::sparse::kernels::ds4_rms_norm(x, w, eps, &stream).ok_or_else(|| {
        anyhow::anyhow!(
            "GLM final DS4-order RMSNorm declined for shape {:?}",
            x.shape()
        )
    })
}

pub(super) fn final_hc_mean(state: &Array) -> anyhow::Result<Array> {
    let shape = state.shape();
    anyhow::ensure!(
        shape.len() == 4 && shape[2] == 4,
        "GLM final mHC collapse expects [B,T,4,H], got {shape:?}"
    );
    let weight_count = usize::try_from(shape[0] * shape[1] * shape[2])?;
    let weights = Array::from_slice(
        &vec![0.25f32; weight_count],
        &[shape[0], shape[1], shape[2]],
    );
    let stream = lisa_metal::Stream::thread_local_or_default();
    lisa_metal::kernels::sparse::kernels::hc_weighted_sum(state, &weights, &stream)
        .ok_or_else(|| anyhow::anyhow!("GLM final DS4-order mHC collapse declined"))
}

pub(super) fn mtp_rms(x: &Array, w: &Array, eps: f32) -> anyhow::Result<Array> {
    let stream = lisa_metal::Stream::thread_local_or_default();
    lisa_metal::kernels::sparse::kernels::mtp_rms_norm(x, w, eps, &stream).ok_or_else(|| {
        anyhow::anyhow!(
            "GLM MTP ds4-order RMSNorm declined for shape {:?}",
            x.shape()
        )
    })
}

pub(super) fn draft_inputs(
    embed: &QuantizedEmbedding,
    target_multi: &Array,
    eh: Array,
    enorm: Array,
    hnorm: Array,
    h: usize,
    streams: usize,
    next_token: u32,
    epsilon: f32,
) -> anyhow::Result<(Array, Array, Array)> {
    let ids = Array::from_slice(&[next_token as i32], &[1i32, 1]);
    let embedded = embed.forward(&ids)?.as_dtype(Dtype::Float32)?;
    let embedded = mtp_rms(&embedded, &enorm, epsilon)?;

    // The reference reduces target mHC streams with uniform 1/C weights via its
    // fused weighted-sum kernel; preserve that kernel and accumulation order.
    anyhow::ensure!(streams == 4, "glm53 MTP mHC width must be four");
    let multi =
        target_multi
            .as_dtype(Dtype::Float32)?
            .reshape(&[1, 1, streams as i32, h as i32])?;
    let weights = Array::from_slice(&[0.25f32; 4], &[1, 1, 4]);
    let stream = lisa_metal::Stream::thread_local_or_default();
    let hidden =
        lisa_metal::kernels::sparse::kernels::hc_weighted_sum(&multi, &weights, &stream)
            .ok_or_else(|| anyhow::anyhow!("glm53 MTP target mHC reduction kernel declined"))?;
    let hidden = mtp_rms(&hidden, &hnorm, epsilon)?;

    let joined = Array::concatenate(&[&embedded, &hidden], 2)?;
    let rows = joined.reshape(&[1, joined.dim(2)])?;
    let projected = if eh.dtype() == Dtype::Bfloat16 {
        let stream = lisa_metal::Stream::thread_local_or_default();
        lisa_metal::kernels::sparse::kernels::bf16_gemv(&eh, &rows, &stream)
            .ok_or_else(|| anyhow::anyhow!("GLM MTP BF16 eh_proj GEMV declined"))?
    } else {
        let weight = eh.transpose_axes(&[1, 0])?.as_dtype(Dtype::Float32)?;
        mm(&joined, &weight)?
    };
    let x = projected.reshape(&[1, 1, h as i32])?;

    Ok((embedded, hidden, x))
}

pub trait TowerOracle2 {
    fn forward_ragged_inner_oracle(
        &mut self,
        ids: &Array,
        per_row_caches: &mut [Vec<LayerCache>],
    ) -> anyhow::Result<(Array, Array)>;
}
impl TowerOracle2 for Tower {
    fn forward_ragged_inner_oracle(
        &mut self,
        ids: &Array,
        per_row_caches: &mut [Vec<LayerCache>],
    ) -> anyhow::Result<(Array, Array)> {
        let (b, s) = (ids.dim(0) as usize, ids.dim(1) as usize);
        anyhow::ensure!(
            b == per_row_caches.len() && s == 1,
            "GLM ragged decode expects one token and cache per row, got ids {:?}, caches {}",
            ids.shape(),
            per_row_caches.len()
        );
        anyhow::ensure!(b > 1, "GLM ragged decode requires at least two rows");
        anyhow::ensure!(
            per_row_caches
                .iter()
                .all(|row| row.len() == self.layers.len()),
            "GLM ragged decode cache depth does not match the tower"
        );
        // Validate the complete cohort before updating any row's cache.
        for caches in per_row_caches.iter() {
            self.check_context_span(caches, 1)?;
        }
        // Preserve-dispatch captures retain their Arrays in Capture::pending.
        // Their buffer leases remain live, so ordered reuse can recycle only
        // unrelated dead scratch while those snapshots wait for readback.
        let _scratch = self
            .capture
            .as_ref()
            .is_none_or(|c| c.layers.is_empty() || c.preserves_dispatch())
            .then(lisa_metal::runtime::enter_ordered_gpu_reuse_scope);
        let dense_compact_limit = self
            .config
            .index_topk
            .saturating_add(self.config.index_kpool.saturating_sub(1));
        let max_position = per_row_caches
            .iter()
            .flat_map(|row| row.iter())
            .filter_map(|cache| match cache {
                LayerCache::Full(cache) => Some(cache.offset),
                LayerCache::Linear(_) | LayerCache::Native(_) => None,
            })
            .max()
            .unwrap_or(0);
        if !ds4_native_session_batch_position_supported(b, max_position, dense_compact_limit) {
            // Match glm53_graph_native_session_batch_supported: the reference declines
            // its native cohort kernel at this row count or sparse-context
            // boundary and evaluates each session through the serial graph.
            let mut mixed_rows = Vec::with_capacity(b);
            let mut multi_rows = Vec::with_capacity(b);
            for (row, caches) in per_row_caches.iter_mut().enumerate() {
                let token = narrow(ids, 0, row, 1)?;
                let (mixed, multi) = self.forward_inner(&token, caches)?;
                mixed_rows.push(mixed);
                multi_rows.push(multi);
            }
            let mixed_refs: Vec<&Array> = mixed_rows.iter().collect();
            let multi_refs: Vec<&Array> = multi_rows.iter().collect();
            return Ok((
                Array::concatenate(&mixed_refs, 0)?,
                Array::concatenate(&multi_refs, 0)?,
            ));
        }
        let h = self.config.hidden_size;
        let c = self.config.hc_mult;
        let x = self.embed.forward(ids)?.as_dtype(Dtype::Float32)?;
        let mut state = x
            .expand_dims(2)?
            .broadcast_to(&[b as i32, 1, c as i32, h as i32])?
            .contiguous()?;

        for (li, layer) in self.layers.iter_mut().enumerate() {
            let (_xc, x_in, post, comb) = if let Some((xc, normed, post, comb)) = layer
                .hc_attn
                .split_normed(&state, &layer.in_norm, self.config.rms_norm_eps)?
            {
                (xc, normed, post, comb)
            } else {
                let (xc, post, comb) = layer.hc_attn.split(&state)?;
                let normed = rms(&xc, &layer.in_norm, self.config.rms_norm_eps)?;
                (xc, normed, post, comb)
            };
            let capture = self.capture.as_ref().filter(|capture| capture.wants(li));
            let step = INC_STEP.load(std::sync::atomic::Ordering::Relaxed);
            if let Some(capture) = capture {
                capture.store(&format!("L{li}.dec_rms_in_s{step}"), &x_in);
            }
            if li == 0 {
                INC_STEP.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
            let attended = match &layer.attn {
                Attn::Kda(kda) => {
                    let mut row_caches = Vec::with_capacity(b);
                    for (row, caches) in per_row_caches.iter_mut().enumerate() {
                        match &mut caches[li] {
                            LayerCache::Linear(cache) => row_caches.push(cache),
                            _ => anyhow::bail!(
                                "glm53: ragged KDA cache kind mismatch at layer {li}, row {row}"
                            ),
                        }
                    }
                    kda.forward_ragged(&x_in, &mut row_caches, self.capture.as_ref(), li)?
                }
                Attn::Dsa(dsa) => {
                    let mut row_caches = Vec::with_capacity(b);
                    for (row, caches) in per_row_caches.iter_mut().enumerate() {
                        match &mut caches[li] {
                            LayerCache::Full(cache) => row_caches.push(cache),
                            _ => anyhow::bail!(
                                "glm53: ragged DSA cache kind mismatch at layer {li}, row {row}"
                            ),
                        }
                    }
                    dsa.forward_ragged(&x_in, &mut row_caches, self.capture.as_ref(), li)?
                }
            };
            if let Some(capture) = capture {
                capture.store(&format!("L{li}.dec_attn_out_s{step}"), &attended);
            }
            state = layer.hc_attn.expand(&attended, &state, &post, &comb)?;
            if let Some(capture) = capture {
                capture.store(&format!("L{li}.state_after_attn"), &state);
                capture.store(&format!("L{li}.dec_state_after_attn_s{step}"), &state);
            }

            let (_xc2, x_in2, post2, comb2) = if let Some((xc, normed, post, comb)) =
                layer
                    .hc_ffn
                    .split_normed(&state, &layer.post_norm, self.config.rms_norm_eps)?
            {
                (xc, normed, post, comb)
            } else {
                let (xc, post, comb) = layer.hc_ffn.split(&state)?;
                let normed = rms(&xc, &layer.post_norm, self.config.rms_norm_eps)?;
                (xc, normed, post, comb)
            };
            if let Some(capture) = capture {
                capture.store(&format!("L{li}.mhc_ffn_normed"), &x_in2);
            }
            let ffn = match &layer.mlp {
                Mlp::Dense(dense) => {
                    dense.forward_with_capture(&x_in2, self.capture.as_ref(), li)?
                }
                Mlp::Sparse(moe) => moe.forward_with_capture(&x_in2, self.capture.as_ref(), li)?,
            };
            if let Some(capture) = capture {
                capture.store(&format!("L{li}.dec_moe_out_s{step}"), &ffn);
            }
            state = layer.hc_ffn.expand(&ffn, &state, &post2, &comb2)?;
            if let Some(capture) = capture {
                capture.store(&format!("L{li}.state_after_ffn"), &state);
                capture.store(&format!("L{li}.dec_state_after_ffn_s{step}"), &state);
            }
        }

        let mixed = final_hc_mean(&state)?.reshape(&[b as i32, 1, h as i32])?;
        let mixed = final_rms(&mixed, &self.final_norm, self.config.rms_norm_eps)?;
        let multi = state.reshape(&[b as i32, 1, (c * h) as i32])?;
        Ok((mixed, multi))
    }
}

pub trait TowerOracle1 {
    fn forward_inner_until_oracle(
        &mut self,
        ids: &Array,
        caches: &mut Vec<LayerCache>,
        stop_after_layer: Option<usize>,
    ) -> anyhow::Result<(Array, Array)>;
}
impl TowerOracle1 for Tower {
    fn forward_inner_until_oracle(
        &mut self,
        ids: &Array,
        caches: &mut Vec<LayerCache>,
        stop_after_layer: Option<usize>,
    ) -> anyhow::Result<(Array, Array)> {
        let _trace_forward = lisa_metal::trace::span("glm53.forward");
        let (b, s) = (ids.dim(0) as usize, ids.dim(1) as usize);
        self.check_context_span(caches, s)?;
        let _scratch = self
            .capture
            .as_ref()
            .is_none_or(|c| c.layers.is_empty() || c.preserves_dispatch())
            .then(lisa_metal::runtime::enter_ordered_gpu_reuse_scope);
        let forward_position = caches.iter().find_map(|cache| match cache {
            LayerCache::Full(cache) => Some(cache.offset),
            LayerCache::Linear(_) | LayerCache::Native(_) => None,
        });
        let prefill_mode = self.prefill_mode;
        let h = self.config.hidden_size;
        let c = self.config.hc_mult;
        if let Some(layer) = stop_after_layer {
            anyhow::ensure!(
                layer < self.layers.len(),
                "GLM prefix layer {layer} exceeds {} layers",
                self.layers.len()
            );
        }
        let x = {
            let _trace_embed = lisa_metal::trace::span("glm53.embed");
            let x = self.embed.forward(ids)?.as_dtype(Dtype::Float32)?;
            if let Some(v) = &self.vision {
                v.splice(x, forward_position.unwrap_or(0))?
            } else {
                x
            } // [B,S,H]
        };
        if let Some(c) = &self.capture {
            c.store("embed_out", &x);
        }
        let mut state = x
            .expand_dims(2)?
            .broadcast_to(&[(b) as i32, (s) as i32, (c) as i32, (h) as i32])?
            .contiguous()?; // [B,S,C,H]

        let layer_count = self.layers.len();
        for (li, layer) in self.layers.iter_mut().enumerate() {
            let _trace_layer = lisa_metal::trace::span_detail("glm53.layer", li as u64 + 1);
            if lisa_metal::trace::enabled_for("glm53.memory") {
                eprintln!(
                    "[glm53-memory] layer={li} rows={} private_bytes={}",
                    b * s,
                    lisa_metal::memory::private_allocated_memory()?,
                );
            }
            // Attention half.
            if let Some(cp) = &self.capture
                && cp.wants(li)
            {
                cp.store(&format!("L{li}.mhc_attn_in"), &state);
            }
            let capture_attn_mixes = self
                .capture
                .as_ref()
                .is_some_and(|capture| capture.wants(li) && !capture.preserves_dispatch());
            let (xc, x_in, post, comb, attn_mixes, attn_split) = {
                let _trace_mhc = lisa_metal::trace::span("glm53.mhc.attn_input");
                if capture_attn_mixes {
                    if let Some((xc, normed, post, comb, mixes, split)) = layer
                        .hc_attn
                        .split_normed_with_mixes(&state, &layer.in_norm, self.config.rms_norm_eps)?
                    {
                        (xc, normed, post, comb, Some(mixes), Some(split))
                    } else {
                        let (xc, post, comb) = layer.hc_attn.split(&state)?;
                        let normed = rms(&xc, &layer.in_norm, self.config.rms_norm_eps)?;
                        (xc, normed, post, comb, None, None)
                    }
                } else if let Some((xc, normed, post, comb)) =
                    layer
                        .hc_attn
                        .split_normed(&state, &layer.in_norm, self.config.rms_norm_eps)?
                {
                    (xc, normed, post, comb, None, None)
                } else {
                    let (xc, post, comb) = layer.hc_attn.split(&state)?;
                    let normed = rms(&xc, &layer.in_norm, self.config.rms_norm_eps)?;
                    (xc, normed, post, comb, None, None)
                }
            };
            if let Some(cp) = &self.capture
                && cp.wants(li)
            {
                cp.store(&format!("L{li}.mhc_attn_collapsed"), &xc);
                cp.store(&format!("L{li}.mhc_attn_post"), &post);
                cp.store(&format!("L{li}.mhc_attn_comb"), &comb);
                cp.store(&format!("L{li}.mhc_attn_normed"), &x_in);
                if let Some(mixes) = &attn_mixes {
                    cp.store(&format!("L{li}.mhc_attn_mixes"), mixes);
                }
                if let Some(split) = &attn_split {
                    cp.store(&format!("L{li}.mhc_attn_split"), split);
                }
            }
            // incremental-decode probe: norm of every stage, every layer
            let step = if li == 0 {
                INC_STEP.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
            } else {
                INC_STEP.load(std::sync::atomic::Ordering::Relaxed)
            };
            let capture_mode = if x_in.dim(1) == 1 { "dec" } else { "rep" };
            if let Some(cp) = &self.capture
                && cp.wants(li)
            {
                let t2 = if x_in.dim(1) == 1 { "dec" } else { "rep" };
                cp.store(&format!("L{li}.{t2}_rms_in_s{step}"), &x_in);
            }
            let attended = {
                let _trace_attn = lisa_metal::trace::span("glm53.attention");
                match (&layer.attn, &mut caches[li]) {
                    (Attn::Kda(k), LayerCache::Linear(g)) => {
                        if prefill_mode {
                            k.forward_prefill(
                                &x_in,
                                &mut g.conv,
                                &mut g.ssm,
                                self.capture.as_ref(),
                                li,
                            )?
                        } else if let Some(position) = forward_position {
                            k.forward_at_position(
                                &x_in,
                                &mut g.conv,
                                &mut g.ssm,
                                self.capture.as_ref(),
                                li,
                                position,
                            )?
                        } else {
                            k.forward(&x_in, &mut g.conv, &mut g.ssm, self.capture.as_ref(), li)?
                        }
                    }
                    (Attn::Dsa(d), LayerCache::Full(f)) => {
                        d.forward(&x_in, f, self.capture.as_ref(), li, prefill_mode)?
                    }
                    _ => anyhow::bail!("glm53: cache kind mismatch at layer {li}"),
                }
            };
            if let Some(cp) = &self.capture
                && cp.wants(li)
            {
                let t2 = if x_in.dim(1) == 1 { "dec" } else { "rep" };
                cp.store(&format!("L{li}.{t2}_attn_out_s{step}"), &attended);
                cp.store(&format!("L{li}.mhc_attn_xc"), &xc);
            }
            state = {
                let _trace_expand = lisa_metal::trace::span("glm53.mhc.attn_expand");
                layer.hc_attn.expand(&attended, &state, &post, &comb)?
            };
            if let Some(cp) = &self.capture
                && cp.wants(li)
            {
                cp.store(&format!("L{li}.state_after_attn"), &state);
                cp.store(
                    &format!("L{li}.{capture_mode}_state_after_attn_s{step}"),
                    &state,
                );
            }

            // FFN half.
            let capture_ffn_mixes = self
                .capture
                .as_ref()
                .is_some_and(|capture| capture.wants(li) && !capture.preserves_dispatch());
            let (xc2, x_in2, post2, comb2, ffn_mixes, ffn_split) = {
                let _trace_mhc = lisa_metal::trace::span("glm53.mhc.ffn_input");
                if capture_ffn_mixes {
                    if let Some((xc, normed, post, comb, mixes, split)) =
                        layer.hc_ffn.split_normed_with_mixes(
                            &state,
                            &layer.post_norm,
                            self.config.rms_norm_eps,
                        )?
                    {
                        (xc, normed, post, comb, Some(mixes), Some(split))
                    } else {
                        let (xc, post, comb) = layer.hc_ffn.split(&state)?;
                        let normed = rms(&xc, &layer.post_norm, self.config.rms_norm_eps)?;
                        (xc, normed, post, comb, None, None)
                    }
                } else if let Some((xc, normed, post, comb)) =
                    layer
                        .hc_ffn
                        .split_normed(&state, &layer.post_norm, self.config.rms_norm_eps)?
                {
                    (xc, normed, post, comb, None, None)
                } else {
                    let (xc, post, comb) = layer.hc_ffn.split(&state)?;
                    let normed = rms(&xc, &layer.post_norm, self.config.rms_norm_eps)?;
                    (xc, normed, post, comb, None, None)
                }
            };
            if let Some(cp) = &self.capture
                && cp.wants(li)
            {
                cp.store(&format!("L{li}.mhc_ffn_post"), &post2);
                cp.store(&format!("L{li}.mhc_ffn_comb"), &comb2);
                cp.store(&format!("L{li}.mhc_ffn_normed"), &x_in2);
                if let Some(mixes) = &ffn_mixes {
                    cp.store(&format!("L{li}.mhc_ffn_mixes"), mixes);
                }
                if let Some(split) = &ffn_split {
                    cp.store(&format!("L{li}.mhc_ffn_split"), split);
                }
            }
            let m = {
                let _trace_ffn = lisa_metal::trace::span("glm53.mlp");
                match &layer.mlp {
                    Mlp::Dense(d) => d.forward_with_capture(&x_in2, self.capture.as_ref(), li)?,
                    Mlp::Sparse(moe) => {
                        moe.forward_with_capture(&x_in2, self.capture.as_ref(), li)?
                    }
                }
            };
            if let Some(cp) = &self.capture
                && cp.wants(li)
            {
                let t2 = if x_in.dim(1) == 1 { "dec" } else { "rep" };
                cp.store(&format!("L{li}.{t2}_moe_out_s{step}"), &m);
                cp.store(&format!("L{li}.mhc_ffn_xc2"), &xc2);
            }
            state = {
                let _trace_expand = lisa_metal::trace::span("glm53.mhc.ffn_expand");
                layer.hc_ffn.expand(&m, &state, &post2, &comb2)?
            };
            if let Some(cp) = &self.capture
                && cp.wants(li)
            {
                cp.store(&format!("L{li}.state_after_ffn"), &state);
                cp.store(
                    &format!("L{li}.{capture_mode}_state_after_ffn_s{step}"),
                    &state,
                );
            }
            // The reference submits indexed GLM decode work every four layers without
            // waiting for completion. This lets Metal execute early layers
            // while the host encodes the remainder of the token. A single
            // command buffer for the whole 45-layer forward adds a large
            // submission bubble even though both paths are otherwise async.
            if b == 1
                && s == 1
                && !prefill_mode
                && self
                    .capture
                    .as_ref()
                    .is_none_or(|capture| capture.preserves_dispatch())
                && (li + 1) % 4 == 0
                && li + 1 < layer_count
            {
                lisa_metal::Stream::thread_local_or_default()
                    .runtime()
                    .commands
                    .flush()?;
            }
            // The reference's final prefill chunk submits every layer above eight
            // rows and drains at each sixteen-layer boundary. Intermediate
            // chunks retain one command stream until the chunk-end drain.
            if prefill_mode
                && (s > DS4_GLM53_PREFILL_CHUNK_TOKENS || (self.prefill_final_chunk && s > 8))
                && li + 1 < layer_count
                && self.capture.as_ref().is_none_or(|c| c.preserves_dispatch())
            {
                let stream = lisa_metal::Stream::thread_local_or_default();
                if (li + 1) % 16 == 0 {
                    let _span = lisa_metal::trace::span("glm53.prefill.layer_drain");
                    stream.runtime().commands.flush_and_wait()?;
                } else {
                    stream.runtime().commands.flush()?;
                }
            }
            if stop_after_layer == Some(li) {
                let mixed = state
                    .contiguous()?
                    .mean_axis(2, true)?
                    .reshape(&[b as i32, s as i32, h as i32])?;
                let multi = state.reshape(&[b as i32, s as i32, (c * h) as i32])?;
                return Ok((mixed, multi));
            }
        }

        if prefill_mode {
            let _span = lisa_metal::trace::span("glm53.prefill.chunk_drain");
            lisa_metal::Stream::thread_local_or_default()
                .runtime()
                .commands
                .flush_and_wait()?;
        }
        // Final: mean over the streams, RMS, head.
        let mixed = final_hc_mean(&state)?;
        let mixed = final_rms(&mixed, &self.final_norm, self.config.rms_norm_eps)?;
        if let Some(c) = &self.capture {
            let mode = if s == 1 { "dec" } else { "rep" };
            let step = INC_STEP.load(std::sync::atomic::Ordering::Relaxed);
            c.store(&format!("final_{mode}_mixed_s{step}"), &mixed);
        }
        let multi = state.reshape(&[(b) as i32, (s) as i32, (c * h) as i32])?;
        Ok((mixed, multi))
    }
}

pub trait MtpBlockOracle0 {
    fn forward_oracle(
        &self,
        config: &ModelConfig,
        embed: &QuantizedEmbedding,
        lm_head: &QuantizedLinear,
        tape: &mut IndexerTape,
        target_multi: &Array,
        next_token: u32,
        position: usize,
        min_position: usize,
    ) -> anyhow::Result<Array>;
}
impl MtpBlockOracle0 for MtpBlock {
    fn forward_oracle(
        &self,
        config: &ModelConfig,
        embed: &QuantizedEmbedding,
        lm_head: &QuantizedLinear,
        tape: &mut IndexerTape,
        target_multi: &Array,
        next_token: u32,
        position: usize,
        min_position: usize,
    ) -> anyhow::Result<Array> {
        anyhow::ensure!(
            target_multi.dim(0) == 1 && target_multi.dim(1) == 1,
            "glm53 MTP expects one target hidden row"
        );
        let h = config.hidden_size;
        let streams = config.hc_mult;
        anyhow::ensure!(
            target_multi.dim(2) == (h * streams) as i32,
            "glm53 MTP target hidden width mismatch"
        );
        anyhow::ensure!(
            (next_token as usize) < config.vocab_size,
            "glm53 MTP token {next_token} exceeds vocabulary size {}",
            config.vocab_size
        );

        let ids = Array::from_slice(&[next_token as i32], &[1i32, 1]);
        let embedded = embed.forward(&ids)?.as_dtype(Dtype::Float32)?;
        let embedded = mtp_rms(&embedded, &self.enorm, config.rms_norm_eps)?;

        // The reference reduces target mHC streams with uniform 1/C weights via its
        // fused weighted-sum kernel; preserve that kernel and accumulation order.
        anyhow::ensure!(streams == 4, "glm53 MTP mHC width must be four");
        let multi =
            target_multi
                .as_dtype(Dtype::Float32)?
                .reshape(&[1, 1, streams as i32, h as i32])?;
        let weights = Array::from_slice(&[0.25f32; 4], &[1, 1, 4]);
        let stream = lisa_metal::Stream::thread_local_or_default();
        let hidden =
            lisa_metal::kernels::sparse::kernels::hc_weighted_sum(&multi, &weights, &stream)
                .ok_or_else(|| anyhow::anyhow!("glm53 MTP target mHC reduction kernel declined"))?;
        let hidden = mtp_rms(&hidden, &self.hnorm, config.rms_norm_eps)?;

        let joined = Array::concatenate(&[&embedded, &hidden], 2)?;
        let rows = joined.reshape(&[1, joined.dim(2)])?;
        let projected = if self.eh_proj.dtype() == Dtype::Bfloat16 {
            let stream = lisa_metal::Stream::thread_local_or_default();
            lisa_metal::kernels::sparse::kernels::bf16_gemv(&self.eh_proj, &rows, &stream)
                .ok_or_else(|| anyhow::anyhow!("GLM MTP BF16 eh_proj GEMV declined"))?
        } else {
            let weight = self
                .eh_proj
                .transpose_axes(&[1, 0])?
                .as_dtype(Dtype::Float32)?;
            mm(&joined, &weight)?
        };
        let x = projected.reshape(&[1, 1, h as i32])?;

        let attn_in = mtp_rms(&x, &self.attn_norm, config.rms_norm_eps)?;
        let attended = self
            .attn
            .forward_mtp(&attn_in, tape, position, min_position)?;
        let stream = lisa_metal::Stream::thread_local_or_default();
        let (after_attn, ffn_in) = lisa_metal::kernels::sparse::kernels::mtp_add_rms_norm(
            &x,
            &attended,
            &self.ffn_norm,
            config.rms_norm_eps,
            &stream,
        )
        .ok_or_else(|| anyhow::anyhow!("GLM MTP DS4 add+RMSNorm kernel declined"))?;
        let (routed, shared) = self.mlp.forward_parts_with_capture(&ffn_in, None, 0)?;
        let shared =
            shared.ok_or_else(|| anyhow::anyhow!("GLM MTP checkpoint has no shared expert"))?;
        let stream = lisa_metal::Stream::thread_local_or_default();
        let next_sum =
            lisa_metal::kernels::sparse::kernels::mtp_add3(&after_attn, &routed, &shared, &stream)
                .ok_or_else(|| anyhow::anyhow!("GLM MTP DS4 ordered add3 kernel declined"))?;
        let next = mtp_rms(&next_sum, &self.shared_head_norm, config.rms_norm_eps)?;
        // The reference's shared output head uses the same native Q8 dispatcher in
        // target and MTP execution, including the vocabulary-sized reduction.
        let logits = lm_head.forward_glm53_prefill(&next)?;
        Ok(logits)
    }
}
