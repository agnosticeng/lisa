#![cfg(test)]
use super::*;
use crate::core::norm::RmsNormExecution;
use crate::core::quant::QuantizedLinearExecution;
use lisa_metal::ops::indexing::{Ellipsis, IndexOp};
use lisa_metal::{Dtype, ops};
pub(crate) trait OriginalPle {
    fn original_short_conv(
        &mut self,
        x: &Array,
        conv_state: &mut Option<Array>,
    ) -> lisa_metal::error::Result<Array>;
    fn original_forward(
        &mut self,
        hidden: &Array,
        ids: &[Vec<i64>],
        previous_context: &[Vec<i64>],
        conv_state: &mut Option<Array>,
        capture: bool,
        capture_full: &mut Option<Array>,
    ) -> anyhow::Result<Array>;
}
impl OriginalPle for PleLayer {
    fn original_short_conv(
        &mut self,
        x: &Array,
        conv_state: &mut Option<Array>,
    ) -> lisa_metal::error::Result<Array> {
        let s = x.dim(1) as usize;
        let n = self.short_conv_state_length;
        let state = match conv_state {
            Some(st) => st.clone(),
            None => ops::zeros::<half::bf16>(&[x.dim(0), n as i32, x.dim(-1)])?,
        };
        let full = ops::concatenate(&[&state, x], 1)?;
        // keep the last n rows as the new state
        let tail = full
            .index((Ellipsis, (full.dim(1) - n as i32).., ..))
            .contiguous()?;
        *conv_state = Some(tail);
        // conv over the last n+S rows (= all of full), kernel 4, dilation 3
        let start = full.dim(1) - (n + s) as i32;
        let window = full.index((Ellipsis, start.., ..)).contiguous()?;
        let conv = ops::conv1d(
            &window,
            &self.conv1d_weight,
            None,
            None,
            Some(self.dilation as i32),
            Some(window.dim(-1)),
        )?;
        let conv = lisa_metal::nn::silu(&conv)?;
        Ok(conv)
    }

    /// hidden: the full hyper stream [B, S, wide]; ids/ctx for the hash.
    fn original_forward(
        &mut self,
        hidden: &Array,
        ids: &[Vec<i64>],
        previous_context: &[Vec<i64>],
        conv_state: &mut Option<Array>,
        capture: bool,
        capture_full: &mut Option<Array>,
    ) -> anyhow::Result<Array> {
        let b = hidden.dim(0);
        let s = hidden.dim(1);
        let h = self.hidden_size as i32;

        let embedded = self.embedding.forward(ids, previous_context)?;
        let embedded = embedded
            .as_dtype(hidden.dtype())
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        self.last_embed = Some(embedded.clone());

        // Three fused kernels replace the ~22-launch op chain (see the engine's
        // `TrackFastPLE.swift`); the reduction between prod and gated stays MLX.
        let fused = true;
        if fused {
            let hc = self.hc_count as i32;
            let hidden_i = self.hidden_size as i32;
            let w = hc * hidden_i;
            let stream = lisa_metal::Stream::thread_local_or_default();
            let eps1 = Array::from_f32(self.norm_key.eps)
                .as_dtype(Dtype::Bfloat16)
                .map_err(|e| anyhow::anyhow!("{e}"))?
                .reshape(&[1])?;
            let key_raw = self.key_proj.forward(&embedded)?;
            let value = self.value_proj.forward(&embedded)?;
            if let Some(prod) = lisa_metal::kernels::ple_prod(
                &key_raw,
                hidden,
                &self.norm_key.weight,
                &self.norm_query.weight,
                &eps1,
                hc,
                hidden_i,
                &stream,
            ) {
                // g0 = the sum over the last axis (MLX's own reduction).
                let g0 = prod.reshape(&[b, s, hc, hidden_i])?.sum_axis(-1, None)?;
                let divisor = Array::from_f32((self.hidden_size as f32).sqrt())
                    .as_dtype(Dtype::Bfloat16)
                    .map_err(|e| anyhow::anyhow!("{e}"))?
                    .reshape(&[1])?;
                let floor = Array::from_f32(1e-6)
                    .as_dtype(Dtype::Bfloat16)
                    .map_err(|e| anyhow::anyhow!("{e}"))?
                    .reshape(&[1])?;
                if let Some((gated, normed)) = lisa_metal::kernels::ple_gated(
                    &g0,
                    &value,
                    &self.norm_conv.weight,
                    &divisor,
                    &floor,
                    &eps1,
                    hc,
                    hidden_i,
                    &stream,
                ) {
                    self.last_gated = Some(gated.clone());
                    let n = self.short_conv_state_length;
                    let state = match conv_state.as_ref() {
                        Some(st) => st.clone(),
                        None => ops::zeros::<half::bf16>(&[b, n as i32, w])?,
                    };
                    let full = ops::concatenate(&[&state, &normed], 1)?;
                    if capture {
                        *capture_full = Some(full.clone());
                    }
                    let tail = full
                        .index((Ellipsis, (full.dim(1) - n as i32).., ..))
                        .contiguous()?;
                    *conv_state = Some(tail);
                    if let Some(out) = lisa_metal::kernels::ple_conv(
                        &full,
                        &self.conv1d_weight,
                        &gated,
                        self.dilation as i32,
                        &stream,
                    ) {
                        return Ok(out);
                    }
                }
            }
        }

        // S == 1: the engine's PLE fusion (`track_ple_prepare_fuse2` +
        // `track_ple_convolution_fuse2`).
        // The fuse2 kernels declare B=1 outputs, so batched windows take the
        // generic path.
        if b == 1 && s == 1 && self.hidden_size == 2560 && self.hc_count == 4 {
            let key = self.key_proj.forward(&embedded)?;
            let value = self.value_proj.forward(&embedded)?;
            if key.dim(-1) == 10240
                && value.dim(-1) == 2560
                && hidden.dim(-1) == 10240
                && self.dilation == 3
                && self.conv1d_weight.shape() == [10240, 4, 1]
            {
                let state = match conv_state.as_ref() {
                    Some(st) => st.clone(),
                    None => ops::zeros::<half::bf16>(&[b, 9, 10240])?,
                };
                let stream = lisa_metal::Stream::thread_local_or_default();
                if let Some((full, out)) = lisa_metal::kernels::ple_fuse2(
                    &key,
                    hidden,
                    &value,
                    &self.norm_key.weight,
                    &self.norm_query.weight,
                    &self.norm_conv.weight,
                    &state,
                    &self.conv1d_weight,
                    self.norm_key.eps,
                    &stream,
                ) {
                    *conv_state = Some(full.index((.., 1..10, ..)).contiguous()?);
                    self.last_gated = None;
                    return Ok(out.reshape(&[b, s, self.hidden_size as i32])?);
                }
            }
        }

        let key_flat = self.norm_key.forward(&self.key_proj.forward(&embedded)?)?;
        let key = key_flat.reshape(&[b, s, self.hc_count as i32, h])?;
        let value = self.value_proj.forward(&embedded)?;
        let query_flat = self.norm_query.forward(hidden)?;
        let query = query_flat.reshape(&[b, s, self.hc_count as i32, h])?;

        let mut gate = key.multiply(&query)?.sum_axis(-1, Some(true))?;
        gate = gate / crate::core::norm::bf16_scalar((self.hidden_size as f32).sqrt());
        // Floor scalar built IN THE GATE'S DTYPE (bf16) — an f32 scalar here
        // would silently promote the whole stream to f32.
        let floor = Array::from_f32(1e-6)
            .as_dtype(gate.dtype())
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        gate = ops::sqrt(&ops::maximum(ops::abs(&gate)?, &floor)?)?.multiply(&ops::sign(&gate)?)?;

        let sg = ops::sigmoid(&gate)?;
        let gated = sg.multiply(&value.expand_dims(-2)?)?;
        let gated = gated.reshape(&[b, s, -1])?;
        self.last_gated = Some(gated.clone());
        let conv = self.original_short_conv(&self.norm_conv.forward(&gated)?, conv_state)?;
        gated.add(&conv).map_err(|e| anyhow::anyhow!("{e}"))
    }
}
