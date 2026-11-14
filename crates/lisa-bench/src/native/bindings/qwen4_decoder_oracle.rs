#![cfg(test)]
use super::*;
pub(crate) trait OriginalDecoder {
    fn original_forward(
        &mut self,
        residual: &Array,
        pending_out: Option<&Array>,
        pending_inject: Option<&Array>,
        cache: Option<&mut LayerCache>,
        rope: &Rotary,
        offset: usize,
        positions: &Array,
        ple_args: Option<(&[Vec<i64>], &[Vec<i64>])>,
        capture: bool,
    ) -> anyhow::Result<(Array, Array, Array)>;
}
impl OriginalDecoder for DecoderLayer {
    /// Forward one layer over the hyper stream.
    ///
    /// `token_rows`/`previous_context` feed the PLE hash (host side).
    #[allow(clippy::too_many_arguments)]
    fn original_forward(
        &mut self,
        residual: &Array,
        pending_out: Option<&Array>,
        pending_inject: Option<&Array>,
        mut cache: Option<&mut LayerCache>,
        rope: &Rotary,
        offset: usize,
        positions: &Array,
        ple_args: Option<(&[Vec<i64>], &[Vec<i64>])>,
        capture: bool,
    ) -> anyhow::Result<(Array, Array, Array)> {
        let attn_scale = self
            .attn_hc
            .norm_scale_q()
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        let mlp_scale = self
            .mlp_hc
            .norm_scale_q()
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        let hc = self.attn_hc.hc_count as i32;
        let hidden = (self.attn_hc.hidden / self.attn_hc.hc_count) as i32;
        let eps = self.attn_hc.hc_norm.eps;
        let (b, s) = (residual.dim(0), residual.dim(1));
        let rows = b * s;
        let stream = lisa_metal::Stream::thread_local_or_default();
        let w = hc * hidden;
        let run_in = |res: &Array,
                      out: Option<&Array>,
                      inj: Option<&Array>,
                      scale: &Array|
         -> anyhow::Result<(Array, Array)> {
            // Wide windows use the `_wide` variant (bit-identical, ~20x fewer
            // threads per row).
            let r = lisa_metal::kernels::inject_norm(
                res, out, inj, scale, hc, hidden, rows, false, eps, &stream,
            );
            let (st, nm) = r.ok_or_else(|| anyhow::anyhow!("inject_norm kernel unavailable"))?;
            Ok((st.reshape(&[b, s, w])?, nm.reshape(&[b, s, w])?))
        };

        // PLE adds into the stream BEFORE this layer's attention mixer: the
        // engine materializes the stream, adds the PLE, then norms again.
        let (stream_v, normed) = match (self.ple.as_mut(), ple_args) {
            (Some(ple), Some((token_rows, previous_context))) => {
                let (s1, _) = run_in(residual, pending_out, pending_inject, &attn_scale)?;
                let ple_out = match cache.as_deref_mut() {
                    Some(c) => {
                        let (slot, cap) = c.ple_conv_mut();
                        ple.forward(&s1, token_rows, previous_context, slot, capture, cap)?
                    }
                    None => ple.forward(
                        &s1,
                        token_rows,
                        previous_context,
                        &mut None,
                        capture,
                        &mut None,
                    )?,
                };
                let s2 = s1.add(&ple_out)?;
                run_in(&s2, None, None, &attn_scale)?
            }
            _ => run_in(residual, pending_out, pending_inject, &attn_scale)?,
        };

        let (input, attn_inject) = self.attn_hc.mix_from_normed(&normed, true)?;
        let attended = match (&mut self.block, cache) {
            (Block::Linear(gdn), Some(LayerCache::Linear(c))) => {
                gdn.forward(&input, Some(c), capture)?
            }
            (Block::Linear(gdn), None) => gdn.forward(&input, None, capture)?,
            (Block::Attention(attn), Some(LayerCache::Full(c))) => {
                attn.forward(&input, rope, Some(c), offset, positions)?
            }
            (Block::Attention(attn), None) => {
                attn.forward(&input, rope, None, offset, positions)?
            }
            _ => anyhow::bail!("cache kind does not match layer kind"),
        };

        let (stream_v2, normed2) =
            run_in(&stream_v, Some(&attended), Some(&attn_inject), &mlp_scale)?;
        let (input2, mlp_inject) = self.mlp_hc.mix_from_normed(&normed2, true)?;
        let out = self.mlp.forward(&input2)?;
        Ok((stream_v2, out, mlp_inject))
    }
}
