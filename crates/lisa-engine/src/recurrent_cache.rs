//! Recurrent state snapshots and rollback, executed through model-owned plans.
/// Per-layer recurrent state for linear-attention layers. Qwen GDN keeps its
/// model-specific conv dtype; GLM KDA stores both conv and SSM state in F32.
pub struct GdnCache<T> {
    pub conv: Option<T>,
    pub ssm: Option<T>,
    /// Speculative-verify capture (only set on a capture forward): the SSM
    /// state after EVERY position `[B*S, Hv, Dv, Dk]` and the raw conv input
    /// `[B, K-1+S, convDim]`. Used to roll the recurrent state back to the
    /// accepted prefix.
    pub capture_ssm: Option<T>,
    /// Compact, exact DSpine verify transaction; absent on other paths.
    pub capture_replay: Option<crate::recurrent::Replay<T>>,
    pub capture_conv_input: Option<T>,
    /// Persistent recurrence-output buffer for the capture window. The MTP
    /// verify rebuilds it every round; reusing the buffer avoids a fresh
    /// (large) device allocation per GDN layer per round.
    pub rec_y_buf: Option<T>,
    /// PLE short-conv state `[B, 9, wide]`, per request (it used to live on the
    /// layer, which made it global mutable state shared by every forward).
    pub ple_conv: Option<T>,
    /// Speculative-verify capture of the PLE conv *input* `[B, 9+S, wide]`, so
    /// the PLE state can be rolled back to the accepted prefix exactly like the
    /// GDN conv. Without this the PLE state keeps the rejected drafts and MTP
    /// drifts on free-form text.
    pub capture_ple_full: Option<T>,
}

impl<T: lisa_backend::TensorAccess + 'static> GdnCache<T> {
    pub fn new() -> Self {
        Self {
            conv: None,
            ssm: None,
            capture_ssm: None,
            capture_replay: None,
            capture_conv_input: None,
            rec_y_buf: None,
            ple_conv: None,
            capture_ple_full: None,
        }
    }

    /// Roll the recurrent state back to the state after `n` positions of the
    /// captured verify window (`n >= 1`). No-op when nothing was captured.
    pub fn rollback_to<D: crate::indexer_cache::CacheDevice<Tensor = T>>(
        &mut self,
        context: &crate::execution::DeviceContext<D>,
        n: usize,
        conv_kernel: usize,
    ) -> anyhow::Result<()> {
        let Some(ci) = &self.capture_conv_input else {
            return Ok(());
        };
        let restore = || -> anyhow::Result<Option<(T, T, Option<T>)>> {
            let state = if let Some(replay) = &self.capture_replay {
                replay.restore(context, n)?
            } else if let Some(seq) = &self.capture_ssm {
                crate::recurrent::snapshot_prefix(context, seq, n)?
            } else {
                return Ok(None);
            };
            let history = conv_kernel
                .checked_sub(1)
                .ok_or_else(|| anyhow::anyhow!("invalid convolution kernel size"))?;
            let conv = crate::recurrent::convolution_prefix(context, ci, n, history)?;
            let ple = match (&self.capture_ple_full, &self.ple_conv) {
                (Some(full), Some(side)) => Some(crate::recurrent::convolution_prefix(
                    context,
                    full,
                    n,
                    side.dim(1) as usize,
                )?),
                _ => None,
            };
            Ok(Some((state, conv, ple)))
        };
        if let Some((state, conv, ple)) =
            restore().map_err(|e| anyhow::Error::msg(e.to_string()))?
        {
            self.ssm = Some(state);
            self.conv = Some(conv);
            if let Some(ple) = ple {
                self.ple_conv = Some(ple);
            }
        }
        Ok(())
    }
    /// Clone the recurrent state for a prefix snapshot.
    pub fn snapshot_state(&self) -> (Option<T>, Option<T>, Option<T>) {
        (self.conv.clone(), self.ssm.clone(), self.ple_conv.clone())
    }

    /// Restore the recurrent state from a prefix snapshot.
    pub fn restore_state(&mut self, s: &(Option<T>, Option<T>, Option<T>)) {
        self.conv = s.0.clone();
        self.ssm = s.1.clone();
        self.ple_conv = s.2.clone();
        if self.capture_replay.take().is_some() {
            self.capture_conv_input = None;
        }
    }
}

impl<T: lisa_backend::TensorAccess + 'static> Default for GdnCache<T> {
    fn default() -> Self {
        Self::new()
    }
}
