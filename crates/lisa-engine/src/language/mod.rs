//! Portable language-session contract. Model plans belong to `lisa-models`;
//! engine adapters bind tensors, cache storage, and execution resources.
//! No device provider is selected or initialized by this interface.

pub mod runtime;

/// Storage and row operations needed by the common language interface.
/// This is an engine adapter, not a device API: devices never receive a cache
/// or model. A runtime binds its concrete storage to device tensor operations.
pub trait LanguageRuntime {
    type Tensor: Clone;
    type Cache;
    type DraftCache;
    type Error: std::error::Error + Send + Sync + 'static;
    fn shape(tensor: &Self::Tensor) -> &[i32];
    fn row(tensor: &Self::Tensor, index: usize) -> anyhow::Result<Self::Tensor>;
    fn join_rows(tensors: &[&Self::Tensor]) -> anyhow::Result<Self::Tensor>;
}

/// The interface the runtime needs from a decoder-only text model.
pub trait LanguageModel<R: LanguageRuntime> {
    /// Explicit engine runtime context; format readers and devices never select it.
    fn runtime(&self) -> &R;
    /// Reference sampler order/RNG is independent of whether a draft head is
    /// present in this checkpoint.
    fn uses_host_sampling(&self) -> bool {
        self.has_target_mtp()
    }
    /// Whether the checkpoint has a supported native still-image encoder.
    fn supports_images(&self) -> bool {
        false
    }
    /// Prepare one complete image prompt. Callers use a fresh session, without
    /// token-only prefix caching or speculative decoding, then clear this state.
    fn prepare_images(
        &mut self,
        _tokens: &mut Vec<u32>,
        _images: &[image::RgbImage],
    ) -> anyhow::Result<()> {
        anyhow::bail!("this checkpoint does not support image input")
    }
    /// Remove request-local image embeddings and positional context.
    fn clear_images(&mut self) {}
    /// Explicit arithmetic identity for cached model state, available before
    /// any lazy pipeline compilation. None disables shared prefix reuse.
    fn cache_numerical_law(&self) -> Option<u64> {
        None
    }
    /// Natural prompt-chunk alignment admitted for exact prefix reuse. A
    /// frontend may split only at multiples of this size, leaving the final
    /// original chunk intact. None leaves new cache integrations disabled.
    fn prefix_cache_granularity(&self) -> Option<usize> {
        None
    }
    /// Preferred progress/snapshot interval for long prompt feeding. This is
    /// scheduling advice, not a numerical alignment; frontends still round it
    /// to prefix_cache_granularity and preserve the original final chunk.
    fn preferred_prefill_step(&self) -> usize {
        128
    }
    /// Effective sequence limit (prompt + generated), including runtime capacity.
    fn max_position_embeddings(&self) -> usize;
    /// Maximum proposal depth, excluding the carried target token.
    fn max_speculative_depth(&self) -> usize {
        usize::MAX
    }
    /// Configure the runtime context limit before warmup or creating sessions.
    /// Models that support this report the effective limit through
    /// `max_position_embeddings`; other models keep their existing behavior.
    fn set_context_capacity(&mut self, capacity: usize) -> anyhow::Result<()> {
        anyhow::bail!(
            "runtime context capacity {capacity} is not supported by this model; omit --ctx-size"
        )
    }
    /// End-of-sequence token id.
    fn eos_token_id(&self) -> i64;
    /// All token ids that end generation. Models with a multi-token stop set
    /// override this while keeping `eos_token_id` as the legacy primary id.
    fn eos_token_ids(&self) -> Vec<i64> {
        vec![self.eos_token_id()]
    }
    /// Fresh per-layer caches for one stream.
    fn new_caches(&self) -> Vec<R::Cache>;
    /// One forward over `ids [B,S]`; returns `(mixed [B,S,H], multi)`.
    fn forward(
        &mut self,
        ids: &R::Tensor,
        caches: Option<&mut Vec<R::Cache>>,
    ) -> anyhow::Result<(R::Tensor, R::Tensor)>;
    /// Forward one decode token for independently positioned streams that keep
    /// separate per-row caches. Architectures with row-local sparse state may
    /// override this to batch the stateless tower operations while advancing
    /// each row's cache at its own absolute position.
    fn forward_ragged(
        &mut self,
        ids: &R::Tensor,
        per_row_caches: &mut [Vec<R::Cache>],
    ) -> anyhow::Result<(R::Tensor, R::Tensor)> {
        anyhow::ensure!(
            R::shape(ids).first().copied() == Some(per_row_caches.len() as i32)
                && R::shape(ids).get(1).copied() == Some(1),
            "ragged forward expects one token and cache per row, got ids {:?}, caches {}",
            R::shape(ids),
            per_row_caches.len()
        );
        let n = per_row_caches.len();
        let mut mixed_rows = Vec::with_capacity(n);
        let mut multi_rows = Vec::with_capacity(n);
        let tails = self.context_tails();
        let mut next_tails = Vec::with_capacity(per_row_caches.len());
        for (row, caches) in per_row_caches.iter_mut().enumerate() {
            if tails.len() == n {
                self.set_context_tails(vec![tails[row].clone()]);
            }
            let ids_row = R::row(ids, row)?;
            let (mixed, multi) = self.forward(&ids_row, Some(caches))?;
            mixed_rows.push(mixed);
            multi_rows.push(multi);
            let updated = self.context_tails();
            if updated.len() == 1 {
                next_tails.push(updated[0].clone());
            }
        }
        if next_tails.len() == n {
            self.set_context_tails(next_tails);
        } else if tails.len() == per_row_caches.len() {
            self.set_context_tails(tails);
        }
        let mixed_refs: Vec<&R::Tensor> = mixed_rows.iter().collect();
        let multi_refs: Vec<&R::Tensor> = multi_rows.iter().collect();
        Ok((R::join_rows(&mixed_refs)?, R::join_rows(&multi_refs)?))
    }
    /// Forward with the speculative-verify capture enabled.
    fn forward_capture(
        &mut self,
        ids: &R::Tensor,
        caches: Option<&mut Vec<R::Cache>>,
        capture: bool,
    ) -> anyhow::Result<(R::Tensor, R::Tensor)>;
    /// Install a layer capture for targeted model-parity tests.
    #[cfg(test)]
    fn set_debug_capture_for_test(&mut self, _layers: Vec<usize>) -> anyhow::Result<()> {
        anyhow::bail!("debug capture is not supported by this model")
    }
    /// Install a test capture limited to selected layer keys.
    #[cfg(test)]
    fn set_debug_capture_filtered_for_test(
        &mut self,
        layers: Vec<usize>,
        _key_filters: Vec<String>,
    ) -> anyhow::Result<()> {
        self.set_debug_capture_for_test(layers)
    }
    /// Flush the targeted model-parity capture to an NPZ file.
    #[cfg(test)]
    fn flush_debug_capture_for_test(&mut self, _path: &str) -> anyhow::Result<()> {
        anyhow::bail!("debug capture is not supported by this model")
    }
    /// Record output-head vectors for a diagnostic parity run. Production
    /// dispatch remains active; the caller excludes this run from timings.
    fn enable_logit_capture(&mut self) -> anyhow::Result<()> {
        anyhow::bail!("logit capture is not supported by this model")
    }
    /// Record selected internal stages while retaining production dispatch.
    fn enable_stage_capture(
        &mut self,
        _layers: Vec<usize>,
        _keys: Vec<String>,
    ) -> anyhow::Result<()> {
        anyhow::bail!("stage capture is not supported by this model")
    }
    /// Write the diagnostic vectors after generation has completed.
    fn write_logit_capture(&self, _path: &str) -> anyhow::Result<()> {
        anyhow::bail!("logit capture is not supported by this model")
    }
    /// Prefill `tokens`; returns the last row's hidden state.
    fn prefill(&mut self, tokens: &[u32], caches: &mut Vec<R::Cache>) -> anyhow::Result<R::Tensor>;
    /// Feed the final prompt suffix immediately before sampling. A model may
    /// preserve a reference-specific final-row boundary here. Cache-only
    /// staging and history replay continue to use `prefill`; this method must
    /// commit every supplied token exactly once and return its last hidden row.
    fn prefill_for_generation(
        &mut self,
        tokens: &[u32],
        caches: &mut Vec<R::Cache>,
    ) -> anyhow::Result<R::Tensor> {
        self.prefill(tokens, caches)
    }
    /// Commit one generated token; returns its last hidden state. Models with
    /// a distinct generation graph override this without changing prompt or
    /// suffix prefill, including one-token prompt chunks.
    fn decode_token(
        &mut self,
        token: u32,
        caches: &mut Vec<R::Cache>,
    ) -> anyhow::Result<R::Tensor> {
        self.prefill(&[token], caches)
    }
    /// Whether ordinary greedy decode can consume one engine-generated device
    /// ID without a host round trip. Opt-in requires the exact serial graph,
    /// reversible cache snapshots, and no request-local capture or media work.
    /// Public forwards retain their validation of caller-supplied token values.
    fn supports_device_serial_decode(&self) -> bool {
        false
    }
    /// Consume a trusted, vocabulary-bounded I32 token `[1,1]` selected from
    /// this model's own ordinary head. Return the same hidden shape and law as
    /// `decode_token`. The engine may roll back this append before exposing its
    /// token; cache snapshots and previous hidden tensors must remain immutable.
    fn decode_generated_token(
        &mut self,
        _ids: &R::Tensor,
        _caches: &mut Vec<R::Cache>,
    ) -> anyhow::Result<R::Tensor> {
        anyhow::bail!("device-generated serial decode is not supported")
    }
    /// Prefill `tokens`; returns the last hidden state and the multi stream.
    fn prefill_multi(
        &mut self,
        tokens: &[u32],
        caches: &mut Vec<R::Cache>,
    ) -> anyhow::Result<(R::Tensor, R::Tensor)>;
    /// Sampling-boundary counterpart of `prefill_multi`, preserving the
    /// model's multi-stream and drafter-capture contract for all supplied rows.
    fn prefill_multi_for_generation(
        &mut self,
        tokens: &[u32],
        caches: &mut Vec<R::Cache>,
    ) -> anyhow::Result<(R::Tensor, R::Tensor)> {
        self.prefill_multi(tokens, caches)
    }
    /// The output head: hidden `[.., H]` -> logits `[.., vocab]`.
    fn head(&self, mixed: &R::Tensor) -> Result<R::Tensor, R::Error>;
    /// Project verified rows using the target's captured numerical law. The
    /// default shares the ordinary readout; implementations may preserve
    /// single-row accumulation explicitly when verification batches rows.
    fn head_verified(&self, mixed: &R::Tensor) -> Result<R::Tensor, R::Error> {
        self.head(mixed)
    }

    /// Apply the output head to independently cached decode rows. A model may
    /// override this when its reference engine falls back from native batch
    /// execution to a row-serial graph at a context or batch-size boundary.
    fn head_ragged(
        &self,
        mixed: &R::Tensor,
        _per_row_caches: &[Vec<R::Cache>],
    ) -> Result<R::Tensor, R::Error> {
        self.head(mixed)
    }
    /// Warm the kernels for the scored shapes.
    fn warmup(&mut self, seed: &[u32]) -> anyhow::Result<()>;

    // --- optional host-side rolling context (e.g. the PLE n-gram window) ---
    /// Whether B>1 batched decode is safe on this tower. `false` clamps the
    /// scheduler to lone streams (the exact single-stream path).
    fn batch_decode_ok(&self) -> bool {
        true
    }
    /// Some batch-capable towers require every live row to share one absolute
    /// cache position. The scheduler then admits only matching prompt lengths.
    fn batch_decode_equal_length_only(&self) -> bool {
        false
    }
    /// Whether independently positioned streams can share a decode batch
    /// while retaining row-local cache state.
    fn batch_decode_ragged_supported(&self) -> bool {
        false
    }
    /// O3: bytes of KV a single token occupies for one stream (0 = unknown, in
    /// which case admission is skipped rather than guessed).
    fn kv_bytes_per_token(&self) -> usize {
        0
    }
    /// O3: fixed per-stream non-KV state bytes (GDN conv/SSM, indexer tape).
    fn stream_state_bytes(&self) -> usize {
        0
    }
    /// Rolling context window size (0 = the model carries none).
    fn context_window(&self) -> usize {
        0
    }
    /// The current per-stream context tails.
    fn context_tails(&self) -> Vec<Vec<i64>> {
        Vec::new()
    }
    /// Replace the per-stream context tails.
    fn set_context_tails(&mut self, _tails: Vec<Vec<i64>>) {}
    /// Drop the rolling context.
    fn clear_context(&mut self) {}

    // --- optional speculative drafting (e.g. the MTP head) ---
    /// Whether this model can draft speculative tokens.
    fn has_drafter(&self) -> bool {
        false
    }
    /// Whether the model has a target-hidden MTP head (GLM 5.3). Unlike the
    /// independent Qwen draft chain, this head must be advanced alongside
    /// target hidden rows and is driven by `target_mtp_step`.
    fn has_target_mtp(&self) -> bool {
        false
    }
    /// One target-conditioned draft step at an absolute target position.
    fn target_mtp_step(
        &mut self,
        _cache: &mut R::DraftCache,
        _target_multi: &R::Tensor,
        _next_token: u32,
        _position: usize,
        _min_position: usize,
    ) -> anyhow::Result<R::Tensor> {
        anyhow::bail!("this model has no target-hidden MTP head")
    }
    /// Reset the drafter's own caches.
    fn drafter_reset(&mut self) {}
    /// Trim the drafter's caches by `n` rows (rollback).
    fn drafter_trim(&mut self, _n: usize) {}
    /// The drafter's cache length in rows (pairs consumed so far).
    fn drafter_offset(&self) -> usize {
        0
    }
    /// Rewind the drafter's cache to a previously captured offset (prefix
    /// snapshot restore).
    fn drafter_restore_offset(&mut self, _n: usize) {}
    /// Whether the model has a block drafter, which drafts tree rounds
    /// ([`Self::draft_tree`]).
    fn has_block_drafter(&self) -> bool {
        false
    }
    /// Whether the model verifies a draft tree in one forward
    /// ([`Self::draft_tree`], [`Self::verify_tree`], [`Self::commit_tree`]).
    fn supports_tree_verify(&self) -> bool {
        false
    }
    /// Fewest rows a tree verification takes (the anchor's included).
    fn tree_min_rows(&self) -> usize {
        2
    }
    /// The learned control state of tree rounds, kept with the loaded model
    /// across conversations.
    fn tree_control(&mut self) -> Option<&mut crate::speculation::tree::TreeControl> {
        None
    }
    /// Grow a draft tree after `anchor` from the drafter's context in
    /// `caches` and lay out its verify rows, all on the device: nothing is
    /// read back.
    fn draft_tree(
        &mut self,
        _anchor: u32,
        _plan: &crate::speculation::tree::TreePlan,
        _caches: &mut Vec<R::Cache>,
    ) -> anyhow::Result<crate::speculation::tree::DeviceTree<R::Tensor>> {
        anyhow::bail!("this model has no tree drafter")
    }
    /// Verify device IDs `I32 [rows]` (row 0 the anchor) under the tree
    /// `I32 [2, rows]` (parents, -1 for a root and each parent before its
    /// children; depths): hidden rows `[1, rows, H]`, with draft taps
    /// captured when capture is on. The caches change only on
    /// [`Self::commit_tree`].
    fn verify_tree(
        &mut self,
        _ids: &R::Tensor,
        _tree: &R::Tensor,
        _caches: &mut Vec<R::Cache>,
    ) -> anyhow::Result<R::Tensor> {
        anyhow::bail!("this model has no tree verification")
    }
    /// Commit the accepted root-to-node path (row indices, starting at the
    /// root) of the last tree verification.
    fn commit_tree(&mut self, _caches: &mut Vec<R::Cache>, _path: &[usize]) -> anyhow::Result<()> {
        anyhow::bail!("this model has no tree verification")
    }
    /// Drop the last tree verification, leaving the caches as they were.
    fn discard_tree(&mut self, _caches: &mut Vec<R::Cache>) -> anyhow::Result<()> {
        anyhow::bail!("this model has no tree verification")
    }
    /// One draft step over `tokens [1,S]` and `multi [1,S,hc*H]`: returns the
    /// drafted token id and the last multi-stream row for the next step.
    fn draft_step(
        &mut self,
        _tokens: &R::Tensor,
        _multi: &R::Tensor,
    ) -> anyhow::Result<(R::Tensor, R::Tensor)> {
        anyhow::bail!("this model has no drafter")
    }
    /// Stable per-architecture row key for the round-cost table (shape, never
    /// a repo id or snapshot hash), e.g. `qwen3_5-64L-5120h`.
    fn mtp_cost_key(&self) -> String {
        String::new()
    }
}
