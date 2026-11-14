//! Numerical policies for one Qwen architecture. Storage-specific kernels
//! preserve their own casts/reductions; the tower owns traversal and sessions.
use crate::{cache::LayerCache, indexer_cache::CacheDevice};
use lisa_backend::ScalarType;
use lisa_models::qwen3_5::config::Qwen35Config;

pub struct Forward<T> {
    pub x: T,
    pub residual: Option<T>,
    pub offset: usize,
    pub rope: Option<(T, T)>,
}
pub trait Execution<D: CacheDevice> {
    fn config(&self) -> &Qwen35Config;
    fn requires_cache(&self) -> bool {
        false
    }
    fn cache_dtype(&self) -> ScalarType {
        ScalarType::BF16
    }
    fn prefill_chunk(&self) -> usize {
        2048
    }
    fn forward_chunk(&self) -> Option<usize> {
        None
    }
    fn initial_capacity(&self) -> usize {
        self.config().max_position_embeddings
    }
    /// Report the loaded recurrent-state policy for measured block-draft runs.
    fn block_drafter_state_replay(&self) -> bool {
        false
    }
    /// Measured default verification depth for a block drafter. Its full
    /// proposal block and maximum supported depth remain independent.
    fn block_drafter_verify_depth(&self) -> usize {
        self.max_speculative_depth()
    }
    fn max_speculative_depth(&self) -> usize {
        usize::MAX
    }
    fn numerical_law(&self) -> Option<u64> {
        None
    }
    fn batch_decode_ok(&self) -> bool {
        true
    }
    fn begin(
        &self,
        ids: &D::Tensor,
        caches: Option<&Vec<LayerCache<D>>>,
    ) -> anyhow::Result<Forward<D::Tensor>>;
    fn layer(
        &mut self,
        index: usize,
        state: &mut Forward<D::Tensor>,
        cache: Option<&mut LayerCache<D>>,
        capture: bool,
    ) -> anyhow::Result<()>;
    /// Target block outputs consumed by an optional external block drafter.
    fn draft_target_layers(&self) -> &[usize] {
        &[]
    }
    fn draft_target_layer_output(&self, _state: &Forward<D::Tensor>) -> anyhow::Result<D::Tensor> {
        anyhow::bail!("this Qwen execution has no block drafter")
    }
    fn has_block_drafter(&self) -> bool {
        false
    }
    fn block_drafter_context_matches(&self, _tokens: &[u32]) -> bool {
        false
    }
    fn append_drafter_context(
        &mut self,
        _target_hidden: &D::Tensor,
        _tokens: &[u32],
    ) -> anyhow::Result<()> {
        anyhow::bail!("this Qwen execution has no block drafter")
    }
    fn draft_block(&mut self, _anchor: u32) -> anyhow::Result<D::Tensor> {
        anyhow::bail!("this Qwen execution has no block drafter")
    }
    fn finish(&self, state: Forward<D::Tensor>) -> anyhow::Result<D::Tensor>;
    fn head(&self, x: &D::Tensor) -> std::result::Result<D::Tensor, D::Error>;
    fn supports_images(&self) -> bool {
        false
    }
    fn prepare_images(
        &mut self,
        _tokens: &mut Vec<u32>,
        _images: &[image::RgbImage],
    ) -> anyhow::Result<()> {
        anyhow::bail!("checkpoint has no supported vision encoder")
    }
    fn clear_images(&mut self) {}
    fn has_drafter(&self) -> bool;
    fn mtp_cost_key(&self) -> String;
    fn take_draft_confidence(&mut self) -> Option<D::Tensor> {
        None
    }
    fn set_conf_capture(&mut self, _on: bool) {}
    fn drafter_reset(&mut self);
    fn drafter_trim(&mut self, n: usize);
    fn drafter_offset(&self) -> usize;
    fn drafter_restore_offset(&mut self, n: usize);
    fn draft_step(
        &mut self,
        tokens: &D::Tensor,
        hidden: &D::Tensor,
    ) -> anyhow::Result<(D::Tensor, D::Tensor)>;
}
