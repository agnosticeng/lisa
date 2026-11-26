//! Numerical policies for one Qwen architecture. Storage-specific kernels
//! preserve their own casts/reductions; the tower owns traversal and sessions.
use crate::cache::{LayerCache, indexer::CacheDevice};
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
    fn finish(&self, state: Forward<D::Tensor>) -> anyhow::Result<D::Tensor>;
    fn head(&self, x: &D::Tensor) -> std::result::Result<D::Tensor, D::Error>;
    fn head_verified(&self, x: &D::Tensor) -> std::result::Result<D::Tensor, D::Error> {
        self.head(x)
    }
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
