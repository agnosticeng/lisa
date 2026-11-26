//! Deferred row providers for model-supplied Engram requests.
//! The execution graph owns no file readers or checkpoint metadata.
use anyhow::Result;
pub use lisa_models::deepseek41::engram::{HEADS, History, LAYERS, Layout, RowIds, WIDTH};

/// Starts bounded background row reads; storage and decoding belong to the provider.
pub trait Tables: Send + Sync {
    fn prefetch(&self, ids: &[RowIds]) -> Result<Box<dyn ReadAhead>>;
}

/// Owns outstanding reads and must join them when dropped, including cancellation.
pub trait ReadAhead {
    fn wait(&mut self, layer: usize) -> Result<Vec<f32>>;
}

/// Batched host rows retain the model's original embedding preparation arithmetic.
/// Return exactly one decoded row per token, in request order. Iteration may
/// defer decoding so the execution graph retains its per-row preparation order.
pub trait EmbeddingRows: Send + Sync {
    fn rows(&self, tokens: &[u32]) -> Result<Box<dyn Iterator<Item = Result<Vec<f32>>>>>;
}
