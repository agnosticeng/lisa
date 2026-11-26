//! Named tensor imports and materialized weight maps.
use std::collections::HashMap;
pub type Weights<T> = HashMap<String, T>;

/// Materialized tensor views with the loader's lifetime-scoped residency lease.
/// Execution receives neither checkpoint metadata nor a file-backed reader.
pub struct PreparedTensors<T, L> {
    // Drain outstanding users before the last tensor handles are released.
    pub(crate) _reservation: L,
    pub tensors: std::collections::BTreeMap<String, T>,
}
pub trait TensorSource<T> {
    fn get(&mut self, name: &str) -> anyhow::Result<T>;
}

/// Execution sees embedding tensors; a loading adapter owns any original
/// row decoding or storage policy needed to produce them.
pub trait EmbeddingRows<T>: Send + Sync {
    fn row(&self, token: u32) -> anyhow::Result<T>;
}

/// A loader-provided image adapter produces request-owned feature tensors.
/// Availability queries must not materialize model weights. File access and
/// any lazy residency policy stay behind this boundary, not in the text tower.
pub trait ImageFeatures<T> {
    fn available(&self) -> bool;
    fn encode(&self, image: &image::RgbImage) -> anyhow::Result<T>;
}
impl<T> TensorSource<T> for Weights<T> {
    fn get(&mut self, name: &str) -> anyhow::Result<T> {
        self.remove(name)
            .ok_or_else(|| anyhow::anyhow!("tensor {name} missing from weight map"))
    }
}
/// Deferred imports expose tensor storage and residency, without a checkpoint
/// container or file-reader type in the execution graph.
pub trait TensorBank<T>: Send + Sync {
    fn tensor_dtype(&self, name: &str) -> anyhow::Result<lisa_backend::ScalarType>;
    fn tensor_nbytes(&self, name: &str) -> anyhow::Result<usize>;
    fn get_mapped(&self, name: &str) -> anyhow::Result<T>;
    fn get_mapped_individual(&self, name: &str) -> anyhow::Result<T>;
    fn get_mapped_weight_bytes(&self, name: &str) -> anyhow::Result<T>;
    fn get_mapped_weight_bytes_individual(&self, name: &str) -> anyhow::Result<T>;
    fn get_packed(&self, names: &[String]) -> anyhow::Result<T>;
    fn prefetch_mapped_tensors(&self, names: &[String]) -> anyhow::Result<(usize, usize)>;
    fn resident_allocations(&self) -> Vec<T>;
    fn resident_nbytes(&self) -> usize;
    fn is_resident(&self) -> bool;
}
