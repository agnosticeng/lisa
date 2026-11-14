//! Named tensor imports and materialized weight maps.
use std::collections::HashMap;
pub type Weights<T> = HashMap<String, T>;
pub trait TensorSource<T> {
    fn get(&mut self, name: &str) -> anyhow::Result<T>;
}
impl<T> TensorSource<T> for Weights<T> {
    fn get(&mut self, name: &str) -> anyhow::Result<T> {
        self.remove(name)
            .ok_or_else(|| anyhow::anyhow!("tensor {name} missing from weight map"))
    }
}
impl<D> TensorSource<D::Tensor> for super::mlx::Shard<D>
where
    D: lisa_backend::StorageDevice + Sync,
    D::Tensor: Clone + Send,
{
    fn get(&mut self, name: &str) -> anyhow::Result<D::Tensor> {
        get_tensor(self, name)
    }
}

pub fn get_tensor<D>(shard: &super::mlx::Shard<D>, name: &str) -> anyhow::Result<D::Tensor>
where
    D: lisa_backend::StorageDevice + Sync,
    D::Tensor: Clone + Send,
{
    for candidate in [name, &format!("language_model.{name}")] {
        if let Some(tensor) = shard.tensor_array(candidate)? {
            return Ok(tensor);
        }
        if let Some((bytes, dtype, shape)) = shard.tensor_bytes(candidate) {
            return Ok(shard.device().pull(
                lisa_backend::HostTensor {
                    bytes,
                    dtype,
                    shape,
                }
                .into(),
                Default::default(),
            )?);
        }
    }
    anyhow::bail!("tensor {name} not found in shard")
}
