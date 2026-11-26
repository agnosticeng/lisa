//! Quantized primitives: affine 4-bit group-32 linear layers and embeddings.
//!
//! The group size is detected per tensor from the packed weight/scale shapes
//! (`detect_group_size`), so group-32 (Flash-Next) and group-64 (Qwen 3.8 27B)
//! checkpoints load without a config hint; `bits` is still taken from the
//! checkpoint config (2/4/8) and passed to `set_quant`.

use lisa_metal::{Array, Dtype, ops};

use crate::core::loader::{Shard, array_from_bytes};

pub const GROUP_SIZE: i32 = 32;
pub const BITS: i32 = 4;
pub use lisa_engine::bindings::glm53::policy::*;
pub use lisa_engine::weights::affine::detect_group_size;
pub type QuantizedLinear = lisa_engine::weights::affine::QuantizedLinear<Array>;
pub type QuantizedEmbedding = lisa_engine::weights::affine::QuantizedEmbedding<Array>;
pub trait QuantizedLinearExecution {
    fn forward(&self, x: &Array) -> lisa_metal::error::Result<Array>;
    fn forward_glm53_prefill(&self, x: &Array) -> lisa_metal::error::Result<Array>;
    fn forward_serial_rows(&self, x: &Array) -> lisa_metal::error::Result<Array>;
}
impl QuantizedLinearExecution for QuantizedLinear {
    /// Forward: [..., K] -> [..., N]
    ///
    /// The input is flattened to 2-D for the quantized matmul: mlx-c's
    /// quantized_matmul mis-handles batched (3-D+) inputs at M > 1 (the values
    /// are wrong entirely), while 2-D is exact.
    fn forward(&self, x: &Array) -> lisa_metal::error::Result<Array> {
        let shape = x.shape();
        let rows: i32 = shape[..shape.len() - 1].iter().product::<i32>().max(1);
        let k = x.dim(-1);
        let flat = x.reshape(&[rows, k])?;
        let out = ops::quantized_matmul(
            &flat,
            &self.weight,
            &self.scales,
            Some(&self.biases),
            true,
            self.group_size,
            self.bits,
        )?;
        out.reshape(&[&shape[..shape.len() - 1], &[self.dims_out() as i32]].concat())
    }

    /// GLM affine projection using the reference's native tiny-row/matrix crossover.
    /// The independent native fixtures and same-weight full-model captures
    /// recorded under resident-port-20261102 establish this arithmetic law.
    fn forward_glm53_prefill(&self, x: &Array) -> lisa_metal::error::Result<Array> {
        crate::bindings::glm53_affine_projection::forward(
            self,
            x,
            glm53_q8_mpp_prefill_override() != Some(false),
        )
        .map_err(|e| lisa_metal::error::Error::msg(e.to_string()))
    }

    /// Forward independent GLM decode rows with the same one-row qmv
    /// arithmetic schedule as serial sessions, sharing one dispatch when the
    /// packed affine shape is qmv-fast eligible.
    fn forward_serial_rows(&self, x: &Array) -> lisa_metal::error::Result<Array> {
        if x.rank() != 3 || x.dim(0) <= 1 || x.dim(1) != 1 {
            return self.forward(x);
        }
        ops::quantized_matmul_serial_rows(
            x,
            &self.weight,
            &self.scales,
            Some(&self.biases),
            true,
            self.group_size,
            self.bits,
        )
    }
}
pub trait QuantizedEmbeddingExecution {
    fn forward(&self, ids: &Array) -> lisa_metal::error::Result<Array>;
    fn as_linear(&self, x: &Array) -> lisa_metal::error::Result<Array>;
}
impl QuantizedEmbeddingExecution for QuantizedEmbedding {
    /// Token ids (any integer dtype) -> [..., K] bf16 rows.
    fn forward(&self, ids: &Array) -> lisa_metal::error::Result<Array> {
        let rows = self.weight.take_axis(ids, 0)?;
        let scales = self.scales.take_axis(ids, 0)?;
        let biases = self.biases.take_axis(ids, 0)?;
        ops::dequantize(&rows, &scales, &biases, self.group_size, self.bits)
    }

    /// The `as_linear` form: [..., K] -> [..., V].
    fn as_linear(&self, x: &Array) -> lisa_metal::error::Result<Array> {
        ops::quantized_matmul(
            x,
            &self.weight,
            &self.scales,
            Some(&self.biases),
            true,
            self.group_size,
            self.bits,
        )
    }
}
/// Fetch one tensor from a shard by (sanitized) name, trying the
/// `language_model.`-prefixed raw name as well (for checkpoints that already
/// store sanitized names the raw probe simply misses).
pub fn get_tensor(shard: &Shard, name: &str) -> anyhow::Result<Array> {
    let candidates = [name, &format!("language_model.{name}")];
    for candidate in candidates {
        if let Some((bytes, dtype, shape)) = shard.tensor_bytes(candidate) {
            if let Some(array) = shard.tensor_array(candidate)? {
                return Ok(array);
            }
            return array_from_bytes(bytes, dtype, shape).map_err(|e| anyhow::anyhow!("{e}"));
        }
    }
    anyhow::bail!("tensor {name} not found in shard")
}

/// Load a plain bf16 tensor (norm weights, A_log, dt_bias, conv kernels...).
pub fn get_bf16(shard: &Shard, name: &str) -> anyhow::Result<Array> {
    let a = get_tensor(shard, name)?;
    if a.dtype() == Dtype::Bfloat16 {
        Ok(a)
    } else {
        a.as_dtype(Dtype::Bfloat16)
            .map_err(|e| anyhow::anyhow!("{e}"))
    }
}
