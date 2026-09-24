//! Quantized primitives: affine 4-bit group-32 linear layers and embeddings.

use lisa_mlx::{ops, Array, Dtype};

use crate::core::loader::{array_from_bytes, Shard, TensorSource};

pub const GROUP_SIZE: i32 = 32;
pub const BITS: i32 = 4;

/// An affine-quantized linear layer: `y = x @ W^T` with W stored as
/// packed 4-bit codes plus per-32-group bf16 scales and biases.
#[derive(Clone)]
pub struct QuantizedLinear {
    pub weight: Array, // u32 [N, K/8]
    pub scales: Array, // bf16 [N, K/32]
    pub biases: Array, // bf16 [N, K/32]
}

impl QuantizedLinear {
    /// Forward: [..., K] -> [..., N]
    ///
    /// The input is flattened to 2-D for the quantized matmul: mlx-c's
    /// quantized_matmul mis-handles batched (3-D+) inputs at M > 1 (values
    /// diverge from the reference entirely), while 2-D is exact.
    pub fn forward(&self, x: &Array) -> lisa_mlx::error::Result<Array> {
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
            GROUP_SIZE,
            BITS,
        )?;
        out.reshape(&[&shape[..shape.len() - 1], &[self.dims_out() as i32]].concat())
    }

    /// Load `{prefix}.{key}.weight/.scales/.biases`. `key` may be empty for
    /// tensors named directly under the prefix.
    pub fn load<S: TensorSource>(src: &mut S, prefix: &str, key: &str) -> anyhow::Result<Self> {
        let path = |suffix: &str| {
            if key.is_empty() {
                format!("{prefix}.{suffix}")
            } else {
                format!("{prefix}.{key}.{suffix}")
            }
        };
        Ok(Self {
            weight: src.get(&path("weight"))?,
            scales: src.get(&path("scales"))?,
            biases: src.get(&path("biases"))?,
        })
    }

    /// Output rows (N).
    pub fn dims_out(&self) -> usize {
        self.weight.shape()[0] as usize
    }

}

/// An affine-quantized embedding table. Lookups dequantize only the
/// gathered rows.
#[derive(Clone)]
pub struct QuantizedEmbedding {
    pub weight: Array, // u32 [V, K/8]
    pub scales: Array, // bf16 [V, K/32]
    pub biases: Array, // bf16 [V, K/32]
}

impl QuantizedEmbedding {
    /// Token ids (any integer dtype) -> [..., K] bf16 rows.
    pub fn forward(&self, ids: &Array) -> lisa_mlx::error::Result<Array> {
        let rows = self.weight.take_axis(ids, 0)?;
        let scales = self.scales.take_axis(ids, 0)?;
        let biases = self.biases.take_axis(ids, 0)?;
        ops::dequantize(&rows, &scales, &biases, GROUP_SIZE, BITS)
    }

    /// The `as_linear` form: [..., K] -> [..., V].
    pub fn as_linear(&self, x: &Array) -> lisa_mlx::error::Result<Array> {
        ops::quantized_matmul(
            x,
            &self.weight,
            &self.scales,
            Some(&self.biases),
            true,
            GROUP_SIZE,
            BITS,
        )
    }

    pub fn load<S: TensorSource>(src: &mut S, prefix: &str) -> anyhow::Result<Self> {
        Ok(Self {
            weight: src.get(&format!("{prefix}.weight"))?,
            scales: src.get(&format!("{prefix}.scales"))?,
            biases: src.get(&format!("{prefix}.biases"))?,
        })
    }
}

/// Fetch one tensor from a shard by (sanitized) name, trying the
/// `language_model.`-prefixed raw name as well (for checkpoints that already
/// store sanitized names the raw probe simply misses).
pub fn get_tensor(shard: &Shard, name: &str) -> anyhow::Result<Array> {
    let candidates = [name, &format!("language_model.{name}")];
    for candidate in candidates {
        if let Some((bytes, dtype, shape)) = shard.tensor_bytes(candidate) {
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
