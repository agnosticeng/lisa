//! Quantized primitives: affine 4-bit group-32 linear layers and embeddings.
//!
//! The group size is detected per tensor from the packed weight/scale shapes
//! (`detect_group_size`), so group-32 (Flash-Next) and group-64 (Qwen 3.8 27B)
//! checkpoints load without a config hint; `bits` is still taken from the
//! checkpoint config (2/4/8) and passed to `set_quant`.

use lisa_mlx::{ops, Array, Dtype};

use crate::core::loader::{array_from_bytes, Shard, TensorSource};

pub const GROUP_SIZE: i32 = 32;
pub const BITS: i32 = 4;

/// Infer the affine group size from the packed tensors, given `bits`.
///
/// `weight: [N, K*bits/32]` and `scales/biases: [N, K/group]`, so the input
/// width `K` is `weight_cols * (32/bits)` and `group = K / scales_cols`. This
/// makes the loader choose group 32 vs 64 (vs 16/128, and 2/4/8-bit) from the
/// checkpoint itself instead of trusting a config block. `None` when the
/// shapes are inconsistent, so callers fall back to their hint.
pub fn detect_group_size(weight_cols: i32, scales_cols: i32, bits: i32) -> Option<i32> {
    if weight_cols <= 0 || scales_cols <= 0 || bits <= 0 || 32 % bits != 0 {
        return None;
    }
    let k = weight_cols.checked_mul(32 / bits)?;
    if k % scales_cols != 0 {
        return None;
    }
    let group = k / scales_cols;
    if (1..=4096).contains(&group) {
        Some(group)
    } else {
        None
    }
}

/// An affine-quantized linear layer: `y = x @ W^T` with W stored as
/// packed 4-bit codes plus per-`group_size` bf16 scales and biases.
#[derive(Clone)]
pub struct QuantizedLinear {
    pub weight: Array, // u32 [N, K*bits/32]
    pub scales: Array, // bf16 [N, K/group_size]
    pub biases: Array, // bf16 [N, K/group_size]
    pub group_size: i32,
    pub bits: i32,
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
            self.group_size,
            self.bits,
        )?;
        out.reshape(&[&shape[..shape.len() - 1], &[self.dims_out() as i32]].concat())
    }

    /// Override the quantization geometry. The group size is re-derived from
    /// the packed shapes for the given `bits`, so group-32/64 (and 2-bit)
    /// checkpoints are handled without a config; `group_size` is only the
    /// fallback when the shapes are inconsistent.
    pub fn set_quant(&mut self, group_size: i32, bits: i32) {
        self.bits = bits;
        self.group_size =
            detect_group_size(self.weight.shape()[1], self.scales.shape()[1], bits)
                .unwrap_or(group_size);
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
        let weight = src.get(&path("weight"))?;
        let scales = src.get(&path("scales"))?;
        let biases = src.get(&path("biases"))?;
        let group_size =
            detect_group_size(weight.shape()[1], scales.shape()[1], BITS).unwrap_or(GROUP_SIZE);
        Ok(Self {
            weight,
            scales,
            biases,
            group_size,
            bits: BITS,
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
    pub weight: Array, // u32 [V, K*bits/32]
    pub scales: Array, // bf16 [V, K/group_size]
    pub biases: Array, // bf16 [V, K/group_size]
    pub group_size: i32,
    pub bits: i32,
}

impl QuantizedEmbedding {
    /// Token ids (any integer dtype) -> [..., K] bf16 rows.
    pub fn forward(&self, ids: &Array) -> lisa_mlx::error::Result<Array> {
        let rows = self.weight.take_axis(ids, 0)?;
        let scales = self.scales.take_axis(ids, 0)?;
        let biases = self.biases.take_axis(ids, 0)?;
        ops::dequantize(&rows, &scales, &biases, self.group_size, self.bits)
    }

    /// The `as_linear` form: [..., K] -> [..., V].
    pub fn as_linear(&self, x: &Array) -> lisa_mlx::error::Result<Array> {
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

    /// Override the quantization geometry. The group size is re-derived from
    /// the packed shapes for the given `bits`; `group_size` is the fallback.
    pub fn set_quant(&mut self, group_size: i32, bits: i32) {
        self.bits = bits;
        self.group_size =
            detect_group_size(self.weight.shape()[1], self.scales.shape()[1], bits)
                .unwrap_or(group_size);
    }

    pub fn load<S: TensorSource>(src: &mut S, prefix: &str) -> anyhow::Result<Self> {
        let weight = src.get(&format!("{prefix}.weight"))?;
        let scales = src.get(&format!("{prefix}.scales"))?;
        let biases = src.get(&format!("{prefix}.biases"))?;
        let group_size =
            detect_group_size(weight.shape()[1], scales.shape()[1], BITS).unwrap_or(GROUP_SIZE);
        Ok(Self {
            weight,
            scales,
            biases,
            group_size,
            bits: BITS,
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
