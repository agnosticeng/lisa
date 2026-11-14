//! Original affine tensor triples and checkpoint geometry, independent of a provider.
use super::source::TensorSource;
use lisa_backend::TensorAccess;
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
pub struct QuantizedLinear<T> {
    pub weight: T, // u32 [N, K*bits/32]
    pub scales: T, // bf16 [N, K/group_size]
    pub biases: T, // bf16 [N, K/group_size]
    pub group_size: i32,
    pub bits: i32,
}

impl<T: TensorAccess> QuantizedLinear<T> {
    /// Override the quantization geometry. The group size is re-derived from
    /// the packed shapes for the given `bits`, so group-32/64 (and 2-bit)
    /// checkpoints are handled without a config; `group_size` is only the
    /// fallback when the shapes are inconsistent.
    pub fn set_quant(&mut self, group_size: i32, bits: i32) {
        self.bits = bits;
        self.group_size = detect_group_size(self.weight.shape()[1], self.scales.shape()[1], bits)
            .unwrap_or(group_size);
    }

    /// Load `{prefix}.{key}.weight/.scales/.biases`. `key` may be empty for
    /// tensors named directly under the prefix.
    pub fn load<S: TensorSource<T>>(src: &mut S, prefix: &str, key: &str) -> anyhow::Result<Self> {
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
pub struct QuantizedEmbedding<T> {
    pub weight: T, // u32 [V, K*bits/32]
    pub scales: T, // bf16 [V, K/group_size]
    pub biases: T, // bf16 [V, K/group_size]
    pub group_size: i32,
    pub bits: i32,
}

impl<T: TensorAccess> QuantizedEmbedding<T> {
    /// Override the quantization geometry. The group size is re-derived from
    /// the packed shapes for the given `bits`; `group_size` is the fallback.
    pub fn set_quant(&mut self, group_size: i32, bits: i32) {
        self.bits = bits;
        self.group_size = detect_group_size(self.weight.shape()[1], self.scales.shape()[1], bits)
            .unwrap_or(group_size);
    }

    pub fn load<S: TensorSource<T>>(src: &mut S, prefix: &str) -> anyhow::Result<Self> {
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
