//! Quantized primitives: affine 4-bit group-32 linear layers and embeddings.
//!
//! The group size is detected per tensor from the packed weight/scale shapes
//! (`detect_group_size`), so group-32 (Flash-Next) and group-64 (Qwen 3.8 27B)
//! checkpoints load without a config hint; `bits` is still taken from the
//! checkpoint config (2/4/8) and passed to `set_quant`.

use lisa_mlx::{Array, Dtype, ops};
use std::cell::Cell;

use crate::core::loader::{Shard, TensorSource, array_from_bytes};

pub const GROUP_SIZE: i32 = 32;
pub const BITS: i32 = 4;
/// DS4's grouped routed-MoE prefill shape gate starts at 96 rows. Keep the
/// production route aligned with that execution path; low-row throughput is
/// an open optimization target and must not silently select a different MoE
/// arithmetic schedule.
pub const GLM53_GROUPED_MOE_MIN_ROWS: usize = 96;

thread_local! {
    static GLM53_Q8_MPP_PREFILL_OVERRIDE: Cell<Option<bool>> = const { Cell::new(None) };
    static GLM53_GROUPED_MOE_PREFILL_OVERRIDE: Cell<Option<bool>> = const { Cell::new(None) };
    static GLM53_GROUPED_MOE_PAIR_SPLIT_OVERRIDE: Cell<Option<bool>> = const { Cell::new(None) };
    static GLM53_GROUPED_MOE_Q2_VECTOR_LOAD_OVERRIDE: Cell<Option<bool>> = const { Cell::new(None) };
    static GLM53_GROUPED_MOE_DOWN_TILE_OVERRIDE: Cell<Option<usize>> = const { Cell::new(None) };
    static GLM53_INDEXER_DECODE_ROWS4_OVERRIDE: Cell<Option<bool>> = const { Cell::new(None) };
    static GLM53_INDEXER_TOPK_SHUFFLE_OVERRIDE: Cell<Option<bool>> = const { Cell::new(None) };
    static GLM53_KDA_NSG_OVERRIDE: Cell<Option<usize>> = const { Cell::new(None) };
}

/// Scoped override used by the in-process prefill A/B harness. `true` selects
/// the native DS4 Q8 schedule; `false` selects stock QMM. Production uses DS4's
/// tiny-row and matrix schedules with the original MLX affine storage.
pub struct Glm53Q8PrefillScope {
    previous: Option<bool>,
}

impl Glm53Q8PrefillScope {
    pub fn new(use_ds4_mpp: bool) -> Self {
        let previous = GLM53_Q8_MPP_PREFILL_OVERRIDE.with(|slot| {
            let previous = slot.get();
            slot.set(Some(use_ds4_mpp));
            previous
        });
        Self { previous }
    }
}

impl Drop for Glm53Q8PrefillScope {
    fn drop(&mut self) {
        GLM53_Q8_MPP_PREFILL_OVERRIDE.with(|slot| slot.set(self.previous));
    }
}

fn glm53_q8_mpp_prefill_override() -> Option<bool> {
    GLM53_Q8_MPP_PREFILL_OVERRIDE.with(Cell::get)
}

/// Keep model-specific projection shortcuts on the same explicitly selected
/// Q8 arithmetic schedule as ordinary GLM linear projections.
pub(crate) fn glm53_q8_native_prefill_enabled() -> bool {
    glm53_q8_mpp_prefill_override() != Some(false)
}

/// Scoped override for the DS4-shaped grouped routed-MoE prefill schedule.
/// The in-process prefill A/B harness uses this to compare the grouped path
/// with the production per-slot schedule on identical model state and rows.
pub struct Glm53GroupedMoePrefillScope {
    previous: Option<bool>,
}

impl Glm53GroupedMoePrefillScope {
    pub fn new(use_grouped: bool) -> Self {
        let previous = GLM53_GROUPED_MOE_PREFILL_OVERRIDE.with(|slot| {
            let previous = slot.get();
            slot.set(Some(use_grouped));
            previous
        });
        Self { previous }
    }
}

impl Drop for Glm53GroupedMoePrefillScope {
    fn drop(&mut self) {
        GLM53_GROUPED_MOE_PREFILL_OVERRIDE.with(|slot| slot.set(self.previous));
    }
}

pub(crate) fn glm53_grouped_moe_prefill_override() -> Option<bool> {
    GLM53_GROUPED_MOE_PREFILL_OVERRIDE.with(Cell::get)
}

/// Scoped override used to interleave the fused and DS4-split grouped gate/up
/// schedules on the same loaded model and prompt.
pub struct Glm53GroupedMoePairScope {
    previous: Option<bool>,
}

impl Glm53GroupedMoePairScope {
    pub fn new(use_split: bool) -> Self {
        let previous = GLM53_GROUPED_MOE_PAIR_SPLIT_OVERRIDE.with(|slot| {
            let previous = slot.get();
            slot.set(Some(use_split));
            previous
        });
        Self { previous }
    }
}

impl Drop for Glm53GroupedMoePairScope {
    fn drop(&mut self) {
        GLM53_GROUPED_MOE_PAIR_SPLIT_OVERRIDE.with(|slot| slot.set(self.previous));
    }
}

pub(crate) fn glm53_grouped_moe_pair_split_override() -> Option<bool> {
    GLM53_GROUPED_MOE_PAIR_SPLIT_OVERRIDE.with(Cell::get)
}

/// Scoped load-mode switch used to interleave scalar byte loads with the
/// four-byte affine-Q2 unpack path in the grouped MoE prefill benchmark.
pub struct Glm53GroupedMoeQ2LoadScope {
    previous: Option<bool>,
}

impl Glm53GroupedMoeQ2LoadScope {
    pub fn new(use_vector_load: bool) -> Self {
        let previous = GLM53_GROUPED_MOE_Q2_VECTOR_LOAD_OVERRIDE.with(|slot| {
            let previous = slot.get();
            slot.set(Some(use_vector_load));
            previous
        });
        Self { previous }
    }
}

impl Drop for Glm53GroupedMoeQ2LoadScope {
    fn drop(&mut self) {
        GLM53_GROUPED_MOE_Q2_VECTOR_LOAD_OVERRIDE.with(|slot| slot.set(self.previous));
    }
}

pub(crate) fn glm53_grouped_moe_q2_vector_load_override() -> Option<bool> {
    GLM53_GROUPED_MOE_Q2_VECTOR_LOAD_OVERRIDE.with(Cell::get)
}

/// Scoped output-row tile override for the grouped affine-Q2 down projection.
/// The 128-row candidate computes two independent DS4 64-row tiles per
/// threadgroup while preserving each output's K accumulation order.
pub struct Glm53GroupedMoeDownTileScope {
    previous: Option<usize>,
}

impl Glm53GroupedMoeDownTileScope {
    pub fn new(output_tile: usize) -> Self {
        assert!(
            matches!(output_tile, 64 | 128),
            "unsupported grouped down tile {output_tile}"
        );
        let previous = GLM53_GROUPED_MOE_DOWN_TILE_OVERRIDE.with(|slot| {
            let previous = slot.get();
            slot.set(Some(output_tile));
            previous
        });
        Self { previous }
    }
}

impl Drop for Glm53GroupedMoeDownTileScope {
    fn drop(&mut self) {
        GLM53_GROUPED_MOE_DOWN_TILE_OVERRIDE.with(|slot| slot.set(self.previous));
    }
}

pub(crate) fn glm53_grouped_moe_down_tile_override() -> Option<usize> {
    GLM53_GROUPED_MOE_DOWN_TILE_OVERRIDE.with(Cell::get)
}

/// Scoped experimental schedule for end-to-end serial-audit A/B runs. `None`
/// selects the device schedule matching DS4; a value selects a different
/// simdgroup count for the fused BF16 KDA input projection only.
pub struct Glm53KdaNsgScope {
    previous: Option<usize>,
}

impl Glm53KdaNsgScope {
    pub fn new(nsg: usize) -> Self {
        assert!(
            matches!(nsg, 4 | 8 | 16),
            "unsupported KDA simdgroup count {nsg}"
        );
        let previous = GLM53_KDA_NSG_OVERRIDE.with(|slot| {
            let previous = slot.get();
            slot.set(Some(nsg));
            previous
        });
        Self { previous }
    }
}

impl Drop for Glm53KdaNsgScope {
    fn drop(&mut self) {
        GLM53_KDA_NSG_OVERRIDE.with(|slot| slot.set(self.previous));
    }
}

pub(crate) fn glm53_kda_nsg_override() -> Option<usize> {
    GLM53_KDA_NSG_OVERRIDE.with(Cell::get)
}

/// Scoped switch for interleaved comparisons of DS4's serial and four-row
/// one-token indexer-score schedules.
pub struct Glm53IndexerDecodeRows4Scope {
    previous: Option<bool>,
}

impl Glm53IndexerDecodeRows4Scope {
    pub fn new(use_rows4: bool) -> Self {
        let previous = GLM53_INDEXER_DECODE_ROWS4_OVERRIDE.with(|slot| {
            let previous = slot.get();
            slot.set(Some(use_rows4));
            previous
        });
        Self { previous }
    }
}

impl Drop for Glm53IndexerDecodeRows4Scope {
    fn drop(&mut self) {
        GLM53_INDEXER_DECODE_ROWS4_OVERRIDE.with(|slot| slot.set(self.previous));
    }
}

pub(crate) fn glm53_indexer_decode_rows4_override() -> Option<bool> {
    GLM53_INDEXER_DECODE_ROWS4_OVERRIDE.with(Cell::get)
}

/// Scoped switch for interleaved comparisons of DS4's threadgroup and SIMD
/// shuffle top-k sort networks.
pub struct Glm53IndexerTopkShuffleScope {
    previous: Option<bool>,
}

impl Glm53IndexerTopkShuffleScope {
    pub fn new(use_shuffle: bool) -> Self {
        let previous = GLM53_INDEXER_TOPK_SHUFFLE_OVERRIDE.with(|slot| {
            let previous = slot.get();
            slot.set(Some(use_shuffle));
            previous
        });
        Self { previous }
    }
}

impl Drop for Glm53IndexerTopkShuffleScope {
    fn drop(&mut self) {
        GLM53_INDEXER_TOPK_SHUFFLE_OVERRIDE.with(|slot| slot.set(self.previous));
    }
}

pub(crate) fn glm53_indexer_topk_shuffle_override() -> Option<bool> {
    GLM53_INDEXER_TOPK_SHUFFLE_OVERRIDE.with(Cell::get)
}

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
    /// quantized_matmul mis-handles batched (3-D+) inputs at M > 1 (the values
    /// are wrong entirely), while 2-D is exact.
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

    /// GLM affine projection using DS4's native tiny-row/matrix crossover.
    /// The independent native fixtures and same-weight full-model captures
    /// recorded under resident-port-20261102 establish this arithmetic law.
    pub fn forward_glm53_prefill(&self, x: &Array) -> lisa_mlx::error::Result<Array> {
        let shape = x.shape();
        let rows: i32 = shape[..shape.len() - 1].iter().product::<i32>().max(1);
        let k = x.dim(-1) as usize;
        let n = self.dims_out();
        let mpp_override = glm53_q8_mpp_prefill_override();
        if mpp_override != Some(false)
            && self.bits == 8
            && self.group_size == 64
            && x.dtype() == Dtype::Float32
            && matches!(self.scales.dtype(), Dtype::Float32 | Dtype::Bfloat16)
            && self.biases.dtype() == self.scales.dtype()
            && self.weight.dtype() == Dtype::Uint32
            && k % 64 == 0
            && self.weight.shape()[1] as usize * 4 == k
        {
            let flat = x.reshape(&[rows, k as i32])?.contiguous()?;
            let output = if rows > 16 {
                Array::new(lisa_mlx::jit::glm53_affine_q8_prefill(
                    &x.t.device(),
                    &flat.t,
                    &self.weight.t,
                    &self.scales.contiguous()?.t,
                    &self.biases.contiguous()?.t,
                )?)
            } else if rows == 1 || k % 128 == 0 {
                Array::new(lisa_mlx::jit::glm53_affine_q8_tiny_native(
                    &x.t.device(),
                    &flat.t,
                    &self.weight.t,
                    &self.scales.contiguous()?.t,
                    &self.biases.contiguous()?.t,
                )?)
            } else {
                lisa_mlx::models::glm53::kernels::glm53_affine_q8_mv_ds4(
                    &self.weight,
                    &self.scales,
                    &self.biases,
                    &flat,
                    &lisa_mlx::Stream::thread_local_or_default(),
                )
                .ok_or_else(|| {
                    lisa_mlx::error::Error::msg(
                        "GLM-5.3 DS4 affine-Q8 prefill kernel declined a supported shape",
                    )
                })?
            };
            return output.reshape(&[&shape[..shape.len() - 1], &[n as i32]].concat());
        }
        self.forward_serial_rows(x)
    }

    /// Forward independent GLM decode rows with the same one-row qmv
    /// arithmetic schedule as serial sessions, sharing one dispatch when the
    /// packed affine shape is qmv-fast eligible.
    pub fn forward_serial_rows(&self, x: &Array) -> lisa_mlx::error::Result<Array> {
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
        self.group_size = detect_group_size(self.weight.shape()[1], self.scales.shape()[1], bits)
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
