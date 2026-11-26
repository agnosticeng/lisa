//! Device-independent binding of model-owned attention append plans.
use crate::{
    cache::indexer::{CacheDevice, IndexerTape},
    execution::DeviceContext,
};
use lisa_backend::{ScalarType as Dtype, TensorAccess};
/// Contiguous KV cache for one full-attention layer: `[B, kvHeads, capacity, D]`
/// buffers for K and V, grown by doubling, with an offset for the live length.
pub struct FullAttentionCache<D: CacheDevice> {
    pub dtype: Dtype,
    /// Optional model-owned PLE convolution history on this attention layer.
    pub ple_conv: Option<D::Tensor>,
    pub capture_ple_full: Option<D::Tensor>,
    pub keys: Option<D::Tensor>,
    pub values: Option<D::Tensor>,
    pub offset: usize,
    pub capacity: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub indexer_tape: IndexerTape<D>,
    /// Whether `indexer_tape` is the full token history of this stream. False for
    /// packed/compacted caches (the tape is not maintained there), so the QSA
    /// indexer is skipped and attention stays exact causal.
    pub indexer_ok: bool,
    /// Ragged batching: each stream's next absolute position. `None` = a single
    /// stream whose position is `offset`. When set, `offset` is the packed
    /// (shared) index and this carries the per-stream token positions for RoPE.
    pub next_pos: Option<Vec<usize>>,
    /// Ragged batching: the per-stream attention keep mask `[B,1,S,L]` (bool),
    /// rebuilt each step because L grows. Replaces the causal/varied mask.
    pub attn_mask: Option<D::Tensor>,
}

impl<D: CacheDevice> FullAttentionCache<D> {
    pub fn new(context: DeviceContext<D>, kv_heads: usize, head_dim: usize) -> Self {
        Self::with_dtype(context, kv_heads, head_dim, Dtype::BF16)
    }

    /// Explicit cache precision is part of the owning model's numerical law.
    pub fn with_dtype(
        context: DeviceContext<D>,
        kv_heads: usize,
        head_dim: usize,
        dtype: impl TryInto<Dtype, Error: std::fmt::Debug>,
    ) -> Self {
        let dtype = dtype.try_into().expect("unsupported cache scalar type");
        assert!(matches!(dtype, Dtype::BF16 | Dtype::F32));
        Self {
            dtype,
            keys: None,
            values: None,
            offset: 0,
            capacity: 0,
            kv_heads,
            head_dim,
            indexer_tape: IndexerTape::new(context),
            indexer_ok: true,
            ple_conv: None,
            capture_ple_full: None,
            next_pos: None,
            attn_mask: None,
        }
    }

    pub fn append_plan(&self, batch: i32, rows: i32) -> lisa_models::attention_cache::Append {
        lisa_models::attention_cache::Append {
            batch,
            heads: self.kv_heads as i32,
            width: self.head_dim as i32,
            rows,
            offset: self.offset,
            capacity: self.capacity,
            previous_batch: self.keys.as_ref().map(|k| k.dim(0)),
            dtype: if self.dtype == Dtype::F32 {
                lisa_backend::ScalarType::F32
            } else {
                lisa_backend::ScalarType::BF16
            },
        }
    }
    /// Execute the model-owned append law and install its resulting storage.
    pub fn update(
        &mut self,
        keys: &D::Tensor,
        values: &D::Tensor,
    ) -> anyhow::Result<(D::Tensor, D::Tensor)> {
        let spec = self.append_plan(keys.dim(0), keys.dim(2));
        let (offset, capacity, _) = spec.layout().map_err(anyhow::Error::msg)?;
        let mut inputs = vec![keys, values];
        if let (Some(k), Some(v)) = (&self.keys, &self.values) {
            inputs.extend([k, v]);
        }
        let output = self.indexer_tape.context.with_slot(
            "attention-cache.append",
            || {
                std::cell::RefCell::new(crate::core::graph::PlanCache::new(
                    self.indexer_tape.context.device().clone(),
                    16,
                ))
            },
            |plans| {
                plans
                    .borrow_mut()
                    .try_get_or_prepare(spec.clone(), || spec.graph().map_err(anyhow::Error::msg))?
                    .run(&inputs, &mut [])
            },
        )?;
        let mut out = output.into_iter();
        self.keys = out.next();
        self.values = out.next();
        self.offset = offset;
        self.capacity = capacity;
        Ok((out.next().unwrap(), out.next().unwrap()))
    }

    /// Roll the offset back (speculative rollback); bytes past the offset stay.
    pub fn trim(&mut self, n: usize) {
        self.offset = self.offset.saturating_sub(n);
        self.indexer_tape.trim(self.offset);
    }

    /// Set the live length back to `boundary` (prefix-snapshot restore); the
    /// KV bytes past it stay and are overwritten by the next append.
    pub fn restore_offset(&mut self, boundary: usize) {
        if boundary <= self.offset {
            self.indexer_tape.trim(boundary);
            self.offset = boundary;
        }
    }
}
