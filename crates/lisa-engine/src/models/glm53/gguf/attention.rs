//! Split K/V GGUF attention, using the shared DS4 indexer and cache contracts.
use super::*;
use crate::core::gguf_resident::{PackedTensor, ResidentWeights};
use crate::models::glm53::gguf::projection::{Projection, plain};
use anyhow::{Context, Result, ensure};
use lisa_mlx::Stream;
use lisa_mlx::models::glm53::gguf as gpu;

pub struct GgufDsa {
    qa: Projection,
    qb: Projection,
    kva: Projection,
    output: Projection,
    key: PackedTensor,
    value: PackedTensor,
    qnorm: Array,
    kvnorm: Array,
    indexer: Indexer,
}
impl GgufDsa {
    pub fn load(w: &ResidentWeights, layer: usize) -> Result<Self> {
        let p = |name: &str| format!("blk.{layer}.{name}");
        let f = |name: &str| plain(w, &p(name));
        let projection = |name: &str| Projection::load(w, &p(name));
        let packed = |name: &str| {
            w.tensors
                .get(&p(name))
                .cloned()
                .context("missing GLM attention weight")
        };
        Ok(Self {
            qa: projection("attn_q_a.weight")?,
            qb: projection("attn_q_b.weight")?,
            kva: projection("attn_kv_a_mqa.weight")?,
            output: projection("attn_output.weight")?,
            key: packed("attn_k_b.weight")?,
            value: packed("attn_v_b.weight")?,
            qnorm: f("attn_q_a_norm.weight")?,
            kvnorm: f("attn_kv_a_norm.weight")?,
            indexer: Indexer {
                wq_b_w: f("indexer.attn_q_b.weight")?,
                wk_w: f("indexer.attn_k.weight")?,
                k_norm_w: f("indexer.k_norm.weight")?,
                k_norm_b: f("indexer.k_norm.bias")?,
                weights_w: f("indexer.proj.weight")?,
                gate_w: f("indexer.pool_gate.weight")?,
                ape: f("indexer.pool_ape.weight")?,
                heads: 32,
                head_dim: 128,
                topk: 2048,
                kpool: 4,
                always_tail: true,
            },
        })
    }
    fn project(&self, x: &Array) -> Result<(Array, Array, Array)> {
        ensure!(
            x.shape() == [1, 1, 4096],
            "GGUF GLM serial attention expects one row"
        );
        let qr = glm53_weighted_rms_norm(&self.qa.forward(x)?, &self.qnorm, 1e-5)?;
        let q = self.qb.forward(&qr)?;
        let latent = glm53_kv_lora_rms_norm(&self.kva.forward(x)?, &self.kvnorm, 1e-5)?;
        Ok((qr, q, latent))
    }
    fn attend(&self, q: &Array, compact: &Array, ids: &Array, dense: bool) -> Result<Array> {
        let s = Stream::thread_local_or_default();
        let low = gpu::qk(q, self.key.packed_bytes(), &s)?;
        let attended = if ids.size() > 2051 {
            gpu::wide_decode(q, &low, compact, ids, self.value.packed_bytes(), &s)?
        } else {
            gpu::decode(&low, compact, ids, self.value.packed_bytes(), dense, &s)?
        };
        self.output.forward(&attended)
    }
    pub fn forward(
        &self,
        x: &Array,
        cache: &mut FullAttentionCache,
        layer: usize,
    ) -> Result<Array> {
        let (qr, q, latent) = self.project(x)?;
        let (key, gate) =
            self.indexer
                .project_pair(x, &self.indexer.wk_w, x, &self.indexer.gate_w)?;
        cache.indexer_tape.append_fields(&key, &gate, 4)?;
        cache.indexer_tape.append_compact_f16(&latent)?;
        self.indexer
            .update_pooled_cache(&mut cache.indexer_tape, None, layer)?;
        cache.offset = cache.indexer_tape.total;
        let total = cache.offset;
        let dense = total <= 2051;
        let ids = if dense {
            Array::from_slice(
                &(0..total as i32).collect::<Vec<_>>(),
                &[1, 1, total as i32],
            )
        } else {
            self.indexer
                .select_ids_gpu(
                    &mut cache.indexer_tape,
                    x,
                    &qr,
                    total,
                    1,
                    None,
                    layer,
                    false,
                )?
                .0
        };
        let compact = cache
            .indexer_tape
            .compact_slice(0, total)
            .context("missing GLM compact cache")?;
        self.attend(&q, &compact, &ids, dense)
    }
    pub fn mtp(
        &self,
        x: &Array,
        tape: &mut IndexerTape,
        position: usize,
        min_position: usize,
    ) -> Result<Array> {
        ensure!(min_position <= position, "invalid GLM MTP window");
        let (_, q, latent) = self.project(x)?;
        tape.write_compact_f16_at(&latent, position)?;
        let compact = tape
            .compact_slice(0, position + 1)
            .context("missing GLM MTP cache")?;
        let ids = Array::from_slice(
            &(min_position as i32..=position as i32).collect::<Vec<_>>(),
            &[1, 1, (position - min_position + 1) as i32],
        );
        self.attend(&q, &compact, &ids, false)
    }
}
