#![cfg(test)]
//! Split K/V GGUF attention, using the shared reference indexer and cache contracts.
use crate::bindings::glm53_projection::ProjectionExecution;
use crate::core::cache::IndexerTape;
use lisa_metal::Array;
fn glm53_weighted_rms_norm(x: &Array, w: &Array, eps: f32) -> Result<Array> {
    lisa_metal::kernels::sparse::kernels::mtp_rms_norm(
        x,
        w,
        eps,
        &Stream::thread_local_or_default(),
    )
    .context("RMS declined")
}
fn glm53_kv_lora_rms_norm(x: &Array, w: &Array, eps: f32) -> Result<Array> {
    lisa_metal::kernels::sparse::kernels::mtp_kv_lora_rms_norm(
        x,
        w,
        eps,
        &Stream::thread_local_or_default(),
    )
    .context("latent RMS declined")
}

use crate::bindings::glm53_projection::Projection;
use crate::core::gguf_resident::PackedTensor;
use anyhow::{Context, Result, ensure};
use lisa_metal::Stream;
use lisa_metal::kernels::sparse::gguf as gpu;

pub struct GgufDsa {
    pub(super) qa: Projection,
    pub(super) qb: Projection,
    pub(super) kva: Projection,
    pub(super) output: Projection,
    pub(super) key: PackedTensor,
    pub(super) value: PackedTensor,
    pub(super) qnorm: Array,
    pub(super) kvnorm: Array,
}
impl GgufDsa {
    pub(super) fn project(&self, x: &Array) -> Result<(Array, Array, Array)> {
        ensure!(
            x.shape() == [1, 1, 4096],
            "GGUF GLM serial attention expects one row"
        );
        let qr = glm53_weighted_rms_norm(&self.qa.forward(x)?, &self.qnorm, 1e-5)?;
        let q = self.qb.forward(&qr)?;
        let latent = glm53_kv_lora_rms_norm(&self.kva.forward(x)?, &self.kvnorm, 1e-5)?;
        Ok((qr, q, latent))
    }
    pub(super) fn attend(
        &self,
        q: &Array,
        compact: &Array,
        ids: &Array,
        dense: bool,
    ) -> Result<Array> {
        let s = Stream::thread_local_or_default();
        let low = gpu::qk(q, self.key.packed_bytes(), &s)?;
        let attended = if ids.size() > 2051 {
            {
                let lora = lisa_metal::kernels::sparse::kernels::dsa_indexed_lora_wide_native(
                    &q.reshape(&[1, 64, 1, 256])?,
                    &low,
                    compact,
                    ids,
                    0.0625,
                    &s,
                )
                .context("wide attention declined")?;
                gpu::value(&lora, self.value.packed_bytes(), &s)?
            }
        } else {
            gpu::decode(&low, compact, ids, self.value.packed_bytes(), dense, &s)?
        };
        self.output.forward(&attended)
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
