#![cfg(test)]
//! The reference's scalar V4.1 attention graph with original GGUF projections.

use crate::bindings::deepseek41::state::{State, store_row, view};
use crate::bindings::deepseek41_projection_oracle as projection;
use crate::core::capture::Capture;
use crate::core::gguf_resident::PackedTensor;
use anyhow::{Context, Result};
use lisa_metal::kernels::ggml::{self as gpu, ActivationFormat};
use lisa_metal::{Array, Stream};
use lisa_models::deepseek41::rope_frequencies;

struct Compressor {
    kv: PackedTensor,
    gate: Option<PackedTensor>,
    norm: Array,
    index_k: PackedTensor,
    index_norm: Array,
}
struct Indexer {
    q: PackedTensor,
    weights: PackedTensor,
}

pub(super) struct Projections {
    qr: Array,
    kv: Array,
    q: Array,
}

impl Projections {
    pub(super) fn row(&self, row: usize) -> Result<Self> {
        Ok(Self {
            qr: view(&self.qr, row * 1280, &[1, 1280])?,
            kv: view(&self.kv, row * 512, &[1, 1, 512])?,
            q: view(&self.q, row * 64 * 512, &[1, 64, 512])?,
        })
    }
}

pub struct Attention {
    layer: usize,
    ratio: usize,
    q_a: PackedTensor,
    q_norm: Array,
    kv: PackedTensor,
    kv_norm: Array,
    q_b: PackedTensor,
    o_a: PackedTensor,
    o_b: PackedTensor,
    sinks: Array,
    compressor: Option<Compressor>,
    indexer: Option<Indexer>,
}

impl Attention {
    /// Row-independent projections retain the original matvec reductions.
    /// The causal cache/index/attention work runs in position order afterward.
    pub(super) fn project(&self, norm: &Array, stream: &Stream) -> Result<Projections> {
        let (qr, kv) = projection::query_kv_pair(&self.q_a, &self.kv, norm, stream)?;
        let (qr, kv) = gpu::rms_norm_pair(&qr, &self.q_norm, &kv, &self.kv_norm, 1e-20, stream)?;
        let q = projection::linear(&self.q_b, &qr, true, stream)?;
        Ok(Projections { qr, kv, q })
    }

    pub(super) fn forward(
        &self,
        norm: &Array,
        projected: Projections,
        state: &mut State,
        selected: &mut Option<Array>,
        candidates: &mut Option<gpu::IndexCandidates>,
        scratch: &mut gpu::DecodeAttentionScratch,
        rope: &Array,
        capture: Option<&Capture>,
        stream: &Stream,
    ) -> Result<Array> {
        let pos = state.position;
        let record = |key: &str, value: &Array| {
            if let Some(capture) = capture.filter(|c| c.wants(self.layer)) {
                capture.store_ephemeral(&format!("p{pos}.l{}.attention.{key}", self.layer), value);
            }
        };
        let owner = if self.layer < 8 {
            0
        } else if self.layer < 14 {
            1
        } else if self.layer < 20 {
            2
        } else {
            3
        };
        let compressed = self.ratio != 0;
        let n_comp = if compressed {
            (pos + 1) / self.ratio
        } else {
            0
        };
        let Projections { qr, kv, mut q } = projected;
        gpu::rope_in_place(
            &mut q,
            pos as u32,
            1,
            rope_frequencies(compressed),
            false,
            false,
            stream,
        )?;
        gpu::rope_quantize_store(
            &kv.reshape(&[1, 512])?,
            &mut state.raw[self.layer],
            pos % 128,
            pos as u32,
            rope_frequencies(compressed),
            ActivationFormat::Fp8E8M0,
            stream,
        )?;
        record("query", &q);
        record(
            "raw",
            &view(&state.raw[self.layer], (pos % 128) * 512, &[1, 512])?,
        );

        if let Some(compressor) = &self.compressor {
            let cache = &mut state.owners[owner];
            let pool_kv = projection::linear(&compressor.kv, norm, self.ratio == 1, stream)?;
            record("pool", &pool_kv);
            let latent = if let Some(gate) = &compressor.gate {
                let score = projection::linear(gate, norm, false, stream)?;
                record("pool_score", &score);
                let latent = if pos % 2 == 1 {
                    Some(gpu::pool_pair(
                        &cache.previous_kv,
                        &cache.previous_score,
                        &pool_kv,
                        &score,
                        stream,
                    )?)
                } else {
                    None
                };
                cache.previous_kv = pool_kv;
                cache.previous_score = score;
                latent
            } else {
                Some(pool_kv)
            };
            if let Some(latent) = latent {
                let latent = gpu::rms_norm(&latent, Some(&compressor.norm), 1e-20, true, stream)?;
                record("latent_norm", &latent);
                let index = projection::linear(&compressor.index_k, &latent, true, stream)?;
                let index =
                    gpu::rms_norm(&index, Some(&compressor.index_norm), 1e-20, true, stream)?;
                let rope_pos = (pos + 1 - self.ratio) as u32;
                gpu::rope_quantize_store(
                    &index,
                    &mut cache.index,
                    n_comp - 1,
                    rope_pos,
                    rope_frequencies(true),
                    ActivationFormat::Fp4E8M0,
                    stream,
                )?;
                gpu::rope_quantize_store(
                    &latent,
                    &mut cache.compressed,
                    n_comp - 1,
                    rope_pos,
                    rope_frequencies(true),
                    ActivationFormat::Fp4E4M3,
                    stream,
                )?;
                let index = view(&cache.index, (n_comp - 1) * 128, &[1, 128])?;
                let latent = view(&cache.compressed, (n_comp - 1) * 512, &[1, 512])?;
                record("index_key", &index);
                record("latent", &latent);
                if let Some(mirror) = &mut cache.index_bf16 {
                    store_row(mirror, n_comp - 1, &gpu::index_keys_bf16(&index, stream)?)?;
                }
            }
        }
        if let Some(indexer) = &self.indexer
            && n_comp > 0
        {
            let mut query =
                projection::linear(&indexer.q, &qr, true, stream)?.reshape(&[32, 128])?;
            gpu::index_query_rope_quantize(&mut query, rope, stream)?;
            record("index_query", &query);
            let weights =
                projection::linear(&indexer.weights, norm, true, stream)?.reshape(&[32])?;
            let cache = &state.owners[owner];
            let picked = if self.layer > 20 && n_comp > 16384 {
                let keys = view(&cache.index, 0, &[n_comp as i32, 128])?;
                gpu::index_candidates_select(
                    &query,
                    &weights,
                    &keys,
                    candidates
                        .as_ref()
                        .context("DeepSeek candidate list was not published")?,
                    stream,
                )?
            } else {
                let keys = if n_comp >= 8192 {
                    view(
                        cache
                            .index_bf16
                            .as_ref()
                            .context("missing DeepSeek BF16 key mirror")?,
                        0,
                        &[n_comp as i32, 128],
                    )?
                } else {
                    view(&cache.index, 0, &[n_comp as i32, 128])?
                };
                let scores = gpu::index_scores(&query, &weights, &keys, stream)?;
                if self.layer == 20 && n_comp > 16384 {
                    *candidates = Some(gpu::index_candidates(&scores, stream)?);
                }
                gpu::index_topk(&scores, n_comp.min(512) as i32, stream)?
            };
            *selected = Some(picked.reshape(&[n_comp.min(512) as i32])?);
        }
        let raw_count = (pos + 1).min(128) as u32;
        let comp = if n_comp > 0 {
            Some(view(
                &state.owners[owner].compressed,
                0,
                &[n_comp as i32, 512],
            )?)
        } else {
            None
        };
        let selection = if let Some(comp) = &comp {
            Some((
                comp,
                selected
                    .as_ref()
                    .context("DeepSeek compressed selection was not published")?,
            ))
        } else {
            None
        };
        let heads = gpu::attention_decode_with_rope(
            &q.reshape(&[64, 512])?,
            &self.sinks,
            &state.raw[self.layer],
            raw_count,
            ((pos + 1 - raw_count as usize) % 128) as u32,
            selection,
            scratch,
            Some((rope, compressed)),
            stream,
        )?
        .reshape(&[1, 64, 512])?;
        let low = projection::attention_output_groups(&self.o_a, &heads, stream)?;
        projection::linear(&self.o_b, &low, true, stream)
    }
}

pub fn fixture(
    w: &[PackedTensor],
    layer: usize,
    ratio: usize,
    compressor: bool,
    gate: bool,
    indexer: bool,
) -> Result<Attention> {
    let plain = |i: usize| projection::plain_view(&w[i]);
    Ok(Attention {
        layer,
        ratio,
        q_a: w[0].clone(),
        q_norm: plain(1)?,
        kv: w[2].clone(),
        kv_norm: plain(3)?,
        q_b: w[4].clone(),
        o_a: w[5].clone(),
        o_b: w[6].clone(),
        sinks: plain(7)?,
        compressor: if compressor {
            Some(Compressor {
                kv: w[8].clone(),
                gate: gate.then(|| w[9].clone()),
                norm: plain(10)?,
                index_k: w[11].clone(),
                index_norm: plain(12)?,
            })
        } else {
            None
        },
        indexer: if indexer {
            Some(Indexer {
                q: w[13].clone(),
                weights: w[14].clone(),
            })
        } else {
            None
        },
    })
}
pub fn compare_projections(a: &Projections, qr: &Array, kv: &Array, q: &Array) -> Result<()> {
    for (x, y) in [(&a.qr, qr), (&a.kv, kv), (&a.q, q)] {
        assert_eq!(x.to_vec1::<u32>()?, y.to_vec1::<u32>()?);
    }
    Ok(())
}
pub fn candidate_tensor(c: &gpu::IndexCandidates) -> Array {
    c.tensor().clone()
}
