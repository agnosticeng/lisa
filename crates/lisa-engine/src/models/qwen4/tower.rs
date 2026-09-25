//! The Qwen 3.8 Flash-Next tower (`qwen4_exp_text`): embeddings, 48 hybrid
//! decoder layers, the final hyper-connection mixer, and the lm head.

use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, RwLock};

use lisa_mlx::ops::indexing::IndexOp;
use lisa_mlx::{ops, Array, Dtype};

use crate::core::cache::{FullAttentionCache, GdnCache, LayerCache};
use crate::models::qwen4::attention::Attention;
use crate::models::qwen4::config::ModelConfig;
use crate::models::qwen4::gdn::GatedDeltaNet;
use crate::models::qwen4::hyper::GatedResidual;
use crate::core::loader::{Checkpoint, Weights};
use crate::core::norm::Rotary;
use crate::models::qwen4::ple::{NgramTable, PleLayer};
use crate::core::quant::{QuantizedEmbedding, QuantizedLinear};



/// One decoder layer.
pub struct DecoderLayer {
    pub block: Block,
    pub mlp: crate::models::qwen4::moe::SparseMoeBlock,
    pub attn_hc: GatedResidual,
    pub mlp_hc: GatedResidual,
    pub ple: Option<PleLayer>,
    pub has_ple: bool,
}

pub enum Block {
    Linear(GatedDeltaNet),
    Attention(Attention),
}

impl DecoderLayer {
    /// Forward one layer over the hyper stream.
    ///
    /// `token_rows`/`previous_context` feed the PLE hash (host side).
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &mut self,
        residual: &Array,
        pending_out: Option<&Array>,
        pending_inject: Option<&Array>,
        mut cache: Option<&mut LayerCache>,
        rope: &Rotary,
        offset: usize,
        positions: &Array,
        ple_args: Option<(&[Vec<i64>], &[Vec<i64>])>,
        capture: bool,
    ) -> anyhow::Result<(Array, Array, Array)> {
        let attn_scale = self.attn_hc.norm_scale_q().map_err(|e| anyhow::anyhow!("{e}"))?;
        let mlp_scale = self.mlp_hc.norm_scale_q().map_err(|e| anyhow::anyhow!("{e}"))?;
        let hc = self.attn_hc.hc_count as i32;
        let hidden = (self.attn_hc.hidden / self.attn_hc.hc_count) as i32;
        let eps = self.attn_hc.hc_norm.eps;
        let (b, s) = (residual.dim(0), residual.dim(1));
        let rows = b * s;
        let stream = lisa_mlx::Stream::thread_local_or_default();
        let w = hc * hidden;
        let run_in =
            |res: &Array, out: Option<&Array>, inj: Option<&Array>, scale: &Array| -> anyhow::Result<(Array, Array)> {
                // Wide windows use the `_wide` variant (bit-identical, ~20x fewer
                // threads per row).
                let r = lisa_mlx::kernels::inject_norm(res, out, inj, scale, hc, hidden, rows, false, eps, &stream);
                let (st, nm) = r.ok_or_else(|| anyhow::anyhow!("inject_norm kernel unavailable"))?;
                Ok((st.reshape(&[b, s, w])?, nm.reshape(&[b, s, w])?))
            };

        // PLE adds into the stream BEFORE this layer's attention mixer: the
        // engine materializes the stream, adds the PLE, then norms again.
        let (stream_v, normed) = match (self.ple.as_mut(), ple_args) {
            (Some(ple), Some((token_rows, previous_context))) => {
                let (s1, _) = run_in(residual, pending_out, pending_inject, &attn_scale)?;
                let ple_out = match cache.as_deref_mut() {
                    Some(c) => {
                        let (slot, cap) = c.ple_conv_mut();
                        ple.forward(&s1, token_rows, previous_context, slot, capture, cap)?
                    }
                    None => ple.forward(&s1, token_rows, previous_context, &mut None, capture, &mut None)?,
                };
                let s2 = s1.add(&ple_out)?;
                run_in(&s2, None, None, &attn_scale)?
            }
            _ => run_in(residual, pending_out, pending_inject, &attn_scale)?,
        };

        let (input, attn_inject) = self.attn_hc.mix_from_normed(&normed, true)?;
        if let Ok(dir) = std::env::var("LISA_DUMP_DIR") {
            static L0: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);
            if s == 1024 && L0.swap(false, std::sync::atomic::Ordering::SeqCst) {
                for (nm, arr) in [("normed", &normed), ("mix_in", &input)] {
                    let f = arr.as_dtype(lisa_mlx::Dtype::Float32).map_err(|e| anyhow::anyhow!("{e}"))?;
                    let a = f.as_slice::<f32>();
                    let _ = std::fs::write(format!("{dir}/{nm}.bin"),
                        unsafe { std::slice::from_raw_parts(a.as_ptr() as *const u8, a.len() * 4) });
                }
            }
        }
        let attended = match (&mut self.block, cache) {
            (Block::Linear(gdn), Some(LayerCache::Linear(c))) => gdn.forward(&input, Some(c), capture)?,
            (Block::Linear(gdn), None) => gdn.forward(&input, None, capture)?,
            (Block::Attention(attn), Some(LayerCache::Full(c))) => {
                attn.forward(&input, rope, Some(c), offset, positions)?
            }
            (Block::Attention(attn), None) => attn.forward(&input, rope, None, offset, positions)?,
            _ => anyhow::bail!("cache kind does not match layer kind"),
        };

        if let Ok(dir) = std::env::var("LISA_DUMP_DIR") {
            static ATTN_CALLS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
            let ac = ATTN_CALLS.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            let want = std::env::var("LISA_MATTN_CALL").ok().and_then(|v| v.parse::<usize>().ok());
            if want == Some(ac) {
                let a = attended.as_dtype(Dtype::Float32).map_err(|e| anyhow::anyhow!("{e}"))?;
                let a = a.as_slice::<f32>();
                std::fs::write(format!("{dir}/attn{ac}.bin"), unsafe {
                    std::slice::from_raw_parts(a.as_ptr() as *const u8, a.len() * 4)
                })?;
            }
        }
        let (stream_v2, normed2) =
            run_in(&stream_v, Some(&attended), Some(&attn_inject), &mlp_scale)?;
        let (input2, mlp_inject) = self.mlp_hc.mix_from_normed(&normed2, true)?;
        let out = self.mlp.forward(&input2)?;
        Ok((stream_v2, out, mlp_inject))
    }
}

/// Low vocabulary ids up to this bound are the frequency-ranked BPE tokens
/// (the tokenizer assigns ids in merge order). The MTP drafter only needs to
/// rank these plus the trailing special ids.
const DRAFT_SHORTLIST_LOW: usize = 98304;
/// Special/added tokens live from here to the end of the vocabulary.
const DRAFT_SHORTLIST_SPECIAL_FROM: usize = 248044;

/// Build the draft-only shortlist head from `lm_head`: gather the low ids and
/// the trailing special ids into a smaller affine-quantized linear layer. Group
/// quantization is along the contraction axis, so gathering output rows is
/// exact. Returns `(None, None)` for a vocabulary that does not fit the shape.
fn build_draft_shortlist(
    lm_head: &QuantizedLinear,
    vocab: usize,
) -> (Option<QuantizedLinear>, Option<Array>) {
    let low = DRAFT_SHORTLIST_LOW.min(vocab);
    let special_from = DRAFT_SHORTLIST_SPECIAL_FROM.min(vocab);
    if special_from <= low {
        return (None, None);
    }
    let mut ids: Vec<i32> = (0..low as i32).collect();
    ids.extend(special_from as i32..vocab as i32);
    // `quantized_matmul` prefers a multiple-of-8 row count; the pad rows map to
    // id 0 (a real row, so an argmax tie still resolves to id 0).
    while ids.len() % 8 != 0 {
        ids.push(0);
    }
    let idx = Array::from_slice(&ids, &[ids.len() as i32]);
    let head = (|| {
        Some(QuantizedLinear {
            weight: lm_head.weight.take_axis(&idx, 0).ok()?,
            scales: lm_head.scales.take_axis(&idx, 0).ok()?,
            biases: lm_head.biases.take_axis(&idx, 0).ok()?,
            group_size: lm_head.group_size,
            bits: lm_head.bits,
        })
    })();
    let map = Array::from_slice(
        &ids.iter().map(|&x| x as u32).collect::<Vec<u32>>(),
        &[ids.len() as i32],
    );
    eprintln!(
        "[mtp] draft shortlist head engaged: {} rows (ids <{} + {}..{})",
        ids.len(),
        low,
        special_from,
        vocab
    );
    (head, Some(map))
}

/// The full text tower.
pub struct Tower {
    pub embed_tokens: QuantizedEmbedding,
    pub layers: Vec<DecoderLayer>,
    pub final_mixer: GatedResidual,
    pub lm_head: QuantizedLinear,
    pub rope: Rotary,
    pub config: ModelConfig,
    /// The 2-token n-gram history per row (host side; eos-filled initially).
    pub ngram_history: Option<Vec<Vec<i64>>>,
    pub eos_token_id: i64,
    /// The embedded MTP head (loaded, driven by the speculative path).
    pub mtp: Option<crate::models::qwen4::mtp::MtpHead>,
    /// Draft-only shortlist of `lm_head` rows: the low (frequency-ranked) ids
    /// plus the trailing special ids. The draft argmax runs on this reduced
    /// head (`mlxfast`/FR-Spec); a miss only rejects a draft, never changes an
    /// emitted token, so the target path stays exact.
    pub draft_head: Option<QuantizedLinear>,
    /// Shortlist row -> real vocabulary id (device, uint32), used to map the
    /// draft argmax back to a token.
    pub draft_ids: Option<Array>,
}

impl Tower {
    /// Load the whole tower from a model directory.
    pub fn load(dir: &Path, config: ModelConfig) -> anyhow::Result<Self> {
        let checkpoint = Checkpoint::open(dir)?;
        let map = checkpoint.weight_map.clone();

        // The n-gram table is read from disk, never loaded as parameters.
        // Two checkpoint layouts: sharded tensors in the index (HF/MLX), or one
        // merged `ngram.safetensors` (the mlx-serve pack).
        let ngram_prefix = "model.layers.1.ple.ple_embedding.ngram_embedding";
        let sharded = map.keys().any(|n| n.contains("ngram_embedding.shard_"));
        let ngram_table = Arc::new(RwLock::new(if sharded {
            NgramTable::open(dir, &map, ngram_prefix, config.split_ngram_parts)?
        } else {
            NgramTable::open_merged(
                &dir.join("ngram.safetensors"),
                config.split_ngram_parts,
            )?
        }));

        // Stream the shards into a sanitized weight map, excluding the
        // n-gram shards (they stay on disk behind the row source) and the
        // vision tower (dropped by sanitize_name).
        let mut shard_order: Vec<String> = Vec::new();
        for shard in map.values() {
            if !shard_order.contains(shard) {
                shard_order.push(shard.clone());
            }
        }
        let mut expected: Vec<String> = map
            .keys()
            .filter_map(|n| crate::core::loader::sanitize_name(n))
            .collect();
        expected.sort();

        let mut weights: Weights = HashMap::new();
        for shard_name in &shard_order {
            let shard = crate::core::loader::Shard::open(&dir.join(shard_name))?;
            for name in shard.tensor_names() {
                if let Some(sanitized) = crate::core::loader::sanitize_name(name) {
                    if weights.contains_key(&sanitized) {
                        continue;
                    }
                    let arr = crate::core::quant::get_tensor(&shard, name)?;
                    weights.insert(sanitized, arr);
                }
            }
        }

        let missing: Vec<&String> = expected
            .iter()
            .filter(|n| !weights.contains_key(*n))
            .collect();
        if !missing.is_empty() {
            anyhow::bail!(
                "missing {} tensors after load, e.g. {:?}",
                missing.len(),
                &missing[..missing.len().min(5)]
            );
        }

        Self::from_weights(weights, config, ngram_table)
    }

    /// Build the tower from a materialized weight map.
    pub fn from_weights(
        mut w: Weights,
        config: ModelConfig,
        ngram_table: Arc<RwLock<crate::models::qwen4::ple::NgramTable>>,
    ) -> anyhow::Result<Self> {
        let hidden = config.hidden_size;
        let hc = config.hc_count;
        let eps = config.rms_norm_eps;

        let embed_tokens = QuantizedEmbedding::load(&mut w, "model.embed_tokens")?;
        let lm_head = QuantizedLinear::load(&mut w, "lm_head", "")?;

        let ngram_table_for_ple = ngram_table.clone();
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        let ple_indices = config.ple_layer_indices();
        for i in 0..config.num_hidden_layers {
            let prefix = format!("model.layers.{i}");
            let is_linear = config.layer_types[i] != "full_attention";
            let block = if is_linear {
                Block::Linear(GatedDeltaNet::load(
                    &mut w,
                    &format!("{prefix}.linear_attn"),
                    eps,
                    config.linear_num_key_heads,
                    config.linear_num_value_heads,
                    config.linear_key_head_dim,
                    config.linear_value_head_dim,
                    config.linear_conv_kernel_dim,
                    crate::core::quant::GROUP_SIZE,
                    crate::core::quant::BITS,
                )?)
            } else {
                Block::Attention(Attention::load(
                    &mut w,
                    &format!("{prefix}.self_attn"),
                    eps,
                    config.num_attention_heads,
                    config.num_key_value_heads,
                    config.head_dim,
                )?)
            };
            let mlp = crate::models::qwen4::moe::SparseMoeBlock::load(&mut w, &format!("{prefix}.mlp"), config.num_experts_per_tok)?;
            let attn_hc = GatedResidual::load(&mut w, &format!("{prefix}.attn_hyper_connection"), hidden, hc, true)?;
            let mlp_hc = GatedResidual::load(&mut w, &format!("{prefix}.mlp_hyper_connection"), hidden, hc, true)?;
            let ple = if let Some(ordinal) = ple_indices.iter().position(|&p| p == i) {
                Some(PleLayer::load(
                    &mut w,
                    &format!("{prefix}.ple"),
                    eps,
                    hidden,
                    hc,
                    config.ngram_size,
                    config.heads_per_ngram,
                    config.ple_embed_dim,
                    config.ple_conv_kernel_size,
                    ordinal, // index INTO ple_layer_ids (NOT the layer number)
                    config.seed,
                    config.vocab_size,
                    config.ngram_vocab_size_base,
                    config.make_ngram_vocab_size_divisible_by,
                    config.split_ngram_parts,
                    ngram_table_for_ple.clone(),
                )?)
            } else {
                None
            };
            let has_ple = ple.is_some();
            layers.push(DecoderLayer {
                block,
                mlp,
                attn_hc,
                mlp_hc,
                ple,
                has_ple,
            });
        }
        let final_mixer = GatedResidual::load(&mut w, "model.hyper_connection_mixer", hidden, hc, false)?;

        // Draft-only shortlist of `lm_head` rows, built once at load.
        let (draft_head, draft_ids) = build_draft_shortlist(&lm_head, config.vocab_size);

        // The embedded MTP head is a sibling of `model` in the checkpoint.
        let mtp = crate::models::qwen4::mtp::MtpHead::load(&mut w, &config)?;

        let leftover: Vec<String> = w.keys().cloned().collect();
        if !leftover.is_empty() {
            anyhow::bail!(
                "{} weight tensors unused, e.g. {:?}",
                leftover.len(),
                &leftover[..leftover.len().min(5)]
            );
        }

        let mut tower = Self {
            embed_tokens,
            layers,
            final_mixer,
            lm_head,
            rope: Rotary::new(config.rotary_dimensions(), config.rope_theta),
            eos_token_id: config.eos_token_id,
            config,
            ngram_history: None,
            mtp: Some(mtp),
            draft_head,
            draft_ids,
        };
        for layer in tower.layers.iter_mut() {
            if let Some(ple) = layer.ple.as_mut() {
                ple.set_eos(tower.eos_token_id);
            }
        }
        Ok(tower)
    }

    pub fn token_rows(ids: &Array) -> anyhow::Result<Vec<Vec<i64>>> {
        let b = ids.dim(0) as usize;
        let s = ids.dim(1) as usize;
        // Read the native i32 (Metal `to_dtype` lacks I32 -> I64).
        let flat: Vec<i64> = ids
            .as_slice::<i32>()
            .iter()
            .map(|&v| v as i64)
            .collect();
        Ok((0..b).map(|bi| flat[bi * s..(bi + 1) * s].to_vec()).collect())
    }

    /// Forward the tower. Returns `(mixed, multi)` where `mixed` is the
    /// collapsed hidden state [B, S, H] and `multi` is the pre-final-mixer
    /// hyper stream [B, S, hc*H] (consumed by the MTP head later).
    pub fn forward(&mut self, ids: &Array, mut caches: Option<&mut Vec<LayerCache>>) -> anyhow::Result<(Array, Array)> {
        self.forward_capture(ids, caches.as_deref_mut(), false)
    }

    /// Long-context prefill window. A single forward over a `S`-token prompt
    /// allocates MoE activation buffers of `[S*top_k, hidden]` and, once the
    /// QSA tape passes its budget, `O(S^2)` attention masks. Feeding the prompt
    /// in windows of this size keeps every kernel on the small/prefill shapes
    /// it already handles and bounds the activation peak. Windows are causally
    /// equivalent to the full prefill (the cache carries the prefix), which
    /// `session::verify_incremental` pins. Prompts at or below this size take
    /// the original single-forward path (so the golden is untouched).
    pub const PREFILL_CHUNK: usize = 2048;

    /// Chunked prefill: forward `ids` through `caches` in windows and return
    /// the hidden state at the last position.
    pub fn prefill(&mut self, ids: &[u32], caches: &mut Vec<LayerCache>) -> anyhow::Result<Array> {
        anyhow::ensure!(!ids.is_empty(), "empty prefill");
        let chunk = Self::PREFILL_CHUNK;
        if ids.len() <= chunk {
            let arr = i32_arr(ids);
            let (mixed, _) = self.forward(&arr, Some(caches))?;
            return Ok(last_row(&mixed));
        }
        let mut last = None;
        for c in ids.chunks(chunk) {
            let arr = i32_arr(c);
            let (mixed, _) = self.forward(&arr, Some(caches))?;
            last = Some(mixed);
            lisa_mlx::memory::trim_cache();
        }
        Ok(last_row(&last.expect("non-empty chunks")))
    }

    /// Chunked prefill returning the hyper stream `multi` for every fed token
    /// (needed to prime the MTP head over a long suffix).
    pub fn prefill_multi(
        &mut self,
        ids: &[u32],
        caches: &mut Vec<LayerCache>,
    ) -> anyhow::Result<(Array, Array)> {
        anyhow::ensure!(!ids.is_empty(), "empty prefill");
        let chunk = Self::PREFILL_CHUNK;
        let mut multis: Vec<Array> = Vec::new();
        let mut last_mixed: Option<Array> = None;
        for c in ids.chunks(chunk) {
            let arr = i32_arr(c);
            let (mixed, multi) = self.forward(&arr, Some(caches))?;
            last_mixed = Some(mixed);
            multis.push(multi);
            lisa_mlx::memory::trim_cache();
        }
        let mixed = last_mixed.expect("non-empty chunks");
        let multi = if multis.len() == 1 {
            multis.pop().expect("one multi")
        } else {
            let refs: Vec<&Array> = multis.iter().collect();
            ops::concatenate(&refs, 1).map_err(|e| anyhow::anyhow!("{e}"))?
        };
        Ok((last_row(&mixed), multi))
    }

    /// Forward with an explicit speculative-capture flag (see `GdnCache`).
    pub fn forward_capture(
        &mut self,
        ids: &Array,
        mut caches: Option<&mut Vec<LayerCache>>,
        capture: bool,
    ) -> anyhow::Result<(Array, Array)> {
        let b = ids.dim(0) as usize;
        let s = ids.dim(1) as usize;
        let token_rows = Self::token_rows(ids)?;

        let hidden = self.embed_tokens.forward(ids)?;
        if let Ok(dir) = std::env::var("LISA_DUMP_DIR") {
            if s == 1024 {
                let f = hidden.as_dtype(lisa_mlx::Dtype::Float32).map_err(|e| anyhow::anyhow!("{e}"))?;
                let a = f.as_slice::<f32>();
                let _ = std::fs::write(
                    format!("{dir}/embed.bin"),
                    unsafe { std::slice::from_raw_parts(a.as_ptr() as *const u8, a.len() * 4) },
                );
            }
        }
        let mut residual =
            ops::tile(&hidden, &[1, 1, self.config.hc_count as i32]).map_err(|e| anyhow::anyhow!("{e}"))?;

        // The n-gram hash reads ids in order; the history carries across calls.
        let ctx_len = self.config.ngram_size - 1;
        let previous_context: Vec<Vec<i64>> = match &self.ngram_history {
            Some(hist) => hist.clone(),
            None => vec![vec![self.eos_token_id; ctx_len]; b],
        };
        let mut new_history = Vec::with_capacity(b);
        for bi in 0..b {
            let mut h = previous_context[bi].clone();
            h.extend_from_slice(&token_rows[bi]);
            new_history.push(h[h.len() - ctx_len..].to_vec());
        }
        self.ngram_history = Some(new_history);

        let offset = caches
            .as_deref()
            .and_then(|c| c.iter().find_map(|lc| match lc {
                LayerCache::Full(f) => Some(f.offset),
                _ => None,
            }))
            .unwrap_or(0);
        // RoPE positions. Ragged batching carries a per-stream next position on
        // the packed attention caches; otherwise every stream shares `offset`.
        let positions: Array = match caches.as_deref().and_then(|c| c.iter().find_map(|lc| match lc {
            LayerCache::Full(f) => f.next_pos.clone(),
            _ => None,
        })) {
            Some(next) => {
                let mut data: Vec<f32> = Vec::with_capacity(next.len() * s);
                for p in &next {
                    for j in 0..s {
                        data.push((*p + j) as f32);
                    }
                }
                Array::from_slice(&data, &[next.len() as i32, s as i32])
            }
            None => crate::core::norm::positions(offset, s)?,
        };

        let profile = lisa_mlx::env_flag("LISA_PROFILE");
        if lisa_mlx::env_flag("LISA_ATTN_DEBUG") {
            eprintln!("[model] ids={:?} s={s} positions={:?}", ids.shape(), positions.shape());
        }
        // Prefill pipeline: the engine overlaps CPU graph construction with GPU
        // execution by `asyncEval`-ing in short chunks. Without it Lisa builds
        // the whole 48-layer graph, then the single terminal `eval` runs it with
        // no overlap. Every N layers we async-eval (0 disables). Chunks of 2-6
        // are all ~equal and worth ~8% prefill; 3 matches the engine.
        let eval_chunk = 3usize;
        let mut pending_out: Option<Array> = None;
        let mut pending_inject: Option<Array> = None;
        static TOWER_CALLS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let tcall = TOWER_CALLS.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let layer_call = std::env::var("LISA_LAYER_CALL").ok().and_then(|v| v.parse::<usize>().ok());
        for (i, layer) in self.layers.iter_mut().enumerate() {
            let cache = caches.as_deref_mut().map(|c| &mut c[i]);
            let ple_args = if layer.has_ple {
                Some((&token_rows[..], &previous_context[..]))
            } else {
                None
            };
            let t0 = std::time::Instant::now();
            let (stream, out, inject_w) = layer.forward(
                &residual,
                pending_out.as_ref(),
                pending_inject.as_ref(),
                cache,
                &self.rope,
                offset,
                &positions,
                ple_args,
                capture,
            )?;
            residual = stream;
            pending_out = Some(out);
            pending_inject = Some(inject_w);
            if profile {
                residual.eval().map_err(|e| anyhow::anyhow!("{e}"))?;
                let kind = if layer.has_ple { "ple" } else if matches!(layer.block, Block::Linear(_)) { "gdn" } else { "attn" };
                eprintln!("layer {i:2} ({kind}) s={s}: {:>8.3} ms", t0.elapsed().as_secs_f64() * 1e3);
            } else if s > 1 && eval_chunk > 0 && (i + 1) % eval_chunk == 0 {
                let mut outs: Vec<&Array> = vec![&residual];
                if let Some(o) = pending_out.as_ref() {
                    outs.push(o);
                }
                if let Some(o) = pending_inject.as_ref() {
                    outs.push(o);
                }
                lisa_mlx::transforms::async_eval(outs).map_err(|e| anyhow::anyhow!("{e}"))?;
            }
            if let Ok(dir) = std::env::var("LISA_DUMP_DIR") {
                let name = if lisa_mlx::env_flag("LISA_DUMP_ALL") {
                    format!("{dir}/T{tcall}_L{i:02}.bin")
                } else if layer_call == Some(tcall) {
                    format!("{dir}/L{tcall}_layer_{i:02}.bin")
                } else {
                    format!("{dir}/layer_{i:02}.bin")
                };
                let a = residual.as_dtype(Dtype::Float32).map_err(|e| anyhow::anyhow!("{e}"))?;
                let a = a.as_slice::<f32>();
                std::fs::write(name, unsafe {
                    std::slice::from_raw_parts(a.as_ptr() as *const u8, a.len() * 4)
                })?;
            }
        }

        // Final mixer: injectNorm(residual, pendingOut, pendingInject) then the
        // inject-less hcMix. `multi` is the resulting stream.
        let final_scale = self.final_mixer.norm_scale_q().map_err(|e| anyhow::anyhow!("{e}"))?;
        let hc = self.config.hc_count as i32;
        let hidden_i = self.config.hidden_size as i32;
        let rows = (b * s) as i32;
        let stream = lisa_mlx::Stream::thread_local_or_default();
        let final_r = lisa_mlx::kernels::inject_norm(
            &residual, pending_out.as_ref(), pending_inject.as_ref(), &final_scale,
            hc, hidden_i, rows, false, self.config.rms_norm_eps, &stream,
        );
        let (multi, final_normed) =
            final_r.ok_or_else(|| anyhow::anyhow!("inject_norm kernel unavailable"))?;
        let multi = multi.reshape(&[b as i32, s as i32, hc * hidden_i])?;
        let final_normed = final_normed.reshape(&[b as i32, s as i32, hc * hidden_i])?;
        let (mixed, _) = self.final_mixer.mix_from_normed(&final_normed, false).map_err(|e| anyhow::anyhow!("{e}"))?;
        if let Ok(dir) = std::env::var("LISA_DUMP_DIR") {
            for (name, arr) in [("mixed", &mixed), ("multi", &multi)] {
                let a = arr.as_dtype(Dtype::Float32).map_err(|e| anyhow::anyhow!("{e}"))?;
                let a = a.as_slice::<f32>();
                std::fs::write(format!("{dir}/{name}.bin"), unsafe {
                    std::slice::from_raw_parts(a.as_ptr() as *const u8, a.len() * 4)
                })?;
            }
        }
        Ok((mixed, multi))
    }

    /// Set the PLE n-gram context tail (the last `ngram_size - 1` committed
    /// tokens). Used by the speculative driver to undo a rejected draft.
    pub fn set_ngram_tail(&mut self, tail: &[i64]) {
        self.ngram_history = Some(vec![tail.to_vec()]);
    }

    /// Logits from the mixed hidden state: [B, S, vocab].
    pub fn head(&self, mixed: &Array) -> lisa_mlx::error::Result<Array> {
        self.lm_head.forward(mixed)
    }

    /// JIT-compile and warm every kernel/MoE path (both the narrow decode paths
    /// and the wide prefill paths) so the first measured forward does not pay
    /// Metal kernel compilation. The reference engine warms at init too.
    ///
    /// The warm inputs are drawn from `seed` (the real prompt) rather than
    /// synthetic zeros: an all-zero activation exercises a different branch and
    /// must not be used here.
    pub fn warmup(&mut self, seed: &[u32]) -> anyhow::Result<()> {
        let shapes: Vec<usize> = vec![1, 9, 40, Self::PREFILL_CHUNK];
        let fallback: Vec<u32> = (0..64).map(|i| 128 + (i * 37) % 4096).collect();
        let base: &[u32] = if seed.is_empty() { &fallback } else { seed };
        for s in shapes {
            let ids: Vec<i32> = (0..s).map(|i| base[i % base.len()] as i32).collect();
            let arr = Array::from_slice(&ids, &[1i32, s as i32]);
            let mut caches = self.new_caches();
            let (mixed, multi) = self.forward(&arr, Some(&mut caches))?;
            let last = mixed.index((.., mixed.dim(1) - 1, ..));
            let logits = self.head(&last)?;
            lisa_mlx::transforms::eval([&logits, &multi]).map_err(|e| anyhow::anyhow!("{e}"))?;
        }
        self.ngram_history = None;
        // Release the warmup buffers before the worker serves (the reference
        // engine's low-memory profile does the same): keeps unified memory free
        // for the real KV/activations. Compiled kernels are unaffected.
        let _ = lisa_mlx::memory::clear_cache();
        Ok(())
    }

    /// Fresh caches for one request.
    pub fn new_caches(&self) -> Vec<LayerCache> {
        self.config
            .layer_types
            .iter()
            .map(|t| {
                if t == "full_attention" {
                    LayerCache::Full(FullAttentionCache::new(
                        self.config.num_key_value_heads,
                        self.config.head_dim,
                    ))
                } else {
                    LayerCache::Linear(GdnCache::new())
                }
            })
            .collect()
    }
}

/// `[1, len]` token ids.
fn i32_arr(ids: &[u32]) -> Array {
    let v: Vec<i32> = ids.iter().map(|&t| t as i32).collect();
    Array::from_slice(&v, &[1i32, ids.len() as i32])
}

/// The last sequence position of a `[B, S, H]` hidden state.
fn last_row(mixed: &Array) -> Array {
    mixed.index((.., mixed.dim(1) - 1, ..))
}

impl crate::models::LanguageModel for Tower {
    fn max_position_embeddings(&self) -> usize {
        self.config.max_position_embeddings
    }
    fn eos_token_id(&self) -> i64 {
        self.eos_token_id
    }
    fn new_caches(&self) -> Vec<LayerCache> {
        self.new_caches()
    }
    fn forward(
        &mut self,
        ids: &Array,
        caches: Option<&mut Vec<LayerCache>>,
    ) -> anyhow::Result<(Array, Array)> {
        self.forward(ids, caches)
    }
    fn forward_capture(
        &mut self,
        ids: &Array,
        caches: Option<&mut Vec<LayerCache>>,
        capture: bool,
    ) -> anyhow::Result<(Array, Array)> {
        self.forward_capture(ids, caches, capture)
    }
    fn prefill(&mut self, tokens: &[u32], caches: &mut Vec<LayerCache>) -> anyhow::Result<Array> {
        self.prefill(tokens, caches)
    }
    fn prefill_multi(
        &mut self,
        tokens: &[u32],
        caches: &mut Vec<LayerCache>,
    ) -> anyhow::Result<(Array, Array)> {
        self.prefill_multi(tokens, caches)
    }
    fn head(&self, mixed: &Array) -> lisa_mlx::error::Result<Array> {
        self.head(mixed)
    }
    fn warmup(&mut self, seed: &[u32]) -> anyhow::Result<()> {
        self.warmup(seed)
    }

    // Host-side rolling n-gram context for the PLE layer.
    fn context_window(&self) -> usize {
        self.config.ngram_size.saturating_sub(1)
    }
    fn context_tails(&self) -> Vec<Vec<i64>> {
        self.ngram_history.clone().unwrap_or_default()
    }
    fn set_context_tails(&mut self, tails: Vec<Vec<i64>>) {
        self.ngram_history = if tails.is_empty() { None } else { Some(tails) };
    }
    fn clear_context(&mut self) {
        self.ngram_history = None;
    }

    // Speculative drafting via the embedded MTP head.
    fn has_drafter(&self) -> bool {
        self.mtp.is_some()
    }
    fn drafter_reset(&mut self) {
        if let Some(h) = self.mtp.as_mut() {
            h.reset_caches();
        }
    }
    fn drafter_trim(&mut self, n: usize) {
        if let Some(h) = self.mtp.as_mut() {
            h.trim_caches(n);
        }
    }
    fn drafter_offset(&self) -> usize {
        self.mtp.as_ref().map(|h| h.cache_offset()).unwrap_or(0)
    }
    fn drafter_restore_offset(&mut self, n: usize) {
        if let Some(h) = self.mtp.as_mut() {
            h.restore_offset(n);
        }
    }
    fn draft_step(&mut self, tokens: &Array, multi: &Array) -> anyhow::Result<(Array, Array)> {
        use lisa_mlx::ops::indexing::IndexOp;
        let offset = self.mtp.as_ref().map(|h| h.cache_offset()).unwrap_or(0);
        let head = self
            .mtp
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("this model has no MTP head"))?;
        let (sample, head_multi) = head.forward(tokens, multi, &self.embed_tokens, offset)?;
        let last = sample.index((.., sample.dim(1) - 1, ..));
        // The draft argmax rides the shortlist head when available; a shortlist
        // miss only rejects a draft (the target verifies every token).
        let logits = match &self.draft_head {
            Some(h) => h.forward(&last)?,
            None => self.lm_head.forward(&last)?,
        };
        if std::env::var("LISA_DUMP_DRAFT").is_ok() {
            static ROUND: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
            let r = ROUND.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            if r >= 200 {
                let l = logits
                    .as_dtype(lisa_mlx::Dtype::Float32)
                    .map_err(|e| anyhow::anyhow!("{e}"))?;
                let v: Vec<f32> = l.as_slice::<f32>().to_vec();
                let mut idx: Vec<usize> = (0..v.len()).collect();
                idx.sort_by(|&a, &b| v[b].partial_cmp(&v[a]).unwrap());
                eprintln!(
                    "[draft-logits] step#{r} top5: {:?}",
                    idx[..5].iter().map(|&i| (i, v[i])).collect::<Vec<_>>()
                );
            }
        }
        // Keep the draft token on the device as `[1, 1]` so the chain does
        // not round-trip through the host between draft steps. The id stays
        // uint32 (the only integer dtype the indexing/cast kernels emit);
        // `token_rows` reads it as i32, same bit pattern.
        let short_idx = lisa_mlx::ops::indexing::argmax_axis(&logits, -1, None)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        let draft_id = match (&self.draft_head, &self.draft_ids) {
            (Some(_), Some(map)) => map
                .take_axis(&short_idx, 0)
                .map_err(|e| anyhow::anyhow!("{e}"))?,
            _ => short_idx,
        };
        let draft_id = draft_id.reshape(&[1, 1])?;
        let m = head_multi
            .index((.., head_multi.dim(1) - 1, ..))
            .contiguous()?;
        if std::env::var("LISA_DUMP_DRAFT").is_ok() {
            static MCALL: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
            let r = MCALL.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            if r >= 200 {
                let f = m
                    .as_dtype(lisa_mlx::Dtype::Float32)
                    .map_err(|e| anyhow::anyhow!("{e}"))?;
                let v: Vec<f32> = f.as_slice::<f32>().to_vec();
                let sum: f32 = v.iter().sum();
                eprintln!("[draft-m] step#{r} n={} sum={:.6e} head={:?}", v.len(), sum, &v[..4.min(v.len())]);
            }
        }
        Ok((draft_id, m))
    }
}
