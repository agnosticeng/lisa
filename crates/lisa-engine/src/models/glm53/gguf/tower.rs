//! GLM's native GGUF graph. Serial execution is the numerical baseline for
//! wider prefill and verification dispatches; no affine weight adapter is used.
use super::super::{config::ModelConfig, kda::Kda, mhc::Hc, sparse::GgufDsa};
use super::{
    moe::Mlp,
    projection::{Projection, plain},
    weights,
};
use crate::{
    core::{
        cache::{FullAttentionCache, GdnCache, IndexerTape, LayerCache},
        capture::Capture,
        gguf_resident::ResidentWeights,
        loader::Weights,
    },
    models::LanguageModel,
};
use anyhow::{Context, Result, ensure};
use lisa_gguf::{Package, Tensor, quant};
use lisa_mlx::models::glm53::kernels as gpu;
use lisa_mlx::{Array, Dtype, Stream};
use std::{os::unix::fs::MetadataExt, path::Path};

enum Attention {
    Kda(Kda),
    Dsa(GgufDsa),
}
struct Layer {
    attn: Attention,
    mlp: Mlp,
    hc_attn: Hc,
    hc_ffn: Hc,
    in_norm: Array,
    post_norm: Array,
}
struct Mtp {
    eh: Projection,
    enorm: Array,
    hnorm: Array,
    anorm: Array,
    fnorm: Array,
    norm: Array,
    attn: GgufDsa,
    mlp: Mlp,
}
pub struct Tower {
    // Drain outstanding commands before any zero-copy weight views disappear.
    _weights: ResidentWeights,
    config: ModelConfig,
    layers: Vec<Layer>,
    mtp: Mtp,
    embedding: Tensor,
    output: Projection,
    norm: Array,
    capacity: usize,
    law: u64,
    capture: Option<Capture>,
    step: std::cell::Cell<usize>,
}
fn rms(x: &Array, w: &Array) -> Result<Array> {
    gpu::glm53_ds4_rms_norm(x, w, 1e-5, &Stream::thread_local_or_default())
        .context("GLM GGUF RMSNorm declined")
}
fn mean(x: &Array) -> Result<Array> {
    let weights = Array::from_slice(&[0.25f32; 4], &[1, 1, 4]);
    gpu::hc_weighted_sum(
        &x.reshape(&[1, 1, 4, 4096])?,
        &weights,
        &Stream::thread_local_or_default(),
    )
    .context("GLM stream reduction declined")
}
impl Tower {
    pub fn load(path: &Path) -> Result<Self> {
        let package = Package::open(path)?;
        let config = weights::validate(&package)?;
        let names: Vec<_> = package
            .tensors
            .keys()
            .filter(|n| n.as_str() != "token_embd.weight")
            .map(String::as_str)
            .collect();
        let w = ResidentWeights::load(&package, &names, &Stream::thread_local_or_default())?;
        let mut layers = Vec::new();
        for layer in 0..45 {
            let hc = |half: &str| -> Result<Hc> {
                let mut views = Weights::new();
                for suffix in ["fn", "base", "scale"] {
                    views.insert(
                        format!("hc_{suffix}"),
                        plain(&w, &format!("blk.{layer}.hc_{half}_{suffix}.weight"))?,
                    );
                }
                Hc::load(&mut views, "hc", 4, 4096, 1e-6, 20, 1e-5)
            };
            layers.push(Layer {
                attn: if config.is_kda(layer) {
                    Attention::Kda(Kda::load_gguf(&w, layer)?)
                } else {
                    Attention::Dsa(GgufDsa::load(&w, layer)?)
                },
                mlp: Mlp::load(&w, layer)?,
                hc_attn: hc("attn")?,
                hc_ffn: hc("ffn")?,
                in_norm: plain(&w, &format!("blk.{layer}.attn_norm.weight"))?,
                post_norm: plain(&w, &format!("blk.{layer}.ffn_norm.weight"))?,
            });
        }
        let p = |name: &str| plain(&w, &format!("blk.45.{name}.weight"));
        let mtp = Mtp {
            eh: Projection::load(&w, "blk.45.nextn.eh_proj.weight")?,
            enorm: p("nextn.enorm")?,
            hnorm: p("nextn.hnorm")?,
            anorm: p("attn_norm")?,
            fnorm: p("ffn_norm")?,
            norm: p("nextn.shared_head_norm")?,
            attn: GgufDsa::load(&w, 45)?,
            mlp: Mlp::load(&w, 45)?,
        };
        let output = Projection::load(&w, "output.weight")?;
        let norm = plain(&w, "output_norm.weight")?;
        let stat = std::fs::metadata(path)?;
        let mut law = 0xcbf29ce484222325u64;
        for byte in b"glm53-gguf-serial-v1".iter().copied().chain(
            [
                stat.dev(),
                stat.ino(),
                stat.len(),
                stat.mtime() as u64,
                stat.mtime_nsec() as u64,
            ]
            .into_iter()
            .flat_map(u64::to_le_bytes),
        ) {
            law = (law ^ u64::from(byte)).wrapping_mul(0x100000001b3);
        }
        Ok(Self {
            _weights: w,
            config,
            layers,
            mtp,
            embedding: package.tensor("token_embd.weight")?.clone(),
            output,
            norm,
            capacity: 32768,
            law,
            capture: None,
            step: std::cell::Cell::new(0),
        })
    }
    fn embed(&self, token: u32) -> Result<Array> {
        ensure!(token < 154880, "GLM token outside vocabulary");
        let mut packed = vec![0u8; self.embedding.info.row_bytes()? as usize];
        self.embedding.read_rows(&[token as u64], &mut packed)?;
        let mut row = vec![0f32; 4096];
        quant::decode(self.embedding.info.kind, &packed, &mut row)?;
        Ok(Array::from_slice(&row, &[1, 1, 4096]))
    }
    fn one(&self, token: u32, caches: &mut [LayerCache]) -> Result<(Array, Array)> {
        ensure!(caches.len() == 45, "GLM cache layer count mismatch");
        let position = match &caches[3] {
            LayerCache::Full(f) => f.offset,
            _ => anyhow::bail!("GLM DSA cache missing"),
        };
        ensure!(position < self.capacity, "GLM context capacity exceeded");
        let _trace = lisa_mlx::trace::span("glm53.gguf.forward");
        let _scratch = lisa_mlx::runtime::enter_ordered_gpu_reuse_scope();
        let embedded = self.embed(token)?;
        let mut state = Array::concatenate(&[&embedded, &embedded, &embedded, &embedded], 2)?
            .reshape(&[1, 1, 4, 4096])?;
        for (li, (layer, cache)) in self.layers.iter().zip(caches.iter_mut()).enumerate() {
            let _trace = lisa_mlx::trace::span_detail("glm53.gguf.layer", li as u64);
            let (_, input, post, comb) = layer
                .hc_attn
                .split_normed(&state, &layer.in_norm, 1e-5)?
                .context("GLM HC attention declined")?;
            let attended = match (&layer.attn, cache) {
                (Attention::Kda(k), LayerCache::Linear(g)) => {
                    k.forward_at_position(&input, &mut g.conv, &mut g.ssm, None, li, position)?
                }
                (Attention::Dsa(d), LayerCache::Full(f)) => d.forward(&input, f, li)?,
                _ => anyhow::bail!("GLM cache kind mismatch at layer {li}"),
            };
            state = layer.hc_attn.expand(&attended, &state, &post, &comb)?;
            let (_, input, post, comb) = layer
                .hc_ffn
                .split_normed(&state, &layer.post_norm, 1e-5)?
                .context("GLM HC FFN declined")?;
            let output = layer.mlp.forward(&input)?;
            state = layer.hc_ffn.expand(&output, &state, &post, &comb)?;
            if let Some(c) = self.capture.as_ref().filter(|c| c.wants(li)) {
                c.store_ephemeral(&format!("L{li}.state_p{position}"), &state);
                c.store_ephemeral(&format!("L{li}.attn_p{position}"), &attended);
                c.store_ephemeral(&format!("L{li}.ffn_p{position}"), &output);
            }
        }
        let multi = state.reshape(&[1, 1, 16384])?;
        let hidden = rms(&mean(&multi)?, &self.norm)?;
        hidden.eval_enqueue()?;
        self.step.set(position);
        Ok((hidden, multi))
    }
    fn feed(&self, tokens: &[u32], caches: &mut [LayerCache], all: bool) -> Result<(Array, Array)> {
        ensure!(!tokens.is_empty(), "GLM requires nonempty input");
        let position = match caches.get(3) {
            Some(LayerCache::Full(f)) => f.offset,
            _ => anyhow::bail!("GLM cache missing"),
        };
        ensure!(
            position
                .checked_add(tokens.len())
                .is_some_and(|end| end <= self.capacity),
            "GLM context capacity exceeded"
        );
        let mut hidden = Vec::new();
        let mut multi = Vec::new();
        for (i, &token) in tokens.iter().enumerate() {
            let (h, m) = self.one(token, caches)?;
            if all || i + 1 == tokens.len() {
                hidden.push(h);
                multi.push(m);
            }
        }
        Ok((
            Array::concatenate(&hidden.iter().collect::<Vec<_>>(), 1)?,
            Array::concatenate(&multi.iter().collect::<Vec<_>>(), 1)?,
        ))
    }
}
impl LanguageModel for Tower {
    fn cache_numerical_law(&self) -> Option<u64> {
        Some(self.law)
    }
    fn max_position_embeddings(&self) -> usize {
        self.capacity
    }
    fn set_context_capacity(&mut self, c: usize) -> Result<()> {
        ensure!(
            c > 0 && c <= self.config.max_position_embeddings,
            "invalid GLM context capacity"
        );
        self.capacity = c;
        Ok(())
    }
    fn eos_token_id(&self) -> i64 {
        self.config.eos_token_id
    }
    fn eos_token_ids(&self) -> Vec<i64> {
        self.config.eos_token_ids.clone()
    }
    fn new_caches(&self) -> Vec<LayerCache> {
        (0..45)
            .map(|i| {
                if self.config.is_kda(i) {
                    LayerCache::Linear(GdnCache::new())
                } else {
                    LayerCache::Full(FullAttentionCache::new(1, 512))
                }
            })
            .collect()
    }
    fn forward(
        &mut self,
        ids: &Array,
        caches: Option<&mut Vec<LayerCache>>,
    ) -> Result<(Array, Array)> {
        ensure!(
            ids.rank() == 2 && ids.dim(0) == 1,
            "GLM GGUF expects one stream"
        );
        let tokens = ids.as_dtype(Dtype::Uint32)?.to_vec1::<u32>()?;
        let mut fresh;
        let caches = match caches {
            Some(c) => c,
            None => {
                fresh = self.new_caches();
                &mut fresh
            }
        };
        self.feed(&tokens, caches, true)
    }
    fn forward_capture(
        &mut self,
        ids: &Array,
        caches: Option<&mut Vec<LayerCache>>,
        _capture: bool,
    ) -> Result<(Array, Array)> {
        self.forward(ids, caches)
    }
    fn prefill(&mut self, tokens: &[u32], caches: &mut Vec<LayerCache>) -> Result<Array> {
        Ok(self.feed(tokens, caches, false)?.0.reshape(&[1, 4096])?)
    }
    fn prefill_multi(
        &mut self,
        tokens: &[u32],
        caches: &mut Vec<LayerCache>,
    ) -> Result<(Array, Array)> {
        let (h, m) = self.feed(tokens, caches, false)?;
        Ok((h.reshape(&[1, 4096])?, m.reshape(&[1, 16384])?))
    }
    fn head(&self, x: &Array) -> lisa_mlx::error::Result<Array> {
        let rows = (x.size() / 4096) as i32;
        let result = self
            .output
            .forward(&x.reshape(&[1, rows, 4096])?)
            .map_err(|e| lisa_mlx::error::Error::msg(e.to_string()))?;
        if let Some(c) = &self.capture {
            c.store(&format!("logits_p{}", self.step.get()), &result);
        }
        let mut shape = x.shape().to_vec();
        *shape.last_mut().expect("head rank") = 154880;
        result.reshape(&shape)
    }
    fn warmup(&mut self, seed: &[u32]) -> Result<()> {
        let t = *seed.first().context("empty GLM warmup seed")?;
        let mut c = self.new_caches();
        let h = self.prefill(&[t], &mut c)?;
        self.head(&h)?.eval()?;
        Ok(())
    }
    fn batch_decode_ok(&self) -> bool {
        false
    }
    fn kv_bytes_per_token(&self) -> usize {
        11 * (512 * 2 + 128 * 4 * 2 + 128 * 4 / 4)
    }
    fn stream_state_bytes(&self) -> usize {
        34 * (64 * 128 * 128 * 4 + 3 * 8192 * 3 * 4)
    }
    fn has_target_mtp(&self) -> bool {
        true
    }
    fn target_mtp_step(
        &mut self,
        tape: &mut IndexerTape,
        target: &Array,
        next: u32,
        position: usize,
        min_position: usize,
    ) -> Result<Array> {
        let _trace = lisa_mlx::trace::span("glm53.gguf.mtp");
        let embed = rms(&self.embed(next)?, &self.mtp.enorm)?;
        let hidden = rms(&mean(target)?, &self.mtp.hnorm)?;
        let x = self
            .mtp
            .eh
            .forward(&Array::concatenate(&[&embed, &hidden], 2)?)?;
        let a = self
            .mtp
            .attn
            .mtp(&rms(&x, &self.mtp.anorm)?, tape, position, min_position)?;
        let s = Stream::thread_local_or_default();
        let (residual, input) = gpu::glm53_mtp_add_rms_norm(&x, &a, &self.mtp.fnorm, 1e-5, &s)
            .context("GLM MTP add/norm declined")?;
        let (routed, shared) = self.mtp.mlp.parts(&input)?;
        let summed = gpu::glm53_mtp_add3(
            &residual,
            &routed.context("missing GLM MTP experts")?,
            &shared,
            &s,
        )
        .context("GLM MTP sum declined")?;
        self.output.forward(&rms(&summed, &self.mtp.norm)?)
    }
    fn enable_logit_capture(&mut self) -> Result<()> {
        self.enable_stage_capture(Vec::new(), Vec::new())
    }
    fn enable_stage_capture(&mut self, layers: Vec<usize>, keys: Vec<String>) -> Result<()> {
        ensure!(layers.iter().all(|&i| i < 45), "invalid GLM capture layer");
        self.capture = Some(Capture {
            layers,
            key_filters: keys,
            preserve_dispatch: true,
            ..Default::default()
        });
        Ok(())
    }
    fn write_logit_capture(&self, path: &str) -> Result<()> {
        self.capture.as_ref().context("no GLM capture")?.flush(path)
    }
}
