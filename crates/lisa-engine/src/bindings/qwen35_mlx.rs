//! The `qwen3_5` dense hybrid tower: embeddings, 64 pre-norm layers (GDN or
//! full attention), a dense SwiGLU MLP per layer, the final norm and `lm_head`.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use crate::{bindings::BindingDevice, execution::DeviceContext};
use lisa_backend::{HostTensor, ManagedDevice, ScalarType as Dtype, TensorAccess};

use crate::bindings::gated_delta::GatedDeltaNet;
use crate::bindings::normalization::{RmsNorm, Rotary};
use crate::bindings::qwen35_attention::Qwen35Attention;
use crate::cache::LayerCache;
use crate::weights::affine::{QuantizedEmbedding, QuantizedLinear};
use crate::weights::{
    mlx::{Checkpoint, Shard},
    source::{Weights, get_tensor},
};
use lisa_models::checkpoint::sanitize_name;
use lisa_models::qwen3_5::config::Qwen35Config;

use crate::bindings::dense::DenseMlp;

use crate::bindings::qwen35_decoder::DecoderLayer;

use lisa_models::qwen3_5::cache_law;

pub struct MlxExecution<D: BindingDevice + ManagedDevice> {
    context: DeviceContext<D>,
    cache_law: u64,
    vision: Option<crate::vision::Vision<D>>,
    embed_tokens: QuantizedEmbedding<D::Tensor>,
    layers: Vec<DecoderLayer<D>>,
    norm: RmsNorm<D::Tensor>,
    lm_head: QuantizedLinear<D::Tensor>,
    /// Draft-only 2-bit requantized copy (`draft_head`); Some only
    /// when the checkpoint carries an MTP head. Never used by verify.
    draft_head: Option<QuantizedLinear<D::Tensor>>,
    dspine_retrieval: Option<crate::bindings::dspine_retrieval::Retrieval<D>>,
    rope: Rotary,
    /// The native MTP head (`mtp/weights.safetensors`), when the checkpoint
    /// ships one. `has_drafter()` is true iff this is `Some`.
    mtp: Option<crate::bindings::qwen35_mtp::MtpHead<D>>,
    /// Optional external DSpine block drafter. When selected, the embedded
    /// target MTP head is left unloaded so session state has one drafter law.
    dspine: Option<crate::bindings::dspine::DSpineDraft<D>>,
    /// The last `draft_step`'s log p_head of its own proposal (unevaluated
    /// `[1]`), for the chunk-A sync. Consumed via `take_draft_confidence`.
    last_conf: Option<D::Tensor>,
    /// Whether draft_step builds the confidence at all (the tiny
    /// softmax+log enqueue is armed only for two-chunk runs).
    conf_capture: bool,
    pub config: Qwen35Config,
}

impl<D: BindingDevice + ManagedDevice> MlxExecution<D> {
    pub fn load(
        context: DeviceContext<D>,
        dir: &Path,
        config: Qwen35Config,
    ) -> anyhow::Result<Self> {
        Self::load_with_dspine(context.clone(), dir, config, None)
    }

    pub fn load_with_dspine(
        context: DeviceContext<D>,
        dir: &Path,
        config: Qwen35Config,
        draft: Option<&Path>,
    ) -> anyhow::Result<Self> {
        let draft = crate::bindings::dspine::resolve_selection(draft)?;
        let checkpoint = Checkpoint::open(dir)?;
        let map = checkpoint.weight_map.clone();
        let mut shard_order: Vec<String> = Vec::new();
        for shard in map.values() {
            if !shard_order.contains(shard) {
                shard_order.push(shard.clone());
            }
        }
        let mut weights: Weights<D::Tensor> = HashMap::new();
        let t_shards = std::time::Instant::now();
        for shard_name in &shard_order {
            let shard = Shard::open(context.device().clone(), &dir.join(shard_name))?;
            for name in shard.tensor_names() {
                if let Some(k) = sanitize_name(name) {
                    if !(draft.is_some() && k.starts_with("mtp.")) && !weights.contains_key(&k) {
                        weights.insert(k, get_tensor(&shard, name)?);
                    }
                }
            }
        }
        crate::ttft_mark(&format!(
            "model.load: {} shards read (mmap+host dequant) {:.2}s",
            shard_order.len(),
            t_shards.elapsed().as_secs_f64()
        ));
        // MTP head: a sibling `mtp/weights.safetensors`. The sidecar's tensor
        // names already carry the `mtp.` prefix (e.g. `mtp.fc.weight`); keys
        // are stored as-is so the head reads them back unchanged.
        let mtp_file = dir.join("mtp").join("weights.safetensors");
        if draft.is_none() && mtp_file.is_file() {
            let shard = Shard::open(context.device().clone(), &mtp_file)?;
            for name in shard.tensor_names() {
                let key = if name.starts_with("mtp.") {
                    sanitize_name(name).unwrap_or_else(|| name.clone())
                } else {
                    format!("mtp.{name}")
                };
                if !weights.contains_key(&key) {
                    weights.insert(key, get_tensor(&shard, name)?);
                }
            }
        }
        let mut tower = Self::from_weights_with_dspine(
            context.clone(),
            weights,
            config,
            Some(dir),
            draft.as_deref(),
        )?;
        tower.vision =
            crate::vision::Vision::from_dir(context.clone(), dir, tower.rope.dimensions)?;
        Ok(tower)
    }

    pub fn from_weights(
        context: DeviceContext<D>,
        w: Weights<D::Tensor>,
        config: Qwen35Config,
    ) -> anyhow::Result<Self> {
        Self::from_weights_cached(context.clone(), w, config, None)
    }

    pub fn from_weights_cached(
        context: DeviceContext<D>,
        w: Weights<D::Tensor>,
        config: Qwen35Config,
        cache_dir: Option<&Path>,
    ) -> anyhow::Result<Self> {
        let draft = crate::bindings::dspine::resolve_selection(None)?;
        Self::from_weights_with_dspine(context.clone(), w, config, cache_dir, draft.as_deref())
    }

    fn from_weights_with_dspine(
        context: DeviceContext<D>,
        mut w: Weights<D::Tensor>,
        config: Qwen35Config,
        cache_dir: Option<&Path>,
        dspine_dir: Option<&Path>,
    ) -> anyhow::Result<Self> {
        if dspine_dir.is_some() {
            anyhow::ensure!(
                crate::bindings::dspine::supports_target(&config),
                "DSpine requires the Qwen 27B target geometry"
            );
            w.retain(|name, _| !name.starts_with("mtp."));
        }
        let t_fw = std::time::Instant::now();
        let eps = config.rms_norm_eps;
        let gs = config.quant_group_size;
        let bits = config.quant_bits;

        crate::memory::report(context.device(), "t0-mmap-only");
        let mut embed_tokens = QuantizedEmbedding::load(&mut w, "model.embed_tokens")?;
        embed_tokens.set_quant(gs, bits);
        let mut lm_head = if config.tie_word_embeddings {
            QuantizedLinear {
                weight: embed_tokens.weight.clone(),
                scales: embed_tokens.scales.clone(),
                biases: embed_tokens.biases.clone(),
                group_size: embed_tokens.group_size,
                bits: embed_tokens.bits,
            }
        } else {
            QuantizedLinear::load(&mut w, "lm_head", "")?
        };
        lm_head.set_quant(gs, bits);
        crate::memory::report(context.device(), "t1-heads");

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            let p = format!("model.layers.{i}");
            let is_full = config.is_full(i);
            let attn = if is_full {
                Some(Qwen35Attention::load(
                    context.clone(),
                    &mut w,
                    &format!("{p}.self_attn"),
                    eps,
                    config.num_attention_heads,
                    config.num_key_value_heads,
                    config.head_dim,
                    config.attn_output_gate,
                    gs,
                    bits,
                )?)
            } else {
                None
            };
            let mut gdn = if is_full {
                None
            } else {
                Some(GatedDeltaNet::load(
                    context.clone(),
                    &mut w,
                    &format!("{p}.linear_attn"),
                    eps,
                    config.linear_num_key_heads,
                    config.linear_num_value_heads,
                    config.linear_key_head_dim,
                    config.linear_value_head_dim,
                    config.linear_conv_kernel_dim,
                    gs,
                    bits,
                )?)
            };
            if let Some(g) = gdn.as_mut() {
                g.output_gate_silu = true;
            }
            layers.push(DecoderLayer {
                context: context.clone(),
                plan: crate::core::decoder::PreparedDecoder::new(lisa_models::qwen3_5::decoder(
                    is_full,
                ))?,
                attn,
                gdn,
                input_norm: RmsNorm::load_on(
                    &context,
                    &mut w,
                    &format!("{p}.input_layernorm"),
                    eps,
                    None,
                )?,
                post_norm: RmsNorm::load_on(
                    &context,
                    &mut w,
                    &format!("{p}.post_attention_layernorm"),
                    eps,
                    None,
                )?,
                mlp: DenseMlp::load(context.clone(), &mut w, &format!("{p}.mlp"), gs, bits)?,
            });
        }
        let norm = RmsNorm::load_on(&context, &mut w, "model.norm", eps, None)?;
        crate::memory::report(context.device(), "t2-64-layers");
        let mtp = if dspine_dir.is_some() {
            None
        } else {
            crate::bindings::qwen35_mtp::MtpHead::load(context.clone(), &mut w, &config)?
        };
        crate::memory::report(context.device(), "t3-mtp-head");
        let t_draft = std::time::Instant::now();
        // The default draft head shortlists with a cached 2-bit projection,
        // then rescores its candidates through the original trunk head.
        // LISA_DRAFT_HEAD_BITS=0 disables it; any positive value enables the
        // supported 2-bit head (see build_draft_lm_head_cached).
        let draft_head = if mtp.is_some() && Self::draft_head_bits_from_env() > 0 {
            Self::build_draft_lm_head_cached(context.clone(), &lm_head, cache_dir)
        } else {
            None
        };
        let dspine = dspine_dir
            .map(|dir| {
                let mut draft = crate::bindings::dspine::DSpineDraft::load(
                    context.clone(),
                    dir,
                    config.vocab_size,
                    config.hidden_size,
                    0,
                )?;
                if let Some(target) = cache_dir {
                    let (block, verify, ssm) =
                        crate::bindings::dspine_policy::load(&context, target, dir)?;
                    draft.block_size = block;
                    draft.verify_width = verify;
                    if std::env::var_os("LISA_DSPINE_SSM").is_none() {
                        draft.replay = ssm == crate::bindings::dspine_policy::SsmMode::Replay;
                    }
                }
                // Explicit calibration override; the model's trained width is
                // still 16. A saved SPS profile is used for normal UI loads.
                if let Some(value) = std::env::var_os("LISA_DSPINE_BLOCK_SIZE") {
                    let block: usize = value
                        .to_str()
                        .ok_or_else(|| anyhow::anyhow!("DSpine block size is not UTF-8"))?
                        .parse()?;
                    anyhow::ensure!((3..=16).contains(&block), "DSpine block size must be 3..16");
                    draft.block_size = block;
                    draft.verify_width = block;
                }
                Ok::<_, anyhow::Error>(draft)
            })
            .transpose()?;
        crate::memory::report(context.device(), "t4-draft-head");
        // The map's explicit teardown. The loader relies on the HashMap being
        // dropped at the end of this function; do it here instead, so the
        // release is visible, and report what was never claimed — an entry still
        // present is a tensor the model does not use, and it must not stay
        // resident. Anything left is either a key this build never asks for or a
        // name the accessors miss, and its byte count is the difference between
        // what the checkpoint holds and what the model actually needs.
        let unclaimed = w.len();
        let unclaimed_bytes: usize = w
            .values()
            .map(|a| {
                a.size()
                    * context
                        .device()
                        .dtype(a)
                        .expect("known imported dtype")
                        .size_of()
            })
            .sum();
        w.clear();
        eprintln!(
            "[load] map teardown: {unclaimed} unclaimed tensors, {:.1} MB freed",
            unclaimed_bytes as f64 / 1e6
        );
        crate::memory::report(context.device(), "t5-map-cleared");
        let dspine_retrieval = if dspine.is_some() && crate::bindings::dspine_retrieval::enabled()?
        {
            crate::bindings::dspine_retrieval::Retrieval::build(context.clone(), &lm_head)?
        } else {
            None
        };
        crate::ttft_mark(&format!(
            "model.load: tower build {:.2}s (draft head {:.2}s inside)",
            t_fw.elapsed().as_secs_f64(),
            t_draft.elapsed().as_secs_f64()
        ));
        let cache_law = cache_law(&config, &context.device().identity().arithmetic);
        Ok(Self {
            context,
            cache_law,
            vision: None,
            embed_tokens,
            layers,
            norm,
            lm_head,
            draft_head,
            dspine_retrieval,
            rope: Rotary::new(config.rotary_dimensions(), config.rope_theta),
            mtp,
            dspine,
            last_conf: None,
            conf_capture: false,
            config,
        })
    }

    /// The draft-only low-bit lm_head: a requantized coarse copy of the
    /// trunk head, used ONLY by `draft_step`. Verification always reads
    /// the trunk head, so the output distribution is untouched — drafts just
    /// read fewer bytes per full-vocab projection, the dominant draft cost.
    /// Width is 2-bit/gs64, lisa's default coarse width (a 3-bit width would
    /// need MLX's permuted 3-bit packing, which our kernels decode but no
    /// host encoder here reproduces yet). Built once at load when the
    /// checkpoint carries an MTP head; `None` keeps the trunk head.
    /// [`Self::build_draft_lm_head`] with an optional on-disk cache. The
    /// requant costs ~5.2 s per process (GPU dequant of the [V, K] trunk
    /// head + host requant) and is DETERMINISTIC, so the packed result is
    /// persisted next to the checkpoint and reloaded verbatim — bit-identical
    /// to a fresh rebuild (same tensors, same layout). A missing/corrupt/
    /// mismatched cache falls back to the rebuild and rewrites the file.
    fn build_draft_lm_head_cached(
        context: DeviceContext<D>,
        lm_head: &QuantizedLinear<D::Tensor>,
        cache_dir: Option<&Path>,
    ) -> Option<QuantizedLinear<D::Tensor>> {
        const DRAFT_BITS: i32 = 2;
        const DRAFT_GROUP: i32 = 64;
        if DRAFT_BITS >= lm_head.bits {
            return None; // no byte saving over the trunk head
        }
        let t0 = std::time::Instant::now();
        if let Some(dir) = cache_dir {
            if let Some(dh) = Self::load_draft_lm_head_cache(
                context.clone(),
                dir,
                DRAFT_GROUP,
                DRAFT_BITS,
                lm_head,
            ) {
                eprintln!(
                    "[mtp] draft-only lm_head loaded from cache in {:.2}s",
                    t0.elapsed().as_secs_f32()
                );
                return Some(dh);
            }
        }
        // Dequantize on the GPU ([V, K] in the scales' dtype), requantize on
        // the host (the lisa-metal `quantize` is a host kernel), materialize.
        let (wq, scales, biases) = crate::bindings::affine::requantize(
            &context,
            &lm_head.weight,
            &lm_head.scales,
            &lm_head.biases,
            (lm_head.group_size, lm_head.bits),
            (DRAFT_GROUP, DRAFT_BITS),
        )
        .ok()?;
        if let Some(dir) = cache_dir {
            let _ = Self::save_draft_lm_head_cache(
                context.clone(),
                dir,
                DRAFT_GROUP,
                DRAFT_BITS,
                &wq,
                &scales,
                &biases,
            );
        }
        eprintln!(
            "[mtp] draft-only lm_head requantized to {DRAFT_BITS}-bit/gs{DRAFT_GROUP} in {:.1}s",
            t0.elapsed().as_secs_f32()
        );
        Some(QuantizedLinear {
            weight: wq,
            scales,
            biases,
            group_size: DRAFT_GROUP,
            bits: DRAFT_BITS,
        })
    }

    /// Draft-head width selector. **Default 2 = ON, like the reference MTP
    /// drafter**, whose per-tick drafts read a coarse 2-bit/gs64 shortlist of
    /// the trunk head and never the full trunk head (mtp.zig draftSelect,
    /// auto-built). The measured trade, both directions:
    ///
    ///   head ON  : drafts shortlist through the 2-bit head, decode **76.9 tok/s**
    ///              median, acceptance 5.86 accepted/round — +386 MB resident,
    ///              measured **15 038 MB total**: STILL under the alignment line
    ///   head OFF : drafts read the trunk's full 4-bit lm_head per draft tick,
    ///              decode **39.1 tok/s** median, acceptance 2.01-4.38/round
    ///
    /// The default follows the reference: RAM is still under the alignment line
    /// with the head on, and the chain cells measure 21-23 % cheaper
    /// (d2 5.38->4.26 .. d6 14.29->11.20 ms, round-cost kv16384, counter-
    /// balanced). `LISA_DRAFT_HEAD_BITS=0` disables for a footprint-first run.
    fn draft_head_bits_from_env() -> i32 {
        std::env::var("LISA_DRAFT_HEAD_BITS")
            .ok()
            .and_then(|v| v.trim().parse::<i32>().ok())
            .unwrap_or(2)
    }

    /// Cache-file layout: magic "LSDH1" | gs:i32 | bits:i32 | V:usize | K:usize
    /// | packed-weight u32 bytes | scales bf16 bytes | biases bf16 bytes. The
    /// V×K/dtype guard makes a stale file (different checkpoint geometry)
    /// detectable by size alone.
    fn draft_head_cache_path(dir: &Path) -> PathBuf {
        dir.join("mtp").join("draft_head_2bit_gs64.lsdh")
    }

    fn load_draft_lm_head_cache(
        context: DeviceContext<D>,
        dir: &Path,
        gs: i32,
        bits: i32,
        reference: &QuantizedLinear<D::Tensor>,
    ) -> Option<QuantizedLinear<D::Tensor>> {
        let data = std::fs::read(Self::draft_head_cache_path(dir)).ok()?;
        if data.len() < 8 + 4 + 4 + 8 + 8 {
            return None;
        }
        if &data[0..5] != b"LSDH1" {
            return None;
        }
        let rd_i32 = |off: usize| -> Option<i32> {
            Some(i32::from_le_bytes(data.get(off..off + 4)?.try_into().ok()?))
        };
        let rd_usize = |off: usize| -> Option<usize> {
            Some(usize::from_le_bytes(
                data.get(off..off + 8)?.try_into().ok()?,
            ))
        };
        if rd_i32(5)? != gs || rd_i32(9)? != bits {
            return None;
        }
        let v = rd_usize(13)?;
        let k = rd_usize(21)?;
        // `k` is the LOGICAL (unpacked) head width; the packed weight rows
        // are k*bits/32 u32 words. The trunk head supplies the expected
        // geometry so a different checkpoint is detected by value.
        // The trunk head's weight is ALSO packed (reference.bits per value),
        // so its logical width is dim(1)*32/bits — compare logical to logical.
        let ref_k = reference.weight.dim(1) as usize * 32 / reference.bits as usize;
        if v != reference.weight.dim(0) as usize || k != ref_k {
            return None;
        }
        let mut off = 29usize;
        let mut take = |n: usize| -> Option<&[u8]> {
            let s = data.get(off..off.checked_add(n)?)?;
            off += n;
            Some(s)
        };
        let wq_bytes = take(v * k * bits as usize / 8)?; // packed: bits/value
        let scales_bytes = take(v * (k / gs as usize) * 2)?; // bf16
        let biases_bytes = take(v * (k / gs as usize) * 2)?;
        let wq = context
            .device()
            .pull(
                HostTensor {
                    bytes: wq_bytes,
                    shape: &[v as i32, (k * bits as usize / 32) as i32],
                    dtype: Dtype::U32,
                }
                .into(),
                Default::default(),
            )
            .ok()?;
        let scales = context
            .device()
            .pull(
                HostTensor {
                    bytes: scales_bytes,
                    shape: &[v as i32, k as i32 / gs],
                    dtype: Dtype::BF16,
                }
                .into(),
                Default::default(),
            )
            .ok()?;
        let biases = context
            .device()
            .pull(
                HostTensor {
                    bytes: biases_bytes,
                    shape: &[v as i32, k as i32 / gs],
                    dtype: Dtype::BF16,
                }
                .into(),
                Default::default(),
            )
            .ok()?;
        Some(QuantizedLinear {
            weight: wq,
            scales,
            biases,
            group_size: gs,
            bits,
        })
    }

    fn save_draft_lm_head_cache(
        context: DeviceContext<D>,
        dir: &Path,
        gs: i32,
        bits: i32,
        wq: &D::Tensor,
        scales: &D::Tensor,
        biases: &D::Tensor,
    ) -> anyhow::Result<()> {
        // `wq` rows are PACKED (bits/value): words per row = k*bits/32, so the
        // logical width k = words*32/bits.
        let v = wq.dim(0) as usize;
        let k = wq.dim(1) as usize * 32 / bits as usize;
        let mut out = Vec::with_capacity(29 + wq.size() * 4 + (scales.size() + biases.size()) * 2);
        out.extend_from_slice(b"LSDH1");
        out.extend_from_slice(&gs.to_le_bytes());
        out.extend_from_slice(&bits.to_le_bytes());
        out.extend_from_slice(&(v as u64).to_le_bytes());
        out.extend_from_slice(&(k as u64).to_le_bytes());
        out.extend_from_slice(&context.device().read_bytes(wq)?);
        out.extend_from_slice(&context.device().read_bytes(scales)?);
        out.extend_from_slice(&context.device().read_bytes(biases)?);
        let path = Self::draft_head_cache_path(dir);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let tmp = path.with_extension("lsdh.tmp");
        std::fs::write(&tmp, &out)?;
        std::fs::rename(&tmp, &path)?;
        Ok(())
    }
}

struct MlxDSpineTarget<'a, D: BindingDevice + ManagedDevice> {
    context: &'a DeviceContext<D>,
    embedding: &'a QuantizedEmbedding<D::Tensor>,
    head: &'a QuantizedLinear<D::Tensor>,
    retrieval: Option<&'a crate::bindings::dspine_retrieval::Retrieval<D>>,
}

impl<D: BindingDevice + ManagedDevice> crate::bindings::dspine::DSpineTarget<D::Tensor>
    for MlxDSpineTarget<'_, D>
{
    fn embed_tokens(&self, ids: &D::Tensor) -> anyhow::Result<D::Tensor> {
        crate::bindings::affine::lookup(
            self.context,
            ids,
            &self.embedding.weight,
            &self.embedding.scales,
            &self.embedding.biases,
            self.embedding.group_size,
            self.embedding.bits,
        )
    }

    fn shortlist(&self, hidden: &D::Tensor) -> anyhow::Result<Option<D::Tensor>> {
        match self.retrieval {
            Some(retrieval) => retrieval.candidates(hidden, self.head),
            None => Ok(None),
        }
    }

    fn head_logits(&self, hidden: &D::Tensor) -> anyhow::Result<D::Tensor> {
        crate::bindings::affine::linear(self.context, hidden, self.head)
    }

    fn head_rows(&self, ids: &D::Tensor) -> anyhow::Result<D::Tensor> {
        crate::bindings::affine::lookup(
            self.context,
            ids,
            &self.head.weight,
            &self.head.scales,
            &self.head.biases,
            self.head.group_size,
            self.head.bits,
        )
    }
}

impl<D: BindingDevice + ManagedDevice> super::qwen35_execution::Execution<D> for MlxExecution<D>
where
    D::Error: From<anyhow::Error>,
{
    fn numerical_law(&self) -> Option<u64> {
        Some(self.cache_law)
    }
    fn layer(
        &mut self,
        i: usize,
        state: &mut super::qwen35_execution::Forward<D::Tensor>,
        cache: Option<&mut LayerCache<D>>,
        capture: bool,
    ) -> anyhow::Result<()> {
        let (cos, sin) = state.rope.as_ref().expect("affine Qwen rotary context");
        let (x, m) = self.layers[i].forward(
            &state.x,
            state.residual.take(),
            cache,
            cos,
            sin,
            state.offset,
            capture,
            self.dspine.as_ref().is_some_and(|draft| draft.replay),
        )?;
        state.x = x;
        state.residual = Some(m);
        Ok(())
    }
    fn draft_target_layers(&self) -> &[usize] {
        self.dspine
            .as_ref()
            .map_or(&[], |draft| draft.target_layers())
    }
    fn draft_target_layer_output(
        &self,
        state: &super::qwen35_execution::Forward<D::Tensor>,
    ) -> anyhow::Result<D::Tensor> {
        crate::bindings::qwen35_entry::capture_hidden(
            &self.context,
            &state.x,
            state.residual.as_ref().expect("Qwen layer residual"),
        )
    }
    fn has_block_drafter(&self) -> bool {
        self.dspine.is_some()
    }
    fn block_drafter_context_matches(&self, tokens: &[u32]) -> bool {
        self.dspine
            .as_ref()
            .is_some_and(|draft| draft.context_matches(tokens))
    }
    fn append_drafter_context(
        &mut self,
        target_hidden: &D::Tensor,
        tokens: &[u32],
    ) -> anyhow::Result<()> {
        self.dspine
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("this model has no DSpine drafter"))?
            .append_target_hidden(target_hidden, tokens)
    }
    fn draft_block(&mut self, anchor: u32) -> anyhow::Result<D::Tensor> {
        let target = MlxDSpineTarget {
            context: &self.context,
            embedding: &self.embed_tokens,
            head: &self.lm_head,
            retrieval: self.dspine_retrieval.as_ref(),
        };
        self.dspine
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("this model has no DSpine drafter"))?
            .draft_block(anchor, &target)
    }
    fn finish(
        &self,
        state: super::qwen35_execution::Forward<D::Tensor>,
    ) -> anyhow::Result<D::Tensor> {
        Ok(self
            .norm
            .forward_residual_add_on(
                &self.context,
                &state.x,
                &state.residual.expect("nonempty Qwen tower"),
            )?
            .1)
    }
    fn config(&self) -> &Qwen35Config {
        &self.config
    }
    fn block_drafter_state_replay(&self) -> bool {
        self.dspine.as_ref().is_some_and(|draft| draft.replay)
    }
    fn block_drafter_verify_depth(&self) -> usize {
        self.dspine
            .as_ref()
            .map_or(0, |draft| draft.verify_width - 1)
    }
    fn max_speculative_depth(&self) -> usize {
        self.dspine
            .as_ref()
            .map_or(usize::MAX, |draft| draft.max_draft_depth())
    }
    fn begin(
        &self,
        ids: &D::Tensor,
        caches: Option<&Vec<LayerCache<D>>>,
    ) -> anyhow::Result<super::qwen35_execution::Forward<D::Tensor>> {
        let s = ids.dim(1) as usize;
        let offset = caches
            .and_then(|c| {
                c.iter().find_map(|l| match l {
                    LayerCache::Full(f) => Some(f.offset),
                    _ => None,
                })
            })
            .unwrap_or(0);
        let next = caches.and_then(|c| {
            c.iter().find_map(|l| match l {
                LayerCache::Full(f) => f.next_pos.as_deref(),
                _ => None,
            })
        });
        let (x, cos_bf, sin_bf) = crate::bindings::qwen35_entry::begin(
            &self.context,
            ids,
            &self.embed_tokens,
            &self.rope,
            self.vision.as_ref(),
            offset,
            s,
            next,
        )?;
        Ok(super::qwen35_execution::Forward {
            x,
            residual: None,
            offset,
            rope: Some((cos_bf, sin_bf)),
        })
    }
    fn head(&self, mixed: &D::Tensor) -> std::result::Result<D::Tensor, D::Error> {
        crate::bindings::affine::linear(&self.context, mixed, &self.lm_head).map_err(D::Error::from)
    }
    fn supports_images(&self) -> bool {
        self.vision.is_some()
    }
    fn prepare_images(
        &mut self,
        tokens: &mut Vec<u32>,
        images: &[image::RgbImage],
    ) -> anyhow::Result<()> {
        let limit = self.config.max_position_embeddings;
        self.vision
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("checkpoint has no supported vision encoder"))?
            .prepare(tokens, images, limit)
    }
    fn clear_images(&mut self) {
        if let Some(v) = &mut self.vision {
            v.clear();
        }
    }
    fn has_drafter(&self) -> bool {
        self.dspine.is_some() || self.mtp.is_some()
    }
    fn mtp_cost_key(&self) -> String {
        format!(
            "qwen3_5-{}L-{}h",
            self.config.num_hidden_layers, self.config.hidden_size
        )
    }
    fn take_draft_confidence(&mut self) -> Option<D::Tensor> {
        self.last_conf.take()
    }
    fn set_conf_capture(&mut self, on: bool) {
        self.conf_capture = on;
    }
    fn drafter_reset(&mut self) {
        if let Some(draft) = self.dspine.as_mut() {
            draft.reset();
        }
        if let Some(h) = self.mtp.as_mut() {
            h.reset_caches();
        }
    }
    fn drafter_trim(&mut self, n: usize) {
        if let Some(draft) = self.dspine.as_mut() {
            draft.trim(n);
        }
        if let Some(h) = self.mtp.as_mut() {
            h.trim_caches(n);
        }
    }
    fn drafter_offset(&self) -> usize {
        self.dspine.as_ref().map_or_else(
            || self.mtp.as_ref().map(|h| h.cache_offset()).unwrap_or(0),
            |d| d.offset(),
        )
    }
    fn drafter_restore_offset(&mut self, n: usize) {
        if let Some(draft) = self.dspine.as_mut() {
            draft.restore_offset(n);
        }
        if let Some(h) = self.mtp.as_mut() {
            h.restore_offset(n);
        }
    }
    fn draft_step(
        &mut self,
        tokens: &D::Tensor,
        multi: &D::Tensor,
    ) -> anyhow::Result<(D::Tensor, D::Tensor)> {
        anyhow::ensure!(
            self.dspine.is_none(),
            "DSpine uses block drafting, not MTP draft_step"
        );
        let offset = self.mtp.as_ref().map(|h| h.cache_offset()).unwrap_or(0);
        let head = self
            .mtp
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("this model has no MTP head"))?;
        // `multi` for qwen3_5 is the trunk's final-norm hidden — the same
        // vector lm_head consumes, which is the head's `hidden` input.
        let (_, last) = head.forward(tokens, multi, &self.embed_tokens, offset)?;
        let selected = crate::bindings::qwen35_readout::select(
            &self.context,
            &last,
            &self.lm_head,
            self.draft_head.as_ref(),
            self.conf_capture,
        )?;
        if let Some(confidence) = selected.confidence {
            self.last_conf = Some(confidence);
        }
        crate::core::oracle::push_arrays(selected.candidates.as_ref(), &selected.scores)?;
        let draft_id = selected.token;
        Ok((draft_id, last))
    }
}

#[cfg(test)]
mod cache_law_tests {
    use super::*;
    #[test]
    fn qwen_mlx_cache_law_is_available_without_lazy_pipelines() -> anyhow::Result<()> {
        let mut config = Qwen35Config::from_str("{}")?;
        let law = cache_law(&config, b"test_s\x01");
        assert_eq!(law, cache_law(&config, b"test_s\x01"));
        assert_ne!(law, cache_law(&config, b"test_d\x01"));
        assert_ne!(law, cache_law(&config, b"test_s\x00"));
        config.quant_bits = 2;
        assert_ne!(law, cache_law(&config, b"test_s\x01"));
        Ok(())
    }
}
