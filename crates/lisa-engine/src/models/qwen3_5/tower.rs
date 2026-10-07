//! The `qwen3_5` dense hybrid tower: embeddings, 64 pre-norm layers (GDN or
//! full attention), a dense SwiGLU MLP per layer, the final norm and `lm_head`.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use lisa_mlx::{Array, Dtype};
use lisa_mlx::ops::indexing::IndexOp;

use crate::core::cache::{FullAttentionCache, GdnCache, LayerCache};
use crate::core::loader::{Checkpoint, Shard, Weights, sanitize_name};
use crate::core::norm::{RmsNorm, Rotary, bf16_silu, positions};
use crate::core::quant::{QuantizedEmbedding, QuantizedLinear, get_tensor};
use crate::models::gdn::GatedDeltaNet;
use crate::models::qwen3_5::attention::Qwen35Attention;
use crate::models::qwen3_5::config::Qwen35Config;

pub(crate) struct DenseMlp {
    pub(crate) gate_proj: QuantizedLinear,
    pub(crate) up_proj: QuantizedLinear,
    pub(crate) down_proj: QuantizedLinear,
    /// Row-concatenation of gate|up (exact on the GEMV paths — same argument
    /// as GDN's `in_proj_all`): ONE gate+up qmv per layer instead of two.
    /// Built only for decode-eligible shapes (group_size/bits shared).
    gate_up_proj: Option<QuantizedLinear>,
}

impl DenseMlp {
    pub(crate) fn load(w: &mut Weights, prefix: &str, gs: i32, bits: i32) -> anyhow::Result<Self> {
        let mut gate_proj = QuantizedLinear::load(w, prefix, "gate_proj")?;
        gate_proj.set_quant(gs, bits);
        let mut up_proj = QuantizedLinear::load(w, prefix, "up_proj")?;
        up_proj.set_quant(gs, bits);
        let mut down_proj = QuantizedLinear::load(w, prefix, "down_proj")?;
        down_proj.set_quant(gs, bits);
        // Fused gate|up: row-concatenation of the packed quantized weights is
        // exact on the GEMV paths (each output row is computed from the same K
        // values with the same group scales — the in_proj_all precedent, which
        // the goldens pin). One qmv per layer instead of two.
        let n_gate = gate_proj.dims_out() as i32;
        let n_up = up_proj.dims_out() as i32;
        let gate_up_proj = if gate_proj.group_size == up_proj.group_size
            && gate_proj.bits == up_proj.bits
        {
            let cat = |a: &Array, b: &Array| {
                lisa_mlx::ops::concatenate(&[a, b], 0)
                    .expect("gate|up weight row-concat")
            };
            Some(QuantizedLinear {
                weight: cat(&gate_proj.weight, &up_proj.weight),
                scales: cat(&gate_proj.scales, &up_proj.scales),
                biases: cat(&gate_proj.biases, &up_proj.biases),
                group_size: gate_proj.group_size,
                bits: gate_proj.bits,
            })
        } else {
            None
        };
        // ROW-JOIN (same trick as the GDN's in_proj_all): once the fused
        // gate|up exists, rebuild
        // gate_proj/up_proj as VIEWS into it and release their originals, so only
        // ONE copy of the gate|up weights stays resident. Before this the layer
        // held THREE — gate, up, and the `cat` copy — and the MLP is 10.16 GB of
        // the 16.05 GB checkpoint, so the duplicate was ~6.8 GB: the bulk of the
        // 24.5 GB MLX active measured at `serve-loaded` against 16.29 GB of
        // weights (measured with `footprint`: all of it IOAccelerator). Rows are
        // dim 0 and contiguous, so `Array::index` is a raw slice view (specs/04)
        // and no data moves; `IndexElem` is implemented for Range<i32>, not
        // Range<usize>. The numeric identity is unchanged: each view holds
        // exactly the rows the original held.
        if let Some(gu) = gate_up_proj.as_ref() {
            let rows = |of: &Array, lo: i32, hi: i32| -> Array { of.index((lo..hi, ..)) };
            gate_proj.weight = rows(&gu.weight, 0, n_gate);
            gate_proj.scales = rows(&gu.scales, 0, n_gate);
            gate_proj.biases = rows(&gu.biases, 0, n_gate);
            up_proj.weight = rows(&gu.weight, n_gate, n_gate + n_up);
            up_proj.scales = rows(&gu.scales, n_gate, n_gate + n_up);
            up_proj.biases = rows(&gu.biases, n_gate, n_gate + n_up);
        }
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
            gate_up_proj,
        })
    }

    fn forward(&self, x: &Array) -> lisa_mlx::error::Result<Array> {
        let stream = lisa_mlx::Stream::thread_local_or_default();
        // Merged gate|up arm (specs/16): ONE qmv over the row-concatenated
        // weights, then the packed SwiGLU reads both halves of its output.
        // Bit-exact vs the split arm (see gate_up_proj note) — the packed
        // swiglu2 consumes the same bf16 elements in the same rounding order.
        if let Some(gu_proj) = self.gate_up_proj.as_ref() {
            // Decode-only arm for now (specs/16): the M=1 qmv path. The M>1
            // prefill GEMM keeps the split weights (golden 310/310 regressed
            // with the merge on the prefill path — see specs/16 §merge).
            // B-generic since the track_swiglu2_packed row-decompose fix
            // (specs/13 re-audit): the kernel used to read gu[i]/gu[F+i]
            // single-row style, corrupting slot >= 1 at B > 1.
            if (x.rank() >= 2 && x.dim(1) == 1) {
                let gu = gu_proj.forward(x)?;
                if let Some(a) = lisa_mlx::models::qwen35::kernels::swiglu2_packed(&gu, &stream) {
                    return self.down_proj.forward(&a);
                }
            }
        }
        let g = self.gate_proj.forward(x)?;
        let u = self.up_proj.forward(x)?;
        // Fused SwiGLU (`track_swiglu2`, the exact_header silu helper already
        // golden-proven on qwen4): one launch replaces silu + multiply.
        // Same op chain, same bf16 rounding points.
        // NOTE: the wrapper flattens the leading dims (`[rows, F]`); restore
        // the input's rank so the residual add can't broadcast across the S
        // axis at B>1 (the B=2 crash in specs/07). Pure reshape — the values
        // and the B=1 rounding points are unchanged.
        let stream = lisa_mlx::Stream::thread_local_or_default();
        let a = match lisa_mlx::kernels::swiglu2(&g, &u, &stream) {
            Some(a) => a.reshape(&g.shape())?,
            None => bf16_silu(&g)?.multiply(&u)?,
        };
        self.down_proj.forward(&a)
    }
}

pub(crate) struct DecoderLayer {
    pub(crate) attn: Option<Qwen35Attention>,
    pub(crate) gdn: Option<GatedDeltaNet>,
    pub(crate) input_norm: RmsNorm,
    pub(crate) post_norm: RmsNorm,
    pub(crate) mlp: DenseMlp,
}

impl DecoderLayer {
    /// The MTP head's decoder layer: the same shape as a trunk layer but built
    /// from `mtp.layers.0.*` weights (always full attention, dense MLP).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new_mtp(
        w: &mut Weights,
        prefix: &str,
        eps: f32,
        heads: usize,
        kv_heads: usize,
        head_dim: usize,
        gated: bool,
        gs: i32,
        bits: i32,
    ) -> anyhow::Result<Self> {
        let mlp = DenseMlp::load(w, &format!("{prefix}.mlp"), gs, bits)?;
        Ok(Self {
            attn: Some(Qwen35Attention::load(
                w,
                &format!("{prefix}.self_attn"),
                eps,
                heads,
                kv_heads,
                head_dim,
                gated,
                gs,
                bits,
            )?),
            gdn: None,
            input_norm: RmsNorm::load(w, &format!("{prefix}.input_layernorm"), eps, None)?,
            post_norm: RmsNorm::load(w, &format!("{prefix}.post_attention_layernorm"), eps, None)?,
            mlp,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn forward(
        &mut self,
        x: &Array,
        m_prev: Option<Array>,
        cache: Option<&mut LayerCache>,
        cos_bf: &Array,
        sin_bf: &Array,
        offset: usize,
        capture: bool,
    ) -> anyhow::Result<(Array, Array)> {
        // Input norm: with the previous layer's MLP output still un-added, the
        // fused add+norm kernel produces (sum, normed) in one dispatch — the
        // same op chain and bf16 rounding points as `x.add(&m)` then
        // `norm.forward` (the fused add+norm pattern; golden-proven at the
        // post-norm site).
        let (sum, normed) = match m_prev {
            None => (x.clone(), self.input_norm.forward(x)?),
            Some(m) => self.input_norm.forward_residual_add(x, &m)?,
        };
        let r = if let Some(a) = self.attn.as_ref() {
            let f = match cache {
                Some(LayerCache::Full(f)) => Some(f),
                _ => None,
            };
            a.forward(&normed, cos_bf, sin_bf, f, offset)?
        } else if let Some(g) = self.gdn.as_ref() {
            let g_c = match cache {
                Some(LayerCache::Linear(gc)) => Some(gc),
                _ => None,
            };
            g.forward(&normed, g_c, capture)?
        } else {
            anyhow::bail!("layer has neither attention nor GDN")
        };
        let (h, h2) = self.post_norm.forward_residual_add(&sum, &r)?;
        let m = self.mlp.forward(&h2)?;
        // The MLP output is NOT added here: the caller folds `sum + m` into the
        // next norm via `forward_residual_add`, removing one badd + one rms
        // dispatch pair per layer. Same values, same rounding points.
        Ok((h, m))
    }
}

pub struct Qwen35Tower {
    embed_tokens: QuantizedEmbedding,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    lm_head: QuantizedLinear,
    /// Draft-only 3-bit requantized copy (`draft_head`); Some only
    /// when the checkpoint carries an MTP head. Never used by verify.
    draft_head: Option<QuantizedLinear>,
    rope: Rotary,
    /// The native MTP head (`mtp/weights.safetensors`), when the checkpoint
    /// ships one. `has_drafter()` is true iff this is `Some`.
    mtp: Option<crate::models::qwen3_5::mtp::MtpHead>,
    /// The last `draft_step`'s log p_head of its own proposal (unevaluated
    /// `[1]`), for the chunk-A sync. Consumed via `take_draft_confidence`.
    last_conf: Option<Array>,
    /// Whether draft_step builds the confidence at all (the tiny
    /// softmax+log enqueue is armed only for two-chunk runs).
    conf_capture: bool,
    pub config: Qwen35Config,
}

impl Qwen35Tower {
    pub fn load(dir: &Path, config: Qwen35Config) -> anyhow::Result<Self> {
        let checkpoint = Checkpoint::open(dir)?;
        let map = checkpoint.weight_map.clone();
        let mut shard_order: Vec<String> = Vec::new();
        for shard in map.values() {
            if !shard_order.contains(shard) {
                shard_order.push(shard.clone());
            }
        }
        let mut weights: Weights = HashMap::new();
        let t_shards = std::time::Instant::now();
        for shard_name in &shard_order {
            let shard = Shard::open(&dir.join(shard_name))?;
            for name in shard.tensor_names() {
                if let Some(k) = sanitize_name(name) {
                    if !weights.contains_key(&k) {
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
        if mtp_file.is_file() {
            let shard = Shard::open(&mtp_file)?;
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
        Self::from_weights_cached(weights, config, Some(dir))
    }

    pub fn from_weights(mut w: Weights, config: Qwen35Config) -> anyhow::Result<Self> {
        Self::from_weights_cached(w, config, None)
    }

    pub fn from_weights_cached(
        mut w: Weights,
        config: Qwen35Config,
        cache_dir: Option<&Path>,
    ) -> anyhow::Result<Self> {
        let t_fw = std::time::Instant::now();
        let eps = config.rms_norm_eps;
        let gs = config.quant_group_size;
        let bits = config.quant_bits;

        crate::core::mem::mlx_mem_line("t0-mmap-only");
        let mut embed_tokens = QuantizedEmbedding::load(&mut w, "model.embed_tokens")?;
        embed_tokens.set_quant(gs, bits);
        let mut lm_head = QuantizedLinear::load(&mut w, "lm_head", "")?;
        lm_head.set_quant(gs, bits);
        crate::core::mem::mlx_mem_line("t1-heads");

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            let p = format!("model.layers.{i}");
            let is_full = config.is_full(i);
            let attn = if is_full {
                Some(Qwen35Attention::load(
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
                g.l2_norm = true;
                g.output_gate_silu = true;
            }
            layers.push(DecoderLayer {
                attn,
                gdn,
                input_norm: RmsNorm::load(&mut w, &format!("{p}.input_layernorm"), eps, None)?,
                post_norm: RmsNorm::load(
                    &mut w,
                    &format!("{p}.post_attention_layernorm"),
                    eps,
                    None,
                )?,
                mlp: DenseMlp::load(&mut w, &format!("{p}.mlp"), gs, bits)?,
            });
        }
        let norm = RmsNorm::load(&mut w, "model.norm", eps, None)?;
        crate::core::mem::mlx_mem_line("t2-64-layers");
        let mtp = crate::models::qwen3_5::mtp::MtpHead::load(&mut w, &config)?;
        crate::core::mem::mlx_mem_line("t3-mtp-head");
        let t_draft = std::time::Instant::now();
        // Draft-only lm_head width: OFF by default, so drafts run through the
        // trunk's own lm_head. A narrower draft head shrinks a read the round
        // barely notices and charges it to ACCEPTANCE, which is the scarce
        // resource — the trunk head is exact either way. `LISA_DRAFT_HEAD_BITS`
        // selects a width (2, 3, 4, 6, 8) for the A/B; 0 or unset keeps it off.
        let draft_head = if mtp.is_some() && Self::draft_head_bits_from_env() > 0 {
            Self::build_draft_lm_head_cached(&lm_head, cache_dir)
        } else {
            None
        };
        crate::core::mem::mlx_mem_line("t4-draft-head");
        // The map's explicit teardown. The loader relies on the HashMap being
        // dropped at the end of this function; do it here instead, so the
        // release is visible, and report what was never claimed — an entry still
        // present is a tensor the model does not use, and it must not stay
        // resident. Anything left is either a key this build never asks for or a
        // name the accessors miss, and its byte count is the difference between
        // what the checkpoint holds and what the model actually needs.
        let unclaimed = w.len();
        let unclaimed_bytes: usize = w.values().map(|a| a.size() * a.dtype().size_of()).sum();
        w.clear();
        eprintln!(
            "[load] map teardown: {unclaimed} unclaimed tensors, {:.1} MB freed",
            unclaimed_bytes as f64 / 1e6
        );
        crate::core::mem::mlx_mem_line("t5-map-cleared");
        crate::ttft_mark(&format!(
            "model.load: tower build {:.2}s (draft head {:.2}s inside)",
            t_fw.elapsed().as_secs_f64(),
            t_draft.elapsed().as_secs_f64()
        ));
        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            draft_head,
            rope: Rotary::new(config.rotary_dimensions(), config.rope_theta),
            mtp,
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
        lm_head: &QuantizedLinear,
        cache_dir: Option<&Path>,
    ) -> Option<QuantizedLinear> {
        const DRAFT_BITS: i32 = 2;
        const DRAFT_GROUP: i32 = 64;
        if DRAFT_BITS >= lm_head.bits {
            return None; // no byte saving over the trunk head
        }
        let t0 = std::time::Instant::now();
        if let Some(dir) = cache_dir {
            if let Some(dh) =
                Self::load_draft_lm_head_cache(dir, DRAFT_GROUP, DRAFT_BITS, lm_head)
            {
                eprintln!(
                    "[mtp] draft-only lm_head loaded from cache in {:.2}s",
                    t0.elapsed().as_secs_f32()
                );
                return Some(dh);
            }
        }
        // Dequantize on the GPU ([V, K] in the scales' dtype), requantize on
        // the host (the lisa-mlx `quantize` is a host kernel), materialize.
        let full = lisa_mlx::ops::dequantize(
            &lm_head.weight,
            &lm_head.scales,
            &lm_head.biases,
            lm_head.group_size,
            lm_head.bits,
        )
        .ok()?;
        full.eval().ok()?;
        let (wq, scales, biases) = lisa_mlx::ops::quantize(&full, DRAFT_GROUP, DRAFT_BITS).ok()?;
        wq.eval().ok()?;
        scales.eval().ok()?;
        biases.eval().ok()?;
        if let Some(dir) = cache_dir {
            let _ = Self::save_draft_lm_head_cache(
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
        dir: &Path,
        gs: i32,
        bits: i32,
        reference: &QuantizedLinear,
    ) -> Option<QuantizedLinear> {
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
            Some(usize::from_le_bytes(data.get(off..off + 8)?.try_into().ok()?))
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
        let wq = unsafe {
            Array::from_raw_data(
                wq_bytes.as_ptr() as *const std::ffi::c_void,
                &[v as i32, (k * bits as usize / 32) as i32],
                Dtype::Uint32,
            )
        };
        let scales = unsafe {
            Array::from_raw_data(
                scales_bytes.as_ptr() as *const std::ffi::c_void,
                &[v as i32, k as i32 / gs],
                Dtype::Bfloat16,
            )
        };
        let biases = unsafe {
            Array::from_raw_data(
                biases_bytes.as_ptr() as *const std::ffi::c_void,
                &[v as i32, k as i32 / gs],
                Dtype::Bfloat16,
            )
        };
        Some(QuantizedLinear {
            weight: wq,
            scales,
            biases,
            group_size: gs,
            bits,
        })
    }

    fn save_draft_lm_head_cache(
        dir: &Path,
        gs: i32,
        bits: i32,
        wq: &lisa_mlx::Array,
        scales: &lisa_mlx::Array,
        biases: &lisa_mlx::Array,
    ) -> anyhow::Result<()> {
        // `wq` rows are PACKED (bits/value): words per row = k*bits/32, so the
        // logical width k = words*32/bits.
        let v = wq.dim(0) as usize;
        let k = wq.dim(1) as usize * 32 / bits as usize;
        let mut out =
            Vec::with_capacity(29 + wq.size() * 4 + (scales.size() + biases.size()) * 2);
        out.extend_from_slice(b"LSDH1");
        out.extend_from_slice(&gs.to_le_bytes());
        out.extend_from_slice(&bits.to_le_bytes());
        out.extend_from_slice(&(v as u64).to_le_bytes());
        out.extend_from_slice(&(k as u64).to_le_bytes());
        out.extend_from_slice(unsafe {
            std::slice::from_raw_parts(wq.as_slice::<u32>().as_ptr() as *const u8, wq.size() * 4)
        });
        out.extend_from_slice(unsafe {
            std::slice::from_raw_parts(
                scales.as_slice::<u16>().as_ptr() as *const u8,
                scales.size() * 2,
            )
        });
        out.extend_from_slice(unsafe {
            std::slice::from_raw_parts(
                biases.as_slice::<u16>().as_ptr() as *const u8,
                biases.size() * 2,
            )
        });
        let path = Self::draft_head_cache_path(dir);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let tmp = path.with_extension("lsdh.tmp");
        std::fs::write(&tmp, &out)?;
        std::fs::rename(&tmp, &path)?;
        Ok(())
    }

    fn forward_inner(
        &mut self,
        ids: &Array,
        caches: Option<&mut Vec<LayerCache>>,
        capture: bool,
    ) -> anyhow::Result<Array> {
        let _trace_fwd = lisa_mlx::trace::span("tower.forward");
        let s = ids.dim(1) as usize;
        let offset = caches
            .as_deref()
            .and_then(|c| {
                c.iter().find_map(|l| match l {
                    LayerCache::Full(f) => Some(f.offset),
                    _ => None,
                })
            })
            .unwrap_or(0);
        let pos = match caches.as_deref().and_then(|c| {
            c.iter().find_map(|l| match l {
                LayerCache::Full(f) => f.next_pos.clone(),
                _ => None,
            })
        }) {
            // Ragged batching: each stream has its own next absolute position
            // (the packed keys are right-aligned to `lmax`). Same shape
            // contract as the qwen4 tower. When absent, every stream shares
            // the packed `offset` — the [1,S] form the single-stream path
            // has always taken (bit-identical).
            Some(next) => {
                let mut data: Vec<f32> = Vec::with_capacity(next.len() * s);
                for p in &next {
                    for j in 0..s {
                        data.push((*p + j) as f32);
                    }
                }
                Array::from_slice(&data, &[next.len() as i32, s as i32])
            }
            None => positions(offset, s)?,
        };
        // cos/sin once per forward: the 16 attention layers used to recompute
        // (exp, mul, concat, cos, sin + a bf16 cast each) them per layer per
        // token. Same ops, same values — hoisted.
        let (cos, sin) = self.rope.cos_sin(&pos)?;
        let cos = cos.expand_dims(1)?;
        let sin = sin.expand_dims(1)?;
        // cos/sin cast to bf16 ONCE per forward (hoisted): every layer consumed
        // identical bf16 values — the per-layer `as_dtype(Bfloat16)` pair was
        // 2 dead dispatches per attention layer per step (specs/16).
        let cos_bf = cos.as_dtype(lisa_mlx::Dtype::Bfloat16)?;
        let sin_bf = sin.as_dtype(lisa_mlx::Dtype::Bfloat16)?;
        let mut caches = caches;
        // The residual stream threads UNADDED pairs: layer i returns
        // (post-attn sum, mlp out) and layer i+1 folds `sum + mlp` into its
        // input norm with the fused add+norm kernel — the per-layer badd + rms
        // pair becomes the one fused dispatch. The first layer norms the
        // embedding directly; the final norm folds the last pair. Rounding
        // chain is identical: bf16 add, then rms (the same fused kernel the
        // post-norm site has used since wave 2).
        let mut x = self.embed_tokens.forward(ids)?;
        let mut m_prev: Option<Array> = None;
        for i in 0..self.layers.len() {
            let _trace_layer = lisa_mlx::trace::span_detail("tower.layer", i as u64);
            let cache = caches.as_deref_mut().map(|c| &mut c[i]);
            let (h, m_i) =
                self.layers[i]
                    .forward(&x, m_prev.take(), cache, &cos_bf, &sin_bf, offset, capture)?;
            x = h;
            m_prev = Some(m_i);
        }
        let m = m_prev.take().expect("the tower has at least one layer");
        Ok(self.norm.forward_residual_add(&x, &m)?.1)
    }

    pub fn head(&self, mixed: &Array) -> lisa_mlx::error::Result<Array> {
        self.lm_head.forward(mixed)
    }

    pub fn warmup(&mut self, seed: &[u32]) -> anyhow::Result<()> {
        let base: Vec<u32> = if seed.is_empty() {
            (0..64).map(|i| 128 + (i * 37) % 4096).collect()
        } else {
            seed.to_vec()
        };
        // s=2048 dropped (specs/05): the shape's throwaway 2048-token forward
        // cost 2.2-2.7 s per warmup call while the JIT it pre-empted is only
        // ~0.45 s — the first real prefill chunk pays that compile instead,
        // for a net ~1.7 s TTFT win at every prompt length. The decode-shape
        // triple (1, 9, 40) stays: it covers the s==1 fused-decode templates
        // for ~0.5 s.
        for s in [1usize, 9, 40] {
            let t_shape = std::time::Instant::now();
            let ids: Vec<i32> = (0..s).map(|i| base[i % base.len()] as i32).collect();
            let arr = Array::from_slice(&ids, &[1i32, s as i32]);
            let mut caches = self.new_caches();
            let h = self.forward_inner(&arr, Some(&mut caches), false)?;
            let logits = self.head(&h)?;
            let _ = logits.eval();
            crate::ttft_mark(&format!(
                "warmup: shape s={s} (JIT+weight upload) {:.2}s",
                t_shape.elapsed().as_secs_f64()
            ));
        }
        let _ = lisa_mlx::memory::clear_cache();
        Ok(())
    }

    fn new_caches(&self) -> Vec<LayerCache> {
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

    pub fn max_position_embeddings(&self) -> usize {
        self.config.max_position_embeddings
    }
    pub fn eos_token_id(&self) -> i64 {
        self.config.eos_token_id
    }
    fn forward_capture(
        &mut self,
        ids: &Array,
        caches: Option<&mut Vec<LayerCache>>,
        capture: bool,
    ) -> anyhow::Result<(Array, Array)> {
        let h = self.forward_inner(ids, caches, capture)?;
        Ok((h.clone(), h))
    }
    fn prefill_multi(
        &mut self,
        tokens: &[u32],
        caches: &mut Vec<LayerCache>,
    ) -> anyhow::Result<(Array, Array)> {
        const CHUNK: usize = 2048;
        let t_prefill = std::time::Instant::now();
        let mut parts: Vec<Array> = Vec::new();
        for c in tokens.chunks(CHUNK) {
            let t_chunk = std::time::Instant::now();
            let ids: Vec<i32> = c.iter().map(|&t| t as i32).collect();
            let arr = Array::from_slice(&ids, &[1i32, c.len() as i32]);
            // Every chunk's full hidden must be kept: `multi` is the MTP
            // head's priming history and needs one row per prompt token.
            // Keeping only the last chunk (the old code) made any prompt over
            // one chunk crash the head prime with `cat: dim mismatch`.
            let (h, _) = self.forward_capture(&arr, Some(caches), false)?;
            parts.push(h);
            lisa_mlx::memory::trim_cache();
            crate::ttft_mark(&format!(
                "prefill: chunk len={} done in {:.2}s (cumulative {:.2}s)",
                c.len(),
                t_chunk.elapsed().as_secs_f64(),
                t_prefill.elapsed().as_secs_f64()
            ));
        }
        let h = if parts.len() == 1 {
            parts.pop().expect("non-empty")
        } else {
            let refs: Vec<&Array> = parts.iter().collect();
            lisa_mlx::ops::concatenate(&refs, 1).map_err(|e| anyhow::anyhow!("{e}"))?
        };
        let last_row = h.index((.., h.dim(1) - 1, ..));
        Ok((last_row, h))
    }
}

impl crate::models::LanguageModel for Qwen35Tower {
    fn batch_decode_ok(&self) -> bool {
        // The B>1 forward is batch-clean (specs/07 fix): the MLP swiglu2 keeps
        // its rank, ragged prefill takes per-stream RoPE positions, and the
        // batched decode consumes the packed keep mask. The fused decode
        // kernels are still mono-batch (item 2), so B>1 runs the generic
        // MLX batched path — slower per step, but correct.
        true
    }
    fn max_position_embeddings(&self) -> usize {
        self.max_position_embeddings()
    }
    fn eos_token_id(&self) -> i64 {
        self.eos_token_id()
    }
    fn new_caches(&self) -> Vec<LayerCache> {
        self.new_caches()
    }
    /// O3: full-attention layers × kv_heads × head_dim × (K+V) × bf16 (2 o).
    fn kv_bytes_per_token(&self) -> usize {
        let full = self
            .config
            .layer_types
            .iter()
            .filter(|t| t.as_str() == "full_attention")
            .count();
        full * self.config.num_key_value_heads * self.config.head_dim * 2 * 2
    }
    fn forward(
        &mut self,
        ids: &Array,
        caches: Option<&mut Vec<LayerCache>>,
    ) -> anyhow::Result<(Array, Array)> {
        self.forward_capture(ids, caches, false)
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
        Ok(self.prefill_multi(tokens, caches)?.0)
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

    // Speculative drafting via the native MTP head.
    fn has_drafter(&self) -> bool {
        self.mtp.is_some()
    }
    fn mtp_cost_key(&self) -> String {
        format!(
            "qwen3_5-{}L-{}h",
            self.config.num_hidden_layers, self.config.hidden_size
        )
    }
    fn take_draft_confidence(&mut self) -> Option<Array> {
        self.last_conf.take()
    }
    fn set_conf_capture(&mut self, on: bool) {
        self.conf_capture = on;
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
        let offset = self.mtp.as_ref().map(|h| h.cache_offset()).unwrap_or(0);
        let head = self
            .mtp
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("this model has no MTP head"))?;
        // `multi` for qwen3_5 is the trunk's final-norm hidden — the same
        // vector lm_head consumes, which is the head's `hidden` input.
        let (_, last) = head.forward(tokens, multi, &self.embed_tokens, offset)?;
        // Draft-side projection: a coarse 2-bit full-vocab readout (the draft-only head
        // built at load) shortlists its top-32, and those rows are RE-SCORED
        // exactly through the trunk head's own gathered rows. The re-score is
        // exact, so the proposal loses acceptance only when the true argmax
        // falls outside the coarse top-32 — verification is untouched either
        // way (proposal side only).
        let draft_id: Array = match &self.draft_head {
            Some(dh) => {
                use lisa_mlx::ops::argpartition_axis;
                use lisa_mlx::ops::indexing::take_axis;
                const K: usize = 32;
                let coarse = dh.forward(&last)?; // [1, V]
                let v: i32 = coarse.dim(-1);
                let kk: i32 = K as i32;
                anyhow::ensure!(v >= kk, "vocab {v} smaller than the top-{K} shortlist");
                let part = argpartition_axis(&(-&coarse), kk - 1, -1)
                    .map_err(|e| anyhow::anyhow!("{e}"))?; // [1, V] ids (repo convention: negated, first-k)
                let flat = part.reshape(&[v])?;
                let cands = flat
                    .index(0..kk)
                    .contiguous()
                    .map_err(|e| anyhow::anyhow!("{e}"))?; // [32] ids
                // Exact re-score through the trunk head's own rows.
                let w32 = self.lm_head.weight.take_axis(&cands, 0)?; // [32, pk]
                let s32 = self.lm_head.scales.take_axis(&cands, 0)?;
                let b32 = self.lm_head.biases.take_axis(&cands, 0)?;
                let rows = last.dim(0 as i32).max(1) as i32;
                let k = last.dim(-1);
                let xf = last.reshape(&[rows, k])?;
                let exact = lisa_mlx::ops::quantized_matmul(
                    &xf,
                    &w32,
                    &s32,
                    Some(&b32),
                    true,
                    self.lm_head.group_size,
                    self.lm_head.bits,
                )?; // [1, 32]
                let local = lisa_mlx::ops::indexing::argmax_axis(&exact, -1, None)
                    .map_err(|e| anyhow::anyhow!("{e}"))?; // [1] local idx
                let local = local.reshape(&[1])?;
                let id = take_axis(&cands, &local, 0)?; // [1] vocab id
                // Chunk-A confidence (port spec §2.3): log p_head of the chosen
                // token, from the exact re-scored shortlist row this step
                // already produced. Stored unevaluated — the round's chunk-A
                // sync reads it back only when the plan considers extension;
                // collapsed rounds never touch it. `None`-safe: two-chunk
                // stays off when the drafter exposes no confidence.
                let sm = exact.softmax_axis(-1).map_err(|e| anyhow::anyhow!("{e}"))?; // [1, 32]
                let p = take_axis(&sm, &local, -1).map_err(|e| anyhow::anyhow!("{e}"))?; // [1]
                let logp = p.log().map_err(|e| anyhow::anyhow!("{e}"))?;
                if self.conf_capture {
                    self.last_conf = Some(logp);
                }
                // Oracle (specs/08): record the exact re-scored shortlist so
                // the driver can rank the target's true token inside it after
                // verify. Diagnostic; a no-op unless the CLI enabled it.
                crate::core::oracle::push_arrays(Some(&cands), &exact)?;
                id.reshape(&[1, 1])?
            }
            None => {
                let logits = self.lm_head.forward(&last)?;
                let id = lisa_mlx::ops::indexing::argmax_axis(&logits, -1, None)
                    .map_err(|e| anyhow::anyhow!("{e}"))?
                    .reshape(&[1, 1])?;
                // Chunk-A confidence from the trunk head's own full
                // distribution (log p_head of the argmax). The 27B ships no
                // 2-bit shortlist head, so THIS is the head confidence the
                // tau gate reads — arguably the more faithful surface.
                if self.conf_capture {
                    let sm = logits.softmax_axis(-1).map_err(|e| anyhow::anyhow!("{e}"))?;
                    let p = lisa_mlx::ops::indexing::take_axis(&sm, &id, -1)
                        .map_err(|e| anyhow::anyhow!("{e}"))?;
                    self.last_conf = Some(p.log().map_err(|e| anyhow::anyhow!("{e}"))?);
                }
                crate::core::oracle::push_arrays(None, &logits)?;
                id
            }
        };
        // Keep the draft token on the device as `[1, 1]` (uint32 — the only
        // integer dtype the indexing kernels emit; same as the qwen4 drafter,
        // read back as the i32 bit pattern).
        let draft_id = draft_id.reshape(&[1, 1])?;
        Ok((draft_id, last))
    }
}
