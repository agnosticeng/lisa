//! Full attention for `qwen3_5`: GQA with a per-head output gate and learned
//! Q/K RMSNorm (no QSA indexer).

use lisa_mlx::ops::indexing::{Ellipsis, IndexOp};
use lisa_mlx::{Array, Dtype, fast, ops};

use crate::core::cache::FullAttentionCache;
use crate::core::loader::TensorSource;
use crate::core::norm::{RmsNorm, rope_partial};
use crate::core::quant::QuantizedLinear;

pub struct Qwen35Attention {
    pub q_proj: QuantizedLinear,
    /// Decode-only merged arm (specs/16 phase 2): q_proj's rows gathered and
    /// re-laid-out as `[Q | G]` (all q rows first, then all gate rows), so the
    /// decode qmv emits q and gate as two CONTIGUOUS row halves and the
    /// s==1 de-interleave `.contiguous()` copies disappear. Exact on the GEMV
    /// paths: each output row is the same dot over the same K values with the
    /// same packed weights/scales — the row gather preserves each row's
    /// packing verbatim (same argument as GDN's `in_proj_all` / MLP's
    /// `gate_up_proj`). `None` when ungated or gs/bits mismatch.
    pub q_gate_proj: Option<QuantizedLinear>,
    pub k_proj: QuantizedLinear,
    pub v_proj: QuantizedLinear,
    pub o_proj: QuantizedLinear,
    pub q_norm: RmsNorm,
    pub k_norm: RmsNorm,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub scale: f32,
    pub gated: bool,
}

impl Qwen35Attention {
    /// Whether to build the merged q|gate retile at load. **OFF by default: RAM
    /// wins.** The retile is a permuted COPY of `q_proj`'s rows resident across
    /// every full-attention layer, and the standing goal is to match the
    /// reference engine's footprint to the MB. The trade is measured, not
    /// guessed — removing it buys **0.77 GB** and costs **+3.56 ms on the serial
    /// step (38.69 -> 42.25, +9.2 %)** — and it is accepted deliberately: the
    /// fallback arm (`q_proj.forward` plus a per-step de-interleave) is exact, so
    /// nothing but speed moves. `LISA_QGATE_RETILE=1` restores it when a run is
    /// about latency instead of footprint.
    fn qgate_retile_enabled() -> bool {
        matches!(
            std::env::var("LISA_QGATE_RETILE").as_deref(),
            Ok("1") | Ok("on") | Ok("true")
        )
    }

    pub fn load<S: TensorSource>(
        src: &mut S,
        prefix: &str,
        eps: f32,
        heads: usize,
        kv_heads: usize,
        head_dim: usize,
        gated: bool,
        group_size: i32,
        bits: i32,
    ) -> anyhow::Result<Self> {
        let mut q_proj = QuantizedLinear::load(src, prefix, "q_proj")?;
        q_proj.set_quant(group_size, bits);
        let mut k_proj = QuantizedLinear::load(src, prefix, "k_proj")?;
        k_proj.set_quant(group_size, bits);
        let mut v_proj = QuantizedLinear::load(src, prefix, "v_proj")?;
        v_proj.set_quant(group_size, bits);
        let mut o_proj = QuantizedLinear::load(src, prefix, "o_proj")?;
        o_proj.set_quant(group_size, bits);
        // Merged q|gate projection (specs/16 phase 2): gather q_proj's rows
        // from the per-head [q|gate] interleave into a [Q | G] row-concat.
        // One-time at load; decode then needs ZERO de-interleave copies.
        let q_gate_proj = if gated && Self::qgate_retile_enabled() {
            let hd = head_dim;
            let h = heads;
            let qn = h * hd;
            let total = 2 * qn;
            let mut idx: Vec<u32> = Vec::with_capacity(total);
            for head in 0..h {
                for j in 0..hd {
                    idx.push((head * 2 * hd + j) as u32);
                }
            }
            for head in 0..h {
                for j in 0..hd {
                    idx.push((head * 2 * hd + hd + j) as u32);
                }
            }
            let idx = Array::from_slice(&idx, &[total as i32]);
            let gather = |a: &Array| -> lisa_mlx::error::Result<Array> {
                a.take_axis(&idx, 0)
            };
            Some(QuantizedLinear {
                weight: gather(&q_proj.weight)?,
                scales: gather(&q_proj.scales)?,
                biases: gather(&q_proj.biases)?,
                group_size: q_proj.group_size,
                bits: q_proj.bits,
            })
        } else {
            None
        };
        Ok(Self {
            q_proj,
            q_gate_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm: RmsNorm::load(src, &format!("{prefix}.q_norm"), eps, None)?,
            k_norm: RmsNorm::load(src, &format!("{prefix}.k_norm"), eps, None)?,
            heads,
            kv_heads,
            head_dim,
            scale: (head_dim as f32).powf(-0.5),
            gated,
        })
    }

    /// x: `[B, S, hidden]`. `cos`/`sin` are precomputed once per forward by the
    /// tower (f32, already expanded to `[B, 1, S, dims]`); each attention layer
    /// used to recompute (and re-cast) them per token.
    pub fn forward(
        &self,
        x: &Array,
        cos: &Array,
        sin: &Array,
        mut cache: Option<&mut FullAttentionCache>,
        _offset: usize,
    ) -> lisa_mlx::error::Result<Array> {
        let b = x.dim(0);
        let s = x.dim(1);
        let hd = self.head_dim as i32;
        let heads = self.heads as i32;

        // q_proj is [heads * 2 * head_dim], laid out per head as
        // [q_head | gate_head]. NOTE (specs/08 §3 evaluated, rejected): the
        // reference's single mlx_split (2 dispatches → 1) does NOT port. Our
        // `index` is a lazy view (zero dispatches), so the two-slice pattern
        // costs only its two contiguous copies; measured, the chunk-split
        // variant (pure strided gate sigmoid·mul, and even chunk→contiguous
        // copies) ran decode 25 → 15–18 tok/s — our strided copy/elementwise
        // kernels are the slow path. The real win here is spec 04's gate-tail
        // fusion, not the split.
        // Decode merged arm (specs/16 phase 2): ONE qmv over the [Q | G]
        // row-concat, q and gate as the two contiguous row halves. At s==1
        // each half is a single dense row, so the fused kernels can consume
        // the SLICE VIEWS raw (bindings honor the buffer offset; the s==1
        // wrapper size checks still hold) — the two de-interleave
        // `.contiguous()` copies per layer disappear. Bit-exact: same GEMV
        // rows, same bytes, same kernel reads. Prefill (s>1) keeps the stock
        // interleave path: multi-row slices are genuinely strided there.
        // NOTE: the MLP gate|up arm (DenseMlp) needed a `b == 1` guard — at
        // B>1 its N-concat qmv CORRUPTS slot >= 1 (specs/13 re-audit: cbatch
        // identical-prompt streams diverge at decode step 1 with token soup).
        // This attention rowgather measured CLEAN at M>1 (identical-prompt
        // parity + faster: 45.6 vs 48.2 ms/step at B=2), so it stays engaged
        // on the batch path. Residual B>1-vs-serial divergence is the §9.6
        // near-tie class (batched sdpa vs solo sdpa rounding), not corruption.
        let merged = self.gated && self.q_gate_proj.is_some() && s == 1;
        // The de-interleave `.contiguous()` copies are DEFERRED on the verify
        // path (specs/01): the fused `qk_norm_rope_rows` arm below reads the
        // interleaved q|gate rows RAW, so materializing q/gate here enqueues
        // two dead slot-priced copies per layer whenever that arm engages
        // (~75 µs GPU-slot each, 32 copies/S3-verify). `q`/`gate` are built
        // lazily from `qg_raw` only when the fused arm declines.
        let (mut q, mut gate, qg_raw) =
            if let (true, Some(qg_proj)) = (merged, self.q_gate_proj.as_ref()) {
                let y = qg_proj.forward(x)?.reshape(&[b, 2 * heads * hd])?; // [b, Q|G]
                let qn = heads * hd;
                let q = Some(
                    y.index((.., 0..qn))
                        .reshape(&[b, s, heads, hd])?,
                );
                let gate = if self.gated {
                    Some(y.index((.., qn..)).reshape(&[b, s, heads, hd])?)
                } else {
                    None
                };
                (q, gate, None)
            } else {
                let qg = self.q_proj.forward(x)?;
                let qg = qg.reshape(&[b, s, heads, 2, hd])?;
                // Pre-fill keeps the eager materialization (multi-row slices
                // are genuinely strided there and the fused arm is b==1-only).
                let eager = s == 1;
                let q = if eager {
                    Some(qg.index((Ellipsis, 0, ..)).contiguous()?)
                } else {
                    None
                };
                let gate = if self.gated {
                    if eager {
                        Some(qg.index((Ellipsis, 1, ..)).contiguous()?)
                    } else {
                        None
                    }
                } else {
                    None
                };
                (q, gate, Some(qg))
            };
        // Lazy de-interleave for the deferred case happens AFTER the fused
        // rows arm below (which consumes `qg_raw` directly and never needs
        // the copies).
        let k = self
            .k_proj
            .forward(x)?
            .reshape(&[b, s, self.kv_heads as i32, hd])?;
        let v = self
            .v_proj
            .forward(x)?
            .reshape(&[b, s, self.kv_heads as i32, hd])?;

        // Ragged batching supplies a per-stream keep mask (`[B,1,S,L]` bool);
        // it replaces the plain causal mask (the packed keys are right-aligned
        // and hide a per-stream padding prefix).
        let attn_mask = cache.as_deref().and_then(|c| c.attn_mask.clone());
        // sdpa_dense scores are [B, hk, r, ql, kl]; the batch's keep mask is
        // `[B, 1, 1, kl]` — re-express it as `[B, 1, 1, ql, kl]` so it
        // broadcasts per kv-head/group. Same values at every ql (decode).
        let mask5: Option<lisa_mlx::Array> = match &attn_mask {
            Some(m) => Some(m.reshape(&[b, 1, 1, s, m.dim(-1)])?),
            None => None,
        };
        let sdpa_mask = match &mask5 {
            Some(m) => fast::ScaledDotProductAttentionMask::Array(m),
            None => fast::ScaledDotProductAttentionMask::Causal,
        };
        // Per-slot batched attention: on a ragged
        // decode group each stream's query rows attend its OWN live window —
        // no padded compute, no mask upload, and the sdpa call takes the same
        // no-pad shape it has solo. We engage at every
        // length (see `per_slot_attention`'s deviation note): our stacked arm
        // is a per-slice GEMM chain, not one fused sdpa.
        // cos/sin arrive DÉJÀ en bf16 (cast hoisté au niveau tower — une seule
        // paire de dispatches par forward au lieu de 2 par couche, specs/16).
        // Valeurs identiques : le cast f32→bf16 est déterministe.
        let cos_bf = cos;
        let sin_bf = sin;
        let per_slot = b > 1 && attn_mask.is_some();

        // Fused decode QK-norm+RoPE (specs/08 §2): at s==1 one launch does
        // q_norm + k_norm + both partial ropes and emits the head-major
        // [1,H,1,hd] layout directly (the transposes vanish). Bit-exact vs
        // the composed chain below — see the kernel's parity contract and
        // `qk_norm_rope_matches_composed`. The wrapper is single-batch (it
        // rejects `q.size() != H*hd`, i.e. every B>1) and falls to the
        // composed path.
        if s == 1 {
            let stream = lisa_mlx::Stream::thread_local_or_default();
            if let Some((qk, kk)) = lisa_mlx::models::qwen35::kernels::qk_norm_rope(
                q.as_ref().expect("s==1 q is eager"),
                &k,
                &self.q_norm.weight,
                &self.k_norm.weight,
                &cos_bf,
                &sin_bf,
                self.q_norm.eps,
                heads,
                self.kv_heads as i32,
                &stream,
            ) {
                let v = v.transpose_axes(&[0, 2, 1, 3])?;
                let (live_k, live_v) = match cache.as_deref_mut() {
                    Some(c) => c.update(&kk, &v)?,
                    None => (kk, v),
                };
                let out = fast::scaled_dot_product_attention(
                    &qk,
                    &live_k,
                    &live_v,
                    self.scale,
                    sdpa_mask,
                    None,
                )?;
                return Ok(self.attention_tail(out, gate, b, s, heads, hd)?);
            }
        }

        // Fused verify QK-norm+RoPE (specs/24): at b==1, s in 2..=8 (the MTP
        // verify block) with the keep-mask absent, ONE launch per layer
        // replaces the two de-interleave `.contiguous()` copies, q_norm,
        // k_norm, both transposes and both `rope_partial` chains — it reads
        // the STOCK interleaved q|gate projection rows directly and emits the
        // head-major [1,h,S,hd] layouts plus the raw de-interleaved gate.
        // Per-(row, head) math is bit-identical to the pinned s==1 kernel
        // (one threadgroup per row+head, same reduction/rounding points);
        // tail-ULP scoped per AGENTS §9.7.
        if b == 1 && s > 1 && self.gated && attn_mask.is_none() {
            if let Some(qg) = qg_raw.as_ref() {
                let stream = lisa_mlx::Stream::thread_local_or_default();
                let qg2 = qg.reshape(&[s, heads * 2 * hd])?;
                let k2 = k.reshape(&[s, self.kv_heads as i32 * hd])?;
                if let Some((qk, kk, gate)) =
                    lisa_mlx::models::qwen35::kernels::qk_norm_rope_rows(
                        &qg2,
                        &k2,
                        &self.q_norm.weight,
                        &self.k_norm.weight,
                        &cos_bf,
                        &sin_bf,
                        self.q_norm.eps,
                        heads,
                        self.kv_heads as i32,
                        s,
                        &stream,
                    )
                {
                    let v = v.transpose_axes(&[0, 2, 1, 3])?;
                    let (live_k, live_v) = match cache.as_deref_mut() {
                        Some(c) => c.update(&kk, &v)?,
                        None => (kk, v),
                    };
                    let out = fast::scaled_dot_product_attention(
                        &qk,
                        &live_k,
                        &live_v,
                        self.scale,
                        fast::ScaledDotProductAttentionMask::Causal,
                        None,
                    )?;
                    let gate = gate.reshape(&[b, s, heads, hd])?;
                    return Ok(self.attention_tail(out, Some(gate), b, s, heads, hd)?);
                }
            }
        }

        // Deferred de-interleave: only reached when the fused rows arm above
        // declined (or the shape is out of its gate) — build q/gate from the
        // raw interleaved projection rows now.
        if q.is_none() {
            if let Some(qg) = qg_raw.as_ref() {
                let qg4 = qg.reshape(&[b, s, heads, 2, hd])?;
                q = Some(qg4.index((Ellipsis, 0, ..)).contiguous()?);
                if gate.is_none() && self.gated {
                    gate = Some(qg4.index((Ellipsis, 1, ..)).contiguous()?);
                }
            }
        }
        let q = q.expect("q materialized for the composed path");
        let gate = if self.gated {
            Some(gate.expect("gate materialized for the composed path"))
        } else {
            None
        };

        let q = self.q_norm.forward(&q)?;
        let k = self.k_norm.forward(&k)?;

        // (cos_bf/sin_bf déjà en bf16 — voir le hoist au-dessus. Le chemin
        // composé d'origine castait vers x.dtype(); x est bf16 sur tout le
        // chemin qwen3_5 (embed quantisé → bf16), donc mêmes valeurs.)
        let q = rope_partial(&q.transpose_axes(&[0, 2, 1, 3])?, &cos_bf, &sin_bf)?;
        let k = rope_partial(&k.transpose_axes(&[0, 2, 1, 3])?, &cos_bf, &sin_bf)?;
        let v = v.transpose_axes(&[0, 2, 1, 3])?;

        let (live_k, live_v) = match cache.as_deref_mut() {
            Some(c) => c.update(&k, &v)?,
            None => (k, v),
        };
        let out = if per_slot {
            // The per-slot read (see the `per_slot` note above). The stacked
            // masked read stays as the fallback (helper returned None).
            match cache
                .as_deref()
                .and_then(|c| Self::per_slot_attention(&q, &live_k, &live_v, c, self.scale).ok())
                .flatten()
            {
                Some(o) => o,
                None => fast::scaled_dot_product_attention(
                    &q,
                    &live_k,
                    &live_v,
                    self.scale,
                    sdpa_mask,
                    None,
                )?,
            }
        } else {
            fast::scaled_dot_product_attention(&q, &live_k, &live_v, self.scale, sdpa_mask, None)?
        };
        self.attention_tail(out, gate, b, s, heads, hd)
    }

    /// Per-slot batched attention read on the packed layout.
    ///
    /// Each stream's query rows attend its OWN live KV window
    /// `[origin_i, offset)` of the packed cache: no padded compute, no mask,
    /// and each sdpa call sees the same `[1, h, 1, kl]` shape it has on a solo
    /// stream. `origin_i = offset - kv_len_i`, and `kv_len_i == next_pos[i]`
    /// is the pack invariant (batch.rs packs `next_pos = lens` and both
    /// advance one per step).
    ///
    /// DEVIATION from the stacked-only policy (specs/13 §5, measured): the
    /// per-slot arm could be gated past BATCHED_PER_SLOT_ATTN_MIN_KV = 1024,
    /// but lisa's stacked arm is `sdpa_dense`'s per-(batch, kv-head, repeat)
    /// GEMM chain — at B=2 that is ~100k `gemm_out` dispatches per 64-token
    /// decode (4.6x the B=1 total). The per-slot read hits the fused
    /// `sdpa_vector` kernel at every length, so on a ragged decode step it is
    /// always the cheaper arm here. Below the floor only the MEMORY cap
    /// (batch.rs `group_keep_count`, the packed-cache bytes) still applies.
    fn per_slot_attention(
        q: &Array,
        live_k: &Array,
        live_v: &Array,
        cache: &FullAttentionCache,
        scale: f32,
    ) -> lisa_mlx::error::Result<Option<Array>> {
        let Some(next_pos) = &cache.next_pos else {
            return Ok(None);
        };
        // q is [b, heads, s, hd] (head-major at the sdpa site); decode = s 1.
        if q.dim(2) != 1 || next_pos.len() != q.dim(0) as usize {
            return Ok(None);
        }
        let offset = cache.offset;
        if next_pos.iter().any(|&l| l > offset) {
            return Ok(None);
        }
        if *next_pos.iter().max().unwrap_or(&0) == 0 {
            return Ok(None);
        }
        let b = q.dim(0);
        // One-shot engagement marker: output equality alone cannot tell the
        // per-slot arm from the stacked one.
        static PER_SLOT_ENGAGED: std::sync::atomic::AtomicBool =
            std::sync::atomic::AtomicBool::new(false);
        if !PER_SLOT_ENGAGED.swap(true, std::sync::atomic::Ordering::Relaxed) {
            eprintln!(
                "[batched] per-slot attention engaged (slots={b}): no KV pad or mask on the read"
            );
        }
        let mut outs = Vec::with_capacity(b as usize);
        for i in 0..b {
            let kv_len = next_pos[i as usize] as i32;
            let origin = offset as i32 - kv_len;
            let q_i = q
                .index((i, ..))
                .reshape(&[1, q.dim(1), q.dim(2), q.dim(3)])?; // [1, heads, 1, hd]
            let k_i = live_k
                .index((i, .., origin..offset as i32, ..))
                .reshape(&[1, live_k.dim(1), kv_len, live_k.dim(3)])?; // [1, kv_h, kl, hd]
            let v_i = live_v
                .index((i, .., origin..offset as i32, ..))
                .reshape(&[1, live_v.dim(1), kv_len, live_v.dim(3)])?;
            // Decode (s == 1): every key is a causal prefix, so the causal
            // mask is vacuous — the same `mode = ""` the per-slot arm uses.
            outs.push(fast::scaled_dot_product_attention(
                &q_i,
                &k_i,
                &v_i,
                scale,
                fast::ScaledDotProductAttentionMask::Causal,
                None,
            )?);
        }
        let refs: Vec<&Array> = outs.iter().collect();
        Ok(Some(ops::concatenate(&refs, 0)?))
    }

    /// The gate + o_proj tail shared by the fused and composed decode paths.
    fn attention_tail(
        &self,
        out: Array,
        gate: Option<Array>,
        b: i32,
        s: i32,
        heads: i32,
        hd: i32,
    ) -> lisa_mlx::error::Result<Array> {
        // `out` is the sdpa output [b, heads, s, hd]. (specs/04) With a gate,
        // the fused `sigmoid_mul_tail` reads gate [b,s,h,hd] and out [b,h,s,hd]
        // through their strides and emits the merged [b, s, h*hd] directly —
        // the per-layer transpose materialization copy disappears. The
        // fallback keeps the transpose + sigmoid_mul chain.
        let out = match gate {
            Some(g) => {
                let stream = lisa_mlx::Stream::thread_local_or_default();
                match lisa_mlx::models::qwen35::kernels::sigmoid_mul_tail(
                    &g,
                    &out,
                    b,
                    s,
                    heads,
                    hd,
                    &stream,
                ) {
                    Some(r) => r,
                    None => {
                        let ot = out.transpose_axes(&[0, 2, 1, 3])?; // [b,s,h,hd]
                        // Fused sigmoid-gate multiply (specs/04, specs/08 §3
                        // tail): `mlx_sigmoid(g) * a` in one launch — same
                        // exact_header helper and rounding structure as the
                        // validated swiglu2.
                        let stream = lisa_mlx::Stream::thread_local_or_default();
                        match lisa_mlx::models::qwen35::kernels::sigmoid_mul(&g, &ot, &stream) {
                            Some(r) => r,
                            None => {
                                let sig = ops::sigmoid(&g)?;
                                ot.multiply(&sig)?
                            }
                        }
                    }
                }
            }
            None => out.transpose_axes(&[0, 2, 1, 3])?.reshape(&[b, s, heads * hd])?,
        };
        let out = out.as_dtype(Dtype::Bfloat16)?;
        self.o_proj.forward(&out)
    }
}
