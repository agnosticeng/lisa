//! Gated DeltaNet linear-attention layer (the 36 recurrent layers).

use lisa_mlx::ops::indexing::IndexOp;
use lisa_mlx::{Array, Dtype, ops};

use crate::core::cache::GdnCache;
use crate::core::loader::TensorSource;
use crate::core::norm::RmsNormGated;
use crate::core::quant::QuantizedLinear;
use lisa_mlx::kernels::{
    gated_delta_kernel, gated_delta_ops, gated_delta_ops_capture, gdn_rows, gdn_two_row,
};

/// A shared f32[1] zero scalar (placeholder projection for the fused path).
/// `Array` is immutable + `Arc`-backed, so a process-wide instance is safe.
fn zero_scalar() -> Array {
    static ZERO: std::sync::OnceLock<Array> = std::sync::OnceLock::new();
    ZERO.get_or_init(|| Array::from_slice(&[0f32], &[1]))
        .clone()
}

/// Gated deltanet block.
pub struct GatedDeltaNet {
    pub in_proj_qkv: QuantizedLinear,
    pub in_proj_z: QuantizedLinear,
    pub in_proj_b: QuantizedLinear,
    pub in_proj_a: QuantizedLinear,
    /// The four input projections concatenated `[qkv | z | b | a]` (one GEMM),
    /// used by the fused S=1 decode kernel.
    pub in_proj_all: QuantizedLinear,
    pub z_offset: i32,
    pub b_offset: i32,
    pub a_offset: i32,
    pub proj_width: i32,
    pub conv1d_weight: Array, // [convDim, K, 1]
    pub a_log: Array,         // bf16 [48]
    /// `-exp(a_log.f32)`, precomputed once at load (the engine's bind-time
    /// `negExpALog`); recomputing it per layer was three lazy ops each.
    pub neg_exp_alog: Array,
    pub dt_bias: Array, // bf16 [48]
    pub norm: RmsNormGated,
    pub out_proj: QuantizedLinear,
    pub value_heads: usize,
    pub key_heads: usize,
    pub key_head_dim: usize,
    pub value_head_dim: usize,
    pub conv_kernel_size: usize,
    pub conv_dim: usize,
    /// Qwen3.5 (27B) normalizes q/k with an L2 norm (divides by `‖x‖`), while
    /// Flash-Next uses MLX mean-RMS. Same scales `1/hd`, `1/√hd`; the base
    /// differs by `√hd`.
    pub l2_norm: bool,
    /// Qwen3.5 (`output_gate_type=swish`) gates the GDN output with `silu(z)`;
    /// Flash-Next uses `sigmoid(z)`.
    pub output_gate_silu: bool,
    /// Per-step constants (specs/08 item 3): the bf16 q/k scales depend only
    /// on the head geometry, so they are built once instead of per layer per
    /// step. Pure host-side reuse — same values, same dtype, bit-identical.
    q_scale: std::sync::OnceLock<Array>,
    k_scale: std::sync::OnceLock<Array>,
}

impl GatedDeltaNet {
    /// The hoisted bf16 q/k scale scalars (`inv_scale²·base`, `inv_scale·base`).
    fn scales(&self) -> (&Array, &Array) {
        let inv_scale = (self.key_head_dim as f32).powf(-0.5);
        // L2 (Qwen3.5) vs mean-RMS (Flash-Next): the mean-RMS base is
        // `‖x‖/√hd`, so one extra `1/√hd` converts it to the L2 base.
        let base_adj = if self.l2_norm {
            (self.key_head_dim as f32).powf(-0.5)
        } else {
            1.0
        };
        (
            self.q_scale
                .get_or_init(|| crate::core::norm::bf16_scalar(inv_scale * inv_scale * base_adj)),
            self.k_scale
                .get_or_init(|| crate::core::norm::bf16_scalar(inv_scale * base_adj)),
        )
    }
    pub fn load<S: TensorSource>(
        src: &mut S,
        prefix: &str,
        eps: f32,
        key_heads: usize,
        value_heads: usize,
        key_head_dim: usize,
        value_head_dim: usize,
        conv_kernel_size: usize,
        group_size: i32,
        bits: i32,
    ) -> anyhow::Result<Self> {
        let key_dim = key_heads * key_head_dim;
        let value_dim = value_heads * value_head_dim;
        let conv_dim = key_dim * 2 + value_dim;
        let mut conv1d_weight = src.get_bf16(&format!("{prefix}.conv1d.weight"))?;
        // Torch ships (C, 1, K); MLX wants (C, K, 1). Idempotent.
        let cs = conv1d_weight.shape();
        if cs.len() == 3 && cs[1] == 1 && cs[2] > 1 {
            conv1d_weight = conv1d_weight.transpose_axes(&[0, 2, 1])?;
        }
        let mut in_proj_qkv = QuantizedLinear::load(src, prefix, "in_proj_qkv")?;
        in_proj_qkv.set_quant(group_size, bits);
        let mut in_proj_z = QuantizedLinear::load(src, prefix, "in_proj_z")?;
        in_proj_z.set_quant(group_size, bits);
        let mut in_proj_b = QuantizedLinear::load(src, prefix, "in_proj_b")?;
        in_proj_b.set_quant(group_size, bits);
        let mut in_proj_a = QuantizedLinear::load(src, prefix, "in_proj_a")?;
        in_proj_a.set_quant(group_size, bits);
        let mut out_proj = QuantizedLinear::load(src, prefix, "out_proj")?;
        out_proj.set_quant(group_size, bits);
        // Fused form for the S=1 decode kernel: one GEMM over the concatenation
        // of the four parts. Row concatenation is exact on the GEMV paths.
        let cat = |a: &Array, b: &Array| lisa_mlx::ops::concatenate(&[a, b], 0).unwrap();
        let in_proj_all = QuantizedLinear {
            weight: {
                let qz = cat(&in_proj_qkv.weight, &in_proj_z.weight);
                let qzb = cat(&qz, &in_proj_b.weight);
                cat(&qzb, &in_proj_a.weight)
            },
            scales: {
                let qz = cat(&in_proj_qkv.scales, &in_proj_z.scales);
                let qzb = cat(&qz, &in_proj_b.scales);
                cat(&qzb, &in_proj_a.scales)
            },
            biases: {
                let qz = cat(&in_proj_qkv.biases, &in_proj_z.biases);
                let qzb = cat(&qz, &in_proj_b.biases);
                cat(&qzb, &in_proj_a.biases)
            },
            group_size: in_proj_qkv.group_size,
            bits: in_proj_qkv.bits,
        };
        let proj_width = in_proj_all.dims_out() as i32;
        let z_offset = in_proj_qkv.dims_out() as i32;
        let b_offset = z_offset + in_proj_z.dims_out() as i32;
        let a_offset = b_offset + in_proj_b.dims_out() as i32;
        // ROW-JOIN: rebuild the four parts as VIEWS into the fused buffer and
        // release their originals, so only ONE copy of the projection weights
        // stays resident. The loader builds the joined buffer, then swaps every
        // part's map entry for a slice of it; the guard ("only map-owned handles
        // are fused") holds here because all four originals came from
        // `QuantizedLinear::load`. Rows are dim 0 and contiguous
        // (weight [N, K*bits/32], scales/biases [N, K/gs]), so `Array::index`
        // yields a raw slice view (specs/04) and no data moves. `IndexElem` is
        // implemented for Range<i32>, not Range<usize>. Before this, `in_proj_all`
        // was a pure `cat` duplicate: four parts PLUS a fused copy (~3.5 GiB of
        // the 27 GiB RSS — the same waste the attention still carries at
        // attention.rs:55, where the merged Q|gate is a `gather` copy).
        {
            let rows = |of: &Array, lo: i32, hi: i32| -> Array { of.index((lo..hi, ..)) };
            let (qn, zn, bn) = (
                in_proj_qkv.dims_out() as i32,
                in_proj_z.dims_out() as i32,
                in_proj_b.dims_out() as i32,
            );
            in_proj_qkv.weight = rows(&in_proj_all.weight, 0, qn);
            in_proj_qkv.scales = rows(&in_proj_all.scales, 0, qn);
            in_proj_qkv.biases = rows(&in_proj_all.biases, 0, qn);
            in_proj_z.weight = rows(&in_proj_all.weight, qn, qn + zn);
            in_proj_z.scales = rows(&in_proj_all.scales, qn, qn + zn);
            in_proj_z.biases = rows(&in_proj_all.biases, qn, qn + zn);
            in_proj_b.weight = rows(&in_proj_all.weight, qn + zn, qn + zn + bn);
            in_proj_b.scales = rows(&in_proj_all.scales, qn + zn, qn + zn + bn);
            in_proj_b.biases = rows(&in_proj_all.biases, qn + zn, qn + zn + bn);
            in_proj_a.weight = rows(&in_proj_all.weight, qn + zn + bn, proj_width);
            in_proj_a.scales = rows(&in_proj_all.scales, qn + zn + bn, proj_width);
            in_proj_a.biases = rows(&in_proj_all.biases, qn + zn + bn, proj_width);
        }
        let a_log = src.get_bf16(&format!("{prefix}.A_log"))?;
        // `negExpALog` is computed once at bind in the engine, not per forward.
        let neg_exp_alog = -(lisa_mlx::ops::exp(&a_log.as_dtype(Dtype::Float32)?)?);
        Ok(Self {
            in_proj_qkv,
            in_proj_z,
            in_proj_b,
            in_proj_a,
            in_proj_all,
            z_offset,
            b_offset,
            a_offset,
            proj_width,
            conv1d_weight,
            a_log,
            neg_exp_alog,
            dt_bias: src.get_bf16(&format!("{prefix}.dt_bias"))?,
            norm: RmsNormGated::load(src, &format!("{prefix}.norm"), eps)?,
            out_proj,
            value_heads,
            key_heads,
            key_head_dim,
            value_head_dim,
            conv_kernel_size,
            conv_dim,
            l2_norm: false,
            output_gate_silu: false,
            q_scale: std::sync::OnceLock::new(),
            k_scale: std::sync::OnceLock::new(),
        })
    }

    /// x: [B, S, hidden]; cache holds conv + ssm state, updated in place.
    ///
    /// `capture` (speculative verify only) additionally records the SSM state
    /// after every position and the raw conv input so the cache can be rolled
    /// back to an accepted prefix.
    pub fn forward(
        &self,
        x: &Array,
        mut cache: Option<&mut GdnCache>,
        capture: bool,
    ) -> lisa_mlx::error::Result<Array> {
        let b = x.dim(0);
        let s = x.dim(1);
        // S=1 (and not a verify capture) runs the engine's fused
        // `track_gdn_decode_complete` when the head geometry matches.
        if !capture && s == 1 {
            if let Some(out) = self.forward_decode_complete(x, cache.as_deref_mut()) {
                return Ok(out);
            }
        }

        // The engine splits the projection into its four parts only for eligible
        // wide prefill windows (batch 1, S > 8, bf16, no capture). Every other
        // window -- the MTP verify/capture and small prefill -- uses the fused
        // `in_proj_all` and reads the gates from offsets in it.
        let separate = !capture && b == 1 && s > 8;
        let value_dim = (self.value_heads * self.value_head_dim) as i32;
        let (mixed_qkv, z, b_proj, a_proj, fused_proj): (
            Array,
            Array,
            Array,
            Array,
            Option<Array>,
        );
        if separate {
            mixed_qkv = self.in_proj_qkv.forward(x)?;
            let z4 = self.in_proj_z.forward(x)?;
            b_proj = self.in_proj_b.forward(x)?;
            a_proj = self.in_proj_a.forward(x)?;
            z = z4.reshape(&[b, s, self.value_heads as i32, self.value_head_dim as i32])?;
            fused_proj = None;
        } else {
            let proj = self.in_proj_all.forward(x)?;
            let z_off = self.z_offset;
            // RAW slice VIEW (specs/04): the z gate rides at offsets in the
            // fused projection and the fused `gated_rms` arm binds it through
            // its strides — the per-layer slice→4D reshape materializing copy
            // (48/verify, ~75 µs GPU-slot each) disappears. The composed
            // fallbacks below materialize only when the fused arm declines.
            z = proj.index((.., .., z_off..(z_off + value_dim)));
            // The gates ride in `proj` at offsets; `mixed_qkv` stays a raw slice
            // VIEW (no contiguous copy): the prep kernel reads proj directly and
            // emits the rollback conv input itself (specs/18).
            mixed_qkv = proj.index((.., .., 0..self.conv_dim as i32));
            // Placeholder zeros for the fused-proj path (the gates are read
            // from offsets in `proj`): per-step constants, built once.
            b_proj = zero_scalar();
            a_proj = zero_scalar();
            fused_proj = Some(proj.clone());
        }

        // Fused prep (`track_gdn_prep` / `track_p12_gdn_prep_split_inputs`):
        // causal conv + silu + q/k RMS + gates in ONE launch. The verify
        // (capture) window builds the rollback conv input separately, instead
        // of the ~10-op chain the prep reproduces bit-for-bit.
        let use_prep = true;
        let key_dim = (self.key_heads * self.key_head_dim) as i32;
        let conv_dim = self.conv_dim as i32;
        let kc = self.conv_kernel_size as i32;
        let (q, k, v, g, beta, new_conv): (Array, Array, Array, Array, Array, Option<Array>);
        let mut capture_conv_input: Option<Array> = None;
        if use_prep {
            let conv_state = match cache.as_deref().and_then(|c| c.conv.clone()) {
                Some(state) => state,
                None => ops::zeros::<half::bf16>(&[
                    b,
                    (self.conv_kernel_size - 1) as i32,
                    self.conv_dim as i32,
                ])?,
            };
            let neg_exp_alog = self.neg_exp_alog.clone();
            let stream = lisa_mlx::Stream::thread_local_or_default();
            let r: Option<(
                Array,
                Array,
                Array,
                Array,
                Array,
                Array,
                Option<Array>,
            )> = match &fused_proj {
                Some(proj) => lisa_mlx::kernels::gdn_prep_fused(
                    proj,
                    &conv_state,
                    &self.conv1d_weight,
                    &neg_exp_alog,
                    &self.dt_bias,
                    self.b_offset,
                    self.a_offset,
                    s,
                    self.key_heads as i32,
                    self.value_heads as i32,
                    self.key_head_dim as i32,
                    self.value_head_dim as i32,
                    kc,
                    self.proj_width,
                    conv_dim,
                    &stream,
                )
                .map(|(qn, kn, vv, gg, bb, co, ci)| (qn, kn, vv, gg, bb, co, Some(ci))),
                None => lisa_mlx::kernels::gdn_prep(
                    &mixed_qkv,
                    &conv_state,
                    &self.conv1d_weight,
                    &neg_exp_alog,
                    &self.dt_bias,
                    &b_proj,
                    &a_proj,
                    s,
                    self.key_heads as i32,
                    self.value_heads as i32,
                    self.key_head_dim as i32,
                    self.value_head_dim as i32,
                    kc,
                    conv_dim,
                    &stream,
                )
                .map(|(qn, kn, vv, gg, bb, co)| (qn, kn, vv, gg, bb, co, None)),
                // (the split-proj arm is wide prefill only — never a capture)
            };
            let (qn, kn, vv, gg, bb, conv_out, conv_in) = r.ok_or_else(|| {
                lisa_mlx::error::Exception::custom("gdn_prep kernel unavailable".to_string())
            })?;
            (q, k, v, g, beta, new_conv) = (qn, kn, vv, gg, bb, Some(conv_out));
            if let Some(ci) = conv_in {
                // The rollback conv input comes out of the prep kernel itself
                // (state rows then the proj rows — the composed concatenate's
                // exact values, one fewer dispatch and no mixed_qkv copy).
                capture_conv_input = Some(ci);
            }
        } else {
            let conv_state = match cache.as_deref() {
                Some(c) => c.conv.clone(),
                None => None,
            };
            let conv_input = match conv_state {
                Some(state) => ops::concatenate(&[&state, &mixed_qkv], 1)?,
                None => {
                    let zeros = ops::zeros::<half::bf16>(&[
                        b,
                        (self.conv_kernel_size - 1) as i32,
                        self.conv_dim as i32,
                    ])?;
                    ops::concatenate(&[&zeros, &mixed_qkv], 1)?
                }
            };
            if let Some(c) = cache.as_deref_mut() {
                let start = -(self.conv_kernel_size as i32 - 1);
                c.conv = Some(conv_input.index((.., start.., ..)).contiguous()?);
            }
            capture_conv_input = Some(conv_input.clone());
            let conv_out = ops::conv1d(
                &conv_input,
                &self.conv1d_weight,
                None,
                None,
                None,
                Some(self.conv_dim as i32),
            )?;
            let conv_out = lisa_mlx::nn::silu(&conv_out)?;
            let qq = conv_out.index((.., .., 0..key_dim));
            let kk = conv_out.index((.., .., key_dim..(2 * key_dim)));
            let vv = conv_out.index((.., .., (2 * key_dim)..));
            let qq = qq.reshape(&[b, s, self.key_heads as i32, self.key_head_dim as i32])?;
            let kk = kk.reshape(&[b, s, self.key_heads as i32, self.key_head_dim as i32])?;
            let vv = vv.reshape(&[b, s, self.value_heads as i32, self.value_head_dim as i32])?;
            let (q_scale, k_scale) = self.scales();
            let qq = fast_rms_none(&qq, 1e-6)? * q_scale;
            let kk = fast_rms_none(&kk, 1e-6)? * k_scale;
            let bb = ops::sigmoid(&b_proj)?.as_dtype(Dtype::Float32)?;
            let neg_exp_alog = self.neg_exp_alog.clone();
            let ax = a_proj.add(&self.dt_bias)?;
            let sp = crate::core::norm::bf16_logaddexp0(&ax)?;
            let sp = sp.as_dtype(Dtype::Float32)?;
            let gg = ops::exp(&(neg_exp_alog * sp))?;
            (q, k, v, g, beta, new_conv) = (qq, kk, vv, gg, bb, None);
        }
        if let Some(c) = cache.as_deref_mut() {
            if let Some(nc) = new_conv {
                c.conv = Some(nc);
            }
        }

        // The fp32 SSM state; zeros on the first call.
        let zeros_state = || {
            let b_usize = b as usize;
            Array::from_slice(
                &vec![0f32; b_usize * self.value_heads * self.value_head_dim * self.key_head_dim],
                &[
                    b,
                    self.value_heads as i32,
                    self.value_head_dim as i32,
                    self.key_head_dim as i32,
                ],
            )
        };
        let state_in = match cache.as_deref() {
            Some(c) => c.ssm.clone().unwrap_or_else(&zeros_state),
            None => zeros_state(),
        };

        // The recurrence: the custom Metal kernel (same source string as the
        // reference); the ops loop is the fallback for non-Metal builds.
        let (out, state) = {
            let stream = lisa_mlx::Stream::thread_local_or_default();
            // Prefill (T > 8, no capture): the engine's 4-row `track_gdn_rows`.
            if s > 8
                && !capture
                && let Some(r) = gdn_rows(&q, &k, &v, &g, &beta, &state_in, false, &stream)
            {
                r
            } else if s > 1 {
                // The engine's two-row recurrence for 1 < T <= 8 (verify and
                // short prefill). The capture window rebuilds its per-position
                // state every round; reuse the persistent buffers so the
                // (large) state tensor is not reallocated per GDN layer.
                let (py, ps) = if capture {
                    match cache.as_deref() {
                        Some(c) => (c.rec_y_buf.clone(), c.capture_ssm.clone()),
                        None => (None, None),
                    }
                } else {
                    (None, None)
                };
                let (hv_i, dv_i, dk_i) = (
                    self.value_heads as i32,
                    self.value_head_dim as i32,
                    self.key_head_dim as i32,
                );
                let reuse = matches!((&py, &ps), (Some(y), Some(st))
                    if y.shape() == [b, s, hv_i, dv_i] && st.shape() == [b * s, hv_i, dv_i, dk_i]);
                let r = if reuse {
                    lisa_mlx::kernels::gdn_two_row_into(
                        &q,
                        &k,
                        &v,
                        &g,
                        &beta,
                        &state_in,
                        capture,
                        py.as_ref().unwrap(),
                        ps.as_ref().unwrap(),
                        &stream,
                    )
                } else {
                    gdn_two_row(&q, &k, &v, &g, &beta, &state_in, capture, &stream)
                };
                match r {
                    Some(r) => r,
                    None => {
                        let ops = || {
                            if capture {
                                gated_delta_ops_capture(&q, &k, &v, &g, &beta, &state_in)
                            } else {
                                gated_delta_ops(&q, &k, &v, &g, &beta, Some(&state_in))
                            }
                        };
                        gated_delta_kernel(&q, &k, &v, &g, &beta, &state_in, capture, &stream)
                            .unwrap_or_else(|| ops().unwrap())
                    }
                }
            } else {
                let ops = || {
                    if capture {
                        gated_delta_ops_capture(&q, &k, &v, &g, &beta, &state_in)
                    } else {
                        gated_delta_ops(&q, &k, &v, &g, &beta, Some(&state_in))
                    }
                };
                gated_delta_kernel(&q, &k, &v, &g, &beta, &state_in, capture, &stream)
                    .unwrap_or_else(|| ops().unwrap())
            }
        };

        if let Some(c) = cache.as_deref_mut() {
            if capture {
                // `state` is the per-position sequence [B*S, Hv, Dv, Dk]; keep
                // the final slot live and stash the sequence + conv input for
                // rollback.
                let last = state.index((s - 1, .., .., ..)).contiguous()?.detach();
                c.rec_y_buf = Some(out.clone());
                c.capture_ssm = Some(state);
                c.ssm = Some(last);
                c.capture_conv_input = capture_conv_input.clone();
            } else {
                c.ssm = Some(state);
            }
        }

        // Output gating: gate(z) * rmsNorm(out) with a plain-scale weight.
        // Fused `track_gated_rms` (butterfly RMS + gate in one launch); the
        // swish-gate arm (`gated_rms_silu`, specs/01) replaces the composed
        // rms_norm + nn::silu (= Sigmoid + bmul) + multiply chain — 4
        // dispatches/layer down to 1 on the qwen3_5 verify path.
        let normed = if self.output_gate_silu {
            let stream = lisa_mlx::Stream::thread_local_or_default();
            // z stays the raw [b, s, hv*dv] slice VIEW; the fused wrapper binds
            // it through its strides (no materializing reshape).
            let hv = self.value_heads as i32;
            let dv = self.value_head_dim as i32;
            match lisa_mlx::kernels::gated_rms_silu(
                &out,
                &z,
                &self.norm.weight,
                hv,
                dv,
                self.norm.eps,
                &stream,
            ) {
                Some(r) => r,
                None => {
                    let z4 = z.copied()?.reshape(&[b, s, hv, dv])?;
                    let rms = lisa_mlx::fast::rms_norm(&out, Some(&self.norm.weight), self.norm.eps)?;
                    crate::core::norm::bf16_silu(&z4)?.multiply(&rms)?
                }
            }
        } else {
            let stream = lisa_mlx::Stream::thread_local_or_default();
            let hv = self.value_heads as i32;
            let dv = self.value_head_dim as i32;
            match lisa_mlx::kernels::gated_rms(
                &out,
                &z,
                &self.norm.weight,
                hv,
                dv,
                self.norm.eps,
                &stream,
            ) {
                Some(r) => r,
                None => {
                    let z4 = z.copied()?.reshape(&[b, s, hv, dv])?;
                    self.norm.forward(&out, &z4)?
                }
            }
        };
        let normed = normed.reshape(&[b, s, -1])?;
        let attended = self.out_proj.forward(&normed)?;
        Ok(attended)
    }
}

impl GatedDeltaNet {
    /// The engine's fused S=1 GDN. Returns `None` when the geometry or a
    /// kernel is unavailable, so the caller falls back to the generic path.
    fn forward_decode_complete(
        &self,
        x: &Array,
        mut cache: Option<&mut GdnCache>,
    ) -> Option<Array> {
        let b = x.dim(0);
        let s = x.dim(1);
        let hk = self.key_heads as i32;
        let hv = self.value_heads as i32;
        let dk = self.key_head_dim as i32;
        let dv = self.value_head_dim as i32;
        let kc = self.conv_kernel_size as i32;
        let conv_dim = self.conv_dim as i32;
        if dk != 128 || dv != 128 || kc <= 1 || conv_dim != (2 * hk + hv) * 128 {
            return None;
        }
        // The gate is templated (sigmoid for Flash-Next, silu for qwen3_5).
        if self.proj_width != self.in_proj_all.dims_out() as i32 {
            return None;
        }
        let proj = self.in_proj_all.forward(x).ok()?;
        let (conv_state, state_in) = match cache.as_deref() {
            Some(c) => (c.conv.clone(), c.ssm.clone()),
            None => (None, None),
        };
        let conv_state = match conv_state {
            Some(c) => c,
            None => ops::zeros::<half::bf16>(&[b, kc - 1, conv_dim]).ok()?,
        };
        let state_in = match state_in {
            Some(t) => t,
            None => ops::zeros::<f32>(&[b, hv, dv, dk]).ok()?,
        };
        let neg_exp_alog = self.neg_exp_alog.clone();
        let conv_w = self.conv1d_weight.reshape(&[conv_dim, kc]).ok()?;
        let stream = lisa_mlx::Stream::thread_local_or_default();
        let (gated, state_out, conv_out) = lisa_mlx::kernels::gdn_decode_complete(
            &proj,
            &conv_state,
            &conv_w,
            &neg_exp_alog,
            &self.dt_bias,
            &state_in,
            &self.norm.weight,
            hk,
            hv,
            dk,
            dv,
            kc,
            conv_dim,
            self.proj_width,
            self.b_offset,
            self.a_offset,
            self.z_offset,
            self.norm.eps,
            self.output_gate_silu,
            &stream,
        )?;
        static ENGAGED: std::sync::Once = std::sync::Once::new();
        ENGAGED.call_once(|| {
            eprintln!(
                "[gdn] decode complete engaged (gate={})",
                if self.output_gate_silu {
                    "silu"
                } else {
                    "sigmoid"
                }
            );
        });
        if let Some(c) = cache.as_deref_mut() {
            c.conv = Some(conv_out);
            c.ssm = Some(state_out);
        }
        let gated = gated.reshape(&[b, s, hv * dv]).ok()?;
        self.out_proj.forward(&gated).ok()
    }
}

/// rmsNorm with no weight (the GDN internal l2norm form).
fn fast_rms_none(x: &Array, eps: f32) -> lisa_mlx::error::Result<Array> {
    lisa_mlx::fast::rms_norm(x, None, eps)
}
