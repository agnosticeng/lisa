//! Gated DeltaNet linear-attention layer (the 36 recurrent layers).

use lisa_mlx::ops::indexing::IndexOp;
use lisa_mlx::{ops, Array, Dtype};

use lisa_mlx::kernels::{gated_delta_kernel, gated_delta_ops, gated_delta_ops_capture, gdn_rows, gdn_two_row};
use crate::core::cache::GdnCache;
use crate::core::norm::RmsNormGated;
use crate::core::loader::TensorSource;
use crate::core::quant::QuantizedLinear;



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
    pub dt_bias: Array,       // bf16 [48]
    pub norm: RmsNormGated,
    pub out_proj: QuantizedLinear,
    pub value_heads: usize,
    pub key_heads: usize,
    pub key_head_dim: usize,
    pub value_head_dim: usize,
    pub conv_kernel_size: usize,
    pub conv_dim: usize,
}

impl GatedDeltaNet {
    pub fn load<S: TensorSource>(
        src: &mut S,
        prefix: &str,
        eps: f32,
        key_heads: usize,
        value_heads: usize,
        key_head_dim: usize,
        value_head_dim: usize,
        conv_kernel_size: usize,
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
        let in_proj_qkv = QuantizedLinear::load(src, prefix, "in_proj_qkv")?;
        let in_proj_z = QuantizedLinear::load(src, prefix, "in_proj_z")?;
        let in_proj_b = QuantizedLinear::load(src, prefix, "in_proj_b")?;
        let in_proj_a = QuantizedLinear::load(src, prefix, "in_proj_a")?;
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
        };
        let proj_width = in_proj_all.dims_out() as i32;
        let z_offset = in_proj_qkv.dims_out() as i32;
        let b_offset = z_offset + in_proj_z.dims_out() as i32;
        let a_offset = b_offset + in_proj_b.dims_out() as i32;
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
            out_proj: QuantizedLinear::load(src, prefix, "out_proj")?,
            value_heads,
            key_heads,
            key_head_dim,
            value_head_dim,
            conv_kernel_size,
            conv_dim,
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
        if !capture && s == 1 && !lisa_mlx::env_flag("LISA_NO_DECODE_COMPLETE") {
            if let Some(out) = self.forward_decode_complete(x, cache.as_deref_mut()) {
                return Ok(out);
            }
        }

        static FIRST_CALL: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);
        static CALLS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let dump_dir = std::env::var("LISA_DUMP_DIR").ok();
        let call = CALLS.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let want_call = std::env::var("LISA_DUMP_CALL")
            .ok()
            .and_then(|v| v.parse::<usize>().ok());
        let first = FIRST_CALL.swap(false, std::sync::atomic::Ordering::SeqCst) && dump_dir.is_some();
        let do_dump = dump_dir.is_some()
            && (first || want_call == Some(call) || std::env::var("LISA_DUMP_ALL").is_ok());
        let dump = |name: &str, arr: &Array| {
            if do_dump {
                let dir = dump_dir.as_ref().unwrap();
                let f = arr
                    .as_dtype(Dtype::Float32)
                    .map_err(|e| anyhow::anyhow!("{e}"))
                    .unwrap();
                let a = f.as_slice::<f32>();
                let file = if first {
                    format!("{dir}/rs_{name}.bin")
                } else {
                    format!("{dir}/c{call}_{name}.bin")
                };
                let _ = std::fs::write(
                    file,
                    unsafe { std::slice::from_raw_parts(a.as_ptr() as *const u8, a.len() * 4) },
                );
            }
        };
        dump("gdn_in", x);
        let _t0 = std::time::Instant::now();
        let _prof = std::env::var("LISA_PROFILE_GDN").is_ok();
        // The engine splits the projection into its four parts only for eligible
        // wide prefill windows (batch 1, S > 8, bf16, no capture). Every other
        // window -- the MTP verify/capture and small prefill -- uses the fused
        // `in_proj_all` and reads the gates from offsets in it.
        let separate = !capture && b == 1 && s > 8;
        let value_dim = (self.value_heads * self.value_head_dim) as i32;
        let (mixed_qkv, z, b_proj, a_proj, fused_proj): (Array, Array, Array, Array, Option<Array>);
        if separate {
            mixed_qkv = self.in_proj_qkv.forward(x)?;
            let z4 = self.in_proj_z.forward(x)?;
            b_proj = self.in_proj_b.forward(x)?;
            a_proj = self.in_proj_a.forward(x)?;
            z = z4.reshape(&[b, s, self.value_heads as i32, self.value_head_dim as i32])?;
            if _prof { let _ = (mixed_qkv.eval(), z.eval(), b_proj.eval(), a_proj.eval()); }
            fused_proj = None;
        } else {
            let proj = self.in_proj_all.forward(x)?;
            let z_off = self.z_offset;
            z = proj
                .index((.., .., z_off..(z_off + value_dim)))
                .reshape(&[b, s, self.value_heads as i32, self.value_head_dim as i32])?;
            mixed_qkv = proj.index((.., .., 0..self.conv_dim as i32)).contiguous()?;
            b_proj = Array::from_slice(&[0f32], &[1]);
            a_proj = Array::from_slice(&[0f32], &[1]);
            if _prof { let _ = (mixed_qkv.eval(), z.eval()); }
            fused_proj = Some(proj);
        }
        dump("qkv", &mixed_qkv);
        let _t_proj = _t0.elapsed();

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
            if lisa_mlx::env_flag("LISA_GDN_SHAPES") {
                eprintln!("[gdn] qkv={:?}/{:?} cs={:?}/{:?} cw={:?}/{:?} neg={:?}/{:?} dt={:?}/{:?} b={:?}/{:?} a={:?}/{:?} | s={s} hk={} hv={} dk={} dv={} kc={kc} conv_dim={conv_dim}",
                    mixed_qkv.shape(), mixed_qkv.dtype(), conv_state.shape(), conv_state.dtype(),
                    self.conv1d_weight.shape(), self.conv1d_weight.dtype(),
                    neg_exp_alog.shape(), neg_exp_alog.dtype(), self.dt_bias.shape(), self.dt_bias.dtype(),
                    b_proj.shape(), b_proj.dtype(), a_proj.shape(), a_proj.dtype(),
                    self.key_heads, self.value_heads, self.key_head_dim, self.value_head_dim);
            }
            let r = match &fused_proj {
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
                ),
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
                ),
            };
            let (qn, kn, vv, gg, bb, conv_out) = r.ok_or_else(|| {
                lisa_mlx::error::Exception::custom("gdn_prep kernel unavailable".to_string())
            })?;
            (q, k, v, g, beta, new_conv) = (qn, kn, vv, gg, bb, Some(conv_out));
            if capture {
                capture_conv_input = Some(ops::concatenate(&[&conv_state, &mixed_qkv], 1)?);
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
            dump("convout", &conv_out);
            let qq = conv_out.index((.., .., 0..key_dim));
            let kk = conv_out.index((.., .., key_dim..(2 * key_dim)));
            let vv = conv_out.index((.., .., (2 * key_dim)..));
            let qq = qq.reshape(&[b, s, self.key_heads as i32, self.key_head_dim as i32])?;
            let kk = kk.reshape(&[b, s, self.key_heads as i32, self.key_head_dim as i32])?;
            let vv = vv.reshape(&[b, s, self.value_heads as i32, self.value_head_dim as i32])?;
            let inv_scale = (self.key_head_dim as f32).powf(-0.5);
            let qq = fast_rms_none(&qq, 1e-6)? * crate::core::norm::bf16_scalar(inv_scale * inv_scale);
            let kk = fast_rms_none(&kk, 1e-6)? * crate::core::norm::bf16_scalar(inv_scale);
            let bb = ops::sigmoid(&b_proj)?.as_dtype(Dtype::Float32)?;
            let neg_exp_alog = self.neg_exp_alog.clone();
            let ax = a_proj.add(&self.dt_bias)?;
            let sp = crate::core::norm::bf16_logaddexp0(&ax)?;
            let sp = sp.as_dtype(Dtype::Float32)?;
            let gg = ops::exp(&(neg_exp_alog * sp))?;
            (q, k, v, g, beta, new_conv) = (qq, kk, vv, gg, bb, None);
        }
        if _prof { let _ = (q.eval(), k.eval(), v.eval(), g.eval(), beta.eval()); }
        let _t_prep = _t0.elapsed();
        if let Some(c) = cache.as_deref_mut() {
            if let Some(nc) = new_conv {
                c.conv = Some(nc);
            }
        }
        dump("q", &q);
        dump("beta", &beta);
        dump("g", &g);

        // The fp32 SSM state; zeros on the first call.
        let zeros_state = || {
            let b_usize = b as usize;
            Array::from_slice(
                &vec![0f32; b_usize * self.value_heads * self.value_head_dim * self.key_head_dim],
                &[b, self.value_heads as i32, self.value_head_dim as i32, self.key_head_dim as i32],
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
                        &q, &k, &v, &g, &beta, &state_in, capture,
                        py.as_ref().unwrap(), ps.as_ref().unwrap(), &stream,
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
        if _prof { let _ = out.eval(); }
        let _t_rec = _t0.elapsed();
        dump("gdn_out", &out);
        dump("gdn_state", &state);

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

        // Output gating: sigmoid(z) * rmsNorm(out) with a plain-scale weight.
        // Fused `track_gated_rms` (butterfly RMS + sigmoid in one launch).
        let normed = {
            let stream = lisa_mlx::Stream::thread_local_or_default();
            let z_flat = z.reshape(&[b, s, (self.value_heads * self.value_head_dim) as i32])?;
            let hv = self.value_heads as i32;
            let dv = self.value_head_dim as i32;
            match lisa_mlx::kernels::gated_rms(&out, &z_flat, &self.norm.weight, hv, dv, self.norm.eps, &stream) {
                Some(r) => r,
                None => self.norm.forward(&out, &z)?,
            }
        };
        dump("gated", &normed);
        if _prof { let _ = normed.eval(); }
        let _t_rms = _t0.elapsed();
        let normed = normed.reshape(&[b, s, -1])?;
        let attended = self.out_proj.forward(&normed)?;
        dump("attended", &attended);
        if _prof {
            let _t_out = _t0.elapsed();
            eprintln!("[gdn-prof] s={s} cap={capture} proj {:.3} prep {:.3} rec {:.3} rms {:.3} out {:.3} ms",
                _t_proj.as_secs_f64()*1e3, (_t_prep-_t_proj).as_secs_f64()*1e3,
                (_t_rec-_t_prep).as_secs_f64()*1e3, (_t_rms-_t_rec).as_secs_f64()*1e3,
                (_t_out-_t_rms).as_secs_f64()*1e3);
        }
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
        static DEC_CALLS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let dec_i = DEC_CALLS.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let dec_want = std::env::var("LISA_DEC_CALL").ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(4);
        let dec_do = dec_i == dec_want;
        let dec_dump = |name: &str, arr: &Array| {
            if !dec_do { return; }
            if let Ok(dir) = std::env::var("LISA_DUMP_DIR") {
                if let Ok(f) = arr.as_dtype(Dtype::Float32) {
                    let a = f.as_slice::<f32>();
                    let _ = std::fs::write(
                        format!("{dir}/dec{dec_i}_{name}.bin"),
                        unsafe { std::slice::from_raw_parts(a.as_ptr() as *const u8, a.len() * 4) },
                    );
                }
            }
        };
        dec_dump("proj", &proj);
        dec_dump("conv_state", &conv_state);
        dec_dump("state_in", &state_in);
        dec_dump("a_log_raw", &self.a_log);
        dec_dump("neg_exp_alog", &neg_exp_alog);
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
            &stream,
        )?;
        dec_dump("gated_raw", &gated);
        dec_dump("conv_out", &conv_out);
        dec_dump("state_out", &state_out);
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
