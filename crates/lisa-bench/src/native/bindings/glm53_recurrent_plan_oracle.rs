#![cfg(test)]
//! KDA (`linear_attention`) layers for GLM-5.3, following the reference's float graph.
//!
//! Kimi Delta Attention: depthwise
//! causal short conv (k=4) over the fused q|k|v stream, l2-normalized q/k
//! (q scaled `head_dim^-0.5`), per-token decay `g = lower_bound * sigmoid(
//! lower_bound * sigmoid(exp(A_log) * (f_b(f_a(x)) + dt_bias)))`, beta
//! `sigmoid(b_proj(x))`, and the
//! delta-rule recurrence on an F32 state `[B, 64, 128, 128]`.
//!
//! The normal single-stream path uses the reference prepare, recurrence, and output
//! kernels. The Array-op path remains an explicit diagnostic fallback.

use crate::bindings::glm53_oracle_helpers::{concat, concat_refs, fscalar, narrow};
use crate::bindings::glm53_projection::Projection;
use crate::core::cache::GdnCache;
use crate::core::capture;
use lisa_metal::kernels::sparse::kernels as kda_kernels;
use lisa_metal::{Array, Dtype, ops};

pub struct Kda {
    pub(super) q_w: Projection, // Plain model weight in source dtype, row-major [out, in]
    pub(super) k_w: Projection,
    pub(super) v_w: Projection,
    pub(super) o_proj: Projection, // Plain KDA projection in source dtype, row-major [out, in]
    pub(super) f_a_w: Projection,  // [4096, 128]
    pub(super) f_b_w: Projection,  // [128, 8192]
    pub(super) g_a_w: Projection,  // [4096, 128]
    pub(super) g_b_w: Projection,  // [128, 8192]
    pub(super) b_w: Projection,    // [4096, 64]
    pub(super) conv_q: Array,      // F32 [channels, 4], taps flattened from [channels, 1, 4]
    pub(super) conv_k: Array,
    pub(super) conv_v: Array,
    pub(super) dt_bias: Array,  // [64, 128] f32
    pub(super) a_log: Array,    // [64] f32
    pub(super) o_norm_w: Array, // [128] f32
    pub(super) heads: usize,
    pub(super) head_dim: usize,
    pub(super) conv_k_sz: usize,
    pub(super) lower_bound: f32,
    pub(super) norm_eps: f32,
}

pub use crate::bindings::glm53_math::gated_rms_norm;
use crate::bindings::glm53_math::l2norm_last;

impl Kda {
    fn linear(&self, x: &Array, weight: &Projection) -> anyhow::Result<Array> {
        crate::bindings::glm53_kda_projection::linear(weight, x)
    }
    fn gate_projections(&self, fa: &Array, ga: &Array) -> anyhow::Result<(Array, Array)> {
        crate::bindings::glm53_kda_projection::gates([&self.f_b_w, &self.g_b_w], fa, ga)
    }
    fn input_projections(
        &self,
        x: &Array,
    ) -> anyhow::Result<(Array, Array, Array, Array, Array, Array)> {
        crate::bindings::glm53_kda_projection::inputs(
            [
                &self.q_w,
                &self.k_w,
                &self.v_w,
                &self.f_a_w,
                &self.g_a_w,
                &self.b_w,
            ],
            x,
            crate::core::quant::glm53_kda_nsg_override(),
        )
    }

    /// Decode a cohort whose KDA recurrent states belong to separate
    /// sessions. The projections and output projection run over the complete
    /// row batch, while the fused recurrence kernel advances each row against
    /// its own convolution and SSM state, matching the reference's native batch path.
    pub fn forward_ragged(
        &self,
        x: &Array,
        caches: &mut [&mut GdnCache],
        cap: Option<&capture::Capture>,
        layer: usize,
    ) -> anyhow::Result<Array> {
        let b = caches.len();
        anyhow::ensure!(
            b > 1 && x.dim(0) == b as i32 && x.dim(1) == 1,
            "glm53 KDA ragged decode expects B>1, one token per row, got {:?} for {b} caches",
            x.shape()
        );
        let d = self.head_dim;
        let heads = self.heads;
        let qkv = heads * d;
        let conv_dim = 3 * qkv;
        let mut conv_rows = Vec::with_capacity(b);
        let mut state_rows = Vec::with_capacity(b);
        for cache in caches.iter() {
            conv_rows.push(match &cache.conv {
                Some(state) => state
                    .as_dtype(Dtype::Float32)?
                    .reshape(&[1, 3, conv_dim as i32])?
                    .contiguous()?,
                None => ops::zeros::<f32>(&[1, 3, conv_dim as i32])?,
            });
            state_rows.push(match &cache.ssm {
                Some(state) => state
                    .as_dtype(Dtype::Float32)?
                    .reshape(&[1, heads as i32, d as i32, d as i32])?
                    .contiguous()?,
                None => ops::zeros::<f32>(&[1, heads as i32, d as i32, d as i32])?,
            });
        }
        let mut conv = Some(concat(conv_rows, 0)?);
        let mut state = Some(concat(state_rows, 0)?);
        let out = self.forward(x, &mut conv, &mut state, cap, layer)?;
        let conv = conv.ok_or_else(|| anyhow::anyhow!("glm53 KDA lost its batched conv state"))?;
        let state = state.ok_or_else(|| anyhow::anyhow!("glm53 KDA lost its batched SSM state"))?;
        for (row, cache) in caches.iter_mut().enumerate() {
            cache.conv = Some(
                narrow(&conv, 0, row, 1)?
                    .reshape(&[1, 3, conv_dim as i32])?
                    .contiguous()?,
            );
            cache.ssm = Some(
                narrow(&state, 0, row, 1)?
                    .reshape(&[1, heads as i32, d as i32, d as i32])?
                    .contiguous()?,
            );
        }
        Ok(out)
    }

    /// Forward `x [B,T,H]` with the layer cache (`GdnCache`: conv window +
    /// F32 recurrence state). Returns the block output `[B,T,H]`.
    pub fn forward(
        &self,
        x: &Array,
        conv_state: &mut Option<Array>,
        ssm_state: &mut Option<Array>,
        cap: Option<&capture::Capture>,
        layer: usize,
    ) -> anyhow::Result<Array> {
        self.forward_with_mode(x, conv_state, ssm_state, cap, layer, false, None)
    }

    /// Forward a prompt chunk using the reference's prefill path, including a one-token
    /// continuation chunk after the 2048-token prefill boundary.
    pub fn forward_prefill(
        &self,
        x: &Array,
        conv_state: &mut Option<Array>,
        ssm_state: &mut Option<Array>,
        cap: Option<&capture::Capture>,
        layer: usize,
    ) -> anyhow::Result<Array> {
        self.forward_with_mode(x, conv_state, ssm_state, cap, layer, true, None)
    }

    fn forward_with_mode(
        &self,
        x: &Array,
        conv_state: &mut Option<Array>,
        ssm_state: &mut Option<Array>,
        cap: Option<&capture::Capture>,
        layer: usize,
        prefill_mode: bool,
        _position: Option<usize>,
    ) -> anyhow::Result<Array> {
        let (b, t, _h) = (x.dim(0) as usize, x.dim(1) as usize, x.dim(2) as usize);
        let heads = self.heads;
        let d = self.head_dim;
        let qkv = heads * d;
        let conv_dim = qkv * 3;
        // The reference keeps prompt chunks on its prefill graph, including a one-token
        // tail chunk after the 2048-token boundary. Generation forwards with
        // prior recurrent state use the fused decode path.
        let has_prior_recurrent_state = ssm_state.is_some();

        // 1. Input projections (six linears on the same input).
        if let Some(c) = cap.filter(|capture| capture.wants(layer)) {
            c.store(&format!("L{layer}.kda_input"), x);
        }
        let _trace_projections =
            lisa_metal::trace::span_detail("glm53.kda.input_projections", layer as u64);
        let (qo, ko, vo, fa, ga, bo) = self.input_projections(x)?;
        let (gate_raw, output_gate) = self.gate_projections(&fa, &ga)?;
        // `gate_raw` is f_b(f_a(x)); `output_gate` is g_b(g_a(x)).
        drop(_trace_projections);
        if let Some(c) = cap.filter(|capture| capture.wants(layer)) {
            c.store(&format!("L{layer}.kda_q_proj"), &qo);
            c.store(&format!("L{layer}.kda_k_proj"), &ko);
            c.store(&format!("L{layer}.kda_v_proj"), &vo);
            c.store(&format!("L{layer}.kda_fa_proj"), &fa);
            c.store(&format!("L{layer}.kda_ga_proj"), &ga);
            c.store(&format!("L{layer}.kda_beta_logits"), &bo);
            c.store(&format!("L{layer}.kda_gate_raw"), &gate_raw);
            c.store(&format!("L{layer}.kda_gb_proj"), &output_gate);
        }
        let stream = lisa_metal::Stream::thread_local_or_default();
        let state = match ssm_state {
            Some(a) => a.clone(),
            None => ops::full::<i32>(
                &[(b) as i32, (heads) as i32, (d) as i32, (d) as i32],
                &Array::from_f32_as(0.0, Dtype::Float32),
            )?,
        };
        // 2. Fused path (B = 1): the reference prepare kernel (conv + silu + l2norm +
        // compute_g_safe) -> recurrence kernel -> gated-output kernel.
        // prepare/output kernels are unit-tested at real geometry
        // (kda_prepare_matches_reference, kda_prepare_two_calls).
        if b > 1 && t == 1 && !prefill_mode {
            // The reference's native batch path shares the row projections across the
            // cohort, then dispatches its one-token decode kernel on each
            // row's independent convolution and recurrent state. Keep these
            // calls row-local: the fused decode's reduction schedule is the
            // same as serial decode, and the batched prepare/recurrence graph
            // has measurably different layer outputs.
            let qo_f = qo.as_dtype(Dtype::Float32)?.contiguous()?;
            let ko_f = ko.as_dtype(Dtype::Float32)?.contiguous()?;
            let vo_f = vo.as_dtype(Dtype::Float32)?.contiguous()?;
            let gate_f = gate_raw.as_dtype(Dtype::Float32)?.contiguous()?;
            let raw_beta_f = bo.as_dtype(Dtype::Float32)?.contiguous()?;
            let output_gate_f = output_gate.as_dtype(Dtype::Float32)?.contiguous()?;
            let conv_in = match conv_state {
                Some(a) => a
                    .as_dtype(Dtype::Float32)?
                    .reshape(&[b as i32, 3, (3 * qkv) as i32])?
                    .contiguous()?,
                None => ops::full::<i32>(
                    &[b as i32, 3, (3 * qkv) as i32],
                    &Array::from_f32_as(0.0, Dtype::Float32),
                )?,
            };
            let state4 = state
                .reshape(&[b as i32, heads as i32, d as i32, d as i32])?
                .contiguous()?;
            if let Some(c) = cap.filter(|capture| capture.wants(layer)) {
                c.store(&format!("L{layer}.kda_initial_conv"), &conv_in);
                c.store(&format!("L{layer}.kda_initial_ssm"), &state4);
                c.store(&format!("L{layer}.kda_q_conv_weight"), &self.conv_q);
                c.store(&format!("L{layer}.kda_k_conv_weight"), &self.conv_k);
                c.store(&format!("L{layer}.kda_v_conv_weight"), &self.conv_v);
                c.store(&format!("L{layer}.kda_a_log"), &self.a_log);
                c.store(&format!("L{layer}.kda_dt_bias"), &self.dt_bias);
                c.store(&format!("L{layer}.kda_output_norm_weight"), &self.o_norm_w);
                c.store(
                    &format!("L{layer}.kda_meta"),
                    &Array::from_slice(
                        &[
                            self.lower_bound,
                            self.norm_eps,
                            self.heads as f32,
                            self.head_dim as f32,
                            self.conv_k_sz as f32,
                        ],
                        &[5],
                    ),
                );
            }
            let q_conv_f = self.conv_q.as_dtype(Dtype::Float32)?.contiguous()?;
            let k_conv_f = self.conv_k.as_dtype(Dtype::Float32)?.contiguous()?;
            let v_conv_f = self.conv_v.as_dtype(Dtype::Float32)?.contiguous()?;
            let a_log_f = self.a_log.as_dtype(Dtype::Float32)?.contiguous()?;
            let dt_bias_f = self.dt_bias.as_dtype(Dtype::Float32)?.contiguous()?;
            let output_norm_f = self.o_norm_w.as_dtype(Dtype::Float32)?.contiguous()?;
            let mut out_rows = Vec::with_capacity(b);
            let mut conv_rows = Vec::with_capacity(b);
            let mut state_rows = Vec::with_capacity(b);
            for row in 0..b {
                let row = row as i32;
                let q_row = narrow(&qo_f, 0, row as usize, 1)?.reshape(&[qkv as i32])?;
                let k_row = narrow(&ko_f, 0, row as usize, 1)?.reshape(&[qkv as i32])?;
                let v_row = narrow(&vo_f, 0, row as usize, 1)?.reshape(&[qkv as i32])?;
                let gate_row = narrow(&gate_f, 0, row as usize, 1)?.reshape(&[qkv as i32])?;
                let beta_row = narrow(&raw_beta_f, 0, row as usize, 1)?.reshape(&[heads as i32])?;
                let output_gate_row =
                    narrow(&output_gate_f, 0, row as usize, 1)?.reshape(&[qkv as i32])?;
                let conv_row = narrow(&conv_in, 0, row as usize, 1)?
                    .reshape(&[3, heads as i32, (3 * d) as i32])?
                    .contiguous()?;
                let state_row = narrow(&state4, 0, row as usize, 1)?
                    .reshape(&[heads as i32, d as i32, d as i32])?
                    .contiguous()?;
                let (out, next_conv, next_state) = {
                    let _trace =
                        lisa_metal::trace::span_detail("glm53.kda.decode_kernel", layer as u64);
                    kda_kernels::kda_decode(
                        &q_row,
                        &k_row,
                        &v_row,
                        &gate_row,
                        &beta_row,
                        &output_gate_row,
                        &q_conv_f,
                        &k_conv_f,
                        &v_conv_f,
                        &a_log_f,
                        &dt_bias_f,
                        &output_norm_f,
                        &conv_row,
                        &state_row,
                        heads,
                        self.lower_bound,
                        self.norm_eps,
                        &stream,
                    )
                    .ok_or_else(|| {
                        anyhow::anyhow!(
                            "glm53 kda: ds4 row decode kernel declined at layer {layer}, row {row}"
                        )
                    })?
                };
                out_rows.push(out.reshape(&[1, 1, qkv as i32])?);
                conv_rows.push(next_conv.reshape(&[1, 3, conv_dim as i32])?);
                state_rows.push(next_state.reshape(&[1, heads as i32, d as i32, d as i32])?);
            }
            let out = concat(out_rows, 0)?;
            let next_conv = concat(conv_rows, 0)?;
            let next_state = concat(state_rows, 0)?;
            if let Some(c) = cap.filter(|capture| capture.wants(layer)) {
                c.store(&format!("L{layer}.kda_decode_out"), &out);
                c.store(&format!("L{layer}.kda_decode_conv_state"), &next_conv);
                c.store(&format!("L{layer}.kda_decode_state"), &next_state);
            }
            *conv_state = Some(next_conv);
            *ssm_state = Some(next_state.contiguous()?);
            let _trace =
                lisa_metal::trace::span_detail("glm53.kda.output_projection", layer as u64);
            let projected = self.linear(&out, &self.o_proj)?;
            if let Some(c) = cap.filter(|capture| capture.wants(layer)) {
                c.store(&format!("L{layer}.kda_o_proj"), &projected);
            }
            return Ok(projected);
        }
        if b == 1 {
            // The kernel binds raw GPU buffers — materialise the lazy casts
            // first (from_slice is eager; as_dtype/mk outputs are not).
            let qo_f = qo.as_dtype(Dtype::Float32)?.contiguous()?;
            let ko_f = ko.as_dtype(Dtype::Float32)?.contiguous()?;
            let vo_f = vo.as_dtype(Dtype::Float32)?.contiguous()?;
            let gate_f = gate_raw.as_dtype(Dtype::Float32)?.contiguous()?;
            // The reference-style: ops dispatch eagerly into the current command buffer
            // (Serial order) — no sync needed before binding raw buffers.
            let conv_in = match conv_state {
                Some(a) => {
                    a.as_dtype(Dtype::Float32)?
                        .reshape(&[3, heads as i32, (d * 3) as i32])?
                }
                None => ops::full::<i32>(
                    &[3, heads as i32, (d * 3) as i32],
                    &Array::from_f32_as(0.0, Dtype::Float32),
                )?,
            };
            let state4 = if state.rank() == 5 {
                state.reshape(&[b as i32, heads as i32, d as i32, d as i32])?
            } else {
                state.clone()
            };
            let state_pn = state4.contiguous()?;
            if let Some(c) = cap.filter(|capture| capture.wants(layer)) {
                c.store(&format!("L{layer}.kda_initial_conv"), &conv_in);
                c.store(&format!("L{layer}.kda_initial_ssm"), &state_pn);
                c.store(&format!("L{layer}.kda_q_conv_weight"), &self.conv_q);
                c.store(&format!("L{layer}.kda_k_conv_weight"), &self.conv_k);
                c.store(&format!("L{layer}.kda_v_conv_weight"), &self.conv_v);
                c.store(&format!("L{layer}.kda_a_log"), &self.a_log);
                c.store(&format!("L{layer}.kda_dt_bias"), &self.dt_bias);
                c.store(&format!("L{layer}.kda_output_norm_weight"), &self.o_norm_w);
                c.store(
                    &format!("L{layer}.kda_meta"),
                    &Array::from_slice(
                        &[
                            self.lower_bound,
                            self.norm_eps,
                            self.heads as f32,
                            self.head_dim as f32,
                            self.conv_k_sz as f32,
                        ],
                        &[5],
                    ),
                );
            }
            if t == 1 && has_prior_recurrent_state && !prefill_mode {
                let q_conv_f = self.conv_q.as_dtype(Dtype::Float32)?.contiguous()?;
                let k_conv_f = self.conv_k.as_dtype(Dtype::Float32)?.contiguous()?;
                let v_conv_f = self.conv_v.as_dtype(Dtype::Float32)?.contiguous()?;
                let a_log_f = self.a_log.as_dtype(Dtype::Float32)?.contiguous()?;
                let dt_bias_f = self.dt_bias.as_dtype(Dtype::Float32)?.contiguous()?;
                let output_norm_f = self.o_norm_w.as_dtype(Dtype::Float32)?.contiguous()?;
                let raw_beta_f = bo.as_dtype(Dtype::Float32)?.contiguous()?;
                let output_gate_f = output_gate.as_dtype(Dtype::Float32)?.contiguous()?;
                let decoded = {
                    let _trace =
                        lisa_metal::trace::span_detail("glm53.kda.decode_kernel", layer as u64);
                    kda_kernels::kda_decode(
                        &qo_f,
                        &ko_f,
                        &vo_f,
                        &gate_f,
                        &raw_beta_f,
                        &output_gate_f,
                        &q_conv_f,
                        &k_conv_f,
                        &v_conv_f,
                        &a_log_f,
                        &dt_bias_f,
                        &output_norm_f,
                        &conv_in,
                        &state_pn,
                        heads,
                        self.lower_bound,
                        self.norm_eps,
                        &stream,
                    )
                };
                if let Some((out, new_conv, new_state)) = decoded {
                    *conv_state = Some(new_conv.reshape(&[1, 3, conv_dim as i32])?);
                    *ssm_state = Some(new_state.reshape(&[1, heads as i32, d as i32, d as i32])?);
                    if let Some(c) = cap
                        && c.wants(layer)
                    {
                        c.store(&format!("L{layer}.kda_decode_out"), &out);
                        c.store(&format!("L{layer}.kda_decode_conv_state"), &new_conv);
                        c.store(&format!("L{layer}.kda_decode_state"), &new_state);
                    }
                    let out = out.reshape(&[1, 1, qkv as i32])?;
                    let _trace =
                        lisa_metal::trace::span_detail("glm53.kda.output_projection", layer as u64);
                    let projected = self.linear(&out, &self.o_proj)?;
                    return Ok(projected);
                }
                anyhow::bail!(
                    "glm53 kda: ds4-parity fused decode kernel declined at layer {layer}"
                );
            }
            if let Some((qf, kf, vf, g, new_conv)) = kda_kernels::kda_prepare(
                &qo_f,
                &ko_f,
                &vo_f,
                &gate_f,
                &self.conv_q.as_dtype(Dtype::Float32)?,
                &self.conv_k.as_dtype(Dtype::Float32)?,
                &self.conv_v.as_dtype(Dtype::Float32)?,
                &self.a_log.as_dtype(Dtype::Float32)?,
                &self
                    .dt_bias
                    .as_dtype(Dtype::Float32)?
                    .reshape(&[(heads * d) as i32])?,
                &conv_in,
                heads,
                self.lower_bound,
                self.norm_eps,
                &stream,
            ) {
                // The kernels bind raw GPU buffers. Eager dispatch (Serial
                // queue order) already materialises the casts before the
                // kernel's dispatch in the same encoder — no wait required.
                // The former 5 × eval() here drained the ENTIRE pipeline
                // per KDA layer (33 layers × 5 syncs/token): 145 s/token
                // decode, GPU 99% idle.
                for o in [&qf, &kf, &vf, &g, &new_conv] {
                    let _ = o.eval_enqueue();
                }
                *conv_state = Some(
                    new_conv
                        .reshape(&[1i32, (self.conv_k_sz - 1) as i32, (conv_dim) as i32])?
                        .as_dtype(Dtype::Float32)?,
                );
                if let Some(c) = cap
                    && c.wants(layer)
                {
                    c.store(&format!("L{layer}.kda_qc"), &qf);
                    c.store(&format!("L{layer}.kda_kc"), &kf);
                    c.store(&format!("L{layer}.kda_vc"), &vf);
                    c.store(&format!("L{layer}.kda_g"), &g);
                    let beta = bo.as_dtype(Dtype::Float32)?.sigmoid()?.reshape(&[
                        b as i32,
                        t as i32,
                        heads as i32,
                        1,
                    ])?;
                    c.store(&format!("L{layer}.kda_beta"), &beta);
                    c.store(&format!("L{layer}.kda_conv_state"), &new_conv);
                }
                // The cache may hold a rank-5 [B,1,H,Dk,Dv] state (written by
                // the naive loop) or a rank-4 [B,H,Dk,Dv] one — normalize.
                let qf4 = qf.reshape(&[(b) as i32, (t) as i32, (heads) as i32, (d) as i32])?;
                let kf4 = kf.reshape(&[(b) as i32, (t) as i32, (heads) as i32, (d) as i32])?;
                let vf4 = vf.reshape(&[(b) as i32, (t) as i32, (heads) as i32, (d) as i32])?;
                let g4 = g.reshape(&[(b) as i32, (t) as i32, (heads) as i32, (d) as i32])?;
                if let Some((y_k, state_k)) = kda_kernels::kda_recurrence(
                    &qf4,
                    &kf4,
                    &vf4,
                    &g4,
                    &bo.as_dtype(Dtype::Float32)?.reshape(&[
                        (b) as i32,
                        (t) as i32,
                        (heads) as i32,
                    ])?,
                    &state_pn,
                    false,
                    &stream,
                ) {
                    if let Some(c) = cap
                        && c.wants(layer)
                    {
                        let _ = y_k.eval_enqueue();
                        let _ = state_k.eval_enqueue();
                        c.store(&format!("L{layer}.fused_y"), &y_k);
                        c.store(&format!("L{layer}.fused_state"), &state_k);
                    }
                    // Kernel outputs wrap real GPU buffers — no sync needed.
                    *ssm_state = Some(state_k.contiguous()?);
                    let gate = output_gate.clone();
                    let gate =
                        gate.reshape(&[(b) as i32, (t) as i32, (heads) as i32, (d) as i32])?;
                    if let Some(outn) = kda_kernels::kda_output_norm(
                        &y_k.reshape(&[(t) as i32, (qkv) as i32])?,
                        &gate.reshape(&[(t) as i32, (qkv) as i32])?,
                        &self.o_norm_w.as_dtype(Dtype::Float32)?,
                        heads,
                        self.norm_eps,
                        &stream,
                    ) {
                        if let Some(c) = cap.filter(|capture| capture.wants(layer)) {
                            c.store(&format!("L{layer}.kda_outnorm"), &outn);
                        }
                        if t == 1
                            && let Some(c) = cap
                            && c.wants(layer)
                        {
                            let _ = outn.eval_enqueue();
                            let _ = y_k.eval_enqueue();
                            c.store(&format!("L{layer}.dec_outnorm"), &outn);
                            c.store(
                                &format!(
                                    "L{layer}.dec_y_s{}",
                                    crate::bindings::glm53_affine_tower::INC_STEP
                                        .load(std::sync::atomic::Ordering::Relaxed)
                                ),
                                &y_k,
                            );
                        }
                        let out = outn.reshape(&[(b) as i32, (t) as i32, (qkv) as i32])?;
                        let projected = self.linear(&out, &self.o_proj)?;
                        return Ok(projected);
                    }
                }
            }
        }

        if b == 1 {
            anyhow::bail!(
                "glm53 kda: fused GPU prepare/recurrence/output path declined at layer {layer}"
            );
        }

        // The fused recurrence consumes raw beta logits and computes the reference's
        // sigmoid expression in-kernel. Only the diagnostic fallback needs a
        // materialized beta tensor.
        let beta = bo.as_dtype(Dtype::Float32)?.sigmoid()?.reshape(&[
            b as i32,
            t as i32,
            heads as i32,
            1,
        ])?;

        // 3. Naive fallback: conv + silu per branch.
        let prefix_len = self.conv_k_sz - 1;
        let window = match conv_state {
            Some(a) => {
                let planar = a.reshape(&[(b) as i32, 3, (3 * qkv) as i32])?;
                let qh = narrow(&planar, 2, 0, qkv)?;
                let kh = narrow(&planar, 2, qkv, qkv)?;
                let vh = narrow(&planar, 2, 2 * qkv, qkv)?;
                concat(vec![qh, kh, vh], 2)?
            }
            None => ops::full::<i32>(
                &[(b) as i32, (prefix_len) as i32, (conv_dim) as i32],
                &Array::from_f32_as(0.0, Dtype::Float32),
            )?,
        };
        // Only the naive convolution fallback needs a joined Q/K/V tensor.
        // The reference-shaped Metal prepare/decode kernels bind the three rows
        // separately; building this concat before those fast paths returned
        // added three GPU copy dispatches per KDA layer on ordinary decode.
        let mixed = concat(vec![qo, ko, vo], 2)?; // [B,T,3*qkv]
        let inp = concat_refs(&[&window, &mixed], 1)?; // [B, K-1+T, convDim]
        let new_window = narrow(&inp, 1, t, prefix_len)?.contiguous()?;
        let new_q = narrow(&new_window, 2, 0, qkv)?;
        let new_k = narrow(&new_window, 2, qkv, qkv)?;
        let new_v = narrow(&new_window, 2, 2 * qkv, qkv)?;
        *conv_state =
            Some(concat(vec![new_q, new_k, new_v], 1)?.reshape(&[b as i32, 3, conv_dim as i32])?);

        let conv_slice = |taps: &Array, off: usize| -> anyhow::Result<Array> {
            let mut acc: Option<Array> = None;
            for k in 0..self.conv_k_sz {
                let tap = narrow(taps, 1, k, 1)?.reshape(&[1, 1, qkv as i32])?;
                let col = narrow(&narrow(&inp, 1, k, t)?, 2, off, qkv)?;
                let term =
                    col.multiply(&tap.broadcast_to(&[(b) as i32, (t) as i32, (qkv) as i32])?)?;
                acc = Some(match acc {
                    Some(a) => a.add(&term)?,
                    None => term,
                });
            }
            Ok(acc.expect("taps").silu()?)
        };
        let qc = conv_slice(&self.conv_q, 0)?;
        let kc = conv_slice(&self.conv_k, qkv)?;
        let vc = conv_slice(&self.conv_v, 2 * qkv)?;

        let q4 = qc.reshape(&[(b) as i32, (t) as i32, (heads) as i32, (d) as i32])?;
        let k4 = kc.reshape(&[(b) as i32, (t) as i32, (heads) as i32, (d) as i32])?;
        let v4 = vc.reshape(&[(b) as i32, (t) as i32, (heads) as i32, (d) as i32])?;
        let qf = l2norm_last(&q4.as_dtype(Dtype::Float32)?.contiguous()?, 1e-6)?;
        let kf = l2norm_last(&k4.as_dtype(Dtype::Float32)?.contiguous()?, 1e-6)?;
        let scale = (d as f64).powf(-0.5) as f32;
        let qf = qf.multiply(&fscalar(&qf, scale)?)?;
        let vf = v4.as_dtype(Dtype::Float32)?.contiguous()?;

        // 4. compute_g_safe on the naive path too.
        let fg = self
            .linear(&fa, &self.f_b_w)?
            .reshape(&[(b) as i32, (t) as i32, (heads) as i32, (d) as i32])?
            .as_dtype(Dtype::Float32)?;
        let dtb =
            self.dt_bias
                .as_dtype(Dtype::Float32)?
                .reshape(&[1, 1, heads as i32, d as i32])?;
        let fgdt =
            fg.add(&dtb.broadcast_to(&[(b) as i32, (t) as i32, (heads) as i32, (d) as i32])?)?;
        let decay =
            self.a_log
                .as_dtype(Dtype::Float32)?
                .exp()?
                .reshape(&[1, 1, heads as i32, 1])?;
        let g = fgdt
            .multiply(&decay.broadcast_to(&[
                (b) as i32,
                (t) as i32,
                (heads) as i32,
                (d) as i32,
            ])?)?
            .sigmoid()?
            .multiply(&fscalar(&fgdt, self.lower_bound)?)?
            .exp()?;
        if let Some(c) = cap
            && c.wants(layer)
        {
            c.store(&format!("L{layer}.kda_g"), &g);
            c.store(&format!("L{layer}.kda_beta"), &beta);
        }

        // 5. Delta-rule recurrence (per token), F32 state [B,Hh,Dk,Dv].
        let mut state = match ssm_state {
            Some(a) => {
                let old = if a.rank() == 5 {
                    a.reshape(&[b as i32, heads as i32, d as i32, d as i32])?
                } else {
                    a.clone()
                };
                old.transpose_axes(&[0, 1, 3, 2])?
            }
            None => ops::full::<i32>(
                &[(b) as i32, (heads) as i32, (d) as i32, (d) as i32],
                &Array::from_f32_as(0.0, Dtype::Float32),
            )?,
        };
        let mut outs: Vec<Array> = Vec::with_capacity(t);
        if t == 1
            && let Some(c) = cap
            && c.wants(layer)
        {
            let _ = qf.eval_enqueue();
            let _ = g.eval_enqueue();
            let _ = state.eval_enqueue();
            let step = crate::bindings::glm53_affine_tower::INC_STEP
                .load(std::sync::atomic::Ordering::Relaxed);
            c.store(&format!("L{layer}.dec_q_s{step}"), &qf);
            c.store(&format!("L{layer}.dec_g_s{step}"), &g);
            c.store(&format!("L{layer}.dec_state_in_s{step}"), &state);
            if layer == 0 {
                let wv = window.to_vec1::<f32>().unwrap_or_default();
                eprintln!(
                    "[dec-conv] naive window[0..16] = {:?}",
                    &wv[..16.min(wv.len())]
                );
                c.store(&format!("L{layer}.dec_conv_s{step}"), &window);
            }
        }
        for ti in 0..t {
            let g_t = narrow(&g, 1, ti, 1)?.contiguous()?; // [B,1,Hh,Dk]
            // g is already exponentiated (compute_g_safe) — no second exp()
            // state *= exp(g): decay on the Dk axis — 3-D reshape (no rank-5
            // broadcast through the wrapper).
            let bh = (b * heads) as i32;
            let state3 = state.reshape(&[bh, d as i32, d as i32])?; // [B*Hh, Dk, Dv]
            let g3 = g_t.reshape(&[bh, d as i32, 1])?; // [B*Hh, Dk, 1]
            let state3 = state3.multiply(&g3)?;
            state = state3.reshape(&[(b) as i32, 1, (heads) as i32, (d) as i32, (d) as i32])?;
            let k_t = narrow(&kf, 1, ti, 1)?.contiguous()?; // [B,1,Hh,Dk]
            let v_t = narrow(&vf, 1, ti, 1)?.contiguous()?; // [B,1,Hh,Dv]
            let q_t = narrow(&qf, 1, ti, 1)?.contiguous()?; // [B,1,Hh,Dk]
            let b_t = narrow(&beta, 1, ti, 1)?.contiguous()?; // [B,1,Hh,1]
            // kv_mem[dv] = Σ_dk state[dk, dv] * k[dk] — reduce on the LAST
            // axis (the wrapper's middle-axis reduce is unreliable on rank-5
            // views): transpose Dk to last, flatten rows, 2-D matmul.
            let bh = (b * heads) as i32;
            let st_t = state
                .transpose_axes(&[0, 1, 2, 4, 3])? // [B,1,Hh,Dv,Dk]
                .contiguous()?
                .reshape(&[bh, d as i32, d as i32])?; // [B*Hh, Dv, Dk]
            let kt2 = k_t.reshape(&[bh, d as i32, 1])?; // [B*Hh, Dk, 1]
            let kv2 = st_t.matmul(&kt2)?; // [B*Hh, Dv, 1]
            let kv_mem = kv2.reshape(&[(b) as i32, 1, (heads) as i32, (d) as i32])?;
            let delta = v_t.subtract(&kv_mem)?.multiply(&b_t)?; // [B,1,Hh,Dv]
            // state += k (outer) delta
            let upd = k_t
                .expand_dims(4)?
                .multiply(&delta.expand_dims(3)?)?
                .reshape(&[(b) as i32, 1, (heads) as i32, (d) as i32, (d) as i32])?;
            state = state.add(&upd)?;
            // out[dv] = Σ_dk state[dk, dv] * q[dk] — same 2-D matmul trick.
            let st_t2 = state
                .transpose_axes(&[0, 1, 2, 4, 3])?
                .contiguous()?
                .reshape(&[bh, d as i32, d as i32])?; // [B*Hh, Dv, Dk]
            let qt2 = q_t.reshape(&[bh, d as i32, 1])?; // [B*Hh, Dk, 1]
            let out_t = st_t2.matmul(&qt2)?; // [B*Hh, Dv, 1]
            let out_r = out_t.reshape(&[(b) as i32, 1, (heads) as i32, (d) as i32])?;
            if let Some(c) = cap
                && c.wants(layer)
                && ti < 2
            {
                c.store(&format!("L{layer}.kda_tok{ti}"), &out_r);
                if ti == 0 {
                    c.store(&format!("L{layer}.kda_state_t0"), &state);
                }
            }
            outs.push(out_r);
        }
        if t == 1
            && let Some(c) = cap
            && c.wants(layer)
        {
            let last = outs.last().unwrap();
            let _ = last.eval_enqueue();
            c.store(
                &format!(
                    "L{layer}.dec_y_s{}",
                    crate::bindings::glm53_affine_tower::INC_STEP
                        .load(std::sync::atomic::Ordering::Relaxed)
                ),
                last,
            );
        }
        let out = concat(outs, 1)?; // [B,T,Hh,D]
        *ssm_state = Some(state.transpose_axes(&[0, 1, 2, 4, 3])?.reshape(&[
            b as i32,
            heads as i32,
            d as i32,
            d as i32,
        ])?);
        if let Some(c) = cap
            && c.wants(layer)
        {
            c.store(&format!("L{layer}.kda_out_prenorm"), &out);
        }

        // 6. Gated output norm + o_proj.
        let gate = output_gate; // [B,T,8192]
        let gate = gate.reshape(&[(b) as i32, (t) as i32, (heads) as i32, (d) as i32])?;
        let out = gated_rms_norm(&out, &self.o_norm_w, &gate, self.norm_eps)?;
        let out = out.reshape(&[(b) as i32, (t) as i32, (qkv) as i32])?;
        self.linear(&out, &self.o_proj)
    }
}

impl Kda {}
