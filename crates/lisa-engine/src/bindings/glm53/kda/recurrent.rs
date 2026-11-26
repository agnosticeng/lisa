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
//! kernels. The D::Tensor-op path remains an explicit diagnostic fallback.

use crate::bindings::glm53::kda::run as run_stage;
use crate::bindings::glm53::packed::projection::Projection;
use crate::bindings::shared::capture;
use crate::cache::recurrent::GdnCache;
use crate::weights::source::{TensorSource, Weights};
use crate::{bindings::BindingDevice, execution::DeviceContext};
use lisa_backend::TensorAccess;
use lisa_models::glm53::kda::{Spec, Stage};

pub struct Kda<D: BindingDevice> {
    pub context: DeviceContext<D>,
    pub q_w: Projection<D>, // Plain model weight in source dtype, row-major [out, in]
    pub k_w: Projection<D>,
    pub v_w: Projection<D>,
    pub o_proj: Projection<D>, // Plain KDA projection in source dtype, row-major [out, in]
    pub f_a_w: Projection<D>,  // [4096, 128]
    pub f_b_w: Projection<D>,  // [128, 8192]
    pub g_a_w: Projection<D>,  // [4096, 128]
    pub g_b_w: Projection<D>,  // [128, 8192]
    pub b_w: Projection<D>,    // [4096, 64]
    pub conv_q: D::Tensor,     // F32 [channels, 4], taps flattened from [channels, 1, 4]
    pub conv_k: D::Tensor,
    pub conv_v: D::Tensor,
    pub dt_bias: D::Tensor,  // [64, 128] f32
    pub a_log: D::Tensor,    // [64] f32
    pub o_norm_w: D::Tensor, // [128] f32
    pub heads: usize,
    pub head_dim: usize,
    pub conv_k_sz: usize,
    pub lower_bound: f32,
    pub norm_eps: f32,
}

pub use crate::bindings::glm53::math::{gated_rms_norm, rms_norm_last};

impl<D: BindingDevice> Kda<D> {
    fn spec(&self, b: usize, t: usize) -> Spec {
        Spec {
            batch: b as i32,
            tokens: t as i32,
            heads: self.heads as i32,
            dimension: self.head_dim as i32,
            lower_bound: self.lower_bound.to_bits(),
            epsilon: self.norm_eps.to_bits(),
        }
    }

    pub fn load_on<S: TensorSource<D::Tensor>>(
        context: DeviceContext<D>,
        w: &mut Weights<D::Tensor>,
        prefix: &str,
        heads: usize,
        head_dim: usize,
        conv_k_sz: usize,
        lower_bound: f32,
        norm_eps: f32,
    ) -> anyhow::Result<Self> {
        let spec = Spec {
            batch: 1,
            tokens: 1,
            heads: heads as i32,
            dimension: head_dim as i32,
            lower_bound: lower_bound.to_bits(),
            epsilon: norm_eps.to_bits(),
        };
        anyhow::ensure!(
            conv_k_sz == lisa_models::glm53::kda::CONVOLUTION_TAPS,
            "unsupported KDA convolution width"
        );
        let transform = |a: D::Tensor, rows, columns| -> anyhow::Result<D::Tensor> {
            let [a] = run_stage(
                &context,
                spec,
                Stage::Weight {
                    rows,
                    columns,
                    cast: true,
                },
                &[&a],
            )?;
            Ok(a)
        };
        let take = |w: &mut Weights<D::Tensor>, name: &str| -> anyhow::Result<D::Tensor> {
            w.remove(name)
                .ok_or_else(|| anyhow::anyhow!("glm53 kda: {name} missing"))
        };
        let p = |suffix: &str| format!("{prefix}.{suffix}");
        // Keep plain projection weights in their source dtype. Decode and
        // small prefill use the reference's BF16/F32 GEMV path; larger prefill uses its
        // F16-staged, F32-accumulating tiled GEMM path.
        let q_w = take(w, &p(lisa_models::glm53::kda::MLX_WEIGHTS[0]))?;
        let k_w = take(w, &p(lisa_models::glm53::kda::MLX_WEIGHTS[1]))?;
        let v_w = take(w, &p(lisa_models::glm53::kda::MLX_WEIGHTS[2]))?;
        let f_a_w = take(w, &p(lisa_models::glm53::kda::MLX_WEIGHTS[4]))?;
        let f_b_w = take(w, &p(lisa_models::glm53::kda::MLX_WEIGHTS[5]))?;
        let g_a_w = take(w, &p(lisa_models::glm53::kda::MLX_WEIGHTS[6]))?;
        let g_b_w = take(w, &p(lisa_models::glm53::kda::MLX_WEIGHTS[7]))?;
        let b_w = take(w, &p(lisa_models::glm53::kda::MLX_WEIGHTS[8]))?;
        let flat_taps = |a: D::Tensor| -> anyhow::Result<D::Tensor> {
            let shape = a.shape();
            anyhow::ensure!(shape.len() == 3, "invalid KDA convolution weight");
            let (rows, columns) = (shape[0], shape[2]);
            transform(a, rows, columns)
        };
        let conv_q = flat_taps(take(w, &p(lisa_models::glm53::kda::MLX_WEIGHTS[9]))?)?;
        let conv_k = flat_taps(take(w, &p(lisa_models::glm53::kda::MLX_WEIGHTS[10]))?)?;
        let conv_v = flat_taps(take(w, &p(lisa_models::glm53::kda::MLX_WEIGHTS[11]))?)?;
        let dt_bias = transform(take(w, &p(lisa_models::glm53::kda::MLX_WEIGHTS[12]))?, 0, 0)?;
        let a_log = transform(take(w, &p(lisa_models::glm53::kda::MLX_WEIGHTS[13]))?, 0, 0)?;
        let o_norm_w = transform(take(w, &p(lisa_models::glm53::kda::MLX_WEIGHTS[14]))?, 0, 0)?;
        let o_proj = take(w, &p(lisa_models::glm53::kda::MLX_WEIGHTS[3]))?;
        Ok(Self {
            context: context.clone(),
            q_w: q_w.into(),
            k_w: k_w.into(),
            v_w: v_w.into(),
            o_proj: o_proj.into(),
            f_a_w: f_a_w.into(),
            f_b_w: f_b_w.into(),
            g_a_w: g_a_w.into(),
            g_b_w: g_b_w.into(),
            b_w: b_w.into(),
            conv_q,
            conv_k,
            conv_v,
            dt_bias,
            a_log,
            o_norm_w,
            heads,
            head_dim,
            conv_k_sz,
            lower_bound,
            norm_eps,
        })
    }

    #[cfg(test)]
    pub fn input_projections(
        &self,
        x: &D::Tensor,
    ) -> anyhow::Result<(
        D::Tensor,
        D::Tensor,
        D::Tensor,
        D::Tensor,
        D::Tensor,
        D::Tensor,
    )> {
        crate::bindings::glm53::kda::projection::inputs(
            &self.context,
            [
                &self.q_w,
                &self.k_w,
                &self.v_w,
                &self.f_a_w,
                &self.g_a_w,
                &self.b_w,
            ],
            x,
            crate::bindings::glm53::policy::glm53_kda_nsg_override(),
        )
    }

    /// Decode a cohort whose KDA recurrent states belong to separate
    /// sessions. The projections and output projection run over the complete
    /// row batch, while the fused recurrence kernel advances each row against
    /// its own convolution and SSM state, matching the reference's native batch path.
    pub fn forward_ragged(
        &self,
        x: &D::Tensor,
        caches: &mut [&mut GdnCache<D::Tensor>],
        cap: Option<&capture::Capture<D::Tensor>>,
        layer: usize,
    ) -> anyhow::Result<D::Tensor> {
        let b = caches.len();
        anyhow::ensure!(
            b > 1 && x.dim(0) == b as i32 && x.dim(1) == 1,
            "glm53 KDA ragged decode expects B>1, one token per row, got {:?} for {b} caches",
            x.shape()
        );
        let spec = self.spec(b, 1);
        let mut conv_rows = Vec::with_capacity(b);
        let mut state_rows = Vec::with_capacity(b);
        for cache in caches.iter() {
            let inputs = cache
                .conv
                .iter()
                .chain(cache.ssm.iter())
                .collect::<Vec<_>>();
            let [c, s] = run_stage(
                &self.context,
                spec,
                Stage::CacheRow {
                    conv: cache.conv.is_some(),
                    ssm: cache.ssm.is_some(),
                },
                &inputs,
            )?;
            conv_rows.push(c);
            state_rows.push(s);
        }
        let [conv] = run_stage(
            &self.context,
            spec,
            Stage::Join { count: b },
            &conv_rows.iter().collect::<Vec<_>>(),
        )?;
        let [state] = run_stage(
            &self.context,
            spec,
            Stage::Join { count: b },
            &state_rows.iter().collect::<Vec<_>>(),
        )?;
        let mut conv = Some(conv);
        let mut state = Some(state);
        let out = self.forward(x, &mut conv, &mut state, cap, layer)?;
        let conv = conv.ok_or_else(|| anyhow::anyhow!("glm53 KDA lost its batched conv state"))?;
        let state = state.ok_or_else(|| anyhow::anyhow!("glm53 KDA lost its batched SSM state"))?;
        for (row, cache) in caches.iter_mut().enumerate() {
            let [c, s] = run_stage(
                &self.context,
                spec,
                Stage::CacheSplit { row: row as i32 },
                &[&conv, &state],
            )?;
            cache.conv = Some(c);
            cache.ssm = Some(s);
        }
        Ok(out)
    }

    /// Forward `x [B,T,H]` with the layer cache (`GdnCache`: conv window +
    /// F32 recurrence state). Returns the block output `[B,T,H]`.
    pub fn forward(
        &self,
        x: &D::Tensor,
        conv_state: &mut Option<D::Tensor>,
        ssm_state: &mut Option<D::Tensor>,
        cap: Option<&capture::Capture<D::Tensor>>,
        layer: usize,
    ) -> anyhow::Result<D::Tensor> {
        self.forward_with_mode(x, conv_state, ssm_state, cap, layer, false, None)
    }

    pub fn forward_at_position(
        &self,
        x: &D::Tensor,
        conv_state: &mut Option<D::Tensor>,
        ssm_state: &mut Option<D::Tensor>,
        cap: Option<&capture::Capture<D::Tensor>>,
        layer: usize,
        position: usize,
    ) -> anyhow::Result<D::Tensor> {
        self.forward_with_mode(x, conv_state, ssm_state, cap, layer, false, Some(position))
    }

    /// Forward a prompt chunk using the reference's prefill path, including a one-token
    /// continuation chunk after the 2048-token prefill boundary.
    pub fn forward_prefill(
        &self,
        x: &D::Tensor,
        conv_state: &mut Option<D::Tensor>,
        ssm_state: &mut Option<D::Tensor>,
        cap: Option<&capture::Capture<D::Tensor>>,
        layer: usize,
    ) -> anyhow::Result<D::Tensor> {
        self.forward_with_mode(x, conv_state, ssm_state, cap, layer, true, None)
    }

    fn forward_with_mode(
        &self,
        x: &D::Tensor,
        conv_state: &mut Option<D::Tensor>,
        ssm_state: &mut Option<D::Tensor>,
        cap: Option<&capture::Capture<D::Tensor>>,
        layer: usize,
        prefill_mode: bool,
        _position: Option<usize>,
    ) -> anyhow::Result<D::Tensor> {
        let (b, t, _h) = (x.dim(0) as usize, x.dim(1) as usize, x.dim(2) as usize);
        let active_cap = cap.filter(|c| c.wants(layer));
        let projections = [
            &self.q_w,
            &self.k_w,
            &self.v_w,
            &self.o_proj,
            &self.f_a_w,
            &self.f_b_w,
            &self.g_a_w,
            &self.g_b_w,
            &self.b_w,
        ];
        let key = lisa_models::glm53::kda::Forward {
            spec: self.spec(b, t),
            projections: projections
                .iter()
                .map(|w| w.descriptor_on(self.context.device()))
                .collect::<anyhow::Result<Vec<_>>>()?
                .try_into()
                .ok()
                .unwrap(),
            conv: conv_state.is_some(),
            state_rank: ssm_state.as_ref().map_or(0, D::Tensor::rank),
            prefill: prefill_mode,
            fused: false,
            triple: false,
            paired: false,
            groups: crate::bindings::glm53::policy::glm53_kda_nsg_override(),
            capture: active_cap.is_some(),
        };
        let mut inputs = vec![x];
        inputs.extend(projections.iter().map(|p| &***p));
        inputs.extend([
            &self.conv_q,
            &self.conv_k,
            &self.conv_v,
            &self.dt_bias,
            &self.a_log,
            &self.o_norm_w,
        ]);
        inputs.extend(conv_state.iter());
        inputs.extend(ssm_state.iter());
        let ([out, next_conv, next_state], observations) =
            crate::bindings::glm53::kda::forward(&self.context, key, &inputs)?;
        if let Some(c) = active_cap {
            let step = crate::bindings::glm53::policy::INC_STEP
                .load(std::sync::atomic::Ordering::Relaxed)
                .to_string();
            for (name, value) in &observations {
                if !name.starts_with('@') && (layer == 0 || !name.starts_with("dec_conv_")) {
                    c.store(
                        &format!("L{layer}.{}", name.replace("{step}", &step)),
                        value,
                    );
                }
            }
            if observations
                .iter()
                .any(|(name, _)| name == "kda_initial_conv")
            {
                for (name, value) in [
                    ("kda_q_conv_weight", &self.conv_q),
                    ("kda_k_conv_weight", &self.conv_k),
                    ("kda_v_conv_weight", &self.conv_v),
                    ("kda_a_log", &self.a_log),
                    ("kda_dt_bias", &self.dt_bias),
                    ("kda_output_norm_weight", &self.o_norm_w),
                ] {
                    c.store(&format!("L{layer}.{name}"), value);
                }
                c.store(
                    &format!("L{layer}.kda_meta"),
                    &pull_f32(
                        &self.context,
                        &[
                            self.lower_bound,
                            self.norm_eps,
                            self.heads as f32,
                            self.head_dim as f32,
                            self.conv_k_sz as f32,
                        ],
                        &[5],
                    )?,
                );
            }
        }
        *conv_state = Some(next_conv);
        *ssm_state = Some(next_state);
        Ok(out)
    }
}

impl<D: BindingDevice> Kda<D> {
    pub fn load_gguf<L>(
        context: DeviceContext<D>,
        w: &crate::weights::prepared::PreparedWeights<D::Tensor, L>,
        layer: usize,
    ) -> anyhow::Result<Self> {
        use crate::bindings::glm53::packed::projection::plain;
        let p = |name: &str| format!("blk.{layer}.{name}.weight");
        let projection = |name: &str| Projection::<D>::load(w, &p(name));
        let f = |name: &str| plain(w, &p(name));
        let spec = lisa_models::glm53::kda::GGUF_SPEC;
        let taps = |a: D::Tensor| -> anyhow::Result<D::Tensor> {
            let [a] = run_stage(
                &context,
                spec,
                Stage::Weight {
                    rows: spec.heads * spec.dimension,
                    columns: lisa_models::glm53::kda::CONVOLUTION_TAPS as i32,
                    cast: false,
                },
                &[&a],
            )?;
            Ok(a)
        };
        Ok(Self {
            context: context.clone(),
            q_w: projection(lisa_models::glm53::kda::GGUF_WEIGHTS[0])?,
            k_w: projection(lisa_models::glm53::kda::GGUF_WEIGHTS[1])?,
            v_w: projection(lisa_models::glm53::kda::GGUF_WEIGHTS[2])?,
            o_proj: projection(lisa_models::glm53::kda::GGUF_WEIGHTS[3])?,
            f_a_w: projection(lisa_models::glm53::kda::GGUF_WEIGHTS[4])?,
            f_b_w: projection(lisa_models::glm53::kda::GGUF_WEIGHTS[5])?,
            g_a_w: projection(lisa_models::glm53::kda::GGUF_WEIGHTS[6])?,
            g_b_w: projection(lisa_models::glm53::kda::GGUF_WEIGHTS[7])?,
            b_w: projection(lisa_models::glm53::kda::GGUF_WEIGHTS[8])?,
            conv_q: taps(f(lisa_models::glm53::kda::GGUF_WEIGHTS[9])?)?,
            conv_k: taps(f(lisa_models::glm53::kda::GGUF_WEIGHTS[10])?)?,
            conv_v: taps(f(lisa_models::glm53::kda::GGUF_WEIGHTS[11])?)?,
            dt_bias: f(lisa_models::glm53::kda::GGUF_WEIGHTS[12])?,
            a_log: f(lisa_models::glm53::kda::GGUF_WEIGHTS[13])?,
            o_norm_w: f(lisa_models::glm53::kda::GGUF_WEIGHTS[14])?,
            heads: spec.heads as usize,
            head_dim: spec.dimension as usize,
            conv_k_sz: lisa_models::glm53::kda::CONVOLUTION_TAPS,
            lower_bound: f32::from_bits(spec.lower_bound),
            norm_eps: f32::from_bits(spec.epsilon),
        })
    }
}

fn pull_f32<D: BindingDevice>(
    context: &DeviceContext<D>,
    values: &[f32],
    shape: &[i32],
) -> anyhow::Result<D::Tensor> {
    Ok(context.device().pull(
        lisa_backend::HostTensor {
            bytes: &values
                .iter()
                .flat_map(|x| x.to_ne_bytes())
                .collect::<Vec<_>>(),
            shape,
            dtype: lisa_backend::ScalarType::F32,
        }
        .into(),
        Default::default(),
    )?)
}
