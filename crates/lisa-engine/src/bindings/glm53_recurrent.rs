//! KDA (`linear_attention`) layers for GLM-5.3, following ds4's float graph.
//!
//! Kimi Delta Attention: depthwise
//! causal short conv (k=4) over the fused q|k|v stream, l2-normalized q/k
//! (q scaled `head_dim^-0.5`), per-token decay `g = lower_bound * sigmoid(
//! lower_bound * sigmoid(exp(A_log) * (f_b(f_a(x)) + dt_bias)))`, beta
//! `sigmoid(b_proj(x))`, and the
//! delta-rule recurrence on an F32 state `[B, 64, 128, 128]`.
//!
//! The normal single-stream path uses the ds4 prepare, recurrence, and output
//! kernels. The D::Tensor-op path remains an explicit diagnostic fallback.

use crate::bindings::glm53_kda::run as run_stage;
use crate::bindings::glm53_projection::Projection;
use crate::core::capture;
use crate::recurrent_cache::GdnCache;
use crate::weights::source::{TensorSource, Weights};
use crate::{bindings::BindingDevice, execution::DeviceContext};
use lisa_backend::TensorAccess;
use lisa_models::glm53::kda::{Spec, Stage};

#[cfg(any(test, feature = "diagnostics"))]
fn capture_decode_diagnostic<T: TensorAccess>(
    prefix: &std::ffi::OsStr,
    position: usize,
    layer: usize,
    name: &str,
    array: &T,
) -> anyhow::Result<()> {
    let values = array.read_f32()?;
    let bytes: Vec<u8> = values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();
    let prefix = std::path::PathBuf::from(prefix).display().to_string();
    std::fs::write(
        format!("{prefix}.kda-pos-{position:05}-L{layer:02}-{name}.f32"),
        bytes,
    )?;
    Ok(())
}

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

pub use crate::bindings::glm53_math::{gated_rms_norm, rms_norm_last};

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
        // small prefill use ds4's BF16/F32 GEMV path; larger prefill uses its
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

    #[cfg(any(test, feature = "diagnostics"))]
    pub fn ds4_decode_weights_for_capture(
        &self,
    ) -> anyhow::Result<(Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>)> {
        Ok((
            self.conv_q.read_f32()?,
            self.conv_k.read_f32()?,
            self.conv_v.read_f32()?,
            self.a_log.read_f32()?,
            self.dt_bias.read_f32()?,
            self.o_norm_w.read_f32()?,
        ))
    }

    #[cfg(any(test, feature = "diagnostics"))]
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
        crate::bindings::glm53_kda_projection::inputs(
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
            super::glm53_policy::glm53_kda_nsg_override(),
        )
    }

    /// Decode a cohort whose KDA recurrent states belong to separate
    /// sessions. The projections and output projection run over the complete
    /// row batch, while the fused recurrence kernel advances each row against
    /// its own convolution and SSM state, matching ds4's native batch path.
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

    /// Forward a prompt chunk using DS4's prefill path, including a one-token
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
        #[cfg(any(test, feature = "diagnostics"))]
        let decode_diagnostic = if layer == 1 && b == 1 && (t == 1 || t == 2) && !prefill_mode {
            _position.and_then(|position| {
                let requested = std::env::var("LISA_GLM53_KDA_DIAG_CAPTURE_POSITIONS").ok()?;
                let selected = requested
                    .split(',')
                    .filter_map(|item| item.trim().parse::<usize>().ok())
                    .any(|item| item == position);
                if selected {
                    Some((
                        std::env::var_os("LISA_GLM53_KDA_DIAG_CAPTURE_PREFIX")?,
                        position,
                    ))
                } else {
                    None
                }
            })
        } else {
            None
        };
        let capture_ds4_prefill =
            t > 1 && std::env::var_os("LISA_GLM53_KDA_PREFILL_DS4_CAPTURE").is_some();
        let active_cap = cap.filter(|c| c.wants(layer));
        let dump = (layer == 0)
            .then(|| std::env::var_os("LISA_GLM53_DUMP"))
            .flatten();
        let debug = layer == 0 && std::env::var_os("LISA_GLM53_DEBUG").is_some();
        #[cfg(any(test, feature = "diagnostics"))]
        let diagnostic = decode_diagnostic.is_some();
        #[cfg(not(any(test, feature = "diagnostics")))]
        let diagnostic = false;
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
            diagnostic: std::env::var_os("LISA_GLM53_NAIVE").is_some(),
            fused: false,
            triple: false,
            paired: false,
            groups: super::glm53_policy::glm53_kda_nsg_override(),
            observe: diagnostic || dump.is_some() || debug,
            capture: active_cap.is_some(),
            capture_prefill: capture_ds4_prefill,
            capture_input: std::env::var_os("LISA_GLM53_KDA_CAPTURE_PREFILL_INPUT").is_some(),
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
            crate::bindings::glm53_kda::forward(&self.context, key, &inputs)?;
        if let Some(c) = active_cap {
            let step = super::glm53_policy::INC_STEP
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
            if std::env::var("LISA_GLM53_KDA_CAPTURE_PROJECTION_WEIGHTS")
                .ok()
                .is_some_and(|v| v == "all" || v.parse::<usize>().ok() == Some(layer))
            {
                for (name, value) in [
                    ("q", &self.q_w),
                    ("k", &self.k_w),
                    ("v", &self.v_w),
                    ("fa", &self.f_a_w),
                    ("ga", &self.g_a_w),
                    ("beta", &self.b_w),
                    ("gate", &self.f_b_w),
                    ("gb", &self.g_b_w),
                ] {
                    c.store(&format!("L{layer}.kda_{name}_proj_weight"), value);
                }
            }
        }
        #[cfg(any(test, feature = "diagnostics"))]
        if let Some((prefix, position)) = decode_diagnostic.as_ref() {
            for (name, value) in &observations {
                if let Some(name) = name.strip_prefix('@') {
                    if !name.starts_with("dump_") {
                        capture_decode_diagnostic(prefix, *position, layer, name, value)?;
                    }
                }
            }
        }
        if let Some(path) = dump {
            // The original binary dump is specific to the prepare path.
            if observations.iter().any(|(name, _)| name == "@q_conv") {
                use std::io::Write;
                let path = std::path::Path::new(&path);
                let mut file = std::fs::File::create(path)?;
                let get = |name: &str| {
                    observations
                        .iter()
                        .find(|(n, _)| n == name)
                        .map(|(_, a)| a)
                        .ok_or_else(|| anyhow::anyhow!("missing recurrent dump observation {name}"))
                };
                for a in [
                    get("@dump_q")?,
                    get("@dump_k")?,
                    get("@dump_v")?,
                    get("@dump_gate")?,
                    &self.conv_q,
                    &self.conv_k,
                    &self.conv_v,
                    &self.a_log,
                    &self.dt_bias,
                    get("@conv_in")?,
                ] {
                    let values = a.read_f32()?;
                    file.write_all(&(values.len() as u32).to_le_bytes())?;
                    for value in values {
                        file.write_all(&value.to_le_bytes())?;
                    }
                }
                eprintln!("[kda-dump] wrote {}", path.display());
            }
        }
        if debug {
            for (name, a) in &observations {
                if matches!(name.as_str(), "@dump_q" | "@q_conv" | "@g") {
                    let values = a.read_f32()?;
                    let norm = values.iter().map(|x| x * x).sum::<f32>().sqrt();
                    eprintln!("[kda-debug] {name}: len {} norm {norm:.4}", values.len());
                }
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
        w: &crate::weights::gguf::ResidentWeights<D, L>,
        layer: usize,
    ) -> anyhow::Result<Self> {
        use crate::bindings::glm53_projection::plain;
        let p = |name: &str| format!("blk.{layer}.{name}.weight");
        let projection = |name: &str| Projection::load(w, &p(name));
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
