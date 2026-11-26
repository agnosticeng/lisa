//! Gated-delta weight and cache bindings. Computation and state layout laws
//! are supplied by lisa-models; execution uses prepared device operations.

use crate::{bindings::BindingDevice, execution::DeviceContext};
use lisa_backend::TensorAccess;

use crate::bindings::shared::normalization::{RmsNormGated, load_bf16};
use crate::cache::recurrent::GdnCache;
use crate::weights::affine::QuantizedLinear;
use crate::weights::source::TensorSource;

/// Gated deltanet block.
pub struct GatedDeltaNet<D: BindingDevice> {
    context: DeviceContext<D>,
    pub in_proj_qkv: QuantizedLinear<D::Tensor>,
    pub in_proj_z: QuantizedLinear<D::Tensor>,
    pub in_proj_b: QuantizedLinear<D::Tensor>,
    pub in_proj_a: QuantizedLinear<D::Tensor>,
    /// The four input projections concatenated `[qkv | z | b | a]` (one GEMM),
    /// used by the fused S=1 decode kernel.
    pub in_proj_all: QuantizedLinear<D::Tensor>,
    pub z_offset: i32,
    pub b_offset: i32,
    pub a_offset: i32,
    pub proj_width: i32,
    pub conv1d_weight: D::Tensor, // [convDim, K, 1]
    pub a_log: D::Tensor,         // bf16 [48]
    /// `-exp(a_log.f32)`, precomputed once at load (the engine's bind-time
    /// `negExpALog`); recomputing it per layer was three lazy ops each.
    pub neg_exp_alog: D::Tensor,
    pub dt_bias: D::Tensor, // bf16 [48]
    pub norm: RmsNormGated<D::Tensor>,
    pub out_proj: QuantizedLinear<D::Tensor>,
    pub value_heads: usize,
    pub key_heads: usize,
    pub key_head_dim: usize,
    pub value_head_dim: usize,
    pub conv_kernel_size: usize,
    pub conv_dim: usize,
    /// Qwen3.5 (`output_gate_type=swish`) gates the GDN output with `silu(z)`;
    /// Flash-Next uses `sigmoid(z)`.
    pub output_gate_silu: bool,
    plans: std::cell::RefCell<
        Vec<(
            lisa_models::gated_delta::Plan,
            crate::core::graph::PreparedGraph<D>,
        )>,
    >,
}

impl<D: BindingDevice> GatedDeltaNet<D> {
    pub fn load<S: TensorSource<D::Tensor>>(
        context: DeviceContext<D>,
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
        use lisa_models::gated_delta::BindingSchema;
        let names = BindingSchema::at(prefix);
        let conv_dim =
            BindingSchema::channels(key_heads, value_heads, key_head_dim, value_head_dim)?;
        let conv1d_weight = load_bf16(&context, src, &names.convolution)?;
        let mut in_proj_qkv = QuantizedLinear::load(src, prefix, BindingSchema::INPUTS[0])?;
        in_proj_qkv.set_quant(group_size, bits);
        let mut in_proj_z = QuantizedLinear::load(src, prefix, BindingSchema::INPUTS[1])?;
        in_proj_z.set_quant(group_size, bits);
        let mut in_proj_b = QuantizedLinear::load(src, prefix, BindingSchema::INPUTS[2])?;
        in_proj_b.set_quant(group_size, bits);
        let mut in_proj_a = QuantizedLinear::load(src, prefix, BindingSchema::INPUTS[3])?;
        in_proj_a.set_quant(group_size, bits);
        let mut out_proj = QuantizedLinear::load(src, prefix, BindingSchema::OUTPUT)?;
        out_proj.set_quant(group_size, bits);
        let a_log = load_bf16(&context, src, &names.decay)?;
        let projections = [&in_proj_qkv, &in_proj_z, &in_proj_b, &in_proj_a];
        let layout = lisa_models::gated_delta::ProjectionLayout::new(
            projections.map(|p| p.dims_out() as i32),
        )?;
        let plan = crate::core::graph::PreparedGraph::new(
            context.device().clone(),
            layout.initialize(conv1d_weight.shape()),
            vec![],
        )?;
        let mut inputs = Vec::with_capacity(14);
        for p in projections {
            inputs.extend([&p.weight, &p.scales, &p.biases]);
        }
        inputs.extend([&conv1d_weight, &a_log]);
        let mut output = plan.run(&inputs, &mut [])?.into_iter();
        let mut projection = || QuantizedLinear {
            weight: output.next().unwrap(),
            scales: output.next().unwrap(),
            biases: output.next().unwrap(),
            group_size,
            bits,
        };
        let in_proj_all = projection();
        in_proj_qkv = projection();
        in_proj_z = projection();
        in_proj_b = projection();
        in_proj_a = projection();
        let conv1d_weight = output.next().unwrap();
        let neg_exp_alog = output.next().unwrap();
        let [_, z_offset, b_offset, a_offset, proj_width] = layout.offsets;
        Ok(Self {
            context: context.clone(),

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
            dt_bias: load_bf16(&context, src, &names.bias)?,
            norm: RmsNormGated::load_on(&context, src, &names.norm, eps)?,
            out_proj,
            value_heads,
            key_heads,
            key_head_dim,
            value_head_dim,
            conv_kernel_size,
            conv_dim,
            output_gate_silu: false,
            plans: std::cell::RefCell::new(Vec::new()),
        })
    }

    /// x: [B, S, hidden]; cache holds conv + ssm state, updated in place.
    ///
    /// `capture` (speculative verify only) additionally records the SSM state
    /// after every position and the raw conv input so the cache can be rolled
    /// back to an accepted prefix.
    pub fn forward(
        &self,
        x: &D::Tensor,
        cache: Option<&mut GdnCache<D::Tensor>>,
        capture: bool,
    ) -> anyhow::Result<D::Tensor> {
        self.forward_with_capture_law(x, cache, capture, false)
    }

    pub fn forward_with_capture_law(
        &self,
        x: &D::Tensor,
        cache: Option<&mut GdnCache<D::Tensor>>,
        capture: bool,
        serial_capture: bool,
    ) -> anyhow::Result<D::Tensor> {
        use lisa_models::gated_delta::{CaptureKind, DecodePlan, Plan};
        let conv = cache.as_deref().and_then(|c| c.conv.as_ref());
        let state = cache.as_deref().and_then(|c| c.ssm.as_ref());
        let workspace = cache
            .as_deref()
            .and_then(|c| c.rec_y_buf.as_ref().zip(c.capture_ssm.as_ref()));
        let projections = [
            &self.in_proj_all,
            &self.in_proj_qkv,
            &self.in_proj_z,
            &self.in_proj_b,
            &self.in_proj_a,
        ];
        let kernel = lisa_backend::DeltaDecode {
            key_heads: self.key_heads as i32,
            value_heads: self.value_heads as i32,
            key_width: self.key_head_dim as i32,
            value_width: self.value_head_dim as i32,
            convolution: self.conv_kernel_size as i32,
            channels: self.conv_dim as i32,
            projection: self.proj_width,
            beta_offset: self.b_offset,
            decay_offset: self.a_offset,
            gate_offset: self.z_offset,
            epsilon: self.norm.eps.to_bits(),
            swish: self.output_gate_silu,
        };
        let mut spec = Plan {
            decode: DecodePlan {
                kernel,
                batch: x.dim(0),
                convolution_present: conv.is_some(),
                state_present: state.is_some(),
                projections: [
                    (self.in_proj_all.group_size, self.in_proj_all.bits),
                    (self.out_proj.group_size, self.out_proj.bits),
                ],
            },
            tokens: x.dim(1),
            capture,
            cache: cache.is_some(),
            verify_fusion: true,
            reuse: false,
            input_projections: projections.map(|p| (p.group_size, p.bits)),
            serial_capture,
        };
        spec.reuse = spec.recurrence().accepts_workspace(
            workspace.map(|(y, _)| y.shape()),
            workspace.map(|(_, s)| s.shape()),
        );
        let run = || -> anyhow::Result<_> {
            let mut plans = self.plans.borrow_mut();
            let index = if let Some(i) = plans.iter().position(|(key, _)| *key == spec) {
                i
            } else {
                let mut weights = Vec::with_capacity(22);
                for p in projections {
                    weights.extend([p.weight.clone(), p.scales.clone(), p.biases.clone()]);
                }
                weights.extend([
                    self.conv1d_weight.clone(),
                    self.neg_exp_alog.clone(),
                    self.dt_bias.clone(),
                    self.norm.weight.clone(),
                    self.out_proj.weight.clone(),
                    self.out_proj.scales.clone(),
                    self.out_proj.biases.clone(),
                ]);
                let plan = crate::core::graph::PreparedGraph::new(
                    self.context.device().clone(),
                    spec.graph()?,
                    weights,
                )?;
                if plans.len() == 16 {
                    plans.remove(0);
                }
                plans.push((spec, plan));
                plans.len() - 1
            };
            let mut inputs = vec![x];
            if let Some(conv) = conv {
                inputs.push(conv);
            }
            if let Some(state) = state {
                inputs.push(state);
            }
            if spec.reuse {
                let (y, s) = workspace.unwrap();
                inputs.extend([y, s]);
            }
            plans[index].1.run(&inputs, &mut [])
        };
        let mut out = run()
            .map_err(|e| anyhow::Error::msg(e.to_string()))?
            .into_iter();
        let activation = out.next().unwrap();
        let conv = out.next().unwrap();
        let state = out.next().unwrap();
        if let Some(c) = cache {
            c.conv = Some(conv);
            c.ssm = Some(state);
            if spec.capture_kind() == CaptureKind::Snapshots {
                c.rec_y_buf = out.next();
                c.capture_ssm = out.next();
                c.capture_conv_input = out.next();
            }
        }
        if spec.is_decode() {
            static ENGAGED: std::sync::Once = std::sync::Once::new();
            ENGAGED.call_once(|| {
                eprintln!(
                    "[gdn] decode complete engaged (gate={})",
                    if self.output_gate_silu {
                        "silu"
                    } else {
                        "sigmoid"
                    }
                )
            });
        }
        Ok(activation)
    }
}
