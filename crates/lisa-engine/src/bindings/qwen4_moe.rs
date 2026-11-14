use super::normalization::load_bf16;
use crate::weights::{affine::QuantizedLinear, source::TensorSource};
// Mixture-of-experts block: fp32 router, top-10 of 512, quantized
// gather-GEMM experts, shared expert with sigmoid gate.

use crate::core::graph::{PlanCache, PreparedGraph};
use crate::{bindings::BindingDevice, execution::DeviceContext};
use lisa_backend::TensorAccess;
use lisa_models::qwen4::moe as model;

/// A quantized SwiGLU expert stack: gate_proj/up_proj [E, H, K/8],
/// down_proj [E, K, H/8].
pub struct SwitchGlu<T> {
    pub gate_proj: T,
    pub gate_scales: T,
    pub gate_biases: T,
    pub up_proj: T,
    pub up_scales: T,
    pub up_biases: T,
    pub down_proj: T,
    pub down_scales: T,
    pub down_biases: T,
}

impl<T: TensorAccess> SwitchGlu<T> {
    pub fn load<S: TensorSource<T>>(src: &mut S, prefix: &str) -> anyhow::Result<Self> {
        let names = model::switch_names(prefix);
        Ok(Self {
            gate_proj: src.get(&names[0])?,
            gate_scales: src.get(&names[1])?,
            gate_biases: src.get(&names[2])?,
            up_proj: src.get(&names[3])?,
            up_scales: src.get(&names[4])?,
            up_biases: src.get(&names[5])?,
            down_proj: src.get(&names[6])?,
            down_scales: src.get(&names[7])?,
            down_biases: src.get(&names[8])?,
        })
    }
}

/// Sparse MoE block.
pub struct SparseMoeBlock<D: BindingDevice> {
    pub context: DeviceContext<D>,
    features: model::Features,
    pub gate: D::Tensor, // bf16 [512, 2560] router weight, NOT quantized
    pub switch_mlp: SwitchGlu<D::Tensor>,
    pub shared_gate_proj: QuantizedLinear<D::Tensor>,
    pub shared_up_proj: QuantizedLinear<D::Tensor>,
    pub shared_down_proj: QuantizedLinear<D::Tensor>,
    pub shared_expert_gate: QuantizedLinear<D::Tensor>,
    pub top_k: usize,
    /// Fused shared `gate|up` quantized weight `[2N, K]` for the decode kernel
    /// (concatenated once; the per-row GEMV is unaffected by the concat).
    pub fused_shared_gate_up: std::sync::OnceLock<(D::Tensor, D::Tensor, D::Tensor)>,
}

impl<D: BindingDevice> SparseMoeBlock<D> {
    pub fn load_on<S: TensorSource<D::Tensor>>(
        context: DeviceContext<D>,
        src: &mut S,
        prefix: &str,
        top_k: usize,
    ) -> anyhow::Result<Self> {
        let names = model::names(prefix);
        Ok(Self {
            context: context.clone(),
            features: features(context.device()),
            gate: load_bf16(&context, src, &names.router)?,
            switch_mlp: SwitchGlu::load(src, &names.switched)?,
            shared_gate_proj: QuantizedLinear::load(src, &names.shared, names.gate)?,
            shared_up_proj: QuantizedLinear::load(src, &names.shared, names.up)?,
            shared_down_proj: QuantizedLinear::load(src, &names.shared, names.down)?,
            shared_expert_gate: QuantizedLinear::load(src, prefix, names.selector)?,
            top_k,
            fused_shared_gate_up: std::sync::OnceLock::new(),
        })
    }

    fn spec(&self) -> model::Spec {
        let q = |w: &QuantizedLinear<D::Tensor>| (w.group_size, w.bits);
        model::Spec {
            experts: self.gate.dim(0),
            hidden: self.gate.dim(1),
            intermediate: self.switch_mlp.gate_proj.dim(1),
            shared: self.shared_gate_proj.dims_out() as i32,
            top_k: self.top_k as i32,
            quant: [
                q(&self.shared_gate_proj),
                q(&self.shared_up_proj),
                q(&self.shared_down_proj),
                q(&self.shared_expert_gate),
            ],
        }
    }
    pub fn forward(&self, x: &D::Tensor) -> anyhow::Result<D::Tensor> {
        self.execute(x)
            .map_err(|e| anyhow::Error::msg(e.to_string()))
    }
    fn execute(&self, x: &D::Tensor) -> anyhow::Result<D::Tensor> {
        let spec = self.spec();
        let features = self.features;
        let branch = spec.branch(x.shape(), features)?;
        let sm = &self.switch_mlp;
        let mut inputs = vec![
            x,
            &self.gate,
            &sm.gate_proj,
            &sm.gate_scales,
            &sm.gate_biases,
            &sm.up_proj,
            &sm.up_scales,
            &sm.up_biases,
            &sm.down_proj,
            &sm.down_scales,
            &sm.down_biases,
            &self.shared_gate_proj.weight,
            &self.shared_gate_proj.scales,
            &self.shared_gate_proj.biases,
            &self.shared_up_proj.weight,
            &self.shared_up_proj.scales,
            &self.shared_up_proj.biases,
            &self.shared_down_proj.weight,
            &self.shared_down_proj.scales,
            &self.shared_down_proj.biases,
            &self.shared_expert_gate.weight,
            &self.shared_expert_gate.scales,
            &self.shared_expert_gate.biases,
        ];
        if branch == model::Branch::Decode {
            if self.fused_shared_gate_up.get().is_none() {
                let prepare = PreparedGraph::new(
                    self.context.device().clone(),
                    model::join_shared(),
                    vec![],
                )?;
                let mut out = prepare.run(&inputs[11..17], &mut [])?.into_iter();
                let _ = self.fused_shared_gate_up.set((
                    out.next().unwrap(),
                    out.next().unwrap(),
                    out.next().unwrap(),
                ));
            }
            let (w, s, b) = self.fused_shared_gate_up.get().unwrap();
            inputs.extend([w, s, b]);
        }
        self.context.with_slot(
            "qwen4.moe.plans",
            || std::cell::RefCell::new(PlanCache::new(self.context.device().clone(), 64)),
            |cache| {
                cache
                    .borrow_mut()
                    .try_get_or_prepare((spec, features, x.shape().to_vec()), || {
                        spec.graph(x.shape(), features)
                    })?
                    .run(&inputs, &mut [])?
                    .into_one()
                    .map_err(anyhow::Error::msg)
            },
        )
    }
}
fn features<D: BindingDevice>(device: &D) -> model::Features {
    use lisa_backend::{AffineExpertsKind as K, Device, KernelKind};
    let caps = device.capabilities();
    let has = |kind| {
        caps.iter()
            .any(|c| c.kind == KernelKind::AffineExperts(kind))
    };
    model::Features {
        matrix_router: has(K::RouterMatrix),
        tiled_experts: has(K::TiledGateUp) && has(K::TiledDown),
    }
}
type Key = (model::Spec, model::Features, Vec<i32>);
