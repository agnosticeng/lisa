//! Dense projection bindings and model-plan execution.
use crate::weights::{affine::QuantizedLinear, source::Weights};
use crate::{bindings::BindingDevice, execution::DeviceContext};

pub struct DenseMlp<D: BindingDevice> {
    context: DeviceContext<D>,
    /// Row-concatenation of gate|up (exact on the GEMV paths — same argument
    /// as GDN's `in_proj_all`): ONE gate+up qmv per layer instead of two.
    /// Built only for decode-eligible shapes (group_size/bits shared).
    plan: crate::core::plan::PreparedPlan<D>,
    pub geometry: [(i32, i32); 3],
    pub joined: Option<(i32, i32)>,
}

impl<D: BindingDevice> DenseMlp<D> {
    pub fn load(
        context: DeviceContext<D>,
        w: &mut Weights<D::Tensor>,
        prefix: &str,
        gs: i32,
        bits: i32,
    ) -> anyhow::Result<Self> {
        let mut gate_proj = QuantizedLinear::load(w, prefix, "gate_proj")?;
        gate_proj.set_quant(gs, bits);
        let mut up_proj = QuantizedLinear::load(w, prefix, "up_proj")?;
        up_proj.set_quant(gs, bits);
        let mut down_proj = QuantizedLinear::load(w, prefix, "down_proj")?;
        down_proj.set_quant(gs, bits);
        let geometry = [&gate_proj, &up_proj].map(|p| (p.group_size, p.bits));
        let graph = lisa_models::dense::join_gate_up(
            [gate_proj.dims_out() as i32, up_proj.dims_out() as i32],
            geometry,
        );
        let gate_up_proj = if let Some(graph) = graph {
            let prepared =
                crate::core::graph::PreparedGraph::new(context.device().clone(), graph, vec![])?;
            let mut output = prepared
                .run(
                    &[
                        &gate_proj.weight,
                        &gate_proj.scales,
                        &gate_proj.biases,
                        &up_proj.weight,
                        &up_proj.scales,
                        &up_proj.biases,
                    ],
                    &mut [],
                )?
                .into_iter();
            let mut projection = || QuantizedLinear {
                weight: output.next().unwrap(),
                scales: output.next().unwrap(),
                biases: output.next().unwrap(),
                group_size: gate_proj.group_size,
                bits: gate_proj.bits,
            };
            let joined = projection();
            let gate = projection();
            let up = projection();
            gate_proj = gate;
            up_proj = up;
            Some(joined)
        } else {
            None
        };
        let mut bindings = Vec::with_capacity(12);
        for projection in [&gate_proj, &up_proj, &down_proj] {
            bindings.extend([
                projection.weight.clone(),
                projection.scales.clone(),
                projection.biases.clone(),
            ]);
        }
        if let Some(projection) = &gate_up_proj {
            bindings.extend([
                projection.weight.clone(),
                projection.scales.clone(),
                projection.biases.clone(),
            ]);
        }
        let geometry = [&gate_proj, &up_proj, &down_proj].map(|p| (p.group_size, p.bits));
        let plans = lisa_models::dense::swiglu(
            geometry,
            gate_up_proj.as_ref().map(|p| (p.group_size, p.bits)),
        );
        let plan = crate::core::plan::PreparedPlan::new(context.device().clone(), plans, bindings)?;
        Ok(Self {
            context: context.clone(),

            plan,
            geometry,
            joined: gate_up_proj.as_ref().map(|p| (p.group_size, p.bits)),
        })
    }

    pub fn weights(&self) -> &[D::Tensor] {
        self.plan.weights()
    }
    pub fn forward(&self, x: &D::Tensor) -> anyhow::Result<D::Tensor> {
        self.plan
            .run(x)
            .map_err(|error| anyhow::Error::msg(error.to_string()))
    }
}
