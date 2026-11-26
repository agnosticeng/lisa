use crate::bindings::shared::normalization::RmsNorm;
use crate::weights::{affine::QuantizedLinear, source::TensorSource};
// Checkpoint storage bound to model-owned gated residual plans.
use crate::core::graph::{PlanCache, PreparedGraph};
use crate::{bindings::BindingDevice, execution::DeviceContext};
use lisa_backend::Outputs;
use lisa_backend::TensorAccess;
use lisa_models::qwen4::hyper::{self as model, Mode, Spec};
#[derive(Clone)]
pub struct GatedResidual<D: BindingDevice> {
    pub context: DeviceContext<D>,
    pub hc_norm: RmsNorm<D::Tensor>,
    pub mix_down: QuantizedLinear<D::Tensor>,
    pub mix_up: QuantizedLinear<D::Tensor>,
    pub decode_up: Option<QuantizedLinear<D::Tensor>>,
    pub block_inject: Option<QuantizedLinear<D::Tensor>>,
    pub hc_count: usize,
    pub hidden: usize,
    pub force_norm_operations: bool,
}
#[derive(PartialEq)]
enum Key {
    Finish(Spec, Vec<i32>, u32, bool, bool),
    InjectNorm(Vec<i32>, i32, i32, u32, bool),
    Forward(Spec, Mode, Vec<i32>, bool),
    Scale(i32),
    Inject(Vec<i32>),
}
fn execute<D: BindingDevice>(
    context: &DeviceContext<D>,
    key: Key,
    inputs: &[&D::Tensor],
    build: impl FnOnce() -> anyhow::Result<lisa_models::graph::Graph>,
) -> anyhow::Result<Outputs<D::Tensor>> {
    context
        .with_slot(
            "qwen4.hyper.plans",
            || std::cell::RefCell::new(PlanCache::new(context.device().clone(), 64)),
            |cache| {
                cache
                    .borrow_mut()
                    .try_get_or_prepare(key, build)?
                    .run(inputs, &mut [])
            },
        )
        .map_err(|e: anyhow::Error| anyhow::Error::msg(e.to_string()))
}
fn one<D: BindingDevice>(out: Outputs<D::Tensor>) -> anyhow::Result<D::Tensor> {
    out.into_one().map_err(anyhow::Error::msg)
}
impl<D: BindingDevice> GatedResidual<D> {
    pub fn load_on<S: TensorSource<D::Tensor>>(
        context: DeviceContext<D>,
        src: &mut S,
        prefix: &str,
        hidden: usize,
        hc_count: usize,
        use_inject: bool,
        force_norm_operations: bool,
    ) -> anyhow::Result<Self> {
        let names = model::names(prefix, use_inject);
        let mix_up = QuantizedLinear::load(src, prefix, names.up)?;
        let decode_up = if let Some(order) =
            model::decode_order(hidden, hc_count, mix_up.weight.dim(0) as usize)
        {
            let prepared =
                PreparedGraph::new(context.device().clone(), model::pack_rows(&order), vec![])?;
            let mut out = prepared
                .run(&[&mix_up.weight, &mix_up.scales, &mix_up.biases], &mut [])?
                .into_iter();
            Some(QuantizedLinear {
                weight: out.next().unwrap(),
                scales: out.next().unwrap(),
                biases: out.next().unwrap(),
                group_size: mix_up.group_size,
                bits: mix_up.bits,
            })
        } else {
            None
        };
        Ok(Self {
            context: context.clone(),
            hc_norm: RmsNorm::load_on(&context, src, &names.norm, 1e-6, Some(hidden))?,
            mix_down: QuantizedLinear::load(src, prefix, names.down)?,
            mix_up,
            decode_up,
            block_inject: names
                .inject
                .map(|name| QuantizedLinear::load(src, prefix, name))
                .transpose()?,
            hc_count,
            hidden: hidden * hc_count,
            force_norm_operations,
        })
    }
    fn spec(&self) -> Spec {
        let q = |x: &QuantizedLinear<D::Tensor>| (x.group_size, x.bits);
        Spec {
            streams: self.hc_count as i32,
            hidden: (self.hidden / self.hc_count) as i32,
            rank: self.mix_down.weight.dim(0),
            down: q(&self.mix_down),
            up: q(&self.mix_up),
            inject: self.block_inject.as_ref().map(q),
            packed: self.decode_up.is_some(),
            epsilon: self.hc_norm.eps.to_bits(),
            force_norm_operations: self.force_norm_operations,
        }
    }
    fn run(&self, x: &D::Tensor, mode: Mode) -> anyhow::Result<Outputs<D::Tensor>> {
        self.run_with_capture_law(x, mode, false)
    }
    fn run_with_capture_law(
        &self,
        x: &D::Tensor,
        mode: Mode,
        captured: bool,
    ) -> anyhow::Result<Outputs<D::Tensor>> {
        let spec = self.spec();
        let inject = self.block_inject.as_ref().unwrap_or(&self.mix_down);
        let packed = self.decode_up.as_ref().unwrap_or(&self.mix_up);
        execute(
            &self.context,
            Key::Forward(spec, mode, x.shape().to_vec(), captured),
            &[
                x,
                &self.hc_norm.weight,
                &self.mix_down.weight,
                &self.mix_down.scales,
                &self.mix_down.biases,
                &self.mix_up.weight,
                &self.mix_up.scales,
                &self.mix_up.biases,
                &inject.weight,
                &inject.scales,
                &inject.biases,
                &packed.weight,
                &packed.scales,
                &packed.biases,
            ],
            || {
                if captured {
                    spec.graph_captured(x.shape(), mode)
                } else {
                    spec.graph(x.shape(), mode)
                }
            },
        )
    }
    pub fn mixed_from_normed(&self, x: &D::Tensor) -> anyhow::Result<D::Tensor> {
        one::<D>(self.run(x, Mode::Mixed)?)
    }
    pub fn hc_norm_quarter(&self, x: &D::Tensor) -> anyhow::Result<D::Tensor> {
        one::<D>(self.run(x, Mode::Norm)?)
    }
    pub fn norm_scale_q(&self) -> anyhow::Result<D::Tensor> {
        one::<D>(execute(
            &self.context,
            Key::Scale(self.hc_count as i32),
            &[&self.hc_norm.weight],
            || Ok(model::norm_scale(self.hc_count as i32)),
        )?)
    }
    pub fn mix_from_normed(
        &self,
        x: &D::Tensor,
        has_inject: bool,
    ) -> anyhow::Result<(D::Tensor, D::Tensor)> {
        let mut out = self.run(x, Mode::Mix(has_inject))?.into_iter();
        Ok((out.next().unwrap(), out.next().unwrap()))
    }
    pub fn mix_from_normed_captured(
        &self,
        x: &D::Tensor,
        has_inject: bool,
    ) -> anyhow::Result<(D::Tensor, D::Tensor)> {
        let mut out = self
            .run_with_capture_law(x, Mode::Mix(has_inject), true)?
            .into_iter();
        Ok((out.next().unwrap(), out.next().unwrap()))
    }
    pub fn mix_with_inject(
        &self,
        x: &D::Tensor,
    ) -> anyhow::Result<(D::Tensor, D::Tensor, D::Tensor)> {
        let mut out = self.run(x, Mode::WithInject)?.into_iter();
        Ok((
            out.next().unwrap(),
            out.next().unwrap(),
            out.next().unwrap(),
        ))
    }
}
pub fn inject<D: BindingDevice>(
    context: &DeviceContext<D>,
    residual: &D::Tensor,
    output: &D::Tensor,
    inject: &D::Tensor,
) -> anyhow::Result<D::Tensor> {
    one::<D>(execute(
        context,
        Key::Inject(output.shape().to_vec()),
        &[residual, output, inject],
        || model::inject(output.shape()),
    )?)
}

pub fn inject_norm<D: BindingDevice>(
    context: &DeviceContext<D>,
    residual: &D::Tensor,
    out: Option<&D::Tensor>,
    inject: Option<&D::Tensor>,
    scale: &D::Tensor,
    hidden: i32,
    streams: i32,
    epsilon: f32,
) -> anyhow::Result<(D::Tensor, D::Tensor)> {
    let enabled = out.is_some() && inject.is_some();
    let mut values = execute(
        context,
        Key::InjectNorm(
            residual.shape().to_vec(),
            hidden,
            streams,
            epsilon.to_bits(),
            enabled,
        ),
        &[
            residual,
            out.unwrap_or(residual),
            inject.unwrap_or(residual),
            scale,
        ],
        || {
            model::inject_norm(
                residual.shape(),
                hidden,
                streams,
                epsilon.to_bits(),
                enabled,
            )
        },
    )?
    .into_iter();
    Ok((values.next().unwrap(), values.next().unwrap()))
}

impl<D: BindingDevice> GatedResidual<D> {
    pub fn finish(
        &self,
        residual: &D::Tensor,
        out: Option<&D::Tensor>,
        inject: Option<&D::Tensor>,
        epsilon: f32,
    ) -> anyhow::Result<(D::Tensor, D::Tensor)> {
        self.finish_with_capture_law(residual, out, inject, epsilon, false)
    }
    pub fn finish_captured(
        &self,
        residual: &D::Tensor,
        out: Option<&D::Tensor>,
        inject: Option<&D::Tensor>,
        epsilon: f32,
    ) -> anyhow::Result<(D::Tensor, D::Tensor)> {
        self.finish_with_capture_law(residual, out, inject, epsilon, true)
    }
    fn finish_with_capture_law(
        &self,
        residual: &D::Tensor,
        out: Option<&D::Tensor>,
        inject: Option<&D::Tensor>,
        epsilon: f32,
        captured: bool,
    ) -> anyhow::Result<(D::Tensor, D::Tensor)> {
        let spec = self.spec();
        let enabled = out.is_some() && inject.is_some();
        let dummy = self.block_inject.as_ref().unwrap_or(&self.mix_down);
        let packed = self.decode_up.as_ref().unwrap_or(&self.mix_up);
        let mut out = execute(
            &self.context,
            Key::Finish(
                spec,
                residual.shape().to_vec(),
                epsilon.to_bits(),
                enabled,
                captured,
            ),
            &[
                residual,
                out.unwrap_or(residual),
                inject.unwrap_or(residual),
                &self.hc_norm.weight,
                &self.mix_down.weight,
                &self.mix_down.scales,
                &self.mix_down.biases,
                &self.mix_up.weight,
                &self.mix_up.scales,
                &self.mix_up.biases,
                &dummy.weight,
                &dummy.scales,
                &dummy.biases,
                &packed.weight,
                &packed.scales,
                &packed.biases,
            ],
            || {
                if captured {
                    model::finish_captured(spec, residual.shape(), epsilon.to_bits(), enabled)
                } else {
                    model::finish(spec, residual.shape(), epsilon.to_bits(), enabled)
                }
            },
        )?
        .into_iter();
        Ok((out.next().unwrap(), out.next().unwrap()))
    }
}
