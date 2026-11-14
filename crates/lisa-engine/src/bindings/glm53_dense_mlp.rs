use crate::tensor_ops::{TensorLaw, TensorOps};
// Affine expert bindings; execution order and arithmetic are model-owned.
use super::{glm53_affine_projection::descriptor, glm53_policy::native_gate_up_enabled};
use crate::{
    bindings::BindingDevice,
    core::{capture::Capture, graph::PlanCache},
    execution::DeviceContext,
    weights::{affine::QuantizedLinear, source::TensorSource},
};
use lisa_backend::{Device, ScalarType, TensorAccess, TensorOperation};
use lisa_models::glm53::dense_mlp::{self as model, Request as R};
fn run<D: BindingDevice>(
    context: &DeviceContext<D>,
    request: R,
    inputs: &[&D::Tensor],
) -> anyhow::Result<Vec<D::Tensor>> {
    context.with_slot(
        "glm53.dense_mlp.plans",
        || std::cell::RefCell::new(PlanCache::<D, R>::new(context.device().clone(), 64)),
        |slot| {
            let recipe = request.clone();
            Ok(slot
                .borrow_mut()
                .try_get_or_prepare(request, || recipe.graph())?
                .run(inputs, &mut [])?
                .into_iter()
                .collect())
        },
    )
}
fn cast<D: BindingDevice>(
    context: &DeviceContext<D>,
    x: &D::Tensor,
    dtype: ScalarType,
) -> anyhow::Result<D::Tensor> {
    TensorOps::new(context.device(), TensorLaw::Native).op(TensorOperation::Cast(dtype), &[x])
}
fn arrays<T>(w: &QuantizedLinear<T>) -> [&T; 3] {
    [&w.weight, &w.scales, &w.biases]
}
fn linear<T>(mut a: impl Iterator<Item = T>, w: &QuantizedLinear<T>) -> QuantizedLinear<T> {
    QuantizedLinear {
        weight: a.next().unwrap(),
        scales: a.next().unwrap(),
        biases: a.next().unwrap(),
        bits: w.bits,
        group_size: w.group_size,
    }
}
/// Concatenate gate and up output rows so one affine-quant projection computes
/// both results. Real layer-0 weights match the separate DS4 production path
/// for decode and prefill shapes; the layer-45 shared expert has the same
/// checkpoint-backed exactness check.
pub fn fuse_gate_up<D: BindingDevice>(
    context: &DeviceContext<D>,
    gate: &QuantizedLinear<D::Tensor>,
    up: &QuantizedLinear<D::Tensor>,
) -> anyhow::Result<QuantizedLinear<D::Tensor>> {
    anyhow::ensure!(
        gate.bits == up.bits && gate.group_size == up.group_size,
        "gate/up quantization mismatch"
    );
    let a = [arrays(gate), arrays(up)].concat();
    Ok(linear(run(context, R::JoinWeights, &a)?.into_iter(), gate))
}

/// Split a fused gate_up projection output [.., 2*inter] into (gate, up).
pub fn split_gate_up<D: BindingDevice>(
    context: &DeviceContext<D>,
    x: &D::Tensor,
    inter: usize,
) -> anyhow::Result<(D::Tensor, D::Tensor)> {
    let mut out = run(
        context,
        R::SplitOutput {
            rank: x.rank(),
            inter: inter as i32,
        },
        &[x],
    )?
    .into_iter();
    Ok((out.next().unwrap(), out.next().unwrap()))
}

/// Clamped SwiGLU: `silu(min(gate, limit)) * clip(up, -limit, limit)`.
pub fn clamped_swiglu<D: BindingDevice>(
    context: &DeviceContext<D>,
    up: &D::Tensor,
    gate: &D::Tensor,
    limit: f32,
) -> anyhow::Result<D::Tensor> {
    Ok(run(
        context,
        R::SwiGlu {
            limit: limit.to_bits(),
        },
        &[up, gate],
    )?
    .remove(0))
}

/// The paired Q8 projection is the production path. The separate mode keeps
/// the original two-dispatch schedule available for controlled end-to-end
/// measurements; it is selected once while loading the model.
pub enum GateUpProjection<T> {
    Paired {
        projection: QuantizedLinear<T>,
        gate: QuantizedLinear<T>,
        up: QuantizedLinear<T>,
    },
    Separate {
        gate: QuantizedLinear<T>,
        up: QuantizedLinear<T>,
    },
}

pub struct ActivatedProjection<T> {
    pub projections: Option<(T, T)>,
    pub activated: T,
}

impl<T: lisa_backend::TensorAccess + Send + Sync + 'static> GateUpProjection<T> {
    pub fn load_mode() -> anyhow::Result<bool> {
        match std::env::var("LISA_GLM53_Q8_GATE_UP").as_deref() {
            Ok("separate") => Ok(false),
            Ok("paired") | Err(std::env::VarError::NotPresent) => Ok(true),
            Ok(value) => {
                anyhow::bail!("LISA_GLM53_Q8_GATE_UP must be `paired` or `separate`, got {value:?}")
            }
            Err(error) => anyhow::bail!("cannot read LISA_GLM53_Q8_GATE_UP: {error}"),
        }
    }

    pub fn from_parts<D: BindingDevice<Tensor = T>>(
        context: &DeviceContext<D>,
        gate: QuantizedLinear<T>,
        up: QuantizedLinear<T>,
        paired: bool,
    ) -> anyhow::Result<Self> {
        if paired {
            let projection = fuse_gate_up(context, &gate, &up)?;
            let (gate, up) = split_gate_up_linear(context, &projection, projection.dims_out() / 2)?;
            Ok(Self::Paired {
                projection,
                gate,
                up,
            })
        } else {
            Ok(Self::Separate { gate, up })
        }
    }

    #[cfg(any(test, feature = "diagnostics"))]
    pub fn inter(&self) -> usize {
        match self {
            Self::Paired { projection, .. } => projection.dims_out() / 2,
            Self::Separate { gate, .. } => gate.dims_out(),
        }
    }

    #[cfg(any(test, feature = "diagnostics"))]
    pub fn split_linears(&self) -> (&QuantizedLinear<T>, &QuantizedLinear<T>) {
        match self {
            Self::Paired { gate, up, .. } | Self::Separate { gate, up } => (gate, up),
        }
    }

    #[cfg(any(test, feature = "diagnostics"))]
    pub fn forward<D: BindingDevice<Tensor = T>>(
        &self,
        context: &DeviceContext<D>,
        x: &T,
    ) -> anyhow::Result<(T, T)> {
        let (g, u, p) = self.bindings();
        let native = super::glm53_policy::glm53_q8_mpp_prefill_override() != Some(false);
        let request = R::Projections {
            gate: descriptor(context, g, x, native)?,
            up: descriptor(context, u, x, native)?,
            paired: p.map(|w| descriptor(context, w, x, native)).transpose()?,
        };
        let mut out = run(context, request, &self.inputs(x))?.into_iter();
        Ok((out.next().unwrap(), out.next().unwrap()))
    }

    pub fn activated<D: BindingDevice<Tensor = T>>(
        &self,
        context: &DeviceContext<D>,
        x: &T,
        limit: f32,
        capture: bool,
    ) -> anyhow::Result<ActivatedProjection<T>> {
        let (g, u, p) = self.bindings();
        let native = super::glm53_policy::glm53_q8_mpp_prefill_override() != Some(false);
        let request = R::Activated {
            gate: descriptor(context, g, x, native)?,
            up: descriptor(context, u, x, native)?,
            paired: p.map(|w| descriptor(context, w, x, native)).transpose()?,
            native: native_gate_up_enabled(),
            capture,
            limit: limit.to_bits(),
        };
        let out = run(context, request, &self.inputs(x))?;
        let count = out.len();
        let mut out = out.into_iter();
        Ok(if count == 1 {
            ActivatedProjection {
                projections: None,
                activated: out.next().unwrap(),
            }
        } else {
            ActivatedProjection {
                projections: Some((out.next().unwrap(), out.next().unwrap())),
                activated: out.next().unwrap(),
            }
        })
    }
    pub fn bindings(
        &self,
    ) -> (
        &QuantizedLinear<T>,
        &QuantizedLinear<T>,
        Option<&QuantizedLinear<T>>,
    ) {
        match self {
            Self::Paired {
                projection,
                gate,
                up,
            } => (gate, up, Some(projection)),
            Self::Separate { gate, up } => (gate, up, None),
        }
    }
    pub fn inputs<'a>(&'a self, x: &'a T) -> Vec<&'a T> {
        let (g, u, p) = self.bindings();
        [
            vec![x],
            arrays(g).to_vec(),
            arrays(u).to_vec(),
            arrays(p.unwrap_or(g)).to_vec(),
        ]
        .concat()
    }
}

/// Clamped GLM MLP. Leading dense layers use a concatenated gate/up Q8
/// projection. Shared experts use the same exact paired-projection schedule.
pub struct DenseMlp<D: BindingDevice> {
    pub context: DeviceContext<D>,
    pub gate_up: GateUpProjection<D::Tensor>,
    pub down: QuantizedLinear<D::Tensor>,
    pub limit: f32,
}

impl<D: BindingDevice> DenseMlp<D> {
    /// The projection mode captured at load, independent of later environment changes.
    pub fn paired_storage(&self) -> bool {
        matches!(&self.gate_up, GateUpProjection::Paired { .. })
    }

    pub fn load_on<S: TensorSource<D::Tensor>>(
        context: &DeviceContext<D>,
        w: &mut S,
        prefix: &str,
        limit: f32,
    ) -> anyhow::Result<Self> {
        let (gate, up, down) = Self::load_parts(context, w, prefix)?;
        Ok(Self {
            context: context.clone(),
            gate_up: GateUpProjection::from_parts(
                context,
                gate,
                up,
                GateUpProjection::<D::Tensor>::load_mode()?,
            )?,
            down,
            limit,
        })
    }

    pub fn load_shared_on<S: TensorSource<D::Tensor>>(
        context: &DeviceContext<D>,
        w: &mut S,
        prefix: &str,
        limit: f32,
    ) -> anyhow::Result<Self> {
        Self::load_on::<S>(context, w, prefix, limit)
    }

    pub fn load_parts<S: TensorSource<D::Tensor>>(
        context: &DeviceContext<D>,
        w: &mut S,
        prefix: &str,
    ) -> anyhow::Result<(
        QuantizedLinear<D::Tensor>,
        QuantizedLinear<D::Tensor>,
        QuantizedLinear<D::Tensor>,
    )> {
        let mut gate =
            QuantizedLinear::load(w, &format!("{prefix}.{}", model::PROJECTION_NAMES[0]), "")?;
        let mut up =
            QuantizedLinear::load(w, &format!("{prefix}.{}", model::PROJECTION_NAMES[1]), "")?;
        let mut down =
            QuantizedLinear::load(w, &format!("{prefix}.{}", model::PROJECTION_NAMES[2]), "")?;
        for q in [&mut gate, &mut up, &mut down] {
            q.set_quant(model::GROUP_SIZE, model::BITS);
            if context.device().dtype(&q.scales) != Some(ScalarType::BF16) {
                q.scales = cast(context, &q.scales, ScalarType::F32)?;
                q.biases = cast(context, &q.biases, ScalarType::F32)?;
            }
        }
        Ok((gate, up, down))
    }

    pub fn forward(&self, x: &D::Tensor) -> anyhow::Result<D::Tensor> {
        self.forward_with_capture(x, None, 0)
    }

    pub fn forward_with_capture(
        &self,
        x: &D::Tensor,
        capture: Option<&Capture<D::Tensor>>,
        layer: usize,
    ) -> anyhow::Result<D::Tensor> {
        use model::Step;
        let projection_capture =
            capture.filter(|c| c.accepts_key(&model::capture_key(layer, Step::CaptureProjection)));
        let mut activated = None;
        let mut projections = None;
        let mut output = None;
        for step in model::FORWARD {
            match step {
                Step::CaptureInput => {
                    if let Some(c) = capture {
                        c.store(&model::capture_key(layer, step), x)
                    }
                }
                Step::Activate => {
                    let a = self.gate_up.activated(
                        &self.context,
                        x,
                        self.limit,
                        projection_capture.is_some(),
                    )?;
                    activated = Some(a.activated);
                    projections = a.projections;
                }
                Step::CaptureProjection => {
                    if let Some(c) = projection_capture {
                        let (g, u) = projections
                            .as_ref()
                            .ok_or_else(|| anyhow::anyhow!("capture projections missing"))?;
                        let joined = run(&self.context, R::CaptureJoin, &[g, u])?.remove(0);
                        c.store(&model::capture_key(layer, step), &joined);
                    }
                }
                Step::CaptureActivation => {
                    if let Some(c) = capture {
                        c.store(
                            &model::capture_key(layer, step),
                            activated
                                .as_ref()
                                .ok_or_else(|| anyhow::anyhow!("activation missing"))?,
                        );
                    }
                }
                Step::Down => {
                    output = Some(
                        self.forward_projection(
                            &self.down,
                            activated
                                .as_ref()
                                .ok_or_else(|| anyhow::anyhow!("activation missing"))?,
                        )?,
                    )
                }
                Step::CaptureOutput => {
                    if let Some(c) = capture {
                        c.store(
                            &model::capture_key(layer, step),
                            output
                                .as_ref()
                                .ok_or_else(|| anyhow::anyhow!("output missing"))?,
                        );
                    }
                }
            }
        }
        output.ok_or_else(|| anyhow::anyhow!("expert plan omitted output"))
    }

    /// Use DS4's row-count-specific Q8 projection schedule over the original
    /// affine storage: M=1, small-row and matrix paths keep their own arithmetic.
    pub fn forward_projection(
        &self,
        projection: &QuantizedLinear<D::Tensor>,
        x: &D::Tensor,
    ) -> anyhow::Result<D::Tensor> {
        super::glm53_affine_projection::forward(
            &self.context,
            projection,
            x,
            super::glm53_policy::glm53_q8_mpp_prefill_override() != Some(false),
        )
    }
}

pub fn split_gate_up_linear<D: BindingDevice>(
    context: &DeviceContext<D>,
    w: &QuantizedLinear<D::Tensor>,
    inter: usize,
) -> anyhow::Result<(QuantizedLinear<D::Tensor>, QuantizedLinear<D::Tensor>)> {
    let mut out = run(
        context,
        R::SplitWeights {
            inter: inter as i32,
        },
        &arrays(w),
    )?
    .into_iter();
    let a = linear(out.by_ref().take(3), w);
    let b = linear(out, w);
    Ok((a, b))
}
