use crate::{bindings::BindingDevice, execution::DeviceContext};
use lisa_backend::TensorAccess;
// Projection bindings and capability discovery; fusion policy is model-owned.
use crate::bindings::glm53::packed::projection::Projection;
use crate::core::graph::PlanCache;
use lisa_backend::{KernelKind, SparseF32Kind};

use lisa_models::glm53::{kda_projection as model, packed::Projection as P};
#[derive(Clone, Debug, PartialEq)]
enum Key {
    One(P, Vec<i32>),
    Inputs([P; 6], Vec<i32>, Option<usize>),
    Gates([P; 2], Vec<i32>, Vec<i32>, bool),
}
struct Plans<D: BindingDevice> {
    plans: PlanCache<D, Key>,
    fused: bool,
    triple: bool,
    paired: bool,
}
fn run<D: BindingDevice>(
    context: &DeviceContext<D>,
    key: Key,
    inputs: &[&D::Tensor],
) -> anyhow::Result<lisa_backend::Outputs<D::Tensor>> {
    context.with_slot(
        "glm53.kda_projection.plans",
        || std::cell::RefCell::new(None::<Plans<D>>),
        |slot| {
            let mut slot = slot.borrow_mut();
            if slot.is_none() {
                let device = context.device().clone();
                let has = |kind| {
                    device
                        .capabilities()
                        .iter()
                        .any(|c| c.kind == KernelKind::SparseF32(kind))
                };
                *slot = Some(Plans {
                    fused: has(SparseF32Kind::Bf16PackedSix),
                    triple: has(SparseF32Kind::Bf16TripleLinear),
                    paired: has(SparseF32Kind::Bf16PairLinear),
                    plans: PlanCache::new(device, 64),
                });
            }
            let p = slot.as_mut().unwrap();
            let recipe = key.clone();
            let (fused, triple, paired) = (p.fused, p.triple, p.paired);
            p.plans
                .try_get_or_prepare(key, || match recipe {
                    Key::One(p, shape) => model::one(p, &shape),
                    Key::Inputs(p, shape, groups) => {
                        model::inputs(p, &shape, fused, triple, groups)
                    }
                    Key::Gates(p, a, b, f32) => model::gates(p, &a, &b, f32, paired),
                })?
                .run(inputs, &mut [])
        },
    )
}
pub fn linear<D: BindingDevice>(
    context: &DeviceContext<D>,
    p: &Projection<D>,
    x: &D::Tensor,
) -> anyhow::Result<D::Tensor> {
    run(
        context,
        Key::One(p.descriptor_on(context.device())?, x.shape().to_vec()),
        &[p, x],
    )?
    .into_one()
    .map_err(anyhow::Error::msg)
}
pub fn inputs<D: BindingDevice>(
    context: &DeviceContext<D>,
    p: [&Projection<D>; 6],
    x: &D::Tensor,
    groups: Option<usize>,
) -> anyhow::Result<(
    D::Tensor,
    D::Tensor,
    D::Tensor,
    D::Tensor,
    D::Tensor,
    D::Tensor,
)> {
    let desc = p
        .iter()
        .map(|p| p.descriptor_on(context.device()))
        .collect::<anyhow::Result<Vec<_>>>()?
        .try_into()
        .ok()
        .unwrap();
    let mut out = run(
        context,
        Key::Inputs(desc, x.shape().to_vec(), groups),
        &[x, p[0], p[1], p[2], p[3], p[4], p[5]],
    )?
    .into_iter();
    Ok((
        out.next().unwrap(),
        out.next().unwrap(),
        out.next().unwrap(),
        out.next().unwrap(),
        out.next().unwrap(),
        out.next().unwrap(),
    ))
}
pub fn gates<D: BindingDevice>(
    context: &DeviceContext<D>,
    p: [&Projection<D>; 2],
    a: &D::Tensor,
    b: &D::Tensor,
) -> anyhow::Result<(D::Tensor, D::Tensor)> {
    let mut out = run(
        context,
        Key::Gates(
            [
                p[0].descriptor_on(context.device())?,
                p[1].descriptor_on(context.device())?,
            ],
            a.shape().to_vec(),
            b.shape().to_vec(),
            context.device().dtype(a) == Some(lisa_backend::ScalarType::F32)
                && context.device().dtype(b) == Some(lisa_backend::ScalarType::F32),
        ),
        &[a, b, p[0], p[1]],
    )?
    .into_iter();
    Ok((out.next().unwrap(), out.next().unwrap()))
}
