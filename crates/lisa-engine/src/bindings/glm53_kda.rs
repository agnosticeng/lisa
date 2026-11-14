use crate::{bindings::BindingDevice, execution::DeviceContext};
// Tensor bindings for the model-owned recurrent stages.
use crate::core::graph::PlanCache;

use lisa_models::glm53::kda::{Spec, Stage};
struct Plans<D: BindingDevice> {
    cache: PlanCache<D, (Spec, Stage)>,
}
pub fn run<D: BindingDevice, const N: usize>(
    context: &DeviceContext<D>,
    spec: Spec,
    stage: Stage,
    inputs: &[&D::Tensor],
) -> anyhow::Result<[D::Tensor; N]> {
    run_many(context, spec, stage, inputs)?
        .try_into()
        .map_err(|_| anyhow::anyhow!("recurrent plan output arity differs"))
}
pub fn run_many<D: BindingDevice>(
    context: &DeviceContext<D>,
    spec: Spec,
    stage: Stage,
    inputs: &[&D::Tensor],
) -> anyhow::Result<Vec<D::Tensor>> {
    context.with_slot(
        "glm53.kda.plans",
        || std::cell::RefCell::new(None::<Plans<D>>),
        |p| {
            let mut p = p.borrow_mut();
            if p.is_none() {
                *p = Some(Plans {
                    cache: PlanCache::new(context.device().clone(), 64),
                });
            }
            Ok(p.as_mut()
                .unwrap()
                .cache
                .try_get_or_prepare((spec, stage), || spec.graph(stage))?
                .run(inputs, &mut [])?
                .into_iter()
                .collect())
        },
    )
}
struct ForwardPlans<D: BindingDevice> {
    device: D,
    fused: bool,
    triple: bool,
    paired: bool,
    plans: std::collections::VecDeque<(
        lisa_models::glm53::kda::Forward,
        crate::core::graph::PreparedGraph<D>,
        Vec<String>,
    )>,
}
pub fn forward<D: BindingDevice>(
    context: &DeviceContext<D>,
    mut key: lisa_models::glm53::kda::Forward,
    inputs: &[&D::Tensor],
) -> anyhow::Result<([D::Tensor; 3], Vec<(String, D::Tensor)>)> {
    use lisa_backend::{Device, KernelKind, SparseF32Kind};
    context.with_slot(
        "glm53.kda.forward",
        || std::cell::RefCell::new(None::<ForwardPlans<D>>),
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
                *slot = Some(ForwardPlans {
                    fused: has(SparseF32Kind::Bf16PackedSix),
                    triple: has(SparseF32Kind::Bf16TripleLinear),
                    paired: has(SparseF32Kind::Bf16PairLinear),
                    device,
                    plans: Default::default(),
                });
            }
            let p = slot.as_mut().unwrap();
            key.fused = p.fused;
            key.triple = p.triple;
            key.paired = p.paired;
            let index = if let Some(i) = p.plans.iter().position(|(k, _, _)| k == &key) {
                i
            } else {
                let plan = key.plan()?;
                let graph =
                    crate::core::graph::PreparedGraph::new(p.device.clone(), plan.graph, vec![])?;
                if p.plans.len() == 64 {
                    p.plans.pop_front();
                }
                p.plans.push_back((key, graph, plan.observations));
                p.plans.len() - 1
            };
            let (_, plan, names) = &p.plans[index];
            let mut out = plan.run(inputs, &mut [])?.into_iter();
            let main = [
                out.next().unwrap(),
                out.next().unwrap(),
                out.next().unwrap(),
            ];
            let observations = names.iter().cloned().zip(out).collect();
            Ok((main, observations))
        },
    )
}
