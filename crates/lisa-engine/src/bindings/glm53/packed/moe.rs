use crate::{bindings::BindingDevice, execution::DeviceContext};
use lisa_backend::TensorAccess;
// Resident original payload binding and execution for the model-owned GLM FFN.
use crate::bindings::glm53::packed::projection::Projection;
use crate::{core::graph::PlanCache, weights::prepared::PreparedWeights};
use anyhow::{Context, Result, ensure};

use lisa_models::glm53::packed as model;
use std::cell::RefCell;
pub struct Mlp<D: BindingDevice> {
    spec: model::Mlp,
    plans: RefCell<PlanCache<D, (Vec<i32>, bool)>>,
    prompt_plans: RefCell<PlanCache<D, usize>>,
}
impl<D: BindingDevice> Mlp<D> {
    pub fn load_on<L>(
        context: DeviceContext<D>,
        w: &PreparedWeights<D::Tensor, L>,
        layer: usize,
    ) -> Result<Self> {
        let mut weights = Vec::new();
        let mut projections = Vec::new();
        for requirement in model::Mlp::requirements(layer) {
            let p = w
                .tensors
                .get(&requirement.name)
                .with_context(|| format!("missing {}", requirement.name))?;
            let a = match requirement.role {
                model::WeightRole::Projection => {
                    let p = Projection::<D>::load(w, &requirement.name)?;
                    projections.push(p.descriptor_on(context.device())?);
                    (*p).clone()
                }
                model::WeightRole::Plain => p.plain_view()?,
                model::WeightRole::Block(kind) => {
                    ensure!(
                        p.layout.projection == lisa_models::packed::Projection::Block(kind),
                        "unexpected block encoding for {}",
                        requirement.name
                    );
                    p.packed_bytes().clone()
                }
            };
            weights.push(a);
        }
        Ok(Self::bound(
            context,
            weights,
            model::Mlp::for_layer(
                layer,
                projections
                    .try_into()
                    .map_err(|_| anyhow::anyhow!("invalid GLM projection bindings"))?,
            ),
        ))
    }
    pub fn bound(context: DeviceContext<D>, weights: Vec<D::Tensor>, spec: model::Mlp) -> Self {
        Self {
            spec,
            prompt_plans: RefCell::new(PlanCache::bound(
                context.device().clone(),
                8,
                weights.clone(),
            )),
            plans: RefCell::new(PlanCache::bound(context.device().clone(), 16, weights)),
        }
    }
    fn run(&self, x: &D::Tensor, combine: bool) -> Result<lisa_backend::Outputs<D::Tensor>> {
        let mut plans = self.plans.borrow_mut();
        let key = (x.shape().to_vec(), combine);
        plans
            .try_get_or_prepare(key, || self.spec.graph(x.shape(), combine))?
            .run(&[x], &mut [])
    }
    pub fn parts(&self, x: &D::Tensor) -> Result<(Option<D::Tensor>, D::Tensor)> {
        let mut values = self.run(x, false)?.into_iter();
        let routed = if self.spec.sparse {
            Some(values.next().context("missing routed output")?)
        } else {
            None
        };
        Ok((routed, values.next().context("missing shared output")?))
    }
    pub fn forward(&self, x: &D::Tensor) -> Result<D::Tensor> {
        self.run(x, true)?.into_one().map_err(anyhow::Error::msg)
    }
    pub fn prompt_rows(&self, rows: &[D::Tensor]) -> Result<Vec<D::Tensor>> {
        ensure!(
            !rows.is_empty() && rows.len() <= 8,
            "invalid GLM FFN prompt block"
        );
        ensure!(
            rows.iter().all(|x| x.shape() == [1, 1, 4096]),
            "invalid GLM FFN prompt row"
        );
        self.prompt_plans
            .borrow_mut()
            .try_get_or_prepare(rows.len(), || self.spec.prompt_graph(rows.len()))?
            .run(&rows.iter().collect::<Vec<_>>(), &mut [])
            .map(|out| out.into_iter().collect())
    }
}
