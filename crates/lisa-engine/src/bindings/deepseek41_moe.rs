//! Bind original resident payloads to the model-owned scalar FFN program.
use crate::indexer_cache::CacheDevice;
use crate::{core::graph::PlanCache, weights::gguf::ResidentWeights};
use anyhow::{Context, Result, ensure};
use lisa_backend::{StorageDevice, TensorAccess};
use lisa_models::{deepseek41::moe as model, packed::Projection};
use std::cell::RefCell;
pub struct Moe<D: CacheDevice + StorageDevice> {
    spec: model::Spec,
    plans: RefCell<PlanCache<D, i32>>,
}
pub struct Routing<T> {
    pub probabilities: T,
    pub selected: T,
    pub weights: T,
}
pub struct Output<T> {
    pub routed: T,
    pub shared: T,
    pub routing: Routing<T>,
}
impl<D: CacheDevice + StorageDevice> Moe<D> {
    pub fn load<L>(
        weights: &ResidentWeights<D, L>,
        layer: usize,
        experts: usize,
        device: &D,
    ) -> Result<Self> {
        let mut arrays = Vec::new();
        let mut projections = Vec::new();
        for (slot, (name, shape)) in model::requirements(layer, i32::try_from(experts)?)
            .into_iter()
            .enumerate()
        {
            let w = weights
                .tensors
                .get(&name)
                .with_context(|| format!("missing {name}"))?;
            ensure!(
                w.info
                    .dimensions
                    .iter()
                    .rev()
                    .copied()
                    .eq(shape.iter().map(|&d| d as u64)),
                "invalid FFN weight geometry for {name}"
            );
            let kind = super::packed::projection(w.info.kind)?;
            if slot == 1 {
                ensure!(
                    kind == Projection::Plain,
                    "router bias must retain unquantized storage"
                );
            } else {
                projections.push(kind);
            }
            arrays.push(if kind == Projection::Plain {
                w.plain_view()?
            } else {
                w.packed_bytes().clone()
            });
        }
        Self::bound(
            arrays,
            model::Spec {
                experts: i32::try_from(experts)?,
                projections: projections
                    .try_into()
                    .map_err(|_| anyhow::anyhow!("invalid FFN bindings"))?,
            },
            device,
        )
    }
    pub fn bound(weights: Vec<D::Tensor>, spec: model::Spec, device: &D) -> Result<Self> {
        // Validate contracts before any forward; actual kernels remain lazily prepared.
        spec.graph(1)?;
        Ok(Self {
            spec,
            plans: RefCell::new(PlanCache::bound(device.clone(), 8, weights)),
        })
    }
    pub fn forward(&self, input: &D::Tensor) -> Result<Output<D::Tensor>> {
        let rows = self.spec.input_rows(input.shape())?;
        let mut plans = self.plans.borrow_mut();
        let mut outputs = plans
            .try_get_or_prepare(rows, || self.spec.graph(rows))?
            .run(&[input], &mut [])?
            .into_iter();
        Ok(Output {
            routed: outputs.next().context("missing routed result")?,
            shared: outputs.next().context("missing shared result")?,
            routing: Routing {
                probabilities: outputs.next().context("missing route probabilities")?,
                selected: outputs.next().context("missing route selection")?,
                weights: outputs.next().context("missing route weights")?,
            },
        })
    }
}
