//! Weight binding and execution of model-owned four-device residual plans.
use crate::indexer_cache::CacheDevice;
use crate::{core::graph::PreparedGraph, weights::gguf::ResidentWeights};
use anyhow::{Context, Result};
use lisa_backend::StorageDevice;
use lisa_models::deepseek41::hyper as model;

pub struct Hc<D: CacheDevice + StorageDevice> {
    plan: PreparedGraph<D>,
}
pub struct Input<T> {
    pub split: T,
    pub collapsed: T,
    pub norm: T,
}
impl<D: CacheDevice + StorageDevice> Hc<D> {
    pub fn load<L>(
        weights: &ResidentWeights<D, L>,
        layer: usize,
        kind: &str,
        device: &D,
    ) -> Result<Self> {
        let mut bound = Vec::new();
        let names = model::requirements(layer, kind)?;
        let mixer = weights
            .tensors
            .get(&names[0])
            .context("missing residual mixer")?;
        for name in names {
            let w = weights
                .tensors
                .get(&name)
                .with_context(|| format!("missing {name}"))?;
            bound.push(super::deepseek41_projection::bound(w)?);
        }
        Self::bound(super::deepseek41_projection::matrix(mixer)?, bound, device)
    }
    pub fn bound(
        matrix: lisa_models::deepseek41::projection::Matrix,
        weights: Vec<D::Tensor>,
        device: &D,
    ) -> Result<Self> {
        Ok(Self {
            plan: PreparedGraph::new(device.clone(), model::input(matrix)?, weights)?,
        })
    }
    pub fn forward(&self, residual: &D::Tensor, previous: &D::Tensor) -> Result<Input<D::Tensor>> {
        let mut results = self.plan.run(&[residual, previous], &mut [])?.into_iter();
        Ok(Input {
            split: results.next().context("missing residual split")?,
            collapsed: results.next().context("missing collapsed residual")?,
            norm: results.next().context("missing residual norm")?,
        })
    }
}
pub struct ResidualPlans<D: CacheDevice + StorageDevice> {
    attention: PreparedGraph<D>,
    ffn: PreparedGraph<D>,
    head: PreparedGraph<D>,
}
impl<D: CacheDevice + StorageDevice> ResidualPlans<D> {
    pub fn new(device: &D) -> Result<Self> {
        let device = device.clone();
        Ok(Self {
            attention: PreparedGraph::new(device.clone(), model::expand(false, false), vec![])?,
            ffn: PreparedGraph::new(device.clone(), model::expand(true, true), vec![])?,
            head: PreparedGraph::new(device, model::head(), vec![])?,
        })
    }
    pub fn attention(
        &self,
        block: &D::Tensor,
        residual: &D::Tensor,
        split: &D::Tensor,
    ) -> Result<D::Tensor> {
        self.attention
            .run(&[block, residual, split], &mut [])?
            .into_one()
            .map_err(anyhow::Error::msg)
    }
    pub fn ffn(
        &self,
        block: &D::Tensor,
        add: &D::Tensor,
        residual: &D::Tensor,
        split: &D::Tensor,
    ) -> Result<D::Tensor> {
        self.ffn
            .run(&[block, add, residual, split], &mut [])?
            .into_one()
            .map_err(anyhow::Error::msg)
    }
    pub fn head(
        &self,
        residual: &D::Tensor,
        previous: &D::Tensor,
        norm: &D::Tensor,
    ) -> Result<D::Tensor> {
        self.head
            .run(&[residual, previous, norm], &mut [])?
            .into_one()
            .map_err(anyhow::Error::msg)
    }
}
