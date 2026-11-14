//! Checkpoint binding and execution of model-owned scalar attention plans.
use super::deepseek41_projection as projection;
use crate::bindings::deepseek41_state::State;
use crate::core::{
    capture::Capture,
    graph::{PlanCache, PreparedGraph},
};
use crate::storage::view;
use crate::{bindings::BindingDevice, weights::gguf::ResidentWeights};
use anyhow::{Context, Result, ensure};
use lisa_backend::TensorAccess;
use lisa_models::deepseek41::{attention as model, config::Config, state::Position};
use std::cell::RefCell;

pub struct Projections<D: BindingDevice> {
    device: D,
    pub qr: D::Tensor,
    pub kv: D::Tensor,
    pub q: D::Tensor,
}
impl<D: BindingDevice> Projections<D> {
    pub fn row(&self, row: usize) -> Result<Self> {
        Ok(Self {
            device: self.device.clone(),
            qr: view(&self.device, &self.qr, row * 1280, &[1, 1280])?,
            kv: view(&self.device, &self.kv, row * 512, &[1, 1, 512])?,
            q: view(&self.device, &self.q, row * 64 * 512, &[1, 64, 512])?,
        })
    }
}
pub struct Scratch<D: BindingDevice> {
    pub arrays: Vec<D::Tensor>,
}
impl<D: BindingDevice> Scratch<D> {
    pub fn new(device: &D) -> Result<Self> {
        Ok(Self {
            arrays: PreparedGraph::new(device.clone(), model::scratch(), vec![])?
                .run(&[], &mut [])?
                .into_iter()
                .collect(),
        })
    }
}
pub struct Attention<D: BindingDevice> {
    device: D,
    spec: model::Spec,
    project: RefCell<PlanCache<D, i32>>,
    forward: RefCell<PlanCache<D, (usize, bool, bool)>>,
}
impl<D: BindingDevice> Attention<D> {
    pub fn load<L>(
        weights: &ResidentWeights<D, L>,
        layer: usize,
        config: &Config,
        device: &D,
    ) -> Result<Self> {
        let names = model::requirements(layer, config);
        let mut matrices = [None; 15];
        let mut bound = Vec::new();
        let placeholder = projection::plain_view(
            weights
                .tensors
                .get(names[1].as_ref().unwrap())
                .context("missing query norm")?,
        )?;
        for (i, name) in names.iter().enumerate() {
            if let Some(name) = name {
                let w = weights
                    .tensors
                    .get(name)
                    .with_context(|| format!("missing {name}"))?;
                if model::is_projection(i) {
                    matrices[i] = Some(projection::matrix(w)?);
                }
                bound.push(projection::bound(w)?);
            } else {
                bound.push(placeholder.clone());
            }
        }
        let spec = model::Spec {
            layer,
            ratio: config.compress_ratios[layer] as usize,
            matrices,
            compressor: names[8].is_some(),
            gate: names[9].is_some(),
            indexer: names[13].is_some(),
        };
        Self::bound(spec, bound, device)
    }
    pub fn bound(spec: model::Spec, weights: Vec<D::Tensor>, device: &D) -> Result<Self> {
        let device = device.clone();
        Ok(Self {
            device: device.clone(),
            spec,
            project: RefCell::new(PlanCache::bound(device.clone(), 8, weights.clone())),
            forward: RefCell::new(PlanCache::bound(device, 64, weights)),
        })
    }
    pub fn project(&self, norm: &D::Tensor, _device: &D) -> Result<Projections<D>> {
        ensure!(norm.rank() == 2, "attention projection requires a matrix");
        let mut plans = self.project.borrow_mut();
        let mut out = plans
            .try_get_or_prepare(norm.dim(0), || self.spec.project(norm.dim(0)))?
            .run(&[norm], &mut [])?
            .into_iter();
        Ok(Projections {
            device: self.device.clone(),
            qr: out.next().context("missing query rank")?,
            kv: out.next().context("missing kv")?,
            q: out.next().context("missing query")?,
        })
    }
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        norm: &D::Tensor,
        projected: Projections<D>,
        state: &mut State<D>,
        selected: &mut Option<D::Tensor>,
        candidates: &mut Option<D::Tensor>,
        scratch: &mut Scratch<D>,
        rope: &D::Tensor,
        capture: Option<&Capture<D::Tensor>>,
        _device: &D,
    ) -> Result<D::Tensor> {
        let pos = state.position;
        let p = Position::new(self.spec.layer, self.spec.ratio, pos)?;
        let cache = &mut state.owners[p.owner];
        let capture = capture.filter(|c| c.wants(self.spec.layer));
        let mirror = cache.index_bf16.is_some();
        let key = (pos, mirror, capture.is_some());
        let mut plans = self.forward.borrow_mut();
        let plan = plans.try_get_or_prepare(key, || {
            Ok(self.spec.forward(pos, mirror, capture.is_some())?.0)
        })?;
        if self.spec.requires_candidates(pos)? {
            ensure!(
                candidates.is_some(),
                "compressed candidate list was not published"
            );
        }
        if self.spec.requires_selection(pos)? {
            ensure!(selected.is_some(), "compressed selection was not published");
        }
        let mut args = vec![
            norm,
            &projected.qr,
            &projected.kv,
            &projected.q,
            &state.raw[self.spec.layer],
            &cache.compressed,
            &cache.index,
            cache.index_bf16.as_ref().unwrap_or(&cache.index),
            &cache.previous_kv,
            &cache.previous_score,
            candidates.as_ref().unwrap_or(&cache.index),
            selected.as_ref().unwrap_or(&cache.index),
            rope,
        ];
        args.extend(scratch.arrays.iter());
        let mut outputs = plan.run(&args, &mut [])?.into_iter();
        let out = outputs.next().context("missing attention output")?;
        cache.previous_kv = outputs.next().context("missing compressor carry")?;
        cache.previous_score = outputs.next().context("missing compressor score")?;
        let candidate = outputs.next().context("missing candidate output")?;
        let selection = outputs.next().context("missing selection output")?;
        if self.spec.publishes_candidates(pos)? {
            *candidates = Some(candidate);
        }
        if self.spec.publishes_selection(pos)? {
            *selected = Some(selection);
        }
        if let Some(capture) = capture {
            for (name, value) in self
                .spec
                .forward(pos, mirror, true)?
                .1
                .into_iter()
                .zip(outputs)
            {
                capture.store_ephemeral(
                    &format!("p{pos}.l{}.attention.{name}", self.spec.layer),
                    &value,
                );
            }
        }
        Ok(out)
    }
}
pub fn rope_table<D: BindingDevice>(position: u32, device: &D) -> Result<D::Tensor> {
    PreparedGraph::new(device.clone(), model::rope_table(position), vec![])?
        .run(&[], &mut [])?
        .into_one()
        .map_err(anyhow::Error::msg)
}
