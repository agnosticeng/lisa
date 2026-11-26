//! One device-independent binding for the model-owned Laya plans.
use super::Laya;
use crate::core::graph::PreparedGraph;
use anyhow::{Context, Result, ensure};
use lisa_backend::{HostTensor, ImportOptions, ReadbackDevice, ScalarType};
use lisa_models::laya::plan::{self, Arithmetic, Plan};
use std::cell::RefCell;

pub(super) trait Execution {
    fn forward(
        &self,
        model: &Laya,
        ids: &[u32],
        qtype: i32,
        markers: &[i32],
    ) -> Result<(Vec<f32>, Vec<f32>)>;
}
struct Plans<D: ReadbackDevice> {
    encoder: Option<(usize, PreparedGraph<D>)>,
    scorer: Option<(usize, usize, PreparedGraph<D>)>,
    action: Option<PreparedGraph<D>>,
}
pub(super) struct Bound<D: ReadbackDevice> {
    device: D,
    arithmetic: Arithmetic,
    plans: RefCell<Plans<D>>,
    weights: RefCell<Box<dyn crate::weights::source::TensorSource<D::Tensor>>>,
    question_types: i32,
}
impl<D: ReadbackDevice + Clone> Bound<D>
where
    D::Tensor: Clone,
{
    pub fn new(
        device: D,
        arithmetic: Arithmetic,
        weights: Box<dyn crate::weights::source::TensorSource<D::Tensor>>,
        question_types: i32,
    ) -> Self {
        Self {
            device,
            arithmetic,
            weights: RefCell::new(weights),
            question_types,
            plans: RefCell::new(Plans {
                encoder: None,
                scorer: None,
                action: None,
            }),
        }
    }
    fn indices(&self, values: &[i32], shape: &[i32]) -> Result<D::Tensor> {
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        Ok(self.device.pull(
            HostTensor {
                bytes: &bytes,
                shape,
                dtype: ScalarType::I32,
            }
            .into(),
            ImportOptions::default(),
        )?)
    }
    fn prepare(&self, plan: Plan) -> Result<PreparedGraph<D>> {
        let weights = plan
            .weights
            .iter()
            .map(|name| self.weights.borrow_mut().get(name))
            .collect::<Result<Vec<_>>>()?;
        PreparedGraph::new(self.device.clone(), plan.graph, weights)
    }
}
impl<D: ReadbackDevice + Clone> Execution for Bound<D>
where
    D::Tensor: Clone,
{
    fn forward(
        &self,
        model: &Laya,
        ids: &[u32],
        qtype: i32,
        markers: &[i32],
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        ensure!(
            !ids.is_empty() && ids.len() <= model.config().max_position_embeddings,
            "invalid Laya sequence length"
        );
        ensure!(
            ids.iter()
                .all(|&id| (id as usize) < model.config().vocab_size),
            "Laya token exceeds vocabulary"
        );
        ensure!(
            !markers.is_empty() && markers.iter().all(|&p| p >= 0 && (p as usize) < ids.len()),
            "invalid Laya marker positions"
        );
        ensure!(
            qtype >= 0 && qtype < self.question_types,
            "invalid Laya question type"
        );
        let sequence = i32::try_from(ids.len())?;
        let input = self.indices(
            &ids.iter().map(|&i| i as i32).collect::<Vec<_>>(),
            &[sequence],
        )?;
        let mut cache = self.plans.borrow_mut();
        if cache.encoder.as_ref().is_none_or(|(n, _)| *n != ids.len()) {
            cache.encoder = Some((
                ids.len(),
                self.prepare(plan::encoder_with(
                    model.config(),
                    sequence,
                    self.arithmetic,
                ))?,
            ));
        }
        let hidden = cache
            .encoder
            .as_ref()
            .unwrap()
            .1
            .run(&[&input], &mut [])?
            .into_one()
            .map_err(anyhow::Error::msg)?;
        let qtype = self.indices(&[qtype], &[1])?;
        let markers_tensor = self.indices(markers, &[i32::try_from(markers.len())?])?;
        if cache
            .scorer
            .as_ref()
            .is_none_or(|(n, k, _)| *n != ids.len() || *k != markers.len())
        {
            cache.scorer = Some((
                ids.len(),
                markers.len(),
                self.prepare(plan::scorer_with(
                    model.config(),
                    sequence,
                    i32::try_from(markers.len())?,
                    self.arithmetic,
                ))?,
            ));
        }
        let mut output = cache
            .scorer
            .as_ref()
            .unwrap()
            .2
            .run(&[&hidden, &qtype, &markers_tensor], &mut [])?
            .into_iter();
        let logits = output.next().context("missing logits")?;
        let probabilities = output.next().context("missing probabilities")?;
        let pooled = output.next().context("missing pooled state")?;
        // Native execution preserves the original readback/submission boundary.
        let native_logits = if self.arithmetic == Arithmetic::Native {
            Some(self.device.read_f32(&logits)?)
        } else {
            None
        };
        let features = plan::action_features(&self.device.read_f32(&probabilities)?)?;
        let bytes: Vec<u8> = features.iter().flat_map(|v| v.to_le_bytes()).collect();
        let features = self.device.pull(
            HostTensor {
                bytes: &bytes,
                shape: &[1, 4],
                dtype: ScalarType::F32,
            }
            .into(),
            ImportOptions::default(),
        )?;
        if cache.action.is_none() {
            cache.action = Some(self.prepare(plan::action_with(self.arithmetic))?);
        }
        let action = cache
            .action
            .as_ref()
            .unwrap()
            .run(&[&pooled, &features], &mut [])?
            .into_one()
            .map_err(anyhow::Error::msg)?;
        Ok((
            match native_logits {
                Some(v) => v,
                None => self.device.read_f32(&logits)?,
            },
            self.device.read_f32(&action)?,
        ))
    }
}
