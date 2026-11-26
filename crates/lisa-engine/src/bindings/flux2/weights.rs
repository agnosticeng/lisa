use crate::{bindings::BindingDevice, execution::DeviceContext};
use lisa_backend::{HostTensor, ScalarType, TensorAccess};
// Prepared affine packs and native BF16/F32 tensors.
use anyhow::Context;

use std::{cell::RefCell, collections::HashMap};

use lisa_models::flux::symbolic::WeightLayout;
pub struct Weights<D: BindingDevice>(
    pub HashMap<String, D::Tensor>,
    pub RefCell<HashMap<(String, WeightLayout), D::Tensor>>,
    pub DeviceContext<D>,
);
impl<D: BindingDevice> Weights<D> {
    pub fn from_map(context: DeviceContext<D>, tensors: HashMap<String, D::Tensor>) -> Self {
        Self(tensors, RefCell::new(HashMap::new()), context)
    }
    pub fn get(&self, key: &str) -> anyhow::Result<D::Tensor> {
        self.0
            .get(key)
            .cloned()
            .with_context(|| format!("missing FLUX tensor {key}"))
    }
    pub fn schema(&self) -> anyhow::Result<lisa_models::flux::symbolic::Schema> {
        self.0
            .iter()
            .map(|(name, tensor)| Ok((name.clone(), info(&self.2, tensor)?)))
            .collect()
    }
    pub fn prepare(
        &self,
        program: lisa_models::flux::symbolic::Program,
    ) -> anyhow::Result<crate::core::graph::PreparedGraph<D>> {
        use crate::core::graph::PreparedGraph;
        let device = self.2.device().clone();
        let mut weights = Vec::with_capacity(program.weights.len());
        for request in program.weights {
            let source = self.get(&request.name)?;
            let weight = if let Some(graph) = request.layout.preparation() {
                let key = (request.name.clone(), request.layout);
                let cached = self.1.borrow().get(&key).cloned();
                match cached {
                    Some(weight) => weight,
                    None => {
                        let weight = PreparedGraph::new(device.clone(), graph, vec![])?
                            .run(&[&source], &mut [])?
                            .into_one()
                            .map_err(anyhow::Error::msg)?;
                        self.1.borrow_mut().insert(key, weight.clone());
                        weight
                    }
                }
            } else {
                source
            };
            weights.push(weight);
        }
        PreparedGraph::new(device, program.graph, weights)
    }
    pub fn lin(&self, key: &str, x: &D::Tensor) -> anyhow::Result<D::Tensor> {
        use lisa_models::flux::symbolic::{Trace, Weights};
        let trace = Trace::new();
        let schema = self.schema()?;
        let shape = info(&self.2, x)?;
        let input = trace.input(&shape.shape, shape.dtype);
        let out = Weights {
            trace: trace.clone(),
            schema: &schema,
        }
        .lin(key, &input)?;
        self.prepare(trace.finish(&[out])?)?
            .run(&[x], &mut [])?
            .into_one()
            .map_err(anyhow::Error::msg)
    }
}

pub fn info<D: BindingDevice>(
    context: &DeviceContext<D>,
    x: &D::Tensor,
) -> anyhow::Result<lisa_models::flux::symbolic::TensorInfo> {
    use lisa_backend::ScalarType as S;
    let dtype = context
        .device()
        .dtype(x)
        .filter(|t| matches!(t, S::F32 | S::BF16 | S::F16 | S::U32 | S::I32 | S::I64))
        .ok_or_else(|| anyhow::anyhow!("unsupported FLUX tensor dtype"))?;
    Ok(lisa_models::flux::symbolic::TensorInfo {
        shape: x.shape().to_vec(),
        dtype,
    })
}

pub fn run<D: BindingDevice>(
    context: &DeviceContext<D>,
    graph: lisa_models::graph::Graph,
    inputs: &[&D::Tensor],
) -> anyhow::Result<D::Tensor> {
    crate::core::graph::PreparedGraph::new(context.device().clone(), graph, vec![])?
        .run(inputs, &mut [])?
        .into_one()
        .map_err(anyhow::Error::msg)
}

pub fn pull<D: BindingDevice>(
    context: &DeviceContext<D>,
    bytes: &[u8],
    shape: &[i32],
    dtype: ScalarType,
) -> anyhow::Result<D::Tensor> {
    Ok(context.device().pull(
        HostTensor {
            bytes,
            shape,
            dtype,
        }
        .into(),
        Default::default(),
    )?)
}
