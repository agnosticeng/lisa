//! Prepared Qwen graph weights, independent of their checkpoint container.
use crate::{bindings::BindingDevice, execution::DeviceContext};
use anyhow::{Context, Result};

pub use lisa_models::qwen3_5::gguf::Config;
/// Named device tensors and the model schema used by prepared programs.
pub struct Bound<D: BindingDevice> {
    pub context: DeviceContext<D>,
    pub arrays: std::collections::BTreeMap<String, D::Tensor>,
    pub schema: lisa_models::program::Schema,
    pub projections: lisa_models::qwen3_5::gguf::Projections,
}
impl<D: BindingDevice> Bound<D> {
    pub fn prepare(
        &self,
        p: lisa_models::program::Program,
    ) -> Result<crate::core::graph::PreparedGraph<D>> {
        let weights = p
            .weights
            .iter()
            .map(|name| {
                self.arrays
                    .get(name)
                    .cloned()
                    .with_context(|| format!("missing {name}"))
            })
            .collect::<Result<Vec<_>>>()?;
        crate::core::graph::PreparedGraph::new(self.context.device().clone(), p.graph, weights)
    }
    pub fn run(
        &self,
        p: lisa_models::program::Program,
        inputs: &[&D::Tensor],
    ) -> Result<D::Tensor> {
        self.prepare(p)?
            .run(inputs, &mut [])?
            .into_one()
            .map_err(anyhow::Error::msg)
    }
}
pub fn run<D: BindingDevice>(
    context: &DeviceContext<D>,
    graph: lisa_models::graph::Graph,
    inputs: &[&D::Tensor],
) -> Result<lisa_backend::Outputs<D::Tensor>> {
    crate::core::graph::PreparedGraph::new(context.device().clone(), graph, vec![])?
        .run(inputs, &mut [])
}
