//! Generic checkpoint-name binding for model-owned computation programs.
use super::graph::PreparedGraph;
use lisa_backend::Device;
use lisa_models::program::Program;
pub fn prepare<D: Device>(
    device: D,
    program: Program,
    mut resolve: impl FnMut(&str) -> anyhow::Result<D::Tensor>,
) -> anyhow::Result<PreparedGraph<D>>
where
    D::Tensor: Clone,
{
    let weights = program
        .weights
        .iter()
        .map(|name| resolve(name))
        .collect::<anyhow::Result<Vec<_>>>()?;
    PreparedGraph::new(device, program.graph, weights)
}
