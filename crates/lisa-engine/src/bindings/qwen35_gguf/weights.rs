//! Validate the complete 27B text/MTP graph before mapping GPU weights.
use crate::{bindings::BindingDevice, execution::DeviceContext, weights::gguf::ResidentWeights};
use anyhow::{Context, Result, ensure};
use lisa_gguf::Package;

pub use lisa_models::qwen3_5::gguf::Config;
pub fn validate(p: &Package) -> Result<Config> {
    use lisa_models::qwen3_5::gguf as model;
    let metadata = &p.header.metadata;
    model::validate(&model::Checkpoint {
        architecture: p.header.architecture()?.to_owned(),
        integers: metadata
            .iter()
            .filter_map(|(k, v)| v.as_u64().map(|n| (k.clone(), n)))
            .collect(),
        reals: metadata
            .iter()
            .filter_map(|(k, v)| v.as_f64().map(|n| (k.clone(), n)))
            .collect(),
        tensors: p
            .tensors
            .iter()
            .map(|(k, t)| {
                (
                    k.clone(),
                    model::TensorInfo {
                        dimensions: t.info.dimensions.clone(),
                        encoding: t.info.kind.0,
                    },
                )
            })
            .collect(),
    })
}

/// Format adapter: preserve original packed bytes and expose portable metadata.
pub struct Bound<D: BindingDevice> {
    pub context: DeviceContext<D>,
    pub arrays: std::collections::BTreeMap<String, D::Tensor>,
    pub schema: lisa_models::program::Schema,
    pub projections: lisa_models::qwen3_5::gguf::Projections,
}
impl<D: BindingDevice> Bound<D> {
    pub fn new<L>(context: DeviceContext<D>, w: &ResidentWeights<D, L>) -> Result<Self> {
        let mut arrays = std::collections::BTreeMap::new();
        let mut schema = lisa_models::program::Schema::new();
        let mut projections = lisa_models::qwen3_5::gguf::Projections::new();
        for (name, t) in &w.tensors {
            let shape = t
                .info
                .dimensions
                .iter()
                .rev()
                .map(|&n| i32::try_from(n))
                .collect::<std::result::Result<Vec<_>, _>>()?;
            let array = if t.info.kind == lisa_gguf::TensorType::F32 {
                t.plain_view()?
            } else {
                ensure!(shape.len() == 2, "expected a packed matrix {name}");
                projections.insert(
                    name.clone(),
                    lisa_models::qwen3_5::gguf::Projection {
                        encoding: t.info.kind.0,
                        input: shape[1],
                        output: shape[0],
                    },
                );
                t.packed_bytes().clone()
            };
            schema.insert(name.clone(), shape);
            arrays.insert(name.clone(), array);
        }
        Ok(Self {
            context,
            arrays,
            schema,
            projections,
        })
    }
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
