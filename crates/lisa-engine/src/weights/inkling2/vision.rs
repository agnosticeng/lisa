//! Lazy original-GGUF image adapter. The text tower receives feature tensors;
//! checkpoint access and request-bounded vision residency stay here.
use crate::{
    bindings::BindingDevice,
    core::graph::PreparedGraph,
    execution::DeviceContext,
    weights::{
        gguf::{ResidentWeights, load_resident},
        source::ImageFeatures,
    },
};
use anyhow::{Result, ensure};
use lisa_backend::{ManagedDevice, TensorAccess};
use lisa_models::inkling2::{
    image as processor,
    vision::{self as plan, Definition},
};
use std::{collections::BTreeSet, path::PathBuf};

/// Construction is metadata-only. Each image retains the existing independent
/// load/encode/release lifetime so image weights do not overlap text KV peaks.
pub(crate) struct LazyEncoder<D: BindingDevice + ManagedDevice> {
    context: DeviceContext<D>,
    path: PathBuf,
}
impl<D: BindingDevice + ManagedDevice> LazyEncoder<D> {
    pub(crate) fn new(context: DeviceContext<D>, path: PathBuf) -> Self {
        Self { context, path }
    }
}
impl<D: BindingDevice + ManagedDevice> ImageFeatures<D::Tensor> for LazyEncoder<D> {
    fn available(&self) -> bool {
        self.path.is_file()
    }

    fn encode(&self, image: &image::RgbImage) -> Result<D::Tensor> {
        let context = &self.context;
        let package = lisa_gguf::Package::open(&self.path)?;
        crate::checkpoint::inkling2::validate_vision(&package.header)?;
        let pixels = processor::preprocess(image)?;
        let (h, w) = (pixels.height, pixels.width);
        let n = (h * w + 1) as i32;
        crate::vision::admit_scratch(context, 2usize << 30)?;
        let mut defs = vec![plan::input(n)];
        defs.extend((0..23).map(|i| plan::layer(i, n)));
        defs.push(plan::project(h as i32, w as i32));
        let names = defs
            .iter()
            .flat_map(|d| d.weights.iter().map(|w| w.name.as_str()))
            .collect::<BTreeSet<_>>();
        let resident = load_resident(
            context.device().clone(),
            &package,
            &names.into_iter().collect::<Vec<_>>(),
        )?;
        let plans = defs
            .into_iter()
            .map(|d| prepare(context, &resident, d))
            .collect::<Result<Vec<_>>>()?;
        let table = package.tensor("v.position_embd.weight")?.disk_reader()?;
        ensure!(
            table.info.dimensions == [1024, 1024] && table.info.kind == lisa_gguf::TensorType::F32,
            "invalid Inkling vision position table"
        );
        let mut bytes = vec![0; 1024 * 1024 * 4];
        table.read_bytes(0, &mut bytes)?;
        let table = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect::<Vec<_>>();
        let positions = processor::positions(&table, h, w)?;
        let (cos, sin) = processor::rotary(h, w);
        let pull = |v: &[f32], shape: &[i32]| -> Result<D::Tensor> {
            crate::bindings::shared::tensor_io::pull_f32(context, v, shape)
        };
        let patches = pull(&pixels.values, &[n - 1, 1200])?;
        let positions = pull(&positions, &[n - 1, 1024])?;
        let cos = pull(&cos, &[n, 1, 2, 16])?;
        let sin = pull(&sin, &[n, 1, 2, 16])?;
        // The executor completes queued image work before returning. Keep the
        // reservation and original mapped payloads alive through that wait.
        crate::vision::inkling2::execute(context, plans, &patches, &positions, &cos, &sin)
    }
}

fn prepare<D: BindingDevice + ManagedDevice>(
    context: &DeviceContext<D>,
    resident: &ResidentWeights<D, D::Reservation>,
    def: Definition,
) -> Result<PreparedGraph<D>> {
    let weights = def
        .weights
        .iter()
        .map(|w| {
            let t = resident
                .tensors
                .get(&w.name)
                .ok_or_else(|| anyhow::anyhow!("missing Inkling vision tensor {}", w.name))?;
            let value = t.plain_view()?;
            ensure!(
                value.shape() == w.shape && context.device().dtype(&value) == Some(w.dtype),
                "Inkling vision weight mismatch: {}",
                w.name
            );
            Ok(value)
        })
        .collect::<Result<Vec<_>>>()?;
    PreparedGraph::new(context.device().clone(), def.graph, weights)
}
