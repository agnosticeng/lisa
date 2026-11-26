//! DeepSeek checkpoint adaptation and disk-backed row providers.
pub mod engram;
pub mod plan;
use crate::{
    bindings::{
        BindingDevice,
        deepseek41::{engram::EmbeddingRows, graph::Graph, tower::Tower},
    },
    cache::LayerCache,
    execution::DeviceContext,
    language::LanguageRuntime,
};
use anyhow::Result;
use lisa_backend::ManagedDevice;
use lisa_gguf::{Package, Tensor, quant};
use lisa_models::deepseek41::tower as model;
use std::{os::unix::fs::MetadataExt, path::Path};

pub fn load<
    D: BindingDevice + ManagedDevice,
    R: LanguageRuntime<Tensor = D::Tensor, Cache = LayerCache<D>>,
>(
    context: DeviceContext<D>,
    runtime: R,
    path: &Path,
) -> Result<Tower<D, R>>
where
    R::Error: From<anyhow::Error>,
{
    let package = crate::checkpoint::deepseek41::open_package(path)?;
    let tokenizer = crate::checkpoint::tokenizer::native_from_header(&package.header)?;
    let mut law = 0xcbf29ce484222325u64;
    let mut hash = |bytes: &[u8]| {
        for &b in bytes {
            law = (law ^ b as u64).wrapping_mul(0x100000001b3);
        }
    };
    hash(crate::bindings::deepseek41::tower::DS4_REVISION.as_bytes());
    hash(model::NUMERICAL_IDENTITY);
    // Bind native snapshots to this checkpoint, including pruned expert
    // payloads and Engram token mappings with otherwise identical shapes.
    // This runs once at load; never hash model files in a dispatch key.
    for path in &package.shards {
        let stat = std::fs::metadata(path)?;
        for value in [
            stat.dev(),
            stat.ino(),
            stat.len(),
            stat.mtime() as u64,
            stat.mtime_nsec() as u64,
            stat.ctime() as u64,
            stat.ctime_nsec() as u64,
        ] {
            hash(&value.to_le_bytes());
        }
    }
    for (name, tensor) in &package.tensors {
        hash(name.as_bytes());
        hash(&tensor.info.kind.0.to_le_bytes());
        for d in &tensor.info.dimensions {
            hash(&d.to_le_bytes());
        }
    }
    let graph = load_graph(context.clone(), &package)?;
    Tower::from_prepared(context, runtime, graph, tokenizer.eos, law)
}

pub fn load_graph<D: BindingDevice + ManagedDevice>(
    context: DeviceContext<D>,
    package: &Package,
) -> Result<Graph<D>> {
    let _trace = context.trace("deepseek41.load");
    let config = crate::checkpoint::deepseek41::config(&package.header)?;
    let layout = crate::checkpoint::deepseek41::engram_layout(&package.header)?;
    let plan = plan::WeightPlan::inspect(package, &config, &layout)?;
    let tables = engram::DiskTables::open(package, &layout)?;
    let names: Vec<&str> = plan.resident.iter().map(String::as_str).collect();
    let resident =
        super::gguf::load_resident(context.device().clone(), package, &names)?.into_prepared()?;
    let embedding = package.tensor(model::EMBEDDING)?.clone();
    Graph::from_prepared(
        context,
        config,
        layout,
        Box::new(tables),
        Box::new(Rows(embedding)),
        resident,
    )
}

struct Rows(Tensor);
impl EmbeddingRows for Rows {
    fn rows(&self, tokens: &[u32]) -> Result<Box<dyn Iterator<Item = Result<Vec<f32>>>>> {
        let row_bytes = self.0.info.row_bytes()? as usize;
        let mut packed = vec![0u8; row_bytes * tokens.len()];
        self.0.read_rows(
            &tokens.iter().map(|&t| t as u64).collect::<Vec<_>>(),
            &mut packed,
        )?;
        let kind = self.0.info.kind;
        Ok(Box::new((0..tokens.len()).map(move |index| {
            let mut row = vec![0f32; model::WIDTH];
            quant::decode(
                kind,
                &packed[index * row_bytes..(index + 1) * row_bytes],
                &mut row,
            )?;
            Ok(row)
        })))
    }
}
