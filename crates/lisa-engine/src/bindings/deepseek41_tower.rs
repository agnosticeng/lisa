//! Integration of the native GGUF graph with LISA sessions and generation.
//! Session caches own shared KV state and Engram history, including snapshots.

use super::deepseek41_graph::{Assembly, assembly};
use crate::cache::{LayerCache, NativeCache};
use crate::language::LanguageModel;
use crate::language::LanguageRuntime;
use crate::{
    bindings::{
        BindingDevice, deepseek41_graph::Graph, deepseek41_projection as projection,
        deepseek41_state::State,
    },
    core::gguf_tokenizer::Tokenizer,
    execution::DeviceContext,
};
use anyhow::{Context, Result, ensure};
use lisa_backend::{ManagedDevice, ScalarType, TensorAccess};
use lisa_gguf::Package;
use lisa_models::deepseek41::tower as model;
use std::os::unix::fs::MetadataExt;
use std::{cell::RefCell, path::Path};

pub struct Tower<
    D: BindingDevice + ManagedDevice,
    R: LanguageRuntime<Tensor = D::Tensor, Cache = LayerCache<D>>,
> {
    context: DeviceContext<D>,
    runtime: R,
    graph: Graph<D>,
    capacity: usize,
    eos: u32,
    law: u64,
    cached_head: RefCell<Option<(D::Tensor, D::Tensor)>>,
}

impl<
    D: BindingDevice + ManagedDevice,
    R: LanguageRuntime<Tensor = D::Tensor, Cache = LayerCache<D>>,
> Tower<D, R>
where
    R::Error: From<anyhow::Error>,
{
    pub fn load(context: DeviceContext<D>, runtime: R, path: &Path) -> Result<Self> {
        let package = Package::open(path)?;
        let tokenizer = Tokenizer::from_header(&package.header)?;
        let mut law = 0xcbf29ce484222325u64;
        let mut hash = |bytes: &[u8]| {
            for &b in bytes {
                law = (law ^ b as u64).wrapping_mul(0x100000001b3);
            }
        };
        hash(DS4_REVISION.as_bytes());
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
        Ok(Self {
            graph: Graph::load(context.clone(), &package)?,
            context,
            runtime,
            capacity: model::DEFAULT_CONTEXT,
            eos: tokenizer.eos,
            law,
            cached_head: RefCell::new(None),
        })
    }

    fn feed(
        &mut self,
        tokens: &[u32],
        caches: &mut [LayerCache<D>],
        all: bool,
    ) -> Result<(D::Tensor, D::Tensor)> {
        ensure!(
            !tokens.is_empty()
                && tokens
                    .iter()
                    .all(|&t| (t as usize) < self.graph.config.vocab),
            "invalid DeepSeek token sequence"
        );
        let [LayerCache::Native(cache)] = caches else {
            anyhow::bail!("DeepSeek requires its native shared cache")
        };
        let state = cache.get_mut::<State<D>>()?;
        ensure!(
            state.capacity() == self.capacity,
            "DeepSeek context capacity changed after session creation"
        );
        ensure!(
            state
                .position()
                .checked_add(tokens.len())
                .is_some_and(|end| end <= self.capacity),
            "DeepSeek input exceeds the configured context capacity"
        );
        state.initialize()?;
        let mut hidden = Vec::new();
        let mut residual = Vec::new();
        *self.cached_head.borrow_mut() = None;
        for (chunk, input) in tokens.chunks(model::PREFILL_CHUNK).enumerate() {
            let last_chunk = (chunk + 1) * model::PREFILL_CHUNK >= tokens.len();
            let results = self.graph.prefill(input, state, last_chunk, None)?;
            for (i, result) in results.into_iter().enumerate() {
                let last = last_chunk && i + 1 == input.len();
                if let Some(logits) = result.logits {
                    *self.cached_head.borrow_mut() = Some((result.hidden.clone(), logits));
                }
                if all || last {
                    hidden.push(result.hidden);
                    residual.push(assembly(
                        Assembly::Reshape(vec![1, (model::WIDTH * model::STREAMS) as i32]),
                        &[&result.residual],
                        &self.context,
                    )?);
                }
            }
        }
        let concatenate = |rows: &[D::Tensor], width| {
            assembly(
                Assembly::Output(rows.len(), width),
                &rows.iter().collect::<Vec<_>>(),
                &self.context,
            )
        };
        Ok((
            concatenate(&hidden, model::WIDTH as i32)?,
            concatenate(&residual, (model::WIDTH * model::STREAMS) as i32)?,
        ))
    }
}

impl<
    D: BindingDevice + ManagedDevice,
    R: LanguageRuntime<Tensor = D::Tensor, Cache = LayerCache<D>>,
> LanguageModel<R> for Tower<D, R>
where
    R::Error: From<anyhow::Error>,
{
    fn runtime(&self) -> &R {
        &self.runtime
    }
    fn uses_ds4_sampling(&self) -> bool {
        true
    }
    fn cache_numerical_law(&self) -> Option<u64> {
        Some(self.law)
    }
    fn max_position_embeddings(&self) -> usize {
        self.capacity
    }
    fn set_context_capacity(&mut self, capacity: usize) -> Result<()> {
        ensure!(
            (1..=self
                .graph
                .config
                .context
                .min(lisa_models::deepseek41::state::MAX_CONTEXT))
                .contains(&capacity),
            "DeepSeek context capacity exceeds the supported checkpoint limit"
        );
        self.capacity = capacity;
        Ok(())
    }
    fn eos_token_id(&self) -> i64 {
        self.eos as i64
    }
    fn new_caches(&self) -> Vec<LayerCache<D>> {
        vec![LayerCache::Native(NativeCache::new(State::empty(
            self.context.device().clone(),
            self.capacity,
        )))]
    }
    fn forward(
        &mut self,
        ids: &D::Tensor,
        caches: Option<&mut Vec<LayerCache<D>>>,
    ) -> Result<(D::Tensor, D::Tensor)> {
        ensure!(
            model::token_shape(ids.shape()),
            "DeepSeek currently supports one sequence per forward"
        );
        ensure!(
            matches!(
                self.context.device().dtype(ids),
                Some(ScalarType::I32 | ScalarType::U32)
            ),
            "DeepSeek token IDs must be 32-bit integers"
        );
        let tokens = assembly(Assembly::Ids, &[ids], &self.context)?.read_raw_ids()?;
        let mut fresh;
        let caches = match caches {
            Some(caches) => caches,
            None => {
                fresh = self.new_caches();
                &mut fresh
            }
        };
        self.feed(&tokens, caches, true)
    }
    fn forward_capture(
        &mut self,
        ids: &D::Tensor,
        caches: Option<&mut Vec<LayerCache<D>>>,
        capture: bool,
    ) -> Result<(D::Tensor, D::Tensor)> {
        ensure!(
            !capture,
            "DeepSeek GGUF has no speculative verifier; use serial generation"
        );
        self.forward(ids, caches)
    }
    fn prefill(&mut self, tokens: &[u32], caches: &mut Vec<LayerCache<D>>) -> Result<D::Tensor> {
        assembly(
            Assembly::Reshape(vec![1, model::WIDTH as i32]),
            &[&self.feed(tokens, caches, false)?.0],
            &self.context,
        )
    }
    fn prefill_multi(
        &mut self,
        tokens: &[u32],
        caches: &mut Vec<LayerCache<D>>,
    ) -> Result<(D::Tensor, D::Tensor)> {
        let (hidden, multi) = self.feed(tokens, caches, false)?;
        Ok((
            assembly(
                Assembly::Reshape(vec![1, model::WIDTH as i32]),
                &[&hidden],
                &self.context,
            )?,
            assembly(
                Assembly::Reshape(vec![1, (model::WIDTH * model::STREAMS) as i32]),
                &[&multi],
                &self.context,
            )?,
        ))
    }
    fn head(&self, mixed: &D::Tensor) -> std::result::Result<D::Tensor, R::Error> {
        let apply = || -> Result<D::Tensor> {
            ensure!(
                self.context.device().dtype(mixed) == Some(ScalarType::F32)
                    && mixed.dim(-1) == 5120,
                "invalid DeepSeek output-head input"
            );
            if let Some((hidden, logits)) = self.cached_head.borrow().as_ref() {
                let device = self.context.device();
                let al = device.layout(hidden);
                let bl = device.layout(mixed);
                if device.same_storage(hidden, mixed)
                    && al.offset == bl.offset
                    && hidden.size() == mixed.size()
                    && bl.is_contiguous(mixed.shape())
                {
                    let mut shape = mixed.shape().to_vec();
                    *shape.last_mut().context("empty head shape")? = self.graph.config.vocab as i32;
                    return assembly(Assembly::Reshape(shape), &[logits], &self.context);
                }
            }
            let matrix = projection::matrix(self.graph.output_weight())?;
            assembly(
                Assembly::Head(matrix, mixed.shape().to_vec()),
                &[mixed, &projection::bound(self.graph.output_weight())?],
                &self.context,
            )
        };
        apply().map_err(R::Error::from)
    }
    fn warmup(&mut self, seed: &[u32]) -> Result<()> {
        let first = *seed.first().context("DeepSeek warmup needs a token")?;
        let mut caches = self.new_caches();
        self.prefill(&[first], &mut caches)?;
        *self.cached_head.borrow_mut() = None;
        Ok(())
    }
    fn batch_decode_ok(&self) -> bool {
        false
    }
    fn kv_bytes_per_token(&self) -> usize {
        model::kv_bytes_per_token(self.capacity)
    }
    fn stream_state_bytes(&self) -> usize {
        State::<D>::storage_bytes(self.capacity)
    }
}

pub const DS4_REVISION: &str = "b3f50180ec2c7324892be1f50bc83555ccb8d5e7";
