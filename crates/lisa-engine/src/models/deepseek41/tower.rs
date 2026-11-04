//! Integration of the native GGUF graph with LISA sessions and generation.
//! Session caches own shared KV state and Engram history, including snapshots.

use super::{
    graph::Graph,
    projection,
    state::{State, view},
    tokenizer::Tokenizer,
};
use crate::core::cache::{LayerCache, NativeCache};
use crate::models::LanguageModel;
use anyhow::{Context, Result, ensure};
use lisa_gguf::Package;
use lisa_mlx::{Array, Dtype, Stream};
use std::os::unix::fs::MetadataExt;
use std::{cell::RefCell, path::Path};

pub struct Tower {
    graph: Graph,
    capacity: usize,
    eos: u32,
    law: u64,
    cached_head: RefCell<Option<(Array, Array)>>,
}

impl Tower {
    pub fn load(path: &Path) -> Result<Self> {
        let package = Package::open(path)?;
        let tokenizer = Tokenizer::from_header(&package.header)?;
        let mut law = 0xcbf29ce484222325u64;
        let mut hash = |bytes: &[u8]| {
            for &b in bytes {
                law = (law ^ b as u64).wrapping_mul(0x100000001b3);
            }
        };
        hash(super::DS4_REVISION.as_bytes());
        hash(b"lisa-ds41-gguf-hc-stable-fused-rope-v3");
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
            graph: Graph::load(&package)?,
            capacity: 4096,
            eos: tokenizer.eos,
            law,
            cached_head: RefCell::new(None),
        })
    }

    fn feed(
        &mut self,
        tokens: &[u32],
        caches: &mut [LayerCache],
        all: bool,
    ) -> Result<(Array, Array)> {
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
        let state = cache.get_mut::<State>()?;
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
        for (chunk, input) in tokens.chunks(8).enumerate() {
            let last_chunk = (chunk + 1) * 8 >= tokens.len();
            let results = self.graph.prefill(input, state, last_chunk, None)?;
            for (i, result) in results.into_iter().enumerate() {
                let last = last_chunk && i + 1 == input.len();
                if let Some(logits) = result.logits {
                    *self.cached_head.borrow_mut() = Some((result.hidden.clone(), logits));
                }
                if all || last {
                    hidden.push(result.hidden);
                    residual.push(result.residual.reshape(&[1, 20480])?);
                }
            }
        }
        let concatenate = |rows: &[Array], width| -> Result<Array> {
            if rows.len() == 1 {
                Ok(rows[0].reshape(&[1, 1, width])?)
            } else {
                Ok(
                    Array::concatenate(&rows.iter().collect::<Vec<_>>(), 0)?.reshape(&[
                        1,
                        rows.len() as i32,
                        width,
                    ])?,
                )
            }
        };
        Ok((concatenate(&hidden, 5120)?, concatenate(&residual, 20480)?))
    }
}

impl LanguageModel for Tower {
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
            (1..=self.graph.config.context.min(super::state::MAX_CONTEXT)).contains(&capacity),
            "DeepSeek context capacity exceeds the supported checkpoint limit"
        );
        self.capacity = capacity;
        Ok(())
    }
    fn eos_token_id(&self) -> i64 {
        self.eos as i64
    }
    fn new_caches(&self) -> Vec<LayerCache> {
        vec![LayerCache::Native(NativeCache::new(State::empty(
            self.capacity,
        )))]
    }
    fn forward(
        &mut self,
        ids: &Array,
        caches: Option<&mut Vec<LayerCache>>,
    ) -> Result<(Array, Array)> {
        ensure!(
            ids.rank() == 2 && ids.dim(0) == 1 && ids.dim(1) > 0,
            "DeepSeek currently supports one sequence per forward"
        );
        ensure!(
            matches!(ids.dtype(), Dtype::Int32 | Dtype::Uint32),
            "DeepSeek token IDs must be 32-bit integers"
        );
        let tokens = ids.as_dtype(Dtype::Uint32)?.as_slice::<u32>().to_vec();
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
        ids: &Array,
        caches: Option<&mut Vec<LayerCache>>,
        capture: bool,
    ) -> Result<(Array, Array)> {
        ensure!(
            !capture,
            "DeepSeek GGUF has no speculative verifier; use serial generation"
        );
        self.forward(ids, caches)
    }
    fn prefill(&mut self, tokens: &[u32], caches: &mut Vec<LayerCache>) -> Result<Array> {
        Ok(self.feed(tokens, caches, false)?.0.reshape(&[1, 5120])?)
    }
    fn prefill_multi(
        &mut self,
        tokens: &[u32],
        caches: &mut Vec<LayerCache>,
    ) -> Result<(Array, Array)> {
        let (hidden, multi) = self.feed(tokens, caches, false)?;
        Ok((hidden.reshape(&[1, 5120])?, multi.reshape(&[1, 20480])?))
    }
    fn head(&self, mixed: &Array) -> lisa_mlx::error::Result<Array> {
        let apply = || -> Result<Array> {
            ensure!(
                mixed.dtype() == Dtype::Float32 && mixed.dim(-1) == 5120,
                "invalid DeepSeek output-head input"
            );
            if let Some((hidden, logits)) = self.cached_head.borrow().as_ref() {
                let (a, al) = hidden.t.buffer_and_layout();
                let (b, bl) = mixed.t.buffer_and_layout();
                if std::sync::Arc::ptr_eq(a, b)
                    && al.offset == bl.offset
                    && hidden.size() == mixed.size()
                    && bl.is_contiguous()
                {
                    let mut shape = mixed.shape().to_vec();
                    *shape.last_mut().context("empty head shape")? = self.graph.config.vocab as i32;
                    return Ok(logits.reshape(&shape)?);
                }
            }
            let flat = mixed.contiguous()?.reshape(&[-1, 5120])?;
            let stream = Stream::gpu();
            let mut rows = Vec::new();
            for row in 0..flat.dim(0) as usize {
                rows.push(projection::linear(
                    self.graph.output_weight(),
                    &view(&flat, row * 5120, &[1, 5120])?,
                    false,
                    &stream,
                )?);
            }
            let mut shape = mixed.shape().to_vec();
            *shape.last_mut().context("empty head shape")? = self.graph.config.vocab as i32;
            Ok(Array::concatenate(&rows.iter().collect::<Vec<_>>(), 0)?.reshape(&shape)?)
        };
        apply().map_err(|e| lisa_mlx::error::Error::msg(e.to_string()))
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
        640 * 4 * 5 / 2
            + if self.capacity >= 8192 {
                128 * 2 * 5 / 2
            } else {
                0
            }
    }
    fn stream_state_bytes(&self) -> usize {
        State::storage_bytes(self.capacity)
    }
}
