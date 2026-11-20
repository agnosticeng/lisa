//! Original MLX payload binding and immutable LFM3 session state.
use super::BindingDevice;
use crate::{
    cache::{LayerCache, NativeCache, NativeState},
    core::{capture::Capture, graph::PreparedGraph},
    language::LanguageModel,
    language_runtime::Runtime,
    sampler::Sampler,
    tensor_ops::{TensorLaw, TensorOps},
    weights::{
        mlx::{Checkpoint, Shard},
        source::{Weights, get_tensor},
    },
};
use anyhow::{Context, Result, ensure};
use lisa_backend::{
    ManagedDevice, ReadbackDevice, ScalarType as Dtype, TensorAccess, TensorOperation as T,
};
use lisa_models::lfm3::{
    config::{Config, Layer},
    graph::{self, Definition},
};
use std::{cell::RefCell, path::Path, sync::Arc};
#[derive(Clone)]
struct State<T> {
    position: usize,
    layers: Vec<Vec<T>>,
    cursors: Vec<usize>,
}
impl<T: Clone + Send + Sync + 'static> NativeState for State<T> {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
    // Every graph produces replacement tensors; no retained snapshot is mutated.
    fn copy_for_write(&self) -> Result<Arc<dyn NativeState>> {
        Ok(Arc::new(self.clone()))
    }
    fn position(&self) -> usize {
        self.position
    }
}
struct BoundLayer<D: BindingDevice> {
    weights: Vec<D::Tensor>,
    plans: Vec<((usize, usize, usize), PreparedGraph<D>)>,
}
pub struct Tower<D: BindingDevice + ManagedDevice + ReadbackDevice> {
    runtime: Runtime<D, Sampler<D>>,
    config: Config,
    capacity: usize,
    layers: Vec<BoundLayer<D>>,
    embedding: Vec<D::Tensor>,
    norm: D::Tensor,
    heads: RefCell<Vec<(usize, PreparedGraph<D>)>>,
    capture: Option<Capture<D::Tensor>>,
}
fn bind<D: BindingDevice>(
    device: &D,
    def: &Definition,
    weights: &Weights<D::Tensor>,
    prefix: &str,
) -> Result<Vec<D::Tensor>> {
    def.weights
        .iter()
        .map(|(name, shape, dtype)| {
            let key = format!("{prefix}{name}");
            let w = weights
                .get(&key)
                .with_context(|| format!("missing LFM3 tensor {key}"))?;
            ensure!(
                w.shape() == shape && device.dtype(w) == Some(*dtype),
                "LFM3 tensor {key} has incompatible shape or dtype"
            );
            Ok(w.clone())
        })
        .collect()
}
impl<D: BindingDevice + ManagedDevice + ReadbackDevice> Tower<D>
where
    D::Error: From<anyhow::Error>,
{
    pub fn load(runtime: Runtime<D, Sampler<D>>, path: &Path) -> Result<Self> {
        let config = Config::parse(&std::fs::read(path.join("config.json"))?)?;
        let checkpoint = Checkpoint::open(path)?;
        let mut files = checkpoint.weight_map.values().cloned().collect::<Vec<_>>();
        files.sort();
        files.dedup();
        let bytes = files.iter().try_fold(0u64, |n, f| {
            Ok::<_, anyhow::Error>(n + std::fs::metadata(path.join(f))?.len())
        })?;
        ensure!(
            bytes.saturating_add(256 * 1024 * 1024) <= crate::memory::ram_budget() as u64,
            "LFM3 weights and working memory exceed the device memory budget"
        );
        let device = runtime.context.device();
        let mut weights = Weights::new();
        for file in files {
            let shard = Shard::open(device.clone(), &path.join(file))?;
            for name in shard.tensor_names() {
                let key = name
                    .strip_prefix("language_model.")
                    .unwrap_or(name)
                    .to_string();
                weights.insert(key, get_tensor(&shard, name)?);
            }
        }
        let head = graph::head(&config, 1);
        let embedding = bind(device, &head, &weights, "")?;
        let norm = weights
            .get("model.embedding_norm.weight")
            .context("missing LFM3 final norm")?
            .clone();
        ensure!(
            norm.shape() == [config.hidden_size as i32] && device.dtype(&norm) == Some(Dtype::BF16),
            "invalid LFM3 final norm"
        );
        let mut layers = vec![];
        for (i, &kind) in config.layer_types.iter().enumerate() {
            let def = graph::layer(&config, kind, 1, 0, 0);
            let w = bind(device, &def, &weights, &format!("model.layers.{i}."))?;
            let plan = PreparedGraph::new(device.clone(), def.graph, w.clone())?;
            layers.push(BoundLayer {
                weights: w,
                plans: vec![((1, 0, 0), plan)],
            });
        }
        let head = PreparedGraph::new(device.clone(), head.graph, embedding.clone())?;
        Ok(Self {
            capacity: config.max_position_embeddings,
            config,
            runtime,
            layers,
            embedding,
            norm,
            heads: RefCell::new(vec![(1, head)]),
            capture: None,
        })
    }
    fn run(&mut self, ids: &D::Tensor, caches: &mut Vec<LayerCache<D>>) -> Result<D::Tensor> {
        let _trace = self.runtime.context.trace("lfm3.forward");
        ensure!(
            ids.rank() == 2 && ids.dim(0) == 1 && ids.dim(1) > 0,
            "LFM3 expects one nonempty token sequence"
        );
        ensure!(caches.len() == 1, "LFM3 requires its native session cache");
        let LayerCache::Native(cache) = &mut caches[0] else {
            anyhow::bail!("invalid LFM3 session cache")
        };
        let old = cache.get::<State<D::Tensor>>()?;
        let offset = old.position;
        let rows = ids.dim(1) as usize;
        ensure!(
            offset.checked_add(rows).is_some_and(|n| n <= self.capacity),
            "LFM3 context capacity exceeded"
        );
        let mut next = old.clone();
        let context = &self.runtime.context;
        let device = context.device();
        let ops = TensorOps::new(device, TensorLaw::Native);
        let pos = (offset..offset + rows)
            .map(|v| v as f32)
            .collect::<Vec<_>>();
        let positions = super::tensor_io::pull_f32(context, &pos, &[rows as i32, 1])?;
        let mut x = super::affine::lookup(
            context,
            ids,
            &self.embedding[0],
            &self.embedding[1],
            &self.embedding[2],
            64,
            4,
        )?;
        if let Some(c) = &self.capture {
            c.store(&format!("p{offset}.embed"), &x);
        }
        let unused = ops.scalar(0.0)?;
        for (i, layer) in self.layers.iter_mut().enumerate() {
            let kind = self.config.layer_types[i];
            let state = &next.layers[i];
            let history = if kind == Layer::Conv || state.is_empty() {
                0
            } else {
                state[0].dim(2) as usize
            };
            let cursor = next.cursors[i];
            let key = (rows, history, cursor);
            if !layer.plans.iter().any(|(k, _)| *k == key) {
                let def = graph::layer(&self.config, kind, rows, history, cursor);
                let p = PreparedGraph::new(device.clone(), def.graph, layer.weights.clone())?;
                if layer.plans.len() == 2 {
                    layer.plans.remove(0);
                }
                layer.plans.push((key, p));
            }
            let plan = &layer.plans.iter().find(|(k, _)| *k == key).unwrap().1;
            let initial;
            let first = if let Some(t) = state.first() {
                t
            } else if kind == Layer::Conv {
                initial = ops.required(
                    lisa_backend::Kernel::Zeros {
                        shape: vec![
                            1,
                            (self.config.conv_length - 1) as i32,
                            self.config.hidden_size as i32,
                        ],
                        dtype: Dtype::BF16,
                    },
                    &[],
                )?;
                &initial
            } else {
                &unused
            };
            let second = state.get(1).unwrap_or(&unused);
            let mask = if kind == Layer::Conv {
                unused.clone()
            } else {
                let values = graph::attention_mask(kind, self.config.sliding_window, history, rows);
                let total = values.len() / rows;
                super::tensor_io::pull_f32(context, &values, &[1, 1, rows as i32, total as i32])?
            };
            let mut output = plan
                .run(&[&x, first, second, &positions, &mask], &mut [])?
                .into_iter();
            x = output.next().context("missing LFM3 layer output")?;
            next.layers[i] = output.collect();
            next.cursors[i] =
                graph::next_cursor(kind, self.config.sliding_window, history, cursor, rows);
            if let Some(c) = &self.capture {
                if c.wants(i) {
                    c.store(&format!("p{offset}.layer{i}"), &x);
                    for (j, t) in next.layers[i].iter().enumerate() {
                        c.store(&format!("p{offset}.layer{i}.state{j}"), t);
                    }
                }
            }
            // Bound outstanding temporaries on mobile without changing graph arithmetic.
            if i % 4 == 3 {
                device.submit(true)?;
                device.trim_unused();
            }
        }
        x = ops.op(
            T::RmsNorm {
                epsilon: self.config.norm_eps.to_bits(),
            },
            &[&x, &self.norm],
        )?;
        next.position += rows;
        *cache.get_mut::<State<D::Tensor>>()? = next;
        Ok(x)
    }
}
impl<D: BindingDevice + ManagedDevice + ReadbackDevice> LanguageModel<Runtime<D, Sampler<D>>>
    for Tower<D>
where
    D::Error: From<anyhow::Error>,
{
    fn runtime(&self) -> &Runtime<D, Sampler<D>> {
        &self.runtime
    }
    fn max_position_embeddings(&self) -> usize {
        self.capacity
    }
    fn set_context_capacity(&mut self, n: usize) -> Result<()> {
        ensure!(
            n > 0 && n <= self.config.max_position_embeddings,
            "invalid LFM3 context capacity"
        );
        self.capacity = n;
        Ok(())
    }
    fn eos_token_id(&self) -> i64 {
        self.config.eos_token_id[0]
    }
    fn eos_token_ids(&self) -> Vec<i64> {
        self.config.eos_token_id.clone()
    }
    fn new_caches(&self) -> Vec<LayerCache<D>> {
        vec![LayerCache::Native(NativeCache::new(State::<D::Tensor> {
            position: 0,
            layers: vec![vec![]; self.layers.len()],
            cursors: vec![0; self.layers.len()],
        }))]
    }
    fn batch_decode_ok(&self) -> bool {
        false
    }
    fn max_speculative_depth(&self) -> usize {
        0
    }
    fn kv_bytes_per_token(&self) -> usize {
        self.config
            .layer_types
            .iter()
            .filter(|&&k| k == Layer::FullAttention)
            .count()
            * self.config.num_key_value_heads
            * self.config.head_dim
            * 4
    }
    fn stream_state_bytes(&self) -> usize {
        self.config
            .layer_types
            .iter()
            .map(|&k| match k {
                Layer::Conv => (self.config.conv_length - 1) * self.config.hidden_size * 2,
                Layer::SlidingAttention => {
                    self.config.sliding_window
                        * self.config.num_key_value_heads
                        * self.config.head_dim
                        * 4
                }
                _ => 0,
            })
            .sum()
    }
    fn forward(
        &mut self,
        ids: &D::Tensor,
        caches: Option<&mut Vec<LayerCache<D>>>,
    ) -> Result<(D::Tensor, D::Tensor)> {
        ensure!(
            ids.rank() == 2 && ids.dim(0) == 1 && ids.dim(1) > 0,
            "LFM3 expects one nonempty token sequence"
        );
        let mut fresh;
        let c = if let Some(c) = caches {
            c
        } else {
            fresh = self.new_caches();
            &mut fresh
        };
        let x = if ids.dim(1) > 8 {
            let mut parts = vec![];
            for start in (0..ids.dim(1)).step_by(8) {
                parts.push(self.run(&ids.slice(1, start, Some((start + 8).min(ids.dim(1))))?, c)?);
            }
            D::Tensor::concatenate(&parts.iter().collect::<Vec<_>>(), 1)?
        } else {
            self.run(ids, c)?
        };
        Ok((x.clone(), x))
    }
    fn forward_capture(
        &mut self,
        ids: &D::Tensor,
        caches: Option<&mut Vec<LayerCache<D>>>,
        capture: bool,
    ) -> Result<(D::Tensor, D::Tensor)> {
        ensure!(!capture, "LFM3 speculative verification is not supported");
        self.forward(ids, caches)
    }
    fn prefill(&mut self, tokens: &[u32], caches: &mut Vec<LayerCache<D>>) -> Result<D::Tensor> {
        ensure!(!tokens.is_empty(), "empty LFM3 prompt");
        let mut last = None;
        for chunk in tokens.chunks(8) {
            let ids = super::tensor_io::pull_i32(
                &self.runtime.context,
                &chunk.iter().map(|&v| v as i32).collect::<Vec<_>>(),
                &[1, chunk.len() as i32],
            )?;
            let out = self.run(&ids, caches)?;
            last = Some(out.select(1, chunk.len() as i32 - 1)?.contiguous()?);
            self.runtime.context.device().submit(true)?;
            self.runtime.context.device().trim_unused();
        }
        Ok(last.unwrap())
    }
    fn prefill_multi(
        &mut self,
        tokens: &[u32],
        caches: &mut Vec<LayerCache<D>>,
    ) -> Result<(D::Tensor, D::Tensor)> {
        let x = self.prefill(tokens, caches)?;
        Ok((x.clone(), x))
    }
    fn head(&self, x: &D::Tensor) -> std::result::Result<D::Tensor, D::Error> {
        let result = (|| -> Result<D::Tensor> {
            let rows = x.size() / self.config.hidden_size;
            let mut heads = self.heads.borrow_mut();
            if !heads.iter().any(|(n, _)| *n == rows) {
                let def = graph::head(&self.config, rows);
                let p = PreparedGraph::new(
                    self.runtime.context.device().clone(),
                    def.graph,
                    self.embedding.clone(),
                )?;
                if heads.len() == 2 {
                    heads.remove(0);
                }
                heads.push((rows, p));
            }
            let out = heads
                .iter()
                .find(|(n, _)| *n == rows)
                .unwrap()
                .1
                .run(&[x], &mut [])?
                .into_one()
                .map_err(anyhow::Error::msg)?;
            let out = out.reshape(&[rows as i32, self.config.vocab_size as i32])?;
            if let Some(c) = &self.capture {
                c.store("logits", &out);
            }
            Ok(out)
        })();
        result.map_err(Into::into)
    }
    fn warmup(&mut self, seed: &[u32]) -> Result<()> {
        let mut c = self.new_caches();
        let x = self.prefill(if seed.is_empty() { &[0] } else { seed }, &mut c)?;
        self.head(&x)?.eval()?;
        self.runtime.context.device().clear_unused();
        Ok(())
    }
    fn enable_logit_capture(&mut self) -> Result<()> {
        self.enable_stage_capture(vec![], vec![])
    }
    fn enable_stage_capture(&mut self, layers: Vec<usize>, keys: Vec<String>) -> Result<()> {
        ensure!(
            layers.iter().all(|&i| i < self.layers.len()),
            "invalid LFM3 capture layer"
        );
        self.capture = Some(Capture {
            layers,
            key_filters: keys,
            preserve_dispatch: true,
            ..Default::default()
        });
        Ok(())
    }
    fn write_logit_capture(&self, path: &str) -> Result<()> {
        self.capture
            .as_ref()
            .context("LFM3 capture not enabled")?
            .flush(path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn immutable_cache_forks_retain_positions_and_ring_cursors() {
        let mut cache = NativeCache::<Vec<u8>>::new(State::<Vec<u8>> {
            position: 256,
            layers: vec![vec![vec![1, 2, 3]]],
            cursors: vec![256],
        });
        let snapshot = cache.clone();
        let state = cache.get_mut::<State<Vec<u8>>>().unwrap();
        state.position += 1;
        state.cursors[0] = 1;
        state.layers[0] = vec![vec![4, 5, 6]];
        let saved = snapshot.get::<State<Vec<u8>>>().unwrap();
        assert_eq!(saved.position, 256);
        assert_eq!(saved.cursors, vec![256]);
        assert_eq!(saved.layers, vec![vec![vec![1, 2, 3]]]);
    }
}
