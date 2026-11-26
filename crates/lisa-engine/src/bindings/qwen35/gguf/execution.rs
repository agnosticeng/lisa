//! Qwen35 text/MTP graph over original mixed GGUF blocks.
use crate::bindings::qwen35::gguf::weights::{self, Config};
use crate::{
    bindings::BindingDevice,
    cache::{LayerCache, attention::FullAttentionCache},
    execution::DeviceContext,
    weights::source::PreparedTensors,
};
use lisa_backend::{ManagedDevice, ScalarType, TensorAccess};
use lisa_models::qwen3_5::gguf as model;

use anyhow::Result;

/// Lazily imports requested embedding rows without exposing checkpoint storage.
pub trait EmbeddingRows<T> {
    fn rows(&self, rows: &[u64]) -> Result<T>;
}

struct Layer<D: BindingDevice> {
    index: usize,
    weights: std::sync::Arc<weights::Bound<D>>,
    plans: std::cell::RefCell<Vec<(model::Layer, crate::core::graph::PreparedGraph<D>)>>,
}
struct Mtp<D: BindingDevice> {
    layer: Layer<D>,
    cache: FullAttentionCache<D>,
}
pub struct GgufExecution<D: BindingDevice + ManagedDevice> {
    pub context: DeviceContext<D>,
    _weights: PreparedTensors<D::Tensor, D::Reservation>,
    mapped_bytes: usize,
    bound: std::sync::Arc<weights::Bound<D>>,
    embedding: Box<dyn EmbeddingRows<D::Tensor>>,
    layers: Vec<Layer<D>>,
    mtp: Mtp<D>,
    config: Config,
    model_config: lisa_models::qwen3_5::config::Qwen35Config,
    law: u64,
}
fn full_cache<D: BindingDevice>(context: &DeviceContext<D>) -> FullAttentionCache<D> {
    FullAttentionCache::with_dtype(
        context.clone(),
        model::CACHE_HEADS,
        model::CACHE_WIDTH,
        ScalarType::F32,
    )
}
impl<D: BindingDevice> Layer<D> {
    fn load(weights: std::sync::Arc<weights::Bound<D>>, index: usize) -> Self {
        Self {
            index,
            weights,
            plans: Default::default(),
        }
    }
    fn forward(
        &self,
        x: &D::Tensor,
        cache: &mut LayerCache<D>,
        capture: bool,
        c: &Config,
    ) -> Result<D::Tensor> {
        let layout = match &*cache {
            LayerCache::Full(c) => model::Cache::Attention(c.append_plan(x.dim(0), x.dim(1))),
            LayerCache::Linear(c) => model::Cache::Delta {
                conv: c.conv.is_some(),
                state: c.ssm.is_some(),
                capture,
            },
            _ => anyhow::bail!("Qwen GGUF cache kind mismatch"),
        };
        anyhow::ensure!(
            model::full_attention(self.index) == matches!(layout, model::Cache::Attention(_)),
            "Qwen GGUF cache kind mismatch"
        );
        let spec = model::Layer {
            index: self.index,
            rows: x.dim(1),
            epsilon: c.eps.to_bits(),
            theta: c.theta.to_bits(),
            cache: layout,
        };
        let mut plans = self.plans.borrow_mut();
        let index = if let Some(i) = plans.iter().position(|(key, _)| key == &spec) {
            i
        } else {
            let plan = self
                .weights
                .prepare(spec.program(&self.weights.schema, &self.weights.projections)?)?;
            if plans.len() == 16 {
                plans.remove(0);
            }
            plans.push((spec.clone(), plan));
            plans.len() - 1
        };
        let mut inputs = vec![x];
        match &*cache {
            LayerCache::Full(c) => {
                if let (Some(k), Some(v)) = (&c.keys, &c.values) {
                    inputs.extend([k, v]);
                }
            }
            LayerCache::Linear(c) => {
                inputs.extend(c.conv.iter());
                inputs.extend(c.ssm.iter());
            }
            _ => unreachable!(),
        }
        let mut out = plans[index].1.run(&inputs, &mut [])?.into_iter();
        let activation = out.next().unwrap();
        match (&spec.cache, cache) {
            (model::Cache::Attention(layout), LayerCache::Full(c)) => {
                let (offset, capacity, _) = layout.layout().map_err(anyhow::Error::msg)?;
                c.keys = out.next();
                c.values = out.next();
                c.offset = offset;
                c.capacity = capacity;
            }
            (model::Cache::Delta { capture, .. }, LayerCache::Linear(c)) => {
                c.conv = out.next();
                c.ssm = out.next();
                c.capture_ssm = if *capture { out.next() } else { None };
                c.capture_conv_input = if *capture { out.next() } else { None };
            }
            _ => unreachable!(),
        }
        Ok(activation)
    }
}
impl<D: BindingDevice + ManagedDevice> GgufExecution<D> {
    fn trim_temporaries(&self) -> Result<()> {
        // Complete pending users before reclaiming scratch. Admission remains
        // strict, and no completion wait happens under the allocator lock.
        let headroom = crate::memory::ram_budget().saturating_sub(self.mapped_bytes);
        let watermark = (headroom / 4 * 3).min(2 << 30);
        if self.context.device().memory_usage()?.private > watermark {
            self.context.device().submit(true)?;
            self.context.device().clear_unused();
        }
        Ok(())
    }

    fn draft_hidden(&mut self, ids: &D::Tensor, hidden: &D::Tensor) -> Result<D::Tensor> {
        let embeddings = self.embed(ids)?;
        let x = self.bound.run(
            model::draft_input(
                &self.bound.schema,
                &self.bound.projections,
                ids.dim(1),
                self.config.eps,
            )?,
            &[&embeddings, hidden],
        )?;
        let mtp = &mut self.mtp;
        let mut cache =
            LayerCache::Full(std::mem::replace(&mut mtp.cache, full_cache(&self.context)));
        let result = mtp.layer.forward(&x, &mut cache, false, &self.config);
        if let LayerCache::Full(c) = cache {
            mtp.cache = c;
        }
        self.bound.run(
            model::draft_finish(&self.bound.schema, ids.dim(1), self.config.eps)?,
            &[&result?],
        )
    }

    pub(crate) fn from_prepared(
        context: DeviceContext<D>,
        weights: crate::weights::source::PreparedTensors<D::Tensor, D::Reservation>,
        mapped_bytes: usize,
        bound: std::sync::Arc<weights::Bound<D>>,
        embedding: Box<dyn EmbeddingRows<D::Tensor>>,
        model_config: lisa_models::qwen3_5::config::Qwen35Config,
        config: Config,
        law: u64,
    ) -> Result<Self> {
        let layers = (0..model::TARGET_LAYERS)
            .map(|i| Layer::load(bound.clone(), i))
            .collect();
        let mtp = Mtp {
            layer: Layer::load(bound.clone(), model::DRAFT_LAYER),
            cache: full_cache(&context),
        };
        Ok(Self {
            context,
            law,
            embedding,
            layers,
            bound,
            mtp,
            model_config,
            config,
            _weights: weights,
            mapped_bytes,
        })
    }
    fn embed(&self, ids: &D::Tensor) -> Result<D::Tensor> {
        let tokens = weights::run(&self.context, model::token_readback(), &[ids])?
            .into_one()
            .map_err(anyhow::Error::msg)?
            .read_raw_ids()?;
        let rows: Vec<_> = tokens.into_iter().map(u64::from).collect();
        self.embedding.rows(&rows)
    }
}

impl<D: BindingDevice + ManagedDevice> crate::bindings::qwen35::execution::Execution<D>
    for GgufExecution<D>
where
    D::Error: From<anyhow::Error>,
{
    fn config(&self) -> &lisa_models::qwen3_5::config::Qwen35Config {
        &self.model_config
    }
    fn requires_cache(&self) -> bool {
        true
    }
    fn cache_dtype(&self) -> lisa_backend::ScalarType {
        lisa_backend::ScalarType::F32
    }
    fn prefill_chunk(&self) -> usize {
        model::CHUNK
    }
    fn forward_chunk(&self) -> Option<usize> {
        Some(model::CHUNK)
    }
    fn initial_capacity(&self) -> usize {
        model::initial_capacity(self.config.context)
    }
    fn max_speculative_depth(&self) -> usize {
        model::MAX_DRAFT_DEPTH
    }
    fn numerical_law(&self) -> Option<u64> {
        Some(self.law)
    }
    fn batch_decode_ok(&self) -> bool {
        false
    }
    fn begin(
        &self,
        ids: &D::Tensor,
        caches: Option<&Vec<LayerCache<D>>>,
    ) -> Result<crate::bindings::qwen35::execution::Forward<D::Tensor>> {
        let offset = caches
            .and_then(|c| {
                c.iter().find_map(|l| match l {
                    LayerCache::Full(f) => Some(f.offset),
                    _ => None,
                })
            })
            .unwrap_or(0);
        Ok(crate::bindings::qwen35::execution::Forward {
            x: self.embed(ids)?,
            residual: None,
            offset,
            rope: None,
        })
    }
    fn layer(
        &mut self,
        i: usize,
        state: &mut crate::bindings::qwen35::execution::Forward<D::Tensor>,
        cache: Option<&mut LayerCache<D>>,
        capture: bool,
    ) -> Result<()> {
        self.trim_temporaries()?;
        state.x = self.layers[i].forward(
            &state.x,
            cache.ok_or_else(|| anyhow::anyhow!("missing Qwen cache"))?,
            capture,
            &self.config,
        )?;
        Ok(())
    }
    fn finish(
        &self,
        state: crate::bindings::qwen35::execution::Forward<D::Tensor>,
    ) -> Result<D::Tensor> {
        self.bound.run(
            model::norm(&self.bound.schema, "output_norm.weight", self.config.eps)?,
            &[&state.x],
        )
    }
    fn head(&self, x: &D::Tensor) -> Result<D::Tensor, D::Error> {
        self.bound
            .run(
                model::projection(
                    &self.bound.schema,
                    &self.bound.projections,
                    "output.weight",
                    x.shape(),
                )
                .map_err(D::Error::from)?,
                &[x],
            )
            .map_err(D::Error::from)
    }
    fn has_drafter(&self) -> bool {
        true
    }
    fn mtp_cost_key(&self) -> String {
        // Fusions change measured round costs, but preserve the numerical law.
        model::COST_KEY.into()
    }
    fn drafter_reset(&mut self) {
        self.mtp.cache = full_cache(&self.context);
    }
    fn drafter_trim(&mut self, n: usize) {
        self.mtp.cache.trim(n);
    }
    fn drafter_offset(&self) -> usize {
        self.mtp.cache.offset
    }
    fn drafter_restore_offset(&mut self, n: usize) {
        self.mtp.cache.restore_offset(n);
    }
    fn draft_step(
        &mut self,
        ids: &D::Tensor,
        hidden: &D::Tensor,
    ) -> Result<(D::Tensor, D::Tensor)> {
        anyhow::ensure!(
            ids.rank() == 2 && ids.dim(0) == 1 && ids.dim(1) > 0,
            "MTP expects nonempty [1,tokens]"
        );
        let mut last = None;
        for (start, end) in model::draft_chunks(ids.dim(1))? {
            self.trim_temporaries()?;
            let mut input = weights::run(
                &self.context,
                model::draft_slice(ids.dim(1), start, end)?,
                &[ids, hidden],
            )?
            .into_iter();
            last = Some(self.draft_hidden(&input.next().unwrap(), &input.next().unwrap())?);
        }
        let x = last.expect("nonempty MTP input");
        let logits = self.head(&x)?;
        let token = weights::run(&self.context, model::select_token(), &[&logits])?
            .into_one()
            .map_err(anyhow::Error::msg)?;
        Ok((token, x))
    }
}
