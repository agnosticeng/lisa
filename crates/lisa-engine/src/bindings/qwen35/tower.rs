//! Shared Qwen dense hybrid tower for MLX and GGUF checkpoints.
//! Numerical execution policies retain format-specific kernels; layer order,
//! cache creation, prefill history, MTP integration and serving have one owner.
use crate::batching::BatchRuntime;
use crate::cache::{attention::FullAttentionCache, recurrent::GdnCache};
use crate::core::{graph::PlanCache, traversal::PreparedTraversal};
use crate::{
    bindings::{BindingDevice, qwen35::execution::Execution},
    execution::DeviceContext,
    language::runtime::Runtime,
    sampling::sampler::Sampler,
};
use crate::{cache::LayerCache, language::LanguageModel};
use anyhow::{Result, ensure};
use lisa_backend::{ManagedDevice, ReadbackDevice, TensorAccess};
use lisa_models::qwen3_5::config::Qwen35Config;
use lisa_models::qwen3_5::tower::{self as definition, View};
fn view<D: BindingDevice>(
    context: &DeviceContext<D>,
    spec: View,
    inputs: &[&D::Tensor],
) -> Result<D::Tensor> {
    context.with_slot(
        "qwen35.views",
        || std::cell::RefCell::new(PlanCache::new(context.device().clone(), 16)),
        |p| {
            p.borrow_mut()
                .get_or_prepare(spec.clone(), || spec.graph())?
                .run(inputs, &mut [])?
                .into_one()
                .map_err(anyhow::Error::msg)
        },
    )
}
pub struct Qwen35Tower<D: BindingDevice + ManagedDevice + ReadbackDevice> {
    runtime: Runtime<D, Sampler<D>>,
    execution: Box<dyn Execution<D>>,
    pub config: Qwen35Config,
    capacity: usize,
    traversal: Option<PreparedTraversal<D>>,
    warmed: bool,
}
impl<D: BindingDevice + ManagedDevice + ReadbackDevice> Qwen35Tower<D> {
    pub fn from_execution(
        runtime: Runtime<D, Sampler<D>>,
        execution: impl Execution<D> + 'static,
    ) -> Self {
        let config = execution.config().clone();
        let capacity = execution.initial_capacity();
        Self {
            runtime,
            execution: Box::new(execution),
            config,
            capacity,
            traversal: None,
            warmed: false,
        }
    }
    fn forward_inner(
        &mut self,
        ids: &D::Tensor,
        caches: Option<&mut Vec<LayerCache<D>>>,
        capture: bool,
    ) -> Result<D::Tensor> {
        let _trace = self.runtime.context.trace("tower.forward");
        ensure!(
            ids.rank() == 2 && ids.dim(0) > 0 && ids.dim(1) > 0,
            "Qwen expects nonempty [batch,tokens]"
        );
        let use_cache = caches.is_some() || self.execution.requires_cache();
        let mut fresh = Vec::new();
        let caches = match caches {
            Some(c) => c,
            None => {
                fresh.extend(self.new_caches());
                &mut fresh
            }
        };
        let offset = caches
            .iter()
            .find_map(|c| match c {
                LayerCache::Full(f) => Some(f.offset),
                _ => None,
            })
            .unwrap_or(0);
        let window = definition::Window {
            batch: ids.dim(0),
            rows: ids.dim(1),
            batch_supported: self.execution.batch_decode_ok(),
            capture,
            max_depth: self.execution.max_speculative_depth(),
            layers: self.config.num_hidden_layers,
            caches: caches.len(),
            offset,
            capacity: self.capacity,
            chunk: self.execution.forward_chunk(),
        };
        window.validate()?;
        if let Some(chunks) = window.chunks() {
            let mut parts = Vec::new();
            for (start, end) in chunks {
                let h = self.forward_inner(
                    &view(&self.runtime.context, View::Tokens(start, end), &[ids])?,
                    Some(caches),
                    false,
                )?;
                h.eval()?;
                parts.push(h);
            }
            return view(
                &self.runtime.context,
                View::Join(parts.len(), 1),
                &parts.iter().collect::<Vec<_>>(),
            );
        }
        if self.traversal.is_none() {
            self.traversal = Some(PreparedTraversal::new(
                self.runtime.context.device().clone(),
                definition::traversal(self.config.num_hidden_layers, &[]),
            )?);
        }
        struct Bound<'a, D: BindingDevice> {
            context: &'a DeviceContext<D>,
            execution: &'a mut dyn Execution<D>,
            ids: &'a D::Tensor,
            caches: &'a mut Vec<LayerCache<D>>,
            use_cache: bool,
            capture: bool,
        }
        impl<D: BindingDevice> crate::core::traversal::Blocks for Bound<'_, D> {
            type Tensor = D::Tensor;
            type State = crate::bindings::qwen35::execution::Forward<D::Tensor>;
            fn begin(&mut self) -> Result<Self::State> {
                self.execution.begin(
                    self.ids,
                    if self.use_cache {
                        Some(self.caches)
                    } else {
                        None
                    },
                )
            }
            fn layer(&mut self, index: usize, state: &mut Self::State) -> Result<()> {
                let _layer = self.context.trace_detail("tower.layer", index as u64);
                self.execution.layer(
                    index,
                    state,
                    if self.use_cache {
                        Some(&mut self.caches[index])
                    } else {
                        None
                    },
                    self.capture,
                )
            }
            fn capture(&self, _: &Self::State) -> Result<D::Tensor> {
                anyhow::bail!("the Qwen traversal has no capture taps")
            }
            fn finish(&self, state: Self::State) -> Result<D::Tensor> {
                self.execution.finish(state)
            }
        }
        let traversal = self.traversal.as_ref().expect("prepared Qwen traversal");
        let (h, _) = traversal.run(&mut Bound {
            context: &self.runtime.context,
            execution: &mut *self.execution,
            ids,
            caches,
            use_cache,
            capture,
        })?;
        Ok(h)
    }

    pub fn head(&self, x: &D::Tensor) -> std::result::Result<D::Tensor, D::Error> {
        self.execution.head(x)
    }
    pub fn head_verified(&self, x: &D::Tensor) -> std::result::Result<D::Tensor, D::Error> {
        self.execution.head_verified(x)
    }
    pub fn max_position_embeddings(&self) -> usize {
        self.capacity
    }
    pub fn eos_token_id(&self) -> i64 {
        self.config.eos_token_id
    }
    pub fn warmup(&mut self, seed: &[u32]) -> Result<()> {
        if self.warmed {
            return Ok(());
        }
        let base: Vec<u32> = if seed.is_empty() {
            (0..64).map(|i| 128 + (i * 37) % 4096).collect()
        } else {
            seed.to_vec()
        };
        for s in [1usize, 9, 40] {
            let started = std::time::Instant::now();
            let ids: Vec<i32> = (0..s).map(|i| base[i % base.len()] as i32).collect();
            let mut caches = self.new_caches();
            let h = self.forward_inner(
                &self.runtime.tokens(&ids, &[1, s as i32])?,
                Some(&mut caches),
                false,
            )?;
            self.head(&h)?.eval()?;
            crate::ttft_mark(&format!(
                "warmup: shape s={s} (JIT+weight upload) {:.2}s",
                started.elapsed().as_secs_f64()
            ));
        }
        self.runtime.context.device().clear_unused();
        self.warmed = true;
        Ok(())
    }
}
impl<D: BindingDevice + ManagedDevice + ReadbackDevice> LanguageModel<Runtime<D, Sampler<D>>>
    for Qwen35Tower<D>
{
    fn runtime(&self) -> &Runtime<D, Sampler<D>> {
        &self.runtime
    }
    fn max_speculative_depth(&self) -> usize {
        self.execution.max_speculative_depth()
    }
    fn supports_images(&self) -> bool {
        self.execution.supports_images()
    }
    fn prepare_images(&mut self, tokens: &mut Vec<u32>, images: &[image::RgbImage]) -> Result<()> {
        self.execution.prepare_images(tokens, images)
    }
    fn clear_images(&mut self) {
        self.execution.clear_images()
    }
    fn cache_numerical_law(&self) -> Option<u64> {
        self.execution.numerical_law()
    }
    fn prefix_cache_granularity(&self) -> Option<usize> {
        Some(self.execution.prefill_chunk())
    }
    fn max_position_embeddings(&self) -> usize {
        self.capacity
    }
    fn eos_token_id(&self) -> i64 {
        self.config.eos_token_id
    }
    fn set_context_capacity(&mut self, n: usize) -> Result<()> {
        self.capacity = definition::context_capacity(
            n,
            self.config.max_position_embeddings,
            self.execution.forward_chunk().is_some(),
        )?;
        Ok(())
    }
    fn new_caches(&self) -> Vec<LayerCache<D>> {
        definition::caches(&self.config)
            .into_iter()
            .map(|c| match c {
                definition::Cache::Full { heads, width } => {
                    LayerCache::Full(FullAttentionCache::with_dtype(
                        self.runtime.context.clone(),
                        heads,
                        width,
                        self.execution.cache_dtype(),
                    ))
                }
                definition::Cache::Linear => LayerCache::Linear(GdnCache::new()),
            })
            .collect()
    }

    fn batch_decode_ok(&self) -> bool {
        self.execution.batch_decode_ok()
    }
    fn kv_bytes_per_token(&self) -> usize {
        definition::bytes(&self.config, self.execution.cache_dtype().size_of()).0
    }
    fn stream_state_bytes(&self) -> usize {
        definition::bytes(&self.config, self.execution.cache_dtype().size_of()).1
    }
    fn forward(
        &mut self,
        ids: &D::Tensor,
        caches: Option<&mut Vec<LayerCache<D>>>,
    ) -> Result<(D::Tensor, D::Tensor)> {
        self.forward_capture(ids, caches, false)
    }
    fn forward_capture(
        &mut self,
        ids: &D::Tensor,
        caches: Option<&mut Vec<LayerCache<D>>>,
        capture: bool,
    ) -> Result<(D::Tensor, D::Tensor)> {
        let h = self.forward_inner(ids, caches, capture)?;
        Ok((h.clone(), h))
    }
    fn prefill(&mut self, tokens: &[u32], caches: &mut Vec<LayerCache<D>>) -> Result<D::Tensor> {
        Ok(self.prefill_multi(tokens, caches)?.0)
    }
    fn prefill_multi(
        &mut self,
        tokens: &[u32],
        caches: &mut Vec<LayerCache<D>>,
    ) -> Result<(D::Tensor, D::Tensor)> {
        ensure!(!tokens.is_empty(), "empty Qwen prefill");
        let started = std::time::Instant::now();
        let mut parts = Vec::new();
        for chunk in tokens.chunks(self.execution.prefill_chunk()) {
            let start = std::time::Instant::now();
            let ids: Vec<i32> = chunk.iter().map(|&t| t as i32).collect();
            let h = self.forward_inner(
                &self.runtime.tokens(&ids, &[1, ids.len() as i32])?,
                Some(caches),
                false,
            )?;
            parts.push(h);
            self.runtime.context.device().trim_unused();
            crate::ttft_mark(&format!(
                "prefill: chunk len={} done in {:.2}s (cumulative {:.2}s)",
                chunk.len(),
                start.elapsed().as_secs_f64(),
                started.elapsed().as_secs_f64()
            ));
        }
        let h = if parts.len() == 1 {
            parts.pop().unwrap()
        } else {
            view(
                &self.runtime.context,
                View::Join(parts.len(), 1),
                &parts.iter().collect::<Vec<_>>(),
            )?
        };
        Ok((view(&self.runtime.context, View::Last(h.dim(1)), &[&h])?, h))
    }
    fn head(&self, x: &D::Tensor) -> std::result::Result<D::Tensor, D::Error> {
        self.head(x)
    }
    fn head_verified(&self, x: &D::Tensor) -> std::result::Result<D::Tensor, D::Error> {
        self.head_verified(x)
    }
    fn warmup(&mut self, seed: &[u32]) -> Result<()> {
        self.warmup(seed)
    }
    fn has_drafter(&self) -> bool {
        self.execution.has_drafter()
    }
    fn mtp_cost_key(&self) -> String {
        self.execution.mtp_cost_key()
    }
    fn drafter_reset(&mut self) {
        self.execution.drafter_reset()
    }
    fn drafter_trim(&mut self, n: usize) {
        self.execution.drafter_trim(n)
    }
    fn drafter_offset(&self) -> usize {
        self.execution.drafter_offset()
    }
    fn drafter_restore_offset(&mut self, n: usize) {
        self.execution.drafter_restore_offset(n)
    }
    fn draft_step(
        &mut self,
        tokens: &D::Tensor,
        hidden: &D::Tensor,
    ) -> Result<(D::Tensor, D::Tensor)> {
        self.execution.draft_step(tokens, hidden)
    }
}
