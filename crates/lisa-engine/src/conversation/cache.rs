//! Prefix snapshots bound to an explicit provider and the loaded model's law.
pub use crate::cache::prefix::MIN_PREFIX_TOKENS;
use crate::{
    bindings::BindingDevice,
    cache::{LayerCache, LayerState},
    conversation::Conversation,
    language::{LanguageModel, runtime::Runtime},
    sampling::sampler::Sampler,
};
use lisa_backend::{ManagedDevice, ReadbackDevice};
pub struct PrefixCache<D: BindingDevice + ManagedDevice + ReadbackDevice>(
    crate::cache::prefix::PrefixCache<D>,
);
impl<D: BindingDevice + ManagedDevice + ReadbackDevice> PrefixCache<D> {
    pub fn new(device: D, max_entries: usize, spill: Option<std::path::PathBuf>) -> Self {
        Self(crate::cache::prefix::PrefixCache::new(
            device,
            max_entries,
            spill,
        ))
    }
    pub fn enabled(&self) -> bool {
        self.0.enabled()
    }
    pub fn insert_with_law(&mut self, law: u64, tokens: &[u32], caches: &[LayerCache<D>]) {
        if self.enabled()
            && tokens.len() >= MIN_PREFIX_TOKENS
            && caches.iter().all(LayerCache::prefix_cacheable)
        {
            self.0.insert_with_law(
                law,
                tokens,
                caches.iter().map(LayerCache::snapshot_prefix).collect(),
            );
        }
    }
    pub fn insert_model(
        &mut self,
        model: &dyn LanguageModel<Runtime<D, Sampler<D>>>,
        tokens: &[u32],
        caches: &[LayerCache<D>],
    ) {
        if let Some(law) = model.cache_numerical_law() {
            self.insert_with_law(law, tokens, caches);
        }
    }
    pub fn restore_session(
        &mut self,
        model: &mut dyn LanguageModel<Runtime<D, Sampler<D>>>,
        prompt: &[u32],
    ) -> Option<(usize, Conversation<Runtime<D, Sampler<D>>>)> {
        let law = model.cache_numerical_law()?;
        let grain = model.prefix_cache_granularity();
        let (n, session) = self.0.restore(prompt, law, |layers, n| {
            // Older persisted entries may have been split inside a natural
            // prefill chunk. Never resume those under the aligned policy.
            if grain.is_some_and(|g| g == 0 || n % g != 0) {
                return None;
            }
            if layers.iter().any(
                |layer| matches!(layer, LayerState::Native(state) if !state.prefix_cacheable()),
            ) {
                return None;
            }
            Some(Conversation::from_prefix(model, layers, prompt, n))
        })?;
        session.map(|session| (n, session))
    }
}
