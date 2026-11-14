//! Prefix snapshots bound to an explicit provider and the loaded model's law.
pub use crate::prefix_cache::MIN_PREFIX_TOKENS;
use crate::{
    bindings::BindingDevice, cache::LayerCache, conversation::Conversation,
    language::LanguageModel, language_runtime::Runtime, sampler::Sampler,
};
use lisa_backend::{ManagedDevice, ReadbackDevice};
pub struct PrefixCache<D: BindingDevice + ManagedDevice + ReadbackDevice>(
    crate::prefix_cache::PrefixCache<D>,
);
impl<D: BindingDevice + ManagedDevice + ReadbackDevice> PrefixCache<D> {
    pub fn new(device: D, max_entries: usize, spill: Option<std::path::PathBuf>) -> Self {
        Self(crate::prefix_cache::PrefixCache::new(
            device,
            max_entries,
            spill,
        ))
    }
    pub fn enabled(&self) -> bool {
        self.0.enabled()
    }
    pub fn insert_with_law(&mut self, law: u64, tokens: &[u32], caches: &[LayerCache<D>]) {
        if self.enabled() && tokens.len() >= MIN_PREFIX_TOKENS {
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
        self.0.restore(prompt, law, |layers, n| {
            Conversation::from_prefix(model, layers, prompt, n)
        })
    }
}
