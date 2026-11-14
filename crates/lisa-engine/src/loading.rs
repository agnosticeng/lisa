//! Provider-independent checkpoint resolution and application loading seam.
use crate::{
    checkpoint::{self, ModelKind},
    decision::DecisionModel,
};
use std::path::{Path, PathBuf};
/// A model kind the runtime can load.
pub enum Loaded<L = std::convert::Infallible, I = std::convert::Infallible> {
    /// A decoder-only text model (generation, batching, speculative decode).
    Language(L),
    /// A separate text-to-image model, driven by image generation requests.
    Image(I),
    /// A non-generative model with its own interface (e.g. typed decisions).
    Decision(Box<dyn DecisionModel>),
}

#[derive(Clone, Default)]
pub struct ModelOptions {
    pub dspine: Option<PathBuf>,
}

/// Implemented by the application adapter, never by a checkpoint reader.
pub trait ModelProvider {
    type Language;
    type Image;
    fn name(&self) -> &str;
    fn supports(&self, kind: ModelKind) -> bool;
    fn bind(
        &self,
        path: &Path,
        kind: ModelKind,
        options: &ModelOptions,
    ) -> anyhow::Result<Loaded<Self::Language, Self::Image>>;
}
pub fn load_with<P: ModelProvider>(
    target: &str,
    provider: &P,
    options: &ModelOptions,
) -> anyhow::Result<Loaded<P::Language, P::Image>> {
    load_dir_with(&checkpoint::resolve_model_dir(target)?, provider, options)
}
pub fn load_dir_with<P: ModelProvider>(
    dir: &Path,
    provider: &P,
    options: &ModelOptions,
) -> anyhow::Result<Loaded<P::Language, P::Image>> {
    let path = checkpoint::hf::checkpoint_path(dir.to_owned());
    let kind = ModelKind::from_model_type(&checkpoint::model_type_of(&path)?)?;
    anyhow::ensure!(
        provider.supports(kind),
        "{} does not support {kind:?}",
        provider.name()
    );
    provider.bind(&path, kind, options)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;

    struct ExternalProvider {
        supported: bool,
        calls: Cell<usize>,
    }
    impl ModelProvider for ExternalProvider {
        type Language = PathBuf;
        type Image = std::convert::Infallible;
        fn name(&self) -> &str {
            "external-test-device"
        }
        fn supports(&self, _: ModelKind) -> bool {
            self.supported
        }
        fn bind(
            &self,
            path: &Path,
            kind: ModelKind,
            options: &ModelOptions,
        ) -> anyhow::Result<Loaded<Self::Language, Self::Image>> {
            self.calls.set(self.calls.get() + 1);
            assert_eq!(kind, ModelKind::from_model_type("qwen3_5")?);
            assert_eq!(options.dspine.as_deref(), Some(Path::new("explicit-draft")));
            Ok(Loaded::Language(path.to_owned()))
        }
    }

    #[test]
    fn provider_admission_precedes_weight_access_and_passes_explicit_options() -> anyhow::Result<()>
    {
        let dir = tempfile::tempdir()?;
        // No weights or tokenizer: resolution/admission must not open them.
        std::fs::write(
            dir.path().join("config.json"),
            br#"{"model_type":"qwen3_5"}"#,
        )?;
        let mut provider = ExternalProvider {
            supported: false,
            calls: Cell::new(0),
        };
        let options = ModelOptions {
            dspine: Some("explicit-draft".into()),
        };
        assert!(load_dir_with(dir.path(), &provider, &options).is_err());
        assert_eq!(provider.calls.get(), 0);
        provider.supported = true;
        let Loaded::Language(bound_path) = load_dir_with(dir.path(), &provider, &options)? else {
            panic!("wrong model kind")
        };
        assert_eq!(bound_path, dir.path());
        assert_eq!(provider.calls.get(), 1);
        Ok(())
    }
}
