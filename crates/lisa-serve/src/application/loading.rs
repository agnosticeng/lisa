//! Checkpoint resolution, explicit device admission, and weight binding.
pub use checkpoint::{ModelKind, hf, model_type_of, resolve_model_dir};
use lisa_engine::checkpoint;
use std::path::Path;

pub type Loaded<L = NativeLanguage, I = NativeImage> = lisa_engine::loading::Loaded<L, I>;

#[cfg(feature = "metal")]
pub type NativeLanguage = super::Language;
#[cfg(not(feature = "metal"))]
pub type NativeLanguage = std::convert::Infallible;
#[cfg(feature = "metal")]
pub type NativeImage = Box<super::ImagePipeline>;
#[cfg(not(feature = "metal"))]
pub type NativeImage = std::convert::Infallible;
pub use lisa_engine::decision::DecisionModel;

/// Load a model from a local directory or a Hugging Face repo id, dispatching
/// on the structure read from its config, on the selected backend.
pub fn load(target: &str) -> anyhow::Result<Loaded> {
    load_dir(&resolve_model_dir(target)?)
}

/// Load a model from an already-resolved directory, on the selected backend.
pub fn load_dir(dir: &Path) -> anyhow::Result<Loaded> {
    load_dir_with(dir, super::device::Backend::selected())
}

/// Load a model from an already-resolved directory on an explicit backend.
/// Weight format comes from the checkpoint; device selection is explicit and
/// never hidden inside a format reader.
pub fn load_dir_with(dir: &Path, backend: super::device::Backend) -> anyhow::Result<Loaded> {
    lisa_engine::loading::load_dir_with(dir, &backend)
}

impl lisa_engine::loading::ModelProvider for super::device::Backend {
    type Language = NativeLanguage;
    type Image = NativeImage;
    fn name(&self) -> &str {
        (*self).name()
    }
    fn supports(&self, kind: ModelKind) -> bool {
        kind == ModelKind::Laya
            || (cfg!(feature = "metal") && *self == super::device::Backend::Metal)
    }
    fn bind(&self, dir: &Path, kind: ModelKind) -> anyhow::Result<Loaded> {
        let backend = *self;
        let model_type = model_type_of(dir)?;
        anyhow::ensure!(
            kind == ModelKind::Laya || backend == super::device::Backend::Metal,
            "{model_type} requires the Metal backend"
        );
        if kind == ModelKind::Laya {
            return Ok(Loaded::Decision(Box::new(
                lisa_engine::weights::laya::load_on(
                    dir,
                    backend.open()?,
                    backend.laya_arithmetic(),
                )?,
            )));
        }
        #[cfg(feature = "metal")]
        {
            use lisa_backend::ManagedDevice;
            let context = super::context()?;
            context.device().clear_unused();
            if matches!(kind, ModelKind::Flux2 | ModelKind::Flux3) {
                return Ok(Loaded::Image(Box::new(
                    lisa_engine::image_generation::Pipeline::load(context, dir)?,
                )));
            }
            if kind == ModelKind::Qwen4 {
                lisa_metal::warm_kernels();
            }
            Ok(Loaded::Language(lisa_engine::model_binding::language(
                super::runtime(context),
                dir,
                kind,
            )?))
        }
        #[cfg(not(feature = "metal"))]
        anyhow::bail!("this build has no language/image provider")
    }
}

#[cfg(test)]
mod load_options_tests {
    use super::*;

    #[test]
    fn resolved_cpu_policy_rejects_language_model_before_weight_import() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("config.json"),
            br#"{"model_type":"qwen3_5"}"#,
        )
        .unwrap();
        let error = match load_dir_with(dir.path(), crate::application::device::Backend::Cpu) {
            Err(error) => error,
            Ok(_) => panic!("CPU Qwen execution is not implemented"),
        };
        assert!(error.to_string().contains("does not support"));
        // No shard/index/tokenizer exists. Failure must precede opening them.
    }
}
