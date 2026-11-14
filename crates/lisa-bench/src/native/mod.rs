//! Native test and benchmark harness for the portable engine.
pub use lisa_engine::batching;
pub use lisa_engine::checkpoint;
pub use lisa_engine::conversation;
pub use lisa_engine::decision;
pub use lisa_engine::eos;
pub use lisa_engine::execution;
pub use lisa_engine::generation;
pub use lisa_engine::language;
pub use lisa_engine::scheduling;
pub use lisa_engine::speculation;
pub use lisa_engine::speculative_runtime;
pub use lisa_engine::weights;
pub mod core;
pub mod device {
    pub use lisa_serve::application::device::*;
}
pub use lisa_engine::laya;
pub mod loading {
    pub use lisa_serve::application::loading::*;
}
#[cfg(feature = "metal")]
pub use bindings::qwen4;
pub use core::tokenizer;
#[cfg(feature = "metal")]
pub use core::{batch, generate, loader, mem, norm, quant, sampler, sched, session};
pub use decision::DecisionModel;
pub use language::LanguageModel;
pub use lisa_engine::{ttft_init, ttft_mark};
#[cfg(feature = "metal")]
pub use lisa_metal::runtime::check_platform_support as check_metal_support;
#[cfg(feature = "metal")]
pub use lisa_metal::shader_pack::initialize as initialize_kernels;
pub use loading::Loaded;
#[cfg(feature = "metal")]
pub mod bindings;
#[cfg(feature = "metal")]
pub mod cli;
#[cfg(feature = "metal")]
pub mod image_generation {
    pub use lisa_engine::image_generation::{Options, is_model_type};
    pub type Pipeline = lisa_serve::application::ImagePipeline;
}
#[cfg(feature = "metal")]
pub mod metal_context;
#[cfg(feature = "metal")]
pub mod metal_language;
#[cfg(feature = "metal")]
mod metal_speculation;
#[cfg(feature = "metal")]
pub(crate) mod metal_storage;
#[cfg(feature = "metal")]
pub mod vision;

#[cfg(all(test, feature = "metal"))]
mod laya_plan_tests;
