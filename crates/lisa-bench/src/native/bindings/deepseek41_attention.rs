//! Application construction for portable sparse attention bindings.
use crate::{
    bindings::deepseek41_state::State,
    core::{capture::Capture, gguf_resident::ResidentWeights},
};
use anyhow::Result;
use lisa_metal::{Array, Stream, device::MetalDevice};
#[cfg(test)]
use lisa_models::deepseek41::attention as model;
use lisa_models::deepseek41::config::Config;
pub type Projections = lisa_engine::bindings::deepseek41::attention::Projections<MetalDevice>;
pub struct Scratch(lisa_engine::bindings::deepseek41::attention::Scratch<MetalDevice>);
impl std::ops::Deref for Scratch {
    type Target = lisa_engine::bindings::deepseek41::attention::Scratch<MetalDevice>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl std::ops::DerefMut for Scratch {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}
impl Scratch {
    pub fn new(stream: &Stream) -> Result<Self> {
        Ok(Self(
            lisa_engine::bindings::deepseek41::attention::Scratch::new(&MetalDevice::on_stream(
                stream,
            ))?,
        ))
    }
}
pub struct Attention(lisa_engine::bindings::deepseek41::attention::Attention<MetalDevice>);
impl Attention {
    pub fn load(
        weights: &ResidentWeights,
        layer: usize,
        config: &Config,
        stream: &Stream,
    ) -> Result<Self> {
        Ok(Self(
            lisa_engine::bindings::deepseek41::attention::Attention::load(
                &**weights,
                layer,
                config,
                &MetalDevice::on_stream(stream),
            )?,
        ))
    }
    #[cfg(test)]
    pub fn bound(spec: model::Spec, weights: Vec<Array>, stream: &Stream) -> Result<Self> {
        Ok(Self(
            lisa_engine::bindings::deepseek41::attention::Attention::bound(
                spec,
                weights,
                &MetalDevice::on_stream(stream),
            )?,
        ))
    }
    pub fn project(&self, norm: &Array, _stream: &Stream) -> Result<Projections> {
        self.0.project(norm, &MetalDevice::on_stream(_stream))
    }
    pub fn forward(
        &self,
        norm: &Array,
        projected: Projections,
        state: &mut State,
        selected: &mut Option<Array>,
        candidates: &mut Option<Array>,
        scratch: &mut Scratch,
        rope: &Array,
        capture: Option<&Capture>,
        _stream: &Stream,
    ) -> Result<Array> {
        self.0.forward(
            norm,
            projected,
            state,
            selected,
            candidates,
            &mut **scratch,
            rope,
            capture,
            &MetalDevice::on_stream(_stream),
        )
    }
}
pub fn rope_table(position: u32, stream: &Stream) -> Result<Array> {
    lisa_engine::bindings::deepseek41::attention::rope_table(
        position,
        &MetalDevice::on_stream(stream),
    )
}
#[cfg(test)]
#[path = "deepseek41_attention_tests.rs"]
mod tests;

#[cfg(test)]
use {super::deepseek41_projection as projection, lisa_models::deepseek41::state::Position};
