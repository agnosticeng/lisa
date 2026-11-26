//! Application construction of portable GatedDeltaNet bindings.
use crate::core::loader::TensorSource;
pub struct GatedDeltaNet(
    pub lisa_engine::bindings::shared::gated_delta::GatedDeltaNet<lisa_metal::device::MetalDevice>,
);
impl std::ops::Deref for GatedDeltaNet {
    type Target =
        lisa_engine::bindings::shared::gated_delta::GatedDeltaNet<lisa_metal::device::MetalDevice>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl std::ops::DerefMut for GatedDeltaNet {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}
impl GatedDeltaNet {
    pub fn load<S: TensorSource>(
        src: &mut S,
        prefix: &str,
        eps: f32,
        key_heads: usize,
        value_heads: usize,
        key_head_dim: usize,
        value_head_dim: usize,
        conv_kernel_size: usize,
        group_size: i32,
        bits: i32,
    ) -> anyhow::Result<Self> {
        Ok(Self(
            lisa_engine::bindings::shared::gated_delta::GatedDeltaNet::load(
                crate::metal_context::execution_context(),
                src,
                prefix,
                eps,
                key_heads,
                value_heads,
                key_head_dim,
                value_head_dim,
                conv_kernel_size,
                group_size,
                bits,
            )?,
        ))
    }
}
