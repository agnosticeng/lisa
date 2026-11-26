//! The `qwen3_5` dense hybrid tower: embeddings, 64 pre-norm layers (GDN or
//! full attention), a dense SwiGLU MLP per layer, the final norm and `lm_head`.

use crate::{bindings::BindingDevice, execution::DeviceContext};
use lisa_backend::{ManagedDevice, TensorAccess};

use crate::bindings::shared::normalization::{RmsNorm, Rotary};
use crate::cache::LayerCache;
use crate::weights::affine::{QuantizedEmbedding, QuantizedLinear};
use lisa_models::qwen3_5::config::Qwen35Config;

use crate::bindings::qwen35::decoder::DecoderLayer;

use lisa_models::qwen3_5::SERIAL_CAPTURE_COST_SUFFIX;
#[cfg(test)]
use lisa_models::qwen3_5::cache_law;

fn native_mtp_cost_key(config: &Qwen35Config) -> String {
    // Auto prices the captured target and verified head under the serial
    // capture schedule.
    format!(
        "qwen3_5-{}L-{}h{SERIAL_CAPTURE_COST_SUFFIX}",
        config.num_hidden_layers, config.hidden_size,
    )
}

pub struct MlxExecution<D: BindingDevice + ManagedDevice> {
    pub(crate) context: DeviceContext<D>,
    pub(crate) cache_law: u64,
    pub(crate) vision: Option<crate::vision::Vision<D>>,
    pub(crate) embed_tokens: QuantizedEmbedding<D::Tensor>,
    pub(crate) layers: Vec<DecoderLayer<D>>,
    pub(crate) norm: RmsNorm<D::Tensor>,
    pub(crate) lm_head: QuantizedLinear<D::Tensor>,
    /// Draft-only 2-bit requantized copy (`draft_head`); Some only
    /// when the checkpoint carries an MTP head. Never used by verify.
    pub(crate) draft_head: Option<QuantizedLinear<D::Tensor>>,
    pub(crate) rope: Rotary,
    /// The native MTP head (`mtp/weights.safetensors`), when the checkpoint
    /// ships one. `has_drafter()` is true iff this is `Some`.
    pub(crate) mtp: Option<crate::bindings::qwen35::mtp::MtpHead<D>>,
    pub config: Qwen35Config,
}

impl<D: BindingDevice + ManagedDevice> crate::bindings::qwen35::execution::Execution<D>
    for MlxExecution<D>
where
    D::Error: From<anyhow::Error>,
{
    fn numerical_law(&self) -> Option<u64> {
        Some(self.cache_law)
    }
    fn layer(
        &mut self,
        i: usize,
        state: &mut crate::bindings::qwen35::execution::Forward<D::Tensor>,
        cache: Option<&mut LayerCache<D>>,
        capture: bool,
    ) -> anyhow::Result<()> {
        let (cos, sin) = state.rope.as_ref().expect("affine Qwen rotary context");
        let (x, m) = self.layers[i].forward(
            &state.x,
            state.residual.take(),
            cache,
            cos,
            sin,
            state.offset,
            capture,
            capture,
        )?;
        state.x = x;
        state.residual = Some(m);
        Ok(())
    }
    fn finish(
        &self,
        state: crate::bindings::qwen35::execution::Forward<D::Tensor>,
    ) -> anyhow::Result<D::Tensor> {
        Ok(self
            .norm
            .forward_residual_add_on(
                &self.context,
                &state.x,
                &state.residual.expect("nonempty Qwen tower"),
            )?
            .1)
    }
    fn config(&self) -> &Qwen35Config {
        &self.config
    }
    fn begin(
        &self,
        ids: &D::Tensor,
        caches: Option<&Vec<LayerCache<D>>>,
    ) -> anyhow::Result<crate::bindings::qwen35::execution::Forward<D::Tensor>> {
        let s = ids.dim(1) as usize;
        let offset = caches
            .and_then(|c| {
                c.iter().find_map(|l| match l {
                    LayerCache::Full(f) => Some(f.offset),
                    _ => None,
                })
            })
            .unwrap_or(0);
        let next = caches.and_then(|c| {
            c.iter().find_map(|l| match l {
                LayerCache::Full(f) => f.next_pos.as_deref(),
                _ => None,
            })
        });
        let (x, cos_bf, sin_bf) = crate::bindings::qwen35::entry::begin(
            &self.context,
            ids,
            &self.embed_tokens,
            &self.rope,
            self.vision.as_ref(),
            offset,
            s,
            next,
        )?;
        Ok(crate::bindings::qwen35::execution::Forward {
            x,
            residual: None,
            offset,
            rope: Some((cos_bf, sin_bf)),
        })
    }
    fn head(&self, mixed: &D::Tensor) -> std::result::Result<D::Tensor, D::Error> {
        crate::bindings::shared::affine::linear(&self.context, mixed, &self.lm_head)
            .map_err(D::Error::from)
    }
    fn head_verified(&self, mixed: &D::Tensor) -> std::result::Result<D::Tensor, D::Error> {
        // The vocabulary projection has no admitted verify reduction exception.
        crate::bindings::shared::affine::linear_serial(&self.context, mixed, &self.lm_head, false)
            .map_err(D::Error::from)
    }
    fn supports_images(&self) -> bool {
        self.vision.is_some()
    }
    fn prepare_images(
        &mut self,
        tokens: &mut Vec<u32>,
        images: &[image::RgbImage],
    ) -> anyhow::Result<()> {
        let limit = self.config.max_position_embeddings;
        self.vision
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("checkpoint has no supported vision encoder"))?
            .prepare(tokens, images, limit)
    }
    fn clear_images(&mut self) {
        if let Some(v) = &mut self.vision {
            v.clear();
        }
    }
    fn has_drafter(&self) -> bool {
        self.mtp.is_some()
    }
    fn mtp_cost_key(&self) -> String {
        native_mtp_cost_key(&self.config)
    }
    fn drafter_reset(&mut self) {
        if let Some(h) = self.mtp.as_mut() {
            h.reset_caches();
        }
    }
    fn drafter_trim(&mut self, n: usize) {
        if let Some(h) = self.mtp.as_mut() {
            h.trim_caches(n);
        }
    }
    fn drafter_offset(&self) -> usize {
        self.mtp.as_ref().map_or(0, |h| h.cache_offset())
    }
    fn drafter_restore_offset(&mut self, n: usize) {
        if let Some(h) = self.mtp.as_mut() {
            h.restore_offset(n);
        }
    }
    fn draft_step(
        &mut self,
        tokens: &D::Tensor,
        multi: &D::Tensor,
    ) -> anyhow::Result<(D::Tensor, D::Tensor)> {
        let offset = self.mtp.as_ref().map(|h| h.cache_offset()).unwrap_or(0);
        let head = self
            .mtp
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("this model has no MTP head"))?;
        // `multi` for qwen3_5 is the trunk's final-norm hidden — the same
        // vector lm_head consumes, which is the head's `hidden` input.
        let (_, last) = head.forward(tokens, multi, &self.embed_tokens, offset)?;
        let selected = crate::bindings::qwen35::readout::select(
            &self.context,
            &last,
            &self.lm_head,
            self.draft_head.as_ref(),
        )?;
        crate::core::oracle::push_arrays(selected.candidates.as_ref(), &selected.scores)?;
        let draft_id = selected.token;
        Ok((draft_id, last))
    }
}

#[cfg(test)]
mod cache_law_tests {
    use super::*;
    #[test]
    fn native_mtp_costs_exclude_legacy_capture_schedule() -> anyhow::Result<()> {
        let mut config = Qwen35Config::parse("{}")?;
        config.num_hidden_layers = 64;
        config.hidden_size = 5120;
        assert_eq!(
            native_mtp_cost_key(&config),
            "qwen3_5-64L-5120h-serial-capture-v3"
        );
        assert_ne!(native_mtp_cost_key(&config), "qwen3_5-64L-5120h");
        assert_ne!(
            native_mtp_cost_key(&config),
            "qwen3_5-64L-5120h-serial-capture-v2"
        );
        Ok(())
    }

    #[test]
    fn qwen_mlx_cache_law_is_available_without_lazy_pipelines() -> anyhow::Result<()> {
        let mut config = Qwen35Config::parse("{}")?;
        let law = cache_law(&config, b"test_s\x01");
        assert_eq!(law, cache_law(&config, b"test_s\x01"));
        assert_ne!(law, cache_law(&config, b"test_d\x01"));
        assert_ne!(law, cache_law(&config, b"test_s\x00"));
        config.quant_bits = 2;
        assert_ne!(law, cache_law(&config, b"test_s\x01"));
        Ok(())
    }
}
