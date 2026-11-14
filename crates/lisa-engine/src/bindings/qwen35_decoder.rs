//! Binding of a model-owned decoder schedule to its weights and cache storage.
use super::{dense::DenseMlp, gated_delta::GatedDeltaNet, qwen35_attention::Qwen35Attention};
use crate::{bindings::BindingDevice, execution::DeviceContext};
use crate::{bindings::normalization::RmsNorm, cache::LayerCache, weights::source::Weights};
pub struct DecoderLayer<D: BindingDevice> {
    pub context: DeviceContext<D>,
    pub plan: crate::core::decoder::PreparedDecoder<8>,
    pub attn: Option<Qwen35Attention<D>>,
    pub gdn: Option<GatedDeltaNet<D>>,
    pub input_norm: RmsNorm<D::Tensor>,
    pub post_norm: RmsNorm<D::Tensor>,
    pub mlp: DenseMlp<D>,
}

impl<D: BindingDevice> DecoderLayer<D> {
    /// The MTP head's decoder layer: the same shape as a trunk layer but built
    /// from `mtp.layers.0.*` weights (always full attention, dense MLP).
    #[allow(clippy::too_many_arguments)]
    pub fn new_mtp(
        context: DeviceContext<D>,
        w: &mut Weights<D::Tensor>,
        prefix: &str,
        eps: f32,
        heads: usize,
        kv_heads: usize,
        head_dim: usize,
        gated: bool,
        gs: i32,
        bits: i32,
    ) -> anyhow::Result<Self> {
        let mlp = DenseMlp::load(context.clone(), w, &format!("{prefix}.mlp"), gs, bits)?;
        Ok(Self {
            context: context.clone(),
            plan: crate::core::decoder::PreparedDecoder::new(lisa_models::qwen3_5::decoder(true))?,
            attn: Some(Qwen35Attention::load(
                context.clone(),
                w,
                &format!("{prefix}.self_attn"),
                eps,
                heads,
                kv_heads,
                head_dim,
                gated,
                gs,
                bits,
            )?),
            gdn: None,
            input_norm: RmsNorm::load_on(
                &context,
                w,
                &format!("{prefix}.input_layernorm"),
                eps,
                None,
            )?,
            post_norm: RmsNorm::load_on(
                &context,
                w,
                &format!("{prefix}.post_attention_layernorm"),
                eps,
                None,
            )?,
            mlp,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &mut self,
        x: &D::Tensor,
        m_prev: Option<D::Tensor>,
        cache: Option<&mut LayerCache<D>>,
        cos_bf: &D::Tensor,
        sin_bf: &D::Tensor,
        offset: usize,
        capture: bool,
        replay: bool,
    ) -> anyhow::Result<(D::Tensor, D::Tensor)> {
        struct Bindings<'a, D: BindingDevice> {
            context: &'a DeviceContext<D>,
            attn: Option<&'a Qwen35Attention<D>>,
            gdn: Option<&'a mut GatedDeltaNet<D>>,
            input_norm: &'a RmsNorm<D::Tensor>,
            post_norm: &'a RmsNorm<D::Tensor>,
            mlp: &'a DenseMlp<D>,
            cache: Option<&'a mut LayerCache<D>>,
            cos: &'a D::Tensor,
            sin: &'a D::Tensor,
            offset: usize,
            capture: bool,
            replay: bool,
        }
        impl<D: BindingDevice> crate::core::decoder::Blocks for Bindings<'_, D> {
            type Tensor = D::Tensor;
            fn residual_norm(
                &mut self,
                norm: usize,
                input: &D::Tensor,
                residual: Option<&D::Tensor>,
            ) -> anyhow::Result<(D::Tensor, D::Tensor)> {
                let norm = match norm {
                    0 => self.input_norm,
                    1 => self.post_norm,
                    _ => anyhow::bail!("unknown bound normalization slot"),
                };
                match residual {
                    Some(r) => Ok(norm.forward_residual_add_on(self.context, input, r)?),
                    None => Ok((input.clone(), norm.forward_on(self.context, input)?)),
                }
            }
            fn mix(
                &mut self,
                kind: lisa_models::decoder::Mixer,
                input: &D::Tensor,
            ) -> anyhow::Result<D::Tensor> {
                match kind {
                    lisa_models::decoder::Mixer::Attention => {
                        let block = self
                            .attn
                            .ok_or_else(|| anyhow::anyhow!("attention block is not bound"))?;
                        let cache = match self.cache.as_deref_mut() {
                            Some(LayerCache::Full(c)) => Some(c),
                            _ => None,
                        };
                        Ok(block.forward(input, self.cos, self.sin, cache, self.offset)?)
                    }
                    lisa_models::decoder::Mixer::GatedDelta => {
                        let block = self
                            .gdn
                            .as_deref_mut()
                            .ok_or_else(|| anyhow::anyhow!("gated-delta block is not bound"))?;
                        let cache = match self.cache.as_deref_mut() {
                            Some(LayerCache::Linear(c)) => Some(c),
                            _ => None,
                        };
                        Ok(block.forward_with_replay(input, cache, self.capture, self.replay)?)
                    }
                }
            }
            fn feed_forward(&mut self, input: &D::Tensor) -> anyhow::Result<D::Tensor> {
                Ok(self.mlp.forward(input)?)
            }
        }
        self.plan.run(
            &mut Bindings {
                context: &self.context,
                attn: self.attn.as_ref(),
                gdn: self.gdn.as_mut(),
                input_norm: &self.input_norm,
                post_norm: &self.post_norm,
                mlp: &self.mlp,
                cache,
                cos: cos_bf,
                sin: sin_bf,
                offset,
                capture,
                replay,
            },
            x,
            m_prev,
        )
    }
}
