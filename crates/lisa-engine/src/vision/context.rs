//! Request-owned image features and bindings for model-owned prompt laws.
use crate::core::graph::PlanCache;
use crate::{bindings::BindingDevice, execution::DeviceContext};
use anyhow::ensure;
use lisa_backend::ScalarType;
use lisa_backend::{HostTensor, ManagedDevice, TensorAccess};
use lisa_models::vision::prompt::{Config, Expansion, Positions, Splice};
pub struct ImageSpan<T> {
    pub start: usize,
    pub features: T,
}
pub struct Prompt<T> {
    pub positions: Positions,
    pub spans: Vec<ImageSpan<T>>,
}
/// An image encoder contains prepared tensors and no checkpoint reader.
pub enum Encoder<D: BindingDevice + ManagedDevice> {
    Qwen(super::qwen::Encoder<D>),
    Glm(super::glm::Encoder<D>),
}
impl<D: BindingDevice + ManagedDevice> Encoder<D> {
    fn spatial_merge_size(&self) -> usize {
        match self {
            Self::Qwen(e) => e.config.spatial_merge_size,
            Self::Glm(e) => e.config.spatial_merge_size,
        }
    }
    fn encode(&self, image: &image::RgbImage) -> anyhow::Result<(D::Tensor, usize, usize)> {
        match self {
            Self::Qwen(e) => e.encode(image),
            Self::Glm(e) => e.encode(image),
        }
    }
}
/// Request-scoped acquisition keeps image weights out of the text KV peak.
pub trait EncoderSource<D: BindingDevice + ManagedDevice> {
    fn load(&self) -> anyhow::Result<Encoder<D>>;
}
pub struct Vision<D: BindingDevice + ManagedDevice> {
    pub context: DeviceContext<D>,
    source: Box<dyn EncoderSource<D>>,
    pub config: Config,
    pub prompt: Option<Prompt<D::Tensor>>,
}

impl<D: BindingDevice + ManagedDevice> Vision<D> {
    pub fn from_prepared(
        context: DeviceContext<D>,
        config: Config,
        source: Box<dyn EncoderSource<D>>,
    ) -> Self {
        Self {
            context,
            source,
            config,
            prompt: None,
        }
    }
    pub fn active(&self) -> bool {
        self.prompt.is_some()
    }
    pub fn clear(&mut self) {
        self.prompt = None;
    }
    pub fn prepare(
        &mut self,
        tokens: &mut Vec<u32>,
        images: &[image::RgbImage],
        context_limit: usize,
    ) -> anyhow::Result<()> {
        self.clear();
        ensure!(
            !images.is_empty() && images.len() <= super::MAX_IMAGES,
            "expected 1..={} images",
            super::MAX_IMAGES
        );
        self.config.validate_markers(tokens, images.len())?;
        let encoder = self.source.load()?;
        let merge = encoder.spatial_merge_size();
        let mut expansion = Expansion::default();
        let mut spans = Vec::new();
        let mut image_index = 0;
        let mut total_len = tokens.len();
        for &token in tokens.iter() {
            if token != self.config.image_id {
                expansion.text(token)?;
                continue;
            }
            let (features, h, w) = encoder.encode(&images[image_index])?;
            let (h, w) = (h / merge, w / merge);
            total_len = total_len
                .checked_add(h * w - 1)
                .ok_or_else(|| anyhow::anyhow!("image prompt length overflow"))?;
            ensure!(
                total_len <= context_limit,
                "image prompt exceeds model context capacity {context_limit}"
            );
            let start = expansion.image(token, h, w, context_limit)?;
            spans.push(ImageSpan { start, features });
            image_index += 1;
        }
        let (expanded, positions) = expansion.finish(context_limit)?;
        self.prompt = Some(Prompt { positions, spans });
        *tokens = expanded;
        // Drop the encoder before text prefill, so its weights do not overlap KV peak.
        Ok(())
    }
    pub fn positions(&self, offset: usize, count: usize) -> anyhow::Result<Option<D::Tensor>> {
        if self.config.glm {
            return Ok(None);
        }
        let Some(prompt) = self.prompt.as_ref() else {
            return Ok(None);
        };
        let values = prompt
            .positions
            .select(&self.config.selector, offset, count);
        Ok(Some(
            self.context.device().pull(
                HostTensor {
                    bytes: &values
                        .iter()
                        .flat_map(|v| v.to_ne_bytes())
                        .collect::<Vec<_>>(),
                    shape: &[1, count as i32, self.config.selector.len() as i32],
                    dtype: ScalarType::F32,
                }
                .into(),
                Default::default(),
            )?,
        ))
    }
    pub fn splice_bindings(
        &self,
        shape: Vec<i32>,
        dtype: ScalarType,
        offset: usize,
    ) -> Option<(Splice, Vec<&D::Tensor>)> {
        let prompt = self.prompt.as_ref()?;
        Some((
            Splice {
                shape,
                dtype,
                offset,
                spans: prompt
                    .spans
                    .iter()
                    .map(|s| (s.start, s.features.dim(1) as usize))
                    .collect(),
            },
            prompt.spans.iter().map(|s| &s.features).collect(),
        ))
    }
    pub fn splice(&self, hidden: D::Tensor, offset: usize) -> anyhow::Result<D::Tensor> {
        let Some(prompt) = &self.prompt else {
            return Ok(hidden);
        };
        let dtype = match self.context.device().dtype(&hidden) {
            Some(ScalarType::F32) => ScalarType::F32,
            Some(ScalarType::F16) => ScalarType::F16,
            Some(ScalarType::BF16) => ScalarType::BF16,
            _ => anyhow::bail!("unsupported image embedding dtype"),
        };
        let spec = Splice {
            shape: hidden.shape().to_vec(),
            dtype,
            offset,
            spans: prompt
                .spans
                .iter()
                .map(|s| (s.start, s.features.dim(1) as usize))
                .collect(),
        };
        let mut inputs = vec![&hidden];
        inputs.extend(prompt.spans.iter().map(|s| &s.features));
        self.context.with_slot(
            "vision.splice",
            || std::cell::RefCell::new(PlanCache::new(self.context.device().clone(), 16)),
            |cache| {
                cache
                    .borrow_mut()
                    .try_get_or_prepare(spec.clone(), || spec.graph())?
                    .run(&inputs, &mut [])?
                    .into_one()
                    .map_err(anyhow::Error::msg)
            },
        )
    }
}
