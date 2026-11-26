use crate::{bindings::BindingDevice, execution::DeviceContext};
use lisa_backend::{ScalarType, TensorAccess};
// Tokenization, prepared tensors and cache handles for FLUX 3 text plans.
use crate::{
    bindings::flux2::weights::{Weights, info},
    core::tokenizer::Tokenizer,
};
use anyhow::{Context, ensure};

use lisa_models::flux::{symbolic::Schema, text3 as model};
use model::DEPTH;
type Cache<T> = Vec<Option<(T, T)>>;
pub struct Encoder<D: BindingDevice> {
    pub context: DeviceContext<D>,
    weights: Weights<D>,
    schema: Schema,
    tokenizer: Tokenizer,
}
impl<D: BindingDevice> Encoder<D> {
    pub fn from_prepared(
        context: DeviceContext<D>,
        weights: Weights<D>,
        schema: Schema,
        tokenizer: Tokenizer,
    ) -> Self {
        Self {
            context,
            weights,
            schema,
            tokenizer,
        }
    }
    fn forward(
        &self,
        ids: &[u32],
        real: usize,
        capture: bool,
        cache: Option<&mut Cache<D::Tensor>>,
    ) -> anyhow::Result<D::Tensor> {
        let cache_schema = cache
            .as_ref()
            .map(|cache| {
                cache
                    .iter()
                    .map(|entry| {
                        entry
                            .as_ref()
                            .map(|(k, v)| Ok((info(&self.context, k)?, info(&self.context, v)?)))
                            .transpose()
                    })
                    .collect::<anyhow::Result<Vec<_>>>()
            })
            .transpose()?;
        let encoding = model::program(
            &self.schema,
            model::Geometry::default(),
            ids.len(),
            real,
            capture,
            cache_schema.as_ref(),
        )?;
        let ids = crate::bindings::flux2::weights::pull(
            &self.context,
            &ids.iter().flat_map(|x| x.to_ne_bytes()).collect::<Vec<_>>(),
            &[ids.len() as i32],
            ScalarType::U32,
        )?;
        let mut inputs = vec![&ids];
        if let Some(cache) = &cache {
            for (k, v) in cache.iter().flatten() {
                inputs.extend([k, v]);
            }
        }
        let prepared = self.weights.prepare(encoding.program)?;
        let mut outputs = prepared.run(&inputs, &mut [])?.into_iter();
        let out = outputs.next().context("missing encoder output")?;
        if let Some(cache) = cache {
            for i in encoding.cache_layers {
                cache[i] = Some((
                    outputs.next().context("missing encoder key")?,
                    outputs.next().context("missing encoder value")?,
                ));
            }
        }
        out.eval()?;
        Ok(out)
    }
    pub fn encode(&self, caption: &str) -> anyhow::Result<D::Tensor> {
        let text = model::prompt(caption);
        let mut ids = self.tokenizer.encode(&text, false)?;
        let pad = self
            .tokenizer
            .inner
            .token_to_id("<|endoftext|>")
            .context("missing Qwen padding token")?;
        let real = model::pad(&mut ids, pad);
        self.forward(&ids, real, true, None)
    }
    /// Uses the bundled instruction model, with a compact adaptation of BFL's
    /// external-LLM caption guide. No second LLM is resident or downloaded.
    pub fn expand(
        &self,
        prompt: &str,
        width: u32,
        height: u32,
        progress: &mut dyn FnMut() -> anyhow::Result<()>,
    ) -> anyhow::Result<String> {
        let input = model::caption_prompt(prompt, width, height);
        let mut ids = self.tokenizer.encode(&input, false)?;
        ensure!(
            ids.len() <= 8192,
            "prompt is too long for local caption expansion"
        );
        let mut cache: Cache<D::Tensor> = (0..DEPTH).map(|_| None).collect();
        let mut generated = Vec::new();
        for step in 0..512 {
            if step % 16 == 0 {
                progress()?;
            }
            let logits = self.forward(&ids, ids.len(), false, Some(&mut cache))?;
            let next = crate::bindings::flux2::weights::run(
                &self.context,
                model::select_token(),
                &[&logits],
            )?;
            next.eval()?;
            let token = next
                .read_raw_ids()?
                .into_iter()
                .next()
                .context("missing caption token")?;
            if self.tokenizer.im_end_ids.contains(&token) {
                break;
            }
            generated.push(token);
            ids.clear();
            ids.push(token);
        }
        let caption = self.tokenizer.decode_clean(&generated)?.trim().to_owned();
        ensure!(
            !caption.is_empty(),
            "local caption expansion returned no text"
        );
        Ok(caption)
    }
}
