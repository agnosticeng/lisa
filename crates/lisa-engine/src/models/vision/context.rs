//! Request-local vision state. Image prompts never share the text prefix cache.
use anyhow::{Context, ensure};
use lisa_mlx::ops::indexing::IndexOp;
use lisa_mlx::{Array, ops};
use serde_json::Value;
use std::path::{Path, PathBuf};

struct ImageSpan {
    start: usize,
    features: Array,
}
struct Prompt {
    positions: Vec<[i32; 3]>,
    delta: i32,
    spans: Vec<ImageSpan>,
}

pub struct Vision {
    dir: PathBuf,
    glm: bool,
    image_id: u32,
    start_id: u32,
    end_id: u32,
    selector: Vec<usize>,
    prompt: Option<Prompt>,
}

impl Vision {
    pub fn from_dir(dir: &Path, rotary_dimensions: usize) -> anyhow::Result<Option<Self>> {
        let root: Value = serde_json::from_slice(&std::fs::read(dir.join("config.json"))?)?;
        if root.get("vision_config").is_none() {
            return Ok(None);
        }
        let token = |name: &str| -> anyhow::Result<u32> {
            u32::try_from(
                root.get(name)
                    .and_then(Value::as_u64)
                    .with_context(|| format!("missing {name}"))?,
            )
            .context("vision token id exceeds u32")
        };
        if root
            .get("model_type")
            .and_then(Value::as_str)
            .is_some_and(|s| s.starts_with("glm5"))
        {
            return Ok(Some(Self {
                dir: dir.into(),
                glm: true,
                image_id: token("image_token_id")?,
                start_id: token("image_start_token_id")?,
                end_id: token("image_end_token_id")?,
                selector: Vec::new(),
                prompt: None,
            }));
        }
        let text = root.get("text_config").unwrap_or(&root);
        let rope = text
            .get("rope_parameters")
            .or_else(|| text.get("rope_scaling"))
            .context("vision checkpoint has no M-RoPE parameters")?;
        let sections = rope
            .get("mrope_section")
            .and_then(Value::as_array)
            .context("missing mrope_section")?;
        ensure!(
            sections.len() == 3,
            "mrope_section must contain three lengths"
        );
        let lengths: Vec<usize> = sections
            .iter()
            .map(|v| {
                v.as_u64()
                    .map(|x| x as usize)
                    .context("invalid mrope_section")
            })
            .collect::<anyhow::Result<_>>()?;
        ensure!(
            rotary_dimensions > 0
                && rotary_dimensions % 2 == 0
                && lengths.iter().sum::<usize>() == rotary_dimensions / 2,
            "M-RoPE sections do not match the rotary width"
        );
        let mut selector = vec![0; rotary_dimensions / 2];
        if rope
            .get("mrope_interleaved")
            .and_then(Value::as_bool)
            .unwrap_or(false)
        {
            for axis in 1..3 {
                for i in (axis..(lengths[axis] * 3).min(selector.len())).step_by(3) {
                    selector[i] = axis;
                }
            }
        } else {
            let mut offset = 0;
            for axis in 0..3 {
                for v in &mut selector[offset..offset + lengths[axis]] {
                    *v = axis;
                }
                offset += lengths[axis];
            }
        }
        Ok(Some(Self {
            dir: dir.into(),
            glm: false,
            image_id: token("image_token_id")?,
            start_id: token("vision_start_token_id")?,
            end_id: token("vision_end_token_id")?,
            selector,
            prompt: None,
        }))
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
        let markers: Vec<usize> = tokens
            .iter()
            .enumerate()
            .filter_map(|(i, t)| (*t == self.image_id).then_some(i))
            .collect();
        ensure!(
            markers.len() == images.len(),
            "prompt has {} image markers for {} images",
            markers.len(),
            images.len()
        );
        for &i in &markers {
            ensure!(
                i > 0 && tokens[i - 1] == self.start_id && tokens.get(i + 1) == Some(&self.end_id),
                "image placeholder must be enclosed by the checkpoint's vision start/end tokens"
            );
        }
        let qwen = if self.glm {
            None
        } else {
            Some(super::qwen::Encoder::load(&self.dir)?)
        };
        let glm = if self.glm {
            Some(super::glm::Encoder::load(&self.dir)?)
        } else {
            None
        };
        let merge = if let Some(e) = &qwen {
            e.config.spatial_merge_size
        } else {
            glm.as_ref().unwrap().config.spatial_merge_size
        };
        let mut expanded = Vec::new();
        let mut positions = Vec::new();
        let mut spans = Vec::new();
        let mut base = 0i32;
        let mut image_index = 0;
        let mut total_len = tokens.len();
        for &token in tokens.iter() {
            if token != self.image_id {
                expanded.push(token);
                positions.push([base; 3]);
                base += 1;
                continue;
            }
            let (features, h, w) = if let Some(e) = &qwen {
                e.encode(&images[image_index])?
            } else {
                glm.as_ref().unwrap().encode(&images[image_index])?
            };
            let (h, w) = (h / merge, w / merge);
            let count = h * w;
            total_len += count - 1;
            ensure!(
                total_len <= context_limit,
                "image prompt exceeds model context capacity {context_limit}"
            );
            spans.push(ImageSpan {
                start: expanded.len(),
                features,
            });
            for y in 0..h {
                for x in 0..w {
                    expanded.push(self.image_id);
                    positions.push([base, base + y as i32, base + x as i32]);
                }
            }
            base += h.max(w) as i32;
            image_index += 1;
        }
        ensure!(
            expanded.len() <= context_limit,
            "image prompt exceeds model context capacity"
        );
        let delta = base - expanded.len() as i32;
        self.prompt = Some(Prompt {
            positions,
            delta,
            spans,
        });
        *tokens = expanded;
        // The encoder is released before text prefill; it is part of the same
        // checkpoint but its resident weights need not overlap the KV peak.
        Ok(())
    }

    pub fn positions(&self, offset: usize, count: usize) -> Option<Array> {
        if self.glm {
            return None;
        }
        let prompt = self.prompt.as_ref()?;
        let mut values = Vec::with_capacity(count * self.selector.len());
        for i in offset..offset + count {
            let p = prompt
                .positions
                .get(i)
                .copied()
                .unwrap_or([i as i32 + prompt.delta; 3]);
            values.extend(self.selector.iter().map(|&axis| p[axis] as f32));
        }
        Some(Array::from_slice(
            &values,
            &[1, count as i32, self.selector.len() as i32],
        ))
    }

    pub fn splice(&self, hidden: Array, offset: usize) -> anyhow::Result<Array> {
        let Some(prompt) = &self.prompt else {
            return Ok(hidden);
        };
        ensure!(
            hidden.dim(0) == 1,
            "image prompts require a single-stream session"
        );
        let count = hidden.dim(1) as usize;
        let end = offset + count;
        let mut parts = Vec::new();
        let mut cursor = offset;
        for span in &prompt.spans {
            let start = span.start.max(offset);
            let stop = (span.start + span.features.dim(1) as usize).min(end);
            if start >= stop {
                continue;
            }
            if cursor < start {
                parts.push(hidden.index((
                    ..,
                    (cursor - offset) as i32..(start - offset) as i32,
                    ..,
                )));
            }
            parts.push(
                span.features
                    .index((
                        ..,
                        (start - span.start) as i32..(stop - span.start) as i32,
                        ..,
                    ))
                    .as_dtype(hidden.dtype())?,
            );
            cursor = stop;
        }
        if cursor == offset {
            return Ok(hidden);
        }
        if cursor < end {
            parts.push(hidden.index((.., (cursor - offset) as i32..count as i32, ..)));
        }
        let refs: Vec<&Array> = parts.iter().collect();
        Ok(ops::concatenate(&refs, 1)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn image_spans_cross_prefill_chunks_and_clear_between_requests() -> anyhow::Result<()> {
        let features = Array::from_slice(&[10f32, 11., 20., 21., 30., 31.], &[1, 3, 2]);
        let mut vision = Vision {
            dir: PathBuf::new(),
            glm: false,
            image_id: 1,
            start_id: 2,
            end_id: 3,
            selector: vec![0, 1, 2],
            prompt: Some(Prompt {
                positions: vec![[0, 0, 0], [1, 1, 1], [2, 2, 2], [2, 2, 3], [2, 3, 2]],
                delta: -1,
                spans: vec![ImageSpan { start: 2, features }],
            }),
        };
        let hidden = Array::from_slice(&[1f32; 8], &[1, 4, 2]);
        let first = vision.splice(hidden.clone(), 0)?;
        first.eval()?;
        assert_eq!(
            first.as_slice::<f32>(),
            &[1., 1., 1., 1., 10., 11., 20., 21.]
        );
        let next = vision.splice(hidden.clone(), 4)?;
        next.eval()?;
        assert_eq!(next.as_slice::<f32>(), &[30., 31., 1., 1., 1., 1., 1., 1.]);
        let positions = vision.positions(3, 3).unwrap();
        positions.eval()?;
        assert_eq!(
            positions.as_slice::<f32>(),
            &[2., 2., 3., 2., 3., 2., 4., 4., 4.]
        );
        vision.clear();
        assert!(vision.positions(0, 4).is_none());
        let plain = vision.splice(hidden, 0)?;
        plain.eval()?;
        assert_eq!(plain.as_slice::<f32>(), &[1.; 8]);
        Ok(())
    }
}
