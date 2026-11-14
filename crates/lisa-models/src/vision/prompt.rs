//! Image token expansion, positional law and feature insertion plans.
use crate::{graph, program::Build};
use anyhow::{Context, ensure};
use lisa_backend::ScalarType;
use serde_json::Value;
#[derive(Clone)]
pub struct Config {
    pub glm: bool,
    pub image_id: u32,
    pub start_id: u32,
    pub end_id: u32,
    pub selector: Vec<usize>,
}
impl Config {
    pub fn parse(root: &Value, rotary_dimensions: usize) -> anyhow::Result<Option<Self>> {
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
                glm: true,
                image_id: token("image_token_id")?,
                start_id: token("image_start_token_id")?,
                end_id: token("image_end_token_id")?,
                selector: Vec::new(),
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
            glm: false,
            image_id: token("image_token_id")?,
            start_id: token("vision_start_token_id")?,
            end_id: token("vision_end_token_id")?,
            selector,
        }))
    }

    pub fn validate_markers(&self, tokens: &[u32], image_count: usize) -> anyhow::Result<()> {
        let markers = tokens
            .iter()
            .enumerate()
            .filter_map(|(i, t)| (*t == self.image_id).then_some(i))
            .collect::<Vec<_>>();
        ensure!(
            markers.len() == image_count,
            "prompt has {} image markers for {} images",
            markers.len(),
            image_count
        );
        for i in markers {
            ensure!(
                i > 0 && tokens[i - 1] == self.start_id && tokens.get(i + 1) == Some(&self.end_id),
                "image placeholder must be enclosed by the checkpoint's vision start/end tokens"
            );
        }
        Ok(())
    }
}
#[derive(Default)]
pub struct Expansion {
    pub tokens: Vec<u32>,
    pub positions: Vec<[i32; 3]>,
    base: i32,
}
impl Expansion {
    pub fn text(&mut self, token: u32) -> anyhow::Result<()> {
        let next = self
            .base
            .checked_add(1)
            .context("vision position overflow")?;
        self.tokens.push(token);
        self.positions.push([self.base; 3]);
        self.base = next;
        Ok(())
    }
    pub fn image(&mut self, token: u32, h: usize, w: usize, limit: usize) -> anyhow::Result<usize> {
        ensure!(h > 0 && w > 0, "empty image feature grid");
        let count = h.checked_mul(w).context("image feature count overflow")?;
        ensure!(
            self.tokens
                .len()
                .checked_add(count)
                .is_some_and(|n| n <= limit),
            "image prompt exceeds model context capacity {limit}"
        );
        let next = self
            .base
            .checked_add(i32::try_from(h.max(w))?)
            .context("vision position overflow")?;
        let start = self.tokens.len();
        for y in 0..h {
            for x in 0..w {
                self.tokens.push(token);
                self.positions
                    .push([self.base, self.base + y as i32, self.base + x as i32]);
            }
        }
        self.base = next;
        Ok(start)
    }
    pub fn finish(self, limit: usize) -> anyhow::Result<(Vec<u32>, Positions)> {
        ensure!(
            self.tokens.len() <= limit,
            "image prompt exceeds model context capacity"
        );
        let delta = self.base - i32::try_from(self.tokens.len())?;
        Ok((
            self.tokens,
            Positions {
                values: self.positions,
                delta,
            },
        ))
    }
}
pub struct Positions {
    pub values: Vec<[i32; 3]>,
    pub delta: i32,
}
impl Positions {
    pub fn select(&self, selector: &[usize], offset: usize, count: usize) -> Vec<f32> {
        let mut values = Vec::with_capacity(count * selector.len());
        for i in offset..offset + count {
            let p = self
                .values
                .get(i)
                .copied()
                .unwrap_or([i as i32 + self.delta; 3]);
            values.extend(selector.iter().map(|&axis| p[axis] as f32));
        }
        values
    }
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Splice {
    pub shape: Vec<i32>,
    pub dtype: ScalarType,
    pub offset: usize,
    /// (absolute start token, number of image tokens); features are inputs 1..N.
    pub spans: Vec<(usize, usize)>,
}
impl Splice {
    pub fn graph(&self) -> anyhow::Result<graph::Graph> {
        ensure!(
            self.shape.len() == 3 && self.shape[0] == 1 && self.shape[1] > 0 && self.shape[2] > 0,
            "image prompts require a single-stream session"
        );
        let count = self.shape[1] as usize;
        let end = self
            .offset
            .checked_add(count)
            .context("image splice range overflow")?;
        let schema = Default::default();
        let mut p = Build::new(self.spans.len() + 1, &schema);
        let hidden = graph::Value::Input(0);
        let mut parts = Vec::new();
        let mut cursor = self.offset;
        for (i, &(span_start, len)) in self.spans.iter().enumerate() {
            let start = span_start.max(self.offset);
            let stop = span_start
                .checked_add(len)
                .context("image span overflow")?
                .min(end);
            if start >= stop {
                continue;
            }
            ensure!(cursor <= start, "overlapping or unordered image spans");
            if cursor < start {
                parts.push(p.slice(
                    hidden,
                    1,
                    i32::try_from(cursor - self.offset)?,
                    i32::try_from(start - self.offset)?,
                ));
            }
            let features = p.slice(
                graph::Value::Input(i + 1),
                1,
                i32::try_from(start - span_start)?,
                i32::try_from(stop - span_start)?,
            );
            parts.push(p.cast(features, self.dtype));
            cursor = stop;
        }
        if cursor == self.offset {
            return Ok(p.finish(vec![hidden])?.graph);
        }
        if cursor < end {
            parts.push(p.slice(
                hidden,
                1,
                i32::try_from(cursor - self.offset)?,
                count as i32,
            ));
        }
        let output = p.concat(parts, 1);
        Ok(p.finish(vec![output])?.graph)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rectangular_image_positions_keep_generated_token_delta() -> anyhow::Result<()> {
        let config = Config {
            glm: false,
            image_id: 1,
            start_id: 2,
            end_id: 3,
            selector: vec![0, 1, 2],
        };
        config.validate_markers(&[9, 2, 1, 3, 8], 1)?;
        assert!(config.validate_markers(&[9, 1, 3], 1).is_err());
        assert!(config.validate_markers(&[2, 1, 3], 2).is_err());
        let mut expansion = Expansion::default();
        expansion.text(9)?;
        expansion.text(2)?;
        assert_eq!(expansion.image(1, 2, 3, 10)?, 2);
        expansion.text(3)?;
        expansion.text(8)?;
        let (tokens, positions) = expansion.finish(10)?;
        assert_eq!(tokens, vec![9, 2, 1, 1, 1, 1, 1, 1, 3, 8]);
        assert_eq!(positions.delta, -3);
        assert_eq!(
            positions.select(&[0, 1, 2], 7, 4),
            vec![2., 3., 4., 5., 5., 5., 6., 6., 6., 7., 7., 7.]
        );
        let mut expansion = Expansion::default();
        assert!(expansion.image(1, 2, 3, 5).is_err());
        assert!(expansion.tokens.is_empty());
        Ok(())
    }
    #[test]
    fn splice_rejects_overlapping_image_features() {
        let splice = Splice {
            shape: vec![1, 8, 4],
            dtype: ScalarType::BF16,
            offset: 0,
            spans: vec![(1, 3), (3, 2)],
        };
        assert!(splice.graph().is_err());
    }
}
