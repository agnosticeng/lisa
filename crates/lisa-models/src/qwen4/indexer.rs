//! Qwen4 indexer projections, incremental block pooling and causal selection.
use crate::{
    graph::{Graph, Value as V},
    program::Build,
};
use anyhow::{Result, ensure};
use lisa_backend::{
    Binary as B, BlockAttention, Comparison, Kernel, ScalarType as D, TensorOperation as T,
    Unary as U,
};
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Spec {
    pub heads: i32,
    pub width: i32,
    pub budget: usize,
    pub ratio: usize,
    pub top_k: usize,
    pub quant: (i32, i32),
    pub epsilon: u32,
    pub key_epsilon: u32,
    pub rotary: usize,
    pub base: u32,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Update {
    pub total: usize,
    pub blocks: usize,
    pub pool: Option<std::ops::Range<usize>>,
    pub active: bool,
}
impl Spec {
    pub fn update(self, rows: usize, tape: Option<(usize, usize)>) -> Result<Update> {
        ensure!(
            rows > 0 && self.ratio > 0 && self.heads > 0 && self.width > 0 && self.top_k > 0,
            "invalid indexer geometry"
        );
        let (previous, pooled) = tape.unwrap_or((0, 0));
        let total = previous
            .checked_add(rows)
            .ok_or_else(|| anyhow::anyhow!("indexer position overflow"))?;
        ensure!(
            total <= i32::MAX as usize,
            "indexer position exceeds tensor range"
        );
        let blocks = total / self.ratio;
        ensure!(pooled <= blocks, "indexer pooled cursor exceeds tape");
        let active = total > self.budget;
        Ok(Update {
            total,
            blocks,
            active,
            pool: (active && pooled < blocks).then_some(pooled..blocks),
        })
    }
    pub fn project_captured(self, shape: &[i32]) -> Result<Graph> {
        Ok(self.project(shape)?.with_serial_affine(false))
    }
    pub fn project(self, shape: &[i32]) -> Result<Graph> {
        ensure!(
            shape.len() == 3 && shape.iter().all(|&d| d > 0),
            "invalid indexer activation"
        );
        let [b, s, _] = <[i32; 3]>::try_from(shape).unwrap();
        let schema = Default::default();
        let mut p = Build::new(4, &schema);
        let qk = p.graph.one(
            Kernel::AffineLinear {
                group_size: self.quant.0,
                bits: self.quant.1,
            },
            vec![V::Input(0), V::Input(1), V::Input(2), V::Input(3)],
        );
        let split = self.heads * self.width;
        let q = p.slice(qk, -1, 0, split);
        let q = p.reshape(q, &[b, s, self.heads, self.width]);
        let k = p.slice(qk, -1, split, split + self.width);
        let k = p.reshape(k, &[b, s, self.width]);
        Ok(p.finish(vec![q, k])?.graph)
    }
    /// Inputs are the engine's raw slice and key-normalization scale.
    pub fn pool(self, batch: i32, start: usize, end: usize, contiguous: bool) -> Result<Graph> {
        ensure!(
            end > start && end <= i32::MAX as usize / self.ratio,
            "invalid pooling interval"
        );
        let n = (end - start) as i32;
        let schema = Default::default();
        let mut p = Build::new(2, &schema);
        let mut raw = V::Input(0);
        if contiguous {
            raw = p.contiguous(raw);
        }
        let raw = p.reshape(raw, &[batch, n, self.ratio as i32, self.width]);
        let raw = p.cast(raw, D::F32);
        let raw = p.op(
            T::Mean {
                axis: 2,
                keep_dims: false,
            },
            vec![raw],
        );
        let raw = p.cast(raw, D::BF16);
        let normalized = rms(&mut p, raw, V::Input(1), self.key_epsilon)?;
        let positions = (start..end)
            .map(|n| (n * self.ratio) as i32)
            .collect::<Vec<_>>();
        let positions = p.graph.constant_i32(&positions, &[1, n]);
        let cs = frequencies(&mut p, self, positions, n)?;
        let out = rotate(&mut p, self, normalized, cs[0], cs[1])?;
        Ok(p.finish(vec![out])?.graph)
    }
    pub fn native_select(
        self,
        batch: i32,
        rows: i32,
        offset: usize,
        blocks: usize,
        available: bool,
    ) -> bool {
        available
            && batch == 1
            && self.heads == 4
            && self.width == 128
            && self.ratio == 4
            && self.top_k == 512
            && offset
                .checked_add(rows as usize)
                .is_some_and(|end| blocks == end / 4)
            && blocks <= 65536
    }

    /// Native selection already has one fixed reduction per query and never
    /// scores future blocks. The generic fallback instead retains the decode
    /// matmul shape and partition extent independently for each prefix.
    pub fn select_captured(
        self,
        batch: i32,
        rows: i32,
        offset: usize,
        blocks: usize,
        native: bool,
    ) -> Result<Graph> {
        if native || rows == 1 {
            return self.select(batch, rows, offset, blocks, native);
        }
        let end = offset
            .checked_add(rows as usize)
            .ok_or_else(|| anyhow::anyhow!("indexer offset overflow"))?;
        ensure!(
            batch > 0 && rows > 0 && self.ratio > 0 && blocks > 0 && end <= i32::MAX as usize,
            "invalid captured indexer selection span"
        );
        let schema = Default::default();
        let mut p = Build::new(3, &schema);
        let width = self.top_k.min(blocks) as i32;
        let mut outputs: [Vec<V>; 4] = Default::default();
        for row in 0..rows {
            let position = offset + row as usize;
            let prefix_blocks = blocks.min((position + 1) / self.ratio);
            let row_width = if prefix_blocks == 0 {
                width
            } else {
                self.top_k.min(prefix_blocks) as i32
            };
            let mut result = if prefix_blocks == 0 {
                vec![
                    p.graph.one(
                        Kernel::Zeros {
                            shape: vec![batch, 1, width],
                            dtype: D::U32,
                        },
                        vec![],
                    ),
                    p.graph.one(
                        Kernel::Zeros {
                            shape: vec![batch, 1, width],
                            dtype: D::Bool,
                        },
                        vec![],
                    ),
                    p.graph.constant_f32(&[0.], &[1, 1]),
                    p.graph.constant_i32(&[position as i32], &[1, 1]),
                ]
            } else {
                let q = p.slice(V::Input(0), 1, row, row + 1);
                let pooled = p.slice(V::Input(1), 1, 0, prefix_blocks as i32);
                p.graph
                    .append(
                        self.select(batch, 1, position, prefix_blocks, false)?,
                        &[q, pooled, V::Input(2)],
                        &[],
                        &[],
                    )
                    .map_err(anyhow::Error::msg)?
            };
            if row_width < width {
                let padding = vec![batch, 1, width - row_width];
                let ids = p.graph.one(
                    Kernel::Zeros {
                        shape: padding.clone(),
                        dtype: D::U32,
                    },
                    vec![],
                );
                let valid = p.graph.one(
                    Kernel::Zeros {
                        shape: padding,
                        dtype: D::Bool,
                    },
                    vec![],
                );
                result[0] = p.concat(vec![result[0], ids], 2);
                result[1] = p.concat(vec![result[1], valid], 2);
            }
            for (all, value) in outputs.iter_mut().zip(result) {
                all.push(value);
            }
        }
        let out: Vec<_> = outputs.into_iter().map(|v| p.concat(v, 1)).collect();
        Ok(p.finish(out)?.graph)
    }

    /// Inputs: query projection, pooled tape, query-normalization scale.
    /// Outputs: block IDs, valid flags, completed-block counts, query positions.
    pub fn select(
        self,
        batch: i32,
        rows: i32,
        offset: usize,
        blocks: usize,
        native: bool,
    ) -> Result<Graph> {
        let end = offset
            .checked_add(rows as usize)
            .ok_or_else(|| anyhow::anyhow!("indexer offset overflow"))?;
        ensure!(
            rows > 0 && blocks > 0 && end <= i32::MAX as usize && blocks <= i32::MAX as usize,
            "invalid indexer selection span"
        );
        let schema = Default::default();
        let mut p = Build::new(3, &schema);
        let pos = (offset..end).map(|p| p as i32).collect::<Vec<_>>();
        let qpos = p.graph.constant_i32(&pos, &[1, rows]);
        let cs = frequencies(&mut p, self, qpos, rows)?;
        let q = rms(&mut p, V::Input(0), V::Input(2), self.epsilon)?;
        let cos = p.reshape(cs[0], &[1, rows, 1, self.rotary as i32]);
        let sin = p.reshape(cs[1], &[1, rows, 1, self.rotary as i32]);
        let q = rotate(&mut p, self, q, cos, sin)?;
        let complete = p.graph.constant_f32(
            &pos.iter()
                .map(|&v| ((v + 1).max(0) / self.ratio as i32) as f32)
                .collect::<Vec<_>>(),
            &[1, rows],
        );
        let (ids, valid) = if native {
            ensure!(
                self.native_select(batch, rows, offset, blocks, true),
                "invalid native indexer geometry"
            );
            let out = p.graph.call(
                Kernel::BlockAttention(BlockAttention::Select {
                    offset: offset as i32,
                    total: end as i32,
                    blocks: blocks as i32,
                }),
                vec![q, V::Input(1)],
            );
            (
                p.reshape(out[0], &[batch, rows, 512]),
                p.reshape(out[1], &[batch, rows, 512]),
            )
        } else {
            let q = p.cast(q, D::F32);
            let pooled = p.cast(V::Input(1), D::F32);
            let q = p.reshape(q, &[batch, rows * self.heads, self.width]);
            let pooled = p.transpose(pooled, &[0, 2, 1]);
            let scores = p.op(T::Matmul, vec![q, pooled]);
            let scores = p.reshape(scores, &[batch, rows, self.heads, blocks as i32]);
            let scores = p.transpose(scores, &[0, 1, 3, 2]);
            let zero = p.scalar(0., D::F32);
            let scores = p.binary(B::Maximum, scores, zero);
            let scores = p.op(
                T::Sum {
                    axis: -1,
                    keep_dims: false,
                },
                vec![scores],
            );
            let divisor = p.scalar((self.width as f32).sqrt(), D::F32);
            let scores = p.binary(B::Divide, scores, divisor);
            let ids = p.graph.constant_i32(
                &(0..blocks as i32).collect::<Vec<_>>(),
                &[1, 1, blocks as i32],
            );
            let ids_f = p.cast(ids, D::F32);
            let visible = p.op(T::Broadcast(vec![batch, rows, blocks as i32]), vec![ids_f]);
            let bounds = p.reshape(complete, &[1, rows, 1]);
            let visible = p.op(T::Compare(Comparison::Less), vec![visible, bounds]);
            let neg = p.scalar(f32::NEG_INFINITY, D::F32);
            let neg = p.op(T::Broadcast(vec![batch, rows, blocks as i32]), vec![neg]);
            let scores = p.op(T::Where, vec![visible, scores, neg]);
            let ids_f = p.cast(ids, D::F32);
            let tie = p.scalar(1e-12, D::F32);
            let tie = p.binary(B::Multiply, ids_f, tie);
            let scores = p.binary(B::Subtract, scores, tie);
            let k = self.top_k.min(blocks) as i32;
            let negative = p.unary(U::Negate, scores);
            let top = p.op(
                T::ArgPartition {
                    kth: k - 1,
                    axis: -1,
                },
                vec![negative],
            );
            let top = p.slice(top, -1, 0, k);
            let top = p.contiguous(top);
            let valid = p.op(T::TakeAlong { axis: -1 }, vec![visible, top]);
            (top, valid)
        };
        Ok(p.finish(vec![ids, valid, complete, qpos])?.graph)
    }
}
fn frequencies(p: &mut Build<'_>, s: Spec, pos: V, rows: i32) -> Result<Vec<V>> {
    p.graph
        .append(
            crate::position::frequencies(s.rotary, f32::from_bits(s.base), &[1, rows], false)?,
            &[pos],
            &[],
            &[],
        )
        .map_err(anyhow::Error::msg)
}
fn rotate(p: &mut Build<'_>, s: Spec, x: V, cos: V, sin: V) -> Result<V> {
    Ok(p.graph
        .append(
            crate::position::partial_rotary(s.rotary as i32, s.width, D::BF16),
            &[x, cos, sin],
            &[],
            &[],
        )
        .map_err(anyhow::Error::msg)?[0])
}
fn rms(p: &mut Build<'_>, x: V, w: V, epsilon: u32) -> Result<V> {
    Ok(p.graph
        .append(
            crate::normalization::rms(f32::from_bits(epsilon), false),
            &[x, w],
            &[],
            &[],
        )
        .map_err(anyhow::Error::msg)?[0])
}
pub fn names(prefix: &str) -> (String, String, &'static str) {
    (
        format!("{prefix}.q_layernorm"),
        format!("{prefix}.k_layernorm"),
        "index_qk_proj",
    )
}
pub fn pool_without_tape(spec: Spec, batch: i32, _rows: i32, blocks: usize) -> Result<Graph> {
    let mut g = crate::graph::Builder::new(2, 0, 0);
    let raw = g.one(
        Kernel::Tensor(T::Slice {
            axis: 1,
            start: 0,
            end: (blocks * spec.ratio) as i32,
        }),
        vec![V::Input(0)],
    );
    let out = g
        .append(
            spec.pool(batch, 0, blocks, false)?,
            &[raw, V::Input(1)],
            &[],
            &[],
        )
        .map_err(anyhow::Error::msg)?;
    Ok(g.finish(out))
}

/// Fixed geometry of this checkpoint family, independently of its provider.
pub const GEOMETRY: (usize, usize, usize, usize, usize) = (4, 128, 2048, 4, 512);
#[cfg(test)]
mod tests {
    use super::*;
    fn spec() -> Spec {
        Spec {
            heads: 4,
            width: 128,
            budget: 2048,
            ratio: 4,
            top_k: 512,
            quant: (32, 4),
            epsilon: 1e-6f32.to_bits(),
            key_epsilon: 1e-6f32.to_bits(),
            rotary: 64,
            base: 10000000f32.to_bits(),
        }
    }

    #[test]
    fn captured_projection_and_fallback_preserve_decode_shapes() {
        let s = spec();
        for rows in [1, 2, 7] {
            let projection = s.project_captured(&[1, rows, 1024]).unwrap();
            projection.validate().unwrap();
            assert!(projection.calls.iter().any(|c| matches!(
                c.kernel,
                Kernel::AffineSerialLinear {
                    allow_verify_splitk: false,
                    ..
                }
            )));
            let offset = 2046;
            let blocks = (offset + rows as usize) / s.ratio;
            let selection = s.select_captured(1, rows, offset, blocks, false).unwrap();
            selection.validate().unwrap();
            let matmuls = selection
                .calls
                .iter()
                .filter(|c| c.kernel == Kernel::Tensor(T::Matmul))
                .count();
            assert_eq!(matmuls, rows as usize);
            assert!(!selection.calls.iter().any(|c| matches!(&c.kernel,
                Kernel::Tensor(T::Reshape(shape)) if shape == &vec![1, rows * s.heads, s.width]
                    && rows > 1)));
            let positions = selection
                .constants
                .iter()
                .filter(|c| c.dtype == D::I32 && c.shape == [1, 1])
                .map(|c| i32::from_ne_bytes(c.bytes[..4].try_into().unwrap()))
                .collect::<Vec<_>>();
            assert_eq!(
                positions,
                (offset as i32..offset as i32 + rows).collect::<Vec<_>>()
            );
        }
        // Before the first complete block, placeholders are invalid rather
        // than attempting a zero-column top-k operation.
        s.select_captured(1, 7, 0, 1, false)
            .unwrap()
            .validate()
            .unwrap();
    }

    #[test]
    fn captured_native_selection_keeps_independent_row_grid() {
        let graph = spec().select_captured(1, 7, 2048, 513, true).unwrap();
        graph.validate().unwrap();
        assert_eq!(
            graph
                .calls
                .iter()
                .filter(|c| matches!(
                    c.kernel,
                    Kernel::BlockAttention(BlockAttention::Select { .. })
                ))
                .count(),
            1
        );
        assert!(
            !graph
                .calls
                .iter()
                .any(|c| c.kernel == Kernel::Tensor(T::Matmul))
        );
    }

    #[test]
    fn pooling_boundary_is_lazy_and_incremental() {
        let s = Spec {
            heads: 4,
            width: 128,
            budget: 2048,
            ratio: 4,
            top_k: 512,
            quant: (32, 4),
            epsilon: 1e-6f32.to_bits(),
            key_epsilon: 1e-6f32.to_bits(),
            rotary: 64,
            base: 10000000f32.to_bits(),
        };
        assert_eq!(
            s.update(2048, Some((0, 0))).unwrap(),
            Update {
                total: 2048,
                blocks: 512,
                pool: None,
                active: false
            }
        );
        assert_eq!(s.update(1, Some((2048, 0))).unwrap().pool, Some(0..512));
        assert_eq!(s.update(1, Some((2049, 512))).unwrap().pool, None);
        assert_eq!(s.update(3, Some((2049, 512))).unwrap().pool, Some(512..513));
        assert!(s.update(1, Some((usize::MAX, 0))).is_err());
        assert!(!s.native_select(2, 1, 2048, 512, true));
        assert!(!s.native_select(1, 1, 2048, 512, false));
    }
}
