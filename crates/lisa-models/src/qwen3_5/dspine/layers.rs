//! Projection, context attention and feed-forward programs. The draft's
//! projection and activation laws deliberately differ from the target's.
use crate::{
    attention_cache::Append,
    graph::{Graph, Value},
    program::Build,
};
use lisa_backend::{AttentionMask, Binary, Kernel, TensorOperation as T, Unary};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Linear {
    pub shape: Vec<i32>,
    pub output: i32,
    pub quantized: bool,
    pub bias: bool,
    pub small_rows: bool,
}
impl Linear {
    pub fn weights(&self) -> usize {
        (if self.quantized { 3 } else { 1 }) + usize::from(self.bias)
    }
    pub(super) fn append(
        &self,
        p: &mut Build<'_>,
        x: Value,
        weights: &[Value],
    ) -> anyhow::Result<Value> {
        anyhow::ensure!(
            self.shape.len() >= 2
                && self.shape.iter().all(|&d| d > 0)
                && self.output > 0
                && weights.len() == self.weights(),
            "invalid draft projection geometry"
        );
        let width = *self.shape.last().unwrap();
        let rows = self.shape[..self.shape.len() - 1]
            .iter()
            .try_fold(1i32, |n, d| n.checked_mul(*d))
            .ok_or_else(|| anyhow::anyhow!("projection row overflow"))?;
        let mut y = if self.quantized || (self.small_rows && (1..=64).contains(&rows)) {
            let x = p.reshape(x, &[rows, width]);
            let mut args = vec![x];
            args.extend_from_slice(&weights[..if self.quantized { 3 } else { 1 }]);
            let y = p.graph.one(
                if self.quantized {
                    Kernel::Affine4WideLinear
                } else {
                    Kernel::Bf16SmallRowLinear
                },
                args,
            );
            let mut shape = self.shape.clone();
            *shape.last_mut().unwrap() = self.output;
            p.reshape(y, &shape)
        } else {
            let w = p.transpose(weights[0], &[1, 0]);
            p.op(T::Matmul, vec![x, w])
        };
        if self.bias {
            y = p.binary(Binary::Add, y, *weights.last().unwrap());
        }
        Ok(y)
    }
    pub fn graph(&self) -> anyhow::Result<Graph> {
        let schema = Default::default();
        let mut p = Build::new(1 + self.weights(), &schema);
        let weights = (1..=self.weights()).map(Value::Input).collect::<Vec<_>>();
        let y = self.append(&mut p, Value::Input(0), &weights)?;
        Ok(p.finish(vec![y])?.graph)
    }
}

pub fn prepare_weight(rank: usize, native_layout: bool) -> anyhow::Result<Graph> {
    anyhow::ensure!(rank == 2 || rank == 3, "invalid projection weight rank");
    let schema = Default::default();
    let mut p = Build::new(1, &schema);
    let mut x = Value::Input(0);
    let axes = if rank == 2 { vec![1, 0] } else { vec![0, 2, 1] };
    if !native_layout {
        x = p.transpose(x, &axes);
    }
    x = p.contiguous(x);
    if !native_layout {
        x = p.transpose(x, &axes);
    }
    Ok(p.finish(vec![x])?.graph)
}

/// Absolute causal positions, including the window's leftmost live token.
pub fn window_mask(
    offset: usize,
    rows: usize,
    key_start: usize,
    keys: usize,
    window: usize,
) -> Vec<f32> {
    (0..rows)
        .flat_map(|row| {
            (0..keys).map(move |key| {
                let query = offset + row;
                let position = key_start + key;
                if position <= query && position >= (query + 1).saturating_sub(window) {
                    0.
                } else {
                    f32::NEG_INFINITY
                }
            })
        })
        .collect()
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Attention {
    pub hidden: i32,
    pub heads: i32,
    pub theta: u32,
    pub epsilon: u32,
    pub quantized: bool,
    pub small_rows: bool,
    pub cache: Append,
    pub cache_start: usize,
    pub window: Option<usize>,
    pub context_only: bool,
}
impl Attention {
    pub fn graph(&self) -> anyhow::Result<Graph> {
        let (end, _, _) = self.cache.layout().map_err(anyhow::Error::msg)?;
        let (b, s, kh, d) = (
            self.cache.batch,
            self.cache.rows,
            self.cache.heads,
            self.cache.width,
        );
        anyhow::ensure!(
            self.hidden > 0 && self.heads > 0 && self.heads % kh == 0 && self.window != Some(0),
            "invalid draft attention geometry"
        );
        let offset = self
            .cache_start
            .checked_add(self.cache.offset)
            .and_then(|n| i32::try_from(n).ok())
            .ok_or_else(|| anyhow::anyhow!("draft position overflow"))?;
        let count = if self.quantized { 3 } else { 1 };
        // Inputs: x; Q/K/V/O projection payloads; Q/K norm; existing K/V.
        let cache_index = 1 + 4 * count + 2;
        let schema = Default::default();
        let mut p = Build::new(
            cache_index
                + if self.cache.previous_batch.is_some() {
                    2
                } else {
                    0
                },
            &schema,
        );
        let project = |p: &mut Build<'_>, index: usize, output: i32, x, shape| {
            let weights = (1 + index * count..1 + (index + 1) * count)
                .map(Value::Input)
                .collect::<Vec<_>>();
            Linear {
                shape,
                output,
                quantized: self.quantized,
                bias: false,
                small_rows: self.small_rows,
            }
            .append(p, x, &weights)
        };
        let rotate = |p: &mut Build<'_>, x| {
            p.op(
                T::Rotary {
                    dimensions: d,
                    base: self.theta,
                    offset,
                },
                vec![x],
            )
        };
        let mut q = None;
        if !self.context_only {
            let x = project(
                &mut p,
                0,
                self.heads * d,
                Value::Input(0),
                vec![b, s, self.hidden],
            )?;
            let x = p.reshape(x, &[b, s, self.heads, d]);
            let x = p.op(
                T::RmsNorm {
                    epsilon: self.epsilon,
                },
                vec![x, Value::Input(1 + 4 * count)],
            );
            let x = p.transpose(x, &[0, 2, 1, 3]);
            let x = p.contiguous(x);
            q = Some(rotate(&mut p, x));
        }
        let k = project(&mut p, 1, kh * d, Value::Input(0), vec![b, s, self.hidden])?;
        let k = p.reshape(k, &[b, s, kh, d]);
        let k = p.op(
            T::RmsNorm {
                epsilon: self.epsilon,
            },
            vec![k, Value::Input(2 + 4 * count)],
        );
        let k = p.transpose(k, &[0, 2, 1, 3]);
        let k = p.contiguous(k);
        let v = project(&mut p, 2, kh * d, Value::Input(0), vec![b, s, self.hidden])?;
        let v = p.reshape(v, &[b, s, kh, d]);
        let v = p.transpose(v, &[0, 2, 1, 3]);
        let v = p.contiguous(v);
        let k = rotate(&mut p, k);
        let mut inputs = vec![k, v];
        if self.cache.previous_batch.is_some() {
            inputs.extend([Value::Input(cache_index), Value::Input(cache_index + 1)]);
        }
        let stored = p
            .graph
            .append(
                self.cache.graph().map_err(anyhow::Error::msg)?,
                &inputs,
                &[],
                &[],
            )
            .map_err(anyhow::Error::msg)?;
        if self.context_only {
            return Ok(p.finish(stored[..2].to_vec())?.graph);
        }
        let (mut keys, mut values) = (stored[2], stored[3]);
        let mut args = vec![q.unwrap()];
        let mask = if let Some(window) = self.window.filter(|&w| offset as usize + s as usize > w) {
            let start = (offset as usize).saturating_sub(window - 1);
            anyhow::ensure!(
                start >= self.cache_start,
                "draft KV window was evicted before its query"
            );
            let local = (start - self.cache_start) as i32;
            keys = p.slice(keys, 2, local, end as i32);
            keys = p.contiguous(keys);
            values = p.slice(values, 2, local, end as i32);
            values = p.contiguous(values);
            let len = end - local as usize;
            let values_mask = window_mask(offset as usize, s as usize, start, len, window);
            let mask = p.graph.constant_f32(&values_mask, &[1, 1, s, len as i32]);
            args.extend([keys, values, mask]);
            AttentionMask::Tensor
        } else {
            args.extend([keys, values]);
            AttentionMask::Causal
        };
        let y = p.op(
            T::Attention {
                scale: (d as f32).powf(-0.5).to_bits(),
                mask,
            },
            args,
        );
        let y = p.transpose(y, &[0, 2, 1, 3]);
        let y = p.contiguous(y);
        let y = p.reshape(y, &[b, s, self.heads * d]);
        let y = project(&mut p, 3, self.hidden, y, vec![b, s, self.heads * d])?;
        Ok(p.finish(vec![y, stored[0], stored[1]])?.graph)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Tail {
    pub shape: Vec<i32>,
    pub intermediate: i32,
    pub quantized: bool,
    pub small_rows: bool,
    pub epsilon: u32,
}
impl Tail {
    pub fn graph(&self) -> anyhow::Result<Graph> {
        let count = if self.quantized { 3 } else { 1 };
        let schema = Default::default();
        let mut p = Build::new(3 + 3 * count, &schema);
        let h = p.binary(Binary::Add, Value::Input(0), Value::Input(1));
        let n = p.op(
            T::RmsNorm {
                epsilon: self.epsilon,
            },
            vec![h, Value::Input(2)],
        );
        let mut lin = Linear {
            shape: self.shape.clone(),
            output: self.intermediate,
            quantized: self.quantized,
            bias: false,
            small_rows: self.small_rows,
        };
        let w = |i| {
            (3 + i * count..3 + (i + 1) * count)
                .map(Value::Input)
                .collect::<Vec<_>>()
        };
        let gate = lin.append(&mut p, n, &w(0))?;
        let up = lin.append(&mut p, n, &w(1))?;
        let mlp = p.unary(Unary::Sigmoid, gate);
        let mlp = p.binary(Binary::Multiply, mlp, gate);
        let mlp = p.binary(Binary::Multiply, mlp, up);
        lin.output = *self
            .shape
            .last()
            .ok_or_else(|| anyhow::anyhow!("missing hidden width"))?;
        *lin.shape.last_mut().unwrap() = self.intermediate;
        let mlp = lin.append(&mut p, mlp, &w(2))?;
        let y = p.binary(Binary::Add, h, mlp);
        Ok(p.finish(vec![y])?.graph)
    }
}
