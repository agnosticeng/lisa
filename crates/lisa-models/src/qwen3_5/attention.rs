//! Dense hybrid attention's projections, rotary law, cache writes and read schedule.
use crate::{
    attention_cache::Append,
    graph::{Builder, Graph, Value},
};
use lisa_backend::{AttentionMask, Kernel, ScalarType, TensorOperation as T};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Plan {
    pub batch: i32,
    pub rows: i32,
    pub heads: i32,
    pub key_heads: i32,
    pub width: i32,
    pub rotary: i32,
    pub frequencies_bf16: bool,
    pub epsilon: u32,
    pub scale: u32,
    pub gated: bool,
    pub quant: [(i32, i32); 4],
    pub joined: bool,
    pub retiled: bool,
    pub cache: Option<Append>,
    pub mask_length: Option<i32>,
    pub next_positions: Option<Vec<usize>>,
}
fn op(g: &mut Builder, op: T, x: Value) -> Value {
    g.one(Kernel::Tensor(op), vec![x])
}
fn reshape(g: &mut Builder, x: Value, shape: &[i32]) -> Value {
    op(g, T::Reshape(shape.to_vec()), x)
}
fn slice(g: &mut Builder, x: Value, axis: i32, start: i32, end: i32) -> Value {
    op(g, T::Slice { axis, start, end }, x)
}
fn transpose(g: &mut Builder, x: Value) -> Value {
    op(g, T::Transpose(vec![0, 2, 1, 3]), x)
}
fn project(g: &mut Builder, x: Value, slot: usize, quant: (i32, i32)) -> Value {
    g.one(
        Kernel::AffineLinear {
            group_size: quant.0,
            bits: quant.1,
        },
        vec![
            x,
            Value::Weight(slot),
            Value::Weight(slot + 1),
            Value::Weight(slot + 2),
        ],
    )
}
impl Plan {
    /// Weights: Q,K,V,O triples; Q/K norm; optional joined triple; optional
    /// retiled triple. Inputs: activation, cosine, sine, optional old K/V, mask.
    /// Outputs: activation, followed by cache storage K/V when requested.
    pub fn graph(&self) -> Result<Graph, String> {
        let (b, s, h, kh, d) = (
            self.batch,
            self.rows,
            self.heads,
            self.key_heads,
            self.width,
        );
        if [b, s, h, kh, d, self.rotary].iter().any(|&v| v <= 0)
            || h % kh != 0
            || self.rotary > d
            || self.rotary % 2 != 0
        {
            return Err("invalid dense attention geometry".into());
        }
        let has_old = self
            .cache
            .as_ref()
            .is_some_and(|c| c.previous_batch.is_some());
        if let Some(c) = &self.cache {
            if (c.batch, c.rows, c.heads, c.width, c.dtype) != (b, s, kh, d, ScalarType::BF16) {
                return Err("attention cache geometry differs".into());
            }
            c.layout()?;
        }
        let mask_slot = 3 + if has_old { 2 } else { 0 };
        let mut g = Builder::new(
            mask_slot + usize::from(self.mask_length.is_some()),
            14 + 3 * usize::from(self.joined) + 3 * usize::from(self.retiled),
            0,
        );
        let x = Value::Input(0);
        let cos = Value::Input(1);
        let sin = Value::Input(2);
        let merged = self.gated && self.retiled && s == 1;
        let joined = if !merged && b == 1 && s <= 8 && self.joined {
            Some(project(&mut g, x, 14, self.quant[0]))
        } else {
            None
        };
        let qn = h * 2 * d;
        let kn = kh * d;
        let mut q = None;
        let mut gate = None;
        let mut raw = None;
        if merged {
            let y = project(&mut g, x, 14 + 3 * usize::from(self.joined), self.quant[0]);
            let y = reshape(&mut g, y, &[b, 2 * h * d]);
            let z = slice(&mut g, y, 1, 0, h * d);
            q = Some(reshape(&mut g, z, &[b, s, h, d]));
            let z = slice(&mut g, y, 1, h * d, 2 * h * d);
            gate = Some(reshape(&mut g, z, &[b, s, h, d]));
        } else {
            let y = match joined {
                Some(v) => slice(&mut g, v, 2, 0, qn),
                None => project(&mut g, x, 0, self.quant[0]),
            };
            let y = op(&mut g, T::SplitLastAxis(vec![h, 2, d]), y);
            raw = Some(y);
            if s == 1 {
                let z = op(&mut g, T::Select { axis: 3, index: 0 }, y);
                q = Some(op(&mut g, T::Contiguous, z));
                if self.gated {
                    let z = op(&mut g, T::Select { axis: 3, index: 1 }, y);
                    gate = Some(op(&mut g, T::Contiguous, z));
                }
            }
        }
        let k = match joined {
            Some(v) => slice(&mut g, v, 2, qn, qn + kn),
            None => project(&mut g, x, 3, self.quant[1]),
        };
        let k = op(&mut g, T::SplitLastAxis(vec![kh, d]), k);
        let v = match joined {
            Some(v) => slice(&mut g, v, 2, qn + kn, qn + 2 * kn),
            None => project(&mut g, x, 6, self.quant[2]),
        };
        let v = op(&mut g, T::SplitLastAxis(vec![kh, d]), v);
        let mask = self
            .mask_length
            .map(|len| reshape(&mut g, Value::Input(mask_slot), &[b, 1, 1, s, len]));
        let fused = b == 1 && d == 256 && self.rotary == 64 && self.frequencies_bf16;
        let fused_rows =
            fused && (2..=8).contains(&s) && self.gated && mask.is_none() && raw.is_some();
        let (q, k) = if fused && s == 1 || fused_rows {
            let (query, key) = if fused_rows {
                if let Some(all) = joined {
                    let all = reshape(&mut g, all, &[s, qn + 2 * kn]);
                    (
                        slice(&mut g, all, 1, 0, qn),
                        slice(&mut g, all, 1, qn, qn + kn),
                    )
                } else {
                    (
                        reshape(&mut g, raw.unwrap(), &[s, qn]),
                        reshape(&mut g, k, &[s, kn]),
                    )
                }
            } else {
                (q.unwrap(), k)
            };
            let out = g.call(
                Kernel::QkNormRotary {
                    query_heads: h,
                    key_heads: kh,
                    rows: s,
                    epsilon: self.epsilon,
                    interleaved: fused_rows,
                },
                vec![query, key, Value::Weight(12), Value::Weight(13), cos, sin],
            );
            if fused_rows {
                gate = Some(out[2]);
            }
            (out[0], out[1])
        } else {
            if q.is_none() {
                let y = reshape(&mut g, raw.unwrap(), &[b, s, h, 2, d]);
                let z = op(&mut g, T::Select { axis: 3, index: 0 }, y);
                q = Some(op(&mut g, T::Contiguous, z));
                if self.gated {
                    let z = op(&mut g, T::Select { axis: 3, index: 1 }, y);
                    gate = Some(op(&mut g, T::Contiguous, z));
                }
            }
            let q = g.one(
                Kernel::Tensor(T::RmsNorm {
                    epsilon: self.epsilon,
                }),
                vec![q.unwrap(), Value::Weight(12)],
            );
            let k = g.one(
                Kernel::Tensor(T::RmsNorm {
                    epsilon: self.epsilon,
                }),
                vec![k, Value::Weight(13)],
            );
            let q = transpose(&mut g, q);
            let q = g.append(
                crate::position::partial_rotary(self.rotary, d, ScalarType::BF16),
                &[q, cos, sin],
                &[],
                &[],
            )?[0];
            let k = transpose(&mut g, k);
            let k = g.append(
                crate::position::partial_rotary(self.rotary, d, ScalarType::BF16),
                &[k, cos, sin],
                &[],
                &[],
            )?[0];
            (q, k)
        };
        let v = transpose(&mut g, v);
        let mut storage = vec![];
        let (k, v) = if let Some(c) = &self.cache {
            let mut inputs = vec![k, v];
            if has_old {
                inputs.extend([Value::Input(3), Value::Input(4)]);
            }
            let out = g.append(c.graph()?, &inputs, &[], &[])?;
            storage.extend([out[0], out[1]]);
            (out[2], out[3])
        } else {
            (k, v)
        };
        let end = self
            .cache
            .as_ref()
            .map(|c| c.offset + s as usize)
            .unwrap_or(s as usize);
        let slots = self.next_positions.as_ref().filter(|p| {
            b > 1
                && s == 1
                && mask.is_some()
                && p.len() == b as usize
                && p.iter().all(|&l| l <= end)
                && p.iter().any(|&l| l > 0)
        });
        let out = if let Some(slots) = slots {
            let mut outs = Vec::new();
            for (i, &len) in slots.iter().enumerate() {
                let qi = op(
                    &mut g,
                    T::Select {
                        axis: 0,
                        index: i as i32,
                    },
                    q,
                );
                let qi = reshape(&mut g, qi, &[1, h, s, d]);
                let ki = op(
                    &mut g,
                    T::Select {
                        axis: 0,
                        index: i as i32,
                    },
                    k,
                );
                let ki = slice(&mut g, ki, 1, (end - len) as i32, end as i32);
                let ki = reshape(&mut g, ki, &[1, kh, len as i32, d]);
                let vi = op(
                    &mut g,
                    T::Select {
                        axis: 0,
                        index: i as i32,
                    },
                    v,
                );
                let vi = slice(&mut g, vi, 1, (end - len) as i32, end as i32);
                let vi = reshape(&mut g, vi, &[1, kh, len as i32, d]);
                outs.push(g.one(
                    Kernel::Tensor(T::Attention {
                        scale: self.scale,
                        mask: AttentionMask::Causal,
                    }),
                    vec![qi, ki, vi],
                ));
            }
            g.one(
                Kernel::Tensor(T::Concatenate {
                    axis: 0,
                    inputs: outs.len(),
                }),
                outs,
            )
        } else {
            let mut inputs = vec![q, k, v];
            if let Some(mask) = mask {
                inputs.push(mask);
            }
            g.one(
                Kernel::Tensor(T::Attention {
                    scale: self.scale,
                    mask: if mask.is_some() {
                        AttentionMask::Tensor
                    } else {
                        AttentionMask::Causal
                    },
                }),
                inputs,
            )
        };
        if fused_rows {
            gate = Some(reshape(&mut g, gate.unwrap(), &[b, s, h, d]));
        }
        let out = if let Some(gate) = gate {
            g.one(
                Kernel::SigmoidGateTail {
                    batch: b,
                    rows: s,
                    heads: h,
                    width: d,
                },
                vec![gate, out],
            )
        } else {
            let out = transpose(&mut g, out);
            reshape(&mut g, out, &[b, s, h * d])
        };
        let out = op(&mut g, T::Cast(ScalarType::BF16), out);
        let out = project(&mut g, out, 9, self.quant[3]);
        let mut outputs = vec![out];
        outputs.extend(storage);
        Ok(g.finish(outputs))
    }
}

/// Binding-time projection storage. Individual Q/K/V views share the joined
/// payload, preserving the original quantized rows without a resident duplicate.
pub fn join_projection(
    widths: [i32; 3],
    inputs: [i32; 3],
    quant: [(i32, i32); 3],
) -> Option<Graph> {
    if widths != [12288, 1024, 1024]
        || inputs != [5120; 3]
        || !matches!(quant[0], (64, 2 | 4))
        || quant.iter().any(|&q| q != quant[0])
    {
        return None;
    }
    let mut g = Builder::new(9, 0, 0);
    let mut joined = Vec::new();
    for component in 0..3 {
        joined.push(g.one(
            Kernel::Tensor(T::Concatenate { axis: 0, inputs: 3 }),
            vec![
                Value::Input(component),
                Value::Input(component + 3),
                Value::Input(component + 6),
            ],
        ));
    }
    let mut outputs = joined.clone();
    let mut start = 0;
    for width in widths {
        for &value in &joined {
            outputs.push(slice(&mut g, value, 0, start, start + width));
        }
        start += width;
    }
    Some(g.finish(outputs))
}
pub fn retile_projection(heads: i32, width: i32) -> Graph {
    let mut g = Builder::new(3, 0, 0);
    let indices = (0..2)
        .flat_map(|part| {
            (0..heads).flat_map(move |h| (0..width).map(move |j| h * 2 * width + part * width + j))
        })
        .collect::<Vec<_>>();
    let indices = g.constant_i32(&indices, &[heads * 2 * width]);
    let outputs = (0..3)
        .map(|i| {
            g.one(
                Kernel::Tensor(T::Take { axis: 0 }),
                vec![Value::Input(i), indices],
            )
        })
        .collect::<Vec<_>>();
    g.finish(outputs)
}
