//! Original affine weights, sandwich RMS norms, paired-half RoPE and causal depthwise convolution.
use super::config::{Config, Layer, Rope};
use crate::graph::{Builder, Graph, Value};
use Value::{Input as I, Weight as W};
use lisa_backend::{AttentionMask, Binary, Kernel, ScalarType as D, TensorOperation as T, Unary};
pub struct Definition {
    pub graph: Graph,
    pub weights: Vec<(String, Vec<i32>, D)>,
}
struct Plan {
    b: Builder,
    weights: Vec<(String, Vec<i32>, D)>,
    rows: i32,
}
impl Plan {
    fn new(inputs: usize, rows: i32) -> Self {
        Self {
            b: Builder::new(inputs, 0, 0),
            weights: vec![],
            rows,
        }
    }
    fn op(&mut self, op: T, args: &[Value]) -> Value {
        self.b.one(Kernel::Tensor(op), args.to_vec())
    }
    fn weight(&mut self, name: String, shape: Vec<i32>, dtype: D) -> Value {
        let i = self.weights.len();
        self.weights.push((name, shape, dtype));
        W(i)
    }
    fn norm(&mut self, x: Value, name: &str, width: usize, eps: f32) -> Value {
        let w = self.weight(format!("{name}.weight"), vec![width as i32], D::BF16);
        self.op(
            T::RmsNorm {
                epsilon: eps.to_bits(),
            },
            &[x, w],
        )
    }
    fn linear(&mut self, x: Value, name: &str, input: usize, output: usize) -> Value {
        let x = self.op(T::Reshape(vec![self.rows, input as i32]), &[x]);
        let x = self.op(T::Contiguous, &[x]);
        let w = self.weight(
            format!("{name}.weight"),
            vec![output as i32, (input / 8) as i32],
            D::U32,
        );
        let s = self.weight(
            format!("{name}.scales"),
            vec![output as i32, (input / 64) as i32],
            D::BF16,
        );
        let b = self.weight(
            format!("{name}.biases"),
            vec![output as i32, (input / 64) as i32],
            D::BF16,
        );
        let kernel = if output % 32 == 0 || output == 16 {
            Kernel::Affine4WideLinear
        } else {
            Kernel::AffineLinear {
                group_size: 64,
                bits: 4,
            }
        };
        let y = self.b.one(kernel, vec![x, w, s, b]);
        self.op(T::Reshape(vec![1, self.rows, output as i32]), &[y])
    }
    fn binary(&mut self, op: Binary, a: Value, b: Value) -> Value {
        self.op(T::Binary(op), &[a, b])
    }
    fn slice(&mut self, x: Value, axis: i32, start: i32, end: i32) -> Value {
        self.op(T::Slice { axis, start, end }, &[x])
    }
    fn cat(&mut self, a: Value, b: Value, axis: i32) -> Value {
        self.op(T::Concatenate { axis, inputs: 2 }, &[a, b])
    }
    fn finish(self, outputs: Vec<Value>) -> Definition {
        let mut graph = self.b.finish(outputs);
        graph.weights = self.weights.len();
        Definition {
            graph,
            weights: self.weights,
        }
    }
    fn rotary(&mut self, x: Value, r: &Rope, d: usize, heads: usize) -> Value {
        let count = (d as f32 * r.partial_rotary_factor) as usize / 2;
        // Build frequencies on the device using the reference's F32 power operation.
        let exps = (0..d / 2)
            .map(|j| -(j as f32) * 2.0 / d as f32)
            .collect::<Vec<_>>();
        let exps = self.b.constant_f32(&exps, &[(d / 2) as i32]);
        let base = self
            .b
            .constant_f32(&vec![r.rope_theta; d / 2], &[(d / 2) as i32]);
        let f = self.binary(Binary::Power, base, exps);
        let mask = (0..d / 2)
            .map(|j| if j < count { 1.0 } else { 0.0 })
            .collect::<Vec<_>>();
        let mask = self.b.constant_f32(&mask, &[(d / 2) as i32]);
        let f = self.binary(Binary::Multiply, f, mask);
        let f = self.op(T::Broadcast(vec![self.rows, (d / 2) as i32]), &[f]);
        let pos = self.op(T::Broadcast(vec![self.rows, (d / 2) as i32]), &[I(3)]);
        let a = self.binary(Binary::Multiply, pos, f);
        let a = self.cat(a, a, -1);
        let c = self.op(T::Unary(Unary::Cos), &[a]);
        let c = self.op(T::Cast(D::BF16), &[c]);
        let s = self.op(T::Unary(Unary::Sin), &[a]);
        let s = self.op(T::Cast(D::BF16), &[s]);
        let c = self.op(
            T::Broadcast(vec![1, heads as i32, self.rows, d as i32]),
            &[c],
        );
        let s = self.op(
            T::Broadcast(vec![1, heads as i32, self.rows, d as i32]),
            &[s],
        );
        let left = self.slice(x, -1, 0, (d / 2) as i32);
        let right = self.slice(x, -1, (d / 2) as i32, d as i32);
        let neg = self.op(T::Unary(Unary::Negate), &[right]);
        let rotated = self.cat(neg, left, -1);
        let a = self.binary(Binary::Multiply, x, c);
        let b = self.binary(Binary::Multiply, rotated, s);
        self.binary(Binary::Add, a, b)
    }
}
/// Inputs: hidden, convolution history or keys, values, absolute positions [S,1], attention mask.
/// Outputs: hidden and immutable replacement state (one convolution tensor or K,V).
pub fn layer(c: &Config, kind: Layer, rows: usize, history: usize, cursor: usize) -> Definition {
    let h = c.hidden_size;
    let s = rows as i32;
    let d = c.head_dim;
    let nh = c.num_attention_heads;
    let nk = c.num_key_value_heads;
    let mut p = Plan::new(5, s);
    let x = p.norm(I(0), "operator_norm", h, c.norm_eps);
    let (mixed, states) = if kind == Layer::Conv {
        let bcx = p.linear(x, "conv.in_proj", h, 3 * h);
        let b = p.slice(bcx, -1, 0, h as i32);
        let gate = p.slice(bcx, -1, h as i32, 2 * h as i32);
        let xx = p.slice(bcx, -1, 2 * h as i32, 3 * h as i32);
        let bx = p.binary(Binary::Multiply, b, xx);
        let full = p.cat(I(1), bx, 1);
        let state = p.slice(full, 1, s, s + c.conv_length as i32 - 1);
        let state = p.op(T::Contiguous, &[state]);
        let w = p.weight(
            "conv.conv.weight".into(),
            vec![h as i32, c.conv_length as i32, 1],
            D::BF16,
        );
        let w = p.op(T::Cast(D::F32), &[w]);
        let xf = p.op(T::Cast(D::F32), &[full]);
        let mut acc = p.b.one(
            Kernel::Zeros {
                shape: vec![1, s, h as i32],
                dtype: D::F32,
            },
            vec![],
        );
        // MLX depthwise_conv_1d accumulates taps in F32, left to right. BF16 products are exact in F32.
        for tap in 0..c.conv_length as i32 {
            let a = p.slice(xf, 1, tap, tap + s);
            let wt = p.slice(w, 1, tap, tap + 1);
            let wt = p.op(T::Contiguous, &[wt]);
            let wt = p.op(T::Reshape(vec![1, 1, h as i32]), &[wt]);
            let wt = p.op(T::Broadcast(vec![1, s, h as i32]), &[wt]);
            let product = p.binary(Binary::Multiply, a, wt);
            acc = p.binary(Binary::Add, acc, product);
        }
        let conv = p.op(T::Cast(D::BF16), &[acc]);
        let conv = p.binary(Binary::Multiply, gate, conv);
        (p.linear(conv, "conv.out_proj", h, h), vec![state])
    } else {
        let q = p.linear(x, "self_attn.q_proj", h, nh * d);
        let q = p.op(T::Reshape(vec![1, s, nh as i32, d as i32]), &[q]);
        let q = p.norm(q, "self_attn.q_norm", d, c.norm_eps);
        let q = p.op(T::Transpose(vec![0, 2, 1, 3]), &[q]);
        let q = p.rotary(q, c.rope(kind), d, nh);
        let k = p.linear(x, "self_attn.k_proj", h, nk * d);
        let k = p.op(T::Reshape(vec![1, s, nk as i32, d as i32]), &[k]);
        let k = p.norm(k, "self_attn.k_norm", d, c.norm_eps);
        let k = p.op(T::Transpose(vec![0, 2, 1, 3]), &[k]);
        let k = p.rotary(k, c.rope(kind), d, nk);
        let v = p.linear(x, "self_attn.v_proj", h, nk * d);
        let v = p.op(T::Reshape(vec![1, s, nk as i32, d as i32]), &[v]);
        let v = p.op(T::Transpose(vec![0, 2, 1, 3]), &[v]);
        let update = |p: &mut Plan, old: Value, new: Value| {
            if history == 0 {
                return new;
            }
            if kind != Layer::SlidingAttention {
                return p.cat(old, new, 2);
            }
            if rows == 1 && history >= c.sliding_window {
                let old = p.slice(old, 2, (history - c.sliding_window) as i32, history as i32);
                let index = if history > c.sliding_window || cursor == c.sliding_window {
                    0
                } else {
                    cursor
                };
                return p.op(
                    T::AssignSlice {
                        axis: 2,
                        start: index as i32,
                        end: index as i32 + 1,
                    },
                    &[old, new],
                );
            }
            let old = if rows > 1 && cursor < history {
                let a = p.slice(old, 2, cursor as i32, history as i32);
                let b = p.slice(old, 2, 0, cursor as i32);
                p.cat(a, b, 2)
            } else {
                old
            };
            let trim = history.saturating_sub(c.sliding_window - 1);
            let old = p.slice(old, 2, trim as i32, history as i32);
            p.cat(old, new, 2)
        };
        let k = update(&mut p, I(1), k);
        let v = update(&mut p, I(2), v);
        let visible_history = if kind == Layer::SlidingAttention {
            history.min(c.sliding_window - 1)
        } else {
            history
        };
        let mask = if rows == 1 {
            AttentionMask::None
        } else if kind == Layer::FullAttention || visible_history + rows <= c.sliding_window {
            AttentionMask::Causal
        } else {
            AttentionMask::Tensor
        };
        let args = if mask == AttentionMask::Tensor {
            let m = p.op(T::Cast(D::BF16), &[I(4)]);
            vec![q, k, v, m]
        } else {
            vec![q, k, v]
        };
        let out = p.op(
            T::Attention {
                scale: (1.0 / (d as f32).sqrt()).to_bits(),
                mask,
            },
            &args,
        );
        let out = p.op(T::Transpose(vec![0, 2, 1, 3]), &[out]);
        let gate = p.linear(x, "self_attn.gate_proj", h, nh);
        let gate = p.op(T::Unary(Unary::Sigmoid), &[gate]);
        let gate = p.op(T::Reshape(vec![1, s, nh as i32, 1]), &[gate]);
        let gate = p.op(T::Broadcast(vec![1, s, nh as i32, d as i32]), &[gate]);
        let out = p.binary(Binary::Multiply, out, gate);
        let out = p.op(T::Contiguous, &[out]);
        let mixed = p.linear(out, "self_attn.o_proj", h, h);
        let k = p.op(T::Contiguous, &[k]);
        let v = p.op(T::Contiguous, &[v]);
        (mixed, vec![k, v])
    };
    let mixed = p.norm(mixed, "post_operator_norm", h, c.norm_eps);
    let residual = p.binary(Binary::Add, I(0), mixed);
    let x = p.norm(residual, "ffn_norm", h, c.norm_eps);
    let gate = p.linear(x, "feed_forward.w1", h, c.intermediate_size);
    let up = p.linear(x, "feed_forward.w3", h, c.intermediate_size);
    let gate = p.op(T::Unary(Unary::Silu), &[gate]);
    let product = p.binary(Binary::Multiply, gate, up);
    let y = p.linear(product, "feed_forward.w2", c.intermediate_size, h);
    let y = p.norm(y, "post_ffn_norm", h, c.norm_eps);
    let y = p.binary(Binary::Add, residual, y);
    let mut outputs = vec![y];
    outputs.extend(states);
    p.finish(outputs)
}
pub fn head(c: &Config, rows: usize) -> Definition {
    let mut p = Plan::new(1, rows as i32);
    let x = p.linear(I(0), "model.embed_tokens", c.hidden_size, c.vocab_size);
    p.finish(vec![x])
}

/// Physical ring order is part of the reference reduction law, including decode after chunked prefill.
pub fn next_cursor(
    kind: Layer,
    window: usize,
    history: usize,
    cursor: usize,
    rows: usize,
) -> usize {
    if kind != Layer::SlidingAttention {
        return history + rows;
    }
    if rows > 1 {
        return history.min(window - 1) + rows;
    }
    if history < window {
        return history + 1;
    }
    if history > window || cursor == window {
        1
    } else {
        cursor + 1
    }
}
pub fn attention_mask(kind: Layer, window: usize, history: usize, rows: usize) -> Vec<f32> {
    let history = if kind == Layer::SlidingAttention {
        history.min(window - 1)
    } else {
        history
    };
    (0..rows)
        .flat_map(|q| {
            (0..history + rows).map(move |k| {
                if k <= history + q && (kind != Layer::SlidingAttention || k + window > history + q)
                {
                    0.0
                } else {
                    f32::NEG_INFINITY
                }
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn plans_preserve_immutable_state_and_window_boundaries() {
        let c = Config::parse(include_bytes!("fixtures/config.json")).unwrap();
        for kind in [Layer::Conv, Layer::FullAttention, Layer::SlidingAttention] {
            for (rows, history, cursor) in [
                (1, 0, 0),
                (8, 248, 248),
                (8, 256, 1),
                (1, 263, 263),
                (1, 256, 256),
                (1, 256, 7),
            ] {
                let def = layer(&c, kind, rows, history, cursor);
                def.graph.validate().unwrap();
                assert!(def.graph.mutable_inputs.is_empty());
                assert!(def.graph.writes.is_empty());
            }
        }
        let mask = attention_mask(Layer::SlidingAttention, 256, 263, 8);
        assert_eq!(mask.len(), 8 * 263);
        for q in 0..8 {
            for k in 0..263 {
                assert_eq!(mask[q * 263 + k].is_finite(), k >= q && k <= 255 + q);
            }
        }
        assert_eq!(next_cursor(Layer::SlidingAttention, 256, 263, 263, 1), 1);
        assert_eq!(next_cursor(Layer::SlidingAttention, 256, 256, 255, 1), 256);
        assert_eq!(next_cursor(Layer::SlidingAttention, 256, 256, 256, 1), 1);
        assert_eq!(next_cursor(Layer::SlidingAttention, 256, 256, 7, 8), 263);
    }
    #[test]
    fn incremental_chat_has_one_bos_and_no_qwen_thinking_whitespace() {
        assert_eq!(
            super::super::prompt::turn(Some(" Be helpful. "), " Hi ", true, false),
            "<|startoftext|><|im_start|>system\nBe helpful.<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think></think>"
        );
        assert_eq!(
            super::super::prompt::turn(Some("ignored"), "Again", false, false),
            "<|im_end|>\n<|im_start|>user\nAgain<|im_end|>\n<|im_start|>assistant\n<think></think>"
        );
    }
}
