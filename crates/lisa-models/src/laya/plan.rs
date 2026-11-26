//! ModernBERT and decision-head dataflow. Weight names and arithmetic order
//! belong to the model; the engine binds/imports them and executes the graph.
use super::config::LayaConfig;
use crate::graph::{Builder, Graph, Value};
use lisa_backend::{Binary as B, Kernel, TensorOperation as T, Unary as U};

pub struct Plan {
    pub graph: Graph,
    pub weights: Vec<String>,
}
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Arithmetic {
    Native,
    HostF32,
}
struct Build {
    b: Builder,
    weights: Vec<String>,
    arithmetic: Arithmetic,
}
impl Build {
    fn new(inputs: usize, arithmetic: Arithmetic) -> Self {
        Self {
            b: Builder::new(inputs, 0, 0),
            weights: vec![],
            arithmetic,
        }
    }
    fn weight(&mut self, name: impl Into<String>) -> Value {
        let name = name.into();
        if let Some(i) = self.weights.iter().position(|w| w == &name) {
            return Value::Weight(i);
        }
        let i = self.weights.len();
        self.weights.push(name);
        Value::Weight(i)
    }
    fn op(&mut self, op: T, args: impl Into<Vec<Value>>) -> Value {
        self.b.one(
            if self.arithmetic == Arithmetic::Native {
                Kernel::Tensor(op)
            } else {
                Kernel::HostTensor(op)
            },
            args,
        )
    }
    fn binary(&mut self, op: B, a: Value, b: Value) -> Value {
        self.op(T::Binary(op), vec![a, b])
    }
    fn unary(&mut self, op: U, x: Value) -> Value {
        self.op(T::Unary(op), vec![x])
    }
    fn slice(&mut self, x: Value, axis: i32, start: i32, end: i32) -> Value {
        self.op(T::Slice { axis, start, end }, vec![x])
    }
    fn reshape(&mut self, x: Value, shape: &[i32]) -> Value {
        self.op(T::Reshape(shape.to_vec()), vec![x])
    }
    fn transpose(&mut self, x: Value, axes: &[i32]) -> Value {
        self.op(T::Transpose(axes.to_vec()), vec![x])
    }
    fn linear(&mut self, x: Value, w: &str, b: Option<&str>) -> Value {
        if self.arithmetic == Arithmetic::HostF32 {
            let mut args = vec![x, self.weight(w)];
            if let Some(b) = b {
                args.push(self.weight(b));
            }
            return self.op(T::SequentialLinear { bias: b.is_some() }, args);
        }
        let w = self.weight(w);
        let wt = self.transpose(w, &[1, 0]);
        let y = self.op(T::Matmul, vec![x, wt]);
        if let Some(b) = b {
            let b = self.weight(b);
            self.binary(B::Add, y, b)
        } else {
            y
        }
    }
    fn norm(&mut self, x: Value, w: &str, b: Option<&str>, eps: f32) -> Value {
        if self.arithmetic == Arithmetic::HostF32 {
            let mut args = vec![x, self.weight(w)];
            if let Some(b) = b {
                args.push(self.weight(b));
            }
            return self.op(
                T::SequentialLayerNorm {
                    epsilon: eps.to_bits(),
                    bias: b.is_some(),
                },
                args,
            );
        }
        let mean = self.op(
            T::Mean {
                axis: -1,
                keep_dims: true,
            },
            vec![x],
        );
        let xc = self.binary(B::Subtract, x, mean);
        let square = self.binary(B::Multiply, xc, xc);
        let variance = self.op(
            T::Mean {
                axis: -1,
                keep_dims: true,
            },
            vec![square],
        );
        let epsilon = self.b.scalar(eps);
        let biased = self.binary(B::Add, variance, epsilon);
        let inv = self.unary(U::Rsqrt, biased);
        let y = self.binary(B::Multiply, xc, inv);
        let weight = self.weight(w);
        let y = self.binary(B::Multiply, y, weight);
        if let Some(b) = b {
            let bias = self.weight(b);
            self.binary(B::Add, y, bias)
        } else {
            y
        }
    }
    fn gelu(&mut self, x: Value) -> Value {
        let c = self.b.scalar(std::f32::consts::FRAC_1_SQRT_2);
        let scaled = self.binary(B::Multiply, x, c);
        let erf = self.unary(U::Erf, scaled);
        let c = self.b.scalar(0.5);
        let half = self.binary(B::Multiply, x, c);
        let one = self.b.scalar(1.0);
        let plus = self.binary(B::Add, erf, one);
        self.binary(B::Multiply, half, plus)
    }
    fn rope(&mut self, x: Value, s: i32, d: i32, base: f32) -> Value {
        let half = d / 2;
        let mut cos = Vec::new();
        let mut sin = Vec::new();
        for pos in 0..s {
            for i in 0..half {
                let inv = base.powf(-2.0 * i as f32 / d as f32);
                let a = pos as f32 * inv;
                cos.push(a.cos());
                sin.push(a.sin());
            }
        }
        let cos = self.b.constant_f32(&cos, &[1, s, half]);
        let sin = self.b.constant_f32(&sin, &[1, s, half]);
        let a = self.slice(x, -1, 0, half);
        let b = self.slice(x, -1, half, d);
        let ac = self.binary(B::Multiply, a, cos);
        let bs = self.binary(B::Multiply, b, sin);
        let bc = self.binary(B::Multiply, b, cos);
        let as_ = self.binary(B::Multiply, a, sin);
        let left = self.binary(B::Subtract, ac, bs);
        let right = self.binary(B::Add, bc, as_);
        self.op(
            T::Concatenate {
                axis: -1,
                inputs: 2,
            },
            vec![left, right],
        )
    }
    fn heads(&mut self, qkv: Value, s: i32, h: i32, d: i32, part: i32) -> Value {
        let a = self.slice(qkv, 1, part, part + 1);
        let a = self.reshape(a, &[s, h, d]);
        let a = self.transpose(a, &[1, 0, 2]);
        self.op(T::Contiguous, vec![a])
    }
    fn attention(
        &mut self,
        qkv: Value,
        s: i32,
        h: i32,
        d: i32,
        mask: Value,
        base: Option<f32>,
    ) -> Value {
        let qkv = self.reshape(qkv, &[s, 3, h, d]);
        let q = self.heads(qkv, s, h, d, 0);
        let k = self.heads(qkv, s, h, d, 1);
        let v = self.heads(qkv, s, h, d, 2);
        let (q, k) = if let Some(base) = base {
            (self.rope(q, s, d, base), self.rope(k, s, d, base))
        } else {
            (q, k)
        };
        if self.arithmetic == Arithmetic::HostF32 {
            let out = self.op(T::SequentialAttention, vec![q, k, v, mask]);
            let out = self.transpose(out, &[1, 0, 2]);
            return self.reshape(out, &[s, h * d]);
        }
        let kt = self.transpose(k, &[0, 2, 1]);
        let scores = self.op(T::Matmul, vec![q, kt]);
        let scale = self.b.scalar(1.0 / (d as f32).sqrt());
        let scores = self.binary(B::Multiply, scores, scale);
        let scores = self.binary(B::Add, scores, mask);
        let probs = self.op(T::Softmax { axis: -1 }, vec![scores]);
        let out = self.op(T::Matmul, vec![probs, v]);
        let out = self.transpose(out, &[1, 0, 2]);
        self.reshape(out, &[s, h * d])
    }
    fn mask(&mut self, s: i32, global: bool, window: usize) -> Value {
        let mut data = vec![0.0; s as usize * s as usize];
        for q in 0..s {
            for k in 0..s {
                if !global && (q - k).unsigned_abs() as usize > window / 2 {
                    data[(q * s + k) as usize] = -1e9;
                }
            }
        }
        self.b.constant_f32(&data, &[s, s])
    }
    fn finish(self, outputs: Vec<Value>) -> Plan {
        let mut graph = self.b.finish(outputs);
        graph.weights = self.weights.len();
        Plan {
            graph,
            weights: self.weights,
        }
    }
}

/// Input: token IDs. Output: final normalized encoder states `[S,H]`.
pub fn encoder(cfg: &LayaConfig, sequence: i32) -> Plan {
    encoder_with(cfg, sequence, Arithmetic::Native)
}
pub fn encoder_with(cfg: &LayaConfig, sequence: i32, arithmetic: Arithmetic) -> Plan {
    let mut p = Build::new(1, arithmetic);
    let h = cfg.hidden_size as i32;
    let emb = p.weight("encoder.embeddings.tok_embeddings.weight");
    let x = p.op(T::Take { axis: 0 }, vec![emb, Value::Input(0)]);
    let mut x = p.norm(x, "encoder.embeddings.norm.weight", None, cfg.norm_eps);
    let full = p.mask(sequence, true, cfg.local_attention);
    let local = p.mask(sequence, false, cfg.local_attention);
    for i in 0..cfg.num_layers {
        let prefix = format!("encoder.layers.{i}");
        let global = cfg.layer_global[i];
        let base = if global {
            cfg.rope_theta_global
        } else {
            cfg.rope_theta_local
        };
        let normed = if i == 0 {
            x
        } else {
            p.norm(x, &format!("{prefix}.attn_norm.weight"), None, cfg.norm_eps)
        };
        let qkv = p.linear(normed, &format!("{prefix}.attn.Wqkv.weight"), None);
        let att = p.attention(
            qkv,
            sequence,
            cfg.num_heads as i32,
            cfg.head_dim as i32,
            if global { full } else { local },
            Some(base),
        );
        let att = p.linear(att, &format!("{prefix}.attn.Wo.weight"), None);
        x = p.binary(B::Add, x, att);
        let normed = p.norm(x, &format!("{prefix}.mlp_norm.weight"), None, cfg.norm_eps);
        let wi = p.linear(normed, &format!("{prefix}.mlp.Wi.weight"), None);
        let inter = cfg.intermediate_size as i32;
        let value = p.slice(wi, -1, 0, inter);
        let gate = p.slice(wi, -1, inter, 2 * inter);
        let value = p.gelu(value);
        let gated = p.binary(B::Multiply, value, gate);
        let mlp = p.linear(gated, &format!("{prefix}.mlp.Wo.weight"), None);
        x = p.binary(B::Add, x, mlp);
    }
    let out = p.norm(x, "encoder.final_norm.weight", None, cfg.norm_eps);
    let out = p.reshape(out, &[sequence, h]);
    p.finish(vec![out])
}

/// Inputs: encoder states, question-type index `[1]`, marker positions `[K]`.
/// Outputs: marker logits `[K]`, probabilities `[K]`, first decision hidden row.
pub fn scorer(cfg: &LayaConfig, sequence: i32, markers: i32) -> Plan {
    scorer_with(cfg, sequence, markers, Arithmetic::Native)
}
pub fn scorer_with(cfg: &LayaConfig, sequence: i32, markers: i32, arithmetic: Arithmetic) -> Plan {
    let mut p = Build::new(3, arithmetic);
    let dims = cfg.hidden_size as i32;
    let type_w = p.weight("type_emb.weight");
    let te = p.op(T::Take { axis: 0 }, vec![type_w, Value::Input(1)]);
    let te = p.reshape(te, &[1, dims]);
    let mut x = p.binary(B::Add, Value::Input(0), te);
    let mask = p.mask(sequence, true, cfg.local_attention);
    let heads = (dims / 64).max(1);
    let d = dims / heads;
    for i in 0..cfg.head_layers {
        let prefix = format!("head.layers.{i}");
        let normed = p.norm(
            x,
            &format!("{prefix}.norm1.weight"),
            Some(&format!("{prefix}.norm1.bias")),
            cfg.norm_eps,
        );
        let qkv = p.linear(
            normed,
            &format!("{prefix}.self_attn.in_proj_weight"),
            Some(&format!("{prefix}.self_attn.in_proj_bias")),
        );
        let att = p.attention(qkv, sequence, heads, d, mask, None);
        let att = p.linear(
            att,
            &format!("{prefix}.self_attn.out_proj.weight"),
            Some(&format!("{prefix}.self_attn.out_proj.bias")),
        );
        x = p.binary(B::Add, x, att);
        let normed = p.norm(
            x,
            &format!("{prefix}.norm2.weight"),
            Some(&format!("{prefix}.norm2.bias")),
            cfg.norm_eps,
        );
        let hid = p.linear(
            normed,
            &format!("{prefix}.linear1.weight"),
            Some(&format!("{prefix}.linear1.bias")),
        );
        let zero = p.b.scalar(0.0);
        let hid = p.binary(B::Maximum, hid, zero);
        let mlp = p.linear(
            hid,
            &format!("{prefix}.linear2.weight"),
            Some(&format!("{prefix}.linear2.bias")),
        );
        x = p.binary(B::Add, x, mlp);
    }
    let marker = p.op(T::Take { axis: 0 }, vec![x, Value::Input(2)]);
    let m = p.norm(
        marker,
        "scorer.0.weight",
        Some("scorer.0.bias"),
        cfg.norm_eps,
    );
    let m = p.linear(m, "scorer.1.weight", Some("scorer.1.bias"));
    let m = p.gelu(m);
    let logits = p.linear(m, "scorer.3.weight", Some("scorer.3.bias"));
    let logits = p.reshape(logits, &[markers]);
    let probs = p.op(T::Softmax { axis: -1 }, vec![logits]);
    let h0 = p.slice(x, 0, 0, 1);
    let h0 = p.reshape(h0, &[1, dims]);
    p.finish(vec![logits, probs, h0])
}

/// Inputs: first hidden row `[1,H]`, calibrated host features `[1,4]`.
pub fn action() -> Plan {
    action_with(Arithmetic::Native)
}
pub fn action_with(arithmetic: Arithmetic) -> Plan {
    let mut p = Build::new(2, arithmetic);
    let pooled = p.op(
        T::Concatenate { axis: 1, inputs: 2 },
        vec![Value::Input(0), Value::Input(1)],
    );
    let a = p.linear(pooled, "act_head.0.weight", Some("act_head.0.bias"));
    let a = p.gelu(a);
    let a = p.linear(a, "act_head.2.weight", Some("act_head.2.bias"));
    p.finish(vec![a])
}

/// Host readback boundary in the original model: keep its f32 order and sorting.
pub fn action_features(probabilities: &[f32]) -> anyhow::Result<[f32; 4]> {
    anyhow::ensure!(
        !probabilities.is_empty() && probabilities.iter().all(|p| p.is_finite()),
        "invalid marker probabilities"
    );
    let k = probabilities.len().max(2) as f32;
    let entropy = -probabilities
        .iter()
        .map(|&x| x * x.max(1e-9).ln())
        .sum::<f32>()
        / k.ln();
    let mut sorted = probabilities.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let top1 = *sorted.last().unwrap();
    let top2 = sorted[sorted.len().saturating_sub(2)];
    Ok([top1, top1 - top2, entropy, k / 255.0])
}
