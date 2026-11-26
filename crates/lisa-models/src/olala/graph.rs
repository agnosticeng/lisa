//! Serial Olala plans over original GGUF projections and immutable F32 state.
//! Architecture follows OVHai-LLM/llama.cpp 55c22dca0c987f247c21b427c3b3a09e1a28187f.
use super::config::Config;
use crate::{
    graph::{Builder, Graph, Value},
    packed::Projection,
};
use Value::{Input as I, Weight as W};
use lisa_backend::{
    Binary as B, Kernel, PackedCompute as P, ScalarType as D, TensorOperation as T, Unary as U,
};
use std::{cell::RefCell, collections::BTreeMap};
#[derive(Clone, Copy)]
pub struct Format {
    pub projection: Projection,
    pub dtype: D,
    pub row_bytes: i32,
}
pub struct Weight {
    pub name: String,
    pub shape: Vec<i32>,
    pub dtype: D,
}
pub struct Definition {
    pub graph: Graph,
    pub weights: Vec<Weight>,
}
struct Plan<'a> {
    b: RefCell<Builder>,
    weights: RefCell<Vec<Weight>>,
    formats: &'a BTreeMap<String, Format>,
    prefix: String,
}
impl<'a> Plan<'a> {
    fn new(formats: &'a BTreeMap<String, Format>, prefix: String, inputs: usize) -> Self {
        Self {
            b: RefCell::new(Builder::new(inputs, 0, 0)),
            weights: RefCell::new(vec![]),
            formats,
            prefix,
        }
    }
    fn op(&self, t: T, a: &[Value]) -> Value {
        self.b.borrow_mut().one(Kernel::Tensor(t), a.to_vec())
    }
    fn scalar(&self, x: f32) -> Value {
        self.b.borrow_mut().constant_f32(&[x], &[1])
    }
    fn binary(&self, t: B, a: Value, b: Value) -> Value {
        self.op(T::Binary(t), &[a, b])
    }
    fn add(&self, a: Value, b: Value) -> Value {
        self.binary(B::Add, a, b)
    }
    fn mul(&self, a: Value, b: Value) -> Value {
        self.binary(B::Multiply, a, b)
    }
    fn div(&self, a: Value, b: Value) -> Value {
        self.binary(B::Divide, a, b)
    }
    fn unary(&self, u: U, a: Value) -> Value {
        self.op(T::Unary(u), &[a])
    }
    fn reshape(&self, x: Value, s: &[i32]) -> Value {
        self.op(T::Reshape(s.to_vec()), &[self.op(T::Contiguous, &[x])])
    }
    fn slice(&self, x: Value, axis: i32, start: i32, end: i32) -> Value {
        self.op(T::Slice { axis, start, end }, &[x])
    }
    fn cat(&self, a: &[Value], axis: i32) -> Value {
        self.op(
            T::Concatenate {
                axis,
                inputs: a.len(),
            },
            a,
        )
    }
    fn weight(&self, name: &str, shape: &[i32], dtype: D) -> Value {
        let name = format!("{}{name}", self.prefix);
        let mut ws = self.weights.borrow_mut();
        if let Some(i) = ws.iter().position(|w| w.name == name) {
            return W(i);
        }
        let i = ws.len();
        ws.push(Weight {
            name,
            shape: shape.to_vec(),
            dtype,
        });
        W(i)
    }
    fn param(&self, name: &str, shape: &[i32]) -> Value {
        self.weight(name, shape, D::F32)
    }
    fn matrix(&self, name: &str, input: i32, output: i32, experts: bool) -> (Value, Format) {
        let f = self.formats[&format!("{}{name}", self.prefix)];
        let cols = if f.projection == Projection::Plain {
            input
        } else {
            f.row_bytes
        };
        // Packed kernels read complete eight-row tiles. The binder pads tiny
        // views once at load, preserving the checkpoint payload byte-for-byte.
        let rows = if matches!(f.projection, Projection::Block(_)) {
            (output + 7) / 8 * 8
        } else {
            output
        };
        let shape = if experts {
            vec![256, output, cols]
        } else {
            vec![rows, cols]
        };
        (self.weight(name, &shape, f.dtype), f)
    }
    fn project(&self, w: Value, f: Format, x: Value, output: i32) -> Value {
        let y = self
            .b
            .borrow_mut()
            .one(f.projection.linear(false), vec![w, x]);
        if matches!(f.projection, Projection::Block(_)) && output % 8 != 0 {
            self.slice(y, 1, 0, output)
        } else {
            y
        }
    }
    fn linear(&self, x: Value, name: &str, input: i32, output: i32) -> Value {
        let (w, f) = self.matrix(name, input, output, false);
        self.project(w, f, self.reshape(x, &[1, input]), output)
    }
    fn norm(&self, x: Value, name: &str) -> Value {
        // The GGUF converter has already applied zero-centered gamma.
        self.op(
            T::RmsNorm {
                epsilon: 1e-6f32.to_bits(),
            },
            &[x, self.param(name, &[128])],
        )
    }
    fn geo(&self, x: Value, y: Value, name: &str, layer: usize) -> Value {
        self.op(
            T::GeodesicResidual {
                inverse_depth: (1.0 / (layer + 1) as f32).to_bits(),
            },
            &[
                x,
                y,
                self.param(&format!("geo_{name}_scale"), &[1]),
                self.param(&format!("geo_{name}_bias"), &[1]),
            ],
        )
    }
    fn relu2(&self, x: Value) -> Value {
        self.b
            .borrow_mut()
            .one(Kernel::PackedCompute(P::ReluSquared), vec![x])
    }
    fn moe(&self, c: &Config, x: Value) -> Value {
        let logits = self.linear(x, "ffn_gate_inp.weight", 1536, 256);
        let bias = self.param("exp_probs_b", &[256]);
        let routes = self.b.borrow_mut().call(
            Kernel::PackedCompute(P::SigmoidRouter256 {
                scale: c.scale.to_bits(),
            }),
            vec![logits, bias],
        );
        let selected = routes[0];
        let probs = routes[1];
        let latent = self.linear(x, "ffn_latent_down.weight", 1536, 384);
        let (up, uf) = self.matrix("ffn_up_exps.weight", 384, 768, true);
        let (down, df) = self.matrix("ffn_down_exps.weight", 768, 384, true);
        // Address selected original blocks directly on the device. Each expert
        // retains the dense projection's dot-product order; the final ordered
        // six-term sum below is deliberately unchanged.
        let up_input = self.reshape(latent, &[1, 1, 384]);
        let h = self.b.borrow_mut().one(
            uf.projection.experts().unwrap(),
            vec![up, up_input, selected],
        );
        let h = self.relu2(h);
        let ys = self
            .b
            .borrow_mut()
            .one(df.projection.experts().unwrap(), vec![down, h, selected]);
        let routed = self
            .b
            .borrow_mut()
            .one(Kernel::PackedCompute(P::WeightedSumSix), vec![ys, probs]);
        let routed = self.linear(routed, "ffn_latent_up.weight", 384, 1536);
        let shared = self.relu2(self.linear(x, "ffn_up_shexp.weight", 1536, 1536));
        let shared = self.linear(shared, "ffn_down_shexp.weight", 1536, 1536);
        self.add(routed, shared)
    }
    fn finish(self, outputs: Vec<Value>) -> Definition {
        let weights = self.weights.into_inner();
        let mut graph = self.b.into_inner().finish(outputs);
        graph.weights = weights.len();
        Definition { graph, weights }
    }
}
/// Inputs: hidden, recurrent state or K, V, previous raw K, previous raw V,
/// and log(min(position + 1, window)) evaluated with the native unary law.
/// Serial rows use the same recurrence and projection law during prefill/decode.
pub fn layer(
    c: &Config,
    index: usize,
    has_history: bool,
    formats: &BTreeMap<String, Format>,
) -> Definition {
    let p = Plan::new(formats, format!("blk.{index}."), 6);
    let (mixed, states) = if c.layers[index] {
        let stat = p.linear(I(0), "ssm_in.weight", 1536, 6288);
        let dynamic = p.linear(I(0), "ssm_in_dyn.weight", 1536, 1056);
        let args = vec![
            dynamic,
            stat,
            p.param("ssm_m3_bias", &[48, 8, 128]),
            p.param("ssm_m3_mxz", &[48, 8, 64]),
            p.param("ssm_mimo_o", &[48, 4, 64]),
            p.param("ssm_m3_norms", &[2, 128]),
            p.param("ssm_m3_misc", &[2, 48]),
            I(1),
        ];
        let out = p.b.borrow_mut().call(
            Kernel::Tensor(T::Mamba3Mimo {
                heads: 48,
                epsilon: c.epsilon.to_bits(),
            }),
            args,
        );
        (
            p.linear(out[0], "attn_output.weight", 3072, 1536),
            vec![out[1]],
        )
    } else {
        let q = p.reshape(p.linear(I(0), "attn_q.weight", 1536, 6144), &[48, 128]);
        let factors = |axis: &str| {
            let a = p.reshape(
                p.linear(I(0), &format!("attn_wa_{axis}.weight"), 1536, 48),
                &[12, 4],
            );
            let b = p.reshape(
                p.linear(I(0), &format!("attn_wb_{axis}.weight"), 1536, 512),
                &[4, 128],
            );
            p.mul(p.op(T::Matmul, &[a, b]), p.scalar(0.25))
        };
        let rawk = factors("k");
        let rawv = factors("v");
        let kg = p.linear(I(0), "attn_shift_k.weight", 1536, 12);
        let vg = p.linear(I(0), "attn_shift_v.weight", 1536, 12);
        let shifted = p.b.borrow_mut().call(
            Kernel::PackedCompute(P::RowGateBlendPair { zero: !has_history }),
            vec![rawk, rawv, I(3), I(4), kg, vg],
        );
        let k = p.norm(shifted[0], "attn_k_norm.weight");
        let v = shifted[1];
        let q = p.norm(q, "attn_q_norm.weight");
        let logpos = I(5);
        let q = p.mul(q, logpos);
        let q = p.mul(
            q,
            p.reshape(p.param("attn_softmax_scaler", &[48]), &[48, 1]),
        );
        let q = p.reshape(q, &[12, 4, 128]);
        let k = p.reshape(k, &[12, 1, 128]);
        let v = p.reshape(v, &[12, 1, 128]);
        let k = if !has_history {
            k
        } else {
            p.cat(&[I(1), k], 1)
        };
        let v = if !has_history {
            v
        } else {
            p.cat(&[I(2), v], 1)
        };
        let scores = p.b.borrow_mut().one(
            Kernel::PackedCompute(P::SmallRowMatmul {
                transpose_rhs: true,
            }),
            vec![q, k],
        );
        let scores = p.mul(scores, p.scalar(1. / 128.));
        let scores = p.mul(
            p.unary(U::Tanh, p.div(scores, p.scalar(c.softcap))),
            p.scalar(c.softcap),
        );
        let prob = p.op(T::Softmax { axis: -1 }, &[scores]);
        let out = p.b.borrow_mut().one(
            Kernel::PackedCompute(P::SmallRowMatmul {
                transpose_rhs: false,
            }),
            vec![prob, v],
        );
        let lambda = p.linear(I(0), "attn_lambda.weight", 1536, 12);
        let gate = p.linear(I(0), "attn_gate.weight", 1536, 4608);
        let combined = p.b.borrow_mut().one(
            Kernel::PackedCompute(P::DifferentialSiluGate {
                bias: c.gate_bias.to_bits(),
            }),
            vec![out, lambda, gate],
        );
        (
            p.linear(combined, "attn_output.weight", 4608, 1536),
            vec![k, v, rawk, rawv],
        )
    };
    let x = p.geo(I(0), mixed, "mixer", index);
    let y = p.moe(c, x);
    let x = p.geo(x, y, "mlp", index);
    let mut out = vec![x];
    out.extend(states);
    p.finish(out)
}
pub fn head(formats: &BTreeMap<String, Format>) -> Definition {
    let p = Plan::new(formats, String::new(), 1);
    let y = p.linear(I(0), "output.weight", 1536, 151936);
    p.finish(vec![y])
}

/// Layer-major prompt rows, retaining sequential Mamba/shift/cache feedback.
pub fn layer_sequence(
    c: &Config,
    index: usize,
    rows: usize,
    formats: &BTreeMap<String, Format>,
) -> Result<Definition, String> {
    use crate::graph::sequence::{self, Feedback};
    let mut def = layer(c, index, true, formats);
    let feedback = (0..rows)
        .map(|row| {
            if row == 0 {
                vec![]
            } else {
                (1..=if c.layers[index] { 1 } else { 4 })
                    .map(|i| Feedback {
                        input: i,
                        output: i,
                        view: None,
                    })
                    .collect()
            }
        })
        .collect::<Vec<_>>();
    let projections = sequence::single_row_projections(&def.graph);
    def.graph = sequence::expand(&def.graph, rows, &feedback, &projections)?;
    Ok(def)
}
