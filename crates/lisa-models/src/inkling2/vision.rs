//! Inkling's ViT, two-axis rotary positions and spatial projector.
//! Reference semantics: the pinned reference's image encoder (provenance in
//! `provenance.json`).
use crate::graph::{Builder, Graph, Value};
use Value::{Input as I, Weight as W};
use lisa_backend::{Binary as B, Kernel, ScalarType as D, TensorOperation as T, Unary as U};
use std::cell::RefCell;

/// A weight a plan binds, by checkpoint name.
pub struct Weight {
    pub name: String,
    pub shape: Vec<i32>,
    pub dtype: D,
}
/// A plan and the weights it binds, in binding order.
pub struct Definition {
    pub graph: Graph,
    pub weights: Vec<Weight>,
}

struct Plan {
    b: RefCell<Builder>,
    weights: RefCell<Vec<Weight>>,
    prefix: String,
}
impl Plan {
    fn new(prefix: String, inputs: usize) -> Self {
        Self {
            b: RefCell::new(Builder::new(inputs, 0, 0)),
            weights: RefCell::new(vec![]),
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
        if a.len() == 1 {
            return a[0];
        }
        self.op(
            T::Concatenate {
                axis,
                inputs: a.len(),
            },
            a,
        )
    }
    fn trans(&self, x: Value, axes: &[i32]) -> Value {
        self.op(T::Transpose(axes.to_vec()), &[x])
    }
    fn mm(&self, a: Value, b: Value) -> Value {
        self.op(T::Matmul, &[a, b])
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
    fn finish(self, outputs: Vec<Value>) -> Definition {
        let weights = self.weights.into_inner();
        let mut graph = self.b.into_inner().finish(outputs);
        graph.weights = weights.len();
        Definition { graph, weights }
    }

    fn vision_linear(&self, x: Value, name: &str, input: i32, output: i32, bias: bool) -> Value {
        let w = self.weight(&format!("{name}.weight"), &[output, input], D::F16);
        let w = self.op(T::Cast(D::F32), &[w]);
        let y = self.mm(x, self.trans(w, &[1, 0]));
        if bias {
            self.add(y, self.param(&format!("{name}.bias"), &[output]))
        } else {
            y
        }
    }
    fn ln(&self, x: Value, name: &str) -> Value {
        self.op(
            T::LayerNorm {
                epsilon: 1e-5f32.to_bits(),
            },
            &[
                x,
                self.param(&format!("{name}.weight"), &[1024]),
                self.param(&format!("{name}.bias"), &[1024]),
            ],
        )
    }
    fn gelu(&self, x: Value) -> Value {
        self.mul(
            self.mul(x, self.scalar(0.5)),
            self.add(
                self.scalar(1.),
                self.unary(
                    U::Erf,
                    self.mul(x, self.scalar(std::f32::consts::FRAC_1_SQRT_2)),
                ),
            ),
        )
    }
    fn vision_rope(&self, x: Value, n: i32) -> Value {
        let pairs = self.reshape(x, &[n, 16, 2, 16, 2]);
        let even = self.reshape(self.slice(pairs, 4, 0, 1), &[n, 16, 2, 16]);
        let odd = self.reshape(self.slice(pairs, 4, 1, 2), &[n, 16, 2, 16]);
        let a = self.binary(B::Subtract, self.mul(even, I(1)), self.mul(odd, I(2)));
        let b = self.add(self.mul(even, I(2)), self.mul(odd, I(1)));
        let pairs = self.cat(
            &[
                self.reshape(a, &[n, 16, 2, 16, 1]),
                self.reshape(b, &[n, 16, 2, 16, 1]),
            ],
            4,
        );
        self.reshape(pairs, &[n, 16, 64])
    }
}
pub fn input(n: i32) -> Definition {
    let p = Plan::new(String::new(), 2);
    let w = p.weight("v.patch_embd.weight", &[1024, 3, 20, 20], D::F16);
    let w = p.op(T::Cast(D::F32), &[w]);
    let w = p.reshape(w, &[1024, 1200]);
    let x = p.mm(I(0), p.trans(w, &[1, 0]));
    let x = p.add(x, I(1));
    let cls = p.add(
        p.param("v.class_embd", &[1024]),
        p.param("v.class_position_embd.weight", &[1024]),
    );
    let x = p.cat(&[p.reshape(cls, &[1, 1024]), x], 0);
    let x = p.ln(p.reshape(x, &[n, 1024]), "v.pre_ln");
    p.finish(vec![x])
}
pub fn layer(i: usize, n: i32) -> Definition {
    let p = Plan::new(format!("v.blk.{i}."), 3);
    let x = p.ln(I(0), "ln1");
    let q = p.vision_linear(x, "attn_q", 1024, 1024, true);
    let q = p.vision_rope(q, n);
    let k = p.vision_linear(x, "attn_k", 1024, 1024, true);
    let k = p.vision_rope(k, n);
    let v = p.vision_linear(x, "attn_v", 1024, 1024, true);
    let v = p.reshape(v, &[n, 16, 64]);
    // Explicit F32 attention is the initial vision law. The fast attention
    // dispatcher is not admitted for these short F32 image sequences.
    let qh = p.trans(q, &[1, 0, 2]);
    let kh = p.trans(k, &[1, 2, 0]);
    let vh = p.trans(v, &[1, 0, 2]);
    let scores = p.mul(p.mm(qh, kh), p.scalar(0.125));
    let probabilities = p.op(T::Softmax { axis: -1 }, &[scores]);
    let attn = p.reshape(p.trans(p.mm(probabilities, vh), &[1, 0, 2]), &[n, 1024]);
    let attn = p.vision_linear(attn, "attn_out", 1024, 1024, true);
    let x = p.add(I(0), p.mul(attn, p.param("ls1.weight", &[1024])));
    let ffn = p.ln(x, "ln2");
    let ffn = p.gelu(p.vision_linear(ffn, "ffn_up", 1024, 4096, true));
    let ffn = p.vision_linear(ffn, "ffn_down", 4096, 1024, true);
    let out = p.add(x, p.mul(ffn, p.param("ls2.weight", &[1024])));
    p.finish(vec![out])
}
pub fn project(h: i32, w: i32) -> Definition {
    let p = Plan::new(String::new(), 1);
    let x = p.slice(I(0), 0, 1, h * w + 1);
    let x = p.reshape(x, &[h / 2, 2, w / 2, 2, 1024]);
    let x = p.reshape(p.trans(x, &[0, 2, 1, 3, 4]), &[h * w / 4, 4096]);
    let x = p.gelu(p.vision_linear(x, "mm.1", 4096, 4096, false));
    let out = p.vision_linear(x, "mm.2", 4096, 2304, false);
    p.finish(vec![out])
}
