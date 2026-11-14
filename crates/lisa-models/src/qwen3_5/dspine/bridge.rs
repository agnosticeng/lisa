//! Code-bridge and exact-sender refinement arithmetic, including every cast.
use super::layers::Linear;
use crate::{
    graph::{Graph, Value},
    program::Build,
};
use lisa_backend::{Binary, Kernel, Refinement as R, ScalarType as D, TensorOperation as T, Unary};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Bridge {
    pub shape: [i32; 3],
    pub layer: usize,
    pub layers: usize,
    pub rank: i32,
    pub codes: i32,
    pub epsilon: u32,
    pub next_epsilon: u32,
    pub stash: bool,
    pub small_rows: bool,
}
impl Bridge {
    /// Inputs: hidden, IDs, next norm, final read, early reads, codebook,
    /// packed projection, gate bias, feature writes. Outputs: normed, updated,
    /// optionally raw/rms/packed refinement state.
    pub fn graph(&self) -> anyhow::Result<Graph> {
        let [b, t, h] = self.shape;
        anyhow::ensure!(
            b > 0
                && t > 0
                && h == 5120
                && self.layer < self.layers
                && self.rank == 512
                && self.codes == 1024,
            "unsupported code bridge geometry"
        );
        let schema = Default::default();
        let mut p = Build::new(9, &schema);
        let raw = p.cast(Value::Input(0), D::BF16);
        let normalized = p.graph.call(
            Kernel::Refinement(R::Normalize {
                epsilon: self.epsilon,
                read: true,
            }),
            vec![raw, Value::Input(2)],
        );
        let read = if self.layer + 1 == self.layers {
            Value::Input(3)
        } else {
            p.op(
                T::Select {
                    axis: 0,
                    index: self.layer as i32,
                },
                vec![Value::Input(4)],
            )
        };
        let lin = |shape, output| Linear {
            shape,
            output,
            quantized: false,
            bias: false,
            small_rows: self.small_rows,
        };
        let projected = lin(vec![b, t, h], self.codes).append(&mut p, normalized[0], &[read])?;
        let codes = p.graph.one(
            Kernel::Refinement(R::NormalizeCodes {
                epsilon: self.epsilon,
            }),
            vec![projected, Value::Input(1), Value::Input(5)],
        );
        let packed =
            lin(vec![b, t, self.codes], 3 * self.rank).append(&mut p, codes, &[Value::Input(6)])?;
        let packed = p.reshape(packed, &[b * t, 3 * self.rank]);
        let message = p.graph.one(
            Kernel::Refinement(R::FeatureMessage { batch: b, rows: t }),
            vec![packed, Value::Input(7)],
        );
        let write = p.op(
            T::Select {
                axis: 0,
                index: self.layer as i32,
            },
            vec![Value::Input(8)],
        );
        let delta = lin(vec![b, t, self.rank], h).append(&mut p, message, &[write])?;
        let updated = p.graph.one(
            Kernel::Refinement(R::FeatureUpdate),
            vec![raw, delta, normalized[1]],
        );
        let normed = p.graph.call(
            Kernel::Refinement(R::Normalize {
                epsilon: self.next_epsilon,
                read: false,
            }),
            vec![updated, Value::Input(2)],
        )[0];
        let mut out = vec![normed, updated];
        if self.stash {
            out.push(p.reshape(raw, &[b * t, h]));
            out.push(p.reshape(normalized[1], &[b * t]));
            out.push(packed);
        }
        Ok(p.finish(out)?.graph)
    }
}

pub fn project_tg(hidden: i32) -> anyhow::Result<Graph> {
    let schema = Default::default();
    let mut p = Build::new(3, &schema);
    let head = p.cast(Value::Input(0), D::F32);
    let norm = p.cast(Value::Input(1), D::F32);
    let norm = p.reshape(norm, &[1, hidden]);
    let weighted = p.binary(Binary::Multiply, head, norm);
    let weighted = p.cast(weighted, D::BF16);
    let write = p.op(T::Select { axis: 0, index: 4 }, vec![Value::Input(2)]);
    let y = p.op(T::Matmul, vec![weighted, write]);
    Ok(p.finish(vec![y])?.graph)
}
pub fn project_tsv(rows: i32, codes: i32, rank: i32, small_rows: bool) -> anyhow::Result<Graph> {
    let schema = Default::default();
    let mut p = Build::new(3, &schema);
    let x = p.op(T::Take { axis: 0 }, vec![Value::Input(1), Value::Input(0)]);
    let x = p.cast(x, D::F32);
    let scale = p.scalar((codes as f32).sqrt(), D::F32);
    let x = p.binary(Binary::Multiply, x, scale);
    let x = p.cast(x, D::BF16);
    let y = Linear {
        shape: vec![rows, codes],
        output: 2 * rank,
        quantized: false,
        bias: false,
        small_rows,
    }
    .append(&mut p, x, &[Value::Input(2)])?;
    Ok(p.finish(vec![y])?.graph)
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Refine {
    pub rows: i32,
    pub count: i32,
    pub hidden: i32,
    pub rank: i32,
    pub codes: i32,
    pub tables: bool,
    pub small_rows: bool,
}
impl Refine {
    /// Inputs: candidates, selected head rows, raw/rms/packed stash, norm,
    /// gate bias, feature writes, sender projection, codebook, anchor IDs,
    /// optionally the two vocabulary tables.
    pub fn graph(&self) -> anyhow::Result<Graph> {
        let (n, k, h, r) = (self.rows, self.count, self.hidden, self.rank);
        anyhow::ensure!(
            n > 0 && k == 16 && h == 5120 && r == 512 && self.codes == 1024,
            "unsupported sender refinement geometry"
        );
        let schema = Default::default();
        let mut p = Build::new(if self.tables { 13 } else { 11 }, &schema);
        let candidates = Value::Input(0);
        let flat_ids = p.reshape(candidates, &[n * k]);
        let head = p.reshape(Value::Input(1), &[n, k, h]);
        let raw = p.slice(Value::Input(2), 0, 1, n + 1);
        let raw = p.reshape(raw, &[n, h]);
        let rms = p.slice(Value::Input(3), 0, 1, n + 1);
        let rms = p.reshape(rms, &[n, 1, 1]);
        let q = p.slice(Value::Input(4), 0, 1, n + 1);
        let q = p.slice(q, 1, 0, r);
        let q = p.reshape(q, &[n, r]);
        let q = p.cast(q, D::F32);
        let norm = p.cast(Value::Input(5), D::F32);
        let tg = if self.tables {
            p.op(T::Take { axis: 0 }, vec![Value::Input(11), flat_ids])
        } else {
            let head_flat = p.reshape(head, &[n * k, h]);
            p.graph
                .append(
                    project_tg(h)?,
                    &[head_flat, Value::Input(5), Value::Input(7)],
                    &[],
                    &[],
                )
                .map_err(anyhow::Error::msg)?[0]
        };
        let tg = p.reshape(tg, &[n, k, r]);
        let tg = p.cast(tg, D::F32);
        let base = p.cast(raw, D::F32);
        let norm = p.reshape(norm, &[1, h]);
        let base = p.binary(Binary::Multiply, base, norm);
        let base = p.reshape(base, &[n, 1, h]);
        let head = p.cast(head, D::F32);
        let base = p.binary(Binary::Multiply, base, head);
        let base = p.op(
            T::Sum {
                axis: -1,
                keep_dims: false,
            },
            vec![base],
        );
        let senders = if n > 1 {
            let prior = p.slice(candidates, 0, 0, n - 1);
            p.concat(vec![Value::Input(10), prior], 0)
        } else {
            Value::Input(10)
        };
        let senders = p.reshape(senders, &[n * k]);
        let sender = if self.tables {
            p.op(T::Take { axis: 0 }, vec![Value::Input(12), senders])
        } else {
            p.graph
                .append(
                    project_tsv(n * k, self.codes, r, self.small_rows)?,
                    &[senders, Value::Input(9), Value::Input(8)],
                    &[],
                    &[],
                )
                .map_err(anyhow::Error::msg)?[0]
        };
        let sender = p.cast(sender, D::F32);
        let gate = p.slice(sender, 1, 0, r);
        let gate = p.reshape(gate, &[n, k, r]);
        let value = p.slice(sender, 1, r, 2 * r);
        let value = p.reshape(value, &[n, k, r]);
        let q = p.reshape(q, &[n, 1, r]);
        let message = p.binary(Binary::Add, q, gate);
        let bias = p.cast(Value::Input(6), D::F32);
        let bias = p.reshape(bias, &[1, 1, r]);
        let message = p.binary(Binary::Add, message, bias);
        let message = p.unary(Unary::Sigmoid, message);
        let message = p.binary(Binary::Multiply, message, value);
        let base = p.reshape(base, &[n, 1, k]);
        let tg = p.transpose(tg, &[0, 2, 1]);
        let score = p.op(T::Matmul, vec![message, tg]);
        let score = p.binary(Binary::Multiply, rms, score);
        let score = p.binary(Binary::Add, base, score);
        let indices = p.op(
            T::Argmax {
                axis: -1,
                keep_dims: false,
            },
            vec![score],
        );
        let result = p.graph.one(
            Kernel::Refinement(R::GreedyChain),
            vec![indices, candidates],
        );
        Ok(p.finish(vec![result])?.graph)
    }
}

pub fn candidates(count: usize) -> Graph {
    let mut p = crate::graph::Builder::new(1, 0, 0);
    let y = p.one(
        Kernel::Refinement(R::Candidates { count }),
        vec![Value::Input(0)],
    );
    p.finish(vec![y])
}
pub fn flat_ids(rows: i32, count: i32) -> Graph {
    let mut p = crate::graph::Builder::new(1, 0, 0);
    let y = p.one(
        Kernel::Tensor(T::Reshape(vec![rows * count])),
        vec![Value::Input(0)],
    );
    p.finish(vec![y])
}
pub fn anchor_ids(anchor: u32, count: usize) -> Vec<u32> {
    vec![anchor; count]
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Utility {
    FlatIds(i32, i32),
    Candidates(usize),
    Tg(i32),
    Tsv(i32, i32, i32, bool),
    Normalize(u32),
    Pack(usize),
    Start(u32),
    Finish(i32),
    Context(Linear, u32),
}
impl Utility {
    pub fn graph(&self) -> anyhow::Result<Graph> {
        let schema = Default::default();
        match self {
            Self::FlatIds(n, k) => Ok(flat_ids(*n, *k)),
            Self::Candidates(k) => Ok(candidates(*k)),
            Self::Tg(h) => project_tg(*h),
            Self::Tsv(n, c, r, small) => project_tsv(*n, *c, *r, *small),
            Self::Pack(n) => Ok(crate::traversal::concatenate(*n, 0)),
            Self::Normalize(eps) => {
                let mut p = Build::new(2, &schema);
                let out = p.graph.call(
                    Kernel::Refinement(R::Normalize {
                        epsilon: *eps,
                        read: false,
                    }),
                    vec![Value::Input(0), Value::Input(1)],
                );
                Ok(p.finish(vec![out[0]])?.graph)
            }
            Self::Start(eps) => {
                let mut p = Build::new(2, &schema);
                let x = p.cast(Value::Input(0), D::BF16);
                let n = p.op(T::RmsNorm { epsilon: *eps }, vec![x, Value::Input(1)]);
                Ok(p.finish(vec![n, x])?.graph)
            }
            Self::Finish(rows) => {
                let mut p = Build::new(1, &schema);
                let x = p.op(T::Select { axis: 0, index: 0 }, vec![Value::Input(0)]);
                let x = p.slice(x, 0, 1, *rows);
                let x = p.contiguous(x);
                Ok(p.finish(vec![x])?.graph)
            }
            Self::Context(lin, eps) => {
                let mut p = Build::new(2 + lin.weights(), &schema);
                let x = p.cast(Value::Input(0), D::BF16);
                let weights = (1..=lin.weights()).map(Value::Input).collect::<Vec<_>>();
                let x = lin.append(&mut p, x, &weights)?;
                let x = p.op(
                    T::RmsNorm { epsilon: *eps },
                    vec![x, Value::Input(1 + lin.weights())],
                );
                Ok(p.finish(vec![x])?.graph)
            }
        }
    }
}
