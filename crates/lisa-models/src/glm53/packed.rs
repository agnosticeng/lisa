//! GLM packed projection and FFN computation. No reader or device dependency.
use crate::graph::{Builder, Graph, Value};
use lisa_backend::{
    Binary, Kernel, PackedCompute as P, PackedEncoding as E, SparseF32 as S, TensorOperation as T,
};
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Plain,
    Bf16,
    Q8,
    Q4K,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Projection {
    pub kind: Kind,
    pub input: i32,
    pub output: i32,
}
impl Projection {
    fn append(self, b: &mut Builder, x: Value, w: Value, shape: &[i32]) -> anyhow::Result<Value> {
        anyhow::ensure!(
            shape.len() == 3
                && shape.iter().all(|&d| d > 0)
                && shape[2] == self.input
                && self.output > 0,
            "invalid GLM projection geometry"
        );
        let rows = shape[0]
            .checked_mul(shape[1])
            .ok_or_else(|| anyhow::anyhow!("projection row overflow"))?;
        let x = b.one(Kernel::Tensor(T::Reshape(vec![rows, self.input])), vec![x]);
        let x = b.one(Kernel::Tensor(T::Contiguous), vec![x]);
        let kernel = match self.kind {
            Kind::Plain => Kernel::PackedCompute(P::PlainLinear {
                round_output: false,
            }),
            Kind::Bf16 => Kernel::SparseF32(S::Bf16Linear { wide: rows > 8 }),
            Kind::Q8 => Kernel::PackedCompute(P::Q8Linear {
                round_output: false,
                round_input: false,
            }),
            Kind::Q4K => Kernel::PackedCompute(P::BlockLinear {
                encoding: E::Q4K,
                round_output: false,
            }),
        };
        let y = b.one(kernel, vec![w, x]);
        Ok(b.one(
            Kernel::Tensor(T::Reshape(vec![shape[0], shape[1], self.output])),
            vec![y],
        ))
    }
    pub fn graph(self, shape: &[i32]) -> anyhow::Result<Graph> {
        let mut b = Builder::new(2, 0, 0);
        let y = self.append(&mut b, Value::Input(1), Value::Input(0), shape)?;
        Ok(b.finish(vec![y]))
    }
}
pub enum WeightRole {
    Projection,
    Plain,
    Block(E),
}
pub struct Requirement {
    pub name: String,
    pub role: WeightRole,
}
#[derive(Clone, Copy)]
pub struct Mlp {
    pub sparse: bool,
    pub projections: [Projection; 3],
}
impl Mlp {
    /// Coalesce the original scalar projections, keeping routing, activation
    /// stores and expert summation in their serial arithmetic order.
    pub fn prompt_graph(self, rows: usize) -> anyhow::Result<Graph> {
        let graph = self.graph(&[1, 1, 4096], true)?;
        let projections = crate::graph::sequence::single_row_projections(&graph);
        crate::graph::sequence::expand(&graph, rows, &vec![vec![]; rows], &projections)
            .map_err(anyhow::Error::msg)
    }
    pub fn for_layer(layer: usize, projections: [Projection; 3]) -> Self {
        Self {
            sparse: layer >= 3,
            projections,
        }
    }
    pub fn requirements(layer: usize) -> Vec<Requirement> {
        let sparse = layer >= 3;
        let suffix = if sparse { "_shexp" } else { "" };
        let mut weights = ["gate", "up", "down"]
            .into_iter()
            .map(|s| Requirement {
                name: format!("blk.{layer}.ffn_{s}{suffix}.weight"),
                role: WeightRole::Projection,
            })
            .collect::<Vec<_>>();
        if sparse {
            for (name, role) in [
                ("ffn_gate_inp.weight", WeightRole::Plain),
                ("exp_probs_b.bias", WeightRole::Plain),
                ("ffn_gate_exps.weight", WeightRole::Block(E::Iq2Xxs)),
                ("ffn_up_exps.weight", WeightRole::Block(E::Iq2Xxs)),
                ("ffn_down_exps.weight", WeightRole::Block(E::Q2K)),
            ] {
                weights.push(Requirement {
                    name: format!("blk.{layer}.{name}"),
                    role,
                });
            }
        }
        weights
    }
    pub fn graph(self, shape: &[i32], combine: bool) -> anyhow::Result<Graph> {
        let mut b = Builder::new(1, if self.sparse { 8 } else { 3 }, 0);
        let x = Value::Input(0);
        let gate = self.projections[0].append(&mut b, x, Value::Weight(0), shape)?;
        let up = self.projections[1].append(&mut b, x, Value::Weight(1), shape)?;
        anyhow::ensure!(
            self.projections[0].output == self.projections[1].output,
            "GLM gate/up width mismatch"
        );
        let y = b.one(
            Kernel::SparseF32(S::ClampedSwiGlu {
                limit: 10f32.to_bits(),
            }),
            vec![gate, up],
        );
        let shared = self.projections[2].append(
            &mut b,
            y,
            Value::Weight(2),
            &[shape[0], shape[1], self.projections[0].output],
        )?;
        if !self.sparse {
            return Ok(b.finish(vec![shared]));
        }
        anyhow::ensure!(
            shape == [1, 1, 4096],
            "GLM packed routed FFN requires a scalar row"
        );
        let x = b.one(Kernel::Tensor(T::Reshape(vec![1, 4096])), vec![x]);
        let logits = b.one(
            Kernel::SparseF32(S::RouterLinear),
            vec![Value::Weight(3), x],
        );
        let routing = b.call(
            Kernel::SparseF32(S::RouteSigmoid {
                top_k: 8,
                scale: 2.5f32.to_bits(),
            }),
            vec![logits, Value::Weight(4)],
        );
        let x = b.one(Kernel::Tensor(T::Reshape(vec![1, 1, 4096])), vec![x]);
        let gate = b.one(
            Kernel::PackedCompute(P::ExpertLinear {
                encoding: E::Iq2Xxs,
            }),
            vec![Value::Weight(5), x, routing[0]],
        );
        let up = b.one(
            Kernel::PackedCompute(P::ExpertLinear {
                encoding: E::Iq2Xxs,
            }),
            vec![Value::Weight(6), x, routing[0]],
        );
        // Preserve the standalone activation's F32 store before route weighting.
        let y = b.one(
            Kernel::SparseF32(S::WeightedClampedSwiGlu {
                limit: 10f32.to_bits(),
            }),
            vec![gate, up, routing[1]],
        );
        let experts = b.one(
            Kernel::PackedCompute(P::ExpertLinear { encoding: E::Q2K }),
            vec![Value::Weight(7), y, routing[0]],
        );
        let experts = b.one(Kernel::Tensor(T::Reshape(vec![8, 4096])), vec![experts]);
        let routed = b.one(
            Kernel::SparseF32(S::SumEight {
                rows: 1,
                width: 4096,
            }),
            vec![experts],
        );
        let routed = b.one(Kernel::Tensor(T::Reshape(vec![1, 1, 4096])), vec![routed]);
        let outputs = if combine {
            vec![b.one(Kernel::Tensor(T::Binary(Binary::Add)), vec![routed, shared])]
        } else {
            vec![routed, shared]
        };
        Ok(b.finish(outputs))
    }
}
