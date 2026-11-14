//! Original scalar shared/routed FFN order, including BF16 boundaries and
//! ordered six-expert reduction. This module only constructs computation data.
use crate::{
    graph::{Builder, Graph, Value},
    packed::Projection,
};
use lisa_backend::{Kernel, PackedCompute as K, TensorOperation as T};
const HIDDEN: i32 = 5120;
const MID: i32 = 2304;
/// Weight order is part of the binding contract. Expert projections remain packed.
pub fn requirements(layer: usize, experts: i32) -> [(String, Vec<i32>); 8] {
    [
        ("ffn_gate_inp.weight", vec![experts, HIDDEN]),
        ("exp_probs_b.bias", vec![experts]),
        ("ffn_gate_shexp.weight", vec![MID, HIDDEN]),
        ("ffn_up_shexp.weight", vec![MID, HIDDEN]),
        ("ffn_down_shexp.weight", vec![HIDDEN, MID]),
        ("ffn_gate_exps.weight", vec![experts, MID, HIDDEN]),
        ("ffn_up_exps.weight", vec![experts, MID, HIDDEN]),
        ("ffn_down_exps.weight", vec![experts, HIDDEN, MID]),
    ]
    .map(|(suffix, shape)| (format!("blk.{layer}.{suffix}"), shape))
}
#[derive(Clone, Copy)]
pub struct Spec {
    pub experts: i32,
    /// Router, shared gate/up/down, then routed gate/up/down.
    pub projections: [Projection; 7],
}
impl Spec {
    pub fn input_rows(self, shape: &[i32]) -> anyhow::Result<i32> {
        anyhow::ensure!(
            shape.len() == 2 && (1..=8).contains(&shape[0]) && shape[1] == HIDDEN,
            "invalid packed FFN input geometry"
        );
        Ok(shape[0])
    }
    pub fn graph(self, rows: i32) -> anyhow::Result<Graph> {
        anyhow::ensure!(
            (1..=8).contains(&rows) && (6..=1024).contains(&self.experts),
            "invalid packed FFN geometry"
        );
        for kind in self.projections {
            anyhow::ensure!(
                !matches!(
                    kind,
                    Projection::Block(
                        lisa_backend::PackedEncoding::Q2K | lisa_backend::PackedEncoding::Q4K
                    )
                ),
                "unsupported DeepSeek packed projection encoding"
            );
        }
        let mut b = Builder::new(1, 8, 0);
        let x = Value::Input(0);
        let logits = b.one(self.projections[0].linear(false), vec![Value::Weight(0), x]);
        let routing = b.call(
            Kernel::PackedCompute(K::Router {
                top_k: 6,
                scale: 1.5f32.to_bits(),
            }),
            vec![logits, Value::Weight(1)],
        );
        let gate = b.one(self.projections[1].linear(true), vec![Value::Weight(2), x]);
        let up = b.one(self.projections[2].linear(true), vec![Value::Weight(3), x]);
        let gate = b.one(Kernel::Tensor(T::Reshape(vec![1, rows * MID])), vec![gate]);
        let up = b.one(Kernel::Tensor(T::Reshape(vec![1, rows * MID])), vec![up]);
        let mid = b.one(
            Kernel::PackedCompute(K::SharedSwiGlu { round_output: true }),
            vec![gate, up],
        );
        let mid = b.one(Kernel::Tensor(T::Reshape(vec![rows, MID])), vec![mid]);
        let shared = b.one(
            self.projections[3].linear(true),
            vec![Value::Weight(4), mid],
        );
        let input = b.one(Kernel::Tensor(T::Reshape(vec![rows, 1, HIDDEN])), vec![x]);
        let gate = b.one(
            self.projections[4].experts()?,
            vec![Value::Weight(5), input, routing[1]],
        );
        let up = b.one(
            self.projections[5].experts()?,
            vec![Value::Weight(6), input, routing[1]],
        );
        let mid = b.one(
            Kernel::PackedCompute(K::WeightedSwiGlu {
                clamp: 10f32.to_bits(),
            }),
            vec![gate, up, routing[2]],
        );
        let experts = b.one(
            self.projections[6].experts()?,
            vec![Value::Weight(7), mid, routing[1]],
        );
        let routed = b.one(Kernel::PackedCompute(K::SumSix), vec![experts]);
        let graph = b.finish(vec![routed, shared, routing[0], routing[1], routing[2]]);
        graph.validate().map_err(anyhow::Error::msg)?;
        Ok(graph)
    }
}
