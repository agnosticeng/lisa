//! Native MTP proposal selection; target verification keeps its own full head.
use crate::graph::{Builder, Graph, Value};
use lisa_backend::{Kernel, TensorOperation as T, Unary};
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Plan {
    pub target: (i32, i32),
    pub coarse: Option<(i32, i32)>,
    pub confidence: bool,
}
impl Plan {
    /// Inputs: hidden, target head triple, optional coarse head triple.
    /// Outputs: token, exact scores, optional shortlist, optional confidence.
    pub fn graph(&self) -> Graph {
        let mut g = Builder::new(if self.coarse.is_some() { 7 } else { 4 }, 0, 0);
        let linear = |g: &mut Builder, x, base, quant: (i32, i32)| {
            g.one(
                Kernel::AffineLinear {
                    group_size: quant.0,
                    bits: quant.1,
                },
                vec![
                    x,
                    Value::Input(base),
                    Value::Input(base + 1),
                    Value::Input(base + 2),
                ],
            )
        };
        let (id, scores, candidates, confidence) = if let Some(coarse) = self.coarse {
            let scores = linear(&mut g, Value::Input(0), 4, coarse);
            let ids = g.one(Kernel::Tensor(T::Top32Indices), vec![scores]);
            let ids = g.one(Kernel::Tensor(T::Reshape(vec![32])), vec![ids]);
            let mut inputs = vec![Value::Input(0)];
            for i in 1..4 {
                inputs.push(g.one(
                    Kernel::Tensor(T::Take { axis: 0 }),
                    vec![Value::Input(i), ids],
                ));
            }
            let exact = g.one(
                Kernel::AffineLinear {
                    group_size: self.target.0,
                    bits: self.target.1,
                },
                inputs,
            );
            let local = g.one(
                Kernel::Tensor(T::Argmax {
                    axis: -1,
                    keep_dims: false,
                }),
                vec![exact],
            );
            let local = g.one(Kernel::Tensor(T::Reshape(vec![1])), vec![local]);
            let id = g.one(Kernel::Tensor(T::Take { axis: 0 }), vec![ids, local]);
            // The original shortlist path computes confidence even when its
            // consumer is disabled. Keep the same operation chain and order.
            let sm = g.one(Kernel::Tensor(T::Softmax { axis: -1 }), vec![exact]);
            let p = g.one(Kernel::Tensor(T::Take { axis: -1 }), vec![sm, local]);
            let logp = g.one(Kernel::Tensor(T::Unary(Unary::Log)), vec![p]);
            let id = g.one(Kernel::Tensor(T::Reshape(vec![1, 1])), vec![id]);
            (id, exact, Some(ids), self.confidence.then_some(logp))
        } else {
            let scores = linear(&mut g, Value::Input(0), 1, self.target);
            let id = g.one(
                Kernel::Tensor(T::Argmax {
                    axis: -1,
                    keep_dims: false,
                }),
                vec![scores],
            );
            let id = g.one(Kernel::Tensor(T::Reshape(vec![1, 1])), vec![id]);
            let confidence = self.confidence.then(|| {
                let sm = g.one(Kernel::Tensor(T::Softmax { axis: -1 }), vec![scores]);
                let p = g.one(Kernel::Tensor(T::Take { axis: -1 }), vec![sm, id]);
                g.one(Kernel::Tensor(T::Unary(Unary::Log)), vec![p])
            });
            (id, scores, None, confidence)
        };
        let id = g.one(Kernel::Tensor(T::Reshape(vec![1, 1])), vec![id]);
        let mut outputs = vec![id, scores];
        outputs.extend(candidates);
        outputs.extend(confidence);
        g.finish(outputs)
    }
}
