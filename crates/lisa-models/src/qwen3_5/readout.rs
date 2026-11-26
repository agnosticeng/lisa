//! Native MTP proposal selection; target verification keeps its own full head.
use crate::graph::{Builder, Graph, Value};
use lisa_backend::{Kernel, TensorOperation as T};
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Plan {
    pub target: (i32, i32),
    pub coarse: Option<(i32, i32)>,
}
impl Plan {
    /// Inputs: hidden, target head triple, optional coarse head triple.
    /// Outputs: token, exact scores, optional shortlist.
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
        let (id, scores, candidates) = if let Some(coarse) = self.coarse {
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
            let id = g.one(Kernel::Tensor(T::Reshape(vec![1, 1])), vec![id]);
            (id, exact, Some(ids))
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
            (id, scores, None)
        };
        let id = g.one(Kernel::Tensor(T::Reshape(vec![1, 1])), vec![id]);
        let mut outputs = vec![id, scores];
        outputs.extend(candidates);
        g.finish(outputs)
    }
}
