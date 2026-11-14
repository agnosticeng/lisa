//! Qwen4 expert routing and execution plans. All shape gates live here.
use crate::{
    graph::{Graph, Value as V},
    program::Build,
};
use anyhow::{Result, ensure};
use lisa_backend::{
    AffineExperts as E, Binary as B, Kernel, ScalarType as D, TensorOperation as T, Unary as U,
};
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Features {
    pub matrix_router: bool,
    pub tiled_experts: bool,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Spec {
    pub experts: i32,
    pub hidden: i32,
    pub intermediate: i32,
    pub shared: i32,
    pub top_k: i32,
    pub quant: [(i32, i32); 4],
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Branch {
    Decode,
    Indirect,
    Gather,
}
pub fn join_shared() -> Graph {
    let mut p = crate::graph::Builder::new(6, 0, 0);
    let out = (0..3)
        .map(|i| {
            p.one(
                Kernel::Tensor(T::Concatenate { axis: 0, inputs: 2 }),
                vec![V::Input(i), V::Input(i + 3)],
            )
        })
        .collect::<Vec<_>>();
    p.finish(out)
}
impl Spec {
    pub fn branch(self, shape: &[i32], features: Features) -> Result<Branch> {
        ensure!(
            shape.len() == 3
                && shape.iter().all(|&n| n > 0)
                && shape[2] == self.hidden
                && self.experts > 0
                && self.top_k > 0
                && self.top_k <= self.experts
                && self.intermediate > 0
                && self.shared > 0,
            "invalid expert geometry"
        );
        let assignments = shape[0]
            .checked_mul(shape[1])
            .and_then(|n| n.checked_mul(self.top_k))
            .ok_or_else(|| anyhow::anyhow!("expert assignment overflow"))?;
        if shape[1] <= 8
            && self.quant.iter().all(|&q| q == (32, 4))
            && self.shared == self.intermediate
            && self.hidden % 128 == 0
            && self.intermediate % 32 == 0
            && self.experts % 128 == 0
            && self.top_k <= 32
        {
            return Ok(Branch::Decode);
        }
        if shape[1] > 8
            && features.tiled_experts
            && self.experts == 512
            && self.hidden == 2560
            && self.intermediate == 640
            && (2048..512 * 64).contains(&assignments)
        {
            return Ok(Branch::Indirect);
        }
        Ok(Branch::Gather)
    }
    /// Inputs: activation, router, routed gate/up/down triples, shared
    /// gate/up/down triples, shared selector triple, optional joined gate/up.
    pub fn graph(self, shape: &[i32], features: Features) -> Result<Graph> {
        let branch = self.branch(shape, features)?;
        let [b, s, h] = <[i32; 3]>::try_from(shape).unwrap();
        let rows = b * s;
        let k = self.top_k;
        let br = rows * k;
        let e = self.experts;
        let n = self.intermediate;
        let schema = Default::default();
        let mut p = Build::new(if branch == Branch::Decode { 26 } else { 23 }, &schema);
        let x = V::Input(0);
        let logits = if b == 1 && s == 1 {
            let xf = p.cast(x, D::F32);
            let xf = p.reshape(xf, &[h]);
            let y = p.graph.one(
                Kernel::AffineExperts(E::RouterVector),
                vec![xf, V::Input(1)],
            );
            p.reshape(y, &[b, s, e])
        } else if features.matrix_router && (32..=1024).contains(&rows) && h == 2560 && e == 512 {
            let y = p.graph.one(
                Kernel::AffineExperts(E::RouterMatrix { rows }),
                vec![x, V::Input(1)],
            );
            p.reshape(y, &[b, s, e])
        } else {
            let xf = p.cast(x, D::F32);
            let w = p.transpose(V::Input(1), &[1, 0]);
            p.op(T::Matmul, vec![xf, w])
        };
        if branch == Branch::Decode {
            let flat = p.reshape(x, &[rows, h]);
            let logits = p.reshape(logits, &[rows, e]);
            let route = p.graph.call(
                Kernel::AffineExperts(E::Route {
                    top_k: k,
                    experts: e,
                    hidden: h,
                    rows,
                }),
                vec![logits, flat, V::Input(20), V::Input(21), V::Input(22)],
            );
            let ids = p.reshape(route[0], &[br]);
            let xrow = p.graph.constant(
                (0..br)
                    .flat_map(|i| ((i / k) as u32).to_le_bytes())
                    .collect(),
                vec![br],
                D::U32,
            );
            let weights = p.reshape(route[1], &[br]);
            let gate = p.reshape(route[2], &[rows]);
            let act = p.graph.one(
                Kernel::AffineExperts(E::GateUp {
                    top_k: k,
                    hidden: h,
                    intermediate: n,
                    rows,
                    wide: s != 1,
                }),
                vec![
                    V::Input(2),
                    V::Input(3),
                    V::Input(4),
                    V::Input(5),
                    V::Input(6),
                    V::Input(7),
                    V::Input(23),
                    V::Input(24),
                    V::Input(25),
                    flat,
                    ids,
                    xrow,
                ],
            );
            let y = p.graph.one(
                Kernel::AffineExperts(E::DownCombine {
                    top_k: k,
                    hidden: h,
                    intermediate: n,
                    rows,
                    wide: s != 1,
                }),
                vec![
                    V::Input(8),
                    V::Input(9),
                    V::Input(10),
                    V::Input(17),
                    V::Input(18),
                    V::Input(19),
                    act,
                    ids,
                    weights,
                    gate,
                ],
            );
            let y = p.reshape(y, &[b, s, h]);
            return Ok(p.finish(vec![y])?.graph);
        }
        let neg = p.unary(U::Negate, logits);
        let ids = p.op(
            T::ArgPartition {
                kth: k - 1,
                axis: -1,
            },
            vec![neg],
        );
        let mut ids = p.slice(ids, -1, 0, k);
        if branch == Branch::Indirect {
            ids = p.contiguous(ids);
        }
        let selected = p.op(T::TakeAlong { axis: -1 }, vec![logits, ids]);
        let weights = p.op(T::Softmax { axis: -1 }, vec![selected]);
        let (down, inverse) = if branch == Branch::Indirect {
            let flat = p.reshape(ids, &[br]);
            let sorted = p.graph.call(
                Kernel::AffineExperts(E::CountingSort {
                    experts: e,
                    top_k: k,
                }),
                vec![flat],
            );
            let tiles = p.graph.one(
                Kernel::AffineExperts(E::Tiles {
                    rows: br,
                    experts: e,
                }),
                vec![sorted[0]],
            );
            let act = p.graph.one(
                Kernel::AffineExperts(E::TiledGateUp {
                    rows: br,
                    experts: e,
                    hidden: h,
                    intermediate: n,
                }),
                vec![
                    x,
                    V::Input(2),
                    V::Input(3),
                    V::Input(4),
                    V::Input(5),
                    V::Input(6),
                    V::Input(7),
                    sorted[0],
                    sorted[1],
                    tiles,
                ],
            );
            let down = p.graph.one(
                Kernel::AffineExperts(E::TiledDown {
                    rows: br,
                    experts: e,
                    hidden: h,
                    intermediate: n,
                }),
                vec![
                    act,
                    V::Input(8),
                    V::Input(9),
                    V::Input(10),
                    sorted[0],
                    tiles,
                ],
            );
            (p.reshape(down, &[br, h]), sorted[2])
        } else {
            let flat = p.reshape(x, &[rows, h]);
            let ids = p.reshape(ids, &[br]);
            let order = p.op(T::Argsort, vec![ids]);
            let inverse = p.op(T::Argsort, vec![order]);
            let divisor = p.graph.constant_i32(&[k], &[]);
            let token = p.op(T::FloorDivide, vec![order, divisor]);
            let sorted_x = p.op(T::Take { axis: 0 }, vec![flat, token]);
            let sorted_ids = p.op(T::Take { axis: 0 }, vec![ids, order]);
            let gather = |p: &mut Build<'_>, x, base, width, out_width| {
                let x = p.reshape(x, &[br, 1, width]);
                let y = p.graph.one(
                    Kernel::AffineExperts(E::Gather),
                    vec![
                        x,
                        V::Input(base),
                        V::Input(base + 1),
                        V::Input(base + 2),
                        sorted_ids,
                    ],
                );
                p.reshape(y, &[br, out_width])
            };
            let gate = gather(&mut p, sorted_x, 2, h, n);
            let up = gather(&mut p, sorted_x, 5, h, n);
            let act = p.unary(U::Silu, gate);
            let act = p.binary(B::Multiply, act, up);
            (gather(&mut p, act, 8, n, h), inverse)
        };
        let linear = |p: &mut Build<'_>, x, base, quant: (i32, i32)| {
            p.graph.one(
                Kernel::AffineLinear {
                    group_size: quant.0,
                    bits: quant.1,
                },
                vec![x, V::Input(base), V::Input(base + 1), V::Input(base + 2)],
            )
        };
        let selector = linear(&mut p, x, 20, self.quant[3]);
        let gate = linear(&mut p, x, 11, self.quant[0]);
        let up = linear(&mut p, x, 14, self.quant[1]);
        let gate = p.reshape(gate, &[rows, self.shared]);
        let up = p.reshape(up, &[rows, self.shared]);
        let shared = p.graph.one(Kernel::SwiGlu, vec![gate, up]);
        let shared = p.reshape(shared, &[b, s, self.shared]);
        let shared = linear(&mut p, shared, 17, self.quant[2]);
        let weights = p.reshape(weights, &[br]);
        let inverse = p.reshape(inverse, &[br]);
        let shared = p.reshape(shared, &[rows, h]);
        let selector = p.reshape(selector, &[rows]);
        let out = p.graph.one(
            Kernel::AffineExperts(E::SortedCombine {
                top_k: k,
                hidden: h,
                rows,
            }),
            vec![down, weights, inverse, shared, selector],
        );
        let out = p.reshape(out, &[b, s, h]);
        Ok(p.finish(vec![out])?.graph)
    }
}

pub struct Names {
    pub router: String,
    pub switched: String,
    pub shared: String,
    pub gate: &'static str,
    pub up: &'static str,
    pub down: &'static str,
    pub selector: &'static str,
}
pub fn names(prefix: &str) -> Names {
    Names {
        router: format!("{prefix}.gate.weight"),
        switched: format!("{prefix}.switch_mlp"),
        shared: format!("{prefix}.shared_expert"),
        gate: "gate_proj",
        up: "up_proj",
        down: "down_proj",
        selector: "shared_expert_gate",
    }
}
pub fn switch_names(prefix: &str) -> [String; 9] {
    std::array::from_fn(|i| {
        format!(
            "{prefix}.{}.{}",
            ["gate_proj", "up_proj", "down_proj"][i / 3],
            ["weight", "scales", "biases"][i % 3]
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn expert_shape_gates_follow_available_contracts() {
        let spec = Spec {
            experts: 512,
            hidden: 2560,
            intermediate: 640,
            shared: 640,
            top_k: 10,
            quant: [(32, 4); 4],
        };
        let enabled = Features {
            matrix_router: true,
            tiled_experts: true,
        };
        let portable = Features {
            matrix_router: false,
            tiled_experts: false,
        };
        for (shape, want) in [
            ([1, 8, 2560], Branch::Decode),
            ([2, 1, 2560], Branch::Decode),
            ([1, 9, 2560], Branch::Gather),
            ([1, 204, 2560], Branch::Gather),
            ([1, 205, 2560], Branch::Indirect),
        ] {
            assert_eq!(spec.branch(&shape, enabled).unwrap(), want);
            spec.graph(&shape, enabled).unwrap().validate().unwrap();
        }
        assert_eq!(
            spec.branch(&[1, 205, 2560], portable).unwrap(),
            Branch::Gather
        );
        let plan = spec.graph(&[1, 205, 2560], portable).unwrap();
        assert!(!plan.calls.iter().any(|call| matches!(
            call.kernel,
            Kernel::AffineExperts(
                E::RouterMatrix { .. } | E::TiledGateUp { .. } | E::TiledDown { .. }
            )
        )));
        assert!(spec.graph(&[1, 205, 128], enabled).is_err());
        assert!(spec.graph(&[i32::MAX, 256, 2560], enabled).is_err());
    }
}
