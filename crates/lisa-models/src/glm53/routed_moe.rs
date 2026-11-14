//! GLM affine routed experts: scheduling, layout and exact reduction plans.
use crate::{
    graph::{Graph, Value as V},
    program::Build,
};
use lisa_backend::{Binary, IndirectAffine as A, Kernel as K, ScalarType as D, SparseF32 as S};
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Spec {
    pub batch: usize,
    pub time: usize,
    pub hidden: usize,
    pub intermediate: usize,
    pub top_k: usize,
    pub experts: usize,
    pub metadata: D,
    pub limit: u32,
    pub grouped: bool,
    pub split: bool,
    pub vector: bool,
    pub down_tile: usize,
    pub workspace: bool,
    pub shared: bool,
    pub capture: bool,
    pub preserve: bool,
}
/// DS4 shape gate for grouped expert prefill.
pub const GROUPED_MOE_MIN_ROWS: usize = 96;
impl Spec {
    pub fn rows(&self) -> usize {
        self.batch * self.time
    }
    pub fn grouped(
        rows: usize,
        top_k: usize,
        intermediate: usize,
        hidden: usize,
        enabled: Option<bool>,
    ) -> bool {
        rows >= GROUPED_MOE_MIN_ROWS
            && top_k == 8
            && intermediate % 64 == 0
            && hidden % 128 == 0
            && enabled.unwrap_or(rows >= GROUPED_MOE_MIN_ROWS)
    }
    pub fn fused_down(&self) -> bool {
        !self.grouped && self.top_k == 8 && (!self.capture || self.preserve)
    }
    pub fn slots(&self) -> bool {
        self.capture || (!self.grouped && !self.fused_down())
    }
    pub fn output_shape(&self) -> [i32; 3] {
        [self.batch as i32, self.time as i32, self.hidden as i32]
    }
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Request {
    /// Diagnostic candidate; never selected by production expert execution.
    SharedDownProbe {
        hidden: usize,
        metadata: D,
    },
    Inputs(Spec),
    RouteMap(Spec),
    Activation(Spec),
    Down(Spec),
    Shadow(Spec),
    Router {
        rows: usize,
        hidden: usize,
    },
    Select {
        top_k: usize,
        scale: u32,
    },
    Rows {
        rows: usize,
        hidden: usize,
    },
    Output {
        shape: Vec<i32>,
    },
    Add,
    Promote,
    GroupedTerm {
        row: usize,
        hidden: usize,
        intermediate: usize,
        gate_bits: i32,
        gate_group: i32,
        down_bits: i32,
        down_group: i32,
        weight: u32,
        limit: u32,
    },
    JoinRows {
        rows: usize,
        shape: Vec<i32>,
    },
}
fn sum(p: &mut Build<'_>, spec: &Spec, slots: V, shared: Option<V>) -> V {
    let sum = p.graph.one(
        K::IndirectAffine(A::Sum8 {
            rows: spec.rows(),
            hidden: spec.hidden,
        }),
        vec![slots],
    );
    if let Some(shared) = shared {
        p.binary(Binary::Add, sum, shared)
    } else {
        sum
    }
}
impl Request {
    pub fn graph(&self) -> anyhow::Result<Graph> {
        let inputs = match self {
            Self::SharedDownProbe { .. } => 6,
            Self::GroupedTerm { .. } => 7,
            Self::JoinRows { rows, .. } => *rows,
            Self::Promote => 1,
            Self::Inputs(_) => 3,
            Self::RouteMap(_) | Self::Rows { .. } | Self::Output { .. } => 1,
            Self::Router { .. } | Self::Select { .. } | Self::Add => 2,
            Self::Activation(s) => {
                if s.grouped {
                    6 + if s.workspace {
                        if s.split { 3 } else { 1 }
                    } else {
                        0
                    }
                } else {
                    3
                }
            }
            Self::Down(s) => {
                if s.grouped {
                    5 + usize::from(s.workspace) + usize::from(s.shared)
                } else {
                    2 + usize::from(s.shared)
                }
            }
            Self::Shadow(s) => 2 + usize::from(s.shared),
        };
        let schema = Default::default();
        let mut p = Build::new(inputs, &schema);
        let out = match self {
            Self::SharedDownProbe { hidden, metadata } => vec![p.graph.one(
                K::IndirectAffine(A::DownSum8SharedProjection {
                    hidden: *hidden,
                    metadata: *metadata,
                }),
                (0..6).map(V::Input).collect::<Vec<_>>(),
            )],
            Self::Promote => vec![p.cast(V::Input(0), D::F32)],
            Self::JoinRows { rows, shape } => {
                let x = p.concat((0..*rows).map(V::Input).collect(), 0);
                vec![p.reshape(x, shape)]
            }
            Self::GroupedTerm {
                row,
                hidden,
                intermediate,
                gate_bits,
                gate_group,
                down_bits,
                down_group,
                weight,
                limit,
            } => {
                let x = p.slice(V::Input(0), 0, *row as i32, *row as i32 + 1);
                let x = p.reshape(x, &[1, *hidden as i32]);
                let y = p.graph.one(
                    K::SparseF32(S::AffineProjection {
                        bits: *gate_bits,
                        group_size: *gate_group,
                        schedule: lisa_backend::AffineSchedule::Generic,
                    }),
                    vec![x, V::Input(1), V::Input(2), V::Input(3)],
                );
                let y = p.reshape(y, &[1, 2 * *intermediate as i32]);
                let gate = p.slice(y, 1, 0, *intermediate as i32);
                let up = p.slice(y, 1, *intermediate as i32, 2 * *intermediate as i32);
                let gate = p.contiguous(gate);
                let up = p.contiguous(up);
                let y = p.graph.one(
                    K::SparseF32(S::ClampedSwiGlu { limit: *limit }),
                    vec![gate, up],
                );
                let weight = p.scalar(f32::from_bits(*weight), D::F32);
                let y = p.binary(Binary::Multiply, y, weight);
                let y = p.reshape(y, &[1, *intermediate as i32]);
                let y = p.graph.one(
                    K::SparseF32(S::AffineProjection {
                        bits: *down_bits,
                        group_size: *down_group,
                        schedule: lisa_backend::AffineSchedule::Generic,
                    }),
                    vec![y, V::Input(4), V::Input(5), V::Input(6)],
                );
                let y = p.reshape(y, &[1, *hidden as i32]);
                vec![p.cast(y, D::F32)]
            }
            Self::Rows { rows, hidden } => {
                let x = p.cast(V::Input(0), D::F32);
                vec![p.reshape(x, &[*rows as i32, *hidden as i32])]
            }
            Self::Output { shape } => vec![p.reshape(V::Input(0), shape)],
            Self::Add => vec![p.binary(Binary::Add, V::Input(0), V::Input(1))],
            Self::Router { rows, hidden } => {
                let x = p.cast(V::Input(0), D::F32);
                let x = p.reshape(x, &[*rows as i32, *hidden as i32]);
                vec![
                    p.graph
                        .one(K::SparseF32(S::RouterLinear), vec![V::Input(1), x]),
                ]
            }
            Self::Select { top_k, scale } => p.graph.call(
                K::SparseF32(S::RouteSigmoid {
                    top_k: *top_k,
                    scale: *scale,
                }),
                vec![V::Input(0), V::Input(1)],
            ),
            Self::Inputs(s) => {
                let sel = p.reshape(V::Input(1), &[(s.rows() * s.top_k) as i32]);
                let weights = p.reshape(V::Input(2), &[(s.rows() * s.top_k) as i32]);
                let x = p.cast(V::Input(0), D::F32);
                let x = p.reshape(x, &[s.rows() as i32, s.hidden as i32]);
                vec![sel, weights, x]
            }
            Self::RouteMap(s) => p.graph.call(
                K::IndirectAffine(A::RouteMap {
                    rows: s.rows(),
                    top_k: s.top_k,
                    experts: s.experts,
                    tile: 32,
                }),
                vec![V::Input(0)],
            ),
            Self::Activation(s) => {
                let k = if s.grouped {
                    let common = vec![
                        V::Input(0),
                        V::Input(2),
                        V::Input(3),
                        V::Input(4),
                        V::Input(5),
                    ];
                    if s.split {
                        let mut projections = vec![];
                        for (i, up) in [false, true].into_iter().enumerate() {
                            let mut args = common.clone();
                            if s.workspace {
                                p.graph.writable_input(6 + i);
                                args.push(V::Input(6 + i));
                            }
                            projections.push(p.graph.one(
                                K::IndirectAffine(A::GroupedProjection {
                                    rows: s.rows(),
                                    top_k: s.top_k,
                                    experts: s.experts,
                                    intermediate: s.intermediate,
                                    up,
                                    vector: s.vector,
                                    metadata: s.metadata,
                                    workspace: s.workspace,
                                }),
                                args,
                            ));
                        }
                        projections.push(V::Input(1));
                        if s.workspace {
                            p.graph.writable_input(8);
                            projections.push(V::Input(8));
                        }
                        p.graph.one(
                            K::IndirectAffine(A::GroupedSwiGlu {
                                rows: s.rows() * s.top_k,
                                intermediate: s.intermediate,
                                limit: s.limit,
                                workspace: s.workspace,
                            }),
                            projections,
                        )
                    } else {
                        let mut args = common;
                        args.push(V::Input(1));
                        if s.workspace {
                            p.graph.writable_input(6);
                            args.push(V::Input(6));
                        }
                        p.graph.one(
                            K::IndirectAffine(A::GroupedPair {
                                rows: s.rows(),
                                top_k: s.top_k,
                                experts: s.experts,
                                intermediate: s.intermediate,
                                limit: s.limit,
                                metadata: s.metadata,
                                workspace: s.workspace,
                            }),
                            args,
                        )
                    }
                } else {
                    p.graph.one(
                        K::IndirectAffine(A::Pair {
                            intermediate: s.intermediate,
                            top_k: s.top_k,
                            limit: s.limit,
                            metadata: s.metadata,
                        }),
                        vec![V::Input(0), V::Input(2), V::Input(1)],
                    )
                };
                vec![k]
            }
            Self::Down(s) | Self::Shadow(s) => {
                let shadow = matches!(self, Self::Shadow(_));
                let shared = s.shared.then_some(V::Input(inputs - 1));
                let (value, slots) = if s.grouped && !shadow {
                    let mut args = (0..5).map(V::Input).collect::<Vec<_>>();
                    if s.workspace {
                        p.graph.writable_input(5);
                        args.push(V::Input(5));
                    }
                    let slots = p.graph.one(
                        K::IndirectAffine(A::GroupedDown {
                            rows: s.rows(),
                            top_k: s.top_k,
                            experts: s.experts,
                            hidden: s.hidden,
                            vector: s.vector,
                            tile: s.down_tile,
                            metadata: s.metadata,
                            workspace: s.workspace,
                        }),
                        args,
                    );
                    (sum(&mut p, s, slots, shared), Some(slots))
                } else if s.fused_down() && !shadow {
                    let args = (0..inputs).map(V::Input).collect::<Vec<_>>();
                    (
                        p.graph.one(
                            K::IndirectAffine(A::DownSum8 {
                                hidden: s.hidden,
                                shared: s.shared,
                                metadata: s.metadata,
                            }),
                            args,
                        ),
                        None,
                    )
                } else {
                    let slots = p.graph.one(
                        K::IndirectAffine(A::Gather {
                            hidden: s.hidden,
                            ds4: shadow,
                            metadata: s.metadata,
                        }),
                        vec![V::Input(0), V::Input(1)],
                    );
                    let value = if shadow {
                        sum(&mut p, s, slots, shared)
                    } else {
                        let down =
                            p.reshape(slots, &[s.rows() as i32, s.top_k as i32, s.hidden as i32]);
                        let first = p.slice(down, 1, 0, 1);
                        let mut acc = p.contiguous(first);
                        for i in 1..s.top_k {
                            let term = p.slice(down, 1, i as i32, i as i32 + 1);
                            let term = p.contiguous(term);
                            acc = p.binary(Binary::Add, acc, term);
                        }
                        p.reshape(acc, &[s.rows() as i32, s.hidden as i32])
                    };
                    (value, Some(slots))
                };
                let mut out = vec![value];
                if s.capture || shadow {
                    if let Some(slots) = slots {
                        out.push(slots);
                    }
                }
                out
            }
        };
        Ok(p.finish(out)?.graph)
    }
}

/// Historical grouped sigmoid router. Host arithmetic and stable tie ordering
/// are part of this reference law; the curated ungrouped path remains on-device.
#[derive(Clone, Copy, Debug)]
pub struct GroupedRouter {
    pub groups: usize,
    pub selected_groups: usize,
    pub top_k: usize,
    pub normalize: bool,
    pub scaling: u32,
}
impl GroupedRouter {
    fn scale(&self) -> f32 {
        f32::from_bits(self.scaling)
    }
    pub fn select(&self, lg: &[f32], bias: &[f32]) -> anyhow::Result<Vec<(usize, f32)>> {
        let e = lg.len();
        anyhow::ensure!(
            self.groups > 0
                && e > 0
                && e % self.groups == 0
                && bias.len() == e
                && self.selected_groups > 0
                && self.selected_groups <= self.groups
                && self.top_k > 0
                && self.top_k <= self.selected_groups * (e / self.groups),
            "invalid grouped router shape"
        );
        let sel_sc: Vec<f32> = lg
            .iter()
            .zip(bias.iter())
            .map(|(l, b)| 1.0 / (1.0 + (-l).exp()) + b)
            .collect();
        let per_group = e / self.groups;
        let mut group_sc: Vec<(f32, usize)> = Vec::with_capacity(self.groups);
        for g in 0..self.groups {
            let lo = g * per_group;
            let mut top = [0f32; 2];
            for si in &sel_sc[lo..lo + per_group] {
                if *si > top[0] {
                    top[1] = top[0];
                    top[0] = *si;
                } else if *si > top[1] {
                    top[1] = *si;
                }
            }
            group_sc.push((top[0] + top[1], g));
        }
        group_sc.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        let mut allowed = vec![false; e];
        for (_, g) in group_sc.iter().take(self.selected_groups) {
            for i in (g * per_group)..((g + 1) * per_group) {
                allowed[i] = true;
            }
        }
        let mut candidates: Vec<(f32, usize)> = (0..e)
            .filter(|&i| allowed[i])
            .map(|i| (sel_sc[i], i))
            .collect();
        candidates.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        let chosen: Vec<usize> = candidates
            .iter()
            .take(self.top_k)
            .map(|(_, i)| *i)
            .collect();
        let mut route_weights: Vec<f32> = chosen
            .iter()
            .map(|&i| 1.0 / (1.0 + (-lg[i]).exp()))
            .collect();
        if self.normalize {
            let sum: f32 = route_weights.iter().sum();
            if sum > 0.0 {
                for weight in &mut route_weights {
                    *weight /= sum;
                }
            }
        }
        for weight in &mut route_weights {
            *weight *= self.scale();
        }

        Ok(chosen.into_iter().zip(route_weights).collect())
    }
}
