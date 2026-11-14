//! KDA recurrent dataflow. The engine binds tensors and persists the returned
//! state; individual device primitives never choose a model execution mode.
use crate::graph::{Builder, Graph, Value as V};
use lisa_backend::{Kernel as K, ScalarType as D, SparseF32 as S, TensorOperation as T};
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Spec {
    pub batch: i32,
    pub tokens: i32,
    pub heads: i32,
    pub dimension: i32,
    pub lower_bound: u32,
    pub epsilon: u32,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stage {
    Decode,
    Cohort,
    Prepare,
    Recurrence,
    Output,
    NaiveConv {
        has_conv: bool,
    },
    NaiveGate {
        projection: super::packed::Projection,
    },
    NaiveRecurrence {
        state_rank: usize,
        capture: bool,
    },
    NaiveOutput,
    Beta,
    InitialState {
        present: bool,
    },
    Inputs {
        cohort: bool,
        conv: bool,
        state_rank: usize,
    },
    CommitDecode,
    CommitConv,
    StateContiguous,
    ReshapeOutput,
    CacheRow {
        conv: bool,
        ssm: bool,
    },
    CacheSplit {
        row: i32,
    },
    Join {
        count: usize,
    },
    Weight {
        rows: i32,
        columns: i32,
        cast: bool,
    },
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    Decode,
    Cohort,
    Prefill,
    Diagnostic,
}
pub fn mode(
    batch: usize,
    tokens: usize,
    prior_state: bool,
    prefill: bool,
    diagnostic: bool,
) -> Mode {
    if diagnostic {
        Mode::Diagnostic
    } else if batch > 1 && tokens == 1 && !prefill {
        Mode::Cohort
    } else if batch == 1 && tokens == 1 && prior_state && !prefill {
        Mode::Decode
    } else if batch == 1 {
        Mode::Prefill
    } else {
        Mode::Diagnostic
    }
}
fn op(b: &mut Builder, t: T, x: V) -> V {
    b.one(K::Tensor(t), vec![x])
}
fn reshape(b: &mut Builder, x: V, s: &[i32]) -> V {
    op(b, T::Reshape(s.to_vec()), x)
}
fn cast(b: &mut Builder, x: V) -> V {
    op(b, T::Cast(D::F32), x)
}
fn ready(b: &mut Builder, x: V) -> V {
    let x = cast(b, x);
    op(b, T::Contiguous, x)
}
fn narrow(b: &mut Builder, x: V, axis: i32, start: i32, len: i32) -> V {
    op(
        b,
        T::Slice {
            axis,
            start,
            end: start + len,
        },
        x,
    )
}
fn concat(b: &mut Builder, v: Vec<V>, axis: i32) -> V {
    b.one(
        K::Tensor(T::Concatenate {
            axis,
            inputs: v.len(),
        }),
        v,
    )
}
impl Spec {
    pub fn graph(self, stage: Stage) -> anyhow::Result<Graph> {
        anyhow::ensure!(
            self.batch > 0 && self.tokens > 0 && self.heads > 0 && self.dimension == 128,
            "invalid KDA geometry"
        );
        anyhow::ensure!(
            self.heads
                .checked_mul(self.dimension)
                .and_then(|n| n.checked_mul(9))
                .is_some(),
            "KDA state overflow"
        );
        match stage {
            Stage::NaiveConv { has_conv } => return self.naive_conv(has_conv),
            Stage::NaiveGate { projection } => return self.naive_gate(projection),
            Stage::NaiveRecurrence {
                state_rank,
                capture,
            } => return self.naive_recurrence(state_rank, capture),
            Stage::NaiveOutput => return self.naive_output(),
            Stage::Beta => return self.beta(),
            Stage::Decode | Stage::Cohort | Stage::Prepare | Stage::Recurrence | Stage::Output => {}
            _ => return self.storage(stage),
        }
        let Self {
            batch,
            tokens,
            heads,
            dimension,
            lower_bound,
            epsilon,
        } = self;
        anyhow::ensure!(
            batch > 0 && tokens > 0 && heads > 0 && dimension == 128,
            "invalid KDA geometry"
        );
        let n = heads
            .checked_mul(dimension)
            .ok_or_else(|| anyhow::anyhow!("KDA width overflow"))?;
        anyhow::ensure!(
            n.checked_mul(9).is_some() && n.checked_mul(dimension).is_some(),
            "KDA state overflow"
        );
        let decode = S::DeltaDecode {
            heads: heads as usize,
            lower_bound,
            epsilon,
        };
        let inputs = match stage {
            Stage::Decode | Stage::Cohort => 14,
            Stage::Prepare => 10,
            Stage::Recurrence => 6,
            Stage::Output => 3,
            _ => unreachable!(),
        };
        let mut b = Builder::new(inputs, 0, 0);
        let out = match stage {
            Stage::Decode => {
                // Input rows and initial states have already been materialized
                // by the caller's preceding state-binding stage.
                let mut a = (0..14).map(V::Input).collect::<Vec<_>>();
                for i in 6..12 {
                    a[i] = ready(&mut b, a[i]);
                }
                a[4] = ready(&mut b, a[4]);
                a[5] = ready(&mut b, a[5]);
                b.call(K::SparseF32(decode), a)
            }
            Stage::Cohort => {
                anyhow::ensure!(tokens == 1, "cohort recurrence expects one token per row");
                let mut a = (0..14).map(V::Input).collect::<Vec<_>>();
                // Projection/state preparation precedes this graph. Preserve
                // weight casts before the row-local fused recurrence sequence.
                for i in 6..12 {
                    a[i] = ready(&mut b, a[i]);
                }
                let (mut ys, mut cs, mut ss) = (vec![], vec![], vec![]);
                for row in 0..batch {
                    let mut r = a.clone();
                    for i in 0..6 {
                        r[i] = narrow(&mut b, r[i], 0, row, 1);
                        r[i] = reshape(&mut b, r[i], &[if i == 4 { heads } else { n }]);
                    }
                    r[12] = narrow(&mut b, a[12], 0, row, 1);
                    r[12] = reshape(&mut b, r[12], &[3, heads, 3 * dimension]);
                    r[12] = op(&mut b, T::Contiguous, r[12]);
                    r[13] = narrow(&mut b, a[13], 0, row, 1);
                    r[13] = reshape(&mut b, r[13], &[heads, dimension, dimension]);
                    r[13] = op(&mut b, T::Contiguous, r[13]);
                    let out = b.call(K::SparseF32(decode.clone()), r);
                    ys.push(reshape(&mut b, out[0], &[1, 1, n]));
                    cs.push(reshape(&mut b, out[1], &[1, 3, 3 * n]));
                    ss.push(reshape(&mut b, out[2], &[1, heads, dimension, dimension]));
                }
                vec![
                    concat(&mut b, ys, 0),
                    concat(&mut b, cs, 0),
                    concat(&mut b, ss, 0),
                ]
            }
            Stage::Prepare => {
                let mut a = (0..10).map(V::Input).collect::<Vec<_>>();
                for i in 4..9 {
                    a[i] = cast(&mut b, a[i]);
                }
                a[8] = reshape(&mut b, a[8], &[n]);
                b.call(
                    K::SparseF32(S::DeltaPrepare {
                        heads: heads as usize,
                        lower_bound,
                        epsilon,
                    }),
                    a,
                )
            }
            Stage::Recurrence => {
                let mut a = (0..6).map(V::Input).collect::<Vec<_>>();
                for i in 0..4 {
                    a[i] = reshape(&mut b, a[i], &[batch, tokens, heads, dimension]);
                }
                a[4] = cast(&mut b, a[4]);
                a[4] = reshape(&mut b, a[4], &[batch, tokens, heads]);
                b.call(K::SparseF32(S::DeltaRecurrence { capture: false }), a)
            }
            Stage::Output => {
                let gate = reshape(&mut b, V::Input(1), &[batch, tokens, heads, dimension]);
                let y = reshape(&mut b, V::Input(0), &[tokens, n]);
                let gate = reshape(&mut b, gate, &[tokens, n]);
                let weight = cast(&mut b, V::Input(2));
                b.call(
                    K::SparseF32(S::DeltaOutput {
                        heads: heads as usize,
                        epsilon,
                    }),
                    vec![y, gate, weight],
                )
            }
            _ => unreachable!(),
        };
        let graph = b.finish(out);
        graph.validate().map_err(anyhow::Error::msg)?;
        Ok(graph)
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn recurrent_modes_preserve_prefill_tail_and_row_local_decode() {
        assert_eq!(mode(1, 1, true, false, false), Mode::Decode);
        assert_eq!(mode(1, 1, false, false, false), Mode::Prefill);
        assert_eq!(mode(1, 1, true, true, false), Mode::Prefill);
        assert_eq!(mode(2, 1, false, false, false), Mode::Cohort);
        assert_eq!(mode(2, 2, true, false, false), Mode::Diagnostic);
        assert_eq!(mode(1, 1, true, false, true), Mode::Diagnostic);
    }
}
use crate::program::Build;
use lisa_backend::{Binary as B, Unary as U};
fn zero(p: &mut Build<'_>, shape: &[i32]) -> V {
    // Legacy full(0) is a broadcast followed by a materialization. A zeros
    // allocation would silently change its dispatch sequence.
    let z = p.scalar(0., D::F32);
    let z = p.op(T::Broadcast(shape.to_vec()), vec![z]);
    p.contiguous(z)
}
impl Spec {
    fn beta(self) -> anyhow::Result<Graph> {
        let schema = Default::default();
        let mut p = Build::new(1, &schema);
        let x = p.cast(V::Input(0), D::F32);
        let x = p.unary(U::Sigmoid, x);
        let y = p.reshape(x, &[self.batch, self.tokens, self.heads, 1]);
        Ok(p.finish(vec![y])?.graph)
    }
    fn naive_conv(self, has_conv: bool) -> anyhow::Result<Graph> {
        let (b, t, h, d) = (self.batch, self.tokens, self.heads, self.dimension);
        let n = h * d;
        let schema = Default::default();
        let mut p = Build::new(6 + usize::from(has_conv), &schema);
        let window = if has_conv {
            let a = p.reshape(V::Input(6), &[b, 3, 3 * n]);
            let q = p.slice(a, 2, 0, n);
            let k = p.slice(a, 2, n, 2 * n);
            let v = p.slice(a, 2, 2 * n, 3 * n);
            p.concat(vec![q, k, v], 2)
        } else {
            zero(&mut p, &[b, 3, 3 * n])
        };
        let mixed = p.concat(vec![V::Input(0), V::Input(1), V::Input(2)], 2);
        let inp = p.concat(vec![window, mixed], 1);
        let nw = p.slice(inp, 1, t, t + 3);
        let nw = p.contiguous(nw);
        let q = p.slice(nw, 2, 0, n);
        let k = p.slice(nw, 2, n, 2 * n);
        let v = p.slice(nw, 2, 2 * n, 3 * n);
        let nc = p.concat(vec![q, k, v], 1);
        let nc = p.reshape(nc, &[b, 3, 3 * n]);
        let mut branches = vec![];
        for branch in 0..3 {
            let mut acc = None;
            for tap in 0..4 {
                let w = p.slice(V::Input(3 + branch), 1, tap, tap + 1);
                let w = p.reshape(w, &[1, 1, n]);
                let col = p.slice(inp, 1, tap, tap + t);
                let col = p.slice(col, 2, branch as i32 * n, (branch as i32 + 1) * n);
                let w = p.op(T::Broadcast(vec![b, t, n]), vec![w]);
                let term = p.binary(B::Multiply, col, w);
                acc = Some(match acc {
                    Some(a) => p.binary(B::Add, a, term),
                    None => term,
                });
            }
            branches.push(p.unary(U::Silu, acc.unwrap()));
        }
        let q = p.reshape(branches[0], &[b, t, h, d]);
        let k = p.reshape(branches[1], &[b, t, h, d]);
        let v = p.reshape(branches[2], &[b, t, h, d]);
        let q = p.cast(q, D::F32);
        let q = p.contiguous(q);
        let q = p
            .graph
            .append(super::math::l2(1e-6, D::F32)?, &[q], &[], &[])
            .map_err(anyhow::Error::msg)?[0];
        let k = p.cast(k, D::F32);
        let k = p.contiguous(k);
        let k = p
            .graph
            .append(super::math::l2(1e-6, D::F32)?, &[k], &[], &[])
            .map_err(anyhow::Error::msg)?[0];
        let scale = p.scalar((d as f64).powf(-0.5) as f32, D::F32);
        let q = p.binary(B::Multiply, q, scale);
        let v = p.cast(v, D::F32);
        let v = p.contiguous(v);
        Ok(p.finish(vec![q, k, v, nc, window])?.graph)
    }
    fn naive_gate(self, projection: super::packed::Projection) -> anyhow::Result<Graph> {
        let (b, t, h, d) = (self.batch, self.tokens, self.heads, self.dimension);
        let schema = Default::default();
        let mut p = Build::new(4, &schema);
        let fg = p
            .graph
            .append(
                super::kda_projection::one(projection, &[b, t, projection.input])?,
                &[V::Input(1), V::Input(0)],
                &[],
                &[],
            )
            .map_err(anyhow::Error::msg)?[0];
        let fg = p.reshape(fg, &[b, t, h, d]);
        let fg = p.cast(fg, D::F32);
        let dt = p.cast(V::Input(2), D::F32);
        let dt = p.reshape(dt, &[1, 1, h, d]);
        let dt = p.op(T::Broadcast(vec![b, t, h, d]), vec![dt]);
        let fg = p.binary(B::Add, fg, dt);
        let decay = p.cast(V::Input(3), D::F32);
        let decay = p.unary(U::Exp, decay);
        let decay = p.reshape(decay, &[1, 1, h, 1]);
        let decay = p.op(T::Broadcast(vec![b, t, h, d]), vec![decay]);
        let g = p.binary(B::Multiply, fg, decay);
        let g = p.unary(U::Sigmoid, g);
        let lb = p.scalar(f32::from_bits(self.lower_bound), D::F32);
        let g = p.binary(B::Multiply, g, lb);
        let g = p.unary(U::Exp, g);
        Ok(p.finish(vec![g])?.graph)
    }
    fn naive_recurrence(self, state_rank: usize, capture: bool) -> anyhow::Result<Graph> {
        let (b, t, h, d) = (self.batch, self.tokens, self.heads, self.dimension);
        let bh = b * h;
        let schema = Default::default();
        let mut p = Build::new(5 + usize::from(state_rank > 0), &schema);
        let mut state = if state_rank > 0 {
            let s = if state_rank == 5 {
                p.reshape(V::Input(5), &[b, h, d, d])
            } else {
                V::Input(5)
            };
            p.transpose(s, &[0, 1, 3, 2])
        } else {
            zero(&mut p, &[b, h, d, d])
        };
        let initial = state;
        let mut outs = vec![];
        let mut captures = vec![];
        for ti in 0..t {
            let gt = p.slice(V::Input(3), 1, ti, ti + 1);
            let gt = p.contiguous(gt);
            let s = p.reshape(state, &[bh, d, d]);
            let g = p.reshape(gt, &[bh, d, 1]);
            let s = p.binary(B::Multiply, s, g);
            state = p.reshape(s, &[b, 1, h, d, d]);
            let k = p.slice(V::Input(1), 1, ti, ti + 1);
            let k = p.contiguous(k);
            let v = p.slice(V::Input(2), 1, ti, ti + 1);
            let v = p.contiguous(v);
            let q = p.slice(V::Input(0), 1, ti, ti + 1);
            let q = p.contiguous(q);
            let beta = p.slice(V::Input(4), 1, ti, ti + 1);
            let beta = p.contiguous(beta);
            let s = p.transpose(state, &[0, 1, 2, 4, 3]);
            let s = p.contiguous(s);
            let s = p.reshape(s, &[bh, d, d]);
            let kt = p.reshape(k, &[bh, d, 1]);
            let mem = p.op(T::Matmul, vec![s, kt]);
            let mem = p.reshape(mem, &[b, 1, h, d]);
            let delta = p.binary(B::Subtract, v, mem);
            let delta = p.binary(B::Multiply, delta, beta);
            let k = p.reshape(k, &[b, 1, h, d, 1]);
            let delta = p.reshape(delta, &[b, 1, h, 1, d]);
            let upd = p.binary(B::Multiply, k, delta);
            let upd = p.reshape(upd, &[b, 1, h, d, d]);
            state = p.binary(B::Add, state, upd);
            let s = p.transpose(state, &[0, 1, 2, 4, 3]);
            let s = p.contiguous(s);
            let s = p.reshape(s, &[bh, d, d]);
            let q = p.reshape(q, &[bh, d, 1]);
            let y = p.op(T::Matmul, vec![s, q]);
            let y = p.reshape(y, &[b, 1, h, d]);
            if capture && ti < 2 {
                captures.push(y);
                if ti == 0 {
                    captures.push(state);
                }
            }
            outs.push(y);
        }
        let last = *outs.last().unwrap();
        let y = p.concat(outs, 1);
        let state = p.transpose(state, &[0, 1, 2, 4, 3]);
        let state = p.reshape(state, &[b, h, d, d]);
        let mut outputs = vec![y, state];
        if capture {
            outputs.extend([initial, last]);
            outputs.extend(captures);
        }
        Ok(p.finish(outputs)?.graph)
    }
    fn naive_output(self) -> anyhow::Result<Graph> {
        let (b, t, h, d) = (self.batch, self.tokens, self.heads, self.dimension);
        let schema = Default::default();
        let mut p = Build::new(3, &schema);
        let gate = p.reshape(V::Input(1), &[b, t, h, d]);
        let out = p
            .graph
            .append(
                super::math::rms(4, f32::from_bits(self.epsilon), true, true)?,
                &[V::Input(0), V::Input(2), gate],
                &[],
                &[],
            )
            .map_err(anyhow::Error::msg)?[0];
        let out = p.reshape(out, &[b, t, h * d]);
        Ok(p.finish(vec![out])?.graph)
    }
}
impl Spec {
    fn storage(self, stage: Stage) -> anyhow::Result<Graph> {
        let (batch, t, h, d) = (self.batch, self.tokens, self.heads, self.dimension);
        let n = h * d;
        let schema = Default::default();
        let inputs = match stage {
            Stage::InitialState { present } => usize::from(present),
            Stage::Inputs { cohort, conv, .. } => (if cohort { 7 } else { 5 }) + usize::from(conv),
            Stage::CommitDecode | Stage::CacheSplit { .. } => 2,
            Stage::CacheRow { conv, ssm } => usize::from(conv) + usize::from(ssm),
            Stage::Join { count } => count,
            _ => 1,
        };
        let mut p = Build::new(inputs, &schema);
        let out = match stage {
            Stage::InitialState { present } => vec![if present {
                V::Input(0)
            } else {
                zero(&mut p, &[batch, h, d, d])
            }],
            Stage::Inputs {
                cohort,
                conv,
                state_rank,
            } => {
                let count = if cohort { 6 } else { 4 };
                let mut a = vec![];
                for i in 0..count {
                    let x = p.cast(V::Input(i), D::F32);
                    a.push(p.contiguous(x));
                }
                let shape = if cohort {
                    vec![batch, 3, 3 * n]
                } else {
                    vec![3, h, 3 * d]
                };
                let c = if conv {
                    let c = p.cast(V::Input(count + 1), D::F32);
                    let c = p.reshape(c, &shape);
                    if cohort { p.contiguous(c) } else { c }
                } else {
                    zero(&mut p, &shape)
                };
                let state = if cohort || state_rank == 5 {
                    p.reshape(V::Input(count), &[batch, h, d, d])
                } else {
                    V::Input(count)
                };
                let state = p.contiguous(state);
                a.extend([c, state]);
                a
            }
            Stage::CommitDecode => vec![
                p.reshape(V::Input(0), &[1, 3, 3 * n]),
                p.reshape(V::Input(1), &[1, h, d, d]),
            ],
            Stage::CommitConv => {
                let c = p.reshape(V::Input(0), &[1, 3, 3 * n]);
                vec![p.cast(c, D::F32)]
            }
            Stage::StateContiguous => vec![p.contiguous(V::Input(0))],
            Stage::ReshapeOutput => vec![p.reshape(V::Input(0), &[batch, t, n])],
            Stage::CacheRow { conv, ssm } => {
                let mut index = 0;
                let mut out = vec![];
                for (present, shape) in [(conv, vec![1, 3, 3 * n]), (ssm, vec![1, h, d, d])] {
                    out.push(if present {
                        let x = p.cast(V::Input(index), D::F32);
                        index += 1;
                        let x = p.reshape(x, &shape);
                        p.contiguous(x)
                    } else {
                        p.graph.one(
                            K::Zeros {
                                shape,
                                dtype: D::F32,
                            },
                            vec![],
                        )
                    });
                }
                out
            }
            Stage::CacheSplit { row } => {
                let mut out = vec![];
                for (i, shape) in [vec![1, 3, 3 * n], vec![1, h, d, d]]
                    .into_iter()
                    .enumerate()
                {
                    let x = p.slice(V::Input(i), 0, row, row + 1);
                    let x = p.reshape(x, &shape);
                    out.push(p.contiguous(x));
                }
                out
            }
            Stage::Join { count } => vec![p.concat((0..count).map(V::Input).collect(), 0)],
            Stage::Weight {
                rows,
                columns,
                cast,
            } => {
                let x = if rows > 0 {
                    p.reshape(V::Input(0), &[rows, columns])
                } else {
                    V::Input(0)
                };
                vec![if cast { p.cast(x, D::F32) } else { x }]
            }
            _ => unreachable!(),
        };
        Ok(p.finish(out)?.graph)
    }
}
/// Tensor identifiers are checkpoint contracts, independent of reader APIs.
pub const MLX_WEIGHTS: [&str; 15] = [
    "q_proj.weight",
    "k_proj.weight",
    "v_proj.weight",
    "o_proj.weight",
    "f_a_proj.weight",
    "f_b_proj.weight",
    "g_a_proj.weight",
    "g_b_proj.weight",
    "b_proj.weight",
    "q_conv1d.weight",
    "k_conv1d.weight",
    "v_conv1d.weight",
    "dt_bias",
    "A_log",
    "o_norm.weight",
];
pub const GGUF_WEIGHTS: [&str; 15] = [
    "kda_q",
    "kda_k",
    "kda_v",
    "kda_output",
    "kda_f_a",
    "kda_f_b",
    "kda_g_a",
    "kda_g_b",
    "kda_beta",
    "kda_q_conv",
    "kda_k_conv",
    "kda_v_conv",
    "kda_dt_bias",
    "kda_a_log",
    "kda_o_norm",
];
pub const GGUF_SPEC: Spec = Spec {
    batch: 1,
    tokens: 1,
    heads: 64,
    dimension: 128,
    lower_bound: (-5.0f32).to_bits(),
    epsilon: (1e-5f32).to_bits(),
};
pub const CONVOLUTION_TAPS: usize = 4;
/// Full recurrent block recipe. Optional observations expose intermediate
/// values without making the model depend on a capture file or a device.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Forward {
    pub spec: Spec,
    pub projections: [super::packed::Projection; 9],
    pub conv: bool,
    pub state_rank: usize,
    pub prefill: bool,
    pub diagnostic: bool,
    pub fused: bool,
    pub triple: bool,
    pub paired: bool,
    pub groups: Option<usize>,
    pub observe: bool,
    pub capture: bool,
    pub capture_prefill: bool,
    pub capture_input: bool,
}
pub struct ForwardPlan {
    pub graph: Graph,
    pub observations: Vec<String>,
}
fn append(b: &mut Builder, graph: Graph, inputs: &[V]) -> anyhow::Result<Vec<V>> {
    b.append(graph, inputs, &[], &[])
        .map_err(anyhow::Error::msg)
}
impl Forward {
    pub fn plan(&self) -> anyhow::Result<ForwardPlan> {
        let spec = self.spec;
        let (batch, t, h, d) = (spec.batch, spec.tokens, spec.heads, spec.dimension);
        let n = h * d;
        let w = |i| V::Input(i + 1);
        let conv = self.conv.then_some(V::Input(16));
        let state = (self.state_rank > 0).then_some(V::Input(16 + usize::from(self.conv)));
        let mut b = Builder::new(
            16 + usize::from(self.conv) + usize::from(state.is_some()),
            0,
            0,
        );
        let mut observations = vec![];
        let mut values = vec![];
        let mut record = |name: &str, value: V| {
            if (self.observe && name.starts_with("@")) || (self.capture && !name.starts_with("@")) {
                observations.push(name.to_string());
                values.push(value);
            }
        };
        let shape = [batch, t, self.projections[0].input];
        let ix = [0, 1, 2, 4, 6, 8];
        let p = ix.map(|i| self.projections[i]);
        let mut args = vec![V::Input(0)];
        args.extend(ix.map(w));
        let a = append(
            &mut b,
            super::kda_projection::inputs(p, &shape, self.fused, self.triple, self.groups)?,
            &args,
        )?;
        let (q, k, v, fa, ga, beta) = (a[0], a[1], a[2], a[3], a[4], a[5]);
        let g = append(
            &mut b,
            super::kda_projection::gates(
                [self.projections[5], self.projections[7]],
                &[batch, t, self.projections[4].output],
                &[batch, t, self.projections[6].output],
                true,
                self.paired,
            )?,
            &[fa, ga, w(5), w(7)],
        )?;
        let (raw_gate, output_gate) = (g[0], g[1]);
        if !self.capture_prefill || self.capture_input {
            record("kda_input", V::Input(0));
        }
        for (name, value) in [
            ("kda_q_proj", q),
            ("kda_k_proj", k),
            ("kda_v_proj", v),
            ("kda_beta_logits", beta),
            ("kda_gate_raw", raw_gate),
            ("kda_gb_proj", output_gate),
        ] {
            record(name, value);
        }
        if !self.capture_prefill {
            record("kda_fa_proj", fa);
            record("kda_ga_proj", ga);
        }
        // Diagnostic observations are separately named to preserve their raw
        // projection values even when the public prefill capture is filtered.
        for (name, value) in [
            ("@input", V::Input(0)),
            ("@q_proj", q),
            ("@k_proj", k),
            ("@v_proj", v),
            ("@fa_proj", fa),
            ("@ga_proj", ga),
            ("@beta_proj", beta),
            ("@raw_gate", raw_gate),
            ("@output_gate", output_gate),
        ] {
            record(name, value);
        }
        let initial = append(
            &mut b,
            spec.graph(Stage::InitialState {
                present: state.is_some(),
            })?,
            &state.into_iter().collect::<Vec<_>>(),
        )?[0];
        let mode = mode(
            batch as usize,
            t as usize,
            state.is_some(),
            self.prefill,
            self.diagnostic,
        );
        let (out, next_conv, next_state) = match mode {
            Mode::Decode | Mode::Prefill | Mode::Cohort => {
                let cohort = mode == Mode::Cohort;
                let mut a = vec![q, k, v, raw_gate];
                if cohort {
                    a.extend([beta, output_gate]);
                }
                a.push(initial);
                a.extend(conv);
                let prepared = append(
                    &mut b,
                    spec.graph(Stage::Inputs {
                        cohort,
                        conv: self.conv,
                        state_rank: if state.is_some() { self.state_rank } else { 4 },
                    })?,
                    &a,
                )?;
                let (c, s) = if cohort {
                    (prepared[6], prepared[7])
                } else {
                    (prepared[4], prepared[5])
                };
                record("kda_initial_conv", c);
                record("kda_initial_ssm", s);
                record("@conv_in", c);
                record("@state_in", s);
                // These observations serve the explicit binary input dump.
                for (name, value) in [
                    ("@dump_q", prepared[0]),
                    ("@dump_k", prepared[1]),
                    ("@dump_v", prepared[2]),
                    ("@dump_gate", prepared[3]),
                ] {
                    record(name, value);
                }
                if mode != Mode::Prefill {
                    let a = if cohort {
                        vec![
                            prepared[0],
                            prepared[1],
                            prepared[2],
                            prepared[3],
                            prepared[4],
                            prepared[5],
                            w(9),
                            w(10),
                            w(11),
                            w(13),
                            w(12),
                            w(14),
                            c,
                            s,
                        ]
                    } else {
                        vec![
                            prepared[0],
                            prepared[1],
                            prepared[2],
                            prepared[3],
                            beta,
                            output_gate,
                            w(9),
                            w(10),
                            w(11),
                            w(13),
                            w(12),
                            w(14),
                            c,
                            s,
                        ]
                    };
                    let out = append(
                        &mut b,
                        spec.graph(if cohort { Stage::Cohort } else { Stage::Decode })?,
                        &a,
                    )?;
                    record("kda_decode_out", out[0]);
                    record("kda_decode_conv_state", out[1]);
                    record("kda_decode_state", out[2]);
                    record("@decode_out", out[0]);
                    record("@conv_out", out[1]);
                    record("@state_out", out[2]);
                    if cohort {
                        let state =
                            append(&mut b, spec.graph(Stage::StateContiguous)?, &[out[2]])?[0];
                        (out[0], out[1], state)
                    } else {
                        let states = append(&mut b, spec.graph(Stage::CommitDecode)?, &out[1..])?;
                        let y = append(&mut b, spec.graph(Stage::ReshapeOutput)?, &[out[0]])?[0];
                        (y, states[0], states[1])
                    }
                } else {
                    let a = append(
                        &mut b,
                        spec.graph(Stage::Prepare)?,
                        &[
                            prepared[0],
                            prepared[1],
                            prepared[2],
                            prepared[3],
                            w(9),
                            w(10),
                            w(11),
                            w(13),
                            w(12),
                            c,
                        ],
                    )?;
                    let nc = append(&mut b, spec.graph(Stage::CommitConv)?, &[a[4]])?[0];
                    for (name, value) in [
                        ("@q_conv", a[0]),
                        ("@k_conv", a[1]),
                        ("@v_conv", a[2]),
                        ("@g", a[3]),
                        ("@conv_out", a[4]),
                    ] {
                        record(name, value);
                    }
                    if self.capture && !self.capture_prefill {
                        for (name, value) in [
                            ("kda_qc", a[0]),
                            ("kda_kc", a[1]),
                            ("kda_vc", a[2]),
                            ("kda_g", a[3]),
                        ] {
                            record(name, value);
                        }
                        let beta = append(&mut b, spec.graph(Stage::Beta)?, &[beta])?[0];
                        record("kda_beta", beta);
                    }
                    record("kda_conv_state", a[4]);
                    let y = append(
                        &mut b,
                        spec.graph(Stage::Recurrence)?,
                        &[a[0], a[1], a[2], a[3], beta, s],
                    )?;
                    record("@recurrence_y", y[0]);
                    record("@state_out", y[1]);
                    if !self.capture_prefill {
                        record("fused_y", y[0]);
                    }
                    record("fused_state", y[1]);
                    let ns = append(&mut b, spec.graph(Stage::StateContiguous)?, &[y[1]])?[0];
                    let out = append(
                        &mut b,
                        spec.graph(Stage::Output)?,
                        &[y[0], output_gate, w(14)],
                    )?[0];
                    record("@decode_out", out);
                    record("kda_outnorm", out);
                    if t == 1 {
                        record("dec_outnorm", out);
                        record("dec_y_s{step}", y[0]);
                    }
                    let out = append(&mut b, spec.graph(Stage::ReshapeOutput)?, &[out])?[0];
                    (out, nc, ns)
                }
            }
            Mode::Diagnostic => {
                let beta = append(&mut b, spec.graph(Stage::Beta)?, &[beta])?[0];
                let mut args = vec![q, k, v, w(9), w(10), w(11)];
                args.extend(conv);
                let a = append(
                    &mut b,
                    spec.graph(Stage::NaiveConv {
                        has_conv: self.conv,
                    })?,
                    &args,
                )?;
                let g = append(
                    &mut b,
                    spec.graph(Stage::NaiveGate {
                        projection: self.projections[5],
                    })?,
                    &[fa, w(5), w(12), w(13)],
                )?[0];
                record("kda_g", g);
                record("kda_beta", beta);
                let mut args = vec![a[0], a[1], a[2], g, beta];
                args.extend(state);
                let y = append(
                    &mut b,
                    spec.graph(Stage::NaiveRecurrence {
                        state_rank: self.state_rank,
                        capture: self.observe || self.capture,
                    })?,
                    &args,
                )?;
                if self.observe || self.capture {
                    if t == 1 {
                        record("dec_q_s{step}", a[0]);
                        record("dec_g_s{step}", g);
                        record("dec_state_in_s{step}", y[2]);
                        record("dec_y_s{step}", y[3]);
                        record("dec_conv_s{step}", a[4]);
                    }
                    record("kda_tok0", y[4]);
                    record("kda_state_t0", y[5]);
                    if t > 1 {
                        record("kda_tok1", y[6]);
                    }
                }
                record("kda_out_prenorm", y[0]);
                let out = append(
                    &mut b,
                    spec.graph(Stage::NaiveOutput)?,
                    &[y[0], output_gate, w(14)],
                )?[0];
                (out, a[3], y[1])
            }
        };
        let out = append(
            &mut b,
            super::kda_projection::one(self.projections[3], &[batch, t, n])?,
            &[w(3), out],
        )?[0];
        record("@o_proj", out);
        if mode == Mode::Cohort {
            record("kda_o_proj", out);
        }
        let mut outputs = vec![out, next_conv, next_state];
        outputs.extend(values);
        Ok(ForwardPlan {
            graph: b.finish(outputs),
            observations,
        })
    }
}
