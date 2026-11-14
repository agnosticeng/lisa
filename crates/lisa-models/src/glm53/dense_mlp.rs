//! Dense/shared affine expert plans and original fusion eligibility.
use super::affine_projection::Spec;
use crate::{
    graph::{Graph, Value as V},
    program::Build,
};
use lisa_backend::{Kernel as K, ScalarType as D, SparseF32 as S};
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Request {
    JoinWeights,
    SplitWeights {
        inter: i32,
    },
    SplitOutput {
        rank: usize,
        inter: i32,
    },
    SwiGlu {
        limit: u32,
    },
    CaptureJoin,
    Projections {
        gate: Spec,
        up: Spec,
        paired: Option<Spec>,
    },
    Activated {
        gate: Spec,
        up: Spec,
        paired: Option<Spec>,
        native: bool,
        capture: bool,
        limit: u32,
    },
}
fn projection(p: &mut Build<'_>, spec: &Spec, x: V, start: usize) -> anyhow::Result<V> {
    Ok(p.graph
        .append(
            spec.graph()?,
            &[x, V::Input(start), V::Input(start + 1), V::Input(start + 2)],
            &[],
            &[],
        )
        .map_err(anyhow::Error::msg)?[0])
}
fn split(p: &mut Build<'_>, y: V, rank: usize, inter: i32) -> [V; 2] {
    [
        p.slice(y, (rank - 1) as i32, 0, inter),
        p.slice(y, (rank - 1) as i32, inter, 2 * inter),
    ]
}
fn projections(
    p: &mut Build<'_>,
    gate: &Spec,
    up: &Spec,
    paired: &Option<Spec>,
) -> anyhow::Result<[V; 2]> {
    anyhow::ensure!(
        gate.shape == up.shape && gate.weight[0] == up.weight[0],
        "gate/up geometry mismatch"
    );
    if let Some(pair) = paired {
        let y = projection(p, pair, V::Input(0), 7)?;
        Ok(split(p, y, gate.shape.len(), gate.weight[0]))
    } else {
        Ok([
            projection(p, gate, V::Input(0), 1)?,
            projection(p, up, V::Input(0), 4)?,
        ])
    }
}
fn swiglu(p: &mut Build<'_>, gate: V, up: V, limit: u32) -> V {
    let g = p.contiguous(gate);
    let u = p.contiguous(up);
    p.graph
        .one(K::SparseF32(S::ClampedSwiGlu { limit }), vec![g, u])
}
pub fn native_eligible(gate: &Spec, up: &Spec, paired: bool, enabled: bool) -> bool {
    if !paired || !enabled || gate.shape.is_empty() || gate.weight.is_empty() {
        return false;
    }
    let Some(rows) = gate.shape[..gate.shape.len() - 1]
        .iter()
        .try_fold(1i32, |a, &b| a.checked_mul(b))
    else {
        return false;
    };
    (1..=16).contains(&rows)
        && gate.shape.last().is_some_and(|k| k % 128 == 0)
        && gate.weight[0] % 2 == 0
        && gate.input_dtype == D::F32
        && [gate, up].iter().all(|w| w.bits == 8 && w.group_size == 64)
}
impl Request {
    pub fn graph(&self) -> anyhow::Result<Graph> {
        let inputs = match self {
            Self::JoinWeights => 6,
            Self::SplitWeights { .. } => 3,
            Self::SplitOutput { .. } => 1,
            Self::SwiGlu { .. } | Self::CaptureJoin => 2,
            _ => 10,
        };
        let schema = Default::default();
        let mut p = Build::new(inputs, &schema);
        let out = match self {
            Self::JoinWeights => (0..3)
                .map(|i| p.concat(vec![V::Input(i), V::Input(i + 3)], 0))
                .collect(),
            Self::SplitWeights { inter } => {
                let mut out = vec![];
                for start in [0, *inter] {
                    for i in 0..3 {
                        let x = p.slice(V::Input(i), 0, start, start + inter);
                        out.push(p.contiguous(x));
                    }
                }
                out
            }
            Self::SplitOutput { rank, inter } => {
                anyhow::ensure!(*rank > 0 && *inter > 0, "invalid fused output shape");
                split(&mut p, V::Input(0), *rank, *inter).to_vec()
            }
            Self::SwiGlu { limit } => vec![swiglu(&mut p, V::Input(1), V::Input(0), *limit)],
            Self::CaptureJoin => vec![p.concat(vec![V::Input(0), V::Input(1)], -1)],
            Self::Projections { gate, up, paired } => {
                projections(&mut p, gate, up, paired)?.to_vec()
            }
            Self::Activated {
                gate,
                up,
                paired,
                native,
                capture,
                limit,
            } => {
                if native_eligible(gate, up, paired.is_some(), *native) {
                    let rows = gate.shape[..gate.shape.len() - 1].iter().product::<i32>();
                    let k = *gate.shape.last().unwrap();
                    let x = p.reshape(V::Input(0), &[rows, k]);
                    let x = p.contiguous(x);
                    let result = p.graph.call(
                        K::SparseF32(S::AffineGateUp {
                            limit: *limit,
                            capture: *capture,
                        }),
                        [vec![x], (1..7).map(V::Input).collect()].concat(),
                    );
                    let mut shape = gate.shape.clone();
                    *shape.last_mut().unwrap() = gate.weight[0];
                    // Legacy capture reshapes G, U, activation in that order.
                    result.into_iter().map(|v| p.reshape(v, &shape)).collect()
                } else {
                    let [g, u] = projections(&mut p, gate, up, paired)?;
                    let y = swiglu(&mut p, g, u, *limit);
                    vec![g, u, y]
                }
            }
        };
        Ok(p.finish(out)?.graph)
    }
}
pub const PROJECTION_NAMES: [&str; 3] = ["gate_proj", "up_proj", "down_proj"];
pub const GROUP_SIZE: i32 = 64;
pub const BITS: i32 = 8;
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Step {
    CaptureInput,
    Activate,
    CaptureProjection,
    CaptureActivation,
    Down,
    CaptureOutput,
}
pub const FORWARD: [Step; 6] = [
    Step::CaptureInput,
    Step::Activate,
    Step::CaptureProjection,
    Step::CaptureActivation,
    Step::Down,
    Step::CaptureOutput,
];
pub fn capture_key(layer: usize, step: Step) -> String {
    format!(
        "L{layer}.{}",
        match step {
            Step::CaptureInput => "dense_mlp_input",
            Step::CaptureProjection => "dense_gate_up",
            Step::CaptureActivation => "dense_swiglu",
            Step::CaptureOutput => "dense_out",
            _ => unreachable!("only capture steps have capture keys"),
        }
    )
}
