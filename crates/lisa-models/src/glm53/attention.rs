//! MLX DSA input, cache-normalization, output and dense fallback dataflow.
use crate::{
    graph::{Graph, Value as V},
    program::Build,
};
use lisa_backend::{
    Binary as B, Kernel as K, ScalarType as D, SparseF32 as S, TensorOperation as T,
};
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Shape {
    pub batch: usize,
    pub tokens: usize,
    pub heads: usize,
    pub query: usize,
    pub value: usize,
    pub cache: usize,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Norm {
    Rows,
    Compact,
    Into(usize),
    Separate,
}
pub fn norm_order(
    batch: usize,
    tokens: usize,
    capture: bool,
    preserve: bool,
    tape: bool,
) -> Vec<Norm> {
    if batch > 1 && tokens == 1 {
        return vec![Norm::Rows, Norm::Separate];
    }
    if batch == 1 && tokens == 1 && (!capture || preserve) {
        let mut order = vec![];
        if tape {
            order.push(Norm::Into(0));
        }
        order.extend([Norm::Compact, Norm::Separate]);
        order
    } else {
        vec![Norm::Separate]
    }
}
pub fn custom_output(shape: &[i32], dtype: D, native: bool) -> bool {
    !native
        && shape.len() == 3
        && shape[2] == 16384
        && dtype == D::F32
        && !(1..=31).contains(&(shape[0] as usize * shape[1] as usize))
}
pub fn dense_limit(topk: usize, pool: usize) -> usize {
    topk + pool.saturating_sub(1)
}
pub fn flash(s: Shape, limit: usize, ragged: bool) -> bool {
    !ragged && s.batch == 1 && s.tokens >= 24 && s.cache <= limit
}
pub fn causal_ids(s: Shape) -> anyhow::Result<Vec<i32>> {
    anyhow::ensure!(
        s.tokens <= s.cache && s.cache <= i32::MAX as usize,
        "invalid causal selection span"
    );
    let start = s.cache - s.tokens;
    Ok((0..s.batch)
        .flat_map(|_| {
            (0..s.tokens)
                .flat_map(|q| (0..s.cache).map(move |p| if p <= start + q { p as i32 } else { -1 }))
        })
        .collect())
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Request {
    Contiguous,
    Inputs {
        latent: super::affine_projection::Spec,
        query: super::affine_projection::Spec,
    },
    MtpInputs {
        qa: super::affine_projection::Spec,
        qb: super::affine_projection::Spec,
        kv: super::affine_projection::Spec,
        shape: Shape,
        epsilon: u32,
    },
    Norm {
        kind: Norm,
        epsilon: u32,
    },
    Query(Shape),
    Output(Shape),
    FinalFallback(Shape),
    OutputProjection {
        shape: Vec<i32>,
        group: usize,
    },
    CastF32,
    SplitKv(Shape),
    DenseDecode {
        shape: Shape,
        scale: u32,
    },
    DenseTranspose,
    DenseOperands {
        shape: Shape,
        row: usize,
    },
    DenseScores {
        shape: Shape,
        scale: u32,
    },
    DenseValue {
        shape: Shape,
    },
    Join {
        count: usize,
        axis: i32,
    },
    BroadcastIds(Shape),
}
impl Request {
    pub fn graph(&self) -> anyhow::Result<Graph> {
        let inputs = match self {
            Self::Inputs { .. } => 7,
            Self::MtpInputs { .. } => 12,
            Self::Norm {
                kind: Norm::Into(_),
                ..
            } => 5,
            Self::Norm { .. } | Self::OutputProjection { .. } | Self::DenseDecode { .. } => 4,
            Self::DenseOperands { .. } => 4,
            Self::DenseScores { .. } => 3,
            Self::DenseValue { .. } => 2,
            Self::Join { count, .. } => *count,
            _ => 1,
        };
        let schema = Default::default();
        let mut p = Build::new(inputs, &schema);
        let out = match self {
            Self::Inputs { latent, query } => {
                let c = projection(&mut p, latent, V::Input(0), 1)?;
                let q = projection(&mut p, query, V::Input(0), 4)?;
                vec![c, q]
            }
            Self::MtpInputs {
                qa,
                qb,
                kv,
                shape: s,
                epsilon,
            } => {
                let raw_q = projection(&mut p, qa, V::Input(0), 1)?;
                let qr = p.graph.one(
                    K::SparseF32(S::WeightedRms {
                        epsilon: *epsilon,
                        latent: false,
                    }),
                    vec![raw_q, V::Input(10)],
                );
                let q = projection(&mut p, qb, qr, 4)?;
                let q = p.reshape(
                    q,
                    &[
                        s.batch as i32,
                        s.tokens as i32,
                        s.heads as i32,
                        s.query as i32,
                    ],
                );
                let q = p.transpose(q, &[0, 2, 1, 3]);
                let q = p.contiguous(q);
                let raw_c = projection(&mut p, kv, V::Input(0), 7)?;
                let c = p.graph.one(
                    K::SparseF32(S::WeightedRms {
                        epsilon: *epsilon,
                        latent: true,
                    }),
                    vec![raw_c, V::Input(11)],
                );
                vec![raw_q, qr, q, raw_c, c]
            }

            Self::Norm { kind, epsilon } => match kind {
                Norm::Separate => {
                    let c = p.graph.one(
                        K::SparseF32(S::WeightedRms {
                            epsilon: *epsilon,
                            latent: true,
                        }),
                        vec![V::Input(2), V::Input(3)],
                    );
                    let q = p.graph.one(
                        K::SparseF32(S::WeightedRms {
                            epsilon: *epsilon,
                            latent: false,
                        }),
                        vec![V::Input(0), V::Input(1)],
                    );
                    vec![q, c]
                }
                _ => {
                    let op = match kind {
                        Norm::Rows => S::QueryLatentRmsRows { epsilon: *epsilon },
                        Norm::Compact => S::QueryLatentRmsCompact { epsilon: *epsilon },
                        Norm::Into(position) => {
                            p.graph.writable_input(4);
                            S::QueryLatentRmsInto {
                                epsilon: *epsilon,
                                position: *position,
                            }
                        }
                        _ => unreachable!(),
                    };
                    p.graph.call(
                        K::SparseF32(op),
                        (0..inputs).map(V::Input).collect::<Vec<_>>(),
                    )
                }
            },
            Self::Query(s) => {
                let y = p.reshape(
                    V::Input(0),
                    &[
                        s.batch as i32,
                        s.tokens as i32,
                        s.heads as i32,
                        s.query as i32,
                    ],
                );
                let y = p.transpose(y, &[0, 2, 1, 3]);
                vec![p.contiguous(y)]
            }
            Self::Output(s) => {
                let y = p.transpose(V::Input(0), &[0, 2, 1, 3]);
                let y = p.contiguous(y);
                vec![p.reshape(
                    y,
                    &[s.batch as i32, s.tokens as i32, (s.heads * s.value) as i32],
                )]
            }
            Self::FinalFallback(s) => {
                let y = p.reshape(
                    V::Input(0),
                    &[
                        s.batch as i32,
                        s.heads as i32,
                        s.tokens as i32,
                        s.value as i32,
                    ],
                );
                let y = p.transpose(y, &[0, 2, 1, 3]);
                vec![p.reshape(
                    y,
                    &[s.batch as i32, s.tokens as i32, (s.heads * s.value) as i32],
                )]
            }
            Self::OutputProjection { shape, group } => {
                anyhow::ensure!(
                    shape.len() == 3 && shape[2] == 16384,
                    "invalid attention output projection"
                );
                let y = p.reshape(
                    V::Input(0),
                    &[
                        shape[0]
                            .checked_mul(shape[1])
                            .ok_or_else(|| anyhow::anyhow!("attention row overflow"))?,
                        16384,
                    ],
                );
                let y = p.graph.one(
                    K::SparseF32(S::AttentionOutput { group: *group }),
                    vec![y, V::Input(1), V::Input(2), V::Input(3)],
                );
                vec![p.reshape(y, &[shape[0], shape[1], 4096])]
            }
            Self::CastF32 => vec![p.cast(V::Input(0), D::F32)],
            Self::Contiguous => vec![p.contiguous(V::Input(0))],
            Self::SplitKv(s) => {
                let kv = p.reshape(
                    V::Input(0),
                    &[
                        s.batch as i32,
                        s.cache as i32,
                        s.heads as i32,
                        (s.query + s.value) as i32,
                    ],
                );
                let k = p.slice(kv, 3, 0, s.query as i32);
                let k = p.transpose(k, &[0, 2, 1, 3]);
                let k = p.contiguous(k);
                let v = p.slice(kv, 3, s.query as i32, (s.query + s.value) as i32);
                let v = p.transpose(v, &[0, 2, 1, 3]);
                let v = p.contiguous(v);
                vec![k, v]
            }
            Self::DenseDecode { shape: s, scale } => {
                let q = p.reshape(
                    V::Input(0),
                    &[s.batch as i32, s.heads as i32, s.query as i32],
                );
                let k = p.contiguous(V::Input(1));
                let v = p.contiguous(V::Input(2));
                let keep = p.reshape(V::Input(3), &[s.batch as i32, s.cache as i32]);
                let sc = p.graph.one(
                    K::SparseF32(S::AttentionScores { scale: *scale }),
                    vec![q, k, keep],
                );
                vec![
                    p.graph
                        .one(K::SparseF32(S::AttentionWeightedValue), vec![sc, v]),
                ]
            }
            Self::DenseTranspose => {
                let k = p.transpose(V::Input(0), &[0, 1, 3, 2]);
                vec![p.contiguous(k)]
            }
            Self::DenseOperands { shape: s, row } => {
                let mut out = vec![];
                for (i, shape) in [
                    (0, vec![s.heads as i32, s.tokens as i32, s.query as i32]),
                    (1, vec![s.heads as i32, s.query as i32, s.cache as i32]),
                    (2, vec![s.heads as i32, s.cache as i32, s.value as i32]),
                ] {
                    let x = p.slice(V::Input(i), 0, *row as i32, (*row + 1) as i32);
                    out.push(p.reshape(x, &shape));
                }
                out.push(p.slice(V::Input(3), 0, *row as i32, (*row + 1) as i32));
                out
            }
            Self::DenseScores { shape: s, scale } => {
                let sc = p.op(T::Matmul, vec![V::Input(0), V::Input(1)]);
                let scale = p.graph.constant_f32(&[f32::from_bits(*scale)], &[1]);
                let sc = p.binary(B::Multiply, sc, scale);
                let sc = p.binary(B::Add, sc, V::Input(2));
                let sc = p.reshape(sc, &[s.heads as i32, s.tokens as i32, s.cache as i32]);
                let probs = p.op(T::Softmax { axis: -1 }, vec![sc]);
                vec![sc, probs]
            }
            Self::DenseValue { shape: s } => {
                let y = p.op(T::Matmul, vec![V::Input(0), V::Input(1)]);
                vec![p.reshape(y, &[(s.heads * s.tokens * s.value) as i32])]
            }
            Self::Join { count, axis } => {
                vec![p.concat((0..*count).map(V::Input).collect(), *axis)]
            }
            Self::BroadcastIds(s) => vec![p.op(
                T::Broadcast(vec![s.batch as i32, s.tokens as i32, s.cache as i32]),
                vec![V::Input(0)],
            )],
        };
        Ok(p.finish(out)?.graph)
    }
}

fn projection(
    p: &mut Build<'_>,
    spec: &super::affine_projection::Spec,
    x: V,
    start: usize,
) -> anyhow::Result<V> {
    Ok(p.graph
        .append(
            spec.graph()?,
            &[x, V::Input(start), V::Input(start + 1), V::Input(start + 2)],
            &[],
            &[],
        )
        .map_err(anyhow::Error::msg)?[0])
}
pub const PROJECTIONS: [&str; 5] = [
    "q_a_proj",
    "q_b_proj",
    "kv_a_proj_with_mqa",
    "kv_b_proj",
    "o_proj",
];
pub const NORMS: [&str; 2] = ["q_a_layernorm.weight", "kv_a_layernorm.weight"];
pub const INDEXER_WEIGHTS: [&str; 7] = [
    "indexer.wq_b.weight",
    "indexer.wk.weight",
    "indexer.k_norm.weight",
    "indexer.k_norm.bias",
    "indexer.weights_proj.weight",
    "indexer.index_kpool_compress_gate",
    "indexer.index_kpool_compress_ape",
];
pub const QUANTIZATION: (i32, i32) = (64, 8);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Selection {
    Causal,
    Prefix,
    Indexed,
    None,
}
pub fn selection(s: Shape, limit: usize) -> Selection {
    if s.tokens > 1 && s.cache <= limit {
        Selection::Causal
    } else if s.tokens == 1 || s.cache > limit {
        if s.cache <= limit {
            Selection::Prefix
        } else {
            Selection::Indexed
        }
    } else {
        Selection::None
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mask {
    GpuKeep,
    Selected,
    Causal,
    HostSelection,
}
pub fn mask(s: Shape, limit: usize, pool_ops: bool, selected: bool) -> Mask {
    if s.tokens == 1 && s.cache > limit && !pool_ops {
        Mask::GpuKeep
    } else if selected {
        Mask::Selected
    } else if s.cache <= limit {
        Mask::Causal
    } else {
        Mask::HostSelection
    }
}
pub fn scale(query_dimension: usize) -> f32 {
    (query_dimension as f64).powf(-0.5) as f32
}
pub fn value_partitioned(s: Shape, prefill: bool, ragged: bool) -> bool {
    prefill && s.batch == 1 && !ragged
}
pub fn dense_fallback_allowed(s: Shape, limit: usize) -> bool {
    s.cache <= limit
}
pub const LATENT: i32 = 512;

pub fn fused_dense(s: Shape, operations: bool) -> bool {
    s.tokens == 1 && !operations
}
