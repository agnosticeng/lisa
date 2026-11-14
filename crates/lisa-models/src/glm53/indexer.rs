//! GLM sparse selection dataflow and cache/chunk policy. No device or I/O.
use crate::{
    graph::{Graph, Value as V},
    program::Build,
};
use lisa_backend::{
    Binary as B, Comparison as C, Kernel as K, ScalarType as D, ScoreSchedule, SparseF32 as S,
    TensorOperation as T, Unary as U,
};
pub const LAYER_NORM_EPS: f32 = 1e-6;
pub const RAW_ROLLBACK_ROWS: usize = 32;
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Spec {
    pub heads: usize,
    pub dimension: usize,
    pub topk: usize,
    pub pool: usize,
    pub tail: bool,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Projection {
    pub shape: Vec<i32>,
    pub weight: Vec<i32>,
    pub dtype: D,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Request {
    Project(Projection),
    Pair(Projection, Projection),
    ScoreInputs {
        spec: Spec,
        query: Projection,
        weights: Projection,
    },
    Pool {
        spec: Spec,
        batch: usize,
        count: usize,
        attempt: bool,
        fused: bool,
    },
    Scores {
        spec: Spec,
        position: usize,
        prefill_tail: bool,
        rows4: bool,
    },
    Expand {
        spec: Spec,
        batch: usize,
        tokens: usize,
        pools: usize,
        offset: usize,
        shuffle: bool,
    },
    Keep {
        spec: Spec,
        batch: usize,
        pools: usize,
        total: usize,
        shuffle: bool,
    },
    Chunk {
        start: usize,
        count: usize,
    },
    Join {
        count: usize,
    },
    LayerNorm {
        epsilon: u32,
    },
    WeightedRms {
        epsilon: u32,
        latent: bool,
    },
}
fn append(p: &mut Build<'_>, g: Graph, a: &[V]) -> anyhow::Result<Vec<V>> {
    p.graph.append(g, a, &[], &[]).map_err(anyhow::Error::msg)
}
fn constant_i32(p: &mut Build<'_>, v: impl IntoIterator<Item = i32>, shape: &[i32]) -> V {
    p.graph.constant(
        v.into_iter().flat_map(i32::to_le_bytes).collect(),
        shape.to_vec(),
        D::I32,
    )
}
fn broadcast(p: &mut Build<'_>, v: V, shape: &[i32]) -> V {
    p.op(T::Broadcast(shape.to_vec()), vec![v])
}
impl Projection {
    fn geometry(&self) -> anyhow::Result<(i32, i32, Vec<i32>)> {
        anyhow::ensure!(
            self.shape.len() >= 2
                && self.weight.len() == 2
                && self.shape.iter().all(|&d| d > 0)
                && self.weight.iter().all(|&d| d > 0),
            "invalid indexer projection geometry"
        );
        let k = *self.shape.last().unwrap();
        anyhow::ensure!(k == self.weight[1], "indexer projection K mismatch");
        let rows = self.shape[..self.shape.len() - 1]
            .iter()
            .try_fold(1i32, |a, &d| a.checked_mul(d))
            .ok_or_else(|| anyhow::anyhow!("indexer projection row overflow"))?;
        let mut out = self.shape.clone();
        *out.last_mut().unwrap() = self.weight[0];
        Ok((rows, k, out))
    }
    fn emit(&self, p: &mut Build<'_>, x: V, w: V) -> anyhow::Result<V> {
        let (rows, k, out) = self.geometry()?;
        if self.dtype == D::BF16 {
            let x = p.reshape(x, &[rows, k]);
            let y = p
                .graph
                .one(K::SparseF32(S::Bf16Linear { wide: rows > 8 }), vec![w, x]);
            Ok(p.reshape(y, &out))
        } else {
            let w = p.transpose(w, &[1, 0]);
            let w = p.cast(w, D::F32);
            Ok(append(
                p,
                super::math::matmul(&self.shape, &[self.weight[1], self.weight[0]])?,
                &[x, w],
            )?[0])
        }
    }
}
fn pair(
    p: &mut Build<'_>,
    a: &Projection,
    b: &Projection,
    inputs: [V; 4],
) -> anyhow::Result<[V; 2]> {
    let (ra, ka, sa) = a.geometry()?;
    let (rb, kb, sb) = b.geometry()?;
    if ra == rb && ra <= 8 && a.dtype == D::BF16 && b.dtype == D::BF16 {
        let x = p.reshape(inputs[0], &[ra, ka]);
        let y = p.reshape(inputs[2], &[rb, kb]);
        let out = p.graph.call(
            K::SparseF32(S::Bf16PairLinear),
            vec![inputs[1], x, inputs[3], y],
        );
        Ok([p.reshape(out[0], &sa), p.reshape(out[1], &sb)])
    } else {
        Ok([
            a.emit(p, inputs[0], inputs[1])?,
            b.emit(p, inputs[2], inputs[3])?,
        ])
    }
}
pub fn layer_norm(epsilon: u32) -> anyhow::Result<Graph> {
    let schema = Default::default();
    let mut p = Build::new(3, &schema);
    let x = p.cast(V::Input(0), D::F32);
    let mean = p.op(
        T::Mean {
            axis: 2,
            keep_dims: true,
        },
        vec![x],
    );
    let cent = p.binary(B::Subtract, x, mean);
    let sq = p.binary(B::Multiply, cent, cent);
    let var = p.op(
        T::Mean {
            axis: 2,
            keep_dims: true,
        },
        vec![sq],
    );
    let eps = p.scalar(f32::from_bits(epsilon), D::F32);
    let inv = p.binary(B::Add, var, eps);
    let inv = p.unary(U::Rsqrt, inv);
    let n = p.binary(B::Multiply, cent, inv);
    let w = p.cast(V::Input(1), D::F32);
    let y = p.binary(B::Multiply, n, w);
    let bias = p.cast(V::Input(2), D::F32);
    let y = p.binary(B::Add, y, bias);
    Ok(p.finish(vec![y])?.graph)
}
impl Request {
    pub fn graph(&self) -> anyhow::Result<Graph> {
        let inputs = match self {
            Self::Project(_) => 2,
            Self::Pair(..) | Self::ScoreInputs { .. } => 4,
            Self::Pool { .. } => 4,
            Self::Scores { .. } | Self::LayerNorm { .. } => 3,
            Self::Expand { .. } | Self::Keep { .. } => 1,
            Self::Chunk { .. } | Self::WeightedRms { .. } => 2,
            Self::Join { count } => *count,
        };
        let schema = Default::default();
        let mut p = Build::new(inputs, &schema);
        let out = match self {
            Self::Project(a) => vec![a.emit(&mut p, V::Input(0), V::Input(1))?],
            Self::Pair(a, b) => pair(
                &mut p,
                a,
                b,
                [V::Input(0), V::Input(1), V::Input(2), V::Input(3)],
            )?
            .to_vec(),
            Self::ScoreInputs {
                spec,
                query,
                weights,
            } => {
                let [q, w] = pair(
                    &mut p,
                    query,
                    weights,
                    [V::Input(0), V::Input(1), V::Input(2), V::Input(3)],
                )?;
                let (b, t) = (weights.shape[0], weights.shape[1]);
                let q = p.cast(q, D::F32);
                let q = p.reshape(q, &[b, t, spec.heads as i32, spec.dimension as i32]);
                let q = p.transpose(q, &[0, 2, 1, 3]);
                let q = p.contiguous(q);
                let w = p.cast(w, D::F32);
                vec![q, w]
            }
            Self::Pool {
                spec,
                batch,
                count,
                attempt,
                fused,
            } => {
                let (b, n, k, d) = (
                    *batch as i32,
                    *count as i32,
                    spec.pool as i32,
                    spec.dimension as i32,
                );
                anyhow::ensure!(
                    b > 0 && n > 0 && k > 0 && d > 0 && (!fused || *attempt && k == 4 && d == 128),
                    "invalid indexer pool geometry"
                );
                let raw = p.slice(V::Input(0), 2, 0, 2 * d);
                let raw = p.contiguous(raw);
                let grouped = p.reshape(raw, &[b, n, k, 2 * d]);
                let fast = if *attempt {
                    let key = p.slice(grouped, 3, 0, d);
                    let key = p.contiguous(key);
                    let gate = p.slice(grouped, 3, d, 2 * d);
                    let gate = p.contiguous(gate);
                    if *fused {
                        Some(p.graph.one(
                            K::SparseF32(S::Pool128 {
                                epsilon: LAYER_NORM_EPS.to_bits(),
                            }),
                            vec![key, gate, V::Input(1), V::Input(2), V::Input(3)],
                        ))
                    } else {
                        None
                    }
                } else {
                    None
                };
                let y = if let Some(y) = fast {
                    y
                } else {
                    let key = p.slice(grouped, 3, 0, d);
                    let key = p.reshape(key, &[b * n * k, d]);
                    let key = p.reshape(key, &[b, n * k, d]);
                    let key = append(
                        &mut p,
                        layer_norm(LAYER_NORM_EPS.to_bits())?,
                        &[key, V::Input(2), V::Input(3)],
                    )?[0];
                    let key = p.reshape(key, &[b, n, k, d]);
                    let gate = p.slice(grouped, 3, d, 2 * d);
                    let gate = p.contiguous(gate);
                    let ape = p.cast(V::Input(1), D::F32);
                    let ape = p.reshape(ape, &[1, k, d]);
                    let ape = p.reshape(ape, &[1, 1, k, d]);
                    let ape = broadcast(&mut p, ape, &[b, n, k, d]);
                    let probs = p.binary(B::Add, gate, ape);
                    let probs = p.op(T::Softmax { axis: 2 }, vec![probs]);
                    let mut sum = None;
                    for j in 0..k {
                        let a = p.slice(probs, 2, j, j + 1);
                        let b = p.slice(key, 2, j, j + 1);
                        let term = p.binary(B::Multiply, a, b);
                        sum = Some(match sum {
                            Some(s) => p.binary(B::Add, s, term),
                            None => term,
                        });
                    }
                    p.reshape(sum.unwrap(), &[b, n, d])
                };
                vec![y]
            }
            Self::Scores {
                spec,
                position,
                prefill_tail,
                rows4,
            } => {
                let scale = ((spec.heads * spec.dimension) as f64).powf(-0.5) as f32;
                let schedule = if *prefill_tail {
                    ScoreSchedule::Batch
                } else if *rows4 {
                    ScoreSchedule::Rows4
                } else {
                    ScoreSchedule::Scalar
                };
                vec![p.graph.one(
                    K::SparseF32(S::IndexScores128 {
                        position: *position,
                        pool_size: spec.pool,
                        scale: scale.to_bits(),
                        schedule,
                    }),
                    vec![V::Input(0), V::Input(1), V::Input(2)],
                )]
            }
            Self::Expand {
                spec,
                batch,
                tokens,
                pools,
                offset,
                shuffle,
            } => {
                let select = spec.selected_pools(*pools);
                anyhow::ensure!(select > 0, "no complete indexer pools available");
                let rows = (batch * tokens) as i32;
                let pools = *pools as i32;
                let start = offset.saturating_sub(*tokens);
                let visible = constant_i32(
                    &mut p,
                    (0..*batch).flat_map(|_| (0..*tokens).map(|q| (start + q + 1) as i32)),
                    &[rows, 1],
                );
                let ends = constant_i32(
                    &mut p,
                    (0..pools).map(|n| (n + 1) * spec.pool as i32),
                    &[1, pools],
                );
                let ends = broadcast(&mut p, ends, &[rows, pools]);
                let visible = broadcast(&mut p, visible, &[rows, pools]);
                let eligible = p.op(T::Compare(C::LessEqual), vec![ends, visible]);
                let scores = p.reshape(V::Input(0), &[rows, pools]);
                let fill = p.scalar(f32::NEG_INFINITY, D::F32);
                let fill = broadcast(&mut p, fill, &[rows, pools]);
                let masked = p.op(T::Where, vec![eligible, scores, fill]);
                let top = p.graph.one(
                    K::SparseF32(S::TopPools {
                        count: select,
                        shuffle: *shuffle,
                    }),
                    vec![masked],
                );
                let expanded = p.graph.one(
                    K::SparseF32(S::ExpandPools {
                        tokens: *tokens,
                        position: start,
                        pool_size: spec.pool,
                        tail: spec.tail,
                    }),
                    vec![top],
                );
                vec![p.reshape(
                    expanded,
                    &[
                        *batch as i32,
                        *tokens as i32,
                        (select * spec.pool + spec.pool - 1) as i32,
                    ],
                )]
            }
            Self::Keep {
                spec,
                batch,
                pools,
                total,
                shuffle,
            } => {
                let (b, tc, select) = (*batch as i32, *total as i32, spec.selected_pools(*pools));
                let sk = select as i32;
                let rows = p.reshape(V::Input(0), &[b, *pools as i32]);
                let top = p.graph.one(
                    K::SparseF32(S::TopPools {
                        count: select,
                        shuffle: *shuffle,
                    }),
                    vec![rows],
                );
                let ar = constant_i32(&mut p, 0..tc, &[tc]);
                let pool = constant_i32(&mut p, [spec.pool as i32], &[]);
                let pool = p.op(T::FloorDivide, vec![ar, pool]);
                let top = p.reshape(top, &[b, 1, sk]);
                let top = broadcast(&mut p, top, &[b, tc, sk]);
                let pool = p.reshape(pool, &[1, tc]);
                let pool = p.reshape(pool, &[1, tc, 1]);
                let pool = broadcast(&mut p, pool, &[b, tc, sk]);
                let hi = p.op(T::Compare(C::LessEqual), vec![pool, top]);
                let member = p.op(T::Compare(C::GreaterEqual), vec![pool, top]);
                let member = p.op(T::LogicalAnd, vec![member, hi]);
                let member = p.cast(member, D::F32);
                let member = p.op(
                    T::Sum {
                        axis: 2,
                        keep_dims: false,
                    },
                    vec![member],
                );
                let zero = p.scalar(0., D::F32);
                let member = p.op(T::Compare(C::Greater), vec![member, zero]);
                let member = p.cast(member, D::F32);
                let tail = constant_i32(&mut p, [(*total - *total % spec.pool) as i32], &[]);
                let tail = p.op(T::Compare(C::GreaterEqual), vec![ar, tail]);
                let tail = p.cast(tail, D::F32);
                let tail = p.reshape(tail, &[1, tc]);
                let tail = broadcast(&mut p, tail, &[b, tc]);
                let keep = p.binary(B::Maximum, member, tail);
                let z = p.scalar(0., D::F32);
                let z = broadcast(&mut p, z, &[b, tc]);
                let neg = p.scalar(-1e30, D::F32);
                let neg = broadcast(&mut p, neg, &[b, tc]);
                let mask = p.op(T::Where, vec![keep, z, neg]);
                vec![p.reshape(mask, &[b, 1, 1, tc])]
            }
            Self::Chunk { start, count } => vec![
                p.slice(V::Input(0), 2, *start as i32, (start + count) as i32),
                p.slice(V::Input(1), 1, *start as i32, (start + count) as i32),
            ],
            Self::Join { count } => {
                let out = p.concat((0..*count).map(V::Input).collect(), 1);
                vec![p.contiguous(out)]
            }
            Self::LayerNorm { epsilon } => return layer_norm(*epsilon),
            Self::WeightedRms { epsilon, latent } => vec![p.graph.one(
                K::SparseF32(S::WeightedRms {
                    epsilon: *epsilon,
                    latent: *latent,
                }),
                vec![V::Input(0), V::Input(1)],
            )],
        };
        Ok(p.finish(out)?.graph)
    }
}
impl Spec {
    pub fn fused_pool(self, dtype: Option<D>) -> bool {
        dtype == Some(D::BF16) && self.dimension == 128 && self.pool == 4
    }
    pub fn tail_width(self) -> usize {
        self.pool - 1
    }
    pub fn validate(self) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.pool > 0 && self.heads > 0 && self.dimension > 0,
            "invalid indexer geometry"
        );
        Ok(())
    }
    pub fn selected_pools(self, pools: usize) -> usize {
        (self.topk / self.pool).min(pools)
    }
    pub fn pool_update(self, total: usize, pooled: usize) -> Option<(usize, usize)> {
        let end = total / self.pool;
        (end > pooled).then_some((pooled, end))
    }
    pub fn score_chunk(self, batch: usize, pools: usize, tokens: usize) -> usize {
        let bytes = batch.saturating_mul(pools).saturating_mul(4);
        (256 * 1024 * 1024 / bytes.max(1))
            .clamp(1, 2048)
            .min(tokens.max(1))
    }
    pub fn valid_prefix(self, pools: usize, tokens: usize) -> usize {
        if tokens == 1 {
            self.selected_pools(pools) * self.pool
        } else {
            0
        }
    }
    pub fn tail_ids(self, batch: usize, tokens: usize, offset: usize) -> Vec<i32> {
        let width = self.pool - 1;
        let mut out = vec![-1; batch * tokens * width];
        for b in 0..batch {
            for q in 0..tokens {
                let visible = offset.saturating_sub(tokens) + q + 1;
                let n = visible % self.pool;
                for m in 0..width.min(n) {
                    out[(b * tokens + q) * width + m] = (visible - n + m) as i32;
                }
            }
        }
        out
    }
    pub fn select_host(
        self,
        scores: &[f32],
        batch: usize,
        tokens: usize,
        pools: usize,
        offset: usize,
        total: usize,
    ) -> Vec<i32> {
        let select = self.selected_pools(pools);
        let width = select * self.pool + self.pool - 1;
        let off = offset.saturating_sub(tokens);
        let mut out = vec![-1; batch * tokens * width];
        for b in 0..batch {
            for q in 0..tokens {
                let abs = off + q;
                let mut candidates = vec![];
                for pp in 0..pools {
                    if pp * self.pool + self.pool - 1 <= abs {
                        candidates.push((scores[(b * tokens + q) * pools + pp], pp));
                    }
                }
                candidates
                    .sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
                let base = (b * tokens + q) * width;
                let mut w = 0;
                for (_, pp) in candidates.iter().take(select) {
                    for j in 0..self.pool {
                        let idx = pp * self.pool + j;
                        if idx < total {
                            out[base + w] = idx as i32;
                            w += 1;
                        }
                    }
                }
                let vis = abs + 1;
                let count = vis % self.pool;
                for m in 0..self.pool - 1 {
                    if m < count {
                        out[base + w] = (vis - count + m) as i32;
                    }
                    w += 1;
                }
            }
        }
        out
    }
}

/// Host fallback masks use the same finite sentinel as the original graph.
pub fn causal_mask(batch: usize, tokens: usize, total: usize) -> anyhow::Result<Vec<f32>> {
    anyhow::ensure!(tokens <= total, "causal mask query exceeds cache");
    let count = batch
        .checked_mul(tokens)
        .and_then(|n| n.checked_mul(total))
        .ok_or_else(|| anyhow::anyhow!("mask size overflow"))?;
    let mut out = vec![0.; count];
    let start = total - tokens;
    for b in 0..batch {
        for q in 0..tokens {
            for k in start + q + 1..total {
                out[(b * tokens + q) * total + k] = -1e30;
            }
        }
    }
    Ok(out)
}
pub fn keep_mask(
    selected: &[i32],
    batch: usize,
    tokens: usize,
    total: usize,
    width: usize,
) -> anyhow::Result<Vec<f32>> {
    anyhow::ensure!(
        tokens <= total
            && batch.checked_mul(tokens).and_then(|n| n.checked_mul(width)) == Some(selected.len()),
        "invalid selection mask geometry"
    );
    let count = batch
        .checked_mul(tokens)
        .and_then(|n| n.checked_mul(total))
        .ok_or_else(|| anyhow::anyhow!("mask size overflow"))?;
    let mut out = vec![-1e30; count];
    let start = total - tokens;
    for b in 0..batch {
        for q in 0..tokens {
            for &id in &selected[(b * tokens + q) * width..(b * tokens + q + 1) * width] {
                if id >= 0 && id as usize <= start + q {
                    out[(b * tokens + q) * total + id as usize] = 0.;
                }
            }
        }
    }
    Ok(out)
}
#[cfg(test)]
mod tests {
    use super::*;
    fn spec() -> Spec {
        Spec {
            heads: 32,
            dimension: 128,
            topk: 8,
            pool: 4,
            tail: true,
        }
    }
    #[test]
    fn indexer_causal_pool_and_chunk_boundaries() {
        let s = spec();
        assert_eq!(s.pool_update(3, 0), None);
        assert_eq!(s.pool_update(4, 0), Some((0, 1)));
        assert_eq!(s.pool_update(7, 1), None);
        assert_eq!(s.pool_update(12, 1), Some((1, 3)));
        assert_eq!(s.tail_ids(1, 3, 3), vec![0, -1, -1, 0, 1, -1, 0, 1, 2]);
        assert_eq!(s.score_chunk(1, 1, 4097), 2048);
        assert_eq!(s.score_chunk(8, 1 << 20, 4097), 8);
        assert_eq!(s.valid_prefix(100, 1), 8);
        assert_eq!(s.valid_prefix(100, 2), 0);
        assert_eq!(
            s.select_host(&[1., 100., 99., 1., 100., 99.], 1, 2, 3, 8, 12),
            vec![
                0, 1, 2, 3, 4, 5, 6, -1, -1, -1, -1, 4, 5, 6, 7, 0, 1, 2, 3, -1, -1, -1
            ]
        );
        assert_eq!(
            causal_mask(1, 2, 3).unwrap(),
            vec![0., 0., -1e30, 0., 0., 0.]
        );
        assert_eq!(
            keep_mask(&[0, 2, -1, 1, 2, -1], 1, 2, 3, 3).unwrap(),
            vec![0., -1e30, -1e30, -1e30, 0., 0.]
        );
        assert!(causal_mask(1, 4, 3).is_err());
        assert!(keep_mask(&[], 1, 2, 3, 3).is_err());
    }
}
