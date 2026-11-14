//! V4.1 attention dataflow, cache mutations and checkpoint schema.
use super::{
    config::Config,
    projection::{self, Matrix},
    rope_frequencies,
    state::{Position, Selection},
};
use crate::graph::{Builder, Graph, Value as V};
use lisa_backend::{
    ActivationEncoding as E, IndexedAttention as A, Kernel as K, PackedCompute as P,
    TensorOperation as T,
};

pub const WEIGHTS: [&str; 15] = [
    "attn_q_a.weight",
    "attn_q_a_norm.weight",
    "attn_kv.weight",
    "attn_kv_a_norm.weight",
    "attn_q_b.weight",
    "attn_output_a.weight",
    "attn_output_b.weight",
    "attn_sinks.weight",
    "attn_compressor_kv.weight",
    "attn_compressor_gate.weight",
    "attn_compressor_norm.weight",
    "indexer.attn_k.weight",
    "indexer.k_norm.weight",
    "indexer.attn_q_b.weight",
    "indexer.proj.weight",
];
pub fn requirements(layer: usize, config: &Config) -> [Option<String>; 15] {
    std::array::from_fn(|i| {
        let enabled = match i {
            0..8 => true,
            8 | 10..13 => config.kv_sources.contains(&(layer as u32)),
            9 => config.kv_sources.contains(&(layer as u32)) && config.compress_ratios[layer] == 2,
            _ => config.index_sources.contains(&(layer as u32)),
        };
        enabled.then(|| format!("blk.{layer}.{}", WEIGHTS[i]))
    })
}
#[derive(Clone, Debug)]
pub struct Spec {
    pub layer: usize,
    pub ratio: usize,
    pub matrices: [Option<Matrix>; 15],
    pub compressor: bool,
    pub gate: bool,
    pub indexer: bool,
}
fn reshape(b: &mut Builder, x: V, shape: &[i32]) -> V {
    b.one(K::Tensor(T::Reshape(shape.to_vec())), vec![x])
}
fn rows(b: &mut Builder, x: V, start: i32, end: i32) -> V {
    b.one(
        K::Tensor(T::Slice {
            axis: 0,
            start,
            end,
        }),
        vec![x],
    )
}
fn norm(b: &mut Builder, x: V, w: V) -> V {
    b.one(
        K::PackedCompute(P::RmsNorm {
            weighted: true,
            round_output: true,
            epsilon: 1e-20f32.to_bits(),
        }),
        vec![x, w],
    )
}
fn frequencies(compressed: bool) -> [u32; 32] {
    rope_frequencies(compressed).map(f32::to_bits)
}
impl Spec {
    fn matrix(&self, slot: usize) -> anyhow::Result<Matrix> {
        self.matrices[slot]
            .ok_or_else(|| anyhow::anyhow!("missing attention matrix contract {slot}"))
    }
    fn linear(&self, b: &mut Builder, slot: usize, x: V, round: bool) -> anyhow::Result<V> {
        let m = self.matrix(slot)?;
        Ok(b.append(
            m.linear(0, m.output, round)?,
            &[V::Weight(slot), x],
            &[],
            &[],
        )
        .map_err(anyhow::Error::msg)?[0])
    }
    pub fn project(&self, count: i32) -> anyhow::Result<Graph> {
        let mut b = Builder::new(1, 15, 0);
        let pair = b
            .append(
                projection::paired(self.matrix(0)?, self.matrix(2)?, count)?,
                &[V::Weight(0), V::Weight(2), V::Input(0)],
                &[],
                &[],
            )
            .map_err(anyhow::Error::msg)?;
        let pair = b.call(
            K::PackedCompute(P::RmsNormPair {
                epsilon: 1e-20f32.to_bits(),
            }),
            vec![pair[0], V::Weight(1), pair[1], V::Weight(3)],
        );
        let q = self.linear(&mut b, 4, pair[0], true)?;
        Ok(b.finish(vec![pair[0], pair[1], q]))
    }
    /// Inputs: norm, qr, kv, q, raw ring, compressed cache, index cache,
    /// optional BF16 mirror, compressor carry pair, candidates, selection,
    /// rope coefficients, then four attention scratch arrays.
    /// Outputs: projected attention, new carry pair, candidates, selection,
    /// followed by requested diagnostic stages. Explicit cache writes retain
    /// the original scalar position order and are invalidated on failure.
    pub fn forward(
        &self,
        pos: usize,
        mirror: bool,
        capture: bool,
    ) -> anyhow::Result<(Graph, Vec<&'static str>)> {
        let p = Position::new(self.layer, self.ratio, pos)?;
        let mut b = Builder::new(17, 15, 0);
        for i in [3, 4, 5, 6, 7, 13, 14, 15, 16] {
            b.writable_input(i);
        }
        let mut records = Vec::new();
        let mut captures = Vec::new();
        let mut record = |name, x| {
            if capture {
                records.push(name);
                captures.push(x);
            }
        };
        let q = b.one(
            K::IndexedAttention(A::Rotary {
                position: pos as u32,
                stride: 1,
                frequencies: frequencies(p.compressed),
                inverse: false,
                round: false,
            }),
            vec![V::Input(3)],
        );
        let kv = reshape(&mut b, V::Input(2), &[1, 512]);
        let raw = b.one(
            K::IndexedAttention(A::RotaryStore {
                row: p.raw_row,
                position: pos as u32,
                frequencies: frequencies(p.compressed),
                encoding: E::Fp8E8M0,
            }),
            vec![kv, V::Input(4)],
        );
        record("query", q);
        if capture {
            record(
                "raw",
                rows(&mut b, raw, p.raw_row as i32, p.raw_row as i32 + 1),
            );
        }
        let mut previous_kv = V::Input(8);
        let mut previous_score = V::Input(9);
        if self.compressor {
            let pool = self.linear(&mut b, 8, V::Input(0), self.ratio == 1)?;
            record("pool", pool);
            let latent = if self.gate {
                let score = self.linear(&mut b, 9, V::Input(0), false)?;
                record("pool_score", score);
                previous_kv = pool;
                previous_score = score;
                if p.pool_pair {
                    Some(b.one(
                        K::IndexedAttention(A::PoolPair),
                        vec![V::Input(8), V::Input(9), pool, score],
                    ))
                } else {
                    None
                }
            } else {
                Some(pool)
            };
            if let Some(latent) = latent {
                let latent = norm(&mut b, latent, V::Weight(10));
                record("latent_norm", latent);
                let index = self.linear(&mut b, 11, latent, true)?;
                let index = norm(&mut b, index, V::Weight(12));
                let row = p
                    .compressed_rows
                    .checked_sub(1)
                    .ok_or_else(|| anyhow::anyhow!("compressed cache has no writable row"))?;
                let start = (pos + 1 - self.ratio) as u32;
                let index = b.one(
                    K::IndexedAttention(A::RotaryStore {
                        row,
                        position: start,
                        frequencies: frequencies(true),
                        encoding: E::Fp4E8M0,
                    }),
                    vec![index, V::Input(6)],
                );
                let latent = b.one(
                    K::IndexedAttention(A::RotaryStore {
                        row,
                        position: start,
                        frequencies: frequencies(true),
                        encoding: E::Fp4E4M3,
                    }),
                    vec![latent, V::Input(5)],
                );
                let index = rows(&mut b, index, row as i32, row as i32 + 1);
                record("index_key", index);
                if capture {
                    record("latent", rows(&mut b, latent, row as i32, row as i32 + 1));
                }
                if mirror {
                    let key = b.one(K::IndexedAttention(A::IndexKeys), vec![index]);
                    b.one(
                        K::Tensor(T::CopySliceInto {
                            axis: 0,
                            start: row as i32,
                            end: row as i32 + 1,
                        }),
                        vec![key, V::Input(7)],
                    );
                }
            }
        }
        let mut candidates = V::Input(10);
        let mut selected = V::Input(11);
        if self.indexer && p.compressed_rows > 0 {
            let query = self.linear(&mut b, 13, V::Input(1), true)?;
            let query = reshape(&mut b, query, &[32, 128]);
            let query = b.one(
                K::IndexedAttention(A::IndexQuery),
                vec![query, V::Input(12)],
            );
            record("index_query", query);
            let weight = self.linear(&mut b, 14, V::Input(0), true)?;
            let weight = reshape(&mut b, weight, &[32]);
            selected = match p.selection {
                Selection::Candidates => {
                    let keys = rows(&mut b, V::Input(6), 0, p.compressed_rows as i32);
                    b.one(
                        K::IndexedAttention(A::CandidateSelect {
                            rows: p.compressed_rows as i32,
                        }),
                        vec![query, weight, keys, candidates],
                    )
                }
                Selection::Full {
                    mirror: use_mirror,
                    publish,
                } => {
                    anyhow::ensure!(!use_mirror || mirror, "missing BF16 index mirror");
                    let keys = rows(
                        &mut b,
                        V::Input(if use_mirror { 7 } else { 6 }),
                        0,
                        p.compressed_rows as i32,
                    );
                    let scores = b.one(
                        K::IndexedAttention(A::IndexScores),
                        vec![query, weight, keys],
                    );
                    if publish {
                        candidates = b.one(K::IndexedAttention(A::Candidates), vec![scores]);
                    }
                    b.one(
                        K::IndexedAttention(A::TopK {
                            count: p.selected as i32,
                        }),
                        vec![scores],
                    )
                }
                Selection::Empty => unreachable!(),
            };
            selected = reshape(&mut b, selected, &[p.selected as i32]);
        }
        let q = reshape(&mut b, q, &[64, 512]);
        let mut args = vec![
            q,
            V::Weight(7),
            V::Input(4),
            V::Input(12),
            V::Input(13),
            V::Input(14),
            V::Input(15),
            V::Input(16),
        ];
        if p.compressed_rows > 0 {
            let comp = rows(&mut b, V::Input(5), 0, p.compressed_rows as i32);
            args.extend([comp, selected]);
        }
        let heads = b.one(
            K::IndexedAttention(A::Decode {
                raw_count: p.raw_count,
                raw_start: p.raw_start,
                compressed: p.compressed,
                selected: p.compressed_rows > 0,
            }),
            args,
        );
        let heads = reshape(&mut b, heads, &[1, 64, 512]);
        let low = b
            .append(
                projection::output_groups(self.matrix(5)?)?,
                &[V::Weight(5), heads],
                &[],
                &[],
            )
            .map_err(anyhow::Error::msg)?[0];
        let out = self.linear(&mut b, 6, low, true)?;
        let mut outputs = vec![out, previous_kv, previous_score, candidates, selected];
        outputs.extend(captures);
        let graph = b.finish(outputs);
        graph.validate().map_err(anyhow::Error::msg)?;
        Ok((graph, records))
    }
}
pub fn rope_table(position: u32) -> Graph {
    let mut b = Builder::new(0, 0, 0);
    let x = b.one(
        K::IndexedAttention(A::RotaryTable {
            position,
            frequencies: [frequencies(false), frequencies(true)],
        }),
        vec![],
    );
    b.finish(vec![x])
}
pub fn is_projection(slot: usize) -> bool {
    matches!(slot, 0 | 2 | 4 | 5 | 6 | 8 | 9 | 11 | 13 | 14)
}
impl Spec {
    pub fn requires_candidates(&self, pos: usize) -> anyhow::Result<bool> {
        Ok(self.indexer
            && Position::new(self.layer, self.ratio, pos)?.selection == Selection::Candidates)
    }
    pub fn requires_selection(&self, pos: usize) -> anyhow::Result<bool> {
        Ok(!self.indexer && Position::new(self.layer, self.ratio, pos)?.compressed_rows > 0)
    }
    pub fn publishes_selection(&self, pos: usize) -> anyhow::Result<bool> {
        Ok(self.indexer && Position::new(self.layer, self.ratio, pos)?.compressed_rows > 0)
    }
    pub fn publishes_candidates(&self, pos: usize) -> anyhow::Result<bool> {
        Ok(self.indexer
            && matches!(
                Position::new(self.layer, self.ratio, pos)?.selection,
                Selection::Full { publish: true, .. }
            ))
    }
}
pub fn scratch() -> Graph {
    use lisa_backend::ScalarType as D;
    let mut b = Builder::new(0, 0, 0);
    let out: Vec<_> = [
        (vec![640, 512], D::F16),
        (vec![640], D::F16),
        (vec![2 * 32 * 512 + 32], D::F16),
        (vec![64 * (512 * 32 + 64)], D::F32),
    ]
    .into_iter()
    .map(|(shape, dtype)| b.one(K::Zeros { shape, dtype }, vec![]))
    .collect();
    b.finish(out)
}
