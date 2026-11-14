//! Original split-Q8 attention dataflow and cache-selection policy.
use super::{indexer::Spec, packed::Projection};
use crate::{
    graph::{Graph, Value as V},
    program::Build,
};
use lisa_backend::{Kernel as K, SparseF32 as S};
pub const INDEXER: Spec = Spec {
    heads: 32,
    dimension: 128,
    topk: 2048,
    pool: 4,
    tail: true,
};
pub const WEIGHTS: [&str; 15] = [
    "attn_q_a.weight",
    "attn_q_b.weight",
    "attn_kv_a_mqa.weight",
    "attn_output.weight",
    "attn_k_b.weight",
    "attn_v_b.weight",
    "attn_q_a_norm.weight",
    "attn_kv_a_norm.weight",
    "indexer.attn_q_b.weight",
    "indexer.attn_k.weight",
    "indexer.k_norm.weight",
    "indexer.k_norm.bias",
    "indexer.proj.weight",
    "indexer.pool_gate.weight",
    "indexer.pool_ape.weight",
];
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Request {
    Project([Projection; 3]),
    Attend {
        selected: usize,
        dense: bool,
        output: Projection,
    },
}
fn project(
    p: &mut Build<'_>,
    descriptor: Projection,
    x: V,
    w: V,
    shape: &[i32],
) -> anyhow::Result<V> {
    Ok(p.graph
        .append(descriptor.graph(shape)?, &[w, x], &[], &[])
        .map_err(anyhow::Error::msg)?[0])
}
impl Request {
    pub fn graph(&self) -> anyhow::Result<Graph> {
        let schema = Default::default();
        let mut p = Build::new(6, &schema);
        let out = match self {
            Self::Project(d) => {
                anyhow::ensure!(
                    d[0].input == 4096
                        && d[0].output == 1536
                        && d[1].input == 1536
                        && d[1].output == 16384
                        && d[2].input == 4096
                        && d[2].output == 512,
                    "invalid split-Q8 attention geometry"
                );
                let qa = project(&mut p, d[0], V::Input(0), V::Input(1), &[1, 1, 4096])?;
                let qr = p.graph.one(
                    K::SparseF32(S::WeightedRms {
                        epsilon: 1e-5f32.to_bits(),
                        latent: false,
                    }),
                    vec![qa, V::Input(4)],
                );
                let q = project(&mut p, d[1], qr, V::Input(2), &[1, 1, 1536])?;
                let kv = project(&mut p, d[2], V::Input(0), V::Input(3), &[1, 1, 4096])?;
                let latent = p.graph.one(
                    K::SparseF32(S::WeightedRms {
                        epsilon: 1e-5f32.to_bits(),
                        latent: true,
                    }),
                    vec![kv, V::Input(5)],
                );
                vec![qr, q, latent]
            }
            Self::Attend {
                selected,
                dense,
                output,
            } => {
                anyhow::ensure!(
                    *selected > 0 && output.input == 16384 && output.output == 4096,
                    "invalid split-Q8 attention output"
                );
                let q = V::Input(0);
                let low = p
                    .graph
                    .one(K::SparseF32(S::Q8HeadQuery), vec![q, V::Input(3)]);
                let attended = if *selected > 2051 {
                    let q = p.reshape(q, &[1, 64, 1, 256]);
                    let latent = p.graph.one(
                        K::SparseF32(S::IndexedLatent {
                            scale: 0.0625f32.to_bits(),
                        }),
                        vec![q, low, V::Input(1), V::Input(2)],
                    );
                    p.graph
                        .one(K::SparseF32(S::Q8HeadValue), vec![latent, V::Input(4)])
                } else {
                    p.graph.one(
                        K::SparseF32(S::Q8IndexedDecode { dense: *dense }),
                        vec![low, V::Input(1), V::Input(2), V::Input(4)],
                    )
                };
                vec![project(
                    &mut p,
                    *output,
                    attended,
                    V::Input(5),
                    &[1, 1, 16384],
                )?]
            }
        };
        Ok(p.finish(out)?.graph)
    }
}
pub fn dense(total: usize) -> bool {
    total <= 2051
}
pub fn ids(start: usize, end: usize) -> anyhow::Result<Vec<i32>> {
    anyhow::ensure!(
        start < end && end <= i32::MAX as usize,
        "invalid attention token range"
    );
    Ok((start as i32..end as i32).collect())
}

pub fn weight_name(layer: usize, suffix: &str) -> String {
    format!("blk.{layer}.{suffix}")
}
pub fn valid_input(shape: &[i32]) -> bool {
    shape == super::packed_tower::INPUT_SHAPE
}
