//! Legacy affine GLM decoder and draft plans. Storage and device execution are
//! engine responsibilities; numerical order and scheduling policy live here.
use super::affine_projection;
use crate::{
    graph::{Graph, Value as V},
    program::Build,
};
use lisa_backend::{
    Binary as B, Kernel as K, ScalarType as D, SparseF32 as S, TensorOperation as T,
};
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Request {
    Promote,
    Reshape(Vec<i32>),
    Slice {
        axis: i32,
        start: i32,
        end: i32,
    },
    Join {
        count: usize,
        axis: i32,
    },
    Embedding {
        group: i32,
        bits: i32,
    },
    Expand {
        batch: usize,
        rows: usize,
        streams: usize,
        hidden: usize,
    },
    Rms {
        epsilon: u32,
    },
    OrderedRms {
        epsilon: u32,
    },
    Mean {
        shape: Vec<i32>,
    },
    PrefixMean {
        shape: Vec<i32>,
    },
    DraftHidden {
        hidden: usize,
        streams: usize,
        epsilon: u32,
    },
    DraftProject {
        hidden: usize,
        dtype: D,
    },
    AddNorm {
        epsilon: u32,
    },
    AddThree,
    Add,
    LastRow {
        rows: usize,
        hidden: usize,
    },
    Head {
        projection: affine_projection::Spec,
    },
}
fn mean(p: &mut Build<'_>, x: V, shape: &[i32]) -> anyhow::Result<V> {
    anyhow::ensure!(
        shape.len() == 4 && shape[2] == 4 && shape.iter().all(|&d| d > 0),
        "GLM collapse expects [B,T,4,H]"
    );
    let count = shape[..3]
        .iter()
        .try_fold(1usize, |a, &b| a.checked_mul(b as usize))
        .ok_or_else(|| anyhow::anyhow!("GLM collapse overflow"))?;
    let weights = p.graph.constant_f32(&vec![0.25; count], &shape[..3]);
    Ok(p.graph.one(K::SparseF32(S::WeightedSum4), vec![x, weights]))
}
fn rms(p: &mut Build<'_>, x: V, w: V, epsilon: u32) -> V {
    p.graph
        .one(K::SparseF32(S::OrderedRms { epsilon }), vec![x, w])
}
impl Request {
    pub fn graph(&self) -> anyhow::Result<Graph> {
        let inputs = match self {
            Self::Embedding { .. } | Self::Head { .. } => 4,
            Self::Join { count, .. } => *count,
            Self::Rms { .. } | Self::OrderedRms { .. } | Self::DraftHidden { .. } | Self::Add => 2,
            Self::DraftProject { .. } | Self::AddNorm { .. } | Self::AddThree => 3,
            _ => 1,
        };
        let schema = Default::default();
        let mut p = Build::new(inputs, &schema);
        let out = match self {
            Self::Promote => vec![p.cast(V::Input(0), D::F32)],
            Self::Reshape(shape) => vec![p.reshape(V::Input(0), shape)],
            Self::Slice { axis, start, end } => vec![p.slice(V::Input(0), *axis, *start, *end)],
            Self::Join { count, axis } => {
                vec![p.concat((0..*count).map(V::Input).collect(), *axis)]
            }
            Self::Embedding { group, bits } => {
                let rows = (1..4)
                    .map(|i| p.op(T::Take { axis: 0 }, vec![V::Input(i), V::Input(0)]))
                    .collect::<Vec<_>>();
                let x = p.graph.one(
                    K::AffineDequantize {
                        group_size: *group,
                        bits: *bits,
                    },
                    rows,
                );
                vec![p.cast(x, D::F32)]
            }
            Self::Expand {
                batch,
                rows,
                streams,
                hidden,
            } => {
                let x = p.reshape(
                    V::Input(0),
                    &[*batch as i32, *rows as i32, 1, *hidden as i32],
                );
                let x = p.op(
                    T::Broadcast(vec![
                        *batch as i32,
                        *rows as i32,
                        *streams as i32,
                        *hidden as i32,
                    ]),
                    vec![x],
                );
                vec![p.contiguous(x)]
            }
            Self::Rms { epsilon } => p
                .graph
                .append(
                    super::math::rms(3, f32::from_bits(*epsilon), true, false)?,
                    &[V::Input(0), V::Input(1)],
                    &[],
                    &[],
                )
                .map_err(anyhow::Error::msg)?,
            Self::OrderedRms { epsilon } => vec![rms(&mut p, V::Input(0), V::Input(1), *epsilon)],
            Self::Mean { shape } => vec![mean(&mut p, V::Input(0), shape)?],
            Self::PrefixMean { shape } => {
                anyhow::ensure!(shape.len() == 4, "invalid prefix state rank");
                let x = p.contiguous(V::Input(0));
                let x = p.op(
                    T::Mean {
                        axis: 2,
                        keep_dims: true,
                    },
                    vec![x],
                );
                vec![p.reshape(x, &[shape[0], shape[1], shape[3]])]
            }
            Self::DraftHidden {
                hidden,
                streams,
                epsilon,
            } => {
                anyhow::ensure!(*streams == 4, "GLM draft requires four streams");
                let x = p.cast(V::Input(0), D::F32);
                let shape = [1, 1, *streams as i32, *hidden as i32];
                let x = p.reshape(x, &shape);
                let x = mean(&mut p, x, &shape)?;
                vec![rms(&mut p, x, V::Input(1), *epsilon)]
            }
            Self::DraftProject { hidden, dtype } => {
                let joined = p.concat(vec![V::Input(0), V::Input(1)], 2);
                let rows = p.reshape(joined, &[1, (2 * hidden) as i32]);
                let x = if *dtype == D::BF16 {
                    p.graph.one(
                        K::SparseF32(S::Bf16Linear { wide: false }),
                        vec![V::Input(2), rows],
                    )
                } else {
                    let w = p.transpose(V::Input(2), &[1, 0]);
                    let w = p.cast(w, D::F32);
                    p.graph
                        .append(
                            super::math::matmul(
                                &[1, 1, (2 * hidden) as i32],
                                &[(2 * hidden) as i32, *hidden as i32],
                            )?,
                            &[joined, w],
                            &[],
                            &[],
                        )
                        .map_err(anyhow::Error::msg)?[0]
                };
                vec![p.reshape(x, &[1, 1, *hidden as i32])]
            }
            Self::AddNorm { epsilon } => p.graph.call(
                K::SparseF32(S::OrderedAddRms { epsilon: *epsilon }),
                vec![V::Input(0), V::Input(1), V::Input(2)],
            ),
            Self::AddThree => vec![p.graph.one(
                K::SparseF32(S::OrderedAddThree),
                vec![V::Input(0), V::Input(1), V::Input(2)],
            )],
            Self::Add => vec![p.binary(B::Add, V::Input(0), V::Input(1))],
            Self::LastRow { rows, hidden } => {
                anyhow::ensure!(*rows > 0, "empty GLM rows");
                let x = p.slice(V::Input(0), 1, (*rows - 1) as i32, *rows as i32);
                vec![p.reshape(x, &[1, *hidden as i32])]
            }
            Self::Head { projection } => {
                anyhow::ensure!(!projection.shape.is_empty(), "empty head rank");
                let shape = &projection.shape;
                let rows = shape[..shape.len() - 1].iter().product::<i32>();
                let x = p.reshape(V::Input(0), &[rows, *shape.last().unwrap()]);
                let mut flat = projection.clone();
                flat.shape = vec![rows, *shape.last().unwrap()];
                let y = p
                    .graph
                    .append(
                        flat.graph()?,
                        &[x, V::Input(1), V::Input(2), V::Input(3)],
                        &[],
                        &[],
                    )
                    .map_err(anyhow::Error::msg)?[0];
                let mut out = shape[..shape.len() - 1].to_vec();
                if out.is_empty() {
                    out.push(1);
                }
                out.push(projection.weight[0]);
                vec![p.reshape(y, &out)]
            }
        };
        Ok(p.finish(out)?.graph)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Step {
    AttentionInput,
    Attention,
    AttentionResidual,
    FfnInput,
    Ffn,
    FfnResidual,
}
pub const LAYER: [Step; 6] = [
    Step::AttentionInput,
    Step::Attention,
    Step::AttentionResidual,
    Step::FfnInput,
    Step::Ffn,
    Step::FfnResidual,
];
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DraftStep {
    EmbeddingNorm,
    HiddenNorm,
    JoinProject,
    AttentionNorm,
    Attention,
    AddNorm,
    Experts,
    AddThree,
    OutputNorm,
    Head,
}
pub const DRAFT: [DraftStep; 10] = [
    DraftStep::EmbeddingNorm,
    DraftStep::HiddenNorm,
    DraftStep::JoinProject,
    DraftStep::AttentionNorm,
    DraftStep::Attention,
    DraftStep::AddNorm,
    DraftStep::Experts,
    DraftStep::AddThree,
    DraftStep::OutputNorm,
    DraftStep::Head,
];
pub const PREFILL_CHUNK: usize = 2048;
pub fn prefill_rows(remaining: usize, position: usize, capacity: usize) -> usize {
    let work = capacity.min(4096);
    let rows = remaining.min(PREFILL_CHUNK);
    if position < work {
        rows.min(work - position)
    } else {
        rows
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Submission {
    None,
    Flush,
    Drain,
}
pub fn submission(
    batch: usize,
    rows: usize,
    prefill: bool,
    final_chunk: bool,
    preserves_dispatch: bool,
    layer: usize,
    layers: usize,
) -> Submission {
    if !preserves_dispatch || layer + 1 >= layers {
        return Submission::None;
    }
    if batch == 1 && rows == 1 && !prefill && (layer + 1).is_multiple_of(4) {
        return Submission::Flush;
    }
    if prefill && (rows > PREFILL_CHUNK || (final_chunk && rows > 8)) {
        if (layer + 1).is_multiple_of(16) {
            Submission::Drain
        } else {
            Submission::Flush
        }
    } else {
        Submission::None
    }
}

pub fn layer_indices(layers: usize, stop: Option<usize>) -> std::ops::Range<usize> {
    0..stop.map_or(layers, |n| n.saturating_add(1).min(layers))
}

pub const NATIVE_BATCH_ROW_LIMIT: usize = 2048;
pub const DEFAULT_CONTEXT: usize = 32768;
pub fn batch_position_supported(rows: usize, position: usize, dense_limit: usize) -> bool {
    (2..=NATIVE_BATCH_ROW_LIMIT).contains(&rows) && position < dense_limit
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn submission_and_chunk_boundaries_retain_legacy_schedule() {
        for b in [1, 2] {
            for rows in [1, 2, 8, 9, 2048, 2049] {
                for prefill in [false, true] {
                    for final_chunk in [false, true] {
                        for preserve in [false, true] {
                            for layer in 0..45 {
                                let serial = b == 1
                                    && rows == 1
                                    && !prefill
                                    && preserve
                                    && (layer + 1) % 4 == 0
                                    && layer + 1 < 45;
                                let chunk = prefill
                                    && (rows > 2048 || (final_chunk && rows > 8))
                                    && layer + 1 < 45
                                    && preserve;
                                let expected = if serial {
                                    Submission::Flush
                                } else if chunk {
                                    if (layer + 1) % 16 == 0 {
                                        Submission::Drain
                                    } else {
                                        Submission::Flush
                                    }
                                } else {
                                    Submission::None
                                };
                                assert_eq!(
                                    submission(b, rows, prefill, final_chunk, preserve, layer, 45),
                                    expected
                                );
                            }
                        }
                    }
                }
            }
        }
        assert_eq!(prefill_rows(4096, 4095, 32768), 1);
        assert_eq!(prefill_rows(4096, 4096, 32768), 2048);
        assert_eq!(
            layer_indices(45, Some(3)).collect::<Vec<_>>(),
            vec![0, 1, 2, 3]
        );
    }
}

pub fn loaded_pairing_law_bits(modes: impl IntoIterator<Item = bool>) -> u64 {
    modes.into_iter().fold(0, |mask, paired| {
        mask | if paired { 1 << 4 } else { 1 << 5 }
    })
}

#[cfg(test)]
mod loaded_pairing_law_tests {
    #[test]
    fn loaded_projection_modes_have_distinct_cache_identities() {
        use super::loaded_pairing_law_bits;
        assert_eq!(loaded_pairing_law_bits([true, true]), 1 << 4);
        assert_eq!(loaded_pairing_law_bits([false, false]), 1 << 5);
        assert_eq!(loaded_pairing_law_bits([true, false]), (1 << 4) | (1 << 5));
    }
}

pub fn kv_bytes_per_token(config: &super::config::ModelConfig, layers: usize) -> usize {
    // Admission conservatively charges F32 indexer key/gate rows per input
    // token because long prefill holds them until the first pool update.
    // After pooling, only a bounded rollback tail remains; compact MLA
    // latents and pooled keys stay F16 for the context lifetime.
    let dsa = (0..layers).filter(|&i| !config.is_kda(i)).count();
    let raw_indexer = 2 * config.index_head_dim * 4;
    let compact_latent = config.kv_lora_rank * 2;
    let pooled_per_token = config.index_head_dim * 2 / config.index_kpool;
    dsa * (raw_indexer + compact_latent + pooled_per_token)
}

pub fn stream_state_bytes(config: &super::config::ModelConfig, layers: usize) -> usize {
    // KDA: the reference-compatible F32 conv window + F32 recurrence state.
    let qkv = config.linear_num_heads * config.linear_head_dim;
    let per_kda = (config.linear_conv_kernel_dim - 1) * qkv * 3 * 4
        + config.linear_num_heads * config.linear_head_dim.pow(2) * 4;
    let kda = (0..layers).filter(|&i| config.is_kda(i)).count();
    kda * per_kda
}
