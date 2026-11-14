//! GLM serial-row decoder and MTP order for original packed weights.
use super::{config::ModelConfig, packed::Projection};
use crate::{
    graph::{Graph, Value as V},
    program::Build,
};
use lisa_backend::{Kernel as K, SparseF32 as S};
pub const WIDTH: usize = 4096;
pub const INPUT_SHAPE: [i32; 3] = [1, 1, WIDTH as i32];
pub const NORM_EPSILON: f32 = 1e-5;
pub const FULL_CACHE: (usize, usize) = (1, 512);
pub const HYPER: super::hyper::Spec = super::hyper::Spec {
    streams: 4,
    hidden: WIDTH,
    epsilon: 1e-6f32.to_bits(),
    rms_epsilon: NORM_EPSILON.to_bits(),
    iterations: 20,
};
pub const EMBEDDING_WEIGHT: &str = "token_embd.weight";
pub const OUTPUT_WEIGHT: &str = "output.weight";
pub const OUTPUT_NORM_WEIGHT: &str = "output_norm.weight";
pub const STREAMS: usize = 4;
pub const VOCAB: usize = 154880;
pub const LAYERS: usize = 45;
pub const POSITION_LAYER: usize = 3;
pub const DEFAULT_CAPACITY: usize = 32768;
pub const NUMERICAL_IDENTITY: &[u8] = b"glm53-gguf-serial-v1";
pub const KV_BYTES_PER_TOKEN: usize = 11 * (512 * 2 + 128 * 4 * 2 + 128 * 4 / 4);
pub const STREAM_STATE_BYTES: usize = 34 * (64 * 128 * 128 * 4 + 3 * 8192 * 3 * 4);
pub const MTP_WEIGHTS: [&str; 6] = [
    "nextn.enorm",
    "nextn.hnorm",
    "attn_norm",
    "ffn_norm",
    "nextn.shared_head_norm",
    "nextn.eh_proj",
];
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Attention {
    Recurrent,
    Sparse,
}
pub fn attention(config: &ModelConfig, layer: usize) -> Attention {
    if config.is_kda(layer) {
        Attention::Recurrent
    } else {
        Attention::Sparse
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Step {
    ExpandEmbedding,
    BeginLayer(usize),
    AttentionInput,
    Attention,
    AttentionResidual,
    FfnInput,
    Ffn,
    FfnResidual,
    CaptureLayer,
    EndLayer,
    FinalNorm,
}
pub fn schedule() -> Vec<Step> {
    let mut steps = vec![Step::ExpandEmbedding];
    for i in 0..LAYERS {
        steps.extend([
            Step::BeginLayer(i),
            Step::AttentionInput,
            Step::Attention,
            Step::AttentionResidual,
            Step::FfnInput,
            Step::Ffn,
            Step::FfnResidual,
            Step::CaptureLayer,
            Step::EndLayer,
        ]);
    }
    steps.push(Step::FinalNorm);
    steps
}
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
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Request {
    Expand,
    Final,
    MeanNorm,
    Rms,
    AddNorm,
    AddThree,
    JoinProject(Projection),
    Head {
        shape: Vec<i32>,
        projection: Projection,
    },
    Join {
        count: usize,
    },
    Reshape(Vec<i32>),
}
fn rms(p: &mut Build<'_>, x: V, w: V) -> V {
    p.graph.one(
        K::SparseF32(S::OrderedRms {
            epsilon: 1e-5f32.to_bits(),
        }),
        vec![x, w],
    )
}
fn mean(p: &mut Build<'_>, x: V) -> V {
    let weights = p.graph.constant(
        [0.25f32; 4]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect(),
        vec![1, 1, 4],
        lisa_backend::ScalarType::F32,
    );
    let x = p.reshape(x, &[1, 1, 4, WIDTH as i32]);
    p.graph.one(K::SparseF32(S::WeightedSum4), vec![x, weights])
}
fn project(
    p: &mut Build<'_>,
    projection: Projection,
    shape: &[i32],
    x: V,
    w: V,
) -> anyhow::Result<V> {
    Ok(p.graph
        .append(projection.graph(shape)?, &[w, x], &[], &[])
        .map_err(anyhow::Error::msg)?[0])
}
impl Request {
    pub fn graph(&self) -> anyhow::Result<Graph> {
        let n = match self {
            Self::Expand | Self::Reshape(_) => 1,
            Self::AddNorm | Self::AddThree | Self::JoinProject(_) => 3,
            Self::Join { count } => *count,
            _ => 2,
        };
        let schema = Default::default();
        let mut p = Build::new(n, &schema);
        let x = V::Input(0);
        let out = match self {
            Self::Expand => {
                let x = p.concat(vec![x; STREAMS], 2);
                vec![p.reshape(x, &[1, 1, STREAMS as i32, WIDTH as i32])]
            }
            Self::Final => {
                let multi = p.reshape(x, &[1, 1, (STREAMS * WIDTH) as i32]);
                let h = mean(&mut p, multi);
                vec![rms(&mut p, h, V::Input(1)), multi]
            }
            Self::MeanNorm => {
                let h = mean(&mut p, x);
                vec![rms(&mut p, h, V::Input(1))]
            }
            Self::Rms => vec![rms(&mut p, x, V::Input(1))],
            Self::AddNorm => p.graph.call(
                K::SparseF32(S::OrderedAddRms {
                    epsilon: 1e-5f32.to_bits(),
                }),
                vec![x, V::Input(1), V::Input(2)],
            ),
            Self::AddThree => vec![p.graph.one(
                K::SparseF32(S::OrderedAddThree),
                vec![x, V::Input(1), V::Input(2)],
            )],
            Self::JoinProject(projection) => {
                let x = p.concat(vec![x, V::Input(1)], 2);
                vec![project(
                    &mut p,
                    *projection,
                    &[1, 1, (2 * WIDTH) as i32],
                    x,
                    V::Input(2),
                )?]
            }
            Self::Head { shape, projection } => {
                anyhow::ensure!(
                    !shape.is_empty()
                        && shape.last() == Some(&(WIDTH as i32))
                        && shape.iter().all(|&n| n > 0),
                    "invalid GLM head shape"
                );
                let rows = shape[..shape.len() - 1]
                    .iter()
                    .try_fold(1i32, |a, &b| a.checked_mul(b))
                    .ok_or_else(|| anyhow::anyhow!("head rows overflow"))?;
                let x = p.reshape(x, &[1, rows, WIDTH as i32]);
                let logits = project(
                    &mut p,
                    *projection,
                    &[1, rows, WIDTH as i32],
                    x,
                    V::Input(1),
                )?;
                let mut out = shape.clone();
                *out.last_mut().unwrap() = VOCAB as i32;
                vec![p.reshape(logits, &out), logits]
            }
            Self::Join { count } => vec![p.concat((0..*count).map(V::Input).collect(), 1)],
            Self::Reshape(shape) => vec![p.reshape(x, shape)],
        };
        Ok(p.finish(out)?.graph)
    }
}
pub fn validate_feed(tokens: &[u32], position: usize, capacity: usize) -> bool {
    !tokens.is_empty()
        && tokens.iter().all(|&t| (t as usize) < VOCAB)
        && position
            .checked_add(tokens.len())
            .is_some_and(|n| n <= capacity)
}
pub fn retain_row(row: usize, count: usize, all: bool) -> bool {
    all || row + 1 == count
}
pub fn block_weight(layer: usize, name: &str) -> String {
    format!("blk.{layer}.{name}.weight")
}
pub fn hc_weights(layer: usize, half: &str) -> [(String, String); 3] {
    ["fn", "base", "scale"].map(|suffix| {
        (
            format!("hc_{suffix}"),
            block_weight(layer, &format!("hc_{half}_{suffix}")),
        )
    })
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn packed_tower_keeps_scalar_layer_and_draft_order() {
        let p = schedule();
        assert_eq!(p.len(), LAYERS * 9 + 2);
        assert_eq!(p[0], Step::ExpandEmbedding);
        assert_eq!(p.last(), Some(&Step::FinalNorm));
        for i in 0..LAYERS {
            assert_eq!(
                &p[1 + 9 * i..1 + 9 * (i + 1)],
                &[
                    Step::BeginLayer(i),
                    Step::AttentionInput,
                    Step::Attention,
                    Step::AttentionResidual,
                    Step::FfnInput,
                    Step::Ffn,
                    Step::FfnResidual,
                    Step::CaptureLayer,
                    Step::EndLayer
                ]
            );
        }
        assert!(validate_feed(&[1, 2], 7, 9));
        assert!(!validate_feed(&[VOCAB as u32], 0, 9));
        assert!(!validate_feed(&[1, 2], 8, 9));
        assert!(!validate_feed(&[1], usize::MAX, usize::MAX));
        assert!(retain_row(2, 3, false));
        assert!(!retain_row(1, 3, false));
        assert!(retain_row(1, 3, true));
    }
}
