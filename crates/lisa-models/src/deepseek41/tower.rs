//! Layer-major scalar execution schedule and tensor assembly for V4.1.
use super::{engram, state};
use crate::graph::{Builder, Graph, Value as V};
use lisa_backend::{Kernel as K, TensorOperation as T};
pub const WIDTH: usize = 5120;
pub const STREAMS: usize = 4;
pub const PREFILL_CHUNK: usize = 8;
pub const DEFAULT_CONTEXT: usize = 4096;
pub const NUMERICAL_IDENTITY: &[u8] = b"lisa-ds41-gguf-hc-stable-fused-rope-v3";
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Step {
    Layer(usize),
    EngramProject(usize),
    EngramRow(usize),
    AttentionInput(usize),
    AttentionProject,
    Attention(usize),
    AttentionResidual(usize),
    FfnInput(usize),
    Experts,
    FfnResidual(usize),
    Submit { wait: bool },
    EndLayer,
    Head { row: usize, logits: bool },
}
pub fn schedule(layers: usize, rows: usize, logits: bool) -> anyhow::Result<Vec<Step>> {
    anyhow::ensure!(
        layers == state::LAYERS && (1..=PREFILL_CHUNK).contains(&rows),
        "unsupported scalar tower geometry"
    );
    let mut out = Vec::new();
    for layer in 0..layers {
        out.push(Step::Layer(layer));
        if let Some(table) = engram::LAYERS.iter().position(|&i| i == layer) {
            out.push(Step::EngramProject(table));
            out.extend((0..rows).map(Step::EngramRow));
        }
        out.extend((0..rows).map(Step::AttentionInput));
        out.push(Step::AttentionProject);
        for row in 0..rows {
            out.push(Step::Attention(row));
            out.push(Step::AttentionResidual(row));
            out.push(Step::FfnInput(row));
        }
        out.push(Step::Experts);
        out.extend((0..rows).map(Step::FfnResidual));
        if rows > 1 {
            out.push(Step::Submit { wait: true });
        } else if layer + 1 < layers && (layer == 0 || (layer + 1) % 2 == 0) {
            out.push(Step::Submit { wait: false });
        }
        out.push(Step::EndLayer);
    }
    out.extend((0..rows).map(|row| Step::Head {
        row,
        logits: logits && row + 1 == rows,
    }));
    Ok(out)
}
pub fn validate_tokens(tokens: &[u32], vocab: usize, position: usize, capacity: usize) -> bool {
    !tokens.is_empty()
        && tokens.len() <= PREFILL_CHUNK
        && position
            .checked_add(tokens.len())
            .is_some_and(|n| n <= capacity)
        && tokens.iter().all(|&t| (t as usize) < vocab)
}
pub fn embedding(row: &[f32]) -> anyhow::Result<Vec<f32>> {
    anyhow::ensure!(row.len() == WIDTH, "invalid embedding width");
    Ok((0..STREAMS).flat_map(|_| row.iter().copied()).collect())
}
pub const INITIAL_MIX: [f32; 4] = [1., 0., 0., 0.];
pub fn join(count: usize) -> Graph {
    let mut b = Builder::new(count, 0, 0);
    let x = if count == 1 {
        V::Input(0)
    } else {
        b.one(
            K::Tensor(T::Concatenate {
                axis: 0,
                inputs: count,
            }),
            (0..count).map(V::Input).collect::<Vec<_>>(),
        )
    };
    b.finish(vec![x])
}
pub fn reshape(shape: Vec<i32>) -> Graph {
    let mut b = Builder::new(1, 0, 0);
    let x = b.one(K::Tensor(T::Reshape(shape)), vec![V::Input(0)]);
    b.finish(vec![x])
}
pub fn concatenate_output(count: usize, width: i32) -> Graph {
    let mut b = Builder::new(count, 0, 0);
    let x = if count == 1 {
        V::Input(0)
    } else {
        b.one(
            K::Tensor(T::Concatenate {
                axis: 0,
                inputs: count,
            }),
            (0..count).map(V::Input).collect::<Vec<_>>(),
        )
    };
    let x = b.one(K::Tensor(T::Reshape(vec![1, count as i32, width])), vec![x]);
    b.finish(vec![x])
}
pub fn ids() -> Graph {
    let mut b = Builder::new(1, 0, 0);
    let x = b.one(
        K::Tensor(T::Cast(lisa_backend::ScalarType::U32)),
        vec![V::Input(0)],
    );
    b.finish(vec![x])
}
pub fn token_shape(shape: &[i32]) -> bool {
    shape.len() == 2 && shape[0] == 1 && shape[1] > 0
}
pub fn head(matrix: super::projection::Matrix, shape: &[i32]) -> anyhow::Result<Graph> {
    anyhow::ensure!(
        shape.last() == Some(&(WIDTH as i32)),
        "invalid output head geometry"
    );
    let count = shape[..shape.len() - 1]
        .iter()
        .try_fold(1i32, |a, &b| a.checked_mul(b))
        .filter(|&n| n > 0)
        .ok_or_else(|| anyhow::anyhow!("invalid output head row count"))?;
    let mut b = Builder::new(2, 0, 0);
    let flat = b.one(K::Tensor(T::Contiguous), vec![V::Input(0)]);
    let flat = b.one(K::Tensor(T::Reshape(vec![-1, WIDTH as i32])), vec![flat]);
    let mut outputs = Vec::new();
    for row in 0..count {
        let x = b.one(
            K::Tensor(T::Slice {
                axis: 0,
                start: row,
                end: row + 1,
            }),
            vec![flat],
        );
        outputs.push(
            b.append(
                matrix.linear(0, matrix.output, false)?,
                &[V::Input(1), x],
                &[],
                &[],
            )
            .map_err(anyhow::Error::msg)?[0],
        );
    }
    let x = b.one(
        K::Tensor(T::Concatenate {
            axis: 0,
            inputs: outputs.len(),
        }),
        outputs,
    );
    let mut shape = shape.to_vec();
    *shape.last_mut().unwrap() = matrix.output;
    let x = b.one(K::Tensor(T::Reshape(shape)), vec![x]);
    Ok(b.finish(vec![x]))
}
pub fn kv_bytes_per_token(capacity: usize) -> usize {
    640 * 4 * 5 / 2 + if capacity >= 8192 { 128 * 2 * 5 / 2 } else { 0 }
}
pub struct EngramWeights {
    pub kv: String,
    pub query: String,
    pub key: Option<String>,
}
pub fn engram_weights(layer: usize, combined: bool) -> Option<EngramWeights> {
    engram::LAYERS.contains(&layer).then(|| EngramWeights {
        kv: format!("blk.{layer}.engram_kv.weight"),
        query: format!(
            "blk.{layer}.{}.weight",
            if combined {
                "engram_gate"
            } else {
                "engram_q_norm"
            }
        ),
        key: (!combined).then(|| format!("blk.{layer}.engram_k_norm.weight")),
    })
}
pub fn identity_gate() -> (Vec<f32>, Vec<i32>) {
    (
        vec![1.; STREAMS * WIDTH],
        vec![STREAMS as i32, WIDTH as i32],
    )
}
pub const OUTPUT_NORM: &str = "output_norm.weight";
pub const OUTPUT: &str = "output.weight";
pub const EMBEDDING: &str = "token_embd.weight";
pub const ATTENTION_HC: &str = "attn";
pub const FFN_HC: &str = "ffn";
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn layer_major_schedule_retains_causal_rows_and_submission_cadence() -> anyhow::Result<()> {
        for rows in [1, 3, 8] {
            let p = schedule(40, rows, true)?;
            let layers = p.iter().filter(|s| matches!(s, Step::Layer(_))).count();
            assert_eq!(layers, 40);
            assert_eq!(
                p.iter().filter(|s| matches!(s, Step::Attention(_))).count(),
                40 * rows
            );
            assert_eq!(
                p.iter()
                    .filter(|s| matches!(s, Step::Submit { wait: true }))
                    .count(),
                if rows == 1 { 0 } else { 40 }
            );
            assert_eq!(
                p.iter()
                    .filter(|s| matches!(s, Step::Submit { wait: false }))
                    .count(),
                if rows == 1 { 20 } else { 0 }
            );
            assert_eq!(
                p.last(),
                Some(&Step::Head {
                    row: rows - 1,
                    logits: true
                })
            );
            let mut layer = 0;
            for s in p {
                match s {
                    Step::Layer(i) => layer = i,
                    Step::EngramProject(table) => assert_eq!(engram::LAYERS[table], layer),
                    _ => {}
                }
            }
        }
        Ok(())
    }
}
