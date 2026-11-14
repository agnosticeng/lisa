//! Tower schema, traversal, host context law and input/readout graphs.
use super::config::ModelConfig;
use crate::{
    graph::{Graph, Value as V},
    program::Build,
    vision::prompt::Splice,
};
use lisa_backend::{Kernel, ScalarType as D, TensorOperation as T};
pub const PREFILL_CHUNK: usize = 2048;
pub const NGRAM_PREFIX: &str = "model.layers.1.ple.ple_embedding.ngram_embedding";
pub const EMBEDDING: &str = "model.embed_tokens";
pub const HEAD: &str = "lm_head";
pub const FINAL_MIXER: &str = "model.hyper_connection_mixer";
#[derive(Clone, Copy, Debug)]
pub enum Step {
    Begin,
    Layer(usize),
    Submit,
    Finish,
}
pub fn traversal(layers: usize, rows: usize) -> Vec<Step> {
    let mut v = vec![Step::Begin];
    for i in 0..layers {
        v.push(Step::Layer(i));
        if rows > 1 && (i + 1) % 3 == 0 {
            v.push(Step::Submit);
        }
    }
    v.push(Step::Finish);
    v
}
pub struct Layer {
    pub linear: bool,
    pub mixer: String,
    pub experts: String,
    pub attn_hyper: String,
    pub mlp_hyper: String,
    pub ple: String,
    pub ple_ordinal: Option<usize>,
}
pub fn layer(prefix: &str, linear: bool, ple_ordinal: Option<usize>) -> Layer {
    Layer {
        linear,
        mixer: format!(
            "{prefix}.{}",
            if linear { "linear_attn" } else { "self_attn" }
        ),
        experts: format!("{prefix}.mlp"),
        attn_hyper: format!("{prefix}.attn_hyper_connection"),
        mlp_hyper: format!("{prefix}.mlp_hyper_connection"),
        ple: format!("{prefix}.ple"),
        ple_ordinal,
    }
}
pub fn layers(c: &ModelConfig) -> Vec<Layer> {
    let ple = c.ple_layer_indices();
    (0..c.num_hidden_layers)
        .map(|i| {
            layer(
                &format!("model.layers.{i}"),
                c.layer_types[i] != "full_attention",
                ple.iter().position(|&p| p == i),
            )
        })
        .collect()
}
#[derive(Clone, Copy, Debug)]
pub enum Cache {
    Full { heads: usize, width: usize },
    Linear,
}
pub fn caches(c: &ModelConfig) -> Vec<Cache> {
    c.layer_types
        .iter()
        .map(|t| {
            if t == "full_attention" {
                Cache::Full {
                    heads: c.num_key_value_heads,
                    width: c.head_dim,
                }
            } else {
                Cache::Linear
            }
        })
        .collect()
}
pub fn kv_bytes(c: &ModelConfig) -> usize {
    caches(c)
        .iter()
        .filter(|c| matches!(c, Cache::Full { .. }))
        .count()
        * c.num_key_value_heads
        * c.head_dim
        * 2
        * 2
}
pub fn cost_key(c: &ModelConfig) -> String {
    format!("qwen4_exp-{}L-{}h", c.num_hidden_layers, c.hidden_size)
}
pub fn token_rows(flat: &[i32], b: usize, s: usize) -> anyhow::Result<Vec<Vec<i64>>> {
    anyhow::ensure!(
        b > 0 && s > 0 && b.checked_mul(s) == Some(flat.len()),
        "invalid token row geometry"
    );
    Ok(flat
        .chunks_exact(s)
        .map(|row| row.iter().map(|&v| v as i64).collect())
        .collect())
}
pub fn history(
    rows: &[Vec<i64>],
    previous: Option<&[Vec<i64>]>,
    ngram: usize,
    eos: i64,
) -> anyhow::Result<(Vec<Vec<i64>>, Vec<Vec<i64>>)> {
    anyhow::ensure!(ngram > 0, "invalid n-gram width");
    let n = ngram - 1;
    let previous = previous
        .map(|p| p.to_vec())
        .unwrap_or_else(|| vec![vec![eos; n]; rows.len()]);
    anyhow::ensure!(
        previous.len() == rows.len() && previous.iter().all(|r| r.len() >= n),
        "invalid n-gram history"
    );
    let next = previous
        .iter()
        .zip(rows)
        .map(|(old, new)| {
            let mut h = old.clone();
            h.extend(new);
            h[h.len() - n..].to_vec()
        })
        .collect();
    Ok((previous, next))
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Entry {
    pub embedding: (i32, i32),
    pub streams: i32,
    pub splice: Option<Splice>,
}
impl Entry {
    pub fn graph(&self) -> anyhow::Result<Graph> {
        let mut g =
            crate::graph::Builder::new(4 + self.splice.as_ref().map_or(0, |s| s.spans.len()), 0, 0);
        let mut x = g
            .append(
                crate::embedding::affine(self.embedding.0, self.embedding.1),
                &(0..4).map(V::Input).collect::<Vec<_>>(),
                &[],
                &[],
            )
            .map_err(anyhow::Error::msg)?[0];
        if let Some(s) = &self.splice {
            let mut inputs = vec![x];
            inputs.extend((4..4 + s.spans.len()).map(V::Input));
            x = g
                .append(s.graph()?, &inputs, &[], &[])
                .map_err(anyhow::Error::msg)?[0];
        }
        let x = g.one(Kernel::Tensor(T::Tile(vec![1, 1, self.streams])), vec![x]);
        Ok(g.finish(vec![x]))
    }
}
pub fn shortlist(vocab: usize) -> Option<Graph> {
    let (low, special) = (98304.min(vocab), 248044.min(vocab));
    if special <= low {
        return None;
    }
    let mut ids = (0..low as i32)
        .chain(special as i32..vocab as i32)
        .collect::<Vec<_>>();
    while ids.len() % 8 != 0 {
        ids.push(0)
    }
    let mut g = crate::graph::Builder::new(3, 0, 0);
    let index = g.constant_i32(&ids, &[ids.len() as i32]);
    let mut out = (0..3)
        .map(|i| {
            g.one(
                Kernel::Tensor(T::Take { axis: 0 }),
                vec![V::Input(i), index],
            )
        })
        .collect::<Vec<_>>();
    let map = g.constant(
        ids.iter().flat_map(|&i| (i as u32).to_le_bytes()).collect(),
        vec![ids.len() as i32],
        D::U32,
    );
    out.push(map);
    Some(g.finish(out))
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Readout {
    pub rows: i32,
    pub multi_rows: i32,
    pub quant: (i32, i32),
    pub shortlist: bool,
}
impl Readout {
    pub fn graph(self) -> anyhow::Result<Graph> {
        let schema = Default::default();
        let mut p = Build::new(if self.shortlist { 6 } else { 5 }, &schema);
        let last = p.op(
            T::Select {
                axis: 1,
                index: self.rows - 1,
            },
            vec![V::Input(0)],
        );
        let logits = p.graph.one(
            Kernel::AffineLinear {
                group_size: self.quant.0,
                bits: self.quant.1,
            },
            vec![last, V::Input(2), V::Input(3), V::Input(4)],
        );
        let mut id = p.op(
            T::Argmax {
                axis: -1,
                keep_dims: false,
            },
            vec![logits],
        );
        if self.shortlist {
            id = p.op(T::Take { axis: 0 }, vec![V::Input(5), id]);
        }
        let id = p.reshape(id, &[1, 1]);
        let multi = p.op(
            T::Select {
                axis: 1,
                index: self.multi_rows - 1,
            },
            vec![V::Input(1)],
        );
        let multi = p.contiguous(multi);
        Ok(p.finish(vec![id, multi, logits])?.graph)
    }
}
pub fn warmup(seed: &[u32]) -> Vec<Vec<i32>> {
    let fallback = (0..64).map(|i| 128 + (i * 37) % 4096).collect::<Vec<_>>();
    let base = if seed.is_empty() { &fallback } else { seed };
    [1, 9, 10, 40, PREFILL_CHUNK]
        .into_iter()
        .map(|s| (0..s).map(|i| base[i % base.len()] as i32).collect())
        .collect()
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn context_and_submission_laws_are_explicit() {
        let (previous, next) = history(&[vec![7], vec![8]], None, 3, 99).unwrap();
        assert_eq!(previous, vec![vec![99, 99]; 2]);
        assert_eq!(next, vec![vec![99, 7], vec![99, 8]]);
        let (_, next) = history(&[vec![1, 2, 3]], Some(&[vec![99, 7]]), 3, 99).unwrap();
        assert_eq!(next, vec![vec![2, 3]]);
        assert!(history(&[vec![1]], Some(&[]), 3, 99).is_err());
        assert!(token_rows(&[1, 2], 1, 3).is_err());
        assert_eq!(
            traversal(7, 1)
                .iter()
                .filter(|s| matches!(s, Step::Submit))
                .count(),
            0
        );
        assert_eq!(
            traversal(7, 8)
                .iter()
                .filter(|s| matches!(s, Step::Submit))
                .count(),
            2
        );
        assert!(shortlist(32768).is_none());
        shortlist(248320).unwrap().validate().unwrap();
    }
}
