//! The complete parallel draft block, including all five bridge writes.
use super::{
    bridge::{Bridge, Utility},
    layers::{Attention, Tail},
};
use crate::graph::{Builder, Graph, Value};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Block {
    pub attention: Vec<Attention>,
    pub intermediate: i32,
    pub epsilon: u32,
    pub bridge_epsilon: u32,
    pub rank: i32,
    pub codes: i32,
}
impl Block {
    pub fn graph(&self) -> anyhow::Result<Graph> {
        anyhow::ensure!(
            self.attention.len() == 5,
            "DSpine requires five draft layers"
        );
        let first = &self.attention[0];
        let (b, t, h) = (first.cache.batch, first.cache.rows, first.hidden);
        let count = if first.quantized { 3 } else { 1 };
        let per_layer = 7 * count + 4;
        // Each layer: input norm, Q/K/V/O, Q/K norm, post norm, gate/up/down.
        // Shared: final norm, final read, early reads, codebook, packed, bias, writes.
        let shared = 2 + 5 * per_layer;
        let mut cache_slot = shared + 7;
        let inputs = cache_slot
            + self
                .attention
                .iter()
                .filter(|a| a.cache.previous_batch.is_some())
                .count()
                * 2;
        let mut g = Builder::new(inputs, 0, 0);
        let start = g
            .append(
                Utility::Start(self.epsilon).graph()?,
                &[Value::Input(0), Value::Input(2)],
                &[],
                &[],
            )
            .map_err(anyhow::Error::msg)?;
        let (mut x, mut residual) = (start[0], start[1]);
        let mut stash = Vec::new();
        let mut caches = Vec::new();
        for (layer, attention) in self.attention.iter().enumerate() {
            anyhow::ensure!(
                !attention.context_only
                    && attention.quantized == first.quantized
                    && attention.cache.batch == b
                    && attention.cache.rows == t
                    && attention.hidden == h,
                "inconsistent draft block layer geometry"
            );
            let base = 2 + layer * per_layer;
            let mut args = vec![x];
            args.extend((base + 1..base + 1 + 4 * count + 2).map(Value::Input));
            if attention.cache.previous_batch.is_some() {
                args.extend([Value::Input(cache_slot), Value::Input(cache_slot + 1)]);
                cache_slot += 2;
            }
            let result = g
                .append(attention.graph()?, &args, &[], &[])
                .map_err(anyhow::Error::msg)?;
            caches.extend_from_slice(&result[1..]);
            let tail = Tail {
                shape: vec![b, t, h],
                intermediate: self.intermediate,
                quantized: attention.quantized,
                small_rows: attention.small_rows,
                epsilon: self.epsilon,
            };
            let mut args = vec![residual, result[0]];
            args.extend((base + 1 + 4 * count + 2..base + per_layer).map(Value::Input));
            let output = g
                .append(tail.graph()?, &args, &[], &[])
                .map_err(anyhow::Error::msg)?[0];
            let last = layer + 1 == self.attention.len();
            let next_norm = Value::Input(if last { shared } else { base + per_layer });
            let mut args = vec![output, Value::Input(1), next_norm];
            args.extend((shared + 1..shared + 7).map(Value::Input));
            let bridge = Bridge {
                shape: [b, t, h],
                layer,
                layers: 5,
                rank: self.rank,
                codes: self.codes,
                epsilon: self.bridge_epsilon,
                next_epsilon: self.epsilon,
                stash: last,
                small_rows: attention.small_rows,
            };
            let output = g
                .append(bridge.graph()?, &args, &[], &[])
                .map_err(anyhow::Error::msg)?;
            x = output[0];
            residual = output[1];
            if last {
                stash.extend_from_slice(&output[2..]);
            }
        }
        let rows = g
            .append(Utility::Finish(t).graph()?, &[x], &[], &[])
            .map_err(anyhow::Error::msg)?[0];
        let mut out = vec![rows];
        out.extend(stash);
        out.extend(caches);
        Ok(g.finish(out))
    }
}
pub fn input_ids(
    anchor: u32,
    mask: usize,
    rows: usize,
    trained_rows: usize,
) -> anyhow::Result<Vec<i32>> {
    anyhow::ensure!(
        rows >= 3 && rows <= trained_rows,
        "invalid DSpine serving block size"
    );
    let anchor = i32::try_from(anchor)?;
    let mask = i32::try_from(mask)?;
    let mut ids = vec![mask; rows];
    ids[0] = anchor;
    Ok(ids)
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Context {
    pub projection: super::layers::Linear,
    pub epsilon: u32,
    pub attention: Vec<Attention>,
}
impl Context {
    pub fn graph(&self) -> anyhow::Result<Graph> {
        anyhow::ensure!(
            self.attention.len() == 5,
            "DSpine context requires five layers"
        );
        let first = &self.attention[0];
        let weights = if first.quantized { 3 } else { 1 };
        let per_layer = 4 * weights + 2;
        let projection_inputs = 2 + self.projection.weights();
        let mut cache_slot = projection_inputs + 5 * per_layer;
        let inputs = cache_slot
            + 2 * self
                .attention
                .iter()
                .filter(|a| a.cache.previous_batch.is_some())
                .count();
        let mut g = Builder::new(inputs, 0, 0);
        let context = g
            .append(
                Utility::Context(self.projection.clone(), self.epsilon).graph()?,
                &(0..projection_inputs).map(Value::Input).collect::<Vec<_>>(),
                &[],
                &[],
            )
            .map_err(anyhow::Error::msg)?[0];
        let mut out = Vec::new();
        for (layer, attention) in self.attention.iter().enumerate() {
            anyhow::ensure!(
                attention.context_only
                    && attention.quantized == first.quantized
                    && attention.hidden == self.projection.output,
                "invalid draft context layer"
            );
            let base = projection_inputs + layer * per_layer;
            let mut args = vec![context];
            args.extend((base..base + per_layer).map(Value::Input));
            if attention.cache.previous_batch.is_some() {
                args.extend([Value::Input(cache_slot), Value::Input(cache_slot + 1)]);
                cache_slot += 2;
            }
            out.extend(
                g.append(attention.graph()?, &args, &[], &[])
                    .map_err(anyhow::Error::msg)?,
            );
        }
        Ok(g.finish(out))
    }
}
