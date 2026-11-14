//! Qwen4 attention computation and sparse/dense mask policy.
use crate::{
    attention_cache::Append,
    graph::{Graph, Value as V},
    program::Build,
};
use anyhow::{Result, ensure};
use lisa_backend::{
    AttentionMask, Binary as B, BlockAttention as A, Comparison as C, Kernel, ScalarType as D,
    TensorOperation as T,
};
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Selection {
    pub width: i32,
    pub ratio: i32,
    pub length: i32,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Plan {
    pub batch: i32,
    pub rows: i32,
    pub heads: i32,
    pub key_heads: i32,
    pub width: i32,
    pub rotary: usize,
    pub base: u32,
    pub epsilon: u32,
    pub scale: u32,
    pub quant: [(i32, i32); 4],
    pub position_shape: Vec<i32>,
    pub offset: usize,
    pub cache: Option<Append>,
    pub ragged: bool,
    pub selection: Option<Selection>,
    pub block_kernel: bool,
}
pub fn use_indexer_tape(ragged: bool, indexer_ok: bool) -> bool {
    !ragged && indexer_ok
}
pub fn retile(heads: i32, width: i32) -> Graph {
    let mut g = crate::graph::Builder::new(3, 0, 0);
    let order = (0..2)
        .flat_map(|part| {
            (0..heads).flat_map(move |h| {
                (0..width).map(move |j| (h * 2 * width + part * width + j) as u32)
            })
        })
        .collect::<Vec<_>>();
    let index = g.constant(
        order.iter().flat_map(|i| i.to_le_bytes()).collect(),
        vec![heads * 2 * width],
        D::U32,
    );
    let outputs = (0..3)
        .map(|i| {
            g.one(
                Kernel::Tensor(T::Take { axis: 0 }),
                vec![V::Input(i), index],
            )
        })
        .collect::<Vec<_>>();
    g.finish(outputs)
}

impl Plan {
    /// Inputs: activation, positions, Q/K/V/O weight triples, Q/K scales,
    /// optional previous cache K/V, optional ragged mask, optional selection
    /// IDs/valid/completed-blocks/query-positions. Outputs: projected activation
    /// and optional cache storage. The graph owns all arithmetic and ordering.
    pub fn graph(&self) -> Result<Graph> {
        let (b, s, h, kh, d) = (
            self.batch,
            self.rows,
            self.heads,
            self.key_heads,
            self.width,
        );
        ensure!(
            [b, s, h, kh, d].iter().all(|&v| v > 0)
                && h % kh == 0
                && self
                    .offset
                    .checked_add(s as usize)
                    .is_some_and(|n| n <= i32::MAX as usize),
            "invalid attention geometry"
        );
        let old = self
            .cache
            .as_ref()
            .is_some_and(|c| c.previous_batch.is_some());
        let mask_slot = 16 + if old { 2 } else { 0 };
        let sparse_slot = mask_slot + usize::from(self.ragged);
        let schema = Default::default();
        let mut p = Build::new(
            sparse_slot + if self.selection.is_some() { 4 } else { 0 },
            &schema,
        );
        let linear = |p: &mut Build<'_>, x, slot: usize, quant: (i32, i32)| {
            p.graph.one(
                Kernel::AffineLinear {
                    group_size: quant.0,
                    bits: quant.1,
                },
                vec![x, V::Input(slot), V::Input(slot + 1), V::Input(slot + 2)],
            )
        };
        let projected = linear(&mut p, V::Input(0), 2, self.quant[0]);
        let q = p.slice(projected, 2, 0, h * d);
        let q = p.contiguous(q);
        let gate = p.slice(projected, 2, h * d, 2 * h * d);
        let gate = p.contiguous(gate);
        let gate = p.reshape(gate, &[b, s, h * d]);
        let cs = p
            .graph
            .append(
                crate::position::frequencies(
                    self.rotary,
                    f32::from_bits(self.base),
                    &self.position_shape,
                    self.position_shape.len() == 3,
                )?,
                &[V::Input(1)],
                &[],
                &[],
            )
            .map_err(anyhow::Error::msg)?;
        let k = linear(&mut p, V::Input(0), 5, self.quant[1]);
        let v = linear(&mut p, V::Input(0), 8, self.quant[2]);
        let cos = p.cast(cs[0], D::BF16);
        let sin = p.cast(cs[1], D::BF16);
        let qkv = p.graph.call(
            Kernel::BlockAttention(A::Prepare {
                heads: h,
                key_heads: kh,
                width: d,
                rotary: self.rotary as i32,
                epsilon: self.epsilon,
            }),
            vec![q, k, v, V::Input(14), V::Input(15), cos, sin],
        );
        let q = qkv[0];
        let mut k = qkv[1];
        let mut v = qkv[2];
        let mut storage = vec![];
        if let Some(cache) = &self.cache {
            ensure!(
                (
                    cache.batch,
                    cache.rows,
                    cache.heads,
                    cache.width,
                    cache.dtype
                ) == (b, s, kh, d, D::BF16),
                "attention cache geometry differs"
            );
            let mut inputs = vec![k, v];
            if old {
                inputs.extend([V::Input(16), V::Input(17)]);
            }
            let output = p
                .graph
                .append(
                    cache.graph().map_err(anyhow::Error::msg)?,
                    &inputs,
                    &[],
                    &[],
                )
                .map_err(anyhow::Error::msg)?;
            storage.extend([output[0], output[1]]);
            k = output[2];
            v = output[3];
        }
        let out = if self.ragged {
            p.op(
                T::Attention {
                    scale: self.scale,
                    mask: AttentionMask::Tensor,
                },
                vec![q, k, v, V::Input(mask_slot)],
            )
        } else if let Some(selection) = &self.selection {
            let total = (self.offset + s as usize) as i32;
            if self.block_kernel
                && b == 1
                && h == 24
                && kh == 2
                && d == 256
                && self.offset >= 2048
                && selection.width == 512
            {
                let ids = p.reshape(V::Input(sparse_slot), &[s, 512]);
                let valid = p.reshape(V::Input(sparse_slot + 1), &[s, 512]);
                p.graph.one(
                    Kernel::BlockAttention(A::Attend {
                        offset: self.offset as i32,
                        total,
                        scale: self.scale,
                    }),
                    vec![q, k, v, ids, valid],
                )
            } else {
                ensure!(b == 1, "dense block-mask contract requires one stream");
                let keep = dense_keep(&mut p, s, selection, sparse_slot)?;
                let right = p
                    .graph
                    .constant_f32(&(0..total).map(|i| i as f32).collect::<Vec<_>>(), &[total]);
                let left = p.graph.constant_f32(
                    &((total - s)..total).map(|i| i as f32).collect::<Vec<_>>(),
                    &[s, 1],
                );
                let left = p.op(T::Broadcast(vec![s, total]), vec![left]);
                let left = p.contiguous(left);
                let right = p.reshape(right, &[1, total]);
                let right = p.op(T::Broadcast(vec![s, total]), vec![right]);
                let right = p.contiguous(right);
                let causal = p.op(T::Compare(C::GreaterEqual), vec![left, right]);
                let causal = p.reshape(causal, &[1, s, total]);
                let mask = p.op(T::LogicalAnd, vec![causal, keep]);
                p.op(
                    T::Attention {
                        scale: self.scale,
                        mask: AttentionMask::Tensor,
                    },
                    vec![q, k, v, mask],
                )
            }
        } else {
            p.op(
                T::Attention {
                    scale: self.scale,
                    mask: AttentionMask::Causal,
                },
                vec![q, k, v],
            )
        };
        let out = p.transpose(out, &[0, 2, 1, 3]);
        let out = p.reshape(out, &[b, s, h * d]);
        let gated = p.graph.one(
            Kernel::BlockAttention(A::Gate { heads: h, width: d }),
            vec![out, gate],
        );
        let output = linear(&mut p, gated, 11, self.quant[3]);
        let mut outputs = vec![output];
        outputs.extend(storage);
        Ok(p.finish(outputs)?.graph)
    }
}
fn dense_keep(p: &mut Build<'_>, s: i32, sel: &Selection, slot: usize) -> Result<V> {
    ensure!(
        sel.ratio > 0 && sel.length > 0 && sel.width > 0,
        "invalid sparse selection geometry"
    );
    let (kv, cr) = (sel.length, sel.ratio);
    let blocks = kv / cr;
    let sentinel = p.scalar(blocks as f32, D::F32);
    let sentinel = p.op(T::Broadcast(vec![1, s, sel.width]), vec![sentinel]);
    let ids = p.cast(V::Input(slot), D::F32);
    let picked = p.op(T::Where, vec![V::Input(slot + 1), ids, sentinel]);
    let picked = p.cast(picked, D::I32);
    let zeros = p.graph.one(
        Kernel::Zeros {
            shape: vec![1, s, blocks + 1],
            dtype: D::U8,
        },
        vec![],
    );
    let yes = p.graph.constant(vec![1], vec![], D::U8);
    let updates = p.op(T::Broadcast(vec![1, s, sel.width]), vec![yes]);
    let keep = p.op(T::PutAlong { axis: -1 }, vec![zeros, picked, updates]);
    let keep = p.slice(keep, -1, 0, blocks);
    let mut keep = p.op(
        T::Repeat {
            count: cr as usize,
            axis: -1,
        },
        vec![keep],
    );
    if kv % cr > 0 {
        let zeros = p.graph.one(
            Kernel::Zeros {
                shape: vec![1, s, kv % cr],
                dtype: D::U8,
            },
            vec![],
        );
        keep = p.concat(vec![keep, zeros], -1);
    }
    let ratio = p.scalar(cr as f32, D::F32);
    let own = p.binary(B::Multiply, V::Input(slot + 2), ratio);
    let tokens = p
        .graph
        .constant_f32(&(0..kv).map(|i| i as f32).collect::<Vec<_>>(), &[1, 1, kv]);
    let positions = p.graph.constant_f32(
        &((kv - s)..kv).map(|i| i as f32).collect::<Vec<_>>(),
        &[1, s],
    );
    let tokens = p.op(T::Broadcast(vec![1, s, kv]), vec![tokens]);
    let own = p.reshape(own, &[1, s, 1]);
    let own = p.op(T::Broadcast(vec![1, s, kv]), vec![own]);
    let positions = p.reshape(positions, &[1, s, 1]);
    let positions = p.op(T::Broadcast(vec![1, s, kv]), vec![positions]);
    let lower = p.op(T::Compare(C::GreaterEqual), vec![tokens, own]);
    let upper = p.op(T::Compare(C::LessEqual), vec![tokens, positions]);
    let own = p.op(T::LogicalAnd, vec![lower, upper]);
    Ok(p.op(T::LogicalOr, vec![keep, own]))
}
pub struct Names {
    pub projections: [&'static str; 4],
    pub q_norm: String,
    pub k_norm: String,
    pub indexer: String,
}
pub fn names(prefix: &str) -> Names {
    Names {
        projections: ["q_proj", "k_proj", "v_proj", "o_proj"],
        q_norm: format!("{prefix}.q_norm"),
        k_norm: format!("{prefix}.k_norm"),
        indexer: format!("{prefix}.indexer"),
    }
}

pub fn scale(width: usize) -> f32 {
    (width as f32).powf(-0.5)
}
