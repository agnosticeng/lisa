//! Native draft-head computation, including its decoder and cache update.
use crate::graph::{Builder, Graph, Value};
use lisa_backend::{Kernel, ScalarType, TensorOperation as T};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Plan {
    pub attention: super::attention::Plan,
    pub embedding: (i32, i32),
    pub fc: (i32, i32),
    pub dense: [(i32, i32); 3],
    pub dense_joined: Option<(i32, i32)>,
    /// Embedding, previous hidden, input, post-attention, final normalization.
    pub epsilon: [u32; 5],
    pub position_offset: usize,
    pub rotary_base: u32,
}
impl Plan {
    /// Inputs: token IDs, previous hidden, packed embedding triple, old K/V.
    /// Weights: pre-norm pair, FC triple, input norm, attention, post norm,
    /// dense MLP and final norm. Outputs: hidden, last hidden row, K/V storage.
    pub fn graph(&self) -> anyhow::Result<Graph> {
        let a = &self.attention;
        anyhow::ensure!(
            a.cache.is_some() && a.mask_length.is_none() && a.next_positions.is_none(),
            "native draft cache must be an unpacked stream"
        );
        let attention = a.graph().map_err(anyhow::Error::msg)?;
        let alternatives = crate::dense::swiglu(self.dense, self.dense_joined);
        let dense = if a.rows == 1 {
            alternatives
                .guarded
                .map(|(_, p)| p)
                .unwrap_or(alternatives.primary)
        } else {
            alternatives.primary
        };
        let dense = dense
            .graph(if self.dense_joined.is_some() { 12 } else { 9 })
            .map_err(anyhow::Error::msg)?;
        let nw = attention.weights;
        let nd = dense.weights;
        let has_old = a.cache.as_ref().unwrap().previous_batch.is_some();
        let mut g = Builder::new(if has_old { 7 } else { 5 }, 8 + nw + nd, 0);
        let mut rows = Vec::new();
        for table in 2..5 {
            rows.push(g.one(
                Kernel::Tensor(T::Take { axis: 0 }),
                vec![Value::Input(table), Value::Input(0)],
            ));
        }
        let embed = g.one(
            Kernel::AffineDequantize {
                group_size: self.embedding.0,
                bits: self.embedding.1,
            },
            rows,
        );
        let e = g.one(
            Kernel::Tensor(T::RmsNorm {
                epsilon: self.epsilon[0],
            }),
            vec![embed, Value::Weight(0)],
        );
        let h = g.one(
            Kernel::Tensor(T::RmsNorm {
                epsilon: self.epsilon[1],
            }),
            vec![Value::Input(1), Value::Weight(1)],
        );
        let cat = g.one(
            Kernel::Tensor(T::Concatenate {
                axis: -1,
                inputs: 2,
            }),
            vec![e, h],
        );
        let x = g.one(
            Kernel::AffineLinear {
                group_size: self.fc.0,
                bits: self.fc.1,
            },
            vec![cat, Value::Weight(2), Value::Weight(3), Value::Weight(4)],
        );
        let end = self
            .position_offset
            .checked_add(a.rows as usize)
            .ok_or_else(|| anyhow::anyhow!("draft position overflow"))?;
        let positions = (self.position_offset..end)
            .map(|i| i as f32)
            .collect::<Vec<_>>();
        let positions = g.constant_f32(&positions, &[1, a.rows]);
        let frequencies = crate::position::frequencies(
            a.rotary as usize,
            f32::from_bits(self.rotary_base),
            &[1, a.rows],
            false,
        )?;
        let frequencies = g
            .append(frequencies, &[positions], &[], &[])
            .map_err(anyhow::Error::msg)?;
        // Preserve the original two reshapes followed by the two BF16 casts.
        let cos = g.one(
            Kernel::Tensor(T::Reshape(vec![1, 1, a.rows, a.rotary])),
            vec![frequencies[0]],
        );
        let sin = g.one(
            Kernel::Tensor(T::Reshape(vec![1, 1, a.rows, a.rotary])),
            vec![frequencies[1]],
        );
        let cos = g.one(Kernel::Tensor(T::Cast(ScalarType::BF16)), vec![cos]);
        let sin = g.one(Kernel::Tensor(T::Cast(ScalarType::BF16)), vec![sin]);
        let normalized = g.one(
            Kernel::Tensor(T::RmsNorm {
                epsilon: self.epsilon[2],
            }),
            vec![x, Value::Weight(5)],
        );
        let mut inputs = vec![normalized, cos, sin];
        if has_old {
            inputs.extend([Value::Input(5), Value::Input(6)]);
        }
        let attn = g
            .append(
                attention,
                &inputs,
                &(6..6 + nw).map(Value::Weight).collect::<Vec<_>>(),
                &[],
            )
            .map_err(anyhow::Error::msg)?;
        let sum_norm = g.call(
            Kernel::Tensor(T::AddRmsNorm {
                epsilon: self.epsilon[3],
            }),
            vec![x, attn[0], Value::Weight(6 + nw)],
        );
        let mlp = g
            .append(
                dense,
                &[sum_norm[1]],
                &(7 + nw..7 + nw + nd).map(Value::Weight).collect::<Vec<_>>(),
                &[],
            )
            .map_err(anyhow::Error::msg)?[0];
        let out = g.call(
            Kernel::Tensor(T::AddRmsNorm {
                epsilon: self.epsilon[4],
            }),
            vec![sum_norm[0], mlp, Value::Weight(7 + nw + nd)],
        )[1];
        let last = g.one(
            Kernel::Tensor(T::Select {
                axis: 1,
                index: a.rows - 1,
            }),
            vec![out],
        );
        let last = g.one(Kernel::Tensor(T::Contiguous), vec![last]);
        Ok(g.finish(vec![out, last, attn[1], attn[2]]))
    }
}
