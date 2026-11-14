//! Optional coarse retrieval, exact original-weight rescoring and bounded
//! auxiliary-head construction. Target projection laws are unchanged.
use crate::{
    graph::{Graph, Value},
    program::Build,
};
use lisa_backend::{Kernel, Refinement, ScalarType, TensorOperation as T};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Rescore {
    pub rows: i32,
    pub count: i32,
    pub group: i32,
    pub bits: i32,
}
impl Rescore {
    /// Inputs: hidden, original packed/scales/biases, selected IDs.
    pub fn graph(&self) -> anyhow::Result<Graph> {
        anyhow::ensure!(
            (1..13).contains(&self.rows) && self.count > 0,
            "invalid retrieval rows"
        );
        let schema = Default::default();
        let mut p = Build::new(5, &schema);
        let flat = p.reshape(Value::Input(4), &[self.rows * self.count]);
        let mut args = vec![Value::Input(0)];
        for slot in 1..4 {
            args.push(p.op(T::Take { axis: 0 }, vec![Value::Input(slot), flat]));
        }
        let logits = p.graph.one(
            Kernel::AffineLinear {
                group_size: self.group,
                bits: self.bits,
            },
            args,
        );
        let mut parts = Vec::new();
        for row in 0..self.rows {
            let part = p.slice(logits, 0, row, row + 1);
            parts.push(p.slice(part, 1, row * self.count, (row + 1) * self.count));
        }
        let out = p.concat(parts, 0);
        Ok(p.finish(vec![out])?.graph)
    }
    /// Inputs: hidden, coarse payload triple, original payload triple.
    pub fn candidates(&self) -> anyhow::Result<Graph> {
        let schema = Default::default();
        let mut p = Build::new(7, &schema);
        let logits = p.graph.one(
            Kernel::AffineLinear {
                group_size: 64,
                bits: 2,
            },
            (0..4).map(Value::Input).collect::<Vec<_>>(),
        );
        let ids = p.graph.one(
            Kernel::Refinement(Refinement::Candidates { count: 128 }),
            vec![logits],
        );
        anyhow::ensure!(
            self.count == 128,
            "coarse retrieval shortlist must contain 128 IDs"
        );
        let score = p
            .graph
            .append(
                self.graph()?,
                &[
                    Value::Input(0),
                    Value::Input(4),
                    Value::Input(5),
                    Value::Input(6),
                    ids,
                ],
                &[],
                &[],
            )
            .map_err(anyhow::Error::msg)?[0];
        let local = p.graph.one(
            Kernel::Refinement(Refinement::Candidates { count: 16 }),
            vec![score],
        );
        let out = p.op(T::TakeAlong { axis: 1 }, vec![ids, local]);
        Ok(p.finish(vec![out])?.graph)
    }
}
pub fn supported(shape: &[i32]) -> bool {
    shape.len() == 2 && (1..13).contains(&shape[0])
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Head {
    pub vocab: i32,
    pub hidden: i32,
    pub source_group: i32,
    pub source_bits: i32,
}
impl Head {
    pub fn bytes(&self) -> anyhow::Result<usize> {
        anyhow::ensure!(
            self.vocab > 0 && self.hidden > 0 && self.hidden % 64 == 0,
            "unsupported coarse head geometry"
        );
        (self.vocab as usize)
            .checked_mul(self.hidden as usize / 4 + self.hidden as usize / 64 * 4)
            .ok_or_else(|| anyhow::anyhow!("coarse head size overflow"))
    }
    pub fn useful(&self) -> bool {
        self.source_bits > 2
    }
    pub fn allocate(&self) -> anyhow::Result<Graph> {
        self.bytes()?;
        let schema = Default::default();
        let mut p = Build::new(0, &schema);
        let out = [
            (self.hidden / 16, ScalarType::U32),
            (self.hidden / 64, ScalarType::BF16),
            (self.hidden / 64, ScalarType::BF16),
        ]
        .map(|(width, dtype)| {
            p.graph.one(
                Kernel::Zeros {
                    shape: vec![self.vocab, width],
                    dtype,
                },
                vec![],
            )
        });
        Ok(p.finish(out.to_vec())?.graph)
    }
    pub fn chunks(&self) -> impl Iterator<Item = (i32, i32)> {
        let vocab = self.vocab;
        (0..vocab)
            .step_by(1024)
            .map(move |start| (start, (start + 1024).min(vocab)))
    }
    /// Original payload triple followed by three writable auxiliary buffers.
    pub fn chunk(&self, start: i32, end: i32) -> anyhow::Result<Graph> {
        anyhow::ensure!(
            start >= 0 && end > start && end <= self.vocab,
            "invalid coarse head chunk"
        );
        let schema = Default::default();
        let mut p = Build::new(6, &schema);
        let mut parts = Vec::new();
        for i in 0..3 {
            parts.push(p.slice(Value::Input(i), 0, start, end));
        }
        let dense = p.graph.one(
            Kernel::AffineDequantize {
                group_size: self.source_group,
                bits: self.source_bits,
            },
            parts,
        );
        let encoded = p.graph.call(
            Kernel::AffineQuantize {
                group_size: 64,
                bits: 2,
            },
            vec![dense],
        );
        let mut out = Vec::new();
        for (i, source) in encoded.into_iter().enumerate() {
            p.graph.writable_input(3 + i);
            out.push(p.op(
                T::CopySliceInto {
                    axis: 0,
                    start,
                    end,
                },
                vec![source, Value::Input(3 + i)],
            ));
        }
        Ok(p.finish(out)?.graph)
    }
}
