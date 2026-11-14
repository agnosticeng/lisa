//! Original small-row projection, paired-query and eight-group output plans.
use crate::{
    graph::{Builder, Graph, Value},
    packed::Projection,
};
use lisa_backend::{Kernel, PackedCompute, PackedEncoding, TensorOperation as T};
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Matrix {
    pub kind: Projection,
    pub input: i32,
    pub output: i32,
    pub row_bytes: i32,
}
pub fn supported(kind: Projection) -> bool {
    !matches!(
        kind,
        Projection::Block(PackedEncoding::Q2K | PackedEncoding::Q4K)
    )
}
impl Matrix {
    pub fn validate(self) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.input > 0 && self.output > 0 && self.row_bytes > 0 && supported(self.kind),
            "unsupported scalar matrix contract"
        );
        Ok(())
    }
    pub fn linear(self, first: i32, count: i32, round: bool) -> anyhow::Result<Graph> {
        self.validate()?;
        anyhow::ensure!(
            first >= 0 && count > 0 && first <= self.output && count <= self.output - first,
            "invalid scalar output-row range"
        );
        let mut b = Builder::new(2, 0, 0);
        let w = b.one(
            Kernel::Tensor(T::Slice {
                axis: 0,
                start: first,
                end: first + count,
            }),
            vec![Value::Input(0)],
        );
        let y = b.one(self.kind.linear(round), vec![w, Value::Input(1)]);
        Ok(b.finish(vec![y]))
    }
}
pub fn paired(a: Matrix, b: Matrix, rows: i32) -> anyhow::Result<Graph> {
    a.validate()?;
    b.validate()?;
    anyhow::ensure!(
        a.input == b.input && (1..=8).contains(&rows),
        "invalid paired projection geometry"
    );
    let mut p = Builder::new(3, 0, 0);
    let results = if a.kind == Projection::Q8 && b.kind == Projection::Q8 && rows == 1 {
        p.call(
            Kernel::PackedCompute(PackedCompute::Q8Pair),
            vec![Value::Input(0), Value::Input(1), Value::Input(2)],
        )
    } else {
        vec![
            p.one(a.kind.linear(true), vec![Value::Input(0), Value::Input(2)]),
            p.one(b.kind.linear(true), vec![Value::Input(1), Value::Input(2)]),
        ]
    };
    Ok(p.finish(results))
}
pub fn output_groups(m: Matrix) -> anyhow::Result<Graph> {
    m.validate()?;
    anyhow::ensure!(
        m.input == 4096 && m.output == 8192,
        "invalid grouped attention output matrix"
    );
    let mut p = Builder::new(2, 0, 0);
    let result = if m.kind == Projection::Block(PackedEncoding::Q5K) {
        let w = p.one(
            Kernel::Tensor(T::Reshape(vec![8, 1024, m.row_bytes])),
            vec![Value::Input(0)],
        );
        let x = p.one(
            Kernel::Tensor(T::Reshape(vec![8, 1, 4096])),
            vec![Value::Input(1)],
        );
        let y = p.one(
            Kernel::PackedCompute(PackedCompute::GroupedLinear {
                encoding: PackedEncoding::Q5K,
                round_output: true,
            }),
            vec![w, x],
        );
        p.one(Kernel::Tensor(T::Reshape(vec![1, 8192])), vec![y])
    } else {
        let x = p.one(
            Kernel::Tensor(T::Reshape(vec![8, 4096])),
            vec![Value::Input(1)],
        );
        let mut outputs = Vec::new();
        for group in 0..8 {
            let w = p.one(
                Kernel::Tensor(T::Slice {
                    axis: 0,
                    start: group * 1024,
                    end: (group + 1) * 1024,
                }),
                vec![Value::Input(0)],
            );
            let x = p.one(
                Kernel::Tensor(T::Slice {
                    axis: 0,
                    start: group,
                    end: group + 1,
                }),
                vec![x],
            );
            outputs.push(p.one(m.kind.linear(true), vec![w, x]));
        }
        p.one(
            Kernel::Tensor(T::Concatenate {
                axis: 1,
                inputs: outputs.len(),
            }),
            outputs,
        )
    };
    Ok(p.finish(vec![result]))
}
