//! Four-stream DS4 residual computation, with original rounding boundaries.
use super::projection::Matrix;
use crate::graph::{Builder, Graph, Value as V};
use lisa_backend::{Kernel as K, PackedCompute as P, TensorOperation as T};

pub fn requirements(layer: usize, kind: &str) -> anyhow::Result<[String; 4]> {
    anyhow::ensure!(
        matches!(kind, "attn" | "ffn"),
        "invalid residual block role"
    );
    Ok([
        format!("blk.{layer}.hc_{kind}_fn.weight"),
        format!("blk.{layer}.hc_{kind}_scale.weight"),
        format!("blk.{layer}.hc_{kind}_base.weight"),
        format!("blk.{layer}.{kind}_norm.weight"),
    ])
}
pub fn input(mixer: Matrix) -> anyhow::Result<Graph> {
    anyhow::ensure!(
        mixer.input == 20480 && mixer.output == 24,
        "invalid residual mixer geometry"
    );
    let mut b = Builder::new(2, 4, 0);
    let flat = b.one(K::Tensor(T::Reshape(vec![1, 20480])), vec![V::Input(0)]);
    let norm = b.one(
        K::PackedCompute(P::RmsNorm {
            weighted: false,
            round_output: false,
            epsilon: 1e-20f32.to_bits(),
        }),
        vec![flat],
    );
    let mix = b
        .append(mixer.linear(0, 24, false)?, &[V::Weight(0), norm], &[], &[])
        .map_err(anyhow::Error::msg)?[0];
    let outputs = b.call(
        K::PackedCompute(P::ResidualInput4 {
            epsilon: 1e-6f32.to_bits(),
            norm_epsilon: 1e-20f32.to_bits(),
        }),
        vec![
            mix,
            V::Input(0),
            V::Input(1),
            V::Weight(1),
            V::Weight(2),
            V::Weight(3),
        ],
    );
    Ok(b.finish(outputs))
}
pub fn expand(add: bool, round_block: bool) -> Graph {
    let count = 3 + usize::from(add);
    let mut b = Builder::new(count, 0, 0);
    let y = b.one(
        K::PackedCompute(P::ResidualExpand4 { add, round_block }),
        (0..count).map(V::Input).collect::<Vec<_>>(),
    );
    b.finish(vec![y])
}
pub fn head() -> Graph {
    let mut b = Builder::new(3, 0, 0);
    let hidden = b.one(
        K::PackedCompute(P::ResidualCollapse4),
        vec![V::Input(0), V::Input(1)],
    );
    let hidden = b.one(
        K::PackedCompute(P::RmsNorm {
            weighted: true,
            round_output: true,
            epsilon: 1e-20f32.to_bits(),
        }),
        vec![hidden, V::Input(2)],
    );
    b.finish(vec![hidden])
}
