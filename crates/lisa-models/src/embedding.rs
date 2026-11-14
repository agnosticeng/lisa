//! Affine embedding lookup and explicit draft-only requantization.
use crate::graph::{Builder, Graph, Value};
use lisa_backend::{Kernel, TensorOperation as T};
/// Inputs: IDs, packed rows, scales and biases. Output: selected dense rows.
pub fn affine(group_size: i32, bits: i32) -> Graph {
    let mut g = Builder::new(4, 0, 0);
    let rows = (1..4)
        .map(|i| {
            g.one(
                Kernel::Tensor(T::Take { axis: 0 }),
                vec![Value::Input(i), Value::Input(0)],
            )
        })
        .collect::<Vec<_>>();
    let out = g.one(Kernel::AffineDequantize { group_size, bits }, rows);
    g.finish(vec![out])
}
pub fn requantize(source: (i32, i32), target: (i32, i32)) -> Graph {
    let mut g = Builder::new(3, 0, 0);
    let full = g.one(
        Kernel::AffineDequantize {
            group_size: source.0,
            bits: source.1,
        },
        (0..3).map(Value::Input).collect::<Vec<_>>(),
    );
    let out = g.call(
        Kernel::AffineQuantize {
            group_size: target.0,
            bits: target.1,
        },
        vec![full],
    );
    g.finish(out)
}
/// Inputs: activations and packed projection triple. Output: original affine law.
pub fn linear(group_size: i32, bits: i32) -> Graph {
    let mut g = Builder::new(4, 0, 0);
    let out = g.one(
        Kernel::AffineLinear { group_size, bits },
        (0..4).map(Value::Input).collect::<Vec<_>>(),
    );
    g.finish(vec![out])
}
