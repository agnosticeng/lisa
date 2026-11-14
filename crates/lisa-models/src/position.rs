//! Rotary operation order, including casts before multiply and the unrotated
//! tail. Runtime shape and storage bindings are supplied by the engine.
use crate::graph::{Builder, Graph, Value};
use lisa_backend::{Binary, Kernel, ScalarType, TensorOperation as T, Unary};
pub fn partial_rotary(rotary: i32, width: i32, dtype: ScalarType) -> Graph {
    let mut b = Builder::new(3, 0, 0);
    let x = Value::Input(0);
    let c = b.one(Kernel::Tensor(T::Cast(dtype)), vec![Value::Input(1)]);
    let s = b.one(Kernel::Tensor(T::Cast(dtype)), vec![Value::Input(2)]);
    let rotated = b.one(
        Kernel::Tensor(T::Slice {
            axis: -1,
            start: 0,
            end: rotary,
        }),
        vec![x],
    );
    let first = b.one(
        Kernel::Tensor(T::Slice {
            axis: -1,
            start: 0,
            end: rotary / 2,
        }),
        vec![rotated],
    );
    let second = b.one(
        Kernel::Tensor(T::Slice {
            axis: -1,
            start: rotary / 2,
            end: rotary,
        }),
        vec![rotated],
    );
    let negative = b.one(Kernel::Tensor(T::Unary(Unary::Negate)), vec![second]);
    let swapped = b.one(
        Kernel::Tensor(T::Concatenate {
            axis: -1,
            inputs: 2,
        }),
        vec![negative, first],
    );
    let left = b.one(
        Kernel::Tensor(T::Binary(Binary::Multiply)),
        vec![rotated, c],
    );
    let right = b.one(
        Kernel::Tensor(T::Binary(Binary::Multiply)),
        vec![swapped, s],
    );
    let mut out = b.one(Kernel::Tensor(T::Binary(Binary::Add)), vec![left, right]);
    if width != rotary {
        let rest = b.one(
            Kernel::Tensor(T::Slice {
                axis: -1,
                start: rotary,
                end: width,
            }),
            vec![x],
        );
        out = b.one(
            Kernel::Tensor(T::Concatenate {
                axis: -1,
                inputs: 2,
            }),
            vec![out, rest],
        );
    }
    b.finish(vec![out])
}

/// Device-side rotary frequency construction. The base logarithm is host f32,
/// while exp/cos/sin remain device primitives with the native numerical law.
pub fn frequencies(
    dimensions: usize,
    base: f32,
    shape: &[i32],
    multimodal: bool,
) -> anyhow::Result<Graph> {
    anyhow::ensure!(
        dimensions > 0 && base.is_finite() && base > 0.,
        "invalid rotary frequencies"
    );
    let schema = Default::default();
    let mut p = crate::program::Build::new(1, &schema);
    let even = (0..dimensions)
        .step_by(2)
        .map(|i| i as f32)
        .collect::<Vec<_>>();
    let even = p.graph.constant_f32(&even, &[i32::try_from(even.len())?]);
    let factor = p.scalar(-base.ln() / dimensions as f32, ScalarType::F32);
    let argument = p.binary(Binary::Multiply, even, factor);
    let inverse = p.unary(Unary::Exp, argument);
    let mut positions = p.cast(Value::Input(0), ScalarType::F32);
    if !multimodal {
        let mut expanded = shape.to_vec();
        expanded.push(1);
        positions = p.reshape(positions, &expanded);
    }
    let frequencies = p.binary(Binary::Multiply, positions, inverse);
    let embedding = p.concat(vec![frequencies, frequencies], -1);
    let cos = p.unary(Unary::Cos, embedding);
    let sin = p.unary(Unary::Sin, embedding);
    Ok(p.finish(vec![cos, sin])?.graph)
}

/// Absolute text positions for a window, optionally one starting offset per
/// packed stream. Engine code imports the returned data without choosing its law.
pub fn indices(
    offset: usize,
    count: usize,
    next: Option<&[usize]>,
) -> anyhow::Result<(Vec<f32>, Vec<i32>)> {
    let default = [offset];
    let starts = next.unwrap_or(&default);
    anyhow::ensure!(
        count > 0
            && count <= i32::MAX as usize
            && !starts.is_empty()
            && starts.len() <= i32::MAX as usize,
        "invalid position window"
    );
    let mut values = Vec::with_capacity(
        starts
            .len()
            .checked_mul(count)
            .ok_or_else(|| anyhow::anyhow!("position window overflow"))?,
    );
    for &start in starts {
        let end = start
            .checked_add(count)
            .ok_or_else(|| anyhow::anyhow!("position offset overflow"))?;
        values.extend((start..end).map(|i| i as f32));
    }
    Ok((values, vec![starts.len() as i32, count as i32]))
}

#[cfg(test)]
mod index_tests {
    #[test]
    fn position_windows_preserve_stream_offsets_and_reject_overflow() {
        assert_eq!(
            super::indices(17, 3, None).unwrap(),
            (vec![17., 18., 19.], vec![1, 3])
        );
        assert_eq!(
            super::indices(99, 2, Some(&[3, 17])).unwrap(),
            (vec![3., 4., 17., 18.], vec![2, 2])
        );
        assert!(super::indices(usize::MAX, 2, None).is_err());
        assert!(super::indices(0, 2, Some(&[])).is_err());
    }
}
