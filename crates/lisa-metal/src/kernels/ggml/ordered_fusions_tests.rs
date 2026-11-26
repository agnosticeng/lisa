use super::*;
use crate::ops;
fn exact(a: &Array, b: &Array) {
    let a = a.as_slice::<f32>();
    let b = b.as_slice::<f32>();
    assert_eq!(a.len(), b.len());
    for (i, (a, b)) in a.iter().zip(b).enumerate() {
        assert_eq!(a.to_bits(), b.to_bits(), "float {i}: {a} != {b}");
    }
}
fn data(n: usize) -> Vec<f32> {
    let mut seed = 9129783u32;
    (0..n)
        .map(|_| {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            seed as i32 as f32 / 219813147.
        })
        .collect()
}
#[test]
fn ordered_concat_copies_strides_offsets_and_float_bits() {
    let s = Stream::gpu();
    for rank in 1..=3 {
        for axis in 0..rank {
            for count in [2, 3, 8] {
                let mut shape = vec![4; rank];
                shape[axis] = 7;
                let n = shape.iter().product::<i32>() as usize;
                let data = (0..n)
                    .map(|i| f32::from_bits((i as u32).wrapping_mul(1376312589)))
                    .collect::<Vec<_>>();
                let x = Array::from_slice(&data, &shape);
                let pieces = (0..count)
                    .map(|i| {
                        lisa_backend::TensorAccess::slice(&x, axis as i32, 1, Some(2 + i % 5))
                            .unwrap()
                    })
                    .collect::<Vec<_>>();
                let refs = pieces.iter().collect::<Vec<_>>();
                let expected = ops::concatenate(&refs, axis as i32).unwrap();
                exact(&concat_f32(&refs, axis, &s).unwrap(), &expected);
            }
        }
    }
    let x = Array::from_slice(&data(3 * 8 * 16), &[3, 8, 16])
        .transpose_axes(&[1, 0, 2])
        .unwrap();
    exact(
        &concat_f32(&[&x, &x], 1, &s).unwrap(),
        &ops::concatenate(&[&x, &x], 1).unwrap(),
    );
}
