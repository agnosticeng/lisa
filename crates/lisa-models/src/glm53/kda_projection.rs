//! KDA projection ordering and fusion eligibility, independent of the device.
use super::packed::{Kind, Projection};
use crate::graph::{Builder, Graph, Value as V};
use lisa_backend::{Kernel as K, ScalarType as D, SparseF32 as S, TensorOperation as T};
fn reshape(b: &mut Builder, x: V, shape: &[i32]) -> V {
    b.one(K::Tensor(T::Reshape(shape.to_vec())), vec![x])
}
fn linear(b: &mut Builder, p: Projection, x: V, w: V, shape: &[i32]) -> anyhow::Result<V> {
    let graph = match p.kind {
        Kind::Plain | Kind::Bf16 => super::math::projection(
            shape,
            &[p.output, p.input],
            if p.kind == Kind::Bf16 {
                D::BF16
            } else {
                D::F32
            },
        )?,
        _ => p.graph(shape)?,
    };
    let inputs = if matches!(p.kind, Kind::Plain | Kind::Bf16) {
        [x, w]
    } else {
        [w, x]
    };
    Ok(b.append(graph, &inputs, &[], &[])
        .map_err(anyhow::Error::msg)?[0])
}
pub fn one(p: Projection, shape: &[i32]) -> anyhow::Result<Graph> {
    let mut b = Builder::new(2, 0, 0);
    let y = linear(&mut b, p, V::Input(1), V::Input(0), shape)?;
    Ok(b.finish(vec![y]))
}
fn qkv(
    b: &mut Builder,
    p: [Projection; 3],
    x: V,
    w: [V; 3],
    shape: &[i32],
    triple: bool,
) -> anyhow::Result<[V; 3]> {
    if triple
        && shape[0] == 1
        && shape[1] == 1
        && p.iter().all(|a| {
            a.kind == Kind::Bf16 && a.input == shape[2] && a.output == p[0].output && a.output > 0
        })
    {
        let x = reshape(b, x, &[1, shape[2]]);
        let out = b.call(K::SparseF32(S::Bf16TripleLinear), vec![w[0], w[1], w[2], x]);
        Ok(std::array::from_fn(|i| {
            reshape(b, out[i], &[1, 1, p[0].output])
        }))
    } else {
        Ok([
            linear(b, p[0], x, w[0], shape)?,
            linear(b, p[1], x, w[1], shape)?,
            linear(b, p[2], x, w[2], shape)?,
        ])
    }
}
pub fn inputs(
    p: [Projection; 6],
    shape: &[i32],
    fused: bool,
    triple: bool,
    groups: Option<usize>,
) -> anyhow::Result<Graph> {
    anyhow::ensure!(
        shape.len() == 3 && shape.iter().all(|&d| d > 0),
        "invalid recurrent input shape"
    );
    let mut b = Builder::new(7, 0, 0);
    let x = V::Input(0);
    let w = std::array::from_fn::<_, 6, _>(|i| V::Input(i + 1));
    let out = if fused
        && (1..=8).contains(&shape[0])
        && shape[1] == 1
        && p.iter()
            .all(|a| a.kind == Kind::Bf16 && a.input == shape[2] && a.output > 0)
        && p[1].output == p[0].output
        && p[2].output == p[0].output
        && p[4].output == p[3].output
        && groups.is_none_or(|n| matches!(n, 4 | 8 | 16))
    {
        let x = reshape(&mut b, x, &[shape[0], shape[2]]);
        let mut args = w.to_vec();
        args.push(x);
        let all = b.call(K::SparseF32(S::Bf16PackedSix { groups }), args);
        let width = p[0].output;
        let qkv = std::array::from_fn::<_, 3, _>(|i| {
            let y = b.one(
                K::Tensor(T::Slice {
                    axis: 1,
                    start: i as i32 * width,
                    end: (i as i32 + 1) * width,
                }),
                vec![all[0]],
            );
            reshape(&mut b, y, &[shape[0], 1, width])
        });
        let beta = b.one(
            K::Tensor(T::Slice {
                axis: 1,
                start: 3 * width,
                end: 3 * width + p[5].output,
            }),
            vec![all[0]],
        );
        let beta = reshape(&mut b, beta, &[shape[0], 1, p[5].output]);
        let fa = reshape(&mut b, all[1], &[shape[0], 1, p[3].output]);
        let ga = reshape(&mut b, all[2], &[shape[0], 1, p[4].output]);
        vec![qkv[0], qkv[1], qkv[2], fa, ga, beta]
    } else {
        let qkv = qkv(
            &mut b,
            [p[0], p[1], p[2]],
            x,
            [w[0], w[1], w[2]],
            shape,
            triple,
        )?;
        vec![
            qkv[0],
            qkv[1],
            qkv[2],
            linear(&mut b, p[3], x, w[3], shape)?,
            linear(&mut b, p[4], x, w[4], shape)?,
            linear(&mut b, p[5], x, w[5], shape)?,
        ]
    };
    Ok(b.finish(out))
}
pub fn gates(
    p: [Projection; 2],
    shape: &[i32],
    other: &[i32],
    input_f32: bool,
    paired: bool,
) -> anyhow::Result<Graph> {
    anyhow::ensure!(
        shape.len() == 3 && other.len() == 3,
        "invalid recurrent gate shape"
    );
    let mut b = Builder::new(4, 0, 0);
    let rows = shape[0].checked_mul(shape[1]).unwrap_or(0);
    let out = if paired
        && shape == other
        && input_f32
        && (1..=8).contains(&rows)
        && p.iter().all(|a| {
            a.kind == Kind::Bf16 && a.input == shape[2] && a.output == p[0].output && a.output > 0
        }) {
        let a = reshape(&mut b, V::Input(0), &[rows, shape[2]]);
        let c = reshape(&mut b, V::Input(1), &[rows, shape[2]]);
        let out = b.call(
            K::SparseF32(S::Bf16PairLinear),
            vec![V::Input(2), a, V::Input(3), c],
        );
        vec![
            reshape(&mut b, out[0], &[shape[0], shape[1], p[0].output]),
            reshape(&mut b, out[1], &[shape[0], shape[1], p[1].output]),
        ]
    } else {
        vec![
            linear(&mut b, p[0], V::Input(0), V::Input(2), shape)?,
            linear(&mut b, p[1], V::Input(1), V::Input(3), other)?,
        ]
    };
    Ok(b.finish(out))
}
