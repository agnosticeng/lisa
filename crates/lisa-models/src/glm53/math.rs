//! Explicit operation chains used by GLM's diagnostic and portable paths.
//! These graphs retain the original casts, contiguation and reduction order.
use crate::{
    graph::{Graph, Value as V},
    program::Build,
};
use lisa_backend::{
    Binary as B, Kernel, ScalarType as D, SparseF32, TensorOperation as T, Unary as U,
};
pub fn rms(rank: usize, epsilon: f32, weighted: bool, gated: bool) -> anyhow::Result<Graph> {
    anyhow::ensure!(
        rank > 0 && (!gated || weighted && rank == 4),
        "invalid GLM normalization geometry"
    );
    let schema = Default::default();
    let mut p = Build::new(1 + usize::from(weighted) + usize::from(gated), &schema);
    let x = p.cast(V::Input(0), D::F32);
    let sq = p.binary(B::Multiply, x, x);
    let sq = p.contiguous(sq);
    let ms = p.op(
        T::Mean {
            axis: (rank - 1) as i32,
            keep_dims: true,
        },
        vec![sq],
    );
    let eps = p.scalar(epsilon, D::F32);
    let shifted = p.binary(B::Add, ms, eps);
    let inverse = p.unary(U::Rsqrt, shifted);
    let mut n = p.binary(B::Multiply, x, inverse);
    if weighted {
        let w = p.cast(V::Input(1), D::F32);
        n = p.binary(B::Multiply, n, w);
    }
    if gated {
        let g = p.cast(V::Input(2), D::F32);
        let g = p.unary(U::Sigmoid, g);
        n = p.binary(B::Multiply, n, g);
    }
    Ok(p.finish(vec![n])?.graph)
}
pub fn l2(epsilon: f32, dtype: D) -> anyhow::Result<Graph> {
    let schema = Default::default();
    let mut p = Build::new(1, &schema);
    let x = V::Input(0);
    let sq = p.binary(B::Multiply, x, x);
    let sq = p.contiguous(sq);
    let ms = p.op(
        T::Sum {
            axis: 3,
            keep_dims: true,
        },
        vec![sq],
    );
    let eps = match dtype {
        D::F32 | D::BF16 => p.scalar(epsilon, dtype),
        D::F16 => p.graph.constant(
            half::f16::from_f32(epsilon).to_le_bytes().to_vec(),
            vec![],
            D::F16,
        ),
        _ => anyhow::bail!("L2 normalization requires floating-point input"),
    };
    let shifted = p.binary(B::Add, ms, eps);
    let inverse = p.unary(U::Rsqrt, shifted);
    let n = p.binary(B::Multiply, x, inverse);
    Ok(p.finish(vec![n])?.graph)
}
pub fn matmul(shape: &[i32], weight_shape: &[i32]) -> anyhow::Result<Graph> {
    anyhow::ensure!(
        !shape.is_empty() && weight_shape.len() == 2,
        "invalid GLM matrix geometry"
    );
    let rows = shape[..shape.len() - 1].iter().product::<i32>().max(1);
    let schema = Default::default();
    let mut p = Build::new(2, &schema);
    let x = p.reshape(V::Input(0), &[rows, shape[shape.len() - 1]]);
    let y = p.op(T::Matmul, vec![x, V::Input(1)]);
    let y = p.reshape(y, &[&shape[..shape.len() - 1], &[weight_shape[1]]].concat());
    Ok(p.finish(vec![y])?.graph)
}
pub fn projection(shape: &[i32], weight_shape: &[i32], dtype: D) -> anyhow::Result<Graph> {
    anyhow::ensure!(
        shape.len() == 3 && weight_shape.len() == 2,
        "invalid GLM projection geometry"
    );
    let (b, t) = (shape[0], shape[1]);
    let rows = b
        .checked_mul(t)
        .ok_or_else(|| anyhow::anyhow!("projection row count overflow"))?;
    let schema = Default::default();
    let mut p = Build::new(2, &schema);
    let y = if rows > 0 && dtype == D::BF16 {
        let x = p.reshape(V::Input(0), &[rows, shape[2]]);
        let y = p.graph.one(
            Kernel::SparseF32(SparseF32::Bf16Linear { wide: rows > 8 }),
            vec![V::Input(1), x],
        );
        p.reshape(y, &[b, t, weight_shape[0]])
    } else {
        let w = p.transpose(V::Input(1), &[1, 0]);
        let w = p.cast(w, D::F32);
        p.graph
            .append(
                matmul(shape, &[weight_shape[1], weight_shape[0]])?,
                &[V::Input(0), w],
                &[],
                &[],
            )
            .map_err(anyhow::Error::msg)?[0]
    };
    Ok(p.finish(vec![y])?.graph)
}
