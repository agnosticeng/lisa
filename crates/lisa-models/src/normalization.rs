//! Shared normalization computation plans. The weight is an explicit input so
//! prepared plans can be reused without rebinding or duplicating layer weights.
use crate::graph::{Builder, Graph, Value};
use lisa_backend::{Binary, Kernel, ScalarType, TensorOperation as T, Unary};
pub fn rms(epsilon: f32, residual: bool) -> Graph {
    let mut b = Builder::new(if residual { 3 } else { 2 }, 0, 0);
    let output = if residual {
        b.call(
            Kernel::Tensor(T::AddRmsNorm {
                epsilon: epsilon.to_bits(),
            }),
            vec![Value::Input(0), Value::Input(1), Value::Input(2)],
        )
    } else {
        b.call(
            Kernel::Tensor(T::RmsNorm {
                epsilon: epsilon.to_bits(),
            }),
            vec![Value::Input(0), Value::Input(1)],
        )
    };
    b.finish(output)
}
pub fn gated_rms(epsilon: f32, dtype: ScalarType) -> Graph {
    let mut b = Builder::new(3, 0, 0);
    let out = b.one(
        Kernel::Tensor(T::RmsNorm {
            epsilon: epsilon.to_bits(),
        }),
        vec![Value::Input(0), Value::Input(2)],
    );
    let gate = b.one(
        Kernel::Tensor(T::Cast(ScalarType::F32)),
        vec![Value::Input(1)],
    );
    let gate = b.one(Kernel::Tensor(T::Unary(Unary::Sigmoid)), vec![gate]);
    let out = b.one(Kernel::Tensor(T::Cast(ScalarType::F32)), vec![out]);
    let out = b.one(Kernel::Tensor(T::Binary(Binary::Multiply)), vec![gate, out]);
    let out = b.one(Kernel::Tensor(T::Cast(dtype)), vec![out]);
    b.finish(vec![out])
}

/// BF16 logaddexp(x, 0), rounding every operation as in MLXNN.softplus.
pub fn bf16_logaddexp0() -> Graph {
    let schema = Default::default();
    let mut p = crate::program::Build::new(1, &schema);
    let zero = p.scalar(0., ScalarType::BF16);
    let x = Value::Input(0);
    let maximum = p.binary(Binary::Maximum, x, zero);
    let minimum = p.binary(Binary::Minimum, x, zero);
    let difference = p.binary(Binary::Subtract, minimum, maximum);
    let exp = p.unary(Unary::Exp, difference);
    let log = p.unary(Unary::Log1p, exp);
    let out = p.binary(Binary::Add, maximum, log);
    p.finish(vec![out])
        .expect("static normalization graph")
        .graph
}

/// Grouped RMS with the original two-stage butterfly and BF16 rounding before
/// applying the checkpoint scale. The 128-wide native path is a distinct law;
/// wider groups retain the explicit operation chain.
pub fn grouped_rms(
    shape: &[i32],
    dtype: ScalarType,
    epsilon: f32,
    width: usize,
    weighted: bool,
    force_operations: bool,
) -> anyhow::Result<Graph> {
    use crate::{graph::Value as V, program::Build};
    anyhow::ensure!(
        !shape.is_empty() && shape.iter().all(|&d| d > 0),
        "invalid grouped RMS shape"
    );
    anyhow::ensure!(
        (128..=4096).contains(&width)
            && width.is_multiple_of(128)
            && (shape[shape.len() - 1] as usize).is_multiple_of(width),
        "invalid grouped RMS width"
    );
    anyhow::ensure!(
        epsilon.is_finite() && epsilon >= 0.,
        "invalid grouped RMS epsilon"
    );
    let schema = Default::default();
    let mut p = Build::new(1 + usize::from(weighted), &schema);
    let x = V::Input(0);
    let normalized = if !force_operations && width == 128 && dtype == ScalarType::BF16 {
        let y = p.op(
            T::GroupedRms128 {
                epsilon: epsilon.to_bits(),
            },
            vec![x],
        );
        p.reshape(y, shape)
    } else {
        let lead = &shape[..shape.len() - 1];
        let groups = shape[shape.len() - 1] / width as i32;
        let slices = (width / 128) as i32;
        let grouped = p.reshape(x, &[lead, &[groups, slices, 32, 4]].concat());
        let left = p.cast(grouped, ScalarType::F32);
        let right = p.cast(grouped, ScalarType::F32);
        let sq = p.binary(Binary::Multiply, left, right);
        let mut lane = p.slice(sq, -1, 0, 1);
        lane = p.contiguous(lane);
        for i in 1..4 {
            let term = p.slice(sq, -1, i, i + 1);
            let term = p.contiguous(term);
            lane = p.binary(Binary::Add, lane, term);
        }
        let lane = p.reshape(lane, &[lead, &[groups, slices, 32]].concat());
        let mut butterfly = |mut v| {
            let mut width = 32;
            while width >= 2 {
                let half = width / 2;
                let a = p.slice(v, -1, 0, half);
                let a = p.contiguous(a);
                let b = p.slice(v, -1, half, width);
                let b = p.contiguous(b);
                v = p.binary(Binary::Add, a, b);
                width = half;
            }
            v
        };
        let partials = butterfly(lane);
        let partials = p.reshape(partials, &[lead, &[groups, slices]].concat());
        let zeros = p.graph.one(
            Kernel::Zeros {
                shape: [lead, &[groups, 32 - slices]].concat(),
                dtype: ScalarType::F32,
            },
            vec![],
        );
        let mut total = p.concat(vec![partials, zeros], -1);
        let mut count = 32;
        while count >= 2 {
            let half = count / 2;
            let a = p.slice(total, -1, 0, half);
            let a = p.contiguous(a);
            let b = p.slice(total, -1, half, count);
            let b = p.contiguous(b);
            total = p.binary(Binary::Add, a, b);
            count = half;
        }
        let total = p.reshape(total, &[lead, &[groups, 1, 1, 1]].concat());
        let divisor = p.scalar(width as f32, ScalarType::F32);
        let mean = p.binary(Binary::Divide, total, divisor);
        let eps = p.scalar(epsilon, ScalarType::F32);
        let mean = p.binary(Binary::Add, mean, eps);
        let inv = p.unary(Unary::Rsqrt, mean);
        let grouped = p.cast(grouped, ScalarType::F32);
        let normalized = p.binary(Binary::Multiply, grouped, inv);
        let normalized = p.cast(normalized, ScalarType::BF16);
        p.reshape(normalized, shape)
    };
    let out = if weighted {
        p.binary(Binary::Multiply, normalized, V::Input(1))
    } else {
        normalized
    };
    Ok(p.finish(vec![out])?.graph)
}
