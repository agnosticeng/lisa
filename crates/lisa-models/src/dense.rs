//! Dense affine SwiGLU dataflow, shared by compatible model definitions.
use crate::{Alternatives, Call, DimensionGuard, Plan, ShapeGuard, Value};
use lisa_backend::Kernel;

/// Weight slots are gate/up/down triples (packed weight, scales, biases), then
/// an optional joined gate/up triple. Bindings carry original checkpoint data.
pub fn swiglu(geometry: [(i32, i32); 3], joined: Option<(i32, i32)>) -> Alternatives {
    let linear = |input, base, (group_size, bits)| Call {
        kernel: Kernel::AffineLinear { group_size, bits },
        inputs: vec![
            input,
            Value::Weight(base),
            Value::Weight(base + 1),
            Value::Weight(base + 2),
        ],
    };
    let primary = Plan {
        calls: vec![
            linear(Value::Input, 0, geometry[0]),
            linear(Value::Input, 3, geometry[1]),
            Call {
                kernel: Kernel::SwiGlu,
                inputs: vec![Value::Temporary(0), Value::Temporary(1)],
            },
            linear(Value::Temporary(2), 6, geometry[2]),
        ],
        output: Value::Temporary(3),
    };
    let guarded = joined.map(|quant| {
        (
            ShapeGuard::Dimension(DimensionGuard {
                min_rank: 2,
                axis: 1,
                equals: 1,
            }),
            Plan {
                calls: vec![
                    linear(Value::Input, 9, quant),
                    Call {
                        kernel: Kernel::PackedSwiGlu,
                        inputs: vec![Value::Temporary(0)],
                    },
                    linear(Value::Temporary(1), 6, geometry[2]),
                ],
                output: Value::Temporary(2),
            },
        )
    });
    Alternatives { primary, guarded }
}

/// Share the existing packed gate/up allocation at narrow, single-stream
/// widths where both projections keep their original per-row matvec schedule.
/// Wider prefill and other geometries retain the separate projection plan.
pub fn swiglu_for_shape(
    geometry: [(i32, i32); 3],
    joined: Option<(i32, i32)>,
    input: i32,
    hidden: i32,
) -> Alternatives {
    let mut plans = swiglu(geometry, joined);
    if input == 5120
        && hidden == 17408
        && geometry[..2] == [(64, 4); 2]
        && let Some((guard, _)) = &mut plans.guarded
    {
        *guard = ShapeGuard::Any(vec![
            guard.clone(),
            ShapeGuard::ExactRank(vec![1..=1, 2..=8, 5120..=5120]),
        ]);
    }
    plans
}

pub fn swiglu_serial_for_shape(
    geometry: [(i32, i32); 3],
    joined: Option<(i32, i32)>,
    input: i32,
    hidden: i32,
) -> Alternatives {
    let mut plans = swiglu_for_shape(geometry, joined, input, hidden);
    for plan in
        std::iter::once(&mut plans.primary).chain(plans.guarded.iter_mut().map(|(_, plan)| plan))
    {
        for call in &mut plan.calls {
            call.kernel = crate::graph::serial_affine_kernel(&call.kernel, true);
        }
    }
    plans
}

/// Binding-time shared storage for a compatible pair of gate/up projections.
/// Inputs are the two packed triples. Outputs are joined/gate/up triples.
pub fn join_gate_up(widths: [i32; 2], geometry: [(i32, i32); 2]) -> Option<crate::graph::Graph> {
    use crate::graph::{Builder, Value};
    use lisa_backend::TensorOperation as T;
    if geometry[0] != geometry[1] {
        return None;
    }
    let mut g = Builder::new(6, 0, 0);
    let joined = (0..3)
        .map(|i| {
            g.one(
                Kernel::Tensor(T::Concatenate { axis: 0, inputs: 2 }),
                vec![Value::Input(i), Value::Input(i + 3)],
            )
        })
        .collect::<Vec<_>>();
    let mut outputs = joined.clone();
    for (start, end) in [(0, widths[0]), (widths[0], widths[0] + widths[1])] {
        for &v in &joined {
            outputs.push(g.one(
                Kernel::Tensor(T::Slice {
                    axis: 0,
                    start,
                    end,
                }),
                vec![v],
            ));
        }
    }
    Some(g.finish(outputs))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn wide_joined_plan_is_limited_to_admitted_single_stream_shapes() {
        let plans = swiglu_for_shape([(64, 4); 3], Some((64, 4)), 5120, 17408);
        let guard = &plans.guarded.unwrap().0;
        assert!(guard.valid());
        for rows in 1..=8 {
            assert!(guard.matches(&[1, rows, 5120]));
        }
        for shape in [
            vec![1, 9, 5120],
            vec![2, 7, 5120],
            vec![7, 5120],
            vec![1, 7, 4096],
        ] {
            assert!(!guard.matches(&shape), "{shape:?}");
        }
        // Preserve the existing length-one batched path.
        assert!(guard.matches(&[2, 1, 5120]));
        for (quant, input, hidden) in [
            ((64, 8), 5120, 17408),
            ((64, 4), 4096, 17408),
            ((64, 4), 5120, 14336),
        ] {
            let other = swiglu_for_shape([quant; 3], Some(quant), input, hidden);
            assert!(!other.guarded.unwrap().0.matches(&[1, 7, input]));
        }
    }
    #[test]
    fn invalid_dataflow_and_missing_bindings_are_rejected() {
        let mut plans = swiglu([(64, 4); 3], Some((64, 4)));
        assert!(plans.primary.validate(9).is_ok());
        assert!(plans.guarded.as_ref().unwrap().1.validate(12).is_ok());
        assert!(plans.guarded.as_ref().unwrap().1.validate(9).is_err());
        plans.primary.calls[0].inputs[0] = Value::Temporary(0);
        assert!(plans.primary.validate(12).is_err());
    }
}
