//! Layer-major expansion of immutable row graphs. Feedback edges retain serial
//! causal order; explicitly selected single-row projections share one dispatch.
use super::{Builder, Graph, Value};
use lisa_backend::{Kernel, PackedCompute as P, TensorOperation as T};

#[derive(Clone, Debug)]
pub struct Feedback {
    pub input: usize,
    pub output: usize,
    /// Optional view applied to the preceding row's output (e.g. a KV window).
    pub view: Option<T>,
}

/// Expand 1..=8 rows, with row-major external inputs and row-major outputs.
/// `feedback[row]` describes recurrent inputs for that row; row zero uses its
/// external state. Each coalesced projection must consume/produce one leading
/// activation row, and its weights must be identical across the sequence.
pub fn expand(
    graph: &Graph,
    rows: usize,
    feedback: &[Vec<Feedback>],
    projections: &[usize],
) -> Result<Graph, String> {
    graph.validate()?;
    if !(1..=8).contains(&rows)
        || feedback.len() != rows
        || !feedback[0].is_empty()
        || graph.states != 0
        || !graph.writes.is_empty()
        || !graph.mutable_inputs.is_empty()
    {
        return Err("invalid immutable row expansion".into());
    }
    for edges in feedback {
        let mut inputs = std::collections::HashSet::new();
        for edge in edges {
            if edge.input >= graph.inputs
                || edge.output >= graph.outputs.len()
                || !inputs.insert(edge.input)
            {
                return Err("invalid row feedback".into());
            }
        }
    }
    let mut batch = vec![false; graph.calls.len()];
    for &i in projections {
        let call = graph.calls.get(i).ok_or("invalid row projection index")?;
        if !matches!(
            call.kernel,
            Kernel::PackedCompute(
                P::PlainLinear {
                    round_output: false
                } | P::Q8Linear {
                    round_output: false,
                    round_input: false
                } | P::BlockLinear {
                    round_output: false,
                    ..
                } | P::ExpertLinear { .. }
                    | P::Q8ExpertLinear
            )
        ) || batch[i]
        {
            return Err("unsupported row projection".into());
        }
        batch[i] = true;
    }
    let mut builder = Builder::new(graph.inputs * rows, graph.weights, 0);
    for c in &graph.constants {
        builder.constant(c.bytes.clone(), c.shape.clone(), c.dtype);
    }
    let mut values = vec![vec![vec![]; graph.calls.len()]; rows];
    let mut row_inputs = vec![vec![None; graph.inputs]; rows];
    fn resolve(
        value: Value,
        row: usize,
        graph: &Graph,
        feedback: &[Vec<Feedback>],
        values: &[Vec<Vec<Value>>],
        inputs: &mut [Vec<Option<Value>>],
        b: &mut Builder,
    ) -> Result<Value, String> {
        match value {
            Value::Input(i) => {
                if let Some(v) = inputs[row][i] {
                    return Ok(v);
                }
                let v = if let Some(edge) = feedback[row].iter().find(|e| e.input == i) {
                    let v = resolve(
                        graph.outputs[edge.output],
                        row - 1,
                        graph,
                        feedback,
                        values,
                        inputs,
                        b,
                    )?;
                    if let Some(view) = &edge.view {
                        b.one(Kernel::Tensor(view.clone()), vec![v])
                    } else {
                        v
                    }
                } else {
                    Value::Input(row * graph.inputs + i)
                };
                inputs[row][i] = Some(v);
                Ok(v)
            }
            Value::Result { call, output } => values[row][call]
                .get(output)
                .copied()
                .ok_or_else(|| "feedback requires a later graph operation".into()),
            Value::State(_) => Err("row expansion cannot bind external state slots".into()),
            v => Ok(v),
        }
    }
    for (index, call) in graph.calls.iter().enumerate() {
        if batch[index] && rows > 1 {
            let args = (0..rows)
                .map(|row| {
                    call.inputs
                        .iter()
                        .map(|&v| {
                            resolve(
                                v,
                                row,
                                graph,
                                feedback,
                                &values,
                                &mut row_inputs,
                                &mut builder,
                            )
                        })
                        .collect::<Result<Vec<_>, _>>()
                })
                .collect::<Result<Vec<_>, _>>()?;
            if args.iter().any(|a| a[0] != args[0][0]) {
                return Err("row projections require shared weights".into());
            }
            let mut joined = vec![args[0][0]];
            for input in 1..call.inputs.len() {
                let kernel = if input == 1 {
                    Kernel::PackedCompute(P::ConcatF32 {
                        axis: 0,
                        inputs: rows as u32,
                    })
                } else {
                    Kernel::Tensor(T::Concatenate {
                        axis: 0,
                        inputs: rows,
                    })
                };
                joined.push(builder.one(kernel, args.iter().map(|a| a[input]).collect::<Vec<_>>()));
            }
            let out = builder.one(call.kernel.clone(), joined);
            for row in 0..rows {
                values[row][index] = vec![builder.one(
                    Kernel::Tensor(T::Slice {
                        axis: 0,
                        start: row as i32,
                        end: row as i32 + 1,
                    }),
                    vec![out],
                )];
            }
        } else {
            for row in 0..rows {
                let args = call
                    .inputs
                    .iter()
                    .map(|&v| {
                        resolve(
                            v,
                            row,
                            graph,
                            feedback,
                            &values,
                            &mut row_inputs,
                            &mut builder,
                        )
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                values[row][index] = builder.call(call.kernel.clone(), args);
            }
        }
    }
    let mut outputs = Vec::new();
    for row in 0..rows {
        for &value in &graph.outputs {
            outputs.push(resolve(
                value,
                row,
                graph,
                feedback,
                &values,
                &mut row_inputs,
                &mut builder,
            )?);
        }
    }
    let expanded = builder.finish(outputs);
    expanded.validate()?;
    Ok(expanded)
}

/// The model opts in after establishing that each projection has one leading
/// activation row. Grouped attention products are intentionally excluded.
pub fn single_row_projections(graph: &Graph) -> Vec<usize> {
    graph
        .calls
        .iter()
        .enumerate()
        .filter_map(|(i, c)| {
            matches!(
                c.kernel,
                Kernel::PackedCompute(
                    P::PlainLinear {
                        round_output: false
                    } | P::Q8Linear {
                        round_output: false,
                        round_input: false
                    } | P::BlockLinear {
                        round_output: false,
                        ..
                    } | P::ExpertLinear { .. }
                        | P::Q8ExpertLinear
                )
            )
            .then_some(i)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use lisa_backend::Binary;

    #[test]
    fn rejects_feedback_that_would_cross_a_projection_barrier() {
        let mut b = Builder::new(2, 1, 0);
        let x = b.one(
            Kernel::Tensor(T::Binary(Binary::Add)),
            vec![Value::Input(0), Value::Input(1)],
        );
        let y = b.one(
            Kernel::PackedCompute(P::Q8Linear {
                round_output: false,
                round_input: false,
            }),
            vec![Value::Weight(0), x],
        );
        let graph = b.finish(vec![y]);
        let feedback = vec![
            vec![],
            vec![Feedback {
                input: 1,
                output: 0,
                view: None,
            }],
        ];
        assert!(
            expand(&graph, 2, &feedback, &[1])
                .unwrap_err()
                .contains("later graph operation")
        );
    }

    #[test]
    fn rejects_ambiguous_feedback_and_rounded_projections() {
        let mut b = Builder::new(1, 1, 0);
        let y = b.one(
            Kernel::PackedCompute(P::Q8Linear {
                round_output: true,
                round_input: false,
            }),
            vec![Value::Weight(0), Value::Input(0)],
        );
        let graph = b.finish(vec![y]);
        let edge = Feedback {
            input: 0,
            output: 0,
            view: None,
        };
        assert!(expand(&graph, 2, &[vec![], vec![edge.clone(), edge]], &[]).is_err());
        assert!(expand(&graph, 2, &[vec![], vec![]], &[0]).is_err());
        assert!(expand(&graph, 0, &[], &[]).is_err());
        assert!(expand(&graph, 9, &vec![vec![]; 9], &[]).is_err());
    }
}
