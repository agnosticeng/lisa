//! Structural partitions of immutable graphs. The model selects boundaries;
//! partitioning preserves every call, operand, constant and output in order.
use super::{Call, Graph, Value};
use std::collections::{HashMap, HashSet};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Source {
    Input(usize),
    Output { part: usize, index: usize },
}

pub struct Region {
    pub graph: Graph,
    /// One source per compact graph input, in binding order.
    pub inputs: Vec<Source>,
    /// Original graph weight indexes, in compact binding order.
    pub weights: Vec<usize>,
}

pub struct SplitGraph {
    pub input_count: usize,
    pub parts: Vec<Region>,
    pub outputs: Vec<Source>,
}

/// Split at ordered call indexes including zero and the original call count.
/// Models must separately prove which regions have replayable fixed geometry.
/// Mutating graphs are rejected: region outputs retain immutable handles, and
/// there is no intermediate state commit or writable input ABI.
pub fn split(graph: &Graph, cuts: &[usize]) -> Result<SplitGraph, String> {
    graph.validate()?;
    if graph.states != 0
        || !graph.writes.is_empty()
        || !graph.mutable_inputs.is_empty()
        || graph
            .calls
            .iter()
            .any(|c| !c.kernel.overwritten_inputs().is_empty())
    {
        return Err("graph regions require immutable inputs and explicit outputs".into());
    }
    if cuts.len() < 2
        || cuts[0] != 0
        || cuts.last() != Some(&graph.calls.len())
        || cuts.windows(2).any(|span| span[0] >= span[1])
    {
        return Err("graph region cuts must cover every call exactly once".into());
    }
    let mut sources = (0..graph.inputs)
        .map(|i| (Value::Input(i), Source::Input(i)))
        .collect::<HashMap<_, _>>();
    let mut parts = Vec::with_capacity(cuts.len() - 1);
    for (part, span) in cuts.windows(2).enumerate() {
        let (start, end) = (span[0], span[1]);
        let later = graph.calls[end..]
            .iter()
            .flat_map(|call| call.inputs.iter().copied())
            .chain(graph.outputs.iter().copied())
            .collect::<HashSet<_>>();
        let mut returned = Vec::new();
        for call in start..end {
            for output in 0..graph.calls[call].kernel.outputs() {
                let value = Value::Result { call, output };
                if later.contains(&value) {
                    returned.push(value);
                }
            }
        }
        if part + 2 == cuts.len() {
            for &value in &graph.outputs {
                if matches!(value, Value::Constant(_) | Value::Weight(_))
                    && !returned.contains(&value)
                {
                    returned.push(value);
                }
            }
        }
        if returned.is_empty() {
            return Err("graph region has no live output".into());
        }
        let mut inputs = Vec::new();
        let mut weights = Vec::new();
        let mut constants = Vec::new();
        let mut remapped = HashMap::new();
        let mut remap = |value| -> Result<Value, String> {
            if let Some(mapped) = remapped.get(&value) {
                return Ok(*mapped);
            }
            let mapped = match value {
                Value::Result { call, output } if (start..end).contains(&call) => Value::Result {
                    call: call - start,
                    output,
                },
                Value::Input(_) | Value::Result { .. } => {
                    let source = sources.get(&value).copied().ok_or_else(|| {
                        "graph region needs an unavailable earlier output".to_string()
                    })?;
                    let index = inputs.len();
                    inputs.push(source);
                    Value::Input(index)
                }
                Value::Weight(index) => {
                    let slot = weights.len();
                    weights.push(index);
                    Value::Weight(slot)
                }
                Value::Constant(index) => {
                    let slot = constants.len();
                    constants.push(graph.constants[index].clone());
                    Value::Constant(slot)
                }
                Value::State(_) => return Err("graph region contains state".into()),
            };
            remapped.insert(value, mapped);
            Ok(mapped)
        };
        let calls = graph.calls[start..end]
            .iter()
            .map(|call| {
                Ok(Call {
                    kernel: call.kernel.clone(),
                    inputs: call
                        .inputs
                        .iter()
                        .copied()
                        .map(&mut remap)
                        .collect::<Result<_, _>>()?,
                })
            })
            .collect::<Result<_, String>>()?;
        let outputs = returned
            .iter()
            .copied()
            .map(&mut remap)
            .collect::<Result<_, _>>()?;
        drop(remap);
        let region = Graph {
            inputs: inputs.len(),
            weights: weights.len(),
            states: 0,
            mutable_inputs: vec![],
            calls,
            constants,
            outputs,
            writes: vec![],
        };
        region.validate()?;
        for (index, value) in returned.into_iter().enumerate() {
            sources.insert(value, Source::Output { part, index });
        }
        parts.push(Region {
            graph: region,
            inputs,
            weights,
        });
    }
    let outputs = graph
        .outputs
        .iter()
        .map(|value| {
            sources
                .get(value)
                .copied()
                .ok_or_else(|| "graph regions lost an output".to_string())
        })
        .collect::<Result<_, _>>()?;
    Ok(SplitGraph {
        input_count: graph.inputs,
        parts,
        outputs,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::Builder;
    use lisa_backend::{Binary, Kernel, TensorOperation as T, Unary};

    fn source(value: Source, previous: &[Vec<Value>]) -> Value {
        match value {
            Source::Input(i) => Value::Input(i),
            Source::Output { part, index } => previous[part][index],
        }
    }
    fn fixture() -> Graph {
        let mut b = Builder::new(2, 2, 0);
        let constant = b.constant_f32(&[0.25], &[1]);
        let a = b.one(
            Kernel::Tensor(T::Binary(Binary::Add)),
            vec![Value::Input(0), Value::Weight(1)],
        );
        let c = b.one(
            Kernel::Tensor(T::Binary(Binary::Multiply)),
            vec![a, constant],
        );
        let d = b.one(
            Kernel::Tensor(T::Binary(Binary::Add)),
            vec![c, Value::Input(1)],
        );
        let e = b.one(Kernel::Tensor(T::Unary(Unary::Tanh)), vec![d]);
        // An earlier output survives past the middle region, and repeated,
        // constant, weight and untouched input outputs preserve their identities.
        let f = b.one(Kernel::Tensor(T::Binary(Binary::Add)), vec![e, a]);
        b.finish(vec![f, a, f, constant, Value::Weight(0), Value::Input(1)])
    }
    fn same_value(a: Value, b: Value, original: &Graph, rebuilt: &Graph) {
        if let (Value::Constant(a), Value::Constant(b)) = (a, b) {
            let (a, b) = (&original.constants[a], &rebuilt.constants[b]);
            assert_eq!(a.dtype, b.dtype);
            assert_eq!(a.shape, b.shape);
            assert_eq!(a.bytes, b.bytes);
        } else {
            assert_eq!(a, b);
        }
    }
    #[test]
    fn regions_reassemble_every_call_operand_and_output() {
        let graph = fixture();
        for cuts in [&[0, 2, 4, 5][..], &[0, 1, 2, 3, 4, 5], &[0, 5]] {
            let split = split(&graph, cuts).unwrap();
            let mut b = Builder::new(graph.inputs, graph.weights, 0);
            let mut previous = Vec::new();
            for (part, region) in split.parts.iter().enumerate() {
                assert!(region.inputs.iter().all(|s| match s {
                    Source::Input(i) => *i < graph.inputs,
                    Source::Output { part: producer, .. } => *producer < part,
                }));
                let inputs = region
                    .inputs
                    .iter()
                    .map(|s| source(*s, &previous))
                    .collect::<Vec<_>>();
                let weights = region
                    .weights
                    .iter()
                    .map(|i| Value::Weight(*i))
                    .collect::<Vec<_>>();
                previous.push(
                    b.append(region.graph.clone(), &inputs, &weights, &[])
                        .unwrap(),
                );
            }
            let outputs: Vec<_> = split
                .outputs
                .iter()
                .map(|s| source(*s, &previous))
                .collect();
            let rebuilt = b.finish(outputs);
            rebuilt.validate().unwrap();
            assert_eq!(graph.calls.len(), rebuilt.calls.len());
            for (before, after) in graph.calls.iter().zip(&rebuilt.calls) {
                assert_eq!(before.kernel, after.kernel);
                assert_eq!(before.inputs.len(), after.inputs.len());
                for (&a, &b) in before.inputs.iter().zip(&after.inputs) {
                    same_value(a, b, &graph, &rebuilt);
                }
            }
            assert_eq!(graph.outputs.len(), rebuilt.outputs.len());
            for (&a, &b) in graph.outputs.iter().zip(&rebuilt.outputs) {
                same_value(a, b, &graph, &rebuilt);
            }
        }
    }
    #[test]
    fn invalid_cuts_and_mutation_are_rejected() {
        for cuts in [
            &[][..],
            &[0],
            &[1, 5],
            &[0, 4],
            &[0, 2, 2, 5],
            &[0, 4, 3, 5],
        ] {
            assert!(split(&fixture(), cuts).is_err());
        }
        let mut graph = fixture();
        graph.mutable_inputs.push(0);
        assert!(split(&graph, &[0, 2, 4, 5]).is_err());
        graph.mutable_inputs.clear();
        graph.states = 1;
        graph.writes.push(crate::graph::StateWrite {
            slot: 0,
            value: graph.outputs[0],
        });
        assert!(split(&graph, &[0, 2, 4, 5]).is_err());
    }
}
