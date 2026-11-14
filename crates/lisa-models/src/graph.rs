//! Portable tensor dataflow with explicit inputs, multi-output calls and state
//! commits. A graph is data; only the engine may prepare or execute it.
use lisa_backend::{HostTensor, Kernel, ScalarType};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Value {
    Input(usize),
    Weight(usize),
    State(usize),
    Constant(usize),
    Result { call: usize, output: usize },
}
#[derive(Clone, Debug)]
pub struct Call {
    pub kernel: Kernel,
    pub inputs: Vec<Value>,
}
#[derive(Clone, Debug)]
pub struct StateWrite {
    pub slot: usize,
    pub value: Value,
}
#[derive(Clone, Debug)]
pub struct Constant {
    pub bytes: Vec<u8>,
    pub shape: Vec<i32>,
    pub dtype: ScalarType,
}
#[derive(Clone, Debug)]
pub struct Graph {
    pub inputs: usize,
    pub weights: usize,
    pub states: usize,
    /// Explicit writable output-workspace inputs. Their bytes may change even
    /// when execution fails; state handle commits remain all-or-nothing.
    pub mutable_inputs: Vec<usize>,
    pub calls: Vec<Call>,
    pub constants: Vec<Constant>,
    pub outputs: Vec<Value>,
    /// Install new handles only after all calls succeed. Explicitly declared
    /// workspace mutations are immediate and cannot be rolled back.
    pub writes: Vec<StateWrite>,
}
impl Graph {
    pub fn validate(&self) -> Result<(), String> {
        let mut mutable = std::collections::HashSet::new();
        if self
            .mutable_inputs
            .iter()
            .any(|&i| i >= self.inputs || !mutable.insert(i))
        {
            return Err("invalid mutable input declaration".into());
        }
        for constant in &self.constants {
            if !(HostTensor {
                bytes: &constant.bytes,
                shape: &constant.shape,
                dtype: constant.dtype,
            })
            .validate()
            {
                return Err("invalid graph constant".into());
            }
        }
        let valid = |value: Value, limit: usize| match value {
            Value::Input(i) => i < self.inputs,
            Value::Weight(i) => i < self.weights,
            Value::State(i) => i < self.states,
            Value::Constant(i) => i < self.constants.len(),
            Value::Result { call, output } => {
                call < limit && output < self.calls[call].kernel.outputs()
            }
        };
        for (i, call) in self.calls.iter().enumerate() {
            if call.inputs.len() != call.kernel.arity() || call.inputs.iter().any(|&v| !valid(v, i))
            {
                return Err(format!("invalid graph call {i}"));
            }
            for &slot in call.kernel.overwritten_inputs() {
                let mut source = call.inputs.get(slot).copied();
                let writable = loop {
                    match source {
                        Some(Value::Input(i)) => break mutable.contains(&i),
                        Some(Value::Result { call, output }) => {
                            let producer = &self.calls[call];
                            if producer.kernel.fresh_output(output) {
                                break true;
                            }
                            source = producer
                                .kernel
                                .alias_input(output)
                                .and_then(|i| producer.inputs.get(i).copied());
                        }
                        _ => break false,
                    }
                };
                if !writable {
                    return Err(format!("undeclared storage mutation in call {i}"));
                }
            }
        }
        if self.outputs.is_empty() || self.outputs.iter().any(|&v| !valid(v, self.calls.len())) {
            return Err("invalid graph result".into());
        }
        let mut assigned = std::collections::HashSet::new();
        for write in &self.writes {
            if write.slot >= self.states
                || !assigned.insert(write.slot)
                || !valid(write.value, self.calls.len())
            {
                return Err("invalid or duplicate state commit".into());
            }
        }
        Ok(())
    }
}

pub struct Builder {
    graph: Graph,
}
impl Builder {
    pub fn new(inputs: usize, weights: usize, states: usize) -> Self {
        Self {
            graph: Graph {
                inputs,
                weights,
                states,
                mutable_inputs: vec![],
                calls: vec![],
                constants: vec![],
                outputs: vec![],
                writes: vec![],
            },
        }
    }
    /// Inline a validated subgraph as data. The model decides composition;
    /// the executor still sees a flat sequence of individual device operations.
    pub fn append(
        &mut self,
        graph: Graph,
        inputs: &[Value],
        weights: &[Value],
        states: &[Value],
    ) -> Result<Vec<Value>, String> {
        graph.validate()?;
        if (graph.inputs, graph.weights, graph.states)
            != (inputs.len(), weights.len(), states.len())
        {
            return Err("subgraph binding count differs".into());
        }
        // Validate external effects before modifying the builder.
        for &i in &graph.mutable_inputs {
            if !matches!(inputs[i], Value::Input(_)) {
                return Err("subgraph output storage must bind an external input".into());
            }
        }
        for write in &graph.writes {
            if !matches!(states[write.slot], Value::State(_)) {
                return Err("subgraph state commit must bind an external state".into());
            }
        }
        let calls = self.graph.calls.len();
        let constants = self.graph.constants.len();
        let remap = |v| match v {
            Value::Input(i) => inputs[i],
            Value::Weight(i) => weights[i],
            Value::State(i) => states[i],
            Value::Constant(i) => Value::Constant(constants + i),
            Value::Result { call, output } => Value::Result {
                call: calls + call,
                output,
            },
        };
        for i in graph.mutable_inputs {
            if let Value::Input(slot) = inputs[i] {
                if !self.graph.mutable_inputs.contains(&slot) {
                    self.graph.mutable_inputs.push(slot);
                }
            }
        }
        self.graph.constants.extend(graph.constants);
        self.graph
            .calls
            .extend(graph.calls.into_iter().map(|call| Call {
                kernel: call.kernel,
                inputs: call.inputs.into_iter().map(remap).collect(),
            }));
        self.graph
            .writes
            .extend(graph.writes.into_iter().map(|write| {
                let Value::State(slot) = states[write.slot] else {
                    unreachable!()
                };
                StateWrite {
                    slot,
                    value: remap(write.value),
                }
            }));
        Ok(graph.outputs.into_iter().map(remap).collect())
    }
    pub fn writable_input(&mut self, slot: usize) {
        self.graph.mutable_inputs.push(slot);
    }
    pub fn constant(&mut self, bytes: Vec<u8>, shape: Vec<i32>, dtype: ScalarType) -> Value {
        let index = self.graph.constants.len();
        self.graph.constants.push(Constant {
            bytes,
            shape,
            dtype,
        });
        Value::Constant(index)
    }
    pub fn constant_i32(&mut self, data: &[i32], shape: &[i32]) -> Value {
        self.constant(
            data.iter().flat_map(|v| v.to_le_bytes()).collect(),
            shape.to_vec(),
            ScalarType::I32,
        )
    }
    pub fn constant_bf16(&mut self, data: &[f32], shape: &[i32]) -> Value {
        self.constant(
            data.iter()
                .flat_map(|&v| half::bf16::from_f32(v).to_le_bytes())
                .collect(),
            shape.to_vec(),
            ScalarType::BF16,
        )
    }
    pub fn constant_f32(&mut self, data: &[f32], shape: &[i32]) -> Value {
        let index = self.graph.constants.len();
        self.graph.constants.push(Constant {
            bytes: data.iter().flat_map(|v| v.to_le_bytes()).collect(),
            shape: shape.to_vec(),
            dtype: ScalarType::F32,
        });
        Value::Constant(index)
    }
    pub fn scalar(&mut self, value: f32) -> Value {
        self.constant_f32(&[value], &[])
    }
    pub fn call(&mut self, kernel: Kernel, inputs: impl Into<Vec<Value>>) -> Vec<Value> {
        let call = self.graph.calls.len();
        let outputs = kernel.outputs();
        self.graph.calls.push(Call {
            kernel,
            inputs: inputs.into(),
        });
        (0..outputs)
            .map(|output| Value::Result { call, output })
            .collect()
    }
    pub fn one(&mut self, kernel: Kernel, inputs: impl Into<Vec<Value>>) -> Value {
        assert_eq!(
            kernel.outputs(),
            1,
            "one() requires a single-output operation"
        );
        self.call(kernel, inputs)[0]
    }
    pub fn commit(&mut self, slot: usize, value: Value) {
        self.graph.writes.push(StateWrite { slot, value });
    }
    pub fn finish(mut self, outputs: impl Into<Vec<Value>>) -> Graph {
        self.graph.outputs = outputs.into();
        self.graph
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lisa_backend::TensorOperation;
    #[test]
    fn composition_rebases_explicit_mutations_and_rejects_weight_destinations() {
        let mut child = Builder::new(8, 0, 0);
        child.writable_input(6);
        child.writable_input(7);
        let out = child.call(
            Kernel::DeltaScan {
                mode: lisa_backend::DeltaScan::TwoRows,
                capture: true,
                reuse: true,
            },
            (0..8).map(Value::Input).collect::<Vec<_>>(),
        );
        let graph = child.finish(out);
        let mut root = Builder::new(10, 1, 0);
        let mut inputs = (2..10).map(Value::Input).collect::<Vec<_>>();
        inputs[7] = Value::Weight(0);
        assert!(root.append(graph.clone(), &inputs, &[], &[]).is_err());
        inputs[7] = Value::Input(9);
        let out = root.append(graph, &inputs, &[], &[]).unwrap();
        let graph = root.finish(out);
        graph.validate().unwrap();
        assert_eq!(graph.mutable_inputs, vec![8, 9]);
    }
    #[test]
    fn fresh_storage_is_writable_but_ordinary_values_are_not() {
        let mut b = Builder::new(1, 1, 0);
        let zero = b.one(
            Kernel::Zeros {
                shape: vec![1, 4, 8],
                dtype: ScalarType::BF16,
            },
            vec![],
        );
        let out = b.one(
            Kernel::CopyRowsInto { offset: 0 },
            vec![Value::Input(0), zero],
        );
        let mut g = b.finish(vec![out]);
        g.validate().unwrap();
        g.calls[1].inputs[1] = Value::Weight(0);
        assert!(g.validate().is_err());
        g.calls[1].inputs[1] = Value::Input(0);
        assert!(g.validate().is_err());
        g.mutable_inputs.push(0);
        g.validate().unwrap();
        g.calls[0].kernel = Kernel::Tensor(TensorOperation::Contiguous);
        g.calls[0].inputs = vec![Value::Input(0)];
        g.calls[1].inputs[1] = zero;
        assert!(g.validate().is_err());
    }
    #[test]
    fn writes_through_views_retain_original_storage_authorization() {
        let mut b = Builder::new(1, 1, 0);
        let v = b.one(
            Kernel::Tensor(TensorOperation::Reshape(vec![1, 64])),
            vec![Value::Weight(0)],
        );
        let out = b.one(
            Kernel::IndexedAttention(lisa_backend::IndexedAttention::IndexQuery),
            vec![v, Value::Input(0)],
        );
        let mut g = b.finish(vec![out]);
        assert!(g.validate().is_err());
        g.calls[0].inputs[0] = Value::Input(0);
        assert!(g.validate().is_err());
        g.mutable_inputs.push(0);
        g.validate().unwrap();
        g.mutable_inputs.clear();
        g.calls.insert(
            0,
            Call {
                kernel: Kernel::PackedCompute(lisa_backend::PackedCompute::RmsNorm {
                    weighted: false,
                    round_output: false,
                    epsilon: 1e-6f32.to_bits(),
                }),
                inputs: vec![Value::Input(0)],
            },
        );
        g.calls[1].inputs[0] = Value::Result { call: 0, output: 0 };
        g.calls[2].inputs[0] = Value::Result { call: 1, output: 0 };
        g.outputs[0] = Value::Result { call: 2, output: 0 };
        g.validate().unwrap();
    }
    #[test]
    fn output_reuse_requires_explicit_writable_bindings() {
        let mut b = Builder::new(8, 0, 0);
        let out = b.call(
            Kernel::DeltaScan {
                mode: lisa_backend::DeltaScan::TwoRows,
                capture: true,
                reuse: true,
            },
            (0..8).map(Value::Input).collect::<Vec<_>>(),
        );
        let mut graph = b.finish(out);
        assert!(graph.validate().is_err());
        graph.mutable_inputs = vec![6];
        assert!(graph.validate().is_err());
        graph.mutable_inputs.push(7);
        graph.validate().unwrap();
        graph.mutable_inputs.push(7);
        assert!(graph.validate().is_err());
        graph.mutable_inputs = vec![6, 7, 8];
        assert!(graph.validate().is_err());
    }
    #[test]
    fn multi_output_dataflow_and_transactional_state_are_validated() {
        let mut b = Builder::new(2, 1, 1);
        let out = b.call(
            Kernel::Tensor(TensorOperation::AddRmsNorm {
                epsilon: 1e-6f32.to_bits(),
            }),
            vec![Value::Input(0), Value::Input(1), Value::Weight(0)],
        );
        b.commit(0, out[0]);
        let mut graph = b.finish(vec![out[1], out[0]]);
        graph.validate().unwrap();
        graph.outputs.push(Value::Result { call: 0, output: 2 });
        assert!(graph.validate().is_err());
        graph.outputs.pop();
        graph.calls[0].inputs[0] = out[0];
        assert!(graph.validate().is_err());
    }
}
