//! Named checkpoint bindings and tensor graph construction. These helpers only
//! emit data; they never own device tensors or invoke an executor.
use crate::graph::{Builder, Graph, Value};
use lisa_backend::{Binary, Kernel, ScalarType, TensorOperation as T, Unary};
use std::collections::BTreeMap;
pub type Schema = BTreeMap<String, Vec<i32>>;
pub struct Program {
    pub graph: Graph,
    pub weights: Vec<String>,
}
pub struct Build<'a> {
    pub graph: Builder,
    pub schema: &'a Schema,
    weights: Vec<String>,
}
impl<'a> Build<'a> {
    pub fn new(inputs: usize, schema: &'a Schema) -> Self {
        Self {
            graph: Builder::new(inputs, 0, 0),
            schema,
            weights: vec![],
        }
    }
    pub fn weight(&mut self, name: &str) -> Value {
        if let Some(i) = self.weights.iter().position(|n| n == name) {
            return Value::Weight(i);
        }
        let i = self.weights.len();
        self.weights.push(name.into());
        Value::Weight(i)
    }
    pub fn width(&self, name: &str) -> anyhow::Result<i32> {
        self.schema
            .get(name)
            .and_then(|s| s.first())
            .copied()
            .filter(|&n| n > 0)
            .ok_or_else(|| anyhow::anyhow!("missing or invalid matrix {name}"))
    }
    pub fn op(&mut self, op: T, args: impl Into<Vec<Value>>) -> Value {
        self.graph.one(Kernel::Tensor(op), args)
    }
    pub fn cast(&mut self, x: Value, dtype: ScalarType) -> Value {
        self.op(T::Cast(dtype), vec![x])
    }
    pub fn reshape(&mut self, x: Value, shape: &[i32]) -> Value {
        self.op(T::Reshape(shape.to_vec()), vec![x])
    }
    pub fn transpose(&mut self, x: Value, axes: &[i32]) -> Value {
        self.op(T::Transpose(axes.to_vec()), vec![x])
    }
    pub fn contiguous(&mut self, x: Value) -> Value {
        self.op(T::Contiguous, vec![x])
    }
    pub fn slice(&mut self, x: Value, axis: i32, start: i32, end: i32) -> Value {
        self.op(T::Slice { axis, start, end }, vec![x])
    }
    pub fn binary(&mut self, op: Binary, a: Value, b: Value) -> Value {
        self.op(T::Binary(op), vec![a, b])
    }
    pub fn unary(&mut self, op: Unary, x: Value) -> Value {
        self.op(T::Unary(op), vec![x])
    }
    pub fn concat(&mut self, values: Vec<Value>, axis: i32) -> Value {
        self.op(
            T::Concatenate {
                axis,
                inputs: values.len(),
            },
            values,
        )
    }
    pub fn scalar(&mut self, v: f32, dtype: ScalarType) -> Value {
        match dtype {
            ScalarType::BF16 => self.graph.constant_bf16(&[v], &[]),
            ScalarType::F32 => self.graph.scalar(v),
            _ => panic!("unsupported model constant dtype"),
        }
    }
    pub fn linear(&mut self, x: Value, name: &str) -> Value {
        let w = self.weight(&format!("{name}.weight"));
        let w = self.transpose(w, &[1, 0]);
        let mut y = self.op(T::Matmul, vec![x, w]);
        let name = format!("{name}.bias");
        if self.schema.contains_key(&name) {
            let bias = self.weight(&name);
            y = self.binary(Binary::Add, y, bias);
        }
        y
    }
    pub fn layer_norm(&mut self, x: Value, name: &str, epsilon: f32) -> Value {
        let w = self.weight(&format!("{name}.weight"));
        let b = self.weight(&format!("{name}.bias"));
        self.op(
            T::LayerNorm {
                epsilon: epsilon.to_bits(),
            },
            vec![x, w, b],
        )
    }
    pub fn rms(&mut self, x: Value, name: &str, epsilon: f32) -> Value {
        let x = self.contiguous(x);
        let w = self.weight(name);
        self.op(
            T::RmsNorm {
                epsilon: epsilon.to_bits(),
            },
            vec![x, w],
        )
    }
    /// BF16 operation order used by the vision merger, distinct from Laya's f32 GELU.
    pub fn gelu_exact(&mut self, x: Value) -> Value {
        let c = self.scalar(std::f32::consts::FRAC_1_SQRT_2, ScalarType::BF16);
        let a = self.binary(Binary::Multiply, x, c);
        let e = self.unary(Unary::Erf, a);
        let one = self.scalar(1., ScalarType::BF16);
        let e = self.binary(Binary::Add, e, one);
        let y = self.binary(Binary::Multiply, x, e);
        let half = self.scalar(0.5, ScalarType::BF16);
        self.binary(Binary::Multiply, y, half)
    }
    pub fn gelu_tanh(&mut self, x: Value, shape: &[i32]) -> Value {
        let three = self.scalar(3., ScalarType::BF16);
        let exponent = self.op(T::Broadcast(shape.to_vec()), vec![three]);
        let cube = self.binary(Binary::Power, x, exponent);
        let c = self.scalar(0.044715, ScalarType::BF16);
        let cube = self.binary(Binary::Multiply, cube, c);
        let a = self.binary(Binary::Add, x, cube);
        let c = self.scalar(0.7978845608028654, ScalarType::BF16);
        let a = self.binary(Binary::Multiply, a, c);
        let a = self.unary(Unary::Tanh, a);
        let one = self.scalar(1., ScalarType::BF16);
        let a = self.binary(Binary::Add, a, one);
        let a = self.binary(Binary::Multiply, x, a);
        let half = self.scalar(0.5, ScalarType::BF16);
        self.binary(Binary::Multiply, a, half)
    }
    pub fn finish(self, outputs: impl Into<Vec<Value>>) -> anyhow::Result<Program> {
        for name in &self.weights {
            anyhow::ensure!(
                self.schema.contains_key(name),
                "missing model weight {name}"
            );
        }
        let mut graph = self.graph.finish(outputs);
        graph.weights = self.weights.len();
        graph.validate().map_err(anyhow::Error::msg)?;
        Ok(Program {
            graph,
            weights: self.weights,
        })
    }
}
