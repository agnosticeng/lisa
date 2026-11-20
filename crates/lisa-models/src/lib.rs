//! Model computations as data. This crate cannot load weights, allocate device
//! tensors, run kernels or execute its plans. `lisa-engine` does those things.
pub mod decoder;
pub mod deepseek41;
pub mod dense;
pub mod glm53;
pub mod graph;
pub mod laya;
pub mod normalization;
pub mod position;
pub mod qwen3_5;
pub mod qwen4;

use lisa_backend::Kernel;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Value {
    Input,
    Weight(usize),
    Temporary(usize),
}

#[derive(Clone, Debug)]
pub struct Call {
    pub kernel: Kernel,
    /// Inputs refer only to weights, the request input or earlier calls.
    pub inputs: Vec<Value>,
}

#[derive(Clone, Debug)]
pub struct Plan {
    pub calls: Vec<Call>,
    pub output: Value,
}
impl Plan {
    pub fn graph(&self, weights: usize) -> Result<graph::Graph, String> {
        self.validate(weights)?;
        let mut g = graph::Builder::new(1, weights, 0);
        let mut outputs = Vec::new();
        for call in &self.calls {
            let bind = |v| match v {
                Value::Input => graph::Value::Input(0),
                Value::Weight(i) => graph::Value::Weight(i),
                Value::Temporary(i) => outputs[i],
            };
            let inputs = call.inputs.iter().copied().map(bind).collect::<Vec<_>>();
            outputs.push(g.one(call.kernel.clone(), inputs));
        }
        let out = match self.output {
            Value::Input => graph::Value::Input(0),
            Value::Weight(i) => graph::Value::Weight(i),
            Value::Temporary(i) => outputs[i],
        };
        Ok(g.finish(vec![out]))
    }

    /// Validate once at preparation, before dispatch or large weight imports.
    pub fn validate(&self, weights: usize) -> Result<(), String> {
        let valid = |value: Value, calls: usize| match value {
            Value::Input => true,
            Value::Weight(i) => i < weights,
            Value::Temporary(i) => i < calls,
        };
        for (i, call) in self.calls.iter().enumerate() {
            if call.kernel.outputs() != 1
                || call.inputs.is_empty()
                || call.inputs.len() > 4
                || call.inputs.len() != call.kernel.arity()
                || !call.inputs.iter().all(|&v| valid(v, i))
            {
                return Err(format!(
                    "invalid arguments or forward reference at plan call {i}"
                ));
            }
            if let Kernel::AffineLinear { group_size, bits } = call.kernel {
                if group_size <= 0 || !matches!(bits, 2 | 4 | 8) {
                    return Err(format!("unsupported affine geometry at plan call {i}"));
                }
            }
        }
        if !valid(self.output, self.calls.len()) {
            return Err("invalid plan output".into());
        }
        Ok(())
    }
}

/// A semantic shape guard supplied by the model, evaluated by the engine.
#[derive(Clone, Copy, Debug)]
pub struct DimensionGuard {
    pub min_rank: usize,
    pub axis: usize,
    pub equals: i32,
}

pub struct Alternatives {
    pub primary: Plan,
    /// Optional optimized plan; a declined kernel restarts the primary plan.
    pub guarded: Option<(DimensionGuard, Plan)>,
}

pub mod program;
pub mod vision;

pub mod media_attention;

pub mod flux;

pub mod gated_delta;

pub mod attention_cache;

pub mod embedding;

pub mod traversal;

pub mod identity;

pub mod packed;

pub mod metadata;

pub mod sequence;

pub mod checkpoint;

pub mod lfm3;
