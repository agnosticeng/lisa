//! Model-owned traversal over bound blocks, with explicit capture and finish order.
use crate::graph::{Builder, Graph, Value};
use lisa_backend::{Kernel, TensorOperation};
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Step {
    Begin,
    Layer(usize),
    Capture,
    Finish,
}
#[derive(Clone, Debug)]
pub struct Plan {
    pub steps: Vec<Step>,
    pub capture_join: Option<Graph>,
}
impl Plan {
    pub fn validate(&self) -> Result<(), String> {
        if self.steps.first() != Some(&Step::Begin)
            || self.steps.last() != Some(&Step::Finish)
            || self.steps[1..].contains(&Step::Begin)
            || self.steps[..self.steps.len() - 1].contains(&Step::Finish)
        {
            return Err("invalid traversal boundary".into());
        }
        let captures = self
            .steps
            .iter()
            .filter(|s| matches!(s, Step::Capture))
            .count();
        match &self.capture_join {
            Some(g) => {
                g.validate()?;
                if g.inputs != captures || g.weights != 0 || g.states != 0 || g.outputs.len() != 1 {
                    return Err("invalid traversal capture join".into());
                }
            }
            None => {
                if captures != 0 {
                    return Err("missing traversal capture join".into());
                }
            }
        }
        Ok(())
    }
}
pub fn concatenate(inputs: usize, axis: i32) -> Graph {
    let mut g = Builder::new(inputs, 0, 0);
    let out = g.one(
        Kernel::Tensor(TensorOperation::Concatenate { axis, inputs }),
        (0..inputs).map(Value::Input).collect::<Vec<_>>(),
    );
    g.finish(vec![out])
}
