//! Dataflow between bound model blocks, including optional carried state.
//! Operations describe subplans; this module never executes them.
#[derive(Clone, Debug)]
pub struct Call<O> {
    pub operation: O,
    pub inputs: Vec<usize>,
    pub outputs: Vec<usize>,
}
#[derive(Clone, Debug)]
pub struct Plan<O> {
    pub inputs: usize,
    pub slots: usize,
    pub calls: Vec<Call<O>>,
    pub outputs: Vec<usize>,
}
impl<O> Plan<O> {
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.inputs > self.slots || self.outputs.is_empty() {
            return Err("invalid block plan geometry");
        }
        let mut defined = vec![false; self.slots];
        defined[..self.inputs].fill(true);
        for call in &self.calls {
            if call.inputs.iter().any(|&i| i >= self.slots || !defined[i]) {
                return Err("undefined block input");
            }
            for &i in &call.outputs {
                if i >= self.slots || defined[i] {
                    return Err("invalid or duplicate block output");
                }
                defined[i] = true;
            }
        }
        if self.outputs.iter().any(|&i| i >= self.slots || !defined[i]) {
            return Err("undefined block result");
        }
        Ok(())
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn block_plan_rejects_undefined_inputs_and_redefinitions() {
        let mut p = Plan {
            inputs: 1,
            slots: 2,
            calls: vec![Call {
                operation: (),
                inputs: vec![0],
                outputs: vec![1],
            }],
            outputs: vec![1],
        };
        assert!(p.validate().is_ok());
        p.calls[0].inputs = vec![1];
        assert!(p.validate().is_err());
        p.calls[0].inputs = vec![0];
        p.calls[0].outputs = vec![0];
        assert!(p.validate().is_err());
    }
}
