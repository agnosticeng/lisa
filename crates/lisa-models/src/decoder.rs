//! Decoder dataflow. These semantic blocks compose kernel subplans; their
//! implementations are supplied by the engine during the incremental migration.
//! No device tensors, device allocations or execution are allowed in this module.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mixer {
    Attention,
    GatedDelta,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Step {
    /// Add an optional carried residual, then normalize with the selected weight.
    /// Writes both the rounded sum and its normalized value.
    ResidualNorm {
        norm: usize,
        input: usize,
        residual: usize,
        sum: usize,
        normalized: usize,
    },
    Mix {
        kind: Mixer,
        input: usize,
        output: usize,
    },
    FeedForward {
        input: usize,
        output: usize,
    },
}
#[derive(Clone, Copy, Debug)]
pub struct Plan {
    pub steps: &'static [Step],
    pub slots: usize,
    pub output: (usize, usize),
}
impl Plan {
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.slots < 2 {
            return Err("decoder needs input and optional residual slots");
        }
        let mut defined = vec![false; self.slots];
        defined[0] = true;
        defined[1] = true;
        for step in self.steps {
            let (inputs, outputs): (&[usize], &[usize]) = match step {
                Step::ResidualNorm {
                    input,
                    residual,
                    sum,
                    normalized,
                    ..
                } => (&[*input, *residual], &[*sum, *normalized]),
                Step::Mix { input, output, .. } | Step::FeedForward { input, output } => {
                    (&[*input], &[*output])
                }
            };
            if inputs.iter().any(|&i| i >= self.slots || !defined[i]) {
                return Err("undefined decoder input");
            }
            for &i in outputs {
                if i >= self.slots || defined[i] {
                    return Err("invalid or multiply defined decoder output");
                }
                defined[i] = true;
            }
        }
        if self.output.0 == self.output.1
            || self.output.0 >= self.slots
            || self.output.1 >= self.slots
            || !defined[self.output.0]
            || !defined[self.output.1]
        {
            return Err("undefined decoder result");
        }
        Ok(())
    }
}
