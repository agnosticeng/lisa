//! Hyper-stream decoder dataflow and placement of the optional PLE block.
use crate::sequence::{Call, Plan};
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Step {
    Scale(usize),
    InjectNorm,
    InjectNormPlain,
    Ple,
    Add,
    HyperMix(usize),
    Mixer,
    Experts,
}
pub fn has_ple(available: bool, context: bool) -> bool {
    available && context
}
pub fn schedule(ple: bool) -> Plan<Step> {
    let mut calls = vec![];
    let mut add = |operation, inputs: &[usize], outputs: &[usize]| {
        calls.push(Call {
            operation,
            inputs: inputs.to_vec(),
            outputs: outputs.to_vec(),
        })
    };
    add(Step::Scale(0), &[], &[3]);
    add(Step::Scale(1), &[], &[4]);
    if ple {
        add(Step::InjectNorm, &[0, 1, 2, 3], &[5, 6]);
        add(Step::Ple, &[5], &[7]);
        add(Step::Add, &[5, 7], &[8]);
        add(Step::InjectNormPlain, &[8, 3], &[9, 10]);
    } else {
        add(Step::InjectNorm, &[0, 1, 2, 3], &[9, 10]);
    }
    add(Step::HyperMix(0), &[10], &[11, 12]);
    add(Step::Mixer, &[11], &[13]);
    add(Step::InjectNorm, &[9, 13, 12, 4], &[14, 15]);
    add(Step::HyperMix(1), &[15], &[16, 17]);
    add(Step::Experts, &[16], &[18]);
    Plan {
        inputs: 3,
        slots: 19,
        calls,
        outputs: vec![14, 18, 17],
    }
}
pub fn add() -> crate::graph::Graph {
    let mut g = crate::graph::Builder::new(2, 0, 0);
    let out = g.one(
        lisa_backend::Kernel::Tensor(lisa_backend::TensorOperation::Binary(
            lisa_backend::Binary::Add,
        )),
        vec![crate::graph::Value::Input(0), crate::graph::Value::Input(1)],
    );
    g.finish(vec![out])
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn schedules_preserve_injection_and_ple_order() {
        for ple in [false, true] {
            let p = schedule(ple);
            p.validate().unwrap();
            assert_eq!(p.outputs, vec![14, 18, 17]);
            assert_eq!(
                p.calls.iter().filter(|c| c.operation == Step::Ple).count(),
                usize::from(ple)
            );
            assert_eq!(p.calls.last().unwrap().operation, Step::Experts);
        }
    }
}
