//! Execution of model-owned decoder dataflow. Block bindings are supplied by
//! load-time adapters; this executor neither selects a model nor chooses order.
use lisa_models::decoder::{Mixer, Plan, Step};

pub trait Blocks {
    type Tensor: Clone;
    fn residual_norm(
        &mut self,
        norm: usize,
        input: &Self::Tensor,
        residual: Option<&Self::Tensor>,
    ) -> anyhow::Result<(Self::Tensor, Self::Tensor)>;
    fn mix(&mut self, kind: Mixer, input: &Self::Tensor) -> anyhow::Result<Self::Tensor>;
    fn feed_forward(&mut self, input: &Self::Tensor) -> anyhow::Result<Self::Tensor>;
}

enum Slot<'a, T> {
    Borrowed(&'a T),
    Owned(T),
}
impl<T: Clone> Slot<'_, T> {
    fn get(&self) -> &T {
        match self {
            Self::Borrowed(t) => t,
            Self::Owned(t) => t,
        }
    }
    fn into_owned(self) -> T {
        match self {
            Self::Borrowed(t) => t.clone(),
            Self::Owned(t) => t,
        }
    }
}

/// Preparation validates every slot. The stack register file avoids per-layer
/// heap allocation; the model's output references determine the returned state.
pub struct PreparedDecoder<const SLOTS: usize> {
    plan: Plan,
}
impl<const SLOTS: usize> PreparedDecoder<SLOTS> {
    pub fn new(plan: Plan) -> anyhow::Result<Self> {
        anyhow::ensure!(plan.slots <= SLOTS, "decoder register capacity exceeded");
        plan.validate().map_err(anyhow::Error::msg)?;
        Ok(Self { plan })
    }
    pub fn run<B: Blocks>(
        &self,
        blocks: &mut B,
        input: &B::Tensor,
        residual: Option<B::Tensor>,
    ) -> anyhow::Result<(B::Tensor, B::Tensor)> {
        let mut values: [Option<Slot<'_, B::Tensor>>; SLOTS] = std::array::from_fn(|_| None);
        values[0] = Some(Slot::Borrowed(input));
        values[1] = residual.map(Slot::Owned);
        for step in self.plan.steps {
            let required = |slot: usize| {
                values[slot]
                    .as_ref()
                    .map(Slot::get)
                    .ok_or_else(|| anyhow::anyhow!("required decoder input {slot} is absent"))
            };
            match *step {
                Step::ResidualNorm {
                    norm,
                    input,
                    residual,
                    sum,
                    normalized,
                } => {
                    let (s, n) = blocks.residual_norm(
                        norm,
                        required(input)?,
                        values[residual].as_ref().map(Slot::get),
                    )?;
                    values[sum] = Some(Slot::Owned(s));
                    values[normalized] = Some(Slot::Owned(n));
                }
                Step::Mix {
                    kind,
                    input,
                    output,
                } => {
                    let result = blocks.mix(kind, required(input)?)?;
                    values[output] = Some(Slot::Owned(result));
                }
                Step::FeedForward { input, output } => {
                    let result = blocks.feed_forward(required(input)?)?;
                    values[output] = Some(Slot::Owned(result));
                }
            }
        }
        let (sum, delta) = self.plan.output;
        let sum = values[sum]
            .take()
            .ok_or_else(|| anyhow::anyhow!("missing decoder output"))?
            .into_owned();
        let delta = values[delta]
            .take()
            .ok_or_else(|| anyhow::anyhow!("missing decoder output"))?
            .into_owned();
        Ok((sum, delta))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    struct Recorder {
        calls: Vec<&'static str>,
    }
    impl Blocks for Recorder {
        type Tensor = i32;
        fn residual_norm(
            &mut self,
            norm: usize,
            input: &i32,
            residual: Option<&i32>,
        ) -> anyhow::Result<(i32, i32)> {
            self.calls
                .push(if norm == 0 { "input_norm" } else { "post_norm" });
            let sum = *input + residual.copied().unwrap_or(0);
            Ok((sum, sum * 2))
        }
        fn mix(&mut self, kind: Mixer, input: &i32) -> anyhow::Result<i32> {
            self.calls.push(if kind == Mixer::Attention {
                "attention"
            } else {
                "gated_delta"
            });
            Ok(*input + 3)
        }
        fn feed_forward(&mut self, input: &i32) -> anyhow::Result<i32> {
            self.calls.push("mlp");
            Ok(*input * 5)
        }
    }
    #[test]
    fn executes_model_order_and_returns_delayed_residual() -> anyhow::Result<()> {
        for full in [false, true] {
            let prepared = PreparedDecoder::<8>::new(lisa_models::qwen3_5::decoder(full))?;
            for residual in [None, Some(2)] {
                let mut b = Recorder { calls: vec![] };
                let (sum, delta) = prepared.run(&mut b, &1, residual)?;
                let want = (1 + residual.unwrap_or(0)) * 3 + 3;
                assert_eq!((sum, delta), (want, want * 10));
                assert_eq!(
                    b.calls,
                    vec![
                        "input_norm",
                        if full { "attention" } else { "gated_delta" },
                        "post_norm",
                        "mlp"
                    ]
                );
            }
        }
        Ok(())
    }
}
