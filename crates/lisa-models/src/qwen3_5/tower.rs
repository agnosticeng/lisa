//! Dense hybrid tower ordering, state geometry and token-window laws.
use super::config::Qwen35Config;
use crate::{
    graph::{Builder, Graph, Value},
    traversal::{Plan, Step},
};
use lisa_backend::{Kernel, TensorOperation as T};
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Cache {
    Full { heads: usize, width: usize },
    Linear,
}
pub fn caches(config: &Qwen35Config) -> Vec<Cache> {
    (0..config.num_hidden_layers)
        .map(|i| {
            if config.is_full(i) {
                Cache::Full {
                    heads: config.num_key_value_heads,
                    width: config.head_dim,
                }
            } else {
                Cache::Linear
            }
        })
        .collect()
}
pub fn bytes(config: &Qwen35Config, scalar_bytes: usize) -> (usize, usize) {
    let full = caches(config)
        .iter()
        .filter(|c| matches!(c, Cache::Full { .. }))
        .count();
    let ssm = config.linear_num_value_heads
        * config.linear_key_head_dim
        * config.linear_value_head_dim
        * 4;
    let conv = (config.linear_conv_kernel_dim - 1)
        * (2 * config.linear_num_key_heads * config.linear_key_head_dim
            + config.linear_num_value_heads * config.linear_value_head_dim)
        * scalar_bytes;
    (
        full * config.num_key_value_heads * config.head_dim * 2 * scalar_bytes,
        (config.num_hidden_layers - full) * (ssm + conv),
    )
}
pub fn traversal(layers: usize, capture: &[usize]) -> Plan {
    let mut steps = vec![Step::Begin];
    let mut count = 0;
    for i in 0..layers {
        steps.push(Step::Layer(i));
        if capture.contains(&i) {
            steps.push(Step::Capture);
            count += 1;
        }
    }
    steps.push(Step::Finish);
    Plan {
        steps,
        capture_join: (count > 0).then(|| crate::traversal::concatenate(count, -1)),
    }
}
#[derive(Clone, Copy, Debug)]
pub struct Window {
    pub batch: i32,
    pub rows: i32,
    pub batch_supported: bool,
    pub capture: bool,
    pub max_depth: usize,
    pub layers: usize,
    pub caches: usize,
    pub offset: usize,
    pub capacity: usize,
    pub chunk: Option<usize>,
}
impl Window {
    pub fn validate(&self) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.batch > 0 && self.rows > 0,
            "Qwen expects nonempty [batch,tokens]"
        );
        anyhow::ensure!(
            self.batch_supported || self.batch == 1,
            "this Qwen execution policy supports one stream per forward"
        );
        anyhow::ensure!(
            !self.capture || self.rows as usize <= self.max_depth.saturating_add(1),
            "verification window exceeds this checkpoint's speculative depth limit"
        );
        anyhow::ensure!(
            self.layers == self.caches,
            "Qwen layer/cache count mismatch"
        );
        anyhow::ensure!(
            self.offset
                .checked_add(self.rows as usize)
                .is_some_and(|n| n <= self.capacity),
            "Qwen context capacity exceeded"
        );
        anyhow::ensure!(self.chunk != Some(0), "Qwen forward chunk must be positive");
        Ok(())
    }
    pub fn chunks(&self) -> Option<Vec<(i32, i32)>> {
        self.chunk
            .filter(|&n| !self.capture && self.rows as usize > n)
            .map(|n| {
                (0..self.rows as usize)
                    .step_by(n)
                    .map(|start| (start as i32, (start + n).min(self.rows as usize) as i32))
                    .collect()
            })
    }
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum View {
    Tokens(i32, i32),
    Join(usize, i32),
    Last(i32),
}
impl View {
    pub fn graph(&self) -> Graph {
        match *self {
            Self::Join(inputs, axis) => crate::traversal::concatenate(inputs, axis),
            _ => {
                let mut g = Builder::new(1, 0, 0);
                let op = match *self {
                    Self::Tokens(start, end) => T::Slice {
                        axis: 1,
                        start,
                        end,
                    },
                    Self::Last(rows) => T::Select {
                        axis: 1,
                        index: rows - 1,
                    },
                    _ => unreachable!(),
                };
                let out = g.one(Kernel::Tensor(op), vec![Value::Input(0)]);
                g.finish(vec![out])
            }
        }
    }
}
pub fn context_capacity(
    requested: usize,
    maximum: usize,
    supported: bool,
) -> anyhow::Result<usize> {
    anyhow::ensure!(
        supported,
        "runtime context capacity is not supported by this checkpoint; omit --ctx-size"
    );
    anyhow::ensure!(
        requested > 0 && requested <= maximum,
        "invalid Qwen context capacity"
    );
    Ok(requested)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn traversal_and_windows_preserve_capture_order_and_bounds() {
        let plan = traversal(4, &[3, 1]);
        plan.validate().unwrap();
        assert_eq!(
            plan.steps,
            vec![
                Step::Begin,
                Step::Layer(0),
                Step::Layer(1),
                Step::Capture,
                Step::Layer(2),
                Step::Layer(3),
                Step::Capture,
                Step::Finish
            ]
        );
        let mut w = Window {
            batch: 1,
            rows: 9,
            batch_supported: true,
            capture: false,
            max_depth: 6,
            layers: 4,
            caches: 4,
            offset: 0,
            capacity: 16,
            chunk: Some(4),
        };
        w.validate().unwrap();
        assert_eq!(w.chunks().unwrap(), vec![(0, 4), (4, 8), (8, 9)]);
        w.capture = true;
        assert!(w.validate().is_err());
        w.rows = 7;
        w.validate().unwrap();
        assert!(w.chunks().is_none());
        w.offset = usize::MAX;
        assert!(w.validate().is_err());
    }
}
