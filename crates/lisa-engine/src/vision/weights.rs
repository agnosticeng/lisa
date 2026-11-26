//! Tensor lookup shared by the vision executors.
use anyhow::Context;
use std::collections::HashMap;
pub(super) fn get<T: Clone>(w: &HashMap<String, T>, name: &str) -> anyhow::Result<T> {
    w.get(name)
        .cloned()
        .with_context(|| format!("missing vision tensor {name}"))
}
