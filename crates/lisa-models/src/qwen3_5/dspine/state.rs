//! Draft cache retention and vocabulary-table construction laws.
use crate::{
    attention_cache::Append,
    graph::{Graph, Value},
    program::Build,
};
use lisa_backend::{Kernel, ScalarType, TensorOperation as T};
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Plan {
    Compact {
        batch: i32,
        heads: i32,
        width: i32,
        offset: usize,
        keep: usize,
        dtype: ScalarType,
    },
    Tables {
        vocab: i32,
        rank: i32,
    },
    Fill {
        start: i32,
        end: i32,
    },
}
impl Plan {
    pub fn graph(&self) -> anyhow::Result<Graph> {
        let schema = Default::default();
        match self {
            Self::Compact {
                batch,
                heads,
                width,
                offset,
                keep,
                dtype,
            } => {
                anyhow::ensure!(
                    *keep > 0 && offset >= keep && *offset <= i32::MAX as usize,
                    "invalid retained draft window"
                );
                let mut p = Build::new(2, &schema);
                let k = p.slice(Value::Input(0), 2, (offset - keep) as i32, *offset as i32);
                let v = p.slice(Value::Input(1), 2, (offset - keep) as i32, *offset as i32);
                let append = Append {
                    batch: *batch,
                    heads: *heads,
                    width: *width,
                    rows: *keep as i32,
                    offset: 0,
                    capacity: 0,
                    previous_batch: None,
                    dtype: *dtype,
                };
                let out = p
                    .graph
                    .append(
                        append.graph().map_err(anyhow::Error::msg)?,
                        &[k, v],
                        &[],
                        &[],
                    )
                    .map_err(anyhow::Error::msg)?;
                Ok(p.finish(out[..2].to_vec())?.graph)
            }
            Self::Tables { vocab, rank } => {
                anyhow::ensure!(*vocab > 0 && *rank > 0, "invalid table geometry");
                let mut p = Build::new(0, &schema);
                let tg = p.graph.one(
                    Kernel::Zeros {
                        shape: vec![*vocab, *rank],
                        dtype: ScalarType::BF16,
                    },
                    vec![],
                );
                let tsv = p.graph.one(
                    Kernel::Zeros {
                        shape: vec![*vocab, 2 * rank],
                        dtype: ScalarType::BF16,
                    },
                    vec![],
                );
                Ok(p.finish(vec![tg, tsv])?.graph)
            }
            Self::Fill { start, end } => {
                anyhow::ensure!(*start >= 0 && end > start, "invalid table chunk");
                let mut p = Build::new(4, &schema);
                p.graph.writable_input(2);
                p.graph.writable_input(3);
                let tg = p.slice(Value::Input(0), 0, 0, end - start);
                let tsv = p.slice(Value::Input(1), 0, 0, end - start);
                let op = T::CopySliceInto {
                    axis: 0,
                    start: *start,
                    end: *end,
                };
                let tg = p.op(op.clone(), vec![tg, Value::Input(2)]);
                let tsv = p.op(op, vec![tsv, Value::Input(3)]);
                Ok(p.finish(vec![tg, tsv])?.graph)
            }
        }
    }
}
pub fn retained_start(offset: usize, window: Option<usize>) -> usize {
    window.map_or(0, |w| offset.saturating_sub(w.saturating_sub(1)))
}
pub fn compaction(live: usize, window: Option<usize>) -> Option<(usize, usize)> {
    let keep = window?.checked_sub(1)?;
    (live > keep.saturating_add(2048)).then_some((keep, live.saturating_sub(keep)))
}
pub fn table_ids(start: usize, chunk: usize, vocab: usize) -> Vec<u32> {
    (start..start + chunk)
        .map(|id| if id < vocab { id as u32 } else { 0 })
        .collect()
}
