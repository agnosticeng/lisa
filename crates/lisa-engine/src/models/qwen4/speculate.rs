//! MTP helpers.
//!
//! The speculative driver lives on [`crate::core::session::Session::generate_mtp`]
//! (single-stream, and the per-stream path the scheduler uses). This module
//! keeps the small helpers it shares.

use lisa_mlx::ops::indexing::argmax;
use lisa_mlx::Array;

pub(crate) fn argmax_id(a: &Array) -> anyhow::Result<u32> {
    let out = argmax(a, None).map_err(|e| anyhow::anyhow!("{e}"))?;
    out.eval().map_err(|e| anyhow::anyhow!("{e}"))?;
    Ok(out.item_cast::<i32>() as u32)
}

pub(crate) fn concat_rows(multis: &[Array]) -> anyhow::Result<Array> {
    let mut rows: Vec<Array> = Vec::with_capacity(multis.len());
    for m in multis {
        rows.push(m.reshape(&[1, 1, m.dim(-1)])?);
    }
    let refs: Vec<&Array> = rows.iter().collect();
    lisa_mlx::ops::concatenate(&refs, 1).map_err(|e| anyhow::anyhow!("{e}"))
}
