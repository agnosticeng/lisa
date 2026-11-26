#![cfg(test)]
//! Temporary tensor helpers for legacy GLM paths still awaiting extraction.
//! Residual computation is defined in lisa-models and executed by its binding.
use lisa_metal::{Array, ops};

pub fn fscalar(a: &Array, v: f32) -> lisa_metal::error::Result<Array> {
    Ok(Array::from_f32_as(v, a.dtype()))
}

/// Concatenate owned arrays along `axis` (the wrapper op borrows).
pub fn concat(arrays: Vec<Array>, axis: i32) -> lisa_metal::error::Result<Array> {
    let refs: Vec<&Array> = arrays.iter().collect();
    ops::concatenate(&refs, axis)
}

/// Concatenate borrowed arrays along `axis`.
pub fn concat_refs(arrays: &[&Array], axis: i32) -> lisa_metal::error::Result<Array> {
    ops::concatenate(arrays, axis)
}

/// Narrow view `axis` in `[start, start+len)` (a view, via the inner array).
pub fn narrow(
    a: &Array,
    axis: usize,
    start: usize,
    len: usize,
) -> lisa_metal::error::Result<Array> {
    Ok(Array::new(a.t.narrow(axis, start, len)?))
}

/// Batched-safe matmul: flatten all leading dims to rows, contract, reshape
/// back (mlx-c matmul is only exact on 2-D here, same as quantized_matmul).
pub fn mm(a: &Array, w: &Array) -> lisa_metal::error::Result<Array> {
    let sh = a.shape();
    let rank = a.rank();
    let rows: i32 = sh[..rank - 1].iter().product::<i32>().max(1);
    let k = sh[rank - 1];
    let flat = a.reshape(&[rows, k])?;
    let out = flat.matmul(w)?;
    out.reshape(&[&sh[..rank - 1], &[w.shape()[1]]].concat())
}
