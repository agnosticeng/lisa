//! Checked storage views and byte-preserving copies; no model computation.
use anyhow::Result;
#[cfg(test)]
use anyhow::ensure;
use lisa_metal::Array;
/// Checked contiguous view; offsets are bytes at the outer Array boundary.
pub(crate) fn view(x: &Array, first: usize, shape: &[i32]) -> Result<Array> {
    lisa_engine::storage::view(&crate::metal_context::device(), x, first, shape)
}

#[cfg(test)]
pub(crate) fn store_row(dst: &mut Array, row: usize, src: &Array) -> Result<()> {
    ensure!(
        dst.rank() == 2
            && src.shape() == [1, dst.dim(1)]
            && row < dst.dim(0) as usize
            && dst.dtype() == src.dtype(),
        "invalid Tensor cache row write"
    );
    let target = view(dst, row * dst.dim(1) as usize, src.shape())?;
    src.eval_enqueue()?;
    src.t.copy_strided_into(&target.t)?;
    *dst = Array::new(dst.t.clone());
    Ok(())
}
