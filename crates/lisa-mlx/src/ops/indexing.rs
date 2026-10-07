//! `lisa_mlx::ops::indexing`-shaped free functions.
use super::*;

fn norm_axis(a: &Array, axis: i32) -> Result<usize> {
    if axis < 0 {
        let r = a.rank() as i32;
        let ax = axis + r;
        if ax < 0 {
            crate::bail!("indexing: axis {axis} out of range for rank {r}");
        }
        Ok(ax as usize)
    } else {
        Ok(axis as usize)
    }
}

/// Global argmax when `axis` is `None`, else along `axis`.
pub fn argmax(a: &Array, axis: Option<i32>) -> Result<Array> {
    match axis {
        None => {
            let flat = a.t.flatten_all()?;
            Ok(Array::new(flat.argmax(0)?))
        }
        Some(ax) => Ok(Array::new(a.t.argmax(norm_axis(a, ax)?)?)),
    }
}

pub fn argmin(a: &Array, axis: Option<i32>) -> Result<Array> {
    match axis {
        None => {
            let flat = a.t.flatten_all()?;
            Ok(Array::new(flat.argmin(0)?))
        }
        Some(ax) => Ok(Array::new(a.t.argmin(norm_axis(a, ax)?)?)),
    }
}

pub fn argmax_axis(a: &Array, axis: i32, keepdims: impl Into<Option<bool>>) -> Result<Array> {
    let keepdims = keepdims.into().unwrap_or(false);
    let ax = norm_axis(a, axis)?;
    let r = a.t.argmax(ax)?;
    Ok(Array::new(if keepdims { r.unsqueeze(ax)? } else { r }))
}

pub fn take_axis(a: &Array, indices: &Array, axis: i32) -> Result<Array> {
    let ax = norm_axis(a, axis)?;
    let idx = index_to_u32(&indices.t)?;
    let flat = idx.flatten_all()?.contiguous()?;
    let sel = a.t.index_select(&flat, ax)?;
    let mut shape: Vec<usize> = a.t.dims()[..ax].to_vec();
    shape.extend(idx.dims().iter().copied());
    shape.extend_from_slice(&a.t.dims()[ax + 1..]);
    Ok(Array::new(sel.reshape_dims(shape)?))
}

// ---- mlx-style slicing (`IndexOp` / `IndexMutOp` / `Ellipsis`) ----

/// Placeholder for `...` in an index tuple.
pub struct Ellipsis;

/// One element of an index tuple.
pub trait IndexElem {
    fn is_ellipsis(&self) -> bool {
        false
    }
    fn ndims(&self) -> usize {
        1
    }
    /// Narrow `t` at `dim`; the bool is "squeeze this axis" (scalar index).
    fn apply(&self, t: &crate::array::Array, dim: usize) -> Result<(crate::array::Array, bool)>;
    /// The equivalent range for assignment (scalar indices keep the axis).
    fn range(&self, dim_len: usize) -> std::ops::Range<usize>;
}

impl IndexElem for std::ops::RangeFull {
    fn apply(&self, t: &crate::array::Array, _dim: usize) -> Result<(crate::array::Array, bool)> {
        Ok((t.clone(), false))
    }
    fn range(&self, dim_len: usize) -> std::ops::Range<usize> {
        0..dim_len
    }
}

impl IndexElem for i32 {
    fn apply(&self, t: &crate::array::Array, dim: usize) -> Result<(crate::array::Array, bool)> {
        Ok((t.narrow(dim, *self as usize, 1)?, true))
    }
    fn range(&self, _dim_len: usize) -> std::ops::Range<usize> {
        *self as usize..*self as usize + 1
    }
}

fn norm_bound(v: i32, dim_len: usize) -> usize {
    if v < 0 {
        (dim_len as i32 + v).max(0) as usize
    } else {
        (v as usize).min(dim_len)
    }
}

impl IndexElem for std::ops::Range<i32> {
    fn apply(&self, t: &crate::array::Array, dim: usize) -> Result<(crate::array::Array, bool)> {
        let dl = t.dims()[dim];
        let a = norm_bound(self.start, dl);
        let b = norm_bound(self.end, dl);
        Ok((t.narrow(dim, a, b.saturating_sub(a))?, false))
    }
    fn range(&self, dim_len: usize) -> std::ops::Range<usize> {
        let a = norm_bound(self.start, dim_len);
        let b = norm_bound(self.end, dim_len);
        a..b
    }
}

impl IndexElem for std::ops::RangeFrom<i32> {
    fn apply(&self, t: &crate::array::Array, dim: usize) -> Result<(crate::array::Array, bool)> {
        let dl = t.dims()[dim];
        let a = norm_bound(self.start, dl);
        Ok((t.narrow(dim, a, dl - a)?, false))
    }
    fn range(&self, dim_len: usize) -> std::ops::Range<usize> {
        norm_bound(self.start, dim_len)..dim_len
    }
}

impl IndexElem for Ellipsis {
    fn is_ellipsis(&self) -> bool {
        true
    }
    fn ndims(&self) -> usize {
        0
    }
    fn apply(&self, _t: &crate::array::Array, _dim: usize) -> Result<(crate::array::Array, bool)> {
        unreachable!("ellipsis is expanded before apply")
    }
    fn range(&self, _dim_len: usize) -> std::ops::Range<usize> {
        unreachable!("ellipsis is expanded before range")
    }
}

fn index_seq(t: &crate::array::Array, elems: &[&dyn IndexElem]) -> Result<crate::array::Array> {
    let rank = t.rank();
    let used: usize = elems.iter().map(|e| e.ndims()).sum();
    let mut cur = t.clone();
    let mut squeezes: Vec<usize> = Vec::new();
    let mut dim = 0usize;
    for e in elems {
        if e.is_ellipsis() {
            dim += rank - used;
            continue;
        }
        let (v, sq) = e.apply(&cur, dim)?;
        cur = v;
        if sq {
            squeezes.push(dim);
        }
        dim += 1;
    }
    for d in squeezes.into_iter().rev() {
        cur = cur.squeeze(d)?;
    }
    Ok(cur)
}

fn index_ranges(t: &crate::array::Array, elems: &[&dyn IndexElem]) -> Vec<std::ops::Range<usize>> {
    let rank = t.rank();
    let used: usize = elems.iter().map(|e| e.ndims()).sum();
    let mut out: Vec<std::ops::Range<usize>> = Vec::with_capacity(rank);
    let mut dim = 0usize;
    for e in elems {
        if e.is_ellipsis() {
            for _ in 0..(rank - used) {
                out.push(0..t.dims()[dim]);
                dim += 1;
            }
            continue;
        }
        out.push(e.range(t.dims()[dim]));
        dim += 1;
    }
    out
}

/// Read-only slicing. `RangeFull`/`Range<i32>`/`RangeFrom<i32>` behave as in
/// mlx; a bare `i32` selects one index and drops the axis.
pub trait TryIndexOp<Idx> {
    fn try_index(&self, i: Idx) -> Result<Array>;
}

pub trait IndexOp<Idx>: TryIndexOp<Idx> {
    fn index(&self, i: Idx) -> Array {
        self.try_index(i).unwrap()
    }
}

impl<T, Idx> IndexOp<Idx> for T where T: TryIndexOp<Idx> {}

pub trait TryIndexMutOp<Idx, Val> {
    fn try_index_mut(&mut self, i: Idx, val: Val) -> Result<()>;
}

pub trait IndexMutOp<Idx, Val>: TryIndexMutOp<Idx, Val> {
    fn index_mut(&mut self, i: Idx, val: Val) {
        self.try_index_mut(i, val).unwrap()
    }
}

impl<T, Idx, Val> IndexMutOp<Idx, Val> for T where T: TryIndexMutOp<Idx, Val> {}

macro_rules! impl_index_tuple {
    ($($name:ident),+) => {
        #[allow(non_snake_case)]
        impl<$($name: IndexElem),+> TryIndexOp<($($name,)+)> for Array {
            fn try_index(&self, i: ($($name,)+)) -> Result<Array> {
                let ($($name,)+) = i;
                Ok(Array::new(index_seq(&self.t, &[$(&$name as &dyn IndexElem),+])?))
            }
        }
        #[allow(non_snake_case)]
        impl<$($name: IndexElem),+> TryIndexMutOp<($($name,)+), Array> for Array {
            fn try_index_mut(&mut self, i: ($($name,)+), val: Array) -> Result<()> {
                let ($($name,)+) = i;
                let elems: [&dyn IndexElem; _] = [$(&$name as &dyn IndexElem),+];
                let ranges = index_ranges(&self.t, &elems);
                self.t = self.t.slice_assign(&ranges, &val.t)?;
                Ok(())
            }
        }
    };
}

impl_index_tuple!(A);
impl_index_tuple!(A, B);
impl_index_tuple!(A, B, C);
impl_index_tuple!(A, B, C, D);

macro_rules! impl_index_single {
    ($t:ty) => {
        impl TryIndexOp<$t> for Array {
            fn try_index(&self, i: $t) -> Result<Array> {
                Ok(Array::new(index_seq(&self.t, &[&i as &dyn IndexElem])?))
            }
        }
    };
}

impl_index_single!(std::ops::RangeFull);
impl_index_single!(i32);
impl_index_single!(std::ops::Range<i32>);
impl_index_single!(std::ops::RangeFrom<i32>);
