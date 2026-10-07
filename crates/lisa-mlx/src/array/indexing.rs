use crate::error::Result;

use super::Array;

/// One element of an `Array::i(...)` index (`IndexOp` for a 2-tuple).

pub trait IdxElem {
    fn apply_idx(&self, a: &Array, dim: usize) -> Result<(Array, bool)>;
}

impl IdxElem for usize {
    fn apply_idx(&self, a: &Array, dim: usize) -> Result<(Array, bool)> {
        Ok((a.narrow(dim, *self, 1)?, true))
    }
}

impl IdxElem for i32 {
    fn apply_idx(&self, a: &Array, dim: usize) -> Result<(Array, bool)> {
        Ok((a.narrow(dim, *self as usize, 1)?, true))
    }
}

impl Array {
    /// `IndexOp::i` for a 3-tuple (integer dims narrow + squeeze).
    pub fn i3<A: IdxElem, B: IdxElem, C: IdxElem>(&self, a0: A, a1: B, a2: C) -> Result<Self> {
        let (x, s0) = a0.apply_idx(self, 0)?;
        let (x, s1) = a1.apply_idx(&x, 1)?;
        let (mut x, s2) = a2.apply_idx(&x, 2)?;
        if s2 {
            x = x.squeeze(2)?;
        }
        if s1 {
            x = x.squeeze(1)?;
        }
        if s0 {
            x = x.squeeze(0)?;
        }
        Ok(x)
    }

    /// `IndexOp::i` for a 2-tuple (integer dims narrow + squeeze).
    pub fn i<A: IdxElem, B: IdxElem>(&self, idx: (A, B)) -> Result<Self> {
        let (a, sq0) = idx.0.apply_idx(self, 0)?;
        let (b, sq1) = idx.1.apply_idx(&a, 1)?;
        let mut out = b;
        if sq1 {
            out = out.squeeze(1)?;
        }
        if sq0 {
            out = out.squeeze(0)?;
        }
        Ok(out)
    }
}
