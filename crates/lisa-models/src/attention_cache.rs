//! Contiguous attention-state layout and append law, independent of storage.
use crate::graph::{Builder, Graph, Value};
use lisa_backend::{Kernel, ScalarType, TensorOperation as T};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Append {
    pub batch: i32,
    pub heads: i32,
    pub width: i32,
    pub rows: i32,
    pub offset: usize,
    pub capacity: usize,
    pub previous_batch: Option<i32>,
    pub dtype: ScalarType,
}
impl Append {
    pub fn layout(&self) -> Result<(usize, usize, bool), String> {
        if self.batch <= 0
            || self.heads <= 0
            || self.width <= 0
            || self.rows <= 0
            || !matches!(self.dtype, ScalarType::BF16 | ScalarType::F32)
            || self.offset > self.capacity
            || (self.previous_batch.is_none() && self.offset != 0)
        {
            return Err("invalid attention cache layout".into());
        }
        let end = self
            .offset
            .checked_add(self.rows as usize)
            .ok_or("attention offset overflow")?;
        let grow = self.previous_batch.is_none()
            || self.capacity < end
            || self.previous_batch != Some(self.batch);
        // A live batch must be repacked by its owner before changing batch size.
        if self.offset != 0 && self.previous_batch != Some(self.batch) {
            return Err("live attention cache batch differs".into());
        }
        let mut capacity = if self.capacity == 0 {
            end.checked_add(256).ok_or("attention capacity overflow")?
        } else {
            self.capacity
        };
        while capacity < end {
            capacity = capacity
                .checked_mul(2)
                .ok_or("attention capacity overflow")?;
        }
        if capacity > i32::MAX as usize {
            return Err("attention capacity exceeds tensor dimensions".into());
        }
        Ok((end, capacity, grow))
    }
    /// Inputs: new K/V, followed by existing K/V when present. Outputs:
    /// storage K/V and live K/V. Only the unused tail of existing storage is
    /// writable; the engine installs offset/capacity after successful execution.
    pub fn graph(&self) -> Result<Graph, String> {
        let (end, capacity, grow) = self.layout()?;
        let old = self.previous_batch.is_some();
        let mut g = Builder::new(if old { 4 } else { 2 }, 0, 0);
        let (mut k, mut v) = if grow {
            let shape = vec![self.batch, self.heads, capacity as i32, self.width];
            let k = g.one(
                Kernel::Zeros {
                    shape: shape.clone(),
                    dtype: self.dtype,
                },
                vec![],
            );
            let v = g.one(
                Kernel::Zeros {
                    shape,
                    dtype: self.dtype,
                },
                vec![],
            );
            (k, v)
        } else {
            g.writable_input(2);
            g.writable_input(3);
            (Value::Input(2), Value::Input(3))
        };
        if grow && old {
            let range = T::Slice {
                axis: 2,
                start: 0,
                end: self.offset as i32,
            };
            let old_k = g.one(Kernel::Tensor(range.clone()), vec![Value::Input(2)]);
            let old_v = g.one(Kernel::Tensor(range), vec![Value::Input(3)]);
            let assign = T::AssignSlice {
                axis: 2,
                start: 0,
                end: self.offset as i32,
            };
            k = g.one(Kernel::Tensor(assign.clone()), vec![k, old_k]);
            v = g.one(Kernel::Tensor(assign), vec![v, old_v]);
        }
        k = g.one(
            Kernel::CopyRowsInto {
                offset: self.offset as i32,
            },
            vec![Value::Input(0), k],
        );
        v = g.one(
            Kernel::CopyRowsInto {
                offset: self.offset as i32,
            },
            vec![Value::Input(1), v],
        );
        let range = T::Slice {
            axis: 2,
            start: 0,
            end: end as i32,
        };
        let live_k = g.one(Kernel::Tensor(range.clone()), vec![k]);
        let live_v = g.one(Kernel::Tensor(range), vec![v]);
        Ok(g.finish(vec![k, v, live_k, live_v]))
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn capacity_and_mutation_are_explicit() {
        let mut p = Append {
            batch: 1,
            heads: 2,
            width: 256,
            rows: 7,
            offset: 0,
            capacity: 0,
            previous_batch: None,
            dtype: ScalarType::BF16,
        };
        assert_eq!(p.layout().unwrap(), (7, 263, true));
        p.graph().unwrap().validate().unwrap();
        p.previous_batch = Some(1);
        p.capacity = 263;
        p.offset = 7;
        let g = p.graph().unwrap();
        g.validate().unwrap();
        assert_eq!(g.mutable_inputs, vec![2, 3]);
        p.offset = 262;
        assert_eq!(p.layout().unwrap(), (269, 526, true));
        let g = p.graph().unwrap();
        g.validate().unwrap();
        assert!(g.mutable_inputs.is_empty());
        p.batch = 2;
        assert!(p.graph().is_err());
        p.capacity = usize::MAX;
        p.offset = usize::MAX;
        assert!(p.graph().is_err());
    }
}
