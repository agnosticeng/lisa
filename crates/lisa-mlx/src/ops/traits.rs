use super::*;

/// Scalar types usable in `Array ⊕ scalar` operators.
pub trait ScalarVal: Copy {
    fn tensor(
        self,
        rt: &std::sync::Arc<crate::runtime::MetalRuntime>,
    ) -> Result<crate::array::Array>;
}

macro_rules! scalar_val {
    ($t:ty) => {
        impl ScalarVal for $t {
            fn tensor(
                self,
                rt: &std::sync::Arc<crate::runtime::MetalRuntime>,
            ) -> Result<crate::array::Array> {
                crate::array::Array::scalar_of(rt, self as f32, Dtype::Float32)
            }
        }
    };
}

scalar_val!(f32);
scalar_val!(f64);
scalar_val!(i32);
scalar_val!(u32);

/// mlx's `promote_types` for the matmul operands: floating wins over integer,
/// and among floats the widest wins.
pub(crate) fn promote_dtype(a: Dtype, b: Dtype) -> Dtype {
    use Dtype::*;
    if a == b {
        return a;
    }
    let rank = |d: Dtype| match d {
        Float64 => 6,
        Float32 => 5,
        Bfloat16 => 4,
        Float16 => 3,
        Int64 => 2,
        Uint32 | Int32 => 1,
        _ => 0,
    };
    let is_float = |d: Dtype| matches!(d, Float64 | Float32 | Bfloat16 | Float16);
    let (fa, fb) = (is_float(a), is_float(b));
    match (fa, fb) {
        (true, true) | (false, false) => {
            if rank(a) >= rank(b) {
                a
            } else {
                b
            }
        }
        (true, false) => a,
        (false, true) => b,
    }
}

/// Element types usable with `ops::zeros`. there is no `WithDType for bool`,
/// so this is explicit rather than a blanket impl.
pub trait ZeroElem {
    const DT: Dtype;
    fn zeros(
        shape: &[usize],
        rt: &std::sync::Arc<crate::runtime::MetalRuntime>,
    ) -> Result<crate::array::Array> {
        crate::array::Array::zeros(rt, shape, Self::DT)
    }
}

macro_rules! zero_elem {
    ($t:ty, $dt:expr) => {
        impl ZeroElem for $t {
            const DT: Dtype = $dt;
        }
    };
}

zero_elem!(f32, Dtype::Float32);
zero_elem!(f64, Dtype::Float64);
zero_elem!(half::bf16, Dtype::Bfloat16);
zero_elem!(half::f16, Dtype::Float16);
zero_elem!(u8, Dtype::Uint8);
zero_elem!(u16, Dtype::Uint32);
zero_elem!(u32, Dtype::Uint32);
zero_elem!(i16, Dtype::Int16);
zero_elem!(i32, Dtype::Int32);
zero_elem!(i64, Dtype::Int64);
zero_elem!(bool, Dtype::Uint8);

/// Element types usable with `Array::from_slice` (bool maps to `u8`).
pub trait SliceElem: Copy {
    const DT: Dtype;
    fn tensor(
        data: &[Self],
        dims: Vec<usize>,
        rt: &std::sync::Arc<crate::runtime::MetalRuntime>,
    ) -> Result<crate::array::Array> {
        crate::array::Array::from_slice_dt(rt, data, &dims, Self::DT)
    }
}

macro_rules! slice_elem {
    ($t:ty, $dt:expr) => {
        impl SliceElem for $t {
            const DT: Dtype = $dt;
        }
    };
}

slice_elem!(f32, Dtype::Float32);
slice_elem!(f64, Dtype::Float64);
slice_elem!(half::bf16, Dtype::Bfloat16);
slice_elem!(half::f16, Dtype::Float16);
slice_elem!(u8, Dtype::Uint8);
slice_elem!(u32, Dtype::Uint32);
slice_elem!(i16, Dtype::Int16);
slice_elem!(i32, Dtype::Int32);
slice_elem!(i64, Dtype::Int64);

impl SliceElem for bool {
    const DT: Dtype = Dtype::Uint8;
    fn tensor(
        data: &[bool],
        dims: Vec<usize>,
        rt: &std::sync::Arc<crate::runtime::MetalRuntime>,
    ) -> Result<crate::array::Array> {
        let v: Vec<u8> = data.iter().map(|&b| b as u8).collect();
        crate::array::Array::from_slice_dt(rt, &v, &dims, Dtype::Uint8)
    }
}

/// Normalise a possibly-negative axis against `rank`.
pub(crate) fn norm_axis(rank: usize, axis: i32) -> usize {
    if axis < 0 {
        (rank as i32 + axis) as usize
    } else {
        axis as usize
    }
}

/// Promoted binary elementwise op (mlx promotes operand dtypes).
pub(crate) fn promoted_binary<F>(
    a: &crate::array::Array,
    b: &crate::array::Array,
    f: F,
) -> Result<crate::array::Array>
where
    F: FnOnce(&crate::array::Array, &crate::array::Array) -> Result<crate::array::Array>,
{
    // The native ops broadcast, so this is just the dtype promotion.
    let dt = promote_dtype(a.dtype(), b.dtype());
    let x = a.to_dtype(dt)?;
    let y = b.to_dtype(dt)?;
    f(&x, &y)
}

/// Dtype cast that works around incomplete Metal `to_dtype` table by
/// falling back to a CPU round-trip.
/// Index tensors for `index_select`/`gather`: Metal `to_dtype` only
/// implements a subset of pairs (I32 -> U32 is missing), so convert on host.
pub(crate) fn index_to_u32(t: &crate::array::Array) -> Result<crate::array::Array> {
    if t.dtype() == Dtype::Uint32 {
        return t.contiguous();
    }
    t.contiguous()?.cast(Dtype::Uint32)
}

pub(crate) fn scalar_op<T: ScalarVal>(a: &Array, s: T, op: &str) -> Array {
    let st = s.tensor(a.t.device()).unwrap();
    let dt = promote_dtype(a.t.dtype(), st.dtype());
    let x = a.t.to_dtype(dt).unwrap();
    let y = st.to_dtype(dt).unwrap();
    let r = match op {
        "add" => x.broadcast_add(&y),
        "sub" => x.broadcast_sub(&y),
        "mul" => x.broadcast_mul(&y),
        _ => x.broadcast_div(&y),
    };
    Array::new(r.unwrap())
}

pub(crate) fn scalar_op_rev<T: ScalarVal>(a: &Array, s: T, op: &str) -> Array {
    let st = s.tensor(a.t.device()).unwrap();
    let dt = promote_dtype(a.t.dtype(), st.dtype());
    let x = st.to_dtype(dt).unwrap();
    let y = a.t.to_dtype(dt).unwrap();
    let r = match op {
        "sub" => x.broadcast_sub(&y),
        _ => x.broadcast_div(&y),
    };
    Array::new(r.unwrap())
}
