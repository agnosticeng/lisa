//! Native implementation of the `lisa_mlx` surface used by the tree.
//!
//! This is built and A/B-validated family-by-family; `lib.rs` keeps
//! re-exporting the real bindings until the whole surface here is complete, at
//! which point this module is promoted to the crate root and the real
//! dependency is removed.
//!
//! Semantics notes:
//! - `eval` maps to a device synchronize: there is no "dispatch without
//!   wait", so unlike MLX this serializes the queue. Revisit before shipping.
//! - Fused/rounding-sensitive ops (`silu`, `sigmoid`, `logaddexp`, `log1p`)
//!   are *not* bit-exact via op chains (see `ab_probe`); they will be
//!   routed through `mlx_rt` from MLX's unary kernel instead.

pub type Result<T> = crate::error::Result<T>;

/// `lisa_mlx::error`-shaped module.
pub mod error {
    pub use crate::error::{Exception, Result};
}

/// mlx-like dtype enum — now the native one (`array::Dtype`), re-exported so
/// `lisa_mlx::Dtype` keeps its meaning.
pub use crate::array::Dtype;

/// Diagnostic: nanoseconds the host spent blocked in GPU waits.
pub fn runtime_wait_ns() -> u64 {
    crate::runtime::WAIT_NS.load(std::sync::atomic::Ordering::Relaxed)
}

/// Diagnostics: dispatch + encoder-open counts since process start.
pub fn runtime_dispatch_count() -> u64 {
    crate::runtime::DISPATCH_COUNT.load(std::sync::atomic::Ordering::Relaxed)
}

pub fn runtime_encoder_count() -> u64 {
    crate::runtime::ENCODER_COUNT.load(std::sync::atomic::Ordering::Relaxed)
}

pub fn runtime_alloc_count() -> u64 {
    crate::runtime::ALLOC_COUNT.load(std::sync::atomic::Ordering::Relaxed)
}

pub fn runtime_poolhit_count() -> u64 {
    crate::runtime::POOLHIT_COUNT.load(std::sync::atomic::Ordering::Relaxed)
}

pub fn label_counts() -> Vec<(String, u64)> {
    crate::runtime::LABEL_COUNTS.lock().unwrap().clone()
}

pub fn runtime_label_counts() -> Vec<(String, u64)> {
    crate::runtime::LABEL_COUNTS.lock().unwrap().clone()
}

pub fn runtime_zero_ns() -> u64 {
    crate::runtime::ZERO_NS.load(std::sync::atomic::Ordering::Relaxed)
}

/// A Metal stream. One command queue per device, so this just
/// carries the device (MLX's thread-local stream semantics land with `eval`).
#[derive(Clone)]
pub struct Stream {
    rt: std::sync::Arc<crate::runtime::MetalRuntime>,
}

impl Stream {
    pub fn gpu() -> Self {
        static RT: std::sync::OnceLock<std::sync::Arc<crate::runtime::MetalRuntime>> =
            std::sync::OnceLock::new();
        let rt = RT
            .get_or_init(|| {
                let per_buffer = std::env::var("LISA_METAL_COMPUTE_PER_BUFFER")
                    .ok()
                    .and_then(|v| v.parse::<usize>().ok())
                    .unwrap_or(16);
                std::sync::Arc::new(
                    crate::runtime::MetalRuntime::new(per_buffer).expect("metal runtime"),
                )
            })
            .clone();
        Self { rt }
    }

    pub fn thread_local_or_default() -> Self {
        Self::gpu()
    }

    /// The Metal runtime (the only device now).
    pub fn device(&self) -> &std::sync::Arc<crate::runtime::MetalRuntime> {
        &self.rt
    }

    /// Our own runtime (the flip's handle).
    pub fn runtime(&self) -> &std::sync::Arc<crate::runtime::MetalRuntime> {
        &self.rt
    }
}

impl Default for Stream {
    fn default() -> Self {
        Self::gpu()
    }
}

/// Scalar types usable in `Array ⊕ scalar` operators.
pub trait ScalarVal: Copy {
    fn tensor(self, rt: &std::sync::Arc<crate::runtime::MetalRuntime>) -> Result<crate::array::Array>;
}

macro_rules! scalar_val {
    ($t:ty) => {
        impl ScalarVal for $t {
            fn tensor(self, rt: &std::sync::Arc<crate::runtime::MetalRuntime>) -> Result<crate::array::Array> {
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
fn promote_dtype(a: Dtype, b: Dtype) -> Dtype {
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
            if rank(a) >= rank(b) { a } else { b }
        }
        (true, false) => a,
        (false, true) => b,
    }
}

/// Element types usable with `ops::zeros`. there is no `WithDType for bool`,
/// so this is explicit rather than a blanket impl.
pub trait ZeroElem {
    const DT: Dtype;
    fn zeros(shape: &[usize], rt: &std::sync::Arc<crate::runtime::MetalRuntime>) -> Result<crate::array::Array> {
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
fn norm_axis(rank: usize, axis: i32) -> usize {
    if axis < 0 {
        (rank as i32 + axis) as usize
    } else {
        axis as usize
    }
}

/// Promoted binary elementwise op (mlx promotes operand dtypes).
fn promoted_binary<F>(
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
fn index_to_u32(t: &crate::array::Array) -> Result<crate::array::Array> {
    if t.dtype() == Dtype::Uint32 {
        return t.contiguous();
    }
    t.contiguous()?.cast(Dtype::Uint32)
}

/// Diagnostic bisection switch: comma-separated op names in `LISA_UNARY_NATIVE`
/// use native implementation instead of the ported MLX kernel.
fn native_unary(op: &str) -> bool {
    // Cached: this is consulted per unary dispatch.
    static NATIVE: std::sync::OnceLock<Vec<String>> = std::sync::OnceLock::new();
    let list = NATIVE.get_or_init(|| {
        std::env::var("LISA_UNARY_NATIVE")
            .unwrap_or_default()
            .split(',')
            .map(|x| x.trim().to_string())
            .collect()
    });
    list.iter().any(|x| x == op)
}

/// A tensor. Mirrors `lisa_mlx::Array`: the shape is cached as `i32` so
/// `shape()`/`dim()` match mlx's signatures exactly (mlx works in `i32`).
#[derive(Clone)]
pub struct Array {
    pub t: crate::array::Array,
    shape: Vec<i32>,
    /// Lazily materialised host bytes for `as_slice`.
    host: std::sync::Arc<std::sync::OnceLock<Vec<u8>>>,
}

impl Array {
    pub fn new(t: crate::array::Array) -> Self {
        let shape = t.shape().iter().map(|&d| d as i32).collect();
        Self {
            t,
            shape,
            host: std::sync::Arc::new(std::sync::OnceLock::new()),
        }
    }

    // ---- constructors ----


    /// The native runtime for this thread.
    fn runtime_ref() -> std::sync::Arc<crate::runtime::MetalRuntime> {
        Stream::thread_local_or_default().runtime().clone()
    }

    /// lisa-mlx `Array::from_slice` (no stream argument).
    pub fn from_slice<T: SliceElem>(data: &[T], shape: &[i32]) -> Self {
        let dims: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
        Self::new(T::tensor(data, dims, &Self::runtime_ref()).expect("from_slice"))
    }

    /// lisa-mlx `Array::from_f32`: a scalar array.
    pub fn from_f32(val: f32) -> Self {
        Self::new(
            crate::array::Array::scalar_of(&Self::runtime_ref(), val, Dtype::Float32)
                .expect("from_f32"),
        )
    }

    /// A scalar array of `dtype`, built on the host (no GPU cast, no eval).
    /// The engine's `bf16_scalar`/norm scales hit this per layer per token; the
    /// `from_f32().as_dtype()` path went through `cast_host`, which evals the
    /// device (a full GPU sync per scalar).
    pub fn from_f32_as(val: f32, dtype: Dtype) -> Self {
        Self::new(
            crate::array::Array::scalar_of(&Self::runtime_ref(), val, dtype)
                .expect("from_f32_as"),
        )
    }

    /// lisa-mlx `Array::from_int`: an int32 scalar array.
    pub fn from_int(val: i32) -> Self {
        Self::new(
            crate::array::Array::scalar_raw(&Self::runtime_ref(), val as i64, Dtype::Int32)
                .expect("from_int"),
        )
    }

    /// lisa-mlx `Array::from_bool`: a uint8 scalar array.
    pub fn from_bool(val: bool) -> Self {
        Self::new(
            crate::array::Array::scalar_raw(
                &Self::runtime_ref(),
                if val { 1 } else { 0 },
                Dtype::Uint8,
            )
            .expect("from_bool"),
        )
    }

    // ---- metadata ----

    /// lisa-mlx `Array::shape`.
    pub fn shape(&self) -> &[i32] {
        &self.shape
    }

    pub fn dtype(&self) -> Dtype {
        self.t.dtype()
    }

    fn axis(&self, dim: i32) -> usize {
        let d = if dim.is_negative() {
            (self.ndim() as i32 + dim) as usize
        } else {
            dim as usize
        };
        d
    }

    /// lisa-mlx `Array::dim` (negative indices count from the end).
    pub fn dim(&self, dim: i32) -> i32 {
        self.shape[self.axis(dim)]
    }

    /// lisa-mlx `Array::ndim`.
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Alias kept for internal callers.
    pub fn rank(&self) -> usize {
        self.ndim()
    }

    pub fn size(&self) -> usize {
        self.t.elem_count()
    }

    /// mlx `Array::len` (element count).
    pub fn len(&self) -> usize {
        self.size()
    }

    pub fn is_empty(&self) -> bool {
        self.size() == 0
    }

    /// Diagnostic only: a pointer to a host-side f32 copy of the array, valid
    /// until the next `as_ptr()` call on this thread. The engine's dump paths
    /// read `len() * 4` bytes from it.
    pub fn as_ptr(&self) -> *const u8 {
        thread_local! {
            static BUF: std::cell::RefCell<Vec<u8>> = const { std::cell::RefCell::new(Vec::new()) };
        }
        let v: Vec<f32> = self.t.to_vec::<f32>().unwrap_or_default();
        let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
        BUF.with(|b| {
            *b.borrow_mut() = bytes;
            b.borrow().as_ptr()
        })
    }

    /// mlx `Array::from_raw_data`: copy `shape`-many `dtype` values from `ptr`.
    ///
    /// # Safety
    /// `ptr` must point to at least `prod(shape) * dtype.size` readable bytes.
    pub unsafe fn from_raw_data(
        ptr: *const std::ffi::c_void,
        shape: &[i32],
        dtype: Dtype,
    ) -> Array {
        let dims: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
        let n: usize = dims.iter().product();
        let rt = Stream::thread_local_or_default().runtime().clone();
        let a = unsafe {
            match dtype {
                Dtype::Bfloat16 => crate::array::Array::from_slice_dt(
                    &rt, std::slice::from_raw_parts(ptr as *const half::bf16, n), &dims, dtype),
                Dtype::Float16 => crate::array::Array::from_slice_dt(
                    &rt, std::slice::from_raw_parts(ptr as *const half::f16, n), &dims, dtype),
                Dtype::Float32 => crate::array::Array::from_slice_dt(
                    &rt, std::slice::from_raw_parts(ptr as *const f32, n), &dims, dtype),
                Dtype::Float64 => crate::array::Array::from_slice_dt(
                    &rt, std::slice::from_raw_parts(ptr as *const f64, n), &dims, dtype),
                Dtype::Uint8 => crate::array::Array::from_slice_dt(
                    &rt, std::slice::from_raw_parts(ptr as *const u8, n), &dims, dtype),
                Dtype::Uint32 => crate::array::Array::from_slice_dt(
                    &rt, std::slice::from_raw_parts(ptr as *const u32, n), &dims, dtype),
                Dtype::Int16 => crate::array::Array::from_slice_dt(
                    &rt, std::slice::from_raw_parts(ptr as *const i16, n), &dims, dtype),
                Dtype::Int32 => crate::array::Array::from_slice_dt(
                    &rt, std::slice::from_raw_parts(ptr as *const i32, n), &dims, dtype),
                Dtype::Int64 => crate::array::Array::from_slice_dt(
                    &rt, std::slice::from_raw_parts(ptr as *const i64, n), &dims, dtype),
                other => panic!("from_raw_data: unsupported dtype {other:?}"),
            }
        }
        .expect("from_raw_data");
        Array::new(a)
    }

    /// mlx `Array::sum_axis` (last-axis sum via the ported reduce kernel).
    pub fn sum_axis(&self, axis: i32, keep_dims: impl Into<Option<bool>>) -> Result<Self> {
        let keep = keep_dims.into().unwrap_or(false);
        let r = crate::mlx_rt::reduce_axis(self.t.device(), &self.t, axis, "sum")?;
        Ok(Self::new(if keep {
            r.unsqueeze(self.axis(axis))?
        } else {
            r
        }))
    }

    /// mlx `Array::mean_axis`.
    pub fn mean_axis(&self, axis: i32, keep_dims: impl Into<Option<bool>>) -> Result<Self> {
        let keep = keep_dims.into().unwrap_or(false);
        let r = crate::mlx_rt::reduce_axis(self.t.device(), &self.t, axis, "mean")?;
        Ok(Self::new(if keep {
            r.unsqueeze(self.axis(axis))?
        } else {
            r
        }))
    }

    /// mlx `Array::as_slice`: a slice of the array's values, materialised to
    /// host memory on first use and cached in the handle.
    pub fn as_slice<T: Copy + 'static>(&self) -> &[T] {
        let bytes = self.host.get_or_init(|| {
            let v: Vec<T> = self.t.to_vec::<T>().unwrap_or_default();
            let mut b = vec![0u8; v.len() * std::mem::size_of::<T>()];
            if !b.is_empty() {
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        v.as_ptr() as *const u8,
                        b.as_mut_ptr(),
                        b.len(),
                    );
                }
            }
            b
        });
        let n = bytes.len() / std::mem::size_of::<T>();
        unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const T, n) }
    }

    /// mlx `Array::take_along_axis`.
    pub fn take_along_axis(&self, indices: &Self, axis: i32) -> Result<Self> {
        let ax = self.axis(axis);
        let idx = index_to_u32(&indices.t)?;
        Ok(Self::new(self.t.gather(&idx, ax)?))
    }

    /// mlx `Array::put_along_axis`.
    pub fn put_along_axis(&self, indices: &Self, values: &Self, axis: i32) -> Result<Self> {
        crate::mlx_rt::put_along_axis(
            self.t.device(),
            &self.t,
            &indices.t,
            &values.t,
            axis,
        )
        .map(Self::new)
    }

    /// mlx `Array::split_equal`.
    pub fn split_equal(&self, num_splits: usize, axis: i32) -> Result<Vec<Self>> {
        let parts = self
            .t
            .chunk(num_splits, self.axis(axis))?;
        Ok(parts.into_iter().map(Self::new).collect())
    }

    /// mlx `Array::logical_or` (0/1 `u8` max).
    pub fn logical_or(&self, rhs: impl AsRef<Array>) -> Result<Self> {
        Ok(Self::new(self.t.maximum(&rhs.as_ref().t)?))
    }

    // ---- transforms ----

    /// mlx `Array::reshape`: one `-1` is inferred and a `0` copies the input
    /// dimension at that position.
    pub fn reshape(&self, shape: &[i32]) -> Result<Self> {
        let total = self.t.elem_count();
        let in_dims = self.t.dims();
        let mut dims: Vec<usize> = Vec::with_capacity(shape.len());
        let mut hole: Option<usize> = None;
        let mut prod = 1usize;
        for (i, &d) in shape.iter().enumerate() {
            if d == -1 {
                hole = Some(i);
                dims.push(1);
            } else if d == 0 {
                let u = in_dims.get(i).copied().unwrap_or(1);
                prod *= u;
                dims.push(u);
            } else {
                let u = d as usize;
                prod *= u;
                dims.push(u);
            }
        }
        if let Some(i) = hole {
            dims[i] = total / prod.max(1);
        }
        Ok(Self::new(self.t.reshape_dims(dims)?))
    }

    pub fn expand_dims(&self, axis: i32) -> Result<Self> {
        let rank = self.t.rank();
        let ax = if axis < 0 {
            (rank as i32 + 1 + axis) as usize
        } else {
            axis as usize
        };
        Ok(Self::new(self.t.unsqueeze(ax)?))
    }

    pub fn as_dtype(&self, dtype: Dtype) -> Result<Self> {
        Ok(Self::new(self.t.to_dtype(dtype)?))
    }

    /// Force pending GPU work. There is no non-blocking eval, so this waits.
    pub fn eval(&self) -> Result<()> {
        self.t.device().synchronize()
    }

    pub fn item<T: Copy + 'static>(&self) -> T {
        self.t.item::<T>()
    }

    pub fn to_vec1<T: Copy + 'static>(&self) -> Result<Vec<T>> {
        self.t.flatten_all()?.contiguous()?.to_vec1::<T>()
    }

    // ---- elementwise ----

    pub fn add(&self, rhs: impl AsRef<Array>) -> Result<Self> {
        Ok(Self::new(promoted_binary(&self.t, &rhs.as_ref().t, |a, b| a.broadcast_add(b))?))
    }

    pub fn subtract(&self, rhs: impl AsRef<Array>) -> Result<Self> {
        Ok(Self::new(promoted_binary(&self.t, &rhs.as_ref().t, |a, b| a.broadcast_sub(b))?))
    }

    pub fn multiply(&self, rhs: impl AsRef<Array>) -> Result<Self> {
        Ok(Self::new(promoted_binary(&self.t, &rhs.as_ref().t, |a, b| a.broadcast_mul(b))?))
    }

    pub fn divide(&self, rhs: impl AsRef<Array>) -> Result<Self> {
        Ok(Self::new(promoted_binary(&self.t, &rhs.as_ref().t, |a, b| a.broadcast_div(b))?))
    }

    pub fn maximum(&self, rhs: impl AsRef<Array>) -> Result<Self> {
        Ok(Self::new(promoted_binary(&self.t, &rhs.as_ref().t, |a, b| a.maximum(b))?))
    }

    pub fn minimum(&self, rhs: impl AsRef<Array>) -> Result<Self> {
        Ok(Self::new(promoted_binary(&self.t, &rhs.as_ref().t, |a, b| a.minimum(b))?))
    }

    pub fn exp(&self) -> Result<Self> {
        // LISA_EXP_ROUTE: "f32" | "bf16" | "all" | "none" (default none) to
        // bisect which dtype/path the ported kernel breaks on.
        static ROUTE: std::sync::OnceLock<String> = std::sync::OnceLock::new();
        let route = ROUTE.get_or_init(|| std::env::var("LISA_EXP_ROUTE").unwrap_or_else(|_| "none".into()));
        let is_f32 = self.t.dtype() == Dtype::Float32;
        let use_kernel = native_unary("exp") == false
            || match route.as_str() {
                "all" => true,
                "f32" => is_f32,
                "bf16" => !is_f32,
                _ => false,
            };
        if use_kernel {
            let t = crate::mlx_rt::exp(self.t.device(), &self.t)?;
            if std::env::var_os("LISA_SYNC_EXP").is_some() {
                t.device().synchronize()?;
            }
            return Ok(Self::new(t));
        }
        Ok(Self::new(self.t.exp()?))
    }

    pub fn log(&self) -> Result<Self> {
        if native_unary("log") {
            return Ok(Self::new(self.t.log()?));
        }
        Ok(Self::new(crate::mlx_rt::log(self.t.device(), &self.t)?))
    }

    pub fn sin(&self) -> Result<Self> {
        Ok(Self::new(self.t.sin()?))
    }

    pub fn cos(&self) -> Result<Self> {
        Ok(Self::new(self.t.cos()?))
    }

    pub fn sqrt(&self) -> Result<Self> {
        if native_unary("sqrt") {
            return Ok(Self::new(self.t.sqrt()?));
        }
        Ok(Self::new(crate::mlx_rt::sqrt(self.t.device(), &self.t)?))
    }

    pub fn rsqrt(&self) -> Result<Self> {
        if native_unary("rsqrt") {
            return Ok(Self::new(self.t.sqrt()?.recip()?));
        }
        Ok(Self::new(crate::mlx_rt::rsqrt(self.t.device(), &self.t)?))
    }

    pub fn abs(&self) -> Result<Self> {
        if native_unary("abs") {
            return Ok(Self::new(self.t.abs()?));
        }
        Ok(Self::new(crate::mlx_rt::abs(self.t.device(), &self.t)?))
    }

    pub fn sign(&self) -> Result<Self> {
        if native_unary("sign") {
            return Ok(Self::new(self.t.sign()?));
        }
        Ok(Self::new(crate::mlx_rt::sign(self.t.device(), &self.t)?))
    }

    pub fn neg(&self) -> Result<Self> {
        Ok(Self::new(self.t.neg()?))
    }

    pub fn sigmoid(&self) -> Result<Self> {
        if native_unary("sigmoid") {
            return Ok(Self::new(self.t.neg()?.exp()?.affine(1.0, 1.0)?.recip()?));
        }
        Ok(Self::new(crate::mlx_rt::sigmoid(self.t.device(), &self.t)?))
    }

    pub fn silu(&self) -> Result<Self> {
        if native_unary("silu") {
            return Ok(Self::new(self.t.silu()?));
        }
        Ok(Self::new(crate::mlx_rt::silu(self.t.device(), &self.t)?))
    }

    pub fn log1p(&self) -> Result<Self> {
        if native_unary("log1p") {
            return Ok(Self::new(self.t.affine(1.0, 1.0)?.log()?));
        }
        Ok(Self::new(crate::mlx_rt::log1p(self.t.device(), &self.t)?))
    }

    pub fn logaddexp(&self, rhs: &Self) -> Result<Self> {
        let a = self.t.maximum(&rhs.t)?;
        let b = self.t.minimum(&rhs.t)?;
        let d = b.sub(&a)?.exp()?.affine(1.0, 1.0)?.log()?;
        Ok(Self::new(a.add(&d)?))
    }

    pub fn is_nan(&self) -> Result<Self> {
        Ok(Self::new(self.t.ne(&self.t)?))
    }

    pub fn floor_divide(&self, rhs: impl AsRef<Array>) -> Result<Self> {
        // floor() is float-only; do the divide/floor in f32 and cast
        // back to the promoted dtype (matches mlx's numpy-style floor_divide).
        let r = rhs.as_ref();
        let dt = promote_dtype(self.t.dtype(), r.t.dtype());
        let a = self.t.to_dtype(Dtype::Float32)?;
        let b = r.t.to_dtype(Dtype::Float32)?;
        let q = a.broadcast_div(&b)?.floor()?;
        Ok(Self::new(q.to_dtype(dt)?))
    }

    /// `select(cond, a, b)` — MLX's `where`.
    pub fn where_(cond: &Self, a: &Self, b: &Self) -> Result<Self> {
        Ok(Self::new(cond.t.where_cond(&a.t, &b.t)?))
    }

    /// Logical NOT for a 0/1 (bool) tensor.
    pub fn logical_not(&self) -> Result<Self> {
        Ok(Self::new(self.t.affine(-1.0, 1.0)?))
    }

    // ---- reductions ----

    pub fn sum(&self, axis: Option<&[i32]>) -> Result<Self> {
        match axis {
            None => Ok(Self::new(self.t.sum_all()?.to_dtype(self.t.dtype())?)),
            Some(&[ax]) => Ok(Self::new(self.t.sum(ax as usize)?)),
            Some(_) => crate::bail!("multi-axis sum not implemented"),
        }
    }

    pub fn mean(&self, axis: Option<&[i32]>) -> Result<Self> {
        match axis {
            None => Ok(Self::new(self.t.mean_all()?.to_dtype(self.t.dtype())?)),
            Some(&[ax]) => Ok(Self::new(self.t.mean(ax as usize)?)),
            Some(_) => crate::bail!("multi-axis mean not implemented"),
        }
    }

    pub fn max(&self, axis: Option<&[i32]>) -> Result<Self> {
        match axis {
            None => Ok(Self::new(self.t.max_all()?)),
            Some(&[ax]) => Ok(Self::new(self.t.max(ax as usize)?)),
            Some(_) => crate::bail!("multi-axis max not implemented"),
        }
    }

    pub fn min(&self, axis: Option<&[i32]>) -> Result<Self> {
        match axis {
            None => Ok(Self::new(self.t.min_all()?)),
            Some(&[ax]) => Ok(Self::new(self.t.min(ax as usize)?)),
            Some(_) => crate::bail!("multi-axis min not implemented"),
        }
    }

    // ---- shape ops ----

    pub fn concatenate(arrays: &[&Self], axis: i32) -> Result<Self> {
        let rank = arrays.first().map(|a| a.t.rank()).unwrap_or(1);
        let inner: Vec<crate::array::Array> = arrays.iter().map(|a| a.t.clone()).collect();
        Ok(Self::new(crate::array::Array::cat(&inner, norm_axis(rank, axis))?))
    }

    pub fn stack(arrays: &[&Self], axis: i32) -> Result<Self> {
        let rank = arrays.first().map(|a| a.t.rank()).unwrap_or(0) + 1;
        let inner: Vec<crate::array::Array> = arrays.iter().map(|a| a.t.clone()).collect();
        Ok(Self::new(crate::array::Array::stack(&inner, norm_axis(rank, axis))?))
    }

    pub fn broadcast_to(&self, shape: &[i32]) -> Result<Self> {
        let dims: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
        Ok(Self::new(self.t.broadcast_as(dims)?))
    }

    /// np.repeat (each element `n` times) along `axis` — MLX's `repeat_axis`
    /// semantics, unlike `repeat` which tiles.
    pub fn repeat_axis(&self, n: usize, axis: i32) -> Result<Self> {
        let rank = self.t.rank();
        let ax = if axis < 0 {
            (rank as i32 + axis) as usize
        } else {
            axis as usize
        };
        let mut dims = self.t.dims().to_vec();
        let d = dims[ax];
        dims.insert(ax + 1, n);
        let expanded = self.t.unsqueeze(ax + 1)?.broadcast_as(dims.clone())?;
        let mut out_dims = dims;
        out_dims[ax] = d * n;
        out_dims.remove(ax + 1);
        Ok(Self::new(expanded.reshape_dims(out_dims)?))
    }

    /// `tile` (np.tile semantics). Same-rank `reps` only.
    pub fn tile(&self, reps: &[usize]) -> Result<Self> {
        let dims = self.t.dims().to_vec();
        if reps.len() != dims.len() {
            crate::bail!("tile: reps rank must match input rank");
        }
        let mut exp = Vec::with_capacity(dims.len() * 2);
        for (i, &d) in dims.iter().enumerate() {
            exp.push(reps[i]);
            exp.push(d);
        }
        let mut t = self.t.clone();
        for i in 0..dims.len() {
            t = t.unsqueeze(2 * i)?;
        }
        let t = t.broadcast_as(exp)?;
        let out_dims: Vec<usize> = dims.iter().zip(reps).map(|(d, r)| d * r).collect();
        Ok(Self::new(t.reshape_dims(out_dims)?))
    }

    pub fn take_axis(&self, indices: &Self, axis: i32) -> Result<Self> {
        let ax = self.axis(axis);
        let idx = index_to_u32(&indices.t)?;
        // mlx take_axis replaces the axis with the *whole* index shape, so
        // flatten for index_select then reshape back.
        let flat = idx.flatten_all()?.contiguous()?;
        let sel = self.t.index_select(&flat, ax)?;
        let mut shape: Vec<usize> = self.t.dims()[..ax].to_vec();
        shape.extend(idx.dims().iter().copied());
        shape.extend_from_slice(&self.t.dims()[ax + 1..]);
        Ok(Self::new(sel.reshape_dims(shape)?))
    }

    // ---- linalg / softmax ----

    /// mlx `matmul` promotes the operand dtypes (`result_type`); the API requires
    /// them to match.
    pub fn matmul(&self, rhs: impl AsRef<Array>) -> Result<Self> {
        let r = rhs.as_ref();
        let dt = promote_dtype(self.t.dtype(), r.t.dtype());
        let a = self.t.to_dtype(dt)?;
        let b = r.t.to_dtype(dt)?;
        Ok(Self::new(a.matmul(&b)?))
    }

    pub fn softmax_axis(&self, axis: i32) -> Result<Self> {
        // Use the ported MLX softmax kernel (bit-exact vs mlx-c); the
        // native softmax differs.
        let ax = norm_axis(self.t.rank(), axis);
        if ax == self.t.rank() - 1 {
            return Ok(Self::new(crate::mlx_rt::softmax_last_axis(
                self.t.device(),
                &self.t,
                true,
            )?));
        }
        let t = self.t.transpose(ax, self.t.rank() - 1)?.contiguous()?;
        let r = crate::mlx_rt::softmax_last_axis(self.t.device(), &t, true)?;
        Ok(Self::new(r.transpose(ax, self.t.rank() - 1)?.contiguous()?))
    }

    pub fn transpose_axes(&self, axes: &[i32]) -> Result<Self> {
        let perm: Vec<usize> = axes.iter().map(|&a| a as usize).collect();
        Ok(Self::new(self.t.permute(&perm)?))
    }

    /// mlx `Array::item_cast`: read a single element (panics on failure).
    pub fn item_cast<T: Copy + 'static>(&self) -> T {
        self.t.item::<T>()
    }

    pub fn contiguous(&self) -> Result<Self> {
        Ok(Self::new(self.t.contiguous()?))
    }

    /// Detached view (shares storage, drops the autograd/graph reference). Used
    /// to stop a small derived tensor from keeping a large parent buffer alive.
    pub fn detach(&self) -> Self {
        self.clone()
    }

    pub fn ge(&self, rhs: impl AsRef<Array>) -> Result<Self> {
        Ok(Self::new(self.t.ge(&rhs.as_ref().t)?))
    }
    pub fn le(&self, rhs: impl AsRef<Array>) -> Result<Self> {
        Ok(Self::new(self.t.le(&rhs.as_ref().t)?))
    }

    pub fn lt(&self, rhs: impl AsRef<Array>) -> Result<Self> {
        Ok(Self::new(self.t.lt(&rhs.as_ref().t)?))
    }

    pub fn gt(&self, rhs: impl AsRef<Array>) -> Result<Self> {
        Ok(Self::new(self.t.gt(&rhs.as_ref().t)?))
    }

    /// mlx `logical_and` on bool arrays; there is no such op, and the inputs
    /// are 0/1 `u8`, so multiplication is the conjunction.
    pub fn logical_and(&self, rhs: impl AsRef<Array>) -> Result<Self> {
        Ok(Self::new(self.t.mul(&rhs.as_ref().t)?))
    }
}

impl AsRef<Array> for Array {
    fn as_ref(&self) -> &Array {
        self
    }
}

/// mlx-style operators. Broadcasts like mlx; panics on error like mlx's.
macro_rules! impl_array_binop {
    ($tr:ident, $op:ident, $cm:ident) => {
        impl std::ops::$tr<&Array> for &Array {
            type Output = Array;
            fn $op(self, rhs: &Array) -> Array {
                let dt = promote_dtype(self.t.dtype(), rhs.t.dtype());
                let a = self.t.to_dtype(dt).unwrap();
                let b = rhs.t.to_dtype(dt).unwrap();
                Array::new(a.$cm(&b).unwrap())
            }
        }
        impl std::ops::$tr<Array> for &Array {
            type Output = Array;
            fn $op(self, rhs: Array) -> Array {
                let dt = promote_dtype(self.t.dtype(), rhs.t.dtype());
                let a = self.t.to_dtype(dt).unwrap();
                let b = rhs.t.to_dtype(dt).unwrap();
                Array::new(a.$cm(&b).unwrap())
            }
        }
        impl std::ops::$tr<&Array> for Array {
            type Output = Array;
            fn $op(self, rhs: &Array) -> Array {
                let dt = promote_dtype(self.t.dtype(), rhs.t.dtype());
                let a = self.t.to_dtype(dt).unwrap();
                let b = rhs.t.to_dtype(dt).unwrap();
                Array::new(a.$cm(&b).unwrap())
            }
        }
        impl std::ops::$tr<Array> for Array {
            type Output = Array;
            fn $op(self, rhs: Array) -> Array {
                let dt = promote_dtype(self.t.dtype(), rhs.t.dtype());
                let a = self.t.to_dtype(dt).unwrap();
                let b = rhs.t.to_dtype(dt).unwrap();
                Array::new(a.$cm(&b).unwrap())
            }
        }
    };
}

impl_array_binop!(Add, add, broadcast_add);
impl_array_binop!(Sub, sub, broadcast_sub);
impl_array_binop!(Mul, mul, broadcast_mul);
impl_array_binop!(Div, div, broadcast_div);

/// Array ⊕ scalar (both orders) for the scalar types the tree uses.
macro_rules! impl_array_scalar {
    ($t:ty) => {
        impl std::ops::Add<$t> for &Array {
            type Output = Array;
            fn add(self, s: $t) -> Array {
                scalar_op(self, s, "add")
            }
        }
        impl std::ops::Add<&Array> for $t {
            type Output = Array;
            fn add(self, a: &Array) -> Array {
                scalar_op(a, self, "add")
            }
        }
        impl std::ops::Sub<$t> for &Array {
            type Output = Array;
            fn sub(self, s: $t) -> Array {
                scalar_op(self, s, "sub")
            }
        }
        impl std::ops::Sub<&Array> for $t {
            type Output = Array;
            fn sub(self, a: &Array) -> Array {
                scalar_op_rev(a, self, "sub")
            }
        }
        impl std::ops::Mul<$t> for &Array {
            type Output = Array;
            fn mul(self, s: $t) -> Array {
                scalar_op(self, s, "mul")
            }
        }
        impl std::ops::Mul<&Array> for $t {
            type Output = Array;
            fn mul(self, a: &Array) -> Array {
                scalar_op(a, self, "mul")
            }
        }
        impl std::ops::Div<$t> for &Array {
            type Output = Array;
            fn div(self, s: $t) -> Array {
                scalar_op(self, s, "div")
            }
        }
        impl std::ops::Div<&Array> for $t {
            type Output = Array;
            fn div(self, a: &Array) -> Array {
                scalar_op_rev(a, self, "div")
            }
        }
    };
}

fn scalar_op<T: ScalarVal>(a: &Array, s: T, op: &str) -> Array {
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

fn scalar_op_rev<T: ScalarVal>(a: &Array, s: T, op: &str) -> Array {
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

impl_array_scalar!(f32);
impl_array_scalar!(f64);
impl_array_scalar!(i32);
impl_array_scalar!(u32);

/// Owned-`Array` variants of the scalar operators.
macro_rules! impl_array_scalar_owned {
    ($t:ty) => {
        impl std::ops::Add<$t> for Array {
            type Output = Array;
            fn add(self, s: $t) -> Array {
                scalar_op(&self, s, "add")
            }
        }
        impl std::ops::Sub<$t> for Array {
            type Output = Array;
            fn sub(self, s: $t) -> Array {
                scalar_op(&self, s, "sub")
            }
        }
        impl std::ops::Mul<$t> for Array {
            type Output = Array;
            fn mul(self, s: $t) -> Array {
                scalar_op(&self, s, "mul")
            }
        }
        impl std::ops::Div<$t> for Array {
            type Output = Array;
            fn div(self, s: $t) -> Array {
                scalar_op(&self, s, "div")
            }
        }
    };
}

impl_array_scalar_owned!(f32);
impl_array_scalar_owned!(f64);
impl_array_scalar_owned!(i32);
impl_array_scalar_owned!(u32);

impl std::ops::Neg for &Array {
    type Output = Array;
    fn neg(self) -> Array {
        Array::new(self.t.neg().unwrap())
    }
}

impl std::ops::Neg for Array {
    type Output = Array;
    fn neg(self) -> Array {
        Array::new(self.t.neg().unwrap())
    }
}

/// `lisa_mlx::ops`-shaped free functions.
pub mod ops {
    use super::*;

    /// mlx exposes indexing under `ops::indexing`.
    pub use crate::shim_api::indexing;

    pub fn exp(a: impl AsRef<Array>) -> Result<Array> {
        a.as_ref().exp()
    }
    pub fn log(a: impl AsRef<Array>) -> Result<Array> {
        a.as_ref().log()
    }
    pub fn sqrt(a: impl AsRef<Array>) -> Result<Array> {
        a.as_ref().sqrt()
    }
    pub fn rsqrt(a: impl AsRef<Array>) -> Result<Array> {
        a.as_ref().rsqrt()
    }
    pub fn sin(a: impl AsRef<Array>) -> Result<Array> {
        a.as_ref().sin()
    }
    pub fn cos(a: impl AsRef<Array>) -> Result<Array> {
        a.as_ref().cos()
    }
    pub fn log1p(a: impl AsRef<Array>) -> Result<Array> {
        a.as_ref().log1p()
    }
    pub fn argsort(a: &Array) -> Result<Array> {
        Ok(Array::new(crate::mlx_rt::argsort_last(a.t.device(), &a.t)?))
    }
    pub fn tile(a: &Array, reps: &[i32]) -> Result<Array> {
        let r: Vec<usize> = reps.iter().map(|&v| v as usize).collect();
        a.tile(&r)
    }

    /// mlx `ops::dequantize` (affine): `w = q * scale + bias` per group.
    pub fn dequantize(
        w: &Array,
        scales: &Array,
        biases: &Array,
        group_size: i32,
        bits: i32,
    ) -> Result<Array> {
        // GPU path (MLX `affine_dequantize`): the host implementation is a full
        // GPU->CPU->GPU round trip and the embedding forward calls this per
        // token.
        return Ok(Array::new(crate::mlx_rt::affine_dequantize(
            w.t.device(),
            &w.t,
            &scales.t,
            &biases.t,
            group_size,
            bits,
        )?));
        #[allow(unreachable_code)]
        let gs = group_size as usize;
        let bits = bits as usize;
        let dims: Vec<usize> = w.shape().iter().map(|&d| d as usize).collect();
        let kq = *dims.last().unwrap();
        let per_word = 32 / bits;
        let k = kq * per_word;
        let rows = w.size() / kq;
        let groups = k / gs;
        let q = w.t.cast(Dtype::Uint32)?.to_vec::<u32>()?;
        let sc: Vec<f32> = scales.t.cast(Dtype::Float32)?.to_vec::<f32>()?;
        let bi: Vec<f32> = biases.t.cast(Dtype::Float32)?.to_vec::<f32>()?;
        let mut out = vec![0f32; rows * k];
        let mask = (1u32 << bits) - 1;
        for r in 0..rows {
            for c in 0..k {
                let word = q[r * kq + c / per_word];
                let qv = (word >> ((c % per_word) * bits)) & mask;
                let g = c / gs;
                out[r * k + c] = qv as f32 * sc[r * groups + g] + bi[r * groups + g];
            }
        }
        let mut shape = dims.clone();
        shape.pop();
        shape.push(k);
        let rt = Stream::thread_local_or_default().runtime().clone();
        let t = crate::array::Array::from_slice_dt(&rt, &out, &shape, Dtype::Float32)?;
        // MLX returns the scales' dtype (the original weight type), not the packed type.
        Ok(Array::new(t.to_dtype(scales.t.dtype())?))
    }

    /// mlx `ops::conv1d`. Only the depthwise (`groups == C_in == C_out`,
    /// one input channel per group) case the GDN fallback uses is implemented.
    pub fn conv1d(
        input: &Array,
        weight: &Array,
        stride: Option<i32>,
        padding: Option<(i32, i32)>,
        dilation: Option<i32>,
        groups: Option<i32>,
    ) -> Result<Array> {
        let ish = input.shape().to_vec();
        let wsh = weight.shape().to_vec();
        let (n, l, cin) = (ish[0] as usize, ish[1] as usize, ish[2] as usize);
        let (cout, k) = (wsh[0] as usize, wsh[1] as usize);
        let g = groups.unwrap_or(1) as usize;
        let s = stride.unwrap_or(1) as usize;
        let dil = dilation.unwrap_or(1) as usize;
        let (pl, pu) = padding.unwrap_or((0, 0));
        if g != cin || cout != cin || wsh[2] != 1 {
            crate::bail!(
                "conv1d: only depthwise (groups == C_in == C_out) is implemented"
            );
        }
        let lp = l + pl as usize + pu as usize;
        let lout = if lp >= (k - 1) * dil + 1 {
            (lp - ((k - 1) * dil + 1)) / s + 1
        } else {
            0
        };
        let rt = Stream::thread_local_or_default().runtime().clone();
        let mut acc = crate::array::Array::zeros(&rt, &[n as usize, lout as usize, cin as usize], Dtype::Float32)?;
        let xf = input.t.to_dtype(Dtype::Float32)?;
        let wf = weight.t.to_dtype(Dtype::Float32)?;
        for t in 0..k {
            let src_start = pl as usize + t * dil;
            let last = src_start + (lout.saturating_sub(1)) * s + 1;
            if lout == 0 || last > lp {
                continue;
            }
            let slice = xf.narrow(1, src_start, lout)?.contiguous()?;
            let wc = wf.narrow(1, t, 1)?.reshape(&[1, 1, cin as usize])?;
            acc = acc.broadcast_add(&slice.broadcast_mul(&wc)?)?;
        }
        Ok(Array::new(acc.to_dtype(input.t.dtype())?))
    }
    pub fn abs(a: impl AsRef<Array>) -> Result<Array> {
        a.as_ref().abs()
    }
    pub fn sign(a: impl AsRef<Array>) -> Result<Array> {
        a.as_ref().sign()
    }
    pub fn sigmoid(a: impl AsRef<Array>) -> Result<Array> {
        a.as_ref().sigmoid()
    }
    pub fn multiply(a: impl AsRef<Array>, b: impl AsRef<Array>) -> Result<Array> {
        a.as_ref().multiply(b.as_ref())
    }
    pub fn add(a: impl AsRef<Array>, b: impl AsRef<Array>) -> Result<Array> {
        a.as_ref().add(b.as_ref())
    }
    pub fn divide(a: impl AsRef<Array>, b: impl AsRef<Array>) -> Result<Array> {
        a.as_ref().divide(b.as_ref())
    }
    pub fn maximum(a: impl AsRef<Array>, b: impl AsRef<Array>) -> Result<Array> {
        a.as_ref().maximum(b.as_ref())
    }
    pub fn minimum(a: impl AsRef<Array>, b: impl AsRef<Array>) -> Result<Array> {
        a.as_ref().minimum(b.as_ref())
    }
    pub fn softmax_axis(a: &Array, axis: i32, _precise: bool) -> Result<Array> {
        a.softmax_axis(axis)
    }
    pub fn matmul(a: impl AsRef<Array>, b: impl AsRef<Array>) -> Result<Array> {
        a.as_ref().matmul(b.as_ref())
    }
    pub fn logaddexp(a: impl AsRef<Array>, b: impl AsRef<Array>) -> Result<Array> {
        a.as_ref().logaddexp(b.as_ref())
    }
    pub fn is_nan(a: impl AsRef<Array>) -> Result<Array> {
        a.as_ref().is_nan()
    }
    pub fn floor_divide(a: impl AsRef<Array>, b: impl AsRef<Array>) -> Result<Array> {
        a.as_ref().floor_divide(b.as_ref())
    }
    pub fn r#where(cond: &Array, a: &Array, b: &Array) -> Result<Array> {
        Array::where_(cond, a, b)
    }
    pub fn concatenate(arrays: &[&Array], axis: i32) -> Result<Array> {
        let rank = arrays.first().map(|a| a.t.rank()).unwrap_or(1);
        let inner: Vec<crate::array::Array> = arrays.iter().map(|a| a.t.clone()).collect();
        Ok(Array::new(crate::array::Array::cat(&inner, norm_axis(rank, axis))?))
    }
    pub fn sum(a: &Array, axis: Option<&[i32]>) -> Result<Array> {
        a.sum(axis)
    }
    pub fn broadcast_to(a: &Array, shape: &[i32]) -> Result<Array> {
        a.broadcast_to(shape)
    }

    /// mlx `ops::repeat_axis` (owned array, `T` only for turbofish parity).
    pub fn repeat_axis<T>(array: Array, count: i32, axis: i32) -> Result<Array> {
        let _ = std::marker::PhantomData::<T>;
        array.repeat_axis(count.max(0) as usize, axis)
    }

    /// mlx `ops::stack`.
    pub fn stack(arrays: &[impl AsRef<Array>], axis: i32) -> Result<Array> {
        let rank = arrays.first().map(|a| a.as_ref().t.rank()).unwrap_or(0) + 1;
        let inner: Vec<crate::array::Array> =
            arrays.iter().map(|a| a.as_ref().t.clone()).collect();
        Ok(Array::new(crate::array::Array::stack(&inner, norm_axis(rank, axis))?))
    }

    /// mlx `ops::full`: broadcast `values` to `shape`.
    pub fn full<T>(shape: &[i32], values: impl AsRef<Array>) -> Result<Array> {
        let _ = std::marker::PhantomData::<T>;
        let dims: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
        let v = values.as_ref();
        Ok(Array::new(
            v.t.broadcast_as(&dims)?.contiguous()?,
        ))
    }

    /// mlx `ops::quantize` (affine). Returns `(wq, scales, biases)` with `wq`
    /// packed `bits`-per-element into `u32` along the last axis, and per-group
    /// `scales`/`biases` in the input dtype, matching mlx's affine scheme:
    /// `scale = (max - min) / (2^bits - 1)`, `bias = min`.
    pub fn quantize(
        w: &Array,
        group_size: impl Into<Option<i32>>,
        bits: impl Into<Option<i32>>,
    ) -> Result<(Array, Array, Array)> {
        let gs = group_size.into().unwrap_or(64) as usize;
        let bits = bits.into().unwrap_or(4) as usize;
        let dims: Vec<usize> = w.shape().iter().map(|&d| d as usize).collect();
        let k = *dims.last().unwrap();
        if k % gs != 0 || gs % (32 / bits) != 0 {
            crate::bail!("quantize: group size {gs} incompatible with K={k}, bits={bits}");
        }
        let vals = w.t.cast(Dtype::Float32)?.to_vec::<f32>()?;
        let rows = vals.len() / k;
        let groups = k / gs;
        let maxq = ((1u32 << bits) - 1) as f32;

        let mut wq = vec![0u32; rows * k * bits / 32];
        let mut sc = vec![0f32; rows * groups];
        let mut bi = vec![0f32; rows * groups];
        let per_word = 32 / bits;
        const EPS: f32 = 1e-7;
        for r in 0..rows {
            for g in 0..groups {
                let base = r * k + g * gs;
                // MLX `affine_quantize` (quantized.h): a signed scale and the
                // outer edge as bias, not (max-min)/bins with bias=min.
                let mut w_min = f32::MAX;
                let mut w_max = 0f32;
                for i in 0..gs {
                    let v = vals[base + i];
                    w_min = w_min.min(v);
                    w_max = w_max.max(v);
                }
                let mut scale = ((w_max - w_min) / maxq).max(EPS);
                let side = w_min.abs() > w_max.abs();
                if !side {
                    scale = -scale;
                }
                let edge = if side { w_min } else { w_max };
                let q0 = (edge / scale).round();
                let at_zero = q0 == 0.0;
                if !at_zero {
                    scale = edge / q0;
                }
                let bias = if at_zero { 0.0 } else { edge };
                sc[r * groups + g] = scale;
                bi[r * groups + g] = bias;
                for i in 0..gs {
                    let q = (((vals[base + i] - bias) / scale).round()).min(maxq) as i32;
                    let word = base / per_word + i / per_word;
                    let shift = (i % per_word) * bits;
                    wq[word] |= ((q as u32) & ((1 << bits) - 1)) << shift;
                }
            }
        }
        let rt = Stream::thread_local_or_default().runtime().clone();
        let mut wq_shape: Vec<usize> = dims.clone();
        wq_shape.pop();
        wq_shape.push(k * bits / 32);
        let wq = crate::array::Array::from_slice_dt(&rt, &wq, &wq_shape, Dtype::Uint32)?;
        let mut ss_shape: Vec<usize> = dims.clone();
        ss_shape.pop();
        ss_shape.push(groups);
        let scales = crate::array::Array::from_slice_dt(&rt, &sc, &ss_shape, Dtype::Float32)?;
        let biases = crate::array::Array::from_slice_dt(&rt, &bi, &ss_shape, Dtype::Float32)?;
        let wdt = w.t.dtype();
        Ok((
            Array::new(wq),
            Array::new(scales.to_dtype(wdt)?),
            Array::new(biases.to_dtype(wdt)?),
        ))
    }
    pub fn zeros<T: ZeroElem>(shape: &[i32]) -> Result<Array> {
        let dims: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
        Ok(Array::new(T::zeros(&dims, &Stream::thread_local_or_default().runtime().clone())?))
    }
    /// mlx `ops::logical_not`.
    pub fn logical_not(a: impl AsRef<Array>) -> Result<Array> {
        a.as_ref().logical_not()
    }
    /// Only the last axis is supported (the ported MLX sort kernel is last-axis).
    pub fn argpartition_axis(a: &Array, kth: i32, axis: i32) -> Result<Array> {
        let last = a.rank().saturating_sub(1) as i32;
        if axis != -1 && axis != last {
            crate::bail!("argpartition_axis: only last axis supported");
        }
        Ok(Array::new(crate::mlx_rt::argpartition_axis(
            a.t.device(),
            &a.t,
            kth,
        )?))
    }

    /// mlx `quantized_matmul` (transpose=true, affine). The input is flattened
    /// to 2-D `[M, K]`: `M == 1` takes the qmv kernel, `M > 1` the split-K qmm.
    pub fn quantized_matmul(
        x: &Array,
        w: &Array,
        scales: &Array,
        biases: Option<&Array>,
        transpose: bool,
        group_size: i32,
        bits: i32,
    ) -> Result<Array> {
        if !transpose {
            crate::bail!("quantized_matmul: only transpose=true supported");
        }
        let bi = biases.ok_or_else(|| {
            crate::error::Error::Msg("quantized_matmul: affine biases required".into())
        })?;
        let rank = x.rank();
        let k = x.dim(rank as i32 - 1) as usize;
        let m = x.size() / k;
        let n = w.dim(0) as usize;
        // mlx: out_type = promote_types(x, scales) and every input is cast to it.
        let dt = promote_dtype(x.t.dtype(), scales.t.dtype());
        let x2 = x.t.to_dtype(dt)?.reshape(&[m, k])?.contiguous()?;
        let sc = scales.t.to_dtype(dt)?.contiguous()?;
        let b2 = bi.t.to_dtype(dt)?.contiguous()?;
        if !matches!(dt, Dtype::Bfloat16 | Dtype::Float16 | Dtype::Float32) {
            crate::bail!("quantized_matmul: unsupported compute dtype {dt:?}");
        }
        let out = if m == 1 {
            crate::mlx_rt::affine_qmv_fast(
                x.t.device(),
                &x2,
                &w.t,
                &sc,
                &b2,
                group_size,
                bits,
            )?
        } else if m < crate::mlx_rt::qmv_batch_limit(k, n) {
            // MLX uses qmv_wide for 2 <= M < vector_limit (affine, gen>=15).
            crate::mlx_rt::qmv_wide(x.t.device(), &x2, &w.t, &sc, &b2, group_size, bits)?
        } else {
            // Split-K when the tiling supports it (A/B-verified); qmm_nax is
            // MLX's other non-split path when split_k would collapse to 1.
            match crate::mlx_rt::affine_qmm_splitk(
                x.t.device(),
                &x2,
                &w.t,
                &sc,
                &b2,
                group_size,
                bits,
            ) {
                Ok(o) => o,
                Err(_) if k % 64 == 0 => {
                    crate::mlx_rt::qmm_nax(x.t.device(), &x2, &w.t, &sc, &b2, group_size, bits)?
                }
                Err(e) => return Err(e),
            }
        };
        let mut shape: Vec<usize> = x.shape()[..rank - 1].iter().map(|&d| d as usize).collect();
        shape.push(n);
        Ok(Array::new(out.reshape_dims(shape)?))
    }

    /// mlx `gather_qmm` (affine, transpose). Only the right-sorted path the
    /// engine's MoE uses is ported: `lhs_indices = None`, `sorted_indices =
    /// true`. An `[rows, 1, K]` lhs yields `M = 1, B = rows`, which is exactly
    /// the branch that maps to `gather_qmm_rhs_nax`.
    pub fn gather_qmm(
        x: &Array,
        w: &Array,
        scales: &Array,
        biases: Option<&Array>,
        lhs_indices: Option<&Array>,
        rhs_indices: Option<&Array>,
        transpose: bool,
        group_size: i32,
        bits: i32,
        sorted_indices: bool,
    ) -> Result<Array> {
        if !transpose {
            crate::bail!("gather_qmm: only transpose=true supported");
        }
        if lhs_indices.is_some() {
            crate::bail!("gather_qmm: lhs_indices path not ported");
        }
        if !sorted_indices {
            crate::bail!("gather_qmm: only sorted_indices=true supported");
        }
        let rhs = rhs_indices.ok_or_else(|| {
            crate::error::Error::Msg("gather_qmm: rhs_indices required".into())
        })?;
        let bi = biases.ok_or_else(|| {
            crate::error::Error::Msg("gather_qmm: affine biases required".into())
        })?;
        let rank = x.rank();
        let n = w.dim(w.rank() as i32 - 2) as usize;
        let mut out_shape: Vec<usize> = x.shape()[..rank - 2].iter().map(|&d| d as usize).collect();
        out_shape.push(x.dim(rank as i32 - 2) as usize);
        out_shape.push(n);
        Ok(Array::new(crate::mlx_rt::gather_qmm_rhs_nax(
            x.t.device(),
            None,
            &x.t,
            &w.t,
            &scales.t,
            &bi.t,
            &rhs.t,
            out_shape,
            group_size,
            bits,
        )?))
    }
}

/// `lisa_mlx::ops::indexing`-shaped free functions.
pub mod indexing {
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
}

/// `lisa_mlx::nn`-shaped wrappers, routed through the MLX unary kernels so the
/// rounding matches (an op chain is not bit-exact for `silu`).
pub mod nn {
    use super::*;
    use crate::mlx_rt;

    pub fn silu(a: &Array) -> Result<Array> {
        Ok(Array::new(mlx_rt::silu(a.t.device(), &a.t)?))
    }
}

/// `lisa_mlx::fast`-shaped wrappers for the fused libraries.
pub mod fast {
    use super::*;
    use crate::mlx_rt;

    /// Mask modes for scaled dot product attention.
    pub enum ScaledDotProductAttentionMask<'a> {
        Array(&'a Array),
        Causal,
    }

    /// Mirrors `lisa_mlx`'s `IntoOption<ScaledDotProductAttentionMask>` so call
    /// sites can pass either `Causal` or `&mask` directly.
    pub trait MaskIntoOption<'a> {
        fn into_option(self) -> Option<ScaledDotProductAttentionMask<'a>>;
    }

    impl<'a> MaskIntoOption<'a> for ScaledDotProductAttentionMask<'a> {
        fn into_option(self) -> Option<ScaledDotProductAttentionMask<'a>> {
            Some(self)
        }
    }

    impl<'a> MaskIntoOption<'a> for &'a Array {
        fn into_option(self) -> Option<ScaledDotProductAttentionMask<'a>> {
            Some(ScaledDotProductAttentionMask::Array(self))
        }
    }

    pub fn scaled_dot_product_attention<'a>(
        queries: &Array,
        keys: &Array,
        values: &Array,
        scale: f32,
        mask: impl MaskIntoOption<'a>,
        _sinks: Option<&Array>,
    ) -> Result<Array> {
        let (do_causal, mask_t) = match mask.into_option() {
            Some(ScaledDotProductAttentionMask::Causal) => (true, None),
            Some(ScaledDotProductAttentionMask::Array(m)) => (false, Some(&m.t)),
            None => (false, None),
        };
        Ok(Array::new(mlx_rt::sdpa(
            queries.t.device(),
            &queries.t,
            &keys.t,
            &values.t,
            scale,
            do_causal,
            mask_t,
        )?))
    }

    pub fn rms_norm(x: &Array, weight: Option<&Array>, eps: f32) -> Result<Array> {
        let w = weight.map(|w| &w.t);
        Ok(Array::new(mlx_rt::rms_norm(
            x.t.device(),
            &x.t,
            w,
            eps,
        )?))
    }

    pub fn rope(
        x: &Array,
        dimensions: i32,
        traditional: bool,
        base: Option<f32>,
        scale: f32,
        offset: i32,
        _freqs: Option<&Array>,
    ) -> Result<Array> {
        let off = Array::from_slice(&[offset], &[1]);
        Ok(Array::new(mlx_rt::rope(
            x.t.device(),
            &x.t,
            dimensions,
            traditional,
            base.unwrap_or(10000.0),
            scale,
            &off.t,
            true,
        )?))
    }
}

/// `lisa_mlx::transforms`. There is no "dispatch without wait" equivalent for
/// `async_eval`; `eval` synchronizes the device.
pub mod transforms {
    use super::*;

    pub fn eval<I, A>(arrays: I) -> Result<()>
    where
        I: IntoIterator<Item = A>,
        A: std::borrow::Borrow<Array>,
    {
        for a in arrays {
            a.borrow().t.device().synchronize()?;
        }
        Ok(())
    }

    pub fn async_eval<I, A>(_arrays: I) -> Result<()>
    where
        I: IntoIterator<Item = A>,
        A: std::borrow::Borrow<Array>,
    {
        Ok(())
    }
}

/// `lisa_mlx::memory`.
///
/// The Metal allocator keeps every buffer it ever allocated in a
/// size-bucketed pool and only releases the ones with `strong_count == 1` from
/// `MetalDevice::drop_unused_buffers`, which is called from
/// `wait_until_completed`/`flush_and_wait_current` — **never** from
/// `new_buffer`/`allocate_buffer` (the field doc promises a sweep on every
/// allocation; the code does not do it). lisa pipelines a whole prefill chunk
/// (or MTP round) without intermediate evals, so without an explicit sweep the
/// pool grows for the entire run and drives the box past its RAM. These mirror
/// The buffer-pool cap (8 GiB).
pub mod memory {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Excess the pool may grow past the baseline before a sweep is forced.
    /// `0` disables the automatic sweep.
    static CACHE_LIMIT: AtomicUsize = AtomicUsize::new(0);
    /// Device allocation observed on the first `trim_cache` (the weights + the
    /// steady-state caches); only growth past this is pool churn.
    static CACHE_BASELINE: AtomicUsize = AtomicUsize::new(0);

    fn with_dev<T>(f: impl FnOnce(&crate::runtime::MetalRuntime) -> T) -> Option<T> {
        let rt = Stream::thread_local_or_default().runtime().clone();
        Some(f(&rt))
    }

    pub fn set_cache_limit(bytes: usize) {
        CACHE_LIMIT.store(bytes, Ordering::Relaxed);
    }

    /// Commit, wait, and release every pooled buffer whose only owner is the
    /// allocator. The public path to `drop_unused_buffers`.
    pub fn clear_cache() {
        with_dev(|m| {
            let _ = m.synchronize();
            m.pool.sweep();
        });
    }

    /// Release pooled buffers once the device's allocation has grown more than
    /// the configured limit past the baseline. Cheap when under the limit (no
    /// sync), so it is safe to call on chunk/round boundaries and periodically
    /// during decode.
    pub fn trim_cache() {
        let limit = CACHE_LIMIT.load(Ordering::Relaxed);
        if limit == 0 {
            return;
        }
        with_dev(|m| {
            let now = m.pool.bytes();
            let base = CACHE_BASELINE.load(Ordering::Relaxed);
            if base == 0 {
                CACHE_BASELINE.store(now, Ordering::Relaxed);
            } else if now > base.saturating_add(limit) {
                if std::env::var("LISA_MEM_TRACE").is_ok() {
                    eprintln!("[mem-trace] pool sweep: {:.2} GB > base {:.2} GB + {:.2} GB",
                        now as f64 / 1e9, base as f64 / 1e9, limit as f64 / 1e9);
                }
                let _ = m.synchronize();
                m.pool.sweep();
            }
        });
    }

    pub fn active_memory() -> Result<usize> {
        // The pool's byte accounting (the runtime owns all allocations).
        Ok(with_dev(|m| m.pool.bytes()).unwrap_or(0))
    }

    /// No high-water mark is tracked, so report the live allocation.
    pub fn peak_memory() -> Result<usize> {
        active_memory()
    }

    pub fn cache_memory() -> Result<usize> {
        Ok(0)
    }

    pub fn memory_limit() -> Result<usize> {
        // No public working-set query on our runtime; report 0 (callers treat
        // 0 as "no cap").
        Ok(0)
    }
}






