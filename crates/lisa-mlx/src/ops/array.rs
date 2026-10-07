use super::*;

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
            crate::array::Array::scalar_of(&Self::runtime_ref(), val, dtype).expect("from_f32_as"),
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
        // The source may be an `mmap` slice at an odd byte offset (safetensors
        // header length is not padded), so element loads through `T*` would be
        // misaligned. Copy through an 8-aligned buffer via unaligned reads when
        // needed; MLX copies the data anyway, so this only doubles a one-shot
        // load copy for the rare misaligned tensors.
        let mut realigned: Vec<u64>;
        let mut ptr = ptr;
        let esize = match dtype {
            Dtype::Bfloat16 | Dtype::Float16 | Dtype::Int16 => 2,
            Dtype::Float32 | Dtype::Uint32 | Dtype::Int32 => 4,
            Dtype::Float64 | Dtype::Int64 => 8,
            Dtype::Uint8 | Dtype::Bool => 1,
            other => panic!("from_raw_data: unsupported dtype {other:?}"),
        };
        if esize > 1 && ptr as usize % esize != 0 {
            let total = n * esize;
            realigned = vec![0u64; total.div_ceil(8)];
            let dst = realigned.as_mut_ptr() as *mut u8;
            for i in 0..total {
                unsafe { dst.add(i).write((ptr as *const u8).add(i).read()) };
            }
            ptr = realigned.as_ptr() as *const std::ffi::c_void;
        }
        let rt = Stream::thread_local_or_default().runtime().clone();
        let a = unsafe {
            match dtype {
                Dtype::Bfloat16 => crate::array::Array::from_slice_dt(
                    &rt,
                    std::slice::from_raw_parts(ptr as *const half::bf16, n),
                    &dims,
                    dtype,
                ),
                Dtype::Float16 => crate::array::Array::from_slice_dt(
                    &rt,
                    std::slice::from_raw_parts(ptr as *const half::f16, n),
                    &dims,
                    dtype,
                ),
                Dtype::Float32 => crate::array::Array::from_slice_dt(
                    &rt,
                    std::slice::from_raw_parts(ptr as *const f32, n),
                    &dims,
                    dtype,
                ),
                Dtype::Float64 => crate::array::Array::from_slice_dt(
                    &rt,
                    std::slice::from_raw_parts(ptr as *const f64, n),
                    &dims,
                    dtype,
                ),
                Dtype::Uint8 => crate::array::Array::from_slice_dt(
                    &rt,
                    std::slice::from_raw_parts(ptr as *const u8, n),
                    &dims,
                    dtype,
                ),
                Dtype::Uint32 => crate::array::Array::from_slice_dt(
                    &rt,
                    std::slice::from_raw_parts(ptr as *const u32, n),
                    &dims,
                    dtype,
                ),
                Dtype::Int16 => crate::array::Array::from_slice_dt(
                    &rt,
                    std::slice::from_raw_parts(ptr as *const i16, n),
                    &dims,
                    dtype,
                ),
                Dtype::Int32 => crate::array::Array::from_slice_dt(
                    &rt,
                    std::slice::from_raw_parts(ptr as *const i32, n),
                    &dims,
                    dtype,
                ),
                Dtype::Int64 => crate::array::Array::from_slice_dt(
                    &rt,
                    std::slice::from_raw_parts(ptr as *const i64, n),
                    &dims,
                    dtype,
                ),
                other => panic!("from_raw_data: unsupported dtype {other:?}"),
            }
        }
        .expect("from_raw_data");
        Array::new(a)
    }

    /// mlx `Array::sum_axis` (last-axis sum via the ported reduce kernel).
    pub fn sum_axis(&self, axis: i32, keep_dims: impl Into<Option<bool>>) -> Result<Self> {
        let keep = keep_dims.into().unwrap_or(false);
        let r = crate::jit::reduce_axis(self.t.device(), &self.t, axis, "sum")?;
        Ok(Self::new(if keep {
            r.unsqueeze(self.axis(axis))?
        } else {
            r
        }))
    }

    /// mlx `Array::mean_axis`.
    pub fn mean_axis(&self, axis: i32, keep_dims: impl Into<Option<bool>>) -> Result<Self> {
        let keep = keep_dims.into().unwrap_or(false);
        let r = crate::jit::reduce_axis(self.t.device(), &self.t, axis, "mean")?;
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
                    std::ptr::copy_nonoverlapping(v.as_ptr() as *const u8, b.as_mut_ptr(), b.len());
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
        crate::jit::put_along_axis(self.t.device(), &self.t, &indices.t, &values.t, axis)
            .map(Self::new)
    }

    /// mlx `Array::split_equal`.
    pub fn split_equal(&self, num_splits: usize, axis: i32) -> Result<Vec<Self>> {
        let parts = self.t.chunk(num_splits, self.axis(axis))?;
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

    /// View-reshape to `[rows, cols]` when the layout admits a uniform row
    /// stride (see `crate::array::Array::reshape_rows_view`); `None` when a
    /// materializing copy would be needed.
    pub fn reshape_rows_view(&self, rows: usize, cols: usize) -> Option<Self> {
        self.t.reshape_rows_view(rows, cols).map(Self::new)
    }

    /// Materialized copy (a fresh contiguous buffer).
    pub fn copied(&self) -> Result<Self> {
        Ok(Self::new(self.t.copied()?))
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

    /// The id of the command buffer dispatches are currently encoding into
    /// (the barrier id for [`Array::eval_wait_through`]).
    pub fn readback_cb_id(&self) -> u64 {
        self.t.readback_cb_id()
    }

    /// Partial readback: wait only through `cb_id`; later command buffers
    /// (e.g. a speculative pre-draft enqueued behind the verify) stay in
    /// flight and run under the host's next step. See
    /// `Commands::flush_wait_through`.
    pub fn eval_wait_through(&self, cb_id: u64) -> Result<()> {
        self.t.eval_wait_through(cb_id)
    }

    pub fn item<T: Copy + 'static>(&self) -> T {
        self.t.item::<T>()
    }

    pub fn to_vec1<T: Copy + 'static>(&self) -> Result<Vec<T>> {
        self.t.flatten_all()?.contiguous()?.to_vec1::<T>()
    }

    // ---- elementwise ----

    pub fn add(&self, rhs: impl AsRef<Array>) -> Result<Self> {
        Ok(Self::new(promoted_binary(
            &self.t,
            &rhs.as_ref().t,
            |a, b| a.broadcast_add(b),
        )?))
    }

    pub fn subtract(&self, rhs: impl AsRef<Array>) -> Result<Self> {
        Ok(Self::new(promoted_binary(
            &self.t,
            &rhs.as_ref().t,
            |a, b| a.broadcast_sub(b),
        )?))
    }

    pub fn multiply(&self, rhs: impl AsRef<Array>) -> Result<Self> {
        Ok(Self::new(promoted_binary(
            &self.t,
            &rhs.as_ref().t,
            |a, b| a.broadcast_mul(b),
        )?))
    }

    pub fn divide(&self, rhs: impl AsRef<Array>) -> Result<Self> {
        Ok(Self::new(promoted_binary(
            &self.t,
            &rhs.as_ref().t,
            |a, b| a.broadcast_div(b),
        )?))
    }

    pub fn maximum(&self, rhs: impl AsRef<Array>) -> Result<Self> {
        Ok(Self::new(promoted_binary(
            &self.t,
            &rhs.as_ref().t,
            |a, b| a.maximum(b),
        )?))
    }

    pub fn minimum(&self, rhs: impl AsRef<Array>) -> Result<Self> {
        Ok(Self::new(promoted_binary(
            &self.t,
            &rhs.as_ref().t,
            |a, b| a.minimum(b),
        )?))
    }

    /// MLX unary op by name (`"erf"`, `"gelu"`, …). Dispatches to the built-in
    /// unary kernel (`unary_ops.metal`) with `op` as the functor type.
    pub fn unary(&self, op: &str) -> Result<Self> {
        Ok(Self::new(self.t.unary(op)?))
    }

    pub fn exp(&self) -> Result<Self> {
        let t = crate::jit::exp(self.t.device(), &self.t)?;
        Ok(Self::new(t))
    }

    pub fn log(&self) -> Result<Self> {
        Ok(Self::new(crate::jit::log(self.t.device(), &self.t)?))
    }

    pub fn sin(&self) -> Result<Self> {
        Ok(Self::new(self.t.sin()?))
    }

    pub fn cos(&self) -> Result<Self> {
        Ok(Self::new(self.t.cos()?))
    }

    pub fn sqrt(&self) -> Result<Self> {
        Ok(Self::new(crate::jit::sqrt(self.t.device(), &self.t)?))
    }

    pub fn rsqrt(&self) -> Result<Self> {
        Ok(Self::new(crate::jit::rsqrt(self.t.device(), &self.t)?))
    }

    pub fn abs(&self) -> Result<Self> {
        Ok(Self::new(crate::jit::abs(self.t.device(), &self.t)?))
    }

    pub fn sign(&self) -> Result<Self> {
        Ok(Self::new(crate::jit::sign(self.t.device(), &self.t)?))
    }

    pub fn neg(&self) -> Result<Self> {
        Ok(Self::new(self.t.neg()?))
    }

    pub fn sigmoid(&self) -> Result<Self> {
        Ok(Self::new(crate::jit::sigmoid(self.t.device(), &self.t)?))
    }

    pub fn silu(&self) -> Result<Self> {
        Ok(Self::new(crate::jit::silu(self.t.device(), &self.t)?))
    }

    pub fn log1p(&self) -> Result<Self> {
        Ok(Self::new(crate::jit::log1p(self.t.device(), &self.t)?))
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
        Ok(Self::new(crate::array::Array::cat(
            &inner,
            norm_axis(rank, axis),
        )?))
    }

    pub fn stack(arrays: &[&Self], axis: i32) -> Result<Self> {
        let rank = arrays.first().map(|a| a.t.rank()).unwrap_or(0) + 1;
        let inner: Vec<crate::array::Array> = arrays.iter().map(|a| a.t.clone()).collect();
        Ok(Self::new(crate::array::Array::stack(
            &inner,
            norm_axis(rank, axis),
        )?))
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
            return Ok(Self::new(crate::jit::softmax_last_axis(
                self.t.device(),
                &self.t,
                true,
            )?));
        }
        let t = self.t.transpose(ax, self.t.rank() - 1)?.contiguous()?;
        let r = crate::jit::softmax_last_axis(self.t.device(), &t, true)?;
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
