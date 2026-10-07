use std::sync::Arc;

use crate::error::Result;
use crate::runtime::{Buffer, MetalRuntime};

use super::copy::{
    COPY_STRIDED_SOURCE, COPY2_STRIDED_SOURCE, Copy2Params, CopyParams, MAX_COPY_RANK, pad_rank,
};
use super::{Dtype, Layout, err};

// ─────────────────────────────── array ───────────────────────────────

/// An eager tensor: a buffer, a layout into it, and a dtype.
#[derive(Clone)]
pub struct Array {
    pub(super) rt: Arc<MetalRuntime>,
    pub(super) buf: Arc<Buffer>,
    pub(super) layout: Layout,
    pub(super) dtype: Dtype,
    /// Host readback cache for `as_slice`. An `Array` is immutable after
    /// construction, so the cache never goes stale.
    pub(super) host: Arc<std::sync::OnceLock<Vec<u8>>>,
}

impl Array {
    /// The one place the struct's fields are assembled.
    pub(super) fn wrap(
        rt: Arc<MetalRuntime>,
        buf: Arc<Buffer>,
        layout: Layout,
        dtype: Dtype,
    ) -> Self {
        Self {
            rt,
            buf,
            layout,
            dtype,
            host: Arc::new(std::sync::OnceLock::new()),
        }
    }

    // ── constructors ──

    /// Copy host data into a fresh contiguous buffer.
    pub fn from_slice<T: Copy>(
        rt: &Arc<MetalRuntime>,
        data: &[T],
        shape: &[usize],
    ) -> Result<Self> {
        let expect: usize = shape.iter().product();
        if expect != data.len() {
            return Err(err(format!(
                "from_slice: {shape:?} needs {expect} elements, got {}",
                data.len()
            )));
        }
        let bytes = expect * std::mem::size_of::<T>();
        let buf = rt.buffer(bytes, "array")?;
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr() as *const u8, buf.contents(), bytes);
        }
        Ok(Self::wrap(
            Arc::clone(rt),
            buf,
            Layout::contiguous(shape),
            Dtype::F32,
        ))
    }

    /// `from_slice` with an explicit dtype.
    pub fn from_slice_dt<T: Copy>(
        rt: &Arc<MetalRuntime>,
        data: &[T],
        shape: &[usize],
        dtype: Dtype,
    ) -> Result<Self> {
        let mut a = Self::from_slice(rt, data, shape)?;
        a.dtype = dtype;
        Ok(a)
    }

    /// Zeroed contiguous storage (dtype-typed).
    pub fn zeros(rt: &Arc<MetalRuntime>, shape: &[usize], dtype: Dtype) -> Result<Self> {
        let buf = rt.buffer(Layout::contiguous(shape).size() * dtype.size_of(), "zeros")?;
        unsafe { std::ptr::write_bytes(buf.contents(), 0, buf.length()) };
        Ok(Self::wrap(
            Arc::clone(rt),
            buf,
            Layout::contiguous(shape),
            dtype,
        ))
    }

    // ── metadata ──

    pub fn shape(&self) -> &[usize] {
        &self.layout.shape
    }

    /// `dims()` (the name `mlx_rt` uses).
    pub fn dims(&self) -> &[usize] {
        &self.layout.shape
    }

    /// `elem_count()`.
    pub fn elem_count(&self) -> usize {
        self.size()
    }

    /// `stride()` (element strides).
    pub fn stride(&self) -> &[usize] {
        &self.layout.strides
    }

    /// `slice_assign`: a copy of `self` with `src` written into the
    /// given ranges. Host-side (the buffers are shared); a kernel path is a
    /// later optimization.
    pub fn slice_assign(&self, ranges: &[std::ops::Range<usize>], src: &Array) -> Result<Self> {
        if ranges.len() != self.rank() {
            return Err(err("slice_assign: rank mismatch"));
        }
        if src.shape() != ranges.iter().map(|r| r.len()).collect::<Vec<_>>() {
            return Err(err("slice_assign: src shape mismatch"));
        }
        // GPU strided->strided copy (slice_assign is a GPU kernel too);
        // a host memcpy here would read the source before its producing kernels
        // complete (the CPU runs ahead of the GPU) and write zeros.
        let out = self.copied()?;
        let mut dst = out.clone();
        for (axis, r) in ranges.iter().enumerate() {
            dst = dst.narrow(axis, r.start, r.len())?;
        }
        src.copy_strided_into(&dst)?;
        Ok(out)
    }

    /// GPU strided->strided copy: `self` (any layout) into `dst` (any layout),
    /// same shape. The dependency is carried by the command encoder.
    pub fn copy_strided_into(&self, dst: &Array) -> Result<()> {
        if self.shape() != dst.shape() || self.dtype != dst.dtype {
            return Err(err("copy_strided_into: shape/dtype mismatch"));
        }
        let esz = self.dtype.size_of();
        let name = format!("copy2_strided_{}", self.dtype.metal_name());
        let pipe = self.rt.compile(COPY2_STRIDED_SOURCE, &name)?;
        let params = Copy2Params {
            ndim: self.rank() as u32,
            shape: pad_rank(self.shape()),
            src_strides: pad_rank(&self.layout.strides),
            dst_strides: pad_rank(&dst.layout.strides),
        };
        {
            let guard = self.rt.commands.encoder()?;
            let enc = guard.encoder();
            enc.set_pipeline(&pipe);
            enc.set_input(0, Some(&self.buf), self.layout.offset * esz);
            enc.set_output(1, Some(&dst.buf), dst.layout.offset * esz);
            enc.set_bytes(2, &params);
            let n = self.size();
            enc.dispatch_threads((n, 1, 1), (256.min(n).max(1), 1, 1));
        }
        Ok(())
    }

    /// `dims2()`.
    pub fn dims2(&self) -> (usize, usize) {
        (self.layout.shape[0], self.layout.shape[1])
    }

    /// `layout()`.
    pub fn layout_ref(&self) -> &Layout {
        &self.layout
    }

    /// Build from a buffer + shape + dtype: the native equivalent of
    /// `Tensor::from_storage` (the shape is taken as contiguous).
    pub fn from_parts(
        rt: &Arc<MetalRuntime>,
        buf: Arc<Buffer>,
        shape: &[usize],
        dtype: Dtype,
    ) -> Self {
        Self::wrap(Arc::clone(rt), buf, Layout::contiguous(shape), dtype)
    }

    /// `storage_and_layout()` equivalent.
    pub fn buffer_and_layout(&self) -> (&Arc<Buffer>, &Layout) {
        (&self.buf, &self.layout)
    }

    pub fn dim(&self, axis: usize) -> usize {
        self.layout.shape[axis]
    }

    pub fn rank(&self) -> usize {
        self.layout.rank()
    }

    pub fn size(&self) -> usize {
        self.layout.size()
    }

    pub fn dtype(&self) -> Dtype {
        self.dtype
    }

    pub fn layout(&self) -> &Layout {
        &self.layout
    }

    pub fn buffer(&self) -> &Arc<Buffer> {
        &self.buf
    }

    pub fn runtime(&self) -> &Arc<MetalRuntime> {
        &self.rt
    }

    // ── views (metadata only) ──

    /// Reshape a contiguous array (a view; strides are recomputed row-major).
    pub fn reshape(&self, shape: &[usize]) -> Result<Self> {
        if !self.layout.is_contiguous() {
            // a non-contiguous view is materialised before reshaping.
            return self.copied()?.reshape(shape);
        }
        let expect: usize = shape.iter().product();
        if expect != self.size() {
            return Err(err(format!(
                "reshape: {shape:?} needs {expect} elements, have {}",
                self.size()
            )));
        }
        let mut out = self.clone();
        out.layout = Layout {
            shape: shape.to_vec(),
            strides: Layout::contiguous(shape).strides,
            offset: self.layout.offset,
        };
        Ok(out)
    }

    /// Reshape to `[rows, cols]` as a VIEW when the layout admits a uniform
    /// row stride (every flattened leading index maps to `idx * row_stride`).
    /// Returns `None` when the general case needs a materializing copy —
    /// callers fall back to `reshape` (which copies non-contiguous input).
    /// Rank-2 and rank-3 inputs cover the fused-kernel call sites (verify
    /// slice views of a fused projection, contiguous `[rows, V]` arrays).
    pub fn reshape_rows_view(&self, rows: usize, cols: usize) -> Option<Self> {
        let sh = self.layout.shape.clone();
        let st = self.layout.strides.clone();
        let total: usize = sh.iter().product();
        if total != rows * cols {
            return None;
        }
        let n = sh.len();
        if st[n - 1] != 1 {
            return None;
        }
        if n == 2 {
            if sh[0] == rows && sh[1] == cols {
                let mut out = self.clone();
                out.layout = Layout {
                    shape: vec![rows, cols],
                    strides: vec![st[0], 1],
                    offset: self.layout.offset,
                };
                return Some(out);
            }
            return None;
        }
        if n == 3 {
            // [b, s, c] -> [b*s, c]: uniform iff st[0] == s * st[1].
            if sh[2] == cols && sh[0] * sh[1] == rows && st[0] == sh[1] * st[1] {
                let mut out = self.clone();
                out.layout = Layout {
                    shape: vec![rows, cols],
                    strides: vec![st[1], 1],
                    offset: self.layout.offset,
                };
                return Some(out);
            }
            return None;
        }
        None
    }

    /// Permute dimensions (a view) — `permute`.
    pub fn permute(&self, perm: &[usize]) -> Result<Self> {
        if perm.len() != self.rank() {
            return Err(err("transpose: rank mismatch"));
        }
        let mut out = self.clone();
        out.layout = Layout {
            shape: perm.iter().map(|&a| self.layout.shape[a]).collect(),
            strides: perm.iter().map(|&a| self.layout.strides[a]).collect(),
            offset: self.layout.offset,
        };
        Ok(out)
    }

    /// Insert a size-1 axis (a view).
    pub fn expand_dims(&self, axis: usize) -> Result<Self> {
        if axis > self.rank() {
            return Err(err("expand_dims: axis out of range"));
        }
        let mut shape = self.layout.shape.clone();
        let mut strides = self.layout.strides.clone();
        shape.insert(axis, 1);
        // A fresh size-1 dim can take any stride; 0 is what MLX uses.
        strides.insert(axis, 0);
        let mut out = self.clone();
        out.layout = Layout {
            shape,
            strides,
            offset: self.layout.offset,
        };
        Ok(out)
    }

    /// A `len`-long slice along `axis` starting at `start` (a view).
    pub fn narrow(&self, axis: usize, start: usize, len: usize) -> Result<Self> {
        if axis >= self.rank() || start + len > self.layout.shape[axis] {
            return Err(err("narrow: out of range"));
        }
        let mut out = self.clone();
        out.layout.shape[axis] = len;
        out.layout.offset += start * self.layout.strides[axis];
        Ok(out)
    }

    /// Broadcast a size-1 axis (a view, stride 0).
    pub fn broadcast(&self, shape: &[usize]) -> Result<Self> {
        if shape.len() != self.rank() {
            return Err(err("broadcast: rank mismatch"));
        }
        let mut strides = self.layout.strides.clone();
        for (i, (&want, &have)) in shape.iter().zip(self.layout.shape.iter()).enumerate() {
            if have == want {
                continue;
            }
            if have != 1 {
                return Err(err(format!("broadcast: incompatible shape {:?} -> {:?}", self.layout.shape, shape)));
            }
            strides[i] = 0;
        }
        let mut out = self.clone();
        out.layout = Layout {
            shape: shape.to_vec(),
            strides,
            offset: self.layout.offset,
        };
        Ok(out)
    }

    // ── materialisation ──

    /// A contiguous copy of this array (itself when already contiguous). A
    /// view with a non-zero offset but contiguous strides is already dense, so
    /// it is returned as-is — matching MLX.
    pub fn contiguous(&self) -> Result<Self> {
        if self.layout.is_contiguous() {
            return Ok(self.clone());
        }
        self.copied()
    }

    /// Always materialise: a fresh offset-0 contiguous buffer. Needed when
    /// handing the data to something that ignores a view's offset (e.g.
    /// `Tensor::from_storage`).
    pub fn copied(&self) -> Result<Self> {
        let out = Self::wrap(
            Arc::clone(&self.rt),
            self.rt
                .buffer(self.size() * self.dtype.size_of(), "contiguous")?,
            Layout::contiguous(&self.layout.shape),
            self.dtype,
        );
        let src_ptr = if self.layout.rank() <= MAX_COPY_RANK {
            self.copy_params()?
        } else {
            return Err(err("contiguous: rank too high"));
        };
        let name = format!("copy_strided_{}", self.dtype.metal_name());
        let pipe = self.rt.compile(COPY_STRIDED_SOURCE, &name)?;
        {
            let guard = self.rt.commands.encoder()?;
            let enc = guard.encoder();
            enc.set_pipeline(&pipe);
            enc.set_input(
                0,
                Some(&self.buf),
                self.layout.offset * self.dtype.size_of(),
            );
            enc.set_output(1, Some(&out.buf), 0);
            enc.set_bytes(2, &src_ptr);
            let n = self.size();
            enc.dispatch_threads((n, 1, 1), (256.min(n).max(1), 1, 1));
        }
        Ok(out)
    }

    fn copy_params(&self) -> Result<CopyParams> {
        if self.rank() > MAX_COPY_RANK {
            return Err(err("copy: rank too high"));
        }
        let mut p = CopyParams {
            ndim: self.rank() as u32,
            shape: [1; MAX_COPY_RANK],
            strides: [0; MAX_COPY_RANK],
        };
        for (i, (&d, &s)) in self
            .layout
            .shape
            .iter()
            .zip(self.layout.strides.iter())
            .enumerate()
        {
            p.shape[i] = d as u32;
            p.strides[i] = s as u32;
        }
        Ok(p)
    }

    // ── eval / readback ──

    /// Flush and wait for everything queued on the runtime.
    pub fn eval(&self) -> Result<()> {
        self.rt.commands.flush_and_wait()
    }

    /// Flush and wait only until command buffer `cb_id` (captured earlier via
    /// `MetalRuntime::commands.current_cb_id()`) has completed; later buffers
    /// stay in flight (step-overlap readback). See
    /// `Commands::flush_wait_through`.
    pub fn eval_wait_through(&self, cb_id: u64) -> Result<()> {
        self.rt.commands.flush_wait_through(cb_id)
    }

    /// The id of the command buffer dispatches are currently encoding into
    /// (the barrier id for [`Array::eval_wait_through`]).
    pub fn readback_cb_id(&self) -> u64 {
        self.rt.commands.current_cb_id()
    }

    /// Read the elements back to the host, walking the layout.
    pub fn to_vec<T: Copy>(&self) -> Result<Vec<T>> {
        if self.dtype.size_of() != std::mem::size_of::<T>() {
            return Err(err("to_vec: element size mismatch"));
        }
        self.eval()?;
        // Readback-race detector (specs/04 §6): after `eval`, every command
        // buffer that ever bound this buffer must be at or below the
        // synchronously-advanced `completed_id`. A hit proves the producing
        // encoder was still in flight at read time. Gated on the static so the
        // off-path cost is one relaxed load, not an environment lookup per
        // readback (this runs per sampled token in serve).
        static RB_DEBUG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if *RB_DEBUG.get_or_init(|| std::env::var_os("LISA_RB_DEBUG").is_some()) {
            let used = self.buf.debug_used_cb();
            let waited = self.rt.commands.completed_id();
            if used > waited {
                eprintln!(
                    "[rb-race] buf#{} used_cb={used} > waited_id={waited}",
                    self.buf.id()
                );
            }
        }
        let base = self.buf.contents() as *const u8;
        let esz = self.dtype.size_of();
        let n = self.size();
        let mut out: Vec<T> = Vec::with_capacity(n);
        let mut idx = vec![0usize; self.rank()];
        for _ in 0..n {
            let off = self.layout.offset
                + idx
                    .iter()
                    .zip(self.layout.strides.iter())
                    .map(|(&i, &s)| i * s)
                    .sum::<usize>();
            unsafe {
                out.push(std::ptr::read_unaligned(base.add(off * esz) as *const T));
            }
            // odometer over shape
            for d in (0..self.rank()).rev() {
                idx[d] += 1;
                if idx[d] < self.layout.shape[d] {
                    break;
                }
                idx[d] = 0;
            }
        }
        Ok(out)
    }
}
