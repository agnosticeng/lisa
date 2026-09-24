//! A native, eager `Array` over `MTLBuffer` with a
//! `Layout` (shape / element strides / element offset) and metadata-only views.
//!
//! Additive — the shim's `shim_api::Array` is still the one the engine uses.
//! Stage 3 migrates the `ops` families onto this type behind that same shim API,
//! each family is validated against the 256/256 golden.
//!
//! Eager here means an op dispatches its kernel into the runtime's batched
//! command buffer as soon as it is called (no lazy graph). The GPU still
//! pipelines while lisa builds a whole chunk, because the runtime only commits
//! every `per_buffer` dispatches (see `runtime::Commands`).

use std::sync::Arc;

use crate::error::{Error, Result};
use crate::runtime::{Buffer, MetalRuntime};

fn err(m: impl Into<String>) -> Error {
    Error::Msg(m.into())
}

// ─────────────────────────────── dtype ───────────────────────────────

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Dtype {
    Bool,
    Uint8,
    Uint16,
    Uint32,
    Int8,
    Int16,
    Int32,
    Int64,
    Float16,
    Float32,
    Float64,
    Bfloat16,
}

impl Dtype {
    // short aliases, so `mlx_rt`'s `DType::F32` etc. keep
    // reading the same.
    #[allow(non_upper_case_globals)]
    pub const U8: Dtype = Dtype::Uint8;
    #[allow(non_upper_case_globals)]
    pub const U16: Dtype = Dtype::Uint16;
    #[allow(non_upper_case_globals)]
    pub const I8: Dtype = Dtype::Int8;
    #[allow(non_upper_case_globals)]
    pub const U32: Dtype = Dtype::Uint32;
    #[allow(non_upper_case_globals)]
    pub const I16: Dtype = Dtype::Int16;
    #[allow(non_upper_case_globals)]
    pub const I32: Dtype = Dtype::Int32;
    #[allow(non_upper_case_globals)]
    pub const I64: Dtype = Dtype::Int64;
    #[allow(non_upper_case_globals)]
    pub const F16: Dtype = Dtype::Float16;
    #[allow(non_upper_case_globals)]
    pub const F32: Dtype = Dtype::Float32;
    #[allow(non_upper_case_globals)]
    pub const F64: Dtype = Dtype::Float64;
    #[allow(non_upper_case_globals)]
    pub const BF16: Dtype = Dtype::Bfloat16;

    /// Bytes per element.
    pub fn size_of(self) -> usize {
        match self {
            Dtype::Bool | Dtype::Uint8 | Dtype::Int8 => 1,
            Dtype::Uint16 | Dtype::Int16 | Dtype::Float16 | Dtype::Bfloat16 => 2,
            Dtype::Uint32 | Dtype::Int32 | Dtype::Float32 => 4,
            Dtype::Int64 | Dtype::Float64 => 8,
        }
    }

    /// `size_in_bytes()`.
    pub fn size_in_bytes(self) -> usize {
        self.size_of()
    }

    pub fn is_float(self) -> bool {
        matches!(self, Dtype::Float16 | Dtype::Float32 | Dtype::Float64 | Dtype::Bfloat16)
    }

    /// The MLX type tag; matches `mlx_rt`'s `type_to_name`.
    pub fn mlx_name(self) -> &'static str {
        match self {
            Dtype::Uint8 => "uint8",
            Dtype::Uint16 => "uint16",
            Dtype::Uint32 => "uint32",
            Dtype::Int16 => "int16",
            Dtype::Int32 => "int32",
            Dtype::Int64 => "int64",
            Dtype::Float16 => "float16",
            Dtype::Float32 => "float32",
            Dtype::Float64 => "double",
            Dtype::Bfloat16 => "bfloat16",
            Dtype::Bool | Dtype::Int8 => "uint8",
        }
    }

    /// The Metal spelling; matches `mlx_rt`'s `type_string`.
    pub fn metal_name(self) -> &'static str {
        match self {
            Dtype::Bool | Dtype::Uint8 => "uint8_t",
            Dtype::Uint16 => "uint16_t",
            Dtype::Uint32 => "uint32_t",
            Dtype::Int8 => "int8_t",
            Dtype::Int16 => "int16_t",
            Dtype::Int32 => "int32_t",
            Dtype::Int64 => "int64_t",
            Dtype::Float16 => "float16_t",
            Dtype::Float32 => "float",
            Dtype::Float64 => "double",
            Dtype::Bfloat16 => "bfloat16_t",
        }
    }
}

// ─────────────────────────────── layout ───────────────────────────────

/// Shape, element strides and element offset. Strides are in elements (not
/// bytes) so a view can be described without knowing the dtype.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Layout {
    pub shape: Vec<usize>,
    pub strides: Vec<usize>,
    pub offset: usize,
}

impl Layout {
    /// Row-major strides for `shape`.
    pub fn contiguous(shape: &[usize]) -> Self {
        let mut strides = vec![0usize; shape.len()];
        let mut acc = 1usize;
        for i in (0..shape.len()).rev() {
            strides[i] = acc;
            acc *= shape[i];
        }
        Self {
            shape: shape.to_vec(),
            strides,
            offset: 0,
        }
    }

    pub fn size(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn rank(&self) -> usize {
        self.shape.len()
    }

    /// `Layout::stride()`.
    pub fn stride(&self) -> &[usize] {
        &self.strides
    }

    /// `Layout::shape()`.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// `Layout::start_offset()`.
    pub fn start_offset(&self) -> usize {
        self.offset
    }

    /// Row-major contiguous, treating size-1 dimensions as stride-agnostic
    /// (as MLX does).
    pub fn is_contiguous(&self) -> bool {
        let mut expected = 1usize;
        for (&d, &s) in self.shape.iter().zip(self.strides.iter()).rev() {
            if d != 1 && s != expected {
                return false;
            }
            expected *= d;
        }
        true
    }
}

// ─────────────────────────────── array ───────────────────────────────

/// An eager tensor: a buffer, a layout into it, and a dtype.
#[derive(Clone)]
pub struct Array {
    rt: Arc<MetalRuntime>,
    buf: Arc<Buffer>,
    layout: Layout,
    dtype: Dtype,
    /// Host readback cache for `as_slice`. An `Array` is immutable after
    /// construction, so the cache never goes stale.
    host: Arc<std::sync::OnceLock<Vec<u8>>>,
}

impl Array {
    /// The one place the struct's fields are assembled.
    fn wrap(rt: Arc<MetalRuntime>, buf: Arc<Buffer>, layout: Layout, dtype: Dtype) -> Self {
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
    pub fn from_slice<T: Copy>(rt: &Arc<MetalRuntime>, data: &[T], shape: &[usize]) -> Result<Self> {
        let expect: usize = shape.iter().product();
        if expect != data.len() {
            return Err(err(format!(
                "from_slice: {shape:?} needs {expect} elements, got {}",
                data.len()
            )));
        }
        let bytes = expect * std::mem::size_of::<T>();
        if crate::runtime::env_flag("LISA_SYNC_HOST_WRITE") {
            rt.commands.flush_and_wait()?;
        }
        let buf = rt.buffer(bytes, "array")?;
        unsafe {
            std::ptr::copy_nonoverlapping(
                data.as_ptr() as *const u8,
                buf.contents(),
                bytes,
            );
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
        Ok(Self::wrap(Arc::clone(rt), buf, Layout::contiguous(shape), dtype))
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
    pub fn slice_assign(
        &self,
        ranges: &[std::ops::Range<usize>],
        src: &Array,
    ) -> Result<Self> {
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
        let pipe = self
            .rt
            .compile(COPY2_STRIDED_SOURCE, &name)?;
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
                return Err(err("broadcast: incompatible shape"));
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
        let out = Self::wrap(Arc::clone(&self.rt), self.rt.buffer(self.size() * self.dtype.size_of(), "contiguous")?, Layout::contiguous(&self.layout.shape), self.dtype);
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
            enc.set_input(0, Some(&self.buf), self.layout.offset * self.dtype.size_of());
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

    /// Read the elements back to the host, walking the layout.
    pub fn to_vec<T: Copy>(&self) -> Result<Vec<T>> {
        if self.dtype.size_of() != std::mem::size_of::<T>() {
            return Err(err("to_vec: element size mismatch"));
        }
        self.eval()?;
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
                out.push(std::ptr::read_unaligned(
                    base.add(off * esz) as *const T
                ));
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

// ───────────────────────── native ops (stage 3) ─────────────────────────

/// MLX `backend/metal/utils.h:75` `get_work_per_thread`.
fn work_per_thread(dt: Dtype, size: usize) -> usize {
    const WPT_THRESHOLD: usize = 1 << 16;
    if size < WPT_THRESHOLD {
        1
    } else {
        (8 / dt.size_of()).max(1)
    }
}

/// `template [[host_name(..)]] [[kernel]] decltype(fn<..>) fn<..>;`
fn tmpl(name: &str, func: &str, args: &[String]) -> String {
    crate::mlx_rt::builtin_template_def(name, func, args)
}

impl Array {
    /// MLX's built-in unary kernel (`unary.cpp:unary_op_gpu`, contiguous path),
    /// bit-exact with `mlx_rt::unary_op`. `op` is the MLX identifier, e.g.
    /// `"Sigmoid"`, `"Exp"`, `"Log"`.
    pub fn unary(&self, op: &str) -> Result<Self> {
        let x = self.contiguous()?;
        let size = x.size();
        if size == 0 {
            return Err(err("unary: empty input"));
        }
        let in_t = self.dtype.metal_name();
        let ty = self.dtype.mlx_name();
        let wpt = work_per_thread(self.dtype, size);
        let mut kernel_name = String::from(if wpt > 1 { "vn" } else { "v" });
        kernel_name.push('_');
        kernel_name.push_str(op);
        kernel_name.push_str(ty);
        kernel_name.push_str(ty);
        let lib_name = kernel_name
            .split_once('_')
            .map(|(_, r)| r)
            .unwrap_or(&kernel_name)
            .to_string();
        let a = |s: &str| s.to_string();
        // Template args are the Metal element types; the kernel name carries the
        // MLX tags (mlx_rt's `type_string` vs `type_to_name`).
        let mut defs = tmpl(&format!("v_{lib_name}"), "unary_v", &[a(in_t), a(in_t), a(op), a("1")]);
        if wpt > 1 {
            defs.push_str(&tmpl(&format!("vn_{lib_name}"), "unary_v", &[a(in_t), a(in_t), a(op)]));
        }
        defs.push_str(&tmpl(&format!("v2_{lib_name}"), "unary_v2", &[a(in_t), a(in_t), a(op)]));
        defs.push_str(&tmpl(
            &format!("gn1_{lib_name}"),
            "unary_g",
            &[a(in_t), a(in_t), a(op), a("1"), a("int")],
        ));
        defs.push_str(&tmpl(
            &format!("gn4large_{lib_name}"),
            "unary_g",
            &[a(in_t), a(in_t), a(op), a("4")],
        ));
        let source = format!(
            "{}{}{}{}",
            crate::mlx_rt::MLX_UTILS_PREAMBLE,
            crate::mlx_rt::MLX_UNARY_OPS_PREAMBLE,
            crate::mlx_rt::MLX_UNARY_PREAMBLE,
            defs
        );
        // MLX builtin kernels are compiled without a language version (matching
        // `mlx_rt`'s `compile_builtin`); setting one changes the codegen.
        let pipe = self
            .rt
            .compile_with(&source, &kernel_name, crate::runtime::Math::SafeNoLang)?;

        let out = Self::wrap(Arc::clone(&self.rt), self.rt.buffer(size * self.dtype.size_of(), "unary_out")?, Layout::contiguous(&x.layout.shape), self.dtype);
        {
            let guard = self.rt.commands.encoder()?;
            let enc = guard.encoder();
            enc.set_pipeline(&pipe);
            enc.set_input(0, Some(&x.buf), x.layout.offset * self.dtype.size_of());
            enc.set_output(1, Some(&out.buf), 0);
            let n = size as i32;
            enc.set_bytes(2, &n);
            let nthreads = size.div_ceil(wpt);
            let tg = pipe
                .max_total_threads_per_threadgroup()
                .min(nthreads)
                .max(1);
            enc.dispatch_threads((nthreads, 1, 1), (tg, 1, 1));
        }
        Ok(out)
    }
}

impl Array {
    /// MLX `ops::softmax_axis` over the last axis (`Softmax::eval_gpu` block
    /// path, `softmax.cpp:16`). `precise` forces an f32 accumulator.
    pub fn softmax_last_axis(&self, precise: bool) -> Result<Self> {
        let x = self.contiguous()?;
        let shape = x.shape().to_vec();
        let axis_size = *shape.last().ok_or_else(|| err("softmax: scalar input"))?;
        if axis_size > 4096 {
            return Err(err("softmax: axis_size > 4096 (looped path) not implemented"));
        }
        let n_rows = x.size() / axis_size.max(1);
        let in_t = x.dtype.metal_name();
        let acc_t = if precise { "float" } else { in_t };
        let ty = x.dtype.mlx_name();
        let mut kernel_name = String::from("block_softmax_");
        if x.dtype != Dtype::F32 && precise {
            kernel_name.push_str("precise_");
        }
        kernel_name.push_str(ty);
        let lib_name = kernel_name
            .split_once('_')
            .map(|(_, r)| r)
            .unwrap_or(&kernel_name)
            .to_string();
        let mut defs = tmpl(
            &format!("block_{lib_name}"),
            "softmax_single_row",
            &[in_t.to_string(), acc_t.to_string()],
        );
        defs.push_str(&tmpl(
            &format!("looped_{lib_name}"),
            "softmax_looped",
            &[in_t.to_string(), acc_t.to_string()],
        ));
        let source = format!(
            "{}{}{}",
            crate::mlx_rt::MLX_UTILS_PREAMBLE,
            crate::mlx_rt::MLX_SOFTMAX_PREAMBLE,
            defs
        );
        let pipe = self.rt.compile(&source, &kernel_name)?;

        let out = Self::wrap(Arc::clone(&self.rt), self.rt.buffer(x.size() * self.dtype.size_of(), "softmax_out")?, Layout::contiguous(&shape), self.dtype);
        {
            let guard = self.rt.commands.encoder()?;
            let enc = guard.encoder();
            enc.set_pipeline(&pipe);
            enc.set_input(0, Some(&x.buf), x.layout.offset * self.dtype.size_of());
            enc.set_output(1, Some(&out.buf), 0);
            let asize = axis_size as i32;
            enc.set_bytes(2, &asize);
            let tgs = 32 * (axis_size.div_ceil(4)).div_ceil(32);
            let n_threads = n_rows * tgs;
            enc.dispatch_threads((n_threads, 1, 1), (tgs, 1, 1));
        }
        Ok(out)
    }

    /// MLX `ops::sum_axis`/`max`/`min`/`mean` on any axis: move it last,
    /// `row_reduce_simple`, move the result back (`mlx_rt::reduce_axis`).
    pub fn reduce_axis(&self, axis: i32, op: &str) -> Result<Self> {
        let rank = self.rank();
        if rank == 0 {
            return Err(err("reduce_axis: scalar input"));
        }
        let ax = if axis < 0 {
            (rank as i32 + axis) as usize
        } else {
            axis as usize
        };
        if ax >= rank {
            return Err(err("reduce_axis: axis out of range"));
        }
        if ax == rank - 1 {
            return self.reduce_last_axis(op);
        }
        let perm: Vec<usize> = (0..rank)
            .map(|i| if i == ax { rank - 1 } else if i == rank - 1 { ax } else { i })
            .collect();
        let t = self.permute(&perm)?.contiguous()?;
        let r = t.reduce_last_axis(op)?;
        // The reduced axis is now last; move it back to `ax`.
        let back: Vec<usize> = (0..rank - 1)
            .map(|i| if i == ax { rank - 2 } else if i == rank - 2 { ax } else { i })
            .collect();
        r.permute(&back)?.contiguous()
    }
}

/// MLX `backend/metal/utils.h` `get_2d_grid_dims`.
fn grid_2d(shape: &[usize]) -> (usize, usize) {
    let strides: Vec<i64> = {
        let mut v = vec![1i64; shape.len()];
        let mut acc = 1i64;
        for i in (0..shape.len()).rev() {
            v[i] = acc;
            acc *= shape[i] as i64;
        }
        v
    };
    let (mut gx, mut gy) = (1usize, 1usize);
    for (i, &d) in shape.iter().enumerate() {
        if strides[i] == 0 {
            continue;
        }
        if (gx as u64) * (d as u64) < u32::MAX as u64 {
            gx *= d;
        } else {
            gy *= d;
        }
    }
    if gy > gx {
        std::mem::swap(&mut gx, &mut gy);
    }
    (gx, gy)
}

/// MLX's `tg_from_row_size` thread-group policy for the row-reduce kernel.
fn tg_from_row_size(row_size: usize) -> usize {
    if row_size <= 512 {
        32
    } else if row_size <= 1024 {
        128
    } else {
        ((row_size.div_ceil(4) + 31) / 32 * 32).min(1024)
    }
}

impl Array {
    /// MLX's `row_reduce_simple` over the last axis. `op` is `"sum"`, `"max"` or
    /// `"min"`; `"mean"` rides the sum kernel and then multiplies by the f32
    /// reciprocal cast to the array dtype (`ops.cpp:2340`), exactly as
    /// `mlx_rt::reduce_last_axis` does.
    pub fn reduce_last_axis(&self, op: &str) -> Result<Self> {
        if op == "mean" {
            let n = *self
                .shape()
                .last()
                .ok_or_else(|| err("reduce: scalar input"))? as f32;
            let sum = self.reduce_last_axis("sum")?;
            let norm = Self::scalar_of(&self.rt, 1.0f32 / n, self.dtype)?;
            return sum.binary(&norm, "bmul");
        }
        let x = self.contiguous()?;
        let shape = x.shape();
        let row_size = *shape.last().ok_or_else(|| err("reduce: scalar input"))?;
        if row_size == 0 {
            return Err(err("reduce: empty axis"));
        }
        let out_dims = shape[..shape.len() - 1].to_vec();
        let in_t = x.dtype.metal_name();
        let ty = x.dtype.mlx_name();
        let mut c = op.chars();
        let op_type = match c.next() {
            Some(f) => f.to_uppercase().collect::<String>() + c.as_str(),
            None => return Err(err("reduce: empty op")),
        };
        let op_t = format!("{op_type}<{in_t}>");
        let kernel_name = format!("row_reduce_simple_{op}{ty}");
        let def = tmpl(
            &kernel_name,
            "row_reduce_simple",
            &[in_t.to_string(), in_t.to_string(), op_t, "size_t".to_string()],
        );
        let source = format!(
            "{}{}{}{}",
            crate::mlx_rt::MLX_UTILS_PREAMBLE,
            crate::mlx_rt::MLX_REDUCE_UTILS_PREAMBLE,
            crate::mlx_rt::MLX_REDUCE_PREAMBLE,
            def
        );
        let pipe = self.rt.compile(&source, &kernel_name)?;

        let out_count: usize = out_dims.iter().product::<usize>().max(1);
        let out = Self::wrap(Arc::clone(&self.rt), self
                .rt
                .buffer(out_count * self.dtype.size_of(), "reduce_out")?, Layout::contiguous(&out_dims), self.dtype);
        let (gx, gy) = grid_2d(&out_dims);
        let gw = gx.div_ceil(4);
        {
            let guard = self.rt.commands.encoder()?;
            let enc = guard.encoder();
            enc.set_pipeline(&pipe);
            enc.set_input(0, Some(&x.buf), x.layout.offset * self.dtype.size_of());
            enc.set_output(1, Some(&out.buf), 0);
            enc.set_bytes(2, &row_size);
            let os = out_count as i64;
            enc.set_bytes(3, &os);
            let tgs = tg_from_row_size(row_size)
                .min(pipe.max_total_threads_per_threadgroup())
                .max(1);
            enc.dispatch_threads((tgs, gw, gy), (tgs, 1, 1));
        }
        Ok(out)
    }

    /// A 0-d array of `dtype` holding `v` (rounded through the dtype).
    pub fn scalar_of(rt: &Arc<MetalRuntime>, v: f32, dtype: Dtype) -> Result<Self> {
        use half::{bf16, f16};
        match dtype {
            Dtype::F32 => Ok(Self::wrap(Arc::clone(rt), {
                let b = rt.buffer(4, "scalar")?;
                unsafe { *(b.contents() as *mut f32) = v };
                b
            }, Layout::contiguous(&[]), dtype)),
            Dtype::F16 => Ok(Self::wrap(Arc::clone(rt), {
                let b = rt.buffer(2, "scalar")?;
                unsafe { *(b.contents() as *mut f16) = f16::from_f32(v) };
                b
            }, Layout::contiguous(&[]), dtype)),
            Dtype::BF16 => Ok(Self::wrap(Arc::clone(rt), {
                let b = rt.buffer(2, "scalar")?;
                unsafe { *(b.contents() as *mut bf16) = bf16::from_f32(v) };
                b
            }, Layout::contiguous(&[]), dtype)),
            other => Err(err(format!("scalar_of: unsupported {other:?}"))),
        }
    }

    /// A scalar array of an integer/bool dtype (mlx `Array::from_int`/`from_bool`).
    pub fn scalar_raw(rt: &Arc<MetalRuntime>, v: i64, dtype: Dtype) -> Result<Self> {
        let bytes: [u8; 8] = match dtype {
            Dtype::Bool | Dtype::Uint8 | Dtype::Int8 => [v as u8, 0, 0, 0, 0, 0, 0, 0],
            Dtype::Uint16 | Dtype::Int16 => {
                let b = (v as u16).to_ne_bytes();
                [b[0], b[1], 0, 0, 0, 0, 0, 0]
            }
            Dtype::Uint32 | Dtype::Int32 => {
                let b = (v as u32).to_ne_bytes();
                [b[0], b[1], b[2], b[3], 0, 0, 0, 0]
            }
            Dtype::Int64 => v.to_ne_bytes(),
            other => return Err(err(format!("scalar_raw: unsupported {other:?}"))),
        };
        let b = rt.buffer(dtype.size_of(), "scalar")?;
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), b.contents() as *mut u8, dtype.size_of());
        }
        Ok(Self::wrap(Arc::clone(rt), b, Layout::contiguous(&[]), dtype))
    }
}

/// The `Dtype` a Rust scalar type corresponds to (for `item` casting).
pub fn dtype_of<T: 'static>() -> Option<Dtype> {
    use std::any::TypeId;
    let t = TypeId::of::<T>();
    if t == TypeId::of::<f32>() {
        Some(Dtype::Float32)
    } else if t == TypeId::of::<f64>() {
        Some(Dtype::Float64)
    } else if t == TypeId::of::<i32>() {
        Some(Dtype::Int32)
    } else if t == TypeId::of::<i64>() {
        Some(Dtype::Int64)
    } else if t == TypeId::of::<u32>() {
        Some(Dtype::Uint32)
    } else if t == TypeId::of::<u8>() {
        Some(Dtype::Uint8)
    } else if t == TypeId::of::<i8>() {
        Some(Dtype::Int8)
    } else if t == TypeId::of::<i16>() {
        Some(Dtype::Int16)
    } else if t == TypeId::of::<u16>() {
        Some(Dtype::Uint16)
    } else if t == TypeId::of::<half::f16>() {
        Some(Dtype::Float16)
    } else if t == TypeId::of::<half::bf16>() {
        Some(Dtype::Bfloat16)
    } else {
        None
    }
}

/// Right-aligned broadcast of two shapes (NumPy semantics).
fn broadcast_shape(a: &[usize], b: &[usize]) -> Result<Vec<usize>> {
    let n = a.len().max(b.len());
    let mut out = vec![0usize; n];
    for i in 0..n {
        let da = if i + a.len() >= n { a[i + a.len() - n] } else { 1 };
        let db = if i + b.len() >= n { b[i + b.len() - n] } else { 1 };
        out[i] = if da == db {
            da
        } else if da == 1 {
            db
        } else if db == 1 {
            da
        } else {
            return Err(err(format!("broadcast: {a:?} vs {b:?}")));
        };
    }
    Ok(out)
}

impl Array {
    /// Broadcast to `shape`, right-aligning ranks first.
    pub fn broadcast_to(&self, shape: &[usize]) -> Result<Self> {
        if shape.len() < self.rank() {
            return Err(err("broadcast_to: fewer dims than the source"));
        }
        let extra = shape.len() - self.rank();
        let mut v = self.clone();
        for _ in 0..extra {
            v = v.expand_dims(0)?;
        }
        v.broadcast(shape)
    }

    /// elementwise binary op, via its vendored kernel. `op` is the
    /// identifier without the dtype suffix: `badd`, `bsub`, `bmul`, `bdiv`,
    /// `bminimum`, `bmaximum`.
    ///
    /// Only the `_strided` variant is dispatched. the API picks between the
    /// contiguous / `_cs` / `_sc` / scalar variants purely as an indexer choice
    /// — the elementwise op and the operands per element are identical — so a
    /// strided dispatch with the broadcast strides is bit-identical, just
    /// without the fast paths.
    pub fn binary(&self, other: &Array, op: &str) -> Result<Self> {
        if self.dtype != other.dtype {
            return Err(err("binary: dtype mismatch"));
        }
        let shape = broadcast_shape(self.shape(), other.shape())?;
        let a = self.broadcast_to(&shape)?;
        let b = other.broadcast_to(&shape)?;
        let ty = match self.dtype {
            Dtype::F32 => "f32",
            Dtype::F16 => "f16",
            Dtype::BF16 => "bf16",
            Dtype::U8 => "u8",
            Dtype::U32 => "u32",
            Dtype::I64 => "i64",
            other => return Err(err(format!("binary: no kernel for {other:?}"))),
        };
        let name = format!("{op}_{ty}_strided");
        let pipe = self.rt.compile_with(ELEMENTWISE_BINARY, &name, crate::runtime::Math::Fast)?;

        let n: usize = shape.iter().product();
        let out = Self::wrap(Arc::clone(&self.rt), self.rt.buffer(n * self.dtype.size_of(), "binary_out")?, Layout::contiguous(&shape), self.dtype);
        let dims: Vec<u64> = shape.iter().map(|&d| d as u64).collect();
        let ls: Vec<u64> = a.layout.strides.iter().map(|&s| s as u64).collect();
        let rs: Vec<u64> = b.layout.strides.iter().map(|&s| s as u64).collect();
        {
            let guard = self.rt.commands.encoder()?;
            let enc = guard.encoder();
            enc.set_pipeline(&pipe);
            enc.set_bytes(0, &n);
            let nd = shape.len();
            enc.set_bytes(1, &nd);
            enc.set_bytes_directly(2, dims.len() * 8, dims.as_ptr().cast());
            enc.set_bytes_directly(3, ls.len() * 8, ls.as_ptr().cast());
            enc.set_bytes_directly(4, rs.len() * 8, rs.as_ptr().cast());
            enc.set_input(5, Some(&a.buf), a.layout.offset * self.dtype.size_of());
            enc.set_input(6, Some(&b.buf), b.layout.offset * self.dtype.size_of());
            enc.set_output(7, Some(&out.buf), 0);
            // `get_tile_size` + `linear_split`.
            let tile = 1.max(8 / self.dtype.size_of());
            let tiles = n.div_ceil(tile);
            let width = pipe.max_total_threads_per_threadgroup().min(tiles).max(1);
            let count = tiles.div_ceil(width);
            enc.dispatch_groups((count, 1, 1), (width, 1, 1));
        }
        Ok(out)
    }
}

/// binary kernel, vendored verbatim (see NOTICE).
const ELEMENTWISE_BINARY: &str = include_str!("kernels/common/elementwise_binary.metal");
/// cast kernel, vendored verbatim (see NOTICE).
const CAST_KERNEL: &str = include_str!("kernels/common/cast.metal");

/// ternary (`where`) kernel, vendored verbatim (see NOTICE).
const SELECT_KERNEL: &str = include_str!("kernels/common/select.metal");

/// indexing kernels (gather / index_select / scatter), vendored
/// verbatim (see NOTICE).
const INDEXING_KERNEL: &str = include_str!("kernels/common/indexing.metal");

impl Array {
    /// The Metal tag for an index/value dtype in kernel names.
    fn kernel_tag(d: Dtype) -> Option<&'static str> {
        Some(match d {
            Dtype::F32 => "f32",
            Dtype::F16 => "f16",
            Dtype::BF16 => "bf16",
            Dtype::I64 => "i64",
            Dtype::U32 => "u32",
            Dtype::U8 | Dtype::Bool => "u8",
            _ => return None,
        })
    }

    /// Stack along a new `axis` (unsqueeze-then-cat).
    pub fn stack(arrays: &[Array], axis: usize) -> Result<Self> {
        let uns: Vec<Array> = arrays
            .iter()
            .map(|a| a.expand_dims(axis))
            .collect::<Result<_>>()?;
        Self::cat(&uns, axis)
    }

    /// np.repeat: each element `n` times along `axis` (MLX's `repeat_axis`
    /// semantics — *not* `repeat`, which tiles).
    pub fn repeat_axis(&self, n: usize, axis: usize) -> Result<Self> {
        let mut dims = self.shape().to_vec();
        let d = dims[axis];
        dims.insert(axis + 1, n);
        let expanded = self.expand_dims(axis + 1)?.broadcast(&dims)?;
        let mut out_dims = dims.clone();
        out_dims[axis] = d * n;
        out_dims.remove(axis + 1);
        // The broadcast is non-contiguous, so it materialises before reshaping.
        expanded.copied()?.reshape(&out_dims)
    }

    /// np.tile (same-rank `reps` only).
    pub fn tile(&self, reps: &[usize]) -> Result<Self> {
        let dims = self.shape().to_vec();
        if reps.len() != dims.len() {
            return Err(err("tile: reps rank must match input rank"));
        }
        let mut exp = Vec::with_capacity(dims.len() * 2);
        for (i, &d) in dims.iter().enumerate() {
            exp.push(reps[i]);
            exp.push(d);
        }
        let mut t = self.clone();
        for i in 0..dims.len() {
            t = t.expand_dims(2 * i)?;
        }
        let t = t.broadcast(&exp)?;
        let out_dims: Vec<usize> = dims.iter().zip(reps).map(|(d, r)| d * r).collect();
        t.copied()?.reshape(&out_dims)
    }

    /// `index_select` along `axis`: `ids` is 1-D and replaces that
    /// axis with the gathered rows.
    pub fn index_select(&self, ids: &Array, axis: usize) -> Result<Self> {
        let x = self.contiguous()?;
        let ids = ids.contiguous()?;
        if ids.rank() != 1 {
            return Err(err("index_select: ids must be 1-D"));
        }
        let (Some(it), Some(tt)) = (Self::kernel_tag(ids.dtype), Self::kernel_tag(x.dtype)) else {
            return Err(err("index_select: unsupported dtype"));
        };
        let name = format!("is_{it}_{tt}");
        let pipe = self
            .rt
            .compile_with(INDEXING_KERNEL, &name, crate::runtime::Math::Fast)?;

        let shape = x.shape().to_vec();
        let left_size: usize = shape[..axis].iter().product();
        let right_size: usize = shape[axis + 1..].iter().product();
        let src_dim_size = shape[axis];
        let ids_size = ids.size();
        let dst_el = ids_size * left_size * right_size;
        let mut out_shape = shape.clone();
        out_shape[axis] = ids_size;
        let out = Self::wrap(Arc::clone(&self.rt), self.rt.buffer(dst_el * x.dtype.size_of(), "index_select_out")?, Layout::contiguous(&out_shape), x.dtype);
        // The input is contiguous, so the kernel's strided path is unused; the
        // dims/strides buffers just have to be well-formed.
        let dims: Vec<u64> = shape.iter().map(|&d| d as u64).collect();
        let strides: Vec<u64> = x.layout.strides.iter().map(|&s| s as u64).collect();
        {
            let guard = self.rt.commands.encoder()?;
            let enc = guard.encoder();
            enc.set_pipeline(&pipe);
            enc.set_bytes(0, &dst_el);
            enc.set_bytes(1, &left_size);
            enc.set_bytes(2, &src_dim_size);
            enc.set_bytes(3, &right_size);
            enc.set_bytes(4, &ids_size);
            let contiguous = true;
            enc.set_bytes(5, &contiguous);
            enc.set_bytes_directly(6, dims.len() * 8, dims.as_ptr().cast());
            enc.set_bytes_directly(7, strides.len() * 8, strides.as_ptr().cast());
            enc.set_input(8, Some(&x.buf), x.layout.offset * x.dtype.size_of());
            enc.set_input(9, Some(&ids.buf), ids.layout.offset * ids.dtype.size_of());
            enc.set_output(10, Some(&out.buf), 0);
            let width = pipe.max_total_threads_per_threadgroup().min(dst_el).max(1);
            let count = dst_el.div_ceil(width);
            enc.dispatch_groups((count, 1, 1), (width, 1, 1));
        }
        Ok(out)
    }

    /// Named unary wrappers (the MLX op identifiers `mlx_rt` uses).
    pub fn exp(&self) -> Result<Self> {
        self.unary("Exp")
    }
    pub fn log(&self) -> Result<Self> {
        self.unary("Log")
    }
    pub fn log1p(&self) -> Result<Self> {
        self.unary("Log1p")
    }
    pub fn sin(&self) -> Result<Self> {
        self.unary("Sin")
    }
    pub fn cos(&self) -> Result<Self> {
        self.unary("Cos")
    }
    pub fn sqrt(&self) -> Result<Self> {
        self.unary("Sqrt")
    }
    pub fn rsqrt(&self) -> Result<Self> {
        self.unary("Rsqrt")
    }
    pub fn abs(&self) -> Result<Self> {
        self.unary("Abs")
    }
    pub fn sign(&self) -> Result<Self> {
        self.unary("Sign")
    }
    pub fn neg(&self) -> Result<Self> {
        self.unary("Negative")
    }
    pub fn sigmoid(&self) -> Result<Self> {
        self.unary("Sigmoid")
    }
    pub fn square(&self) -> Result<Self> {
        self.unary("Square")
    }
    /// MLX `nn.silu` is `x * sigmoid(x)` (`mlx_rt::silu`).
    pub fn silu(&self) -> Result<Self> {
        let s = self.sigmoid()?;
        self.binary(&s, "bmul")
    }

    /// Named binary wrappers (kernel identifiers).
    pub fn add(&self, other: &Array) -> Result<Self> {
        self.binary(other, "badd")
    }
    pub fn subtract(&self, other: &Array) -> Result<Self> {
        self.binary(other, "bsub")
    }
    pub fn multiply(&self, other: &Array) -> Result<Self> {
        self.binary(other, "bmul")
    }
    pub fn divide(&self, other: &Array) -> Result<Self> {
        self.binary(other, "bdiv")
    }
    pub fn maximum(&self, other: &Array) -> Result<Self> {
        self.binary(other, "bmaximum")
    }
    pub fn minimum(&self, other: &Array) -> Result<Self> {
        self.binary(other, "bminimum")
    }

    /// boolean comparisons: the kernel is `{op}_{ty}_strided` with the
    /// operand dtype, and the output is `u8`.
    pub fn cmp(&self, other: &Array, op: &str) -> Result<Self> {
        if self.dtype != other.dtype {
            return Err(err("cmp: dtype mismatch"));
        }
        let shape = broadcast_shape(self.shape(), other.shape())?;
        let a = self.broadcast_to(&shape)?;
        let b = other.broadcast_to(&shape)?;
        let ty = Self::kernel_tag(self.dtype).ok_or_else(|| err("cmp: unsupported dtype"))?;
        let name = format!("{op}_{ty}_strided");
        let pipe = self
            .rt
            .compile_with(ELEMENTWISE_BINARY, &name, crate::runtime::Math::Fast)?;
        let n: usize = shape.iter().product();
        let out = Self::wrap(Arc::clone(&self.rt), self.rt.buffer(n, "cmp_out")?, Layout::contiguous(&shape), Dtype::U8);
        let dims: Vec<u64> = shape.iter().map(|&d| d as u64).collect();
        let ls: Vec<u64> = a.layout.strides.iter().map(|&s| s as u64).collect();
        let rs: Vec<u64> = b.layout.strides.iter().map(|&s| s as u64).collect();
        {
            let guard = self.rt.commands.encoder()?;
            let enc = guard.encoder();
            enc.set_pipeline(&pipe);
            enc.set_bytes(0, &n);
            let nd = shape.len();
            enc.set_bytes(1, &nd);
            enc.set_bytes_directly(2, dims.len() * 8, dims.as_ptr().cast());
            enc.set_bytes_directly(3, ls.len() * 8, ls.as_ptr().cast());
            enc.set_bytes_directly(4, rs.len() * 8, rs.as_ptr().cast());
            enc.set_input(5, Some(&a.buf), a.layout.offset * self.dtype.size_of());
            enc.set_input(6, Some(&b.buf), b.layout.offset * self.dtype.size_of());
            enc.set_output(7, Some(&out.buf), 0);
            let tile = 1.max(8 / self.dtype.size_of());
            let tiles = n.div_ceil(tile);
            let width = pipe.max_total_threads_per_threadgroup().min(tiles).max(1);
            let count = tiles.div_ceil(width);
            enc.dispatch_groups((count, 1, 1), (width, 1, 1));
        }
        Ok(out)
    }
    pub fn ge(&self, other: &Array) -> Result<Self> {
        self.cmp(other, "ge")
    }
    pub fn le(&self, other: &Array) -> Result<Self> {
        self.cmp(other, "le")
    }
    pub fn lt(&self, other: &Array) -> Result<Self> {
        self.cmp(other, "lt")
    }
    pub fn gt(&self, other: &Array) -> Result<Self> {
        self.cmp(other, "gt")
    }

    /// Whole-array reductions (flatten, then the MLX row-reduce) — the
    /// `sum_all`/`mean_all`/`max_all`/`min_all`.
    pub fn sum_all(&self) -> Result<Self> {
        self.flatten_all()?.reduce_last_axis("sum")
    }
    pub fn mean_all(&self) -> Result<Self> {
        self.flatten_all()?.reduce_last_axis("mean")
    }
    pub fn max_all(&self) -> Result<Self> {
        self.flatten_all()?.reduce_last_axis("max")
    }
    pub fn min_all(&self) -> Result<Self> {
        self.flatten_all()?.reduce_last_axis("min")
    }

    /// Axis reductions — `sum`/`mean`/`max`/`min` (MLX row-reduce).
    pub fn sum(&self, axis: usize) -> Result<Self> {
        self.reduce_axis(axis as i32, "sum")
    }
    pub fn mean(&self, axis: usize) -> Result<Self> {
        self.reduce_axis(axis as i32, "mean")
    }
    pub fn max(&self, axis: usize) -> Result<Self> {
        self.reduce_axis(axis as i32, "max")
    }
    pub fn min(&self, axis: usize) -> Result<Self> {
        self.reduce_axis(axis as i32, "min")
    }

    /// MLX `logaddexp` = `max(a,b) + log(exp(min(a,b) - max(a,b)) + 1)` — the
    /// shim's own composition (`affine(1,1)` then `log`, *not* `log1p`, which
    /// rounds differently).
    pub fn logaddexp(&self, other: &Array) -> Result<Self> {
        let a = self.maximum(other)?;
        let b = self.minimum(other)?;
        let one = Self::scalar_of(&self.rt, 1.0, self.dtype)?;
        let d = b.subtract(&a)?.exp()?.add(&one)?.log()?;
        a.add(&d)
    }

    /// Host readback as a typed slice. Cached on the array (which is immutable
    /// after construction), so repeated reads are free.
    pub fn as_slice<T: Copy>(&self) -> &[T] {
        let bytes = self.host.get_or_init(|| {
            let v = self.to_vec::<T>().expect("as_slice");
            unsafe {
                std::slice::from_raw_parts(v.as_ptr().cast::<u8>(), std::mem::size_of_val(v.as_slice()))
            }
            .to_vec()
        });
        let n = bytes.len() / std::mem::size_of::<T>();
        unsafe { std::slice::from_raw_parts(bytes.as_ptr().cast::<T>(), n) }
    }

    /// First element as `T`, casting the stored dtype first (
    /// `to_scalar`/`item` semantics — `item::<f32>()` on a bf16 array casts).
    pub fn item<T: Copy + 'static>(&self) -> T {
        match dtype_of::<T>() {
            Some(dt) if dt != self.dtype => {
                self.to_dtype(dt).expect("item: cast").as_slice::<T>()[0]
            }
            _ => self.as_slice::<T>()[0],
        }
    }

    /// Split into `num_splits` equal parts along `axis` (`chunk`).
    pub fn split_equal(&self, num_splits: usize, axis: usize) -> Result<Vec<Self>> {
        if num_splits == 0 || self.dim(axis) % num_splits != 0 {
            return Err(err("split_equal: axis not divisible"));
        }
        let step = self.dim(axis) / num_splits;
        (0..num_splits)
            .map(|i| self.narrow(axis, i * step, step))
            .collect()
    }

    /// Logical NOT for a 0/1 tensor: `affine(-1, 1)`.
    pub fn logical_not(&self) -> Result<Self> {
        self.affine(-1.0, 1.0)
    }

    /// Logical OR for 0/1 tensors (`maximum`).
    pub fn logical_or(&self, other: &Array) -> Result<Self> {
        self.maximum(other)
    }

    /// Logical AND for 0/1 tensors (`mul`).
    pub fn logical_and(&self, other: &Array) -> Result<Self> {
        self.multiply(other)
    }

    /// `x != x` (`is_nan`).
    pub fn is_nan(&self) -> Result<Self> {
        self.cmp(self, "ne")
    }

    /// numpy-style floor division (divide/floor in f32, cast back).
    pub fn floor_divide(&self, other: &Array) -> Result<Self> {
        if self.dtype != other.dtype {
            return Err(err("floor_divide: dtype mismatch"));
        }
        let dt = self.dtype;
        let a = self.cast(Dtype::F32)?;
        let b = other.cast(Dtype::F32)?;
        let q = a.divide(&b)?.unary("Floor")?;
        q.cast(dt)
    }

    /// `x * a + b` (`affine`).
    pub fn affine(&self, a: f32, b: f32) -> Result<Self> {
        let sa = Self::scalar_of(&self.rt, a, self.dtype)?;
        let sb = Self::scalar_of(&self.rt, b, self.dtype)?;
        self.multiply(&sa)?.add(&sb)
    }

    /// Eager: there is no autograd graph to detach from.
    pub fn detach(&self) -> Result<Self> {
        Ok(self.clone())
    }

    /// MLX `argmax_axis` (`ArgReduce::eval_gpu`, `arg_reduce_general`). Returns
    /// u32 indices along `axis`. Bit-identical to `Tensor::argmax` on
    /// the golden's geometry (verified against it), so re-routing is safe.
    pub fn argmax_axis(&self, axis: i32) -> Result<Self> {
        self.arg_reduce(axis, "ArgMax", "argmax")
    }

    /// MLX `argmin_axis`.
    pub fn argmin_axis(&self, axis: i32) -> Result<Self> {
        self.arg_reduce(axis, "ArgMin", "argmin")
    }

    fn arg_reduce(&self, axis: i32, op_struct: &str, prefix: &str) -> Result<Self> {
        let x = self.contiguous()?;
        let dims = x.shape().to_vec();
        let rank = dims.len();
        if rank == 0 {
            return Err(err("arg_reduce: scalar input"));
        }
        let ax = if axis < 0 {
            (rank as i32 + axis) as usize
        } else {
            axis as usize
        };
        if ax >= rank {
            return Err(err("arg_reduce: axis out of range"));
        }
        let ty = x.dtype.mlx_name();
        let t = x.dtype.metal_name();
        let kernel_name = format!("{prefix}_{ty}");
        let inst = format!(
            "\ntemplate [[host_name(\"{kernel_name}\")]] [[kernel]] decltype(arg_reduce_general<{t}, {op_struct}<{t}>>) arg_reduce_general<{t}, {op_struct}<{t}>>;\n"
        );
        let source = format!(
            "{}{}{}",
            crate::mlx_rt::MLX_UTILS_PREAMBLE,
            crate::mlx_rt::MLX_ARG_REDUCE_SOURCE,
            inst
        );
        let pipe = self.rt.compile(&source, &kernel_name)?;

        let strides = x.layout.strides.clone();
        let mut shape: Vec<i32> = Vec::new();
        let mut in_strides: Vec<i64> = Vec::new();
        let mut axis_stride = 1i64;
        let mut axis_size = 1usize;
        for i in 0..rank {
            if i == ax {
                axis_stride = strides[i] as i64;
                axis_size = dims[i];
                continue;
            }
            shape.push(dims[i] as i32);
            in_strides.push(strides[i] as i64);
        }
        let out_dims: Vec<usize> = dims
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != ax)
            .map(|(_, d)| *d)
            .collect();
        let mut out_strides = vec![1i64; out_dims.len()];
        let mut acc = 1i64;
        for i in (0..out_dims.len()).rev() {
            out_strides[i] = acc;
            acc *= out_dims[i] as i64;
        }
        let ndim = out_dims.len();
        if ndim > 2 {
            return Err(err("arg_reduce: out rank > 2 not implemented"));
        }
        let out_count: usize = out_dims.iter().product::<usize>().max(1);
        let out = Self::wrap(Arc::clone(&self.rt), self.rt.buffer(out_count * 4, "arg_out")?, Layout::contiguous(&out_dims), Dtype::U32);
        let (gd_w, gd_h) = match ndim {
            0 => (1usize, 1usize),
            1 => (out_dims[0], 1usize),
            _ => (out_dims[1], out_dims[0]),
        };
        {
            let guard = self.rt.commands.encoder()?;
            let enc = guard.encoder();
            enc.set_pipeline(&pipe);
            enc.set_input(0, Some(&x.buf), x.layout.offset * x.dtype.size_of());
            enc.set_output(1, Some(&out.buf), 0);
            if ndim == 0 {
                let zero_i: i32 = 0;
                let zero_l: i64 = 0;
                enc.set_bytes(2, &zero_i);
                enc.set_bytes(3, &zero_l);
                enc.set_bytes(4, &zero_l);
            } else {
                enc.set_bytes_directly(2, shape.len() * 4, shape.as_ptr().cast());
                enc.set_bytes_directly(3, in_strides.len() * 8, in_strides.as_ptr().cast());
                enc.set_bytes_directly(4, out_strides.len() * 8, out_strides.as_ptr().cast());
            }
            enc.set_bytes(5, &ndim);
            enc.set_bytes(6, &axis_stride);
            enc.set_bytes(7, &axis_size);
            let tgs = axis_size
                .div_ceil(4)
                .min(pipe.max_total_threads_per_threadgroup())
                .div_ceil(32)
                * 32;
            enc.dispatch_threads((tgs, gd_w, gd_h), (tgs, 1, 1));
        }
        Ok(out)
    }

    /// MLX `max`/`min` over the last axis (`max_all`/`max` route to
    /// own reduce kernel; this is the MLX row-reduce instead, verified
    /// bit-identical to `max_all`).
    pub fn max_last_axis(&self) -> Result<Self> {
        self.reduce_last_axis("max")
    }

    /// Concatenate along `axis`. All inputs must share a rank and dtype, and
    /// match on every dim but `axis`.
    ///
    /// each input is copied into the output with a blit; the buffers here
    /// are shared-storage, so the copy is host-side (a memcpy per outer block).
    /// A blit/kernel path is a later optimization.
    pub fn cat(arrays: &[Array], axis: usize) -> Result<Self> {
        let first = arrays.first().ok_or_else(|| err("cat: no arrays"))?;
        let rank = first.rank();
        if axis >= rank {
            return Err(err("cat: axis out of range"));
        }
        let shape0 = first.shape().to_vec();
        for a in arrays {
            if a.rank() != rank || a.dtype != first.dtype {
                return Err(err("cat: rank/dtype mismatch"));
            }
            for (d, (&x, &y)) in shape0.iter().zip(a.shape().iter()).enumerate() {
                if d != axis && x != y {
                    return Err(err("cat: dim mismatch"));
                }
            }
        }
        let mut out_shape = shape0.clone();
        out_shape[axis] = arrays.iter().map(|a| a.dim(axis)).sum();
        let out = Self::zeros(&first.rt, &out_shape, first.dtype)?;
        // MLX `concatenate_gpu` (backend/metal/slicing.cpp): slice the output on
        // the concat axis, then dispatch one GeneralGeneral copy per input —
        // src and dst share the iteration shape but have their own strides. No
        // host readback; ordering is carried by the command encoder.
        let esz = first.dtype.size_of();
        let tag = first.dtype.metal_name();
        let mut off = 0usize;
        for a in arrays {
            let d = a.dim(axis);
            // The dst slice view (MLX's `out_slice` with
            // data_offset = strides[axis] * offset).
            let dst = out.narrow(axis, off, d)?;
            let dst_off_elems = dst.layout.offset;
            // Collapse runs of contiguous dims across BOTH stride sets (MLX's
            // `copy_gpu_inplace` maybe_collapse), then pick the specialised
            // nd1/nd2/nd3 gg kernel.
            let shape_i32: Vec<i32> = a.shape().iter().map(|&x| x as i32).collect();
            let src_str: Vec<i64> = a.layout.strides.iter().map(|&x| x as i64).collect();
            let dst_str: Vec<i64> = out.layout.strides.iter().map(|&x| x as i64).collect();
            let (cshape, cstrides) = crate::mlx_rt::collapse_contiguous_dims(
                &shape_i32,
                &[src_str, dst_str],
                i32::MAX as i64,
            );
            let ndim = cshape.len();
            if ndim == 0 || ndim > 3 {
                return Err(err(format!("cat: collapsed rank {ndim} unsupported")));
            }
            let src_s = &cstrides[0];
            let dst_s = &cstrides[1];
            let data_size: usize = cshape.iter().map(|&x| x as usize).product();
            let dim0 = cshape[ndim - 1] as usize;
            let dim1 = if ndim > 1 { cshape[ndim - 2] as usize } else { 1 };
            let rest = data_size / (dim0 * dim1).max(1);

            let name = format!("copy_gg_nd{ndim}_{tag}");
            let pipe = first.rt.compile(&copy_gg_source(), &name)?;
            {
                let guard = first.rt.commands.encoder()?;
                let enc = guard.encoder();
                enc.set_pipeline(&pipe);
                enc.set_input(0, Some(&a.buf), a.layout.offset * esz);
                enc.set_output(1, Some(&out.buf), dst_off_elems * esz);
                match ndim {
                    1 => {
                        enc.set_bytes(3, &(src_s[0] as i64));
                        enc.set_bytes(4, &(dst_s[0] as i64));
                    }
                    _ => {
                        enc.set_bytes_directly(3, ndim * 8, src_s.as_ptr().cast());
                        enc.set_bytes_directly(4, ndim * 8, dst_s.as_ptr().cast());
                    }
                }
                // MLX: grid (dim0, dim1, rest), block dims summing to 1024.
                let (gw, gh, gd) = crate::mlx_rt::get_block_dims(dim0, dim1, rest, 10);
                enc.dispatch_threads_3d(
                    (dim0.max(1), dim1.max(1), rest.max(1)),
                    (gw, gh, gd),
                );
            }
            off += d;
        }
        Ok(out)
    }

    /// Permute dimensions by an explicit order (alias of `permute`).
    pub fn transpose_axes(&self, perm: &[usize]) -> Result<Self> {
        self.permute(perm)
    }

    /// `transpose(a, b)`: swap two dims (a view).
    pub fn transpose(&self, a: usize, b: usize) -> Result<Self> {
        let rank = self.rank();
        if a >= rank || b >= rank {
            return Err(err("transpose: axis out of range"));
        }
        let perm: Vec<usize> = (0..rank)
            .map(|i| if i == a { b } else if i == b { a } else { i })
            .collect();
        self.permute(&perm)
    }

    /// Drop a size-1 axis (a view).
    pub fn squeeze(&self, axis: usize) -> Result<Self> {
        if axis >= self.rank() || self.layout.shape[axis] != 1 {
            return Err(err("squeeze: axis is not size 1"));
        }
        let mut shape = self.layout.shape.clone();
        let mut strides = self.layout.strides.clone();
        shape.remove(axis);
        strides.remove(axis);
        let mut out = self.clone();
        out.layout = Layout {
            shape,
            strides,
            offset: self.layout.offset,
        };
        Ok(out)
    }

    /// Flatten to 1-D (a view when contiguous, else a copy).
    pub fn flatten_all(&self) -> Result<Self> {
        let n = self.size();
        if self.layout.is_contiguous() {
            self.reshape(&[n])
        } else {
            self.copied()?.reshape(&[n])
        }
    }

    /// The first element as `T` (a scalar readback).
    pub fn item_cast<T: Copy + 'static>(&self) -> Result<T> {
        if self.size() != 1 {
            return Err(err("item_cast: not a single element"));
        }
        Ok(self.item::<T>())
    }

    /// mlx `take_axis`: `ids` (any shape) replaces `axis` in the output shape
    /// (`mlx_rt`'s flatten-index_select-reshape, which the shim uses).
    pub fn take_axis(&self, ids: &Array, axis: usize) -> Result<Self> {
        let flat = ids.flatten_all()?;
        let sel = self.index_select(&flat, axis)?;
        let mut shape: Vec<usize> = self.shape()[..axis].to_vec();
        shape.extend(ids.shape().iter().copied());
        shape.extend(self.shape()[axis + 1..].iter().copied());
        sel.reshape(&shape)
    }

    /// `gather` along `axis`: `out[.., i, ..] = self[.., ids[.., i, ..], ..]`.
    /// `ids` must share the rank and every dim but `axis`.
    pub fn gather(&self, ids: &Array, axis: usize) -> Result<Self> {
        let x = self.contiguous()?;
        if ids.rank() != x.rank() {
            return Err(err("gather: ids rank must match input rank"));
        }
        let shape = x.shape().to_vec();
        for (d, (&a, &b)) in shape.iter().zip(ids.shape().iter()).enumerate() {
            if d != axis && a != b {
                return Err(err(format!("gather: dim {d} mismatch {a} vs {b}")));
            }
        }
        let (Some(it), Some(tt)) = (Self::kernel_tag(ids.dtype), Self::kernel_tag(x.dtype)) else {
            return Err(err("gather: unsupported dtype"));
        };
        let name = format!("gather_{it}_{tt}");
        let pipe = self
            .rt
            .compile_with(INDEXING_KERNEL, &name, crate::runtime::Math::Fast)?;

        let left_size: usize = shape[..axis].iter().product();
        let right_size: usize = shape[axis + 1..].iter().product();
        let src_dim_size = shape[axis];
        let ids_size = ids.dim(axis);
        let dst_el = ids_size * left_size * right_size;
        let out = Self::wrap(Arc::clone(&self.rt), self.rt.buffer(dst_el * x.dtype.size_of(), "gather_out")?, Layout::contiguous(ids.shape()), x.dtype);
        let ids = ids.contiguous()?;
        {
            let guard = self.rt.commands.encoder()?;
            let enc = guard.encoder();
            enc.set_pipeline(&pipe);
            enc.set_bytes(0, &dst_el);
            enc.set_bytes(1, &left_size);
            enc.set_bytes(2, &src_dim_size);
            enc.set_bytes(3, &right_size);
            enc.set_bytes(4, &ids_size);
            enc.set_input(5, Some(&x.buf), x.layout.offset * x.dtype.size_of());
            enc.set_input(6, Some(&ids.buf), ids.layout.offset * ids.dtype.size_of());
            enc.set_output(7, Some(&out.buf), 0);
            let width = pipe.max_total_threads_per_threadgroup().min(dst_el).max(1);
            let count = dst_el.div_ceil(width);
            enc.dispatch_groups((count, 1, 1), (width, 1, 1));
        }
        Ok(out)
    }
}

impl Array {
    /// A contiguous array with every element set to `value` (rounded through
    /// the dtype). `full`/`zeros` are a host write, matching blit fill.
    pub fn full(
        rt: &Arc<MetalRuntime>,
        value: f32,
        shape: &[usize],
        dtype: Dtype,
    ) -> Result<Self> {
        use half::{bf16, f16};
        let out = Self::zeros(rt, shape, dtype)?;
        let p = out.buf.contents();
        match dtype {
            Dtype::F32 => unsafe { (0..out.size()).for_each(|i| *(p as *mut f32).add(i) = value) },
            Dtype::BF16 => {
                let v = bf16::from_f32(value);
                unsafe { (0..out.size()).for_each(|i| *(p as *mut bf16).add(i) = v) }
            }
            Dtype::F16 => {
                let v = f16::from_f32(value);
                unsafe { (0..out.size()).for_each(|i| *(p as *mut f16).add(i) = v) }
            }
            Dtype::U8 | Dtype::Bool => {
                let v = value as u8;
                unsafe { (0..out.size()).for_each(|i| *(p as *mut u8).add(i) = v) }
            }
            Dtype::U32 => {
                let v = value as u32;
                unsafe { (0..out.size()).for_each(|i| *(p as *mut u32).add(i) = v) }
            }
            Dtype::I64 => {
                let v = value as i64;
                unsafe { (0..out.size()).for_each(|i| *(p as *mut i64).add(i) = v) }
            }
            other => return Err(err(format!("full: unsupported {other:?}"))),
        }
        Ok(out)
    }

    /// `where_cond`: `cond` selects elementwise between `on_true` and
    /// `on_false`. All three must share a shape (no broadcasting).
    pub fn where_cond(&self, on_true: &Array, on_false: &Array) -> Result<Self> {
        if self.shape() != on_true.shape() || self.shape() != on_false.shape() {
            return Err(err("where_cond: shapes must match"));
        }
        if on_true.dtype != on_false.dtype {
            return Err(err("where_cond: value dtypes must match"));
        }
        let name = match (self.dtype, on_true.dtype) {
            (Dtype::U8 | Dtype::Bool, Dtype::F32) => "where_u8_f32",
            (Dtype::U32, Dtype::F32) => "where_u32_f32",
            (Dtype::U8 | Dtype::Bool, Dtype::BF16) => "where_u8_bf16",
            (Dtype::U8 | Dtype::Bool, Dtype::F16) => "where_u8_f16",
            (Dtype::U8 | Dtype::Bool, Dtype::I64) => "where_u8_i64",
            (Dtype::U8 | Dtype::Bool, Dtype::U32) => "where_u8_u32",
            (Dtype::U8 | Dtype::Bool, Dtype::U8 | Dtype::Bool) => "where_u8_u8",
            (l, r) => return Err(err(format!("where_cond {l:?} {r:?} not implemented"))),
        };
        let cond = self.contiguous()?;
        let t = on_true.contiguous()?;
        let f = on_false.contiguous()?;
        let size = cond.size();
        let out = Self::wrap(Arc::clone(&self.rt), self.rt.buffer(size * t.dtype.size_of(), "where_out")?, Layout::contiguous(cond.shape()), t.dtype);
        // All three operands are contiguous, so the kernel's per-operand
        // function constants are true and the strides go unused.
        let pipe = self.rt.compile_with_constants(
            SELECT_KERNEL,
            name,
            crate::runtime::Math::Fast,
            &[(0, crate::runtime::ConstVal::Bool(true)), (1, crate::runtime::ConstVal::Bool(true)), (2, crate::runtime::ConstVal::Bool(true))],
        )?;
        let rank = cond.rank();
        let dims: Vec<u64> = cond.shape().iter().map(|&d| d as u64).collect();
        let zeros: Vec<u64> = vec![0; rank];
        {
            let guard = self.rt.commands.encoder()?;
            let enc = guard.encoder();
            enc.set_pipeline(&pipe);
            enc.set_bytes(0, &size);
            enc.set_bytes(1, &rank);
            enc.set_bytes_directly(2, dims.len() * 8, dims.as_ptr().cast());
            enc.set_bytes_directly(3, zeros.len() * 8, zeros.as_ptr().cast());
            enc.set_bytes_directly(4, zeros.len() * 8, zeros.as_ptr().cast());
            enc.set_bytes_directly(5, zeros.len() * 8, zeros.as_ptr().cast());
            enc.set_input(6, Some(&cond.buf), cond.layout.offset * self.dtype.size_of());
            enc.set_input(7, Some(&t.buf), t.layout.offset * t.dtype.size_of());
            enc.set_input(8, Some(&f.buf), f.layout.offset * f.dtype.size_of());
            enc.set_output(9, Some(&out.buf), 0);
            let tile = 1.max(8 / t.dtype.size_of());
            let tiles = size.div_ceil(tile);
            let width = pipe.max_total_threads_per_threadgroup().min(tiles).max(1);
            let count = tiles.div_ceil(width);
            enc.dispatch_groups((count, 1, 1), (width, 1, 1));
        }
        Ok(out)
    }
}

impl Array {
    /// Element type conversion, via cast kernel. Only the `_strided`
    /// variant is dispatched — the contiguous one differs solely in the
    /// indexer, so the values are identical.
    ///
    /// The kernel set is (f32/f16/bf16/i64/u32/u8 as source and
    /// destination). Other pairs (e.g. i32) fall back to a host conversion,
    /// mirroring `device_cast` CPU fallback.
    pub fn cast(&self, dtype: Dtype) -> Result<Self> {
        if self.dtype == dtype {
            return Ok(self.clone());
        }
        let tag = |d: Dtype| -> Option<&'static str> {
            Some(match d {
                Dtype::F32 => "f32",
                Dtype::F16 => "f16",
                Dtype::BF16 => "bf16",
                Dtype::I64 => "i64",
                Dtype::U32 => "u32",
                Dtype::U8 | Dtype::Bool => "u8",
                _ => return None,
            })
        };
        let (Some(src), Some(dst)) = (tag(self.dtype), tag(dtype)) else {
            return self.cast_host(dtype);
        };
        let x = self.contiguous()?;
        // The strided kernel reads `dims[0]`; a 0-d input has no dims to bind
        // (a zero-length `setBytes` is not a buffer binding). Reshape to [1]:
        // the values are identical and the GPU path avoids `cast_host`, which
        // evals the device (a full GPU sync per call — bf16_scalar hits this
        // per layer per token).
        let (x, rank_pad) = if x.rank() == 0 {
            (x.reshape(&[1])?, true)
        } else {
            (x, false)
        };
        let name = format!("cast_{src}_{dst}_strided");
        let pipe = self
            .rt
            .compile_with(CAST_KERNEL, &name, crate::runtime::Math::Fast)?;
        let n = x.size();
        let out = Self::wrap(
            Arc::clone(&self.rt),
            self.rt.buffer(n * dtype.size_of(), "cast_out")?,
            Layout::contiguous(x.shape()),
            dtype,
        );
        let dims: Vec<u64> = x.shape().iter().map(|&d| d as u64).collect();
        let strides: Vec<u64> = x.layout.strides.iter().map(|&s| s as u64).collect();
        {
            let guard = self.rt.commands.encoder()?;
            let enc = guard.encoder();
            enc.set_pipeline(&pipe);
            enc.set_bytes(0, &n);
            let nd = x.rank();
            enc.set_bytes(1, &nd);
            enc.set_bytes_directly(2, dims.len() * 8, dims.as_ptr().cast());
            enc.set_bytes_directly(3, strides.len() * 8, strides.as_ptr().cast());
            enc.set_input(4, Some(&x.buf), x.layout.offset * self.dtype.size_of());
            enc.set_output(5, Some(&out.buf), 0);
            // strided cast uses `linear_split` over the element count.
            let width = pipe.max_total_threads_per_threadgroup().min(n).max(1);
            let count = n.div_ceil(width);
            enc.dispatch_groups((count, 1, 1), (width, 1, 1));
        }
        if rank_pad {
            return out.reshape(&[]);
        }
        Ok(out)
    }

    /// Host conversion for dtype pairs Metal cast does not cover.
    /// Mirrors `device_cast` CPU fallback; Rust `as` casts match it.
    fn cast_host(&self, dtype: Dtype) -> Result<Self> {
        if std::env::var("LISA_CAST_TRACE").is_ok() {
            static CT: std::sync::Mutex<Vec<(String, String)>> = std::sync::Mutex::new(Vec::new());
            let mut c = CT.lock().unwrap();
            let key = (format!("{:?}", self.dtype), format!("{dtype:?}"));
            if !c.contains(&key) {
                c.push(key.clone());
                eprintln!("[cast-host] {:?} -> {:?} (first occurrence)", key.0, key.1);
            }
        }
        use half::{bf16, f16};
        let x = self.contiguous()?;
        let out = Self::wrap(
            Arc::clone(&self.rt),
            self.rt.buffer(x.size() * dtype.size_of(), "cast_host")?,
            Layout::contiguous(x.shape()),
            dtype,
        );
        let put = |bytes: &[u8]| unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), out.buf.contents(), bytes.len())
        };
        // Read the source through f64 (exact for every supported pair).
        let vals: Vec<f64> = match x.dtype {
            Dtype::F32 => x.to_vec::<f32>()?.iter().map(|&v| v as f64).collect(),
            Dtype::F64 => x.to_vec::<f64>()?,
            Dtype::F16 => x.to_vec::<f16>()?.iter().map(|&v| v.to_f64()).collect(),
            Dtype::BF16 => x.to_vec::<bf16>()?.iter().map(|&v| v.to_f64()).collect(),
            Dtype::I64 => x.to_vec::<i64>()?.iter().map(|&v| v as f64).collect(),
            Dtype::I32 => x.to_vec::<i32>()?.iter().map(|&v| v as f64).collect(),
            Dtype::I16 => x.to_vec::<i16>()?.iter().map(|&v| v as f64).collect(),
            Dtype::I8 => x.to_vec::<i8>()?.iter().map(|&v| v as f64).collect(),
            Dtype::U32 => x.to_vec::<u32>()?.iter().map(|&v| v as f64).collect(),
            Dtype::U16 => x.to_vec::<u16>()?.iter().map(|&v| v as f64).collect(),
            Dtype::U8 | Dtype::Bool => x.to_vec::<u8>()?.iter().map(|&v| v as f64).collect(),
        };
        match dtype {
            Dtype::F32 => put(bytes_of(&vals.iter().map(|&v| v as f32).collect::<Vec<_>>())),
            Dtype::F64 => put(bytes_of(&vals)),
            Dtype::F16 => put(bytes_of(
                &vals.iter().map(|&v| f16::from_f64(v)).collect::<Vec<_>>(),
            )),
            Dtype::BF16 => put(bytes_of(
                &vals.iter().map(|&v| bf16::from_f64(v)).collect::<Vec<_>>(),
            )),
            Dtype::I64 => put(bytes_of(
                &vals.iter().map(|&v| v as i64).collect::<Vec<_>>(),
            )),
            Dtype::I32 => put(bytes_of(
                &vals.iter().map(|&v| v as i32).collect::<Vec<_>>(),
            )),
            Dtype::I16 => put(bytes_of(
                &vals.iter().map(|&v| v as i16).collect::<Vec<_>>(),
            )),
            Dtype::I8 => put(bytes_of(&vals.iter().map(|&v| v as i8).collect::<Vec<_>>())),
            Dtype::U32 => put(bytes_of(
                &vals.iter().map(|&v| v as u32).collect::<Vec<_>>(),
            )),
            Dtype::U16 => put(bytes_of(
                &vals.iter().map(|&v| v as u16).collect::<Vec<_>>(),
            )),
            Dtype::U8 | Dtype::Bool => {
                put(bytes_of(&vals.iter().map(|&v| v as u8).collect::<Vec<_>>()))
            }
        }
        Ok(out)
    }
}

/// View a POD slice as bytes.
fn bytes_of<T: Copy>(v: &[T]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr().cast::<u8>(), std::mem::size_of_val(v)) }
}

/// Max rank the strided-copy kernel takes (kept small so the params struct is
/// a plain `setBytes`).
pub const MAX_COPY_RANK: usize = 8;

#[repr(C)]
#[derive(Clone, Copy)]
struct CopyParams {
    ndim: u32,
    shape: [u32; MAX_COPY_RANK],
    strides: [u32; MAX_COPY_RANK],
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Copy2Params {
    ndim: u32,
    shape: [u32; MAX_COPY_RANK],
    src_strides: [u32; MAX_COPY_RANK],
    dst_strides: [u32; MAX_COPY_RANK],
}

fn pad_rank(v: &[usize]) -> [u32; MAX_COPY_RANK] {
    let mut o = [0u32; MAX_COPY_RANK];
    for (i, &x) in v.iter().enumerate() {
        o[i] = x as u32;
    }
    o
}

/// `copy_gg` (concatenate slices): template kernels + per-type instantiations,
/// prepended with the MLX utils preamble the helpers live in.
const COPY_GG_SOURCE: &str = include_str!("kernels/common/copy_gg.metal");
/// Strided -> strided copy body (`slice_assign` into a view).
const COPY2_STRIDED_SOURCE: &str = include_str!("kernels/common/copy2_strided.metal");
/// Strided -> contiguous materialisation of a view.
const COPY_STRIDED_SOURCE: &str = include_str!("kernels/common/copy_strided.metal");

/// Source for a `copy_gg_{ndim}` kernel of `dtype`: the preamble plus the static
/// template file (which carries every explicit instantiation).
fn copy_gg_source() -> String {
    format!("{}{COPY_GG_SOURCE}", crate::mlx_rt::MLX_UTILS_PREAMBLE)
}

fn promote(a: Dtype, b: Dtype) -> Dtype {
    if a == b {
        return a;
    }
    let rank = |d: Dtype| match d {
        Dtype::Float64 => 6,
        Dtype::Float32 => 5,
        Dtype::Bfloat16 => 4,
        Dtype::Float16 => 3,
        Dtype::Int64 => 2,
        Dtype::Uint32 | Dtype::Int32 => 1,
        _ => 0,
    };
    if a.is_float() == b.is_float() {
        if rank(a) >= rank(b) { a } else { b }
    } else if a.is_float() { a } else { b }
}

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

// ─────────── API-compatible facade ───────────
//

// `mlx_rt` and the shim were written against `Tensor`; these are
// the method names they call, mapped onto the native ops above. Kept as the
// seam so the conversion is a type change rather than a rewrite.
impl Array {
    pub fn device(&self) -> &Arc<MetalRuntime> {
        &self.rt
    }
    pub fn to_dtype(&self, dt: Dtype) -> Result<Self> {
        self.cast(dt)
    }
    pub fn unsqueeze(&self, axis: usize) -> Result<Self> {
        self.expand_dims(axis)
    }
    /// `broadcast_as` (accepts a slice or a `Vec`).
    pub fn broadcast_as<S: AsRef<[usize]>>(&self, shape: S) -> Result<Self> {
        self.broadcast_to(shape.as_ref())
    }

    /// `reshape` (accepts a slice or a `Vec`).
    pub fn reshape_dims<S: AsRef<[usize]>>(&self, shape: S) -> Result<Self> {
        self.reshape(shape.as_ref())
    }
    pub fn argmax(&self, axis: usize) -> Result<Self> {
        self.argmax_axis(axis as i32)
    }
    pub fn argmin(&self, axis: usize) -> Result<Self> {
        self.argmin_axis(axis as i32)
    }
    pub fn mul(&self, other: &Array) -> Result<Self> {
        self.multiply(other)
    }
    pub fn broadcast_mul(&self, other: &Array) -> Result<Self> {
        self.multiply(other)
    }
    pub fn broadcast_add(&self, other: &Array) -> Result<Self> {
        self.add(other)
    }
    pub fn broadcast_sub(&self, other: &Array) -> Result<Self> {
        self.subtract(other)
    }
    pub fn broadcast_div(&self, other: &Array) -> Result<Self> {
        self.divide(other)
    }
    pub fn eq(&self, other: &Array) -> Result<Self> {
        self.cmp(other, "eq")
    }
    pub fn sub(&self, other: &Array) -> Result<Self> {
        self.subtract(other)
    }
    pub fn floor(&self) -> Result<Self> {
        self.unary("Floor")
    }
    /// Matrix multiply (`matmul`), via the NAX GEMM. Rank-3 inputs
    /// are batched (each batch row flattened into M).
    /// Zero-pad a contiguous 2-D `[m, k]` array to `[rows, cols]` (host copy;
    /// only used to align GEMM operands, so the sizes are small).
    pub fn pad2(&self, rows: usize, cols: usize) -> Result<Self> {
        let x = self.contiguous()?;
        let (m, k) = (x.dim(0), x.dim(1));
        if m > rows || k > cols {
            return Err(err("pad2: target smaller than source"));
        }
        let out = Self::zeros(&self.rt, &[rows, cols], self.dtype)?;
        self.rt.commands.flush_and_wait()?;
        let esz = self.dtype.size_of();
        unsafe {
            let src = x.buf.contents().add(x.layout.offset * esz);
            let dst = out.buf.contents();
            for r in 0..m {
                std::ptr::copy_nonoverlapping(src.add(r * k * esz), dst.add(r * cols * esz), k * esz);
            }
        }
        Ok(out)
    }

    pub fn matmul(&self, other: &Array) -> Result<Self> {
        // `matmul_nax` reads both operands linearly, so a transposed view (e.g.
        // `w.t()`) must be materialised first.
        if self.rank() == 2 && other.rank() == 2 {
            // the operands are promoted to a common dtype before the GEMM.
            let dt = promote(self.dtype(), other.dtype());
            let a = self.to_dtype(dt)?.contiguous()?;
            let b = other.to_dtype(dt)?.contiguous()?;
            let (m, k, n) = (a.dim(0), a.dim(1), b.dim(1));
            let out = Self::zeros(&self.rt, &[m, n], dt)?;
            crate::mlx_rt::dense_gemm(
                &self.rt,
                (1, m, n, k),
                a.layout.strides.as_slice(),
                a.layout.offset * dt.size_of(),
                &a.buf,
                b.layout.strides.as_slice(),
                b.layout.offset * dt.size_of(),
                &b.buf,
                &out,
                dt,
            )?;
            return Ok(out);
        }
        if self.rank() == 3 && other.rank() == 2 {
            // broadcast_matmul broadcasts the 2-D rhs; every scored
            // caller has b == 1, so one (m, n, k) GEMM covers it.
            let dt = promote(self.dtype(), other.dtype());
            let a = self.to_dtype(dt)?.contiguous()?;
            let bb = other.to_dtype(dt)?.contiguous()?;
            let (bm, m, k) = (a.dim(0), a.dim(1), a.dim(2));
            let n = bb.dim(1);
            let out = Self::zeros(&self.rt, &[bm, m, n], dt)?;
            crate::mlx_rt::dense_gemm(
                &self.rt,
                (bm, m, n, k),
                a.layout.strides.as_slice(),
                a.layout.offset * dt.size_of(),
                &a.buf,
                bb.layout.strides.as_slice(),
                bb.layout.offset * dt.size_of(),
                &bb.buf,
                &out,
                dt,
            )?;
            return Ok(out);
        }
        if self.rank() == 3 && other.rank() == 3 && self.dim(0) == other.dim(0) {
            let dt = promote(self.dtype(), other.dtype());
            let a = self.to_dtype(dt)?.contiguous()?;
            let bb = other.to_dtype(dt)?.contiguous()?;
            let (bn, m, k) = (a.dim(0), a.dim(1), a.dim(2));
            let n = bb.dim(2);
            let out = Self::zeros(&self.rt, &[bn, m, n], dt)?;
            crate::mlx_rt::dense_gemm(
                &self.rt,
                (bn, m, n, k),
                a.layout.strides.as_slice(),
                a.layout.offset * dt.size_of(),
                &a.buf,
                bb.layout.strides.as_slice(),
                bb.layout.offset * dt.size_of(),
                &bb.buf,
                &out,
                dt,
            )?;
            return Ok(out);
        }
        Err(err(format!("matmul: shapes {:?} x {:?}", self.shape(), other.shape())))
    }

    /// `broadcast_matmul` (a plain matmul for the shapes the tree uses).
    pub fn broadcast_matmul(&self, other: &Array) -> Result<Self> {
        self.matmul(other)
    }

    /// `chunk`: `num_splits` equal parts along `axis`.
    pub fn chunk(&self, num_splits: usize, axis: usize) -> Result<Vec<Self>> {
        self.split_equal(num_splits, axis)
    }
    pub fn ne(&self, other: &Array) -> Result<Self> {
        self.cmp(other, "ne")
    }
    /// `1 / x`.
    pub fn recip(&self) -> Result<Self> {
        let one = Self::scalar_of(&self.rt, 1.0, self.dtype)?;
        one.divide(self)
    }
    pub fn to_scalar<T: Copy + 'static>(&self) -> Result<T> {
        Ok(self.item::<T>())
    }
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
    pub fn to_vec1<T: Copy>(&self) -> Result<Vec<T>> {
        self.to_vec::<T>()
    }
}

// ─────────────────────────────── tests ───────────────────────────────
