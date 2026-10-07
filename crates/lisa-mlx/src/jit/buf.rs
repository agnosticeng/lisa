use objc2_metal::MTLSize;

use super::compile::{
    builtin_template_def, compile_builtin, get_block_dims, type_string, type_to_name,
};
use super::{Device, Tensor};
use super::{MLX_REDUCE_UTILS_PREAMBLE, MLX_SCATTER_AXIS_PREAMBLE, MLX_UTILS_PREAMBLE};
use crate::error::Result;

/// MLX `collapse_contiguous_dims` (`backend/common/utils.cpp:24`).
///
/// Collapses runs of axes that are contiguous for every input into a single
/// axis, returning one shape plus the collapsed strides per input. Used by the
/// gather/qmm kernels to describe batched shapes and gather-index strides.
pub fn collapse_contiguous_dims(
    shape: &[i32],
    strides: &[Vec<i64>],
    size_cap: i64,
) -> (Vec<i32>, Vec<Vec<i64>>) {
    let mut to_collapse: Vec<i32> = Vec::new();
    if !shape.is_empty() {
        if shape[0] != 1 {
            to_collapse.push(0);
        }
        let mut size: i64 = shape[0] as i64;
        for i in 1..shape.len() {
            let mut contiguous = true;
            size *= shape[i] as i64;
            for st in strides {
                if st[i] * shape[i] as i64 != st[i - 1] || size > size_cap {
                    contiguous = false;
                    size = shape[i] as i64;
                    break;
                }
            }
            if !contiguous {
                to_collapse.push(-1);
            }
            if shape[i] != 1 {
                to_collapse.push(i as i32);
            }
        }
        to_collapse.push(-1);
    }

    let mut out_shape: Vec<i32> = Vec::new();
    let mut out_strides: Vec<Vec<i64>> = vec![Vec::new(); strides.len()];
    let mut i = 0usize;
    loop {
        while i < to_collapse.len() && to_collapse[i] == -1 {
            i += 1;
        }
        if i == to_collapse.len() {
            break;
        }
        let mut current_shape = shape[to_collapse[i] as usize];
        let mut k = i;
        loop {
            k += 1;
            if to_collapse[k] == -1 {
                break;
            }
            current_shape *= shape[to_collapse[k] as usize];
        }
        out_shape.push(current_shape);
        for (j, st) in strides.iter().enumerate() {
            out_strides[j].push(st[to_collapse[k - 1] as usize]);
        }
        i = k + 1;
    }

    if !shape.is_empty() && out_shape.is_empty() {
        out_shape.push(1);
        for os in out_strides.iter_mut() {
            os.push(0);
        }
    }
    (out_shape, out_strides)
}

pub fn collapse_default_cap() -> i64 {
    i64::from(i32::MAX)
}

/// Accumulates the small scalar/vector blobs MLX writes with `set_bytes`
/// (`int`) and `set_vector_bytes` (`vector<int>` / `vector<int64_t>`), keeping
/// the order in which they were written so they can be bound to consecutive
/// buffer indices. Metal copies on set, but the bytes must outlive dispatch.
pub struct ByteBlob {
    buf: Vec<u8>,
    fields: Vec<(usize, usize)>,
}

impl ByteBlob {
    pub fn new() -> Self {
        Self {
            buf: Vec::new(),
            fields: Vec::new(),
        }
    }
    fn push(&mut self, bytes: &[u8]) {
        self.fields.push((self.buf.len(), bytes.len()));
        self.buf.extend_from_slice(bytes);
    }
    pub fn i32(&mut self, v: i32) {
        self.push(&v.to_le_bytes());
    }
    pub fn i32s(&mut self, vs: &[i32]) {
        self.push(&vs.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>());
    }
    pub fn i64s(&mut self, vs: &[i64]) {
        self.push(&vs.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>());
    }
    pub fn len_fields(&self) -> usize {
        self.fields.len()
    }
    /// Bind each field to `base`, `base+1`, … via `set_bytes_directly`.
    pub fn bind(&self, enc: &crate::runtime::ComputeEncoder, base: usize) {
        for (i, (off, len)) in self.fields.iter().enumerate() {
            enc.set_bytes_directly(base + i, *len, self.buf[*off..].as_ptr().cast());
        }
    }
}

impl Default for ByteBlob {
    fn default() -> Self {
        Self::new()
    }
}

/// MLX `add_strides_and_shapes` (`quantized.cpp:151`). Each argument is a
/// `(shape, strides)` pair in elements (strides stored `int64_t`).
pub fn add_strides_and_shapes(
    x: (&[usize], &[usize]),
    w: (&[usize], &[usize]),
    scales_strides: &[usize],
    biases_strides: Option<&[usize]>,
) -> ByteBlob {
    let mut b = ByteBlob::new();
    b.i32(x.0.len() as i32 - 2);
    b.i32s(&x.0.iter().map(|&v| v as i32).collect::<Vec<_>>());
    b.i64s(&x.1.iter().map(|&v| v as i64).collect::<Vec<_>>());
    b.i32(w.0.len() as i32 - 2);
    b.i32s(&w.0.iter().map(|&v| v as i32).collect::<Vec<_>>());
    b.i64s(&w.1.iter().map(|&v| v as i64).collect::<Vec<_>>());
    b.i64s(&scales_strides.iter().map(|&v| v as i64).collect::<Vec<_>>());
    if let Some(bs) = biases_strides {
        b.i64s(&bs.iter().map(|&v| v as i64).collect::<Vec<_>>());
    }
    b
}

/// MLX `add_gather_strides_and_shapes` (`quantized.cpp:181`). Collapses the lhs
/// index shape against the lhs/rhs index strides, then writes the shared shape
/// and both collapsed stride vectors.
pub fn add_gather_strides_and_shapes(
    lhs: (&[usize], &[usize]),
    rhs: (&[usize], &[usize]),
) -> ByteBlob {
    let shape: Vec<i32> = lhs.0.iter().map(|&v| v as i32).collect();
    let strides = vec![
        lhs.1.iter().map(|&v| v as i64).collect::<Vec<_>>(),
        rhs.1.iter().map(|&v| v as i64).collect::<Vec<_>>(),
    ];
    let (shape, strides) = collapse_contiguous_dims(&shape, &strides, collapse_default_cap());
    let mut b = ByteBlob::new();
    b.i32(shape.len() as i32);
    b.i32s(&shape);
    b.i64s(&strides[0]);
    b.i64s(&strides[1]);
    b
}

/// MLX `ops::put_along_axis` / `ScatterAxis::eval_gpu` (`indexing.cpp:522`).
/// `out = copy(src)` then scatter `updates` at `indices` along `axis`.
pub fn put_along_axis(
    device: &Device,
    src: &Tensor,
    indices: &Tensor,
    updates: &Tensor,
    axis: i32,
) -> Result<Tensor> {
    let mdev = device;
    let out = src.copied()?;
    let dims = indices.dims().to_vec();
    let rank = dims.len();
    let ax = if axis < 0 {
        (rank as i32 + axis) as usize
    } else {
        axis as usize
    };
    let out_t = type_string(src.dtype())?;
    let idx_t = type_string(indices.dtype())?;
    let out_ty = type_to_name(src.dtype())?;
    let idx_ty = type_to_name(indices.dtype())?;
    let lib_name = format!("scatter_axis{out_ty}{idx_ty}_none_int");
    let kernel_name = format!("{lib_name}cc");
    let a = |s: &str| s.to_string();
    let mut defs = String::new();
    for (uc, ic) in [(true, true), (true, false), (false, true), (false, false)] {
        defs.push_str(&builtin_template_def(
            &format!(
                "{lib_name}{}{}",
                if uc { "c" } else { "nc" },
                if ic { "c" } else { "nc" }
            ),
            "scatter_axis",
            &[
                a(out_t),
                a(idx_t),
                a("int"),
                a("None"),
                a(if uc { "true" } else { "false" }),
                a(if ic { "true" } else { "false" }),
            ],
        ));
    }
    let source =
        format!("{MLX_UTILS_PREAMBLE}{MLX_REDUCE_UTILS_PREAMBLE}{MLX_SCATTER_AXIS_PREAMBLE}{defs}");
    let pipeline = compile_builtin(device, &source, &kernel_name)?;

    let mut shape: Vec<i32> = Vec::new();
    let mut upd_strides: Vec<i64> = Vec::new();
    let mut idx_strides: Vec<i64> = Vec::new();
    let (_s, idx_layout) = indices.buffer_and_layout();
    let (_s, upd_layout) = updates.buffer_and_layout();
    let mut out_axis_size = 0i32;
    let mut upd_ax_stride = 0usize;
    let mut idx_ax_stride = 0usize;
    for i in 0..rank {
        if i == ax {
            out_axis_size = src.dims()[i] as i32;
            upd_ax_stride = upd_layout.stride()[i];
            idx_ax_stride = idx_layout.stride()[i];
            continue;
        }
        shape.push(dims[i] as i32);
        upd_strides.push(upd_layout.stride()[i] as i64);
        idx_strides.push(idx_layout.stride()[i] as i64);
    }
    let ndim = rank - 1;

    let guard = mdev.commands.encoder()?;
    let enc = guard.encoder();
    enc.set_pipeline(&pipeline);
    let bind = |index: usize, t: &Tensor| -> Result<()> {
        let (ms, layout) = t.buffer_and_layout();
        enc.set_input(index, Some(ms), layout.offset * t.dtype().size_of());
        Ok(())
    };
    bind(0, updates)?;
    bind(1, indices)?;
    {
        let (ms, layout) = out.buffer_and_layout();
        enc.set_output(2, Some(ms), layout.offset * out.dtype().size_of());
    }
    if ndim == 0 {
        let z: i32 = 0;
        let zl: i64 = 0;
        enc.set_bytes(3, &z);
        enc.set_bytes(4, &zl);
        enc.set_bytes(5, &zl);
    } else {
        enc.set_bytes_directly(
            3,
            std::mem::size_of_val(shape.as_slice()),
            shape.as_ptr().cast(),
        );
        enc.set_bytes_directly(
            4,
            std::mem::size_of_val(upd_strides.as_slice()),
            upd_strides.as_ptr().cast(),
        );
        enc.set_bytes_directly(
            5,
            std::mem::size_of_val(idx_strides.as_slice()),
            idx_strides.as_ptr().cast(),
        );
    }
    enc.set_bytes(6, &ndim);
    enc.set_bytes(7, &(ax as i32));
    enc.set_bytes(8, &out_axis_size);
    enc.set_bytes(9, &upd_ax_stride);
    enc.set_bytes(10, &idx_ax_stride);

    let size_pre: usize = dims[..ax].iter().product();
    let size_post: usize = dims[ax + 1..].iter().product();
    let idx_ax_size = dims[ax];
    let group = get_block_dims(size_post, idx_ax_size, size_pre, 10);
    enc.dispatch_threads_size(
        MTLSize {
            width: size_post,
            height: idx_ax_size,
            depth: size_pre,
        },
        MTLSize {
            width: group.0,
            height: group.1,
            depth: group.2,
        },
    );
    Ok(out)
}
