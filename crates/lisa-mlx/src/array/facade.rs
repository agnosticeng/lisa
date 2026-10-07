use std::sync::Arc;

use crate::error::Result;
use crate::runtime::MetalRuntime;

use super::dtype::promote;
use super::{Array, Dtype, err};

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
                std::ptr::copy_nonoverlapping(
                    src.add(r * k * esz),
                    dst.add(r * cols * esz),
                    k * esz,
                );
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
            crate::jit::dense_gemm(
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
            crate::jit::dense_gemm(
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
            crate::jit::dense_gemm(
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
        Err(err(format!(
            "matmul: shapes {:?} x {:?}",
            self.shape(),
            other.shape()
        )))
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
    pub fn to_vec1<T: Copy>(&self) -> Result<Vec<T>> {
        self.to_vec::<T>()
    }
}
