//! Custom Metal kernel interface, now backed by the runtime.
//!
//! This used to be a raw FFI wrapper. The engine kernel modules
//! (`kernels`, `moe_decode`, `prefill_indirect`, `qsa`) call it as
//! `MetalKernel::new(...)` / `apply(&[Array], ...)`; here that maps onto
//! [`crate::mlx_rt::MetalKernel`] (a port of MLX's `metal_kernel.cpp`), with
//! the shim `Array`/`Dtype` translated at the boundary.

use crate::shim_api::{Array, Dtype, Stream};

/// Template argument for a Metal kernel (shim-typed).
pub enum TemplateArg {
    Dtype(&'static str, Dtype),
    Int(&'static str, i32),
    Bool(&'static str, bool),
}

/// Output argument description: shape + dtype (shim-typed).
pub struct OutputArg {
    pub shape: Vec<i32>,
    pub dtype: Dtype,
}

fn to_template_arg(a: &TemplateArg) -> lisa_mlx::error::Result<crate::mlx_rt::TemplateArg> {
    Ok(match a {
        TemplateArg::Dtype(n, d) => crate::mlx_rt::TemplateArg::Dtype(n, *d),
        TemplateArg::Int(n, v) => crate::mlx_rt::TemplateArg::Int(n, *v),
        TemplateArg::Bool(n, v) => crate::mlx_rt::TemplateArg::Bool(n, *v),
    })
}

/// A JIT-compiled custom Metal kernel, equivalent to `mx.fast.metal_kernel`.
pub struct MetalKernel {
    inner: crate::mlx_rt::MetalKernel,
}

unsafe impl Send for MetalKernel {}
unsafe impl Sync for MetalKernel {}

impl MetalKernel {
    pub fn new(
        name: &str,
        input_names: &[&str],
        output_names: &[&str],
        source: &str,
        header: &str,
        ensure_row_contiguous: bool,
        atomic_outputs: bool,
    ) -> lisa_mlx::error::Result<Self> {
        Ok(Self {
            inner: crate::mlx_rt::MetalKernel::new(
                name,
                input_names,
                output_names,
                source,
                header,
                ensure_row_contiguous,
                atomic_outputs,
            )?,
        })
    }

    /// Apply the kernel with the given inputs, template args, grid and outputs.
    pub fn apply(
        &self,
        inputs: &[&Array],
        template: &[TemplateArg],
        grid: (i32, i32, i32),
        thread_group: (i32, i32, i32),
        outputs: &[OutputArg],
        stream: &Stream,
    ) -> lisa_mlx::error::Result<Vec<Array>> {
        let ts: Vec<&crate::array::Array> = inputs.iter().map(|a| &a.t).collect();
        let tmpl: Vec<crate::mlx_rt::TemplateArg> = template
            .iter()
            .map(to_template_arg)
            .collect::<lisa_mlx::error::Result<_>>()?;
        let outs: Vec<crate::mlx_rt::OutputArg> = outputs
            .iter()
            .map(|o| -> lisa_mlx::error::Result<_> {
                                Ok(crate::mlx_rt::OutputArg {
                    shape: o.shape.clone(),
                    dtype: o.dtype,
                })
            })
            .collect::<lisa_mlx::error::Result<_>>()?;
        let res = self
            .inner
            .apply(stream.runtime(), &ts, &tmpl, grid, thread_group, &outs, None)?;
        Ok(res.into_iter().map(Array::new).collect())
    }

    /// Like [`apply`], but writes into the caller's pre-allocated output
    /// tensors instead of allocating fresh buffers. `prealloc` must match the
    /// kernel's output count and shapes.
    pub fn apply_into(
        &self,
        inputs: &[&Array],
        template: &[TemplateArg],
        grid: (i32, i32, i32),
        thread_group: (i32, i32, i32),
        outputs: &[OutputArg],
        prealloc: &[&Array],
        stream: &Stream,
    ) -> lisa_mlx::error::Result<Vec<Array>> {
        let ts: Vec<&crate::array::Array> = inputs.iter().map(|a| &a.t).collect();
        let tmpl: Vec<crate::mlx_rt::TemplateArg> = template
            .iter()
            .map(to_template_arg)
            .collect::<lisa_mlx::error::Result<_>>()?;
        let outs: Vec<crate::mlx_rt::OutputArg> = outputs
            .iter()
            .map(|o| -> lisa_mlx::error::Result<_> {
                Ok(crate::mlx_rt::OutputArg {
                    shape: o.shape.clone(),
                    dtype: o.dtype,
                })
            })
            .collect::<lisa_mlx::error::Result<_>>()?;
        let pa: Vec<crate::array::Array> = prealloc.iter().map(|a| a.t.clone()).collect();
        let res = self
            .inner
            .apply(stream.runtime(), &ts, &tmpl, grid, thread_group, &outs, Some(&pa))?;
        Ok(res.into_iter().map(Array::new).collect())
    }
}

/// Kept for the CLI: with the runtime the AOT metallib is selected per
/// kernel (see `mlx_rt::build_metallib`), so there is nothing global to set.
pub fn set_metallib_path(_path: &str) -> Result<(), String> {
    Ok(())
}

/// Kept for the CLI; the runtime reports errors through `Result`.
pub fn install_error_handler() {}
