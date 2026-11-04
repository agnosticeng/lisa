//! Custom Metal kernel interface, now backed by the runtime.
//!
//! This used to be a raw FFI wrapper. The engine kernel modules
//! (`kernels`, `moe_decode`, `prefill_indirect`, `qsa`) call it as
//! `MetalKernel::new(...)` / `apply(&[Array], ...)`; here that maps onto
//! [`crate::jit::MetalKernel`] (a port of MLX's `metal_kernel.cpp`), with
//! the shim `Array`/`Dtype` translated at the boundary.

pub use crate::jit::KernelInput;
use crate::ops::{Array, Dtype, Stream};
use crate::runtime::ResidencySet;

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

fn to_template_arg(a: &TemplateArg) -> lisa_mlx::error::Result<crate::jit::TemplateArg> {
    Ok(match a {
        TemplateArg::Dtype(n, d) => crate::jit::TemplateArg::Dtype(n, *d),
        TemplateArg::Int(n, v) => crate::jit::TemplateArg::Int(n, *v),
        TemplateArg::Bool(n, v) => crate::jit::TemplateArg::Bool(n, *v),
    })
}

/// A JIT-compiled custom Metal kernel, equivalent to `mx.fast.metal_kernel`.
pub struct MetalKernel {
    inner: crate::jit::MetalKernel,
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
        Self::new_with_math(
            name,
            input_names,
            output_names,
            source,
            header,
            ensure_row_contiguous,
            atomic_outputs,
            crate::runtime::Math::Jit,
        )
    }

    /// Create a kernel with an explicit Metal math law.
    pub fn new_with_math(
        name: &str,
        input_names: &[&str],
        output_names: &[&str],
        source: &str,
        header: &str,
        ensure_row_contiguous: bool,
        atomic_outputs: bool,
        math: crate::runtime::Math,
    ) -> lisa_mlx::error::Result<Self> {
        Self::new_with_math_and_threadgroup_memory(
            name,
            input_names,
            output_names,
            source,
            header,
            ensure_row_contiguous,
            atomic_outputs,
            math,
            0,
        )
    }

    /// Create a kernel with dynamic threadgroup scratch bound at index zero.
    pub fn new_with_math_and_threadgroup_memory(
        name: &str,
        input_names: &[&str],
        output_names: &[&str],
        source: &str,
        header: &str,
        ensure_row_contiguous: bool,
        atomic_outputs: bool,
        math: crate::runtime::Math,
        threadgroup_memory_bytes: usize,
    ) -> lisa_mlx::error::Result<Self> {
        Ok(Self {
            inner: crate::jit::MetalKernel::new_with_math_and_threadgroup_memory(
                name,
                input_names,
                output_names,
                source,
                header,
                ensure_row_contiguous,
                atomic_outputs,
                math,
                threadgroup_memory_bytes,
            )?,
        })
    }

    /// Mark a kernel whose validated dispatch writes every logical output
    /// element before reading it. Partial or atomic outputs must not opt in.
    pub fn with_full_write_outputs(mut self) -> Self {
        self.inner = self.inner.with_full_write_outputs();
        self
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
        let tmpl: Vec<crate::jit::TemplateArg> = template
            .iter()
            .map(to_template_arg)
            .collect::<lisa_mlx::error::Result<_>>()?;
        let outs: Vec<crate::jit::OutputArg> = outputs
            .iter()
            .map(|o| -> lisa_mlx::error::Result<_> {
                Ok(crate::jit::OutputArg {
                    shape: o.shape.clone(),
                    dtype: o.dtype,
                })
            })
            .collect::<lisa_mlx::error::Result<_>>()?;
        let res = self.inner.apply(
            stream.runtime(),
            &ts,
            &tmpl,
            grid,
            thread_group,
            &outs,
            None,
        )?;
        Ok(res.into_iter().map(Array::new).collect())
    }

    /// Apply with inline scalar arguments and ordinary tensor inputs mixed in.
    pub fn apply_with_inline_inputs(
        &self,
        inputs: &[KernelInput<'_>],
        template: &[TemplateArg],
        grid: (i32, i32, i32),
        thread_group: (i32, i32, i32),
        outputs: &[OutputArg],
        stream: &Stream,
    ) -> lisa_mlx::error::Result<Vec<Array>> {
        let tmpl: Vec<crate::jit::TemplateArg> = template
            .iter()
            .map(to_template_arg)
            .collect::<lisa_mlx::error::Result<_>>()?;
        let outs: Vec<crate::jit::OutputArg> = outputs
            .iter()
            .map(|o| -> lisa_mlx::error::Result<_> {
                Ok(crate::jit::OutputArg {
                    shape: o.shape.clone(),
                    dtype: o.dtype,
                })
            })
            .collect::<lisa_mlx::error::Result<_>>()?;
        let res = self.inner.apply_with_inline_inputs(
            stream.runtime(),
            inputs,
            &tmpl,
            grid,
            thread_group,
            &outs,
            None,
        )?;
        Ok(res.into_iter().map(Array::new).collect())
    }

    /// Apply mixed tensor/scalar inputs while retaining indirect tensor
    /// buffers until command-buffer completion.
    pub fn apply_with_inline_inputs_and_keepalive(
        &self,
        inputs: &[KernelInput<'_>],
        template: &[TemplateArg],
        grid: (i32, i32, i32),
        thread_group: (i32, i32, i32),
        outputs: &[OutputArg],
        keepalive: &[&Array],
        stream: &Stream,
    ) -> lisa_mlx::error::Result<Vec<Array>> {
        self.apply_with_inline_resource_mode(
            inputs,
            template,
            grid,
            thread_group,
            outputs,
            keepalive,
            stream,
            false,
        )
    }

    /// Apply mixed tensor/scalar inputs using the queue-wide residency set for
    /// indirectly addressed buffers.
    pub fn apply_with_inline_inputs_and_residency_set(
        &self,
        inputs: &[KernelInput<'_>],
        template: &[TemplateArg],
        grid: (i32, i32, i32),
        thread_group: (i32, i32, i32),
        outputs: &[OutputArg],
        keepalive: &[&Array],
        stream: &Stream,
    ) -> lisa_mlx::error::Result<Vec<Array>> {
        self.apply_with_inline_resource_mode(
            inputs,
            template,
            grid,
            thread_group,
            outputs,
            keepalive,
            stream,
            true,
        )
    }

    fn apply_with_inline_resource_mode(
        &self,
        inputs: &[KernelInput<'_>],
        template: &[TemplateArg],
        grid: (i32, i32, i32),
        thread_group: (i32, i32, i32),
        outputs: &[OutputArg],
        keepalive: &[&Array],
        stream: &Stream,
        queue_resident: bool,
    ) -> lisa_mlx::error::Result<Vec<Array>> {
        let keepalive_tensors: Vec<&crate::array::Array> = keepalive.iter().map(|a| &a.t).collect();
        let tmpl: Vec<crate::jit::TemplateArg> = template
            .iter()
            .map(to_template_arg)
            .collect::<lisa_mlx::error::Result<_>>()?;
        let outs: Vec<crate::jit::OutputArg> = outputs
            .iter()
            .map(|o| -> lisa_mlx::error::Result<_> {
                Ok(crate::jit::OutputArg {
                    shape: o.shape.clone(),
                    dtype: o.dtype,
                })
            })
            .collect::<lisa_mlx::error::Result<_>>()?;
        let res = if queue_resident {
            self.inner.apply_with_inline_inputs_and_residency_set(
                stream.runtime(),
                inputs,
                &tmpl,
                grid,
                thread_group,
                &outs,
                None,
                &keepalive_tensors,
            )?
        } else {
            self.inner.apply_with_inline_inputs_and_keepalive(
                stream.runtime(),
                inputs,
                &tmpl,
                grid,
                thread_group,
                &outs,
                None,
                &keepalive_tensors,
            )?
        };
        Ok(res.into_iter().map(Array::new).collect())
    }

    /// Apply the kernel while retaining tensors referenced indirectly through
    /// GPU virtual addresses until their command buffer completes.
    pub fn apply_with_keepalive(
        &self,
        inputs: &[&Array],
        template: &[TemplateArg],
        grid: (i32, i32, i32),
        thread_group: (i32, i32, i32),
        outputs: &[OutputArg],
        keepalive: &[&Array],
        stream: &Stream,
    ) -> lisa_mlx::error::Result<Vec<Array>> {
        self.apply_with_resource_mode(
            inputs,
            template,
            grid,
            thread_group,
            outputs,
            keepalive,
            stream,
            false,
            None,
        )
    }

    /// Like `apply_with_keepalive`, but the indirect inputs are already
    /// covered by a queue-wide Metal residency set.
    pub fn apply_with_residency_set(
        &self,
        inputs: &[&Array],
        template: &[TemplateArg],
        grid: (i32, i32, i32),
        thread_group: (i32, i32, i32),
        outputs: &[OutputArg],
        keepalive: &[&Array],
        stream: &Stream,
    ) -> lisa_mlx::error::Result<Vec<Array>> {
        self.apply_with_resource_mode(
            inputs,
            template,
            grid,
            thread_group,
            outputs,
            keepalive,
            stream,
            true,
            None,
        )
    }

    /// Apply while attaching a per-layer indirect-resource set to the active
    /// command buffer.
    pub fn apply_with_command_residency_set(
        &self,
        inputs: &[&Array],
        template: &[TemplateArg],
        grid: (i32, i32, i32),
        thread_group: (i32, i32, i32),
        outputs: &[OutputArg],
        keepalive: &[&Array],
        stream: &Stream,
        residency_set: &ResidencySet,
    ) -> lisa_mlx::error::Result<Vec<Array>> {
        self.apply_with_resource_mode(
            inputs,
            template,
            grid,
            thread_group,
            outputs,
            keepalive,
            stream,
            true,
            Some(residency_set),
        )
    }

    fn apply_with_resource_mode(
        &self,
        inputs: &[&Array],
        template: &[TemplateArg],
        grid: (i32, i32, i32),
        thread_group: (i32, i32, i32),
        outputs: &[OutputArg],
        keepalive: &[&Array],
        stream: &Stream,
        queue_resident: bool,
        command_residency_set: Option<&ResidencySet>,
    ) -> lisa_mlx::error::Result<Vec<Array>> {
        let ts: Vec<&crate::array::Array> = inputs.iter().map(|a| &a.t).collect();
        let keepalive_tensors: Vec<&crate::array::Array> = keepalive.iter().map(|a| &a.t).collect();
        let tmpl: Vec<crate::jit::TemplateArg> = template
            .iter()
            .map(to_template_arg)
            .collect::<lisa_mlx::error::Result<_>>()?;
        let outs: Vec<crate::jit::OutputArg> = outputs
            .iter()
            .map(|o| -> lisa_mlx::error::Result<_> {
                Ok(crate::jit::OutputArg {
                    shape: o.shape.clone(),
                    dtype: o.dtype,
                })
            })
            .collect::<lisa_mlx::error::Result<_>>()?;
        let res = if let Some(set) = command_residency_set {
            self.inner.apply_with_command_residency_set(
                stream.runtime(),
                &ts,
                &tmpl,
                grid,
                thread_group,
                &outs,
                None,
                &keepalive_tensors,
                set,
            )?
        } else if queue_resident {
            self.inner.apply_with_residency_set(
                stream.runtime(),
                &ts,
                &tmpl,
                grid,
                thread_group,
                &outs,
                None,
                &keepalive_tensors,
            )?
        } else {
            self.inner.apply_with_keepalive(
                stream.runtime(),
                &ts,
                &tmpl,
                grid,
                thread_group,
                &outs,
                None,
                &keepalive_tensors,
            )?
        };
        Ok(res.into_iter().map(Array::new).collect())
    }

    /// Like [`Self::apply`], but writes into the caller's pre-allocated output
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
        let tmpl: Vec<crate::jit::TemplateArg> = template
            .iter()
            .map(to_template_arg)
            .collect::<lisa_mlx::error::Result<_>>()?;
        let outs: Vec<crate::jit::OutputArg> = outputs
            .iter()
            .map(|o| -> lisa_mlx::error::Result<_> {
                Ok(crate::jit::OutputArg {
                    shape: o.shape.clone(),
                    dtype: o.dtype,
                })
            })
            .collect::<lisa_mlx::error::Result<_>>()?;
        let pa: Vec<crate::array::Array> = prealloc.iter().map(|a| a.t.clone()).collect();
        let res = self.inner.apply(
            stream.runtime(),
            &ts,
            &tmpl,
            grid,
            thread_group,
            &outs,
            Some(&pa),
        )?;
        Ok(res.into_iter().map(Array::new).collect())
    }
}

/// Kept for the CLI; the runtime reports errors through `Result`.
pub fn install_error_handler() {}
