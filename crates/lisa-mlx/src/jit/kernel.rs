use std::collections::HashMap;
use std::sync::Mutex;

use objc2_metal::MTLSize;

use crate::array::Array;
use crate::runtime::{ComputePipeline, Math};

use super::MLX_UTILS_PREAMBLE;
use super::compile::type_string;
use super::{DType, Device, Tensor};
use crate::error::Result;

/// Inputs with fewer elements are passed in the `constant` address space.
const MAX_CONSTANT_ARRAY_SIZE: usize = 8;

/// Metal attributes MLX auto-declares when the source references them, in the
/// exact order of `metal_kernel.cpp`.
const METAL_ATTRIBUTES: &[(&str, &str)] = &[
    ("dispatch_quadgroups_per_threadgroup", "uint"),
    ("dispatch_simdgroups_per_threadgroup", "uint"),
    ("dispatch_threads_per_threadgroup", "uint3"),
    ("grid_origin", "uint3"),
    ("grid_size", "uint3"),
    ("quadgroup_index_in_threadgroup", "uint"),
    ("quadgroups_per_threadgroup", "uint"),
    ("simdgroup_index_in_threadgroup", "uint"),
    ("simdgroups_per_threadgroup", "uint"),
    ("thread_execution_width", "uint"),
    ("thread_index_in_quadgroup", "uint"),
    ("thread_index_in_simdgroup", "uint"),
    ("thread_index_in_threadgroup", "uint"),
    ("thread_position_in_grid", "uint3"),
    ("thread_position_in_threadgroup", "uint3"),
    ("threadgroup_position_in_grid", "uint3"),
    ("threadgroups_per_grid", "uint3"),
    ("threads_per_grid", "uint3"),
    ("threads_per_simdgroup", "uint"),
    ("threads_per_threadgroup", "uint3"),
];

/// Template argument for a Metal kernel (mirrors MLX's `mx.fast.metal_kernel`).
#[derive(Clone)]
pub enum TemplateArg {
    Dtype(&'static str, DType),
    Int(&'static str, i32),
    Bool(&'static str, bool),
}

impl TemplateArg {
    fn name(&self) -> &'static str {
        match self {
            TemplateArg::Dtype(n, _) | TemplateArg::Int(n, _) | TemplateArg::Bool(n, _) => n,
        }
    }
}

/// Output argument: shape + dtype.
#[derive(Clone)]
pub struct OutputArg {
    pub shape: Vec<i32>,
    pub dtype: DType,
}

/// The `template <...>` value list MLX appends to the kernel name and uses to
/// explicitly instantiate the kernel (`write_template`).
fn write_template(args: &[TemplateArg]) -> Result<String> {
    let mut s = String::from("<");
    for (i, arg) in args.iter().enumerate() {
        if i > 0 {
            s.push_str(", ");
        }
        match arg {
            TemplateArg::Int(_, v) => s.push_str(&v.to_string()),
            TemplateArg::Bool(_, v) => s.push_str(if *v { "1" } else { "0" }),
            TemplateArg::Dtype(_, dt) => s.push_str(type_string(*dt)?),
        }
    }
    s.push('>');
    Ok(s)
}

/// `make_template_hash`: `<>` -> `_`, `", "` -> `_`, then drop the last char.
fn make_template_hash(template_def: &str) -> String {
    let bytes = template_def.as_bytes();
    let mut s = String::with_capacity(template_def.len());
    let mut i = 0;
    while i < bytes.len() {
        let c = bytes[i] as char;
        if c == '<' || c == '>' {
            s.push('_');
        } else if c == ',' && i + 1 < bytes.len() && bytes[i + 1] == b' ' {
            s.push('_');
            i += 1;
        } else {
            s.push(c);
        }
        i += 1;
    }
    s.pop();
    s
}

#[allow(clippy::too_many_arguments)]
fn write_signature(
    func_name: &str,
    header: &str,
    source: &str,
    input_names: &[String],
    input_dtypes: &[DType],
    input_ndims: &[usize],
    input_sizes: &[usize],
    output_names: &[String],
    output_dtypes: &[DType],
    template_args: &[TemplateArg],
    attributes: &[String],
    shape_infos: &[(bool, bool, bool)],
    atomic_outputs: bool,
) -> Result<String> {
    let mut ks = String::with_capacity(header.len() + source.len() + 16384);
    ks.push_str(header);
    if !template_args.is_empty() {
        ks.push_str("template <");
        for (i, arg) in template_args.iter().enumerate() {
            let param_type = match arg {
                TemplateArg::Int(..) => "int",
                TemplateArg::Bool(..) => "bool",
                TemplateArg::Dtype(..) => "typename",
            };
            if i > 0 {
                ks.push_str(", ");
            }
            ks.push_str(param_type);
            ks.push(' ');
            ks.push_str(arg.name());
        }
        ks.push_str(">\n");
    }
    ks.push_str("[[kernel]] void ");
    ks.push_str(func_name);
    ks.push_str("(\n");

    let mut index = 0usize;
    for i in 0..input_names.len() {
        let name = &input_names[i];
        let dtype = type_string(input_dtypes[i])?;
        let location = if input_sizes[i] < MAX_CONSTANT_ARRAY_SIZE {
            "constant"
        } else {
            "device"
        };
        let reference = if input_ndims[i] == 0 { "&" } else { "*" };
        ks.push_str("  const ");
        ks.push_str(location);
        ks.push(' ');
        ks.push_str(dtype);
        ks.push_str(reference);
        ks.push(' ');
        ks.push_str(name);
        ks.push_str(&format!(" [[buffer({index})]],\n"));
        index += 1;
        if input_ndims[i] > 0 {
            let (shape, strides, ndim) = shape_infos[i];
            if shape {
                ks.push_str(&format!(
                    "  const constant int* {name}_shape [[buffer({index})]],\n"
                ));
                index += 1;
            }
            if strides {
                ks.push_str(&format!(
                    "  const constant int64_t* {name}_strides [[buffer({index})]],\n"
                ));
                index += 1;
            }
            if ndim {
                ks.push_str(&format!(
                    "  const constant int& {name}_ndim [[buffer({index})]],\n"
                ));
                index += 1;
            }
        }
    }
    for i in 0..output_names.len() {
        let name = &output_names[i];
        let ts = type_string(output_dtypes[i])?;
        ks.push_str("  device ");
        if atomic_outputs {
            ks.push_str("atomic<");
        }
        ks.push_str(ts);
        if atomic_outputs {
            ks.push('>');
        }
        ks.push_str("* ");
        ks.push_str(name);
        ks.push_str(&format!(" [[buffer({index})]]"));
        if index < input_names.len() + output_names.len() - 1 || !attributes.is_empty() {
            ks.push_str(",\n");
        } else {
            ks.push_str(") {\n");
        }
        index += 1;
    }
    for (i, attr) in attributes.iter().enumerate() {
        ks.push_str(attr);
        if i < attributes.len() - 1 {
            ks.push_str(",\n");
        } else {
            ks.push_str(") {\n");
        }
    }
    ks.push_str(source);
    ks.push_str("\n}\n");
    Ok(ks)
}

/// A JIT-compiled custom Metal kernel; the replacement for MLX's
/// `mlx_fast_metal_kernel`.
pub struct MetalKernel {
    name: String,
    input_names: Vec<String>,
    output_names: Vec<String>,
    source: String,
    header: String,
    atomic_outputs: bool,
    shape_infos: Vec<(bool, bool, bool)>,
    attributes: Vec<String>,
    pipelines: Mutex<HashMap<String, ComputePipeline>>,
    /// `apply` fast path: cheap u64 key -> (kernel name, pipeline). Avoids
    /// rebuilding the template string + hash per dispatch.
    apply_keys: Mutex<HashMap<u64, (String, ComputePipeline)>>,
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
        _ensure_row_contiguous: bool,
        atomic_outputs: bool,
    ) -> Result<Self> {
        if output_names.is_empty() {
            crate::bail!("[metal_kernel] must specify at least one output");
        }
        let shape_infos = input_names
            .iter()
            .map(|n| {
                (
                    source.contains(&format!("{n}_shape")),
                    source.contains(&format!("{n}_strides")),
                    source.contains(&format!("{n}_ndim")),
                )
            })
            .collect();
        let attributes = METAL_ATTRIBUTES
            .iter()
            .filter(|(attr, _)| source.contains(attr))
            .map(|(attr, ty)| format!("  {ty} {attr} [[{attr}]]"))
            .collect();
        Ok(Self {
            name: name.to_string(),
            input_names: input_names.iter().map(|s| s.to_string()).collect(),
            output_names: output_names.iter().map(|s| s.to_string()).collect(),
            source: source.to_string(),
            header: header.to_string(),
            atomic_outputs,
            shape_infos,
            attributes,
            pipelines: Mutex::new(HashMap::new()),
            apply_keys: Mutex::new(HashMap::new()),
        })
    }

    /// Match MLX's `device.cpp::set_compile_options` + `build_library_`
    /// exactly. On macOS >= 15 MLX sets **only** `MathModeSafe` (never
    /// `fastMathEnabled`) and additionally sets the language version
    /// (`get_metal_version`: 4.1 on macOS 27, 4.0 on 26, 3.2 on 15). Calling
    /// `setFastMathEnabled(false)` and leaving the language version at its
    /// default changed `metal::exp` codegen and made `silu_head` differ from
    /// the engine by ~6e-5 on large inputs.
    fn pipeline(
        &self,
        device: &Device,
        kernel_source: &str,
        kernel_name: &str,
    ) -> Result<ComputePipeline> {
        {
            let cache = self.pipelines.lock().unwrap();
            if let Some(p) = cache.get(kernel_name) {
                return Ok(p.clone());
            }
        }
        let mdev = device;
        let full_source = format!("{MLX_UTILS_PREAMBLE}{kernel_source}");
        let pipeline = mdev.compile_with(&full_source, kernel_name, Math::Jit)?;
        self.pipelines
            .lock()
            .unwrap()
            .insert(kernel_name.to_string(), pipeline.clone());
        Ok(pipeline)
    }

    /// Apply the kernel. `inputs` are Metal tensors; the returned tensors
    /// are allocated on the same device.
    pub fn apply(
        &self,
        device: &Device,
        inputs: &[&Tensor],
        template: &[TemplateArg],
        grid: (i32, i32, i32),
        thread_group: (i32, i32, i32),
        outputs: &[OutputArg],
        prealloc: Option<&[Tensor]>,
    ) -> Result<Vec<Tensor>> {
        if inputs.len() != self.input_names.len() {
            crate::bail!(
                "kernel {}: expected {} inputs, got {}",
                self.name,
                self.input_names.len(),
                inputs.len()
            );
        }
        let _mdev = device;

        let mut input_dtypes = Vec::with_capacity(inputs.len());
        let mut input_ndims = Vec::with_capacity(inputs.len());
        let mut input_sizes = Vec::with_capacity(inputs.len());
        for t in inputs {
            let _ = t.device();
            input_dtypes.push(t.dtype());
            input_ndims.push(t.rank());
            input_sizes.push(t.elem_count());
        }
        let output_dtypes: Vec<DType> = outputs.iter().map(|o| o.dtype).collect();

        // Cheap u64 cache key (name, template args, input dtypes + the
        // scalar/buffer flag, output dtypes). `apply` runs for every dispatch;
        // the template string + hash + `write_signature` are only needed on a
        // miss (first launch), and the name string itself only to compile.
        let mut key: u64 = 0xcbf29ce484222325;
        for b in self.name.as_bytes() {
            key = (key ^ *b as u64).wrapping_mul(0x100000001b3);
        }
        for arg in template {
            match arg {
                TemplateArg::Dtype(n, d) => {
                    for b in n.as_bytes() {
                        key = (key ^ *b as u64).wrapping_mul(0x100000001b3);
                    }
                    key = (key ^ (*d as u32 as u64)).wrapping_mul(0x100000001b3);
                }
                TemplateArg::Int(n, v) => {
                    for b in n.as_bytes() {
                        key = (key ^ *b as u64).wrapping_mul(0x100000001b3);
                    }
                    key = (key ^ (*v as u64)).wrapping_mul(0x100000001b3);
                }
                TemplateArg::Bool(n, v) => {
                    for b in n.as_bytes() {
                        key = (key ^ *b as u64).wrapping_mul(0x100000001b3);
                    }
                    key = (key ^ (*v as u64)).wrapping_mul(0x100000001b3);
                }
            }
        }
        for (i, dt) in input_dtypes.iter().enumerate() {
            key = (key ^ (*dt as u32 as u64)).wrapping_mul(0x100000001b3);
            if input_ndims[i] == 0 {
                key = (key ^ 0x11).wrapping_mul(0x100000001b3);
            } else if input_sizes[i] < MAX_CONSTANT_ARRAY_SIZE {
                key = (key ^ 0x22).wrapping_mul(0x100000001b3);
            }
        }
        for dt in output_dtypes.iter() {
            key = (key ^ (*dt as u32 as u64)).wrapping_mul(0x100000001b3);
        }

        // Pipeline fast path: `apply_keys` maps the u64 key to the kernel name
        // (which encodes dtypes/templates, as MLX builds it) + pipeline.
        {
            let cache = self.apply_keys.lock().unwrap();
            if let Some((kernel_name, p)) = cache.get(&key) {
                let (_kernel_name, p) = (kernel_name.clone(), p.clone());
                drop(cache);
                return self.apply_with_pipeline(
                    device,
                    inputs,
                    grid,
                    thread_group,
                    outputs,
                    prealloc,
                    p,
                    &input_ndims,
                );
            }
        }

        let mut kernel_name = format!("custom_kernel_{}", self.name);
        if !template.is_empty() {
            let template_def = write_template(template)?;
            let hash = make_template_hash(&template_def);
            kernel_name.push('_');
            kernel_name.push_str(&hash);
        }
        for (i, dt) in input_dtypes.iter().enumerate() {
            kernel_name.push('_');
            kernel_name.push_str(type_string(*dt)?);
            if input_ndims[i] == 0 {
                kernel_name.push('s');
            } else if input_sizes[i] < MAX_CONSTANT_ARRAY_SIZE {
                kernel_name.push('c');
            }
        }
        for dt in output_dtypes.iter() {
            kernel_name.push('_');
            kernel_name.push_str(type_string(*dt)?);
        }

        let mut kernel_source = write_signature(
            &kernel_name,
            &self.header,
            &self.source,
            &self.input_names,
            &input_dtypes,
            &input_ndims,
            &input_sizes,
            &self.output_names,
            &output_dtypes,
            template,
            &self.attributes,
            &self.shape_infos,
            self.atomic_outputs,
        )?;
        if !template.is_empty() {
            let template_def = write_template(template)?;
            let template_def = format!("{kernel_name}{template_def}");
            kernel_source.push_str(&format!(
                "\ntemplate [[host_name(\"{kernel_name}\")]] [[kernel]] decltype({template_def}) {template_def};\n"
            ));
        }

        let pipeline = self.pipeline(device, &kernel_source, &kernel_name)?;
        {
            let mut cache = self.apply_keys.lock().unwrap();
            cache.insert(key, (kernel_name.clone(), pipeline.clone()));
        }

        self.apply_with_pipeline(
            device,
            inputs,
            grid,
            thread_group,
            outputs,
            prealloc,
            pipeline,
            &input_ndims,
        )
    }

    /// The tail of `apply` once the pipeline is known: allocate outputs, bind,
    /// dispatch. Shared by the cache-hit fast path.
    fn apply_with_pipeline(
        &self,
        device: &Device,
        inputs: &[&Tensor],
        grid: (i32, i32, i32),
        thread_group: (i32, i32, i32),
        outputs: &[OutputArg],
        prealloc: Option<&[Tensor]>,
        pipeline: ComputePipeline,
        input_ndims: &[usize],
    ) -> Result<Vec<Tensor>> {
        let mdev = device;
        // Allocate outputs up front so their buffers can be bound.
        let mut out_tensors = Vec::with_capacity(outputs.len());
        for (oi, out) in outputs.iter().enumerate() {
            let count: usize = out.shape.iter().map(|&d| d.max(0) as usize).product();
            if let Some(pa) = prealloc {
                out_tensors.push(pa[oi].clone());
                continue;
            }
            let buffer =
                mdev.buffer((count) as usize * (out.dtype).size_of(), "lisa_kernel_out")?;
            let shape: Vec<usize> = out.shape.iter().map(|&d| d as usize).collect();
            out_tensors.push(Array::from_parts(mdev, buffer, &shape, out.dtype));
        }

        let guard = mdev.commands.encoder()?;
        let enc = guard.encoder();
        enc.set_pipeline(&pipeline);

        let mut index = 0usize;
        for (i, t) in inputs.iter().enumerate() {
            let (ms, layout) = t.buffer_and_layout();
            // A kernel that declares `{name}_strides` indexes through them, so a
            // non-contiguous view (e.g. a KV-cache slice) is fine; the shape /
            // stride / ndim buffers below carry its layout.
            if !layout.is_contiguous() && !self.shape_infos[i].1 {
                crate::bail!(
                    "kernel {}: input {} is not contiguous and the kernel declares no strides",
                    self.name,
                    self.input_names[i]
                );
            }
            enc.set_input(index, Some(ms), layout.offset * t.dtype().size_of());
            index += 1;
            if input_ndims[i] > 0 {
                let (want_shape, want_strides, want_ndim) = self.shape_infos[i];
                if want_shape {
                    let dims: Vec<i32> = layout.shape().iter().map(|&d| d as i32).collect();
                    enc.set_bytes_directly(
                        index,
                        std::mem::size_of_val(dims.as_slice()),
                        dims.as_ptr().cast(),
                    );
                    index += 1;
                }
                if want_strides {
                    let strides: Vec<i64> = layout.stride().iter().map(|&s| s as i64).collect();
                    enc.set_bytes_directly(
                        index,
                        std::mem::size_of_val(strides.as_slice()),
                        strides.as_ptr().cast(),
                    );
                    index += 1;
                }
                if want_ndim {
                    let ndim = input_ndims[i] as i32;
                    enc.set_bytes(index, &ndim);
                    index += 1;
                }
            }
        }
        for t in &out_tensors {
            let (ms, _) = t.buffer_and_layout();
            enc.set_output(index, Some(ms), 0);
            index += 1;
        }

        let _ = index;

        let tg_size = (thread_group.0 as usize)
            .saturating_mul(thread_group.1 as usize)
            .saturating_mul(thread_group.2 as usize);
        if tg_size > pipeline.max_total_threads_per_threadgroup() {
            crate::bail!(
                "kernel {}: thread group size {tg_size} exceeds max {}",
                self.name,
                pipeline.max_total_threads_per_threadgroup()
            );
        }
        let (gx, gy, gz) = grid;
        let (tx, ty, tz) = thread_group;
        let group = MTLSize {
            width: tx.min(gx) as usize,
            height: ty.min(gy) as usize,
            depth: tz.min(gz) as usize,
        };
        let grid_dims = MTLSize {
            width: gx.max(0) as usize,
            height: gy.max(0) as usize,
            depth: gz.max(0) as usize,
        };
        enc.dispatch_threads_size(grid_dims, group);

        Ok(out_tensors)
    }
}
