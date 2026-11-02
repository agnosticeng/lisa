use std::collections::HashMap;
use std::sync::Mutex;

use objc2_metal::MTLSize;

use crate::array::Array;
use crate::runtime::{ComputePipeline, Math, ResidencySet};

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

/// A kernel input that can be bound directly as a Metal inline scalar instead
/// of allocating a one-element tensor buffer.
#[derive(Clone, Copy)]
pub enum KernelInput<'a> {
    Tensor(&'a Tensor),
    I32(i32),
    F32(f32),
}

impl KernelInput<'_> {
    fn dtype(self) -> DType {
        match self {
            Self::Tensor(t) => t.dtype(),
            Self::I32(_) => DType::Int32,
            Self::F32(_) => DType::Float32,
        }
    }

    fn rank(self) -> usize {
        match self {
            Self::Tensor(t) => t.rank(),
            Self::I32(_) | Self::F32(_) => 0,
        }
    }

    fn elem_count(self) -> usize {
        match self {
            Self::Tensor(t) => t.elem_count(),
            Self::I32(_) | Self::F32(_) => 1,
        }
    }
}

#[derive(Clone, Copy)]
enum KernelInputs<'a> {
    Tensors(&'a [&'a Tensor]),
    Mixed(&'a [KernelInput<'a>]),
}

impl<'a> KernelInputs<'a> {
    fn len(self) -> usize {
        match self {
            Self::Tensors(inputs) => inputs.len(),
            Self::Mixed(inputs) => inputs.len(),
        }
    }

    fn get(self, index: usize) -> KernelInput<'a> {
        match self {
            Self::Tensors(inputs) => KernelInput::Tensor(inputs[index]),
            Self::Mixed(inputs) => inputs[index],
        }
    }
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
    dynamic_threadgroup_memory: bool,
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
        if index < input_names.len() + output_names.len() - 1
            || !attributes.is_empty()
            || dynamic_threadgroup_memory
        {
            ks.push_str(",\n");
        } else {
            ks.push_str(") {\n");
        }
        index += 1;
    }
    if dynamic_threadgroup_memory {
        ks.push_str("  threadgroup float* shared [[threadgroup(0)]],\n");
    }
    for (i, attr) in attributes.iter().enumerate() {
        ks.push_str(attr);
        if i < attributes.len() - 1 {
            ks.push_str(",\n");
        } else {
            ks.push_str(") {\n");
        }
    }
    if attributes.is_empty() && dynamic_threadgroup_memory {
        ks.push_str(") {\n");
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
    math: Math,
    atomic_outputs: bool,
    full_write_outputs: bool,
    threadgroup_memory_bytes: usize,
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
        Self::new_with_math(
            name,
            input_names,
            output_names,
            source,
            header,
            _ensure_row_contiguous,
            atomic_outputs,
            Math::Jit,
        )
    }

    /// Create a kernel with an explicit Metal math law. This is for kernels
    /// whose reference implementation depends on the compiler's default math
    /// mode rather than MLX's safe JIT mode.
    pub fn new_with_math(
        name: &str,
        input_names: &[&str],
        output_names: &[&str],
        source: &str,
        header: &str,
        _ensure_row_contiguous: bool,
        atomic_outputs: bool,
        math: Math,
    ) -> Result<Self> {
        Self::new_with_math_and_threadgroup_memory(
            name,
            input_names,
            output_names,
            source,
            header,
            _ensure_row_contiguous,
            atomic_outputs,
            math,
            0,
        )
    }

    /// Create a kernel with runtime-sized threadgroup scratch at index zero.
    /// The byte length is a dispatch property and does not alter emitted
    /// arithmetic, so it is intentionally excluded from the pipeline key.
    pub fn new_with_math_and_threadgroup_memory(
        name: &str,
        input_names: &[&str],
        output_names: &[&str],
        source: &str,
        header: &str,
        _ensure_row_contiguous: bool,
        atomic_outputs: bool,
        math: Math,
        threadgroup_memory_bytes: usize,
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
            math,
            atomic_outputs,
            full_write_outputs: false,
            threadgroup_memory_bytes,
            shape_infos,
            attributes,
            pipelines: Mutex::new(HashMap::new()),
            apply_keys: Mutex::new(HashMap::new()),
        })
    }

    /// Declare that every logical element of every output is assigned before
    /// it is read, for every shape admitted by this kernel's wrapper. This is
    /// an allocation contract, not a change to shader arithmetic or its cache
    /// key. Partial writes and atomic accumulation must retain the default.
    pub fn with_full_write_outputs(mut self) -> Self {
        assert!(
            !self.atomic_outputs,
            "atomic outputs require explicit initialization"
        );
        self.full_write_outputs = true;
        self
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
        let pipeline = mdev.compile_with(&full_source, kernel_name, self.math)?;
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
        self.apply_with_keepalive_mode(
            device,
            KernelInputs::Tensors(inputs),
            template,
            grid,
            thread_group,
            outputs,
            prealloc,
            &[],
            false,
            None,
        )
    }

    /// Apply a kernel with selected scalar arguments passed through Metal's
    /// inline `setBytes` path. The generated signature and arithmetic are the
    /// same as for rank-zero tensor inputs.
    pub fn apply_with_inline_inputs(
        &self,
        device: &Device,
        inputs: &[KernelInput<'_>],
        template: &[TemplateArg],
        grid: (i32, i32, i32),
        thread_group: (i32, i32, i32),
        outputs: &[OutputArg],
        prealloc: Option<&[Tensor]>,
    ) -> Result<Vec<Tensor>> {
        self.apply_with_keepalive_mode(
            device,
            KernelInputs::Mixed(inputs),
            template,
            grid,
            thread_group,
            outputs,
            prealloc,
            &[],
            false,
            None,
        )
    }

    /// Apply mixed tensor/scalar inputs while retaining buffers addressed
    /// indirectly by the kernel.
    pub fn apply_with_inline_inputs_and_keepalive(
        &self,
        device: &Device,
        inputs: &[KernelInput<'_>],
        template: &[TemplateArg],
        grid: (i32, i32, i32),
        thread_group: (i32, i32, i32),
        outputs: &[OutputArg],
        prealloc: Option<&[Tensor]>,
        keepalive: &[&Tensor],
    ) -> Result<Vec<Tensor>> {
        self.apply_with_keepalive_mode(
            device,
            KernelInputs::Mixed(inputs),
            template,
            grid,
            thread_group,
            outputs,
            prealloc,
            keepalive,
            false,
            None,
        )
    }

    /// Apply mixed tensor/scalar inputs while using an existing queue-wide
    /// residency set for indirectly addressed buffers.
    pub fn apply_with_inline_inputs_and_residency_set(
        &self,
        device: &Device,
        inputs: &[KernelInput<'_>],
        template: &[TemplateArg],
        grid: (i32, i32, i32),
        thread_group: (i32, i32, i32),
        outputs: &[OutputArg],
        prealloc: Option<&[Tensor]>,
        keepalive: &[&Tensor],
    ) -> Result<Vec<Tensor>> {
        self.apply_with_keepalive_mode(
            device,
            KernelInputs::Mixed(inputs),
            template,
            grid,
            thread_group,
            outputs,
            prealloc,
            keepalive,
            true,
            None,
        )
    }

    /// Apply a kernel while retaining buffers accessed indirectly through GPU
    /// virtual addresses until the dispatch command buffer completes.
    pub fn apply_with_keepalive(
        &self,
        device: &Device,
        inputs: &[&Tensor],
        template: &[TemplateArg],
        grid: (i32, i32, i32),
        thread_group: (i32, i32, i32),
        outputs: &[OutputArg],
        prealloc: Option<&[Tensor]>,
        keepalive: &[&Tensor],
    ) -> Result<Vec<Tensor>> {
        self.apply_with_keepalive_mode(
            device,
            KernelInputs::Tensors(inputs),
            template,
            grid,
            thread_group,
            outputs,
            prealloc,
            keepalive,
            false,
            None,
        )
    }

    /// Apply with indirect buffers already registered in a queue residency set.
    pub fn apply_with_residency_set(
        &self,
        device: &Device,
        inputs: &[&Tensor],
        template: &[TemplateArg],
        grid: (i32, i32, i32),
        thread_group: (i32, i32, i32),
        outputs: &[OutputArg],
        prealloc: Option<&[Tensor]>,
        keepalive: &[&Tensor],
    ) -> Result<Vec<Tensor>> {
        self.apply_with_keepalive_mode(
            device,
            KernelInputs::Tensors(inputs),
            template,
            grid,
            thread_group,
            outputs,
            prealloc,
            keepalive,
            true,
            None,
        )
    }

    /// Apply with an indirect-resource set attached to this command buffer.
    /// The set's allocations are made resident for this buffer's execution.
    pub fn apply_with_command_residency_set(
        &self,
        device: &Device,
        inputs: &[&Tensor],
        template: &[TemplateArg],
        grid: (i32, i32, i32),
        thread_group: (i32, i32, i32),
        outputs: &[OutputArg],
        prealloc: Option<&[Tensor]>,
        keepalive: &[&Tensor],
        residency_set: &ResidencySet,
    ) -> Result<Vec<Tensor>> {
        self.apply_with_keepalive_mode(
            device,
            KernelInputs::Tensors(inputs),
            template,
            grid,
            thread_group,
            outputs,
            prealloc,
            keepalive,
            true,
            Some(residency_set),
        )
    }

    fn apply_with_keepalive_mode(
        &self,
        device: &Device,
        inputs: KernelInputs<'_>,
        template: &[TemplateArg],
        grid: (i32, i32, i32),
        thread_group: (i32, i32, i32),
        outputs: &[OutputArg],
        prealloc: Option<&[Tensor]>,
        keepalive: &[&Tensor],
        queue_resident: bool,
        command_residency_set: Option<&ResidencySet>,
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
        for i in 0..inputs.len() {
            let input = inputs.get(i);
            if let KernelInput::Tensor(t) = input {
                let _ = t.device();
            }
            input_dtypes.push(input.dtype());
            input_ndims.push(input.rank());
            input_sizes.push(input.elem_count());
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
                    keepalive,
                    queue_resident,
                    command_residency_set,
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
            self.threadgroup_memory_bytes != 0,
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
            keepalive,
            queue_resident,
            command_residency_set,
        )
    }

    /// The tail of `apply` once the pipeline is known: allocate outputs, bind,
    /// dispatch. Shared by the cache-hit fast path.
    fn apply_with_pipeline(
        &self,
        device: &Device,
        inputs: KernelInputs<'_>,
        grid: (i32, i32, i32),
        thread_group: (i32, i32, i32),
        outputs: &[OutputArg],
        prealloc: Option<&[Tensor]>,
        pipeline: ComputePipeline,
        input_ndims: &[usize],
        keepalive: &[&Tensor],
        queue_resident: bool,
        command_residency_set: Option<&ResidencySet>,
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
            let bytes = count * out.dtype.size_of();
            let buffer = if self.full_write_outputs {
                mdev.buffer_for_full_gpu_write(bytes, "lisa_kernel_out")?
            } else {
                mdev.buffer_for_gpu_write(bytes, "lisa_kernel_out")?
            };
            let shape: Vec<usize> = out.shape.iter().map(|&d| d as usize).collect();
            out_tensors.push(Array::from_parts(mdev, buffer, &shape, out.dtype));
        }

        {
            let guard = mdev.commands.encoder()?;
            let enc = guard.encoder();
            if let Some(set) = command_residency_set {
                enc.use_residency_set(set);
            }
            enc.set_pipeline(&pipeline);
            for tensor in keepalive {
                if queue_resident {
                    enc.retain_resident_until_completion(tensor.buffer());
                } else {
                    enc.retain_until_completion(tensor.buffer());
                }
            }

            let mut index = 0usize;
            for i in 0..inputs.len() {
                let input = inputs.get(i);
                match input {
                    KernelInput::Tensor(t) => {
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
                                let dims: Vec<i32> =
                                    layout.shape().iter().map(|&d| d as i32).collect();
                                enc.set_bytes_directly(
                                    index,
                                    std::mem::size_of_val(dims.as_slice()),
                                    dims.as_ptr().cast(),
                                );
                                index += 1;
                            }
                            if want_strides {
                                let strides: Vec<i64> =
                                    layout.stride().iter().map(|&s| s as i64).collect();
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
                    KernelInput::I32(value) => {
                        if input_ndims[i] != 0 {
                            crate::bail!(
                                "kernel {}: inline integer input {} is not scalar",
                                self.name,
                                self.input_names[i]
                            );
                        }
                        enc.set_bytes(index, &value);
                        index += 1;
                    }
                    KernelInput::F32(value) => {
                        if input_ndims[i] != 0 {
                            crate::bail!(
                                "kernel {}: inline float input {} is not scalar",
                                self.name,
                                self.input_names[i]
                            );
                        }
                        enc.set_bytes(index, &value);
                        index += 1;
                    }
                }
            }
            for t in &out_tensors {
                let (ms, _) = t.buffer_and_layout();
                enc.set_output(index, Some(ms), 0);
                index += 1;
            }

            if self.threadgroup_memory_bytes != 0 {
                enc.set_threadgroup_memory_length(0, self.threadgroup_memory_bytes);
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
        } // drop the encoder guard (commits + updates the cross-encoder fence)

        Ok(out_tensors)
    }
}
