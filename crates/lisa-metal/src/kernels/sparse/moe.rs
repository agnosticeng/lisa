//! Moe operations with the pinned DS4 arithmetic.
use super::*;


pub(super) const MOE_GATHER_SOURCE: &crate::shader_pack::Source =
    crate::kernel_source!("glm53/moe_gather.metal");
pub(super) const MOE_PAIR_SWIGLU_SOURCE: &crate::shader_pack::Source =
    crate::kernel_source!("glm53/moe_pair_swiglu.metal");
pub(super) const MOE_GATHER_ADDR_SOURCE: &crate::shader_pack::Source =
    crate::kernel_source!("glm53/moe_gather_addr.metal");
pub(super) const MOE_GATHER_ADDR_DS4_SOURCE: &crate::shader_pack::Source =
    crate::kernel_source!("glm53/moe_gather_addr_ds4.metal");
pub(super) const MOE_GATHER_SUM8_ADDR_DS4_SOURCE: &crate::shader_pack::Source =
    crate::kernel_source!("glm53/moe_gather_sum8_addr_ds4.metal");
pub(super) const MOE_GATHER_SUM8_ADDR_SOURCE: &crate::shader_pack::Source =
    crate::kernel_source!("glm53/moe_gather_sum8_addr.metal");
pub(super) const MOE_PAIR_SWIGLU_ADDR_SOURCE: &crate::shader_pack::Source =
    crate::kernel_source!("glm53/moe_pair_swiglu_addr.metal");
pub(super) const MOE_EXPERT_ROUTE_MAP_SOURCE: &crate::shader_pack::Source =
    crate::kernel_source!("glm53/moe_expert_route_map.metal");
pub(super) const MOE_MAPPED_RESOURCE_HELPER: &crate::shader_pack::Source =
    crate::kernel_source!("glm53/moe_mapped_resource.metal");
pub(super) const MOE_PAIR_SWIGLU_MAPPED_BODY: &crate::shader_pack::Source =
    crate::kernel_source!("glm53/moe_pair_swiglu_mapped.metal");
pub(super) const MOE_GATHER_SUM8_MAPPED_BODY: &crate::shader_pack::Source =
    crate::kernel_source!("glm53/moe_gather_sum8_mapped.metal");
pub(super) const MOE_SUM8_SOURCE: &crate::shader_pack::Source = crate::kernel_source!("glm53/moe_sum8.metal");
pub(super) const MOE_MAPPED_RESOURCE_INPUTS: [&str; 16] = [
    "resource_00",
    "resource_01",
    "resource_02",
    "resource_03",
    "resource_04",
    "resource_05",
    "resource_06",
    "resource_07",
    "resource_08",
    "resource_09",
    "resource_10",
    "resource_11",
    "resource_12",
    "resource_13",
    "resource_14",
    "resource_15",
];

/// Build the per-expert `(resource slot, byte offset)` table used by the
/// ordinary buffer-bound MoE kernels. Each expert contributes the same nine
/// fields as the address-table path: gate/up/down weight, scale, and bias.
/// The high byte of each Int64 entry stores the resource slot; the lower 56
/// bits store the byte offset into that encoder-bound MTLBuffer.
pub fn moe_mapped_resource_table(fields: &[&Array]) -> Option<(Array, Vec<Array>)> {
    if fields.is_empty() || fields.len() % 9 != 0 {
        return None;
    }

    let mut resource_slots = std::collections::HashMap::<usize, u8>::new();
    let mut resources = Vec::<Array>::new();
    let mut entries = Vec::<i64>::with_capacity(fields.len());
    for field in fields {
        let buffer = field.t.buffer();
        let address = field.gpu_address()?;
        let offset = address.checked_sub(buffer.gpu_address())?;
        let offset = usize::try_from(offset).ok()?;
        let byte_len = field.size().checked_mul(field.dtype().size_of())?;
        if offset.checked_add(byte_len)? > buffer.length() || offset >= (1usize << 56) {
            return None;
        }

        let id = field.backing_buffer_id();
        let slot = if let Some(&slot) = resource_slots.get(&id) {
            slot
        } else {
            if resources.len() == MOE_MAPPED_RESOURCE_INPUTS.len() || buffer.length() < 8 {
                return None;
            }
            let slot = u8::try_from(resources.len()).ok()?;
            let base =
                Array::from_buffer_offset(field.t.buffer().clone(), 0, &[8], Dtype::Uint8).ok()?;
            resource_slots.insert(id, slot);
            resources.push(base);
            slot
        };
        let packed = (u64::from(slot) << 56) | offset as u64;
        entries.push(packed as i64);
    }
    if resources.is_empty() {
        return None;
    }
    Some((
        Array::from_slice(&entries, &[entries.len() as i32]),
        resources,
    ))
}

/// The MLX quantized-matmul preamble chain — the SAME headers the jit qmv
/// dispatch composes (`jit/matmul.rs`), so the gathered kernel calls the very
/// same `qdot`/`load_vector` helpers as the certified per-expert qmv.
pub(super) fn moe_gather_header() -> String {
    format!(
        "{}{}{}",
        crate::jit::MLX_GEMM_PREAMBLE,
        crate::jit::MLX_QUANTIZED_UTILS_PREAMBLE,
        crate::jit::MLX_QUANTIZED_PREAMBLE,
    )
}

pub(super) fn moe_gather_kernel() -> &'static Option<MetalKernel> {
    static K: std::sync::OnceLock<Option<MetalKernel>> = std::sync::OnceLock::new();
    K.get_or_init(|| {
        let header = moe_gather_header();
        MetalKernel::new_with_math(
            "glm53_moe_gather_qmv_ds4_default_v4",
            &[
                "n_slots",
                "groups_per_slot",
                "out_rows",
                "in_vec_size",
                "x_per_slot",
                "w",
                "scales",
                "biases",
                "x",
                "selected",
            ],
            &["y"],
            MOE_GATHER_SOURCE,
            &header,
            true,
            false,
            crate::runtime::Math::CompilerDefault,
        )
        .ok()
    })
}

pub(super) fn moe_pair_swiglu_kernel() -> &'static Option<MetalKernel> {
    static K: std::sync::OnceLock<Option<MetalKernel>> = std::sync::OnceLock::new();
    K.get_or_init(|| {
        let header = moe_gather_header();
        MetalKernel::new_with_math(
            "glm53_moe_pair_swiglu_ds4_default_v2",
            &[
                "n_slots",
                "groups_per_slot",
                "mid_dim",
                "in_vec_size",
                "slots_per_token",
                "limit",
                "w",
                "scales",
                "biases",
                "x",
                "selected",
                "weights",
            ],
            &["y"],
            MOE_PAIR_SWIGLU_SOURCE,
            &header,
            true,
            false,
            crate::runtime::Math::CompilerDefault,
        )
        .ok()
    })
}

pub(super) fn moe_gather_addr_kernel() -> &'static Option<MetalKernel> {
    static K: std::sync::OnceLock<Option<MetalKernel>> = std::sync::OnceLock::new();
    K.get_or_init(|| {
        let header = moe_gather_header();
        MetalKernel::new_with_math(
            "glm53_moe_gather_qmv_addr_ds4_default_v3",
            &[
                "n_slots",
                "groups_per_slot",
                "out_rows",
                "in_vec_size",
                "x_per_slot",
                "n_experts",
                "address_table",
                "x",
                "selected",
            ],
            &["y"],
            MOE_GATHER_ADDR_SOURCE,
            &header,
            true,
            false,
            crate::runtime::Math::CompilerDefault,
        )
        .ok()
    })
}

pub(super) fn moe_gather_addr_ds4_kernel() -> &'static Option<MetalKernel> {
    static K: std::sync::OnceLock<Option<MetalKernel>> = std::sync::OnceLock::new();
    K.get_or_init(|| {
        let header = moe_gather_header();
        MetalKernel::new_with_math(
            "glm53_moe_gather_qmv_addr_ds4_exact_v1",
            &[
                "n_slots",
                "out_rows",
                "in_vec_size",
                "x_per_slot",
                "n_experts",
                "address_table",
                "x",
                "selected",
            ],
            &["y"],
            MOE_GATHER_ADDR_DS4_SOURCE,
            &header,
            true,
            false,
            crate::runtime::Math::CompilerDefault,
        )
        .ok()
    })
}

pub(super) fn moe_gather_sum8_addr_ds4_kernel() -> &'static Option<MetalKernel> {
    static K: std::sync::OnceLock<Option<MetalKernel>> = std::sync::OnceLock::new();
    K.get_or_init(|| {
        let header = moe_gather_header();
        MetalKernel::new_with_math(
            "glm53_moe_gather_sum8_addr_ds4_exact_v5",
            &[
                "n_tokens",
                "out_rows",
                "in_vec_size",
                "n_experts",
                "has_shared",
                "address_table",
                "x",
                "selected",
                "shared",
            ],
            &["y"],
            MOE_GATHER_SUM8_ADDR_DS4_SOURCE,
            &header,
            true,
            false,
            crate::runtime::Math::CompilerDefault,
        )
        // Full-write contract: Every row/token is assigned, including zero sums for invalid expert IDs.
        .map(MetalKernel::with_full_write_outputs)
        .ok()
    })
}

pub(super) fn moe_gather_sum8_addr_kernel() -> &'static Option<MetalKernel> {
    static K: std::sync::OnceLock<Option<MetalKernel>> = std::sync::OnceLock::new();
    K.get_or_init(|| {
        let header = moe_gather_header();
        match MetalKernel::new_with_math(
            "glm53_moe_gather_sum8_addr_ds4_default_v9",
            &[
                "n_tokens",
                "groups_per_token",
                "out_rows",
                "in_vec_size",
                "slots_per_token",
                "n_experts",
                "address_table",
                "x",
                "selected",
                "has_shared",
                "shared",
                "has_shared_down",
                "shared_down_in_vec_size",
                "shared_down_x",
                "shared_down_w",
                "shared_down_scales",
                "shared_down_biases",
            ],
            &["out"],
            MOE_GATHER_SUM8_ADDR_SOURCE,
            &header,
            true,
            false,
            crate::runtime::Math::CompilerDefault,
        ) {
            Ok(kernel) => Some(kernel),
            Err(error) => {
                eprintln!("[glm53-moe-sum8] kernel compilation failed: {error}");
                None
            }
        }
    })
}

pub(super) fn moe_pair_swiglu_addr_kernel() -> &'static Option<MetalKernel> {
    static K: std::sync::OnceLock<Option<MetalKernel>> = std::sync::OnceLock::new();
    K.get_or_init(|| {
        let header = moe_gather_header();
        MetalKernel::new_with_math(
            "glm53_moe_pair_swiglu_addr_ds4_default_v9",
            &[
                "n_slots",
                "groups_per_slot",
                "mid_dim",
                "in_vec_size",
                "slots_per_token",
                "n_experts",
                "limit",
                "address_table",
                "x",
                "selected",
                "weights",
            ],
            &["y"],
            MOE_PAIR_SWIGLU_ADDR_SOURCE,
            &header,
            true,
            false,
            crate::runtime::Math::CompilerDefault,
        )
        .ok()
    })
}

pub(super) fn moe_expert_route_map_kernel() -> &'static Option<MetalKernel> {
    static K: std::sync::OnceLock<Option<MetalKernel>> = std::sync::OnceLock::new();
    K.get_or_init(|| {
        MetalKernel::new_with_math(
            "glm53_moe_expert_route_map_ds4_v1",
            &["n_tokens", "top_k", "n_experts", "tile_size", "selected"],
            &["route_ids", "counts", "work_count", "work_items"],
            MOE_EXPERT_ROUTE_MAP_SOURCE,
            "",
            true,
            false,
            crate::runtime::Math::CompilerDefault,
        )
        .ok()
    })
}

/// Build the expert-major route lists and compact tile work map used by the
/// grouped affine-Q2 MoE prefill. Route IDs remain flattened token/slot
/// indices, so grouped projections can write results back in the existing
/// token-major order. This is a mapping primitive; the production per-slot
/// MoE path remains unchanged until grouped projection parity is established.
pub fn moe_expert_route_map(
    selected: &Array,
    n_tokens: usize,
    top_k: usize,
    n_experts: usize,
    tile_size: usize,
    stream: &Stream,
) -> Option<(Array, Array, Array, Array)> {
    let kernel = moe_expert_route_map_kernel().as_ref()?;
    if selected.rank() != 1
        || selected.dtype() != Dtype::Int32
        || n_tokens == 0
        || top_k == 0
        || n_experts == 0
        || n_experts > 1024
        || tile_size == 0
        || n_tokens > i32::MAX as usize
        || top_k > i32::MAX as usize
        || tile_size > i32::MAX as usize
        || selected.dim(0) as usize != n_tokens.checked_mul(top_k)?
        || n_tokens.checked_mul(n_experts)? > i32::MAX as usize
    {
        return None;
    }
    selected.eval_enqueue().ok()?;

    // Router top-k selection produces unique expert IDs within each token,
    // which bounds each expert's route list to n_tokens entries.
    let route_id_count = n_tokens.checked_mul(n_experts)?;
    let max_work_items = selected.dim(0).checked_div(tile_size as i32)? as usize
        + usize::from(selected.dim(0) as usize % tile_size != 0)
        + n_experts;
    let work_item_values = max_work_items.checked_mul(2)?;
    if work_item_values > i32::MAX as usize {
        return None;
    }
    let inputs = [
        KernelInput::I32(n_tokens as i32),
        KernelInput::I32(top_k as i32),
        KernelInput::I32(n_experts as i32),
        KernelInput::I32(tile_size as i32),
        KernelInput::Tensor(&selected.t),
    ];
    let outputs = [
        OutputArg {
            shape: vec![route_id_count as i32],
            dtype: Dtype::Int32,
        },
        OutputArg {
            shape: vec![n_experts as i32],
            dtype: Dtype::Int32,
        },
        OutputArg {
            shape: vec![1],
            dtype: Dtype::Int32,
        },
        OutputArg {
            shape: vec![work_item_values as i32],
            dtype: Dtype::Int32,
        },
    ];
    let mut result = kernel
        .apply_with_inline_inputs(
            &inputs,
            &[],
            (n_experts as i32, 1, 1),
            (n_experts as i32, 1, 1),
            &outputs,
            stream,
        )
        .ok()?
        .into_iter();
    Some((
        result.next()?,
        result.next()?,
        result.next()?,
        result.next()?,
    ))
}

pub(super) fn moe_pair_swiglu_mapped_kernel() -> &'static Option<MetalKernel> {
    static K: std::sync::OnceLock<Option<MetalKernel>> = std::sync::OnceLock::new();
    K.get_or_init(|| {
        let header = format!("{}\n{}", moe_gather_header(), MOE_MAPPED_RESOURCE_HELPER);
        let source = MOE_PAIR_SWIGLU_MAPPED_BODY;
        let mut inputs = vec![
            "n_slots",
            "groups_per_slot",
            "mid_dim",
            "in_vec_size",
            "slots_per_token",
            "n_experts",
            "limit",
            "address_table",
            "x",
            "selected",
            "weights",
        ];
        inputs.extend(MOE_MAPPED_RESOURCE_INPUTS.iter().copied());
        MetalKernel::new_with_math(
            "glm53_moe_pair_swiglu_mapped_ds4_exact_v2",
            &inputs,
            &["y"],
            source,
            &header,
            true,
            false,
            crate::runtime::Math::CompilerDefault,
        )
        .ok()
    })
}

pub(super) fn moe_gather_sum8_mapped_kernel() -> &'static Option<MetalKernel> {
    static K: std::sync::OnceLock<Option<MetalKernel>> = std::sync::OnceLock::new();
    K.get_or_init(|| {
        let header = format!("{}\n{}", moe_gather_header(), MOE_MAPPED_RESOURCE_HELPER);
        let source = MOE_GATHER_SUM8_MAPPED_BODY;
        let mut inputs = vec![
            "n_tokens",
            "out_rows",
            "in_vec_size",
            "n_experts",
            "has_shared",
            "address_table",
            "x",
            "selected",
            "shared",
        ];
        inputs.extend(MOE_MAPPED_RESOURCE_INPUTS.iter().copied());
        MetalKernel::new_with_math(
            "glm53_moe_gather_sum8_mapped_ds4_exact_v2",
            &inputs,
            &["out"],
            source,
            &header,
            true,
            false,
            crate::runtime::Math::CompilerDefault,
        )
        .ok()
    })
}

pub(super) fn moe_sum8_kernel() -> &'static Option<MetalKernel> {
    static K: std::sync::OnceLock<Option<MetalKernel>> = std::sync::OnceLock::new();
    K.get_or_init(|| {
        MetalKernel::new_with_math(
            "glm53_moe_sum8_ds4_default_v2",
            &["n_tokens", "width", "src"],
            &["out"],
            MOE_SUM8_SOURCE,
            EXACT_HEADER,
            true,
            false,
            crate::runtime::Math::CompilerDefault,
        )
        // Full-write contract: Each token group covers every output column.
        .map(MetalKernel::with_full_write_outputs)
        .ok()
    })
}

/// Fused gate/up projection through a compact MLX-expert GPU address table.
/// `address_table` stores nine Int64 addresses per expert; the first six are
/// the separate gate/up weight, scale, and bias buffers. `keepalive` holds the
/// underlying arrays through completion because address indirection is
/// invisible to Metal's normal resource tracker.
pub fn moe_pair_swiglu_addr(
    address_table: &Array,
    x: &Array,
    selected: &Array,
    weights: &Array,
    keepalive: &[&Array],
    mid_dim: usize,
    slots_per_token: usize,
    limit: f32,
    metadata_dtype: Dtype,
    residency_set: Option<&crate::runtime::ResidencySet>,
    stream: &Stream,
) -> Option<Array> {
    if address_table.rank() != 1
        || address_table.dtype() != Dtype::Int64
        || address_table.dim(0) <= 0
        || address_table.dim(0) % 9 != 0
        || !matches!(metadata_dtype, Dtype::Bfloat16 | Dtype::Float32)
        || selected.rank() != 1
        || selected.dtype() != Dtype::Int32
        || weights.rank() != 1
        || weights.dtype() != Dtype::Float32
        || x.rank() != 2
        || !matches!(x.dtype(), Dtype::Float32 | Dtype::Float16)
        || mid_dim == 0
        || slots_per_token == 0
    {
        return None;
    }
    let n_slots = selected.dim(0) as usize;
    let in_vec_size = x.dim(1) as usize;
    let n_experts = address_table.dim(0) as usize / 9;
    if n_slots == 0
        || n_slots % slots_per_token != 0
        || x.dim(0) != (n_slots / slots_per_token) as i32
        || weights.dim(0) as usize != n_slots
        || in_vec_size == 0
        || in_vec_size % 512 != 0
        || n_experts == 0
    {
        return None;
    }
    let kernel = moe_pair_swiglu_addr_kernel().as_ref()?;
    for a in [address_table, x, selected, weights] {
        a.eval_enqueue().ok()?;
    }
    // Four output rows per SIMD group, two SIMD groups per threadgroup.
    let groups_per_slot = mid_dim.div_ceil(8);
    let grid_y = i32::try_from(groups_per_slot.checked_mul(n_slots)?.checked_mul(2)?).ok()?;
    let inputs = [
        KernelInput::I32(n_slots as i32),
        KernelInput::I32(groups_per_slot as i32),
        KernelInput::I32(mid_dim as i32),
        KernelInput::I32(in_vec_size as i32),
        KernelInput::I32(slots_per_token as i32),
        KernelInput::I32(n_experts as i32),
        KernelInput::F32(limit),
        KernelInput::Tensor(&address_table.t),
        KernelInput::Tensor(&x.t),
        KernelInput::Tensor(&selected.t),
        KernelInput::Tensor(&weights.t),
    ];
    let outputs = [OutputArg {
        shape: vec![n_slots as i32, mid_dim as i32],
        dtype: Dtype::Float32,
    }];
    let result = if residency_set.is_some() {
        kernel.apply_with_inline_inputs_and_residency_set(
            &inputs,
            &[
                TemplateArg::Dtype("T", Dtype::Float32),
                TemplateArg::Dtype("X", x.dtype()),
                TemplateArg::Dtype("M", metadata_dtype),
            ],
            (32, grid_y, 1),
            (32, 2, 1),
            &outputs,
            keepalive,
            stream,
        )
    } else {
        kernel.apply_with_inline_inputs_and_keepalive(
            &inputs,
            &[
                TemplateArg::Dtype("T", Dtype::Float32),
                TemplateArg::Dtype("M", metadata_dtype),
            ],
            (32, grid_y, 1),
            (32, 2, 1),
            &outputs,
            keepalive,
            stream,
        )
    };
    result.ok()?.into_iter().next()
}

pub(super) fn padded_moe_resource_inputs(resources: &[Array]) -> Option<Vec<Array>> {
    if resources.is_empty() || resources.len() > MOE_MAPPED_RESOURCE_INPUTS.len() {
        return None;
    }
    if resources
        .iter()
        .any(|resource| resource.dtype() != Dtype::Uint8 || resource.shape() != [8])
    {
        return None;
    }
    let mut inputs = resources.to_vec();
    let first = inputs.first()?.clone();
    inputs.resize(MOE_MAPPED_RESOURCE_INPUTS.len(), first);
    Some(inputs)
}

/// Fused gate/up SwiGLU with MLX expert buffers bound as ordinary Metal
/// resources. The table stores one packed resource slot and byte offset for
/// each of the nine expert fields. This mirrors ds4's normal encoder bindings
/// and avoids both raw GPU-address dereferences and selected-ID readback.
pub fn moe_pair_swiglu_mapped(
    resource_table: &Array,
    x: &Array,
    selected: &Array,
    weights: &Array,
    resources: &[Array],
    mid_dim: usize,
    slots_per_token: usize,
    limit: f32,
    metadata_dtype: Dtype,
    stream: &Stream,
) -> Option<Array> {
    let kernel = moe_pair_swiglu_mapped_kernel().as_ref()?;
    if resource_table.rank() != 1
        || resource_table.dtype() != Dtype::Int64
        || resource_table.dim(0) <= 0
        || resource_table.dim(0) % 9 != 0
        || !matches!(metadata_dtype, Dtype::Bfloat16 | Dtype::Float32)
        || selected.rank() != 1
        || selected.dtype() != Dtype::Int32
        || weights.rank() != 1
        || weights.dtype() != Dtype::Float32
        || x.rank() != 2
        || x.dtype() != Dtype::Float32
        || mid_dim == 0
        || slots_per_token == 0
    {
        return None;
    }
    let bound = padded_moe_resource_inputs(resources)?;
    let n_slots = selected.dim(0) as usize;
    let in_vec_size = x.dim(1) as usize;
    let n_experts = resource_table.dim(0) as usize / 9;
    if n_slots == 0
        || n_slots % slots_per_token != 0
        || x.dim(0) != (n_slots / slots_per_token) as i32
        || weights.dim(0) as usize != n_slots
        || in_vec_size == 0
        || in_vec_size % 512 != 0
        || n_experts == 0
    {
        return None;
    }

    let slots = Array::from_int(n_slots as i32);
    let groups_per_slot = Array::from_int(mid_dim.div_ceil(4) as i32);
    let mid = Array::from_int(mid_dim as i32);
    let k = Array::from_int(in_vec_size as i32);
    let spt = Array::from_int(slots_per_token as i32);
    let ne = Array::from_int(n_experts as i32);
    let lim = Array::from_f32_as(limit, Dtype::Float32);
    let mut inputs: Vec<&Array> = vec![
        &slots,
        &groups_per_slot,
        &mid,
        &k,
        &spt,
        &ne,
        &lim,
        resource_table,
        x,
        selected,
        weights,
    ];
    inputs.extend(bound.iter());
    for input in &inputs {
        input.eval_enqueue().ok()?;
    }
    let outputs = [OutputArg {
        shape: vec![n_slots as i32, mid_dim as i32],
        dtype: Dtype::Float32,
    }];
    let result = match kernel.apply(
        &inputs,
        &[
            TemplateArg::Dtype("T", Dtype::Float32),
            TemplateArg::Dtype("M", metadata_dtype),
        ],
        (32, (mid_dim.div_ceil(4) * n_slots * 4) as i32, 1),
        (32, 4, 1),
        &outputs,
        stream,
    ) {
        Ok(result) => result,
        Err(error) => {
            eprintln!("[glm53-moe-mapped] pair dispatch declined: {error}");
            return None;
        }
    };
    result.into_iter().next()
}

/// Per-route down projection through the same compact nine-address-per-expert
/// table. Output is `[n_slots, out_rows]` in F32.
pub fn moe_gather_qmv_addr(
    address_table: &Array,
    x: &Array,
    selected: &Array,
    keepalive: &[&Array],
    out_rows: usize,
    x_per_slot: bool,
    metadata_dtype: Dtype,
    residency_set: Option<&crate::runtime::ResidencySet>,
    stream: &Stream,
) -> Option<Array> {
    let kernel = moe_gather_addr_kernel().as_ref()?;
    if address_table.rank() != 1
        || address_table.dtype() != Dtype::Int64
        || address_table.dim(0) <= 0
        || address_table.dim(0) % 9 != 0
        || !matches!(metadata_dtype, Dtype::Bfloat16 | Dtype::Float32)
        || selected.rank() != 1
        || selected.dtype() != Dtype::Int32
        || x.rank() != 2
        || !matches!(x.dtype(), Dtype::Float32 | Dtype::Float16)
        || out_rows == 0
        || out_rows % 8 != 0
    {
        return None;
    }
    let n_slots = selected.dim(0) as usize;
    let in_vec_size = x.dim(1) as usize;
    let n_experts = address_table.dim(0) as usize / 9;
    if n_slots == 0
        || x.dim(0) as usize != if x_per_slot { n_slots } else { 1 }
        || in_vec_size == 0
        || in_vec_size % 512 != 0
        || n_experts == 0
    {
        return None;
    }
    for a in [address_table, x, selected] {
        a.eval_enqueue().ok()?;
    }
    let slots = Array::from_int(n_slots as i32);
    let groups_per_slot = Array::from_int(out_rows.div_ceil(8) as i32);
    let rows = Array::from_int(out_rows as i32);
    let k = Array::from_int(in_vec_size as i32);
    let xps = Array::from_int(i32::from(x_per_slot));
    let ne = Array::from_int(n_experts as i32);
    let inputs: [&Array; 9] = [
        &slots,
        &groups_per_slot,
        &rows,
        &k,
        &xps,
        &ne,
        address_table,
        x,
        selected,
    ];
    let outputs = [OutputArg {
        shape: vec![n_slots as i32, out_rows as i32],
        dtype: Dtype::Float32,
    }];
    let result = if residency_set.is_some() {
        kernel.apply_with_residency_set(
            &inputs,
            &[
                TemplateArg::Dtype("T", Dtype::Float32),
                TemplateArg::Dtype("X", x.dtype()),
                TemplateArg::Dtype("M", metadata_dtype),
            ],
            (32, (out_rows.div_ceil(8) * n_slots * 2) as i32, 1),
            (32, 2, 1),
            &outputs,
            keepalive,
            stream,
        )
    } else {
        kernel.apply_with_keepalive(
            &inputs,
            &[
                TemplateArg::Dtype("T", Dtype::Float32),
                TemplateArg::Dtype("X", x.dtype()),
                TemplateArg::Dtype("M", metadata_dtype),
            ],
            (32, (out_rows.div_ceil(8) * n_slots * 2) as i32, 1),
            (32, 2, 1),
            &outputs,
            keepalive,
            stream,
        )
    };
    result.ok()?.into_iter().next()
}

/// Q2 affine per-route down projection with DS4's exact one-token arithmetic
/// schedule. Every output/route pair uses one SIMD group, whose lanes visit
/// `k = lane + 32n`, FMA-accumulate in order, then perform one SIMD reduction.
pub fn moe_gather_qmv_addr_ds4(
    address_table: &Array,
    x: &Array,
    selected: &Array,
    keepalive: &[&Array],
    out_rows: usize,
    x_per_slot: bool,
    metadata_dtype: Dtype,
    residency_set: Option<&crate::runtime::ResidencySet>,
    stream: &Stream,
) -> Option<Array> {
    let kernel = moe_gather_addr_ds4_kernel().as_ref()?;
    if address_table.rank() != 1
        || address_table.dtype() != Dtype::Int64
        || address_table.dim(0) <= 0
        || address_table.dim(0) % 9 != 0
        || !matches!(metadata_dtype, Dtype::Bfloat16 | Dtype::Float32)
        || selected.rank() != 1
        || selected.dtype() != Dtype::Int32
        || x.rank() != 2
        || x.dtype() != Dtype::Float32
        || out_rows == 0
        || out_rows > i32::MAX as usize
    {
        return None;
    }
    let n_slots = selected.dim(0) as usize;
    let in_vec_size = x.dim(1) as usize;
    let n_experts = address_table.dim(0) as usize / 9;
    if n_slots == 0
        || n_slots > i32::MAX as usize
        || x.dim(0) as usize != if x_per_slot { n_slots } else { 1 }
        || in_vec_size == 0
        || in_vec_size % 128 != 0
        || in_vec_size > i32::MAX as usize
        || n_experts == 0
        || n_experts > i32::MAX as usize
    {
        return None;
    }
    for input in [address_table, x, selected] {
        input.eval_enqueue().ok()?;
    }
    let slots = Array::from_int(n_slots as i32);
    let rows = Array::from_int(out_rows as i32);
    let k = Array::from_int(in_vec_size as i32);
    let xps = Array::from_int(i32::from(x_per_slot));
    let ne = Array::from_int(n_experts as i32);
    let inputs: [&Array; 8] = [&slots, &rows, &k, &xps, &ne, address_table, x, selected];
    let outputs = [OutputArg {
        shape: vec![n_slots as i32, out_rows as i32],
        dtype: Dtype::Float32,
    }];
    let grid_x = i32::try_from(out_rows.checked_mul(32)?).ok()?;
    let grid_y = i32::try_from(n_slots).ok()?;
    let template = [TemplateArg::Dtype("M", metadata_dtype)];
    let result = if residency_set.is_some() {
        kernel.apply_with_residency_set(
            &inputs,
            &template,
            (grid_x, grid_y, 1),
            (32, 1, 1),
            &outputs,
            keepalive,
            stream,
        )
    } else {
        kernel.apply_with_keepalive(
            &inputs,
            &template,
            (grid_x, grid_y, 1),
            (32, 1, 1),
            &outputs,
            keepalive,
            stream,
        )
    };
    result.ok()?.into_iter().next()
}

/// DS4-exact Q2 down projection for eight routed experts in one dispatch.
/// Each SIMD group computes one expert's DS4 affine matvec for four output
/// rows; the group leader adds the eight reduced values in route order.
pub fn moe_gather_qmv_sum8_addr_ds4(
    address_table: &Array,
    x: &Array,
    selected: &Array,
    keepalive: &[&Array],
    shared: Option<&Array>,
    out_rows: usize,
    metadata_dtype: Dtype,
    residency_set: Option<&crate::runtime::ResidencySet>,
    stream: &Stream,
) -> Option<Array> {
    let kernel = moe_gather_sum8_addr_ds4_kernel().as_ref()?;
    if address_table.rank() != 1
        || address_table.dtype() != Dtype::Int64
        || address_table.dim(0) <= 0
        || address_table.dim(0) % 9 != 0
        || !matches!(metadata_dtype, Dtype::Bfloat16 | Dtype::Float32)
        || selected.rank() != 1
        || selected.dtype() != Dtype::Int32
        || selected.dim(0) <= 0
        || selected.dim(0) % 8 != 0
        || x.rank() != 2
        || x.dtype() != Dtype::Float32
        || x.dim(0) != selected.dim(0)
        || out_rows == 0
        || out_rows > i32::MAX as usize
    {
        return None;
    }
    let n_slots = selected.dim(0) as usize;
    let n_tokens = n_slots / 8;
    let in_vec_size = x.dim(1) as usize;
    let n_experts = address_table.dim(0) as usize / 9;
    if in_vec_size == 0
        || in_vec_size % 128 != 0
        || in_vec_size > i32::MAX as usize
        || n_experts == 0
        || n_experts > i32::MAX as usize
        || n_tokens > i32::MAX as usize
    {
        return None;
    }
    for input in [address_table, x, selected] {
        input.eval_enqueue().ok()?;
    }
    let (has_shared, shared_values) = if let Some(shared) = shared {
        if shared.rank() != 2
            || shared.dtype() != Dtype::Float32
            || shared.dim(0) != n_tokens as i32
            || shared.dim(1) != out_rows as i32
        {
            return None;
        }
        (1, shared.contiguous().ok()?)
    } else {
        // The shader never reads this binding when has_shared is zero.
        (0, x.clone())
    };
    let inputs = [
        KernelInput::I32(n_tokens as i32),
        KernelInput::I32(out_rows as i32),
        KernelInput::I32(in_vec_size as i32),
        KernelInput::I32(n_experts as i32),
        KernelInput::I32(has_shared),
        KernelInput::Tensor(&address_table.t),
        KernelInput::Tensor(&x.t),
        KernelInput::Tensor(&selected.t),
        KernelInput::Tensor(&shared_values.t),
    ];
    let outputs = [OutputArg {
        shape: vec![n_tokens as i32, out_rows as i32],
        dtype: Dtype::Float32,
    }];
    let grid_x = i32::try_from(out_rows.div_ceil(4).checked_mul(32)?).ok()?;
    let grid_y = i32::try_from(n_tokens.checked_mul(8)?).ok()?;
    let template = [TemplateArg::Dtype("M", metadata_dtype)];
    let result = if residency_set.is_some() {
        kernel.apply_with_inline_inputs_and_residency_set(
            &inputs,
            &template,
            (grid_x, grid_y, 1),
            (32, 8, 1),
            &outputs,
            keepalive,
            stream,
        )
    } else {
        kernel.apply_with_inline_inputs_and_keepalive(
            &inputs,
            &template,
            (grid_x, grid_y, 1),
            (32, 8, 1),
            &outputs,
            keepalive,
            stream,
        )
    };
    result.ok()?.into_iter().next()
}

/// Selected-expert QMV for eight routed rows per token. This follows ds4's
/// down-projection schedule: each output tile visits expert slots in order,
/// keeps their F32 reductions local, and writes one `[token, out_rows]` row.
/// `x` contains one already route-weighted input row per selected expert.
pub fn moe_gather_qmv_sum8_addr(
    address_table: &Array,
    x: &Array,
    selected: &Array,
    keepalive: &[&Array],
    out_rows: usize,
    metadata_dtype: Dtype,
    residency_set: Option<&crate::runtime::ResidencySet>,
    stream: &Stream,
) -> Option<Array> {
    moe_gather_qmv_sum8_addr_impl(
        address_table,
        x,
        selected,
        keepalive,
        None,
        None,
        out_rows,
        metadata_dtype,
        residency_set,
        stream,
    )
}

/// Selected-expert sum8 down projection with the shared-expert output added
/// after the ordered routed sum in the same dispatch.
pub fn moe_gather_qmv_sum8_addr_add_shared(
    address_table: &Array,
    x: &Array,
    selected: &Array,
    keepalive: &[&Array],
    shared: &Array,
    out_rows: usize,
    metadata_dtype: Dtype,
    residency_set: Option<&crate::runtime::ResidencySet>,
    stream: &Stream,
) -> Option<Array> {
    moe_gather_qmv_sum8_addr_impl(
        address_table,
        x,
        selected,
        keepalive,
        Some(shared),
        None,
        out_rows,
        metadata_dtype,
        residency_set,
        stream,
    )
}

/// Selected-expert sum8 down projection with the shared Q8/group-64 down
/// projection fused after the routed sum. The shared activation is one row per
/// token and uses the stock MLX packed affine layout.
pub fn moe_gather_qmv_sum8_addr_add_shared_down(
    address_table: &Array,
    x: &Array,
    selected: &Array,
    keepalive: &[&Array],
    shared_x: &Array,
    shared_w: &Array,
    shared_scales: &Array,
    shared_biases: &Array,
    out_rows: usize,
    metadata_dtype: Dtype,
    residency_set: Option<&crate::runtime::ResidencySet>,
    stream: &Stream,
) -> Option<Array> {
    moe_gather_qmv_sum8_addr_impl(
        address_table,
        x,
        selected,
        keepalive,
        None,
        Some((shared_x, shared_w, shared_scales, shared_biases)),
        out_rows,
        metadata_dtype,
        residency_set,
        stream,
    )
}

pub(super) fn moe_gather_qmv_sum8_addr_impl(
    address_table: &Array,
    x: &Array,
    selected: &Array,
    keepalive: &[&Array],
    shared: Option<&Array>,
    shared_down: Option<(&Array, &Array, &Array, &Array)>,
    out_rows: usize,
    metadata_dtype: Dtype,
    residency_set: Option<&crate::runtime::ResidencySet>,
    stream: &Stream,
) -> Option<Array> {
    let kernel = moe_gather_sum8_addr_kernel().as_ref()?;
    if address_table.rank() != 1
        || address_table.dtype() != Dtype::Int64
        || address_table.dim(0) <= 0
        || address_table.dim(0) % 9 != 0
        || !matches!(metadata_dtype, Dtype::Bfloat16 | Dtype::Float32)
        || selected.rank() != 1
        || selected.dtype() != Dtype::Int32
        || x.rank() != 2
        || !matches!(x.dtype(), Dtype::Float32 | Dtype::Float16)
        || out_rows == 0
        || out_rows % 8 != 0
    {
        return None;
    }
    let n_slots = selected.dim(0) as usize;
    let in_vec_size = x.dim(1) as usize;
    let n_experts = address_table.dim(0) as usize / 9;
    if n_slots == 0
        || n_slots % 8 != 0
        || x.dim(0) as usize != n_slots
        || in_vec_size == 0
        || in_vec_size % 512 != 0
        || n_experts == 0
    {
        return None;
    }
    for a in [address_table, x, selected] {
        a.eval_enqueue().ok()?;
    }
    let n_tokens = n_slots / 8;
    let shared_input = if let Some(shared) = shared {
        if shared.rank() != 2
            || shared.dtype() != Dtype::Float32
            || shared.dim(0) != n_tokens as i32
            || shared.dim(1) != out_rows as i32
        {
            return None;
        }
        shared.contiguous().ok()?
    } else {
        Array::from_slice(&[0.0f32; 8], &[8])
    };
    let (
        shared_down_in_vec_size,
        shared_down_x,
        shared_down_w,
        shared_down_scales,
        shared_down_biases,
    ) = if let Some((shared_x, shared_w, shared_scales, shared_biases)) = shared_down {
        let shared_in = shared_x.dim(-1) as usize;
        if shared_x.rank() != 2
            || shared_x.dtype() != Dtype::Float32
            || shared_x.dim(0) != n_tokens as i32
            || shared_in == 0
            || shared_in % 256 != 0
            || shared_w.rank() != 2
            || shared_w.dtype() != Dtype::Uint32
            || shared_w.dim(0) != out_rows as i32
            || shared_w.dim(1) as usize != shared_in / 4
            || shared_scales.rank() != 2
            || shared_scales.dtype() != Dtype::Float32
            || shared_scales.dim(0) != out_rows as i32
            || shared_scales.dim(1) as usize != shared_in / 64
            || shared_biases.shape() != shared_scales.shape()
            || shared_biases.dtype() != Dtype::Float32
        {
            return None;
        }
        (
            shared_in,
            shared_x.contiguous().ok()?,
            shared_w.contiguous().ok()?,
            shared_scales.contiguous().ok()?,
            shared_biases.contiguous().ok()?,
        )
    } else {
        let placeholder = Array::from_slice(&[0.0f32; 8], &[8]);
        (
            0,
            placeholder.clone(),
            placeholder.clone(),
            placeholder.clone(),
            placeholder,
        )
    };
    let inputs = [
        KernelInput::I32(n_tokens as i32),
        KernelInput::I32(out_rows.div_ceil(4) as i32),
        KernelInput::I32(out_rows as i32),
        KernelInput::I32(in_vec_size as i32),
        KernelInput::I32(8),
        KernelInput::I32(n_experts as i32),
        KernelInput::Tensor(&address_table.t),
        KernelInput::Tensor(&x.t),
        KernelInput::Tensor(&selected.t),
        KernelInput::I32(i32::from(shared.is_some())),
        KernelInput::Tensor(&shared_input.t),
        KernelInput::I32(i32::from(shared_down.is_some())),
        KernelInput::I32(shared_down_in_vec_size as i32),
        KernelInput::Tensor(&shared_down_x.t),
        KernelInput::Tensor(&shared_down_w.t),
        KernelInput::Tensor(&shared_down_scales.t),
        KernelInput::Tensor(&shared_down_biases.t),
    ];
    for a in [
        &shared_input,
        &shared_down_x,
        &shared_down_w,
        &shared_down_scales,
        &shared_down_biases,
    ] {
        a.eval_enqueue().ok()?;
    }
    let outputs = [OutputArg {
        shape: vec![n_tokens as i32, out_rows as i32],
        dtype: Dtype::Float32,
    }];
    let result = if residency_set.is_some() {
        kernel.apply_with_inline_inputs_and_residency_set(
            &inputs,
            &[
                TemplateArg::Dtype("T", Dtype::Float32),
                TemplateArg::Dtype("X", x.dtype()),
                TemplateArg::Dtype("M", metadata_dtype),
            ],
            (32, (out_rows.div_ceil(4) * n_tokens * 4) as i32, 1),
            (32, 4, 1),
            &outputs,
            keepalive,
            stream,
        )
    } else {
        kernel.apply_with_inline_inputs_and_keepalive(
            &inputs,
            &[
                TemplateArg::Dtype("T", Dtype::Float32),
                TemplateArg::Dtype("X", x.dtype()),
                TemplateArg::Dtype("M", metadata_dtype),
            ],
            (32, (out_rows.div_ceil(4) * n_tokens * 4) as i32, 1),
            (32, 4, 1),
            &outputs,
            keepalive,
            stream,
        )
    };
    match result {
        Ok(outputs) => outputs.into_iter().next(),
        Err(error) => {
            eprintln!("[glm53-moe-sum8] dispatch failed: {error}");
            None
        }
    }
}

/// Selected-expert sum8 down projection with each MLX backing buffer supplied
/// as an ordinary encoder input, matching ds4's normal resource binding.
pub fn moe_gather_qmv_sum8_mapped(
    resource_table: &Array,
    x: &Array,
    selected: &Array,
    resources: &[Array],
    out_rows: usize,
    metadata_dtype: Dtype,
    stream: &Stream,
) -> Option<Array> {
    let kernel = moe_gather_sum8_mapped_kernel().as_ref()?;
    if resource_table.rank() != 1
        || resource_table.dtype() != Dtype::Int64
        || resource_table.dim(0) <= 0
        || resource_table.dim(0) % 9 != 0
        || !matches!(metadata_dtype, Dtype::Bfloat16 | Dtype::Float32)
        || selected.rank() != 1
        || selected.dtype() != Dtype::Int32
        || x.rank() != 2
        || x.dtype() != Dtype::Float32
        || out_rows == 0
        || out_rows > i32::MAX as usize
    {
        return None;
    }
    let bound = padded_moe_resource_inputs(resources)?;
    let n_slots = selected.dim(0) as usize;
    let in_vec_size = x.dim(1) as usize;
    let n_experts = resource_table.dim(0) as usize / 9;
    if n_slots == 0
        || n_slots % 8 != 0
        || x.dim(0) as usize != n_slots
        || in_vec_size == 0
        || in_vec_size % 128 != 0
        || in_vec_size > i32::MAX as usize
        || n_experts == 0
        || n_experts > i32::MAX as usize
    {
        return None;
    }

    let n_tokens = n_slots / 8;
    let n_tokens_a = Array::from_int(n_tokens as i32);
    let rows = Array::from_int(out_rows as i32);
    let k = Array::from_int(in_vec_size as i32);
    let ne = Array::from_int(n_experts as i32);
    let has_shared = Array::from_int(0);
    let shared = Array::from_slice(&[0.0f32; 8], &[8]);
    let mut inputs: Vec<&Array> = vec![
        &n_tokens_a,
        &rows,
        &k,
        &ne,
        &has_shared,
        resource_table,
        x,
        selected,
        &shared,
    ];
    inputs.extend(bound.iter());
    for input in &inputs {
        input.eval_enqueue().ok()?;
    }
    let outputs = [OutputArg {
        shape: vec![n_tokens as i32, out_rows as i32],
        dtype: Dtype::Float32,
    }];
    let grid_x = i32::try_from(out_rows.div_ceil(4).checked_mul(32)?).ok()?;
    let grid_y = i32::try_from(n_tokens.checked_mul(8)?).ok()?;
    let result = match kernel.apply(
        &inputs,
        &[
            TemplateArg::Dtype("T", Dtype::Float32),
            TemplateArg::Dtype("M", metadata_dtype),
        ],
        (grid_x, grid_y, 1),
        (32, 8, 1),
        &outputs,
        stream,
    ) {
        Ok(result) => result,
        Err(error) => {
            eprintln!("[glm53-moe-mapped] down sum8 dispatch declined: {error}");
            return None;
        }
    };
    result.into_iter().next()
}

/// Reduce eight ordered routed-expert rows per token into one F32 row. This
/// follows ds4's `kernel_dsv4_moe_sum8_f32` and replaces seven separate tensor
/// additions after the down projections.
pub fn moe_sum8(experts: &Array, n_tokens: usize, width: usize, stream: &Stream) -> Option<Array> {
    let kernel = moe_sum8_kernel().as_ref()?;
    if experts.rank() != 2
        || experts.dtype() != Dtype::Float32
        || n_tokens == 0
        || width == 0
        || n_tokens > i32::MAX as usize
        || width > i32::MAX as usize
        || experts.dim(0) as usize != n_tokens.checked_mul(8)?
        || experts.dim(1) as usize != width
    {
        return None;
    }
    let grid_x = i32::try_from(n_tokens.checked_mul(256)?).ok()?;
    experts.eval_enqueue().ok()?;
    let n_tokens_a = Array::from_int(n_tokens as i32);
    let width_a = Array::from_int(width as i32);
    let inputs: [&Array; 3] = [&n_tokens_a, &width_a, experts];
    let outputs = [OutputArg {
        shape: vec![n_tokens as i32, width as i32],
        dtype: Dtype::Float32,
    }];
    kernel
        .apply(&inputs, &[], (grid_x, 1, 1), (256, 1, 1), &outputs, stream)
        .ok()?
        .into_iter()
        .next()
}

/// Gathered quantized matvec over stacked per-expert pools. `selected` is
/// read on the GPU, and `x_per_slot` controls whether each route uses the
/// shared input row or its corresponding route row. GLM-5.3 uses F32 for both
/// input and quantization metadata. Returns `None` when declined.
pub fn moe_gather_qmv(
    pools_w: &Array,
    pools_s: &Array,
    pools_b: &Array,
    x: &Array,
    selected: &Array,
    out_rows: usize,
    x_per_slot: bool,
    dtype: Dtype,
    stream: &Stream,
) -> Option<Array> {
    let Some(kernel) = moe_gather_kernel().as_ref() else {
        eprintln!("[moe-gather-decline] Metal gathered-QMV kernel failed to initialize");
        return None;
    };
    // Materialize every input before dispatch: in the lazy zero-flush decode
    // step nothing else forces these, and the kernel reads raw buffers.
    for a in [pools_w, pools_s, pools_b, x, selected] {
        if let Err(e) = a.eval_enqueue() {
            eprintln!("[moe-gather-decline] {e}");
            return None;
        }
    }
    let n_slots = selected.dim(0) as usize;
    let k = x.dim(x.rank() as i32 - 1) as usize;
    if n_slots == 0 || out_rows == 0 || out_rows % 8 != 0 || k == 0 || k % 512 != 0 {
        return None;
    }
    let template = [TemplateArg::Dtype("T", dtype)];
    let na = Array::from_int(n_slots as i32);
    let gps = out_rows / 8;
    let gps_a = Array::from_int(gps as i32);
    let orows = Array::from_int(out_rows as i32);
    let k_a = Array::from_int(k as i32);
    let xps = Array::from_int(if x_per_slot { 1 } else { 0 });
    let inputs: [&Array; 10] = [
        &na, &gps_a, &orows, &k_a, &xps, pools_w, pools_s, pools_b, x, selected,
    ];
    let outputs = [OutputArg {
        shape: vec![n_slots as i32, out_rows as i32],
        dtype,
    }];
    let all: Vec<&Array> = inputs.iter().map(|a| &**a).collect();
    let out = match kernel.apply(
        &all,
        &template,
        (32, (gps * n_slots * 2) as i32, 1),
        (32, 2, 1),
        &outputs,
        stream,
    ) {
        Ok(o) => o,
        Err(e) => {
            eprintln!("[moe-gather-decline] {e}");
            return None;
        }
    };
    Some(out.into_iter().next()?)
}

/// Gathered gate/up projection fused with ds4's clamped SwiGLU and route
/// weighting. MLX packed Q2 weights remain in their native layout; `selected`
/// and `weights` are read per slot on the GPU. Returns `[n_slots, mid_dim]`.
pub fn moe_pair_swiglu(
    pools_w: &Array,
    pools_s: &Array,
    pools_b: &Array,
    x: &Array,
    selected: &Array,
    weights: &Array,
    mid_dim: usize,
    slots_per_token: usize,
    limit: f32,
    stream: &Stream,
) -> Option<Array> {
    let kernel = moe_pair_swiglu_kernel().as_ref()?;
    if selected.rank() != 1
        || weights.rank() != 1
        || x.rank() != 2
        || pools_w.rank() != 3
        || pools_s.rank() != 3
        || pools_b.rank() != 3
        || mid_dim == 0
        || slots_per_token == 0
        || pools_w.dim(1) != (2 * mid_dim) as i32
        || pools_s.dtype() != Dtype::Float32
        || pools_b.dtype() != Dtype::Float32
        || x.dtype() != Dtype::Float32
        || weights.dtype() != Dtype::Float32
        || selected.dtype() != Dtype::Int32
    {
        return None;
    }
    let n_slots = selected.dim(0) as usize;
    let in_vec_size = x.dim(1) as usize;
    if n_slots == 0
        || n_slots % slots_per_token != 0
        || x.dim(0) != (n_slots / slots_per_token) as i32
        || in_vec_size == 0
        || in_vec_size % 512 != 0
        || pools_s.dim(1) != (2 * mid_dim) as i32
        || pools_s.dim(2) != (in_vec_size / 128) as i32
    {
        return None;
    }
    for a in [pools_w, pools_s, pools_b, x, selected, weights] {
        if a.eval_enqueue().is_err() {
            return None;
        }
    }
    let slots = Array::from_int(n_slots as i32);
    let groups_per_slot = Array::from_int(mid_dim.div_ceil(8) as i32);
    let mid = Array::from_int(mid_dim as i32);
    let k = Array::from_int(in_vec_size as i32);
    let spt = Array::from_int(slots_per_token as i32);
    let lim = Array::from_f32_as(limit, Dtype::Float32);
    let inputs: [&Array; 12] = [
        &slots,
        &groups_per_slot,
        &mid,
        &k,
        &spt,
        &lim,
        pools_w,
        pools_s,
        pools_b,
        x,
        selected,
        weights,
    ];
    let outputs = [OutputArg {
        shape: vec![n_slots as i32, mid_dim as i32],
        dtype: Dtype::Float32,
    }];
    let all: Vec<&Array> = inputs.iter().map(|a| &**a).collect();
    let out = kernel
        .apply(
            &all,
            &[TemplateArg::Dtype("T", Dtype::Float32)],
            (32, (mid_dim.div_ceil(8) * n_slots * 2) as i32, 1),
            (32, 2, 1),
            &outputs,
            stream,
        )
        .ok()?;
    out.into_iter().next()
}