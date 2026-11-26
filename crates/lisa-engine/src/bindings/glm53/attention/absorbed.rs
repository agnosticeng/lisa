use crate::bindings::glm53::attention::mode::*;
use crate::core::graph::PlanCache;
use crate::{bindings::BindingDevice, execution::DeviceContext};
use lisa_backend::TensorAccess;
use lisa_models::glm53::absorbed_attention::{Request, Spec, Stage};
pub fn run<D: BindingDevice>(
    request: Request,
    inputs: &[&D::Tensor],
    context: &DeviceContext<D>,
) -> anyhow::Result<Vec<D::Tensor>> {
    context.with_slot(
        "glm53.absorbed_attention.plans",
        || std::cell::RefCell::new(PlanCache::<D, Request>::new(context.device().clone(), 64)),
        |slot| {
            let mut p = slot.borrow_mut();
            let recipe = request.clone();
            Ok(p.try_get_or_prepare(request, || recipe.graph())?
                .run(inputs, &mut [])?
                .into_iter()
                .collect())
        },
    )
}
fn execute<D: BindingDevice + lisa_backend::ManagedDevice>(
    spec: Spec,
    inputs: &[&D::Tensor],
    context: &DeviceContext<D>,
) -> anyhow::Result<(D::Tensor, Option<D::Tensor>, Option<D::Tensor>)> {
    let mut prepared = vec![];
    let mut query = None;
    let mut latent = None;
    let mut projected = None;
    let mut output = None;
    for stage in spec.stages()? {
        let request = Request::Stage(spec.clone(), stage);
        match stage {
            Stage::Prepare => prepared = run(request, inputs, context)?,
            Stage::Query => {
                query = Some(
                    run(
                        request,
                        &[&prepared[0], &prepared[1], &prepared[2], &prepared[3]],
                        context,
                    )?
                    .remove(0),
                )
            }
            Stage::Boundary => context.device().execution_boundary(),
            Stage::Attend => {
                let mut out = run(
                    request,
                    &[
                        &prepared[0],
                        query.as_ref().unwrap(),
                        &prepared[4],
                        &prepared[5],
                        &prepared[1],
                        &prepared[2],
                        &prepared[3],
                    ],
                    context,
                )?
                .into_iter();
                if spec.fused_decode() {
                    projected = out.next();
                    latent = out.next();
                } else {
                    latent = out.next();
                }
            }
            Stage::Value => {
                projected = Some(
                    run(
                        request,
                        &[
                            latent.as_ref().unwrap(),
                            &prepared[1],
                            &prepared[2],
                            &prepared[3],
                        ],
                        context,
                    )?
                    .remove(0),
                )
            }
            Stage::Output => {
                output = Some(run(request, &[projected.as_ref().unwrap()], context)?.remove(0))
            }
        }
    }
    Ok((
        output.unwrap(),
        if spec.fused_decode() && !spec.capture {
            None
        } else {
            query
        },
        latent,
    ))
}
#[allow(clippy::too_many_arguments)]
fn attention<D: BindingDevice + lisa_backend::ManagedDevice>(
    q: &D::Tensor,
    w: &D::Tensor,
    s: &D::Tensor,
    b: &D::Tensor,
    group: usize,
    cache: &D::Tensor,
    selected: Option<&D::Tensor>,
    prefix: usize,
    scale: f32,
    context: &DeviceContext<D>,
    capture: bool,
    fused: bool,
    capture_fused: bool,
    boundaries: bool,
    flash: Option<usize>,
) -> Option<(D::Tensor, Option<D::Tensor>, Option<D::Tensor>)> {
    let dummy = if selected.is_none() {
        Some(context.with_slot(
            "glm53.attention.dummy",
            || {
                context.device().pull(
                    lisa_backend::HostTensor {
                        bytes: &[0u8; 32],
                        shape: &[1, 1, 8],
                        dtype: lisa_backend::ScalarType::I32,
                    }
                    .into(),
                    Default::default(),
                )
            },
            |value| value.as_ref().ok().cloned(),
        )?)
    } else {
        None
    };
    let ids = selected
        .or(dummy.as_ref())
        .expect("selected or dummy indices");
    let inputs = [q, w, s, b, cache, ids];
    let device = context.device();
    let spec = Spec {
        shapes: inputs.iter().map(|a| a.shape().to_vec()).collect(),
        dtypes: inputs
            .iter()
            .map(|a| device.dtype(a))
            .collect::<Option<Vec<_>>>()?,
        group,
        selected: selected.is_some(),
        prefix,
        scale: scale.to_bits(),
        capture,
        fused,
        capture_fused,
        boundaries,
        native: glm53_dsa_native_enabled(),
        dense_prefix: GLM53_DSA_VALUE_DENSE_PREFIX.with(std::cell::Cell::get),
        flash,
    };
    if !spec.validate() {
        return None;
    }
    match execute(spec, &inputs, context) {
        Ok(v) => Some(v),
        Err(e) => {
            eprintln!("[absorbed-attention-plan] {e:#}");
            None
        }
    }
}
#[allow(clippy::too_many_arguments)]
pub fn dsa_absorbed_attention_stages_inner<D: BindingDevice + lisa_backend::ManagedDevice>(
    q: &D::Tensor,
    w: &D::Tensor,
    s: &D::Tensor,
    b: &D::Tensor,
    group: usize,
    cache: &D::Tensor,
    selected: Option<&D::Tensor>,
    prefix: usize,
    scale: f32,
    context: &DeviceContext<D>,
    capture: bool,
    fused: bool,
    capture_fused: bool,
    boundaries: bool,
) -> Option<(D::Tensor, Option<D::Tensor>, Option<D::Tensor>)> {
    attention(
        q,
        w,
        s,
        b,
        group,
        cache,
        selected,
        prefix,
        scale,
        context,
        capture,
        fused,
        capture_fused,
        boundaries,
        None,
    )
}
/// GLM DSA attention in absorbed MLA form, reading the packed MLX affine
/// `kv_b` weights directly. `q=[B,heads,T,256]`, `kv_weight=[32768,128]`
/// packed U32 (8-bit, 512 inputs), scales/biases `[32768,8]` F32,
/// `compact_latents=[B,tc,512]` F16, and `selected=[B,T,M]` absolute token
/// ids. Returns `[B,heads,T,256]` F32. Native single-row decode fuses attention
/// and value projection through 2051 selected rows; longer private MTP windows
/// use the reference's staged indexed-lora kernel followed by its value projection.
pub fn dsa_absorbed_attention<D: BindingDevice + lisa_backend::ManagedDevice>(
    q: &D::Tensor,
    kv_weight: &D::Tensor,
    kv_scales: &D::Tensor,
    kv_biases: &D::Tensor,
    group_size: usize,
    compact_latents: &D::Tensor,
    selected: &D::Tensor,
    scale: f32,
    context: &DeviceContext<D>,
) -> Option<D::Tensor> {
    dsa_absorbed_attention_stages_inner(
        q,
        kv_weight,
        kv_scales,
        kv_biases,
        group_size,
        compact_latents,
        Some(selected),
        0,
        scale,
        context,
        false,
        false,
        false,
        false,
    )
    .map(|(out, _, _)| out)
}

/// Explicit fused one-token decode path. The fused schedule is separate from
/// the reference's staged MLX-affine path and must only be selected by callers that have
/// validated its numerics for their weight representation.
#[allow(clippy::too_many_arguments)]
pub fn dsa_absorbed_attention_decode<D: BindingDevice + lisa_backend::ManagedDevice>(
    q: &D::Tensor,
    kv_weight: &D::Tensor,
    kv_scales: &D::Tensor,
    kv_biases: &D::Tensor,
    group_size: usize,
    compact_latents: &D::Tensor,
    selected: Option<&D::Tensor>,
    selected_valid_prefix: usize,
    scale: f32,
    context: &DeviceContext<D>,
) -> Option<D::Tensor> {
    dsa_absorbed_attention_stages_inner(
        q,
        kv_weight,
        kv_scales,
        kv_biases,
        group_size,
        compact_latents,
        selected,
        selected_valid_prefix,
        scale,
        context,
        false,
        true,
        false,
        false,
    )
    .map(|(out, _, _)| out)
}

/// Capture the intermediate weighted latent from the native fused decode
/// kernel. This diagnostic entry point adds one output buffer and is never
/// used by production inference.
#[allow(clippy::too_many_arguments)]
pub fn dsa_absorbed_attention_decode_capture<D: BindingDevice + lisa_backend::ManagedDevice>(
    q: &D::Tensor,
    kv_weight: &D::Tensor,
    kv_scales: &D::Tensor,
    kv_biases: &D::Tensor,
    group_size: usize,
    compact_latents: &D::Tensor,
    selected: Option<&D::Tensor>,
    selected_valid_prefix: usize,
    scale: f32,
    context: &DeviceContext<D>,
) -> Option<(D::Tensor, D::Tensor)> {
    dsa_absorbed_attention_stages_inner(
        q,
        kv_weight,
        kv_scales,
        kv_biases,
        group_size,
        compact_latents,
        selected,
        selected_valid_prefix,
        scale,
        context,
        false,
        true,
        true,
        false,
    )
    .and_then(|(out, _, fused_lora)| Some((out, fused_lora?)))
}

/// DSA decode variant for selectors that guarantee a live-cache prefix in
/// every selected row. Entries at and after `selected_valid_prefix` remain
/// bounds-checked, so the causal-tail padding can use `-1`/`UINT32_MAX`.
/// Prefill and arbitrary selections should use [`dsa_absorbed_attention`].
#[allow(clippy::too_many_arguments)]
pub fn dsa_absorbed_attention_with_valid_prefix<D: BindingDevice + lisa_backend::ManagedDevice>(
    q: &D::Tensor,
    kv_weight: &D::Tensor,
    kv_scales: &D::Tensor,
    kv_biases: &D::Tensor,
    group_size: usize,
    compact_latents: &D::Tensor,
    selected: &D::Tensor,
    selected_valid_prefix: usize,
    scale: f32,
    context: &DeviceContext<D>,
    prefill_encoder_boundaries: bool,
) -> Option<D::Tensor> {
    dsa_absorbed_attention_stages_inner(
        q,
        kv_weight,
        kv_scales,
        kv_biases,
        group_size,
        compact_latents,
        Some(selected),
        selected_valid_prefix,
        scale,
        context,
        false,
        false,
        false,
        prefill_encoder_boundaries,
    )
    .map(|(out, _, _)| out)
}

/// Dense-equivalent one-token DSA decode with contiguous cache rows supplied
/// implicitly. This avoids uploading `[0..tc)` as a selected-ID tensor.
#[allow(clippy::too_many_arguments)]
pub fn dsa_absorbed_attention_dense<D: BindingDevice + lisa_backend::ManagedDevice>(
    q: &D::Tensor,
    kv_weight: &D::Tensor,
    kv_scales: &D::Tensor,
    kv_biases: &D::Tensor,
    group_size: usize,
    compact_latents: &D::Tensor,
    scale: f32,
    context: &DeviceContext<D>,
) -> Option<D::Tensor> {
    dsa_absorbed_attention_stages_inner(
        q,
        kv_weight,
        kv_scales,
        kv_biases,
        group_size,
        compact_latents,
        None,
        0,
        scale,
        context,
        false,
        false,
        false,
        false,
    )
    .map(|(out, _, _)| out)
}

/// Dense compact-cache attention with QK-lowrank and attention-weighted latent
/// intermediates returned for non-perturbing differential capture. At short
/// windows this uses the same dispatch sequence as `dsa_absorbed_attention_dense`;
/// the capture flag only prevents the long-window fused decode branch.
#[allow(clippy::too_many_arguments)]
pub fn dsa_absorbed_attention_dense_stages<D: BindingDevice + lisa_backend::ManagedDevice>(
    q: &D::Tensor,
    kv_weight: &D::Tensor,
    kv_scales: &D::Tensor,
    kv_biases: &D::Tensor,
    group_size: usize,
    compact_latents: &D::Tensor,
    scale: f32,
    context: &DeviceContext<D>,
) -> Option<(D::Tensor, D::Tensor, D::Tensor)> {
    dsa_absorbed_attention_stages_inner(
        q,
        kv_weight,
        kv_scales,
        kv_biases,
        group_size,
        compact_latents,
        None,
        0,
        scale,
        context,
        true,
        false,
        false,
        false,
    )
    .and_then(|(out, q_low, lora)| Some((out, q_low?, lora?)))
}

/// Same DSA path with the QK low-rank and attention-weighted latent outputs
/// returned for differential capture. Runtime callers should use
/// `dsa_absorbed_attention` unless they need those intermediate tensors.
#[allow(clippy::too_many_arguments)]
pub fn dsa_absorbed_attention_stages<D: BindingDevice + lisa_backend::ManagedDevice>(
    q: &D::Tensor,
    kv_weight: &D::Tensor,
    kv_scales: &D::Tensor,
    kv_biases: &D::Tensor,
    group_size: usize,
    compact_latents: &D::Tensor,
    selected: &D::Tensor,
    scale: f32,
    context: &DeviceContext<D>,
    prefill_encoder_boundaries: bool,
) -> Option<(D::Tensor, D::Tensor, D::Tensor)> {
    dsa_absorbed_attention_stages_inner(
        q,
        kv_weight,
        kv_scales,
        kv_biases,
        group_size,
        compact_latents,
        Some(selected),
        0,
        scale,
        context,
        true,
        false,
        false,
        prefill_encoder_boundaries,
    )
    .and_then(|(out, q_low, lora)| Some((out, q_low?, lora?)))
}

#[allow(clippy::too_many_arguments)]
pub fn dsa_absorbed_attention_flash_causal_prefill_stages<
    D: BindingDevice + lisa_backend::ManagedDevice,
>(
    q: &D::Tensor,
    w: &D::Tensor,
    s: &D::Tensor,
    b: &D::Tensor,
    group: usize,
    cache: &D::Tensor,
    position: usize,
    scale: f32,
    context: &DeviceContext<D>,
    boundaries: bool,
) -> Option<(D::Tensor, D::Tensor, D::Tensor)> {
    let (y, q, l) = attention(
        q,
        w,
        s,
        b,
        group,
        cache,
        None,
        0,
        scale,
        context,
        true,
        false,
        false,
        boundaries,
        Some(position),
    )?;
    Some((y, q?, l?))
}
pub fn value_rows<D: BindingDevice + lisa_backend::ManagedDevice>(
    x: &D::Tensor,
    w: &D::Tensor,
    s: &D::Tensor,
    b: &D::Tensor,
    group: usize,
    heads: usize,
    context: &DeviceContext<D>,
) -> Option<D::Tensor> {
    run(
        Request::Value {
            rows: x.dim(0) as usize,
            heads,
            group,
            native: glm53_dsa_native_enabled(),
            prefix: GLM53_DSA_VALUE_DENSE_PREFIX.with(std::cell::Cell::get),
        },
        &[x, w, s, b],
        context,
    )
    .ok()?
    .into_iter()
    .next()
}
