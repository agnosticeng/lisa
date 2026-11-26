use crate::{bindings::BindingDevice, execution::DeviceContext};
use crate::{cache::indexer::IndexerTape, core::graph::PlanCache};
use lisa_models::glm53::attention::{self as model, Norm, Request};
pub fn run<D: BindingDevice>(
    context: &DeviceContext<D>,
    request: Request,
    inputs: &[&D::Tensor],
) -> anyhow::Result<Vec<D::Tensor>> {
    context.with_slot(
        "glm53.attention_ops.plans",
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
pub fn one<D: BindingDevice>(
    context: &DeviceContext<D>,
    request: Request,
    inputs: &[&D::Tensor],
) -> anyhow::Result<D::Tensor> {
    Ok(run(context, request, inputs)?.remove(0))
}
#[allow(clippy::too_many_arguments)]
pub fn normalize<D: BindingDevice>(
    context: &DeviceContext<D>,
    inputs: [&D::Tensor; 4],
    batch: usize,
    tokens: usize,
    epsilon: f32,
    capture: bool,
    preserve: bool,
    mut tape: Option<&mut IndexerTape<D>>,
    layer: usize,
) -> anyhow::Result<(Option<D::Tensor>, D::Tensor, Option<(usize, usize)>)> {
    for kind in model::norm_order(batch, tokens, capture, preserve, tape.is_some()) {
        if let Norm::Into(_) = kind {
            let (cache, start) = tape
                .as_mut()
                .unwrap()
                .reserve_compact_f16_append(batch as i32, tokens, model::LATENT)
                .map_err(|e| anyhow::anyhow!("glm53 compact cache reserve: {e}"))?;
            let _trace = context.trace_detail("glm53.dsa.qkv_norm_compact_into", layer as u64);
            let direct = one(
                context,
                Request::Norm {
                    kind: Norm::Into(start),
                    epsilon: epsilon.to_bits(),
                },
                &[inputs[0], inputs[1], inputs[2], inputs[3], &cache],
            );
            if let Ok(q) = direct {
                return Ok((None, q, Some((start, tokens))));
            }
        } else {
            let result = run(
                context,
                Request::Norm {
                    kind,
                    epsilon: epsilon.to_bits(),
                },
                &inputs,
            );
            if kind == Norm::Separate || result.is_ok() {
                let mut out = result?.into_iter();
                let q = out.next().unwrap();
                return Ok((out.next(), q, None));
            }
        }
    }
    unreachable!("normalization plan includes mandatory separate fallback")
}
