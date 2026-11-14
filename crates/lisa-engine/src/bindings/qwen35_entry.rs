//! Tower input bindings; positions and embedding composition are model-owned.
use crate::{bindings::BindingDevice, execution::DeviceContext};
use crate::{
    bindings::normalization::Rotary, core::graph::PlanCache, weights::affine::QuantizedEmbedding,
};
use lisa_backend::ScalarType;
use lisa_backend::{HostTensor, ManagedDevice, TensorAccess};
use lisa_models::qwen3_5::entry::Plan;
#[derive(PartialEq)]
enum Key {
    Entry(Plan),
    Capture,
}
pub fn begin<D: BindingDevice + ManagedDevice>(
    context: &DeviceContext<D>,
    ids: &D::Tensor,
    embedding: &QuantizedEmbedding<D::Tensor>,
    rope: &Rotary,
    vision: Option<&crate::vision::Vision<D>>,
    offset: usize,
    count: usize,
    next: Option<&[usize]>,
) -> anyhow::Result<(D::Tensor, D::Tensor, D::Tensor)> {
    let (positions, multimodal) = if let Some(pos) = vision
        .map(|v| v.positions(offset, count))
        .transpose()?
        .flatten()
    {
        (pos, true)
    } else {
        let (values, shape) = lisa_models::position::indices(offset, count, next)?;
        (
            context.device().pull(
                HostTensor {
                    bytes: &values
                        .iter()
                        .flat_map(|v| v.to_ne_bytes())
                        .collect::<Vec<_>>(),
                    shape: &shape,
                    dtype: ScalarType::F32,
                }
                .into(),
                Default::default(),
            )?,
            false,
        )
    };
    let splice = vision.and_then(|v| {
        v.splice_bindings(
            vec![
                ids.dim(0),
                ids.dim(1),
                embedding.scales.dim(-1) * embedding.group_size,
            ],
            ScalarType::BF16,
            offset,
        )
    });
    let spec = Plan {
        positions_shape: positions.shape().to_vec(),
        multimodal,
        rotary: rope.dimensions,
        base: rope.base.to_bits(),
        embedding: (embedding.group_size, embedding.bits),
        splice: splice.as_ref().map(|(s, _)| s.clone()),
    };
    let mut inputs = vec![
        ids,
        &embedding.weight,
        &embedding.scales,
        &embedding.biases,
        &positions,
    ];
    if let Some((_, features)) = &splice {
        inputs.extend(features);
    }
    let mut output = context
        .with_slot(
            "qwen35.entry",
            || std::cell::RefCell::new(PlanCache::new(context.device().clone(), 16)),
            |p| {
                p.borrow_mut()
                    .try_get_or_prepare(Key::Entry(spec.clone()), || spec.graph())?
                    .run(&inputs, &mut [])
            },
        )?
        .into_iter();
    Ok((
        output.next().unwrap(),
        output.next().unwrap(),
        output.next().unwrap(),
    ))
}
pub fn capture_hidden<D: BindingDevice>(
    context: &DeviceContext<D>,
    x: &D::Tensor,
    residual: &D::Tensor,
) -> anyhow::Result<D::Tensor> {
    context.with_slot(
        "qwen35.entry",
        || std::cell::RefCell::new(PlanCache::new(context.device().clone(), 16)),
        |p| {
            p.borrow_mut()
                .get_or_prepare(Key::Capture, lisa_models::qwen3_5::entry::capture_hidden)?
                .run(&[x, residual], &mut [])?
                .into_one()
                .map_err(anyhow::Error::msg)
        },
    )
}
