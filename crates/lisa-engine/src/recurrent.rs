//! Engine-owned immutable tensor bindings for model-defined recurrent history.
use crate::core::graph::{PlanCache, PreparedGraph};
use crate::{execution::DeviceContext, indexer_cache::CacheDevice};
use lisa_backend::Device;
use lisa_backend::TensorAccess;
use lisa_models::gated_delta::ReplayGeometry;
#[derive(Clone)]
pub struct Replay<T> {
    pub geometry: ReplayGeometry,
    pub tensors: [T; 5],
}
impl<T: Clone> Replay<T> {
    pub fn restore_on<D: Device<Tensor = T>>(&self, device: D, prefix: usize) -> anyhow::Result<T> {
        let prepared = PreparedGraph::new(device, self.geometry.restore(prefix)?, vec![])?;
        prepared
            .run(&self.tensors.iter().collect::<Vec<_>>(), &mut [])?
            .into_one()
            .map_err(anyhow::Error::msg)
    }
}

impl<T: TensorAccess + 'static> Replay<T> {
    pub fn restore<D: CacheDevice<Tensor = T>>(
        &self,
        context: &DeviceContext<D>,
        prefix: usize,
    ) -> anyhow::Result<T> {
        context.with_slot(
            "recurrent.restore",
            || std::cell::RefCell::new(PlanCache::new(context.device().clone(), 16)),
            |cache| {
                cache
                    .borrow_mut()
                    .try_get_or_prepare((self.geometry, prefix), || self.geometry.restore(prefix))?
                    .run(&self.tensors.iter().collect::<Vec<_>>(), &mut [])?
                    .into_one()
                    .map_err(anyhow::Error::msg)
            },
        )
    }
}
pub fn capture<D: CacheDevice>(
    context: &DeviceContext<D>,
    q: &D::Tensor,
    k: &D::Tensor,
    v: &D::Tensor,
    g: &D::Tensor,
    beta: &D::Tensor,
    state: &D::Tensor,
) -> anyhow::Result<Option<(D::Tensor, D::Tensor, Replay<D::Tensor>)>> {
    if k.rank() != 4 || v.rank() != 4 {
        return Ok(None);
    }
    let geometry = ReplayGeometry {
        batch: k.dim(0),
        tokens: k.dim(1),
        key_heads: k.dim(2),
        value_heads: v.dim(2),
        key_width: k.dim(3),
        value_width: v.dim(3),
    };
    let device = context.device();
    if !geometry.accepts_types([q, k, v, g, beta, state].map(|v| device.dtype(v))) {
        return Ok(None);
    }
    let output = context.with_slot(
        "recurrent.capture",
        || std::cell::RefCell::new(PlanCache::new(context.device().clone(), 8)),
        |cache| {
            cache
                .borrow_mut()
                .try_get_or_prepare(geometry, || geometry.capture())?
                .run(&[q, k, v, g, beta, state], &mut [])
        },
    )?;
    let mut values = output.into_iter();
    let y = values.next().expect("validated activation");
    let final_state = values.next().expect("validated final state");
    let tensors = std::array::from_fn(|_| values.next().expect("validated history"));
    Ok(Some((y, final_state, Replay { geometry, tensors })))
}

#[derive(PartialEq)]
enum PrefixKey {
    Snapshot(usize, Vec<i32>),
    Convolution(usize, Vec<i32>, usize),
}

pub fn snapshot_prefix<D: CacheDevice>(
    context: &DeviceContext<D>,
    tensor: &D::Tensor,
    prefix: usize,
) -> anyhow::Result<D::Tensor> {
    context.with_slot(
        "recurrent.prefix",
        || std::cell::RefCell::new(PlanCache::new(context.device().clone(), 32)),
        |cache| {
            cache
                .borrow_mut()
                .try_get_or_prepare(PrefixKey::Snapshot(prefix, tensor.shape().to_vec()), || {
                    lisa_models::gated_delta::snapshot_state(prefix, tensor.shape())
                })?
                .run(&[tensor], &mut [])?
                .into_one()
                .map_err(anyhow::Error::msg)
        },
    )
}
pub fn convolution_prefix<D: CacheDevice>(
    context: &DeviceContext<D>,
    tensor: &D::Tensor,
    prefix: usize,
    history: usize,
) -> anyhow::Result<D::Tensor> {
    context.with_slot(
        "recurrent.prefix",
        || std::cell::RefCell::new(PlanCache::new(context.device().clone(), 32)),
        |cache| {
            cache
                .borrow_mut()
                .try_get_or_prepare(
                    PrefixKey::Convolution(prefix, tensor.shape().to_vec(), history),
                    || {
                        lisa_models::gated_delta::convolution_prefix(
                            prefix,
                            tensor.shape(),
                            history,
                        )
                    },
                )?
                .run(&[tensor], &mut [])?
                .into_one()
                .map_err(anyhow::Error::msg)
        },
    )
}
