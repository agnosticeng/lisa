//! Accepted-prefix restoration of captured recurrent state through model-defined plans.
use crate::core::graph::PlanCache;
use crate::{cache::indexer::CacheDevice, execution::DeviceContext};
use lisa_backend::TensorAccess;
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
