//! Packed weight views and bounded program preparation, with an explicit caller device.
use crate::{
    cache::indexer::CacheDevice, core::graph::PlanCache, execution::DeviceContext,
    weights::prepared::PreparedTensor,
};
use anyhow::{Result, ensure};
use lisa_backend::TensorAccess;
use lisa_backend::{Outputs, PackedEncoding};
use lisa_models::{deepseek41::projection as model, packed::Projection};
use std::cell::RefCell;
#[derive(Clone, Copy, PartialEq)]
enum Key {
    Linear(model::Matrix, i32, i32, bool),
    Pair(model::Matrix, model::Matrix, i32),
    Groups(model::Matrix),
}
fn run<D: CacheDevice + lisa_backend::StorageDevice>(
    key: Key,
    inputs: &[&D::Tensor],
    context: &DeviceContext<D>,
) -> Result<Outputs<D::Tensor>> {
    context.with_slot(
        "deepseek41.projection",
        || RefCell::new(PlanCache::new(context.device().clone(), 128)),
        |slot| {
            let recipe = key;
            let graph = || match recipe {
                Key::Linear(m, first, count, round) => m.linear(first, count, round),
                Key::Pair(a, b, rows) => model::paired(a, b, rows),
                Key::Groups(m) => model::output_groups(m),
            };
            slot.borrow_mut()
                .try_get_or_prepare(key, graph)?
                .run(inputs, &mut [])
        },
    )
}
pub fn matrix<T: Clone>(w: &PreparedTensor<T>) -> Result<model::Matrix> {
    ensure!(
        w.layout.logical_shape.len() == 2,
        "projection weight is not a matrix"
    );
    Ok(model::Matrix {
        kind: w.layout.projection,
        input: w.layout.logical_shape[1],
        output: w.layout.logical_shape[0],
        row_bytes: w.layout.row_bytes,
    })
}
pub fn bound<T: Clone>(w: &PreparedTensor<T>) -> Result<T> {
    if w.layout.projection == Projection::Plain {
        w.plain_view()
    } else {
        Ok(w.packed_bytes().clone())
    }
}
pub fn plain_view<T: Clone>(w: &PreparedTensor<T>) -> Result<T> {
    w.plain_view()
}
pub fn block_format(p: Projection) -> Result<PackedEncoding> {
    ensure!(
        model::supported(p),
        "unsupported scalar packed encoding {p:?}"
    );
    let Projection::Block(encoding) = p else {
        anyhow::bail!("expert weights require packed block storage")
    };
    Ok(encoding)
}
pub fn linear<D: CacheDevice + lisa_backend::StorageDevice>(
    w: &PreparedTensor<D::Tensor>,
    x: &D::Tensor,
    round: bool,
    stream: &DeviceContext<D>,
) -> Result<D::Tensor> {
    ensure!(
        w.layout.logical_shape.len() == 2,
        "projection weight is not a matrix"
    );
    linear_rows(
        w,
        0,
        u64::try_from(w.layout.logical_shape[0])?,
        x,
        round,
        stream,
    )
}
pub fn linear_rows<D: CacheDevice + lisa_backend::StorageDevice>(
    w: &PreparedTensor<D::Tensor>,
    first: u64,
    count: u64,
    x: &D::Tensor,
    round: bool,
    stream: &DeviceContext<D>,
) -> Result<D::Tensor> {
    ensure!(
        x.rank() == 2 && (1..=8).contains(&x.dim(0)),
        "scalar projection requires one to eight rows"
    );
    let m = matrix(w)?;
    run(
        Key::Linear(m, i32::try_from(first)?, i32::try_from(count)?, round),
        &[&bound(w)?, x],
        stream,
    )?
    .into_one()
    .map_err(anyhow::Error::msg)
}
pub fn query_kv_pair<D: CacheDevice + lisa_backend::StorageDevice>(
    a: &PreparedTensor<D::Tensor>,
    b: &PreparedTensor<D::Tensor>,
    x: &D::Tensor,
    stream: &DeviceContext<D>,
) -> Result<(D::Tensor, D::Tensor)> {
    ensure!(x.rank() == 2, "paired projection requires a matrix input");
    let mut result = run(
        Key::Pair(matrix(a)?, matrix(b)?, x.dim(0)),
        &[&bound(a)?, &bound(b)?, x],
        stream,
    )?
    .into_iter();
    Ok((result.next().unwrap(), result.next().unwrap()))
}
pub fn attention_output_groups<D: CacheDevice + lisa_backend::StorageDevice>(
    w: &PreparedTensor<D::Tensor>,
    x: &D::Tensor,
    stream: &DeviceContext<D>,
) -> Result<D::Tensor> {
    ensure!(
        x.shape() == [1, 64, 512],
        "invalid grouped attention output input"
    );
    run(Key::Groups(matrix(w)?), &[&bound(w)?, x], stream)?
        .into_one()
        .map_err(anyhow::Error::msg)
}
