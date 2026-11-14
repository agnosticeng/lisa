//! Packed weight views and bounded program preparation, with an explicit caller device.
use crate::{
    core::graph::PlanCache, execution::DeviceContext, indexer_cache::CacheDevice,
    weights::gguf::PackedTensor,
};
use anyhow::{Result, ensure};
use lisa_backend::TensorAccess;
use lisa_backend::{Outputs, PackedEncoding};
use lisa_gguf::TensorType;
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
pub fn matrix<D: CacheDevice + lisa_backend::StorageDevice>(
    w: &PackedTensor<D>,
) -> Result<model::Matrix> {
    ensure!(
        w.info.dimensions.len() == 2,
        "{} is not a matrix",
        w.info.name
    );
    Ok(model::Matrix {
        kind: super::packed::projection(w.info.kind)?,
        input: i32::try_from(w.info.dimensions[0])?,
        output: i32::try_from(w.info.dimensions[1])?,
        row_bytes: i32::try_from(w.info.row_bytes()?)?,
    })
}
pub fn bound<D: CacheDevice + lisa_backend::StorageDevice>(
    w: &PackedTensor<D>,
) -> Result<D::Tensor> {
    if matches!(
        w.info.kind,
        TensorType::F32 | TensorType::F16 | TensorType::BF16
    ) {
        w.plain_view()
    } else {
        Ok(w.packed_bytes().clone())
    }
}
pub fn plain_view<D: CacheDevice + lisa_backend::StorageDevice>(
    w: &PackedTensor<D>,
) -> Result<D::Tensor> {
    w.plain_view()
}
pub fn block_format(kind: TensorType) -> Result<PackedEncoding> {
    let p = super::packed::projection(kind)?;
    ensure!(
        model::supported(p),
        "unsupported scalar packed encoding {kind:?}"
    );
    let Projection::Block(encoding) = p else {
        anyhow::bail!("expert weights require packed block storage")
    };
    Ok(encoding)
}
pub fn linear<D: CacheDevice + lisa_backend::StorageDevice>(
    w: &PackedTensor<D>,
    x: &D::Tensor,
    round: bool,
    stream: &DeviceContext<D>,
) -> Result<D::Tensor> {
    ensure!(
        w.info.dimensions.len() == 2,
        "{} is not a matrix",
        w.info.name
    );
    linear_rows(w, 0, w.info.dimensions[1], x, round, stream)
}
pub fn linear_rows<D: CacheDevice + lisa_backend::StorageDevice>(
    w: &PackedTensor<D>,
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
    a: &PackedTensor<D>,
    b: &PackedTensor<D>,
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
    w: &PackedTensor<D>,
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
