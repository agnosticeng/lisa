//! Application bindings for portable packed projections.
use crate::core::gguf_resident::PackedTensor;
use anyhow::Result;
pub fn block_format(kind: lisa_gguf::TensorType) -> Result<lisa_backend::PackedEncoding> {
    lisa_engine::bindings::deepseek41::projection::block_format(
        lisa_engine::weights::gguf::projection(kind)?,
    )
}
pub fn matrix(w: &PackedTensor) -> Result<lisa_models::deepseek41::projection::Matrix> {
    lisa_engine::bindings::deepseek41::projection::matrix(&w.clone().into_prepared()?)
}
pub fn bound(w: &PackedTensor) -> Result<Array> {
    lisa_engine::bindings::deepseek41::projection::bound(&w.clone().into_prepared()?)
}
pub fn plain_view(w: &PackedTensor) -> Result<Array> {
    lisa_engine::bindings::deepseek41::projection::plain_view(&w.clone().into_prepared()?)
}
use lisa_metal::{Array, Stream};
struct StreamContext {
    runtime: std::sync::Arc<lisa_metal::runtime::MetalRuntime>,
    context: lisa_engine::execution::DeviceContext<lisa_metal::device::MetalDevice>,
}
crate::metal_context_local! { static CONTEXT: std::cell::RefCell<Option<StreamContext>> = const {std::cell::RefCell::new(None)}; }
pub(super) fn context_for(
    stream: &Stream,
) -> lisa_engine::execution::DeviceContext<lisa_metal::device::MetalDevice> {
    CONTEXT.with(|slot| {
        let mut slot = slot.borrow_mut();
        if slot
            .as_ref()
            .is_none_or(|p| !std::sync::Arc::ptr_eq(&p.runtime, stream.runtime()))
        {
            *slot = Some(StreamContext {
                runtime: stream.runtime().clone(),
                context: lisa_engine::execution::DeviceContext::new(
                    lisa_metal::device::MetalDevice::on_stream(stream),
                ),
            });
        }
        slot.as_ref().unwrap().context.clone()
    })
}
pub fn linear(w: &PackedTensor, x: &Array, round: bool, stream: &Stream) -> Result<Array> {
    let owned = context_for(stream);
    let context = &owned;
    lisa_engine::bindings::deepseek41::projection::linear(
        &w.clone().into_prepared()?,
        x,
        round,
        context,
    )
}
pub fn linear_rows(
    w: &PackedTensor,
    first: u64,
    count: u64,
    x: &Array,
    round: bool,
    stream: &Stream,
) -> Result<Array> {
    let owned = context_for(stream);
    let context = &owned;
    lisa_engine::bindings::deepseek41::projection::linear_rows(
        &w.clone().into_prepared()?,
        first,
        count,
        x,
        round,
        context,
    )
}
pub fn query_kv_pair(
    a: &PackedTensor,
    b: &PackedTensor,
    x: &Array,
    stream: &Stream,
) -> Result<(Array, Array)> {
    let owned = context_for(stream);
    let context = &owned;
    lisa_engine::bindings::deepseek41::projection::query_kv_pair(
        &a.clone().into_prepared()?,
        &b.clone().into_prepared()?,
        x,
        context,
    )
}
pub fn attention_output_groups(w: &PackedTensor, x: &Array, stream: &Stream) -> Result<Array> {
    let owned = context_for(stream);
    let context = &owned;
    lisa_engine::bindings::deepseek41::projection::attention_output_groups(
        &w.clone().into_prepared()?,
        x,
        context,
    )
}
