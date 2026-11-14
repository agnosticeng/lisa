//! Application bindings for portable packed projections.
use crate::core::gguf_resident::PackedTensor;
use anyhow::Result;
pub use lisa_engine::bindings::deepseek41_projection::{block_format, bound, matrix, plain_view};
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
    lisa_engine::bindings::deepseek41_projection::linear(w, x, round, context)
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
    lisa_engine::bindings::deepseek41_projection::linear_rows(w, first, count, x, round, context)
}
pub fn query_kv_pair(
    a: &PackedTensor,
    b: &PackedTensor,
    x: &Array,
    stream: &Stream,
) -> Result<(Array, Array)> {
    let owned = context_for(stream);
    let context = &owned;
    lisa_engine::bindings::deepseek41_projection::query_kv_pair(a, b, x, context)
}
pub fn attention_output_groups(w: &PackedTensor, x: &Array, stream: &Stream) -> Result<Array> {
    let owned = context_for(stream);
    let context = &owned;
    lisa_engine::bindings::deepseek41_projection::attention_output_groups(w, x, context)
}
