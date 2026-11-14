//! Native oracle harness. The shared fixture context is confined to diagnostics;
//! production applications pass explicit owners to engine model bindings.
use lisa_engine::execution::DeviceContext;
use lisa_metal::device::MetalDevice;
thread_local! {
    static FIXTURE: DeviceContext<MetalDevice> = lisa_serve::application::context().expect("Metal oracle provider");
}
pub fn execution_context() -> DeviceContext<MetalDevice> {
    FIXTURE.with(Clone::clone)
}
pub(crate) fn device() -> MetalDevice {
    execution_context().device().clone()
}
/// A typed key, not a cache owner. Each context holds its own initialized value.
pub struct ContextSlot<T: 'static> {
    key: &'static str,
    init: fn() -> T,
}
impl<T: 'static> ContextSlot<T> {
    pub const fn new(key: &'static str, init: fn() -> T) -> Self {
        Self { key, init }
    }
    pub fn with<R>(&self, f: impl FnOnce(&T) -> R) -> R {
        execution_context().with_slot(self.key, self.init, f)
    }
}
#[macro_export]
macro_rules! metal_context_local {
    ($(static $name:ident: $ty:ty = $init:expr;)*) => {
        $(static $name: $crate::metal_context::ContextSlot<$ty> =
            $crate::metal_context::ContextSlot::new(concat!(module_path!(), "::", line!(), "::", stringify!($name)), || $init);)*
    };
}
