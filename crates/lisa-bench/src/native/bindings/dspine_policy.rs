//! Application composition of portable DSpine profile identity.
pub use lisa_engine::bindings::dspine_policy::{Identity, Policy, SsmMode, cache_path};
pub fn detect(target: &std::path::Path, draft: &std::path::Path) -> anyhow::Result<Identity> {
    lisa_engine::bindings::dspine_policy::detect(
        &crate::metal_context::execution_context(),
        target,
        draft,
    )
}
pub(crate) fn load(
    target: &std::path::Path,
    draft: &std::path::Path,
) -> anyhow::Result<(usize, usize, SsmMode)> {
    lisa_engine::bindings::dspine_policy::load(
        &crate::metal_context::execution_context(),
        target,
        draft,
    )
}
