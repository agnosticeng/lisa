use lisa_metal::{Array, device::IndirectResources};
use lisa_models::glm53::routed_moe::Request;
pub(crate) fn run(
    request: Request,
    inputs: &[&Array],
    resources: Option<&IndirectResources>,
) -> anyhow::Result<Vec<Array>> {
    lisa_engine::bindings::glm53_moe_ops::run(
        &crate::metal_context::execution_context(),
        request,
        inputs,
        resources,
    )
}
