//! Individual projection contracts; shape selection and composition are caller-owned.
use super::*;
pub(super) fn capabilities() -> Vec<KernelCapability> {
    [
        (
            KernelKind::Bf16SmallRowLinear,
            NumericalLaw::MlxSmallRowBf16V1,
            &[BF16, BF16][..],
        ),
        (
            KernelKind::Affine4WideLinear,
            NumericalLaw::MlxWideAffine4V1,
            &[BF16, PACKED, BF16, BF16][..],
        ),
    ]
    .into_iter()
    .map(|(kind, law, inputs)| KernelCapability {
        kind,
        law,
        inputs,
        variadic: false,
        shape: ShapeRelation::TensorOperation,
        encoding: None,
        workspace: Workspace::ProviderManaged,
    })
    .collect()
}
pub(super) fn invoke(
    kernel: &Kernel,
    inputs: &[&Array],
    _: &Stream,
) -> Result<Option<Outputs<Array>>> {
    let x = inputs[0];
    let output = match kernel {
        Kernel::Bf16SmallRowLinear => {
            if !x.t.device().nax() {
                return Ok(None);
            }
            crate::jit::dspine_linear_nax(x.t.device(), &x.t, &inputs[1].t)?
        }
        Kernel::Affine4WideLinear => {
            crate::jit::dspine_affine(x.t.device(), &x.t, &inputs[1].t, &inputs[2].t, &inputs[3].t)?
        }
        _ => return Ok(None),
    };
    Ok(Some(Outputs::One(Array::new(output))))
}
