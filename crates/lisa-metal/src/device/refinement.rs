//! Model-unaware normalization, causal writes and stable selection kernels.
use super::*;
use crate::kernels::refinement as native;
const STRIDED_BF16: TensorConstraint = TensorConstraint {
    layout: LayoutSupport::Strided,
    ..BF16
};
const IDS: TensorConstraint = TensorConstraint {
    dtypes: &[ScalarType::I32, ScalarType::U32],
    layout: LayoutSupport::Strided,
    ..BF16
};
const SCORES: TensorConstraint = TensorConstraint {
    dtypes: &[ScalarType::F32, ScalarType::BF16, ScalarType::F16],
    layout: LayoutSupport::Strided,
    ..BF16
};
pub(super) fn capabilities() -> Vec<KernelCapability> {
    use RefinementKind as K;
    [
        (K::Normalize, &[STRIDED_BF16, STRIDED_BF16][..]),
        (K::NormalizeCodes, &[STRIDED_BF16, IDS, STRIDED_BF16][..]),
        (K::FeatureMessage, &[STRIDED_BF16, STRIDED_BF16][..]),
        (
            K::FeatureUpdate,
            &[
                STRIDED_BF16,
                STRIDED_BF16,
                TensorConstraint {
                    layout: LayoutSupport::Strided,
                    dtypes: &[ScalarType::F32],
                    ..BF16
                },
            ][..],
        ),
        (K::Candidates(16), &[SCORES][..]),
        (K::Candidates(128), &[SCORES][..]),
        (K::GreedyChain, &[IDS, IDS][..]),
    ]
    .into_iter()
    .map(|(kind, inputs)| KernelCapability {
        kind: KernelKind::Refinement(kind),
        law: NumericalLaw::MlxRefinementV1,
        inputs,
        variadic: false,
        shape: ShapeRelation::TensorOperation,
        encoding: None,
        workspace: Workspace::ProviderManaged,
    })
    .collect()
}
// Scratch is provider-owned, bounded by the current selection shape. The native
// workspace checks context, completion and surviving leases before every reuse.
thread_local! {
    static SELECT16:std::cell::RefCell<native::SelectionWorkspace>=Default::default();
    static SELECT128:std::cell::RefCell<native::SelectionWorkspace>=Default::default();
}
pub(super) fn invoke(
    kernel: &Kernel,
    a: &[&Array],
    stream: &Stream,
) -> Result<Option<Outputs<Array>>> {
    let Kernel::Refinement(op) = kernel else {
        return Ok(None);
    };
    let output = match op {
        Refinement::Normalize { epsilon, read } => {
            let (x, rms) =
                native::normalize_on(a[0], a[1], f32::from_bits(*epsilon), *read, stream)?;
            Outputs::Two([x, rms])
        }
        Refinement::NormalizeCodes { epsilon } => Outputs::One(native::normalize_codes_on(
            a[0],
            a[1],
            a[2],
            f32::from_bits(*epsilon),
            stream,
        )?),
        Refinement::FeatureMessage { batch, rows } => Outputs::One(native::feature_message_on(
            a[0], a[1], *batch, *rows, stream,
        )?),
        Refinement::FeatureUpdate => {
            Outputs::One(native::feature_update_on(a[0], a[1], a[2], stream)?)
        }
        Refinement::Candidates { count: 16 } => Outputs::One(SELECT16.with(|w| {
            native::top16_candidates_with_workspace_on(a[0], &mut w.borrow_mut(), stream)
        })?),
        Refinement::Candidates { count: 128 } => Outputs::One(SELECT128.with(|w| {
            native::top128_candidates_with_workspace_on(a[0], &mut w.borrow_mut(), stream)
        })?),
        Refinement::Candidates { .. } => return Ok(None),
        Refinement::GreedyChain => Outputs::One(native::greedy_chain_on(a[0], a[1], stream)?),
    };
    Ok(Some(output))
}
