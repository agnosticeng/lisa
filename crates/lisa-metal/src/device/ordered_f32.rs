//! Prepared individual F32 kernels; no checkpoint or model knowledge.
use super::*;
use crate::kernels::blockwise as native;
const F32: TensorConstraint = TensorConstraint {
    dtypes: &[ScalarType::F32],
    min_rank: 1,
    max_rank: None,
    layout: LayoutSupport::RowContiguous,
    alignment: 4,
};
const ATTENTION_STATE: TensorConstraint = TensorConstraint {
    min_rank: 4,
    max_rank: Some(4),
    layout: LayoutSupport::Strided,
    ..F32
};
const BYTES: TensorConstraint = TensorConstraint {
    dtypes: &[ScalarType::U8],
    min_rank: 2,
    max_rank: Some(2),
    layout: LayoutSupport::RowContiguous,
    alignment: 2,
};
pub(super) fn capabilities() -> Vec<KernelCapability> {
    use OrderedF32Kind as K;
    let capability = |kind, inputs| KernelCapability {
        kind: KernelKind::OrderedF32(kind),
        law: NumericalLaw::OrderedF32V1,
        inputs,
        variadic: false,
        shape: ShapeRelation::TensorOperation,
        encoding: None,
        workspace: Workspace::ProviderManaged,
    };
    let mut caps = [
        (K::UnitRms, &[F32][..]),
        (K::DepthwiseConv4Silu, &[F32, F32][..]),
        (K::DecayGates, &[F32, F32, F32, F32][..]),
        (K::SiluProduct, &[F32, F32][..]),
        (K::Delta128, &[F32, F32, F32, F32, F32, F32][..]),
        (
            K::VectorAttention,
            &[
                TensorConstraint {
                    min_rank: 4,
                    max_rank: Some(4),
                    ..F32
                },
                ATTENTION_STATE,
                ATTENTION_STATE,
            ][..],
        ),
    ]
    .into_iter()
    .map(|(k, a)| capability(k, a))
    .collect::<Vec<_>>();
    for encoding in OrderedBlockEncoding::ALL {
        caps.push(capability(K::BlockLinear(encoding), &[BYTES, F32]));
    }
    caps
}
pub(super) fn invoke(
    kernel: &Kernel,
    a: &[&Array],
    stream: &Stream,
) -> Result<Option<Outputs<Array>>> {
    let Kernel::OrderedF32(op) = kernel else {
        return Ok(None);
    };
    if a.iter().any(|x| x.shape().iter().any(|&d| d <= 0)) {
        return Ok(None);
    }
    if matches!(op, OrderedF32::VectorAttention { .. }) {
        let (q, k, v) = (a[0], a[1], a[2]);
        if q.rank() != 4
            || k.rank() != 4
            || v.rank() != 4
            || q.dim(0) != k.dim(0)
            || k.dim(0) != v.dim(0)
            || k.dim(1) != v.dim(1)
            || q.dim(1) % k.dim(1) != 0
            || k.dim(2) != v.dim(2)
            || q.dim(2) > k.dim(2)
            || q.dim(3) != k.dim(3)
            || q.dim(3) != v.dim(3)
            || !matches!(q.dim(3), 128 | 256)
            || k.t.stride()[3] != 1
            || v.t.stride()[3] != 1
        {
            return Ok(None);
        }
    }
    let out = match op {
        OrderedF32::UnitRms { epsilon } => {
            Outputs::One(crate::fast::rms_norm(a[0], None, f32::from_bits(*epsilon))?)
        }
        OrderedF32::BlockLinear { encoding } => {
            Outputs::One(native::matvec(*encoding, a[0], a[1], stream)?)
        }
        OrderedF32::DepthwiseConv4Silu => Outputs::One(native::conv_silu_on(a[0], a[1], stream)?),
        OrderedF32::DecayGates => {
            let (g, b) = native::gates_on(a[0], a[1], a[2], a[3], stream)?;
            Outputs::Two([g, b])
        }
        OrderedF32::SiluProduct { reverse } => {
            Outputs::One(native::silu_product_on(a[0], a[1], *reverse, stream)?)
        }
        OrderedF32::Delta128 => {
            let (y, s) = native::gdn(a[0], a[1], a[2], a[3], a[4], a[5], stream)?;
            Outputs::Two([y, s])
        }
        OrderedF32::VectorAttention { scale, causal } => {
            Outputs::One(Array::new(crate::jit::sdpa_vector(
                stream.runtime(),
                &a[0].t,
                &a[1].t,
                &a[2].t,
                f32::from_bits(*scale),
                *causal,
                None,
            )?))
        }
    };
    Ok(Some(out))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn prepared_f32_contracts_cover_block_encodings_and_reject_invalid_geometry() -> Result<()> {
        let device = MetalDevice::system_default();
        let stream = device.stream();
        for encoding in OrderedBlockEncoding::ALL {
            assert!(
                device
                    .prepare(Kernel::OrderedF32(OrderedF32::BlockLinear { encoding }))?
                    .is_some()
            );
        }
        assert!(
            device
                .prepare(Kernel::OrderedF32(OrderedF32::UnitRms {
                    epsilon: f32::NAN.to_bits()
                }))?
                .is_none()
        );
        let norm = device
            .prepare(Kernel::OrderedF32(OrderedF32::UnitRms {
                epsilon: 1e-6f32.to_bits(),
            }))?
            .unwrap();
        let bf16 = Array::from_slice(&[1f32; 128], &[1, 128]).as_dtype(Dtype::Bfloat16)?;
        assert!(device.dispatch(&norm, &[&bf16], &stream).is_err());
        let attention = device
            .prepare(Kernel::OrderedF32(OrderedF32::VectorAttention {
                scale: 0.0625f32.to_bits(),
                causal: false,
            }))?
            .unwrap();
        let q = Array::from_slice(&[1f32; 3 * 128], &[1, 3, 1, 128]);
        let k = Array::from_slice(&[1f32; 2 * 128], &[1, 2, 1, 128]);
        assert!(
            device
                .dispatch(&attention, &[&q, &k, &k], &stream)?
                .is_none()
        );
        let small = Array::from_slice(&[1f32; 32], &[1, 1, 1, 32]);
        assert!(
            device
                .dispatch(&attention, &[&small, &small, &small], &stream)?
                .is_none()
        );
        let zero = Array::from_slice::<f32>(&[], &[1, 0, 1, 128]);
        assert!(
            device
                .dispatch(&attention, &[&zero, &zero, &zero], &stream)?
                .is_none()
        );
        Ok(())
    }
}
