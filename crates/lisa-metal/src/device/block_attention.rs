//! One prepared operation per existing launch; model policy stays outside Metal.
use super::*;
pub(super) fn capabilities() -> Vec<KernelCapability> {
    const STRIDED: TensorConstraint = TensorConstraint {
        layout: LayoutSupport::Strided,
        ..BF16
    };
    const IDS: TensorConstraint = TensorConstraint {
        dtypes: &[ScalarType::I32, ScalarType::U32],
        ..STRIDED
    };
    const BOOL: TensorConstraint = TensorConstraint {
        dtypes: &[ScalarType::Bool],
        ..STRIDED
    };
    let cap = |kind, inputs| KernelCapability {
        kind: KernelKind::BlockAttention(kind),
        law: NumericalLaw::BlockAttentionV1,
        inputs,
        variadic: false,
        shape: ShapeRelation::TensorOperation,
        encoding: None,
        workspace: Workspace::ProviderManaged,
    };
    let mut out = vec![
        cap(BlockAttentionKind::Select, &[STRIDED, STRIDED]),
        cap(
            BlockAttentionKind::Prepare,
            &[
                STRIDED, STRIDED, STRIDED, STRIDED, STRIDED, STRIDED, STRIDED,
            ],
        ),
        cap(BlockAttentionKind::Gate, &[STRIDED, STRIDED]),
    ];
    if crate::runtime::nax_available() {
        out.push(cap(
            BlockAttentionKind::Attend,
            &[STRIDED, STRIDED, STRIDED, IDS, BOOL],
        ));
    }
    out
}
pub(super) fn invoke(
    kernel: &Kernel,
    a: &[&Array],
    stream: &Stream,
) -> Result<Option<Outputs<Array>>> {
    let Kernel::BlockAttention(op) = kernel else {
        return Ok(None);
    };
    if a.iter().any(|x| x.shape().iter().any(|&d| d <= 0)) {
        return Ok(None);
    }
    Ok(match *op {
        BlockAttention::Select {
            offset,
            total,
            blocks,
        } => {
            if a[0].shape() != [1, total - offset, 4, 128] || a[1].shape() != [1, blocks, 128] {
                return Ok(None);
            }
            crate::qsa::select_blocks(a[0], a[1], offset, total, blocks as usize, stream)
                .map(|(x, y)| Outputs::Two([x, y]))
        }
        BlockAttention::Attend {
            offset,
            total,
            scale,
        } => {
            if a[0].shape() != [1, 24, total - offset, 256]
                || a[1].rank() != 4
                || a[1].dim(0) != 1
                || a[1].dim(1) != 2
                || a[1].dim(2) < total
                || a[1].dim(3) != 256
                || a[2].shape() != a[1].shape()
                || a[3].shape() != [total - offset, 512]
                || a[4].shape() != a[3].shape()
            {
                return Ok(None);
            }
            crate::qsa::prefill_flash(
                a[0],
                a[1],
                a[2],
                a[3],
                a[4],
                offset,
                total,
                f32::from_bits(scale),
                stream,
            )
            .map(Outputs::One)
        }
        BlockAttention::Prepare {
            heads,
            key_heads,
            width,
            rotary,
            epsilon,
        } => {
            if a[0].rank() != 3 {
                return Ok(None);
            }
            let (b, s) = (a[0].dim(0), a[0].dim(1));
            if a[0].dim(2) != heads * width
                || a[1].shape() != [b, s, key_heads * width]
                || a[2].shape() != a[1].shape()
                || a[3].shape() != [width]
                || a[4].shape() != [width]
                || a[5].dim(-1) != rotary
                || !([
                    s as usize * rotary as usize,
                    b as usize * s as usize * rotary as usize,
                ]
                .contains(&a[5].size()))
                || a[6].shape() != a[5].shape()
            {
                return Ok(None);
            }
            crate::kernels::attn_prep_split(
                a[0],
                a[1],
                a[2],
                a[3],
                a[4],
                a[5],
                a[6],
                heads,
                key_heads,
                width,
                rotary,
                f32::from_bits(epsilon),
                stream,
            )
            .map(|(x, y, z)| Outputs::Many(vec![x, y, z]))
        }
        BlockAttention::Gate { heads, width } => {
            if a[0].dim(-1) != heads * width || a[1].size() != a[0].size() {
                return Ok(None);
            }
            crate::kernels::attn_gate(a[0], a[1], heads, width, stream).map(Outputs::One)
        }
    })
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn boolean_gather_preserves_selected_bytes() -> Result<()> {
        let d = MetalDevice::system_default();
        let values = Array::from_slice(&[true, false, true, false, false, true], &[2, 3]);
        let indices = Array::from_slice(&[2u32, 1, 0, 2], &[2, 2]);
        let op = d
            .prepare(Kernel::Tensor(TensorOperation::TakeAlong { axis: -1 }))?
            .unwrap();
        let out = d
            .dispatch(&op, &[&values, &indices], &d.stream())?
            .unwrap()
            .output
            .into_one()
            .map_err(Error::msg)?;
        assert_eq!(out.to_vec1::<bool>()?, vec![true, false, false, true]);
        Ok(())
    }
    #[test]
    fn block_primitives_reject_incompatible_bindings() -> Result<()> {
        let d = MetalDevice::system_default();
        assert!(
            d.prepare(Kernel::BlockAttention(BlockAttention::Select {
                offset: 0,
                total: 1,
                blocks: 65537
            }))?
            .is_none()
        );
        let op = d
            .prepare(Kernel::BlockAttention(BlockAttention::Select {
                offset: 2048,
                total: 2049,
                blocks: 512,
            }))?
            .unwrap();
        let q = Array::from_slice(&vec![half::bf16::ZERO; 512], &[1, 1, 4, 128]);
        let short = Array::from_slice(&vec![half::bf16::ZERO; 128], &[1, 1, 128]);
        assert!(d.dispatch(&op, &[&q, &short], &d.stream())?.is_none());
        Ok(())
    }
}
#[cfg(test)]
mod scatter_layout_tests {
    use super::*;
    #[test]
    fn scatter_honors_broadcast_updates_and_strided_indices() -> Result<()> {
        let d = MetalDevice::system_default();
        let op = d
            .prepare(Kernel::Tensor(TensorOperation::PutAlong { axis: -1 }))?
            .unwrap();
        for strided in [false, true] {
            let ids = if strided {
                Array::from_slice(&[2u32, 0, 1, 2], &[2, 2]).transpose_axes(&[1, 0])?
            } else {
                Array::from_slice(&[2u32, 0, 1, 2], &[2, 2])
            };
            for broadcast in [false, true] {
                let updates = if broadcast {
                    Array::from_bool(true).broadcast_to(&[2, 2])?
                } else {
                    Array::from_slice(&[1u8, 1, 1, 1], &[2, 2])
                };
                let src = Array::from_slice(&[0u8; 6], &[2, 3]);
                let out = d
                    .dispatch(&op, &[&src, &ids, &updates], &d.stream())?
                    .unwrap()
                    .output
                    .into_one()
                    .map_err(Error::msg)?;
                assert_eq!(
                    out.to_vec1::<u8>()?,
                    if strided {
                        vec![0, 1, 1, 1, 0, 1]
                    } else {
                        vec![1, 0, 1, 0, 1, 1]
                    },
                    "strided={strided} broadcast={broadcast}"
                );
                assert_eq!(src.to_vec1::<u8>()?, vec![0; 6]);
            }
        }
        Ok(())
    }
}
