//! Independent residual launches. All graph and shape policy lives in models.
use super::*;
pub(super) fn capabilities() -> Vec<KernelCapability> {
    use ResidualBf16Kind as K;
    let cap = |kind, inputs| KernelCapability {
        kind: KernelKind::ResidualBf16(kind),
        law: NumericalLaw::ResidualBf16V1,
        inputs,
        variadic: false,
        shape: ShapeRelation::TensorOperation,
        encoding: None,
        workspace: Workspace::ProviderManaged,
    };
    // Static metadata constraints keep prepared handles independent of weights.
    const M: TensorConstraint = TensorConstraint {
        min_rank: 2,
        max_rank: Some(2),
        ..BF16
    };
    vec![
        cap(K::InjectNorm, &[BF16, BF16, BF16, BF16]),
        cap(K::DownInject, &[BF16, PACKED, M, M, PACKED, M, M]),
        cap(K::UpMix, &[BF16, BF16, PACKED, M, M, BF16]),
        cap(K::Mix, &[BF16, BF16, BF16]),
        cap(K::SiluHead, &[BF16]),
        cap(K::NormalizedProduct, &[BF16, BF16, BF16, BF16, BF16]),
        cap(K::GatedNorm, &[BF16, BF16, BF16, BF16, BF16, BF16]),
        cap(K::DilatedConvAdd, &[BF16, BF16, BF16]),
    ]
}
fn affine(a: &[&Array], rows: i32, width: i32) -> bool {
    a[0].shape() == [rows, width / 8]
        && a[1].shape() == [rows, width / 32]
        && a[2].shape() == a[1].shape()
}
pub(super) fn invoke(
    kernel: &Kernel,
    a: &[&Array],
    stream: &Stream,
) -> Result<Option<Outputs<Array>>> {
    let Kernel::ResidualBf16(op) = kernel else {
        return Ok(None);
    };
    if a.iter().any(|t| t.shape().iter().any(|&n| n <= 0)) {
        return Ok(None);
    }
    let out = match *op {
        ResidualBf16::InjectNorm {
            hidden,
            streams,
            rows,
            inject,
            tile,
            epsilon,
        } => {
            let width = hidden * streams;
            if a[0].size() != rows as usize * if tile { hidden } else { width } as usize
                || a[3].size() != width as usize
                || (inject
                    && (a[1].size() != rows as usize * hidden as usize
                        || a[2].size() != rows as usize * streams as usize))
            {
                return Ok(None);
            }
            crate::kernels::inject_norm(
                a[0],
                inject.then_some(a[1]),
                inject.then_some(a[2]),
                a[3],
                streams,
                hidden,
                rows,
                tile,
                f32::from_bits(epsilon),
                stream,
            )
            .map(|(a, b)| Outputs::Two([a, b]))
        }
        ResidualBf16::DownInject {
            width,
            rank,
            streams,
            rows,
            inject,
        } => {
            if a[0].size() != rows as usize * width as usize
                || a[0].dim(-1) != width
                || !affine(&a[1..4], rank, width)
                || (inject && !affine(&a[4..7], streams, width))
            {
                return Ok(None);
            }
            crate::moe_decode::down_inject(
                a[0], a[1], a[2], a[3], a[4], a[5], a[6], width, rank, streams, rows, inject,
                stream,
            )
            .map(|(a, b, c)| Outputs::Many(vec![a, b, c]))
        }
        ResidualBf16::UpMix {
            hidden,
            streams,
            rows,
            inject,
            packed,
        } => {
            let rank = a[0].dim(-1);
            if rank <= 0
                || rank % 32 != 0
                || a[0].size() != rows as usize * rank as usize
                || a[1].size() != rows as usize * (hidden * streams) as usize
                || !affine(&a[2..5], hidden * streams, rank)
                || a[5].size() != rows as usize * streams as usize
            {
                return Ok(None);
            }
            crate::moe_decode::up_mix(
                a[0], a[1], a[2], a[3], a[4], a[5], hidden, streams, rows, inject, packed, stream,
            )
            .map(|(a, b)| Outputs::Two([a, b]))
        }
        ResidualBf16::Mix {
            hidden,
            streams,
            rows,
            inject,
        } => {
            let n = rows as usize * (hidden * streams) as usize;
            if a[0].size() != n
                || a[1].size() != n
                || (inject && a[2].size() != rows as usize * streams as usize)
            {
                return Ok(None);
            }
            crate::kernels::hc_mix(a[0], a[1], a[2], streams, hidden, rows, inject, stream)
                .map(|(a, b)| Outputs::Two([a, b]))
        }
        ResidualBf16::NormalizedProduct { hidden, streams } => {
            let width = hidden * streams;
            if a[0].rank() != 3
                || a[0].dim(2) != width
                || a[1].shape() != a[0].shape()
                || a[2].size() != width as usize
                || a[3].size() != width as usize
                || a[4].size() != 1
            {
                return Ok(None);
            }
            crate::kernels::ple_prod(a[0], a[1], a[2], a[3], a[4], streams, hidden, stream)
                .map(Outputs::One)
        }
        ResidualBf16::GatedNorm { hidden, streams } => {
            if a[1].rank() != 3
                || a[1].dim(2) != hidden
                || a[0].shape() != [a[1].dim(0), a[1].dim(1), streams]
                || a[2].size() != (hidden * streams) as usize
                || a[3..].iter().any(|v| v.size() != 1)
            {
                return Ok(None);
            }
            crate::kernels::ple_gated(a[0], a[1], a[2], a[3], a[4], a[5], streams, hidden, stream)
                .map(|(a, b)| Outputs::Two([a, b]))
        }
        ResidualBf16::DilatedConvAdd { dilation } => {
            if a[0].rank() != 3
                || a[1].rank() != 3
                || a[2].rank() != 3
                || a[1].dim(2) != 1
                || a[1].dim(1) <= 0
                || a[0].dim(0) != a[2].dim(0)
                || a[0].dim(2) != a[2].dim(2)
                || a[1].dim(0) != a[2].dim(2)
                || (a[1].dim(1) - 1)
                    .checked_mul(dilation)
                    .and_then(|n| n.checked_add(a[2].dim(1)))
                    != Some(a[0].dim(1))
            {
                return Ok(None);
            }
            crate::kernels::ple_conv(a[0], a[1], a[2], dilation, stream).map(Outputs::One)
        }
        ResidualBf16::SiluHead { width } => {
            if a[0].dim(-1) < width || a[0].dim(-1) <= 0 {
                return Ok(None);
            }
            crate::kernels::silu_head(a[0], width, stream).map(Outputs::One)
        }
    };
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn residual_primitives_reject_invalid_geometry_before_launch() -> Result<()> {
        let device = MetalDevice::system_default();
        let stream = device.stream();
        for op in [
            ResidualBf16::DownInject {
                width: 10240,
                rank: 320,
                streams: 3,
                rows: 1,
                inject: true,
            },
            ResidualBf16::UpMix {
                hidden: 2560,
                streams: 4,
                rows: 8,
                packed: true,
                inject: true,
            },
            ResidualBf16::NormalizedProduct {
                hidden: 4097,
                streams: 4,
            },
            ResidualBf16::DilatedConvAdd { dilation: 0 },
        ] {
            assert!(device.prepare(Kernel::ResidualBf16(op))?.is_none());
        }
        let kernel = device
            .prepare(Kernel::ResidualBf16(ResidualBf16::NormalizedProduct {
                hidden: 128,
                streams: 4,
            }))?
            .unwrap();
        let x = Array::from_slice(&vec![half::bf16::ZERO; 512], &[1, 1, 512]);
        let scale = Array::from_slice(&vec![half::bf16::ONE; 512], &[512]);
        let epsilon = Array::from_slice(&[half::bf16::from_f32(1e-6)], &[1]);
        let short = Array::from_slice(&vec![half::bf16::ONE; 128], &[128]);
        assert!(
            device
                .dispatch(&kernel, &[&x, &x, &short, &scale, &epsilon], &stream)?
                .is_none()
        );
        let wrong = x.as_dtype(Dtype::Float32)?;
        assert!(
            device
                .dispatch(&kernel, &[&wrong, &x, &scale, &scale, &epsilon], &stream)
                .is_err()
        );
        Ok(())
    }
}
