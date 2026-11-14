//! Checked expert primitives, independent of model traversal and checkpoints.
use super::*;
const E: TensorConstraint = TensorConstraint {
    min_rank: 3,
    max_rank: Some(3),
    ..PACKED
};
const M: TensorConstraint = TensorConstraint {
    min_rank: 3,
    max_rank: Some(3),
    ..BF16
};
const S: TensorConstraint = TensorConstraint {
    min_rank: 2,
    max_rank: Some(2),
    ..BF16
};
const I: TensorConstraint = TensorConstraint {
    dtypes: &[ScalarType::I32, ScalarType::U32],
    min_rank: 1,
    max_rank: None,
    layout: LayoutSupport::RowContiguous,
    alignment: 4,
};
const F: TensorConstraint = TensorConstraint {
    dtypes: &[ScalarType::F32],
    ..BF16
};
pub(super) fn capabilities() -> Vec<KernelCapability> {
    use AffineExpertsKind as K;
    let cap = |kind, inputs| KernelCapability {
        kind: KernelKind::AffineExperts(kind),
        law: NumericalLaw::AffineExpertsV1,
        inputs,
        variadic: false,
        shape: ShapeRelation::TensorOperation,
        encoding: None,
        workspace: Workspace::ProviderManaged,
    };
    let mut caps = vec![
        cap(K::RouterVector, &[F, BF16]),
        cap(K::Route, &[F, BF16, PACKED, S, S]),
        cap(K::CountingSort, &[I]),
        cap(K::Tiles, &[I]),
        cap(K::Gather, &[BF16, E, M, M, I]),
        cap(K::SortedCombine, &[BF16, F, I, BF16, BF16]),
    ];
    for wide in [false, true] {
        caps.push(cap(
            K::GateUp(wide),
            &[E, M, M, E, M, M, PACKED, S, S, BF16, I, I],
        ));
        caps.push(cap(
            K::DownCombine(wide),
            &[E, M, M, PACKED, S, S, BF16, I, F, BF16],
        ));
    }
    if crate::runtime::nax_available() {
        caps.extend([
            cap(K::RouterMatrix, &[BF16, BF16]),
            cap(K::TiledGateUp, &[BF16, E, M, M, E, M, M, I, I, I]),
            cap(K::TiledDown, &[BF16, E, M, M, I, I]),
        ]);
    }
    caps
}
fn matrix(a: &[&Array], n: i32, k: i32) -> bool {
    a[0].shape() == [n, k / 8] && a[1].shape() == [n, k / 32] && a[2].shape() == a[1].shape()
}
fn experts(a: &[&Array], e: i32, n: i32, k: i32) -> bool {
    a[0].shape() == [e, n, k / 8] && a[1].shape() == [e, n, k / 32] && a[2].shape() == a[1].shape()
}
pub(super) fn invoke(
    kernel: &Kernel,
    a: &[&Array],
    stream: &Stream,
) -> Result<Option<Outputs<Array>>> {
    let Kernel::AffineExperts(op) = kernel else {
        return Ok(None);
    };
    if a.iter().any(|v| v.shape().iter().any(|&n| n <= 0)) {
        return Ok(None);
    }
    use AffineExperts as K;
    let out = match *op {
        K::RouterVector => {
            if a[1].rank() != 2
                || a[0].size() != a[1].dim(1) as usize
                || a[1].dim(0) % 16 != 0
                || a[1].dim(1) % 32 != 0
            {
                return Ok(None);
            }
            crate::kernels::router_gemv(a[0], a[1], stream)
        }
        K::RouterMatrix { rows } => {
            if a[0].size() != rows as usize * 2560
                || a[0].dim(-1) != 2560
                || a[1].shape() != [512, 2560]
            {
                return Ok(None);
            }
            crate::prefill_indirect::router(a[0], a[1], rows, stream)
        }
        K::Route {
            top_k,
            experts,
            hidden,
            rows,
        } => {
            if a[0].shape() != [rows, experts]
                || a[1].shape() != [rows, hidden]
                || !matrix(&a[2..5], 1, hidden)
            {
                return Ok(None);
            }
            return Ok(crate::moe_decode::route(
                a[0], a[1], a[2], a[3], a[4], top_k, experts, hidden, rows, stream,
            )
            .map(|(a, b, c)| Outputs::Many(vec![a, b, c])));
        }
        K::GateUp {
            top_k,
            hidden,
            intermediate,
            rows,
            wide,
        } => {
            let br = rows * top_k;
            let e = a[0].dim(0);
            if !experts(&a[0..3], e, intermediate, hidden)
                || !experts(&a[3..6], e, intermediate, hidden)
                || !matrix(&a[6..9], 2 * intermediate, hidden)
                || a[9].shape() != [rows, hidden]
                || a[10].size() != br as usize
                || a[11].size() != br as usize
            {
                return Ok(None);
            }
            let f = if wide {
                crate::moe_decode::gate_up_act_wide
            } else {
                crate::moe_decode::gate_up_act
            };
            f(
                a[0],
                a[1],
                a[2],
                a[3],
                a[4],
                a[5],
                a[6],
                a[7],
                a[8],
                a[9],
                a[10],
                a[11],
                br,
                intermediate,
                hidden,
                rows,
                stream,
            )
        }
        K::DownCombine {
            top_k,
            hidden,
            intermediate,
            rows,
            wide,
        } => {
            let br = rows * top_k;
            let e = a[0].dim(0);
            if !experts(&a[0..3], e, hidden, intermediate)
                || !matrix(&a[3..6], hidden, intermediate)
                || a[6].shape() != [br + rows, intermediate]
                || a[7].size() != br as usize
                || a[8].size() != br as usize
                || a[9].size() != rows as usize
            {
                return Ok(None);
            }
            let f = if wide {
                crate::moe_decode::down_combine_wide
            } else {
                crate::moe_decode::down_combine
            };
            f(
                a[0],
                a[1],
                a[2],
                a[3],
                a[4],
                a[5],
                a[6],
                a[7],
                a[8],
                a[9],
                top_k,
                intermediate,
                hidden,
                br,
                rows,
                stream,
            )
        }
        K::CountingSort { experts, top_k } => {
            if a[0].rank() != 1 || a[0].dim(0) % top_k != 0 {
                return Ok(None);
            }
            return Ok(crate::kernels::route_counting_sort(
                a[0],
                experts as usize,
                top_k as usize,
                stream,
            )
            .map(|(a, b, c)| Outputs::Many(vec![a, b, c])));
        }
        K::Tiles { rows, experts } => {
            if a[0].size() != rows as usize {
                return Ok(None);
            }
            crate::prefill_indirect::tile_table(a[0], rows, experts, stream)
        }
        K::TiledGateUp {
            rows,
            experts: e,
            hidden,
            intermediate,
        } => {
            let max = crate::prefill_indirect::max_tiles(rows, e);
            if a[0].dim(-1) != hidden
                || !experts(&a[1..4], e, intermediate, hidden)
                || !experts(&a[4..7], e, intermediate, hidden)
                || a[7].size() != rows as usize
                || a[8].size() != rows as usize
                || a[9].size() != 2 * max as usize
            {
                return Ok(None);
            }
            crate::prefill_indirect::gate_up(
                a[0],
                a[1],
                a[2],
                a[3],
                a[4],
                a[5],
                a[6],
                a[7],
                a[8],
                a[9],
                max,
                intermediate,
                hidden,
                rows,
                stream,
            )
        }
        K::TiledDown {
            rows,
            experts: e,
            hidden,
            intermediate,
        } => {
            let max = crate::prefill_indirect::max_tiles(rows, e);
            if a[0].shape() != [rows, 1, intermediate]
                || !experts(&a[1..4], e, hidden, intermediate)
                || a[4].size() != rows as usize
                || a[5].size() != 2 * max as usize
            {
                return Ok(None);
            }
            crate::prefill_indirect::down(
                a[0],
                a[1],
                a[2],
                a[3],
                a[4],
                a[5],
                max,
                hidden,
                intermediate,
                rows,
                stream,
            )
        }
        K::Gather => {
            if a[0].rank() != 3
                || a[0].dim(1) != 1
                || a[0].dim(2) % 32 != 0
                || !experts(&a[1..4], a[1].dim(0), a[1].dim(1), a[0].dim(2))
                || a[4].size() != a[0].dim(0) as usize
            {
                return Ok(None);
            }
            Some(crate::ops::gather_qmm(
                a[0],
                a[1],
                a[2],
                Some(a[3]),
                None,
                Some(a[4]),
                true,
                32,
                4,
                true,
            )?)
        }
        K::SortedCombine {
            top_k,
            hidden,
            rows,
        } => {
            let br = rows * top_k;
            if a[0].shape() != [br, hidden]
                || a[1].size() != br as usize
                || a[2].size() != br as usize
                || a[3].shape() != [rows, hidden]
                || a[4].size() != rows as usize
            {
                return Ok(None);
            }
            crate::kernels::moe_combine(a[0], a[1], a[2], a[3], a[4], top_k, hidden, rows, stream)
        }
    };
    Ok(out.map(Outputs::One))
}

#[cfg(test)]
mod tests {
    use super::*;
    /// Batch decode uses the narrow arithmetic independently for every row,
    /// including the shared expert. Compare every output with a one-row call;
    /// comparing two batched wrappers would share an unwritten-row bug.
    #[test]
    fn narrow_expert_batch_matches_independent_rows() -> Result<()> {
        let device = MetalDevice::system_default();
        let stream = device.stream();
        let (e, h, n, k) = (4, 2560, 640, 2);
        let packed = |shape: &[i32], salt: u32| {
            let count = shape.iter().map(|&v| v as usize).product();
            Array::from_slice(
                &(0..count)
                    .map(|i| (i as u32).wrapping_mul(2654435761).wrapping_add(salt))
                    .collect::<Vec<_>>(),
                shape,
            )
        };
        let meta = |shape: &[i32], value| {
            Array::from_slice(
                &vec![half::bf16::from_f32(value); shape.iter().map(|&v| v as usize).product()],
                shape,
            )
        };
        let wg = packed(&[e, n, h / 8], 17);
        let wu = packed(&[e, n, h / 8], 31);
        let wsh = packed(&[2 * n, h / 8], 53);
        let scales = meta(&[e, n, h / 32], 0.001);
        let biases = meta(&[e, n, h / 32], -0.007);
        let sh_scales = meta(&[2 * n, h / 32], 0.001);
        let sh_biases = meta(&[2 * n, h / 32], -0.007);
        for rows in [1, 2, 3, 8] {
            let values = (0..rows * h)
                .map(|i| half::bf16::from_f32(((i * 13 % 251) as f32 - 125.) / 311.))
                .collect::<Vec<_>>();
            let ids = (0..rows * k).map(|i| i % e).collect::<Vec<_>>();
            let invoke = |x: &Array, ids: &[i32], xrows: &[i32], rows| -> Result<Vec<u16>> {
                let handle = device
                    .prepare(Kernel::AffineExperts(AffineExperts::GateUp {
                        top_k: k,
                        hidden: h,
                        intermediate: n,
                        rows,
                        wide: false,
                    }))?
                    .unwrap();
                let ids = Array::from_slice(ids, &[rows * k]);
                let xrows = Array::from_slice(xrows, &[rows * k]);
                let output = device
                    .dispatch(
                        &handle,
                        &[
                            &wg, &scales, &biases, &wu, &scales, &biases, &wsh, &sh_scales,
                            &sh_biases, x, &ids, &xrows,
                        ],
                        &stream,
                    )?
                    .unwrap();
                let Outputs::One(output) = output.output else {
                    panic!("one activation tensor expected")
                };
                output.to_vec1::<u16>()
            };
            let batched = invoke(
                &Array::from_slice(&values, &[rows, h]),
                &ids,
                &(0..rows * k).map(|i| i / k).collect::<Vec<_>>(),
                rows,
            )?;
            for row in 0..rows as usize {
                let single = invoke(
                    &Array::from_slice(&values[row * h as usize..(row + 1) * h as usize], &[1, h]),
                    &ids[row * k as usize..(row + 1) * k as usize],
                    &vec![0; k as usize],
                    1,
                )?;
                let n = n as usize;
                let k = k as usize;
                assert_eq!(
                    &batched[row * k * n..(row + 1) * k * n],
                    &single[..k * n],
                    "routed rows={rows} row={row}"
                );
                assert_eq!(
                    &batched[(rows as usize * k + row) * n..(rows as usize * k + row + 1) * n],
                    &single[k * n..],
                    "shared rows={rows} row={row}"
                );
            }
        }
        Ok(())
    }
    #[test]
    fn expert_contracts_reject_overflow_and_incompatible_storage() -> Result<()> {
        let device = MetalDevice::system_default();
        let stream = device.stream();
        for op in [
            AffineExperts::Route {
                top_k: 513,
                experts: 512,
                hidden: 2560,
                rows: 1,
            },
            AffineExperts::GateUp {
                top_k: 10,
                hidden: 2560,
                intermediate: 640,
                rows: i32::MAX,
                wide: true,
            },
            AffineExperts::Tiles {
                rows: i32::MAX,
                experts: 512,
            },
        ] {
            assert!(device.prepare(Kernel::AffineExperts(op))?.is_none());
        }
        let h = device
            .prepare(Kernel::AffineExperts(AffineExperts::RouterVector))?
            .unwrap();
        let x = Array::from_slice(&[0f32; 32], &[32]);
        let w = Array::from_slice(&[half::bf16::ZERO; 1024], &[16, 64]);
        assert!(device.dispatch(&h, &[&x, &w], &stream)?.is_none());
        Ok(())
    }
}
