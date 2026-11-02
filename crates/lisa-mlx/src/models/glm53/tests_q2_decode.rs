//! Exact comparisons against the unchanged scalar-row affine GPU paths.

use super::*;
use crate::ops::indexing::IndexOp;

const EXPERTS: usize = 3;
const ROUTES: usize = 8;

struct Bank {
    weight: Array,
    scale: Array,
    bias: Array,
    rows: usize,
    experts: usize,
}

impl Bank {
    fn new(rows: usize, columns: usize, seed: usize, dtype: Dtype) -> Self {
        Self::with_experts(EXPERTS, rows, columns, seed, dtype)
    }

    fn with_experts(
        experts: usize,
        rows: usize,
        columns: usize,
        seed: usize,
        dtype: Dtype,
    ) -> Self {
        // Each expert has a leading unused row, so all bound projection views
        // exercise nonzero offsets rather than only whole-buffer bindings.
        let stored_rows = rows + 1;
        let packed_columns = columns / 16;
        let packed = (0..experts * stored_rows * packed_columns)
            .map(|index| {
                let mut word = 0u32;
                for lane in 0..16 {
                    let hash =
                        (index * 37 + lane * 13 + seed * 19) ^ ((index / 7 + lane * 3 + seed) * 11);
                    word |= ((hash % 4) as u32) << (2 * lane);
                }
                word
            })
            .collect::<Vec<_>>();
        let groups = columns / 128;
        let scale = (0..experts * stored_rows * groups)
            .map(|i| 0.00231 + ((i * 13 + seed * 7) % 29) as f32 * 0.000713)
            .collect::<Vec<_>>();
        let bias = scale
            .iter()
            .enumerate()
            .map(|(i, &scale)| {
                -1.37 * scale + (((i * 17 + seed * 11) % 13) as f32 - 6.0) * 0.000419
            })
            .collect::<Vec<_>>();
        let metadata_shape = [experts as i32, stored_rows as i32, groups as i32];
        Self {
            weight: Array::from_slice(
                &packed,
                &[experts as i32, stored_rows as i32, packed_columns as i32],
            ),
            scale: Array::from_slice(&scale, &metadata_shape)
                .as_dtype(dtype)
                .expect("metadata dtype"),
            bias: Array::from_slice(&bias, &metadata_shape)
                .as_dtype(dtype)
                .expect("metadata dtype"),
            rows,
            experts,
        }
    }

    fn fields(&self, expert: usize) -> [Array; 3] {
        let end = (self.rows + 1) as i32;
        [
            self.weight.index((expert as i32, 1..end, ..)),
            self.scale.index((expert as i32, 1..end, ..)),
            self.bias.index((expert as i32, 1..end, ..)),
        ]
    }
}

fn tables(gate: &Bank, up: &Bank, down: &Bank) -> (Array, Array, Vec<Array>, Vec<Array>) {
    assert_eq!(gate.experts, up.experts);
    assert_eq!(gate.experts, down.experts);
    let mut fields = Vec::with_capacity(gate.experts * 9);
    for expert in 0..gate.experts {
        fields.extend(gate.fields(expert));
        fields.extend(up.fields(expert));
        fields.extend(down.fields(expert));
    }
    let addresses = fields
        .iter()
        .map(|tensor| tensor.gpu_address().expect("expert GPU address") as i64)
        .collect::<Vec<_>>();
    let address_table = Array::from_slice(&addresses, &[(gate.experts * 9) as i32]);
    let refs = fields.iter().collect::<Vec<_>>();
    let (mapped_table, resources) = moe_mapped_resource_table(&refs).expect("mapped expert table");
    (address_table, mapped_table, resources, fields)
}

fn assert_exact(actual: &[f32], expected: &[f32], label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}: shape");
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            actual.is_finite() && expected.is_finite(),
            "{label}[{index}]: finite"
        );
        assert_eq!(actual.to_bits(), expected.to_bits(), "{label}[{index}]");
    }
}

#[test]
fn glm53_q2_pair_multi_output_matches_scalar_rows_exactly() {
    if !crate::device::metal_available() {
        return;
    }
    let stream = Stream::thread_local_or_default();
    for dtype in [Dtype::Float32, Dtype::Bfloat16] {
        for (tokens, columns, rows) in [
            (1, 512, 1),
            (1, 512, 3),
            (2, 512, 7),
            (2, 512, 9),
            (2, 4096, 2048),
            (3, 512, 9),
        ] {
            let gate = Bank::new(rows, columns, 1, dtype);
            let up = Bank::new(rows, columns, 2, dtype);
            // The pair does not read the down fields, but all nine addresses
            // must remain valid while either independently bound path runs.
            let down = Bank::new(1, 128, 3, dtype);
            let (addresses, mapped, resources, fields) = tables(&gate, &up, &down);
            let keepalive = fields.iter().collect::<Vec<_>>();
            let input = (0..tokens * columns)
                .map(|i| (((i * 23 + i / columns * 31) % 127) as f32 - 63.0) * 0.0137)
                .collect::<Vec<_>>();
            let x = Array::from_slice(&input, &[tokens as i32, columns as i32]);
            let ids = (0..tokens * ROUTES)
                .map(|i| ((i * 5 + i / ROUTES) % EXPERTS) as i32)
                .collect::<Vec<_>>();
            let selected = Array::from_slice(&ids, &[(tokens * ROUTES) as i32]);
            let routes = (0..tokens * ROUTES)
                .map(|i| 0.0317 + (i % ROUTES) as f32 * 0.0173)
                .collect::<Vec<_>>();
            let routes = Array::from_slice(&routes, &[(tokens * ROUTES) as i32]);
            for limit in [0.0, 0.05, 10.0] {
                let expected = moe_pair_swiglu_mapped(
                    &mapped, &x, &selected, &routes, &resources, rows, ROUTES, limit, dtype,
                    &stream,
                )
                .expect("unchanged scalar-row pair")
                .to_vec1::<f32>()
                .expect("reference pair readback");
                let actual = moe_pair_swiglu_addr(
                    &addresses, &x, &selected, &routes, &keepalive, rows, ROUTES, limit, dtype,
                    None, &stream,
                )
                .expect("multi-output pair")
                .to_vec1::<f32>()
                .expect("candidate pair readback");
                assert_exact(
                    &actual,
                    &expected,
                    &format!("pair M={tokens} K={columns} N={rows} {dtype:?} clamp={limit}"),
                );
            }
        }
    }
}

#[test]
fn glm53_q2_pair_overlap_ownership_matches_scalar_rows_exactly() {
    if !crate::device::metal_available() {
        return;
    }
    const OVERLAP_EXPERTS: usize = 16;
    const COLUMNS: usize = 512;
    let stream = Stream::thread_local_or_default();
    let patterns: [(&str, [i32; 16]); 6] = [
        (
            "disjoint",
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        ),
        ("full", [0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7]),
        (
            "reordered",
            [0, 1, 2, 3, 4, 5, 6, 7, 7, 6, 5, 4, 3, 2, 1, 0],
        ),
        (
            "partial",
            [0, 1, 2, 3, 4, 5, 6, 7, 3, 9, 5, 11, 0, 13, 14, 15],
        ),
        (
            "duplicates",
            [0, 0, 1, 2, 2, 3, 4, 4, 4, 0, 4, 1, 6, 6, 3, 0],
        ),
        (
            "invalid",
            [-1, 0, 1, 2, 16, 3, 4, 4, 4, -1, 4, 1, 16, 6, 3, 0],
        ),
    ];
    for dtype in [Dtype::Float32, Dtype::Bfloat16] {
        for rows in [9, 32] {
            let gate = Bank::with_experts(OVERLAP_EXPERTS, rows, COLUMNS, 1, dtype);
            let up = Bank::with_experts(OVERLAP_EXPERTS, rows, COLUMNS, 2, dtype);
            let down = Bank::with_experts(OVERLAP_EXPERTS, 1, 128, 3, dtype);
            let (addresses, mapped, resources, fields) = tables(&gate, &up, &down);
            let keepalive = fields.iter().collect::<Vec<_>>();
            let input = (0..2 * COLUMNS)
                .map(|i| (((i * 23 + i / COLUMNS * 31) % 127) as f32 - 63.0) * 0.0137)
                .collect::<Vec<_>>();
            let x = Array::from_slice(&input, &[2, COLUMNS as i32]);
            // Token1 duplicate slots deliberately have different route weights.
            let route_weights = (0..16)
                .map(|i| 0.0317 + i as f32 * 0.0173)
                .collect::<Vec<_>>();
            let route_weights = Array::from_slice(&route_weights, &[16]);
            for (pattern, ids) in &patterns {
                let selected = Array::from_slice(ids, &[16]);
                for limit in [0.0, 0.05, 10.0] {
                    let expected = moe_pair_swiglu_mapped(
                        &mapped,
                        &x,
                        &selected,
                        &route_weights,
                        &resources,
                        rows,
                        ROUTES,
                        limit,
                        dtype,
                        &stream,
                    )
                    .expect("unchanged scalar-row pair")
                    .to_vec1::<f32>()
                    .expect("reference pair readback");
                    let actual = moe_pair_swiglu_addr(
                        &addresses,
                        &x,
                        &selected,
                        &route_weights,
                        &keepalive,
                        rows,
                        ROUTES,
                        limit,
                        dtype,
                        None,
                        &stream,
                    )
                    .expect("rows2 pair with expert reuse")
                    .to_vec1::<f32>()
                    .expect("candidate pair readback");
                    assert_exact(
                        &actual,
                        &expected,
                        &format!("rows2 {pattern} N={rows} {dtype:?} clamp={limit}"),
                    );
                }
            }
        }
    }
}

#[test]
fn glm53_q2_down_group_metadata_matches_scalar_routes_exactly() {
    if !crate::device::metal_available() {
        return;
    }
    let stream = Stream::thread_local_or_default();
    for dtype in [Dtype::Float32, Dtype::Bfloat16] {
        for (tokens, columns, rows) in [(1, 128, 1), (1, 128, 3), (2, 512, 7), (2, 2048, 4096)] {
            let gate = Bank::new(1, 128, 1, dtype);
            let up = Bank::new(1, 128, 2, dtype);
            let down = Bank::new(rows, columns, 3, dtype);
            let (addresses, _, _, fields) = tables(&gate, &up, &down);
            let keepalive = fields.iter().collect::<Vec<_>>();
            let input = (0..tokens * ROUTES * columns)
                .map(|i| (((i * 19 + i / columns * 37) % 113) as f32 - 56.0) * 0.0179)
                .collect::<Vec<_>>();
            let x = Array::from_slice(&input, &[(tokens * ROUTES) as i32, columns as i32]);
            let ids = (0..tokens * ROUTES)
                .map(|i| match i % ROUTES {
                    3 => -1,
                    6 => EXPERTS as i32,
                    _ => ((i * 5 + i / ROUTES) % EXPERTS) as i32,
                })
                .collect::<Vec<_>>();
            let selected = Array::from_slice(&ids, &[(tokens * ROUTES) as i32]);
            let per_route = moe_gather_qmv_addr_ds4(
                &addresses, &x, &selected, &keepalive, rows, true, dtype, None, &stream,
            )
            .expect("unchanged scalar-route down")
            .to_vec1::<f32>()
            .expect("reference down readback");
            let shared_values = (0..tokens * rows)
                .map(|i| ((i * 17 % 43) as f32 - 21.0) * 0.00371)
                .collect::<Vec<_>>();
            let shared = Array::from_slice(&shared_values, &[tokens as i32, rows as i32]);
            for include_shared in [false, true] {
                let actual = moe_gather_qmv_sum8_addr_ds4(
                    &addresses,
                    &x,
                    &selected,
                    &keepalive,
                    include_shared.then_some(&shared),
                    rows,
                    dtype,
                    None,
                    &stream,
                )
                .expect("metadata-hoisted down sum8")
                .to_vec1::<f32>()
                .expect("candidate down readback");
                let mut expected = vec![0.0f32; tokens * rows];
                for token in 0..tokens {
                    for row in 0..rows {
                        let index = token * rows + row;
                        for route in 0..ROUTES {
                            let slot = token * ROUTES + route;
                            let contribution = if (0..EXPERTS as i32).contains(&ids[slot]) {
                                per_route[slot * rows + row]
                            } else {
                                0.0
                            };
                            expected[index] += contribution;
                        }
                        if include_shared {
                            expected[index] += shared_values[index];
                        }
                    }
                }
                assert_exact(
                    &actual,
                    &expected,
                    &format!(
                        "down M={tokens} K={columns} N={rows} {dtype:?} shared={include_shared}"
                    ),
                );
            }
        }
    }
}
