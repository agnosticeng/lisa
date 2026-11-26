use super::*;
use crate::ops::{Array as A, Stream};

fn values(count: usize, seed: u32) -> Vec<f32> {
    let mut state = seed;
    (0..count)
        .map(|index| {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            let bits = if index % 131 == 0 {
                state & 0x8000_0000
            } else {
                (state & 0x807f_0000) | (((state >> 25) % 24 + 112) << 23)
            };
            f32::from_bits(bits)
        })
        .collect()
}

#[test]
fn canonical_small_rows_match_original_eight_row_projection_bits() -> Result<()> {
    let rt = Stream::gpu().runtime().clone();
    if !rt.nax() {
        return Ok(());
    }
    // Include all final-block projection shapes, both K-lane partitions, a
    // nonmultiple K tail and the large-N grid-x special case. Nonzero views
    // preserve the original BF16 payload and exercise offset bindings.
    for (k, n) in [
        (260, 37),
        (260, 64),
        (260, 65),
        (3072, 2304),
        (2304, 9216),
        (9216, 2304),
        (128, 65536),
    ] {
        let a = A::from_slice(&values(9 * k, 71), &[9, k as i32]).as_dtype(Dtype::Bfloat16)?;
        let w = A::from_slice(&values((n + 1) * k, 113), &[(n + 1) as i32, k as i32])
            .as_dtype(Dtype::Bfloat16)?;
        let a = a.t.narrow(0, 1, 8)?;
        let w = w.t.narrow(0, 1, n)?;
        let full = bf16_small_row_linear(&rt, &a, &w)?.to_vec::<u16>()?;
        for rows in 1..=7 {
            let selected = a.narrow(0, 1, rows)?;
            let actual = bf16_small_row_linear_canonical(&rt, &selected, &w, 8)?.to_vec::<u16>()?;
            assert_eq!(actual, full[n..(rows + 1) * n], "rows={rows} K={k} N={n}");
        }
        let same = bf16_small_row_linear_canonical(&rt, &a, &w, 8)?.to_vec::<u16>()?;
        assert_eq!(same, full, "full canonical K={k} N={n}");
    }
    Ok(())
}

#[test]
fn canonical_small_rows_reject_unqualified_reference_shapes_before_encoding() -> Result<()> {
    let rt = Stream::gpu().runtime().clone();
    if !rt.nax() {
        return Ok(());
    }
    let a = A::from_slice(&values(9 * 128, 17), &[9, 128]).as_dtype(Dtype::Bfloat16)?;
    let w = A::from_slice(&values(65 * 128, 19), &[65, 128]).as_dtype(Dtype::Bfloat16)?;
    let selected = a.t.narrow(0, 1, 2)?;
    for reference in [0, 1, 2, 7, 9, 16] {
        let before = crate::ops::runtime_dispatch_count();
        assert!(bf16_small_row_linear_canonical(&rt, &selected, &w.t, reference).is_err());
        assert_eq!(before, crate::ops::runtime_dispatch_count());
    }
    let before = crate::ops::runtime_dispatch_count();
    assert!(bf16_small_row_linear_canonical(&rt, &a.t, &w.t, 8).is_err());
    let foreign = std::sync::Arc::new(crate::runtime::MetalRuntime::new(4)?);
    let foreign_view = Tensor::from_parts_offset(
        &foreign,
        selected.buffer().clone(),
        selected.shape(),
        selected.dtype(),
        selected.layout().offset,
    );
    assert!(bf16_small_row_linear_canonical(&rt, &foreign_view, &w.t, 8).is_err());
    let malformed = Tensor::from_parts_offset(
        &rt,
        selected.buffer().clone(),
        selected.shape(),
        selected.dtype(),
        usize::MAX,
    );
    assert!(bf16_small_row_linear_canonical(&rt, &malformed, &w.t, 8).is_err());
    assert_eq!(before, crate::ops::runtime_dispatch_count());
    Ok(())
}
