//! Causal verification must preserve each ordinary decode prefix's arithmetic.
use super::*;
use crate::{Array, ops::indexing::IndexOp};

fn fixture(shape: &[i32], mut seed: u32) -> Array {
    let data = (0..shape.iter().product::<i32>())
        .map(|_| {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            half::bf16::from_f32((seed as i32 as f32) / i32::MAX as f32)
        })
        .collect::<Vec<_>>();
    Array::from_slice(&data, shape)
}

fn same_bits(label: &str, actual: &Array, expected: &Array) -> Result<bool> {
    let a = actual.to_vec1::<u16>()?;
    let e = expected.to_vec1::<u16>()?;
    assert_eq!(a.len(), e.len(), "{label} element count");
    let different = a.iter().zip(&e).filter(|(a, e)| a != e).count();
    if let Some(i) = a.iter().zip(&e).position(|(a, e)| a != e) {
        eprintln!(
            "{label}: {different}/{} differ; first[{i}]={:#06x} expected {:#06x}",
            a.len(),
            a[i],
            e[i]
        );
    }
    Ok(different == 0)
}

#[test]
fn serial_partition_selection_uses_single_query_geometry() {
    assert_eq!(serial_vector_blocks('s', 6, 1023), 0);
    assert_eq!(serial_vector_blocks('s', 6, 1024), 64);
    assert_eq!(serial_vector_blocks('s', 6, 1025), 128);
    assert_eq!(serial_vector_blocks('s', 6, 8192), 128);
    assert_eq!(serial_vector_blocks('s', 6, 8193), 256);
    assert_eq!(serial_vector_blocks('s', 4, 8193), 64);
    assert_eq!(serial_vector_blocks('d', 6, 16383), 128);
    assert_eq!(serial_vector_blocks('d', 6, 16384), 512);
    assert_eq!(serial_vector_blocks('d', 2, 8192), 128);
    assert_eq!(serial_vector_blocks('d', 2, 8193), 256);
    assert_eq!(serial_vector_blocks('g', 6, 4095), 0);
    assert_eq!(serial_vector_blocks('g', 6, 4096), 64);
    assert_eq!(serial_vector_blocks('g', 2, 4096), 32);
    assert_eq!(serial_vector_blocks('g', 1, 8192), 0);
}

#[test]
fn captured_causal_attention_matches_serial_prefixes() -> Result<()> {
    let mut exact = true;
    for (heads, key_heads) in [(24, 4), (8, 2)] {
        for length in [41, 1024, 2048, 8192, 8200] {
            // Offset/strided cache views reproduce a live cache's larger capacity.
            let k =
                fixture(&[1, key_heads, length + 9, 256], 113).index((.., .., 3..length + 3, ..));
            let v =
                fixture(&[1, key_heads, length + 11, 256], 197).index((.., .., 5..length + 5, ..));
            for rows in [1, 2, 3, 5, 6, 7, 8, 16] {
                let q = fixture(&[1, heads, rows, 256], 251);
                let device = q.t.device();
                let mut decoded = Vec::new();
                for row in 0..rows {
                    let qr = q.index((.., .., row..row + 1, ..)).contiguous()?;
                    let visible = length - rows + row + 1;
                    let kr = k.index((.., .., 0..visible, ..));
                    let vr = v.index((.., .., 0..visible, ..));
                    decoded.push(Array::new(sdpa(
                        device, &qr.t, &kr.t, &vr.t, 0.0625, true, None,
                    )?));
                }
                let expected = Array::concatenate(&decoded.iter().collect::<Vec<_>>(), 2)?;
                let actual = Array::new(sdpa_serial_causal(device, &q.t, &k.t, &v.t, 0.0625)?);
                let label = format!("heads={heads}/{key_heads} KV={length} S={rows}");
                exact &= same_bits(&format!("dispatch {label}"), &actual, &expected)?;
            }
        }
    }
    assert!(
        exact,
        "captured attention must preserve serial prefix arithmetic"
    );
    Ok(())
}

#[test]
fn batched_captured_attention_preserves_strided_serial_prefixes() -> Result<()> {
    let mut exact = true;
    for (heads, key_heads) in [(24, 4), (8, 2)] {
        for length in [1024, 8192] {
            let k =
                fixture(&[2, key_heads, length + 9, 256], 313).index((.., .., 3..length + 3, ..));
            let v =
                fixture(&[2, key_heads, length + 11, 256], 397).index((.., .., 5..length + 5, ..));
            for rows in [2, 7] {
                // Both batch and head strides include padding outside the view.
                let q = fixture(&[2, heads, rows + 3, 256], 451).index((.., .., 1..rows + 1, ..));
                let device = q.t.device();
                let mut decoded = Vec::new();
                for row in 0..rows {
                    let qr = q.index((.., .., row..row + 1, ..)).contiguous()?;
                    let visible = length - rows + row + 1;
                    let kr = k.index((.., .., 0..visible, ..));
                    let vr = v.index((.., .., 0..visible, ..));
                    decoded.push(Array::new(sdpa(
                        device, &qr.t, &kr.t, &vr.t, 0.0625, true, None,
                    )?));
                }
                let expected = Array::concatenate(&decoded.iter().collect::<Vec<_>>(), 2)?;
                let actual = Array::new(sdpa_serial_causal(device, &q.t, &k.t, &v.t, 0.0625)?);
                exact &= same_bits(
                    &format!("B=2 heads={heads}/{key_heads} KV={length} S={rows}"),
                    &actual,
                    &expected,
                )?;
            }
        }
    }
    assert!(
        exact,
        "batched captured attention must preserve serial prefix arithmetic"
    );
    Ok(())
}
