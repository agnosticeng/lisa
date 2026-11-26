//! Direct serial/wide projection differential before recurrent-state updates.
use super::*;
use crate::ops::indexing::IndexOp;

#[test]
fn captured_projections_preserve_single_row_reductions() -> Result<()> {
    let mut state = 0x438f_193du32;
    let mut random = || {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        state
    };
    let mut exact = true;
    for k in [1536, 5120, 6144, 17408] {
        for bits in [2, 4, 8] {
            let n = 64;
            let codes = Array::from_slice(
                &(0..n * k * bits / 32).map(|_| random()).collect::<Vec<_>>(),
                &[n, k * bits / 32],
            );
            let mut metadata = |bias: f32| {
                Array::from_slice(
                    &(0..n * k / 64)
                        .map(|_| half::bf16::from_f32(bias + (random() % 1024) as f32 * 0.00001))
                        .collect::<Vec<_>>(),
                    &[n, k / 64],
                )
            };
            let scales = metadata(0.001);
            let biases = metadata(-0.05);
            for (batch, rows) in [(1, 1), (1, 2), (1, 7), (1, 16), (2, 7), (2, 16)] {
                let count = batch * rows;
                let x = Array::from_slice(
                    &(0..count * k)
                        .map(|_| half::bf16::from_f32(random() as i32 as f32 / i32::MAX as f32))
                        .collect::<Vec<_>>(),
                    &[batch, rows, k],
                );
                let flat = x.reshape(&[count, k])?;
                let serial = (0..count)
                    .map(|row| {
                        quantized_matmul(
                            &flat.index((row..row + 1, ..)),
                            &codes,
                            &scales,
                            Some(&biases),
                            true,
                            64,
                            bits,
                        )
                    })
                    .collect::<Result<Vec<_>>>()?;
                let serial =
                    Array::concatenate(&serial.iter().collect::<Vec<_>>(), 0)?.to_vec1::<u16>()?;
                let actual = quantized_matmul_serial_capture(
                    &x,
                    &codes,
                    &scales,
                    Some(&biases),
                    true,
                    64,
                    bits,
                    false,
                )?;
                assert_eq!(actual.shape(), &[batch, rows, n]);
                let actual = actual.to_vec1::<u16>()?;
                let count = actual.iter().zip(&serial).filter(|(a, b)| a != b).count();
                if let Some(i) = actual.iter().zip(&serial).position(|(a, b)| a != b) {
                    eprintln!(
                        "K={k} bits={bits} B={batch} S={rows}: {count}/{} differ; first[{i}]={:#06x} expected {:#06x}",
                        actual.len(),
                        actual[i],
                        serial[i]
                    );
                }
                exact &= count == 0;
            }
        }
    }
    assert!(
        exact,
        "captured projections must preserve single-row reductions"
    );
    Ok(())
}

#[test]
fn serial_projection_exception_requires_flag_scope_and_single_stream() -> Result<()> {
    let (k, n, bits) = (1536, 512, 4);
    let mut state = 0x7913_b42fu32;
    let mut random = || {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        state
    };
    let codes = Array::from_slice(
        &(0..n * k * bits / 32).map(|_| random()).collect::<Vec<_>>(),
        &[n, k * bits / 32],
    );
    let scales = Array::from_slice(
        &vec![half::bf16::from_f32(0.007); (n * k / 64) as usize],
        &[n, k / 64],
    );
    let biases = Array::from_slice(
        &vec![half::bf16::from_f32(-0.049); (n * k / 64) as usize],
        &[n, k / 64],
    );
    for (batch, rows) in [(1, 7), (2, 2), (1, 16)] {
        let count = batch * rows;
        let x = Array::from_slice(
            &(0..count * k)
                .map(|_| half::bf16::from_f32(random() as i32 as f32 / i32::MAX as f32))
                .collect::<Vec<_>>(),
            &[batch, rows, k],
        );
        let flat = x.reshape(&[count, k])?;
        let serial = (0..count)
            .map(|row| {
                quantized_matmul(
                    &flat.index((row..row + 1, ..)),
                    &codes,
                    &scales,
                    Some(&biases),
                    true,
                    64,
                    bits,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let serial = Array::concatenate(&serial.iter().collect::<Vec<_>>(), 0)?.to_vec1::<u16>()?;
        let project = |allow| {
            quantized_matmul_serial_capture(
                &x,
                &codes,
                &scales,
                Some(&biases),
                true,
                64,
                bits,
                allow,
            )?
            .to_vec1::<u16>()
        };
        assert!(project(true)? == serial, "unscoped B={batch} S={rows}");
        let _guard = enter_verify_splitk_scope();
        assert!(project(false)? == serial, "disabled B={batch} S={rows}");
        if batch == 1 && rows <= 8 {
            let expected = crate::jit::affine_verify_qmm_splitk(
                x.t.device(),
                &flat.t,
                &codes.t,
                &scales.t,
                &biases.t,
                64,
                bits,
            )?;
            let expected = Array::new(expected).to_vec1::<u16>()?;
            assert!(project(true)? == expected, "admitted verify scope");
        } else {
            assert!(
                project(true)? == serial,
                "strict geometry B={batch} S={rows}"
            );
        }
    }
    Ok(())
}
