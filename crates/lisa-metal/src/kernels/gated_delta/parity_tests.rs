//! Cross-schedule checks: the serial fused decoder is the state oracle.
use super::*;
use crate::error::Result;

fn values(shape: &[i32], scale: f32, bias: f32, dtype: Dtype) -> Result<Array> {
    let len: i32 = shape.iter().product();
    Array::from_slice(
        &(0..len)
            .map(|i| (((i * 37 + 11) % 257) as f32 - 128.) * scale + bias)
            .collect::<Vec<_>>(),
        shape,
    )
    .as_dtype(dtype)
}
fn same_bits(label: &str, actual: &Array, expected: &Array) -> Result<bool> {
    let bits = |x: &Array| -> Result<Vec<u32>> {
        if x.dtype() == Dtype::Float32 {
            x.to_vec1::<u32>()
        } else {
            Ok(x.to_vec1::<u16>()?.into_iter().map(u32::from).collect())
        }
    };
    let actual = bits(actual)?;
    let expected = bits(expected)?;
    assert_eq!(actual.len(), expected.len(), "{label} element count");
    let count = actual.iter().zip(&expected).filter(|(a, b)| a != b).count();
    if let Some(index) = actual.iter().zip(&expected).position(|(a, b)| a != b) {
        eprintln!(
            "{label}: {count}/{} bits differ, first[{index}]={:#010x} expected {:#010x}",
            actual.len(),
            actual[index],
            expected[index]
        );
    }
    Ok(count == 0)
}

#[test]
fn short_capture_preserves_complete_convolution_history() -> Result<()> {
    let stream = Stream::thread_local_or_default();
    let (hk, hv, kc) = (4, 16, 4);
    let channels = (2 * hk + hv) * 128;
    let gate = channels;
    let beta = gate + hv * 128;
    let decay = beta + hv;
    let width = decay + hv;
    let mut exact = true;
    for batch in [1, 2] {
        for rows in [1, 2, 3, 7] {
            let projection = values(&[batch, rows, width], 0.0023, 0.1, Dtype::Bfloat16)?;
            let history = values(&[batch, kc - 1, channels], 0.0031, 0.7, Dtype::Bfloat16)?;
            let weights = values(&[channels, kc, 1], 0.0013, 0.15, Dtype::Bfloat16)?;
            let a = values(&[hv], 0.0007, -0.3, Dtype::Float32)?;
            let bias = values(&[hv], 0.0031, 0.1, Dtype::Bfloat16)?;
            let (_, _, _, _, _, next, captured) = gdn_prep_fused(
                &projection,
                &history,
                &weights,
                &a,
                &bias,
                beta,
                decay,
                rows,
                hk,
                hv,
                128,
                128,
                kc,
                width,
                channels,
                &stream,
            )
            .expect("supported GDN prep");
            let projected_rows = projection.index((.., .., 0..channels)).contiguous()?;
            let expected = Array::concatenate(&[&history, &projected_rows], 1)?;
            exact &= same_bits(
                &format!("B={batch} S={rows} captured convolution"),
                &captured,
                &expected,
            )?;
            for keep in 1..=rows {
                let restored = captured.index((.., keep..keep + kc - 1, ..)).contiguous()?;
                let expected_prefix = expected.index((.., keep..keep + kc - 1, ..)).contiguous()?;
                exact &= same_bits(
                    &format!("B={batch} S={rows} keep={keep}"),
                    &restored,
                    &expected_prefix,
                )?;
            }
            let last = expected.index((.., rows..rows + kc - 1, ..)).contiguous()?;
            exact &= same_bits(
                &format!("B={batch} S={rows} final convolution"),
                &next,
                &last,
            )?;
        }
    }
    assert!(exact, "capture must preserve every convolution history row");
    Ok(())
}

#[test]
fn serial_decode_matches_every_captured_state_prefix() -> Result<()> {
    let stream = Stream::thread_local_or_default();
    let mut exact = true;
    for (hk, hv) in [(4, 16), (16, 48)] {
        let kc = 4;
        let channels = (2 * hk + hv) * 128;
        let gate = channels;
        let beta = gate + hv * 128;
        let decay = beta + hv;
        let width = decay + hv;
        for rows in [1, 2, 7] {
            for swish in [false, true] {
                let projection = values(&[1, rows, width], 0.0023, 0.1, Dtype::Bfloat16)?;
                let history = values(&[1, kc - 1, channels], 0.0031, 0.7, Dtype::Bfloat16)?;
                let weights = values(&[channels, kc, 1], 0.0013, 0.15, Dtype::Bfloat16)?;
                let a = values(&[hv], 0.0007, -0.3, Dtype::Float32)?;
                let bias = values(&[hv], 0.0031, 0.1, Dtype::Bfloat16)?;
                let state = values(&[1, hv, 128, 128], 0.0023, 0., Dtype::Float32)?;
                let norm = values(&[128], 0.0019, 1., Dtype::Bfloat16)?;
                let (q, k, v, g, b, conv, _) = gdn_prep_fused(
                    &projection,
                    &history,
                    &weights,
                    &a,
                    &bias,
                    beta,
                    decay,
                    rows,
                    hk,
                    hv,
                    128,
                    128,
                    kc,
                    width,
                    channels,
                    &stream,
                )
                .expect("supported GDN prep");
                let z = projection.index((.., .., gate..gate + hv * 128));
                let (output, snapshots) = if rows > 1 && swish {
                    gdn_verify_fused_with_serial_gate(
                        &q, &k, &v, &g, &b, &state, &z, &norm, 1e-6, None, true, &stream,
                    )
                    .expect("supported fused verification")
                } else {
                    let (raw, snapshots) = if rows > 1 {
                        gdn_two_row(&q, &k, &v, &g, &b, &state, true, &stream)
                    } else {
                        gated_delta_kernel(&q, &k, &v, &g, &b, &state, true, &stream)
                    }
                    .expect("supported scalar/paired capture");
                    let output = if swish {
                        crate::kernels::gated_rms_silu_with_serial_gate(
                            &raw, &z, &norm, hv, 128, 1e-6, true, &stream,
                        )
                    } else {
                        crate::kernels::gated_rms(&raw, &z, &norm, hv, 128, 1e-6, &stream)
                    }
                    .expect("supported gated RMS");
                    (output, snapshots)
                };
                let mut serial_state = state;
                let mut serial_conv = history;
                for t in 0..rows {
                    let token = projection.index((.., t..t + 1, ..)).contiguous()?;
                    let (activation, next_state, next_conv) = delta_decode_checked(
                        &token,
                        &serial_conv,
                        &weights,
                        &a,
                        &bias,
                        &serial_state,
                        &norm,
                        hk,
                        hv,
                        128,
                        128,
                        kc,
                        channels,
                        width,
                        beta,
                        decay,
                        gate,
                        1e-6,
                        swish,
                        &stream,
                    )?;
                    let captured_state = snapshots.index((t..t + 1, .., .., ..)).contiguous()?;
                    let captured_output = output.index((.., t..t + 1, .., ..)).contiguous()?;
                    let label = format!("Hk={hk} Hv={hv} S={rows} prefix={} swish={swish}", t + 1);
                    exact &= same_bits(&format!("{label} state"), &captured_state, &next_state)?;
                    exact &= same_bits(
                        &format!("{label} activation"),
                        &captured_output,
                        &activation,
                    )?;
                    serial_state = next_state;
                    serial_conv = next_conv;
                }
                exact &= same_bits("serial/captured final convolution", &conv, &serial_conv)?;
            }
        }
    }
    assert!(
        exact,
        "captured prefixes must preserve serial decode activation and state bits"
    );
    Ok(())
}

#[test]
fn serial_decode_matches_capture_for_all_finite_bf16_swish_gates() -> Result<()> {
    finite_bf16_capture_gates(true)
}

#[test]
fn serial_decode_matches_capture_for_all_finite_bf16_sigmoid_gates() -> Result<()> {
    finite_bf16_capture_gates(false)
}

fn finite_bf16_capture_gates(swish: bool) -> Result<()> {
    let stream = Stream::thread_local_or_default();
    let (hk, hv, kc) = if swish { (16, 48, 4) } else { (4, 16, 4) };
    let channels = (2 * hk + hv) * 128;
    let gate = channels;
    let beta = gate + hv * 128;
    let decay = beta + hv;
    let width = decay + hv;
    let history = values(&[1, kc - 1, channels], 0.0031, 0.7, Dtype::Bfloat16)?;
    let weights = values(&[channels, kc, 1], 0.0013, 0.15, Dtype::Bfloat16)?;
    let a = values(&[hv], 0.0007, -0.3, Dtype::Float32)?;
    let bias = values(&[hv], 0.0031, 0.1, Dtype::Bfloat16)?;
    let state = values(&[1, hv, 128, 128], 0.0023, 0., Dtype::Float32)?;
    let norm = values(&[128], 0.0019, 1., Dtype::Bfloat16)?;
    let finite_gates: Vec<_> = (0..=u16::MAX)
        .map(half::bf16::from_bits)
        .filter(|value| value.is_finite())
        .collect();
    let mut exact = true;
    for rows in [1, 2, 7] {
        let mut projection_values: Vec<_> = (0..rows * width)
            .map(|i| half::bf16::from_f32((((i * 37 + 11) % 257) as f32 - 128.) * 0.0023 + 0.1))
            .collect();
        for (chunk, gates) in finite_gates.chunks((rows * hv * 128) as usize).enumerate() {
            // Only z changes: equal recurrent states isolate the gate arithmetic.
            // Check every finite BF16 input for scalar and fused wide capture.
            for row in 0..rows as usize {
                let start = row * (hv * 128) as usize;
                let end = (start + (hv * 128) as usize).min(gates.len());
                let values = &gates[start.min(gates.len())..end];
                let offset = row * width as usize + gate as usize;
                let gate_values = &mut projection_values[offset..offset + (hv * 128) as usize];
                gate_values.fill(half::bf16::from_f32(0.));
                gate_values[..values.len()].copy_from_slice(values);
            }
            let projection = Array::from_slice(&projection_values, &[1, rows, width]);
            let (q, k, v, g, b, conv, _) = gdn_prep_fused(
                &projection,
                &history,
                &weights,
                &a,
                &bias,
                beta,
                decay,
                rows,
                hk,
                hv,
                128,
                128,
                kc,
                width,
                channels,
                &stream,
            )
            .expect("supported GDN prep");
            let z = projection.index((.., .., gate..beta));
            let (captured, snapshots) = if rows == 1 || !swish {
                let (raw, snapshots) = if rows == 1 {
                    gated_delta_kernel(&q, &k, &v, &g, &b, &state, true, &stream)
                } else {
                    gdn_two_row(&q, &k, &v, &g, &b, &state, true, &stream)
                }
                .expect("supported scalar/paired capture");
                let output = if swish {
                    crate::kernels::gated_rms_silu_with_serial_gate(
                        &raw, &z, &norm, hv, 128, 1e-6, true, &stream,
                    )
                } else {
                    crate::kernels::gated_rms(&raw, &z, &norm, hv, 128, 1e-6, &stream)
                }
                .expect("supported gated RMS");
                (output, snapshots)
            } else {
                gdn_verify_fused_with_serial_gate(
                    &q, &k, &v, &g, &b, &state, &z, &norm, 1e-6, None, true, &stream,
                )
                .expect("supported fused verification")
            };
            let mut serial_state = state.clone();
            let mut serial_conv = history.clone();
            for row in 0..rows {
                let token = projection.index((.., row..row + 1, ..)).contiguous()?;
                let (serial, next_state, next_conv) = delta_decode_checked(
                    &token,
                    &serial_conv,
                    &weights,
                    &a,
                    &bias,
                    &serial_state,
                    &norm,
                    hk,
                    hv,
                    128,
                    128,
                    kc,
                    channels,
                    width,
                    beta,
                    decay,
                    gate,
                    1e-6,
                    swish,
                    &stream,
                )?;
                let label =
                    format!("finite bf16 gate S={rows} chunk={chunk} row={row} swish={swish}");
                let captured_state = snapshots.index((row..row + 1, .., .., ..)).contiguous()?;
                let captured_output = captured.index((.., row..row + 1, .., ..)).contiguous()?;
                exact &= same_bits(&format!("{label} state"), &captured_state, &next_state)?;
                let captured_bits = captured_output.to_vec1::<u16>()?;
                let serial_bits = serial.to_vec1::<u16>()?;
                if let Some(index) = captured_bits
                    .iter()
                    .zip(&serial_bits)
                    .position(|(a, b)| a != b)
                {
                    let z = projection_values[(row * width + gate) as usize + index];
                    eprintln!(
                        "{label}: first differing z={z:?}, bits={:#06x}",
                        z.to_bits()
                    );
                }
                exact &= same_bits(&format!("{label} activation"), &captured_output, &serial)?;
                serial_state = next_state;
                serial_conv = next_conv;
            }
            exact &= same_bits(
                &format!("finite bf16 gate S={rows} chunk={chunk} convolution"),
                &conv,
                &serial_conv,
            )?;
        }
    }
    assert!(
        exact,
        "finite bf16 output gates must preserve serial decode arithmetic"
    );
    Ok(())
}
