#![cfg(test)]
use super::*;

#[test]
fn glm53_kda_projection_uses_eight_row_boundary_and_preserves_shape() -> anyhow::Result<()> {
    if !lisa_metal::device::metal_available() {
        eprintln!("[glm53-kda-projection] skipping: no Metal device available");
        return Ok(());
    }
    let (k, n) = (323usize, 9usize);
    let weights_f32: Vec<f32> = (0..n * k)
        .map(|i| (((i * 7 + 3) % 53) as f32 - 26.0) * 0.003)
        .collect();
    let weights =
        Array::from_slice(&weights_f32, &[n as i32, k as i32]).as_dtype(Dtype::Bfloat16)?;
    let rounded_weights = weights.as_dtype(Dtype::Float32)?.to_vec1::<f32>()?;

    // Eight rows use the reference's row GEMV. Nine rows cross into its tiled GEMM,
    // which rounds both operands through F16 before F32 accumulation.
    for rows in [8usize, 9] {
        let x_values: Vec<f32> = (0..rows * k)
            .map(|i| (((i * 11 + 5) % 47) as f32 - 23.0) * 0.002)
            .collect();
        let x = Array::from_slice(&x_values, &[1, rows as i32, k as i32]);
        let got = linear_projection(&x, &weights)?;
        assert_eq!(got.shape(), [1, rows as i32, n as i32]);
        let got = got.to_vec1::<f32>()?;
        for row in 0..rows {
            for out in 0..n {
                let mut want = 0.0f64;
                for inner in 0..k {
                    let (weight, activation) = if rows > 8 {
                        (
                            half::f16::from_f32(rounded_weights[out * k + inner]).to_f32(),
                            half::f16::from_f32(x_values[row * k + inner]).to_f32(),
                        )
                    } else {
                        (rounded_weights[out * k + inner], x_values[row * k + inner])
                    };
                    want += f64::from(weight) * f64::from(activation);
                }
                assert!(
                    (f64::from(got[row * n + out]) - want).abs() < 2e-4,
                    "rows={rows} output=({row},{out}): {} vs {want}",
                    got[row * n + out]
                );
            }
        }
    }
    Ok(())
}
