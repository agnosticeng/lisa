#![cfg(test)]
use anyhow::Result;

#[test]
fn glm53_gguf_indexed_attention_short_and_wide_windows() -> Result<()> {
    use lisa_metal::kernels::sparse::gguf as gpu;
    use lisa_metal::{Array, Dtype, Stream};
    let s = Stream::thread_local_or_default();
    let q = Array::from_slice(&vec![0f32; 64 * 256], &[1, 1, 16384]);
    let low = Array::from_slice(&vec![0f32; 64 * 512], &[1, 1, 64, 512]);
    let mut packed = vec![1u8; 64 * 256 * 544];
    for b in packed.as_chunks_mut::<34>().0.iter_mut() {
        b[..2].copy_from_slice(&0x3c00u16.to_le_bytes());
    }
    let weights = Array::from_slice(&packed, &[64, 256, 544]);
    for n in [1, 7, 2051, 2052, 4097] {
        let latent = Array::from_slice(&vec![0.5f32; n * 512], &[1, n as i32, 512])
            .as_dtype(Dtype::Float16)?;
        let selected = Array::from_slice(&(0..n as i32).collect::<Vec<_>>(), &[1, 1, n as i32]);
        let result = if n <= 2051 {
            gpu::decode(&low, &latent, &selected, &weights, false, &s)?
        } else {
            {
                let lora = lisa_metal::kernels::sparse::kernels::dsa_indexed_lora_wide_native(
                    &q.reshape(&[1, 64, 1, 256])?,
                    &low,
                    &latent,
                    &selected,
                    0.0625,
                    &s,
                )
                .expect("wide attention");
                gpu::value(&lora, &weights, &s)?
            }
        };
        for x in result.to_vec1::<f32>()? {
            assert!((x - 256.0).abs() < 0.002, "window={n}, value={x}");
        }
    }
    Ok(())
}
