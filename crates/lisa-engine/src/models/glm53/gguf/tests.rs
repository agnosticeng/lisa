use super::*;
use crate::core::{
    generate,
    sampler::Sampler,
    session::{MtpDepth, Session},
    tokenizer::{GlmReasoning, Tokenizer, glm_generation_prompt},
};
use crate::models::LanguageModel;
use anyhow::{Context, Result};

/// Frozen from unmodified DS4 b3f50180, the pinned Q2 checkpoint, greedy
/// --nothink, the default system prompt, and a 64-token continuation.
#[test]
#[ignore = "requires LISA_GLM53_GGUF and exclusive GPU access"]
fn glm53_gguf_serial_and_mtp_match_ds4_twice() -> Result<()> {
    let path = std::env::var("LISA_GLM53_GGUF").context("set LISA_GLM53_GGUF")?;
    let path = std::path::Path::new(&path);
    let tok = Tokenizer::load(path)?;
    let prompt = glm_generation_prompt(
        "Write a Python function that adds two numbers.",
        Some("You are a helpful assistant"),
        GlmReasoning::Disabled,
    );
    let ids = tok.encode_special(&prompt)?;
    assert_eq!(ids.len(), 21);
    let expected: &[u32] = &[
        8419, 594, 264, 4285, 13020, 729, 429, 11362, 1378, 5109, 1447, 73022, 12663, 198, 750,
        912, 32807, 2877, 11, 293, 982, 262, 470, 264, 488, 293, 198, 13865, 19288, 334, 13306,
        10426, 25, 55798, 73022, 12663, 198, 1382, 284, 912, 32807, 7, 20, 11, 220, 18, 340, 1350,
        4456, 8, 220, 671, 9255, 25, 220, 23, 271, 1382, 284, 912, 32807, 7, 17, 13,
    ];
    let mut model = tower::Tower::load(path)?;
    model.set_context_capacity(256)?;
    generate::set_eos_ids_for_model(tok.im_end_ids.clone(), &model);
    for round in 0..2 {
        let mut sampler = Sampler {
            temperature: 0.0,
            ..Default::default()
        };
        let (serial, _) =
            generate::generate(&mut model, &ids, 64, &mut sampler, 0, &mut |_| Ok(()))?;
        assert_eq!(serial, expected, "serial round {round}");
        let mut session = Session::new(&mut model);
        let mtp = session.generate_mtp_depth(
            &mut model,
            &ids,
            64,
            MtpDepth::Fixed(2),
            None,
            None,
            &mut |_| Ok(()),
        )?;
        assert_eq!(mtp, expected, "MTP round {round}");
    }
    Ok(())
}

#[test]
fn glm53_gguf_indexed_attention_short_and_wide_windows() -> Result<()> {
    use lisa_mlx::models::glm53::gguf as gpu;
    use lisa_mlx::{Array, Dtype, Stream};
    let s = Stream::thread_local_or_default();
    let q = Array::from_slice(&vec![0f32; 64 * 256], &[1, 1, 16384]);
    let low = Array::from_slice(&vec![0f32; 64 * 512], &[1, 1, 64, 512]);
    let mut packed = vec![1u8; 64 * 256 * 544];
    for b in packed.chunks_exact_mut(34) {
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
            gpu::wide_decode(&q, &low, &latent, &selected, &weights, &s)?
        };
        for x in result.to_vec1::<f32>()? {
            assert!((x - 256.0).abs() < 0.002, "window={n}, value={x}");
        }
    }
    Ok(())
}
