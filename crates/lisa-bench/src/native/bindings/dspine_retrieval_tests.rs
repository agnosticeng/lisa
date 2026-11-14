#![cfg(test)]
use crate::core::config::ReadConfig;
use crate::core::quant::QuantizedLinearExecution;
use lisa_metal::ops;
#[cfg(test)]
fn rescore(hidden: &Array, target: &QuantizedLinear, ids: &Array) -> anyhow::Result<Array> {
    let spec = Rescore {
        rows: ids.dim(0),
        count: ids.dim(1),
        group: target.group_size,
        bits: target.bits,
    };
    bound::rescore(
        spec,
        &[hidden, &target.weight, &target.scales, &target.biases, ids],
        false,
    )
}

use super::*;
use crate::bindings::dspine_retrieval_plans as bound;
use lisa_metal::Dtype;
use lisa_models::qwen3_5::dspine::retrieval::Rescore;

#[test]
#[ignore = "requires an explicit local Qwen 27B target checkpoint"]
fn coarse_retrieval_real_head_rescore_bits() -> anyhow::Result<()> {
    use crate::core::loader::{Checkpoint, Shard, sanitize_name};
    use crate::core::quant::get_tensor;
    let model =
        std::env::var("LISA_QWEN_DSPINE_TARGET_MODEL").context("target checkpoint required")?;
    let path = crate::checkpoint::resolve_model_dir(&model)?;
    let config = crate::bindings::qwen35::Qwen35Config::from_json(&path.join("config.json"))?;
    let checkpoint = Checkpoint::open(&path)?;
    let load = |suffix: &str| -> anyhow::Result<Array> {
        let key = if config.tie_word_embeddings {
            format!("model.embed_tokens.{suffix}")
        } else {
            format!("lm_head.{suffix}")
        };
        let (name, shard) = checkpoint
            .weight_map
            .iter()
            .find(|(name, _)| sanitize_name(name).as_deref() == Some(&key))
            .with_context(|| format!("missing target tensor {key}"))?;
        get_tensor(&Shard::open(&path.join(shard))?, name)
    };
    let target = QuantizedLinear {
        weight: load("weight")?,
        scales: load("scales")?,
        biases: load("biases")?,
        group_size: config.quant_group_size,
        bits: config.quant_bits,
    };
    let h = config.hidden_size as i32;
    for rows in [1, 6, 8, 12] {
        let x: Vec<f32> = (0..rows * h)
            .map(|i| (((i * 13 + 41) % 251) as f32 - 125.) / 512.)
            .collect();
        let x = Array::from_slice(&x, &[rows, h]).as_dtype(Dtype::Bfloat16)?;
        let ids: Vec<u32> = (0..rows * 128)
            .map(|i| ((i as usize * 7879 + 19) % config.vocab_size) as u32)
            .collect();
        let ids = Array::from_slice(&ids, &[rows, 128]);
        let expected = target.forward(&x)?.take_along_axis(&ids, 1)?;
        assert_eq!(
            rescore(&x, &target, &ids)?.to_vec1::<u16>()?,
            expected.to_vec1::<u16>()?
        );
    }
    Ok(())
}

#[test]
fn coarse_retrieval_rescores_original_head_bits() -> anyhow::Result<()> {
    let h = 5120;
    let vocab = 4096;
    let values: Vec<f32> = (0..vocab * h)
        .map(|i| (((i * 37 + 11) % 257) as f32 - 128.) / 256.)
        .collect();
    let dense = Array::from_slice(&values, &[vocab, h]).as_dtype(Dtype::Bfloat16)?;
    for bits in [2, 4] {
        let (weight, scales, biases) = ops::quantize(&dense, 64, bits)?;
        let target = QuantizedLinear {
            weight,
            scales,
            biases,
            group_size: 64,
            bits,
        };
        for rows in [1, 2, 6, 7, 8, 12] {
            let x: Vec<f32> = (0..rows * h)
                .map(|i| (((i * 13 + 41) % 251) as f32 - 125.) / 512.)
                .collect();
            let x = Array::from_slice(&x, &[rows, h]).as_dtype(Dtype::Bfloat16)?;
            let ids: Vec<u32> = (0..rows * 128)
                .map(|i| ((i * 47 + 19) % vocab) as u32)
                .collect();
            let ids = Array::from_slice(&ids, &[rows, 128]);
            let expected = target.forward(&x)?.take_along_axis(&ids, 1)?;
            assert_eq!(
                rescore(&x, &target, &ids)?.to_vec1::<u16>()?,
                expected.to_vec1::<u16>()?
            );
        }
        let coarse = Retrieval::build(crate::metal_context::execution_context(), &target)?;
        if bits == 2 {
            assert!(coarse.is_none());
        } else {
            let coarse = coarse.unwrap();
            let restored =
                ops::dequantize(&target.weight, &target.scales, &target.biases, 64, bits)?;
            let (w, s, b) = ops::quantize(&restored, 64, 2)?;
            assert_eq!(coarse.head.weight.to_vec1::<u32>()?, w.to_vec1::<u32>()?);
            assert_eq!(coarse.head.scales.to_vec1::<u16>()?, s.to_vec1::<u16>()?);
            assert_eq!(coarse.head.biases.to_vec1::<u16>()?, b.to_vec1::<u16>()?);
        }
    }
    Ok(())
}
