#![cfg(test)]
use super::*;
use lisa_metal::{Dtype, Stream};
#[path = "qwen4_ple_oracle.rs"]
mod original;
use original::OriginalPle;
fn matrix(output: i32, input: i32) -> anyhow::Result<QuantizedLinear> {
    Ok(QuantizedLinear {
        weight: Array::from_slice(
            &(0..output * input / 8)
                .map(|i| (i as u32).wrapping_mul(2654435761))
                .collect::<Vec<_>>(),
            &[output, input / 8],
        ),
        scales: Array::from_slice(
            &vec![0.0001f32; (output * input / 32) as usize],
            &[output, input / 32],
        )
        .as_dtype(Dtype::Bfloat16)?,
        biases: Array::from_slice(
            &vec![-0.0007f32; (output * input / 32) as usize],
            &[output, input / 32],
        )
        .as_dtype(Dtype::Bfloat16)?,
        group_size: 32,
        bits: 4,
    })
}
pub(crate) fn layer() -> anyhow::Result<PleLayer> {
    use std::io::Write;
    let constants = NgramConstants::new(3, 8, 248320, 19, 128, 4, 0, 0);
    let rows = constants.rows_per_shard * constants.shard_count;
    let mut file = tempfile::NamedTempFile::new()?;
    for i in 0..rows * 20 {
        file.write_all(&(i as u32).wrapping_mul(2654435761).to_le_bytes())?;
    }
    for _ in 0..rows * 5 {
        file.write_all(&bf16::from_f32(0.02).to_le_bytes())?;
    }
    for _ in 0..rows * 5 {
        file.write_all(&bf16::from_f32(-0.14).to_le_bytes())?;
    }
    file.flush()?;
    let map = Arc::new(unsafe { Mmap::map(file.as_file())? });
    let maps = (0..4)
        .map(|i| {
            Some(ShardMap {
                map: map.clone(),
                weight_off: i * constants.rows_per_shard * 80,
                scales_off: rows * 80 + i * constants.rows_per_shard * 10,
                biases_off: rows * 90 + i * constants.rows_per_shard * 10,
                rows: constants.rows_per_shard,
            })
        })
        .collect();
    let table = NgramTable {
        maps,
        rows_per_shard: constants.rows_per_shard,
        row_dims: 160,
    };
    let norm = || -> anyhow::Result<RmsNorm> {
        Ok(RmsNorm {
            weight: Array::from_slice(&vec![1f32; 10240], &[10240]).as_dtype(Dtype::Bfloat16)?,
            eps: 1e-6,
            group_size: Some(2560),
        })
    };
    Ok(PleLayer {
        context: crate::metal_context::execution_context(),
        embedding: NgramEmbedding {
            context: crate::metal_context::execution_context(),
            constants,
            table: Arc::new(std::sync::RwLock::new(table)),
            ngram_heads: 16,
            eos_token_id: 248044,
            head_dimensions: 160,
            ngram_size: 3,
            heads_per_ngram: 8,
        },
        key_proj: matrix(10240, 2560)?,
        value_proj: matrix(2560, 2560)?,
        norm_key: norm()?,
        norm_query: norm()?,
        norm_conv: norm()?,
        conv1d_weight: Array::from_slice(
            &(0..40960)
                .map(|i| ((i % 13) as f32 - 6.) / 53.)
                .collect::<Vec<_>>(),
            &[10240, 4, 1],
        )
        .as_dtype(Dtype::Bfloat16)?,
        hidden_size: 2560,
        hc_count: 4,
        dilation: 3,
        short_conv_state_length: 9,
        last_embed: None,
        last_gated: None,
    })
}
fn bits(a: &Array, b: &Array) -> anyhow::Result<()> {
    assert_eq!(a.shape(), b.shape());
    assert_eq!(a.dtype(), b.dtype());
    assert_eq!(a.to_vec1::<u16>()?, b.to_vec1::<u16>()?);
    Ok(())
}
#[test]
fn embedding_convolution_plans_preserve_history_captures_and_dispatch() -> anyhow::Result<()> {
    use crate::bindings::deepseek41_moe::tests::{counts, difference};
    let stream = Stream::gpu();
    let mut layer = layer()?;
    for batch in [1, 2] {
        let mut old_state = None;
        let mut new_state = None;
        let mut history = vec![vec![17, 248044]; batch];
        for rows in [1, 8, 1, 16, 3] {
            let ids = (0..batch)
                .map(|b| {
                    (0..rows)
                        .map(|t| {
                            if t == 2 {
                                248044
                            } else {
                                (b * 17 + t + 13) as i64
                            }
                        })
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();
            let x = Array::from_slice(
                &(0..batch * rows * 10240)
                    .map(|i| ((i * 17 % 251) as f32 - 125.) / 131.)
                    .collect::<Vec<_>>(),
                &[batch as i32, rows as i32, 10240],
            )
            .as_dtype(Dtype::Bfloat16)?;
            let mut old_full = None;
            let mut new_full = None;
            stream.runtime().commands.flush_and_wait()?;
            let before = counts();
            let a =
                layer.original_forward(&x, &ids, &history, &mut old_state, true, &mut old_full)?;
            stream.runtime().commands.flush_and_wait()?;
            let middle = counts();
            let embed = layer.last_embed.clone().unwrap();
            let gated = layer.last_gated.clone().unwrap();
            let b = layer.forward(&x, &ids, &history, &mut new_state, true, &mut new_full)?;
            stream.runtime().commands.flush_and_wait()?;
            let after = counts();
            bits(&a, &b)?;
            bits(&embed, layer.last_embed.as_ref().unwrap())?;
            bits(&gated, layer.last_gated.as_ref().unwrap())?;
            bits(old_state.as_ref().unwrap(), new_state.as_ref().unwrap())?;
            bits(old_full.as_ref().unwrap(), new_full.as_ref().unwrap())?;
            if !after.is_empty() {
                assert_eq!(
                    difference(&before, &middle),
                    difference(&middle, &after),
                    "batch={batch} rows={rows}"
                );
            }
            for (h, ids) in history.iter_mut().zip(&ids) {
                h.extend_from_slice(ids);
            }
        }
    }
    Ok(())
}

#[test]
fn ngram_mapping_validates_format_ranges_and_model_geometry() -> anyhow::Result<()> {
    use std::io::Write;
    let mut file = tempfile::NamedTempFile::new()?;
    let header = serde_json::to_vec(&serde_json::json!({
        "ngram.weight":{"dtype":"U32","shape":[4,20],"data_offsets":[0,320]},
        "ngram.scales":{"dtype":"BF16","shape":[4,5],"data_offsets":[320,360]},
        "ngram.biases":{"dtype":"BF16","shape":[4,5],"data_offsets":[360,400]}
    }))?;
    file.write_all(&(header.len() as u64).to_le_bytes())?;
    file.write_all(&header)?;
    let data = (0..400).map(|i| (i % 251) as u8).collect::<Vec<_>>();
    file.write_all(&data)?;
    file.flush()?;
    let mut table = NgramTable::open_merged(file.path(), 2)?;
    assert_eq!(table.rows_per_shard, 2);
    assert!(table.set_rows_per_shard(3).is_err());
    table.set_rows_per_shard(2)?;
    for (i, shard) in table.maps.iter().enumerate() {
        let shard = shard.as_ref().unwrap();
        assert_eq!(
            &shard.map[shard.weight_off..shard.weight_off + 160],
            &data[i * 160..i * 160 + 160]
        );
        assert_eq!(
            &shard.map[shard.scales_off..shard.scales_off + 20],
            &data[320 + i * 20..340 + i * 20]
        );
    }
    assert!(
        table
            .gather(&crate::metal_context::execution_context(), &[-1])
            .is_err()
    );
    assert!(
        table
            .gather(&crate::metal_context::execution_context(), &[4])
            .is_err()
    );
    assert!(NgramTable::open_merged(file.path(), 3).is_err());
    drop(table);
    file.as_file_mut()
        .set_len((8 + header.len() + 399) as u64)?;
    assert!(NgramTable::open_merged(file.path(), 2).is_err());
    Ok(())
}
