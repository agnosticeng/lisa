#![cfg(test)]
use super::{plan_oracle, weights::Bound};
use crate::core::cache::{FullAttentionCache, GdnCache, LayerCache};
use lisa_metal::{Array, Dtype};
use lisa_models::qwen3_5::gguf as model;
use std::collections::BTreeMap;
fn fixture(index: usize, h: i32) -> Bound {
    let p = format!("blk.{index}");
    let mut matrices = vec![
        ("ffn_gate", 512, h),
        ("ffn_up", 512, h),
        ("ffn_down", h, 512),
    ];
    let mut vectors = vec![("attn_norm.weight", h), ("post_attention_norm.weight", h)];
    if model::full_attention(index) {
        matrices.extend([
            ("attn_q", 12288, h),
            ("attn_k", 1024, h),
            ("attn_v", 1024, h),
            ("attn_output", h, 6144),
        ]);
        vectors.extend([("attn_q_norm.weight", 256), ("attn_k_norm.weight", 256)]);
    } else {
        matrices.extend([
            ("attn_qkv", 10240, h),
            ("attn_gate", 6144, h),
            ("ssm_alpha", 48, h),
            ("ssm_beta", 48, h),
            ("ssm_out", h, 6144),
        ]);
        vectors.extend([("ssm_a", 48), ("ssm_dt.bias", 48), ("ssm_norm.weight", 128)]);
    }
    if index == 64 {
        matrices.push(("nextn.eh_proj", 5120, 10240));
        vectors.extend([
            ("nextn.enorm.weight", 5120),
            ("nextn.hnorm.weight", 5120),
            ("nextn.shared_head_norm.weight", 5120),
        ]);
    }
    let mut arrays = BTreeMap::new();
    let mut schema = BTreeMap::new();
    let mut projections = BTreeMap::new();
    let mut seed = 8123u32;
    for (name, n, k) in matrices {
        let name = format!("{p}.{name}.weight");
        let mut bytes = vec![0u8; (n * k / 256 * 84) as usize];
        for block in bytes.chunks_exact_mut(84) {
            for value in &mut block[..80] {
                seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                *value = (seed >> 24) as u8;
            }
            block[80..82].copy_from_slice(&half::f16::from_f32(0.0005).to_le_bytes());
            block[82..84].copy_from_slice(&half::f16::from_f32(0.0003).to_le_bytes());
        }
        arrays.insert(name.clone(), Array::from_slice(&bytes, &[n, k / 256 * 84]));
        schema.insert(name.clone(), vec![n, k]);
        projections.insert(
            name,
            model::Projection {
                encoding: 10,
                input: k,
                output: n,
            },
        );
    }
    for (name, n) in vectors {
        let name = format!("{p}.{name}");
        let values = (0..n)
            .map(|i| {
                if name.ends_with("ssm_a") {
                    -0.1 - (i % 7) as f32 / 50.
                } else if name.ends_with("bias") {
                    (i % 11) as f32 / 50.
                } else {
                    0.9 + (i % 13) as f32 / 100.
                }
            })
            .collect::<Vec<_>>();
        arrays.insert(name.clone(), Array::from_slice(&values, &[n]));
        schema.insert(name, vec![n]);
    }
    if !model::full_attention(index) {
        let name = format!("{p}.ssm_conv1d.weight");
        let values = (0..40960)
            .map(|i| ((i * 7 % 31) as f32 - 15.) / 100.)
            .collect::<Vec<_>>();
        arrays.insert(name.clone(), Array::from_slice(&values, &[10240, 4]));
        schema.insert(name, vec![10240, 4]);
    }
    Bound {
        context: crate::metal_context::execution_context(),
        arrays,
        schema,
        projections,
    }
}
fn bits(a: &Array) -> anyhow::Result<Vec<u32>> {
    Ok(a.t.to_vec::<f32>()?.iter().map(|v| v.to_bits()).collect())
}
fn equal(a: &Array, b: &Array) -> anyhow::Result<()> {
    assert_eq!(a.shape(), b.shape());
    assert_eq!(bits(a)?, bits(b)?);
    Ok(())
}
#[test]
fn packed_layer_plans_preserve_f32_state_and_capture() -> anyhow::Result<()> {
    for index in [0, 3] {
        let weights = fixture(index, 256);
        let oracle = plan_oracle::Layer::load(&weights, index)?;
        let config = model::Config {
            eps: 1e-6,
            theta: 10_000_000.,
            context: 4096,
            eos: 0,
        };
        let mut cache = if index == 3 {
            LayerCache::Full(FullAttentionCache::with_dtype(
                crate::metal_context::execution_context(),
                4,
                256,
                Dtype::Float32,
            ))
        } else {
            LayerCache::Linear(GdnCache::new())
        };
        let mut actual_cache = if index == 3 {
            LayerCache::Full(FullAttentionCache::with_dtype(
                crate::metal_context::execution_context(),
                4,
                256,
                Dtype::Float32,
            ))
        } else {
            LayerCache::Linear(GdnCache::new())
        };
        for (rows, capture) in [(1, false), (3, false), (7, true), (1, false)] {
            let values = (0..rows * 256)
                .map(|i| ((i * 13 % 59) as f32 - 29.) / 31.7)
                .collect::<Vec<_>>();
            let x = Array::from_slice(&values, &[1, rows, 256]);
            let expected = oracle.forward(&x, &mut cache, capture, &config)?;
            let layout = match &actual_cache {
                LayerCache::Full(c) => model::Cache::Attention(c.append_plan(1, rows)),
                LayerCache::Linear(c) => model::Cache::Delta {
                    conv: c.conv.is_some(),
                    state: c.ssm.is_some(),
                    capture,
                },
                _ => unreachable!(),
            };
            let spec = model::Layer {
                index,
                rows,
                epsilon: config.eps.to_bits(),
                theta: config.theta.to_bits(),
                cache: layout,
            };
            let mut input = vec![&x];
            match &actual_cache {
                LayerCache::Full(c) => {
                    input.extend(c.keys.iter());
                    input.extend(c.values.iter());
                }
                LayerCache::Linear(c) => {
                    input.extend(c.conv.iter());
                    input.extend(c.ssm.iter());
                }
                _ => unreachable!(),
            }
            let mut out = weights
                .prepare(spec.program(&weights.schema, &weights.projections)?)?
                .run(&input, &mut [])?
                .into_iter();
            equal(&out.next().unwrap(), &expected)?;
            match (&cache, &mut actual_cache) {
                (LayerCache::Full(expected), LayerCache::Full(actual)) => {
                    actual.keys = out.next();
                    actual.values = out.next();
                    actual.offset = expected.offset;
                    actual.capacity = expected.capacity;
                    equal(
                        actual.keys.as_ref().unwrap(),
                        expected.keys.as_ref().unwrap(),
                    )?;
                    equal(
                        actual.values.as_ref().unwrap(),
                        expected.values.as_ref().unwrap(),
                    )?;
                }
                (LayerCache::Linear(expected), LayerCache::Linear(actual)) => {
                    actual.conv = out.next();
                    actual.ssm = out.next();
                    equal(
                        actual.conv.as_ref().unwrap(),
                        expected.conv.as_ref().unwrap(),
                    )?;
                    equal(actual.ssm.as_ref().unwrap(), expected.ssm.as_ref().unwrap())?;
                    if capture {
                        actual.capture_ssm = out.next();
                        actual.capture_conv_input = out.next();
                        equal(
                            actual.capture_ssm.as_ref().unwrap(),
                            expected.capture_ssm.as_ref().unwrap(),
                        )?;
                        equal(
                            actual.capture_conv_input.as_ref().unwrap(),
                            expected.capture_conv_input.as_ref().unwrap(),
                        )?;
                    }
                }
                _ => unreachable!(),
            }
        }
    }
    Ok(())
}

#[test]
fn packed_mtp_plans_preserve_input_last_row_and_cache() -> anyhow::Result<()> {
    use lisa_metal::{fast, ops::indexing::IndexOp};
    let weights = fixture(64, 5120);
    let oracle = plan_oracle::Layer::load(&weights, 64)?;
    let config = model::Config {
        eps: 1e-6,
        theta: 10_000_000.,
        context: 4096,
        eos: 0,
    };
    let mut expected_cache = LayerCache::Full(FullAttentionCache::with_dtype(
        crate::metal_context::execution_context(),
        4,
        256,
        Dtype::Float32,
    ));
    let mut actual_cache = FullAttentionCache::with_dtype(
        crate::metal_context::execution_context(),
        4,
        256,
        Dtype::Float32,
    );
    for rows in [1, 3, 7] {
        let values = (0..rows * 5120)
            .map(|i| ((i * 17 % 71) as f32 - 35.) / 37.1)
            .collect::<Vec<_>>();
        let embeddings = Array::from_slice(&values, &[1, rows, 5120]);
        let hidden = Array::from_slice(
            &values.iter().map(|v| v * -0.91).collect::<Vec<_>>(),
            &[1, rows, 5120],
        );
        let norm = |x: &Array, name: &str| {
            fast::rms_norm(
                x,
                Some(&weights.arrays[&format!("blk.64.nextn.{name}.weight")]),
                config.eps,
            )
        };
        let e = norm(&embeddings, "enorm")?;
        let h = norm(&hidden, "hnorm")?;
        let input = Array::concatenate(&[&e, &h], -1)?;
        let input = lisa_metal::kernels::blockwise::matvec(
            10,
            &weights.arrays["blk.64.nextn.eh_proj.weight"],
            &input.reshape(&[rows, 10240])?,
            &lisa_metal::Stream::thread_local_or_default(),
        )?
        .reshape(&[1, rows, 5120])?;
        let actual_input = weights.run(
            model::draft_input(&weights.schema, &weights.projections, rows, config.eps)?,
            &[&embeddings, &hidden],
        )?;
        equal(&actual_input, &input)?;
        let expected = oracle.forward(&input, &mut expected_cache, false, &config)?;
        let spec = model::Layer {
            index: 64,
            rows,
            epsilon: config.eps.to_bits(),
            theta: config.theta.to_bits(),
            cache: model::Cache::Attention(actual_cache.append_plan(1, rows)),
        };
        let mut inputs = vec![&actual_input];
        inputs.extend(actual_cache.keys.iter());
        inputs.extend(actual_cache.values.iter());
        let mut out = weights
            .prepare(spec.program(&weights.schema, &weights.projections)?)?
            .run(&inputs, &mut [])?
            .into_iter();
        let actual = out.next().unwrap();
        equal(&actual, &expected)?;
        let LayerCache::Full(expected_cache) = &expected_cache else {
            unreachable!()
        };
        actual_cache.keys = out.next();
        actual_cache.values = out.next();
        actual_cache.offset = expected_cache.offset;
        actual_cache.capacity = expected_cache.capacity;
        equal(
            actual_cache.keys.as_ref().unwrap(),
            expected_cache.keys.as_ref().unwrap(),
        )?;
        equal(
            actual_cache.values.as_ref().unwrap(),
            expected_cache.values.as_ref().unwrap(),
        )?;
        let expected = norm(
            &expected.index((.., rows - 1, ..)).contiguous()?,
            "shared_head_norm",
        )?;
        let actual = weights.run(
            model::draft_finish(&weights.schema, rows, config.eps)?,
            &[&actual],
        )?;
        equal(&actual, &expected)?;
    }
    Ok(())
}
