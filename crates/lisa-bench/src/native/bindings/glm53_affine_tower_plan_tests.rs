#![cfg(test)]
use super::*;
use crate::bindings::glm53_oracle_helpers::mm;
use crate::core::quant::QuantizedEmbeddingExecution;
#[path = "glm53_affine_tower_oracle.rs"]
mod oracle;
fn values(shape: &[i32], seed: usize) -> Array {
    Array::from_slice(
        &(0..shape.iter().product::<i32>() as usize)
            .map(|i| ((i * 17 + seed) % 127) as f32 / 151. - 0.3)
            .collect::<Vec<_>>(),
        shape,
    )
}
fn equal(a: &Array, b: &Array) -> anyhow::Result<()> {
    assert_eq!(a.shape(), b.shape());
    assert_eq!(a.dtype(), b.dtype());
    if a.dtype().size_of() == 2 {
        assert_eq!(a.to_vec1::<u16>()?, b.to_vec1::<u16>()?);
    } else {
        assert_eq!(a.to_vec1::<u32>()?, b.to_vec1::<u32>()?);
    }
    Ok(())
}
fn check(
    old: impl FnOnce() -> anyhow::Result<Array>,
    new: impl FnOnce() -> anyhow::Result<Array>,
) -> anyhow::Result<()> {
    let a = old()?;
    let b = new()?;
    equal(&a, &b)?;
    Ok(())
}
fn quant(n: i32, k: i32, bits: i32, group: i32, seed: usize) -> QuantizedLinear {
    QuantizedLinear {
        weight: Array::from_slice(
            &(0..n * k * bits / 32)
                .map(|i| (i as u32).wrapping_mul(1729).wrapping_add(seed as u32))
                .collect::<Vec<_>>(),
            &[n, k * bits / 32],
        ),
        scales: Array::from_slice(&vec![0.0001f32; (n * k / group) as usize], &[n, k / group]),
        biases: Array::from_slice(&vec![-0.001f32; (n * k / group) as usize], &[n, k / group]),
        bits,
        group_size: group,
    }
}
#[test]
fn affine_tower_arithmetic_plans_preserve_original_chains() -> anyhow::Result<()> {
    for h in [128, 4096] {
        for (b, t) in [(1, 1), (2, 1), (1, 3), (1, 9), (2, 7)] {
            let x = values(&[b, t, h], 17);
            let w = values(&[h], 23);
            check(|| oracle::rms(&x, &w, 1e-5), || rms(&x, &w, 1e-5))?;
            check(
                || oracle::final_rms(&x, &w, 1e-5),
                || final_rms(&x, &w, 1e-5),
            )?;
            let state = values(&[b, t, 4, h], 29);
            check(|| oracle::final_hc_mean(&state), || final_hc_mean(&state))?;
            check(
                || {
                    Ok(x.expand_dims(2)?
                        .broadcast_to(&[b, t, 4, h])?
                        .contiguous()?)
                },
                || {
                    plan_one(
                        R::Expand {
                            batch: b as usize,
                            rows: t as usize,
                            streams: 4,
                            hidden: h as usize,
                        },
                        &[&x],
                    )
                },
            )?;
            check(
                || {
                    Ok(state
                        .contiguous()?
                        .mean_axis(2, true)?
                        .reshape(&[b, t, h])?)
                },
                || {
                    plan_one(
                        R::PrefixMean {
                            shape: state.shape().to_vec(),
                        },
                        &[&state],
                    )
                },
            )?;
            let head = quant(512, h, 8, 64, 31);
            check(
                || {
                    let flat = x.reshape(&[b * t, h])?;
                    Ok(head.forward_glm53_prefill(&flat)?.reshape(&[b, t, 512])?)
                },
                || plans::head(&head, &x),
            )?;
            let embed = QuantizedEmbedding {
                weight: head.weight,
                scales: head.scales,
                biases: head.biases,
                bits: 8,
                group_size: 64,
            };
            let ids = Array::from_slice(
                &(0..b * t).map(|i| (i * 37) % 512).collect::<Vec<_>>(),
                &[b, t],
            );
            check(
                || Ok(embed.forward(&ids)?.as_dtype(Dtype::Float32)?),
                || plans::embedding(&embed, &ids),
            )?;
        }
    }
    Ok(())
}
#[test]
fn affine_draft_input_and_residual_plans_preserve_original_chains() -> anyhow::Result<()> {
    let h = 4096;
    let w = quant(32, h, 8, 64, 37);
    let embed = QuantizedEmbedding {
        weight: w.weight,
        scales: w.scales,
        biases: w.biases,
        bits: 8,
        group_size: 64,
    };
    for dtype in [Dtype::Float32, Dtype::Bfloat16] {
        let target = values(&[1, 1, 4 * h], 41);
        let enorm = values(&[h], 43);
        let hnorm = values(&[h], 47);
        let eh = values(&[h, 2 * h], 53).as_dtype(dtype)?;
        let (a, b, c) = oracle::draft_inputs(
            &embed,
            &target,
            eh.clone(),
            enorm.clone(),
            hnorm.clone(),
            h as usize,
            4,
            7,
            1e-5,
        )?;
        let ids = Array::from_slice(&[7i32], &[1, 1]);
        let x = plans::embedding(&embed, &ids)?;
        let new_a = mtp_rms(&x, &enorm, 1e-5)?;
        let new_b = plan_one(
            R::DraftHidden {
                hidden: h as usize,
                streams: 4,
                epsilon: 1e-5f32.to_bits(),
            },
            &[&target, &hnorm],
        )?;
        let new_c = plan_one(
            R::DraftProject {
                hidden: h as usize,
                dtype: if dtype == Dtype::Float32 {
                    lisa_backend::ScalarType::F32
                } else {
                    lisa_backend::ScalarType::BF16
                },
            },
            &[&new_a, &new_b, &eh],
        )?;
        equal(&a, &new_a)?;
        equal(&b, &new_b)?;
        equal(&c, &new_c)?;
        let s = lisa_metal::Stream::thread_local_or_default();
        let attended = values(&[1, 1, h], 59);
        let (sum, norm) =
            lisa_metal::kernels::sparse::kernels::mtp_add_rms_norm(&c, &attended, &enorm, 1e-5, &s)
                .unwrap();
        let out = plans::run(
            R::AddNorm {
                epsilon: 1e-5f32.to_bits(),
            },
            &[&c, &attended, &enorm],
        )?;
        equal(&sum, &out[0])?;
        equal(&norm, &out[1])?;
        check(
            || Ok(lisa_metal::kernels::sparse::kernels::mtp_add3(&c, &a, &b, &s).unwrap()),
            || plan_one(R::AddThree, &[&c, &a, &b]),
        )?;
    }
    Ok(())
}
fn dense(h: i32, seed: usize) -> anyhow::Result<DenseMlp> {
    Ok(DenseMlp {
        context: crate::metal_context::execution_context(),
        gate_up: crate::bindings::glm53_dense_mlp::GateUpProjection::from_parts(
            &crate::metal_context::execution_context(),
            quant(512, h, 8, 64, seed),
            quant(512, h, 8, 64, seed + 1),
            true,
        )?,
        down: quant(h, 512, 8, 64, seed + 2),
        limit: 10.,
    })
}
fn hyper(h: i32, seed: usize) -> anyhow::Result<Hc> {
    let mut w = Weights::new();
    w.insert(
        "hc_fn".into(),
        values(&[24, 4 * h], seed).as_dtype(Dtype::Bfloat16)?,
    );
    w.insert("hc_scale".into(), values(&[3], seed + 1));
    w.insert("hc_base".into(), values(&[24], seed + 2));
    Hc::load_on(
        crate::metal_context::execution_context(),
        &mut w,
        "hc",
        4,
        h as usize,
        1e-6,
        4,
        1e-5,
    )
}
fn tower_fixture() -> anyhow::Result<Tower> {
    let config = ModelConfig::parse(
        r#"{"hidden_size":4096,"num_hidden_layers":2,"vocab_size":32,"layer_types":["linear_attention","deepseek_sparse_attention"]}"#,
    )?;
    let emb = quant(32, 4096, 8, 64, 61);
    let layers = vec![
        Layer {
            attn: Attn::Kda(crate::bindings::glm53_recurrent::plan_tests::tower_fixture(
                4096,
            )?),
            mlp: Mlp::Dense(dense(4096, 67)?),
            hc_attn: hyper(4096, 71)?,
            hc_ffn: hyper(4096, 73)?,
            in_norm: values(&[4096], 79),
            post_norm: values(&[4096], 83),
        },
        Layer {
            attn: Attn::Dsa(crate::bindings::glm53_attention_ops::tests::tower_fixture()?),
            mlp: Mlp::Sparse(
                crate::bindings::glm53_affine_moe::address_tests::tower_fixture(
                    Dtype::Float32,
                    4096,
                    Some(dense(4096, 89)?),
                )?,
            ),
            hc_attn: hyper(4096, 97)?,
            hc_ffn: hyper(4096, 101)?,
            in_norm: values(&[4096], 103),
            post_norm: values(&[4096], 107),
        },
    ];
    Ok(Tower {
        context: crate::metal_context::execution_context(),
        runtime: crate::metal_language::runtime(),
        vision: None,
        config,
        context_capacity: 4096,
        capture: None,
        prefill_mode: false,
        prefill_final_chunk: true,
        embed: QuantizedEmbedding {
            weight: emb.weight,
            scales: emb.scales,
            biases: emb.biases,
            bits: 8,
            group_size: 64,
        },
        layers,
        final_norm: values(&[4096], 109),
        lm_head: quant(32, 4096, 8, 64, 113),
        mtp: None,
        _expert_residency_set: None,
        n_layers: 2,
        _resident_memory: None,
    })
}
fn caches_equal(a: &[LayerCache], b: &[LayerCache]) -> anyhow::Result<()> {
    for (a, b) in a.iter().zip(b) {
        match (a, b) {
            (LayerCache::Linear(a), LayerCache::Linear(b)) => {
                for (a, b) in [(&a.conv, &b.conv), (&a.ssm, &b.ssm)] {
                    assert_eq!(a.is_some(), b.is_some());
                    if let (Some(a), Some(b)) = (a, b) {
                        equal(a, b)?;
                    }
                }
            }
            (LayerCache::Full(a), LayerCache::Full(b)) => {
                assert_eq!(a.offset, b.offset);
                if a.offset > 0 {
                    equal(
                        &a.indexer_tape.compact_slice(0, a.offset).unwrap(),
                        &b.indexer_tape.compact_slice(0, b.offset).unwrap(),
                    )?;
                }
            }
            _ => panic!("cache kind differs"),
        }
    }
    Ok(())
}
#[test]
fn affine_decoder_and_ragged_plans_preserve_original_outputs_and_caches() -> anyhow::Result<()> {
    use crate::language::LanguageModel;
    let mut tower = tower_fixture()?;
    for native in [true, false] {
        let _q8 = crate::core::quant::Glm53Q8PrefillScope::new(native);
        let _dsa = crate::bindings::glm53_attention_mode::Glm53DsaNativeScope::new(native);
        for (b, t, prefill) in [
            (1, 1, false),
            (1, 3, true),
            (2, 1, false),
            (2, 3, true),
            (1, 9, true),
        ] {
            let mut ac = tower.new_caches();
            let mut bc = tower.new_caches();
            for n in [t, 1] {
                tower.prefill_mode = prefill && n == t;
                let ids = Array::from_slice(
                    &(0..b * n).map(|i| (i * 7 + 3) % 32).collect::<Vec<_>>(),
                    &[b, n],
                );
                let (a, am) = tower.forward_inner_until_oracle(&ids, &mut ac, None)?;
                let (z, zm) = tower.forward_inner_until(&ids, &mut bc, None)?;
                equal(&a, &z)?;
                equal(&am, &zm)?;
                caches_equal(&ac, &bc)?;
            }
        }
        tower.prefill_mode = false;
        for preserve in [false, true] {
            let ids = Array::from_slice(&[3i32, 5, 7], &[1, 3]);
            let mut ac = tower.new_caches();
            let mut bc = tower.new_caches();
            tower.capture = Some(Capture {
                layers: vec![0, 1],
                preserve_dispatch: preserve,
                ..Default::default()
            });
            INC_STEP.store(0, std::sync::atomic::Ordering::Relaxed);
            let (a, am) = tower.forward_inner_until_oracle(&ids, &mut ac, Some(0))?;
            let old = tower.capture.take().unwrap();
            tower.capture = Some(Capture {
                layers: vec![0, 1],
                preserve_dispatch: preserve,
                ..Default::default()
            });
            INC_STEP.store(0, std::sync::atomic::Ordering::Relaxed);
            let (b, bm) = tower.forward_inner_until(&ids, &mut bc, Some(0))?;
            let new = tower.capture.take().unwrap();
            equal(&a, &b)?;
            equal(&am, &bm)?;
            caches_equal(&ac, &bc)?;
            if preserve {
                let a = old.pending.lock().unwrap();
                let b = new.pending.lock().unwrap();
                assert!(!a.is_empty());
                assert_eq!(a.len(), b.len());
                for (k, v) in a.iter() {
                    equal(v, &b[k])?;
                }
            } else {
                let a = old.entries.lock().unwrap();
                let b = new.entries.lock().unwrap();
                assert!(!a.is_empty());
                assert_eq!(a.len(), b.len());
                for (k, v) in a.iter() {
                    assert_eq!(
                        v.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                        b[k].iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                        "{k}"
                    );
                }
            }
        }
        let mut ac = vec![tower.new_caches(), tower.new_caches()];
        let mut bc = vec![tower.new_caches(), tower.new_caches()];
        for row in 0..2 {
            let ids = Array::from_slice(&vec![3i32; row + 1], &[1, (row + 1) as i32]);
            tower.forward_inner_until_oracle(&ids, &mut ac[row], None)?;
            tower.forward_inner_until(&ids, &mut bc[row], None)?;
        }
        for _ in 0..2 {
            let ids = Array::from_slice(&[5i32, 7], &[2, 1]);
            let (a, am) = tower.forward_ragged_inner_oracle(&ids, &mut ac)?;
            let (b, bm) = tower.forward_ragged_inner(&ids, &mut bc)?;
            equal(&a, &b)?;
            equal(&am, &bm)?;
            for (a, b) in ac.iter().zip(&bc) {
                caches_equal(a, b)?;
            }
        }
    }
    Ok(())
}
#[test]
fn affine_mtp_plan_preserves_original_logits_and_private_tape() -> anyhow::Result<()> {
    let mut tower = tower_fixture()?;
    let block = MtpBlock {
        context: crate::metal_context::execution_context(),
        eh_proj: values(&[4096, 8192], 127).as_dtype(Dtype::Bfloat16)?,
        enorm: values(&[4096], 131),
        hnorm: values(&[4096], 137),
        attn_norm: values(&[4096], 139),
        ffn_norm: values(&[4096], 149),
        shared_head_norm: values(&[4096], 151),
        attn: crate::bindings::glm53_attention_ops::tests::tower_fixture()?,
        mlp: crate::bindings::glm53_affine_moe::address_tests::tower_fixture(
            Dtype::Float32,
            4096,
            Some(dense(4096, 157)?),
        )?,
    };
    // Release trunk-only weights before exercising the independent draft plan.
    tower.layers.clear();
    for native in [true, false] {
        let _q8 = crate::core::quant::Glm53Q8PrefillScope::new(native);
        let _dsa = crate::bindings::glm53_attention_mode::Glm53DsaNativeScope::new(native);
        let mut ac = IndexerTape::new(crate::metal_context::execution_context());
        let mut bc = IndexerTape::new(crate::metal_context::execution_context());
        for position in 0..4 {
            let target = values(&[1, 1, 16384], 163 + position);
            let a = block.forward_oracle(
                &tower.config,
                &tower.embed,
                &tower.lm_head,
                &mut ac,
                &target,
                7,
                position,
                0,
            )?;
            let b = block.forward(
                &tower.config,
                &tower.embed,
                &tower.lm_head,
                &mut bc,
                &target,
                7,
                position,
                0,
            )?;
            equal(&a, &b)?;
            assert_eq!(ac.compact_total(), bc.compact_total());
            equal(
                &ac.compact_slice(0, position + 1).unwrap(),
                &bc.compact_slice(0, position + 1).unwrap(),
            )?;
        }
    }
    Ok(())
}

use oracle::{MtpBlockOracle0, TowerOracle1, TowerOracle2};

use crate::core::quant::QuantizedLinearExecution;
