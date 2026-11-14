#![cfg(test)]
use super::*;
use crate::bindings::deepseek41_moe::tests::{counts, difference};
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
fn fixture(metadata: Dtype) -> anyhow::Result<MoE> {
    let linear = |n: usize, k: usize, seed: usize| -> anyhow::Result<QuantizedLinear> {
        let w = (0..n * k / 16)
            .map(|i| {
                (i as u32)
                    .wrapping_mul(1729)
                    .wrapping_add(seed as u32 * 7919)
            })
            .collect::<Vec<_>>();
        let scales = (0..n * k / 128)
            .map(|i| 0.003 + ((i + seed) % 7) as f32 * 0.0002)
            .collect::<Vec<_>>();
        let biases = (0..n * k / 128)
            .map(|i| -0.003 + ((i + seed) % 5) as f32 * 0.0001)
            .collect::<Vec<_>>();
        Ok(QuantizedLinear {
            weight: Array::from_slice(&w, &[n as i32, (k / 16) as i32]),
            scales: Array::from_slice(&scales, &[n as i32, (k / 128) as i32]).as_dtype(metadata)?,
            biases: Array::from_slice(&biases, &[n as i32, (k / 128) as i32]).as_dtype(metadata)?,
            group_size: 128,
            bits: 2,
        })
    };
    let experts = (0..12)
        .map(|i| {
            Ok(ExpertParameters {
                gate: linear(512, 512, i * 3)?,
                up: linear(512, 512, i * 3 + 1)?,
                down: linear(512, 512, i * 3 + 2)?,
            })
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    let table = ExpertPools::address_table(&crate::metal_context::device(), &experts)?.0;
    Ok(MoE {
        context: crate::metal_context::execution_context(),
        gate_w: Array::from_slice(&vec![0f32; 12 * 512], &[12, 512]),
        bias: Array::from_slice(&[0f32; 12], &[12]),
        pools: Arc::new(ExpertPools {
            device: crate::metal_context::device(),
            observer: crate::metal_context::execution_context().observer(),
            worker: Arc::new(Default::default()),
            expert_cache_bytes: experts
                .iter()
                .map(|e| e.byte_size(&crate::metal_context::device()))
                .collect(),
            experts,
            lazy_store: None,
            expert_prefix: None,
            inter: 512,
            address_table: table,
            metadata_dtype: lisa_backend::ScalarType::try_from(metadata)?,
            residency_set: Default::default(),
            gpu_address_all_experts: false,
            resources: Default::default(),
            cache_capacity_bytes: AtomicUsize::new(0),
            cache: Default::default(),
        }),
        shared: None,
        grouped_workspace: Some(Arc::new(GroupedMoeWorkspace::default())),
        top_k: 8,
        n_group: 1,
        topk_group: 1,
        routed_scaling: 2.5,
        norm_topk: true,
        limit: 10.,
        n_experts: 12,
    })
}
#[test]
fn routed_expert_plans_match_original_bits_dispatch_and_reused_workspace() -> anyhow::Result<()> {
    for resident in [false, true] {
        for metadata in [Dtype::Float32, Dtype::Bfloat16] {
            let mut model = fixture(metadata)?;
            if resident {
                Arc::get_mut(&mut model.pools)
                    .unwrap()
                    .gpu_address_all_experts = true;
                let set = crate::metal_context::device()
                    .new_residency(12 * 9)
                    .expect("residency support");
                assert!(model.add_expert_resources_to_residency_set(&set));
                set.commit();
                model.set_model_residency_set(set);
            }
            for (split, vector, tile) in [
                (true, false, 64),
                (false, false, 64),
                (true, true, 64),
                (false, true, 64),
                (true, false, 128),
                (false, false, 128),
                (true, true, 128),
                (false, true, 128),
            ] {
                let _scope = crate::core::quant::Glm53GroupedMoePairScope::new(split);
                let _vector = crate::core::quant::Glm53GroupedMoeQ2LoadScope::new(vector);
                let _tile = crate::core::quant::Glm53GroupedMoeDownTileScope::new(tile);
                for (batch, time) in [(1, 1), (1, 7), (2, 3), (1, 95), (1, 96), (1, 128), (1, 96)] {
                    for mode in [0, 1, 2] {
                        let n = batch * time;
                        let x = Array::from_slice(
                            &(0..n * 512)
                                .map(|i| ((i * 17 + time) % 101) as f32 * 0.002 - 0.08)
                                .collect::<Vec<_>>(),
                            &[batch as i32, time as i32, 512],
                        );
                        let ids = Array::from_slice(
                            &(0..n)
                                .flat_map(|row| (0..8).map(move |slot| ((row + slot) % 12) as i32))
                                .collect::<Vec<_>>(),
                            &[n as i32, 8],
                        );
                        let weights = Array::from_slice(
                            &(0..n * 8)
                                .map(|i| 0.07 + (i % 8) as f32 * 0.01)
                                .collect::<Vec<_>>(),
                            &[n as i32, 8],
                        );
                        let shared = Array::from_slice(
                            &(0..n * 512)
                                .map(|i| ((i * 23) % 131) as f32 * 0.0001)
                                .collect::<Vec<_>>(),
                            &[n as i32, 512],
                        );
                        let a = Capture {
                            layers: vec![3],
                            preserve_dispatch: mode == 2,
                            ..Default::default()
                        };
                        let b = Capture {
                            layers: vec![3],
                            preserve_dispatch: mode == 2,
                            ..Default::default()
                        };
                        let prepare = || {
                            Some((
                                model.pools.address_table.clone(),
                                model
                                    .pools
                                    .experts
                                    .iter()
                                    .flat_map(ExpertPools::arrays)
                                    .cloned()
                                    .collect(),
                            ))
                        };
                        let before = counts();
                        let old = model.forward_address_gpu_oracle(
                            &x,
                            &ids,
                            &weights,
                            batch,
                            time,
                            512,
                            Some(&shared),
                            prepare(),
                            (mode > 0).then_some(&a),
                            3,
                        )?;
                        let middle = counts();
                        let new = model.forward_address_gpu(
                            &x,
                            &ids,
                            &weights,
                            batch,
                            time,
                            512,
                            Some(&shared),
                            prepare(),
                            (mode > 0).then_some(&b),
                            3,
                        )?;
                        let after = counts();
                        equal(&old, &new)?;
                        if std::env::var_os("LISA_TRACE").is_some() {
                            let old = difference(&before, &middle);
                            assert!(!old.is_empty());
                            assert_eq!(
                                old,
                                difference(&middle, &after),
                                "metadata{metadata:?} split{split} B{batch} T{time} capture{mode}"
                            );
                        }
                        if mode == 1 {
                            let a = a.entries.lock().unwrap();
                            let b = b.entries.lock().unwrap();
                            assert_eq!(a.len(), b.len());
                            for (k, v) in a.iter() {
                                assert_eq!(
                                    v.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                                    b[k].iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                                    "{k}"
                                );
                            }
                        } else if mode == 2 {
                            let a = a.pending.lock().unwrap();
                            let b = b.pending.lock().unwrap();
                            assert_eq!(a.len(), b.len());
                            for (k, v) in a.iter() {
                                equal(v, &b[k])?;
                            }
                        }
                    }
                }
            }
        }
    }
    Ok(())
}
#[test]
fn router_and_grouped_fallback_plans_match_original() -> anyhow::Result<()> {
    for metadata in [Dtype::Float32, Dtype::Bfloat16] {
        let mut model = fixture(metadata)?;
        model.gate_w = Array::from_slice(
            &(0..12 * 512)
                .map(|i| ((i * 31) % 101) as f32 * 0.0003 - 0.01)
                .collect::<Vec<_>>(),
            &[12, 512],
        );
        model.bias = Array::from_slice(
            &(0..12)
                .map(|i| ((i * 7) % 13) as f32 * 0.003)
                .collect::<Vec<_>>(),
            &[12],
        );
        for groups in [1, 3] {
            model.n_group = groups;
            model.topk_group = if groups == 1 { 1 } else { 2 };
            for (b, t) in [(1, 1), (1, 7), (2, 3), (1, 16)] {
                let x = Array::from_slice(
                    &(0..b * t * 512)
                        .map(|i| ((i * 17 + t) % 101) as f32 * 0.002 - 0.08)
                        .collect::<Vec<_>>(),
                    &[b as i32, t as i32, 512],
                );
                let before = counts();
                let (a, _) = model.forward_parts_oracle(&x, None, 3, false)?;
                let middle = counts();
                let (b, _) = model.forward_parts_with_capture_mode(&x, None, 3, false)?;
                let after = counts();
                equal(&a, &b)?;
                if std::env::var_os("LISA_TRACE").is_some() {
                    assert_eq!(
                        difference(&before, &middle),
                        difference(&middle, &after),
                        "metadata{metadata:?} groups{groups} T{t}"
                    );
                }
            }
        }
    }
    Ok(())
}

pub(crate) fn tower_fixture(
    metadata: Dtype,
    width: i32,
    shared: Option<DenseMlp>,
) -> anyhow::Result<MoE> {
    let linear = |n: usize, k: usize, seed: usize| -> anyhow::Result<QuantizedLinear> {
        let w = (0..n * k / 16)
            .map(|i| {
                (i as u32)
                    .wrapping_mul(1729)
                    .wrapping_add(seed as u32 * 7919)
            })
            .collect::<Vec<_>>();
        let scales = (0..n * k / 128)
            .map(|i| 0.003 + ((i + seed) % 7) as f32 * 0.0002)
            .collect::<Vec<_>>();
        let biases = (0..n * k / 128)
            .map(|i| -0.003 + ((i + seed) % 5) as f32 * 0.0001)
            .collect::<Vec<_>>();
        Ok(QuantizedLinear {
            weight: Array::from_slice(&w, &[n as i32, (k / 16) as i32]),
            scales: Array::from_slice(&scales, &[n as i32, (k / 128) as i32]).as_dtype(metadata)?,
            biases: Array::from_slice(&biases, &[n as i32, (k / 128) as i32]).as_dtype(metadata)?,
            group_size: 128,
            bits: 2,
        })
    };
    let experts = (0..12)
        .map(|i| {
            Ok(ExpertParameters {
                gate: linear(512, width as usize, i * 3)?,
                up: linear(512, width as usize, i * 3 + 1)?,
                down: linear(width as usize, 512, i * 3 + 2)?,
            })
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    let table = ExpertPools::address_table(&crate::metal_context::device(), &experts)?.0;
    Ok(MoE {
        context: crate::metal_context::execution_context(),
        gate_w: Array::from_slice(&vec![0f32; 12 * width as usize], &[12, width]),
        bias: Array::from_slice(&[0f32; 12], &[12]),
        pools: Arc::new(ExpertPools {
            device: crate::metal_context::device(),
            observer: crate::metal_context::execution_context().observer(),
            worker: Arc::new(Default::default()),
            expert_cache_bytes: experts
                .iter()
                .map(|e| e.byte_size(&crate::metal_context::device()))
                .collect(),
            experts,
            lazy_store: None,
            expert_prefix: None,
            inter: 512,
            address_table: table,
            metadata_dtype: lisa_backend::ScalarType::try_from(metadata)?,
            residency_set: Default::default(),
            gpu_address_all_experts: false,
            resources: Default::default(),
            cache_capacity_bytes: AtomicUsize::new(0),
            cache: Default::default(),
        }),
        shared,
        grouped_workspace: Some(Arc::new(GroupedMoeWorkspace::default())),
        top_k: 8,
        n_group: 1,
        topk_group: 1,
        routed_scaling: 2.5,
        norm_topk: true,
        limit: 10.,
        n_experts: 12,
    })
}

use super::address_oracle::{MoEOracle0, MoEOracle1};
