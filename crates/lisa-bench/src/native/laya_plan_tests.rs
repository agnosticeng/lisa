//! Synthetic two-layer reference, captured before replacing the execution path.
use crate::core::graph::PreparedGraph;
use crate::laya::LayaConfig;
use std::collections::BTreeMap;
fn config() -> LayaConfig {
    LayaConfig::from_values(&serde_json::json!({"model_type":"modernbert","hidden_size":8,"num_attention_heads":2,"num_hidden_layers":2,"intermediate_size":12,"vocab_size":16,"local_attention":2}),&serde_json::json!({"head_layers":1,"act_costs":{"a":1}}),None).unwrap()
}
fn fixture() -> (LayaConfig, BTreeMap<String, (Vec<f32>, Vec<i32>)>) {
    let cfg = config();
    let mut shapes = BTreeMap::new();
    for plan in [
        lisa_models::laya::plan::encoder(&cfg, 5),
        lisa_models::laya::plan::scorer(&cfg, 5, 3),
        lisa_models::laya::plan::action(),
    ] {
        for name in plan.weights {
            let shape = if name.contains("tok_embeddings") {
                vec![16, 8]
            } else if name == "type_emb.weight" {
                vec![3, 8]
            } else if name.contains("norm") || name.starts_with("scorer.0") {
                vec![8]
            } else if name.contains("Wqkv") || name.contains("in_proj") {
                if name.ends_with("bias") {
                    vec![24]
                } else {
                    vec![24, 8]
                }
            } else if name.contains("mlp.Wi") {
                vec![24, 8]
            } else if name.contains("mlp.Wo") {
                vec![8, 12]
            } else if name.contains("linear1") {
                if name.ends_with("bias") {
                    vec![12]
                } else {
                    vec![12, 8]
                }
            } else if name.contains("linear2") {
                if name.ends_with("bias") {
                    vec![8]
                } else {
                    vec![8, 12]
                }
            } else if name.starts_with("scorer.1") {
                if name.ends_with("bias") {
                    vec![5]
                } else {
                    vec![5, 8]
                }
            } else if name.starts_with("scorer.3") {
                if name.ends_with("bias") {
                    vec![1]
                } else {
                    vec![1, 5]
                }
            } else if name.starts_with("act_head.0") {
                if name.ends_with("bias") {
                    vec![6]
                } else {
                    vec![6, 12]
                }
            } else if name.starts_with("act_head.2") {
                if name.ends_with("bias") {
                    vec![2]
                } else {
                    vec![2, 6]
                }
            } else if name.ends_with("bias") {
                vec![8]
            } else {
                vec![8, 8]
            };
            shapes.insert(name, shape);
        }
    }
    let mut seed = 319u64;
    let weights = shapes
        .into_iter()
        .map(|(name, shape)| {
            let values = (0..shape.iter().product::<i32>())
                .map(|_| {
                    seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                    let v = ((seed >> 40) as f32 / 8_388_608.0 - 1.0) * 0.25;
                    if name.contains("norm") && name.ends_with("weight") {
                        v + 1.0
                    } else {
                        v
                    }
                })
                .collect();
            (name, (values, shape))
        })
        .collect();
    (cfg, weights)
}

use lisa_metal::{Array, device::MetalDevice};
#[test]
#[cfg(feature = "metal")]
fn complete_metal_plans_preserve_original_f32_bits() {
    let (cfg, weights) = fixture();
    let prepare = |plan: lisa_models::laya::plan::Plan| {
        let arrays = plan
            .weights
            .iter()
            .map(|n| {
                let (v, s) = &weights[n];
                Array::from_slice(v, s)
            })
            .collect();
        PreparedGraph::new(MetalDevice::system_default(), plan.graph, arrays).unwrap()
    };
    // Captured from the pre-plan Metal forward with the deterministic synthetic
    // weights above: full/local encoder attention, decision head and act head.
    for (ids, markers, expected_logits, expected_action) in [
        (
            vec![1i32, 4, 2, 3, 8],
            vec![0i32, 2, 4],
            vec![1031150940, 1033714386, 1031998937],
            vec![1012870032, 3182291450],
        ),
        (
            vec![1],
            vec![0],
            vec![1030921898],
            vec![3145917152, 3183115701],
        ),
    ] {
        let encoder = prepare(lisa_models::laya::plan::encoder(&cfg, ids.len() as i32));
        let scorer = prepare(lisa_models::laya::plan::scorer(
            &cfg,
            ids.len() as i32,
            markers.len() as i32,
        ));
        let action = prepare(lisa_models::laya::plan::action());
        for _ in 0..2 {
            let ids = Array::from_slice(&ids, &[ids.len() as i32]);
            let hidden = encoder.run(&[&ids], &mut []).unwrap().into_one().unwrap();
            let qtype = Array::from_slice(&[1i32], &[1]);
            let marker_ids = Array::from_slice(&markers, &[markers.len() as i32]);
            let out = scorer
                .run(&[&hidden, &qtype, &marker_ids], &mut [])
                .unwrap();
            let logits = out.get(0).unwrap().as_slice::<f32>();
            assert_eq!(
                logits.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                expected_logits
            );
            let probabilities = out.get(1).unwrap().as_slice::<f32>();
            let features = lisa_models::laya::plan::action_features(probabilities).unwrap();
            let features = Array::from_slice(&features, &[1, 4]);
            let a = action
                .run(&[out.get(2).unwrap(), &features], &mut [])
                .unwrap()
                .into_one()
                .unwrap();
            assert_eq!(
                a.as_slice::<f32>()
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>(),
                expected_action
            );
        }
    }
}

#[test]
fn application_loader_runs_generic_laya_binding_on_metal() -> anyhow::Result<()> {
    let (_, weights) = fixture();
    let tmp = tempfile::tempdir()?;
    let root = tmp.path();
    std::fs::create_dir(root.join("encoder"))?;
    std::fs::create_dir(root.join("tokenizer"))?;
    std::fs::write(
        root.join("encoder/config.json"),
        serde_json::to_vec(&serde_json::json!({
            "model_type":"modernbert","hidden_size":8,"num_attention_heads":2,
            "num_hidden_layers":2,"intermediate_size":12,"vocab_size":16,"local_attention":2
        }))?,
    )?;
    std::fs::write(
        root.join("rl_agent_config.json"),
        br#"{"head_layers":1,"act_costs":{"a":1}}"#,
    )?;
    let vocab =
        std::collections::HashMap::from([("[UNK]".to_owned(), 0), ("[MASK]".to_owned(), 1)]);
    let model = tokenizers::models::wordlevel::WordLevel::builder()
        .vocab(vocab.into_iter().collect())
        .unk_token("[UNK]".into())
        .build()
        .map_err(anyhow::Error::msg)?;
    tokenizers::Tokenizer::new(model)
        .save(root.join("tokenizer/tokenizer.json"), false)
        .map_err(anyhow::Error::msg)?;
    let mut header = serde_json::Map::new();
    let mut payload = Vec::new();
    for (name, (values, shape)) in weights {
        let start = payload.len();
        for value in values {
            payload.extend(value.to_le_bytes());
        }
        header.insert(
            name,
            serde_json::json!({"dtype":"F32", "shape":shape, "data_offsets":[start,payload.len()]}),
        );
    }
    let mut header = serde_json::to_vec(&header)?;
    header.resize(header.len().div_ceil(8) * 8, b' ');
    let mut file = (header.len() as u64).to_le_bytes().to_vec();
    file.extend(header);
    file.extend(payload);
    std::fs::write(root.join("model.safetensors"), file)?;
    let crate::loading::Loaded::Decision(model) =
        crate::loading::load_dir_with(root, crate::device::Backend::Metal)?
    else {
        anyhow::bail!("expected decision model");
    };
    for _ in 0..2 {
        let (logits, action) = model.decide(&[1, 4, 2, 3, 8], 1, &[0, 2, 4])?;
        assert_eq!(
            logits.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            [1031150940, 1033714386, 1031998937]
        );
        assert_eq!(
            action.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            [1012870032, 3182291450]
        );
    }
    Ok(())
}
