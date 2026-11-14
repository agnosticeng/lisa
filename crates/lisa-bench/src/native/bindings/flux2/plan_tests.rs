#![cfg(test)]
//! Strict migration checks against the previous operation chain.
use super::{
    plan_oracle as oracle,
    weights::{Weights, info},
};
use lisa_metal::{Array, Dtype, ops};
use lisa_models::flux::{dit2, dit3, text2};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, HashMap};
fn shapes2() -> BTreeMap<String, Vec<i32>> {
    let mut s = BTreeMap::new();
    for (key, n, k) in [
        ("x_embedder", 128, 128),
        ("context_embedder", 128, 12288),
        ("time_guidance_embed.linear_1", 128, 256),
        ("time_guidance_embed.linear_2", 128, 128),
        ("double_stream_modulation_img.linear", 768, 128),
        ("double_stream_modulation_txt.linear", 768, 128),
        ("single_stream_modulation.linear", 384, 128),
        ("norm_out.linear", 256, 128),
        ("proj_out", 128, 128),
    ] {
        s.insert(format!("{key}.weight"), vec![n, k]);
        s.insert(format!("{key}.bias"), vec![n]);
    }
    let p = "transformer_blocks.0";
    for name in [
        "to_q",
        "to_k",
        "to_v",
        "to_out",
        "add_q_proj",
        "add_k_proj",
        "add_v_proj",
        "to_add_out",
    ] {
        s.insert(format!("{p}.attn.{name}.weight"), vec![128, 128]);
    }
    for name in ["norm_q", "norm_k", "norm_added_q", "norm_added_k"] {
        s.insert(format!("{p}.attn.{name}.weight"), vec![128]);
    }
    for ff in ["ff", "ff_context"] {
        s.insert(format!("{p}.{ff}.linear_in.weight"), vec![384, 128]);
        s.insert(format!("{p}.{ff}.linear_out.weight"), vec![128, 192]);
    }
    let p = "single_transformer_blocks.0.attn";
    s.insert(format!("{p}.to_qkv_mlp_proj.weight"), vec![768, 128]);
    s.insert(format!("{p}.to_out.weight"), vec![128, 320]);
    for name in ["norm_q", "norm_k"] {
        s.insert(format!("{p}.{name}.weight"), vec![128]);
    }
    s
}
fn shapes3() -> BTreeMap<String, Vec<i32>> {
    let mut s = BTreeMap::new();
    for (key, n, k) in [
        ("emb_in.image", 128, 128),
        ("txt_in", 128, 20480),
        ("time_in.in_layer", 128, 256),
        ("time_in.out_layer", 128, 128),
        ("ref_time_in.in_layer", 128, 256),
        ("ref_time_in.out_layer", 128, 128),
        ("final_layer.image.adaLN_modulation.1", 256, 128),
        ("final_layer.image.linear", 128, 128),
    ] {
        s.insert(format!("{key}.weight"), vec![n, k]);
        s.insert(format!("{key}.bias"), vec![n]);
    }
    for stream in [
        "early_stream_modulations.image",
        "early_stream_modulations.txt",
        "single_stream_modulations.image",
        "single_stream_modulations.txt",
    ] {
        s.insert(format!("{stream}.lin.weight"), vec![384, 128]);
    }
    for p in [
        "content_mode_blocks.image.0",
        "txt_mode_blocks.0",
        "single_blocks.0",
    ] {
        for key in ["q_proj", "k_proj", "v_proj", "attn_out"] {
            s.insert(format!("{p}.{key}.weight"), vec![128, 128]);
        }
        for key in ["query_norm", "key_norm"] {
            s.insert(format!("{p}.norm.{key}.scale"), vec![128]);
        }
        s.insert(format!("{p}.mlp_in.weight"), vec![384, 128]);
        s.insert(format!("{p}.mlp_out.weight"), vec![128, 192]);
    }
    s
}
fn shapes_text2() -> BTreeMap<String, Vec<i32>> {
    let mut s = BTreeMap::from([("embed_tokens.weight".to_owned(), vec![64, 128])]);
    for i in 0..27 {
        let p = format!("layers.{i}");
        for key in [
            "input_layernorm",
            "post_attention_layernorm",
            "self_attn.q_norm",
            "self_attn.k_norm",
        ] {
            s.insert(format!("{p}.{key}.weight"), vec![128]);
        }
        for (key, n, k) in [
            ("self_attn.q_proj", 256, 128),
            ("self_attn.k_proj", 128, 128),
            ("self_attn.v_proj", 128, 128),
            ("self_attn.o_proj", 128, 256),
            ("mlp.gate_proj", 192, 128),
            ("mlp.up_proj", 192, 128),
            ("mlp.down_proj", 128, 192),
        ] {
            s.insert(format!("{p}.{key}.weight"), vec![n, k]);
        }
    }
    s
}
fn weights(
    shapes: BTreeMap<String, Vec<i32>>,
    quantized: bool,
) -> anyhow::Result<HashMap<String, Array>> {
    let mut seed = 7919u64;
    let mut out = HashMap::new();
    for (key, shape) in shapes {
        let norm = shape.len() == 1 && (key.contains("norm") || key.ends_with(".scale"));
        let data = (0..shape.iter().product::<i32>())
            .map(|_| {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                ((seed >> 40) as f32 / 8_388_608. - 1.) * 0.025 + if norm { 1. } else { 0. }
            })
            .collect::<Vec<_>>();
        let w = Array::from_slice(&data, &shape).as_dtype(Dtype::Bfloat16)?;
        if quantized && shape.len() == 2 {
            let (w, s, b) = ops::quantize(&w, 64, 4)?;
            let prefix = key.strip_suffix(".weight").unwrap();
            out.insert(format!("{prefix}.scales"), s);
            out.insert(format!("{prefix}.biases"), b);
            out.insert(key, w);
        } else {
            out.insert(key, w);
        }
    }
    Ok(out)
}
fn input(shape: &[i32], dtype: Dtype) -> anyhow::Result<Array> {
    Ok(Array::from_slice(
        &(0..shape.iter().product::<i32>())
            .map(|i| ((i * 7 % 43) as f32 - 21.) / 32.)
            .collect::<Vec<_>>(),
        shape,
    )
    .as_dtype(dtype)?)
}
fn bits(x: &Array) -> anyhow::Result<Vec<u32>> {
    let x = x.as_dtype(Dtype::Float32)?.contiguous()?;
    x.eval()?;
    Ok(x.as_slice::<f32>().iter().map(|x| x.to_bits()).collect())
}
fn shapes_text3() -> BTreeMap<String, Vec<i32>> {
    let mut s = shapes_text2();
    let layer = s
        .iter()
        .filter(|(k, _)| k.starts_with("layers.0."))
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect::<Vec<_>>();
    for i in 27..36 {
        for (k, v) in &layer {
            s.insert(
                k.replacen("layers.0.", &format!("layers.{i}."), 1),
                v.clone(),
            );
        }
    }
    s.insert("norm.weight".into(), vec![128]);
    s
}
// These hashes were independently agreed by the original chains under both
// per-dispatch submission and Metal shader validation. After fixing the
// small-row reduction underwrite, normal execution also compares both chains.
fn verify(label: &str, actual: &Array, reference: Option<&Array>) -> anyhow::Result<()> {
    verify_bits(
        label,
        &bits(actual)?,
        reference.map(bits).transpose()?.as_deref(),
    )
}
fn verify_bits(label: &str, actual: &[u32], reference: Option<&[u32]>) -> anyhow::Result<()> {
    let expected: std::collections::BTreeMap<String, String> =
        serde_json::from_str(include_str!("plan_expected.json"))?;
    let digest = |v: &[u32]| {
        format!(
            "{:x}",
            Sha256::digest(v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<_>>())
        )
    };
    assert_eq!(digest(actual), expected[label], "{label}");
    if let Some(reference) = reference {
        assert_eq!(actual, reference, "{label}");
    }
    Ok(())
}
fn dit2_check(reference: bool) -> anyhow::Result<()> {
    for quantized in [false, true] {
        let map = weights(shapes2(), quantized)?;
        let oracle = oracle::dit2::Dit::fixture(oracle::weights::Weights::from_map(map.clone()));
        let weights = Weights::from_map(crate::metal_context::execution_context(), map);
        let schema = weights.schema()?;
        let latent = input(&[1, 6, 128], Dtype::Bfloat16)?;
        let enc = input(&[1, 3, 12288], Dtype::Bfloat16)?;
        for time in [500., 250.] {
            let expected = reference
                .then(|| oracle.forward(&latent, &enc, time, 2, 3))
                .transpose()?;
            for _ in 0..2 {
                let p = dit2::program(
                    &schema,
                    dit2::Geometry {
                        inner: 128,
                        heads: 1,
                        hd: 128,
                        doubles: 1,
                        singles: 1,
                    },
                    info(&latent)?,
                    info(&enc)?,
                    time,
                    2,
                    3,
                )?;
                let out = weights
                    .prepare(p)?
                    .run(&[&latent, &enc], &mut [])?
                    .into_one()
                    .map_err(anyhow::Error::msg)?;
                verify(
                    &format!("dit2-q{quantized}-t{time}"),
                    &out,
                    expected.as_ref(),
                )?;
            }
        }
    }
    Ok(())
}
fn dit3_check(reference: bool) -> anyhow::Result<()> {
    let map = weights(shapes3(), false)?;
    let oracle = oracle::dit3::Dit::fixture(oracle::weights::Weights::from_map(map.clone()));
    let weights = Weights::from_map(crate::metal_context::execution_context(), map);
    let schema = weights.schema()?;
    let latent = input(&[1, 6, 128], Dtype::Float32)?;
    let enc = input(&[1, 3, 20480], Dtype::Bfloat16)?;
    for (time, target) in [(0.5, 0.25), (0.25, 0.)] {
        let expected = reference
            .then(|| oracle.forward(&latent, &enc, time, target, 2, 3))
            .transpose()?;
        for _ in 0..2 {
            let p = dit3::program(
                &schema,
                dit3::Geometry {
                    hidden: 128,
                    heads: 1,
                    head_dim: 128,
                    early: 1,
                    joint: 1,
                },
                info(&latent)?,
                info(&enc)?,
                time,
                target,
                2,
                3,
            )?;
            let out = weights
                .prepare(p)?
                .run(&[&latent, &enc], &mut [])?
                .into_one()
                .map_err(anyhow::Error::msg)?;
            verify(&format!("dit3-t{time}-{target}"), &out, expected.as_ref())?;
        }
    }
    Ok(())
}
fn text2_check(reference: bool) -> anyhow::Result<()> {
    let map = weights(shapes_text2(), true)?;
    let oracle = oracle::weights::Weights::from_map(map.clone());
    let weights = Weights::from_map(crate::metal_context::execution_context(), map);
    let schema = weights.schema()?;
    let ids = (0..512).map(|i| (i % 64) as u32).collect::<Vec<_>>();
    let input = Array::from_slice(&ids, &[512]);
    for real in [7, 512] {
        let expected = reference
            .then(|| oracle::text2::forward(&oracle, &ids, real))
            .transpose()?;
        for _ in 0..2 {
            let p = text2::program(
                &schema,
                text2::Geometry {
                    hidden: 128,
                    hd: 128,
                    heads: 2,
                    kv: 1,
                    depth: 27,
                },
                real,
            )?;
            let out = weights
                .prepare(p)?
                .run(&[&input], &mut [])?
                .into_one()
                .map_err(anyhow::Error::msg)?;
            verify(&format!("text2-real{real}"), &out, expected.as_ref())?;
        }
    }
    Ok(())
}
fn text3_check(reference: bool) -> anyhow::Result<()> {
    use lisa_models::flux::text3;
    let map = weights(shapes_text3(), false)?;
    let oracle = oracle::text3::Encoder {
        weights: oracle::weights::Weights::from_map(map.clone()),
    };
    let weights = Weights::from_map(crate::metal_context::execution_context(), map);
    let schema = weights.schema()?;
    let geometry = text3::Geometry {
        hidden: 128,
        heads: 2,
        kv_heads: 1,
        depth: 36,
    };
    let execute = |ids: &[u32],
                   real: usize,
                   capture: bool,
                   cache: Option<&mut oracle::text3::Cache>|
     -> anyhow::Result<Array> {
        let cache_schema = cache
            .as_ref()
            .map(|c| {
                c.iter()
                    .map(|v| {
                        v.as_ref()
                            .map(|(k, v)| Ok((info(k)?, info(v)?)))
                            .transpose()
                    })
                    .collect::<anyhow::Result<Vec<_>>>()
            })
            .transpose()?;
        let p = text3::program(
            &schema,
            geometry,
            ids.len(),
            real,
            capture,
            cache_schema.as_ref(),
        )?;
        let ids = Array::from_slice(ids, &[ids.len() as i32]);
        let mut inputs = vec![&ids];
        if let Some(c) = &cache {
            for (k, v) in c.iter().flatten() {
                inputs.extend([k, v]);
            }
        }
        let mut out = weights
            .prepare(p.program)?
            .run(&inputs, &mut [])?
            .into_iter();
        let y = out.next().unwrap();
        if let Some(c) = cache {
            for i in p.cache_layers {
                c[i] = Some((out.next().unwrap(), out.next().unwrap()));
            }
        }
        Ok(y)
    };
    for _ in 0..2 {
        let ids = [2, 7, 11, 3, 0, 0, 0, 0];
        let expected = reference
            .then(|| oracle.forward(&ids, 4, true, None))
            .transpose()?;
        verify(
            "text3-capture",
            &execute(&ids, 4, true, None)?,
            expected.as_ref(),
        )?;
        let mut rc = (0..36).map(|_| None).collect::<Vec<_>>();
        let mut ac = rc.clone();
        for (step, ids) in [vec![2, 7, 11], vec![3], vec![4, 5]].iter().enumerate() {
            let expected = reference
                .then(|| oracle.forward(ids, ids.len(), false, Some(&mut rc)))
                .transpose()?;
            verify(
                &format!("text3-logits-{step}"),
                &execute(ids, ids.len(), false, Some(&mut ac))?,
                expected.as_ref(),
            )?;
            let mut all = vec![];
            let mut original = vec![];
            for (a, b) in ac.iter().zip(&rc) {
                let (ak, av) = a.as_ref().unwrap();
                all.extend(bits(ak)?);
                all.extend(bits(av)?);
                if let Some((bk, bv)) = b {
                    assert_eq!(ak.shape(), bk.shape());
                    assert_eq!(av.shape(), bv.shape());
                    original.extend(bits(bk)?);
                    original.extend(bits(bv)?);
                }
            }
            verify_bits(
                &format!("text3-cache-{step}"),
                &all,
                reference.then_some(original.as_slice()),
            )?;
        }
    }
    Ok(())
}
#[test]
fn flux2_denoiser_plan_matches_validated_bits() -> anyhow::Result<()> {
    dit2_check(true)
}
#[test]
fn flux3_denoiser_plan_matches_validated_bits() -> anyhow::Result<()> {
    dit3_check(true)
}
#[test]
fn flux2_text_plan_preserves_original_bits() -> anyhow::Result<()> {
    text2_check(true)
}
#[test]
fn flux3_text_plan_preserves_original_bits_and_cache() -> anyhow::Result<()> {
    text3_check(true)
}
#[test]
fn latent_and_pixel_plans_preserve_family_arithmetic() -> anyhow::Result<()> {
    use super::weights::run;
    use lisa_models::flux::pipeline::{self as model, Family};
    for family in [Family::Klein2, Family::Klein3] {
        let data = model::normal_noise(6 * 128, 42);
        let noise = Array::from_slice(&data, &model::noise_shape(family, 2, 3));
        let rounded = noise.as_dtype(Dtype::Bfloat16)?;
        let expected = match family {
            Family::Klein2 => rounded.transpose_axes(&[0, 2, 1])?.contiguous()?,
            Family::Klein3 => rounded.as_dtype(Dtype::Float32)?,
        };
        let mut actual = run(model::initialize(family, 2, 3)?, &[&noise])?;
        assert_eq!(bits(&actual)?, bits(&expected)?);
        let mut expected = expected;
        let velocity = input(&[1, 6, 128], Dtype::Bfloat16)?;
        for dt in [-0.17, -0.33333334] {
            let mut delta = velocity.multiply(&Array::from_f32(dt))?;
            if family == Family::Klein3 {
                delta = delta.as_dtype(Dtype::Bfloat16)?.as_dtype(Dtype::Float32)?;
            }
            expected = expected.add(&delta)?;
            actual = run(model::step(family, dt), &[&actual, &velocity])?;
            assert_eq!(bits(&actual)?, bits(&expected)?);
        }
        let decoded = run(model::decoder_input(2, 3), &[&actual])?;
        let expected = expected
            .reshape(&[1, 2, 3, 128])?
            .transpose_axes(&[0, 3, 1, 2])?;
        assert_eq!(decoded.shape(), expected.shape());
        assert_eq!(bits(&decoded)?, bits(&expected)?);
    }
    let pixels = [-2., -1., -0.5, 0., 0.5, 1., 2.];
    assert_eq!(
        model::rgb(Family::Klein2, &pixels)?,
        [0, 0, 64, 128, 191, 255, 255]
    );
    assert_eq!(
        model::rgb(Family::Klein3, &pixels)?,
        [0, 0, 63, 127, 191, 255, 255]
    );
    assert!(model::rgb(Family::Klein2, &[f32::NAN]).is_err());
    Ok(())
}
