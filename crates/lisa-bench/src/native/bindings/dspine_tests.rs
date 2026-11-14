#![cfg(test)]
use super::*;
use crate::core::norm::RmsNormExecution;
use lisa_metal::ops::indexing::{IndexOp, take_axis};
use lisa_metal::{fast, ops};
use lisa_models::qwen3_5::dspine::layers::window_mask;

#[test]
fn sliding_window_masks_old_context_and_future_proposals() {
    for offset in [8189usize, 8191, 8192, 16384] {
        let rows = 7;
        let start = offset.saturating_sub(8191);
        let keys = offset + rows - start;
        let mask = window_mask(offset, rows, start, keys, 8192);
        for row in 0..rows {
            let live: Vec<_> = mask[row * keys..(row + 1) * keys]
                .iter()
                .enumerate()
                .filter(|(_, v)| **v == 0.0)
                .map(|(key, _)| key + start)
                .collect();
            assert_eq!(live.len(), (offset + row + 1).min(8192));
            assert_eq!(live.last(), Some(&(offset + row)));
            assert_eq!(live.first(), Some(&(offset + row + 1).saturating_sub(8192)));
        }
    }
}

#[test]
fn sliding_attention_excludes_evicted_values_on_gpu() -> anyhow::Result<()> {
    let zero = || -> anyhow::Result<DenseLinear> {
        Ok(DenseLinear {
            weight: ops::zeros::<f32>(&[128, 128])?.as_dtype(Dtype::Bfloat16)?,
            bias: None,
            quantized: None,
        })
    };
    let identity = || -> anyhow::Result<DenseLinear> {
        let v: Vec<f32> = (0..128 * 128)
            .map(|i| if i / 128 == i % 128 { 1.0 } else { 0.0 })
            .collect();
        Ok(DenseLinear {
            weight: Array::from_slice(&v, &[128, 128]).as_dtype(Dtype::Bfloat16)?,
            bias: None,
            quantized: None,
        })
    };
    let norm = || -> anyhow::Result<RmsNorm> {
        Ok(RmsNorm {
            weight: Array::from_slice(&vec![1.0f32; 128], &[128]).as_dtype(Dtype::Bfloat16)?,
            eps: 1e-6,
            group_size: None,
        })
    };
    let attention = DraftAttention {
        q: zero()?,
        k: zero()?,
        v: identity()?,
        o: identity()?,
        q_norm: norm()?,
        k_norm: norm()?,
        heads: 1,
        kv_heads: 1,
        head_dim: 128,
        theta: 10000000.0,
        window: Some(8192),
    };
    let mut cache = FullAttentionCache::with_dtype(
        crate::metal_context::execution_context(),
        1,
        128,
        Dtype::Bfloat16,
    );
    let keys = ops::zeros::<f32>(&[1, 1, 8195, 128])?.as_dtype(Dtype::Bfloat16)?;
    let values: Vec<f32> = (0..8195 * 128)
        .map(|i| if i < 4 * 128 { 10000.0 } else { 1.0 })
        .collect();
    let values = Array::from_slice(&values, &[1, 1, 8195, 128]).as_dtype(Dtype::Bfloat16)?;
    let _ = cache.update(&keys, &values)?;
    let x = ops::zeros::<f32>(&[1, 7, 128])?.as_dtype(Dtype::Bfloat16)?;
    let output = attention
        .forward_block(&x, &mut cache, 0)?
        .as_dtype(Dtype::Float32)?
        .to_vec1::<f32>()?;
    assert!(
        output.iter().all(|&x| (x - 1.0).abs() <= 0.00390625),
        "old values leaked through sliding window: {:?}",
        &output[..8]
    );
    assert_eq!(cache.offset, 8195, "proposal rows must remain transient");
    let mut compact = compact_draft_caches(&[cache], 8191)?.remove(0);
    let compact_output = attention
        .forward_block(&x, &mut compact, 4)?
        .as_dtype(Dtype::Float32)?
        .to_vec1::<f32>()?;
    assert_eq!(output, compact_output);
    assert_eq!(compact.offset, 8191);
    Ok(())
}

#[test]
fn compact_draft_window_preserves_absolute_rope_and_values() -> anyhow::Result<()> {
    let linear = || -> anyhow::Result<DenseLinear> {
        let weights: Vec<f32> = (0..128 * 128)
            .map(|i| if i / 128 == i % 128 { 1.0 } else { 0.0 })
            .collect();
        Ok(DenseLinear {
            weight: Array::from_slice(&weights, &[128, 128]).as_dtype(Dtype::Bfloat16)?,
            bias: None,
            quantized: None,
        })
    };
    let norm = || -> anyhow::Result<RmsNorm> {
        Ok(RmsNorm {
            weight: Array::from_slice(&[1f32; 128], &[128]).as_dtype(Dtype::Bfloat16)?,
            eps: 1e-6,
            group_size: None,
        })
    };
    let attention = DraftAttention {
        q: linear()?,
        k: linear()?,
        v: linear()?,
        o: linear()?,
        q_norm: norm()?,
        k_norm: norm()?,
        heads: 1,
        kv_heads: 1,
        head_dim: 128,
        theta: 10000000.,
        window: Some(8192),
    };
    let mut dense = FullAttentionCache::new(crate::metal_context::execution_context(), 1, 128);
    let mut bounded = FullAttentionCache::new(crate::metal_context::execution_context(), 1, 128);
    let mut origin = 0;
    for chunk in 0..9 {
        let data: Vec<f32> = (0..2048 * 128)
            .map(|i| (((i + chunk * 271) * 37 % 1009) as f32 - 504.) / 512.)
            .collect();
        let input = Array::from_slice(&data, &[1, 2048, 128]).as_dtype(Dtype::Bfloat16)?;
        attention.append_context(&input, &mut dense, 0)?;
        attention.append_context(&input, &mut bounded, origin)?;
        if bounded.offset > 8191 + 2048 {
            let dropped = bounded.offset - 8191;
            bounded = compact_draft_caches(&[bounded], 8191)?.remove(0);
            origin += dropped;
        }
        assert_eq!(origin + bounded.offset, dense.offset);
        assert!(bounded.capacity <= 2 * (8191 + 2048 + 256));
        let probe = input.index((.., 0..7, ..));
        let expected = attention
            .forward_block(&probe, &mut dense, 0)?
            .to_vec1::<u16>()?;
        let actual = attention
            .forward_block(&probe, &mut bounded, origin)?
            .to_vec1::<u16>()?;
        assert_eq!(actual, expected, "window after chunk {chunk}");
        for (full, compact) in [
            (&dense.keys, &bounded.keys),
            (&dense.values, &bounded.values),
        ] {
            let full =
                full.as_ref()
                    .unwrap()
                    .index((.., .., origin as i32..dense.offset as i32, ..));
            let compact = compact
                .as_ref()
                .unwrap()
                .index((.., .., 0..bounded.offset as i32, ..));
            assert_eq!(full.to_vec1::<u16>()?, compact.to_vec1::<u16>()?);
        }
    }
    assert!(origin > 8192);
    Ok(())
}

#[test]
#[ignore = "requires the real DSpine checkpoint and independent MLX fixture"]
fn model_draft_matches_mlx_fixture() -> anyhow::Result<()> {
    check_mlx_fixture(
        "LISA_QWEN_DSPINE_MODEL",
        "LISA_DSPINE_FIXTURE_DIR",
        include_str!("../../../tests/fixtures/dspine/inventory.json"),
        7,
    )
}

#[test]
#[ignore = "requires the derived 4-bit drafter and independent quantized MLX fixture"]
fn quantized_draft_matches_mlx_fixture() -> anyhow::Result<()> {
    check_mlx_fixture(
        "LISA_QWEN_DSPINE_4BIT_MODEL",
        "LISA_DSPINE_4BIT_FIXTURE_DIR",
        include_str!("../../../tests/fixtures/dspine/inventory-4bit.json"),
        7,
    )
}

#[test]
#[ignore = "requires the DSpine checkpoint and independent full-block MLX fixture"]
fn full_block_draft_matches_mlx_fixture() -> anyhow::Result<()> {
    check_mlx_fixture(
        "LISA_QWEN_DSPINE_MODEL",
        "LISA_DSPINE_B16_FIXTURE_DIR",
        include_str!("../../../tests/fixtures/dspine/inventory-b16.json"),
        16,
    )
}

fn check_mlx_fixture(
    model_env: &str,
    fixture_env: &str,
    inventory_json: &str,
    block: i32,
) -> anyhow::Result<()> {
    struct Target(Array);
    impl DSpineTarget<Array> for Target {
        fn embed_tokens(&self, _ids: &Array) -> anyhow::Result<Array> {
            Ok(self.0.clone())
        }
        fn head_logits(&self, hidden: &Array) -> anyhow::Result<Array> {
            let values: Vec<f32> = (0..hidden.dim(0))
                .flat_map(|row| (0..512).map(move |id| ((id * 37 + row * 17) % 257) as f32 / 256.0))
                .collect();
            Ok(Array::from_slice(&values, &[hidden.dim(0), 512]))
        }
        fn head_rows(&self, ids: &Array) -> anyhow::Result<Array> {
            let ids = ids.as_dtype(Dtype::Uint32)?.to_vec1::<u32>()?;
            let values: Vec<f32> = ids
                .iter()
                .flat_map(|&id| {
                    (0..5120).map(move |column| {
                        ((((id * 5120 + column) * 37 + 11) % 257) as f32 - 128.0) / 256.0
                    })
                })
                .collect();
            Ok(Array::from_slice(&values, &[ids.len() as i32, 5120]).as_dtype(Dtype::Bfloat16)?)
        }
    }
    let model = std::env::var(model_env).with_context(|| format!("set {model_env}"))?;
    let fixture = std::path::PathBuf::from(
        std::env::var(fixture_env).with_context(|| format!("set {fixture_env}"))?,
    );
    use sha2::{Digest, Sha256};
    let inventory: serde_json::Value = serde_json::from_str(inventory_json)?;
    for (name, expected) in inventory.as_object().context("invalid fixture inventory")? {
        let bytes = std::fs::read(fixture.join(name))?;
        ensure!(
            bytes.len() as u64 == expected["bytes"].as_u64().unwrap(),
            "fixture size mismatch: {name}"
        );
        ensure!(
            format!("{:x}", Sha256::digest(&bytes)) == expected["sha256"].as_str().unwrap(),
            "fixture hash mismatch: {name}"
        );
    }
    let mut draft = DSpineDraft::load(
        crate::metal_context::execution_context(),
        &crate::checkpoint::resolve_model_dir(&model)?,
        248320,
        5120,
        0,
    )?;
    ensure!(
        draft.fc.weight.t.device().nax(),
        "this pinned fixture requires the M5 NAX draft projection path"
    );
    draft.block_size = block as usize;
    let values = |shape: &[i32]| -> anyhow::Result<Array> {
        let n: i32 = shape.iter().product();
        let v: Vec<f32> = (0..n)
            .map(|i| (((i * 37 + 11) % 257) as f32 - 128.0) / 256.0)
            .collect();
        Ok(Array::from_slice(&v, shape).as_dtype(Dtype::Bfloat16)?)
    };
    let target = Target(values(&[1, block, 5120])?);
    draft.append_target_hidden(&values(&[1, 8, 25600])?, &[0; 8])?;
    let read = |name: &str, shape: &[i32]| -> anyhow::Result<Array> {
        let bytes = std::fs::read(fixture.join(format!("{name}.f32")))?;
        let v: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|x| f32::from_le_bytes(x.try_into().unwrap()))
            .collect();
        Ok(Array::from_slice(&v, shape).as_dtype(Dtype::Bfloat16)?)
    };
    let mut captures = Vec::new();
    let projection_rows: &[i32] = if draft.cfg.quantization.is_some() {
        &[1, 2, 3, 4, 5, 6, 7, 8, 12, 13, 16, 32, 36, 64]
    } else {
        &[1, block, 16, 32, 64]
    };
    for &m in projection_rows {
        captures.push((
            format!("fc_m{m}"),
            draft.fc.forward(&values(&[1, m, 25600])?)?,
        ));
    }
    if draft.cfg.quantization.is_some() {
        for (i, layer) in draft.layers.iter().enumerate() {
            for &m in projection_rows {
                let input = values(&[1, m, 5120])?;
                captures.push((
                    format!("context_k{i}_m{m}"),
                    layer.attention.k.forward(&input)?,
                ));
                captures.push((
                    format!("context_v{i}_m{m}"),
                    layer.attention.v.forward(&input)?,
                ));
            }
        }
    }
    captures.push(("fc".to_owned(), draft.fc.forward(&values(&[1, 8, 25600])?)?));
    for i in 0..5 {
        let gate = read(&format!("gate{i}"), &[1, block, 17408])?;
        let up = read(&format!("up{i}"), &[1, block, 17408])?;
        captures.push((
            format!("act{i}"),
            ops::sigmoid(&gate)?.multiply(&gate)?.multiply(&up)?,
        ));
    }
    let context = draft
        .hidden_norm
        .forward(&draft.fc.forward(&values(&[1, 8, 25600])?)?)?;
    captures.push(("context".to_owned(), context));
    let context_ref = read("context", &[1, 8, 5120])?;
    let x_ref = read("input", &[1, block, 5120])?;
    let a = &draft.layers[0].attention;
    captures.push(("qproj0".to_owned(), a.q.forward(&x_ref)?));
    let q = a
        .q_norm
        .forward(&read("qproj0", &[1, block, 4096])?.reshape(&[1, block, 32, 128])?)?
        .transpose_axes(&[0, 2, 1, 3])?
        .contiguous()?;
    captures.push(("qnorm0".to_owned(), q));
    let q = fast::rope(
        &read("qnorm0", &[1, 32, block, 128])?,
        128,
        false,
        Some(a.theta),
        1.0,
        8,
        None,
    )?;
    captures.push(("qrope0".to_owned(), q));
    let (ck, cv) = a.project_kv(&context_ref, 0)?;
    captures.push(("ck0".to_owned(), ck));
    captures.push(("cv0".to_owned(), cv));
    let (pk, pv) = a.project_kv(&x_ref, 8)?;
    captures.push(("pk0".to_owned(), pk));
    captures.push(("pv0".to_owned(), pv));
    let keys = Array::concatenate(
        &[
            &read("ck0", &[1, 8, 8, 128])?,
            &read("pk0", &[1, 8, block, 128])?,
        ],
        2,
    )?;
    let vals = Array::concatenate(
        &[
            &read("cv0", &[1, 8, 8, 128])?,
            &read("pv0", &[1, 8, block, 128])?,
        ],
        2,
    )?;
    let av = fast::scaled_dot_product_attention(
        &read("qrope0", &[1, 32, block, 128])?,
        &keys,
        &vals,
        128f32.powf(-0.5),
        fast::ScaledDotProductAttentionMask::Causal,
        None,
    )?;
    captures.push(("av0".to_owned(), av));
    let mut x = draft.layers[0].input_norm.forward(&target.0)?;
    let mut residual = target.0.clone();
    captures.push(("input".to_owned(), x.clone()));
    let ids = Array::from_slice(
        &std::iter::once(42i32)
            .chain(std::iter::repeat_n(248070, block as usize - 1))
            .collect::<Vec<_>>(),
        &[1, block],
    );
    for i in 0..draft.layers.len() {
        let layer = &draft.layers[i];
        let attn = layer
            .attention
            .forward_block(&x, &mut draft.caches[i], draft.cache_start)?;
        captures.push((format!("attn{i}"), attn.clone()));
        let h = residual.add(&attn)?;
        let norm = layer.post_norm.forward(&h)?;
        captures.push((format!("norm{i}"), norm.clone()));
        let gate = layer.gate.forward(&norm)?;
        let up = layer.up.forward(&norm)?;
        captures.push((format!("gate{i}"), gate.clone()));
        let act = ops::sigmoid(&gate)?.multiply(&gate)?.multiply(&up)?;
        captures.push((format!("act{i}"), act.clone()));
        let raw = h.add(&layer.down.forward(&act)?)?;
        captures.push((format!("layer{i}"), raw.clone()));
        let next_norm = if i < 4 {
            &draft.layers[i + 1].input_norm
        } else {
            &draft.final_norm
        };
        let (normed, updated, _) = draft.bridge_layer(&raw, &ids, i, next_norm, false)?;
        x = normed;
        residual = updated;
        captures.push((format!("bridge{i}"), x.clone()));
    }
    let (rows, stash) = draft.draft_hidden(42, &target)?;
    let candidates = draft.candidates(&rows, &target)?;
    let proposals = draft.greedy_refine(
        &rows,
        DraftStash {
            raw: stash.raw.clone(),
            packed: stash.packed.clone(),
            rms: stash.rms.clone(),
        },
        42,
        &target,
    )?;
    let tables = draft.build_serial_tables(&target, 512)?;
    let count = tables.rows_per_projection;
    let ids: Vec<u32> = (0..count).map(|i| ((i * 37 + 503) % 512) as u32).collect();
    let ids = Array::from_slice(&ids, &[count as i32]);
    assert_eq!(
        take_axis(&tables.tg, &ids, 0)?.to_vec1::<u16>()?,
        draft
            .project_tg(&target.head_rows(&ids)?)?
            .to_vec1::<u16>()?,
        "cached TG rows differ from selected projections"
    );
    assert_eq!(
        take_axis(&tables.tsv, &ids, 0)?.to_vec1::<u16>()?,
        draft.project_tsv(&ids)?.to_vec1::<u16>()?,
        "cached TSV rows differ from selected projections"
    );
    draft.serial_tables = Some(tables);
    let cached_proposals = draft.greedy_refine(
        &rows,
        DraftStash {
            raw: stash.raw.clone(),
            packed: stash.packed.clone(),
            rms: stash.rms.clone(),
        },
        42,
        &target,
    )?;
    assert_eq!(
        cached_proposals.to_vec1::<u32>()?,
        proposals.to_vec1::<u32>()?
    );
    captures.push(("candidates".to_owned(), candidates));
    captures.push(("proposals".to_owned(), proposals));
    captures.extend([
        ("rows".to_owned(), rows),
        ("raw".to_owned(), stash.raw),
        ("packed".to_owned(), stash.packed),
        ("rms".to_owned(), stash.rms),
    ]);
    let mut failures = Vec::new();
    for (name, actual) in captures {
        let bytes = std::fs::read(fixture.join(format!("{name}.f32")))?;
        let expected: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|x| f32::from_le_bytes(x.try_into().unwrap()))
            .collect();
        let actual = actual.as_dtype(Dtype::Float32)?.to_vec1::<f32>()?;
        assert_eq!(actual.len(), expected.len());
        let sq: f64 = actual
            .iter()
            .zip(&expected)
            .map(|(a, b)| (*a as f64 - *b as f64).powi(2))
            .sum();
        let norm: f64 = expected.iter().map(|x| (*x as f64).powi(2)).sum();
        let relative_l2 = (sq / norm.max(1e-30)).sqrt();
        let max_abs = actual
            .iter()
            .zip(&expected)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        eprintln!("MLX draft {name}: relative_l2={relative_l2:.8}, max_abs={max_abs:.8}");
        if actual != expected {
            failures.push(name);
        }
    }
    assert_eq!(draft.offset(), 8);
    // Exercise the actual companion's append/evict/rewind/rebuild lifecycle.
    // Full attention arithmetic and retained values are checked separately
    // against dense history above; this checks the model-owned metadata.
    let initial = draft.draft_block(42, &target)?.to_vec1::<u32>()?;
    let taps = values(&[1, 8, 25600])?.repeat_axis(256, 1)?;
    for _ in 0..6 {
        draft.append_target_hidden(&taps, &vec![1; 2048])?;
        lisa_metal::Stream::thread_local_or_default()
            .runtime()
            .synchronize()?;
    }
    assert_eq!(draft.offset(), 8 + 6 * 2048);
    assert!(draft.cache_start > 0);
    assert!(draft.caches.iter().all(|c| c.offset <= 8191 + 2048));
    let offset = draft.offset();
    draft.restore_offset(offset);
    assert_eq!(draft.offset(), offset);
    assert!(draft.context_matches(&[vec![0; 8], vec![1; 6 * 2048]].concat()));
    // This prefix predates retained history. A reset must force the tower's
    // context mismatch path to reconstruct taps from the target snapshot.
    draft.restore_offset(8);
    assert_eq!(draft.offset(), 0);
    assert!(!draft.context_matches(&[0; 8]));
    draft.append_target_hidden(&values(&[1, 8, 25600])?, &[0; 8])?;
    assert!(draft.context_matches(&[0; 8]));
    assert_eq!(draft.draft_block(42, &target)?.to_vec1::<u32>()?, initial);
    assert!(
        failures.is_empty(),
        "Draft differs from independent MLX reference: {failures:?}"
    );
    Ok(())
}

#[test]
fn prepared_dense_weights_preserve_bf16_projection_bits() -> anyhow::Result<()> {
    for (input, output) in [(128, 512), (5120, 4096)] {
        let values: Vec<f32> = (0..input * output)
            .map(|i| (((i * 37 + 11) % 257) as f32 - 128.0) / 256.0)
            .collect();
        let weight = Array::from_slice(&values, &[output, input]).as_dtype(Dtype::Bfloat16)?;
        let prepared =
            DenseLinear::prepare_weight(&crate::metal_context::execution_context(), &weight)?;
        // Verify the complete stored tensor, not only a greedy prediction.
        assert_eq!(
            weight.to_vec1::<u16>()?,
            prepared.contiguous()?.to_vec1::<u16>()?
        );
        for rows in [1, 7, 32] {
            let x: Vec<f32> = (0..rows * input)
                .map(|i| (((i * 13 + 5) % 251) as f32 - 125.0) / 128.0)
                .collect();
            let x = Array::from_slice(&x, &[1, rows, input]).as_dtype(Dtype::Bfloat16)?;
            let expected = x.matmul(&weight.transpose_axes(&[1, 0])?)?;
            let actual = x.matmul(&prepared.transpose_axes(&[1, 0])?)?;
            assert_eq!(
                expected.to_vec1::<u16>()?,
                actual.to_vec1::<u16>()?,
                "projection changed at M={rows}, K={input}, N={output}"
            );
        }
    }
    Ok(())
}

trait DenseLinearOracle0 {
    fn forward(&self, x: &Array) -> anyhow::Result<Array>;
}
impl DenseLinearOracle0 for DenseLinear {
    fn forward(&self, x: &Array) -> anyhow::Result<Array> {
        let spec = lisa_models::qwen3_5::dspine::layers::Linear {
            shape: x.shape().to_vec(),
            output: self.weight.dim(0),
            quantized: self.quantized.is_some(),
            bias: self.bias.is_some(),
            small_rows: x.t.device().nax(),
        };
        let mut inputs = vec![x];
        inputs.extend(self.bindings());
        crate::bindings::dspine_layers::linear(spec, &inputs)
    }
}
trait DraftAttentionOracle1 {
    fn project_kv(&self, x: &Array, offset: usize) -> anyhow::Result<(Array, Array)>;
    fn forward_block(
        &self,
        x: &Array,
        cache: &mut FullAttentionCache,
        cache_start: usize,
    ) -> anyhow::Result<Array>;
}
impl DraftAttentionOracle1 for DraftAttention {
    fn project_kv(&self, x: &Array, offset: usize) -> anyhow::Result<(Array, Array)> {
        let (b, s) = (x.dim(0), x.dim(1));
        let k = self
            .k_norm
            .forward(&self.k.forward(x)?.reshape(&[
                b,
                s,
                self.kv_heads as i32,
                self.head_dim as i32,
            ])?)?
            .transpose_axes(&[0, 2, 1, 3])?
            .contiguous()?;
        let v = self
            .v
            .forward(x)?
            .reshape(&[b, s, self.kv_heads as i32, self.head_dim as i32])?
            .transpose_axes(&[0, 2, 1, 3])?
            .contiguous()?;
        let k = fast::rope(
            &k,
            self.head_dim as i32,
            false,
            Some(self.theta),
            1.0,
            offset as i32,
            None,
        )?;
        Ok((k, v))
    }
    fn forward_block(
        &self,
        x: &Array,
        cache: &mut FullAttentionCache,
        cache_start: usize,
    ) -> anyhow::Result<Array> {
        self.execute(x, cache, cache_start, false)?
            .context("draft attention returned no activation")
    }
}
trait DSpineDraftOracle2 {
    fn bridge_layer(
        &self,
        hidden: &Array,
        input_ids: &Array,
        layer: usize,
        next_norm: &RmsNorm,
        stash: bool,
    ) -> anyhow::Result<(Array, Array, Option<DraftStash>)>;
}
impl DSpineDraftOracle2 for DSpineDraft {
    fn bridge_layer(
        &self,
        hidden: &Array,
        input_ids: &Array,
        layer: usize,
        next_norm: &RmsNorm,
        stash: bool,
    ) -> anyhow::Result<(Array, Array, Option<DraftStash>)> {
        let spec = lisa_models::qwen3_5::dspine::bridge::Bridge {
            shape: [hidden.dim(0), hidden.dim(1), hidden.dim(2)],
            layer,
            layers: self.layers.len(),
            rank: self.bridge.rank as i32,
            codes: self.bridge.code_size as i32,
            epsilon: self.bridge.eps.to_bits(),
            next_epsilon: next_norm.eps.to_bits(),
            stash,
            small_rows: hidden.t.device().nax(),
        };
        let output = bound_bridge::bridge(
            spec,
            &[
                hidden,
                input_ids,
                &next_norm.weight,
                &self.bridge.read_weight,
                &self.bridge.early_read_weights,
                &self.bridge.codebook,
                &self.bridge.packed_weight,
                &self.bridge.gate_bias,
                &self.bridge.feature_write_weights,
            ],
        )?;
        let mut output = output.into_iter();
        let normed = output.next().context("missing normalized bridge result")?;
        let updated = output.next().context("missing bridge residual")?;
        let saved = if stash {
            Some(DraftStash {
                raw: output.next().context("missing bridge raw state")?,
                rms: output.next().context("missing bridge RMS state")?,
                packed: output.next().context("missing bridge projection state")?,
            })
        } else {
            None
        };
        Ok((normed, updated, saved))
    }
}

trait DraftAttentionOracle3 {
    fn execute(
        &self,
        x: &Array,
        cache: &mut FullAttentionCache,
        cache_start: usize,
        context_only: bool,
    ) -> anyhow::Result<Option<Array>>;
    fn append_context(
        &self,
        x: &Array,
        cache: &mut FullAttentionCache,
        cache_start: usize,
    ) -> anyhow::Result<()>;
}
impl DraftAttentionOracle3 for DraftAttention {
    fn execute(
        &self,
        x: &Array,
        cache: &mut FullAttentionCache,
        cache_start: usize,
        context_only: bool,
    ) -> anyhow::Result<Option<Array>> {
        let spec = self.description(
            x.shape(),
            x.t.device().nax(),
            cache,
            cache_start,
            context_only,
        );
        let mut inputs = vec![x];
        for projection in [&self.q, &self.k, &self.v, &self.o] {
            inputs.extend(projection.bindings());
        }
        inputs.extend([&self.q_norm.weight, &self.k_norm.weight]);
        crate::bindings::dspine_layers::attention(spec, &inputs, cache)
    }
    fn append_context(
        &self,
        x: &Array,
        cache: &mut FullAttentionCache,
        cache_start: usize,
    ) -> anyhow::Result<()> {
        if x.dim(1) != 0 {
            self.execute(x, cache, cache_start, true)?;
        }
        Ok(())
    }
}
