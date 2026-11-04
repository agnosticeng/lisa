//! Load-and-generate smoke for the glm53 tower (env-gated: requires the
//! local artifact and `LISA_GLM53_SMOKE=1`).

use super::mhc::narrow;
use crate::core::cache::LayerCache;
use crate::models::LanguageModel;
use crate::models::glm53::config::ModelConfig;
use crate::models::glm53::tower::Tower;
use lisa_mlx::ops::indexing::IndexOp;
use lisa_mlx::{Array, Dtype};

fn verify_capture_layers(spec: Option<&str>, layer_count: usize) -> anyhow::Result<Vec<usize>> {
    let layers = match spec {
        Some(spec) => spec
            .split(',')
            .map(|item| {
                item.trim().parse::<usize>().map_err(|error| {
                    anyhow::anyhow!("invalid GLM verify capture layer {item:?}: {error}")
                })
            })
            .collect::<anyhow::Result<Vec<_>>>()?,
        None => (0..layer_count).collect(),
    };
    anyhow::ensure!(
        !layers.is_empty() && layers.iter().all(|&layer| layer < layer_count),
        "verify capture layers must be valid indices below {layer_count}"
    );
    Ok(layers)
}

fn capture_glm53_cache_state(
    capture: &crate::core::capture::Capture,
    caches: &[LayerCache],
    mtp_cache: &crate::core::cache::IndexerTape,
) {
    let mut metadata = Vec::with_capacity(caches.len() * 7);
    for (layer, cache) in caches.iter().enumerate() {
        let prefix = format!("glm53.cache.layer_{layer:02}");
        match cache {
            LayerCache::Native(_) => panic!("GLM smoke received a native cache"),
            LayerCache::Full(full) => {
                let tape = &full.indexer_tape;
                metadata.extend_from_slice(&[
                    0.0,
                    full.offset as f32,
                    tape.total as f32,
                    tape.raw_start() as f32,
                    tape.pooled_upto as f32,
                    tape.compact_total() as f32,
                    tape.compression_ratio() as f32,
                ]);
                if let Some(keys) = full.keys.as_ref() {
                    capture.store(
                        &format!("{prefix}.keys"),
                        &keys.index((.., .., 0..full.offset as i32, ..)),
                    );
                }
                if let Some(values) = full.values.as_ref() {
                    capture.store(
                        &format!("{prefix}.values"),
                        &values.index((.., .., 0..full.offset as i32, ..)),
                    );
                }
                if let Some(raw) = tape.raw_slice(tape.raw_start(), tape.total) {
                    capture.store(&format!("{prefix}.indexer_raw"), &raw);
                }
                if let Some(compact) = tape.compact_slice(0, tape.compact_total()) {
                    capture.store(&format!("{prefix}.compact_cache"), &compact);
                }
                if let Some(pooled) = tape.pooled_view() {
                    capture.store(&format!("{prefix}.pooled_cache"), &pooled);
                }
                if let Some(ple) = &full.ple_conv {
                    capture.store(&format!("{prefix}.ple_conv"), ple);
                }
            }
            LayerCache::Linear(linear) => {
                metadata.extend_from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
                if let Some(conv) = &linear.conv {
                    capture.store(&format!("{prefix}.kda_conv"), conv);
                }
                if let Some(ssm) = &linear.ssm {
                    capture.store(&format!("{prefix}.kda_ssm"), ssm);
                }
                if let Some(ple) = &linear.ple_conv {
                    capture.store(&format!("{prefix}.ple_conv"), ple);
                }
            }
        }
    }
    capture.store(
        "glm53.cache_meta",
        &Array::from_slice(&metadata, &[caches.len() as i32, 7]),
    );
    capture.store(
        "glm53.mtp_cache_meta",
        &Array::from_slice(
            &[
                mtp_cache.raw_start() as f32,
                mtp_cache.total as f32,
                mtp_cache.pooled_upto as f32,
                mtp_cache.compact_total() as f32,
                mtp_cache.compression_ratio() as f32,
            ],
            &[5],
        ),
    );
    if let Some(compact) = mtp_cache.compact_slice(0, mtp_cache.compact_total()) {
        capture.store("glm53.mtp_compact_cache", &compact);
    }
}

fn save_glm53_cache_transaction(
    prefix: &std::ffi::OsStr,
    cycle: usize,
    carry: u32,
    draft: u32,
    layers: &[usize],
    caches: &[LayerCache],
    mtp_cache: &crate::core::cache::IndexerTape,
) -> anyhow::Result<()> {
    let capture = crate::core::capture::Capture {
        layers: layers.to_vec(),
        state_only: false,
        kda_prefill_only: false,
        preserve_dispatch: true,
        entries: std::sync::Mutex::new(std::collections::HashMap::new()),
        shapes: std::sync::Mutex::new(std::collections::HashMap::new()),
        flushed: std::sync::atomic::AtomicBool::new(false),
        ..Default::default()
    };
    capture_glm53_cache_state(&capture, caches, mtp_cache);
    capture.store(
        "glm53.transaction_meta",
        &Array::from_slice(&[cycle as f32, carry as f32, draft as f32], &[3]),
    );
    let prefix = std::path::Path::new(prefix)
        .to_str()
        .ok_or_else(|| anyhow::anyhow!("transaction capture prefix is not UTF-8"))?;
    let path = format!("{prefix}.cycle-{cycle:03}.npz");
    capture.flush(&path)?;
    eprintln!("[glm53-kda-accept-capture] saved transaction cycle {cycle} to {path}");
    Ok(())
}

fn assert_f32_state_equal(a: &Array, b: &Array, label: &str) -> anyhow::Result<()> {
    anyhow::ensure!(
        a.shape() == b.shape(),
        "{label} cache shapes differ: {:?} vs {:?}",
        a.shape(),
        b.shape()
    );
    let as_f32 = |value: &Array| -> anyhow::Result<Vec<f32>> {
        Ok(value
            .as_dtype(Dtype::Float32)?
            .contiguous()?
            .reshape(&[value.size() as i32])?
            .to_vec1::<f32>()?)
    };
    let av = as_f32(a)?;
    let bv = as_f32(b)?;
    let mismatches = av
        .iter()
        .zip(&bv)
        .filter(|(x, y)| x.to_bits() != y.to_bits())
        .count();
    let max_abs = av
        .iter()
        .zip(&bv)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    let first_mismatch = av
        .iter()
        .zip(&bv)
        .position(|(x, y)| x.to_bits() != y.to_bits());
    anyhow::ensure!(
        mismatches == 0,
        "{label} cache differs: {mismatches}/{} values, max_abs={max_abs:e}, first={first_mismatch:?} ({:?} vs {:?}), shape={:?}/{:?}",
        av.len(),
        first_mismatch.map(|i| av[i]),
        first_mismatch.map(|i| bv[i]),
        a.shape(),
        b.shape()
    );
    Ok(())
}

fn assert_optional_f32_state_equal(
    a: &Option<Array>,
    b: &Option<Array>,
    label: &str,
) -> anyhow::Result<()> {
    match (a, b) {
        (Some(a), Some(b)) => assert_f32_state_equal(a, b, label),
        (None, None) => Ok(()),
        _ => anyhow::bail!("{label} cache presence differs"),
    }
}

fn assert_optional_array_layout(
    a: Option<Array>,
    b: Option<Array>,
    label: &str,
) -> anyhow::Result<()> {
    match (a, b) {
        (Some(a), Some(b)) => {
            anyhow::ensure!(
                a.shape() == b.shape(),
                "{label} cache shapes differ: {:?} vs {:?}",
                a.shape(),
                b.shape()
            );
            Ok(())
        }
        (None, None) => Ok(()),
        _ => anyhow::bail!("{label} cache presence differs"),
    }
}

fn assert_session_cache_layout_equal(a: &[LayerCache], b: &[LayerCache]) -> anyhow::Result<()> {
    anyhow::ensure!(a.len() == b.len(), "serial/MTP cache depth differs");
    for (layer, (a, b)) in a.iter().zip(b).enumerate() {
        match (a, b) {
            (LayerCache::Full(a), LayerCache::Full(b)) => {
                anyhow::ensure!(a.offset == b.offset, "layer {layer} KV offsets differ");
                assert_optional_array_layout(
                    a.keys.clone(),
                    b.keys.clone(),
                    &format!("layer {layer} keys"),
                )?;
                assert_optional_array_layout(
                    a.values.clone(),
                    b.values.clone(),
                    &format!("layer {layer} values"),
                )?;
                anyhow::ensure!(
                    a.indexer_tape.total == b.indexer_tape.total
                        && a.indexer_tape.pooled_upto == b.indexer_tape.pooled_upto,
                    "layer {layer} indexer cache lengths differ"
                );
                for (name, av, bv) in [
                    (
                        "indexer raw",
                        a.indexer_tape.raw_live_for_test(),
                        b.indexer_tape.raw_live_for_test(),
                    ),
                    (
                        "indexer pooled",
                        a.indexer_tape.pooled_view(),
                        b.indexer_tape.pooled_view(),
                    ),
                    (
                        "compact KV",
                        a.indexer_tape.compact_live_for_test(),
                        b.indexer_tape.compact_live_for_test(),
                    ),
                ] {
                    assert_optional_array_layout(av, bv, &format!("layer {layer} {name}"))?;
                }
                assert_optional_array_layout(
                    a.ple_conv.clone(),
                    b.ple_conv.clone(),
                    &format!("layer {layer} PLE"),
                )?;
            }
            (LayerCache::Linear(a), LayerCache::Linear(b)) => {
                assert_optional_array_layout(
                    a.conv.clone(),
                    b.conv.clone(),
                    &format!("layer {layer} KDA convolution"),
                )?;
                assert_optional_array_layout(
                    a.ssm.clone(),
                    b.ssm.clone(),
                    &format!("layer {layer} KDA recurrence"),
                )?;
                assert_optional_array_layout(
                    a.ple_conv.clone(),
                    b.ple_conv.clone(),
                    &format!("layer {layer} PLE"),
                )?;
            }
            _ => anyhow::bail!("layer {layer} serial/MTP cache kinds differ"),
        }
    }
    Ok(())
}

fn assert_session_caches_equal(a: &[LayerCache], b: &[LayerCache]) -> anyhow::Result<()> {
    anyhow::ensure!(a.len() == b.len(), "serial/MTP cache depth differs");
    let mut differences = Vec::new();
    macro_rules! check {
        ($result:expr) => {
            if let Err(error) = $result {
                differences.push(format!("{error:#}"));
            }
        };
    }
    for (layer, (a, b)) in a.iter().zip(b).enumerate() {
        match (a, b) {
            (LayerCache::Full(a), LayerCache::Full(b)) => {
                anyhow::ensure!(a.offset == b.offset, "layer {layer} KV offsets differ");
                let offset = a.offset as i32;
                for (name, a, b) in [("keys", &a.keys, &b.keys), ("values", &a.values, &b.values)] {
                    match (a, b) {
                        (Some(a), Some(b)) => {
                            let a = a.index((.., .., 0..offset, ..)).contiguous()?;
                            let b = b.index((.., .., 0..offset, ..)).contiguous()?;
                            check!(assert_f32_state_equal(
                                &a,
                                &b,
                                &format!("layer {layer} {name}")
                            ));
                        }
                        (None, None) => {}
                        _ => {
                            differences.push(format!("layer {layer} {name} cache presence differs"))
                        }
                    }
                }
                if a.indexer_tape.total != b.indexer_tape.total
                    || a.indexer_tape.pooled_upto != b.indexer_tape.pooled_upto
                {
                    differences.push(format!("layer {layer} indexer cache lengths differ"));
                }
                for (name, av, bv) in [
                    (
                        "indexer raw",
                        a.indexer_tape.raw_live_for_test(),
                        b.indexer_tape.raw_live_for_test(),
                    ),
                    (
                        "indexer pooled",
                        a.indexer_tape.pooled_view(),
                        b.indexer_tape.pooled_view(),
                    ),
                    (
                        "compact KV",
                        a.indexer_tape.compact_live_for_test(),
                        b.indexer_tape.compact_live_for_test(),
                    ),
                ] {
                    match (av, bv) {
                        (Some(av), Some(bv)) => {
                            check!(assert_f32_state_equal(
                                &av,
                                &bv,
                                &format!("layer {layer} {name}")
                            ));
                        }
                        (None, None) => {}
                        _ => {
                            differences.push(format!("layer {layer} {name} cache presence differs"))
                        }
                    }
                }
                check!(assert_optional_f32_state_equal(
                    &a.ple_conv,
                    &b.ple_conv,
                    &format!("layer {layer} PLE"),
                ));
            }
            (LayerCache::Linear(a), LayerCache::Linear(b)) => {
                check!(assert_optional_f32_state_equal(
                    &a.conv,
                    &b.conv,
                    &format!("layer {layer} KDA convolution"),
                ));
                check!(assert_optional_f32_state_equal(
                    &a.ssm,
                    &b.ssm,
                    &format!("layer {layer} KDA recurrence"),
                ));
                check!(assert_optional_f32_state_equal(
                    &a.ple_conv,
                    &b.ple_conv,
                    &format!("layer {layer} PLE"),
                ));
            }
            _ => differences.push(format!("layer {layer} serial/MTP cache kinds differ")),
        }
    }
    anyhow::ensure!(
        differences.is_empty(),
        "serial/MTP cache differences:\n{}",
        differences.join("\n")
    );
    Ok(())
}

const DEFAULT_ARTIFACT: &str = concat!(
    env!("HOME"),
    "/.cache/huggingface/hub/models--lisa--glm53-flash-2bit/snapshots/main"
);

fn artifact_dir() -> std::path::PathBuf {
    std::env::var_os("LISA_GLM53_MODEL")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| std::path::PathBuf::from(DEFAULT_ARTIFACT))
}

#[test]
fn capture_real_serial_generation_for_ds4() -> anyhow::Result<()> {
    let Some(prefix) = std::env::var_os("LISA_GLM53_SERIAL_CAPTURE_PREFIX") else {
        eprintln!(
            "LISA_GLM53_SERIAL_CAPTURE_PREFIX=<path> to capture real-checkpoint serial token IDs and timings"
        );
        return Ok(());
    };
    if !lisa_mlx::device::metal_available() {
        eprintln!("[glm53-serial-capture] skipping: no Metal device available");
        return Ok(());
    }

    let dir = artifact_dir();
    let config = ModelConfig::from_json(&dir.join("config.json"))?;
    let mut tower = Tower::load(&dir, config)?;
    let tokenizer = crate::core::tokenizer::Tokenizer::load_file(&dir.join("tokenizer.json"))?;
    let prompt_text = if let Some(path) = std::env::var_os("LISA_GLM53_SERIAL_CAPTURE_PROMPT_FILE")
    {
        std::fs::read_to_string(path)?
    } else {
        std::env::var("LISA_GLM53_SERIAL_CAPTURE_PROMPT")
            .unwrap_or_else(|_| "The capital of France is".to_owned())
    };
    let prompt = tokenizer.encode(&prompt_text, false)?;
    anyhow::ensure!(
        !prompt.is_empty(),
        "serial capture prompt tokenized to empty input"
    );
    let token_limit = std::env::var("LISA_GLM53_SERIAL_CAPTURE_TOKENS")
        .ok()
        .map(|value| value.parse::<usize>())
        .transpose()?
        .unwrap_or(32);
    anyhow::ensure!(
        token_limit > 0,
        "serial capture token limit must be positive"
    );
    anyhow::ensure!(
        prompt.len() + token_limit <= tower.max_position_embeddings(),
        "serial capture prompt and token limit exceed the model context"
    );
    crate::core::generate::set_eos_ids_for_model(tokenizer.im_end_ids.clone(), &tower);

    let mut sampler = crate::core::sampler::Sampler::default();
    let (generated, stats) = crate::core::generate::generate(
        &mut tower,
        &prompt,
        token_limit,
        &mut sampler,
        0,
        &mut |_| Ok(()),
    )?;
    anyhow::ensure!(!generated.is_empty(), "serial capture generated no tokens");

    let prefix = std::path::PathBuf::from(prefix);
    if let Some(parent) = prefix.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let write_ids = |suffix: &str, ids: &[u32]| -> anyhow::Result<()> {
        let mut bytes = Vec::with_capacity(ids.len() * std::mem::size_of::<u32>());
        for &id in ids {
            bytes.extend_from_slice(&id.to_le_bytes());
        }
        std::fs::write(prefix.with_extension(suffix), bytes)?;
        Ok(())
    };
    write_ids("prompt-ids.u32", &prompt)?;
    write_ids("generated-token-ids.u32", &generated)?;
    std::fs::write(prefix.with_extension("prompt.txt"), &prompt_text)?;
    std::fs::write(
        prefix.with_extension("timing.txt"),
        format!(
            "prompt_tokens={}\ngenerated_tokens={}\nprefill_seconds={:.6}\ndecode_seconds={:.6}\nprefill_tokens_per_second={:.6}\ndecode_tokens_per_second={:.6}\n",
            stats.prompt_tokens,
            stats.generated_tokens,
            stats.prefill_seconds,
            stats.decode_seconds,
            stats.prefill_tps(),
            stats.decode_tps(),
        ),
    )?;
    eprintln!(
        "[glm53-serial-capture] prompt_tokens={} generated_tokens={} prefill={:.2} tok/s decode={:.2} tok/s artifacts={}",
        stats.prompt_tokens,
        stats.generated_tokens,
        stats.prefill_tps(),
        stats.decode_tps(),
        prefix.display()
    );
    if std::env::var_os("LISA_TRACE").is_some() || std::env::var_os("LISA_TRACE_JSON").is_some() {
        lisa_mlx::trace::finish();
    }
    Ok(())
}

#[test]
fn target_mtp_generation_matches_serial_real_model() -> anyhow::Result<()> {
    if std::env::var("LISA_GLM53_MTP_PARITY").is_err() {
        eprintln!("LISA_GLM53_MTP_PARITY=1 to run (loads the full GLM checkpoint)");
        return Ok(());
    }
    if !lisa_mlx::device::metal_available() {
        eprintln!("[glm53-mtp-parity] skipping: no Metal device available");
        return Ok(());
    }

    let dir = artifact_dir();
    let config = ModelConfig::from_json(&dir.join("config.json"))?;
    let mut tower = Tower::load(&dir, config)?;
    let tokenizer = crate::core::tokenizer::Tokenizer::load_file(&dir.join("tokenizer.json"))?;
    let prompt = tokenizer.encode("The capital of France is", false)?;
    let max_tokens = 9;

    let mut serial_session = crate::core::session::Session::new(&mut tower);
    let mut serial_sampler = crate::core::sampler::Sampler::default();
    let mut serial_on_token = |_: u32| Ok(());
    let serial_start = std::time::Instant::now();
    let serial = serial_session.generate(
        &mut tower,
        &prompt,
        max_tokens,
        &mut serial_sampler,
        0,
        None,
        &mut serial_on_token,
    )?;
    let serial_elapsed = serial_start.elapsed();

    let mut mtp_session = crate::core::session::Session::new(&mut tower);
    let mut mtp_on_token = |_: u32| Ok(());
    let mtp_start = std::time::Instant::now();
    let mtp = mtp_session.generate_mtp(&mut tower, &prompt, max_tokens, 2, &mut mtp_on_token)?;
    let mtp_elapsed = mtp_start.elapsed();
    eprintln!("[glm53-mtp-parity] serial={serial:?} target_mtp={mtp:?}");
    eprintln!(
        "[glm53-mtp-parity] diagnostic only: serial={:.2} ms/token target_mtp={:.2} ms/token over {max_tokens} tokens",
        serial_elapsed.as_secs_f64() * 1000.0 / max_tokens as f64,
        mtp_elapsed.as_secs_f64() * 1000.0 / max_tokens as f64,
    );
    anyhow::ensure!(
        mtp == serial,
        "real-checkpoint greedy target-MTP tokens differ from serial: mtp={mtp:?}, serial={serial:?}"
    );
    let final_token = *serial.last().expect("non-empty greedy generation");
    let mut expected_mtp_fed = serial_session.fed.clone();
    expected_mtp_fed.push(final_token);
    anyhow::ensure!(
        mtp_session.fed == expected_mtp_fed,
        "target-MTP must retain the DS4 committed boundary including the final emitted token"
    );
    // Serial generation leaves its last token unfed; DS4-style target MTP has
    // already committed it. Complete the serial boundary and compare the next
    // greedy token without feeding the MTP token twice.
    let serial_next = serial_session.decode_token(&mut tower, final_token)?;
    let mtp_next = tower.head(
        mtp_session
            .last_mixed
            .as_ref()
            .expect("target MTP committed final hidden row"),
    )?;
    assert_session_cache_layout_equal(&serial_session.caches, &mtp_session.caches)?;
    // Serial decode and speculative verification deliberately enter different
    // target kernels. Keep this cache comparison diagnostic; verify-vs-DS4
    // fixtures remain the numerical gate.
    if let Err(error) = assert_session_caches_equal(&serial_session.caches, &mtp_session.caches) {
        let details = error.to_string();
        let lines: Vec<_> = details.lines().collect();
        eprintln!(
            "[glm53-mtp-parity] serial vs verify caches differ in {} groups; first differences: {}",
            lines.len().saturating_sub(1),
            lines
                .iter()
                .skip(1)
                .take(3)
                .copied()
                .collect::<Vec<_>>()
                .join(" | ")
        );
    } else {
        eprintln!("[glm53-mtp-parity] serial and verify cache states are bit-identical");
    }
    let serial_next_token =
        lisa_mlx::ops::indexing::argmax_axis(&serial_next, -1, false)?.item_cast::<i32>();
    let mtp_next_token =
        lisa_mlx::ops::indexing::argmax_axis(&mtp_next, -1, false)?.item_cast::<i32>();
    anyhow::ensure!(
        serial_next_token == mtp_next_token,
        "serial/MTP continuation greedy tokens differ: {serial_next_token} vs {mtp_next_token}"
    );
    // Two suffix rows keep this resume probe away from position 15, where the
    // model test intentionally enforces the separate natural-target fixture.
    let suffix = [serial_next_token as u32, 2];
    let mut serial_sampler = crate::core::sampler::Sampler::default();
    let serial_continuation = serial_session.generate(
        &mut tower,
        &suffix,
        3,
        &mut serial_sampler,
        0,
        None,
        &mut |_| Ok(()),
    )?;
    let mtp_continuation = mtp_session.generate_mtp(&mut tower, &suffix, 3, 2, &mut |_| Ok(()))?;
    anyhow::ensure!(
        serial_continuation == mtp_continuation,
        "serial/MTP multi-turn continuation differs: {serial_continuation:?} vs {mtp_continuation:?}"
    );
    let continuation_tail = *serial_continuation
        .last()
        .expect("non-empty continuation generation");
    let mut expected_mtp_fed = serial_session.fed.clone();
    expected_mtp_fed.push(continuation_tail);
    if crate::core::generate::is_eos(continuation_tail) {
        anyhow::ensure!(
            mtp_session.fed == serial_session.fed || mtp_session.fed == expected_mtp_fed,
            "multi-turn MTP EOS boundary is neither sampled-stop nor accepted-draft commit"
        );
    } else {
        anyhow::ensure!(
            mtp_session.fed == expected_mtp_fed,
            "multi-turn MTP continuation did not retain DS4's committed boundary"
        );
    }
    eprintln!(
        "[glm53-mtp-parity] continuation greedy token matches ({serial_next_token}); multi-turn continuation also matches"
    );
    Ok(())
}

#[test]
fn target_mtp_rejection_replay_restores_serial_cache_real_model() -> anyhow::Result<()> {
    if std::env::var("LISA_GLM53_MTP_REJECTION_PARITY").is_err() {
        eprintln!("LISA_GLM53_MTP_REJECTION_PARITY=1 to run (loads the full GLM checkpoint)");
        return Ok(());
    }
    if !lisa_mlx::device::metal_available() {
        eprintln!("[glm53-mtp-rejection] skipping: no Metal device available");
        return Ok(());
    }

    let dir = artifact_dir();
    let config = ModelConfig::from_json(&dir.join("config.json"))?;
    let mut tower = Tower::load(&dir, config)?;
    let tokenizer = crate::core::tokenizer::Tokenizer::load_file(&dir.join("tokenizer.json"))?;
    let prompt = tokenizer.encode("The capital of France is", false)?;
    let mut serial_session = crate::core::session::Session::new(&mut tower);
    let mut serial_sampler = crate::core::sampler::Sampler::default();
    let mut serial_on_token = |_: u32| Ok(());
    let serial = serial_session.generate(
        &mut tower,
        &prompt,
        3,
        &mut serial_sampler,
        0,
        None,
        &mut serial_on_token,
    )?;
    let mut mtp_session = crate::core::session::Session::new(&mut tower);
    let mut mtp_on_token = |_: u32| Ok(());
    let mtp = mtp_session.generate_mtp(&mut tower, &prompt, 3, 2, &mut mtp_on_token)?;

    eprintln!("[glm53-mtp-rejection] serial={serial:?} target_mtp={mtp:?}");
    anyhow::ensure!(
        serial == vec![12089, 13, 12089] && mtp == serial,
        "expected the real first MTP proposal rejection and serial-equivalent output, got serial={serial:?}, mtp={mtp:?}"
    );
    let final_token = *serial.last().expect("non-empty greedy generation");
    let _ = serial_session.decode_token(&mut tower, final_token)?;
    anyhow::ensure!(
        mtp_session.fed == serial_session.fed,
        "target-MTP and serial sessions differ after completing the serial tail"
    );
    assert_session_cache_layout_equal(&serial_session.caches, &mtp_session.caches)?;
    assert_session_caches_equal(&serial_session.caches, &mtp_session.caches)?;
    eprintln!(
        "[glm53-mtp-rejection] rollback, replay, and committed target caches match serial bit-for-bit"
    );
    if std::env::var_os("LISA_TRACE").is_some() || std::env::var_os("LISA_TRACE_JSON").is_some() {
        lisa_mlx::trace::finish();
    }
    Ok(())
}

#[test]
fn capture_real_two_row_target_mtp_kda_input_for_ds4() -> anyhow::Result<()> {
    let Some(capture_path) = std::env::var_os("LISA_GLM53_KDA_VERIFY_CAPTURE") else {
        eprintln!("LISA_GLM53_KDA_VERIFY_CAPTURE=<path>.npz to capture real MTP verify KDA stages");
        return Ok(());
    };
    if !lisa_mlx::device::metal_available() {
        eprintln!("[glm53-kda-verify-capture] skipping: no Metal device available");
        return Ok(());
    }

    let dir = artifact_dir();
    let config = ModelConfig::from_json(&dir.join("config.json"))?;
    anyhow::ensure!(config.is_kda(1), "GLM layer 1 is not a KDA layer");
    let layer_count = config.num_hidden_layers;
    let mut tower = Tower::load(&dir, config)?;
    let tokenizer = crate::core::tokenizer::Tokenizer::load_file(&dir.join("tokenizer.json"))?;
    let prompt = tokenizer.encode("The capital of France is", false)?;
    let mut caches = tower.new_caches();

    let prefix_hidden = tower.prefill(&prompt, &mut caches)?;
    let first = crate::models::speculate::argmax_id(&tower.head(&prefix_hidden)?)?;
    let first_ids = Array::from_slice(&[first as i32], &[1, 1]);
    let (first_mixed, first_multi) = tower.forward_capture(&first_ids, Some(&mut caches), false)?;
    let carry = crate::models::speculate::argmax_id(&tower.head(&first_mixed)?)?;
    let mut mtp_cache = crate::core::cache::IndexerTape::new();
    let draft_logits = tower.target_mtp_step(
        &mut mtp_cache,
        &first_multi,
        carry,
        prompt.len(),
        prompt.len(),
    )?;
    let draft = crate::models::speculate::argmax_id(&draft_logits)?;
    eprintln!(
        "[glm53-kda-verify-capture] target rows=[{carry},{draft}] at {}",
        prompt.len() + 1
    );

    let capture_layers = std::env::var("LISA_GLM53_VERIFY_CAPTURE_LAYERS")
        .ok()
        .map(|spec| {
            spec.split(',')
                .filter_map(|item| item.trim().parse::<usize>().ok())
                .collect::<Vec<_>>()
        })
        .unwrap_or_else(|| vec![1]);
    anyhow::ensure!(
        !capture_layers.is_empty() && capture_layers.iter().all(|&layer| layer < layer_count),
        "verify capture layers must be valid indices below {}",
        layer_count
    );
    tower.capture = Some(crate::core::capture::Capture {
        layers: capture_layers.clone(),
        state_only: false,
        kda_prefill_only: false,
        // This differential is against the production MTP path. The default
        // capture mode can select a different MoE down-projection dispatch.
        preserve_dispatch: true,
        entries: std::sync::Mutex::new(std::collections::HashMap::new()),
        shapes: std::sync::Mutex::new(std::collections::HashMap::new()),
        flushed: std::sync::atomic::AtomicBool::new(false),
        ..Default::default()
    });
    let verify_ids = Array::from_slice(&[carry as i32, draft as i32], &[1, 2]);
    let _ = tower.forward_capture(&verify_ids, Some(&mut caches), false)?;
    tower.capture.as_ref().expect("capture installed").flush(
        std::path::Path::new(&capture_path)
            .to_str()
            .ok_or_else(|| anyhow::anyhow!("capture path is not UTF-8"))?,
    )?;
    if std::env::var_os("LISA_TRACE").is_some() || std::env::var_os("LISA_TRACE_JSON").is_some() {
        lisa_mlx::trace::finish();
    }
    Ok(())
}

#[test]
fn capture_real_accepted_target_mtp_kda_input_for_ds4() -> anyhow::Result<()> {
    let Some(capture_path) = std::env::var_os("LISA_GLM53_KDA_ACCEPT_VERIFY_CAPTURE") else {
        eprintln!(
            "LISA_GLM53_KDA_ACCEPT_VERIFY_CAPTURE=<path>.npz to capture an accepted real MTP verify"
        );
        return Ok(());
    };
    if !lisa_mlx::device::metal_available() {
        eprintln!("[glm53-kda-accept-capture] skipping: no Metal device available");
        return Ok(());
    }

    let dir = artifact_dir();
    let config = ModelConfig::from_json(&dir.join("config.json"))?;
    let kda_layers: Vec<_> = (0..config.num_hidden_layers)
        .filter(|&layer| config.is_kda(layer))
        .collect();
    anyhow::ensure!(!kda_layers.is_empty(), "GLM checkpoint has no KDA layers");
    // The accepted two-row verify can capture either all trunk layers or a
    // selected subset. For DSA-only DS4 replay, selecting layers 3,7,...,43
    // avoids retaining the unrelated KDA stage arrays in the NPZ.
    let capture_layers = verify_capture_layers(
        std::env::var("LISA_GLM53_VERIFY_CAPTURE_LAYERS")
            .ok()
            .as_deref(),
        config.num_hidden_layers,
    )?;
    let mut tower = Tower::load(&dir, config)?;
    let tokenizer = crate::core::tokenizer::Tokenizer::load_file(&dir.join("tokenizer.json"))?;
    let prompt = tokenizer.encode("The capital of France is", false)?;
    let mut caches = tower.new_caches();

    let prompt_hidden = tower.prefill(&prompt, &mut caches)?;
    let first = crate::models::speculate::argmax_id(&tower.head(&prompt_hidden)?)?;
    let first_ids = Array::from_slice(&[first as i32], &[1, 1]);
    let (first_mixed, first_multi) = tower.forward_capture(&first_ids, Some(&mut caches), false)?;
    let mut carry = crate::models::speculate::argmax_id(&tower.head(&first_mixed)?)?;
    let min_position = prompt.len();
    let mut mtp_cache = crate::core::cache::IndexerTape::new();
    let mut draft_logits = tower.target_mtp_step(
        &mut mtp_cache,
        &first_multi,
        carry,
        min_position,
        min_position,
    )?;
    let mut draft = crate::models::speculate::argmax_id(&draft_logits)?;
    let mut verify_position = min_position + 1;
    let transaction_capture_prefix = std::env::var_os("LISA_GLM53_MTP_TRANSACTION_CAPTURE_PREFIX");
    if let Some(prefix) = transaction_capture_prefix.as_deref() {
        save_glm53_cache_transaction(
            prefix,
            0,
            carry,
            draft,
            &capture_layers,
            &caches,
            &mtp_cache,
        )?;
    }

    std::fs::create_dir_all(
        std::path::Path::new(&capture_path)
            .parent()
            .unwrap_or_else(|| std::path::Path::new(".")),
    )?;
    tower.capture = Some(crate::core::capture::Capture {
        layers: capture_layers.clone(),
        state_only: false,
        kda_prefill_only: false,
        // Preserve the production MoE dispatch while capturing this verify.
        preserve_dispatch: true,
        entries: std::sync::Mutex::new(std::collections::HashMap::new()),
        shapes: std::sync::Mutex::new(std::collections::HashMap::new()),
        flushed: std::sync::atomic::AtomicBool::new(false),
        ..Default::default()
    });

    let mut accepted = false;
    for attempt in 0..32 {
        let before: Vec<_> = caches.iter().map(|cache| cache.snapshot_prefix()).collect();
        let verify_ids = Array::from_slice(&[carry as i32, draft as i32], &[1, 2]);
        let (verify_mixed, verify_multi) =
            tower.forward_capture(&verify_ids, Some(&mut caches), true)?;
        let logits = tower.head(&verify_mixed)?;
        let row0_logits = logits.index((.., 0, ..)).contiguous()?;
        let target = crate::models::speculate::argmax_id(&row0_logits)?;
        if target == draft {
            let next_target =
                crate::models::speculate::argmax_id(&logits.index((.., 1, ..)).contiguous()?)?;
            let first_verify_multi = narrow(&verify_multi, 1, 0, 1)?;
            let second_verify_multi = narrow(&verify_multi, 1, 1, 1)?;
            let _ = tower.target_mtp_step(
                &mut mtp_cache,
                &first_verify_multi,
                draft,
                verify_position,
                min_position,
            )?;
            let _ = tower.target_mtp_step(
                &mut mtp_cache,
                &second_verify_multi,
                next_target,
                verify_position + 1,
                min_position,
            )?;
            if let Some(prefix) = transaction_capture_prefix.as_deref() {
                save_glm53_cache_transaction(
                    prefix,
                    attempt + 1,
                    carry,
                    draft,
                    &capture_layers,
                    &caches,
                    &mtp_cache,
                )?;
            }
            let cache_capture = tower.capture.as_ref().expect("accepted capture installed");
            capture_glm53_cache_state(cache_capture, &caches, &mtp_cache);
            if let Some(path) = std::env::var_os("LISA_GLM53_MTP_ACCEPTED_LOGITS_CAPTURE") {
                let flat = logits
                    .as_dtype(Dtype::Float32)?
                    .contiguous()?
                    .reshape(&[logits.size() as i32])?
                    .to_vec1::<f32>()?;
                let mut bytes = Vec::with_capacity(flat.len() * std::mem::size_of::<f32>());
                for value in flat {
                    bytes.extend_from_slice(&value.to_le_bytes());
                }
                std::fs::write(&path, bytes)?;
                eprintln!(
                    "[glm53-kda-accept-capture] saved accepted verify logits rows=[carry,draft] to {}",
                    std::path::Path::new(&path).display()
                );
            }
            accepted = true;
            eprintln!(
                "[glm53-kda-accept-capture] accepted target rows=[{carry},{draft}] at position {verify_position} after {attempt} rejected proposals"
            );
            break;
        }

        for (cache, state) in caches.iter_mut().zip(&before) {
            cache.restore_prefix(state);
        }
        let saved_capture = tower.capture.take();
        let carry_ids = Array::from_slice(&[carry as i32], &[1, 1]);
        let (carry_mixed, carry_multi) =
            tower.forward_capture(&carry_ids, Some(&mut caches), false)?;
        tower.capture = saved_capture;
        carry = crate::models::speculate::argmax_id(&tower.head(&carry_mixed)?)?;
        draft_logits = tower.target_mtp_step(
            &mut mtp_cache,
            &carry_multi,
            carry,
            verify_position,
            min_position,
        )?;
        draft = crate::models::speculate::argmax_id(&draft_logits)?;
        verify_position += 1;
        if let Some(prefix) = transaction_capture_prefix.as_deref() {
            save_glm53_cache_transaction(
                prefix,
                attempt + 1,
                carry,
                draft,
                &capture_layers,
                &caches,
                &mtp_cache,
            )?;
        }
    }
    anyhow::ensure!(accepted, "no accepted target-MTP proposal in 32 attempts");
    let capture = tower.capture.as_ref().expect("accepted capture installed");
    capture.store(
        "glm53.verify_meta",
        &Array::from_slice(&[verify_position as f32, carry as f32, draft as f32], &[3]),
    );
    capture.flush(
        std::path::Path::new(&capture_path)
            .to_str()
            .ok_or_else(|| anyhow::anyhow!("capture path is not UTF-8"))?,
    )?;
    if std::env::var_os("LISA_TRACE").is_some() || std::env::var_os("LISA_TRACE_JSON").is_some() {
        lisa_mlx::trace::finish();
    }
    Ok(())
}

#[test]
fn capture_real_target_mtp_final_session_cache_for_ds4() -> anyhow::Result<()> {
    let Some(prefix) = std::env::var_os("LISA_GLM53_MTP_TRANSACTION_CAPTURE_PREFIX") else {
        eprintln!(
            "LISA_GLM53_MTP_TRANSACTION_CAPTURE_PREFIX=<path> to capture final target-MTP session state"
        );
        return Ok(());
    };
    if !lisa_mlx::device::metal_available() {
        eprintln!("[glm53-mtp-final-capture] skipping: no Metal device available");
        return Ok(());
    }

    let dir = artifact_dir();
    let config = ModelConfig::from_json(&dir.join("config.json"))?;
    let layer_count = config.num_hidden_layers;
    let eos_token_ids = config.eos_token_ids.clone();
    let mut tower = if std::env::var_os("LISA_GLM53_MTP_MAPPED_EXPERTS_CAPTURE").is_some() {
        Tower::load_mapped_experts_without_residency(&dir, config)?
    } else if std::env::var_os("LISA_GLM53_MTP_NO_EXPERT_RESIDENCY_CAPTURE").is_some() {
        Tower::load_without_expert_residency(&dir, config)?
    } else {
        Tower::load(&dir, config)?
    };
    let tokenizer = crate::core::tokenizer::Tokenizer::load_file(&dir.join("tokenizer.json"))?;
    let prompt_text = std::env::var("LISA_GLM53_MTP_CAPTURE_PROMPT").unwrap_or_else(|_| {
        "Write the numbers from 1 to 1000 in order, separated by commas. Continue listing them without explanation.".to_owned()
    });
    let prompt = tokenizer.encode(&prompt_text, false)?;
    let token_limit = std::env::var("LISA_GLM53_MTP_CAPTURE_TOKENS")
        .ok()
        .map(|value| value.parse::<usize>())
        .transpose()?
        .unwrap_or(32);
    anyhow::ensure!(token_limit > 0, "MTP capture token limit must be positive");
    if let Some(serial_capture_path) = std::env::var_os("LISA_GLM53_MTP_SERIAL_CACHE_CAPTURE") {
        let mut serial_session = crate::core::session::Session::new(&mut tower);
        let mut serial_sampler = crate::core::sampler::Sampler {
            temperature: 0.0,
            ..Default::default()
        };
        let mut serial_on_token = |_| Ok(());
        let serial_tokens = serial_session.generate(
            &mut tower,
            &prompt,
            4,
            &mut serial_sampler,
            0,
            None,
            &mut serial_on_token,
        )?;
        anyhow::ensure!(serial_tokens.len() == 4, "serial cache probe stopped early");
        let _ = serial_session.decode_token(&mut tower, serial_tokens[3])?;
        anyhow::ensure!(
            serial_session.fed.len() == prompt.len() + serial_tokens.len(),
            "serial cache probe did not feed its final emitted token"
        );
        let serial_capture = crate::core::capture::Capture {
            layers: (0..layer_count).collect(),
            state_only: false,
            kda_prefill_only: false,
            preserve_dispatch: true,
            entries: std::sync::Mutex::new(std::collections::HashMap::new()),
            shapes: std::sync::Mutex::new(std::collections::HashMap::new()),
            flushed: std::sync::atomic::AtomicBool::new(false),
            ..Default::default()
        };
        capture_glm53_cache_state(
            &serial_capture,
            &serial_session.caches,
            serial_session.glm_mtp_cache_for_test(),
        );
        serial_capture.flush(
            std::path::Path::new(&serial_capture_path)
                .to_str()
                .ok_or_else(|| anyhow::anyhow!("serial cache capture path is not UTF-8"))?,
        )?;
        eprintln!(
            "[glm53-mtp-serial-cache] captured fed={} tokens={serial_tokens:?}",
            serial_session.fed.len()
        );
    }
    let prefix_text = std::path::Path::new(&prefix)
        .to_str()
        .ok_or_else(|| anyhow::anyhow!("final session capture prefix is not UTF-8"))?
        .to_owned();
    let state_capture_counts = std::env::var("LISA_GLM53_MTP_STATE_CAPTURE_COUNTS")
        .ok()
        .map(|value| {
            value
                .split(',')
                .map(|item| item.parse::<usize>())
                .collect::<Result<Vec<_>, _>>()
        })
        .transpose()?;
    let prefix_parent = std::path::Path::new(&prefix_text)
        .parent()
        .unwrap_or_else(|| std::path::Path::new("."));
    std::fs::create_dir_all(prefix_parent)?;
    std::fs::write(format!("{prefix_text}.prompt.txt"), &prompt_text)?;
    let mut session = crate::core::session::Session::new(&mut tower);
    let exact_sampling = std::env::var_os("LISA_GLM53_MTP_EXACT_CAPTURE").is_some();
    let sampler_seed = 0x9E37_79B9_7F4A_7C15;
    let mut sampler = crate::core::sampler::Sampler {
        temperature: 1.0,
        top_p: 0.95,
        mtp_exact_sampling: exact_sampling,
        seed: sampler_seed,
        ..Default::default()
    };
    let boundary_prefix = prefix_text.clone();
    session.set_glm_mtp_state_capture_for_test(move |token_count, caches, mtp_cache| {
        if state_capture_counts
            .as_ref()
            .is_some_and(|counts| !counts.contains(&token_count))
        {
            return Ok(());
        }
        let capture = crate::core::capture::Capture {
            layers: (0..layer_count).collect(),
            state_only: false,
            kda_prefill_only: false,
            preserve_dispatch: true,
            entries: std::sync::Mutex::new(std::collections::HashMap::new()),
            shapes: std::sync::Mutex::new(std::collections::HashMap::new()),
            flushed: std::sync::atomic::AtomicBool::new(false),
            ..Default::default()
        };
        capture_glm53_cache_state(&capture, caches, mtp_cache);
        capture.store(
            "glm53.mtp_boundary_meta",
            &Array::from_slice(&[token_count as f32], &[1]),
        );
        let path = format!("{boundary_prefix}.cache-fed-{token_count:05}.npz");
        capture.flush(&path)?;
        Ok(())
    });
    let mut emitted = Vec::with_capacity(token_limit);
    let mut on_token = |token| {
        emitted.push(token);
        Ok(())
    };
    let generated = session.generate_mtp_depth(
        &mut tower,
        &prompt,
        token_limit,
        crate::core::session::MtpDepth::Fixed(2),
        None,
        exact_sampling.then_some(&mut sampler),
        &mut on_token,
    )?;
    anyhow::ensure!(
        !generated.is_empty()
            && (generated.len() == token_limit
                || generated
                    .last()
                    .is_some_and(|token| eos_token_ids.contains(&i64::from(*token)))),
        "expected up to {token_limit} generated tokens, optionally ending in EOS; got {}",
        generated.len()
    );
    anyhow::ensure!(
        generated == emitted,
        "emitted tokens differ from the generated result"
    );

    let mut captured_logit_rows = 0usize;
    for (capture_index, (token_count, flat)) in
        session.glm_mtp_target_logits_for_test().iter().enumerate()
    {
        anyhow::ensure!(
            flat.len() == 154_880,
            "expected one GLM target logit vector, got {} values",
            flat.len()
        );
        let mut bytes = Vec::with_capacity(flat.len() * std::mem::size_of::<f32>());
        for value in flat {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        let logits_path = format!("{prefix_text}.target-logits-fed-{token_count:05}.f32");
        std::fs::write(&logits_path, bytes)?;
        let occurrence_path = format!(
            "{prefix_text}.target-logits-capture-{capture_index:03}-fed-{token_count:05}.f32"
        );
        std::fs::copy(&logits_path, occurrence_path)?;
        captured_logit_rows += 1;
    }

    let capture = crate::core::capture::Capture {
        layers: (0..layer_count).collect(),
        state_only: false,
        kda_prefill_only: false,
        preserve_dispatch: true,
        entries: std::sync::Mutex::new(std::collections::HashMap::new()),
        shapes: std::sync::Mutex::new(std::collections::HashMap::new()),
        flushed: std::sync::atomic::AtomicBool::new(false),
        ..Default::default()
    };
    capture_glm53_cache_state(&capture, &session.caches, session.glm_mtp_cache_for_test());
    capture.store(
        "glm53.final_session_meta",
        &Array::from_slice(
            &[
                prompt.len() as f32,
                generated.len() as f32,
                session.fed.len() as f32,
            ],
            &[3],
        ),
    );
    let cache_path = format!("{prefix_text}.final-session.npz");
    capture.flush(&cache_path)?;
    let tokens_path = format!("{prefix_text}.final-session.tokens.txt");
    let token_text = generated
        .iter()
        .map(u32::to_string)
        .collect::<Vec<_>>()
        .join(",");
    std::fs::write(&tokens_path, token_text)?;
    std::fs::write(
        format!("{prefix_text}.sampling.txt"),
        if exact_sampling {
            format!(
                "mode=ds4-exact-mtp\ntokens={token_limit}\ntemperature=1.0\ntop_k=0\ntop_p=0.95\nmin_p=0.0\nseed={sampler_seed:#018x}\nrng_final={:#018x}\n",
                sampler.seed
            )
        } else {
            "mode=greedy\n".to_owned()
        },
    )?;
    eprintln!(
        "[glm53-mtp-final-capture] saved {} generated tokens (exact_sampling={}), {} target-logit boundaries, and trunk/private cache state to {}",
        generated.len(),
        exact_sampling,
        captured_logit_rows,
        cache_path
    );
    if std::env::var_os("LISA_TRACE").is_some() || std::env::var_os("LISA_TRACE_JSON").is_some() {
        lisa_mlx::trace::finish();
    }
    Ok(())
}

#[test]
fn verify_capture_layer_filter_defaults_and_validates() -> anyhow::Result<()> {
    anyhow::ensure!(verify_capture_layers(None, 4)? == vec![0, 1, 2, 3]);
    anyhow::ensure!(
        verify_capture_layers(Some("1, 3"), 4)? == vec![1, 3],
        "selected verify capture layers were not preserved"
    );
    anyhow::ensure!(
        verify_capture_layers(Some("3,7,11,15,19,23,27,31,35,39,43"), 45)?
            == vec![3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43],
        "standard DSA layer filter was not preserved"
    );
    anyhow::ensure!(verify_capture_layers(Some(""), 4).is_err());
    anyhow::ensure!(verify_capture_layers(Some("1,nope"), 4).is_err());
    anyhow::ensure!(verify_capture_layers(Some("4"), 4).is_err());
    Ok(())
}

#[test]
fn smoke_logits_and_generation() -> anyhow::Result<()> {
    if std::env::var("LISA_GLM53_SMOKE").is_err() {
        eprintln!("LISA_GLM53_SMOKE=1 to run (loads 102 GB)");
        return Ok(());
    }
    let dir = artifact_dir();
    let config = ModelConfig::from_json(&dir.join("config.json")).expect("config");
    let mut tower = Tower::load(&dir, config).expect("load");

    // Raw prompt (no chat template) for a clean comparison with the
    // Pipenetwork runtime.
    let prompt = "The capital of France is";
    let tok = crate::core::tokenizer::Tokenizer::load_file(&dir.join("tokenizer.json"))
        .expect("tokenizer");
    let ids = tok.encode(prompt, false).expect("encode");
    println!("[glm53] ids: {:?}", ids);
    for &t in &ids {
        println!("[glm53]   {} {:?}", t, tok.decode(&[t]).unwrap_or_default());
    }
    let i32s: Vec<i32> = ids.iter().map(|&t| t as i32).collect();
    let arr = Array::from_slice(&i32s, &[1i32, i32s.len() as i32]);

    let mut caches = tower.new_caches();
    let (mixed, _) = tower.forward(&arr, Some(&mut caches)).expect("forward");
    let last = narrow(&mixed, 1, mixed.dim(1) as usize - 1, 1)?.reshape(&[1, mixed.dim(2)])?;
    let logits = tower.head(&last).expect("head");
    let neg = -&logits;
    let top = lisa_mlx::ops::argpartition_axis(&neg, 7, -1).expect("argpartition");
    let top = narrow(&top, 1, 0, 8)?;
    let vals = logits.take_along_axis(&top, -1).expect("take");
    let _ = vals.eval();
    let toks: Vec<u32> = top.as_slice::<u32>().to_vec();
    let vf: Vec<f32> = vals
        .as_dtype(Dtype::Float32)
        .expect("cast")
        .as_slice::<f32>()
        .to_vec();
    println!("[glm53] top-8 first-token logits for {prompt:?}:");
    for (t, v) in toks.iter().zip(vf.iter()) {
        let piece = tok.decode(&[*t]).unwrap_or_default();
        println!("  {:>7} {:>8.3} {piece:?}", t, v);
    }

    // Bounded loader and real-weight prefill check. This lets us validate a
    // model-scale forward without also running the repeated re-prefill loop
    // below, which is intended for numerical diagnosis rather than load QA.
    if std::env::var_os("LISA_GLM53_PREFILL_ONLY").is_some() {
        anyhow::ensure!(vf.iter().all(|v| v.is_finite()), "non-finite GLM logits");
        return Ok(());
    }

    // Greedy generation of 8 more tokens.
    let mut gen_ids: Vec<u32> = ids.clone();
    let mut first = toks[0];

    // INCREMENTAL-DECODE check: carry the caches from the prefill and run
    // t=1 steps; each step's logits must match the re-prefill reference.
    {
        let mut caches_inc = tower.new_caches();
        // Prime the incremental cache with the prefix WITHOUT the generated
        // token (it is fed again at step 0) — otherwise the token would be
        // counted twice and the comparison against the re-prefill is biased.
        {
            let pre: Vec<i32> = ids.iter().map(|&t| t as i32).collect();
            let arr_pre = Array::from_slice(&pre, &[1i32, pre.len() as i32]);
            let _ = tower
                .forward(&arr_pre, Some(&mut caches_inc))
                .expect("inc prefill");
        }
        // Run the 4 incremental steps FIRST, storing logits; the re-prefill
        // references are computed AFTER the block — interleaving fresh caches
        // with the incremental forwards perturbs the buffer pool (readback
        // race) and corrupts the measurement.
        let mut inc_logits: Vec<Vec<f32>> = Vec::new();
        let mut next_toks: Vec<u32> = Vec::new();
        let mut cur = first;
        for _ in 0..4 {
            let arr1 = Array::from_slice(&[cur as i32], &[1i32, 1]);
            let (mixed_inc, _) = tower.forward(&arr1, Some(&mut caches_inc)).expect("inc");
            let last = narrow(&mixed_inc, 1, mixed_inc.dim(1) as usize - 1, 1)?
                .reshape(&[1, mixed_inc.dim(2)])?;
            let logits_inc = tower
                .head(&last)
                .expect("head")
                .as_dtype(Dtype::Float32)
                .expect("cast")
                .to_vec1::<f32>()
                .unwrap();
            let am = logits_inc
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i)
                .unwrap_or(0);
            cur = u32::try_from(am).unwrap_or(0);
            next_toks.push(cur);
            inc_logits.push(logits_inc);
        }
        // References: re-prefill over the prefix + the tokens fed so far.
        let mut fed: Vec<u32> = vec![first];
        for step in 0..4 {
            let mut full: Vec<i32> = ids.iter().map(|&t| t as i32).collect();
            full.extend(fed.iter().map(|&t| t as i32));
            let arr_full = Array::from_slice(&full, &[1i32, full.len() as i32]);
            let mut caches_full = tower.new_caches();
            let (mixed_ref, _) = tower
                .forward(&arr_full, Some(&mut caches_full))
                .expect("ref");
            let last_ref = narrow(&mixed_ref, 1, mixed_ref.dim(1) as usize - 1, 1)?
                .reshape(&[1, mixed_ref.dim(2)])?;
            let logits_ref = tower
                .head(&last_ref)
                .expect("head")
                .as_dtype(Dtype::Float32)
                .expect("cast")
                .to_vec1::<f32>()
                .unwrap();
            let li = &inc_logits[step];
            let ni: f32 = li.iter().map(|a| a * a).sum::<f32>().sqrt();
            let nr: f32 = logits_ref.iter().map(|a| a * a).sum::<f32>().sqrt();
            let any_nan_i = li.iter().any(|a| a.is_nan());
            let any_nan_r = logits_ref.iter().any(|a| a.is_nan());
            let cos: f64 = li.iter().zip(&logits_ref).map(|(a, b)| a * b).sum::<f32>() as f64
                / (ni as f64 * nr as f64 + 1e-9);
            anyhow::ensure!(
                !any_nan_i && !any_nan_r,
                "GLM incremental/re-prefill logits contain NaN at step {step}"
            );
            anyhow::ensure!(
                li.iter()
                    .zip(&logits_ref)
                    .all(|(a, b)| a.is_finite() && b.is_finite()),
                "GLM incremental/re-prefill logits contain non-finite values at step {step}"
            );
            let argmax = |xs: &[f32]| {
                xs.iter()
                    .enumerate()
                    .max_by(|a, b| a.1.total_cmp(b.1))
                    .map(|(index, _)| index)
                    .unwrap_or(0)
            };
            anyhow::ensure!(
                argmax(li) == argmax(&logits_ref),
                "GLM incremental/re-prefill greedy token mismatch at step {step}: {} vs {}",
                argmax(li),
                argmax(&logits_ref)
            );
            anyhow::ensure!(
                cos >= 0.995,
                "GLM incremental/re-prefill logits cosine is {cos:.6} at step {step}"
            );
            println!(
                "[glm53] inc-step {step}: logits cos {cos:.6} |ni| {ni:.4} |nr| {nr:.4} nan_i {any_nan_i} nan_r {any_nan_r}"
            );
            fed.push(next_toks[step]);
        }
        println!("[glm53] incremental decode matches re-prefill on four consecutive steps");
    }
    gen_ids.push(first);
    for _ in 0..7 {
        // Re-prefill each step to produce an independent greedy continuation.
        let full: Vec<i32> = gen_ids.iter().map(|&t| t as i32).collect();
        let arr_full = Array::from_slice(&full, &[1i32, full.len() as i32]);
        let mut caches_full = tower.new_caches();
        let (mixed, _) = tower
            .forward(&arr_full, Some(&mut caches_full))
            .expect("fwd");
        let last = narrow(&mixed, 1, mixed.dim(1) as usize - 1, 1)?.reshape(&[1, mixed.dim(2)])?;
        let logits = tower.head(&last).expect("head");
        let am = lisa_mlx::ops::indexing::argmax_axis(&logits, -1, false)?;
        first = u32::try_from(am.item_cast::<i32>()).unwrap_or(0);
        gen_ids.push(first);
    }
    let text: String = gen_ids[ids.len()..]
        .iter()
        .map(|&t| tok.decode(&[t]).unwrap_or_default())
        .collect();
    println!("[glm53] generated: {text:?}");
    // The single-token continuation of "The capital of France is" must not
    // be a control/unknown token; the numerics assertion comes with the
    // golden harness (oracle).
    assert!(!text.trim().is_empty(), "empty generation");
    let _ = &mut first;
    Ok(())
}
