//! In-process serial/DSpine comparison with identical target weights and ABBA order.
use anyhow::{Context, Result, ensure};
use clap::Parser;
use lisa_bench::{
    checkpoint,
    core::{
        generate,
        session::{MtpDepth, Session},
        tokenizer::{self, Tokenizer},
    },
    language::LanguageModel,
    loading::{self, Loaded},
};
use serde_json::json;
use std::{path::PathBuf, process::Command, time::Instant};

#[derive(Parser)]
struct Args {
    #[arg(long, default_value = "agnosticeng/Qwen3.8-27B-4bit")]
    model: String,
    #[arg(long, default_value = "tencent/Qwen3.8-27B-DSpine-b16")]
    draft: String,
    #[arg(
        long,
        default_value = "Explain how a hash table works, including collisions, resizing, and practical tradeoffs. Give a concrete example."
    )]
    prompt: String,
    #[arg(long, default_value_t = 128)]
    tokens: usize,
    #[arg(long, default_value_t = 2)]
    repeats: usize,
    #[arg(
        long,
        help = "Run one serial/DSpine pair for diagnostics; excluded from scored repeats"
    )]
    diagnostic: bool,
    #[arg(long, requires = "audit_rounds")]
    audit_reference: Option<PathBuf>,
    #[arg(long, requires = "audit_reference")]
    audit_rounds: Option<PathBuf>,
    /// Verify width including the carried target token; drafting keeps its configured block.
    #[arg(long, value_parser = clap::value_parser!(u8).range(2..=16))]
    verify_width: Option<u8>,
    /// Profile BS1 SPS widths in forward/reverse order with serial controls.
    #[arg(long, value_delimiter = ',', conflicts_with = "verify_width", value_parser = clap::value_parser!(u8).range(2..=16))]
    verify_widths: Vec<u8>,
    /// Compare the request-local controller with a fixed measured width.
    #[arg(long, requires = "verify_width", conflicts_with_all = ["verify_widths", "diagnostic", "save_policy"])]
    compare_adaptive: bool,
    /// Save the fastest measured width for this device/checkpoint pair (Lisa UI loads it).
    #[arg(long, requires = "verify_widths", conflicts_with = "diagnostic")]
    save_policy: bool,
    #[arg(long)]
    out: PathBuf,
}

fn safety() -> Result<()> {
    for (name, args) in [("sysctl", vec!["hw.memsize"]), ("memory_pressure", vec![])] {
        let result = Command::new(name).args(args).output()?;
        ensure!(result.status.success(), "{name} failed");
        eprintln!("{}", String::from_utf8_lossy(&result.stdout));
    }
    let app = Command::new("pgrep").args(["-x", "lisa-ui"]).output()?;
    ensure!(
        app.status.code() == Some(1),
        "stop Lisa before benchmarking"
    );
    Ok(())
}

fn run(
    model: &mut dyn LanguageModel<lisa_bench::metal_language::MetalLanguageRuntime>,
    prompt: &[u32],
    count: usize,
    draft: bool,
    verify_width: Option<u8>,
) -> Result<serde_json::Value> {
    safety()?;
    model.drafter_reset();
    model.set_context_tails(Vec::new());
    let mut session = Session::new(model);
    let stream = lisa_metal::Stream::thread_local_or_default();
    stream.runtime().synchronize()?;
    let _eos = generate::ignore_eos_scope();
    let mut first = None;
    let start = Instant::now();
    let tokens = session.generate_mtp_depth(
        model,
        prompt,
        count,
        if draft {
            verify_width.map_or(MtpDepth::Auto, |w| MtpDepth::Fixed(w as usize - 1))
        } else {
            MtpDepth::Fixed(0)
        },
        None,
        None,
        &mut |_| {
            first.get_or_insert_with(Instant::now);
            Ok(())
        },
    )?;
    stream.runtime().synchronize()?;
    let end = Instant::now();
    ensure!(
        tokens.len() == count,
        "expected {count} tokens, got {}",
        tokens.len()
    );
    let first = first.context("no tokens emitted")?;
    let decode_s = end.duration_since(first).as_secs_f64();
    let result = json!({
        "mode": if draft { "dspine" } else { "serial" },
        "verify_width": verify_width,
        "ttft_s": first.duration_since(start).as_secs_f64(),
        "decode_s": decode_s,
        "decode_tokens": count - 1,
        "decode_tps": (count - 1) as f64 / decode_s,
        "total_s": end.duration_since(start).as_secs_f64(),
        "active_bytes_after_decode": lisa_metal::memory::active_memory()?,
        "tokens": tokens,
    });
    eprintln!(
        "{}: TTFT {:.3}s; decode {:.2} tok/s; total {:.3}s",
        result["mode"],
        result["ttft_s"].as_f64().unwrap(),
        result["decode_tps"].as_f64().unwrap(),
        result["total_s"].as_f64().unwrap()
    );
    drop(session);
    stream.runtime().synchronize()?;
    lisa_metal::memory::clear_cache();
    Ok(result)
}

fn audit(
    model: &mut dyn LanguageModel<lisa_bench::metal_language::MetalLanguageRuntime>,
    reference: &std::path::Path,
    rounds: &std::path::Path,
) -> Result<serde_json::Value> {
    use lisa_metal::{Array, Dtype, ops::indexing::IndexOp};
    use lisa_bench::core::cache::LayerCache;
    safety()?;
    let reference: serde_json::Value = serde_json::from_slice(&std::fs::read(reference)?)?;
    let runs = reference["runs"].as_array().context("missing runs")?;
    let runs = if runs.is_empty() {
        reference["warmups"]
            .as_array()
            .context("missing diagnostic warmups")?
    } else {
        runs
    };
    let ids = |value: &serde_json::Value| -> Vec<u32> {
        value
            .as_array()
            .unwrap()
            .iter()
            .map(|id| id.as_u64().unwrap() as u32)
            .collect()
    };
    let serial = ids(&runs.iter().find(|run| run["mode"] == "serial").unwrap()["tokens"]);
    let draft_run = runs
        .iter()
        .find(|run| run["mode"] == "dspine")
        .context("missing draft run")?;
    let draft = ids(&draft_run["tokens"]);
    let prompt = ids(&reference["prompt_tokens"]);
    let mismatch = serial
        .iter()
        .zip(&draft)
        .position(|(a, b)| a != b)
        .context("no token mismatch")?;
    let logs = std::fs::read_to_string(rounds)?;
    let mut parsed = Vec::new();
    for line in logs
        .lines()
        .filter(|line| line.starts_with("[dspine.round]"))
    {
        let number = |key: &str| -> usize {
            line.split(key)
                .nth(1)
                .unwrap()
                .split_whitespace()
                .next()
                .unwrap()
                .parse()
                .unwrap()
        };
        let proposals: Vec<u32> = serde_json::from_str(
            line.split("proposals=")
                .nth(1)
                .unwrap()
                .split(" target=")
                .next()
                .unwrap(),
        )?;
        let capture = line
            .split(" capture=")
            .nth(1)
            .map(|value| value.split_whitespace().next().unwrap().parse::<bool>())
            .transpose()?;
        ensure!(
            capture.is_some() || !draft_run["verify_width"].is_null(),
            "Auto replay requires an explicit capture flag in the round log"
        );
        parsed.push((
            number("start="),
            number("emitted="),
            proposals,
            capture.unwrap_or(true),
        ));
    }
    let selected = parsed
        .iter()
        .find(|(start, emitted, _, _)| *start <= mismatch && mismatch < start + emitted)
        .context("missing mismatching round")?;
    let capture = |logits: Array| -> Result<Vec<f32>> {
        Ok(logits
            .as_dtype(Dtype::Float32)?
            .reshape(&[-1])?
            .to_vec1::<f32>()?)
    };
    let mut serial_session = Session::new(model);
    let mut logits = serial_session.feed(model, &prompt)?;
    for &token in serial.iter().take(mismatch) {
        logits = serial_session.decode_token(model, token)?;
    }
    let serial_logits = capture(logits)?;
    drop(serial_session);

    // Replay the same prefix with serial target arithmetic, then change only
    // the final forward's row count. This isolates immediate wide arithmetic.
    let mut same_prefix = Session::new(model);
    same_prefix.feed(model, &prompt)?;
    for &token in serial.iter().take(selected.0 - 1) {
        same_prefix.decode_token(model, token)?;
    }
    let mut verify = vec![draft[selected.0 - 1] as i32];
    verify.extend(selected.2.iter().map(|&x| x as i32));
    let rows = Array::from_slice(&verify, &[1, verify.len() as i32]);
    let (hidden, _) = model.forward_capture(&rows, Some(&mut same_prefix.caches), selected.3)?;
    let wide_serial_prefix = capture(model.head(&hidden)?.index((
        ..,
        (mismatch - selected.0) as i32,
        ..,
    )))?;
    drop(same_prefix);

    // Reproduce all earlier accepted/rejected verify transactions without the
    // drafter, using its recorded inputs and commit counts exactly.
    let mut wide_session = Session::new(model);
    wide_session.feed(model, &prompt)?;
    let mut wide_logits = None;
    for (start, emitted, proposals, capture_state) in parsed {
        let mut input = vec![draft[start - 1] as i32];
        input.extend(proposals.iter().map(|&x| x as i32));
        let (hidden, _) = model.forward_capture(
            &Array::from_slice(&input, &[1, input.len() as i32]),
            Some(&mut wide_session.caches),
            capture_state,
        )?;
        if start <= mismatch && mismatch < start + emitted {
            wide_logits = Some(capture(model.head(&hidden)?.index((
                ..,
                (mismatch - start) as i32,
                ..,
            )))?);
            break;
        }
        for cache in &mut wide_session.caches {
            match cache {
                LayerCache::Full(full) => full.trim(input.len() - emitted),
                LayerCache::Linear(linear) if capture_state => linear.rollback_to(&model.runtime().context, emitted, 4)?,
                LayerCache::Linear(_) => {}
                LayerCache::Native(_) => anyhow::bail!("unexpected native cache"),
            }
        }
    }
    let wide_logits = wide_logits.context("no wide logits")?;
    let argmax = |values: &[f32]| {
        values
            .iter()
            .enumerate()
            .max_by(|(ia, a), (ib, b)| a.total_cmp(b).then(ib.cmp(ia)))
            .map(|(index, _)| index as u32)
            .unwrap()
    };
    ensure!(
        argmax(&serial_logits) == serial[mismatch],
        "serial audit did not reproduce the recorded token"
    );
    ensure!(
        argmax(&wide_logits) == draft[mismatch],
        "transaction replay did not reproduce the recorded draft token"
    );
    let summarize = |v: &[f32]| {
        let mut order: Vec<_> = v.iter().copied().enumerate().collect();
        order.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        json!({"top8": &order[..8], "margin": order[0].1 - order[1].1,
            "serial_token_logit": v[serial[mismatch] as usize], "draft_token_logit": v[draft[mismatch] as usize]})
    };
    Ok(json!({"token_position_1based": mismatch + 1,
        "serial": summarize(&serial_logits), "wide_on_serial_prefix": summarize(&wide_serial_prefix),
        "wide_replayed_prefix": summarize(&wide_logits),
        "serial_logits": serial_logits, "wide_on_serial_prefix_logits": wide_serial_prefix,
        "wide_replayed_prefix_logits": wide_logits}))
}

fn main() -> Result<()> {
    let args = Args::parse();
    if args.diagnostic {
        lisa_bench::ttft_init();
    }
    ensure!(
        args.tokens >= 2 && args.repeats > 0,
        "need >=2 tokens and >0 repeats"
    );
    safety()?;
    let _ = lisa_metal::memory::set_cache_limit(8 << 30);
    let target = checkpoint::resolve_model_dir(&args.model)?;
    let draft = checkpoint::resolve_model_dir(&args.draft)?;
    let checkpoint: serde_json::Value =
        serde_json::from_slice(&std::fs::read(target.join("config.json"))?)?;
    let geometry = checkpoint.get("text_config").unwrap_or(&checkpoint);
    let target_geometry = json!({
        "hidden_size": geometry["hidden_size"],
        "query_heads": geometry["num_attention_heads"],
        "kv_heads": geometry["num_key_value_heads"],
        "head_dim": geometry["head_dim"],
        "layers": geometry["num_hidden_layers"],
        "vocab": geometry["vocab_size"],
    });
    let tok = Tokenizer::load(&target)?;
    let prompt = tok.encode(
        &tokenizer::generation_prompt(&[("user".into(), args.prompt.clone())]),
        false,
    )?;
    let load_start = Instant::now();
    let Loaded::Language(mut model) = loading::load_dir_with_dspine(&target, &draft)? else {
        anyhow::bail!("expected a language model");
    };
    let model_load_s = load_start.elapsed().as_secs_f64();
    ensure!(model.has_block_drafter(), "DSpine was not loaded");
    lisa_metal::memory::clear_cache();
    let loaded_active_bytes = lisa_metal::memory::active_memory()?;
    ensure!(
        prompt.len() + args.tokens <= model.max_position_embeddings(),
        "prompt/output exceed context capacity"
    );
    generate::set_eos_ids_for_model(tok.im_end_ids.clone(), &*model);
    if let (Some(reference), Some(rounds)) = (&args.audit_reference, &args.audit_rounds) {
        let report = audit(&mut *model, reference, rounds)?;
        std::fs::write(&args.out, serde_json::to_vec_pretty(&report)?)?;
        return Ok(());
    }
    eprintln!("prompt: {} tokens; warming both modes", prompt.len());
    let policy_identity = lisa_bench::bindings::qwen35::dspine_policy::detect(&target, &draft)?;
    let policy_path = lisa_bench::bindings::qwen35::dspine_policy::cache_path(&policy_identity)?;
    let mut report = json!({"model": target, "draft": draft, "prompt": args.prompt,
        "policy_identity": policy_identity, "policy_path": policy_path,
        "draft_block_size": model.max_speculative_depth() + 1,
        "ssm": if model.block_drafter_state_replay() { "replay" } else { "capture" },
        "prompt_tokens": prompt, "output_tokens": args.tokens,
        "loaded_active_bytes": loaded_active_bytes, "model_load_s": model_load_s,
        "target_geometry": target_geometry,
        "retrieval_mode": std::env::var("LISA_DSPINE_RETRIEVAL").unwrap_or_else(|_| "native".into()),
        "table_mode": std::env::var("LISA_DSPINE_TABLES").unwrap_or_else(|_| "selected".into()),
        "warmups": [], "runs": []});
    let warmups: Vec<_> = if args.compare_adaptive {
        vec![(false, None), (true, args.verify_width), (true, None)]
    } else if args.verify_widths.is_empty() {
        vec![(false, args.verify_width), (true, args.verify_width)]
    } else {
        std::iter::once((false, None))
            .chain(args.verify_widths.iter().map(|&w| (true, Some(w))))
            .collect()
    };
    for (warm, width) in warmups {
        let mut result = run(&mut *model, &prompt, args.tokens, warm, width)?;
        let ids: Vec<u32> = serde_json::from_value(result["tokens"].clone())?;
        result["text"] = json!(tok.decode(&ids)?);
        report["warmups"].as_array_mut().unwrap().push(result);
        std::fs::write(&args.out, serde_json::to_vec_pretty(&report)?)?;
    }
    if args.diagnostic {
        lisa_metal::trace::finish();
        return Ok(());
    }
    for _ in 0..args.repeats {
        let schedule: Vec<_> = if args.compare_adaptive {
            vec![
                (false, None),
                (true, args.verify_width),
                (true, None),
                (true, None),
                (true, args.verify_width),
                (false, None),
            ]
        } else if args.verify_widths.is_empty() {
            [false, true, true, false]
                .into_iter()
                .map(|draft| (draft, args.verify_width))
                .collect()
        } else {
            std::iter::once((false, None))
                .chain(
                    args.verify_widths
                        .iter()
                        .chain(args.verify_widths.iter().rev())
                        .map(|&w| (true, Some(w))),
                )
                .chain(std::iter::once((false, None)))
                .collect()
        };
        for (draft, width) in schedule {
            let mut result = run(&mut *model, &prompt, args.tokens, draft, width)?;
            let ids: Vec<u32> = serde_json::from_value(result["tokens"].clone())?;
            result["text"] = json!(tok.decode(&ids)?);
            report["runs"].as_array_mut().unwrap().push(result);
            std::fs::write(&args.out, serde_json::to_vec_pretty(&report)?)?;
        }
    }
    let runs = report["runs"].as_array().unwrap();
    let matches = runs.iter().all(|run| run["tokens"] == runs[0]["tokens"]);
    if args.save_policy {
        let mut totals = std::collections::BTreeMap::<usize, (u64, f64)>::new();
        for run in runs.iter().filter(|run| run["mode"] == "dspine") {
            let width = run["verify_width"]
                .as_u64()
                .context("SPS run missing width")? as usize;
            let total = totals.entry(width).or_default();
            total.0 += run["decode_tokens"].as_u64().unwrap();
            total.1 += run["decode_s"].as_f64().unwrap();
        }
        let (&verify_width, _) = totals
            .iter()
            .max_by(|(wa, a), (wb, b)| {
                (a.0 as f64 / a.1)
                    .total_cmp(&(b.0 as f64 / b.1))
                    .then(wa.cmp(wb))
            })
            .context("no SPS measurements")?;
        let policy = lisa_bench::bindings::qwen35::dspine_policy::Policy {
            format: "lisa-dspine-sps-metal-v1".into(),
            identity: policy_identity,
            verify_width,
            draft_width: model.max_speculative_depth() + 1,
            ssm: if model.block_drafter_state_replay() {
                lisa_bench::bindings::qwen35::dspine_policy::SsmMode::Replay
            } else {
                lisa_bench::bindings::qwen35::dspine_policy::SsmMode::Capture
            },
        };
        policy.validate(&policy.identity)?;
        std::fs::create_dir_all(policy_path.parent().unwrap())?;
        let tmp = policy_path.with_extension("json.tmp");
        std::fs::write(&tmp, serde_json::to_vec_pretty(&policy)?)?;
        std::fs::rename(tmp, &policy_path)?;
        report["saved_policy"] = serde_json::to_value(&policy)?;
        eprintln!("Saved SPS profile {}", policy_path.display());
    }

    report["all_tokens_match"] = json!(matches);
    std::fs::write(&args.out, serde_json::to_vec_pretty(&report)?)?;
    println!(
        "report: {}; all tokens match: {matches}",
        args.out.display()
    );
    Ok(())
}
