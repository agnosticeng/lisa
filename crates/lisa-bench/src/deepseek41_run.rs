//! Diagnostics for the native GGUF graph and bounded prefill. Timings do
//! not certify parity with DS4's wide prefill or optimized decode schedule.

use anyhow::{Context, Result, ensure};
use lisa_engine::models::deepseek41::{self, graph::Graph, state::State, tokenizer::Tokenizer};
use lisa_gguf::Package;
use std::{io::Write, path::PathBuf, time::Instant};

#[derive(clap::Args)]
pub struct Args {
    /// First GGUF shard (split.no=0). All parts must be fully downloaded.
    #[arg(long)]
    model: PathBuf,
    #[arg(long)]
    prompt: String,
    #[arg(long)]
    raw: bool,
    #[arg(long, default_value_t = 0)]
    reasoning_effort: u8,
    #[arg(long, default_value_t = 2048)]
    ctx_size: usize,
    #[arg(long, default_value_t = 16)]
    max_tokens: usize,
    /// Bounded prompt row schedule. One retains the scalar reference path.
    #[arg(long, default_value_t = 8)]
    prefill_chunk: usize,
    /// Only inspect the GGUF tokenizer; do not map or execute model weights.
    #[arg(long)]
    tokenize_only: bool,
    /// Save diagnostic intermediates. Captures are excluded from timing claims.
    #[arg(long)]
    capture: Option<PathBuf>,
    #[arg(long, value_delimiter = ',', requires = "capture")]
    capture_layers: Vec<usize>,
    /// Replay one decode token from an independent cache snapshot, comparing
    /// every logit bit. This validates LISA state reuse, not DS4 model parity.
    #[arg(long)]
    snapshot_check: bool,
    /// New JSON evidence file; existing evidence is never overwritten.
    #[arg(long)]
    output: Option<PathBuf>,
}

fn greedy(logits: &lisa_mlx::Array) -> Result<u32> {
    let values = logits.as_slice::<f32>();
    ensure!(
        !values.is_empty() && values.iter().all(|v| v.is_finite()),
        "DeepSeek produced non-finite logits"
    );
    let mut best = 0;
    for i in 1..values.len() {
        if values[i] > values[best] {
            best = i;
        }
    }
    Ok(best as u32)
}

pub fn run(args: Args) -> Result<()> {
    ensure!(args.max_tokens > 0, "--max-tokens must be positive");
    ensure!(
        (1..=8).contains(&args.prefill_chunk),
        "--prefill-chunk must be between 1 and 8"
    );
    ensure!(
        (1..=deepseek41::state::MAX_CONTEXT).contains(&args.ctx_size),
        "DeepSeek --ctx-size must be between 1 and 1048576"
    );
    ensure!(
        args.capture_layers.iter().all(|&l| l < 40),
        "capture layer is out of range"
    );
    for path in [&args.output, &args.capture].into_iter().flatten() {
        ensure!(!path.exists(), "refusing to overwrite {}", path.display());
    }
    let package = Package::open(&args.model)?;
    let tokenizer = Tokenizer::from_header(&package.header)?;
    let prompt = if args.raw {
        tokenizer.encode(&args.prompt)?
    } else {
        tokenizer.chat(None, &args.prompt, args.reasoning_effort)?
    };
    ensure!(!prompt.is_empty(), "empty DeepSeek prompt");
    if args.tokenize_only {
        println!(
            "{}",
            serde_json::json!({ "prompt_ids": prompt, "ds4_revision": deepseek41::DS4_REVISION })
        );
        return Ok(());
    }
    ensure!(
        prompt
            .len()
            .checked_add(args.max_tokens)
            .is_some_and(|n| n <= args.ctx_size),
        "prompt and generation exceed context capacity"
    );
    eprintln!(
        "DeepSeek diagnostic: {} prompt tokens, rows {}, DS4 {}",
        prompt.len(),
        args.prefill_chunk,
        deepseek41::DS4_REVISION
    );
    let start = Instant::now();
    let mut graph = Graph::load(&package)?;
    let load_seconds = start.elapsed().as_secs_f64();
    eprintln!(
        "DeepSeek: mapped {:.2} GiB resident weights; Engram stays on disk",
        graph.mapped_weight_bytes() as f64 / (1u64 << 30) as f64
    );
    let mut state = State::new(args.ctx_size)?;
    let capture = args
        .capture
        .as_ref()
        .map(|_| Graph::diagnostic_capture(args.capture_layers.clone()));
    let start = Instant::now();
    let mut last = None;
    for (i, tokens) in prompt.chunks(args.prefill_chunk).enumerate() {
        let end = ((i + 1) * args.prefill_chunk).min(prompt.len());
        last = graph
            .prefill(tokens, &mut state, end == prompt.len(), capture.as_ref())?
            .pop();
        eprintln!("DeepSeek prefill {end}/{}", prompt.len());
    }
    let prefill_seconds = start.elapsed().as_secs_f64();
    let mut logits = last
        .context("missing prefill output")?
        .logits
        .context("missing prefill logits")?;
    if args.snapshot_check {
        let next = greedy(&logits)?;
        let snapshot = state.snapshot()?;
        let actual = graph
            .step(next, &mut state, true, None)?
            .logits
            .context("missing check logits")?;
        let expected: Vec<u32> = actual
            .as_slice::<f32>()
            .iter()
            .map(|v| v.to_bits())
            .collect();
        state = snapshot.snapshot()?;
        let replay = graph
            .step(next, &mut state, true, None)?
            .logits
            .context("missing replay logits")?;
        ensure!(
            replay
                .as_slice::<f32>()
                .iter()
                .map(|v| v.to_bits())
                .eq(expected),
            "DeepSeek snapshot replay changed logits"
        );
        state = snapshot;
        eprintln!("DeepSeek snapshot replay: all logit bits match");
    }
    let start = Instant::now();
    let mut generated = Vec::new();
    let mut decode_steps = 0usize;
    for i in 0..args.max_tokens {
        let token = greedy(&logits)?;
        generated.push(token);
        if token == tokenizer.eos || i + 1 == args.max_tokens {
            break;
        }
        logits = graph
            .step(token, &mut state, true, capture.as_ref())?
            .logits
            .context("missing decode logits")?;
        decode_steps += 1;
    }
    let decode_seconds = start.elapsed().as_secs_f64();
    if let (Some(capture), Some(path)) = (&capture, &args.capture) {
        capture.flush(path.to_str().context("capture path is not UTF-8")?)?;
    }
    let report = serde_json::json!({
        "model": args.model, "ds4_revision": deepseek41::DS4_REVISION,
        "schedule": "bounded layer-major prefill / scalar decode", "prefill_chunk": args.prefill_chunk, "parity_certified": false,
        "captured": args.capture.is_some(), "snapshot_checked": args.snapshot_check,
        "prompt_ids": prompt, "generated_ids": generated,
        "text": String::from_utf8_lossy(&tokenizer.decode_bytes(&generated)?),
        "load_seconds": load_seconds, "prefill_seconds": prefill_seconds,
        "decode_seconds": decode_seconds, "decode_steps": decode_steps,
        "decode_tokens_per_second": if args.capture.is_none() && decode_steps > 0 {
            Some(decode_steps as f64 / decode_seconds) } else { None },
    });
    let json = serde_json::to_string_pretty(&report)?;
    if let Some(path) = &args.output {
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)?;
        file.write_all(json.as_bytes())?;
        file.write_all(b"\n")?;
    }
    println!("{json}");
    Ok(())
}
