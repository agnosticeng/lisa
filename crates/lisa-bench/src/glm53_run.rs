//! One production GLM generation with machine-readable timing and token IDs.
//! There is no benchmark warmup or token printing inside the timed interval.

use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, ensure};
use lisa_engine::cli::ModelDir;
use lisa_engine::models::{self, Loaded};
use lisa_engine::{generate, sampler, session, tokenizer};

#[derive(clap::Args)]
pub struct Args {
    #[arg(long)]
    model: ModelDir,
    #[arg(
        long,
        conflicts_with = "prompt_file",
        required_unless_present = "prompt_file"
    )]
    prompt: Option<String>,
    #[arg(long, conflicts_with = "prompt", required_unless_present = "prompt")]
    prompt_file: Option<PathBuf>,
    /// Encode prompt text directly, without the production GLM chat prefix.
    #[arg(long, default_value_t = false)]
    raw: bool,
    #[arg(long, default_value_t = 4096)]
    ctx_size: usize,
    #[arg(long, default_value_t = 64)]
    max_tokens: usize,
    /// 0 uses serial generation; 2 uses GLM's native target-conditioned MTP.
    #[arg(long, default_value_t = 0, value_parser = parse_depth)]
    depth: usize,
    #[arg(long, default_value_t = 0)]
    seed: u64,
    #[arg(long, visible_alias = "temp", default_value_t = 0.0)]
    temperature: f32,
    /// New JSON result file. Existing evidence files are never overwritten.
    #[arg(long)]
    output: PathBuf,
    /// Save every target logit vector as NPZ; diagnostic run, excluded from timings.
    #[arg(long)]
    capture_logits: Option<PathBuf>,
    /// Comma-separated trunk layers to include in the diagnostic NPZ.
    #[arg(long, value_delimiter = ',', requires = "capture_logits")]
    capture_layers: Vec<usize>,
    /// Comma-separated stage-name substrings; empty records every selected stage.
    #[arg(long, value_delimiter = ',', requires = "capture_logits")]
    capture_keys: Vec<String>,
    /// Explicitly select the native DS4 Q8 schedule, also used by default.
    #[arg(long, default_value_t = false)]
    matrix_prefill: bool,
    /// Explicitly select native DS4 attention, also used by default.
    #[arg(long, default_value_t = false)]
    native_dsa: bool,
    /// Explicitly select native DS4 fused Q8 gate/up/SwiGLU, also used by default.
    #[arg(long, default_value_t = false)]
    native_q8_gate_up: bool,
}

fn parse_depth(value: &str) -> Result<usize, String> {
    match value {
        "0" => Ok(0),
        "2" => Ok(2),
        _ => Err("glm53-run --depth must be 0 (serial) or 2 (target MTP)".into()),
    }
}

pub fn run(args: Args) -> anyhow::Result<()> {
    ensure!(args.max_tokens > 0, "--max-tokens must be positive");
    ensure!(args.ctx_size > 0, "--ctx-size must be positive");
    ensure!(
        args.temperature.is_finite() && args.temperature >= 0.0,
        "--temperature must be finite and nonnegative"
    );
    ensure!(
        !args.output.exists(),
        "output already exists: {}",
        args.output.display()
    );
    if let Some(path) = &args.capture_logits {
        ensure!(
            !path.exists(),
            "logit capture already exists: {}",
            path.display()
        );
    }
    let model_type = models::model_type_of(&args.model)?;
    ensure!(
        matches!(model_type.as_str(), "glm5_next" | "glm5_next_text"),
        "glm53-run requires a GLM-5.3 checkpoint, got {model_type}"
    );
    let prompt = match (&args.prompt, &args.prompt_file) {
        (Some(text), None) => text.clone(),
        (None, Some(path)) => std::fs::read_to_string(path)
            .with_context(|| format!("reading prompt {}", path.display()))?,
        _ => anyhow::bail!("provide exactly one of --prompt or --prompt-file"),
    };
    let tokenization_started = Instant::now();
    let tokenizer = tokenizer::Tokenizer::load(&args.model)?;
    let rendered_prompt = if args.raw {
        prompt.clone()
    } else {
        tokenizer::glm_generation_prompt(
            &prompt,
            Some("You are a helpful assistant"),
            tokenizer::GlmReasoning::High,
        )
    };
    let input_ids = if args.raw {
        tokenizer.encode(&rendered_prompt, false)?
    } else {
        tokenizer.encode_special(&rendered_prompt)?
    };
    let tokenization_seconds = tokenization_started.elapsed().as_secs_f64();
    ensure!(!input_ids.is_empty(), "prompt encodes to no tokens");
    ensure!(
        input_ids
            .len()
            .checked_add(args.max_tokens)
            .is_some_and(|n| n <= args.ctx_size),
        "context overflow: {} prompt tokens + {} generated tokens exceed --ctx-size {}",
        input_ids.len(),
        args.max_tokens,
        args.ctx_size
    );
    if let Some(parent) = args
        .output
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        std::fs::create_dir_all(parent)?;
    }

    let _q8_scope = args
        .matrix_prefill
        .then(|| lisa_engine::core::quant::Glm53Q8PrefillScope::new(true));
    let _dsa_scope = args
        .native_dsa
        .then(|| lisa_mlx::models::glm53::kernels::Glm53DsaNativeScope::new(true));
    let _gate_up_scope = args
        .native_q8_gate_up
        .then(|| lisa_engine::models::glm53::moe::Glm53Q8GateUpNativeScope::new(true));
    let load_started = Instant::now();
    let mut tower = match models::load_dir(&args.model)? {
        Loaded::Language(tower) => tower,
        Loaded::Decision(_) => anyhow::bail!("glm53-run requires a language model"),
    };
    tower.set_context_capacity(args.ctx_size)?;
    if args.capture_logits.is_some() {
        if args.capture_layers.is_empty() {
            tower.enable_logit_capture()?;
        } else {
            tower.enable_stage_capture(args.capture_layers.clone(), args.capture_keys.clone())?;
        }
    }
    let runtime = lisa_mlx::Stream::thread_local_or_default()
        .runtime()
        .clone();
    runtime.synchronize()?;
    let model_load_seconds = load_started.elapsed().as_secs_f64();
    ensure!(
        args.depth == 0 || tower.has_target_mtp(),
        "checkpoint has no target MTP head"
    );
    generate::set_eos_ids_for_model(tokenizer.im_end_ids.clone(), tower.as_ref());
    let mut sampler = sampler::Sampler {
        temperature: args.temperature,
        seed: args.seed,
        ..Default::default()
    };
    let active_loaded = lisa_mlx::memory::active_memory()?;
    let private_loaded = lisa_mlx::memory::private_allocated_memory()?;
    let mut first_token_seconds = None;
    let generation_started = Instant::now();
    let mut on_token = |_token: u32| -> anyhow::Result<()> {
        first_token_seconds.get_or_insert_with(|| generation_started.elapsed().as_secs_f64());
        Ok(())
    };
    let (output_ids, serial_stats) = if args.depth == 0 {
        let (ids, stats) = generate::generate(
            tower.as_mut(),
            &input_ids,
            args.max_tokens,
            &mut sampler,
            0,
            &mut on_token,
        )?;
        (ids, Some(stats))
    } else {
        let mut session = session::Session::new(tower.as_mut());
        let stochastic = (!sampler.greedy()).then_some(&mut sampler);
        let ids = session.generate_mtp_depth(
            tower.as_mut(),
            &input_ids,
            args.max_tokens,
            session::MtpDepth::Fixed(2),
            None,
            stochastic,
            &mut on_token,
        )?;
        (ids, None)
    };
    runtime.synchronize()?;
    let total_generation_seconds = generation_started.elapsed().as_secs_f64();
    if let Some(path) = &args.capture_logits {
        tower.write_logit_capture(path.to_str().context("capture path is not UTF-8")?)?;
    }
    let active_final = lisa_mlx::memory::active_memory()?;
    let private_final = lisa_mlx::memory::private_allocated_memory()?;
    let decode_steps = serial_stats
        .as_ref()
        .map(|stats| stats.generated_tokens.saturating_sub(1));
    let prefill_seconds = serial_stats.as_ref().map(|stats| stats.prefill_seconds);
    let decode_seconds = serial_stats.as_ref().map(|stats| stats.decode_seconds);
    let decode_tokens_per_second = serial_stats.as_ref().and_then(|stats| {
        (stats.decode_seconds > 0.0)
            .then(|| stats.generated_tokens.saturating_sub(1) as f64 / stats.decode_seconds)
    });
    let result = serde_json::json!({
        "format": "lisa-glm53-run-v1",
        "source": "lisa",
        "model": args.model.as_ref().to_string_lossy(),
        "model_type": model_type,
        "prompt": prompt,
        "rendered_prompt": rendered_prompt,
        "raw": args.raw,
        "ctx_size": args.ctx_size,
        "max_tokens": args.max_tokens,
        "depth": args.depth,
        "capture_logits": args.capture_logits,
        "q8_schedule": "native_ds4",
        "matrix_prefill_explicit": args.matrix_prefill,
        "dsa_schedule": if lisa_mlx::models::glm53::kernels::glm53_dsa_native_enabled() { "native_ds4" } else { "legacy_affine" },
        "native_dsa_explicit": args.native_dsa,
        "q8_gate_up_schedule": if lisa_engine::models::glm53::moe::native_gate_up_enabled() && std::env::var("LISA_GLM53_Q8_GATE_UP").as_deref() != Ok("separate") { "native_ds4_fused" } else { "separate" },
        "native_q8_gate_up_explicit": args.native_q8_gate_up,
        "timing_valid_for_performance": args.capture_logits.is_none(),
        "sampling": {"seed": args.seed, "temperature": args.temperature,
            "top_k": 0, "top_p": 1.0, "min_p": 0.0, "repetition_penalty": 1.0,
            "mtp_exact_sampling": false},
        "input_ids": input_ids,
        "output_ids": output_ids,
        "prompt_tokens": input_ids.len(),
        "generated_tokens": output_ids.len(),
        "tokenization_seconds": tokenization_seconds,
        "model_load_seconds": model_load_seconds,
        "time_to_first_token_seconds": first_token_seconds,
        "prefill_seconds": prefill_seconds,
        "decode_seconds": decode_seconds,
        "decode_steps": decode_steps,
        "decode_tokens_per_second": decode_tokens_per_second,
        "total_generation_seconds": total_generation_seconds,
        "generation_tokens_per_second": output_ids.len() as f64 / total_generation_seconds.max(1e-9),
        "memory": {
            "resident_model_bytes": active_loaded.saturating_sub(private_loaded),
            "active_after_load_bytes": active_loaded,
            "private_after_load_bytes": private_loaded,
            "active_after_generation_bytes": active_final,
            "private_after_generation_bytes": private_final,
            "ram_budget_bytes": lisa_mlx::memory::ram_budget(),
        },
    });
    // All token serialization and filesystem work happens after timing ends.
    // The external acceptance runner owns source and output artifact hashes.
    let mut output = std::fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&args.output)
        .with_context(|| format!("creating result {}", args.output.display()))?;
    serde_json::to_writer_pretty(&mut output, &result)?;
    output.write_all(b"\n")?;
    eprintln!(
        "GLM result: {} ({} tokens in {:.3}s)",
        args.output.display(),
        output_ids.len(),
        total_generation_seconds
    );
    Ok(())
}
