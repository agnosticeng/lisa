use std::path::PathBuf;

use clap::{Parser, Subcommand};
use lisa_engine::cli::{DepthArg, ModelDir, build_sampler};
use lisa_engine::core::mem::mlx_mem_line;
use lisa_engine::models::LanguageModel;
use lisa_engine::models::laya::{Laya, LayaDevice};
use lisa_engine::models::qwen4::config;
use lisa_engine::models::qwen4::tower as model;
use lisa_engine::{batch, generate, session, tokenizer};
use lisa_serve as serve;

/// These commands currently use Qwen4-specific loading and turn framing.
fn require_qwen4_command(model: &std::path::Path, command: &str) -> anyhow::Result<()> {
    let family = lisa_engine::models::model_type_of(model)?;
    anyhow::ensure!(
        matches!(family.as_str(), "qwen4_exp" | "qwen4_exp_text"),
        "`lisa {command}` currently supports Qwen4 checkpoints; use `run` or `serve` for {family}"
    );
    Ok(())
}

fn tokenize(model: &std::path::Path, prompt: &str, raw: bool) -> anyhow::Result<()> {
    let tok = tokenizer::Tokenizer::load(model)?;
    if lisa_engine::models::model_type_of(model)? == "deepseek41" {
        let ids = if raw {
            tok.encode_plain(prompt)?
        } else {
            tok.native_chat(None, prompt, 0)?
        };
        println!("ids: {:?}", &ids[..ids.len().min(40)]);
        return Ok(());
    }
    let is_glm = matches!(
        lisa_engine::models::model_type_of(model)?.as_str(),
        "glm5_next" | "glm5_next_text" | "glm5-next"
    );
    let text = if raw {
        prompt.to_owned()
    } else if is_glm {
        tokenizer::glm_generation_prompt(
            &prompt,
            Some("You are a helpful assistant"),
            tokenizer::GlmReasoning::High,
        )
    } else {
        tokenizer::generation_prompt(&[("user".to_string(), prompt.to_owned())])
    };
    let ids = if is_glm && !raw {
        tok.encode_special(&text)?
    } else if raw {
        tok.encode_plain(&text)?
    } else {
        tok.encode(&text, false)?
    };
    println!("ids: {:?}", &ids[..ids.len().min(40)]);
    let markers = if is_glm {
        ["[gMASK]", "<|user|>", "<|assistant|>"]
    } else {
        ["<|im_start|>", "<|im_end|>", "<think>"]
    };
    for want in markers {
        println!("{want} -> {:?}", tok.inner.token_to_id(want));
    }
    Ok(())
}

/// Parse the runtime context capacity before loading model weights.
fn parse_context_capacity(value: &str) -> Result<usize, String> {
    let capacity = value
        .parse::<usize>()
        .map_err(|_| "--ctx-size must be a positive integer".to_owned())?;
    if capacity == 0 {
        return Err("--ctx-size must be greater than zero".to_owned());
    }
    Ok(capacity)
}

/// A `--state` value: a JSON object/array is parsed; anything else is a string.
fn parse_state(s: &str) -> serde_json::Value {
    match serde_json::from_str::<serde_json::Value>(s) {
        Ok(v @ serde_json::Value::Object(_)) | Ok(v @ serde_json::Value::Array(_)) => v,
        _ => serde_json::Value::String(s.to_string()),
    }
}

/// Parse `--temperature-by-options TYPE:SIZE=TEMP` into `(bucket, temp)` pairs.
fn parse_temp_by_options(items: &[String]) -> anyhow::Result<Vec<(String, f32)>> {
    items
        .iter()
        .map(|s| {
            let (k, v) = s.rsplit_once('=').ok_or_else(|| {
                anyhow::anyhow!("--temperature-by-options wants TYPE:SIZE=TEMP, got {s:?}")
            })?;
            let val: f32 = v
                .trim()
                .parse()
                .map_err(|_| anyhow::anyhow!("invalid temperature {v:?} in {s:?}"))?;
            Ok((k.trim().to_string(), val))
        })
        .collect()
}

/// Keep only the `k` highest probabilities in each answer's `probabilities` map.
fn trim_topk(v: &mut serde_json::Value, k: usize) {
    let Some(answers) = v.get_mut("answers").and_then(|a| a.as_object_mut()) else {
        return;
    };
    for (_, ans) in answers.iter_mut() {
        let Some(probs) = ans.get_mut("probabilities").and_then(|p| p.as_object_mut()) else {
            continue;
        };
        let mut items: Vec<(String, f64)> = probs
            .iter()
            .map(|(key, val)| (key.clone(), val.as_f64().unwrap_or(0.0)))
            .collect();
        items.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        items.truncate(k);
        let kept: serde_json::Map<String, serde_json::Value> = items
            .into_iter()
            .map(|(key, val)| (key, serde_json::json!(val)))
            .collect();
        *probs = kept;
    }
}

#[derive(Parser)]
#[command(
    name = "lisa",
    about = "Local Qwen, GLM and Laya inference with Rust and Metal"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Prepare GLM resident weights from the original checkpoint (CPU only).
    Prepare {
        #[arg(long)]
        model: PathBuf,
        /// Report required disk space without writing weight files.
        #[arg(long)]
        plan_only: bool,
    },
    /// Print checkpoint diagnostics; GGUF inspection does not load a GPU model.
    Inspect {
        #[arg(long)]
        model: ModelDir,
        /// Inspect only the first GGUF header, without requiring other shards.
        #[arg(long)]
        header_only: bool,
    },
    /// Load a typed-decision model (e.g. Laya) and run it.
    Decide {
        #[arg(long)]
        model: ModelDir,
        /// Run a fixed probe input and print the logits/action.
        #[arg(long)]
        probe: bool,
        /// JSON state (object/array, or a plain string).
        #[arg(long)]
        state: Option<String>,
        /// JSON questions object, keyed by question id.
        #[arg(long)]
        questions: Option<String>,
        /// Read `{"state":…,"questions":…}` from a JSON file.
        #[arg(long)]
        input: Option<PathBuf>,

        // Overrides of `rl_agent_config.json` (validated 4 < head_max_len < max_len <= 8192).
        /// Option-prompt token budget shared across a question's options.
        #[arg(long)]
        head_max_len: Option<usize>,
        /// Total token budget (option prompt + state).
        #[arg(long)]
        max_len: Option<usize>,
        /// Per-type temperatures for choice,score,noul (e.g. `0.7,1.0,1.0`).
        /// Beats the shipped config buckets; clamped to [0.5, 5.0] (noul excepted).
        /// `--temperature-by-options` is applied after this and wins.
        #[arg(long, value_delimiter = ',')]
        temperature: Option<Vec<f32>>,
        /// Bucketed temperature, repeatable, `TYPE:SIZE=TEMP`; SIZE must be one of
        /// `2`, `3-5`, `6-10`, `11+` (e.g. `choice:6-10=0.7`). Wins over
        /// `--temperature` and the shipped config. Clamped to [0.5, 5.0] (noul excepted).
        #[arg(long = "temperature-by-options")]
        temperature_by_options: Vec<String>,

        /// Device override: `metal` | `cpu`.
        #[arg(long)]
        device: Option<String>,

        /// Keep only the top-k probabilities per answer (display).
        #[arg(long)]
        top_k: Option<usize>,
    },
    /// Generate a PNG with a local FLUX.2 or FLUX 3 Klein checkpoint.
    Image {
        #[arg(long, default_value = "mlx-community/flux2-klein-4b-4bit")]
        model: ModelDir,
        #[arg(long)]
        prompt: String,
        #[arg(long, default_value = "image.png")]
        output: std::path::PathBuf,
        #[arg(long, default_value_t = 512)]
        width: u32,
        #[arg(long, default_value_t = 512)]
        height: u32,
        /// Denoising steps (default: 4 for FLUX.2, 16 for FLUX 3).
        #[arg(long)]
        steps: Option<u32>,
        /// Skip FLUX 3 prompt expansion when providing a complete scene caption.
        #[arg(long)]
        prompt_is_caption: bool,
        #[arg(long, default_value_t = 0)]
        seed: u64,
    },
    /// Load a model and generate.
    Run {
        #[arg(long)]
        model: ModelDir,
        /// Runtime prompt + output context capacity (GLM default: 32768).
        #[arg(long, value_parser = parse_context_capacity)]
        ctx_size: Option<usize>,
        #[arg(long, default_value = "Explain what a MoE layer is in two sentences.")]
        prompt: String,
        /// PNG, JPEG or WebP image; repeat for multiple images.
        #[arg(long)]
        image: Vec<std::path::PathBuf>,
        /// System message; GLM defaults to ds4's "You are a helpful assistant".
        #[arg(long)]
        system: Option<String>,
        #[arg(long, default_value_t = 128)]
        max_tokens: usize,
        /// Sampling temperature; GLM defaults to ds4's 1.0, other models to 0.
        #[arg(long)]
        temperature: Option<f32>,
        #[arg(long, default_value_t = false)]
        raw: bool,
        /// Nucleus sampling; GLM defaults to ds4's 0.95, other models to 1.0.
        #[arg(long)]
        top_p: Option<f32>,
        #[arg(long, default_value_t = 0)]
        top_k: usize,
        /// Minimum sampling probability; GLM and other models default to 0.
        #[arg(long)]
        min_p: Option<f32>,
        #[arg(long, default_value_t = 1.0)]
        rep_penalty: f32,
        #[arg(long, default_value_t = 0)]
        seed: u64,
        /// Use ds4's maximum GLM reasoning effort instead of its default High.
        #[arg(long, conflicts_with = "nothink")]
        think_max: bool,
        /// Close an empty GLM think block and omit the reasoning-effort prefix.
        #[arg(long, conflicts_with = "think_max")]
        nothink: bool,
        /// Explicitly keep the default High GLM reasoning effort.
        #[arg(long, conflicts_with_all = ["think_max", "nothink"])]
        think: bool,
        /// MTP policy (`auto` = Qwen EV controller or GLM target MTP;
        /// 0 = serial; 2..=7 = fixed Qwen depth or GLM target MTP).
        #[arg(long, default_value = "auto")]
        depth: DepthArg,
        /// Use DS4's exact stochastic acceptance and residual sampling for GLM MTP.
        #[arg(long, default_value_t = false)]
        mtp_exact_sampling: bool,
        /// Prompt-lookup (PLD) speculative decode (greedy only).
        #[arg(long, default_value_t = false)]
        pld: bool,
        /// PLD draft depth (used when --pld is set; 0 = default 4).
        #[arg(long, default_value_t = 0)]
        pld_depth: usize,
        /// Ignore EOS: generate exactly max_tokens (the clean-throughput
        /// measurement arm).
        #[arg(long, default_value_t = false)]
        ignore_eos: bool,
        /// Echo router: score the prompt's 4-gram recurrence and route to PLD
        /// when it clears `--router-threshold` (greedy only).
        #[arg(long, default_value_t = false)]
        router: bool,
        /// Echo score at or above which the router picks PLD (guards the
        /// long-KV branch; below LONG_KV PLD is the default regardless).
        #[arg(long, default_value_t = 0.80)]
        router_threshold: f32,
        /// Draft-ranking oracle (specs/08): rank the target's true token in
        /// the draft head's own distribution on every round verify rejects,
        /// and print the top-k repair floor at the end. Diagnostics only —
        /// the proposal, verify, and the committed tokens are untouched.
        #[arg(long, default_value_t = false)]
        oracle: bool,
    },
    /// Print token ids for a prompt.
    Tok {
        #[arg(long)]
        model: ModelDir,
        #[arg(long, default_value = "What is the capital of France?")]
        prompt: String,
        #[arg(long, default_value_t = false)]
        raw: bool,
    },
    /// Qwen4 cohort batching: N prompts decoded together in one batched forward.
    Batch {
        #[arg(long)]
        model: ModelDir,
        /// One prompt per stream (repeat --prompt).
        #[arg(long = "prompt", num_args = 1.., required = true)]
        prompts: Vec<String>,
        #[arg(long, default_value_t = 32)]
        max_tokens: usize,
        #[arg(long, default_value_t = 0.0)]
        temperature: f32,
        #[arg(long, default_value_t = false)]
        raw: bool,
        #[arg(long, default_value_t = 1.0)]
        top_p: f32,
        #[arg(long, default_value_t = 0)]
        top_k: usize,
        #[arg(long, default_value_t = 0.0)]
        min_p: f32,
        #[arg(long, default_value_t = 1.0)]
        rep_penalty: f32,
        #[arg(long, default_value_t = 0)]
        seed: u64,
    },
    /// Qwen4 multi-turn chat with incremental prefill and persistent caches.
    Chat {
        #[arg(long)]
        model: ModelDir,
        #[arg(long, default_value_t = 256)]
        max_tokens: usize,
        #[arg(long, default_value_t = 0.0)]
        temperature: f32,
        #[arg(long, default_value_t = 0)]
        verify: usize,
        /// MTP draft depth for the chat (`auto` = EV controller, the default;
        /// 0 = serial; 2..=7 = fixed MTP; 1 is rejected; speculative chat requires greedy sampling.
        #[arg(long, default_value = "auto")]
        depth: DepthArg,
        /// Prompt-lookup (PLD) speculative decode (greedy only).
        #[arg(long, default_value_t = false)]
        pld: bool,
        /// PLD draft depth (used when --pld is set; 0 = default 4).
        #[arg(long, default_value_t = 0)]
        pld_depth: usize,
        #[arg(long, default_value_t = 1.0)]
        top_p: f32,
        #[arg(long, default_value_t = 0)]
        top_k: usize,
        #[arg(long, default_value_t = 0.0)]
        min_p: f32,
        #[arg(long, default_value_t = 1.0)]
        rep_penalty: f32,
        #[arg(long, default_value_t = 0)]
        seed: u64,
    },
    /// Minimal OpenAI-compatible HTTP server (POST /v1/chat/completions, SSE).
    Serve {
        #[arg(long)]
        model: ModelDir,
        /// Runtime prompt + output context capacity (GLM default: 32768).
        #[arg(long, value_parser = parse_context_capacity)]
        ctx_size: Option<usize>,
        #[arg(long, default_value = "127.0.0.1:8080")]
        addr: String,
        #[arg(long, default_value_t = 256)]
        max_tokens: usize,
        /// Sampling overrides; without them the checkpoint's
        /// `generation_config.json` recommendations apply (greedy only if the
        /// checkpoint declares none).
        #[arg(long)]
        temperature: Option<f32>,
        #[arg(long)]
        top_p: Option<f32>,
        #[arg(long)]
        top_k: Option<usize>,
        #[arg(long)]
        min_p: Option<f32>,
        #[arg(long)]
        rep_penalty: Option<f32>,
        #[arg(long, default_value = "auto")]
        depth: DepthArg,
        /// Use DS4's exact stochastic acceptance and residual sampling for GLM MTP.
        #[arg(long, default_value_t = false)]
        mtp_exact_sampling: bool,
        /// Prompt-lookup (PLD) speculative decode (greedy only).
        #[arg(long, default_value_t = false)]
        pld: bool,
        /// PLD draft depth (used when --pld is set; 0 = default 4).
        #[arg(long, default_value_t = 0)]
        pld_depth: usize,
        /// Cross-request prompt prefix cache: max entries, LRU (0 = off).
        #[arg(long, default_value_t = 2)]
        prefix_cache: usize,
        /// Maximum streams stepped together (continuous-batch width).
        #[arg(long, default_value_t = 4)]
        max_batch: usize,
    },
}

fn main() -> anyhow::Result<()> {
    lisa_engine::ttft_init();
    let result = run();
    lisa_mlx::trace::finish();
    result
}

fn run() -> anyhow::Result<()> {
    let cli = Cli::parse();
    // Weight preparation never initializes a Metal device or allocator.
    if let Command::Prepare { model, plan_only } = &cli.command {
        let summary = lisa_engine::core::prepare::prepare_model(model, *plan_only)?;
        println!("{}", serde_json::to_string_pretty(&summary)?);
        return Ok(());
    }
    if let Command::Tok { model, prompt, raw } = &cli.command {
        return tokenize(model, prompt, *raw);
    }
    if let Command::Inspect { model, header_only } = &cli.command {
        if model.is_file() {
            use lisa_gguf::{Header, Package};
            let header = Header::open(model)?;
            println!("GGUF architecture: {}", header.architecture()?);
            if header.architecture()? == "deepseek41" {
                let config = lisa_engine::models::deepseek41::config::Config::from_header(&header)?;
                let engram = lisa_engine::models::deepseek41::engram::Layout::from_header(&header)?;
                println!(
                    "{} layers, {} hidden, {}/{} active experts",
                    config.layers, config.width, config.selected_experts, config.experts
                );
                println!("Engram rows: {:?} (disk lookup)", engram.rows);
            }
            if *header_only {
                println!(
                    "First-shard header: {} tensors; payloads and other shards not checked",
                    header.tensors.len()
                );
            } else {
                let package = Package::open(model)?;
                println!("{}", package.summary()?);
                if header.architecture()? == "glm5-next" {
                    let c = lisa_engine::models::glm53::gguf::weights::validate(&package)?;
                    println!(
                        "{} trunk layers, {} hidden, {}/{} active experts; embedded MTP block",
                        c.num_hidden_layers,
                        c.hidden_size,
                        c.num_experts_per_tok,
                        c.n_routed_experts
                    );
                    println!(
                        "GLM native GGUF: original Q2 payloads; serial prefill baseline. Full DS4 parity is not certified."
                    );
                }
                if header.architecture()? == "deepseek41" {
                    use lisa_engine::models::deepseek41::{
                        config::Config, engram::Layout, weights::WeightPlan,
                    };
                    let plan = WeightPlan::inspect(
                        &package,
                        &Config::from_header(&header)?,
                        &Layout::from_header(&header)?,
                    )?;
                    println!(
                        "Resident payload: {:.2} GiB; disk-only Engram: {:.2} GiB",
                        plan.resident_payload_bytes as f64 / (1u64 << 30) as f64,
                        plan.disk_payload_bytes as f64 / (1u64 << 30) as f64
                    );
                }
            }
            if header.architecture()? == "deepseek41" {
                println!(
                    "DeepSeek GGUF: native scalar text inference; default context 4096, no draft head. DS4 parity is not yet certified."
                );
            }
            return Ok(());
        }
        anyhow::ensure!(!header_only, "--header-only requires a local GGUF file");
    }
    // Resolve the device backend: `LISA_DEVICE`, else Metal when present, else
    // CPU. Announce a non-default (CPU) selection.
    let backend = lisa_mlx::backend::Backend::from_env().map_err(|e| anyhow::anyhow!("{e}"))?;
    if backend != lisa_mlx::backend::Backend::Metal {
        eprintln!("device: {}", backend.name());
    }
    lisa_mlx::ffi::install_error_handler();
    // Cap how far the Metal buffer pool may grow past load-steady-state before
    // a sweep is forced. Without a cap a pipelined prefill or MTP round pins
    // every temporary it ever made (buffers are only released at a flush).
    let _ = lisa_mlx::memory::set_cache_limit(8 << 30);
    match cli.command {
        Command::Prepare { .. } => unreachable!("preparation is handled before runtime setup"),
        Command::Decide {
            model,
            probe,
            state,
            questions,
            input,
            head_max_len,
            max_len,
            temperature,
            temperature_by_options,
            device,
            top_k,
        } => {
            let backend =
                match device.as_deref() {
                    Some(d) => lisa_mlx::backend::Backend::from_name(d)
                        .map_err(|e| anyhow::anyhow!("{e}"))?,
                    None => lisa_mlx::backend::Backend::from_env()
                        .map_err(|e| anyhow::anyhow!("{e}"))?,
                };
            let model_type = lisa_engine::models::model_type_of(&model)?;
            anyhow::ensure!(
                model_type == "laya",
                "{} has model_type {model_type:?}, not a decision model",
                model.display()
            );
            let dev = match backend {
                lisa_mlx::backend::Backend::Metal => LayaDevice::Metal,
                lisa_mlx::backend::Backend::Cpu => LayaDevice::Cpu,
            };
            let mut laya = Laya::load_device(&model, dev)?;
            let by_options = parse_temp_by_options(&temperature_by_options)?;
            laya.apply_overrides(head_max_len, max_len, temperature, &by_options)?;

            let emit = |v: &serde_json::Value| -> anyhow::Result<()> {
                let mut v = v.clone();
                if let Some(k) = top_k {
                    trim_topk(&mut v, k);
                }
                println!("{}", serde_json::to_string_pretty(&v)?);
                Ok(())
            };
            if let Some(path) = input {
                let v: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&path)?)?;
                let st = v.get("state").cloned().unwrap_or(serde_json::Value::Null);
                let qs = v
                    .get("questions")
                    .cloned()
                    .unwrap_or(serde_json::Value::Null);
                emit(&laya.system_one(&st, &qs)?)
            } else if let (Some(st), Some(qs)) = (state, questions) {
                let st = parse_state(&st);
                let qs: serde_json::Value = serde_json::from_str(&qs)?;
                emit(&laya.system_one(&st, &qs)?)
            } else {
                println!("{}", laya.summary());
                if probe {
                    // Fixed input matching the Python reference harness:
                    // ids [1000..6000, 7, 8], qtype 0, markers [6, 7].
                    let ids = [1000u32, 2000, 3000, 4000, 5000, 6000, 7, 8];
                    let (logits, action) = laya.decide(&ids, 0, &[6, 7])?;
                    println!("probe logits: {:?}", logits);
                    println!("probe action: {:?}", action);
                }
                Ok(())
            }
        }
        Command::Inspect { model, .. } => {
            let t0 = std::time::Instant::now();
            let _tower = lisa_engine::models::load_dir(&model)?;
            println!("loaded in {:.1}s", t0.elapsed().as_secs_f64());
            mlx_mem_line("inspect");
            Ok(())
        }
        Command::Tok { .. } => unreachable!("tokenization is handled before runtime setup"),
        Command::Batch {
            model,
            prompts,
            max_tokens,
            temperature,
            raw,
            top_p,
            top_k,
            min_p,
            rep_penalty,
            seed,
        } => {
            require_qwen4_command(&model, "batch")?;
            let config = config::ModelConfig::from_json(&model.join("config.json"))?;
            let mut tower = model::Tower::load(&model, config)?;
            let tok = tokenizer::Tokenizer::load(&model)?;
            generate::set_eos_ids_for_model(tok.im_end_ids.clone(), &tower);
            let texts: Vec<String> = prompts
                .iter()
                .map(|p| {
                    if raw {
                        p.clone()
                    } else {
                        tokenizer::generation_prompt(&[("user".to_string(), p.clone())])
                    }
                })
                .collect();
            let ids: Vec<Vec<u32>> = texts
                .iter()
                .map(|t| tok.encode(t, false))
                .collect::<anyhow::Result<_>>()?;
            let lens: Vec<usize> = ids.iter().map(|v| v.len()).collect();
            tower.warmup(&ids[0])?;
            eprintln!("batch: {} streams of {} tokens", ids.len(), lens[0]);
            let t0 = std::time::Instant::now();
            let mut acc: Vec<String> = vec![String::new(); ids.len()];
            let mut sampler = build_sampler(temperature, top_k, top_p, min_p, rep_penalty, seed);
            let (_b, out) =
                batch::Batch::generate(&mut tower, &ids, max_tokens, &mut sampler, &mut |s, t| {
                    acc[s].push_str(&tok.decode(&[t])?);
                    Ok(())
                })?;
            for (i, o) in out.iter().enumerate() {
                println!(
                    "--- stream {i}: {} tokens in {:.2}s ---",
                    o.len(),
                    t0.elapsed().as_secs_f64()
                );
                println!("{}", acc[i]);
            }
            Ok(())
        }
        Command::Chat {
            model,
            max_tokens,
            temperature,
            verify,
            depth,
            pld,
            pld_depth,
            top_p,
            top_k,
            min_p,
            rep_penalty,
            seed,
        } => {
            require_qwen4_command(&model, "chat")?;
            let pld = lisa_engine::cli::pld_enable(pld, pld_depth);
            let config = config::ModelConfig::from_json(&model.join("config.json"))?;
            let t0 = std::time::Instant::now();
            let mut tower = model::Tower::load(&model, config)?;
            eprintln!("loaded in {:.1}s", t0.elapsed().as_secs_f64());
            let tok = tokenizer::Tokenizer::load(&model)?;
            generate::set_eos_ids_for_model(tok.im_end_ids.clone(), &tower);

            // Warm on a short synthetic turn, then start the session fresh.
            let warm = tok.encode(
                "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n thinking\n",
                false,
            )?;
            tower.warmup(&warm)?;

            let mut session = session::Session::new(&mut tower);
            let mut turn = 0usize;
            let stdin = std::io::stdin();
            loop {
                use std::io::Write;
                print!("user> ");
                std::io::stdout().flush().ok();
                let mut line = String::new();
                if stdin.read_line(&mut line)? == 0 {
                    break;
                }
                let msg = line.trim_end_matches('\n').to_string();
                if msg.is_empty() {
                    continue;
                }
                // Suffix: close the previous assistant turn, then the new user
                // turn and a fresh assistant header (the template's own text).
                let suffix_ids =
                    tok.encode(&tokenizer::chat_turn_suffix(&msg, turn == 0), false)?;

                // Optional: prove the incremental prefill matches a full one at
                // this point (same token ids, fresh caches).
                if verify > 0 {
                    let all: Vec<u32> = session
                        .fed
                        .iter()
                        .copied()
                        .chain(suffix_ids.iter().copied())
                        .collect();
                    let (eq, dlog, dhid) = session::verify_incremental(&mut tower, &all, verify)?;
                    eprintln!(
                        "[verify] chunk={verify} argmax_equal={eq} logit_diff={dlog:.6} hidden_diff={dhid:.6}"
                    );
                }

                print!("assistant> ");
                std::io::stdout().flush().ok();
                let mut sampler =
                    build_sampler(temperature, top_k, top_p, min_p, rep_penalty, seed);
                let mut emit = |t: u32| -> anyhow::Result<()> {
                    print!("{}", tok.decode(&[t])?);
                    std::io::stdout().flush().ok();
                    Ok(())
                };
                let mdepth = lisa_engine::cli::resolve_mtp_depth(
                    depth,
                    sampler.greedy(),
                    tower.has_drafter(),
                )?;
                let out_tokens = if let Some(md) = mdepth {
                    session.generate_mtp_depth(
                        &mut tower,
                        &suffix_ids,
                        max_tokens,
                        md,
                        None,
                        if sampler.greedy() {
                            None
                        } else {
                            Some(&mut sampler)
                        },
                        &mut emit,
                    )?
                } else {
                    session.generate(
                        &mut tower,
                        &suffix_ids,
                        max_tokens,
                        &mut sampler,
                        pld,
                        None,
                        &mut emit,
                    )?
                };
                println!();
                eprintln!("[turn {turn}: {} tokens]", out_tokens.len());
                turn += 1;
            }
            Ok(())
        }

        Command::Serve {
            model,
            ctx_size,
            addr,
            max_tokens,
            temperature,
            top_p,
            top_k,
            min_p,
            rep_penalty,
            depth,
            mtp_exact_sampling,
            pld,
            pld_depth,
            prefix_cache,
            max_batch,
        } => {
            let depth_us = match depth {
                DepthArg::Auto => lisa_engine::core::round_cost::AUTO_DEPTH,
                DepthArg::Fixed(d) => d,
            };
            let cfg = serve::ServerConfig {
                pld: lisa_engine::cli::pld_enable(pld, pld_depth),
                prefix_cache,
                addr,
                max_tokens,
                temperature,
                top_p,
                top_k,
                min_p,
                rep_penalty,
                mtp_exact_sampling,
                gen_defaults: serve::GenDefaults::from_model_dir(&model),
                depth: depth_us,
                max_batch,
                chat_template: tokenizer::load_chat_template(&model)?,
                model_id: serve::model_id_of(&model),
                supports_images: false,
            };
            if let Some(t) = cfg.gen_defaults.thinking {
                eprintln!(
                    "[serve] generation_config: thinking default {}",
                    if t { "on" } else { "off" }
                );
            }
            eprintln!(
                "[serve] generation_config sampling: temp {:?} top_p {:?} top_k {:?}",
                cfg.gen_defaults.temperature, cfg.gen_defaults.top_p, cfg.gen_defaults.top_k
            );
            // Non-generative decision models get their own endpoint.
            if lisa_engine::models::model_type_of(&model)? == "laya" {
                anyhow::ensure!(
                    ctx_size.is_none(),
                    "--ctx-size requires a language model with configurable context; omit it for Laya"
                );
                anyhow::ensure!(
                    !mtp_exact_sampling,
                    "--mtp-exact-sampling requires a GLM model with target MTP"
                );
                let t0 = std::time::Instant::now();
                match lisa_engine::models::load_dir(&model)? {
                    lisa_engine::Loaded::Decision(m) => {
                        eprintln!("loaded in {:.1}s", t0.elapsed().as_secs_f64());
                        return serve::run_decisions(m, cfg);
                    }
                    _ => unreachable!("model_type was laya"),
                }
            }
            let t0 = std::time::Instant::now();
            let mut tower = match lisa_engine::models::load_dir(&model)? {
                lisa_engine::Loaded::Language(m) => m,
                lisa_engine::Loaded::Image(pipeline) => return serve::images::run(pipeline, cfg),
                lisa_engine::Loaded::Decision(_) => unreachable!("model_type was not laya"),
            };
            if let Some(capacity) = ctx_size {
                tower.set_context_capacity(capacity)?;
            }
            anyhow::ensure!(
                !mtp_exact_sampling || tower.has_target_mtp(),
                "--mtp-exact-sampling requires a GLM model with target MTP"
            );
            eprintln!("loaded in {:.1}s", t0.elapsed().as_secs_f64());
            mlx_mem_line("serve-loaded");
            let tok = tokenizer::Tokenizer::load(&model)?;
            generate::set_eos_ids_for_model(tok.im_end_ids.clone(), tower.as_ref());
            let warm = tok.encode("user\nhi\nassistant\n thinking\n", false)?;
            tower.warmup(&warm)?;
            // Freeze the numerical law now that the launch-time kernels are
            // compiled: caches key on it, and a lazily-compiled shape later in
            // the session must NOT change the identity (that emptied the prefix
            // cache mid-run — measured 683/710 resumes → 0 after one long
            // request).
            let _ = lisa_mlx::runtime::kernel_law();
            mlx_mem_line("serve-warm");
            serve::run(tower.as_mut(), &tok, cfg)
        }

        Command::Image {
            model,
            prompt,
            output,
            width,
            height,
            steps,
            seed,
            prompt_is_caption,
        } => {
            anyhow::ensure!(
                backend == lisa_mlx::backend::Backend::Metal,
                "image generation requires the Metal backend"
            );
            let pipeline = lisa_engine::models::image_generation::Pipeline::load(&model)?;
            let options = lisa_engine::models::image_generation::Options {
                width,
                height,
                steps,
                seed,
                prompt_is_caption,
            };
            let started = std::time::Instant::now();
            let png = pipeline.generate(&prompt, &options, &mut |stage, i, n| {
                eprintln!("{stage} {i}/{n}");
                Ok(())
            })?;
            std::fs::write(&output, png)?;
            println!(
                "{} ({width}x{height}, {:.2}s, seed {seed})",
                output.display(),
                started.elapsed().as_secs_f64()
            );
            Ok(())
        }
        Command::Run {
            model,
            ctx_size,
            mut prompt,
            image,
            system,
            max_tokens,
            temperature,
            raw,
            top_p,
            top_k,
            min_p,
            rep_penalty,
            seed,
            think_max,
            nothink,
            think,
            depth,
            mtp_exact_sampling,
            pld,
            pld_depth,
            ignore_eos,
            router,
            router_threshold,
            oracle,
        } => {
            if oracle {
                lisa_engine::core::oracle::enable();
            }
            let pld = lisa_engine::cli::pld_enable(pld, pld_depth);
            let t0 = std::time::Instant::now();
            let mut tower = match lisa_engine::models::load_dir(&model)? {
                lisa_engine::models::Loaded::Language(m) => m,
                lisa_engine::models::Loaded::Image(_)
                | lisa_engine::models::Loaded::Decision(_) => {
                    anyhow::bail!(
                        "run expects a language model; use lisa image for image generation"
                    )
                }
            };
            if let Some(capacity) = ctx_size {
                tower.set_context_capacity(capacity)?;
            }
            println!("loaded in {:.1}s", t0.elapsed().as_secs_f64());
            lisa_engine::ttft_mark("model.load done (load_dir returned)");
            let t_tok = std::time::Instant::now();
            let tok = tokenizer::Tokenizer::load(&model)?;
            lisa_engine::ttft_mark(&format!(
                "tokenizer load {:.2}s",
                t_tok.elapsed().as_secs_f64()
            ));
            generate::set_eos_ids_for_model(tok.im_end_ids.clone(), tower.as_ref());
            if ignore_eos {
                // `ignore_eos`: EOS never ends the reply — exactly max_tokens
                // forced out, both arms of an A/B.
                generate::set_eos_ids(Vec::new());
            }

            let is_glm = lisa_engine::models::model_type_of(&model)?
                .as_str()
                .starts_with("glm5");
            let is_deepseek = lisa_engine::models::model_type_of(&model)? == "deepseek41";
            anyhow::ensure!(
                !is_deepseek || (pld == 0 && !router),
                "DeepSeek GGUF currently supports serial generation; omit --pld and --router"
            );
            anyhow::ensure!(
                !mtp_exact_sampling || (is_glm && tower.has_target_mtp()),
                "--mtp-exact-sampling requires a GLM model with target MTP"
            );
            anyhow::ensure!(
                is_glm || is_deepseek || (!think_max && !nothink && !think),
                "--think, --think-max, and --nothink require GLM or DeepSeek"
            );
            let temperature = temperature.unwrap_or(if is_glm { 1.0 } else { 0.0 });
            let top_p = top_p.unwrap_or(if is_glm { 0.95 } else { 1.0 });
            let min_p = min_p.unwrap_or(0.0);
            let glm_reasoning = if think_max {
                tokenizer::GlmReasoning::Max
            } else if nothink {
                tokenizer::GlmReasoning::Disabled
            } else {
                tokenizer::GlmReasoning::High
            };
            let images = image
                .iter()
                .map(|path| {
                    let url = lisa_engine::models::vision::file_data_url(path)?;
                    lisa_engine::models::vision::decode_data_url(&url)
                })
                .collect::<anyhow::Result<Vec<_>>>()?;
            if !images.is_empty() {
                anyhow::ensure!(!raw, "--image cannot be combined with --raw");
                let marker = if is_glm {
                    "<|begin_of_image|><|image|><|end_of_image|>"
                } else {
                    "<|vision_start|><|image_pad|><|vision_end|>"
                };
                prompt = format!("{}{}", marker.repeat(images.len()), prompt);
            }
            let (text, special) = if raw {
                (prompt.clone(), false)
            } else if is_glm {
                (
                    tokenizer::glm_generation_prompt(
                        &prompt,
                        system.as_deref().or(Some("You are a helpful assistant")),
                        glm_reasoning,
                    ),
                    true,
                )
            } else if let Some(system) = system.as_deref() {
                (
                    tokenizer::generation_prompt(&[
                        ("system".to_string(), system.to_string()),
                        ("user".to_string(), prompt.clone()),
                    ]),
                    false,
                )
            } else {
                (
                    tokenizer::generation_prompt(&[("user".to_string(), prompt.clone())]),
                    false,
                )
            };
            let t_enc = std::time::Instant::now();
            let mut prompt_ids: Vec<u32> = if is_deepseek {
                if raw {
                    tok.encode_plain(&prompt)?
                } else {
                    tok.native_chat(
                        system.as_deref(),
                        &prompt,
                        if think_max {
                            100
                        } else if think {
                            75
                        } else {
                            0
                        },
                    )?
                }
            } else if raw {
                tok.encode_plain(&text)?
            } else if special {
                tok.encode_special(&text)?
            } else {
                tok.encode(&text, false)?
            };
            lisa_engine::ttft_mark(&format!(
                "tokenize {} ids in {:.2}s",
                prompt_ids.len(),
                t_enc.elapsed().as_secs_f64()
            ));
            if !images.is_empty() {
                tower.prepare_images(&mut prompt_ids, &images)?;
            }
            let has_images = !images.is_empty();
            drop(images);
            println!("prompt: {} tokens", prompt_ids.len());

            let mut sampler = build_sampler(temperature, top_k, top_p, min_p, rep_penalty, seed);
            sampler.mtp_exact_sampling = mtp_exact_sampling;
            let mut first_tok: Option<std::time::Instant> = None;
            let emit = &mut |t: u32| -> anyhow::Result<()> {
                if first_tok.is_none() {
                    first_tok = Some(std::time::Instant::now());
                    lisa_engine::ttft_mark("FIRST TOKEN (decode step returned)");
                }
                let piece = tok.decode(&[t])?;
                print!("{piece}");
                use std::io::Write;
                std::io::stdout().flush().ok();
                Ok(())
            };
            // Router (specs/router): score the prompt's 4-gram recurrence and
            // route. Measured decision table (27B, 2k-KV, 128 tok):
            //   echo:  serial 22.6 / pld 33.5 / mtp-auto 17.3 tok/s
            //   prose: serial 22.9 / pld 22.8 / mtp-auto 21.6
            //   mixed: serial 23.5 / pld 24.8 / mtp-auto 20.8
            // → at short KV PLD never loses (its copy-miss fallback IS a
            // serial step, and drafts are always verified), so it is the
            // default below LONG_KV. At long KV MTP wins on prose
            // (specs/03: auto 41.6 vs serial ~27.8 at 12.5k) while PLD keeps
            // winning on echo text (24.9→59.4), so above LONG_KV only a high
            // echo score overrides the --depth policy. Note the raw token
            // 4-gram score separates weakly (measured: echo 0.857, prose
            // 0.707, mixed 0.594) — the threshold guards the long-KV branch
            // only, where misrouting prose→pld is the costly direction.
            let mut pld = pld;
            let mut mdepth = lisa_engine::cli::resolve_mtp_depth(
                depth,
                tower.has_target_mtp() || sampler.greedy() || sampler.mtp_exact_sampling,
                tower.has_drafter() || tower.has_target_mtp(),
            )?;
            if router && sampler.greedy() {
                let echo = lisa_engine::core::copy_draft::echo_score(&prompt_ids);
                const LONG_KV: usize = 8192;
                let short = prompt_ids.len() < LONG_KV;
                let routed = echo >= router_threshold || short;
                eprintln!(
                    "[router] echo_score {:.3} threshold {:.3} prompt_len {} -> {}",
                    echo,
                    router_threshold,
                    prompt_ids.len(),
                    if routed { "pld" } else { "depth policy" }
                );
                if routed {
                    if pld == 0 {
                        pld = 4;
                    }
                    mdepth = None;
                }
            }
            if has_images {
                mdepth = None;
                pld = 0;
            }
            let (generated, stats) = if let Some(md) = mdepth {
                let mut sess = session::Session::new(&mut *tower);
                let t = std::time::Instant::now();
                let mtp_sampler =
                    if !tower.has_target_mtp() && sampler.greedy() && !sampler.mtp_exact_sampling {
                        None
                    } else {
                        Some(&mut sampler)
                    };
                let g = sess.generate_mtp_depth(
                    &mut *tower,
                    &prompt_ids,
                    max_tokens,
                    md,
                    None,
                    mtp_sampler,
                    emit,
                )?;
                let secs = t.elapsed().as_secs_f64();
                println!(
                    "prefill+decode: {:.1} tok/s ({:.2}s for {} tokens)",
                    g.len() as f64 / secs,
                    secs,
                    g.len()
                );
                (g, None)
            } else {
                let (g, s) = generate::generate(
                    &mut *tower,
                    &prompt_ids,
                    max_tokens,
                    &mut sampler,
                    pld,
                    emit,
                )?;
                (g, Some(s))
            };
            println!();
            match stats {
                Some(s) => println!(
                    "prefill: {:.1} tok/s ({:.2}s for {} tokens) | decode: {:.1} tok/s ({:.2}s for {} tokens)",
                    s.prefill_tps(),
                    s.prefill_seconds,
                    s.prompt_tokens,
                    s.decode_tps(),
                    s.decode_seconds,
                    s.generated_tokens,
                ),
                None => println!(
                    "mtp depth {depth}: {} tokens in this process",
                    generated.len()
                ),
            }
            let _ = generated;
            Ok(())
        }
    }
}
