use lisa_engine::checkpoint::config::ReadConfig;

use std::path::PathBuf;

use clap::{Parser, Subcommand};
use lisa_models::qwen4::config;

use lisa_engine::cli::{DepthArg, ModelDir, build_sampler};
use lisa_engine::language::LanguageModel;
use lisa_engine::{generation as generate, tokenizer};
use lisa_serve as serve;
use lisa_serve::application::memory_line as mlx_mem_line;
use lisa_serve::application::{Batch, Session};

/// These commands currently use Qwen4-specific loading and turn framing.
fn require_qwen4_command(model: &std::path::Path, command: &str) -> anyhow::Result<()> {
    let family = lisa_engine::checkpoint::model_type_of(model)?;
    anyhow::ensure!(
        matches!(family.as_str(), "qwen4_exp" | "qwen4_exp_text"),
        "`lisa {command}` currently supports Qwen4 checkpoints; use `run` or `serve` for {family}"
    );
    Ok(())
}

fn checkpoint_prompt(
    model: &std::path::Path,
    prompt: &str,
    system: Option<&str>,
    thinking: bool,
    images: usize,
) -> anyhow::Result<String> {
    let source = lisa_engine::checkpoint::tokenizer::load_chat_template(model)?
        .ok_or_else(|| anyhow::anyhow!("GGUF has no chat template"))?;
    let template = lisa_serve::ChatTemplate::new(&source)?;
    let mut messages = Vec::new();
    for (role, content) in system
        .map(|s| ("system", s))
        .into_iter()
        .chain(std::iter::once(("user", prompt)))
    {
        messages.push(lisa_serve::ChatMessage {
            role: role.into(),
            content: content.into(),
            reasoning_content: None,
            tool_calls: vec![],
            tool_call_id: None,
            parts: None,
        });
    }
    if images > 0 {
        let mut parts = (0..images)
            .map(|_| serde_json::json!({"type":"image"}))
            .collect::<Vec<_>>();
        parts.push(serde_json::json!({"type":"text","text":prompt}));
        messages.last_mut().unwrap().parts = Some(parts);
    }
    template.render(&messages, true, &[], thinking)
}

fn tokenize(model: &std::path::Path, prompt: &str, raw: bool) -> anyhow::Result<()> {
    let tok = lisa_engine::checkpoint::tokenizer::load(model)?;
    if lisa_engine::checkpoint::model_type_of(model)? == "deepseek41" {
        let ids = if raw {
            tok.encode_plain(prompt)?
        } else {
            tok.native_chat(None, prompt, 0)?
        };
        println!("ids: {:?}", &ids[..ids.len().min(40)]);
        return Ok(());
    }
    let is_glm = matches!(
        lisa_engine::checkpoint::model_type_of(model)?.as_str(),
        "glm5_next" | "glm5_next_text" | "glm5-next"
    );
    let text = if raw {
        prompt.to_owned()
    } else if matches!(
        lisa_engine::checkpoint::model_type_of(model)?.as_str(),
        "olala" | "inkling2"
    ) {
        checkpoint_prompt(model, prompt, None, false, 0)?
    } else if is_glm {
        tokenizer::glm_generation_prompt(
            prompt,
            Some("You are a helpful assistant"),
            tokenizer::ReasoningEffort::High,
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
    /// Compute device: `metal`, `cpu` or `auto` (Metal when present).
    #[arg(long, global = true)]
    device: Option<String>,
    /// Cap the RAM budget in GiB; it can only lower the detected budget and
    /// also arms HTTP request admission.
    #[arg(long, global = true, value_parser = clap::value_parser!(u64).range(1..))]
    ram_cap_gb: Option<u64>,
    /// Key/value cache format: `auto` (8-bit where the model implements it),
    /// `native` (each model's own) or `int8` (required; others refuse to load).
    #[arg(long, global = true, default_value = "auto", value_parser = ["auto", "native", "int8"])]
    kv_format: String,
    /// Most rows a greedy tree round verifies (1 to 16). Each round chooses
    /// its rows from measured costs and learned acceptance, up to this cap.
    #[arg(long, global = true, default_value_t = 16, value_parser = clap::value_parser!(u64).range(1..=16))]
    tree_nodes: u64,
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
        /// Override prompt + output context capacity (default: checkpoint metadata).
        #[arg(long, value_parser = parse_context_capacity)]
        ctx_size: Option<usize>,
        #[arg(long, default_value = "Explain what a MoE layer is in two sentences.")]
        prompt: String,
        /// PNG, JPEG or WebP image; repeat for multiple images.
        #[arg(long)]
        image: Vec<std::path::PathBuf>,
        /// System message; GLM defaults to the reference's "You are a helpful assistant".
        #[arg(long)]
        system: Option<String>,
        #[arg(long, default_value_t = 128)]
        max_tokens: usize,
        /// Sampling temperature; GLM defaults to the reference's 1.0, other models to 0.
        #[arg(long)]
        temperature: Option<f32>,
        #[arg(long, default_value_t = false)]
        raw: bool,
        /// Nucleus sampling; GLM defaults to the reference's 0.95, other models to 1.0.
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
        /// Use the reference's maximum GLM reasoning effort instead of its default High.
        #[arg(long, conflicts_with = "nothink")]
        think_max: bool,
        /// Close an empty GLM think block and omit the reasoning-effort prefix.
        #[arg(long, conflicts_with = "think_max")]
        nothink: bool,
        /// Explicitly keep the default High GLM reasoning effort.
        #[arg(long, conflicts_with_all = ["think_max", "nothink"])]
        think: bool,
        /// MTP policy (`auto` = Qwen EV controller or GLM target MTP;
        /// 0 = serial; 2..=6 = fixed native Qwen depth or GLM target MTP;
        /// model-specific limits apply).
        #[arg(long, default_value = "auto")]
        depth: DepthArg,
        /// Use the reference's exact stochastic acceptance and residual sampling for GLM MTP.
        #[arg(long, default_value_t = false)]
        mtp_exact_sampling: bool,
        /// Ignore EOS: generate exactly max_tokens (the clean-throughput
        /// measurement arm).
        #[arg(long, default_value_t = false)]
        ignore_eos: bool,
        /// Draft-ranking oracle: rank the target's true token in
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
        /// 0 = serial; 2..=6 = fixed native MTP; model-specific limits apply;
        /// 1 is rejected; speculative chat requires greedy sampling).
        #[arg(long, default_value = "auto")]
        depth: DepthArg,
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
        /// Override prompt + output context capacity (default: checkpoint metadata).
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
        /// Use the reference's exact stochastic acceptance and residual sampling for GLM MTP.
        #[arg(long, default_value_t = false)]
        mtp_exact_sampling: bool,
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
    lisa_metal::trace::finish();
    result
}

fn run() -> anyhow::Result<()> {
    let cli = Cli::parse();
    // Weight preparation never initializes a Metal device or allocator.
    if let Command::Prepare { model, plan_only } = &cli.command {
        let summary = lisa_engine::weights::prepare::prepare_model(model, *plan_only)?;
        println!("{}", serde_json::to_string_pretty(&summary)?);
        return Ok(());
    }
    if let Command::Tok { model, prompt, raw } = &cli.command {
        return tokenize(model, prompt, *raw);
    }
    if let Command::Inspect { model, header_only } = &cli.command {
        if model.is_file() {
            use lisa_gguf::Header;
            let header = Header::open(model)?;
            println!("GGUF architecture: {}", header.architecture()?);
            if header.architecture()? == "deepseek41" {
                let config = lisa_engine::checkpoint::deepseek41::config(&header)?;
                let engram = lisa_engine::checkpoint::deepseek41::engram_layout(&header)?;
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
                let package = lisa_engine::checkpoint::deepseek41::open_package(model)?;
                println!("{}", package.summary()?);
                if header.architecture()? == "qwen35" {
                    let resident = lisa_engine::weights::qwen35::gguf::inspect(model)?;
                    println!(
                        "Qwen 27B GGUF: 64 trunk layers, embedded MTP, text only; {:.2} GiB resident weights. Experimental F32 graph.",
                        resident as f64 / (1u64 << 30) as f64
                    );
                }
                if header.architecture()? == "glm5-next" {
                    let c = lisa_engine::checkpoint::glm53::validate(&package)?;
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
                    use lisa_engine::weights::deepseek41::plan::WeightPlan;
                    let plan = WeightPlan::inspect(
                        &package,
                        &lisa_engine::checkpoint::deepseek41::config(&header)?,
                        &lisa_engine::checkpoint::deepseek41::engram_layout(&header)?,
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
    // Select the device backend (Metal when present unless `--device` says
    // otherwise) and the RAM cap before anything loads.
    if let Some(gib) = cli.ram_cap_gb {
        lisa_engine::memory::set_ram_cap_gib(gib as usize);
    }
    lisa_engine::memory::set_kv_format(match cli.kv_format.as_str() {
        "native" => lisa_engine::memory::KvFormat::Native,
        "int8" => lisa_engine::memory::KvFormat::Int8,
        _ => lisa_engine::memory::KvFormat::Auto,
    });
    lisa_engine::loading::set_tree_nodes(cli.tree_nodes as usize);
    let backend = lisa_serve::application::device::Backend::resolve(cli.device.as_deref())?;
    backend.select()?;
    if backend != lisa_serve::application::device::Backend::Metal {
        eprintln!("device: {}", backend.name());
    }
    lisa_metal::ffi::install_error_handler();
    // Cap how far the Metal buffer pool may grow past load-steady-state before
    // a sweep is forced. Without a cap a pipelined prefill or MTP round pins
    // every temporary it ever made (buffers are only released at a flush).
    lisa_metal::memory::set_cache_limit(8 << 30);
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
            top_k,
        } => {
            let model_type = lisa_engine::checkpoint::model_type_of(&model)?;
            anyhow::ensure!(
                model_type == "laya",
                "{} has model_type {model_type:?}, not a decision model",
                model.display()
            );
            let mut laya = lisa_engine::weights::laya::load_on(
                &model,
                backend.open()?,
                backend.laya_arithmetic(),
            )?;
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
            let _tower = lisa_serve::application::loading::load_dir(&model)?;
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
            let mut tower = lisa_serve::application::load_qwen4(&model, config)?;
            let tok = lisa_engine::checkpoint::tokenizer::load(&model)?;
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
                Batch::generate(&mut tower, &ids, max_tokens, &mut sampler, &mut |s, t| {
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
            top_p,
            top_k,
            min_p,
            rep_penalty,
            seed,
        } => {
            require_qwen4_command(&model, "chat")?;
            let config = config::ModelConfig::from_json(&model.join("config.json"))?;
            let t0 = std::time::Instant::now();
            let mut tower = lisa_serve::application::load_qwen4(&model, config)?;
            eprintln!("loaded in {:.1}s", t0.elapsed().as_secs_f64());
            let tok = lisa_engine::checkpoint::tokenizer::load(&model)?;
            generate::set_eos_ids_for_model(tok.im_end_ids.clone(), &tower);

            // Warm on a short synthetic turn, then start the session fresh.
            let warm = tok.encode(
                "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n thinking\n",
                false,
            )?;
            tower.warmup(&warm)?;

            let mut session = Session::new(&mut tower);
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
                    let (eq, dlog, dhid) =
                        lisa_engine::session_check::verify_incremental(&mut tower, &all, verify)?;
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
                        if sampler.can_use_raw_argmax() {
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
            prefix_cache,
            max_batch,
        } => {
            let depth_us = match depth {
                DepthArg::Auto => lisa_engine::core::round_cost::AUTO_DEPTH,
                DepthArg::Fixed(d) => d,
            };
            let cfg = serve::ServerConfig {
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
                chat_template: lisa_engine::checkpoint::tokenizer::load_chat_template(&model)?,
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
            if lisa_engine::checkpoint::model_type_of(&model)? == "laya" {
                anyhow::ensure!(
                    ctx_size.is_none(),
                    "--ctx-size requires a language model with configurable context; omit it for Laya"
                );
                anyhow::ensure!(
                    !mtp_exact_sampling,
                    "--mtp-exact-sampling requires a GLM model with target MTP"
                );
                let t0 = std::time::Instant::now();
                match lisa_serve::application::loading::load_dir(&model)? {
                    lisa_serve::application::Loaded::Decision(m) => {
                        eprintln!("loaded in {:.1}s", t0.elapsed().as_secs_f64());
                        return serve::run_decisions(m, cfg);
                    }
                    _ => unreachable!("model_type was laya"),
                }
            }
            let t0 = std::time::Instant::now();
            let mut tower = match lisa_serve::application::loading::load_dir(&model)? {
                lisa_serve::application::Loaded::Language(m) => m,
                lisa_serve::application::Loaded::Image(pipeline) => {
                    return serve::images::run(pipeline, cfg);
                }
                lisa_serve::application::Loaded::Decision(_) => {
                    unreachable!("model_type was not laya")
                }
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
            let tok = lisa_engine::checkpoint::tokenizer::load(&model)?;
            generate::set_eos_ids_for_model(tok.im_end_ids.clone(), tower.as_ref());
            let warm = tok.encode("user\nhi\nassistant\n thinking\n", false)?;
            tower.warmup(&warm)?;
            // Freeze the numerical law now that the launch-time kernels are
            // compiled: caches key on it, and a lazily-compiled shape later in
            // the session must NOT change the identity (that emptied the prefix
            // cache mid-run — measured 683/710 resumes → 0 after one long
            // request).
            let _ = lisa_metal::runtime::kernel_law();
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
                backend == lisa_serve::application::device::Backend::Metal,
                "image generation requires the Metal backend"
            );
            let pipeline = lisa_serve::application::ImagePipeline::load(
                lisa_serve::application::context()?,
                &model,
            )?;
            let options = lisa_engine::image_generation::Options {
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
            ignore_eos,
            oracle,
        } => {
            if oracle {
                lisa_engine::core::oracle::enable();
            }
            let t0 = std::time::Instant::now();
            let mut tower = match lisa_serve::application::loading::load_dir(&model)? {
                lisa_serve::application::loading::Loaded::Language(m) => m,
                lisa_serve::application::loading::Loaded::Image(_)
                | lisa_serve::application::loading::Loaded::Decision(_) => {
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
            let tok = lisa_engine::checkpoint::tokenizer::load(&model)?;
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

            let is_glm = lisa_engine::checkpoint::model_type_of(&model)?
                .as_str()
                .starts_with("glm5");
            let is_deepseek = lisa_engine::checkpoint::model_type_of(&model)? == "deepseek41";
            let is_inkling = lisa_engine::checkpoint::model_type_of(&model)? == "inkling2";
            let has_control_frames =
                lisa_engine::checkpoint::model_type_of(&model)? == "olala" || is_inkling;
            anyhow::ensure!(
                !mtp_exact_sampling || (is_glm && tower.has_target_mtp()),
                "--mtp-exact-sampling requires a GLM model with target MTP"
            );
            anyhow::ensure!(
                is_glm || is_deepseek || has_control_frames || (!think_max && !nothink && !think),
                "--think, --think-max, and --nothink require a supported reasoning model"
            );
            let temperature = temperature.unwrap_or(if is_glm { 1.0 } else { 0.0 });
            let top_p = top_p.unwrap_or(if is_glm { 0.95 } else { 1.0 });
            let min_p = min_p.unwrap_or(0.0);
            let glm_reasoning = if think_max {
                tokenizer::ReasoningEffort::Max
            } else if nothink {
                tokenizer::ReasoningEffort::Disabled
            } else {
                tokenizer::ReasoningEffort::High
            };
            let images = image
                .iter()
                .map(|path| {
                    let url = lisa_engine::vision::file_data_url(path)?;
                    lisa_engine::vision::decode_data_url(&url)
                })
                .collect::<anyhow::Result<Vec<_>>>()?;
            anyhow::ensure!(
                images.is_empty() || !raw,
                "--image cannot be combined with --raw"
            );
            if !images.is_empty() && !is_inkling {
                let marker = if is_glm {
                    "<|begin_of_image|><|image|><|end_of_image|>"
                } else {
                    "<|vision_start|><|image_pad|><|vision_end|>"
                };
                prompt = format!("{}{}", marker.repeat(images.len()), prompt);
            }
            let (text, special) = if raw {
                (prompt.clone(), false)
            } else if has_control_frames {
                (
                    checkpoint_prompt(
                        &model,
                        &prompt,
                        system.as_deref(),
                        think || think_max,
                        if is_inkling { images.len() } else { 0 },
                    )?,
                    false,
                )
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
            } else if tok.is_lfm3() {
                tok.encode(
                    &lisa_models::lfm3::prompt::turn(
                        system.as_deref(),
                        &prompt,
                        true,
                        think || think_max,
                    ),
                    false,
                )?
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
            let output_template = if has_control_frames && !raw {
                Some(lisa_serve::ChatTemplate::new(
                    &lisa_engine::checkpoint::tokenizer::load_chat_template(&model)?
                        .ok_or_else(|| anyhow::anyhow!("missing checkpoint template"))?,
                )?)
            } else {
                None
            };
            let mut decoder = tokenizer::IncrementalDecoder::new(&tok, false);
            let mut displayed = String::new();
            let emit = &mut |t: u32| -> anyhow::Result<()> {
                if first_tok.is_none() {
                    first_tok = Some(std::time::Instant::now());
                    lisa_engine::ttft_mark("FIRST TOKEN (decode step returned)");
                }
                if let Some(template) = &output_template {
                    decoder.push(t)?;
                    let (_, content) =
                        template.streaming_sections(decoder.stable_text(), think || think_max);
                    if let Some(delta) = content.strip_prefix(&displayed) {
                        print!("{delta}");
                    }
                    displayed = content.to_owned();
                } else {
                    print!("{}", tok.decode(&[t])?);
                }
                use std::io::Write;
                std::io::stdout().flush().ok();
                Ok(())
            };
            let mut mdepth = lisa_engine::cli::resolve_mtp_depth(
                depth,
                tower.has_target_mtp() || sampler.greedy() || sampler.mtp_exact_sampling,
                tower.has_drafter() || tower.has_target_mtp(),
            )?;
            if has_images {
                mdepth = None;
            }
            let (generated, stats) = if let Some(md) = mdepth {
                let mut sess = Session::new(&mut *tower);
                let t = std::time::Instant::now();
                let mtp_sampler = if !tower.has_target_mtp()
                    && sampler.can_use_raw_argmax()
                    && !sampler.mtp_exact_sampling
                {
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
                let (g, s) =
                    generate::generate(&mut *tower, &prompt_ids, max_tokens, &mut sampler, emit)?;
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
