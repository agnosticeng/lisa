use std::path::PathBuf;

use clap::{Parser, Subcommand};
use lisa_engine::cli::{DepthArg, ModelDir, build_sampler};
use lisa_engine::core::mem::mlx_mem_line;
use lisa_engine::models::LanguageModel;
use lisa_engine::models::laya::{Laya, LayaDevice};
use lisa_engine::models::qwen4::tower as model;
use lisa_engine::models::qwen4::config;
use lisa_engine::{batch, generate, session, tokenizer};
use lisa_serve as serve;

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
    about = "Rust MLX engine for Qwen 3.8 Flash-Next (qwen4_exp)"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Load a model and print load-time diagnostics only.
    Inspect {
        #[arg(long)]
        model: ModelDir,
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
    /// Load a model and generate.
    Run {
        #[arg(long)]
        model: ModelDir,
        #[arg(long, default_value = "Explain what a MoE layer is in two sentences.")]
        prompt: String,
        #[arg(long, default_value_t = 128)]
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
        /// MTP draft depth for the run (`auto` = EV controller, the default;
        /// 0 = serial; 2..=7 = fixed MTP; greedy only).
        #[arg(long, default_value = "auto")]
        depth: DepthArg,
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
    /// Cohort batching: N prompts decoded together in one batched forward.
    Batch {
        #[arg(long)]
        model: ModelDir,
        /// One prompt per stream (repeat --prompt).
        #[arg(long = "prompt", num_args = 1..)]
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
    /// Multi-turn chat (incremental prefill; caches persist across turns).
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
        /// 0 = serial; 2..=7 = fixed MTP; 1 is rejected).
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
    // The GPU command buffer is committed every 50 dispatches by default,
    // which is late enough that the MTP head's sequential draft steps do not
    // overlap the CPU graph build. Committing at ~16 recovers ~2 ms of the
    // draft (d5 round 69.6 -> 67.4 ms); the boundary does not change results.
    match cli.command {
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
        Command::Inspect { model } => {
            let config = config::ModelConfig::from_json(&model.join("config.json"))?;
            println!(
                "config: {} layers, hidden {}, vocab {}",
                config.num_hidden_layers, config.hidden_size, config.vocab_size
            );
            let t0 = std::time::Instant::now();
            let _tower = model::Tower::load(&model, config)?;
            println!("loaded in {:.1}s", t0.elapsed().as_secs_f64());
            mlx_mem_line("inspect");
            Ok(())
        }
        Command::Tok { model, prompt, raw } => {
            let tok = tokenizer::Tokenizer::load(&model)?;
            let text = if raw {
                prompt.clone()
            } else {
                tokenizer::generation_prompt(&[("user".to_string(), prompt.clone())])
            };
            let ids = tok.encode(&text, false)?;
            println!("ids: {:?}", &ids[..ids.len().min(40)]);
            for want in ["<|im_start|>", "<|im_end|>", "<think>"] {
                println!("{want} -> {:?}", tok.inner.token_to_id(want));
            }
            Ok(())
        }
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
            let config = config::ModelConfig::from_json(&model.join("config.json"))?;
            let mut tower = model::Tower::load(&model, config)?;
            let tok = tokenizer::Tokenizer::load(&model)?;
            generate::set_eos_ids(tok.im_end_ids.clone());
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
            let pld = lisa_engine::cli::pld_enable(pld, pld_depth);
            let config = config::ModelConfig::from_json(&model.join("config.json"))?;
            let t0 = std::time::Instant::now();
            let mut tower = model::Tower::load(&model, config)?;
            eprintln!("loaded in {:.1}s", t0.elapsed().as_secs_f64());
            let tok = tokenizer::Tokenizer::load(&model)?;
            generate::set_eos_ids(tok.im_end_ids.clone());

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
            addr,
            max_tokens,
            temperature,
            top_p,
            top_k,
            min_p,
            rep_penalty,
            depth,
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
                gen_defaults: serve::GenDefaults::from_model_dir(&model),
                depth: depth_us,
                max_batch,
                chat_template: std::fs::read_to_string(model.join("chat_template.jinja")).ok(),
                model_id: serve::model_id_of(&model),
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
                let t0 = std::time::Instant::now();
                match lisa_engine::models::load_dir(&model)? {
                    lisa_engine::Loaded::Decision(m) => {
                        eprintln!("loaded in {:.1}s", t0.elapsed().as_secs_f64());
                        return serve::run_decisions(m, cfg);
                    }
                    lisa_engine::Loaded::Language(_) => unreachable!("model_type was laya"),
                }
            }
            let t0 = std::time::Instant::now();
            let mut tower = match lisa_engine::models::load_dir(&model)? {
                lisa_engine::Loaded::Language(m) => m,
                lisa_engine::Loaded::Decision(_) => unreachable!("model_type was not laya"),
            };
            eprintln!("loaded in {:.1}s", t0.elapsed().as_secs_f64());
            mlx_mem_line("serve-loaded");
            let tok = tokenizer::Tokenizer::load(&model)?;
            generate::set_eos_ids(tok.im_end_ids.clone());
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

        Command::Run {
            model,
            prompt,
            max_tokens,
            temperature,
            raw,
            top_p,
            top_k,
            min_p,
            rep_penalty,
            seed,
            depth,
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
                lisa_engine::models::Loaded::Decision(_) => {
                    anyhow::bail!("run expects a language model")
                }
            };
            println!("loaded in {:.1}s", t0.elapsed().as_secs_f64());
            lisa_engine::ttft_mark("model.load done (load_dir returned)");
            let t_tok = std::time::Instant::now();
            let tok = tokenizer::Tokenizer::load(&model)?;
            lisa_engine::ttft_mark(&format!(
                "tokenizer load {:.2}s",
                t_tok.elapsed().as_secs_f64()
            ));
            generate::set_eos_ids(tok.im_end_ids.clone());
            if ignore_eos {
                // `ignore_eos`: EOS never ends the reply — exactly max_tokens
                // forced out, both arms of an A/B.
                generate::set_eos_ids(Vec::new());
            }

            let text = if raw {
                prompt.clone()
            } else {
                tokenizer::generation_prompt(&[("user".to_string(), prompt.clone())])
            };
            let t_enc = std::time::Instant::now();
            let prompt_ids: Vec<u32> = tok.encode(&text, false)?;
            lisa_engine::ttft_mark(&format!(
                "tokenize {} ids in {:.2}s",
                prompt_ids.len(),
                t_enc.elapsed().as_secs_f64()
            ));
            println!("prompt: {} tokens", prompt_ids.len());

            let mut sampler = build_sampler(temperature, top_k, top_p, min_p, rep_penalty, seed);
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
                sampler.greedy(),
                tower.has_drafter(),
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
            let (generated, stats) = if let Some(md) = mdepth {
                let mut sess = session::Session::new(&mut *tower);
                let t = std::time::Instant::now();
                let g = sess.generate_mtp_depth(
                    &mut *tower,
                    &prompt_ids,
                    max_tokens,
                    md,
                    None,
                    if sampler.greedy() {
                        None
                    } else {
                        Some(&mut sampler)
                    },
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
                anyhow::ensure!(
                    depth == DepthArg::Fixed(0) || router,
                    "--depth requires greedy sampling (temperature 0)"
                );
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
