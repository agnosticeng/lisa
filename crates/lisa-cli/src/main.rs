
use std::path::PathBuf;

use clap::{Parser, Subcommand};
use lisa_engine::{batch, generate, sampler, sched, session, tokenizer};
use lisa_engine::models::qwen4::{config, layerdiff, smoke};
use lisa_engine::models::qwen4::tower as model;
use lisa_engine::models::laya::{Laya, LayaDevice};
use lisa_engine::core::mem::mlx_mem_line;
use lisa_serve as serve;
use lisa_mlx::ops::indexing::IndexOp;
use lisa_mlx::Dtype;

/// `--model`: a local model directory or a Hugging Face repo id, resolved to a
/// local directory when the arguments are parsed.
#[derive(Clone)]
struct ModelDir(PathBuf);

impl std::ops::Deref for ModelDir {
    type Target = PathBuf;
    fn deref(&self) -> &PathBuf {
        &self.0
    }
}

impl AsRef<std::path::Path> for ModelDir {
    fn as_ref(&self) -> &std::path::Path {
        &self.0
    }
}

impl std::str::FromStr for ModelDir {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        lisa_engine::models::resolve_model_dir(s)
            .map(ModelDir)
            .map_err(|e| e.to_string())
    }
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
            let (k, v) = s
                .rsplit_once('=')
                .ok_or_else(|| anyhow::anyhow!("--temperature-by-options wants TYPE:SIZE=TEMP, got {s:?}"))?;
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

/// MTP draft depth: 0 = serial, 2..=6 = speculative. A single draft is never
/// worth a round, so depth 1 is rejected rather than silently routed.
fn parse_depth(s: &str) -> Result<usize, String> {
    let v: usize = s.parse().map_err(|e| format!("{e}"))?;
    if v == 1 {
        return Err("depth 1 is not supported; use 0 (serial) or 2..=6 (MTP)".to_string());
    }
    Ok(v)
}

#[derive(Parser)]
#[command(name = "lisa", about = "Rust MLX engine for Qwen 3.8 Flash-Next (qwen4_exp)")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Verify the compute stack: quantized matmul, fast ops, custom kernels.
    Smoke,
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
    },
    /// Run the public golden correctness check.
    Golden {
        #[arg(long)]
        model: ModelDir,
        #[arg(long)]
        golden: PathBuf,
        #[arg(long, default_value_t = 256)]
        max_tokens: usize,
        /// MTP speculative draft depth (0 = serial, 2..=6 = MTP; 1 is rejected).
        #[arg(long, default_value_t = 0, value_parser = parse_depth)]
        depth: usize,
    },
    /// Diff layer-0 intermediates against reference dumps.
    LayerDiff {
        #[arg(long)]
        model: ModelDir,
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
    /// Print top-8 first-token logits for a prompt.
    Logits {
        #[arg(long)]
        model: ModelDir,
        #[arg(long, default_value = "")]
        prompt: String,
        #[arg(long, default_value_t = false)]
        raw: bool,
    },
    /// KV cache write/read unit test.
    CacheTest,
    /// Negative slice indexing check.
    NegSlice,
    /// Rounding semantics checks.
    Rounding,
    /// Quantized matmul M>1 isolation.
    QmmM,
    /// Benchmark the NAX indirect expert GEMMs vs a dense quantized matmul.
    IndirectBench,
    /// Benchmark the S=1 decode MoE kernels (route/gate_up_act/down_combine).
    DecodeMoeBench,
    /// Verify incremental prefill (session prefix reuse) against a full prefill.
    SessionCheck {
        #[arg(long)]
        model: ModelDir,
        #[arg(long)]
        golden: PathBuf,
        #[arg(long, default_value_t = 16)]
        chunk: usize,
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
        /// MTP draft depth for the chat (0 = serial; 2..=6 = MTP; 1 is rejected).
        #[arg(long, default_value_t = 0, value_parser = parse_depth)]
        depth: usize,
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
    /// Repeat a golden prompt's prefill in-process for low-noise sampling.
    PrefillBench {
        #[arg(long)]
        model: ModelDir,
        #[arg(long)]
        golden: PathBuf,
        #[arg(long, default_value_t = 9)]
        iters: usize,
    },
    /// Continuous batching: streams admitted/retired on the fly (CBv2-style).
    Cbatch {
        #[arg(long)]
        model: ModelDir,
        /// One prompt per stream (repeat --prompt).
        #[arg(long = "prompt", num_args = 1..)]
        prompts: Vec<String>,
        #[arg(long, default_value_t = 2)]
        max_batch: usize,
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
        /// MTP draft depth (0 = continuous batching; 2..=6 = per-stream MTP; 1 is rejected).
        #[arg(long, default_value_t = 0, value_parser = parse_depth)]
        depth: usize,
    },
    /// Minimal OpenAI-compatible HTTP server (POST /v1/chat/completions, SSE).
    Serve {
        #[arg(long)]
        model: ModelDir,
        #[arg(long, default_value = "127.0.0.1:8080")]
        addr: String,
        #[arg(long, default_value_t = 256)]
        max_tokens: usize,
        #[arg(long, default_value_t = 0.0)]
        temperature: f32,
        #[arg(long, default_value_t = 1.0)]
        top_p: f32,
        #[arg(long, default_value_t = 0)]
        top_k: usize,
        #[arg(long, default_value_t = 0.0)]
        min_p: f32,
        #[arg(long, default_value_t = 1.0)]
        rep_penalty: f32,
        #[arg(long, default_value_t = 0, value_parser = parse_depth)]
        depth: usize,
        /// Maximum streams stepped together (continuous-batch width).
        #[arg(long, default_value_t = 4)]
        max_batch: usize,
    },
}

fn build_sampler(
    temperature: f32,
    top_k: usize,
    top_p: f32,
    min_p: f32,
    rep_penalty: f32,
    seed: u64,
) -> sampler::Sampler {
    let mut s = sampler::Sampler {
        temperature,
        top_k,
        top_p,
        min_p,
        repetition_penalty: rep_penalty,
        ..Default::default()
    };
    if seed != 0 {
        s.seed = seed;
    }
    s
}


fn main() -> anyhow::Result<()> {
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
    // Read at device creation, so set it before the model loads.
    if std::env::var_os("LISA_METAL_COMPUTE_PER_BUFFER").is_none() {
        // Safety: single-threaded here, before any device or worker thread.
        unsafe { std::env::set_var("LISA_METAL_COMPUTE_PER_BUFFER", "16") };
    }
    // Use an AOT metallib when present: identical GPU kernels = identical
    // arithmetic (the mean/col_reduce association is metallib-build-sensitive).
    if let Ok(meta) = std::env::var("LISA_METALLIB") {
        lisa_mlx::ffi::set_metallib_path(&meta).map_err(|e| anyhow::anyhow!("{e}"))?;
        println!("metallib: {meta}");
    }
    match cli.command {
        Command::Smoke => smoke::run_all(),
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
            let backend = match device.as_deref() {
                Some(d) => lisa_mlx::backend::Backend::from_name(d).map_err(|e| anyhow::anyhow!("{e}"))?,
                None => lisa_mlx::backend::Backend::from_env().map_err(|e| anyhow::anyhow!("{e}"))?,
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
                let qs = v.get("questions").cloned().unwrap_or(serde_json::Value::Null);
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
                config.num_hidden_layers,
                config.hidden_size,
                config.vocab_size
            );
            let t0 = std::time::Instant::now();
            let _tower = model::Tower::load(&model, config)?;
            println!("loaded in {:.1}s", t0.elapsed().as_secs_f64());
            mlx_mem_line("inspect");
            Ok(())
        }
        Command::LayerDiff { model } => layerdiff::run(&model),
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
        Command::Logits { model, prompt, raw } => {
            let config = config::ModelConfig::from_json(&model.join("config.json"))?;
            let mut tower = model::Tower::load(&model, config)?;
            let tok = tokenizer::Tokenizer::load(&model)?;
            let text = if raw {
                prompt.clone()
            } else {
                tokenizer::generation_prompt(&[("user".to_string(), prompt.clone())])
            };
            let ids = tok.encode(&text, false)?;
            println!("{} tokens", ids.len());
            let mut caches = tower.new_caches();
            let i32s: Vec<i32> = ids.iter().map(|&t| t as i32).collect();
            let arr = lisa_mlx::Array::from_slice(&i32s, &[1i32, i32s.len() as i32]);
            let (mixed, _) = tower.forward(&arr, Some(&mut caches))?;
            let last = mixed.index((.., mixed.dim(1) - 1, ..));
            let logits = tower.head(&last)?;
            let neg = -&logits;
            let top = lisa_mlx::ops::argpartition_axis(&neg, 7, -1)?;
            let top = top.index((.., 0..8));
            let vals = logits.take_along_axis(&top, -1)?;
            let _ = vals.eval();
            let toks: Vec<u32> = top.as_slice::<u32>().to_vec();
            let vf: Vec<f32> = vals
                .as_dtype(Dtype::Float32)
                .map_err(|e| anyhow::anyhow!("{e}"))?
                .as_slice::<f32>()
                .to_vec();
            for (t, v) in toks.iter().zip(vf.iter()) {
                let piece = tok.decode(&[*t]).unwrap_or_default();
                println!("  {:>7} {:>8.3} {:?}", t, v, piece);
            }
            Ok(())
        }
        Command::CacheTest => smoke::run_cache_test(),
        Command::NegSlice => smoke::run_negslice_test(),
        Command::Rounding => smoke::run_rounding_test(),
        Command::IndirectBench => {
            lisa_mlx::kernels::indirect_bench();
            Ok(())
        }
        Command::DecodeMoeBench => {
            lisa_mlx::moe_decode::decode_bench();
            Ok(())
        }
        Command::Batch { model, prompts, max_tokens, temperature, raw, top_p, top_k, min_p, rep_penalty, seed } => {
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
            let (_b, out) = batch::Batch::generate(&mut tower, &ids, max_tokens, &mut sampler, &mut |s, t| {
                acc[s].push_str(&tok.decode(&[t])?);
                Ok(())
            })?;
            for (i, o) in out.iter().enumerate() {
                println!("--- stream {i}: {} tokens in {:.2}s ---", o.len(), t0.elapsed().as_secs_f64());
                println!("{}", acc[i]);
            }
            Ok(())
        }
        Command::SessionCheck { model, golden, chunk } => {
            let data = std::fs::read_to_string(&golden)?;
            let g: serde_json::Value = serde_json::from_str(&data)?;
            let prompt: Vec<u32> = g["cases"][0]["prompt_tokens"]
                .as_array()
                .unwrap()
                .iter()
                .filter_map(|v| v.as_u64().map(|x| x as u32))
                .collect();
            let config = config::ModelConfig::from_json(&model.join("config.json"))?;
            let mut tower = model::Tower::load(&model, config)?;
            generate::set_eos_ids(vec![248_046, 248_044]);
            tower.warmup(&prompt[..8])?;
            let (eq, dlog, dhid) = session::verify_incremental(&mut tower, &prompt, chunk)?;
            println!("chunk={chunk} argmax_equal={eq} logit_diff={dlog:.6} hidden_diff={dhid:.6}");
            Ok(())
        }
        Command::Chat { model, max_tokens, temperature, verify, depth, top_p, top_k, min_p, rep_penalty, seed } => {
            let config = config::ModelConfig::from_json(&model.join("config.json"))?;
            let t0 = std::time::Instant::now();
            let mut tower = model::Tower::load(&model, config)?;
            eprintln!("loaded in {:.1}s", t0.elapsed().as_secs_f64());
            let tok = tokenizer::Tokenizer::load(&model)?;
            generate::set_eos_ids(tok.im_end_ids.clone());

            // Warm on a short synthetic turn, then start the session fresh.
            let warm = tok.encode("<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n thinking\n", false)?;
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
                let suffix_ids = tok.encode(&tokenizer::chat_turn_suffix(&msg, turn == 0), false)?;
                if std::env::var("LISA_DEBUG_SESSION").is_ok() {
                    eprintln!("[chat] suffix {} ids: {:?}", suffix_ids.len(), &suffix_ids[..suffix_ids.len().min(24)]);
                }

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
                    eprintln!("[verify] chunk={verify} argmax_equal={eq} logit_diff={dlog:.6} hidden_diff={dhid:.6}");
                }

                print!("assistant> ");
                std::io::stdout().flush().ok();
                let mut sampler = build_sampler(temperature, top_k, top_p, min_p, rep_penalty, seed);
                let mut emit = |t: u32| -> anyhow::Result<()> {
                    print!("{}", tok.decode(&[t])?);
                    std::io::stdout().flush().ok();
                    Ok(())
                };
                let out_tokens = if depth > 0 && sampler.greedy() {
                    session.generate_mtp(&mut tower, &suffix_ids, max_tokens, depth, &mut emit)?
                } else {
                    session.generate(&mut tower, &suffix_ids, max_tokens, &mut sampler, &mut emit)?
                };
                println!();
                if std::env::var("LISA_DEBUG_MTP").is_ok() {
                    eprintln!("[turn {turn} ids] {:?}", &out_tokens[..out_tokens.len().min(16)]);
                }
                eprintln!("[turn {turn}: {} tokens]", out_tokens.len());
                turn += 1;
            }
            Ok(())
        }

        Command::Cbatch {
            model,
            prompts,
            max_batch,
            max_tokens,
            temperature,
            raw,
            top_p,
            top_k,
            min_p,
            rep_penalty,
            seed,
            depth,
        } => {
            let config = config::ModelConfig::from_json(&model.join("config.json"))?;
            let mut tower = model::Tower::load(&model, config)?;
            let tok = tokenizer::Tokenizer::load(&model)?;
            generate::set_eos_ids(tok.im_end_ids.clone());
            let requests: Vec<sched::Request> = prompts
                .iter()
                .enumerate()
                .map(|(i, p)| {
                    sched::request_from_text(
                        i,
                        &tok,
                        p,
                        raw,
                        max_tokens,
                        build_sampler(temperature, top_k, top_p, min_p, rep_penalty, seed),
                    )
                })
                .collect::<anyhow::Result<_>>()?;
            tower.warmup(&requests[0].prompt)?;
            eprintln!("cbatch: {} streams, max_batch {max_batch}, depth {depth}", requests.len());
            let t0 = std::time::Instant::now();
            let mut acc: Vec<String> = vec![String::new(); requests.len()];
            let out = sched::run(&mut tower, requests, max_batch, depth, &mut |id, t| {
                acc[id].push_str(&tok.decode(&[t])?);
                Ok(())
            })?;
            for (i, o) in out.iter().enumerate() {
                println!("--- stream {i}: {} tokens ---", o.len());
                println!("{}", acc[i]);
            }
            eprintln!("cbatch done in {:.2}s", t0.elapsed().as_secs_f64());
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
            max_batch,
        } => {
            let cfg = serve::ServerConfig {
                addr,
                max_tokens,
                temperature,
                top_p,
                top_k,
                min_p,
                rep_penalty,
                depth,
                max_batch,
                chat_template: std::fs::read_to_string(model.join("chat_template.jinja")).ok(),
            };
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
            let config = config::ModelConfig::from_json(&model.join("config.json"))?;
            let t0 = std::time::Instant::now();
            let mut tower = model::Tower::load(&model, config)?;
            eprintln!("loaded in {:.1}s", t0.elapsed().as_secs_f64());
            let tok = tokenizer::Tokenizer::load(&model)?;
            generate::set_eos_ids(tok.im_end_ids.clone());
            let warm = tok.encode(
                "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n thinking\n",
                false,
            )?;
            tower.warmup(&warm)?;
            mlx_mem_line("serve-warm");
            serve::run(&mut tower, &tok, cfg)
        }

        Command::PrefillBench {
            model,
            golden,
            iters,
        } => {
            let data = std::fs::read_to_string(&golden)?;
            let g: serde_json::Value = serde_json::from_str(&data)?;
            let prompt: Vec<i32> = g["cases"][0]["prompt_tokens"]
                .as_array()
                .unwrap()
                .iter()
                .filter_map(|v| v.as_u64().map(|x| x as i32))
                .collect();
            let n = prompt.len();
            let config = config::ModelConfig::from_json(&model.join("config.json"))?;
            let t0 = std::time::Instant::now();
            let mut tower = model::Tower::load(&model, config)?;
            eprintln!("loaded in {:.1}s", t0.elapsed().as_secs_f64());
            generate::set_eos_ids(vec![248_046, 248_044]);
            let arr = lisa_mlx::Array::from_slice(&prompt, &[1i32, n as i32]);
            let mut times = Vec::new();
            for i in 0..iters {
                tower.ngram_history = None;
                let mut caches = tower.new_caches();
                let t = std::time::Instant::now();
                let (mixed, _) = tower.forward(&arr, Some(&mut caches))?;
                let last = mixed.index((.., mixed.dim(1) - 1, ..));
                let logits = tower.head(&last)?;
                let next = lisa_mlx::ops::indexing::argmax(&logits, None)?;
                next.eval().map_err(|e| anyhow::anyhow!("{e}"))?;
                let secs = t.elapsed().as_secs_f64();
                times.push(secs);
                println!("iter {i:2}: {:>7.1} tok/s ({:>7.1} ms)", n as f64 / secs, secs * 1e3);
            }
            times.sort_by(|a, b| a.partial_cmp(b).unwrap());
            println!(
                "median {:.1} tok/s | best {:.1} tok/s ({:.1} ms)",
                n as f64 / times[times.len() / 2],
                n as f64 / times[0],
                times[0] * 1e3
            );
            Ok(())
        }
        Command::QmmM => smoke::run_qmm_m_test(),
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
        } => {
            let t0 = std::time::Instant::now();
            let mut tower = match lisa_engine::models::load_dir(&model)? {
                lisa_engine::models::Loaded::Language(m) => m,
                lisa_engine::models::Loaded::Decision(_) => {
                    anyhow::bail!("run expects a language model")
                }
            };
            println!("loaded in {:.1}s", t0.elapsed().as_secs_f64());
            let tok = tokenizer::Tokenizer::load(&model)?;
            generate::set_eos_ids(tok.im_end_ids.clone());

            let text = if raw {
                prompt.clone()
            } else {
                tokenizer::generation_prompt(&[("user".to_string(), prompt.clone())])
            };
            let prompt_ids: Vec<u32> = if let Ok(s) = std::env::var("LISA_PROMPT_IDS") {
                s.split(',').filter_map(|x| x.trim().parse::<u32>().ok()).collect()
            } else {
                tok.encode(&text, false)?
            };
            println!("prompt: {} tokens", prompt_ids.len());

            let dump_ids = std::env::var("LISA_DUMP_IDS").is_ok();
            let mut sampler = build_sampler(temperature, top_k, top_p, min_p, rep_penalty, seed);
            let (generated, stats) = generate::generate(&mut *tower, &prompt_ids, max_tokens, &mut sampler, &mut |t| {
                if dump_ids {
                    eprintln!("ID {}", t);
                }
                let piece = tok.decode(&[t])?;
                print!("{piece}");
                use std::io::Write;
                std::io::stdout().flush().ok();
                Ok(())
            })?;
            println!();
            println!(
                "prefill: {:.1} tok/s ({:.2}s for {} tokens) | decode: {:.1} tok/s ({:.2}s for {} tokens)",
                stats.prefill_tps(),
                stats.prefill_seconds,
                stats.prompt_tokens,
                stats.decode_tps(),
                stats.decode_seconds,
                stats.generated_tokens,
            );
            let _ = generated;
            Ok(())
        }
        Command::Golden {
            model,
            golden,
            max_tokens,
            depth,
        } => {
            let data = std::fs::read_to_string(&golden)?;
            let g: serde_json::Value = serde_json::from_str(&data)?;
            let case = &g["cases"][0];
            let prompt: Vec<u32> = case["prompt_tokens"]
                .as_array()
                .unwrap()
                .iter()
                .filter_map(|v| v.as_u64().map(|x| x as u32))
                .collect();
            let expected: Vec<u32> = case["expected_tokens"]
                .as_array()
                .unwrap()
                .iter()
                .filter_map(|v| v.as_u64().map(|x| x as u32))
                .collect();

            let config = config::ModelConfig::from_json(&model.join("config.json"))?;
            let mut tower = model::Tower::load(&model, config)?;
            generate::set_eos_ids(vec![248_046, 248_044]);

            let truncate = std::env::var("LISA_TRUNC")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or(0);
            let mut prompt = &prompt[..prompt.len() - truncate];
            if let Ok(keep) = std::env::var("LISA_KEEP") {
                prompt = &prompt[..keep.parse::<usize>().unwrap()];
            }
            if let Ok(n) = std::env::var("LISA_DUMP_TOKENS") {
                prompt = &prompt[..n.parse::<usize>().unwrap()];
            }

            if std::env::var("LISA_DIAG").is_ok() {
                // per-position self-consistency over a window
                let mut caches = tower.new_caches();
                let ids: Vec<i32> = prompt.iter().map(|&t| t as i32).collect();
                let arr = lisa_mlx::Array::from_slice(&ids, &[1i32, ids.len() as i32]);
                let (mixed, _) = tower.forward(&arr, Some(&mut caches))?;
                let logits = tower.head(&mixed)?;
                let top1 = lisa_mlx::ops::argpartition_axis(&(-&logits), 0, -1)?;
                let top1 = top1.index((.., .., 0..1)).contiguous().unwrap();
                let _ = top1.eval();
                let flat = top1.as_slice::<u32>().to_vec();
                let start = std::env::var("LISA_FROM")
                    .ok()
                    .and_then(|v| v.parse::<usize>().ok())
                    .unwrap_or(prompt.len().saturating_sub(24));
                let end = std::env::var("LISA_TO")
                    .ok()
                    .and_then(|v| v.parse::<usize>().ok())
                    .unwrap_or(prompt.len() - 1);
                let mut agree = 0usize;
                let tot = end - start;
                for p in start..end {
                    let ok = flat[p] == prompt[p + 1];
                    if ok {
                        agree += 1;
                    }
                    println!(
                        "pos {p}: argmax {:>6} want {:>6} {}",
                        flat[p],
                        prompt[p + 1],
                        if ok { "OK" } else { "MISS" }
                    );
                }
                println!("agreement {agree}/{tot}");
            }

            if std::env::var("LISA_TOPK").is_ok() {
                let mut caches = tower.new_caches();
                let ids: Vec<i32> = prompt.iter().map(|&t| t as i32).collect();
                let arr = lisa_mlx::Array::from_slice(&ids, &[1i32, ids.len() as i32]);
                let (mixed, _) = tower.forward(&arr, Some(&mut caches))?;
                let last = mixed.index((.., mixed.dim(1) - 1, ..));
                let logits = tower.head(&last)?;
                if let Ok(dir) = std::env::var("LISA_DUMP_DIR") {
                    let f = logits.as_dtype(Dtype::Float32).map_err(|e| anyhow::anyhow!("{e}"))?;
                    let a = f.as_slice::<f32>();
                    let _ = std::fs::write(
                        format!("{dir}/rs_logits.bin"),
                        unsafe { std::slice::from_raw_parts(a.as_ptr() as *const u8, a.len() * 4) },
                    );
                }
                let top = lisa_mlx::ops::argpartition_axis(&(-&logits), 7, -1)?;
                let top = top.index((.., 0..8));
                let vals = logits.take_along_axis(&top, -1)?;
                let _ = vals.eval();
                let toks: Vec<u32> = top.as_slice::<u32>().to_vec();
                let vf: Vec<f32> = vals.as_dtype(Dtype::Float32).map_err(|e| anyhow::anyhow!("{e}"))?.as_slice::<f32>().to_vec();
                println!("RUST top8: {:?}", toks.iter().zip(vf.iter()).map(|(t,v)| (*t, (*v*100.0).round()/100.0)).collect::<Vec<_>>());
            }
            let n = max_tokens.min(expected.len());
            let (generated, prefill_tps, decode_tps, spec_line) = if depth > 0 {
                let mut sess = session::Session::new(&mut tower);
                let t0 = std::time::Instant::now();
                let g = sess.generate_mtp(&mut tower, &prompt, n, depth, &mut |_t| Ok(()))?;
                let secs = t0.elapsed().as_secs_f64();
                (g, 0.0, 0.0, Some(format!("mtp depth {depth} in {secs:.2}s")))
            } else {
                let mut greedy = sampler::Sampler::default();
                let (g, s) = generate::generate(&mut tower, &prompt, n, &mut greedy, &mut |_t| Ok(()))?;
                (g, s.prefill_tps(), s.decode_tps(), None)
            };
            let mut matches = 0usize;
            let mut first_bad = None;
            for (i, (a, b)) in generated.iter().zip(expected.iter()).enumerate() {
                if a == b {
                    matches += 1;
                } else if first_bad.is_none() {
                    first_bad = Some(i);
                }
            }
            println!(
                "tokens: {} generated / {} expected | matches {}/{} ({:.1}%) | first mismatch at {:?}",
                generated.len(),
                expected.len(),
                matches,
                generated.len(),
                100.0 * matches as f64 / generated.len() as f64,
                first_bad,
            );
            println!("expected head:  {:?}", &expected[..expected.len().min(12)]);
            println!("generated head: {:?}", &generated[..generated.len().min(12)]);
            if let Some(line) = spec_line {
                println!("{line}");
            }
            println!("prefill: {:.1} tok/s | decode: {:.1} tok/s", prefill_tps, decode_tps);
            Ok(())
        }
    }
}
