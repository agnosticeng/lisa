use std::path::PathBuf;

use clap::{Parser, Subcommand};
use lisa_engine::cli::{DepthArg, ModelDir, parse_depth, pld_enable};
use lisa_engine::core::prefix_cache;
use lisa_engine::models::qwen4::tower as model;
use lisa_engine::models::qwen4::{config, layerdiff, smoke};
use lisa_engine::models::{self, Loaded};
use lisa_engine::{generate, sampler, sched, session, tokenizer};
use lisa_mlx::ops::indexing::IndexOp;

mod glm53_run;

#[derive(Parser)]
#[command(
    name = "lisa-bench",
    about = "lisa test / parity / benchmark commands (golden, smoke, benches)"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Time one production GLM serial/MTP generation and save token IDs as JSON.
    Glm53Run(glm53_run::Args),
    /// Verify the compute stack: quantized matmul, fast ops, custom kernels.
    Smoke,
    /// Run the public golden correctness check.
    Golden {
        /// A local model directory or a HF repo id (resolved against the hub
        /// cache only — the golden never triggers a download).
        #[arg(long)]
        model: String,
        /// Golden JSON. Defaults to the tracked qwen3_5 golden
        /// (`crates/lisa-bench/golden/qwen3_5.json`, embedded at compile time).
        #[arg(long)]
        golden: Option<PathBuf>,
        #[arg(long, default_value_t = 256)]
        max_tokens: usize,
        /// MTP speculative draft depth (0 = serial, 2..=7 = MTP; 1 is rejected).
        /// The golden deliberately stays SERIAL by default: it is a parity tool
        /// (the tracked baseline was captured serially), not a benchmark.
        #[arg(long, default_value_t = 0, value_parser = parse_depth)]
        depth: usize,
        /// Prompt-lookup (PLD) speculative decode (greedy only).
        #[arg(long, default_value_t = false)]
        pld: bool,
        /// PLD draft depth (used when --pld is set; 0 = default 4).
        #[arg(long, default_value_t = 0)]
        pld_depth: usize,
        /// Instead of comparing, generate greedily and write the tokens back
        /// into the golden file as the baseline (qwen3_5 capture mode; needs
        /// an explicit --golden to write into).
        #[arg(long, default_value_t = false)]
        capture: bool,
    },
    /// Diff layer-0 intermediates against reference dumps.
    LayerDiff {
        #[arg(long)]
        model: ModelDir,
    },
    /// Print top-8 first-token logits for a prompt; optionally dump the full vector as JSON.
    Logits {
        #[arg(long)]
        model: ModelDir,
        #[arg(long, default_value = "")]
        prompt: String,
        #[arg(long, default_value_t = false)]
        raw: bool,
        /// Write prompt IDs and the full first-token logit vector as JSON.
        #[arg(long)]
        dump_logits: Option<PathBuf>,
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
    /// Prove the cross-request prefix cache: the greedy continuation from a
    /// restored prefix state must equal the full-prefill continuation.
    PrefixCheck {
        #[arg(long)]
        model: ModelDir,
        #[arg(long)]
        golden: PathBuf,
        #[arg(long, default_value_t = 310)]
        tokens: usize,
        /// The uncached tail of the prompt (the suffix prefilled on a hit).
        #[arg(long, default_value_t = 64)]
        suffix: usize,
    },
    /// Verify incremental prefill (session prefix reuse) against a full prefill.
    SessionCheck {
        #[arg(long)]
        model: ModelDir,
        #[arg(long)]
        golden: PathBuf,
        #[arg(long, default_value_t = 16)]
        chunk: usize,
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
        /// MTP draft depth (`auto` = EV controller, the default; 0 = continuous
        /// batching; 2..=7 = fixed per-stream MTP; 1 is rejected).
        #[arg(long, default_value = "auto")]
        depth: DepthArg,
        /// Print raw token ids per stream (for batched-vs-serial parity diffs).
        #[arg(long, default_value_t = false)]
        show_ids: bool,
    },
    /// Measure the MTP round-cost table for a model: wall ms of a serial step,
    /// a draft step, and a verify forward per width S = depth+1, at each KV
    /// bucket. Merged into `--out` (default `$HOME/.lisa/round_cost.json`,
    /// what the EV controller loads); `--docs` also refreshes the tracked copy
    /// under `docs/round-cost/`.
    RoundCost {
        #[arg(long)]
        model: ModelDir,
        /// Output JSON (merged with the file already there).
        #[arg(long)]
        out: Option<PathBuf>,
        /// KV lengths to measure (one per bucket row, e.g. 2k and 16k).
        #[arg(long, default_value = "1024,16384", value_delimiter = ',')]
        kv: Vec<usize>,
        /// Timed iterations per cell (median after 4 warmup rounds).
        #[arg(long, default_value_t = 24)]
        steps: usize,
        /// Verify widths S = depth+1 to measure.
        #[arg(long, default_value = "3,4,5,6,7", value_delimiter = ',')]
        widths: Vec<u32>,
        /// Also write the tracked copy under `docs/round-cost/`.
        #[arg(long, default_value_t = false)]
        docs: bool,
    },
    /// Verify-round GPU audit: wall ms/verify, dispatches/verify and
    /// per-kernel GPU ms/verify over N S-row verify forwards at `kv`
    /// (Port 2 decomposition, specs/24). Same counter method as serial-audit.
    VerifyAudit {
        #[arg(long)]
        model: ModelDir,
        #[arg(long, default_value_t = 16384)]
        kv: usize,
        /// Verify width S = depth+1.
        #[arg(long, default_value_t = 7)]
        s: u32,
        #[arg(long, default_value_t = 24)]
        steps: usize,
    },
    /// Repeat a golden prompt's prefill in-process for low-noise sampling.
    PrefillBench {
        #[arg(long)]
        model: ModelDir,
        #[arg(long)]
        golden: PathBuf,
        #[arg(long, default_value_t = 9)]
        iters: usize,
        /// Select the GLM Q8 prefill route or run an interleaved DS4/stock A/B.
        #[arg(long, default_value = "ds4-mpp", value_parser = ["ds4-mpp", "stock", "abba"])]
        q8_prefill: String,
        /// Select the default, grouped, or per-slot routed-MoE prefill schedule.
        #[arg(long, default_value = "auto", value_parser = ["auto", "grouped", "stock", "abba"])]
        moe_grouped: String,
        /// Select the fused or DS4-split grouped gate/up schedule, or interleave both.
        #[arg(long, default_value = "auto", value_parser = ["auto", "fused", "split", "abba"])]
        moe_pair: String,
        /// Select scalar or four-byte-vector affine-Q2 unpacking for grouped MoE.
        #[arg(long, default_value = "scalar", value_parser = ["vector", "scalar", "abba"])]
        moe_q2_load: String,
        /// Select the grouped affine-Q2 down output tile or interleave 64/128.
        #[arg(long, default_value = "auto", value_parser = ["auto", "64", "128", "abba"])]
        moe_down_tile: String,
        /// Write both warmup logit vectors as JSON when --moe-grouped abba is selected.
        #[arg(long)]
        dump_schedule_logits_prefix: Option<PathBuf>,
    },
    /// Serial-step GPU audit: wall ms/step, dispatches/step and per-label GPU
    /// intervals over N serial decode steps (specs/16). Pair with
    /// LISA_GPU_PROBE=1 for per-command-buffer GPU intervals attributed to
    /// one kernel per buffer; use a normal run for wall/step because probe
    /// mode changes command-buffer submission cadence.
    SerialAudit {
        #[arg(long)]
        model: ModelDir,
        /// Synthetic prompt length (KV bucket context).
        #[arg(long, default_value_t = 1024)]
        kv: usize,
        /// Optional prefill window for long-context diagnostics. Zero keeps the
        /// single-forward prefill used by the default serial audit.
        #[arg(long, default_value_t = 0)]
        prefill_chunk: usize,
        /// Counted steps after 8 warmup steps.
        #[arg(long, default_value_t = 32)]
        steps: usize,
        /// Compare the DS4 KDA projection schedule, a 4/16-simdgroup
        /// candidate, or interleave them in ABBA order.
        #[arg(
            long,
            default_value = "ds4",
            value_parser = ["ds4", "4", "16", "abba", "abba4", "abba16"]
        )]
        kda_nsg: String,
        /// Compare DS4's one-pool indexer scorer with its four-pool SIMD-group schedule.
        #[arg(long, default_value = "rows4", value_parser = ["legacy", "rows4", "abba"])]
        dsa_score: String,
        /// Select the DS4 threadgroup or SIMD-shuffle top-k network, or interleave both.
        #[arg(long, default_value = "shuffle", value_parser = ["legacy", "shuffle", "abba"])]
        dsa_topk: String,
        /// Compare the reference 4-row output-head QMV tile with the 8-row candidate.
        #[arg(long, default_value = "4", value_parser = ["4", "8", "abba"])]
        lm_head_qmv: String,
    },
}

/// Short host tag for the table's `measured` provenance field.
fn whoami() -> String {
    std::env::var("USER")
        .or_else(|_| std::env::var("LOGNAME"))
        .unwrap_or_else(|_| "unknown".to_string())
}

fn main() -> anyhow::Result<()> {
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
        Command::Glm53Run(args) => glm53_run::run(args),
        Command::Smoke => smoke::run_all(),
        Command::LayerDiff { model } => layerdiff::run(&model),
        Command::Logits {
            model,
            prompt,
            raw,
            dump_logits,
        } => {
            let mut tower = match models::load_dir(&model)? {
                Loaded::Language(tower) => tower,
                Loaded::Decision(model) => {
                    anyhow::bail!("logits are unavailable for decision model {}", model.name())
                }
            };
            let tok = tokenizer::Tokenizer::load(&model)?;
            let text = if raw {
                prompt.clone()
            } else {
                tokenizer::generation_prompt(&[("user".to_string(), prompt.clone())])
            };
            let ids = tok.encode(&text, false)?;
            println!("{} tokens", ids.len());
            println!("input_ids: {ids:?}");
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
                .as_dtype(lisa_mlx::Dtype::Float32)
                .map_err(|e| anyhow::anyhow!("{e}"))?
                .as_slice::<f32>()
                .to_vec();
            for (t, v) in toks.iter().zip(vf.iter()) {
                let piece = tok.decode(&[*t]).unwrap_or_default();
                println!("  {:>7} {:>8.3} {:?}", t, v, piece);
            }
            if let Some(path) = dump_logits {
                let values = logits
                    .as_dtype(lisa_mlx::Dtype::Float32)
                    .map_err(|e| anyhow::anyhow!("{e}"))?
                    .as_slice::<f32>()
                    .to_vec();
                let (argmax_id, argmax_logit) = values
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.total_cmp(b.1))
                    .map(|(id, value)| (id, *value))
                    .ok_or_else(|| anyhow::anyhow!("logit vector is empty"))?;
                let payload = serde_json::json!({
                    "source": "lisa",
                    "model": model.to_string_lossy(),
                    "prompt": prompt,
                    "raw": raw,
                    "input_ids": ids,
                    "vocab": values.len(),
                    "argmax_token": {"id": argmax_id, "logit": argmax_logit},
                    "logits": values,
                });
                std::fs::write(path, serde_json::to_vec(&payload)?)?;
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
        Command::QmmM => smoke::run_qmm_m_test(),
        Command::PrefixCheck {
            model,
            golden,
            tokens,
            suffix,
        } => {
            let _ = &golden; // kept for CLI symmetry with session-check
            // The tracked golden prompts are tiny; build a natural long prompt
            // (repeated text + a counting question) so the greedy continuation
            // runs the full token budget instead of hitting EOS immediately.
            let tok = tokenizer::Tokenizer::load(&model)?;
            let story = "The quick brown fox jumps over the lazy dog. ".repeat(276);
            let text = tokenizer::generation_prompt(&[(
                "user".to_string(),
                format!(
                    "{story}Repeat the sentence above 30 times, then count from 1 to 50 in words, one per line."
                ),
            )]);
            let prompt: Vec<u32> = tok.encode(&text, false)?;
            anyhow::ensure!(prompt.len() > 256, "prompt too short for a prefix test");
            let mut tower = match lisa_engine::models::load_dir(&model)? {
                lisa_engine::models::Loaded::Language(m) => m,
                lisa_engine::models::Loaded::Decision(_) => {
                    anyhow::bail!("prefix-check expects a language model")
                }
            };
            generate::set_eos_ids(vec![248_046, 248_044]);
            tower.warmup(&prompt[..8])?;
            let split = prompt.len().saturating_sub(suffix.min(prompt.len()));

            // Pass 1 is discarded (first-call Metal JIT / lazy page faults).
            for pass in 0..2 {
                // Reference: one full prefill, greedy decode.
                let mut greedy = sampler::Sampler::default();
                let (ref_tokens, ref_stats) =
                    generate::generate(&mut *tower, &prompt, tokens, &mut greedy, 0, &mut |_t| {
                        Ok(())
                    })?;
                // Cached: prefill the prefix once, snapshot it, resume from the
                // entry on a fresh session, prefill only the suffix, decode.
                let mut cache = prefix_cache::PrefixCache::new(4);
                {
                    let mut s = session::Session::new(&mut *tower);
                    s.feed(&mut *tower, &prompt[..split])?;
                    cache.insert_with_law(
                        prefix_cache::model_numerical_law(&*tower),
                        &prompt[..split],
                        &s.caches,
                    );
                }
                let (matched, mut s) = cache
                    .restore_session(&mut *tower, &prompt)
                    .expect("prefix cache hit");
                let mut greedy = sampler::Sampler::default();
                let cached_tokens = s.generate(
                    &mut *tower,
                    &prompt[matched..],
                    tokens,
                    &mut greedy,
                    0,
                    None,
                    &mut |_t| Ok(()),
                )?;
                let first_bad = ref_tokens
                    .iter()
                    .zip(cached_tokens.iter())
                    .position(|(a, b)| a != b);
                if pass == 0 {
                    continue;
                }
                println!(
                    "prefix-check: matched {matched}/{} | ref {} tok ({:.1} tok/s) | cached {} tok | argmax_equal={} first_mismatch={:?}",
                    prompt.len(),
                    ref_tokens.len(),
                    ref_stats.decode_tps(),
                    cached_tokens.len(),
                    first_bad.is_none(),
                    first_bad
                );
                anyhow::ensure!(
                    first_bad.is_none() && ref_tokens.len() == cached_tokens.len(),
                    "prefix-cache restore diverged"
                );
            }
            println!("prefix-check OK");
            Ok(())
        }
        Command::SessionCheck {
            model,
            golden,
            chunk,
        } => {
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
            show_ids,
        } => {
            // models::load_dir routes by model_type (qwen4 AND qwen3_5);
            // the qwen4-only `model::Tower::load` needs a PLE ngram sidecar
            // the 27B does not ship, which locked cbatch to Flash-Next.
            let mut tower = match lisa_engine::models::load_dir(&model)? {
                lisa_engine::Loaded::Language(m) => m,
                lisa_engine::Loaded::Decision(_) => {
                    anyhow::bail!("cbatch needs a generative language model")
                }
            };
            let tower = tower.as_mut();
            let tok = tokenizer::Tokenizer::load(&model)?;
            generate::set_eos_ids_for_model(tok.im_end_ids.clone(), tower);
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
                        lisa_engine::cli::build_sampler(
                            temperature,
                            top_k,
                            top_p,
                            min_p,
                            rep_penalty,
                            seed,
                        ),
                    )
                })
                .collect::<anyhow::Result<_>>()?;
            tower.warmup(&requests[0].prompt)?;
            let depth_us = match depth {
                DepthArg::Auto => lisa_engine::core::round_cost::AUTO_DEPTH,
                DepthArg::Fixed(d) => d,
            };
            eprintln!(
                "cbatch: {} streams, max_batch {max_batch}, depth {depth}",
                requests.len()
            );
            let t0 = std::time::Instant::now();
            let mut acc: Vec<String> = vec![String::new(); requests.len()];
            let out = sched::run(tower, requests, max_batch, depth_us, &mut |id, t| {
                acc[id].push_str(&tok.decode(&[t])?);
                Ok(())
            })?;
            for (i, o) in out.iter().enumerate() {
                println!("--- stream {i}: {} tokens ---", o.len());
                if show_ids {
                    println!("ids: {:?}", o);
                }
                println!("{}", acc[i]);
            }
            eprintln!("cbatch done in {:.2}s", t0.elapsed().as_secs_f64());
            Ok(())
        }
        Command::SerialAudit {
            model,
            kv,
            prefill_chunk,
            steps,
            kda_nsg,
            dsa_score,
            dsa_topk,
            lm_head_qmv,
        } => {
            anyhow::ensure!(
                dsa_score != "abba" || kda_nsg == "ds4",
                "--dsa-score abba requires the default DS4 KDA schedule"
            );
            anyhow::ensure!(
                dsa_score != "abba" || !kda_nsg.starts_with("abba"),
                "run one ABBA schedule experiment at a time"
            );
            anyhow::ensure!(
                dsa_topk != "abba" || kda_nsg == "ds4",
                "--dsa-topk abba requires the default DS4 KDA schedule"
            );
            anyhow::ensure!(
                dsa_topk != "abba" || dsa_score != "abba",
                "run one ABBA schedule experiment at a time"
            );
            anyhow::ensure!(
                dsa_topk != "abba" || !kda_nsg.starts_with("abba"),
                "run one ABBA schedule experiment at a time"
            );
            anyhow::ensure!(
                lm_head_qmv != "abba"
                    || (kda_nsg == "ds4" && dsa_score != "abba" && dsa_topk != "abba"),
                "--lm-head-qmv abba requires all other schedules to use one fixed path"
            );
            use lisa_mlx::{
                runtime_dispatch_count, runtime_label_counts, runtime_label_dispatches,
                runtime_label_gpu_ms,
            };
            let mut tower = match lisa_engine::models::load_dir(&model)? {
                lisa_engine::models::Loaded::Language(m) => m,
                lisa_engine::models::Loaded::Decision(_) => {
                    anyhow::bail!("serial-audit expects a language model")
                }
            };
            let tok = tokenizer::Tokenizer::load(&model)?;
            let text = tokenizer::generation_prompt(&[(
                "user".to_string(),
                "The quick brown fox jumps over the lazy dog. ".repeat(kv.div_ceil(10) + 1),
            )]);
            let mut ids: Vec<u32> = tok.encode(&text, false)?;
            ids.truncate(kv);
            tower.warmup(&ids[..ids.len().min(64)])?;
            let mut sess = lisa_engine::core::session::Session::new(&mut *tower);
            let logits = if prefill_chunk == 0 {
                sess.feed_multi(&mut *tower, &ids)?.0
            } else {
                let mut logits = None;
                for chunk in ids.chunks(prefill_chunk) {
                    let next = sess.feed(&mut *tower, chunk)?;
                    next.eval()?;
                    logits = Some(next);
                    lisa_mlx::memory::trim_cache();
                }
                logits.expect("non-empty serial-audit prompt")
            };
            let argmax_id = |a: &lisa_mlx::Array| -> anyhow::Result<u32> {
                let out =
                    lisa_mlx::ops::indexing::argmax(a, None).map_err(|e| anyhow::anyhow!("{e}"))?;
                out.eval().map_err(|e| anyhow::anyhow!("{e}"))?;
                Ok(out.item_cast::<i32>() as u32)
            };
            let mut tok_id = argmax_id(&logits)?;
            let warm = 8usize;
            for _ in 0..warm {
                let logits = sess.decode_token(&mut *tower, tok_id)?;
                tok_id = argmax_id(&logits)?;
            }
            // Snapshot BEFORE the counted steps (counters are cumulative since
            // process start, prefill included) so deltas cover exactly them.
            let pre_d = runtime_dispatch_count();
            let pre_lc = runtime_label_counts();
            let pre_ld = runtime_label_dispatches();
            let pre_lg = runtime_label_gpu_ms();
            std::thread::sleep(std::time::Duration::from_millis(300));
            let pre_lg2 = runtime_label_gpu_ms();
            println!(
                "pre GPU cumulative: {:.1} ms / after settle {:.1} ms (labels={})",
                pre_lg.iter().map(|(_, g)| g).sum::<u64>() as f64 / 1e6,
                pre_lg2.iter().map(|(_, g)| g).sum::<u64>() as f64 / 1e6,
                pre_lg.len()
            );
            let mut walls: Vec<f64> = Vec::new();
            let mut ds4_walls: Vec<f64> = Vec::new();
            let mut candidate_walls: Vec<f64> = Vec::new();
            let mut dsa_baseline_walls: Vec<f64> = Vec::new();
            let mut dsa_rows4_walls: Vec<f64> = Vec::new();
            let mut dsa_topk_legacy_walls: Vec<f64> = Vec::new();
            let mut dsa_topk_shuffle_walls: Vec<f64> = Vec::new();
            let mut lm_head_qmv_reference_walls: Vec<f64> = Vec::new();
            let mut lm_head_qmv_wide_walls: Vec<f64> = Vec::new();
            let mut sampled_tokens = Vec::with_capacity(steps);
            for step in 0..steps {
                let use_candidate = match kda_nsg.as_str() {
                    "ds4" => false,
                    "4" | "16" => true,
                    "abba" | "abba16" | "abba4" => matches!(step % 4, 1 | 2),
                    _ => unreachable!("clap validates kda-nsg"),
                };
                let use_rows4 = match dsa_score.as_str() {
                    "legacy" => false,
                    "rows4" => true,
                    "abba" => matches!(step % 4, 1 | 2),
                    _ => unreachable!("clap validates dsa-score"),
                };
                let use_shuffle = match dsa_topk.as_str() {
                    "legacy" => false,
                    "shuffle" => true,
                    "abba" => matches!(step % 4, 1 | 2),
                    _ => unreachable!("clap validates dsa-topk"),
                };
                let use_wide_lm_head = match lm_head_qmv.as_str() {
                    "4" => false,
                    "8" => true,
                    "abba" => matches!(step % 4, 1 | 2),
                    _ => unreachable!("clap validates lm-head-qmv"),
                };
                let t0 = std::time::Instant::now();
                let _dsa_score_scope = (dsa_score != "rows4").then(|| {
                    lisa_engine::core::quant::Glm53IndexerDecodeRows4Scope::new(use_rows4)
                });
                let _dsa_topk_scope = (dsa_topk != "shuffle").then(|| {
                    lisa_engine::core::quant::Glm53IndexerTopkShuffleScope::new(use_shuffle)
                });
                let _lm_head_qmv_scope = (lm_head_qmv != "4").then(|| {
                    lisa_mlx::ops::Glm53LmHeadQmvTileScope::new(if use_wide_lm_head {
                        8
                    } else {
                        4
                    })
                });
                let logits = if use_candidate {
                    let nsg = if kda_nsg == "4" || kda_nsg == "abba4" {
                        4
                    } else {
                        16
                    };
                    let _scope = lisa_engine::core::quant::Glm53KdaNsgScope::new(nsg);
                    sess.decode_token(&mut *tower, tok_id)?
                } else {
                    sess.decode_token(&mut *tower, tok_id)?
                };
                let id = argmax_id(&logits)?;
                let dt = t0.elapsed().as_secs_f64() * 1e3;
                walls.push(dt);
                if use_candidate {
                    candidate_walls.push(dt);
                } else {
                    ds4_walls.push(dt);
                }
                if dsa_score != "ds4" {
                    if use_rows4 {
                        dsa_rows4_walls.push(dt);
                    } else {
                        dsa_baseline_walls.push(dt);
                    }
                }
                if dsa_topk != "legacy" {
                    if use_shuffle {
                        dsa_topk_shuffle_walls.push(dt);
                    } else {
                        dsa_topk_legacy_walls.push(dt);
                    }
                }
                if lm_head_qmv != "4" {
                    if use_wide_lm_head {
                        lm_head_qmv_wide_walls.push(dt);
                    } else {
                        lm_head_qmv_reference_walls.push(dt);
                    }
                }
                sampled_tokens.push(id);
                tok_id = id;
            }
            let d = runtime_dispatch_count();
            // GPU time lands in the completion handlers (async); give them a
            // beat to drain the counted buffers before the post snapshot.
            std::thread::sleep(std::time::Duration::from_millis(500));
            let lc = runtime_label_counts();
            let ld = runtime_label_dispatches();
            let lg = runtime_label_gpu_ms();
            let pre = (pre_d, &pre_lc, &pre_lg2);
            let _ = &pre_lg;
            let n = steps as f64;
            let mut steps_v: Vec<f64> = walls.clone();
            steps_v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let wall_med = steps_v[steps_v.len() / 2];
            println!("greedy tokens: {sampled_tokens:?}");
            println!(
                "serial-audit: {steps} steps @ kv {kv} (prefill_chunk={prefill_chunk}, kda_nsg={kda_nsg}, dsa_score={dsa_score}, dsa_topk={dsa_topk}, lm_head_qmv={lm_head_qmv}, probe={:?})",
                std::env::var_os("LISA_GPU_PROBE").is_some(),
            );
            println!(
                "wall ms/step: median {:.2}  mean {:.2}",
                wall_med,
                walls.iter().sum::<f64>() / n
            );
            if kda_nsg.starts_with("abba") {
                let median = |values: &mut Vec<f64>| -> f64 {
                    values.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    values[values.len() / 2]
                };
                let candidate_nsg = if kda_nsg == "abba4" { 4 } else { 16 };
                println!(
                    "ABBA wall ms/step: DS4 median {:.2} ({} steps), nsg{candidate_nsg} median {:.2} ({} steps)",
                    median(&mut ds4_walls),
                    ds4_walls.len(),
                    median(&mut candidate_walls),
                    candidate_walls.len(),
                );
            }
            if dsa_score == "abba" {
                let median = |values: &mut Vec<f64>| -> f64 {
                    values.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    values[values.len() / 2]
                };
                println!(
                    "DSA score ABBA wall ms/step: legacy median {:.2} ({} steps), rows4 median {:.2} ({} steps)",
                    median(&mut dsa_baseline_walls),
                    dsa_baseline_walls.len(),
                    median(&mut dsa_rows4_walls),
                    dsa_rows4_walls.len(),
                );
            }
            if dsa_topk == "abba" {
                let median = |values: &mut Vec<f64>| -> f64 {
                    values.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    values[values.len() / 2]
                };
                println!(
                    "DSA top-k ABBA wall ms/step: legacy median {:.2} ({} steps), shuffle median {:.2} ({} steps)",
                    median(&mut dsa_topk_legacy_walls),
                    dsa_topk_legacy_walls.len(),
                    median(&mut dsa_topk_shuffle_walls),
                    dsa_topk_shuffle_walls.len(),
                );
            }
            if lm_head_qmv == "abba" {
                let median = |values: &mut Vec<f64>| -> f64 {
                    values.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    values[values.len() / 2]
                };
                println!(
                    "LM-head QMV ABBA wall ms/step: 4-row median {:.2} ({} steps), 8-row median {:.2} ({} steps)",
                    median(&mut lm_head_qmv_reference_walls),
                    lm_head_qmv_reference_walls.len(),
                    median(&mut lm_head_qmv_wide_walls),
                    lm_head_qmv_wide_walls.len(),
                );
            }
            println!("dispatches/step: {:.1}", (d - pre.0) as f64 / n);
            let gpu: std::collections::HashMap<String, u64> = pre.2.iter().cloned().collect();
            // Buffer-allocation labels and kernel labels are separate
            // universes; GPU timing and dispatch counts use the latter.
            let mut dl: Vec<(&String, i64)> = lg
                .iter()
                .map(|(k, g)| (k, *g as i64 - gpu.get(k).copied().unwrap_or(0) as i64))
                .collect();
            dl.sort_by(|a, b| b.1.cmp(&a.1));
            let gpu_sum: f64 = dl.iter().map(|(_, g)| *g).sum::<i64>() as f64 / 1e6;
            println!(
                "attributed GPU interval ms/step: {:.2}  wall-minus-interval: {:+.2} ms/step",
                gpu_sum / n,
                wall_med - gpu_sum / n
            );
            println!("--- kernels by attributed GPU interval/step ---");
            for (k, g) in dl.iter().filter(|(_, g)| *g > 10_000) {
                println!("{:>8.3} ms  {}", *g as f64 / 1e6 / n, k);
            }
            let dispatches: std::collections::HashMap<String, u64> =
                pre_ld.iter().cloned().collect();
            let mut dlc: Vec<(&String, i64)> = ld
                .iter()
                .map(|(k, v)| {
                    (
                        k,
                        *v as i64 - dispatches.get(k).copied().unwrap_or(0) as i64,
                    )
                })
                .collect();
            dlc.sort_by(|a, b| b.1.cmp(&a.1));
            println!("--- kernels by dispatch count/step ---");
            for (k, c) in dlc.iter().filter(|(_, c)| *c > 0) {
                println!("{:>7.1}/step  {}", *c as f64 / n, k);
            }
            let allocations: std::collections::HashMap<String, u64> =
                pre.1.iter().cloned().collect();
            let mut alc: Vec<(&String, i64)> = lc
                .iter()
                .map(|(k, v)| {
                    (
                        k,
                        *v as i64 - allocations.get(k).copied().unwrap_or(0) as i64,
                    )
                })
                .collect();
            alc.sort_by(|a, b| b.1.cmp(&a.1));
            println!("--- output allocations/step ---");
            for (k, c) in alc.iter().filter(|(_, c)| *c > 0) {
                println!("{:>7.1}/step  {}", *c as f64 / n, k);
            }
            Ok(())
        }
        Command::VerifyAudit {
            model,
            kv,
            s,
            steps,
        } => {
            use lisa_mlx::{runtime_dispatch_count, runtime_label_counts, runtime_label_gpu_ms};
            let mut tower = match lisa_engine::models::load_dir(&model)? {
                lisa_engine::models::Loaded::Language(m) => m,
                lisa_engine::models::Loaded::Decision(_) => {
                    anyhow::bail!("verify-audit expects a language model")
                }
            };
            let tok = tokenizer::Tokenizer::load(&model)?;
            let text = tokenizer::generation_prompt(&[(
                "user".to_string(),
                "The quick brown fox jumps over the lazy dog. ".repeat(kv.div_ceil(10) + 1),
            )]);
            let mut ids: Vec<u32> = tok.encode(&text, false)?;
            ids.truncate(kv);
            tower.warmup(&ids[..ids.len().min(64)])?;
            let mut sess = lisa_engine::core::session::Session::new(&mut *tower);
            let ctx_len = tower.context_window();
            let (logits, _) = sess.feed_multi(&mut *tower, &ids)?;
            let _ = logits;
            let s = s.max(2);
            let verify_ids: Vec<i32> = std::iter::repeat(1i32).take(s as usize).collect();
            // One S-row verify forward with snapshot/rollback (round-cost's
            // verify pass); wall ms includes head+argmax, splitk scope armed.
            let verify_once = |sess: &mut lisa_engine::core::session::Session,
                               tower: &mut dyn lisa_engine::models::LanguageModel|
             -> anyhow::Result<f64> {
                let snaps: Vec<_> = sess
                    .caches
                    .iter()
                    .map(|c| match c {
                        lisa_engine::core::cache::LayerCache::Full(f) => (Some(f.offset), None),
                        lisa_engine::core::cache::LayerCache::Linear(g) => {
                            (None, Some(g.snapshot_state()))
                        }
                    })
                    .collect();
                tower.set_context_tails(vec![
                    sess.fed[sess.fed.len().saturating_sub(ctx_len)..]
                        .iter()
                        .map(|&t| t as i64)
                        .collect(),
                ]);
                let arr = lisa_mlx::Array::from_slice(&verify_ids, &[1i32, s as i32]);
                let _splitk = lisa_mlx::ops::enter_verify_splitk_scope();
                let t0 = std::time::Instant::now();
                let (mixed, _multi) = tower.forward_capture(&arr, Some(&mut sess.caches), true)?;
                let logits = tower.head(&mixed)?;
                let top = lisa_mlx::ops::indexing::argmax_axis(&logits, -1, None)
                    .map_err(|e| anyhow::anyhow!("{e}"))?;
                top.eval().map_err(|e| anyhow::anyhow!("{e}"))?;
                let dt = t0.elapsed().as_secs_f64() * 1e3;
                drop(_splitk);
                for (c, sn) in sess.caches.iter_mut().zip(snaps.iter()) {
                    match (c, sn) {
                        (lisa_engine::core::cache::LayerCache::Full(f), (Some(off), _)) => {
                            f.trim(f.offset - off)
                        }
                        (lisa_engine::core::cache::LayerCache::Linear(g), (_, Some(st))) => {
                            g.restore_state(st)
                        }
                        _ => {}
                    }
                }
                Ok(dt)
            };
            let warm = 4usize;
            for _ in 0..warm {
                verify_once(&mut sess, &mut *tower)?;
            }
            let pre_d = runtime_dispatch_count();
            let pre_lc = runtime_label_counts();
            std::thread::sleep(std::time::Duration::from_millis(300));
            let pre_lg2 = runtime_label_gpu_ms();
            let mut walls: Vec<f64> = Vec::new();
            for _ in 0..steps {
                walls.push(verify_once(&mut sess, &mut *tower)?);
            }
            let d = runtime_dispatch_count();
            std::thread::sleep(std::time::Duration::from_millis(500));
            let lc = runtime_label_counts();
            let lg = runtime_label_gpu_ms();
            let pre = (pre_d, &pre_lc, &pre_lg2);
            let n = steps as f64;
            let mut walls_v = walls.clone();
            walls_v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let wall_med = walls_v[walls_v.len() / 2];
            println!("verify-audit: {steps} x S{s} @ kv {kv}");
            println!(
                "wall ms/verify: median {:.2}  mean {:.2}",
                wall_med,
                walls.iter().sum::<f64>() / n
            );
            println!("dispatches/verify: {:.1}", (d - pre.0) as f64 / n);
            let count: std::collections::HashMap<String, u64> = pre.1.iter().cloned().collect();
            let gpu: std::collections::HashMap<String, u64> = pre.2.iter().cloned().collect();
            let mut dl: Vec<(&String, i64)> = lg
                .iter()
                .map(|(k, g)| (k, *g as i64 - gpu.get(k).copied().unwrap_or(0) as i64))
                .collect();
            dl.sort_by(|a, b| b.1.cmp(&a.1));
            let gpu_sum: f64 = dl.iter().map(|(_, g)| *g).sum::<i64>() as f64 / 1e6;
            println!(
                "GPU ms/verify (sum over kernels): {:.2}  -> un-attributed {:.2} ms/verify",
                gpu_sum / n,
                (wall_med - gpu_sum / n).max(0.0)
            );
            println!("--- kernels by GPU ms/verify ---");
            for (k, g) in dl.iter().filter(|(_, g)| *g > 10_000) {
                println!("{:>8.3} ms  {}", *g as f64 / 1e6 / n, k);
            }
            let mut cl: Vec<(&String, i64)> = lc
                .iter()
                .map(|(k, v)| (k, *v as i64 - count.get(k).copied().unwrap_or(0) as i64))
                .collect();
            cl.sort_by(|a, b| b.1.cmp(&a.1));
            println!("--- ops by dispatch count/verify ---");
            for (k, c) in cl.iter().filter(|(_, c)| *c > 0) {
                println!("{:>7.1}/verify  {}", *c as f64 / n, k);
            }
            Ok(())
        }
        Command::RoundCost {
            model,
            out,
            kv,
            steps,
            widths,
            docs,
        } => {
            use lisa_engine::core::round_cost::{
                ModelCost, RoundCostTable, bucket_for, default_path, measure_at_kv,
            };
            let mut tower = match lisa_engine::models::load_dir(&model)? {
                lisa_engine::models::Loaded::Language(m) => m,
                lisa_engine::models::Loaded::Decision(_) => {
                    anyhow::bail!("round-cost expects a language model")
                }
            };
            let widths = if tower.has_target_mtp() {
                anyhow::ensure!(
                    widths == [3, 4, 5, 6, 7] || widths.iter().all(|&width| width == 2),
                    "GLM target-MTP supports only verify width S=2; pass --widths 2"
                );
                eprintln!("round-cost: target-conditioned MTP supports only verify width S=2");
                vec![2]
            } else {
                widths
            };
            let tok = tokenizer::Tokenizer::load(&model)?;
            let key = tower.mtp_cost_key();
            let out_path = out.unwrap_or_else(default_path);
            let mut table = RoundCostTable::load(&out_path).unwrap_or_default();
            table.version = 2;
            for &kv_len in &kv {
                let text = tokenizer::generation_prompt(&[(
                    "user".to_string(),
                    "The quick brown fox jumps over the lazy dog. ".repeat(kv_len.div_ceil(10) + 1),
                )]);
                let mut ids: Vec<u32> = tok.encode(&text, false)?;
                anyhow::ensure!(
                    ids.len() >= kv_len,
                    "synthetic prompt too short ({} < {})",
                    ids.len(),
                    kv_len
                );
                ids.truncate(kv_len);
                tower.warmup(&ids[..ids.len().min(64)])?;
                let bucket = bucket_for(kv_len);
                let (serial, draft, chain, verify) =
                    measure_at_kv(&mut *tower, &ids, &widths, steps)?;
                {
                    let mc: &mut ModelCost = table.get_or_insert(&key);
                    mc.key = key.clone();
                    mc.measured = format!(
                        "{} {}",
                        std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)?
                            .as_secs(),
                        whoami(),
                    );
                    mc.serial_ms[bucket].observe(serial as f32);
                    mc.draft_ms[bucket].observe(draft as f32);
                    for (d_w, ms) in chain.iter() {
                        let cells = mc.chain_ms.entry(*d_w).or_default();
                        cells[bucket].observe(*ms as f32);
                    }
                    for (s_w, ms) in verify.iter() {
                        let cells = mc.verify_ms.entry(*s_w).or_default();
                        cells[bucket].observe(*ms as f32);
                    }
                }
                println!(
                    "bucket {} (kv {}): serial {:.2} ms | draft {:.2} ms | chain {} | verify {}",
                    lisa_engine::core::round_cost::BUCKET_NAMES[bucket],
                    kv_len,
                    serial,
                    draft,
                    chain
                        .iter()
                        .map(|(d_w, ms)| format!("d{d_w}:{ms:.2}ms"))
                        .collect::<Vec<_>>()
                        .join(" "),
                    verify
                        .iter()
                        .map(|(s_w, ms)| format!("S{s_w}:{ms:.2}ms"))
                        .collect::<Vec<_>>()
                        .join(" ")
                );
            }
            table.save(&out_path)?;
            println!("table written: {} (model row {key})", out_path.display());
            if docs {
                let dir = std::path::Path::new("docs/round-cost");
                std::fs::create_dir_all(dir)?;
                let p = dir.join(format!("{key}.json"));
                table.save(&p)?;
                println!("docs copy: {}", p.display());
            }
            Ok(())
        }
        Command::PrefillBench {
            model,
            golden,
            iters,
            q8_prefill,
            moe_grouped,
            moe_pair,
            moe_q2_load,
            moe_down_tile,
            dump_schedule_logits_prefix,
        } => {
            anyhow::ensure!(
                [
                    q8_prefill == "abba",
                    moe_grouped == "abba",
                    moe_pair == "abba",
                    moe_q2_load == "abba",
                    moe_down_tile == "abba"
                ]
                .into_iter()
                .filter(|active| *active)
                .count()
                    <= 1,
                "select ABBA for only one prefill schedule at a time"
            );
            anyhow::ensure!(
                moe_pair != "abba" || moe_grouped == "grouped",
                "--moe-pair abba requires --moe-grouped grouped"
            );
            anyhow::ensure!(
                moe_q2_load != "abba"
                    || (moe_grouped == "grouped" && q8_prefill != "abba" && moe_pair != "abba"),
                "--moe-q2-load abba requires grouped MoE and fixed Q8/pair schedules"
            );
            anyhow::ensure!(
                moe_down_tile != "abba"
                    || (moe_grouped == "grouped"
                        && q8_prefill != "abba"
                        && moe_pair != "abba"
                        && moe_q2_load != "abba"),
                "--moe-down-tile abba requires grouped MoE and fixed Q8/pair/load schedules"
            );
            anyhow::ensure!(
                dump_schedule_logits_prefix.is_none()
                    || (q8_prefill != "abba" && moe_grouped == "abba" && moe_pair != "abba"),
                "schedule logit dumps require --moe-grouped abba with a fixed Q8 prefill route"
            );
            let data = std::fs::read_to_string(&golden)?;
            let g: serde_json::Value = serde_json::from_str(&data)?;
            let prompt: Vec<u32> = g["cases"][0]["prompt_tokens"]
                .as_array()
                .unwrap()
                .iter()
                .filter_map(|v| v.as_u64().map(|x| x as u32))
                .collect();
            let n = prompt.len();
            let t0 = std::time::Instant::now();
            let mut tower = match models::load_dir(&model)? {
                Loaded::Language(tower) => tower,
                Loaded::Decision(model) => {
                    anyhow::bail!(
                        "prefill-bench requires a language model, got {}",
                        model.name()
                    )
                }
            };
            eprintln!("loaded in {:.1}s", t0.elapsed().as_secs_f64());
            // Both schedules can compile first-use Metal pipelines during
            // prefill. Warm each path before collecting the interleaved
            // samples so one route does not pay that one-time cost in the
            // measured ABBA window.
            let warm_moe_modes: &[Option<bool>] = match moe_grouped.as_str() {
                "auto" => &[None],
                "grouped" => &[Some(true)],
                "stock" => &[Some(false)],
                "abba" => &[Some(true), Some(false)],
                _ => unreachable!("clap validates moe-grouped"),
            };
            let warm_pair_modes: &[Option<bool>] = match moe_pair.as_str() {
                "auto" => &[None],
                "fused" => &[Some(false)],
                "split" => &[Some(true)],
                "abba" => &[Some(false), Some(true)],
                _ => unreachable!("clap validates moe-pair"),
            };
            let warm_q2_load_modes: &[Option<bool>] = match moe_q2_load.as_str() {
                "vector" => &[Some(true)],
                "scalar" => &[Some(false)],
                "abba" => &[Some(false), Some(true)],
                _ => unreachable!("clap validates moe-q2-load"),
            };
            let warm_down_tile_modes: &[Option<usize>] = match moe_down_tile.as_str() {
                "auto" => &[None],
                "64" => &[Some(64)],
                "128" => &[Some(128)],
                "abba" => &[Some(64), Some(128)],
                _ => unreachable!("clap validates moe-down-tile"),
            };
            let mut grouped_warm_logits: Option<(Vec<f32>, u32)> = None;
            let mut stock_warm_logits: Option<(Vec<f32>, u32)> = None;
            let mut vector_load_warm_logits: Option<(Vec<f32>, u32)> = None;
            let mut scalar_load_warm_logits: Option<(Vec<f32>, u32)> = None;
            let mut tile64_warm_logits: Option<(Vec<f32>, u32)> = None;
            let mut tile128_warm_logits: Option<(Vec<f32>, u32)> = None;
            for use_ds4_mpp in [true, false] {
                for &grouped_override in warm_moe_modes {
                    for &pair_override in warm_pair_modes {
                        for &q2_load_override in warm_q2_load_modes {
                            for &down_tile_override in warm_down_tile_modes {
                                let q8_name = if use_ds4_mpp { "ds4-mpp" } else { "stock" };
                                let moe_name = match grouped_override {
                                    Some(true) => "grouped",
                                    Some(false) => "stock",
                                    None => "auto",
                                };
                                let pair_split = pair_override.unwrap_or(true);
                                let pair_name = if pair_split { "split" } else { "fused" };
                                let q2_vector_load = q2_load_override.unwrap_or(true);
                                let q2_load_name = if q2_vector_load { "vector" } else { "scalar" };
                                let _q8_scope =
                                    lisa_engine::core::quant::Glm53Q8PrefillScope::new(use_ds4_mpp);
                                let _moe_scope = grouped_override.map(
                                    lisa_engine::core::quant::Glm53GroupedMoePrefillScope::new,
                                );
                                let _pair_scope = pair_override
                                    .map(lisa_engine::core::quant::Glm53GroupedMoePairScope::new);
                                let _q2_load_scope = q2_load_override
                                    .map(lisa_engine::core::quant::Glm53GroupedMoeQ2LoadScope::new);
                                let _down_tile_scope = down_tile_override.map(
                                    lisa_engine::core::quant::Glm53GroupedMoeDownTileScope::new,
                                );
                                let mut caches = tower.new_caches();
                                tower.clear_context();
                                let t = std::time::Instant::now();
                                let last = tower.prefill(&prompt, &mut caches)?;
                                let logits = tower.head(&last)?;
                                let next = lisa_mlx::ops::indexing::argmax(&logits, None)?;
                                next.eval().map_err(|e| anyhow::anyhow!("{e}"))?;
                                let next_token = next.item_cast::<i32>() as u32;
                                if use_ds4_mpp && moe_grouped == "abba" {
                                    let values = logits.to_vec1::<f32>()?;
                                    match grouped_override {
                                        Some(true) => {
                                            grouped_warm_logits = Some((values, next_token))
                                        }
                                        Some(false) => {
                                            stock_warm_logits = Some((values, next_token))
                                        }
                                        _ => unreachable!("ABBA warmup forces each MoE schedule"),
                                    }
                                }
                                if use_ds4_mpp && moe_q2_load == "abba" {
                                    let values = logits.to_vec1::<f32>()?;
                                    match q2_load_override {
                                        Some(true) => {
                                            vector_load_warm_logits = Some((values, next_token))
                                        }
                                        Some(false) => {
                                            scalar_load_warm_logits = Some((values, next_token))
                                        }
                                        _ => unreachable!("Q2 load ABBA warms each load path"),
                                    }
                                }
                                if use_ds4_mpp && moe_down_tile == "abba" {
                                    let values = logits.to_vec1::<f32>()?;
                                    match down_tile_override {
                                        Some(64) => tile64_warm_logits = Some((values, next_token)),
                                        Some(128) => {
                                            tile128_warm_logits = Some((values, next_token))
                                        }
                                        _ => unreachable!("down tile ABBA warms each tile path"),
                                    }
                                }
                                eprintln!(
                                    "warmup q8={q8_name:>7} moe={moe_name:>7} pair={pair_name:>5} q2={q2_load_name:>6} down={down_tile_override:?}: {:>7.1} ms, next_token={}",
                                    t.elapsed().as_secs_f64() * 1e3,
                                    next_token
                                );
                            }
                        }
                    }
                }
            }
            if let (Some((grouped, grouped_token)), Some((stock, stock_token))) =
                (grouped_warm_logits.as_ref(), stock_warm_logits.as_ref())
            {
                anyhow::ensure!(
                    grouped.len() == stock.len(),
                    "MoE schedule logits have different lengths"
                );
                let mut exact = 0usize;
                let mut max_abs = 0.0f64;
                let mut diff_sq = 0.0f64;
                let mut reference_sq = 0.0f64;
                for (&a, &b) in grouped.iter().zip(stock) {
                    if a.to_bits() == b.to_bits() {
                        exact += 1;
                    }
                    let delta = (a as f64) - (b as f64);
                    max_abs = max_abs.max(delta.abs());
                    diff_sq += delta * delta;
                    reference_sq += (b as f64) * (b as f64);
                }
                let rel_l2 = (diff_sq / reference_sq.max(f64::MIN_POSITIVE)).sqrt();
                println!(
                    "warmup logits q8=ds4-mpp: grouped vs stock exact {exact}/{} ({:.4}%), max_abs {:.6e}, rel_l2 {:.6e}, next_token {} / {}{}",
                    grouped.len(),
                    exact as f64 * 100.0 / grouped.len() as f64,
                    max_abs,
                    rel_l2,
                    grouped_token,
                    stock_token,
                    if grouped_token == stock_token {
                        " (same)"
                    } else {
                        " (DIFFERENT)"
                    }
                );
                if let Some(prefix) = dump_schedule_logits_prefix.as_ref() {
                    for (name, values) in [("grouped", grouped), ("stock", stock)] {
                        let path =
                            std::path::PathBuf::from(format!("{}.{}.json", prefix.display(), name));
                        let payload = serde_json::json!({
                            "prompt_tokens": n,
                            "vocab": values.len(),
                            "logits": values,
                        });
                        std::fs::write(&path, serde_json::to_vec(&payload)?)?;
                        println!("wrote {name} warmup logits: {}", path.display());
                    }
                }
            }
            if let (Some((tile64, token64)), Some((tile128, token128))) =
                (tile64_warm_logits.as_ref(), tile128_warm_logits.as_ref())
            {
                anyhow::ensure!(
                    tile64.len() == tile128.len(),
                    "down tile logits have different lengths"
                );
                let exact = tile64
                    .iter()
                    .zip(tile128)
                    .filter(|(a, b)| a.to_bits() == b.to_bits())
                    .count();
                let max_abs = tile64
                    .iter()
                    .zip(tile128)
                    .map(|(a, b)| (f64::from(*a) - f64::from(*b)).abs())
                    .fold(0.0, f64::max);
                println!(
                    "grouped down tile warmup: 64/128 exact logits {exact}/{}, max_abs={max_abs:.8}, next_token={token64}/{token128}",
                    tile64.len()
                );
            }
            if let (Some((vector, vector_token)), Some((scalar, scalar_token))) = (
                vector_load_warm_logits.as_ref(),
                scalar_load_warm_logits.as_ref(),
            ) {
                anyhow::ensure!(
                    vector.len() == scalar.len(),
                    "Q2 load logits have different lengths"
                );
                let exact = vector
                    .iter()
                    .zip(scalar)
                    .filter(|(a, b)| a.to_bits() == b.to_bits())
                    .count();
                let max_abs = vector
                    .iter()
                    .zip(scalar)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f32, f32::max);
                println!(
                    "warmup logits q8=ds4-mpp moe=grouped: vector vs scalar exact {exact}/{}, max_abs {:.6e}, next_token {} / {}",
                    vector.len(),
                    max_abs,
                    vector_token,
                    scalar_token
                );
                anyhow::ensure!(
                    exact == vector.len() && vector_token == scalar_token,
                    "vector and scalar affine-Q2 load paths changed prefill logits"
                );
            }
            let mut times = Vec::new();
            for i in 0..iters {
                let use_ds4_mpp = match q8_prefill.as_str() {
                    "ds4-mpp" => true,
                    "stock" => false,
                    "abba" => matches!(i % 4, 0 | 3),
                    _ => unreachable!("clap validates q8-prefill"),
                };
                let use_grouped_moe = match moe_grouped.as_str() {
                    "auto" => n >= lisa_engine::core::quant::GLM53_GROUPED_MOE_MIN_ROWS,
                    "grouped" => true,
                    "stock" => false,
                    "abba" => matches!(i % 4, 0 | 3),
                    _ => unreachable!("clap validates moe-grouped"),
                };
                let use_split_pair = match moe_pair.as_str() {
                    "auto" => true,
                    "fused" => false,
                    "split" => true,
                    "abba" => !matches!(i % 4, 0 | 3),
                    _ => unreachable!("clap validates moe-pair"),
                };
                let use_vector_q2_load = match moe_q2_load.as_str() {
                    "vector" => true,
                    "scalar" => false,
                    "abba" => matches!(i % 4, 0 | 3),
                    _ => unreachable!("clap validates moe-q2-load"),
                };
                let down_tile = match moe_down_tile.as_str() {
                    "auto" => 64,
                    "64" => 64,
                    "128" => 128,
                    "abba" => {
                        if matches!(i % 4, 0 | 3) {
                            64
                        } else {
                            128
                        }
                    }
                    _ => unreachable!("clap validates moe-down-tile"),
                };
                let q8_name = if use_ds4_mpp { "ds4-mpp" } else { "stock" };
                let moe_name = if use_grouped_moe { "grouped" } else { "stock" };
                let pair_name = if use_split_pair { "split" } else { "fused" };
                let q2_load_name = if use_vector_q2_load {
                    "vector"
                } else {
                    "scalar"
                };
                let _q8_scope = lisa_engine::core::quant::Glm53Q8PrefillScope::new(use_ds4_mpp);
                let _moe_scope = (moe_grouped != "auto").then(|| {
                    lisa_engine::core::quant::Glm53GroupedMoePrefillScope::new(use_grouped_moe)
                });
                let _pair_scope = (moe_pair != "auto").then(|| {
                    lisa_engine::core::quant::Glm53GroupedMoePairScope::new(use_split_pair)
                });
                let _q2_load_scope = (moe_q2_load != "vector").then(|| {
                    lisa_engine::core::quant::Glm53GroupedMoeQ2LoadScope::new(use_vector_q2_load)
                });
                let _down_tile_scope = (moe_down_tile != "auto").then(|| {
                    lisa_engine::core::quant::Glm53GroupedMoeDownTileScope::new(down_tile)
                });
                let mut caches = tower.new_caches();
                tower.clear_context();
                let t = std::time::Instant::now();
                let last = tower.prefill(&prompt, &mut caches)?;
                let logits = tower.head(&last)?;
                let next = lisa_mlx::ops::indexing::argmax(&logits, None)?;
                next.eval().map_err(|e| anyhow::anyhow!("{e}"))?;
                let next_token = next.item_cast::<i32>() as u32;
                let secs = t.elapsed().as_secs_f64();
                times.push((
                    use_ds4_mpp,
                    use_grouped_moe,
                    use_split_pair,
                    use_vector_q2_load,
                    secs,
                    down_tile,
                ));
                println!(
                    "iter {i:2} q8={q8_name:>7} moe={moe_name:>7} pair={pair_name:>5} q2={q2_load_name:>6} down={down_tile:>3}: {:>7.1} tok/s ({:>7.1} ms), next_token={next_token}",
                    n as f64 / secs,
                    secs * 1e3
                );
            }
            let median = |mut values: Vec<f64>| {
                values.sort_by(|a, b| a.partial_cmp(b).unwrap());
                if values.len() % 2 == 0 {
                    (values[values.len() / 2 - 1] + values[values.len() / 2]) * 0.5
                } else {
                    values[values.len() / 2]
                }
            };
            for (use_ds4_mpp, q8_name) in [(true, "ds4-mpp"), (false, "stock")] {
                for (use_grouped_moe, moe_name) in [(true, "grouped"), (false, "stock")] {
                    for (use_split_pair, pair_name) in [(true, "split"), (false, "fused")] {
                        for (use_vector_load, q2_load_name) in [(true, "vector"), (false, "scalar")]
                        {
                            let path_times: Vec<f64> = times
                                .iter()
                                .filter_map(|(q8, moe, pair, q2, seconds, _down_tile)| {
                                    (*q8 == use_ds4_mpp
                                        && *moe == use_grouped_moe
                                        && *pair == use_split_pair
                                        && *q2 == use_vector_load)
                                        .then_some(*seconds)
                                })
                                .collect();
                            if path_times.is_empty() {
                                continue;
                            }
                            let median_secs = median(path_times.clone());
                            let best_secs =
                                path_times.iter().copied().min_by(f64::total_cmp).unwrap();
                            println!(
                                "q8={q8_name:>7} moe={moe_name:>7} pair={pair_name:>5} q2={q2_load_name:>6} median {:.1} tok/s | best {:.1} tok/s ({:.1} ms)",
                                n as f64 / median_secs,
                                n as f64 / best_secs,
                                best_secs * 1e3
                            );
                        }
                    }
                }
            }
            if moe_grouped == "abba" {
                for (cycle, samples) in times.chunks_exact(4).enumerate() {
                    let grouped_ms = median(vec![samples[0].4, samples[3].4]) * 1e3;
                    let stock_ms = median(vec![samples[1].4, samples[2].4]) * 1e3;
                    println!(
                        "MoE ABBA cycle {cycle}: grouped {:.1} ms, stock {:.1} ms, delta {:+.2}%",
                        grouped_ms,
                        stock_ms,
                        (grouped_ms / stock_ms - 1.0) * 100.0
                    );
                }
            }
            if moe_down_tile == "abba" {
                for (cycle, samples) in times.chunks_exact(4).enumerate() {
                    let tile64_ms = median(vec![samples[0].4, samples[3].4]) * 1e3;
                    let tile128_ms = median(vec![samples[1].4, samples[2].4]) * 1e3;
                    println!(
                        "grouped down tile ABBA cycle {cycle}: 64-row {:.1} ms, 128-row {:.1} ms, delta {:+.2}%",
                        tile64_ms,
                        tile128_ms,
                        (tile128_ms / tile64_ms - 1.0) * 100.0
                    );
                }
            }
            if q8_prefill == "abba" {
                for (use_grouped_moe, moe_name) in [(true, "grouped"), (false, "stock")] {
                    let q8_times: Vec<f64> = times
                        .iter()
                        .filter_map(|(_, moe, _, _, seconds, _down_tile)| {
                            (*moe == use_grouped_moe).then_some(*seconds)
                        })
                        .collect();
                    for (cycle, samples) in q8_times.chunks_exact(4).enumerate() {
                        let mpp_ms = median(vec![samples[0], samples[3]]) * 1e3;
                        let stock_ms = median(vec![samples[1], samples[2]]) * 1e3;
                        println!(
                            "Q8 ABBA ({moe_name}) cycle {cycle}: ds4-mpp {:.1} ms, stock {:.1} ms, delta {:+.2}%",
                            mpp_ms,
                            stock_ms,
                            (mpp_ms / stock_ms - 1.0) * 100.0
                        );
                    }
                }
            }
            if moe_pair == "abba" {
                for (cycle, samples) in times.chunks_exact(4).enumerate() {
                    let fused_ms = median(vec![samples[0].4, samples[3].4]) * 1e3;
                    let split_ms = median(vec![samples[1].4, samples[2].4]) * 1e3;
                    println!(
                        "MoE pair ABBA cycle {cycle}: fused {:.1} ms, split {:.1} ms, delta {:+.2}%",
                        fused_ms,
                        split_ms,
                        (split_ms / fused_ms - 1.0) * 100.0
                    );
                }
            }
            if moe_q2_load == "abba" {
                for (cycle, samples) in times.chunks_exact(4).enumerate() {
                    let vector_ms = median(vec![samples[0].4, samples[3].4]) * 1e3;
                    let scalar_ms = median(vec![samples[1].4, samples[2].4]) * 1e3;
                    println!(
                        "MoE Q2-load ABBA cycle {cycle}: vector {:.1} ms, scalar {:.1} ms, delta {:+.2}%",
                        vector_ms,
                        scalar_ms,
                        (vector_ms / scalar_ms - 1.0) * 100.0
                    );
                }
            }
            Ok(())
        }
        Command::Golden {
            model,
            golden,
            max_tokens,
            depth,
            pld,
            pld_depth,
            capture,
        } => {
            let pld = pld_enable(pld, pld_depth);
            // The golden is cache-only: it never triggers a model download.
            let model_dir = resolve_cached_model(&model)?;
            // qwen3_5 (27B) golden over every case in the file; `--depth 2..=7`
            // runs the same cases through the native MTP head (lossless greedy
            // verify). `--capture` writes the serial greedy baselines.
            if lisa_engine::models::model_type_of(&model_dir)?.starts_with("qwen3_5") {
                anyhow::ensure!(!capture || depth == 0, "--capture is serial greedy only");
                let content = match golden.as_deref() {
                    Some(path) => std::fs::read_to_string(path)?,
                    None => {
                        anyhow::ensure!(
                            !capture,
                            "--capture needs an explicit --golden to write into"
                        );
                        embedded_qwen35_golden(&model_dir).to_string()
                    }
                };
                return golden_qwen35(
                    &model_dir,
                    &content,
                    golden.as_deref(),
                    max_tokens,
                    capture,
                    depth,
                    pld,
                );
            }
            // qwen4 (Flash-Next) golden: one long-prompt greedy case over the
            // Tower surface. `--capture` re-captures the baseline into an
            // explicit --golden path; without it, the tracked golden embedded
            // at compile time is the default.
            let content = match golden.as_deref() {
                Some(path) => std::fs::read_to_string(path)?,
                None => {
                    anyhow::ensure!(
                        !capture,
                        "--capture needs an explicit --golden to write into"
                    );
                    QWEN4_GOLDEN.to_string()
                }
            };
            if capture {
                return golden_qwen4_capture(&model_dir, &content, golden.as_deref(), max_tokens);
            }
            let g: serde_json::Value = serde_json::from_str(&content)?;
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

            let config = config::ModelConfig::from_json(&model_dir.join("config.json"))?;
            let mut tower = model::Tower::load(&model_dir, config)?;
            generate::set_eos_ids(vec![248_046, 248_044]);

            let n = max_tokens.min(expected.len());
            let (generated, prefill_tps, decode_tps, spec_line) = if depth > 0 {
                let mut sess = session::Session::new(&mut tower);
                let t0 = std::time::Instant::now();
                let g = sess.generate_mtp(&mut tower, &prompt, n, depth, &mut |_t| Ok(()))?;
                let secs = t0.elapsed().as_secs_f64();
                (
                    g,
                    0.0,
                    0.0,
                    Some(format!("mtp depth {depth} in {secs:.2}s")),
                )
            } else {
                let mut greedy = sampler::Sampler::default();
                let (g, s) =
                    generate::generate(&mut tower, &prompt, n, &mut greedy, pld, &mut |_t| Ok(()))?;
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
            if depth > 0 {
                // Governed wide-verify semantics (AGENTS.md §9.6): the verify
                // runs one S=depth+1 forward while the golden was captured
                // serially (S=1), so near-tie tokens can flip and cascade.
                // Divergence here is expected and deterministic — do not
                // "fix" it by narrowing the verify.
                println!(
                    "note: --depth {depth} verifies wide (S={}); near-tie flips vs the serial golden are governed semantics (AGENTS.md §9.6)",
                    depth + 1
                );
            }
            println!("expected head:  {:?}", &expected[..expected.len().min(12)]);
            println!(
                "generated head: {:?}",
                &generated[..generated.len().min(12)]
            );
            if let Some(line) = spec_line {
                println!("{line}");
            }
            println!(
                "prefill: {:.1} tok/s | decode: {:.1} tok/s",
                prefill_tps, decode_tps
            );
            Ok(())
        }
    }
}

/// The tracked qwen3_5 golden (the single source of truth on disk is
/// `crates/lisa-bench/golden/qwen3_5.json`; this copy is embedded at compile
/// time so `lisa-bench golden` is self-contained against a cached model).
const QWEN35_GOLDEN: &str = include_str!("../golden/qwen3_5.json");
/// The MiMo 9B's own baselines. Two checkpoints share `model_type = qwen3_5`, so
/// goldens are keyed by MODEL, not by type: `qwen3_5.json` names the 27B and
/// scoring the 9B against it produced a `37/310 MISMATCH` that said nothing about
/// the 9B. Pick the embedded golden that names the model under test.
const MIMO9B_GOLDEN: &str = include_str!("../golden/mimo_9b.json");

/// The embedded `qwen3_5`-family golden that names `dir`, else the 27B default
/// (which the guard in `golden_qwen35` then refuses if it does not match).
fn embedded_qwen35_golden(dir: &std::path::Path) -> &'static str {
    let path = dir.to_string_lossy();
    for g in [QWEN35_GOLDEN, MIMO9B_GOLDEN] {
        let Ok(v) = serde_json::from_str::<serde_json::Value>(g) else {
            continue;
        };
        let Some(m) = v["model"].as_str() else {
            continue;
        };
        let short = m.rsplit('/').next().unwrap_or(m);
        if path.contains(&m.replace('/', "--")) || path.contains(short) {
            return g;
        }
    }
    QWEN35_GOLDEN
}

/// Resolve a golden `--model` value to a local directory WITHOUT downloading:
/// an existing path is used as-is; a repo id must already be in the HF hub
/// cache (the golden is an integrity check, not a fetcher).
fn resolve_cached_model(target: &str) -> anyhow::Result<std::path::PathBuf> {
    use lisa_engine::models::hf;
    let direct = std::path::Path::new(target);
    if direct.is_dir() {
        return Ok(direct.to_path_buf());
    }
    anyhow::ensure!(
        target.contains('/') && !target.starts_with('~') && !target.starts_with('/'),
        "unknown model {target:?}: pass a local path or a Hugging Face repo id"
    );
    let snapshots = hf::repo_dir(target).join("snapshots");
    if let Some(dir) = std::fs::read_dir(&snapshots).ok().and_then(|entries| {
        entries
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| hf::is_model_dir(p))
            .max()
    }) {
        return Ok(dir);
    }
    anyhow::bail!(
        "model {target} is not in the HF hub cache and the golden never downloads.\n\
         Fetch it first, e.g.: hf download {target}"
    )
}

/// The tracked qwen4 golden (same layout as the qwen3_5 golden; embedded at
/// compile time so `lisa-bench golden` is self-contained against a cached model).
const QWEN4_GOLDEN: &str = include_str!("../golden/qwen4.json");

/// Capture the qwen4 golden baseline: greedy tokens over cases[0]'s `prompt`
/// (raw-encoded, no chat wrapper), written back as `prompt_tokens` /
/// `expected_tokens` into the explicit --golden path.
fn golden_qwen4_capture(
    model_dir: &std::path::Path,
    content: &str,
    writeback: Option<&std::path::Path>,
    max_tokens: usize,
) -> anyhow::Result<()> {
    let path = writeback.ok_or_else(|| anyhow::anyhow!("--capture needs an explicit --golden"))?;
    let mut g: serde_json::Value = serde_json::from_str(content)?;
    {
        let case = g["cases"][0]
            .as_object_mut()
            .ok_or_else(|| anyhow::anyhow!("golden file has no cases[0]"))?;
        let prompt_text = case
            .get("prompt")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("capture needs cases[0].prompt text"))?;
        let tok = tokenizer::Tokenizer::load(model_dir)?;
        let text = tokenizer::generation_prompt(&[("user".to_string(), prompt_text.to_string())]);
        let prompt_ids: Vec<u32> = tok.encode(&text, false)?;
        let config = config::ModelConfig::from_json(&model_dir.join("config.json"))?;
        let mut tower = model::Tower::load(model_dir, config)?;
        generate::set_eos_ids(vec![248_046, 248_044]);
        tower.warmup(&prompt_ids[..prompt_ids.len().min(64)])?;
        let mut greedy = sampler::Sampler::default();
        let (generated, _stats) = generate::generate(
            &mut tower,
            &prompt_ids,
            max_tokens,
            &mut greedy,
            0,
            &mut |_t| Ok(()),
        )?;
        case.insert(
            "prompt_tokens".to_string(),
            serde_json::Value::Array(prompt_ids.iter().map(|t| serde_json::json!(t)).collect()),
        );
        case.insert(
            "expected_tokens".to_string(),
            serde_json::Value::Array(generated.iter().map(|t| serde_json::json!(t)).collect()),
        );
        println!(
            "captured {}: {} prompt + {} generated tokens",
            case.get("name").and_then(|v| v.as_str()).unwrap_or("case"),
            prompt_ids.len(),
            generated.len()
        );
    }
    std::fs::write(path, serde_json::to_string_pretty(&g)?)?;
    println!("golden written: {}", path.display());
    Ok(())
}

/// The qwen3_5 golden: greedy tokens per case over the generic `LanguageModel`
/// surface. `capture` fills `prompt_tokens`/`expected_tokens` from a greedy
/// generation (the baseline pass; written back to `writeback`); otherwise every
/// case is replayed and compared token-for-token.
fn golden_qwen35(
    model: &std::path::Path,
    content: &str,
    writeback: Option<&std::path::Path>,
    max_tokens: usize,
    capture: bool,
    depth: usize,
    pld: usize,
) -> anyhow::Result<()> {
    let mut g: serde_json::Value = serde_json::from_str(content)?;
    // A golden declares the model it was captured FROM. Comparing a different
    // model against it is meaningless and produces a MISMATCH that says nothing
    // about the model under test — which is exactly what happened when the 9B
    // (same `model_type` as the 27B) was run against the 27B's embedded baseline
    // and scored 37/310. Goldens are model-specific, not model_type-specific:
    // refuse rather than lie. Only the embedded default is guarded — an explicit
    // `--golden <path>` is the caller's call (a local capture has no HF path).
    if !capture && writeback.is_none() {
        if let Some(gm) = g["model"].as_str() {
            let dir = model.to_string_lossy();
            let slug = gm.replace('/', "--");
            let short = gm.rsplit('/').next().unwrap_or(gm);
            if !dir.contains(&slug) && !dir.contains(short) {
                anyhow::bail!(
                    "the embedded golden is for `{gm}` but the model under test is \
                     `{dir}` — goldens are model-specific, not model_type-specific. \
                     Run `--capture --golden <path>` to record this model's own \
                     baseline, or pass --golden <path> to compare against another."
                );
            }
        }
    }
    let mut tower = match lisa_engine::models::load_dir(model)? {
        lisa_engine::models::Loaded::Language(m) => m,
        lisa_engine::models::Loaded::Decision(_) => {
            anyhow::bail!("golden expects a language model")
        }
    };
    let tok = tokenizer::Tokenizer::load(model)?;
    generate::set_eos_ids_for_model(tok.im_end_ids.clone(), tower.as_ref());

    let cases = g["cases"]
        .as_array_mut()
        .ok_or_else(|| anyhow::anyhow!("golden file has no cases array"))?;
    let mut total_matches = 0usize;
    let mut total_tokens = 0usize;
    let mut all_ok = true;
    for case in cases {
        let name = case["name"].as_str().unwrap_or("case").to_string();
        let prompt_text = case["prompt"].as_str().unwrap_or_default().to_string();
        let text = tokenizer::generation_prompt(&[("user".to_string(), prompt_text.clone())]);
        let prompt_ids: Vec<u32> = tok.encode(&text, false)?;
        tower.warmup(&prompt_ids[..prompt_ids.len().min(64)])?;

        let case_max = case["max_tokens"]
            .as_u64()
            .map(|v| v as usize)
            .unwrap_or(max_tokens);
        let n = if capture {
            case_max
        } else {
            let expected_len = case["expected_tokens"]
                .as_array()
                .map(|a| a.len())
                .unwrap_or(0);
            case_max.min(expected_len)
        };

        let mut greedy = sampler::Sampler::default();
        let t0 = std::time::Instant::now();
        let (generated, stats) = if depth > 0 {
            let mut sess = session::Session::new(&mut *tower);
            let g = sess.generate_mtp(&mut *tower, &prompt_ids, n, depth, &mut |_t| Ok(()))?;
            (g, None)
        } else {
            let (g, s) =
                generate::generate(&mut *tower, &prompt_ids, n, &mut greedy, pld, &mut |_t| {
                    Ok(())
                })?;
            (g, Some(s))
        };
        let secs = t0.elapsed().as_secs_f64();

        if capture {
            case["prompt_tokens"] =
                serde_json::Value::Array(prompt_ids.iter().map(|t| serde_json::json!(t)).collect());
            case["expected_tokens"] =
                serde_json::Value::Array(generated.iter().map(|t| serde_json::json!(t)).collect());
            println!(
                "captured {name}: {} prompt + {} generated tokens",
                prompt_ids.len(),
                generated.len()
            );
            continue;
        }

        let expected: Vec<u32> = case["expected_tokens"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|v| v.as_u64().map(|x| x as u32))
            .collect();
        let matches = generated
            .iter()
            .zip(expected.iter())
            .filter(|(a, b)| a == b)
            .count();
        let first_bad = generated
            .iter()
            .zip(expected.iter())
            .position(|(a, b)| a != b);
        let tps = stats.map(|s| s.decode_tps());
        println!(
            "case {name}: matches {matches}/{} ({:.1}%) | first mismatch at {:?} | decode {} ({:.2}s)",
            generated.len(),
            100.0 * matches as f64 / generated.len().max(1) as f64,
            first_bad,
            match tps {
                Some(t) => format!("{t:.1} tok/s"),
                None => format!("depth {depth}"),
            },
            secs,
        );
        if first_bad.is_some() || generated.len() != expected.len() {
            all_ok = false;
            println!(
                "  expected head:  {:?}",
                &expected[..expected.len().min(12)]
            );
            println!(
                "  generated head: {:?}",
                &generated[..generated.len().min(12)]
            );
        }
        total_matches += matches;
        total_tokens += generated.len();
    }
    if capture {
        let path = writeback
            .ok_or_else(|| anyhow::anyhow!("--capture needs an explicit --golden to write into"))?;
        std::fs::write(path, serde_json::to_string_pretty(&g)?)?;
        println!("golden written: {}", path.display());
    } else {
        println!(
            "TOTAL: {total_matches}/{total_tokens} ({:.1}%) {}",
            100.0 * total_matches as f64 / total_tokens.max(1) as f64,
            if all_ok { "OK" } else { "MISMATCH" },
        );
        anyhow::ensure!(all_ok, "qwen3_5 golden mismatch");
    }
    Ok(())
}
