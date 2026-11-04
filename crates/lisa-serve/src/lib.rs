//! Minimal OpenAI-compatible serving layer.
//!
//! No async runtime: an accept thread creates connection handlers that parse
//! HTTP requests and enqueue jobs over an `mpsc` channel. One worker owns the
//! [`LanguageModel`] and its thread-affine arrays, processing eligible request
//! waves or serial completions. Connection handlers frame replies as chunked
//! SSE or collect them into JSON responses.
//!
//! Requests can reuse prompt snapshots or an explicit `session_id`. Models
//! without batch decode support use the serial, cache-aware completion path.

use std::collections::HashMap;
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::{Value, json};

mod api_state;
mod http;
pub mod images;
mod responses;
use api_state::ApiState;
use http::{read_request, write_response};
#[cfg(test)]
use responses::{ReplyData, prompt_tokens_details_json, usage_json};
use responses::{
    collect_choices, collect_surface, drain_reply, empty_response_resource,
    responses_payload_with_id, store_conversation, stream_surface,
};
mod deepseek41;
mod streaming;

use lisa_engine::DecisionModel;
use lisa_engine::core::generate::is_eos;
use lisa_engine::core::prefix_cache::PrefixCache;
use lisa_engine::core::sampler::{Sampler, token_logprob};
use lisa_engine::core::session::{MtpDepth, Session, TargetMtpEosPolicy, TargetMtpSamplingMode};
use lisa_engine::core::tokenizer::{self, IncrementalDecoder, Tokenizer};
use lisa_engine::models::LanguageModel;

pub struct ServerConfig {
    pub addr: String,
    pub max_tokens: usize,
    /// Request/flag-level sampling overrides; `None` falls through to the
    /// checkpoint's `generation_config.json` recommendations, then hardcoded
    /// greedy defaults (see `sampler_from`).
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<usize>,
    pub min_p: Option<f32>,
    pub rep_penalty: Option<f32>,
    /// Enable DS4's exact stochastic acceptance/residual sampling for GLM MTP.
    pub mtp_exact_sampling: bool,
    /// The checkpoint's own sampling/thinking recommendations
    /// (`generation_config.json`): `temperature`, `top_p`, `top_k`,
    /// `repetition_penalty`, and `default_chat_template_kwargs
    /// .enable_thinking` (the model author's thinking default).
    pub gen_defaults: GenDefaults,
    pub depth: usize,
    /// Prompt-lookup (PLD) speculative draft depth (0 = off; greedy only).
    pub pld: usize,
    /// Cross-request prompt prefix cache: max entries (LRU; 0 = off).
    pub prefix_cache: usize,
    /// Maximum streams stepped together (admission queue width).
    pub max_batch: usize,
    /// The checkpoint's `chat_template.jinja`, when available.
    pub chat_template: Option<String>,
    /// The loaded model's id as advertised on `/v1/models` and echoed in
    /// response payloads. Derived from the resolved checkpoint (never a
    /// hardcoded placeholder).
    pub model_id: String,
    /// Whether the loaded language checkpoint includes a native vision tower.
    pub supports_images: bool,
}

/// The id of the model loaded from `dir`: the Hugging Face repo id when the
/// checkpoint lives in a hub cache (`models--ORG--NAME/snapshots/...`), the
/// directory's own name otherwise.
pub fn model_id_of(dir: &std::path::Path) -> String {
    for anc in dir.ancestors() {
        if let Some(name) = anc.file_name().and_then(|n| n.to_str()) {
            if let Some(rest) = name.strip_prefix("models--") {
                return rest.replacen("--", "/", 1);
            }
        }
    }
    dir.file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("unknown")
        .to_string()
}

/// The checkpoint's own defaults, from `generation_config.json`. `None` fields
/// fall through to the hardcoded greedy defaults. `thinking` is
/// `default_chat_template_kwargs.enable_thinking` — the model author's
/// declared thinking default for silent requests.
#[derive(Default, Debug, Clone)]
pub struct GenDefaults {
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<usize>,
    pub rep_penalty: Option<f32>,
    pub thinking: Option<bool>,
}

impl GenDefaults {
    /// Parse a checkpoint's `generation_config.json`. Missing file or fields
    /// are fine — every default stays `None`/greedy.
    pub fn from_model_dir(dir: &std::path::Path) -> Self {
        let Ok(text) = std::fs::read_to_string(dir.join("generation_config.json")) else {
            return Self::default();
        };
        let Ok(v) = serde_json::from_str::<Value>(&text) else {
            eprintln!("[serve] generation_config.json: unparseable, ignoring");
            return Self::default();
        };
        let f = |k: &str| v.get(k).and_then(|x| x.as_f64()).map(|x| x as f32);
        let mut out = Self {
            temperature: f("temperature"),
            top_p: f("top_p"),
            top_k: v
                .get("top_k")
                .and_then(|x| x.as_i64())
                .and_then(|x| usize::try_from(x).ok()),
            rep_penalty: f("repetition_penalty"),
            thinking: None,
        };
        if let Some(et) = v
            .get("default_chat_template_kwargs")
            .and_then(|c| c.get("enable_thinking"))
            .and_then(|x| x.as_bool())
        {
            out.thinking = Some(et);
        }
        out
    }
}

/// Sentinel prefixed onto a refusal carried through `anyhow` so the HTTP layer
/// can tell "out of budget" (503) from a genuine engine failure (500).
const REFUSED_MARKER: &str = "\u{1}refused\u{1}";

/// HTTP status for a drained-reply error: 503 when the engine refused for
/// memory (O3), 500 otherwise.
fn http_status_for(e: &anyhow::Error) -> &'static str {
    if e.to_string().starts_with(REFUSED_MARKER) {
        "503 Service Unavailable"
    } else {
        "500 Internal Server Error"
    }
}

/// User-facing message for a drained-reply error: strips the refusal sentinel
/// (and the `retry-after=` prefix) so it never leaks into the JSON body.
fn error_message_for(e: &anyhow::Error) -> String {
    match e.to_string().strip_prefix(REFUSED_MARKER) {
        Some(rest) => rest
            .split_once("; ")
            .map(|(_, m)| m.to_string())
            .unwrap_or_else(|| rest.to_string()),
        None => e.to_string(),
    }
}

enum Reply {
    /// Sent once at the start of a streamed reply, carrying the prompt size.
    Start {
        prompt_tokens: usize,
    },
    /// A streamed content or reasoning delta, optionally carrying a per-token logprob entry.
    Delta {
        text: String,
        reasoning: Option<String>,
        logprobs: Option<Value>,
    },
    Done {
        content: String,
        reasoning: Option<String>,
        tool_calls: Vec<Value>,
        prompt_tokens: usize,
        completion_tokens: usize,
        /// Prompt tokens served from the cross-request prefix cache (O2/P2).
        /// Reported as OpenAI's `prompt_tokens_details.cached_tokens` — the only
        /// way a client (and llmprobe) can SEE that the cache engaged.
        cached_tokens: usize,
        finish: String,
        logprobs: Option<Value>,
        stop_sequence: Option<String>,
    },
    Error(String),
    /// O3: the request was refused BEFORE any forward because it would exceed
    /// the memory budget. Maps to HTTP 503, not 500.
    Refused {
        retry_after_s: u64,
        reason: String,
    },
}

/// Which HTTP surface a job came in on. The engine path is shared; only the
/// response serialization differs.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Surface {
    Chat,
    Completions,
    Messages,
    Responses,
    CountTokens,
    Embeddings,
}

struct Job {
    body: Value,
    stream: bool,
    /// When set, the request continues a persistent multi-turn conversation
    /// (prefix reuse); it is handled on its own `Session`, not in a batch.
    session_id: Option<String>,
    surface: Surface,
    reply: Sender<Reply>,
}

pub fn run(
    tower: &mut dyn LanguageModel,
    tok: &Tokenizer,
    cfg: ServerConfig,
) -> anyhow::Result<()> {
    run_with_stop(tower, tok, cfg, std::sync::Arc::new(AtomicBool::new(false)))
}

/// A non-blocking view of the HTTP server, for hosts that also need the engine
/// (e.g. the desktop UI). `start` binds the socket and spawns the accept thread;
/// the caller drives inference by calling [`Server::step`] whenever convenient,
/// so the same engine thread can interleave its own work with served requests.
/// Requests are serialized (the model is single-threaded); a chat turn waits at
/// most for an in-flight HTTP request, and vice versa.
pub struct Server {
    cfg: std::sync::Arc<ServerConfig>,
    jobs: Receiver<Job>,
    sessions: HashMap<String, (Session, Vec<ChatMessage>)>,
    prefix: PrefixCache,
    chat_tmpl: Option<ChatTemplate>,
    stop: std::sync::Arc<AtomicBool>,
    /// Held so the bound port stays open until [`Server::shutdown`] drops it.
    listener: Option<std::sync::Arc<TcpListener>>,
    accept: Option<std::thread::JoinHandle<()>>,
    addr: std::net::SocketAddr,
    /// O3: byte-level admission budget. `None` = the kill-switch is off (no
    /// `LISA_RAM_CAP_GB`), in which case no request is ever refused.
    admission: Option<lisa_engine::core::admission::MemoryBudget>,
}

impl Server {
    pub fn start(cfg: ServerConfig, stop: std::sync::Arc<AtomicBool>) -> anyhow::Result<Server> {
        let api = ApiState::new(cfg.model_id.clone());
        let listener =
            TcpListener::bind(&cfg.addr).map_err(|e| anyhow::anyhow!("bind {}: {e}", cfg.addr))?;
        listener.set_nonblocking(true)?;
        let addr = listener.local_addr()?;
        let chat_tmpl = cfg
            .chat_template
            .as_deref()
            .map(ChatTemplate::new)
            .transpose()?;

        let (tx, rx) = mpsc::channel::<Job>();
        let prefix = PrefixCache::new(cfg.prefix_cache);
        let cfg = std::sync::Arc::new(cfg);
        // Share the listener with the accept thread but keep it owned here too,
        // so joining that thread on shutdown actually closes the socket before
        // the `Server` drops (otherwise a stop→start toggle can race into
        // `Address already in use`).
        let listener = std::sync::Arc::new(listener);
        let accept = {
            let cfg = cfg.clone();
            let tx = tx.clone();
            let stop = stop.clone();
            let listener = listener.clone();
            std::thread::spawn(move || {
                while !stop.load(Ordering::Relaxed) {
                    match listener.accept() {
                        Ok((stream, _)) => {
                            let _ = stream.set_nonblocking(false);
                            let tx = tx.clone();
                            let cfg = cfg.clone();
                            let api = api.clone();
                            std::thread::spawn(move || {
                                if let Err(e) = handle_conn(&api, stream, tx, &cfg) {
                                    eprintln!("[serve] connection error: {e}");
                                }
                            });
                        }
                        Err(_) => std::thread::sleep(std::time::Duration::from_millis(50)),
                    }
                }
            })
        };
        // O3: admission is kill-switched — armed only when the operator sets
        // `LISA_RAM_CAP_GB`, so the default path can never refuse a request.
        let admission = std::env::var_os("LISA_RAM_CAP_GB")
            .map(|_| lisa_engine::core::admission::MemoryBudget::from_host());
        Ok(Server {
            cfg,
            jobs: rx,
            sessions: HashMap::new(),
            prefix,
            chat_tmpl,
            stop,
            listener: Some(listener),
            accept: Some(accept),
            addr,
            admission,
        })
    }

    pub fn local_addr(&self) -> std::net::SocketAddr {
        self.addr
    }

    /// Process one ready wave of jobs, if any. Returns `true` when it did work
    /// (so a host loop can prefer its own commands between waves).
    pub fn step(&mut self, tower: &mut dyn LanguageModel, tok: &Tokenizer) -> bool {
        self.step_with_context(tower, tok, |_| {})
    }

    /// Restore a host's model context immediately before a ready wave runs.
    /// An idle poll leaves model-owned context and draft caches untouched.
    pub fn step_with_context(
        &mut self,
        tower: &mut dyn LanguageModel,
        tok: &Tokenizer,
        before_wave: impl FnOnce(&mut dyn LanguageModel),
    ) -> bool {
        let Ok(first) = self.jobs.try_recv() else {
            return false;
        };
        let mut wave = vec![first];
        while wave.len() < self.cfg.max_batch.max(1) {
            match self.jobs.try_recv() {
                Ok(j) => wave.push(j),
                Err(_) => break,
            }
        }

        // Auto speculation is a capability choice. A checkpoint without a
        // draft head uses serial generation; an explicitly requested fixed
        // depth still receives the existing unsupported-head error.
        if !tower.has_drafter() && !tower.has_target_mtp() {
            for job in &mut wave {
                if depth_for(job, &self.cfg) == lisa_engine::core::round_cost::AUTO_DEPTH {
                    job.body["depth"] = json!(0);
                }
            }
        }

        // O3: byte-level admission at wave intake, kill-switched on
        // `LISA_RAM_CAP_GB`. A request that cannot fit the budget is refused
        // BEFORE any forward (503 upstream) instead of driving the box into
        // swap. Admitted bytes cover this wave's transient KV and are released
        // at the end of the step; long-lived session state stays bounded by the
        // existing pad-waste / max-batch limits.
        let mut admitted = 0usize;
        if let Some(b) = self.admission.as_mut() {
            let kv = tower.kv_bytes_per_token();
            if kv > 0 {
                let state = tower.stream_state_bytes();
                let mut kept = Vec::with_capacity(wave.len());
                for job in wave {
                    let want = lisa_engine::core::admission::stream_bytes(
                        request_max_tokens(&job, &self.cfg),
                        kv,
                        state,
                        2,
                    );
                    if b.try_admit(want) {
                        admitted += want;
                        kept.push(job);
                    } else {
                        let _ = job.reply.send(Reply::Refused {
                            retry_after_s: 1,
                            reason: format!(
                                "out of memory budget: round needs {} MiB, headroom {} MiB",
                                want >> 20,
                                b.headroom() >> 20
                            ),
                        });
                    }
                }
                wave = kept;
            }
        }

        if !wave.is_empty() {
            before_wave(tower);
        }

        // Stateful (session) requests are serial; stateless ones batch.
        let mut sess_jobs = Vec::new();
        let mut batch_jobs = Vec::new();
        for job in wave {
            if job.session_id.is_some() {
                sess_jobs.push(job);
            } else {
                batch_jobs.push(job);
            }
        }
        let cfg = self.cfg.clone();
        let chat_tmpl = self.chat_tmpl.as_ref();
        for job in sess_jobs {
            eprintln!("[batched] job serial: session stateful (prefix reuse)");
            if let Err(e) = run_session(
                tower,
                tok,
                &mut self.sessions,
                &mut self.prefix,
                &job,
                &cfg,
                chat_tmpl,
            ) {
                let _ = job.reply.send(Reply::Error(e.to_string()));
            }
        }
        if !batch_jobs.is_empty() {
            // Qwen chain-MTP requests are serialized one stream at a time
            // (`maximumSpeculativeBatch == 1`), so N concurrent streams pay N
            // sequential prefill+decode passes. Batching them trades per-stream
            // speculation for shared work: measured at equal steps (4 streams
            // x 64 engine steps, 3 interleaved pairs)
            // batched 3.44 s vs serialized MTP 4.71 s = **1.37x**, and ~3x on
            // the worst TTFT (2183 vs 6569 ms). GLM target-MTP uses its own
            // one-cycle-per-session wave path below.
            //
            // The decision is PER WAVE, not per job: a lone request keeps its
            // speculation (which is where MTP wins). Requests that need the
            // serial paths (tools, stops, logprobs) stay serial either way.
            //
            // Crossover measured on BOTH sides (4 streams x 64 engine steps,
            // medians of 3 interleaved pairs): B=1 MTP
            // 1.11 s; B=2 MTP 2.22 s vs batched 2.81 s (MTP wins); B=4 MTP
            // 4.71 s vs batched 3.91 s (batched wins). B=3 was measured batched
            // only (3.07 s), so the threshold is 4 BY MEASUREMENT, not by
            // interpolation. The batched path emits different tokens than the
            // speculative one (near-tie class, AGENTS.md §9.6).
            const BATCH_WAVE_MIN: usize = 4;
            let target_mtp_model = tower.has_target_mtp();
            let (spec, wave_jobs): (Vec<Job>, Vec<Job>) =
                if !tower.batch_decode_ok() && !target_mtp_model {
                    // A single-stream tower gains no batching work. Use the
                    // completion path so its native prefix snapshots are reused.
                    (batch_jobs, Vec::new())
                } else if target_mtp_model {
                    // Preserve GLM's two-draft target-MTP cycle for eligible
                    // multi-request waves, interleaving one verified round
                    // per stream.
                    batch_jobs
                        .into_iter()
                        .partition(|job| !is_target_mtp_wave_eligible(job, depth_for(job, &cfg)))
                } else if batch_jobs.len() >= BATCH_WAVE_MIN {
                    eprintln!(
                        "[batched] wave of {} jobs -> continuous batch \
                         (per-stream speculation dropped, maximumSpeculativeBatch == 1)",
                        batch_jobs.len()
                    );
                    batch_jobs.into_iter().partition(|j| !is_wave_eligible(j))
                } else {
                    batch_jobs.into_iter().partition(|j| !is_plain_job(j, &cfg))
                };
            for job in &spec {
                let why = if depth_for(job, &cfg) > 0 {
                    format!("mtp depth={}", depth_for(job, &cfg))
                } else if has_tools(job) {
                    "tool calls".to_string()
                } else if !request_stops(job).is_empty() {
                    "stop sequences".to_string()
                } else if wants_logprobs(&job.body) {
                    "logprobs".to_string()
                } else if !tower.batch_decode_ok() {
                    "single-stream model".to_string()
                } else {
                    "non-chat surface".to_string()
                };
                eprintln!("[batched] job serial: {why}");
                let r = if job.surface == Surface::CountTokens {
                    count_tokens_job(tok, &job, chat_tmpl, &cfg)
                } else if job.surface == Surface::Embeddings {
                    run_embeddings(&self.cfg.model_id, tower, tok, &job)
                } else {
                    complete(tower, tok, &job, &cfg, &mut self.prefix, chat_tmpl)
                };
                if let Err(e) = r {
                    let _ = job.reply.send(Reply::Error(e.to_string()));
                }
            }
            let (mtp_wave, plain) = if target_mtp_model {
                wave_jobs
                    .into_iter()
                    .partition(|job| depth_for(job, &cfg) > 0)
            } else {
                (Vec::new(), wave_jobs)
            };
            if !plain.is_empty() {
                // Late arrivals are polled mid-wave so they join the live
                // batch; anything non-plain comes back and runs serial here.
                let mut leftover: Vec<Job> = Vec::new();
                if let Err(e) = run_wave(
                    tower,
                    tok,
                    plain,
                    &cfg,
                    chat_tmpl,
                    0,
                    Some(&self.jobs),
                    &mut leftover,
                ) {
                    eprintln!("[serve] wave error: {e}");
                }
                for job in leftover {
                    let r = if job.surface == Surface::CountTokens {
                        count_tokens_job(tok, &job, chat_tmpl, &cfg)
                    } else if job.surface == Surface::Embeddings {
                        run_embeddings(&self.cfg.model_id, tower, tok, &job)
                    } else {
                        complete(tower, tok, &job, &cfg, &mut self.prefix, chat_tmpl)
                    };
                    if let Err(e) = r {
                        let _ = job.reply.send(Reply::Error(e.to_string()));
                    }
                }
            }
            if mtp_wave.len() == 1 {
                let job = mtp_wave.into_iter().next().expect("one MTP request");
                if let Err(e) = complete(tower, tok, &job, &cfg, &mut self.prefix, chat_tmpl) {
                    let _ = job.reply.send(Reply::Error(e.to_string()));
                }
            } else if !mtp_wave.is_empty() {
                // GLM uses a fixed two-draft cycle. A supported positive depth
                // selects this model path while each request retains its own
                // sampler and output budget.
                let mut leftover = Vec::new();
                if let Err(e) = run_wave(
                    tower,
                    tok,
                    mtp_wave,
                    &cfg,
                    chat_tmpl,
                    2,
                    // The admission ledger currently reserves the initial
                    // wave as a unit. Defer late requests when it is armed.
                    self.admission.is_none().then_some(&self.jobs),
                    &mut leftover,
                ) {
                    eprintln!("[serve] GLM MTP wave error: {e}");
                }
                for job in leftover {
                    let result = if job.surface == Surface::CountTokens {
                        count_tokens_job(tok, &job, chat_tmpl, &cfg)
                    } else if job.surface == Surface::Embeddings {
                        run_embeddings(&self.cfg.model_id, tower, tok, &job)
                    } else {
                        complete(tower, tok, &job, &cfg, &mut self.prefix, chat_tmpl)
                    };
                    if let Err(e) = result {
                        let _ = job.reply.send(Reply::Error(e.to_string()));
                    }
                }
            }
        }
        if let Some(b) = self.admission.as_mut() {
            b.release(admitted);
        }
        true
    }

    /// Stop accepting and join the accept thread. The port is freed when the
    /// `Server` itself is dropped (it holds the last listener reference).
    pub fn shutdown(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(h) = self.accept.take() {
            let _ = h.join();
        }
        // Drop the last listener reference so the port is free immediately.
        self.listener = None;
    }
}

/// Like [`run`], but returns once `stop` is set (checked between waves).
pub fn run_with_stop(
    tower: &mut dyn LanguageModel,
    tok: &Tokenizer,
    mut cfg: ServerConfig,
    stop: std::sync::Arc<AtomicBool>,
) -> anyhow::Result<()> {
    cfg.supports_images = tower.supports_images();
    lisa_engine::core::generate::set_eos_ids_for_model(tok.im_end_ids.clone(), tower);
    let mut server = Server::start(cfg, stop.clone())?;
    println!("lisa serve listening on http://{}", server.local_addr());
    println!("  POST /v1/chat/completions   GET /v1/models   GET /health");
    println!("  max_batch {}", server.cfg.max_batch);
    if server.chat_tmpl.is_some() {
        println!("  chat_template.jinja loaded");
    }
    while !stop.load(Ordering::Relaxed) {
        if !server.step(tower, tok) {
            std::thread::sleep(std::time::Duration::from_millis(50));
        }
    }
    server.shutdown();
    Ok(())
}

fn depth_for(job: &Job, cfg: &ServerConfig) -> usize {
    job.body
        .get("depth")
        .and_then(|v| v.as_u64())
        .map(|v| v as usize)
        .unwrap_or(cfg.depth)
}

/// Map a configured depth to the generation policy: the `auto` sentinel
/// becomes the EV controller when the tower ships a drafter, fixed otherwise.
fn mtp_depth(depth: usize, tower: &dyn LanguageModel) -> MtpDepth {
    if depth == lisa_engine::core::round_cost::AUTO_DEPTH && tower.has_drafter() {
        MtpDepth::Auto
    } else {
        MtpDepth::Fixed(depth.clamp(2, 6))
    }
}

/// Resolve the request's draft depth, rejecting depth 1 (a single draft never
/// pays for a speculative round; use 0 or 2..=6).
fn resolve_depth(job: &Job, cfg: &ServerConfig) -> anyhow::Result<usize> {
    let depth = depth_for(job, cfg);
    anyhow::ensure!(
        depth != 1,
        "depth 1 is not supported; use 0 (serial) or 2..=6 (MTP)"
    );
    Ok(depth)
}

/// Build the scheduler request for a stateless HTTP job.
fn build_request(
    id: usize,
    tok: &Tokenizer,
    job: &Job,
    cfg: &ServerConfig,
    tmpl: Option<&ChatTemplate>,
) -> anyhow::Result<lisa_engine::core::sched::Request> {
    let messages = parse_messages(&job.body)?;
    let prompt = render_prompt(
        tmpl,
        &messages,
        true,
        &request_tools(job),
        request_thinking(&job.body, cfg),
    )?;
    let ids = tok.encode(&prompt, false)?;
    let max_tokens = request_max_tokens(job, cfg);
    Ok(lisa_engine::core::sched::Request {
        id,
        prompt: ids,
        max_tokens,
        sampler: sampler_from(&job.body, cfg),
        ignore_eos: job
            .body
            .get("ignore_eos")
            .and_then(|v| v.as_bool())
            .unwrap_or(false),
    })
}

/// Whether a job can use a stateless scheduler wave: a plain chat completion
/// that needs none of the features only the serial paths implement (stop
/// sequences, per-token logprobs, tool calls and structured output).
fn is_plain_job(job: &Job, cfg: &ServerConfig) -> bool {
    depth_for(job, cfg) == 0 && is_wave_eligible(job)
}

/// GLM target-MTP can join the speculative scheduler wave unless request
/// semantics require the serial completion path.
fn is_target_mtp_wave_eligible(job: &Job, depth: usize) -> bool {
    is_wave_eligible(job)
        && depth != 1
        && !(depth > 0
            && job
                .body
                .get("ignore_eos")
                .and_then(|value| value.as_bool())
                .unwrap_or(false))
}

/// Everything [`is_plain_job`] requires EXCEPT the depth clause: the features
/// that genuinely need the serial paths (tool calls, stop sequences, per-token
/// logprobs, structured output, non-chat surfaces, persistent sessions). The wave-level batching
/// policy may drop a request's SPECULATION, but never these.
fn is_wave_eligible(job: &Job) -> bool {
    !lisa_engine::models::vision::messages_have_images(&job.body)
        && !has_tools(job)
        && job.surface == Surface::Chat
        && job.session_id.is_none()
        && request_stops(job).is_empty()
        && !wants_logprobs(&job.body)
        && response_format_of(&job.body).is_none()
}

/// Whether a request asked for per-token logprobs.
fn wants_logprobs(body: &Value) -> bool {
    body.get("logprobs")
        .and_then(|v| v.as_bool())
        .unwrap_or(false)
        || body
            .get("top_logprobs")
            .and_then(|v| v.as_u64())
            .unwrap_or(0)
            > 0
}

/// Serve a wave of stateless requests through the generation scheduler,
/// streaming each request's tokens back to its own connection. `depth == 0`
/// uses continuous batching; a positive depth uses per-request speculation.
/// `late`, when given, is polled between plain decode rounds so eligible
/// arrivals can join the live batch. Other late jobs return to the caller.
fn run_wave(
    tower: &mut dyn LanguageModel,
    tok: &Tokenizer,
    jobs: Vec<Job>,
    cfg: &ServerConfig,
    tmpl: Option<&ChatTemplate>,
    depth: usize,
    late: Option<&Receiver<Job>>,
    leftover: &mut Vec<Job>,
) -> anyhow::Result<()> {
    // Job registry: slot i is job i; `dead[i]` jobs already errored.
    let jobs: Vec<Job> = jobs;
    let mut dead: Vec<bool> = vec![false; jobs.len()];
    let mut reqs: Vec<lisa_engine::core::sched::Request> = Vec::with_capacity(jobs.len());
    for (i, job) in jobs.iter().enumerate() {
        match build_request(i, tok, job, cfg, tmpl) {
            Ok(r) => reqs.push(r),
            Err(e) => {
                dead[i] = true;
                let _ = job.reply.send(Reply::Error(e.to_string()));
            }
        }
    }
    if reqs.is_empty() && late.is_none() {
        return Ok(());
    }
    let n = jobs.len();
    let mut plens: Vec<usize> = vec![0; n];
    for r in &reqs {
        plens[r.id] = r.prompt.len();
    }
    // Shared between the token callback and the late-arrival poll (which both
    // run inside the scheduler loop) — RefCell keeps the borrows disjoint.
    let acc: std::cell::RefCell<Vec<Vec<u32>>> = std::cell::RefCell::new(vec![Vec::new(); n]);
    let stopped: std::cell::RefCell<Vec<bool>> = std::cell::RefCell::new(vec![false; n]);
    let last: std::cell::RefCell<Vec<String>> = std::cell::RefCell::new(vec![String::new(); n]);
    let last_reasoning: std::cell::RefCell<Vec<String>> =
        std::cell::RefCell::new(vec![String::new(); n]);
    let stream_flags: std::cell::RefCell<Vec<bool>> =
        std::cell::RefCell::new(jobs.iter().map(|j| j.stream).collect());
    // `ignore_eos` must reach the wave's token callback: testing `is_eos` there
    // without consulting it truncated every forced-length batched reply at the
    // model's natural EOS (measured 53 of 64 tokens) while the engine — which
    // does honour it after the decode_round fix — kept decoding.
    let ignore_eos_flags: std::cell::RefCell<Vec<bool>> = std::cell::RefCell::new(
        jobs.iter()
            .map(|j| {
                j.body
                    .get("ignore_eos")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false)
            })
            .collect(),
    );
    let replies: Vec<Sender<Reply>> = jobs.iter().map(|j| j.reply.clone()).collect();
    let replies: std::cell::RefCell<Vec<Sender<Reply>>> = std::cell::RefCell::new(replies);
    let jobs: std::cell::RefCell<Vec<Job>> = std::cell::RefCell::new(jobs);
    let dead: std::cell::RefCell<Vec<bool>> = std::cell::RefCell::new(dead);
    let plens: std::cell::RefCell<Vec<usize>> = std::cell::RefCell::new(plens);

    // A job is admissible to the batch when it is a plain stateless chat
    // completion (see `is_plain_job`); anything else takes a serial path.
    let is_plain = |job: &Job| {
        if depth > 0 {
            depth_for(job, cfg) > 0 && is_target_mtp_wave_eligible(job, depth_for(job, cfg))
        } else {
            is_plain_job(job, cfg)
        }
    };

    // Late-arrival poll: pull plain jobs into the batch mid-wave, defer the rest.
    let decoded = std::cell::RefCell::new(
        (0..n)
            .map(|_| IncrementalDecoder::new(tok, false))
            .collect::<Vec<_>>(),
    );
    let mut next = || -> Option<lisa_engine::core::sched::Request> {
        let rx = late?;
        loop {
            let job = rx.try_recv().ok()?;
            if is_plain(&job) {
                let id = jobs.borrow().len();
                match build_request(id, tok, &job, cfg, tmpl) {
                    Ok(r) => {
                        plens.borrow_mut().push(r.prompt.len());
                        acc.borrow_mut().push(Vec::new());
                        decoded
                            .borrow_mut()
                            .push(IncrementalDecoder::new(tok, false));
                        stopped.borrow_mut().push(false);
                        last.borrow_mut().push(String::new());
                        last_reasoning.borrow_mut().push(String::new());
                        dead.borrow_mut().push(false);
                        stream_flags.borrow_mut().push(job.stream);
                        ignore_eos_flags.borrow_mut().push(
                            job.body
                                .get("ignore_eos")
                                .and_then(|v| v.as_bool())
                                .unwrap_or(false),
                        );
                        replies.borrow_mut().push(job.reply.clone());
                        jobs.borrow_mut().push(job);
                        return Some(r);
                    }
                    Err(e) => {
                        let _ = job.reply.send(Reply::Error(e.to_string()));
                    }
                }
            } else {
                leftover.push(job);
            }
        }
    };

    let mut emit = |id: usize, t: u32| -> anyhow::Result<()> {
        if is_eos(t) {
            // `ignore_eos`: EOS never ends the reply (the engine's
            // decode_round agrees after the same fix). Skip the token — it
            // is not part of the answer — and keep collecting.
            if ignore_eos_flags.borrow()[id] {
                return Ok(());
            }
            stopped.borrow_mut()[id] = true;
            return Ok(());
        }
        acc.borrow_mut()[id].push(t);
        if stream_flags.borrow()[id] {
            let mut decoded = decoded.borrow_mut();
            decoded[id].push(t)?;
            let text = decoded[id].stable_text();
            let thinking = request_thinking(&jobs.borrow()[id].body, cfg);
            let (reasoning, content) = streaming_sections(&text, thinking);
            let reasoning = reasoning.to_string();
            let content = content.to_string();
            let content_delta = stream_delta(&last.borrow()[id], &content);
            let reasoning_delta = stream_delta(&last_reasoning.borrow()[id], &reasoning);
            if content_delta.is_some() || reasoning_delta.is_some() {
                let _ = replies.borrow()[id].send(Reply::Delta {
                    text: content_delta.unwrap_or_default(),
                    reasoning: reasoning_delta,
                    logprobs: None,
                });
            }
            last.borrow_mut()[id] = content;
            last_reasoning.borrow_mut()[id] = reasoning;
        }
        Ok(())
    };
    if depth == 0 {
        lisa_engine::core::sched::run_streaming(tower, reqs, cfg.max_batch, &mut emit, &mut next)?;
    } else {
        lisa_engine::core::sched::run_speculative_streaming(
            tower,
            reqs,
            cfg.max_batch,
            depth,
            &mut emit,
            &mut next,
        )?;
    }

    let n = jobs.borrow().len();
    for id in 0..n {
        if dead.borrow()[id] {
            continue;
        }
        // Same shaping as the serial paths: always decode from the raw token
        // stream (NOT the shaped stream text — that has the `<think>` block
        // stripped, which would drop `reasoning` from the streamed Done), so a
        // streamed reply and a non-streamed one carry identical content and
        // reasoning.
        let full = tok.decode(&acc.borrow()[id])?;
        let thinking = request_thinking(&jobs.borrow()[id].body, cfg);
        let (reasoning, content) = split_reasoning_for_request(&full, thinking);
        if stream_flags.borrow()[id] {
            // Reconcile against the same request-mode parser used while
            // streaming. Completed-response shaping may classify a late
            // `</think>` marker differently; a stream cannot retract text
            // already sent in DS4's text mode.
            let (stream_reasoning, stream_content) = streaming_reply_sections(&full, thinking, &[]);
            let content_delta = stream_delta(&last.borrow()[id], &stream_content);
            let reasoning_delta = stream_delta(&last_reasoning.borrow()[id], &stream_reasoning);
            if content_delta.is_some() || reasoning_delta.is_some() {
                let _ = replies.borrow()[id].send(Reply::Delta {
                    text: content_delta.unwrap_or_default(),
                    reasoning: reasoning_delta,
                    logprobs: None,
                });
            }
        }
        let _ = replies.borrow()[id].send(Reply::Done {
            content,
            reasoning,
            tool_calls: Vec::new(),
            logprobs: None,
            stop_sequence: None,
            cached_tokens: 0,
            prompt_tokens: plens.borrow()[id],
            completion_tokens: acc.borrow()[id].len(),
            finish: if stopped.borrow()[id] {
                "stop".into()
            } else {
                "length".into()
            },
        });
    }
    Ok(())
}

/// Serve one multi-turn request on its persistent `Session`, reusing the
/// already-prefilled prefix and feeding only the new suffix.
fn run_session(
    tower: &mut dyn LanguageModel,
    tok: &Tokenizer,
    sessions: &mut HashMap<String, (Session, Vec<ChatMessage>)>,
    prefix: &mut PrefixCache,
    job: &Job,
    cfg: &ServerConfig,
    tmpl: Option<&ChatTemplate>,
) -> anyhow::Result<()> {
    let sid = job.session_id.clone().expect("session job");
    let depth = match resolve_depth(job, cfg) {
        Ok(d) => d,
        Err(e) => {
            let _ = job.reply.send(Reply::Error(e.to_string()));
            return Ok(());
        }
    };
    let messages = parse_messages(&job.body)?;
    let tools = request_tools(job);
    let thinking = request_thinking(&job.body, cfg);
    let new_str = render_prompt(tmpl, &messages, true, &tools, thinking)?;

    let (mut sess, mut conversation) = sessions
        .remove(&sid)
        .filter(|(session, _)| session.is_valid())
        .unwrap_or_else(|| {
            let mut s = Session::new(tower);
            s.enable_snapshots();
            (s, Vec::new())
        });
    sess.ensure_valid()?;
    // Cross-request prefix cache: a brand-new session whose prompt extends a
    // cached prompt token-exactly resumes at the split point.
    let ids_full = tok.encode(&new_str, false)?;
    if sess.fed.is_empty() && prefix.enabled() {
        if let Some((l, restored)) = prefix.restore_session(tower, &ids_full) {
            eprintln!(
                "[serve] session {sid}: prefix-cache resume {l}/{}",
                ids_full.len()
            );
            sess = restored;
            sess.enable_snapshots();
        }
    }
    sess.capture_snapshot(conversation.len(), tower.drafter_offset());

    // Structural match against the tracked conversation. Raw token matching
    // fails (re-encoding BPE-merges the generation prompt's newline into the
    // assistant content); message-level matching picks the right turn boundary.
    let k = structural_prefix(&messages, &conversation).min(conversation.len());
    let diverges = k < conversation.len();

    // String-suffix path (canonical template only): feed only the messages
    // after the shared prefix. On divergence, rewind the recurrent state to the
    // snapshot taken at that message boundary first.
    // Native DeepSeek uses exact token-prefix matching below: its EOS closes
    // the prior assistant, and the final emitted token may still be unfed.
    let string_suffix: Option<Vec<u32>> = if tmpl.is_some_and(|t| !t.deepseek41) && k > 0 {
        if diverges && sess.restore_snapshot(tower, k).is_none() {
            let mut s = Session::new(tower);
            s.enable_snapshots();
            sess = s;
        }
        render_prompt(tmpl, &messages[..k], false, &tools, thinking)
            .ok()
            .and_then(|prefix| new_str.strip_prefix(&prefix).map(|r| r.to_string()))
            .filter(|r| !r.is_empty())
            // The canonical GLM template has no newline between turns, and
            // DS4's rendered prompt reuses that exact boundary. Other model
            // templates can leave a closing assistant newline outside the
            // cached token prefix, so preserve the historical separator for
            // those formats only.
            .map(|rest| {
                let suffix = session_suffix_text(&new_str, &rest);
                tok.encode(&suffix, false)
            })
            .transpose()?
    } else {
        None
    };

    let suffix: Vec<u32> = match string_suffix {
        Some(sfx) if !sfx.is_empty() => {
            eprintln!(
                "[serve] session {sid}: {} reuse, {} prefilled",
                if diverges {
                    "snapshot"
                } else {
                    "string-suffix"
                },
                sfx.len()
            );
            sfx
        }
        _ => {
            let ids = ids_full.clone();
            let cp0 = common_prefix(&sess.fed, &ids);
            let fed_len = sess.fed.len();
            if cp0 < fed_len {
                let mut s = Session::new(tower);
                s.enable_snapshots();
                sess = s;
            }
            let cp = cp0.min(sess.fed.len());
            let sfx = ids[cp..].to_vec();
            eprintln!("[serve] session {sid}: full prefill {}", sfx.len());
            sfx
        }
    };
    anyhow::ensure!(!suffix.is_empty(), "session request has no new tokens");
    sess.set_target_mtp_eos_policy(TargetMtpEosPolicy::StopBeforeVerified);

    let max_tokens = request_max_tokens(job, cfg);
    let mut sampler = sampler_from(&job.body, cfg);
    let greedy = !tower.has_target_mtp() && sampler.greedy() && !sampler.mtp_exact_sampling;
    let native_dsml = tmpl.is_some_and(|t| t.deepseek41);
    let dsml_tools = !tools.is_empty() && (tower.has_target_mtp() || native_dsml);
    let streaming = job.stream;
    if streaming {
        let _ = job.reply.send(Reply::Start {
            prompt_tokens: suffix.len(),
        });
    }
    let reply = job.reply.clone();

    let stops = request_stops(job);
    let parallel = parallel_tools(job);
    // `ignore_eos`: EOS never ends the reply.
    // The serial paths stop inside `generate*` via the process EOS set, so
    // clear it for this one job and restore after (one job at a time here;
    // the continuous-batch path carries the flag per Request instead).
    let ignore_eos = job
        .body
        .get("ignore_eos")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let _eos_scope = ignore_eos.then(lisa_engine::core::generate::ignore_eos_scope);
    let mut acc: Vec<u32> = Vec::new();
    let mut decoded = IncrementalDecoder::new(tok, false);
    let mut last = String::new();
    let mut last_reasoning = String::new();
    let hit_stop = std::cell::Cell::new(false);
    let tool_mode = std::cell::Cell::new(TargetMtpSamplingMode::Configured);
    let mut tool_parser = streaming::DsmlParser::for_template(thinking, native_dsml);
    let mut emit = |t: u32| -> anyhow::Result<()> {
        if is_eos(t) {
            return Ok(());
        }
        acc.push(t);
        decoded.push(t)?;
        let text = decoded.text();
        if !stops.is_empty() {
            if stops
                .iter()
                .any(|s| !s.is_empty() && text.contains(s.as_str()))
            {
                hit_stop.set(true);
                if streaming {
                    // DS4 emits the safe prefix before ending the stream when
                    // a text stop is found. The normal holdback path runs
                    // after this check, so flush the trimmed prefix here.
                    let (reasoning, content) = streaming_reply_sections(&text, thinking, &stops);
                    let content_delta = stream_delta(&last, &content);
                    let reasoning_delta = stream_delta(&last_reasoning, &reasoning);
                    if content_delta.is_some() || reasoning_delta.is_some() {
                        let _ = reply.send(Reply::Delta {
                            text: content_delta.unwrap_or_default(),
                            reasoning: reasoning_delta,
                            logprobs: None,
                        });
                    }
                    last = content;
                    last_reasoning = reasoning;
                }
                return Err(anyhow::anyhow!("stop sequence"));
            }
        }
        if streaming {
            let stable = decoded.stable_text();
            let safe_len = streaming_stop_safe_len(stable, &stops);
            let (reasoning, answer) = streaming_sections(&stable[..safe_len], thinking);
            let shown = answer[..deepseek41::visible_end(answer)].to_string();
            let content_delta = stream_delta(&last, &shown);
            let reasoning_delta = stream_delta(&last_reasoning, reasoning);
            if content_delta.is_some() || reasoning_delta.is_some() {
                let _ = reply.send(Reply::Delta {
                    text: content_delta.unwrap_or_default(),
                    reasoning: reasoning_delta,
                    logprobs: None,
                });
            }
            last = shown;
            last_reasoning = reasoning.to_string();
        }
        if dsml_tools {
            let state = tool_parser.update(text);
            tool_mode.set(if state == DsmlDecodeState::Greedy {
                TargetMtpSamplingMode::Greedy
            } else {
                TargetMtpSamplingMode::Configured
            });
            if state == DsmlDecodeState::Done {
                hit_stop.set(true);
                return Err(anyhow::anyhow!("tool call complete"));
            }
        }
        Ok(())
    };
    // Snapshot the full prompt at the clean boundary (prompt fed, nothing
    // decoded) so later requests sharing it resume there.
    let explicit_cache_law = tower.cache_numerical_law();
    let mut after_prefill = |caches: &[lisa_engine::core::cache::LayerCache]| {
        let cache_law = explicit_cache_law.unwrap_or_else(lisa_mlx::runtime::kernel_law);
        prefix.insert_with_law(cache_law, &ids_full, caches);
    };
    let generated = if depth > 0 {
        let md = mtp_depth(depth, tower);
        if dsml_tools {
            let mut on_token_with_mode = |token| {
                emit(token)?;
                Ok(tool_mode.get())
            };
            sess.generate_mtp_depth_with_sampling_mode(
                tower,
                &suffix,
                max_tokens,
                md,
                Some(&mut after_prefill),
                if greedy { None } else { Some(&mut sampler) },
                &mut on_token_with_mode,
            )
        } else {
            sess.generate_mtp_depth(
                tower,
                &suffix,
                max_tokens,
                md,
                Some(&mut after_prefill),
                if greedy { None } else { Some(&mut sampler) },
                &mut emit,
            )
        }
    } else if dsml_tools {
        let mut on_token_with_mode = |token| {
            emit(token)?;
            Ok(tool_mode.get())
        };
        sess.generate_with_sampling_mode(
            tower,
            &suffix,
            max_tokens,
            &mut sampler,
            Some(&mut after_prefill),
            &mut on_token_with_mode,
        )
    } else {
        sess.generate(
            tower,
            &suffix,
            max_tokens,
            &mut sampler,
            cfg.pld,
            Some(&mut after_prefill),
            &mut emit,
        )
    };
    let generated = match generated {
        Ok(g) => g,
        Err(_) if hit_stop.get() => {
            // A callback stop can leave the last visible token unfed or a
            // verified suffix discarded. Rebuild from the next full prompt.
            sess.invalidate();
            Vec::new()
        }
        Err(e) => return Err(e),
    };
    let stopped = generated.last().map(|&t| is_eos(t)).unwrap_or(false);
    let full = tok.decode(&acc)?;
    let (content, reasoning, tool_calls) = shape_reply(&full, &tools, parallel, &stops, thinking);
    if streaming {
        // Completed-response shaping can reinterpret a late `</think>`
        // marker, but an SSE stream cannot retract its earlier text-mode
        // deltas. Reconcile with the incremental parser's final state.
        let (stream_reasoning, stream_content) = streaming_reply_sections(&full, thinking, &stops);
        let content_delta = stream_delta(&last, &stream_content);
        let reasoning_delta = stream_delta(&last_reasoning, &stream_reasoning);
        if content_delta.is_some() || reasoning_delta.is_some() {
            let _ = reply.send(Reply::Delta {
                text: content_delta.unwrap_or_default(),
                reasoning: reasoning_delta,
                logprobs: None,
            });
        }
    }

    // Track the conversation including this assistant turn.
    conversation = messages.clone();
    conversation.push(ChatMessage {
        role: "assistant".to_string(),
        content: content.clone(),
        reasoning_content: reasoning.clone(),
        tool_calls: tool_calls.clone(),
        tool_call_id: None,
        parts: None,
    });
    if sess.is_valid() {
        sessions.insert(sid, (sess, conversation));
    }
    let finish = if !tool_calls.is_empty() {
        "tool_calls"
    } else if stopped || hit_stop.get() {
        "stop"
    } else {
        "length"
    };
    let _ = job.reply.send(Reply::Done {
        content,
        reasoning,
        tool_calls,
        prompt_tokens: suffix.len(),
        completion_tokens: acc.len(),
        cached_tokens: 0, // the session path does not consult the prefix cache for reporting
        finish: finish.into(),
        logprobs: None,
        stop_sequence: None,
    });
    Ok(())
}

fn structural_prefix(a: &[ChatMessage], b: &[ChatMessage]) -> usize {
    let n = a.len().min(b.len());
    let mut i = 0;
    while i < n
        && a[i].role == b[i].role
        && a[i].content == b[i].content
        && a[i].reasoning_content == b[i].reasoning_content
    {
        i += 1;
    }
    i
}

fn common_prefix(a: &[u32], b: &[u32]) -> usize {
    let n = a.len().min(b.len());
    let mut i = 0;
    while i < n && a[i] == b[i] {
        i += 1;
    }
    i
}

fn models_payload(id: &str, supports_images: bool) -> Value {
    json!({
        "object": "list",
        "data": [{
            "id": id,
            "object": "model",
            "owned_by": "lisa",
            "architecture": {
                "input_modalities": if supports_images { json!(["text", "image"]) } else { json!(["text"]) },
                "output_modalities": ["text"]
            }
        }]
    })
}

fn handle_conn(
    api: &ApiState,
    mut stream: TcpStream,
    tx: Sender<Job>,
    cfg: &ServerConfig,
) -> anyhow::Result<()> {
    let req = match read_request(&mut stream) {
        Ok(Some(r)) => r,
        Ok(None) => return Ok(()),
        Err(e) => {
            let payload =
                json!({"error": {"message": e.to_string(), "type": "invalid_request_error"}});
            let _ = write_response(
                &mut stream,
                "400 Bad Request",
                "application/json",
                payload.to_string().as_bytes(),
            );
            return Ok(());
        }
    };
    let (method, path, body) = req;
    if method == "GET" && (path == "/health" || path == "/healthz") {
        let payload = json!({"status": "ok"});
        return write_response(
            &mut stream,
            "200 OK",
            "application/json",
            payload.to_string().as_bytes(),
        );
    }
    if method == "GET" && path.starts_with("/v1/models") {
        let payload = models_payload(&cfg.model_id, cfg.supports_images);
        return write_response(
            &mut stream,
            "200 OK",
            "application/json",
            payload.to_string().as_bytes(),
        );
    }
    if method == "POST" && (path == "/v1/chat/completions" || path == "/v1/completions") {
        let surface = if path == "/v1/completions" {
            Surface::Completions
        } else {
            Surface::Chat
        };
        return dispatch_or_400(api, &mut stream, &body, tx, cfg, surface);
    }
    if method == "POST" && path == "/v1/messages" {
        return dispatch_or_400(api, &mut stream, &body, tx, cfg, Surface::Messages);
    }
    if method == "POST" && path == "/v1/messages/count_tokens" {
        return dispatch_or_400(api, &mut stream, &body, tx, cfg, Surface::CountTokens);
    }
    if method == "POST" && path == "/v1/responses" {
        return dispatch_or_400(api, &mut stream, &body, tx, cfg, Surface::Responses);
    }
    if method == "POST" && path == "/v1/embeddings" {
        return dispatch_or_400(api, &mut stream, &body, tx, cfg, Surface::Embeddings);
    }
    if method == "GET" && path.starts_with("/v1/responses/") {
        let id = path.trim_start_matches("/v1/responses/").to_string();
        let stored = api.response(&id);
        let status = if stored.is_some() {
            "200 OK"
        } else {
            "404 Not Found"
        };
        let payload = stored.unwrap_or_else(
            || json!({"error": {"message": "response not found", "type": "invalid_request_error"}}),
        );
        return write_response(
            &mut stream,
            status,
            "application/json",
            payload.to_string().as_bytes(),
        );
    }

    write_response(
        &mut stream,
        "404 Not Found",
        "application/json",
        br#"{"error":{"message":"not found"}}"#,
    )
}

/// Normalize a legacy `/v1/completions` body into the internal chat shape so the
/// engine path is shared.
fn normalize_completion_prompt(body: &mut Value) {
    if body.get("messages").is_some() {
        return;
    }
    let text = match body.get("prompt") {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Array(a)) => a
            .iter()
            .map(|v| v.as_str().unwrap_or(""))
            .collect::<Vec<_>>()
            .join(""),
        _ => String::new(),
    };
    body["messages"] = json!([{ "role": "user", "content": text }]);
}

/// Run a completion handler, turning client-shape errors into a 400.
fn dispatch_or_400(
    api: &ApiState,
    stream: &mut TcpStream,
    body: &[u8],
    tx: Sender<Job>,
    cfg: &ServerConfig,
    surface: Surface,
) -> anyhow::Result<()> {
    if let Err(e) = dispatch_completion(api, stream, body, tx, cfg, surface) {
        let payload = if surface == Surface::Messages {
            json!({"type": "error", "error": {"type": "invalid_request_error", "message": e.to_string()}})
        } else {
            json!({"error": {"message": e.to_string(), "type": "invalid_request_error"}})
        };
        let _ = write_response(
            stream,
            "400 Bad Request",
            "application/json",
            payload.to_string().as_bytes(),
        );
    }
    Ok(())
}

fn dispatch_completion(
    api: &ApiState,
    stream: &mut TcpStream,
    body: &[u8],
    tx: Sender<Job>,
    cfg: &ServerConfig,
    surface: Surface,
) -> anyhow::Result<()> {
    let mut body: Value =
        serde_json::from_slice(body).map_err(|e| anyhow::anyhow!("invalid JSON body: {e}"))?;
    anyhow::ensure!(body.is_object(), "request body must be a JSON object");
    match surface {
        Surface::Completions => normalize_completion_prompt(&mut body),
        Surface::Messages | Surface::CountTokens => normalize_messages_body(&mut body),
        Surface::Responses => normalize_responses_body(&mut body),
        Surface::Chat => {}
        Surface::Embeddings => {}
    }
    if surface == Surface::Responses {
        api.extend_conversation(&mut body)?;
    }
    if surface == Surface::Messages && body.get("max_tokens").is_none() {
        // Anthropic's spec makes `max_tokens` required.
        anyhow::bail!("max_tokens is required");
    }
    if surface != Surface::CountTokens && surface != Surface::Embeddings {
        // Validate the request shape up front so client errors are 400s, not 500s.
        parse_messages(&body)?;
    }
    anyhow::ensure!(
        response_format_of(&body).is_none()
            || body.get("session_id").and_then(Value::as_str).is_none(),
        "structured output does not support session_id; send complete message history"
    );
    if lisa_engine::models::vision::messages_have_images(&body) {
        anyhow::ensure!(
            cfg.supports_images,
            "the loaded checkpoint does not support image input"
        );
        anyhow::ensure!(
            body.get("session_id").is_none(),
            "image requests must include full message history and omit session_id"
        );
        anyhow::ensure!(
            surface != Surface::CountTokens,
            "image token counting requires the vision encoder; use completion usage"
        );
        lisa_engine::models::vision::decode_message_images(&body)?;
    }
    let stream_mode = body
        .get("stream")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let n = match body.get("n") {
        None => 1,
        Some(value) => value
            .as_u64()
            .filter(|n| (1..=16).contains(n))
            .ok_or_else(|| anyhow::anyhow!("n must be an integer between 1 and 16"))?
            as usize,
    };
    anyhow::ensure!(
        n == 1 || (!stream_mode && matches!(surface, Surface::Chat | Surface::Completions)),
        "multiple choices require non-streaming chat or completions"
    );
    // Background Responses return a queued resource and retain the final result.
    if surface == Surface::Responses
        && body
            .get("background")
            .and_then(|v| v.as_bool())
            .unwrap_or(false)
    {
        let id = response_id();
        let mut payload = empty_response_resource(api, &id, "queued");
        payload["background"] = json!(true);
        let (rtx, rrx) = mpsc::channel::<Reply>();
        tx.send(Job {
            body: body.clone(),
            stream: false,
            session_id: None,
            surface,
            reply: rtx,
        })
        .map_err(|_| anyhow::anyhow!("engine worker is gone"))?;
        api.store(&id, payload.clone(), None);
        let key = id.clone();
        let background_api = api.clone();
        std::thread::spawn(move || match drain_reply(rrx) {
            Ok(d) => {
                let mut payload = responses_payload_with_id(&background_api, &d, &key);
                payload["background"] = json!(true);
                store_conversation(&background_api, &key, &body, &d, &payload);
            }
            Err(error) => {
                let mut payload = empty_response_resource(&background_api, &key, "failed");
                payload["background"] = json!(true);
                payload["error"] =
                    json!({"code": "engine_error", "message": error_message_for(&error)});
                background_api.store(&key, payload, None);
            }
        });
        return write_response(
            stream,
            "200 OK",
            "application/json",
            payload.to_string().as_bytes(),
        );
    }
    let include_usage = body
        .get("stream_options")
        .and_then(|v| v.get("include_usage"))
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let session_id = body
        .get("session_id")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());
    if n > 1 {
        return collect_choices(api, stream, &body, tx, n, surface);
    }

    let (rtx, rrx) = mpsc::channel::<Reply>();
    tx.send(Job {
        body: body.clone(),
        stream: stream_mode,
        session_id,
        surface,
        reply: rtx,
    })
    .map_err(|_| anyhow::anyhow!("engine worker is gone"))?;

    if surface == Surface::CountTokens {
        return collect_count_tokens(stream, rrx);
    }
    if stream_mode {
        stream_surface(api, stream, rrx, include_usage, surface, &body)
    } else {
        collect_surface(api, stream, rrx, surface, &body)
    }
}

fn anthropic_system_text(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Array(blocks) => blocks
            .iter()
            .filter_map(|b| b.get("text").and_then(|t| t.as_str()))
            .collect::<Vec<_>>()
            .join(""),
        _ => String::new(),
    }
}

/// A tool_result block's content (string or array of text blocks).
fn block_content_text(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Array(blocks) => blocks
            .iter()
            .filter_map(|b| {
                b.get("text")
                    .and_then(|t| t.as_str())
                    .or_else(|| b.as_str())
            })
            .collect::<Vec<_>>()
            .join(""),
        _ => String::new(),
    }
}

/// Convert an Anthropic `/v1/messages` body into the internal chat shape.
fn normalize_messages_body(body: &mut Value) {
    let mut out: Vec<Value> = Vec::new();
    if let Some(sys) = body.get("system") {
        let text = anthropic_system_text(sys);
        if !text.is_empty() {
            out.push(json!({ "role": "system", "content": text }));
        }
    }
    if let Some(arr) = body.get("messages").and_then(|v| v.as_array()) {
        for m in arr {
            let role = m.get("role").and_then(|v| v.as_str()).unwrap_or("user");
            match m.get("content") {
                Some(Value::String(s)) => {
                    out.push(json!({ "role": role, "content": s }));
                }
                Some(Value::Array(blocks)) => {
                    let mut text = String::new();
                    let mut reasoning = String::new();
                    let mut tool_calls: Vec<Value> = Vec::new();
                    let mut parts: Vec<Value> = Vec::new();
                    let mut has_image = false;
                    let mut only_tool_results = !blocks.is_empty();
                    for b in blocks {
                        let ty = b.get("type").and_then(|t| t.as_str()).unwrap_or("");
                        match ty {
                            "text" => {
                                only_tool_results = false;
                                text.push_str(b.get("text").and_then(|t| t.as_str()).unwrap_or(""));
                                parts.push(b.clone());
                            }
                            "thinking" => {
                                only_tool_results = false;
                                reasoning.push_str(
                                    b.get("thinking").and_then(|t| t.as_str()).unwrap_or(""),
                                );
                            }
                            "tool_use" => {
                                only_tool_results = false;
                                tool_calls.push(json!({
                                    "id": b.get("id").cloned().unwrap_or(json!("")),
                                    "type": "function",
                                    "function": {
                                        "name": b.get("name").cloned().unwrap_or(json!("")),
                                        "arguments": b.get("input").cloned().unwrap_or(json!({})),
                                    }
                                }));
                            }
                            "tool_result" => {
                                let id =
                                    b.get("tool_use_id").and_then(|v| v.as_str()).unwrap_or("");
                                let content =
                                    block_content_text(b.get("content").unwrap_or(&Value::Null));
                                out.push(json!({
                                    "role": "tool",
                                    "tool_call_id": id,
                                    "content": content,
                                }));
                            }
                            "image" => {
                                only_tool_results = false;
                                has_image = true;
                                parts.push(b.clone());
                            }
                            _ => {}
                        }
                    }
                    if !only_tool_results {
                        let mut msg = json!({ "role": role, "content": text });
                        if has_image {
                            msg["content"] = Value::Array(parts);
                        }
                        if !reasoning.is_empty() {
                            msg["reasoning_content"] = json!(reasoning);
                        }
                        if !tool_calls.is_empty() {
                            msg["tool_calls"] = Value::Array(tool_calls);
                        }
                        out.push(msg);
                    }
                }
                _ => {}
            }
        }
    }
    body["messages"] = Value::Array(out);
    if let Some(tools) = body.get("tools").and_then(|v| v.as_array()) {
        let openai: Vec<Value> = tools
            .iter()
            .map(|t| {
                json!({
                    "type": "function",
                    "function": {
                        "name": t.get("name").and_then(|v| v.as_str()).unwrap_or(""),
                        "description": t.get("description").cloned().unwrap_or(Value::Null),
                        "parameters": t.get("input_schema").cloned().unwrap_or(json!({"type": "object"})),
                    }
                })
            })
            .collect();
        body["tools"] = Value::Array(openai);
    }
    let mut parallel_off = false;
    if let Some(tc) = body.get("tool_choice").cloned() {
        if let Some(ty) = tc.get("type").and_then(|v| v.as_str()) {
            let mapped = match ty {
                "auto" => json!("auto"),
                "any" => json!("required"),
                "none" => json!("none"),
                "tool" => json!({
                    "type": "function",
                    "function": { "name": tc.get("name").and_then(|v| v.as_str()).unwrap_or("") }
                }),
                _ => json!("auto"),
            };
            body["tool_choice"] = mapped;
        }
        if tc
            .get("disable_parallel_tool_use")
            .and_then(|v| v.as_bool())
            .unwrap_or(false)
        {
            parallel_off = true;
        }
    }
    if parallel_off {
        body["parallel_tool_calls"] = json!(false);
    }
    if let Some(s) = body.get("stop_sequences") {
        body["stop"] = s.clone();
    }
    if body
        .get("thinking")
        .and_then(|t| t.get("type"))
        .and_then(|v| v.as_str())
        == Some("disabled")
    {
        body["chat_template_kwargs"] = json!({ "enable_thinking": false });
    }
}

// ─────────────────────── OpenAI Responses ───────────────────────

/// Convert an OpenAI `/v1/responses` body into the internal chat shape.
fn normalize_responses_body(body: &mut Value) {
    let mut out: Vec<Value> = Vec::new();
    if let Some(ins) = body.get("instructions").and_then(|v| v.as_str()) {
        out.push(json!({ "role": "system", "content": ins }));
    }
    match body.get("input").cloned() {
        Some(Value::String(s)) => out.push(json!({ "role": "user", "content": s })),
        Some(Value::Array(items)) => {
            for item in &items {
                match item.get("type").and_then(|t| t.as_str()) {
                    Some("function_call") => {
                        out.push(json!({
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [{
                                "id": item.get("call_id").cloned().unwrap_or(json!("")),
                                "type": "function",
                                "function": {
                                    "name": item.get("name").cloned().unwrap_or(json!("")),
                                    "arguments": item.get("arguments").cloned().unwrap_or(json!("")),
                                }
                            }]
                        }));
                    }
                    Some("function_call_output") => {
                        out.push(json!({
                            "role": "tool",
                            "tool_call_id": item.get("call_id").cloned().unwrap_or(json!("")),
                            "content": item.get("output").cloned().unwrap_or(json!("")),
                        }));
                    }
                    _ => {
                        let role = item.get("role").and_then(|v| v.as_str()).unwrap_or("user");
                        let mut text = String::new();
                        let mut image = false;
                        let mut content_parts = Vec::new();
                        if let Some(parts) = item.get("content").and_then(|v| v.as_array()) {
                            for p in parts {
                                match p.get("type").and_then(|t| t.as_str()) {
                                    Some("input_text") | Some("output_text") => {
                                        let part_text =
                                            p.get("text").and_then(|t| t.as_str()).unwrap_or("");
                                        text.push_str(part_text);
                                        content_parts
                                            .push(json!({"type": "text", "text": part_text}));
                                    }
                                    Some("input_image") | Some("image") => {
                                        image = true;
                                        let mut part = p.clone();
                                        part["type"] = json!("image");
                                        content_parts.push(part);
                                    }
                                    _ => {}
                                }
                            }
                        }
                        let mut msg = json!({ "role": role, "content": text });
                        if image {
                            msg["content"] = Value::Array(content_parts);
                        }
                        out.push(msg);
                    }
                }
            }
        }
        _ => {}
    }
    body["messages"] = Value::Array(out);
    if let Some(tools) = body.get("tools").and_then(|v| v.as_array()) {
        let openai: Vec<Value> = tools
            .iter()
            .filter(|t| t.get("type").and_then(|x| x.as_str()).unwrap_or("function") == "function")
            .map(|t| {
                json!({
                    "type": "function",
                    "function": {
                        "name": t.get("name").and_then(|v| v.as_str()).unwrap_or(""),
                        "description": t.get("description").cloned().unwrap_or(Value::Null),
                        "parameters": t.get("parameters").cloned().unwrap_or(json!({"type": "object"})),
                    }
                })
            })
            .collect();
        body["tools"] = Value::Array(openai);
    }
    if let Some(tc) = body.get("tool_choice").cloned() {
        if let Some(ty) = tc.get("type").and_then(|v| v.as_str()) {
            if ty == "function" {
                body["tool_choice"] = json!({
                    "type": "function",
                    "function": { "name": tc.get("name").and_then(|v| v.as_str()).unwrap_or("") }
                });
            } else {
                body["tool_choice"] = json!(ty);
            }
        }
    }
    // `text.format` -> response_format
    if let Some(fmt) = body.get("text").and_then(|t| t.get("format")) {
        let ty = fmt.get("type").and_then(|v| v.as_str()).unwrap_or("");
        let rf = match ty {
            "json_object" => Some(json!({ "type": "json_object" })),
            "json_schema" => Some(json!({
                "type": "json_schema",
                "json_schema": {
                    "name": fmt.get("name").cloned().unwrap_or(json!("schema")),
                    "schema": fmt.get("schema").cloned().unwrap_or(json!({"type": "object"})),
                }
            })),
            _ => None,
        };
        if let Some(rf) = rf {
            body["response_format"] = rf;
        }
    }
    if let Some(m) = body.get("max_output_tokens") {
        body["max_tokens"] = m.clone();
    }
    if body
        .get("reasoning")
        .and_then(|r| r.get("effort"))
        .and_then(|v| v.as_str())
        == Some("none")
    {
        body["chat_template_kwargs"] = json!({ "enable_thinking": false });
    }
}

/// A response id for the Responses API and the chaining store.
fn response_id() -> String {
    let n = SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    format!("resp_{:x}{:x}", now(), n)
}

// ─────────────────────── token counting ───────────────────────

/// Worker-side `messages/count_tokens`: render the prompt and count its tokens.
fn count_tokens_job(
    tok: &Tokenizer,
    job: &Job,
    tmpl: Option<&ChatTemplate>,
    cfg: &ServerConfig,
) -> anyhow::Result<()> {
    let messages = parse_messages(&job.body)?;
    let prompt = render_prompt(tmpl, &messages, true, &[], request_thinking(&job.body, cfg))?;
    let ids = tok.encode(&prompt, false)?;
    let _ = job.reply.send(Reply::Done {
        content: String::new(),
        reasoning: None,
        tool_calls: Vec::new(),
        prompt_tokens: ids.len(),
        completion_tokens: 0,
        cached_tokens: 0,
        finish: "stop".to_string(),
        logprobs: None,
        stop_sequence: None,
    });
    Ok(())
}

fn collect_count_tokens(stream: &mut TcpStream, rx: Receiver<Reply>) -> anyhow::Result<()> {
    let d = match drain_reply(rx) {
        Ok(d) => d,
        Err(e) => {
            let payload =
                json!({"error": {"message": error_message_for(&e), "type": "engine_error"}});
            return write_response(
                stream,
                http_status_for(&e),
                "application/json",
                payload.to_string().as_bytes(),
            );
        }
    };
    let payload = json!({ "input_tokens": d.prompt_tokens });
    write_response(
        stream,
        "200 OK",
        "application/json",
        payload.to_string().as_bytes(),
    )
}

/// The prompt tokens still to feed: everything after the resumed prefix-cache
/// boundary and after the two-stage prefill boundary.
fn unfed_prefix(ids: &[u32], resumed: Option<&usize>, prefeed: &Option<Vec<u32>>) -> Vec<u32> {
    let skip = resumed
        .copied()
        .unwrap_or(0)
        .max(prefeed.as_ref().map_or(0, |p| p.len()));
    ids[skip..].to_vec()
}

fn complete(
    tower: &mut dyn LanguageModel,
    tok: &Tokenizer,
    job: &Job,
    cfg: &ServerConfig,
    prefix: &mut PrefixCache,
    tmpl: Option<&ChatTemplate>,
) -> anyhow::Result<()> {
    tower.clear_images();
    let result = complete_inner(tower, tok, job, cfg, prefix, tmpl);
    tower.clear_images();
    result
}

fn complete_inner(
    tower: &mut dyn LanguageModel,
    tok: &Tokenizer,
    job: &Job,
    cfg: &ServerConfig,
    prefix: &mut PrefixCache,
    tmpl: Option<&ChatTemplate>,
) -> anyhow::Result<()> {
    let mut messages = parse_messages(&job.body)?;
    let json_schema = response_format_of(&job.body);
    if let Some(schema) = &json_schema {
        let instr = json_instruction(Some(schema));
        if let Some(sys) = messages.iter_mut().find(|m| m.role == "system") {
            sys.content = format!("{}\n\n{instr}", sys.content);
        } else {
            messages.insert(
                0,
                ChatMessage {
                    role: "system".to_string(),
                    content: instr,
                    reasoning_content: None,
                    tool_calls: Vec::new(),
                    tool_call_id: None,
                    parts: None,
                },
            );
        }
    }
    let tools = request_tools(job);
    let stops = request_stops(job);
    let parallel = parallel_tools(job);
    // `ignore_eos` for the serial completion path (see the session path note).
    let ignore_eos = job
        .body
        .get("ignore_eos")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let _eos_scope = ignore_eos.then(lisa_engine::core::generate::ignore_eos_scope);
    // Raw completions never get a think-prompt (parity with the old
    // surface-gated resolution); explicit enable_thinking there is ignored.
    let thinking = job.surface != Surface::Completions && request_thinking(&job.body, cfg);
    // Anthropic semantics: a trailing assistant message is a prefill. Render the
    // generation header, then the partial assistant text, and let the model
    // continue it (rather than closing the turn and starting a new one).
    let prompt = if job.surface == Surface::Messages
        && messages
            .last()
            .map(|m| m.role == "assistant")
            .unwrap_or(false)
    {
        let prefill = messages.pop().map(|m| m.content).unwrap_or_default();
        format!(
            "{}{}",
            render_prompt(tmpl, &messages, true, &tools, false)?,
            prefill
        )
    } else {
        render_prompt(tmpl, &messages, true, &tools, thinking)?
    };
    let mut ids = tok.encode(&prompt, false)?;
    let images = lisa_engine::models::vision::decode_message_images(&job.body)?;
    let has_images = !images.is_empty();
    if has_images {
        anyhow::ensure!(
            tmpl.is_some(),
            "image input requires the checkpoint chat template"
        );
        tower.prepare_images(&mut ids, &images)?;
    }
    drop(images);

    let max_tokens = request_max_tokens(job, cfg);
    let depth = match resolve_depth(job, cfg) {
        Ok(d) => d,
        Err(e) => {
            let _ = job.reply.send(Reply::Error(e.to_string()));
            return Ok(());
        }
    };
    let depth = if has_images { 0 } else { depth };
    let mut sampler = sampler_from(&job.body, cfg);
    let greedy = !tower.has_target_mtp() && sampler.greedy() && !sampler.mtp_exact_sampling;
    let native_dsml = tmpl.is_some_and(|t| t.deepseek41);
    let dsml_tools = !tools.is_empty() && (tower.has_target_mtp() || native_dsml);
    let streaming = job.stream;
    if streaming {
        let _ = job.reply.send(Reply::Start {
            prompt_tokens: ids.len(),
        });
    }

    let want_logprobs = job
        .body
        .get("logprobs")
        .and_then(|v| v.as_bool())
        .unwrap_or(false)
        || job
            .body
            .get("top_logprobs")
            .and_then(|v| v.as_u64())
            .unwrap_or(0)
            > 0;
    let mut collected: Vec<u32> = Vec::new();
    let mut decoded = IncrementalDecoder::new(tok, false);
    let mut last_text = String::new();
    let mut last_reasoning = String::new();
    let reply = job.reply.clone();
    let hit_stop = std::cell::Cell::new(false);
    let tool_mode = std::cell::Cell::new(TargetMtpSamplingMode::Configured);
    let mut tool_parser = streaming::DsmlParser::for_template(thinking, native_dsml);
    let mut lp_entries: Vec<Value> = Vec::new();

    // Cross-request prefix cache: resume at a cached prompt boundary when the
    // prompt extends it token-exactly; otherwise prefill in stages so a
    // mid-prompt boundary (the shared system-prefix cell) gets snapshotted.
    let mut resumed: Option<usize> = None;
    let mut session = if !has_images && prefix.enabled() {
        match prefix.restore_session(tower, &ids) {
            Some((l, s)) => {
                eprintln!("[serve] prefix-cache resume {l}/{}", ids.len());
                resumed = Some(l);
                s
            }
            None => Session::new(tower),
        }
    } else {
        Session::new(tower)
    };
    session.ensure_valid()?;
    session.set_target_mtp_eos_policy(TargetMtpEosPolicy::StopBeforeVerified);
    let mut prefeed: Option<Vec<u32>> = None;
    // Populate the cache BEFORE generation, with the boundary one token short
    // of the prompt end. `PrefixCache::lookup` serves only STRICT extensions
    // (entry len < prompt len — a resumed state must be fed a fresh token to
    // yield logits), so an entry at the full boundary can never serve an
    // identical repeat. llmprobe's prefix probe (1580-token prompt, warm ==
    // cold prompt) measured 0/1580 cached and warm TTFT SLOWER than cold
    // (8.5 s vs 6.5 s): every warm request re-prefilled the whole prompt AND
    // paid the snapshot clone. One token short, a warm repeat IS a strict
    // extension: it resumes at the boundary and prefills exactly one token.
    if !has_images
        && prefix.enabled()
        && ids.len() > lisa_engine::core::prefix_cache::MIN_PREFIX_TOKENS
    {
        let split = ids.len() - 1;
        let pos = resumed.unwrap_or(0);
        if pos < split {
            // Mid-prompt boundary only past the natural prefill-chunk size,
            // and ON a chunk multiple: feeding [..b] then the rest dispatches
            // exactly the windows a plain chunked prefill would, so the cache
            // never changes the computed values.
            if ids.len() > 2048 {
                let b = (ids.len() / 2048) * 2048;
                if b > pos && b <= split {
                    session.feed(tower, &ids[pos..b])?;
                    session.ensure_valid()?;
                    prefix.insert_with_law(
                        lisa_engine::core::prefix_cache::model_numerical_law(tower),
                        &ids[..b],
                        &session.caches,
                    );
                    prefeed = Some(ids[..b].to_vec());
                }
            }
            let start = prefeed.as_ref().map_or(pos, |p| p.len());
            session.feed(tower, &ids[start..split])?;
            session.ensure_valid()?;
            prefix.insert_with_law(
                lisa_engine::core::prefix_cache::model_numerical_law(tower),
                &ids[..split],
                &session.caches,
            );
            prefeed = Some(ids[..split].to_vec());
        }
    }
    let generated = {
        let mut emit = |t: u32, lp: Option<Value>| -> anyhow::Result<()> {
            if let Some(entry) = &lp {
                lp_entries.push(entry.clone());
            }
            let eos = is_eos(t);
            if !eos {
                collected.push(t);
                let text = decoded.push(t)?;
                if !stops.is_empty() {
                    if stops
                        .iter()
                        .any(|s| !s.is_empty() && text.contains(s.as_str()))
                    {
                        hit_stop.set(true);
                        if streaming && json_schema.is_none() {
                            let visible = strip_stops(&text, &stops);
                            let (reasoning, answer) = streaming_sections(visible, thinking);
                            let shown = match tool_call_start(answer) {
                                Some(i) => answer[..i].to_string(),
                                None => answer.to_string(),
                            };
                            let delta = stream_delta(&last_text, &shown);
                            let reasoning_delta = stream_delta(&last_reasoning, reasoning);
                            last_text = shown;
                            last_reasoning = reasoning.to_string();
                            if delta.is_some() || reasoning_delta.is_some() || lp.is_some() {
                                let _ = reply.send(Reply::Delta {
                                    text: delta.unwrap_or_default(),
                                    reasoning: reasoning_delta,
                                    logprobs: lp.map(|e| json!({ "content": [e] })),
                                });
                            }
                        }
                        return Err(anyhow::anyhow!("stop sequence"));
                    }
                }
            }
            if streaming && json_schema.is_none() {
                let text = decoded.stable_text();
                let safe_len = streaming_stop_safe_len(&text, &stops);
                let (reasoning, answer) = streaming_sections(&text[..safe_len], thinking);
                // Past a `<tool_call>` the text is not assistant content; don't stream it.
                let shown = answer[..deepseek41::visible_end(answer)].to_string();
                let delta = stream_delta(&last_text, &shown);
                let reasoning_delta = stream_delta(&last_reasoning, reasoning);
                last_text = shown;
                last_reasoning = reasoning.to_string();
                // An EOS token contributes a logprob entry but no text; stream it
                // so the streamed logprob list matches the non-streamed one.
                if delta.is_some() || reasoning_delta.is_some() || lp.is_some() {
                    let _ = reply.send(Reply::Delta {
                        text: delta.unwrap_or_default(),
                        reasoning: reasoning_delta,
                        logprobs: lp.map(|e| json!({ "content": [e] })),
                    });
                }
            }
            if dsml_tools {
                let state = tool_parser.update(decoded.text());
                tool_mode.set(if state == DsmlDecodeState::Greedy {
                    TargetMtpSamplingMode::Greedy
                } else {
                    TargetMtpSamplingMode::Configured
                });
                if state == DsmlDecodeState::Done {
                    hit_stop.set(true);
                    return Err(anyhow::anyhow!("tool call complete"));
                }
            }
            Ok(())
        };
        if want_logprobs {
            // Logprobs require the logits at every step; MTP drafts would skip
            // them, so this path runs one token at a time.
            let mut out: Vec<u32> = Vec::new();
            let gen_ids = unfed_prefix(&ids, resumed.as_ref(), &prefeed);
            let mut logits = session.feed(tower, &gen_ids)?;
            let mut sampling_mode = TargetMtpSamplingMode::Configured;
            while out.len() < max_tokens {
                let t = if dsml_tools && sampling_mode == TargetMtpSamplingMode::Greedy {
                    Sampler::default().draw_ds4(&logits, &[])?
                } else if tower.uses_ds4_sampling() {
                    sampler.draw_ds4(&logits, &session.fed)?
                } else {
                    sampler.draw(&logits, &session.fed)?
                };
                let entry = json!({
                    "token": tok.decode(&[t]).unwrap_or_default(),
                    "logprob": token_logprob(&logits, t)?,
                    "bytes": Value::Null,
                    "top_logprobs": [],
                });
                match emit(t, Some(entry)) {
                    Ok(()) => {}
                    Err(_) if hit_stop.get() => break,
                    Err(e) => return Err(e),
                }
                out.push(t);
                if dsml_tools {
                    sampling_mode = tool_mode.get();
                }
                if out.len() >= max_tokens || is_eos(t) {
                    break;
                }
                logits = session.decode_token(tower, t)?;
            }
            Ok(out)
        } else if depth > 0 {
            let gen_ids = unfed_prefix(&ids, resumed.as_ref(), &prefeed);
            anyhow::ensure!(!gen_ids.is_empty(), "request has no new tokens");
            let md = mtp_depth(depth, tower);
            if dsml_tools {
                session.generate_mtp_depth_with_sampling_mode(
                    tower,
                    &gen_ids,
                    max_tokens,
                    md,
                    None,
                    Some(&mut sampler),
                    &mut |token| {
                        emit(token, None)?;
                        Ok(tool_mode.get())
                    },
                )
            } else {
                session.generate_mtp_depth(
                    tower,
                    &gen_ids,
                    max_tokens,
                    md,
                    None,
                    if greedy { None } else { Some(&mut sampler) },
                    &mut |t| emit(t, None),
                )
            }
        } else {
            let gen_ids = unfed_prefix(&ids, resumed.as_ref(), &prefeed);
            anyhow::ensure!(!gen_ids.is_empty(), "request has no new tokens");
            if dsml_tools {
                session.generate_with_sampling_mode(
                    tower,
                    &gen_ids,
                    max_tokens,
                    &mut sampler,
                    None,
                    &mut |token| {
                        emit(token, None)?;
                        Ok(tool_mode.get())
                    },
                )
            } else {
                session.generate(
                    tower,
                    &gen_ids,
                    max_tokens,
                    &mut sampler,
                    if has_images { 0 } else { cfg.pld },
                    None,
                    &mut |t| emit(t, None),
                )
            }
        }
    };
    let generated = match generated {
        Ok(g) => g,
        Err(e) if hit_stop.get() => {
            let _ = e;
            Vec::new()
        }
        Err(e) => return Err(e),
    };
    let logprobs = if want_logprobs {
        Some(json!({ "content": lp_entries }))
    } else {
        None
    };

    let full = tok.decode(&collected)?;
    let (content, reasoning, tool_calls) = if json_schema.is_some() {
        let sc = json_schema.clone().unwrap_or(json!({ "type": "object" }));
        let raw = strip_stops(&full, &stops);
        let body = strip_think_prefix(raw);
        let (mut content, mut errors) = match extract_json(body) {
            Some(s) => {
                let mut v: Value = serde_json::from_str(&s).unwrap_or(Value::String(s));
                repair_json(&mut v, &sc);
                let errs = json_schema_errors(&v, &sc);
                (v.to_string(), errs)
            }
            None => (
                body.trim().to_string(),
                vec!["the reply was not parseable as JSON".to_string()],
            ),
        };
        // Validate-and-retry: the engine has no grammar-constrained decoding,
        // so a reply that still violates the schema after repair gets ONE
        // corrective turn — the failed output plus the violations are fed back
        // and the model regenerates. The second attempt goes through the same
        // extract/repair/validate pipeline. Exhausting validation returns an
        // engine error instead of presenting invalid content as success.
        if !errors.is_empty() && !content.trim().is_empty() {
            eprintln!("[serve] json-schema retry: {}", errors.join("; "));
            let corrective = format!(
                "\n{}\nThat reply violates the requested JSON Schema ({}). \
                 Output only the corrected JSON value, nothing else.",
                content.trim(),
                errors.join("; ")
            );
            let retry_ids = tok.encode(&corrective, false)?;
            let mut noop = |_t: u32| -> anyhow::Result<()> { Ok(()) };
            let retry = session.generate(
                tower,
                &retry_ids,
                max_tokens,
                &mut sampler,
                0,
                None,
                &mut noop,
            );
            if let Ok(retry) = retry {
                let retry_text = tok.decode(&retry)?;
                let rbody = strip_think_prefix(strip_stops(&retry_text, &stops));
                if let Some(s) = extract_json(rbody) {
                    let mut v: Value = serde_json::from_str(&s).unwrap_or(Value::String(s));
                    repair_json(&mut v, &sc);
                    let rerrs = json_schema_errors(&v, &sc);
                    if rerrs.len() <= errors.len() {
                        content = v.to_string();
                        errors = rerrs;
                    }
                }
            }
        }
        anyhow::ensure!(
            errors.is_empty(),
            "generated output does not satisfy the requested JSON schema after repair/retry: {}",
            errors.join("; ")
        );
        (content, None, Vec::new())
    } else {
        shape_reply(&full, &tools, parallel, &stops, thinking)
    };
    if streaming && json_schema.is_none() {
        let (stream_reasoning, stream_content) = streaming_reply_sections(&full, thinking, &stops);
        let content_delta = stream_delta(&last_text, &stream_content);
        let reasoning_delta = stream_delta(&last_reasoning, &stream_reasoning);
        if content_delta.is_some() || reasoning_delta.is_some() {
            let _ = job.reply.send(Reply::Delta {
                text: content_delta.unwrap_or_default(),
                reasoning: reasoning_delta,
                logprobs: None,
            });
        }
    }
    if streaming && json_schema.is_some() {
        let _ = job.reply.send(Reply::Delta {
            text: content.clone(),
            reasoning: None,
            logprobs: None,
        });
    }
    let stopped_by = if hit_stop.get() {
        stops
            .iter()
            .find(|sq| !sq.is_empty() && full.contains(sq.as_str()))
            .cloned()
    } else {
        None
    };
    let finish = if !tool_calls.is_empty() {
        "tool_calls"
    } else if hit_stop.get() || generated.last().is_some_and(|&t| is_eos(t)) {
        "stop"
    } else {
        "length"
    };
    let _ = job.reply.send(Reply::Done {
        content,
        reasoning,
        tool_calls,
        prompt_tokens: ids.len(),
        completion_tokens: collected.len(),
        cached_tokens: resumed.unwrap_or(0),
        finish: finish.to_string(),
        logprobs,
        stop_sequence: stopped_by,
    });
    Ok(())
}

/// Serve POST /v1/embeddings: encode each input, run one forward over the
/// trunk, mean-pool the last hidden states over the sequence, L2-normalize.
/// The OpenAI payload is serialized into `Reply::Done.content` (the surface
/// writer hands it straight through).
fn run_embeddings(
    model_id: &str,
    tower: &mut dyn LanguageModel,
    tok: &Tokenizer,
    job: &Job,
) -> anyhow::Result<()> {
    use lisa_mlx::Array as MlxArray;
    let inputs: Vec<String> = match job.body.get("input") {
        Some(Value::String(s)) => vec![s.clone()],
        Some(Value::Array(a)) => {
            let mut v = Vec::with_capacity(a.len());
            for x in a {
                match x {
                    Value::String(s) => v.push(s.clone()),
                    other => v.push(other.to_string()),
                }
            }
            v
        }
        _ => anyhow::bail!("input must be a string or an array of strings"),
    };
    anyhow::ensure!(!inputs.is_empty(), "input must not be empty");

    let mut data = Vec::with_capacity(inputs.len());
    let mut total_tokens = 0usize;
    for text in &inputs {
        let ids = tok.encode(text, false)?;
        anyhow::ensure!(!ids.is_empty(), "input encoded to zero tokens: {text:?}");
        total_tokens += ids.len();
        let arr = MlxArray::from_slice(
            &ids.iter().map(|&t| t as i32).collect::<Vec<_>>(),
            &[1i32, ids.len() as i32],
        );
        let mut caches = tower.new_caches();
        let (mixed, _multi) = tower.forward(&arr, Some(&mut caches))?;
        // mixed: [1, S, H] -> mean over S -> [1, H]
        let pooled = mixed
            .mean_axis(1, false)?
            .as_dtype(lisa_mlx::Dtype::Float32)?;
        let _ = pooled.eval();
        let host = pooled.as_slice::<f32>();
        // L2 normalization in host order (H is small).
        let norm: f32 = host.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
        data.push(host.iter().map(|x| x / norm).collect::<Vec<f32>>());
    }

    let embeddings: Vec<Value> = data
        .into_iter()
        .enumerate()
        .map(|(i, e)| json!({ "object": "embedding", "index": i, "embedding": e }))
        .collect();
    let payload = json!({
        "object": "list",
        "data": embeddings,
        "model": model_id,
        "usage": { "prompt_tokens": total_tokens, "total_tokens": total_tokens },
    });
    let _ = job.reply.send(Reply::Done {
        content: payload.to_string(),
        reasoning: None,
        tool_calls: Vec::new(),
        prompt_tokens: total_tokens,
        completion_tokens: 0,
        cached_tokens: 0,
        finish: "stop".into(),
        logprobs: None,
        stop_sequence: None,
    });
    Ok(())
}

/// One parsed chat message. `reasoning_content` carries the assistant's
/// thinking (separate from `content`) so the checkpoint template can render the
/// canonical ` thinking… response` block. `tool_calls`/`tool_call_id` and raw
/// multimodal `parts` round-trip the OpenAI tool/vision fields.
#[derive(Clone)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
    pub reasoning_content: Option<String>,
    pub tool_calls: Vec<Value>,
    pub tool_call_id: Option<String>,
    pub parts: Option<Vec<Value>>,
}

/// The checkpoint's `chat_template.jinja`, rendered with `hf-chat-template`:
/// a minijinja-based engine carrying Hugging Face's compatibility layer
/// (pycompat string methods, Python `tojson`, `strftime_now`, `raise_exception`
/// and the `{% generation %}` block), targeting byte-identical
/// `transformers.apply_chat_template` output.
pub struct ChatTemplate {
    inner: hf_chat_template::ChatTemplate,
    deepseek41: bool,
}

impl ChatTemplate {
    pub fn new(src: &str) -> anyhow::Result<Self> {
        let inner = hf_chat_template::ChatTemplate::from_str(src)
            .map_err(|e| anyhow::anyhow!("chat template parse: {e}"))?;
        Ok(Self {
            inner,
            deepseek41: src.contains("<｜System｜>") && src.contains("reasoning_effort"),
        })
    }

    pub fn render(
        &self,
        messages: &[ChatMessage],
        add_generation_prompt: bool,
        tools: &[Value],
        thinking: bool,
    ) -> anyhow::Result<String> {
        if self.deepseek41 {
            return deepseek41::render(messages, add_generation_prompt, tools, thinking);
        }
        let mut input = hf_chat_template::RenderInput {
            add_generation_prompt,
            ..Default::default()
        };
        input.tools = tools.to_vec();
        for m in messages {
            let mut msg = hf_chat_template::Message::new(m.role.clone(), m.content.clone());
            if let Some(parts) = &m.parts {
                msg.content = Some(hf_chat_template::Content::Parts(parts.clone()));
            }
            if let Some(r) = &m.reasoning_content {
                msg.extra.insert("reasoning_content".to_string(), json!(r));
            }
            if let Some(id) = &m.tool_call_id {
                msg.extra.insert("tool_call_id".to_string(), json!(id));
            }
            if !m.tool_calls.is_empty() {
                let mut calls = m.tool_calls.clone();
                // OpenAI wire format carries `arguments` as a JSON string;
                // the template iterates it as an object (`arguments|items`).
                // Parse it here so the canonical render never sees a string.
                for c in &mut calls {
                    if let Some(f) = c.get_mut("function") {
                        if let Some(a) = f.get("arguments").and_then(|v| v.as_str()) {
                            let parsed: Value =
                                serde_json::from_str(a).unwrap_or_else(|_| json!({}));
                            f["arguments"] = parsed;
                        }
                    }
                }
                msg.tool_calls = calls;
            }
            input.messages.push(msg);
        }
        input
            .extra
            .insert("enable_thinking".to_string(), json!(thinking));
        input.extra.insert(
            "reasoning_effort".to_string(),
            json!(if self.deepseek41 { "high" } else { "xhigh" }),
        );
        input
            .extra
            .insert("preserve_thinking".to_string(), json!(true));
        let mut rendered = self
            .inner
            .render(&input)
            .map_err(|e| anyhow::anyhow!("chat template render: {e}"))?;

        // The GLM checkpoint template defaults an absent or unsupported
        // reasoning_effort to "Max" and emits that system instruction even
        // when enable_thinking=false. DS4 omits the reasoning-effort system
        // message entirely for a no-thinking request; leaving it here changes
        // the prompt tokens and can make the model emit private reasoning as
        // visible content. Match DS4's GLM prompt contract while preserving
        // the checkpoint template for thinking requests and every other model.
        const GLM_PREFIX: &str = "[gMASK]<sop>";
        const GLM_MAX_EFFORT: &str = "<|system|>Reasoning Effort: Max";
        if rendered.starts_with(GLM_PREFIX) {
            normalize_glm_tool_prompt(&mut rendered, tools);
            if !thinking {
                let effort_start = GLM_PREFIX.len();
                let effort_end = effort_start + GLM_MAX_EFFORT.len();
                if rendered
                    .get(effort_start..effort_end)
                    .is_some_and(|text| text == GLM_MAX_EFFORT)
                {
                    rendered.replace_range(effort_start..effort_end, "");
                }
                if add_generation_prompt && rendered.ends_with("<|assistant|><think>") {
                    rendered.push_str("</think>");
                }
            }
        }

        Ok(rendered)
    }
}

/// Match DS4's GLM tool-schema serialization. The checkpoint Jinja template
/// uses Python-style `tojson` spacing inside nested values, while DS4 emits
/// compact nested JSON and preserves only the spaces between top-level fields.
fn ds4_glm_tool_schema_json(tool: &Value) -> Option<String> {
    let schema = tool.get("function").unwrap_or(tool);
    let fields = schema.as_object()?;
    if fields
        .get("defer_loading")
        .and_then(Value::as_bool)
        .unwrap_or(false)
    {
        return None;
    }

    let mut out = String::from("{");
    let mut wrote = false;
    for (key, value) in fields {
        if key == "defer_loading" || key == "strict" {
            continue;
        }
        if wrote {
            out.push_str(", ");
        }
        out.push_str(&serde_json::to_string(key).ok()?);
        out.push_str(": ");
        out.push_str(&serde_json::to_string(value).ok()?);
        wrote = true;
    }
    out.push('}');
    Some(out)
}

fn normalize_glm_tool_prompt(rendered: &mut String, tools: &[Value]) {
    if tools.is_empty() {
        return;
    }
    // The fixed instruction text also contains the literal `<tools></tools>`;
    // match the rendered opening tag followed by its schema newline instead.
    let Some(start) = rendered.find("<tools>\n") else {
        return;
    };
    let Some(relative_end) = rendered[start..].find("</tools>") else {
        return;
    };
    let end = start + relative_end + "</tools>".len();

    let schemas: Vec<String> = tools.iter().filter_map(ds4_glm_tool_schema_json).collect();
    let replacement = if schemas.is_empty() {
        "<tools>\n\n\n</tools>".to_string()
    } else {
        format!("<tools>\n\n{}\n\n\n</tools>", schemas.join("\n\n\n"))
    };
    rendered.replace_range(start..end, &replacement);
}

/// Render the prompt with the checkpoint template when available, else fall
/// back to the engine's text-only rendering.
fn render_prompt(
    tmpl: Option<&ChatTemplate>,
    messages: &[ChatMessage],
    add_generation_prompt: bool,
    tools: &[Value],
    thinking: bool,
) -> anyhow::Result<String> {
    if let Some(t) = tmpl {
        return t.render(messages, add_generation_prompt, tools, thinking);
    }
    let pairs: Vec<(String, String)> = messages
        .iter()
        .map(|m| (m.role.clone(), m.content.clone()))
        .collect();
    Ok(tokenizer::generation_prompt(&pairs))
}

fn session_suffix_text(full_prompt: &str, rest: &str) -> String {
    if full_prompt.starts_with("[gMASK]<sop>") {
        rest.to_string()
    } else {
        format!("\n{rest}")
    }
}

/// Close the ` thinking` block a generation prompt opened, splitting the model's
/// raw output into (reasoning, answer). Falls back to all-content.
fn split_reasoning(text: &str) -> (Option<String>, String) {
    // Markers are ` thinking` / `` — the ASCII forms the checkpoint emits.
    const OPEN: &str = "\u{3c}think\u{3e}";
    const CLOSE: &str = "\u{3c}/think\u{3e}";
    if let Some(idx) = deepseek41::think_end(text) {
        let raw = &text[..idx];
        let raw = raw.strip_prefix(OPEN).unwrap_or(raw);
        let reasoning = raw.trim().to_string();
        let answer = text[idx + CLOSE.len()..].trim_start().to_string();
        (Some(reasoning), answer)
    } else if let Some(idx) = text.find(OPEN) {
        // Unclosed thinking (e.g. truncated by a token cap): keep whatever came
        // before the marker as the answer, not the scratchpad.
        let reasoning = text[idx + OPEN.len()..].trim().to_string();
        let answer = text[..idx].trim_end().to_string();
        (Some(reasoning), answer)
    } else {
        (None, text.to_string())
    }
}

/// Incremental response sections for SSE, following DS4's request-selected
/// parser mode. With thinking disabled, raw generated text stays in content;
/// with thinking enabled, text inside the think block goes to reasoning.
fn streaming_sections(text: &str, thinking_enabled: bool) -> (&str, &str) {
    const OPEN: &str = "\u{3c}think\u{3e}";
    const CLOSE: &str = "\u{3c}/think\u{3e}";
    // DS4 selects the streaming parser mode from the request's thinking
    // setting. In text mode, an incidental closing marker does not switch
    // already-streamed content into reasoning after the fact.
    if !thinking_enabled {
        return ("", text);
    }
    if let Some(idx) = deepseek41::think_end(text) {
        let raw = &text[..idx];
        let reasoning = raw.strip_prefix(OPEN).unwrap_or(raw);
        return (reasoning, &text[idx + CLOSE.len()..]);
    }
    if let Some(idx) = text.find(OPEN) {
        return (&text[idx + OPEN.len()..], &text[..idx]);
    }
    (text, "")
}

/// Final visible sections for an SSE stream. Keep this aligned with the
/// incremental parser: once text mode emitted a prefix, completed-response
/// shaping cannot retroactively move it into `reasoning_content`.
fn streaming_reply_sections(
    full: &str,
    thinking_enabled: bool,
    stops: &[String],
) -> (String, String) {
    let text = strip_stops(full, stops);
    let (reasoning, answer) = streaming_sections(text, thinking_enabled);
    let content = match tool_call_start(answer) {
        Some(i) => answer[..i].trim_end_matches(|ch: char| ch.is_ascii_whitespace()),
        None => answer,
    };
    (reasoning.to_string(), content.to_string())
}

/// Split a completed response according to the requested thinking mode.
/// An unclosed thinking block is reasoning, never user-visible answer text.
fn split_reasoning_for_request(text: &str, thinking_enabled: bool) -> (Option<String>, String) {
    if thinking_enabled && deepseek41::think_end(text).is_none() {
        let reasoning = text.strip_prefix("<think>").unwrap_or(text);
        return (Some(reasoning.trim().to_string()), String::new());
    }
    split_reasoning(text)
}

/// Delta to append so a partially-streamed reply grows to `cur`. Decoding a
/// partial token sequence can rewrite earlier characters (a multi-byte glyph
/// split across tokens), so this diffs against the longest common prefix rather
/// than assuming `prev` is a prefix of `cur`. `None` when nothing new.
fn stream_delta(prev: &str, cur: &str) -> Option<String> {
    if let Some(delta) = cur.strip_prefix(prev) {
        return (!delta.is_empty()).then(|| delta.to_string());
    }
    if cur.len() <= prev.len() {
        return None;
    }
    let mut n = 0;
    for (a, b) in prev.chars().zip(cur.chars()) {
        if a != b {
            break;
        }
        n += a.len_utf8();
    }
    Some(cur[n..].to_string())
}

/// Keep a suffix large enough that a stop string cannot begin in already-sent
/// text and finish in a later token. This mirrors DS4's byte-based streaming
/// holdback while ensuring the returned Rust string slice ends on a UTF-8
/// boundary.
fn streaming_stop_safe_len(text: &str, stops: &[String]) -> usize {
    let hold = stops
        .iter()
        .filter(|stop| !stop.is_empty())
        .map(String::len)
        .max()
        .unwrap_or(0)
        .saturating_sub(1);
    let mut safe_len = text.len().saturating_sub(hold);
    while !text.is_char_boundary(safe_len) {
        safe_len -= 1;
    }
    safe_len
}

/// Split a generated reply into visible content and OpenAI-style `tool_calls`,
/// parsing the Qwen/MiMo text form
/// `<tool_call><function=NAME><parameter=K>V</parameter>…</function></tool_call>`.
/// Any reasoning block is stripped first.
fn extract_tool_calls(text: &str, tools: &[Value]) -> (String, Vec<Value>) {
    let (_, answer) = split_reasoning(text);
    if let Some(result) = deepseek41::extract(&answer) {
        return result.unwrap_or_else(|_| (answer, Vec::new()));
    }
    const OPEN: &str = "<tool_call>";
    const CLOSE: &str = "</tool_call>";
    let Some(start) = answer.find(OPEN) else {
        return (answer, Vec::new());
    };
    let content = answer[..start].trim().to_string();
    let types = tool_types(tools);
    let mut calls = Vec::new();
    let mut rest = &answer[start..];
    while let Some(i) = rest.find(OPEN) {
        let after = &rest[i + OPEN.len()..];
        match after.find(CLOSE) {
            Some(j) => {
                calls.push(parse_tool_call(&after[..j], &types));
                rest = &after[j + CLOSE.len()..];
            }
            None => {
                // Some models omit `</tool_call>` (or get cut off); parse the
                // remainder, which still carries the function and parameters.
                let body = after.split("</tool_call").next().unwrap_or(after);
                calls.push(parse_tool_call(body, &types));
                break;
            }
        }
    }
    (content, calls)
}

/// Per-tool parameter type map (`name -> {param -> "string"|"integer"|…}`) from
/// the request's JSON-schema tool definitions, used to type arguments.
fn tool_types(
    tools: &[Value],
) -> std::collections::HashMap<String, std::collections::HashMap<String, String>> {
    let mut out = std::collections::HashMap::new();
    for t in tools {
        let Some(f) = t.get("function") else {
            continue;
        };
        let name = f
            .get("name")
            .and_then(|n| n.as_str())
            .unwrap_or_default()
            .to_string();
        let mut params = std::collections::HashMap::new();
        if let Some(props) = f
            .get("parameters")
            .and_then(|p| p.get("properties"))
            .and_then(|p| p.as_object())
        {
            for (k, v) in props {
                if let Some(ty) = v.get("type").and_then(|x| x.as_str()) {
                    params.insert(k.clone(), ty.to_string());
                }
            }
        }
        out.insert(name, params);
    }
    out
}

/// Parse one `<function=NAME>…<parameter=K>V</parameter>…</function>` body into
/// an OpenAI tool call, with `arguments` serialized as a JSON string. Models are
/// inconsistent: some emit `<parameter>` tags, others put a raw JSON object
/// inside the `<function=…>` tag — accept both. Values are typed from the tool
/// schema so `4` stays a number and `true` a boolean.
fn parse_tool_call(
    body: &str,
    types: &std::collections::HashMap<String, std::collections::HashMap<String, String>>,
) -> Value {
    if body.contains("<arg_key>") {
        let Some((raw_name, _)) = body.split_once("<arg_key>") else {
            unreachable!();
        };
        let name = raw_name.trim();
        let mut qwen_form = format!("<function={name}>");
        let mut rest = body;
        loop {
            let Some(start) = rest.find("<arg_key>") else {
                break;
            };
            rest = &rest[start + "<arg_key>".len()..];
            let Some((key, after_key)) = rest.split_once("</arg_key>") else {
                break;
            };
            let Some(value_start) = after_key.find("<arg_value>") else {
                break;
            };
            let value_start = value_start + "<arg_value>".len();
            let after_value_start = &after_key[value_start..];
            let (value, after_value) = match after_value_start.split_once("</arg_value>") {
                Some((value, after)) => (value, Some(after)),
                None => (after_value_start, None),
            };
            qwen_form.push_str("<parameter=");
            qwen_form.push_str(key.trim());
            qwen_form.push('>');
            qwen_form.push_str(value.trim());
            qwen_form.push_str("</parameter>");
            match after_value {
                Some(after) => rest = after,
                None => break,
            }
        }
        qwen_form.push_str("</function>");
        return parse_tool_call(&qwen_form, types);
    }

    // Name comes from `<function=NAME>` or, for JSON-form calls, `"name"`.
    let mut name = body
        .split_once("<function=")
        .and_then(|(_, r)| r.split_once('>'))
        .map(|(n, _)| n.trim().to_string())
        .unwrap_or_default();
    let mut raw = Value::Null;
    if name.is_empty() {
        if let Some(obj) = find_json_object(body) {
            name = obj
                .get("name")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            raw = obj
                .get("arguments")
                .or_else(|| obj.get("parameters"))
                .cloned()
                .unwrap_or(Value::Null);
        }
    }
    let declared = types.get(&name);
    let coerce = |key: &str, val: &str| -> Value {
        match declared.and_then(|m| m.get(key)).map(String::as_str) {
            Some("integer") => val
                .trim()
                .parse::<i64>()
                .map(Value::from)
                .unwrap_or_else(|_| Value::String(val.to_string())),
            Some("number") => val
                .trim()
                .parse::<f64>()
                .map(Value::from)
                .unwrap_or_else(|_| Value::String(val.to_string())),
            Some("boolean") => match val.trim() {
                "true" => Value::Bool(true),
                "false" => Value::Bool(false),
                _ => Value::String(val.to_string()),
            },
            Some("object") | Some("array") | Some("null") => {
                serde_json::from_str(val.trim()).unwrap_or_else(|_| Value::String(val.to_string()))
            }
            // No schema (or a string): parse a bare JSON scalar, else keep the text.
            _ => serde_json::from_str::<Value>(val.trim())
                .ok()
                .filter(|v| matches!(v, Value::Number(_) | Value::Bool(_) | Value::Null))
                .unwrap_or_else(|| Value::String(val.to_string())),
        }
    };
    let mut args = serde_json::Map::new();
    if let Value::Object(o) = raw {
        args = o;
    }
    if args.is_empty() {
        let mut rest = body;
        while let Some((_, r)) = rest.split_once("<parameter") {
            // Accept `<parameter=K>V</parameter>` and `<parameter name="K">V`.
            let (key, r2) = if let Some(x) = r.strip_prefix('=') {
                match x.split_once('>') {
                    Some((k, r2)) => (k.to_string(), r2),
                    None => break,
                }
            } else {
                let Some(nv) = r.split_once('>').map(|(a, _)| a) else {
                    break;
                };
                let Some(eq) = nv.find('=') else {
                    break;
                };
                let k = nv[eq + 1..]
                    .trim()
                    .trim_matches(|c| c == '"' || c == '\'')
                    .to_string();
                let Some((_, r2)) = r.split_once('>') else {
                    break;
                };
                (k, r2)
            };
            let key = key.trim().to_string();
            let (val, after) = match r2.split_once("</parameter>") {
                Some((v, a)) => (v.to_string(), Some(a)),
                None => {
                    // No closing tag: take up to the next parameter, function
                    // or tool call, or the end of the text.
                    let end = r2
                        .find("</function")
                        .or_else(|| r2.find("<parameter"))
                        .or_else(|| r2.find("</tool_call"))
                        .unwrap_or(r2.len());
                    (r2[..end].to_string(), None)
                }
            };
            if !key.is_empty() {
                // The template's documented XML form puts the value on its own
                // line(s); edge whitespace is format, not data.
                args.insert(key.clone(), coerce(&key, val.trim()));
            }
            match after {
                Some(a) => rest = a,
                None => break,
            }
        }
    }
    if args.is_empty() {
        // Last resort: a JSON object anywhere in the body, repaired if the
        // model left it unbalanced.
        if let Some(obj) = find_json_object(body).or_else(|| salvage_json_object(body)) {
            args = obj;
        }
    }
    json!({
        "id": tool_call_id(),
        "type": "function",
        "function": {
            "name": name,
            "arguments": Value::Object(args).to_string(),
        },
    })
}

/// Recover a JSON object the model left unbalanced (a missing closing quote or
/// brace, or a stray `</parameter>`), by cutting at the next tag and closing
/// whatever is still open. Enough to keep a truncated `write_file` valid.
fn salvage_json_object(s: &str) -> Option<serde_json::Map<String, Value>> {
    let start = s.find('{')?;
    let tail = &s[start..];
    let cut = ["</function", "</tool_call", "</parameter"]
        .iter()
        .filter_map(|m| tail.find(m))
        .min()
        .unwrap_or(tail.len());
    let chars: Vec<char> = tail[..cut].chars().collect();
    let mut out = String::new();
    let mut in_str = false;
    let mut esc = false;
    let mut stack: Vec<char> = Vec::new();
    let mut expect_key = false;
    let mut expect_val = false;
    let mut i = 0;
    while i < chars.len() {
        let ch = chars[i];
        if in_str {
            // JSON strings cannot carry a raw newline/CR/tab; escape them so a
            // truncated multi-line value can still be parsed.
            if esc {
                out.push(ch);
                esc = false;
            } else {
                match ch {
                    '\\' => {
                        out.push('\\');
                        esc = true;
                    }
                    '"' => {
                        out.push('"');
                        in_str = false;
                    }
                    '\n' => out.push_str("\\n"),
                    '\r' => out.push_str("\\r"),
                    '\t' => out.push_str("\\t"),
                    _ => out.push(ch),
                }
            }
            i += 1;
            continue;
        }
        match ch {
            '"' => {
                in_str = true;
                out.push(ch);
                expect_key = false;
                expect_val = false;
            }
            '{' => {
                stack.push('}');
                out.push(ch);
                expect_key = true;
                expect_val = false;
            }
            '[' => {
                stack.push(']');
                out.push(ch);
                expect_key = false;
                expect_val = false;
            }
            '}' | ']' => {
                if stack.last() == Some(&ch) {
                    stack.pop();
                    out.push(ch);
                }
                expect_key = false;
                expect_val = false;
            }
            ',' => {
                out.push(ch);
                expect_key = stack.last() == Some(&'}');
                expect_val = false;
            }
            ':' => {
                out.push(ch);
                expect_val = true;
                expect_key = false;
            }
            c if c.is_whitespace() => out.push(c),
            // Stray tag fragments the model emitted in place of the closing
            // brace (e.g. `"</}`) are not JSON structure.
            '<' | '>' | '/' => {}
            c if expect_key && (c.is_alphabetic() || c == '_' || c == '$') => {
                let mut id = String::new();
                while i < chars.len()
                    && (chars[i].is_alphanumeric() || chars[i] == '_' || chars[i] == '$')
                {
                    id.push(chars[i]);
                    i += 1;
                }
                out.push('"');
                out.push_str(&id);
                out.push('"');
                expect_key = false;
                continue;
            }
            c if expect_val && (c.is_alphabetic() || c == '_') => {
                let mut id = String::new();
                while i < chars.len()
                    && !chars[i].is_whitespace()
                    && chars[i] != ','
                    && chars[i] != '}'
                    && chars[i] != ']'
                {
                    id.push(chars[i]);
                    i += 1;
                }
                if matches!(id.as_str(), "true" | "false" | "null") {
                    out.push_str(&id);
                } else {
                    out.push('"');
                    out.push_str(&id);
                    out.push('"');
                }
                expect_val = false;
                continue;
            }
            c => out.push(c),
        }
        i += 1;
    }
    if in_str {
        out.push('"');
    }
    while let Some(c) = stack.pop() {
        out.push(c);
    }
    match serde_json::from_str::<Value>(&out) {
        Ok(Value::Object(o)) => Some(o),
        _ => None,
    }
}

/// The first balanced `[…]` JSON array found in `s`, if any.
fn find_json_object(s: &str) -> Option<serde_json::Map<String, Value>> {
    let bytes = s.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'{' {
            let mut depth = 0i32;
            let mut in_str = false;
            let mut esc = false;
            let mut j = i;
            while j < bytes.len() {
                let c = bytes[j];
                if in_str {
                    if esc {
                        esc = false;
                    } else if c == b'\\' {
                        esc = true;
                    } else if c == b'"' {
                        in_str = false;
                    }
                } else if c == b'"' {
                    in_str = true;
                } else if c == b'{' {
                    depth += 1;
                } else if c == b'}' {
                    depth -= 1;
                    if depth == 0 {
                        if let Ok(Value::Object(o)) = serde_json::from_str::<Value>(&s[i..=j]) {
                            return Some(o);
                        }
                        break;
                    }
                }
                j += 1;
            }
        }
        i += 1;
    }
    None
}

fn tool_call_id() -> String {
    static N: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let n = N.fetch_add(1, Ordering::Relaxed);
    format!("call_{:x}{:x}", now(), n)
}

/// Whether the streamed/generated text has opened a tool call (so further text
/// must not be streamed as assistant content).
fn tool_call_start(text: &str) -> Option<usize> {
    text.find("<tool_call>")
        .into_iter()
        .chain(deepseek41::start(text))
        .min()
}

/// Match ds4's live GLM/Qwen DSML sampling boundary. Structural tool syntax is
/// sampled greedily; parameter bodies use the request's configured sampler.
#[cfg(test)]
fn dsml_sampling_mode(output: &str, thinking_enabled: bool) -> TargetMtpSamplingMode {
    match dsml_decode_state(output, thinking_enabled) {
        DsmlDecodeState::Greedy => TargetMtpSamplingMode::Greedy,
        DsmlDecodeState::Configured | DsmlDecodeState::Done => TargetMtpSamplingMode::Configured,
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum DsmlDecodeState {
    Configured,
    Greedy,
    Done,
}

#[cfg(test)]
fn dsml_decode_state(output: &str, thinking_enabled: bool) -> DsmlDecodeState {
    const THINK_END: &str = "</think>";
    const TOOL_OPEN: &str = "<tool_call>";
    const TOOL_CLOSE: &str = "</tool_call>";
    const GLM_VALUE_OPEN: &str = "<arg_value>";
    const GLM_VALUE_CLOSE: &str = "</arg_value>";
    const QWEN_VALUE_OPEN: &str = "<parameter=";
    const QWEN_VALUE_CLOSE: &str = "</parameter>";
    const QWEN_VALUE_OPEN_ALT: &str = "<parameter ";

    fn partial_suffix_len(text: &str, literal: &str) -> usize {
        let max = text.len().min(literal.len().saturating_sub(1));
        (1..=max)
            .rev()
            .find(|&len| text.ends_with(&literal[..len]))
            .unwrap_or(0)
    }

    let mut pos = 0usize;
    if thinking_enabled {
        let Some(end) = output.find(THINK_END) else {
            return DsmlDecodeState::Configured;
        };
        pos = end + THINK_END.len();
    }

    let mut structural = false;
    let mut payload_close: Option<&str> = None;
    while pos < output.len() {
        if let Some(close) = payload_close {
            if let Some(relative) = output[pos..].find(close) {
                pos += relative + close.len();
                payload_close = None;
                structural = true;
                continue;
            }
            return if partial_suffix_len(&output[pos..], close) >= 2 {
                DsmlDecodeState::Greedy
            } else {
                DsmlDecodeState::Configured
            };
        }

        if !structural {
            if let Some(relative) = output[pos..].find(TOOL_OPEN) {
                pos += relative + TOOL_OPEN.len();
                structural = true;
                continue;
            }
            return DsmlDecodeState::Configured;
        }

        let tail = &output[pos..];
        if tail.starts_with(TOOL_CLOSE) {
            return DsmlDecodeState::Done;
        }
        if tail.starts_with(GLM_VALUE_OPEN) {
            pos += GLM_VALUE_OPEN.len();
            payload_close = Some(GLM_VALUE_CLOSE);
            continue;
        }
        if tail.starts_with(QWEN_VALUE_OPEN) || tail.starts_with(QWEN_VALUE_OPEN_ALT) {
            let Some(end) = tail.find('>') else {
                return DsmlDecodeState::Greedy;
            };
            pos += end + 1;
            payload_close = Some(QWEN_VALUE_CLOSE);
            continue;
        }

        let ch = tail.chars().next().expect("non-empty structural tail");
        pos += ch.len_utf8();
    }

    if payload_close.is_some() {
        DsmlDecodeState::Configured
    } else if structural {
        DsmlDecodeState::Greedy
    } else {
        DsmlDecodeState::Configured
    }
}

fn parse_messages(body: &Value) -> anyhow::Result<Vec<ChatMessage>> {
    let mut out: Vec<ChatMessage> = Vec::new();
    let as_content = |m: &Value| -> String {
        match m.get("content") {
            Some(Value::String(s)) => s.clone(),
            Some(Value::Array(parts)) => parts
                .iter()
                .filter_map(|p| p.get("text").and_then(|t| t.as_str()))
                .collect::<Vec<_>>()
                .join(""),
            _ => String::new(),
        }
    };
    if let Some(sys) = body.get("system").and_then(|v| v.as_str()) {
        out.push(ChatMessage {
            role: "system".to_string(),
            content: sys.to_string(),
            reasoning_content: None,
            tool_calls: Vec::new(),
            tool_call_id: None,
            parts: None,
        });
    }
    if let Some(arr) = body.get("messages").and_then(|v| v.as_array()) {
        for m in arr {
            let role = m
                .get("role")
                .and_then(|v| v.as_str())
                .unwrap_or("user")
                .to_string();
            let raw = as_content(m);
            let mut reasoning = m
                .get("reasoning_content")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string());
            let mut content = raw;
            if role == "assistant" && reasoning.is_none() {
                let (r, c) = split_reasoning(&content);
                if r.is_some() {
                    reasoning = r;
                    content = c;
                }
            }
            let tool_calls = m
                .get("tool_calls")
                .and_then(|v| v.as_array())
                .cloned()
                .unwrap_or_default();
            let tool_call_id = m
                .get("tool_call_id")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string());
            let parts = m.get("content").and_then(|c| c.as_array()).cloned();
            out.push(ChatMessage {
                role,
                content,
                reasoning_content: reasoning,
                tool_calls,
                tool_call_id,
                parts,
            });
        }
    } else if let Some(p) = body.get("prompt").and_then(|v| v.as_str()) {
        out.push(ChatMessage {
            role: "user".to_string(),
            content: p.to_string(),
            reasoning_content: None,
            tool_calls: Vec::new(),
            tool_call_id: None,
            parts: None,
        });
    }
    anyhow::ensure!(!out.is_empty(), "request has no messages or prompt");
    Ok(out)
}

/// The tools to advertise for a request, honoring `tool_choice: "none"` (empty)
/// and a named-function choice (only that tool).
fn request_tools(job: &Job) -> Vec<Value> {
    let Some(tools) = job.body.get("tools").and_then(|v| v.as_array()) else {
        return Vec::new();
    };
    match job.body.get("tool_choice") {
        Some(Value::String(s)) if s == "none" => Vec::new(),
        Some(Value::Object(o)) if o.get("type").and_then(|t| t.as_str()) == Some("function") => {
            match o
                .get("function")
                .and_then(|f| f.get("name"))
                .and_then(|n| n.as_str())
            {
                Some(n) => tools
                    .iter()
                    .filter(|t| {
                        t.get("function")
                            .and_then(|f| f.get("name"))
                            .and_then(|x| x.as_str())
                            == Some(n)
                    })
                    .cloned()
                    .collect(),
                None => tools.clone(),
            }
        }
        _ => tools.clone(),
    }
}

fn parallel_tools(job: &Job) -> bool {
    job.body
        .get("parallel_tool_calls")
        .and_then(|v| v.as_bool())
        .unwrap_or(true)
}

fn request_stops(job: &Job) -> Vec<String> {
    match job.body.get("stop") {
        Some(Value::String(s)) => vec![s.clone()],
        Some(Value::Array(a)) => a
            .iter()
            .filter_map(|v| v.as_str().map(|s| s.to_string()))
            .collect(),
        _ => Vec::new(),
    }
}

fn request_max_tokens(job: &Job, cfg: &ServerConfig) -> usize {
    job.body
        .get("max_completion_tokens")
        .or_else(|| job.body.get("max_tokens"))
        .and_then(|v| v.as_u64())
        .map(|v| v as usize)
        .unwrap_or(cfg.max_tokens)
}

fn has_tools(job: &Job) -> bool {
    !request_tools(job).is_empty()
}

fn shape_reply(
    full: &str,
    tools: &[Value],
    parallel: bool,
    stops: &[String],
    thinking_enabled: bool,
) -> (String, Option<String>, Vec<Value>) {
    let text = strip_stops(full, stops);
    let (reasoning, answer) = split_reasoning_for_request(text, thinking_enabled);
    if tools.is_empty() {
        (answer, reasoning, Vec::new())
    } else {
        let (content, mut calls) = extract_tool_calls(&answer, tools);
        if !parallel && calls.len() > 1 {
            calls.truncate(1);
        }
        (content, reasoning, calls)
    }
}

/// The JSON Schema (if any) a `response_format` asks for: `Some(schema)` for a
/// structured-output request, an empty-object schema for `json_object`.
fn response_format_of(body: &Value) -> Option<Value> {
    let rf = body.get("response_format")?;
    match rf.get("type").and_then(|t| t.as_str())? {
        "json_object" => Some(json!({ "type": "object" })),
        "json_schema" => rf
            .get("json_schema")
            .and_then(|j| j.get("schema"))
            .or_else(|| rf.get("schema"))
            .cloned()
            .or_else(|| Some(json!({ "type": "object" }))),
        _ => None,
    }
}

/// A system instruction that pushes the model into emitting the requested JSON
/// shape (the engine has no grammar-constrained decoding).
fn json_instruction(schema: Option<&Value>) -> String {
    match schema {
        Some(sc) if !sc.as_object().map(|o| o.is_empty()).unwrap_or(false) => format!(
            "Respond with a single valid JSON value that conforms to this JSON Schema and \
nothing else. Do not include prose, explanations, or markdown code fences.\n{sc}"
        ),
        _ => "Respond with a single valid JSON object and nothing else. Do not include \
prose, explanations, or markdown code fences."
            .to_string(),
    }
}

/// Pull the JSON value out of a model reply, stripping markdown fences and any
/// surrounding prose. Returns a compact serialization, or `None`.
fn extract_json(text: &str) -> Option<String> {
    let mut t = text.trim();
    if let Some(r) = t.strip_prefix("```json").or_else(|| t.strip_prefix("```")) {
        t = r.trim_start();
    }
    if let Some(r) = t.strip_suffix("```") {
        t = r.trim_end();
    }
    if let Ok(v) = serde_json::from_str::<Value>(t) {
        return Some(v.to_string());
    }
    if let Some(o) = find_json_object(t) {
        return Some(Value::Object(o).to_string());
    }
    find_json_array(t).map(|a| a.to_string())
}

/// Truncate `text` at the earliest stop sequence.
fn strip_stops<'a>(text: &'a str, stops: &[String]) -> &'a str {
    match stops
        .iter()
        .filter(|s| !s.is_empty())
        .filter_map(|s| text.find(s.as_str()))
        .min()
    {
        Some(cut) => &text[..cut],
        None => text,
    }
}

/// Drop a leading ` thinking…</think>` block so a JSON body is not scanned
/// through marker text (marker strings inside the JSON are data, not markers).
fn strip_think_prefix(text: &str) -> &str {
    const OPEN: &str = "\u{3c}think\u{3e}";
    const CLOSE: &str = "\u{3c}/think\u{3e}";
    let t = text.trim_start();
    if t.starts_with(OPEN) {
        if let Some(i) = t.find(CLOSE) {
            return t[i + CLOSE.len()..].trim_start();
        }
    }
    text
}

/// Nudge a parsed value toward a JSON Schema where the model only approximated
/// it: fill missing required properties, drop extras, coerce primitive types,
/// and complete a string `enum` value the model truncated (an enum constrains
/// the value, so the schema decides it — the same output a constrained decoder
/// would emit).
fn repair_json(v: &mut Value, schema: &Value) {
    match schema.get("type").and_then(|t| t.as_str()) {
        Some("object") => {
            let Value::Object(mut obj) = std::mem::take(v) else {
                *v = json!({});
                return;
            };
            if let Some(props) = schema.get("properties").and_then(|p| p.as_object()) {
                for (k, ps) in props {
                    if obj.contains_key(k) {
                        repair_json(obj.get_mut(k).unwrap(), ps);
                    } else if schema
                        .get("required")
                        .and_then(|r| r.as_array())
                        .map(|r| r.iter().any(|x| x.as_str() == Some(k.as_str())))
                        .unwrap_or(false)
                    {
                        let mut nv = Value::Null;
                        repair_json(&mut nv, ps);
                        obj.insert(k.clone(), nv);
                    }
                }
            }
            if schema.get("additionalProperties").and_then(|a| a.as_bool()) == Some(false) {
                if let Some(props) = schema.get("properties").and_then(|p| p.as_object()) {
                    obj.retain(|k, _| props.contains_key(k));
                }
            }
            *v = Value::Object(obj);
        }
        Some("array") => {
            if let (Value::Array(a), Some(items)) = (v, schema.get("items")) {
                for x in a.iter_mut() {
                    repair_json(x, items);
                }
            }
        }
        Some("string") => {
            if let Some(e) = schema.get("enum").and_then(|e| e.as_array()) {
                if !e.iter().any(|x| x == &*v) {
                    let cur = v.as_str().unwrap_or("");
                    let pick = if e.len() == 1 {
                        e.first()
                    } else {
                        e.iter()
                            .find(|x| x.as_str().map(|s| s.starts_with(cur)).unwrap_or(false))
                    };
                    if let Some(p) = pick {
                        *v = p.clone();
                    }
                }
            } else if !v.is_string() {
                *v = Value::String(match &*v {
                    Value::Null => String::new(),
                    other => other.to_string(),
                });
            }
        }
        Some("integer") => {
            if let Some(n) = v.as_f64() {
                *v = json!(n as i64);
            } else if let Some(n) = v.as_str().and_then(|s| s.trim().parse::<i64>().ok()) {
                *v = json!(n);
            }
        }
        Some("number") => {
            if let Some(n) = v.as_str().and_then(|s| s.trim().parse::<f64>().ok()) {
                *v = json!(n);
            }
        }
        Some("boolean") => {
            if let Some(b) = v.as_str().and_then(|s| match s {
                "true" => Some(true),
                "false" => Some(false),
                _ => None,
            }) {
                *v = json!(b);
            }
        }
        _ => {}
    }
}

/// Validate a parsed JSON value against (the subset of JSON Schema the server
/// honors): types, `required` object properties, `enum`, and nested
/// `properties`/`items`. Returns human-readable violations; empty == valid.
/// This backs the validate+retry path for `response_format` — the engine has
/// no grammar-constrained decoding, so conformance is checked after
/// generation (repair first, then one corrective retry, documented in the
/// serving-api spec).
fn json_schema_errors(v: &Value, schema: &Value) -> Vec<String> {
    let mut errs = Vec::new();
    let ty = schema.get("type").and_then(|t| t.as_str());
    let type_ok = |v: &Value, ty: &str| match ty {
        "object" => v.is_object(),
        "array" => v.is_array(),
        "string" => v.is_string(),
        "integer" => v.is_i64() || v.is_u64(),
        "number" => v.is_number(),
        "boolean" => v.is_boolean(),
        "null" => v.is_null(),
        _ => true,
    };
    if let Some(ty) = ty {
        if !type_ok(v, ty) {
            errs.push(format!("expected {ty}, got {}", type_name_of(v)));
            return errs;
        }
    }
    if let Some(e) = schema.get("enum").and_then(|e| e.as_array()) {
        if !e.iter().any(|x| x == v) {
            errs.push(format!("value {v} not in enum"));
        }
    }
    match ty {
        Some("object") | None if v.is_object() => {
            if let Some(props) = schema.get("properties").and_then(|p| p.as_object()) {
                for (k, ps) in props {
                    if let Some(x) = v.get(k) {
                        for e in json_schema_errors(x, ps) {
                            errs.push(format!("{k}: {e}"));
                        }
                    }
                }
            }
            if let Some(req) = schema.get("required").and_then(|r| r.as_array()) {
                for r in req {
                    if let Some(k) = r.as_str() {
                        if !v.get(k).map(|x| !x.is_null()).unwrap_or(false) {
                            errs.push(format!("missing required property {k:?}"));
                        }
                    }
                }
            }
        }
        Some("array") => {
            if let (Some(items), Value::Array(a)) = (schema.get("items"), v) {
                for (i, x) in a.iter().enumerate() {
                    for e in json_schema_errors(x, items) {
                        errs.push(format!("[{i}]: {e}"));
                    }
                }
            }
        }
        _ => {}
    }
    errs
}

fn type_name_of(v: &Value) -> &'static str {
    match v {
        Value::Null => "null",
        Value::Bool(_) => "boolean",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}

fn find_json_array(s: &str) -> Option<Value> {
    let bytes = s.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'[' {
            let mut depth = 0i32;
            let mut in_str = false;
            let mut esc = false;
            let mut j = i;
            while j < bytes.len() {
                let c = bytes[j];
                if in_str {
                    if esc {
                        esc = false;
                    } else if c == b'\\' {
                        esc = true;
                    } else if c == b'"' {
                        in_str = false;
                    }
                } else if c == b'"' {
                    in_str = true;
                } else if c == b'[' {
                    depth += 1;
                } else if c == b']' {
                    depth -= 1;
                    if depth == 0 {
                        if let Ok(v) = serde_json::from_str::<Value>(&s[i..=j]) {
                            return Some(v);
                        }
                        break;
                    }
                }
                j += 1;
            }
        }
        i += 1;
    }
    None
}

/// Sampling decision chain: request body >
/// server flags > checkpoint `generation_config.json` > hardcoded defaults
/// (greedy). Applied identically on every generation path (batched, serial,
/// MTP-verify), so a silent request samples the same way everywhere.
fn sampler_from(body: &Value, cfg: &ServerConfig) -> Sampler {
    let f = |k: &str, flag: Option<f32>, generation: Option<f32>, d: f32| {
        body.get(k)
            .and_then(|v| v.as_f64())
            .map(|v| v as f32)
            .or(flag)
            .or(generation)
            .unwrap_or(d)
    };
    let u = |k: &str, flag: Option<usize>, generation: Option<usize>, d: usize| {
        body.get(k)
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .or(flag)
            .or(generation)
            .unwrap_or(d)
    };
    let seed = body
        .get("seed")
        .and_then(|v| v.as_u64())
        .unwrap_or(0x9E37_79B9_7F4A_7C15);
    Sampler {
        temperature: f(
            "temperature",
            cfg.temperature,
            cfg.gen_defaults.temperature,
            0.0,
        ),
        top_k: u("top_k", cfg.top_k, cfg.gen_defaults.top_k, 0),
        top_p: f("top_p", cfg.top_p, cfg.gen_defaults.top_p, 1.0),
        min_p: f("min_p", cfg.min_p, None, 0.0),
        repetition_penalty: f(
            "repetition_penalty",
            cfg.rep_penalty,
            cfg.gen_defaults.rep_penalty,
            1.0,
        ),
        mtp_exact_sampling: cfg.mtp_exact_sampling,
        seed,
        ..Default::default()
    }
}

/// Whether a request renders its generation prompt with thinking on: an
/// explicit request value wins (`chat_template_kwargs.enable_thinking`,
/// `reasoning_effort` / `reasoning.effort`), then the checkpoint's own
/// declaration (`generation_config.json` →
/// `default_chat_template_kwargs.enable_thinking`), then off. Silent requests
/// default to off.
fn request_thinking(body: &Value, cfg: &ServerConfig) -> bool {
    if let Some(v) = body
        .get("chat_template_kwargs")
        .and_then(|c| c.get("enable_thinking"))
        .and_then(|v| v.as_bool())
    {
        return v;
    }
    let effort = body
        .get("reasoning_effort")
        .and_then(|v| v.as_str())
        .or_else(|| {
            body.get("reasoning")
                .and_then(|r| r.get("effort"))
                .and_then(|v| v.as_str())
        });
    if let Some(e) = effort {
        return e != "none";
    }
    cfg.gen_defaults.thinking.unwrap_or(false)
}

fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn completion_id() -> String {
    let n = SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    format!("chatcmpl-lisa-{:x}{:x}", now(), n)
}

static SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

// ─────────────────────────── decisions ───────────────────────────

/// Serve a non-generative decision model (e.g. Laya) over HTTP.
///
/// Endpoints: `POST /v1/decisions` (`{"state":…,"questions":…}` → answers),
/// `GET /v1/models`, `GET /health`. The model is a pure function, so requests
/// are handled inline on the accept thread (single-threaded; fast, ~30 ms).
pub fn run_decisions(model: Box<dyn DecisionModel>, cfg: ServerConfig) -> anyhow::Result<()> {
    let listener = TcpListener::bind(&cfg.addr)?;
    let addr = listener.local_addr()?;
    println!("lisa decisions listening on http://{addr}");
    println!("  POST /v1/decisions   GET /v1/models   GET /health");
    for stream in listener.incoming() {
        match stream {
            Ok(stream) => {
                if let Err(e) = handle_decision_conn(stream, model.as_ref()) {
                    eprintln!("decisions: {e}");
                }
            }
            Err(e) => eprintln!("accept: {e}"),
        }
    }
    Ok(())
}

fn handle_decision_conn(mut stream: TcpStream, model: &dyn DecisionModel) -> anyhow::Result<()> {
    let Some((method, path, body)) = read_request(&mut stream)? else {
        return Ok(());
    };
    if method == "GET" && (path == "/health" || path == "/healthz") {
        return write_response(
            &mut stream,
            "200 OK",
            "application/json",
            br#"{"status":"ok"}"#,
        );
    }
    if method == "GET" && path.starts_with("/v1/models") {
        let payload = json!({"object":"list","data":[{"id":model.name(),"object":"model"}]});
        return write_response(
            &mut stream,
            "200 OK",
            "application/json",
            payload.to_string().as_bytes(),
        );
    }
    if method == "POST" && path == "/v1/decisions" {
        let req: Value = match serde_json::from_slice(&body) {
            Ok(v) => v,
            Err(e) => return bad_request(&mut stream, format!("invalid JSON: {e}")),
        };
        let state = req.get("state").cloned().unwrap_or(Value::Null);
        let questions = req.get("questions").cloned().unwrap_or(Value::Null);
        return match model.system_one(&state, &questions) {
            Ok(ans) => write_response(
                &mut stream,
                "200 OK",
                "application/json",
                ans.to_string().as_bytes(),
            ),
            Err(e) => bad_request(&mut stream, e.to_string()),
        };
    }
    write_response(
        &mut stream,
        "404 Not Found",
        "application/json",
        br#"{"error":{"message":"not found"}}"#,
    )
}

fn bad_request(stream: &mut TcpStream, msg: String) -> anyhow::Result<()> {
    let body = json!({ "error": { "message": msg } }).to_string();
    write_response(
        stream,
        "400 Bad Request",
        "application/json",
        body.as_bytes(),
    )
}

#[cfg(test)]
#[path = "tests.rs"]
mod render_test;
