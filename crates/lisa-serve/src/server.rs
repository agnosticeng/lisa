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

use crate::application;
use std::collections::HashMap;
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::{Value, json};

#[path = "api_state.rs"]
mod api_state;
#[path = "http.rs"]
mod http;
#[path = "images.rs"]
pub mod images;
#[path = "responses.rs"]
mod responses;
use api_state::ApiState;
use http::{read_request, write_response};
#[cfg(test)]
use responses::{ReplyData, prompt_tokens_details_json, usage_json};
use responses::{
    collect_choices, collect_surface, drain_reply, empty_response_resource,
    responses_payload_with_id, store_conversation, stream_surface,
};
#[path = "deepseek41.rs"]
mod deepseek41;
#[path = "streaming.rs"]
mod streaming;
#[path = "wait.rs"]
mod wait;
use wait::WakeSender;

use crate::application::PrefixCache;
use crate::application::Sampler;
use crate::application::Session;
use lisa_engine::DecisionModel;
use lisa_engine::core::tokenizer::{self, IncrementalDecoder, Tokenizer};
use lisa_engine::generation::is_eos;
use lisa_engine::language::LanguageModel;
use lisa_engine::sampling::sampler::token_logprob;
use lisa_engine::speculation::{MtpDepth, TargetMtpEosPolicy, TargetMtpSamplingMode};
#[path = "server/waves.rs"]
mod waves;
use waves::*;
#[path = "server/requests.rs"]
mod requests;
use requests::*;
#[path = "server/completion.rs"]
mod completion;
use completion::*;
#[path = "server/templates.rs"]
mod templates;
pub use templates::*;
#[path = "server/tools.rs"]
mod tools;
pub use tools::*;
#[path = "server/structured.rs"]
mod structured;
use structured::*;

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
    /// Enable the reference's exact stochastic acceptance/residual sampling for GLM MTP.
    pub mtp_exact_sampling: bool,
    /// The checkpoint's own sampling/thinking recommendations
    /// (`generation_config.json`): `temperature`, `top_p`, `top_k`,
    /// `repetition_penalty`, and `default_chat_template_kwargs
    /// .enable_thinking` (the model author's thinking default).
    pub gen_defaults: GenDefaults,
    pub depth: usize,
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
        if let Some(name) = anc.file_name().and_then(|n| n.to_str())
            && let Some(rest) = name.strip_prefix("models--")
        {
            return rest.replacen("--", "/", 1);
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
    tower: &mut dyn LanguageModel<crate::application::LanguageRuntime>,
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
    sessions: HashMap<String, StoredSession>,
    prefix: PrefixCache,
    chat_tmpl: Option<ChatTemplate>,
    stop: std::sync::Arc<AtomicBool>,
    /// Held so the bound port stays open until [`Server::shutdown`] drops it.
    listener: Option<std::sync::Arc<TcpListener>>,
    accept: Option<std::thread::JoinHandle<()>>,
    addr: std::net::SocketAddr,
    /// Byte-level admission budget; `None` without an explicit RAM cap, in
    /// which case no request is ever refused.
    admission: Option<lisa_engine::core::admission::MemoryBudget>,
}

impl Server {
    pub fn start(
        cfg: ServerConfig,
        stop: std::sync::Arc<AtomicBool>,
        device: application::Device,
    ) -> anyhow::Result<Server> {
        Self::start_inner(cfg, stop, device, None)
    }

    /// Wake a host's existing engine loop after a request is enqueued. The
    /// callback must only notify that loop; it must not execute the model.
    pub fn start_with_waker(
        cfg: ServerConfig,
        stop: std::sync::Arc<AtomicBool>,
        device: application::Device,
        wake: std::sync::Arc<dyn Fn() + Send + Sync>,
    ) -> anyhow::Result<Server> {
        Self::start_inner(cfg, stop, device, Some(wake))
    }

    fn start_inner(
        cfg: ServerConfig,
        stop: std::sync::Arc<AtomicBool>,
        device: application::Device,
        wake: Option<std::sync::Arc<dyn Fn() + Send + Sync>>,
    ) -> anyhow::Result<Server> {
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
        let tx = WakeSender::new(tx, wake);
        let prefix = PrefixCache::new(device, cfg.prefix_cache, application::prefix_spill());
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
            wait::accept(listener, stop, move |stream| {
                application::trace_http("serve.http.accept");
                let _ = stream.set_nonblocking(false);
                let tx = tx.clone();
                let cfg = cfg.clone();
                let api = api.clone();
                std::thread::spawn(move || {
                    if let Err(e) = handle_conn(&api, stream, tx, &cfg) {
                        eprintln!("[serve] connection error: {e}");
                    }
                });
            })?
        };
        // Admission is armed only by an explicit RAM cap, so the default path
        // never refuses a request.
        let admission = lisa_engine::memory::ram_cap()
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
    pub fn step(
        &mut self,
        tower: &mut dyn LanguageModel<crate::application::LanguageRuntime>,
        tok: &Tokenizer,
    ) -> bool {
        self.step_with_context(tower, tok, |_| {})
    }

    /// Restore a host's model context immediately before a ready wave runs.
    /// An idle poll leaves model-owned context and draft caches untouched.
    pub fn step_with_context(
        &mut self,
        tower: &mut dyn LanguageModel<crate::application::LanguageRuntime>,
        tok: &Tokenizer,
        before_wave: impl FnOnce(&mut dyn LanguageModel<crate::application::LanguageRuntime>),
    ) -> bool {
        let Some(first) = wait::receive_first(&self.jobs, None) else {
            return false;
        };
        self.step_first(tower, tok, first, before_wave)
    }

    fn wait_step(
        &mut self,
        tower: &mut dyn LanguageModel<crate::application::LanguageRuntime>,
        tok: &Tokenizer,
    ) -> bool {
        let Some(first) = wait::receive_first(&self.jobs, Some(wait::STOP_CHECK)) else {
            return false;
        };
        self.step_first(tower, tok, first, |_| {})
    }

    fn step_first(
        &mut self,
        tower: &mut dyn LanguageModel<crate::application::LanguageRuntime>,
        tok: &Tokenizer,
        first: Job,
        before_wave: impl FnOnce(&mut dyn LanguageModel<crate::application::LanguageRuntime>),
    ) -> bool {
        let mut wave = wait::ready_wave(first, &self.jobs, self.cfg.max_batch);

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

        // Byte-level admission at wave intake, armed by an explicit RAM cap.
        // A request that cannot fit the budget is refused
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
                    count_tokens_job(tok, job, chat_tmpl, &cfg)
                } else if job.surface == Surface::Embeddings {
                    run_embeddings(&self.cfg.model_id, tower, tok, job)
                } else {
                    complete(tower, tok, job, &cfg, &mut self.prefix, chat_tmpl)
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
    tower: &mut dyn LanguageModel<crate::application::LanguageRuntime>,
    tok: &Tokenizer,
    mut cfg: ServerConfig,
    stop: std::sync::Arc<AtomicBool>,
) -> anyhow::Result<()> {
    cfg.supports_images = tower.supports_images();
    lisa_engine::generation::set_eos_ids_for_model(tok.im_end_ids.clone(), tower);
    let mut server = Server::start(cfg, stop.clone(), tower.runtime().context.device().clone())?;
    println!("lisa serve listening on http://{}", server.local_addr());
    println!("  POST /v1/chat/completions   GET /v1/models   GET /health");
    println!("  max_batch {}", server.cfg.max_batch);
    if server.chat_tmpl.is_some() {
        println!("  chat_template.jinja loaded");
    }
    while !stop.load(Ordering::Relaxed) {
        server.wait_step(tower, tok);
    }
    server.shutdown();
    Ok(())
}

// ─────────────────────── OpenAI Responses ───────────────────────

// ─────────────────────── token counting ───────────────────────

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

#[cfg(test)]
#[path = "tests.rs"]
mod render_test;

#[cfg(test)]
#[path = "sampling_tests.rs"]
mod sampling_tests;

#[path = "inkling2.rs"]
mod inkling2;

#[path = "olala.rs"]
mod olala;
