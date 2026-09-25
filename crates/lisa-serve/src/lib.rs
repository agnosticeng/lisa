//! Minimal OpenAI-compatible serving layer.
//!
//! No async runtime: a `TcpListener` accept thread parses each HTTP request and
//! hands it to the worker (the thread that owns the [`Tower`] — MLX arrays are
//! thread-affine) over an `mpsc` channel. The worker runs one completion at a
//! time and streams tokens back to the connection thread, which frames them as
//! SSE (chunked) or collects them into a single JSON response.
//!
//! Requests are stateless: each one prefills its own prompt from scratch. (A
//! `session_id`-keyed cache would reuse prefixes, but the scheduler/batch
//! machinery for that is a separate step.)

use std::collections::HashMap;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::{json, Value};

use lisa_engine::core::generate::is_eos;
use lisa_engine::models::LanguageModel;
use lisa_engine::DecisionModel;
use lisa_engine::core::sampler::Sampler;
use lisa_engine::core::session::Session;
use lisa_engine::core::tokenizer::{self, Tokenizer};

pub struct ServerConfig {
    pub addr: String,
    pub max_tokens: usize,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
    pub min_p: f32,
    pub rep_penalty: f32,
    pub depth: usize,
    /// Maximum streams stepped together (admission queue width).
    pub max_batch: usize,
    /// The checkpoint's `chat_template.jinja`, when available.
    pub chat_template: Option<String>,
}

enum Reply {
    /// Sent once at the start of a streamed reply, carrying the prompt size.
    Start { prompt_tokens: usize },
    /// A streamed text delta, optionally carrying a per-token logprob entry.
    Delta {
        text: String,
        logprobs: Option<Value>,
    },
    Done {
        content: String,
        reasoning: Option<String>,
        tool_calls: Vec<Value>,
        prompt_tokens: usize,
        completion_tokens: usize,
        finish: String,
        logprobs: Option<Value>,
        stop_sequence: Option<String>,
    },
    Error(String),
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
    chat_tmpl: Option<ChatTemplate>,
    stop: std::sync::Arc<AtomicBool>,
    /// Held so the bound port stays open until [`Server::shutdown`] drops it.
    listener: Option<std::sync::Arc<TcpListener>>,
    accept: Option<std::thread::JoinHandle<()>>,
    addr: std::net::SocketAddr,
}

impl Server {
    pub fn start(cfg: ServerConfig, stop: std::sync::Arc<AtomicBool>) -> anyhow::Result<Server> {
        let listener = TcpListener::bind(&cfg.addr)
            .map_err(|e| anyhow::anyhow!("bind {}: {e}", cfg.addr))?;
        listener.set_nonblocking(true)?;
        let addr = listener.local_addr()?;
        let chat_tmpl = cfg
            .chat_template
            .as_deref()
            .map(ChatTemplate::new)
            .transpose()?;

        let (tx, rx) = mpsc::channel::<Job>();
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
                            std::thread::spawn(move || {
                                if let Err(e) = handle_conn(stream, tx, &cfg) {
                                    eprintln!("[serve] connection error: {e}");
                                }
                            });
                        }
                        Err(_) => std::thread::sleep(std::time::Duration::from_millis(50)),
                    }
                }
            })
        };
        Ok(Server {
            cfg,
            jobs: rx,
            sessions: HashMap::new(),
            chat_tmpl,
            stop,
            listener: Some(listener),
            accept: Some(accept),
            addr,
        })
    }

    pub fn local_addr(&self) -> std::net::SocketAddr {
        self.addr
    }

    /// Process one ready wave of jobs, if any. Returns `true` when it did work
    /// (so a host loop can prefer its own commands between waves).
    pub fn step(&mut self, tower: &mut dyn LanguageModel, tok: &Tokenizer) -> bool {
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
            if let Err(e) = run_session(tower, tok, &mut self.sessions, &job, &cfg, chat_tmpl) {
                let _ = job.reply.send(Reply::Error(e.to_string()));
            }
        }
        if !batch_jobs.is_empty() {
            let (spec, plain): (Vec<Job>, Vec<Job>) = batch_jobs
                .into_iter()
                .partition(|j| depth_for(j, &cfg) > 0 || has_tools(j) || j.surface != Surface::Chat);
            for job in spec {
                let r = if job.surface == Surface::CountTokens {
                    count_tokens_job(tok, &job, chat_tmpl)
                } else {
                    complete(tower, tok, &job, &cfg, chat_tmpl)
                };
                if let Err(e) = r {
                    let _ = job.reply.send(Reply::Error(e.to_string()));
                }
            }
            if !plain.is_empty() {
                if plain.len() == 1 {
                    if let Err(e) = complete(tower, tok, &plain[0], &cfg, chat_tmpl) {
                        let _ = plain[0].reply.send(Reply::Error(e.to_string()));
                    }
                } else if let Err(e) = run_wave(tower, tok, plain, &cfg, chat_tmpl) {
                    eprintln!("[serve] wave error: {e}");
                }
            }
        }
        if std::env::var("LISA_PROFILE").is_ok() {
            lisa_engine::core::mem::mlx_mem_line("serve-wave");
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
    cfg: ServerConfig,
    stop: std::sync::Arc<AtomicBool>,
) -> anyhow::Result<()> {
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
fn build_request(id: usize, tok: &Tokenizer, job: &Job, cfg: &ServerConfig, tmpl: Option<&ChatTemplate>) -> anyhow::Result<lisa_engine::core::sched::Request> {
    let messages = parse_messages(&job.body)?;
    let prompt = render_prompt(tmpl, &messages, true, &request_tools(job), true)?;
    let ids = tok.encode(&prompt, false)?;
    let max_tokens = request_max_tokens(job, cfg);
    Ok(lisa_engine::core::sched::Request {
        id,
        prompt: ids,
        max_tokens,
        sampler: sampler_from(&job.body, cfg),
    })
}

/// Serve a wave of stateless requests through the continuous-batch scheduler,
/// streaming each request's tokens back to its own connection.
fn run_wave(tower: &mut dyn LanguageModel, tok: &Tokenizer, jobs: Vec<Job>, cfg: &ServerConfig, tmpl: Option<&ChatTemplate>) -> anyhow::Result<()> {
    let mut job_of: Vec<usize> = Vec::new();
    let mut reqs: Vec<lisa_engine::core::sched::Request> = Vec::new();
    for (ji, job) in jobs.iter().enumerate() {
        match build_request(reqs.len(), tok, job, cfg, tmpl) {
            Ok(r) => {
                job_of.push(ji);
                reqs.push(r);
            }
            Err(e) => {
                let _ = job.reply.send(Reply::Error(e.to_string()));
            }
        }
    }
    if reqs.is_empty() {
        return Ok(());
    }
    let n = reqs.len();
    let plens: Vec<usize> = reqs.iter().map(|r| r.prompt.len()).collect();
    let mut acc: Vec<Vec<u32>> = vec![Vec::new(); n];
    let mut stopped: Vec<bool> = vec![false; n];
    let mut last: Vec<String> = vec![String::new(); n];
    let stream_flags: Vec<bool> = job_of.iter().map(|&ji| jobs[ji].stream).collect();
    let replies: Vec<Sender<Reply>> = job_of.iter().map(|&ji| jobs[ji].reply.clone()).collect();

    lisa_engine::core::sched::run(tower, reqs, cfg.max_batch, 0, &mut |id, t| {
        if is_eos(t) {
            stopped[id] = true;
            return Ok(());
        }
        acc[id].push(t);
        if stream_flags[id] {
            let text = tok.decode(&acc[id])?;
            if text.len() > last[id].len() {
                let delta = text[last[id].len()..].to_string();
                let _ = replies[id].send(Reply::Delta { text: delta, logprobs: None });
            }
            last[id] = text;
        }
        Ok(())
    })?;

    for id in 0..n {
        let content = if stream_flags[id] {
            std::mem::take(&mut last[id])
        } else {
            tok.decode(&acc[id])?
        };
        let _ = replies[id].send(Reply::Done {
            content,
            reasoning: None,
            tool_calls: Vec::new(),
            logprobs: None,
            stop_sequence: None,
            prompt_tokens: plens[id],
            completion_tokens: acc[id].len(),
            finish: if stopped[id] { "stop".into() } else { "length".into() },
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
    let new_str = render_prompt(tmpl, &messages, true, &tools, true)?;

    let (mut sess, mut conversation) = sessions.remove(&sid).unwrap_or_else(|| {
        let mut s = Session::new(tower);
        s.enable_snapshots();
        (s, Vec::new())
    });
    sess.capture_snapshot(conversation.len(), tower.drafter_offset());

    // Structural match against the tracked conversation. Raw token matching
    // fails (re-encoding BPE-merges the generation prompt's newline into the
    // assistant content); message-level matching picks the right turn boundary.
    let k = structural_prefix(&messages, &conversation).min(conversation.len());
    let diverges = k < conversation.len();

    // String-suffix path (canonical template only): feed only the messages
    // after the shared prefix. On divergence, rewind the recurrent state to the
    // snapshot taken at that message boundary first.
    let string_suffix: Option<Vec<u32>> = if tmpl.is_some() && k > 0 {
        if diverges && sess.restore_snapshot(tower, k).is_none() {
            let mut s = Session::new(tower);
            s.enable_snapshots();
            sess = s;
        }
        render_prompt(tmpl, &messages[..k], false, &tools, true)
            .ok()
            .and_then(|prefix| new_str.strip_prefix(&prefix).map(|r| r.to_string()))
            .filter(|r| !r.is_empty())
            // The assistant turn's closing newline lives in the stripped
            // prefix but not in the cached tokens; carry it over.
            .map(|rest| tok.encode(&format!("\n{rest}"), false))
            .transpose()?
    } else {
        None
    };

    let suffix: Vec<u32> = match string_suffix {
        Some(sfx) if !sfx.is_empty() => {
            eprintln!(
                "[serve] session {sid}: {} reuse, {} prefilled",
                if diverges { "snapshot" } else { "string-suffix" },
                sfx.len()
            );
            sfx
        }
        _ => {
            let ids = tok.encode(&new_str, false)?;
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

    let max_tokens = request_max_tokens(job, cfg);
    let mut sampler = sampler_from(&job.body, cfg);
    let greedy = sampler.greedy();
    let streaming = job.stream;
    if streaming {
        let _ = job.reply.send(Reply::Start {
            prompt_tokens: suffix.len(),
        });
    }
    let reply = job.reply.clone();

    let stops = request_stops(job);
    let parallel = parallel_tools(job);
    let mut acc: Vec<u32> = Vec::new();
    let mut last = String::new();
    let hit_stop = std::cell::Cell::new(false);
    let mut emit = |t: u32| -> anyhow::Result<()> {
        if is_eos(t) {
            return Ok(());
        }
        acc.push(t);
        if !stops.is_empty() {
            let text = tok.decode(&acc)?;
            if stops.iter().any(|s| !s.is_empty() && text.contains(s.as_str())) {
                hit_stop.set(true);
                return Err(anyhow::anyhow!("stop sequence"));
            }
        }
        if streaming {
            let text = tok.decode(&acc)?;
            let ans = streaming_answer(&text);
            let shown = match tool_call_start(ans) {
                Some(i) => ans[..i].to_string(),
                None => ans.to_string(),
            };
            if let Some(delta) = stream_delta(&last, &shown) {
                let _ = reply.send(Reply::Delta { text: delta, logprobs: None });
            }
            last = shown;
        }
        Ok(())
    };
    let generated = if depth > 0 && greedy {
        sess.generate_mtp(tower, &suffix, max_tokens, depth, &mut emit)
    } else {
        sess.generate(tower, &suffix, max_tokens, &mut sampler, &mut emit)
    };
    let generated = match generated {
        Ok(g) => g,
        Err(_) if hit_stop.get() => Vec::new(),
        Err(e) => return Err(e),
    };
    let stopped = generated.last().map(|&t| is_eos(t)).unwrap_or(false);
    let full = tok.decode(&acc)?;
    let (content, reasoning, tool_calls) = shape_reply(&full, &tools, parallel, &stops);

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
    sessions.insert(sid, (sess, conversation));
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

fn handle_conn(mut stream: TcpStream, tx: Sender<Job>, cfg: &ServerConfig) -> anyhow::Result<()> {
    let req = match read_request(&mut stream) {
        Ok(Some(r)) => r,
        Ok(None) => return Ok(()),
        Err(e) => {
            let payload = json!({"error": {"message": e.to_string(), "type": "invalid_request_error"}});
            let _ = write_response(&mut stream, "400 Bad Request", "application/json", payload.to_string().as_bytes(), "");
            return Ok(());
        }
    };
    let (method, path, body) = req;
    if method == "GET" && (path == "/health" || path == "/healthz") {
        let payload = json!({"status": "ok"});
        return write_response(&mut stream, "200 OK", "application/json", payload.to_string().as_bytes(), "");
    }
    if method == "GET" && path.starts_with("/v1/models") {
        let payload = json!({
            "object": "list",
            "data": [{ "id": MODEL_ID, "object": "model", "owned_by": "lisa" }]
        });
        return write_response(&mut stream, "200 OK", "application/json", payload.to_string().as_bytes(), "");
    }
    if method == "POST" && (path == "/v1/chat/completions" || path == "/v1/completions") {
        let surface = if path == "/v1/completions" {
            Surface::Completions
        } else {
            Surface::Chat
        };
        return dispatch_or_400(&mut stream, &body, tx, cfg, surface);
    }
    if method == "POST" && path == "/v1/messages" {
        return dispatch_or_400(&mut stream, &body, tx, cfg, Surface::Messages);
    }
    if method == "POST" && path == "/v1/messages/count_tokens" {
        return dispatch_or_400(&mut stream, &body, tx, cfg, Surface::CountTokens);
    }
    if method == "POST" && path == "/v1/responses" {
        return dispatch_or_400(&mut stream, &body, tx, cfg, Surface::Responses);
    }
    if method == "GET" && path.starts_with("/v1/responses/") {
        let id = path.trim_start_matches("/v1/responses/").to_string();
        let payload = stored_response(&id)
            .unwrap_or_else(|| json!({"error": {"message": "response not found", "type": "invalid_request_error"}}));
        return write_response(
            &mut stream,
            "200 OK",
            "application/json",
            payload.to_string().as_bytes(),
            "",
        );
    }

    write_response(&mut stream, "404 Not Found", "application/json", br#"{"error":{"message":"not found"}}"#, "")
}

const MODEL_ID: &str = "qwen3.8-flash-next";

/// Rate/quota headers advertised on every response (llmprobe's frontier
/// rate-limit check, and honest for a single-stream server).
fn rate_headers() -> String {
    static SERVED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    const LIMIT: u64 = 100_000;
    let n = SERVED.fetch_add(1, Ordering::Relaxed);
    let remaining = LIMIT.saturating_sub(n + 1);
    format!(
        "X-RateLimit-Limit-Requests: {LIMIT}\r\n\
         X-RateLimit-Remaining-Requests: {remaining}\r\n\
         X-RateLimit-Reset-Requests: 1s"
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
    stream: &mut TcpStream,
    body: &[u8],
    tx: Sender<Job>,
    cfg: &ServerConfig,
    surface: Surface,
) -> anyhow::Result<()> {
    if let Err(e) = dispatch_completion(stream, body, tx, cfg, surface) {
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
            "",
        );
    }
    Ok(())
}

fn dispatch_completion(
    stream: &mut TcpStream,
    body: &[u8],
    tx: Sender<Job>,
    _cfg: &ServerConfig,
    surface: Surface,
) -> anyhow::Result<()> {
    let mut body: Value = serde_json::from_slice(body)
        .map_err(|e| anyhow::anyhow!("invalid JSON body: {e}"))?;
    match surface {
        Surface::Completions => normalize_completion_prompt(&mut body),
        Surface::Messages | Surface::CountTokens => normalize_messages_body(&mut body),
        Surface::Responses => normalize_responses_body(&mut body),
        Surface::Chat => {}
    }
    if surface == Surface::Responses {
        apply_previous_response(&mut body);
    }
    if surface == Surface::Messages && body.get("max_tokens").is_none() {
        // Anthropic's spec makes `max_tokens` required.
        anyhow::bail!("max_tokens is required");
    }
    if surface != Surface::CountTokens {
        // Validate the request shape up front so client errors are 400s, not 500s.
        parse_messages(&body)?;
    }
    // `background: true` runs the job off the request (llmprobe checks the
    // immediate status is queued/in_progress).
    if surface == Surface::Responses && body.get("background").and_then(|v| v.as_bool()).unwrap_or(false) {
        let id = response_id();
        let (rtx, rrx) = mpsc::channel::<Reply>();
        if tx
            .send(Job {
                body,
                stream: false,
                session_id: None,
                surface,
                reply: rtx,
            })
            .is_err()
        {
            anyhow::bail!("engine worker is gone");
        }
        let key = id.clone();
        std::thread::spawn(move || {
            if let Ok(d) = drain_reply(rrx) {
                store_response(&key, responses_payload_with_id(&d, &key));
            }
        });
        let payload = json!({
            "id": id,
            "object": "response",
            "created_at": now(),
            "status": "queued",
            "model": MODEL_ID,
            "output": [],
            "output_text": "",
            "usage": Value::Null,
        });
        return write_response(
            stream,
            "200 OK",
            "application/json",
            payload.to_string().as_bytes(),
            "",
        );
    }
    let stream_mode = body.get("stream").and_then(|v| v.as_bool()).unwrap_or(false);
    let include_usage = body
        .get("stream_options")
        .and_then(|v| v.get("include_usage"))
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let session_id = body
        .get("session_id")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());
    let n = body.get("n").and_then(|v| v.as_u64()).unwrap_or(1).max(1) as usize;

    if n > 1 && !stream_mode && (surface == Surface::Chat || surface == Surface::Completions) {
        return collect_choices(stream, &body, tx, n, surface);
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
        stream_surface(stream, rrx, include_usage, surface, &body)
    } else {
        collect_surface(stream, rrx, surface, &body)
    }
}

/// A fully-collected engine reply.
struct ReplyData {
    content: String,
    reasoning: Option<String>,
    tool_calls: Vec<Value>,
    prompt_tokens: usize,
    completion_tokens: usize,
    finish: String,
    logprobs: Option<Value>,
    stop_sequence: Option<String>,
}

fn drain_reply(rx: Receiver<Reply>) -> anyhow::Result<ReplyData> {
    let mut content = String::new();
    loop {
        match rx.recv() {
            Ok(Reply::Start { .. }) => {}
            Ok(Reply::Delta { text, .. }) => content.push_str(&text),
            Ok(Reply::Done {
                content: c,
                reasoning,
                tool_calls,
                prompt_tokens,
                completion_tokens,
                finish,
                logprobs,
                stop_sequence,
            }) => {
                return Ok(ReplyData {
                    content: c,
                    reasoning,
                    tool_calls,
                    prompt_tokens,
                    completion_tokens,
                    finish,
                    logprobs,
                    stop_sequence,
                });
            }
            Ok(Reply::Error(e)) => anyhow::bail!("{e}"),
            Err(_) => anyhow::bail!("engine worker dropped the request"),
        }
    }
}

/// Log-softmax probability of `token` under `logits` (last axis = vocab).
fn token_logprob(logits: &lisa_mlx::Array, token: u32) -> anyhow::Result<f32> {
    let v = logits
        .as_dtype(lisa_mlx::Dtype::Float32)?
        .to_vec1::<f32>()?;
    let m = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let sum: f32 = v.iter().map(|x| (x - m).exp()).sum();
    let lse = m + sum.ln();
    let idx = token as usize;
    Ok(if idx < v.len() { v[idx] - lse } else { 0.0 })
}

fn usage_json(prompt_tokens: usize, completion_tokens: usize) -> Value {
    json!({
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    })
}

fn chat_message(d: &ReplyData) -> Value {
    let mut message = json!({ "role": "assistant", "content": d.content });
    if let Some(r) = &d.reasoning {
        message["reasoning_content"] = json!(r);
    }
    if !d.tool_calls.is_empty() {
        message["tool_calls"] = json!(d.tool_calls);
    }
    message
}

fn chat_choice(d: &ReplyData, index: usize) -> Value {
    let mut choice = json!({
        "index": index,
        "message": chat_message(d),
        "finish_reason": d.finish,
    });
    if let Some(lp) = &d.logprobs {
        choice["logprobs"] = lp.clone();
    }
    choice
}

fn completion_choice(d: &ReplyData, index: usize) -> Value {
    json!({
        "index": index,
        "text": d.content,
        "logprobs": d.logprobs,
        "finish_reason": d.finish,
    })
}

fn single_payload(d: &ReplyData, surface: Surface) -> Value {
    let object = if surface == Surface::Completions {
        "text_completion"
    } else {
        "chat.completion"
    };
    let choice = if surface == Surface::Completions {
        completion_choice(d, 0)
    } else {
        chat_choice(d, 0)
    };
    json!({
        "id": completion_id(),
        "object": object,
        "created": now(),
        "model": MODEL_ID,
        "choices": [choice],
        "usage": usage_json(d.prompt_tokens, d.completion_tokens),
    })
}

fn collect_surface(
    stream: &mut TcpStream,
    rx: Receiver<Reply>,
    surface: Surface,
    req_body: &Value,
) -> anyhow::Result<()> {
    let d = match drain_reply(rx) {
        Ok(d) => d,
        Err(e) => {
            let payload = json!({"error": {"message": e.to_string(), "type": "engine_error"}});
            return write_response(
                stream,
                "500 Internal Server Error",
                "application/json",
                payload.to_string().as_bytes(),
                &rate_headers(),
            );
        }
    };
    let payload = match surface {
        Surface::Messages => messages_payload(&d),
        Surface::Responses => {
            let id = response_id();
            let p = responses_payload_with_id(&d, &id);
            store_conversation(&id, req_body, &d, &p);
            p
        }
        _ => single_payload(&d, surface),
    };
    write_response(
        stream,
        "200 OK",
        "application/json",
        payload.to_string().as_bytes(),
        &rate_headers(),
    )
}

/// Anthropic Messages response body.
fn messages_payload(d: &ReplyData) -> Value {
    let mut content: Vec<Value> = Vec::new();
    if let Some(r) = &d.reasoning {
        if !r.is_empty() {
            content.push(json!({ "type": "thinking", "thinking": r, "signature": "" }));
        }
    }
    if !d.content.is_empty() {
        content.push(json!({ "type": "text", "text": d.content }));
    }
    for c in &d.tool_calls {
        let name = c
            .get("function")
            .and_then(|f| f.get("name"))
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let args = c
            .get("function")
            .and_then(|f| f.get("arguments"))
            .map(|a| match a {
                Value::String(s) => serde_json::from_str::<Value>(s).unwrap_or(json!({})),
                other => other.clone(),
            })
            .unwrap_or(json!({}));
        content.push(json!({
            "type": "tool_use",
            "id": c.get("id").cloned().unwrap_or(json!("")),
            "name": name,
            "input": args,
        }));
    }
    let (stop_reason, stop_sequence) = if let Some(sq) = &d.stop_sequence {
        ("stop_sequence", json!(sq))
    } else {
        (
            match d.finish.as_str() {
                "tool_calls" => "tool_use",
                "length" => "max_tokens",
                _ => "end_turn",
            },
            Value::Null,
        )
    };
    json!({
        "id": format!("msg_{:x}{:x}", now(), SEQ.fetch_add(1, Ordering::Relaxed)),
        "type": "message",
        "role": "assistant",
        "model": MODEL_ID,
        "content": content,
        "stop_reason": stop_reason,
        "stop_sequence": stop_sequence,
        "usage": {
            "input_tokens": d.prompt_tokens,
            "output_tokens": d.completion_tokens,
        },
    })
}

/// OpenAI Responses resource.
fn empty_response_resource(id: &str, status: &str) -> Value {
    json!({
        "id": id,
        "object": "response",
        "created_at": now(),
        "completed_at": Value::Null,
        "status": status,
        "incomplete_details": Value::Null,
        "model": MODEL_ID,
        "previous_response_id": Value::Null,
        "instructions": Value::Null,
        "output": [],
        "error": Value::Null,
        "tools": [],
        "tool_choice": "auto",
        "truncation": "disabled",
        "parallel_tool_calls": true,
        "text": { "format": { "type": "text" } },
        "top_p": 1.0,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
        "top_logprobs": 0,
        "temperature": 1.0,
        "reasoning": Value::Null,
        "usage": Value::Null,
        "max_output_tokens": Value::Null,
        "max_tool_calls": Value::Null,
        "store": true,
        "background": false,
        "service_tier": "default",
        "metadata": {},
        "safety_identifier": Value::Null,
        "prompt_cache_key": Value::Null,
    })
}

fn responses_payload_with_id(d: &ReplyData, id: &str) -> Value {
    let mut output: Vec<Value> = Vec::new();
    if let Some(r) = &d.reasoning {
        if !r.is_empty() {
            output.push(json!({
                "type": "reasoning",
                "id": format!("rs_{:x}", now()),
                "summary": [{ "type": "summary_text", "text": r }],
            }));
        }
    }
    if !d.content.is_empty() {
        output.push(json!({
            "type": "message",
            "id": format!("msg_{:x}", now()),
            "status": "completed",
            "role": "assistant",
            "content": [{ "type": "output_text", "text": d.content, "annotations": [] }],
        }));
    }
    for c in &d.tool_calls {
        output.push(json!({
            "type": "function_call",
            "id": format!("fc_{:x}", now()),
            "call_id": c.get("id").cloned().unwrap_or(json!("")),
            "name": c.get("function").and_then(|f| f.get("name")).cloned().unwrap_or(json!("")),
            "arguments": c.get("function").and_then(|f| f.get("arguments")).cloned().unwrap_or(json!("")),
            "status": "completed",
        }));
    }
    let incomplete = d.finish == "length";
    let mut resp = empty_response_resource(id, if incomplete { "incomplete" } else { "completed" });
    resp["completed_at"] = if incomplete { Value::Null } else { json!(now()) };
    resp["output"] = Value::Array(output);
    resp["output_text"] = json!(d.content);
    resp["usage"] = json!({
        "input_tokens": d.prompt_tokens,
        "output_tokens": d.completion_tokens,
        "total_tokens": d.prompt_tokens + d.completion_tokens,
        "input_tokens_details": { "cached_tokens": 0 },
        "output_tokens_details": { "reasoning_tokens": 0 },
    });
    if incomplete {
        resp["incomplete_details"] = json!({ "reason": "max_output_tokens" });
    }
    resp
}

/// `n > 1`: run `n` independent generations (distinct seeds) and merge them into
/// one choices array.
fn collect_choices(
    stream: &mut TcpStream,
    body: &Value,
    tx: Sender<Job>,
    n: usize,
    surface: Surface,
) -> anyhow::Result<()> {
    let base = body
        .get("seed")
        .and_then(|v| v.as_u64())
        .filter(|&v| v != 0)
        .unwrap_or(0x9E37_79B9_7F4A_7C15);
    let mut choices = Vec::with_capacity(n);
    let mut prompt_tokens = 0;
    let mut completion_tokens = 0;
    for i in 0..n {
        let mut b = body.clone();
        b["seed"] = json!(base.wrapping_add(i as u64));
        let (rtx, rrx) = mpsc::channel::<Reply>();
        tx.send(Job {
            body: b,
            stream: false,
            session_id: None,
            surface,
            reply: rtx,
        })
        .map_err(|_| anyhow::anyhow!("engine worker is gone"))?;
        let d = match drain_reply(rrx) {
            Ok(d) => d,
            Err(e) => {
                let payload = json!({"error": {"message": e.to_string(), "type": "engine_error"}});
                return write_response(
                    stream,
                    "500 Internal Server Error",
                    "application/json",
                    payload.to_string().as_bytes(),
                    &rate_headers(),
                );
            }
        };
        prompt_tokens = d.prompt_tokens;
        completion_tokens += d.completion_tokens;
        choices.push(if surface == Surface::Completions {
            completion_choice(&d, i)
        } else {
            chat_choice(&d, i)
        });
    }
    let object = if surface == Surface::Completions {
        "text_completion"
    } else {
        "chat.completion"
    };
    let payload = json!({
        "id": completion_id(),
        "object": object,
        "created": now(),
        "model": MODEL_ID,
        "choices": choices,
        "usage": usage_json(prompt_tokens, completion_tokens),
    });
    write_response(
        stream,
        "200 OK",
        "application/json",
        payload.to_string().as_bytes(),
        &rate_headers(),
    )
}

fn stream_surface(
    stream: &mut TcpStream,
    rx: Receiver<Reply>,
    include_usage: bool,
    surface: Surface,
    req_body: &Value,
) -> anyhow::Result<()> {
    let mut head = String::from(
        "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\nTransfer-Encoding: chunked\r\n",
    );
    head.push_str(&rate_headers());
    head.push_str("\r\n\r\n");
    stream.write_all(head.as_bytes())?;
    match surface {
        Surface::Completions => stream_completions(stream, rx, include_usage),
        Surface::Messages => stream_messages(stream, rx),
        Surface::Responses => stream_responses(stream, rx, req_body),
        _ => stream_chat(stream, rx, include_usage),
    }
}

fn stream_chunk(object: &str, id: &str, choice: Value) -> Value {
    json!({
        "id": id,
        "object": object,
        "created": now(),
        "model": MODEL_ID,
        "choices": [choice],
    })
}

fn stream_chat(stream: &mut TcpStream, rx: Receiver<Reply>, include_usage: bool) -> anyhow::Result<()> {
    let id = completion_id();
    let chunk = |delta: Value, finish: Value, logprobs: Option<Value>| {
        let mut choice = json!({ "index": 0, "delta": delta, "finish_reason": finish });
        if let Some(lp) = logprobs {
            choice["logprobs"] = lp;
        }
        stream_chunk("chat.completion.chunk", &id, choice)
    };
    write_sse(stream, &chunk(json!({"role": "assistant"}), Value::Null, None))?;

    for reply in rx {
        match reply {
            Reply::Start { .. } => {}
            Reply::Delta { text, logprobs } => {
                write_sse(stream, &chunk(json!({"content": text}), Value::Null, logprobs))?
            }
            Reply::Done {
                finish,
                tool_calls,
                prompt_tokens,
                completion_tokens,
                ..
            } => {
                if !tool_calls.is_empty() {
                    let deltas: Vec<Value> = tool_calls
                        .iter()
                        .enumerate()
                        .map(|(i, c)| {
                            json!({
                                "index": i,
                                "id": c.get("id"),
                                "type": "function",
                                "function": c.get("function"),
                            })
                        })
                        .collect();
                    write_sse(stream, &chunk(json!({"tool_calls": deltas}), Value::Null, None))?;
                }
                write_sse(stream, &chunk(json!({}), json!(finish), None))?;
                if include_usage {
                    let usage = json!({
                        "id": id,
                        "object": "chat.completion.chunk",
                        "created": now(),
                        "model": MODEL_ID,
                        "choices": [],
                        "usage": usage_json(prompt_tokens, completion_tokens),
                    });
                    write_sse(stream, &usage)?;
                }
                write_chunk(stream, b"data: [DONE]\n\n")?;
                write_chunk(stream, b"")?;
                break;
            }
            Reply::Error(e) => {
                write_sse(stream, &json!({"error": {"message": e}}))?;
                write_chunk(stream, b"")?;
                break;
            }
        }
    }
    stream.flush().ok();
    Ok(())
}

fn stream_completions(stream: &mut TcpStream, rx: Receiver<Reply>, include_usage: bool) -> anyhow::Result<()> {
    let id = completion_id();
    for reply in rx {
        match reply {
            Reply::Start { .. } => {}
            Reply::Delta { text, .. } => {
                write_sse(
                    stream,
                    &stream_chunk(
                        "text_completion",
                        &id,
                        json!({ "index": 0, "text": text, "finish_reason": Value::Null }),
                    ),
                )?;
            }
            Reply::Done {
                finish,
                prompt_tokens,
                completion_tokens,
                ..
            } => {
                write_sse(
                    stream,
                    &stream_chunk(
                        "text_completion",
                        &id,
                        json!({ "index": 0, "text": "", "finish_reason": finish }),
                    ),
                )?;
                if include_usage {
                    let usage = json!({
                        "id": id,
                        "object": "text_completion",
                        "created": now(),
                        "model": MODEL_ID,
                        "choices": [],
                        "usage": usage_json(prompt_tokens, completion_tokens),
                    });
                    write_sse(stream, &usage)?;
                }
                write_chunk(stream, b"data: [DONE]\n\n")?;
                write_chunk(stream, b"")?;
                break;
            }
            Reply::Error(e) => {
                write_sse(stream, &json!({"error": {"message": e}}))?;
                write_chunk(stream, b"")?;
                break;
            }
        }
    }
    stream.flush().ok();
    Ok(())
}

/// Write one SSE event (`event:` line optional) with a JSON `data:` payload.
fn sse_event(stream: &mut TcpStream, event: Option<&str>, payload: &Value) -> std::io::Result<()> {
    let mut body = String::new();
    if let Some(e) = event {
        body.push_str("event: ");
        body.push_str(e);
        body.push('\n');
    }
    body.push_str("data: ");
    body.push_str(&payload.to_string());
    body.push_str("\n\n");
    write_chunk(stream, body.as_bytes())
}

fn stream_messages(stream: &mut TcpStream, rx: Receiver<Reply>) -> anyhow::Result<()> {
    let id = format!("msg_{:x}{:x}", now(), SEQ.fetch_add(1, Ordering::Relaxed));
    let mut started = false;
    let mut block_index = 0usize;
    let mut text_started = false;
    for reply in rx {
        match reply {
            Reply::Start { prompt_tokens } => {
                if !started {
                    started = true;
                    sse_event(
                        stream,
                        Some("message_start"),
                        &json!({
                            "type": "message_start",
                            "message": {
                                "id": id,
                                "type": "message",
                                "role": "assistant",
                                "model": MODEL_ID,
                                "content": [],
                                "stop_reason": Value::Null,
                                "stop_sequence": Value::Null,
                                "usage": { "input_tokens": prompt_tokens, "output_tokens": 0 },
                            }
                        }),
                    )?;
                }
            }
            Reply::Delta { text, .. } => {
                if text.is_empty() {
                    continue;
                }
                if !text_started {
                    sse_event(
                        stream,
                        Some("content_block_start"),
                        &json!({
                            "type": "content_block_start",
                            "index": block_index,
                            "content_block": { "type": "text", "text": "" },
                        }),
                    )?;
                    text_started = true;
                }
                sse_event(
                    stream,
                    Some("content_block_delta"),
                    &json!({
                        "type": "content_block_delta",
                        "index": block_index,
                        "delta": { "type": "text_delta", "text": text },
                    }),
                )?;
            }
            Reply::Done {
                finish,
                tool_calls,
                completion_tokens,
                ..
            } => {
                if text_started {
                    sse_event(
                        stream,
                        Some("content_block_stop"),
                        &json!({ "type": "content_block_stop", "index": block_index }),
                    )?;
                    block_index += 1;
                }
                for c in &tool_calls {
                    let cid = c.get("id").cloned().unwrap_or(json!(""));
                    let name = c
                        .get("function")
                        .and_then(|f| f.get("name"))
                        .cloned()
                        .unwrap_or(json!(""));
                    let args = c
                        .get("function")
                        .and_then(|f| f.get("arguments"))
                        .and_then(|a| a.as_str())
                        .unwrap_or("{}")
                        .to_string();
                    sse_event(
                        stream,
                        Some("content_block_start"),
                        &json!({
                            "type": "content_block_start",
                            "index": block_index,
                            "content_block": {
                                "type": "tool_use", "id": cid, "name": name, "input": {}
                            },
                        }),
                    )?;
                    sse_event(
                        stream,
                        Some("content_block_delta"),
                        &json!({
                            "type": "content_block_delta",
                            "index": block_index,
                            "delta": { "type": "input_json_delta", "partial_json": args },
                        }),
                    )?;
                    sse_event(
                        stream,
                        Some("content_block_stop"),
                        &json!({ "type": "content_block_stop", "index": block_index }),
                    )?;
                    block_index += 1;
                }
                let stop_reason = match finish.as_str() {
                    "tool_calls" => "tool_use",
                    "length" => "max_tokens",
                    _ => "end_turn",
                };
                sse_event(
                    stream,
                    Some("message_delta"),
                    &json!({
                        "type": "message_delta",
                        "delta": { "stop_reason": stop_reason, "stop_sequence": Value::Null },
                        "usage": { "output_tokens": completion_tokens },
                    }),
                )?;
                sse_event(stream, Some("message_stop"), &json!({ "type": "message_stop" }))?;
                write_chunk(stream, b"")?;
                break;
            }
            Reply::Error(e) => {
                sse_event(
                    stream,
                    Some("error"),
                    &json!({ "type": "error", "error": { "type": "engine_error", "message": e } }),
                )?;
                write_chunk(stream, b"")?;
                break;
            }
        }
    }
    stream.flush().ok();
    Ok(())
}

fn stream_responses(stream: &mut TcpStream, rx: Receiver<Reply>, req_body: &Value) -> anyhow::Result<()> {
    let id = response_id();
    let mut seq = 0u64;
    let mut next_seq = || {
        let s = seq;
        seq += 1;
        s
    };
    sse_event(
        stream,
        None,
        &json!({
            "type": "response.created",
            "sequence_number": next_seq(),
            "response": empty_response_resource(&id, "in_progress"),
        }),
    )?;
    let item_id = format!("msg_{:x}", now());
    let mut text_started = false;
    for reply in rx {
        match reply {
            Reply::Start { .. } => {}
            Reply::Delta { text, .. } => {
                if text.is_empty() {
                    continue;
                }
                if !text_started {
                    text_started = true;
                    sse_event(
                        stream,
                        None,
                        &json!({
                            "type": "response.output_item.added",
                            "sequence_number": next_seq(),
                            "output_index": 0,
                            "item": {
                                "type": "message", "id": item_id, "status": "in_progress",
                                "role": "assistant", "content": []
                            }
                        }),
                    )?;
                    sse_event(
                        stream,
                        None,
                        &json!({
                            "type": "response.content_part.added",
                            "sequence_number": next_seq(),
                            "item_id": item_id, "output_index": 0, "content_index": 0,
                            "part": { "type": "output_text", "text": "", "annotations": [] }
                        }),
                    )?;
                }
                sse_event(
                    stream,
                    None,
                    &json!({
                        "type": "response.output_text.delta",
                        "sequence_number": next_seq(),
                        "item_id": item_id, "output_index": 0, "content_index": 0,
                        "delta": text
                    }),
                )?;
            }
            Reply::Done {
                content,
                reasoning,
                tool_calls,
                prompt_tokens,
                completion_tokens,
                finish,
                logprobs,
                stop_sequence,
            } => {
                let d = ReplyData {
                    content: content.clone(),
                    reasoning,
                    tool_calls: tool_calls.clone(),
                    prompt_tokens,
                    completion_tokens,
                    finish: finish.clone(),
                    logprobs,
                    stop_sequence,
                };
                if text_started {
                    sse_event(
                        stream,
                        None,
                        &json!({
                            "type": "response.output_text.done",
                            "sequence_number": next_seq(),
                            "item_id": item_id, "output_index": 0, "content_index": 0,
                            "text": content
                        }),
                    )?;
                    sse_event(
                        stream,
                        None,
                        &json!({
                            "type": "response.content_part.done",
                            "sequence_number": next_seq(),
                            "item_id": item_id, "output_index": 0, "content_index": 0,
                            "part": { "type": "output_text", "text": content, "annotations": [] }
                        }),
                    )?;
                    sse_event(
                        stream,
                        None,
                        &json!({
                            "type": "response.output_item.done",
                            "sequence_number": next_seq(),
                            "output_index": 0,
                            "item": {
                                "type": "message", "id": item_id, "status": "completed",
                                "role": "assistant",
                                "content": [{ "type": "output_text", "text": content, "annotations": [] }]
                            }
                        }),
                    )?;
                }
                let base = if text_started { 1 } else { 0 };
                for (i, c) in tool_calls.iter().enumerate() {
                    let idx = base + i;
                    let call_id = c.get("id").cloned().unwrap_or(json!(""));
                    let name = c
                        .get("function")
                        .and_then(|f| f.get("name"))
                        .cloned()
                        .unwrap_or(json!(""));
                    let args = c
                        .get("function")
                        .and_then(|f| f.get("arguments"))
                        .and_then(|a| a.as_str())
                        .unwrap_or("{}")
                        .to_string();
                    let fc_id = format!("fc_{:x}", now());
                    sse_event(
                        stream,
                        None,
                        &json!({
                            "type": "response.output_item.added",
                            "sequence_number": next_seq(),
                            "output_index": idx,
                            "item": {
                                "type": "function_call", "id": fc_id, "call_id": call_id,
                                "name": name, "arguments": "", "status": "in_progress"
                            }
                        }),
                    )?;
                    sse_event(
                        stream,
                        None,
                        &json!({
                            "type": "response.function_call_arguments.delta",
                            "sequence_number": next_seq(),
                            "item_id": fc_id,
                            "output_index": idx,
                            "delta": args
                        }),
                    )?;
                    sse_event(
                        stream,
                        None,
                        &json!({
                            "type": "response.output_item.done",
                            "sequence_number": next_seq(),
                            "output_index": idx,
                            "item": {
                                "type": "function_call", "id": fc_id, "call_id": call_id,
                                "name": name, "arguments": args, "status": "completed"
                            }
                        }),
                    )?;
                }
                let payload = responses_payload_with_id(&d, &id);
                store_conversation(&id, req_body, &d, &payload);
                let terminal = if finish == "length" {
                    "response.incomplete"
                } else {
                    "response.completed"
                };
                sse_event(
                    stream,
                    None,
                    &json!({
                        "type": terminal,
                        "sequence_number": next_seq(),
                        "response": payload
                    }),
                )?;
                write_chunk(stream, b"data: [DONE]\n\n")?;
                write_chunk(stream, b"")?;
                break;
            }
            Reply::Error(e) => {
                sse_event(
                    stream,
                    None,
                    &json!({ "type": "error", "error": { "type": "engine_error", "message": e } }),
                )?;
                write_chunk(stream, b"")?;
                break;
            }
        }
    }
    stream.flush().ok();
    Ok(())
}

// ─────────────────────── Anthropic Messages ───────────────────────

/// Flatten an Anthropic `system` field (string or array of text blocks).
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
                    let mut only_tool_results = !blocks.is_empty();
                    for b in blocks {
                        let ty = b.get("type").and_then(|t| t.as_str()).unwrap_or("");
                        match ty {
                            "text" => {
                                only_tool_results = false;
                                text.push_str(b.get("text").and_then(|t| t.as_str()).unwrap_or(""));
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
                                let id = b
                                    .get("tool_use_id")
                                    .and_then(|v| v.as_str())
                                    .unwrap_or("");
                                let content = block_content_text(
                                    b.get("content").unwrap_or(&Value::Null),
                                );
                                out.push(json!({
                                    "role": "tool",
                                    "tool_call_id": id,
                                    "content": content,
                                }));
                            }
                            "image" => {
                                only_tool_results = false;
                                parts.push(json!({ "type": "image" }));
                            }
                            _ => {}
                        }
                    }
                    if !only_tool_results {
                        let mut msg = json!({ "role": role, "content": text });
                        if !parts.is_empty() {
                            let mut arr = vec![json!({ "type": "text", "text": text })];
                            arr.extend(parts.iter().cloned());
                            msg["content"] = Value::Array(arr);
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
        if tc.get("disable_parallel_tool_use").and_then(|v| v.as_bool()).unwrap_or(false) {
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
                        if let Some(parts) = item.get("content").and_then(|v| v.as_array()) {
                            for p in parts {
                                match p.get("type").and_then(|t| t.as_str()) {
                                    Some("input_text") | Some("output_text") => {
                                        text.push_str(p.get("text").and_then(|t| t.as_str()).unwrap_or(""));
                                    }
                                    Some("input_image") | Some("image") => image = true,
                                    _ => {}
                                }
                            }
                        }
                        let mut msg = json!({ "role": role, "content": text });
                        if image {
                            msg["content"] = json!([
                                { "type": "text", "text": text },
                                { "type": "image" }
                            ]);
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

fn store_conversation(id: &str, req_body: &Value, d: &ReplyData, payload: &Value) {
    let mut conv = req_body
        .get("messages")
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default();
    let mut assistant = json!({ "role": "assistant", "content": d.content });
    if let Some(r) = &d.reasoning {
        assistant["reasoning_content"] = json!(r);
    }
    if !d.tool_calls.is_empty() {
        assistant["tool_calls"] = json!(d.tool_calls);
    }
    conv.push(assistant);
    conv_store().lock().unwrap().insert(id.to_string(), conv);
    payload_store().lock().unwrap().insert(id.to_string(), payload.clone());
}

fn conv_store() -> &'static std::sync::Mutex<std::collections::HashMap<String, Vec<Value>>> {
    static S: std::sync::OnceLock<
        std::sync::Mutex<std::collections::HashMap<String, Vec<Value>>>,
    > = std::sync::OnceLock::new();
    S.get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()))
}

fn payload_store() -> &'static std::sync::Mutex<std::collections::HashMap<String, Value>> {
    static S: std::sync::OnceLock<std::sync::Mutex<std::collections::HashMap<String, Value>>> =
        std::sync::OnceLock::new();
    S.get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()))
}

fn store_response(id: &str, payload: Value) {
    payload_store()
        .lock()
        .unwrap()
        .insert(id.to_string(), payload);
}

fn stored_response(id: &str) -> Option<Value> {
    payload_store().lock().unwrap().get(id).cloned()
}

/// Prepend a chained response's conversation when `previous_response_id` is set.
fn apply_previous_response(body: &mut Value) {
    let Some(prev) = body.get("previous_response_id").and_then(|v| v.as_str()) else {
        return;
    };
    let prior = conv_store().lock().unwrap().get(prev).cloned();
    if let Some(mut prior) = prior {
        if let Some(arr) = body.get("messages").and_then(|v| v.as_array()) {
            prior.extend(arr.iter().cloned());
            body["messages"] = Value::Array(prior);
        }
    }
}

// ─────────────────────── token counting ───────────────────────

/// Worker-side `messages/count_tokens`: render the prompt and count its tokens.
fn count_tokens_job(tok: &Tokenizer, job: &Job, tmpl: Option<&ChatTemplate>) -> anyhow::Result<()> {
    let messages = parse_messages(&job.body)?;
    let prompt = render_prompt(tmpl, &messages, true, &[], true)?;
    let ids = tok.encode(&prompt, false)?;
    let _ = job.reply.send(Reply::Done {
        content: String::new(),
        reasoning: None,
        tool_calls: Vec::new(),
        prompt_tokens: ids.len(),
        completion_tokens: 0,
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
            let payload = json!({"error": {"message": e.to_string(), "type": "engine_error"}});
            return write_response(
                stream,
                "500 Internal Server Error",
                "application/json",
                payload.to_string().as_bytes(),
                "",
            );
        }
    };
    let payload = json!({ "input_tokens": d.prompt_tokens });
    write_response(
        stream,
        "200 OK",
        "application/json",
        payload.to_string().as_bytes(),
        "",
    )
}

fn complete(
    tower: &mut dyn LanguageModel,
    tok: &Tokenizer,
    job: &Job,
    cfg: &ServerConfig,
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
    let think_off = job
        .body
        .get("chat_template_kwargs")
        .and_then(|c| c.get("enable_thinking"))
        .and_then(|v| v.as_bool())
        == Some(false)
        || job.body.get("reasoning_effort").and_then(|v| v.as_str()) == Some("none")
        || job
            .body
            .get("reasoning")
            .and_then(|r| r.get("effort"))
            .and_then(|v| v.as_str())
            == Some("none");
    let thinking = job.surface != Surface::Completions && !think_off;
    // Anthropic semantics: a trailing assistant message is a prefill. Render the
    // generation header, then the partial assistant text, and let the model
    // continue it (rather than closing the turn and starting a new one).
    let prompt = if job.surface == Surface::Messages
        && messages.last().map(|m| m.role == "assistant").unwrap_or(false)
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
    let ids = tok.encode(&prompt, false)?;

    let max_tokens = request_max_tokens(job, cfg);
    let depth = match resolve_depth(job, cfg) {
        Ok(d) => d,
        Err(e) => {
            let _ = job.reply.send(Reply::Error(e.to_string()));
            return Ok(());
        }
    };
    let mut sampler = sampler_from(&job.body, cfg);
    let greedy = sampler.greedy();
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
        || job.body.get("top_logprobs").and_then(|v| v.as_u64()).unwrap_or(0) > 0;
    let mut collected: Vec<u32> = Vec::new();
    let mut last_text = String::new();
    let reply = job.reply.clone();
    let hit_stop = std::cell::Cell::new(false);
    let mut lp_entries: Vec<Value> = Vec::new();

    let mut session = Session::new(tower);
    let generated = {
        let mut emit = |t: u32, lp: Option<Value>| -> anyhow::Result<()> {
            if let Some(entry) = &lp {
                lp_entries.push(entry.clone());
            }
            let eos = is_eos(t);
            if !eos {
                collected.push(t);
                if !stops.is_empty() {
                    let text = tok.decode(&collected)?;
                    if stops.iter().any(|s| !s.is_empty() && text.contains(s.as_str())) {
                        hit_stop.set(true);
                        return Err(anyhow::anyhow!("stop sequence"));
                    }
                }
            }
            if streaming && json_schema.is_none() {
                let text = tok.decode(&collected)?;
                let ans = streaming_answer(&text);
                // Past a `<tool_call>` the text is not assistant content; don't stream it.
                let shown = match tool_call_start(ans) {
                    Some(i) => ans[..i].to_string(),
                    None => ans.to_string(),
                };
                let delta = stream_delta(&last_text, &shown);
                last_text = shown;
                // An EOS token contributes a logprob entry but no text; stream it
                // so the streamed logprob list matches the non-streamed one.
                if delta.is_some() || lp.is_some() {
                    let _ = reply.send(Reply::Delta {
                        text: delta.unwrap_or_default(),
                        logprobs: lp.map(|e| json!({ "content": [e] })),
                    });
                }
            }
            Ok(())
        };
        if want_logprobs {
            // Logprobs require the logits at every step; MTP drafts would skip
            // them, so this path runs one token at a time.
            let mut out: Vec<u32> = Vec::new();
            let mut logits = session.feed(tower, &ids)?;
            loop {
                let t = sampler.draw(&logits, &session.fed)?;
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
                if out.len() >= max_tokens || is_eos(t) {
                    break;
                }
                logits = session.feed(tower, &[t])?;
            }
            Ok(out)
        } else if depth > 0 && greedy {
            session.generate_mtp(tower, &ids, max_tokens, depth, &mut |t| emit(t, None))
        } else {
            session.generate(tower, &ids, max_tokens, &mut sampler, &mut |t| emit(t, None))
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
    if (!tools.is_empty() || json_schema.is_some())
        && std::env::var_os("LISA_DEBUG_TOOLS").is_some()
    {
        eprintln!("[raw] {full}");
    }
    let (content, reasoning, tool_calls) = if json_schema.is_some() {
        let raw = strip_stops(&full, &stops);
        let body = strip_think_prefix(raw);
        let content = match extract_json(body) {
            Some(s) => {
                let mut v: Value = serde_json::from_str(&s).unwrap_or(Value::String(s));
                if let Some(sc) = &json_schema {
                    repair_json(&mut v, sc);
                }
                v.to_string()
            }
            None => body.trim().to_string(),
        };
        (content, None, Vec::new())
    } else {
        shape_reply(&full, &tools, parallel, &stops)
    };
    if streaming && json_schema.is_some() {
        let _ = job.reply.send(Reply::Delta { text: content.clone(), logprobs: None });
    }
    if !tools.is_empty() && std::env::var_os("LISA_DEBUG_TOOLS").is_some() {
        eprintln!("[parsed] {tool_calls:?}");
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
    } else if hit_stop.get() {
        "stop"
    } else if generated.last().map(|&t| is_eos(t)).unwrap_or(false) {
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
        finish: finish.to_string(),
        logprobs,
        stop_sequence: stopped_by,
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
}

impl ChatTemplate {
    pub fn new(src: &str) -> anyhow::Result<Self> {
        let inner = hf_chat_template::ChatTemplate::from_str(src)
            .map_err(|e| anyhow::anyhow!("chat template parse: {e}"))?;
        Ok(Self { inner })
    }

    pub fn render(
        &self,
        messages: &[ChatMessage],
        add_generation_prompt: bool,
        tools: &[Value],
        thinking: bool,
    ) -> anyhow::Result<String> {
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
                msg.tool_calls = m.tool_calls.clone();
            }
            input.messages.push(msg);
        }
        input.extra.insert("enable_thinking".to_string(), json!(thinking));
        input.extra.insert("reasoning_effort".to_string(), json!("xhigh"));
        input.extra.insert("preserve_thinking".to_string(), json!(true));
        self.inner
            .render(&input)
            .map_err(|e| anyhow::anyhow!("chat template render: {e}"))
    }
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

/// Close the ` thinking` block a generation prompt opened, splitting the model's
/// raw output into (reasoning, answer). Falls back to all-content.
fn split_reasoning(text: &str) -> (Option<String>, String) {
    // Markers are ` thinking` / `` — the ASCII forms the checkpoint emits.
    const OPEN: &str = "\u{3c}think\u{3e}";
    const CLOSE: &str = "\u{3c}/think\u{3e}";
    if let Some(idx) = text.find(CLOSE) {
        let reasoning = text[..idx].trim().to_string();
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

/// The visible answer inside the text generated so far, for streaming. Mirrors
/// [`split_reasoning`] so a streamed reply reassembles to the non-streamed
/// `content`: nothing while a ` thinking` block is open, the answer once it
/// closes, the whole text when the model never thinks.
fn streaming_answer(text: &str) -> &str {
    const OPEN: &str = "\u{3c}think\u{3e}";
    const CLOSE: &str = "\u{3c}/think\u{3e}";
    if let Some(idx) = text.find(CLOSE) {
        return &text[idx + CLOSE.len()..];
    }
    if let Some(idx) = text.find(OPEN) {
        return &text[..idx];
    }
    text
}

/// Delta to append so a partially-streamed reply grows to `cur`. Decoding a
/// partial token sequence can rewrite earlier characters (a multi-byte glyph
/// split across tokens), so this diffs against the longest common prefix rather
/// than assuming `prev` is a prefix of `cur`. `None` when nothing new.
fn stream_delta(prev: &str, cur: &str) -> Option<String> {
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

/// Split a generated reply into visible content and OpenAI-style `tool_calls`,
/// parsing the Qwen/MiMo text form
/// `<tool_call><function=NAME><parameter=K>V</parameter>…</function></tool_call>`.
/// Any reasoning block is stripped first.
fn extract_tool_calls(text: &str, tools: &[Value]) -> (String, Vec<Value>) {
    let (_, answer) = split_reasoning(text);
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
    // Name comes from `<function=NAME>` or, for JSON-form calls, `"name"`.
    let mut name = body
        .split_once("<function=")
        .and_then(|(_, r)| r.split_once('>'))
        .map(|(n, _)| n.trim().to_string())
        .unwrap_or_default();
    let mut raw = Value::Null;
    if name.is_empty() {
        if let Some(obj) = find_json_object(body) {
            name = obj.get("name").and_then(|v| v.as_str()).unwrap_or("").to_string();
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
            Some("integer") => val.trim().parse::<i64>().map(Value::from).unwrap_or_else(|_| Value::String(val.to_string())),
            Some("number") => val.trim().parse::<f64>().map(Value::from).unwrap_or_else(|_| Value::String(val.to_string())),
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
                let k = nv[eq + 1..].trim().trim_matches(|c| c == '"' || c == '\'').to_string();
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
                args.insert(key.clone(), coerce(&key, &val));
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
) -> (String, Option<String>, Vec<Value>) {
    let text = strip_stops(full, stops);
    let (reasoning, answer) = split_reasoning(text);
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

fn sampler_from(body: &Value, cfg: &ServerConfig) -> Sampler {
    let f = |k: &str, d: f32| {
        body.get(k)
            .and_then(|v| v.as_f64())
            .map(|v| v as f32)
            .unwrap_or(d)
    };
    let u = |k: &str, d: usize| {
        body.get(k)
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(d)
    };
    let seed = body
        .get("seed")
        .and_then(|v| v.as_u64())
        .filter(|&v| v != 0)
        .unwrap_or(0x9E37_79B9_7F4A_7C15);
    Sampler {
        temperature: f("temperature", cfg.temperature),
        top_k: u("top_k", cfg.top_k),
        top_p: f("top_p", cfg.top_p),
        min_p: f("min_p", cfg.min_p),
        repetition_penalty: f("repetition_penalty", cfg.rep_penalty),
        seed,
        ..Default::default()
    }
}

fn read_request(stream: &mut TcpStream) -> anyhow::Result<Option<(String, String, Vec<u8>)>> {
    let mut reader = BufReader::new(stream.try_clone()?);
    let mut line = String::new();
    if reader.read_line(&mut line)? == 0 {
        return Ok(None);
    }
    let mut parts = line.split_whitespace();
    let method = parts.next().unwrap_or("").to_string();
    let path = parts.next().unwrap_or("").to_string();

    let mut content_length = 0usize;
    loop {
        let mut header = String::new();
        if reader.read_line(&mut header)? == 0 {
            break;
        }
        if header == "\r\n" || header == "\n" {
            break;
        }
        let lower = header.to_ascii_lowercase();
        if let Some(v) = lower.strip_prefix("content-length:") {
            content_length = v.trim().parse().unwrap_or(0);
        }
    }
    const MAX_BODY: usize = 8 << 20;
    anyhow::ensure!(content_length <= MAX_BODY, "request body too large ({content_length} bytes)");
    let mut body = vec![0u8; content_length];
    if content_length > 0 {
        reader.read_exact(&mut body)?;
    }
    Ok(Some((method, path, body)))
}
fn write_response(
    stream: &mut TcpStream,
    status: &str,
    content_type: &str,
    body: &[u8],
    extra: &str,
) -> anyhow::Result<()> {
    let mut head = format!(
        "HTTP/1.1 {status}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\n",
        body.len()
    );
    if !extra.is_empty() {
        head.push_str(extra);
        head.push_str("\r\n");
    }
    head.push_str("\r\n");
    stream.write_all(head.as_bytes())?;
    stream.write_all(body)?;
    stream.flush()?;
    Ok(())
}

fn write_chunk(stream: &mut TcpStream, data: &[u8]) -> std::io::Result<()> {
    if data.is_empty() {
        stream.write_all(b"0\r\n\r\n")
    } else {
        stream.write_all(format!("{:x}\r\n", data.len()).as_bytes())?;
        stream.write_all(data)?;
        stream.write_all(b"\r\n")
    }
}

fn write_sse(stream: &mut TcpStream, value: &Value) -> std::io::Result<()> {
    write_chunk(stream, format!("data: {value}\n\n").as_bytes())
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
        return write_response(&mut stream, "200 OK", "application/json", br#"{"status":"ok"}"#, "");
    }
    if method == "GET" && path.starts_with("/v1/models") {
        let payload = json!({"object":"list","data":[{"id":model.name(),"object":"model"}]});
        return write_response(&mut stream, "200 OK", "application/json", payload.to_string().as_bytes(), "");
    }
    if method == "POST" && path == "/v1/decisions" {
        let req: Value = match serde_json::from_slice(&body) {
            Ok(v) => v,
            Err(e) => return bad_request(&mut stream, format!("invalid JSON: {e}")),
        };
        let state = req.get("state").cloned().unwrap_or(Value::Null);
        let questions = req.get("questions").cloned().unwrap_or(Value::Null);
        return match model.system_one(&state, &questions) {
            Ok(ans) => write_response(&mut stream, "200 OK", "application/json", ans.to_string().as_bytes(), ""),
            Err(e) => bad_request(&mut stream, e.to_string()),
        };
    }
    write_response(&mut stream, "404 Not Found", "application/json", br#"{"error":{"message":"not found"}}"#, "")
}

fn bad_request(stream: &mut TcpStream, msg: String) -> anyhow::Result<()> {
    let body = json!({ "error": { "message": msg } }).to_string();
    write_response(stream, "400 Bad Request", "application/json", body.as_bytes(), "")
}

#[cfg(test)]
mod render_test {
    use super::{extract_tool_calls, find_json_object, salvage_json_object, ChatMessage, ChatTemplate};
    use serde_json::json;

    #[test]
    fn parses_tool_calls() {
        let text = "Sure.\n<tool_call>\n<function=get_weather>\n<parameter=city>Paris</parameter>\n<parameter=units>metric</parameter>\n</function>\n</tool_call>";
        let (content, calls) = extract_tool_calls(text, &[]);
        assert_eq!(content, "Sure.");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0]["type"], "function");
        assert_eq!(calls[0]["function"]["name"], "get_weather");
        let args: serde_json::Value =
            serde_json::from_str(calls[0]["function"]["arguments"].as_str().unwrap()).unwrap();
        assert_eq!(args["city"], "Paris");
        assert_eq!(args["units"], "metric");
    }

    #[test]
    fn parses_json_form_tool_calls() {
        let t1 = "<tool_call><function=run_command>{\"command\":\"npm test\"}</function></tool_call>";
        let (_, c1) = extract_tool_calls(t1, &[]);
        assert_eq!(c1.len(), 1);
        assert_eq!(c1[0]["function"]["name"], "run_command");
        let a1: serde_json::Value =
            serde_json::from_str(c1[0]["function"]["arguments"].as_str().unwrap()).unwrap();
        assert_eq!(a1["command"], "npm test");

        // Missing closing brace, plus a stray tag — must still yield `command`.
        // Unquoted keys/values (`{path:"README.md"}`) must still parse.
        let t1b = "<tool_call><function=read_file>{path:\"README.md\"}</function></tool_call>";
        let (_, c1b) = extract_tool_calls(t1b, &[]);
        let a1b: serde_json::Value =
            serde_json::from_str(c1b[0]["function"]["arguments"].as_str().unwrap()).unwrap();
        assert_eq!(a1b["path"], "README.md");

        let t2 = "<tool_call><function=run_command>{\"command\":\"node -e \\\"x\\\"\"</parameter></function></tool_call>";
        let (_, c2) = extract_tool_calls(t2, &[]);
        let a2: serde_json::Value =
            serde_json::from_str(c2[0]["function"]["arguments"].as_str().unwrap()).unwrap();
        assert_eq!(a2["command"], "node -e \"x\"");
    }

    #[test]
    fn salvages_truncated_calls() {
        // Truncated mid-value, no closing tags at all.
        let t = "<tool_call><function=write_file>{\"path\":\"src/defaults.js\",\"content\":\"module.exports = { timeoutMs: 10000, retries: 2 };\n";
        let (_, c) = extract_tool_calls(t, &[]);
        let a: serde_json::Value =
            serde_json::from_str(c[0]["function"]["arguments"].as_str().unwrap()).unwrap();
        assert_eq!(a["path"], "src/defaults.js");

        // Missing final brace, stray closing tag.
        let t2 = "<tool_call><function=run_command>{\"command\":\"node -e \\\"const {slugify}=x\\\"\"</parameter></function></tool_call>";
        let (_, c2) = extract_tool_calls(t2, &[]);
        let a2: serde_json::Value =
            serde_json::from_str(c2[0]["function"]["arguments"].as_str().unwrap()).unwrap();
        assert_eq!(a2["command"], "node -e \"const {slugify}=x\"");
    }

    #[test]
    fn parses_slugify_call() {
        let t = "<tool_call><function=run_command>{\"command\":\"node -e \\\"const {slugify}=require('./src/slug'); console.log(slugify('Hello World'));\\\"\"</parameter></function></tool_call>";
        let (_, c) = extract_tool_calls(t, &[]);
        eprintln!("c={c:?}");
        let a: serde_json::Value =
            serde_json::from_str(c[0]["function"]["arguments"].as_str().unwrap()).unwrap();
        assert!(a["command"].as_str().unwrap().contains("slugify"));
    }

    #[test]
    fn parses_u_call() {
        let t = "<think></think><tool_call><function=run_command>{\"command\":\"node -e \\\"const u=require('./src/user'); console.log(u.getInitials({first:'Ada',last:'Lovelace'}));\\\" && npm test 2>&1 | tail -20\"</}</function></tool_call>";
        let (_, c) = extract_tool_calls(t, &[]);
        assert!(!c.is_empty());
        let a: serde_json::Value =
            serde_json::from_str(c[0]["function"]["arguments"].as_str().unwrap()).unwrap();
        assert!(a["command"].as_str().unwrap().contains("npm test"));
    }

    #[test]
    fn supports_generation_markers() {
        // MiMo-style template block that minijinja does not understand.
        let src = "{%- macro m(x) -%}\n{%- generation -%}hello {{ x }}{%- endgeneration -%}\n{%- endmacro -%}{{ m(1) }}";
        let t = ChatTemplate::new(src).expect("template parse");
        let out = t.render(&[], false, &[], true).expect("render");
        assert!(out.contains("hello 1"), "got {out:?}");
    }

    #[test]
    #[ignore]
    fn render_mimo_tools_and_parts() {
        let dir = match lisa_engine::models::resolve_model_dir(
            "agnosticeng/MiMo-V2.6-Distill-Qwen-9B-4bit",
        ) {
            Ok(d) => d,
            Err(_) => return,
        };
        let src = match std::fs::read_to_string(dir.join("chat_template.jinja")) {
            Ok(s) => s,
            Err(_) => return,
        };
        // Exercise the branches our text-only ChatMessage wrapper never feeds:
        // tools, tool_calls, tool results, and multimodal content parts.
        let t = hf_chat_template::ChatTemplate::from_str(&src).expect("template parse");
        let mut input = hf_chat_template::RenderInput {
            add_generation_prompt: true,
            ..Default::default()
        };
        input.tools.push(json!({
            "type": "function",
            "function": { "name": "get_weather", "parameters": { "type": "object" } }
        }));
        input
            .messages
            .push(hf_chat_template::Message::user("weather in Paris?"));
        let mut call = hf_chat_template::Message::assistant("");
        call.tool_calls.push(json!({
            "function": { "name": "get_weather", "arguments": { "city": "Paris" } }
        }));
        input.messages.push(call);
        input
            .messages
            .push(hf_chat_template::Message::tool("sunny, 21C"));
        let mut img = hf_chat_template::Message::user("");
        img.content = Some(hf_chat_template::Content::Parts(vec![
            json!({"type": "image"}),
            json!({"type": "text", "text": "describe"}),
        ]));
        input.messages.push(img);
        input.extra.insert("enable_thinking".into(), json!(false));
        let out = t.render(&input).expect("render");
        eprintln!("---RENDER---\n{out}\n---END---");
    }

    #[test]
    #[ignore]
    fn render_mimo_template() {
        let dir = match lisa_engine::models::resolve_model_dir(
            "agnosticeng/MiMo-V2.6-Distill-Qwen-9B-4bit",
        ) {
            Ok(d) => d,
            Err(_) => return,
        };
        let src = match std::fs::read_to_string(dir.join("chat_template.jinja")) {
            Ok(s) => s,
            Err(_) => return,
        };
        let t = ChatTemplate::new(&src).expect("template parse");
        let msgs = vec![ChatMessage {
            role: "user".into(),
            content: "Name three colors.".into(),
            reasoning_content: None,
            tool_calls: Vec::new(),
            tool_call_id: None,
            parts: None,
        }];
        let out = t.render(&msgs, true, &[], true).expect("render");
        eprintln!("---RENDER---\n{out}\n---END---");
    }

    #[test]
    #[ignore]
    fn render_real_template() {
        let dir = match lisa_engine::models::resolve_model_dir(
            "agnosticeng/Qwen3.8-Flash-Next-4bit",
        ) {
            Ok(d) => d,
            Err(_) => return,
        };
        let src = match std::fs::read_to_string(dir.join("chat_template.jinja")) {
            Ok(s) => s,
            Err(_) => return,
        };
        let t = ChatTemplate::new(&src).expect("template parse");
        let msgs = vec![
            ChatMessage { role: "user".into(), content: "Name three colors.".into(), reasoning_content: None, tool_calls: Vec::new(), tool_call_id: None, parts: None },
            ChatMessage { role: "assistant".into(), content: "Red, green, blue.".into(), reasoning_content: Some("Thinking about colors.".into()), tool_calls: Vec::new(), tool_call_id: None, parts: None },
            ChatMessage { role: "user".into(), content: "Which is warmest?".into(), reasoning_content: None, tool_calls: Vec::new(), tool_call_id: None, parts: None },
        ];
        let out = t.render(&msgs, true, &[], true).expect("render");
        eprintln!("---RENDER---\n{out}\n---END---");
    }
}
