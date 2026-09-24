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
use std::sync::mpsc::{self, Receiver, Sender};
use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::{json, Value};

use lisa_engine::core::generate::is_eos;
use lisa_engine::models::qwen4::tower::Tower;
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
}

enum Reply {
    Delta(String),
    Done {
        content: String,
        prompt_tokens: usize,
        completion_tokens: usize,
        finish: String,
    },
    Error(String),
}

struct Job {
    body: Value,
    stream: bool,
    /// When set, the request continues a persistent multi-turn conversation
    /// (prefix reuse); it is handled on its own `Session`, not in a batch.
    session_id: Option<String>,
    reply: Sender<Reply>,
}

pub fn run(tower: &mut Tower, tok: &Tokenizer, cfg: ServerConfig) -> anyhow::Result<()> {
    let listener = TcpListener::bind(&cfg.addr)
        .map_err(|e| anyhow::anyhow!("bind {}: {e}", cfg.addr))?;
    let addr = listener.local_addr()?;
    println!("lisa serve listening on http://{addr}");
    println!("  POST /v1/chat/completions   GET /v1/models   GET /health");
    println!("  max_batch {}", cfg.max_batch);

    let (tx, rx) = mpsc::channel::<Job>();
    let cfg = std::sync::Arc::new(cfg);
    {
        let cfg = cfg.clone();
        let tx = tx.clone();
        std::thread::spawn(move || {
            for stream in listener.incoming() {
                let Ok(stream) = stream else { continue };
                let tx = tx.clone();
                let cfg = cfg.clone();
                std::thread::spawn(move || {
                    if let Err(e) = handle_conn(stream, tx, &cfg) {
                        eprintln!("[serve] connection error: {e}");
                    }
                });
            }
        });
    }

    let mut sessions: HashMap<String, Session> = HashMap::new();
    loop {
        // Block for one request, then drain whatever else is queued (up to the
        // batch width) so several arrivals are served together.
        let Ok(first) = rx.recv() else { break };
        let mut wave = vec![first];
        while wave.len() < cfg.max_batch.max(1) {
            match rx.try_recv() {
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
        for job in sess_jobs {
            if let Err(e) = run_session(tower, tok, &mut sessions, &job, &cfg) {
                let _ = job.reply.send(Reply::Error(e.to_string()));
            }
        }
        // Speculative requests are per-stream with batch size 1 (the reference
        // engine's `maximumSpeculativeBatch`): they take the single-request
        // `Session` MTP path. Non-speculative requests share the continuous
        // batch.
        if !batch_jobs.is_empty() {
            let (spec, plain): (Vec<Job>, Vec<Job>) = batch_jobs
                .into_iter()
                .partition(|j| depth_for(j, &cfg) > 0);
            for job in spec {
                if let Err(e) = complete(tower, tok, &job, &cfg) {
                    let _ = job.reply.send(Reply::Error(e.to_string()));
                }
            }
            if !plain.is_empty() {
                if plain.len() == 1 {
                    if let Err(e) = complete(tower, tok, &plain[0], &cfg) {
                        let _ = plain[0].reply.send(Reply::Error(e.to_string()));
                    }
                } else if let Err(e) = run_wave(tower, tok, plain, &cfg) {
                    eprintln!("[serve] wave error: {e}");
                }
            }
        }
        if std::env::var("LISA_PROFILE").is_ok() {
            lisa_engine::core::mem::mlx_mem_line("serve-wave");
        }
    }
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
fn build_request(id: usize, tok: &Tokenizer, job: &Job, cfg: &ServerConfig) -> anyhow::Result<lisa_engine::core::sched::Request> {
    let messages = parse_messages(&job.body)?;
    let prompt = tokenizer::generation_prompt(&messages);
    let ids = tok.encode(&prompt, false)?;
    let max_tokens = job
        .body
        .get("max_tokens")
        .and_then(|v| v.as_u64())
        .map(|v| v as usize)
        .unwrap_or(cfg.max_tokens);
    Ok(lisa_engine::core::sched::Request {
        id,
        prompt: ids,
        max_tokens,
        sampler: sampler_from(&job.body, cfg),
    })
}

/// Serve a wave of stateless requests through the continuous-batch scheduler,
/// streaming each request's tokens back to its own connection.
fn run_wave(tower: &mut Tower, tok: &Tokenizer, jobs: Vec<Job>, cfg: &ServerConfig) -> anyhow::Result<()> {
    let mut job_of: Vec<usize> = Vec::new();
    let mut reqs: Vec<lisa_engine::core::sched::Request> = Vec::new();
    for (ji, job) in jobs.iter().enumerate() {
        match build_request(reqs.len(), tok, job, cfg) {
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
                let _ = replies[id].send(Reply::Delta(delta));
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
    tower: &mut Tower,
    tok: &Tokenizer,
    sessions: &mut HashMap<String, Session>,
    job: &Job,
    cfg: &ServerConfig,
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
    let prompt = tokenizer::generation_prompt(&messages);
    let ids = tok.encode(&prompt, false)?;

    let mut sess = sessions.remove(&sid).unwrap_or_else(|| Session::new(tower));
    let cp = common_prefix(&sess.fed, &ids);
    if cp < sess.fed.len() {
        // The request diverges from the cached conversation: start over.
        sess = Session::new(tower);
    }
    let cp = cp.min(sess.fed.len());
    let suffix: Vec<u32> = ids[cp..].to_vec();
    anyhow::ensure!(!suffix.is_empty(), "session request has no new tokens");

    let max_tokens = job
        .body
        .get("max_tokens")
        .and_then(|v| v.as_u64())
        .map(|v| v as usize)
        .unwrap_or(cfg.max_tokens);
    let mut sampler = sampler_from(&job.body, cfg);
    let greedy = sampler.greedy();
    let streaming = job.stream;
    let reply = job.reply.clone();

    let mut acc: Vec<u32> = Vec::new();
    let mut last = String::new();
    let mut emit = |t: u32| -> anyhow::Result<()> {
        if is_eos(t) {
            return Ok(());
        }
        acc.push(t);
        if streaming {
            let text = tok.decode(&acc)?;
            if text.len() > last.len() {
                let delta = text[last.len()..].to_string();
                let _ = reply.send(Reply::Delta(delta));
            }
            last = text;
        }
        Ok(())
    };
    let generated = if depth > 0 && greedy {
        sess.generate_mtp(tower, &suffix, max_tokens, depth, &mut emit)?
    } else {
        sess.generate(tower, &suffix, max_tokens, &mut sampler, &mut emit)?
    };
    sessions.insert(sid, sess);

    let stopped = generated.last().map(|&t| is_eos(t)).unwrap_or(false);
    let content = if streaming { last } else { tok.decode(&acc)? };
    let _ = job.reply.send(Reply::Done {
        content,
        prompt_tokens: suffix.len(),
        completion_tokens: acc.len(),
        finish: if stopped { "stop".into() } else { "length".into() },
    });
    Ok(())
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
            let payload = json!({"error": {"message": e.to_string()}});
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
            "data": [{ "id": "qwen3.8-flash-next", "object": "model", "owned_by": "lisa" }]
        });
        return write_response(&mut stream, "200 OK", "application/json", payload.to_string().as_bytes(), "");
    }
    if method == "POST" && path == "/v1/chat/completions" {
        if let Err(e) = handle_completion(&mut stream, &body, tx, cfg) {
            let payload = json!({"error": {"message": e.to_string(), "type": "invalid_request_error"}});
            let _ = write_response(
                &mut stream,
                "400 Bad Request",
                "application/json",
                payload.to_string().as_bytes(),
                "",
            );
        }
        return Ok(());
    }

    write_response(&mut stream, "404 Not Found", "application/json", br#"{"error":{"message":"not found"}}"#, "")
}

fn handle_completion(
    stream: &mut TcpStream,
    body: &[u8],
    tx: Sender<Job>,
    _cfg: &ServerConfig,
) -> anyhow::Result<()> {
    let body: Value = serde_json::from_slice(body)
        .map_err(|e| anyhow::anyhow!("invalid JSON body: {e}"))?;
    let stream_mode = body.get("stream").and_then(|v| v.as_bool()).unwrap_or(false);
    let session_id = body
        .get("session_id")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());

    let (rtx, rrx) = mpsc::channel::<Reply>();
    tx.send(Job {
        body,
        stream: stream_mode,
        session_id,
        reply: rtx,
    })
    .map_err(|_| anyhow::anyhow!("engine worker is gone"))?;

    if stream_mode {
        stream_completion(stream, rrx)
    } else {
        collect_completion(stream, rrx)
    }
}

fn collect_completion(stream: &mut TcpStream, rx: Receiver<Reply>) -> anyhow::Result<()> {
    let mut content = String::new();
    let (prompt_tokens, completion_tokens, finish) = loop {
        match rx.recv() {
            Ok(Reply::Delta(d)) => content.push_str(&d),
            Ok(Reply::Done {
                content: c,
                prompt_tokens,
                completion_tokens,
                finish,
            }) => {
                content = c;
                break (prompt_tokens, completion_tokens, finish);
            }
            Ok(Reply::Error(e)) => {
                let payload = json!({"error": {"message": e, "type": "engine_error"}});
                return write_response(stream, "500 Internal Server Error", "application/json", payload.to_string().as_bytes(), "");
            }
            Err(_) => anyhow::bail!("engine worker dropped the request"),
        }
    };
    let payload = json!({
        "id": completion_id(),
        "object": "chat.completion",
        "created": now(),
        "model": "qwen3.8-flash-next",
        "choices": [{
            "index": 0,
            "message": { "role": "assistant", "content": content },
            "finish_reason": finish,
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    });
    write_response(stream, "200 OK", "application/json", payload.to_string().as_bytes(), "")
}

fn stream_completion(stream: &mut TcpStream, rx: Receiver<Reply>) -> anyhow::Result<()> {
    stream.write_all(
        b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\nTransfer-Encoding: chunked\r\n\r\n",
    )?;
    let id = completion_id();
    let chunk = |delta: Value, finish: Value| {
        json!({
            "id": id,
            "object": "chat.completion.chunk",
            "created": now(),
            "model": "qwen3.8-flash-next",
            "choices": [{ "index": 0, "delta": delta, "finish_reason": finish }],
        })
    };
    write_sse(stream, &chunk(json!({"role": "assistant"}), Value::Null))?;

    for reply in rx {
        match reply {
            Reply::Delta(d) => write_sse(stream, &chunk(json!({"content": d}), Value::Null))?,
            Reply::Done { finish, .. } => {
                write_sse(stream, &chunk(json!({}), json!(finish)))?;
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

fn complete(
    tower: &mut Tower,
    tok: &Tokenizer,
    job: &Job,
    cfg: &ServerConfig,
) -> anyhow::Result<()> {
    let messages = parse_messages(&job.body)?;
    let prompt = tokenizer::generation_prompt(&messages);
    let ids = tok.encode(&prompt, false)?;

    let max_tokens = job
        .body
        .get("max_tokens")
        .and_then(|v| v.as_u64())
        .map(|v| v as usize)
        .unwrap_or(cfg.max_tokens);
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

    let mut collected: Vec<u32> = Vec::new();
    let mut last_text = String::new();
    let reply = job.reply.clone();
    let mut emit = |t: u32| -> anyhow::Result<()> {
        if is_eos(t) {
            return Ok(());
        }
        collected.push(t);
        if streaming {
            let text = tok.decode(&collected)?;
            if text.len() > last_text.len() {
                let delta = text[last_text.len()..].to_string();
                let _ = reply.send(Reply::Delta(delta));
            }
            last_text = text;
        }
        Ok(())
    };

    let mut session = Session::new(tower);
    let generated = if depth > 0 && greedy {
        session.generate_mtp(tower, &ids, max_tokens, depth, &mut emit)?
    } else {
        session.generate(tower, &ids, max_tokens, &mut sampler, &mut emit)?
    };

    let content = if streaming {
        last_text
    } else {
        tok.decode(&collected)?
    };
    let finish = if generated.last().map(|&t| is_eos(t)).unwrap_or(false) {
        "stop"
    } else {
        "length"
    };
    let _ = job.reply.send(Reply::Done {
        content,
        prompt_tokens: ids.len(),
        completion_tokens: collected.len(),
        finish: finish.to_string(),
    });
    Ok(())
}

fn parse_messages(body: &Value) -> anyhow::Result<Vec<(String, String)>> {
    let mut out: Vec<(String, String)> = Vec::new();
    if let Some(sys) = body.get("system").and_then(|v| v.as_str()) {
        out.push(("system".to_string(), sys.to_string()));
    }
    if let Some(arr) = body.get("messages").and_then(|v| v.as_array()) {
        for m in arr {
            let role = m
                .get("role")
                .and_then(|v| v.as_str())
                .unwrap_or("user")
                .to_string();
            let content = match m.get("content") {
                Some(Value::String(s)) => s.clone(),
                Some(Value::Array(parts)) => parts
                    .iter()
                    .filter_map(|p| p.get("text").and_then(|t| t.as_str()))
                    .collect::<Vec<_>>()
                    .join(""),
                _ => String::new(),
            };
            out.push((role, content));
        }
    } else if let Some(p) = body.get("prompt").and_then(|v| v.as_str()) {
        out.push(("user".to_string(), p.to_string()));
    }
    anyhow::ensure!(!out.is_empty(), "request has no messages or prompt");
    Ok(out)
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
    let head = format!(
        "HTTP/1.1 {status}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\n{extra}\r\n",
        body.len()
    );
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
