//! Connection handling and normalization of Chat, Completions, Messages and Responses requests.
use super::*;

pub(super) fn models_payload(id: &str, supports_images: bool) -> Value {
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

pub(super) fn handle_conn(
    api: &ApiState,
    mut stream: TcpStream,
    tx: WakeSender<Job>,
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
    application::trace_http("serve.http.request_read");
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
pub(super) fn normalize_completion_prompt(body: &mut Value) {
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
pub(super) fn dispatch_or_400(
    api: &ApiState,
    stream: &mut TcpStream,
    body: &[u8],
    tx: WakeSender<Job>,
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

pub(super) fn dispatch_completion(
    api: &ApiState,
    stream: &mut TcpStream,
    body: &[u8],
    tx: WakeSender<Job>,
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
    if lisa_engine::vision::messages_have_images(&body) {
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
        lisa_engine::vision::decode_message_images(&body)?;
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

pub(super) fn anthropic_system_text(v: &Value) -> String {
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
pub(super) fn block_content_text(v: &Value) -> String {
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
pub(super) fn normalize_messages_body(body: &mut Value) {
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

/// Convert an OpenAI `/v1/responses` body into the internal chat shape.
pub(super) fn normalize_responses_body(body: &mut Value) {
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
    if let Some(tc) = body.get("tool_choice").cloned()
        && let Some(ty) = tc.get("type").and_then(|v| v.as_str())
    {
        if ty == "function" {
            body["tool_choice"] = json!({
                "type": "function",
                "function": { "name": tc.get("name").and_then(|v| v.as_str()).unwrap_or("") }
            });
        } else {
            body["tool_choice"] = json!(ty);
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
pub(super) fn response_id() -> String {
    let n = SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    format!("resp_{:x}{:x}", now(), n)
}

/// Worker-side `messages/count_tokens`: render the prompt and count its tokens.
pub(super) fn count_tokens_job(
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

pub(super) fn collect_count_tokens(
    stream: &mut TcpStream,
    rx: Receiver<Reply>,
) -> anyhow::Result<()> {
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

pub(super) fn parse_messages(body: &Value) -> anyhow::Result<Vec<ChatMessage>> {
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
pub(super) fn request_tools(job: &Job) -> Vec<Value> {
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

pub(super) fn parallel_tools(job: &Job) -> bool {
    job.body
        .get("parallel_tool_calls")
        .and_then(|v| v.as_bool())
        .unwrap_or(true)
}

pub(super) fn request_stops(job: &Job) -> Vec<String> {
    match job.body.get("stop") {
        Some(Value::String(s)) => vec![s.clone()],
        Some(Value::Array(a)) => a
            .iter()
            .filter_map(|v| v.as_str().map(|s| s.to_string()))
            .collect(),
        _ => Vec::new(),
    }
}

pub(super) fn request_max_tokens(job: &Job, cfg: &ServerConfig) -> usize {
    job.body
        .get("max_completion_tokens")
        .or_else(|| job.body.get("max_tokens"))
        .and_then(|v| v.as_u64())
        .map(|v| v as usize)
        .unwrap_or(cfg.max_tokens)
}

pub(super) fn has_tools(job: &Job) -> bool {
    !request_tools(job).is_empty()
}

pub(super) fn shape_reply(
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

/// Keep history-dependent greedy requests on the configured sampler path.
/// Target MTP also keeps its existing reference sampling and RNG contract.
pub(super) fn can_omit_configured_sampler(sampler: &Sampler, target_mtp: bool) -> bool {
    !target_mtp && sampler.can_use_raw_argmax() && !sampler.mtp_exact_sampling
}

/// Sampling decision chain: request body >
/// server flags > checkpoint `generation_config.json` > hardcoded defaults
/// (greedy). Applied identically on every generation path (batched, serial,
/// MTP-verify), so a silent request samples the same way everywhere.
pub(super) fn sampler_from(body: &Value, cfg: &ServerConfig) -> Sampler {
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
    Sampler::new(lisa_engine::sampling::Sampler {
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
    })
}

/// Whether a request renders its generation prompt with thinking on: an
/// explicit request value wins (`chat_template_kwargs.enable_thinking`,
/// `reasoning_effort` / `reasoning.effort`), then the checkpoint's own
/// declaration (`generation_config.json` →
/// `default_chat_template_kwargs.enable_thinking`), then off. Silent requests
/// default to off.
pub(super) fn request_thinking(body: &Value, cfg: &ServerConfig) -> bool {
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
            body.get("chat_template_kwargs")
                .and_then(|c| c.get("reasoning_effort"))
                .and_then(Value::as_str)
        })
        .or_else(|| {
            body.get("reasoning")
                .and_then(|r| r.get("effort"))
                .and_then(|v| v.as_str())
        });
    if let Some(e) = effort {
        return !matches!(e, "none" | "disabled");
    }
    cfg.gen_defaults.thinking.unwrap_or(false)
}

pub(super) fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

pub(super) fn completion_id() -> String {
    let n = SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    format!("chatcmpl-lisa-{:x}{:x}", now(), n)
}

pub(super) static SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

pub(super) fn bad_request(stream: &mut TcpStream, msg: String) -> anyhow::Result<()> {
    let body = json!({ "error": { "message": msg } }).to_string();
    write_response(
        stream,
        "400 Bad Request",
        "application/json",
        body.as_bytes(),
    )
}
