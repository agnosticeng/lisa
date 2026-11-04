//! Wire representations and streaming for the supported completion APIs.

use std::io::Write;
use std::net::TcpStream;
use std::sync::atomic::Ordering;
use std::sync::mpsc::{self, Receiver, Sender};

use serde_json::{Value, json};

use super::http::{write_chunk, write_response, write_sse};
use super::{
    ApiState, Job, REFUSED_MARKER, Reply, SEQ, Surface, completion_id, error_message_for,
    http_status_for, now, response_id,
};

/// A fully-collected engine reply.
pub(super) struct ReplyData {
    pub(super) content: String,
    pub(super) reasoning: Option<String>,
    pub(super) tool_calls: Vec<Value>,
    pub(super) prompt_tokens: usize,
    pub(super) completion_tokens: usize,
    pub(super) cached_tokens: usize,
    pub(super) finish: String,
    pub(super) logprobs: Option<Value>,
    pub(super) stop_sequence: Option<String>,
}

pub(super) fn drain_reply(rx: Receiver<Reply>) -> anyhow::Result<ReplyData> {
    loop {
        match rx.recv() {
            Ok(Reply::Start { .. } | Reply::Delta { .. }) => {}
            Ok(Reply::Done {
                content: c,
                reasoning,
                tool_calls,
                prompt_tokens,
                completion_tokens,
                cached_tokens,
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
                    cached_tokens,
                    finish,
                    logprobs,
                    stop_sequence,
                });
            }
            Ok(Reply::Error(e)) => anyhow::bail!("{e}"),
            Ok(Reply::Refused {
                retry_after_s,
                reason,
            }) => anyhow::bail!("{REFUSED_MARKER}retry-after={retry_after_s}; {reason}"),
            Err(_) => anyhow::bail!("engine worker dropped the request"),
        }
    }
}

pub(super) fn prompt_tokens_details_json(prompt_tokens: usize, cached_tokens: usize) -> Value {
    let cached_tokens = cached_tokens.min(prompt_tokens);
    let cache_write_tokens = prompt_tokens.saturating_sub(cached_tokens);
    json!({
        "cached_tokens": cached_tokens,
        "cache_write_tokens": cache_write_tokens,
    })
}

pub(super) fn usage_json(
    prompt_tokens: usize,
    completion_tokens: usize,
    cached_tokens: usize,
) -> Value {
    json!({
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        // Match DS4's cache accounting and expose prefix-cache hits to clients.
        "prompt_tokens_details": prompt_tokens_details_json(prompt_tokens, cached_tokens),
        "total_tokens": prompt_tokens + completion_tokens,
    })
}

pub(super) fn chat_message(d: &ReplyData) -> Value {
    let mut message = json!({ "role": "assistant", "content": d.content });
    if let Some(r) = &d.reasoning {
        message["reasoning_content"] = json!(r);
    }
    if !d.tool_calls.is_empty() {
        message["tool_calls"] = json!(d.tool_calls);
    }
    message
}

pub(super) fn chat_choice(d: &ReplyData, index: usize) -> Value {
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

pub(super) fn completion_choice(d: &ReplyData, index: usize) -> Value {
    json!({
        "index": index,
        "text": d.content,
        "logprobs": d.logprobs,
        "finish_reason": d.finish,
    })
}

pub(super) fn single_payload(api: &ApiState, d: &ReplyData, surface: Surface) -> Value {
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
        "model": api.model_id,
        "choices": [choice],
        "usage": usage_json(d.prompt_tokens, d.completion_tokens, d.cached_tokens),
    })
}

pub(super) fn collect_surface(
    api: &ApiState,
    stream: &mut TcpStream,
    rx: Receiver<Reply>,
    surface: Surface,
    req_body: &Value,
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
    let payload = match surface {
        Surface::Messages => messages_payload(api, &d),
        Surface::Responses => {
            let id = response_id();
            let p = responses_payload_with_id(api, &d, &id);
            store_conversation(api, &id, req_body, &d, &p);
            p
        }
        // The embeddings job serializes its whole OpenAI payload itself.
        Surface::Embeddings => serde_json::from_str(&d.content)
            .unwrap_or_else(|_| json!({"error": {"message": "embedding payload corrupt"}})),
        _ => single_payload(api, &d, surface),
    };
    write_response(
        stream,
        "200 OK",
        "application/json",
        payload.to_string().as_bytes(),
    )
}

/// Anthropic Messages response body.
pub(super) fn messages_payload(api: &ApiState, d: &ReplyData) -> Value {
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
        "model": api.model_id,
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
pub(super) fn empty_response_resource(api: &ApiState, id: &str, status: &str) -> Value {
    json!({
        "id": id,
        "object": "response",
        "created_at": now(),
        "completed_at": Value::Null,
        "status": status,
        "incomplete_details": Value::Null,
        "model": api.model_id,
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

pub(super) fn responses_payload_with_id(api: &ApiState, d: &ReplyData, id: &str) -> Value {
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
    let mut resp = empty_response_resource(
        api,
        id,
        if incomplete {
            "incomplete"
        } else {
            "completed"
        },
    );
    resp["completed_at"] = if incomplete {
        Value::Null
    } else {
        json!(now())
    };
    resp["output"] = Value::Array(output);
    resp["output_text"] = json!(d.content);
    resp["usage"] = json!({
        "input_tokens": d.prompt_tokens,
        "output_tokens": d.completion_tokens,
        "total_tokens": d.prompt_tokens + d.completion_tokens,
        "input_tokens_details": prompt_tokens_details_json(d.prompt_tokens, d.cached_tokens),
        "output_tokens_details": { "reasoning_tokens": 0 },
    });
    if incomplete {
        resp["incomplete_details"] = json!({ "reason": "max_output_tokens" });
    }
    resp
}

/// `n > 1`: run `n` independent generations (distinct seeds) and merge them into
/// one choices array.
pub(super) fn collect_choices(
    api: &ApiState,
    stream: &mut TcpStream,
    body: &Value,
    tx: Sender<Job>,
    n: usize,
    surface: Surface,
) -> anyhow::Result<()> {
    let base = body
        .get("seed")
        .and_then(|v| v.as_u64())
        .unwrap_or(0x9E37_79B9_7F4A_7C15);
    let mut choices = Vec::with_capacity(n);
    let mut prompt_tokens = 0;
    let mut completion_tokens = 0;
    let mut cached_tokens = 0;
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
        prompt_tokens = d.prompt_tokens;
        cached_tokens = d.cached_tokens;
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
        "model": api.model_id,
        "choices": choices,
        "usage": usage_json(prompt_tokens, completion_tokens, cached_tokens),
    });
    write_response(
        stream,
        "200 OK",
        "application/json",
        payload.to_string().as_bytes(),
    )
}

pub(super) fn stream_surface(
    api: &ApiState,
    stream: &mut TcpStream,
    rx: Receiver<Reply>,
    include_usage: bool,
    surface: Surface,
    req_body: &Value,
) -> anyhow::Result<()> {
    stream.write_all(
        b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\nTransfer-Encoding: chunked\r\n\r\n",
    )?;
    match surface {
        Surface::Completions => stream_completions(api, stream, rx, include_usage),
        Surface::Messages => stream_messages(api, stream, rx),
        Surface::Responses => stream_responses(api, stream, rx, req_body),
        _ => stream_chat(api, stream, rx, include_usage),
    }
}

pub(super) fn stream_chunk(api: &ApiState, object: &str, id: &str, choice: Value) -> Value {
    json!({
        "id": id,
        "object": object,
        "created": now(),
        "model": api.model_id,
        "choices": [choice],
    })
}

pub(super) fn stream_chat(
    api: &ApiState,
    stream: &mut TcpStream,
    rx: Receiver<Reply>,
    include_usage: bool,
) -> anyhow::Result<()> {
    let id = completion_id();
    let chunk = |delta: Value, finish: Value, logprobs: Option<Value>| {
        let mut choice = json!({ "index": 0, "delta": delta, "finish_reason": finish });
        if let Some(lp) = logprobs {
            choice["logprobs"] = lp;
        }
        stream_chunk(api, "chat.completion.chunk", &id, choice)
    };
    write_sse(
        stream,
        &chunk(json!({"role": "assistant"}), Value::Null, None),
    )?;

    for reply in rx {
        match reply {
            Reply::Start { .. } => {}
            Reply::Delta {
                text,
                reasoning,
                logprobs,
            } => {
                let mut delta = json!({});
                if !text.is_empty() || logprobs.is_some() {
                    delta["content"] = json!(text);
                }
                if let Some(reasoning) = reasoning {
                    delta["reasoning_content"] = json!(reasoning);
                }
                if !delta.as_object().map_or(true, |object| object.is_empty()) {
                    write_sse(stream, &chunk(delta, Value::Null, logprobs))?;
                }
            }
            Reply::Done {
                finish,
                tool_calls,
                prompt_tokens,
                completion_tokens,
                cached_tokens,
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
                    write_sse(
                        stream,
                        &chunk(json!({"tool_calls": deltas}), Value::Null, None),
                    )?;
                }
                write_sse(stream, &chunk(json!({}), json!(finish), None))?;
                if include_usage {
                    let usage = json!({
                        "id": id,
                        "object": "chat.completion.chunk",
                        "created": now(),
                        "model": api.model_id,
                        "choices": [],
                        "usage": usage_json(prompt_tokens, completion_tokens, cached_tokens),
                    });
                    write_sse(stream, &usage)?;
                }
                write_chunk(stream, b"data: [DONE]\n\n")?;
                write_chunk(stream, b"")?;
                break;
            }
            Reply::Refused {
                retry_after_s,
                reason,
            } => {
                write_sse(
                    stream,
                    &json!({"error": {"message": reason, "type": "insufficient_memory", "retry_after_s": retry_after_s}}),
                )?;
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

pub(super) fn stream_completions(
    api: &ApiState,
    stream: &mut TcpStream,
    rx: Receiver<Reply>,
    include_usage: bool,
) -> anyhow::Result<()> {
    let id = completion_id();
    for reply in rx {
        match reply {
            Reply::Start { .. } => {}
            Reply::Delta { text, .. } => {
                write_sse(
                    stream,
                    &stream_chunk(
                        api,
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
                cached_tokens,
                ..
            } => {
                write_sse(
                    stream,
                    &stream_chunk(
                        api,
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
                        "model": api.model_id,
                        "choices": [],
                        "usage": usage_json(prompt_tokens, completion_tokens, cached_tokens),
                    });
                    write_sse(stream, &usage)?;
                }
                write_chunk(stream, b"data: [DONE]\n\n")?;
                write_chunk(stream, b"")?;
                break;
            }
            Reply::Refused {
                retry_after_s,
                reason,
            } => {
                write_sse(
                    stream,
                    &json!({"error": {"message": reason, "type": "insufficient_memory", "retry_after_s": retry_after_s}}),
                )?;
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
pub(super) fn sse_event(
    stream: &mut TcpStream,
    event: Option<&str>,
    payload: &Value,
) -> std::io::Result<()> {
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

pub(super) fn stream_messages(
    api: &ApiState,
    stream: &mut TcpStream,
    rx: Receiver<Reply>,
) -> anyhow::Result<()> {
    let id = format!("msg_{:x}{:x}", now(), SEQ.fetch_add(1, Ordering::Relaxed));
    let mut started = false;
    let mut block_index = 0usize;
    let mut text_started = false;
    let mut thinking_started = false;
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
                                "model": api.model_id,
                                "content": [],
                                "stop_reason": Value::Null,
                                "stop_sequence": Value::Null,
                                "usage": { "input_tokens": prompt_tokens, "output_tokens": 0 },
                            }
                        }),
                    )?;
                }
            }
            Reply::Delta {
                text, reasoning, ..
            } => {
                if let Some(reasoning) = reasoning.filter(|r| !r.is_empty()) {
                    if text_started {
                        sse_event(
                            stream,
                            Some("content_block_stop"),
                            &json!({ "type": "content_block_stop", "index": block_index }),
                        )?;
                        block_index += 1;
                        text_started = false;
                    }
                    if !thinking_started {
                        sse_event(
                            stream,
                            Some("content_block_start"),
                            &json!({
                                "type": "content_block_start",
                                "index": block_index,
                                "content_block": { "type": "thinking", "thinking": "" },
                            }),
                        )?;
                        thinking_started = true;
                    }
                    sse_event(
                        stream,
                        Some("content_block_delta"),
                        &json!({
                            "type": "content_block_delta",
                            "index": block_index,
                            "delta": { "type": "thinking_delta", "thinking": reasoning },
                        }),
                    )?;
                }
                if !text.is_empty() {
                    if thinking_started {
                        sse_event(
                            stream,
                            Some("content_block_stop"),
                            &json!({ "type": "content_block_stop", "index": block_index }),
                        )?;
                        block_index += 1;
                        thinking_started = false;
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
            }
            Reply::Done {
                finish,
                tool_calls,
                completion_tokens,
                ..
            } => {
                if text_started || thinking_started {
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
                sse_event(
                    stream,
                    Some("message_stop"),
                    &json!({ "type": "message_stop" }),
                )?;
                write_chunk(stream, b"")?;
                break;
            }
            Reply::Refused {
                retry_after_s,
                reason,
            } => {
                sse_event(
                    stream,
                    Some("error"),
                    &json!({ "type": "error", "error": { "type": "insufficient_memory", "message": reason, "retry_after_s": retry_after_s } }),
                )?;
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

pub(super) fn stream_responses(
    api: &ApiState,
    stream: &mut TcpStream,
    rx: Receiver<Reply>,
    req_body: &Value,
) -> anyhow::Result<()> {
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
            "response": empty_response_resource(api, &id, "in_progress"),
        }),
    )?;
    let item_id = format!("msg_{:x}", now());
    let reasoning_item_id = format!("rs_{:x}", now());
    let mut reasoning_started = false;
    let mut text_started = false;
    for reply in rx {
        match reply {
            Reply::Start { .. } => {}
            Reply::Delta {
                text, reasoning, ..
            } => {
                if let Some(reasoning) = reasoning.filter(|r| !r.is_empty()) {
                    if !reasoning_started {
                        reasoning_started = true;
                        sse_event(
                            stream,
                            None,
                            &json!({
                                "type": "response.output_item.added",
                                "sequence_number": next_seq(),
                                "output_index": 0,
                                "item": {
                                    "type": "reasoning", "id": reasoning_item_id,
                                    "status": "in_progress", "summary": []
                                }
                            }),
                        )?;
                        sse_event(
                            stream,
                            None,
                            &json!({
                                "type": "response.reasoning_summary_part.added",
                                "sequence_number": next_seq(),
                                "item_id": reasoning_item_id, "output_index": 0,
                                "summary_index": 0,
                                "part": { "type": "summary_text", "text": "" }
                            }),
                        )?;
                    }
                    sse_event(
                        stream,
                        None,
                        &json!({
                            "type": "response.reasoning_summary_text.delta",
                            "sequence_number": next_seq(),
                            "item_id": reasoning_item_id, "output_index": 0,
                            "summary_index": 0, "delta": reasoning
                        }),
                    )?;
                }
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
                            "output_index": usize::from(reasoning_started),
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
                            "item_id": item_id, "output_index": usize::from(reasoning_started), "content_index": 0,
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
                        "item_id": item_id, "output_index": usize::from(reasoning_started), "content_index": 0,
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
                cached_tokens,
                finish,
                logprobs,
                stop_sequence,
            } => {
                let d = ReplyData {
                    content: content.clone(),
                    reasoning: reasoning.clone(),
                    tool_calls: tool_calls.clone(),
                    prompt_tokens,
                    completion_tokens,
                    cached_tokens,
                    finish: finish.clone(),
                    logprobs,
                    stop_sequence,
                };
                if reasoning_started {
                    let text = reasoning.as_deref().unwrap_or("");
                    sse_event(
                        stream,
                        None,
                        &json!({
                            "type": "response.reasoning_summary_text.done",
                            "sequence_number": next_seq(),
                            "item_id": reasoning_item_id, "output_index": 0,
                            "summary_index": 0, "text": text
                        }),
                    )?;
                    sse_event(
                        stream,
                        None,
                        &json!({
                            "type": "response.reasoning_summary_part.done",
                            "sequence_number": next_seq(),
                            "item_id": reasoning_item_id, "output_index": 0,
                            "summary_index": 0,
                            "part": { "type": "summary_text", "text": text }
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
                                "type": "reasoning", "id": reasoning_item_id,
                                "status": "completed",
                                "summary": [{ "type": "summary_text", "text": text }]
                            }
                        }),
                    )?;
                }
                if text_started {
                    sse_event(
                        stream,
                        None,
                        &json!({
                            "type": "response.output_text.done",
                            "sequence_number": next_seq(),
                            "item_id": item_id, "output_index": usize::from(reasoning_started), "content_index": 0,
                            "text": content
                        }),
                    )?;
                    sse_event(
                        stream,
                        None,
                        &json!({
                            "type": "response.content_part.done",
                            "sequence_number": next_seq(),
                            "item_id": item_id, "output_index": usize::from(reasoning_started), "content_index": 0,
                            "part": { "type": "output_text", "text": content, "annotations": [] }
                        }),
                    )?;
                    sse_event(
                        stream,
                        None,
                        &json!({
                            "type": "response.output_item.done",
                            "sequence_number": next_seq(),
                            "output_index": usize::from(reasoning_started),
                            "item": {
                                "type": "message", "id": item_id, "status": "completed",
                                "role": "assistant",
                                "content": [{ "type": "output_text", "text": content, "annotations": [] }]
                            }
                        }),
                    )?;
                }
                let base = usize::from(reasoning_started) + usize::from(text_started);
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
                let payload = responses_payload_with_id(api, &d, &id);
                store_conversation(api, &id, req_body, &d, &payload);
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
            Reply::Refused {
                retry_after_s,
                reason,
            } => {
                sse_event(
                    stream,
                    None,
                    &json!({ "type": "error", "error": { "type": "insufficient_memory", "message": reason, "retry_after_s": retry_after_s } }),
                )?;
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

/// Retain the completed resource and its normalized continuation history together.
pub(super) fn store_conversation(
    api: &ApiState,
    id: &str,
    req_body: &Value,
    d: &ReplyData,
    payload: &Value,
) {
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
    api.store(id, payload.clone(), Some(conv));
}
