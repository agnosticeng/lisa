//! Serial cache-aware completion, streaming emission and embeddings.
use super::*;

/// The prompt tokens still to feed: everything after the resumed prefix-cache
/// boundary and after the two-stage prefill boundary.
pub(super) fn unfed_prefix(
    ids: &[u32],
    resumed: Option<&usize>,
    prefeed: &Option<Vec<u32>>,
) -> Vec<u32> {
    let skip = resumed
        .copied()
        .unwrap_or(0)
        .max(prefeed.as_ref().map_or(0, |p| p.len()));
    ids[skip..].to_vec()
}

pub(super) fn complete(
    tower: &mut dyn LanguageModel<crate::application::LanguageRuntime>,
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

pub(super) fn complete_inner(
    tower: &mut dyn LanguageModel<crate::application::LanguageRuntime>,
    tok: &Tokenizer,
    job: &Job,
    cfg: &ServerConfig,
    prefix: &mut PrefixCache,
    tmpl: Option<&ChatTemplate>,
) -> anyhow::Result<()> {
    // Raw completions have no channel prompt and must remain raw text.
    let raw_olala =
        job.surface == Surface::Completions && tmpl.is_some_and(|t| t.uses_control_channels());
    let tmpl = if raw_olala { None } else { tmpl };
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
    let _eos_scope = ignore_eos.then(lisa_engine::generation::ignore_eos_scope);
    // Raw completions never get a think-prompt (parity with the old
    // surface-gated resolution); explicit enable_thinking there is ignored.
    let thinking = job.surface != Surface::Completions && request_thinking(&job.body, cfg);
    // Anthropic semantics: a trailing assistant message is a prefill. Render the
    // generation header, then the partial assistant text, and let the model
    // continue it (rather than closing the turn and starting a new one).
    let prompt = if raw_olala {
        messages
            .iter()
            .map(|m| m.content.as_str())
            .collect::<String>()
    } else if job.surface == Surface::Messages
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
    application::trace_http("serve.http.prompt_ready");
    let images = lisa_engine::vision::decode_message_images(&job.body)?;
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
    let greedy = can_omit_configured_sampler(&sampler, tower.has_target_mtp());
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
    // Save an immutable prefix while retaining the original final prefill
    // chunk. Strict extension lookup needs at least one fresh token for logits;
    // admitted batched models also require aligned chunk boundaries.
    if !has_images
        && prefix.enabled()
        && ids.len() > lisa_engine::conversation::cache::MIN_PREFIX_TOKENS
    {
        let grain = tower.prefix_cache_granularity().unwrap_or(1).max(1);
        let split = (ids.len() - 1) / grain * grain;
        let pos = resumed.unwrap_or(0);
        if pos < split {
            // Mid-prompt boundary only past the natural prefill-chunk size,
            // and ON a chunk multiple: feeding [..b] then the rest dispatches
            // exactly the windows a plain chunked prefill would, so the cache
            // never changes the computed values.
            if ids.len() > 2048 {
                let b = (ids.len() / 2048) * 2048;
                if b > pos && b <= split && b % grain == 0 {
                    session.feed(tower, &ids[pos..b])?;
                    session.ensure_valid()?;
                    prefix.insert_model(tower, &ids[..b], &session.caches);
                    prefeed = Some(ids[..b].to_vec());
                }
            }
            let start = prefeed.as_ref().map_or(pos, |p| p.len());
            if start < split {
                session.feed(tower, &ids[start..split])?;
                session.ensure_valid()?;
                prefix.insert_model(tower, &ids[..split], &session.caches);
                prefeed = Some(ids[..split].to_vec());
            }
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
                if collected.len() == 1 {
                    application::trace_http("serve.http.first_token");
                }
                let text = decoded.push(t)?;
                if !stops.is_empty()
                    && stops
                        .iter()
                        .any(|s| !s.is_empty() && text.contains(s.as_str()))
                {
                    hit_stop.set(true);
                    if streaming && json_schema.is_none() {
                        let visible = strip_stops(text, &stops);
                        let (reasoning, answer) = template_sections(visible, thinking, tmpl);
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
            if streaming && json_schema.is_none() {
                let text = decoded.stable_text();
                let safe_len = streaming_stop_safe_len(text, &stops);
                let (reasoning, answer) = template_sections(&text[..safe_len], thinking, tmpl);
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
            let mut logits = session.feed_for_generation(tower, &gen_ids)?;
            let mut sampling_mode = TargetMtpSamplingMode::Configured;
            while out.len() < max_tokens {
                let t = if dsml_tools && sampling_mode == TargetMtpSamplingMode::Greedy {
                    Sampler::default().draw_host(tower.runtime(), &logits, &[])?
                } else if tower.uses_host_sampling() {
                    sampler.draw_host(tower.runtime(), &logits, &session.fed)?
                } else {
                    sampler.draw(tower.runtime(), &logits, &session.fed)?
                };
                let entry = json!({
                    "token": tok.decode(&[t]).unwrap_or_default(),
                    "logprob": token_logprob(tower.runtime(), &logits, t)?,
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
                session.generate(tower, &gen_ids, max_tokens, &mut sampler, None, &mut |t| {
                    emit(t, None)
                })
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
        let body = if tmpl.is_some_and(|t| t.uses_control_channels()) {
            template_sections(raw, thinking, tmpl).1
        } else {
            strip_think_prefix(raw)
        };
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
            let retry =
                session.generate(tower, &retry_ids, max_tokens, &mut sampler, None, &mut noop);
            if let Ok(retry) = retry {
                let retry_text = tok.decode(&retry)?;
                let raw = strip_stops(&retry_text, &stops);
                let rbody = if tmpl.is_some_and(|t| t.uses_control_channels()) {
                    template_sections(raw, thinking, tmpl).1
                } else {
                    strip_think_prefix(raw)
                };
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
        template_reply(&full, &tools, parallel, &stops, thinking, tmpl)
    };
    if streaming && json_schema.is_none() {
        let (stream_reasoning, stream_content) =
            template_stream_reply(&full, thinking, &stops, tmpl);
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
pub(super) fn run_embeddings(
    model_id: &str,
    tower: &mut dyn LanguageModel<crate::application::LanguageRuntime>,
    tok: &Tokenizer,
    job: &Job,
) -> anyhow::Result<()> {
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
        let mut caches = tower.new_caches();
        let (mixed, _) = tower.forward(&crate::application::token_row(&ids), Some(&mut caches))?;
        let host = crate::application::mean_pool(&mixed)?;
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
