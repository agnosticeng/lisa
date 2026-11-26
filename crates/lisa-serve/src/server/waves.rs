//! Speculative depth selection, batched waves and persistent sessions.
use super::*;

pub(super) fn depth_for(job: &Job, cfg: &ServerConfig) -> usize {
    job.body
        .get("depth")
        .and_then(|v| v.as_u64())
        .map(|v| v as usize)
        .unwrap_or(cfg.depth)
}

/// Map a configured depth to the generation policy: the `auto` sentinel
/// becomes the EV controller when the tower ships a drafter, fixed otherwise.
pub(super) fn mtp_depth(
    depth: usize,
    tower: &dyn LanguageModel<crate::application::LanguageRuntime>,
) -> MtpDepth {
    configured_mtp_depth(depth, tower.has_drafter())
}

pub(super) fn configured_mtp_depth(depth: usize, has_drafter: bool) -> MtpDepth {
    if depth == lisa_engine::core::round_cost::AUTO_DEPTH && has_drafter {
        MtpDepth::Auto
    } else {
        MtpDepth::Fixed(depth)
    }
}

/// Resolve the request's draft depth, rejecting depth 1 (a single draft never
/// pays for a speculative round; use 0 or at least 2).
pub(super) fn resolve_depth(job: &Job, cfg: &ServerConfig) -> anyhow::Result<usize> {
    let depth = depth_for(job, cfg);
    anyhow::ensure!(
        depth != 1,
        "depth 1 is not supported; use 0 (serial) or at least 2 (MTP)"
    );
    Ok(depth)
}

/// Build the scheduler request for a stateless HTTP job.
pub(super) fn build_request(
    id: usize,
    tok: &Tokenizer,
    job: &Job,
    cfg: &ServerConfig,
    tmpl: Option<&ChatTemplate>,
) -> anyhow::Result<crate::application::Request> {
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
    Ok(crate::application::Request {
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
pub(super) fn is_plain_job(job: &Job, cfg: &ServerConfig) -> bool {
    depth_for(job, cfg) == 0 && is_wave_eligible(job)
}

/// GLM target-MTP can join the speculative scheduler wave unless request
/// semantics require the serial completion path.
pub(super) fn is_target_mtp_wave_eligible(job: &Job, depth: usize) -> bool {
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
pub(super) fn is_wave_eligible(job: &Job) -> bool {
    !lisa_engine::vision::messages_have_images(&job.body)
        && !has_tools(job)
        && job.surface == Surface::Chat
        && job.session_id.is_none()
        && request_stops(job).is_empty()
        && !wants_logprobs(&job.body)
        && response_format_of(&job.body).is_none()
}

/// Whether a request asked for per-token logprobs.
pub(super) fn wants_logprobs(body: &Value) -> bool {
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
pub(super) fn run_wave(
    tower: &mut dyn LanguageModel<crate::application::LanguageRuntime>,
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
    let mut reqs: Vec<crate::application::Request> = Vec::with_capacity(jobs.len());
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
    let mut next = || -> Option<crate::application::Request> {
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
            let (reasoning, content) = template_sections(text, thinking, tmpl);
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
        lisa_engine::scheduling::run_streaming(tower, reqs, cfg.max_batch, &mut emit, &mut next)?;
    } else {
        lisa_engine::scheduling::run_speculative_streaming(
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
        let (reasoning, content) = if tmpl.is_some_and(|t| t.uses_control_channels()) {
            let (content, reasoning, _) = template_reply(&full, &[], true, &[], thinking, tmpl);
            (reasoning, content)
        } else {
            split_reasoning_for_request(&full, thinking)
        };
        if stream_flags.borrow()[id] {
            // Reconcile against the same request-mode parser used while
            // streaming. Completed-response shaping may classify a late
            // `</think>` marker differently; a stream cannot retract text
            // already sent in the reference's text mode.
            let (stream_reasoning, stream_content) =
                template_stream_reply(&full, thinking, &[], tmpl);
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
pub(super) fn run_session(
    tower: &mut dyn LanguageModel<crate::application::LanguageRuntime>,
    tok: &Tokenizer,
    sessions: &mut HashMap<String, StoredSession>,
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
        .filter(|stored| stored.session.is_valid() && stored.compatible(&tools, thinking))
        .map(|stored| (stored.session, stored.messages))
        .unwrap_or_else(|| {
            let mut s = Session::new(tower);
            s.enable_snapshots();
            (s, Vec::new())
        });
    sess.ensure_valid()?;
    // Cross-request prefix cache: a brand-new session whose prompt extends a
    // cached prompt token-exactly resumes at the split point.
    let ids_full = tok.encode(&new_str, false)?;
    if sess.fed.is_empty()
        && prefix.enabled()
        && let Some((l, restored)) = prefix.restore_session(tower, &ids_full)
    {
        eprintln!(
            "[serve] session {sid}: prefix-cache resume {l}/{}",
            ids_full.len()
        );
        sess = restored;
        sess.enable_snapshots();
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
        if diverges && sess.restore_snapshot_exact(tower, k).is_none() {
            let mut s = Session::new(tower);
            s.enable_snapshots();
            sess = s;
            None
        } else {
            render_prompt(tmpl, &messages[..k], false, &tools, thinking)
                .ok()
                .and_then(|prefix| new_str.strip_prefix(&prefix).map(|r| r.to_string()))
                .filter(|r| !r.is_empty())
                // The canonical GLM template has no newline between turns, and
                // the reference's rendered prompt reuses that exact boundary. Other model
                // templates can leave a closing assistant newline outside the
                // cached token prefix, so preserve the historical separator for
                // those formats only.
                .map(|rest| {
                    let suffix = session_suffix_text(&new_str, &rest);
                    tok.encode(&suffix, false)
                })
                .transpose()?
        }
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
    let greedy = can_omit_configured_sampler(&sampler, tower.has_target_mtp());
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
    let _eos_scope = ignore_eos.then(lisa_engine::generation::ignore_eos_scope);
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
        if !stops.is_empty()
            && stops
                .iter()
                .any(|s| !s.is_empty() && text.contains(s.as_str()))
        {
            hit_stop.set(true);
            if streaming {
                // The reference emits the safe prefix before ending the stream when
                // a text stop is found. The normal holdback path runs
                // after this check, so flush the trimmed prefix here.
                let (reasoning, content) = template_stream_reply(text, thinking, &stops, tmpl);
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
        if streaming {
            let stable = decoded.stable_text();
            let safe_len = streaming_stop_safe_len(stable, &stops);
            let (reasoning, answer) = template_sections(&stable[..safe_len], thinking, tmpl);
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
    let mut after_prefill = |caches: &[crate::application::LayerCache]| {
        let cache_law = explicit_cache_law.unwrap_or_else(crate::application::kernel_law);
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
    let (content, reasoning, tool_calls) =
        template_reply(&full, &tools, parallel, &stops, thinking, tmpl);
    if streaming {
        // Completed-response shaping can reinterpret a late `</think>`
        // marker, but an SSE stream cannot retract its earlier text-mode
        // deltas. Reconcile with the incremental parser's final state.
        let (stream_reasoning, stream_content) =
            template_stream_reply(&full, thinking, &stops, tmpl);
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
        sessions.insert(
            sid,
            StoredSession {
                session: sess,
                messages: conversation,
                tools,
                thinking,
            },
        );
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

pub(super) struct StoredSession {
    session: Session,
    messages: Vec<ChatMessage>,
    tools: Vec<Value>,
    thinking: bool,
}

pub(super) fn session_render_options_match(
    old_tools: &[Value],
    old_thinking: bool,
    tools: &[Value],
    thinking: bool,
) -> bool {
    old_tools == tools && old_thinking == thinking
}

impl StoredSession {
    fn compatible(&self, tools: &[Value], thinking: bool) -> bool {
        session_render_options_match(&self.tools, self.thinking, tools, thinking)
    }
}

pub(super) fn structural_prefix(a: &[ChatMessage], b: &[ChatMessage]) -> usize {
    let n = a.len().min(b.len());
    let mut i = 0;
    while i < n && a[i] == b[i] {
        i += 1;
    }
    i
}

pub(super) fn common_prefix(a: &[u32], b: &[u32]) -> usize {
    let n = a.len().min(b.len());
    let mut i = 0;
    while i < n && a[i] == b[i] {
        i += 1;
    }
    i
}
