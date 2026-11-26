//! Chat templates, prompt rendering and reasoning/content stream sections.
use super::*;

/// One parsed chat message. `reasoning_content` carries the assistant's
/// thinking (separate from `content`) so the checkpoint template can render the
/// canonical ` thinking… response` block. `tool_calls`/`tool_call_id` and raw
/// multimodal `parts` round-trip the OpenAI tool/vision fields.
#[derive(Clone, PartialEq)]
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
    pub(super) inner: hf_chat_template::ChatTemplate,
    pub(super) deepseek41: bool,
    pub(super) olala: bool,
    pub(super) inkling2: bool,
}

impl ChatTemplate {
    /// Parse model control channels and declared tool calls before display cleanup.
    pub fn parse_reply(
        &self,
        text: &str,
        tools: &[Value],
        thinking: bool,
    ) -> (String, Option<String>, Vec<Value>) {
        template_reply(text, tools, true, &[], thinking, Some(self))
    }
    pub fn uses_control_channels(&self) -> bool {
        self.olala || self.inkling2
    }

    /// Visible response sections, with incomplete control markers held back.
    pub fn streaming_sections<'a>(&self, text: &'a str, thinking: bool) -> (&'a str, &'a str) {
        template_sections(text, thinking, Some(self))
    }

    pub fn new(src: &str) -> anyhow::Result<Self> {
        let inner = hf_chat_template::ChatTemplate::from_str(src)
            .map_err(|e| anyhow::anyhow!("chat template parse: {e}"))?;
        Ok(Self {
            inner,
            deepseek41: src.contains("<｜System｜>") && src.contains("reasoning_effort"),
            olala: src.contains("<|channel_start|>"),
            inkling2: src.contains("<|message_model|>"),
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
                    if let Some(f) = c.get_mut("function")
                        && let Some(a) = f.get("arguments").and_then(|v| v.as_str())
                    {
                        let parsed: Value = serde_json::from_str(a).unwrap_or_else(|_| json!({}));
                        f["arguments"] = parsed;
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
            json!(if self.olala {
                if thinking { "medium" } else { "disabled" }
            } else if self.inkling2 {
                if thinking { "high" } else { "none" }
            } else if self.deepseek41 {
                "high"
            } else {
                "xhigh"
            }),
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
        // when enable_thinking=false. The reference omits the reasoning-effort system
        // message entirely for a no-thinking request; leaving it here changes
        // the prompt tokens and can make the model emit private reasoning as
        // visible content. Match the reference's GLM prompt contract while preserving
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

/// Match the reference's GLM tool-schema serialization. The checkpoint Jinja template
/// uses Python-style `tojson` spacing inside nested values, while the reference emits
/// compact nested JSON and preserves only the spaces between top-level fields.
pub(super) fn ds4_glm_tool_schema_json(tool: &Value) -> Option<String> {
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

pub(super) fn normalize_glm_tool_prompt(rendered: &mut String, tools: &[Value]) {
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
pub(super) fn render_prompt(
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

pub(super) fn session_suffix_text(full_prompt: &str, rest: &str) -> String {
    if full_prompt.starts_with("[gMASK]<sop>")
        || full_prompt.starts_with("<|endoftext|><|im_start|>system<|channel_start|>")
    {
        rest.to_string()
    } else {
        format!("\n{rest}")
    }
}

/// Close the ` thinking` block a generation prompt opened, splitting the model's
/// raw output into (reasoning, answer). Falls back to all-content.
pub(super) fn split_reasoning(text: &str) -> (Option<String>, String) {
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

/// Incremental response sections for SSE, following the reference's request-selected
/// parser mode. With thinking disabled, raw generated text stays in content;
/// with thinking enabled, text inside the think block goes to reasoning.
pub(super) fn streaming_sections(text: &str, thinking_enabled: bool) -> (&str, &str) {
    const OPEN: &str = "\u{3c}think\u{3e}";
    const CLOSE: &str = "\u{3c}/think\u{3e}";
    // The reference selects the streaming parser mode from the request's thinking
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
pub(super) fn streaming_reply_sections(
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
pub(super) fn split_reasoning_for_request(
    text: &str,
    thinking_enabled: bool,
) -> (Option<String>, String) {
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
pub(super) fn stream_delta(prev: &str, cur: &str) -> Option<String> {
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
/// text and finish in a later token. This mirrors the reference's byte-based streaming
/// holdback while ensuring the returned Rust string slice ends on a UTF-8
/// boundary.
pub(super) fn streaming_stop_safe_len(text: &str, stops: &[String]) -> usize {
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

pub(super) fn template_sections<'a>(
    text: &'a str,
    thinking: bool,
    template: Option<&ChatTemplate>,
) -> (&'a str, &'a str) {
    if template.is_some_and(|t| t.inkling2) {
        inkling2::sections(text, thinking)
    } else if template.is_some_and(|t| t.olala) {
        olala::sections(text, thinking)
    } else {
        streaming_sections(text, thinking)
    }
}

pub(super) fn template_stream_reply(
    text: &str,
    thinking: bool,
    stops: &[String],
    template: Option<&ChatTemplate>,
) -> (String, String) {
    if template.is_some_and(|t| t.uses_control_channels()) {
        let (r, c) = template_sections(strip_stops(text, stops), thinking, template);
        (r.into(), c.into())
    } else {
        streaming_reply_sections(text, thinking, stops)
    }
}

pub(super) fn template_reply(
    text: &str,
    tools: &[Value],
    parallel: bool,
    stops: &[String],
    thinking: bool,
    template: Option<&ChatTemplate>,
) -> (String, Option<String>, Vec<Value>) {
    if template.is_some_and(|t| t.inkling2) {
        inkling2::reply(strip_stops(text, stops), thinking, tools, parallel)
    } else if template.is_some_and(|t| t.olala) {
        olala::reply(strip_stops(text, stops), thinking, tools, parallel)
    } else {
        shape_reply(text, tools, parallel, stops, thinking)
    }
}
