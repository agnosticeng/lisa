//! DS4 V4.1 chat rendering and DSML boundaries, ported from ds4_server.c and
//! ds4_tool_text.h at b3f50180ec2c7324892be1f50bc83555ccb8d5e7 (MIT).
//! Copyright (c) 2026 The ds4.c authors. See LICENSE-DS4 and NOTICE.

use super::{ChatMessage, DsmlDecodeState};
use anyhow::{Result, bail, ensure};
use serde_json::{Map, Value, json};

#[derive(Clone, Copy)]
struct Syntax {
    open: &'static str,
    close: &'static str,
    invoke: &'static str,
    invoke_end: &'static str,
    param: &'static str,
    param_end: &'static str,
}

const SYNTAXES: [Syntax; 4] = [
    Syntax {
        open: "<｜DSML｜ calls>",
        close: "</｜DSML｜ calls>",
        invoke: "<｜DSML｜ invoke",
        invoke_end: "</｜DSML｜ invoke>",
        param: "<｜DSML｜ parameter",
        param_end: "</｜DSML｜ parameter>",
    },
    Syntax {
        open: "<｜DSML｜tool_calls>",
        close: "</｜DSML｜tool_calls>",
        invoke: "<｜DSML｜invoke",
        invoke_end: "</｜DSML｜invoke>",
        param: "<｜DSML｜parameter",
        param_end: "</｜DSML｜parameter>",
    },
    Syntax {
        open: "<DSML｜tool_calls>",
        close: "</DSML｜tool_calls>",
        invoke: "<DSML｜invoke",
        invoke_end: "</DSML｜invoke>",
        param: "<DSML｜parameter",
        param_end: "</DSML｜parameter>",
    },
    Syntax {
        open: "<tool_calls>",
        close: "</tool_calls>",
        invoke: "<invoke",
        invoke_end: "</invoke>",
        param: "<parameter",
        param_end: "</parameter>",
    },
];

fn find_start(text: &str) -> Option<(usize, Syntax)> {
    SYNTAXES
        .iter()
        .filter_map(|&s| text.find(s.open).map(|at| (at, s)))
        .min_by_key(|&(at, _)| at)
}

pub(super) fn start(text: &str) -> Option<usize> {
    find_start(text).map(|(at, _)| at)
}

/// A closing think marker in a parameter is data, not a role transition.
pub(super) fn think_end(text: &str) -> Option<usize> {
    let mut pos = 0;
    while pos < text.len() {
        let tail = &text[pos..];
        if tail.starts_with("</think>") {
            return Some(pos);
        }
        if let Some(close) = SYNTAXES
            .iter()
            .find(|s| tail.starts_with(s.param))
            .map(|s| s.param_end)
            .or_else(|| tail.starts_with("<arg_value>").then_some("</arg_value>"))
            .or_else(|| tail.starts_with("<arg_key>").then_some("</arg_key>"))
        {
            let after = tail.find('>')? + 1;
            pos += after + tail[after..].find(close)? + close.len();
        } else {
            pos += tail.chars().next().unwrap().len_utf8();
        }
    }
    None
}

/// Withhold incomplete protocol openers from SSE content. Do not cut Unicode.
pub(super) fn visible_end(text: &str) -> usize {
    let mut end = start(text).unwrap_or(text.len());
    for marker in SYNTAXES.iter().map(|s| s.open).chain(["<tool_call>"]) {
        if let Some(at) = text.find(marker) {
            end = end.min(at);
        }
        for n in 1..marker.len().min(text.len() + 1) {
            if text.as_bytes().ends_with(&marker.as_bytes()[..n]) {
                end = end.min(text.len() - n);
            }
        }
    }
    // DS4 trims separator whitespace before calls. Keep it pending until a
    // later ordinary character proves it belongs to content; SSE cannot retract.
    while end > 0 && text.as_bytes()[end - 1].is_ascii_whitespace() {
        end -= 1;
    }
    end
}

fn escaped_close(text: &str, close: &str) -> bool {
    let Some(mut tail) = text.strip_prefix('&') else {
        return false;
    };
    while let Some(rest) = tail.strip_prefix("amp;") {
        tail = rest;
    }
    tail.strip_prefix("lt;")
        .is_some_and(|rest| rest.starts_with(&close[1..]))
}

fn body_escape(text: &str, close: &str) -> String {
    let mut out = String::new();
    for (at, ch) in text.char_indices() {
        if text[at..].starts_with(close) {
            out.push_str("&lt;");
        } else if escaped_close(&text[at..], close) {
            out.push_str("&amp;");
        } else {
            out.push(ch);
        }
    }
    out
}

fn body_unescape(mut text: &str, close: &str) -> String {
    let mut out = String::new();
    while !text.is_empty() {
        if escaped_close(text, close) {
            if let Some(rest) = text.strip_prefix("&amp;") {
                out.push('&');
                text = rest;
            } else {
                out.push('<');
                text = &text[4..];
            }
        } else {
            let ch = text.chars().next().unwrap();
            out.push(ch);
            text = &text[ch.len_utf8()..];
        }
    }
    out
}

fn attr_escape(text: &str) -> String {
    text.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

fn attr(tag: &str, key: &str) -> Option<String> {
    let (_, tail) = tag.split_once(&format!("{key}=\""))?;
    let raw = tail.split_once('"')?.0;
    let mut out = String::new();
    let mut tail = raw;
    while !tail.is_empty() {
        if let Some((escape, value)) = [
            ("&amp;", '&'),
            ("&lt;", '<'),
            ("&gt;", '>'),
            ("&quot;", '"'),
            ("&apos;", '\''),
        ]
        .into_iter()
        .find(|(escape, _)| tail.starts_with(escape))
        {
            out.push(value);
            tail = &tail[escape.len()..];
        } else {
            let ch = tail.chars().next().unwrap();
            out.push(ch);
            tail = &tail[ch.len_utf8()..];
        }
    }
    Some(out)
}

fn user_like(m: &ChatMessage) -> bool {
    matches!(m.role.as_str(), "user" | "tool" | "function")
}
fn system(m: &ChatMessage) -> bool {
    matches!(m.role.as_str(), "system" | "developer")
}

fn ordered(messages: &[ChatMessage]) -> Vec<&ChatMessage> {
    let mut out: Vec<_> = messages.iter().collect();
    let mut calls = std::collections::HashMap::new();
    let mut i = 0;
    while i < messages.len() {
        let m = &messages[i];
        if m.role == "assistant" && !m.tool_calls.is_empty() {
            calls.clear();
            for (rank, c) in m.tool_calls.iter().enumerate() {
                if let Some(id) = c.get("id").and_then(Value::as_str) {
                    calls.insert(id, rank);
                }
            }
        }
        if !user_like(m) {
            i += 1;
            continue;
        }
        let begin = i;
        while i < messages.len() && user_like(&messages[i]) {
            i += 1;
        }
        let mut results: Vec<_> = messages[begin..i]
            .iter()
            .filter(|m| m.role != "user")
            .collect();
        results.sort_by_key(|m| {
            m.tool_call_id
                .as_deref()
                .and_then(|id| calls.get(id))
                .copied()
                .unwrap_or(0)
        });
        let mut results = results.into_iter();
        for m in &mut out[begin..i] {
            if m.role != "user" {
                *m = results.next().unwrap();
            }
        }
    }
    out
}

fn append_calls(out: &mut String, calls: &[Value]) -> Result<()> {
    if calls.is_empty() {
        return Ok(());
    }
    let s = SYNTAXES[0];
    out.push_str(&format!("\n\n{}\n", s.open));
    for call in calls {
        let function = call.get("function").unwrap_or(call);
        let name = function
            .get("name")
            .and_then(Value::as_str)
            .unwrap_or_default();
        out.push_str(&format!("{} name=\"{}\">\n", s.invoke, attr_escape(name)));
        let args = function
            .get("arguments")
            .cloned()
            .unwrap_or_else(|| json!({}));
        let args = if let Some(raw) = args.as_str() {
            serde_json::from_str(raw).unwrap_or_else(|_| json!({"arguments": raw}))
        } else {
            args
        };
        let args = args
            .as_object()
            .ok_or_else(|| anyhow::anyhow!("DeepSeek tool arguments must be an object"))?;
        for (key, value) in args {
            let body = if let Some(raw) = value.as_str() {
                body_escape(raw, s.param_end)
            } else {
                value
                    .to_string()
                    .replace(s.param_end, &format!("\\u003c{}", &s.param_end[1..]))
            };
            out.push_str(&format!(
                "{} name=\"{}\" string=\"{}\">{}{}\n",
                s.param,
                attr_escape(key),
                value.is_string(),
                body,
                s.param_end
            ));
        }
        out.push_str(s.invoke_end);
        out.push('\n');
    }
    out.push_str(s.close);
    Ok(())
}

const TOOLS_PREFIX: &str = "## Tools\n\nYou can invoke tools using this format:\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"$TOOL_NAME\">\n<｜DSML｜ parameter name=\"$PARAMETER_NAME\" string=\"true|false\">$PARAMETER_VALUE</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n</｜DSML｜ calls>\n\nString values use string=\"true\"; all other values use JSON and string=\"false\". Inside string values only, escape a literal </｜DSML｜ parameter> as &lt;/｜DSML｜ parameter>. To write that escaped spelling literally, use &amp;lt;/｜DSML｜ parameter>. Other HTML entities are unchanged.\n\nFinish reasoning with </think> before tool calls or a final response.\n\n### Available Tool Schemas\n\n";

pub(super) fn render(
    messages: &[ChatMessage],
    generation: bool,
    tools: &[Value],
    thinking: bool,
) -> Result<String> {
    let mut out = String::from("<｜begin▁of▁sentence｜>");
    let initial_system = thinking || !tools.is_empty();
    if initial_system {
        out.push_str("<｜System｜>");
    }
    if thinking {
        out.push_str("Reasoning Effort: 75 (range 1-100, the higher the value, the more thorough the reasoning)\n\n");
    }
    if !tools.is_empty() {
        out.push_str(TOOLS_PREFIX);
        out.push_str(
            &tools
                .iter()
                .map(|t| t.get("function").unwrap_or(t).to_string())
                .collect::<Vec<_>>()
                .join("\n"),
        );
        out.push_str("\n\nYou MUST strictly follow the above defined tool name and parameter schemas to invoke tool calls. Use the exact parameter names from the schemas.");
    }
    let keep_reasoning = !tools.is_empty()
        || messages
            .iter()
            .any(|m| !m.tool_calls.is_empty() || matches!(m.role.as_str(), "tool" | "function"));
    let last_user = messages
        .iter()
        .enumerate()
        .rfind(|(i, m)| user_like(m) || (*i > 0 && system(m)))
        .map(|(i, _)| i);
    let mut pending = false;
    let mut user_open = false;
    for (i, m) in ordered(messages).into_iter().enumerate() {
        if system(m) {
            if i == 0 && initial_system {
                if !tools.is_empty() {
                    out.push_str("\n\n");
                }
            } else {
                out.push_str("<｜System｜>");
            }
            out.push_str(&m.content);
            pending = i > 0;
            user_open = false;
        } else if user_like(m) {
            out.push_str(if user_open { "\n\n" } else { "<｜User｜>" });
            if m.role == "user" {
                out.push_str(&m.content);
            } else {
                out.push_str("<tool_result>");
                out.push_str(&body_escape(&m.content, "</tool_result>"));
                out.push_str("</tool_result>");
            }
            pending = true;
            user_open = true;
        } else if m.role == "assistant" {
            if pending {
                out.push_str("<｜Assistant｜>");
                if thinking && (keep_reasoning || last_user.is_none_or(|last| i > last)) {
                    out.push_str("<think>");
                    out.push_str(m.reasoning_content.as_deref().unwrap_or(""));
                }
                out.push_str("</think>");
            }
            out.push_str(&m.content);
            append_calls(&mut out, &m.tool_calls)?;
            out.push_str("<｜end▁of▁sentence｜>");
            pending = false;
            user_open = false;
        } else {
            bail!("unsupported DeepSeek message role: {}", m.role)
        }
    }
    if pending && generation {
        out.push_str(if thinking {
            "<｜Assistant｜><think>"
        } else {
            "<｜Assistant｜></think>"
        });
    }
    Ok(out)
}

fn parameter<'a>(text: &'a str, s: Syntax, depth: usize) -> Result<(String, Value, &'a str)> {
    ensure!(depth < 32, "DSML nesting exceeds 32");
    ensure!(text.starts_with(s.param), "expected DSML parameter");
    let (tag, mut tail) = text
        .split_once('>')
        .ok_or_else(|| anyhow::anyhow!("unclosed DSML parameter"))?;
    let key = attr(tag, "name").ok_or_else(|| anyhow::anyhow!("missing DSML parameter name"))?;
    let kind = attr(tag, "string");
    if kind.is_none() && tail.trim_start().starts_with(s.param) {
        let mut args = Map::new();
        while tail.trim_start().starts_with(s.param) {
            let (key, value, next) = parameter(tail.trim_start(), s, depth + 1)?;
            args.insert(key, value);
            tail = next;
        }
        tail = tail.trim_start();
        return Ok((
            key,
            Value::Object(args),
            tail.strip_prefix(s.param_end).unwrap_or(tail),
        ));
    }
    let (raw, tail) = tail
        .split_once(s.param_end)
        .ok_or_else(|| anyhow::anyhow!("missing DSML parameter close"))?;
    let value = if kind.as_deref().unwrap_or("true") == "true" {
        Value::String(body_unescape(raw, s.param_end))
    } else if raw.trim().is_empty() {
        Value::Null
    } else {
        serde_json::from_str(raw)?
    };
    Ok((key, value, tail))
}

/// A malformed or truncated block is never promoted to an executable call.
pub(super) fn extract(answer: &str) -> Option<Result<(String, Vec<Value>)>> {
    let (at, s) = find_start(answer)?;
    Some((|| {
        let mut tail = &answer[at + s.open.len()..];
        let mut calls = Vec::new();
        loop {
            tail = tail.trim_start();
            if tail.starts_with(s.close) {
                return Ok((answer[..at].trim_end().to_string(), calls));
            }
            ensure!(tail.starts_with(s.invoke), "expected DSML invoke");
            let (tag, rest) = tail
                .split_once('>')
                .ok_or_else(|| anyhow::anyhow!("unclosed DSML invoke"))?;
            let name =
                attr(tag, "name").ok_or_else(|| anyhow::anyhow!("missing DSML invoke name"))?;
            tail = rest;
            let mut args = Map::new();
            loop {
                tail = tail.trim_start();
                if let Some(rest) = tail.strip_prefix(s.invoke_end) {
                    tail = rest;
                    break;
                }
                let (key, value, rest) = parameter(tail, s, 0)?;
                args.insert(key, value);
                tail = rest;
            }
            calls.push(json!({"id": super::tool_call_id(), "type":"function", "function":{"name":name,"arguments":Value::Object(args).to_string()}}));
        }
    })())
}

#[derive(Clone, Copy)]
enum Phase {
    Thinking,
    Search,
    Structure,
    String,
    Json,
    Done,
}

pub(super) struct Parser {
    phase: Phase,
    pos: usize,
    syntax: Syntax,
    quoted: bool,
    escaped: bool,
}

impl Parser {
    pub(super) fn new(thinking: bool) -> Self {
        Self {
            phase: if thinking {
                Phase::Thinking
            } else {
                Phase::Search
            },
            pos: 0,
            syntax: SYNTAXES[0],
            quoted: false,
            escaped: false,
        }
    }
    /// Byte cursor matches DS4, including JSON strings and partial closing tags.
    /// A replacement glyph at the decoder's unfinished UTF-8 tail stays pending.
    pub(super) fn update(&mut self, text: &str) -> DsmlDecodeState {
        // The decoder can replace one or two unfinished UTF-8 bytes with this
        // glyph. Keep their preceding marker prefix intact until it settles.
        let text = text.strip_suffix('\u{FFFD}').unwrap_or(text);
        let bytes = text.as_bytes();
        loop {
            let tail = &bytes[self.pos..];
            match self.phase {
                Phase::Done => return DsmlDecodeState::Done,
                Phase::Thinking | Phase::Search => {
                    let from =
                        std::str::from_utf8(tail).expect("search position is a character boundary");
                    let found = if matches!(self.phase, Phase::Thinking) {
                        from.find("</think>")
                            .map(|at| (at + "</think>".len(), None))
                    } else {
                        find_start(from).map(|(at, s)| (at + s.open.len(), Some(s)))
                    };
                    if let Some((after, s)) = found {
                        self.pos += after;
                        if let Some(s) = s {
                            self.syntax = s;
                            self.phase = Phase::Structure;
                        } else {
                            self.phase = Phase::Search;
                        }
                    } else {
                        let keep = SYNTAXES.iter().map(|s| s.open.len()).max().unwrap();
                        self.pos = bytes.len().saturating_sub(keep);
                        while !text.is_char_boundary(self.pos) {
                            self.pos -= 1;
                        }
                        return DsmlDecodeState::Configured;
                    }
                }
                Phase::String | Phase::Json => {
                    let string = matches!(self.phase, Phase::String);
                    let close = self.syntax.param_end.as_bytes();
                    if string || !self.quoted {
                        if tail.starts_with(close) {
                            self.pos += close.len();
                            self.phase = Phase::Structure;
                            continue;
                        }
                        if !tail.is_empty() && close.starts_with(tail) {
                            return if tail.len() >= 2 || !string {
                                DsmlDecodeState::Greedy
                            } else {
                                DsmlDecodeState::Configured
                            };
                        }
                    }
                    if tail.is_empty() || tail == "\u{FFFD}".as_bytes() {
                        return if string || self.quoted {
                            DsmlDecodeState::Configured
                        } else {
                            DsmlDecodeState::Greedy
                        };
                    }
                    let c = bytes[self.pos];
                    self.pos += 1;
                    if !string {
                        if self.quoted {
                            if self.escaped {
                                self.escaped = false;
                            } else if c == b'\\' {
                                self.escaped = true;
                            } else if c == b'"' {
                                self.quoted = false;
                            }
                        } else if c == b'"' {
                            self.quoted = true;
                        }
                    }
                }
                Phase::Structure => {
                    if tail.first().is_some_and(u8::is_ascii_whitespace) {
                        self.pos += 1;
                        continue;
                    }
                    if tail.starts_with(self.syntax.close.as_bytes()) {
                        self.phase = Phase::Done;
                        continue;
                    }
                    if tail.starts_with(self.syntax.invoke_end.as_bytes()) {
                        self.pos += self.syntax.invoke_end.len();
                        continue;
                    }
                    let param = tail.starts_with(self.syntax.param.as_bytes());
                    if param || tail.starts_with(self.syntax.invoke.as_bytes()) {
                        let Some(end) = tail.iter().position(|&c| c == b'>') else {
                            return DsmlDecodeState::Greedy;
                        };
                        if param {
                            let tag =
                                std::str::from_utf8(&tail[..end]).expect("complete tag UTF-8");
                            self.phase = if attr(tag, "string").as_deref() == Some("true") {
                                Phase::String
                            } else {
                                Phase::Json
                            };
                            self.quoted = false;
                            self.escaped = false;
                        }
                        self.pos += end + 1;
                    } else {
                        return DsmlDecodeState::Greedy;
                    }
                }
            }
        }
    }
}

#[cfg(test)]
#[path = "deepseek41_tests.rs"]
mod tests;
