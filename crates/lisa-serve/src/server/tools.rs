//! Tool-call extraction, JSON salvage and DSML decode state.
use super::*;

/// Split a generated reply into visible content and OpenAI-style `tool_calls`,
/// parsing the Qwen/MiMo text form
/// `<tool_call><function=NAME><parameter=K>V</parameter>…</function></tool_call>`.
/// Complete bare function blocks are also accepted when their tools are declared.
/// Any reasoning block is stripped first.
pub fn extract_tool_calls(text: &str, tools: &[Value]) -> (String, Vec<Value>) {
    let (_, answer) = split_reasoning(text);
    if let Some(result) = deepseek41::extract(&answer) {
        return result.unwrap_or_else(|_| (answer, Vec::new()));
    }
    const OPEN: &str = "<tool_call>";
    const CLOSE: &str = "</tool_call>";
    let types = tool_types(tools);
    let Some(start) = answer.find(OPEN) else {
        if let Some(calls) = parse_bare_tool_calls(&answer, &types) {
            return (String::new(), calls);
        }
        return (answer, Vec::new());
    };
    let content = answer[..start].trim().to_string();
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

/// A missing outer wrapper is recoverable only for a reply consisting entirely
/// of complete calls to declared tools. Prose and code examples remain text.
pub(super) fn parse_bare_tool_calls(
    answer: &str,
    types: &std::collections::HashMap<String, std::collections::HashMap<String, String>>,
) -> Option<Vec<Value>> {
    let mut rest = answer.trim();
    while let Some(prefix) = rest
        .strip_suffix("<|im_end|>")
        .or_else(|| rest.strip_suffix("<|endoftext|>"))
    {
        rest = prefix.trim_end();
    }
    let mut calls = Vec::new();
    while !rest.is_empty() {
        if !rest.starts_with("<function=") {
            return None;
        }
        let end = rest.find("</function>")? + "</function>".len();
        let call = parse_tool_call(&rest[..end], types);
        let name = call.get("function")?.get("name")?.as_str()?;
        if !types.contains_key(name) {
            return None;
        }
        calls.push(call);
        rest = rest[end..].trim_start();
    }
    (!calls.is_empty()).then_some(calls)
}

/// Per-tool parameter type map (`name -> {param -> "string"|"integer"|…}`) from
/// the request's JSON-schema tool definitions, used to type arguments.
pub(super) fn tool_types(
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
pub(super) fn parse_tool_call(
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
        while let Some(start) = rest.find("<arg_key>") {
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
    if name.is_empty()
        && let Some(obj) = find_json_object(body)
    {
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
pub(super) fn salvage_json_object(s: &str) -> Option<serde_json::Map<String, Value>> {
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
pub(super) fn find_json_object(s: &str) -> Option<serde_json::Map<String, Value>> {
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

pub(super) fn tool_call_id() -> String {
    static N: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let n = N.fetch_add(1, Ordering::Relaxed);
    format!("call_{:x}{:x}", now(), n)
}

/// Whether the streamed/generated text has opened a tool call (so further text
/// must not be streamed as assistant content).
pub(super) fn tool_call_start(text: &str) -> Option<usize> {
    text.find("<tool_call>")
        .into_iter()
        .chain(deepseek41::start(text))
        .min()
}

/// Match the reference's live GLM/Qwen DSML sampling boundary. Structural tool syntax is
/// sampled greedily; parameter bodies use the request's configured sampler.
#[cfg(test)]
pub(super) fn dsml_sampling_mode(output: &str, thinking_enabled: bool) -> TargetMtpSamplingMode {
    match dsml_decode_state(output, thinking_enabled) {
        DsmlDecodeState::Greedy => TargetMtpSamplingMode::Greedy,
        DsmlDecodeState::Configured | DsmlDecodeState::Done => TargetMtpSamplingMode::Configured,
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum DsmlDecodeState {
    Configured,
    Greedy,
    Done,
}

#[cfg(test)]
pub(super) fn dsml_decode_state(output: &str, thinking_enabled: bool) -> DsmlDecodeState {
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
