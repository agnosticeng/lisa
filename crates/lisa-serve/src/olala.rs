//! Olala channel framing. Called only for the checkpoint's channel template.
use serde_json::{Value, json};
const START: &str = "<|channel_start|>";
const END: &str = "<|channel_end|>";
const CONTENT: &str = "<|content|>";
fn safe(text: &str) -> &str {
    let mut end = text.len();
    for marker in [START, END, "<|im_end|>"] {
        if let Some(i) = text.find(marker) {
            end = end.min(i);
        }
        for n in 1..marker.len() {
            if text.ends_with(&marker[..n]) {
                end = end.min(text.len() - n);
            }
        }
    }
    &text[..end]
}
fn channel<'a>(text: &'a str, name: &str) -> Option<&'a str> {
    let marker = format!("{START}{name}{CONTENT}");
    text.find(&marker).map(|i| safe(&text[i + marker.len()..]))
}
/// The generation prompt opens analysis (or closes it immediately when disabled).
/// Partial control markers are held back until their meaning is known.
pub(crate) fn sections(text: &str, thinking: bool) -> (&str, &str) {
    let reasoning = if thinking {
        channel(text, "analysis").unwrap_or_else(|| safe(text))
    } else {
        ""
    };
    let answer = channel(text, "final").unwrap_or("");
    (reasoning, answer)
}
pub(crate) fn reply(
    text: &str,
    thinking: bool,
    tools: &[Value],
    parallel: bool,
) -> (String, Option<String>, Vec<Value>) {
    let (reasoning, content) = sections(text, thinking);
    let mut calls = vec![];
    if let Some(raw) = channel(text, "tools") {
        let mut rest = raw;
        while let Some(start) = rest.find("<call ") {
            rest = &rest[start + 6..];
            let Some((attrs, body)) = rest.split_once('>') else {
                break;
            };
            let Some((body, tail)) = body.split_once("</call>") else {
                break;
            };
            rest = tail;
            let tag = |name: &str| {
                body.split_once(&format!("<{name}>"))?
                    .1
                    .split_once(&format!("</{name}>"))
                    .map(|(v, _)| v)
            };
            let (Some(name), Some(arguments)) = (tag("name"), tag("arguments")) else {
                continue;
            };
            if !tools
                .iter()
                .any(|t| t["function"]["name"].as_str() == Some(name))
            {
                continue;
            }
            let Ok(arguments) = serde_json::from_str::<Value>(arguments) else {
                continue;
            };
            if !arguments.is_object() {
                continue;
            }
            let id = attrs
                .split_once("id=\"")
                .and_then(|(_, v)| v.split_once('"'))
                .map(|(v, _)| v.to_owned())
                .unwrap_or_else(|| format!("call_{}", calls.len()));
            calls.push(json!({"id":id,"type":"function","function":{"name":name,"arguments":arguments.to_string()}}));
            if !parallel {
                break;
            }
        }
    }
    (
        content.to_owned(),
        thinking.then(|| reasoning.to_owned()),
        calls,
    )
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn session_suffix_keeps_channel_bytes_unchanged() {
        let prefix = "<|endoftext|><|im_start|>system<|channel_start|>instructions<|content|>";
        let suffix = "<|im_end|><|im_start|>user<|channel_start|>text<|content|>Next";
        assert_eq!(super::super::session_suffix_text(prefix, suffix), suffix);
    }
    #[test]
    fn checkpoint_template_renders_channels_and_tool_history() {
        let template =
            super::super::ChatTemplate::new(include_str!("fixtures/olala.jinja")).unwrap();
        let msg = |role: &str, content: &str| super::super::ChatMessage {
            role: role.into(),
            content: content.into(),
            reasoning_content: None,
            tool_calls: vec![],
            tool_call_id: None,
            parts: None,
        };
        let mut messages = vec![msg("system", "Be concise."), msg("user", "Weather?")];
        let disabled = template.render(&messages, true, &[], false).unwrap();
        assert!(disabled.contains("Reasoning effort: disabled"));
        assert!(disabled.contains("user<|channel_start|>text<|content|>Weather?"));
        assert!(disabled.ends_with("assistant<|channel_start|>analysis<|content|><|channel_end|>"));
        let enabled = template.render(&messages, true, &[], true).unwrap();
        assert!(enabled.contains("Reasoning effort: medium"));
        assert!(enabled.ends_with("assistant<|channel_start|>analysis<|content|>"));
        let mut call = msg("assistant", "");
        call.tool_calls = vec![
            json!({"id":"call_7","type":"function","function":{"name":"weather","arguments":"{\"city\":\"Paris\"}"}}),
        ];
        messages.push(call);
        let mut result = msg("tool", "Sunny");
        result.tool_call_id = Some("call_7".into());
        messages.push(result);
        let rendered = template.render(&messages, true, &[], false).unwrap();
        assert!(rendered.contains("<call id=\"call_7\"><name>weather</name><arguments>{"));
        assert!(rendered.contains("tool_output<|content|>Sunny"));
        assert!(rendered.contains("call_7"));
    }
    #[test]
    fn channels_never_stream_markers_or_tool_payloads() {
        let text = "Réfléchir🙂<|channel_end|><|channel_start|>final<|content|>Bonjour🙂<|channel_end|><|channel_start|>tools<|content|><toolcalls><call id=\"call_7\"><name>weather</name><arguments>{\"city\":\"Paris\"}</arguments></call></toolcalls><|channel_end|>";
        let mut last = (String::new(), String::new());
        for end in 0..=text.len() {
            if !text.is_char_boundary(end) {
                continue;
            }
            let (r, c) = sections(&text[..end], true);
            assert!(r.starts_with(&last.0));
            assert!(c.starts_with(&last.1));
            assert!("Réfléchir🙂".starts_with(r));
            assert!("Bonjour🙂".starts_with(c));
            last = (r.into(), c.into());
        }
        let tools = vec![json!({"type":"function","function":{"name":"weather"}})];
        let (c, r, calls) = reply(text, true, &tools, true);
        assert_eq!(c, "Bonjour🙂");
        assert_eq!(r.as_deref(), Some("Réfléchir🙂"));
        assert_eq!(calls[0]["id"], "call_7");
        assert_eq!(calls[0]["function"]["arguments"], "{\"city\":\"Paris\"}");
        assert!(reply(text, true, &[], true).2.is_empty());
    }
    #[test]
    fn disabled_and_truncated_channels_are_hidden() {
        for text in ["", "<|channel_start|>", "<|channel_start|>final<|cont"] {
            assert_eq!(sections(text, false), ("", ""));
        }
        assert_eq!(
            sections("<|channel_start|>final<|content|>Hi<|channel_en", false),
            ("", "Hi")
        );
    }
}
