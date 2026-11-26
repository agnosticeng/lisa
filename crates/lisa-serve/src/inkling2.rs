//! Inkling message/content framing, including incremental control-token tails.
use serde_json::{Value, json};
const MARKERS: &[&str] = &[
    "<|message_model|>",
    "<|message_user|>",
    "<|message_system|>",
    "<|message_tool|>",
    "<|content_text|>",
    "<|content_thinking|>",
    "<|content_invoke_tool_json|>",
    "<|content_invoke_tool_text|>",
    "<|end_message|>",
    "<|content_model_end_sampling|>",
];
fn safe(text: &str) -> &str {
    let mut end = text.len();
    for marker in MARKERS {
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
fn channel<'a>(text: &'a str, marker: &str) -> &'a str {
    text.split_once(marker)
        .map(|(_, tail)| safe(tail))
        .unwrap_or("")
}
pub(crate) fn sections(text: &str, thinking: bool) -> (&str, &str) {
    (
        if thinking {
            channel(text, "<|content_thinking|>")
        } else {
            ""
        },
        channel(text, "<|content_text|>"),
    )
}
pub(crate) fn reply(
    text: &str,
    thinking: bool,
    tools: &[Value],
    parallel: bool,
) -> (String, Option<String>, Vec<Value>) {
    let (reasoning, content) = sections(text, thinking);
    let mut calls = Vec::new();
    for part in text.split("<|content_invoke_tool_json|>").skip(1) {
        let raw = safe(part);
        let Ok(v) = serde_json::from_str::<Value>(raw) else {
            continue;
        };
        let Some(name) = v["name"].as_str() else {
            continue;
        };
        if !v["args"].is_object()
            || !tools
                .iter()
                .any(|t| t["function"]["name"].as_str() == Some(name))
        {
            continue;
        }
        calls.push(json!({"id":format!("call_{}",calls.len()),"type":"function","function":{"name":name,"arguments":v["args"].to_string()}}));
        if !parallel {
            break;
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
    fn streamed_frames_hide_control_tokens() {
        let raw = "<|message_model|><|content_thinking|>Réfléchir🙂<|end_message|><|message_model|><|content_text|>Bonjour🙂<|end_message|><|content_model_end_sampling|>";
        let mut old = (String::new(), String::new());
        for end in 0..=raw.len() {
            if raw.is_char_boundary(end) {
                let (r, c) = sections(&raw[..end], true);
                assert!(r.starts_with(&old.0));
                assert!(c.starts_with(&old.1));
                assert!("Réfléchir🙂".starts_with(r));
                assert!("Bonjour🙂".starts_with(c));
                old = (r.into(), c.into());
            }
        }
        assert_eq!(old, ("Réfléchir🙂".into(), "Bonjour🙂".into()));
    }
}

#[cfg(test)]
mod template_tests {
    use super::super::{ChatMessage, ChatTemplate};
    use serde_json::json;
    #[test]
    fn checkpoint_template_preserves_images_and_effort() {
        let template = ChatTemplate::new(include_str!("fixtures/inkling2.jinja")).unwrap();
        let message = ChatMessage {
            role: "user".into(),
            content: String::new(),
            reasoning_content: None,
            tool_calls: vec![],
            tool_call_id: None,
            parts: Some(vec![
                json!({"type":"image_url","image_url":{"url":"unused"}}),
                json!({"type":"text","text":"What color?"}),
            ]),
        };
        let text = template.render(&[message], true, &[], false).unwrap();
        assert_eq!(
            text,
            "<|message_system|><|content_text|>Thinking effort level: 0<|end_message|><|message_user|><|content_image|><|unused_200054|><|end_message|><|message_user|><|content_text|>What color?<|end_message|>"
        );
    }
}
