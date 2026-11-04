use super::{
    ChatMessage, ChatTemplate, Job, ReplyData, Surface, dsml_sampling_mode, extract_tool_calls,
    is_target_mtp_wave_eligible, models_payload, normalize_glm_tool_prompt,
    prompt_tokens_details_json, responses_payload_with_id, session_suffix_text,
    split_reasoning_for_request, stream_delta, streaming_reply_sections, streaming_sections,
    streaming_stop_safe_len, usage_json,
};
use lisa_engine::core::session::TargetMtpSamplingMode;
use serde_json::json;

#[test]
fn image_blocks_survive_api_adapters_in_order() {
    let mut anthropic = json!({"messages":[{"role":"user","content":[
        {"type":"text","text":"before"},
        {"type":"image","source":{"type":"base64","media_type":"image/png","data":"abc"}},
        {"type":"text","text":"after"}
    ]}]});
    super::normalize_messages_body(&mut anthropic);
    assert!(lisa_engine::models::vision::messages_have_images(
        &anthropic
    ));
    assert_eq!(
        anthropic["messages"][0]["content"][1]["source"]["data"],
        "abc"
    );
    let mut responses = json!({"input":[{"role":"user","content":[
        {"type":"input_text","text":"before"},
        {"type":"input_image","image_url":"data:image/png;base64,abc"},
        {"type":"input_text","text":"after"}
    ]}]});
    super::normalize_responses_body(&mut responses);
    assert!(lisa_engine::models::vision::messages_have_images(
        &responses
    ));
    let parts = responses["messages"][0]["content"].as_array().unwrap();
    assert_eq!(parts[0]["text"], "before");
    assert_eq!(parts[2]["text"], "after");
    assert_eq!(
        models_payload("vision", true)["data"][0]["architecture"]["input_modalities"],
        json!(["text", "image"])
    );
}

#[test]
fn model_list_reports_text_only_modalities_for_loaded_model() {
    let payload = models_payload("org/glm53-flash", false);
    assert_eq!(payload["object"], "list");
    assert_eq!(payload["data"][0]["id"], "org/glm53-flash");
    assert_eq!(
        payload["data"][0]["architecture"]["input_modalities"],
        json!(["text"])
    );
    assert_eq!(
        payload["data"][0]["architecture"]["output_modalities"],
        json!(["text"])
    );
}

#[test]
fn parses_tool_calls() {
    let text = "Sure.\n<tool_call>\n<function=get_weather>\n<parameter=city>Paris</parameter>\n<parameter=units>metric</parameter>\n</function>\n</tool_call>";
    let (content, calls) = extract_tool_calls(text, &[]);
    assert_eq!(content, "Sure.");
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0]["type"], "function");
    assert_eq!(calls[0]["function"]["name"], "get_weather");
    let args: serde_json::Value =
        serde_json::from_str(calls[0]["function"]["arguments"].as_str().unwrap()).unwrap();
    assert_eq!(args["city"], "Paris");
    assert_eq!(args["units"], "metric");
}

#[test]
fn parses_ds4_glm_arg_key_value_tool_calls() {
    let text =
        "Before.\n<tool_call>bash<arg_key>command</arg_key><arg_value>pwd</arg_value></tool_call>";
    let (content, calls) = extract_tool_calls(text, &[]);
    assert_eq!(content, "Before.");
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0]["function"]["name"], "bash");
    let args: serde_json::Value =
        serde_json::from_str(calls[0]["function"]["arguments"].as_str().unwrap()).unwrap();
    assert_eq!(args["command"], "pwd");
}

#[test]
fn dsml_sampling_switches_at_glm_and_qwen_payload_boundaries() {
    use TargetMtpSamplingMode::{Configured, Greedy};

    assert_eq!(dsml_sampling_mode("ordinary answer", false), Configured);
    assert_eq!(
        dsml_sampling_mode("<tool_call>bash<arg_key>command</arg_key>", false),
        Greedy
    );
    assert_eq!(
        dsml_sampling_mode(
            "<tool_call>bash<arg_key>command</arg_key><arg_value>printf ok",
            false
        ),
        Configured
    );
    assert_eq!(
        dsml_sampling_mode(
            "<tool_call>bash<arg_key>command</arg_key><arg_value>printf ok</ar",
            false
        ),
        Greedy
    );
    assert_eq!(
        dsml_sampling_mode(
            "<tool_call>bash<arg_key>command</arg_key><arg_value>printf ok</arg_value></tool_call>",
            false
        ),
        Configured
    );
    assert_eq!(
        dsml_sampling_mode("<tool_call><function=search><parameter=query>hello", false),
        Configured
    );
    assert_eq!(
        dsml_sampling_mode("<tool_call><function=search><parameter=query", false),
        Greedy
    );
    assert_eq!(
        dsml_sampling_mode("<think>drafting <tool_call>ignored", true),
        Configured
    );
    assert_eq!(
        dsml_sampling_mode(
            "<think>drafting <tool_call>ignored</think><tool_call>bash<arg_key>command",
            true
        ),
        Greedy
    );
}

#[test]
fn glm_target_mtp_wave_preserves_serial_only_request_modes() {
    let make_job = |body: serde_json::Value| {
        let (reply, _receiver) = std::sync::mpsc::channel();
        Job {
            body,
            stream: false,
            session_id: None,
            surface: Surface::Chat,
            reply,
        }
    };

    assert!(is_target_mtp_wave_eligible(&make_job(json!({})), 2));
    assert!(!is_target_mtp_wave_eligible(
        &make_job(json!({"depth": 1})),
        1
    ));
    assert!(!is_target_mtp_wave_eligible(
        &make_job(json!({"ignore_eos": true})),
        2
    ));
    assert!(!is_target_mtp_wave_eligible(
        &make_job(json!({"tools": [{"type": "function"}]})),
        2
    ));
}

#[test]
fn streaming_thinking_matches_open_prompt_and_unclosed_final_state() {
    assert_eq!(
        streaming_sections("private scratchpad", true),
        ("private scratchpad", "")
    );
    assert_eq!(
        streaming_sections("private scratchpad</think><tool_call>x", true),
        ("private scratchpad", "<tool_call>x")
    );
    assert_eq!(
        streaming_sections("direct response", false),
        ("", "direct response")
    );
    assert_eq!(
        streaming_sections("private scratchpad</think>tool call", false),
        ("", "private scratchpad</think>tool call")
    );
    assert_eq!(
        streaming_reply_sections(
            "private scratchpad</think><tool_call>bash</tool_call>",
            false,
            &[],
        ),
        (String::new(), "private scratchpad</think>".into())
    );
    assert_eq!(
        streaming_reply_sections(
            "<think>private scratchpad</think><tool_call>bash</tool_call>",
            true,
            &[],
        ),
        ("private scratchpad".into(), String::new())
    );
    assert_eq!(
        split_reasoning_for_request("unfinished private scratchpad", true),
        (Some("unfinished private scratchpad".into()), String::new())
    );
    assert_eq!(
        split_reasoning_for_request("private scratchpad</think>visible answer", true),
        (Some("private scratchpad".into()), "visible answer".into())
    );
}

#[test]
fn streaming_stop_tail_holds_cross_token_prefix_and_flushes_only_safe_text() {
    let stops = vec![" gamma delta".to_string()];
    let partial = "alpha beta gamma";
    let safe_len = streaming_stop_safe_len(partial, &stops);
    assert_eq!(&partial[..safe_len], "alpha");

    // The next token completes the stop string. Final response shaping
    // flushes the safe prefix that was held back but never exposes the
    // stop sequence itself.
    let complete = "alpha beta gamma delta epsilon";
    let (_, final_content) = streaming_reply_sections(complete, false, &stops);
    assert_eq!(final_content, "alpha beta");
    assert_eq!(
        stream_delta(&partial[..safe_len], &final_content).as_deref(),
        Some(" beta")
    );

    // Holdback is measured in UTF-8 bytes but may only slice at a valid
    // character boundary.
    let unicode_stop = vec!["☃".to_string()];
    let unicode_text = "A☃";
    let unicode_safe_len = streaming_stop_safe_len(unicode_text, &unicode_stop);
    assert_eq!(&unicode_text[..unicode_safe_len], "A");
    assert_eq!(streaming_stop_safe_len("complete", &[]), "complete".len());
}

#[test]
fn openai_usage_reports_ds4_cache_read_and_write_counts() {
    assert_eq!(
        usage_json(19, 4, 0),
        json!({
            "prompt_tokens": 19,
            "completion_tokens": 4,
            "prompt_tokens_details": {
                "cached_tokens": 0,
                "cache_write_tokens": 19
            },
            "total_tokens": 23
        })
    );
    assert_eq!(
        usage_json(19, 4, 7)["prompt_tokens_details"],
        json!({"cached_tokens": 7, "cache_write_tokens": 12})
    );
    assert_eq!(
        prompt_tokens_details_json(19, 25),
        json!({"cached_tokens": 19, "cache_write_tokens": 0})
    );
}

#[test]
fn responses_usage_reports_ds4_cache_read_and_write_counts() {
    let data = ReplyData {
        content: String::new(),
        reasoning: None,
        tool_calls: Vec::new(),
        prompt_tokens: 19,
        completion_tokens: 4,
        cached_tokens: 7,
        finish: "stop".to_string(),
        logprobs: None,
        stop_sequence: None,
    };
    let payload = responses_payload_with_id(
        &super::ApiState::new("test-model".into()),
        &data,
        "resp_test",
    );
    assert_eq!(
        payload["usage"]["input_tokens_details"],
        json!({"cached_tokens": 7, "cache_write_tokens": 12})
    );
}

#[test]
fn parses_json_form_tool_calls() {
    let t1 = "<tool_call><function=run_command>{\"command\":\"npm test\"}</function></tool_call>";
    let (_, c1) = extract_tool_calls(t1, &[]);
    assert_eq!(c1.len(), 1);
    assert_eq!(c1[0]["function"]["name"], "run_command");
    let a1: serde_json::Value =
        serde_json::from_str(c1[0]["function"]["arguments"].as_str().unwrap()).unwrap();
    assert_eq!(a1["command"], "npm test");

    // Missing closing brace, plus a stray tag — must still yield `command`.
    // Unquoted keys/values (`{path:"README.md"}`) must still parse.
    let t1b = "<tool_call><function=read_file>{path:\"README.md\"}</function></tool_call>";
    let (_, c1b) = extract_tool_calls(t1b, &[]);
    let a1b: serde_json::Value =
        serde_json::from_str(c1b[0]["function"]["arguments"].as_str().unwrap()).unwrap();
    assert_eq!(a1b["path"], "README.md");

    let t2 = "<tool_call><function=run_command>{\"command\":\"node -e \\\"x\\\"\"</parameter></function></tool_call>";
    let (_, c2) = extract_tool_calls(t2, &[]);
    let a2: serde_json::Value =
        serde_json::from_str(c2[0]["function"]["arguments"].as_str().unwrap()).unwrap();
    assert_eq!(a2["command"], "node -e \"x\"");
}

#[test]
fn salvages_truncated_calls() {
    // Truncated mid-value, no closing tags at all.
    let t = "<tool_call><function=write_file>{\"path\":\"src/defaults.js\",\"content\":\"module.exports = { timeoutMs: 10000, retries: 2 };\n";
    let (_, c) = extract_tool_calls(t, &[]);
    let a: serde_json::Value =
        serde_json::from_str(c[0]["function"]["arguments"].as_str().unwrap()).unwrap();
    assert_eq!(a["path"], "src/defaults.js");

    // Missing final brace, stray closing tag.
    let t2 = "<tool_call><function=run_command>{\"command\":\"node -e \\\"const {slugify}=x\\\"\"</parameter></function></tool_call>";
    let (_, c2) = extract_tool_calls(t2, &[]);
    let a2: serde_json::Value =
        serde_json::from_str(c2[0]["function"]["arguments"].as_str().unwrap()).unwrap();
    assert_eq!(a2["command"], "node -e \"const {slugify}=x\"");
}

#[test]
fn parses_slugify_call() {
    let t = "<tool_call><function=run_command>{\"command\":\"node -e \\\"const {slugify}=require('./src/slug'); console.log(slugify('Hello World'));\\\"\"</parameter></function></tool_call>";
    let (_, c) = extract_tool_calls(t, &[]);
    eprintln!("c={c:?}");
    let a: serde_json::Value =
        serde_json::from_str(c[0]["function"]["arguments"].as_str().unwrap()).unwrap();
    assert!(a["command"].as_str().unwrap().contains("slugify"));
}

#[test]
fn parses_u_call() {
    let t = "<think></think><tool_call><function=run_command>{\"command\":\"node -e \\\"const u=require('./src/user'); console.log(u.getInitials({first:'Ada',last:'Lovelace'}));\\\" && npm test 2>&1 | tail -20\"</}</function></tool_call>";
    let (_, c) = extract_tool_calls(t, &[]);
    assert!(!c.is_empty());
    let a: serde_json::Value =
        serde_json::from_str(c[0]["function"]["arguments"].as_str().unwrap()).unwrap();
    assert!(a["command"].as_str().unwrap().contains("npm test"));
}

#[test]
fn supports_generation_markers() {
    // MiMo-style template block that minijinja does not understand.
    let src = "{%- macro m(x) -%}\n{%- generation -%}hello {{ x }}{%- endgeneration -%}\n{%- endmacro -%}{{ m(1) }}";
    let t = ChatTemplate::new(src).expect("template parse");
    let out = t.render(&[], false, &[], true).expect("render");
    assert!(out.contains("hello 1"), "got {out:?}");
}

#[test]
fn glm_no_thinking_prompt_omits_default_max_effort_message() {
    let src = "[gMASK]<sop><|system|>Reasoning Effort: Max<|user|>hello<|assistant|><think>";
    let t = ChatTemplate::new(src).expect("template parse");

    let direct = t
        .render(&[], true, &[], false)
        .expect("render direct prompt");
    assert_eq!(
        direct,
        "[gMASK]<sop><|user|>hello<|assistant|><think></think>"
    );

    let thinking = t
        .render(&[], true, &[], true)
        .expect("render thinking prompt");
    assert_eq!(thinking, src);

    let partial = t
        .render(&[], false, &[], false)
        .expect("render partial prompt");
    assert_eq!(partial, "[gMASK]<sop><|user|>hello<|assistant|><think>");
}

#[test]
fn glm_session_suffix_preserves_ds4_turn_boundary() {
    let full = "[gMASK]<sop><|user|>first<|assistant|><think></think>BLUE<|user|>second";
    let rest = "<|user|>second";
    assert_eq!(session_suffix_text(full, rest), rest);
    assert_eq!(
        session_suffix_text("<|im_start|>assistant", rest),
        "\n<|user|>second"
    );
}

#[test]
fn glm_tool_prompt_matches_ds4_json_and_whitespace() {
    let tools = vec![json!({
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Run a shell command.",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
                "additionalProperties": false
            },
            "strict": true,
            "defer_loading": false
        }
    })];
    let mut rendered = "You are provided with signatures within <tools></tools> XML tags:\n<tools>\n{\"name\": \"bash\", \"parameters\": { \"type\": \"object\" }}\n</tools>\n\nFor each function call".to_string();
    normalize_glm_tool_prompt(&mut rendered, &tools);
    assert_eq!(
        rendered,
        "You are provided with signatures within <tools></tools> XML tags:\n<tools>\n\n{\"name\": \"bash\", \"description\": \"Run a shell command.\", \"parameters\": {\"type\":\"object\",\"properties\":{\"command\":{\"type\":\"string\"}},\"required\":[\"command\"],\"additionalProperties\":false}}\n\n\n</tools>\n\nFor each function call"
    );
}

#[test]
fn glm_tool_prompt_matches_ds4_multiple_and_deferred_tools() {
    let tools = vec![
        json!({"type":"function", "function":{"name":"first", "parameters":{"type":"object"}}}),
        json!({"type":"function", "function":{"name":"hidden", "defer_loading":true}}),
        json!({"type":"function", "function":{"name":"last", "parameters":{"type":"array"}}}),
    ];
    let mut rendered = "<tools>\nfirst and last\n</tools>".to_string();
    normalize_glm_tool_prompt(&mut rendered, &tools);
    assert_eq!(
        rendered,
        "<tools>\n\n{\"name\": \"first\", \"parameters\": {\"type\":\"object\"}}\n\n\n{\"name\": \"last\", \"parameters\": {\"type\":\"array\"}}\n\n\n</tools>"
    );

    let deferred_only = vec![json!({
        "type": "function",
        "function": {"name": "hidden", "defer_loading": true}
    })];
    let mut rendered = "<tools>\n</tools>".to_string();
    normalize_glm_tool_prompt(&mut rendered, &deferred_only);
    assert_eq!(rendered, "<tools>\n\n\n</tools>");
}

#[test]
#[ignore]
fn render_mimo_tools_and_parts() {
    let dir = match lisa_engine::models::resolve_model_dir(
        "agnosticeng/MiMo-V2.6-Distill-Qwen-9B-4bit",
    ) {
        Ok(d) => d,
        Err(_) => return,
    };
    let src = match std::fs::read_to_string(dir.join("chat_template.jinja")) {
        Ok(s) => s,
        Err(_) => return,
    };
    // Exercise the branches our text-only ChatMessage wrapper never feeds:
    // tools, tool_calls, tool results, and multimodal content parts.
    let t = hf_chat_template::ChatTemplate::from_str(&src).expect("template parse");
    let mut input = hf_chat_template::RenderInput {
        add_generation_prompt: true,
        ..Default::default()
    };
    input.tools.push(json!({
        "type": "function",
        "function": { "name": "get_weather", "parameters": { "type": "object" } }
    }));
    input
        .messages
        .push(hf_chat_template::Message::user("weather in Paris?"));
    let mut call = hf_chat_template::Message::assistant("");
    call.tool_calls.push(json!({
        "function": { "name": "get_weather", "arguments": { "city": "Paris" } }
    }));
    input.messages.push(call);
    input
        .messages
        .push(hf_chat_template::Message::tool("sunny, 21C"));
    let mut img = hf_chat_template::Message::user("");
    img.content = Some(hf_chat_template::Content::Parts(vec![
        json!({"type": "image"}),
        json!({"type": "text", "text": "describe"}),
    ]));
    input.messages.push(img);
    input.extra.insert("enable_thinking".into(), json!(false));
    let out = t.render(&input).expect("render");
    eprintln!("---RENDER---\n{out}\n---END---");
}

#[test]
#[ignore]
fn render_mimo_template() {
    let dir = match lisa_engine::models::resolve_model_dir(
        "agnosticeng/MiMo-V2.6-Distill-Qwen-9B-4bit",
    ) {
        Ok(d) => d,
        Err(_) => return,
    };
    let src = match std::fs::read_to_string(dir.join("chat_template.jinja")) {
        Ok(s) => s,
        Err(_) => return,
    };
    let t = ChatTemplate::new(&src).expect("template parse");
    let msgs = vec![ChatMessage {
        role: "user".into(),
        content: "Name three colors.".into(),
        reasoning_content: None,
        tool_calls: Vec::new(),
        tool_call_id: None,
        parts: None,
    }];
    let out = t.render(&msgs, true, &[], true).expect("render");
    eprintln!("---RENDER---\n{out}\n---END---");
}

#[test]
#[ignore]
fn render_real_template() {
    let dir = match lisa_engine::models::resolve_model_dir("agnosticeng/Qwen3.8-Flash-Next-4bit") {
        Ok(d) => d,
        Err(_) => return,
    };
    let src = match std::fs::read_to_string(dir.join("chat_template.jinja")) {
        Ok(s) => s,
        Err(_) => return,
    };
    let t = ChatTemplate::new(&src).expect("template parse");
    let msgs = vec![
        ChatMessage {
            role: "user".into(),
            content: "Name three colors.".into(),
            reasoning_content: None,
            tool_calls: Vec::new(),
            tool_call_id: None,
            parts: None,
        },
        ChatMessage {
            role: "assistant".into(),
            content: "Red, green, blue.".into(),
            reasoning_content: Some("Thinking about colors.".into()),
            tool_calls: Vec::new(),
            tool_call_id: None,
            parts: None,
        },
        ChatMessage {
            role: "user".into(),
            content: "Which is warmest?".into(),
            reasoning_content: None,
            tool_calls: Vec::new(),
            tool_call_id: None,
            parts: None,
        },
    ];
    let out = t.render(&msgs, true, &[], true).expect("render");
    eprintln!("---RENDER---\n{out}\n---END---");
}
