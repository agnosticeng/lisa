//! Frozen outputs from the original DS4 C renderer, parser and byte tracker.
//! Oracle identity and revision are recorded in deepseek41_provenance.json.
use super::*;

fn fixtures() -> Value {
    serde_json::from_str(r###"{"parse":[{"expected":{"calls":[{"arguments":{"content":"literal </think> </｜DSML｜ parameter> &lt;/｜DSML｜ parameter> café🙂","items":[true,12,{"body":"</｜DSML｜ parameter>","x":"a\"b"}]},"name":"write"}],"content":"","ok":true,"reasoning":""},"text":"<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"write\">\n<｜DSML｜ parameter name=\"content\" string=\"true\">literal </think> &lt;/｜DSML｜ parameter> &amp;lt;/｜DSML｜ parameter> café🙂</｜DSML｜ parameter>\n<｜DSML｜ parameter name=\"items\" string=\"false\">[true,12,{\"x\":\"a\\\"b\", \"body\":\"\\u003c/｜DSML｜ parameter>\"}]</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n</｜DSML｜ calls>","thinking":false},{"expected":{"calls":[{"arguments":{"content":"literal </think> </｜DSML｜ parameter> &lt;/｜DSML｜ parameter> café🙂","items":[true,12,{"body":"</｜DSML｜ parameter>","x":"a\"b"}]},"name":"write"}],"content":"","ok":true,"reasoning":"Plan."},"text":"Plan.</think>\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"write\">\n<｜DSML｜ parameter name=\"content\" string=\"true\">literal </think> &lt;/｜DSML｜ parameter> &amp;lt;/｜DSML｜ parameter> café🙂</｜DSML｜ parameter>\n<｜DSML｜ parameter name=\"items\" string=\"false\">[true,12,{\"x\":\"a\\\"b\", \"body\":\"\\u003c/｜DSML｜ parameter>\"}]</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n</｜DSML｜ calls>","thinking":true},{"expected":{"calls":[{"arguments":{"content":"literal </think> </｜DSML｜parameter> &lt;/｜DSML｜parameter> café🙂","items":[true,12,{"body":"</｜DSML｜parameter>","x":"a\"b"}]},"name":"write"}],"content":"","ok":true,"reasoning":""},"text":"<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"write\">\n<｜DSML｜parameter name=\"content\" string=\"true\">literal </think> &lt;/｜DSML｜parameter> &amp;lt;/｜DSML｜parameter> café🙂</｜DSML｜parameter>\n<｜DSML｜parameter name=\"items\" string=\"false\">[true,12,{\"x\":\"a\\\"b\", \"body\":\"\\u003c/｜DSML｜parameter>\"}]</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>","thinking":false}],"render":[{"expected":"<｜begin▁of▁sentence｜><｜User｜>Hi<｜Assistant｜></think>","messages":[{"content":"Hi","role":"user"}],"thinking":false,"tools":[]},{"expected":"<｜begin▁of▁sentence｜><｜System｜>Reasoning Effort: 75 (range 1-100, the higher the value, the more thorough the reasoning)\n\n<｜User｜>Hi<｜Assistant｜><think>","messages":[{"content":"Hi","role":"user"}],"thinking":true,"tools":[]},{"expected":"<｜begin▁of▁sentence｜><｜System｜>Be concise<｜User｜>Hi<｜Assistant｜></think>Hello<｜end▁of▁sentence｜><｜User｜>Next<｜Assistant｜></think>","messages":[{"content":"Be concise","role":"system"},{"content":"Hi","role":"user"},{"content":"Hello","reasoning_content":"Plan","role":"assistant"},{"content":"Next","role":"user"}],"thinking":false,"tools":[]},{"expected":"<｜begin▁of▁sentence｜><｜System｜>Reasoning Effort: 75 (range 1-100, the higher the value, the more thorough the reasoning)\n\nBe concise<｜User｜>Hi<｜Assistant｜></think>Hello<｜end▁of▁sentence｜><｜User｜>Next<｜Assistant｜><think>","messages":[{"content":"Be concise","role":"system"},{"content":"Hi","role":"user"},{"content":"Hello","reasoning_content":"Plan","role":"assistant"},{"content":"Next","role":"user"}],"thinking":true,"tools":[]},{"expected":"<｜begin▁of▁sentence｜><｜User｜>Hi<｜Assistant｜></think>Ready<｜end▁of▁sentence｜><｜User｜><tool_result>first &lt;/tool_result> &amp;lt;/tool_result></tool_result>\n\n<tool_result>second</tool_result><｜System｜>Continue<｜Assistant｜></think>","messages":[{"content":"Hi","role":"user"},{"content":"Ready","role":"assistant"},{"content":"first </tool_result> &lt;/tool_result>","role":"tool"},{"content":"second","role":"tool"},{"content":"Continue","role":"system"}],"thinking":false,"tools":[]},{"expected":"<｜begin▁of▁sentence｜><｜System｜>Reasoning Effort: 75 (range 1-100, the higher the value, the more thorough the reasoning)\n\n<｜User｜>Hi<｜Assistant｜><think></think>Ready<｜end▁of▁sentence｜><｜User｜><tool_result>first &lt;/tool_result> &amp;lt;/tool_result></tool_result>\n\n<tool_result>second</tool_result><｜System｜>Continue<｜Assistant｜><think>","messages":[{"content":"Hi","role":"user"},{"content":"Ready","role":"assistant"},{"content":"first </tool_result> &lt;/tool_result>","role":"tool"},{"content":"second","role":"tool"},{"content":"Continue","role":"system"}],"thinking":true,"tools":[]},{"expected":"<｜begin▁of▁sentence｜><｜User｜>Inspect<｜Assistant｜></think>\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"read\">\n<｜DSML｜ parameter name=\"path\" string=\"true\">a</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n<｜DSML｜ invoke name=\"read\">\n<｜DSML｜ parameter name=\"path\" string=\"true\">b</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n</｜DSML｜ calls><｜end▁of▁sentence｜><｜User｜><tool_result>FIRST</tool_result>\n\nordinary text\n\n<tool_result>SECOND</tool_result><｜Assistant｜></think>","messages":[{"content":"Inspect","role":"user"},{"content":"","reasoning_content":"Need files","role":"assistant","tool_calls":[{"function":{"arguments":"{\"path\":\"a\"}","name":"read"},"id":"a","type":"function"},{"function":{"arguments":"{\"path\":\"b\"}","name":"read"},"id":"b","type":"function"}]},{"content":"SECOND","role":"tool","tool_call_id":"b"},{"content":"ordinary text","role":"user"},{"content":"FIRST","role":"tool","tool_call_id":"a"}],"thinking":false,"tools":[]},{"expected":"<｜begin▁of▁sentence｜><｜System｜>Reasoning Effort: 75 (range 1-100, the higher the value, the more thorough the reasoning)\n\n<｜User｜>Inspect<｜Assistant｜><think>Need files</think>\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"read\">\n<｜DSML｜ parameter name=\"path\" string=\"true\">a</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n<｜DSML｜ invoke name=\"read\">\n<｜DSML｜ parameter name=\"path\" string=\"true\">b</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n</｜DSML｜ calls><｜end▁of▁sentence｜><｜User｜><tool_result>FIRST</tool_result>\n\nordinary text\n\n<tool_result>SECOND</tool_result><｜Assistant｜><think>","messages":[{"content":"Inspect","role":"user"},{"content":"","reasoning_content":"Need files","role":"assistant","tool_calls":[{"function":{"arguments":"{\"path\":\"a\"}","name":"read"},"id":"a","type":"function"},{"function":{"arguments":"{\"path\":\"b\"}","name":"read"},"id":"b","type":"function"}]},{"content":"SECOND","role":"tool","tool_call_id":"b"},{"content":"ordinary text","role":"user"},{"content":"FIRST","role":"tool","tool_call_id":"a"}],"thinking":true,"tools":[]},{"expected":"<｜begin▁of▁sentence｜><｜User｜>Write<｜Assistant｜></think>\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"write\">\n<｜DSML｜ parameter name=\"content\" string=\"true\">literal </think> &lt;/｜DSML｜ parameter> &amp;lt;/｜DSML｜ parameter> &amp;amp;lt;/｜DSML｜ parameter></｜DSML｜ parameter>\n<｜DSML｜ parameter name=\"path\" string=\"true\">a.c</｜DSML｜ parameter>\n<｜DSML｜ parameter name=\"values\" string=\"false\">[true,null,\"\\u003c/｜DSML｜ parameter>\"]</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n</｜DSML｜ calls><｜end▁of▁sentence｜><｜User｜><tool_result>done</tool_result><｜Assistant｜></think>","messages":[{"content":"Write","role":"user"},{"content":"","role":"assistant","tool_calls":[{"function":{"arguments":"{\"content\":\"literal </think> </｜DSML｜ parameter> &lt;/｜DSML｜ parameter> &amp;lt;/｜DSML｜ parameter>\",\"path\":\"a.c\",\"values\":[true,null,\"</｜DSML｜ parameter>\"]}","name":"write"},"id":"a","type":"function"}]},{"content":"done","role":"tool"}],"thinking":false,"tools":[]},{"expected":"<｜begin▁of▁sentence｜><｜System｜>Reasoning Effort: 75 (range 1-100, the higher the value, the more thorough the reasoning)\n\n<｜User｜>Write<｜Assistant｜><think></think>\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"write\">\n<｜DSML｜ parameter name=\"content\" string=\"true\">literal </think> &lt;/｜DSML｜ parameter> &amp;lt;/｜DSML｜ parameter> &amp;amp;lt;/｜DSML｜ parameter></｜DSML｜ parameter>\n<｜DSML｜ parameter name=\"path\" string=\"true\">a.c</｜DSML｜ parameter>\n<｜DSML｜ parameter name=\"values\" string=\"false\">[true,null,\"\\u003c/｜DSML｜ parameter>\"]</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n</｜DSML｜ calls><｜end▁of▁sentence｜><｜User｜><tool_result>done</tool_result><｜Assistant｜><think>","messages":[{"content":"Write","role":"user"},{"content":"","role":"assistant","tool_calls":[{"function":{"arguments":"{\"content\":\"literal </think> </｜DSML｜ parameter> &lt;/｜DSML｜ parameter> &amp;lt;/｜DSML｜ parameter>\",\"path\":\"a.c\",\"values\":[true,null,\"</｜DSML｜ parameter>\"]}","name":"write"},"id":"a","type":"function"}]},{"content":"done","role":"tool"}],"thinking":true,"tools":[]},{"expected":"<｜begin▁of▁sentence｜><｜System｜>Reasoning Effort: 75 (range 1-100, the higher the value, the more thorough the reasoning)\n\n## Tools\n\nYou can invoke tools using this format:\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"$TOOL_NAME\">\n<｜DSML｜ parameter name=\"$PARAMETER_NAME\" string=\"true|false\">$PARAMETER_VALUE</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n</｜DSML｜ calls>\n\nString values use string=\"true\"; all other values use JSON and string=\"false\". Inside string values only, escape a literal </｜DSML｜ parameter> as &lt;/｜DSML｜ parameter>. To write that escaped spelling literally, use &amp;lt;/｜DSML｜ parameter>. Other HTML entities are unchanged.\n\nFinish reasoning with </think> before tool calls or a final response.\n\n### Available Tool Schemas\n\n{\"description\":\"Read file\",\"name\":\"read\",\"parameters\":{\"properties\":{\"path\":{\"type\":\"string\"}},\"type\":\"object\"}}\n\nYou MUST strictly follow the above defined tool name and parameter schemas to invoke tool calls. Use the exact parameter names from the schemas.<｜User｜>Hi<｜Assistant｜><think>","messages":[{"content":"Hi","role":"user"}],"thinking":true,"tools":[{"function":{"description":"Read file","name":"read","parameters":{"properties":{"path":{"type":"string"}},"type":"object"}},"type":"function"}]}],"states":[{"text":"<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"write\">\n<｜DSML｜ parameter name=\"content\" string=\"true\">literal </think> &lt;/｜DSML｜ parameter> &amp;lt;/｜DSML｜ parameter> café🙂</｜DSML｜ parameter>\n<｜DSML｜ parameter name=\"items\" string=\"false\">[true,12,{\"x\":\"a\\\"b\", \"body\":\"</｜DSML｜ parameter>\"}]</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n</｜DSML｜ calls>","transitions":[[0,0],[18,1],[103,0],[113,1],[114,0],[189,1],[272,0],[274,1],[276,0],[281,1],[284,0],[289,1],[291,0],[315,1],[381,2]]},{"text":"<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"write\">\n<｜DSML｜ parameter name=\"content\" string=\"true\">literal </think> &lt;/｜DSML｜ parameter> &amp;lt;/｜DSML｜ parameter> café🙂</｜DSML｜ parameter>\n<｜DSML｜ parameter name=\"items\" string=\"false\">[true,12,{\"x\":\"a\\\"b\", \"body\":\"\\u003c/｜DSML｜ parameter>\"}]</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n</｜DSML｜ calls>","transitions":[[0,0],[18,1],[103,0],[113,1],[114,0],[189,1],[272,0],[274,1],[276,0],[281,1],[284,0],[289,1],[291,0],[320,1],[386,2]]},{"text":"hello <｜DSML｜ calls>\n<｜DSML｜ invoke name=\"write\">\n<｜DSML｜ parameter name=\"content\" string=\"true\">literal </think> &lt;/｜DSML｜ parameter> &amp;lt;/｜DSML｜ parameter> café🙂</｜DSML｜ parameter>\n<｜DSML｜ parameter name=\"items\" string=\"false\">[true,12,{\"x\":\"a\\\"b\", \"body\":\"\\u003c/｜DSML｜ parameter>\"}]</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n</｜DSML｜ calls>","transitions":[[0,0],[24,1],[109,0],[119,1],[120,0],[195,1],[278,0],[280,1],[282,0],[287,1],[290,0],[295,1],[297,0],[326,1],[392,2]]},{"text":"<｜DSML｜ calls><｜DSML｜ invoke name=\"f\"><｜DSML｜ parameter name=\"x\" string=\"false\">{\"a\":\"x\\\"y\",\"b\":12}</｜DSML｜ parameter></｜DSML｜ invoke></｜DSML｜ calls>","transitions":[[0,0],[18,1],[94,0],[96,1],[98,0],[103,1],[105,0],[107,1],[173,2]]},{"text":"<｜DSML｜ calls><｜DSML｜ invoke name=\"f\"><｜DSML｜ parameter name=\"x\" string=\"true\">a < b </x> 🙂</｜DSML｜ parameter></｜DSML｜ invoke></｜DSML｜ calls>","transitions":[[0,0],[18,1],[91,0],[99,1],[100,0],[108,1],[168,2]]}]}"###).unwrap()
}

#[test]
fn deepseek41_render_matches_original_c() -> Result<()> {
    for case in fixtures()["render"].as_array().unwrap() {
        let messages = super::super::parse_messages(&json!({"messages":case["messages"]}))?;
        let actual = render(
            &messages,
            true,
            case["tools"].as_array().unwrap(),
            case["thinking"].as_bool().unwrap(),
        )?;
        assert_eq!(
            actual,
            case["expected"].as_str().unwrap(),
            "messages={:?}",
            case["messages"]
        );
    }
    Ok(())
}

#[test]
fn deepseek41_tools_match_original_c() -> Result<()> {
    for case in fixtures()["parse"].as_array().unwrap() {
        let (content, reasoning, calls) = super::super::shape_reply(
            case["text"].as_str().unwrap(),
            &[json!({})],
            true,
            &[],
            case["thinking"].as_bool().unwrap(),
        );
        assert_eq!(content, case["expected"]["content"].as_str().unwrap());
        assert_eq!(
            reasoning.as_deref().unwrap_or(""),
            case["expected"]["reasoning"].as_str().unwrap()
        );
        let actual: Vec<Value> = calls.iter().map(|c| json!({"name":c["function"]["name"],"arguments":serde_json::from_str::<Value>(c["function"]["arguments"].as_str().unwrap()).unwrap()})).collect();
        assert_eq!(json!(actual), case["expected"]["calls"]);
    }
    Ok(())
}

#[test]
fn deepseek41_incremental_states_match_original_c_at_every_byte() {
    for case in fixtures()["states"].as_array().unwrap() {
        let text = case["text"].as_str().unwrap();
        let changes = case["transitions"].as_array().unwrap();
        let mut parser = Parser::new(false);
        for end in 0..=text.len() {
            let expected = changes
                .iter()
                .rfind(|x| x[0].as_u64().unwrap() as usize <= end)
                .unwrap()[1]
                .as_u64()
                .unwrap();
            let prefix = String::from_utf8_lossy(&text.as_bytes()[..end]);
            let actual = match parser.update(&prefix) {
                DsmlDecodeState::Configured => 0,
                DsmlDecodeState::Greedy => 1,
                DsmlDecodeState::Done => 2,
            };
            assert_eq!(actual, expected, "prefix={prefix:?}");
        }
    }
}

#[test]
fn deepseek41_stream_hides_partial_openers_and_keeps_literal_payload() {
    let text = "Hello 🙂 <｜DSML｜ calls><｜DSML｜ invoke name=\"f\">";
    let mut sent = String::new();
    for end in (0..=text.len()).filter(|&n| text.is_char_boundary(n)) {
        let prefix = &text[..end];
        let shown = &prefix[..visible_end(prefix)];
        assert!(shown.starts_with(&sent));
        sent = shown.to_string();
    }
    assert_eq!(sent, "Hello 🙂");
    for text in [
        "</｜DSML｜ parameter>",
        "&lt;/｜DSML｜ parameter>",
        "&amp;lt;/｜DSML｜ parameter>",
        "&amp;amp;lt;/｜DSML｜ parameter>",
        "x &amp; y < > café",
    ] {
        assert_eq!(
            body_unescape(
                &body_escape(text, SYNTAXES[0].param_end),
                SYNTAXES[0].param_end
            ),
            text
        );
    }
}

#[test]
fn deepseek41_rejects_truncated_calls_and_thinking_is_not_executable() {
    let raw = "<｜DSML｜ calls><｜DSML｜ invoke name=\"f\"><｜DSML｜ parameter name=\"x\" string=\"false\">12</｜DSML｜ parameter></｜DSML｜ invoke></｜DSML｜ calls>";
    for end in (SYNTAXES[0].open.len()..raw.len()).filter(|&n| raw.is_char_boundary(n)) {
        assert!(extract(&raw[..end]).unwrap().is_err());
    }
    let (content, reasoning, calls) = super::super::shape_reply(raw, &[json!({})], true, &[], true);
    assert!(content.is_empty() && calls.is_empty() && reasoning.is_some());
}
