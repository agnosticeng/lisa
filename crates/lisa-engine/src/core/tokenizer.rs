//! Tokenizer wrapper and ChatML template application.

use tokenizers::Tokenizer as HfTokenizer;

pub struct Tokenizer {
    pub inner: HfTokenizer,
    pub im_end_ids: Vec<u32>,
}

impl Tokenizer {
    pub fn load(dir: &std::path::Path) -> anyhow::Result<Self> {
        let inner = HfTokenizer::from_file(dir.join("tokenizer.json"))
            .map_err(|e| anyhow::anyhow!("tokenizer: {e}"))?;
        // <|im_end|> and <|endoftext|>
        let mut im_end_ids = Vec::new();
        for token in ["<|im_end|>", "<|endoftext|>"] {
            if let Some(id) = inner.token_to_id(token) {
                im_end_ids.push(id);
            }
        }
        Ok(Self { inner, im_end_ids })
    }

    pub fn encode(&self, text: &str, add_bos: bool) -> anyhow::Result<Vec<u32>> {
        let mut ids = self
            .inner
            .encode(text, false)
            .map_err(|e| anyhow::anyhow!("encode: {e}"))?
            .get_ids()
            .to_vec();
        if add_bos {
            if let Some(bos) = self.inner.token_to_id("<|im_start|>") {
                ids.insert(0, bos);
            }
        }
        Ok(ids)
    }

    pub fn decode(&self, ids: &[u32]) -> anyhow::Result<String> {
        self.inner
            .decode(ids, false)
            .map_err(|e| anyhow::anyhow!("decode: {e}"))
    }
}

/// A minimal ChatML applier matching the checkpoint's chat template for
/// text-only conversations (system/user/assistant), with the default
/// thinking-mode generation prompt.
pub fn apply_chat_template(messages: &[(String, String)]) -> String {
    let mut out = String::new();
    for (role, content) in messages {
        match role.as_str() {
            "system" => {
                out.push_str("<|im_start|>system\n");
                out.push_str(content);
                out.push_str("<|im_end|>\n");
            }
            "user" => {
                out.push_str("<|im_start|>user\n");
                out.push_str(content);
                out.push_str("<|im_end|>\n");
            }
            "assistant" => {
                out.push_str("<|im_start|>assistant\n");
                out.push_str(content);
                out.push_str("<|im_end|>\n");
            }
            other => {
                out.push_str(&format!("<|im_start|>{other}\n{content}<|im_end|>\n"));
            }
        }
    }
    out
}

/// The generation prompt: everything above plus the assistant header with an
/// open `<think>` block (thinking enabled, matching the template default).
/// The assistant header with an open think block, verbatim from the
/// checkpoint's `chat_template.jinja`.
pub const ASSISTANT_PROMPT: &str = "<|im_start|>assistant\n<think\n";

/// The generation prompt: the rendered conversation plus the assistant header.
pub fn generation_prompt(messages: &[(String, String)]) -> String {
    let mut out = apply_chat_template(messages);
    out.push_str(ASSISTANT_PROMPT);
    out
}

/// The tokens a new chat turn appends to the committed conversation: close the
/// previous assistant turn, render the new user turn, open a fresh assistant
/// turn. `first` is the very first turn (nothing to close).
pub fn chat_turn_suffix(msg: &str, first: bool) -> String {
    let mut out = String::new();
    if !first {
        out.push_str("<|im_end|>\n");
    }
    out.push_str(&apply_chat_template(&[("user".to_string(), msg.to_string())]));
    out.push_str(ASSISTANT_PROMPT);
    out
}
