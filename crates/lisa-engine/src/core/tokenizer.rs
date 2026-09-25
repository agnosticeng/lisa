//! Tokenizer wrapper and ChatML template application.

use tokenizers::Tokenizer as HfTokenizer;

pub struct Tokenizer {
    pub inner: HfTokenizer,
    pub im_end_ids: Vec<u32>,
}

impl Tokenizer {
    pub fn load(dir: &std::path::Path) -> anyhow::Result<Self> {
        Self::load_file(&dir.join("tokenizer.json"))
    }

    /// Load a `tokenizer.json` from an explicit path (some checkpoints keep it
    /// in a `tokenizer/` subdirectory).
    pub fn load_file(path: &std::path::Path) -> anyhow::Result<Self> {
        let inner = HfTokenizer::from_file(path)
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

    /// Decode without rendering special tokens (`<im_end>` and friends).
    pub fn decode_clean(&self, ids: &[u32]) -> anyhow::Result<String> {
        self.inner
            .decode(ids, true)
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
pub const ASSISTANT_PROMPT: &str = "assistant\n<think>\n";

/// The generation prompt: the rendered conversation plus the assistant header.
pub fn generation_prompt(messages: &[(String, String)]) -> String {
    let mut out = apply_chat_template(messages);
    out.push_str(ASSISTANT_PROMPT);
    out
}

/// Non-thinking generation prompt: a pre-closed `<think>` block, matching the
/// template's `enable_thinking=false`, so the model answers directly with no
/// visible reasoning. Used by the UI.
pub fn generation_prompt_no_think(messages: &[(String, String)]) -> String {
    let mut out = apply_chat_template(messages);
    out.push_str("assistant\n<think>\n\n</think>\n\n");
    out
}

/// Build a Qwen ChatML prompt using the real special tokens. With
/// `think == false` the think block is pre-closed so the model answers directly.
pub fn chat_prompt_special(text: &str, think: bool) -> String {
    chat_prompt_special_with_system(text, None, think)
}

/// Like [`chat_prompt_special`], optionally prefixed with a `system` turn.
pub fn chat_prompt_special_with_system(text: &str, system: Option<&str>, think: bool) -> String {
    let start = concat!("<|", "im_start", "|>");
    let end = concat!("<|", "im_end", "|>");
    let tail = if think {
        "<think>\n"
    } else {
        "<think>\n\n</think>\n\n"
    };
    match system.map(str::trim).filter(|s| !s.is_empty()) {
        Some(sys) => format!(
            "{start}system\n{sys}{end}\n{start}user\n{text}{end}\n{start}assistant\n{tail}"
        ),
        None => format!("{start}user\n{text}{end}\n{start}assistant\n{tail}"),
    }
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

/// Like [`chat_turn_suffix`], but using the real ChatML special tokens and a
/// pre-closed think block when `think == false`, matching
/// [`chat_prompt_special_with_system`]. Used for multi-turn sessions so each
/// turn continues the committed conversation.
///
/// `assistant_im_end` distinguishes templates that close an assistant turn with
/// `` (e.g. MiMo) from Qwen's, which relies on a plain newline.
pub fn chat_turn_suffix_special(
    msg: &str,
    first: bool,
    think: bool,
    assistant_im_end: bool,
) -> String {
    let start = concat!("<|", "im_start", "|>");
    let end = concat!("<|", "im_end", "|>");
    let tail = if think {
        " thinking\n"
    } else {
        " thinking\n\n response\n\n"
    };
    let lead = if first {
        String::new()
    } else if assistant_im_end {
        format!("{end}\n")
    } else {
        "\n".to_string()
    };
    format!("{lead}{start}user\n{msg}{end}\n{start}assistant\n{tail}")
}
