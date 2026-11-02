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
        let is_glm53 = match path.parent() {
            Some(dir) => is_glm53_model_dir(dir)?,
            None => false,
        };
        let inner = if is_glm53 {
            load_glm53_tokenizer(path)?
        } else {
            HfTokenizer::from_file(path).map_err(|e| anyhow::anyhow!("tokenizer: {e}"))?
        };
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

    /// Special-token strings worth splitting on, present in this vocab.
    pub fn special_tokens(&self) -> Vec<(&'static str, u32)> {
        const CANDIDATES: &[&str] = &[
            "[gMASK]",
            "<sop>",
            "<|system|>",
            "<|user|>",
            "<|assistant|>",
            "<|observation|>",
            "<think>",
            "</think>",
            "<|im_start|>",
            "<|im_end|>",
            "<|endoftext|>",
        ];
        CANDIDATES
            .iter()
            .filter_map(|t| self.inner.token_to_id(t).map(|id| (*t, id)))
            .collect()
    }

    /// Encode with special-token strings mapped to their single ids (the HF
    /// `tokenizers` plain encode does NOT split on tokens that are absent from
    /// `added_tokens` — glm5_next's `<|system|>` et al. are vocab entries, not
    /// added tokens, so a plain encode yields literal text).
    pub fn encode_special(&self, text: &str) -> anyhow::Result<Vec<u32>> {
        let specials = self.special_tokens();
        if specials.is_empty() {
            return self.encode(text, false);
        }
        // Greedy longest-match scan. GLM's markers are ordinary vocab entries,
        // not added tokens — split on them explicitly.
        let specials = {
            let mut sp = specials;
            for tok in [
                "\u{ff5c}user",
                "\u{ff5c}assistant",
                "\u{ff5c}system",
                "\u{ff5c}observation",
                "gMASK",
                "<sop>",
                "ressed\n",
            ] {
                if let Some(id) = self.inner.token_to_id(tok) {
                    sp.push((tok, id));
                }
            }
            sp
        };
        let mut ids = Vec::new();
        let mut i = 0usize;
        while i < text.len() {
            let mut hit: Option<(usize, u32)> = None; // (len, id)
            for (s, id) in &specials {
                if text[i..].starts_with(s) {
                    let better = hit.map(|(l, _)| s.len() > l).unwrap_or(true);
                    if better {
                        hit = Some((s.len(), *id));
                    }
                }
            }
            match hit {
                Some((l, id)) => {
                    ids.push(id);
                    i += l;
                }
                None => {
                    // advance one utf-8 char, accumulating a plain segment
                    let start = i;
                    i += 1;
                    while i < text.len() && !text.is_char_boundary(i) {
                        i += 1;
                    }
                    // accumulate consecutive plain chars into one encode call
                    let mut j = i;
                    while j < text.len() {
                        if specials.iter().any(|(s, _)| text[j..].starts_with(s)) {
                            break;
                        }
                        j += 1;
                        while j < text.len() && !text.is_char_boundary(j) {
                            j += 1;
                        }
                    }
                    let seg = &text[start..j];
                    ids.extend(self.encode(seg, false)?);
                    i = j;
                }
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

fn is_glm53_model_dir(dir: &std::path::Path) -> anyhow::Result<bool> {
    let config_path = dir.join("config.json");
    if !config_path.is_file() {
        return Ok(false);
    }
    let config: serde_json::Value = serde_json::from_slice(&std::fs::read(&config_path)?)?;
    let model_type = config.get("model_type").and_then(serde_json::Value::as_str);
    let text_model_type = config
        .pointer("/text_config/model_type")
        .and_then(serde_json::Value::as_str);
    Ok(matches!(model_type, Some("glm5_next" | "glm5_next_text"))
        || text_model_type == Some("glm5_next_text"))
}

fn load_glm53_tokenizer(path: &std::path::Path) -> anyhow::Result<HfTokenizer> {
    let bytes = std::fs::read(path)?;
    load_glm53_tokenizer_bytes(&bytes)
}

fn load_glm53_tokenizer_bytes(bytes: &[u8]) -> anyhow::Result<HfTokenizer> {
    let mut document: serde_json::Value = serde_json::from_slice(bytes)?;
    let model = document
        .get_mut("model")
        .and_then(serde_json::Value::as_object_mut)
        .ok_or_else(|| anyhow::anyhow!("GLM-5.3 tokenizer model is missing"))?;
    anyhow::ensure!(
        model.get("type").and_then(serde_json::Value::as_str) == Some("BPE"),
        "GLM-5.3 DS4 parity requires a BPE tokenizer"
    );

    // The MLX tokenizer JSON sets ignore_merges=true, allowing a whole vocab
    // entry to bypass BPE merge ranks. DS4's GGUF tokenizer always applies its
    // merge table. Force the same semantics while keeping the MLX vocab and
    // pre-tokenizer, so raw text maps to the same token IDs as DS4.
    model.insert("ignore_merges".to_string(), serde_json::Value::Bool(false));
    let bytes = serde_json::to_vec(&document)?;
    HfTokenizer::from_bytes(bytes).map_err(|e| anyhow::anyhow!("GLM-5.3 tokenizer: {e}"))
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
        Some(sys) => {
            format!("{start}system\n{sys}{end}\n{start}user\n{text}{end}\n{start}assistant\n{tail}")
        }
        None => format!("{start}user\n{text}{end}\n{start}assistant\n{tail}"),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GlmReasoning {
    High,
    Max,
    Disabled,
}

/// GLM-5.3 generation prompt, matching ds4's text-only chat prefix. The
/// control markers must be encoded as their special-token ids; rendering
/// lookalike separators as ordinary text changes the model input.
pub fn glm_generation_prompt(text: &str, system: Option<&str>, reasoning: GlmReasoning) -> String {
    let mut out = String::from("[gMASK]<sop>");
    let effort = match reasoning {
        GlmReasoning::High => Some("Reasoning Effort: High"),
        GlmReasoning::Max => Some("Reasoning Effort: Max"),
        GlmReasoning::Disabled => None,
    };
    if let Some(effort) = effort {
        out.push_str("<|system|>");
        out.push_str(effort);
    }
    if let Some(system) = system.filter(|system| !system.is_empty()) {
        out.push_str("<|system|>");
        out.push_str(system);
    }
    out.push_str("<|user|>");
    out.push_str(text);
    out.push_str("<|assistant|><think>");
    if reasoning == GlmReasoning::Disabled {
        out.push_str("</think>");
    }
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
    out.push_str(&apply_chat_template(&[(
        "user".to_string(),
        msg.to_string(),
    )]));
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn glm53_tokenizer_uses_ds4_bpe_merge_semantics() -> anyhow::Result<()> {
        let source = br#"{
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "added_tokens": [],
            "normalizer": null,
            "pre_tokenizer": {"type": "WhitespaceSplit"},
            "post_processor": null,
            "decoder": null,
            "model": {
                "type": "BPE",
                "dropout": null,
                "unk_token": null,
                "continuing_subword_prefix": null,
                "end_of_word_suffix": null,
                "fuse_unk": false,
                "byte_fallback": false,
                "ignore_merges": true,
                "vocab": {"a": 0, "b": 1, "ab": 2},
                "merges": []
            }
        }"#;
        let source_tokenizer = HfTokenizer::from_bytes(source)
            .map_err(|e| anyhow::anyhow!("source tokenizer: {e}"))?;
        let ds4_tokenizer = load_glm53_tokenizer_bytes(source)?;
        let source_ids = source_tokenizer
            .encode("ab", false)
            .map_err(|e| anyhow::anyhow!("source encode: {e}"))?;
        let ds4_ids = ds4_tokenizer
            .encode("ab", false)
            .map_err(|e| anyhow::anyhow!("DS4-compatible encode: {e}"))?;
        assert_eq!(source_ids.get_ids(), &[2]);
        assert_eq!(ds4_ids.get_ids(), &[0, 1]);
        Ok(())
    }

    #[test]
    fn glm53_real_tokenizer_matches_ds4_natural_prompt_fixture() -> anyhow::Result<()> {
        let Some(model_dir) = std::env::var_os("LISA_GLM53_MODEL") else {
            eprintln!("LISA_GLM53_MODEL=<snapshot> to check real GLM tokenizer parity");
            return Ok(());
        };
        let tokenizer = Tokenizer::load(std::path::Path::new(&model_dir))?;
        assert_eq!(tokenizer.encode(" attenzione", false)?, [1619, 16581, 6339]);

        let fixture_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("src/core/fixtures/glm53-tokenizer");
        let read_u32 = |name: &str| -> anyhow::Result<Vec<u32>> {
            let bytes = std::fs::read(fixture_dir.join(name))?;
            anyhow::ensure!(bytes.len() % 4 == 0, "{name} is not a u32 array");
            Ok(bytes
                .chunks_exact(4)
                .map(|chunk| u32::from_le_bytes(chunk.try_into().expect("four bytes")))
                .collect())
        };
        let mlx_ids = read_u32("prompt-row-1.u32")?;
        let ds4_ids = read_u32("prompt-row-1.ds4.u32")?;
        let text = tokenizer.decode(&mlx_ids)?;
        let actual = tokenizer.encode(&text, false)?;
        assert_eq!(actual.len(), 8_195);
        assert_eq!(actual, ds4_ids);

        let cases_path = fixture_dir.join("tokenizer-ds4-merge-cases-20261101.json");
        let fixture: serde_json::Value = serde_json::from_slice(&std::fs::read(cases_path)?)?;
        assert_eq!(
            fixture["ds4_revision"],
            "c97964b8aa77034a8e17afb10ca7a686c40ab511"
        );
        let cases = fixture["cases"]
            .as_array()
            .ok_or_else(|| anyhow::anyhow!("DS4 tokenizer cases are missing"))?;
        assert_eq!(cases.len(), 12);
        for (index, case) in cases.iter().enumerate() {
            let text = case["text"]
                .as_str()
                .ok_or_else(|| anyhow::anyhow!("tokenizer case {index} has no text"))?;
            let expected = case["ids"]
                .as_array()
                .ok_or_else(|| anyhow::anyhow!("tokenizer case {index} has no ids"))?
                .iter()
                .map(|id| {
                    id.as_u64()
                        .and_then(|id| u32::try_from(id).ok())
                        .ok_or_else(|| anyhow::anyhow!("tokenizer case {index} has an invalid id"))
                })
                .collect::<anyhow::Result<Vec<_>>>()?;
            assert_eq!(
                tokenizer.encode(text, false)?,
                expected,
                "tokenizer case {index}"
            );
        }
        Ok(())
    }

    #[test]
    fn glm_generation_prompt_matches_ds4_wire_format() {
        assert_eq!(
            glm_generation_prompt(
                "Hello",
                Some("You are a helpful assistant"),
                GlmReasoning::High,
            ),
            "[gMASK]<sop><|system|>Reasoning Effort: High<|system|>You are a helpful assistant<|user|>Hello<|assistant|><think>"
        );
        assert_eq!(
            glm_generation_prompt("Hello", None, GlmReasoning::Max),
            "[gMASK]<sop><|system|>Reasoning Effort: Max<|user|>Hello<|assistant|><think>"
        );
        assert_eq!(
            glm_generation_prompt("Hello", None, GlmReasoning::Disabled),
            "[gMASK]<sop><|user|>Hello<|assistant|><think></think>"
        );
    }

    #[test]
    fn glm_generation_prompt_encodes_ds4_special_token_ids() -> anyhow::Result<()> {
        let Some(model_dir) = std::env::var_os("LISA_GLM53_MODEL") else {
            eprintln!("[glm53-prompt] skipping: LISA_GLM53_MODEL is unset");
            return Ok(());
        };
        let tokenizer = Tokenizer::load(std::path::Path::new(&model_dir))?;
        let special = |text: &str| {
            tokenizer
                .inner
                .token_to_id(text)
                .ok_or_else(|| anyhow::anyhow!("GLM tokenizer is missing {text}"))
        };
        let mut expected = vec![
            special("[gMASK]")?,
            special("<sop>")?,
            special("<|system|>")?,
        ];
        expected.extend(tokenizer.encode("Reasoning Effort: High", false)?);
        expected.push(special("<|system|>")?);
        expected.extend(tokenizer.encode("You are a helpful assistant", false)?);
        expected.push(special("<|user|>")?);
        expected.extend(tokenizer.encode("Hello", false)?);
        expected.push(special("<|assistant|>")?);
        expected.push(special("<think>")?);

        assert_eq!(
            tokenizer.encode_special(&glm_generation_prompt(
                "Hello",
                Some("You are a helpful assistant"),
                GlmReasoning::High,
            ))?,
            expected
        );
        Ok(())
    }
}
