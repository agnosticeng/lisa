//! Tokenizer wrapper and ChatML template application.

use tokenizers::Tokenizer as HfTokenizer;

pub struct Tokenizer {
    pub inner: HfTokenizer,
    pub im_end_ids: Vec<u32>,
    pub(crate) lfm3: bool,
    pub(crate) native: Option<crate::core::native_tokenizer::Tokenizer>,
}

/// Incremental decoding with the same text as `Tokenizer::decode` at every
/// token boundary, including incomplete UTF-8. Qwen and GLM use ByteLevel;
/// other decoder types retain the tokenizer's complete-sequence fallback.
pub struct IncrementalDecoder<'a> {
    tokenizer: &'a Tokenizer,
    skip_special_tokens: bool,
    byte_level: bool,
    ids: Vec<u32>,
    pending: Vec<u8>,
    text: String,
    stable_len: usize,
}

impl<'a> IncrementalDecoder<'a> {
    pub fn new(tokenizer: &'a Tokenizer, skip_special_tokens: bool) -> Self {
        Self {
            tokenizer,
            skip_special_tokens,
            byte_level: matches!(
                tokenizer.inner.get_decoder(),
                Some(tokenizers::DecoderWrapper::ByteLevel(_))
            ),
            ids: Vec::new(),
            pending: Vec::new(),
            text: String::new(),
            stable_len: 0,
        }
    }

    pub fn text(&self) -> &str {
        &self.text
    }

    /// The suffix after this boundary may change when a split UTF-8 character
    /// is completed. Consumers that cannot retract text should hold it back.
    pub fn stable_text(&self) -> &str {
        &self.text[..self.stable_len]
    }

    pub fn push(&mut self, id: u32) -> anyhow::Result<&str> {
        if !self.byte_level {
            self.ids.push(id);
            self.text = self
                .tokenizer
                .inner
                .decode(&self.ids, self.skip_special_tokens)
                .map_err(|error| anyhow::anyhow!("decode: {error}"))?;
            self.stable_len = self.text.len();
            return Ok(&self.text);
        }
        let Some(token) = self.tokenizer.inner.id_to_token(id) else {
            return Ok(&self.text);
        };
        if self.skip_special_tokens
            && self
                .tokenizer
                .inner
                .get_added_vocabulary()
                .is_special_token(&token)
        {
            return Ok(&self.text);
        }
        // ByteLevel maps the visible Latin-1 alphabet directly and assigns
        // the remaining bytes consecutive code points starting at U+0100.
        static BYTES: std::sync::OnceLock<[Option<u8>; 512]> = std::sync::OnceLock::new();
        let bytes = BYTES.get_or_init(|| {
            let mut map = [None; 512];
            let mut escaped = 256;
            for byte in 0u16..=255 {
                let visible =
                    (33..=126).contains(&byte) || (161..=172).contains(&byte) || byte >= 174;
                let code = if visible {
                    byte as usize
                } else {
                    let code = escaped;
                    escaped += 1;
                    code
                };
                map[code] = Some(byte as u8);
            }
            map
        });
        let start = self.pending.len();
        for ch in token.chars() {
            if let Some(Some(byte)) = bytes.get(ch as usize) {
                self.pending.push(*byte);
            } else {
                // The ByteLevel decoder preserves the entire literal token
                // if any character is outside its byte alphabet.
                self.pending.truncate(start);
                self.pending.extend_from_slice(token.as_bytes());
                break;
            }
        }
        self.text.truncate(self.stable_len);
        let mut consumed = 0;
        while consumed < self.pending.len() {
            match std::str::from_utf8(&self.pending[consumed..]) {
                Ok(valid) => {
                    self.text.push_str(valid);
                    consumed = self.pending.len();
                }
                Err(error) => {
                    let end = consumed + error.valid_up_to();
                    self.text.push_str(
                        std::str::from_utf8(&self.pending[consumed..end]).expect("valid prefix"),
                    );
                    consumed = end;
                    if let Some(length) = error.error_len() {
                        self.text.push('\u{FFFD}');
                        consumed += length;
                    } else {
                        break;
                    }
                }
            }
        }
        self.pending.drain(..consumed);
        self.stable_len = self.text.len();
        if !self.pending.is_empty() {
            self.text.push('\u{FFFD}');
        }
        Ok(&self.text)
    }
}

impl Tokenizer {
    pub fn is_lfm3(&self) -> bool {
        self.lfm3
    }
    pub fn encode(&self, text: &str, add_bos: bool) -> anyhow::Result<Vec<u32>> {
        if let Some(native) = &self.native {
            let mut ids = native.encode_rendered(text)?;
            if add_bos {
                ids.insert(0, native.bos);
            }
            return Ok(ids);
        }
        let mut ids = self
            .inner
            .encode(text, false)
            .map_err(|e| anyhow::anyhow!("encode: {e}"))?
            .get_ids()
            .to_vec();
        if add_bos && let Some(bos) = self.inner.token_to_id("<|im_start|>") {
            ids.insert(0, bos);
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
        if let Some(native) = &self.native {
            return native.encode_rendered(text);
        }
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
        if let Some(native) = &self.native {
            return Ok(String::from_utf8_lossy(&native.decode_bytes(ids)?).into_owned());
        }
        self.inner
            .decode(ids, false)
            .map_err(|e| anyhow::anyhow!("decode: {e}"))
    }

    /// Decode without rendering special tokens (`<im_end>` and friends).
    pub fn decode_clean(&self, ids: &[u32]) -> anyhow::Result<String> {
        if let Some(native) = &self.native {
            return Ok(String::from_utf8_lossy(&native.decode_clean(ids)?).into_owned());
        }
        self.inner
            .decode(ids, true)
            .map_err(|e| anyhow::anyhow!("decode: {e}"))
    }
}

impl Tokenizer {
    /// Native no-thinking turn for a persistent chat. Generation stops before
    /// EOS, so subsequent turns close the previous assistant and omit BOS.
    pub fn native_chat_turn(
        &self,
        system: Option<&str>,
        prompt: &str,
        first: bool,
    ) -> anyhow::Result<Option<Vec<u32>>> {
        if self.lfm3 {
            return Ok(Some(self.encode(
                &lisa_models::lfm3::prompt::turn(system, prompt, first, false),
                false,
            )?));
        }
        let Some(native) = &self.native else {
            return Ok(None);
        };
        if native.is_glm4() {
            let rendered = if first {
                glm_generation_prompt(prompt, system, ReasoningEffort::Disabled)
            } else {
                format!("<|user|>{prompt}<|assistant|><think></think>")
            };
            return Ok(Some(native.encode_rendered(&rendered)?));
        }
        let mut ids = native.chat(if first { system } else { None }, prompt, 0)?;
        if !first {
            ids[0] = native.eos;
        }
        Ok(Some(ids))
    }

    /// Preserve each the reference family's raw-text control-marker rules.
    pub fn encode_plain(&self, text: &str) -> anyhow::Result<Vec<u32>> {
        if let Some(native) = &self.native {
            if native.is_glm4() {
                native.encode_rendered(text)
            } else {
                native.encode(text)
            }
        } else {
            self.encode(text, false)
        }
    }
    pub fn native_chat(
        &self,
        system: Option<&str>,
        prompt: &str,
        effort: u8,
    ) -> anyhow::Result<Vec<u32>> {
        self.native
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("model has no native GGUF tokenizer"))?
            .chat(system, prompt, effort)
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
pub enum ReasoningEffort {
    High,
    Max,
    Disabled,
}

/// GLM-5.3 generation prompt, matching the reference's text-only chat prefix. The
/// control markers must be encoded as their special-token ids; rendering
/// lookalike separators as ordinary text changes the model input.
pub fn glm_generation_prompt(
    text: &str,
    system: Option<&str>,
    reasoning: ReasoningEffort,
) -> String {
    let mut out = String::from("[gMASK]<sop>");
    let effort = match reasoning {
        ReasoningEffort::High => Some("Reasoning Effort: High"),
        ReasoningEffort::Max => Some("Reasoning Effort: Max"),
        ReasoningEffort::Disabled => None,
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
    if reasoning == ReasoningEffort::Disabled {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn native_inkling_sampling_terminator_is_loaded_from_tokenizer() -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        let source = serde_json::json!({
            "version":"1.0", "truncation":null, "padding":null,
            "added_tokens":[], "normalizer":null, "pre_tokenizer":null,
            "post_processor":null, "decoder":null,
            "model":{"type":"WordLevel", "vocab":{
                "unknown":0, "<|content_model_end_sampling|>":1, "word":2
            }, "unk_token":"unknown"}
        });
        std::fs::write(
            dir.path().join("tokenizer.json"),
            serde_json::to_vec(&source)?,
        )?;
        let tokenizer = crate::checkpoint::tokenizer::load(dir.path())?;
        assert_eq!(tokenizer.im_end_ids, vec![1]);
        Ok(())
    }

    #[test]
    fn incremental_byte_level_matches_full_decode_at_every_token() -> anyhow::Result<()> {
        let mut vocab = serde_json::Map::new();
        let mut escaped = 256u32;
        for byte in 0u32..=255 {
            let visible = (33..=126).contains(&byte) || (161..=172).contains(&byte) || byte >= 174;
            let code = if visible {
                byte
            } else {
                let code = escaped;
                escaped += 1;
                code
            };
            vocab.insert(char::from_u32(code).unwrap().to_string(), byte.into());
        }
        vocab.insert("<special>".into(), 256.into());
        vocab.insert("literal🙂token".into(), 257.into());
        let source = serde_json::json!({
            "version":"1.0", "truncation":null, "padding":null,
            "added_tokens":[{"id":256,"content":"<special>","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true}],
            "normalizer":null,"pre_tokenizer":null,"post_processor":null,
            "decoder":{"type":"ByteLevel","add_prefix_space":true,"trim_offsets":true,"use_regex":true},
            "model":{"type":"WordLevel","vocab":vocab,"unk_token":"<special>"}
        });
        let inner = HfTokenizer::from_bytes(serde_json::to_vec(&source)?)
            .map_err(|error| anyhow::anyhow!("{error}"))?;
        let tokenizer = Tokenizer {
            native: None,
            lfm3: false,
            inner,
            im_end_ids: vec![],
        };
        // Pairs covering all byte values exercise valid, invalid and incomplete
        // UTF-8; add split multibyte sequences, specials and unknown IDs.
        let mut ids = Vec::new();
        for byte in 0..256 {
            ids.extend([byte, (byte * 137) % 256]);
        }
        ids.extend(
            "é🙂中文<think>reason</think><tool_call>"
                .bytes()
                .map(u32::from),
        );
        ids.extend([256, 257, 999999, 0xf0, 0x9f, 256, 0x99, 0x82, 0xe2]);
        for skip in [false, true] {
            let mut incremental = IncrementalDecoder::new(&tokenizer, skip);
            let mut streamed = String::new();
            for (index, &id) in ids.iter().enumerate() {
                let expected = tokenizer.inner.decode(&ids[..=index], skip).unwrap();
                assert_eq!(incremental.push(id)?, expected, "skip={skip} index={index}");
                let stable = incremental.stable_text();
                assert!(
                    stable.starts_with(&streamed),
                    "a sent prefix cannot be retracted"
                );
                streamed.push_str(&stable[streamed.len()..]);
            }
            // End-of-generation flush includes any permanently incomplete
            // suffix using the tokenizer's replacement-character semantics.
            streamed.push_str(&incremental.text()[streamed.len()..]);
            assert_eq!(streamed, tokenizer.inner.decode(&ids, skip).unwrap());
        }
        // Unknown decoder kinds retain their original joining semantics.
        let mut tokenizer = tokenizer;
        tokenizer
            .inner
            .with_decoder(None::<tokenizers::DecoderWrapper>);
        let mut incremental = IncrementalDecoder::new(&tokenizer, false);
        for (index, &id) in ids[..32].iter().enumerate() {
            assert_eq!(incremental.push(id)?, tokenizer.decode(&ids[..=index])?);
        }
        Ok(())
    }

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
        let ds4_tokenizer = crate::checkpoint::tokenizer::load_glm53_tokenizer_bytes(source)?;
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
    fn glm_generation_prompt_matches_ds4_wire_format() {
        assert_eq!(
            glm_generation_prompt(
                "Hello",
                Some("You are a helpful assistant"),
                ReasoningEffort::High,
            ),
            "[gMASK]<sop><|system|>Reasoning Effort: High<|system|>You are a helpful assistant<|user|>Hello<|assistant|><think>"
        );
        assert_eq!(
            glm_generation_prompt("Hello", None, ReasoningEffort::Max),
            "[gMASK]<sop><|system|>Reasoning Effort: Max<|user|>Hello<|assistant|><think>"
        );
        assert_eq!(
            glm_generation_prompt("Hello", None, ReasoningEffort::Disabled),
            "[gMASK]<sop><|user|>Hello<|assistant|><think></think>"
        );
    }
}
