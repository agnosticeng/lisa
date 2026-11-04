//! The pinned DS4 JoyAI/GLM4 pre-tokenizers and leftmost-minimum-rank BPE.
//! Vocabulary, merges and stop IDs come from the GGUF itself. In particular,
//! a Unicode-regex tokenizer is not substituted for DS4's manual split rules.

use anyhow::{Context, Result, ensure};
use lisa_gguf::{Header, Value};
use std::collections::HashMap;

#[path = "gguf_glm4.rs"]
mod glm4;

pub struct Tokenizer {
    glm4: bool,
    tokens: Vec<String>,
    ids: HashMap<String, u32>,
    ranks: HashMap<(String, String), usize>,
    bytes: [char; 256],
    inverse: HashMap<char, u8>,
    pub bos: u32,
    pub eos: u32,
}

impl Tokenizer {
    pub fn is_glm4(&self) -> bool {
        self.glm4
    }
    pub fn decoder_view(&self) -> Result<tokenizers::Tokenizer> {
        let mut merges: Vec<_> = self.ranks.iter().collect();
        merges.sort_by_key(|(_, rank)| **rank);
        let merges: Vec<_> = merges.into_iter().map(|((a, b), _)| vec![a, b]).collect();
        let added: Vec<_> = self
            .specials()
            .into_iter()
            .filter(|(text, id)| self.tokens[*id as usize] == *text)
            .map(|(text, id)| {
                serde_json::json!({ "id": id, "content": text, "single_word": false,
                "lstrip": false, "rstrip": false, "normalized": false, "special": true })
            })
            .collect();
        let data = serde_json::json!({ "version": "1.0", "added_tokens": added,
            "normalizer": null, "post_processor": null, "truncation": null, "padding": null,
            "pre_tokenizer": null,
            "decoder": { "type": "ByteLevel", "add_prefix_space": false, "trim_offsets": false, "use_regex": false },
            "model": { "type": "BPE", "vocab": self.ids, "merges": merges, "ignore_merges": false } });
        tokenizers::Tokenizer::from_bytes(serde_json::to_vec(&data)?)
            .map_err(|e| anyhow::anyhow!("GGUF tokenizer decoder: {e}"))
    }

    fn specials(&self) -> Vec<(&'static str, u32)> {
        if self.glm4 {
            return [
                "[gMASK]",
                "<sop>",
                "<|system|>",
                "<|user|>",
                "<|assistant|>",
                "<|observation|>",
                "<|endoftext|>",
                "<think>",
                "</think>",
            ]
            .into_iter()
            .filter_map(|s| self.ids.get(s).map(|&id| (s, id)))
            .collect();
        }
        let mut out: Vec<_> = [
            "<｜begin▁of▁sentence｜>",
            "<｜end▁of▁sentence｜>",
            "<｜System｜>",
            "<｜User｜>",
            "<｜Assistant｜>",
            "<think>",
            "</think>",
            "｜DSML｜",
        ]
        .into_iter()
        .filter_map(|s| self.ids.get(s).map(|&id| (s, id)))
        .collect();
        // DS4's rendered-chat scanner accepts these aliases too.
        for (alias, original) in [
            ("[gMASK]", "<｜begin▁of▁sentence｜>"),
            ("<|system|>", "<｜System｜>"),
            ("<|user|>", "<｜User｜>"),
            ("<|assistant|>", "<｜Assistant｜>"),
        ] {
            if let Some(&id) = self.ids.get(original) {
                out.push((alias, id));
            }
        }
        out
    }

    pub fn encode_rendered(&self, text: &str) -> Result<Vec<u32>> {
        let specials = self.specials();
        let mut out = Vec::new();
        let mut tail = text;
        while !tail.is_empty() {
            let next = specials
                .iter()
                .filter_map(|&(marker, id)| tail.find(marker).map(|at| (at, marker, id)))
                .min_by_key(|&(at, _, _)| at);
            let Some((at, marker, id)) = next else {
                out.extend(self.encode(tail)?);
                break;
            };
            out.extend(self.encode(&tail[..at])?);
            out.push(id);
            tail = &tail[at + marker.len()..];
        }
        Ok(out)
    }

    pub fn decode_clean(&self, ids: &[u32]) -> Result<Vec<u8>> {
        let specials = self.specials();
        let ids: Vec<u32> = ids
            .iter()
            .copied()
            .filter(|id| !specials.iter().any(|(_, special)| id == special))
            .collect();
        self.decode_bytes(&ids)
    }
    pub fn from_header(header: &Header) -> Result<Self> {
        let pre = header
            .metadata
            .get("tokenizer.ggml.pre")
            .and_then(Value::as_str);
        let glm4 = pre == Some("glm4");
        ensure!(
            header
                .metadata
                .get("tokenizer.ggml.model")
                .and_then(Value::as_str)
                == Some("gpt2")
                && (glm4 || pre == Some("joyai-llm")),
            "unsupported GGUF tokenizer"
        );
        let strings = |key| -> Result<Vec<String>> {
            header
                .metadata
                .get(key)
                .and_then(Value::as_array)
                .context("missing GGUF tokenizer array")?
                .iter()
                .map(|v| {
                    v.as_str()
                        .map(str::to_owned)
                        .context("invalid GGUF tokenizer string")
                })
                .collect()
        };
        let tokens = strings("tokenizer.ggml.tokens")?;
        let mut ids = HashMap::new();
        for (index, token) in tokens.iter().enumerate() {
            ensure!(
                ids.insert(token.clone(), index as u32).is_none(),
                "duplicate GGUF vocabulary entry"
            );
        }
        let mut ranks = HashMap::new();
        for (rank, merge) in strings("tokenizer.ggml.merges")?.iter().enumerate() {
            let (left, right) = merge.split_once(' ').context("invalid GGUF BPE merge")?;
            ensure!(
                !left.is_empty() && !right.is_empty() && !right.contains(' '),
                "invalid GGUF BPE merge symbols"
            );
            ensure!(
                ids.contains_key(left)
                    && ids.contains_key(right)
                    && ids.contains_key(&format!("{left}{right}")),
                "GGUF BPE merge references a missing vocabulary symbol"
            );
            // DS4's table_put replaces the value for a duplicate merge.
            ranks.insert((left.to_owned(), right.to_owned()), rank);
        }
        let mut escape = 256u32;
        let bytes = std::array::from_fn(|i| {
            let visible = (33..=126).contains(&i) || (161..=172).contains(&i) || i >= 174;
            let cp = if visible {
                i as u32
            } else {
                let cp = escape;
                escape += 1;
                cp
            };
            char::from_u32(cp).expect("byte alphabet is valid Unicode")
        });
        for ch in bytes {
            ensure!(
                ids.contains_key(&ch.to_string()),
                "incomplete GGUF byte vocabulary"
            );
        }
        let inverse = bytes
            .iter()
            .enumerate()
            .map(|(b, &c)| (c, b as u8))
            .collect();
        let bos = *ids
            .get(if glm4 {
                "[gMASK]"
            } else {
                "<｜begin▁of▁sentence｜>"
            })
            .context("missing DeepSeek BOS")?;
        let eos = *ids
            .get(if glm4 {
                "<|endoftext|>"
            } else {
                "<｜end▁of▁sentence｜>"
            })
            .context("missing DeepSeek EOS")?;
        for (key, id) in [
            ("tokenizer.ggml.bos_token_id", bos),
            ("tokenizer.ggml.eos_token_id", eos),
        ] {
            ensure!(
                header.metadata.get(key).and_then(Value::as_u64) == Some(id as u64),
                "DeepSeek special token metadata mismatch"
            );
        }
        Ok(Self {
            glm4,
            tokens,
            ids,
            ranks,
            bytes,
            inverse,
            bos,
            eos,
        })
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        ensure!(
            !text.contains('\0'),
            "DS4 text tokenizer does not accept embedded NUL"
        );
        let mut result = Vec::new();
        for piece in if self.glm4 {
            glm4::pieces(text)
        } else {
            pieces(text)
        } {
            let mut symbols: Vec<String> = piece
                .bytes()
                .map(|b| self.bytes[b as usize].to_string())
                .collect();
            loop {
                let mut best = None;
                for i in 0..symbols.len().saturating_sub(1) {
                    if let Some(&rank) = self
                        .ranks
                        .get(&(symbols[i].clone(), symbols[i + 1].clone()))
                    {
                        if best.is_none_or(|(_, old)| rank < old) {
                            best = Some((i, rank));
                        }
                    }
                }
                let Some((index, _)) = best else { break };
                let right = symbols.remove(index + 1);
                symbols[index].push_str(&right);
            }
            for symbol in symbols {
                result.push(
                    *self
                        .ids
                        .get(&symbol)
                        .context("BPE produced a missing vocabulary symbol")?,
                );
            }
        }
        Ok(result)
    }

    /// DS4's single-user-turn template, including its reasoning prefix.
    pub fn chat(&self, system: Option<&str>, prompt: &str, effort: u8) -> Result<Vec<u32>> {
        if self.glm4 {
            let reasoning = if effort == 0 {
                super::tokenizer::GlmReasoning::Disabled
            } else {
                super::tokenizer::GlmReasoning::High
            };
            return self.encode_rendered(&super::tokenizer::glm_generation_prompt(
                prompt, system, reasoning,
            ));
        }
        ensure!(
            effort <= 100,
            "DeepSeek reasoning effort must be between 0 and 100"
        );
        let id = |token: &str| {
            self.ids
                .get(token)
                .copied()
                .with_context(|| format!("missing DeepSeek marker {token}"))
        };
        let mut out = vec![self.bos];
        if effort > 0 {
            out.push(id("<｜System｜>")?);
            out.extend(self.encode(&format!("Reasoning Effort: {effort} (range 1-100, the higher the value, the more thorough the reasoning)\n\n"))?);
        }
        if let Some(system) = system.filter(|s| !s.is_empty()) {
            if effort == 0 {
                out.push(id("<｜System｜>")?);
            }
            out.extend(self.encode(system)?);
        }
        out.push(id("<｜User｜>")?);
        out.extend(self.encode(prompt)?);
        out.push(id("<｜Assistant｜>")?);
        out.push(id(if effort > 0 { "<think>" } else { "</think>" })?);
        Ok(out)
    }

    /// Bytes, not lossy per-token UTF-8: callers retain incomplete sequences
    /// across token boundaries when streaming text.
    pub fn decode_bytes(&self, ids: &[u32]) -> Result<Vec<u8>> {
        let mut out = Vec::new();
        for &id in ids {
            let token = self
                .tokens
                .get(id as usize)
                .context("invalid DeepSeek token ID")?;
            for ch in token.chars() {
                if let Some(&byte) = self.inverse.get(&ch) {
                    out.push(byte);
                } else {
                    let mut buf = [0u8; 4];
                    out.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
                }
            }
        }
        Ok(out)
    }
}

fn punct(c: u8) -> bool {
    matches!(c, b'!'..=b'/' | b':'..=b'@' | b'['..=b'`' | b'{'..=b'~')
}
fn space(c: u8) -> bool {
    matches!(c, b' ' | b'\t' | b'\n' | b'\r' | 11 | 12)
}
fn newline(c: u8) -> bool {
    matches!(c, b'\r' | b'\n')
}
fn letter(c: u8) -> bool {
    c.is_ascii_alphabetic() || c >= 128
}
fn cjk(text: &str) -> bool {
    text.chars()
        .next()
        .is_some_and(|c| matches!(c as u32, 0x4e00..=0x9fa5 | 0x3040..=0x309f | 0x30a0..=0x30ff))
}

fn pieces(text: &str) -> Vec<&str> {
    let bytes = text.as_bytes();
    let (mut pos, len) = (0, bytes.len());
    let mut result = Vec::new();
    while pos < len {
        let start = pos;
        let c = bytes[pos];
        if c.is_ascii_digit() {
            while pos < len && bytes[pos].is_ascii_digit() && pos - start < 3 {
                pos += 1;
            }
        } else if cjk(&text[pos..]) {
            loop {
                pos += text[pos..].chars().next().unwrap().len_utf8();
                if pos == len || !cjk(&text[pos..]) {
                    break;
                }
            }
        } else if punct(c) && pos + 1 < len && bytes[pos + 1].is_ascii_alphabetic() {
            pos += 1;
            while pos < len && bytes[pos].is_ascii_alphabetic() {
                pos += 1;
            }
        } else if letter(c) {
            while pos < len && letter(bytes[pos]) {
                pos += text[pos..].chars().next().unwrap().len_utf8();
            }
        } else if !newline(c) && !punct(c) && pos + 1 < len && letter(bytes[pos + 1]) {
            pos += 1;
            while pos < len && letter(bytes[pos]) {
                pos += text[pos..].chars().next().unwrap().len_utf8();
            }
        } else if c == b' ' && pos + 1 < len && punct(bytes[pos + 1]) {
            pos += 1;
            while pos < len && punct(bytes[pos]) {
                pos += 1;
            }
            while pos < len && newline(bytes[pos]) {
                pos += 1;
            }
        } else if punct(c) {
            while pos < len && punct(bytes[pos]) {
                pos += 1;
            }
            while pos < len && newline(bytes[pos]) {
                pos += 1;
            }
        } else if space(c) {
            let (mut p, mut last_newline) = (pos, 0);
            while p < len && space(bytes[p]) {
                if newline(bytes[p]) {
                    last_newline = p + 1;
                }
                p += 1;
            }
            pos = if last_newline != 0 {
                last_newline
            } else if p < len && p > pos + 1 && (letter(bytes[p]) || punct(bytes[p])) {
                p - 1
            } else {
                p
            };
        } else {
            pos += text[pos..].chars().next().unwrap().len_utf8();
        }
        result.push(&text[start..pos]);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deepseek41_ds4_joyai_boundaries() {
        assert_eq!(
            pieces("    int x = 1234567;\n\n"),
            ["   ", " int", " x", " =", " ", "123", "456", "7", ";\n\n"]
        );
        assert_eq!(
            pieces("émoji🙂中abc 中文abc .foo!\n"),
            ["émoji🙂中abc", " 中文abc", " .", "foo", "!\n"]
        );
        assert_eq!(pieces("中文abc"), ["中文", "abc"]);
        assert_eq!(
            pieces("a\r\n   \tb\u{000b}  "),
            ["a", "\r\n", "   ", "\tb", "\u{000b}  "]
        );
    }
}

#[cfg(test)]
#[path = "../models/deepseek41/tokenizer_tests.rs"]
mod checkpoint_tests;

#[cfg(test)]
#[path = "gguf_glm4_tests.rs"]
mod glm4_checkpoint_tests;
