//! Request-local parser cursors for growing decoded output.

use super::DsmlDecodeState;

#[derive(Clone, Copy)]
enum Phase {
    Thinking,
    Text,
    Structure,
    Payload(&'static str),
    Done,
}

pub(crate) struct DsmlParser {
    phase: Phase,
    position: usize,
    native: Option<super::deepseek41::Parser>,
}

fn suffix_start(text: &str, keep: usize) -> usize {
    let mut start = text.len().saturating_sub(keep);
    while !text.is_char_boundary(start) {
        start -= 1;
    }
    start
}

impl DsmlParser {
    pub(crate) fn new(thinking: bool) -> Self {
        Self {
            phase: if thinking {
                Phase::Thinking
            } else {
                Phase::Text
            },
            position: 0,
            native: None,
        }
    }

    pub(crate) fn for_template(thinking: bool, deepseek41: bool) -> Self {
        let mut parser = Self::new(thinking);
        if deepseek41 {
            parser.native = Some(super::deepseek41::Parser::new(thinking));
        }
        parser
    }

    /// Consume only newly appended text and an incomplete marker suffix.
    /// `IncrementalDecoder` may rewrite an incomplete UTF-8 suffix; a final
    /// replacement character is never committed into this cursor.
    pub(crate) fn update(&mut self, text: &str) -> DsmlDecodeState {
        if let Some(native) = &mut self.native {
            return native.update(text);
        }
        loop {
            let tail = &text[self.position..];
            match self.phase {
                Phase::Thinking | Phase::Text => {
                    let marker = if matches!(self.phase, Phase::Thinking) {
                        "</think>"
                    } else {
                        "<tool_call>"
                    };
                    if let Some(relative) = tail.find(marker) {
                        self.position += relative + marker.len();
                        self.phase = if matches!(self.phase, Phase::Thinking) {
                            Phase::Text
                        } else {
                            Phase::Structure
                        };
                    } else {
                        self.position = self.position.max(suffix_start(text, marker.len() - 1));
                        return DsmlDecodeState::Configured;
                    }
                }
                Phase::Payload(close) => {
                    if let Some(relative) = tail.find(close) {
                        self.position += relative + close.len();
                        self.phase = Phase::Structure;
                    } else {
                        let partial = (1..=tail.len().min(close.len() - 1))
                            .rev()
                            .find(|&len| tail.ends_with(&close[..len]))
                            .unwrap_or(0);
                        self.position = self.position.max(suffix_start(text, close.len() - 1));
                        return if partial >= 2 {
                            DsmlDecodeState::Greedy
                        } else {
                            DsmlDecodeState::Configured
                        };
                    }
                }
                Phase::Structure => {
                    if tail.is_empty() || tail == "\u{FFFD}" {
                        return DsmlDecodeState::Greedy;
                    }
                    if tail.starts_with("</tool_call>") {
                        self.phase = Phase::Done;
                        continue;
                    }
                    if tail.starts_with("<arg_value>") {
                        self.position += "<arg_value>".len();
                        self.phase = Phase::Payload("</arg_value>");
                    } else if tail.starts_with("<parameter=") || tail.starts_with("<parameter ") {
                        let Some(end) = tail.find('>') else {
                            return DsmlDecodeState::Greedy;
                        };
                        self.position += end + 1;
                        self.phase = Phase::Payload("</parameter>");
                    } else if ["</tool_call>", "<arg_value>", "<parameter=", "<parameter "]
                        .iter()
                        .any(|marker| marker.starts_with(tail))
                    {
                        return DsmlDecodeState::Greedy;
                    } else {
                        self.position += tail.chars().next().expect("nonempty").len_utf8();
                    }
                }
                Phase::Done => return DsmlDecodeState::Done,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn incremental_dsml_matches_reference_at_every_character() {
        for thinking in [false, true] {
            for text in [
                "normal é🙂 text <tool_call><arg_key>x</arg_key><arg_value>hello 🙂</arg_value></tool_call>",
                "<think>plan</think><tool_call><function=f><parameter=x>abc</parameter></function></tool_call>",
                "</think><tool_call><parameter name=\"x\">abc</parameter></tool_call>",
                "<tool_call><arg_value>abc<abc</arg_</arg_value>stuff</tool_call>tail",
                "<think>unfinished <tool_call><arg_value>value</arg_value>",
                "<tool_call><parameter=x unfinished",
            ] {
                let mut parser = DsmlParser::new(thinking);
                for end in (0..=text.len()).filter(|&end| text.is_char_boundary(end)) {
                    assert!(
                        parser.update(&text[..end])
                            == super::super::dsml_decode_state(&text[..end], thinking),
                        "thinking={thinking} prefix={:?}",
                        &text[..end]
                    );
                }
            }
        }
    }

    #[test]
    fn incremental_dsml_matches_reference_with_rewritten_utf8_suffixes() {
        for thinking in [false, true] {
            for text in [
                "🙂</think>héllo<tool_call>🙂<arg_value>é🙂</arg_value>é</tool_call>",
                "</think><tool_call>�<parameter=x>�🙂</parameter></tool_call>",
                "</think><tool_call><arg_value>text </arg_é</arg_value></tool_call>",
            ] {
                let mut parser = DsmlParser::new(thinking);
                for end in 0..=text.len() {
                    let prefix = String::from_utf8_lossy(&text.as_bytes()[..end]);
                    assert!(
                        parser.update(&prefix)
                            == super::super::dsml_decode_state(&prefix, thinking),
                        "thinking={thinking} prefix={prefix:?}"
                    );
                }
            }
        }
    }
}
