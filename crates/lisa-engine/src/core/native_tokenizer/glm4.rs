//! DS4 b3f50180 GLM4 split rules and explicit Unicode classes.
fn whitespace(c: char) -> bool {
    let cp = c as u32;
    if cp < 128 {
        return matches!(c, ' ' | '\t' | '\n' | '\r' | '\u{b}' | '\u{c}');
    }
    cp == 0x0085
        || cp == 0x00a0
        || cp == 0x1680
        || (0x2000..=0x200a).contains(&cp)
        || cp == 0x2028
        || cp == 0x2029
        || cp == 0x202f
        || cp == 0x205f
        || cp == 0x3000
}
fn number(c: char) -> bool {
    let cp = c as u32;
    if cp < 128 {
        return c.is_ascii_digit();
    }
    (0x0660..=0x0669).contains(&cp)
        || (0x06f0..=0x06f9).contains(&cp)
        || (0x07c0..=0x07c9).contains(&cp)
        || (0x0966..=0x096f).contains(&cp)
        || (0x09e6..=0x09ef).contains(&cp)
        || (0x0a66..=0x0a6f).contains(&cp)
        || (0x0ae6..=0x0aef).contains(&cp)
        || (0x0b66..=0x0b6f).contains(&cp)
        || (0x0be6..=0x0bef).contains(&cp)
        || (0x0c66..=0x0c6f).contains(&cp)
        || (0x0ce6..=0x0cef).contains(&cp)
        || (0x0d66..=0x0d6f).contains(&cp)
        || (0x0de6..=0x0def).contains(&cp)
        || (0x0e50..=0x0e59).contains(&cp)
        || (0x0ed0..=0x0ed9).contains(&cp)
        || (0x0f20..=0x0f29).contains(&cp)
        || (0x1040..=0x1049).contains(&cp)
        || (0x1090..=0x1099).contains(&cp)
        || (0x17e0..=0x17e9).contains(&cp)
        || (0x1810..=0x1819).contains(&cp)
        || (0xff10..=0xff19).contains(&cp)
}
fn punct_symbol(c: char) -> bool {
    let cp = c as u32;
    if cp < 128 {
        return c.is_ascii_punctuation();
    }
    (0x00a1..=0x00a9).contains(&cp)
        || (0x00ab..=0x00ac).contains(&cp)
        || (0x00ae..=0x00b1).contains(&cp)
        || cp == 0x00b4
        || (0x00b6..=0x00b8).contains(&cp)
        || cp == 0x00bb
        || cp == 0x00bf
        || cp == 0x00d7
        || cp == 0x00f7
        || (0x02c2..=0x02df).contains(&cp)
        || (0x02e5..=0x02eb).contains(&cp)
        || (0x02ed..=0x02ff).contains(&cp)
        || (0x0375..=0x037e).contains(&cp)
        || (0x0384..=0x0385).contains(&cp)
        || cp == 0x0387
        || (0x055a..=0x055f).contains(&cp)
        || (0x0589..=0x058a).contains(&cp)
        || (0x05be..=0x05c0).contains(&cp)
        || cp == 0x05c3
        || (0x05c6..=0x05c7).contains(&cp)
        || (0x0609..=0x060a).contains(&cp)
        || (0x060c..=0x060d).contains(&cp)
        || cp == 0x061b
        || (0x061e..=0x061f).contains(&cp)
        || cp == 0x066a
        || cp == 0x066d
        || cp == 0x06d4
        || (0x2000..=0x206f).contains(&cp)
        || (0x20a0..=0x20cf).contains(&cp)
        || (0x2100..=0x214f).contains(&cp)
        || (0x2190..=0x23ff).contains(&cp)
        || (0x2460..=0x24ff).contains(&cp)
        || (0x2500..=0x2775).contains(&cp)
        || (0x2794..=0x2bff).contains(&cp)
        || (0x2e00..=0x2e7f).contains(&cp)
        || (0x3000..=0x303f).contains(&cp)
        || (0xfd3e..=0xfd3f).contains(&cp)
        || (0xfe10..=0xfe6f).contains(&cp)
        || (0xff01..=0xff0f).contains(&cp)
        || (0xff1a..=0xff20).contains(&cp)
        || (0xff3b..=0xff40).contains(&cp)
        || (0xff5b..=0xff65).contains(&cp)
        || (0x1f000..=0x1faff).contains(&cp)
}
fn letter(c: char) -> bool {
    if c.is_ascii() {
        c.is_ascii_alphabetic()
    } else {
        !whitespace(c) && !number(c) && !punct_symbol(c)
    }
}
pub fn pieces(text: &str) -> Vec<&str> {
    let chars: Vec<_> = text.char_indices().collect();
    let len = chars.len();
    let mut pos = 0;
    let mut result = Vec::new();
    while pos < len {
        let start = pos;
        let c = chars[pos].1;
        if c == '\'' && pos + 1 < len {
            let n = chars[pos + 1].1.to_ascii_lowercase();
            let two = matches!(n, 's' | 't' | 'm' | 'd');
            let three = pos + 2 < len
                && matches!(
                    (n, chars[pos + 2].1.to_ascii_lowercase()),
                    ('r', 'e') | ('v', 'e') | ('l', 'l')
                );
            if two || three {
                pos += if two { 2 } else { 3 };
                result.push(&text[chars[start].0..chars.get(pos).map_or(text.len(), |c| c.0)]);
                continue;
            }
        }
        if !matches!(c, '\r' | '\n')
            && !number(c)
            && (letter(c) || pos + 1 < len && letter(chars[pos + 1].1))
        {
            pos += 1;
            while pos < len && letter(chars[pos].1) {
                pos += 1;
            }
        } else if number(c) {
            while pos < len && pos - start < 3 && number(chars[pos].1) {
                pos += 1;
            }
        } else {
            let punct_pos = pos + usize::from(c == ' ');
            if punct_pos < len
                && !whitespace(chars[punct_pos].1)
                && !letter(chars[punct_pos].1)
                && !number(chars[punct_pos].1)
            {
                pos = punct_pos;
                while pos < len
                    && !whitespace(chars[pos].1)
                    && !letter(chars[pos].1)
                    && !number(chars[pos].1)
                {
                    pos += 1;
                }
                while pos < len && matches!(chars[pos].1, '\r' | '\n') {
                    pos += 1;
                }
            } else if whitespace(c) {
                let mut p = pos;
                let mut last_newline = None;
                while p < len && whitespace(chars[p].1) {
                    if matches!(chars[p].1, '\r' | '\n') {
                        last_newline = Some(p + 1);
                    }
                    p += 1;
                }
                pos = last_newline.unwrap_or(if p - start > 1 && p < len { p - 1 } else { p });
            } else {
                pos += 1;
            }
        }
        result.push(&text[chars[start].0..chars.get(pos).map_or(text.len(), |c| c.0)]);
    }
    result
}
