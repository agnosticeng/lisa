//! Text-only incremental turns matching the checkpoint template; tools use the full Jinja template.
pub fn turn(system: Option<&str>, user: &str, first: bool, thinking: bool) -> String {
    let mut out = if first {
        "<|startoftext|>".to_owned()
    } else {
        "<|im_end|>\n".to_owned()
    };
    if first {
        if let Some(s) = system.map(str::trim).filter(|s| !s.is_empty()) {
            out.push_str(&format!("<|im_start|>system\n{s}<|im_end|>\n"));
        }
    }
    out.push_str(&format!(
        "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n<think>",
        user.trim()
    ));
    if !thinking {
        out.push_str("</think>");
    }
    out
}
