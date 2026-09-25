use std::cell::{Cell, OnceCell, RefCell};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::thread;

use objc2::rc::Retained;
use objc2::runtime::{AnyObject, Bool, ProtocolObject, Sel};
use objc2::{
    define_class, msg_send, sel, AnyThread, ClassType, DefinedClass, MainThreadMarker,
    MainThreadOnly,
};
use objc2_app_kit::*;
use objc2_foundation::*;

use crate::ui;

#[derive(Clone, Debug)]
pub struct Model {
    pub repo_id: String,
    #[allow(dead_code)]
    pub model_type: String,
    pub path: String,
    pub local: bool,
    pub size: u64,
    pub too_big: bool,
}

/// Model types the engine can load (mirrors `models::load_dir`).
const SUPPORTED_MODEL_TYPES: &[&str] = &[
    "qwen4_exp",
    "qwen4_exp_text",
    "qwen3_5",
    "qwen3_5_text",
    "laya",
];

/// Slash commands offered in the chat input's "/" dropdown: `(name, help)`.
const CHAT_COMMANDS: &[(&str, &str)] = &[
    ("/clear", "Reset the conversation"),
    ("/help", "Show available commands"),
];

fn config_model_type(dir: &std::path::Path) -> Option<String> {
    let text = std::fs::read_to_string(dir.join("config.json")).ok()?;
    let v: serde_json::Value = serde_json::from_str(&text).ok()?;
    v.get("text_config")
        .and_then(|t| t.get("model_type"))
        .and_then(|m| m.as_str())
        .map(str::to_string)
        .or_else(|| {
            v.get("model_type")
                .and_then(|m| m.as_str())
                .map(str::to_string)
        })
}

fn display_name(repo_id: &str) -> &str {
    repo_id.strip_prefix("agnosticeng/").unwrap_or(repo_id)
}

/// Scan the local Hugging Face hub cache for checkpoints the engine supports.
fn discover_models() -> Vec<Model> {
    let hub = std::path::PathBuf::from(std::env::var("HOME").unwrap_or_default())
        .join(".cache/huggingface/hub");
    let mut out = Vec::new();
    let Ok(entries) = std::fs::read_dir(&hub) else {
        return out;
    };
    for entry in entries.flatten() {
        let name = entry.file_name().to_string_lossy().to_string();
        let Some(rest) = name.strip_prefix("models--") else {
            continue;
        };
        let repo_id = rest.replacen("--", "/", 1);
        if !repo_id.starts_with("agnosticeng/") {
            continue;
        }
        let Ok(snaps) = std::fs::read_dir(entry.path().join("snapshots")) else {
            continue;
        };
        for snap in snaps.flatten() {
            let path = snap.path();
            if !path.is_dir() {
                continue;
            }
            if let Some(mt) = config_model_type(&path) {
                if SUPPORTED_MODEL_TYPES.contains(&mt.as_str()) {
                    out.push(Model {
                        repo_id: repo_id.clone(),
                        model_type: mt,
                        path: path.to_string_lossy().to_string(),
                        local: true,
                        size: dir_size(&path),
                        too_big: false,
                    });
                    break;
                }
            }
        }
    }
    out.sort_by(|a, b| a.repo_id.cmp(&b.repo_id));
    out
}

/// Recursively total the bytes under `path` (following symlinks, e.g. the HF
/// hub's snapshot links).
fn dir_size(path: &std::path::Path) -> u64 {
    let mut total = 0u64;
    let Ok(entries) = std::fs::read_dir(path) else {
        return 0;
    };
    for e in entries.flatten() {
        let p = e.path();
        let Ok(md) = std::fs::metadata(&p) else {
            continue;
        };
        if md.is_dir() {
            total += dir_size(&p);
        } else {
            total += md.len();
        }
    }
    total
}

/// Physical memory in bytes (`sysctl hw.memsize`).
fn ram_total() -> u64 {
    std::process::Command::new("sysctl")
        .args(["-n", "hw.memsize"])
        .output()
        .ok()
        .and_then(|o| {
            String::from_utf8_lossy(&o.stdout)
                .trim()
                .parse::<u64>()
                .ok()
        })
        .unwrap_or(0)
}

fn curl_json(url: &str) -> Option<serde_json::Value> {
    let out = std::process::Command::new("curl")
        .args(["-sSL", "--max-time", "20", url])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    serde_json::from_slice(&out.stdout).ok()
}

/// The Hub cache directory for a repo (`models--org--name`).
fn hf_hub_dir(repo: &str) -> PathBuf {
    let hub = match std::env::var_os("HF_HOME") {
        Some(h) => PathBuf::from(h).join("hub"),
        None => PathBuf::from(std::env::var("HOME").unwrap_or_default())
            .join(".cache")
            .join("huggingface")
            .join("hub"),
    };
    hub.join(format!("models--{}", repo.replace('/', "--")))
}

/// Bytes already written for a repo's blobs (in-progress downloads included).
fn hf_blobs_size(repo: &str) -> u64 {
    dir_size(&hf_hub_dir(repo).join("blobs"))
}

/// Byte size of a Hub repo (sum of its files), 0 when unknown.
fn remote_size(id: &str) -> u64 {
    let url = format!("https://huggingface.co/api/models/{id}/tree/main?recursive=true&expand=true");
    let Some(v) = curl_json(&url) else {
        return 0;
    };
    let Some(arr) = v.as_array() else {
        return 0;
    };
    arr.iter()
        .filter(|e| e.get("type").and_then(|t| t.as_str()) == Some("file"))
        .filter_map(|e| e.get("size").and_then(|s| s.as_u64()))
        .sum()
}

/// Fetch the agnosticeng models (id, size) from the Hugging Face Hub (`curl`,
/// no HTTP dependency). Runs on a background thread; empty on failure.
fn fetch_remote_models() -> Vec<(String, u64)> {
    let Some(v) = curl_json("https://huggingface.co/api/models?author=agnosticeng&limit=200") else {
        return Vec::new();
    };
    let Some(arr) = v.as_array() else {
        return Vec::new();
    };
    arr.iter()
        .filter_map(|m| m.get("id").and_then(|x| x.as_str()).map(str::to_string))
        .filter(|id| {
            let l = id.to_lowercase();
            !l.contains("gguf") && !l.contains("dataset")
        })
        .map(|id| {
            let size = remote_size(&id);
            (id, size)
        })
        .collect()
}

/// Commands sent from the UI to the engine thread.
enum Command {
    Load(PathBuf),
    Download(String),
    /// Reset the conversation (drop the session state; keep the model loaded).
    Clear,
    Chat { text: String, system: String },
    /// Start the OpenAI-compatible server on the loaded model; runs until the
    /// shared `stop` flag is set, then the engine resumes handling commands.
    StartServe {
        addr: String,
        stop: Arc<AtomicBool>,
    },
}

/// Events pushed from the engine thread and drained on the main thread.
#[derive(Debug)]
enum UiEvent {
    Log(String),
    UserMessage(String),
    AssistantDelta(String),
    /// Replace the streaming assistant text wholesale (used when hidden
    /// reasoning is stripped and the visible reply no longer extends the
    /// previously emitted prefix).
    AssistantReset(String),
    AssistantDone,
    /// A model finished loading (true) or failed (false).
    Loaded(bool),
    /// The serve endpoint is listening on the given URL.
    ServeStarted(String),
    /// The serve endpoint stopped.
    ServeStopped,
    /// Remote model ids fetched from the Hub.
    RemoteModels(Vec<(String, u64)>),
    /// The local model set changed (e.g. a download finished).
    ModelsChanged,
}

/// Upcast any Foundation object whose superclass is `NSObject` to `AnyObject`.
fn as_any_object<T: ClassType<Super = NSObject>>(o: &T) -> &AnyObject {
    ClassType::as_super(ClassType::as_super(o))
}

/// Base style for a block of inline text.
#[derive(Clone, Copy, PartialEq)]
enum MdStyle {
    Base,
    Heading,
}

/// Build an attributed string for one inline-markdown run (no block layout).
/// `size` is the base body size; headings and math scale from it.
fn styled_attributed(
    mtm: MainThreadMarker,
    text: &str,
    color: &NSColor,
    base_style: MdStyle,
    size: f64,
) -> Retained<NSMutableAttributedString> {
    let base_font = NSFont::systemFontOfSize(size);
    let manager = NSFontManager::sharedFontManager(mtm);
    let bold_font = manager.convertFont_toHaveTrait(&base_font, NSFontTraitMask::BoldFontMask);
    let italic_font = manager.convertFont_toHaveTrait(&base_font, NSFontTraitMask::ItalicFontMask);
    let code_font =
        NSFont::userFixedPitchFontOfSize(size - 1.0).unwrap_or_else(|| base_font.clone());
    let heading_font = NSFont::boldSystemFontOfSize(size);

    let out = NSMutableAttributedString::initWithString(
        NSMutableAttributedString::alloc(),
        &NSString::from_str(""),
    );
    let font_key = NSString::from_str("NSFont");
    let lines: Vec<&str> = text.split('\n').collect();
    for (idx, line) in lines.iter().enumerate() {
        append_inline_attr(
            &out,
            mtm,
            line,
            base_style,
            &base_font,
            &bold_font,
            &italic_font,
            &code_font,
            &heading_font,
            size,
            color,
        );
        if idx + 1 < lines.len() {
            append_plain(&out, &font_key, "\n", &base_font);
        }
    }
    let color_key = NSString::from_str("NSColor");
    unsafe {
        out.addAttribute_value_range(
            &color_key,
            as_any_object(&*color),
            NSRange::new(0, out.length()),
        );
    }
    out
}

fn append_plain(
    out: &NSMutableAttributedString,
    font_key: &NSString,
    text: &str,
    font: &Retained<NSFont>,
) {
    if text.is_empty() {
        return;
    }
    let a = NSMutableAttributedString::initWithString(
        NSMutableAttributedString::alloc(),
        &NSString::from_str(text),
    );
    let len = a.length();
    unsafe {
        a.addAttribute_value_range(font_key, as_any_object(&**font), NSRange::new(0, len));
    }
    out.appendAttributedString(&a);
}

/// Row title for the "/" command menu: the command name in the label colour,
/// its description after a gap in the secondary colour. `selected` brightens
/// both runs so the keyboard-highlighted row stands out.
fn command_row_title(name: &str, desc: &str, selected: bool) -> Retained<NSMutableAttributedString> {
    let font_key = NSString::from_str("NSFont");
    let color_key = NSString::from_str("NSColor");
    let name_font = NSFont::boldSystemFontOfSize(13.0);
    let desc_font = NSFont::systemFontOfSize(12.0);
    let name_color = if selected {
        NSColor::whiteColor()
    } else {
        NSColor::labelColor()
    };
    let desc_color = if selected {
        NSColor::labelColor()
    } else {
        NSColor::secondaryLabelColor()
    };

    let out =
        NSMutableAttributedString::initWithString(NSMutableAttributedString::alloc(), ns_string!(""));
    // Keep the leading slash off the command word so it reads as a marker.
    let display = match name.strip_prefix('/') {
        Some(rest) => format!("/ {rest}"),
        None => name.to_string(),
    };
    let head = NSMutableAttributedString::initWithString(
        NSMutableAttributedString::alloc(),
        &NSString::from_str(&display),
    );
    let head_len = head.length();
    unsafe {
        head.addAttribute_value_range(
            &font_key,
            as_any_object(&*name_font),
            NSRange::new(0, head_len),
        );
        head.addAttribute_value_range(
            &color_key,
            as_any_object(&*name_color),
            NSRange::new(0, head_len),
        );
    }
    out.appendAttributedString(&head);

    let tail = NSMutableAttributedString::initWithString(
        NSMutableAttributedString::alloc(),
        &NSString::from_str(&format!("\t{desc}")),
    );
    let tail_len = tail.length();
    unsafe {
        tail.addAttribute_value_range(
            &font_key,
            as_any_object(&*desc_font),
            NSRange::new(0, tail_len),
        );
        tail.addAttribute_value_range(
            &color_key,
            as_any_object(&*desc_color),
            NSRange::new(0, tail_len),
        );
    }
    out.appendAttributedString(&tail);
    // A single tab stop lines every description up in the same column.
    let style = NSMutableParagraphStyle::new();
    style.setDefaultTabInterval(74.0);
    let style_key = NSString::from_str("NSParagraphStyle");
    let style_ref: &NSParagraphStyle = &style;
    let total = out.length();
    unsafe {
        out.addAttribute_value_range(&style_key, as_any_object(style_ref), NSRange::new(0, total));
    }
    out
}

/// An iMessage-style three-dot typing indicator: three dots with a bright
/// "head" that walks across them.
fn typing_dots(mtm: MainThreadMarker, time: f64) -> Retained<NSMutableAttributedString> {
    let _ = mtm;
    let font = NSFont::systemFontOfSize(7.0);
    let font_key = NSString::from_str("NSFont");
    let color_key = NSString::from_str("NSColor");
    let period = 1.1f64;
    let out = NSMutableAttributedString::initWithString(
        NSMutableAttributedString::alloc(),
        &NSString::from_str(""),
    );
    for i in 0..3 {
        if i > 0 {
            append_plain(&out, &font_key, " ", &font);
        }
        let p = (time / period - i as f64 * 0.16).rem_euclid(1.0);
        let s = 0.5 - 0.5 * (std::f64::consts::TAU * p).cos();
        let eased = s * s * (3.0 - 2.0 * s);
        let alpha = 0.16 + 0.84 * eased;
        let color = NSColor::labelColor().colorWithAlphaComponent(alpha);
        let a = NSMutableAttributedString::initWithString(
            NSMutableAttributedString::alloc(),
            &NSString::from_str("\u{25CF}"),
        );
        let len = a.length();
        unsafe {
            a.addAttribute_value_range(&font_key, as_any_object(&*font), NSRange::new(0, len));
            a.addAttribute_value_range(&color_key, as_any_object(&*color), NSRange::new(0, len));
        }
        out.appendAttributedString(&a);
    }
    out
}

#[allow(clippy::too_many_arguments)]
fn append_inline_attr(
    out: &NSMutableAttributedString,
    mtm: MainThreadMarker,
    line: &str,
    base_style: MdStyle,
    base_font: &Retained<NSFont>,
    bold_font: &Retained<NSFont>,
    italic_font: &Retained<NSFont>,
    code_font: &Retained<NSFont>,
    heading_font: &Retained<NSFont>,
    size: f64,
    color: &NSColor,
) {
    let font_key = NSString::from_str("NSFont");
    let mut buf = String::new();
    let (mut bold, mut italic, mut code) = (false, false, false);
    let mut i = 0;

    macro_rules! flush {
        () => {
            if !buf.is_empty() {
                let f: &Retained<NSFont> = if code {
                    code_font
                } else if bold {
                    bold_font
                } else if italic {
                    italic_font
                } else if base_style == MdStyle::Heading {
                    heading_font
                } else {
                    base_font
                };
                append_plain(out, &font_key, &buf, f);
                buf.clear();
            }
        };
    }

    while i < line.len() {
        let rest = &line[i..];
        if rest.starts_with("**") {
            flush!();
            bold = !bold;
            i += 2;
            continue;
        }
        if rest.starts_with('`') {
            flush!();
            code = !code;
            i += 1;
            continue;
        }
        if rest.starts_with('$') {
            flush!();
            let delim = if rest.starts_with("$$") { 2 } else { 1 };
            let after = &rest[delim..];
            let close = if delim == 2 {
                after.find("$$")
            } else {
                after.find('$')
            };
            let (inner, adv) = match close {
                Some(j) => (after[..j].to_string(), i + delim + j + delim),
                None => (after.to_string(), line.len()),
            };
            math_attributed(out, mtm, &inner, size, color);
            i = adv;
            continue;
        }
        if rest.starts_with('*') {
            flush!();
            italic = !italic;
            i += 1;
            continue;
        }
        let ch = rest.chars().next().unwrap();
        buf.push(ch);
        i += ch.len_utf8();
    }
    flush!();
}

#[derive(Clone, Copy, PartialEq)]
enum MKind {
    Normal,
    Super,
    Sub,
}

struct MRun {
    text: String,
    kind: MKind,
    upright: bool,
}

/// Look up a LaTeX command as a (glyph, upright) pair.
fn math_symbol(cmd: &str) -> Option<(&'static str, bool)> {
    Some(match cmd {
        "gamma" => ("\u{3b3}", false),
        "delta" => ("\u{3b4}", false),
        "epsilon" => ("\u{3b5}", false),
        "zeta" => ("\u{3b6}", false),
        "eta" => ("\u{3b7}", false),
        "theta" => ("\u{3b8}", false),
        "iota" => ("\u{3b9}", false),
        "kappa" => ("\u{3ba}", false),
        "lambda" => ("\u{3bb}", false),
        "mu" => ("\u{3bc}", false),
        "nu" => ("\u{3bd}", false),
        "xi" => ("\u{3be}", false),
        "pi" => ("\u{3c0}", false),
        "rho" => ("\u{3c1}", false),
        "sigma" => ("\u{3c3}", false),
        "tau" => ("\u{3c4}", false),
        "phi" => ("\u{3c6}", false),
        "chi" => ("\u{3c7}", false),
        "psi" => ("\u{3c8}", false),
        "omega" => ("\u{3c9}", false),
        "Gamma" => ("\u{393}", false),
        "Delta" => ("\u{394}", false),
        "Theta" => ("\u{398}", false),
        "Lambda" => ("\u{39b}", false),
        "Xi" => ("\u{39e}", false),
        "Pi" => ("\u{3a0}", false),
        "Sigma" => ("\u{3a3}", false),
        "Phi" => ("\u{3a6}", false),
        "Psi" => ("\u{3a8}", false),
        "Omega" => ("\u{3a9}", false),
        "times" => ("\u{d7}", true),
        "div" => ("\u{f7}", true),
        "cdot" => ("\u{b7}", true),
        "pm" => ("\u{b1}", true),
        "mp" => ("\u{2213}", true),
        "le" | "leq" => ("\u{2264}", true),
        "ge" | "geq" => ("\u{2265}", true),
        "ne" | "neq" => ("\u{2260}", true),
        "approx" => ("\u{2248}", true),
        "equiv" => ("\u{2261}", true),
        "propto" => ("\u{221d}", true),
        "infty" => ("\u{221e}", true),
        "sum" => ("\u{2211}", true),
        "prod" => ("\u{220f}", true),
        "int" => ("\u{222b}", true),
        "partial" => ("\u{2202}", true),
        "nabla" => ("\u{2207}", true),
        "to" | "rightarrow" => ("\u{2192}", true),
        "leftarrow" => ("\u{2190}", true),
        "leftrightarrow" => ("\u{2194}", true),
        "forall" => ("\u{2200}", true),
        "exists" => ("\u{2203}", true),
        "in" => ("\u{2208}", true),
        "notin" => ("\u{2209}", true),
        "subset" => ("\u{2282}", true),
        "cup" => ("\u{222a}", true),
        "cap" => ("\u{2229}", true),
        "cdots" | "dots" | "ldots" => ("\u{2026}", true),
        "quad" => ("  ", true),
        "qquad" => ("    ", true),
        "," | ";" | " " => (" ", true),
        "!" => ("", true),
        _ => return None,
    })
}

/// Upright multi-letter function names (\sin, \log, …).
fn math_func(cmd: &str) -> Option<&'static str> {
    Some(match cmd {
        "sin" => "sin",
        "cos" => "cos",
        "tan" => "tan",
        "cot" => "cot",
        "sec" => "sec",
        "csc" => "csc",
        "arcsin" => "arcsin",
        "arccos" => "arccos",
        "arctan" => "arctan",
        "sinh" => "sinh",
        "cosh" => "cosh",
        "tanh" => "tanh",
        "log" => "log",
        "ln" => "ln",
        "exp" => "exp",
        "max" => "max",
        "min" => "min",
        "lim" => "lim",
        "det" => "det",
        "gcd" => "gcd",
        _ => return None,
    })
}

fn math_parse(input: &str) -> Vec<MRun> {
    let chars: Vec<char> = input.chars().collect();
    let mut out = Vec::new();
    parse_seq(&chars, MKind::Normal, &mut out);
    out
}

fn parse_seq(chars: &[char], kind: MKind, out: &mut Vec<MRun>) {
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        match c {
            '{' => {
                let (g, ni) = read_atom(chars, i);
                parse_seq(&g.chars().collect::<Vec<_>>(), kind, out);
                i = ni;
            }
            '^' => {
                let (g, ni) = read_atom(chars, i + 1);
                parse_seq(&g.chars().collect::<Vec<_>>(), MKind::Super, out);
                i = ni;
            }
            '_' => {
                let (g, ni) = read_atom(chars, i + 1);
                parse_seq(&g.chars().collect::<Vec<_>>(), MKind::Sub, out);
                i = ni;
            }
            '\\' => {
                let (cmd, ni) = read_command(chars, i + 1);
                if cmd == "frac" {
                    let (a, n1) = read_atom(chars, ni);
                    let (b, n2) = read_atom(chars, n1);
                    let an = math_plain(&a);
                    let bn = math_plain(&b);
                    let wrap = |x: &str| {
                        if x.chars().count() > 1 {
                            format!("({x})")
                        } else {
                            x.to_string()
                        }
                    };
                    out.push(MRun {
                        text: wrap(&an),
                        kind,
                        upright: true,
                    });
                    out.push(MRun {
                        text: "/".into(),
                        kind,
                        upright: true,
                    });
                    out.push(MRun {
                        text: wrap(&bn),
                        kind,
                        upright: true,
                    });
                    i = n2;
                } else if cmd == "sqrt" {
                    let (a, n1) = read_atom(chars, ni);
                    let an = math_plain(&a);
                    let body = if an.chars().count() > 1 {
                        format!("({an})")
                    } else {
                        an
                    };
                    out.push(MRun {
                        text: "\u{221a}".into(),
                        kind,
                        upright: true,
                    });
                    out.push(MRun {
                        text: body,
                        kind,
                        upright: false,
                    });
                    i = n1;
                } else if let Some((sym, upright)) = math_symbol(&cmd) {
                    out.push(MRun {
                        text: sym.into(),
                        kind,
                        upright,
                    });
                    i = ni;
                } else if let Some(f) = math_func(&cmd) {
                    out.push(MRun {
                        text: f.into(),
                        kind,
                        upright: true,
                    });
                    i = ni;
                } else {
                    out.push(MRun {
                        text: cmd,
                        kind,
                        upright: false,
                    });
                    i = ni;
                }
            }
            _ => {
                out.push(MRun {
                    text: c.to_string(),
                    kind,
                    upright: !c.is_alphabetic(),
                });
                i += 1;
            }
        }
    }
}

/// Render an atom (group, command, or single char) to plain text.
fn math_plain(input: &str) -> String {
    math_parse(input).into_iter().map(|r| r.text).collect()
}

/// Read an atom starting at `i`: a `{…}` group, a `\command`, or one char.
fn read_atom(chars: &[char], i: usize) -> (String, usize) {
    if i >= chars.len() {
        return (String::new(), i);
    }
    if chars[i] == '{' {
        let mut depth = 0;
        let mut j = i;
        let mut s = String::new();
        while j < chars.len() {
            match chars[j] {
                '{' => {
                    depth += 1;
                    if depth > 1 {
                        s.push('{');
                    }
                }
                '}' => {
                    depth -= 1;
                    if depth == 0 {
                        return (s, j + 1);
                    }
                    s.push('}');
                }
                other => s.push(other),
            }
            j += 1;
        }
        (s, j)
    } else if chars[i] == '\\' {
        let (cmd, ni) = read_command(chars, i + 1);
        (format!("\\{cmd}"), ni)
    } else {
        (chars[i].to_string(), i + 1)
    }
}

/// Read a `\command` name (letters, or one symbol char).
fn read_command(chars: &[char], i: usize) -> (String, usize) {
    let mut j = i;
    let mut s = String::new();
    while j < chars.len() && chars[j].is_ascii_alphabetic() {
        s.push(chars[j]);
        j += 1;
    }
    if s.is_empty() && j < chars.len() {
        s.push(chars[j]);
        j += 1;
    }
    (s, j)
}

/// Typeset a math expression into baseline-shifted attributed runs.
fn math_attributed(
    out: &NSMutableAttributedString,
    mtm: MainThreadMarker,
    input: &str,
    size: f64,
    color: &NSColor,
) {
    let small = size * 0.72;
    let italic = NSFont::fontWithName_size(
        &NSString::from_str("TimesNewRomanPS-ItalicMT"),
        size,
    )
    .unwrap_or_else(|| {
        NSFontManager::sharedFontManager(mtm).convertFont_toHaveTrait(
            &NSFont::systemFontOfSize(size),
            NSFontTraitMask::ItalicFontMask,
        )
    });
    let italic_small = NSFont::fontWithName_size(
        &NSString::from_str("TimesNewRomanPS-ItalicMT"),
        small,
    )
    .unwrap_or_else(|| {
        NSFontManager::sharedFontManager(mtm).convertFont_toHaveTrait(
            &NSFont::systemFontOfSize(small),
            NSFontTraitMask::ItalicFontMask,
        )
    });
    let upright = NSFont::fontWithName_size(&NSString::from_str("TimesNewRomanPSMT"), size)
        .unwrap_or_else(|| NSFont::systemFontOfSize(size));
    let upright_small = NSFont::fontWithName_size(&NSString::from_str("TimesNewRomanPSMT"), small)
        .unwrap_or_else(|| NSFont::systemFontOfSize(small));

    let font_key = NSString::from_str("NSFont");
    let color_key = NSString::from_str("NSColor");
    let base_key = NSString::from_str("NSBaselineOffset");

    for run in math_parse(input) {
        if run.text.is_empty() {
            continue;
        }
        let (font, offset) = match run.kind {
            MKind::Normal => {
                let f = if run.upright { &upright } else { &italic };
                (f, 0.0)
            }
            MKind::Super => {
                let f = if run.upright {
                    &upright_small
                } else {
                    &italic_small
                };
                (f, size * 0.42)
            }
            MKind::Sub => {
                let f = if run.upright {
                    &upright_small
                } else {
                    &italic_small
                };
                (f, -size * 0.16)
            }
        };
        let a = NSMutableAttributedString::initWithString(
            NSMutableAttributedString::alloc(),
            &NSString::from_str(&run.text),
        );
        let len = a.length();
        unsafe {
            a.addAttribute_value_range(&font_key, as_any_object(&**font), NSRange::new(0, len));
            a.addAttribute_value_range(&color_key, as_any_object(&*color), NSRange::new(0, len));
            if offset != 0.0 {
                let n = NSNumber::new_f64(offset);
                let v: &NSValue = ClassType::as_super(&*n);
                let o: &NSObject = ClassType::as_super(v);
                let any: &AnyObject = ClassType::as_super(o);
                a.addAttribute_value_range(&base_key, any, NSRange::new(0, len));
            }
        }
        out.appendAttributedString(&a);
    }
}
/// A parsed markdown block.
enum Block {
    Heading(String),
    Paragraph(String),
    Bullets(Vec<String>),
    Rule,
    Table(Vec<Vec<String>>, bool),
    Code(String),
}

fn parse_cells(line: &str) -> Vec<String> {
    let t = line.trim();
    let t = t.strip_prefix('|').unwrap_or(t);
    let t = t.strip_suffix('|').unwrap_or(t);
    t.split('|').map(|c| sanitize_cell(c.trim())).collect()
}

/// Drop math delimiters and inline markers from a table cell.
fn sanitize_cell(c: &str) -> String {
    c.replace("**", "").chars().filter(|ch| *ch != '`').collect()
}

/// True for a `|---|---|` alignment-specifier row.
fn is_sep_row(row: &[String]) -> bool {
    !row.is_empty()
        && row
            .iter()
            .all(|c| !c.is_empty() && c.chars().all(|ch| ch == '-' || ch == ':'))
}

fn is_table_line(line: &str) -> bool {
    let t = line.trim_start();
    t.starts_with('|') && t.matches('|').count() >= 2
}

fn is_heading_line(line: &str) -> bool {
    let t = line.trim_start();
    let h = t.chars().take_while(|c| *c == '#').count();
    h >= 1 && h <= 6 && t[h..].starts_with(' ')
}

fn is_bullet_line(line: &str) -> bool {
    let t = line.trim_start();
    t.starts_with("- ") || t.starts_with("* ")
}

/// Split a reply into block-level markdown units.
fn parse_blocks(text: &str) -> Vec<Block> {
    let lines: Vec<&str> = text.split('\n').collect();
    let mut blocks = Vec::new();
    let mut i = 0;
    while i < lines.len() {
        let t = lines[i].trim();
        if t.is_empty() {
            i += 1;
            continue;
        }
        if t.starts_with("```") {
            let mut body = Vec::new();
            i += 1;
            while i < lines.len() && !lines[i].trim_start().starts_with("```") {
                body.push(lines[i]);
                i += 1;
            }
            if i < lines.len() {
                i += 1;
            }
            blocks.push(Block::Code(body.join("\n")));
            continue;
        }
        if is_table_line(lines[i]) {
            let mut rows: Vec<Vec<String>> = Vec::new();
            let mut header = false;
            while i < lines.len() && is_table_line(lines[i]) {
                let cells = parse_cells(lines[i]);
                if is_sep_row(&cells) {
                    if rows.len() == 1 {
                        header = true;
                    }
                } else {
                    rows.push(cells);
                }
                i += 1;
            }
            if !rows.is_empty() {
                blocks.push(Block::Table(rows, header));
            }
            continue;
        }
        if t == "---" || t == "***" || t == "___" {
            blocks.push(Block::Rule);
            i += 1;
            continue;
        }
        if is_heading_line(lines[i]) {
            let s = lines[i].trim_start();
            let h = s.chars().take_while(|c| *c == '#').count();
            blocks.push(Block::Heading(s[h + 1..].trim().to_string()));
            i += 1;
            continue;
        }
        if is_bullet_line(lines[i]) {
            let mut items = Vec::new();
            while i < lines.len() && is_bullet_line(lines[i]) {
                let s = lines[i].trim_start();
                let rest = s.strip_prefix("- ").or_else(|| s.strip_prefix("* ")).unwrap();
                items.push(rest.to_string());
                i += 1;
            }
            blocks.push(Block::Bullets(items));
            continue;
        }
        let mut para = Vec::new();
        while i < lines.len() {
            let t = lines[i].trim();
            if t.is_empty()
                || t.starts_with("```")
                || is_table_line(lines[i])
                || t == "---"
                || t == "***"
                || t == "___"
                || is_heading_line(lines[i])
                || is_bullet_line(lines[i])
            {
                break;
            }
            para.push(lines[i]);
            i += 1;
        }
        blocks.push(Block::Paragraph(para.join("\n")));
    }
    blocks
}

/// Render every block of a reply into `stack`.
fn render_blocks(mtm: MainThreadMarker, stack: &NSStackView, text: &str, color: &NSColor) {
    for block in parse_blocks(text) {
        match block {
            Block::Heading(t) => {
                let l = paragraph_label(mtm, &t, color, MdStyle::Heading);
                stack.addArrangedSubview(&l);
            }
            Block::Paragraph(t) => {
                let l = paragraph_label(mtm, &t, color, MdStyle::Base);
                stack.addArrangedSubview(&l);
            }
            Block::Bullets(items) => {
                let v = bullets_view(mtm, &items, color);
                stack.addArrangedSubview(&v);
            }
            Block::Rule => {
                let v = LineView::new(mtm);
                ui::height(&v, 1.0);
                stack.addArrangedSubview(&v);
                v.widthAnchor()
                    .constraintEqualToAnchor(&stack.widthAnchor())
                    .setActive(true);
            }
            Block::Table(rows, header) => {
                if let Some(prev) = stack.arrangedSubviews().lastObject() {
                    stack.setCustomSpacing_afterView(TABLE_GAP, &prev);
                }
                let grid = table_view(mtm, &rows, header, color);
                stack.addArrangedSubview(&grid);
                grid.widthAnchor()
                    .constraintGreaterThanOrEqualToAnchor(&stack.widthAnchor())
                    .setActive(true);
                stack.setCustomSpacing_afterView(TABLE_GAP, &grid);
            }
            Block::Code(t) => {
                let l = code_label(mtm, &t, color);
                stack.addArrangedSubview(&l);
            }
        }
    }
}

fn paragraph_label(
    mtm: MainThreadMarker,
    text: &str,
    color: &NSColor,
    base_style: MdStyle,
) -> Retained<NSTextField> {
    let label = NSTextField::wrappingLabelWithString(&NSString::from_str(""), mtm);
    label.setTranslatesAutoresizingMaskIntoConstraints(false);
    label.setMaximumNumberOfLines(0);
    label.setPreferredMaxLayoutWidth(BUBBLE_MAX_WIDTH - 30.0);
    label.setAlignment(NSTextAlignment::Left);
    label.setFont(Some(&NSFont::systemFontOfSize(BUBBLE_FONT_SIZE)));
    label.setAttributedStringValue(&styled_attributed(mtm, text, color, base_style, BUBBLE_FONT_SIZE));
    ui::make_static(&label);
    label
}

fn code_label(mtm: MainThreadMarker, text: &str, color: &NSColor) -> Retained<NSTextField> {
    let label = NSTextField::wrappingLabelWithString(&NSString::from_str(text), mtm);
    if let Some(font) = NSFont::userFixedPitchFontOfSize(BUBBLE_FONT_SIZE) {
        label.setFont(Some(&font));
    }
    label.setTextColor(Some(color));
    label.setMaximumNumberOfLines(0);
    label.setPreferredMaxLayoutWidth(BUBBLE_MAX_WIDTH - 30.0);
    label.setAlignment(NSTextAlignment::Left);
    label.setTranslatesAutoresizingMaskIntoConstraints(false);
    ui::make_static(&label);
    label
}

fn bullets_view(
    mtm: MainThreadMarker,
    items: &[String],
    color: &NSColor,
) -> Retained<NSStackView> {
    let stack = NSStackView::new(mtm);
    stack.setOrientation(NSUserInterfaceLayoutOrientation::Vertical);
    stack.setAlignment(NSLayoutAttribute::Leading);
    stack.setSpacing(6.0);
    stack.setTranslatesAutoresizingMaskIntoConstraints(false);
    for item in items {
        let label = paragraph_label(mtm, &format!("\u{2022}  {item}"), color, MdStyle::Base);
        stack.addArrangedSubview(&label);
    }
    stack
}

/// Draw a full-width horizontal rule.
#[derive(Debug)]
pub struct LineViewIvars;

define_class!(
    #[unsafe(super(NSView))]
    #[thread_kind = MainThreadOnly]
    #[ivars = LineViewIvars]
    #[derive(Debug)]
    pub struct LineView;

    unsafe impl NSObjectProtocol for LineView {}

    impl LineView {
        #[unsafe(method(drawRect:))]
        fn draw_rect(&self, _dirty_rect: NSRect) {
            NSColor::separatorColor().setFill();
            NSBezierPath::bezierPathWithRect(self.bounds()).fill();
        }
    }
);

impl LineView {
    fn new(mtm: MainThreadMarker) -> Retained<Self> {
        let this = Self::alloc(mtm).set_ivars(LineViewIvars);
        unsafe { msg_send![super(this), init] }
    }
}

/// A table cell that paints its own background and a bottom hairline.
#[derive(Debug)]
pub struct TableCellIvars {
    fill: OnceCell<Retained<NSColor>>,
    bottom: Cell<bool>,
}

define_class!(
    #[unsafe(super(NSView))]
    #[thread_kind = MainThreadOnly]
    #[ivars = TableCellIvars]
    #[derive(Debug)]
    pub struct TableCellView;

    unsafe impl NSObjectProtocol for TableCellView {}

    impl TableCellView {
        #[unsafe(method(drawRect:))]
        fn draw_rect(&self, _dirty_rect: NSRect) {
            if let Some(fill) = self.ivars().fill.get() {
                fill.setFill();
                NSBezierPath::bezierPathWithRect(self.bounds()).fill();
            }
            if self.ivars().bottom.get() {
                NSColor::separatorColor().setFill();
                let b = self.bounds();
                let line = NSRect::new(
                    NSPoint::new(0.0, 0.0),
                    NSSize::new(b.size.width, 1.0),
                );
                NSBezierPath::bezierPathWithRect(line).fill();
            }
        }
    }
);

impl TableCellView {
    fn new(mtm: MainThreadMarker, _header: bool, bottom: bool) -> Retained<Self> {
        let this = Self::alloc(mtm).set_ivars(TableCellIvars {
            fill: OnceCell::new(),
            bottom: Cell::new(false),
        });
        let this: Retained<Self> = unsafe { msg_send![super(this), init] };
        let fill = NSColor::clearColor();
        this.ivars().fill.set(fill).ok();
        this.ivars().bottom.set(bottom);
        this
    }
}

fn table_cell(
    mtm: MainThreadMarker,
    text: &str,
    header: bool,
    color: &NSColor,
    bottom: bool,
) -> Retained<TableCellView> {
    let cell = TableCellView::new(mtm, header, bottom);
    cell.setTranslatesAutoresizingMaskIntoConstraints(false);
    let label = NSTextField::labelWithString(&NSString::from_str(""), mtm);
    if header {
        label.setStringValue(&NSString::from_str(text));
        label.setFont(Some(&NSFont::boldSystemFontOfSize(BUBBLE_FONT_SIZE)));
        label.setTextColor(Some(color));
    } else {
        let attr = styled_attributed(mtm, text, color, MdStyle::Base, BUBBLE_FONT_SIZE);
        label.setAttributedStringValue(&attr);
    }
    label.setAlignment(NSTextAlignment::Left);
    label.setTranslatesAutoresizingMaskIntoConstraints(false);
    ui::make_static(&label);
    cell.addSubview(&label);
    label
        .leadingAnchor()
        .constraintEqualToAnchor_constant(&cell.leadingAnchor(), 12.0)
        .setActive(true);
    label
        .trailingAnchor()
        .constraintEqualToAnchor_constant(&cell.trailingAnchor(), -12.0)
        .setActive(true);
    label
        .topAnchor()
        .constraintEqualToAnchor_constant(&cell.topAnchor(), 6.0)
        .setActive(true);
    label
        .bottomAnchor()
        .constraintEqualToAnchor_constant(&cell.bottomAnchor(), -6.0)
        .setActive(true);
    cell
}

/// Render a pipe table as a real `NSGridView` with padded cells, a tinted
/// header row and hairline row separators.
fn table_view(
    mtm: MainThreadMarker,
    rows: &[Vec<String>],
    has_header: bool,
    color: &NSColor,
) -> Retained<NSGridView> {
    let ncols = rows.iter().map(|r| r.len()).max().unwrap_or(0);
    let mut row_arrays: Vec<Retained<NSArray<NSView>>> = Vec::new();
    for (ri, row) in rows.iter().enumerate() {
        let mut cells: Vec<Retained<NSView>> = Vec::new();
        for c in 0..ncols {
            let text = row.get(c).cloned().unwrap_or_default();
            let header = has_header && ri == 0;
            let bottom = ri + 1 < rows.len();
            let cell = table_cell(mtm, &text, header, color, bottom);
            cells.push(cell.into_super());
        }
        row_arrays.push(NSArray::from_retained_slice(&cells));
    }
    let rows_arr = NSArray::from_retained_slice(&row_arrays);
    let grid = NSGridView::gridViewWithViews(&rows_arr, mtm);
    grid.setRowSpacing(0.0);
    grid.setColumnSpacing(0.0);
    grid.setXPlacement(NSGridCellPlacement::Fill);
    grid.setYPlacement(NSGridCellPlacement::Fill);
    grid.setRowAlignment(NSGridRowAlignment::None);
    for i in 0..grid.numberOfRows() {
        let row = grid.rowAtIndex(i);
        row.setTopPadding(0.0);
        row.setBottomPadding(0.0);
        row.setHeight(TABLE_ROW_HEIGHT);
    }
    grid.setTranslatesAutoresizingMaskIntoConstraints(false);
    grid.widthAnchor()
        .constraintLessThanOrEqualToConstant(BUBBLE_MAX_WIDTH - 30.0)
        .setActive(true);
    grid
}
/// Clean a raw model reply for display: drop `<|…|>` special-token markers and
/// any ` thinking…</think>` reasoning, leaving just the answer. The prompt
/// pre-fills ` thinking`, so the reply often has only the closing tag.
fn clean_reply(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut rest = s;
    while let Some(i) = rest.find("<|") {
        out.push_str(&rest[..i]);
        match rest[i..].find("|>") {
            Some(j) => rest = &rest[i + j + 2..],
            None => rest = "",
        }
    }
    out.push_str(rest);
    for (open, close) in [(" thinking", "</think>"), ("<think", "</think>")] {
        while let Some(cpos) = out.find(close) {
            let start = out[..cpos].rfind(open).unwrap_or(0);
            out.replace_range(start..cpos + close.len(), "");
        }
        if let Some(opos) = out.find(open) {
            out.truncate(opos);
        }
    }
    out.trim().to_string()
}

const CHAT_MAX_TOKENS: usize = 2048;
const BUBBLE_MAX_WIDTH: f64 = 640.0;
const BUBBLE_FONT_SIZE: f64 = 13.0;
/// Poll interval; also the typing-indicator frame step.
const TICK_DT: f64 = 0.03;
const TABLE_GAP: f64 = 18.0;
const DEFAULT_SYSTEM_PROMPT: &str =
    "You are Lisa, a helpful and knowledgeable AI assistant. Answer clearly and concisely.";
const TABLE_ROW_HEIGHT: f64 = 34.0;

// A top-left origin document view so the transcript scrolls from the top.
define_class!(
    #[unsafe(super(NSView))]
    #[thread_kind = MainThreadOnly]
    #[derive(Debug)]
    pub struct FlippedView;

    unsafe impl NSObjectProtocol for FlippedView {}

    impl FlippedView {
        #[unsafe(method(isFlipped))]
        fn is_flipped(&self) -> bool {
            true
        }
    }
);

impl FlippedView {
    fn new(mtm: MainThreadMarker) -> Retained<Self> {
        let this = Self::alloc(mtm);
        unsafe { msg_send![this, init] }
    }
}

/// A rounded message bubble that paints its own background.
#[derive(Debug)]
pub struct BubbleViewIvars {
    fill: OnceCell<Retained<NSColor>>,
}

define_class!(
    #[unsafe(super(NSView))]
    #[thread_kind = MainThreadOnly]
    #[ivars = BubbleViewIvars]
    #[derive(Debug)]
    pub struct BubbleView;

    unsafe impl NSObjectProtocol for BubbleView {}

    impl BubbleView {
        #[unsafe(method(drawRect:))]
        fn draw_rect(&self, _dirty_rect: NSRect) {
            let Some(fill) = self.ivars().fill.get() else {
                return;
            };
            let path = NSBezierPath::bezierPathWithRoundedRect_xRadius_yRadius(
                self.bounds(),
                12.0,
                12.0,
            );
            fill.setFill();
            path.fill();
        }
    }
);

impl BubbleView {
    fn new(mtm: MainThreadMarker, fill: &Retained<NSColor>) -> Retained<Self> {
        let this = Self::alloc(mtm).set_ivars(BubbleViewIvars {
            fill: OnceCell::new(),
        });
        let this: Retained<Self> = unsafe { msg_send![super(this), init] };
        this.ivars().fill.set(fill.clone()).ok();
        this
    }
}

/// A rounded, self-painting view used for icon badges and settings cards.
#[derive(Debug)]
pub struct RoundedViewIvars {
    fill: OnceCell<Retained<NSColor>>,
    radius: Cell<f64>,
}

define_class!(
    #[unsafe(super(NSView))]
    #[thread_kind = MainThreadOnly]
    #[ivars = RoundedViewIvars]
    #[derive(Debug)]
    pub struct RoundedView;

    unsafe impl NSObjectProtocol for RoundedView {}

    impl RoundedView {
        #[unsafe(method(drawRect:))]
        fn draw_rect(&self, _dirty_rect: NSRect) {
            let Some(fill) = self.ivars().fill.get() else {
                return;
            };
            fill.setFill();
            let r = self.ivars().radius.get();
            NSBezierPath::bezierPathWithRoundedRect_xRadius_yRadius(self.bounds(), r, r).fill();
        }
    }
);

impl RoundedView {
    fn new(mtm: MainThreadMarker, fill: &Retained<NSColor>, radius: f64) -> Retained<Self> {
        let this = Self::alloc(mtm).set_ivars(RoundedViewIvars {
            fill: OnceCell::new(),
            radius: Cell::new(radius),
        });
        let this: Retained<Self> = unsafe { msg_send![super(this), init] };
        this.ivars().fill.set(fill.clone()).ok();
        this
    }
}

/// A determinate progress ring (track + accent arc) used for downloads.
#[derive(Debug)]
pub struct ProgressRingIvars {
    progress: Cell<f64>,
}

define_class!(
    #[unsafe(super(NSView))]
    #[thread_kind = MainThreadOnly]
    #[ivars = ProgressRingIvars]
    #[derive(Debug)]
    pub struct ProgressRing;

    unsafe impl NSObjectProtocol for ProgressRing {}

    impl ProgressRing {
        #[unsafe(method(drawRect:))]
        fn draw_rect(&self, _dirty_rect: NSRect) {
            let b = self.bounds();
            let inset = 2.0;
            let w = b.size.width - inset * 2.0;
            let h = b.size.height - inset * 2.0;
            let rect = NSRect::new(
                NSPoint::new(b.origin.x + inset, b.origin.y + inset),
                NSSize::new(w, h),
            );
            let track = NSBezierPath::bezierPathWithOvalInRect(rect);
            track.setLineWidth(2.0);
            NSColor::quaternaryLabelColor().setStroke();
            track.stroke();

            let p = self.ivars().progress.get().clamp(0.0, 1.0);
            if p > 0.0 {
                let center = NSPoint::new(
                    b.origin.x + b.size.width / 2.0,
                    b.origin.y + b.size.height / 2.0,
                );
                let radius = w.min(h) / 2.0;
                let arc = NSBezierPath::bezierPath();
                arc.appendBezierPathWithArcWithCenter_radius_startAngle_endAngle_clockwise(
                    center,
                    radius,
                    90.0,
                    90.0 - 360.0 * p,
                    true,
                );
                arc.setLineWidth(2.0);
                arc.setLineCapStyle(NSLineCapStyle::Round);
                NSColor::controlAccentColor().setStroke();
                arc.stroke();
            }
        }
    }
);

impl ProgressRing {
    fn new(mtm: MainThreadMarker, progress: f64) -> Retained<Self> {
        let this = Self::alloc(mtm).set_ivars(ProgressRingIvars {
            progress: Cell::new(progress),
        });
        let this: Retained<Self> = unsafe { msg_send![super(this), init] };
        this
    }
}

/// Shorten a path under the user's home directory to `~/…`.
fn tilde(path: &str) -> String {
    if let Ok(home) = std::env::var("HOME") {
        if let Some(rest) = path.strip_prefix(&home) {
            return format!("~{rest}");
        }
    }
    path.to_string()
}

/// Owns the loaded model on a dedicated thread (the model is not `Send`) and
/// funnels log/chat events back to the main thread through `events`.
fn spawn_engine(rx: mpsc::Receiver<Command>, events: Arc<Mutex<Vec<UiEvent>>>) {
    let _ = thread::Builder::new().name("lisa-engine".into()).spawn(move || {
        let push = |e: UiEvent| events.lock().unwrap().push(e);
        let mut model: Option<Box<dyn lisa_engine::models::LanguageModel>> = None;
        let mut tokenizer: Option<lisa_engine::core::tokenizer::Tokenizer> = None;
        let mut sampler = lisa_engine::core::sampler::Sampler::default();
        let mut model_dir: Option<PathBuf> = None;
        // Multi-turn conversation state; kept in lockstep with the loaded model.
        let mut session: Option<lisa_engine::core::session::Session> = None;
        let mut turn = 0usize;
        // Whether the checkpoint's template closes assistant turns with
        // `` (MiMo-style) rather than Qwen's bare newline.
        let mut assistant_im_end = false;

        // One engine thread serves both UI chat and (optionally) the
        // in-process HTTP server. Commands are preferred; between them the
        // server advances one wave at a time. The model is not `Send` and a
        // forward pass is not preemptible, so requests are serialized — a chat
        // turn waits at most for an in-flight HTTP request, and vice versa.
        let mut server: Option<lisa_serve::Server> = None;
        let mut serve_stop: Option<Arc<AtomicBool>> = None;
        loop {
            let cmd = match rx.try_recv() {
                Ok(c) => c,
                Err(mpsc::TryRecvError::Disconnected) => break,
                Err(mpsc::TryRecvError::Empty) => {
                    if let Some(stop) = serve_stop.as_ref() {
                        if stop.load(Ordering::Relaxed) {
                            if let Some(mut srv) = server.take() {
                                srv.shutdown();
                            }
                            serve_stop = None;
                            push(UiEvent::ServeStopped);
                            continue;
                        }
                    }
                    if let (Some(srv), Some(m), Some(t)) =
                        (server.as_mut(), model.as_mut(), tokenizer.as_ref())
                    {
                        if srv.step(&mut **m, t) {
                            continue;
                        }
                    }
                    match rx.recv_timeout(std::time::Duration::from_millis(50)) {
                        Ok(c) => c,
                        Err(mpsc::RecvTimeoutError::Timeout) => continue,
                        Err(mpsc::RecvTimeoutError::Disconnected) => break,
                    }
                }
            };
            match cmd {
                Command::Load(path) => {
                    // Swapping the model requires stopping any running server.
                    if let Some(mut srv) = server.take() {
                        srv.shutdown();
                        serve_stop = None;
                        push(UiEvent::ServeStopped);
                    }
                    model = None;
                    tokenizer = None;
                    push(UiEvent::Log(format!(
                        "loading {} …",
                        tilde(&path.to_string_lossy())
                    )));
                    match lisa_engine::models::load_dir(&path) {
                        Ok(lisa_engine::models::Loaded::Language(m)) => {
                            match lisa_engine::core::tokenizer::Tokenizer::load(&path) {
                                Ok(t) => {
                                    lisa_engine::core::generate::set_eos_ids(t.im_end_ids.clone());
                                    tokenizer = Some(t);
                                    model = Some(m);
                                    model_dir = Some(path.clone());
                                    assistant_im_end = std::fs::read_to_string(
                                        path.join("chat_template.jinja"),
                                    )
                                    .map(|t| t.contains("render_assistant_message"))
                                    .unwrap_or(false);
                                    if let Some(m) = model.as_mut() {
                                        session = Some(lisa_engine::core::session::Session::new(
                                            &mut **m,
                                        ));
                                        turn = 0;
                                    }
                                    push(UiEvent::Log("model loaded".into()));
                                    push(UiEvent::Loaded(true));
                                }
                                Err(e) => {
                                    push(UiEvent::Log(format!("tokenizer error: {e}")));
                                    push(UiEvent::Loaded(false));
                                }
                            }
                        }
                        Ok(lisa_engine::models::Loaded::Decision(_)) => {
                            push(UiEvent::Log("decision models are not chattable".into()));
                            push(UiEvent::Loaded(false));
                        }
                        Err(e) => {
                            push(UiEvent::Log(format!("load error: {e}")));
                            push(UiEvent::Loaded(false));
                        }
                    }
                }
                Command::Download(repo) => {
                    // Download on a side thread so the active model keeps
                    // serving chat, and don't swap models when it finishes.
                    push(UiEvent::Log(format!("downloading {repo} …")));
                    let events = events.clone();
                    let repo2 = repo.clone();
                    let _ = thread::Builder::new()
                        .name("lisa-download".into())
                        .spawn(move || {
                            let push = |e: UiEvent| events.lock().unwrap().push(e);
                            match lisa_engine::models::resolve_model_dir(&repo2) {
                                Ok(_) => {
                                    push(UiEvent::Log(format!("downloaded {repo2}")));
                                    push(UiEvent::ModelsChanged);
                                }
                                Err(e) => {
                                    push(UiEvent::Log(format!("download error: {e}")));
                                    push(UiEvent::ModelsChanged);
                                }
                            }
                        });
                }
                Command::Clear => {
                    if let Some(m) = model.as_mut() {
                        session = Some(lisa_engine::core::session::Session::new(&mut **m));
                        turn = 0;
                    }
                    push(UiEvent::Log("conversation cleared".into()));
                }
                Command::Chat { text, system } => {
                    let (Some(model), Some(tokenizer)) = (model.as_mut(), tokenizer.as_ref())
                    else {
                        push(UiEvent::Log("no model loaded".into()));
                        continue;
                    };
                    if session.is_none() {
                        session =
                            Some(lisa_engine::core::session::Session::new(&mut **model));
                        turn = 0;
                    }
                    let sys = if system.trim().is_empty() {
                        None
                    } else {
                        Some(system.as_str())
                    };
                    // First turn renders the whole (system + user) prompt; later
                    // turns append just the new user turn to the session.
                    let prompt = if turn == 0 {
                        lisa_engine::core::tokenizer::chat_prompt_special_with_system(
                            &text, sys, false,
                        )
                    } else {
                        lisa_engine::core::tokenizer::chat_turn_suffix_special(
                            &text,
                            false,
                            false,
                            assistant_im_end,
                        )
                    };
                    let ids = match tokenizer.encode(&prompt, false) {
                        Ok(ids) => ids,
                        Err(e) => {
                            push(UiEvent::Log(format!("encode error: {e}")));
                            continue;
                        }
                    };
                    push(UiEvent::UserMessage(text.clone()));
                    let mut all = Vec::new();
                    let mut emitted = String::new();
                    let sess = session.as_mut().expect("session");
                    let result = sess.generate(
                        &mut **model,
                        &ids,
                        CHAT_MAX_TOKENS,
                        &mut sampler,
                        &mut |id| {
                            all.push(id);
                            let full = tokenizer.decode_clean(&all).unwrap_or_default();
                            let display = clean_reply(&full);
                            if display.len() >= emitted.len() && display.starts_with(&emitted) {
                                let delta = display[emitted.len()..].to_string();
                                if !delta.is_empty() && !delta.ends_with('\u{FFFD}') {
                                    emitted = display;
                                    push(UiEvent::AssistantDelta(delta));
                                }
                            } else if display.len() > emitted.len() {
                                // Non-monotonic (a stripped think/special span):
                                // only grow the bubble, never clear it mid-stream.
                                emitted = display.clone();
                                push(UiEvent::AssistantReset(display));
                            }
                            Ok(())
                        },
                    );
                    if let Err(e) = result {
                        push(UiEvent::AssistantDelta(format!("\n[error: {e}]")));
                    } else {
                        // Settle on the authoritative final text (the streaming
                        // pass may have skipped transient truncations).
                        let full = tokenizer.decode_clean(&all).unwrap_or_default();
                        let display = clean_reply(&full);
                        if display != emitted {
                            emitted = display.clone();
                            push(UiEvent::AssistantReset(display));
                        }
                        if emitted.is_empty() {
                            push(UiEvent::AssistantDelta("(no response)".to_string()));
                        }
                    }
                    turn += 1;
                    push(UiEvent::AssistantDone);
                }
                Command::StartServe { addr, stop } => {
                    if model.is_none() || tokenizer.is_none() {
                        push(UiEvent::Log("serve: no model loaded".into()));
                        push(UiEvent::ServeStopped);
                        continue;
                    }
                    let chat_template = model_dir
                        .as_ref()
                        .and_then(|d| std::fs::read_to_string(d.join("chat_template.jinja")).ok());
                    let cfg = lisa_serve::ServerConfig {
                        addr: addr.clone(),
                        max_tokens: 2048,
                        temperature: 0.0,
                        top_p: 1.0,
                        top_k: 0,
                        min_p: 0.0,
                        rep_penalty: 1.0,
                        depth: 0,
                        max_batch: 4,
                        chat_template,
                    };
                    match lisa_serve::Server::start(cfg, stop.clone()) {
                        Ok(srv) => {
                            let local = srv.local_addr();
                            server = Some(srv);
                            serve_stop = Some(stop);
                            push(UiEvent::ServeStarted(format!("http://{local}")));
                        }
                        Err(e) => {
                            push(UiEvent::Log(format!("serve error: {e}")));
                            push(UiEvent::ServeStopped);
                        }
                    }
                }
            }
        }
    });
}

#[derive(Debug)]
pub struct AppDelegateIvars {
    window: OnceCell<Retained<NSWindow>>,
    sidebar_item: OnceCell<Retained<NSSplitViewItem>>,
    logs_item: OnceCell<Retained<NSSplitViewItem>>,
    models: RefCell<Vec<Model>>,
    remote_ids: RefCell<Vec<(String, u64)>>,
    models_table: OnceCell<Retained<NSTableView>>,
    loading_row: RefCell<Option<usize>>,
    /// Repo id of the model currently loaded/active. Tracked by id (not row
    /// index) so list re-sorting never moves the active highlight.
    active_repo: RefCell<Option<String>>,
    downloading: Cell<bool>,
    download_progress: Cell<f64>,
    progress_tick: Cell<usize>,
    selected_row: RefCell<Option<usize>>,
    updating_table: Cell<bool>,
    chat_stack: OnceCell<Retained<NSStackView>>,
    chat_scroll: OnceCell<Retained<NSScrollView>>,
    chat_placeholder: OnceCell<Retained<NSTextField>>,
    assistant_label: RefCell<Option<Retained<NSTextField>>>,
    assistant_stack: RefCell<Option<Retained<NSStackView>>>,
    typing: Cell<bool>,
    typing_tick: Cell<usize>,
    chat_input: OnceCell<Retained<NSTextField>>,
    chat_send: OnceCell<Retained<NSButton>>,
    cmd_menu: OnceCell<Retained<RoundedView>>,
    cmd_rows: RefCell<Vec<Retained<NSButton>>>,
    cmd_bgs: RefCell<Vec<Retained<RoundedView>>>,
    cmd_sel: Cell<isize>,
    system_view: OnceCell<Retained<NSTextView>>,
    engine_log: OnceCell<Retained<NSTextView>>,
    engine_tx: OnceCell<mpsc::Sender<Command>>,
    engine_events: OnceCell<Arc<Mutex<Vec<UiEvent>>>>,
    serve_item: OnceCell<Retained<NSSplitViewItem>>,
    serve_port: OnceCell<Retained<NSTextField>>,
    serve_switch: OnceCell<Retained<NSSwitch>>,
    serve_status: OnceCell<Retained<NSTextField>>,
    serve_state: OnceCell<Retained<NSTextField>>,
    serve_dot: OnceCell<Retained<NSImageView>>,
    serve_stop: RefCell<Option<Arc<AtomicBool>>>,
    serving: Cell<bool>,
    /// Whether a model is currently loaded (chat is enabled once it is, and
    /// disabled while the in-process server is running).
    model_loaded: Cell<bool>,
}

define_class!(
    #[unsafe(super(NSObject))]
    #[thread_kind = MainThreadOnly]
    #[ivars = AppDelegateIvars]
    #[derive(Debug)]
    pub struct AppDelegate;

    unsafe impl NSObjectProtocol for AppDelegate {}

    unsafe impl NSControlTextEditingDelegate for AppDelegate {
        #[unsafe(method(controlTextDidChange:))]
        fn control_text_did_change(&self, _notification: &NSNotification) {
            self.update_command_menu();
        }

        #[unsafe(method(controlTextDidEndEditing:))]
        fn control_text_did_end_editing(&self, _notification: &NSNotification) {
            // Keep the menu open when the click landed on one of its rows
            // (the row's action hides it); drop it for any other focus loss.
            let Some(menu) = self.ivars().cmd_menu.get() else {
                return;
            };
            if menu.isHidden() {
                return;
            }
            let Some(window) = self.ivars().window.get() else {
                return;
            };
            let point = NSEvent::mouseLocation();
            let base = menu.convertRect_toView(menu.bounds(), None);
            let screen = window.convertRectToScreen(base);
            let inside = point.x >= screen.origin.x
                && point.x <= screen.origin.x + screen.size.width
                && point.y >= screen.origin.y
                && point.y <= screen.origin.y + screen.size.height;
            if !inside {
                menu.setHidden(true);
            }
        }

        #[unsafe(method(control:textView:doCommandBySelector:))]
        unsafe fn control_text_view_do_command_by_selector(
            &self,
            _control: &NSControl,
            _text_view: &NSTextView,
            command_selector: Sel,
        ) -> Bool {
            if command_selector == sel!(moveDown:) {
                return self.move_command_selection(1).into();
            }
            if command_selector == sel!(moveUp:) {
                return self.move_command_selection(-1).into();
            }
            if command_selector == sel!(cancelOperation:) {
                if let Some(menu) = self.ivars().cmd_menu.get() {
                    menu.setHidden(true);
                }
                self.ivars().cmd_sel.set(-1);
                return Bool::YES;
            }
            if command_selector == sel!(insertNewline:) {
                let open = self
                    .ivars()
                    .cmd_menu
                    .get()
                    .map(|m| !m.isHidden())
                    .unwrap_or(false);
                let selected = self.ivars().cmd_sel.get();
                if open && selected >= 0 {
                    if let Some(&(name, _)) = CHAT_COMMANDS.get(selected as usize) {
                        self.run_chat_command(name);
                        return Bool::YES;
                    }
                }
                // Enter sends the message.
                self.send_current();
                return Bool::YES;
            }
            Bool::NO
        }
    }

    unsafe impl NSTextFieldDelegate for AppDelegate {}

    unsafe impl NSApplicationDelegate for AppDelegate {
        #[unsafe(method(applicationDidFinishLaunching:))]
        fn did_finish_launching(&self, _notification: &NSNotification) {
            let mtm = self.mtm();
            let app = NSApplication::sharedApplication(mtm);
            app.setActivationPolicy(NSApplicationActivationPolicy::Regular);
            // Show "lisa" in the menu bar instead of the binary name.
            NSProcessInfo::processInfo().setProcessName(ns_string!("lisa"));

            self.build_menu(&app, mtm);

            let outer = NSSplitViewController::new(mtm);

            let sidebar_view = self.build_sidebar(mtm);
            let sidebar_vc = NSViewController::new(mtm);
            sidebar_vc.setView(&sidebar_view);
            let sidebar_item = NSSplitViewItem::splitViewItemWithViewController(&sidebar_vc);
            sidebar_item.setMinimumThickness(250.0);
            sidebar_item.setMaximumThickness(300.0);
            sidebar_item.setPreferredThicknessFraction(0.09);
            outer.addSplitViewItem(&sidebar_item);
            sidebar_view
                .widthAnchor()
                .constraintGreaterThanOrEqualToConstant(250.0)
                .setActive(true);
            sidebar_view
                .widthAnchor()
                .constraintLessThanOrEqualToConstant(300.0)
                .setActive(true);
            self.ivars().sidebar_item.set(sidebar_item).ok();

            // Content: a vertical split with the chat on top and the log panel
            // underneath (resizable via the divider).
            let content = NSSplitViewController::new(mtm);
            content.splitView().setVertical(false);

            let chat_vc = NSViewController::new(mtm);
            chat_vc.setView(&self.build_chat(mtm));
            let chat_item = NSSplitViewItem::splitViewItemWithViewController(&chat_vc);
            chat_item.setMinimumThickness(200.0);
            content.addSplitViewItem(&chat_item);

            let logs_vc = NSViewController::new(mtm);
            logs_vc.setView(&self.build_logs(mtm));
            let logs_item = NSSplitViewItem::splitViewItemWithViewController(&logs_vc);
            logs_item.setMinimumThickness(80.0);
            logs_item.setMaximumThickness(600.0);
            content.addSplitViewItem(&logs_item);
            logs_item.setCollapsed(true);
            self.ivars().logs_item.set(logs_item).ok();

            let content_item = NSSplitViewItem::splitViewItemWithViewController(&content);
            content_item.setMinimumThickness(420.0);
            outer.addSplitViewItem(&content_item);

            // Right column: serve the loaded model over HTTP for other tools.
            let serve_view = self.build_serve(mtm);
            let serve_vc = NSViewController::new(mtm);
            serve_vc.setView(&serve_view);
            let serve_item = NSSplitViewItem::splitViewItemWithViewController(&serve_vc);
            serve_item.setMinimumThickness(320.0);
            serve_item.setMaximumThickness(320.0);
            outer.addSplitViewItem(&serve_item);
            serve_view
                .widthAnchor()
                .constraintEqualToConstant(320.0)
                .setActive(true);
            self.ivars().serve_item.set(serve_item).ok();

            let split_view = outer.view();
            let container = NSView::new(mtm);
            let backdrop = ui::backdrop(mtm);
            container.addSubview(&backdrop);
            for constraint in [
                backdrop
                    .leadingAnchor()
                    .constraintEqualToAnchor(&container.leadingAnchor()),
                backdrop
                    .trailingAnchor()
                    .constraintEqualToAnchor(&container.trailingAnchor()),
                backdrop.topAnchor().constraintEqualToAnchor(&container.topAnchor()),
                backdrop
                    .bottomAnchor()
                    .constraintEqualToAnchor(&container.bottomAnchor()),
            ] {
                constraint.setActive(true);
            }
            split_view.setTranslatesAutoresizingMaskIntoConstraints(false);
            container.addSubview(&split_view);
            for constraint in [
                split_view
                    .leadingAnchor()
                    .constraintEqualToAnchor(&container.leadingAnchor()),
                split_view
                    .trailingAnchor()
                    .constraintEqualToAnchor(&container.trailingAnchor()),
                split_view.topAnchor().constraintEqualToAnchor(&container.topAnchor()),
                split_view
                    .bottomAnchor()
                    .constraintEqualToAnchor(&container.bottomAnchor()),
            ] {
                constraint.setActive(true);
            }
            let container_vc = NSViewController::new(mtm);
            container_vc.setView(&container);
            container_vc.addChildViewController(&outer);

            let window = unsafe {
                NSWindow::initWithContentRect_styleMask_backing_defer(
                    NSWindow::alloc(mtm),
                    NSRect::new(NSPoint::new(0.0, 0.0), NSSize::new(1440.0, 900.0)),
                    NSWindowStyleMask::Titled
                        | NSWindowStyleMask::Closable
                        | NSWindowStyleMask::Miniaturizable
                        | NSWindowStyleMask::Resizable
                        | NSWindowStyleMask::FullSizeContentView,
                    NSBackingStoreType::Buffered,
                    false,
                )
            };
            unsafe { window.setReleasedWhenClosed(false) };
            window.setTitle(ns_string!("Lisa"));
            window.setTitlebarAppearsTransparent(true);
            window.setTitleVisibility(NSWindowTitleVisibility::Hidden);
            window.setOpaque(false);
            window.setBackgroundColor(Some(&NSColor::clearColor()));
            window.setMovableByWindowBackground(true);
            window.setContentViewController(Some(&container_vc));
            window.setContentSize(NSSize::new(1360.0, 860.0));
            window.setContentMinSize(NSSize::new(1000.0, 640.0));
            window.setDelegate(Some(ProtocolObject::from_ref(self)));
            window.makeKeyAndOrderFront(None);
            window.setFrameOrigin(NSPoint::new(90.0, 150.0));

            // Open with the serve panel at its 320pt minimum and the models
            // panel at its 150pt minimum.
            if let Some(content) = window.contentView() {
                content.layoutSubtreeIfNeeded();
            }
            let split = outer.splitView();
            let total = split.frame().size.width;
            if total > 320.0 {
                split.setPosition_ofDividerAtIndex(total - 320.0, 1);
            }
            split.setPosition_ofDividerAtIndex(250.0, 0);

            self.ivars().window.set(window).ok();
            self.reload_models_table();

            // Drain engine events (log / chat / loaded) on the main thread.
            unsafe {
                NSTimer::scheduledTimerWithTimeInterval_target_selector_userInfo_repeats(
                    TICK_DT,
                    self.target(),
                    sel!(pollEngine:),
                    None,
                    true,
                );
            }

            #[allow(deprecated)]
            app.activateIgnoringOtherApps(true);
        }
    }

    unsafe impl NSWindowDelegate for AppDelegate {
        #[unsafe(method(windowWillClose:))]
        fn window_will_close(&self, _notification: &NSNotification) {
            let app = NSApplication::sharedApplication(self.mtm());
            app.terminate(None);
        }
    }

    unsafe impl NSTableViewDataSource for AppDelegate {
        #[unsafe(method(numberOfRowsInTableView:))]
        fn number_of_rows(&self, _table_view: &NSTableView) -> NSInteger {
            self.ivars().models.borrow().len() as NSInteger
        }
    }

    unsafe impl NSTableViewDelegate for AppDelegate {
        #[unsafe(method_id(tableView:viewForTableColumn:row:))]
        fn view_for_row(
            &self,
            _table_view: &NSTableView,
            _table_column: Option<&NSTableColumn>,
            row: NSInteger,
        ) -> Option<Retained<NSView>> {
            self.make_model_cell(row)
        }

        #[unsafe(method(tableViewSelectionDidChange:))]
        fn selection_did_change(&self, _notification: &NSNotification) {
            // Ignore selection changes caused by our own reloadData (which
            // would otherwise recurse), and anything but a real mouse click.
            if self.ivars().updating_table.get() {
                return;
            }
            let is_click = NSApplication::sharedApplication(self.mtm())
                .currentEvent()
                .map(|e| {
                    matches!(
                        e.r#type(),
                        NSEventType::LeftMouseDown | NSEventType::LeftMouseUp
                    )
                })
                .unwrap_or(false);
            if !is_click {
                return;
            }
            let Some(table) = self.ivars().models_table.get() else {
                return;
            };
            let row = table.selectedRow();
            if row < 0 {
                return;
            }
            let selected = {
                let models = self.ivars().models.borrow();
                models
                    .get(row as usize)
                    .map(|m| (m.repo_id.clone(), m.path.clone(), m.local, m.too_big))
            };
            let Some((repo_id, path, local, too_big)) = selected else {
                return;
            };
            if too_big {
                self.append_engine_log(&format!("{repo_id} is too large for this Mac's memory"));
                return;
            }
            if local {
                // Already the active model?
                if self.ivars().active_repo.borrow().as_deref() == Some(repo_id.as_str()) {
                    return;
                }
                if let Some(window) = self.ivars().window.get() {
                    window.setTitle(&NSString::from_str(&format!(
                        "Lisa — {}",
                        display_name(&repo_id)
                    )));
                }
                self.ivars().downloading.set(false);
                *self.ivars().active_repo.borrow_mut() = Some(repo_id);
                *self.ivars().loading_row.borrow_mut() = Some(row as usize);
                *self.ivars().selected_row.borrow_mut() = Some(row as usize);
                self.reload_models_table();
                self.ivars().model_loaded.set(false);
                self.set_chat_enabled(false);
                if let Some(tx) = self.ivars().engine_tx.get() {
                    let _ = tx.send(Command::Load(PathBuf::from(path)));
                }
            } else {
                // Download in the background and leave the current model
                // loaded so the user can keep chatting meanwhile.
                if self.ivars().downloading.get() {
                    self.append_engine_log("a download is already in progress");
                    return;
                }
                self.ivars().downloading.set(true);
                self.ivars().download_progress.set(0.0);
                self.ivars().progress_tick.set(0);
                *self.ivars().loading_row.borrow_mut() = Some(row as usize);
                self.reload_models_table();
                if let Some(tx) = self.ivars().engine_tx.get() {
                    let _ = tx.send(Command::Download(repo_id));
                }
            }
        }
    }

    impl AppDelegate {
        #[unsafe(method(toggleSidebar:))]
        fn toggle_sidebar(&self, _sender: Option<&AnyObject>) {
            if let Some(item) = self.ivars().sidebar_item.get() {
                item.setCollapsed(!item.isCollapsed());
            }
        }

        #[unsafe(method(deleteModel:))]
        fn delete_model(&self, _sender: Option<&AnyObject>) {
            let Some(table) = self.ivars().models_table.get() else {
                return;
            };
            let row = table.clickedRow();
            if row < 0 {
                return;
            }
            let selected = {
                let models = self.ivars().models.borrow();
                models
                    .get(row as usize)
                    .map(|m| (m.repo_id.clone(), m.path.clone(), m.local))
            };
            let Some((repo_id, path, local)) = selected else {
                return;
            };
            if !local {
                self.append_engine_log("only local models can be deleted");
                return;
            }
            if *self.ivars().selected_row.borrow() == Some(row as usize) {
                self.append_engine_log("select another model before deleting this one");
                return;
            }
            let mtm = self.mtm();
            let alert = NSAlert::new(mtm);
            alert.setMessageText(&NSString::from_str("Delete model?"));
            alert.setInformativeText(&NSString::from_str(&format!(
                "{repo_id} will be removed from disk."
            )));
            alert.addButtonWithTitle(&NSString::from_str("Delete"));
            alert.addButtonWithTitle(&NSString::from_str("Cancel"));
            if alert.runModal() != NSAlertFirstButtonReturn {
                return;
            }
            let p = PathBuf::from(&path);
            let is_hub_snapshot = p
                .parent()
                .and_then(|x| x.file_name())
                .map(|n| n.to_string_lossy() == "snapshots")
                .unwrap_or(false);
            let repo_dir = if is_hub_snapshot {
                p.parent().and_then(|x| x.parent()).map(|x| x.to_path_buf())
            } else {
                Some(p.clone())
            };
            if let Some(dir) = repo_dir {
                match std::fs::remove_dir_all(&dir) {
                    Ok(()) => self.append_engine_log(&format!("deleted {repo_id}")),
                    Err(e) => self.append_engine_log(&format!("delete failed: {e}")),
                }
            }
            self.refresh_models();
        }

        #[unsafe(method(toggleLogs:))]
        fn toggle_logs(&self, _sender: Option<&AnyObject>) {
            if let Some(item) = self.ivars().logs_item.get() {
                item.setCollapsed(!item.isCollapsed());
            }
        }

        #[unsafe(method(pollEngine:))]
        fn poll_engine(&self, _timer: &NSTimer) {
            self.tick_typing();
            self.tick_download_progress();
            let Some(events) = self.ivars().engine_events.get() else {
                return;
            };
            let drained: Vec<UiEvent> = std::mem::take(&mut *events.lock().unwrap());
            for event in drained {
                match event {
                    UiEvent::Log(line) => self.append_engine_log(&line),
                    UiEvent::UserMessage(text) => self.add_user_message(&text),
                    UiEvent::AssistantDelta(piece) => self.append_assistant_delta(&piece),
                    UiEvent::AssistantReset(text) => self.set_assistant_text(&text),
                    UiEvent::AssistantDone => self.finalize_assistant(),
                    UiEvent::Loaded(ok) => {
                        *self.ivars().loading_row.borrow_mut() = None;
                        self.reload_models_table();
                        self.ivars().model_loaded.set(ok);
                        self.update_chat_enabled();
                    }
                    UiEvent::ServeStarted(url) => {
                        self.set_serving(true, &format!("listening on {url}"))
                    }
                    UiEvent::ServeStopped => self.set_serving(false, "stopped"),
                    UiEvent::RemoteModels(ids) => {
                        *self.ivars().remote_ids.borrow_mut() = ids;
                        self.refresh_models();
                    }
                    UiEvent::ModelsChanged => {
                        if self.ivars().downloading.get() {
                            self.ivars().downloading.set(false);
                            *self.ivars().loading_row.borrow_mut() = None;
                        }
                        self.refresh_models();
                    }
                }
            }
        }

        #[unsafe(method(toggleServe:))]
        fn toggle_serve(&self, _sender: Option<&AnyObject>) {
            if self.ivars().serving.get() {
                if let Some(stop) = self.ivars().serve_stop.borrow().as_ref() {
                    stop.store(true, Ordering::Relaxed);
                }
                if let Some(status) = self.ivars().serve_status.get() {
                    status.setStringValue(&NSString::from_str("stopping…"));
                }
                return;
            }
            let port = self
                .ivars()
                .serve_port
                .get()
                .map(|p| p.stringValue().to_string())
                .unwrap_or_else(|| "5472".into());
            let port: u16 = match port.trim().parse() {
                Ok(p) => p,
                Err(_) => {
                    if let Some(status) = self.ivars().serve_status.get() {
                        status.setStringValue(&NSString::from_str("invalid port"));
                    }
                    return;
                }
            };
            let stop = Arc::new(AtomicBool::new(false));
            *self.ivars().serve_stop.borrow_mut() = Some(stop.clone());
            let addr = format!("127.0.0.1:{port}");
            if let Some(tx) = self.ivars().engine_tx.get() {
                let _ = tx.send(Command::StartServe { addr, stop });
            }
            self.set_serving(true, "starting…");
        }

        #[unsafe(method(runCommand:))]
        fn run_command(&self, sender: Option<&AnyObject>) {
            let Some(sender) = sender else {
                return;
            };
            let tag: NSInteger = unsafe { msg_send![sender, tag] };
            let Some(&(name, _)) = CHAT_COMMANDS.get(tag as usize) else {
                return;
            };
            self.run_chat_command(name);
        }

        #[unsafe(method(sendMessage:))]
        fn send_message(&self, _sender: Option<&AnyObject>) {
            self.send_current();
        }
    }
);

impl AppDelegate {
    pub fn new(mtm: MainThreadMarker) -> Retained<Self> {
        let (tx, rx) = mpsc::channel::<Command>();
        let events: Arc<Mutex<Vec<UiEvent>>> = Arc::new(Mutex::new(Vec::new()));
        spawn_engine(rx, events.clone());
        let this = Self::alloc(mtm).set_ivars(AppDelegateIvars {
            window: OnceCell::new(),
            sidebar_item: OnceCell::new(),
            logs_item: OnceCell::new(),
            models: RefCell::new(discover_models()),
            remote_ids: RefCell::new(Vec::new()),
            models_table: OnceCell::new(),
            loading_row: RefCell::new(None),
            active_repo: RefCell::new(None),
            downloading: Cell::new(false),
            download_progress: Cell::new(0.0),
            progress_tick: Cell::new(0),
            selected_row: RefCell::new(None),
            updating_table: Cell::new(false),
            chat_stack: OnceCell::new(),
            chat_scroll: OnceCell::new(),
            chat_placeholder: OnceCell::new(),
            assistant_label: RefCell::new(None),
            assistant_stack: RefCell::new(None),
            typing: Cell::new(false),
            typing_tick: Cell::new(0),
            chat_input: OnceCell::new(),
            chat_send: OnceCell::new(),
            cmd_menu: OnceCell::new(),
            cmd_rows: RefCell::new(Vec::new()),
            cmd_bgs: RefCell::new(Vec::new()),
            cmd_sel: Cell::new(-1),
            system_view: OnceCell::new(),
            engine_log: OnceCell::new(),
            engine_tx: OnceCell::new(),
            engine_events: OnceCell::new(),
            serve_item: OnceCell::new(),
            serve_port: OnceCell::new(),
            serve_switch: OnceCell::new(),
            serve_status: OnceCell::new(),
            serve_state: OnceCell::new(),
            serve_dot: OnceCell::new(),
            serve_stop: RefCell::new(None),
            serving: Cell::new(false),
            model_loaded: Cell::new(false),
        });
        let this: Retained<Self> = unsafe { msg_send![super(this), init] };
        this.ivars().engine_tx.set(tx).ok();
        this.ivars().engine_events.set(events.clone()).ok();
        {
            let events = events.clone();
            let _ = thread::Builder::new()
                .name("lisa-model-list".into())
                .spawn(move || {
                    let ids = fetch_remote_models();
                    events.lock().unwrap().push(UiEvent::RemoteModels(ids));
                });
        }
        this
    }

    /// Send the current input: run a slash command locally, else hand the turn
    /// to the engine worker thread.
    fn send_current(&self) {
        let Some(input) = self.ivars().chat_input.get() else {
            return;
        };
        let text = input.stringValue().to_string();
        let text = text.trim().to_string();
        if text.is_empty() {
            return;
        }
        input.setStringValue(ns_string!(""));
        self.update_command_menu();
        // Slash commands run locally instead of going to the model.
        if text.starts_with('/') && !text.contains(char::is_whitespace) {
            if let Some(&(name, _)) = CHAT_COMMANDS.iter().find(|(n, _)| *n == text) {
                self.run_chat_command(name);
                return;
            }
        }
        let system = self
            .ivars()
            .system_view
            .get()
            .map(|v| v.string().to_string())
            .unwrap_or_default();
        if let Some(tx) = self.ivars().engine_tx.get() {
            let _ = tx.send(Command::Chat { text, system });
        }
    }

    fn target(&self) -> &AnyObject {
        let object: &NSObject = ClassType::as_super(self);
        ClassType::as_super(object)
    }

    fn make_model_cell(&self, row: NSInteger) -> Option<Retained<NSView>> {
        let mtm = self.mtm();
        let models = self.ivars().models.borrow();
        let model = models.get(row as usize)?;

        let cell = NSView::new(mtm);
        cell.setFrame(NSRect::new(NSPoint::new(0.0, 0.0), NSSize::new(150.0, 30.0)));

        // Custom highlight for the selected model (clear even when the table
        // isn't focused).
        if *self.ivars().selected_row.borrow() == Some(row as usize) {
            let bg = NSBox::new(mtm);
            bg.setBoxType(NSBoxType::Custom);
            bg.setBorderWidth(0.0);
            bg.setCornerRadius(6.0);
            bg.setFillColor(&NSColor::controlAccentColor().colorWithAlphaComponent(0.22));
            bg.setTranslatesAutoresizingMaskIntoConstraints(false);
            cell.addSubview(&bg);
            bg.leadingAnchor()
                .constraintEqualToAnchor_constant(&cell.leadingAnchor(), 4.0)
                .setActive(true);
            bg.trailingAnchor()
                .constraintEqualToAnchor_constant(&cell.trailingAnchor(), -4.0)
                .setActive(true);
            bg.topAnchor()
                .constraintEqualToAnchor_constant(&cell.topAnchor(), 1.0)
                .setActive(true);
            bg.bottomAnchor()
                .constraintEqualToAnchor_constant(&cell.bottomAnchor(), -1.0)
                .setActive(true);
        }

        let symbol = if model.too_big {
            ns_string!("nosign")
        } else if model.local {
            ns_string!("cube")
        } else {
            ns_string!("arrow.down.circle")
        };
        let icon = NSImageView::new(mtm);
        if let Some(image) =
            NSImage::imageWithSystemSymbolName_accessibilityDescription(symbol, None)
        {
            icon.setImage(Some(&image));
        }
        let name_color = if model.too_big {
            NSColor::tertiaryLabelColor()
        } else if model.local {
            NSColor::labelColor()
        } else {
            NSColor::secondaryLabelColor()
        };
        icon.setContentTintColor(Some(&name_color));
        icon.setTranslatesAutoresizingMaskIntoConstraints(false);
        cell.addSubview(&icon);

        let name = ui::label(mtm, display_name(&model.repo_id), 13.0, false, &name_color);
        name.setTranslatesAutoresizingMaskIntoConstraints(false);
        let tip = if model.too_big {
            format!(
                "{} — too large for this Mac's memory ({:.0} GB)",
                model.repo_id,
                model.size as f64 / 1e9
            )
        } else if model.local {
            model.repo_id.clone()
        } else {
            format!(
                "{} — click to download from huggingface.co/agnosticeng",
                model.repo_id
            )
        };
        name.setToolTip(Some(&NSString::from_str(&tip)));
        if let Some(cell) = name.cell() {
            cell.setLineBreakMode(NSLineBreakMode::ByTruncatingTail);
            cell.setUsesSingleLineMode(true);
        }
        cell.addSubview(&name);

        icon.leadingAnchor()
            .constraintEqualToAnchor_constant(&cell.leadingAnchor(), 10.0)
            .setActive(true);
        icon.centerYAnchor()
            .constraintEqualToAnchor(&cell.centerYAnchor())
            .setActive(true);
        ui::size(&icon, 18.0, 18.0);

        name.leadingAnchor()
            .constraintEqualToAnchor_constant(&icon.trailingAnchor(), 10.0)
            .setActive(true);
        name.centerYAnchor()
            .constraintEqualToAnchor(&cell.centerYAnchor())
            .setActive(true);

        if *self.ivars().loading_row.borrow() == Some(row as usize) {
            if self.ivars().downloading.get() {
                let ring = ProgressRing::new(mtm, self.ivars().download_progress.get());
                ring.setTranslatesAutoresizingMaskIntoConstraints(false);
                cell.addSubview(&ring);
                ring.trailingAnchor()
                    .constraintEqualToAnchor_constant(&cell.trailingAnchor(), -10.0)
                    .setActive(true);
                ring.centerYAnchor()
                    .constraintEqualToAnchor(&cell.centerYAnchor())
                    .setActive(true);
                ui::size(&ring, 18.0, 18.0);
                name.trailingAnchor()
                    .constraintEqualToAnchor_constant(&ring.leadingAnchor(), -8.0)
                    .setActive(true);
            } else {
                let spinner = NSProgressIndicator::new(mtm);
                spinner.setStyle(NSProgressIndicatorStyle::Spinning);
                spinner.setIndeterminate(true);
                spinner.setControlSize(NSControlSize::Small);
                spinner.setDisplayedWhenStopped(false);
                spinner.setTranslatesAutoresizingMaskIntoConstraints(false);
                unsafe { spinner.startAnimation(None) };
                cell.addSubview(&spinner);
                spinner.trailingAnchor()
                    .constraintEqualToAnchor_constant(&cell.trailingAnchor(), -10.0)
                    .setActive(true);
                spinner.centerYAnchor()
                    .constraintEqualToAnchor(&cell.centerYAnchor())
                    .setActive(true);
                ui::size(&spinner, 16.0, 16.0);
                name.trailingAnchor()
                    .constraintEqualToAnchor_constant(&spinner.leadingAnchor(), -8.0)
                    .setActive(true);
            }
        } else {
            name.trailingAnchor()
                .constraintEqualToAnchor_constant(&cell.trailingAnchor(), -10.0)
                .setActive(true);
        }

        Some(cell)
    }

    /// Append a line to the engine-log panel, called from engine callbacks (the
    /// first call replaces the placeholder text).
    pub fn append_engine_log(&self, line: &str) {
        if let Some(log) = self.ivars().engine_log.get() {
            let current = log.string().to_string();
            let updated = if current.trim().is_empty() {
                format!("{line}\n")
            } else {
                format!("{current}{line}\n")
            };
            log.setString(&NSString::from_str(&updated));
            let length = log.string().length();
            log.scrollRangeToVisible(NSRange::new(length, 0));
        }
    }

    /// Build a message bubble row; returns `(row, content_stack)` where block
    /// views (or a streaming label) are placed.
    fn make_bubble(&self, is_user: bool) -> (Retained<NSView>, Retained<NSStackView>) {
        let mtm = self.mtm();
        let row = NSView::new(mtm);
        row.setTranslatesAutoresizingMaskIntoConstraints(false);

        let fill = if is_user {
            NSColor::controlAccentColor()
        } else {
            NSColor::controlBackgroundColor()
        };
        let bubble = BubbleView::new(mtm, &fill);
        bubble.setTranslatesAutoresizingMaskIntoConstraints(false);
        row.addSubview(&bubble);

        let content = NSStackView::new(mtm);
        content.setOrientation(NSUserInterfaceLayoutOrientation::Vertical);
        content.setAlignment(NSLayoutAttribute::Leading);
        content.setSpacing(8.0);
        content.setTranslatesAutoresizingMaskIntoConstraints(false);
        bubble.addSubview(&content);

        content
            .leadingAnchor()
            .constraintEqualToAnchor_constant(&bubble.leadingAnchor(), 14.0)
            .setActive(true);
        content
            .trailingAnchor()
            .constraintEqualToAnchor_constant(&bubble.trailingAnchor(), -14.0)
            .setActive(true);
        content
            .topAnchor()
            .constraintEqualToAnchor_constant(&bubble.topAnchor(), 10.0)
            .setActive(true);
        content
            .bottomAnchor()
            .constraintEqualToAnchor_constant(&bubble.bottomAnchor(), -10.0)
            .setActive(true);

        bubble
            .widthAnchor()
            .constraintLessThanOrEqualToConstant(BUBBLE_MAX_WIDTH)
            .setActive(true);
        if is_user {
            bubble
                .trailingAnchor()
                .constraintEqualToAnchor_constant(&row.trailingAnchor(), -8.0)
                .setActive(true);
        } else {
            bubble
                .leadingAnchor()
                .constraintEqualToAnchor_constant(&row.leadingAnchor(), 8.0)
                .setActive(true);
        }
        bubble
            .topAnchor()
            .constraintEqualToAnchor(&row.topAnchor())
            .setActive(true);
        bubble
            .bottomAnchor()
            .constraintEqualToAnchor(&row.bottomAnchor())
            .setActive(true);

        (row, content)
    }

    /// Show/hide the "/" command menu for the current input text: visible
    /// while the text starts with "/" and matches at least one command. The
    /// first match is selected so Enter runs it straight away.
    fn update_command_menu(&self) {
        let (Some(input), Some(menu)) = (self.ivars().chat_input.get(), self.ivars().cmd_menu.get())
        else {
            return;
        };
        let text = input.stringValue().to_string();
        let mut first: Option<usize> = None;
        {
            let rows = self.ivars().cmd_rows.borrow();
            for (i, row) in rows.iter().enumerate() {
                let Some(&(name, _)) = CHAT_COMMANDS.get(i) else {
                    continue;
                };
                let hit = text.starts_with('/') && name.starts_with(&text);
                row.setHidden(!hit);
                if hit && first.is_none() {
                    first = Some(i);
                }
            }
        }
        match first {
            Some(i) => {
                menu.setHidden(false);
                self.ivars().cmd_sel.set(i as isize);
            }
            None => {
                menu.setHidden(true);
                self.ivars().cmd_sel.set(-1);
            }
        }
        self.apply_command_selection();
    }

    /// Repaint the command rows so only the selected one has a highlight.
    fn apply_command_selection(&self) {
        let sel = self.ivars().cmd_sel.get();
        {
            let rows = self.ivars().cmd_rows.borrow();
            for (i, row) in rows.iter().enumerate() {
                let Some(&(name, desc)) = CHAT_COMMANDS.get(i) else {
                    continue;
                };
                row.setAttributedTitle(&command_row_title(name, desc, i as isize == sel));
            }
        }
        let bgs = self.ivars().cmd_bgs.borrow();
        for (i, bg) in bgs.iter().enumerate() {
            bg.setHidden(i as isize != sel);
        }
    }

    /// Move the dropdown selection by `delta` rows (wrapping). True if the
    /// menu is open and the key was consumed.
    fn move_command_selection(&self, delta: isize) -> bool {
        if self
            .ivars()
            .cmd_menu
            .get()
            .map(|m| m.isHidden())
            .unwrap_or(true)
        {
            return false;
        }
        let visible: Vec<usize> = self
            .ivars()
            .cmd_rows
            .borrow()
            .iter()
            .enumerate()
            .filter(|(_, r)| !r.isHidden())
            .map(|(i, _)| i)
            .collect();
        if visible.is_empty() {
            return false;
        }
        let cur = self.ivars().cmd_sel.get();
        let pos = visible
            .iter()
            .position(|&i| i as isize == cur)
            .unwrap_or(0);
        let next = (pos as isize + delta).rem_euclid(visible.len() as isize) as usize;
        self.ivars().cmd_sel.set(visible[next] as isize);
        self.apply_command_selection();
        true
    }

    /// Run a slash command (`name` is one of `CHAT_COMMANDS`' names).
    fn run_chat_command(&self, name: &str) {
        match name {
            "/clear" => {
                self.clear_chat();
                if let Some(tx) = self.ivars().engine_tx.get() {
                    let _ = tx.send(Command::Clear);
                }
            }
            "/help" => self.show_command_help(),
            _ => {}
        }
        if let Some(input) = self.ivars().chat_input.get() {
            input.setStringValue(ns_string!(""));
        }
        self.update_command_menu();
    }

    /// Append a locally-rendered note bubble (no model round-trip).
    fn add_note(&self, text: &str) {
        let Some(chat) = self.ivars().chat_stack.get() else {
            return;
        };
        if let Some(placeholder) = self.ivars().chat_placeholder.get() {
            placeholder.setHidden(true);
        }
        let (row, content) = self.make_bubble(false);
        render_blocks(self.mtm(), &content, text, &NSColor::labelColor());
        chat.addArrangedSubview(&row);
        self.scroll_chat_to_bottom();
    }

    /// `/help` — list the available slash commands.
    fn show_command_help(&self) {
        let mut md = String::from("**Commands**\n\n");
        for (name, desc) in CHAT_COMMANDS {
            md.push_str(&format!("- `{name}` — {desc}\n"));
        }
        self.add_note(md.trim_end());
    }

    /// `/clear` — empty the transcript and restore the empty-state placeholder.
    fn clear_chat(&self) {
        if let Some(chat) = self.ivars().chat_stack.get() {
            let rows = chat.arrangedSubviews();
            for i in (0..rows.count()).rev() {
                let view = rows.objectAtIndex(i);
                chat.removeArrangedSubview(&view);
                view.removeFromSuperview();
            }
        }
        *self.ivars().assistant_label.borrow_mut() = None;
        *self.ivars().assistant_stack.borrow_mut() = None;
        self.ivars().typing.set(false);
        if let Some(placeholder) = self.ivars().chat_placeholder.get() {
            placeholder.setHidden(false);
        }
        self.scroll_chat_to_bottom();
    }

    fn add_user_message(&self, text: &str) {
        let Some(chat) = self.ivars().chat_stack.get() else {
            return;
        };
        if let Some(placeholder) = self.ivars().chat_placeholder.get() {
            placeholder.setHidden(true);
        }
        let (row, content) = self.make_bubble(true);
        render_blocks(self.mtm(), &content, text, &NSColor::whiteColor());
        chat.addArrangedSubview(&row);

        // Placeholder assistant bubble with an animated typing indicator.
        let (row, content) = self.make_bubble(false);
        let label = ui::wrapping_label(self.mtm(), "", BUBBLE_FONT_SIZE, &NSColor::labelColor());
        label.setTranslatesAutoresizingMaskIntoConstraints(false);
        label.setMaximumNumberOfLines(0);
        label.setPreferredMaxLayoutWidth(BUBBLE_MAX_WIDTH - 30.0);
        label.setAttributedStringValue(&typing_dots(self.mtm(), 0.0));
        content.addArrangedSubview(&label);
        chat.addArrangedSubview(&row);
        *self.ivars().assistant_label.borrow_mut() = Some(label);
        *self.ivars().assistant_stack.borrow_mut() = Some(content);
        self.ivars().typing.set(true);
        self.ivars().typing_tick.set(0);
        self.scroll_chat_to_bottom();
    }

    /// Advance the typing-indicator animation (called from the poll timer).
    fn tick_typing(&self) {
        if !self.ivars().typing.get() {
            return;
        }
        let tick = self.ivars().typing_tick.get() + 1;
        self.ivars().typing_tick.set(tick);
        let time = tick as f64 * TICK_DT;
        if let Some(label) = self.ivars().assistant_label.borrow().as_ref() {
            label.setAttributedStringValue(&typing_dots(self.mtm(), time));
        }
    }

    /// Refresh the download ring from the bytes written to the Hub cache.
    /// Throttled to ~3 Hz so the directory walk stays cheap.
    fn tick_download_progress(&self) {
        if !self.ivars().downloading.get() {
            return;
        }
        let tick = self.ivars().progress_tick.get() + 1;
        self.ivars().progress_tick.set(tick);
        if tick % 10 != 0 {
            return;
        }
        let Some(row) = *self.ivars().loading_row.borrow() else {
            return;
        };
        let (repo, total) = {
            let models = self.ivars().models.borrow();
            match models.get(row) {
                Some(m) => (m.repo_id.clone(), m.size),
                None => return,
            }
        };
        if total > 0 {
            let done = hf_blobs_size(&repo);
            let progress = (done as f64 / total as f64).clamp(0.0, 1.0);
            self.ivars().download_progress.set(progress);
        }
        if let Some(table) = self.ivars().models_table.get() {
            let rows = NSIndexSet::indexSetWithIndex(row as NSUInteger);
            let columns = NSIndexSet::indexSetWithIndex(0);
            table.reloadDataForRowIndexes_columnIndexes(&rows, &columns);
        }
    }

    /// Stream a chunk into the active assistant bubble (plain while generating).
    fn append_assistant_delta(&self, piece: &str) {
        let stick = self.chat_near_bottom();
        let was_typing = self.ivars().typing.get();
        self.ivars().typing.set(false);
        let label = self.ivars().assistant_label.borrow().clone();
        if let Some(label) = label {
            let updated = if was_typing {
                piece.to_string()
            } else {
                let current = label.stringValue().to_string();
                format!("{current}{piece}")
            };
            label.setStringValue(&NSString::from_str(&updated));
        }
        if stick {
            self.scroll_chat_to_bottom();
        }
    }

    /// Replace the active assistant bubble's text (streaming reset).
    fn set_assistant_text(&self, text: &str) {
        let stick = self.chat_near_bottom();
        self.ivars().typing.set(false);
        let label = self.ivars().assistant_label.borrow().clone();
        if let Some(label) = label {
            label.setStringValue(&NSString::from_str(text));
        }
        if stick {
            self.scroll_chat_to_bottom();
        }
    }

    /// Markdown-format the finished reply into block views.
    fn finalize_assistant(&self) {
        self.ivars().typing.set(false);
        let label = self.ivars().assistant_label.borrow().clone();
        let content = self.ivars().assistant_stack.borrow().clone();
        if let (Some(label), Some(content)) = (label, content) {
            let text = label.stringValue().to_string();
            label.removeFromSuperview();
            render_blocks(self.mtm(), &content, &text, &NSColor::labelColor());
        }
        *self.ivars().assistant_label.borrow_mut() = None;
        *self.ivars().assistant_stack.borrow_mut() = None;
        self.scroll_chat_to_bottom();
    }

    /// True when the transcript is scrolled at (or within a hair of) the
    /// bottom, so streaming should keep following the growing reply.
    fn chat_near_bottom(&self) -> bool {
        let Some(scroll) = self.ivars().chat_scroll.get() else {
            return true;
        };
        let Some(doc) = scroll.documentView() else {
            return true;
        };
        let clip = scroll.contentView();
        let content_h = doc.frame().size.height;
        let max_y = (content_h - clip.bounds().size.height).max(0.0);
        max_y - clip.bounds().origin.y < 80.0
    }

    fn scroll_chat_to_bottom(&self) {
        let Some(scroll) = self.ivars().chat_scroll.get() else {
            return;
        };
        scroll.layoutSubtreeIfNeeded();
        let Some(doc) = scroll.documentView() else {
            return;
        };
        doc.layoutSubtreeIfNeeded();
        let clip = scroll.contentView();
        let max_y = (doc.frame().size.height - clip.bounds().size.height).max(0.0);
        clip.scrollToPoint(NSPoint::new(0.0, max_y));
        scroll.reflectScrolledClipView(&clip);
    }

    /// Chat is enabled whenever a model is loaded; the engine thread services
    /// chat and served HTTP requests from the same queue.
    fn update_chat_enabled(&self) {
        self.set_chat_enabled(self.ivars().model_loaded.get());
    }

    /// Enable the chat input + Send button once a model is loaded (and focus
    /// the input). While no model is loaded the input is greyed and disabled.
    fn set_chat_enabled(&self, on: bool) {
        if let Some(input) = self.ivars().chat_input.get() {
            input.setEnabled(on);
            let color = if on {
                NSColor::whiteColor()
            } else {
                NSColor::tertiaryLabelColor()
            };
            input.setTextColor(Some(&color));
            if on {
                if let Some(window) = self.ivars().window.get() {
                    window.makeFirstResponder(Some(&**input));
                }
            }
        }
        if let Some(send) = self.ivars().chat_send.get() {
            send.setEnabled(on);
        }
    }

    /// Merge the local models with the remote (agnosticeng) ids, flag models
    /// too large for RAM, and refresh the table.
    fn refresh_models(&self) {
        let local = discover_models();
        let remote = self.ivars().remote_ids.borrow().clone();
        let threshold = (ram_total() as f64 * 0.9) as u64;
        let mut merged = local;
        for (id, size) in remote {
            if merged.iter().any(|m| m.repo_id == id) {
                continue;
            }
            merged.push(Model {
                repo_id: id,
                model_type: String::new(),
                path: String::new(),
                local: false,
                size,
                too_big: false,
            });
        }
        for m in &mut merged {
            m.too_big = m.size > 0 && threshold > 0 && m.size > threshold;
        }
        merged.sort_by(|a, b| {
            (a.too_big, !a.local, &a.repo_id).cmp(&(b.too_big, !b.local, &b.repo_id))
        });
        *self.ivars().models.borrow_mut() = merged;
        // Re-anchor the highlight to the active model after the re-sort.
        let active = self.ivars().active_repo.borrow().clone();
        let index = active
            .as_ref()
            .and_then(|r| self.ivars().models.borrow().iter().position(|m| &m.repo_id == r));
        *self.ivars().selected_row.borrow_mut() = index;
        self.reload_models_table();
    }

    /// Reload the model list while keeping the selected row highlighted.
    fn reload_models_table(&self) {
        let Some(table) = self.ivars().models_table.get() else {
            return;
        };
        let selected = *self.ivars().selected_row.borrow();
        self.ivars().updating_table.set(true);
        table.reloadData();
        if let Some(row) = selected {
            let index = NSIndexSet::indexSetWithIndex(row);
            table.selectRowIndexes_byExtendingSelection(&index, false);
        }
        self.ivars().updating_table.set(false);
    }

    fn build_menu(&self, app: &NSApplication, mtm: MainThreadMarker) {
        let menu = NSMenu::initWithTitle(NSMenu::alloc(mtm), ns_string!(""));
        let app_item = unsafe {
            NSMenuItem::initWithTitle_action_keyEquivalent(
                NSMenuItem::alloc(mtm),
                ns_string!("lisa"),
                None,
                ns_string!(""),
            )
        };
        let app_menu = NSMenu::initWithTitle(NSMenu::alloc(mtm), ns_string!("lisa"));
        unsafe {
            app_menu.addItemWithTitle_action_keyEquivalent(
                ns_string!("Quit Lisa"),
                Some(sel!(terminate:)),
                ns_string!("q"),
            )
        };
        app_item.setSubmenu(Some(&app_menu));
        menu.addItem(&app_item);

        let view_item = unsafe {
            NSMenuItem::initWithTitle_action_keyEquivalent(
                NSMenuItem::alloc(mtm),
                ns_string!("View"),
                None,
                ns_string!(""),
            )
        };
        let view_menu = NSMenu::initWithTitle(NSMenu::alloc(mtm), ns_string!("View"));
        let toggles = [
            ("Toggle Sidebar", sel!(toggleSidebar:), "s"),
            ("Toggle Logs Panel", sel!(toggleLogs:), "l"),
        ];
        for (title, action, key) in toggles {
            let item = unsafe {
                NSMenuItem::initWithTitle_action_keyEquivalent(
                    NSMenuItem::alloc(mtm),
                    &NSString::from_str(title),
                    Some(action),
                    &NSString::from_str(key),
                )
            };
            item.setKeyEquivalentModifierMask(NSEventModifierFlags::Control | NSEventModifierFlags::Command);
            unsafe { item.setTarget(Some(self.target())) };
            view_menu.addItem(&item);
        }
        view_item.setSubmenu(Some(&view_menu));
        menu.addItem(&view_item);

        app.setMainMenu(Some(&menu));
        // AppKit may re-title the app menu from the process name; force it.
        if let Some(main) = app.mainMenu() {
            if let Some(item) = main.itemAtIndex(0) {
                item.setTitle(ns_string!("lisa"));
            }
        }
    }

    fn build_sidebar(&self, mtm: MainThreadMarker) -> Retained<NSView> {
        let root = NSView::new(mtm);

        let label = ui::label(mtm, "MODELS", 10.0, true, &NSColor::secondaryLabelColor());
        label.setTranslatesAutoresizingMaskIntoConstraints(false);
        root.addSubview(&label);

        let table = NSTableView::new(mtm);
        let column = NSTableColumn::new(mtm);
        column.setWidth(150.0);
        table.addTableColumn(&column);
        table.setHeaderView(None);
        table.setRowHeight(30.0);
        table.setColumnAutoresizingStyle(
            NSTableViewColumnAutoresizingStyle::LastColumnOnlyAutoresizingStyle,
        );
        table.setBackgroundColor(&NSColor::clearColor());
        table.setSelectionHighlightStyle(NSTableViewSelectionHighlightStyle::None);
        table.setIntercellSpacing(NSSize::new(0.0, 1.0));
        unsafe { table.setDataSource(Some(ProtocolObject::from_ref(self))) };
        unsafe { table.setDelegate(Some(ProtocolObject::from_ref(self))) };
        let menu = NSMenu::initWithTitle(NSMenu::alloc(mtm), ns_string!(""));
        let delete_item = unsafe {
            NSMenuItem::initWithTitle_action_keyEquivalent(
                NSMenuItem::alloc(mtm),
                ns_string!("Delete Model"),
                Some(sel!(deleteModel:)),
                ns_string!(""),
            )
        };
        unsafe { delete_item.setTarget(Some(self.target())) };
        menu.addItem(&delete_item);
        unsafe { table.setMenu(Some(&menu)) };
        self.ivars().models_table.set(table.clone()).ok();

        let scroll = ui::scroll(mtm, &table);
        root.addSubview(&scroll);

        label
            .leadingAnchor()
            .constraintEqualToAnchor_constant(&root.leadingAnchor(), 16.0)
            .setActive(true);
        label
            .topAnchor()
            .constraintEqualToAnchor_constant(&root.topAnchor(), 44.0)
            .setActive(true);

        scroll
            .leadingAnchor()
            .constraintEqualToAnchor(&root.leadingAnchor())
            .setActive(true);
        scroll
            .trailingAnchor()
            .constraintEqualToAnchor(&root.trailingAnchor())
            .setActive(true);
        scroll
            .topAnchor()
            .constraintEqualToAnchor_constant(&label.bottomAnchor(), 10.0)
            .setActive(true);
        scroll
            .bottomAnchor()
            .constraintEqualToAnchor(&root.bottomAnchor())
            .setActive(true);

        root
    }

    fn build_chat(&self, mtm: MainThreadMarker) -> Retained<NSView> {
        let root = NSView::new(mtm);

        let document = FlippedView::new(mtm);
        document.setTranslatesAutoresizingMaskIntoConstraints(false);
        let stack = ui::stack(
            mtm,
            NSUserInterfaceLayoutOrientation::Vertical,
            10.0,
            NSLayoutAttribute::Width,
        );
        stack.setEdgeInsets(NSEdgeInsets {
            top: 22.0,
            left: 16.0,
            bottom: 90.0,
            right: 16.0,
        });
        document.addSubview(&stack);
        stack
            .leadingAnchor()
            .constraintEqualToAnchor(&document.leadingAnchor())
            .setActive(true);
        stack
            .trailingAnchor()
            .constraintEqualToAnchor(&document.trailingAnchor())
            .setActive(true);
        stack
            .topAnchor()
            .constraintEqualToAnchor(&document.topAnchor())
            .setActive(true);
        stack
            .bottomAnchor()
            .constraintEqualToAnchor(&document.bottomAnchor())
            .setActive(true);
        self.ivars().chat_stack.set(stack).ok();

        let scroll = NSScrollView::new(mtm);
        scroll.setDocumentView(Some(&document));
        scroll.setHasVerticalScroller(true);
        scroll.setDrawsBackground(false);
        scroll.setBorderType(NSBorderType::NoBorder);
        scroll.setTranslatesAutoresizingMaskIntoConstraints(false);
        scroll.setAutomaticallyAdjustsContentInsets(false);
        scroll.setContentInsets(NSEdgeInsets {
            top: 0.0,
            left: 0.0,
            bottom: 0.0,
            right: 0.0,
        });
        document
            .widthAnchor()
            .constraintEqualToAnchor(&scroll.contentView().widthAnchor())
            .setActive(true);
        self.ivars().chat_scroll.set(scroll.clone()).ok();
        root.addSubview(&scroll);

        // Empty-state placeholder, hidden once the first message arrives.
        let placeholder = ui::label(
            mtm,
            "Select a model and start chatting",
            13.0,
            false,
            &NSColor::tertiaryLabelColor(),
        );
        placeholder.setTranslatesAutoresizingMaskIntoConstraints(false);
        root.addSubview(&placeholder);
        placeholder
            .centerXAnchor()
            .constraintEqualToAnchor(&scroll.centerXAnchor())
            .setActive(true);
        placeholder
            .centerYAnchor()
            .constraintEqualToAnchor(&scroll.centerYAnchor())
            .setActive(true);
        self.ivars().chat_placeholder.set(placeholder).ok();

        let input_bg = NSBox::new(mtm);
        input_bg.setBoxType(NSBoxType::Custom);
        input_bg.setCornerRadius(10.0);
        input_bg.setBorderWidth(0.0);
        input_bg.setFillColor(&NSColor::colorWithSRGBRed_green_blue_alpha(
            0.09, 0.09, 0.09, 0.85,
        ));
        input_bg.setTranslatesAutoresizingMaskIntoConstraints(false);
        root.addSubview(&input_bg);

        let input = NSTextField::textFieldWithString(ns_string!(""), mtm);
        input.setBezeled(false);
        input.setBordered(false);
        input.setDrawsBackground(false);
        input.setFocusRingType(NSFocusRingType::None);
        input.setTextColor(Some(&NSColor::tertiaryLabelColor()));
        input.setEnabled(false);
        input.setPlaceholderString(Some(ns_string!(
            "Ask Lisa anything, or type / to run commands"
        )));
        input.setFont(Some(&NSFont::systemFontOfSize(13.0)));
        input.setTranslatesAutoresizingMaskIntoConstraints(false);
        unsafe {
            input.setTarget(Some(self.target()));
            input.setAction(Some(sel!(sendMessage:)));
        }
        self.ivars().chat_input.set(input.clone()).ok();
        input_bg.addSubview(&input);

        let send = ui::button(mtm, "Send", Some(self.target()), Some(sel!(sendMessage:)), true);
        send.setControlSize(NSControlSize::Large);
        send.setFont(Some(&NSFont::systemFontOfSize(13.0)));
        send.setEnabled(false);
        send.setBordered(false);
        self.ivars().chat_send.set(send.clone()).ok();
        send.setTranslatesAutoresizingMaskIntoConstraints(false);
        send
            .widthAnchor()
            .constraintGreaterThanOrEqualToConstant(80.0)
            .setActive(true);
        send
            .heightAnchor()
            .constraintEqualToConstant(32.0)
            .setActive(true);

        let send_bg = RoundedView::new(
            mtm,
            &NSColor::colorWithSRGBRed_green_blue_alpha(0.24, 0.24, 0.24, 1.0),
            10.0,
        );
        send_bg.setTranslatesAutoresizingMaskIntoConstraints(false);
        root.addSubview(&send_bg);
        root.addSubview(&send);

        // Slash-command dropdown, parked above the input and revealed by
        // `update_command_menu` once the text starts with "/".
        let cmd_menu = RoundedView::new(
            mtm,
            &NSColor::colorWithSRGBRed_green_blue_alpha(0.20, 0.20, 0.20, 1.0),
            10.0,
        );
        cmd_menu.setTranslatesAutoresizingMaskIntoConstraints(false);
        cmd_menu.setHidden(true);
        root.addSubview(&cmd_menu);

        let cmd_stack = ui::stack(
            mtm,
            NSUserInterfaceLayoutOrientation::Vertical,
            0.0,
            NSLayoutAttribute::Leading,
        );
        cmd_stack.setEdgeInsets(NSEdgeInsets {
            top: 10.0,
            left: 12.0,
            bottom: 10.0,
            right: 12.0,
        });
        cmd_menu.addSubview(&cmd_stack);
        cmd_stack
            .leadingAnchor()
            .constraintEqualToAnchor(&cmd_menu.leadingAnchor())
            .setActive(true);
        cmd_stack
            .trailingAnchor()
            .constraintEqualToAnchor(&cmd_menu.trailingAnchor())
            .setActive(true);
        cmd_stack
            .topAnchor()
            .constraintEqualToAnchor(&cmd_menu.topAnchor())
            .setActive(true);
        cmd_stack
            .bottomAnchor()
            .constraintEqualToAnchor(&cmd_menu.bottomAnchor())
            .setActive(true);

        let mut cmd_rows: Vec<Retained<NSButton>> = Vec::new();
        let mut cmd_bgs: Vec<Retained<RoundedView>> = Vec::new();
        for (i, (name, desc)) in CHAT_COMMANDS.iter().enumerate() {
            let row = NSView::new(mtm);
            row.setTranslatesAutoresizingMaskIntoConstraints(false);

            let bg = RoundedView::new(
                mtm,
                &NSColor::controlAccentColor().colorWithAlphaComponent(0.35),
                6.0,
            );
            bg.setTranslatesAutoresizingMaskIntoConstraints(false);
            bg.setHidden(true);
            row.addSubview(&bg);

            let btn = ui::button(mtm, "", Some(self.target()), Some(sel!(runCommand:)), false);
            btn.setControlSize(NSControlSize::Regular);
            btn.setBordered(false);
            btn.setAlignment(NSTextAlignment::Left);
            btn.setAttributedTitle(&command_row_title(name, desc, false));
            btn.setTranslatesAutoresizingMaskIntoConstraints(false);
            btn.setTag(i as NSInteger);
            row.addSubview(&btn);

            for constraint in [
                bg.leadingAnchor().constraintEqualToAnchor(&row.leadingAnchor()),
                bg.trailingAnchor().constraintEqualToAnchor(&row.trailingAnchor()),
                bg.topAnchor().constraintEqualToAnchor(&row.topAnchor()),
                bg.bottomAnchor().constraintEqualToAnchor(&row.bottomAnchor()),
                btn.leadingAnchor().constraintEqualToAnchor_constant(&row.leadingAnchor(), 10.0),
                btn.trailingAnchor().constraintEqualToAnchor_constant(&row.trailingAnchor(), -10.0),
                btn.topAnchor().constraintEqualToAnchor(&row.topAnchor()),
                btn.bottomAnchor().constraintEqualToAnchor(&row.bottomAnchor()),
            ] {
                constraint.setActive(true);
            }

            cmd_stack.addArrangedSubview(&row);
            row.widthAnchor()
                .constraintEqualToAnchor_constant(&cmd_stack.widthAnchor(), -24.0)
                .setActive(true);
            row.heightAnchor()
                .constraintEqualToConstant(26.0)
                .setActive(true);
            cmd_rows.push(btn);
            cmd_bgs.push(bg);
        }
        *self.ivars().cmd_rows.borrow_mut() = cmd_rows;
        *self.ivars().cmd_bgs.borrow_mut() = cmd_bgs;
        self.ivars().cmd_menu.set(cmd_menu.clone()).ok();

        cmd_menu
            .leadingAnchor()
            .constraintEqualToAnchor(&input_bg.leadingAnchor())
            .setActive(true);
        cmd_menu
            .bottomAnchor()
            .constraintEqualToAnchor_constant(&input_bg.topAnchor(), -8.0)
            .setActive(true);
        cmd_menu
            .widthAnchor()
            .constraintEqualToConstant(300.0)
            .setActive(true);
        cmd_menu
            .widthAnchor()
            .constraintLessThanOrEqualToAnchor(&input_bg.widthAnchor())
            .setActive(true);

        unsafe { input.setDelegate(Some(ProtocolObject::from_ref(self))) };

        scroll
            .leadingAnchor()
            .constraintEqualToAnchor(&root.leadingAnchor())
            .setActive(true);
        scroll
            .trailingAnchor()
            .constraintEqualToAnchor(&root.trailingAnchor())
            .setActive(true);
        scroll
            .topAnchor()
            .constraintEqualToAnchor(&root.topAnchor())
            .setActive(true);
        scroll
            .bottomAnchor()
            .constraintEqualToAnchor(&root.bottomAnchor())
            .setActive(true);

        input_bg
            .leadingAnchor()
            .constraintEqualToAnchor_constant(&root.leadingAnchor(), 18.0)
            .setActive(true);
        input_bg
            .trailingAnchor()
            .constraintEqualToAnchor_constant(&send.leadingAnchor(), -10.0)
            .setActive(true);
        input_bg
            .bottomAnchor()
            .constraintEqualToAnchor_constant(&root.bottomAnchor(), -18.0)
            .setActive(true);
        input_bg
            .heightAnchor()
            .constraintEqualToAnchor(&send.heightAnchor())
            .setActive(true);

        input
            .leadingAnchor()
            .constraintEqualToAnchor_constant(&input_bg.leadingAnchor(), 14.0)
            .setActive(true);
        input
            .trailingAnchor()
            .constraintEqualToAnchor_constant(&input_bg.trailingAnchor(), -14.0)
            .setActive(true);
        input
            .centerYAnchor()
            .constraintEqualToAnchor(&input_bg.centerYAnchor())
            .setActive(true);

        send
            .trailingAnchor()
            .constraintEqualToAnchor_constant(&root.trailingAnchor(), -18.0)
            .setActive(true);
        send
            .bottomAnchor()
            .constraintEqualToAnchor(&input_bg.bottomAnchor())
            .setActive(true);

        send_bg
            .leadingAnchor()
            .constraintEqualToAnchor(&send.leadingAnchor())
            .setActive(true);
        send_bg
            .trailingAnchor()
            .constraintEqualToAnchor(&send.trailingAnchor())
            .setActive(true);
        send_bg
            .topAnchor()
            .constraintEqualToAnchor(&send.topAnchor())
            .setActive(true);
        send_bg
            .bottomAnchor()
            .constraintEqualToAnchor(&send.bottomAnchor())
            .setActive(true);

        root
    }

    fn set_serving(&self, on: bool, msg: &str) {
        self.ivars().serving.set(on);
        if let Some(switch) = self.ivars().serve_switch.get() {
            switch.setState(if on {
                NSControlStateValueOn
            } else {
                NSControlStateValueOff
            });
        }
        if let Some(state) = self.ivars().serve_state.get() {
            state.setStringValue(&NSString::from_str(if on {
                "Available"
            } else {
                "Unavailable"
            }));
        }
        if let Some(dot) = self.ivars().serve_dot.get() {
            let color = if on {
                NSColor::systemGreenColor()
            } else {
                NSColor::secondaryLabelColor()
            };
            dot.setContentTintColor(Some(&color));
        }
        if let Some(status) = self.ivars().serve_status.get() {
            status.setStringValue(&NSString::from_str(msg));
        }
        if let Some(port) = self.ivars().serve_port.get() {
            port.setEnabled(!on);
        }
        if !on {
            *self.ivars().serve_stop.borrow_mut() = None;
        }
        self.update_chat_enabled();
    }

    fn build_serve(&self, mtm: MainThreadMarker) -> Retained<NSView> {
        let root = NSView::new(mtm);

        let body = NSStackView::new(mtm);
        body.setOrientation(NSUserInterfaceLayoutOrientation::Vertical);
        body.setAlignment(NSLayoutAttribute::Leading);
        body.setSpacing(12.0);
        body.setTranslatesAutoresizingMaskIntoConstraints(false);
        root.addSubview(&body);

        let card = RoundedView::new(mtm, &NSColor::quaternaryLabelColor(), 10.0);
        card.setTranslatesAutoresizingMaskIntoConstraints(false);
        body.addArrangedSubview(&card);
        card.trailingAnchor()
            .constraintEqualToAnchor_constant(&root.trailingAnchor(), -18.0)
            .setActive(true);

        let inner = NSStackView::new(mtm);
        inner.setOrientation(NSUserInterfaceLayoutOrientation::Vertical);
        inner.setAlignment(NSLayoutAttribute::Leading);
        inner.setSpacing(10.0);
        inner.setTranslatesAutoresizingMaskIntoConstraints(false);
        card.addSubview(&inner);
        inner
            .leadingAnchor()
            .constraintEqualToAnchor_constant(&card.leadingAnchor(), 12.0)
            .setActive(true);
        inner
            .trailingAnchor()
            .constraintEqualToAnchor_constant(&card.trailingAnchor(), -12.0)
            .setActive(true);
        inner
            .topAnchor()
            .constraintEqualToAnchor_constant(&card.topAnchor(), 12.0)
            .setActive(true);
        inner
            .bottomAnchor()
            .constraintEqualToAnchor_constant(&card.bottomAnchor(), -12.0)
            .setActive(true);

        // Top row: icon badge + title/description + switch.
        let row1 = NSView::new(mtm);
        row1.setTranslatesAutoresizingMaskIntoConstraints(false);

        let icon = RoundedView::new(mtm, &NSColor::controlAccentColor(), 7.0);
        icon.setTranslatesAutoresizingMaskIntoConstraints(false);
        row1.addSubview(&icon);

        let glyph = NSImageView::new(mtm);
        if let Some(img) = NSImage::imageWithSystemSymbolName_accessibilityDescription(
            &NSString::from_str("network"),
            None,
        ) {
            glyph.setImage(Some(&img));
        }
        glyph.setContentTintColor(Some(&NSColor::whiteColor()));
        glyph.setTranslatesAutoresizingMaskIntoConstraints(false);
        icon.addSubview(&glyph);

        let card_title = ui::label(mtm, "Server", 13.0, true, &NSColor::labelColor());
        card_title.setTranslatesAutoresizingMaskIntoConstraints(false);
        row1.addSubview(&card_title);

        let card_desc = ui::wrapping_label(
            mtm,
            "Expose the loaded model as an OpenAI-compatible API.",
            11.0,
            &NSColor::secondaryLabelColor(),
        );
        card_desc.setTranslatesAutoresizingMaskIntoConstraints(false);
        card_desc.setMaximumNumberOfLines(0);
        row1.addSubview(&card_desc);

        let switch = NSSwitch::new(mtm);
        switch.setTranslatesAutoresizingMaskIntoConstraints(false);
        unsafe {
            switch.setTarget(Some(self.target()));
            switch.setAction(Some(sel!(toggleServe:)));
        }
        row1.addSubview(&switch);
        self.ivars().serve_switch.set(switch.clone()).ok();

        icon.leadingAnchor()
            .constraintEqualToAnchor(&row1.leadingAnchor())
            .setActive(true);
        icon.topAnchor()
            .constraintEqualToAnchor(&row1.topAnchor())
            .setActive(true);
        icon.widthAnchor().constraintEqualToConstant(28.0).setActive(true);
        icon.heightAnchor().constraintEqualToConstant(28.0).setActive(true);
        glyph
            .centerXAnchor()
            .constraintEqualToAnchor(&icon.centerXAnchor())
            .setActive(true);
        glyph
            .centerYAnchor()
            .constraintEqualToAnchor(&icon.centerYAnchor())
            .setActive(true);
        glyph.widthAnchor().constraintEqualToConstant(16.0).setActive(true);
        glyph.heightAnchor().constraintEqualToConstant(16.0).setActive(true);
        switch
            .trailingAnchor()
            .constraintEqualToAnchor(&row1.trailingAnchor())
            .setActive(true);
        switch
            .centerYAnchor()
            .constraintEqualToAnchor(&icon.centerYAnchor())
            .setActive(true);
        card_title
            .leadingAnchor()
            .constraintEqualToAnchor_constant(&icon.trailingAnchor(), 10.0)
            .setActive(true);
        card_title
            .topAnchor()
            .constraintEqualToAnchor(&row1.topAnchor())
            .setActive(true);
        card_title
            .trailingAnchor()
            .constraintLessThanOrEqualToAnchor_constant(&switch.leadingAnchor(), -12.0)
            .setActive(true);
        card_desc
            .leadingAnchor()
            .constraintEqualToAnchor(&card_title.leadingAnchor())
            .setActive(true);
        card_desc
            .topAnchor()
            .constraintEqualToAnchor_constant(&card_title.bottomAnchor(), 2.0)
            .setActive(true);
        card_desc
            .trailingAnchor()
            .constraintEqualToAnchor_constant(&switch.leadingAnchor(), -12.0)
            .setActive(true);
        card_desc
            .bottomAnchor()
            .constraintEqualToAnchor(&row1.bottomAnchor())
            .setActive(true);
        inner.addArrangedSubview(&row1);
        row1.widthAnchor()
            .constraintEqualToAnchor(&inner.widthAnchor())
            .setActive(true);

        // Hairline between the switch row and the endpoint status.
        let card_rule = LineView::new(mtm);
        ui::height(&card_rule, 1.0);
        inner.addArrangedSubview(&card_rule);
        card_rule
            .widthAnchor()
            .constraintEqualToAnchor(&inner.widthAnchor())
            .setActive(true);

        let endpoint = ui::label(mtm, "Endpoint", 12.0, true, &NSColor::labelColor());
        endpoint.setTranslatesAutoresizingMaskIntoConstraints(false);
        inner.addArrangedSubview(&endpoint);

        let status_row = NSView::new(mtm);
        status_row.setTranslatesAutoresizingMaskIntoConstraints(false);
        let dot = NSImageView::new(mtm);
        if let Some(img) = NSImage::imageWithSystemSymbolName_accessibilityDescription(
            &NSString::from_str("circle.fill"),
            None,
        ) {
            dot.setImage(Some(&img));
        }
        dot.setContentTintColor(Some(&NSColor::secondaryLabelColor()));
        dot.setTranslatesAutoresizingMaskIntoConstraints(false);
        status_row.addSubview(&dot);
        let state = ui::label(mtm, "Unavailable", 12.0, false, &NSColor::secondaryLabelColor());
        state.setTranslatesAutoresizingMaskIntoConstraints(false);
        status_row.addSubview(&state);
        self.ivars().serve_dot.set(dot.clone()).ok();
        self.ivars().serve_state.set(state.clone()).ok();
        dot.leadingAnchor()
            .constraintEqualToAnchor(&status_row.leadingAnchor())
            .setActive(true);
        dot.topAnchor()
            .constraintEqualToAnchor(&status_row.topAnchor())
            .setActive(true);
        dot.widthAnchor().constraintEqualToConstant(9.0).setActive(true);
        dot.heightAnchor().constraintEqualToConstant(9.0).setActive(true);
        state
            .leadingAnchor()
            .constraintEqualToAnchor_constant(&dot.trailingAnchor(), 6.0)
            .setActive(true);
        state
            .centerYAnchor()
            .constraintEqualToAnchor(&dot.centerYAnchor())
            .setActive(true);
        state
            .trailingAnchor()
            .constraintLessThanOrEqualToAnchor(&status_row.trailingAnchor())
            .setActive(true);
        state
            .bottomAnchor()
            .constraintEqualToAnchor(&status_row.bottomAnchor())
            .setActive(true);
        inner.addArrangedSubview(&status_row);
        status_row
            .widthAnchor()
            .constraintEqualToAnchor(&inner.widthAnchor())
            .setActive(true);

        let port_card = NSBox::new(mtm);
        port_card.setBoxType(NSBoxType::Custom);
        port_card.setBorderWidth(0.0);
        port_card.setCornerRadius(10.0);
        port_card.setFillColor(&NSColor::quaternaryLabelColor());
        port_card.setTranslatesAutoresizingMaskIntoConstraints(false);
        ui::height(&port_card, 40.0);
        body.addArrangedSubview(&port_card);
        port_card
            .trailingAnchor()
            .constraintEqualToAnchor_constant(&root.trailingAnchor(), -18.0)
            .setActive(true);

        let port_label = ui::label(mtm, "Port", 13.0, false, &NSColor::labelColor());
        port_label.setTranslatesAutoresizingMaskIntoConstraints(false);
        port_card.addSubview(&port_label);

        let port = NSTextField::textFieldWithString(&NSString::from_str("5472"), mtm);
        port.setTranslatesAutoresizingMaskIntoConstraints(false);
        port.setFont(Some(&NSFont::systemFontOfSize(13.0)));
        port.setAlignment(NSTextAlignment::Right);
        port.setBezeled(false);
        port.setDrawsBackground(false);
        port.setBordered(false);
        port.setFocusRingType(NSFocusRingType::None);
        port_card.addSubview(&port);
        self.ivars().serve_port.set(port.clone()).ok();

        port_label
            .leadingAnchor()
            .constraintEqualToAnchor_constant(&port_card.leadingAnchor(), 12.0)
            .setActive(true);
        port_label
            .centerYAnchor()
            .constraintEqualToAnchor(&port_card.centerYAnchor())
            .setActive(true);
        port
            .trailingAnchor()
            .constraintEqualToAnchor_constant(&port_card.trailingAnchor(), -12.0)
            .setActive(true);
        port
            .centerYAnchor()
            .constraintEqualToAnchor(&port_card.centerYAnchor())
            .setActive(true);
        port
            .leadingAnchor()
            .constraintGreaterThanOrEqualToAnchor_constant(&port_label.trailingAnchor(), 8.0)
            .setActive(true);
        port.widthAnchor().constraintEqualToConstant(120.0).setActive(true);

        let sys_card = RoundedView::new(mtm, &NSColor::quaternaryLabelColor(), 10.0);
        sys_card.setTranslatesAutoresizingMaskIntoConstraints(false);
        body.addArrangedSubview(&sys_card);
        sys_card
            .trailingAnchor()
            .constraintEqualToAnchor_constant(&root.trailingAnchor(), -18.0)
            .setActive(true);

        let sys_label = ui::label(mtm, "System", 12.0, true, &NSColor::labelColor());
        sys_label.setTranslatesAutoresizingMaskIntoConstraints(false);
        sys_card.addSubview(&sys_label);

        let sys_text = NSTextView::new(mtm);
        sys_text.setEditable(true);
        sys_text.setRichText(false);
        sys_text.setDrawsBackground(false);
        sys_text.setTextColor(Some(&NSColor::labelColor()));
        sys_text.setFont(Some(&NSFont::systemFontOfSize(12.0)));
        sys_text.setTextContainerInset(NSSize::new(0.0, 0.0));
        if let Some(tc) = unsafe { sys_text.textContainer() } {
            tc.setLineFragmentPadding(0.0);
        }
        sys_text.setVerticallyResizable(true);
        sys_text.setHorizontallyResizable(false);
        sys_text.setAutoresizingMask(NSAutoresizingMaskOptions::ViewWidthSizable);
        sys_text.setFrame(NSRect::new(NSPoint::new(0.0, 0.0), NSSize::new(220.0, 72.0)));
        sys_text.setString(&NSString::from_str(DEFAULT_SYSTEM_PROMPT));
        self.ivars().system_view.set(sys_text.clone()).ok();
        let sys_scroll = ui::scroll(mtm, &sys_text);
        sys_card.addSubview(&sys_scroll);

        sys_label
            .leadingAnchor()
            .constraintEqualToAnchor_constant(&sys_card.leadingAnchor(), 12.0)
            .setActive(true);
        sys_label
            .topAnchor()
            .constraintEqualToAnchor_constant(&sys_card.topAnchor(), 10.0)
            .setActive(true);
        sys_scroll
            .leadingAnchor()
            .constraintEqualToAnchor_constant(&sys_card.leadingAnchor(), 12.0)
            .setActive(true);
        sys_scroll
            .trailingAnchor()
            .constraintEqualToAnchor_constant(&sys_card.trailingAnchor(), -12.0)
            .setActive(true);
        sys_scroll
            .topAnchor()
            .constraintEqualToAnchor_constant(&sys_label.bottomAnchor(), 6.0)
            .setActive(true);
        sys_scroll
            .bottomAnchor()
            .constraintEqualToAnchor_constant(&sys_card.bottomAnchor(), -12.0)
            .setActive(true);
        ui::height(&sys_scroll, 72.0);


        body
            .leadingAnchor()
            .constraintEqualToAnchor_constant(&root.leadingAnchor(), 18.0)
            .setActive(true);
        body
            .trailingAnchor()
            .constraintEqualToAnchor_constant(&root.trailingAnchor(), -18.0)
            .setActive(true);
        body
            .topAnchor()
            .constraintEqualToAnchor_constant(&root.topAnchor(), 20.0)
            .setActive(true);

        root
    }

    fn build_logs(&self, mtm: MainThreadMarker) -> Retained<NSView> {
        let root = NSView::new(mtm);

        let header = NSView::new(mtm);
        header.setTranslatesAutoresizingMaskIntoConstraints(false);
        let title = ui::label(mtm, "Logs", 14.0, true, &NSColor::labelColor());
        title.setTranslatesAutoresizingMaskIntoConstraints(false);
        header.addSubview(&title);
        root.addSubview(&header);

        let header_rule = ui::rule(mtm);
        root.addSubview(&header_rule);

        let text = NSTextView::new(mtm);
        text.setEditable(false);
        text.setRichText(false);
        text.setDrawsBackground(false);
        text.setTextColor(Some(&NSColor::secondaryLabelColor()));
        if let Some(font) = NSFont::userFixedPitchFontOfSize(12.0) {
            text.setFont(Some(&font));
        }
        text.setTextContainerInset(NSSize::new(16.0, 14.0));
        text.setVerticallyResizable(true);
        text.setHorizontallyResizable(false);
        text.setAutoresizingMask(NSAutoresizingMaskOptions::ViewWidthSizable);
        text.setFrame(NSRect::new(NSPoint::new(0.0, 0.0), NSSize::new(420.0, 600.0)));
        text.setString(&NSString::from_str(""));
        self.ivars().engine_log.set(text.clone()).ok();

        let scroll = ui::scroll(mtm, &text);
        root.addSubview(&scroll);

        header
            .leadingAnchor()
            .constraintEqualToAnchor(&root.leadingAnchor())
            .setActive(true);
        header
            .trailingAnchor()
            .constraintEqualToAnchor(&root.trailingAnchor())
            .setActive(true);
        header
            .topAnchor()
            .constraintEqualToAnchor(&root.topAnchor())
            .setActive(true);
        ui::height(&header, 44.0);

        title
            .leadingAnchor()
            .constraintEqualToAnchor_constant(&header.leadingAnchor(), 18.0)
            .setActive(true);
        title
            .centerYAnchor()
            .constraintEqualToAnchor(&header.centerYAnchor())
            .setActive(true);

        header_rule
            .leadingAnchor()
            .constraintEqualToAnchor(&root.leadingAnchor())
            .setActive(true);
        header_rule
            .trailingAnchor()
            .constraintEqualToAnchor(&root.trailingAnchor())
            .setActive(true);
        header_rule
            .topAnchor()
            .constraintEqualToAnchor(&header.bottomAnchor())
            .setActive(true);

        scroll
            .leadingAnchor()
            .constraintEqualToAnchor(&root.leadingAnchor())
            .setActive(true);
        scroll
            .trailingAnchor()
            .constraintEqualToAnchor(&root.trailingAnchor())
            .setActive(true);
        scroll
            .topAnchor()
            .constraintEqualToAnchor(&header_rule.bottomAnchor())
            .setActive(true);
        scroll
            .bottomAnchor()
            .constraintEqualToAnchor(&root.bottomAnchor())
            .setActive(true);

        root
    }
}
