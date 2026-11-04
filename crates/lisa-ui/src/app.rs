use std::cell::{Cell, OnceCell, RefCell};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, mpsc};
use std::thread;

use objc2::rc::Retained;
use objc2::runtime::{AnyObject, Bool, ProtocolObject, Sel};
use objc2::{
    AnyThread, ClassType, DefinedClass, MainThreadMarker, MainThreadOnly, define_class, msg_send,
    sel,
};
use objc2_app_kit::*;
use objc2_foundation::*;

use crate::models::{
    Model, config_model_type, discover_models, display_name, fetch_remote_models, hf_blobs_size,
    model_ram_budget,
};
use crate::{catalog, chats, ui};

/// Slash commands offered in the chat input's "/" dropdown: `(name, help)`.
const CHAT_COMMANDS: &[(&str, &str)] = &[
    ("/clear", "Start a new chat"),
    ("/help", "Show available commands"),
];

/// Commands sent from the UI to the engine thread.
enum Command {
    Load(PathBuf),
    Download(String),
    /// Reset the conversation (drop the session state; keep the model loaded).
    Clear,
    Restore {
        turns: Vec<chats::Turn>,
        system: String,
    },
    Chat {
        text: String,
        system: String,
        images: Vec<PathBuf>,
    },
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
    UserImages(Vec<PathBuf>),
    AssistantImage(PathBuf),
    ImageMode(Option<u32>),
    AssistantDelta(String),
    /// Replace the streaming assistant text wholesale (used when hidden
    /// reasoning is stripped and the visible reply no longer extends the
    /// previously emitted prefix).
    AssistantReset(String),
    AssistantDone,
    /// The authoritative successful reply, safe to replay in a later session.
    TurnCommitted(String),
    /// A model finished loading (true) or failed (false).
    Loaded(bool),
    SupportsImages(bool),
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
fn command_row_title(
    name: &str,
    desc: &str,
    _selected: bool,
) -> Retained<NSMutableAttributedString> {
    let font_key = NSString::from_str("NSFont");
    let color_key = NSString::from_str("NSColor");
    let name_font = NSFont::boldSystemFontOfSize(13.0);
    let desc_font = NSFont::systemFontOfSize(12.0);
    let name_color = NSColor::labelColor();
    let desc_color = NSColor::secondaryLabelColor();

    let out = NSMutableAttributedString::initWithString(
        NSMutableAttributedString::alloc(),
        ns_string!(""),
    );
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
    let italic = NSFont::fontWithName_size(&NSString::from_str("TimesNewRomanPS-ItalicMT"), size)
        .unwrap_or_else(|| {
            NSFontManager::sharedFontManager(mtm).convertFont_toHaveTrait(
                &NSFont::systemFontOfSize(size),
                NSFontTraitMask::ItalicFontMask,
            )
        });
    let italic_small =
        NSFont::fontWithName_size(&NSString::from_str("TimesNewRomanPS-ItalicMT"), small)
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
    c.replace("**", "")
        .chars()
        .filter(|ch| *ch != '`')
        .collect()
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
                let rest = s
                    .strip_prefix("- ")
                    .or_else(|| s.strip_prefix("* "))
                    .unwrap();
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
                l.widthAnchor()
                    .constraintEqualToAnchor(&stack.widthAnchor())
                    .setActive(true);
            }
            Block::Paragraph(t) => {
                let l = paragraph_label(mtm, &t, color, MdStyle::Base);
                stack.addArrangedSubview(&l);
                l.widthAnchor()
                    .constraintEqualToAnchor(&stack.widthAnchor())
                    .setActive(true);
            }
            Block::Bullets(items) => {
                let v = bullets_view(mtm, &items, color);
                stack.addArrangedSubview(&v);
                v.widthAnchor()
                    .constraintEqualToAnchor(&stack.widthAnchor())
                    .setActive(true);
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
                l.widthAnchor()
                    .constraintEqualToAnchor(&stack.widthAnchor())
                    .setActive(true);
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
    let label = ui::wrapping_label(mtm, "", BUBBLE_FONT_SIZE, color);
    label.setTranslatesAutoresizingMaskIntoConstraints(false);
    label.setMaximumNumberOfLines(0);
    label.setPreferredMaxLayoutWidth(BUBBLE_MAX_WIDTH - 30.0);
    label.setAlignment(NSTextAlignment::Left);
    label.setFont(Some(&NSFont::systemFontOfSize(BUBBLE_FONT_SIZE)));
    label.setAttributedStringValue(&styled_attributed(
        mtm,
        text,
        color,
        base_style,
        BUBBLE_FONT_SIZE,
    ));
    ui::make_static(&label);
    label
}

fn code_label(mtm: MainThreadMarker, text: &str, color: &NSColor) -> Retained<NSTextField> {
    let label = ui::wrapping_label(mtm, text, BUBBLE_FONT_SIZE, color);
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

fn bullets_view(mtm: MainThreadMarker, items: &[String], color: &NSColor) -> Retained<NSStackView> {
    let stack = NSStackView::new(mtm);
    stack.setOrientation(NSUserInterfaceLayoutOrientation::Vertical);
    stack.setAlignment(NSLayoutAttribute::Leading);
    stack.setSpacing(6.0);
    stack.setTranslatesAutoresizingMaskIntoConstraints(false);
    for item in items {
        let label = paragraph_label(mtm, &format!("\u{2022}  {item}"), color, MdStyle::Base);
        stack.addArrangedSubview(&label);
        label
            .widthAnchor()
            .constraintEqualToAnchor(&stack.widthAnchor())
            .setActive(true);
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
/// The unstripped decode — what a reasoning-only reply still shows.
fn raw_display(ids: &[u32], tok: &lisa_engine::core::tokenizer::Tokenizer) -> String {
    tok.decode_clean(ids).unwrap_or_default().trim().to_string()
}

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
const BUBBLE_MAX_WIDTH: f64 = 720.0;
const BUBBLE_FONT_SIZE: f64 = 14.0;
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

fn chat_input_ids(
    tokenizer: &lisa_engine::core::tokenizer::Tokenizer,
    text: &str,
    system: Option<&str>,
    first: bool,
    is_glm: bool,
) -> anyhow::Result<Vec<u32>> {
    use lisa_engine::core::tokenizer::{self, GlmReasoning};
    if let Some(ids) = tokenizer.native_chat_turn(system, text, first)? {
        return Ok(ids);
    }
    let prompt = if is_glm {
        if first {
            tokenizer::glm_generation_prompt(text, system, GlmReasoning::Disabled)
        } else {
            format!("<|user|>{text}<|assistant|><think></think>")
        }
    } else if first {
        tokenizer::chat_prompt_special_with_system(text, system, false)
    } else {
        // The previous response stops before EOS. Close it and use the same
        // Qwen markers as the first turn and the checkpoint's chat template.
        format!(
            "<|im_end|>\n{}",
            tokenizer::chat_prompt_special(text, false)
        )
    };
    if is_glm {
        tokenizer.encode_special(&prompt)
    } else {
        tokenizer.encode(&prompt, false)
    }
}

fn new_chat_session(
    model: &mut dyn lisa_engine::models::LanguageModel,
) -> lisa_engine::core::session::Session {
    use lisa_engine::core::session::{Session, TargetMtpEosPolicy};
    model.drafter_reset();
    let mut session = Session::new(model);
    // Chat turn suffixes start after assistant text, without a committed EOS.
    session.set_target_mtp_eos_policy(TargetMtpEosPolicy::StopBeforeVerified);
    session
}

fn replay_chat_history(
    model: &mut dyn lisa_engine::models::LanguageModel,
    tokenizer: &lisa_engine::core::tokenizer::Tokenizer,
    history: &[(String, String)],
    system: &str,
    is_glm: bool,
) -> anyhow::Result<lisa_engine::core::session::Session> {
    let mut session = new_chat_session(model);
    let mut feed = |ids: &[u32]| -> anyhow::Result<()> {
        if ids.is_empty() {
            return Ok(());
        }
        if model.has_drafter() {
            let (_, multi) = session.feed_multi(model, ids)?;
            session.mtp_tokens.extend_from_slice(ids);
            session.mtp_multi.push(multi);
        } else {
            session.feed(model, ids)?;
        }
        Ok(())
    };
    for (index, (user, reply)) in history.iter().enumerate() {
        let ids = chat_input_ids(
            tokenizer,
            user,
            Some(system).filter(|s| !s.trim().is_empty()),
            index == 0,
            is_glm,
        )?;
        feed(&ids)?;
        feed(&tokenizer.encode_plain(reply)?)?;
    }
    Ok(session)
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum EngineOwner {
    Chat,
    Server,
}

/// The target's rolling context and Qwen draft head live on the shared model.
/// Keep each caller's target context and re-prime its own draft history when
/// ownership changes; consecutive HTTP requests retain their warmed head.
struct EngineContexts {
    owner: EngineOwner,
    chat: Vec<Vec<i64>>,
    server: Vec<Vec<i64>>,
}

impl EngineContexts {
    fn new() -> Self {
        Self {
            owner: EngineOwner::Chat,
            chat: Vec::new(),
            server: Vec::new(),
        }
    }

    fn activate(&mut self, owner: EngineOwner, model: &mut dyn lisa_engine::models::LanguageModel) {
        if self.owner == owner {
            return;
        }
        let previous = model.context_tails();
        let next = match owner {
            EngineOwner::Chat => {
                self.server = previous;
                self.chat.clone()
            }
            EngineOwner::Server => {
                self.chat = previous;
                self.server.clone()
            }
        };
        model.set_context_tails(next);
        model.drafter_reset();
        self.owner = owner;
    }
}

/// Owns the loaded model on a dedicated thread (the model is not `Send`) and
/// funnels log/chat events back to the main thread through `events`.
fn spawn_engine(rx: mpsc::Receiver<Command>, events: Arc<Mutex<Vec<UiEvent>>>) {
    let _ = thread::Builder::new()
        .name("lisa-engine".into())
        .spawn(move || {
            let push = |e: UiEvent| events.lock().unwrap().push(e);
            let mut model: Option<Box<dyn lisa_engine::models::LanguageModel>> = None;
            let mut tokenizer: Option<lisa_engine::core::tokenizer::Tokenizer> = None;
            let mut sampler = lisa_engine::core::sampler::Sampler::default();
            let mut model_dir: Option<PathBuf> = None;
            // Multi-turn conversation state; kept in lockstep with the loaded model.
            let mut session: Option<lisa_engine::core::session::Session> = None;
            let mut turn = 0usize;
            // The conversation as the engine saw it — user turns, assistant
            // replies, and the first-turn system — kept so a model swap can
            // replay the history into the new model's session and the chat
            // keeps its context across the switch.
            let mut history: Vec<(String, String)> = Vec::new();
            let mut history_images: Vec<Vec<String>> = Vec::new();
            let mut image_model: Option<Box<lisa_engine::models::image_generation::Pipeline>>=None;
            let mut image_server: Option<lisa_serve::images::ImageServer>=None;
            let mut history_system = String::new();
            let mut history_error: Option<String> = None;
            let mut is_glm = false;
            let mut contexts = EngineContexts::new();

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
                                if let Some(mut srv) = server.take() {srv.shutdown();}
                                if let Some(mut srv) = image_server.take() {srv.shutdown();}
                                serve_stop = None;
                                push(UiEvent::ServeStopped);
                                continue;
                            }
                        }
                        if let (Some(srv), Some(m), Some(t)) =
                            (server.as_mut(), model.as_mut(), tokenizer.as_ref())
                        {
                            if srv.step_with_context(&mut **m, t, |model| {
                                contexts.activate(EngineOwner::Server, model);
                            }) {
                                continue;
                            }
                        }
                        if let (Some(server),Some(model))=(&image_server,&image_model) {
                            if server.step(model) {continue;}
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
                        let path = lisa_engine::models::hf::checkpoint_path(path);
                        if let Some(mut srv)=image_server.take() {srv.shutdown();serve_stop=None;push(UiEvent::ServeStopped);}
                        image_model=None;
                        push(UiEvent::ImageMode(None));
                        // Swapping the model requires stopping any running server.
                        if let Some(mut srv) = server.take() {
                            srv.shutdown();
                            serve_stop = None;
                            push(UiEvent::ServeStopped);
                        }
                        session = None;
                        turn = 0;
                        model = None;
                        tokenizer = None;
                        model_dir = None;
                        contexts = EngineContexts::new();
                        push(UiEvent::Log(format!(
                            "loading {} …",
                            tilde(&path.to_string_lossy())
                        )));
                        if matches!(config_model_type(&path).as_deref(), Some("glm5_next" | "glm5_next_text"))
                            && !path.join(".lisa-resident/index.json").is_file()
                        {
                            push(UiEvent::Log("Preparing GLM weights for the first load. This uses extra disk space and may take several minutes.".into()));
                            if let Err(e) = lisa_engine::core::prepare::prepare_model(&path, false) {
                                push(UiEvent::Log(format!("preparation error: {e}")));
                                push(UiEvent::Loaded(false));
                                continue;
                            }
                        }
                        match lisa_engine::models::load_dir(&path) {
                            Ok(lisa_engine::models::Loaded::Language(m)) => {
                                match lisa_engine::core::tokenizer::Tokenizer::load(&path) {
                                    Ok(t) => {
                                        lisa_engine::core::generate::set_eos_ids_for_model(
                                            t.im_end_ids.clone(),
                                            m.as_ref(),
                                        );
                                        tokenizer = Some(t);
                                        is_glm = matches!(
                                            config_model_type(&path).as_deref(),
                                            Some("glm5_next" | "glm5_next_text" | "glm5-next")
                                        );
                                        model = Some(m);
                                        model_dir = Some(path.clone());
                                        if let Some(m) = model.as_mut().filter(|_| history_images.iter().all(Vec::is_empty)) {
                                            // Prefill recorded turns without sampling. Qwen
                                            // also retains the hidden rows needed to prime MTP.
                                            match replay_chat_history(
                                                &mut **m,
                                                tokenizer.as_ref().expect("tokenizer"),
                                                &history,
                                                &history_system,
                                                is_glm,
                                            ) {
                                                Ok(replayed) => {
                                                    session = Some(replayed);
                                                    turn = history.len();
                                                    if turn > 0 {
                                                        push(UiEvent::Log(format!(
                                                            "history replayed ({turn} turns)"
                                                        )));
                                                    }
                                                }
                                                Err(error) => {
                                                    push(UiEvent::Log(format!(
                                                        "history replay failed: {error}"
                                                    )));
                                                    session = None;
                                                    turn = history.len();
                                                }
                                            }
                                        }
                                        push(UiEvent::Log("model loaded".into()));
                                        push(UiEvent::SupportsImages(model.as_ref().is_some_and(|m| m.supports_images())));
                                        push(UiEvent::Loaded(true));
                                    }
                                    Err(e) => {
                                        push(UiEvent::Log(format!("tokenizer error: {e}")));
                                        push(UiEvent::Loaded(false));
                                    }
                                }
                            }
                            Ok(lisa_engine::models::Loaded::Image(pipeline)) => {
                                let steps=pipeline.default_steps();
                                image_model=Some(pipeline);model_dir=Some(path.clone());
                                push(UiEvent::ImageMode(Some(steps)));push(UiEvent::Loaded(true));
                                push(UiEvent::Log("Image model ready: enter a prompt to generate a 512×512 image.".into()));
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
                        if catalog::find(&repo).is_none() {
                            push(UiEvent::Log(format!("{repo} is not in the curated download catalog")));
                            push(UiEvent::ModelsChanged);
                            continue;
                        }
                        // Download on a side thread so the active model keeps
                        // serving chat, and don't swap models when it finishes.
                        push(UiEvent::Log(format!("downloading {repo} …")));
                        let events = events.clone();
                        let repo2 = repo.clone();
                        let _ =
                            thread::Builder::new()
                                .name("lisa-download".into())
                                .spawn(move || {
                                    let push = |e: UiEvent| events.lock().unwrap().push(e);
                                    match lisa_engine::models::hf::download_repo(&repo2) {
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
                        history.clear();
                        history_images.clear();
                        history_system.clear();
                        history_error = None;
                        session = None;
                        turn = 0;
                        if let Some(m) = model.as_mut() {
                            contexts.activate(EngineOwner::Chat, &mut **m);
                            session = Some(new_chat_session(&mut **m));
                            turn = 0;
                        }
                        push(UiEvent::Log("conversation cleared".into()));
                    }
                    Command::Restore { turns, system } => {
                        session = None;
                        history.clear();
                        history_images.clear();
                        history_system = system;
                        history_error = None;
                        for saved in turns.into_iter().filter(|t| t.complete && t.generated_image.is_none()) {
                            let images = saved.images.iter()
                                .map(|path| lisa_engine::models::vision::file_data_url(path))
                                .collect::<anyhow::Result<Vec<_>>>();
                            match images {
                                Ok(images) => {
                                    history.push((saved.user, saved.assistant));
                                    history_images.push(images);
                                }
                                Err(error) => {
                                    history_error = Some(format!("A saved attachment could not be read: {error}. Restore the file or start a new chat."));
                                    break;
                                }
                            }
                        }
                        turn = history.len();
                        if let Some(m) = model.as_mut() {
                            contexts.activate(EngineOwner::Chat, &mut **m);
                            m.clear_images();
                            m.drafter_reset();
                        }
                    }
                    Command::Chat { text, system, images } => {
                        push(UiEvent::UserMessage(text.clone()));
                        if let Some(error) = &history_error {
                            push(UiEvent::AssistantReset(error.clone()));
                            push(UiEvent::AssistantDone);
                            continue;
                        }
                        if let Some(pipeline)=&image_model {
                            let result=(||->anyhow::Result<PathBuf> {
                                anyhow::ensure!(images.is_empty(),"this image model currently supports text-to-image; remove attached images");
                                let stamp=std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)?.as_nanos();
                                let options=lisa_engine::models::image_generation::Options {seed:stamp as u64,..Default::default()};
                                let png=pipeline.generate(&text,&options,&mut |stage,i,n| {push(UiEvent::AssistantReset(format!("{stage} {i}/{n}")));Ok(())})?;
                                let root=PathBuf::from(std::env::var_os("HOME").ok_or_else(||anyhow::anyhow!("HOME is not set"))?).join("Pictures/Lisa");
                                std::fs::create_dir_all(&root)?;
                                let path=root.join(format!("image-{stamp}.png"));std::fs::write(&path,png)?;Ok(path)
                            })();
                            match result {
                                Ok(path)=> {push(UiEvent::Log(format!("Saved image: {}",path.display())));push(UiEvent::AssistantImage(path));push(UiEvent::TurnCommitted(String::new()));}
                                Err(e)=>push(UiEvent::AssistantReset(format!("Image generation failed: {e}"))),
                            }
                            push(UiEvent::AssistantDone);continue;
                        }
                        let (Some(model), Some(tokenizer)) = (model.as_mut(), tokenizer.as_ref())
                        else {
                            push(UiEvent::Log("no model loaded".into()));
                            push(UiEvent::AssistantReset("Choose a model before sending a message.".into()));
                            push(UiEvent::AssistantDone);
                            continue;
                        };
                        contexts.activate(EngineOwner::Chat, &mut **model);
                        if !images.is_empty() || history_images.iter().any(|i|!i.is_empty()) {
                            let result=(||->anyhow::Result<(String,Vec<String>)> {
                                let urls=images.iter().map(|p|lisa_engine::models::vision::file_data_url(p)).collect::<anyhow::Result<Vec<_>>>()?;
                                push(UiEvent::UserImages(images.clone()));
                                let mut messages=Vec::new();
                                let sys=if history.is_empty() {&system} else {&history_system};
                                if !sys.trim().is_empty() {messages.push(serde_json::json!({"role":"system","content":sys}));}
                                let user=|text:&str,urls:&[String]| {
                                    let mut parts=urls.iter().map(|url|serde_json::json!({"type":"image_url","image_url":{"url":url}})).collect::<Vec<_>>();
                                    parts.push(serde_json::json!({"type":"text","text":text}));
                                    serde_json::json!({"role":"user","content":parts})
                                };
                                for (i,(text,reply)) in history.iter().enumerate() {
                                    messages.push(user(text,history_images.get(i).map(Vec::as_slice).unwrap_or(&[])));
                                    messages.push(serde_json::json!({"role":"assistant","content":reply}));
                                }
                                messages.push(user(&text,&urls));
                                let body=serde_json::json!({"messages":messages});
                                let decoded=lisa_engine::models::vision::decode_message_images(&body)?;
                                let dir=model_dir.as_ref().ok_or_else(||anyhow::anyhow!("no model directory"))?;
                                let template=lisa_serve::ChatTemplate::new(&std::fs::read_to_string(dir.join("chat_template.jinja"))?)?;
                                let chat:Vec<lisa_serve::ChatMessage>=body["messages"].as_array().unwrap().iter().map(|m|lisa_serve::ChatMessage {
                                    role:m["role"].as_str().unwrap().into(),content:m["content"].as_str().unwrap_or("").into(),
                                    reasoning_content:None,tool_calls:Vec::new(),tool_call_id:None,parts:m["content"].as_array().cloned()
                                }).collect();
                                let prompt=template.render(&chat,true,&[],false)?;
                                let mut ids=tokenizer.encode(&prompt,false)?;
                                model.clear_images();
                                model.prepare_images(&mut ids,&decoded)?;
                                drop(decoded);
                                let mut sess=new_chat_session(&mut **model);
                                let mut decoder=lisa_engine::core::tokenizer::IncrementalDecoder::new(tokenizer,true);
                                let output=sess.generate(&mut **model,&ids,CHAT_MAX_TOKENS,&mut sampler,0,None,&mut |id| {
                                    decoder.push(id)?;
                                    push(UiEvent::AssistantReset(clean_reply(decoder.stable_text())));
                                    Ok(())
                                })?;
                                let reply=clean_reply(&tokenizer.decode_clean(&output)?);
                                push(UiEvent::AssistantReset(reply.clone()));
                                Ok((reply,urls))
                            })();
                            model.clear_images();session=None;
                            match result {
                                Ok((reply,urls))=> {
                                    if history.is_empty() {history_system=system;}
                                    history_images.resize(history.len(),Vec::new());
                                    push(UiEvent::TurnCommitted(reply.clone()));
                                    history_images.push(urls);history.push((text,reply));turn=history.len();
                                }
                                Err(e)=> {
                                    push(UiEvent::Log(format!("image request failed: {e}")));
                                    push(UiEvent::AssistantReset(format!("Image request failed: {e}")));
                                }
                            }
                            push(UiEvent::AssistantDone);
                            continue;
                        }
                        if session.as_ref().is_none_or(|session| !session.is_valid()) {
                            match replay_chat_history(
                                &mut **model,
                                tokenizer,
                                &history,
                                &history_system,
                                is_glm,
                            ) {
                                Ok(replayed) => {
                                    session = Some(replayed);
                                    turn = history.len();
                                }
                                Err(error) => {
                                    push(UiEvent::Log(format!("history replay failed: {error}")));
                                    push(UiEvent::AssistantReset(format!("Could not restore this conversation: {error}")));
                                    push(UiEvent::AssistantDone);
                                    continue;
                                }
                            }
                        }
                        if turn == 0 {
                            history.clear();
                            history_system = system.clone();
                        }
                        let sys = if system.trim().is_empty() {
                            None
                        } else {
                            Some(system.as_str())
                        };
                        let ids = match chat_input_ids(
                            tokenizer,
                            &text,
                            sys,
                            turn == 0,
                            is_glm,
                        ) {
                            Ok(ids) => ids,
                            Err(e) => {
                                push(UiEvent::Log(format!("encode error: {e}")));
                                push(UiEvent::AssistantReset(format!("Could not prepare this message: {e}")));
                                push(UiEvent::AssistantDone);
                                continue;
                            }
                        };
                        let mut all = Vec::new();
                        let mut emitted = String::new();
                        let sess = session.as_mut().expect("session");
                        let mut decoded =
                            lisa_engine::core::tokenizer::IncrementalDecoder::new(tokenizer, true);
                        let mut emit = |id| {
                            all.push(id);
                            decoded.push(id)?;
                            let display = clean_reply(decoded.stable_text());
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
                        };
                        let target_mtp = model.has_target_mtp();
                        let use_mtp = target_mtp || (model.has_drafter() && sampler.greedy());
                        let result = if use_mtp {
                            let mtp_sampler = if target_mtp { Some(&mut sampler) } else { None };
                            sess.generate_mtp_depth(
                                &mut **model,
                                &ids,
                                CHAT_MAX_TOKENS,
                                lisa_engine::core::session::MtpDepth::Auto,
                                None,
                                mtp_sampler,
                                &mut emit,
                            )
                        } else {
                            sess.generate(
                                &mut **model,
                                &ids,
                                CHAT_MAX_TOKENS,
                                &mut sampler,
                                0,
                                None,
                                &mut emit,
                            )
                        };
                        if let Err(e) = result {
                            // Rebuild from completed turns before another request;
                            // a failed MTP transaction may have invalidated caches.
                            session = None;
                            turn = history.len();
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
                                // Reasoning checkpoints spend their whole
                                // budget inside ` presence think` and emit
                                // nothing after it. The bubble still shows the
                                // thinking: re-decode without the strip.
                                let raw = tokenizer.decode_clean(&all).unwrap_or_default();
                                let raw = raw.trim().to_string();
                                if raw.is_empty() {
                                    push(UiEvent::AssistantDelta("(no response)".to_string()));
                                } else {
                                    push(UiEvent::AssistantReset(raw));
                                }
                            }
                            // Record the turn so a later model swap replays it.
                            let reply = if emitted.is_empty() {
                                raw_display(&all, tokenizer)
                            } else {
                                emitted.clone()
                            };
                            history_images.push(Vec::new());
                            push(UiEvent::TurnCommitted(reply.clone()));
                            history.push((text.clone(), reply));
                            turn += 1;
                        }
                        push(UiEvent::AssistantDone);
                    }
                    Command::StartServe { addr, stop } => {
                        if image_model.is_some() {
                            let id=model_dir.as_ref().map(|p|lisa_serve::model_id_of(p)).unwrap_or_else(||"flux2-klein".into());
                            match lisa_serve::images::ImageServer::bind(&addr,&id,stop.clone()) {
                                Ok(srv)=>{image_server=Some(srv);serve_stop=Some(stop);push(UiEvent::ServeStarted(format!("http://{addr}")));}
                                Err(e)=>{push(UiEvent::Log(format!("serve failed: {e}")));push(UiEvent::ServeStopped);}
                            }
                            continue;
                        }
                        if model.is_none() || tokenizer.is_none() {
                            push(UiEvent::Log("serve: no model loaded".into()));
                            push(UiEvent::ServeStopped);
                            continue;
                        }
                        let chat_template = model_dir.as_ref().and_then(|d| {
                            std::fs::read_to_string(d.join("chat_template.jinja")).ok()
                        });
                        let cfg = lisa_serve::ServerConfig {
                            supports_images: model.as_ref().is_some_and(|m|m.supports_images()),
                            addr: addr.clone(),
                            max_tokens: 2048,
                            temperature: Some(0.0),
                            top_p: Some(1.0),
                            top_k: Some(0),
                            min_p: Some(0.0),
                            rep_penalty: Some(1.0),
                            mtp_exact_sampling: false,
                            gen_defaults: model_dir
                                .as_deref()
                                .map(lisa_serve::GenDefaults::from_model_dir)
                                .unwrap_or_default(),
                            depth: if model
                                .as_ref()
                                .is_some_and(|model| model.has_drafter() || model.has_target_mtp())
                            {
                                lisa_engine::core::round_cost::AUTO_DEPTH
                            } else {
                                0
                            },
                            pld: 0,
                            prefix_cache: 2,
                            max_batch: 4,
                            chat_template,
                            model_id: model_dir
                                .as_deref()
                                .map(lisa_serve::model_id_of)
                                .unwrap_or_else(|| "unknown".to_string()),
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
    models_list: OnceCell<Retained<NSStackView>>,
    models_scroll: OnceCell<Retained<NSScrollView>>,
    loading_row: RefCell<Option<usize>>,
    /// Repo id of the model currently loaded/active. Tracked by id (not row
    /// index) so list re-sorting never moves the active highlight.
    active_repo: RefCell<Option<String>>,
    downloading: Cell<bool>,
    download_progress: Cell<f64>,
    progress_tick: Cell<usize>,
    selected_row: RefCell<Option<usize>>,
    chat_stack: OnceCell<Retained<NSStackView>>,
    chat_scroll: OnceCell<Retained<NSScrollView>>,
    chat_placeholder: OnceCell<Retained<NSStackView>>,
    welcome_title: OnceCell<Retained<NSTextField>>,
    welcome_detail: OnceCell<Retained<NSTextField>>,
    model_title: OnceCell<Retained<NSTextField>>,
    model_status: OnceCell<Retained<NSTextField>>,
    composer_hint: OnceCell<Retained<NSTextField>>,
    new_chat: OnceCell<Retained<NSButton>>,
    new_chat_row: OnceCell<Retained<ui::SidebarRow>>,
    chat_store: chats::Store,
    chats: RefCell<Vec<chats::Chat>>,
    active_chat: RefCell<Option<String>>,
    chat_list: OnceCell<Retained<NSStackView>>,
    chat_list_height: OnceCell<Retained<NSLayoutConstraint>>,
    chat_section: OnceCell<ui::SidebarSection>,
    models_section: OnceCell<ui::SidebarSection>,
    chat_buttons: RefCell<Vec<Retained<NSButton>>>,
    busy: Cell<bool>,
    assistant_label: RefCell<Option<Retained<NSTextField>>>,
    assistant_stack: RefCell<Option<Retained<NSStackView>>>,
    user_stack: RefCell<Option<Retained<NSStackView>>>,
    typing: Cell<bool>,
    typing_tick: Cell<usize>,
    chat_input: OnceCell<Retained<NSTextField>>,
    chat_send: OnceCell<Retained<NSButton>>,
    chat_attach: OnceCell<Retained<NSButton>>,
    attachment_clear: OnceCell<Retained<NSButton>>,
    attachments: RefCell<Vec<PathBuf>>,
    image_mode: Cell<Option<u32>>,
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
    supports_images: Cell<bool>,
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
            text_view: &NSTextView,
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
                if NSApplication::sharedApplication(self.mtm()).currentEvent()
                    .is_some_and(|event| event.modifierFlags().contains(NSEventModifierFlags::Shift))
                {
                    unsafe { text_view.insertNewlineIgnoringFieldEditor(None) };
                    return Bool::YES;
                }
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
        #[unsafe(method(applicationWillTerminate:))]
        fn will_terminate(&self, _notification: &NSNotification) {
            self.save_chat_draft();
        }

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
            sidebar_item.setMinimumThickness(240.0);
            sidebar_item.setCanCollapse(true);
            sidebar_item.setCanCollapseFromWindowResize(true);
            sidebar_item.setMaximumThickness(300.0);
            sidebar_item.setPreferredThicknessFraction(0.09);
            // Keep the chosen sidebar width when Settings changes the available space.
            sidebar_item.setHoldingPriority(260.0);
            outer.addSplitViewItem(&sidebar_item);
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
            content_item.setHoldingPriority(250.0);
            outer.addSplitViewItem(&content_item);

            // Right column: serve the loaded model over HTTP for other tools.
            let serve_view = self.build_serve(mtm);
            let serve_vc = NSViewController::new(mtm);
            serve_vc.setView(&serve_view);
            let serve_item = NSSplitViewItem::splitViewItemWithViewController(&serve_vc);
            serve_item.setMinimumThickness(320.0);
            serve_item.setMaximumThickness(320.0);
            serve_item.setCanCollapse(true);
            // Resizing must preserve the user's choice to show or hide Settings.
            serve_item.setCanCollapseFromWindowResize(false);
            serve_item.setCollapsed(true);
            outer.addSplitViewItem(&serve_item);
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
            window.setContentSize(NSSize::new(1180.0, 820.0));
            window.setContentMinSize(NSSize::new(720.0, 560.0));
            window.setDelegate(Some(ProtocolObject::from_ref(self)));
            window.makeKeyAndOrderFront(None);
            window.setFrameOrigin(NSPoint::new(90.0, 150.0));

            // Chat is the primary workspace; the inspector opens on demand.
            if let Some(content) = window.contentView() {
                content.layoutSubtreeIfNeeded();
            }
            outer.splitView().setPosition_ofDividerAtIndex(260.0, 0);

            self.ivars().window.set(window).ok();
            self.refresh_model_list();
            self.refresh_chat_list();

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

    impl AppDelegate {
        #[unsafe(method(selectModel:))]
        fn select_model(&self, sender: Option<&AnyObject>) {
            if self.chat_locked() {
                return;
            }
            let Some(button) = sender.and_then(|sender| sender.downcast_ref::<NSButton>()) else {
                return;
            };
            let row = button.tag();
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
                self.load_chat_model(repo_id, PathBuf::from(path), row as usize);
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
                self.refresh_model_list();
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

        #[unsafe(method(toggleChatSection:))]
        fn toggle_chat_section(&self, _sender: Option<&AnyObject>) {
            if let Some(section) = self.ivars().chat_section.get() {
                section.toggle();
            }
        }

        #[unsafe(method(toggleModelsSection:))]
        fn toggle_models_section(&self, _sender: Option<&AnyObject>) {
            if let Some(section) = self.ivars().models_section.get() {
                section.toggle();
            }
        }

        #[unsafe(method(newChat:))]
        fn new_chat(&self, _sender: Option<&AnyObject>) {
            if !self.chat_locked() {
                self.run_chat_command("/clear");
                self.update_chat_enabled();
            }
        }

        #[unsafe(method(selectChat:))]
        fn select_chat(&self, sender: &NSButton) {
            if self.chat_locked() { return; }
            let chat = self.ivars().chats.borrow().get(sender.tag() as usize).cloned();
            if let Some(chat) = chat { self.open_chat(chat); }
        }

        #[unsafe(method(deleteChat:))]
        fn delete_chat(&self, sender: &NSButton) {
            self.confirm_delete_chat(sender.tag());
        }

        #[unsafe(method(deleteChatFromMenu:))]
        fn delete_chat_from_menu(&self, sender: &NSMenuItem) {
            self.confirm_delete_chat(sender.tag());
        }

        #[unsafe(method(validateMenuItem:))]
        fn validate_menu_item(&self, item: &NSMenuItem) -> bool {
            if item.action() == Some(sel!(deleteChatFromMenu:)) {
                !self.chat_locked() && self.ivars().chats.borrow().get(item.tag() as usize).is_some()
            } else {
                true
            }
        }

        #[unsafe(method(copyEndpoint:))]
        fn copy_endpoint(&self, _sender: Option<&AnyObject>) {
            let port = self.ivars().serve_port.get().map(|field|field.stringValue().to_string()).unwrap_or_else(||"5472".into());
            if let Ok(port) = port.trim().parse::<u16>() {
                let pasteboard = NSPasteboard::generalPasteboard();
                pasteboard.clearContents();
                unsafe { pasteboard.setString_forType(&NSString::from_str(&format!("http://127.0.0.1:{port}/v1")), NSPasteboardTypeString); }
            }
        }

        #[unsafe(method(toggleInspector:))]
        fn toggle_inspector(&self, _sender: Option<&AnyObject>) {
            if let Some(item) = self.ivars().serve_item.get() {
                let collapse = !item.isCollapsed();
                if !collapse {
                    if let Some(window) = self.ivars().window.get() {
                        let frame = window.frame();
                        if frame.size.width < 1120.0 {
                            window.setContentSize(NSSize::new(1120.0, frame.size.height));
                        }
                    }
                }
                item.setCollapsed(collapse);
            }
        }

        #[unsafe(method(deleteModel:))]
        fn delete_model(&self, sender: Option<&AnyObject>) {
            let Some(item) = sender.and_then(|sender| sender.downcast_ref::<NSMenuItem>()) else {
                return;
            };
            let row = item.tag();
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
            self.refresh_sidebar_hover();
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
                    UiEvent::ImageMode(steps)=> {
                        self.ivars().image_mode.set(steps);
                        self.ivars().supports_images.set(false);
                        let on=steps.is_some();
                        if let Some(input)=self.ivars().chat_input.get() {
                            input.setPlaceholderString(Some(if on {ns_string!("Describe an image to generate")} else {ns_string!("Message Lisa…")}));
                        }
                        self.ivars().attachments.borrow_mut().clear();
                        if let Some(button)=self.ivars().chat_send.get() {button.setTitle(if on {ns_string!("Generate")} else {ns_string!("Send")});}
                        if let Some(button)=self.ivars().chat_attach.get() {button.setTitle(ns_string!("Attach"));button.setEnabled(false);}
                        self.refresh_chat_chrome();
                    }
                    UiEvent::SupportsImages(supported) => {
                        self.ivars().supports_images.set(supported);
                        self.update_chat_enabled();
                    }
                    UiEvent::AssistantImage(path)=> {
                        self.update_saved_turn(|turn| {turn.assistant.clear();turn.generated_image=Some(path.clone());});
                        self.add_generated_image(&path);
                    }
                    UiEvent::UserImages(paths) => {
                        for path in paths {self.add_user_image(&path);}
                    }
                    UiEvent::AssistantDelta(piece) => {
                        self.update_saved_turn(|turn| turn.assistant.push_str(&piece));
                        self.append_assistant_delta(&piece);
                    }
                    UiEvent::AssistantReset(text) => {
                        self.update_saved_turn(|turn| turn.assistant=text.clone());
                        self.set_assistant_text(&text);
                    }
                    UiEvent::TurnCommitted(reply) => self.update_saved_turn(|turn| {
                        turn.assistant=reply;turn.complete=true;
                    }),
                    UiEvent::AssistantDone => {
                        self.finalize_assistant();
                        self.ivars().busy.set(false);
                        self.persist_active_chat();
                        self.refresh_chat_list();
                        self.update_chat_enabled();
                    },
                    UiEvent::Loaded(ok) => {
                        *self.ivars().loading_row.borrow_mut() = None;
                        self.refresh_model_list();
                        self.ivars().model_loaded.set(ok);
                        if ok {
                            let model = self.ivars().active_repo.borrow().clone();
                            self.update_saved_chat(|chat| chat.model=model);
                            self.persist_active_chat();
                        }
                        if !ok {
                            *self.ivars().active_repo.borrow_mut() = None;
                            *self.ivars().selected_row.borrow_mut() = None;
                            self.refresh_model_list();
                            if let Some(item) = self.ivars().logs_item.get() { item.setCollapsed(false); }
                        }
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

        #[unsafe(method(removeAttachments:))]
        fn remove_attachments(&self, _sender: Option<&AnyObject>) {
            self.ivars().attachments.borrow_mut().clear();
            if let Some(button) = self.ivars().chat_attach.get() {button.setTitle(ns_string!("Attach"));}
            self.refresh_chat_chrome();
        }

        #[unsafe(method(attachImage:))]
        fn attach_image(&self, _sender: Option<&AnyObject>) {
            if !self.ivars().model_loaded.get() || !self.ivars().supports_images.get() {
                return;
            }
            let panel=NSOpenPanel::openPanel(self.mtm());
            panel.setCanChooseFiles(true);
            panel.setCanChooseDirectories(false);
            panel.setAllowsMultipleSelection(true);
            #[allow(deprecated)]
            panel.setAllowedFileTypes(Some(&NSArray::from_slice(&[ns_string!("png"),ns_string!("jpg"),ns_string!("jpeg"),ns_string!("webp")])));
            if panel.runModal()==NSModalResponseOK {
                let paths:Vec<PathBuf>=panel.URLs().iter().filter_map(|url|url.path().map(|p|PathBuf::from(p.to_string()))).collect();
                if paths.len()>lisa_engine::models::vision::MAX_IMAGES {self.append_engine_log("Select at most eight images.");return;}
                if let Some(button)=self.ivars().chat_attach.get() {button.setTitle(&NSString::from_str(&format!("{} {}",paths.len(),if paths.len()==1 {"image"} else {"images"})));}
                *self.ivars().attachments.borrow_mut()=paths;
                self.refresh_chat_chrome();
            }
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
        let chat_store = chats::Store::new();
        let (saved_chats, errors) = chat_store.load();
        for error in errors {
            events.lock().unwrap().push(UiEvent::Log(error));
        }
        let this = Self::alloc(mtm).set_ivars(AppDelegateIvars {
            window: OnceCell::new(),
            sidebar_item: OnceCell::new(),
            logs_item: OnceCell::new(),
            models: RefCell::new(Vec::new()),
            remote_ids: RefCell::new(Vec::new()),
            models_list: OnceCell::new(),
            models_scroll: OnceCell::new(),
            loading_row: RefCell::new(None),
            active_repo: RefCell::new(None),
            downloading: Cell::new(false),
            download_progress: Cell::new(0.0),
            progress_tick: Cell::new(0),
            selected_row: RefCell::new(None),
            chat_stack: OnceCell::new(),
            chat_scroll: OnceCell::new(),
            chat_placeholder: OnceCell::new(),
            welcome_title: OnceCell::new(),
            welcome_detail: OnceCell::new(),
            model_title: OnceCell::new(),
            model_status: OnceCell::new(),
            composer_hint: OnceCell::new(),
            new_chat: OnceCell::new(),
            new_chat_row: OnceCell::new(),
            chat_store,
            chats: RefCell::new(saved_chats),
            active_chat: RefCell::new(None),
            chat_list: OnceCell::new(),
            chat_list_height: OnceCell::new(),
            chat_section: OnceCell::new(),
            models_section: OnceCell::new(),
            chat_buttons: RefCell::new(Vec::new()),
            busy: Cell::new(false),
            assistant_label: RefCell::new(None),
            assistant_stack: RefCell::new(None),
            user_stack: RefCell::new(None),
            typing: Cell::new(false),
            typing_tick: Cell::new(0),
            chat_input: OnceCell::new(),
            chat_send: OnceCell::new(),
            chat_attach: OnceCell::new(),
            attachment_clear: OnceCell::new(),
            attachments: RefCell::new(Vec::new()),
            image_mode: Cell::new(None),
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
            supports_images: Cell::new(false),
        });
        let this: Retained<Self> = unsafe { msg_send![super(this), init] };
        this.ivars().engine_tx.set(tx).ok();
        this.ivars().engine_events.set(events.clone()).ok();
        this.refresh_models();
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

    fn chat_locked(&self) -> bool {
        self.ivars().busy.get()
            || (self.ivars().loading_row.borrow().is_some() && !self.ivars().downloading.get())
    }

    fn refresh_sidebar_hover(&self) {
        // Button tracking and row replacement can swallow mouse-exit events.
        // Reconcile on the existing UI tick even with a stationary pointer:
        // scrolling, resizing and collapsing sections can move rows underneath it.
        if let Some(row) = self.ivars().new_chat_row.get() {
            row.refresh_hover();
        }
        if let Some(section) = self.ivars().chat_section.get() {
            section.refresh_hover();
        }
        if let Some(section) = self.ivars().models_section.get() {
            section.refresh_hover();
        }
        if let Some(list) = self.ivars().chat_list.get() {
            for row in list.arrangedSubviews().iter() {
                ui::refresh_sidebar_row(&row);
            }
        }
        if let Some(list) = self.ivars().models_list.get() {
            for row in list.arrangedSubviews().iter() {
                ui::refresh_sidebar_row(&row);
            }
        }
    }

    fn confirm_delete_chat(&self, index: NSInteger) {
        if self.chat_locked() {
            return;
        }
        let chat = self.ivars().chats.borrow().get(index as usize).cloned();
        let Some(chat) = chat else { return };
        let alert = NSAlert::new(self.mtm());
        alert.setAlertStyle(NSAlertStyle::Warning);
        alert.setMessageText(ns_string!("Delete chat?"));
        alert.setInformativeText(&NSString::from_str(&format!(
            "\"{}\" will be permanently deleted, including its messages and attached copies. Images saved in Pictures will be kept.",
            chat.title(),
        )));
        alert.addButtonWithTitle(ns_string!("Cancel"));
        let delete = alert.addButtonWithTitle(ns_string!("Delete"));
        delete.setHasDestructiveAction(true);
        if alert.runModal() != NSAlertSecondButtonReturn || self.chat_locked() {
            return;
        }
        let warning = match self.ivars().chat_store.delete(&chat.id) {
            Ok(warning) => warning,
            Err(error) => {
                self.append_engine_log(&format!("Could not delete chat: {error:#}"));
                let alert = NSAlert::new(self.mtm());
                alert.setMessageText(ns_string!("Could not delete chat"));
                alert.setInformativeText(&NSString::from_str(&format!(
                    "Your chat is still available. {error:#}",
                )));
                alert.addButtonWithTitle(ns_string!("OK"));
                alert.runModal();
                return;
            }
        };
        self.ivars()
            .chats
            .borrow_mut()
            .retain(|record| record.id != chat.id);
        if self.ivars().active_chat.borrow().as_ref() == Some(&chat.id) {
            // Do not use /clear here: it saves the current draft before resetting.
            self.reset_chat();
        } else {
            self.refresh_chat_list();
        }
        if let Some(warning) = warning {
            self.append_engine_log(&warning);
            let alert = NSAlert::new(self.mtm());
            alert.setMessageText(ns_string!("Chat deleted"));
            alert.setInformativeText(ns_string!(
                "Some attachment copies could not be removed from disk. Open View > Toggle Logs Panel for details."
            ));
            alert.addButtonWithTitle(ns_string!("OK"));
            alert.runModal();
        }
    }

    /// Reset the visible chat and engine context without saving the old draft.
    fn reset_chat(&self) {
        *self.ivars().active_chat.borrow_mut() = None;
        self.ivars().attachments.borrow_mut().clear();
        if let Some(input) = self.ivars().chat_input.get() {
            input.setStringValue(ns_string!(""));
        }
        if let Some(button) = self.ivars().chat_attach.get() {
            button.setTitle(ns_string!("Attach"));
        }
        self.clear_chat();
        if let Some(tx) = self.ivars().engine_tx.get() {
            let _ = tx.send(Command::Clear);
        }
        self.refresh_chat_list();
        self.update_command_menu();
        self.update_chat_enabled();
        self.focus_composer();
    }

    fn update_saved_chat(&self, update: impl FnOnce(&mut chats::Chat)) {
        let id = self.ivars().active_chat.borrow().clone();
        if let Some(chat) = self
            .ivars()
            .chats
            .borrow_mut()
            .iter_mut()
            .find(|c| Some(&c.id) == id.as_ref())
        {
            update(chat);
        }
    }

    fn update_saved_turn(&self, update: impl FnOnce(&mut chats::Turn)) {
        self.update_saved_chat(|chat| {
            if let Some(turn) = chat.turns.last_mut() {
                update(turn);
            }
        });
    }

    fn persist_active_chat(&self) {
        let id = self.ivars().active_chat.borrow().clone();
        let chat = self
            .ivars()
            .chats
            .borrow()
            .iter()
            .find(|c| Some(&c.id) == id.as_ref())
            .cloned();
        if let Some(chat) = chat {
            if let Err(error) = self.ivars().chat_store.save(&chat) {
                self.append_engine_log(&format!("Could not save chat: {error}"));
                self.add_note("This chat could not be saved to disk. Keep Lisa open and open View > Toggle Logs Panel for details.");
            }
        }
    }

    fn save_chat_draft(&self) {
        let draft = self
            .ivars()
            .chat_input
            .get()
            .map(|v| v.stringValue().to_string())
            .unwrap_or_default();
        if self.ivars().active_chat.borrow().is_none() && !draft.trim().is_empty() {
            let system = self
                .ivars()
                .system_view
                .get()
                .map(|v| v.string().to_string())
                .unwrap_or_default();
            let mut chat = chats::Chat::new(self.ivars().active_repo.borrow().clone(), system);
            chat.draft = draft;
            *self.ivars().active_chat.borrow_mut() = Some(chat.id.clone());
            self.ivars().chats.borrow_mut().insert(0, chat);
        } else {
            self.update_saved_chat(|chat| chat.draft = draft);
        }
        self.persist_active_chat();
    }

    fn begin_saved_turn(
        &self,
        text: &str,
        system: &str,
        images: &[PathBuf],
    ) -> anyhow::Result<Vec<PathBuf>> {
        let id = self.ivars().active_chat.borrow().clone();
        let current = self
            .ivars()
            .chats
            .borrow()
            .iter()
            .find(|c| Some(&c.id) == id.as_ref())
            .cloned();
        let mut chat = current.unwrap_or_else(|| {
            chats::Chat::new(self.ivars().active_repo.borrow().clone(), system.into())
        });
        if chat.turns.is_empty() {
            chat.system = system.into();
        }
        chat.model = self.ivars().active_repo.borrow().clone();
        let images = self.ivars().chat_store.keep_images(&chat.id, images)?;
        chat.turns.push(chats::Turn {
            user: text.into(),
            images: images.clone(),
            ..Default::default()
        });
        chat.draft.clear();
        chat.updated = chats::timestamp();
        if let Err(error) = self.ivars().chat_store.save(&chat) {
            for path in &images {
                let _ = std::fs::remove_file(path);
            }
            return Err(error);
        }
        *self.ivars().active_chat.borrow_mut() = Some(chat.id.clone());
        {
            let mut records = self.ivars().chats.borrow_mut();
            records.retain(|old| old.id != chat.id);
            records.insert(0, chat);
        }
        self.refresh_chat_list();
        Ok(images)
    }

    fn load_chat_model(&self, repo: String, path: PathBuf, row: usize) {
        self.ivars().downloading.set(false);
        *self.ivars().active_repo.borrow_mut() = Some(repo);
        *self.ivars().loading_row.borrow_mut() = Some(row);
        *self.ivars().selected_row.borrow_mut() = Some(row);
        self.ivars().model_loaded.set(false);
        self.refresh_model_list();
        self.set_chat_enabled(false);
        if let Some(tx) = self.ivars().engine_tx.get() {
            let _ = tx.send(Command::Load(path));
        }
    }

    fn open_chat(&self, chat: chats::Chat) {
        if self.ivars().active_chat.borrow().as_ref() == Some(&chat.id) {
            return;
        }
        self.save_chat_draft();
        self.clear_chat();
        *self.ivars().active_chat.borrow_mut() = Some(chat.id.clone());
        self.ivars().attachments.borrow_mut().clear();
        if let Some(button) = self.ivars().chat_attach.get() {
            button.setTitle(ns_string!("Attach"));
        }
        if let Some(input) = self.ivars().chat_input.get() {
            input.setStringValue(&NSString::from_str(&chat.draft));
        }
        if let Some(system) = self.ivars().system_view.get() {
            system.setString(&NSString::from_str(&chat.system));
        }
        for turn in &chat.turns {
            self.add_user_message(&turn.user);
            for path in &turn.images {
                self.add_user_image(path);
            }
            if let Some(path) = &turn.generated_image {
                self.add_generated_image(path);
            } else {
                let text = if turn.complete {
                    turn.assistant.clone()
                } else {
                    format!("{}\n\n*This response was not completed.*", turn.assistant)
                        .trim()
                        .to_string()
                };
                self.set_assistant_text(&text);
            }
            self.finalize_assistant();
        }
        if let Some(tx) = self.ivars().engine_tx.get() {
            let _ = tx.send(Command::Restore {
                turns: chat.turns,
                system: chat.system,
            });
        }
        if let Some(repo) = chat.model {
            if self.ivars().active_repo.borrow().as_ref() != Some(&repo)
                || !self.ivars().model_loaded.get()
            {
                let local = self
                    .ivars()
                    .models
                    .borrow()
                    .iter()
                    .enumerate()
                    .find(|(_, model)| model.repo_id == repo && model.local && !model.too_big)
                    .map(|(row, model)| (row, PathBuf::from(&model.path)));
                if let Some((row, path)) = local {
                    self.load_chat_model(repo, path, row);
                } else {
                    *self.ivars().active_repo.borrow_mut() = None;
                    *self.ivars().selected_row.borrow_mut() = None;
                    self.ivars().model_loaded.set(false);
                    self.refresh_model_list();
                    self.add_note(
                        "This chat's model is unavailable. Choose a local model to continue.",
                    );
                }
            }
        }
        self.refresh_chat_list();
        self.update_command_menu();
        self.update_chat_enabled();
        self.focus_composer();
    }

    fn refresh_chat_list(&self) {
        let Some(list) = self.ivars().chat_list.get() else {
            return;
        };
        for view in list.arrangedSubviews().iter() {
            list.removeArrangedSubview(&view);
            view.removeFromSuperview();
        }
        self.ivars().chat_buttons.borrow_mut().clear();
        let records = self.ivars().chats.borrow();
        let active = self.ivars().active_chat.borrow();
        if records.is_empty() {
            let empty = ui::label(
                self.mtm(),
                "Your chats will appear here",
                12.0,
                false,
                &NSColor::tertiaryLabelColor(),
            );
            empty.setTranslatesAutoresizingMaskIntoConstraints(false);
            list.addArrangedSubview(&empty);
            empty
                .heightAnchor()
                .constraintEqualToConstant(32.0)
                .setActive(true);
        }
        for (index, chat) in records.iter().enumerate() {
            let selected = active.as_ref() == Some(&chat.id);
            let title = chat.title();
            let delete =
                ui::symbol_button(self.mtm(), "", "trash", self.target(), sel!(deleteChat:));
            delete.setTag(index as NSInteger);
            delete.setImagePosition(NSCellImagePosition::ImageOnly);
            delete.setSymbolConfiguration(Some(&ui::sidebar_symbol_configuration()));
            delete.setContentTintColor(Some(&NSColor::secondaryLabelColor()));
            delete.setToolTip(Some(ns_string!("Delete chat")));
            delete
                .setAccessibilityLabel(Some(&NSString::from_str(&format!("Delete chat: {title}"))));
            delete.setEnabled(!self.chat_locked());
            let row = ui::chat_row(self.mtm(), selected, delete.clone());
            let icon = if chat.turns.iter().any(|t| t.generated_image.is_some()) {
                "photo"
            } else {
                "bubble.left"
            };
            let button =
                ui::symbol_button(self.mtm(), &title, icon, self.target(), sel!(selectChat:));
            button.setSymbolConfiguration(Some(&ui::sidebar_symbol_configuration()));
            button.setImageScaling(objc2_app_kit::NSImageScaling::ScaleNone);
            button.setTitle(&NSString::from_str(&format!("\u{2002}{title}")));
            button.setTag(index as NSInteger);
            button.setAlignment(NSTextAlignment::Left);
            button.setFont(Some(&NSFont::systemFontOfSize(12.0)));
            let title_color = if selected {
                NSColor::labelColor()
            } else {
                NSColor::secondaryLabelColor()
            };
            button.setContentTintColor(Some(&title_color));
            button.setEnabled(!self.chat_locked());
            button.setToolTip(Some(&NSString::from_str(&format!(
                "{}\n{}",
                title,
                chat.model.as_deref().unwrap_or("Choose a model")
            ))));
            button.setAccessibilityLabel(Some(&NSString::from_str(&format!(
                "{}{}",
                title,
                if selected { ", current chat" } else { "" }
            ))));
            if let Some(cell) = button.cell() {
                cell.setLineBreakMode(NSLineBreakMode::ByTruncatingTail);
            }
            button.setContentCompressionResistancePriority_forOrientation(
                249.0,
                NSLayoutConstraintOrientation::Horizontal,
            );
            let menu = NSMenu::initWithTitle(NSMenu::alloc(self.mtm()), ns_string!(""));
            let delete_item = unsafe {
                NSMenuItem::initWithTitle_action_keyEquivalent(
                    NSMenuItem::alloc(self.mtm()),
                    ns_string!("Delete Chat…"),
                    Some(sel!(deleteChatFromMenu:)),
                    ns_string!(""),
                )
            };
            delete_item.setTag(index as NSInteger);
            unsafe {
                delete_item.setTarget(Some(self.target()));
                row.setMenu(Some(&menu));
                button.setMenu(Some(&menu));
            }
            menu.addItem(&delete_item);
            row.addSubview(&button);
            row.addSubview(&delete);
            list.addArrangedSubview(&row);
            for constraint in [
                row.widthAnchor()
                    .constraintEqualToAnchor(&list.widthAnchor()),
                row.heightAnchor().constraintEqualToConstant(34.0),
                button
                    .leadingAnchor()
                    .constraintEqualToAnchor_constant(&row.leadingAnchor(), 10.0),
                button
                    .trailingAnchor()
                    .constraintEqualToAnchor_constant(&delete.leadingAnchor(), -4.0),
                button.topAnchor().constraintEqualToAnchor(&row.topAnchor()),
                button
                    .bottomAnchor()
                    .constraintEqualToAnchor(&row.bottomAnchor()),
                delete
                    .trailingAnchor()
                    .constraintEqualToAnchor_constant(&row.trailingAnchor(), -5.0),
                delete
                    .centerYAnchor()
                    .constraintEqualToAnchor(&row.centerYAnchor()),
                delete.widthAnchor().constraintEqualToConstant(26.0),
                delete.heightAnchor().constraintEqualToConstant(26.0),
            ] {
                constraint.setActive(true);
            }
            self.ivars()
                .chat_buttons
                .borrow_mut()
                .extend([button, delete]);
        }
        if let Some(height) = self.ivars().chat_list_height.get() {
            height.setConstant((records.len().max(1) as f64 * 38.0).min(200.0));
        }
        list.layoutSubtreeIfNeeded();
        self.refresh_sidebar_hover();
    }

    fn add_generated_image(&self, path: &std::path::Path) {
        let Some(stack) = self.ivars().assistant_stack.borrow().clone() else {
            return;
        };
        let Some(card) = crate::media::GeneratedImageCard::new(self.mtm(), path) else {
            self.set_assistant_text("Your image was saved, but its preview could not be opened. You can find it in Pictures → Lisa.");
            return;
        };
        if let Some(label) = self.ivars().assistant_label.borrow_mut().take() {
            label.removeFromSuperview();
        }
        self.ivars().typing.set(false);
        stack.addArrangedSubview(&card);
        card.widthAnchor()
            .constraintLessThanOrEqualToAnchor(&stack.widthAnchor())
            .setActive(true);
    }

    /// Add attachments inside the user message that owns them.
    fn add_user_image(&self, path: &std::path::Path) {
        let stack = self.ivars().user_stack.borrow().clone();
        let Some(stack) = stack else {
            return;
        };
        let Some(image) = NSImage::initWithContentsOfFile(
            NSImage::alloc(),
            &NSString::from_str(&path.to_string_lossy()),
        ) else {
            return;
        };
        let view = NSImageView::new(self.mtm());
        view.setImage(Some(&image));
        view.setImageScaling(NSImageScaling::ScaleProportionallyUpOrDown);
        view.setTranslatesAutoresizingMaskIntoConstraints(false);
        stack.addArrangedSubview(&view);
        let max_width = 220.0;
        view.widthAnchor()
            .constraintLessThanOrEqualToConstant(max_width)
            .setActive(true);
        view.widthAnchor()
            .constraintLessThanOrEqualToAnchor(&stack.widthAnchor())
            .setActive(true);
        let preferred = view.widthAnchor().constraintEqualToConstant(max_width);
        preferred.setPriority(NSLayoutPriorityDefaultLow);
        preferred.setActive(true);
        let size = image.size();
        let aspect = if size.width > 0.0 {
            size.height / size.width
        } else {
            1.0
        };
        view.heightAnchor()
            .constraintEqualToAnchor_multiplier(&view.widthAnchor(), aspect)
            .setActive(true);
    }

    /// Run a slash command locally or hand the current turn to the engine.
    fn send_current(&self) {
        if self.chat_locked() {
            return;
        }
        let Some(input) = self.ivars().chat_input.get() else {
            return;
        };
        let text = input.stringValue().to_string();
        let text = text.trim().to_string();
        if text.is_empty() && self.ivars().attachments.borrow().is_empty() {
            return;
        }
        // Slash commands run locally instead of going to the model.
        if text.starts_with('/') && !text.contains(char::is_whitespace) {
            if let Some(&(name, _)) = CHAT_COMMANDS.iter().find(|(n, _)| *n == text) {
                input.setStringValue(ns_string!(""));
                self.run_chat_command(name);
                return;
            }
        }
        if !self.ivars().model_loaded.get() {
            return;
        }
        let system = self
            .ivars()
            .system_view
            .get()
            .map(|v| v.string().to_string())
            .unwrap_or_default();
        let original_images = self.ivars().attachments.borrow().clone();
        let images = match self.begin_saved_turn(&text, &system, &original_images) {
            Ok(images) => images,
            Err(error) => {
                self.append_engine_log(&format!("Could not save chat: {error}"));
                self.add_note(&format!(
                    "Could not save this message: {error}. Your draft is still in the composer."
                ));
                return;
            }
        };
        input.setStringValue(ns_string!(""));
        self.update_command_menu();
        self.ivars().busy.set(true);
        self.refresh_chat_chrome();
        if let Some(tx) = self.ivars().engine_tx.get() {
            self.ivars().attachments.borrow_mut().clear();
            if let Some(button) = self.ivars().chat_attach.get() {
                button.setTitle(ns_string!("Attach"));
            }
            self.refresh_chat_chrome();
            if tx
                .send(Command::Chat {
                    text,
                    system,
                    images,
                })
                .is_err()
            {
                self.ivars().busy.set(false);
                self.update_chat_enabled();
                self.add_note(
                    "The inference engine stopped. Reopen Lisa to continue this saved chat.",
                );
            }
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
        let selected = *self.ivars().selected_row.borrow() == Some(row as usize);
        let cell = ui::model_row(mtm, selected);
        cell.setTranslatesAutoresizingMaskIntoConstraints(false);
        cell.heightAnchor()
            .constraintEqualToConstant(58.0)
            .setActive(true);
        let color = if model.too_big {
            NSColor::tertiaryLabelColor()
        } else {
            NSColor::labelColor()
        };
        let (symbol, model_kind) =
            if lisa_engine::models::image_generation::is_model_type(&model.model_type) {
                ("photo", "Image generation")
            } else {
                ("bubble.left", "Chat model")
            };
        let icon = NSImageView::new(mtm);
        icon.setSymbolConfiguration(Some(&ui::sidebar_symbol_configuration()));
        icon.setImageScaling(objc2_app_kit::NSImageScaling::ScaleNone);
        icon.setImage(
            NSImage::imageWithSystemSymbolName_accessibilityDescription(
                &NSString::from_str(symbol),
                Some(&NSString::from_str(model_kind)),
            )
            .as_deref(),
        );
        icon.setContentTintColor(Some(&NSColor::secondaryLabelColor()));
        let curated = catalog::find(&model.repo_id);
        let name = ui::label(
            mtm,
            curated.map_or_else(|| display_name(&model.repo_id), |entry| entry.name),
            12.0,
            selected,
            &color,
        );
        let owner = model.repo_id.split('/').next().unwrap_or("Local");
        let detail = if model.too_big {
            "Exceeds available memory".to_string()
        } else {
            format!(
                "{} · {:.1} GB",
                curated.map_or(owner, |entry| entry.quantization),
                model.size as f64 / 1e9
            )
        };
        let subtitle = ui::label(mtm, &detail, 10.0, false, &NSColor::secondaryLabelColor());
        cell.setToolTip(Some(&NSString::from_str(&format!(
            "{}\n{}\n{}",
            model.repo_id,
            curated.map_or("Local checkpoint", |entry| entry.description),
            detail
        ))));
        for label in [&name, &subtitle] {
            if let Some(c) = label.cell() {
                c.setLineBreakMode(NSLineBreakMode::ByTruncatingTail);
                c.setUsesSingleLineMode(true);
            }
        }
        for view in [&*icon as &NSView, &*name, &*subtitle] {
            view.setTranslatesAutoresizingMaskIntoConstraints(false);
            cell.addSubview(view);
        }
        let loading = *self.ivars().loading_row.borrow() == Some(row as usize);
        let show_status = loading || !model.local || model.too_big;
        for c in [
            icon.leadingAnchor().constraintEqualToAnchor_constant(
                &cell.leadingAnchor(),
                ui::MODEL_ROW_CONTENT_INSET,
            ),
            icon.topAnchor()
                .constraintEqualToAnchor_constant(&cell.topAnchor(), 14.0),
            icon.widthAnchor()
                .constraintEqualToConstant(ui::SIDEBAR_ICON_SIZE),
            icon.heightAnchor()
                .constraintEqualToConstant(ui::SIDEBAR_ICON_SIZE),
            name.leadingAnchor()
                .constraintEqualToAnchor_constant(&icon.trailingAnchor(), 8.0),
            name.topAnchor()
                .constraintEqualToAnchor_constant(&cell.topAnchor(), 11.0),
            name.trailingAnchor().constraintEqualToAnchor_constant(
                &cell.trailingAnchor(),
                -ui::MODEL_ROW_CONTENT_INSET - if show_status { 24.0 } else { 0.0 },
            ),
            subtitle
                .leadingAnchor()
                .constraintEqualToAnchor(&name.leadingAnchor()),
            subtitle
                .topAnchor()
                .constraintEqualToAnchor_constant(&name.bottomAnchor(), 4.0),
            subtitle
                .trailingAnchor()
                .constraintEqualToAnchor(&name.trailingAnchor()),
        ] {
            c.setActive(true);
        }
        if show_status {
            let indicator: Retained<NSView> = if loading && self.ivars().downloading.get() {
                ProgressRing::new(mtm, self.ivars().download_progress.get()).into_super()
            } else if loading {
                let spinner = NSProgressIndicator::new(mtm);
                spinner.setStyle(NSProgressIndicatorStyle::Spinning);
                spinner.setControlSize(NSControlSize::Small);
                spinner.setIndeterminate(true);
                unsafe { spinner.startAnimation(None) };
                spinner.into_super()
            } else {
                let (symbol, description) = if model.too_big {
                    ("nosign", "Exceeds available memory")
                } else {
                    ("arrow.down.circle", "Download model")
                };
                let status = NSImageView::new(mtm);
                status.setImage(
                    NSImage::imageWithSystemSymbolName_accessibilityDescription(
                        &NSString::from_str(symbol),
                        Some(&NSString::from_str(description)),
                    )
                    .as_deref(),
                );
                status.setContentTintColor(Some(&NSColor::secondaryLabelColor()));
                status.setToolTip(Some(&NSString::from_str(description)));
                status.into_super().into_super()
            };
            indicator.setTranslatesAutoresizingMaskIntoConstraints(false);
            cell.addSubview(&indicator);
            ui::size(&indicator, 16.0, 16.0);
            indicator
                .trailingAnchor()
                .constraintEqualToAnchor_constant(
                    &cell.trailingAnchor(),
                    -ui::MODEL_ROW_CONTENT_INSET,
                )
                .setActive(true);
            indicator
                .centerYAnchor()
                .constraintEqualToAnchor(&cell.centerYAnchor())
                .setActive(true);
        }
        let button = ui::button(mtm, "", Some(self.target()), Some(sel!(selectModel:)), false);
        button.setTransparent(true);
        button.setTag(row);
        button.setAccessibilityLabel(Some(&NSString::from_str(&format!(
            "{} · {}", name.stringValue(), detail,
        ))));
        button.setAccessibilitySelected(selected);
        button.setTranslatesAutoresizingMaskIntoConstraints(false);
        cell.addSubview(&button);
        for constraint in [
            button.leadingAnchor().constraintEqualToAnchor(&cell.leadingAnchor()),
            button.trailingAnchor().constraintEqualToAnchor(&cell.trailingAnchor()),
            button.topAnchor().constraintEqualToAnchor(&cell.topAnchor()),
            button.bottomAnchor().constraintEqualToAnchor(&cell.bottomAnchor()),
        ] {
            constraint.setActive(true);
        }
        if model.local {
            let menu = NSMenu::initWithTitle(NSMenu::alloc(mtm), ns_string!(""));
            let delete = unsafe {
                NSMenuItem::initWithTitle_action_keyEquivalent(
                    NSMenuItem::alloc(mtm), ns_string!("Delete Model"), Some(sel!(deleteModel:)), ns_string!(""),
                )
            };
            delete.setTag(row);
            unsafe {
                delete.setTarget(Some(self.target()));
                button.setMenu(Some(&menu));
            }
            menu.addItem(&delete);
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
            NSColor::quaternaryLabelColor()
        } else {
            NSColor::clearColor()
        };
        let bubble = BubbleView::new(mtm, &fill);
        bubble.setTranslatesAutoresizingMaskIntoConstraints(false);
        row.addSubview(&bubble);

        let content = NSStackView::new(mtm);
        content.setOrientation(NSUserInterfaceLayoutOrientation::Vertical);
        content.setAlignment(NSLayoutAttribute::Leading);
        content.setSpacing(12.0);
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
        bubble
            .leadingAnchor()
            .constraintGreaterThanOrEqualToAnchor(&row.leadingAnchor())
            .setActive(true);
        bubble
            .trailingAnchor()
            .constraintLessThanOrEqualToAnchor(&row.trailingAnchor())
            .setActive(true);
        if is_user {
            bubble
                .trailingAnchor()
                .constraintEqualToAnchor(&row.trailingAnchor())
                .setActive(true);
            let preferred = bubble
                .widthAnchor()
                .constraintEqualToAnchor_multiplier(&row.widthAnchor(), 0.82);
            preferred.setPriority(NSLayoutPriorityDefaultLow);
            preferred.setActive(true);
        } else {
            bubble
                .leadingAnchor()
                .constraintEqualToAnchor(&row.leadingAnchor())
                .setActive(true);
            bubble
                .trailingAnchor()
                .constraintEqualToAnchor(&row.trailingAnchor())
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
        let (Some(input), Some(menu)) =
            (self.ivars().chat_input.get(), self.ivars().cmd_menu.get())
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
        let pos = visible.iter().position(|&i| i as isize == cur).unwrap_or(0);
        let next = (pos as isize + delta).rem_euclid(visible.len() as isize) as usize;
        self.ivars().cmd_sel.set(visible[next] as isize);
        self.apply_command_selection();
        true
    }

    /// Run a slash command (`name` is one of `CHAT_COMMANDS`' names).
    fn run_chat_command(&self, name: &str) {
        match name {
            "/clear" => {
                if self.chat_locked() {
                    return;
                }
                if let Some(input) = self.ivars().chat_input.get() {
                    let text = input.stringValue().to_string();
                    if text.starts_with('/')
                        && CHAT_COMMANDS
                            .iter()
                            .any(|(command, _)| command.starts_with(text.trim()))
                    {
                        input.setStringValue(ns_string!(""));
                    }
                }
                self.save_chat_draft();
                self.reset_chat();
            }
            "/help" => self.show_command_help(),
            _ => {}
        }
        if let Some(input) = self.ivars().chat_input.get() {
            input.setStringValue(ns_string!(""));
        }
        self.update_command_menu();
        self.refresh_chat_chrome();
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

    /// Empty the visible transcript; saved conversations remain in the sidebar.
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
        *self.ivars().user_stack.borrow_mut() = None;
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
        render_blocks(self.mtm(), &content, text, &NSColor::labelColor());
        *self.ivars().user_stack.borrow_mut() = Some(content);
        chat.addArrangedSubview(&row);

        // Placeholder assistant bubble with an animated typing indicator.
        let (row, content) = self.make_bubble(false);
        let label = ui::wrapping_label(self.mtm(), "", BUBBLE_FONT_SIZE, &NSColor::labelColor());
        label.setTranslatesAutoresizingMaskIntoConstraints(false);
        label.setMaximumNumberOfLines(0);
        label.setPreferredMaxLayoutWidth(BUBBLE_MAX_WIDTH - 30.0);
        label.setAttributedStringValue(&typing_dots(self.mtm(), 0.0));
        content.addArrangedSubview(&label);
        label
            .widthAnchor()
            .constraintEqualToAnchor(&content.widthAnchor())
            .setActive(true);
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
        self.refresh_model_row(row);
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

    fn focus_composer(&self) {
        let Some(input) = self
            .ivars()
            .chat_input
            .get()
            .filter(|input| input.isEnabled())
        else {
            return;
        };
        if let Some(window) = self.ivars().window.get() {
            window.makeFirstResponder(Some(&**input));
            if let Some(editor) = input.currentEditor() {
                editor.setSelectedRange(NSRange::new(input.stringValue().length(), 0));
            }
        }
    }

    /// Enable the chat input + Send button once a model is loaded (and focus
    /// the input). While no model is loaded the input is greyed and disabled.
    fn set_chat_enabled(&self, on: bool) {
        if let Some(switch) = self.ivars().serve_switch.get() {
            switch.setEnabled(self.ivars().serving.get() || (on && !self.ivars().busy.get()));
        }
        if let Some(button) = self.ivars().chat_attach.get() {
            button.setEnabled(
                on && self.ivars().supports_images.get() && self.ivars().image_mode.get().is_none(),
            );
        }
        if let Some(input) = self.ivars().chat_input.get() {
            let newly_enabled = on && !input.isEnabled();
            input.setEnabled(on);
            let color = if on {
                NSColor::labelColor()
            } else {
                NSColor::tertiaryLabelColor()
            };
            input.setTextColor(Some(&color));
            if newly_enabled {
                self.focus_composer();
            }
        }
        self.refresh_chat_chrome();
    }

    /// Put downloaded checkpoints first, retaining curated order within each
    /// group, and flag models too large for RAM.
    fn refresh_models(&self) {
        let local = discover_models();
        let remote = self.ivars().remote_ids.borrow().clone();
        let threshold = model_ram_budget();
        let loading_repo = self.ivars().loading_row.borrow().and_then(|row| {
            self.ivars()
                .models
                .borrow()
                .get(row)
                .map(|model| model.repo_id.clone())
        });
        let mut merged = local;
        for entry in catalog::MODELS {
            if merged.iter().any(|m| m.repo_id == entry.repo_id) {
                continue;
            }
            let size = remote
                .iter()
                .find(|(id, _)| id == entry.repo_id)
                .map_or(entry.download_bytes, |(_, size)| *size);
            merged.push(Model {
                repo_id: entry.repo_id.into(),
                model_type: entry.model_type.into(),
                path: String::new(),
                local: false,
                size,
                resident_bytes: entry.resident_bytes,
                too_big: false,
            });
        }
        for m in &mut merged {
            let resident = m.resident_bytes.unwrap_or(m.size);
            m.too_big = resident > 0 && threshold > 0 && resident > threshold;
        }
        merged.sort_by(|a, b| {
            (!a.local, a.too_big, catalog::rank(&a.repo_id), &a.repo_id).cmp(&(
                !b.local,
                b.too_big,
                catalog::rank(&b.repo_id),
                &b.repo_id,
            ))
        });
        if let Some(repo) = loading_repo {
            *self.ivars().loading_row.borrow_mut() =
                merged.iter().position(|model| model.repo_id == repo);
        }
        *self.ivars().models.borrow_mut() = merged;
        // Re-anchor the highlight to the active model after the re-sort.
        let active = self.ivars().active_repo.borrow().clone();
        let index = active.as_ref().and_then(|r| {
            self.ivars()
                .models
                .borrow()
                .iter()
                .position(|m| &m.repo_id == r)
        });
        *self.ivars().selected_row.borrow_mut() = index;
        self.refresh_model_list();
    }

    fn refresh_model_list(&self) {
        let (Some(list), Some(scroll)) = (
            self.ivars().models_list.get(),
            self.ivars().models_scroll.get(),
        ) else {
            return;
        };
        ui::preserve_scroll_position(scroll, || {
            for row in list.arrangedSubviews().iter() {
                list.removeArrangedSubview(&row);
                row.removeFromSuperview();
            }
            for index in 0..self.ivars().models.borrow().len() {
                if let Some(row) = self.make_model_cell(index as NSInteger) {
                    list.addArrangedSubview(&row);
                    row.widthAnchor()
                        .constraintEqualToAnchor(&list.widthAnchor())
                        .setActive(true);
                }
            }
        });
    }

    fn refresh_model_row(&self, index: usize) {
        let (Some(list), Some(scroll)) = (
            self.ivars().models_list.get(),
            self.ivars().models_scroll.get(),
        ) else {
            return;
        };
        let rows = list.arrangedSubviews();
        if index >= rows.len() {
            return;
        }
        let Some(row) = self.make_model_cell(index as NSInteger) else {
            return;
        };
        ui::preserve_scroll_position(scroll, || {
            let old = rows.objectAtIndex(index);
            list.removeArrangedSubview(&old);
            old.removeFromSuperview();
            list.insertArrangedSubview_atIndex(&row, index as NSInteger);
            row.widthAnchor()
                .constraintEqualToAnchor(&list.widthAnchor())
                .setActive(true);
        });
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
        let quit = unsafe {
            app_menu.addItemWithTitle_action_keyEquivalent(
                ns_string!("Quit Lisa"),
                Some(sel!(terminate:)),
                ns_string!("q"),
            )
        };
        unsafe { quit.setTarget(Some(app)) };
        let new_chat = unsafe {
            app_menu.addItemWithTitle_action_keyEquivalent(
                ns_string!("New Chat"),
                Some(sel!(newChat:)),
                ns_string!("n"),
            )
        };
        unsafe { new_chat.setTarget(Some(self.target())) };
        app_item.setSubmenu(Some(&app_menu));
        menu.addItem(&app_item);

        let edit_item = unsafe {
            NSMenuItem::initWithTitle_action_keyEquivalent(
                NSMenuItem::alloc(mtm),
                ns_string!("Edit"),
                None,
                ns_string!(""),
            )
        };
        let edit_menu = NSMenu::initWithTitle(NSMenu::alloc(mtm), ns_string!("Edit"));
        for (title, action, key) in [
            ("Undo", sel!(undo:), "z"),
            ("Cut", sel!(cut:), "x"),
            ("Copy", sel!(copy:), "c"),
            ("Paste", sel!(paste:), "v"),
            ("Select All", sel!(selectAll:), "a"),
        ] {
            unsafe {
                edit_menu.addItemWithTitle_action_keyEquivalent(
                    &NSString::from_str(title),
                    Some(action),
                    &NSString::from_str(key),
                );
            }
        }
        edit_item.setSubmenu(Some(&edit_menu));
        menu.addItem(&edit_item);

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
            ("Model Settings", sel!(toggleInspector:), "i"),
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
            item.setKeyEquivalentModifierMask(
                NSEventModifierFlags::Control | NSEventModifierFlags::Command,
            );
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
        let root = RoundedView::new(mtm, &NSColor::windowBackgroundColor(), 0.0);
        let new_chat = ui::symbol_button(
            mtm,
            "New chat",
            "square.and.pencil",
            self.target(),
            sel!(newChat:),
        );
        new_chat.setAlignment(NSTextAlignment::Left);
        new_chat.setFont(Some(&NSFont::systemFontOfSize(13.0)));
        let new_bg = ui::sidebar_action(mtm);
        let chats_label = ui::section_header(mtm, "Chats", self.target(), sel!(toggleChatSection:));
        let label = ui::section_header(mtm, "Models", self.target(), sel!(toggleModelsSection:));
        for view in [&*new_bg as &NSView, &*chats_label, &*label] {
            view.setTranslatesAutoresizingMaskIntoConstraints(false);
            root.addSubview(view);
        }
        new_bg.addSubview(&new_chat);
        self.ivars().new_chat.set(new_chat.clone()).ok();
        self.ivars().new_chat_row.set(new_bg.clone()).ok();

        let history_document = FlippedView::new(mtm);
        history_document.setTranslatesAutoresizingMaskIntoConstraints(false);
        let history_list = ui::stack(
            mtm,
            NSUserInterfaceLayoutOrientation::Vertical,
            4.0,
            NSLayoutAttribute::Width,
        );
        history_document.addSubview(&history_list);
        let history_scroll = ui::scroll(mtm, &history_document);
        history_scroll.setAutomaticallyAdjustsContentInsets(false);
        root.addSubview(&history_scroll);
        for constraint in [
            history_document
                .widthAnchor()
                .constraintEqualToAnchor(&history_scroll.contentView().widthAnchor()),
            history_list
                .leadingAnchor()
                .constraintEqualToAnchor_constant(&history_document.leadingAnchor(), 4.0),
            history_list
                .trailingAnchor()
                .constraintEqualToAnchor_constant(&history_document.trailingAnchor(), -4.0),
            history_list
                .topAnchor()
                .constraintEqualToAnchor(&history_document.topAnchor()),
            history_list
                .bottomAnchor()
                .constraintEqualToAnchor(&history_document.bottomAnchor()),
        ] {
            constraint.setActive(true);
        }
        let history_height = history_scroll
            .heightAnchor()
            .constraintEqualToConstant(38.0);
        history_height.setPriority(NSLayoutPriorityDefaultLow);
        self.ivars()
            .chat_list_height
            .set(history_height.clone())
            .ok();
        self.ivars().chat_list.set(history_list).ok();

        let models_document = FlippedView::new(mtm);
        models_document.setTranslatesAutoresizingMaskIntoConstraints(false);
        let list = ui::stack(
            mtm,
            NSUserInterfaceLayoutOrientation::Vertical,
            2.0,
            NSLayoutAttribute::Width,
        );
        models_document.addSubview(&list);
        let scroll = ui::sidebar_scroll(mtm, &models_document);
        scroll.setAutomaticallyAdjustsContentInsets(false);
        for constraint in [
            models_document
                .widthAnchor()
                .constraintEqualToAnchor(&scroll.contentView().widthAnchor()),
            list.leadingAnchor()
                .constraintEqualToAnchor(&models_document.leadingAnchor()),
            list.trailingAnchor()
                .constraintEqualToAnchor(&models_document.trailingAnchor()),
            list.topAnchor()
                .constraintEqualToAnchor(&models_document.topAnchor()),
            list.bottomAnchor()
                .constraintEqualToAnchor_constant(&models_document.bottomAnchor(), -20.0),
        ] {
            constraint.setActive(true);
        }
        self.ivars().models_list.set(list).ok();
        self.ivars().models_scroll.set(scroll.clone()).ok();
        root.addSubview(&scroll);

        let history_top = history_scroll
            .topAnchor()
            .constraintEqualToAnchor_constant(&chats_label.bottomAnchor(), 4.0);
        let models_top = scroll
            .topAnchor()
            .constraintEqualToAnchor(&label.bottomAnchor());
        self.ivars()
            .chat_section
            .set(ui::SidebarSection::new(
                chats_label.clone(),
                history_scroll.clone(),
                vec![
                    history_height,
                    history_scroll
                        .heightAnchor()
                        .constraintGreaterThanOrEqualToConstant(32.0),
                    history_scroll
                        .heightAnchor()
                        .constraintLessThanOrEqualToAnchor_multiplier(&root.heightAnchor(), 0.27),
                ],
                history_top.clone(),
                "sidebar.chatsCollapsed",
            ))
            .ok();
        self.ivars()
            .models_section
            .set(ui::SidebarSection::new(
                label.clone(),
                scroll.clone(),
                vec![
                    scroll
                        .bottomAnchor()
                        .constraintEqualToAnchor(&root.bottomAnchor()),
                ],
                models_top.clone(),
                "sidebar.modelsCollapsed",
            ))
            .ok();
        for c in [
            new_bg
                .leadingAnchor()
                .constraintEqualToAnchor_constant(&root.leadingAnchor(), 14.0),
            new_bg
                .trailingAnchor()
                .constraintEqualToAnchor_constant(&root.trailingAnchor(), -14.0),
            new_bg
                .topAnchor()
                .constraintEqualToAnchor_constant(&root.topAnchor(), 52.0),
            new_bg.heightAnchor().constraintEqualToConstant(36.0),
            new_chat
                .leadingAnchor()
                .constraintEqualToAnchor_constant(&new_bg.leadingAnchor(), 10.0),
            new_chat
                .trailingAnchor()
                .constraintEqualToAnchor_constant(&new_bg.trailingAnchor(), -10.0),
            new_chat
                .topAnchor()
                .constraintEqualToAnchor(&new_bg.topAnchor()),
            new_chat
                .bottomAnchor()
                .constraintEqualToAnchor(&new_bg.bottomAnchor()),
            label
                .leadingAnchor()
                .constraintEqualToAnchor_constant(&root.leadingAnchor(), 22.0),
            label
                .trailingAnchor()
                .constraintEqualToAnchor_constant(&root.trailingAnchor(), -18.0),
            chats_label
                .leadingAnchor()
                .constraintEqualToAnchor_constant(&root.leadingAnchor(), 22.0),
            chats_label
                .trailingAnchor()
                .constraintEqualToAnchor_constant(&root.trailingAnchor(), -18.0),
            chats_label
                .topAnchor()
                .constraintEqualToAnchor_constant(&new_bg.bottomAnchor(), 22.0),
            history_top,
            history_scroll
                .leadingAnchor()
                .constraintEqualToAnchor_constant(&root.leadingAnchor(), 10.0),
            history_scroll
                .trailingAnchor()
                .constraintEqualToAnchor_constant(&root.trailingAnchor(), -10.0),
            label
                .topAnchor()
                .constraintEqualToAnchor_constant(&history_scroll.bottomAnchor(), 22.0),
            scroll
                .leadingAnchor()
                .constraintEqualToAnchor_constant(&root.leadingAnchor(), 8.0),
            scroll
                .trailingAnchor()
                .constraintEqualToAnchor_constant(&root.trailingAnchor(), -8.0),
            models_top,
        ] {
            c.setActive(true);
        }
        root.into_super()
    }

    fn build_chat(&self, mtm: MainThreadMarker) -> Retained<NSView> {
        let root = RoundedView::new(mtm, &NSColor::textBackgroundColor(), 0.0);
        let header = NSView::new(mtm);
        header.setTranslatesAutoresizingMaskIntoConstraints(false);
        root.addSubview(&header);
        let sidebar =
            ui::symbol_button(mtm, "", "sidebar.left", self.target(), sel!(toggleSidebar:));
        sidebar.setToolTip(Some(ns_string!("Toggle sidebar")));
        let title = ui::label(mtm, "New chat", 13.0, true, &NSColor::labelColor());
        let status = ui::label(
            mtm,
            "Choose a model to begin",
            11.0,
            false,
            &NSColor::secondaryLabelColor(),
        );
        if let Some(cell) = title.cell() {
            cell.setLineBreakMode(NSLineBreakMode::ByTruncatingTail);
        }
        title.setContentCompressionResistancePriority_forOrientation(
            NSLayoutPriorityDefaultLow,
            NSLayoutConstraintOrientation::Horizontal,
        );
        status.setContentCompressionResistancePriority_forOrientation(
            NSLayoutPriorityDefaultLow,
            NSLayoutConstraintOrientation::Horizontal,
        );
        let settings = ui::symbol_button(
            mtm,
            "Settings",
            "slider.horizontal.3",
            self.target(),
            sel!(toggleInspector:),
        );
        for view in [&*sidebar as &NSView, &*title, &*status, &*settings] {
            view.setTranslatesAutoresizingMaskIntoConstraints(false);
            header.addSubview(view);
        }
        self.ivars().model_title.set(title.clone()).ok();
        self.ivars().model_status.set(status.clone()).ok();
        for c in [
            header
                .leadingAnchor()
                .constraintEqualToAnchor(&root.leadingAnchor()),
            header
                .trailingAnchor()
                .constraintEqualToAnchor(&root.trailingAnchor()),
            header
                .topAnchor()
                .constraintEqualToAnchor(&root.topAnchor()),
            header.heightAnchor().constraintEqualToConstant(66.0),
            sidebar
                .leadingAnchor()
                .constraintEqualToAnchor_constant(&header.leadingAnchor(), 20.0),
            sidebar
                .centerYAnchor()
                .constraintEqualToAnchor(&header.centerYAnchor()),
            sidebar.widthAnchor().constraintEqualToConstant(28.0),
            sidebar.heightAnchor().constraintEqualToConstant(28.0),
            title
                .leadingAnchor()
                .constraintEqualToAnchor_constant(&sidebar.trailingAnchor(), 12.0),
            title
                .topAnchor()
                .constraintEqualToAnchor_constant(&header.topAnchor(), 17.0),
            title
                .trailingAnchor()
                .constraintLessThanOrEqualToAnchor_constant(&settings.leadingAnchor(), -16.0),
            status
                .leadingAnchor()
                .constraintEqualToAnchor(&title.leadingAnchor()),
            status
                .topAnchor()
                .constraintEqualToAnchor_constant(&title.bottomAnchor(), 4.0),
            status
                .trailingAnchor()
                .constraintLessThanOrEqualToAnchor_constant(&settings.leadingAnchor(), -16.0),
            settings
                .trailingAnchor()
                .constraintEqualToAnchor_constant(&header.trailingAnchor(), -24.0),
            settings
                .centerYAnchor()
                .constraintEqualToAnchor(&sidebar.centerYAnchor()),
        ] {
            c.setActive(true);
        }

        let document = FlippedView::new(mtm);
        document.setTranslatesAutoresizingMaskIntoConstraints(false);
        let stack = ui::stack(
            mtm,
            NSUserInterfaceLayoutOrientation::Vertical,
            28.0,
            NSLayoutAttribute::Width,
        );
        stack.setEdgeInsets(NSEdgeInsets {
            top: 24.0,
            left: 0.0,
            bottom: 28.0,
            right: 0.0,
        });
        document.addSubview(&stack);
        for c in [
            stack
                .centerXAnchor()
                .constraintEqualToAnchor(&document.centerXAnchor()),
            stack
                .widthAnchor()
                .constraintLessThanOrEqualToConstant(BUBBLE_MAX_WIDTH),
            stack
                .widthAnchor()
                .constraintLessThanOrEqualToAnchor_constant(&document.widthAnchor(), -48.0),
            stack
                .topAnchor()
                .constraintEqualToAnchor(&document.topAnchor()),
            stack
                .bottomAnchor()
                .constraintEqualToAnchor(&document.bottomAnchor()),
        ] {
            c.setActive(true);
        }
        // Keep visual width preferences below AppKit's window-size priority:
        // a capped reading column must not cap the resizable window itself.
        let preferred = stack
            .widthAnchor()
            .constraintEqualToAnchor_constant(&document.widthAnchor(), -48.0);
        preferred.setPriority(NSLayoutPriorityDefaultLow);
        preferred.setActive(true);
        self.ivars().chat_stack.set(stack).ok();
        let scroll = ui::scroll(mtm, &document);
        scroll.setAutomaticallyAdjustsContentInsets(false);
        document
            .widthAnchor()
            .constraintEqualToAnchor(&scroll.contentView().widthAnchor())
            .setActive(true);
        self.ivars().chat_scroll.set(scroll.clone()).ok();
        root.addSubview(&scroll);

        let welcome = ui::stack(
            mtm,
            NSUserInterfaceLayoutOrientation::Vertical,
            14.0,
            NSLayoutAttribute::CenterX,
        );
        let mark = NSImageView::new(mtm);
        mark.setImage(
            NSImage::imageWithSystemSymbolName_accessibilityDescription(
                ns_string!("sparkle"),
                Some(ns_string!("Lisa")),
            )
            .as_deref(),
        );
        mark.setContentTintColor(Some(&NSColor::labelColor()));
        ui::size(&mark, 36.0, 36.0);
        welcome.addArrangedSubview(&mark);
        let welcome_title = ui::label(
            mtm,
            "A little space to think.",
            28.0,
            true,
            &NSColor::labelColor(),
        );
        welcome_title.setAlignment(NSTextAlignment::Center);
        welcome.addArrangedSubview(&welcome_title);
        let detail = ui::wrapping_label(
            mtm,
            "Choose a model in the sidebar, then make something of your ideas.",
            13.0,
            &NSColor::secondaryLabelColor(),
        );
        detail.setAlignment(NSTextAlignment::Center);
        detail.setMaximumNumberOfLines(3);
        detail.setTranslatesAutoresizingMaskIntoConstraints(false);
        welcome.addArrangedSubview(&detail);
        detail
            .widthAnchor()
            .constraintEqualToAnchor(&welcome.widthAnchor())
            .setActive(true);
        root.addSubview(&welcome);
        for c in [
            welcome
                .centerXAnchor()
                .constraintEqualToAnchor(&scroll.centerXAnchor()),
            welcome
                .centerYAnchor()
                .constraintEqualToAnchor_constant(&scroll.centerYAnchor(), -24.0),
            welcome
                .widthAnchor()
                .constraintLessThanOrEqualToConstant(460.0),
            welcome
                .widthAnchor()
                .constraintLessThanOrEqualToAnchor_constant(&root.widthAnchor(), -48.0),
        ] {
            c.setActive(true);
        }
        let width = welcome.widthAnchor().constraintEqualToConstant(460.0);
        width.setPriority(NSLayoutPriorityDefaultLow);
        width.setActive(true);
        self.ivars().chat_placeholder.set(welcome).ok();
        self.ivars().welcome_title.set(welcome_title).ok();
        self.ivars().welcome_detail.set(detail).ok();

        let composer = NSBox::new(mtm);
        composer.setBoxType(NSBoxType::Custom);
        composer.setCornerRadius(18.0);
        composer.setBorderWidth(1.0);
        composer.setBorderColor(&NSColor::separatorColor());
        composer.setFillColor(&NSColor::controlBackgroundColor());
        composer.setTranslatesAutoresizingMaskIntoConstraints(false);
        root.addSubview(&composer);
        let input = NSTextField::textFieldWithString(ns_string!(""), mtm);
        input.setBezeled(false);
        input.setBordered(false);
        input.setDrawsBackground(false);
        input.setFocusRingType(NSFocusRingType::None);
        input.setTextColor(Some(&NSColor::labelColor()));
        input.setEnabled(false);
        input.setPlaceholderString(Some(ns_string!("Message Lisa…")));
        input.setFont(Some(&NSFont::systemFontOfSize(14.0)));
        input.setMaximumNumberOfLines(3);
        if let Some(cell) = input.cell() {
            cell.setUsesSingleLineMode(false);
            cell.setWraps(true);
            cell.setScrollable(true);
        }
        input.setTranslatesAutoresizingMaskIntoConstraints(false);
        unsafe {
            input.setDelegate(Some(ProtocolObject::from_ref(self)));
        }
        composer.addSubview(&input);
        self.ivars().chat_input.set(input.clone()).ok();
        let attach = ui::symbol_button(mtm, "Attach", "plus", self.target(), sel!(attachImage:));
        attach.setEnabled(false);
        composer.addSubview(&attach);
        self.ivars().chat_attach.set(attach.clone()).ok();
        let remove = ui::symbol_button(
            mtm,
            "",
            "xmark.circle.fill",
            self.target(),
            sel!(removeAttachments:),
        );
        remove.setToolTip(Some(ns_string!("Remove attached images")));
        remove.setHidden(true);
        composer.addSubview(&remove);
        self.ivars().attachment_clear.set(remove.clone()).ok();
        let hint = ui::label(
            mtm,
            "Select a model to start",
            11.0,
            false,
            &NSColor::secondaryLabelColor(),
        );
        hint.setTranslatesAutoresizingMaskIntoConstraints(false);
        if let Some(cell) = hint.cell() {
            cell.setLineBreakMode(NSLineBreakMode::ByTruncatingTail);
        }
        composer.addSubview(&hint);
        self.ivars().composer_hint.set(hint.clone()).ok();
        let send = ui::symbol_button(
            mtm,
            "Send",
            "arrow.up.circle.fill",
            self.target(),
            sel!(sendMessage:),
        );
        send.setFont(Some(&NSFont::boldSystemFontOfSize(12.0)));
        send.setEnabled(false);
        composer.addSubview(&send);
        self.ivars().chat_send.set(send.clone()).ok();
        let keyboard = ui::label(
            mtm,
            "Return to send  ·  Shift Return for a new line",
            10.0,
            false,
            &NSColor::tertiaryLabelColor(),
        );
        keyboard.setTranslatesAutoresizingMaskIntoConstraints(false);
        root.addSubview(&keyboard);
        for c in [
            scroll
                .leadingAnchor()
                .constraintEqualToAnchor(&root.leadingAnchor()),
            scroll
                .trailingAnchor()
                .constraintEqualToAnchor(&root.trailingAnchor()),
            scroll
                .topAnchor()
                .constraintEqualToAnchor(&header.bottomAnchor()),
            scroll
                .bottomAnchor()
                .constraintEqualToAnchor_constant(&composer.topAnchor(), -12.0),
            composer
                .centerXAnchor()
                .constraintEqualToAnchor(&root.centerXAnchor()),
            composer
                .widthAnchor()
                .constraintLessThanOrEqualToConstant(760.0),
            composer
                .widthAnchor()
                .constraintLessThanOrEqualToAnchor_constant(&root.widthAnchor(), -48.0),
            composer
                .bottomAnchor()
                .constraintEqualToAnchor_constant(&root.bottomAnchor(), -42.0),
            composer.heightAnchor().constraintEqualToConstant(118.0),
            input
                .leadingAnchor()
                .constraintEqualToAnchor_constant(&composer.leadingAnchor(), 18.0),
            input
                .trailingAnchor()
                .constraintEqualToAnchor_constant(&composer.trailingAnchor(), -18.0),
            input
                .topAnchor()
                .constraintEqualToAnchor_constant(&composer.topAnchor(), 16.0),
            input.heightAnchor().constraintEqualToConstant(54.0),
            attach
                .leadingAnchor()
                .constraintEqualToAnchor_constant(&composer.leadingAnchor(), 16.0),
            attach
                .bottomAnchor()
                .constraintEqualToAnchor_constant(&composer.bottomAnchor(), -12.0),
            attach.heightAnchor().constraintEqualToConstant(24.0),
            remove
                .leadingAnchor()
                .constraintEqualToAnchor_constant(&attach.trailingAnchor(), 6.0),
            remove
                .centerYAnchor()
                .constraintEqualToAnchor(&attach.centerYAnchor()),
            remove.widthAnchor().constraintEqualToConstant(18.0),
            hint.leadingAnchor()
                .constraintEqualToAnchor_constant(&remove.trailingAnchor(), 10.0),
            hint.centerYAnchor()
                .constraintEqualToAnchor(&attach.centerYAnchor()),
            hint.trailingAnchor()
                .constraintLessThanOrEqualToAnchor_constant(&send.leadingAnchor(), -12.0),
            send.trailingAnchor()
                .constraintEqualToAnchor_constant(&composer.trailingAnchor(), -16.0),
            send.centerYAnchor()
                .constraintEqualToAnchor(&attach.centerYAnchor()),
            send.heightAnchor().constraintEqualToConstant(28.0),
            keyboard
                .centerXAnchor()
                .constraintEqualToAnchor(&composer.centerXAnchor()),
            keyboard
                .topAnchor()
                .constraintEqualToAnchor_constant(&composer.bottomAnchor(), 12.0),
        ] {
            c.setActive(true);
        }
        let width = composer
            .widthAnchor()
            .constraintEqualToAnchor_constant(&root.widthAnchor(), -48.0);
        width.setPriority(NSLayoutPriorityDefaultLow);
        width.setActive(true);

        let cmd_menu = RoundedView::new(mtm, &NSColor::controlBackgroundColor(), 12.0);
        cmd_menu.setTranslatesAutoresizingMaskIntoConstraints(false);
        cmd_menu.setHidden(true);
        root.addSubview(&cmd_menu);
        let cmd_stack = ui::stack(
            mtm,
            NSUserInterfaceLayoutOrientation::Vertical,
            2.0,
            NSLayoutAttribute::Width,
        );
        cmd_stack.setEdgeInsets(NSEdgeInsets {
            top: 8.0,
            left: 8.0,
            bottom: 8.0,
            right: 8.0,
        });
        cmd_menu.addSubview(&cmd_stack);
        for c in [
            cmd_stack
                .leadingAnchor()
                .constraintEqualToAnchor(&cmd_menu.leadingAnchor()),
            cmd_stack
                .trailingAnchor()
                .constraintEqualToAnchor(&cmd_menu.trailingAnchor()),
            cmd_stack
                .topAnchor()
                .constraintEqualToAnchor(&cmd_menu.topAnchor()),
            cmd_stack
                .bottomAnchor()
                .constraintEqualToAnchor(&cmd_menu.bottomAnchor()),
            cmd_menu
                .leadingAnchor()
                .constraintEqualToAnchor(&composer.leadingAnchor()),
            cmd_menu
                .bottomAnchor()
                .constraintEqualToAnchor_constant(&composer.topAnchor(), -8.0),
            cmd_menu.widthAnchor().constraintEqualToConstant(320.0),
        ] {
            c.setActive(true);
        }
        let mut rows = Vec::new();
        let mut backgrounds = Vec::new();
        for (i, (name, description)) in CHAT_COMMANDS.iter().enumerate() {
            let row = NSView::new(mtm);
            row.setTranslatesAutoresizingMaskIntoConstraints(false);
            let bg = RoundedView::new(mtm, &NSColor::quaternaryLabelColor(), 6.0);
            bg.setTranslatesAutoresizingMaskIntoConstraints(false);
            bg.setHidden(true);
            row.addSubview(&bg);
            let button = ui::button(mtm, "", Some(self.target()), Some(sel!(runCommand:)), false);
            button.setTag(i as NSInteger);
            button.setAlignment(NSTextAlignment::Left);
            button.setAttributedTitle(&command_row_title(name, description, false));
            button.setTranslatesAutoresizingMaskIntoConstraints(false);
            row.addSubview(&button);
            for c in [
                bg.leadingAnchor()
                    .constraintEqualToAnchor(&row.leadingAnchor()),
                bg.trailingAnchor()
                    .constraintEqualToAnchor(&row.trailingAnchor()),
                bg.topAnchor().constraintEqualToAnchor(&row.topAnchor()),
                bg.bottomAnchor()
                    .constraintEqualToAnchor(&row.bottomAnchor()),
                button
                    .leadingAnchor()
                    .constraintEqualToAnchor_constant(&row.leadingAnchor(), 8.0),
                button
                    .trailingAnchor()
                    .constraintEqualToAnchor_constant(&row.trailingAnchor(), -8.0),
                button.topAnchor().constraintEqualToAnchor(&row.topAnchor()),
                button
                    .bottomAnchor()
                    .constraintEqualToAnchor(&row.bottomAnchor()),
                row.heightAnchor().constraintEqualToConstant(32.0),
            ] {
                c.setActive(true);
            }
            cmd_stack.addArrangedSubview(&row);
            rows.push(button);
            backgrounds.push(bg);
        }
        *self.ivars().cmd_rows.borrow_mut() = rows;
        *self.ivars().cmd_bgs.borrow_mut() = backgrounds;
        self.ivars().cmd_menu.set(cmd_menu).ok();
        root.into_super()
    }

    fn refresh_chat_chrome(&self) {
        let loaded = self.ivars().model_loaded.get();
        let busy = self.ivars().busy.get();
        if let Some(button) = self.ivars().attachment_clear.get() {
            button.setHidden(self.ivars().attachments.borrow().is_empty());
        }
        let image_steps = self.ivars().image_mode.get();
        let image = image_steps.is_some();
        let active = self.ivars().active_repo.borrow();
        let chat_id = self.ivars().active_chat.borrow().clone();
        let chat_title = self
            .ivars()
            .chats
            .borrow()
            .iter()
            .find(|chat| Some(&chat.id) == chat_id.as_ref())
            .map(chats::Chat::title);
        if let Some(title) = self.ivars().model_title.get() {
            title.setStringValue(&NSString::from_str(
                chat_title.as_deref().unwrap_or("New chat"),
            ));
            title.setToolTip(active.as_deref().map(NSString::from_str).as_deref());
        }
        let status = if busy {
            if image {
                "Generating an image…"
            } else {
                "Lisa is responding…"
            }
        } else if loaded {
            if self.ivars().serving.get() {
                "Local model · API running"
            } else {
                "Local model · Ready"
            }
        } else if active.is_some() {
            "Loading model…"
        } else {
            "Choose a model to begin"
        };
        if let Some(label) = self.ivars().model_status.get() {
            let status = active
                .as_deref()
                .map(|model| {
                    format!(
                        "{} · {status}",
                        catalog::find(model)
                            .map_or_else(|| display_name(model), |entry| entry.name,)
                    )
                })
                .unwrap_or_else(|| status.into());
            label.setStringValue(&NSString::from_str(&status));
        }
        if let Some(window) = self.ivars().window.get() {
            window.setTitle(&NSString::from_str(
                &chat_title
                    .map(|title| format!("Lisa — {title}"))
                    .unwrap_or_else(|| "Lisa".into()),
            ));
        }
        if let Some(label) = self.ivars().composer_hint.get() {
            let hint = if loaded {
                if let Some(steps) = image_steps {
                    format!("512 × 512 · {steps} steps")
                } else {
                    "On your Mac".into()
                }
            } else {
                status.into()
            };
            label.setStringValue(&NSString::from_str(&hint));
        }
        if let Some(label) = self.ivars().welcome_title.get() {
            label.setStringValue(&NSString::from_str(if image && loaded {
                "Imagine something new."
            } else {
                "A little space to think."
            }));
        }
        if let Some(label) = self.ivars().welcome_detail.get() {
            label.setStringValue(&NSString::from_str(if !loaded {
                "Choose a model in the sidebar, then make something of your ideas."
            } else if image {
                "Describe what you see in your mind. Lisa will create it here, on your Mac."
            } else if !self.ivars().supports_images.get() {
                "Ask a question, work through an idea, or write some code."
            } else {
                "Ask a question, work through an idea, or bring an image into the conversation."
            }));
        }
        if let Some(send) = self.ivars().chat_send.get() {
            send.setEnabled(loaded && !busy);
            send.setTitle(&NSString::from_str(if busy {
                "Working…"
            } else if image {
                "Generate"
            } else {
                "Send"
            }));
        }
        if let Some(button) = self.ivars().new_chat.get() {
            button.setEnabled(!self.chat_locked());
        }
        for button in self.ivars().chat_buttons.borrow().iter() {
            button.setEnabled(!self.chat_locked());
        }
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
        let root = RoundedView::new(mtm, &NSColor::windowBackgroundColor(), 0.0);

        let heading = ui::label(mtm, "Model settings", 13.0, true, &NSColor::labelColor());
        heading.setTranslatesAutoresizingMaskIntoConstraints(false);
        root.addSubview(&heading);
        heading
            .leadingAnchor()
            .constraintEqualToAnchor_constant(&root.leadingAnchor(), 20.0)
            .setActive(true);
        heading
            .topAnchor()
            .constraintEqualToAnchor_constant(&root.topAnchor(), 40.0)
            .setActive(true);
        let close = ui::symbol_button(mtm, "", "xmark", self.target(), sel!(toggleInspector:));
        close.setToolTip(Some(ns_string!("Close settings")));
        root.addSubview(&close);
        close
            .trailingAnchor()
            .constraintEqualToAnchor_constant(&root.trailingAnchor(), -20.0)
            .setActive(true);
        close
            .centerYAnchor()
            .constraintEqualToAnchor(&heading.centerYAnchor())
            .setActive(true);

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
        switch.setEnabled(false);
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
        icon.widthAnchor()
            .constraintEqualToConstant(28.0)
            .setActive(true);
        icon.heightAnchor()
            .constraintEqualToConstant(28.0)
            .setActive(true);
        glyph
            .centerXAnchor()
            .constraintEqualToAnchor(&icon.centerXAnchor())
            .setActive(true);
        glyph
            .centerYAnchor()
            .constraintEqualToAnchor(&icon.centerYAnchor())
            .setActive(true);
        glyph
            .widthAnchor()
            .constraintEqualToConstant(16.0)
            .setActive(true);
        glyph
            .heightAnchor()
            .constraintEqualToConstant(16.0)
            .setActive(true);
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
        let state = ui::label(
            mtm,
            "Unavailable",
            12.0,
            false,
            &NSColor::secondaryLabelColor(),
        );
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
        dot.widthAnchor()
            .constraintEqualToConstant(9.0)
            .setActive(true);
        dot.heightAnchor()
            .constraintEqualToConstant(9.0)
            .setActive(true);
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

        let status = ui::wrapping_label(
            mtm,
            "Start the server to connect another app.",
            11.0,
            &NSColor::secondaryLabelColor(),
        );
        status.setTranslatesAutoresizingMaskIntoConstraints(false);
        inner.addArrangedSubview(&status);
        status
            .widthAnchor()
            .constraintEqualToAnchor(&inner.widthAnchor())
            .setActive(true);
        self.ivars().serve_status.set(status).ok();
        let copy = ui::symbol_button(
            mtm,
            "Copy API URL",
            "doc.on.doc",
            self.target(),
            sel!(copyEndpoint:),
        );
        inner.addArrangedSubview(&copy);

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
        port.trailingAnchor()
            .constraintEqualToAnchor_constant(&port_card.trailingAnchor(), -12.0)
            .setActive(true);
        port.centerYAnchor()
            .constraintEqualToAnchor(&port_card.centerYAnchor())
            .setActive(true);
        port.leadingAnchor()
            .constraintGreaterThanOrEqualToAnchor_constant(&port_label.trailingAnchor(), 8.0)
            .setActive(true);
        port.widthAnchor()
            .constraintEqualToConstant(120.0)
            .setActive(true);

        let sys_card = RoundedView::new(mtm, &NSColor::quaternaryLabelColor(), 10.0);
        sys_card.setTranslatesAutoresizingMaskIntoConstraints(false);
        body.addArrangedSubview(&sys_card);
        sys_card
            .trailingAnchor()
            .constraintEqualToAnchor_constant(&root.trailingAnchor(), -18.0)
            .setActive(true);

        let sys_label = ui::label(mtm, "Instructions", 12.0, true, &NSColor::labelColor());
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
        sys_text.setFrame(NSRect::new(
            NSPoint::new(0.0, 0.0),
            NSSize::new(220.0, 72.0),
        ));
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

        body.leadingAnchor()
            .constraintEqualToAnchor_constant(&root.leadingAnchor(), 18.0)
            .setActive(true);
        body.trailingAnchor()
            .constraintEqualToAnchor_constant(&root.trailingAnchor(), -18.0)
            .setActive(true);
        body.topAnchor()
            .constraintEqualToAnchor_constant(&root.topAnchor(), 86.0)
            .setActive(true);

        root.into_super()
    }

    fn build_logs(&self, mtm: MainThreadMarker) -> Retained<NSView> {
        let root = RoundedView::new(mtm, &NSColor::textBackgroundColor(), 0.0);

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
        text.setFrame(NSRect::new(
            NSPoint::new(0.0, 0.0),
            NSSize::new(420.0, 600.0),
        ));
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

        root.into_super()
    }
}
