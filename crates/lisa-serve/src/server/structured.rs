//! Structured output: schema instructions, JSON extraction, repair and validation.
use super::*;

/// The JSON Schema (if any) a `response_format` asks for: `Some(schema)` for a
/// structured-output request, an empty-object schema for `json_object`.
pub(super) fn response_format_of(body: &Value) -> Option<Value> {
    let rf = body.get("response_format")?;
    match rf.get("type").and_then(|t| t.as_str())? {
        "json_object" => Some(json!({ "type": "object" })),
        "json_schema" => rf
            .get("json_schema")
            .and_then(|j| j.get("schema"))
            .or_else(|| rf.get("schema"))
            .cloned()
            .or_else(|| Some(json!({ "type": "object" }))),
        _ => None,
    }
}

/// A system instruction that pushes the model into emitting the requested JSON
/// shape (the engine has no grammar-constrained decoding).
pub(super) fn json_instruction(schema: Option<&Value>) -> String {
    match schema {
        Some(sc) if !sc.as_object().map(|o| o.is_empty()).unwrap_or(false) => format!(
            "Respond with a single valid JSON value that conforms to this JSON Schema and \
nothing else. Do not include prose, explanations, or markdown code fences.\n{sc}"
        ),
        _ => "Respond with a single valid JSON object and nothing else. Do not include \
prose, explanations, or markdown code fences."
            .to_string(),
    }
}

/// Pull the JSON value out of a model reply, stripping markdown fences and any
/// surrounding prose. Returns a compact serialization, or `None`.
pub(super) fn extract_json(text: &str) -> Option<String> {
    let mut t = text.trim();
    if let Some(r) = t.strip_prefix("```json").or_else(|| t.strip_prefix("```")) {
        t = r.trim_start();
    }
    if let Some(r) = t.strip_suffix("```") {
        t = r.trim_end();
    }
    if let Ok(v) = serde_json::from_str::<Value>(t) {
        return Some(v.to_string());
    }
    if let Some(o) = find_json_object(t) {
        return Some(Value::Object(o).to_string());
    }
    find_json_array(t).map(|a| a.to_string())
}

/// Truncate `text` at the earliest stop sequence.
pub(super) fn strip_stops<'a>(text: &'a str, stops: &[String]) -> &'a str {
    match stops
        .iter()
        .filter(|s| !s.is_empty())
        .filter_map(|s| text.find(s.as_str()))
        .min()
    {
        Some(cut) => &text[..cut],
        None => text,
    }
}

/// Drop a leading ` thinking…</think>` block so a JSON body is not scanned
/// through marker text (marker strings inside the JSON are data, not markers).
pub(super) fn strip_think_prefix(text: &str) -> &str {
    const OPEN: &str = "\u{3c}think\u{3e}";
    const CLOSE: &str = "\u{3c}/think\u{3e}";
    let t = text.trim_start();
    if t.starts_with(OPEN)
        && let Some(i) = t.find(CLOSE)
    {
        return t[i + CLOSE.len()..].trim_start();
    }
    text
}

/// Nudge a parsed value toward a JSON Schema where the model only approximated
/// it: fill missing required properties, drop extras, coerce primitive types,
/// and complete a string `enum` value the model truncated (an enum constrains
/// the value, so the schema decides it — the same output a constrained decoder
/// would emit).
pub(super) fn repair_json(v: &mut Value, schema: &Value) {
    match schema.get("type").and_then(|t| t.as_str()) {
        Some("object") => {
            let Value::Object(mut obj) = std::mem::take(v) else {
                *v = json!({});
                return;
            };
            if let Some(props) = schema.get("properties").and_then(|p| p.as_object()) {
                for (k, ps) in props {
                    if obj.contains_key(k) {
                        repair_json(obj.get_mut(k).unwrap(), ps);
                    } else if schema
                        .get("required")
                        .and_then(|r| r.as_array())
                        .map(|r| r.iter().any(|x| x.as_str() == Some(k.as_str())))
                        .unwrap_or(false)
                    {
                        let mut nv = Value::Null;
                        repair_json(&mut nv, ps);
                        obj.insert(k.clone(), nv);
                    }
                }
            }
            if schema.get("additionalProperties").and_then(|a| a.as_bool()) == Some(false)
                && let Some(props) = schema.get("properties").and_then(|p| p.as_object())
            {
                obj.retain(|k, _| props.contains_key(k));
            }
            *v = Value::Object(obj);
        }
        Some("array") => {
            if let (Value::Array(a), Some(items)) = (v, schema.get("items")) {
                for x in a.iter_mut() {
                    repair_json(x, items);
                }
            }
        }
        Some("string") => {
            if let Some(e) = schema.get("enum").and_then(|e| e.as_array()) {
                if !e.iter().any(|x| x == &*v) {
                    let cur = v.as_str().unwrap_or("");
                    let pick = if e.len() == 1 {
                        e.first()
                    } else {
                        e.iter()
                            .find(|x| x.as_str().map(|s| s.starts_with(cur)).unwrap_or(false))
                    };
                    if let Some(p) = pick {
                        *v = p.clone();
                    }
                }
            } else if !v.is_string() {
                *v = Value::String(match &*v {
                    Value::Null => String::new(),
                    other => other.to_string(),
                });
            }
        }
        Some("integer") => {
            if let Some(n) = v.as_f64() {
                *v = json!(n as i64);
            } else if let Some(n) = v.as_str().and_then(|s| s.trim().parse::<i64>().ok()) {
                *v = json!(n);
            }
        }
        Some("number") => {
            if let Some(n) = v.as_str().and_then(|s| s.trim().parse::<f64>().ok()) {
                *v = json!(n);
            }
        }
        Some("boolean") => {
            if let Some(b) = v.as_str().and_then(|s| match s {
                "true" => Some(true),
                "false" => Some(false),
                _ => None,
            }) {
                *v = json!(b);
            }
        }
        _ => {}
    }
}

/// Validate a parsed JSON value against (the subset of JSON Schema the server
/// honors): types, `required` object properties, `enum`, and nested
/// `properties`/`items`. Returns human-readable violations; empty == valid.
/// This backs the validate+retry path for `response_format` — the engine has
/// no grammar-constrained decoding, so conformance is checked after
/// generation (repair first, then one corrective retry, documented in the
/// serving-api spec).
pub(super) fn json_schema_errors(v: &Value, schema: &Value) -> Vec<String> {
    let mut errs = Vec::new();
    let ty = schema.get("type").and_then(|t| t.as_str());
    let type_ok = |v: &Value, ty: &str| match ty {
        "object" => v.is_object(),
        "array" => v.is_array(),
        "string" => v.is_string(),
        "integer" => v.is_i64() || v.is_u64(),
        "number" => v.is_number(),
        "boolean" => v.is_boolean(),
        "null" => v.is_null(),
        _ => true,
    };
    if let Some(ty) = ty
        && !type_ok(v, ty)
    {
        errs.push(format!("expected {ty}, got {}", type_name_of(v)));
        return errs;
    }
    if let Some(e) = schema.get("enum").and_then(|e| e.as_array())
        && !e.iter().any(|x| x == v)
    {
        errs.push(format!("value {v} not in enum"));
    }
    match ty {
        Some("object") | None if v.is_object() => {
            if let Some(props) = schema.get("properties").and_then(|p| p.as_object()) {
                for (k, ps) in props {
                    if let Some(x) = v.get(k) {
                        for e in json_schema_errors(x, ps) {
                            errs.push(format!("{k}: {e}"));
                        }
                    }
                }
            }
            if let Some(req) = schema.get("required").and_then(|r| r.as_array()) {
                for r in req {
                    if let Some(k) = r.as_str()
                        && !v.get(k).map(|x| !x.is_null()).unwrap_or(false)
                    {
                        errs.push(format!("missing required property {k:?}"));
                    }
                }
            }
        }
        Some("array") => {
            if let (Some(items), Value::Array(a)) = (schema.get("items"), v) {
                for (i, x) in a.iter().enumerate() {
                    for e in json_schema_errors(x, items) {
                        errs.push(format!("[{i}]: {e}"));
                    }
                }
            }
        }
        _ => {}
    }
    errs
}

pub(super) fn type_name_of(v: &Value) -> &'static str {
    match v {
        Value::Null => "null",
        Value::Bool(_) => "boolean",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}

pub(super) fn find_json_array(s: &str) -> Option<Value> {
    let bytes = s.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'[' {
            let mut depth = 0i32;
            let mut in_str = false;
            let mut esc = false;
            let mut j = i;
            while j < bytes.len() {
                let c = bytes[j];
                if in_str {
                    if esc {
                        esc = false;
                    } else if c == b'\\' {
                        esc = true;
                    } else if c == b'"' {
                        in_str = false;
                    }
                } else if c == b'"' {
                    in_str = true;
                } else if c == b'[' {
                    depth += 1;
                } else if c == b']' {
                    depth -= 1;
                    if depth == 0 {
                        if let Ok(v) = serde_json::from_str::<Value>(&s[i..=j]) {
                            return Some(v);
                        }
                        break;
                    }
                }
                j += 1;
            }
        }
        i += 1;
    }
    None
}
