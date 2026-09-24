//! Laya prompt construction and answer derivation.
//!
//! Mirrors the upstream `rl_agent` layout (and the `laya-mlx` port):
//! `[CLS] <type> question: <instructions> [SEP] [MASK] opt0 [MASK] opt1 … [SEP]
//! <state> [SEP]`, with one marker per option. Answers are the temperature-scaled
//! softmax over the marker logits; `choice`/`score`/`noul` each report a
//! different view of it.

use anyhow::{Context, Result};
use serde_json::{json, Value};

use super::Laya;

/// Question types and their `qtype` index (the `type_emb` row).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QType {
    Choice = 0,
    Score = 1,
    Noul = 2,
}

impl QType {
    pub fn parse(s: &str) -> Result<Self> {
        match s {
            "choice" => Ok(QType::Choice),
            "score" => Ok(QType::Score),
            "noul" => Ok(QType::Noul),
            other => anyhow::bail!("unknown question type {other:?}; expected choice, score, or noul"),
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            QType::Choice => "choice",
            QType::Score => "score",
            QType::Noul => "noul",
        }
    }
    fn index(self) -> i32 {
        self as i32
    }
}

/// A validated, normalized question.
pub struct Question {
    pub t: QType,
    pub ins: String,
    pub crit: Value,
}

impl Question {
    /// Validate and normalize a question definition (upstream `_to_internal`).
    pub fn parse(v: &Value) -> Result<Self> {
        let obj = v.as_object().context("each question must be a dictionary")?;
        let t = QType::parse(obj.get("type").and_then(|v| v.as_str()).context("question is missing type")?)?;
        let ins = obj.get("instructions").context("question is missing instructions")?;
        let ins = match ins {
            Value::String(s) => s.clone(),
            other => serde_json::to_string(other)?,
        };
        let crit = obj.get("criteria").cloned().unwrap_or(Value::Null);
        match t {
            QType::Choice => {
                let c = normalize_choice(&crit)?;
                Ok(Self { t, ins, crit: c })
            }
            QType::Score => {
                anyhow::ensure!(
                    crit.as_array().is_some_and(|a| !a.is_empty()),
                    "score criteria must be a nonempty list"
                );
                Ok(Self { t, ins, crit })
            }
            QType::Noul => {
                anyhow::ensure!(
                    crit.is_null() || crit.is_object(),
                    "noul criteria must be a dictionary with false/true descriptions"
                );
                Ok(Self { t, ins, crit })
            }
        }
    }

    /// Option texts in label-index order (upstream `render_options`).
    pub fn render_options(&self) -> Result<Vec<String>> {
        match self.t {
            QType::Choice => {
                let obj = self.crit.as_object().context("choice criteria must be a dictionary")?;
                let mut out = Vec::with_capacity(obj.len());
                for (k, v) in obj {
                    out.push(if v.is_null() || v.as_str() == Some("") {
                        k.clone()
                    } else {
                        format!("{k}: {}", render_criterion(v))
                    });
                }
                Ok(out)
            }
            QType::Score => {
                let arr = self.crit.as_array().context("score criteria must be a list")?;
                Ok(arr
                    .iter()
                    .enumerate()
                    .map(|(i, c)| format!("level {i}: {}", render_criterion(c)))
                    .collect())
            }
            QType::Noul => {
                let obj = self.crit.as_object();
                let get = |k: &str| obj.and_then(|o| o.get(k));
                let false_c = get("false");
                let true_c = get("true");
                Ok(vec![
                    format!("false: {}", noul_text(false_c, "no, the statement does not hold")),
                    format!("true: {}", noul_text(true_c, "yes, the statement holds")),
                ])
            }
        }
    }
}

fn normalize_choice(crit: &Value) -> Result<Value> {
    let obj = match crit {
        Value::Array(a) => {
            let mut map = serde_json::Map::new();
            for c in a {
                let s = c.as_str().context("choice labels must be strings")?;
                anyhow::ensure!(!map.contains_key(s), "choice labels must be unique");
                map.insert(s.to_string(), Value::String(s.to_string()));
            }
            Value::Object(map)
        }
        Value::Object(o) => Value::Object(o.clone()),
        _ => anyhow::bail!("choice criteria must be a nonempty dictionary or list"),
    };
    anyhow::ensure!(obj.as_object().is_some_and(|o| !o.is_empty()), "choice criteria must be nonempty");
    Ok(obj)
}

fn render_criterion(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        other => serde_json::to_string(other).unwrap_or_default(),
    }
}

fn noul_text(c: Option<&Value>, fallback: &str) -> String {
    match c {
        Some(Value::String(s)) if !s.is_empty() => s.clone(),
        Some(v) if !v.is_null() => render_criterion(v),
        _ => fallback.to_string(),
    }
}

/// `1 - H(p) / log(k)`, clipped to `[0, 1]` (upstream `confidence_from_probs`).
pub fn confidence_from_probs(p: &[f32]) -> f32 {
    let k = p.len();
    if k < 2 {
        return 1.0;
    }
    let ent: f32 = -p.iter().map(|&x| x * x.max(1e-12).ln()).sum::<f32>();
    (1.0 - ent / (k as f32).ln()).clamp(0.0, 1.0)
}

/// The `temperature_by_options` bucket key (upstream `temp_bucket`).
pub fn temp_bucket(qt: QType, k: usize) -> String {
    let size = if k <= 2 {
        "2"
    } else if k <= 5 {
        "3-5"
    } else if k <= 10 {
        "6-10"
    } else {
        "11+"
    };
    format!("{}:{}", qt.name(), size)
}

/// The only option-count buckets the lookup recognises.
pub const TEMP_BUCKET_SIZES: [&str; 4] = ["2", "3-5", "6-10", "11+"];

/// Reject a `temperature_by_options` key that the size lookup can never select.
/// The four canonical size buckets are the only valid sizes.
pub fn validate_bucket(key: &str) -> Result<()> {
    let (ty, size) = key
        .split_once(':')
        .ok_or_else(|| anyhow::anyhow!("temperature bucket {key:?} must be TYPE:SIZE"))?;
    anyhow::ensure!(
        matches!(ty, "choice" | "score" | "noul"),
        "temperature bucket {key:?}: type must be choice, score, or noul"
    );
    anyhow::ensure!(
        TEMP_BUCKET_SIZES.contains(&size),
        "temperature bucket {key:?}: size {size:?} must be one of {}",
        TEMP_BUCKET_SIZES.join(", ")
    );
    Ok(())
}

/// The effective calibration scale for `qtype` with `k` options: the bucketed
/// temperature when present, else the per-type one (`temp_bucket`'s lookup).
pub fn effective_scale(cfg: &super::LayaConfig, qt: QType, k: usize) -> (String, f32) {
    let bucket = temp_bucket(qt, k);
    let scale = cfg
        .temperature_by_options
        .get(&bucket)
        .copied()
        .unwrap_or(cfg.temperature[qt as usize]);
    (bucket, scale)
}

pub fn softmax(z: &[f32]) -> Vec<f32> {
    let m = z.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let e: Vec<f32> = z.iter().map(|&x| (x - m).exp()).collect();
    let s: f32 = e.iter().sum();
    e.iter().map(|&x| x / s.max(1e-30)).collect()
}

impl Laya {
    fn encode_text(&self, text: &str) -> Result<Vec<u32>> {
        Ok(self
            .tokenizer()
            .inner
            .encode(text, false)
            .map_err(|e| anyhow::anyhow!("laya encode: {e}"))?
            .get_ids()
            .to_vec())
    }

    fn serialize_state(state: &Value) -> String {
        match state {
            Value::String(s) => s.clone(),
            other => serde_json::to_string(other).unwrap_or_default(),
        }
    }

    /// Build the `(input_ids, marker_pos)` for one question (upstream
    /// `build_sequence`).
    pub fn build_sequence(&self, state: &Value, q: &Question) -> Result<(Vec<u32>, Vec<i32>)> {
        let cfg = self.config();
        let max_len = cfg.max_len;
        let head_max_len = cfg.head_max_len;
        let mask_id = cfg.mask_id;
        let mask_tok = cfg.mask_token.clone();

        let opts = q.render_options()?;
        let ins = q.ins.replace(&mask_tok, " ");
        let mut head_ids = self.encode_text(&format!("{} question: {}", q.t.name(), ins))?;

        let mut opt_ids: Vec<Vec<u32>> = Vec::with_capacity(opts.len());
        for o in &opts {
            let body = o.replace(&mask_tok, " ");
            let mut id = vec![mask_id];
            id.extend(self.encode_text(&format!(" {body}"))?.into_iter().take(48));
            opt_ids.push(id);
        }
        let mut opt_budget = head_max_len as i64 - opt_ids.iter().map(|o| o.len() as i64).sum::<i64>();
        if opt_budget < 16 {
            let per = (4).max((head_max_len.saturating_sub(16)) / opt_ids.len().max(1));
            for o in &mut opt_ids {
                o.truncate(per);
            }
            opt_budget = head_max_len as i64 - opt_ids.iter().map(|o| o.len() as i64).sum::<i64>();
        }
        head_ids.truncate((8).max(opt_budget.max(0) as usize));

        let mut ids: Vec<u32> = vec![cfg.cls_id];
        ids.extend(head_ids);
        ids.push(cfg.sep_id);
        let mut markers: Vec<i32> = Vec::with_capacity(opt_ids.len());
        for o in &opt_ids {
            markers.push(ids.len() as i32);
            ids.extend(o);
        }
        ids.push(cfg.sep_id);

        let room = max_len.saturating_sub(ids.len() + 1);
        let mut st = self.encode_text(&Self::serialize_state(state).replace(&mask_tok, " "))?;
        st.truncate(room);
        ids.extend(st);
        ids.push(cfg.sep_id);
        ids.truncate(max_len);
        markers.retain(|&m| (m as usize) < max_len);
        Ok((ids, markers))
    }

    /// Run one state + questions and produce the upstream answer JSON.
    pub fn system_one(&self, state: &Value, questions: &Value) -> Result<Value> {
        let qmap = questions.as_object().context("questions must be a dictionary")?;
        let mut answers = serde_json::Map::new();
        let mut input_tokens = 0usize;
        for (qid, qdef) in qmap {
            let q = Question::parse(qdef).with_context(|| format!("question {qid:?}"))?;
            let (ids, markers) = self.build_sequence(state, &q)?;
            input_tokens += ids.len();
            let k = markers.len();
            anyhow::ensure!(k > 0, "question {qid:?} has no options");
            let (logits, action) = self.decide(&ids, q.t.index(), &markers)?;

            let scale = effective_scale(self.config(), q.t, k);
            let (bucket, scale) = scale;
            let z: Vec<f32> = logits.iter().map(|&x| x / scale).collect();
            let p = softmax(&z);
            let act = softmax(&action);

            let mut ans = json!({
                "type": q.t.name(),
                "confidence": round4(confidence_from_probs(&p)),
                "action": { "act_probability": round4(act.first().copied().unwrap_or(0.0)) },
                "temperature": round4(scale),
                "bucket": bucket,
            });
            match q.t {
                QType::Choice => {
                    let labels: Vec<String> = q
                        .crit
                        .as_object()
                        .map(|o| o.keys().cloned().collect())
                        .unwrap_or_default();
                    let best = p
                        .iter()
                        .enumerate()
                        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                        .map(|(i, _)| i)
                        .unwrap_or(0);
                    let probs: serde_json::Map<String, Value> = labels
                        .iter()
                        .zip(p.iter())
                        .map(|(l, &v)| (l.clone(), json!(round4(v))))
                        .collect();
                    ans["choice"] = json!(labels.get(best).cloned().unwrap_or_default());
                    ans["probabilities"] = Value::Object(probs);
                }
                QType::Score => {
                    let legend: serde_json::Map<String, Value> = q
                        .crit
                        .as_array()
                        .map(|a| a.iter().enumerate().map(|(i, c)| (i.to_string(), c.clone())).collect())
                        .unwrap_or_default();
                    let probs: serde_json::Map<String, Value> = p
                        .iter()
                        .enumerate()
                        .map(|(i, &v)| (i.to_string(), json!(round4(v))))
                        .collect();
                    let expected: f32 = p.iter().enumerate().map(|(i, &v)| i as f32 * v).sum();
                    ans["score"] = json!(round4(expected));
                    ans["legend"] = Value::Object(legend);
                    ans["probabilities"] = Value::Object(probs);
                }
                QType::Noul => {
                    let yes = p.get(1).copied().unwrap_or(0.0);
                    ans["noul"] = json!(round4(yes));
                    ans["confidence"] = json!(round4(yes.max(1.0 - yes)));
                }
            }
            answers.insert(qid.clone(), ans);
        }
        Ok(json!({
            "model": "laya",
            "answers": Value::Object(answers),
            "usage": { "input_tokens": input_tokens, "output_tokens": 0 },
        }))
    }
}

fn round4(x: f32) -> f64 {
    (x as f64 * 1e4).round() / 1e4
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_canonical_buckets_and_rejects_the_rest() {
        for ok in ["choice:2", "choice:3-5", "score:6-10", "noul:11+"] {
            assert!(validate_bucket(ok).is_ok(), "{ok} should be allowed");
        }
        let err = validate_bucket("choice:21-40").unwrap_err().to_string();
        assert!(err.contains("choice:21-40") && err.contains("11+"), "{err}");
        assert!(validate_bucket("bogus:2").is_err());
        assert!(validate_bucket("choice").is_err());
    }

    #[test]
    fn bucket_sizes_match_temp_bucket() {
        for (k, size) in [(2usize, "2"), (5, "3-5"), (10, "6-10"), (40, "11+")] {
            let b = temp_bucket(QType::Choice, k);
            assert_eq!(b.split_once(':').unwrap().1, size);
            assert!(TEMP_BUCKET_SIZES.contains(&size));
        }
    }
}
