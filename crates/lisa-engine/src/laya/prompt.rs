//! Tokenizer and request adaptation for the Laya decision model.
use super::Laya;
use anyhow::{Context, Result};
pub use lisa_models::laya::prompt::*;
use serde_json::{Value, json};
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
        let mut opt_budget =
            head_max_len as i64 - opt_ids.iter().map(|o| o.len() as i64).sum::<i64>();
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
        let qmap = questions
            .as_object()
            .context("questions must be a dictionary")?;
        let mut answers = serde_json::Map::new();
        let mut input_tokens = 0usize;
        for (qid, qdef) in qmap {
            let q = Question::parse(qdef).with_context(|| format!("question {qid:?}"))?;
            let (ids, markers) = self.build_sequence(state, &q)?;
            input_tokens += ids.len();
            let k = markers.len();
            anyhow::ensure!(k > 0, "question {qid:?} has no options");
            let (logits, action) = self.decide(&ids, q.t.index(), &markers)?;

            let ans = answer(self.config(), &q, &logits, &action);
            answers.insert(qid.clone(), ans);
        }
        Ok(json!({
            "model": "laya",
            "answers": Value::Object(answers),
            "usage": { "input_tokens": input_tokens, "output_tokens": 0 },
        }))
    }
}
