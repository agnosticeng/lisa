//! GGUF metadata adaptation. Model geometry remains in lisa-models.
use anyhow::{Context, Result, ensure};
use lisa_gguf::{Header, Value};
pub fn config(h: &Header) -> Result<lisa_models::inkling2::config::Config> {
    ensure!(
        h.metadata
            .get("tokenizer.ggml.eos_token_id")
            .and_then(Value::as_u64)
            == Some(200006),
        "unsupported Inkling 2 EOS token"
    );
    for name in [
        "tokenizer.ggml.add_bos_token",
        "tokenizer.ggml.add_eos_token",
    ] {
        ensure!(
            matches!(h.metadata.get(name), Some(Value::Bool(false))),
            "unsupported Inkling 2 {name}"
        );
    }
    ensure!(
        h.metadata
            .get("inkling2.attention.sliding_window_pattern")
            .and_then(Value::as_array)
            .is_some_and(|a| a.len() == 48
                && a.iter()
                    .enumerate()
                    .all(|(i, v)| matches!(v,Value::Bool(b) if *b==(i%6!=5)))),
        "unsupported Inkling attention schedule"
    );
    lisa_models::inkling2::config::Config::from_metadata(&Metadata {
        header: h,
        prefix: "inkling2",
    })
}
pub struct Metadata<'a> {
    pub header: &'a Header,
    pub prefix: &'a str,
}

impl Metadata<'_> {
    pub fn get(&self, names: &[&str]) -> Result<&Value> {
        let mut result = None;
        for name in names {
            if let Some(value) = self.header.metadata.get(&format!("{}.{name}", self.prefix)) {
                if let Some(previous) = result {
                    ensure!(
                        previous == value,
                        "conflicting Inkling 2 metadata aliases {names:?}"
                    );
                }
                result = Some(value);
            }
        }
        result.with_context(|| format!("missing Inkling 2 metadata {names:?}"))
    }
}
impl lisa_models::metadata::Metadata for Metadata<'_> {
    fn architecture(&self) -> Result<&str> {
        self.header.architecture()
    }
    fn contains(&self, name: &str) -> bool {
        self.header
            .metadata
            .contains_key(&format!("{}.{name}", self.prefix))
    }
    fn vocabulary_len(&self) -> Result<usize> {
        Ok(self
            .header
            .metadata
            .get("tokenizer.ggml.tokens")
            .and_then(Value::as_array)
            .context("Inkling 2 GGUF is missing tokenizer vocabulary")?
            .len())
    }
    fn boolean(&self, names: &[&str]) -> Result<bool> {
        match self.get(names)? {
            Value::Bool(v) => Ok(*v),
            _ => anyhow::bail!("invalid boolean {names:?}"),
        }
    }
    fn string(&self, names: &[&str]) -> Result<&str> {
        self.get(names)?
            .as_str()
            .with_context(|| format!("invalid string {names:?}"))
    }
    fn floats(&self, names: &[&str]) -> Result<Vec<f64>> {
        self.get(names)?
            .as_array()
            .context("expected numeric array")?
            .iter()
            .map(|v| {
                v.as_f64()
                    .with_context(|| format!("invalid array element {names:?}"))
            })
            .collect()
    }
    fn integer(&self, names: &[&str]) -> Result<u64> {
        self.get(names)?
            .as_u64()
            .with_context(|| format!("invalid integer {names:?}"))
    }

    fn float(&self, names: &[&str]) -> Result<f32> {
        let x = self
            .get(names)?
            .as_f64()
            .with_context(|| format!("invalid float {names:?}"))?;
        ensure!(
            x.is_finite() && (x as f32).is_finite(),
            "non-finite metadata {names:?}"
        );
        Ok(x as f32)
    }

    fn integers(&self, names: &[&str]) -> Result<Vec<u64>> {
        self.get(names)?
            .as_array()
            .context("expected integer array")?
            .iter()
            .map(|x| {
                x.as_u64()
                    .with_context(|| format!("invalid array element {names:?}"))
            })
            .collect()
    }
}

pub fn validate_vision(h: &Header) -> Result<()> {
    ensure!(
        matches!(
            h.metadata.get("clip.has_vision_encoder"),
            Some(Value::Bool(true))
        ),
        "missing Inkling vision encoder"
    );
    lisa_models::inkling2::config::validate_vision(&Metadata {
        header: h,
        prefix: "clip.vision",
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    fn fixture() -> serde_json::Value {
        serde_json::from_str(include_str!("../fixtures/inkling2-q4-k-m.json")).unwrap()
    }
    fn value(v: &serde_json::Value) -> Value {
        match v {
            serde_json::Value::Bool(x) => Value::Bool(*x),
            serde_json::Value::String(x) => Value::String(x.clone()),
            serde_json::Value::Array(v) => Value::Array(v.iter().map(value).collect()),
            serde_json::Value::Number(n) => n
                .as_u64()
                .map(Value::Unsigned)
                .unwrap_or_else(|| Value::Float(n.as_f64().unwrap())),
            _ => panic!("invalid fixture"),
        }
    }
    fn header() -> Header {
        let f = fixture();
        let mut metadata = f["metadata"]
            .as_object()
            .unwrap()
            .iter()
            .map(|(k, v)| (k.clone(), value(v)))
            .collect::<std::collections::BTreeMap<_, _>>();
        metadata.insert(
            "tokenizer.ggml.tokens".into(),
            Value::Array(vec![Value::String(String::new()); 200058]),
        );
        Header {
            metadata,
            tensors: Default::default(),
            data_offset: 0,
            file_bytes: 0,
        }
    }
    #[test]
    fn projector_metadata_is_validated() {
        let f: serde_json::Value =
            serde_json::from_str(include_str!("../fixtures/inkling2-vision.json")).unwrap();
        let mut h = header();
        h.metadata = f
            .as_object()
            .unwrap()
            .iter()
            .map(|(k, v)| (k.clone(), value(v)))
            .collect();
        validate_vision(&h).unwrap();
        for (key, v) in [
            ("clip.vision.spatial_merge_size", Value::Unsigned(4)),
            (
                "clip.vision.attention.layer_norm_epsilon",
                Value::Float(1e-6),
            ),
            (
                "clip.vision.image_std",
                Value::Array(vec![Value::Float(0.4); 3]),
            ),
        ] {
            let mut h = Header {
                metadata: h.metadata.clone(),
                tensors: Default::default(),
                data_offset: 0,
                file_bytes: 0,
            };
            h.metadata.insert(key.into(), v);
            assert!(validate_vision(&h).is_err(), "{key}");
        }
    }
    #[test]
    fn release_geometry_and_arithmetic_are_validated() {
        let h = header();
        assert_eq!(config(&h).unwrap().context, 1048576);
        for (key, v) in [
            ("inkling2.num_loops", Value::Unsigned(2)),
            ("inkling2.attention.qk_clip", Value::Float(22.)),
            ("tokenizer.ggml.eos_token_id", Value::Unsigned(0)),
        ] {
            let mut h = header();
            h.metadata.insert(key.into(), v);
            assert!(config(&h).is_err(), "{key}");
        }
    }
    #[test]
    fn released_tensors_match_the_step_geometry() {
        let f = fixture();
        let actual = f["tensors"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| (v["name"].as_str().unwrap(), v))
            .collect::<std::collections::BTreeMap<_, _>>();
        let mut matched = 0;
        for (name, shape) in lisa_models::inkling2::config::tensors() {
            let Some(t) = actual.get(name.as_str()) else {
                assert!(name.ends_with(".scale"));
                continue;
            };
            assert_eq!(
                shape,
                serde_json::from_value::<Vec<u64>>(t["shape"].clone()).unwrap(),
                "{name}"
            );
            matched += 1;
        }
        assert_eq!(matched, actual.len());
    }
}
