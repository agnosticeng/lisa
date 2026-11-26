//! GGUF metadata adaptation. Model geometry remains in lisa-models.
use anyhow::{Context, Result, ensure};
use lisa_gguf::{Header, Value};
pub fn config(h: &Header) -> Result<lisa_models::olala::config::Config> {
    ensure!(
        h.metadata
            .get("tokenizer.ggml.eos_token_id")
            .and_then(Value::as_u64)
            == Some(151645),
        "unsupported Olala EOS token"
    );
    for name in [
        "tokenizer.ggml.add_bos_token",
        "tokenizer.ggml.add_eos_token",
    ] {
        ensure!(
            matches!(h.metadata.get(name), Some(Value::Bool(false))),
            "unsupported Olala {name}"
        );
    }
    lisa_models::olala::config::Config::from_metadata(&Metadata(h))
}
pub struct Metadata<'a>(pub &'a Header);

impl Metadata<'_> {
    pub fn get(&self, names: &[&str]) -> Result<&Value> {
        let mut result = None;
        for name in names {
            if let Some(value) = self.0.metadata.get(&format!("olala.{name}")) {
                if let Some(previous) = result {
                    ensure!(
                        previous == value,
                        "conflicting Olala metadata aliases {names:?}"
                    );
                }
                result = Some(value);
            }
        }
        result.with_context(|| format!("missing Olala metadata {names:?}"))
    }
}
impl lisa_models::metadata::Metadata for Metadata<'_> {
    fn architecture(&self) -> Result<&str> {
        self.0.architecture()
    }
    fn contains(&self, name: &str) -> bool {
        self.0.metadata.contains_key(&format!("olala.{name}"))
    }
    fn vocabulary_len(&self) -> Result<usize> {
        Ok(self
            .0
            .metadata
            .get("tokenizer.ggml.tokens")
            .and_then(Value::as_array)
            .context("Olala GGUF is missing tokenizer vocabulary")?
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

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    pub(crate) fn header() -> Header {
        let v: serde_json::Value =
            serde_json::from_str(include_str!("fixtures/olala-q5-k-m.json")).unwrap();
        let mut metadata = std::collections::BTreeMap::new();
        for (k, v) in v["metadata"].as_object().unwrap() {
            let value = match v {
                serde_json::Value::Bool(x) => Value::Bool(*x),
                serde_json::Value::String(x) => Value::String(x.clone()),
                serde_json::Value::Number(x) => {
                    if let Some(n) = x.as_u64() {
                        Value::Unsigned(n)
                    } else {
                        Value::Float(x.as_f64().unwrap())
                    }
                }
                _ => unreachable!(),
            };
            metadata.insert(k.clone(), value);
        }
        metadata.insert(
            "tokenizer.ggml.tokens".into(),
            Value::Array(vec![Value::String(String::new()); 151936]),
        );
        Header {
            metadata,
            tensors: Default::default(),
            data_offset: 0,
            file_bytes: 0,
        }
    }
    #[test]
    fn validates_release_metadata_and_rejects_changed_arithmetic() {
        let h = header();
        let c = config(&h).unwrap();
        assert_eq!(c.layers.iter().filter(|&&v| v).count(), 29);
        for (key, v) in [
            ("tokenizer.ggml.eos_token_id", Value::Unsigned(0)),
            ("tokenizer.ggml.add_bos_token", Value::Bool(true)),
            ("olala.expert_count", Value::Unsigned(128)),
            ("olala.layer_types", Value::String("M".into())),
            ("olala.final_norm", Value::Bool(true)),
            ("olala.context_length", Value::Unsigned(0)),
            ("olala.attn_logit_softcapping", Value::Float(f64::NAN)),
        ] {
            let mut h = header();
            h.metadata.insert(key.into(), v);
            assert!(config(&h).is_err(), "{key}");
        }
    }
    #[test]
    fn manifest_matches_portable_shapes_and_graphs() {
        use lisa_models::olala::{
            config::tensors,
            graph::{self, Format},
        };
        let c = config(&header()).unwrap();
        let fixture: serde_json::Value =
            serde_json::from_str(include_str!("fixtures/olala-q5-k-m.json")).unwrap();
        let mut formats = std::collections::BTreeMap::new();
        for (name, shape, _) in tensors(&c) {
            let t = &fixture["tensors"][&name];
            let actual: Vec<u64> = serde_json::from_value(t["dimensions"].clone()).unwrap();
            assert_eq!(shape, actual, "{name}");
            let kind = lisa_gguf::TensorType(t["kind"].as_u64().unwrap() as u32);
            let dtype = match kind {
                lisa_gguf::TensorType::F32 => lisa_backend::ScalarType::F32,
                lisa_gguf::TensorType::BF16 => lisa_backend::ScalarType::BF16,
                _ => lisa_backend::ScalarType::U8,
            };
            formats.insert(
                name,
                Format {
                    projection: crate::weights::gguf::projection(kind).unwrap(),
                    dtype,
                    row_bytes: kind.row_bytes(shape[0]).unwrap() as i32,
                },
            );
        }
        for i in 0..36 {
            for has_history in [false, true] {
                let def = graph::layer(&c, i, has_history, &formats);
                def.graph.validate().unwrap();
                assert!(def.graph.mutable_inputs.is_empty());
                assert!(def.graph.writes.is_empty());
            }
        }
        graph::head(&formats).graph.validate().unwrap();
    }
}
