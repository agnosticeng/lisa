//! GGUF metadata adaptation; geometry and Engram hashing live in lisa-models.
use anyhow::{Context, Result, ensure};
use lisa_gguf::{Header, Value};
pub use lisa_models::deepseek41::{config::Config, engram::Layout};

pub fn config(header: &Header) -> Result<Config> {
    Config::from_metadata(&Metadata(header))
}
pub fn engram_layout(header: &Header) -> Result<Layout> {
    Layout::from_metadata(&Metadata(header))
}

/// Both DS4's original export and the standard GGUF converter are accepted.
/// Conflicting aliases are rejected rather than choosing one silently.
pub struct Metadata<'a>(pub &'a Header);

impl Metadata<'_> {
    pub fn get(&self, names: &[&str]) -> Result<&Value> {
        let mut result = None;
        for name in names {
            if let Some(value) = self.0.metadata.get(&format!("deepseek41.{name}")) {
                if let Some(previous) = result {
                    ensure!(
                        previous == value,
                        "conflicting DeepSeek metadata aliases {names:?}"
                    );
                }
                result = Some(value);
            }
        }
        result.with_context(|| format!("missing DeepSeek metadata {names:?}"))
    }
}
impl lisa_models::metadata::Metadata for Metadata<'_> {
    fn architecture(&self) -> Result<&str> {
        self.0.architecture()
    }
    fn contains(&self, name: &str) -> bool {
        self.0.metadata.contains_key(&format!("deepseek41.{name}"))
    }
    fn vocabulary_len(&self) -> Result<usize> {
        Ok(self
            .0
            .metadata
            .get("tokenizer.ggml.tokens")
            .and_then(Value::as_array)
            .context("DeepSeek GGUF is missing tokenizer vocabulary")?
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
mod tests {
    use super::*;
    use lisa_models::metadata::Metadata as _;
    #[test]
    fn metadata_preserves_alias_conflicts_and_scalar_types() {
        let mut h = Header {
            metadata: Default::default(),
            tensors: Default::default(),
            data_offset: 0,
            file_bytes: 0,
        };
        h.metadata
            .insert("deepseek41.a".into(), Value::Unsigned(40));
        h.metadata
            .insert("deepseek41.b".into(), Value::Unsigned(40));
        assert_eq!(Metadata(&h).integer(&["a", "b"]).unwrap(), 40);
        h.metadata.insert("deepseek41.b".into(), Value::Signed(40));
        // Numerically equal but differently typed aliases remain conflicting.
        assert!(Metadata(&h).integer(&["a", "b"]).is_err());
        h.metadata.insert("deepseek41.b".into(), Value::Signed(-1));
        assert!(Metadata(&h).integer(&["b"]).is_err());
        assert!(Metadata(&h).boolean(&["a"]).is_err());
        assert!(Metadata(&h).integer(&["missing"]).is_err());
        for value in [f64::NAN, f64::INFINITY, f64::MAX] {
            h.metadata
                .insert("deepseek41.b".into(), Value::Float(value));
            assert!(Metadata(&h).float(&["b"]).is_err());
        }
    }
}
