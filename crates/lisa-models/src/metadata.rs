//! Read-only scalar checkpoint metadata supplied by an engine format adapter.
//! Alias lists and geometry validation belong to the model schema.
use anyhow::{Result, ensure};
pub trait Metadata {
    fn architecture(&self) -> Result<&str>;
    fn contains(&self, name: &str) -> bool;
    fn vocabulary_len(&self) -> Result<usize>;
    fn integer(&self, names: &[&str]) -> Result<u64>;
    fn float(&self, names: &[&str]) -> Result<f32>;
    fn integers(&self, names: &[&str]) -> Result<Vec<u64>>;
    fn floats(&self, names: &[&str]) -> Result<Vec<f64>>;
    fn boolean(&self, names: &[&str]) -> Result<bool>;
    fn string(&self, names: &[&str]) -> Result<&str>;
    fn expect(&self, names: &[&str], expected: u64) -> Result<()> {
        let actual = self.integer(names)?;
        ensure!(
            actual == expected,
            "unsupported {names:?}: {actual}, expected {expected}"
        );
        Ok(())
    }
}
