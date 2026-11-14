//! Native Qwen GGUF execution, independent of the legacy MLX arithmetic law.
pub(crate) mod execution;

mod weights;
pub use crate::checkpoint::qwen35_tokenizer::from_header as tokenizer;

/// Validate the selected text package without creating a Metal runtime.
/// Embeddings are decoded a row at a time from disk and are not GPU-resident.
pub fn inspect(path: &std::path::Path) -> anyhow::Result<u64> {
    let package = lisa_gguf::Package::open(path)?;
    weights::validate(&package)?;
    package
        .tensors
        .iter()
        .filter(|(name, _)| name.as_str() != "token_embd.weight")
        .try_fold(0u64, |total, (_, tensor)| {
            total
                .checked_add(tensor.mapped_region_bytes()? as u64)
                .ok_or_else(|| anyhow::anyhow!("Qwen resident size overflow"))
        })
}

#[cfg(test)]
mod plan_oracle;
#[cfg(test)]
mod plan_tests;

pub use crate::checkpoint::catalog::qwen35::{ENTRY, REPO};
