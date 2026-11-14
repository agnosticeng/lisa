//! Portable MLX checkpoint storage: safetensors payloads and shard indices.
//!
//! This crate owns immutable host bytes and metadata. It does not allocate
//! device buffers, select hardware, rename model tensors or dequantize weights.
//! MLX affine projections retain their original packed weights, scales and
//! biases as separate named tensors; model schema adapters interpret them.

use anyhow::{Context, bail, ensure};
use memmap2::Mmap;
use std::collections::HashMap;
use std::io::Read;
use std::path::{Component, Path, PathBuf};
use std::sync::Arc;

mod dtype;
pub use dtype::Dtype;

/// Bound metadata reads independently of checkpoint payload size.
pub fn read_index_bytes(path: &Path) -> anyhow::Result<Vec<u8>> {
    const LIMIT: u64 = 256 * 1024 * 1024;
    let file =
        std::fs::File::open(path).with_context(|| format!("opening index {}", path.display()))?;
    let metadata = file.metadata()?;
    ensure!(
        metadata.is_file() && metadata.len() <= LIMIT,
        "index must be a regular file no larger than 256 MiB: {}",
        path.display()
    );
    let mut bytes = Vec::new();
    file.take(LIMIT + 1).read_to_end(&mut bytes)?;
    ensure!(
        bytes.len() as u64 <= LIMIT,
        "index grew beyond 256 MiB: {}",
        path.display()
    );
    Ok(bytes)
}

/// One tensor record from the safetensors header.
#[derive(Debug)]
pub struct TensorInfo {
    pub dtype: Dtype,
    pub shape: Vec<i32>,
    /// Absolute byte offset of the tensor payload within the shard file.
    pub start: usize,
    pub end: usize,
}

/// A validated, read-only safetensors mapping. Cloned mapping owners keep
/// payloads alive after the shard handle is dropped.
pub struct Shard {
    map: Arc<Mmap>,
    tensors: HashMap<String, TensorInfo>,
}

impl Shard {
    pub fn open(path: &Path) -> anyhow::Result<Self> {
        let file = std::fs::File::open(path)
            .with_context(|| format!("opening shard {}", path.display()))?;
        let map = Arc::new(unsafe { Mmap::map(&file)? });
        let n = map.len();
        if n < 8 {
            bail!("shard {} too small", path.display());
        }
        let header_len = usize::try_from(u64::from_le_bytes(map[0..8].try_into()?))
            .context("safetensors header length exceeds address space")?;
        let payload_start = header_len
            .checked_add(8)
            .context("safetensors header length overflow")?;
        ensure!(
            payload_start <= n && header_len <= 64 * 1024 * 1024,
            "shard {} has an invalid or oversized header",
            path.display()
        );
        let header: serde_json::Value = serde_json::from_slice(&map[8..payload_start])
            .with_context(|| format!("parsing shard header {}", path.display()))?;

        let mut tensors = HashMap::new();
        let obj = header
            .as_object()
            .context("safetensors header must be an object")?;
        for (name, info) in obj {
            if name == "__metadata__" {
                continue;
            }
            let dtype_str = info["dtype"].as_str().unwrap_or_default();
            let dtype = match dtype_str {
                "BF16" => Dtype::Bfloat16,
                "F32" => Dtype::Float32,
                "F16" => Dtype::Float16,
                "I64" => Dtype::Int64,
                "I32" => Dtype::Int32,
                "U32" => Dtype::Uint32,
                "U8" => Dtype::Uint8,
                "BOOL" => Dtype::Bool,
                other => bail!("unsupported dtype {other} for {name}"),
            };
            let shape = info["shape"]
                .as_array()
                .with_context(|| format!("missing shape for {name}"))?
                .iter()
                .map(|v| {
                    v.as_u64()
                        .and_then(|d| i32::try_from(d).ok())
                        .with_context(|| format!("invalid dimension for {name}: {v}"))
                })
                .collect::<anyhow::Result<Vec<_>>>()?;
            let expected_bytes = shape
                .iter()
                .try_fold(dtype.size_of(), |bytes, &dim| {
                    bytes.checked_mul(dim as usize)
                })
                .with_context(|| format!("tensor shape overflows for {name}"))?;
            let range = info["data_offsets"]
                .as_array()
                .with_context(|| format!("missing data_offsets for {name}"))?;
            ensure!(
                range.len() == 2,
                "data_offsets must have two entries for {name}"
            );
            let offset = |v: &serde_json::Value| -> anyhow::Result<usize> {
                v.as_u64()
                    .and_then(|n| usize::try_from(n).ok())
                    .with_context(|| format!("invalid data offset for {name}: {v}"))
            };
            let start = offset(&range[0])?;
            let end = offset(&range[1])?;
            ensure!(
                start <= end && end <= n - payload_start,
                "tensor {name} exceeds shard payload bounds"
            );
            ensure!(
                end - start == expected_bytes,
                "tensor byte size differs from shape for {name}"
            );
            tensors.insert(
                name.clone(),
                TensorInfo {
                    dtype,
                    shape,
                    start: payload_start + start,
                    end: payload_start + end,
                },
            );
        }
        Ok(Self { map, tensors })
    }

    /// Raw byte slice of one tensor.
    pub fn tensor_bytes(&self, name: &str) -> Option<(&[u8], Dtype, &[i32])> {
        let info = self.tensors.get(name)?;
        Some((&self.map[info.start..info.end], info.dtype, &info.shape[..]))
    }

    pub fn tensor_names(&self) -> impl Iterator<Item = &String> {
        self.tensors.keys()
    }

    /// Validated storage metadata; ranges are absolute offsets in `mapping()`.
    pub fn tensor_info(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.get(name)
    }

    /// Read-only backing owner for zero-copy host views or backend import.
    /// Importers must retain an owner through the last asynchronous read.
    pub fn mapping(&self) -> &Arc<Mmap> {
        &self.map
    }
}

/// Checkpoint view over a model directory: index + shard file names.
pub struct Checkpoint {
    pub dir: PathBuf,
    /// Original checkpoint tensor name -> shard file name.
    pub weight_map: HashMap<String, String>,
}

impl Checkpoint {
    pub fn open(dir: &Path) -> anyhow::Result<Self> {
        let index_path = dir.join("model.safetensors.index.json");
        let data = read_index_bytes(&index_path)
            .with_context(|| format!("reading {}", index_path.display()))?;
        let index: serde_json::Value = serde_json::from_slice(&data)?;
        let map = index["weight_map"]
            .as_object()
            .context("missing weight_map")?;
        let mut weight_map = HashMap::new();
        for (name, shard) in map {
            let shard = shard
                .as_str()
                .context("shard name not a string")?
                .to_string();
            ensure!(
                !shard.is_empty()
                    && Path::new(&shard)
                        .components()
                        .all(|c| matches!(c, Component::Normal(_))),
                "checkpoint shard must be a relative path inside the model directory: {shard}"
            );
            weight_map.insert(name.to_string(), shard);
        }
        Ok(Self {
            dir: dir.to_path_buf(),
            weight_map,
        })
    }
}

#[cfg(test)]
mod tests;
