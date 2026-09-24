//! Safetensors checkpoint loading: shard index, mmap, name filtering, and
//! materialization into MLX arrays.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, bail};
use lisa_mlx::{Array, Dtype};
use memmap2::Mmap;

/// One tensor record from the safetensors header.
struct TensorInfo {
    dtype: Dtype,
    shape: Vec<i32>,
    /// Absolute byte offset of the tensor payload within the shard file.
    start: usize,
    end: usize,
}

/// An mmap'd safetensors shard.
pub struct Shard {
    map: Mmap,
    tensors: HashMap<String, TensorInfo>,
}

impl Shard {
    pub fn open(path: &Path) -> anyhow::Result<Self> {
        let file = std::fs::File::open(path)
            .with_context(|| format!("opening shard {}", path.display()))?;
        let map = unsafe { Mmap::map(&file)? };
        let n = map.len();
        if n < 8 {
            bail!("shard {} too small", path.display());
        }
        let header_len = u64::from_le_bytes(map[0..8].try_into().unwrap()) as usize;
        if 8 + header_len > n {
            bail!("shard {} corrupt header length", path.display());
        }
        let header: serde_json::Value = serde_json::from_slice(&map[8..8 + header_len])?;

        let mut tensors = HashMap::new();
        if let Some(obj) = header.as_object() {
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
                let shape: Vec<i32> = info["shape"]
                    .as_array()
                    .map(|a| a.iter().filter_map(|v| v.as_i64().map(|x| x as i32)).collect())
                    .unwrap_or_default();
                let range = info["data_offsets"].as_array().unwrap();
                let start = range[0].as_u64().unwrap() as usize;
                let end = range[1].as_u64().unwrap() as usize;
                tensors.insert(
                    name.clone(),
                    TensorInfo {
                        dtype,
                        shape,
                        start: 8 + header_len + start,
                        end: 8 + header_len + end,
                    },
                );
            }
        }
        Ok(Self {
            map,
            tensors,
        })
    }

    /// Raw byte slice of one tensor.
    pub fn tensor_bytes(&self, name: &str) -> Option<(&[u8], Dtype, &[i32])> {
        let info = self.tensors.get(name)?;
        Some((
            &self.map[info.start..info.end],
            info.dtype,
            &info.shape[..],
        ))
    }

    pub fn tensor_names(&self) -> impl Iterator<Item = &String> {
        self.tensors.keys()
    }
}

/// Checkpoint view over a model directory: index + shard file names.
pub struct Checkpoint {
    pub dir: PathBuf,
    /// weight_map: sanitized tensor name -> shard file name
    pub weight_map: HashMap<String, String>,
}

impl Checkpoint {
    pub fn open(dir: &Path) -> anyhow::Result<Self> {
        let index_path = dir.join("model.safetensors.index.json");
        let data = std::fs::read_to_string(&index_path)
            .with_context(|| format!("reading {}", index_path.display()))?;
        let index: serde_json::Value = serde_json::from_str(&data)?;
        let map = index["weight_map"]
            .as_object()
            .context("missing weight_map")?
            .clone();
        let mut weight_map = HashMap::new();
        for (name, shard) in map {
            let shard = shard.as_str().context("shard name not a string")?.to_string();
            weight_map.insert(name.to_string(), shard);
        }
        Ok(Self {
            dir: dir.to_path_buf(),
            weight_map,
        })
    }

}

/// Create an MLX array from raw tensor bytes (copies onto the device).
pub fn array_from_bytes(bytes: &[u8], dtype: Dtype, shape: &[i32]) -> lisa_mlx::error::Result<Array> {
    unsafe {
        match dtype {
            Dtype::Bfloat16 => Ok(Array::from_raw_data(
                bytes.as_ptr() as *const std::ffi::c_void,
                shape,
                Dtype::Bfloat16,
            )),
            Dtype::Uint32 => Ok(Array::from_raw_data(
                bytes.as_ptr() as *const std::ffi::c_void,
                shape,
                Dtype::Uint32,
            )),
            Dtype::Int64 => Ok(Array::from_raw_data(
                bytes.as_ptr() as *const std::ffi::c_void,
                shape,
                Dtype::Int64,
            )),
            Dtype::Float32 => Ok(Array::from_raw_data(
                bytes.as_ptr() as *const std::ffi::c_void,
                shape,
                Dtype::Float32,
            )),
            Dtype::Int32 => Ok(Array::from_raw_data(
                bytes.as_ptr() as *const std::ffi::c_void,
                shape,
                Dtype::Int32,
            )),
            _ => Err(lisa_mlx::error::Exception::custom(format!(
                "array_from_bytes: unsupported dtype {dtype:?}"
            ))),
        }
    }
}

/// Sanitize a checkpoint tensor name the way the reference does:
/// drop `language_model.` / `model.language_model.`, remap `model.mtp.` ->
/// `mtp.`, drop the vision tower, and drop n-gram shards.
pub fn sanitize_name(raw: &str) -> Option<String> {
    if raw.starts_with("vision_tower.")
        || raw.starts_with("visual.")
        || raw.starts_with("model.visual.")
    {
        return None;
    }
    let mut key = raw.to_string();
    if let Some(rest) = key.strip_prefix("model.language_model.") {
        key = format!("model.{rest}");
    } else if let Some(rest) = key.strip_prefix("language_model.") {
        key = rest.to_string();
    }
    if let Some(rest) = key.strip_prefix("model.mtp.") {
        key = format!("mtp.{rest}");
    }
    if key.contains(".ngram_embedding.shard_") {
        return None;
    }
    // The checkpoint's I64 n-gram buffers are never used: the values are
    // rebuilt from configuration (the reference does the same).
    if key.ends_with(".layer_multipliers")
        || key.ends_with(".ngram_heads_vocab_sizes")
        || key.ends_with(".ngram_heads_offsets")
    {
        return None;
    }
    // Torch stores a depthwise kernel as (C, 1, K); MLX wants (C, K, 1).
    // Handled by the caller at load time (shape inspection).
    Some(key)
}


/// Fully materialized weight tensors keyed by sanitized name.
pub type Weights = HashMap<String, Array>;

/// Source of weight tensors by name. Implemented for a single opened shard
/// (loading straight from disk) and for the fully materialized weight map.
pub trait TensorSource {
    fn get(&mut self, name: &str) -> anyhow::Result<Array>;
    fn get_bf16(&mut self, name: &str) -> anyhow::Result<Array> {
        let a = self.get(name)?;
        if a.dtype() == Dtype::Bfloat16 {
            Ok(a)
        } else {
            a.as_dtype(Dtype::Bfloat16)
                .map_err(|e| anyhow::anyhow!("{e}"))
        }
    }
}

impl TensorSource for Shard {
    fn get(&mut self, name: &str) -> anyhow::Result<Array> {
        crate::core::quant::get_tensor(self, name)
    }
}

impl TensorSource for Weights {
    fn get(&mut self, name: &str) -> anyhow::Result<Array> {
        self.remove(name)
            .ok_or_else(|| anyhow::anyhow!("tensor {name} missing from weight map"))
    }
}
