//! Safetensors checkpoint loading: shard index, mmap, name filtering, and
//! materialization into native arrays.

use std::path::Path;
use std::sync::Arc;

use lisa_metal::{Array, Dtype};

#[cfg(test)]
use crate::weights::mlx::mapped_tensor_window;
pub use lisa_mlx::{Checkpoint, read_index_bytes};

/// Concrete application binding of the portable mapped-checkpoint adapter.
pub struct Shard(crate::weights::mlx::Shard<lisa_metal::device::MetalDevice>);
impl std::ops::Deref for Shard {
    type Target = crate::weights::mlx::Shard<lisa_metal::device::MetalDevice>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl Shard {
    pub fn open(path: &Path) -> anyhow::Result<Self> {
        Ok(Self(crate::weights::mlx::Shard::open(
            crate::metal_context::device(),
            path,
        )?))
    }
    pub fn tensor_bytes(&self, name: &str) -> Option<(&[u8], Dtype, &[i32])> {
        self.0
            .tensor_bytes(name)
            .map(|(bytes, dtype, shape)| (bytes, crate::metal_storage::dtype(dtype), shape))
    }
}
#[cfg(test)]
unsafe extern "C" {
    fn getpagesize() -> i32;
}

/// Metal application binding of the portable lazy checkpoint importer.
pub struct LazyTensorStore(crate::weights::lazy::LazyTensorStore<lisa_metal::device::MetalDevice>);
impl std::ops::Deref for LazyTensorStore {
    type Target = crate::weights::lazy::LazyTensorStore<lisa_metal::device::MetalDevice>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl LazyTensorStore {
    pub fn into_inner(
        self,
    ) -> crate::weights::lazy::LazyTensorStore<lisa_metal::device::MetalDevice> {
        self.0
    }

    fn bind(
        mut store: crate::weights::lazy::LazyTensorStore<lisa_metal::device::MetalDevice>,
    ) -> Self {
        store.set_copy_observer(|tensor, shard| {
            if lisa_metal::trace::enabled_for("glm53-loader") {
                eprintln!("[glm53-loader] copying tensor {tensor} from {shard}: no legal mapped Metal window");
            }
        });
        Self(store)
    }
    pub fn open(dir: &Path) -> anyhow::Result<Self> {
        Ok(Self::bind(crate::weights::lazy::LazyTensorStore::open(
            crate::metal_context::device(),
            dir,
            sanitize_name,
        )?))
    }
    pub fn open_resident(dir: &Path) -> anyhow::Result<Self> {
        Self::open_resident_index(dir, &dir.join(".lisa-resident/index.json"))
    }
    pub fn open_resident_index(dir: &Path, index: &Path) -> anyhow::Result<Self> {
        Ok(Self::bind(
            crate::weights::lazy::LazyTensorStore::open_resident_index(
                crate::metal_context::device(),
                dir,
                index,
                sanitize_name,
            )?,
        ))
    }
    #[cfg(test)]
    pub(crate) fn from_checkpoint(dir: &Path, checkpoint: &Checkpoint) -> anyhow::Result<Self> {
        Ok(Self::bind(
            crate::weights::lazy::LazyTensorStore::from_checkpoint(
                crate::metal_context::device(),
                dir,
                checkpoint,
                sanitize_name,
            )?,
        ))
    }
    pub fn tensor_dtype(&self, name: &str) -> anyhow::Result<Dtype> {
        self.0.tensor_dtype(name).map(crate::metal_storage::dtype)
    }
    pub fn resident_buffers(&self) -> Vec<Arc<lisa_metal::runtime::Buffer>> {
        self.0
            .resident_allocations()
            .iter()
            .map(|a| a.t.buffer().clone())
            .collect()
    }
}

/// Whether a sanitized tensor name belongs to one routed GLM expert projection.
pub fn is_glm_routed_expert_tensor(name: &str) -> bool {
    let parts: Vec<&str> = name.split('.').collect();
    parts.len() == 8
        && matches!(parts[0], "model" | "mtp")
        && parts[1] == "layers"
        && parts[2].parse::<usize>().is_ok()
        && parts[3] == "mlp"
        && parts[4] == "experts"
        && parts[5].parse::<usize>().is_ok()
        && matches!(parts[6], "gate_proj" | "up_proj" | "down_proj")
        && matches!(parts[7], "weight" | "scales" | "biases")
}

/// Create a native array from checked tensor bytes (copies onto the device).
pub fn array_from_bytes(
    bytes: &[u8],
    dtype: Dtype,
    shape: &[i32],
) -> lisa_metal::error::Result<Array> {
    use lisa_backend::{Device, HostTensor, ScalarType};
    let dtype = match dtype {
        Dtype::Bfloat16 => ScalarType::BF16,
        Dtype::Uint32 => ScalarType::U32,
        Dtype::Uint8 => ScalarType::U8,
        Dtype::Int64 => ScalarType::I64,
        Dtype::Float32 => ScalarType::F32,
        Dtype::Float16 => ScalarType::F16,
        Dtype::Int32 => ScalarType::I32,
        _ => {
            return Err(lisa_metal::error::Error::msg(format!(
                "array_from_bytes: unsupported dtype {dtype:?}"
            )));
        }
    };
    let source = HostTensor {
        bytes,
        dtype,
        shape,
    };
    // Reject invalid descriptors before creating a device context.
    if !source.validate() {
        return Err(lisa_metal::error::Error::msg(
            "array_from_bytes: payload size differs from tensor shape",
        ));
    }
    crate::metal_context::device().pull(source.into(), Default::default())
}

pub use lisa_models::checkpoint::sanitize_name;

/// Fully materialized weight tensors keyed by sanitized name.
pub type Weights = lisa_engine::weights::source::Weights<Array>;

/// Source of weight tensors by name. Implemented for a single opened shard
/// (loading straight from disk) and for the fully materialized weight map.
pub trait TensorSource: lisa_engine::weights::source::TensorSource<Array> {
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

impl lisa_engine::weights::source::TensorSource<Array> for Shard {
    fn get(&mut self, name: &str) -> anyhow::Result<Array> {
        crate::core::quant::get_tensor(self, name)
    }
}

impl<T: lisa_engine::weights::source::TensorSource<Array>> TensorSource for T {}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn prefetch_many_disjoint_ranges_preserves_all_page_hints() {
        let page = unsafe { getpagesize() } as usize;
        let count = 40usize;
        let stride = page * 3;
        let mut tensors = serde_json::Map::new();
        let names = (0..count)
            .map(|index| {
                let name = format!("probe_{index}");
                let start = (index + 2) * stride;
                tensors.insert(
                    name.clone(),
                    serde_json::json!({
                        "dtype": "U8",
                        "shape": [1],
                        "data_offsets": [start, start + 1]
                    }),
                );
                name
            })
            .collect::<Vec<_>>();
        let mut header = serde_json::to_vec(&serde_json::Value::Object(tensors)).unwrap();
        while header.len() % 8 != 0 {
            header.push(b' ');
        }

        let path = std::env::temp_dir().join(format!(
            "lisa-prefetch-ranges-{}-{}.safetensors",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let mut file = std::fs::File::create(&path).unwrap();
        file.write_all(&(header.len() as u64).to_le_bytes())
            .unwrap();
        file.write_all(&header).unwrap();
        let data_start = 8 + header.len();
        let file_len = data_start + (count + 1) * stride + 1;
        file.set_len(file_len.div_ceil(page).saturating_mul(page) as u64)
            .unwrap();
        drop(file);

        let shard = Shard::open(&path).unwrap();
        let (ranges, bytes) = shard.prefetch_tensor_ranges(&names);
        assert_eq!(ranges, count);
        assert_eq!(bytes, count * page);

        drop(shard);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn mincore_marks_touched_mapped_pages_resident() {
        let page = unsafe { getpagesize() } as usize;
        let payload_len = page * 3;
        let mut tensors = serde_json::Map::new();
        tensors.insert(
            "probe".to_owned(),
            serde_json::json!({
                "dtype": "U8",
                "shape": [payload_len],
                "data_offsets": [0, payload_len]
            }),
        );
        let mut header = serde_json::to_vec(&serde_json::Value::Object(tensors)).unwrap();
        while header.len() % 8 != 0 {
            header.push(b' ');
        }
        let path = std::env::temp_dir().join(format!(
            "lisa-mincore-resident-{}-{}.safetensors",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let mut file = std::fs::File::create(&path).unwrap();
        file.write_all(&(header.len() as u64).to_le_bytes())
            .unwrap();
        file.write_all(&header).unwrap();
        file.write_all(&vec![0x5a; payload_len]).unwrap();
        drop(file);

        let shard = Shard::open(&path).unwrap();
        for offset in (0..shard.source.mapping().len()).step_by(page) {
            std::hint::black_box(shard.source.mapping()[offset]);
        }
        assert_eq!(
            shard.mapped_range_mostly_resident(0, shard.source.mapping().len(), page),
            Some(true)
        );
        drop(shard);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn mapped_window_recenters_tensor_that_crosses_chunk_boundary() {
        let page = 16 * 1024;
        let max = page * 8;
        let chunk = max;
        let start = chunk - page * 2;
        let end = chunk + page * 2;
        assert_eq!(
            mapped_tensor_window(start, end, chunk * 3, page, max, max),
            Some((start, page * 4))
        );
    }

    #[test]
    fn owned_tensor_import_preserves_misaligned_u32_data() {
        if !lisa_metal::device::metal_available() {
            eprintln!("skipping Metal raw tensor import assertion: no Metal device available");
            return;
        }

        let words = [0x0123_4567u32, 0x89ab_cdef, 0xfedc_ba98];
        let payload_bytes: Vec<u8> = words.iter().flat_map(|word| word.to_le_bytes()).collect();
        let mut header = serde_json::to_vec(&serde_json::json!({
            "probe": {
                "dtype": "U32",
                "shape": [words.len()],
                "data_offsets": [0, payload_bytes.len()]
            }
        }))
        .unwrap();
        while (8 + header.len()) % std::mem::size_of::<u32>() != 1 {
            header.push(b' ');
        }

        let path = std::env::temp_dir().join(format!(
            "lisa-misaligned-u32-{}-{}.safetensors",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let mut file = std::fs::File::create(&path).unwrap();
        file.write_all(&(header.len() as u64).to_le_bytes())
            .unwrap();
        file.write_all(&header).unwrap();
        file.write_all(&payload_bytes).unwrap();
        drop(file);

        let shard = Shard::open(&path).unwrap();
        assert!(shard.tensor_array("probe").unwrap().is_none());
        let imported = crate::core::quant::get_tensor(&shard, "probe").unwrap();
        assert_eq!(imported.as_slice::<u32>(), words);
        drop(imported);
        drop(shard);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn mapped_tensor_view_reads_file_backed_payload() {
        let payload = [1.25f32, -2.5, 3.75, 8.0];
        let payload_bytes: Vec<u8> = payload.iter().flat_map(|v| v.to_le_bytes()).collect();
        let mut header = serde_json::to_vec(&serde_json::json!({
            "probe": {
                "dtype": "F32",
                "shape": [payload.len()],
                "data_offsets": [0, std::mem::size_of_val(&payload)]
            }
        }))
        .unwrap();
        while header.len() % 8 != 0 {
            header.push(b' ');
        }

        let path = std::env::temp_dir().join(format!(
            "lisa-mmap-tensor-{}-{}.safetensors",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let mut file = std::fs::File::create(&path).unwrap();
        file.write_all(&(header.len() as u64).to_le_bytes())
            .unwrap();
        file.write_all(&header).unwrap();
        file.write_all(&payload_bytes).unwrap();
        let page_size = unsafe { getpagesize() } as usize;
        let file_len = 8 + header.len() + payload_bytes.len();
        let padded_len = file_len.div_ceil(page_size) * page_size;
        file.set_len(padded_len as u64).unwrap();
        drop(file);

        let shard = Shard::open(&path).unwrap();
        let (bytes, dtype, shape) = shard.tensor_bytes("probe").unwrap();
        assert_eq!(dtype, Dtype::Float32);
        assert_eq!(shape, [payload.len() as i32]);
        assert_eq!(bytes, payload_bytes);
        let source_bytes = bytes.to_vec();
        let shape = shape.to_vec();
        if !lisa_metal::device::metal_available() {
            eprintln!("skipping Metal mapped-buffer assertion: no Metal device available");
            drop(shard);
            std::fs::remove_file(path).unwrap();
            return;
        }
        let mapped = shard
            .tensor_array("probe")
            .unwrap()
            .expect("aligned tensor should use the mapped buffer");
        shard.discard_cached_pages().unwrap();
        let doubled = mapped.add(&mapped).unwrap();
        assert_eq!(doubled.as_slice::<f32>(), [2.5, -5.0, 7.5, 16.0]);
        let copied = array_from_bytes(&source_bytes, dtype, &shape).unwrap();
        assert_eq!(mapped.as_slice::<f32>(), payload);
        assert_eq!(copied.as_slice::<f32>(), payload);

        drop(mapped);
        drop(copied);
        drop(shard);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn mapped_expert_views_share_one_shard_buffer() {
        if !lisa_metal::device::metal_available() {
            eprintln!("skipping Metal shared-shard assertion: no Metal device available");
            return;
        }

        let first = [3u8, 5, 7, 11];
        let second = [13u8, 17, 19, 23];
        let typed = [0x80u8, 0x3f, 0x00, 0x3f];
        let large = vec![0x5au8; unsafe { getpagesize() } as usize + 777];
        let mut header = serde_json::to_vec(&serde_json::json!({
            "expert_a": {
                "dtype": "U8",
                "shape": [first.len()],
                "data_offsets": [0, first.len()]
            },
            "expert_b": {
                "dtype": "U8",
                "shape": [second.len()],
                "data_offsets": [first.len(), first.len() + second.len()]
            },
            "expert_scales": {
                "dtype": "BF16",
                "shape": [2],
                "data_offsets": [first.len() + second.len(), first.len() + second.len() + typed.len()]
            },
            "large_expert_weight": {
                "dtype": "U8",
                "shape": [large.len()],
                "data_offsets": [first.len() + second.len() + typed.len(), first.len() + second.len() + typed.len() + large.len()]
            }
        }))
        .unwrap();
        while header.len() % 8 != 0 {
            header.push(b' ');
        }

        let path = std::env::temp_dir().join(format!(
            "lisa-mmap-shared-shard-{}-{}.safetensors",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let mut file = std::fs::File::create(&path).unwrap();
        file.write_all(&(header.len() as u64).to_le_bytes())
            .unwrap();
        file.write_all(&header).unwrap();
        file.write_all(&first).unwrap();
        file.write_all(&second).unwrap();
        file.write_all(&typed).unwrap();
        file.write_all(&large).unwrap();
        let page_size = unsafe { getpagesize() } as usize;
        let file_len = 8 + header.len() + first.len() + second.len() + typed.len() + large.len();
        file.set_len((file_len.div_ceil(page_size) * page_size) as u64)
            .unwrap();
        drop(file);

        let shard = Shard::open(&path).unwrap();
        let uncached_large = shard
            .tensor_byte_array_individual_uncached("large_expert_weight")
            .unwrap()
            .expect("large individual tensor should map without a shard cache entry");
        assert_eq!(uncached_large.to_vec1::<u8>().unwrap(), large);
        assert!(shard.mapped_buffers.lock().unwrap().is_empty());
        drop(uncached_large);

        let a = shard
            .tensor_byte_array_shard_view("expert_a")
            .unwrap()
            .unwrap();
        let b = shard
            .tensor_byte_array_shard_view("expert_b")
            .unwrap()
            .unwrap();
        let scales = shard
            .tensor_array_shard_view("expert_scales")
            .unwrap()
            .unwrap();
        assert_eq!(a.backing_buffer_id(), b.backing_buffer_id());
        assert_eq!(a.backing_buffer_id(), scales.backing_buffer_id());
        assert_eq!(a.to_vec1::<u8>().unwrap(), first);
        assert_eq!(b.to_vec1::<u8>().unwrap(), second);
        assert_eq!(
            scales.to_vec1::<half::bf16>().unwrap(),
            [half::bf16::from_f32(1.0), half::bf16::from_f32(0.5)]
        );
        assert_eq!(shard.mapped_buffers.lock().unwrap().len(), 1);

        drop(a);
        drop(b);
        drop(scales);
        drop(shard);
        std::fs::remove_file(path).unwrap();
    }
}
