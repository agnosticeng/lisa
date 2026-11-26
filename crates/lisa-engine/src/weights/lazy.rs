//! On-demand checkpoint import and prepared residency without a compute provider.
use super::mlx::{Checkpoint, Shard, scalar_type};
use anyhow::{Context, ensure};
use lisa_backend::{ScalarType as Dtype, StorageDevice};
use std::{
    collections::{BTreeMap, HashMap},
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
};

/// On-demand tensor views for GLM routed experts. Payloads stay file-backed
/// and are exposed as page-aligned external device buffers when possible.
/// A prepared resident store returns views of its selected mapped or uploaded
/// allocations, including from compatibility methods named `get_owned`.
/// Per-tensor lookups never copy a prepared allocation.
pub struct LazyTensorStore<D: StorageDevice> {
    device: D,
    on_copy: fn(&str, &str),
    dir: PathBuf,
    tensors: HashMap<String, (String, String)>,
    shards: Mutex<HashMap<String, Arc<Shard<D>>>>,
    resident: Option<crate::weights::weight_arena::ResidentWeights<D>>,
}

impl<D: StorageDevice + Clone + Sync> LazyTensorStore<D>
where
    D::Tensor: Clone + Send,
{
    fn copy_bytes(&self, bytes: &[u8], dtype: Dtype, shape: &[i32]) -> anyhow::Result<D::Tensor> {
        Ok(self.device.pull(
            lisa_backend::HostTensor {
                bytes,
                dtype,
                shape,
            }
            .into(),
            Default::default(),
        )?)
    }

    pub fn open(
        device: D,
        dir: &Path,
        normalize: fn(&str) -> Option<String>,
    ) -> anyhow::Result<Self> {
        let checkpoint = Checkpoint::open(dir)?;
        Self::from_checkpoint(device, dir, &checkpoint, normalize)
    }

    /// Open a prepared, byte-validated resident checkpoint. Missing preparation
    /// is an error; this route never silently falls back to private copies.
    pub fn open_resident(
        device: D,
        dir: &Path,
        normalize: fn(&str) -> Option<String>,
    ) -> anyhow::Result<Self> {
        Self::open_resident_index(
            device,
            dir,
            &dir.join(".lisa-resident/index.json"),
            normalize,
        )
    }

    pub fn open_resident_index(
        device: D,
        dir: &Path,
        index: &Path,
        normalize: fn(&str) -> Option<String>,
    ) -> anyhow::Result<Self> {
        Self::open_resident_index_with_policy(
            device,
            dir,
            index,
            normalize,
            super::ResidencyPolicy::Mapped,
        )
    }

    pub fn open_resident_index_with_policy(
        device: D,
        dir: &Path,
        index: &Path,
        normalize: fn(&str) -> Option<String>,
        policy: super::ResidencyPolicy,
    ) -> anyhow::Result<Self> {
        let checkpoint = Checkpoint::open(dir)?;
        let mut store = Self::from_checkpoint(device, dir, &checkpoint, normalize)?;
        let mut source_files = checkpoint
            .weight_map
            .values()
            .map(PathBuf::from)
            .collect::<std::collections::HashSet<_>>();
        source_files.insert(PathBuf::from("model.safetensors.index.json"));
        let resident = crate::weights::weight_arena::ResidentWeights::open_with_policy(
            store.device.clone(),
            dir,
            index,
            &source_files,
            policy,
        )?;
        anyhow::ensure!(
            resident.len() == store.tensors.len(),
            "resident index tensor inventory differs from checkpoint"
        );
        for name in store.tensors.keys() {
            anyhow::ensure!(
                resident.contains(name),
                "resident index is missing checkpoint tensor {name}"
            );
        }
        store.resident = Some(resident);
        Ok(store)
    }

    pub fn is_resident(&self) -> bool {
        self.resident.is_some()
    }

    /// Return one sanitized tensor through the store's mapped loading policy.
    pub fn get(&self, name: &str) -> anyhow::Result<D::Tensor> {
        self.get_mapped(name)
    }

    /// Imported backing allocations retained by the prepared resident store.
    pub fn resident_allocations(&self) -> Vec<D::Tensor> {
        self.resident
            .as_ref()
            .map_or_else(Vec::new, |r| r.buffers().to_vec())
    }

    /// Optional engine diagnostic hook; it does not change import policy.
    pub fn set_copy_observer(&mut self, observer: fn(&str, &str)) {
        self.on_copy = observer;
    }

    /// Unique mapped file bytes; overlapping device views are counted once.
    pub fn residency(&self) -> super::ResidencyBytes {
        self.resident
            .as_ref()
            .map_or(super::ResidencyBytes::default(), |r| r.residency())
    }

    pub fn resident_nbytes(&self) -> usize {
        self.resident.as_ref().map_or(0, |r| r.mapped_bytes())
    }

    /// Build a lazy tensor view from an index the caller already parsed.
    /// Reusing the checkpoint avoids reparsing its index and lets eager and
    /// lazy model loading share the same shard mappings and parsed headers.
    pub fn from_checkpoint(
        device: D,
        dir: &Path,
        checkpoint: &Checkpoint,
        normalize: fn(&str) -> Option<String>,
    ) -> anyhow::Result<Self> {
        let mut tensors = HashMap::new();
        tensors.reserve(checkpoint.weight_map.len());
        for (raw_name, shard_name) in &checkpoint.weight_map {
            if let Some(name) = normalize(raw_name) {
                ensure!(
                    tensors
                        .insert(name.clone(), (shard_name.clone(), raw_name.clone()))
                        .is_none(),
                    "checkpoint tensor names collide after normalization: {name}"
                );
            }
        }
        Ok(Self {
            device,
            on_copy: |_, _| {},
            dir: dir.to_path_buf(),
            tensors,
            shards: Mutex::new(HashMap::new()),
            resident: None,
        })
    }

    pub fn shard(&self, shard_name: &str) -> anyhow::Result<Arc<Shard<D>>> {
        let mut shards = self
            .shards
            .lock()
            .map_err(|_| anyhow::anyhow!("lazy checkpoint shard cache poisoned"))?;
        if let Some(shard) = shards.get(shard_name) {
            return Ok(Arc::clone(shard));
        }
        let shard = Arc::new(Shard::open(
            self.device.clone(),
            &self.dir.join(shard_name),
        )?);
        shards.insert(shard_name.to_string(), Arc::clone(&shard));
        Ok(shard)
    }

    /// Prefetch selected checkpoint tensors using a bounded number of
    /// per-shard advice calls. The advice is best effort and never changes
    /// the mapped tensor contents or inference result.
    pub fn prefetch_mapped_tensors(&self, names: &[String]) -> anyhow::Result<(usize, usize)> {
        if self.resident.is_some() {
            return Ok((0, 0));
        }
        let mut by_shard = BTreeMap::<String, Vec<String>>::new();
        for name in names {
            let (shard_name, raw_name) = self
                .tensors
                .get(name)
                .ok_or_else(|| anyhow::anyhow!("lazy checkpoint tensor {name} missing"))?;
            by_shard
                .entry(shard_name.clone())
                .or_default()
                .push(raw_name.clone());
        }

        let mut ranges = 0usize;
        let mut bytes = 0usize;
        for (shard_name, raw_names) in by_shard {
            let shard = self.shard(&shard_name)?;
            let (shard_ranges, shard_bytes) = shard.prefetch_tensor_ranges(&raw_names);
            ranges = ranges.saturating_add(shard_ranges);
            bytes = bytes.saturating_add(shard_bytes);
        }
        Ok((ranges, bytes))
    }

    /// Load one sanitized tensor into a private device allocation. Expert cache
    /// eviction can then release it without keeping the source shard wired in
    /// memory by a no-copy device buffer.
    pub fn get_owned(&self, name: &str) -> anyhow::Result<D::Tensor> {
        if let Some(resident) = &self.resident {
            return resident.get(name);
        }
        let (shard_name, raw_name) = self
            .tensors
            .get(name)
            .ok_or_else(|| anyhow::anyhow!("lazy checkpoint tensor {name} missing"))?;
        let shard = self.shard(shard_name)?;
        let (bytes, dtype, shape) = shard
            .tensor_bytes(raw_name)
            .ok_or_else(|| anyhow::anyhow!("tensor {raw_name} absent from shard {shard_name}"))?;
        self.copy_bytes(bytes, dtype, shape)
            .map_err(|e| anyhow::anyhow!("loading lazy tensor {name}: {e}"))
    }

    /// Copy one packed quantized tensor into a private byte array. Routed
    /// experts use this on demand so a device no-copy buffer cannot keep an
    /// entire checkpoint shard resident for the lifetime of the model.
    pub fn get_owned_weight_bytes(&self, name: &str) -> anyhow::Result<D::Tensor> {
        if let Some(resident) = &self.resident {
            return resident.get_bytes(name);
        }
        let (shard_name, raw_name) = self
            .tensors
            .get(name)
            .ok_or_else(|| anyhow::anyhow!("lazy checkpoint tensor {name} missing"))?;
        let shard = self.shard(shard_name)?;
        let (bytes, _, _) = shard
            .tensor_bytes(raw_name)
            .ok_or_else(|| anyhow::anyhow!("tensor {raw_name} absent from shard {shard_name}"))?;
        let byte_len = i32::try_from(bytes.len())
            .with_context(|| format!("expert weight {name} exceeds byte-array limits"))?;
        self.copy_bytes(bytes, Dtype::U8, &[byte_len])
    }

    /// Return a page-scoped typed view whose device buffer lifetime is owned by
    /// the caller. Unsupported or unaligned tensors fall back to a private
    /// allocation rather than retaining a view in the shard cache.
    pub fn get_mapped_individual(&self, name: &str) -> anyhow::Result<D::Tensor> {
        if let Some(resident) = &self.resident {
            return resident.get(name);
        }
        let (shard_name, raw_name) = self
            .tensors
            .get(name)
            .ok_or_else(|| anyhow::anyhow!("lazy checkpoint tensor {name} missing"))?;
        let shard = self.shard(shard_name)?;
        if let Some(array) = shard.tensor_array_individual_uncached(raw_name)? {
            return Ok(array);
        }
        self.get_owned(name)
    }

    /// Return a page-scoped byte view for packed quantized weights. The
    /// returned device buffer is retained only by the array, so an expert-cache
    /// eviction releases it without retaining the rest of the safetensors
    /// shard.
    pub fn get_mapped_weight_bytes_individual(&self, name: &str) -> anyhow::Result<D::Tensor> {
        if let Some(resident) = &self.resident {
            return resident.get_bytes(name);
        }
        let (shard_name, raw_name) = self
            .tensors
            .get(name)
            .ok_or_else(|| anyhow::anyhow!("lazy checkpoint tensor {name} missing"))?;
        let shard = self.shard(shard_name)?;
        if let Some(array) = shard.tensor_byte_array_individual_uncached(raw_name)? {
            return Ok(array);
        }
        self.get_owned_weight_bytes(name)
    }

    /// Return a tensor payload size from the safetensors index without
    /// materializing its data. Used to enforce the routed-expert cache budget
    /// before loading a selected expert.
    pub fn tensor_nbytes(&self, name: &str) -> anyhow::Result<usize> {
        if let Some(resident) = &self.resident {
            return resident.nbytes(name);
        }
        let (shard_name, raw_name) = self
            .tensors
            .get(name)
            .ok_or_else(|| anyhow::anyhow!("lazy checkpoint tensor {name} missing"))?;
        let shard = self.shard(shard_name)?;
        let (_, _, _) = shard
            .tensor_bytes(raw_name)
            .ok_or_else(|| anyhow::anyhow!("tensor {raw_name} absent from shard {shard_name}"))?;
        Ok(shard.source.tensor_info(raw_name).unwrap().end
            - shard.source.tensor_info(raw_name).unwrap().start)
    }

    /// Return a tensor dtype without copying or materializing its payload.
    pub fn tensor_dtype(&self, name: &str) -> anyhow::Result<Dtype> {
        if let Some(resident) = &self.resident {
            return resident.dtype(name);
        }
        let (shard_name, raw_name) = self
            .tensors
            .get(name)
            .ok_or_else(|| anyhow::anyhow!("lazy checkpoint tensor {name} missing"))?;
        let shard = self.shard(shard_name)?;
        shard
            .source
            .tensor_info(raw_name)
            .map(|info| scalar_type(info.dtype))
            .ok_or_else(|| anyhow::anyhow!("tensor {raw_name} absent from shard {shard_name}"))
    }

    /// Return a tensor's indexed shape without materializing its payload.
    pub fn tensor_shape(&self, name: &str) -> anyhow::Result<Vec<i32>> {
        if let Some(resident) = &self.resident {
            return resident.shape(name);
        }
        let (shard_name, raw_name) = self
            .tensors
            .get(name)
            .ok_or_else(|| anyhow::anyhow!("lazy checkpoint tensor {name} missing"))?;
        let shard = self.shard(shard_name)?;
        shard
            .source
            .tensor_info(raw_name)
            .map(|info| info.shape.clone())
            .ok_or_else(|| anyhow::anyhow!("tensor {raw_name} absent from shard {shard_name}"))
    }

    /// Load same-shaped tensors into one contiguous device array. GLM expert
    /// safetensors payloads often start at offsets that cannot be represented
    /// as typed device views; packing their raw bytes here keeps one backing
    /// allocation per expert parameter instead of one allocation per expert.
    pub fn get_packed(&self, names: &[String]) -> anyhow::Result<D::Tensor> {
        anyhow::ensure!(
            self.resident.is_none(),
            "resident tensors must use their shared views instead of runtime packing"
        );
        anyhow::ensure!(!names.is_empty(), "cannot pack an empty tensor list");
        let mut packed = Vec::<u8>::new();
        let mut dtype = None;
        let mut packed_shape: Option<Vec<i32>> = None;
        for name in names {
            let (shard_name, raw_name) = self
                .tensors
                .get(name)
                .ok_or_else(|| anyhow::anyhow!("lazy checkpoint tensor {name} missing"))?;
            let shard = self.shard(shard_name)?;
            let (bytes, tensor_dtype, shape) = shard.tensor_bytes(raw_name).ok_or_else(|| {
                anyhow::anyhow!("tensor {raw_name} absent from shard {shard_name}")
            })?;
            match (&mut dtype, &mut packed_shape) {
                (None, None) => {
                    anyhow::ensure!(!shape.is_empty(), "cannot pack scalar tensor {name}");
                    dtype = Some(tensor_dtype);
                    packed_shape = Some(shape.to_vec());
                }
                (Some(expected_dtype), Some(expected_shape)) => {
                    anyhow::ensure!(
                        *expected_dtype == tensor_dtype
                            && expected_shape.len() == shape.len()
                            && expected_shape[1..] == shape[1..],
                        "incompatible tensor {name} in packed tensor list"
                    );
                    expected_shape[0] = expected_shape[0]
                        .checked_add(shape[0])
                        .ok_or_else(|| anyhow::anyhow!("packed tensor shape overflow"))?;
                }
                _ => unreachable!("packed metadata is initialized together"),
            }
            let expected_bytes = shape
                .iter()
                .try_fold(1usize, |n, &d| n.checked_mul(usize::try_from(d).ok()?));
            let expected_bytes = expected_bytes
                .and_then(|n| n.checked_mul(tensor_dtype.size_of()))
                .ok_or_else(|| anyhow::anyhow!("tensor {name} byte size overflow"))?;
            anyhow::ensure!(
                bytes.len() == expected_bytes,
                "tensor {name} has {} bytes; expected {expected_bytes}",
                bytes.len()
            );
            if packed.capacity() == 0 {
                let capacity = expected_bytes
                    .checked_mul(names.len())
                    .ok_or_else(|| anyhow::anyhow!("packed tensor byte size overflow"))?;
                packed.reserve_exact(capacity);
            }
            packed.extend_from_slice(bytes);
        }
        self.copy_bytes(
            &packed,
            dtype.expect("nonempty packed tensors have dtype"),
            &packed_shape.expect("nonempty packed tensors have shape"),
        )
        .map_err(|e| anyhow::anyhow!("loading packed expert tensors: {e}"))
    }

    /// Return a no-copy view of one tensor in the checkpoint mapping. The
    /// GLM expert address-table path keeps these arrays alive with the model,
    /// so router-selected weights remain directly addressable by the GPU.
    pub fn get_mapped(&self, name: &str) -> anyhow::Result<D::Tensor> {
        if let Some(resident) = &self.resident {
            return resident.get(name);
        }
        let (shard_name, raw_name) = self
            .tensors
            .get(name)
            .ok_or_else(|| anyhow::anyhow!("lazy checkpoint tensor {name} missing"))?;
        let shard = self.shard(shard_name)?;
        if let Some(array) = shard.tensor_array_shard_view(raw_name)? {
            return Ok(array);
        }
        if let Some(array) = shard.tensor_array(raw_name)? {
            return Ok(array);
        }
        (self.on_copy)(raw_name, shard_name);
        self.get_owned(name)
    }

    /// Return packed quantized weights through a page-aligned view of the
    /// individual tensor. Byte loads permit payload offsets that are not
    /// aligned to the dtype; tight mappings limit amplification for scattered
    /// expert tensors in MLX shards.
    pub fn get_mapped_weight_bytes(&self, name: &str) -> anyhow::Result<D::Tensor> {
        if let Some(resident) = &self.resident {
            return resident.get_bytes(name);
        }
        let (shard_name, raw_name) = self
            .tensors
            .get(name)
            .ok_or_else(|| anyhow::anyhow!("lazy checkpoint tensor {name} missing"))?;
        let shard = self.shard(shard_name)?;
        if let Some(array) = shard.tensor_byte_array_shard_view(raw_name)? {
            return Ok(array);
        }
        if let Some(array) = shard.tensor_byte_array_individual(raw_name)? {
            return Ok(array);
        }
        let (bytes, _, _) = shard
            .tensor_bytes(raw_name)
            .ok_or_else(|| anyhow::anyhow!("tensor {raw_name} absent from shard {shard_name}"))?;
        let byte_len = i32::try_from(bytes.len())
            .with_context(|| format!("mapped weight tensor {raw_name} exceeds byte-view limits"))?;
        self.copy_bytes(bytes, Dtype::U8, &[byte_len])
    }

    /// Advise the OS that clean pages can be reclaimed after mapped expert
    /// views are established. Their external device buffers keep each mapping
    /// alive, and later GPU reads can fault clean pages back from disk.
    pub fn discard_cached_pages(&self) -> anyhow::Result<()> {
        if self.resident.is_some() {
            return Ok(());
        }
        let shards = self
            .shards
            .lock()
            .map_err(|_| anyhow::anyhow!("lazy checkpoint shard cache poisoned"))?;
        for shard in shards.values() {
            shard.discard_cached_pages()?;
        }
        Ok(())
    }
}
