//! Safetensors checkpoint loading: shard index, mmap, name filtering, and
//! materialization into native arrays.

use std::collections::{BTreeMap, HashMap};
use std::ffi::c_void;
use std::io::Read;
use std::path::{Component, Path, PathBuf};
use std::ptr::NonNull;
use std::sync::{Arc, Mutex};

use anyhow::{Context, bail, ensure};
use lisa_mlx::{Array, Dtype};
use memmap2::{Advice, Mmap};

/// Bounded window size for generic mapped tensors and the safe fallback path.
const MAPPED_EXPERT_WINDOW_BYTES: usize = 4 * 1024 * 1024;
const PREFETCH_SKIP_RESIDENT_PERCENT: usize = 90;

/// Bound metadata reads independently of checkpoint payload size.
pub(crate) fn read_index_bytes(path: &Path) -> anyhow::Result<Vec<u8>> {
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
struct TensorInfo {
    dtype: Dtype,
    shape: Vec<i32>,
    /// Absolute byte offset of the tensor payload within the shard file.
    start: usize,
    end: usize,
}

/// An mmap'd safetensors shard.
pub struct Shard {
    map: Arc<Mmap>,
    tensors: HashMap<String, TensorInfo>,
    mapped_buffers: Mutex<HashMap<(usize, usize), Arc<lisa_mlx::runtime::Buffer>>>,
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
        Ok(Self {
            map,
            tensors,
            mapped_buffers: Mutex::new(HashMap::new()),
        })
    }

    /// Raw byte slice of one tensor.
    pub fn tensor_bytes(&self, name: &str) -> Option<(&[u8], Dtype, &[i32])> {
        let info = self.tensors.get(name)?;
        Some((&self.map[info.start..info.end], info.dtype, &info.shape[..]))
    }

    pub fn tensor_names(&self) -> impl Iterator<Item = &String> {
        self.tensors.keys()
    }

    /// Tell the OS that the file-backed pages used during this shard's load
    /// can be discarded. External Metal buffers retain the mapping itself, so
    /// mapped tensors can fault their clean pages back in when inference uses
    /// them; copied tensors no longer need their source pages resident.
    pub fn discard_cached_pages(&self) -> anyhow::Result<()> {
        // SAFETY: the mmap is read-only, no Rust slice into it is live here,
        // and the model treats external checkpoint buffers as immutable. The
        // OS may discard clean pages, which subsequent reads fault back from
        // the same file without changing the mapped contents.
        unsafe {
            self.map
                .unchecked_advise(memmap2::UncheckedAdvice::DontNeed)
        }
        .with_context(|| "advising safetensors mapping pages")
    }

    /// Hint the OS to fetch a set of selected tensor payloads by advising
    /// merged, page-aligned ranges. This keeps the advice count proportional
    /// to the selected regions rather than issuing one syscall per tensor.
    pub fn prefetch_tensor_ranges(&self, names: &[String]) -> (usize, usize) {
        if names.is_empty() {
            return (0, 0);
        }
        let page_size = unsafe { getpagesize() } as usize;
        if page_size == 0 {
            return (0, 0);
        }

        let mut ranges = names
            .iter()
            .filter_map(|name| self.tensors.get(name))
            .filter_map(|info| {
                let start = info.start / page_size * page_size;
                let end = info
                    .end
                    .div_ceil(page_size)
                    .saturating_mul(page_size)
                    .min(self.map.len());
                (start < end).then_some((start, end))
            })
            .collect::<Vec<_>>();
        ranges.sort_unstable_by_key(|&(start, end)| (start, end));

        let mut merged = Vec::<(usize, usize)>::with_capacity(ranges.len());
        for (start, end) in ranges {
            if let Some(last) = merged.last_mut()
                && start <= last.1.saturating_add(page_size)
            {
                last.1 = last.1.max(end);
            } else {
                merged.push((start, end));
            }
        }

        let advise_chunk = |ranges: &[(usize, usize)]| {
            ranges
                .iter()
                .fold((0usize, 0usize), |(count, bytes), &(start, end)| {
                    let length = end.saturating_sub(start);
                    if self
                        .mapped_range_mostly_resident(start, length, page_size)
                        .unwrap_or(false)
                    {
                        return (count, bytes);
                    }
                    if self
                        .map
                        .advise_range(Advice::WillNeed, start, length)
                        .is_ok()
                    {
                        (count.saturating_add(1), bytes.saturating_add(length))
                    } else {
                        (count, bytes)
                    }
                })
        };
        if merged.len() < 32 {
            return advise_chunk(&merged);
        }

        // Large routed-prefill selections can resolve to hundreds of disjoint
        // mapped ranges. Submit those bounded ranges concurrently so the OS
        // can overlap page reads; small decode selections stay on the cheaper
        // serial path. The per-pool cache budget bounds the selected payload
        // before this helper is called.
        let workers = std::thread::available_parallelism()
            .map_or(1, usize::from)
            .min(8)
            .min(merged.len());
        let chunk_size = merged.len().div_ceil(workers);
        std::thread::scope(|scope| {
            let handles = merged
                .chunks(chunk_size)
                .map(|chunk| scope.spawn(move || advise_chunk(chunk)))
                .collect::<Vec<_>>();
            handles.into_iter().fold((0usize, 0usize), |total, handle| {
                let result = handle.join().unwrap_or_default();
                (
                    total.0.saturating_add(result.0),
                    total.1.saturating_add(result.1),
                )
            })
        })
    }

    fn mapped_range_mostly_resident(
        &self,
        start: usize,
        length: usize,
        page_size: usize,
    ) -> Option<bool> {
        if page_size == 0 || start % page_size != 0 || length == 0 {
            return None;
        }
        let pages = length.div_ceil(page_size);
        let mut residency = vec![0u8; pages];
        let address = unsafe { self.map.as_ptr().add(start) }.cast::<c_void>();
        let result = unsafe { mincore(address, length, residency.as_mut_ptr().cast()) };
        if result != 0 {
            return None;
        }
        let resident = residency.iter().filter(|&&page| page & 1 != 0).count();
        Some(resident.saturating_mul(100) >= pages.saturating_mul(PREFETCH_SKIP_RESIDENT_PERCENT))
    }

    /// Return a zero-copy tensor view when its payload fits in one page-aligned
    /// Metal buffer window. Unsupported layouts use the ordinary copy path.
    pub fn tensor_array(&self, name: &str) -> anyhow::Result<Option<Array>> {
        self.tensor_array_with_window(name, MAPPED_EXPERT_WINDOW_BYTES)
    }

    /// Return a typed tensor view backed only by its own page-aligned payload.
    /// Whole-shard Metal views can wire unrelated routed experts into memory,
    /// so model weights that stay resident use these tight windows instead.
    pub fn tensor_array_individual(&self, name: &str) -> anyhow::Result<Option<Array>> {
        let Some(info) = self.tensors.get(name) else {
            return Ok(None);
        };
        self.tensor_array_with_window(name, info.end.saturating_sub(info.start))
    }

    /// Return an individually mapped typed view without retaining its Metal
    /// buffer in the shard. Owners such as the routed-expert LRU then control
    /// the Metal buffer lifetime directly.
    pub fn tensor_array_individual_uncached(&self, name: &str) -> anyhow::Result<Option<Array>> {
        let Some(info) = self.tensors.get(name) else {
            return Ok(None);
        };
        let Some((buffer, offset_bytes, payload_bytes)) =
            self.mapped_tensor_buffer_uncached(name, info.end.saturating_sub(info.start))?
        else {
            return Ok(None);
        };
        let element_size = info.dtype.size_of();
        if offset_bytes % element_size != 0 {
            return Ok(None);
        }
        let expected_bytes = info
            .shape
            .iter()
            .try_fold(1usize, |n, &d| n.checked_mul(d as usize))
            .and_then(|n| n.checked_mul(element_size));
        if expected_bytes != Some(payload_bytes) {
            return Ok(None);
        }
        match Array::from_buffer_offset(buffer, offset_bytes, &info.shape, info.dtype) {
            Ok(array) => Ok(Some(array)),
            Err(_) => Ok(None),
        }
    }

    fn tensor_array_with_window(
        &self,
        name: &str,
        window_limit: usize,
    ) -> anyhow::Result<Option<Array>> {
        let Some(info) = self.tensors.get(name) else {
            return Ok(None);
        };
        let element_size = info.dtype.size_of();
        let Some((buffer, offset_bytes, payload_bytes)) =
            self.mapped_tensor_buffer(name, window_limit)?
        else {
            return Ok(None);
        };
        if offset_bytes % element_size != 0 {
            return Ok(None);
        }
        let expected_bytes = info
            .shape
            .iter()
            .try_fold(1usize, |n, &d| n.checked_mul(d as usize))
            .and_then(|n| n.checked_mul(element_size));
        if expected_bytes != Some(payload_bytes) {
            return Ok(None);
        }
        match Array::from_buffer_offset(buffer, offset_bytes, &info.shape, info.dtype) {
            Ok(array) => Ok(Some(array)),
            Err(_) => Ok(None),
        }
    }

    /// Return an untyped byte view of a tensor in a bounded mapping window.
    /// Byte loads allow packed U32 quantized weights whose safetensors payload
    /// begins at a non-U32-aligned offset to remain zero-copy and GPU-readable.
    pub fn tensor_byte_array(&self, name: &str) -> anyhow::Result<Option<Array>> {
        self.tensor_byte_array_with_window(name, MAPPED_EXPERT_WINDOW_BYTES)
    }

    /// Map one packed tensor with only its page-aligned byte range. GLM expert
    /// weights are scattered through the shards; sharing a 4 MiB window for
    /// each tensor retained tens of GiB of unrelated mapped data.
    pub fn tensor_byte_array_individual(&self, name: &str) -> anyhow::Result<Option<Array>> {
        let page_size = unsafe { getpagesize() } as usize;
        self.tensor_byte_array_with_window(name, page_size)
    }

    /// Map a packed tensor's byte range without retaining the Metal buffer in
    /// the shard cache. This is the routed-expert cache's bounded-view path.
    pub fn tensor_byte_array_individual_uncached(
        &self,
        name: &str,
    ) -> anyhow::Result<Option<Array>> {
        let Some(info) = self.tensors.get(name) else {
            return Ok(None);
        };
        let payload_bytes = info.end.saturating_sub(info.start);
        let Some((buffer, offset_bytes, payload_bytes)) =
            self.mapped_tensor_buffer_uncached(name, payload_bytes)?
        else {
            return Ok(None);
        };
        let Ok(byte_len) = i32::try_from(payload_bytes) else {
            return Ok(None);
        };
        match Array::from_buffer_offset(buffer, offset_bytes, &[byte_len], Dtype::Uint8) {
            Ok(array) => Ok(Some(array)),
            Err(_) => Ok(None),
        }
    }

    /// Return a byte view over one tensor through a shared whole-shard Metal
    /// buffer when the shard fits the device's maximum buffer length. This
    /// mirrors ds4's model-view layout: expert addresses point into a small
    /// number of shard allocations instead of creating one Metal allocation
    /// per expert tensor. The file remains mmap-backed; creating the view does
    /// not fault the shard payload into memory.
    pub fn tensor_byte_array_shard_view(&self, name: &str) -> anyhow::Result<Option<Array>> {
        let Some(info) = self.tensors.get(name) else {
            return Ok(None);
        };
        let payload_bytes = info.end.saturating_sub(info.start);
        let Ok(byte_len) = i32::try_from(payload_bytes) else {
            return Ok(None);
        };
        if payload_bytes == 0 {
            return Ok(None);
        }

        let page_size = unsafe { getpagesize() } as usize;
        if page_size == 0 {
            return Ok(None);
        }
        let Some(buffer_length) = self.map.len().checked_add(page_size - 1) else {
            return Ok(None);
        };
        let buffer_length = buffer_length / page_size * page_size;
        if buffer_length > Array::max_buffer_length() {
            return Ok(None);
        }
        let Some(buffer) = self.mapped_buffer(0, buffer_length)? else {
            return Ok(None);
        };
        match Array::from_buffer_offset(buffer, info.start, &[byte_len], Dtype::Uint8) {
            Ok(array) => Ok(Some(array)),
            Err(_) => Ok(None),
        }
    }

    /// Return a typed tensor view through the same whole-shard Metal buffer
    /// used by packed expert weights. Keeping scales and biases on this backing
    /// allocation avoids overlapping per-tensor Metal views of the same mmap
    /// pages in the model-wide residency set.
    pub fn tensor_array_shard_view(&self, name: &str) -> anyhow::Result<Option<Array>> {
        let Some(info) = self.tensors.get(name) else {
            return Ok(None);
        };
        let payload_bytes = info.end.saturating_sub(info.start);
        let expected_bytes = info
            .shape
            .iter()
            .try_fold(1usize, |n, &d| n.checked_mul(d as usize))
            .and_then(|n| n.checked_mul(info.dtype.size_of()));
        if payload_bytes == 0 || expected_bytes != Some(payload_bytes) {
            return Ok(None);
        }

        let page_size = unsafe { getpagesize() } as usize;
        if page_size == 0 {
            return Ok(None);
        }
        let Some(buffer_length) = self.map.len().checked_add(page_size - 1) else {
            return Ok(None);
        };
        let buffer_length = buffer_length / page_size * page_size;
        if buffer_length > Array::max_buffer_length() {
            return Ok(None);
        }
        let Some(buffer) = self.mapped_buffer(0, buffer_length)? else {
            return Ok(None);
        };
        match Array::from_buffer_offset(buffer, info.start, &info.shape, info.dtype) {
            Ok(array) => Ok(Some(array)),
            Err(_) => Ok(None),
        }
    }

    fn tensor_byte_array_with_window(
        &self,
        name: &str,
        window_limit: usize,
    ) -> anyhow::Result<Option<Array>> {
        let Some((buffer, offset_bytes, payload_bytes)) =
            self.mapped_tensor_buffer(name, window_limit)?
        else {
            return Ok(None);
        };
        let Ok(byte_len) = i32::try_from(payload_bytes) else {
            return Ok(None);
        };
        match Array::from_buffer_offset(buffer, offset_bytes, &[byte_len], Dtype::Uint8) {
            Ok(array) => Ok(Some(array)),
            Err(_) => Ok(None),
        }
    }

    fn mapped_tensor_buffer(
        &self,
        name: &str,
        window_limit: usize,
    ) -> anyhow::Result<Option<(Arc<lisa_mlx::runtime::Buffer>, usize, usize)>> {
        self.mapped_tensor_buffer_inner(name, window_limit, true)
    }

    fn mapped_tensor_buffer_uncached(
        &self,
        name: &str,
        window_limit: usize,
    ) -> anyhow::Result<Option<(Arc<lisa_mlx::runtime::Buffer>, usize, usize)>> {
        self.mapped_tensor_buffer_inner(name, window_limit, false)
    }

    fn mapped_tensor_buffer_inner(
        &self,
        name: &str,
        window_limit: usize,
        retain_buffer: bool,
    ) -> anyhow::Result<Option<(Arc<lisa_mlx::runtime::Buffer>, usize, usize)>> {
        let Some(info) = self.tensors.get(name) else {
            return Ok(None);
        };
        let payload_bytes = info.end.saturating_sub(info.start);
        let expected_bytes = info
            .shape
            .iter()
            .try_fold(1usize, |n, &d| n.checked_mul(d as usize))
            .and_then(|n| n.checked_mul(info.dtype.size_of()));
        if expected_bytes != Some(payload_bytes) || payload_bytes == 0 {
            return Ok(None);
        }

        let page_size = unsafe { getpagesize() } as usize;
        let max_length = Array::max_buffer_length();
        let Some((chunk_start, buffer_length)) = mapped_tensor_window(
            info.start,
            info.end,
            self.map.len(),
            page_size,
            max_length,
            window_limit,
        ) else {
            return Ok(None);
        };
        let offset_bytes = info.start - chunk_start;
        let mapped = if retain_buffer {
            self.mapped_buffer(chunk_start, buffer_length)?
        } else {
            self.mapped_buffer_uncached(chunk_start, buffer_length)?
        };
        let Some(buffer) = mapped else {
            return Ok(None);
        };
        Ok(Some((buffer, offset_bytes, payload_bytes)))
    }

    fn mapped_buffer(
        &self,
        start: usize,
        length: usize,
    ) -> anyhow::Result<Option<Arc<lisa_mlx::runtime::Buffer>>> {
        let mut buffers = self.mapped_buffers.lock().unwrap();
        let key = (start, length);
        if let Some(buffer) = buffers.get(&key) {
            return Ok(Some(Arc::clone(buffer)));
        }
        let Some(buffer) = self.new_mapped_buffer(start, length)? else {
            return Ok(None);
        };
        buffers.insert(key, Arc::clone(&buffer));
        Ok(Some(buffer))
    }

    fn mapped_buffer_uncached(
        &self,
        start: usize,
        length: usize,
    ) -> anyhow::Result<Option<Arc<lisa_mlx::runtime::Buffer>>> {
        self.new_mapped_buffer(start, length)
    }

    fn new_mapped_buffer(
        &self,
        start: usize,
        length: usize,
    ) -> anyhow::Result<Option<Arc<lisa_mlx::runtime::Buffer>>> {
        let Some(end) = start.checked_add(length) else {
            return Ok(None);
        };
        let page_size = unsafe { getpagesize() } as usize;
        if page_size == 0
            || start % page_size != 0
            || end > self.map.len().div_ceil(page_size) * page_size
        {
            return Ok(None);
        }
        let pointer =
            unsafe { NonNull::new_unchecked(self.map.as_ptr().add(start) as *mut c_void) };
        let owner: Arc<dyn std::any::Any + Send + Sync> = self.map.clone();
        let Ok(buffer) = Array::buffer_from_external(pointer, length, owner, "safetensors_mmap")
        else {
            return Ok(None);
        };
        Ok(Some(buffer))
    }
}

/// Pick a page-aligned external-buffer window that contains one tensor. Nearby
/// tensors share windows up to `window_limit`. If a tensor straddles its chunk
/// boundary, shift the window back to the tensor's page so eligible tensors
/// do not fall back to copies because of their safetensors offset.
fn mapped_tensor_window(
    tensor_start: usize,
    tensor_end: usize,
    mapping_len: usize,
    page_size: usize,
    max_length: usize,
    window_limit: usize,
) -> Option<(usize, usize)> {
    if page_size == 0 || tensor_start >= tensor_end || tensor_end > mapping_len {
        return None;
    }
    let chunk_size = window_limit.min(max_length) / page_size * page_size;
    if chunk_size < page_size {
        return None;
    }

    let mut window_start = tensor_start / chunk_size * chunk_size;
    let chunk_end = window_start.saturating_add(chunk_size).min(mapping_len);
    let mut window_length = chunk_end.saturating_sub(window_start).div_ceil(page_size) * page_size;
    if tensor_end > chunk_end || window_length > max_length {
        window_start = tensor_start / page_size * page_size;
        window_length = tensor_end
            .checked_sub(window_start)?
            .div_ceil(page_size)
            .checked_mul(page_size)?;
    }
    if tensor_start < window_start
        || tensor_end.saturating_sub(window_start) > window_length
        || window_length > max_length
    {
        return None;
    }
    Some((window_start, window_length))
}

unsafe extern "C" {
    fn getpagesize() -> i32;
    fn mincore(address: *const c_void, length: usize, residency: *mut c_void) -> i32;
}

/// Checkpoint view over a model directory: index + shard file names.
pub struct Checkpoint {
    pub dir: PathBuf,
    /// Original checkpoint tensor name -> shard file name.
    pub weight_map: HashMap<String, String>,
}

/// On-demand tensor views for GLM routed experts. Payloads stay file-backed
/// and are exposed as page-aligned external Metal buffers when possible.
/// A prepared resident store always returns shared model views, including
/// from the compatibility methods named `get_owned`; it never copies weights.
pub struct LazyTensorStore {
    dir: PathBuf,
    tensors: HashMap<String, (String, String)>,
    shards: Mutex<HashMap<String, Arc<Shard>>>,
    resident: Option<crate::core::weight_arena::ResidentWeights>,
}

impl LazyTensorStore {
    pub fn open(dir: &Path) -> anyhow::Result<Self> {
        let checkpoint = Checkpoint::open(dir)?;
        Self::from_checkpoint(dir, &checkpoint)
    }

    /// Open a prepared, byte-validated resident checkpoint. Missing preparation
    /// is an error; this route never silently falls back to private copies.
    pub fn open_resident(dir: &Path) -> anyhow::Result<Self> {
        Self::open_resident_index(dir, &dir.join(".lisa-resident/index.json"))
    }

    pub fn open_resident_index(dir: &Path, index: &Path) -> anyhow::Result<Self> {
        let checkpoint = Checkpoint::open(dir)?;
        let mut store = Self::from_checkpoint(dir, &checkpoint)?;
        let mut source_files = checkpoint
            .weight_map
            .values()
            .map(PathBuf::from)
            .collect::<std::collections::HashSet<_>>();
        source_files.insert(PathBuf::from("model.safetensors.index.json"));
        let resident = crate::core::weight_arena::ResidentWeights::open(dir, index, &source_files)?;
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
    pub fn get(&self, name: &str) -> anyhow::Result<Array> {
        self.get_mapped(name)
    }

    /// The complete shared backing resources, deduplicated at model-view level.
    pub fn resident_buffers(&self) -> Vec<Arc<lisa_mlx::runtime::Buffer>> {
        self.resident
            .as_ref()
            .map_or_else(Vec::new, |r| r.buffers().to_vec())
    }

    /// Unique mapped file bytes; overlapping Metal views are counted once.
    pub fn resident_nbytes(&self) -> usize {
        self.resident.as_ref().map_or(0, |r| r.mapped_bytes())
    }

    /// Build a lazy tensor view from an index the caller already parsed.
    /// Reusing the checkpoint avoids reparsing its index and lets eager and
    /// lazy model loading share the same shard mappings and parsed headers.
    pub(crate) fn from_checkpoint(dir: &Path, checkpoint: &Checkpoint) -> anyhow::Result<Self> {
        let mut tensors = HashMap::new();
        tensors.reserve(checkpoint.weight_map.len());
        for (raw_name, shard_name) in &checkpoint.weight_map {
            if let Some(name) = sanitize_name(raw_name) {
                ensure!(
                    tensors
                        .insert(name.clone(), (shard_name.clone(), raw_name.clone()))
                        .is_none(),
                    "checkpoint tensor names collide after normalization: {name}"
                );
            }
        }
        Ok(Self {
            dir: dir.to_path_buf(),
            tensors,
            shards: Mutex::new(HashMap::new()),
            resident: None,
        })
    }

    pub(crate) fn shard(&self, shard_name: &str) -> anyhow::Result<Arc<Shard>> {
        let mut shards = self
            .shards
            .lock()
            .map_err(|_| anyhow::anyhow!("lazy checkpoint shard cache poisoned"))?;
        if let Some(shard) = shards.get(shard_name) {
            return Ok(Arc::clone(shard));
        }
        let shard = Arc::new(Shard::open(&self.dir.join(shard_name))?);
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

    /// Load one sanitized tensor into a private MLX allocation. Expert cache
    /// eviction can then release it without keeping the source shard wired in
    /// memory by a no-copy Metal buffer.
    pub fn get_owned(&self, name: &str) -> anyhow::Result<Array> {
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
        array_from_bytes(bytes, dtype, shape)
            .map_err(|e| anyhow::anyhow!("loading lazy tensor {name}: {e}"))
    }

    /// Copy one packed quantized tensor into a private byte array. Routed
    /// experts use this on demand so a Metal no-copy buffer cannot keep an
    /// entire checkpoint shard resident for the lifetime of the model.
    pub fn get_owned_weight_bytes(&self, name: &str) -> anyhow::Result<Array> {
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
        Ok(Array::from_slice(bytes, &[byte_len]))
    }

    /// Return a page-scoped typed view whose Metal buffer lifetime is owned by
    /// the caller. Unsupported or unaligned tensors fall back to a private
    /// allocation rather than retaining a view in the shard cache.
    pub fn get_mapped_individual(&self, name: &str) -> anyhow::Result<Array> {
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
    /// returned Metal buffer is retained only by the array, so an expert-cache
    /// eviction releases it without retaining the rest of the safetensors
    /// shard.
    pub fn get_mapped_weight_bytes_individual(&self, name: &str) -> anyhow::Result<Array> {
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
        Ok(shard.tensors[raw_name].end - shard.tensors[raw_name].start)
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
            .tensors
            .get(raw_name)
            .map(|info| info.dtype)
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
            .tensors
            .get(raw_name)
            .map(|info| info.shape.clone())
            .ok_or_else(|| anyhow::anyhow!("tensor {raw_name} absent from shard {shard_name}"))
    }

    /// Load same-shaped tensors into one contiguous MLX array. GLM expert
    /// safetensors payloads often start at offsets that cannot be represented
    /// as typed Metal views; packing their raw bytes here keeps one backing
    /// allocation per expert parameter instead of one allocation per expert.
    pub fn get_packed(&self, names: &[String]) -> anyhow::Result<Array> {
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
        array_from_bytes(
            &packed,
            dtype.expect("nonempty packed tensors have dtype"),
            &packed_shape.expect("nonempty packed tensors have shape"),
        )
        .map_err(|e| anyhow::anyhow!("loading packed expert tensors: {e}"))
    }

    /// Return a no-copy view of one tensor in the checkpoint mapping. The
    /// GLM expert address-table path keeps these arrays alive with the model,
    /// so router-selected weights remain directly addressable by the GPU.
    pub fn get_mapped(&self, name: &str) -> anyhow::Result<Array> {
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
        if lisa_mlx::trace::enabled_for("glm53-loader") {
            eprintln!(
                "[glm53-loader] copying tensor {raw_name} from {shard_name}: no legal mapped Metal window"
            );
        }
        self.get_owned(name)
    }

    /// Return packed quantized weights through a page-aligned view of the
    /// individual tensor. Byte loads permit payload offsets that are not
    /// aligned to the dtype; tight mappings limit amplification for scattered
    /// expert tensors in MLX shards.
    pub fn get_mapped_weight_bytes(&self, name: &str) -> anyhow::Result<Array> {
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
        Ok(Array::from_slice(bytes, &[byte_len]))
    }

    /// Advise the OS that clean pages can be reclaimed after mapped expert
    /// views are established. Their external Metal buffers keep each mapping
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

/// Create a native array from checked tensor bytes (copies onto the device).
pub fn array_from_bytes(
    bytes: &[u8],
    dtype: Dtype,
    shape: &[i32],
) -> lisa_mlx::error::Result<Array> {
    let expected = shape.iter().try_fold(dtype.size_of(), |bytes, &dim| {
        usize::try_from(dim)
            .ok()
            .and_then(|dim| bytes.checked_mul(dim))
    });
    if expected != Some(bytes.len()) {
        return Err(lisa_mlx::error::Exception::custom(
            "array_from_bytes: payload size differs from tensor shape".into(),
        ));
    }
    match dtype {
        Dtype::Bfloat16
        | Dtype::Uint32
        | Dtype::Uint8
        | Dtype::Int64
        | Dtype::Float32
        | Dtype::Float16
        | Dtype::Int32 => {
            // SAFETY: the complete source range was validated above; construction
            // copies it into an owned buffer before this slice can be released.
            Ok(unsafe { Array::from_raw_data(bytes.as_ptr().cast(), shape, dtype) })
        }
        _ => Err(lisa_mlx::error::Exception::custom(format!(
            "array_from_bytes: unsupported dtype {dtype:?}"
        ))),
    }
}

/// Sanitize a checkpoint tensor name:
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
    // rebuilt from configuration.
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
        for offset in (0..shard.map.len()).step_by(page) {
            std::hint::black_box(shard.map[offset]);
        }
        assert_eq!(
            shard.mapped_range_mostly_resident(0, shard.map.len(), page),
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
        if !lisa_mlx::device::metal_available() {
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
        if !lisa_mlx::device::metal_available() {
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
        if !lisa_mlx::device::metal_available() {
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
