//! Backend-independent imports and bounded mapped views of MLX payloads.
//! The engine chooses sharing/copy policy; the format parser never sees a device.
use anyhow::Context;
use lisa_backend::{ScalarType as Dtype, StorageDevice};
use memmap2::Advice;
use std::{
    collections::HashMap,
    ffi::c_void,
    path::Path,
    ptr::NonNull,
    sync::{Arc, Mutex},
};
/// Bounded window size for generic mapped tensors and the safe fallback path.
const MAPPED_EXPERT_WINDOW_BYTES: usize = 4 * 1024 * 1024;
const PREFETCH_SKIP_RESIDENT_PERCENT: usize = 90;

pub use lisa_mlx::{Checkpoint, read_index_bytes};

/// device import adapter over a portable MLX checkpoint mapping.
/// The format crate never sees a device buffer or a device policy.
pub struct Shard<D: StorageDevice> {
    device: D,
    pub source: lisa_mlx::Shard,
    pub mapped_buffers: Mutex<HashMap<(usize, usize), D::Tensor>>,
}

pub fn scalar_type(dtype: lisa_mlx::Dtype) -> Dtype {
    match dtype {
        lisa_mlx::Dtype::Bool => Dtype::Bool,
        lisa_mlx::Dtype::Uint8 => Dtype::U8,
        lisa_mlx::Dtype::Uint32 => Dtype::U32,
        lisa_mlx::Dtype::Int32 => Dtype::I32,
        lisa_mlx::Dtype::Int64 => Dtype::I64,
        lisa_mlx::Dtype::Float16 => Dtype::F16,
        lisa_mlx::Dtype::Float32 => Dtype::F32,
        lisa_mlx::Dtype::Bfloat16 => Dtype::BF16,
    }
}

impl<D: StorageDevice + Sync> Shard<D>
where
    D::Tensor: Clone + Send,
{
    pub fn device(&self) -> &D {
        &self.device
    }

    pub fn open(device: D, path: &Path) -> anyhow::Result<Self> {
        Ok(Self {
            device,
            source: lisa_mlx::Shard::open(path)?,
            mapped_buffers: Mutex::new(HashMap::new()),
        })
    }

    pub fn tensor_bytes(&self, name: &str) -> Option<(&[u8], Dtype, &[i32])> {
        self.source
            .tensor_bytes(name)
            .map(|(bytes, dtype, shape)| (bytes, scalar_type(dtype), shape))
    }

    pub fn tensor_names(&self) -> impl Iterator<Item = &String> {
        self.source.tensor_names()
    }

    /// Tell the OS that the file-backed pages used during this shard's load
    /// can be discarded. External device buffers retain the mapping itself, so
    /// mapped tensors can fault their clean pages back in when inference uses
    /// them; copied tensors no longer need their source pages resident.
    pub fn discard_cached_pages(&self) -> anyhow::Result<()> {
        // SAFETY: the mmap is read-only, no Rust slice into it is live here,
        // and the model treats external checkpoint buffers as immutable. The
        // OS may discard clean pages, which subsequent reads fault back from
        // the same file without changing the mapped contents.
        unsafe {
            self.source
                .mapping()
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
            .filter_map(|name| self.source.tensor_info(name))
            .filter_map(|info| {
                let start = info.start / page_size * page_size;
                let end = info
                    .end
                    .div_ceil(page_size)
                    .saturating_mul(page_size)
                    .min(self.source.mapping().len());
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
                        .source
                        .mapping()
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

    pub fn mapped_range_mostly_resident(
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
        let address = unsafe { self.source.mapping().as_ptr().add(start) }.cast::<c_void>();
        let result = unsafe { mincore(address, length, residency.as_mut_ptr().cast()) };
        if result != 0 {
            return None;
        }
        let resident = residency.iter().filter(|&&page| page & 1 != 0).count();
        Some(resident.saturating_mul(100) >= pages.saturating_mul(PREFETCH_SKIP_RESIDENT_PERCENT))
    }

    /// Return a zero-copy tensor view when its payload fits in one page-aligned
    /// device buffer window. Unsupported layouts use the ordinary copy path.
    pub fn tensor_array(&self, name: &str) -> anyhow::Result<Option<D::Tensor>> {
        self.tensor_array_with_window(name, MAPPED_EXPERT_WINDOW_BYTES)
    }

    /// Return a typed tensor view backed only by its own page-aligned payload.
    /// Whole-shard device views can wire unrelated routed experts into memory,
    /// so model weights that stay resident use these tight windows instead.
    pub fn tensor_array_individual(&self, name: &str) -> anyhow::Result<Option<D::Tensor>> {
        let Some(info) = self.source.tensor_info(name) else {
            return Ok(None);
        };
        self.tensor_array_with_window(name, info.end.saturating_sub(info.start))
    }

    /// Return an individually mapped typed view without retaining its device
    /// buffer in the shard. Owners such as the routed-expert LRU then control
    /// the device buffer lifetime directly.
    pub fn tensor_array_individual_uncached(
        &self,
        name: &str,
    ) -> anyhow::Result<Option<D::Tensor>> {
        let Some(info) = self.source.tensor_info(name) else {
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
        match self
            .device
            .byte_view(&buffer, offset_bytes, &info.shape, scalar_type(info.dtype))
        {
            Ok(array) => Ok(Some(array)),
            Err(_) => Ok(None),
        }
    }

    fn tensor_array_with_window(
        &self,
        name: &str,
        window_limit: usize,
    ) -> anyhow::Result<Option<D::Tensor>> {
        let Some(info) = self.source.tensor_info(name) else {
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
        match self
            .device
            .byte_view(&buffer, offset_bytes, &info.shape, scalar_type(info.dtype))
        {
            Ok(array) => Ok(Some(array)),
            Err(_) => Ok(None),
        }
    }

    /// Return an untyped byte view of a tensor in a bounded mapping window.
    /// Byte loads allow packed U32 quantized weights whose safetensors payload
    /// begins at a non-U32-aligned offset to remain zero-copy and GPU-readable.
    pub fn tensor_byte_array(&self, name: &str) -> anyhow::Result<Option<D::Tensor>> {
        self.tensor_byte_array_with_window(name, MAPPED_EXPERT_WINDOW_BYTES)
    }

    /// Map one packed tensor with only its page-aligned byte range. GLM expert
    /// weights are scattered through the shards; sharing a 4 MiB window for
    /// each tensor retained tens of GiB of unrelated mapped data.
    pub fn tensor_byte_array_individual(&self, name: &str) -> anyhow::Result<Option<D::Tensor>> {
        let page_size = unsafe { getpagesize() } as usize;
        self.tensor_byte_array_with_window(name, page_size)
    }

    /// Map a packed tensor's byte range without retaining the device buffer in
    /// the shard cache. This is the routed-expert cache's bounded-view path.
    pub fn tensor_byte_array_individual_uncached(
        &self,
        name: &str,
    ) -> anyhow::Result<Option<D::Tensor>> {
        let Some(info) = self.source.tensor_info(name) else {
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
        match self
            .device
            .byte_view(&buffer, offset_bytes, &[byte_len], Dtype::U8)
        {
            Ok(array) => Ok(Some(array)),
            Err(_) => Ok(None),
        }
    }

    /// Return a byte view over one tensor through a shared whole-shard device
    /// buffer when the shard fits the device's maximum buffer length. This
    /// mirrors ds4's model-view layout: expert addresses point into a small
    /// number of shard allocations instead of creating one device allocation
    /// per expert tensor. The file remains mmap-backed; creating the view does
    /// not fault the shard payload into memory.
    pub fn tensor_byte_array_shard_view(&self, name: &str) -> anyhow::Result<Option<D::Tensor>> {
        let Some(info) = self.source.tensor_info(name) else {
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
        let Some(buffer_length) = self.source.mapping().len().checked_add(page_size - 1) else {
            return Ok(None);
        };
        let buffer_length = buffer_length / page_size * page_size;
        if buffer_length > self.device.max_allocation_bytes() {
            return Ok(None);
        }
        let Some(buffer) = self.mapped_buffer(0, buffer_length)? else {
            return Ok(None);
        };
        match self
            .device
            .byte_view(&buffer, info.start, &[byte_len], Dtype::U8)
        {
            Ok(array) => Ok(Some(array)),
            Err(_) => Ok(None),
        }
    }

    /// Return a typed tensor view through the same whole-shard device buffer
    /// used by packed expert weights. Keeping scales and biases on this backing
    /// allocation avoids overlapping per-tensor device views of the same mmap
    /// pages in the model-wide residency set.
    pub fn tensor_array_shard_view(&self, name: &str) -> anyhow::Result<Option<D::Tensor>> {
        let Some(info) = self.source.tensor_info(name) else {
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
        let Some(buffer_length) = self.source.mapping().len().checked_add(page_size - 1) else {
            return Ok(None);
        };
        let buffer_length = buffer_length / page_size * page_size;
        if buffer_length > self.device.max_allocation_bytes() {
            return Ok(None);
        }
        let Some(buffer) = self.mapped_buffer(0, buffer_length)? else {
            return Ok(None);
        };
        match self
            .device
            .byte_view(&buffer, info.start, &info.shape, scalar_type(info.dtype))
        {
            Ok(array) => Ok(Some(array)),
            Err(_) => Ok(None),
        }
    }

    fn tensor_byte_array_with_window(
        &self,
        name: &str,
        window_limit: usize,
    ) -> anyhow::Result<Option<D::Tensor>> {
        let Some((buffer, offset_bytes, payload_bytes)) =
            self.mapped_tensor_buffer(name, window_limit)?
        else {
            return Ok(None);
        };
        let Ok(byte_len) = i32::try_from(payload_bytes) else {
            return Ok(None);
        };
        match self
            .device
            .byte_view(&buffer, offset_bytes, &[byte_len], Dtype::U8)
        {
            Ok(array) => Ok(Some(array)),
            Err(_) => Ok(None),
        }
    }

    fn mapped_tensor_buffer(
        &self,
        name: &str,
        window_limit: usize,
    ) -> anyhow::Result<Option<(D::Tensor, usize, usize)>> {
        self.mapped_tensor_buffer_inner(name, window_limit, true)
    }

    fn mapped_tensor_buffer_uncached(
        &self,
        name: &str,
        window_limit: usize,
    ) -> anyhow::Result<Option<(D::Tensor, usize, usize)>> {
        self.mapped_tensor_buffer_inner(name, window_limit, false)
    }

    fn mapped_tensor_buffer_inner(
        &self,
        name: &str,
        window_limit: usize,
        retain_buffer: bool,
    ) -> anyhow::Result<Option<(D::Tensor, usize, usize)>> {
        let Some(info) = self.source.tensor_info(name) else {
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
        let max_length = self.device.max_allocation_bytes();
        let Some((chunk_start, buffer_length)) = mapped_tensor_window(
            info.start,
            info.end,
            self.source.mapping().len(),
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

    fn mapped_buffer(&self, start: usize, length: usize) -> anyhow::Result<Option<D::Tensor>> {
        let mut buffers = self.mapped_buffers.lock().unwrap();
        let key = (start, length);
        if let Some(buffer) = buffers.get(&key) {
            return Ok(Some(buffer.clone()));
        }
        let Some(buffer) = self.new_mapped_buffer(start, length)? else {
            return Ok(None);
        };
        buffers.insert(key, buffer.clone());
        Ok(Some(buffer))
    }

    fn mapped_buffer_uncached(
        &self,
        start: usize,
        length: usize,
    ) -> anyhow::Result<Option<D::Tensor>> {
        self.new_mapped_buffer(start, length)
    }

    fn new_mapped_buffer(&self, start: usize, length: usize) -> anyhow::Result<Option<D::Tensor>> {
        let Some(end) = start.checked_add(length) else {
            return Ok(None);
        };
        let page_size = unsafe { getpagesize() } as usize;
        if page_size == 0
            || start % page_size != 0
            || end > self.source.mapping().len().div_ceil(page_size) * page_size
        {
            return Ok(None);
        }
        let pointer =
            unsafe { NonNull::new_unchecked(self.source.mapping().as_ptr().add(start) as *mut u8) };
        let owner: Arc<dyn std::any::Any + Send + Sync> = self.source.mapping().clone();
        // mmap covers the final partial page. The retained mapping owns this
        // exact immutable region; the device never receives its filename.
        let storage = unsafe { lisa_backend::HostStorage::new(pointer, length, owner) };
        let Ok(pages) = i32::try_from(length / page_size) else {
            return Ok(None);
        };
        let shape = [pages, page_size as i32];
        let Ok(array) = self.device.pull(
            lisa_backend::TensorSource::Shared {
                storage,
                offset: 0,
                shape: &shape,
                dtype: lisa_backend::ScalarType::U8,
            },
            lisa_backend::ImportOptions {
                mode: lisa_backend::ImportMode::SharedRequired,
            },
        ) else {
            return Ok(None);
        };
        Ok(Some(array))
    }
}

/// Pick a page-aligned external-buffer window that contains one tensor. Nearby
/// tensors share windows up to `window_limit`. If a tensor straddles its chunk
/// boundary, shift the window back to the tensor's page so eligible tensors
/// do not fall back to copies because of their safetensors offset.
pub fn mapped_tensor_window(
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
