//! Our own Metal runtime: device/queue, byte-accounted buffer pool with a real
//! sweep, command batching (`Commands`/`ComputeEncoder`) with in-session
//! barriers and cross-encoder fences, pipeline cache, and the `Math` compile
//! regimes.
//!
//! Everything the tree dispatches flows through `mdev.commands` on ONE queue:
//! Metal does not order work across queues, so any dispatch outside this
//! runtime would race the custom kernels it shares buffers with.
//!
//! The pool is byte-accounted and swept once past a cap (8 GiB); reuse is
//! gated on a full flush since the buffer's last
//! bind, and taken buffers are zeroed (the eager arrays have no graph retaining
//! intermediates, so a stale region a kernel reads before its producer writes
//! would otherwise carry garbage).

use std::collections::{HashMap, HashSet};
use std::ffi::c_void;
use block2::RcBlock;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, RwLock};

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBarrierScope, MTLBuffer, MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandEncoder,
    MTLCommandQueue, MTLCompileOptions, MTLComputeCommandEncoder, MTLComputePipelineState,
    MTLDataType, MTLDevice, MTLDispatchType, MTLFence, MTLFunction, MTLFunctionConstantValues,
    MTLLibrary, MTLMathFloatingPointFunctions, MTLMathMode, MTLCreateSystemDefaultDevice,
    MTLResource, MTLResourceOptions, MTLSize,
};

pub use crate::error::{Error, Result};

/// Kept for call sites that name the runtime error explicitly.
pub type RuntimeError = Error;

fn err(m: impl Into<String>) -> Error {
    Error::Msg(m.into())
}


pub static WAIT_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static DISPATCH_COUNT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static ALLOC_COUNT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static POOLHIT_COUNT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static ZERO_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static LABEL_COUNTS: std::sync::Mutex<Vec<(String, u64)>> = std::sync::Mutex::new(Vec::new());
pub static ENCODER_COUNT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// Set at startup when any allocation/dispatch tracing env var is present; keeps
/// the per-op hot paths free of atomics and env lookups otherwise.
static TRACE_ON: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);


/// Cached diagnostic env flag (read once; these sit in per-op hot paths and a
/// libc `getenv` per dispatch is measurable in samples).
pub fn env_flag(name: &'static str) -> bool {
    static CACHE: std::sync::OnceLock<std::collections::HashMap<String, bool>> =
        std::sync::OnceLock::new();
    let cache = CACHE.get_or_init(|| {
        let mut m = std::collections::HashMap::new();
        for k in [
            "LISA_NO_POOL_REUSE",
            "LISA_NO_ZERO_REUSE",
            "LISA_NO_SWEEP",
            "LISA_NO_KEEPALIVE",
            "LISA_SERIAL_ENCODER",
            "LISA_SYNC_ROTATE",
            "LISA_SYNC_EVERY_OP",
            "LISA_CHECK_REUSE",
            "LISA_BARRIER_TRACE",
            "LISA_WAIT_TRACE",
            "LISA_DISPATCH_TRACE",
            "LISA_PLAIN_REUSE",
            "LISA_NO_TRIM",
            "LISA_RMS_TRACE",
            "LISA_RMS_EVAL_TOP",
            "LISA_DUMP_NORMED",
            "LISA_SHAPE_DEBUG",
            "LISA_PROFILE_DECODE",
            "LISA_PROFILE",
            "LISA_PROFILE_MTP",
            "LISA_PROFILE_MOE",
            "LISA_DEBUG_MTP",
            "LISA_DEBUG_SESSION",
            "LISA_DEBUG_INDIRECT",
            "LISA_INDEXER_DEBUG",
            "LISA_QSA_PROF",
            "LISA_TOPK",
            "LISA_KEEP",
            "LISA_TRUNC",
            "LISA_DUMP_TOKENS",
            "LISA_LDTOKENS",
            "LISA_NO_WARMUP",
            "LISA_NO_SWEEP",
            "LISA_TOTALS",
            "LISA_SYNC_EXP",
            "LISA_MEM_TRACE",
            "LISA_NO_DECODE_COMPLETE",
            "LISA_GDN_SHAPES",
            "LISA_NOOP_DISPATCH",
            "LISA_NO_TRACKING",
            "LISA_LABEL_TRACE",
        ] {
            m.insert((*k).to_string(), std::env::var(k).is_ok());
        }
        m
    });
    cache.get(name).copied().unwrap_or(false)
}

fn ns(s: &str) -> Retained<NSString> {
    NSString::from_str(s)
}

// ─────────────────────────────── buffers ───────────────────────────────

static NEXT_BUFFER_ID: AtomicUsize = AtomicUsize::new(1);

pub struct Buffer {
    raw: Retained<ProtocolObject<dyn MTLBuffer>>,
    id: usize,
    len: usize,
    /// The command buffer in which this buffer was last bound. A pooled buffer
    /// may be reused once that command buffer has *completed* on the GPU (see
    /// `Commands::completed_watermark`). This is `strong_count == 1`
    /// reuse made sound: command buffers of one queue execute in commit order,
    /// so once the last one touching a buffer is done, nothing in flight reads
    /// or writes it.
    used_cb: AtomicU64,
}

unsafe impl Send for Buffer {}
unsafe impl Sync for Buffer {}

impl Buffer {
    fn wrap(raw: Retained<ProtocolObject<dyn MTLBuffer>>) -> Self {
        let len = raw.length();
        Self {
            raw,
            id: NEXT_BUFFER_ID.fetch_add(1, Ordering::Relaxed),
            len,
            used_cb: AtomicU64::new(0),
        }
    }

    /// Record the command buffer that last bound this buffer.
    pub fn mark_used_cb(&self, cb: u64) {
        self.used_cb.fetch_max(cb, Ordering::AcqRel);
    }

    /// True once the command buffer that last touched this buffer (if any) has
    /// completed on the GPU.
    pub fn reusable_after(&self, completed: u64) -> bool {
        let used = self.used_cb.load(Ordering::Acquire);
        used == 0 || used <= completed
    }

    /// Adopt an externally-owned `MTLBuffer` (the migration bridge shares
    /// buffers without copying). Ownership is ordinary
    /// objc2 refcounting, so neither side frees it while the other lives.
    pub fn shared(raw: Retained<ProtocolObject<dyn MTLBuffer>>) -> Self {
        Self::wrap(raw)
    }

    /// Retain a second owner of the same `MTLBuffer`.
    pub fn retain_shared(&self) -> Self {
        // SAFETY: `self.raw` is a valid object; retain bumps its count.
        let raw = unsafe { Retained::retain(Retained::as_ptr(&self.raw) as *mut _) }
            .expect("retain MTLBuffer");
        Self::wrap(raw)
    }

    pub fn mtlbuffer(&self) -> &ProtocolObject<dyn MTLBuffer> {
        &self.raw
    }

    /// A retained `MTLBuffer` handle for handing to another owner (the shim);
    /// the buffer stays alive through ordinary objc2 refcounting.
    pub fn retain_shared_mtlbuffer(&self) -> Retained<ProtocolObject<dyn MTLBuffer>> {
        self.retain_shared().raw.clone()
    }

    /// Stable identity for the encoder's dependency map: the `MTLBuffer` object.
    pub fn id(&self) -> usize {
        self.id
    }

    pub fn length(&self) -> usize {
        self.len
    }

    pub fn set_label(&self, label: &str) {
        self.raw.setLabel(Some(&ns(label)));
    }

    /// CPU pointer for shared-storage buffers.
    pub fn contents(&self) -> *mut u8 {
        self.raw.contents().as_ptr() as *mut u8
    }
}

impl AsRef<ProtocolObject<dyn MTLBuffer>> for Buffer {
    fn as_ref(&self) -> &ProtocolObject<dyn MTLBuffer> {
        &self.raw
    }
}

/// Sizes are rounded to a 1 KiB bucket so similar shapes reuse a buffer.
fn bucket_size(size: usize) -> usize {
    size.max(1).div_ceil(1024) * 1024
}

/// A size-bucketed buffer pool with a real, byte-accounted cap.
pub struct BufferPool {
    buckets: RwLock<HashMap<usize, Vec<Arc<Buffer>>>>,
    bytes: AtomicUsize,
    limit: AtomicUsize,
}

impl BufferPool {
    pub fn new() -> Self {
        Self {
            buckets: RwLock::new(HashMap::new()),
            bytes: AtomicUsize::new(0),
            limit: AtomicUsize::new(0),
        }
    }

    /// Bytes the pool may retain; `0` disables the cap.
    pub fn set_limit(&self, bytes: usize) {
        self.limit.store(bytes, Ordering::Relaxed);
    }

    pub fn bytes(&self) -> usize {
        self.bytes.load(Ordering::Relaxed)
    }

    /// The smallest free pooled buffer that can hold `size` and whose last
    /// command buffer has completed.
    pub fn take(&self, size: usize, completed: u64) -> Option<Arc<Buffer>> {
        if env_flag("LISA_NO_POOL_REUSE") {
            return None;
        }
        let plain = env_flag("LISA_PLAIN_REUSE");
        let eligible = |s: &Arc<Buffer>| -> bool {
            Arc::strong_count(s) == 1 && (plain || s.reusable_after(completed))
        };
        let buckets = self.buckets.read().unwrap();
        // Exact-fit first: most takes re-request the size just freed, and the
        // best-fit scan over every bucket is measurable per dispatch.
        if let Some(subs) = buckets.get(&bucket_size(size)) {
            if let Some(s) = subs.iter().find(|s| eligible(s)) {
                return Some(Arc::clone(s));
            }
        }
        let mut best: Option<(usize, Arc<Buffer>)> = None;
        for (&b, subs) in buckets.iter() {
            if b < size || best.as_ref().is_some_and(|(bb, _)| b >= *bb) {
                continue;
            }
            if let Some(s) = subs.iter().find(|s| eligible(s)) {
                best = Some((b, Arc::clone(s)));
            }
        }
        best.map(|(_, b)| {
            // Our arrays are eager (no autograd graph retaining intermediates),
            // so a pooled buffer can be handed back while a kernel still reads a
            // region its producer never wrote. The reference relied on the
            // graph for this; zero on reuse instead. `LISA_NO_ZERO_REUSE=1`
            // disables it for diagnostics only.
            if !env_flag("LISA_NO_ZERO_REUSE") {
                let t0 = std::time::Instant::now();
                unsafe { std::ptr::write_bytes(b.contents(), 0, b.length()) };
                ZERO_NS.fetch_add(t0.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
            }
            b
        })
    }

    fn put(&self, buf: Arc<Buffer>) {
        self.bytes.fetch_add(buf.length(), Ordering::Relaxed);
        let b = bucket_size(buf.length());
        self.buckets
            .write()
            .unwrap()
            .entry(b)
            .or_default()
            .push(buf);
    }

    /// Drop every pooled buffer whose only owner is the pool; returns bytes freed.
    pub fn sweep(&self) -> usize {
        let mut freed = 0usize;
        let mut buckets = self.buckets.write().unwrap();
        for subs in buckets.values_mut() {
            subs.retain(|s| {
                if Arc::strong_count(s) == 1 {
                    freed += s.length();
                    false
                } else {
                    true
                }
            });
        }
        self.bytes.fetch_sub(freed, Ordering::Relaxed);
        freed
    }

    /// Sweep once the pool holds more than its cap. Cheap (no sync) otherwise.
    pub fn sweep_if_over(&self) -> bool {
        if env_flag("LISA_NO_SWEEP") {
            return false;
        }
        let limit = self.limit.load(Ordering::Relaxed);
        if limit != 0 && self.bytes() > limit {
            self.sweep();
            return true;
        }
        false
    }
}

impl Default for BufferPool {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────── fences ───────────────────────────────

/// Cross-encoder synchronization for `HazardTrackingModeUntracked` buffers.
/// `memoryBarrierWithScope` only orders work inside one encoder, so a buffer
/// written by one encoder and read by the next needs an explicit fence.
pub struct Fence {
    raw: Retained<ProtocolObject<dyn MTLFence>>,
}

unsafe impl Send for Fence {}
unsafe impl Sync for Fence {}

impl Fence {
    fn new(device: &ProtocolObject<dyn MTLDevice>) -> Result<Self> {
        let raw = device.newFence().ok_or_else(|| err("newFence"))?;
        Ok(Self { raw })
    }

    pub fn raw(&self) -> &ProtocolObject<dyn MTLFence> {
        &self.raw
    }
}

// ──────────────────────────── command encoder ────────────────────────────

/// Per-encoder RAII state for read-after-write / write-after-write /
/// write-after-read detection between dispatches of the same session.
#[derive(Default)]
struct EncoderState {
    prev_outputs: HashSet<usize>,
    next_outputs: HashSet<usize>,
    prev_inputs: HashSet<usize>,
    next_inputs: HashSet<usize>,
    needs_barrier: bool,
    /// Every buffer this session wrote, registered globally on `end`.
    all_outputs: HashSet<usize>,
}

unsafe impl Send for ComputeEncoder {}
unsafe impl Sync for ComputeEncoder {}

/// A compute encoder whose session spans several dispatches (lisa builds a
/// whole chunk / round before committing), with in-session barriers and a
/// session fence for the next encoder.
#[derive(Clone)]
pub struct ComputeEncoder {
    raw: Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>,
    fence: Arc<Fence>,
    state: Arc<Mutex<EncoderState>>,
    /// The command buffer this encoder encodes into (kept here too, for
    /// the completed-handler that cleans the cross-encoder fence map).
    command_buffer: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
    /// Id of `command_buffer`, recorded on every bound buffer so the pool can
    /// reuse it once this command buffer completes.
    cb_id: u64,
    /// Diagnostic: `LISA_NO_TRACKING` (read once at encoder creation).
    no_tracking: bool,
}

impl ComputeEncoder {
    fn new(
        raw: Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>,
        fence: Arc<Fence>,
        command_buffer: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
        cb_id: u64,
    ) -> Self {
        Self {
            raw,
            fence,
            state: Arc::new(Mutex::new(EncoderState::default())),
            command_buffer,
            cb_id,
            no_tracking: env_flag("LISA_NO_TRACKING"),
        }
    }

    pub fn set_pipeline(&self, pipeline: &ComputePipeline) {
        self.raw.setComputePipelineState(pipeline.raw());
    }

    /// Bind a read-only input. Registers it as read so a later in-session write
    /// (or a read after an earlier write) inserts a barrier.
    pub fn set_input(&self, index: usize, buffer: Option<&Arc<Buffer>>, offset: usize) {
        if let Some(b) = buffer {
            b.mark_used_cb(self.cb_id);
            let b: &Buffer = b;
            let id = b.id();
            if !self.no_tracking {
                let mut s = self.state.lock().unwrap();
                if s.prev_outputs.contains(&id) {
                    s.needs_barrier = true;
                }
                s.next_inputs.insert(id);
            }
        }
        unsafe {
            self.raw.setBuffer_offset_atIndex(
                buffer.map(|b| AsRef::<ProtocolObject<dyn MTLBuffer>>::as_ref(&**b)),
                offset,
                index,
            )
        }
    }

    /// Bind a writable output. Passing the same buffer the host later reads (or
    /// a buffer the same session already touched) forces a barrier.
    pub fn set_output(&self, index: usize, buffer: Option<&Arc<Buffer>>, offset: usize) {
        let buf_arc = buffer.expect("output buffer");
        buf_arc.mark_used_cb(self.cb_id);
        let buffer: &Buffer = buf_arc;
        let id = buffer.id();
        if !self.no_tracking {
            let mut s = self.state.lock().unwrap();
            if s.prev_outputs.contains(&id) || s.prev_inputs.contains(&id) {
                s.needs_barrier = true;
            }
            s.next_outputs.insert(id);
            s.all_outputs.insert(id);
        }
        unsafe {
            self.raw
                .setBuffer_offset_atIndex(Some(buffer.as_ref()), offset, index)
        }
    }

    /// A raw byte blob (e.g. an array of `size_t` dims/strides), copied into
    /// the command buffer at encode time.
    pub fn set_bytes_directly(&self, index: usize, length: usize, bytes: *const c_void) {
        let ptr = NonNull::new(bytes as *mut c_void).unwrap();
        unsafe {
            self.raw.setBytes_length_atIndex(ptr, length, index)
        }
    }

    /// A 0-d / scalar argument.
    pub fn set_bytes<T>(&self, index: usize, data: &T) {
        let ptr = NonNull::new(data as *const T as *mut c_void).unwrap();
        unsafe {
            self.raw
                .setBytes_length_atIndex(ptr, std::mem::size_of::<T>(), index)
        }
    }

    pub fn dispatch_threads(&self, grid: (usize, usize, usize), group: (usize, usize, usize)) {
        DISPATCH_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.barrier_if_needed();
        self.raw.dispatchThreads_threadsPerThreadgroup(
            MTLSize {
                width: grid.0,
                height: grid.1,
                depth: grid.2,
            },
            MTLSize {
                width: group.0,
                height: group.1,
                depth: group.2,
            },
        );
    }

    pub fn dispatch_groups(&self, groups: (usize, usize, usize), group: (usize, usize, usize)) {
        DISPATCH_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.barrier_if_needed();
        self.raw.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: groups.0,
                height: groups.1,
                depth: groups.2,
            },
            MTLSize {
                width: group.0,
                height: group.1,
                depth: group.2,
            },
        );
    }

    /// Tuple-taking dispatch (avoids exporting the objc2 MTLSize type).
    pub fn dispatch_threads_3d(
        &self,
        grid: (usize, usize, usize),
        group: (usize, usize, usize),
    ) {
        self.dispatch_threads(grid, group);
    }

    /// Tuple-taking threadgroup-count dispatch (
    /// `dispatch_thread_groups`).
    pub fn dispatch_groups_3d(
        &self,
        groups: (usize, usize, usize),
        group: (usize, usize, usize),
    ) {
        self.dispatch_groups(groups, group);
    }

    /// `MTLSize`-taking dispatch (shape), for `mlx_rt`'s sites.
    pub fn dispatch_threads_size(&self, grid: MTLSize, group: MTLSize) {
        DISPATCH_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.barrier_if_needed();
        self.raw
            .dispatchThreads_threadsPerThreadgroup(grid, group);
    }

    /// `MTLSize`-taking threadgroup dispatch.
    pub fn dispatch_groups_size(&self, groups: MTLSize, group: MTLSize) {
        DISPATCH_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.barrier_if_needed();
        self.raw
            .dispatchThreadgroups_threadsPerThreadgroup(groups, group);
    }

    /// Fold this dispatch's accesses into the previous set, inserting a buffer
    /// barrier when one of them depends on an earlier dispatch.
    fn barrier_if_needed(&self) {
        let mut s = self.state.lock().unwrap();
        if s.needs_barrier {
            if env_flag("LISA_BARRIER_TRACE") {
                static N: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
                eprintln!("[barrier] #{}", N.fetch_add(1, std::sync::atomic::Ordering::Relaxed));
            }
            self.raw.memoryBarrierWithScope(MTLBarrierScope::Buffers);
            s.needs_barrier = false;
            s.prev_outputs = std::mem::take(&mut s.next_outputs);
            s.prev_inputs = std::mem::take(&mut s.next_inputs);
        } else {
            let outs = std::mem::take(&mut s.next_outputs);
            s.prev_outputs.extend(outs);
            let ins = std::mem::take(&mut s.next_inputs);
            s.prev_inputs.extend(ins);
        }
    }

    fn end(&self, prev_outputs: &Arc<Mutex<HashMap<usize, Arc<Fence>>>>) {
        let outs: Vec<usize> = {
            let s = self.state.lock().unwrap();
            s.all_outputs.iter().copied().collect()
        };
        if !outs.is_empty() {
            {
                let mut map = prev_outputs.lock().unwrap();
                for id in &outs {
                    map.insert(*id, Arc::clone(&self.fence));
                }
            }
            // The `end_encoding` cleanup: once this command buffer
            // completes, drop its entries so the map only holds in-flight work.
            // Without it the map grows for the whole run and every new session
            // encodes a `waitForFence` per stale entry (decode CPU cost).
            let map_for_cleanup = Arc::clone(prev_outputs);
            let fence_for_cleanup = Arc::clone(&self.fence);
            let all_outputs = Arc::new(outs);
            let block = RcBlock::new(move |_cb: NonNull<ProtocolObject<dyn MTLCommandBuffer>>| {
                let mut map = map_for_cleanup.lock().unwrap();
                for &buf in all_outputs.iter() {
                    if let Some(f) = map.get(&buf) {
                        if Arc::ptr_eq(f, &fence_for_cleanup) {
                            map.remove(&buf);
                        }
                    }
                }
            });
            unsafe {
                self.command_buffer
                    .addCompletedHandler(RcBlock::as_ptr(&block));
            }
        }
        self.raw.updateFence(self.fence.raw());
        self.raw.endEncoding();
    }
}

/// A command buffer plus its open encoder; commit/rotate at a dispatch budget.
struct EntryState {
    current: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
    current_id: u64,
    in_flight: Vec<(u64, Retained<ProtocolObject<dyn MTLCommandBuffer>>)>,
    encoder: Option<ComputeEncoder>,
}

unsafe impl Send for EntryState {}

/// Contiguous completion watermark over command-buffer ids. Command buffers are
/// committed in id order; completion may be reported out of order, so a set of
/// completed ids is held until the gap below the watermark fills. Only ids at or
/// below the watermark are treated as done.
#[derive(Default)]
struct CompletionState {
    watermark: u64,
    pending: HashSet<u64>,
}

impl CompletionState {
    fn note(&mut self, id: u64) {
        if id <= self.watermark {
            return;
        }
        self.pending.insert(id);
        while self.pending.remove(&(self.watermark + 1)) {
            self.watermark += 1;
        }
    }
}

/// Batching compute dispatches into command buffers, with the cross-encoder
/// fence map. Mirrors the shape of `the runtime's `Commands``.
pub struct Commands {
    state: Mutex<EntryState>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    compute_count: AtomicUsize,
    per_buffer: usize,
    prev_outputs: Arc<Mutex<HashMap<usize, Arc<Fence>>>>,
    /// Monotonic command-buffer ids and the highest one known complete.
    next_id: AtomicU64,
    completed_id: AtomicU64,
    /// Contiguous completion watermark over command-buffer ids, advanced by the
    /// per-command-buffer completion handlers. `take` reuses a pooled buffer
    /// only once the command buffer that last bound it is at or below this.
    completed: Arc<Mutex<CompletionState>>,
    /// Incremented by every `flush_and_wait`; pooled buffers may only be reused
    /// across an epoch boundary.
    epoch: AtomicU64,
    /// Diagnostic: counts `encoder()` opens for `LISA_SYNC_EVERY_OP`.
    op_count: AtomicUsize,
}

unsafe impl Send for Commands {}
unsafe impl Sync for Commands {}

pub struct EncoderGuard<'a> {
    guard: std::sync::MutexGuard<'a, EntryState>,
}

impl EncoderGuard<'_> {
    pub fn encoder(&self) -> &ComputeEncoder {
        self.guard.encoder.as_ref().expect("encoder open")
    }
}

impl Commands {
    pub fn new(
        queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
        device: Retained<ProtocolObject<dyn MTLDevice>>,
        per_buffer: usize,
    ) -> Result<Self> {
        let current = queue.commandBuffer().ok_or_else(|| err("commandBuffer"))?;
        Ok(Self {
            state: Mutex::new(EntryState {
                current,
                current_id: 1,
                in_flight: Vec::new(),
                encoder: None,
            }),
            queue,
            device,
            compute_count: AtomicUsize::new(0),
            per_buffer: per_buffer.max(1),
            prev_outputs: Arc::new(Mutex::new(HashMap::new())),
            next_id: AtomicU64::new(2),
            completed_id: AtomicU64::new(0),
            completed: Arc::new(Mutex::new(CompletionState::default())),
            epoch: AtomicU64::new(1),
            op_count: AtomicUsize::new(0),
        })
    }

    /// The highest command-buffer id known to have completed.
    pub fn completed_id(&self) -> u64 {
        self.completed_id.load(Ordering::Acquire)
    }

    /// The contiguous completion watermark; a pooled buffer is reusable once
    /// every command buffer that bound it is at or below this.
    pub fn completed_watermark(&self) -> u64 {
        self.completed.lock().unwrap().watermark
    }

    /// Attach a completion handler to `cb` that advances the watermark to `id`.
    /// Called for every command buffer before it is committed.
    fn arm_completion(&self, cb: &ProtocolObject<dyn MTLCommandBuffer>, id: u64) {
        let tracker = Arc::clone(&self.completed);
        let block = RcBlock::new(move |_cb: NonNull<ProtocolObject<dyn MTLCommandBuffer>>| {
            tracker.lock().unwrap().note(id);
        });
        unsafe {
            cb.addCompletedHandler(RcBlock::as_ptr(&block));
        }
    }

    /// The current flush epoch.
    pub fn epoch(&self) -> u64 {
        self.epoch.load(Ordering::Acquire)
    }

    /// Open (or reuse) the current session's encoder.
    pub fn encoder(&self) -> Result<EncoderGuard<'_>> {
        if env_flag("LISA_SYNC_EVERY_OP") {
            if self.op_count.fetch_add(1, Ordering::Relaxed) % 1 == 0 {
                let _ = self.flush_and_wait();
            }
        }
        if env_flag("LISA_DISPATCH_TRACE") {
            ENCODER_COUNT.fetch_add(1, Ordering::Relaxed);
        }
        let mut st = self.state.lock().unwrap();
        if self.compute_count.fetch_add(1, Ordering::Relaxed) >= self.per_buffer {
            self.rotate(&mut st)?;
            // `rotate` reset the counter; this open still belongs to the new
            // command buffer, so count it (otherwise a flush can miss it).
            self.compute_count.store(1, Ordering::Release);
        }
        if st.encoder.is_none() {
            let fence = Arc::new(Fence::new(&self.device)?);
            let serial = env_flag("LISA_SERIAL_ENCODER");
            let raw = if serial {
                st.current.computeCommandEncoder()
            } else {
                st.current.computeCommandEncoderWithDispatchType(MTLDispatchType::Concurrent)
            }
            .ok_or_else(|| err("computeCommandEncoder"))?;
            // Every buffer written by a previous session must be visible before
            // the first dispatch here.
            {
                let prev = self.prev_outputs.lock().unwrap();
                let mut seen = HashSet::new();
                for f in prev.values() {
                    if seen.insert(Arc::as_ptr(f) as usize) {
                        raw.waitForFence(f.raw());
                    }
                }
            }
            let cb = st.current.clone();
            st.encoder = Some(ComputeEncoder::new(raw, fence, cb, st.current_id));
        }
        Ok(EncoderGuard { guard: st })
    }

    fn rotate(&self, st: &mut EntryState) -> Result<()> {
            if let Some(enc) = st.encoder.take() {
                enc.end(&self.prev_outputs);
            }
        match st.current.status() {
            MTLCommandBufferStatus::NotEnqueued | MTLCommandBufferStatus::Enqueued => {
                // Arm before commit so a completion that races the caller is not
                // missed; the watermark only moves on this id's completion.
                self.arm_completion(&st.current, st.current_id);
                st.current.commit()
            }
            _ => {}
        }
        let next = self.queue.commandBuffer().ok_or_else(|| err("commandBuffer"))?;
        let old = std::mem::replace(&mut st.current, next);
        let old_id = st.current_id;
        st.current_id = self.next_id.fetch_add(1, Ordering::AcqRel);
        if env_flag("LISA_SYNC_ROTATE") {
            let _ = wait(&old);
        }
        st.in_flight.push((old_id, old));
        self.compute_count.store(0, Ordering::Release);
        Ok(())
    }

    /// Commit and wait for everything queued so far; drops the fence map.
    pub fn flush_and_wait(&self) -> Result<()> {
        let to_wait = {
            let mut st = self.state.lock().unwrap();
            if st.encoder.is_some() || self.compute_count.load(Ordering::Acquire) > 0 {
                self.rotate(&mut st)?;
            }
            std::mem::take(&mut st.in_flight)
        };
        if let Some((last_id, last)) = to_wait.last() {
            let t0 = std::time::Instant::now();
            wait(last)?;
            if env_flag("LISA_WAIT_TRACE") {
                WAIT_NS.fetch_add(
                    t0.elapsed().as_nanos() as u64,
                    std::sync::atomic::Ordering::Relaxed,
                );
            }
            self.completed_id.fetch_max(*last_id, Ordering::AcqRel);
        }
        if env_flag("LISA_CHECK_REUSE") {
            for (id, cb) in &to_wait {
                if cb.status() == MTLCommandBufferStatus::Completed {
                    self.completed_id.fetch_max(*id, Ordering::AcqRel);
                } else {
                    eprintln!(
                        "[flush] cb {id} not completed after wait: {:?}",
                        cb.status()
                    );
                }
            }
        }
        for (_, cb) in &to_wait {
            if cb.status() == MTLCommandBufferStatus::Error {
                let msg = cb
                    .error()
                    .map(|e| e.localizedDescription().to_string())
                    .unwrap_or_default();
                return Err(err(format!("command buffer error: {msg}")));
            }
        }
        self.prev_outputs.lock().unwrap().clear();
        self.epoch.fetch_add(1, Ordering::AcqRel);
        Ok(())
    }
}

impl Drop for Commands {
    fn drop(&mut self) {
        // End the open encoder and drain in-flight buffers; a command encoder
        // released without `endEncoding` trips a Metal assertion.
        let _ = self.flush_and_wait();
    }
}

fn wait(cb: &ProtocolObject<dyn MTLCommandBuffer>) -> Result<()> {
    match cb.status() {
        MTLCommandBufferStatus::NotEnqueued | MTLCommandBufferStatus::Enqueued => {
            cb.commit();
            cb.waitUntilCompleted();
        }
        MTLCommandBufferStatus::Committed | MTLCommandBufferStatus::Scheduled => {
            cb.waitUntilCompleted();
        }
        MTLCommandBufferStatus::Completed => {}
        MTLCommandBufferStatus::Error => {
            let msg = cb
                .error()
                .map(|e| e.localizedDescription().to_string())
                .unwrap_or_default();
            return Err(err(format!("command buffer error: {msg}")));
        }
        _ => {}
    }
    Ok(())
}

// ───────────────────────────── pipelines ─────────────────────────────

#[derive(Clone)]
pub struct ComputePipeline {
    raw: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
}

unsafe impl Send for ComputePipeline {}
unsafe impl Sync for ComputePipeline {}

impl ComputePipeline {
    /// Wrap an already-built pipeline (the AOT metallib path).
    pub fn from_raw(raw: Retained<ProtocolObject<dyn MTLComputePipelineState>>) -> Self {
        Self { raw }
    }

    pub fn raw(&self) -> &ProtocolObject<dyn MTLComputePipelineState> {
        &self.raw
    }

    pub fn max_total_threads_per_threadgroup(&self) -> usize {
        self.raw.maxTotalThreadsPerThreadgroup()
    }
}

// ───────────────────────────── runtime ─────────────────────────────

unsafe impl Send for MetalRuntime {}
unsafe impl Sync for MetalRuntime {}

/// Owns the device, queue, pool, pipeline cache and command batching.
pub struct MetalRuntime {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    pub pool: Arc<BufferPool>,
    pub commands: Arc<Commands>,
    pipelines: Mutex<HashMap<String, ComputePipeline>>,
}

/// Which compile options a kernel needs. The two paths in the tree disagree,
/// and the disagreement is load-bearing for bit-exactness:
///
/// - `Safe` matches MLX's `device.cpp::set_compile_options` (and the runtime's
///   `compile_builtin`): `MathModeSafe` + `Precise` + an explicit language
///   version. Leaving `Precise` off shifted `metal::exp` by 1 ulp on the MLX
///   unary kernels; `fastMathEnabled(false)` with a default language version
///   moved `silu_head` by ~6e-5 on large inputs.
/// - `Fast` matches the kernels crate' `get_compile_options`, which compiles
///   *own* kernels (binary, reduce, indexing, ...) with
///   `MathModeFast` + `Fast` and the default language version. Without it,
///   `bdiv` differs by 1 ulp (fast math turns division into a reciprocal
///   multiply).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Math {
    /// mlx_rt's JIT `MetalKernel` path: Safe + an explicit language version,
    /// but **no** `Precise` floating-point functions (its own `compile_options`).
    Jit,
    /// MLX's `device.cpp::set_compile_options`: Safe + Precise + an explicit
    /// language version.
    Safe,
    /// `compile_builtin` path (the MLX builtin/quantized kernels):
    /// Safe + Precise, but **no** language version — setting one changes the
    /// generated code for these kernels.
    SafeNoLang,
    Fast,
}


/// A specialised Metal function constant.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ConstVal {
    Bool(bool),
    Int(i32),
}

fn compile_options(math: Math) -> Retained<MTLCompileOptions> {
    use objc2_metal::MTLLanguageVersion;
    let opts = MTLCompileOptions::new();
    match math {
        Math::SafeNoLang => {
            opts.setMathMode(MTLMathMode::Safe);
            opts.setMathFloatingPointFunctions(MTLMathFloatingPointFunctions::Precise);
        }
        Math::Jit | Math::Safe => {
            opts.setMathMode(MTLMathMode::Safe);
            if math == Math::Safe {
                opts.setMathFloatingPointFunctions(MTLMathFloatingPointFunctions::Precise);
            }
            let lang = if objc2::available!(macos = 27.0) {
                (4 << 16) + 1
            } else if objc2::available!(macos = 26.0) {
                4 << 16
            } else {
                (3 << 16) + 2
            };
            opts.setLanguageVersion(MTLLanguageVersion(lang));
        }
        Math::Fast => {
            opts.setMathMode(MTLMathMode::Fast);
            opts.setMathFloatingPointFunctions(MTLMathFloatingPointFunctions::Fast);
        }
    }
    opts
}

impl MetalRuntime {
    pub fn new(per_buffer: usize) -> Result<Self> {
        if env_flag("LISA_DISPATCH_TRACE") || env_flag("LISA_LABEL_TRACE") {
            TRACE_ON.store(true, Ordering::Relaxed);
        }

        let device: Retained<ProtocolObject<dyn MTLDevice>> =
            MTLCreateSystemDefaultDevice().ok_or_else(|| err("no Metal device"))?;
        let queue = device.newCommandQueue().ok_or_else(|| err("newCommandQueue"))?;
        let pool = Arc::new(BufferPool::new());
        let commands = Arc::new(Commands::new(queue.clone(), device.clone(), per_buffer)?);
        Ok(Self {
            device,
            queue,
            pool,
            commands,
            pipelines: Mutex::new(HashMap::new()),
        })
    }

    pub fn device(&self) -> &ProtocolObject<dyn MTLDevice> {
        &self.device
    }

    /// The raw `MTLDevice` (for callers that need it directly).
    pub fn metal_device(&self) -> &ProtocolObject<dyn MTLDevice> {
        &self.device
    }

    /// The GPU architecture name (e.g. `"applegpu_g17s"`), used to pick NAX
    /// tiles.
    pub fn architecture_name(&self) -> String {
        self.device.architecture().name().to_string()
    }

    /// Commit and wait (`Device::synchronize`).
    pub fn synchronize(&self) -> Result<()> {
        self.commands.flush_and_wait()
    }

    pub fn queue(&self) -> &ProtocolObject<dyn MTLCommandQueue> {
        &self.queue
    }

    /// Allocate (or reuse) a shared-storage buffer of at least `bytes`.
    pub fn buffer(&self, bytes: usize, label: &str) -> Result<Arc<Buffer>> {
        if TRACE_ON.load(Ordering::Relaxed) {
            ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
            if env_flag("LISA_LABEL_TRACE") {
                let mut c = LABEL_COUNTS.lock().unwrap();
                match c.iter_mut().find(|(l, _)| l == label) {
                    Some(e) => e.1 += 1,
                    None => c.push((label.to_string(), 1)),
                }
            }
        }
        self.pool.sweep_if_over();
        if let Some(b) = self.pool.take(bytes, self.commands.completed_watermark()) {
            if TRACE_ON.load(Ordering::Relaxed) {
                POOLHIT_COUNT.fetch_add(1, Ordering::Relaxed);
            }
            return Ok(b);
        }
        let raw = self
            .device
            .newBufferWithLength_options(bucket_size(bytes), MTLResourceOptions::StorageModeShared)
            .ok_or_else(|| err(format!("newBuffer({bytes})")))?;
        let buf = Arc::new(Buffer::wrap(raw));
        buf.set_label(label);
        // Metal buffer contents are undefined at allocation; a kernel that reads
        // a region its producer never wrote would see recycled garbage.
        // (Same switch as take()'s zeroing: `LISA_NO_ZERO_REUSE=1` disables.)
        if !env_flag("LISA_NO_ZERO_REUSE") {
            unsafe { std::ptr::write_bytes(buf.contents(), 0, buf.length()) };
        }
        self.pool.put(Arc::clone(&buf));
        Ok(buf)
    }

    /// Compile `source`'s kernel `name` with the MLX (`Safe`) options, cached.
    pub fn compile(&self, source: &str, name: &str) -> Result<ComputePipeline> {
        self.compile_with(source, name, Math::Safe)
    }

    /// Compile `source`'s kernel `name` with explicit math options, cached by
    /// `(name, math)`.
    pub fn compile_with(
        &self,
        source: &str,
        name: &str,
        math: Math,
    ) -> Result<ComputePipeline> {
        self.compile_full(source, name, math, &[])
    }

    /// Like [`compile_with`], specialising Metal bool function constants
    /// (`[[function_constant(n)]]`), which ternary kernel uses to pick
    /// per-operand indexers.
    pub fn compile_with_constants(
        &self,
        source: &str,
        name: &str,
        math: Math,
        consts: &[(usize, ConstVal)],
    ) -> Result<ComputePipeline> {
        self.compile_full(source, name, math, consts)
    }

    fn compile_full(
        &self,
        source: &str,
        name: &str,
        math: Math,
        consts: &[(usize, ConstVal)],
    ) -> Result<ComputePipeline> {
        let key = format!(
            "{name}|{math:?}|{}",
            consts
                .iter()
                .map(|(i, v)| format!("{i}{v:?}"))
                .collect::<String>()
        );
        if let Some(p) = self.pipelines.lock().unwrap().get(&key) {
            return Ok(p.clone());
        }
        let opts = compile_options(math);
        let lib = self
            .device
            .newLibraryWithSource_options_error(&ns(source), Some(&opts))
            .map_err(|e| err(format!("library `{name}`: {}", e.localizedDescription())))?;
        let f: Retained<ProtocolObject<dyn MTLFunction>> = if consts.is_empty() {
            lib.newFunctionWithName(&ns(name))
                .ok_or_else(|| err(format!("function `{name}`")))?
        } else {
            let cvs = MTLFunctionConstantValues::new();
            for (i, v) in consts {
                let (ptr, ty) = match v {
                    ConstVal::Bool(b) => (NonNull::from(b).cast(), MTLDataType::Bool),
                    ConstVal::Int(x) => (NonNull::from(x).cast(), MTLDataType::Int),
                };
                unsafe { cvs.setConstantValue_type_atIndex(ptr, ty, *i) };
            }
            lib.newFunctionWithName_constantValues_error(&ns(name), &cvs)
                .map_err(|e| err(format!("function `{name}`: {}", e.localizedDescription())))?
        };
        let state = self
            .device
            .newComputePipelineStateWithFunction_error(&f)
            .map_err(|e| err(format!("pipeline `{name}`: {}", e.localizedDescription())))?;
        let p = ComputePipeline { raw: state };
        self.pipelines.lock().unwrap().insert(key, p.clone());
        Ok(p)
    }
}

// ───────────────────────────── tests ─────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const KERNEL: &str = r#"
#include <metal_stdlib>
using namespace metal;
[[kernel]] void double_it(
    const device float* x [[buffer(0)]],
    device float* out [[buffer(1)]],
    uint i [[thread_position_in_grid]])
{
    out[i] = x[i] * 2.0f;
}
"#;

    #[test]
    fn dispatch_and_read_back() {
        let rt = MetalRuntime::new(4).unwrap();
        let x = rt.buffer(64, "x").unwrap();
        let out = rt.buffer(64, "out").unwrap();
        unsafe {
            let p = x.contents() as *mut f32;
            for i in 0..16 {
                *p.add(i) = i as f32;
            }
        }
        let pipe = rt.compile(KERNEL, "double_it").unwrap();
        {
            let guard = rt.commands.encoder().unwrap();
            let enc = guard.encoder();
            enc.set_pipeline(&pipe);
            enc.set_input(0, Some(&x), 0);
            enc.set_output(1, Some(&out), 0);
            enc.dispatch_threads((16, 1, 1), (16, 1, 1));
        }
        rt.commands.flush_and_wait().unwrap();
        let got: Vec<f32> = unsafe {
            let p = out.contents() as *const f32;
            (0..16).map(|i| *p.add(i)).collect()
        };
        assert_eq!(got, (0..16).map(|i| (i * 2) as f32).collect::<Vec<_>>());
    }

    #[test]
    fn pool_reuses_then_caps() {
        let rt = MetalRuntime::new(4).unwrap();
        rt.pool.set_limit(1 << 20);
        // A released buffer is reused, not re-allocated.
        let a = rt.buffer(4096, "a").unwrap();
        drop(a);
        let before = rt.pool.bytes();
        let b = rt.buffer(4096, "b").unwrap();
        assert_eq!(rt.pool.bytes(), before, "released buffer should be reused");
        drop(b);

        // Past the cap, `sweep_if_over` releases what nobody holds.
        let mut held = Vec::new();
        for _ in 0..8 {
            held.push(rt.buffer(512 * 1024, "big").unwrap());
        }
        let peak = rt.pool.bytes();
        held.truncate(2);
        assert!(rt.pool.bytes() > (1 << 20), "pool over cap: {peak}");
        assert!(rt.pool.sweep_if_over(), "sweep should fire");
        assert!(rt.pool.bytes() <= (1 << 20), "pool bounded after sweep");
    }
}