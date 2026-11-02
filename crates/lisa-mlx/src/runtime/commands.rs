use block2::RcBlock;
use std::collections::{HashMap, HashSet};
use std::ffi::c_void;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandEncoder, MTLCommandQueue,
    MTLComputeCommandEncoder, MTLDevice, MTLDispatchType, MTLResource, MTLResourceUsage, MTLSize,
};

use crate::error::Result;
use crate::trace;

use super::buffer::Buffer;
use super::env::{DISPATCH_COUNT, ENCODER_COUNT, WAIT_NS, add_gpu_ns_split, err};
use super::fence::Fence;
use super::pipeline::ComputePipeline;
use super::residency::ResidencySet;

// ──────────────────────────── command encoder ────────────────────────────

/// Per-encoder state: the buffers this session wrote, registered globally on
/// `end` (cross-encoder fence map). The former per-dispatch hazard sets
/// (prev/next inputs/outputs) are gone: the encoder uses serial dispatch, and
/// command-buffer submission establishes ordering for later consumers.
#[derive(Default)]
struct EncoderState {
    /// Every buffer this session wrote, registered globally on `end`.
    all_outputs: Vec<usize>,
}

unsafe impl Send for ComputeEncoder {}
unsafe impl Sync for ComputeEncoder {}

/// A compute encoder whose session spans several dispatches (lisa builds a
/// whole chunk / round before committing), with a session fence for the next
/// encoder.
#[derive(Clone)]
pub struct ComputeEncoder {
    raw: Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>,
    command_buffer: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
    fence: Arc<Fence>,
    state: Arc<Mutex<EncoderState>>,
    /// Id of `command_buffer`, recorded on every bound buffer so the pool can
    /// reuse it once this command buffer completes.
    cb_id: u64,
    /// Host time from the first encoder creation until its command buffer is
    /// submitted. This separates CPU encoding gaps from GPU execution time.
    encode_started: std::time::Instant,
    /// Resources reached through GPU virtual addresses are invisible to Metal's
    /// ordinary buffer binding tracker; retain them through command completion.
    cb_keepalive: Arc<Mutex<HashMap<u64, HashMap<usize, Arc<Buffer>>>>>,
    /// Interned kernel label of the last `set_pipeline` (u32::MAX = none), so
    /// each dispatch is attributed. Atomics only: no lock, no String clone on
    /// the hot path.
    current_label: Arc<AtomicU32>,
    /// Dispatch count per interned kernel label encoded into this encoder's
    /// session. Captured when tracing, GPU probing, or command-buffer recording
    /// is enabled.
    kernel_counts: Arc<Mutex<Vec<(u32, u64)>>>,
}

impl ComputeEncoder {
    fn new(
        raw: Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>,
        command_buffer: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
        fence: Arc<Fence>,
        cb_id: u64,
        cb_keepalive: Arc<Mutex<HashMap<u64, HashMap<usize, Arc<Buffer>>>>>,
    ) -> Self {
        Self {
            raw,
            command_buffer,
            fence,
            state: Arc::new(Mutex::new(EncoderState::default())),
            cb_id,
            encode_started: std::time::Instant::now(),
            cb_keepalive,
            current_label: Arc::new(AtomicU32::new(u32::MAX)),
            kernel_counts: Arc::new(Mutex::new(Vec::new())),
        }
    }

    pub fn set_pipeline(&self, pipeline: &ComputePipeline) {
        self.current_label
            .store(pipeline.label_id(), Ordering::Relaxed);
        self.raw.setComputePipelineState(pipeline.raw());
    }

    /// Attach an indirect-resource set to the command buffer that owns this
    /// encoder. Metal makes those allocations resident for its execution.
    pub fn use_residency_set(&self, set: &ResidencySet) {
        self.command_buffer.useResidencySet(set.raw());
    }

    /// Bind dynamic threadgroup scratch declared with `[[threadgroup(0)]]`.
    pub fn set_threadgroup_memory_length(&self, index: usize, length: usize) {
        unsafe {
            self.raw.setThreadgroupMemoryLength_atIndex(length, index);
        }
    }

    /// Attribute this dispatch to the current kernel label. When diagnostics
    /// are disabled, return before the label lookup and lock.
    fn note_dispatch(&self) {
        if !trace::enabled() && !super::env::kernel_label_capture_enabled() {
            return;
        }
        let label = self.current_label.load(Ordering::Relaxed);
        if label != u32::MAX {
            let mut counts = self.kernel_counts.lock().unwrap();
            match counts.iter_mut().find(|(l, _)| *l == label) {
                Some(entry) => entry.1 += 1,
                None => counts.push((label, 1)),
            }
        }
    }

    /// Bind a read-only input. (Hazard bookkeeping removed: Serial dispatch
    /// type already orders memory across dispatches — see `EncoderState`.)
    pub fn set_input(&self, index: usize, buffer: Option<&Arc<Buffer>>, offset: usize) {
        if let Some(b) = buffer {
            b.mark_used_cb(self.cb_id);
            let b: &Buffer = b;
            unsafe {
                self.raw
                    .setBuffer_offset_atIndex(Some(b.as_ref()), offset, index)
            }
        } else {
            unsafe {
                self.raw.setBuffer_offset_atIndex(None, offset, index);
            }
        }
    }

    /// Retain a buffer referenced indirectly through a GPU address until the
    /// command buffer that contains the dispatch has completed, and declare
    /// its read residency to Metal before the indirect dispatch.
    pub fn retain_until_completion(&self, buffer: &Arc<Buffer>) {
        self.retain_until_completion_with_usage(buffer, true);
    }

    /// Retain an indirectly-addressed buffer for the command buffer when it is
    /// already covered by a queue-wide residency set.
    pub fn retain_resident_until_completion(&self, buffer: &Arc<Buffer>) {
        self.retain_until_completion_with_usage(buffer, false);
    }

    fn retain_until_completion_with_usage(&self, buffer: &Arc<Buffer>, declare_usage: bool) {
        let first_use = {
            let mut keepalive = self.cb_keepalive.lock().unwrap();
            let resources = keepalive.entry(self.cb_id).or_default();
            if resources.contains_key(&buffer.id()) {
                false
            } else {
                resources.insert(buffer.id(), Arc::clone(buffer));
                true
            }
        };
        buffer.mark_used_cb(self.cb_id);
        if first_use && declare_usage {
            let resource: &ProtocolObject<dyn MTLResource> = buffer.mtlbuffer().as_ref();
            self.raw.useResource_usage(resource, MTLResourceUsage::Read);
        }
    }

    /// Bind a writable output.
    pub fn set_output(&self, index: usize, buffer: Option<&Arc<Buffer>>, offset: usize) {
        let buf_arc = buffer.expect("output buffer");
        buf_arc.mark_used_cb(self.cb_id);
        let buffer: &Buffer = buf_arc;
        {
            let mut s = self.state.lock().unwrap();
            s.all_outputs.push(buffer.id());
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
        unsafe { self.raw.setBytes_length_atIndex(ptr, length, index) }
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
        self.note_dispatch();
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
        self.note_dispatch();
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
    pub fn dispatch_threads_3d(&self, grid: (usize, usize, usize), group: (usize, usize, usize)) {
        self.dispatch_threads(grid, group);
    }

    /// Tuple-taking threadgroup-count dispatch (
    /// `dispatch_thread_groups`).
    pub fn dispatch_groups_3d(&self, groups: (usize, usize, usize), group: (usize, usize, usize)) {
        self.dispatch_groups(groups, group);
    }

    /// `MTLSize`-taking dispatch (shape), for `mlx_rt`'s sites.
    pub fn dispatch_threads_size(&self, grid: MTLSize, group: MTLSize) {
        DISPATCH_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.note_dispatch();
        self.raw.dispatchThreads_threadsPerThreadgroup(grid, group);
    }

    /// `MTLSize`-taking threadgroup dispatch.
    pub fn dispatch_groups_size(&self, groups: MTLSize, group: MTLSize) {
        DISPATCH_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.note_dispatch();
        self.raw
            .dispatchThreadgroups_threadsPerThreadgroup(groups, group);
    }

    fn end(
        &self,
        prev_outputs: &Arc<Mutex<HashMap<usize, Arc<Fence>>>>,
        cb_labels: &Arc<Mutex<HashMap<u64, Vec<(String, u64)>>>>,
    ) -> Option<(Vec<usize>, Arc<Fence>)> {
        trace::record_elapsed(
            "mlx.commands.encoder_lifetime",
            self.cb_id,
            self.encode_started.elapsed(),
        );
        // Hand the per-label dispatch counts to the command buffer's completion
        // handler, which splits its GPU duration across them.
        {
            let counts: Vec<(String, u64)> = {
                let counts = self.kernel_counts.lock().unwrap();
                counts
                    .iter()
                    .map(|(id, c)| (super::env::label_name(*id), *c))
                    .collect()
            };
            if !counts.is_empty() {
                cb_labels
                    .lock()
                    .unwrap()
                    .entry(self.cb_id)
                    .or_default()
                    .extend(counts);
            }
        }
        let outs: Vec<usize> = {
            let mut s = self.state.lock().unwrap();
            s.all_outputs.sort_unstable();
            s.all_outputs.dedup();
            std::mem::take(&mut s.all_outputs)
        };
        if outs.is_empty() {
            self.raw.updateFence(self.fence.raw());
            self.raw.endEncoding();
            return None;
        }
        {
            let mut map = prev_outputs.lock().unwrap();
            for id in &outs {
                map.insert(*id, Arc::clone(&self.fence));
            }
        }
        self.raw.updateFence(self.fence.raw());
        self.raw.endEncoding();
        // The map-entry cleanup runs from `Commands::arm_completion`'s
        // kept-alive block (this command buffer's completion). It used to be a
        // second `addCompletedHandler` block here — dropped when `end`
        // returned. The objc2 binding passes a raw `*mut DynBlock` and does
        // NOT copy it (proven by the arm_completion fix), so the completion
        // call invoked a FREED block: benign while the heap kept the bytes
        // intact, but under parallel-test heap churn the stale captures
        // removed the WRONG fence entry (or none), the next encoder skipped
        // its `waitForFence`, and kernels/readbacks read buffers whose
        // producer command buffer was still in flight — the specs/04 §6
        // readback-race flake family. Returning the (outputs, fence) to
        // `end_encoder` here hands the cleanup to a block that provably outlives
        // the buffer.
        Some((outs, Arc::clone(&self.fence)))
    }
}

/// A command buffer plus its open encoder; commit/rotate at a dispatch budget.
struct EntryState {
    current: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
    current_id: u64,
    in_flight: Vec<(u64, Retained<ProtocolObject<dyn MTLCommandBuffer>>)>,
    encoder: Option<ComputeEncoder>,
    /// Each encoder in the current command buffer owns a distinct fence.
    /// Keep all cleanup records until that command buffer completes.
    fence_cleanups: Vec<(Vec<usize>, Arc<Fence>)>,
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
    per_buffer: AtomicUsize,
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
    /// Per-command-buffer kernel dispatch counts, consumed by the completion
    /// handler to split the buffer's GPU time across labels.
    cb_labels: Arc<Mutex<HashMap<u64, Vec<(String, u64)>>>>,
    /// Buffers indirectly accessed by GPU virtual address, keyed by command
    /// buffer and buffer identity to avoid duplicate retains.
    cb_keepalive: Arc<Mutex<HashMap<u64, HashMap<usize, Arc<Buffer>>>>>,
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
                fence_cleanups: Vec::new(),
            }),
            queue,
            device,
            compute_count: AtomicUsize::new(0),
            per_buffer: AtomicUsize::new(per_buffer.max(1)),
            prev_outputs: Arc::new(Mutex::new(HashMap::new())),
            next_id: AtomicU64::new(2),
            completed_id: AtomicU64::new(0),
            completed: Arc::new(Mutex::new(CompletionState::default())),
            epoch: AtomicU64::new(1),
            cb_labels: Arc::new(Mutex::new(HashMap::new())),
            cb_keepalive: Arc::new(Mutex::new(HashMap::new())),
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
    fn arm_completion(
        &self,
        cb: &ProtocolObject<dyn MTLCommandBuffer>,
        id: u64,
        fence_cleanups: Vec<(Vec<usize>, Arc<Fence>)>,
    ) {
        let submitted_at = std::time::Instant::now();
        let tracker = Arc::clone(&self.completed);
        // Cross-encoder fence-map cleanup for THIS command buffer's outputs:
        // on completion, drop the map entries so the map only holds in-flight
        // work (without it the map grows for the whole run and every new
        // session encodes a `waitForFence` per stale entry — decode CPU cost).
        // Lives in this block because this is the one completion block that is
        // provably kept alive (the leak below): the former dedicated cleanup
        // block in `ComputeEncoder::end` was dropped while still registered —
        // a use-after-free at completion time (specs/04 §6).
        let map_for_cleanup = Arc::clone(&self.prev_outputs);
        // The completion block itself is kept alive below. Release these
        // per-encoder vectors and fences when it fires, not at process exit.
        let fence_cleanups = Arc::new(Mutex::new(Some(fence_cleanups)));
        // Take this buffer's kernel dispatch counts; the handler splits the
        // buffer's GPU duration (GPUStartTime..GPUEndTime) across them.
        let labels = self
            .cb_labels
            .lock()
            .unwrap()
            .remove(&id)
            .unwrap_or_default();
        let keepalive = Arc::new(Mutex::new(Some(
            self.cb_keepalive
                .lock()
                .unwrap()
                .remove(&id)
                .unwrap_or_default(),
        )));
        let keepalive_for_callback = Arc::clone(&keepalive);
        let block = RcBlock::new(move |_cb: NonNull<ProtocolObject<dyn MTLCommandBuffer>>| {
            trace::record_elapsed(
                "mlx.commands.submit_to_completion",
                id,
                submitted_at.elapsed(),
            );
            drop(keepalive_for_callback.lock().unwrap().take());
            tracker.lock().unwrap().note(id);
            if let Some(fence_cleanups) = fence_cleanups.lock().unwrap().take() {
                let mut map = map_for_cleanup.lock().unwrap();
                for (outs, fence) in &fence_cleanups {
                    for &buf in outs {
                        if let Some(f) = map.get(&buf) {
                            if Arc::ptr_eq(f, fence) {
                                map.remove(&buf);
                            }
                        }
                    }
                }
            }
            if !labels.is_empty() {
                let cb: &ProtocolObject<dyn MTLCommandBuffer> = unsafe { _cb.as_ref() };
                let start = cb.GPUStartTime();
                let end = cb.GPUEndTime();
                let dur_ns = ((end - start).max(0.0) * 1e9) as u64;
                add_gpu_ns_split(&labels, dur_ns);
                super::env::add_dispatch_counts(&labels);
                if let Some(path) = super::env::cb_record_file() {
                    use std::io::Write;
                    if let Ok(mut f) = std::fs::OpenOptions::new()
                        .create(true)
                        .append(true)
                        .open(&path)
                    {
                        let esc = |s: &str| {
                            s.chars()
                                .map(|c| match c {
                                    '"' | '\\' => format!("\\{}", c),
                                    c if (c as u32) < 0x20 => format!("\\u{:04x}", c as u32),
                                    c => c.to_string(),
                                })
                                .collect::<String>()
                        };
                        let pairs = labels
                            .iter()
                            .map(|(n, c)| format!("[\"{}\",{}]", esc(n), c))
                            .collect::<Vec<_>>()
                            .join(",");
                        let json =
                            format!("{{\"id\":{},\"d\":{},\"l\":[{}]}}\n", id, dur_ns, pairs);
                        let _ = f.write_all(json.as_bytes());
                    }
                }
                // Probe mode: one dispatch per buffer, so this buffer's
                // duration is that kernel's real GPU occupancy — record the
                // absolute span too, for inter-buffer gap accounting.
                if super::env::gpu_probe_enabled() && labels.len() == 1 {
                    super::env::GPU_PROBE_RECORDS.lock().unwrap().push((
                        labels[0].0.clone(),
                        start,
                        end,
                    ));
                }
            }
        });
        // The block must outlive the buffer: objc2's `&Block` argument is a
        // non-owning pointer and the callee does NOT copy it on this binding,
        // so a dropped RcBlock meant the completion handler never fired
        // (LABEL_GPU_NS/GPU_PROBE_RECORDS stayed empty). Keep it alive for the
        // process; diagnostics only, bounded by the buffer count.
        std::mem::forget(block.clone());
        unsafe {
            cb.addCompletedHandler(RcBlock::as_ptr(&block));
        }
    }

    /// The current flush epoch.
    pub fn epoch(&self) -> u64 {
        self.epoch.load(Ordering::Acquire)
    }

    /// Override the dispatch budget per command buffer at runtime (rotate
    /// experiments: a large budget holds rotations until the step-boundary
    /// flush, removing mid-step cb commit/fence boundaries).
    pub fn set_per_buffer(&self, n: usize) {
        self.per_buffer.store(n.max(1), Ordering::Release);
    }

    pub fn per_buffer(&self) -> usize {
        self.per_buffer.load(Ordering::Acquire)
    }

    /// Make a model residency set visible to the whole queue once at setup.
    pub fn add_residency_set_to_queue(&self, set: &ResidencySet) -> bool {
        set.add_to_queue(&self.queue)
    }

    /// Register a model residency set without prewarming all of its allocations.
    pub fn add_residency_set_to_queue_without_request(&self, set: &ResidencySet) -> bool {
        set.add_to_queue_without_request(&self.queue)
    }

    /// Open (or reuse) the current session's encoder.
    pub fn encoder(&self) -> Result<EncoderGuard<'_>> {
        ENCODER_COUNT.fetch_add(1, Ordering::Relaxed);
        let mut st = self.state.lock().unwrap();
        if self.compute_count.fetch_add(1, Ordering::Relaxed)
            >= self.per_buffer.load(Ordering::Acquire)
        {
            self.rotate(&mut st)?;
            // `rotate` reset the counter; this open still belongs to the new
            // command buffer, so count it (otherwise a flush can miss it).
            self.compute_count.store(1, Ordering::Release);
        }
        if st.encoder.is_none() {
            let fence = Arc::new(Fence::new(&self.device)?);
            let raw = st
                .current
                .computeCommandEncoderWithDispatchType(MTLDispatchType::Serial)
                .ok_or_else(|| err("computeCommandEncoder"))?;
            // Every buffer written by a previous encoder must be visible before
            // this encoder's first dispatch. Skipped in GPU-probe mode because the
            // wait would contaminate the measured GPU execution window.
            if !super::env::gpu_probe_enabled() {
                let prev = self.prev_outputs.lock().unwrap();
                let mut seen = HashSet::new();
                for f in prev.values() {
                    if seen.insert(Arc::as_ptr(f) as usize) {
                        raw.waitForFence(f.raw());
                    }
                }
            }
            st.encoder = Some(ComputeEncoder::new(
                raw,
                st.current.clone(),
                fence,
                st.current_id,
                Arc::clone(&self.cb_keepalive),
            ));
        }
        Ok(EncoderGuard { guard: st })
    }

    fn end_encoder(&self, st: &mut EntryState) {
        if let Some(cleanup) = st
            .encoder
            .take()
            .and_then(|enc| enc.end(&self.prev_outputs, &self.cb_labels))
        {
            st.fence_cleanups.push(cleanup);
        }
    }

    /// End only the compute encoder, keeping the current command buffer open.
    /// The next dispatch opens another encoder in that same buffer, with the
    /// usual cross-encoder fences. This never commits, waits, advances the
    /// completion watermark, or makes its temporary buffers reusable.
    pub fn flush_encoder(&self) {
        let mut st = self.state.lock().unwrap();
        let _span = trace::span_detail("mlx.commands.flush_encoder", st.current_id);
        self.end_encoder(&mut st);
    }

    fn rotate(&self, st: &mut EntryState) -> Result<()> {
        let _span = trace::span_detail("mlx.commands.rotate", st.current_id);
        self.end_encoder(st);
        let fence_cleanups = std::mem::take(&mut st.fence_cleanups);
        match st.current.status() {
            MTLCommandBufferStatus::NotEnqueued | MTLCommandBufferStatus::Enqueued => {
                // Arm before commit so a completion that races the caller is not
                // missed; the watermark only moves on this id's completion.
                self.arm_completion(&st.current, st.current_id, fence_cleanups);
                st.current.commit()
            }
            _ => {}
        }
        let next = self
            .queue
            .commandBuffer()
            .ok_or_else(|| err("commandBuffer"))?;
        let old = std::mem::replace(&mut st.current, next);
        let old_id = st.current_id;
        st.current_id = self.next_id.fetch_add(1, Ordering::AcqRel);
        st.in_flight.push((old_id, old));
        self.compute_count.store(0, Ordering::Release);
        Ok(())
    }

    /// The id of the command buffer dispatches are currently encoding into.
    /// Captured before speculative work is enqueued, this is the barrier id
    /// for `flush_wait_through`: everything up to and including this buffer is
    /// required for a host readback, later buffers are overlap-only.
    pub fn current_cb_id(&self) -> u64 {
        self.state.lock().unwrap().current_id
    }

    /// Commit everything queued so far and wait only until the command buffer
    /// `id` has completed. Buffers committed AFTER `id` stay in flight and run
    /// under the host's next step (readback overlap). Buffer reuse stays sound:
    /// a buffer's `used_cb` is the max id that bound it, so any buffer read by
    /// a still-in-flight later buffer is not handed out until that buffer
    /// completes (`reusable_after` gates the pool on the watermark).
    ///
    /// CONCURRENCY CONTRACT (specs/04 §6): committed command buffers are NEVER
    /// removed from `in_flight` by the thread that waits them — the entry is
    /// cloned, waited outside the lock, and only pruned afterwards. The former
    /// drain-then-wait let a concurrent `eval` observe an empty `in_flight`
    /// (the other thread had "checked out" the entries) and return while the
    /// GPU was still writing the very buffer the caller was about to read.
    pub fn flush_wait_through(&self, id: u64) -> Result<()> {
        let _span = trace::span_detail("mlx.commands.flush_wait_through", id);
        let target = {
            let mut st = self.state.lock().unwrap();
            if st.encoder.is_some() || self.compute_count.load(Ordering::Acquire) > 0 {
                self.rotate(&mut st)?;
            }
            // The queue executes in commit order, so waiting for `id` implies
            // everything committed before it is done too. Clone the entry;
            // entries after it stay in flight.
            let pos = st.in_flight.iter().position(|(cid, _)| *cid >= id);
            match pos {
                Some(p) => Some(st.in_flight[p].clone()),
                None => st.in_flight.last().cloned(),
            }
        };
        if let Some((wait_id, cb)) = target {
            let t0 = std::time::Instant::now();
            let _wait_span = trace::span_detail("mlx.commands.wait_completed", wait_id);
            wait(&cb)?;
            drop(_wait_span);
            WAIT_NS.fetch_add(
                t0.elapsed().as_nanos() as u64,
                std::sync::atomic::Ordering::Relaxed,
            );
            self.completed_id.fetch_max(wait_id, Ordering::AcqRel);
            if cb.status() == MTLCommandBufferStatus::Error {
                let msg = cb
                    .error()
                    .map(|e| e.localizedDescription().to_string())
                    .unwrap_or_default();
                return Err(err(format!("command buffer error: {msg}")));
            }
            // Prune everything at or below the waited id; it and all earlier
            // command buffers have completed (FIFO queue). NOTE: prev_outputs
            // is deliberately NOT cleared here — entries for buffers still
            // bound by later in-flight command buffers must keep fencing the
            // next encoder (step-overlap readback).
            self.state
                .lock()
                .unwrap()
                .in_flight
                .retain(|(cid, _)| *cid > wait_id);
        }
        Ok(())
    }

    /// Commit everything queued so far **without waiting** (ds4/MLX parity:
    /// a kernel-input materialization enqueues the producer into the serial
    /// queue; the queue's FIFO order makes the buffer readable by any later
    /// dispatch — no `waitUntilCompleted`). Buffer reuse stays sound via the
    /// pool's `reusable_after(completed_watermark)` gate.
    pub fn flush(&self) -> Result<()> {
        let mut st = self.state.lock().unwrap();
        if st.encoder.is_some() || self.compute_count.load(Ordering::Acquire) > 0 {
            self.rotate(&mut st)?;
        }
        Ok(())
    }

    /// Commit and wait for everything queued so far; drops the fence map.
    pub fn flush_and_wait(&self) -> Result<()> {
        let _span = trace::span("mlx.commands.flush_and_wait");
        let to_wait = {
            let mut st = self.state.lock().unwrap();
            if st.encoder.is_some() || self.compute_count.load(Ordering::Acquire) > 0 {
                self.rotate(&mut st)?;
            }
            // Clone, don't drain: a concurrent eval must still see (and wait)
            // this entry — see the contract on `flush_wait_through`.
            st.in_flight.last().cloned()
        };
        if let Some((last_id, last)) = to_wait {
            let t0 = std::time::Instant::now();
            let _wait_span = trace::span_detail("mlx.commands.wait_completed", last_id);
            wait(&last)?;
            drop(_wait_span);
            WAIT_NS.fetch_add(
                t0.elapsed().as_nanos() as u64,
                std::sync::atomic::Ordering::Relaxed,
            );
            self.completed_id.fetch_max(last_id, Ordering::AcqRel);
            // Everything at or below `last_id` completed (FIFO queue); check
            // their statuses while pruning.
            {
                let mut st = self.state.lock().unwrap();
                let drained: Vec<_> = st
                    .in_flight
                    .iter()
                    .filter(|(cid, _)| *cid <= last_id)
                    .cloned()
                    .collect();
                st.in_flight.retain(|(cid, _)| *cid > last_id);
                for (_, cb) in &drained {
                    if cb.status() == MTLCommandBufferStatus::Error {
                        let msg = cb
                            .error()
                            .map(|e| e.localizedDescription().to_string())
                            .unwrap_or_default();
                        return Err(err(format!("command buffer error: {msg}")));
                    }
                }
            }
            self.prev_outputs.lock().unwrap().clear();
            self.epoch.fetch_add(1, Ordering::AcqRel);
        }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encoder_only_boundaries_preserve_command_buffer_and_clean_all_fences() -> Result<()> {
        if !crate::device::metal_available() {
            return Ok(());
        }
        let rt = crate::runtime::MetalRuntime::new(64)?;
        let commands = &rt.commands;
        let cb_id = commands.current_cb_id();
        let epoch = commands.epoch();
        for output_id in [11, 22, 33] {
            {
                let guard = commands.encoder()?;
                // Distinct output identities exercise cleanup of every closed
                // encoder, including outputs the final encoder never writes.
                guard
                    .encoder()
                    .state
                    .lock()
                    .unwrap()
                    .all_outputs
                    .push(output_id);
            }
            commands.flush_encoder();
            assert_eq!(commands.current_cb_id(), cb_id);
            assert_eq!(commands.epoch(), epoch);
            assert_eq!(commands.completed_watermark(), 0);
            let state = commands.state.lock().unwrap();
            assert!(state.encoder.is_none());
            assert!(state.in_flight.is_empty());
            assert_eq!(state.current.status(), MTLCommandBufferStatus::NotEnqueued);
        }
        assert_eq!(commands.state.lock().unwrap().fence_cleanups.len(), 3);
        assert_eq!(commands.prev_outputs.lock().unwrap().len(), 3);
        commands.flush_wait_through(cb_id)?;
        // Unlike flush_and_wait, this does not clear the fence map itself:
        // all three entries must be retired by the completion callback.
        assert!(commands.prev_outputs.lock().unwrap().is_empty());
        assert!(commands.state.lock().unwrap().fence_cleanups.is_empty());
        assert!(commands.completed_watermark() >= cb_id);
        Ok(())
    }
}
