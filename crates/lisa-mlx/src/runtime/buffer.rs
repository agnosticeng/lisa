use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, RwLock};

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLResource};

use super::env::ns;
use crate::error::{Error, Result};

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
    pub(super) fn wrap(raw: Retained<ProtocolObject<dyn MTLBuffer>>) -> Self {
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

    /// GPU virtual address for shaders that dereference Metal address tables.
    /// The owning `Buffer` must be retained until every dispatch using the
    /// address has completed.
    pub fn gpu_address(&self) -> u64 {
        self.raw.gpuAddress()
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

    /// Id of the last command buffer that bound this buffer.
    pub fn last_used_cb(&self) -> u64 {
        self.used_cb.load(Ordering::Acquire)
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
pub(super) fn bucket_size(size: usize) -> usize {
    size.max(1).div_ceil(1024) * 1024
}

#[derive(Default)]
struct ResidentAdmission {
    reservations: HashMap<usize, (usize, usize)>,
    next_id: usize,
    reserved: usize,
    budget: usize,
}

/// Accounts immutable mapped model storage for its owner's lifetime. Private
/// allocations share the remaining RAM budget while any reservation is alive.
pub struct ResidentMemoryReservation {
    pool: Arc<BufferPool>,
    commands: Arc<super::commands::Commands>,
    id: usize,
}

impl Drop for ResidentMemoryReservation {
    fn drop(&mut self) {
        // The last model owner can disappear before submitted Metal work
        // releases its buffers. Retire that work before returning its budget.
        // This wait is outside the allocator/admission lock.
        if let Err(error) = self.commands.flush_and_wait() {
            eprintln!("[memory] retaining resident reservation after GPU wait failed: {error}");
            return;
        }
        self.pool.release_model_memory(self.id);
    }
}

/// A size-bucketed buffer pool with a real, byte-accounted cap.
pub struct BufferPool {
    buckets: RwLock<HashMap<usize, Vec<Arc<Buffer>>>>,
    bytes: AtomicUsize,
    limit: AtomicUsize,
    last_sweep_bytes: AtomicUsize,
    admission: Mutex<ResidentAdmission>,
    reserved_bytes: AtomicUsize,
}

impl BufferPool {
    pub fn new() -> Self {
        Self {
            buckets: RwLock::new(HashMap::new()),
            bytes: AtomicUsize::new(0),
            limit: AtomicUsize::new(0),
            last_sweep_bytes: AtomicUsize::new(0),
            admission: Mutex::new(ResidentAdmission::default()),
            reserved_bytes: AtomicUsize::new(0),
        }
    }

    /// Bytes the pool may retain; `0` disables the cap.
    pub fn set_limit(&self, bytes: usize) {
        self.limit.store(bytes, Ordering::Relaxed);
    }

    pub fn bytes(&self) -> usize {
        self.bytes.load(Ordering::Relaxed)
    }

    /// Mapped model bytes are counted once, independently of overlapping views.
    pub fn reserved_model_bytes(&self) -> usize {
        self.reserved_bytes.load(Ordering::Acquire)
    }

    pub fn total_bytes(&self) -> usize {
        self.bytes().saturating_add(self.reserved_model_bytes())
    }

    fn release_model_memory(&self, id: usize) {
        let mut admission = self.admission.lock().unwrap();
        if let Some((bytes, _)) = admission.reservations.remove(&id) {
            admission.reserved -= bytes;
            admission.budget = admission
                .reservations
                .values()
                .map(|(_, cap)| *cap)
                .min()
                .unwrap_or(0);
            self.reserved_bytes
                .store(admission.reserved, Ordering::Release);
        }
    }

    pub(super) fn reserve_model_memory(
        self: &Arc<Self>,
        bytes: usize,
        budget: usize,
        commands: Arc<super::commands::Commands>,
    ) -> Result<ResidentMemoryReservation> {
        if bytes == 0 || budget == 0 {
            return Err(Error::msg(
                "resident model reservation and RAM budget must be nonzero",
            ));
        }
        let mut admission = self.admission.lock().unwrap();
        let reserved = admission
            .reserved
            .checked_add(bytes)
            .ok_or_else(|| Error::msg("resident model reservation overflow"))?;
        let cap = if admission.reservations.is_empty() {
            budget
        } else {
            budget.min(admission.budget)
        };
        self.ensure_room(reserved, cap, 0, commands.completed_watermark())?;
        let id = admission
            .next_id
            .checked_add(1)
            .ok_or_else(|| Error::msg("resident model reservation id overflow"))?;
        admission.next_id = id;
        admission.reservations.insert(id, (bytes, budget));
        admission.reserved = reserved;
        admission.budget = cap;
        self.reserved_bytes.store(reserved, Ordering::Release);
        Ok(ResidentMemoryReservation {
            pool: self.clone(),
            commands,
            id,
        })
    }

    fn ensure_room(
        &self,
        reserved: usize,
        budget: usize,
        bytes: usize,
        completed: u64,
    ) -> Result<()> {
        let fits = || {
            reserved
                .checked_add(self.bytes())
                .and_then(|used| used.checked_add(bytes))
                .is_some_and(|total| total <= budget)
        };
        if !fits() {
            self.sweep_completed(completed);
        }
        if !fits() {
            return Err(Error::msg(format!(
                "RAM budget exceeded: {reserved} resident model bytes + {} private bytes + {bytes} requested bytes > {budget} byte cap",
                self.bytes(),
            )));
        }
        Ok(())
    }

    /// Admit and register a new private allocation atomically with model
    /// reservations. Pool reuse bypasses this slow path. No GPU wait occurs
    /// while this lock is held: only already-completed buffers may be freed.
    pub(super) fn allocate_new(
        &self,
        bytes: usize,
        completed: u64,
        create: impl FnOnce() -> Result<Arc<Buffer>>,
    ) -> Result<Arc<Buffer>> {
        let admission = self.admission.lock().unwrap();
        if admission.reserved > 0 {
            self.ensure_room(admission.reserved, admission.budget, bytes, completed)?;
        }
        let buffer = create()?;
        self.put(buffer.clone());
        Ok(buffer)
    }

    /// The smallest free pooled buffer that can hold `size`. Recent matching
    /// buffers are checked first. A pool-owned reference keeps each allocation
    /// reusable after the caller releases its lease.
    pub fn take(&self, size: usize, completed: u64) -> Option<Arc<Buffer>> {
        self.take_inner(size, completed, false)
    }

    /// Lease storage only for a later GPU write on this pool's ordered queue.
    /// Existing GPU readers have already been encoded when the last Array
    /// owner disappears. Their commands precede the new write; CPU writers
    /// must continue to use `take` and wait for the completion watermark.
    pub(super) fn take_for_ordered_gpu_write(&self, size: usize) -> Option<Arc<Buffer>> {
        self.take_inner(size, 0, true)
    }

    fn take_inner(
        &self,
        size: usize,
        completed: u64,
        ordered_gpu_write: bool,
    ) -> Option<Arc<Buffer>> {
        let eligible = |s: &Arc<Buffer>| -> bool {
            Arc::strong_count(s) == 1 && (ordered_gpu_write || s.reusable_after(completed))
        };
        // The pool retains its owner Arc while callers hold leases. A read
        // lock is not sufficient here: two concurrent allocators could both
        // observe strong_count == 1 and clone the same pooled buffer before
        // either clone becomes visible, then bind it for unrelated writes.
        // Serialize the eligibility check and lease acquisition with sweeps.
        let buckets = self.buckets.write().unwrap();
        // Keep the pool's strong reference while a buffer is leased. The
        // caller gets a clone; `strong_count == 1` makes it available again
        // when the caller drops it. Search newest-first because the most
        // recently allocated sizes are the likeliest to recur on the next op.
        let want = bucket_size(size);
        let take_from =
            |buckets: &HashMap<usize, Vec<Arc<Buffer>>>, b: usize| -> Option<Arc<Buffer>> {
                buckets
                    .get(&b)?
                    .iter()
                    .rev()
                    // The newest buffers are the ones most likely to have
                    // just been released. Bound this lookup: model loading
                    // retains many same-sized weight buffers in the pool, and
                    // scanning all of them per allocation becomes quadratic.
                    .take(64)
                    .find(|buffer| eligible(buffer))
                    .cloned()
            };
        if let Some(s) = take_from(&buckets, want) {
            return Some(s);
        }
        // Best fit among strictly larger buckets: smallest bucket >= size.
        let mut best_b: Option<usize> = None;
        for &b in buckets.keys() {
            if b < size {
                continue;
            }
            if best_b.is_none_or(|bb| b < bb) {
                best_b = Some(b);
            }
        }
        best_b.and_then(|b| take_from(&buckets, b))
    }

    pub(super) fn put(&self, buf: Arc<Buffer>) {
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
        self.sweep_inner(None)
    }

    /// Reclaim only buffers whose last GPU user is known to have completed.
    pub fn sweep_completed(&self, completed: u64) -> usize {
        self.sweep_inner(Some(completed))
    }

    fn sweep_inner(&self, completed: Option<u64>) -> usize {
        let mut freed = 0usize;
        let mut buckets = self.buckets.write().unwrap();
        for subs in buckets.values_mut() {
            subs.retain(|s| {
                if Arc::strong_count(s) == 1
                    && completed.is_none_or(|watermark| s.reusable_after(watermark))
                {
                    freed += s.length();
                    false
                } else {
                    true
                }
            });
        }
        self.bytes.fetch_sub(freed, Ordering::Relaxed);
        self.last_sweep_bytes
            .store(self.bytes.load(Ordering::Relaxed), Ordering::Relaxed);
        freed
    }

    /// Sweep periodically after the pool crosses its cap. Pool accounting also
    /// includes buffers with active Array leases, which cannot be freed; avoid
    /// rescanning those entries before every tensor allocation.
    pub fn sweep_if_over(&self) -> bool {
        self.sweep_if_over_inner(None)
    }

    pub fn sweep_if_over_completed(&self, completed: u64) -> bool {
        self.sweep_if_over_inner(Some(completed))
    }

    fn sweep_if_over_inner(&self, completed: Option<u64>) -> bool {
        let limit = self.limit.load(Ordering::Relaxed);
        if limit == 0 {
            return false;
        }
        let bytes = self.bytes();
        if bytes <= limit {
            return false;
        }
        let last = self.last_sweep_bytes.load(Ordering::Relaxed);
        if bytes.saturating_sub(last) < limit {
            return false;
        }
        if self
            .last_sweep_bytes
            .compare_exchange(last, bytes, Ordering::Relaxed, Ordering::Relaxed)
            .is_err()
        {
            return false;
        }
        self.sweep_inner(completed);
        true
    }
}

impl Default for BufferPool {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod resident_memory_tests {
    use super::*;
    use std::cell::Cell;

    // These checks exercise only host bookkeeping. They never create a Metal
    // device, command queue, or buffer and can run without a GPU slot.
    #[test]
    fn resident_memory_cap_refuses_before_allocation_and_accepts_exact_boundary() {
        let pool = BufferPool::new();
        pool.bytes.store(64, Ordering::Relaxed);
        {
            let mut admission = pool.admission.lock().unwrap();
            admission.reserved = 128;
            admission.budget = 256;
        }
        let called = Cell::new(false);
        let refused = pool.allocate_new(65, 0, || {
            called.set(true);
            Err(Error::msg("allocator reached"))
        });
        assert!(!called.get());
        assert!(
            refused
                .err()
                .unwrap()
                .to_string()
                .contains("RAM budget exceeded")
        );
        let exact = pool.allocate_new(64, 0, || {
            called.set(true);
            Err(Error::msg("allocator reached"))
        });
        assert!(called.get());
        assert_eq!(exact.err().unwrap().to_string(), "allocator reached");
        assert!(pool.ensure_room(usize::MAX, usize::MAX, 1, 0).is_err());
    }

    #[test]
    fn resident_memory_reservations_release_only_their_own_bytes() {
        let pool = BufferPool::new();
        {
            let mut admission = pool.admission.lock().unwrap();
            admission.reservations.insert(1, (100, 500));
            admission.reservations.insert(2, (200, 600));
            admission.reserved = 300;
            admission.budget = 500;
        }
        pool.reserved_bytes.store(300, Ordering::Relaxed);
        pool.release_model_memory(1);
        assert_eq!(pool.reserved_model_bytes(), 200);
        assert_eq!(pool.admission.lock().unwrap().budget, 600);
        pool.release_model_memory(1);
        assert_eq!(pool.reserved_model_bytes(), 200);
        pool.release_model_memory(2);
        assert_eq!(pool.reserved_model_bytes(), 0);
        assert_eq!(pool.admission.lock().unwrap().budget, 0);
        let reached = pool.allocate_new(usize::MAX, 0, || Err(Error::msg("uncapped")));
        assert_eq!(reached.err().unwrap().to_string(), "uncapped");
    }
}
