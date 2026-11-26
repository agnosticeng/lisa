use std::collections::{BTreeMap, HashMap};
use std::ops::Bound::{Excluded, Unbounded};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, RwLock, Weak};

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLResource};

use super::env::ns;
use super::residency::ResidencySet;
use crate::error::{Error, Result};

// ─────────────────────────────── buffers ───────────────────────────────

static NEXT_BUFFER_ID: AtomicUsize = AtomicUsize::new(1);

struct BufferLabel {
    current: Option<String>,
    // External owners can set the underlying resource label without this
    // cache. Keep those resources on the original unconditional path.
    owned: bool,
    #[cfg(test)]
    updates: usize,
}

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
    /// Retained Buffer aliases share one label cache for their Metal object.
    label: Arc<Mutex<BufferLabel>>,
}

// SAFETY: MTLBuffer objects may be retained and bound from any thread; `used_cb` is
// atomic and the label cache is behind a `Mutex`. CPU access to `contents()` is
// ordered by the pool's completion watermark, not by this type.
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
            label: Arc::new(Mutex::new(BufferLabel {
                current: None,
                owned: true,
                #[cfg(test)]
                updates: 0,
            })),
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
        let result = Self::wrap(raw);
        result.label.lock().unwrap().owned = false;
        result
    }

    /// Retain a second owner of the same `MTLBuffer`.
    pub fn retain_shared(&self) -> Self {
        // SAFETY: `self.raw` is a valid object; retain bumps its count.
        let raw = unsafe { Retained::retain(Retained::as_ptr(&self.raw) as *mut _) }
            .expect("retain MTLBuffer");
        Self {
            raw,
            id: NEXT_BUFFER_ID.fetch_add(1, Ordering::Relaxed),
            len: self.len,
            used_cb: AtomicU64::new(0),
            label: self.label.clone(),
        }
    }

    pub fn mtlbuffer(&self) -> &ProtocolObject<dyn MTLBuffer> {
        // The public handle also exposes MTLResource::setLabel. Its owner can
        // change that property without going through our shared label cache.
        self.label.lock().unwrap().owned = false;
        &self.raw
    }

    /// Internal bindings borrow the resource without changing its label.
    pub(super) fn raw_buffer(&self) -> &ProtocolObject<dyn MTLBuffer> {
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
        self.label.lock().unwrap().owned = false;
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
        let mut cached = self.label.lock().unwrap();
        if cached.owned && cached.current.as_deref() == Some(label) {
            return;
        }
        self.raw.setLabel(Some(&ns(label)));
        match &mut cached.current {
            Some(current) => label.clone_into(current),
            None => cached.current = Some(label.to_owned()),
        }
        #[cfg(test)]
        {
            cached.updates += 1;
        }
    }

    /// CPU pointer for shared-storage buffers.
    pub fn contents(&self) -> *mut u8 {
        self.raw.contents().as_ptr() as *mut u8
    }
}

impl AsRef<ProtocolObject<dyn MTLBuffer>> for Buffer {
    fn as_ref(&self) -> &ProtocolObject<dyn MTLBuffer> {
        self.mtlbuffer()
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

impl ResidentMemoryReservation {
    pub(crate) fn shrink_to(&mut self, pool: &Arc<BufferPool>, bytes: usize) -> Result<()> {
        if !Arc::ptr_eq(pool, &self.pool) {
            return Err(Error::msg(
                "resident reservation belongs to another allocator",
            ));
        }
        // Synchronization is outside the admission lock. The caller has also
        // released the source owners whose charge is being removed.
        self.commands.flush_and_wait()?;
        self.pool.shrink_model_memory(self.id, bytes)
    }
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
    buckets: RwLock<PoolBuckets>,
    bytes: AtomicUsize,
    limit: AtomicUsize,
    explicit_limit: AtomicBool,
    last_sweep_bytes: AtomicUsize,
    admission: Mutex<ResidentAdmission>,
    reserved_bytes: AtomicUsize,
    resident_budget: AtomicUsize,
    /// A held residency set that keeps every pooled allocation wired across
    /// idle gaps, while a loaded model holds it.
    resident: Mutex<Weak<ResidencySet>>,
}

#[derive(Default)]
struct PoolBuckets {
    general: BTreeMap<usize, Vec<Arc<Buffer>>>,
    // Only nonescaping recorded-program scratch can enter this lane. Its key
    // includes the Commands identity: encoding order proves reuse on one
    // serial queue, never across queues that happen to share this allocator.
    program: BTreeMap<(usize, usize), Vec<Arc<Buffer>>>,
}

impl BufferPool {
    pub fn new() -> Self {
        Self {
            buckets: RwLock::new(PoolBuckets::default()),
            bytes: AtomicUsize::new(0),
            limit: AtomicUsize::new(0),
            explicit_limit: AtomicBool::new(false),
            last_sweep_bytes: AtomicUsize::new(0),
            admission: Mutex::new(ResidentAdmission::default()),
            reserved_bytes: AtomicUsize::new(0),
            resident_budget: AtomicUsize::new(0),
            resident: Mutex::new(Weak::new()),
        }
    }

    /// Keep every pooled allocation, present and future, in `set` while it
    /// lives: new allocations join it and swept ones leave it. The caller
    /// commits the set.
    pub(crate) fn keep_resident(&self, set: &Arc<ResidencySet>) {
        let buckets = self.buckets.read().unwrap();
        for buffer in buckets
            .general
            .values()
            .chain(buckets.program.values())
            .flatten()
        {
            set.add(buffer);
        }
        *self.resident.lock().unwrap() = Arc::downgrade(set);
    }

    fn resident_set(&self) -> Option<Arc<ResidencySet>> {
        self.resident.lock().unwrap().upgrade()
    }

    /// Explicit soft retained-buffer limit; `0` disables automatic sweeping.
    /// This override remains active when a resident RAM reservation exists.
    pub fn set_limit(&self, bytes: usize) {
        self.limit.store(bytes, Ordering::Relaxed);
        self.explicit_limit.store(true, Ordering::Release);
    }

    /// Install the runtime default before publishing the pool. A live resident
    /// reservation replaces only this default miss-time policy with hard-budget
    /// reclamation; caller overrides and explicit trims remain independent.
    pub(super) fn set_default_limit(&self, bytes: usize) {
        self.limit.store(bytes, Ordering::Relaxed);
        self.explicit_limit.store(false, Ordering::Release);
    }

    /// An unrelated shape miss must not discard reusable arenas merely because
    /// live private weights put the pool above its default soft threshold. With
    /// a resident reservation, allocate_new_in_lane atomically checks mapped +
    /// private + requested bytes and reclaims completed idle storage if needed.
    /// A zero-byte reservation retains that cap until its identity is released.
    pub(super) fn sweep_on_allocation_miss(&self, completed: u64, program_scratch: bool) {
        let admitted = self.has_resident_admission();
        if admitted && !self.explicit_limit.load(Ordering::Acquire) {
            return;
        }
        if admitted || program_scratch {
            self.sweep_if_over_completed(completed);
        } else {
            self.sweep_if_over();
        }
    }

    pub fn bytes(&self) -> usize {
        self.bytes.load(Ordering::Relaxed)
    }

    /// Mapped model bytes are counted once, independently of overlapping views.
    pub fn reserved_model_bytes(&self) -> usize {
        self.reserved_bytes.load(Ordering::Acquire)
    }

    pub fn has_resident_admission(&self) -> bool {
        self.resident_budget.load(Ordering::Acquire) != 0
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
            self.resident_budget
                .store(admission.budget, Ordering::Release);
        }
    }

    fn shrink_model_memory(&self, id: usize, bytes: usize) -> Result<()> {
        let mut admission = self.admission.lock().unwrap();
        let &(old, cap) = admission
            .reservations
            .get(&id)
            .ok_or_else(|| Error::msg("unknown resident reservation"))?;
        if bytes > old {
            return Err(Error::msg("resident reservation can only shrink"));
        }
        admission.reservations.insert(id, (bytes, cap));
        admission.reserved -= old - bytes;
        self.reserved_bytes
            .store(admission.reserved, Ordering::Release);
        // Preserve the original cap and identity even when bytes reaches zero.
        self.resident_budget
            .store(admission.budget, Ordering::Release);
        Ok(())
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
        self.resident_budget.store(cap, Ordering::Release);
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
            return Err(Error::AllocationUnavailable(format!(
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
        self.allocate_new_in_lane(bytes, completed, None, create)
    }

    pub(super) fn allocate_program_scratch(
        &self,
        bytes: usize,
        completed: u64,
        queue: usize,
        create: impl FnOnce() -> Result<Arc<Buffer>>,
    ) -> Result<Arc<Buffer>> {
        self.allocate_new_in_lane(bytes, completed, Some(queue), create)
    }

    fn allocate_new_in_lane(
        &self,
        bytes: usize,
        completed: u64,
        program_queue: Option<usize>,
        create: impl FnOnce() -> Result<Arc<Buffer>>,
    ) -> Result<Arc<Buffer>> {
        let admission = self.admission.lock().unwrap();
        if admission.reserved > 0 || !admission.reservations.is_empty() {
            self.ensure_room(admission.reserved, admission.budget, bytes, completed)?;
        }
        let buffer = create()?;
        // The set's heartbeat commits the addition.
        if let Some(set) = self.resident_set() {
            set.add(&buffer);
        }
        if let Some(queue) = program_queue {
            self.bytes.fetch_add(buffer.length(), Ordering::Relaxed);
            self.buckets
                .write()
                .unwrap()
                .program
                .entry((queue, bucket_size(buffer.length())))
                .or_default()
                .push(buffer.clone());
        } else {
            self.put(buffer.clone());
        }
        Ok(buffer)
    }

    /// Lease exact-capacity private scratch after its last host owner has
    /// encoded every access and released it. Only DispatchProgram may use this
    /// lane; it never exports an arena, writes it on the CPU, or binds it on a
    /// different queue. The next full GPU producer is therefore ordered after
    /// every former GPU reader even when completion has not happened yet.
    pub(super) fn take_program_scratch(&self, size: usize, queue: usize) -> Option<Arc<Buffer>> {
        // As with general leases, eligibility plus cloning must be exclusive.
        #[expect(
            clippy::readonly_write_lock,
            reason = "atomic private scratch lease acquisition"
        )]
        let buckets = self.buckets.write().unwrap();
        buckets
            .program
            .get(&(queue, bucket_size(size)))?
            .iter()
            .rev()
            .take(64)
            .find(|buffer| Arc::strong_count(buffer) == 1)
            .cloned()
    }

    /// The smallest free pooled buffer that can hold `size`. Recent matching
    /// buffers are checked first. A pool-owned reference keeps each allocation
    /// reusable after the caller releases its lease.
    pub fn take(&self, size: usize, completed: u64) -> Option<Arc<Buffer>> {
        self.take_inner(size, completed, false, false)
    }

    /// Lease storage only for a later GPU write on this pool's ordered queue.
    /// Existing GPU readers have already been encoded when the last Array
    /// owner disappears. Their commands precede the new write; CPU writers
    /// must continue to use `take` and wait for the completion watermark.
    pub(super) fn take_for_ordered_gpu_write(&self, size: usize) -> Option<Arc<Buffer>> {
        self.take_inner(size, 0, true, false)
    }

    /// Retained small state must not keep an arbitrarily larger scratch
    /// allocation alive. Restrict reuse to its exact 1 KiB size bucket while
    /// preserving the caller's existing completion or ordered-queue policy.
    pub(super) fn take_exact(
        &self,
        size: usize,
        completed: u64,
        ordered_gpu_write: bool,
    ) -> Option<Arc<Buffer>> {
        self.take_inner(size, completed, ordered_gpu_write, true)
    }

    fn take_inner(
        &self,
        size: usize,
        completed: u64,
        ordered_gpu_write: bool,
        exact_bucket: bool,
    ) -> Option<Arc<Buffer>> {
        let eligible = |s: &Arc<Buffer>| -> bool {
            Arc::strong_count(s) == 1 && (ordered_gpu_write || s.reusable_after(completed))
        };
        // The pool retains its owner Arc while callers hold leases. A read
        // lock is not sufficient here: two concurrent allocators could both
        // observe strong_count == 1 and clone the same pooled buffer before
        // either clone becomes visible, then bind it for unrelated writes.
        // Serialize the eligibility check and lease acquisition with sweeps.
        #[expect(
            clippy::readonly_write_lock,
            reason = "eligibility and Arc lease acquisition must be exclusive"
        )]
        let buckets = self.buckets.write().unwrap();
        // Keep the pool's strong reference while a buffer is leased. The
        // caller gets a clone; `strong_count == 1` makes it available again
        // when the caller drops it. Search newest-first because the most
        // recently allocated sizes are the likeliest to recur on the next op.
        let want = bucket_size(size);
        let take_from = |buffers: &[Arc<Buffer>]| -> Option<Arc<Buffer>> {
            buffers
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
        if let Some(s) = buckets
            .general
            .get(&want)
            .and_then(|buffers| take_from(buffers))
        {
            return Some(s);
        }
        if exact_bucket {
            return None;
        }
        // Ascending sizes make the first eligible bucket the exact same best
        // fit as a full scan, without visiting unrelated smaller/larger sizes.
        for (_, buffers) in buckets.general.range((Excluded(want), Unbounded)) {
            // A busy exact/smaller bucket must not hide an eligible larger
            // buffer. The lease and completion checks remain under this lock.
            if let Some(buffer) = take_from(buffers) {
                return Some(buffer);
            }
        }
        None
    }

    pub(super) fn put(&self, buf: Arc<Buffer>) {
        self.bytes.fetch_add(buf.length(), Ordering::Relaxed);
        let b = bucket_size(buf.length());
        self.buckets
            .write()
            .unwrap()
            .general
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
        let mut released = Vec::new();
        let mut buckets = self.buckets.write().unwrap();
        buckets.general.retain(|_, subs| {
            subs.retain(|s| {
                if Arc::strong_count(s) == 1
                    && completed.is_none_or(|watermark| s.reusable_after(watermark))
                {
                    freed += s.length();
                    released.push(s.clone());
                    false
                } else {
                    true
                }
            });
            // Context growth creates many sizes that never recur. Do not
            // scan empty historical buckets on subsequent allocation misses.
            !subs.is_empty()
        });
        // Keep in-flight private scratch resident and charged even when a
        // general unbounded sweep was requested. Known-complete trim/admission
        // paths reclaim it; no wait occurs while either allocator lock is held.
        if let Some(completed) = completed {
            buckets.program.retain(|_, subs| {
                subs.retain(|buffer| {
                    if Arc::strong_count(buffer) == 1 && buffer.reusable_after(completed) {
                        freed += buffer.length();
                        released.push(buffer.clone());
                        false
                    } else {
                        true
                    }
                });
                !subs.is_empty()
            });
        }
        self.bytes.fetch_sub(freed, Ordering::Relaxed);
        self.last_sweep_bytes
            .store(self.bytes.load(Ordering::Relaxed), Ordering::Relaxed);
        drop(buckets);
        // Released allocations leave the resident set before their memory
        // is freed.
        if let Some(set) = self.resident_set()
            && !released.is_empty()
        {
            for buffer in &released {
                set.remove(buffer);
            }
            set.commit();
        }
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
mod reuse_tests {
    use super::*;
    use crate::runtime::MetalRuntime;

    #[test]
    fn recorded_scratch_is_private_exact_queue_bound_and_completion_accounted() {
        let rt = MetalRuntime::new(4).unwrap();
        let queue = rt.commands.program_identity();
        let scratch = rt.buffer_for_recorded_scratch(1536).unwrap();
        let id = scratch.id();
        assert_eq!(scratch.length(), 2048);
        scratch.mark_used_cb(100);
        drop(scratch);
        assert_eq!(rt.pool.bytes(), 2048);
        assert!(rt.pool.take(1536, u64::MAX).is_none());
        assert!(rt.pool.take_exact(1536, u64::MAX, true).is_none());
        assert!(rt.pool.take_program_scratch(1024, queue).is_none());
        assert!(rt.pool.take_program_scratch(1536, queue ^ 1).is_none());
        let leased = rt.pool.take_program_scratch(1536, queue).unwrap();
        assert_eq!(leased.id(), id, "private reuse does not wait for cb100");
        let cpu = rt
            .buffer_for_compact_full_write(1536, "scratch_cpu_import")
            .unwrap();
        assert_ne!(cpu.id(), id, "CPU imports cannot enter the private lane");
        assert_eq!(rt.pool.bytes(), 4096);
        leased.mark_used_cb(120);
        drop(leased);
        assert_eq!(
            rt.pool.sweep(),
            0,
            "in-flight scratch stays resident and charged"
        );
        assert_eq!(rt.pool.sweep_completed(119), 0);
        assert_eq!(rt.pool.sweep_completed(120), 2048);
        assert_eq!(rt.pool.bytes(), 2048);
    }

    #[test]
    fn recorded_scratch_concurrent_leases_are_exclusive_before_completion() {
        let rt = MetalRuntime::new(4).unwrap();
        let queue = rt.commands.program_identity();
        let scratch = rt.buffer_for_recorded_scratch(4096).unwrap();
        let id = scratch.id();
        scratch.mark_used_cb(100);
        drop(scratch);
        let barrier = Arc::new(std::sync::Barrier::new(16));
        let workers = (0..16)
            .map(|_| {
                let pool = rt.pool.clone();
                let barrier = barrier.clone();
                std::thread::spawn(move || {
                    barrier.wait();
                    pool.take_program_scratch(4096, queue)
                })
            })
            .collect::<Vec<_>>();
        let leases = workers
            .into_iter()
            .filter_map(|worker| worker.join().unwrap())
            .collect::<Vec<_>>();
        assert_eq!(leases.len(), 1, "one encoding owner per arena");
        assert_eq!(leases[0].id(), id);
        assert!(rt.pool.take_program_scratch(4096, queue).is_none());
        assert_eq!(
            rt.pool.sweep_completed(100),
            0,
            "a live lease cannot be reclaimed"
        );
        drop(leases);
        assert_eq!(rt.pool.sweep_completed(99), 0);
        assert_eq!(rt.pool.sweep_completed(100), 4096);
    }

    #[test]
    fn ordered_buckets_preserve_newest_window_and_completed_best_fit() {
        let rt = MetalRuntime::new(4).unwrap();
        let mut held = (0..66)
            .map(|_| rt.buffer(4096, "held_small").unwrap())
            .collect::<Vec<_>>();
        let oldest = held.remove(0);
        let oldest_id = oldest.id();
        let newest_id = held.last().unwrap().id();
        let next_id = held[held.len() - 2].id();
        held.last().unwrap().mark_used_cb(100);
        let larger = rt.buffer(8192, "larger").unwrap();
        let larger_id = larger.id();
        drop(oldest);
        drop(larger);
        // The only free small allocation is outside the unchanged newest-64
        // search window. A larger eligible bucket must still be selected.
        let larger = rt.pool.take(4096, 0).unwrap();
        assert_eq!(larger.id(), larger_id);
        assert_ne!(larger.id(), oldest_id);
        drop(held.pop()); // Newest is in flight.
        drop(held.pop()); // Second-newest is completed.
        let next = rt.pool.take(4096, 0).unwrap();
        assert_eq!(next.id(), next_id);
        drop(next);
        let newest = rt.pool.take(4096, 100).unwrap();
        assert_eq!(newest.id(), newest_id);
    }

    #[test]
    fn concurrent_larger_bucket_leases_preserve_completion_and_exclusivity() {
        let rt = MetalRuntime::new(4).unwrap();
        let exact = rt.buffer(1024, "held_exact").unwrap();
        let mut seeds = Vec::new();
        let mut expected = Vec::new();
        for index in 0..8 {
            let seed = rt.buffer((index + 2) * 1024, "larger_seed").unwrap();
            if index % 2 == 0 {
                expected.push(seed.id());
            } else {
                seed.mark_used_cb(100);
            }
            seeds.push(seed);
        }
        let first_in_flight_id = seeds[1].id();
        drop(seeds);
        let workers = 16;
        let barrier = Arc::new(std::sync::Barrier::new(workers));
        let joins = (0..workers)
            .map(|_| {
                let pool = rt.pool.clone();
                let barrier = barrier.clone();
                std::thread::spawn(move || {
                    barrier.wait();
                    pool.take(1024, 50)
                })
            })
            .collect::<Vec<_>>();
        let leases = joins
            .into_iter()
            .filter_map(|thread| thread.join().unwrap())
            .collect::<Vec<_>>();
        let mut actual = leases.iter().map(|lease| lease.id()).collect::<Vec<_>>();
        actual.sort_unstable();
        expected.sort_unstable();
        assert_eq!(
            actual, expected,
            "each completed allocation yields one lease"
        );
        assert!(actual.iter().all(|&id| id != exact.id()));
        let completed = rt.pool.take(1024, 100).unwrap();
        assert_eq!(completed.id(), first_in_flight_id);
    }

    #[test]
    fn buffer_labels_skip_equal_values_and_share_retained_alias_updates() {
        let rt = MetalRuntime::new(4).unwrap();
        let buffer = rt.buffer(1024, "original label").unwrap();
        assert_eq!(buffer.label.lock().unwrap().updates, 1);
        for _ in 0..8 {
            buffer.set_label("original label");
        }
        assert_eq!(buffer.label.lock().unwrap().updates, 1);
        let alias = buffer.retain_shared();
        assert!(Arc::ptr_eq(&buffer.label, &alias.label));
        alias.set_label("changed label");
        buffer.set_label("changed label");
        assert_eq!(buffer.label.lock().unwrap().updates, 2);
        assert_eq!(buffer.raw.label().unwrap().to_string(), "changed label");
        buffer.set_label("original label");
        assert_eq!(alias.raw.label().unwrap().to_string(), "original label");
        alias.set_label("");
        alias.set_label("");
        assert_eq!(buffer.label.lock().unwrap().updates, 4);
        assert_eq!(buffer.raw.label().unwrap().to_string(), "");
    }

    #[test]
    fn externally_shared_buffer_labels_keep_unconditional_updates() {
        let rt = MetalRuntime::new(4).unwrap();
        let buffer = rt.buffer(1024, "owner label").unwrap();
        let raw = buffer.retain_shared_mtlbuffer();
        raw.setLabel(Some(&ns("external label")));
        buffer.set_label("owner label");
        assert_eq!(raw.label().unwrap().to_string(), "owner label");
        assert_eq!(buffer.label.lock().unwrap().updates, 2);
        let imported = Buffer::shared(raw.clone());
        imported.set_label("import label");
        raw.setLabel(Some(&ns("another external label")));
        imported.set_label("import label");
        assert_eq!(raw.label().unwrap().to_string(), "import label");
        assert_eq!(imported.label.lock().unwrap().updates, 2);

        for via_as_ref in [false, true] {
            let buffer = rt.buffer(1024, "borrowed label").unwrap();
            let initial_updates = buffer.label.lock().unwrap().updates;
            let raw: &ProtocolObject<dyn MTLBuffer> = if via_as_ref {
                buffer.as_ref().as_ref()
            } else {
                buffer.mtlbuffer()
            };
            raw.setLabel(Some(&ns("externally relabeled borrow")));
            buffer.set_label("borrowed label");
            assert_eq!(raw.label().unwrap().to_string(), "borrowed label");
            assert_eq!(buffer.label.lock().unwrap().updates, initial_updates + 1);
        }
    }

    #[test]
    fn concurrent_retained_alias_labels_stay_coherent() {
        let rt = MetalRuntime::new(4).unwrap();
        let buffer = rt.buffer(1024, "start").unwrap();
        let joins = (0..8)
            .map(|index| {
                let alias = buffer.retain_shared();
                std::thread::spawn(move || {
                    for _ in 0..16 {
                        alias.set_label(&format!("worker {index}"));
                    }
                })
            })
            .collect::<Vec<_>>();
        for thread in joins {
            thread.join().unwrap();
        }
        let cached = buffer.label.lock().unwrap();
        assert_eq!(
            cached.current.as_deref(),
            Some(buffer.raw.label().unwrap().to_string().as_str())
        );
        drop(cached);
        buffer.set_label("final");
        assert_eq!(buffer.raw.label().unwrap().to_string(), "final");
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
    fn resident_memory_shrink_preserves_cap_zero_identity_and_exact_release() {
        let pool = BufferPool::new();
        pool.bytes.store(64, Ordering::Relaxed);
        {
            let mut admission = pool.admission.lock().unwrap();
            admission.reservations.insert(1, (128, 256));
            admission.reservations.insert(2, (32, 384));
            admission.reserved = 160;
            admission.budget = 256;
        }
        pool.reserved_bytes.store(160, Ordering::Relaxed);
        pool.resident_budget.store(256, Ordering::Relaxed);
        assert!(pool.shrink_model_memory(1, 129).is_err());
        assert!(pool.shrink_model_memory(3, 0).is_err());
        assert_eq!(pool.reserved_model_bytes(), 160);
        pool.shrink_model_memory(1, 64).unwrap();
        assert_eq!(pool.reserved_model_bytes(), 96);
        pool.shrink_model_memory(1, 0).unwrap();
        pool.shrink_model_memory(2, 0).unwrap();
        assert_eq!(pool.reserved_model_bytes(), 0);
        assert!(pool.has_resident_admission());
        assert_eq!(pool.admission.lock().unwrap().budget, 256);
        let called = Cell::new(false);
        let denied = pool.allocate_new(193, 0, || {
            called.set(true);
            Err(Error::msg("reached"))
        });
        assert!(denied.is_err());
        assert!(
            !called.get(),
            "zero mapped bytes must not disable private admission"
        );
        let exact = pool.allocate_new(192, 0, || {
            called.set(true);
            Err(Error::msg("reached"))
        });
        assert!(called.get());
        assert_eq!(exact.err().unwrap().to_string(), "reached");
        pool.release_model_memory(1);
        assert!(pool.has_resident_admission());
        assert_eq!(pool.admission.lock().unwrap().budget, 384);
        pool.release_model_memory(2);
        assert!(!pool.has_resident_admission());
        assert_eq!(pool.reserved_model_bytes(), 0);
        assert_eq!(pool.admission.lock().unwrap().budget, 0);
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

    #[test]
    fn held_residency_tracks_pooled_allocations() {
        let rt = crate::runtime::MetalRuntime::new(4).unwrap();
        let before = rt.buffer(4096, "resident_before").unwrap();
        // Residency sets need macOS 15; older systems keep no set.
        let Some(set) = ResidencySet::new(rt.device(), 4) else {
            return;
        };
        let set = Arc::new(set);
        rt.pool.keep_resident(&set);
        assert!(set.contains(before.id()), "existing allocations join");
        let after = rt.buffer(1 << 20, "resident_after").unwrap();
        assert!(set.contains(after.id()), "new allocations join");
        let (before_id, after_id) = (before.id(), after.id());
        drop((before, after));
        rt.pool.sweep();
        assert!(!set.contains(before_id) && !set.contains(after_id));
        assert_eq!(set.allocation_count(), 0, "swept allocations leave");
        drop(set);
        let untracked = rt.buffer(8192, "resident_dropped").unwrap();
        assert!(
            rt.pool.resident_set().is_none(),
            "a dropped set is not kept"
        );
        drop(untracked);
    }

    #[test]
    fn device_zeroed_storage_reads_zero_after_reuse() {
        let rt = crate::runtime::MetalRuntime::new(4).unwrap();
        let dirty = rt.buffer_for_full_write(1 << 16, "dirty").unwrap();
        // SAFETY: a fresh CPU-import lease of `length()` bytes with no GPU user.
        unsafe { std::ptr::write_bytes(dirty.contents(), 0xa5, dirty.length()) };
        let id = dirty.id();
        drop(dirty);
        let zeroed = rt.buffer_zeroed_on_gpu(1 << 16, "device_zeros").unwrap();
        assert_eq!(zeroed.id(), id, "the dirty lease is reused uncleared");
        rt.commands.flush_and_wait().unwrap();
        // SAFETY: the queue completed; the lease holds `1 << 16` bytes.
        let bytes = unsafe { std::slice::from_raw_parts(zeroed.contents(), 1 << 16) };
        assert!(bytes.iter().all(|&b| b == 0), "the GPU cleared every byte");
    }
}
