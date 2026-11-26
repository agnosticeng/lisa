use std::collections::HashSet;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLAllocation, MTLCommandQueue, MTLDevice, MTLResidencySet, MTLResidencySetDescriptor,
};

use super::Buffer;

/// How often a held set requests residency again.
const HEARTBEAT: Duration = Duration::from_millis(500);

/// Residency for buffers reached indirectly through GPU addresses. A command
/// buffer can attach this set so Metal makes its allocations resident only for
/// that command buffer's execution.
pub struct ResidencySet {
    raw: Retained<ProtocolObject<dyn MTLResidencySet>>,
    buffers: Mutex<HashSet<usize>>,
    allocation_bytes: AtomicUsize,
    /// Membership changed since the last commit.
    dirty: AtomicBool,
    queue: Mutex<Option<Retained<ProtocolObject<dyn MTLCommandQueue>>>>,
}

// SAFETY: membership edits run under the `buffers` lock and queue registration under
// the `queue` lock; `commit`/`requestResidency` rely on MTLResidencySet being
// thread-safe, as Metal objects other than encoders are.
unsafe impl Send for ResidencySet {}
unsafe impl Sync for ResidencySet {}

impl ResidencySet {
    pub fn new(device: &ProtocolObject<dyn MTLDevice>, initial_capacity: usize) -> Option<Self> {
        if !objc2::available!(macos = 15.0) {
            return None;
        }
        let descriptor = MTLResidencySetDescriptor::new();
        // SAFETY: capacity is a model-derived allocation count, not a byte size.
        unsafe { descriptor.setInitialCapacity(initial_capacity) };
        let raw = device
            .newResidencySetWithDescriptor_error(&descriptor)
            .ok()?;
        Some(Self {
            raw,
            buffers: Mutex::new(HashSet::new()),
            allocation_bytes: AtomicUsize::new(0),
            dirty: AtomicBool::new(false),
            queue: Mutex::new(None),
        })
    }

    /// Add one indirectly addressed allocation before committing the set.
    pub fn add(&self, buffer: &Buffer) -> bool {
        let mut buffers = self.buffers.lock().unwrap();
        if !buffers.insert(buffer.id()) {
            return true;
        }
        self.allocation_bytes
            .fetch_add(buffer.length(), Ordering::Relaxed);
        self.raw.addAllocation(as_allocation(buffer));
        self.dirty.store(true, Ordering::Release);
        true
    }

    /// Remove one allocation from a dynamic residency set before releasing
    /// its last cache owner. The caller commits the set after batching updates.
    pub fn remove(&self, buffer: &Buffer) -> bool {
        let mut buffers = self.buffers.lock().unwrap();
        if !buffers.remove(&buffer.id()) {
            return false;
        }
        self.allocation_bytes
            .fetch_sub(buffer.length(), Ordering::Relaxed);
        self.raw.removeAllocation(as_allocation(buffer));
        self.dirty.store(true, Ordering::Release);
        true
    }

    pub fn contains(&self, buffer_id: usize) -> bool {
        self.buffers.lock().unwrap().contains(&buffer_id)
    }

    pub fn allocation_count(&self) -> usize {
        self.buffers.lock().unwrap().len()
    }

    /// Total byte length of the unique allocations added to this set.
    pub fn allocation_bytes(&self) -> usize {
        self.allocation_bytes.load(Ordering::Relaxed)
    }

    /// Register the committed set once for every command buffer on this queue.
    /// This follows the model-wide reference path; per-command-buffer residency sets
    /// repeatedly validate the same large file-backed weight mappings.
    pub fn add_to_queue(&self, queue: &Retained<ProtocolObject<dyn MTLCommandQueue>>) -> bool {
        self.add_to_queue_with_request(queue, true)
    }

    /// Register the set on the queue without asking Metal to prepare all of its
    /// allocations as resident up front. Command buffers may still attach the
    /// set when they use indirect GPU addresses.
    pub fn add_to_queue_without_request(
        &self,
        queue: &Retained<ProtocolObject<dyn MTLCommandQueue>>,
    ) -> bool {
        self.add_to_queue_with_request(queue, false)
    }

    fn add_to_queue_with_request(
        &self,
        queue: &Retained<ProtocolObject<dyn MTLCommandQueue>>,
        request_residency: bool,
    ) -> bool {
        if !objc2::available!(macos = 15.0) {
            return false;
        }
        let mut registered = self.queue.lock().unwrap();
        if registered.is_some() {
            return true;
        }
        if request_residency {
            self.raw.requestResidency();
        }
        queue.addResidencySet(&self.raw);
        *registered = Some(queue.clone());
        true
    }

    /// Finalize the resource list before attaching the set to command buffers.
    pub fn commit(&self) {
        // Membership edits hold the same lock.
        let _buffers = self.buffers.lock().unwrap();
        self.dirty.store(false, Ordering::Release);
        self.raw.commit();
    }

    /// Request residency after a committed dynamic membership update.
    pub fn request_residency(&self) {
        self.raw.requestResidency();
    }

    /// Keep the committed set resident for as long as it lives. One request
    /// holds its allocations only about two seconds, so a heartbeat thread
    /// requests residency every 500 ms until the last owner drops the set,
    /// first committing allocations added since the last commit. The set is
    /// not attached to a queue: an attached set is made resident before
    /// every command buffer starts.
    pub fn hold(self: &Arc<Self>) {
        self.raw.requestResidency();
        let set = Arc::downgrade(self);
        let spawned = std::thread::Builder::new()
            .name("lisa-residency".into())
            .spawn(move || {
                loop {
                    std::thread::sleep(HEARTBEAT);
                    let Some(set) = set.upgrade() else { break };
                    if set.dirty.load(Ordering::Acquire) {
                        set.commit();
                    }
                    set.raw.requestResidency();
                }
            });
        // Without the thread, the initial request still holds the set.
        drop(spawned);
    }

    pub fn raw(&self) -> &ProtocolObject<dyn MTLResidencySet> {
        &self.raw
    }
}

impl Drop for ResidencySet {
    fn drop(&mut self) {
        if let Some(queue) = self.queue.get_mut().unwrap().take() {
            queue.removeResidencySet(&self.raw);
        }
        self.raw.endResidency();
    }
}

fn as_allocation(buffer: &Buffer) -> &ProtocolObject<dyn MTLAllocation> {
    // SAFETY: MTLBuffer inherits MTLResource, which inherits MTLAllocation.
    unsafe {
        &*(buffer.raw_buffer() as *const ProtocolObject<dyn objc2_metal::MTLBuffer>
            as *const ProtocolObject<dyn MTLAllocation>)
    }
}
