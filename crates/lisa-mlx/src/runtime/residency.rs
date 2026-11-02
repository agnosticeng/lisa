use std::collections::HashSet;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLAllocation, MTLCommandQueue, MTLDevice, MTLResidencySet, MTLResidencySetDescriptor,
};

use super::Buffer;

/// Residency for buffers reached indirectly through GPU addresses. A command
/// buffer can attach this set so Metal makes its allocations resident only for
/// that command buffer's execution.
pub struct ResidencySet {
    raw: Retained<ProtocolObject<dyn MTLResidencySet>>,
    buffers: Mutex<HashSet<usize>>,
    allocation_bytes: AtomicUsize,
    queue: Mutex<Option<Retained<ProtocolObject<dyn MTLCommandQueue>>>>,
}

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
    /// This follows the model-wide DS4 path; per-command-buffer residency sets
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
        self.raw.commit();
    }

    /// Request residency after a committed dynamic membership update.
    pub fn request_residency(&self) {
        self.raw.requestResidency();
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
        &*(buffer.as_ref() as *const ProtocolObject<dyn objc2_metal::MTLBuffer>
            as *const ProtocolObject<dyn MTLAllocation>)
    }
}
