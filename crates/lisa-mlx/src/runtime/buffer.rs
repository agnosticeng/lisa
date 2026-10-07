use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, RwLock};

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLResource};

use super::env::{ZERO_NS, ns};

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

    /// A retained `MTLBuffer` handle for handing to another owner (the shim);
    /// the buffer stays alive through ordinary objc2 refcounting.
    pub fn retain_shared_mtlbuffer(&self) -> Retained<ProtocolObject<dyn MTLBuffer>> {
        self.retain_shared().raw.clone()
    }

    /// Stable identity for the encoder's dependency map: the `MTLBuffer` object.
    pub fn id(&self) -> usize {
        self.id
    }

    /// Diagnostic (`LISA_RB_DEBUG`): the id of the last command buffer that
    /// bound this buffer.
    pub fn debug_used_cb(&self) -> u64 {
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
        let eligible =
            |s: &Arc<Buffer>| -> bool { Arc::strong_count(s) == 1 && s.reusable_after(completed) };
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
            // graph for this; zero on reuse instead.
            {
                let t0 = std::time::Instant::now();
                unsafe { std::ptr::write_bytes(b.contents(), 0, b.length()) };
                ZERO_NS.fetch_add(
                    t0.elapsed().as_nanos() as u64,
                    std::sync::atomic::Ordering::Relaxed,
                );
            }
            b
        })
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
