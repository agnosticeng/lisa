use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLDevice, MTLFence};

use crate::error::Result;

use super::env::err;

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
    pub(super) fn new(device: &ProtocolObject<dyn MTLDevice>) -> Result<Self> {
        let raw = device.newFence().ok_or_else(|| err("newFence"))?;
        Ok(Self { raw })
    }

    pub fn raw(&self) -> &ProtocolObject<dyn MTLFence> {
        &self.raw
    }
}
