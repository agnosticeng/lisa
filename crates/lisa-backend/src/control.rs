//! Queue and allocation mechanisms. Model admission and scheduling are callers' policy.
use crate::HostAccessDevice;

#[derive(Clone, Copy, Debug)]
pub struct MemoryUsage {
    pub active: usize,
    pub private: usize,
}

#[derive(Clone, Copy, Debug)]
pub enum ExecutionScope {
    /// Reuse only audited output allocations ordered on the same serial queue.
    OrderedOutputReuse,
    /// Enable the explicitly admitted affine verify arithmetic implementation.
    AffineVerifySplitK,
}

/// Provider identity and the stable arithmetic implementation key. The key
/// must differ whenever stored numerical state cannot be reused safely.
#[derive(Clone, Debug)]
pub struct DeviceIdentity {
    pub provider: String,
    pub device: String,
    pub architecture: String,
    pub arithmetic: Vec<u8>,
}

pub trait ManagedDevice: HostAccessDevice {
    fn identity(&self) -> DeviceIdentity;
    type Reservation;
    type Scope;
    fn memory_usage(&self) -> anyhow::Result<MemoryUsage>;
    fn reserve_residency(&self, bytes: usize, budget: usize) -> anyhow::Result<Self::Reservation>;
    /// End the current execution encoder without submitting or waiting.
    fn execution_boundary(&self);
    fn submit(&self, wait: bool) -> anyhow::Result<()>;
    fn enter_scope(&self, scope: ExecutionScope) -> Self::Scope;
    fn trim_unused(&self);
    /// Complete queued work and release all unleased pooled allocations.
    fn clear_unused(&self);
}
