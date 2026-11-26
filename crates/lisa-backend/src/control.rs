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
    /// Reduce an existing reservation after the caller has completed all users
    /// and released the storage no longer covered by it. Never increase bytes
    /// or change the reservation's budget. A zero-byte reservation must keep
    /// private-allocation admission active until it is dropped. Providers that
    /// do not support shrinking retain their original charge and return false.
    fn shrink_residency(
        &self,
        _reservation: &mut Self::Reservation,
        _bytes: usize,
    ) -> anyhow::Result<bool> {
        Ok(false)
    }
    /// End the current execution encoder without submitting or waiting.
    fn execution_boundary(&self);
    fn submit(&self, wait: bool) -> anyhow::Result<()>;
    /// Commit queued work, then wait only for the submission marked by
    /// `previous`. Returns this submission's mark, so a caller can keep one
    /// batch executing while it encodes the next. Providers without marks
    /// wait for everything.
    fn submit_after(&self, previous: Option<u64>) -> anyhow::Result<u64> {
        let _ = previous;
        self.submit(true)?;
        Ok(0)
    }
    fn enter_scope(&self, scope: ExecutionScope) -> Self::Scope;
    fn trim_unused(&self);
    /// Complete queued work and release all unleased pooled allocations.
    fn clear_unused(&self);
}
