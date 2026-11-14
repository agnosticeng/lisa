//! Opaque residency, indirect storage tables and serial full-write scratch.
//! Allocation policy and model traversal remain in the engine.
use crate::{IndirectDevice, ManagedDevice, ScalarType};

pub trait ResourceDevice: ManagedDevice + IndirectDevice<Resources: Send + Sync + 'static> {
    type Residency: Clone + Send + Sync + 'static;
    type Table: Clone + Send + Sync + 'static;
    type Workspace: Send + 'static;
    fn allocation_id(&self, tensor: &Self::Tensor) -> usize;
    fn completed_watermark(&self) -> u64;
    fn reusable_after(&self, tensor: &Self::Tensor, completed: u64) -> bool;
    fn new_residency(&self, capacity: usize) -> Option<Self::Residency>;
    fn residency_add(&self, set: &Self::Residency, tensor: &Self::Tensor) -> bool;
    fn residency_remove(&self, set: &Self::Residency, tensor: &Self::Tensor) -> bool;
    fn residency_contains(&self, set: &Self::Residency, allocation: usize) -> bool;
    fn residency_count(&self, set: &Self::Residency) -> usize;
    fn residency_bytes(&self, set: &Self::Residency) -> usize;
    fn residency_commit(&self, set: &Self::Residency);
    fn residency_request(&self, set: &Self::Residency);
    fn residency_attach(&self, set: &Self::Residency, request: bool) -> bool;
    fn indirect_table(&self, slots: &[&Self::Tensor]) -> anyhow::Result<Self::Table>;
    fn resources_with_residency(
        &self,
        slots: &[&Self::Tensor],
        set: Option<Self::Residency>,
    ) -> anyhow::Result<Self::Resources>;
    /// Bind a previously constructed table to its selected allocation leases.
    ///
    /// # Safety
    /// Every entry reachable by the dispatched operation must retain its
    /// original payload and extent in `owners` or the enclosing resident owner.
    unsafe fn bind_indirect_table(
        &self,
        table: Self::Table,
        owners: Vec<Self::Tensor>,
        set: Option<Self::Residency>,
    ) -> anyhow::Result<Self::Resources>;
    /// Retain a scratch allocation independently of temporary tensor views.
    ///
    /// # Safety
    /// All active elements must be overwritten. The caller must serialize
    /// consumers and subsequent writers on this device's queue, including
    /// concurrent host callers, and must not retain views for later readback.
    unsafe fn full_write_workspace(
        &self,
        slot: &mut Option<Self::Workspace>,
        shape: &[i32],
        dtype: ScalarType,
        label: &str,
    ) -> anyhow::Result<Self::Tensor>;
}
