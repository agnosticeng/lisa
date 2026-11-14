//! Model-independent residency and ordered scratch allocation mechanisms.
use super::*;
use std::sync::Weak;

#[derive(Clone)]
pub struct Residency {
    runtime: Weak<crate::runtime::MetalRuntime>,
    set: Arc<crate::runtime::ResidencySet>,
}
pub struct SerialWorkspace {
    runtime: Weak<crate::runtime::MetalRuntime>,
    buffer: Arc<crate::runtime::Buffer>,
}
#[derive(Clone)]
pub struct IndirectTable(Array);
impl MetalDevice {
    fn owns_residency(&self, set: &Residency) -> bool {
        set.runtime.ptr_eq(&Arc::downgrade(&self.runtime))
    }
    fn native_residency(
        &self,
        set: Option<Residency>,
    ) -> anyhow::Result<Option<Arc<crate::runtime::ResidencySet>>> {
        set.map(|set| {
            anyhow::ensure!(self.owns_residency(&set), "foreign residency context");
            Ok(set.set)
        })
        .transpose()
    }
}
impl ResourceDevice for MetalDevice {
    type Residency = Residency;
    type Table = IndirectTable;
    type Workspace = SerialWorkspace;
    fn allocation_id(&self, t: &Array) -> usize {
        t.backing_buffer_id()
    }
    fn completed_watermark(&self) -> u64 {
        self.runtime.commands.completed_watermark()
    }
    fn reusable_after(&self, t: &Array, completed: u64) -> bool {
        Arc::ptr_eq(t.t.runtime(), &self.runtime) && t.t.buffer().reusable_after(completed)
    }
    fn new_residency(&self, capacity: usize) -> Option<Residency> {
        Some(Residency {
            runtime: Arc::downgrade(&self.runtime),
            set: Arc::new(crate::runtime::ResidencySet::new(
                self.runtime.device(),
                capacity,
            )?),
        })
    }
    fn residency_add(&self, set: &Residency, t: &Array) -> bool {
        self.owns_residency(set)
            && Arc::ptr_eq(t.t.runtime(), &self.runtime)
            && set.set.add(t.t.buffer())
    }
    fn residency_remove(&self, set: &Residency, t: &Array) -> bool {
        self.owns_residency(set)
            && Arc::ptr_eq(t.t.runtime(), &self.runtime)
            && set.set.remove(t.t.buffer())
    }
    fn residency_contains(&self, set: &Residency, id: usize) -> bool {
        self.owns_residency(set) && set.set.contains(id)
    }
    fn residency_count(&self, set: &Residency) -> usize {
        assert!(self.owns_residency(set), "foreign residency context");
        set.set.allocation_count()
    }
    fn residency_bytes(&self, set: &Residency) -> usize {
        assert!(self.owns_residency(set), "foreign residency context");
        set.set.allocation_bytes()
    }
    fn residency_commit(&self, set: &Residency) {
        assert!(self.owns_residency(set), "foreign residency context");
        set.set.commit()
    }
    fn residency_request(&self, set: &Residency) {
        assert!(self.owns_residency(set), "foreign residency context");
        set.set.request_residency()
    }
    fn residency_attach(&self, set: &Residency, request: bool) -> bool {
        self.owns_residency(set)
            && if request {
                self.runtime.commands.add_residency_set_to_queue(&set.set)
            } else {
                self.runtime
                    .commands
                    .add_residency_set_to_queue_without_request(&set.set)
            }
    }
    fn indirect_table(&self, slots: &[&Array]) -> anyhow::Result<IndirectTable> {
        Ok(IndirectTable(MetalDevice::indirect_table(self, slots)?))
    }
    fn resources_with_residency(
        &self,
        slots: &[&Array],
        set: Option<Residency>,
    ) -> anyhow::Result<IndirectResources> {
        Ok(MetalDevice::resources_with_residency(
            self,
            slots,
            self.native_residency(set)?,
        )?)
    }
    unsafe fn bind_indirect_table(
        &self,
        table: IndirectTable,
        owners: Vec<Array>,
        set: Option<Residency>,
    ) -> anyhow::Result<IndirectResources> {
        // SAFETY: The caller retains and validates all reachable selected slots.
        Ok(unsafe { self.bind_indirect_resources(table.0, owners, self.native_residency(set)?)? })
    }
    unsafe fn full_write_workspace(
        &self,
        slot: &mut Option<SerialWorkspace>,
        shape: &[i32],
        scalar: ScalarType,
        label: &str,
    ) -> anyhow::Result<Array> {
        let bytes = lisa_backend::tensor_bytes(shape, scalar)
            .ok_or_else(|| anyhow::anyhow!("{label}: workspace size overflow"))?;
        if let Some(w) = slot {
            anyhow::ensure!(
                w.runtime.ptr_eq(&Arc::downgrade(&self.runtime)),
                "foreign workspace context"
            );
        }
        if slot.as_ref().is_none_or(|w| w.buffer.length() < bytes) {
            *slot = Some(SerialWorkspace {
                runtime: Arc::downgrade(&self.runtime),
                buffer: self.runtime.buffer_for_full_write(bytes, label)?,
            });
        }
        let shape = shape.iter().map(|&n| n as usize).collect::<Vec<_>>();
        Ok(Array::new(crate::jit::Tensor::from_parts(
            &self.runtime,
            slot.as_ref().unwrap().buffer.clone(),
            &shape,
            dtype(scalar),
        )))
    }
}

impl std::ops::Deref for IndirectTable {
    type Target = Array;
    fn deref(&self) -> &Array {
        &self.0
    }
}
impl std::ops::Deref for Residency {
    type Target = crate::runtime::ResidencySet;
    fn deref(&self) -> &Self::Target {
        &self.set
    }
}
impl AsRef<crate::runtime::ResidencySet> for Residency {
    fn as_ref(&self) -> &crate::runtime::ResidencySet {
        &self.set
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn opaque_resources_and_snapshot_keep_payloads_alive() -> anyhow::Result<()> {
        let device = MetalDevice::system_default();
        let bytes: Vec<u8> = (0u32..16).flat_map(u32::to_ne_bytes).collect();
        let source = device.pull(
            HostTensor {
                bytes: &bytes,
                shape: &[16],
                dtype: ScalarType::U32,
            }
            .into(),
            Default::default(),
        )?;
        let snapshot = device.copy_tensor(&source)?;
        assert!(!device.same_storage(&source, &snapshot));
        let table = <MetalDevice as ResourceDevice>::indirect_table(&device, &[&source])?;
        let set = device.new_residency(1);
        if let Some(set) = &set {
            assert!(device.residency_add(set, &source));
            assert!(device.residency_contains(set, device.allocation_id(&source)));
            device.residency_commit(set);
        }
        // SAFETY: Every address in this table has a retained owner.
        let resources = unsafe { device.bind_indirect_table(table, vec![source], set)? };
        device.submit(true)?;
        assert_eq!(device.read_bytes(&snapshot)?, bytes);
        drop(resources);
        assert_eq!(device.read_bytes(&snapshot)?, bytes);
        Ok(())
    }
    #[test]
    fn scratch_growth_preserves_old_leases_and_reuses_capacity() -> anyhow::Result<()> {
        let device = MetalDevice::system_default();
        let mut slot = None;
        // SAFETY: These allocation-only checks never read or dispatch scratch.
        let first = unsafe {
            device.full_write_workspace(&mut slot, &[16], ScalarType::F32, "scratch-test")?
        };
        let first_id = device.allocation_id(&first);
        let grown = unsafe {
            device.full_write_workspace(&mut slot, &[8192], ScalarType::F32, "scratch-test")?
        };
        assert_ne!(first_id, device.allocation_id(&grown));
        let grown_id = device.allocation_id(&grown);
        drop(grown);
        let reused = unsafe {
            device.full_write_workspace(&mut slot, &[32], ScalarType::F32, "scratch-test")?
        };
        assert_eq!(grown_id, device.allocation_id(&reused));
        assert_eq!(first_id, device.allocation_id(&first));
        Ok(())
    }
}
