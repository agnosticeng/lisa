//! Metal admission binding for the portable GGUF mapped-weight importer.
use anyhow::{Result, ensure};
use lisa_gguf::Package;
use lisa_metal::{Stream, device::MetalDevice, runtime::ResidentMemoryReservation};
pub type PackedTensor = crate::weights::gguf::PackedTensor<MetalDevice>;
pub struct ResidentWeights {
    prepared: lisa_engine::weights::prepared::PreparedWeights<
        lisa_metal::Array,
        ResidentMemoryReservation,
    >,
    // Original descriptors are retained only for independent oracle fixtures.
    pub tensors: std::collections::BTreeMap<String, PackedTensor>,
}
impl std::ops::Deref for ResidentWeights {
    type Target = lisa_engine::weights::prepared::PreparedWeights<
        lisa_metal::Array,
        ResidentMemoryReservation,
    >;
    fn deref(&self) -> &Self::Target {
        &self.prepared
    }
}
impl ResidentWeights {
    pub fn load(package: &Package, names: &[&str], stream: &Stream) -> Result<Self> {
        let imported = crate::weights::gguf::ResidentWeights::pull(
            MetalDevice::on_stream(stream),
            package,
            names,
            |mapped_bytes| {
                let budget = lisa_engine::memory::ram_budget();
                ensure!(
                    mapped_bytes <= budget,
                    "resident GGUF weights need {} GiB, RAM budget is {} GiB",
                    mapped_bytes as f64 / (1u64 << 30) as f64,
                    budget as f64 / (1u64 << 30) as f64
                );
                Ok(stream
                    .runtime()
                    .reserve_model_memory(mapped_bytes, budget)?)
            },
        )?;
        let tensors = imported.tensors.clone();
        Ok(Self {
            prepared: imported.into_prepared()?,
            tensors,
        })
    }
}
