//! Metal admission binding for the portable GGUF mapped-weight importer.
use anyhow::{Result, ensure};
use lisa_gguf::Package;
use lisa_metal::{Stream, device::MetalDevice, runtime::ResidentMemoryReservation};
pub type PackedTensor = crate::weights::gguf::PackedTensor<MetalDevice>;
pub struct ResidentWeights(
    crate::weights::gguf::ResidentWeights<MetalDevice, ResidentMemoryReservation>,
);
impl std::ops::Deref for ResidentWeights {
    type Target = crate::weights::gguf::ResidentWeights<MetalDevice, ResidentMemoryReservation>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl ResidentWeights {
    pub fn load(package: &Package, names: &[&str], stream: &Stream) -> Result<Self> {
        Ok(Self(crate::weights::gguf::ResidentWeights::pull(
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
        )?))
    }
}
