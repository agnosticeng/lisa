//! Engine-selected residency policy; devices only implement tensor imports.
use lisa_backend::ImportMode;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ResidencyPolicy {
    /// Immutable mapped views. Unsupported sharing is an error, never a copy.
    #[default]
    Mapped,
    /// Upload unchanged bytes to private device allocations.
    Uploaded,
}
impl ResidencyPolicy {
    pub fn import_mode(self) -> ImportMode {
        match self {
            Self::Mapped => ImportMode::SharedRequired,
            Self::Uploaded => ImportMode::Copy,
        }
    }
}

/// Separate address-space costs for admission before importing any weights.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ResidencyBytes {
    pub host_resident: usize,
    pub device_resident: usize,
    /// Maximum temporary host mapping while sequentially uploading weights.
    pub host_staging: usize,
}
