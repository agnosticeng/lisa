//! Device backends.
//!
//! The tensor/op surface the engine uses is backend-agnostic in intent: a
//! backend supplies `Array`/`Stream` and the kernels behind them. Today the
//! only backend is [`metal`]; a CPU backend would be a sibling module and a
//! [`Backend`] variant.
//!
//! [`Backend::from_env`] resolves the backend for a run (`LISA_DEVICE`,
//! default `metal`) so callers fail early and clearly when a backend is not
//! built in.

pub mod metal;

use crate::error::{Error, Result};

/// The available device backends.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Backend {
    /// Apple-Silicon GPU via Metal.
    Metal,
    /// Host CPU (only models with a host forward path, e.g. Laya).
    Cpu,
}

impl Backend {
    /// Resolve a backend by name (`metal`/`gpu`, `cpu`, …).
    pub fn from_name(name: &str) -> Result<Self> {
        match name.to_ascii_lowercase().as_str() {
            "metal" | "gpu" => Ok(Backend::Metal),
            "cpu" => Ok(Backend::Cpu),
            other => Err(Error::Msg(format!(
                "unknown device {other:?}; available: metal, cpu"
            ))),
        }
    }

    /// Resolve an explicit name, or auto-select when none is given: Metal if a
    /// Metal device exists, else CPU.
    pub fn resolve(explicit: Option<&str>) -> Result<Self> {
        match explicit {
            Some(name) => Self::from_name(name),
            None => Ok(if metal_available() { Backend::Metal } else { Backend::Cpu }),
        }
    }

    /// The backend selected by `LISA_DEVICE`, else auto (Metal if available,
    /// otherwise CPU).
    pub fn from_env() -> Result<Self> {
        let name = std::env::var("LISA_DEVICE").ok();
        Self::resolve(name.as_deref())
    }

    /// The backend's canonical name.
    pub fn name(self) -> &'static str {
        match self {
            Backend::Metal => "metal",
            Backend::Cpu => "cpu",
        }
    }
}

/// Whether a Metal device is present. `MTLCreateSystemDefaultDevice` returns
/// `None` when there is no GPU (or Metal is unavailable), in which case
/// [`Backend::resolve`] falls back to CPU.
pub fn metal_available() -> bool {
    objc2_metal::MTLCreateSystemDefaultDevice().is_some()
}