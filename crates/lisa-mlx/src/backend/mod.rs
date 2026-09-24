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
}

impl Backend {
    /// Resolve a backend by name (`metal`/`gpu`, `cpu`, …).
    pub fn from_name(name: &str) -> Result<Self> {
        match name.to_ascii_lowercase().as_str() {
            "metal" | "gpu" => Ok(Backend::Metal),
            "cpu" => Err(Error::Msg(
                "the cpu backend is not implemented yet".to_string(),
            )),
            other => Err(Error::Msg(format!(
                "unknown device {other:?}; available: metal"
            ))),
        }
    }

    /// The backend selected by `LISA_DEVICE` (default `metal`).
    pub fn from_env() -> Result<Self> {
        let name = std::env::var("LISA_DEVICE").unwrap_or_else(|_| "metal".to_string());
        Self::from_name(&name)
    }

    /// The backend's canonical name.
    pub fn name(self) -> &'static str {
        match self {
            Backend::Metal => "metal",
        }
    }
}