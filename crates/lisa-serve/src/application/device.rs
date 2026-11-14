//! Application-facing device selection policy, independent of weight format.
//! Concrete runtimes report availability; they do not read `LISA_DEVICE` or
//! decide which other backend a request should use. Only implemented model
//! paths are admitted by the loader.

use anyhow::{Context, Result, bail};

#[cfg(feature = "metal")]
fn metal_available() -> bool {
    lisa_metal::device::metal_available()
}
#[cfg(not(feature = "metal"))]
fn metal_available() -> bool {
    false
}

/// The available device backends.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Backend {
    /// Apple-Silicon GPU via Metal.
    Metal,
    /// Host CPU (only models with a host forward path, e.g. Laya).
    Cpu,
}

impl Backend {
    /// Construct a model-unaware device handle at the application boundary.
    pub fn open(self) -> Result<lisa_backend::erased::DeviceHandle> {
        use lisa_backend::erased::DeviceHandle;
        match self {
            Self::Cpu => Ok(DeviceHandle::new(lisa_cpu::CpuDevice)),
            #[cfg(feature = "metal")]
            Self::Metal => Ok(DeviceHandle::with_indirect(
                lisa_metal::device::MetalDevice::system_default(),
            )),
            #[cfg(not(feature = "metal"))]
            Self::Metal => bail!("Metal support is disabled in this build"),
        }
    }

    /// The model chooses arithmetic explicitly; device selection cannot silently
    /// substitute the host reduction law for the native tensor law.
    pub fn laya_arithmetic(self) -> lisa_models::laya::plan::Arithmetic {
        match self {
            Self::Cpu => lisa_models::laya::plan::Arithmetic::HostF32,
            Self::Metal => lisa_models::laya::plan::Arithmetic::Native,
        }
    }

    /// Parse an implemented backend. Device zero is the only addressable device
    /// in the current Metal runtime; reject other ordinals rather than silently
    /// running on the system default. `gpu` remains a legacy Metal alias.
    pub fn from_name(name: &str) -> Result<Self> {
        let request: lisa_engine::device::DeviceRequest = name.parse()?;
        let provider = request.provider.as_str();
        let ordinal = request.index;
        let backend = match provider {
            "metal" | "gpu" => {
                anyhow::ensure!(
                    cfg!(feature = "metal"),
                    "Metal support is disabled in this build"
                );
                Self::Metal
            }
            "cpu" => Self::Cpu,
            "rocm" | "hip" => bail!(
                "ROCm/HIP execution is not implemented in this build; portable checkpoint readers do not provide a compute backend"
            ),
            other => bail!("unknown device {other:?}; available: metal, cpu"),
        };
        anyhow::ensure!(
            ordinal == 0,
            "{} device index {ordinal} is not supported; this build only addresses device 0",
            backend.name()
        );
        Ok(backend)
    }

    /// Resolve a supplied device, otherwise auto-select from runtime discovery.
    /// Model support is validated separately before importing weights.
    pub fn resolve(explicit: Option<&str>) -> Result<Self> {
        match explicit {
            Some(name) if !name.eq_ignore_ascii_case("auto") => Self::from_name(name),
            _ => Ok(if metal_available() {
                Self::Metal
            } else {
                Self::Cpu
            }),
        }
    }

    /// Explicit API/CLI choices take precedence over the environment. Only the
    /// application boundary should call this; resolved loaders never reread it.
    pub fn from_options(explicit: Option<&str>) -> Result<Self> {
        if explicit.is_some() {
            return Self::resolve(explicit);
        }
        Self::from_env()
    }

    pub fn from_env() -> Result<Self> {
        let name = match std::env::var("LISA_DEVICE") {
            Ok(name) => Some(name),
            Err(std::env::VarError::NotPresent) => None,
            Err(error) => return Err(error).context("LISA_DEVICE must be UTF-8"),
        };
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn named_devices_reject_unimplemented_providers_and_ordinals() {
        #[cfg(feature = "metal")]
        assert_eq!(Backend::from_name("METAL:0").unwrap(), Backend::Metal);
        assert_eq!(Backend::from_name("cpu:0").unwrap(), Backend::Cpu);
        #[cfg(feature = "metal")]
        assert_eq!(Backend::from_name("gpu").unwrap(), Backend::Metal);
        for name in ["metal:1", "cpu:1", "metal:-1", "metal:0:1", "cpu:"] {
            assert!(Backend::from_name(name).is_err(), "accepted {name}");
        }
        assert!(
            Backend::from_name("rocm:0")
                .unwrap_err()
                .to_string()
                .contains("not implemented")
        );
    }
    #[test]
    #[cfg(not(feature = "metal"))]
    fn portable_build_selects_cpu_and_rejects_metal_before_loading() {
        assert_eq!(Backend::resolve(None).unwrap(), Backend::Cpu);
        assert_eq!(Backend::resolve(Some("auto")).unwrap(), Backend::Cpu);
        for name in ["metal", "METAL:0", "gpu"] {
            assert!(
                Backend::from_name(name)
                    .unwrap_err()
                    .to_string()
                    .contains("disabled")
            );
        }
        let result = Backend::Metal.open();
        assert!(result.err().unwrap().to_string().contains("disabled"));
    }
}
