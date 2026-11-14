//! Loaded attention arithmetic and scoped diagnostic overrides.
thread_local! {
    pub static GLM53_DSA_NATIVE_OVERRIDE: std::cell::Cell<bool> = const { std::cell::Cell::new(true) };
    pub static GLM53_DSA_VALUE_DENSE_PREFIX: std::cell::Cell<Option<usize>> = const { std::cell::Cell::new(None) };
}

/// DS4 ends an attention/value slice at the dense-to-sparse cache boundary.
/// Preserve the resulting value-projection row count when selecting its
/// scalar (<32 rows) or matrix schedule.
pub struct Glm53DsaValueSliceScope {
    previous: Option<usize>,
}

impl Glm53DsaValueSliceScope {
    pub fn new(query_start: usize, dense_limit: usize) -> Self {
        let prefix = dense_limit.saturating_sub(query_start);
        let previous = GLM53_DSA_VALUE_DENSE_PREFIX.with(|value| value.replace(Some(prefix)));
        Self { previous }
    }
}

impl Drop for Glm53DsaValueSliceScope {
    fn drop(&mut self) {
        GLM53_DSA_VALUE_DENSE_PREFIX.with(|value| value.set(self.previous));
    }
}

/// Scoped override for DS4's native QK, value and indexed-decode schedules.
/// Production uses the native schedules; diagnostics may select the legacy path.
pub struct Glm53DsaNativeScope {
    previous: bool,
}

impl Glm53DsaNativeScope {
    pub fn new(enabled: bool) -> Self {
        let previous = GLM53_DSA_NATIVE_OVERRIDE.with(|value| value.replace(enabled));
        Self { previous }
    }
}

impl Drop for Glm53DsaNativeScope {
    fn drop(&mut self) {
        GLM53_DSA_NATIVE_OVERRIDE.with(|value| value.set(self.previous));
    }
}

/// Arithmetic mode used by GLM's explicit prefix/session cache identity.
pub fn glm53_dsa_native_enabled() -> bool {
    GLM53_DSA_NATIVE_OVERRIDE.with(std::cell::Cell::get)
}
