//! Process-wide diagnostics, configured once by the application before the
//! first Metal runtime starts. Everything is off by default; the production
//! path pays one relaxed load per span or dispatch.
use std::path::PathBuf;
use std::sync::OnceLock;

#[derive(Clone, Debug, Default)]
pub struct Diagnostics {
    /// Record hierarchical spans. `Some(prefixes)` keeps only spans whose name
    /// starts with one of the prefixes; an empty list keeps every span.
    pub trace: Option<Vec<String>>,
    /// Write the recorded spans as a Chrome trace to this path at exit.
    pub trace_json: Option<PathBuf>,
    /// Submit one dispatch per command buffer so each buffer's GPU interval
    /// can be attributed to a single kernel. Slow; diagnostic only.
    pub gpu_probe: bool,
    /// Append one JSON line per completed command buffer to this path.
    pub cb_record: Option<PathBuf>,
}

impl Diagnostics {
    /// Whether span recording is on, for the report or the Chrome trace.
    pub fn tracing(&self) -> bool {
        self.trace.is_some() || self.trace_json.is_some()
    }
}

static DIAGNOSTICS: OnceLock<Diagnostics> = OnceLock::new();

/// Install the process diagnostics. Fails if they were already read or set.
pub fn configure(diagnostics: Diagnostics) -> Result<(), Diagnostics> {
    DIAGNOSTICS.set(diagnostics)
}

/// The installed diagnostics, or the all-off default.
pub fn get() -> &'static Diagnostics {
    DIAGNOSTICS.get_or_init(Diagnostics::default)
}
