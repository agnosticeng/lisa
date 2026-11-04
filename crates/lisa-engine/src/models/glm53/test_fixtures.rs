//! External data for explicitly enabled GLM model integration tests.

/// Load a captured oracle only after its integration test has been enabled.
/// Missing data is an error, never a reason to skip an enabled test.
#[track_caller]
pub(crate) fn read(name: &str) -> Vec<u8> {
    let root = std::env::var_os("LISA_GLM53_FIXTURE_DIR")
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| {
            panic!(
                "GLM integration fixture {name} requires LISA_GLM53_FIXTURE_DIR; \
                 set it to the external capture directory containing {name}; \
                 see specs/models/glm53.md for validation scope"
            )
        });
    let path = std::path::Path::new(&root).join(name);
    std::fs::read(&path).unwrap_or_else(|error| {
        panic!(
            "cannot read required GLM integration fixture {}: {error}; \
             restore the captured file or set LISA_GLM53_FIXTURE_DIR to its directory",
            path.display()
        )
    })
}
