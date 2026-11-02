//! External data for explicitly enabled GLM model integration tests.

/// Load a captured oracle only after its integration test has been enabled.
/// Missing data is an error, never a reason to skip an enabled test.
#[track_caller]
pub(super) fn read(name: &str) -> Vec<u8> {
    let root = std::env::var_os("LISA_GLM53_FIXTURE_DIR")
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| {
            panic!(
                "GLM integration fixture {name} requires LISA_GLM53_FIXTURE_DIR; \
                 set it to the directory containing the captured fixture files listed in \
                 crates/lisa-engine/src/models/glm53/fixtures/integration-fixtures.json"
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
