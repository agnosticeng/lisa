// Shared by build.rs (through include!) and the crate's provenance test, so
// both see exactly the packed text.

/// Replace each `//@include <path>` line with the named fragment, verbatim.
/// Fragments hold runs of source shared by several kernels; the packed text
/// is exactly the text the kernel would have without the indirection.
fn expand(root: &Path, source: &[u8]) -> Vec<u8> {
    const MARKER: &[u8] = b"//@include ";
    let mut out = Vec::with_capacity(source.len());
    for line in source.split_inclusive(|&byte| byte == b'\n') {
        if let Some(name) = line.strip_prefix(MARKER) {
            let name = std::str::from_utf8(name).expect("UTF-8 include").trim_end();
            assert!(
                name.ends_with(".inc") && !name.contains(".."),
                "invalid shader include {name:?}"
            );
            let fragment = std::fs::read(root.join(name)).expect("read shader fragment");
            assert!(
                !fragment.windows(MARKER.len()).any(|w| w == MARKER),
                "nested shader include in {name}"
            );
            out.extend_from_slice(&fragment);
        } else {
            out.extend_from_slice(line);
        }
    }
    out
}
