use std::env;
use std::fs;
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

fn sources(dir: &Path, paths: &mut Vec<PathBuf>) {
    for entry in fs::read_dir(dir).expect("read shader directory") {
        let path = entry.expect("shader directory entry").path();
        if path.is_dir() {
            sources(&path, paths);
        } else if matches!(
            path.extension().and_then(|s| s.to_str()),
            Some("metal" | "h")
        ) {
            paths.push(path);
        }
    }
}

fn write_changed(path: &Path, bytes: &[u8]) {
    if fs::read(path).ok().as_deref() != Some(bytes) {
        let temporary = path.with_extension(format!("{}.tmp", std::process::id()));
        fs::write(&temporary, bytes).expect("write kernel package");
        fs::rename(temporary, path).expect("publish kernel package");
    }
}

fn main() {
    let root = Path::new("src/shaders");
    println!("cargo:rerun-if-changed={}", root.display());
    let mut paths = Vec::new();
    sources(root, &mut paths);
    paths.sort();
    let mut source = Vec::new();
    let mut index = String::from("macro_rules! kernel_source {\n");
    for path in paths {
        let bytes = fs::read(&path).expect("read shader source");
        std::str::from_utf8(&bytes).expect("UTF-8 shader source");
        let name = path.strip_prefix(root).unwrap().to_str().unwrap();
        index.push_str(&format!(
            "    ({name:?}) => {{{{ static SOURCE: crate::shader_pack::Source = crate::shader_pack::Source::new({}, {}); &SOURCE }}}};\n",
            source.len(), bytes.len(),
        ));
        source.extend_from_slice(&bytes);
    }
    index.push_str("}\npub(crate) use kernel_source;\n");
    let digest: [u8; 32] = Sha256::digest(&source).into();
    let mut packed = b"LISAKR01".to_vec();
    packed.extend_from_slice(&(source.len() as u64).to_le_bytes());
    packed.extend_from_slice(&miniz_oxide::deflate::compress_to_vec_zlib(&source, 9));
    index.push_str(&format!(
        "pub(crate) const KERNEL_BYTES: usize = {};\n\
         pub(crate) const PACK_BYTES: usize = {};\n\
         pub(crate) const KERNEL_SHA256: [u8; 32] = {digest:?};\n",
        source.len(),
        packed.len(),
    ));
    let out = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR"));
    write_changed(&out.join("kernel_index.rs"), index.as_bytes());
    // OUT_DIR is <profile>/build/<package>/out, including custom target dirs.
    let profile = out.ancestors().nth(3).expect("Cargo profile directory");
    // Regenerate a removed sidecar even when Rust sources have not changed.
    println!(
        "cargo:rerun-if-changed={}",
        profile.join("kernels.bin").display()
    );
    write_changed(&profile.join("kernels.bin"), &packed);
}
