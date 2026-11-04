//! Lossless Metal source package. Loaded and authenticated once, before JIT use.

use std::fmt;
use std::io::Read;
use std::ops::Deref;
use std::path::PathBuf;
use std::sync::OnceLock;

use sha2::{Digest, Sha256};

use crate::error::{Error, Result};

static SOURCES: OnceLock<std::result::Result<Box<str>, String>> = OnceLock::new();

fn locations() -> std::result::Result<Vec<PathBuf>, String> {
    let executable = std::env::current_exe().map_err(|e| e.to_string())?;
    let directory = executable.parent().ok_or("executable has no directory")?;
    let mut paths = vec![directory.join("kernels.bin")];
    if directory.file_name().is_some_and(|name| name == "MacOS") {
        paths.insert(0, directory.join("../Resources/kernels.bin"));
    } else if directory.file_name().is_some_and(|name| name == "deps") {
        paths.push(directory.join("../kernels.bin"));
    }
    Ok(paths)
}

fn load() -> std::result::Result<Box<str>, String> {
    let paths = locations()?;
    let path = paths.iter().find(|p| p.is_file()).ok_or_else(|| {
        "kernels.bin is missing; rebuild the app or copy the matching kernel package beside the executable".to_string()
    })?;
    let file = std::fs::File::open(path).map_err(|e| format!("open kernels.bin: {e}"))?;
    if file.metadata().map_err(|e| e.to_string())?.len() != crate::PACK_BYTES as u64 {
        return Err(
            "kernels.bin does not match this executable; rebuild or reinstall the complete app"
                .into(),
        );
    }
    let mut packed = Vec::with_capacity(crate::PACK_BYTES);
    file.take(crate::PACK_BYTES as u64 + 1)
        .read_to_end(&mut packed)
        .map_err(|e| format!("read kernels.bin: {e}"))?;
    if packed.len() != crate::PACK_BYTES
        || packed.get(..8) != Some(b"LISAKR01")
        || packed.get(8..16) != Some((crate::KERNEL_BYTES as u64).to_le_bytes().as_slice())
    {
        return Err("invalid kernel package header".into());
    }
    let source =
        miniz_oxide::inflate::decompress_to_vec_zlib_with_limit(&packed[16..], crate::KERNEL_BYTES)
            .map_err(|_| "invalid compressed kernel package".to_string())?;
    if source.len() != crate::KERNEL_BYTES
        || <[u8; 32]>::from(Sha256::digest(&source)) != crate::KERNEL_SHA256
    {
        return Err(
            "kernel package checksum mismatch; rebuild or reinstall the complete app".into(),
        );
    }
    String::from_utf8(source)
        .map(String::into_boxed_str)
        .map_err(|_| "kernel package is not UTF-8".into())
}

/// Load the matching kernel package without creating a Metal device.
pub fn initialize() -> Result<()> {
    SOURCES
        .get_or_init(load)
        .as_ref()
        .map(|_| ())
        .map_err(|e| Error::msg(e.clone()))
}

/// A compile-time byte range in the authenticated, immutable source package.
/// Source access never reads files, inflates data or hashes bytes after initialization.
#[derive(Debug)]
pub struct Source {
    offset: usize,
    length: usize,
}

impl Source {
    pub(crate) const fn new(offset: usize, length: usize) -> Self {
        Self { offset, length }
    }

    pub fn as_str(&self) -> &'static str {
        let source = SOURCES
            .get_or_init(load)
            .as_ref()
            .unwrap_or_else(|e| panic!("cannot load Metal kernels: {e}"));
        &source[self.offset..self.offset + self.length]
    }
}

impl Deref for Source {
    type Target = str;
    fn deref(&self) -> &str {
        self.as_str()
    }
}

impl fmt::Display for Source {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}
