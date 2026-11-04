use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
#[cfg(target_os = "macos")]
use std::os::fd::AsRawFd;
use std::os::unix::fs::FileExt;
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result, ensure};
use memmap2::{Mmap, MmapOptions};

use super::{Header, TensorInfo};

const MAX_READ_BYTES: usize = 64 * 1024 * 1024;

/// A validated file range. Keeping the open file avoids reopening a replaced
/// path when reading disk-only Engram rows later in a session.
#[derive(Clone, Debug)]
pub struct Tensor {
    pub info: TensorInfo,
    file: Arc<File>,
    pub path: PathBuf,
}

impl Tensor {
    /// Page-aligned mapping size, including the tensor's offset within its
    /// first page and padding in its last page. Use this for memory admission
    /// before calling [`Self::mapped_region`].
    pub fn mapped_region_bytes(&self) -> Result<usize> {
        let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as u64;
        ensure!(page.is_power_of_two(), "invalid host page size");
        let inner = self.info.offset % page;
        let length = inner
            .checked_add(self.info.bytes)
            .context("GGUF mapping size overflow")?;
        usize::try_from(
            length
                .checked_add(page - 1)
                .context("GGUF mapping rounding overflow")?
                & !(page - 1),
        )
        .context("GGUF mapping does not fit address space")
    }

    /// Map the pages containing this tensor and return its byte offset within
    /// the mapping. Tensor views must stay within `info.bytes` from that offset;
    /// padding after the file's end must never be read. Keep checkpoint files
    /// immutable while any mapping exists, and retain the mapping for all users.
    pub fn mapped_region(&self) -> Result<(Arc<Mmap>, usize)> {
        let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as u64;
        let length = self.mapped_region_bytes()?;
        let inner = self.info.offset % page;
        ensure!(
            self.file.metadata()?.len() >= self.info.offset + self.info.bytes,
            "GGUF shard was truncated: {}",
            self.path.display()
        );
        // mmap extends only to the end of the file's last page; tensor views
        // are bounded to real payload bytes and never read that padding.
        let map = unsafe {
            MmapOptions::new()
                .offset(self.info.offset - inner)
                .len(length)
                .map(&*self.file)?
        };
        Ok((Arc::new(map), inner as usize))
    }

    /// Independent descriptor for random disk lookups. Do not apply these
    /// flags to the descriptor used by resident weight mappings.
    pub fn disk_reader(&self) -> Result<Self> {
        let file = File::open(&self.path)?;
        let original = self.file.metadata()?;
        let current = file.metadata()?;
        ensure!(
            original.dev() == current.dev()
                && original.ino() == current.ino()
                && current.len() == original.len(),
            "GGUF file changed before opening disk reader"
        );
        #[cfg(target_os = "macos")]
        for (command, value) in [(libc::F_NOCACHE, 1), (libc::F_RDAHEAD, 0)] {
            if unsafe { libc::fcntl(file.as_raw_fd(), command, value) } == -1 {
                return Err(std::io::Error::last_os_error().into());
            }
        }
        Ok(Self {
            info: self.info.clone(),
            file: Arc::new(file),
            path: self.path.clone(),
        })
    }

    /// Map only this tensor. Callers must keep checkpoint files immutable while
    /// mappings exist; the same constraint applies to safetensors mappings.
    pub fn map(&self) -> Result<Arc<Mmap>> {
        let length = usize::try_from(self.info.bytes)?;
        ensure!(
            self.file.metadata()?.len() >= self.info.offset + self.info.bytes,
            "GGUF shard was truncated: {}",
            self.path.display()
        );
        let map = unsafe {
            MmapOptions::new()
                .offset(self.info.offset)
                .len(length)
                .map(&*self.file)?
        };
        Ok(Arc::new(map))
    }

    pub fn read_bytes(&self, offset: u64, output: &mut [u8]) -> Result<()> {
        ensure!(
            output.len() <= MAX_READ_BYTES,
            "GGUF read exceeds 64 MiB staging limit"
        );
        let end = offset
            .checked_add(output.len() as u64)
            .context("GGUF read overflow")?;
        ensure!(
            end <= self.info.bytes,
            "read outside GGUF tensor {}",
            self.info.name
        );
        self.file
            .read_exact_at(output, self.info.offset + offset)
            .with_context(|| format!("reading {} in {}", self.info.name, self.path.display()))
    }

    /// Read the requested rows in caller order. This never reads the full
    /// Engram table or admits it into the resident Metal weight allocation.
    pub fn read_rows(&self, rows: &[u64], output: &mut [u8]) -> Result<()> {
        let bytes = usize::try_from(self.info.row_bytes()?)?;
        let total = rows
            .len()
            .checked_mul(bytes)
            .context("GGUF row read size overflow")?;
        ensure!(
            total == output.len() && total <= MAX_READ_BYTES,
            "invalid or oversized GGUF row output buffer"
        );
        ensure!(
            rows.iter().all(|&row| row < self.info.rows()),
            "GGUF row index out of range"
        );
        for (&row, out) in rows.iter().zip(output.chunks_exact_mut(bytes)) {
            self.read_bytes(row * bytes as u64, out)?;
        }
        Ok(())
    }
}

#[derive(Debug)]
pub struct Package {
    pub header: Header,
    pub shards: Vec<PathBuf>,
    pub tensors: BTreeMap<String, Tensor>,
}

impl Package {
    pub fn open(first: &Path) -> Result<Self> {
        let header = Header::open(first)?;
        let count = match header.metadata.get("split.count") {
            Some(v) => v.as_u64().context("invalid split.count")?,
            None => 1,
        };
        ensure!(
            (1..=1024).contains(&count),
            "invalid GGUF shard count {count}"
        );
        let mut paths = vec![first.to_owned()];
        if count > 1 {
            ensure!(
                header.integer("split.no")? == 0,
                "pass the first GGUF shard (split.no=0)"
            );
            let name = first
                .file_name()
                .and_then(|s| s.to_str())
                .context("invalid GGUF filename")?;
            let suffix = format!("-00001-of-{count:05}.gguf");
            let stem = name
                .strip_suffix(&suffix)
                .context("split GGUF filename does not match metadata")?;
            let parent = first.parent().unwrap_or(Path::new("."));
            for part in 2..=count {
                let expected = parent.join(format!("{stem}-{part:05}-of-{count:05}.gguf"));
                if expected.is_file() {
                    paths.push(expected);
                    continue;
                }
                // ISTA publishes the shared Engram file under its original
                // name, outside IQ2_XXS/. Accept that exact package layout;
                // never guess among unrelated GGUF weights in the directory.
                if part == count && header.architecture()? == "deepseek41" {
                    let engram_name = "DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf";
                    let mut candidates = BTreeSet::new();
                    for directory in [Some(parent), parent.parent()].into_iter().flatten() {
                        let path = directory.join(engram_name);
                        if path.is_file() {
                            candidates.insert(path.canonicalize()?);
                        }
                    }
                    if candidates.len() == 1 {
                        paths.push(candidates.into_iter().next().unwrap());
                        continue;
                    }
                }
                anyhow::bail!("missing GGUF shard {}", expected.display());
            }
        }
        Self::open_paths(paths)
    }

    /// Explicit shard list for callers with a pinned download manifest.
    pub fn open_paths(paths: Vec<PathBuf>) -> Result<Self> {
        ensure!(!paths.is_empty(), "GGUF package has no shards");
        let mut first = None;
        let mut tensors = BTreeMap::new();
        let mut seen_files = BTreeSet::new();
        for (part, path) in paths.iter().enumerate() {
            ensure!(
                seen_files.insert(path.canonicalize()?),
                "duplicate GGUF shard path"
            );
            let file = Arc::new(File::open(path)?);
            let stat = file.metadata()?;
            ensure!(stat.is_file(), "GGUF shard is not a regular file");
            let mut reader = std::io::BufReader::new(file.try_clone()?);
            let header = Header::read(&mut reader, stat.len())?;
            if paths.len() > 1 || header.metadata.contains_key("split.count") {
                ensure!(
                    header.integer("split.count")? == paths.len() as u64,
                    "GGUF shard count mismatch in {}",
                    path.display()
                );
                ensure!(
                    header.integer("split.no")? == part as u64,
                    "GGUF shard number mismatch in {}",
                    path.display()
                );
            }
            if let Some(first) = &first {
                let first: &Header = first;
                ensure!(
                    header.integer("split.tensors.count")?
                        == first.integer("split.tensors.count")?,
                    "GGUF total tensor count differs between shards"
                );
                for (key, value) in &header.metadata {
                    if key.starts_with("split.") || key.starts_with("general.") {
                        continue;
                    }
                    if let Some(original) = first.metadata.get(key) {
                        ensure!(original == value, "conflicting GGUF metadata {key}");
                    }
                }
                if let Some(kind) = header.metadata.get("general.architecture") {
                    ensure!(
                        kind.as_str() == Some(first.architecture()?),
                        "conflicting GGUF architectures"
                    );
                }
            }
            for (name, info) in &header.tensors {
                let tensor = Tensor {
                    info: info.clone(),
                    file: file.clone(),
                    path: path.clone(),
                };
                ensure!(
                    tensors.insert(name.clone(), tensor).is_none(),
                    "duplicate GGUF tensor {name} across shards"
                );
            }
            if first.is_none() {
                first = Some(header);
            }
        }
        let header = first.context("missing first GGUF shard")?;
        if paths.len() > 1 {
            ensure!(
                header.integer("split.tensors.count")? == tensors.len() as u64,
                "GGUF package tensor count mismatch"
            );
        }
        Ok(Self {
            header,
            shards: paths,
            tensors,
        })
    }

    pub fn tensor(&self, name: &str) -> Result<&Tensor> {
        self.tensors
            .get(name)
            .with_context(|| format!("missing GGUF tensor {name}"))
    }

    pub fn summary(&self) -> Result<String> {
        let mut kinds = BTreeMap::new();
        let mut bytes = 0u64;
        for t in self.tensors.values() {
            let entry = kinds
                .entry(t.info.kind.layout()?.0)
                .or_insert((0usize, 0u64));
            entry.0 += 1;
            entry.1 = entry
                .1
                .checked_add(t.info.bytes)
                .context("GGUF size overflow")?;
            bytes = bytes
                .checked_add(t.info.bytes)
                .context("GGUF size overflow")?;
        }
        let mut out = format!(
            "GGUF v3 · {} · {} shards · {} tensors · {:.2} GB payload\n",
            self.header.architecture()?,
            self.shards.len(),
            self.tensors.len(),
            bytes as f64 / 1e9
        );
        for (name, (count, bytes)) in kinds {
            out.push_str(&format!(
                "  {name:8} {count:5} tensors  {:8.3} GB\n",
                bytes as f64 / 1e9
            ));
        }
        Ok(out)
    }
}
