//! Immutable, validated file-backed tensor views for resident model loading.
//!
//! The preparation tool verifies original payload bytes once. Load checks the
//! recorded file identities and tensor bounds before wrapping a bounded number
//! of shared Metal views; it never repairs alignment by copying a tensor.

use std::collections::{HashMap, HashSet};
use std::ffi::c_void;
use std::os::unix::fs::MetadataExt;
use std::path::{Component, Path, PathBuf};
use std::ptr::NonNull;
use std::sync::Arc;

use anyhow::{Context, ensure};
use lisa_mlx::{Array, Dtype};
use memmap2::Mmap;
use serde::{Deserialize, Serialize};

#[derive(Clone, PartialEq, Eq, Deserialize, Serialize)]
pub(crate) struct FileIdentity {
    pub(crate) path: PathBuf,
    pub(crate) size: u64,
    pub(crate) mtime_ns: i128,
    pub(crate) inode: u64,
    pub(crate) device: u64,
}

impl FileIdentity {
    fn open_checked(&self, root: &Path) -> anyhow::Result<std::fs::File> {
        ensure!(
            !self.path.as_os_str().is_empty()
                && self
                    .path
                    .components()
                    .all(|c| matches!(c, Component::Normal(_))),
            "resident weight identity must be relative to its model/cache directory: {}",
            self.path.display()
        );
        let path = root.join(&self.path);
        let file = std::fs::File::open(&path)
            .with_context(|| format!("resident weight file {}", path.display()))?;
        // Validate the descriptor that will actually be mapped, so replacing a
        // path between a metadata query and open cannot select a different file.
        let meta = file.metadata()?;
        let mtime_ns = i128::from(meta.mtime()) * 1_000_000_000 + i128::from(meta.mtime_nsec());
        ensure!(
            meta.is_file()
                && meta.len() == self.size
                && mtime_ns == self.mtime_ns
                && meta.ino() == self.inode
                && meta.dev() == self.device,
            "resident weight file changed: {}; rebuild its validated resident index",
            path.display()
        );
        Ok(file)
    }
}

#[derive(Deserialize, Serialize)]
pub(crate) struct ArenaIndex {
    pub(crate) format: String,
    pub(crate) source_files: Vec<FileIdentity>,
    pub(crate) backing_files: Vec<FileIdentity>,
    pub(crate) tensors: HashMap<String, TensorRecord>,
}

#[derive(Deserialize, Serialize)]
pub(crate) struct TensorRecord {
    pub(crate) file: usize,
    pub(crate) offset: usize,
    pub(crate) nbytes: usize,
    pub(crate) dtype: String,
    pub(crate) shape: Vec<i32>,
    pub(crate) sha256: String,
}

struct TensorView {
    buffer: Arc<lisa_mlx::runtime::Buffer>,
    offset: usize,
    nbytes: usize,
    dtype: Dtype,
    shape: Vec<i32>,
}

pub(crate) struct ResidentWeights {
    tensors: HashMap<String, TensorView>,
    buffers: Vec<Arc<lisa_mlx::runtime::Buffer>>,
    mapped_bytes: usize,
}

impl ResidentWeights {
    pub(crate) fn open(
        model_dir: &Path,
        index_path: &Path,
        expected_sources: &HashSet<PathBuf>,
    ) -> anyhow::Result<Self> {
        let index_bytes = super::loader::read_index_bytes(index_path).with_context(|| {
            format!(
                "cannot read resident weight index {}; run `lisa prepare --model '{}'` to create it",
                index_path.display(), model_dir.display()
            )
        })?;
        let index: ArenaIndex = serde_json::from_slice(&index_bytes)?;
        ensure!(
            index.format == "lisa-resident-weights-v1",
            "unsupported resident weight index format"
        );
        ensure!(!index.tensors.is_empty(), "resident weight index is empty");
        let actual_sources: HashSet<_> =
            index.source_files.iter().map(|f| f.path.clone()).collect();
        ensure!(
            actual_sources.len() == index.source_files.len() && &actual_sources == expected_sources,
            "resident source file inventory differs from the current checkpoint"
        );
        for identity in &index.source_files {
            identity.open_checked(model_dir)?;
        }
        let index_dir = index_path
            .parent()
            .context("resident index has no parent")?;
        let page_size = unsafe { getpagesize() };
        ensure!(page_size > 0, "invalid OS page size");
        let page = page_size as usize;
        let max_buffer = Array::max_buffer_length() / page * page;
        let mut buffers = Vec::new();
        let mut file_views = Vec::new();
        let mut mapped_bytes = 0usize;
        let mut backing_identities = HashSet::new();

        for (file_index, identity) in index.backing_files.iter().enumerate() {
            ensure!(
                backing_identities.insert((identity.device, identity.inode)),
                "resident index contains duplicate backing storage"
            );
            let file = identity.open_checked(index_dir)?;
            let map = Arc::new(unsafe { Mmap::map(&file)? });
            ensure!(
                !map.is_empty(),
                "empty resident backing file {}",
                identity.path.display()
            );
            let largest = index
                .tensors
                .values()
                .filter(|t| t.file == file_index)
                .map(|t| t.nbytes)
                .max()
                .context("unused resident backing file")?;
            let overlap = largest
                .checked_add(page - 1)
                .context("tensor span overflow")?
                / page
                * page;
            let overlap = overlap.checked_add(page).context("view overlap overflow")?;
            ensure!(
                max_buffer > overlap,
                "largest resident tensor exceeds Metal buffer capacity"
            );
            let step = max_buffer - overlap;
            let rounded_len = map
                .len()
                .checked_add(page - 1)
                .context("mapping length overflow")?
                / page
                * page;
            mapped_bytes = mapped_bytes
                .checked_add(rounded_len)
                .context("resident byte count overflow")?;
            let mut views = Vec::new();
            let mut start = 0;
            while start < map.len() {
                let length = max_buffer.min(rounded_len - start);
                let pointer =
                    unsafe { NonNull::new_unchecked(map.as_ptr().add(start) as *mut c_void) };
                let owner: Arc<dyn std::any::Any + Send + Sync> = map.clone();
                let buffer =
                    Array::buffer_from_external(pointer, length, owner, "resident_model_view")?;
                views.push((start, length, buffer.clone()));
                buffers.push(buffer);
                if start + length >= map.len() {
                    break;
                }
                start = start.checked_add(step).context("view position overflow")?;
            }
            file_views.push((map.len(), views));
        }

        let mut tensors = HashMap::with_capacity(index.tensors.len());
        for (name, record) in index.tensors {
            let dtype = parse_dtype(&record.dtype)?;
            ensure!(
                record.sha256.len() == 64 && record.sha256.bytes().all(|b| b.is_ascii_hexdigit()),
                "missing verified payload digest for {name}"
            );
            let expected = record
                .shape
                .iter()
                .try_fold(dtype.size_of(), |bytes, &dim| {
                    usize::try_from(dim)
                        .ok()
                        .and_then(|dim| bytes.checked_mul(dim))
                })
                .context("resident tensor shape overflow")?;
            ensure!(
                record.nbytes > 0 && record.nbytes == expected,
                "resident tensor byte size mismatch: {name}"
            );
            ensure!(
                record.offset % dtype.size_of() == 0,
                "unaligned resident tensor: {name}"
            );
            let (file_len, views) = file_views
                .get(record.file)
                .context("invalid resident backing file index")?;
            let end = record
                .offset
                .checked_add(record.nbytes)
                .context("resident tensor range overflow")?;
            ensure!(
                end <= *file_len,
                "resident tensor exceeds backing file: {name}"
            );
            let (start, _, buffer) = views
                .iter()
                .find(|(start, length, _)| record.offset >= *start && end <= *start + *length)
                .with_context(|| format!("no resident Metal view contains {name}"))?;
            tensors.insert(
                name,
                TensorView {
                    buffer: buffer.clone(),
                    offset: record.offset - *start,
                    nbytes: record.nbytes,
                    dtype,
                    shape: record.shape,
                },
            );
        }
        Ok(Self {
            tensors,
            buffers,
            mapped_bytes,
        })
    }

    fn tensor(&self, name: &str) -> anyhow::Result<&TensorView> {
        self.tensors
            .get(name)
            .with_context(|| format!("resident tensor {name} missing"))
    }

    pub(crate) fn contains(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }
    pub(crate) fn len(&self) -> usize {
        self.tensors.len()
    }
    pub(crate) fn get(&self, name: &str) -> anyhow::Result<Array> {
        let t = self.tensor(name)?;
        Ok(Array::from_buffer_offset(
            t.buffer.clone(),
            t.offset,
            &t.shape,
            t.dtype,
        )?)
    }
    pub(crate) fn get_bytes(&self, name: &str) -> anyhow::Result<Array> {
        let t = self.tensor(name)?;
        let length = i32::try_from(t.nbytes).context("resident byte tensor exceeds array limit")?;
        Ok(Array::from_buffer_offset(
            t.buffer.clone(),
            t.offset,
            &[length],
            Dtype::Uint8,
        )?)
    }
    pub(crate) fn nbytes(&self, name: &str) -> anyhow::Result<usize> {
        Ok(self.tensor(name)?.nbytes)
    }
    pub(crate) fn dtype(&self, name: &str) -> anyhow::Result<Dtype> {
        Ok(self.tensor(name)?.dtype)
    }
    pub(crate) fn shape(&self, name: &str) -> anyhow::Result<Vec<i32>> {
        Ok(self.tensor(name)?.shape.clone())
    }
    pub(crate) fn buffers(&self) -> &[Arc<lisa_mlx::runtime::Buffer>] {
        &self.buffers
    }
    pub(crate) fn mapped_bytes(&self) -> usize {
        self.mapped_bytes
    }
}

fn parse_dtype(dtype: &str) -> anyhow::Result<Dtype> {
    Ok(match dtype {
        "BF16" => Dtype::Bfloat16,
        "F16" => Dtype::Float16,
        "F32" => Dtype::Float32,
        "I64" => Dtype::Int64,
        "I32" => Dtype::Int32,
        "U32" => Dtype::Uint32,
        "U8" => Dtype::Uint8,
        "BOOL" => Dtype::Bool,
        _ => anyhow::bail!("unsupported resident dtype {dtype}"),
    })
}

unsafe extern "C" {
    fn getpagesize() -> i32;
}
