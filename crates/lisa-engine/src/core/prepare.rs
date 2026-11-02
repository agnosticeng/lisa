//! Explicit, CPU-only preparation of lossless resident checkpoint storage.
//!
//! Original safetensors remain authoritative. Payloads are streamed without
//! conversion, read back for SHA-256 verification, then published atomically.

use std::collections::{BTreeMap, HashMap};
use std::ffi::CString;
use std::fs::{self, File, Metadata, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::marker::PhantomData;
use std::os::unix::ffi::OsStrExt;
use std::os::unix::fs::{DirBuilderExt, MetadataExt};
use std::path::{Component, Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use anyhow::{Context, ensure};
use serde::{Deserialize, Serialize, de};
use sha2::{Digest, Sha256};

use super::loader::sanitize_name;
use super::weight_arena::{ArenaIndex, FileIdentity, TensorRecord};

const INDEX_NAME: &str = "model.safetensors.index.json";
const MAX_HEADER_BYTES: u64 = 64 * 1024 * 1024;
const COPY_CHUNK_BYTES: usize = 8 * 1024 * 1024;
const PROGRESS_BYTES: u64 = 4 * 1024 * 1024 * 1024;
const DISK_MARGIN_BYTES: u64 = 64 * 1024 * 1024;

/// Disk requirements and the destination of an explicit preparation request.
#[derive(Debug, Serialize)]
pub struct Summary {
    pub index_path: PathBuf,
    pub tensor_count: usize,
    pub source_file_count: usize,
    pub original_bytes: u64,
    pub copied_storage_bytes: u64,
    pub required_free_disk_bytes: u64,
    pub available_disk_bytes: u64,
    pub plan_only: bool,
}

struct SourceTensor {
    name: String,
    offset: u64,
    nbytes: u64,
    dtype: String,
    shape: Vec<i32>,
}

struct SourceShard {
    identity: usize,
    tensors: Vec<SourceTensor>,
}

struct Plan {
    summary: Summary,
    sources: Vec<FileIdentity>,
    shards: Vec<SourceShard>,
}

#[derive(Deserialize)]
struct CheckpointIndex {
    weight_map: UniqueMap<String>,
}

#[derive(Deserialize)]
struct HeaderTensor {
    dtype: String,
    shape: Vec<u64>,
    data_offsets: [u64; 2],
}

// Duplicate JSON names must not silently replace checkpoint or tensor records.
struct UniqueMap<T>(BTreeMap<String, T>);

impl<'de, T: Deserialize<'de>> Deserialize<'de> for UniqueMap<T> {
    fn deserialize<D: de::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct Visitor<T>(PhantomData<T>);
        impl<'de, T: Deserialize<'de>> de::Visitor<'de> for Visitor<T> {
            type Value = UniqueMap<T>;

            fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                formatter.write_str("an object with unique names")
            }

            fn visit_map<A: de::MapAccess<'de>>(self, mut map: A) -> Result<Self::Value, A::Error> {
                let mut entries = BTreeMap::new();
                while let Some((name, value)) = map.next_entry::<String, T>()? {
                    if entries.insert(name.clone(), value).is_some() {
                        return Err(de::Error::custom(format!("duplicate JSON name: {name}")));
                    }
                }
                Ok(UniqueMap(entries))
            }
        }
        deserializer.deserialize_map(Visitor(PhantomData))
    }
}

/// Prepare `<dir>/.lisa-resident/index.json` without loading a model or using Metal.
///
/// `plan_only` reads bounded JSON headers and reports disk requirements without
/// reading tensor payloads or creating files. Preparation never overwrites an
/// existing resident directory and never modifies original checkpoint files.
pub fn prepare_model(dir: &Path, plan_only: bool) -> anyhow::Result<Summary> {
    let dir = dir
        .canonicalize()
        .with_context(|| format!("opening checkpoint directory {}", dir.display()))?;
    ensure!(
        dir.is_dir(),
        "checkpoint path is not a directory: {}",
        dir.display()
    );
    let family = crate::models::model_type_of(&dir)?;
    ensure!(
        matches!(family.as_str(), "glm5_next" | "glm5_next_text"),
        "resident preparation currently supports GLM-5.3 checkpoints; found model_type {family:?}"
    );
    let output = dir.join(".lisa-resident");
    if !plan_only {
        ensure_absent(&output)?;
    }
    let plan = make_plan(&dir, plan_only)?;
    if plan_only {
        return Ok(plan.summary);
    }
    ensure!(
        plan.summary.available_disk_bytes >= plan.summary.required_free_disk_bytes,
        "insufficient free disk space at {}: need {} bytes, have {} bytes",
        dir.display(),
        plan.summary.required_free_disk_bytes,
        plan.summary.available_disk_bytes,
    );

    let mut staging = StagingDirectory::create(&dir)?;
    let weights_path = staging.path.join("weights.bin");
    eprintln!(
        "[prepare] copying {} tensors ({:.2} GiB) into {}",
        plan.summary.tensor_count,
        plan.summary.original_bytes as f64 / (1u64 << 30) as f64,
        staging.path.display(),
    );
    let started = Instant::now();
    let mut weights = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&weights_path)?;
    let mut records = HashMap::with_capacity(plan.summary.tensor_count);
    let mut buffer = vec![0u8; COPY_CHUNK_BYTES];
    let mut cursor = 0u64;
    let mut completed = 0u64;
    let mut next_progress = PROGRESS_BYTES;
    for shard in &plan.shards {
        let identity = &plan.sources[shard.identity];
        let mut source = open_unchanged(&dir, identity)?;
        for tensor in &shard.tensors {
            let offset = align_up(cursor, 64)?;
            write_padding(&mut weights, offset - cursor)?;
            source.seek(SeekFrom::Start(tensor.offset))?;
            let mut digest = Sha256::new();
            stream_payload(&mut source, tensor.nbytes, &mut buffer, |bytes| {
                weights.write_all(bytes)?;
                digest.update(bytes);
                Ok(())
            })
            .with_context(|| format!("copying tensor {}", tensor.name))?;
            records.insert(
                tensor.name.clone(),
                TensorRecord {
                    file: 0,
                    offset: usize::try_from(offset).context("resident tensor offset overflow")?,
                    nbytes: usize::try_from(tensor.nbytes)
                        .context("resident tensor size overflow")?,
                    dtype: tensor.dtype.clone(),
                    shape: tensor.shape.clone(),
                    sha256: format!("{:x}", digest.finalize()),
                },
            );
            cursor = offset
                .checked_add(tensor.nbytes)
                .context("resident size overflow")?;
            completed = completed
                .checked_add(tensor.nbytes)
                .context("copy progress overflow")?;
            if completed >= next_progress {
                eprintln!(
                    "[prepare] copied {:.1} / {:.1} GiB in {:.1}s",
                    completed as f64 / (1u64 << 30) as f64,
                    plan.summary.original_bytes as f64 / (1u64 << 30) as f64,
                    started.elapsed().as_secs_f64(),
                );
                next_progress = completed.saturating_add(PROGRESS_BYTES);
            }
        }
        check_identity(&source.metadata()?, identity)?;
    }
    write_padding(
        &mut weights,
        plan.summary
            .copied_storage_bytes
            .checked_sub(cursor)
            .context("copied payloads exceed the planned backing size")?,
    )?;
    weights.sync_all()?;
    drop(weights);

    eprintln!("[prepare] verifying stored SHA-256 payloads");
    let mut weights = File::open(&weights_path)?;
    ensure!(
        weights.metadata()?.len() == plan.summary.copied_storage_bytes,
        "prepared backing file has an unexpected size"
    );
    // Verify in source/copy order to keep disk reads sequential.
    let mut verified = 0u64;
    next_progress = PROGRESS_BYTES;
    for shard in &plan.shards {
        for tensor in &shard.tensors {
            let record = &records[&tensor.name];
            weights.seek(SeekFrom::Start(record.offset as u64))?;
            let mut digest = Sha256::new();
            stream_payload(&mut weights, tensor.nbytes, &mut buffer, |bytes| {
                digest.update(bytes);
                Ok(())
            })?;
            ensure!(
                format!("{:x}", digest.finalize()) == record.sha256,
                "stored payload digest differs from the original: {}",
                tensor.name,
            );
            verified = verified
                .checked_add(tensor.nbytes)
                .context("verification progress overflow")?;
            if verified >= next_progress {
                eprintln!(
                    "[prepare] readback verified {:.1} GiB",
                    verified as f64 / (1u64 << 30) as f64
                );
                next_progress = verified.saturating_add(PROGRESS_BYTES);
            }
        }
    }
    for identity in &plan.sources {
        open_unchanged(&dir, identity)?;
    }
    let backing = file_identity(PathBuf::from("weights.bin"), &weights.metadata()?)?;
    let index = ArenaIndex {
        format: "lisa-resident-weights-v1".into(),
        source_files: plan.sources,
        backing_files: vec![backing],
        tensors: records,
    };
    let mut index_file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(staging.path.join("index.json"))?;
    serde_json::to_writer(&mut index_file, &index)?;
    index_file.write_all(b"\n")?;
    index_file.sync_all()?;
    File::open(&staging.path)?.sync_all()?;
    publish_exclusive(&staging.path, &output)?;
    staging.published = true;
    File::open(&dir)?
        .sync_all()
        .context("resident index was published, but syncing its parent directory failed")?;
    eprintln!(
        "[prepare] prepared {} in {:.1}s",
        plan.summary.index_path.display(),
        started.elapsed().as_secs_f64()
    );
    Ok(plan.summary)
}

fn make_plan(dir: &Path, plan_only: bool) -> anyhow::Result<Plan> {
    let mut index_file =
        File::open(dir.join(INDEX_NAME)).context("opening checkpoint shard index")?;
    let index_identity = file_identity(PathBuf::from(INDEX_NAME), &index_file.metadata()?)?;
    ensure!(
        index_identity.size <= MAX_HEADER_BYTES,
        "checkpoint index exceeds 64 MiB"
    );
    let mut bytes = Vec::new();
    (&mut index_file)
        .take(MAX_HEADER_BYTES + 1)
        .read_to_end(&mut bytes)?;
    ensure!(
        bytes.len() as u64 <= MAX_HEADER_BYTES,
        "checkpoint index exceeds 64 MiB"
    );
    check_identity(&index_file.metadata()?, &index_identity)?;
    let index: CheckpointIndex =
        serde_json::from_slice(&bytes).context("parsing checkpoint shard index")?;
    let mut by_shard = BTreeMap::<PathBuf, Vec<String>>::new();
    for (name, shard) in index.weight_map.0 {
        let shard = PathBuf::from(shard);
        ensure!(
            !shard.as_os_str().is_empty()
                && shard
                    .components()
                    .all(|part| matches!(part, Component::Normal(_))),
            "checkpoint shard must be relative to its model directory: {}",
            shard.display(),
        );
        by_shard.entry(shard).or_default().push(name);
    }
    let mut sources = vec![index_identity];
    let mut shards = Vec::new();
    let mut names = std::collections::HashSet::new();
    let mut original_bytes = 0u64;
    let mut storage_bytes = 0u64;
    let mut index_allowance = 0u64;
    for (relative, raw_names) in by_shard {
        let mut file = File::open(dir.join(&relative))
            .with_context(|| format!("opening shard {}", relative.display()))?;
        let identity = file_identity(relative.clone(), &file.metadata()?)?;
        let (header, data_start) = read_header(&mut file, identity.size)
            .with_context(|| format!("reading shard {}", relative.display()))?;
        check_identity(&file.metadata()?, &identity)?;
        let mut tensors = Vec::new();
        for raw in raw_names {
            let Some(name) = sanitize_name(&raw) else {
                continue;
            };
            ensure!(
                names.insert(name.clone()),
                "duplicate sanitized tensor: {name}"
            );
            let value = header
                .0
                .get(&raw)
                .with_context(|| format!("tensor {raw} missing from {}", relative.display()))?;
            let tensor: HeaderTensor = serde_json::from_value(value.clone())
                .with_context(|| format!("invalid tensor header: {raw}"))?;
            let width = dtype_size(&tensor.dtype)?;
            let mut shape = Vec::with_capacity(tensor.shape.len());
            let mut nbytes = width;
            for dim in tensor.shape {
                shape.push(
                    i32::try_from(dim).with_context(|| format!("invalid dimension for {raw}"))?,
                );
                nbytes = nbytes
                    .checked_mul(dim)
                    .context("tensor shape size overflow")?;
            }
            let [start, end] = tensor.data_offsets;
            let offset = data_start
                .checked_add(start)
                .context("tensor offset overflow")?;
            ensure!(
                nbytes > 0
                    && end.checked_sub(start) == Some(nbytes)
                    && data_start
                        .checked_add(end)
                        .is_some_and(|end| end <= identity.size),
                "invalid source tensor byte range: {raw}",
            );
            index_allowance = index_allowance
                .checked_add(
                    512u64
                        // JSON can escape each UTF-8 byte into six ASCII bytes.
                        .checked_add((name.len() as u64).saturating_mul(6))
                        .and_then(|n| n.checked_add((raw.len() as u64).saturating_mul(6)))
                        .and_then(|n| n.checked_add(shape.len() as u64 * 12))
                        .context("index allowance overflow")?,
                )
                .context("index allowance overflow")?;
            tensors.push(SourceTensor {
                name,
                offset,
                nbytes,
                dtype: tensor.dtype,
                shape,
            });
        }
        tensors.sort_by_key(|tensor| tensor.offset);
        for pair in tensors.windows(2) {
            let previous_end = pair[0]
                .offset
                .checked_add(pair[0].nbytes)
                .context("source tensor range overflow")?;
            ensure!(
                previous_end <= pair[1].offset,
                "overlapping source tensors {} and {} in {}",
                pair[0].name,
                pair[1].name,
                relative.display(),
            );
        }
        for tensor in &tensors {
            original_bytes = original_bytes
                .checked_add(tensor.nbytes)
                .context("checkpoint size overflow")?;
            storage_bytes = align_up(storage_bytes, 64)?
                .checked_add(tensor.nbytes)
                .context("resident size overflow")?;
        }
        let identity_index = sources.len();
        sources.push(identity);
        shards.push(SourceShard {
            identity: identity_index,
            tensors,
        });
    }
    ensure!(
        !names.is_empty(),
        "checkpoint has no supported tensor payloads"
    );
    let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    ensure!(page_size > 0, "invalid OS page size");
    storage_bytes = align_up(storage_bytes, page_size as u64)?;
    let required_free_disk_bytes = storage_bytes
        .checked_add(index_allowance)
        .and_then(|bytes| bytes.checked_add(DISK_MARGIN_BYTES))
        .context("required disk space overflow")?;
    for identity in &sources {
        open_unchanged(dir, identity)?;
    }
    Ok(Plan {
        summary: Summary {
            index_path: dir.join(".lisa-resident/index.json"),
            tensor_count: names.len(),
            source_file_count: sources.len(),
            original_bytes,
            copied_storage_bytes: storage_bytes,
            required_free_disk_bytes,
            available_disk_bytes: available_disk_bytes(dir)?,
            plan_only,
        },
        sources,
        shards,
    })
}

fn read_header(file: &mut File, size: u64) -> anyhow::Result<(UniqueMap<serde_json::Value>, u64)> {
    ensure!(
        size >= 8,
        "safetensors shard is smaller than its header prefix"
    );
    let mut length = [0u8; 8];
    file.read_exact(&mut length)?;
    let length = u64::from_le_bytes(length);
    let data_start = length
        .checked_add(8)
        .context("safetensors header size overflow")?;
    ensure!(
        length <= MAX_HEADER_BYTES && data_start <= size,
        "invalid or oversized safetensors header"
    );
    let mut bytes = vec![0; usize::try_from(length)?];
    file.read_exact(&mut bytes)?;
    Ok((serde_json::from_slice(&bytes)?, data_start))
}

fn dtype_size(dtype: &str) -> anyhow::Result<u64> {
    Ok(match dtype {
        "BF16" | "F16" => 2,
        "F32" | "I32" | "U32" => 4,
        "I64" => 8,
        "U8" | "BOOL" => 1,
        _ => anyhow::bail!("unsupported checkpoint dtype {dtype}"),
    })
}

fn align_up(value: u64, alignment: u64) -> anyhow::Result<u64> {
    value
        .checked_add(alignment - 1)
        .map(|value| value / alignment * alignment)
        .context("aligned storage size overflow")
}

fn file_identity(path: PathBuf, metadata: &Metadata) -> anyhow::Result<FileIdentity> {
    ensure!(
        metadata.is_file(),
        "checkpoint/backing path is not a regular file: {}",
        path.display()
    );
    Ok(FileIdentity {
        path,
        size: metadata.len(),
        mtime_ns: i128::from(metadata.mtime()) * 1_000_000_000 + i128::from(metadata.mtime_nsec()),
        inode: metadata.ino(),
        device: metadata.dev(),
    })
}

fn check_identity(metadata: &Metadata, expected: &FileIdentity) -> anyhow::Result<()> {
    ensure!(
        &file_identity(expected.path.clone(), metadata)? == expected,
        "original checkpoint changed during preparation: {}",
        expected.path.display(),
    );
    Ok(())
}

fn open_unchanged(root: &Path, identity: &FileIdentity) -> anyhow::Result<File> {
    let file = File::open(root.join(&identity.path)).with_context(|| {
        format!(
            "opening original checkpoint file {}",
            identity.path.display()
        )
    })?;
    check_identity(&file.metadata()?, identity)?;
    Ok(file)
}

fn stream_payload(
    source: &mut File,
    mut left: u64,
    buffer: &mut [u8],
    mut consume: impl FnMut(&[u8]) -> anyhow::Result<()>,
) -> anyhow::Result<()> {
    while left > 0 {
        let count = left.min(buffer.len() as u64) as usize;
        source.read_exact(&mut buffer[..count])?;
        consume(&buffer[..count])?;
        left -= count as u64;
    }
    Ok(())
}

fn write_padding(file: &mut File, mut length: u64) -> anyhow::Result<()> {
    let zeros = [0u8; 16 * 1024];
    while length > 0 {
        let count = length.min(zeros.len() as u64) as usize;
        file.write_all(&zeros[..count])?;
        length -= count as u64;
    }
    Ok(())
}

fn available_disk_bytes(path: &Path) -> anyhow::Result<u64> {
    let path = CString::new(path.as_os_str().as_bytes())?;
    let mut stats = std::mem::MaybeUninit::<libc::statvfs>::uninit();
    // SAFETY: the path is NUL-terminated and statvfs initializes stats on success.
    ensure!(
        unsafe { libc::statvfs(path.as_ptr(), stats.as_mut_ptr()) } == 0,
        "reading free disk space: {}",
        std::io::Error::last_os_error()
    );
    let stats = unsafe { stats.assume_init() };
    let block = if stats.f_frsize > 0 {
        stats.f_frsize
    } else {
        stats.f_bsize
    };
    (stats.f_bavail as u64)
        .checked_mul(block as u64)
        .context("free disk space overflow")
}

fn ensure_absent(path: &Path) -> anyhow::Result<()> {
    match fs::symlink_metadata(path) {
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error.into()),
        Ok(_) => anyhow::bail!(
            "resident output already exists; preserve or remove it explicitly before rebuilding: {}",
            path.display()
        ),
    }
}

struct StagingDirectory {
    path: PathBuf,
    published: bool,
}

impl StagingDirectory {
    fn create(parent: &Path) -> anyhow::Result<Self> {
        static NEXT_ID: AtomicU64 = AtomicU64::new(0);
        let stamp = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
        for _ in 0..128 {
            let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);
            let path = parent.join(format!(
                ".lisa-resident-build-{}-{stamp}-{id}",
                std::process::id()
            ));
            match fs::DirBuilder::new().mode(0o700).create(&path) {
                Ok(()) => {
                    return Ok(Self {
                        path,
                        published: false,
                    });
                }
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(error) => return Err(error).context("creating resident preparation directory"),
            }
        }
        anyhow::bail!("could not allocate a unique resident preparation directory")
    }
}

impl Drop for StagingDirectory {
    fn drop(&mut self) {
        if !self.published {
            let _ = fs::remove_dir_all(&self.path);
        }
    }
}

fn publish_exclusive(from: &Path, to: &Path) -> anyhow::Result<()> {
    let from = CString::new(from.as_os_str().as_bytes())?;
    let to_c = CString::new(to.as_os_str().as_bytes())?;
    #[cfg(target_os = "macos")]
    // SAFETY: both paths are NUL-terminated. RENAME_EXCL forbids replacing a
    // destination created by another process after our initial existence check.
    let result = unsafe { libc::renamex_np(from.as_ptr(), to_c.as_ptr(), libc::RENAME_EXCL) };
    #[cfg(target_os = "linux")]
    let result = unsafe {
        libc::renameat2(
            libc::AT_FDCWD,
            from.as_ptr(),
            libc::AT_FDCWD,
            to_c.as_ptr(),
            libc::RENAME_NOREPLACE,
        )
    };
    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    anyhow::bail!("atomic resident preparation is unsupported on this operating system");
    #[cfg(any(target_os = "macos", target_os = "linux"))]
    ensure!(
        result == 0,
        "publishing resident directory {} without replacement: {}",
        to.display(),
        std::io::Error::last_os_error()
    );
    Ok(())
}
