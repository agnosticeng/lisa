//! A second storage provider exercises real mapped imports without Metal.
use super::*;
use lisa_backend::*;
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

#[derive(Clone)]
struct Storage {
    owner: HostStorage,
    offset: usize,
    shape: Vec<i32>,
    dtype: ScalarType,
}
impl Storage {
    fn bytes(&self) -> &[u8] {
        &self.owner.bytes()
            [self.offset..self.offset + tensor_bytes(&self.shape, self.dtype).unwrap()]
    }
}
#[derive(Clone, Default)]
struct HostViews(Arc<AtomicUsize>, bool);
impl Device for HostViews {
    type Tensor = Storage;
    type Error = std::io::Error;
    type KernelHandle = ();
    type Stream = ();
    type Completion = ();
    fn capabilities(&self) -> &[KernelCapability] {
        &[]
    }
    fn dtype(&self, t: &Storage) -> Option<ScalarType> {
        Some(t.dtype)
    }
    fn shape<'a>(&self, t: &'a Storage) -> &'a [i32] {
        &t.shape
    }
    fn stream(&self) {}
    fn wait(&self, _: &()) -> Result<(), Self::Error> {
        Ok(())
    }
    fn prepare(&self, _: Kernel) -> Result<Option<()>, Self::Error> {
        Ok(None)
    }
    fn dispatch(
        &self,
        _: &(),
        _: &[&Storage],
        _: &(),
    ) -> Result<Option<Dispatch<Outputs<Storage>, ()>>, Self::Error> {
        Err(std::io::Error::other("storage-only provider"))
    }
    fn pull(
        &self,
        source: TensorSource<'_>,
        options: ImportOptions,
    ) -> Result<Storage, Self::Error> {
        if !source.host_tensor().is_some_and(|h| h.validate()) {
            return Err(std::io::Error::other("invalid import"));
        }
        if matches!(options.mode, ImportMode::Copy | ImportMode::CopyCompact) {
            let host = source.host_tensor().unwrap();
            let owner = Arc::new(host.bytes.to_vec());
            let pointer = std::ptr::NonNull::new(owner.as_ptr() as *mut u8).unwrap();
            // SAFETY: `owner` is an immutable Vec whose exact buffer is `pointer..len`.
            let storage = unsafe { HostStorage::new(pointer, owner.len(), owner) };
            self.0.fetch_add(1, Ordering::SeqCst);
            return Ok(Storage {
                owner: storage,
                offset: 0,
                shape: host.shape.to_vec(),
                dtype: host.dtype,
            });
        }
        if self.1 {
            return Err(std::io::Error::other("this provider requires uploads"));
        }
        let TensorSource::Shared {
            storage,
            offset,
            shape,
            dtype,
        } = source
        else {
            return Err(std::io::Error::other("owned mapping required"));
        };
        self.0.fetch_add(1, Ordering::SeqCst);
        Ok(Storage {
            owner: storage,
            offset,
            shape: shape.to_vec(),
            dtype,
        })
    }
}
impl StorageDevice for HostViews {
    fn max_allocation_bytes(&self) -> usize {
        1024 * 1024
    }
    fn byte_view(
        &self,
        t: &Storage,
        offset: usize,
        shape: &[i32],
        dtype: ScalarType,
    ) -> Result<Storage, Self::Error> {
        let size =
            tensor_bytes(shape, dtype).ok_or_else(|| std::io::Error::other("invalid shape"))?;
        if !offset.is_multiple_of(dtype.size_of())
            || offset.checked_add(size).is_none_or(|n| n > t.bytes().len())
        {
            return Err(std::io::Error::other("view out of range"));
        }
        Ok(Storage {
            owner: t.owner.clone(),
            offset: t.offset + offset,
            shape: shape.to_vec(),
            dtype,
        })
    }
}

#[test]
fn gguf_import_preserves_payloads_and_reserves_before_import() -> anyhow::Result<()> {
    fn string(bytes: &mut Vec<u8>, text: &str) {
        bytes.extend((text.len() as u64).to_le_bytes());
        bytes.extend(text.as_bytes());
    }
    let tmp = tempfile::tempdir()?;
    let path = tmp.path().join("weights.gguf");
    let mut bytes = b"GGUF".to_vec();
    bytes.extend(3u32.to_le_bytes());
    bytes.extend(2u64.to_le_bytes());
    bytes.extend(0u64.to_le_bytes());
    for (name, width, kind, offset) in [
        ("packed", 32u64, lisa_gguf::TensorType::Q8_0, 0u64),
        ("plain", 2u64, lisa_gguf::TensorType::BF16, 64u64),
    ] {
        string(&mut bytes, name);
        bytes.extend(1u32.to_le_bytes());
        bytes.extend(width.to_le_bytes());
        bytes.extend(kind.0.to_le_bytes());
        bytes.extend(offset.to_le_bytes());
    }
    bytes.resize(bytes.len().div_ceil(32) * 32, 0);
    let packed: Vec<u8> = (0..34).collect();
    bytes.extend(&packed);
    bytes.extend([0; 30]);
    let plain = [0x80, 0x3f, 0x00, 0x40];
    bytes.extend(plain);
    std::fs::write(&path, bytes)?;
    let package = lisa_gguf::Package::open(&path)?;
    let device = HostViews::default();
    let denied = gguf::ResidentWeights::<_, ()>::pull(
        device.clone(),
        &package,
        &["packed", "plain"],
        |_| anyhow::bail!("budget denied"),
    );
    assert!(denied.is_err());
    assert_eq!(device.0.load(Ordering::SeqCst), 0);
    let lease = Arc::new(());
    let weak = Arc::downgrade(&lease);
    let store = gguf::ResidentWeights::pull(
        device.clone(),
        &package,
        &["packed", "plain"],
        |mapped_bytes| {
            assert!(mapped_bytes >= packed.len() + plain.len());
            Ok(lease)
        },
    )?;
    assert_eq!(device.0.load(Ordering::SeqCst), 2);
    assert!(store.tensors["packed"].plain_view().is_err());
    let packed_view = store.tensors["packed"].packed_bytes().clone();
    let plain_view = store.tensors["plain"].plain_view()?;
    let copy_only = HostViews(Arc::default(), true);
    assert!(
        gguf::ResidentWeights::pull(copy_only.clone(), &package, &["packed"], |_| Ok(())).is_err()
    );
    let uploaded = gguf::ResidentWeights::pull_with_policy(
        copy_only,
        &package,
        &["packed", "plain"],
        ResidencyPolicy::Uploaded,
        |bytes| {
            assert_eq!(bytes.host_resident, 0);
            assert_eq!(bytes.device_resident, packed.len() + plain.len());
            assert!(bytes.host_staging >= packed.len());
            Ok(())
        },
    )?;
    assert_eq!(uploaded.tensors["packed"].packed_bytes().bytes(), packed);
    assert_eq!(uploaded.tensors["plain"].plain_view()?.bytes(), plain);
    let packed_pointer = uploaded.tensors["packed"].packed_bytes().bytes().as_ptr();
    let plain_pointer = uploaded.tensors["plain"].plain_view()?.bytes().as_ptr();
    let uploaded = uploaded.into_prepared()?;
    assert_eq!(uploaded.tensors["packed"].packed_bytes().bytes(), packed);
    assert_eq!(
        uploaded.tensors["packed"].packed_bytes().bytes().as_ptr(),
        packed_pointer
    );
    assert_eq!(
        uploaded.tensors["plain"].plain_view()?.bytes().as_ptr(),
        plain_pointer
    );
    assert_eq!(uploaded.tensors["plain"].layout.dtype, ScalarType::BF16);
    assert_eq!(
        uploaded.tensors["packed"].layout.projection,
        lisa_models::packed::Projection::Q8
    );

    // A failed metadata adaptation must drain before releasing either the
    // already converted views or the unconverted original allocations.
    type Owner = dyn std::any::Any + Send + Sync;
    struct Drain {
        owners: Arc<std::sync::Mutex<Vec<std::sync::Weak<Owner>>>>,
        count: Arc<AtomicUsize>,
    }
    impl Drop for Drain {
        fn drop(&mut self) {
            assert!(
                self.owners
                    .lock()
                    .unwrap()
                    .iter()
                    .all(|p| p.upgrade().is_some())
            );
            self.count.fetch_add(1, Ordering::SeqCst);
        }
    }
    let owners = Arc::new(std::sync::Mutex::new(Vec::new()));
    let drains = Arc::new(AtomicUsize::new(0));
    let mut invalid =
        gguf::ResidentWeights::pull(device.clone(), &package, &["packed", "plain"], |_| {
            Ok(Drain {
                owners: owners.clone(),
                count: drains.clone(),
            })
        })?;
    for tensor in invalid.tensors.values() {
        owners
            .lock()
            .unwrap()
            .push(Arc::downgrade(&tensor.packed_bytes().owner.owner()));
    }
    invalid
        .tensors
        .get_mut("plain")
        .unwrap()
        .info
        .dimensions
        .clear();
    drop(package);
    assert!(invalid.into_prepared().is_err());
    assert_eq!(drains.load(Ordering::SeqCst), 1);
    assert!(owners.lock().unwrap().iter().all(|p| p.upgrade().is_none()));
    assert!(weak.upgrade().is_some());
    // Loading can erase GGUF metadata while preserving the exact imported
    // tensor views and the original admission lease for tensor-only execution.
    let (reservation, original) = store.into_parts();
    let mut tensors = std::collections::BTreeMap::new();
    tensors.insert(
        "packed".to_owned(),
        original["packed"].packed_bytes().clone(),
    );
    tensors.insert("plain".to_owned(), original["plain"].plain_view()?);
    let prepared = source::PreparedTensors {
        _reservation: reservation,
        tensors,
    };
    drop(original);
    assert!(weak.upgrade().is_some());
    assert_eq!(prepared.tensors["packed"].bytes(), packed);
    assert_eq!(prepared.tensors["plain"].bytes(), plain);
    drop(prepared);
    assert!(weak.upgrade().is_none());
    assert_eq!(packed_view.bytes(), packed);
    assert_eq!(plain_view.bytes(), plain);
    assert_eq!(plain_view.dtype, ScalarType::BF16);
    Ok(())
}
#[test]
fn mapped_mlx_import_preserves_packed_bytes_and_outlives_reader() -> anyhow::Result<()> {
    let tmp = tempfile::tempdir()?;
    let path = tmp.path().join("weights.safetensors");
    let mut header = serde_json::to_vec(
        &serde_json::json!({"packed":{"dtype":"U32","shape":[2,2],"data_offsets":[0,16]},"scales":{"dtype":"BF16","shape":[2],"data_offsets":[16,20]}}),
    )?;
    while header.len() % 8 != 0 {
        header.push(b' ');
    }
    let payload = [
        0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0xaa, 0xbb, 0xcc, 0xdd, 0xee,
        0xff, 0x80, 0x3f, 0x00, 0x40,
    ];
    let mut file = (header.len() as u64).to_le_bytes().to_vec();
    file.extend(header);
    file.extend(payload);
    std::fs::write(&path, file)?;
    let device = HostViews::default();
    let shard = mlx::Shard::open(device.clone(), &path)?;
    let packed = shard.tensor_array_shard_view("packed")?.unwrap();
    let scales = shard.tensor_array_shard_view("scales")?.unwrap();
    assert_eq!(
        device.0.load(Ordering::SeqCst),
        1,
        "one retained allocation per shard"
    );
    drop(shard);
    assert_eq!(packed.bytes(), &payload[..16]);
    assert_eq!(scales.bytes(), &payload[16..]);
    assert_eq!(packed.dtype, ScalarType::U32);
    assert_eq!(scales.dtype, ScalarType::BF16);
    assert!(
        device
            .byte_view(&packed, 12, &[2], ScalarType::U32)
            .is_err()
    );
    assert!(device.byte_view(&packed, 1, &[1], ScalarType::U32).is_err());
    let checkpoint = lisa_mlx::Checkpoint {
        dir: tmp.path().to_path_buf(),
        weight_map: std::collections::HashMap::from([
            ("packed".into(), "weights.safetensors".into()),
            ("scales".into(), "weights.safetensors".into()),
        ]),
    };
    let lazy =
        lazy::LazyTensorStore::from_checkpoint(device.clone(), tmp.path(), &checkpoint, |name| {
            Some(name.to_owned())
        })?;
    let mapped = lazy.get_mapped("packed")?;
    let owned = lazy.get_owned("packed")?;
    let combined = lazy.get_packed(&["packed".into(), "packed".into()])?;
    assert_eq!(lazy.tensor_dtype("scales")?, ScalarType::BF16);
    assert_eq!(lazy.tensor_shape("packed")?, [2, 2]);
    assert!(lazy.get("missing").is_err());
    assert!(
        lazy::LazyTensorStore::from_checkpoint(device, tmp.path(), &checkpoint, |_| Some(
            "collision".into()
        ),)
        .is_err()
    );
    drop(lazy);
    assert_eq!(mapped.bytes(), &payload[..16]);
    assert_eq!(owned.bytes(), mapped.bytes());
    assert_eq!(combined.shape, [4, 2]);
    assert_eq!(combined.bytes(), [&payload[..16], &payload[..16]].concat());
    Ok(())
}
#[test]
fn resident_index_checks_identity_and_ranges_with_non_metal_storage() -> anyhow::Result<()> {
    use crate::weights::weight_arena::{ArenaIndex, FileIdentity, ResidentWeights, TensorRecord};
    use std::os::unix::fs::MetadataExt;
    let tmp = tempfile::tempdir()?;
    let root = tmp.path();
    std::fs::write(root.join("source"), [17u8; 8])?;
    let payload = [0x80, 0x3f, 0x00, 0x40];
    std::fs::write(root.join("backing"), payload)?;
    let identity = |name: &str| -> anyhow::Result<FileIdentity> {
        let m = std::fs::metadata(root.join(name))?;
        Ok(FileIdentity {
            path: name.into(),
            size: m.len(),
            mtime_ns: i128::from(m.mtime()) * 1_000_000_000 + i128::from(m.mtime_nsec()),
            inode: m.ino(),
            device: m.dev(),
        })
    };
    let mut index = ArenaIndex {
        format: "lisa-resident-weights-v1".into(),
        source_files: vec![identity("source")?],
        backing_files: vec![identity("backing")?],
        tensors: std::collections::HashMap::from([(
            "scales".into(),
            TensorRecord {
                file: 0,
                offset: 0,
                nbytes: 4,
                dtype: "BF16".into(),
                shape: vec![2],
                sha256: "a".repeat(64),
            },
        )]),
    };
    let path = root.join("index.json");
    std::fs::write(&path, serde_json::to_vec(&index)?)?;
    let sources = std::collections::HashSet::from([std::path::PathBuf::from("source")]);
    let store = ResidentWeights::open(HostViews::default(), root, &path, &sources)?;
    let tensor = store.get("scales")?;
    let copy_only = HostViews(Arc::default(), true);
    assert!(ResidentWeights::open(copy_only.clone(), root, &path, &sources).is_err());
    let uploaded = ResidentWeights::open_with_policy(
        copy_only,
        root,
        &path,
        &sources,
        ResidencyPolicy::Uploaded,
    )?;
    assert_eq!(uploaded.residency().host_resident, 0);
    assert!(uploaded.residency().device_resident >= payload.len());
    assert_eq!(uploaded.get("scales")?.bytes(), payload);
    drop(store);
    assert_eq!(tensor.bytes(), payload);
    index.tensors.get_mut("scales").unwrap().offset = 4;
    std::fs::write(&path, serde_json::to_vec(&index)?)?;
    assert!(ResidentWeights::open(HostViews::default(), root, &path, &sources).is_err());
    index.tensors.get_mut("scales").unwrap().offset = 0;
    std::fs::write(&path, serde_json::to_vec(&index)?)?;
    std::fs::write(root.join("source"), [18u8; 9])?;
    assert!(ResidentWeights::open(HostViews::default(), root, &path, &sources).is_err());
    Ok(())
}
