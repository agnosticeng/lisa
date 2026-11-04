use super::*;
use std::io::Cursor;

fn string(out: &mut Vec<u8>, value: &str) {
    out.extend_from_slice(&(value.len() as u64).to_le_bytes());
    out.extend_from_slice(value.as_bytes());
}

fn file_bytes(tensors: &[(&str, &[u64], TensorType, u64)], payload: &[u8]) -> Vec<u8> {
    let mut bytes = b"GGUF".to_vec();
    bytes.extend_from_slice(&3u32.to_le_bytes());
    bytes.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
    bytes.extend_from_slice(&1u64.to_le_bytes());
    string(&mut bytes, "general.architecture");
    bytes.extend_from_slice(&8u32.to_le_bytes());
    string(&mut bytes, "deepseek41");
    for &(name, dimensions, kind, offset) in tensors {
        string(&mut bytes, name);
        bytes.extend_from_slice(&(dimensions.len() as u32).to_le_bytes());
        for d in dimensions {
            bytes.extend_from_slice(&d.to_le_bytes());
        }
        bytes.extend_from_slice(&kind.0.to_le_bytes());
        bytes.extend_from_slice(&offset.to_le_bytes());
    }
    bytes.resize(bytes.len().div_ceil(32) * 32, 0);
    bytes.extend_from_slice(payload);
    bytes
}

#[test]
fn gguf_header_geometry_and_range_checks() {
    let tensors = [
        ("a", &[256, 2][..], TensorType::IQ1_M, 0),
        ("b", &[32, 4][..], TensorType::IQ4_NL, 128),
    ];
    let bytes = file_bytes(&tensors, &[0; 200]);
    let header = Header::read(&mut Cursor::new(&bytes), bytes.len() as u64).unwrap();
    assert_eq!(header.tensors["a"].bytes, 112);
    assert_eq!(header.tensors["b"].bytes, 72);
    assert_eq!(header.tensors["b"].offset, header.data_offset + 128);
    assert!(Header::read(&mut Cursor::new(&bytes), bytes.len() as u64 - 1).is_err());
    let overlap = file_bytes(
        &[
            ("a", &[32, 2], TensorType::Q8_0, 0),
            ("b", &[32, 2], TensorType::Q8_0, 32),
        ],
        &[0; 100],
    );
    assert!(Header::read(&mut Cursor::new(&overlap), overlap.len() as u64).is_err());
}

#[test]
fn gguf_header_rejects_invalid_types_dimensions_and_duplicates() {
    for tensor in [
        ("a", &[31][..], TensorType::IQ4_NL, 0),
        ("a", &[0][..], TensorType::F32, 0),
        ("a", &[u64::MAX, 2][..], TensorType::F32, 0),
        ("a", &[32][..], TensorType(999), 0),
        ("a", &[32][..], TensorType::Q8_0, 1),
    ] {
        let bytes = file_bytes(&[tensor], &[0; 128]);
        assert!(Header::read(&mut Cursor::new(&bytes), bytes.len() as u64).is_err());
    }
    let bytes = file_bytes(
        &[
            ("a", &[1], TensorType::F32, 0),
            ("a", &[1], TensorType::F32, 32),
        ],
        &[0; 36],
    );
    assert!(Header::read(&mut Cursor::new(&bytes), bytes.len() as u64).is_err());
}

#[test]
fn gguf_header_truncation_is_an_error() {
    let bytes = file_bytes(&[("a", &[32], TensorType::Q8_0, 0)], &[0; 34]);
    let header = Header::read(&mut Cursor::new(&bytes), bytes.len() as u64).unwrap();
    // Header-only remote inspection can omit tensor payloads, but never part
    // of the metadata or tensor descriptors.
    for length in 0..header.data_offset as usize - 31 {
        assert!(Header::read(&mut Cursor::new(&bytes[..length]), bytes.len() as u64).is_err());
    }
}

struct TempFile(std::path::PathBuf);
impl TempFile {
    pub fn new(bytes: &[u8]) -> Self {
        use std::io::Write;
        static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let id = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!("lisa-gguf-{}-{id}.gguf", std::process::id()));
        let mut file = std::fs::OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&path)
            .unwrap();
        file.write_all(bytes).unwrap();
        Self(path)
    }
}
impl Drop for TempFile {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

#[test]
fn gguf_disk_reads_are_bounded_and_ordered() {
    let values: Vec<u8> = (0..16).flat_map(|i| (i as f32).to_le_bytes()).collect();
    let file = TempFile::new(&file_bytes(&[("a", &[4, 4], TensorType::F32, 0)], &values));
    let package = Package::open(&file.0).unwrap();
    let tensor = package.tensor("a").unwrap().disk_reader().unwrap();
    let mut result = [0u8; 48];
    tensor.read_rows(&[3, 0, 3], &mut result).unwrap();
    assert_eq!(&result[..16], &values[48..64]);
    assert_eq!(&result[16..32], &values[..16]);
    assert_eq!(&result[32..48], &values[48..64]);
    assert!(tensor.read_rows(&[4], &mut result[..16]).is_err());
    assert!(tensor.read_bytes(63, &mut [0; 2]).is_err());
}

#[test]
fn gguf_mapping_preserves_payload_offset_and_lifetime() {
    let values: Vec<u8> = (0..16).flat_map(|i| (i as f32).to_le_bytes()).collect();
    let file = TempFile::new(&file_bytes(&[("a", &[4, 4], TensorType::F32, 0)], &values));
    let package = Package::open(&file.0).unwrap();
    let tensor = package.tensor("a").unwrap();
    let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as usize;
    let (mapping, offset) = tensor.mapped_region().unwrap();
    assert_eq!(mapping.len(), tensor.mapped_region_bytes().unwrap());
    assert_eq!(mapping.len() % page, 0);
    assert_eq!(mapping.as_ptr() as usize % page, 0);
    assert_eq!(offset, tensor.info.offset as usize % page);
    assert_ne!(offset, 0);
    assert!(offset + values.len() <= mapping.len());
    let payload = tensor.map().unwrap();
    drop(package);
    assert_eq!(&mapping[offset..offset + values.len()], &values);
    assert_eq!(&payload[..], &values);
}
