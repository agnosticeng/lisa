use super::*;
use std::io::Write;

fn shard_file(header: serde_json::Value, payload: &[u8]) -> tempfile::NamedTempFile {
    let mut file = tempfile::NamedTempFile::new().unwrap();
    let header = serde_json::to_vec(&header).unwrap();
    file.write_all(&(header.len() as u64).to_le_bytes())
        .unwrap();
    file.write_all(&header).unwrap();
    file.write_all(payload).unwrap();
    file.flush().unwrap();
    file
}

#[test]
fn original_affine_payloads_and_metadata_survive_without_a_device() {
    // Deliberately unaligned U32 payload, plus BF16 scales/biases: no decoding,
    // tensor-name normalization, padding or precision conversion is allowed.
    let file = shard_file(
        serde_json::json!({
            "prefix": {"dtype":"U8", "shape":[1], "data_offsets":[0,1]},
            "model.layers.0.proj.weight": {"dtype":"U32", "shape":[1,2], "data_offsets":[1,9]},
            "model.layers.0.proj.scales": {"dtype":"BF16", "shape":[1], "data_offsets":[9,11]},
            "model.layers.0.proj.biases": {"dtype":"BF16", "shape":[1], "data_offsets":[11,13]},
            "__metadata__": {"format":"mlx"}
        }),
        &[0xff, 1, 2, 3, 4, 5, 6, 7, 8, 0x80, 0x3f, 0x80, 0xbf],
    );
    let shard = Shard::open(file.path()).unwrap();
    let name = "model.layers.0.proj.weight";
    let (bytes, dtype, shape) = shard.tensor_bytes(name).unwrap();
    assert_eq!(bytes, &[1, 2, 3, 4, 5, 6, 7, 8]);
    assert_eq!(dtype, Dtype::Uint32);
    assert_eq!(shape, &[1, 2]);
    assert_eq!(shard.tensor_names().count(), 4);
    assert_eq!(
        shard.tensor_bytes("model.layers.0.proj.scales").unwrap().0,
        &[0x80, 0x3f]
    );
    assert_eq!(
        shard.tensor_bytes("model.layers.0.proj.biases").unwrap().0,
        &[0x80, 0xbf]
    );
    let info = shard.tensor_info(name).unwrap();
    let range = info.start..info.end;
    let owner = Arc::clone(shard.mapping());
    drop(shard);
    assert_eq!(&owner[range], &[1, 2, 3, 4, 5, 6, 7, 8]);
}

#[test]
fn scalar_and_empty_shapes_preserve_the_storage_contract() {
    let file = shard_file(
        serde_json::json!({
            "scalar": {"dtype":"F32", "shape":[], "data_offsets":[0,4]},
            "empty": {"dtype":"I64", "shape":[0,8], "data_offsets":[4,4]}
        }),
        &1.25f32.to_le_bytes(),
    );
    let shard = Shard::open(file.path()).unwrap();
    assert_eq!(
        shard.tensor_bytes("scalar").unwrap().0,
        &1.25f32.to_le_bytes()
    );
    assert_eq!(shard.tensor_bytes("scalar").unwrap().2, &[] as &[i32]);
    assert_eq!(shard.tensor_bytes("empty").unwrap().0, &[] as &[u8]);
    assert!(shard.tensor_info("absent").is_none());
}

#[test]
fn malformed_tensor_metadata_is_rejected_before_import() {
    for info in [
        serde_json::json!({"dtype":"UNKNOWN", "shape":[1], "data_offsets":[0,1]}),
        serde_json::json!({"dtype":"F32", "shape":[-1], "data_offsets":[0,4]}),
        serde_json::json!({"dtype":"F32", "shape":[2147483648u64], "data_offsets":[0,4]}),
        serde_json::json!({"dtype":"F32", "shape":[2147483647,2147483647,2147483647], "data_offsets":[0,4]}),
        serde_json::json!({"dtype":"F32", "shape":[1], "data_offsets":[0,3]}),
        serde_json::json!({"dtype":"F32", "shape":[1], "data_offsets":[4,8]}),
        serde_json::json!({"dtype":"F32", "shape":[1], "data_offsets":[4,0]}),
        serde_json::json!({"dtype":"F32", "shape":[1], "data_offsets":[0]}),
    ] {
        let file = shard_file(serde_json::json!({"x":info}), &[0; 4]);
        assert!(Shard::open(file.path()).is_err(), "accepted {info}");
    }
    for bytes in [
        vec![0; 7],
        u64::MAX.to_le_bytes().to_vec(),
        65_000_000u64.to_le_bytes().to_vec(),
    ] {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        file.write_all(&bytes).unwrap();
        assert!(Shard::open(file.path()).is_err());
    }
}

#[test]
fn shard_index_requires_relative_contained_paths() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("model.safetensors.index.json");
    for bad in [
        "",
        "/outside.safetensors",
        "../outside.safetensors",
        "weights/../../outside",
    ] {
        std::fs::write(
            &path,
            serde_json::to_vec(&serde_json::json!({"weight_map":{"x":bad}})).unwrap(),
        )
        .unwrap();
        assert!(Checkpoint::open(dir.path()).is_err(), "accepted {bad}");
    }
    std::fs::write(
        &path,
        br#"{"weight_map":{"model.x":"weights/part-01.safetensors"}}"#,
    )
    .unwrap();
    let checkpoint = Checkpoint::open(dir.path()).unwrap();
    assert_eq!(
        checkpoint.weight_map["model.x"],
        "weights/part-01.safetensors"
    );
    assert_eq!(checkpoint.dir, dir.path());
}

#[test]
fn oversized_index_is_rejected_without_reading_the_payload() {
    let file = tempfile::NamedTempFile::new().unwrap();
    file.as_file().set_len(256 * 1024 * 1024 + 1).unwrap();
    assert!(
        read_index_bytes(file.path())
            .unwrap_err()
            .to_string()
            .contains("256 MiB")
    );
}
