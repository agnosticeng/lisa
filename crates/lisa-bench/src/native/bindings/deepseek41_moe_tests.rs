#![cfg(test)]
//! Full-width non-dyadic migration oracle. No external weights or recaptured goldens.
use super::*;
use crate::core::gguf_resident::PackedTensor;
use lisa_gguf::{TensorInfo, TensorType as T};
use lisa_metal::Dtype;
use std::collections::BTreeMap;
#[path = "deepseek41_moe_oracle.rs"]
mod oracle;
pub(crate) fn tensor(kind: T, shape: &[i32], seed: u32) -> Result<PackedTensor> {
    let dims: Vec<u64> = shape.iter().rev().map(|&d| d as u64).collect();
    let width = *shape.last().unwrap() as usize;
    let rows = shape.iter().take(shape.len() - 1).product::<i32>() as usize;
    let row_bytes = kind.row_bytes(width as u64)? as usize;
    let mut data = vec![0u8; row_bytes * rows];
    let mut rng = seed;
    let mut next = || {
        rng ^= rng << 13;
        rng ^= rng >> 17;
        rng ^= rng << 5;
        (rng >> 24) as u8
    };
    if matches!(kind, T::F32 | T::F16 | T::BF16) {
        let bytes = kind.layout().unwrap().2 as usize;
        for (i, dst) in data.chunks_exact_mut(bytes).enumerate() {
            let v = ((i * 19 + seed as usize) % 8191) as f32 / 8191.0 - 0.5;
            match kind {
                T::F32 => dst.copy_from_slice(&(v / 32.).to_le_bytes()),
                T::F16 => dst.copy_from_slice(&half::f16::from_f32(v / 32.).to_le_bytes()),
                _ => dst.copy_from_slice(&half::bf16::from_f32(v / 32.).to_le_bytes()),
            }
        }
    } else {
        let bytes = kind.layout().unwrap().2 as usize;
        for b in data.chunks_exact_mut(bytes) {
            for v in b.iter_mut() {
                *v = next();
            }
            if kind == T::Q2_K {
                b[80..82].copy_from_slice(&0x1800u16.to_le_bytes());
                b[82..84].copy_from_slice(&0x1400u16.to_le_bytes());
            } else if kind == T::IQ1_M {
                for i in 0..4 {
                    let at = 48 + 2 * i;
                    let v = u16::from_le_bytes([b[at], b[at + 1]]) & 0xfff;
                    b[at..at + 2].copy_from_slice(
                        &(v | (((0x1800u16 >> (4 * i)) & 15) << 12)).to_le_bytes(),
                    );
                }
            } else {
                b[..2].copy_from_slice(&0x1800u16.to_le_bytes());
                if matches!(kind, T::Q5_K | T::Q4_K) {
                    b[2..4].copy_from_slice(&0x1400u16.to_le_bytes());
                }
            }
        }
    }
    // Nonzero byte offset exercises mapped-view addressing without mutating weights.
    let mut storage = vec![0u8; 16];
    storage.extend_from_slice(&data);
    let storage = Array::from_slice(&storage, &[storage.len() as i32]);
    let (buffer, _) = storage.t.buffer_and_layout();
    let mut storage_shape = shape[..shape.len() - 1].to_vec();
    storage_shape.push(row_bytes as i32);
    let bytes = Array::from_buffer_offset(buffer.clone(), 16, &storage_shape, Dtype::Uint8)?;
    Ok(PackedTensor::from_parts_for_test(
        lisa_metal::device::MetalDevice::system_default(),
        bytes,
        TensorInfo {
            name: "fixture".into(),
            dimensions: dims,
            kind,
            offset: 0,
            bytes: data.len() as u64,
        },
    ))
}
pub(crate) fn counts() -> BTreeMap<String, u64> {
    // Label counts are published when command buffers complete. Reading them
    // before the fence can compare two empty/stale censuses and miss dispatches.
    if std::env::var_os("LISA_TRACE").is_some() {
        lisa_metal::Stream::thread_local_or_default()
            .runtime()
            .commands
            .flush_and_wait()
            .expect("complete the measured dispatch interval");
    }
    lisa_metal::runtime_label_dispatches().into_iter().collect()
}
pub(crate) fn difference(
    a: &BTreeMap<String, u64>,
    b: &BTreeMap<String, u64>,
) -> BTreeMap<String, u64> {
    b.iter()
        .filter_map(|(k, n)| {
            let d = n - a.get(k).unwrap_or(&0);
            (d > 0).then(|| (k.clone(), d))
        })
        .collect()
}
#[test]
fn packed_ffn_preserves_routing_shared_and_routed_bits() -> Result<()> {
    let stream = Stream::gpu();
    for kinds in [
        [
            T::F32,
            T::F32,
            T::Q8_0,
            T::Q8_0,
            T::Q8_0,
            T::IQ2_XXS,
            T::IQ1_M,
            T::Q5_K,
        ],
        [
            T::BF16,
            T::F32,
            T::Q5_K,
            T::IQ2_XS,
            T::Q5_K,
            T::Q5_K,
            T::IQ2_XS,
            T::IQ2_XXS,
        ],
    ] {
        let experts = 8;
        let weights = model::requirements(0, experts)
            .iter()
            .zip(kinds)
            .enumerate()
            .map(|(i, ((_, shape), kind))| tensor(kind, shape, 171 + i as u32))
            .collect::<Result<Vec<_>>>()?;
        let original = oracle::from_weights(&weights)?;
        let mut projections = Vec::new();
        let mut arrays = Vec::new();
        for (i, w) in weights.iter().enumerate() {
            let kind = crate::bindings::packed::projection(w.info.kind)?;
            if i != 1 {
                projections.push(kind);
            }
            arrays.push(if kind == Projection::Plain {
                w.plain_view()?
            } else {
                w.packed_bytes().clone()
            });
        }
        let migrated = Moe::bound(
            arrays,
            model::Spec {
                experts,
                projections: projections.try_into().unwrap(),
            },
            &stream,
        )?;
        for rows in [1, 3, 8, 1] {
            let x: Vec<f32> = (0..rows * 5120)
                .map(|i| ((i * 37 + i / 17) % 4093) as f32 / 4093. - 0.5)
                .collect();
            let x = Array::from_slice(&x, &[rows, 5120]);
            stream.runtime().commands.flush_and_wait()?;
            let before = counts();
            let old = original.forward(&x, &stream)?;
            stream.runtime().commands.flush_and_wait()?;
            let middle = counts();
            let new = migrated.forward(&x)?;
            stream.runtime().commands.flush_and_wait()?;
            let after = counts();
            for (name, a, b) in [
                ("shared", &old.shared, &new.shared),
                ("routed", &old.routed, &new.routed),
                (
                    "probabilities",
                    &old.routing.probabilities,
                    &new.routing.probabilities,
                ),
                ("selected", &old.routing.selected, &new.routing.selected),
                ("route weights", &old.routing.weights, &new.routing.weights),
            ] {
                assert_eq!(a.shape(), b.shape(), "{name}");
                assert_eq!(
                    a.to_vec1::<u32>()?,
                    b.to_vec1::<u32>()?,
                    "{name}, rows={rows}, encodings={kinds:?}"
                );
            }
            if !after.is_empty() {
                let reference = difference(&before, &middle);
                let actual = difference(&middle, &after);
                assert_eq!(reference, actual, "dispatch census rows={rows}");
                eprintln!(
                    "packed FFN trace rows={rows}: {} dispatches, {} variants",
                    actual.values().sum::<u64>(),
                    actual.len()
                );
            }
        }
        stream.runtime().commands.flush_and_wait()?;
    }
    Ok(())
}

pub(crate) fn projection_check(
    stream: &Stream,
    old: impl FnOnce() -> Result<Array>,
    new: impl FnOnce() -> Result<Array>,
) -> Result<()> {
    stream.runtime().commands.flush_and_wait()?;
    let before = counts();
    let a = old()?;
    stream.runtime().commands.flush_and_wait()?;
    let middle = counts();
    let b = new()?;
    stream.runtime().commands.flush_and_wait()?;
    let after = counts();
    assert_eq!(a.shape(), b.shape());
    assert_eq!(a.to_vec1::<u32>()?, b.to_vec1::<u32>()?);
    if !after.is_empty() {
        assert_eq!(difference(&before, &middle), difference(&middle, &after));
    }
    Ok(())
}
#[test]
fn packed_projection_plans_preserve_slices_pairing_and_group_order() -> Result<()> {
    use crate::bindings::deepseek41_projection as new;
    use oracle::projection as old;
    let stream = Stream::gpu();
    for kind in [
        T::F32,
        T::F16,
        T::BF16,
        T::Q8_0,
        T::IQ2_XXS,
        T::IQ2_XS,
        T::IQ1_M,
        T::Q5_K,
    ] {
        let w = tensor(kind, &[24, 512], 181)?;
        for rows in [1, 3, 8] {
            let data: Vec<f32> = (0..rows * 512)
                .map(|i| (i % 251) as f32 / 251. - 0.5)
                .collect();
            let x = Array::from_slice(&data, &[rows, 512]);
            for (first, count) in [(0, 24), (8, 8)] {
                for round in [false, true] {
                    projection_check(
                        &stream,
                        || old::linear_rows(&w, first, count, &x, round, &stream),
                        || new::linear_rows(&w, first, count, &x, round, &stream),
                    )?;
                }
            }
        }
    }
    for second in [T::Q8_0, T::BF16] {
        let a = tensor(T::Q8_0, &[24, 512], 182)?;
        let b = tensor(second, &[32, 512], 183)?;
        for rows in [1, 3, 8] {
            let x = Array::from_slice(
                &(0..rows * 512)
                    .map(|i| (i % 251) as f32 / 251. - 0.5)
                    .collect::<Vec<_>>(),
                &[rows, 512],
            );
            stream.runtime().commands.flush_and_wait()?;
            let before = counts();
            let expected = old::query_kv_pair(&a, &b, &x, &stream)?;
            stream.runtime().commands.flush_and_wait()?;
            let middle = counts();
            let actual = new::query_kv_pair(&a, &b, &x, &stream)?;
            stream.runtime().commands.flush_and_wait()?;
            let after = counts();
            assert_eq!(expected.0.to_vec1::<u32>()?, actual.0.to_vec1::<u32>()?);
            assert_eq!(expected.1.to_vec1::<u32>()?, actual.1.to_vec1::<u32>()?);
            if !after.is_empty() {
                assert_eq!(difference(&before, &middle), difference(&middle, &after));
            }
        }
    }
    for kind in [T::Q5_K, T::Q8_0, T::F16] {
        let w = tensor(kind, &[8192, 4096], 184)?;
        let x = Array::from_slice(
            &(0..64 * 512)
                .map(|i| (i % 4093) as f32 / 4093. - 0.5)
                .collect::<Vec<_>>(),
            &[1, 64, 512],
        );
        projection_check(
            &stream,
            || old::attention_output_groups(&w, &x, &stream),
            || new::attention_output_groups(&w, &x, &stream),
        )?;
    }
    Ok(())
}
