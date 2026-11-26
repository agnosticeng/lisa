#![cfg(test)]
//! Packed-weight integration checks against the GGML-validated CPU decoder.
//! Dyadic inputs make these sums exact in f32; arbitrary inputs additionally
//! require comparisons against the original Metal reduction, not this dot.

use lisa_gguf::{TensorType, quant};
use lisa_metal::kernels::ggml::{self as gpu, BlockFormat};
use lisa_metal::{Array, Dtype, Stream};

const FORMATS: [(TensorType, BlockFormat); 7] = [
    (TensorType::IQ2_XXS, BlockFormat::Iq2Xxs),
    (TensorType::IQ2_XS, BlockFormat::Iq2Xs),
    (TensorType::IQ1_M, BlockFormat::Iq1M),
    (TensorType::Q5_K, BlockFormat::Q5K),
    (TensorType::Q6_K, BlockFormat::Q6K),
    (TensorType::Q2_K, BlockFormat::Q2K),
    (TensorType::Q4_K, BlockFormat::Q4K),
];

fn blocks(kind: TensorType, count: usize) -> Vec<u8> {
    let size = kind.layout().unwrap().2 as usize;
    let mut output = vec![0; size * count];
    let mut state = 0x12345678u32;
    for block in output.chunks_exact_mut(size) {
        for byte in block.iter_mut() {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            *byte = (state >> 24) as u8;
        }
        if kind == TensorType::Q6_K {
            block[208..210].copy_from_slice(&0x1800u16.to_le_bytes());
        } else if kind == TensorType::Q2_K {
            block[80..82].copy_from_slice(&0x1800u16.to_le_bytes());
            block[82..84].copy_from_slice(&0x1400u16.to_le_bytes());
        } else if kind == TensorType::IQ1_M {
            for i in 0..4 {
                let at = 48 + 2 * i;
                let value = u16::from_le_bytes([block[at], block[at + 1]]) & 0xfff;
                let value = value | (((0x1800u16 >> (4 * i)) & 15) << 12);
                block[at..at + 2].copy_from_slice(&value.to_le_bytes());
            }
        } else {
            block[..2].copy_from_slice(&0x1800u16.to_le_bytes());
            if matches!(kind, TensorType::Q5_K | TensorType::Q4_K) {
                block[2..4].copy_from_slice(&0x1400u16.to_le_bytes());
            }
        }
    }
    output
}

fn packed_view(bytes: &[u8], shape: &[i32]) -> Array {
    let mut storage = vec![0u8; 2];
    storage.extend_from_slice(bytes);
    let storage = Array::from_slice(&storage, &[storage.len() as i32]);
    let (buffer, _) = storage.t.buffer_and_layout();
    Array::from_buffer_offset(buffer.clone(), 2, shape, Dtype::Uint8).unwrap()
}

fn inputs(count: usize) -> Vec<f32> {
    (0..count)
        .map(|i| ((i * 19 + i / 7) % 3) as f32 * 0.125 - 0.125)
        .collect()
}

#[test]
fn deepseek41_native_quantized_dense_matches_exact_dyadic_dot() {
    let stream = Stream::gpu();
    for (kind, format) in FORMATS {
        for width in [256, 1280, 2304, 5120, 8192] {
            let rows = 16;
            let tokens = 3;
            let bytes = blocks(kind, width * rows / 256);
            let row_bytes = bytes.len() / rows;
            let weights = packed_view(&bytes, &[rows as i32, row_bytes as i32]);
            let x = inputs(tokens * width);
            let x_gpu = Array::from_slice(&x, &[tokens as i32, width as i32]);
            let result = gpu::packed_matvec(format, &weights, &x_gpu, &stream).unwrap();
            let mut decoded = vec![0.0; rows * width];
            quant::decode(kind, &bytes, &mut decoded).unwrap();
            let actual = result.as_slice::<f32>();
            for token in 0..tokens {
                for row in 0..rows {
                    let expected: f32 = decoded[row * width..(row + 1) * width]
                        .iter()
                        .zip(&x[token * width..(token + 1) * width])
                        .map(|(w, x)| w * x)
                        .sum();
                    assert_eq!(
                        actual[token * rows + row],
                        expected,
                        "{kind:?} width={width} row={row} token={token}"
                    );
                }
            }
        }
    }
}

#[test]
fn deepseek41_native_expert_ids_and_broadcast_match_dense() {
    let stream = Stream::gpu();
    for (kind, format) in FORMATS {
        for experts in [192, 384] {
            let (width, rows, tokens, top_k) = (256, 8, 3, 6);
            let bytes = blocks(kind, width * rows * experts / 256);
            let row_bytes = bytes.len() / (rows * experts);
            let weights = packed_view(&bytes, &[experts as i32, rows as i32, row_bytes as i32]);
            let selected: Vec<i32> = (0..tokens * top_k)
                .map(|i| ((i * 79 + 1) % experts) as i32)
                .collect();
            let mut selected = selected;
            selected[0] = experts as i32 - 1;
            selected[1] = -1;
            selected[2] = experts as i32;
            let ids = Array::from_slice(&selected, &[tokens as i32, top_k as i32]);
            let mut decoded = vec![0.0; experts * rows * width];
            quant::decode(kind, &bytes, &mut decoded).unwrap();
            for slots in [1, top_k] {
                let x = inputs(tokens * slots * width);
                let x_gpu = Array::from_slice(&x, &[tokens as i32, slots as i32, width as i32]);
                let result = gpu::expert_matvec(format, &weights, &x_gpu, &ids, &stream).unwrap();
                let actual = result.as_slice::<f32>();
                for token in 0..tokens {
                    for slot in 0..top_k {
                        let expert = selected[token * top_k + slot];
                        let begin = (token * slots + slot % slots) * width;
                        for row in 0..rows {
                            let expected = if expert < 0 || expert >= experts as i32 {
                                0.0
                            } else {
                                let at = (expert as usize * rows + row) * width;
                                decoded[at..at + width]
                                    .iter()
                                    .zip(&x[begin..begin + width])
                                    .map(|(w, x)| w * x)
                                    .sum::<f32>()
                            };
                            assert_eq!(
                                actual[(token * top_k + slot) * rows + row],
                                expected,
                                "{kind:?} experts={experts} token={token} slot={slot} row={row} slots={slots}"
                            );
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn deepseek41_grouped_projection_matches_separate_logit_bits() {
    let stream = Stream::gpu();
    for (kind, format) in FORMATS {
        // The full Q5 shape exercises the checkpoint's eight O_A groups.
        let shapes = if kind == TensorType::Q5_K {
            vec![(8, 1024, 4096, 1), (3, 16, 1280, 3)]
        } else {
            vec![(3, 16, 1280, 3)]
        };
        for (groups, rows, width, tokens) in shapes {
            let bytes = blocks(kind, groups * rows * width / 256);
            let row_bytes = bytes.len() / (groups * rows);
            let packed = packed_view(&bytes, &[groups as i32, rows as i32, row_bytes as i32]);
            // Non-dyadic data checks reduction order, beyond exact-sum fixtures.
            let x: Vec<f32> = (0..groups * tokens * width)
                .map(|i| ((i * 131 + i / 13) % 4093) as f32 / 4093.0 - 0.5)
                .collect();
            let input = Array::from_slice(&x, &[groups as i32, tokens as i32, width as i32]);
            let grouped = gpu::grouped_packed_matvec(format, &packed, &input, &stream).unwrap();
            let actual = grouped.as_slice::<f32>();
            for group in 0..groups {
                let (buffer, layout) = packed.t.buffer_and_layout();
                let weights = Array::from_buffer_offset(
                    buffer.clone(),
                    layout.offset + group * rows * row_bytes,
                    &[rows as i32, row_bytes as i32],
                    Dtype::Uint8,
                )
                .unwrap();
                let (buffer, layout) = input.t.buffer_and_layout();
                let one = Array::from_buffer_offset(
                    buffer.clone(),
                    (layout.offset + group * tokens * width) * 4,
                    &[tokens as i32, width as i32],
                    Dtype::Float32,
                )
                .unwrap();
                let separate = gpu::packed_matvec(format, &weights, &one, &stream).unwrap();
                assert!(
                    actual[group * tokens * rows..(group + 1) * tokens * rows]
                        .iter()
                        .zip(separate.as_slice::<f32>())
                        .all(|(a, b)| a.to_bits() == b.to_bits()),
                    "grouped {kind:?} group={group} rows={rows} tokens={tokens}"
                );
            }
        }
    }
}
