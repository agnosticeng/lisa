//! Native DS4 Q8 schedules over original MLX affine storage.

use super::{DType, Device, Tensor};
use crate::array::Array;
use crate::error::Result;
use crate::runtime::{ConstVal, Math};

const MV_SOURCE: &str = include_str!("../shaders/common/matmul/glm53_affine_q8_mv_native.metal");
const EXT_SOURCE: &str =
    include_str!("../shaders/common/matmul/glm53_affine_q8_ext_small_native.metal");

/// Preserve upstream's M=1 block-dot and M=2..16 vector-dot dispatch gates.
/// GLM chooses this path after the corresponding native fixture and complete
/// model-logit comparisons; other models do not call this wrapper.
pub fn glm53_affine_q8_tiny_native(
    device: &Device,
    x: &Tensor,
    weights: &Tensor,
    scales: &Tensor,
    biases: &Tensor,
) -> Result<Tensor> {
    if [x, weights, scales, biases].iter().any(|t| t.rank() != 2) {
        crate::bail!("glm53_affine_q8_tiny_native requires rank-2 tensors");
    }
    let (m, k) = x.dims2();
    let (n, packed_k) = weights.dims2();
    if !(1..=16).contains(&m)
        || n == 0
        || k == 0
        || k % 64 != 0
        || (m > 1 && k % 128 != 0)
        || packed_k.checked_mul(4) != Some(k)
        || x.dtype() != DType::F32
        || weights.dtype() != DType::U32
        || !matches!(scales.dtype(), DType::F32 | DType::BF16)
        || biases.dtype() != scales.dtype()
        || scales.dims2() != (n, k / 64)
        || biases.dims2() != (n, k / 64)
        || [m, k, n].into_iter().any(|d| d > i32::MAX as usize)
    {
        crate::bail!("unsupported native tiny Q8 shape/dtype: M={m}, K={k}, N={n}");
    }
    for tensor in [x, weights, scales, biases] {
        if !tensor.buffer_and_layout().1.is_contiguous() {
            crate::bail!("native tiny Q8 requires contiguous inputs");
        }
    }
    let (pipeline, groups, nsg, shared_bytes) = if m == 1 {
        let (name, nsg) = if n > 65536 {
            ("glm53_affine_q8_mv_native_nsg8", 8)
        } else {
            ("glm53_affine_q8_mv_native_nsg4", 4)
        };
        (
            device.compile_with(MV_SOURCE, name, Math::CompilerDefault)?,
            (n.div_ceil(2), 1, 1),
            nsg,
            32 * 2 * 4,
        )
    } else {
        let nxpsg = if k % 256 == 0 && m == 2 { 16 } else { 8 };
        let r1ptg = match m {
            2 => 2,
            3 | 6 => 3,
            5 => 5,
            _ => 4,
        };
        let name = format!("glm53_affine_q8_ext_small_native_r{r1ptg}");
        (
            device.compile_with_constants(
                EXT_SOURCE,
                &name,
                Math::CompilerDefault,
                &[(0, ConstVal::Int(2)), (1, ConstVal::Int(nxpsg))],
            )?,
            (n.div_ceil((32 / nxpsg as usize) * 2), m.div_ceil(r1ptg), 1),
            2,
            0,
        )
    };
    let output_bytes = m
        .checked_mul(n)
        .and_then(|elements| elements.checked_mul(4))
        .ok_or_else(|| crate::error::Error::msg("native tiny Q8 output size overflow"))?;
    let output_buffer = device.buffer_for_full_gpu_write(output_bytes, "glm53_q8_tiny_native")?;
    let output = Array::from_parts(device, output_buffer.clone(), &[m, n], DType::F32);
    let guard = device.commands.encoder()?;
    let encoder = guard.encoder();
    encoder.set_pipeline(&pipeline);
    encoder.set_bytes(
        0,
        &[
            k as u32,
            n as u32,
            m as u32,
            u32::from(scales.dtype() == DType::BF16),
        ],
    );
    for (index, tensor) in [(1, weights), (2, x), (4, scales), (5, biases)] {
        let (buffer, layout) = tensor.buffer_and_layout();
        encoder.set_input(
            index,
            Some(buffer),
            layout.offset * tensor.dtype().size_of(),
        );
    }
    encoder.set_output(3, Some(&output_buffer), 0);
    if shared_bytes != 0 {
        encoder.set_threadgroup_memory_length(0, shared_bytes);
    }
    encoder.dispatch_groups(groups, (32, nsg, 1));
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn glm53_affine_q8_tiny_matches_native_ds4_fixtures() {
        let Some(root) = std::env::var_os("LISA_GLM53_Q8_NATIVE_FIXTURE_DIR") else {
            eprintln!("[glm53-q8-tiny] skipping: LISA_GLM53_Q8_NATIVE_FIXTURE_DIR is unset");
            return;
        };
        if !crate::device::metal_available() {
            return;
        }
        let root = std::path::Path::new(&root);
        let mut cases = [1usize, 2, 3, 4, 5, 6, 8, 9, 16]
            .into_iter()
            .map(|m| (m, 4096usize, 4096usize))
            .collect::<Vec<_>>();
        cases.extend([(1, 128, 65536), (1, 128, 65538), (1, 2048, 65538)]);
        let scale = half::f16::from_bits(0x3580).to_f32();
        assert_eq!(scale, 0.34375);
        for (m, k, n) in cases {
            let path = root.join(format!("q8-native-tiny-{m}-{k}-{n}.f32"));
            let expected =
                std::fs::read(&path).unwrap_or_else(|error| panic!("{}: {error}", path.display()));
            assert_eq!(expected.len(), m * n * 4, "native fixture shape");
            let mut packed = vec![0u32; n * k / 4];
            for row in 0..n {
                for col in 0..k {
                    let signed = ((row * 13 + (col / 32) * 17 + (col % 32) * 7) % 9) as i32 - 4;
                    packed[row * (k / 4) + col / 4] |= ((signed + 128) as u32) << ((col % 4) * 8);
                }
            }
            let input = (0..m)
                .flat_map(|token| {
                    (0..k).map(move |col| {
                        (((token * 31 + col * 17) % 257) as f32 - 128.0) * 0.00037f32
                    })
                })
                .collect::<Vec<_>>();
            let input = crate::Array::from_slice(&input, &[m as i32, k as i32]);
            let weights = crate::Array::from_slice(&packed, &[n as i32, (k / 4) as i32]);
            let scales =
                crate::Array::from_slice(&vec![scale; n * k / 64], &[n as i32, (k / 64) as i32]);
            let biases = crate::Array::from_slice(
                &vec![-128.0 * scale; n * k / 64],
                &[n as i32, (k / 64) as i32],
            );
            for dtype in [crate::Dtype::Float32, crate::Dtype::Bfloat16] {
                let scales = scales.as_dtype(dtype).unwrap();
                let biases = biases.as_dtype(dtype).unwrap();
                let output = glm53_affine_q8_tiny_native(
                    input.t.device(),
                    &input.t,
                    &weights.t,
                    &scales.t,
                    &biases.t,
                )
                .expect("native tiny Q8 dispatch");
                let actual = crate::Array::new(output).to_vec1::<f32>().unwrap();
                let mismatches = actual
                    .iter()
                    .zip(expected.chunks_exact(4))
                    .filter(|(value, bytes)| {
                        !value.is_finite()
                            || value.to_bits() != u32::from_le_bytes((*bytes).try_into().unwrap())
                    })
                    .count();
                eprintln!(
                    "[glm53-q8-tiny] M={m} K={k} N={n} metadata={dtype:?} exact={}/{}",
                    actual.len() - mismatches,
                    actual.len()
                );
                assert_eq!(mismatches, 0, "native Q8 M={m} K={k} N={n} dtype={dtype:?}");
            }
        }
    }

    #[test]
    fn glm53_affine_q8_pair_matches_affine_residual_fixtures() {
        let Some(root) = std::env::var_os("LISA_GLM53_Q8_NATIVE_FIXTURE_DIR") else {
            return;
        };
        if !crate::device::metal_available() {
            return;
        }
        const K: usize = 4096;
        const N: usize = 2048;
        const GROUPS: usize = K / 64;
        let root = std::path::Path::new(&root);
        let mut packed = vec![0u32; 2 * N * K / 4];
        let mut scale_values = vec![0.0f32; 2 * N * GROUPS];
        let mut bias_values = vec![0.0f32; 2 * N * GROUPS];
        for bank in 0..2 {
            for row in 0..N {
                for col in 0..K {
                    let signed = if bank == 0 {
                        ((row * 13 + (col / 32) * 17 + (col % 32) * 7) % 9) as i32 - 4
                    } else {
                        ((row * 11 + (col / 32) * 19 + (col % 32) * 5) % 11) as i32 - 5
                    };
                    packed[(bank * N + row) * (K / 4) + col / 4] |=
                        ((signed + 128) as u32) << ((col % 4) * 8);
                }
                for group in 0..GROUPS {
                    let scale = 0.25 + ((row * 3 + group * 5 + bank * 7) % 8) as f32 * 0.03125;
                    let sign = if (row + group + bank) % 2 == 0 {
                        1.0
                    } else {
                        -1.0
                    };
                    let residual = (1 + (row + group * 3 + bank) % 3) as f32 * 0.5 * sign;
                    let index = (bank * N + row) * GROUPS + group;
                    scale_values[index] = scale;
                    bias_values[index] = -128.0 * scale + residual;
                }
            }
        }
        let weights = crate::Array::from_slice(&packed, &[(2 * N) as i32, (K / 4) as i32]);
        let gate_w = crate::Array::new(weights.t.narrow(0, 0, N).unwrap());
        let up_w = crate::Array::new(weights.t.narrow(0, N, N).unwrap());
        let scales = crate::Array::from_slice(&scale_values, &[(2 * N) as i32, GROUPS as i32]);
        let biases = crate::Array::from_slice(&bias_values, &[(2 * N) as i32, GROUPS as i32]);
        for (dtype_tag, dtype) in [
            ("bf16", crate::Dtype::Bfloat16),
            ("f32", crate::Dtype::Float32),
        ] {
            let scales = scales.as_dtype(dtype).unwrap();
            let biases = biases.as_dtype(dtype).unwrap();
            let gate_s = crate::Array::new(scales.t.narrow(0, 0, N).unwrap());
            let gate_b = crate::Array::new(biases.t.narrow(0, 0, N).unwrap());
            let up_s = crate::Array::new(scales.t.narrow(0, N, N).unwrap());
            let up_b = crate::Array::new(biases.t.narrow(0, N, N).unwrap());
            for rows in [1usize, 2, 16] {
                let values = (0..rows)
                    .flat_map(|token| {
                        (0..K).map(move |col| {
                            (((token * 31 + col * 17) % 257) as f32 - 128.0) * 0.00037
                        })
                    })
                    .collect::<Vec<_>>();
                let input = crate::Array::from_slice(&values, &[rows as i32, K as i32]);
                for (clamp_tag, clamp) in [("0", 0.0f32), ("005", 0.05)] {
                    let (gate, up, mid) = crate::jit::glm53_affine_q8_gate_up_native(
                        input.t.device(),
                        &input.t,
                        &gate_w.t,
                        &gate_s.t,
                        &gate_b.t,
                        &up_w.t,
                        &up_s.t,
                        &up_b.t,
                        clamp,
                    )
                    .expect("affine residual gate/up dispatch");
                    for (stage, tensor) in [("gate", gate), ("up", up), ("mid", mid)] {
                        let path = root.join(format!(
                            "q8-affine-pair-residual-{rows}-{dtype_tag}-clamp{clamp_tag}.{stage}-affine.f32"
                        ));
                        let expected = std::fs::read(&path)
                            .unwrap_or_else(|error| panic!("{}: {error}", path.display()));
                        let actual = crate::Array::new(tensor).to_vec1::<f32>().unwrap();
                        assert_eq!(expected.len(), actual.len() * 4);
                        let mismatches = actual
                            .iter()
                            .zip(expected.chunks_exact(4))
                            .filter(|(value, bytes)| {
                                !value.is_finite()
                                    || value.to_bits()
                                        != u32::from_le_bytes((*bytes).try_into().unwrap())
                            })
                            .count();
                        eprintln!(
                            "[glm53-q8-pair-residual] M={rows} dtype={dtype_tag} clamp={clamp} stage={stage} exact={}/{}",
                            actual.len() - mismatches,
                            actual.len()
                        );
                        assert_eq!(
                            mismatches, 0,
                            "M={rows} dtype={dtype_tag} clamp={clamp} stage={stage}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn glm53_affine_q8_pair_matches_native_ds4_fixtures_and_weight_views() {
        let Some(root) = std::env::var_os("LISA_GLM53_Q8_NATIVE_FIXTURE_DIR") else {
            eprintln!("[glm53-q8-pair] skipping: LISA_GLM53_Q8_NATIVE_FIXTURE_DIR is unset");
            return;
        };
        if !crate::device::metal_available() {
            return;
        }
        const K: usize = 4096;
        const N: usize = 2048;
        let root = std::path::Path::new(&root);
        let mut packed = vec![0u32; 2 * N * K / 4];
        for bank in 0..2 {
            for row in 0..N {
                for col in 0..K {
                    let signed = if bank == 0 {
                        ((row * 13 + (col / 32) * 17 + (col % 32) * 7) % 9) as i32 - 4
                    } else {
                        ((row * 11 + (col / 32) * 19 + (col % 32) * 5) % 11) as i32 - 5
                    };
                    packed[(bank * N + row) * (K / 4) + col / 4] |=
                        ((signed + 128) as u32) << ((col % 4) * 8);
                }
            }
        }
        let weights = crate::Array::from_slice(&packed, &[(2 * N) as i32, (K / 4) as i32]);
        let gate_w = crate::Array::new(weights.t.narrow(0, 0, N).unwrap());
        let up_w = crate::Array::new(weights.t.narrow(0, N, N).unwrap());
        let scale_values = [0x3580u16, 0x3480]
            .into_iter()
            .flat_map(|bits| std::iter::repeat_n(half::f16::from_bits(bits).to_f32(), N * K / 64))
            .collect::<Vec<_>>();
        let bias_values = scale_values.iter().map(|s| -128.0 * s).collect::<Vec<_>>();
        let scales = crate::Array::from_slice(&scale_values, &[(2 * N) as i32, (K / 64) as i32]);
        let biases = crate::Array::from_slice(&bias_values, &[(2 * N) as i32, (K / 64) as i32]);
        for dtype in [crate::Dtype::Float32, crate::Dtype::Bfloat16] {
            let scales = scales.as_dtype(dtype).unwrap();
            let biases = biases.as_dtype(dtype).unwrap();
            let gate_s = crate::Array::new(scales.t.narrow(0, 0, N).unwrap());
            let gate_b = crate::Array::new(biases.t.narrow(0, 0, N).unwrap());
            let up_s = crate::Array::new(scales.t.narrow(0, N, N).unwrap());
            let up_b = crate::Array::new(biases.t.narrow(0, N, N).unwrap());
            for m in [1usize, 2, 3, 4, 5, 6, 8, 9, 16] {
                let values = (0..m)
                    .flat_map(|token| {
                        (0..K).map(move |col| {
                            (((token * 31 + col * 17) % 257) as f32 - 128.0) * 0.00037f32
                        })
                    })
                    .collect::<Vec<_>>();
                let input = crate::Array::from_slice(&values, &[m as i32, K as i32]);
                for (tag, clamp) in [("0", 0.0f32), ("005", 0.05)] {
                    let (gate, up, mid) = crate::jit::glm53_affine_q8_gate_up_native(
                        input.t.device(),
                        &input.t,
                        &gate_w.t,
                        &gate_s.t,
                        &gate_b.t,
                        &up_w.t,
                        &up_s.t,
                        &up_b.t,
                        clamp,
                    )
                    .expect("native fused gate/up dispatch");
                    for (stage, tensor) in [("gate", gate), ("up", up), ("mid", mid)] {
                        let path =
                            root.join(format!("q8-native-pair-{m}-clamp{tag}.{stage}-native.f32"));
                        let expected = std::fs::read(&path)
                            .unwrap_or_else(|error| panic!("{}: {error}", path.display()));
                        let actual = crate::Array::new(tensor).to_vec1::<f32>().unwrap();
                        assert_eq!(expected.len(), actual.len() * 4);
                        let mismatches = actual
                            .iter()
                            .zip(expected.chunks_exact(4))
                            .filter(|(value, bytes)| {
                                !value.is_finite()
                                    || value.to_bits()
                                        != u32::from_le_bytes((*bytes).try_into().unwrap())
                            })
                            .count();
                        eprintln!(
                            "[glm53-q8-pair] M={m} clamp={clamp} dtype={dtype:?} stage={stage} exact={}/{}",
                            actual.len() - mismatches,
                            actual.len()
                        );
                        assert_eq!(
                            mismatches, 0,
                            "M={m} clamp={clamp} dtype={dtype:?} stage={stage}"
                        );
                    }
                }
            }
        }
    }
}
