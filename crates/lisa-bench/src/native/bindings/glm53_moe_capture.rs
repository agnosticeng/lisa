#![cfg(test)]
#![cfg(feature = "glm53-fixtures")]
use super::*;
impl MoE {
    /// Re-run the GLM router for the MTP DS4 differential without changing
    /// the production MoE dispatch or its fused down-sum path.
    #[cfg(all(test, feature = "glm53-fixtures"))]
    pub(crate) fn capture_or_assert_mtp_router_ds4(&self, x: &Array) -> anyhow::Result<()> {
        const HIDDEN: usize = 4096;
        const EXPERTS: usize = 288;
        const TOP_K: usize = 8;
        let external_fixture: &[u8] =
            &crate::bindings::glm53_test_fixtures::read("mtp_router_layer45_ds4.bin");
        anyhow::ensure!(
            x.dim(x.rank() as i32 - 1) == HIDDEN as i32
                && self.gate_w.shape() == [EXPERTS as i32, HIDDEN as i32]
                && self.bias.size() == EXPERTS
                && self.top_k == TOP_K
                && self.n_group == 1
                && self.topk_group == 1,
            "layer-45 MTP router does not match the DS4 fixture geometry"
        );
        let input = x.as_dtype(Dtype::Float32)?.reshape(&[1, HIDDEN as i32])?;
        let stream = lisa_metal::Stream::thread_local_or_default();
        let logits =
            lisa_metal::kernels::sparse::kernels::router_matvec_f32(&self.gate_w, &input, &stream)
                .ok_or_else(|| anyhow::anyhow!("GLM MTP router projection declined"))?;
        let (selected, weights) = lisa_metal::kernels::sparse::kernels::router_select_fused(
            &logits,
            &self.bias,
            TOP_K,
            self.routed_scaling,
            &stream,
        )
        .ok_or_else(|| anyhow::anyhow!("GLM MTP router selection declined"))?;
        let input_values = input.to_vec1::<f32>()?;
        let router_weight_values = self.gate_w.to_vec1::<f32>()?;
        let router_bias_values = self.bias.to_vec1::<f32>()?;
        let logits_values = logits.to_vec1::<f32>()?;
        let selected_values = selected.to_vec1::<i32>()?;
        let weight_values = weights.to_vec1::<f32>()?;
        anyhow::ensure!(
            input_values.len() == HIDDEN
                && router_weight_values.len() == EXPERTS * HIDDEN
                && router_bias_values.len() == EXPERTS
                && logits_values.len() == EXPERTS
                && selected_values.len() == TOP_K
                && weight_values.len() == TOP_K,
            "layer-45 MTP router output shape mismatch"
        );

        if let Some(path) = std::env::var_os("LISA_GLM53_MTP_ROUTER_CAPTURE_DIR") {
            let path = std::path::Path::new(&path);
            std::fs::create_dir_all(path)?;
            for (name, data) in [
                ("input.f32", encode_f32(&input_values)),
                ("router_weight.f32", encode_f32(&router_weight_values)),
                ("router_bias.f32", encode_f32(&router_bias_values)),
                ("lisa_logits.f32", encode_f32(&logits_values)),
                ("lisa_weights.f32", encode_f32(&weight_values)),
            ] {
                std::fs::write(path.join(name), data)?;
            }
            std::fs::write(path.join("lisa_selected.i32"), encode_i32(&selected_values))?;
            return Ok(());
        }

        const F32_BYTES: usize = (HIDDEN + EXPERTS * HIDDEN + 2 * EXPERTS + TOP_K) * 4;
        const SELECTED_BYTES: usize = TOP_K * 4;
        anyhow::ensure!(
            external_fixture.len() == F32_BYTES + SELECTED_BYTES,
            "DS4 layer-45 MTP router fixture has an invalid size"
        );
        let mut offset = 0usize;
        let mut expected_f32 = Vec::with_capacity(5);
        for count in [HIDDEN, EXPERTS * HIDDEN, EXPERTS, EXPERTS, TOP_K] {
            let end = offset + count * 4;
            expected_f32.push(
                external_fixture[offset..end]
                    .chunks_exact(4)
                    .map(|bytes| f32::from_le_bytes(bytes.try_into().expect("F32 word")))
                    .collect::<Vec<_>>(),
            );
            offset = end;
        }
        let expected_selected = external_fixture[offset..]
            .chunks_exact(4)
            .map(|bytes| i32::from_le_bytes(bytes.try_into().expect("I32 id")))
            .collect::<Vec<_>>();
        for (name, actual, expected) in [
            ("input", &input_values, &expected_f32[0]),
            ("router weight", &router_weight_values, &expected_f32[1]),
            ("router bias", &router_bias_values, &expected_f32[2]),
        ] {
            anyhow::ensure!(
                actual
                    .iter()
                    .map(|v| v.to_bits())
                    .eq(expected.iter().map(|v| v.to_bits())),
                "layer-45 MTP DS4 {name} fixture input changed"
            );
        }
        anyhow::ensure!(
            selected_values == expected_selected,
            "layer-45 MTP router expert IDs differ from DS4"
        );
        let logits_max_abs = logits_values
            .iter()
            .zip(&expected_f32[3])
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0f32, f32::max);
        let weight_max_abs = weight_values
            .iter()
            .zip(&expected_f32[4])
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0f32, f32::max);
        anyhow::ensure!(
            logits_max_abs <= 2e-5 && weight_max_abs <= 1e-6,
            "layer-45 MTP router differs from DS4: logits max_abs={logits_max_abs:e}, weights max_abs={weight_max_abs:e}"
        );
        eprintln!(
            "[glm53-mtp-router] selected IDs match DS4; logits max_abs={logits_max_abs:.3e}; weights max_abs={weight_max_abs:.3e}"
        );
        Ok(())
    }

    /// Compare the MTP shared expert against DS4 using the checkpoint's
    /// MLX-affine weights dequantized by the direct Metal reference harness.
    #[cfg(all(test, feature = "glm53-fixtures"))]
    pub(crate) fn capture_or_assert_mtp_shared_ds4(
        &self,
        x: &Array,
        actual: &Array,
    ) -> anyhow::Result<()> {
        const HIDDEN: usize = 4096;
        let external_fixture: &[u8] =
            &crate::bindings::glm53_test_fixtures::read("mtp_shared_layer45_ds4.bin");
        let shared = self
            .shared
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("GLM MTP shared expert is missing"))?;
        let inter = shared.gate_up.inter();
        let (gate, up) = shared.gate_up.split_linears();
        let hidden = HIDDEN;
        let group = usize::try_from(gate.group_size)?;
        let bits = usize::try_from(gate.bits)?;
        anyhow::ensure!(
            hidden == x.dim(x.rank() as i32 - 1) as usize
                && inter > 0
                && group == 64
                && bits == 8
                && up.dims_out() == inter
                && shared.down.dims_out() == hidden
                && gate.group_size == up.group_size
                && gate.group_size == shared.down.group_size
                && gate.bits == up.bits
                && gate.bits == shared.down.bits,
            "layer-45 MTP shared expert geometry differs from the DS4 fixture"
        );

        let input = x.as_dtype(Dtype::Float32)?.to_vec1::<f32>()?;
        let output = actual.as_dtype(Dtype::Float32)?.to_vec1::<f32>()?;
        let gate_weight = gate.weight.to_vec1::<u32>()?;
        let gate_scales = gate.scales.to_vec1::<f32>()?;
        let gate_biases = gate.biases.to_vec1::<f32>()?;
        let up_weight = up.weight.to_vec1::<u32>()?;
        let up_scales = up.scales.to_vec1::<f32>()?;
        let up_biases = up.biases.to_vec1::<f32>()?;
        let down_weight = shared.down.weight.to_vec1::<u32>()?;
        let down_scales = shared.down.scales.to_vec1::<f32>()?;
        let down_biases = shared.down.biases.to_vec1::<f32>()?;
        anyhow::ensure!(
            input.len() == hidden
                && output.len() == hidden
                && gate_weight.len() == inter * hidden * bits / 32
                && up_weight.len() == gate_weight.len()
                && down_weight.len() == hidden * inter * bits / 32
                && gate_scales.len() == inter * hidden / group
                && gate_biases.len() == gate_scales.len()
                && up_scales.len() == gate_scales.len()
                && up_biases.len() == gate_scales.len()
                && down_scales.len() == hidden * inter / group
                && down_biases.len() == down_scales.len(),
            "layer-45 MTP shared expert tensor sizes do not match its geometry"
        );

        let mut prefix = Vec::new();
        for value in [hidden as u32, inter as u32, group as u32, bits as u32] {
            prefix.extend_from_slice(&value.to_le_bytes());
        }
        prefix.extend_from_slice(&encode_f32(&input));
        prefix.extend_from_slice(&encode_u32(&gate_weight));
        prefix.extend_from_slice(&encode_f32(&gate_scales));
        prefix.extend_from_slice(&encode_f32(&gate_biases));
        prefix.extend_from_slice(&encode_u32(&up_weight));
        prefix.extend_from_slice(&encode_f32(&up_scales));
        prefix.extend_from_slice(&encode_f32(&up_biases));
        prefix.extend_from_slice(&encode_u32(&down_weight));
        prefix.extend_from_slice(&encode_f32(&down_scales));
        prefix.extend_from_slice(&encode_f32(&down_biases));

        if let Some(path) = std::env::var_os("LISA_GLM53_MTP_SHARED_CAPTURE_DIR") {
            let path = std::path::Path::new(&path);
            std::fs::create_dir_all(path)?;
            std::fs::write(
                path.join("geometry.u32"),
                [hidden as u32, inter as u32, group as u32, bits as u32]
                    .into_iter()
                    .flat_map(u32::to_le_bytes)
                    .collect::<Vec<_>>(),
            )?;
            std::fs::write(path.join("input.f32"), encode_f32(&input))?;
            std::fs::write(path.join("gate_weight.u32"), encode_u32(&gate_weight))?;
            std::fs::write(path.join("gate_scales.f32"), encode_f32(&gate_scales))?;
            std::fs::write(path.join("gate_biases.f32"), encode_f32(&gate_biases))?;
            std::fs::write(path.join("up_weight.u32"), encode_u32(&up_weight))?;
            std::fs::write(path.join("up_scales.f32"), encode_f32(&up_scales))?;
            std::fs::write(path.join("up_biases.f32"), encode_f32(&up_biases))?;
            std::fs::write(path.join("down_weight.u32"), encode_u32(&down_weight))?;
            std::fs::write(path.join("down_scales.f32"), encode_f32(&down_scales))?;
            std::fs::write(path.join("down_biases.f32"), encode_f32(&down_biases))?;
            std::fs::write(path.join("lisa_output.f32"), encode_f32(&output))?;
            return Ok(());
        }

        anyhow::ensure!(
            external_fixture.len() == prefix.len() + hidden * 4
                && external_fixture.starts_with(&prefix),
            "layer-45 MTP shared expert DS4 fixture inputs changed"
        );
        let expected = external_fixture[prefix.len()..]
            .chunks_exact(4)
            .map(|bytes| f32::from_le_bytes(bytes.try_into().expect("F32 output word")))
            .collect::<Vec<_>>();
        let expected_norm2 = expected
            .iter()
            .map(|value| f64::from(*value).powi(2))
            .sum::<f64>();

        // Exercise the shared Q8 down projection inside the DS4-default sum8
        // kernel against the direct DS4 branch fixture. Zero routed activations
        // isolate this branch while retaining the production address-table and
        // dispatch path.
        let gate_output = gate.forward(x)?;
        let up_output = up.forward(x)?;
        let shared_activation =
            clamped_swiglu(&up_output, &gate_output, shared.limit)?.reshape(&[1, inter as i32])?;
        let selected_ids = [0i32; 8];
        let selected = Array::from_slice(&selected_ids, &[selected_ids.len() as i32]);
        let routed_activation = Array::from_slice(
            &vec![0.0f32; selected_ids.len() * inter],
            &[selected_ids.len() as i32, inter as i32],
        );
        let selected_resources = if self.pools.residency_set.get().is_some() {
            Vec::new()
        } else {
            ExpertPools::indirect_resources(&self.pools.experts, &selected_ids)?
        };
        let keepalive: Vec<&Array> = selected_resources.iter().collect();
        let fused_shared =
            lisa_metal::kernels::sparse::kernels::moe_gather_qmv_sum8_addr_add_shared_down(
                &self.pools.address_table,
                &routed_activation,
                &selected,
                &keepalive,
                &shared_activation,
                &shared.down.weight,
                &shared.down.scales,
                &shared.down.biases,
                hidden,
                self.pools.metadata_dtype,
                self.pools.residency_set.get().map(Arc::as_ref),
                &lisa_metal::Stream::thread_local_or_default(),
            )
            .ok_or_else(|| anyhow::anyhow!("shared Q8 down/sum8 DS4 fixture kernel declined"))?
            .to_vec1::<f32>()?;
        let fused_stock_mismatches = fused_shared
            .iter()
            .zip(&output)
            .filter(|(fused, stock)| fused.to_bits() != stock.to_bits())
            .count();
        let fused_stock_max_abs = fused_shared
            .iter()
            .zip(&output)
            .map(|(fused, stock)| (fused - stock).abs())
            .fold(0.0f32, f32::max);
        eprintln!(
            "[glm53-mtp-shared-down-exact] mismatches={fused_stock_mismatches}/{} max_abs={fused_stock_max_abs:e}",
            output.len()
        );
        let fused_max_abs = fused_shared
            .iter()
            .zip(&expected)
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0f32, f32::max);
        let fused_delta2 = fused_shared
            .iter()
            .zip(&expected)
            .map(|(actual, expected)| f64::from(actual - expected).powi(2))
            .sum::<f64>();
        let fused_rel_l2 = (fused_delta2 / expected_norm2.max(f64::MIN_POSITIVE)).sqrt();
        anyhow::ensure!(
            fused_max_abs <= 1e-4 && fused_rel_l2 <= 1e-5,
            "fused layer-45 MTP shared expert differs from DS4: max_abs={fused_max_abs:e}, rel_l2={fused_rel_l2:e}"
        );
        eprintln!(
            "[glm53-mtp-shared-fused-down] output max_abs={fused_max_abs:.3e} rel_l2={fused_rel_l2:.3e} against DS4"
        );
        let mut delta2 = 0.0f64;
        let max_abs = output
            .iter()
            .zip(&expected)
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0f32, f32::max);
        for (actual, expected) in output.iter().zip(&expected) {
            delta2 += f64::from(actual - expected).powi(2);
        }
        let rel_l2 = (delta2 / expected_norm2.max(f64::MIN_POSITIVE)).sqrt();
        anyhow::ensure!(
            max_abs <= 1e-4 && rel_l2 <= 1e-5,
            "layer-45 MTP shared expert differs from DS4: max_abs={max_abs:e}, rel_l2={rel_l2:e}"
        );
        eprintln!(
            "[glm53-mtp-shared] output max_abs={max_abs:.3e} rel_l2={rel_l2:.3e} against DS4"
        );
        Ok(())
    }

    /// Capture only the eight selected MTP experts, then compare the routed
    /// output with DS4's F32 Metal graph replay.
    #[cfg(all(test, feature = "glm53-fixtures"))]
    pub(crate) fn capture_or_assert_mtp_routed_ds4(
        &self,
        x: &Array,
        actual: &Array,
    ) -> anyhow::Result<()> {
        const HIDDEN: usize = 4096;
        const INTER: usize = 2048;
        const TOP_K: usize = 8;
        let external_fixture: &[u8] =
            &crate::bindings::glm53_test_fixtures::read("mtp_routed_layer45_ds4.bin");
        anyhow::ensure!(
            x.dim(x.rank() as i32 - 1) == HIDDEN as i32
                && self.pools.inter == INTER
                && self.top_k == TOP_K
                && self.n_group == 1
                && self.topk_group == 1,
            "layer-45 MTP routed expert geometry differs from DS4 fixture"
        );
        let input = x.as_dtype(Dtype::Float32)?.reshape(&[1, HIDDEN as i32])?;
        let stream = lisa_metal::Stream::thread_local_or_default();
        let logits =
            lisa_metal::kernels::sparse::kernels::router_matvec_f32(&self.gate_w, &input, &stream)
                .ok_or_else(|| {
                    anyhow::anyhow!("GLM MTP routed fixture router projection declined")
                })?;
        let (selected, route_weights) = lisa_metal::kernels::sparse::kernels::router_select_fused(
            &logits,
            &self.bias,
            TOP_K,
            self.routed_scaling,
            &stream,
        )
        .ok_or_else(|| anyhow::anyhow!("GLM MTP routed fixture router selector declined"))?;
        let input_values = input.to_vec1::<f32>()?;
        let selected_values = selected.to_vec1::<i32>()?;
        let route_values = route_weights.to_vec1::<f32>()?;
        let output_values = actual.as_dtype(Dtype::Float32)?.to_vec1::<f32>()?;
        anyhow::ensure!(
            input_values.len() == HIDDEN
                && selected_values.len() == TOP_K
                && route_values.len() == TOP_K
                && output_values.len() == HIDDEN,
            "layer-45 MTP routed expert fixture shape mismatch"
        );

        if let Some(path) = std::env::var_os("LISA_GLM53_MTP_ROUTED_CAPTURE_DIR") {
            let path = std::path::Path::new(&path);
            std::fs::create_dir_all(path)?;
            std::fs::write(path.join("input.f32"), encode_f32(&input_values))?;
            std::fs::write(path.join("selected.i32"), encode_i32(&selected_values))?;
            std::fs::write(path.join("route_weights.f32"), encode_f32(&route_values))?;
            std::fs::write(path.join("lisa_output.f32"), encode_f32(&output_values))?;
            for (slot, expert_id) in selected_values.iter().enumerate() {
                let expert = self.pools.experts.get(*expert_id as usize).ok_or_else(|| {
                    anyhow::anyhow!("selected MTP expert id {expert_id} is invalid")
                })?;
                for (name, packed, scales, biases) in [
                    (
                        "gate",
                        &expert.gate.weight,
                        &expert.gate.scales,
                        &expert.gate.biases,
                    ),
                    (
                        "up",
                        &expert.up.weight,
                        &expert.up.scales,
                        &expert.up.biases,
                    ),
                    (
                        "down",
                        &expert.down.weight,
                        &expert.down.scales,
                        &expert.down.biases,
                    ),
                ] {
                    let packed = match packed.dtype() {
                        Dtype::Uint8 => packed.to_vec1::<u8>()?,
                        Dtype::Uint32 => encode_u32(&packed.to_vec1::<u32>()?),
                        dtype => anyhow::bail!(
                            "MTP expert {slot} {name} uses unsupported packed dtype {dtype:?}"
                        ),
                    };
                    let scales =
                        scales
                            .as_dtype(Dtype::Float32)?
                            .to_vec1::<f32>()
                            .map_err(|error| {
                                anyhow::anyhow!("read MTP expert {slot} {name} scales: {error}")
                            })?;
                    let biases =
                        biases
                            .as_dtype(Dtype::Float32)?
                            .to_vec1::<f32>()
                            .map_err(|error| {
                                anyhow::anyhow!("read MTP expert {slot} {name} biases: {error}")
                            })?;
                    let stem = format!("expert_{slot:02}_{name}");
                    std::fs::write(path.join(format!("{stem}_weight.u8")), packed)?;
                    std::fs::write(path.join(format!("{stem}_scales.f32")), encode_f32(&scales))?;
                    std::fs::write(path.join(format!("{stem}_biases.f32")), encode_f32(&biases))?;
                }
            }
            return Ok(());
        }

        const SELECTED_BYTES: usize = TOP_K * 4;
        anyhow::ensure!(
            external_fixture.len() == SELECTED_BYTES + HIDDEN * 8,
            "DS4 layer-45 MTP routed fixture has an invalid size"
        );
        let expected_selected = external_fixture[..SELECTED_BYTES]
            .chunks_exact(4)
            .map(|bytes| i32::from_le_bytes(bytes.try_into().expect("I32 expert id")))
            .collect::<Vec<_>>();
        anyhow::ensure!(
            selected_values == expected_selected,
            "layer-45 MTP routed expert ids differ from DS4 fixture"
        );
        let input_end = SELECTED_BYTES + HIDDEN * 4;
        let expected_input = external_fixture[SELECTED_BYTES..input_end]
            .chunks_exact(4)
            .map(|bytes| f32::from_le_bytes(bytes.try_into().expect("F32 input word")))
            .collect::<Vec<_>>();
        anyhow::ensure!(
            input_values
                .iter()
                .zip(&expected_input)
                .all(|(actual, expected)| actual.to_bits() == expected.to_bits()),
            "layer-45 MTP routed fixture input changed"
        );
        let expected_output = external_fixture[input_end..]
            .chunks_exact(4)
            .map(|bytes| f32::from_le_bytes(bytes.try_into().expect("F32 output word")))
            .collect::<Vec<_>>();
        let max_abs = output_values
            .iter()
            .zip(&expected_output)
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0f32, f32::max);
        let delta2 = output_values
            .iter()
            .zip(&expected_output)
            .map(|(actual, expected)| f64::from(actual - expected).powi(2))
            .sum::<f64>();
        let ref2 = expected_output
            .iter()
            .map(|expected| f64::from(*expected).powi(2))
            .sum::<f64>();
        let rel_l2 = (delta2 / ref2.max(f64::MIN_POSITIVE)).sqrt();
        anyhow::ensure!(
            max_abs <= 1e-4 && rel_l2 <= 1e-5,
            "layer-45 MTP routed output differs from DS4: max_abs={max_abs:e}, rel_l2={rel_l2:e}"
        );
        eprintln!(
            "[glm53-mtp-routed] selected IDs match DS4; output max_abs={max_abs:.3e} rel_l2={rel_l2:.3e}"
        );
        Ok(())
    }
}
